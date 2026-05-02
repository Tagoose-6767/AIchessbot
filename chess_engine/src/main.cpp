// main.cpp — UCI loop, perft, bench. All UCI traffic on stdout; debug stderr.

#include "types.h"
#include "board.h"
#include "movegen.h"
#include "search.h"
#include "tt.h"
#include "book.h"
#include "evaluate.h"
#include "nnue.h"
#include "syzygy.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

// Directory containing the running executable (no trailing slash). Used as
// NNUE's fallback search dir so relative paths like "nets/foo.nnue" resolve
// even when a GUI launches the engine from a different working directory.
static std::string executable_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf, n);
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t sz = sizeof(buf);
    std::string p = (_NSGetExecutablePath(buf, &sz) == 0) ? std::string(buf) : "";
#else
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf));
    std::string p = (n > 0) ? std::string(buf, n) : "";
#endif
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? std::string(".") : p.substr(0, s);
}

static const char* ENGINE_NAME    = "AIChessbot 0.2 (C++)";
static const char* ENGINE_AUTHOR  = "AIChessbot";
static const char* ENGINE_VERSION = "0.2";

static Board g_board;
static OpeningBook g_book;
static std::thread g_search_thread;

struct EngineOptions {
    int  hash_mb = 256;
    int  threads = 1;            // overwritten at init() to hardware_concurrency()
    bool ponder  = false;        // surfaced via UCI; protocol-level hint to GUI.
};
static EngineOptions g_opts;
static int g_max_threads = 1;  // populated at startup from hardware_concurrency()

// Limits and side-to-move from the most recent "go" so the ponderhit handler
// can compute the time budget for the now-time-limited search. Written by
// uci_go before spawning the search thread; read by the ponderhit handler.
static SearchLimits g_pending_limits;
static Color        g_pending_stm = WHITE;

static std::string score_to_uci(int s) {
    if (std::abs(s) >= VALUE_MATE_IN_MAX_PLY) {
        int plies = VALUE_MATE - std::abs(s);
        int moves = (plies + 1) / 2;
        std::ostringstream ss;
        ss << "mate " << (s > 0 ? moves : -moves);
        return ss.str();
    }
    std::ostringstream ss;
    ss << "cp " << s;
    return ss.str();
}

static void on_info(const SearchInfo& i) {
    std::ostringstream ss;
    ss << "info depth " << i.depth
       << " seldepth " << i.seldepth
       << " score " << score_to_uci(i.score)
       << " nodes " << i.nodes
       << " nps " << i.nps
       << " time " << i.time_ms
       << " hashfull " << TT.hashfull()
       << " pv";
    for (int k = 0; k < i.pv_len; k++) ss << ' ' << move_to_uci(i.pv[k]);
    std::cout << ss.str() << std::endl;
}

static void on_bestmove(Move m, Move ponder) {
    std::cout << "bestmove " << (m == MOVE_NONE ? "0000" : move_to_uci(m));
    if (ponder != MOVE_NONE) std::cout << " ponder " << move_to_uci(ponder);
    std::cout << std::endl;
}

static void wait_for_search() {
    if (g_search_thread.joinable()) g_search_thread.join();
}

// ---------------------------------------------------------------------------
// UCI command handlers
// ---------------------------------------------------------------------------
static void uci_position(std::istringstream& iss) {
    wait_for_search();
    std::string token;
    iss >> token;
    if (token == "startpos") {
        g_board.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        iss >> token;  // expect "moves" or end
    } else if (token == "fen") {
        std::string fen, t;
        for (int i = 0; i < 6 && iss >> t; ++i) {
            if (t == "moves") { token = "moves"; break; }
            if (!fen.empty()) fen += ' ';
            fen += t;
        }
        g_board.set(fen);
        // Caller may have consumed "moves" into `token`; if not, peek next.
        if (token != "moves") iss >> token;
    } else {
        return;
    }
    if (token == "moves") {
        std::string mv;
        while (iss >> mv) {
            Move m = uci_to_move(mv, g_board);
            if (m == MOVE_NONE) {
                std::cerr << "info string illegal move " << mv << '\n';
                break;
            }
            g_board.make_move(m);
        }
    }
}

static void uci_go(std::istringstream& iss) {
    wait_for_search();

    SearchLimits lim;
    std::string tok;
    while (iss >> tok) {
        if      (tok == "depth")     iss >> lim.depth;
        else if (tok == "movetime")  iss >> lim.movetime;
        else if (tok == "wtime")     iss >> lim.time[WHITE];
        else if (tok == "btime")     iss >> lim.time[BLACK];
        else if (tok == "winc")      iss >> lim.inc[WHITE];
        else if (tok == "binc")      iss >> lim.inc[BLACK];
        else if (tok == "movestogo") iss >> lim.movestogo;
        else if (tok == "infinite")  lim.infinite = true;
        else if (tok == "ponder")    lim.ponder = true;
    }

    // Stash for ponderhit, which converts the running search to time-limited.
    g_pending_limits = lim;
    g_pending_stm    = g_board.side_to_move();
    g_is_pondering.store(lim.ponder, std::memory_order_relaxed);

    // Try book first. Skip when pondering: under "go ponder" the engine must
    // keep searching until ponderhit/stop, even if the predicted position
    // happens to be in book.
    if (!lim.ponder && g_book.loaded()) {
        Move bm = g_book.find_move(g_board);
        if (bm != MOVE_NONE) {
            std::cout << "bestmove " << move_to_uci(bm) << std::endl;
            return;
        }
    }

    // Root Syzygy probe — perfect endgame play once <=5 pieces. Same ponder
    // exclusion as the book: don't short-circuit a pondering search.
    if (!lim.ponder
        && Syzygy::active() && popcount(g_board.pieces()) <= Syzygy::largest()
        && g_board.castling() == 0) {
        int tb_score = 0;
        Move tb_move = MOVE_NONE;
        if (Syzygy::probe_root(g_board, tb_score, tb_move) && tb_move != MOVE_NONE) {
            SearchInfo info{};
            info.depth = info.seldepth = 1;
            info.score = tb_score;
            info.nodes = 1;
            info.nps   = 0;
            info.time_ms = 0;
            info.pv[0] = tb_move;
            info.pv_len = 1;
            on_info(info);
            std::cout << "bestmove " << move_to_uci(tb_move) << std::endl;
            return;
        }
    }

    g_search_stop.store(false);
    int n_threads = g_opts.threads < 1 ? 1 : g_opts.threads;
    Board root = g_board;  // canonical root copied here before threads diverge

    g_search_thread = std::thread([lim, n_threads, root]() mutable {
        // Lazy SMP: each thread gets its own Board + Searcher; they all share
        // the global TT. The main thread (index 0) reports info and bestmove;
        // helpers run silently. Helpers diverge naturally via TT race effects
        // and their own killers/history.
        std::vector<Board> boards;
        std::vector<std::unique_ptr<Searcher>> searchers;
        boards.reserve(n_threads);
        searchers.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            boards.push_back(root);
            searchers.push_back(std::make_unique<Searcher>());
            searchers.back()->set_thread_id(i);
        }

        std::vector<std::thread> helpers;
        helpers.reserve(n_threads - 1);
        for (int i = 1; i < n_threads; ++i) {
            helpers.emplace_back([&, i]() {
#ifdef _WIN32
                // Pin helper i to logical CPU i for cache-locality on NUMA hosts.
                DWORD_PTR mask = DWORD_PTR(1) << (unsigned(i) % 64u);
                SetThreadAffinityMask(GetCurrentThread(), mask);
#endif
                searchers[i]->start(boards[i], lim,
                                    [](const SearchInfo&) {},     // no info from helpers
                                    [](Move, Move) {});           // no bestmove from helpers
            });
        }

        // Main search.
#ifdef _WIN32
        SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1));
#endif
        searchers[0]->start(boards[0], lim, on_info, on_bestmove);

        // Stop and join helpers before destroying their boards/searchers.
        g_search_stop.store(true);
        for (auto& s : searchers) s->stop();
        for (auto& t : helpers) if (t.joinable()) t.join();
    });
}

static void uci_setoption(std::istringstream& iss) {
    std::string tok, name, value;
    iss >> tok;  // "name"
    while (iss >> tok && tok != "value") { if (!name.empty()) name += ' '; name += tok; }
    while (iss >> tok) { if (!value.empty()) value += ' '; value += tok; }

    if (name == "Hash") {
        size_t mb = std::stoul(value);
        g_opts.hash_mb = int(mb);
        TT.resize(mb);
    } else if (name == "Book") {
        g_book.close();
        if (!value.empty()) g_book.load(value);
    } else if (name == "Threads") {
        int n = std::stoi(value);
        if (n < 1) n = 1;
        if (n > g_max_threads) n = g_max_threads;
        g_opts.threads = n;
    } else if (name == "EvalFile") {
        // "" / "<auto>" / "<empty>" all mean: re-try the bundled default net.
        // "<none>" / "off" forces HCE. Anything else is treated as an explicit
        // path (resolved by NNUE against CWD, then the executable dir).
        if (value == "<none>" || value == "off") {
            NNUE::unload();
            std::cout << "info string nnue: unloaded; using HCE\n";
        } else if (value.empty() || value == "<auto>" || value == "<empty>") {
            if (!NNUE::load(NNUE::DEFAULT_NET))
                std::cout << "info string EvalFile auto-load failed; using HCE\n";
        } else if (!NNUE::load(value)) {
            std::cout << "info string EvalFile load failed; using HCE\n";
        }
    } else if (name == "SyzygyPath") {
        Syzygy::init(value);
    } else if (name == "Ponder") {
        // Stored only — pondering is driven entirely by the GUI sending
        // "go ponder"; this option is the protocol-level capability hint.
        g_opts.ponder = (value == "true" || value == "1" || value == "True");
    }
}

// ---------------------------------------------------------------------------
// Bench: search a fixed set of positions to fixed depth, report total NPS.
// ---------------------------------------------------------------------------
static const std::vector<std::string> BENCH_FENS = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r2q1rk1/pp1bbppp/2np1n2/2p1p3/2P1P3/2NPBN2/PP2BPPP/R2Q1RK1 w - - 0 1",
    "rnbq1rk1/pp2ppbp/3p1np1/2pP4/4P3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 1",
    "8/8/4k3/8/2p5/3K4/8/8 w - - 0 1",
    "8/p3kp2/1p2p1p1/8/8/4P1P1/PP3PKP/8 w - - 0 1",
    "r4rk1/pp3ppp/2pq1n2/3p4/3P1B2/2PB1Q2/PP3PPP/R4RK1 w - - 0 1",
    "rnbqkb1r/pp1p1ppp/4pn2/2p5/2P5/2N2N2/PP1PPPPP/R1BQKB1R w KQkq - 0 1",
};

static void uci_bench(int depth) {
    uint64_t total_nodes = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (const auto& fen : BENCH_FENS) {
        Board b;
        b.set(fen);
        SearchLimits lim;
        lim.depth = depth;
        Searcher s;
        std::atomic<uint64_t> nodes_here{ 0 };
        SearchInfo last_info{};
        Move bench_best = MOVE_NONE, bench_ponder = MOVE_NONE;
        s.start(b, lim,
                [&](const SearchInfo& si) { last_info = si; },
                [&](Move best, Move ponder) { bench_best = best; bench_ponder = ponder; });
        std::cout << "bestmove "
                  << (bench_best == MOVE_NONE ? "0000" : move_to_uci(bench_best));
        if (bench_ponder != MOVE_NONE)
            std::cout << " ponder " << move_to_uci(bench_ponder);
        std::cout << std::endl;
        total_nodes += s.nodes_searched();
    }
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    uint64_t nps = ms > 0 ? (total_nodes * 1000) / ms : 0;
    std::cerr << "Bench: " << total_nodes << " nodes  "
              << ms << " ms  " << nps << " NPS  (depth " << depth << ")\n";
    std::cout << "info string bench " << total_nodes << " nodes "
              << nps << " nps" << std::endl;
}

static void uci_perft(int depth) {
    auto t0 = std::chrono::steady_clock::now();
    uint64_t n = perft(g_board, depth);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    uint64_t nps = ms > 0 ? (n * 1000) / ms : 0;
    std::cout << "info string perft(" << depth << ") = " << n
              << " (" << ms << " ms, " << nps << " NPS)" << std::endl;
}

// Per-move perft breakdown (used by external test harnesses).
static void uci_perft_divide(int depth) {
    MoveList ml;
    generate_legal(g_board, ml);
    uint64_t total = 0;
    for (int i = 0; i < ml.size; i++) {
        Move m = ml.moves[i].move;
        g_board.make_move(m);
        uint64_t n = perft(g_board, depth - 1);
        g_board.unmake_move(m);
        total += n;
        std::cout << move_to_uci(m) << ": " << n << '\n';
    }
    std::cout << "Total: " << total << std::endl;
}

// ---------------------------------------------------------------------------
// UCI loop
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Zobrist::init();
    init_movegen();
    TT.resize(256);
    g_max_threads = int(std::thread::hardware_concurrency());
    if (g_max_threads <= 0) g_max_threads = 1;
    g_opts.threads = g_max_threads;          // default to all cores
    clear_shared_history();
    g_board.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Auto-load the bundled NNUE relative to the executable, so GUIs that
    // launch the engine from a different CWD get NNUE eval out of the box.
    NNUE::set_search_dir(executable_dir());
    NNUE::load(NNUE::DEFAULT_NET);

    // CLI: `chess_engine bench [depth]` runs bench and exits.
    if (argc >= 2 && std::string(argv[1]) == "bench") {
        int d = 13;
        if (argc >= 3) d = std::atoi(argv[2]);
        uci_bench(d);
        return 0;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "uci") {
            std::cout << "id name " << ENGINE_NAME << '\n';
            std::cout << "id author " << ENGINE_AUTHOR << '\n';
            std::cout << "option name Hash type spin default 256 min 1 max 65536\n";
            std::cout << "option name Threads type spin default " << g_max_threads
                      << " min 1 max " << g_max_threads << "\n";
            std::cout << "option name Book type string default \n";
            std::cout << "option name EvalFile type string default <auto>\n";
            std::cout << "option name SyzygyPath type string default <empty>\n";
            std::cout << "option name Ponder type check default false\n";
            std::cout << "uciok" << std::endl;
        } else if (cmd == "isready") {
            wait_for_search();
            std::cout << "readyok" << std::endl;
        } else if (cmd == "ucinewgame") {
            wait_for_search();
            TT.clear();
            clear_shared_history();
            g_board.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        } else if (cmd == "position") {
            uci_position(iss);
        } else if (cmd == "go") {
            uci_go(iss);
        } else if (cmd == "stop") {
            // Stops every searcher (main + Lazy SMP helpers) via the global
            // flag that all of them poll inside time_up(). Also covers the
            // "ponder, then opponent played a different move" case.
            g_is_pondering.store(false, std::memory_order_relaxed);
            g_search_stop.store(true);
            wait_for_search();
        } else if (cmd == "ponderhit") {
            // Opponent played the move we predicted -- the running ponder
            // search IS the search for the current position. Convert it from
            // "no time limit" to time-limited by atomically programming the
            // global deadline; the main searcher will pick it up on its next
            // poll inside time_up(). Helpers ride along on the same deadline
            // (and on g_search_stop), so SMP cleanup is automatic.
            g_is_pondering.store(false, std::memory_order_relaxed);
            int64_t budget = compute_time_budget(g_pending_limits, g_pending_stm);
            int64_t deadline = budget > 0 ? now_steady_ms() + budget : INT64_MAX;
            g_search_deadline_ms.store(deadline, std::memory_order_relaxed);
        } else if (cmd == "setoption") {
            uci_setoption(iss);
        } else if (cmd == "quit") {
            g_search_stop.store(true);
            wait_for_search();
            Syzygy::free();
            break;
        } else if (cmd == "perft") {
            int d = 4;
            iss >> d;
            uci_perft(d);
        } else if (cmd == "divide") {
            int d = 4;
            iss >> d;
            uci_perft_divide(d);
        } else if (cmd == "bench") {
            int d = 13;
            iss >> d;
            uci_bench(d);
        } else if (cmd == "d" || cmd == "print") {
            g_board.print();
        } else if (cmd == "eval") {
            std::cerr << "eval = " << evaluate(g_board) << '\n';
        }
        // Unknown commands silently ignored, per UCI spec.
    }
    wait_for_search();
    return 0;
}

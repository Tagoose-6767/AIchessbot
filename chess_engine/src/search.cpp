// search.cpp — alpha-beta with PVS, NMP, LMR, futility, razoring, qsearch.

#include "search.h"
#include "movegen.h"
#include "evaluate.h"
#include "tt.h"
#include "syzygy.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

std::atomic<bool> g_search_stop{false};
std::atomic<bool> g_is_pondering{false};
std::atomic<int64_t> g_search_deadline_ms{INT64_MAX};

// Process-wide history shared across Lazy SMP threads.
std::atomic<int> g_history[16][64];

int64_t compute_time_budget(const SearchLimits& lim, Color stm) {
    if (lim.infinite) return 0;
    if (lim.movetime > 0) return lim.movetime;
    int64_t our_time = lim.time[stm];
    int64_t our_inc  = lim.inc[stm];
    if (our_time <= 0) return 0;
    int mtg = lim.movestogo > 0 ? lim.movestogo : 30;
    int64_t target = our_time / mtg + our_inc * 70 / 100;
    if (target < 10) target = 10;
    if (target > our_time / 4) target = our_time / 4;
    return target;
}

void clear_shared_history() {
    for (int p = 0; p < 16; ++p)
        for (int sq = 0; sq < 64; ++sq)
            g_history[p][sq].store(0, std::memory_order_relaxed);
}

namespace {
constexpr int ASPIRATION_DELTA = 25;
constexpr int RAZOR_MARGIN[4]  = { 0, 250, 400, 0 };
constexpr int FUT_MARGIN[5]    = { 0, 150, 300, 450, 0 };
constexpr int LMR_MIN_DEPTH    = 3;
constexpr int LMR_MIN_MOVES    = 4;

inline bool is_mate(int s) { return std::abs(s) >= VALUE_MATE_IN_MAX_PLY; }
}

bool Searcher::time_up() {
    if (stop_flag_.load(std::memory_order_relaxed)) return true;
    if (g_search_stop.load(std::memory_order_relaxed)) {
        stop_flag_.store(true, std::memory_order_relaxed);
        return true;
    }
    int64_t deadline = g_search_deadline_ms.load(std::memory_order_relaxed);
    if (deadline == INT64_MAX) return false;
    if (now_steady_ms() >= deadline) {
        stop_flag_.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Quiescence: only captures (and check evasions) until position is "quiet".
// ---------------------------------------------------------------------------
int Searcher::quiescence(Board& b, int ply, int alpha, int beta) {
    if ((nodes_ & 2047) == 0 && time_up()) return 0;
    nodes_++;
    if (ply > seldepth_) seldepth_ = ply;
    if (ply >= MAX_PLY - 1) return evaluate(b);

    int stand_pat;
    bool in_check = b.is_check();
    if (!in_check) {
        stand_pat = evaluate(b);
        if (stand_pat >= beta)  return stand_pat;
        if (stand_pat > alpha)  alpha = stand_pat;
    } else {
        stand_pat = -VALUE_INFINITE;
    }

    MoveList ml;
    if (in_check) generate_legal(b, ml);
    else          generate_captures(b, ml);

    // Order: TT not used in qsearch; use MVV-LVA / SEE.
    for (int i = 0; i < ml.size; i++) {
        Move m = ml.moves[i].move;
        int s = 0;
        if (b.is_capture(m)) {
            PieceType v = type_of_move(m) == MT_ENPASSANT ? PAWN : b.type_on(to_sq(m));
            PieceType a = b.type_on(from_sq(m));
            s = PieceValueSimple[v] * 16 - int(a);
        }
        if (type_of_move(m) == MT_PROMOTION) s += PieceValueSimple[promotion_of(m)];
        ml.moves[i].score = s;
    }

    bool any_legal = false;
    for (int i = 0; i < ml.size; i++) {
        ExtMove& em = pick_next(ml, i);
        Move m = em.move;

        if (!in_check) {
            // Skip captures with negative SEE (bad captures).
            if (b.is_capture(m) && !b.see_ge(m, 0)) continue;
            // Delta pruning: even capturing this victim plus a slack margin
            // can't raise alpha — skip.
            if (b.is_capture(m) && type_of_move(m) != MT_PROMOTION) {
                int v = type_of_move(m) == MT_ENPASSANT ? PieceValueSimple[PAWN]
                                                         : PieceValueSimple[b.type_on(to_sq(m))];
                if (stand_pat + v + 200 < alpha) continue;
            }
        }

        b.make_move(m);
        if (!in_check) {
            // Pseudo-legal generation requires a legality check after make.
            Color mover = ~b.side_to_move();
            if (b.attackers_to(b.king_sq(mover), b.pieces()) & b.pieces(b.side_to_move())) {
                b.unmake_move(m);
                continue;
            }
        }
        any_legal = true;
        int v = -quiescence(b, ply + 1, -beta, -alpha);
        b.unmake_move(m);
        if (stop_flag_.load(std::memory_order_relaxed)) return 0;

        if (v >= beta)  return v;
        if (v > alpha)  alpha = v;
    }

    if (in_check && !any_legal) return -VALUE_MATE + ply;
    return alpha;
}

// ---------------------------------------------------------------------------
// Negamax with all the trimmings.
// ---------------------------------------------------------------------------
int Searcher::negamax(Board& b, int depth, int ply, int alpha, int beta, bool allow_null) {
    pv_len_[ply] = 0;
    if ((nodes_ & 2047) == 0 && time_up()) return 0;
    if (ply >= MAX_PLY - 1) return evaluate(b);

    // Mate-distance pruning.
    alpha = std::max(alpha, -VALUE_MATE + ply);
    beta  = std::min(beta,   VALUE_MATE - ply - 1);
    if (alpha >= beta) return alpha;

    bool in_check = b.is_check();
    if (in_check) ++depth;  // check extension — never drop into qsearch in check.

    if (depth <= 0) return quiescence(b, ply, alpha, beta);
    nodes_++;
    if (ply > seldepth_) seldepth_ = ply;

    if (ply > 0 && (b.is_repetition() || b.is_50move_draw() || b.is_insufficient_material()))
        return VALUE_DRAW;

    bool is_pv = beta - alpha > 1;

    // --- TT probe ----
    bool tt_hit;
    TTEntry* tte = TT.probe(b.key(), tt_hit);
    Move tt_move = tt_hit ? tte->move : MOVE_NONE;
    if (tt_hit && !is_pv && ply > 0 && tte->depth >= depth) {
        int s = score_from_tt(tte->score, ply);
        if (tte->bound == BOUND_EXACT) return s;
        if (tte->bound == BOUND_LOWER && s >= beta)  return s;
        if (tte->bound == BOUND_UPPER && s <= alpha) return s;
    }

    // --- Syzygy WDL probe ----
    // Cheap and safe: requires no castling rights (TB ignores castling), and the
    // 50-move clock must have just been reset by an irreversible move so that
    // TB rule50 effects don't mislead us. Skip on root and inside qsearch.
    if (ply > 0 && Syzygy::active()
        && popcount(b.pieces()) <= Syzygy::largest()
        && b.castling() == 0
        && b.halfmove_clock() == 0) {
        int tb_score = 0;
        if (Syzygy::probe_wdl(b, tb_score)) {
            // Clamp into TB-mate band (already done by probe_wdl). Store as exact.
            if (!stop_flag_.load(std::memory_order_relaxed))
                TT.store(b.key(), depth, score_to_tt(tb_score, ply),
                         BOUND_EXACT, MOVE_NONE);
            return tb_score;
        }
    }

    int static_eval = in_check ? VALUE_NONE : evaluate(b);

    // --- Pre-move pruning (skip when in check or PV) ---
    if (!in_check && !is_pv && std::abs(beta) < VALUE_MATE_IN_MAX_PLY) {
        // Reverse futility / static null-move.
        if (depth <= 4 && static_eval - 80 * depth >= beta) return static_eval;

        // Razoring: at low depth, when far below alpha, drop to qsearch.
        if (depth <= 2 && static_eval + RAZOR_MARGIN[depth] <= alpha) {
            int q = quiescence(b, ply, alpha, beta);
            if (q <= alpha) return q;
        }

        // Null-move pruning. Skip if our side has no non-pawn material
        // (zugzwang risk in pure pawn endings).
        if (allow_null && depth >= 3 && static_eval >= beta
            && b.non_pawn_material(b.side_to_move()) > 0) {
            int R = 2 + depth / 6;
            b.make_null_move();
            int v = -negamax(b, depth - 1 - R, ply + 1, -beta, -beta + 1, false);
            b.unmake_null_move();
            if (stop_flag_.load(std::memory_order_relaxed)) return 0;
            if (v >= beta && !is_mate(v)) return v;
        }
    }

    // --- Staged move generation ---
    MovePicker mp(b, tt_move, killers_[ply], g_history);

    // Frontier futility precondition.
    bool do_futility = (!in_check && !is_pv && depth <= 4
                        && static_eval + FUT_MARGIN[depth] <= alpha
                        && std::abs(alpha) < VALUE_MATE_IN_MAX_PLY);

    int best_score = -VALUE_INFINITE;
    Move best_move = MOVE_NONE;
    Bound flag = BOUND_UPPER;
    int moves_searched = 0;
    int legal_count = 0;

    Move m;
    while ((m = mp.next()) != MOVE_NONE) {
        bool is_cap = b.is_capture(m);
        bool is_promo = type_of_move(m) == MT_PROMOTION;
        bool gives_chk = b.gives_check(m);
        bool is_quiet = !is_cap && !is_promo && !gives_chk;

        if (do_futility && is_quiet && moves_searched > 0) continue;

        b.make_move(m);
        // Legality check (we used pseudo-legal generation).
        Color mover = ~b.side_to_move();
        if (b.attackers_to(b.king_sq(mover), b.pieces()) & b.pieces(b.side_to_move())) {
            b.unmake_move(m);
            continue;
        }
        ++legal_count;

        int new_depth = depth - 1;

        // LMR: trust ordering for late, quiet, non-checking moves.
        int reduction = 0;
        if (depth >= LMR_MIN_DEPTH && moves_searched >= LMR_MIN_MOVES
            && is_quiet && !in_check) {
            reduction = 1 + int(std::log(double(depth)) * std::log(double(moves_searched + 1)) / 2.0);
            if (reduction > new_depth - 1) reduction = new_depth - 1;
            if (reduction < 0) reduction = 0;
        }

        int v;
        if (moves_searched == 0) {
            v = -negamax(b, new_depth, ply + 1, -beta, -alpha, true);
        } else {
            v = -negamax(b, new_depth - reduction, ply + 1, -alpha - 1, -alpha, true);
            if (v > alpha && reduction > 0)
                v = -negamax(b, new_depth, ply + 1, -alpha - 1, -alpha, true);
            if (v > alpha && v < beta)
                v = -negamax(b, new_depth, ply + 1, -beta, -alpha, true);
        }
        b.unmake_move(m);
        if (stop_flag_.load(std::memory_order_relaxed)) return 0;

        ++moves_searched;
        if (v > best_score) {
            best_score = v;
            best_move = m;
            if (v > alpha) {
                alpha = v;
                flag = BOUND_EXACT;
                pv_table_[ply][0] = m;
                int child_len = pv_len_[ply + 1];
                for (int i2 = 0; i2 < child_len; i2++)
                    pv_table_[ply][i2 + 1] = pv_table_[ply + 1][i2];
                pv_len_[ply] = child_len + 1;

                if (v >= beta) {
                    flag = BOUND_LOWER;
                    if (is_quiet) {
                        if (killers_[ply][0] != m) {
                            killers_[ply][1] = killers_[ply][0];
                            killers_[ply][0] = m;
                        }
                        Piece p = b.piece_on(from_sq(m));
                        int new_val = g_history[p][to_sq(m)]
                            .fetch_add(depth * depth, std::memory_order_relaxed)
                            + depth * depth;
                        // Cap history to avoid runaway numbers. Race-tolerant:
                        // multiple threads may divide concurrently; values are
                        // heuristic so transient inconsistency is harmless.
                        if (new_val > (1 << 18)) {
                            for (int pp = 0; pp < 16; pp++)
                                for (int sq = 0; sq < 64; sq++) {
                                    int v = g_history[pp][sq].load(std::memory_order_relaxed);
                                    g_history[pp][sq].store(v / 2, std::memory_order_relaxed);
                                }
                        }
                    }
                    break;
                }
            }
        }
    }

    if (legal_count == 0)
        return in_check ? -VALUE_MATE + ply : VALUE_DRAW;

    if (!stop_flag_.load(std::memory_order_relaxed))
        TT.store(b.key(), depth, score_to_tt(best_score, ply), flag, best_move);
    return best_score;
}

// ---------------------------------------------------------------------------
// Iterative deepening driver.
// ---------------------------------------------------------------------------
void Searcher::start(Board& b, const SearchLimits& lim,
                     std::function<void(const SearchInfo&)> on_info,
                     std::function<void(Move, Move)> on_bestmove) {
    nodes_ = 0;
    seldepth_ = 0;
    stop_flag_.store(false);
    start_tp_ = std::chrono::steady_clock::now();

    // Only the main thread programs the global deadline. Helpers piggy-back on
    // it (and on g_search_stop). Pondering and infinite searches install no
    // deadline; ponderhit will rewrite it from the UCI thread when the
    // expected move actually arrives.
    if (thread_id_ == 0) {
        int64_t budget = compute_time_budget(lim, b.side_to_move());
        int64_t deadline = (lim.ponder || budget <= 0) ? INT64_MAX
                                                       : now_steady_ms() + budget;
        g_search_deadline_ms.store(deadline, std::memory_order_relaxed);
    }

    std::memset(killers_, 0, sizeof(killers_));
    std::memset(countermove_, 0, sizeof(countermove_));
    std::memset(pv_len_, 0, sizeof(pv_len_));
    // Main thread (or single-threaded bench) clears shared atomic history so
    // each search starts with a clean ordering signal. Helpers piggy-back.
    if (thread_id_ == 0) clear_shared_history();
    TT.new_search();

    // Lazy SMP depth staggering: helpers offset their starting depth and skip
    // some plies to diversify the search tree relative to the main thread.
    // Stockfish-style 20-entry skip pattern.
    constexpr int SKIP_PLIES[20] = { 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4 };
    constexpr int SKIP_PHASE[20] = { 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7 };

    Move best_move = MOVE_NONE;
    Move ponder_move = MOVE_NONE;   // PV[1] from the last completed iteration
    int  best_score = 0;
    int max_d = std::min(lim.depth, MAX_PLY - 1);

    for (int depth = 1; depth <= max_d; depth++) {
        // Skip selected depths on helper threads to diversify search.
        if (thread_id_ > 0) {
            int idx = (thread_id_ - 1) % 20;
            if (((depth + SKIP_PHASE[idx]) / SKIP_PLIES[idx]) % 2 != 0) continue;
        }
        seldepth_ = 0;
        int alpha, beta, delta = ASPIRATION_DELTA;
        if (depth >= 4 && !is_mate(best_score)) {
            alpha = best_score - delta;
            beta  = best_score + delta;
        } else {
            alpha = -VALUE_INFINITE;
            beta  =  VALUE_INFINITE;
        }

        int score;
        for (;;) {
            std::memset(pv_len_, 0, sizeof(pv_len_));
            score = negamax(b, depth, 0, alpha, beta, true);
            if (stop_flag_.load(std::memory_order_relaxed)) break;
            if (score <= alpha) {
                alpha = std::max(-VALUE_INFINITE, alpha - delta);
                delta *= 2;
            } else if (score >= beta) {
                beta = std::min(VALUE_INFINITE, beta + delta);
                delta *= 2;
            } else {
                break;
            }
        }

        if (stop_flag_.load(std::memory_order_relaxed) && depth > 1) break;
        best_score = score;

        if (pv_len_[0] > 0) {
            best_move = pv_table_[0][0];
            // Capture the predicted opponent reply *here* (only when we just
            // completed an iteration). Reading pv_len_[0] after the for-loop
            // exits would race with the aspiration window's memset, which can
            // wipe the PV before we reach the on_bestmove call.
            ponder_move = (pv_len_[0] >= 2) ? pv_table_[0][1] : MOVE_NONE;
        }

        // Emit info.
        SearchInfo info;
        info.depth = depth;
        info.seldepth = seldepth_;
        info.score = best_score;
        info.nodes = nodes_;
        auto now = std::chrono::steady_clock::now();
        info.time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_tp_).count();
        info.nps = info.time_ms > 0 ? (info.nodes * 1000) / info.time_ms : info.nodes * 1000;
        info.pv_len = pv_len_[0];
        for (int i = 0; i < pv_len_[0]; i++) info.pv[i] = pv_table_[0][i];
        if (on_info) on_info(info);

        if (is_mate(best_score)) break;
    }

    if (on_bestmove) on_bestmove(best_move, ponder_move);
}

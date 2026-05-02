// search.h — search driver: iterative deepening, time control, info reporting.
#pragma once

#include "types.h"
#include "board.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>

// Process-wide history table, shared across all Lazy SMP search threads.
// Updated atomically via fetch_add on cutoffs; read with relaxed loads in the
// MovePicker. Cleared on `ucinewgame`.
extern std::atomic<int> g_history[16][64];
void clear_shared_history();

struct SearchLimits {
    int    depth     = MAX_PLY;
    int64_t movetime = 0;          // 0 = ignore
    int64_t time[2]  = { 0, 0 };   // [WHITE], [BLACK]
    int64_t inc[2]   = { 0, 0 };
    int    movestogo = 0;
    bool   infinite  = false;
    bool   ponder    = false;      // "go ponder": no time limit until ponderhit.
};

// Global stop flag. UCI "stop" sets it; cleared at the start of every "go".
// All Searchers check it in addition to their own per-instance stop, so a
// stop request reaches every helper thread under Lazy SMP.
extern std::atomic<bool> g_search_stop;

// Pondering state. Set true by uci_go when "go ponder" is received and cleared
// by the ponderhit handler. The search treats it as "no time limit"; ponderhit
// converts the running search to time-limited by atomically updating the
// deadline below.
extern std::atomic<bool> g_is_pondering;

// Absolute deadline in steady_clock milliseconds-since-epoch. INT64_MAX means
// "no deadline" (used for infinite / pondering searches). Written at search
// start by the searcher and at ponderhit by the UCI thread; read by every
// thread inside time_up(). One global is enough because uci_go always waits
// for any prior search before starting a new one.
extern std::atomic<int64_t> g_search_deadline_ms;

// Returns the time budget (ms) we should spend on this move given the limits
// and side-to-move. 0 means "no time control" (infinite / no time fields).
int64_t compute_time_budget(const SearchLimits& lim, Color stm);

// Helper: steady_clock::now() in ms-since-epoch.
inline int64_t now_steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct SearchInfo {
    int depth, seldepth, score;
    uint64_t nodes;
    uint64_t nps;
    int64_t  time_ms;
    Move pv[MAX_PLY];
    int  pv_len;
};

class Searcher {
public:
    // on_bestmove takes (best, ponder). ponder is the second move of the PV
    // (i.e. the predicted opponent reply); MOVE_NONE if the PV is too short.
    void start(Board& b, const SearchLimits& lim,
               std::function<void(const SearchInfo&)> on_info,
               std::function<void(Move, Move)> on_bestmove);
    void stop() { stop_flag_.store(true); }

    uint64_t nodes_searched() const { return nodes_; }

    // Thread index: 0 = main thread, >=1 = helper. Used for depth staggering
    // so helpers explore different plies first, diversifying search.
    void set_thread_id(int id) { thread_id_ = id; }

private:
    std::atomic<bool> stop_flag_{false};
    uint64_t nodes_ = 0;
    int      seldepth_ = 0;
    int      thread_id_ = 0;
    std::chrono::steady_clock::time_point start_tp_;

    Move killers_[MAX_PLY][2];
    Move countermove_[16][64];      // countermove_[prev_piece_moved][prev_to]
    Move pv_table_[MAX_PLY][MAX_PLY];
    int  pv_len_[MAX_PLY];

    bool time_up();
    int  negamax(Board& b, int depth, int ply, int alpha, int beta, bool allow_null);
    int  quiescence(Board& b, int ply, int alpha, int beta);
};

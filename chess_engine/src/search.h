// search.h — search driver: iterative deepening, time control, info reporting.
#pragma once

#include "types.h"
#include "board.h"
#include <atomic>
#include <chrono>
#include <functional>

struct SearchLimits {
    int    depth     = MAX_PLY;
    int64_t movetime = 0;          // 0 = ignore
    int64_t time[2]  = { 0, 0 };   // [WHITE], [BLACK]
    int64_t inc[2]   = { 0, 0 };
    int    movestogo = 0;
    bool   infinite  = false;
};

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
    void start(Board& b, const SearchLimits& lim,
               std::function<void(const SearchInfo&)> on_info,
               std::function<void(Move)> on_bestmove);
    void stop() { stop_flag_.store(true); }

    uint64_t nodes_searched() const { return nodes_; }

private:
    std::atomic<bool> stop_flag_{false};
    uint64_t nodes_ = 0;
    int      seldepth_ = 0;
    std::chrono::steady_clock::time_point start_tp_;
    int64_t  hard_time_limit_ = 0;  // milliseconds; -1 = none

    Move killers_[MAX_PLY][2];
    int  history_[16][64];          // history_[piece][to_square]
    Move countermove_[16][64];      // countermove_[prev_piece_moved][prev_to]
    Move pv_table_[MAX_PLY][MAX_PLY];
    int  pv_len_[MAX_PLY];

    bool time_up();
    int  negamax(Board& b, int depth, int ply, int alpha, int beta, bool allow_null);
    int  quiescence(Board& b, int ply, int alpha, int beta);
};

// tt.h — transposition table with Zobrist keying.
#pragma once

#include "types.h"

enum Bound : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct TTEntry {
    uint64_t key;        // 64-bit full key (no collisions worth catching at our scale)
    int16_t  score;
    Move     move;
    int8_t   depth;
    uint8_t  bound;
    uint8_t  age;
};

class TranspositionTable {
public:
    void resize(size_t mb);
    void clear();
    void new_search() { ++age_; }
    TTEntry* probe(uint64_t key, bool& found) const;
    void store(uint64_t key, int depth, int score, Bound flag, Move m);

    int hashfull() const;  // permille
private:
    TTEntry* table_ = nullptr;
    size_t   count_ = 0;
    size_t   mask_  = 0;
    uint8_t  age_   = 0;
};

extern TranspositionTable TT;

// Per-engine Zobrist keys (separate from Polyglot book hashes — TT only).
namespace Zobrist {
    void init();
    extern U64 piece_square[16][64];
    extern U64 castling[16];
    extern U64 ep_file[8];
    extern U64 side;
}

// Adjust mate scores when storing/retrieving so they're independent of ply.
inline int score_to_tt(int s, int ply) {
    if (s >=  VALUE_MATE_IN_MAX_PLY) return s + ply;
    if (s <= -VALUE_MATE_IN_MAX_PLY) return s - ply;
    return s;
}
inline int score_from_tt(int s, int ply) {
    if (s >=  VALUE_MATE_IN_MAX_PLY) return s - ply;
    if (s <= -VALUE_MATE_IN_MAX_PLY) return s + ply;
    return s;
}

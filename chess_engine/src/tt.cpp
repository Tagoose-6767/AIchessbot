// tt.cpp — transposition table implementation + Zobrist key generation.

#include "tt.h"

#include <cstring>
#include <random>

TranspositionTable TT;

namespace Zobrist {
    U64 piece_square[16][64];
    U64 castling[16];
    U64 ep_file[8];
    U64 side;

    void init() {
        std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
        for (int p = 0; p < 16; p++)
            for (int s = 0; s < 64; s++)
                piece_square[p][s] = rng();
        for (int i = 0; i < 16; i++) castling[i] = rng();
        for (int i = 0; i < 8; i++)  ep_file[i]  = rng();
        side = rng();

        // NO_PIECE entries should not contribute to keys.
        for (int s = 0; s < 64; s++) piece_square[NO_PIECE][s] = 0;
        // Zero castling[0] so empty rights contribute nothing.
        castling[0] = 0;
    }
}

void TranspositionTable::resize(size_t mb) {
    delete[] table_;
    size_t bytes = mb * 1024 * 1024;
    size_t entries = 1;
    while ((entries * 2) * sizeof(TTEntry) <= bytes) entries *= 2;
    if (entries < 1024) entries = 1024;
    count_ = entries;
    mask_  = entries - 1;
    table_ = new TTEntry[entries];
    clear();
}

void TranspositionTable::clear() {
    if (table_) std::memset(table_, 0, count_ * sizeof(TTEntry));
    age_ = 0;
}

TTEntry* TranspositionTable::probe(uint64_t key, bool& found) const {
    TTEntry* e = &table_[key & mask_];
    found = (e->key == key && e->bound != BOUND_NONE);
    return e;
}

void TranspositionTable::store(uint64_t key, int depth, int score, Bound flag, Move m) {
    TTEntry* e = &table_[key & mask_];
    // Depth-preferred replacement: keep deeper entries from the same search;
    // always replace stale (different age) entries.
    if (e->bound == BOUND_NONE
        || e->age != age_
        || depth + 2 >= e->depth
        || flag == BOUND_EXACT) {
        if (m == MOVE_NONE && e->key == key) m = e->move;  // preserve TT move on bound updates
        e->key = key;
        e->depth = int8_t(depth);
        e->score = int16_t(score);
        e->bound = uint8_t(flag);
        e->move = m;
        e->age = age_;
    }
}

int TranspositionTable::hashfull() const {
    int filled = 0;
    int sample = std::min<size_t>(1000, count_);
    for (int i = 0; i < sample; i++)
        if (table_[i].bound != BOUND_NONE && table_[i].age == age_) ++filled;
    return filled;
}

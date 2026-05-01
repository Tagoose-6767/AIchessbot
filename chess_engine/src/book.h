// book.h — Polyglot opening book (stub in this release).
//
// Note: a real Polyglot reader requires the 781-entry Polyglot Random64 table
// to compute book-compatible Zobrist keys. That table is not embedded yet;
// see TODO in book.cpp. The chess_engine/book.py reader (Python) is fully
// functional if you need book moves now.
#pragma once

#include "types.h"
class Board;

class OpeningBook {
public:
    bool load(const std::string& path);
    Move find_move(const Board& b);  // returns MOVE_NONE if no entry / not loaded
    void close();
    bool loaded() const { return loaded_; }
private:
    bool loaded_ = false;
};

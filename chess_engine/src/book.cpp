// book.cpp — Polyglot opening book (stub).
//
// A real Polyglot reader requires the 781-entry Polyglot Random64 constant
// table to compute book-compatible Zobrist keys. That table is not embedded
// in this release. Calling find_move() always returns MOVE_NONE; the engine
// falls back to its search. The Python implementation in chess_engine/book.py
// has a working Polyglot reader if you need book moves now.

#include "book.h"
#include "board.h"

#include <fstream>
#include <iostream>

bool OpeningBook::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "info string book: failed to open " << path << '\n';
        loaded_ = false;
        return false;
    }
    // Sanity: a polyglot file is a multiple of 16 bytes.
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    if (sz < 16 || (sz % 16) != 0) {
        std::cerr << "info string book: not a valid polyglot file\n";
        loaded_ = false;
        return false;
    }
    std::cerr << "info string book: loaded " << path
              << " (" << (sz / 16) << " entries) but key generation is not "
              << "implemented; book is currently inert\n";
    loaded_ = true;
    return true;
}

Move OpeningBook::find_move(const Board& /*b*/) {
    // TODO: implement polyglot key (requires the 781 Random64 constants).
    return MOVE_NONE;
}

void OpeningBook::close() { loaded_ = false; }

// syzygy.h — Fathom-based Syzygy tablebase wrapper.
#pragma once

#include "types.h"
class Board;

namespace Syzygy {

// Initialize from a path string (semicolons on Windows, colons on POSIX).
// Returns true if at least one tablebase file was loaded.
bool init(const std::string& path);

// Free all loaded tablebases.
void free();

// Largest tablebase piece count actually loaded (0 if none).
int largest();

// Whether init has been attempted with a non-empty path and at least one
// tablebase was found.
bool active();

// Probe at the root using DTZ. On hit, returns true and writes:
//  - score  — engine score (mate or cp), from side-to-move POV
//  - best   — the suggested best move
// Caller must check that it's safe to probe (piece count <= largest()).
bool probe_root(const Board& b, int& score, Move& best);

// Probe in-search using WDL. On hit, returns true and writes engine score.
// Returns false if probe failed or position is not eligible (castling, etc.).
bool probe_wdl(const Board& b, int& score);

// Convert a Fathom (from,to,promotion) to our Move encoding given a Board
// to disambiguate en-passant. Returns MOVE_NONE if it can't be matched.
Move fathom_move_to_engine(const Board& b, unsigned from, unsigned to, unsigned promotes);

}  // namespace Syzygy

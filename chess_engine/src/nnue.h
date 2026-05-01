// nnue.h — Stockfish-12-compatible HalfKP-256x2-32-32 NNUE inference.
//
// File format: native Stockfish .nnue (version 0x7AF32F16, ~21 MB).
//
// Architecture:
//   - HalfKP feature transformer: 41024 features per perspective × 256 hidden
//   - Concatenated dual-perspective accumulator → 512 dims
//   - Affine 512→32 + ClippedReLU
//   - Affine 32→32  + ClippedReLU
//   - Affine 32→1
//   - Final score = output / FV_SCALE (16)
//
// Quantization: feature transformer weights/biases int16, hidden layer
// weights int8 with int32 biases, ClippedReLU shifts by 6 bits before
// clamping to [0, 127].
//
// This is a NON-INCREMENTAL implementation: the accumulator is rebuilt from
// scratch on every eval call. The reference implementation incrementally
// updates per move, which is roughly 10× faster but requires hooks in
// make/unmake. Cost: ~25K mul-adds per eval (scalar), comparable to HCE.
//
// If no .nnue file is loaded, evaluate() falls back to HCE — no regression.
#pragma once

#include "types.h"
#include <string>

class Board;

namespace NNUE {
    bool load(const std::string& path);
    void unload();
    bool is_loaded();
    int  evaluate(const Board& b);  // centipawns from side-to-move's perspective
}

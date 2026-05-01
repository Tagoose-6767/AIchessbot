// nnue.h — NNUE-lite (768→256→1) inference infrastructure.
//
// Architecture:
//   - 768 input features (12 piece types × 64 squares). Each piece on the
//     board activates one feature.
//   - 256-dim hidden layer with ReLU.
//   - 1-dim linear output (centipawns from white's perspective).
//
// File format ("NNUELITE" magic, little-endian, all int16 quantized):
//   char     magic[8]                 = "NNUELITE"
//   uint32_t version                  = 1
//   uint16_t input_dim                = 768
//   uint16_t hidden_dim               = 256
//   uint16_t output_dim               = 1
//   int16_t  output_scale             (divisor for the int32 output)
//   int16_t  W_input[768 * 256]       (input weights, indexed [feat][hidden])
//   int16_t  b_hidden[256]
//   int16_t  W_output[256]
//   int32_t  b_output
//
// Without a loaded .nnue file evaluate() falls back to HCE — no regression.
// You must supply trained weights to actually gain Elo. Random/untrained
// weights will WEAKEN the engine, so we never auto-initialize them.
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

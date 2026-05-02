// evaluate.h — tapered static evaluation.
#pragma once

#include "types.h"
class Board;

// Score is centipawns from the side-to-move's perspective.
int evaluate(const Board& b);

// HCE-only evaluator (skips NNUE dispatch). Used by the Texel tuner.
int evaluate_hce(const Board& b);

// All HCE weights tunable by the Texel pipeline. Layout is stable; the tuner
// rewrites the literal block in evaluate.cpp between TEXEL-TUNED-WEIGHTS-BEGIN
// and TEXEL-TUNED-WEIGHTS-END markers.
struct EvalWeights {
    int mg_val[7];                  // [PT]: P, N, B, R, Q (KING entries unused)
    int eg_val[7];
    int pst_mg[7][64];              // [PT][sq]
    int pst_eg[7][64];
    int passed_mg[8];               // by relative rank (0,7 unused)
    int passed_eg[8];
    int connected_passed_mg;
    int connected_passed_eg;
    int doubled_mg;
    int doubled_eg;
    int isolated_mg;
    int isolated_eg;
    int rook_open_mg;
    int rook_open_eg;
    int rook_semi_mg;
    int rook_semi_eg;
    int king_shelter_close;         // pawn 1 square in front
    int king_shelter_far;           // pawn 2 squares in front
    int king_shelter_missing;       // file with no own pawn
    int king_shelter_behind;        // pawn behind king
    int mobility_mg;
    int mobility_eg;
    int bishop_pair_mg;
    int bishop_pair_eg;
    int tempo;
    // --- Section 3 additions (no NNUE; HCE-only) ---
    // Per Chebyshev-step that any of my N/B/R/Q is closer than 7 to the
    // enemy king. Sum over my minor+major pieces.
    int king_tropism_mg;
    int king_tropism_eg;
    // Two same-color rooks on the same file or rank with no own/opponent
    // pieces strictly between them.
    int connected_rooks_mg;
    int connected_rooks_eg;
    // Bishop attacks at least 5 squares of the a1-h8 / a8-h1 long
    // diagonals (saturated; outpost-style "controls a long diagonal").
    int bishop_long_diag_mg;
    int bishop_long_diag_eg;
};

extern EvalWeights g_eval;

// Reset g_eval to the hand-tuned (or last Texel-tuned) literals.
void evaluate_reload_weights();

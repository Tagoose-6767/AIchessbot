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
};

extern EvalWeights g_eval;

// Reset g_eval to the hand-tuned (or last Texel-tuned) literals.
void evaluate_reload_weights();

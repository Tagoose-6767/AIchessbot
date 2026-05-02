// evaluate.cpp — tapered (MG/EG) static evaluation with PeSTO PSTs.

#include "evaluate.h"
#include "board.h"
#include "movegen.h"
#include "nnue.h"

#include <algorithm>
#include <cstring>

// Mutable global parameter set. Default-initialized from the literal block
// below at startup via evaluate_reload_weights() (called from a static ctor).
EvalWeights g_eval;

namespace {

// === TEXEL-TUNED-WEIGHTS-BEGIN ===
// Tuner edits the contents of this block in place. Keep formatting stable.

// PeSTO MG/EG piece values.
constexpr int LIT_MG_VAL[7] = { 0, 101, 407, 421, 550, 1252, 0 };
constexpr int LIT_EG_VAL[7] = { 0, 77, 297, 315, 561, 1005, 0 };

// PeSTO piece-square tables. Index 0..63 in (file + rank*8) order.
constexpr int LIT_PAWN_MG[64] = {
         0,     0,     0,     0,     0,     0,     0,     0,
      -21,   -5,  -10,  -18,  -13,    25,    31,  -16,
       -9,  -10,    10,   -2,    11,    18,    27,   -3,
      -15,  -11,    11,    25,    30,    20,     2,  -20,
       -2,    16,    17,    36,    38,    28,    18,  -13,
        18,    13,    43,    37,    86,   104,    40,     9,
        54,   112,    33,    87,    55,   106,  -35,  -95,
         0,     0,     0,     0,     0,     0,     0,     0
};
constexpr int LIT_PAWN_EG[64] = {
         0,     0,     0,     0,     0,     0,     0,     0,
        34,    23,    27,    22,    31,    16,     8,     7,
        25,    25,    12,    18,    18,    12,     6,     8,
        37,    32,    14,     7,     7,     8,    17,    19,
        49,    35,    21,     0,     5,    11,    26,    31,
        69,    63,    33,   -3,  -23,     0,    36,    48,
       163,   139,   124,    86,   103,    85,   147,   187,
         0,     0,     0,     0,     0,     0,     0,     0
};
constexpr int LIT_KNIGHT_MG[64] = {
     -120,     4,  -28,   -4,    22,     5,     7,     3,
       -8,  -28,    18,    32,    38,    42,    13,    12,
       -7,    15,    31,    40,    50,    45,    51,     4,
         7,    27,    37,    33,    50,    38,    45,    14,
         7,    38,    29,    70,    48,    95,    38,    48,
      -41,    74,    45,    77,   114,   153,   100,    62,
      -80,  -37,    92,    33,    24,    83,     1,   -6,
     -194, -105,  -61,  -49,    71, -113,  -23, -113
};
constexpr int LIT_KNIGHT_EG[64] = {
         6,  -40,   -8,     2,  -12,   -9,  -41,  -58,
      -23,   -6,     0,   -1,     0,  -13,  -10,  -39,
      -12,     4,     0,    17,    12,   -5,  -17,  -10,
       -4,     3,    21,    31,    20,    24,    15,   -8,
       -4,    11,    31,    25,    28,     9,    15,  -11,
      -11,  -16,    13,     8,  -13,  -18,  -24,  -41,
       -8,     8,  -27,     8,   -4,  -30,  -12,  -43,
      -35,  -22,     6,  -16,  -28,  -16,  -57,  -84
};
constexpr int LIT_BISHOP_MG[64] = {
      -20,    17,     9,     5,    10,     5,  -37,  -12,
        18,    39,    33,    22,    32,    35,    59,    17,
        13,    32,    34,    21,    25,    50,    30,    22,
         5,    22,    17,    40,    43,    13,    17,    17,
         1,    13,    20,    58,    41,    35,    16,     6,
      -24,    32,    45,    31,    41,    58,    32,   -7,
      -31,    14,  -29,  -44,    21,    58,    26,  -51,
      -36,  -18, -150, -109,  -77,  -69,  -34,     1
};
constexpr int LIT_BISHOP_EG[64] = {
      -11,   -1,  -14,     0,   -2,   -6,     9,   -9,
       -7,  -18,   -8,   -4,   -4,   -6,  -19,  -28,
       -5,   -1,     7,     9,    13,   -7,   -1,  -11,
       -3,     1,    11,    12,   -1,     7,   -5,   -6,
         1,    10,    10,     4,     7,     4,   -2,     3,
         9,   -4,   -4,   -4,   -9,   -5,     3,     8,
         0,   -3,    13,   -1,   -2,   -9,   -7,   -7,
      -10,  -14,    13,    10,    13,     2,   -4,  -23
};
constexpr int LIT_ROOK_MG[64] = {
         3,     4,    11,    22,    25,    31,   -9,    10,
      -28,  -10,  -15,   -2,     9,    26,    15,  -49,
      -38,  -20,  -15,  -19,   -3,    14,    11,  -16,
      -39,  -34,  -23,  -17,   -2,   -2,    18,  -24,
      -37,  -34,  -14,   -3,  -10,    33,  -11,  -25,
      -24,   -9,   -7,   -4,  -27,    39,    77,     5,
        10,     9,    49,    60,    80,    98,     4,    39,
         9,    25,   -8,    42,    47,   -4,   -5,     3
};
constexpr int LIT_ROOK_EG[64] = {
        12,    15,    16,     9,     4,     2,    11,  -10,
        19,    13,    20,    21,     8,     3,     0,    21,
        23,    23,    15,    20,    11,     4,     7,     6,
        32,    33,    32,    26,    17,    14,     7,    18,
        35,    32,    38,    24,    26,    18,    21,    32,
        35,    33,    30,    30,    30,    12,     5,    18,
        34,    35,    26,    20,     3,     8,    34,    24,
        32,    24,    37,    22,    22,    32,    31,    30
};
constexpr int LIT_QUEEN_MG[64] = {
        31,    30,    40,    54,    28,    19,    13,  -17,
       -3,    21,    41,    39,    48,    51,    35,    48,
         0,    26,     8,    15,    12,    19,    31,    25,
         7,  -34,   -4,   -9,     0,     1,     3,     9,
      -30,  -29,  -31,  -30,   -9,     2,   -4,     2,
       -4,  -20,     7,  -28,    20,    78,    56,    71,
      -17,  -44,  -17,  -22,  -52,    75,    49,    85,
      -11,  -21,   -3,   -3,   132,   126,    95,    80
};
constexpr int LIT_QUEEN_EG[64] = {
      -16,  -31,  -19,  -38,    10,  -13,   -3,  -23,
         8,   -6,  -17,   -9,   -9,   -7,  -26,  -12,
        29,  -17,    37,    19,    36,    45,    51,    48,
        10,    79,    47,    80,    60,    66,    82,    61,
        57,    60,    66,    83,    93,    82,   111,    94,
        15,    47,    29,   106,    92,    55,    64,    40,
        22,    61,    78,   100,   128,    51,    56,    25,
        20,    79,    76,    76,    13,    12,     8,    47
};
constexpr int LIT_KING_MG[64] = {
      -18,    44,    27,  -62,    11,  -35,    38,    27,
       -4,     0,  -30,  -86,  -68,  -50,   -1,    11,
      -24,  -30,  -50,  -82,  -88,  -83,  -41,  -60,
      -91,     9,  -72, -115, -128,  -99, -104, -124,
      -15,  -24,    15,  -49,  -58,  -51,  -39, -115,
        63,    66,    97,    30,    65,   114,   128,  -33,
       197,    84,    33,   117,    43,    27,  -25, -153,
     -146,   192,   187,   112, -134,  -73,    70,    14
};
constexpr int LIT_KING_EG[64] = {
      -71,  -53,  -27,   -1,  -28,   -6,  -41,  -72,
      -35,  -11,    16,    32,    32,    21,   -5,  -30,
      -22,     4,    26,    42,    46,    36,    14,   -3,
      -11,   -3,    39,    52,    56,    45,    29,     5,
      -16,    28,    27,    42,    42,    48,    38,    18,
       -9,    12,    11,    13,     9,    36,    29,    13,
      -59,   -1,    10,   -1,    13,    38,    29,    33,
      -77,  -79,  -60,  -44,     8,    26,  -14,  -31
};

// Pawn structure / king safety / mobility / misc weights.
constexpr int LIT_PASSED_MG[8] = { 0, -2, -8, -11, 6, 12, 64, 0 };
constexpr int LIT_PASSED_EG[8] = { 0, -3, 2, 26, 55, 129, 118, 0 };
constexpr int LIT_CONNECTED_PASSED_MG = 13;
constexpr int LIT_CONNECTED_PASSED_EG = 11;
constexpr int LIT_DOUBLED_MG = -1;
constexpr int LIT_DOUBLED_EG = -8;
constexpr int LIT_ISOLATED_MG = -19;
constexpr int LIT_ISOLATED_EG = -6;
constexpr int LIT_ROOK_OPEN_MG = 63;
constexpr int LIT_ROOK_OPEN_EG = -12;
constexpr int LIT_ROOK_SEMI_MG = 19;
constexpr int LIT_ROOK_SEMI_EG = 14;
constexpr int LIT_KING_SHELTER_CLOSE = 12;
constexpr int LIT_KING_SHELTER_FAR = 6;
constexpr int LIT_KING_SHELTER_MISSING = -20;
constexpr int LIT_KING_SHELTER_BEHIND = 21;
constexpr int LIT_MOBILITY_MG = 4;
constexpr int LIT_MOBILITY_EG = 3;
constexpr int LIT_BISHOP_PAIR_MG = 32;
constexpr int LIT_BISHOP_PAIR_EG = 54;
constexpr int LIT_TEMPO = 21;

// Section 3 additions. Hand-set defaults; the Texel pipeline doesn't tune
// these yet so they keep these literals. With NNUE loaded these have no
// effect (evaluate() short-circuits to NNUE::evaluate()); they only affect
// the HCE fallback (`setoption name EvalFile value <none>`) and the tuner.
constexpr int LIT_KING_TROPISM_MG     = 2;
constexpr int LIT_KING_TROPISM_EG     = 1;
constexpr int LIT_CONNECTED_ROOKS_MG  = 14;
constexpr int LIT_CONNECTED_ROOKS_EG  = 8;
constexpr int LIT_BISHOP_LONG_DIAG_MG = 12;
constexpr int LIT_BISHOP_LONG_DIAG_EG = 6;

// === TEXEL-TUNED-WEIGHTS-END ===

// Phase weight per piece type. Sums to 24 in the starting position; we use it
// to interpolate MG -> EG.
constexpr int PHASE[7] = { 0, 0, 1, 1, 2, 4, 0 };
constexpr int TOTAL_PHASE = 24;

// Precomputed "passed pawn front-cone" masks.
struct PassedMasks {
    Bitboard m[2][64];
    PassedMasks() {
        for (int sq = 0; sq < 64; sq++) {
            int f = file_of(sq), r = rank_of(sq);
            Bitboard files = file_bb(f);
            if (f > 0) files |= file_bb(f - 1);
            if (f < 7) files |= file_bb(f + 1);
            Bitboard ahead_w = 0, ahead_b = 0;
            for (int rr = r + 1; rr < 8; rr++) ahead_w |= rank_bb(rr);
            for (int rr = 0; rr < r; rr++)     ahead_b |= rank_bb(rr);
            m[WHITE][sq] = files & ahead_w;
            m[BLACK][sq] = files & ahead_b;
        }
    }
};
const PassedMasks passed_masks;

inline int pst_value(Color c, PieceType pt, int sq, bool eg) {
    int idx = c == WHITE ? sq : sq ^ 56;
    return eg ? g_eval.pst_eg[pt][idx] : g_eval.pst_mg[pt][idx];
}

int pawn_structure_term(const Board& b, Color us, bool eg) {
    int score = 0;
    Bitboard own = b.pieces(us, PAWN);
    Bitboard opp = b.pieces(~us, PAWN);
    int file_count[8] = { 0 };
    Bitboard tmp = own;
    while (tmp) {
        int sq = pop_lsb(tmp);
        file_count[file_of(sq)]++;
    }

    tmp = own;
    while (tmp) {
        int sq = pop_lsb(tmp);
        int f = file_of(sq), r = rank_of(sq);
        int rel = us == WHITE ? r : 7 - r;

        if (file_count[f] > 1) score += eg ? g_eval.doubled_eg : g_eval.doubled_mg;

        bool isolated = true;
        if (f > 0 && file_count[f - 1] > 0) isolated = false;
        if (f < 7 && file_count[f + 1] > 0) isolated = false;
        if (isolated) score += eg ? g_eval.isolated_eg : g_eval.isolated_mg;

        if (!(passed_masks.m[us][sq] & opp)) {
            score += eg ? g_eval.passed_eg[rel] : g_eval.passed_mg[rel];
            // Connected passers
            Bitboard adj = 0;
            if (f > 0) adj |= file_bb(f - 1);
            if (f < 7) adj |= file_bb(f + 1);
            Bitboard near_ranks = rank_bb(r);
            if (r > 0) near_ranks |= rank_bb(r - 1);
            if (r < 7) near_ranks |= rank_bb(r + 1);
            if (own & adj & near_ranks) score += eg ? g_eval.connected_passed_eg : g_eval.connected_passed_mg;
        }
    }
    return score;
}

int rook_files_term(const Board& b, Color us, bool eg) {
    int score = 0;
    Bitboard own_p = b.pieces(us, PAWN);
    Bitboard opp_p = b.pieces(~us, PAWN);
    Bitboard rooks = b.pieces(us, ROOK);
    while (rooks) {
        int sq = pop_lsb(rooks);
        Bitboard fbb = file_bb(file_of(sq));
        if (!(fbb & own_p)) {
            if (!(fbb & opp_p)) score += eg ? g_eval.rook_open_eg : g_eval.rook_open_mg;
            else                score += eg ? g_eval.rook_semi_eg : g_eval.rook_semi_mg;
        }
    }
    return score;
}

int king_safety_term(const Board& b, Color us) {
    int ksq = b.king_sq(us);
    int f = file_of(ksq);
    Bitboard own_pawns = b.pieces(us, PAWN);
    int score = 0;
    int kr = rank_of(ksq);
    int lo_f = std::max(0, f - 1), hi_f = std::min(7, f + 1);
    for (int ff = lo_f; ff <= hi_f; ff++) {
        Bitboard col = file_bb(ff) & own_pawns;
        if (!col) { score += g_eval.king_shelter_missing; continue; }
        int p_sq;
        int rel;
        if (us == WHITE) {
            p_sq = lsb(col);
            rel = rank_of(p_sq) - kr;
        } else {
            p_sq = msb(col);
            rel = kr - rank_of(p_sq);
        }
        if (rel == 1)      score += g_eval.king_shelter_close;
        else if (rel == 2) score += g_eval.king_shelter_far;
        else if (rel <= 0) score += g_eval.king_shelter_behind;
    }
    return score;
}

// Sum of (7 - chebyshev distance) over my N/B/R/Q to enemy king. Encourages
// piece placement near the enemy monarch -- the classic "tropism" term used in
// older HCE engines as a proxy for attacking potential.
int king_tropism_term(const Board& b, Color us, bool eg) {
    int ek = b.king_sq(~us);
    int ek_f = file_of(ek), ek_r = rank_of(ek);
    int sum = 0;
    Bitboard pieces = b.pieces(us, KNIGHT) | b.pieces(us, BISHOP)
                    | b.pieces(us, ROOK)   | b.pieces(us, QUEEN);
    while (pieces) {
        int sq = pop_lsb(pieces);
        int d = std::max(std::abs(file_of(sq) - ek_f),
                         std::abs(rank_of(sq) - ek_r));
        if (d < 7) sum += 7 - d;
    }
    return sum * (eg ? g_eval.king_tropism_eg : g_eval.king_tropism_mg);
}

// Two same-color rooks on the same file or rank with no pieces strictly
// between them. Connected rooks are notoriously powerful (back-rank and 7th-
// rank batteries); a flat per-pair bonus is the standard HCE shorthand.
int connected_rooks_term(const Board& b, Color us, bool eg) {
    Bitboard rooks = b.pieces(us, ROOK);
    if (popcount(rooks) < 2) return 0;
    Bitboard occ = b.pieces();
    int score = 0;
    int per = eg ? g_eval.connected_rooks_eg : g_eval.connected_rooks_mg;
    // Iterate all pairs (cheap: at most 2 rooks usually, occasionally 3 with
    // promotion). Inner check is "no pieces strictly between on shared line".
    int squares[8]; int n = 0;
    Bitboard tmp = rooks;
    while (tmp && n < 8) squares[n++] = pop_lsb(tmp);
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
        int s1 = squares[i], s2 = squares[j];
        Bitboard between = 0;
        if (file_of(s1) == file_of(s2)) {
            int lo = std::min(s1, s2), hi = std::max(s1, s2);
            for (int s = lo + 8; s < hi; s += 8) between |= sq_bb(s);
        } else if (rank_of(s1) == rank_of(s2)) {
            int lo = std::min(s1, s2), hi = std::max(s1, s2);
            for (int s = lo + 1; s < hi; ++s)  between |= sq_bb(s);
        } else {
            continue;
        }
        if (!(between & occ)) score += per;
    }
    return score;
}

// Bishop "controls" a long diagonal if it attacks at least 5 of the 16
// squares on a1-h8 / a8-h1 (counts both diagonals at once). A fianchettoed
// bishop with an empty long diagonal fits this naturally.
int bishop_long_diag_term(const Board& b, Color us, bool eg) {
    constexpr Bitboard LONG_DIAG_A1H8 = 0x8040201008040201ULL;
    constexpr Bitboard LONG_DIAG_A8H1 = 0x0102040810204080ULL;
    Bitboard long_diags = LONG_DIAG_A1H8 | LONG_DIAG_A8H1;
    Bitboard bishops = b.pieces(us, BISHOP);
    Bitboard occ = b.pieces();
    int score = 0;
    int per = eg ? g_eval.bishop_long_diag_eg : g_eval.bishop_long_diag_mg;
    while (bishops) {
        int sq = pop_lsb(bishops);
        Bitboard atts = bishop_attacks(sq, occ);
        if (popcount(atts & long_diags) >= 5) score += per;
    }
    return score;
}

int mobility_term(const Board& b, Color us, bool eg) {
    Bitboard occ = b.pieces();
    Bitboard us_bb = b.pieces(us);
    int total = 0;
    Bitboard knights = b.pieces(us, KNIGHT);
    while (knights) {
        int sq = pop_lsb(knights);
        Bitboard atts = knight_attacks_bb(sq) & ~us_bb;
        total += popcount(atts);
    }
    Bitboard bishops = b.pieces(us, BISHOP);
    while (bishops) {
        int sq = pop_lsb(bishops);
        Bitboard atts = bishop_attacks(sq, occ) & ~us_bb;
        total += popcount(atts);
    }
    Bitboard rooks = b.pieces(us, ROOK);
    while (rooks) {
        int sq = pop_lsb(rooks);
        Bitboard atts = rook_attacks(sq, occ) & ~us_bb;
        total += popcount(atts);
    }
    Bitboard queens = b.pieces(us, QUEEN);
    while (queens) {
        int sq = pop_lsb(queens);
        Bitboard atts = queen_attacks(sq, occ) & ~us_bb;
        total += popcount(atts);
    }
    return total * (eg ? g_eval.mobility_eg : g_eval.mobility_mg);
}

}  // namespace

void evaluate_reload_weights() {
    std::memset(&g_eval, 0, sizeof(g_eval));
    for (int pt = 0; pt < 7; ++pt) {
        g_eval.mg_val[pt] = LIT_MG_VAL[pt];
        g_eval.eg_val[pt] = LIT_EG_VAL[pt];
    }
    auto copy_pst = [&](int pt, const int* mg, const int* eg) {
        for (int sq = 0; sq < 64; sq++) {
            g_eval.pst_mg[pt][sq] = mg[sq];
            g_eval.pst_eg[pt][sq] = eg[sq];
        }
    };
    copy_pst(PAWN,   LIT_PAWN_MG,   LIT_PAWN_EG);
    copy_pst(KNIGHT, LIT_KNIGHT_MG, LIT_KNIGHT_EG);
    copy_pst(BISHOP, LIT_BISHOP_MG, LIT_BISHOP_EG);
    copy_pst(ROOK,   LIT_ROOK_MG,   LIT_ROOK_EG);
    copy_pst(QUEEN,  LIT_QUEEN_MG,  LIT_QUEEN_EG);
    copy_pst(KING,   LIT_KING_MG,   LIT_KING_EG);
    for (int r = 0; r < 8; r++) {
        g_eval.passed_mg[r] = LIT_PASSED_MG[r];
        g_eval.passed_eg[r] = LIT_PASSED_EG[r];
    }
    g_eval.connected_passed_mg = LIT_CONNECTED_PASSED_MG;
    g_eval.connected_passed_eg = LIT_CONNECTED_PASSED_EG;
    g_eval.doubled_mg = LIT_DOUBLED_MG;
    g_eval.doubled_eg = LIT_DOUBLED_EG;
    g_eval.isolated_mg = LIT_ISOLATED_MG;
    g_eval.isolated_eg = LIT_ISOLATED_EG;
    g_eval.rook_open_mg = LIT_ROOK_OPEN_MG;
    g_eval.rook_open_eg = LIT_ROOK_OPEN_EG;
    g_eval.rook_semi_mg = LIT_ROOK_SEMI_MG;
    g_eval.rook_semi_eg = LIT_ROOK_SEMI_EG;
    g_eval.king_shelter_close = LIT_KING_SHELTER_CLOSE;
    g_eval.king_shelter_far = LIT_KING_SHELTER_FAR;
    g_eval.king_shelter_missing = LIT_KING_SHELTER_MISSING;
    g_eval.king_shelter_behind = LIT_KING_SHELTER_BEHIND;
    g_eval.mobility_mg = LIT_MOBILITY_MG;
    g_eval.mobility_eg = LIT_MOBILITY_EG;
    g_eval.bishop_pair_mg = LIT_BISHOP_PAIR_MG;
    g_eval.bishop_pair_eg = LIT_BISHOP_PAIR_EG;
    g_eval.tempo = LIT_TEMPO;
    g_eval.king_tropism_mg     = LIT_KING_TROPISM_MG;
    g_eval.king_tropism_eg     = LIT_KING_TROPISM_EG;
    g_eval.connected_rooks_mg  = LIT_CONNECTED_ROOKS_MG;
    g_eval.connected_rooks_eg  = LIT_CONNECTED_ROOKS_EG;
    g_eval.bishop_long_diag_mg = LIT_BISHOP_LONG_DIAG_MG;
    g_eval.bishop_long_diag_eg = LIT_BISHOP_LONG_DIAG_EG;
}

namespace {
struct WeightInit { WeightInit() { evaluate_reload_weights(); } };
WeightInit weight_init_;
}  // namespace

int evaluate_hce(const Board& b) {
    int mg[2] = { 0, 0 };
    int eg[2] = { 0, 0 };
    int phase = 0;

    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = b.pieces(col, PieceType(pt));
            int n = popcount(bb);
            mg[c] += n * g_eval.mg_val[pt];
            eg[c] += n * g_eval.eg_val[pt];
            phase += n * PHASE[pt];
            while (bb) {
                int sq = pop_lsb(bb);
                mg[c] += pst_value(col, PieceType(pt), sq, false);
                eg[c] += pst_value(col, PieceType(pt), sq, true);
            }
        }
        if (popcount(b.pieces(col, BISHOP)) >= 2) {
            mg[c] += g_eval.bishop_pair_mg; eg[c] += g_eval.bishop_pair_eg;
        }
        mg[c] += pawn_structure_term(b, col, false);
        eg[c] += pawn_structure_term(b, col, true);
        mg[c] += rook_files_term(b, col, false);
        eg[c] += rook_files_term(b, col, true);
        mg[c] += king_safety_term(b, col);  // mg only
        mg[c] += mobility_term(b, col, false);
        eg[c] += mobility_term(b, col, true);
        mg[c] += king_tropism_term(b, col, false);
        eg[c] += king_tropism_term(b, col, true);
        mg[c] += connected_rooks_term(b, col, false);
        eg[c] += connected_rooks_term(b, col, true);
        mg[c] += bishop_long_diag_term(b, col, false);
        eg[c] += bishop_long_diag_term(b, col, true);
    }

    int mg_score = mg[WHITE] - mg[BLACK];
    int eg_score = eg[WHITE] - eg[BLACK];
    if (phase > TOTAL_PHASE) phase = TOTAL_PHASE;
    int score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE;

    // Tempo bonus
    score += (b.side_to_move() == WHITE ? g_eval.tempo : -g_eval.tempo);

    return b.side_to_move() == WHITE ? score : -score;
}

int evaluate(const Board& b) {
    if (NNUE::is_loaded()) return NNUE::evaluate(b);
    return evaluate_hce(b);
}

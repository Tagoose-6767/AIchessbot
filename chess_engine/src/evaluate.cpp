// evaluate.cpp — tapered (MG/EG) static evaluation with PeSTO PSTs.

#include "evaluate.h"
#include "board.h"
#include "movegen.h"

namespace {

// PeSTO MG/EG piece values.
constexpr int MG_VAL[7] = { 0, 82, 337, 365, 477, 1025, 0 };
constexpr int EG_VAL[7] = { 0, 94, 281, 297, 512,  936, 0 };

// PeSTO piece-square tables. Index 0..63 in (file + rank*8) order.
constexpr int PAWN_MG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    -35,  -1, -20, -23, -15,  24,  38, -22,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -14,  13,   6,  21,  23,  12,  17, -23,
     -6,   7,  26,  31,  65,  56,  25, -20,
     98, 134,  61,  95,  68, 126,  34, -11,
      0,   0,   0,   0,   0,   0,   0,   0,
};
constexpr int PAWN_EG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     13,   8,   8,  10,  13,   0,   2,  -7,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     32,  24,  13,   5,  -2,   4,  17,  17,
     94, 100,  85,  67,  56,  53,  82,  84,
    178, 173, 158, 134, 147, 132, 165, 187,
      0,   0,   0,   0,   0,   0,   0,   0,
};
constexpr int KNIGHT_MG[64] = {
   -105, -21, -58, -33, -17, -28, -19, -23,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -13,   4,  16,  13,  28,  19,  21,  -8,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -47,  60,  37,  65,  84, 129,  73,  44,
    -73, -41,  72,  36,  23,  62,   7, -17,
   -167, -89, -34, -49,  61, -97, -15,-107,
};
constexpr int KNIGHT_EG[64] = {
    -29, -51, -23, -15, -22, -18, -50, -64,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -58, -38, -13, -28, -31, -27, -63, -99,
};
constexpr int BISHOP_MG[64] = {
    -33,  -3, -14, -21, -13, -12, -39, -21,
      4,  15,  16,   0,   7,  21,  33,   1,
      0,  15,  15,  15,  14,  27,  18,  10,
     -6,  13,  13,  26,  34,  12,  10,   4,
     -4,   5,  19,  50,  37,  37,   7,  -2,
    -16,  37,  43,  40,  35,  50,  37,  -2,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -29,   4, -82, -37, -25, -42,   7,  -8,
};
constexpr int BISHOP_EG[64] = {
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
     -3,   9,  12,   9,  14,  10,   3,   2,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
};
constexpr int ROOK_MG[64] = {
    -19, -13,   1,  17,  16,   7, -37, -26,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -24, -11,   7,  26,  24,  35,  -8, -20,
     -5,  19,  26,  36,  17,  45,  61,  16,
     27,  32,  58,  62,  80,  67,  26,  44,
     32,  42,  32,  51,  63,   9,  31,  43,
};
constexpr int ROOK_EG[64] = {
     -9,   2,   3,  -1,  -5, -13,   4, -20,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
      4,   3,  13,   1,   2,   1,  -1,   2,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
     11,  13,  13,  11,  -3,   3,   8,   3,
     13,  10,  18,  15,  12,  12,   8,   5,
};
constexpr int QUEEN_MG[64] = {
     -1, -18,  -9,  10, -15, -25, -31, -50,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -28,   0,  29,  12,  59,  44,  43,  45,
};
constexpr int QUEEN_EG[64] = {
    -33, -28, -22, -43,  -5, -32, -20, -41,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -18,  28,  19,  47,  31,  34,  39,  23,
      3,  22,  24,  45,  57,  40,  57,  36,
    -20,   6,   9,  49,  47,  35,  19,   9,
    -17,  20,  32,  41,  58,  25,  30,   0,
     -9,  22,  22,  27,  27,  19,  10,  20,
};
constexpr int KING_MG[64] = {
    -15,  36,  12, -54,   8, -28,  24,  14,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -14, -14, -22, -46, -44, -30, -15, -27,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -17, -20, -12, -27, -30, -25, -14, -36,
     -9,  24,   2, -16, -20,   6,  22, -22,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -65,  23,  16, -15, -56, -34,   2,  13,
};
constexpr int KING_EG[64] = {
    -53, -34, -21, -11, -28, -14, -24, -43,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -18,  -4,  21,  24,  27,  23,   9, -11,
     -8,  22,  24,  27,  26,  33,  26,   3,
     10,  17,  23,  15,  20,  45,  44,  13,
    -12,  17,  14,  17,  17,  38,  23,  11,
    -74, -35, -18, -18, -11,  15,   4, -17,
};

const int* PST_MG[7] = { nullptr, PAWN_MG, KNIGHT_MG, BISHOP_MG, ROOK_MG, QUEEN_MG, KING_MG };
const int* PST_EG[7] = { nullptr, PAWN_EG, KNIGHT_EG, BISHOP_EG, ROOK_EG, QUEEN_EG, KING_EG };

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

constexpr int PASSED_BONUS_MG[8] = { 0, 5, 10, 20, 35, 60, 100, 0 };
constexpr int PASSED_BONUS_EG[8] = { 0, 10, 20, 35, 60, 100, 150, 0 };

inline int pst_value(Color c, PieceType pt, int sq, bool eg) {
    int idx = c == WHITE ? sq : sq ^ 56;
    return eg ? PST_EG[pt][idx] : PST_MG[pt][idx];
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

        if (file_count[f] > 1) score += eg ? -20 : -10;

        bool isolated = true;
        if (f > 0 && file_count[f - 1] > 0) isolated = false;
        if (f < 7 && file_count[f + 1] > 0) isolated = false;
        if (isolated) score += eg ? -10 : -15;

        if (!(passed_masks.m[us][sq] & opp)) {
            score += eg ? PASSED_BONUS_EG[rel] : PASSED_BONUS_MG[rel];
            // Connected passers
            Bitboard adj = 0;
            if (f > 0) adj |= file_bb(f - 1);
            if (f < 7) adj |= file_bb(f + 1);
            Bitboard near_ranks = rank_bb(r);
            if (r > 0) near_ranks |= rank_bb(r - 1);
            if (r < 7) near_ranks |= rank_bb(r + 1);
            if (own & adj & near_ranks) score += eg ? 20 : 10;
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
            if (!(fbb & opp_p)) score += eg ? 15 : 25;
            else                score += eg ? 7  : 12;
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
        if (!col) { score -= 25; continue; }
        int p_sq;
        int rel;
        if (us == WHITE) {
            p_sq = lsb(col);
            rel = rank_of(p_sq) - kr;
        } else {
            p_sq = msb(col);
            rel = kr - rank_of(p_sq);
        }
        if (rel == 1) score += 5;
        else if (rel == 2) score += 2;
        else if (rel <= 0) score -= 15;
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
    return eg ? total * 2 : total * 3;
}

}  // namespace

int evaluate(const Board& b) {
    int mg[2] = { 0, 0 };
    int eg[2] = { 0, 0 };
    int phase = 0;

    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = b.pieces(col, PieceType(pt));
            int n = popcount(bb);
            mg[c] += n * MG_VAL[pt];
            eg[c] += n * EG_VAL[pt];
            phase += n * PHASE[pt];
            while (bb) {
                int sq = pop_lsb(bb);
                mg[c] += pst_value(col, PieceType(pt), sq, false);
                eg[c] += pst_value(col, PieceType(pt), sq, true);
            }
        }
        if (popcount(b.pieces(col, BISHOP)) >= 2) {
            mg[c] += 30; eg[c] += 50;
        }
        mg[c] += pawn_structure_term(b, col, false);
        eg[c] += pawn_structure_term(b, col, true);
        mg[c] += rook_files_term(b, col, false);
        eg[c] += rook_files_term(b, col, true);
        mg[c] += king_safety_term(b, col);  // mg only
        mg[c] += mobility_term(b, col, false);
        eg[c] += mobility_term(b, col, true);
    }

    int mg_score = mg[WHITE] - mg[BLACK];
    int eg_score = eg[WHITE] - eg[BLACK];
    if (phase > TOTAL_PHASE) phase = TOTAL_PHASE;
    int score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE;

    // Tempo bonus
    score += (b.side_to_move() == WHITE ? 10 : -10);

    return b.side_to_move() == WHITE ? score : -score;
}

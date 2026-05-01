// movegen.cpp — magic bitboards, attack lookups, pseudo-legal generation,
// move ordering, and perft.

#include "movegen.h"
#include "board.h"

#include <cstring>
#include <iostream>
#include <algorithm>

// --- Pawn / knight / king attack tables --------------------------------------
static Bitboard PAWN_ATTACKS_T[2][64];
static Bitboard KNIGHT_ATTACKS_T[64];
static Bitboard KING_ATTACKS_T[64];

Bitboard knight_attacks_bb(int sq) { return KNIGHT_ATTACKS_T[sq]; }
Bitboard king_attacks_bb(int sq)   { return KING_ATTACKS_T[sq]; }
Bitboard pawn_attacks_bb(Color c, int sq) { return PAWN_ATTACKS_T[c][sq]; }

Bitboard pawn_attacks_set(Color c, Bitboard pawns) {
    if (c == WHITE) {
        return ((pawns & ~FILE_A_BB) << 7) | ((pawns & ~FILE_H_BB) << 9);
    } else {
        return ((pawns & ~FILE_A_BB) >> 9) | ((pawns & ~FILE_H_BB) >> 7);
    }
}

// --- Magic bitboards ---------------------------------------------------------
struct Magic {
    Bitboard mask;
    Bitboard magic;
    Bitboard* attacks;
    int shift;
    int index(Bitboard occ) const {
        return int(((occ & mask) * magic) >> shift);
    }
};

static Magic ROOK_MAGICS[64];
static Magic BISHOP_MAGICS[64];

// Plain magic — fixed 4096 / 512 table per square. Some squares use only a
// fraction of these; the unused entries are harmless.
static Bitboard ROOK_TABLE[64][4096];
static Bitboard BISHOP_TABLE[64][512];

Bitboard rook_attacks(int sq, Bitboard occ)   { return ROOK_MAGICS[sq].attacks[ROOK_MAGICS[sq].index(occ)]; }
Bitboard bishop_attacks(int sq, Bitboard occ) { return BISHOP_MAGICS[sq].attacks[BISHOP_MAGICS[sq].index(occ)]; }

// Slow ray-based attacks for magic-table initialization.
static Bitboard slide_attacks(int sq, Bitboard occ, const int* deltas) {
    Bitboard atts = 0;
    int sf = file_of(sq), sr = rank_of(sq);
    for (int d = 0; d < 4; d++) {
        int df = deltas[d * 2], dr = deltas[d * 2 + 1];
        int f = sf + df, r = sr + dr;
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            int s = make_square(f, r);
            atts |= sq_bb(s);
            if (occ & sq_bb(s)) break;
            f += df; r += dr;
        }
    }
    return atts;
}

static Bitboard rook_attacks_slow(int sq, Bitboard occ) {
    static const int deltas[] = { 1,0, -1,0, 0,1, 0,-1 };
    return slide_attacks(sq, occ, deltas);
}
static Bitboard bishop_attacks_slow(int sq, Bitboard occ) {
    static const int deltas[] = { 1,1, 1,-1, -1,1, -1,-1 };
    return slide_attacks(sq, occ, deltas);
}

// Edge-stripped "relevant occupancy" mask for magic indexing.
static Bitboard rook_mask(int sq) {
    Bitboard m = rook_attacks_slow(sq, 0);
    int f = file_of(sq), r = rank_of(sq);
    if (r != 0) m &= ~RANK_1_BB;
    if (r != 7) m &= ~RANK_8_BB;
    if (f != 0) m &= ~FILE_A_BB;
    if (f != 7) m &= ~FILE_H_BB;
    return m;
}
static Bitboard bishop_mask(int sq) {
    Bitboard m = bishop_attacks_slow(sq, 0);
    return m & ~(RANK_1_BB | RANK_8_BB | FILE_A_BB | FILE_H_BB);
}

// Build a blocker subset corresponding to `index` over the bits of `mask`.
static Bitboard index_to_blockers(int index, Bitboard mask) {
    Bitboard b = 0;
    int bits = popcount(mask);
    for (int i = 0; i < bits; i++) {
        int sq = pop_lsb(mask);
        if (index & (1 << i)) b |= sq_bb(sq);
    }
    return b;
}

// Tiny xorshift PRNG with sparse-bit "magic candidates".
struct PRNG {
    U64 s;
    U64 next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 2685821657736338717ULL; }
    U64 sparse() { return next() & next() & next(); }
};

static U64 find_magic(int sq, bool bishop, Bitboard mask, Bitboard* table) {
    int bits = popcount(mask);
    int n = 1 << bits;
    Bitboard blockers[4096], attacks[4096];
    for (int i = 0; i < n; i++) {
        blockers[i] = index_to_blockers(i, mask);
        attacks[i]  = bishop ? bishop_attacks_slow(sq, blockers[i])
                             : rook_attacks_slow(sq, blockers[i]);
    }
    Bitboard used[4096];
    PRNG rng{ 0x1234567ABCDEFULL ^ U64(sq) * 0x9E3779B97F4A7C15ULL ^ U64(bishop) };
    for (int tries = 0; tries < 100000000; tries++) {
        U64 magic = rng.sparse();
        if (popcount((magic * mask) & 0xFF00000000000000ULL) < 6) continue;
        std::memset(used, 0, sizeof(Bitboard) * n);
        bool ok = true;
        for (int i = 0; i < n; i++) {
            int idx = int((blockers[i] * magic) >> (64 - bits));
            if (used[idx] == 0) used[idx] = attacks[i];
            else if (used[idx] != attacks[i]) { ok = false; break; }
        }
        if (ok) {
            for (int i = 0; i < n; i++) {
                int idx = int((blockers[i] * magic) >> (64 - bits));
                table[idx] = attacks[i];
            }
            return magic;
        }
    }
    return 0;
}

void init_movegen() {
    // Pawn attacks
    for (int sq = 0; sq < 64; sq++) {
        Bitboard b = sq_bb(sq);
        PAWN_ATTACKS_T[WHITE][sq] = ((b & ~FILE_A_BB) << 7) | ((b & ~FILE_H_BB) << 9);
        PAWN_ATTACKS_T[BLACK][sq] = ((b & ~FILE_A_BB) >> 9) | ((b & ~FILE_H_BB) >> 7);
    }
    // Knight attacks
    static const int kn_dx[] = { 1, 2, 2, 1, -1, -2, -2, -1 };
    static const int kn_dy[] = { 2, 1, -1, -2, -2, -1, 1, 2 };
    for (int sq = 0; sq < 64; sq++) {
        int f = file_of(sq), r = rank_of(sq);
        Bitboard b = 0;
        for (int i = 0; i < 8; i++) {
            int nf = f + kn_dx[i], nr = r + kn_dy[i];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) b |= sq_bb(make_square(nf, nr));
        }
        KNIGHT_ATTACKS_T[sq] = b;
    }
    // King attacks
    for (int sq = 0; sq < 64; sq++) {
        int f = file_of(sq), r = rank_of(sq);
        Bitboard b = 0;
        for (int df = -1; df <= 1; df++) for (int dr = -1; dr <= 1; dr++) {
            if (!df && !dr) continue;
            int nf = f + df, nr = r + dr;
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) b |= sq_bb(make_square(nf, nr));
        }
        KING_ATTACKS_T[sq] = b;
    }
    // Magic init.
    for (int sq = 0; sq < 64; sq++) {
        ROOK_MAGICS[sq].mask = rook_mask(sq);
        ROOK_MAGICS[sq].shift = 64 - popcount(ROOK_MAGICS[sq].mask);
        ROOK_MAGICS[sq].attacks = ROOK_TABLE[sq];
        ROOK_MAGICS[sq].magic = find_magic(sq, false, ROOK_MAGICS[sq].mask, ROOK_TABLE[sq]);

        BISHOP_MAGICS[sq].mask = bishop_mask(sq);
        BISHOP_MAGICS[sq].shift = 64 - popcount(BISHOP_MAGICS[sq].mask);
        BISHOP_MAGICS[sq].attacks = BISHOP_TABLE[sq];
        BISHOP_MAGICS[sq].magic = find_magic(sq, true, BISHOP_MAGICS[sq].mask, BISHOP_TABLE[sq]);
    }
}

// ---------------------------------------------------------------------------
// Pseudo-legal generation. Output is filtered to legal in generate_legal.
// ---------------------------------------------------------------------------
static inline void emit_promos(MoveList& ml, int from, int to) {
    ml.push(make_promo(from, to, QUEEN));
    ml.push(make_promo(from, to, ROOK));
    ml.push(make_promo(from, to, BISHOP));
    ml.push(make_promo(from, to, KNIGHT));
}

static void gen_pawn_moves(const Board& b, MoveList& ml, bool only_caps) {
    Color us = b.side_to_move(), them = ~us;
    Bitboard pawns = b.pieces(us, PAWN);
    Bitboard empty = ~b.pieces();
    Bitboard their_pieces = b.pieces(them);

    int up    = us == WHITE ? 8 : -8;
    Bitboard rank3 = us == WHITE ? RANK_4_BB >> 8 : RANK_5_BB << 8;  // rank3 from white perspective
    // Let's redo: doubled push squares require single push to land on rank3 (white) / rank6 (black).
    Bitboard pre_promotion = us == WHITE ? RANK_7_BB : RANK_2_BB;

    // Pushes (only when not in capture-only mode)
    if (!only_caps) {
        Bitboard non_promo = pawns & ~pre_promotion;
        Bitboard promo     = pawns & pre_promotion;

        Bitboard single = (us == WHITE ? non_promo << 8 : non_promo >> 8) & empty;
        Bitboard dbl_src = (us == WHITE ? (single & RANK_4_BB >> 8) : (single & RANK_5_BB << 8));
        // dbl pushes only from rank 2 (white) / rank 7 (black). Source pawn must be on rank2/7.
        Bitboard r2_pawns = pawns & (us == WHITE ? RANK_2_BB : RANK_7_BB);
        Bitboard one_step = (us == WHITE ? r2_pawns << 8 : r2_pawns >> 8) & empty;
        Bitboard two_step = (us == WHITE ? one_step << 8 : one_step >> 8) & empty;

        Bitboard pushes = single;
        while (pushes) {
            int to = pop_lsb(pushes);
            int from = to - up;
            ml.push(make_normal(from, to));
        }
        while (two_step) {
            int to = pop_lsb(two_step);
            int from = to - 2 * up;
            ml.push(make_normal(from, to));
        }
        // Promotion pushes
        Bitboard promo_push = (us == WHITE ? promo << 8 : promo >> 8) & empty;
        while (promo_push) {
            int to = pop_lsb(promo_push);
            int from = to - up;
            emit_promos(ml, from, to);
        }
    }

    // Captures
    Bitboard non_promo = pawns & ~pre_promotion;
    Bitboard promo     = pawns & pre_promotion;
    int up_l = us == WHITE ? 7 : -9;  // up-left
    int up_r = us == WHITE ? 9 : -7;  // up-right
    Bitboard left_caps  = (us == WHITE ? (non_promo & ~FILE_A_BB) << 7 : (non_promo & ~FILE_A_BB) >> 9) & their_pieces;
    Bitboard right_caps = (us == WHITE ? (non_promo & ~FILE_H_BB) << 9 : (non_promo & ~FILE_H_BB) >> 7) & their_pieces;
    while (left_caps) {
        int to = pop_lsb(left_caps);
        int from = to - up_l;
        ml.push(make_normal(from, to));
    }
    while (right_caps) {
        int to = pop_lsb(right_caps);
        int from = to - up_r;
        ml.push(make_normal(from, to));
    }
    // Promotion captures
    Bitboard pl = (us == WHITE ? (promo & ~FILE_A_BB) << 7 : (promo & ~FILE_A_BB) >> 9) & their_pieces;
    Bitboard pr = (us == WHITE ? (promo & ~FILE_H_BB) << 9 : (promo & ~FILE_H_BB) >> 7) & their_pieces;
    while (pl) { int to = pop_lsb(pl); emit_promos(ml, to - up_l, to); }
    while (pr) { int to = pop_lsb(pr); emit_promos(ml, to - up_r, to); }

    // En passant
    int ep = b.ep_sq();
    if (ep != SQ_NONE) {
        Bitboard from_pawns = pawns & PAWN_ATTACKS_T[them][ep];
        while (from_pawns) {
            int from = pop_lsb(from_pawns);
            ml.push(make_ep(from, ep));
        }
    }
}

static void gen_piece_moves(const Board& b, MoveList& ml, PieceType pt, Bitboard target) {
    Color us = b.side_to_move();
    Bitboard occ = b.pieces();
    Bitboard pieces = b.pieces(us, pt);
    while (pieces) {
        int from = pop_lsb(pieces);
        Bitboard atts;
        switch (pt) {
            case KNIGHT: atts = KNIGHT_ATTACKS_T[from]; break;
            case BISHOP: atts = bishop_attacks(from, occ); break;
            case ROOK:   atts = rook_attacks(from, occ); break;
            case QUEEN:  atts = queen_attacks(from, occ); break;
            case KING:   atts = KING_ATTACKS_T[from]; break;
            default: atts = 0;
        }
        atts &= target;
        while (atts) ml.push(make_normal(from, pop_lsb(atts)));
    }
}

static void gen_castling(const Board& b, MoveList& ml) {
    Color us = b.side_to_move();
    if (b.is_check()) return;
    int rights = b.castling();
    int home_rank = us == WHITE ? 0 : 56;
    int king_from = home_rank + 4;  // e1 / e8
    Bitboard occ = b.pieces();

    auto path_safe = [&](int from, int to) {
        // Squares the king crosses, including from and to, must not be attacked.
        int s = std::min(from, to);
        int e = std::max(from, to);
        for (int sq = s; sq <= e; ++sq)
            if (b.attackers_to(sq, occ) & b.pieces(~us)) return false;
        return true;
    };

    int oo  = (us == WHITE) ? WHITE_OO  : BLACK_OO;
    int ooo = (us == WHITE) ? WHITE_OOO : BLACK_OOO;

    if (rights & oo) {
        int rook = home_rank + 7;
        // f1, g1 must be empty (or f8, g8).
        if (!(occ & (sq_bb(home_rank + 5) | sq_bb(home_rank + 6)))
            && b.piece_on(rook) == make_piece(us, ROOK)
            && path_safe(king_from, home_rank + 6)) {
            ml.push(make_castle(king_from, home_rank + 6));
        }
    }
    if (rights & ooo) {
        int rook = home_rank + 0;
        // b1, c1, d1 must be empty (b1 needn't be path-safe but must be empty).
        if (!(occ & (sq_bb(home_rank + 1) | sq_bb(home_rank + 2) | sq_bb(home_rank + 3)))
            && b.piece_on(rook) == make_piece(us, ROOK)
            && path_safe(king_from, home_rank + 2)) {
            ml.push(make_castle(king_from, home_rank + 2));
        }
    }
}

void generate_pseudo(const Board& b, MoveList& list) {
    Color us = b.side_to_move();
    Bitboard target = ~b.pieces(us);
    gen_pawn_moves(b, list, false);
    gen_piece_moves(b, list, KNIGHT, target);
    gen_piece_moves(b, list, BISHOP, target);
    gen_piece_moves(b, list, ROOK,   target);
    gen_piece_moves(b, list, QUEEN,  target);
    gen_piece_moves(b, list, KING,   target);
    gen_castling(b, list);
}

void generate_captures(const Board& b, MoveList& list) {
    Color us = b.side_to_move();
    Bitboard target = b.pieces(~us);
    gen_pawn_moves(b, list, true);
    gen_piece_moves(b, list, KNIGHT, target);
    gen_piece_moves(b, list, BISHOP, target);
    gen_piece_moves(b, list, ROOK,   target);
    gen_piece_moves(b, list, QUEEN,  target);
    gen_piece_moves(b, list, KING,   target);
}

void generate_legal(const Board& b, MoveList& list) {
    MoveList pseudo;
    generate_pseudo(b, pseudo);
    Board& nb = const_cast<Board&>(b);
    for (int i = 0; i < pseudo.size; i++) {
        Move m = pseudo.moves[i].move;
        nb.make_move(m);
        // After make_move, side to move flipped. The mover's king is the
        // OPPOSITE of the now-current stm. So we check whether the stm's
        // pieces attack the mover's king.
        Color mover = ~nb.side_to_move();
        bool legal = !(nb.attackers_to(nb.king_sq(mover), nb.pieces()) & nb.pieces(nb.side_to_move()));
        nb.unmake_move(m);
        if (legal) list.push(m);
    }
}

// ---------------------------------------------------------------------------
// Move ordering
// ---------------------------------------------------------------------------
// MVV-LVA: capturing high-value piece with low-value piece scores best.
static int mvv_lva(PieceType victim, PieceType attacker) {
    return PieceValueSimple[victim] * 16 - int(attacker);
}

void score_moves(const Board& b, MoveList& list,
                 Move tt_move,
                 const Move* killers,
                 const int (*history)[64],
                 Move counter) {
    for (int i = 0; i < list.size; i++) {
        Move m = list.moves[i].move;
        if (m == tt_move) { list.moves[i].score = 1 << 30; continue; }

        bool cap = b.is_capture(m);
        if (cap) {
            PieceType victim = (type_of_move(m) == MT_ENPASSANT) ? PAWN : b.type_on(to_sq(m));
            PieceType attacker = b.type_on(from_sq(m));
            int s = 1 << 22;
            s += mvv_lva(victim, attacker);
            if (!b.see_ge(m, 0)) s -= (1 << 22) - (1 << 19);  // bad capture: demote below quiets
            list.moves[i].score = s;
            continue;
        }
        if (type_of_move(m) == MT_PROMOTION) {
            list.moves[i].score = (1 << 24) + int(promotion_of(m));
            continue;
        }
        if (m == killers[0]) { list.moves[i].score = (1 << 21); continue; }
        if (m == killers[1]) { list.moves[i].score = (1 << 21) - 1; continue; }
        if (m == counter)    { list.moves[i].score = (1 << 20); continue; }

        Piece p = b.piece_on(from_sq(m));
        list.moves[i].score = history[p][to_sq(m)];
    }
}

// ---------------------------------------------------------------------------
// Perft
// ---------------------------------------------------------------------------
uint64_t perft(Board& b, int depth) {
    if (depth == 0) return 1;
    MoveList ml;
    generate_legal(b, ml);
    if (depth == 1) return uint64_t(ml.size);
    uint64_t n = 0;
    for (int i = 0; i < ml.size; i++) {
        b.make_move(ml.moves[i].move);
        n += perft(b, depth - 1);
        b.unmake_move(ml.moves[i].move);
    }
    return n;
}

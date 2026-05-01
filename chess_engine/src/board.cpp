// board.cpp — position representation, FEN, make/unmake, queries.

#include "board.h"
#include "movegen.h"
#include "tt.h"

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <iostream>
#include <sstream>

// AND'ing this in when a piece moves on/off a square cheaply clears the
// affected castling rights (king's home or rook's home).
const int CASTLING_MASK[64] = {
    /* a1 */ 0xF & ~WHITE_OOO, 0xF, 0xF, 0xF, /* e1 */ 0xF & ~(WHITE_OO|WHITE_OOO), 0xF, 0xF, /* h1 */ 0xF & ~WHITE_OO,
    0xF,0xF,0xF,0xF,0xF,0xF,0xF,0xF,
    0xF,0xF,0xF,0xF,0xF,0xF,0xF,0xF,
    0xF,0xF,0xF,0xF,0xF,0xF,0xF,0xF,
    0xF,0xF,0xF,0xF,0xF,0xF,0xF,0xF,
    0xF,0xF,0xF,0xF,0xF,0xF,0xF,0xF,
    0xF,0xF,0xF,0xF,0xF,0xF,0xF,0xF,
    /* a8 */ 0xF & ~BLACK_OOO, 0xF, 0xF, 0xF, /* e8 */ 0xF & ~(BLACK_OO|BLACK_OOO), 0xF, 0xF, /* h8 */ 0xF & ~BLACK_OO,
};

Board::Board() { clear(); }

void Board::clear() {
    for (auto& bb : byType) bb = 0;
    for (auto& bb : byColor) bb = 0;
    occupied = 0;
    for (auto& p : board) p = NO_PIECE;
    king_square[0] = king_square[1] = SQ_NONE;
    stm = WHITE;
    game_ply = 0;
    st = state_stack;
    st->castling_rights = 0;
    st->ep_square = SQ_NONE;
    st->halfmove_clock = 0;
    st->captured = NO_PIECE_TYPE;
    st->key = 0;
    st->previous = nullptr;
}

void Board::put_piece(Piece p, int sq) {
    Bitboard b = sq_bb(sq);
    byColor[color_of(p)] |= b;
    byType[type_of_piece(p)]   |= b;
    occupied |= b;
    board[sq] = p;
    if (type_of_piece(p) == KING) king_square[color_of(p)] = sq;
    st->key ^= Zobrist::piece_square[p][sq];
}

void Board::remove_piece(int sq) {
    Piece p = board[sq];
    Bitboard b = sq_bb(sq);
    byColor[color_of(p)] ^= b;
    byType[type_of_piece(p)]   ^= b;
    occupied ^= b;
    board[sq] = NO_PIECE;
    st->key ^= Zobrist::piece_square[p][sq];
}

void Board::move_piece(int from, int to) {
    Piece p = board[from];
    Bitboard fromto = sq_bb(from) | sq_bb(to);
    byColor[color_of(p)] ^= fromto;
    byType[type_of_piece(p)] ^= fromto;
    occupied ^= fromto;
    board[to] = p;
    board[from] = NO_PIECE;
    if (type_of_piece(p) == KING) king_square[color_of(p)] = to;
    st->key ^= Zobrist::piece_square[p][from] ^ Zobrist::piece_square[p][to];
}

// ---------------------------------------------------------------------------
// FEN parsing
// ---------------------------------------------------------------------------
void Board::set(const std::string& fen) {
    clear();
    std::istringstream iss(fen);
    std::string board_part, stm_part, castle_part, ep_part;
    int hm = 0, fm = 1;
    iss >> board_part >> stm_part >> castle_part >> ep_part >> hm >> fm;

    int rank = 7, file = 0;
    for (char c : board_part) {
        if (c == '/') { --rank; file = 0; }
        else if (isdigit((unsigned char)c)) file += c - '0';
        else {
            Color col = isupper((unsigned char)c) ? WHITE : BLACK;
            PieceType pt = NO_PIECE_TYPE;
            switch (tolower((unsigned char)c)) {
                case 'p': pt = PAWN;   break;
                case 'n': pt = KNIGHT; break;
                case 'b': pt = BISHOP; break;
                case 'r': pt = ROOK;   break;
                case 'q': pt = QUEEN;  break;
                case 'k': pt = KING;   break;
            }
            put_piece(make_piece(col, pt), make_square(file, rank));
            ++file;
        }
    }

    stm = (stm_part == "w" ? WHITE : BLACK);
    if (stm == BLACK) st->key ^= Zobrist::side;

    int rights = 0;
    for (char c : castle_part) {
        switch (c) {
            case 'K': rights |= WHITE_OO;  break;
            case 'Q': rights |= WHITE_OOO; break;
            case 'k': rights |= BLACK_OO;  break;
            case 'q': rights |= BLACK_OOO; break;
        }
    }
    st->castling_rights = rights;
    st->key ^= Zobrist::castling[rights];

    if (ep_part != "-") {
        int f = ep_part[0] - 'a';
        int r = ep_part[1] - '1';
        st->ep_square = make_square(f, r);
        st->key ^= Zobrist::ep_file[f];
    }
    st->halfmove_clock = hm;
    game_ply = 0;
    history_keys[0] = st->key;
}

std::string Board::fen() const {
    std::string s;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board[make_square(f, r)];
            if (p == NO_PIECE) { ++empty; continue; }
            if (empty) { s += char('0' + empty); empty = 0; }
            const char* names = " PNBRQK pnbrqk";
            char ch = names[(color_of(p) == WHITE) ? type_of_piece(p) : (type_of_piece(p) + 7)];
            s += ch;
        }
        if (empty) s += char('0' + empty);
        if (r) s += '/';
    }
    s += stm == WHITE ? " w " : " b ";
    int cr = st->castling_rights;
    std::string cs;
    if (cr & WHITE_OO) cs += 'K';
    if (cr & WHITE_OOO) cs += 'Q';
    if (cr & BLACK_OO) cs += 'k';
    if (cr & BLACK_OOO) cs += 'q';
    s += cs.empty() ? "-" : cs;
    s += ' ';
    if (st->ep_square == SQ_NONE) s += '-';
    else { s += char('a' + file_of(st->ep_square)); s += char('1' + rank_of(st->ep_square)); }
    s += ' ' + std::to_string(st->halfmove_clock) + ' ' + std::to_string(1 + game_ply / 2);
    return s;
}

void Board::print() const {
    std::cerr << "  +-----------------+\n";
    for (int r = 7; r >= 0; --r) {
        std::cerr << r + 1 << " | ";
        for (int f = 0; f < 8; ++f) {
            Piece p = board[make_square(f, r)];
            const char* names = ".PNBRQK.pnbrqk";
            char ch = (p == NO_PIECE) ? '.' :
                      names[(color_of(p) == WHITE) ? type_of_piece(p) : (type_of_piece(p) + 7)];
            std::cerr << ch << ' ';
        }
        std::cerr << "|\n";
    }
    std::cerr << "  +-----------------+\n    a b c d e f g h\n";
    std::cerr << "fen: " << fen() << '\n';
}

// ---------------------------------------------------------------------------
// Attackers
// ---------------------------------------------------------------------------
Bitboard Board::attackers_to(int sq, Bitboard occ) const {
    Bitboard rooks_queens   = byType[ROOK]   | byType[QUEEN];
    Bitboard bishops_queens = byType[BISHOP] | byType[QUEEN];
    return (pawn_attacks_bb(BLACK, sq) & pieces(WHITE, PAWN))
         | (pawn_attacks_bb(WHITE, sq) & pieces(BLACK, PAWN))
         | (knight_attacks_bb(sq)      & byType[KNIGHT])
         | (king_attacks_bb(sq)        & byType[KING])
         | (rook_attacks(sq, occ)      & rooks_queens)
         | (bishop_attacks(sq, occ)    & bishops_queens);
}

bool Board::gives_check(Move m) const {
    int from = from_sq(m), to = to_sq(m);
    int ksq = king_square[~stm];
    MoveType mt = type_of_move(m);
    Piece moved = board[from];
    PieceType pt = type_of_piece(moved);
    if (mt == MT_PROMOTION) pt = promotion_of(m);

    // Direct check: does moved piece attack king from `to`?
    Bitboard occ_after = (occupied ^ sq_bb(from)) | sq_bb(to);
    if (mt == MT_ENPASSANT) occ_after ^= sq_bb(to + (stm == WHITE ? -8 : 8));
    Bitboard direct = 0;
    switch (pt) {
        case PAWN:   direct = pawn_attacks_bb(stm, to); break;
        case KNIGHT: direct = knight_attacks_bb(to); break;
        case BISHOP: direct = bishop_attacks(to, occ_after); break;
        case ROOK:   direct = rook_attacks(to, occ_after); break;
        case QUEEN:  direct = queen_attacks(to, occ_after); break;
        default: break;
    }
    if (direct & sq_bb(ksq)) return true;

    // Discovered check: any of our sliders attack king through the new occupancy?
    Bitboard our_rooks_q   = pieces(stm, ROOK)   | pieces(stm, QUEEN);
    Bitboard our_bishops_q = pieces(stm, BISHOP) | pieces(stm, QUEEN);
    if (mt == MT_CASTLING) {
        // Rook moves to a known square; check after both king & rook moved.
        int rook_from, rook_to;
        if (file_of(to) == 6) { rook_from = stm == WHITE ? 7 : 63; rook_to = stm == WHITE ? 5 : 61; }
        else                  { rook_from = stm == WHITE ? 0 : 56; rook_to = stm == WHITE ? 3 : 59; }
        Bitboard occ2 = ((occupied ^ sq_bb(from) ^ sq_bb(rook_from)) | sq_bb(to)) | sq_bb(rook_to);
        if (rook_attacks(ksq, occ2) & sq_bb(rook_to)) return true;
    }
    if ((rook_attacks(ksq, occ_after) & our_rooks_q & ~sq_bb(to))) return true;
    if ((bishop_attacks(ksq, occ_after) & our_bishops_q & ~sq_bb(to))) return true;
    return false;
}

bool Board::is_capture(Move m) const {
    return board[to_sq(m)] != NO_PIECE || type_of_move(m) == MT_ENPASSANT;
}

bool Board::is_pseudo_legal(Move m) const {
    if (m == MOVE_NONE) return false;
    int from = from_sq(m), to = to_sq(m);
    if (from == to) return false;
    Piece p = board[from];
    if (p == NO_PIECE || color_of(p) != stm) return false;
    PieceType pt = type_of_piece(p);
    Piece target = board[to];
    if (target != NO_PIECE && color_of(target) == stm) return false;

    MoveType mt = type_of_move(m);

    if (mt == MT_CASTLING) {
        // Rare; verify by full pseudo-legal generation.
        if (pt != KING) return false;
        MoveList ml;
        generate_pseudo(*this, ml);
        for (int i = 0; i < ml.size; i++)
            if (ml.moves[i].move == m) return true;
        return false;
    }
    if (mt == MT_ENPASSANT) {
        if (pt != PAWN) return false;
        if (st->ep_square != to) return false;
        if (target != NO_PIECE) return false;
        return (pawn_attacks_bb(stm, from) & sq_bb(to)) != 0;
    }
    if (mt == MT_PROMOTION) {
        if (pt != PAWN) return false;
        int rank_to = rank_of(to), rank_from = rank_of(from);
        int up = stm == WHITE ? 8 : -8;
        if (stm == WHITE && (rank_to != 7 || rank_from != 6)) return false;
        if (stm == BLACK && (rank_to != 0 || rank_from != 1)) return false;
        if (to == from + up) return target == NO_PIECE;
        return target != NO_PIECE && (pawn_attacks_bb(stm, from) & sq_bb(to)) != 0;
    }

    // Normal move.
    switch (pt) {
        case PAWN: {
            int up = stm == WHITE ? 8 : -8;
            int rank_from = rank_of(from), rank_to = rank_of(to);
            // Promotions must use MT_PROMOTION; reject if pawn lands on last rank.
            if (stm == WHITE && rank_to == 7) return false;
            if (stm == BLACK && rank_to == 0) return false;
            if (to == from + up) return target == NO_PIECE;
            if ((stm == WHITE && rank_from == 1 && to == from + 16
                 && board[from + 8] == NO_PIECE && target == NO_PIECE)
             || (stm == BLACK && rank_from == 6 && to == from - 16
                 && board[from - 8] == NO_PIECE && target == NO_PIECE))
                return true;
            if (target != NO_PIECE && color_of(target) != stm
                && (pawn_attacks_bb(stm, from) & sq_bb(to)))
                return true;
            return false;
        }
        case KNIGHT: return (knight_attacks_bb(from)            & sq_bb(to)) != 0;
        case BISHOP: return (bishop_attacks(from, occupied)     & sq_bb(to)) != 0;
        case ROOK:   return (rook_attacks(from, occupied)       & sq_bb(to)) != 0;
        case QUEEN:  return (queen_attacks(from, occupied)      & sq_bb(to)) != 0;
        case KING:   return (king_attacks_bb(from)              & sq_bb(to)) != 0;
        default: return false;
    }
}
bool Board::is_capture_or_promotion(Move m) const {
    return is_capture(m) || type_of_move(m) == MT_PROMOTION;
}

// ---------------------------------------------------------------------------
// Make / unmake
// ---------------------------------------------------------------------------
void Board::make_move(Move m) {
    int from = from_sq(m), to = to_sq(m);
    MoveType mt = type_of_move(m);

    StateInfo* prev = st;
    ++st;
    *st = *prev;
    st->previous = prev;
    st->captured = NO_PIECE_TYPE;
    st->halfmove_clock++;

    // Clear ep from key (re-add only if a new double push happens this move).
    if (st->ep_square != SQ_NONE) {
        st->key ^= Zobrist::ep_file[file_of(st->ep_square)];
        st->ep_square = SQ_NONE;
    }

    Piece moved = board[from];
    PieceType moved_pt = type_of_piece(moved);

    if (mt == MT_CASTLING) {
        int rook_from, rook_to;
        if (file_of(to) == 6) { rook_from = stm == WHITE ? 7 : 63; rook_to = stm == WHITE ? 5 : 61; }
        else                  { rook_from = stm == WHITE ? 0 : 56; rook_to = stm == WHITE ? 3 : 59; }
        move_piece(from, to);
        move_piece(rook_from, rook_to);
    } else if (mt == MT_ENPASSANT) {
        int cap_sq = to + (stm == WHITE ? -8 : 8);
        st->captured = PAWN;
        remove_piece(cap_sq);
        move_piece(from, to);
        st->halfmove_clock = 0;
    } else if (mt == MT_PROMOTION) {
        if (board[to] != NO_PIECE) {
            st->captured = type_of_piece(board[to]);
            remove_piece(to);
        }
        remove_piece(from);
        put_piece(make_piece(stm, promotion_of(m)), to);
        st->halfmove_clock = 0;
    } else {
        if (board[to] != NO_PIECE) {
            st->captured = type_of_piece(board[to]);
            remove_piece(to);
        }
        move_piece(from, to);
        if (moved_pt == PAWN) {
            st->halfmove_clock = 0;
            if (std::abs(to - from) == 16) {
                int ep_sq = (from + to) / 2;
                st->ep_square = ep_sq;
                st->key ^= Zobrist::ep_file[file_of(ep_sq)];
            }
        } else if (st->captured != NO_PIECE_TYPE) {
            st->halfmove_clock = 0;
        }
    }

    int new_rights = st->castling_rights & CASTLING_MASK[from] & CASTLING_MASK[to];
    if (new_rights != st->castling_rights) {
        st->key ^= Zobrist::castling[st->castling_rights];
        st->key ^= Zobrist::castling[new_rights];
        st->castling_rights = new_rights;
    }

    st->key ^= Zobrist::side;
    stm = ~stm;
    history_keys[++game_ply] = st->key;
    (void)moved_pt;
}

void Board::unmake_move(Move m) {
    stm = ~stm;
    --game_ply;
    int from = from_sq(m), to = to_sq(m);
    MoveType mt = type_of_move(m);

    if (mt == MT_PROMOTION) {
        Piece promoted_p = board[to];
        Color c = color_of(promoted_p);
        Bitboard tobb = sq_bb(to), frombb = sq_bb(from);
        // Remove promoted piece
        byType[type_of_piece(promoted_p)] ^= tobb;
        byColor[c] ^= tobb;
        occupied ^= tobb;
        board[to] = NO_PIECE;
        // Restore pawn at `from`
        byType[PAWN] ^= frombb;
        byColor[c] ^= frombb;
        occupied ^= frombb;
        board[from] = make_piece(c, PAWN);
        // Restore captured, if any
        if (st->captured != NO_PIECE_TYPE) {
            byType[st->captured] ^= tobb;
            byColor[~c] ^= tobb;
            occupied ^= tobb;
            board[to] = make_piece(~c, PieceType(st->captured));
        }
    } else if (mt == MT_ENPASSANT) {
        int cap_sq = to + (stm == WHITE ? -8 : 8);
        Color c = stm;
        Bitboard fromto = sq_bb(from) | sq_bb(to);
        byType[PAWN] ^= fromto;
        byColor[c] ^= fromto;
        occupied ^= fromto;
        board[from] = make_piece(c, PAWN);
        board[to] = NO_PIECE;
        Bitboard cbb = sq_bb(cap_sq);
        byType[PAWN] ^= cbb;
        byColor[~c] ^= cbb;
        occupied ^= cbb;
        board[cap_sq] = make_piece(~c, PAWN);
    } else if (mt == MT_CASTLING) {
        int rook_from, rook_to;
        if (file_of(to) == 6) { rook_from = stm == WHITE ? 7 : 63; rook_to = stm == WHITE ? 5 : 61; }
        else                  { rook_from = stm == WHITE ? 0 : 56; rook_to = stm == WHITE ? 3 : 59; }
        // King back to->from
        Bitboard fromto = sq_bb(from) | sq_bb(to);
        byType[KING] ^= fromto;
        byColor[stm] ^= fromto;
        occupied ^= fromto;
        board[from] = make_piece(stm, KING);
        board[to] = NO_PIECE;
        king_square[stm] = from;
        // Rook rook_to->rook_from
        Bitboard rfromto = sq_bb(rook_from) | sq_bb(rook_to);
        byType[ROOK] ^= rfromto;
        byColor[stm] ^= rfromto;
        occupied ^= rfromto;
        board[rook_from] = make_piece(stm, ROOK);
        board[rook_to] = NO_PIECE;
    } else {
        Piece p = board[to];
        Color c = color_of(p);
        Bitboard fromto = sq_bb(from) | sq_bb(to);
        byType[type_of_piece(p)] ^= fromto;
        byColor[c] ^= fromto;
        occupied ^= fromto;
        board[from] = p;
        board[to] = NO_PIECE;
        if (type_of_piece(p) == KING) king_square[c] = from;
        if (st->captured != NO_PIECE_TYPE) {
            Bitboard tobb = sq_bb(to);
            byType[st->captured] ^= tobb;
            byColor[~c] ^= tobb;
            occupied ^= tobb;
            board[to] = make_piece(~c, PieceType(st->captured));
        }
    }
    --st;
}

void Board::make_null_move() {
    StateInfo* prev = st;
    ++st;
    *st = *prev;
    st->previous = prev;
    st->captured = NO_PIECE_TYPE;
    st->halfmove_clock++;
    if (st->ep_square != SQ_NONE) {
        st->key ^= Zobrist::ep_file[file_of(st->ep_square)];
        st->ep_square = SQ_NONE;
    }
    st->key ^= Zobrist::side;
    stm = ~stm;
    history_keys[++game_ply] = st->key;
}

void Board::unmake_null_move() {
    stm = ~stm;
    --game_ply;
    --st;
}

// ---------------------------------------------------------------------------
// Draws
// ---------------------------------------------------------------------------
bool Board::is_repetition() const {
    // Need to see if current key has appeared at least once before within the
    // halfmove window (since a pawn move or capture clears repetition history).
    int end = game_ply;
    int start = std::max(0, end - st->halfmove_clock);
    for (int i = end - 4; i >= start; i -= 2) {
        if (history_keys[i] == st->key) return true;
    }
    return false;
}

bool Board::is_insufficient_material() const {
    // Insufficient: lone kings, K+B vs K, K+N vs K, K+B vs K+B (same color squares).
    if (byType[PAWN] || byType[ROOK] || byType[QUEEN]) return false;
    int wb = popcount(pieces(WHITE, BISHOP));
    int wn = popcount(pieces(WHITE, KNIGHT));
    int bb = popcount(pieces(BLACK, BISHOP));
    int bn = popcount(pieces(BLACK, KNIGHT));
    int total_minor = wb + wn + bb + bn;
    if (total_minor == 0) return true;       // K vs K
    if (total_minor == 1) return true;       // K+minor vs K
    return false;                            // be conservative; > 1 minor: not insufficient
}

int Board::non_pawn_material(Color c) const {
    int m = 0;
    m += popcount(pieces(c, KNIGHT)) * PieceValueSimple[KNIGHT];
    m += popcount(pieces(c, BISHOP)) * PieceValueSimple[BISHOP];
    m += popcount(pieces(c, ROOK))   * PieceValueSimple[ROOK];
    m += popcount(pieces(c, QUEEN))  * PieceValueSimple[QUEEN];
    return m;
}

// ---------------------------------------------------------------------------
// SEE — Stockfish-style "swap" using attackers_to (Pradyumna Kannan style).
// ---------------------------------------------------------------------------
bool Board::see_ge(Move m, int threshold) const {
    MoveType mt = type_of_move(m);
    // Accept non-NORMAL moves (promotions, castling, ep) without SEE filtering.
    if (mt != MT_NORMAL) return threshold <= 0;

    int from = from_sq(m), to = to_sq(m);
    int swap = PieceValueSimple[type_of_piece(board[to])] - threshold;
    if (swap < 0) return false;
    swap = PieceValueSimple[type_of_piece(board[from])] - swap;
    if (swap <= 0) return true;

    Bitboard occ = (occupied ^ sq_bb(from)) | sq_bb(to);
    Bitboard attackers = attackers_to(to, occ) & occ;
    Color side = ~stm;
    int result = 1;

    while (true) {
        Bitboard our_attackers = attackers & byColor[side];
        if (!our_attackers) break;

        // Stop if our last attacker would move our own king into check.
        // (Approximation: skip; strong engines handle this rigorously.)

        int attacker_pt;
        Bitboard bb = 0;
        for (attacker_pt = PAWN; attacker_pt <= KING; ++attacker_pt) {
            bb = our_attackers & byType[attacker_pt];
            if (bb) break;
        }
        if (attacker_pt > KING) break;

        occ ^= bb & -bb;  // remove the chosen LVA from occupancy

        // X-ray: opening up sliders behind the captured piece.
        if (attacker_pt == PAWN || attacker_pt == BISHOP || attacker_pt == QUEEN)
            attackers |= bishop_attacks(to, occ) & (byType[BISHOP] | byType[QUEEN]);
        if (attacker_pt == ROOK || attacker_pt == QUEEN)
            attackers |= rook_attacks(to, occ) & (byType[ROOK] | byType[QUEEN]);
        attackers &= occ;

        result ^= 1;
        swap = PieceValueSimple[attacker_pt] - swap;
        if (swap < result) break;
        side = ~side;
    }
    return bool(result);
}

// ---------------------------------------------------------------------------
// UCI move helpers
// ---------------------------------------------------------------------------
std::string move_to_uci(Move m) {
    if (m == MOVE_NONE) return "0000";
    int from = from_sq(m), to = to_sq(m);
    std::string s;
    s += char('a' + file_of(from));
    s += char('1' + rank_of(from));
    s += char('a' + file_of(to));
    s += char('1' + rank_of(to));
    if (type_of_move(m) == MT_PROMOTION) {
        const char promo_chars[] = { 'n','b','r','q' };
        s += promo_chars[(int(m) >> 12) & 3];
    }
    return s;
}

Move uci_to_move(const std::string& s, Board& b) {
    if (s.size() < 4) return MOVE_NONE;
    int from = (s[0] - 'a') + 8 * (s[1] - '1');
    int to   = (s[2] - 'a') + 8 * (s[3] - '1');

    MoveList ml;
    generate_legal(b, ml);
    for (int i = 0; i < ml.size; ++i) {
        Move m = ml.moves[i].move;
        if (from_sq(m) == from && to_sq(m) == to) {
            if (s.size() == 5 && type_of_move(m) == MT_PROMOTION) {
                char p = s[4];
                PieceType want = (p == 'n' ? KNIGHT : p == 'b' ? BISHOP :
                                  p == 'r' ? ROOK   : QUEEN);
                if (promotion_of(m) != want) continue;
            }
            return m;
        }
    }
    return MOVE_NONE;
}

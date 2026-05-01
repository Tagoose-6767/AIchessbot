// types.h — fundamental engine types: enums, Move encoding, constants.
#pragma once

#include <cstdint>
#include <string>

using U64 = uint64_t;
using Bitboard = uint64_t;

// --- Color & PieceType --------------------------------------------------------
enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
inline Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int {
    NO_PIECE_TYPE = 0,
    PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6,
    PIECE_TYPE_NB = 7
};

enum Piece : int {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_NB = 16
};

inline Color color_of(Piece p) { return Color(p >> 3); }
inline PieceType type_of_piece(Piece p) { return PieceType(p & 7); }
inline Piece make_piece(Color c, PieceType pt) { return Piece((int(c) << 3) | int(pt)); }

// --- Squares -----------------------------------------------------------------
// 0 = a1, 1 = b1, ..., 7 = h1, 8 = a2, ..., 63 = h8.
inline int file_of(int sq) { return sq & 7; }
inline int rank_of(int sq) { return sq >> 3; }
inline int make_square(int file, int rank) { return (rank << 3) | file; }
inline int relative_rank(Color c, int r) { return c == WHITE ? r : 7 - r; }
inline int relative_sq(Color c, int sq) { return c == WHITE ? sq : sq ^ 56; }
constexpr int SQ_NONE = 64;

// --- Move encoding (16 bits) -------------------------------------------------
//   bits  0- 5 : from square
//   bits  6-11 : to square
//   bits 12-13 : promotion piece minus KNIGHT (0=N, 1=B, 2=R, 3=Q)
//   bits 14-15 : move type (NORMAL / PROMOTION / EN_PASSANT / CASTLING)
using Move = uint16_t;
constexpr Move MOVE_NONE = 0;
constexpr Move MOVE_NULL = 65;  // a1->b1 with type=0; reserved sentinel

enum MoveType : uint16_t {
    MT_NORMAL    = 0u << 14,
    MT_PROMOTION = 1u << 14,
    MT_ENPASSANT = 2u << 14,
    MT_CASTLING  = 3u << 14,
    MT_MASK      = 3u << 14
};

inline Move make_normal(int from, int to)        { return Move(from | (to << 6)); }
inline Move make_ep(int from, int to)            { return Move(from | (to << 6) | MT_ENPASSANT); }
inline Move make_castle(int from, int to)        { return Move(from | (to << 6) | MT_CASTLING); }
inline Move make_promo(int from, int to, PieceType pt) {
    return Move(from | (to << 6) | (((int(pt) - int(KNIGHT)) & 3) << 12) | MT_PROMOTION);
}

inline int from_sq(Move m)            { return int(m) & 0x3F; }
inline int to_sq(Move m)              { return (int(m) >> 6) & 0x3F; }
inline MoveType type_of_move(Move m)  { return MoveType(m & MT_MASK); }
inline PieceType promotion_of(Move m) { return PieceType(((int(m) >> 12) & 3) + int(KNIGHT)); }

std::string move_to_uci(Move m);
Move uci_to_move(const std::string& s, struct Board& b);

// --- Castling ----------------------------------------------------------------
enum CastlingRights : int {
    NO_CASTLING = 0,
    WHITE_OO  = 1,
    WHITE_OOO = 2,
    BLACK_OO  = 4,
    BLACK_OOO = 8,
    WHITE_CASTLING = WHITE_OO | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING
};

// --- Search constants --------------------------------------------------------
constexpr int MAX_PLY = 128;
constexpr int MAX_MOVES = 256;
constexpr int VALUE_INFINITE = 30000;
constexpr int VALUE_MATE = 29000;
constexpr int VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;
constexpr int VALUE_NONE = 32001;
constexpr int VALUE_DRAW = 0;

// Simple piece values used for SEE / MVV-LVA / delta pruning.
constexpr int PieceValueSimple[PIECE_TYPE_NB] = { 0, 100, 320, 330, 500, 900, 20000 };

// --- Bit helpers -------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
inline int popcount(U64 b) { return __builtin_popcountll(b); }
inline int lsb(U64 b)      { return __builtin_ctzll(b); }
inline int msb(U64 b)      { return 63 - __builtin_clzll(b); }
#elif defined(_MSC_VER)
#include <intrin.h>
inline int popcount(U64 b) { return int(__popcnt64(b)); }
inline int lsb(U64 b)      { unsigned long i; _BitScanForward64(&i, b); return int(i); }
inline int msb(U64 b)      { unsigned long i; _BitScanReverse64(&i, b); return int(i); }
#else
inline int popcount(U64 b) { int n=0; while(b){b&=b-1;++n;} return n; }
inline int lsb(U64 b)      { for (int i = 0; i < 64; i++) if (b & (1ULL<<i)) return i; return 64; }
inline int msb(U64 b)      { for (int i = 63; i >= 0; --i) if (b & (1ULL<<i)) return i; return -1; }
#endif

inline int pop_lsb(U64& b) { int s = lsb(b); b &= b - 1; return s; }
inline U64 sq_bb(int sq)   { return 1ULL << sq; }

// File / rank bitboards.
constexpr U64 FILE_A_BB = 0x0101010101010101ULL;
constexpr U64 FILE_B_BB = FILE_A_BB << 1;
constexpr U64 FILE_G_BB = FILE_A_BB << 6;
constexpr U64 FILE_H_BB = FILE_A_BB << 7;
constexpr U64 RANK_1_BB = 0xFFULL;
constexpr U64 RANK_2_BB = RANK_1_BB << 8;
constexpr U64 RANK_4_BB = RANK_1_BB << 24;
constexpr U64 RANK_5_BB = RANK_1_BB << 32;
constexpr U64 RANK_7_BB = RANK_1_BB << 48;
constexpr U64 RANK_8_BB = RANK_1_BB << 56;

inline U64 file_bb(int f) { return FILE_A_BB << f; }
inline U64 rank_bb(int r) { return RANK_1_BB << (r * 8); }

// board.h — bitboard-based position representation with make/unmake.
#pragma once

#include "types.h"
#include <string>

struct StateInfo {
    int castling_rights;
    int ep_square;        // SQ_NONE if not available
    int halfmove_clock;
    int captured;         // PieceType captured by the move that produced this state
    U64 key;
    StateInfo* previous;
};

class Board {
public:
    Board();
    void set(const std::string& fen);
    std::string fen() const;
    void print() const;

    // --- Accessors ------------------------------------------------------------
    Bitboard pieces() const                          { return occupied; }
    Bitboard pieces(Color c) const                   { return byColor[c]; }
    Bitboard pieces(PieceType pt) const              { return byType[pt]; }
    Bitboard pieces(Color c, PieceType pt) const     { return byColor[c] & byType[pt]; }
    Bitboard pieces(PieceType p1, PieceType p2) const{ return byType[p1] | byType[p2]; }

    Piece piece_on(int sq) const         { return board[sq]; }
    PieceType type_on(int sq) const      { return type_of_piece(board[sq]); }
    Color side_to_move() const           { return stm; }
    int castling() const                 { return st->castling_rights; }
    int ep_sq() const                    { return st->ep_square; }
    int halfmove_clock() const           { return st->halfmove_clock; }
    U64 key() const                      { return st->key; }
    int king_sq(Color c) const           { return king_square[c]; }
    int game_ply_count() const           { return game_ply; }
    int non_pawn_material(Color c) const;

    // --- Move handling --------------------------------------------------------
    void make_move(Move m);
    void unmake_move(Move m);
    void make_null_move();
    void unmake_null_move();

    // --- Queries --------------------------------------------------------------
    bool is_check() const                { return checkers() != 0; }
    Bitboard checkers() const            { return attackers_to(king_square[stm], occupied) & byColor[~stm]; }
    bool gives_check(Move m) const;
    bool is_legal(Move m) const;
    bool is_capture(Move m) const;
    // Cheap static check: would generate_pseudo emit this move from this position?
    // Used to validate TT / killer / counter moves before make_move.
    bool is_pseudo_legal(Move m) const;
    bool is_capture_or_promotion(Move m) const;
    bool is_repetition() const;
    bool is_50move_draw() const          { return st->halfmove_clock >= 100; }
    bool is_insufficient_material() const;

    Bitboard attackers_to(int sq, Bitboard occ) const;

    // SEE: returns true if the static exchange evaluation of `m` is >= threshold.
    bool see_ge(Move m, int threshold = 0) const;

private:
    Bitboard byType[PIECE_TYPE_NB];
    Bitboard byColor[COLOR_NB];
    Bitboard occupied;
    Piece board[64];
    int king_square[COLOR_NB];
    Color stm;
    int game_ply;
    StateInfo* st;
    StateInfo state_stack[MAX_PLY + 32];

    // Repetition history: keys of every position reached this game.
    U64 history_keys[2048];

    void put_piece(Piece p, int sq);
    void remove_piece(int sq);
    void move_piece(int from, int to);
    U64 compute_key() const;
    void clear();
};

// Castling-rights update mask: ANDing this into castling_rights when a piece
// moves onto/off these squares clears the appropriate rights cheaply.
extern const int CASTLING_MASK[64];

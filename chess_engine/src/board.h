// board.h — bitboard-based position representation with make/unmake.
#pragma once

#include "types.h"
#include <memory>
#include <string>

struct StateInfo {
    int castling_rights;
    int ep_square;        // SQ_NONE if not available
    int halfmove_clock;
    int captured;         // PieceType captured by the move that produced this state
    U64 key;
    StateInfo* previous;

    // --- NNUE incremental accumulator -------------------------------------
    // One 256-dim int16 vector per perspective (white, black). Updated in
    // make_move from the previous StateInfo's accumulator; popped automatically
    // by unmake_move when it pops the state stack. The `computed` flags let
    // evaluate() lazily refresh from scratch when the chain is broken (e.g.,
    // freshly-set position, freshly-loaded net, or a king move that we mark
    // invalid for the moving side instead of refreshing eagerly).
    alignas(32) int16_t nnue_acc[2][256];
    bool    nnue_acc_computed[2];
};

class Board {
public:
    Board();
    Board(const Board& other);             // deep copy with state_stack pointer fixup (for Lazy SMP)
    Board& operator=(const Board& other);

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

    // Exposed so NNUE can read/refresh the per-position accumulator cache.
    StateInfo* state_info() const        { return st; }

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
    // Capacity sized for the longest game we'll ever see (2048 plies = 1024
    // full moves, well past any practical Cute Chess time control) plus the
    // search's own MAX_PLY excursion on top, plus a small safety margin.
    // state_stack is heap-allocated because each StateInfo carries a 1 KB
    // NNUE accumulator: an inline array of this size would be ~2.4 MB and
    // would blow Windows' default 1 MB thread stack for any local Board.
    static constexpr int MAX_GAME_PLIES = 2048;
    static constexpr int STATE_STACK_SIZE = MAX_GAME_PLIES + MAX_PLY + 32;

    Bitboard byType[PIECE_TYPE_NB];
    Bitboard byColor[COLOR_NB];
    Bitboard occupied;
    Piece board[64];
    int king_square[COLOR_NB];
    Color stm;
    int game_ply;
    StateInfo* st;
    std::unique_ptr<StateInfo[]> state_stack;

    // Repetition history: keys of every position reached this game. Sized to
    // match state_stack so a search at any in-game ply still has room for the
    // MAX_PLY recursion on top before history_keys[++game_ply] would overflow.
    std::unique_ptr<U64[]> history_keys;

    void put_piece(Piece p, int sq);
    void remove_piece(int sq);
    void move_piece(int from, int to);
    U64 compute_key() const;
    void clear();
};

// Castling-rights update mask: ANDing this into castling_rights when a piece
// moves onto/off these squares clears the appropriate rights cheaply.
extern const int CASTLING_MASK[64];

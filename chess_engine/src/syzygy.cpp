// syzygy.cpp — translates engine Board <-> Fathom calls.

#include "syzygy.h"
#include "board.h"
#include "movegen.h"

#include "fathom/tbprobe.h"

#include <atomic>
#include <iostream>
#include <string>

namespace Syzygy {

static std::atomic<bool> s_active{false};
static std::atomic<int>  s_largest{0};

bool init(const std::string& path) {
    // Re-init: free first.
    if (s_active.load()) {
        ::tb_free();
        s_active.store(false);
        s_largest.store(0);
    }
    if (path.empty() || path == "<empty>") return false;

    if (!::tb_init(path.c_str())) {
        std::cerr << "info string SyzygyPath: tb_init failed for '" << path << "'\n";
        return false;
    }
    int n = int(TB_LARGEST);
    s_largest.store(n);
    if (n == 0) {
        std::cerr << "info string SyzygyPath: no tablebase files found at '" << path << "'\n";
        return false;
    }
    s_active.store(true);
    std::cerr << "info string SyzygyPath: loaded, TB_LARGEST=" << n << '\n';
    return true;
}

void free() {
    if (s_active.load()) {
        ::tb_free();
        s_active.store(false);
        s_largest.store(0);
    }
}

int  largest() { return s_largest.load(); }
bool active()  { return s_active.load() && s_largest.load() > 0; }

// Castling-rights translation: engine uses {WHITE_OO=1,WHITE_OOO=2,BLACK_OO=4,BLACK_OOO=8}
// — same bit assignments as Fathom's TB_CASTLING_*. We can pass the value through.
// Fathom requires castling==0 for any probe, so non-zero rights => skip probing.

static inline unsigned fathom_castling(int cr) { return unsigned(cr); }

// Fathom uses 0..63 with A1=0, H8=63 — same as our encoding.
static void board_to_fathom(const Board& b,
                            uint64_t& white, uint64_t& black,
                            uint64_t& kings, uint64_t& queens, uint64_t& rooks,
                            uint64_t& bishops, uint64_t& knights, uint64_t& pawns) {
    white   = b.pieces(WHITE);
    black   = b.pieces(BLACK);
    kings   = b.pieces(KING);
    queens  = b.pieces(QUEEN);
    rooks   = b.pieces(ROOK);
    bishops = b.pieces(BISHOP);
    knights = b.pieces(KNIGHT);
    pawns   = b.pieces(PAWN);
}

bool probe_wdl(const Board& b, int& score) {
    if (!active()) return false;
    if (b.castling() != 0) return false;
    int total = popcount(b.pieces());
    if (total > s_largest.load()) return false;

    uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    board_to_fathom(b, white, black, kings, queens, rooks, bishops, knights, pawns);
    unsigned ep = b.ep_sq() == SQ_NONE ? 0 : unsigned(b.ep_sq());
    bool turn = b.side_to_move() == WHITE;

    unsigned res = ::tb_probe_wdl(white, black, kings, queens, rooks, bishops,
                                  knights, pawns,
                                  unsigned(b.halfmove_clock()),
                                  fathom_castling(b.castling()),
                                  ep, turn);
    if (res == TB_RESULT_FAILED) return false;

    // Map WDL to a coarse engine score from the side-to-move's POV.
    // VALUE_MATE_IN_MAX_PLY is the upper bound for tablebase wins so they
    // don't shadow real mates. Use slightly inside that band.
    constexpr int TB_WIN_SCORE = VALUE_MATE_IN_MAX_PLY - 100;
    switch (res) {
        case TB_LOSS:          score = -TB_WIN_SCORE; return true;
        case TB_BLESSED_LOSS:  score = -1;            return true;  // rule50-saved
        case TB_DRAW:          score = 0;             return true;
        case TB_CURSED_WIN:    score = 1;             return true;
        case TB_WIN:           score = TB_WIN_SCORE;  return true;
        default:               return false;
    }
}

Move fathom_move_to_engine(const Board& b, unsigned from, unsigned to, unsigned promotes) {
    // Generate all legal moves and find one with matching from/to/promo.
    MoveList ml;
    generate_legal(b, ml);
    PieceType wanted_promo = NO_PIECE_TYPE;
    switch (promotes) {
        case TB_PROMOTES_QUEEN:  wanted_promo = QUEEN;  break;
        case TB_PROMOTES_ROOK:   wanted_promo = ROOK;   break;
        case TB_PROMOTES_BISHOP: wanted_promo = BISHOP; break;
        case TB_PROMOTES_KNIGHT: wanted_promo = KNIGHT; break;
        default:                 wanted_promo = NO_PIECE_TYPE; break;
    }
    for (int i = 0; i < ml.size; i++) {
        Move m = ml.moves[i].move;
        if (unsigned(from_sq(m)) != from) continue;
        if (unsigned(to_sq(m))   != to)   continue;
        if (wanted_promo != NO_PIECE_TYPE) {
            if (type_of_move(m) != MT_PROMOTION)        continue;
            if (promotion_of(m) != wanted_promo)        continue;
        } else {
            if (type_of_move(m) == MT_PROMOTION)        continue;
        }
        return m;
    }
    return MOVE_NONE;
}

bool probe_root(const Board& b, int& score, Move& best) {
    if (!active()) return false;
    if (b.castling() != 0) return false;
    int total = popcount(b.pieces());
    if (total > s_largest.load()) return false;

    uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    board_to_fathom(b, white, black, kings, queens, rooks, bishops, knights, pawns);
    unsigned ep = b.ep_sq() == SQ_NONE ? 0 : unsigned(b.ep_sq());
    bool turn = b.side_to_move() == WHITE;

    unsigned res = ::tb_probe_root(white, black, kings, queens, rooks, bishops,
                                   knights, pawns,
                                   unsigned(b.halfmove_clock()),
                                   fathom_castling(b.castling()),
                                   ep, turn, nullptr);
    if (res == TB_RESULT_FAILED || res == TB_RESULT_CHECKMATE
        || res == TB_RESULT_STALEMATE) return false;

    unsigned wdl = TB_GET_WDL(res);
    unsigned from = TB_GET_FROM(res);
    unsigned to   = TB_GET_TO(res);
    unsigned pr   = TB_GET_PROMOTES(res);

    Move m = fathom_move_to_engine(b, from, to, pr);
    if (m == MOVE_NONE) return false;

    constexpr int TB_WIN_SCORE = VALUE_MATE_IN_MAX_PLY - 100;
    switch (wdl) {
        case TB_LOSS:          score = -TB_WIN_SCORE; break;
        case TB_BLESSED_LOSS:  score = -1;            break;
        case TB_DRAW:          score = 0;             break;
        case TB_CURSED_WIN:    score = 1;             break;
        case TB_WIN:           score = TB_WIN_SCORE;  break;
        default:               return false;
    }
    best = m;
    return true;
}

}  // namespace Syzygy

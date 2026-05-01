// book.cpp — Polyglot opening book (.bin) reader.
//
// File format (per http://hgm.nubati.net/book_format.html):
//   Each entry is 16 bytes, big-endian:
//     uint64 key      Zobrist hash of the position (Polyglot scheme — see
//                     polyglot_random.h)
//     uint16 move     packed move (to-file, to-rank, from-file, from-rank,
//                     promotion-piece)
//     uint16 weight   relative move weight (caller picks proportionally)
//     uint32 learn    self-tuning bookkeeping (we ignore)
//   Entries are sorted by key; ties allowed.
//
// We binary-search the file by offset (no full load — books may be hundreds
// of MB). Then we collect all entries sharing the target key and pick one
// proportional to weight.

#include "book.h"
#include "board.h"
#include "movegen.h"
#include "polyglot_random.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace {

struct PEntry {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;
};

std::ifstream g_file;
int64_t       g_num_entries = 0;
std::mt19937  g_rng{ std::random_device{}() };

uint64_t read_be64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}
uint16_t read_be16(const uint8_t* p) {
    return uint16_t((p[0] << 8) | p[1]);
}
uint32_t read_be32(const uint8_t* p) {
    return uint32_t((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
}

bool read_entry(int64_t idx, PEntry& out) {
    g_file.seekg(idx * 16);
    uint8_t buf[16];
    g_file.read(reinterpret_cast<char*>(buf), 16);
    if (!g_file) return false;
    out.key    = read_be64(buf);
    out.move   = read_be16(buf + 8);
    out.weight = read_be16(buf + 10);
    out.learn  = read_be32(buf + 12);
    return true;
}

// --- Polyglot Zobrist key ---------------------------------------------------
// Layout per spec:
//   piece-square: 64*polyglot_piece + sq          for sq in 0..63
//                 polyglot_piece order: bp=0, wp=1, bn=2, wn=3, bb=4, wb=5,
//                                       br=6, wr=7, bq=8, wq=9, bk=10, wk=11
//   castling:    768..771  (white OO, white OOO, black OO, black OOO)
//   en-passant:  772..779  (file a..h) — only XOR-ed if a real ep capture exists
//   turn:        780       (XOR-ed iff white to move)
uint64_t polyglot_key(const Board& b) {
    uint64_t key = 0;

    Bitboard occ = b.pieces();
    while (occ) {
        int sq = pop_lsb(occ);
        Piece p = b.piece_on(sq);
        int pt = int(type_of_piece(p)) - 1;        // 0..5
        int pp = pt * 2 + (color_of(p) == WHITE ? 1 : 0);  // 0..11
        key ^= POLYGLOT_RANDOM64[64 * pp + sq];
    }

    int rights = b.castling();
    if (rights & WHITE_OO)  key ^= POLYGLOT_RANDOM64[768];
    if (rights & WHITE_OOO) key ^= POLYGLOT_RANDOM64[769];
    if (rights & BLACK_OO)  key ^= POLYGLOT_RANDOM64[770];
    if (rights & BLACK_OOO) key ^= POLYGLOT_RANDOM64[771];

    int ep = b.ep_sq();
    if (ep != SQ_NONE) {
        // Polyglot only includes the ep file if our side can actually capture.
        Color us = b.side_to_move();
        int ep_file = file_of(ep);
        Bitboard our_pawns = b.pieces(us, PAWN);
        // Capturing pawn must be on same rank as the just-moved enemy pawn.
        // White captures from rank 5 (index 4); black captures from rank 4 (3).
        int our_pawn_rank = (us == WHITE) ? 4 : 3;
        Bitboard adj = 0;
        if (ep_file > 0) adj |= file_bb(ep_file - 1);
        if (ep_file < 7) adj |= file_bb(ep_file + 1);
        if (our_pawns & adj & rank_bb(our_pawn_rank))
            key ^= POLYGLOT_RANDOM64[772 + ep_file];
    }

    if (b.side_to_move() == WHITE) key ^= POLYGLOT_RANDOM64[780];
    return key;
}

// --- Polyglot move (16 bits) -> internal Move ------------------------------
Move polyglot_to_internal(uint16_t pm, const Board& b) {
    int to_file   = pm & 7;
    int to_rank   = (pm >> 3) & 7;
    int from_file = (pm >> 6) & 7;
    int from_rank = (pm >> 9) & 7;
    int promo     = (pm >> 12) & 7;

    int from = make_square(from_file, from_rank);
    int to   = make_square(to_file, to_rank);

    // Polyglot encodes castling as king-rook (e1-h1, e1-a1, e8-h8, e8-a8).
    if (b.type_on(from) == KING) {
        if ((from == 4  && to == 7)  || (from == 60 && to == 63))
            return make_castle(from, from + 2);   // kingside
        if ((from == 4  && to == 0)  || (from == 60 && to == 56))
            return make_castle(from, from - 2);   // queenside
    }
    // EP: pawn moves diagonally onto an empty square.
    if (b.type_on(from) == PAWN && from_file != to_file && b.piece_on(to) == NO_PIECE) {
        return make_ep(from, to);
    }
    // Promotion.
    if (promo > 0) {
        PieceType pt = promo == 1 ? KNIGHT
                     : promo == 2 ? BISHOP
                     : promo == 3 ? ROOK
                     : QUEEN;
        return make_promo(from, to, pt);
    }
    return make_normal(from, to);
}

}  // namespace

// ---------------------------------------------------------------------------
bool OpeningBook::load(const std::string& path) {
    g_file.close();
    g_file.clear();
    g_file.open(path, std::ios::binary);
    if (!g_file) {
        std::cerr << "info string book: failed to open " << path << '\n';
        loaded_ = false;
        return false;
    }
    g_file.seekg(0, std::ios::end);
    int64_t sz = g_file.tellg();
    if (sz < 16 || (sz % 16) != 0) {
        std::cerr << "info string book: not a valid polyglot file (size=" << sz << ")\n";
        g_file.close();
        loaded_ = false;
        return false;
    }
    g_num_entries = sz / 16;
    std::cerr << "info string book: loaded " << path
              << " (" << g_num_entries << " entries)\n";
    loaded_ = true;
    return true;
}

void OpeningBook::close() {
    g_file.close();
    g_num_entries = 0;
    loaded_ = false;
}

Move OpeningBook::find_move(const Board& b) {
    if (!loaded_ || g_num_entries == 0) return MOVE_NONE;

    uint64_t target = polyglot_key(b);

    // Binary search for the first entry with key >= target.
    int64_t lo = 0, hi = g_num_entries;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        PEntry e;
        if (!read_entry(mid, e)) return MOVE_NONE;
        if (e.key < target) lo = mid + 1;
        else                hi = mid;
    }
    if (lo == g_num_entries) return MOVE_NONE;

    // Collect contiguous run of matching entries.
    std::vector<PEntry> matches;
    while (lo < g_num_entries) {
        PEntry e;
        if (!read_entry(lo, e) || e.key != target) break;
        matches.push_back(e);
        ++lo;
    }
    if (matches.empty()) return MOVE_NONE;

    // Weighted random pick. Reject entries whose decoded move isn't legal
    // (corrupt/suspicious book entries).
    uint64_t total = 0;
    for (auto& e : matches) total += e.weight;

    auto pick_index = [&]() {
        if (total == 0) return size_t(0);
        std::uniform_int_distribution<uint64_t> dist(0, total - 1);
        uint64_t r = dist(g_rng);
        uint64_t acc = 0;
        for (size_t i = 0; i < matches.size(); ++i) {
            acc += matches[i].weight;
            if (r < acc) return i;
        }
        return matches.size() - 1;
    };

    // Try up to N picks; if all picks decode to illegal moves, give up.
    for (int attempt = 0; attempt < 8; ++attempt) {
        size_t i = pick_index();
        Move m = polyglot_to_internal(matches[i].move, b);
        if (m != MOVE_NONE && b.is_pseudo_legal(m)) {
            // Verify legality (not leaving king in check) by simulation.
            Board nb = b;
            nb.make_move(m);
            Color mover = ~nb.side_to_move();
            bool legal = !(nb.attackers_to(nb.king_sq(mover), nb.pieces()) & nb.pieces(nb.side_to_move()));
            if (legal) return m;
        }
        // Drop this entry's weight to avoid picking it again next attempt.
        total -= matches[i].weight;
        matches[i].weight = 0;
        if (total == 0) break;
    }
    return MOVE_NONE;
}

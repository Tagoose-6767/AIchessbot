// nnue.cpp — NNUE-lite forward pass (non-incremental).
//
// Inference per position:
//   accumulator[h] = b_hidden[h]
//   for each piece P on square S:
//       feat = feature_index(P, S)         // 0..767
//       accumulator[h] += W_input[feat][h]
//   hidden[h] = max(0, accumulator[h])    // ReLU
//   output     = b_output + sum_h hidden[h] * W_output[h]
//   score_cp   = output / output_scale
//
// Cost: ~32 pieces × 256 mul-adds for input, plus 256 for output. ~10us
// per call on modern hardware — roughly doubles eval cost vs HCE.

#include "nnue.h"
#include "board.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr int INPUT_DIM  = 768;
constexpr int HIDDEN_DIM = 256;

bool loaded_ = false;
std::vector<int16_t> W_input_;     // [INPUT_DIM][HIDDEN_DIM] flattened (input-major for cache)
std::vector<int16_t> b_hidden_;    // [HIDDEN_DIM]
std::vector<int16_t> W_output_;    // [HIDDEN_DIM]
int32_t b_output_ = 0;
int16_t output_scale_ = 1;

// Feature index: 0..767. Layout — WP=0..63, WN=64..127, WB, WR, WQ, WK,
// then BP=384..447, BN, BB, BR, BQ, BK.
inline int feature_idx(Piece p, int sq) {
    int pt = type_of_piece(p) - 1;       // 0..5
    int co = color_of(p) == WHITE ? 0 : 6;
    return (pt + co) * 64 + sq;
}

}  // namespace

namespace NNUE {

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "info string nnue: failed to open " << path << '\n';
        return false;
    }
    char magic[8] = { 0 };
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "NNUELITE", 8) != 0) {
        std::cerr << "info string nnue: bad magic (expected NNUELITE)\n";
        return false;
    }
    uint32_t version = 0;
    uint16_t input_dim = 0, hidden_dim = 0, output_dim = 0;
    int16_t scale = 1;
    f.read(reinterpret_cast<char*>(&version),    4);
    f.read(reinterpret_cast<char*>(&input_dim),  2);
    f.read(reinterpret_cast<char*>(&hidden_dim), 2);
    f.read(reinterpret_cast<char*>(&output_dim), 2);
    f.read(reinterpret_cast<char*>(&scale),      2);

    if (version != 1 || input_dim != INPUT_DIM || hidden_dim != HIDDEN_DIM || output_dim != 1) {
        std::cerr << "info string nnue: incompatible header (v=" << version
                  << " in=" << input_dim << " h=" << hidden_dim
                  << " out=" << output_dim << ")\n";
        return false;
    }

    W_input_.assign(size_t(INPUT_DIM) * HIDDEN_DIM, 0);
    b_hidden_.assign(HIDDEN_DIM, 0);
    W_output_.assign(HIDDEN_DIM, 0);

    f.read(reinterpret_cast<char*>(W_input_.data()),  W_input_.size()  * sizeof(int16_t));
    f.read(reinterpret_cast<char*>(b_hidden_.data()), b_hidden_.size() * sizeof(int16_t));
    f.read(reinterpret_cast<char*>(W_output_.data()), W_output_.size() * sizeof(int16_t));
    f.read(reinterpret_cast<char*>(&b_output_), 4);

    if (!f) {
        std::cerr << "info string nnue: file truncated\n";
        unload();
        return false;
    }
    output_scale_ = scale > 0 ? scale : 1;
    loaded_ = true;
    std::cerr << "info string nnue: loaded " << path
              << " (768->256->1, " << (W_input_.size() * sizeof(int16_t) / 1024) << " KB)\n";
    return true;
}

void unload() {
    loaded_ = false;
    W_input_.clear();
    b_hidden_.clear();
    W_output_.clear();
    b_output_ = 0;
    output_scale_ = 1;
}

bool is_loaded() { return loaded_; }

int evaluate(const Board& b) {
    int32_t acc[HIDDEN_DIM];
    for (int h = 0; h < HIDDEN_DIM; ++h) acc[h] = b_hidden_[h];

    // Sparse accumulator: sum the input columns for each piece on the board.
    // ~32 columns out of 768; ~32×256 mul-free additions.
    Bitboard occ = b.pieces();
    while (occ) {
        int sq = pop_lsb(occ);
        Piece p = b.piece_on(sq);
        int feat = feature_idx(p, sq);
        const int16_t* col = &W_input_[size_t(feat) * HIDDEN_DIM];
        for (int h = 0; h < HIDDEN_DIM; ++h) acc[h] += col[h];
    }

    int64_t out = b_output_;
    for (int h = 0; h < HIDDEN_DIM; ++h) {
        int32_t v = acc[h];
        if (v < 0) v = 0;             // ReLU
        out += int64_t(v) * W_output_[h];
    }
    int score = int(out / output_scale_);
    if (score >  30000) score =  30000;
    if (score < -30000) score = -30000;
    return b.side_to_move() == WHITE ? score : -score;
}

}  // namespace NNUE

// nnue.cpp — Stockfish-12-compatible HalfKP-256x2-32-32 NNUE inference.
//
// All arithmetic, layout, and quantization scheme are taken from
// Stockfish 12 source (GPLv3, 2020). Reference files:
//   src/nnue/nnue_common.h         (constants, version, kpp ordering)
//   src/nnue/nnue_feature_transformer.h  (FT load + Transform)
//   src/nnue/features/half_kp.cpp  (orient + MakeIndex)
//   src/nnue/layers/affine_transform.h   (Affine load + Propagate)
//   src/nnue/layers/clipped_relu.h (>> 6, clamp [0,127])
//   src/nnue/architectures/halfkp_256x2-32-32.h (network topology)

#include "nnue.h"
#include "board.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <immintrin.h>
#  define NNUE_X86 1
#else
#  define NNUE_X86 0
#endif

namespace {

// Runtime CPU dispatch. Detected once at startup; AVX2 paths are gated by it.
bool g_use_avx2 = false;

struct SimdInit {
    SimdInit() {
#if NNUE_X86 && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        g_use_avx2 = __builtin_cpu_supports("avx2") != 0;
#elif NNUE_X86 && defined(_MSC_VER)
        int info[4]; __cpuid(info, 0);
        if (info[0] >= 7) {
            __cpuidex(info, 7, 0);
            g_use_avx2 = (info[1] & (1 << 5)) != 0;
        }
#endif
    }
};
SimdInit g_simd_init;


constexpr uint32_t NN_VERSION   = 0x7AF32F16u;
constexpr int      HALF_DIMS    = 256;
constexpr int      FT_INPUTS    = 41024;          // 64 king squares × 641 piece-square slots
constexpr int      PS_END       = 641;            // pieces-without-kings stride per king square
constexpr int      FV_SCALE     = 16;             // final divisor on output
constexpr int      WEIGHT_SHIFT = 6;              // ClippedReLU >> 6 before clamp

// Layer dimensions (not padded — for HalfKP-256 all inputs are already
// multiples of kMaxSimdWidth=32, so no padding bytes appear in the file).
constexpr int L1_IN = 512, L1_OUT = 32;
constexpr int L2_IN =  32, L2_OUT = 32;
constexpr int LO_IN =  32, LO_OUT =  1;

// Stockfish's PieceSquare offsets.
constexpr int PS_NONE     = 0;
constexpr int PS_W_PAWN   = 1;
constexpr int PS_B_PAWN   = 1 * 64 + 1;       //  65
constexpr int PS_W_KNIGHT = 2 * 64 + 1;       // 129
constexpr int PS_B_KNIGHT = 3 * 64 + 1;       // 193
constexpr int PS_W_BISHOP = 4 * 64 + 1;       // 257
constexpr int PS_B_BISHOP = 5 * 64 + 1;       // 321
constexpr int PS_W_ROOK   = 6 * 64 + 1;       // 385
constexpr int PS_B_ROOK   = 7 * 64 + 1;       // 449
constexpr int PS_W_QUEEN  = 8 * 64 + 1;       // 513
constexpr int PS_B_QUEEN  = 9 * 64 + 1;       // 577
// Kings have entries in SF's table but HalfKP excludes them (used as context only).

// kpp_board_index[piece][perspective] — from SF12 evaluate_nnue.cpp.
// Indexed by our Piece enum (NO_PIECE=0; W_PAWN=1..W_KING=6; B_PAWN=9..B_KING=14).
constexpr uint32_t KPP_BOARD_INDEX[16][2] = {
    /*  0 NO_PIECE */ { PS_NONE,     PS_NONE     },
    /*  1 W_PAWN   */ { PS_W_PAWN,   PS_B_PAWN   },
    /*  2 W_KNIGHT */ { PS_W_KNIGHT, PS_B_KNIGHT },
    /*  3 W_BISHOP */ { PS_W_BISHOP, PS_B_BISHOP },
    /*  4 W_ROOK   */ { PS_W_ROOK,   PS_B_ROOK   },
    /*  5 W_QUEEN  */ { PS_W_QUEEN,  PS_B_QUEEN  },
    /*  6 W_KING   */ { PS_NONE,     PS_NONE     },   // excluded from HalfKP
    /*  7         */ { PS_NONE,     PS_NONE     },
    /*  8         */ { PS_NONE,     PS_NONE     },
    /*  9 B_PAWN   */ { PS_B_PAWN,   PS_W_PAWN   },
    /* 10 B_KNIGHT */ { PS_B_KNIGHT, PS_W_KNIGHT },
    /* 11 B_BISHOP */ { PS_B_BISHOP, PS_W_BISHOP },
    /* 12 B_ROOK   */ { PS_B_ROOK,   PS_W_ROOK   },
    /* 13 B_QUEEN  */ { PS_B_QUEEN,  PS_W_QUEEN  },
    /* 14 B_KING   */ { PS_NONE,     PS_NONE     },   // excluded from HalfKP
    /* 15         */ { PS_NONE,     PS_NONE     },
};

// HalfKP "Friend" orient: XOR by 63 for black perspective (180° rotation).
inline int orient(Color persp, int sq) {
    return sq ^ ((persp == BLACK) ? 63 : 0);
}

inline uint32_t feature_idx(Color persp, Piece pc, int sq, int ksq_oriented) {
    return uint32_t(orient(persp, sq))
         + KPP_BOARD_INDEX[pc][persp]
         + uint32_t(PS_END) * uint32_t(ksq_oriented);
}

// --- Stored weights ----------------------------------------------------------
bool                 loaded_       = false;
std::vector<int16_t> ft_biases_;     //                256
std::vector<int16_t> ft_weights_;    //  41024 × 256 = 10,502,144
std::vector<int32_t> l1_biases_;     //                 32
std::vector<int8_t>  l1_weights_;    //     32 × 512 =  16,384
std::vector<int32_t> l2_biases_;     //                 32
std::vector<int8_t>  l2_weights_;    //     32 ×  32 =   1,024
std::vector<int32_t> out_biases_;    //                  1
std::vector<int8_t>  out_weights_;   //      1 ×  32 =      32
std::string          arch_string_;

// --- Little-endian readers ---------------------------------------------------
template <typename T>
T read_le(std::istream& s) {
    uint8_t buf[sizeof(T)];
    s.read(reinterpret_cast<char*>(buf), sizeof(T));
    typename std::make_unsigned<T>::type v = 0;
    for (size_t i = 0; i < sizeof(T); ++i)
        v |= typename std::make_unsigned<T>::type(buf[i]) << (8 * i);
    T r;
    std::memcpy(&r, &v, sizeof(T));
    return r;
}

template <typename T>
bool read_array(std::istream& s, std::vector<T>& v, size_t n) {
    v.resize(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = read_le<T>(s);
        if (!s) return false;
    }
    return true;
}

}  // namespace

namespace NNUE {

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "info string nnue: cannot open " << path << '\n';
        return false;
    }

    // --- Header ---
    uint32_t version  = read_le<uint32_t>(f);
    uint32_t nn_hash  = read_le<uint32_t>(f);
    uint32_t arch_sz  = read_le<uint32_t>(f);
    if (!f || version != NN_VERSION) {
        std::cerr << "info string nnue: bad version 0x" << std::hex << version
                  << " (want 0x" << NN_VERSION << ")\n" << std::dec;
        return false;
    }
    arch_string_.resize(arch_sz);
    f.read(&arch_string_[0], arch_sz);
    if (!f) return false;

    // --- Feature transformer ---
    uint32_t ft_hash = read_le<uint32_t>(f);
    if (!read_array(f, ft_biases_,  HALF_DIMS))                 return false;
    if (!read_array(f, ft_weights_, size_t(FT_INPUTS) * HALF_DIMS)) return false;

    // --- Network ---
    uint32_t net_hash = read_le<uint32_t>(f);
    if (!read_array(f, l1_biases_,  L1_OUT))           return false;
    if (!read_array(f, l1_weights_, size_t(L1_OUT) * L1_IN))  return false;
    if (!read_array(f, l2_biases_,  L2_OUT))           return false;
    if (!read_array(f, l2_weights_, size_t(L2_OUT) * L2_IN))  return false;
    if (!read_array(f, out_biases_, LO_OUT))           return false;
    if (!read_array(f, out_weights_,size_t(LO_OUT) * LO_IN))  return false;

    // Should be at EOF now.
    f.peek();
    if (!f.eof()) {
        std::cerr << "info string nnue: trailing bytes after network — wrong size?\n";
    }

    loaded_ = true;
    std::cerr << "info string nnue: loaded " << path << '\n'
              << "info string nnue: arch=\"" << arch_string_ << "\"\n"
              << "info string nnue: header_hash=0x" << std::hex << nn_hash
              << " ft_hash=0x" << ft_hash
              << " net_hash=0x" << net_hash << std::dec << '\n';
    return true;
}

void unload() {
    loaded_ = false;
    ft_biases_.clear();   ft_biases_.shrink_to_fit();
    ft_weights_.clear();  ft_weights_.shrink_to_fit();
    l1_biases_.clear();   l1_weights_.clear();
    l2_biases_.clear();   l2_weights_.clear();
    out_biases_.clear();  out_weights_.clear();
    arch_string_.clear();
}

bool is_loaded() { return loaded_; }

// Refresh: build one perspective's accumulator from the position, ignoring any
// cached state. Used by evaluate() when the incremental chain is broken.
void refresh_perspective(const Board& b, Color persp, int16_t* acc) {
    std::memcpy(acc, ft_biases_.data(), HALF_DIMS * sizeof(int16_t));
    int ksq_oriented = orient(persp, b.king_sq(persp));
    Bitboard pieces = b.pieces() & ~b.pieces(KING);
    while (pieces) {
        int sq = pop_lsb(pieces);
        Piece pc = b.piece_on(sq);
        uint32_t feat = feature_idx(persp, pc, sq, ksq_oriented);
        const int16_t* col = &ft_weights_[size_t(feat) * HALF_DIMS];
        for (int h = 0; h < HALF_DIMS; ++h) acc[h] += col[h];
    }
}

// AVX2 helpers — gated by runtime g_use_avx2 detection. The target attribute
// allows these to compile even when the TU isn't built with -mavx2; the
// runtime check prevents calling them on non-AVX2 CPUs (which would SIGILL).
#if NNUE_X86

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
static void add_col_avx2(int16_t* acc, const int16_t* col) {
    // 256 int16 = 16 lanes × 16 int16 per __m256i.
    for (int i = 0; i < HALF_DIMS / 16; ++i) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i * 16));
        __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + i * 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i * 16),
                            _mm256_add_epi16(a, c));
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
static void sub_col_avx2(int16_t* acc, const int16_t* col) {
    for (int i = 0; i < HALF_DIMS / 16; ++i) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i * 16));
        __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col + i * 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i * 16),
                            _mm256_sub_epi16(a, c));
    }
}

// Affine layer: in_dim must be a multiple of 32. uint8 inputs × int8 weights.
// Pattern from Stockfish 12's AffineTransform::Propagate (AVX2 branch):
// pmaddubsw → pmaddwd → accumulate to int32, then horizontal sum.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
static void affine_avx2(const uint8_t* input, const int8_t* weights,
                        const int32_t* biases, int32_t* output,
                        int in_dim, int out_dim) {
    const __m256i ones = _mm256_set1_epi16(1);
    for (int i = 0; i < out_dim; ++i) {
        __m256i sum = _mm256_setzero_si256();
        const int8_t* row = weights + size_t(i) * in_dim;
        for (int j = 0; j < in_dim; j += 32) {
            __m256i in_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + j));
            __m256i w_v  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + j));
            // Saturating int8×uint8 multiply with adjacent-pair int16 add.
            __m256i prod = _mm256_maddubs_epi16(in_v, w_v);
            // Then int16×1 multiply with adjacent-pair int32 add to widen.
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(prod, ones));
        }
        // Horizontal reduce: 8 int32 → 1.
        sum = _mm256_hadd_epi32(sum, sum);
        sum = _mm256_hadd_epi32(sum, sum);
        int32_t total = _mm256_extract_epi32(sum, 0) + _mm256_extract_epi32(sum, 4);
        output[i] = biases[i] + total;
    }
}

#endif  // NNUE_X86

// Helper: add or remove a single piece's feature column from one perspective.
namespace {
inline void add_feature(int16_t* acc, Color persp, Piece pc, int sq,
                        int ksq_oriented, int sign) {
    if (type_of_piece(pc) == KING) return;            // kings excluded
    uint32_t feat = feature_idx(persp, pc, sq, ksq_oriented);
    const int16_t* col = &ft_weights_[size_t(feat) * HALF_DIMS];
#if NNUE_X86
    if (g_use_avx2) {
        if (sign > 0) add_col_avx2(acc, col);
        else          sub_col_avx2(acc, col);
        return;
    }
#endif
    if (sign > 0)
        for (int h = 0; h < HALF_DIMS; ++h) acc[h] += col[h];
    else
        for (int h = 0; h < HALF_DIMS; ++h) acc[h] -= col[h];
}
}  // namespace

// Incremental update on make_move. After the move has been applied to the
// board, deduce the piece changes from the move + the captured piece type,
// and apply them to a copy of prev's accumulator. King moves on a perspective
// invalidate that perspective (full refresh deferred to evaluate()).
void update_after_move(const Board& b,
                       const StateInfo* prev, StateInfo* curr,
                       Move m) {
    if (!loaded_) {
        curr->nnue_acc_computed[0] = curr->nnue_acc_computed[1] = false;
        return;
    }
    int from = from_sq(m), to = to_sq(m);
    MoveType mt = type_of_move(m);
    // After make_move, side-to-move is flipped. The mover is the opposite.
    Color mover = ~b.side_to_move();

    bool king_moved_mover = (mt == MT_CASTLING)
                         || (mt == MT_NORMAL && b.type_on(to) == KING);

    for (int p = 0; p < 2; ++p) {
        Color persp = Color(p);

        if (!prev->nnue_acc_computed[p]) {
            curr->nnue_acc_computed[p] = false;
            continue;
        }
        // For the perspective whose king moved, lazy-invalidate. evaluate()
        // will rebuild from scratch the next time we evaluate this position.
        if (king_moved_mover && persp == mover) {
            curr->nnue_acc_computed[p] = false;
            continue;
        }

        // Apply incremental delta from prev to curr.
        std::memcpy(curr->nnue_acc[p], prev->nnue_acc[p],
                    HALF_DIMS * sizeof(int16_t));
        int ksq_oriented = orient(persp, b.king_sq(persp));
        int16_t* acc = curr->nnue_acc[p];

        if (mt == MT_NORMAL) {
            Piece moved = b.piece_on(to);
            add_feature(acc, persp, moved, from, ksq_oriented, -1);
            add_feature(acc, persp, moved, to,   ksq_oriented, +1);
            if (curr->captured != NO_PIECE_TYPE) {
                Piece cap = make_piece(~mover, PieceType(curr->captured));
                add_feature(acc, persp, cap, to, ksq_oriented, -1);
            }
        } else if (mt == MT_ENPASSANT) {
            Piece pawn = make_piece(mover, PAWN);
            add_feature(acc, persp, pawn, from, ksq_oriented, -1);
            add_feature(acc, persp, pawn, to,   ksq_oriented, +1);
            int cap_sq = to + (mover == WHITE ? -8 : 8);
            Piece enemy = make_piece(~mover, PAWN);
            add_feature(acc, persp, enemy, cap_sq, ksq_oriented, -1);
        } else if (mt == MT_PROMOTION) {
            Piece pawn     = make_piece(mover, PAWN);
            Piece promoted = make_piece(mover, promotion_of(m));
            add_feature(acc, persp, pawn,     from, ksq_oriented, -1);
            add_feature(acc, persp, promoted, to,   ksq_oriented, +1);
            if (curr->captured != NO_PIECE_TYPE) {
                Piece cap = make_piece(~mover, PieceType(curr->captured));
                add_feature(acc, persp, cap, to, ksq_oriented, -1);
            }
        } else if (mt == MT_CASTLING) {
            // Mover's king moved → handled via king_moved_mover above (this
            // branch only runs for the non-mover perspective). The rook moved
            // too; reflect just the rook for the non-mover perspective.
            int rook_from, rook_to;
            if (file_of(to) == 6) {
                rook_from = mover == WHITE ? 7 : 63;
                rook_to   = mover == WHITE ? 5 : 61;
            } else {
                rook_from = mover == WHITE ? 0 : 56;
                rook_to   = mover == WHITE ? 3 : 59;
            }
            Piece rook = make_piece(mover, ROOK);
            add_feature(acc, persp, rook, rook_from, ksq_oriented, -1);
            add_feature(acc, persp, rook, rook_to,   ksq_oriented, +1);
        }

        curr->nnue_acc_computed[p] = true;
    }
}

// --- Forward pass ------------------------------------------------------------
int evaluate(const Board& b) {
    if (!loaded_) return 0;

    // 1. Get / refresh the accumulator. The board's current StateInfo holds
    // the cached state; if either perspective is stale, rebuild it here and
    // mark valid so subsequent evaluations on the same node are O(1).
    StateInfo* st_mut = const_cast<StateInfo*>(b.state_info());
    int16_t (*acc)[HALF_DIMS] = st_mut->nnue_acc;
    for (int p = 0; p < 2; ++p) {
        if (!st_mut->nnue_acc_computed[p]) {
            refresh_perspective(b, Color(p), acc[p]);
            st_mut->nnue_acc_computed[p] = true;
        }
    }

    // 2. Transform: clamp [0, 127], concatenate STM perspective first.
    Color stm = b.side_to_move();
    Color them = ~stm;
    uint8_t input[L1_IN];  // 512
    for (int h = 0; h < HALF_DIMS; ++h) {
        int v = acc[stm][h];
        input[h] = uint8_t(v < 0 ? 0 : v > 127 ? 127 : v);
    }
    for (int h = 0; h < HALF_DIMS; ++h) {
        int v = acc[them][h];
        input[HALF_DIMS + h] = uint8_t(v < 0 ? 0 : v > 127 ? 127 : v);
    }

    // 3. Affine 512 → 32 (int8 weights, int32 bias/output).
    int32_t l1[L1_OUT];
#if NNUE_X86
    if (g_use_avx2) {
        affine_avx2(input, l1_weights_.data(), l1_biases_.data(), l1, L1_IN, L1_OUT);
    } else
#endif
    {
        for (int i = 0; i < L1_OUT; ++i) {
            int32_t sum = l1_biases_[i];
            const int8_t* row = &l1_weights_[size_t(i) * L1_IN];
            for (int j = 0; j < L1_IN; ++j) sum += int32_t(row[j]) * int32_t(input[j]);
            l1[i] = sum;
        }
    }
    // 4. ClippedReLU: >> 6, clamp [0, 127].
    uint8_t l1r[L1_OUT];
    for (int i = 0; i < L1_OUT; ++i) {
        int v = l1[i] >> WEIGHT_SHIFT;
        l1r[i] = uint8_t(v < 0 ? 0 : v > 127 ? 127 : v);
    }

    // 5. Affine 32 → 32.
    int32_t l2[L2_OUT];
#if NNUE_X86
    if (g_use_avx2) {
        affine_avx2(l1r, l2_weights_.data(), l2_biases_.data(), l2, L2_IN, L2_OUT);
    } else
#endif
    {
        for (int i = 0; i < L2_OUT; ++i) {
            int32_t sum = l2_biases_[i];
            const int8_t* row = &l2_weights_[size_t(i) * L2_IN];
            for (int j = 0; j < L2_IN; ++j) sum += int32_t(row[j]) * int32_t(l1r[j]);
            l2[i] = sum;
        }
    }
    uint8_t l2r[L2_OUT];
    for (int i = 0; i < L2_OUT; ++i) {
        int v = l2[i] >> WEIGHT_SHIFT;
        l2r[i] = uint8_t(v < 0 ? 0 : v > 127 ? 127 : v);
    }

    // 6. Output 32 → 1 (no ReLU on the final layer). Output layer is small
    // enough that scalar wins (single output, 32 muls). Skip SIMD here.
    int32_t out = out_biases_[0];
    for (int j = 0; j < LO_IN; ++j)
        out += int32_t(out_weights_[j]) * int32_t(l2r[j]);

    int score = out / FV_SCALE;
    // SF score is from STM's perspective (since STM accumulator is concatenated
    // first into the input). Our engine convention matches.
    if (score >  29000) score =  29000;
    if (score < -29000) score = -29000;
    return score;
}

}  // namespace NNUE

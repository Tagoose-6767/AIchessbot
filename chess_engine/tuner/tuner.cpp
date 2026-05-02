// tuner.cpp — Texel tuner for AIChessbot HCE weights.
// Builds against the engine's Board / movegen / evaluate modules and tunes
// `g_eval` to minimize sigmoid(K*eval) - label MSE on a labeled EPD dataset.
//
//   Build: see Makefile target `tuner`
//   Run:   ./tuner.exe --epd quiet-labeled.epd --iters 10000 --sample 200000
//
// Output: rewrites the literal block in src/evaluate.cpp between the markers
//         `// === TEXEL-TUNED-WEIGHTS-BEGIN ===` and
//         `// === TEXEL-TUNED-WEIGHTS-END ===`.

#include "../src/board.h"
#include "../src/movegen.h"
#include "../src/evaluate.h"
#include "../src/nnue.h"
#include "../src/types.h"
#include "../src/tt.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Param table — every tunable slot in EvalWeights, in stable order.
// Each row maps a flat index -> {pointer to int slot, kind, presentation}.
// ---------------------------------------------------------------------------
enum ParamKind : uint8_t { K_MG = 0, K_EG = 1, K_ALWAYS = 2 };

struct ParamMeta {
    int*        slot;        // -> g_eval member
    ParamKind   kind;
    std::string name;        // for debug only
};

struct ParamTable {
    std::vector<ParamMeta> p;
    int idx_mg_val(int pt) const         { return pt; }                       // 0..6
    int idx_eg_val(int pt) const         { return 7 + pt; }                   // 7..13
    int idx_pst_mg(int pt, int sq) const { return 14 + (pt - 1) * 64 + sq; }  // 14..397 (pt 1..6)
    int idx_pst_eg(int pt, int sq) const { return 398 + (pt - 1) * 64 + sq; } // 398..781
    int idx_passed_mg(int r) const       { return 782 + r; }                  // 782..789
    int idx_passed_eg(int r) const       { return 790 + r; }                  // 790..797
    int IDX_CONN_MG, IDX_CONN_EG;
    int IDX_DOUBLED_MG, IDX_DOUBLED_EG;
    int IDX_ISOLATED_MG, IDX_ISOLATED_EG;
    int IDX_ROOK_OPEN_MG, IDX_ROOK_OPEN_EG;
    int IDX_ROOK_SEMI_MG, IDX_ROOK_SEMI_EG;
    int IDX_KS_CLOSE, IDX_KS_FAR, IDX_KS_MISSING, IDX_KS_BEHIND;
    int IDX_MOB_MG, IDX_MOB_EG;
    int IDX_BPAIR_MG, IDX_BPAIR_EG;
    int IDX_TEMPO;
};

static ParamTable g_params;

static void build_params() {
    g_params.p.clear();
    auto add = [&](int* slot, ParamKind k, const std::string& nm) {
        ParamMeta m{ slot, k, nm };
        g_params.p.push_back(m);
        return int(g_params.p.size() - 1);
    };

    // 0..6: mg_val
    for (int pt = 0; pt < 7; pt++) add(&g_eval.mg_val[pt], K_MG, "mg_val[" + std::to_string(pt) + "]");
    // 7..13: eg_val
    for (int pt = 0; pt < 7; pt++) add(&g_eval.eg_val[pt], K_EG, "eg_val[" + std::to_string(pt) + "]");
    // 14..397: pst_mg[pt][sq] for pt in 1..6
    for (int pt = 1; pt <= 6; pt++)
        for (int sq = 0; sq < 64; sq++)
            add(&g_eval.pst_mg[pt][sq], K_MG, "pst_mg[" + std::to_string(pt) + "][" + std::to_string(sq) + "]");
    // 398..781: pst_eg
    for (int pt = 1; pt <= 6; pt++)
        for (int sq = 0; sq < 64; sq++)
            add(&g_eval.pst_eg[pt][sq], K_EG, "pst_eg[" + std::to_string(pt) + "][" + std::to_string(sq) + "]");
    // 782..789: passed_mg
    for (int r = 0; r < 8; r++) add(&g_eval.passed_mg[r], K_MG, "passed_mg[" + std::to_string(r) + "]");
    // 790..797: passed_eg
    for (int r = 0; r < 8; r++) add(&g_eval.passed_eg[r], K_EG, "passed_eg[" + std::to_string(r) + "]");
    g_params.IDX_CONN_MG       = add(&g_eval.connected_passed_mg, K_MG, "connected_mg");
    g_params.IDX_CONN_EG       = add(&g_eval.connected_passed_eg, K_EG, "connected_eg");
    g_params.IDX_DOUBLED_MG    = add(&g_eval.doubled_mg, K_MG, "doubled_mg");
    g_params.IDX_DOUBLED_EG    = add(&g_eval.doubled_eg, K_EG, "doubled_eg");
    g_params.IDX_ISOLATED_MG   = add(&g_eval.isolated_mg, K_MG, "isolated_mg");
    g_params.IDX_ISOLATED_EG   = add(&g_eval.isolated_eg, K_EG, "isolated_eg");
    g_params.IDX_ROOK_OPEN_MG  = add(&g_eval.rook_open_mg, K_MG, "rook_open_mg");
    g_params.IDX_ROOK_OPEN_EG  = add(&g_eval.rook_open_eg, K_EG, "rook_open_eg");
    g_params.IDX_ROOK_SEMI_MG  = add(&g_eval.rook_semi_mg, K_MG, "rook_semi_mg");
    g_params.IDX_ROOK_SEMI_EG  = add(&g_eval.rook_semi_eg, K_EG, "rook_semi_eg");
    g_params.IDX_KS_CLOSE      = add(&g_eval.king_shelter_close,   K_MG, "ks_close");
    g_params.IDX_KS_FAR        = add(&g_eval.king_shelter_far,     K_MG, "ks_far");
    g_params.IDX_KS_MISSING    = add(&g_eval.king_shelter_missing, K_MG, "ks_missing");
    g_params.IDX_KS_BEHIND     = add(&g_eval.king_shelter_behind,  K_MG, "ks_behind");
    g_params.IDX_MOB_MG        = add(&g_eval.mobility_mg, K_MG, "mob_mg");
    g_params.IDX_MOB_EG        = add(&g_eval.mobility_eg, K_EG, "mob_eg");
    g_params.IDX_BPAIR_MG      = add(&g_eval.bishop_pair_mg, K_MG, "bpair_mg");
    g_params.IDX_BPAIR_EG      = add(&g_eval.bishop_pair_eg, K_EG, "bpair_eg");
    g_params.IDX_TEMPO         = add(&g_eval.tempo, K_ALWAYS, "tempo");
}

// ---------------------------------------------------------------------------
// Sample storage. Sparse features, integer counts.
// ---------------------------------------------------------------------------
struct Sample {
    int16_t phase;          // 0..24
    int8_t  tempo_sign;     // +1 white-to-move, -1 black-to-move (added to ALWAYS-kind)
    float   label;          // 1.0 / 0.5 / 0.0
    std::vector<std::pair<int32_t, int16_t>> mg;   // (idx, signed count)
    std::vector<std::pair<int32_t, int16_t>> eg;
};

// ---------------------------------------------------------------------------
// Feature extraction — mirrors evaluate_hce(b) exactly.
// We accumulate signed contributions per param (white -> +, black -> -).
// ---------------------------------------------------------------------------
constexpr int PHASE_W[7] = { 0, 0, 1, 1, 2, 4, 0 };

namespace {
struct PassedMasksT {
    Bitboard m[2][64];
    PassedMasksT() {
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
}  // namespace
static const PassedMasksT TUNER_PASSED;

struct FeatAcc {
    std::unordered_map<int32_t, int32_t> mg, eg;
    void add_mg(int idx, int delta) { if (delta) mg[idx] += delta; }
    void add_eg(int idx, int delta) { if (delta) eg[idx] += delta; }
};

static void accumulate_pawn_struct(const Board& b, Color us, int sign, FeatAcc& acc) {
    Bitboard own = b.pieces(us, PAWN);
    Bitboard opp = b.pieces(~us, PAWN);
    int file_count[8] = { 0 };
    Bitboard tmp = own;
    while (tmp) { file_count[file_of(pop_lsb(tmp))]++; }
    tmp = own;
    while (tmp) {
        int sq = pop_lsb(tmp);
        int f = file_of(sq), r = rank_of(sq);
        int rel = us == WHITE ? r : 7 - r;
        if (file_count[f] > 1) {
            acc.add_mg(g_params.IDX_DOUBLED_MG, sign);
            acc.add_eg(g_params.IDX_DOUBLED_EG, sign);
        }
        bool isolated = true;
        if (f > 0 && file_count[f - 1] > 0) isolated = false;
        if (f < 7 && file_count[f + 1] > 0) isolated = false;
        if (isolated) {
            acc.add_mg(g_params.IDX_ISOLATED_MG, sign);
            acc.add_eg(g_params.IDX_ISOLATED_EG, sign);
        }
        if (!(TUNER_PASSED.m[us][sq] & opp)) {
            acc.add_mg(g_params.idx_passed_mg(rel), sign);
            acc.add_eg(g_params.idx_passed_eg(rel), sign);
            Bitboard adj = 0;
            if (f > 0) adj |= file_bb(f - 1);
            if (f < 7) adj |= file_bb(f + 1);
            Bitboard near_ranks = rank_bb(r);
            if (r > 0) near_ranks |= rank_bb(r - 1);
            if (r < 7) near_ranks |= rank_bb(r + 1);
            if (own & adj & near_ranks) {
                acc.add_mg(g_params.IDX_CONN_MG, sign);
                acc.add_eg(g_params.IDX_CONN_EG, sign);
            }
        }
    }
}

static void accumulate_rook_files(const Board& b, Color us, int sign, FeatAcc& acc) {
    Bitboard own_p = b.pieces(us, PAWN);
    Bitboard opp_p = b.pieces(~us, PAWN);
    Bitboard rooks = b.pieces(us, ROOK);
    while (rooks) {
        int sq = pop_lsb(rooks);
        Bitboard fbb = file_bb(file_of(sq));
        if (!(fbb & own_p)) {
            if (!(fbb & opp_p)) {
                acc.add_mg(g_params.IDX_ROOK_OPEN_MG, sign);
                acc.add_eg(g_params.IDX_ROOK_OPEN_EG, sign);
            } else {
                acc.add_mg(g_params.IDX_ROOK_SEMI_MG, sign);
                acc.add_eg(g_params.IDX_ROOK_SEMI_EG, sign);
            }
        }
    }
}

static void accumulate_king_safety(const Board& b, Color us, int sign, FeatAcc& acc) {
    int ksq = b.king_sq(us);
    int f = file_of(ksq);
    int kr = rank_of(ksq);
    Bitboard own_pawns = b.pieces(us, PAWN);
    int lo_f = std::max(0, f - 1), hi_f = std::min(7, f + 1);
    for (int ff = lo_f; ff <= hi_f; ff++) {
        Bitboard col = file_bb(ff) & own_pawns;
        if (!col) {
            acc.add_mg(g_params.IDX_KS_MISSING, sign);
            continue;
        }
        int p_sq = us == WHITE ? lsb(col) : msb(col);
        int rel  = us == WHITE ? rank_of(p_sq) - kr : kr - rank_of(p_sq);
        if (rel == 1)      acc.add_mg(g_params.IDX_KS_CLOSE, sign);
        else if (rel == 2) acc.add_mg(g_params.IDX_KS_FAR,   sign);
        else if (rel <= 0) acc.add_mg(g_params.IDX_KS_BEHIND, sign);
    }
}

static int compute_mobility(const Board& b, Color us) {
    Bitboard occ = b.pieces();
    Bitboard us_bb = b.pieces(us);
    int total = 0;
    Bitboard kn = b.pieces(us, KNIGHT);
    while (kn) total += popcount(knight_attacks_bb(pop_lsb(kn)) & ~us_bb);
    Bitboard bp = b.pieces(us, BISHOP);
    while (bp) total += popcount(bishop_attacks(pop_lsb(bp), occ) & ~us_bb);
    Bitboard rk = b.pieces(us, ROOK);
    while (rk) total += popcount(rook_attacks(pop_lsb(rk), occ) & ~us_bb);
    Bitboard qn = b.pieces(us, QUEEN);
    while (qn) total += popcount(queen_attacks(pop_lsb(qn), occ) & ~us_bb);
    return total;
}

static void extract_features(const Board& b, Sample& s) {
    FeatAcc acc;
    int phase = 0;

    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        int sign = (col == WHITE) ? +1 : -1;
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = b.pieces(col, PieceType(pt));
            int n = popcount(bb);
            phase += n * PHASE_W[pt];
            // mg_val / eg_val
            acc.add_mg(g_params.idx_mg_val(pt), sign * n);
            acc.add_eg(g_params.idx_eg_val(pt), sign * n);
            // PSTs
            Bitboard tmp = bb;
            while (tmp) {
                int sq = pop_lsb(tmp);
                int idx = (col == WHITE) ? sq : sq ^ 56;
                acc.add_mg(g_params.idx_pst_mg(pt, idx), sign);
                acc.add_eg(g_params.idx_pst_eg(pt, idx), sign);
            }
        }
        if (popcount(b.pieces(col, BISHOP)) >= 2) {
            acc.add_mg(g_params.IDX_BPAIR_MG, sign);
            acc.add_eg(g_params.IDX_BPAIR_EG, sign);
        }
        accumulate_pawn_struct(b, col, sign, acc);
        accumulate_rook_files(b, col, sign, acc);
        accumulate_king_safety(b, col, sign, acc);
        int mob = compute_mobility(b, col);
        acc.add_mg(g_params.IDX_MOB_MG, sign * mob);
        acc.add_eg(g_params.IDX_MOB_EG, sign * mob);
    }

    // Engine: score = base + (stm==WHITE ? tempo : -tempo); return stm==WHITE ? score : -score
    // Convert that stm-pov return back to white-pov (we want labels {1.0,0.5,0.0}):
    //   stm=WHITE: white-pov = base + tempo
    //   stm=BLACK: white-pov = -(base - tempo) ... but `base` here is already the
    //   white-pov mg_score (mg[W]-mg[B]); so the engine's `base` term, viewed in
    //   white-pov, is just `base`. Therefore:
    //     white-pov = base + sign(stm) * tempo
    s.tempo_sign = (b.side_to_move() == WHITE) ? +1 : -1;
    s.label = 0.f;  // filled by caller
    s.phase = (phase > 24) ? 24 : int16_t(phase);

    s.mg.clear(); s.eg.clear();
    s.mg.reserve(acc.mg.size());
    s.eg.reserve(acc.eg.size());
    for (auto& kv : acc.mg) if (kv.second) s.mg.push_back({ kv.first, int16_t(kv.second) });
    for (auto& kv : acc.eg) if (kv.second) s.eg.push_back({ kv.first, int16_t(kv.second) });
}

// ---------------------------------------------------------------------------
// Eval from (sample, theta) — must equal evaluate_hce(b) (modulo /24 trunc).
// Returns WHITE-POV eval (so labels {1.0, 0.5, 0.0} apply directly).
// ---------------------------------------------------------------------------
static double eval_from_features(const Sample& s, const std::vector<double>& theta) {
    double mg = 0, eg = 0;
    for (auto& f : s.mg) mg += theta[f.first] * f.second;
    for (auto& f : s.eg) eg += theta[f.first] * f.second;
    double base = (mg * s.phase + eg * (24 - s.phase)) / 24.0;
    return base + theta[g_params.IDX_TEMPO] * s.tempo_sign;
}

// ---------------------------------------------------------------------------
// Sigmoid loss + K solver.
// ---------------------------------------------------------------------------
static inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

static double mse(const std::vector<Sample>& samples, const std::vector<double>& theta, double K) {
    double total = 0;
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<double> sums(n_threads, 0.0);
    std::vector<std::thread> tt;
    size_t per = samples.size() / n_threads + 1;
    for (int t = 0; t < n_threads; ++t) {
        size_t lo = size_t(t) * per;
        size_t hi = std::min(samples.size(), lo + per);
        tt.emplace_back([&, t, lo, hi]() {
            double s = 0;
            for (size_t i = lo; i < hi; ++i) {
                double e = eval_from_features(samples[i], theta);
                double p = sigmoid(K * e);
                double d = p - samples[i].label;
                s += d * d;
            }
            sums[t] = s;
        });
    }
    for (auto& th : tt) th.join();
    for (double s : sums) total += s;
    return total / double(samples.size());
}

static double solve_K(const std::vector<Sample>& samples, const std::vector<double>& theta) {
    // Coarse search then golden section.
    double best_K = 1.0, best_loss = 1e18;
    for (double K = 0.0005; K <= 0.05; K += 0.0005) {
        double l = mse(samples, theta, K);
        if (l < best_loss) { best_loss = l; best_K = K; }
    }
    // Refine
    double a = std::max(0.0001, best_K - 0.0005);
    double b = best_K + 0.0005;
    for (int it = 0; it < 25; it++) {
        double m1 = a + (b - a) / 3, m2 = b - (b - a) / 3;
        double l1 = mse(samples, theta, m1);
        double l2 = mse(samples, theta, m2);
        if (l1 < l2) b = m2; else a = m1;
    }
    return (a + b) / 2;
}

// ---------------------------------------------------------------------------
// Adam optimizer — analytic gradient.
// ---------------------------------------------------------------------------
struct AdamState {
    std::vector<double> m, v;
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    double lr = 1.0;
    int step = 0;
};

static void adam_update(AdamState& a, std::vector<double>& theta,
                        const std::vector<double>& grad) {
    a.step++;
    double bias1 = 1 - std::pow(a.beta1, a.step);
    double bias2 = 1 - std::pow(a.beta2, a.step);
    for (size_t i = 0; i < theta.size(); ++i) {
        a.m[i] = a.beta1 * a.m[i] + (1 - a.beta1) * grad[i];
        a.v[i] = a.beta2 * a.v[i] + (1 - a.beta2) * grad[i] * grad[i];
        double mh = a.m[i] / bias1;
        double vh = a.v[i] / bias2;
        theta[i] -= a.lr * mh / (std::sqrt(vh) + a.eps);
    }
}

// Gradient = (2/N) sum_i (sigmoid(K*eval) - label) * sigma' * K * d_eval/d_theta
// d_eval/d_theta_mg[i] += count * phase / 24
// d_eval/d_theta_eg[i] += count * (24-phase) / 24
// d_eval/d_theta_tempo = tempo_sign
static void compute_gradient(const std::vector<Sample>& samples,
                             const std::vector<double>& theta,
                             double K,
                             std::vector<double>& grad_out) {
    std::fill(grad_out.begin(), grad_out.end(), 0.0);
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::vector<double>> grads(n_threads, std::vector<double>(theta.size(), 0.0));
    std::vector<std::thread> tt;
    size_t per = samples.size() / n_threads + 1;
    for (int t = 0; t < n_threads; ++t) {
        size_t lo = size_t(t) * per;
        size_t hi = std::min(samples.size(), lo + per);
        tt.emplace_back([&, t, lo, hi]() {
            auto& g = grads[t];
            for (size_t i = lo; i < hi; ++i) {
                const Sample& s = samples[i];
                double e = eval_from_features(s, theta);
                double p = sigmoid(K * e);
                double err = p - s.label;
                double coef = err * p * (1.0 - p) * K * 2.0 / double(samples.size());
                double mg_mul = coef * double(s.phase) / 24.0;
                double eg_mul = coef * double(24 - s.phase) / 24.0;
                for (auto& f : s.mg) g[f.first] += mg_mul * f.second;
                for (auto& f : s.eg) g[f.first] += eg_mul * f.second;
                g[g_params.IDX_TEMPO] += coef * double(s.tempo_sign);
            }
        });
    }
    for (auto& th : tt) th.join();
    for (auto& g : grads)
        for (size_t i = 0; i < g.size(); ++i) grad_out[i] += g[i];
}

// ---------------------------------------------------------------------------
// EPD loader: <fen> c9 "1-0|0-1|1/2-1/2";
// ---------------------------------------------------------------------------
static bool parse_epd_line(const std::string& line, std::string& fen, float& label) {
    auto pos_c9 = line.find("c9");
    if (pos_c9 == std::string::npos) return false;
    fen = line.substr(0, pos_c9);
    while (!fen.empty() && (fen.back() == ' ' || fen.back() == '\t')) fen.pop_back();
    if (fen.size() < 6) return false;
    auto qa = line.find('"', pos_c9);
    auto qb = line.find('"', qa + 1);
    if (qa == std::string::npos || qb == std::string::npos) return false;
    std::string r = line.substr(qa + 1, qb - qa - 1);
    if (r == "1-0")           label = 1.0f;
    else if (r == "0-1")      label = 0.0f;
    else if (r == "1/2-1/2")  label = 0.5f;
    else return false;
    // EPD has only 4 fields; pad halfmove + fullmove for FEN.
    int spaces = 0;
    for (char c : fen) if (c == ' ') spaces++;
    if (spaces == 3) fen += " 0 1";
    return true;
}

static std::vector<Sample> load_epd(const std::string& path, size_t cap) {
    std::vector<Sample> out;
    std::ifstream fp(path);
    if (!fp) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        std::exit(2);
    }
    std::string line;
    size_t total = 0, parsed = 0, valid = 0;
    while (std::getline(fp, line)) {
        ++total;
        std::string fen;
        float label = 0;
        if (!parse_epd_line(line, fen, label)) continue;
        ++parsed;
        Board b;
        try { b.set(fen); } catch (...) { continue; }
        Sample s;
        extract_features(b, s);
        s.label = label;
        out.push_back(std::move(s));
        ++valid;
        if (cap > 0 && out.size() >= cap) break;
        if (parsed % 50000 == 0)
            std::cerr << "  loaded " << parsed << " positions...\n";
    }
    std::cerr << "Loaded " << valid << " / " << parsed << " parsed / " << total << " lines\n";
    return out;
}

// ---------------------------------------------------------------------------
// Sanity check: feature-eval should equal evaluate_hce within a small tolerance.
// We verify on the first N samples.
// ---------------------------------------------------------------------------
static void sanity_check(const std::string& epd_path, const std::vector<double>& theta) {
    std::ifstream fp(epd_path);
    if (!fp) return;
    std::string line;
    int ok = 0, bad = 0, total = 0;
    while (std::getline(fp, line) && total < 200) {
        std::string fen; float label;
        if (!parse_epd_line(line, fen, label)) continue;
        ++total;
        Board b;
        try { b.set(fen); } catch (...) { continue; }
        Sample s;
        extract_features(b, s);
        double feat_eval = eval_from_features(s, theta);
        int eng_white_pov = evaluate_hce(b);
        // evaluate_hce returns from STM POV. Convert to white POV.
        if (b.side_to_move() == BLACK) eng_white_pov = -eng_white_pov;
        // Our white-pov accounting via sample uses tempo_sign=+1 always; engine
        // flips tempo with stm. So white-pov should match exactly modulo /24 trunc.
        // Adjust: sample stores tempo_sign=+1 but engine adds +tempo for WHITE,
        // -tempo for BLACK then flips. After flip-to-white-pov, the engine has:
        //   sign(stm)*base + tempo  (always)
        // and our feature-eval is base + tempo (since base is built from white-pov
        // signed contributions). So the only sign flip is on `base` for BLACK stm.
        if (b.side_to_move() == BLACK) {
            // Our base is white-pov-signed; engine's base in white-pov is also
            // white-pov-signed (since sign(stm)*base = -base for BLACK, but we
            // multiplied by -1 above to flip back, so it's +base). Same. OK.
        }
        if (std::abs(feat_eval - eng_white_pov) <= 2) ok++;
        else {
            if (bad < 3) std::cerr << "  MISMATCH " << fen << " feat=" << feat_eval
                                   << " eng=" << eng_white_pov << "\n";
            bad++;
        }
    }
    std::cerr << "Sanity: " << ok << " match, " << bad << " mismatch / " << total << "\n";
}

// ---------------------------------------------------------------------------
// Write back: copy theta into g_eval (clamped to int), then rewrite the
// literal block in src/evaluate.cpp between the sentinel markers.
// ---------------------------------------------------------------------------
static void copy_theta_to_eval(const std::vector<double>& theta) {
    for (size_t i = 0; i < g_params.p.size(); ++i) {
        long v = std::lround(theta[i]);
        if (v >  9999) v =  9999;
        if (v < -9999) v = -9999;
        *g_params.p[i].slot = int(v);
    }
}

static std::string fmt_pst(const int* arr) {
    std::ostringstream ss;
    for (int r = 0; r < 8; r++) {
        ss << "    ";
        for (int f = 0; f < 8; f++) {
            int v = arr[r * 8 + f];
            ss << " ";
            if (v >= 0) ss << " ";
            char buf[16]; std::snprintf(buf, sizeof(buf), "%4d", v);
            ss << buf;
            if (!(r == 7 && f == 7)) ss << ",";
        }
        ss << "\n";
    }
    return ss.str();
}

static void rewrite_evaluate_cpp(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::cerr << "Cannot open " << path << " for read\n"; return; }
    std::ostringstream all; all << in.rdbuf();
    std::string src = all.str();

    const std::string BEG = "// === TEXEL-TUNED-WEIGHTS-BEGIN ===";
    const std::string END = "// === TEXEL-TUNED-WEIGHTS-END ===";
    auto a = src.find(BEG);
    auto b = src.find(END);
    if (a == std::string::npos || b == std::string::npos) {
        std::cerr << "Sentinel markers not found in " << path << "\n";
        return;
    }

    std::ostringstream out;
    out << BEG << "\n"
        << "// Tuner edits the contents of this block in place. Keep formatting stable.\n\n";

    out << "// PeSTO MG/EG piece values.\n";
    out << "constexpr int LIT_MG_VAL[7] = { ";
    for (int i = 0; i < 7; i++) out << g_eval.mg_val[i] << (i==6?" };\n":", ");
    out << "constexpr int LIT_EG_VAL[7] = { ";
    for (int i = 0; i < 7; i++) out << g_eval.eg_val[i] << (i==6?" };\n":", ");

    out << "\n// PeSTO piece-square tables. Index 0..63 in (file + rank*8) order.\n";
    auto write_pst = [&](const char* name, const int* arr) {
        out << "constexpr int " << name << "[64] = {\n" << fmt_pst(arr) << "};\n";
    };
    write_pst("LIT_PAWN_MG",   g_eval.pst_mg[PAWN]);
    write_pst("LIT_PAWN_EG",   g_eval.pst_eg[PAWN]);
    write_pst("LIT_KNIGHT_MG", g_eval.pst_mg[KNIGHT]);
    write_pst("LIT_KNIGHT_EG", g_eval.pst_eg[KNIGHT]);
    write_pst("LIT_BISHOP_MG", g_eval.pst_mg[BISHOP]);
    write_pst("LIT_BISHOP_EG", g_eval.pst_eg[BISHOP]);
    write_pst("LIT_ROOK_MG",   g_eval.pst_mg[ROOK]);
    write_pst("LIT_ROOK_EG",   g_eval.pst_eg[ROOK]);
    write_pst("LIT_QUEEN_MG",  g_eval.pst_mg[QUEEN]);
    write_pst("LIT_QUEEN_EG",  g_eval.pst_eg[QUEEN]);
    write_pst("LIT_KING_MG",   g_eval.pst_mg[KING]);
    write_pst("LIT_KING_EG",   g_eval.pst_eg[KING]);

    out << "\n// Pawn structure / king safety / mobility / misc weights.\n";
    out << "constexpr int LIT_PASSED_MG[8] = { ";
    for (int i = 0; i < 8; i++) out << g_eval.passed_mg[i] << (i==7?" };\n":", ");
    out << "constexpr int LIT_PASSED_EG[8] = { ";
    for (int i = 0; i < 8; i++) out << g_eval.passed_eg[i] << (i==7?" };\n":", ");
    out << "constexpr int LIT_CONNECTED_PASSED_MG = " << g_eval.connected_passed_mg << ";\n";
    out << "constexpr int LIT_CONNECTED_PASSED_EG = " << g_eval.connected_passed_eg << ";\n";
    out << "constexpr int LIT_DOUBLED_MG = " << g_eval.doubled_mg << ";\n";
    out << "constexpr int LIT_DOUBLED_EG = " << g_eval.doubled_eg << ";\n";
    out << "constexpr int LIT_ISOLATED_MG = " << g_eval.isolated_mg << ";\n";
    out << "constexpr int LIT_ISOLATED_EG = " << g_eval.isolated_eg << ";\n";
    out << "constexpr int LIT_ROOK_OPEN_MG = " << g_eval.rook_open_mg << ";\n";
    out << "constexpr int LIT_ROOK_OPEN_EG = " << g_eval.rook_open_eg << ";\n";
    out << "constexpr int LIT_ROOK_SEMI_MG = " << g_eval.rook_semi_mg << ";\n";
    out << "constexpr int LIT_ROOK_SEMI_EG = " << g_eval.rook_semi_eg << ";\n";
    out << "constexpr int LIT_KING_SHELTER_CLOSE = " << g_eval.king_shelter_close << ";\n";
    out << "constexpr int LIT_KING_SHELTER_FAR = " << g_eval.king_shelter_far << ";\n";
    out << "constexpr int LIT_KING_SHELTER_MISSING = " << g_eval.king_shelter_missing << ";\n";
    out << "constexpr int LIT_KING_SHELTER_BEHIND = " << g_eval.king_shelter_behind << ";\n";
    out << "constexpr int LIT_MOBILITY_MG = " << g_eval.mobility_mg << ";\n";
    out << "constexpr int LIT_MOBILITY_EG = " << g_eval.mobility_eg << ";\n";
    out << "constexpr int LIT_BISHOP_PAIR_MG = " << g_eval.bishop_pair_mg << ";\n";
    out << "constexpr int LIT_BISHOP_PAIR_EG = " << g_eval.bishop_pair_eg << ";\n";
    out << "constexpr int LIT_TEMPO = " << g_eval.tempo << ";\n";
    out << "\n";

    std::string replacement = out.str();
    std::string before = src.substr(0, a);
    std::string after  = src.substr(b);
    std::ofstream of(path, std::ios::binary);
    of << before << replacement << after;
    std::cerr << "Wrote tuned weights to " << path << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string epd = "tuner/quiet-labeled.epd";
    std::string eval_cpp = "src/evaluate.cpp";
    int iters = 10000;
    int sample_cap = 200000;
    double lr = 1.0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](double dflt) {
            if (i + 1 >= argc) return dflt;
            return std::atof(argv[++i]);
        };
        auto nexts = [&]() -> std::string {
            if (i + 1 >= argc) return "";
            return argv[++i];
        };
        if      (a == "--epd")    epd = nexts();
        else if (a == "--evalcpp") eval_cpp = nexts();
        else if (a == "--iters")  iters = int(next(iters));
        else if (a == "--sample") sample_cap = int(next(sample_cap));
        else if (a == "--lr")     lr = next(lr);
    }

    Zobrist::init();
    init_movegen();
    NNUE::unload();
    evaluate_reload_weights();
    build_params();
    std::cerr << "Params: " << g_params.p.size() << "\n";

    std::vector<double> theta(g_params.p.size());
    for (size_t i = 0; i < theta.size(); ++i) theta[i] = double(*g_params.p[i].slot);

    std::cerr << "Loading EPD: " << epd << " (cap " << sample_cap << ")\n";
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Sample> samples = load_epd(epd, sample_cap);
    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "Loaded in " << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()
              << "s; " << samples.size() << " samples\n";

    if (samples.empty()) { std::cerr << "No samples; aborting.\n"; return 2; }

    sanity_check(epd, theta);

    std::cerr << "Solving K...\n";
    double K = solve_K(samples, theta);
    std::cerr << "K = " << K << "\n";
    std::cerr << "Initial loss = " << mse(samples, theta, K) << "\n";

    AdamState adam;
    adam.lr = lr;
    adam.m.assign(theta.size(), 0.0);
    adam.v.assign(theta.size(), 0.0);
    std::vector<double> grad(theta.size());

    double best_loss = 1e18;
    std::vector<double> best_theta = theta;
    auto train_t0 = std::chrono::steady_clock::now();

    for (int it = 1; it <= iters; ++it) {
        compute_gradient(samples, theta, K, grad);
        adam_update(adam, theta, grad);
        if (it % 100 == 0 || it == 1 || it == iters) {
            double l = mse(samples, theta, K);
            auto now = std::chrono::steady_clock::now();
            auto secs = std::chrono::duration_cast<std::chrono::seconds>(now - train_t0).count();
            std::cerr << "iter " << it << " loss=" << l << " (" << secs << "s)\n";
            if (l < best_loss) { best_loss = l; best_theta = theta; }
        }
    }

    theta = best_theta;
    std::cerr << "Final loss = " << best_loss << "\n";
    copy_theta_to_eval(theta);
    rewrite_evaluate_cpp(eval_cpp);
    std::cerr << "Done.\n";
    return 0;
}

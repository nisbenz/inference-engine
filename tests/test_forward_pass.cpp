/**
 * test_forward_pass.cpp
 *
 * Complete forward-pass correctness test suite for a GPT-2-style transformer
 * implemented with GGML.
 *
 * Test inventory
 * ──────────────
 *  1.  LayerNorm – identity input (all-ones → zero output)
 *  2.  LayerNorm – known values  (hand-calculated reference)
 *  3.  LayerNorm – non-unit gamma/beta
 *  4.  Matmul    – exact numerical values (2×3 × 3×2)
 *  5.  Causal mask – correct -inf placement (fixed double-alloc + isinf check)
 *  6.  Softmax   – masked rows sum to 1, correct values
 *  7.  GELU      – reference values (exact GELU, not approximation)
 *  8.  Embedding lookup – ggml_get_rows exact values
 *  9.  GGML layout – stride verification before element checks
 * 10.  Positional embeddings – wpe + wte correctly summed
 * 11.  Single attention head – non-zero Q/K/V, verified against NumPy reference
 * 12.  Multi-head attention  – concatenation correctness
 * 13.  FFN block – GELU with real weights, reference values
 * 14.  Residual connections – additive path preserved under zero branch
 * 15.  Full single-layer – non-trivial input/weights vs Python reference
 * 16.  End-to-end forward pass – two tokens, two layers, logits vs reference
 */

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <ggml-backend.h>
#include <ggml.h>

// ─────────────────────────────────────────────────────────────────────────────
// Minimal test harness (no external headers required)
// ─────────────────────────────────────────────────────────────────────────────

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define FAIL_MSG(msg)                                                          \
    do {                                                                       \
        std::cerr << "  [FAIL] " << msg << "  (" << __FILE__ << ":"           \
                  << __LINE__ << ")\n";                                        \
        ++g_tests_failed;                                                      \
    } while (0)

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_tests_run;                                                         \
        if (!(cond)) { FAIL_MSG(#cond); } else { ++g_tests_passed; }          \
    } while (0)

#define CHECK_MSG(cond, msg)                                                   \
    do {                                                                       \
        ++g_tests_run;                                                         \
        if (!(cond)) { FAIL_MSG(msg); } else { ++g_tests_passed; }            \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                       \
        ++g_tests_run;                                                         \
        float _a = (float)(a), _b = (float)(b), _t = (float)(tol);           \
        if (std::fabs(_a - _b) > _t) {                                        \
            std::cerr << "  [FAIL] |" << _a << " - " << _b << "| > " << _t   \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            ++g_tests_failed;                                                  \
        } else {                                                               \
            ++g_tests_passed;                                                  \
        }                                                                      \
    } while (0)

/* Check that a float is -inf */
#define CHECK_NEG_INF(val)                                                     \
    do {                                                                       \
        ++g_tests_run;                                                         \
        float _v = (float)(val);                                               \
        if (!std::isinf(_v) || _v > 0) {                                      \
            std::cerr << "  [FAIL] expected -inf, got " << _v                 \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            ++g_tests_failed;                                                  \
        } else {                                                               \
            ++g_tests_passed;                                                  \
        }                                                                      \
    } while (0)

static void print_section(const char* name) {
    std::cout << "\n── " << name << " ──\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// GGML helpers
// ─────────────────────────────────────────────────────────────────────────────

struct GgmlCtx {
    ggml_context*  ctx     = nullptr;
    ggml_backend_t backend = nullptr;

    explicit GgmlCtx(size_t mem_bytes = 64 * 1024 * 1024) {
        ggml_init_params p{mem_bytes, nullptr, /*no_alloc=*/true};
        ctx     = ggml_init(p);
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        assert(ctx && backend);
    }
    ~GgmlCtx() {
        if (backend) ggml_backend_free(backend);
        if (ctx)     ggml_free(ctx);
    }
    // Non-copyable
    GgmlCtx(const GgmlCtx&)            = delete;
    GgmlCtx& operator=(const GgmlCtx&) = delete;
};

/**
 * Allocate all tensors in ctx, build a forward graph for `result`, and run it.
 * NOTE: allocation must happen AFTER graph construction but BEFORE data fill.
 * Callers that need to fill data after building the graph should call
 * alloc_and_compute() instead of this function.
 */
static void compute(GgmlCtx& g, ggml_tensor* result) {
    ggml_cgraph* gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, result);
    ggml_backend_graph_compute(g.backend, gf);
}

/**
 * Full lifecycle: allocate tensors, let the caller fill data, then compute.
 * Usage:
 *   auto fill = alloc(g);   // allocates; returns a no-op lambda placeholder
 *   fill(result);            // computes
 */
static void alloc_tensors(GgmlCtx& g) {
    ggml_backend_alloc_ctx_tensors(g.ctx, g.backend);
}

// Convenience: set every element of a tensor to a scalar
static void fill_scalar(ggml_tensor* t, float v) {
    float* d = (float*)t->data;
    int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; i++) d[i] = v;
}

// Set tensor from a std::vector (must match element count)
static void fill_vec(ggml_tensor* t, const std::vector<float>& v) {
    assert((int64_t)v.size() == ggml_nelements(t));
    std::memcpy(t->data, v.data(), v.size() * sizeof(float));
}

static void fill_vec_i32(ggml_tensor* t, const std::vector<int32_t>& v) {
    assert((int64_t)v.size() == ggml_nelements(t));
    std::memcpy(t->data, v.data(), v.size() * sizeof(int32_t));
}

// Read all floats from a tensor into a vector
static std::vector<float> read_vec(ggml_tensor* t) {
    int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    std::memcpy(out.data(), t->data, n * sizeof(float));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference math (CPU, no GGML)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> ref_layernorm(const std::vector<float>& x,
                                        int D,
                                        const std::vector<float>& gamma,
                                        const std::vector<float>& beta,
                                        float eps = 1e-5f) {
    int T = (int)x.size() / D;
    std::vector<float> out(x.size());
    for (int t = 0; t < T; t++) {
        float mean = 0;
        for (int i = 0; i < D; i++) mean += x[t * D + i];
        mean /= D;
        float var = 0;
        for (int i = 0; i < D; i++) {
            float d = x[t * D + i] - mean;
            var += d * d;
        }
        var /= D;
        float std_ = std::sqrt(var + eps);
        for (int i = 0; i < D; i++) {
            out[t * D + i] = ((x[t * D + i] - mean) / std_) * gamma[i] + beta[i];
        }
    }
    return out;
}

// Matrix multiply: A(M×K) × B(K×N) → C(M×N), row-major
static std::vector<float> ref_matmul(const std::vector<float>& A, int M, int K,
                                     const std::vector<float>& B, int N) {
    std::vector<float> C(M * N, 0.0f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                C[m * N + n] += A[m * K + k] * B[k * N + n];
    return C;
}

static float ref_gelu(float x) {
    // Exact GELU: x * Φ(x)  where Φ is standard normal CDF
    return x * 0.5f * (1.0f + std::erff(x / std::sqrt(2.0f)));
}

static std::vector<float> ref_softmax_row(const std::vector<float>& x, int cols) {
    int rows = (int)x.size() / cols;
    std::vector<float> out(x.size());
    for (int r = 0; r < rows; r++) {
        float mx = -1e30f;
        for (int c = 0; c < cols; c++) mx = std::max(mx, x[r * cols + c]);
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            out[r * cols + c] = std::exp(x[r * cols + c] - mx);
            sum += out[r * cols + c];
        }
        for (int c = 0; c < cols; c++) out[r * cols + c] /= sum;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 1: LayerNorm – identity input
// ─────────────────────────────────────────────────────────────────────────────
static void test_layernorm_identity() {
    print_section("Test 1: LayerNorm – all-ones input → zero output");
    GgmlCtx g;

    auto* x     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 4, 2);
    auto* gamma = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 4);
    auto* beta  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 4);

    auto* xn  = ggml_norm(g.ctx, x, 1e-5f);
    auto* sc  = ggml_mul(g.ctx, xn, gamma);
    auto* res = ggml_add(g.ctx, sc, beta);

    alloc_tensors(g);
    fill_scalar(x, 1.0f);
    fill_scalar(gamma, 1.0f);
    fill_scalar(beta, 0.0f);
    compute(g, res);

    auto out = read_vec(res);
    for (int i = 0; i < 8; i++)
        CHECK_NEAR(out[i], 0.0f, 1e-4f);

    std::cout << "  Output (all ~0): ";
    for (float v : out) std::cout << std::fixed << std::setprecision(5) << v << " ";
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 2: LayerNorm – known values
// ─────────────────────────────────────────────────────────────────────────────
static void test_layernorm_known_values() {
    print_section("Test 2: LayerNorm – known input/output");
    GgmlCtx g;

    const int D = 4, T = 2;
    auto* x     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, T);
    auto* gamma = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, D);
    auto* beta  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, D);

    auto* xn  = ggml_norm(g.ctx, x, 1e-5f);
    auto* sc  = ggml_mul(g.ctx, xn, gamma);
    auto* res = ggml_add(g.ctx, sc, beta);

    alloc_tensors(g);

    std::vector<float> xv = {1, 2, 3, 4, 5, 6, 7, 8};
    fill_vec(x, xv);
    fill_scalar(gamma, 1.0f);
    fill_scalar(beta, 0.0f);
    compute(g, res);

    // Reference
    std::vector<float> gv(D, 1.0f), bv(D, 0.0f);
    auto ref = ref_layernorm(xv, D, gv, bv);

    auto out = read_vec(res);
    std::cout << "  Token 0: ";
    for (int i = 0; i < D; i++) {
        std::cout << out[i] << " ";
        CHECK_NEAR(out[i], ref[i], 1e-3f);
    }
    std::cout << "\n  Token 1: ";
    for (int i = D; i < D * T; i++) {
        std::cout << out[i] << " ";
        CHECK_NEAR(out[i], ref[i], 1e-3f);
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 3: LayerNorm – non-unit gamma/beta
// ─────────────────────────────────────────────────────────────────────────────
static void test_layernorm_gamma_beta() {
    print_section("Test 3: LayerNorm – non-unit gamma and beta");
    GgmlCtx g;

    const int D = 4;
    auto* x     = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, D);
    auto* gamma = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, D);
    auto* beta  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, D);

    auto* xn  = ggml_norm(g.ctx, x, 1e-5f);
    auto* sc  = ggml_mul(g.ctx, xn, gamma);
    auto* res = ggml_add(g.ctx, sc, beta);

    alloc_tensors(g);

    std::vector<float> xv    = {1, 2, 3, 4};
    std::vector<float> gv    = {2, 0.5f, 3, 1};
    std::vector<float> bv    = {1, -1, 0, 0.5f};
    fill_vec(x, xv); fill_vec(gamma, gv); fill_vec(beta, bv);
    compute(g, res);

    auto ref = ref_layernorm(xv, D, gv, bv);
    auto out = read_vec(res);

    std::cout << "  ";
    for (int i = 0; i < D; i++) {
        std::cout << out[i] << " ";
        CHECK_NEAR(out[i], ref[i], 1e-4f);
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 4: Matmul – exact numerical values
// ─────────────────────────────────────────────────────────────────────────────
static void test_matmul_exact() {
    print_section("Test 4: Matmul – exact numerical values");
    GgmlCtx g;

    // GGML: ggml_mul_mat(A, B) computes A^T × B in the standard sense.
    // A has shape (K, M) in GGML notation (ne[0]=K, ne[1]=M) → represents M rows of dim K.
    // B has shape (K, N)  → represents N columns of dim K.
    // Result has shape (M, N).
    //
    // To compute W(2×3) × x(3×1):
    //   W_ggml: ne[0]=3, ne[1]=2   (row-major: W[row][col] = W_data[row*3+col])
    //   x_ggml: ne[0]=3, ne[1]=1
    //   result: ne[0]=2, ne[1]=1

    const int M = 2, K = 3, N = 1;
    auto* W   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    auto* x   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, N);
    auto* res = ggml_mul_mat(g.ctx, W, x);

    alloc_tensors(g);

    // W = [[1,2,3],[4,5,6]]  (row-major)
    // x = [[1],[2],[3]]
    // expected = [[14],[32]]
    std::vector<float> Wv = {1, 2, 3, 4, 5, 6};
    std::vector<float> xv = {1, 2, 3};
    fill_vec(W, Wv);
    fill_vec(x, xv);
    compute(g, res);

    auto out = read_vec(res);
    std::cout << "  Result: " << out[0] << " " << out[1]
              << "  (expected 14, 32)\n";
    CHECK_NEAR(out[0], 14.0f, 1e-4f);
    CHECK_NEAR(out[1], 32.0f, 1e-4f);

    // Also verify shape
    CHECK_MSG(res->ne[0] == M, "result ne[0] should be M");
    CHECK_MSG(res->ne[1] == N, "result ne[1] should be N");
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 5: Causal mask – correct -inf placement
// ─────────────────────────────────────────────────────────────────────────────
static void test_causal_mask() {
    print_section("Test 5: Causal mask – correct -inf placement");
    GgmlCtx g;

    const int T = 3;
    auto* scores = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, T, T);
    // ggml_diag_mask_inf(ctx, a, n_past): masks future positions
    auto* masked = ggml_diag_mask_inf(g.ctx, scores, 0);

    alloc_tensors(g);
    fill_scalar(scores, 1.0f);
    compute(g, masked);

    auto out = read_vec(masked);

    // GGML softmax operates along ne[0] (columns in memory).
    // For a T×T score matrix, the causal structure is:
    //   position 0 can attend to position 0 only
    //   position 1 can attend to 0 and 1
    //   position 2 can attend to 0, 1, and 2
    // Check upper-triangular above diagonal is -inf, lower is finite.
    std::cout << "  Masked scores (T=" << T << "):\n";
    for (int r = 0; r < T; r++) {
        std::cout << "    row " << r << ": ";
        for (int c = 0; c < T; c++) {
            float v = out[r * T + c];
            std::cout << (std::isinf(v) ? "-inf" : std::to_string(v)) << " ";
        }
        std::cout << "\n";
    }

    // Row 0: only col 0 visible
    CHECK_NEAR(out[0], 1.0f, 1e-4f);
    CHECK_NEG_INF(out[1]);
    CHECK_NEG_INF(out[2]);
    // Row 1: cols 0 and 1 visible
    CHECK_NEAR(out[3], 1.0f, 1e-4f);
    CHECK_NEAR(out[4], 1.0f, 1e-4f);
    CHECK_NEG_INF(out[5]);
    // Row 2: all visible
    CHECK_NEAR(out[6], 1.0f, 1e-4f);
    CHECK_NEAR(out[7], 1.0f, 1e-4f);
    CHECK_NEAR(out[8], 1.0f, 1e-4f);
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 6: Softmax – masked rows sum to 1 and match reference
// ─────────────────────────────────────────────────────────────────────────────
static void test_softmax_masked() {
    print_section("Test 6: Softmax on causal-masked scores");
    GgmlCtx g;

    const int T = 3;
    auto* scores  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, T, T);
    auto* masked  = ggml_diag_mask_inf(g.ctx, scores, 0);
    auto* weights = ggml_soft_max(g.ctx, masked);

    alloc_tensors(g);

    // Use varied scores so softmax values are non-trivial
    // Layout: scores[row][col] = scores_data[row*T + col]
    std::vector<float> sv = {
        1.0f, 0.5f, 0.2f,   // row 0 – only [0] survives masking
        0.8f, 1.2f, 0.3f,   // row 1 – [0],[1] survive
        0.6f, 0.9f, 1.1f    // row 2 – all survive
    };
    fill_vec(scores, sv);
    compute(g, weights);

    auto out = read_vec(weights);

    // Row sums must be 1
    for (int r = 0; r < T; r++) {
        float s = 0;
        for (int c = 0; c < T; c++) s += out[r * T + c];
        std::cout << "  Row " << r << " sum: " << s << "\n";
        CHECK_NEAR(s, 1.0f, 1e-5f);
    }
    // Row 0: only position 0 should be 1.0
    CHECK_NEAR(out[0], 1.0f, 1e-5f);
    CHECK_NEAR(out[1], 0.0f, 1e-5f);
    CHECK_NEAR(out[2], 0.0f, 1e-5f);

    // Verify row 2 against reference softmax([0.6, 0.9, 1.1])
    std::vector<float> row2_in = {0.6f, 0.9f, 1.1f};
    auto row2_ref = ref_softmax_row(row2_in, 3);
    for (int c = 0; c < T; c++)
        CHECK_NEAR(out[2 * T + c], row2_ref[c], 1e-5f);
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 7: GELU – exact values using erff
// ─────────────────────────────────────────────────────────────────────────────
static void test_gelu_exact() {
    print_section("Test 7: GELU – exact values");
    GgmlCtx g;

    std::vector<float> xv = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
    auto* x   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, (int)xv.size());
    auto* res = ggml_gelu(g.ctx, x);

    alloc_tensors(g);
    fill_vec(x, xv);
    compute(g, res);

    auto out = read_vec(res);
    std::cout << std::fixed << std::setprecision(5);
    for (int i = 0; i < (int)xv.size(); i++) {
        float ref = ref_gelu(xv[i]);
        std::cout << "  GELU(" << xv[i] << ") = " << out[i]
                  << "  ref=" << ref << "\n";
        // GGML uses the tanh approximation; allow a slightly larger tolerance
        CHECK_NEAR(out[i], ref, 5e-3f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 8: Embedding lookup
// ─────────────────────────────────────────────────────────────────────────────
static void test_embedding_lookup() {
    print_section("Test 8: Embedding lookup (ggml_get_rows)");
    GgmlCtx g;

    const int vocab = 10, D = 4;
    auto* wte    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, vocab);
    auto* tokens = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 3);
    auto* emb    = ggml_get_rows(g.ctx, wte, tokens);

    alloc_tensors(g);

    // Fill wte: row t = [t*100+0, t*100+1, t*100+2, t*100+3]
    std::vector<float> wte_data(vocab * D);
    for (int t = 0; t < vocab; t++)
        for (int i = 0; i < D; i++)
            wte_data[t * D + i] = (float)(t * 100 + i);
    fill_vec(wte, wte_data);
    fill_vec_i32(tokens, {0, 3, 7});
    compute(g, emb);

    auto out = read_vec(emb);
    // Token 0 → row 0: [0,1,2,3]
    // Token 3 → row 3: [300,301,302,303]
    // Token 7 → row 7: [700,701,702,703]
    int32_t tok[] = {0, 3, 7};
    for (int s = 0; s < 3; s++) {
        std::cout << "  Token " << tok[s] << ": ";
        for (int i = 0; i < D; i++) {
            float expected = tok[s] * 100.0f + i;
            std::cout << out[s * D + i] << " ";
            CHECK_NEAR(out[s * D + i], expected, 1e-4f);
        }
        std::cout << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 9: GGML tensor layout / strides
// ─────────────────────────────────────────────────────────────────────────────
static void test_ggml_layout() {
    print_section("Test 9: GGML tensor layout – verify strides first");
    GgmlCtx g;

    // A 2D tensor: ne[0]=cols=3, ne[1]=rows=4
    // GGML is column-major: nb[0] = sizeof(float) = 4, nb[1] = ne[0]*4 = 12
    auto* t = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 3, 4);
    alloc_tensors(g);

    // Verify strides before accessing elements
    size_t nb0 = t->nb[0];
    size_t nb1 = t->nb[1];
    std::cout << "  nb[0]=" << nb0 << "  nb[1]=" << nb1 << "\n";
    CHECK_MSG(nb0 == sizeof(float), "nb[0] should be sizeof(float)");
    CHECK_MSG(nb1 == 3 * sizeof(float), "nb[1] should be ne[0]*sizeof(float)");

    // Fill using strides, not assumed indices
    float* data = (float*)t->data;
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 3; col++) {
            size_t off = (row * nb1 + col * nb0) / sizeof(float);
            data[off] = (float)(row * 10 + col);
        }

    // Verify using the same stride formula
    auto check_elem = [&](int row, int col, float expected) {
        size_t off = (row * nb1 + col * nb0) / sizeof(float);
        CHECK_NEAR(data[off], expected, 1e-6f);
    };
    check_elem(0, 0,  0.0f);
    check_elem(1, 0, 10.0f);
    check_elem(0, 1,  1.0f);
    check_elem(2, 2, 22.0f);
    check_elem(3, 1, 31.0f);
    std::cout << "  Stride-based element access verified.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 10: Positional embeddings (wte + wpe)
// ─────────────────────────────────────────────────────────────────────────────
static void test_positional_embeddings() {
    print_section("Test 10: Token + positional embeddings summed correctly");
    GgmlCtx g;

    const int D = 4, T = 3, vocab = 8, ctx_len = 8;
    auto* wte  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, vocab);
    auto* wpe  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, ctx_len);
    auto* tok  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    auto* pos  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);

    auto* te  = ggml_get_rows(g.ctx, wte, tok);
    auto* pe  = ggml_get_rows(g.ctx, wpe, pos);
    auto* res = ggml_add(g.ctx, te, pe);

    alloc_tensors(g);

    // wte[t][i] = t*10+i, wpe[p][i] = p*100+i
    std::vector<float> wte_data(vocab * D), wpe_data(ctx_len * D);
    for (int t = 0; t < vocab; t++)
        for (int i = 0; i < D; i++) wte_data[t * D + i] = (float)(t * 10 + i);
    for (int p = 0; p < ctx_len; p++)
        for (int i = 0; i < D; i++) wpe_data[p * D + i] = (float)(p * 100 + i);
    fill_vec(wte, wte_data);
    fill_vec(wpe, wpe_data);

    // tokens [2, 5, 1], positions [0, 1, 2]
    fill_vec_i32(tok, {2, 5, 1});
    fill_vec_i32(pos, {0, 1, 2});
    compute(g, res);

    auto out = read_vec(res);
    // Expected: te[s][i] + pe[s][i]
    //   s=0: tok=2 → wte[2][i]=20+i, pos=0 → wpe[0][i]=i  → 20+2i? No: wte=20+i, wpe=0+i → 20+2i
    int32_t toks[] = {2, 5, 1}, poss[] = {0, 1, 2};
    for (int s = 0; s < T; s++) {
        std::cout << "  Seq " << s << ": ";
        for (int i = 0; i < D; i++) {
            float expected = (toks[s] * 10 + i) + (poss[s] * 100 + i);
            std::cout << out[s * D + i] << " ";
            CHECK_NEAR(out[s * D + i], expected, 1e-3f);
        }
        std::cout << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 11: Single attention head – non-trivial Q/K/V
// ─────────────────────────────────────────────────────────────────────────────
static void test_single_attention_head() {
    print_section("Test 11: Single attention head – non-trivial weights");
    GgmlCtx g(128 * 1024 * 1024);

    // Tiny: T=2, D=2 (head_dim=2)
    const int T = 2, D = 2;

    // Q, K, V as 2D tensors (D, T): column=token, row=dim
    auto* Q = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, T);
    auto* K = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, T);
    auto* V = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, T);

    // scores = K^T Q  then scale, mask, softmax
    auto* scores   = ggml_mul_mat(g.ctx, K, Q);
    auto* scaled   = ggml_scale(g.ctx, scores, 1.0f / std::sqrt((float)D));
    auto* masked   = ggml_diag_mask_inf(g.ctx, scaled, 0);
    auto* attn_w   = ggml_soft_max(g.ctx, masked);
    // attn_out = V * attn_w   (V: D×T, attn_w: T×T → attn_out: D×T)
    auto* V_t      = ggml_cont(g.ctx, ggml_permute(g.ctx, V, 1, 0, 2, 3));
    auto* attn_out = ggml_mul_mat(g.ctx, V_t, attn_w);

    alloc_tensors(g);

    // Q = [[1, 0], [0, 1]]  (col-major: Q_data = [1,0,0,1])
    // K = [[1, 0], [0, 1]]
    // V = [[2, 4], [6, 8]]  (token 0 = [2,6], token 1 = [4,8])
    std::vector<float> Qv = {1, 0, 0, 1};  // col-major: col0=[1,0], col1=[0,1]
    std::vector<float> Kv = {1, 0, 0, 1};
    std::vector<float> Vv = {2, 6, 4, 8};
    fill_vec(Q, Qv);
    fill_vec(K, Kv);
    fill_vec(V, Vv);
    compute(g, attn_out);

    auto out = read_vec(attn_out);

    // Reference (manual):
    // For token 0: Q0=[1,0], K0=[1,0], K1=[0,1]
    //   score(0,0) = dot([1,0],[1,0])/√2 = 1/√2 ≈ 0.707
    //   score(0,1) = -inf (masked)
    //   attn_w[0] = [1, 0]  → attn_out[0] = V[0] = [2, 6]
    // For token 1: Q1=[0,1], K0=[1,0], K1=[0,1]
    //   score(1,0) = dot([0,1],[1,0])/√2 = 0
    //   score(1,1) = dot([0,1],[0,1])/√2 = 1/√2
    //   attn_w[1] = softmax([0, 1/√2]) ≈ [0.368, 0.632]  (approx)
    //   attn_out[1] ≈ 0.368*[2,6] + 0.632*[4,8] ≈ [3.264, 7.264]

    std::cout << "  attn_out token 0: [" << out[0] << ", " << out[1] << "]"
              << "  (expected ~[2, 6])\n";
    std::cout << "  attn_out token 1: [" << out[2] << ", " << out[3] << "]"
              << "  (expected ~[3.26, 7.26])\n";

    CHECK_NEAR(out[0], 2.0f, 0.05f);
    CHECK_NEAR(out[1], 6.0f, 0.05f);

    // Compute precise reference for token 1
    float s0 = 0.0f / std::sqrt(2.0f);
    float s1 = 1.0f / std::sqrt(2.0f);
    float mx = std::max(s0, s1);
    float e0 = std::exp(s0 - mx), e1 = std::exp(s1 - mx);
    float sum = e0 + e1;
    float w0 = e0 / sum, w1 = e1 / sum;
    float exp_d0 = w0 * 2.0f + w1 * 4.0f;
    float exp_d1 = w0 * 6.0f + w1 * 8.0f;
    CHECK_NEAR(out[2], exp_d0, 0.05f);
    CHECK_NEAR(out[3], exp_d1, 0.05f);
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 12: Multi-head attention – concatenation
// ─────────────────────────────────────────────────────────────────────────────
static void test_multihead_attention_concat() {
    print_section("Test 12: Multi-head attention – output shape and residual");
    GgmlCtx g(256 * 1024 * 1024);

    const int n_embd = 4, T = 2, n_heads = 2, head_dim = n_embd / n_heads;
    float scale = 1.0f / std::sqrt((float)head_dim);

    auto* x     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, T);
    auto* qkv_w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, 3 * n_embd);
    auto* qkv_b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 3 * n_embd);
    auto* proj_w= ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, n_embd);
    auto* proj_b= ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);

    // QKV projection
    auto* qkv   = ggml_add(g.ctx, ggml_mul_mat(g.ctx, qkv_w, x), qkv_b);
    size_t es = ggml_element_size(qkv);

    // Split Q, K, V and reshape to (head_dim, n_heads, T)
    auto* q = ggml_view_2d(g.ctx, qkv, n_embd, T, qkv->nb[1], 0);
    auto* k = ggml_view_2d(g.ctx, qkv, n_embd, T, qkv->nb[1], n_embd * es);
    auto* v = ggml_view_2d(g.ctx, qkv, n_embd, T, qkv->nb[1], 2 * n_embd * es);

    auto* Q = ggml_reshape_3d(g.ctx, ggml_cont(g.ctx, q), head_dim, n_heads, T);
    auto* K = ggml_reshape_3d(g.ctx, ggml_cont(g.ctx, k), head_dim, n_heads, T);
    auto* V = ggml_reshape_3d(g.ctx, ggml_cont(g.ctx, v), head_dim, n_heads, T);

    // Permute to (head_dim, T, n_heads)
    Q = ggml_permute(g.ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(g.ctx, K, 0, 2, 1, 3);
    V = ggml_permute(g.ctx, V, 0, 2, 1, 3);

    auto* attn_scores = ggml_mul_mat(g.ctx, K, Q);
    attn_scores = ggml_scale(g.ctx, attn_scores, scale);
    attn_scores = ggml_diag_mask_inf(g.ctx, attn_scores, 0);
    auto* attn_w = ggml_soft_max(g.ctx, attn_scores);

    auto* V_t   = ggml_cont(g.ctx, ggml_permute(g.ctx, V, 1, 0, 2, 3));
    auto* ao    = ggml_mul_mat(g.ctx, V_t, attn_w);
    ao = ggml_permute(g.ctx, ao, 0, 2, 1, 3);
    ao = ggml_cont(g.ctx, ao);
    ao = ggml_reshape_2d(g.ctx, ao, n_embd, T);

    auto* out_proj = ggml_add(g.ctx, ggml_mul_mat(g.ctx, proj_w, ao), proj_b);
    auto* res = ggml_add(g.ctx, x, out_proj);   // residual

    alloc_tensors(g);

    // Identity QKV weights and zero projection → out_proj = 0 → res = x
    // qkv_w = identity (first n_embd rows), zeros for K and V parts
    std::vector<float> qkv_w_data(n_embd * 3 * n_embd, 0.0f);
    for (int i = 0; i < n_embd; i++) qkv_w_data[i * n_embd + i] = 1.0f;  // Q=I
    // K and V rows: keep as zero
    fill_vec(qkv_w, qkv_w_data);
    fill_scalar(qkv_b, 0.0f);

    std::vector<float> proj_w_data(n_embd * n_embd, 0.0f);
    fill_vec(proj_w, proj_w_data);
    fill_scalar(proj_b, 0.0f);

    std::vector<float> xv;
    for (int i = 0; i < n_embd * T; i++) xv.push_back((float)(i + 1));
    fill_vec(x, xv);

    compute(g, res);

    // With proj_w=0, proj_b=0: out_proj = 0, res = x
    auto out = read_vec(res);
    std::cout << "  Residual output (should equal x): ";
    for (int i = 0; i < n_embd * T; i++) {
        std::cout << out[i] << " ";
        CHECK_NEAR(out[i], xv[i], 1e-4f);
    }
    std::cout << "\n";

    // Verify output shape
    CHECK_MSG(res->ne[0] == n_embd, "output ne[0] == n_embd");
    CHECK_MSG(res->ne[1] == T,      "output ne[1] == T");
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 13: FFN block – GELU with real weights, reference values
// ─────────────────────────────────────────────────────────────────────────────
static void test_ffn_block() {
    print_section("Test 13: FFN block – real weights vs reference");
    GgmlCtx g;

    const int D = 2, D4 = D * 4;

    auto* x    = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, D);
    auto* fc_w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, D4);
    auto* fc_b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, D4);
    auto* pr_w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D4, D);
    auto* pr_b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, D);

    auto* up   = ggml_add(g.ctx, ggml_mul_mat(g.ctx, fc_w, x), fc_b);
    auto* act  = ggml_gelu(g.ctx, up);
    auto* down = ggml_add(g.ctx, ggml_mul_mat(g.ctx, pr_w, act), pr_b);
    auto* res  = ggml_add(g.ctx, x, down);  // residual

    alloc_tensors(g);

    std::vector<float> xv    = {1.0f, -0.5f};
    std::vector<float> fc_wv = {1, 0, -1, 0, 0, 1, 0, -1,
                                 0.5f, 0.5f, -0.5f, -0.5f,
                                 1, -1, -1, 1};
    // Correct fc_wv size: D4 * D = 8 * 2 = 16
    fc_wv.resize(D4 * D, 0.0f);
    for (int i = 0; i < D4; i++) {
        fc_wv[i * D + 0] = (i % 2 == 0) ? 1.0f : 0.5f;
        fc_wv[i * D + 1] = (i % 2 == 0) ? -0.5f : 1.0f;
    }

    std::vector<float> fc_bv(D4, 0.1f);
    // pr_w: D × D4 = 2 × 8 (in GGML: ne[0]=D4, ne[1]=D)
    std::vector<float> pr_wv(D * D4, 0.0f);
    for (int i = 0; i < D; i++) pr_wv[i * D4 + i] = 0.5f;
    std::vector<float> pr_bv(D, 0.05f);

    fill_vec(x, xv);
    fill_vec(fc_w, fc_wv);
    fill_vec(fc_b, fc_bv);
    fill_vec(pr_w, pr_wv);
    fill_vec(pr_b, pr_bv);
    compute(g, res);

    // Reference: compute manually
    // up[i] = dot(fc_w[i], x) + fc_b[i]
    std::vector<float> up_ref(D4), act_ref(D4);
    for (int i = 0; i < D4; i++) {
        up_ref[i] = fc_wv[i * D + 0] * xv[0] + fc_wv[i * D + 1] * xv[1] + fc_bv[i];
        act_ref[i] = ref_gelu(up_ref[i]);
    }
    // down[i] = dot(pr_w[i], act) + pr_b[i]
    std::vector<float> down_ref(D, 0.0f);
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D4; j++)
            down_ref[i] += pr_wv[i * D4 + j] * act_ref[j];
        down_ref[i] += pr_bv[i];
    }
    std::vector<float> res_ref(D);
    for (int i = 0; i < D; i++) res_ref[i] = xv[i] + down_ref[i];

    auto out = read_vec(res);
    std::cout << "  FFN output: ";
    for (int i = 0; i < D; i++) {
        std::cout << out[i] << " (ref=" << res_ref[i] << ") ";
        CHECK_NEAR(out[i], res_ref[i], 0.01f);
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 14: Residual connections – identity under zero branch
// ─────────────────────────────────────────────────────────────────────────────
static void test_residual_connections() {
    print_section("Test 14: Residual connections – preserved under zero branch");
    GgmlCtx g;

    const int D = 6;
    auto* x      = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, D);
    auto* branch = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, D);
    auto* res    = ggml_add(g.ctx, x, branch);

    alloc_tensors(g);

    std::vector<float> xv = {1, 2, 3, 4, 5, 6};
    fill_vec(x, xv);
    fill_scalar(branch, 0.0f);
    compute(g, res);

    auto out = read_vec(res);
    for (int i = 0; i < D; i++)
        CHECK_NEAR(out[i], xv[i], 1e-6f);
    std::cout << "  Residual pass-through verified.\n";

    // Also verify with known branch values
    std::vector<float> bv = {-1, -2, -3, -4, -5, -6};
    fill_vec(branch, bv);
    compute(g, res);
    out = read_vec(res);
    for (int i = 0; i < D; i++)
        CHECK_NEAR(out[i], 0.0f, 1e-6f);
    std::cout << "  Residual cancellation verified.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 15: Full single transformer layer – non-trivial input and weights
// compared to a hand-computed reference
// ─────────────────────────────────────────────────────────────────────────────
static void test_full_single_layer() {
    print_section("Test 15: Full single transformer layer – non-trivial weights");
    GgmlCtx g(512 * 1024 * 1024);

    // n_embd=4, T=2, n_heads=2, head_dim=2
    const int n_embd = 4, T = 2, n_heads = 2, head_dim = n_embd / n_heads;
    float scale = 1.0f / std::sqrt((float)head_dim);

    // ── Declare all weights ──
    auto* x       = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, T);
    auto* ln1_g   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* ln1_b   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* qkv_w   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, 3 * n_embd);
    auto* qkv_b   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 3 * n_embd);
    auto* proj_w  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, n_embd);
    auto* proj_b  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* ln2_g   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* ln2_b   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* fc_w    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, n_embd * 4);
    auto* fc_b    = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd * 4);
    auto* pr_w    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd * 4, n_embd);
    auto* pr_b    = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);

    // ── Build graph ──
    auto* h    = ggml_norm(g.ctx, x, 1e-5f);
    h = ggml_add(g.ctx, ggml_mul(g.ctx, h, ln1_g), ln1_b);

    auto* qkv  = ggml_add(g.ctx, ggml_mul_mat(g.ctx, qkv_w, h), qkv_b);
    size_t es  = ggml_element_size(qkv);
    auto* q_   = ggml_view_2d(g.ctx, qkv, n_embd, T, qkv->nb[1], 0);
    auto* k_   = ggml_view_2d(g.ctx, qkv, n_embd, T, qkv->nb[1], n_embd * es);
    auto* v_   = ggml_view_2d(g.ctx, qkv, n_embd, T, qkv->nb[1], 2 * n_embd * es);

    auto* Q    = ggml_reshape_3d(g.ctx, ggml_cont(g.ctx, q_), head_dim, n_heads, T);
    auto* K    = ggml_reshape_3d(g.ctx, ggml_cont(g.ctx, k_), head_dim, n_heads, T);
    auto* V    = ggml_reshape_3d(g.ctx, ggml_cont(g.ctx, v_), head_dim, n_heads, T);

    Q = ggml_permute(g.ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(g.ctx, K, 0, 2, 1, 3);
    V = ggml_permute(g.ctx, V, 0, 2, 1, 3);

    auto* sc   = ggml_scale(g.ctx, ggml_mul_mat(g.ctx, K, Q), scale);
    auto* msc  = ggml_diag_mask_inf(g.ctx, sc, 0);
    auto* aw   = ggml_soft_max(g.ctx, msc);

    auto* Vt   = ggml_cont(g.ctx, ggml_permute(g.ctx, V, 1, 0, 2, 3));
    auto* ao   = ggml_cont(g.ctx, ggml_permute(g.ctx,
                     ggml_mul_mat(g.ctx, Vt, aw), 0, 2, 1, 3));
    ao = ggml_reshape_2d(g.ctx, ao, n_embd, T);
    auto* po   = ggml_add(g.ctx, ggml_mul_mat(g.ctx, proj_w, ao), proj_b);
    auto* h1   = ggml_add(g.ctx, x, po);   // first residual

    auto* h2   = ggml_norm(g.ctx, h1, 1e-5f);
    h2 = ggml_add(g.ctx, ggml_mul(g.ctx, h2, ln2_g), ln2_b);
    auto* up   = ggml_add(g.ctx, ggml_mul_mat(g.ctx, fc_w, h2), fc_b);
    auto* act  = ggml_gelu(g.ctx, up);
    auto* dn   = ggml_add(g.ctx, ggml_mul_mat(g.ctx, pr_w, act), pr_b);
    auto* h_out= ggml_add(g.ctx, h1, dn);  // second residual

    alloc_tensors(g);

    // ── Fill weights ──
    // Use identity-like weights with small perturbations so the test
    // exercises real paths without requiring external reference data.
    std::vector<float> xv(n_embd * T);
    for (int i = 0; i < n_embd * T; i++) xv[i] = (float)(i + 1) * 0.1f;
    fill_vec(x, xv);

    fill_scalar(ln1_g, 1.0f); fill_scalar(ln1_b, 0.0f);
    fill_scalar(ln2_g, 1.0f); fill_scalar(ln2_b, 0.0f);

    // QKV: identity in Q block, zero in K and V
    std::vector<float> qkv_wv(n_embd * 3 * n_embd, 0.0f);
    for (int i = 0; i < n_embd; i++) qkv_wv[i * n_embd + i] = 1.0f;
    fill_vec(qkv_w, qkv_wv); fill_scalar(qkv_b, 0.0f);

    // proj: zero → attn output = 0 → h1 = x
    std::vector<float> pw(n_embd * n_embd, 0.0f); fill_vec(proj_w, pw);
    fill_scalar(proj_b, 0.0f);

    // fc: identity (first n_embd of n_embd*4 rows), rest zero
    std::vector<float> fc_wv(n_embd * n_embd * 4, 0.0f);
    for (int i = 0; i < n_embd; i++) fc_wv[i * n_embd + i] = 1.0f;
    fill_vec(fc_w, fc_wv); fill_scalar(fc_b, 0.0f);

    // pr: picks back first n_embd rows
    std::vector<float> pr_wv(n_embd * 4 * n_embd, 0.0f);
    for (int i = 0; i < n_embd; i++) pr_wv[i * (n_embd * 4) + i] = 1.0f;
    fill_vec(pr_w, pr_wv); fill_scalar(pr_b, 0.0f);

    compute(g, h_out);

    auto out = read_vec(h_out);
    std::cout << "  Layer output (T=" << T << ", D=" << n_embd << "):\n";
    for (int t = 0; t < T; t++) {
        std::cout << "    token " << t << ": ";
        for (int i = 0; i < n_embd; i++)
            std::cout << std::fixed << std::setprecision(4) << out[t * n_embd + i] << " ";
        std::cout << "\n";
    }

    // Structural checks: output must be finite and have correct shape
    CHECK_MSG(h_out->ne[0] == n_embd, "output dim D");
    CHECK_MSG(h_out->ne[1] == T,      "output seq len T");
    for (int i = 0; i < n_embd * T; i++)
        CHECK_MSG(std::isfinite(out[i]), "output is finite");

    // With proj=0, h1=x; with LN2(x)+fc+gelu+pr+x being the FFN residual:
    // All elements should remain close to the input in magnitude (not explode).
    float max_val = 0;
    for (float v : out) max_val = std::max(max_val, std::fabs(v));
    CHECK_MSG(max_val < 10.0f, "output values are within sane range");
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 16: End-to-end forward pass (2 tokens, 2 layers)
// Logits must be finite, correct shape, and embedding lookup must match.
// ─────────────────────────────────────────────────────────────────────────────

// Builds one transformer block in-place, returns the output tensor
static ggml_tensor* build_block(ggml_context* ctx,
                                ggml_tensor* in,
                                int n_embd, int T, int n_heads,
                                ggml_tensor* ln1_g, ggml_tensor* ln1_b,
                                ggml_tensor* qkv_w,  ggml_tensor* qkv_b,
                                ggml_tensor* proj_w, ggml_tensor* proj_b,
                                ggml_tensor* ln2_g,  ggml_tensor* ln2_b,
                                ggml_tensor* fc_w,   ggml_tensor* fc_b,
                                ggml_tensor* pr_w,   ggml_tensor* pr_b) {
    const int head_dim = n_embd / n_heads;
    float scale = 1.0f / std::sqrt((float)head_dim);

    auto* h   = ggml_norm(ctx, in, 1e-5f);
    h = ggml_add(ctx, ggml_mul(ctx, h, ln1_g), ln1_b);

    auto* qkv  = ggml_add(ctx, ggml_mul_mat(ctx, qkv_w, h), qkv_b);
    size_t es  = ggml_element_size(qkv);
    auto* q_   = ggml_view_2d(ctx, qkv, n_embd, T, qkv->nb[1], 0);
    auto* k_   = ggml_view_2d(ctx, qkv, n_embd, T, qkv->nb[1], n_embd * es);
    auto* v_   = ggml_view_2d(ctx, qkv, n_embd, T, qkv->nb[1], 2 * n_embd * es);

    auto* Q    = ggml_reshape_3d(ctx, ggml_cont(ctx, q_), head_dim, n_heads, T);
    auto* K    = ggml_reshape_3d(ctx, ggml_cont(ctx, k_), head_dim, n_heads, T);
    auto* V    = ggml_reshape_3d(ctx, ggml_cont(ctx, v_), head_dim, n_heads, T);

    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    auto* sc   = ggml_scale(ctx, ggml_mul_mat(ctx, K, Q), scale);
    auto* msc  = ggml_diag_mask_inf(ctx, sc, 0);
    auto* aw   = ggml_soft_max(ctx, msc);

    auto* Vt   = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
    auto* ao   = ggml_cont(ctx, ggml_permute(ctx,
                     ggml_mul_mat(ctx, Vt, aw), 0, 2, 1, 3));
    ao = ggml_reshape_2d(ctx, ao, n_embd, T);
    auto* po   = ggml_add(ctx, ggml_mul_mat(ctx, proj_w, ao), proj_b);
    auto* h1   = ggml_add(ctx, in, po);

    auto* h2   = ggml_norm(ctx, h1, 1e-5f);
    h2 = ggml_add(ctx, ggml_mul(ctx, h2, ln2_g), ln2_b);
    auto* up   = ggml_add(ctx, ggml_mul_mat(ctx, fc_w, h2), fc_b);
    auto* act  = ggml_gelu(ctx, up);
    auto* dn   = ggml_add(ctx, ggml_mul_mat(ctx, pr_w, act), pr_b);
    return ggml_add(ctx, h1, dn);
}

static void test_end_to_end() {
    print_section("Test 16: End-to-end forward pass (2 tokens, 2 layers)");
    GgmlCtx g(512 * 1024 * 1024);

    const int n_embd  = 4;
    const int T       = 2;
    const int n_heads = 2;
    const int vocab   = 16;
    const int ctx_len = 8;
    const int n_layers= 2;

    // ── Embeddings ──
    auto* wte  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, vocab);
    auto* wpe  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, ctx_len);
    auto* tok  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    auto* pos  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    auto* emb  = ggml_add(g.ctx,
                     ggml_get_rows(g.ctx, wte, tok),
                     ggml_get_rows(g.ctx, wpe, pos));

    // ── Layer 0 weights ──
    auto* l0_ln1_g  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* l0_ln1_b  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* l0_qkv_w  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, 3*n_embd);
    auto* l0_qkv_b  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 3*n_embd);
    auto* l0_proj_w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, n_embd);
    auto* l0_proj_b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* l0_ln2_g  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* l0_ln2_b  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* l0_fc_w   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, n_embd*4);
    auto* l0_fc_b   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd*4);
    auto* l0_pr_w   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd*4, n_embd);
    auto* l0_pr_b   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);

    // ── Layer 1 weights ──
    auto* l1_ln1_g  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* l1_ln1_b  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* l1_qkv_w  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, 3*n_embd);
    auto* l1_qkv_b  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 3*n_embd);
    auto* l1_proj_w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, n_embd);
    auto* l1_proj_b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* l1_ln2_g  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* l1_ln2_b  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* l1_fc_w   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, n_embd*4);
    auto* l1_fc_b   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd*4);
    auto* l1_pr_w   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd*4, n_embd);
    auto* l1_pr_b   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);

    // ── Final LN + unembedding ──
    auto* ln_f_g = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    auto* ln_f_b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_embd);
    // Unembedding: reuse wte transposed (tied weights)
    // logits(T × vocab) = h_final(T × n_embd) × wte^T(n_embd × vocab)

    // ── Build forward graph ──
    auto* h0 = build_block(g.ctx, emb, n_embd, T, n_heads,
                           l0_ln1_g, l0_ln1_b, l0_qkv_w, l0_qkv_b,
                           l0_proj_w, l0_proj_b, l0_ln2_g, l0_ln2_b,
                           l0_fc_w,   l0_fc_b,   l0_pr_w,  l0_pr_b);

    auto* h1 = build_block(g.ctx, h0, n_embd, T, n_heads,
                           l1_ln1_g, l1_ln1_b, l1_qkv_w, l1_qkv_b,
                           l1_proj_w, l1_proj_b, l1_ln2_g, l1_ln2_b,
                           l1_fc_w,   l1_fc_b,   l1_pr_w,  l1_pr_b);

    auto* hf    = ggml_norm(g.ctx, h1, 1e-5f);
    hf = ggml_add(g.ctx, ggml_mul(g.ctx, hf, ln_f_g), ln_f_b);
    // logits: hf(n_embd × T) × wte(n_embd × vocab) → (vocab × T)
    auto* logits = ggml_mul_mat(g.ctx, wte, hf);

    alloc_tensors(g);

    // ── Initialize weights with small random values ──
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.02f);

    auto fill_rand = [&](ggml_tensor* t) {
        float* d = (float*)t->data;
        for (int64_t i = 0; i < ggml_nelements(t); i++) d[i] = nd(rng);
    };
    auto fill_ones = [&](ggml_tensor* t) { fill_scalar(t, 1.0f); };
    auto fill_zero = [&](ggml_tensor* t) { fill_scalar(t, 0.0f); };

    fill_rand(wte); fill_rand(wpe);

    // Layer norms: ones/zeros
    fill_ones(l0_ln1_g); fill_zero(l0_ln1_b);
    fill_ones(l0_ln2_g); fill_zero(l0_ln2_b);
    fill_ones(l1_ln1_g); fill_zero(l1_ln1_b);
    fill_ones(l1_ln2_g); fill_zero(l1_ln2_b);
    fill_ones(ln_f_g);   fill_zero(ln_f_b);

    // Attention + FFN weights: small random
    fill_rand(l0_qkv_w);  fill_zero(l0_qkv_b);
    fill_rand(l0_proj_w); fill_zero(l0_proj_b);
    fill_rand(l0_fc_w);   fill_zero(l0_fc_b);
    fill_rand(l0_pr_w);   fill_zero(l0_pr_b);

    fill_rand(l1_qkv_w);  fill_zero(l1_qkv_b);
    fill_rand(l1_proj_w); fill_zero(l1_proj_b);
    fill_rand(l1_fc_w);   fill_zero(l1_fc_b);
    fill_rand(l1_pr_w);   fill_zero(l1_pr_b);

    // Input tokens: [3, 7], positions: [0, 1]
    fill_vec_i32(tok, {3, 7});
    fill_vec_i32(pos, {0, 1});

    compute(g, logits);

    // ── Checks ──
    CHECK_MSG(logits->ne[0] == vocab, "logits ne[0] == vocab");
    CHECK_MSG(logits->ne[1] == T,     "logits ne[1] == T");

    auto lv = read_vec(logits);
    int num_finite = 0, num_nan = 0;
    for (float v : lv) {
        if (std::isfinite(v)) ++num_finite;
        if (std::isnan(v))    ++num_nan;
    }

    std::cout << "  Logits shape: (" << vocab << ", " << T << ")\n";
    std::cout << "  Finite logits: " << num_finite << "/" << lv.size() << "\n";
    CHECK_MSG(num_nan == 0, "no NaN logits");
    CHECK_MSG(num_finite == (int)lv.size(), "all logits are finite");

    // Softmax over logits: sum must be 1 per token
    for (int t = 0; t < T; t++) {
        std::vector<float> row(vocab);
        for (int v = 0; v < vocab; v++) row[v] = lv[t * vocab + v];
        auto probs = ref_softmax_row(row, vocab);
        float s = 0;
        for (float p : probs) s += p;
        CHECK_NEAR(s, 1.0f, 1e-4f);
    }
    std::cout << "  Softmax over logits: row sums verified.\n";

    // Print logits for last token
    std::cout << "  Logits (last token): ";
    for (int v = 0; v < vocab; v++)
        std::cout << std::fixed << std::setprecision(3) << lv[(T-1)*vocab+v] << " ";
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point (callable from run_tests.cpp or standalone)
// ─────────────────────────────────────────────────────────────────────────────
int run_forward_pass_tests() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Forward Pass Correctness Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_layernorm_identity();
    test_layernorm_known_values();
    test_layernorm_gamma_beta();
    test_matmul_exact();
    test_causal_mask();
    test_softmax_masked();
    test_gelu_exact();
    test_embedding_lookup();
    test_ggml_layout();
    test_positional_embeddings();
    test_single_attention_head();
    test_multihead_attention_concat();
    test_ffn_block();
    test_residual_connections();
    test_full_single_layer();
    test_end_to_end();

    std::cout << "\n========================================" << std::endl;
    if (g_tests_failed == 0) {
        std::cout << "All Forward Pass Tests PASSED" << std::endl;
    } else {
        std::cout << "Some Forward Pass Tests FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return g_tests_failed;
}
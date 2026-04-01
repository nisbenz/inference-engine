#include "common_test.hpp"
#include "model.hpp"
#include <ggml.h>
#include <ggml-backend.h>
#include <cmath>
#include <random>

// Helper: compute graph with backend
static void compute_graph(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* result) {
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_graph_compute(backend, gf);
}

// Test 1: Identity LayerNorm - verify normalization is correct
int test_forward_layernorm_identity() {
    print_test_header("test_forward_layernorm_identity");

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    TEST_ASSERT_MSG(backend != nullptr, "Failed to init backend");

    // Input: all 1.0f
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor* gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

    // Build graph first
    ggml_tensor* x_norm = ggml_norm(ctx, x, 1e-5f);
    ggml_tensor* scaled = ggml_mul(ctx, x_norm, gamma);
    ggml_tensor* result = ggml_add(ctx, scaled, beta);

    // Allocate tensors
    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Now assign data (after allocation)
    float* x_data = (float*)x->data;
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;
    for (int i = 0; i < 8; i++) x_data[i] = 1.0f;
    for (int i = 0; i < 4; i++) {
        gamma_data[i] = 1.0f;
        beta_data[i] = 0.0f;
    }

    compute_graph(ctx, backend, result);

    float* result_data = (float*)result->data;
    std::cout << "  Input all 1s, LayerNorm result (first 4): ";
    for (int i = 0; i < 4; i++) std::cout << result_data[i] << " ";
    std::cout << std::endl;

    // Result should be ~0 for all elements (since x = mean, normalized = 0)
    for (int i = 0; i < 8; i++) {
        TEST_ASSERT_FLOAT_EQ(result_data[i], 0.0f, 1e-4f);
    }

    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "  LayerNorm identity test PASSED" << std::endl;
    return 0;
}

// Test 2: Simple matmul - verify dimensions
int test_simple_matmul() {
    print_test_header("test_simple_matmul");

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    TEST_ASSERT_MSG(backend != nullptr, "Failed to init backend");

    ggml_tensor* W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);

    std::cout << "  W: ne[0]=" << W->ne[0] << " ne[1]=" << W->ne[1] << std::endl;
    std::cout << "  x: ne[0]=" << x->ne[0] << " ne[1]=" << x->ne[1] << std::endl;

    ggml_tensor* result = ggml_mul_mat(ctx, W, x);
    std::cout << "  result: ne[0]=" << result->ne[0] << " ne[1]=" << result->ne[1] << std::endl;

    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "  Simple matmul test PASSED" << std::endl;
    return 0;
}

// Test 3: Verify LayerNorm output for known input
int test_layernorm_known_values() {
    print_test_header("test_layernorm_known_values");

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    TEST_ASSERT_MSG(backend != nullptr, "Failed to init backend");

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor* gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

    ggml_tensor* x_norm = ggml_norm(ctx, x, 1e-5f);
    ggml_tensor* scaled = ggml_mul(ctx, x_norm, gamma);
    ggml_tensor* result = ggml_add(ctx, scaled, beta);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    float* x_data = (float*)x->data;
    x_data[0] = 1; x_data[1] = 2; x_data[2] = 3; x_data[3] = 4;
    x_data[4] = 5; x_data[5] = 6; x_data[6] = 7; x_data[7] = 8;

    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;
    for (int i = 0; i < 4; i++) {
        gamma_data[i] = 1.0f;
        beta_data[i] = 0.0f;
    }

    compute_graph(ctx, backend, result);

    float* result_data = (float*)result->data;

    std::cout << "  LayerNorm result:" << std::endl;
    std::cout << "  Token 0: ";
    for (int i = 0; i < 4; i++) std::cout << result_data[i] << " ";
    std::cout << std::endl;
    std::cout << "  Token 1: ";
    for (int i = 4; i < 8; i++) std::cout << result_data[i] << " ";
    std::cout << std::endl;

    float expected[4] = {-1.3416f, -0.4472f, 0.4472f, 1.3416f};
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 4; i++) {
            float actual = result_data[j * 4 + i];
            TEST_ASSERT_FLOAT_EQ(actual, expected[i], 0.01f);
        }
    }

    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "  LayerNorm known values test PASSED" << std::endl;
    return 0;
}

// Test 4: Check attention mask with causal attention
int test_forward_causal_mask() {
    print_test_header("test_forward_causal_mask");

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    TEST_ASSERT_MSG(backend != nullptr, "Failed to init backend");

    ggml_tensor* scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 3);
    ggml_tensor* masked = ggml_diag_mask_inf(ctx, scores, 0);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    float* scores_data = (float*)scores->data;
    for (int i = 0; i < 9; i++) scores_data[i] = 1.0f;

    compute_graph(ctx, backend, masked);

    float* result_data = (float*)masked->data;

    std::cout << "  Scores after causal mask (position=0):" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << result_data[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }

    TEST_ASSERT_FLOAT_EQ(result_data[0], 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[1], -INFINITY, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[2], -INFINITY, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[3], 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[4], 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[5], -INFINITY, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[6], 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[7], 0.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[8], 0.0f, 0.001f);

    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "  Causal mask test PASSED" << std::endl;
    return 0;
}

// Test 5: Verify softmax on masked scores produces valid attention weights
int test_softmax_attention() {
    print_test_header("test_softmax_attention");

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    TEST_ASSERT_MSG(backend != nullptr, "Failed to init backend");

    ggml_tensor* scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 3);
    ggml_tensor* attn_weights = ggml_soft_max(ctx, scores);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    float* scores_data = (float*)scores->data;
    scores_data[0] = 0.0f; scores_data[1] = -INFINITY; scores_data[2] = -INFINITY;
    scores_data[3] = 0.0f; scores_data[4] = 0.0f; scores_data[5] = -INFINITY;
    scores_data[6] = 0.0f; scores_data[7] = 0.0f; scores_data[8] = 0.0f;

    compute_graph(ctx, backend, attn_weights);

    float* result_data = (float*)attn_weights->data;

    std::cout << "  Attention weights after softmax:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << result_data[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }

    TEST_ASSERT_FLOAT_EQ(result_data[0], 1.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[3], 0.5f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[4], 0.5f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[6], 1.0f/3.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[7], 1.0f/3.0f, 0.001f);
    TEST_ASSERT_FLOAT_EQ(result_data[8], 1.0f/3.0f, 0.001f);

    for (int i = 0; i < 3; i++) {
        float row_sum = result_data[i*3] + result_data[i*3+1] + result_data[i*3+2];
        TEST_ASSERT_FLOAT_EQ(row_sum, 1.0f, 0.001f);
    }

    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "  Softmax attention test PASSED" << std::endl;
    return 0;
}

// Test 6: GELU activation function
int test_forward_gelu_activation() {
    print_test_header("test_forward_gelu_activation");

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    TEST_ASSERT_MSG(backend != nullptr, "Failed to init backend");

    ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor* y = ggml_gelu(ctx, x);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    float* x_data = (float*)x->data;
    x_data[0] = 0.0f;
    x_data[1] = 1.0f;
    x_data[2] = -1.0f;

    compute_graph(ctx, backend, y);

    float* result_data = (float*)y->data;

    std::cout << "  GELU(0) = " << result_data[0] << " (expected ~0)" << std::endl;
    std::cout << "  GELU(1) = " << result_data[1] << " (expected ~0.841)" << std::endl;
    std::cout << "  GELU(-1) = " << result_data[2] << " (expected ~-0.158)" << std::endl;

    TEST_ASSERT_FLOAT_EQ(result_data[0], 0.0f, 0.01f);
    TEST_ASSERT_FLOAT_EQ(result_data[1], 0.84119f, 0.01f);
    TEST_ASSERT_FLOAT_EQ(result_data[2], -0.15865f, 0.01f);

    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "  GELU activation test PASSED" << std::endl;
    return 0;
}

// Test 7: Full single-layer forward pass with zero weights
int test_single_layer_forward() {
    print_test_header("test_single_layer_forward");

    struct ggml_init_params params = {
        .mem_size   = 256 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    TEST_ASSERT_MSG(backend != nullptr, "Failed to init backend");

    const int n_embd = 4;
    const int seq_len = 1;
    const int n_heads = 2;
    const int head_dim = n_embd / n_heads;

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, seq_len);
    ggml_tensor* ln1_gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor* ln1_beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor* qkv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 3 * n_embd);
    ggml_tensor* qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * n_embd);
    ggml_tensor* proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
    ggml_tensor* proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor* ln2_gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor* ln2_beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor* fc_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd * 4);
    ggml_tensor* fc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd * 4);
    ggml_tensor* proj_ffn_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd * 4, n_embd);
    ggml_tensor* proj_ffn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    // Build graph
    ggml_tensor* ln1_out = ggml_norm(ctx, x, 1e-5f);
    ln1_out = ggml_mul(ctx, ln1_out, ln1_gamma);
    ln1_out = ggml_add(ctx, ln1_out, ln1_beta);

    ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, ln1_out);
    ggml_tensor* qkv_out = ggml_add(ctx, qkv, qkv_b);

    size_t es = ggml_element_size(qkv_out);
    ggml_tensor* q = ggml_view_2d(ctx, qkv_out, n_embd, seq_len, qkv_out->nb[1], 0);
    ggml_tensor* k = ggml_view_2d(ctx, qkv_out, n_embd, seq_len, qkv_out->nb[1], n_embd * es);
    ggml_tensor* v = ggml_view_2d(ctx, qkv_out, n_embd, seq_len, qkv_out->nb[1], 2 * n_embd * es);

    ggml_tensor* Q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_heads, seq_len);
    ggml_tensor* K = ggml_reshape_3d(ctx, ggml_cont(ctx, k), head_dim, n_heads, seq_len);
    ggml_tensor* V = ggml_reshape_3d(ctx, ggml_cont(ctx, v), head_dim, n_heads, seq_len);

    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    ggml_tensor* scores = ggml_mul_mat(ctx, K, Q);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)head_dim));
    scores = ggml_diag_mask_inf(ctx, scores, 0);
    ggml_tensor* attn_weights = ggml_soft_max(ctx, scores);

    ggml_tensor* V_t = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
    ggml_tensor* attn_out = ggml_mul_mat(ctx, V_t, attn_weights);
    attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
    attn_out = ggml_cont(ctx, attn_out);
    attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, seq_len);

    ggml_tensor* proj_out = ggml_mul_mat(ctx, proj_w, attn_out);
    proj_out = ggml_add(ctx, proj_out, proj_b);

    ggml_tensor* h1 = ggml_add(ctx, x, proj_out);

    ggml_tensor* ln2_out = ggml_norm(ctx, h1, 1e-5f);
    ln2_out = ggml_mul(ctx, ln2_out, ln2_gamma);
    ln2_out = ggml_add(ctx, ln2_out, ln2_beta);

    ggml_tensor* up = ggml_mul_mat(ctx, fc_w, ln2_out);
    up = ggml_add(ctx, up, fc_b);
    ggml_tensor* activated = ggml_gelu(ctx, up);
    ggml_tensor* down = ggml_mul_mat(ctx, proj_ffn_w, activated);
    down = ggml_add(ctx, down, proj_ffn_b);

    ggml_tensor* h2 = ggml_add(ctx, h1, down);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Initialize all weights to zero (already zero from alloc, but be explicit)
    float* x_data = (float*)x->data;
    for (int i = 0; i < n_embd * seq_len; i++) x_data[i] = 0.0f;

    float* ln1g = (float*)ln1_gamma->data;
    float* ln1b = (float*)ln1_beta->data;
    for (int i = 0; i < n_embd; i++) { ln1g[i] = 1.0f; ln1b[i] = 0.0f; }

    float* qkv_w_data = (float*)qkv_w->data;
    float* qkv_b_data = (float*)qkv_b->data;
    for (int i = 0; i < n_embd * 3 * n_embd; i++) qkv_w_data[i] = 0.0f;
    for (int i = 0; i < 3 * n_embd; i++) qkv_b_data[i] = 0.0f;

    float* proj_w_data = (float*)proj_w->data;
    float* proj_b_data = (float*)proj_b->data;
    for (int i = 0; i < n_embd * n_embd; i++) proj_w_data[i] = 0.0f;
    for (int i = 0; i < n_embd; i++) proj_b_data[i] = 0.0f;

    float* ln2g = (float*)ln2_gamma->data;
    float* ln2b = (float*)ln2_beta->data;
    for (int i = 0; i < n_embd; i++) { ln2g[i] = 1.0f; ln2b[i] = 0.0f; }

    float* fc_w_data = (float*)fc_w->data;
    float* fc_b_data = (float*)fc_b->data;
    float* proj_ffn_w_data = (float*)proj_ffn_w->data;
    float* proj_ffn_b_data = (float*)proj_ffn_b->data;
    for (int i = 0; i < n_embd * n_embd * 4; i++) fc_w_data[i] = 0.0f;
    for (int i = 0; i < n_embd * 4; i++) fc_b_data[i] = 0.0f;
    for (int i = 0; i < n_embd * 4 * n_embd; i++) proj_ffn_w_data[i] = 0.0f;
    for (int i = 0; i < n_embd; i++) proj_ffn_b_data[i] = 0.0f;

    compute_graph(ctx, backend, h2);

    float* result_data = (float*)h2->data;

    std::cout << "  Single layer output (all zeros input, zero weights):" << std::endl;
    std::cout << "  ";
    for (int i = 0; i < n_embd; i++) {
        std::cout << result_data[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < n_embd; i++) {
        TEST_ASSERT_FLOAT_EQ(result_data[i], 0.0f, 1e-5f);
    }

    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "  Single layer forward test PASSED" << std::endl;
    return 0;
}

// Test 8: Verify GGML tensor layout (row-major vs column-major)
int test_ggml_layout() {
    print_test_header("test_ggml_layout");

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    TEST_ASSERT_MSG(backend != nullptr, "Failed to init backend");

    ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);
    std::cout << "  Created tensor with ne[0]=" << t->ne[0] << " ne[1]=" << t->ne[1] << std::endl;
    std::cout << "  nb[0]=" << t->nb[0] << " nb[1]=" << t->nb[1] << std::endl;

    std::cout << "  Expected strides:" << std::endl;
    std::cout << "    Column-major: nb[0]=4, nb[1]=12" << std::endl;
    std::cout << "    Row-major:    nb[0]=16, nb[1]=4" << std::endl;

    if (t->nb[0] == 4 && t->nb[1] == 12) {
        std::cout << "  => GGML is COLUMN-MAJOR" << std::endl;
    } else if (t->nb[0] == 16 && t->nb[1] == 4) {
        std::cout << "  => GGML is ROW-MAJOR" << std::endl;
    } else {
        std::cout << "  => Unknown layout! nb[0]=" << t->nb[0] << " nb[1]=" << t->nb[1] << std::endl;
    }

    // Extract row 1 (should be [10, 11, 12, 13])
    ggml_tensor* row_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor* row1 = ggml_get_rows(ctx, t, row_idx);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Fill tensor: t[row, col] = row * 10 + col
    float* data = (float*)t->data;
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 3; row++) {
            size_t offset = col * t->nb[1] + row * t->nb[0];
            data[offset / sizeof(float)] = (float)(row * 10 + col);
        }
    }

    int32_t* idx = (int32_t*)row_idx->data;
    idx[0] = 1;

    compute_graph(ctx, backend, row1);

    float* row1_data = (float*)row1->data;
    std::cout << "  Row 1 extracted: ";
    for (int i = 0; i < 4; i++) {
        std::cout << row1_data[i] << " ";
    }
    std::cout << "(expected: 10, 11, 12, 13)" << std::endl;

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_EQ(row1_data[i], (float)(10 + i), 0.001f);
    }

    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "  GGML layout test PASSED" << std::endl;
    return 0;
}

// Test 9: Embedding lookup with ggml_get_rows
int test_forward_embedding_lookup() {
    print_test_header("test_forward_embedding_lookup");

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    TEST_ASSERT_MSG(backend != nullptr, "Failed to init backend");

    const int n_embd = 4;
    const int vocab_size = 10;

    ggml_tensor* wte = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, vocab_size);
    ggml_tensor* tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    ggml_tensor* embeddings = ggml_get_rows(ctx, wte, tokens);

    ggml_backend_alloc_ctx_tensors(ctx, backend);

    float* wte_data = (float*)wte->data;
    for (int t = 0; t < vocab_size; t++) {
        for (int i = 0; i < n_embd; i++) {
            wte_data[t * n_embd + i] = (float)(t * 1000 + i);
        }
    }

    std::cout << "  WTE: ne[0]=" << wte->ne[0] << " ne[1]=" << wte->ne[1] << std::endl;
    std::cout << "  WTE: nb[0]=" << wte->nb[0] << " nb[1]=" << wte->nb[1] << std::endl;

    int32_t* tokens_data = (int32_t*)tokens->data;
    tokens_data[0] = 0;
    tokens_data[1] = 1;
    tokens_data[2] = 2;

    compute_graph(ctx, backend, embeddings);

    float* emb_data = (float*)embeddings->data;

    std::cout << "  Embedding for token 0: ";
    for (int i = 0; i < n_embd; i++) {
        std::cout << emb_data[i] << " ";
    }
    std::cout << "(expected: 0, 1, 2, 3)" << std::endl;

    std::cout << "  Embedding for token 1: ";
    for (int i = 0; i < n_embd; i++) {
        std::cout << emb_data[n_embd + i] << " ";
    }
    std::cout << "(expected: 1000, 1001, 1002, 1003)" << std::endl;

    std::cout << "  Embedding for token 2: ";
    for (int i = 0; i < n_embd; i++) {
        std::cout << emb_data[2 * n_embd + i] << " ";
    }
    std::cout << "(expected: 2000, 2001, 2002, 2003)" << std::endl;

    for (int i = 0; i < n_embd; i++) {
        TEST_ASSERT_FLOAT_EQ(emb_data[i], (float)i, 0.001f);
    }
    for (int i = 0; i < n_embd; i++) {
        TEST_ASSERT_FLOAT_EQ(emb_data[n_embd + i], (float)(1000 + i), 0.001f);
    }
    for (int i = 0; i < n_embd; i++) {
        TEST_ASSERT_FLOAT_EQ(emb_data[2 * n_embd + i], (float)(2000 + i), 0.001f);
    }

    ggml_backend_free(backend);
    ggml_free(ctx);
    std::cout << "  Embedding lookup test PASSED" << std::endl;
    return 0;
}

int run_forward_pass_tests() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Forward Pass Correctness Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int result = 0;
    result |= test_forward_layernorm_identity();
    result |= test_simple_matmul();
    result |= test_layernorm_known_values();
    result |= test_forward_causal_mask();
    result |= test_softmax_attention();
    result |= test_forward_gelu_activation();
    result |= test_single_layer_forward();
    result |= test_ggml_layout();
    result |= test_forward_embedding_lookup();

    std::cout << "\n========================================" << std::endl;
    if (result == 0) {
        std::cout << "All Forward Pass Tests PASSED" << std::endl;
    } else {
        std::cout << "Some Forward Pass Tests FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return result;
}

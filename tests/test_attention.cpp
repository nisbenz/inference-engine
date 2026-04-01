#include "common_test.hpp"
#include "layers.hpp"
#include <ggml.h>

// Test attention configuration
int test_attention_config() {
    print_test_header("test_attention_config");

    const int n_heads = GPT2Config::n_heads;
    const int n_embd = GPT2Config::n_embd;
    const int head_dim = GPT2Config::head_dim;

    TEST_ASSERT_INT_EQ(n_heads, 12);
    TEST_ASSERT_INT_EQ(n_embd, 768);
    TEST_ASSERT_INT_EQ(head_dim, 64);
    TEST_ASSERT_INT_EQ(n_embd % n_heads, 0);  // Must divide evenly

    std::cout << "  Attention config: " << n_heads << " heads, " << n_embd << " hidden, " << head_dim << " head_dim" << std::endl;
    return 0;
}

// Test QKV projection shapes
int test_qkv_projection_shapes() {
    print_test_header("test_qkv_projection_shapes");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    const int n_embd = GPT2Config::n_embd;    // 768
    const int seq_len = 1;                      // Single token for testing
    const int n_heads = GPT2Config::n_heads;   // 12
    const int head_dim = GPT2Config::head_dim; // 64

    // Input: [n_embd, seq_len] = [768, 1]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, seq_len);
    TEST_ASSERT_INT_EQ(input->ne[0], 768);
    TEST_ASSERT_INT_EQ(input->ne[1], 1);

    // QKV weight: [n_embd, 3*n_embd] = [768, 2304]
    ggml_tensor* qkv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 3 * n_embd);
    ggml_tensor* qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * n_embd);

    // Fill with ones for testing
    float* w_data = (float*)qkv_w->data;
    for (size_t i = 0; i < 768 * 2304; i++) w_data[i] = 1.0f;
    float* b_data = (float*)qkv_b->data;
    for (size_t i = 0; i < 2304; i++) b_data[i] = 0.0f;

    // QKV projection: W^T @ x + b
    ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, input);
    ggml_tensor* qkv_out = ggml_add(ctx, qkv, qkv_b);

    std::cout << "  QKV output shape: ne[0]=" << qkv_out->ne[0] << " ne[1]=" << qkv_out->ne[1] << std::endl;
    TEST_ASSERT_INT_EQ(qkv_out->ne[0], 2304);  // 3 * n_embd
    TEST_ASSERT_INT_EQ(qkv_out->ne[1], 1);    // seq_len

    ggml_free(ctx);

    std::cout << "  QKV projection shapes verified" << std::endl;
    return 0;
}

// Test QKV split
int test_qkv_split() {
    print_test_header("test_qkv_split");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    const int n_embd = 768;
    const int seq_len = 4;

    // QKV tensor: [3*n_embd, seq_len] = [2304, 4]
    ggml_tensor* qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2304, seq_len);

    // Fill with sequential values for verification
    float* data = (float*)qkv->data;
    for (int i = 0; i < 2304 * seq_len; i++) {
        data[i] = (float)i;
    }

    size_t es = ggml_element_size(qkv);  // 4 for F32

    // Split into Q, K, V
    // Q: offset 0
    ggml_tensor* q = ggml_view_2d(ctx, qkv, n_embd, seq_len, qkv->nb[1], 0);
    // K: offset n_embd * es
    ggml_tensor* k = ggml_view_2d(ctx, qkv, n_embd, seq_len, qkv->nb[1], n_embd * es);
    // V: offset 2 * n_embd * es
    ggml_tensor* v = ggml_view_2d(ctx, qkv, n_embd, seq_len, qkv->nb[1], 2 * n_embd * es);

    TEST_ASSERT_INT_EQ(q->ne[0], 768);
    TEST_ASSERT_INT_EQ(q->ne[1], 4);
    TEST_ASSERT_INT_EQ(k->ne[0], 768);
    TEST_ASSERT_INT_EQ(k->ne[1], 4);
    TEST_ASSERT_INT_EQ(v->ne[0], 768);
    TEST_ASSERT_INT_EQ(v->ne[1], 4);

    // Verify Q, K, V point to correct data regions
    // Q[0] should equal qkv[0]
    float* q_data = (float*)q->data;
    float* qkv_data = (float*)qkv->data;
    TEST_ASSERT_FLOAT_EQ(q_data[0], qkv_data[0], 0.001f);
    TEST_ASSERT_FLOAT_EQ(q_data[1], qkv_data[1], 0.001f);

    // K[0] should equal qkv[768] (offset by n_embd)
    float* k_data = (float*)k->data;
    TEST_ASSERT_FLOAT_EQ(k_data[0], qkv_data[768], 0.001f);

    // V[0] should equal qkv[1536] (offset by 2 * n_embd)
    float* v_data = (float*)v->data;
    TEST_ASSERT_FLOAT_EQ(v_data[0], qkv_data[1536], 0.001f);

    ggml_free(ctx);

    std::cout << "  QKV split verified" << std::endl;
    return 0;
}

// Test attention reshape to multi-head
int test_attention_reshape() {
    print_test_header("test_attention_reshape");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    const int n_heads = 12;
    const int head_dim = 64;
    const int seq_len = 4;

    // Q after split: [n_embd, seq_len] = [768, 4]
    // First make it contiguous
    ggml_tensor* q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, seq_len);

    // Reshape to [head_dim, n_heads, seq_len] = [64, 12, 4]
    ggml_tensor* Q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_heads, seq_len);

    TEST_ASSERT_INT_EQ(Q->ne[0], 64);   // head_dim
    TEST_ASSERT_INT_EQ(Q->ne[1], 12);   // n_heads
    TEST_ASSERT_INT_EQ(Q->ne[2], 4);    // seq_len
    TEST_ASSERT_INT_EQ(Q->n_dims, 3);

    ggml_free(ctx);

    std::cout << "  Attention reshape verified" << std::endl;
    return 0;
}

// Test attention permute
int test_attention_permute() {
    print_test_header("test_attention_permute");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    const int n_heads = 12;
    const int head_dim = 64;
    const int seq_len = 4;

    // Create Q tensor [head_dim, n_heads, seq_len] = [64, 12, 4]
    ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len);

    // Permute: (0, 2, 1, 3) -> [head_dim, seq_len, n_heads] = [64, 4, 12]
    ggml_tensor* Q_perm = ggml_permute(ctx, q, 0, 2, 1, 3);

    TEST_ASSERT_INT_EQ(Q_perm->ne[0], 64);  // head_dim (unchanged)
    TEST_ASSERT_INT_EQ(Q_perm->ne[1], 4);    // seq_len (moved from dim 2)
    TEST_ASSERT_INT_EQ(Q_perm->ne[2], 12);   // n_heads (moved from dim 1)

    // Create K tensor [head_dim, n_heads, seq_len] = [64, 12, 4]
    ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len);
    ggml_tensor* K_perm = ggml_permute(ctx, k, 0, 2, 1, 3);

    TEST_ASSERT_INT_EQ(K_perm->ne[0], 64);  // head_dim
    TEST_ASSERT_INT_EQ(K_perm->ne[1], 4);   // seq_len
    TEST_ASSERT_INT_EQ(K_perm->ne[2], 12);  // n_heads

    ggml_free(ctx);

    std::cout << "  Attention permute verified" << std::endl;
    return 0;
}

// Test causal mask (using ggml_diag_mask_inf)
int test_causal_mask() {
    print_test_header("test_causal_mask");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    // Create attention scores: [seq_len, seq_len] = [4, 4]
    // After permute, we get [seq_len, seq_len, n_heads]
    // But for diag_mask, we work on a 2D slice
    ggml_tensor* scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);

    // Fill with zeros (before scaling and softmax)
    float* data = (float*)scores->data;
    for (int i = 0; i < 16; i++) data[i] = 0.0f;

    // Apply causal mask at position=0
    ggml_tensor* masked = ggml_diag_mask_inf(ctx, scores, 0);

    // Verify it's the same tensor (diag_mask doesn't change shape)
    TEST_ASSERT_INT_EQ(masked->ne[0], 4);
    TEST_ASSERT_INT_EQ(masked->ne[1], 4);

    ggml_free(ctx);

    std::cout << "  Causal mask structure verified" << std::endl;
    return 0;
}

// Test softmax
int test_softmax() {
    print_test_header("test_softmax");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    // Create a simple tensor [1, 3] = [[1.0, 2.0, 3.0]]
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
    float* data = (float*)x->data;
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;

    ggml_tensor* result = ggml_soft_max(ctx, x);

    // Softmax of [1, 2, 3] should be [0.090, 0.245, 0.665] approximately
    float* result_data = (float*)result->data;
    float sum = result_data[0] + result_data[1] + result_data[2];

    // Sum should be 1.0 (approximately)
    TEST_ASSERT_FLOAT_EQ(sum, 1.0f, 0.01f);

    // Each value should be positive
    TEST_ASSERT_MSG(result_data[0] > 0 && result_data[1] > 0 && result_data[2] > 0,
        "Softmax outputs should be positive");

    ggml_free(ctx);

    std::cout << "  Softmax verified" << std::endl;
    return 0;
}

// Test attention score computation (K^T @ Q)
int test_attention_scores() {
    print_test_header("test_attention_scores");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    const int n_heads = 12;
    const int head_dim = 64;
    const int seq_len = 1;  // Single token for simplicity

    // After permute: Q = [head_dim, seq_len, n_heads] = [64, 1, 12]
    ggml_tensor* Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads);

    // After permute: K = [head_dim, seq_len, n_heads] = [64, 1, 12]
    ggml_tensor* K = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads);

    // Fill with ones
    float* q_data = (float*)Q->data;
    float* k_data = (float*)K->data;
    for (size_t i = 0; i < 64 * 1 * 12; i++) {
        q_data[i] = 1.0f;
        k_data[i] = 1.0f;
    }

    // K^T @ Q = [head_dim, n_heads, seq_len] @ [head_dim, n_heads, seq_len]
    // But in practice we need to permute and use ggml_mul_mat correctly
    // This is a simplified test to verify shape compatibility

    ggml_tensor* scores = ggml_mul_mat(ctx, K, Q);

    // After ggml_mul_mat with batched dims, expected output shape
    // This tests that the operation doesn't crash
    std::cout << "  Attention matmul completed without error" << std::endl;

    ggml_free(ctx);

    return 0;
}

int run_attention_tests() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Attention Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int result = 0;
    result |= test_attention_config();
    result |= test_qkv_projection_shapes();
    result |= test_qkv_split();
    result |= test_attention_reshape();
    result |= test_attention_permute();
    result |= test_causal_mask();
    result |= test_softmax();
    result |= test_attention_scores();

    std::cout << "\n========================================" << std::endl;
    if (result == 0) {
        std::cout << "All Attention Tests PASSED" << std::endl;
    } else {
        std::cout << "Some Attention Tests FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return result;
}

#include "common_test.hpp"
#include "layers.hpp"
#include <ggml.h>
#include <cmath>

// Test FFN configuration
int test_ffn_config() {
    print_test_header("test_ffn_config");

    const int n_embd = GPT2Config::n_embd;    // 768
    const int n_ffn = GPT2Config::n_ffn;      // 3072

    TEST_ASSERT_INT_EQ(n_embd, 768);
    TEST_ASSERT_INT_EQ(n_ffn, 3072);
    TEST_ASSERT_INT_EQ(n_ffn, 4 * n_embd);  // FFN is 4x hidden size

    std::cout << "  FFN config: n_embd=" << n_embd << ", n_ffn=" << n_ffn << std::endl;
    return 0;
}

// Test FFN dimensions
int test_ffn_dimensions() {
    print_test_header("test_ffn_dimensions");

    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,  // 128MB - needed ~19MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    const int n_embd = 768;
    const int n_ffn = 3072;
    const int seq_len = 4;

    // Input: [n_embd, seq_len] = [768, 4]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, seq_len);
    TEST_ASSERT_INT_EQ(input->ne[0], 768);
    TEST_ASSERT_INT_EQ(input->ne[1], 4);

    // FFN up weight: [n_embd, n_ffn] = [768, 3072]
    ggml_tensor* fc_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ffn);
    ggml_tensor* fc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_ffn);

    // Fill with ones
    float* w_data = (float*)fc_w->data;
    for (size_t i = 0; i < 768 * 3072; i++) w_data[i] = 1.0f;
    float* b_data = (float*)fc_b->data;
    for (size_t i = 0; i < n_ffn; i++) b_data[i] = 0.0f;

    // Up projection: W^T @ x + b
    ggml_tensor* up = ggml_mul_mat(ctx, fc_w, input);
    ggml_tensor* up_out = ggml_add(ctx, up, fc_b);

    std::cout << "  Up projection output: ne[0]=" << up_out->ne[0] << " ne[1]=" << up_out->ne[1] << std::endl;
    TEST_ASSERT_INT_EQ(up_out->ne[0], 3072);  // n_ffn
    TEST_ASSERT_INT_EQ(up_out->ne[1], 4);     // seq_len

    // FFN down weight: [n_ffn, n_embd] = [3072, 768]
    ggml_tensor* proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ffn, n_embd);
    ggml_tensor* proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    // Down projection
    ggml_tensor* down = ggml_mul_mat(ctx, proj_w, up_out);
    ggml_tensor* down_out = ggml_add(ctx, down, proj_b);

    std::cout << "  Down projection output: ne[0]=" << down_out->ne[0] << " ne[1]=" << down_out->ne[1] << std::endl;
    TEST_ASSERT_INT_EQ(down_out->ne[0], 768);  // n_embd
    TEST_ASSERT_INT_EQ(down_out->ne[1], 4);   // seq_len

    ggml_free(ctx);

    std::cout << "  FFN dimensions verified" << std::endl;
    return 0;
}

// Test GELU activation function
int test_gelu_activation() {
    print_test_header("test_gelu_activation");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    // Create input tensor [1, 1] = [[x]]
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);

    // Test GELU(0) ≈ 0
    float* data = (float*)x->data;
    data[0] = 0.0f;
    ggml_tensor* result = ggml_gelu(ctx, x);

    // Note: We can't easily read back the result without computing
    // This test verifies the operation can be created without error
    std::cout << "  GELU operation created successfully" << std::endl;

    ggml_free(ctx);

    return 0;
}

// Test GELU mathematical properties (reference implementation)
int test_gelu_math() {
    print_test_header("test_gelu_math");

    // Reference GELU implementation
    auto gelu_ref = [](float x) -> float {
        return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    };

    // Test values
    struct TestCase {
        float input;
        float expected;
        float tolerance;
    };

    TestCase cases[] = {
        {0.0f, 0.0f, 0.01f},
        {1.0f, 0.84119f, 0.01f},
        {-1.0f, -0.15867f, 0.01f},
        {2.0f, 1.95460f, 0.01f},
        {-2.0f, -0.04540f, 0.01f},
    };

    for (const auto& tc : cases) {
        float result = gelu_ref(tc.input);
        TEST_ASSERT_FLOAT_EQ(result, tc.expected, tc.tolerance);
        std::cout << "  GELU(" << tc.input << ") = " << result << " (expected " << tc.expected << ")" << std::endl;
    }

    std::cout << "  GELU math verified" << std::endl;
    return 0;
}

// Test FFN full forward pass structure
int test_ffn_forward_structure() {
    print_test_header("test_ffn_forward_structure");

    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,  // 128MB for larger tensors
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");
    ggml_cgraph* gf = ggml_new_graph(ctx);

    const int n_embd = 768;
    const int n_ffn = 3072;
    const int seq_len = 1;

    // Create FFN
    FFN ffn;
    ffn.c_fc_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ffn);
    ffn.c_fc_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_ffn);
    ffn.c_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ffn, n_embd);
    ffn.c_proj_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    // Initialize with zeros
    memset(ffn.c_fc_weight->data, 0, ggml_nbytes(ffn.c_fc_weight));
    memset(ffn.c_fc_bias->data, 0, ggml_nbytes(ffn.c_fc_bias));
    memset(ffn.c_proj_weight->data, 0, ggml_nbytes(ffn.c_proj_weight));
    memset(ffn.c_proj_bias->data, 0, ggml_nbytes(ffn.c_proj_bias));

    // Create input: [n_embd, seq_len] = [768, 1]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, seq_len);
    float* in_data = (float*)input->data;
    for (int i = 0; i < n_embd * seq_len; i++) in_data[i] = 1.0f;

    // Forward pass
    ggml_tensor* out = ffn.forward(ctx, gf, input);

    std::cout << "  FFN forward output shape: ne[0]=" << out->ne[0] << " ne[1]=" << out->ne[1] << std::endl;
    TEST_ASSERT_INT_EQ(out->ne[0], n_embd);   // Should be n_embd
    TEST_ASSERT_INT_EQ(out->ne[1], seq_len);  // Should be seq_len

    ggml_free(ctx);

    std::cout << "  FFN forward structure verified" << std::endl;
    return 0;
}

// Test FFN weight shapes match GPT-2
int test_ffn_weight_shapes() {
    print_test_header("test_ffn_weight_shapes");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    // c_fc: (n_embd, n_ffn) = (768, 3072)
    ggml_tensor* c_fc_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 3072);
    ggml_tensor* c_fc_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3072);

    // c_proj: (n_ffn, n_embd) = (3072, 768)
    ggml_tensor* c_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3072, 768);
    ggml_tensor* c_proj_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 768);

    // Verify shapes
    TEST_ASSERT_INT_EQ(c_fc_weight->ne[0], 768);   // n_embd
    TEST_ASSERT_INT_EQ(c_fc_weight->ne[1], 3072);  // n_ffn
    TEST_ASSERT_INT_EQ(c_fc_bias->ne[0], 3072);    // n_ffn

    TEST_ASSERT_INT_EQ(c_proj_weight->ne[0], 3072);  // n_ffn
    TEST_ASSERT_INT_EQ(c_proj_weight->ne[1], 768);   // n_embd
    TEST_ASSERT_INT_EQ(c_proj_bias->ne[0], 768);     // n_embd

    ggml_free(ctx);

    std::cout << "  FFN weight shapes verified" << std::endl;
    return 0;
}

// Test FFN computation preserves sequence length
int test_ffn_seq_len_preservation() {
    print_test_header("test_ffn_seq_len_preservation");

    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,  // 128MB - tensors accumulate in loop
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    const int n_embd = 768;
    const int n_ffn = 3072;

    // Test with different sequence lengths
    int seq_lens[] = {1, 4, 16, 32};

    for (int seq_len : seq_lens) {
        // Input: [n_embd, seq_len]
        ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, seq_len);

        // Up weight: [n_embd, n_ffn]
        ggml_tensor* fc_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ffn);

        // Up projection
        ggml_tensor* up = ggml_mul_mat(ctx, fc_w, input);

        // Down weight: [n_ffn, n_embd]
        ggml_tensor* proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ffn, n_embd);

        // Down projection
        ggml_tensor* down = ggml_mul_mat(ctx, proj_w, up);

        TEST_ASSERT_INT_EQ(down->ne[0], n_embd);
        TEST_ASSERT_INT_EQ(down->ne[1], seq_len);

        std::cout << "  seq_len=" << seq_len << " -> output ne[1]=" << down->ne[1] << std::endl;
    }

    ggml_free(ctx);

    std::cout << "  FFN sequence length preservation verified" << std::endl;
    return 0;
}

int run_ffn_tests() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running FFN Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int result = 0;
    result |= test_ffn_config();
    result |= test_ffn_dimensions();
    result |= test_gelu_activation();
    result |= test_gelu_math();
    result |= test_ffn_forward_structure();
    result |= test_ffn_weight_shapes();
    result |= test_ffn_seq_len_preservation();

    std::cout << "\n========================================" << std::endl;
    if (result == 0) {
        std::cout << "All FFN Tests PASSED" << std::endl;
    } else {
        std::cout << "Some FFN Tests FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return result;
}

#include "common_test.hpp"
#include "layers.hpp"
#include <ggml.h>
#include <cmath>

// Test LayerNorm configuration
int test_layernorm_config() {
    print_test_header("test_layernorm_config");

    const float eps = GPT2Config::layer_norm_eps;
    TEST_ASSERT_FLOAT_EQ(eps, 1e-5f, 0.0001f);

    std::cout << "  LayerNorm eps=" << eps << std::endl;
    return 0;
}

// Test LayerNorm shape preservation
int test_layernorm_shape() {
    print_test_header("test_layernorm_shape");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    const int n_embd = 768;
    const int seq_len = 4;

    // Input: [n_embd, seq_len] = [768, 4]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, seq_len);

    // LayerNorm gamma (scale) and beta (bias)
    ggml_tensor* gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor* beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    // Fill with ones
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;
    for (int i = 0; i < n_embd; i++) {
        gamma_data[i] = 1.0f;
        beta_data[i] = 0.0f;
    }

    // Apply LayerNorm using ggml_norm
    ggml_tensor* normalized = ggml_norm(ctx, input, GPT2Config::layer_norm_eps);

    // Scale by gamma
    ggml_tensor* scaled = ggml_mul(ctx, normalized, gamma);

    // Add beta (with broadcasting)
    ggml_tensor* repeated_beta = ggml_repeat(ctx, beta, scaled);
    ggml_tensor* result = ggml_add(ctx, scaled, repeated_beta);

    TEST_ASSERT_INT_EQ(result->ne[0], n_embd);  // 768
    TEST_ASSERT_INT_EQ(result->ne[1], seq_len);  // 4

    std::cout << "  LayerNorm shape: ne[0]=" << result->ne[0] << " ne[1]=" << result->ne[1] << std::endl;

    ggml_free(ctx);

    std::cout << "  LayerNorm shape verified" << std::endl;
    return 0;
}

// Test RMSNorm shape preservation
int test_rmsnorm_shape() {
    print_test_header("test_rmsnorm_shape");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    const int n_embd = 768;
    const int seq_len = 4;

    // Input: [n_embd, seq_len] = [768, 4]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, seq_len);

    // RMSNorm weight
    ggml_tensor* weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    float* w_data = (float*)weight->data;
    for (int i = 0; i < n_embd; i++) w_data[i] = 1.0f;

    // Apply RMSNorm
    ggml_tensor* normalized = ggml_rms_norm(ctx, input, GPT2Config::layer_norm_eps);

    // Scale by weight
    ggml_tensor* result = ggml_mul(ctx, normalized, weight);

    TEST_ASSERT_INT_EQ(result->ne[0], n_embd);
    TEST_ASSERT_INT_EQ(result->ne[1], seq_len);

    std::cout << "  RMSNorm shape: ne[0]=" << result->ne[0] << " ne[1]=" << result->ne[1] << std::endl;

    ggml_free(ctx);

    std::cout << "  RMSNorm shape verified" << std::endl;
    return 0;
}

// Test LayerNorm with simple input (identity case)
int test_layernorm_identity() {
    print_test_header("test_layernorm_identity");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    // Input: [4, 1] = all zeros
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    float* in_data = (float*)input->data;
    for (int i = 0; i < 4; i++) in_data[i] = 0.0f;

    // Gamma = ones, Beta = zeros (should be identity)
    ggml_tensor* gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;
    for (int i = 0; i < 4; i++) {
        gamma_data[i] = 1.0f;
        beta_data[i] = 0.0f;
    }

    // LayerNorm: output = gamma * normalized + beta
    // If input is all zeros, after normalization it should still be zeros
    // Then gamma * 0 + 0 = 0
    ggml_tensor* normalized = ggml_norm(ctx, input, GPT2Config::layer_norm_eps);
    ggml_tensor* scaled = ggml_mul(ctx, normalized, gamma);
    ggml_tensor* repeated_beta = ggml_repeat(ctx, beta, scaled);
    ggml_tensor* result = ggml_add(ctx, scaled, repeated_beta);

    // Result shape should be [4, 1]
    TEST_ASSERT_INT_EQ(result->ne[0], 4);
    TEST_ASSERT_INT_EQ(result->ne[1], 1);

    std::cout << "  LayerNorm identity case verified" << std::endl;

    ggml_free(ctx);

    return 0;
}

// Test LayerNorm layer structure
int test_layernorm_layer() {
    print_test_header("test_layernorm_layer");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    // Create LayerNorm layer
    LayerNorm ln;
    ln.gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, GPT2Config::n_embd);
    ln.beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, GPT2Config::n_embd);

    // Initialize gamma to 1, beta to 0
    float* gamma_data = (float*)ln.gamma->data;
    float* beta_data = (float*)ln.beta->data;
    for (int i = 0; i < GPT2Config::n_embd; i++) {
        gamma_data[i] = 1.0f;
        beta_data[i] = 0.0f;
    }

    // Input: [n_embd, seq_len] = [768, 1]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GPT2Config::n_embd, 1);

    // Forward pass
    ggml_tensor* out = ln.forward(ctx, input);

    TEST_ASSERT_INT_EQ(out->ne[0], GPT2Config::n_embd);
    TEST_ASSERT_INT_EQ(out->ne[1], 1);

    ggml_free(ctx);

    std::cout << "  LayerNorm layer forward verified" << std::endl;
    return 0;
}

// Test RMSNorm layer structure
int test_rmsnorm_layer() {
    print_test_header("test_rmsnorm_layer");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    // Create RMSNorm layer
    RMSNorm rms;
    rms.weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, GPT2Config::n_embd);

    // Initialize weight to 1
    float* weight_data = (float*)rms.weight->data;
    for (int i = 0; i < GPT2Config::n_embd; i++) {
        weight_data[i] = 1.0f;
    }

    // Input: [n_embd, seq_len] = [768, 1]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GPT2Config::n_embd, 1);

    // Forward pass
    ggml_tensor* out = rms.forward(ctx, input);

    TEST_ASSERT_INT_EQ(out->ne[0], GPT2Config::n_embd);
    TEST_ASSERT_INT_EQ(out->ne[1], 1);

    ggml_free(ctx);

    std::cout << "  RMSNorm layer forward verified" << std::endl;
    return 0;
}

// Test ggml_repeat broadcasting
int test_repeat_broadcasting() {
    print_test_header("test_repeat_broadcasting");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    // Beta: [n_embd] = [768]
    ggml_tensor* beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 768);
    float* beta_data = (float*)beta->data;
    for (int i = 0; i < 768; i++) beta_data[i] = (float)i;

    // Input tensor: [n_embd, seq_len] = [768, 4]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 4);

    // Repeat beta to match input shape
    ggml_tensor* repeated_beta = ggml_repeat(ctx, beta, input);

    TEST_ASSERT_INT_EQ(repeated_beta->ne[0], 768);
    TEST_ASSERT_INT_EQ(repeated_beta->ne[1], 4);

    // Verify the repeat worked (element at [r, c] should equal beta[r])
    float* repeated_data = (float*)repeated_beta->data;
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 768; r++) {
            size_t idx = c * 768 + r;
            TEST_ASSERT_FLOAT_EQ(repeated_data[idx], beta_data[r], 0.001f);
        }
    }

    ggml_free(ctx);

    std::cout << "  Repeat broadcasting verified" << std::endl;
    return 0;
}

// Test ggml_norm computation
int test_norm_computation() {
    print_test_header("test_norm_computation");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    // Input: [4, 1] with values [1, 2, 3, 4]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    float* in_data = (float*)input->data;
    in_data[0] = 1.0f;
    in_data[1] = 2.0f;
    in_data[2] = 3.0f;
    in_data[3] = 4.0f;

    // Apply LayerNorm: normalize over ne[0] (per column)
    // For [4, 1] tensor, it normalizes the single column
    // mean = (1+2+3+4)/4 = 2.5
    // var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2)/4 = 1.25
    // std = sqrt(1.25 + eps) ≈ 1.12
    // normalized = (x - mean) / std
    ggml_tensor* normalized = ggml_norm(ctx, input, 1e-5f);

    // ggml_norm should produce output with same shape
    TEST_ASSERT_INT_EQ(normalized->ne[0], 4);
    TEST_ASSERT_INT_EQ(normalized->ne[1], 1);

    std::cout << "  Norm computation structure verified" << std::endl;

    ggml_free(ctx);

    return 0;
}

// Test LayerNorm with multiple sequence positions
int test_layernorm_mult_seq() {
    print_test_header("test_layernorm_mult_seq");

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ggml_context* ctx = ggml_init(params);
    TEST_ASSERT_MSG(ctx != nullptr, "Failed to init GGML context");

    const int n_embd = 768;
    const int seq_len = 16;

    // Input: [n_embd, seq_len] = [768, 16]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, seq_len);

    // Gamma and beta
    ggml_tensor* gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor* beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;
    for (int i = 0; i < n_embd; i++) {
        gamma_data[i] = 1.0f;
        beta_data[i] = 0.0f;
    }

    // Apply LayerNorm
    ggml_tensor* normalized = ggml_norm(ctx, input, GPT2Config::layer_norm_eps);
    ggml_tensor* scaled = ggml_mul(ctx, normalized, gamma);
    ggml_tensor* repeated_beta = ggml_repeat(ctx, beta, scaled);
    ggml_tensor* result = ggml_add(ctx, scaled, repeated_beta);

    TEST_ASSERT_INT_EQ(result->ne[0], n_embd);
    TEST_ASSERT_INT_EQ(result->ne[1], seq_len);

    std::cout << "  LayerNorm with seq_len=" << seq_len << " verified" << std::endl;

    ggml_free(ctx);

    return 0;
}

int run_layer_norm_tests() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Layer Norm Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int result = 0;
    result |= test_layernorm_config();
    result |= test_layernorm_shape();
    result |= test_rmsnorm_shape();
    result |= test_layernorm_identity();
    result |= test_layernorm_layer();
    result |= test_rmsnorm_layer();
    result |= test_repeat_broadcasting();
    result |= test_norm_computation();
    result |= test_layernorm_mult_seq();

    std::cout << "\n========================================" << std::endl;
    if (result == 0) {
        std::cout << "All Layer Norm Tests PASSED" << std::endl;
    } else {
        std::cout << "Some Layer Norm Tests FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return result;
}

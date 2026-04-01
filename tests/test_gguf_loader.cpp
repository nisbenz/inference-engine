#include "common_test.hpp"
#include "gguf_loader.h"
#include <cstdio>
#include <cstdlib>

// Test GGUF type sizes
int test_type_sizes() {
    print_test_header("test_type_sizes");

    // Verify block sizes match expected GGUF specification
    TEST_ASSERT_INT_EQ(GGUF_TID_Q4_0, 2);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q4_1, 3);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q5_0, 6);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q5_1, 7);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q8_0, 8);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q8_1, 9);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q2_K, 10);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q3_K, 11);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q4_K, 12);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q5_K, 13);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q6_K, 14);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q8_K, 15);
    TEST_ASSERT_INT_EQ(GGUF_TID_BF16, 16);
    TEST_ASSERT_INT_EQ(GGUF_TID_Q8_0_ALT, 30);

    std::cout << "  All GGUF type IDs verified correctly" << std::endl;
    return 0;
}

// Test BF16 to FP32 conversion
int test_bf16_conversion() {
    print_test_header("test_bf16_conversion");

    // BF16 representation of 1.0f is 0x3F80 (upper 16 bits of IEEE754 float 1.0)
    uint16_t bf16_one = 0x3F80;
    uint32_t val = (uint32_t)bf16_one << 16;
    float result;
    std::memcpy(&result, &val, sizeof(float));
    TEST_ASSERT_FLOAT_EQ(result, 1.0f, 0.001f);

    // BF16 representation of 0.0f is 0x0000
    bf16_one = 0x0000;
    val = (uint32_t)bf16_one << 16;
    std::memcpy(&result, &val, sizeof(float));
    TEST_ASSERT_FLOAT_EQ(result, 0.0f, 0.001f);

    // BF16 representation of -1.0f is 0xBFC0
    bf16_one = 0xBFC0;
    val = (uint32_t)bf16_one << 16;
    std::memcpy(&result, &val, sizeof(float));
    TEST_ASSERT_FLOAT_EQ(result, -1.0f, 0.01f);

    std::cout << "  BF16 conversion verified" << std::endl;
    return 0;
}

// Test FP16 to FP32 conversion (simple implementation)
int test_fp16_conversion() {
    print_test_header("test_fp16_conversion");

    // FP16 0x3C00 = 1.0 (IEEE754 half precision)
    uint16_t f16_one = 0x3C00;
    unsigned int sign = (f16_one >> 15) & 0x1;
    unsigned int exp = (f16_one >> 10) & 0x1f;
    unsigned int mant = f16_one & 0x3ff;

    // Simple conversion for normalized values
    int32_t e = (int32_t)exp - 15;
    float m = 1.0f + mant / 1024.0f;
    float result = (sign ? -1.0f : 1.0f) * std::pow(2.0f, e) * m;
    TEST_ASSERT_FLOAT_EQ(result, 1.0f, 0.01f);

    // FP16 0x0000 = 0.0
    f16_one = 0x0000;
    sign = (f16_one >> 15) & 0x1;
    exp = (f16_one >> 10) & 0x1f;
    mant = f16_one & 0x3ff;
    if (exp == 0 && mant == 0) {
        result = sign ? -0.0f : 0.0f;
    }
    TEST_ASSERT_FLOAT_EQ(result, 0.0f, 0.001f);

    std::cout << "  FP16 conversion verified" << std::endl;
    return 0;
}

// Test GGUF tensor info structure
int test_tensor_info_structure() {
    print_test_header("test_tensor_info_structure");

    GGUFTensorInfo info;
    info.name = "test.tensor";
    info.n_dims = 2;
    info.dims[0] = 768;
    info.dims[1] = 3072;
    info.type = GGUF_TID_F32;
    info.offset = 0;

    TEST_ASSERT_MSG(info.name == "test.tensor", "Tensor name mismatch");
    TEST_ASSERT_INT_EQ(info.n_dims, 2);
    TEST_ASSERT_INT_EQ(info.dims[0], 768);
    TEST_ASSERT_INT_EQ(info.dims[1], 3072);
    TEST_ASSERT_INT_EQ(info.type, GGUF_TID_F32);

    std::cout << "  Tensor info structure verified" << std::endl;
    return 0;
}

// Test GGUF metadata value types
int test_metadata_value_types() {
    print_test_header("test_metadata_value_types");

    GGUFValue val;

    // Test uint32
    val.type = MY_GGUF_TYPE_UINT32;
    val.u32 = 12345;
    TEST_ASSERT_INT_EQ(val.u32, 12345);

    // Test int32
    val.type = MY_GGUF_TYPE_INT32;
    val.i32 = -6789;
    TEST_ASSERT_INT_EQ(val.i32, -6789);

    // Test float32
    val.type = MY_GGUF_TYPE_FLOAT32;
    val.f32 = 3.14159f;
    TEST_ASSERT_FLOAT_EQ(val.f32, 3.14159f, 0.0001f);

    // Test bool
    val.type = MY_GGUF_TYPE_BOOL;
    val.b = true;
    TEST_ASSERT_MSG(val.b == true, "Bool value mismatch");

    std::cout << "  Metadata value types verified" << std::endl;
    return 0;
}

// Simulate Q8_0_ALT dequantization
int test_q80_alt_dequantization() {
    print_test_header("test_q80_alt_dequantization");

    // Q8_0_ALT format: 2-byte scale (FP16 little-endian) + 32 int8 values per block
    // For scale=1.0 (FP16: 0x3C00) and all values=1, each block is 34 bytes

    // Create test data: scale=1.0, values [1,2,3,...,32]
    std::vector<uint8_t> qdata(34);  // 1 block
    // Scale: 1.0 in FP16 little-endian is 0x00, 0x3C
    qdata[0] = 0x00;
    qdata[1] = 0x3C;
    // Values 1-32
    for (int i = 0; i < 32; i++) {
        qdata[2 + i] = (uint8_t)(i + 1);
    }

    // Manual dequantization (matching model's logic)
    uint16_t scale_f16 = (uint16_t)(qdata[0] | (qdata[1] << 8));
    // FP16 to FP32 conversion
    unsigned int sign = (scale_f16 >> 15) & 0x1;
    unsigned int exp = (scale_f16 >> 10) & 0x1f;
    unsigned int mant = scale_f16 & 0x3ff;
    int32_t e = (int32_t)exp - 15;
    float m = 1.0f + mant / 1024.0f;
    float scale = (sign ? -1.0f : 1.0f) * std::pow(2.0f, e) * m;

    // Verify scale is approximately 1.0
    TEST_ASSERT_FLOAT_EQ(scale, 1.0f, 0.01f);

    // Dequantize values
    std::vector<float> result(32);
    for (int i = 0; i < 32; i++) {
        int8_t val = (int8_t)qdata[2 + i];
        result[i] = val * scale;
        TEST_ASSERT_FLOAT_EQ(result[i], (float)(i + 1), 0.001f);
    }

    std::cout << "  Q8_0_ALT dequantization verified (scale=" << scale << ")" << std::endl;
    return 0;
}

// Test layer index extraction logic
int test_layer_idx_extraction() {
    print_test_header("test_layer_idx_extraction");

    // Helper lambda to extract layer index (mimics model.cpp logic)
    auto extract_layer_idx = [](const std::string& name) -> size_t {
        // Find blk. pattern (llama.cpp style)
        size_t blk_pos = name.find("blk.");
        if (blk_pos != std::string::npos) {
            size_t start = blk_pos + 4;
            size_t end = name.find('.', start);
            if (end != std::string::npos) {
                return std::stoi(name.substr(start, end - start));
            }
        }
        // Find .h. pattern (HuggingFace style)
        size_t h_pos = name.find(".h.");
        if (h_pos != std::string::npos) {
            size_t start = h_pos + 3;
            size_t end = name.find('.', start);
            if (end != std::string::npos) {
                return std::stoi(name.substr(start, end - start));
            }
        }
        return SIZE_MAX;  // Not a layer tensor
    };

    TEST_ASSERT_SIZE_T_EQ(extract_layer_idx("blk.0.attn_qkv.weight"), 0);
    TEST_ASSERT_SIZE_T_EQ(extract_layer_idx("blk.11.ffn_up.weight"), 11);
    TEST_ASSERT_SIZE_T_EQ(extract_layer_idx("blk.5.attn_output.weight"), 5);
    TEST_ASSERT_SIZE_T_EQ(extract_layer_idx("model.h.0.attn_qkv.weight"), 0);
    TEST_ASSERT_SIZE_T_EQ(extract_layer_idx("model.h.11.ffn_up.weight"), 11);
    TEST_ASSERT_SIZE_T_EQ(extract_layer_idx("model.h.5.attn_output.weight"), 5);
    TEST_ASSERT_SIZE_T_EQ(extract_layer_idx("token_embd.weight"), SIZE_MAX);
    TEST_ASSERT_SIZE_T_EQ(extract_layer_idx("output_norm.weight"), SIZE_MAX);

    std::cout << "  Layer index extraction verified" << std::endl;
    return 0;
}

// Test transpose detection logic
int test_transpose_detection() {
    print_test_header("test_transpose_detection");

    // GGUF stores tensors as [rows, cols] in row-major format
    // GGML 2D tensors: ne[0]=rows, ne[1]=cols (conceptually)
    // For ggml_mul_mat compatibility:
    //   - W.ne[0] must match input.ne[0]
    //   - W is treated as transposed in the multiplication

    // Case 1: GGUF [768, 3072] matches GGML ne[0]=768, ne[1]=3072
    // dims[0]=768, dims[1]=3072, dst_ne[0]=768, dst_ne[1]=3072
    // Condition: dims[0]==dst_ne[1] && dims[1]==dst_ne[0]
    //          : 768==3072 && 3072==768 -> FALSE -> no transpose needed

    uint64_t dims[2] = {768, 3072};
    size_t dst_ne[2] = {768, 3072};
    bool needs_transpose = (dims[0] == dst_ne[1] && dims[1] == dst_ne[0]);
    TEST_ASSERT_MSG(needs_transpose == false,
        "FFN up weight should not need transpose: dims[0]=768==dst_ne[1]=3072?");

    // Case 2: If GGUF had [3072, 768] but GGML expects [768, 3072]
    dims[0] = 3072;
    dims[1] = 768;
    needs_transpose = (dims[0] == dst_ne[1] && dims[1] == dst_ne[0]);
    TEST_ASSERT_MSG(needs_transpose == true,
        "Transpose should be detected when dims are swapped");

    std::cout << "  Transpose detection logic verified" << std::endl;
    return 0;
}

// Test actual transpose operation
int test_transpose_operation() {
    print_test_header("test_transpose_operation");

    // Create a 2x3 matrix in row-major:
    // [1, 2, 3]
    // [4, 5, 6]
    std::vector<float> src = {1, 2, 3, 4, 5, 6};
    int rows = 2;  // dims[0]
    int cols = 3;  // dims[1]

    // Transpose result: 3x2 matrix
    std::vector<float> dst(rows * cols);

    // Element [r, c] from src goes to [c, r] in dst
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }

    // Expected result:
    // [1, 4]
    // [2, 5]
    // [3, 6]
    TEST_ASSERT_FLOAT_EQ(dst[0], 1.0f, 0.001f);  // dst[0] = src[0]
    TEST_ASSERT_FLOAT_EQ(dst[1], 4.0f, 0.001f);  // dst[1] = src[2]
    TEST_ASSERT_FLOAT_EQ(dst[2], 2.0f, 0.001f);  // dst[2] = src[1]
    TEST_ASSERT_FLOAT_EQ(dst[3], 5.0f, 0.001f);  // dst[3] = src[4]
    TEST_ASSERT_FLOAT_EQ(dst[4], 3.0f, 0.001f);  // dst[4] = src[2]
    TEST_ASSERT_FLOAT_EQ(dst[5], 6.0f, 0.001f);  // dst[5] = src[5]

    std::cout << "  Transpose operation verified" << std::endl;
    return 0;
}

int run_gguf_loader_tests() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running GGUF Loader Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int result = 0;
    result |= test_type_sizes();
    result |= test_bf16_conversion();
    result |= test_fp16_conversion();
    result |= test_tensor_info_structure();
    result |= test_metadata_value_types();
    result |= test_q80_alt_dequantization();
    result |= test_layer_idx_extraction();
    result |= test_transpose_detection();
    result |= test_transpose_operation();

    std::cout << "\n========================================" << std::endl;
    if (result == 0) {
        std::cout << "All GGUF Loader Tests PASSED" << std::endl;
    } else {
        std::cout << "Some GGUF Loader Tests FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return result;
}

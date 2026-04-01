#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>

// Test assertion macros
#define TEST_ASSERT(condition, msg) \
    do { if (!(condition)) { \
        std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return 1; \
    }} while(0)

#define TEST_ASSERT_MSG(condition, msg, ...) \
    do { if (!(condition)) { \
        std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "  Additional info: " << __VA_ARGS__ << std::endl; \
        return 1; \
    }} while(0)

#define TEST_ASSERT_FLOAT_EQ(a, b, eps) \
    do { float _a = (a); float _b = (b); if (std::abs(_a - _b) >= (eps)) { \
        std::cerr << "FAIL: " << #a << " (" << _a << ") != " << #b << " (" << _b \
                  << ") within " << (eps) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return 1; \
    }} while(0)

#define TEST_ASSERT_INT_EQ(a, b) \
    do { if ((a) != (b)) { \
        std::cerr << "FAIL: " << #a << " (" << (a) << ") != " << #b << " (" << (b) \
                  << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return 1; \
    }} while(0)

#define TEST_ASSERT_SIZE_T_EQ(a, b) \
    do { if ((size_t)(a) != (size_t)(b)) { \
        std::cerr << "FAIL: " << #a << " (" << (a) << ") != " << #b << " (" << (b) \
                  << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return 1; \
    }} while(0)

// Helper to print a vector for debugging
template<typename T>
void print_vector(const std::string& name, const std::vector<T>& vec, size_t max_elements = 10) {
    std::cout << name << " [" << vec.size() << " elements]: {";
    for (size_t i = 0; i < std::min(vec.size(), max_elements); i++) {
        std::cout << vec[i];
        if (i < vec.size() - 1 && i < max_elements - 1) std::cout << ", ";
    }
    if (vec.size() > max_elements) std::cout << ", ...";
    std::cout << "}" << std::endl;
}

// Helper to compare floats with tolerance
inline bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Helper: create a tensor filled with a specific pattern
inline std::vector<float> create_pattern_vec(size_t size, float start = 0.0f, float step = 1.0f) {
    std::vector<float> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = start + i * step;
    }
    return result;
}

// Helper: create a tensor filled with zeros
inline std::vector<float> create_zeros_vec(size_t size) {
    return std::vector<float>(size, 0.0f);
}

// Helper: create a tensor filled with ones
inline std::vector<float> create_ones_vec(size_t size) {
    return std::vector<float>(size, 1.0f);
}

// Helper: verify two vectors are approximately equal
inline bool vectors_approx_eq(const std::vector<float>& a, const std::vector<float>& b, float eps = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (!float_eq(a[i], b[i], eps)) return false;
    }
    return true;
}

// Print test header
inline void print_test_header(const char* test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
}

// Print test result
inline void print_test_result(int result, const char* test_name) {
    if (result == 0) {
        std::cout << "[PASS] " << test_name << std::endl;
    } else {
        std::cout << "[FAIL] " << test_name << std::endl;
    }
}

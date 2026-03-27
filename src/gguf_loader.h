#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <cstdio>

// GGUF Type IDs
enum GGUF_TYPE_ID {
    GGUF_TID_F32 = 0,
    GGUF_TID_F16 = 1,
    GGUF_TID_Q4_0 = 2,
    GGUF_TID_Q4_1 = 3,
    GGUF_TID_Q5_0 = 6,
    GGUF_TID_Q5_1 = 7,
    GGUF_TID_Q8_0 = 8,
    GGUF_TID_Q8_1 = 9,
    GGUF_TID_Q2_K = 10,
    GGUF_TID_Q3_K = 11,
    GGUF_TID_Q4_K = 12,
    GGUF_TID_Q5_K = 13,
    GGUF_TID_Q6_K = 14,
    GGUF_TID_Q8_K = 15,
    GGUF_TID_BF16 = 16,
    GGUF_TID_I8 = 17,
    GGUF_TID_I16 = 18,
    GGUF_TID_I32 = 19,
    GGUF_TID_I64 = 20,
    GGUF_TID_F64 = 21,
};

// GGUF Metadata Value Types (renamed to avoid conflict with GGML's gguf_type)
enum GGUFMetadataValueType {
    MY_GGUF_TYPE_UINT8 = 0,
    MY_GGUF_TYPE_INT8 = 1,
    MY_GGUF_TYPE_UINT16 = 2,
    MY_GGUF_TYPE_INT16 = 3,
    MY_GGUF_TYPE_UINT32 = 4,
    MY_GGUF_TYPE_INT32 = 5,
    MY_GGUF_TYPE_FLOAT32 = 6,
    MY_GGUF_TYPE_BOOL = 7,
    MY_GGUF_TYPE_STRING = 8,
    MY_GGUF_TYPE_ARRAY = 9,
    MY_GGUF_TYPE_UINT64 = 10,
    MY_GGUF_TYPE_INT64 = 11,
    MY_GGUF_TYPE_FLOAT64 = 12,
};

// GGUF Value union for metadata
struct GGUFValue {
    GGUFMetadataValueType type;
    union {
        uint8_t u8;
        int8_t i8;
        uint16_t u16;
        int16_t i16;
        uint32_t u32;
        int32_t i32;
        float f32;
        bool b;
        uint64_t u64;
        int64_t i64;
        double f64;
    };
    std::string str;
    GGUFMetadataValueType arr_type;
    std::vector<GGUFValue> arr;
    GGUFValue() : type(MY_GGUF_TYPE_UINT8), u64(0) {}
};

// Tensor info
struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4];
    GGUF_TYPE_ID type;
    uint64_t offset;
};

// GGUF File handle
struct GGUFFile {
    FILE* fp = nullptr;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_kv_count = 0;
    std::map<std::string, GGUFValue> metadata;
    std::vector<GGUFTensorInfo> tensors;
    uint64_t tensor_data_offset = 0;

    // Helper to get metadata values
    std::string get_str(const std::string& key, const std::string& default_val = "") {
        auto it = metadata.find(key);
        if (it != metadata.end() && it->second.type == MY_GGUF_TYPE_STRING) {
            return it->second.str;
        }
        return default_val;
    }

    uint32_t get_u32(const std::string& key, uint32_t default_val = 0) {
        auto it = metadata.find(key);
        if (it != metadata.end() && it->second.type == MY_GGUF_TYPE_UINT32) {
            return it->second.u32;
        }
        return default_val;
    }

    uint32_t get_u32_or(const std::string& key, uint32_t fallback) {
        auto it = metadata.find(key);
        if (it != metadata.end()) {
            switch (it->second.type) {
                case MY_GGUF_TYPE_UINT32: return it->second.u32;
                case MY_GGUF_TYPE_INT32: return (uint32_t)it->second.i32;
                default: break;
            }
        }
        return fallback;
    }

    int32_t get_i32(const std::string& key, int32_t default_val = 0) {
        auto it = metadata.find(key);
        if (it != metadata.end() && it->second.type == MY_GGUF_TYPE_INT32) {
            return it->second.i32;
        }
        return default_val;
    }

    float get_f32(const std::string& key, float default_val = 0.0f) {
        auto it = metadata.find(key);
        if (it != metadata.end() && it->second.type == MY_GGUF_TYPE_FLOAT32) {
            return it->second.f32;
        }
        return default_val;
    }

    bool get_bool(const std::string& key, bool default_val = false) {
        auto it = metadata.find(key);
        if (it != metadata.end() && it->second.type == MY_GGUF_TYPE_BOOL) {
            return it->second.b;
        }
        return default_val;
    }

    int64_t get_i64(const std::string& key, int64_t default_val = 0) {
        auto it = metadata.find(key);
        if (it != metadata.end() && it->second.type == MY_GGUF_TYPE_INT64) {
            return it->second.i64;
        }
        return default_val;
    }
};

// Load GGUF file (reads metadata, leaves file open for tensor reading)
GGUFFile load_gguf(const char* path);

// Read tensor data from GGUF file into pre-allocated buffer
void read_tensor_data(GGUFFile& gguf, const GGUFTensorInfo& info, void* dst, size_t nbytes);

// Calculate tensor size in bytes (for allocation)
size_t gguf_tensor_nbytes(const GGUFTensorInfo& info);
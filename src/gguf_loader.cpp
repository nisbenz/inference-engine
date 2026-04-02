#include "gguf_loader.h"
#include <algorithm>
#include <cstring>

static inline void read_bytes(FILE* fp, void* dst, size_t n) {
    if (fread(dst, 1, n, fp) != n) throw std::runtime_error("GGUF: unexpected end of file");
}

template<typename T> static inline T read_val(FILE* fp) {
    T val; read_bytes(fp, &val, sizeof(T)); return val;
}

static std::string read_gguf_string(FILE* fp) {
    uint64_t len = read_val<uint64_t>(fp);
    std::string s(len, '\0');
    read_bytes(fp, s.data(), len);
    return s;
}

static void get_type_info(GGUF_TYPE_ID type, size_t& block_size, size_t& type_size) {
    switch (type) {
        case GGUF_TID_F32:     block_size = 1;   type_size = 4;  break;
        case GGUF_TID_F16:     block_size = 1;   type_size = 2;  break;
        case GGUF_TID_Q4_0:    block_size = 32;  type_size = 18; break;
        case GGUF_TID_Q4_1:    block_size = 32;  type_size = 20; break;
        case GGUF_TID_Q5_0:    block_size = 32;  type_size = 22; break;
        case GGUF_TID_Q5_1:    block_size = 32;  type_size = 24; break;
        case GGUF_TID_Q8_0:    block_size = 32;  type_size = 34; break;
        case GGUF_TID_Q8_1:    block_size = 32;  type_size = 36; break;
        case GGUF_TID_Q2_K:    block_size = 256; type_size = 84; break;
        case GGUF_TID_Q3_K:    block_size = 256; type_size = 110; break;
        case GGUF_TID_Q4_K:    block_size = 256; type_size = 144; break;
        case GGUF_TID_Q5_K:    block_size = 256; type_size = 176; break;
        case GGUF_TID_Q6_K:    block_size = 256; type_size = 210; break;
        case GGUF_TID_Q8_K:    block_size = 256; type_size = 292; break;
        case GGUF_TID_BF16:    block_size = 1;   type_size = 2;  break;
        case GGUF_TID_I8:      block_size = 1;   type_size = 1;  break;
        case GGUF_TID_I16:     block_size = 1;   type_size = 2;  break;
        case GGUF_TID_I32:     block_size = 1;   type_size = 4;  break;
        case GGUF_TID_I64:     block_size = 1;   type_size = 8;  break;
        case GGUF_TID_F64:     block_size = 1;   type_size = 8;  break;
        case GGUF_TID_Q8_0_ALT: block_size = 32; type_size = 34; break;  // Q8_0 variant
        default:
            throw std::runtime_error("GGUF: unsupported tensor type");
    }
}

size_t gguf_tensor_nbytes(const GGUFTensorInfo& info) {
    uint64_t n_elements = 1;
    for (uint32_t i = 0; i < info.n_dims; i++) n_elements *= info.dims[i];
    size_t block_size, type_size;
    get_type_info(info.type, block_size, type_size);
    return (n_elements / block_size) * type_size;
}

static GGUFValue read_metadata_value(FILE* fp, GGUFMetadataValueType type) {
    GGUFValue val; val.type = type;
    switch (type) {
        case MY_GGUF_TYPE_UINT8:   val.u8  = read_val<uint8_t>(fp);  break;
        case MY_GGUF_TYPE_INT8:    val.i8  = read_val<int8_t>(fp);   break;
        case MY_GGUF_TYPE_UINT16:  val.u16 = read_val<uint16_t>(fp); break;
        case MY_GGUF_TYPE_INT16:   val.i16 = read_val<int16_t>(fp);  break;
        case MY_GGUF_TYPE_UINT32:  val.u32 = read_val<uint32_t>(fp); break;
        case MY_GGUF_TYPE_INT32:   val.i32 = read_val<int32_t>(fp);  break;
        case MY_GGUF_TYPE_FLOAT32: val.f32 = read_val<float>(fp);    break;
        case MY_GGUF_TYPE_BOOL:    val.b   = read_val<uint8_t>(fp);  break;
        case MY_GGUF_TYPE_STRING:   val.str = read_gguf_string(fp);   break;
        case MY_GGUF_TYPE_UINT64:  val.u64 = read_val<uint64_t>(fp); break;
        case MY_GGUF_TYPE_INT64:   val.i64 = read_val<int64_t>(fp);  break;
        case MY_GGUF_TYPE_FLOAT64: val.f64 = read_val<double>(fp);   break;
        case MY_GGUF_TYPE_ARRAY: {
            GGUFMetadataValueType arr_type = read_val<GGUFMetadataValueType>(fp);
            uint64_t arr_len = read_val<uint64_t>(fp);
            val.arr_type = arr_type;
            val.arr.resize(arr_len);
            for (uint64_t i = 0; i < arr_len; i++) val.arr[i] = read_metadata_value(fp, arr_type);
            break;
        }
        default: throw std::runtime_error("GGUF: unknown metadata type");
    }
    return val;
}

GGUFFile load_gguf(const char* path) {
    GGUFFile gguf; FILE* fp = fopen(path, "rb");
    if (!fp) throw std::runtime_error("Cannot open GGUF file");
    gguf.fp = fp;
    uint32_t magic = read_val<uint32_t>(fp);
    if (magic != 0x46554747) {
        char buf[5] = {0};
        memcpy(buf, &magic, 4);
        throw std::runtime_error("GGUF: invalid magic");
    }
    gguf.version = read_val<uint32_t>(fp);
    gguf.tensor_count = read_val<uint64_t>(fp);
    gguf.metadata_kv_count = read_val<uint64_t>(fp);
    for (uint64_t i = 0; i < gguf.metadata_kv_count; i++) {
        std::string key = read_gguf_string(fp);
        GGUFMetadataValueType vtype = read_val<GGUFMetadataValueType>(fp);
        gguf.metadata[key] = read_metadata_value(fp, vtype);
    }
    gguf.tensors.resize(gguf.tensor_count);
    for (uint64_t i = 0; i < gguf.tensor_count; i++) {
        auto& t = gguf.tensors[i];
        t.name = read_gguf_string(fp);
        t.n_dims = read_val<uint32_t>(fp);
        for (uint32_t d = 0; d < t.n_dims; d++) t.dims[d] = read_val<uint64_t>(fp);
        t.type = (GGUF_TYPE_ID)read_val<uint32_t>(fp);
        t.offset = read_val<uint64_t>(fp);
    }
    uint32_t align = gguf.get_u32_or("general.alignment", 32);
    gguf.tensor_data_offset = (ftell(fp) + align - 1) & ~(uint64_t)(align - 1);

    fseek(fp, 0, SEEK_END);
    const uint64_t file_size = (uint64_t)ftell(fp);
    const uint64_t tensor_data_end = file_size;
    for (uint64_t i = 0; i < gguf.tensor_count; i++) {
        const uint64_t next_offset = (i + 1 < gguf.tensor_count)
            ? gguf.tensors[i + 1].offset
            : (tensor_data_end - gguf.tensor_data_offset);
        if (next_offset < gguf.tensors[i].offset) {
            throw std::runtime_error("GGUF: tensor offsets are not monotonic");
        }
        gguf.tensors[i].data_size = next_offset - gguf.tensors[i].offset;
    }

    fseek(fp, (long)gguf.tensor_data_offset, SEEK_SET);
    return gguf;
}

void read_tensor_data(GGUFFile& gguf, const GGUFTensorInfo& info, void* dst, size_t nbytes) {
    fseek(gguf.fp, gguf.tensor_data_offset + info.offset, SEEK_SET);
    if (fread(dst, 1, nbytes, gguf.fp) != nbytes) throw std::runtime_error("GGUF: read fail");
}

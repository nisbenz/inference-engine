#include "kv_cache.hpp"
#include "layers.hpp"
#include <cstring>
#include <iostream>

// Use CONTEXT_LENGTH from model.hpp as MAX_SEQ_LEN
static constexpr int MAX_SEQ_LEN = 1024;

// ============== KVCacheEntry ==============

void KVCacheEntry::init(ggml_context* ctx, int n_heads, int head_dim, int max_seq_len) {
    // Allocate tensors: (n_heads, seq_len, head_dim)
    k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, max_seq_len);
    v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_heads, head_dim, max_seq_len);

    ggml_set_name(k, "k_cache");
    ggml_set_name(v, "v_cache");

    current_length = 0;
}

void KVCacheEntry::update(int position, const float* k_data, const float* v_data) {
    // Copy k_data into k[:, position, :]
    // k is stored as (head_dim, n_heads, seq_len)
    // We need to copy k_data (head_dim * n_heads) to position

    int n_heads = GPT2Config::n_heads;
    int head_dim = GPT2Config::head_dim;

    float* k_dst = (float*)k->data + position * n_heads * head_dim;
    memcpy(k_dst, k_data, n_heads * head_dim * sizeof(float));

    // Copy v_data into v[:, position, :]
    // v is stored as (n_heads, head_dim, seq_len)
    float* v_dst = (float*)v->data + position * n_heads * head_dim;
    memcpy(v_dst, v_data, n_heads * head_dim * sizeof(float));

    current_length = position + 1;
}

void KVCacheEntry::get(int position, float* k_out, float* v_out) {
    int n_heads = GPT2Config::n_heads;
    int head_dim = GPT2Config::head_dim;

    // Get all keys up to and including position
    float* k_src = (float*)k->data;
    memcpy(k_out, k_src, (position + 1) * n_heads * head_dim * sizeof(float));

    float* v_src = (float*)v->data;
    memcpy(v_out, v_src, (position + 1) * n_heads * head_dim * sizeof(float));
}

// ============== KVCache ==============

KVCache::KVCache() : current_length_(0) {
    layers_.resize(N_LAYERS);
}

void KVCache::init(ggml_context* ctx) {
    for (int i = 0; i < N_LAYERS; i++) {
        layers_[i].init(ctx, N_HEADS, HEAD_DIM, MAX_SEQ_LEN);
    }
    current_length_ = 0;
}

void KVCache::reset() {
    for (int i = 0; i < N_LAYERS; i++) {
        memset(layers_[i].k->data, 0, ggml_nbytes(layers_[i].k));
        memset(layers_[i].v->data, 0, ggml_nbytes(layers_[i].v));
        layers_[i].current_length = 0;
    }
    current_length_ = 0;
}

void KVCache::update(int layer_idx, int position, const float* k_data, const float* v_data) {
    if (layer_idx < 0 || layer_idx >= N_LAYERS) {
        std::cerr << "KVCache: invalid layer index " << layer_idx << std::endl;
        return;
    }
    layers_[layer_idx].update(position, k_data, v_data);
    current_length_ = std::max(current_length_, position + 1);
}

// ============== Helper Functions ==============

void copy_tensor_slice(
    ggml_tensor* dst,
    const ggml_tensor* src,
    int dst_offset,
    int n_tokens
) {
    // Assumes dst and src have same shape except possibly seq_len dimension
    // Copies n_tokens from src to dst at dst_offset

    int n_heads = GPT2Config::n_heads;
    int head_dim = GPT2Config::head_dim;

    size_t src_row_size = n_heads * head_dim * sizeof(float);
    size_t dst_row_size = ggml_nbytes(dst) / (MAX_SEQ_LEN);

    const float* src_ptr = (const float*)src->data;
    float* dst_ptr = (float*)dst->data + dst_offset * n_heads * head_dim;

    memcpy(dst_ptr, src_ptr, n_tokens * n_heads * head_dim * sizeof(float));
}

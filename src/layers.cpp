#include "layers.hpp"
#include <cmath>
#include <iostream>

// Removed ggml_new_scalar helper since native ops are safer

// ============== LayerNorm ==============

ggml_tensor* LayerNorm::forward(ggml_context* ctx, ggml_tensor* x) {
    // x: ne[0]=n_embd, ne[1]=seq_len
    // ggml_norm normalizes along ne[0] (per-token), which is exactly LayerNorm
    ggml_tensor* x_norm = ggml_norm(ctx, x, GPT2Config::layer_norm_eps);
    // gamma, beta: ne[0]=n_embd — broadcasts over ne[1]=seq_len
    ggml_tensor* scaled = ggml_mul(ctx, x_norm, gamma);
    ggml_tensor* result = ggml_add(ctx, scaled, beta);
    return result;
}

// ============== RMSNorm ==============

ggml_tensor* RMSNorm::forward(ggml_context* ctx, ggml_tensor* x) {
    // ggml_rms_norm normalizes along ne[0] per row
    ggml_tensor* normalized = ggml_rms_norm(ctx, x, GPT2Config::layer_norm_eps);
    ggml_tensor* result = ggml_mul(ctx, normalized, weight);
    return result;
}

// ============== GELU ==============

ggml_tensor* GELU::forward(ggml_context* ctx, ggml_tensor* x) {
    // Native GGML GELU operator prevents scalar broadcasting memory leaks
    return ggml_gelu(ctx, x);
}

// ============== Attention ==============

Attention::Attention()
    : n_heads(GPT2Config::n_heads)
    , n_embd(GPT2Config::n_embd)
    , head_dim(GPT2Config::head_dim)
    , seq_len(0)
    , c_attn_weight(nullptr)
    , c_attn_bias(nullptr)
    , c_proj_weight(nullptr)
    , c_proj_bias(nullptr)
    , k_cache(nullptr)
    , v_cache(nullptr)
{}

void Attention::init_cache(ggml_context* ctx) {
    // KV cache: ne[0]=head_dim, ne[1]=context_length, ne[2]=n_heads
    k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                  GPT2Config::head_dim,
                                  GPT2Config::context_length,
                                  GPT2Config::n_heads);
    v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                  GPT2Config::head_dim,
                                  GPT2Config::context_length,
                                  GPT2Config::n_heads);
    ggml_set_name(k_cache, "k_cache");
    ggml_set_name(v_cache, "v_cache");
}

ggml_tensor* Attention::forward(
    ggml_context* ctx,
    ggml_cgraph* gf,
    ggml_tensor* x,
    int position,
    bool use_cache
) {
    // x: ne[0]=n_embd, ne[1]=seq_len
    int n_heads = GPT2Config::n_heads;
    int n_embd = GPT2Config::n_embd;
    int head_dim = GPT2Config::head_dim;
    int seq_len = (int)x->ne[1];

    // ---- QKV projection ----
    // c_attn_weight: ne[0]=n_embd, ne[1]=3*n_embd
    // ggml_mul_mat(W, x) = W^T @ x; both ne[0]=n_embd
    // Result: ne[0]=3*n_embd, ne[1]=seq_len
    ggml_tensor* qkv = ggml_mul_mat(ctx, c_attn_weight, x);
    ggml_tensor* repeated_attn_bias = ggml_repeat(ctx, c_attn_bias, qkv);
    ggml_tensor* qkv_out = ggml_add(ctx, qkv, repeated_attn_bias);
    qkv = qkv_out;

    // ---- Split Q, K, V along ne[0] ----
    // Each chunk: ne[0]=n_embd, ne[1]=seq_len
    size_t es = ggml_element_size(qkv);
    ggml_tensor* q = ggml_view_2d(ctx, qkv, n_embd, seq_len,
                                   qkv->nb[1], 0);
    ggml_tensor* k = ggml_view_2d(ctx, qkv, n_embd, seq_len,
                                   qkv->nb[1], n_embd * es);
    ggml_tensor* v = ggml_view_2d(ctx, qkv, n_embd, seq_len,
                                   qkv->nb[1], 2 * n_embd * es);

    // ---- Reshape to separate heads ----
    // n_embd = n_heads * head_dim, so reshape ne[0] into (head_dim, n_heads)
    // Result: ne[0]=head_dim, ne[1]=n_heads, ne[2]=seq_len
    ggml_tensor* Q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_heads, seq_len);
    ggml_tensor* K = ggml_reshape_3d(ctx, ggml_cont(ctx, k), head_dim, n_heads, seq_len);
    ggml_tensor* V = ggml_reshape_3d(ctx, ggml_cont(ctx, v), head_dim, n_heads, seq_len);

    // Permute to (head_dim, seq_len, n_heads) for batched per-head matmul
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    // ---- Attention scores ----
    // ggml_mul_mat(K, Q) = K^T @ Q, batched over ne[2]=n_heads
    // Both ne[0]=head_dim; result: ne[0]=seq_len, ne[1]=seq_len, ne[2]=n_heads
    ggml_tensor* scores = ggml_mul_mat(ctx, K, Q);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)head_dim));

    // ---- Causal mask using ggml_diag_mask_inf ----
    // position = n_past (number of tokens already processed in the full sequence)
    // This masks future tokens: element at (row, col) where col > row + n_past gets -inf
    // Official GGML pattern: ggml_diag_mask_inf(ctx, scores, n_past)
    scores = ggml_diag_mask_inf(ctx, scores, position);

    // ---- Softmax along ne[0] (over kv positions per query) ----
    ggml_tensor* attn_weights = ggml_soft_max(ctx, scores);

    // ---- Weighted sum of values ----
    // V: ne[0]=head_dim, ne[1]=seq_len, ne[2]=n_heads (after permute, non-contiguous)
    // Need V with ne[0]=seq_len for matmul with attn_weights
    // Permute V: swap ne[0] and ne[1] -> (seq_len, head_dim, n_heads)
    // ggml_cont required: double-permute makes nb[0] > nb[1] (transposed),
    // and ggml_mul_mat asserts !ggml_is_transposed(a)
    ggml_tensor* V_t = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));

    // ggml_mul_mat(V_t, attn_weights) = V_t^T @ attn_weights
    // V_t ne[0]=seq_len, attn_weights ne[0]=seq_len — match
    // Result: ne[0]=head_dim, ne[1]=seq_len, ne[2]=n_heads
    ggml_tensor* attn_out = ggml_mul_mat(ctx, V_t, attn_weights);

    // ---- Recombine heads ----
    // Permute to (head_dim, n_heads, seq_len) then reshape to (n_embd, seq_len)
    attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
    attn_out = ggml_cont(ctx, attn_out);
    attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, seq_len);

    // ---- Output projection ----
    // c_proj_weight: ne[0]=n_embd, ne[1]=n_embd
    // ggml_mul_mat(W, x) = W^T @ x; both ne[0]=n_embd
    // Result: ne[0]=n_embd, ne[1]=seq_len
    ggml_tensor* out = ggml_mul_mat(ctx, c_proj_weight, attn_out);
    ggml_tensor* repeated_proj_bias = ggml_repeat(ctx, c_proj_bias, out);
    out = ggml_add(ctx, out, repeated_proj_bias);

    ggml_build_forward_expand(gf, out);
    return out;
}

void Attention::set_weights(
    const float* qkv_w, const float* qkv_b,
    const float* proj_w, const float* proj_b
) {
    // Weights are loaded directly into tensors during model loading
}

// ============== FFN ==============

FFN::FFN()
    : c_fc_weight(nullptr)
    , c_fc_bias(nullptr)
    , c_proj_weight(nullptr)
    , c_proj_bias(nullptr)
{}

ggml_tensor* FFN::gelu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_gelu(ctx, x);
}

ggml_tensor* FFN::forward(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x) {
    // x: ne[0]=n_embd, ne[1]=seq_len
    //
    // Up projection: c_fc_weight ne[0]=n_embd, ne[1]=n_ffn
    // ggml_mul_mat(W, x) = W^T @ x; both ne[0]=n_embd
    // Result: ne[0]=n_ffn, ne[1]=seq_len
    ggml_tensor* up = ggml_mul_mat(ctx, c_fc_weight, x);
    ggml_tensor* repeated_fc_bias = ggml_repeat(ctx, c_fc_bias, up);
    up = ggml_add(ctx, up, repeated_fc_bias);

    // GELU activation
    ggml_tensor* activated = gelu(ctx, up);

    // Down projection: c_proj_weight ne[0]=n_ffn, ne[1]=n_embd
    // ggml_mul_mat(W, x) = W^T @ x; both ne[0]=n_ffn
    // Result: ne[0]=n_embd, ne[1]=seq_len
    ggml_tensor* down = ggml_mul_mat(ctx, c_proj_weight, activated);
    ggml_tensor* repeated_down_bias = ggml_repeat(ctx, c_proj_bias, down);
    down = ggml_add(ctx, down, repeated_down_bias);

    return down;
}

void FFN::set_weights(
    const float* fc_w, const float* fc_b,
    const float* proj_w, const float* proj_b
) {
    // Copy weights to tensors
}

// ============== TransformerBlock ==============

TransformerBlock::TransformerBlock() {}

ggml_tensor* TransformerBlock::forward(
    ggml_context* ctx,
    ggml_cgraph* gf,
    ggml_tensor* x,
    int position,
    bool use_cache
) {
    // Pre-norm architecture: LN1 -> Attention -> Residual
    ggml_tensor* ln1_out = layer_norm(ctx, x, ln1.gamma, ln1.beta, GPT2Config::layer_norm_eps);
    ggml_tensor* attn_out = attention.forward(ctx, gf, ln1_out, position, use_cache);
    ggml_tensor* h1 = ggml_add(ctx, x, attn_out);

    // LN2 -> FFN -> Residual
    ggml_tensor* ln2_out = layer_norm(ctx, h1, ln2.gamma, ln2.beta, GPT2Config::layer_norm_eps);
    ggml_tensor* ffn_out = ffn.forward(ctx, gf, ln2_out);
    ggml_tensor* h2 = ggml_add(ctx, h1, ffn_out);

    return h2;
}

void TransformerBlock::build_graph(
    ggml_context* ctx,
    ggml_cgraph* gf,
    ggml_tensor* x,
    int position,
    bool use_cache
) {
    ggml_tensor* ln1_out = layer_norm(ctx, x, ln1.gamma, ln1.beta, GPT2Config::layer_norm_eps);
    ggml_build_forward_expand(gf, ln1_out);

    ggml_tensor* attn_out = attention.forward(ctx, gf, ln1_out, position, use_cache);
    ggml_build_forward_expand(gf, attn_out);

    ggml_tensor* h1 = ggml_add(ctx, x, attn_out);
    ggml_build_forward_expand(gf, h1);

    ggml_tensor* ln2_out = layer_norm(ctx, h1, ln2.gamma, ln2.beta, GPT2Config::layer_norm_eps);
    ggml_build_forward_expand(gf, ln2_out);

    ggml_tensor* ffn_out = ffn.forward(ctx, gf, ln2_out);
    ggml_build_forward_expand(gf, ffn_out);

    ggml_tensor* h2 = ggml_add(ctx, h1, ffn_out);
    ggml_build_forward_expand(gf, h2);
}

// ============== Utilities ==============

ggml_tensor* linear(
    ggml_context* ctx,
    ggml_tensor* input,
    ggml_tensor* weight,
    ggml_tensor* bias
) {
    // ggml_mul_mat(W, x) = W^T @ x
    // weight ne[0] must match input ne[0]
    ggml_tensor* result = ggml_mul_mat(ctx, weight, input);
    if (bias != nullptr) {
        result = ggml_add(ctx, result, bias);
    }
    return result;
}

ggml_tensor* layer_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* gamma,
    ggml_tensor* beta,
    float eps
) {
    ggml_tensor* x_norm = ggml_norm(ctx, x, eps);
    // Explicitly repeat beta across seq_len axis for structured memory broadcasting!
    ggml_tensor* repeated_gamma = ggml_repeat(ctx, gamma, x_norm);
    ggml_tensor* scaled = ggml_mul(ctx, x_norm, repeated_gamma);
    ggml_tensor* repeated_beta = ggml_repeat(ctx, beta, scaled);
    ggml_tensor* result = ggml_add(ctx, scaled, repeated_beta);
    return result;
}

ggml_tensor* rms_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    float eps
) {
    ggml_tensor* normalized = ggml_rms_norm(ctx, x, eps);
    ggml_tensor* result = ggml_mul(ctx, normalized, weight);
    return result;
}

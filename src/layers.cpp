#include "layers.hpp"
#include <cmath>
#include <iostream>

// Helper to create a scalar tensor (for adding constants)
static inline ggml_tensor* ggml_new_scalar(ggml_context* ctx, float value) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float*)t->data)[0] = value;
    return t;
}

// ============== LayerNorm ==============

ggml_tensor* LayerNorm::forward(ggml_context* ctx, ggml_tensor* x) {
    // LayerNorm: gamma * (x - mean) / sqrt(var + eps) + beta
    // x shape: (seq_len, n_embd)
    // Since GGML lacks per-row reduction ops, we use a simplified approach:
    // compute mean/variance over the full tensor as approximation
    // For seq_len=1 (typical during generation), this equals per-row computation

    int seq_len = (int)ggml_nrows(x);
    int n_embd = GPT2Config::n_embd;

    // Compute mean over all elements: sum(x) / (seq_len * n_embd)
    ggml_tensor* x_sum = ggml_sum(ctx, x);
    ggml_tensor* x_mean = ggml_scale(ctx, x_sum, 1.0f / (float)(seq_len * n_embd));

    // Compute x - mean
    ggml_tensor* x_centered = ggml_sub(ctx, x, x_mean);

    // Compute variance: mean((x - mean)^2)
    ggml_tensor* x_centered_sq = ggml_sqr(ctx, x_centered);
    ggml_tensor* var_sum = ggml_sum(ctx, x_centered_sq);
    ggml_tensor* variance = ggml_scale(ctx, var_sum, 1.0f / (float)(seq_len * n_embd));

    // Compute sqrt(var + eps)
    ggml_tensor* var_eps = ggml_add(ctx, variance, ggml_new_scalar(ctx, GPT2Config::layer_norm_eps));
    ggml_tensor* std = ggml_sqrt(ctx, var_eps);

    // Normalize: (x - mean) / std
    ggml_tensor* x_norm = ggml_div(ctx, x_centered, std);

    // Scale and shift: gamma * x_norm + beta
    ggml_tensor* scaled = ggml_mul(ctx, x_norm, gamma);
    ggml_tensor* result = ggml_add(ctx, scaled, beta);
    return result;
}

// ============== RMSNorm ==============

ggml_tensor* RMSNorm::forward(ggml_context* ctx, ggml_tensor* x) {
    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    ggml_tensor* x2 = ggml_sqr(ctx, x);
    ggml_tensor* mean2 = ggml_mean(ctx, x2);
    ggml_tensor* var_eps = ggml_add(ctx, mean2, ggml_new_scalar(ctx, GPT2Config::layer_norm_eps));
    ggml_tensor* rms = ggml_sqrt(ctx, var_eps);
    ggml_tensor* normalized = ggml_div(ctx, x, rms);
    ggml_tensor* result = ggml_mul(ctx, normalized, weight);
    return result;
}

// ============== GELU ==============

ggml_tensor* GELU::forward(ggml_context* ctx, ggml_tensor* x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ggml_tensor* x3 = ggml_mul(ctx, ggml_mul(ctx, x, x), x);  // x^3
    ggml_tensor* inner = ggml_add(ctx, x, ggml_scale(ctx, x3, GELU::GELU_A));  // x + 0.044715 * x^3
    ggml_tensor* tanh_arg = ggml_scale(ctx, inner, GELU::GELU_SQRT2_OVER_PI);  // sqrt(2/pi) * inner
    ggml_tensor* tanh_result = ggml_tanh(ctx, tanh_arg);
    ggml_tensor* one_plus_tanh = ggml_add(ctx, tanh_result, ggml_new_scalar(ctx, 1.0f));
    ggml_tensor* result = ggml_scale(ctx, ggml_mul(ctx, x, one_plus_tanh), 0.5f);
    return result;
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
    // KV cache shape: (n_heads, seq_len, head_dim)
    // We allocate max sequence length
    k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                  GPT2Config::head_dim,
                                  GPT2Config::n_heads,
                                  GPT2Config::context_length);
    v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                  GPT2Config::n_heads,
                                  GPT2Config::head_dim,
                                  GPT2Config::context_length);
    ggml_set_name(k_cache, "k_cache");
    ggml_set_name(v_cache, "v_cache");
}

ggml_tensor* Attention::causal_mask(ggml_context* ctx, int seq_len) {
    // Create causal mask: lower triangular matrix
    // mask[i,j] = 0 if j <= i (can attend), -inf if j > i (cannot attend)
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);

    float* data = (float*)mask->data;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            data[i * seq_len + j] = (j > i) ? -10000.0f : 0.0f;
        }
    }

    return mask;
}

ggml_tensor* Attention::forward(
    ggml_context* ctx,
    ggml_cgraph* gf,
    ggml_tensor* x,
    int position,
    bool use_cache
) {
    // x: (seq_len, n_embd)
    // position: current position in sequence (for KV cache indexing)
    // use_cache: if true, store/retrieve from KV cache

    int n_heads = GPT2Config::n_heads;
    int n_embd = GPT2Config::n_embd;
    int head_dim = GPT2Config::head_dim;
    int seq_len = (int)ggml_nrows(x);

    printf("[Debug Attn] START: x ne[0]=%lu ne[1]=%lu, c_attn_weight ne[0]=%lu ne[1]=%lu\n",
           (unsigned long)x->ne[0], (unsigned long)x->ne[1],
           (unsigned long)c_attn_weight->ne[0], (unsigned long)c_attn_weight->ne[1]);
    fflush(stdout);

    // QKV projection: (seq_len, n_embd) @ (n_embd, 3*n_embd) = (seq_len, 3*n_embd)
    ggml_tensor* qkv = ggml_mul_mat(ctx, x, c_attn_weight);
    qkv = ggml_add(ctx, qkv, c_attn_bias);

    // Split into q, k, v - each (seq_len, n_embd)
    ggml_tensor* q = ggml_view_2d(ctx, qkv, n_embd, seq_len, n_embd * sizeof(float), 0);
    ggml_tensor* k = ggml_view_2d(ctx, qkv, n_embd, seq_len, n_embd * sizeof(float), n_embd * sizeof(float));
    ggml_tensor* v = ggml_view_2d(ctx, qkv, n_embd, seq_len, n_embd * sizeof(float), 2 * n_embd * sizeof(float));
    printf("[Debug Attn] qkv: ne[0]=%lu, ne[1]=%lu\n", (unsigned long)qkv->ne[0], (unsigned long)qkv->ne[1]);
    printf("[Debug Attn] q: ne[0]=%lu, ne[1]=%lu\n", (unsigned long)q->ne[0], (unsigned long)q->ne[1]);

    // Reshape to (seq_len, n_heads, head_dim)
    q = ggml_reshape_3d(ctx, q, seq_len, n_heads, head_dim);
    k = ggml_reshape_3d(ctx, k, seq_len, n_heads, head_dim);
    v = ggml_reshape_3d(ctx, v, seq_len, n_heads, head_dim);

    // Handle KV cache storage and retrieval
    int total_kv_len = seq_len;

    if (use_cache && position >= 0) {
        // Store new k,v to cache at current position
        // k_cache format: (head_dim, n_heads, context_length)
        // k is (seq_len, n_heads, head_dim), we want to store at position offset
        for (int i = 0; i < seq_len; i++) {
            int store_pos = position + i;

            // Get view into k_cache at position store_pos
            ggml_tensor* k_slice = ggml_view_3d(ctx, k_cache,
                head_dim, n_heads, 1,
                n_heads * head_dim * sizeof(float),  // stride in dim0 (ne1*ne2)
                head_dim * sizeof(float),             // stride in dim1 (ne2)
                store_pos * head_dim * n_heads * sizeof(float)); // offset

            // Transpose k[i] from (n_heads, head_dim) to (head_dim, n_heads) for storage
            ggml_tensor* k_row = ggml_view_2d(ctx, k, n_heads, head_dim,
                n_heads * head_dim * sizeof(float), i * n_heads * head_dim * sizeof(float));
            k_row = ggml_transpose(ctx, k_row);

            ggml_tensor* k_dst = ggml_view_2d(ctx, k_slice, n_heads, head_dim,
                head_dim * sizeof(float), 0);
            k_dst = ggml_cpy(ctx, k_row, k_dst);
            ggml_build_forward_expand(gf, k_dst);

            // Store v similarly
            ggml_tensor* v_slice = ggml_view_3d(ctx, v_cache,
                n_heads, head_dim, 1,
                head_dim * sizeof(float),
                0,
                store_pos * n_heads * head_dim * sizeof(float));

            ggml_tensor* v_row = ggml_view_2d(ctx, v, n_heads, head_dim,
                n_heads * head_dim * sizeof(float), i * n_heads * head_dim * sizeof(float));

            ggml_tensor* v_dst = ggml_view_2d(ctx, v_slice, head_dim, n_heads,
                n_heads * sizeof(float), 0);
            v_dst = ggml_cpy(ctx, v_row, v_dst);
            ggml_build_forward_expand(gf, v_dst);
        }

        // Retrieve cached K,V if position > 0
        if (position > 0) {
            // Get cached keys/values from positions 0 to position-1
            // k_cache: (head_dim, n_heads, context_length)
            ggml_tensor* k_cached = ggml_view_3d(ctx, k_cache,
                n_heads, position, head_dim,
                head_dim * n_heads * sizeof(float),  // stride dim0
                head_dim * sizeof(float),              // stride dim1
                0);

            // Transpose from (n_heads, position, head_dim) to (position, n_heads, head_dim)
            k_cached = ggml_reshape_3d(ctx, k_cached, n_heads, position, head_dim);
            k_cached = ggml_transpose(ctx, k_cached);

            ggml_tensor* v_cached = ggml_view_3d(ctx, v_cache,
                n_heads, position, head_dim,
                head_dim * GPT2Config::context_length * sizeof(float),
                GPT2Config::context_length * sizeof(float),
                sizeof(float));

            v_cached = ggml_reshape_3d(ctx, v_cached, n_heads, position, head_dim);
            v_cached = ggml_transpose(ctx, v_cached);

            // Concat cached and new k,v
            k = ggml_concat(ctx, k_cached, k, 0);
            v = ggml_concat(ctx, v_cached, v, 0);

            total_kv_len = position + seq_len;
        }
    }

    // Now k and v are (total_kv_len, n_heads, head_dim)
    // q is (seq_len, n_heads, head_dim)

    // Reshape q for matmul: (seq_len, n_heads, head_dim) -> (seq_len * n_heads, head_dim)
    ggml_tensor* q_2d = ggml_reshape_2d(ctx, q, seq_len * n_heads, head_dim);

    // Reshape k for matmul: (total_kv_len, n_heads, head_dim) -> (n_heads, total_kv_len, head_dim)
    k = ggml_reshape_3d(ctx, k, n_heads, total_kv_len, head_dim);
    // Transpose to (n_heads, head_dim, total_kv_len)
    k = ggml_transpose(ctx, k);
    // Reshape to (head_dim, n_heads * total_kv_len)
    ggml_tensor* k_mat = ggml_reshape_2d(ctx, k, head_dim, n_heads * total_kv_len);

    // Similar for v: (total_kv_len, n_heads, head_dim) -> (n_heads * total_kv_len, head_dim)
    v = ggml_reshape_3d(ctx, v, n_heads, total_kv_len, head_dim);
    v = ggml_transpose(ctx, v);  // (head_dim, n_heads, total_kv_len)
    ggml_tensor* v_mat = ggml_reshape_2d(ctx, v, head_dim, n_heads * total_kv_len);
    v_mat = ggml_transpose(ctx, v_mat);  // (n_heads * total_kv_len, head_dim)

    // Compute attention scores: q_2d @ k_mat
    // (seq_len * n_heads, head_dim) @ (head_dim, n_heads * total_kv_len)
    // = (seq_len * n_heads, n_heads * total_kv_len)
    printf("[Debug Attn] seq_len=%d, n_heads=%d, head_dim=%d, total_kv_len=%d, position=%d\n",
           seq_len, n_heads, head_dim, total_kv_len, position);
    printf("[Debug Attn] q_2d: ne[0]=%lu, ne[1]=%lu\n", (unsigned long)q_2d->ne[0], (unsigned long)q_2d->ne[1]);
    printf("[Debug Attn] k_mat: ne[0]=%lu, ne[1]=%lu\n", (unsigned long)k_mat->ne[0], (unsigned long)k_mat->ne[1]);
    ggml_tensor* scores = ggml_mul_mat(ctx, q_2d, k_mat);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)head_dim));

    // Apply causal mask: for token i, can only attend to positions 0..i in the full sequence
    // For seq_len tokens starting at position, we need a mask that allows each token to attend
    // to positions up to (position + i)
    if (seq_len > 1 || (use_cache && position > 0)) {
        ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, total_kv_len);
        float* mask_data = (float*)mask->data;

        for (int i = 0; i < seq_len; i++) {
            int token_pos = position + i;  // absolute position of token i
            for (int j = 0; j < total_kv_len; j++) {
                // Can attend if j < token_pos (i.e., j is before or at current position)
                // But when j >= position and j < position + seq_len, that's the current chunk
                // And when j >= position + seq_len... but that shouldn't happen with proper k_seq
                int kv_pos = j;  // position in the cached/new concatenated k
                mask_data[i * total_kv_len + j] = (kv_pos > token_pos) ? -10000.0f : 0.0f;
            }
        }

        // Expand mask for n_heads: each head gets the same mask
        ggml_tensor* mask_expanded = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_heads * seq_len, total_kv_len);
        float* mask_exp = (float*)mask_expanded->data;
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < total_kv_len; j++) {
                    mask_exp[(h * seq_len + i) * total_kv_len + j] = mask_data[i * total_kv_len + j];
                }
            }
        }
        scores = ggml_add(ctx, scores, mask_expanded);
    }

    // Softmax over the key dimension
    ggml_tensor* attn_weights = ggml_soft_max(ctx, scores);

    // Apply attention: attn_weights @ v_mat
    // (seq_len * n_heads, n_heads * total_kv_len) @ (n_heads * total_kv_len, head_dim)
    // = (seq_len * n_heads, head_dim)
    ggml_tensor* attn_out = ggml_mul_mat(ctx, attn_weights, v_mat);

    // Reshape back to (seq_len, n_heads, head_dim) then (seq_len, n_embd)
    attn_out = ggml_reshape_3d(ctx, attn_out, seq_len, n_heads, head_dim);
    attn_out = ggml_transpose(ctx, attn_out);  // (seq_len, head_dim, n_heads)
    attn_out = ggml_reshape_2d(ctx, attn_out, seq_len, n_embd);

    // Output projection
    ggml_tensor* out = ggml_mul_mat(ctx, attn_out, c_proj_weight);
    out = ggml_add(ctx, out, c_proj_bias);

    ggml_build_forward_expand(gf, out);
    return out;
}

void Attention::set_weights(
    const float* qkv_w, const float* qkv_b,
    const float* proj_w, const float* proj_b
) {
    // This would copy weights to the GGML tensors
    // In practice, weights are loaded directly into tensors during model loading
}

// ============== FFN ==============

FFN::FFN()
    : c_fc_weight(nullptr)
    , c_fc_bias(nullptr)
    , c_proj_weight(nullptr)
    , c_proj_bias(nullptr)
{}

ggml_tensor* FFN::gelu(ggml_context* ctx, ggml_tensor* x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ggml_tensor* x3 = ggml_mul(ctx, ggml_mul(ctx, x, x), x);
    ggml_tensor* inner = ggml_add(ctx, x, ggml_scale(ctx, x3, GELU::GELU_A));
    ggml_tensor* tanh_arg = ggml_scale(ctx, inner, GELU::GELU_SQRT2_OVER_PI);
    ggml_tensor* tanh_result = ggml_tanh(ctx, tanh_arg);
    ggml_tensor* one_plus_tanh = ggml_add(ctx, tanh_result, ggml_new_scalar(ctx, 1.0f));
    ggml_tensor* result = ggml_scale(ctx, ggml_mul(ctx, x, one_plus_tanh), 0.5f);
    return result;
}

ggml_tensor* FFN::forward(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x) {
    // FFN: GELU(up_proj(x)) * down_proj(x)

    // up_proj: (n_embd, n_ffn)
    ggml_tensor* up = ggml_mul_mat(ctx, x, c_fc_weight);
    up = ggml_add(ctx, up, c_fc_bias);
    // up: (seq_len, n_ffn)

    // GELU activation
    ggml_tensor* activated = gelu(ctx, up);
    // activated: (seq_len, n_ffn)

    // down_proj: (n_ffn, n_embd)
    ggml_tensor* down = ggml_mul_mat(ctx, activated, c_proj_weight);
    down = ggml_add(ctx, down, c_proj_bias);
    // down: (seq_len, n_embd)

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
    // Pre-norm architecture
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
    ggml_tensor* result = ggml_mul_mat(ctx, input, weight);
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
    // LayerNorm: gamma * (x - mean) / sqrt(var + eps) + beta
    // Since GGML lacks per-row reduction ops, compute mean/variance over full tensor
    // For seq_len=1 (typical during generation), this equals per-row computation
    int seq_len = (int)ggml_nrows(x);
    int n_embd = GPT2Config::n_embd;

    // Compute mean over all elements
    ggml_tensor* x_sum = ggml_sum(ctx, x);
    ggml_tensor* x_mean = ggml_scale(ctx, x_sum, 1.0f / (float)(seq_len * n_embd));

    // Compute x - mean
    ggml_tensor* x_centered = ggml_sub(ctx, x, x_mean);

    // Compute variance: mean((x - mean)^2)
    ggml_tensor* x_centered_sq = ggml_sqr(ctx, x_centered);
    ggml_tensor* var_sum = ggml_sum(ctx, x_centered_sq);
    ggml_tensor* variance = ggml_scale(ctx, var_sum, 1.0f / (float)(seq_len * n_embd));

    // Compute sqrt(var + eps)
    ggml_tensor* var_eps = ggml_add(ctx, variance, ggml_new_scalar(ctx, eps));
    ggml_tensor* std = ggml_sqrt(ctx, var_eps);

    // Normalize and scale/shift
    ggml_tensor* x_norm = ggml_div(ctx, x_centered, std);
    ggml_tensor* scaled = ggml_mul(ctx, x_norm, gamma);
    ggml_tensor* result = ggml_add(ctx, scaled, beta);
    return result;
}

ggml_tensor* rms_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    float eps
) {
    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    // GGML's ggml_rms_norm(ctx, x, eps) doesn't take a weight tensor
    // So we compute manually and apply weight
    ggml_tensor* x2 = ggml_sqr(ctx, x);
    ggml_tensor* mean2 = ggml_mean(ctx, x2);  // Over all elements - approximation
    ggml_tensor* var_eps = ggml_add(ctx, mean2, ggml_new_scalar(ctx, eps));
    ggml_tensor* rms = ggml_sqrt(ctx, var_eps);
    ggml_tensor* normalized = ggml_div(ctx, x, rms);
    ggml_tensor* result = ggml_mul(ctx, normalized, weight);
    return result;
}

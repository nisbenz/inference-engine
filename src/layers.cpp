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
    // x shape: (n_rows, n_cols) = (seq_len, n_embd)

    int n_rows = (int)x->ne[0];
    int n_cols = (int)x->ne[1];

    // Compute mean over all elements
    ggml_tensor* x_sum = ggml_sum(ctx, x);
    ggml_tensor* x_mean = ggml_scale(ctx, x_sum, 1.0f / (float)(n_rows * n_cols));

    // Compute x - mean and variance
    ggml_tensor* x_centered = ggml_sub(ctx, x, x_mean);
    ggml_tensor* x_centered_sq = ggml_sqr(ctx, x_centered);
    ggml_tensor* var_sum = ggml_sum(ctx, x_centered_sq);
    ggml_tensor* variance = ggml_scale(ctx, var_sum, 1.0f / (float)(n_rows * n_cols));

    // Compute std = sqrt(var + eps)
    ggml_tensor* var_eps = ggml_add(ctx, variance, ggml_new_scalar(ctx, GPT2Config::layer_norm_eps));
    ggml_tensor* std = ggml_sqrt(ctx, var_eps);

    // Normalize and scale/shift
    ggml_tensor* x_norm = ggml_div(ctx, x_centered, std);

    // gamma and beta are 1D (n_cols,), need to reshape to 2D (1, n_cols) for broadcasting
    ggml_tensor* gamma_2d = ggml_reshape_2d(ctx, gamma, 1, n_cols);
    ggml_tensor* beta_2d = ggml_reshape_2d(ctx, beta, 1, n_cols);

    ggml_tensor* scaled = ggml_mul(ctx, x_norm, gamma_2d);
    ggml_tensor* result = ggml_add(ctx, scaled, beta_2d);
    return result;
}

// ============== RMSNorm ==============

ggml_tensor* RMSNorm::forward(ggml_context* ctx, ggml_tensor* x) {
    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    ggml_tensor* x2 = ggml_sqr(ctx, x);
    ggml_tensor* mean2 = ggml_mean(ctx, x2);
    ggml_tensor* rms_eps = ggml_add(ctx, mean2, ggml_new_scalar(ctx, GPT2Config::layer_norm_eps));
    ggml_tensor* rms = ggml_sqrt(ctx, rms_eps);
    // No need for explicit repeat - ggml_div handles broadcasting
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

    // QKV projection: (seq_len, n_embd) @ (n_embd, 3*n_embd) = (seq_len, 3*n_embd)
    //
    // GGML's ggml_mul_mat requires BOTH tensors to have the same ne[0] (first dimension).
    // This is different from standard matmul!
    // For standard A @ B = C where A is (m,n) and B is (n,k), we get C is (m,k).
    // GGML expects: A has ne=(k,m), B has ne=(k,n), result ne=(m,n).
    //
    // Our x is (seq_len, n_embd) = (16, 768) and weight is (n_embd, 3*n_embd) = (768, 2304)
    // To use GGML: transpose x to (n_embd, seq_len) = (768, 16), weight stays (768, 2304)
    // Then GGML matmul gives (seq_len, 3*n_embd) = (16, 2304)

    ggml_tensor* x_t = ggml_transpose(ctx, x);
    ggml_tensor* x_reshaped = ggml_reshape_2d(ctx, x_t, x->ne[1], x->ne[0]); // (n_embd, seq_len)
    ggml_tensor* w_reshaped = ggml_reshape_2d(ctx, c_attn_weight, c_attn_weight->ne[0], c_attn_weight->ne[1]); // (n_embd, 3*n_embd)

    ggml_tensor* qkv = ggml_mul_mat(ctx, x_reshaped, w_reshaped);
    qkv = ggml_add(ctx, qkv, c_attn_bias);

    // Split into q, k, v - each (n_embd, seq_len) in row-major
    // qkv is (3*n_embd, seq_len), we split by rows
    ggml_tensor* q = ggml_view_2d(ctx, qkv, n_embd, seq_len, n_embd * sizeof(float), 0);
    ggml_tensor* k = ggml_view_2d(ctx, qkv, n_embd, seq_len, n_embd * sizeof(float), n_embd * sizeof(float));
    ggml_tensor* v = ggml_view_2d(ctx, qkv, n_embd, seq_len, n_embd * sizeof(float), 2 * n_embd * sizeof(float));

    // Handle KV cache storage and retrieval
    // Cache is complex - skip it for now and just use k, v directly
    int total_kv_len = seq_len;
    // k, v are (n_embd, seq_len) where n_embd = n_heads * head_dim
    // Memory layout: for head h, dimension d: element at row (h*head_dim + d)
    //
    // Simple per-head attention: for each head h
    //   Q_h = rows h*head_dim to (h+1)*head_dim of q     -> (head_dim, seq_len)
    //   K_h = rows h*head_dim to (h+1)*head_dim of k     -> (head_dim, total_kv_len)
    //   V_h = rows h*head_dim to (h+1)*head_dim of v     -> (head_dim, total_kv_len)
    //
    //   scores_h = Q_h^T @ K_h  -> (seq_len, total_kv_len) [GGML: same ne[0]=head_dim]
    //   scores_h = mask + softmax
    //   attn_h = scores_h @ V_h^T -> (seq_len, head_dim)

    // Allocate output: (seq_len, n_embd) in row-major
    ggml_tensor* attn_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, n_embd);

    // Loop over each head
    for (int h = 0; h < n_heads; h++) {
        int head_offset = h * head_dim;

        // Extract Q_h: rows head_offset to head_offset+head_dim from q
        // q: (n_embd, seq_len), we want rows [head_offset, head_offset+head_dim), all seq_len cols
        // This view is non-contiguous (stride n_embd instead of head_dim), so copy to contiguous tensor
        ggml_tensor* q_h_view = ggml_view_2d(ctx, q,
                                        head_dim, seq_len,
                                        q->ne[0] * sizeof(float),  // row stride = n_embd * sizeof(float)
                                        head_offset * sizeof(float)); // byte offset to start of head

        // K_h: same extraction from k
        ggml_tensor* k_h_view = ggml_view_2d(ctx, k,
                                        head_dim, total_kv_len,
                                        k->ne[0] * sizeof(float),
                                        head_offset * sizeof(float));

        // V_h: same extraction from v
        ggml_tensor* v_h_view = ggml_view_2d(ctx, v,
                                        head_dim, total_kv_len,
                                        v->ne[0] * sizeof(float),
                                        head_offset * sizeof(float));

        // Copy views to contiguous tensors (ggml_cpy works element-by-element)
        ggml_tensor* q_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, seq_len);
        q_h = ggml_cpy(ctx, q_h_view, q_h);

        ggml_tensor* k_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, total_kv_len);
        k_h = ggml_cpy(ctx, k_h_view, k_h);

        ggml_tensor* v_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, total_kv_len);
        v_h = ggml_cpy(ctx, v_h_view, v_h);

        // Q_h is (head_dim, seq_len), K_h is (head_dim, total_kv_len)
        // For GGML matmul: same ne[0] on both -> result is (seq_len, total_kv_len)
        ggml_tensor* scores_h = ggml_mul_mat(ctx, q_h, k_h);

        // Scale by 1/sqrt(head_dim)
        scores_h = ggml_scale(ctx, scores_h, 1.0f / std::sqrt((float)head_dim));

        // Apply causal mask
        ggml_tensor* mask_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, total_kv_len);
        float* mask_data = (float*)mask_h->data;
        for (int i = 0; i < seq_len; i++) {
            int token_pos = position + i;
            for (int j = 0; j < total_kv_len; j++) {
                size_t idx = i + j * seq_len;  // column-major
                mask_data[idx] = (j > token_pos) ? -10000.0f : 0.0f;
            }
        }
        scores_h = ggml_add(ctx, scores_h, mask_h);

        // Softmax
        ggml_tensor* attn_h = ggml_soft_max(ctx, scores_h);

        // attn_h: (seq_len, total_kv_len), V_h: (head_dim, total_kv_len)
        // For GGML: need same ne[0], so transpose attn_h to (total_kv_len, seq_len)
        // Result: (seq_len, head_dim)
        ggml_tensor* attn_h_t = ggml_transpose(ctx, attn_h);
        ggml_tensor* output_h = ggml_mul_mat(ctx, attn_h_t, v_h);

        // Copy output_h into attn_out at column offset h*head_dim
        ggml_tensor* out_cols = ggml_view_2d(ctx, attn_out,
                                             seq_len, head_dim,
                                             n_embd * sizeof(float),
                                             head_offset * sizeof(float));
        ggml_tensor* copy_h = ggml_cpy(ctx, output_h, out_cols);
        ggml_build_forward_expand(gf, copy_h);
    }

    // attn_out is now (seq_len, n_embd)

    // Output projection: (seq_len, n_embd) @ (n_embd, n_embd) = (seq_len, n_embd)
    // For GGML matmul: transpose attn_out to (n_embd, seq_len)
    ggml_tensor* attn_out_t = ggml_transpose(ctx, attn_out);
    ggml_tensor* attn_out_reshaped = ggml_reshape_2d(ctx, attn_out_t, attn_out->ne[1], attn_out->ne[0]); // (n_embd, seq_len)
    ggml_tensor* proj_reshaped = ggml_reshape_2d(ctx, c_proj_weight, c_proj_weight->ne[0], c_proj_weight->ne[1]); // (n_embd, n_embd)

    ggml_tensor* out = ggml_mul_mat(ctx, attn_out_reshaped, proj_reshaped);
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
    //
    // For GGML matmul, we need to transpose x since GGML requires same ne[0] for both tensors.
    // x: (seq_len, n_embd) -> transpose to (n_embd, seq_len)
    // up_proj weight: (n_embd, n_ffn)
    // After GGML matmul: (seq_len, n_ffn)

    ggml_tensor* x_t = ggml_transpose(ctx, x);
    ggml_tensor* x_reshaped = ggml_reshape_2d(ctx, x_t, x->ne[1], x->ne[0]); // (n_embd, seq_len)

    ggml_tensor* up_reshaped = ggml_reshape_2d(ctx, c_fc_weight, c_fc_weight->ne[0], c_fc_weight->ne[1]); // (n_embd, n_ffn)
    ggml_tensor* up = ggml_mul_mat(ctx, x_reshaped, up_reshaped);
    up = ggml_add(ctx, up, c_fc_bias);
    // up: (seq_len, n_ffn)

    // GELU activation
    ggml_tensor* activated = gelu(ctx, up);
    // activated: (seq_len, n_ffn)

    // down_proj: (n_ffn, n_embd)
    ggml_tensor* activated_t = ggml_transpose(ctx, activated);
    ggml_tensor* activated_reshaped = ggml_reshape_2d(ctx, activated_t, activated->ne[1], activated->ne[0]); // (n_ffn, seq_len)

    ggml_tensor* down_reshaped = ggml_reshape_2d(ctx, c_proj_weight, c_proj_weight->ne[0], c_proj_weight->ne[1]); // (n_ffn, n_embd)
    ggml_tensor* down = ggml_mul_mat(ctx, activated_reshaped, down_reshaped);
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
    // x shape: (n_rows, n_cols) = (seq_len, n_embd)
    // For seq_len=1, we can use per-token norm which is same as full tensor

    int n_rows = (int)x->ne[0];  // number of rows (tokens)
    int n_cols = (int)x->ne[1];  // number of columns (embedding dim)

    // Compute mean over all elements: sum(x) / (n_rows * n_cols)
    ggml_tensor* x_sum = ggml_sum(ctx, x);
    ggml_tensor* x_mean = ggml_scale(ctx, x_sum, 1.0f / (float)(n_rows * n_cols));

    // Compute variance: sum((x - mean)^2) / (n_rows * n_cols)
    ggml_tensor* x_centered = ggml_sub(ctx, x, x_mean);
    ggml_tensor* x_centered_sq = ggml_sqr(ctx, x_centered);
    ggml_tensor* var_sum = ggml_sum(ctx, x_centered_sq);
    ggml_tensor* variance = ggml_scale(ctx, var_sum, 1.0f / (float)(n_rows * n_cols));

    // Compute std = sqrt(var + eps)
    ggml_tensor* var_eps = ggml_add(ctx, variance, ggml_new_scalar(ctx, eps));
    ggml_tensor* std = ggml_sqrt(ctx, var_eps);

    // Normalize: (x - mean) / std
    ggml_tensor* x_norm = ggml_div(ctx, x_centered, std);

    // Scale and shift: gamma * x_norm + beta
    // gamma and beta are 1D (n_cols,), need to reshape to 2D (1, n_cols) for broadcasting
    ggml_tensor* gamma_2d = ggml_reshape_2d(ctx, gamma, 1, n_cols);
    ggml_tensor* beta_2d = ggml_reshape_2d(ctx, beta, 1, n_cols);

    ggml_tensor* scaled = ggml_mul(ctx, x_norm, gamma_2d);
    ggml_tensor* result = ggml_add(ctx, scaled, beta_2d);
    return result;
}

ggml_tensor* rms_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    float eps
) {
    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    ggml_tensor* x2 = ggml_sqr(ctx, x);
    ggml_tensor* mean2 = ggml_mean(ctx, x2);
    ggml_tensor* var_eps = ggml_add(ctx, mean2, ggml_new_scalar(ctx, eps));
    ggml_tensor* rms = ggml_sqrt(ctx, var_eps);
    // No need for explicit repeat - ggml_div handles broadcasting
    ggml_tensor* normalized = ggml_div(ctx, x, rms);
    ggml_tensor* result = ggml_mul(ctx, normalized, weight);
    return result;
}

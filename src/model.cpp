#include "model.hpp"
#include "gguf_loader.h"
#include <ggml-cpu.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>

// ============== GPT2Model ==============

GPT2Model::GPT2Model()
    : ctx_(nullptr)
    , gf_(nullptr)
    , backend_(nullptr)
    , allocr_(nullptr)
    , buffer_w_(nullptr)
    , use_gpu_(false)
    , wte_(nullptr)
    , wpe_(nullptr)
    , lm_head_(nullptr)
    , logits_data_(nullptr)
    , logits_size_(0)
    , logits_tensor_(nullptr)
    , input_ids_tensor_(nullptr)
    , position_tensor_(nullptr)
{
    layers_.resize(N_LAYERS);
}

GPT2Model::~GPT2Model() {
    if (ctx_) {
        ggml_free(ctx_);
    }
    if (logits_data_) {
        delete[] logits_data_;
    }
    if (allocr_) {
        ggml_gallocr_free(allocr_);
    }
    if (buffer_w_) {
        ggml_backend_buffer_free(buffer_w_);
    }
    if (backend_) {
        ggml_backend_free(backend_);
    }
}

bool GPT2Model::init(bool use_gpu) {
    use_gpu_ = use_gpu;

    // Initialize the backend first
    backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend_) {
        std::cerr << "Failed to initialize GGML backend" << std::endl;
        return false;
    }

    // Calculate weight buffer size
    size_t buffer_size = 0;
    buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD); // ln_f_gamma
    buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD); // ln_f_beta
    buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD * VOCAB_SIZE); // wte
    buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD * CONTEXT_LENGTH); // wpe
    buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD * VOCAB_SIZE); // lm_head (tied to wte)

    for (int i = 0; i < N_LAYERS; i++) {
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD); // ln1_gamma
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD); // ln1_beta
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD); // ln2_gamma
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD); // ln2_beta
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD * 3 * N_EMBD); // c_attn_weight
        buffer_size += ggml_row_size(GGML_TYPE_F32, 3 * N_EMBD); // c_attn_bias
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD * N_EMBD); // c_proj_weight
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD); // c_proj_bias
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD * N_FFN); // c_fc_weight
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_FFN); // c_fc_bias
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_FFN * N_EMBD); // c_proj_weight
        buffer_size += ggml_row_size(GGML_TYPE_F32, N_EMBD); // c_proj_bias
    }
    buffer_size += (6 + 12 * N_LAYERS) * 128; // alignment overhead

    // Calculate total tensor count: model tensors + KV cache tensors
    // Model: 4 base + 12 per layer = 4 + 12*12 = 148
    // KV cache: 2 per layer = 2*12 = 24
    // Total: 172, add buffer for safety
    size_t n_tensors = 4 + 12 * N_LAYERS + 2 * N_LAYERS + 32;

    // Initialize GGML context with no_alloc=true
    struct ggml_init_params params = {
        .mem_size   = ggml_tensor_overhead() * n_tensors,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ctx_ = ggml_init(params);
    if (!ctx_) {
        std::cerr << "Failed to initialize GGML context" << std::endl;
        return false;
    }

    // Allocate weight buffer
    buffer_w_ = ggml_backend_alloc_buffer(backend_, buffer_size);

    // Initialize KV cache
    kv_cache_.init(ctx_);

    // Allocate model weights (will be filled by load_weights)
    // wte: (VOCAB_SIZE, N_EMBD) = (50257, 768)
    wte_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, VOCAB_SIZE);
    ggml_set_name(wte_, "wte");

    // wpe: (CONTEXT_LENGTH, N_EMBD) = (1024, 768)
    wpe_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, CONTEXT_LENGTH);
    ggml_set_name(wpe_, "wpe");

    // Final layer norm gamma and beta
    ln_f_.gamma = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
    ln_f_.beta = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
    ggml_set_name(ln_f_.gamma, "ln_f_gamma");
    ggml_set_name(ln_f_.beta, "ln_f_beta");

    // LM head (tied to wte)
    lm_head_ = wte_;  // Tie weights

    // Allocate weight tensors into buffer
    ggml_tallocr alloc = ggml_tallocr_new(buffer_w_);
    ggml_tallocr_alloc(&alloc, wte_);
    ggml_tallocr_alloc(&alloc, wpe_);
    ggml_tallocr_alloc(&alloc, ln_f_.gamma);
    ggml_tallocr_alloc(&alloc, ln_f_.beta);

    // Initialize transformer layers
    for (int i = 0; i < N_LAYERS; i++) {
        // LayerNorm 1
        layers_[i].ln1.gamma = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        layers_[i].ln1.beta = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        ggml_set_name(layers_[i].ln1.gamma, ("ln1_gamma_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ln1.beta, ("ln1_beta_" + std::to_string(i)).c_str());

        // Attention weights
        layers_[i].attention.c_attn_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, 3 * N_EMBD);
        layers_[i].attention.c_attn_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 3 * N_EMBD);
        layers_[i].attention.c_proj_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, N_EMBD);
        layers_[i].attention.c_proj_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);

        ggml_set_name(layers_[i].attention.c_attn_weight, ("attn_c_attn_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].attention.c_attn_bias, ("attn_c_attn_bias_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].attention.c_proj_weight, ("attn_c_proj_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].attention.c_proj_bias, ("attn_c_proj_bias_" + std::to_string(i)).c_str());

        // LayerNorm 2
        layers_[i].ln2.gamma = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        layers_[i].ln2.beta = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        ggml_set_name(layers_[i].ln2.gamma, ("ln2_gamma_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ln2.beta, ("ln2_beta_" + std::to_string(i)).c_str());

        // FFN weights
        layers_[i].ffn.c_fc_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, N_FFN);
        layers_[i].ffn.c_fc_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_FFN);
        layers_[i].ffn.c_proj_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_FFN, N_EMBD);
        layers_[i].ffn.c_proj_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);

        ggml_set_name(layers_[i].ffn.c_fc_weight, ("ffn_c_fc_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ffn.c_fc_bias, ("ffn_c_fc_bias_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ffn.c_proj_weight, ("ffn_c_proj_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ffn.c_proj_bias, ("ffn_c_proj_bias_" + std::to_string(i)).c_str());

        // Allocate layer tensors
        ggml_tallocr_alloc(&alloc, layers_[i].ln1.gamma);
        ggml_tallocr_alloc(&alloc, layers_[i].ln1.beta);
        ggml_tallocr_alloc(&alloc, layers_[i].attention.c_attn_weight);
        ggml_tallocr_alloc(&alloc, layers_[i].attention.c_attn_bias);
        ggml_tallocr_alloc(&alloc, layers_[i].attention.c_proj_weight);
        ggml_tallocr_alloc(&alloc, layers_[i].attention.c_proj_bias);
        ggml_tallocr_alloc(&alloc, layers_[i].ln2.gamma);
        ggml_tallocr_alloc(&alloc, layers_[i].ln2.beta);
        ggml_tallocr_alloc(&alloc, layers_[i].ffn.c_fc_weight);
        ggml_tallocr_alloc(&alloc, layers_[i].ffn.c_fc_bias);
        ggml_tallocr_alloc(&alloc, layers_[i].ffn.c_proj_weight);
        ggml_tallocr_alloc(&alloc, layers_[i].ffn.c_proj_bias);
    }

    // Create graph allocator (will be used for compute graph temporary buffers)
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));

    // Initialize logits buffer
    logits_size_ = VOCAB_SIZE;
    logits_data_ = new float[logits_size_];
    memset(logits_data_, 0, logits_size_ * sizeof(float));

    std::cout << "GPT-2 Large model initialized" << std::endl;
    std::cout << "  Layers: " << N_LAYERS << std::endl;
    std::cout << "  Hidden: " << N_EMBD << std::endl;
    std::cout << "  Heads: " << N_HEADS << std::endl;
    std::cout << "  FFN: " << N_FFN << std::endl;
    std::cout << "  Vocab: " << VOCAB_SIZE << std::endl;

    return true;
}

bool GPT2Model::load_weights(const std::string& path) {
    // Try GGUF format first (most common for ggml-based models)
    if (path.find(".gguf") != std::string::npos ||
        path.find(".bin") != std::string::npos) {
        return load_gguf_weights(path);
    }
    // Fallback to legacy ggml format
    return load_ggml_weights(path);
}


// Forward declarations for helper functions
static size_t extract_layer_idx(const std::string& name);
static float fp16_to_fp32(uint16_t f16);
static float bf16_to_fp32(uint16_t bf16);

bool GPT2Model::load_gguf_weights(const std::string& path) {
    std::cout << "Loading GGUF model from: " << path << std::endl;

    try {
        GGUFFile gguf = load_gguf(path.c_str());

        // Print model info from metadata
        std::string model_name = gguf.get_str("general.architecture", "unknown");
        std::cout << "Model architecture: " << model_name << std::endl;

        uint32_t n_ctx = gguf.get_u32("llama.context_length", gguf.get_u32("gpt2.context_length", 1024));
        uint32_t n_embd = gguf.get_u32("llama.embedding_length", gguf.get_u32("gpt2.embedding_length", 768));
        uint32_t n_head = gguf.get_u32("llama.attention.head_count", gguf.get_u32("gpt2.attention.head_count", 12));
        uint32_t n_layer = gguf.get_u32("llama.block_count", gguf.get_u32("gpt2.block_count", 12));
        uint32_t n_ffn = gguf.get_u32("llama.feed_forward_length", gguf.get_u32("gpt2.feed_forward_length", 3072));

        std::cout << "Config: ctx=" << n_ctx << " embd=" << n_embd << " head=" << n_head
                  << " layer=" << n_layer << " ffn=" << n_ffn << std::endl;

        // Print first few tensor names for verification
        std::cout << "\nSample tensor names:" << std::endl;
        for (size_t i = 0; i < gguf.tensors.size() && i < 5; i++) {
            auto& t = gguf.tensors[i];
            std::cout << "  " << t.name << std::endl;
        }
        std::cout << "  ... (" << gguf.tensors.size() << " total tensors)" << std::endl;

        // Map tensor names to our model weights
        // Tensor names in this GGUF file use blk.X.* pattern:
        //   blk.X.attn_qkv.weight/bias
        //   blk.X.attn_output.weight/bias
        //   blk.X.attn_norm.weight/bias (ln1)
        //   blk.X.ffn_norm.weight/bias (ln2)
        //   blk.X.ffn_up.weight/bias
        //   blk.X.ffn_down.weight/bias
        //   token_embd.weight, position_embd.weight
        //   output_norm.weight/bias

        std::cout << "\nLoading tensors..." << std::endl;

        int loaded = 0;
        int failed = 0;

        for (auto& t : gguf.tensors) {
            ggml_tensor* dst = nullptr;
            size_t layer_idx = extract_layer_idx(t.name);

            // Token embeddings
            if (t.name == "token_embd.weight") {
                dst = wte_;
            }
            // Position embeddings
            else if (t.name == "position_embd.weight") {
                dst = wpe_;
            }
            // Final layer norm
            else if (t.name == "output_norm.weight") {
                dst = ln_f_.gamma;
            }
            else if (t.name == "output_norm.bias") {
                dst = ln_f_.beta;
            }
            // Layer tensors (blk.X.* pattern)
            else if (layer_idx < N_LAYERS) {
                if (t.name.find(".attn_qkv.weight") != std::string::npos) {
                    dst = layers_[layer_idx].attention.c_attn_weight;
                } else if (t.name.find(".attn_qkv.bias") != std::string::npos) {
                    dst = layers_[layer_idx].attention.c_attn_bias;
                } else if (t.name.find(".attn_output.weight") != std::string::npos) {
                    dst = layers_[layer_idx].attention.c_proj_weight;
                } else if (t.name.find(".attn_output.bias") != std::string::npos) {
                    dst = layers_[layer_idx].attention.c_proj_bias;
                } else if (t.name.find(".attn_norm.weight") != std::string::npos) {
                    dst = layers_[layer_idx].ln1.gamma;
                } else if (t.name.find(".attn_norm.bias") != std::string::npos) {
                    dst = layers_[layer_idx].ln1.beta;
                } else if (t.name.find(".ffn_norm.weight") != std::string::npos) {
                    dst = layers_[layer_idx].ln2.gamma;
                } else if (t.name.find(".ffn_norm.bias") != std::string::npos) {
                    dst = layers_[layer_idx].ln2.beta;
                } else if (t.name.find(".ffn_up.weight") != std::string::npos) {
                    dst = layers_[layer_idx].ffn.c_fc_weight;
                } else if (t.name.find(".ffn_up.bias") != std::string::npos) {
                    dst = layers_[layer_idx].ffn.c_fc_bias;
                } else if (t.name.find(".ffn_down.weight") != std::string::npos) {
                    dst = layers_[layer_idx].ffn.c_proj_weight;
                } else if (t.name.find(".ffn_down.bias") != std::string::npos) {
                    dst = layers_[layer_idx].ffn.c_proj_bias;
                }
            }

            if (dst) {
                // Calculate expected size
                size_t expected_nbytes = ggml_nbytes(dst);
                size_t actual_nbytes = gguf_tensor_nbytes(t);

                // GGUF row-major [R, C] and GGML column-major [C, R] have the SAME flat memory layout.
                // Element (r, c) is at index r*C+c in both formats.
                // No transpose needed — direct memcpy is correct.

                if (expected_nbytes == actual_nbytes || t.type == GGUF_TID_Q4_K || t.type == GGUF_TID_Q8_0_ALT || t.type == GGUF_TID_BF16 || t.type == GGUF_TID_F16) {
                    
                    std::vector<float> buffer_f(ggml_nelements(dst));
                    bool read_success = false;

                    // Type matches or quantized (will be handled separately)
                    if (t.type == GGUF_TID_F32) {
                        read_tensor_data(gguf, t, buffer_f.data(), actual_nbytes);
                        read_success = true;
                        loaded++;
                    } else if (t.type == GGUF_TID_F16) {
                        // Convert F16 to F32
                        std::vector<uint16_t> f16_data(actual_nbytes / 2);
                        read_tensor_data(gguf, t, f16_data.data(), actual_nbytes);
                        for (size_t j = 0; j < f16_data.size(); j++) {
                            buffer_f[j] = fp16_to_fp32(f16_data[j]);
                        }
                        read_success = true;
                        loaded++;
                    } else if (t.type == GGUF_TID_Q4_K) {
                        // Q4_K_M quantization - needs special dequantization
                        std::cout << "  Warning: Q4_K tensor '" << t.name << "' needs dequantization, using random" << std::endl;
                        failed++;
                    } else if (t.type == GGUF_TID_Q8_0_ALT) {
                        // Q8_0_ALT (type 30) in this GGUF file: actually stored as BF16 (2 bytes per element)
                        // Confirmed by tensor data size: 768*2304*2 = 3,538,944 bytes
                        std::vector<uint16_t> bf16_data(ggml_nelements(dst));
                        read_tensor_data(gguf, t, bf16_data.data(), actual_nbytes);
                        for (size_t j = 0; j < bf16_data.size(); j++) {
                            buffer_f[j] = bf16_to_fp32(bf16_data[j]);
                        }
                        read_success = true;
                        loaded++;
                    } else if (t.type == GGUF_TID_BF16) {
                        // BF16 type - convert to F32
                        std::vector<uint16_t> bf16_data(actual_nbytes / 2);
                        read_tensor_data(gguf, t, bf16_data.data(), actual_nbytes);
                        for (size_t j = 0; j < bf16_data.size(); j++) {
                            buffer_f[j] = bf16_to_fp32(bf16_data[j]);
                        }
                        read_success = true;
                        loaded++;
                    } else {
                        std::cout << "  Warning: Unsupported type " << t.type << " for tensor '" << t.name << "'" << std::endl;
                        failed++;
                    }
                    
                    if (read_success) {
                        auto* dst_ptr = (float*)dst->data;
                        std::memcpy(dst_ptr, buffer_f.data(), buffer_f.size() * sizeof(float));
                    }
                } else {
                    std::cout << "  Size mismatch for '" << t.name << "': expected " << expected_nbytes << " got " << actual_nbytes << std::endl;
                    failed++;
                }
            }
        }

        std::cout << "\nLoaded " << loaded << " tensors, " << failed << " failed/skipped" << std::endl;

        // Debug: verify weights are non-zero
        {
            const float* wte_data = (const float*)wte_->data;
            float wte_sum = 0, wte_max = 0;
            for (size_t i = 0; i < ggml_nelements(wte_); i++) {
                wte_sum += std::abs(wte_data[i]);
                if (std::abs(wte_data[i]) > wte_max) wte_max = std::abs(wte_data[i]);
            }
            std::cout << "[DEBUG] WTE: sum=" << wte_sum << " max=" << wte_max << " elements=" << ggml_nelements(wte_) << std::endl;

            if (N_LAYERS > 0) {
                const float* ln1_gamma = (const float*)layers_[0].ln1.gamma->data;
                float ln1_sum = 0;
                for (size_t i = 0; i < ggml_nelements(layers_[0].ln1.gamma); i++) {
                    ln1_sum += std::abs(ln1_gamma[i]);
                }
                std::cout << "[DEBUG] Layer 0 LN1 gamma: sum=" << ln1_sum << " elements=" << ggml_nelements(layers_[0].ln1.gamma) << std::endl;

                const float* qkv_w = (const float*)layers_[0].attention.c_attn_weight->data;
                float qkv_sum = 0, qkv_max = 0;
                for (size_t i = 0; i < ggml_nelements(layers_[0].attention.c_attn_weight); i++) {
                    qkv_sum += std::abs(qkv_w[i]);
                    if (std::abs(qkv_w[i]) > qkv_max) qkv_max = std::abs(qkv_w[i]);
                }
                std::cout << "[DEBUG] Layer 0 QKV weight: sum=" << qkv_sum << " max=" << qkv_max << " elements=" << ggml_nelements(layers_[0].attention.c_attn_weight) << std::endl;
            }
        }

        fclose(gguf.fp);
        std::cout << "GGUF model loaded successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error loading GGUF: " << e.what() << std::endl;
        return false;
    }
}

// Helper to extract layer index from tensor name
// Handles both "model.h.0.xxx" and "blk.0.xxx" patterns
// Returns SIZE_MAX if not a layer tensor (e.g., token_embd.weight, output_norm.weight)
static size_t extract_layer_idx(const std::string& name) {
    // Find .h. pattern (HuggingFace style)
    size_t h_pos = name.find(".h.");
    if (h_pos != std::string::npos) {
        size_t start = h_pos + 3;
        size_t end = name.find('.', start);
        if (end != std::string::npos) {
            return std::stoi(name.substr(start, end - start));
        }
    }
    // Find blk. pattern (llama.cpp style)
    size_t blk_pos = name.find("blk.");
    if (blk_pos != std::string::npos) {
        size_t start = blk_pos + 4;
        size_t end = name.find('.', start);
        if (end != std::string::npos) {
            return std::stoi(name.substr(start, end - start));
        }
    }
    return SIZE_MAX;  // Not a layer tensor - use SIZE_MAX as sentinel
}

// FP16 to FP32 conversion
static float fp16_to_fp32(uint16_t f16) {
    unsigned int sign = (f16 >> 15) & 0x1;
    unsigned int exp = (f16 >> 10) & 0x1f;
    unsigned int mant = f16 & 0x3ff;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        else return sign ? -pow(2, -14) * (mant / 1024.0f) : pow(2, -14) * (mant / 1024.0f);
    } else if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        else return NAN;
    }

    int32_t e = exp - 15;
    float m = 1.0f + mant / 1024.0f;
    return sign ? -pow(2, e) * m : pow(2, e) * m;
}

// BF16 (Brain Float 16) to FP32 conversion
// BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits
static float bf16_to_fp32(uint16_t bf16) {
    uint32_t val = (uint32_t)bf16 << 16;
    float result;
    std::memcpy(&result, &val, sizeof(float));
    return result;
}

bool GPT2Model::load_ggml_weights(const std::string& path) {
    std::cerr << "GGML weight format not yet implemented" << std::endl;
    return false;
}

std::vector<float> GPT2Model::forward(
    const std::vector<int>& input_ids,
    int position,
    bool use_cache
) {
    if (input_ids.empty()) {
        return std::vector<float>();
    }

    int seq_len = (int)input_ids.size();

    // Allocate a temporary ggml context for this forward pass graph
    struct ggml_init_params params0 = {
        .mem_size   = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false),
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context* ctx0 = ggml_init(params0);

    // Build computation graph
    build_graph(ctx0, input_ids, position, use_cache);
    
    // Create a fresh graph allocator for this forward pass
    // (graph structure changes with seq_len, so we can't reuse the allocator)
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    
    // Allocate the graph
    ggml_gallocr_alloc_graph(alloc, gf_);

    // Set input tensors
    struct ggml_tensor* inp_tokens = ggml_graph_get_tensor(gf_, "inp_tokens");
    if (inp_tokens) {
        ggml_backend_tensor_set(inp_tokens, input_ids.data(), 0, seq_len * sizeof(int32_t));
        std::cout << "[DEBUG] Set inp_tokens, seq_len=" << seq_len << std::endl;
    } else {
        std::cerr << "[ERROR] inp_tokens not found in graph!" << std::endl;
    }

    struct ggml_tensor* pos_tensor = ggml_graph_get_tensor(gf_, "position");
    if (pos_tensor) {
        std::vector<int32_t> pos_ids(seq_len);
        for (int i = 0; i < seq_len; i++) {
            int p = position + i;
            pos_ids[i] = (p >= CONTEXT_LENGTH) ? (CONTEXT_LENGTH - 1) : p;
        }
        for (int i = 0; i < seq_len; i++) {
            int32_t v = pos_ids[i];
            ggml_backend_tensor_set(pos_tensor, &v, i * sizeof(int32_t), sizeof(v));
        }
    }

    // Set backend options
    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, 4);
    }

    // Compute
    ggml_backend_graph_compute(backend_, gf_);

    // Extract logits for the last token only
    std::vector<float> logits(VOCAB_SIZE, 0.0f);

    struct ggml_tensor* logits_out = ggml_graph_get_tensor(gf_, "logits");
    if (logits_out) {
        // logits_out: ne[0]=VOCAB_SIZE, ne[1]=seq_len
        // Get the last token's logits
        ggml_backend_tensor_get(logits_out, logits.data(), (seq_len - 1) * VOCAB_SIZE * sizeof(float), VOCAB_SIZE * sizeof(float));
        
        // Debug: check logits
        float logit_sum = 0, logit_max = -1e30, logit_min = 1e30;
        int nan_count = 0;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            logit_sum += logits[i];
            if (logits[i] > logit_max) logit_max = logits[i];
            if (logits[i] < logit_min) logit_min = logits[i];
            if (std::isnan(logits[i]) || std::isinf(logits[i])) nan_count++;
        }
        std::cout << "[DEBUG] Logits: sum=" << logit_sum << " max=" << logit_max << " min=" << logit_min << " nan=" << nan_count << std::endl;
        
        // Print top 5 logits
        std::vector<std::pair<float, int>> top_logits;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            top_logits.push_back({logits[i], i});
        }
        std::partial_sort(top_logits.begin(), top_logits.begin() + 5, top_logits.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        std::cout << "[DEBUG] Top 5 logits: ";
        for (int i = 0; i < 5; i++) {
            std::cout << "token=" << top_logits[i].second << " logit=" << top_logits[i].first << " ";
        }
        std::cout << std::endl;
    } else {
        std::cerr << "[ERROR] logits tensor not found in graph!" << std::endl;
    }

    // Free the temporary context and allocator
    ggml_gallocr_free(alloc);
    ggml_free(ctx0);
    gf_ = nullptr;

    return logits;
}

void GPT2Model::build_graph(
    ggml_context* ctx0,
    const std::vector<int>& input_ids,
    int position,
    bool use_cache
) {
    // Allocate a fresh computation graph within the local context ctx0
    gf_ = ggml_new_graph_custom(ctx0, 4096, false);

    int seq_len = input_ids.size();

    // Build tokens tensor for row extraction
    ggml_tensor* tokens_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(tokens_tensor, "inp_tokens");
    ggml_set_input(tokens_tensor);

    // Build positional indices tensor
    ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos_tensor, "position");
    ggml_set_input(pos_tensor);

    // wte + wpe
    ggml_tensor* input_embd = ggml_get_rows(ctx0, wte_, tokens_tensor);
    ggml_tensor* pos_embd = ggml_get_rows(ctx0, wpe_, pos_tensor);
    ggml_tensor* h = ggml_add(ctx0, input_embd, pos_embd);
    // h: ne[0]=N_EMBD, ne[1]=seq_len
    
    // Pass through transformer layers
    for (int i = 0; i < N_LAYERS; i++) {
        h = layers_[i].forward(ctx0, gf_, h, position, use_cache);
        ggml_build_forward_expand(gf_, h);
    }

    // Final layer norm
    ggml_tensor* h_norm = layer_norm(ctx0, h, ln_f_.gamma, ln_f_.beta, GPT2Config::layer_norm_eps);
    ggml_build_forward_expand(gf_, h_norm);

    // LM head: lm_head^T @ h_norm
    // Result: ne[0]=VOCAB_SIZE, ne[1]=seq_len
    ggml_tensor* logits = ggml_mul_mat(ctx0, lm_head_, h_norm);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf_, logits);
}

std::vector<int> GPT2Model::generate(
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    int top_k
) {
    std::vector<int> tokens = prompt_tokens;

    // First forward pass message
    std::cout << "Processing prompt (" << tokens.size() << " tokens)..." << std::endl;

    // Subsequent passes: process the full sequence since KV cache isn't fully implemented
    while ((int)tokens.size() < (int)prompt_tokens.size() + max_new_tokens) {

        // Forward pass with full sequence, disable cache
        // Trim to last CONTEXT_LENGTH tokens if needed
        std::vector<int> window(tokens);
        if ((int)window.size() > CONTEXT_LENGTH)
            window = std::vector<int>(tokens.end() - CONTEXT_LENGTH, tokens.end());
        std::vector<float> logits = forward(window, 0, false);

        // Sample next token (use logits directly - they correspond to the single token)
        int next_token = sample(logits, temperature, top_k);

        // [DEBUG] Check what exactly is being sampled
        // std::cout << " [Sampled token ID: " << next_token << "] ";

        // Check for EOS
        if (next_token == EOS_TOKEN) {
            break;
        }

        // Append to sequence
        tokens.push_back(next_token);

        // Print progress
        std::cout << "." << std::flush;
    }

    std::cout << std::endl;
    return tokens;
}

int GPT2Model::sample(const std::vector<float>& logits, float temperature, int top_k) {
    // Apply temperature
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
    }

    // Top-k filtering
    if (top_k > 0 && top_k < (int)probs.size()) {
        std::vector<std::pair<float, int>> pairs;
        for (int i = 0; i < (int)probs.size(); i++) {
            pairs.push_back({probs[i], i});
        }
        std::partial_sort(pairs.begin(), pairs.begin() + top_k, pairs.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        float top_k_sum = 0;
        for (int i = 0; i < top_k; i++) {
            top_k_sum += pairs[i].first;
        }
        for (int i = 0; i < top_k; i++) {
            probs[pairs[i].second] /= top_k_sum;
        }
        for (int i = top_k; i < (int)probs.size(); i++) {
            probs[pairs[i].second] = 0;
        }
    }

    // Sample from distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());

    return dist(gen);
}

// ============== Helper Functions ==============

ggml_tensor* create_tensor_2d(
    ggml_context* ctx,
    const std::string& name,
    int rows,
    int cols,
    const float* data
) {
    ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    ggml_set_name(tensor, name.c_str());
    if (data) {
        memcpy(tensor->data, data, ggml_nbytes(tensor));
    }
    return tensor;
}

ggml_tensor* create_tensor_1d(
    ggml_context* ctx,
    const std::string& name,
    int size,
    const float* data
) {
    ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    ggml_set_name(tensor, name.c_str());
    if (data) {
        memcpy(tensor->data, data, ggml_nbytes(tensor));
    }
    return tensor;
}

// ============== MappedFile ==============

MappedFile::MappedFile(const std::string& path) : data(nullptr), size(0) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        // Note: For production, use mmap or equivalent
        // This is simplified
        data = malloc(size);
        if (data) {
            file.read((char*)data, size);
        }
    }
}

MappedFile::~MappedFile() {
    if (data) {
        free(data);
    }
}

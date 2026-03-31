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
}

bool GPT2Model::init(bool use_gpu) {
    use_gpu_ = use_gpu;

    // Initialize GGML
    struct ggml_init_params params = {
        .mem_size   = static_cast<size_t>(1024) * 1024 * 1024,  // 1 GB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };

    ctx_ = ggml_init(params);
    if (!ctx_) {
        std::cerr << "Failed to initialize GGML context" << std::endl;
        return false;
    }

    // Initialize KV cache
    kv_cache_.init(ctx_);

    // Allocate model weights (will be filled by load_weights)
    // wte: (VOCAB_SIZE, N_EMBD) = (50257, 768)
    wte_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, VOCAB_SIZE);
    ggml_set_name(wte_, "wte");
    memset(wte_->data, 0, ggml_nbytes(wte_));

    // wpe: (CONTEXT_LENGTH, N_EMBD) = (1024, 768)
    wpe_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, CONTEXT_LENGTH);
    ggml_set_name(wpe_, "wpe");
    memset(wpe_->data, 0, ggml_nbytes(wpe_));

    // Final layer norm gamma and beta
    ln_f_.gamma = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
    ln_f_.beta = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
    ggml_set_name(ln_f_.gamma, "ln_f_gamma");
    ggml_set_name(ln_f_.beta, "ln_f_beta");
    memset(ln_f_.gamma->data, 0, ggml_nbytes(ln_f_.gamma));
    memset(ln_f_.beta->data, 0, ggml_nbytes(ln_f_.beta));

    // LM head (tied to wte)
    lm_head_ = wte_;  // Tie weights

    // Initialize transformer layers
    for (int i = 0; i < N_LAYERS; i++) {
        // LayerNorm 1
        layers_[i].ln1.gamma = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        layers_[i].ln1.beta = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        ggml_set_name(layers_[i].ln1.gamma, ("ln1_gamma_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ln1.beta, ("ln1_beta_" + std::to_string(i)).c_str());
        memset(layers_[i].ln1.gamma->data, 0, ggml_nbytes(layers_[i].ln1.gamma));
        memset(layers_[i].ln1.beta->data, 0, ggml_nbytes(layers_[i].ln1.beta));

        // Attention weights
        // c_attn_weight: (3 * N_EMBD, N_EMBD) = (3840, 1280)
        layers_[i].attention.c_attn_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, 3 * N_EMBD);
        layers_[i].attention.c_attn_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 3 * N_EMBD);
        // c_proj_weight: (N_EMBD, N_EMBD) = (1280, 1280)
        layers_[i].attention.c_proj_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, N_EMBD);
        layers_[i].attention.c_proj_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);

        ggml_set_name(layers_[i].attention.c_attn_weight, ("attn_c_attn_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].attention.c_attn_bias, ("attn_c_attn_bias_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].attention.c_proj_weight, ("attn_c_proj_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].attention.c_proj_bias, ("attn_c_proj_bias_" + std::to_string(i)).c_str());

        memset(layers_[i].attention.c_attn_weight->data, 0, ggml_nbytes(layers_[i].attention.c_attn_weight));
        memset(layers_[i].attention.c_attn_bias->data, 0, ggml_nbytes(layers_[i].attention.c_attn_bias));
        memset(layers_[i].attention.c_proj_weight->data, 0, ggml_nbytes(layers_[i].attention.c_proj_weight));
        memset(layers_[i].attention.c_proj_bias->data, 0, ggml_nbytes(layers_[i].attention.c_proj_bias));

        // Initialize KV cache for this layer (disabled for now)
        // layers_[i].attention.init_cache(ctx_);

        // LayerNorm 2
        layers_[i].ln2.gamma = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        layers_[i].ln2.beta = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);
        ggml_set_name(layers_[i].ln2.gamma, ("ln2_gamma_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ln2.beta, ("ln2_beta_" + std::to_string(i)).c_str());
        memset(layers_[i].ln2.gamma->data, 0, ggml_nbytes(layers_[i].ln2.gamma));
        memset(layers_[i].ln2.beta->data, 0, ggml_nbytes(layers_[i].ln2.beta));

        // FFN weights
        // c_fc_weight: (N_FFN, N_EMBD) = (5120, 1280)
        layers_[i].ffn.c_fc_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_EMBD, N_FFN);
        layers_[i].ffn.c_fc_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_FFN);
        // c_proj_weight: (N_EMBD, N_FFN) = (1280, 5120)
        layers_[i].ffn.c_proj_weight = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, N_FFN, N_EMBD);
        layers_[i].ffn.c_proj_bias = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, N_EMBD);

        ggml_set_name(layers_[i].ffn.c_fc_weight, ("ffn_c_fc_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ffn.c_fc_bias, ("ffn_c_fc_bias_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ffn.c_proj_weight, ("ffn_c_proj_weight_" + std::to_string(i)).c_str());
        ggml_set_name(layers_[i].ffn.c_proj_bias, ("ffn_c_proj_bias_" + std::to_string(i)).c_str());

        memset(layers_[i].ffn.c_fc_weight->data, 0, ggml_nbytes(layers_[i].ffn.c_fc_weight));
        memset(layers_[i].ffn.c_fc_bias->data, 0, ggml_nbytes(layers_[i].ffn.c_fc_bias));
        memset(layers_[i].ffn.c_proj_weight->data, 0, ggml_nbytes(layers_[i].ffn.c_proj_weight));
        memset(layers_[i].ffn.c_proj_bias->data, 0, ggml_nbytes(layers_[i].ffn.c_proj_bias));
    }

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

                // Check if dimensions are transposed
                bool needs_transpose = false;
                if (t.n_dims == 2) {
                    // All GPT-2 linear transformations (c_attn, c_proj, c_fc) are HuggingFace Conv1D. 
                    // Conv1D stores memory as [in_features, out_features] where out_features is contiguous.
                    // ggml_mul_mat requires in_features to be contiguous. Thus, we MUST transpose them all.
                    if (t.name.find("blk.") != std::string::npos && 
                        t.name.find(".weight") != std::string::npos && 
                        t.name.find("norm") == std::string::npos) {
                        needs_transpose = true;
                    } 
                    // Fallback for statically inverted dimensions
                    else if (t.dims[0] != (uint64_t)dst->ne[0] && t.dims[1] == (uint64_t)dst->ne[0]) {
                        needs_transpose = true;
                    } 
                }

                // [DEBUG] Print dimension mapping for first layer to verify
                if (t.name.find("blk.0.") != std::string::npos) {
                    std::cout << "[DEBUG] Tensor " << t.name << " file_dims=[" << t.dims[0] << "," << t.dims[1] 
                              << "] dst_ne=[" << dst->ne[0] << "," << dst->ne[1] 
                              << "] transposed=" << (needs_transpose ? "YES" : "NO") << std::endl;
                }

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
                        // Q8_0 variant (type 30): 2-byte scale (float16) + 32 int8 values per block
                        std::vector<uint8_t> qdata(actual_nbytes);
                        read_tensor_data(gguf, t, qdata.data(), actual_nbytes);
                        size_t n_blocks = actual_nbytes / 34;  // 2 bytes scale + 32 bytes data
                        size_t j = 0;
                        for (size_t b = 0; b < n_blocks; b++) {
                            // Read scale (float16, 2 bytes, little-endian)
                            uint16_t scale_bits = qdata[b * 34] | (qdata[b * 34 + 1] << 8);
                            float scale = fp16_to_fp32(scale_bits);
                            // Read 32 quantized values
                            for (int i = 0; i < 32; i++) {
                                int8_t val = (int8_t)qdata[b * 34 + 2 + i];
                                buffer_f[j++] = val * scale;
                            }
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
                        if (needs_transpose) {
                            // PyTorch Conv1D physically stores memory as [in_features, out_features]
                            // where out_features is the contiguous fast dimension.
                            // GGML linear layers require in_features to be the contiguous fast dimension.
                            // Because GGUF conversion scripts often mislabel t.dims for Conv1D, 
                            // we bypass t.dims entirely and read the physical memory relying on dst_ne.
                            int in_features = (int)dst->ne[0];
                            int out_features = (int)dst->ne[1];

                            for (int y = 0; y < in_features; y++) {
                                for (int x = 0; x < out_features; x++) {
                                    dst_ptr[x * in_features + y] = buffer_f[y * out_features + x];
                                }
                            }
                        } else {
                            std::memcpy(dst_ptr, buffer_f.data(), buffer_f.size() * sizeof(float));
                        }
                    }
                } else {
                    std::cout << "  Size mismatch for '" << t.name << "': expected " << expected_nbytes << " got " << actual_nbytes << std::endl;
                    failed++;
                }
            }
        }

        std::cout << "\nLoaded " << loaded << " tensors, " << failed << " failed/skipped" << std::endl;

       
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
    return 0;
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
    unsigned int sign = (bf16 >> 15) & 0x1;
    unsigned int exp = (bf16 >> 7) & 0xff;
    unsigned int mant = bf16 & 0x7f;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        else return sign ? -pow(2, -126) * (mant / 128.0f) : pow(2, -126) * (mant / 128.0f);
    } else if (exp == 255) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        else return NAN;
    }

    int32_t e = exp - 127;
    float m = 1.0f + mant / 128.0f;
    return sign ? -pow(2, e) * m : pow(2, e) * m;
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

    // Allocate a temporary ggml context for this forward pass graph
    struct ggml_init_params params0 = {
        .mem_size   = 256 * 1024 * 1024,  // 256 MB for the compute graph
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    ggml_context* ctx0 = ggml_init(params0);

    // Build computation graph
    build_graph(ctx0, input_ids, position, use_cache);
    
    // [DEBUG] Track memory footprint after building the full architecture graph
    std::cout << "[DEBUG] Forward pass (seq_len=" << input_ids.size() 
              << ") build complete. ctx0 memory used: " 
              << ggml_used_mem(ctx0) / (1024.0 * 1024.0) << " MB" << std::endl;

    // Compute
    compute();

    // Extract logits from computed tensor
    // Get logits for the last position only
    std::vector<float> logits(VOCAB_SIZE, 0.0f);

    if (logits_tensor_) {
        // logits_tensor_: ne[0]=VOCAB_SIZE, ne[1]=seq_len
        // GGML uses row-major storage where ne[0] is the fast dimension.
        // Element [v, s] (where v is vocab index, s is sequence index)
        // is at offset s * ne[0] + v = s * VOCAB_SIZE + v
        int seq_len = input_ids.size();
        const float* logits_data = (const float*)logits_tensor_->data;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            logits[v] = logits_data[(seq_len - 1) * VOCAB_SIZE + v];
        }
    }

    // Free the temporary context
    ggml_free(ctx0);
    gf_ = nullptr;
    logits_tensor_ = nullptr;

    return logits;
}

void GPT2Model::build_graph(
    ggml_context* ctx0,
    const std::vector<int>& input_ids,
    int position,
    bool use_cache
) {
    // Allocate a fresh computation graph within the local context ctx0
    gf_ = ggml_new_graph(ctx0);

    int seq_len = input_ids.size();

    // Build embeddings by selecting rows from wte
    ggml_tensor* input_embd = nullptr;
    for (int i = 0; i < seq_len; i++) {
        // Get embedding for token input_ids[i]
        ggml_tensor* row = ggml_view_2d(ctx0, wte_, N_EMBD, 1,
                                        N_EMBD * sizeof(float),
                                        input_ids[i] * N_EMBD * sizeof(float));
        if (i == 0) {
            input_embd = row;
        } else {
            input_embd = ggml_concat(ctx0, input_embd, row, 0);
        }
    }
    // input_embd: (seq_len * N_EMBD, 1) = (12288, 1) for seq_len=16
    // Reshape to ne[0]=N_EMBD, ne[1]=seq_len (standard ggml convention)
    input_embd = ggml_reshape_2d(ctx0, input_embd, N_EMBD, seq_len);

    // Add positional embeddings
    ggml_tensor* pos_embd = nullptr;
    for (int i = 0; i < seq_len; i++) {
        int pos = position + i;
        if (pos >= CONTEXT_LENGTH) pos = CONTEXT_LENGTH - 1;  // Clip to max

        ggml_tensor* pos_row = ggml_view_2d(ctx0, wpe_, N_EMBD, 1,
                                            N_EMBD * sizeof(float),
                                            pos * N_EMBD * sizeof(float));
        if (i == 0) {
            pos_embd = pos_row;
        } else {
            pos_embd = ggml_concat(ctx0, pos_embd, pos_row, 0);
        }
    }
    // pos_embd: (seq_len * N_EMBD, 1) = (12288, 1)
    pos_embd = ggml_reshape_2d(ctx0, pos_embd, N_EMBD, seq_len);
    // pos_embd: ne[0]=N_EMBD, ne[1]=seq_len

    ggml_tensor* h = ggml_add(ctx0, input_embd, pos_embd);
    // h: ne[0]=N_EMBD, ne[1]=seq_len
    
    // [DEBUG] Add log to ensure forward pass graphs compile correctly
    // std::cout << "[DEBUG] Adding debug markers inside graph." << std::endl;

    // Pass through transformer layers
    for (int i = 0; i < N_LAYERS; i++) {
        // position is the sequence position of the FIRST token in input_ids
        // When use_cache=true and seq_len=1 (single new token), position is that token's position
        h = layers_[i].forward(ctx0, gf_, h, position, use_cache);
        ggml_build_forward_expand(gf_, h);
    }

    // Final layer norm
    ggml_tensor* h_norm = layer_norm(ctx0, h, ln_f_.gamma, ln_f_.beta, GPT2Config::layer_norm_eps);
    ggml_build_forward_expand(gf_, h_norm);

    // LM head: lm_head^T @ h_norm
    // lm_head: ne[0]=N_EMBD, ne[1]=VOCAB_SIZE; h_norm: ne[0]=N_EMBD, ne[1]=seq_len
    // ggml_mul_mat(lm_head, h_norm) = lm_head^T @ h_norm
    // Result: ne[0]=VOCAB_SIZE, ne[1]=seq_len
    logits_tensor_ = ggml_mul_mat(ctx0, lm_head_, h_norm);
    ggml_build_forward_expand(gf_, logits_tensor_);
}

void GPT2Model::compute() {
    // Use CPU backend since tensors are allocated in CPU memory (no_alloc=false)
    // Using GPU backend would require ggml_backend_alloc_ctx_tensors for GPU memory
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);

    if (!backend) {
        std::cerr << "Failed to initialize GGML CPU backend" << std::endl;
        return;
    }

    ggml_backend_graph_compute(backend, gf_);
    ggml_backend_free(backend);
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
        std::vector<float> logits = forward(tokens, 0, false);

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

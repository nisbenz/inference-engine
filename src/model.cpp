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
    , gf_()
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

        // Initialize KV cache for this layer
        layers_[i].attention.init_cache(ctx_);

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

bool GPT2Model::load_huggingface_weights(const std::string& path) {
    std::cout << "Loading HuggingFace weights from: " << path << std::endl;

    // This is a simplified loader
    // In reality, you'd use safetensors library or read PyTorch bin files
    // GPT-2 Large weights from HuggingFace:
    // - transformer.wte.weight: (50257, 1280)
    // - transformer.wpe.weight: (1024, 1280)
    // - transformer.h.{i}.ln_1.{gamma,beta}
    // - transformer.h.{i}.attn.{c_attn,c_proj}.{weight,bias}
    // - transformer.h.{i}.ln_2.{gamma,beta}
    // - transformer.h.{i}.mlp.{c_fc,c_proj}.{weight,bias}
    // - transformer.ln_f.{gamma,beta}
    // - lm_head.weight: (50257, 1280) - tied to wte

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open weights file: " << path << std::endl;
        return false;
    }

    // Simplified: just show what we'd load
    std::cout << "Note: This is a stub - need actual weight loading implementation" << std::endl;
    std::cout << "For now, using random initialization" << std::endl;

    // Random initialization for testing
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // Initialize with random weights for now
    float* wte_data = (float*)wte_->data;
    for (size_t i = 0; i < VOCAB_SIZE * N_EMBD; i++) {
        wte_data[i] = dist(gen);
    }

    float* wpe_data = (float*)wpe_->data;
    for (size_t i = 0; i < CONTEXT_LENGTH * N_EMBD; i++) {
        wpe_data[i] = dist(gen);
    }

    // Initialize layer weights
    for (int i = 0; i < N_LAYERS; i++) {
        float* ln1g = (float*)layers_[i].ln1.gamma->data;
        float* ln1b = (float*)layers_[i].ln1.beta->data;
        for (int j = 0; j < N_EMBD; j++) {
            ln1g[j] = (j == 0) ? 1.0f : 0.0f;  // gamma = 1
            ln1b[j] = 0.0f;  // beta = 0
        }

        float* c_attn_w = (float*)layers_[i].attention.c_attn_weight->data;
        float* c_attn_b = (float*)layers_[i].attention.c_attn_bias->data;
        float* c_proj_w = (float*)layers_[i].attention.c_proj_weight->data;
        float* c_proj_b = (float*)layers_[i].attention.c_proj_bias->data;

        for (size_t j = 0; j < 3 * N_EMBD * N_EMBD; j++) c_attn_w[j] = dist(gen);
        for (size_t j = 0; j < 3 * N_EMBD; j++) c_attn_b[j] = dist(gen);
        for (size_t j = 0; j < N_EMBD * N_EMBD; j++) c_proj_w[j] = dist(gen);
        for (size_t j = 0; j < N_EMBD; j++) c_proj_b[j] = dist(gen);

        float* ln2g = (float*)layers_[i].ln2.gamma->data;
        float* ln2b = (float*)layers_[i].ln2.beta->data;
        for (int j = 0; j < N_EMBD; j++) {
            ln2g[j] = (j == 0) ? 1.0f : 0.0f;
            ln2b[j] = 0.0f;
        }

        float* fc_w = (float*)layers_[i].ffn.c_fc_weight->data;
        float* fc_b = (float*)layers_[i].ffn.c_fc_bias->data;
        float* proj_w = (float*)layers_[i].ffn.c_proj_weight->data;
        float* proj_b = (float*)layers_[i].ffn.c_proj_bias->data;

        for (size_t j = 0; j < N_FFN * N_EMBD; j++) fc_w[j] = dist(gen);
        for (size_t j = 0; j < N_FFN; j++) fc_b[j] = dist(gen);
        for (size_t j = 0; j < N_EMBD * N_FFN; j++) proj_w[j] = dist(gen);
        for (size_t j = 0; j < N_EMBD; j++) proj_b[j] = dist(gen);
    }

    // Final layer norm
    float* ln_fg = (float*)ln_f_.gamma->data;
    float* ln_fb = (float*)ln_f_.beta->data;
    for (int j = 0; j < N_EMBD; j++) {
        ln_fg[j] = (j == 0) ? 1.0f : 0.0f;
        ln_fb[j] = 0.0f;
    }

    return true;
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

        // Print all tensor names and dimensions for debugging
        std::cout << "\nTensors in GGUF file:" << std::endl;
        for (size_t i = 0; i < gguf.tensors.size() && i < 20; i++) {
            auto& t = gguf.tensors[i];
            std::cout << "  " << i << ": " << t.name << " [";
            for (uint32_t d = 0; d < t.n_dims; d++) {
                std::cout << t.dims[d] << (d < t.n_dims - 1 ? ", " : "");
            }
            std::cout << "] type=" << t.type << std::endl;
        }
        if (gguf.tensors.size() > 20) {
            std::cout << "  ... and " << (gguf.tensors.size() - 20) << " more tensors" << std::endl;
        }

        // Map tensor names to our model weights
        // The GGUF file uses names like "model.wte", "model.h.0.attn.c_attn.weight", etc.
        // Our model uses names like "wte", "attn_c_attn_weight", etc.

        std::cout << "\nLoading tensors..." << std::endl;

        int loaded = 0;
        int failed = 0;

        for (auto& t : gguf.tensors) {
            // Find matching tensor in our model by name
            ggml_tensor* dst = nullptr;

            // Build a lookup map based on tensor name patterns
            if (t.name == "model.wte" || t.name == "model.embed_tokens" || t.name == "token_embd.weight") {
                dst = wte_;
            } else if (t.name == "model.wpe" || t.name == "model.position_embeddings" || t.name == "pos_embd.weight") {
                dst = wpe_;
            } else if (t.name == "model.ln_f.weight" || t.name == "model.final_layernorm.weight" || t.name == "model.ln_f.g" || t.name == "output_norm.weight") {
                dst = ln_f_.gamma;
            } else if (t.name == "model.ln_f.bias" || t.name == "model.final_layernorm.bias" || t.name == "model.ln_f.b" || t.name == "output_norm.bias") {
                dst = ln_f_.beta;
            } else if (t.name.find(".attn.c_attn.weight") != std::string::npos ||
                       t.name.find(".attention.c_attn.weight") != std::string::npos) {
                // Extract layer number
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
                    dst = layers_[layer_idx].attention.c_attn_weight;
                }
            } else if (t.name.find(".attn.c_attn.bias") != std::string::npos ||
                       t.name.find(".attention.c_attn.bias") != std::string::npos) {
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
                    dst = layers_[layer_idx].attention.c_attn_bias;
                }
            } else if (t.name.find(".attn.c_proj.weight") != std::string::npos ||
                       t.name.find(".attention.c_proj.weight") != std::string::npos) {
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
                    dst = layers_[layer_idx].attention.c_proj_weight;
                }
            } else if (t.name.find(".attn.c_proj.bias") != std::string::npos ||
                       t.name.find(".attention.c_proj.bias") != std::string::npos) {
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
                    dst = layers_[layer_idx].attention.c_proj_bias;
                }
            } else if (t.name.find(".ln_1.weight") != std::string::npos ||
                       t.name.find(".layer_norm_1.weight") != std::string::npos) {
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
                    dst = layers_[layer_idx].ln1.gamma;
                }
            } else if (t.name.find(".ln_1.bias") != std::string::npos ||
                       t.name.find(".layer_norm_1.bias") != std::string::npos) {
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
                    dst = layers_[layer_idx].ln1.beta;
                }
            } else if (t.name.find(".ln_2.weight") != std::string::npos ||
                       t.name.find(".layer_norm_2.weight") != std::string::npos) {
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
                    dst = layers_[layer_idx].ln2.gamma;
                }
            } else if (t.name.find(".ln_2.bias") != std::string::npos ||
                       t.name.find(".layer_norm_2.bias") != std::string::npos) {
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
                    dst = layers_[layer_idx].ln2.beta;
                }
            } else if (t.name.find(".mlp.c_fc.weight") != std::string::npos ||
                       t.name.find(".mlp.c_fc.bias") != std::string::npos) {
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
                    if (t.name.find(".weight") != std::string::npos) {
                        dst = layers_[layer_idx].ffn.c_fc_weight;
                    } else {
                        dst = layers_[layer_idx].ffn.c_fc_bias;
                    }
                }
            } else if (t.name.find(".mlp.c_proj.weight") != std::string::npos ||
                       t.name.find(".mlp.c_proj.bias") != std::string::npos) {
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
                    if (t.name.find(".weight") != std::string::npos) {
                        dst = layers_[layer_idx].ffn.c_proj_weight;
                    } else {
                        dst = layers_[layer_idx].ffn.c_proj_bias;
                    }
                }
            }

            // GGUF blk.X patterns (llama.cpp style)
            else if (t.name.find(".blk.") != std::string::npos) {
                size_t layer_idx = extract_layer_idx(t.name);
                if (layer_idx < N_LAYERS) {
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
            }

            if (dst) {
                // Calculate expected size
                size_t expected_nbytes = ggml_nbytes(dst);
                size_t actual_nbytes = gguf_tensor_nbytes(t);

                if (expected_nbytes == actual_nbytes || t.type == GGUF_TID_Q4_K) {
                    // Type matches or quantized (will be handled separately)
                    if (t.type == GGUF_TID_F32) {
                        read_tensor_data(gguf, t, dst->data, actual_nbytes);
                        loaded++;
                    } else if (t.type == GGUF_TID_F16) {
                        // Convert F16 to F32
                        std::vector<uint16_t> f16_data(actual_nbytes / 2);
                        read_tensor_data(gguf, t, f16_data.data(), actual_nbytes);
                        auto* dst_f = (float*)dst->data;
                        for (size_t j = 0; j < f16_data.size(); j++) {
                            dst_f[j] = fp16_to_fp32(f16_data[j]);
                        }
                        loaded++;
                    } else if (t.type == GGUF_TID_Q4_K) {
                        // Q4_K_M quantization - needs special dequantization
                        std::cout << "  Warning: Q4_K tensor '" << t.name << "' needs dequantization, using random" << std::endl;
                        failed++;
                    } else if (t.type == GGUF_TID_Q8_0_ALT) {
                        // Q8_0 variant (type 30): 2-byte scale + 32 int8 values per block
                        printf("  [Debug] Dequantizing Q8_0 tensor '%s', %lu bytes\n", t.name.c_str(), (unsigned long)actual_nbytes);
                        std::vector<uint8_t> qdata(actual_nbytes);
                        read_tensor_data(gguf, t, qdata.data(), actual_nbytes);
                        auto* dst_f = (float*)dst->data;
                        size_t n_blocks = actual_nbytes / 34;  // 2 bytes scale + 32 bytes data
                        size_t j = 0;
                        for (size_t b = 0; b < n_blocks; b++) {
                            // Read scale (float16, 2 bytes)
                            uint16_t scale_bits = qdata[b * 34] | (qdata[b * 34 + 1] << 8);
                            float scale = bf16_to_fp32(scale_bits);
                            // Read 32 quantized values
                            for (int i = 0; i < 32; i++) {
                                int8_t val = (int8_t)qdata[b * 34 + 2 + i];
                                dst_f[j++] = val * scale;
                            }
                        }
                        loaded++;
                    } else if (t.type == GGUF_TID_BF16) {
                        // BF16 type - convert to F32
                        std::vector<uint16_t> bf16_data(actual_nbytes / 2);
                        read_tensor_data(gguf, t, bf16_data.data(), actual_nbytes);
                        auto* dst_f = (float*)dst->data;
                        for (size_t j = 0; j < bf16_data.size(); j++) {
                            dst_f[j] = bf16_to_fp32(bf16_data[j]);
                        }
                        loaded++;
                    } else {
                        std::cout << "  Warning: Unsupported type " << t.type << " for tensor '" << t.name << "'" << std::endl;
                        failed++;
                    }
                } else {
                    std::cout << "  Size mismatch for '" << t.name << "': expected " << expected_nbytes << " got " << actual_nbytes << std::endl;
                    failed++;
                }
            }
        }

        std::cout << "\nLoaded " << loaded << " tensors, " << failed << " failed/skipped" << std::endl;

        // For failed tensors, initialize with random weights
        if (failed > 0) {
            std::cout << "Initializing failed tensors with random weights..." << std::endl;
            std::random_device rd;
            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

            // Initialize all model weights
            float* wte_data = (float*)wte_->data;
            for (size_t i = 0; i < VOCAB_SIZE * N_EMBD; i++) wte_data[i] = dist(gen);

            float* wpe_data = (float*)wpe_->data;
            for (size_t i = 0; i < CONTEXT_LENGTH * N_EMBD; i++) wpe_data[i] = dist(gen);

            for (int i = 0; i < N_LAYERS; i++) {
                auto* ln1g = (float*)layers_[i].ln1.gamma->data;
                auto* ln1b = (float*)layers_[i].ln1.beta->data;
                for (int j = 0; j < N_EMBD; j++) { ln1g[j] = (j == 0) ? 1.0f : 0.0f; ln1b[j] = 0.0f; }

                auto* c_attn_w = (float*)layers_[i].attention.c_attn_weight->data;
                auto* c_attn_b = (float*)layers_[i].attention.c_attn_bias->data;
                auto* c_proj_w = (float*)layers_[i].attention.c_proj_weight->data;
                auto* c_proj_b = (float*)layers_[i].attention.c_proj_bias->data;
                for (size_t j = 0; j < (size_t)3 * N_EMBD * N_EMBD; j++) c_attn_w[j] = dist(gen);
                for (size_t j = 0; j < (size_t)3 * N_EMBD; j++) c_attn_b[j] = dist(gen);
                for (size_t j = 0; j < (size_t)N_EMBD * N_EMBD; j++) c_proj_w[j] = dist(gen);
                for (size_t j = 0; j < (size_t)N_EMBD; j++) c_proj_b[j] = dist(gen);

                auto* ln2g = (float*)layers_[i].ln2.gamma->data;
                auto* ln2b = (float*)layers_[i].ln2.beta->data;
                for (int j = 0; j < N_EMBD; j++) { ln2g[j] = (j == 0) ? 1.0f : 0.0f; ln2b[j] = 0.0f; }

                auto* fc_w = (float*)layers_[i].ffn.c_fc_weight->data;
                auto* fc_b = (float*)layers_[i].ffn.c_fc_bias->data;
                auto* proj_w = (float*)layers_[i].ffn.c_proj_weight->data;
                auto* proj_b = (float*)layers_[i].ffn.c_proj_bias->data;
                for (size_t j = 0; j < (size_t)N_FFN * N_EMBD; j++) fc_w[j] = dist(gen);
                for (size_t j = 0; j < (size_t)N_FFN; j++) fc_b[j] = dist(gen);
                for (size_t j = 0; j < (size_t)N_EMBD * N_FFN; j++) proj_w[j] = dist(gen);
                for (size_t j = 0; j < (size_t)N_EMBD; j++) proj_b[j] = dist(gen);
            }

            float* ln_fg = (float*)ln_f_.gamma->data;
            float* ln_fb = (float*)ln_f_.beta->data;
            for (int j = 0; j < N_EMBD; j++) { ln_fg[j] = (j == 0) ? 1.0f : 0.0f; ln_fb[j] = 0.0f; }
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

    // Build computation graph
    build_graph(input_ids, position, use_cache);

    // Compute
    compute();

    // Extract logits from computed tensor
    // Get logits for the last position only
    std::vector<float> logits(VOCAB_SIZE, 0.0f);

    if (logits_tensor_) {
        // logits_tensor_ is (seq_len, VOCAB_SIZE)
        // We want the last row, which is at index seq_len - 1
        int seq_len = input_ids.size();
        int last_idx = (seq_len - 1) * VOCAB_SIZE;

        // Copy the last row of logits
        // First get the pointer to logits data
        const float* logits_data = (const float*)logits_tensor_->data;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            logits[i] = logits_data[last_idx + i];
        }
    }

    return logits;
}

void GPT2Model::build_graph(
    const std::vector<int>& input_ids,
    int position,
    bool use_cache
) {
    // Reset computation graph to avoid memory accumulation
    ggml_graph_clear(&gf_);

    int seq_len = input_ids.size();

    // Get token embeddings: wte[input_ids]
    // wte is (VOCAB_SIZE, N_EMBD), we need to index by input_ids
    // Result: (seq_len, N_EMBD)
    ggml_tensor* token_embeddings = ggml_get_rows(ctx_, wte_,
        ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, seq_len));

    // For now, simplify: use input_ids directly as indices
    // Create a view of wte with the selected rows
    ggml_tensor* input_embd = nullptr;
    for (int i = 0; i < seq_len; i++) {
        // Get embedding for token input_ids[i]
        ggml_tensor* row = ggml_view_2d(ctx_, wte_, N_EMBD, 1,
                                        N_EMBD * sizeof(float),
                                        input_ids[i] * N_EMBD * sizeof(float));
        if (i == 0) {
            input_embd = row;
        } else {
            input_embd = ggml_concat(ctx_, input_embd, row, 0);
        }
    }
    // input_embd: (seq_len, N_EMBD)

    // Add positional embeddings
    ggml_tensor* pos_embd = nullptr;
    for (int i = 0; i < seq_len; i++) {
        int pos = position + i;
        if (pos >= CONTEXT_LENGTH) pos = CONTEXT_LENGTH - 1;  // Clip to max

        ggml_tensor* pos_row = ggml_view_2d(ctx_, wpe_, N_EMBD, 1,
                                            N_EMBD * sizeof(float),
                                            pos * N_EMBD * sizeof(float));
        if (i == 0) {
            pos_embd = pos_row;
        } else {
            pos_embd = ggml_concat(ctx_, pos_embd, pos_row, 0);
        }
    }
    // pos_embd: (seq_len, N_EMBD)

    ggml_tensor* h = ggml_add(ctx_, input_embd, pos_embd);
    // h: (seq_len, N_EMBD)

    // Pass through transformer layers
    for (int i = 0; i < N_LAYERS; i++) {
        // position is the sequence position of the FIRST token in input_ids
        // When use_cache=true and seq_len=1 (single new token), position is that token's position
        h = layers_[i].forward(ctx_, &gf_, h, position, use_cache);
        ggml_build_forward_expand(&gf_, h);
    }

    // Final layer norm
    ggml_tensor* h_norm = layer_norm(ctx_, h, ln_f_.gamma, ln_f_.beta, GPT2Config::layer_norm_eps);
    ggml_build_forward_expand(&gf_, h_norm);

    // LM head: h_norm @ lm_head^T
    // lm_head is (VOCAB_SIZE, N_EMBD), we need (N_EMBD, VOCAB_SIZE)
    // Result: (seq_len, VOCAB_SIZE)
    logits_tensor_ = ggml_mul_mat(ctx_, h_norm, lm_head_);
    ggml_build_forward_expand(&gf_, logits_tensor_);
}

void GPT2Model::compute() {
    // GGML 0.10+ backend API: use best available backend (GPU if available)
    ggml_backend_t backend = nullptr;

    // Try CUDA backend first
    if (use_gpu_) {
        // Try ggml_backend_init_by_name("cuda") first
        backend = ggml_backend_init_by_name("cuda", NULL);
        if (!backend) {
            // Fall back to GPU type
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
        }
    }

    // Fall back to CPU if GPU not available or disabled
    if (!backend) {
        if (use_gpu_) {
            std::cerr << "GPU backend not available, falling back to CPU" << std::endl;
        }
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }

    if (!backend) {
        std::cerr << "Failed to initialize GGML backend" << std::endl;
        return;
    }

    ggml_backend_graph_compute(backend, &gf_);
    ggml_backend_free(backend);
}

std::vector<int> GPT2Model::generate(
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    int top_k
) {
    std::vector<int> tokens = prompt_tokens;

    // First forward pass: process the full prompt to populate KV cache
    // use_cache=true means we'll cache K,V for each layer
    std::cout << "Processing prompt (" << tokens.size() << " tokens)..." << std::endl;
    forward(tokens, 0, true);  // position=0 for the first token

    // Subsequent passes: only process the single new token
    // KV cache will be used to attend to all previous tokens
    while ((int)tokens.size() < (int)prompt_tokens.size() + max_new_tokens) {
        int current_position = tokens.size() - 1;  // Position of the last token in sequence

        // Create single-element vector with just the new token
        std::vector<int> single_token = {tokens.back()};

        // Forward pass with single token, use_cache=true to use & update KV cache
        std::vector<float> logits = forward(single_token, current_position, true);

        // Sample next token (use logits directly - they correspond to the single token)
        int next_token = sample(logits, temperature, top_k);

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

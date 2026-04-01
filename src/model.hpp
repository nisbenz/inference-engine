#pragma once

#include "layers.hpp"
#include "kv_cache.hpp"
#include "tokenizer.hpp"
#include <ggml.h>
#include <ggml-impl.h>
#include <string>
#include <vector>
#include <memory>

// GPT-2 (124M) Model
// Reference: https://huggingface.co/openai-community/gpt2
class GPT2Model {
public:
    GPT2Model();
    ~GPT2Model();

    // Initialize GGML context and allocate tensors
    bool init(bool use_gpu = true);

    // Load weights from file (GGUF format)
    bool load_weights(const std::string& path);

    // Run inference on input token IDs
    // Returns logits over vocabulary (shape: vocab_size)
    std::vector<float> forward(
        const std::vector<int>& input_ids,
        int position = 0,
        bool use_cache = true
    );

    // Generate tokens autoregressively
    // Starts with prompt tokens, generates up to max_new_tokens
    std::vector<int> generate(
        const std::vector<int>& prompt_tokens,
        int max_new_tokens = 100,
        float temperature = 1.0f,
        int top_k = 50
    );

    // Tokenize string to token IDs
    std::vector<int> tokenize(const std::string& text) { return tokenizer_.encode(text); }

    // Decode token IDs to string
    std::string decode(const std::vector<int>& tokens) { return tokenizer_.decode(tokens); }

    // Load tokenizer from vocab and merges files
    bool load_tokenizer(const std::string& vocab_path, const std::string& merges_path) {
        return tokenizer_.load(vocab_path, merges_path);
    }

    // Get logits for next token prediction at position
    // Must be called after forward()
    const float* get_logits() const { return logits_data_; }

    // Sample next token from logits
    int sample(const std::vector<float>& logits, float temperature, int top_k);

    // Configuration (GPT-2 base: 124M parameters)
    static constexpr int N_LAYERS = 12;       // 12 layers
    static constexpr int N_HEADS = 12;        // 12 attention heads
    static constexpr int N_EMBD = 768;        // 768 hidden size
    static constexpr int N_FFN = 3072;        // 3072 FFN intermediate size
    static constexpr int VOCAB_SIZE = 50257;  // GPT-2 vocabulary
    static constexpr int CONTEXT_LENGTH = 1024;
    static constexpr int HEAD_DIM = N_EMBD / N_HEADS;  // 64
    static constexpr int EOS_TOKEN = 50256;

private:
    // GGML context and compute graph
    ggml_context* ctx_;
    ggml_cgraph* gf_;
    bool use_gpu_;

    // Model weights (stored as GGML tensors)
    // Token embeddings
    ggml_tensor* wte_;  // (VOCAB_SIZE, N_EMBD) - token embeddings
    ggml_tensor* wpe_;  // (CONTEXT_LENGTH, N_EMBD) - position embeddings

    // Final layer norm
    LayerNorm ln_f_;

    // LM head (tied to wte)
    ggml_tensor* lm_head_;  // (VOCAB_SIZE, N_EMBD) - tied to wte

    // Transformer layers
    std::vector<TransformerBlock> layers_;

    // KV Cache
    KVCache kv_cache_;

    // Current logits
    float* logits_data_;
    int logits_size_;

    // Computed logits tensor (for extraction after compute)
    ggml_tensor* logits_tensor_;

    // Tokenizer
    GPT2Tokenizer tokenizer_;

    // Temporary tensors for forward pass
    ggml_tensor* input_ids_tensor_;
    ggml_tensor* position_tensor_;

    // Build computation graph for forward pass
    void build_graph(
        ggml_context* ctx0,
        const std::vector<int>& input_ids,
        int position,
        bool use_cache
    );

    // Actually compute the graph
    void compute(ggml_context* ctx0);

    // Load weights from GGUF format
    bool load_gguf_weights(const std::string& path);
    // Load weights from HuggingFace safetensors or pytorch bin
    bool load_huggingface_weights(const std::string& path);
    bool load_ggml_weights(const std::string& path);
};

// Helper: Create a 2D tensor with given shape and data
ggml_tensor* create_tensor_2d(
    ggml_context* ctx,
    const std::string& name,
    int rows,
    int cols,
    const float* data
);

// Helper: Create a 1D tensor with given shape and data
ggml_tensor* create_tensor_1d(
    ggml_context* ctx,
    const std::string& name,
    int size,
    const float* data
);

// Memory mapping utilities for large files
struct MappedFile {
    void* data;
    size_t size;

    MappedFile(const std::string& path);
    ~MappedFile();

    bool is_valid() const { return data != nullptr; }
};

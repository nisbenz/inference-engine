# GPT-2 Inference Engine

A C++ inference engine for GPT-2 model using `ggml`. This engine can load GPT-2 model in GGUF format and perform text generation.

## 🏗️ Architecture

The inference pipeline follows a standard autoregressive generation loop.

```mermaid
graph TD
    A[User Input / Prompt] --> B{Tokenizer}
    B -->|Tokens| C(GPT-2 Model)
    C --> D[Token Embeddings]
    C --> E[Positional Embeddings]
    D & E --> F[Transformer Blocks]
    
    subgraph Transformer Block
        F --> G[Layer Norm]
        G --> H[Multi-Head Self-Attention]
        H <--> |Read/Write Past Keys/Values| K[(KV Cache)]
        H --> I[Add & Norm]
        I --> J[MLP Feed-Forward]
        J --> L[Add & Norm]
    end
    
    L --> M[LM Head / Output Layer]
    M --> N[Logits]
    N -->|Sampling temperature, top_k| O[Next Token ID]
    O --> P{Detokenizer}
    O -->|Autoregressive| C
    P --> Q[Generated Text]
```

### Execution Flow

```mermaid
sequenceDiagram
    participant Main
    participant Model
    participant Loader
    participant Tokenizer

    Main->>Model: init(use_gpu)
    Main->>Model: load_weights(gguf_path)
    Model->>Loader: parse GGUF file
    Loader-->>Model: weights & tensors
    Main->>Model: load_tokenizer(vocab, merges)
    Model->>Tokenizer: load BPE files
    Main-->>Tokenizer: tokenize("Prompt text")
    Tokenizer-->>Main: prompt_tokens
    
    loop Autoregressive Generation
        Main->>Model: generate(prompt_tokens, max_tokens)
        Model->>Model: Forward Pass (Embeddings, Attention, MLP)
        Model->>Model: Sample output logits (temperature, top_k)
        Model-->>Main: Generated Token ID
    end

    Main->>Tokenizer: decode(new_tokens)
    Tokenizer-->>Main: Generated Text Output
```

## 🚀 How to Run

### Prerequisites
- **CMake** (3.16+)
- **CUDA Toolkit** (for GPU acceleration, target sm_75/T4 by default)
- **ggml** library (C/C++ tensor library)

### Building

1. **Clone and build the `ggml` library** (must be parallel to this repository by default, or provide path via `-DGGML_DIR`):
   ```bash
   git clone https://github.com/ggerganov/ggml.git ../ggml
   cd ../ggml
   mkdir build && cd build
   cmake .. 
   make -j
   cd ../../inference-engine
   ```

2. **Build the Engine**:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j
   ```

### Running Inference

Run the generated executable located in the `bin/` directory:

```bash
./build/bin/gpt2 <prompt> [max_tokens] [temperature] [top_k]
```

**Example:**
```bash
./build/bin/gpt2 "Once upon a time" 50 0.8 50
```

*Note: The current implementation looks for the model, vocabulary, and merges at specific hardcoded paths (e.g. `/content/gpt2-model/`). Make sure these files are present or adjust `src/main.cpp` accordingly.*

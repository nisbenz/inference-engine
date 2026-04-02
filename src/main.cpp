#include "model.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

// Usage: gpt2 <prompt> [max_tokens] [temperature] [top_k]
// Example: gpt2 "Hello, world!" 100 0.8 50

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <prompt> [max_tokens] [temperature] [top_k]" << std::endl;
    std::cout << "  prompt:      Input text (required)" << std::endl;
    std::cout << "  max_tokens:  Maximum tokens to generate (default: 100)" << std::endl;
    std::cout << "  temperature: Sampling temperature (default: 1.0)" << std::endl;
    std::cout << "  top_k:       Top-k sampling parameter (default: 50)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " \"Once upon a time\" 50 0.7 40" << std::endl;
}

int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string prompt = argv[1];
    int max_tokens = (argc > 2) ? std::stoi(argv[2]) : 100;
    float temperature = (argc > 3) ? std::stof(argv[3]) : 1.0f;
    int top_k = (argc > 4) ? std::stoi(argv[4]) : 50;

    std::cout << "=== GPT-2  Inference ===" << std::endl;
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Max tokens: " << max_tokens << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    std::cout << "Top-k: " << top_k << std::endl;
    std::cout << "==============================" << std::endl;

    // Initialize model
    GPT2Model model;
    if (!model.init(true)) {  // true = use GPU
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }

    // Load weights (GGUF format - downloaded in Colab)
    std::string weights_path = "/content/gpt2-model/gpt2-bf16.gguf";
    std::cout << "Loading weights from: " << weights_path << std::endl;

    if (!model.load_weights(weights_path)) {
        std::cerr << "Warning: Could not load weights, using random initialization" << std::endl;
    }

    // Load tokenizer
    std::string vocab_path = "/content/gpt2-model/vocab.json";
    std::string merges_path = "/content/gpt2-model/merges.txt";
    std::cout << "Loading tokenizer from: " << vocab_path << " and " << merges_path << std::endl;
    if (!model.load_tokenizer(vocab_path, merges_path)) {
        std::cerr << "Warning: Could not load tokenizer, using default encoding" << std::endl;
    }

    // Tokenize the prompt
    std::cout << "Tokenizing prompt..." << std::endl;
    std::vector<int> prompt_tokens = model.tokenize(prompt);
    std::cout << "Prompt tokens: " << prompt_tokens.size() << " tokens" << std::endl;

    // Generate
    std::cout << std::endl << "Generating..." << std::endl;
    std::cout << "Output: ";

    std::vector<int> generated_tokens = model.generate(prompt_tokens, max_tokens, temperature, top_k);

    // Decode and print only the newly generated tokens (skip prompt)
    std::vector<int> new_tokens(generated_tokens.begin() + prompt_tokens.size(), generated_tokens.end());
    std::string output = model.decode(new_tokens);
    std::cout << std::endl << "Prompt: " << prompt << std::endl;
    std::cout << "Generated: " << output << std::endl;

    std::cout << std::endl << "Done!" << std::endl;

    return 0;
}


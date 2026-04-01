#include <iostream>

// Forward declarations of test functions
int run_gguf_loader_tests();
int run_model_loading_tests();
int run_attention_tests();
int run_ffn_tests();
int run_layer_norm_tests();
int run_wte_diagnosis_tests(const char* gguf_path = nullptr);
int run_forward_pass_tests();

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << "       GPT-2 Inference Engine Unit Tests         " << std::endl;
    std::cout << "==================================================" << std::endl;

    int total_failures = 0;

    total_failures += run_gguf_loader_tests();
    total_failures += run_model_loading_tests();
    total_failures += run_attention_tests();
    total_failures += run_ffn_tests();
    total_failures += run_layer_norm_tests();
    total_failures += run_forward_pass_tests();

    // WTE diagnosis tests - optionally pass path to GGUF model file
    const char* gguf_path = (argc > 1) ? argv[1] : nullptr;
    total_failures += run_wte_diagnosis_tests(gguf_path);

    std::cout << "\n==================================================" << std::endl;
    std::cout << "                 FINAL RESULTS                   " << std::endl;
    std::cout << "==================================================" << std::endl;

    if (total_failures == 0) {
        std::cout << "  ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << "  " << total_failures << " TEST(S) FAILED" << std::endl;
    }

    std::cout << "==================================================" << std::endl;

    return total_failures;
}

cmake_minimum_required(VERSION 3.16)

# Test executable
add_executable(run_tests
    run_tests.cpp
    test_gguf_loader.cpp
    test_model_loading.cpp
    test_attention.cpp
    test_ffn.cpp
    test_layer_norm.cpp
    ${CMAKE_SOURCE_DIR}/src/layers.cpp
    ${CMAKE_SOURCE_DIR}/src/gguf_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/model.cpp
    ${CMAKE_SOURCE_DIR}/src/tokenizer.cpp
    ${CMAKE_SOURCE_DIR}/src/kv_cache.cpp
)

# Link against GGML and pthread
target_link_libraries(run_tests
    ${GGML_LIBRARIES}
    pthread
    m
)

# C++17
set_target_properties(run_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(run_tests PRIVATE
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${GGML_BUILD_DIR}/src
    ${CMAKE_SOURCE_DIR}/src
)

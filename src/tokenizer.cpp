#include "tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

// GPT-2 tokenizer regex pattern
// Matches: contractions | ASCII letters | numbers | punctuation | whitespace
// Using POSIX character classes for C++ regex compatibility
static const char* GPT2_REGEX_PATTERN =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^[:space:][[:alnum:]]]+|[[:space:]]+)";

GPT2Tokenizer::GPT2Tokenizer() {
    // GPT-2 byte-level BPE uses 50257 tokens (256 byte tokens + 50k merged)
    byte_to_id_.reserve(256);

    // Initialize byte to id mapping (same as GPT-2)
    // This maps raw bytes 0-255 to token IDs 0-255
    for (int i = 0; i < 256; i++) {
        byte_to_id_[i] = i;
    }

    // Compile regex pattern
    try {
        pat_ = std::regex(GPT2_REGEX_PATTERN, std::regex_constants::optimize);
    } catch (const std::exception& e) {
        std::cerr << "Regex compilation failed: " << e.what() << std::endl;
    }
}

GPT2Tokenizer::~GPT2Tokenizer() {}

// Load vocab and merges files
bool GPT2Tokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "Failed to open vocab file: " << vocab_path << std::endl;
        return false;
    }

    // Parse vocab JSON (simplified - expects the GPT-2 vocab format)
    // GPT-2 vocab is a dict of {token: id} where token may contain UTF-8 or special chars like Ġ
    std::string line;
    while (std::getline(vocab_file, line)) {
        // Expected format: "token": id
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;

        // Extract token string (between quotes)
        size_t token_start = line.find('"');
        size_t token_end = line.rfind('"');
        if (token_start == std::string::npos || token_end == std::string::npos) continue;
        token_start++; // Skip opening quote

        std::string token = line.substr(token_start, token_end - token_start);
        std::string id_str = line.substr(colon_pos + 1);
        int id = std::stoi(id_str);

        // Convert GPT-2 special bytes (Ġ = 0xC4 0xA0 = 256+32=288 in GPT-2)
        // These are GPT-2's way of representing bytes as Unicode chars
        if (token.size() >= 2 && (unsigned char)token[0] == 0xC4 && (unsigned char)token[1] == 0xA0) {
            // This represents byte 32 (space) - we already have it
            continue;
        }
    }

    // Parse merges file
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Failed to open merges file: " << merges_path << std::endl;
        return false;
    }

    int merge_rank = 256;  // Merges start after 256 base tokens
    std::string merge_line;
    while (std::getline(merges_file, merge_line)) {
        // Skip header lines
        if (merge_line.find("version") != std::string::npos) continue;
        if (merge_line.empty()) continue;

        // Parse "bpe1 bpe2" format
        std::istringstream iss(merge_line);
        std::string bpe1, bpe2;
        iss >> bpe1 >> bpe2;

        if (bpe1.empty() || bpe2.empty()) continue;

        // Convert byte sequences to IDs
        int id1 = -1, id2 = -1;

        // bpe1 and bpe2 are UTF-8 encoded byte sequences
        // Convert to bytes and look up IDs
        std::vector<unsigned char> bytes1, bytes2;

        for (size_t i = 0; i < bpe1.size(); ) {
            unsigned char c = (unsigned char)bpe1[i];
            if (c >= 0x80) {
                // UTF-8 continuation byte or multi-byte
                // For GPT-2, we need to decode properly
                int codepoint = 0;
                if ((c & 0xF0) == 0xF0) {
                    codepoint = c & 0x07;
                    if (i + 1 < bpe1.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe1[i+1] & 0x3F);
                    if (i + 2 < bpe1.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe1[i+2] & 0x3F);
                    if (i + 3 < bpe1.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe1[i+3] & 0x3F);
                    i += 4;
                } else if ((c & 0xE0) == 0xE0) {
                    codepoint = c & 0x0F;
                    if (i + 1 < bpe1.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe1[i+1] & 0x3F);
                    if (i + 2 < bpe1.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe1[i+2] & 0x3F);
                    i += 3;
                } else if ((c & 0xC0) == 0xC0) {
                    codepoint = c & 0x1F;
                    if (i + 1 < bpe1.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe1[i+1] & 0x3F);
                    i += 2;
                }

                // GPT-2 special: bytes 32-255 are encoded as Unicode chars 256-431
                if (codepoint >= 256 && codepoint < 432) {
                    bytes1.push_back((unsigned char)(codepoint - 256));
                } else {
                    // Convert Unicode to UTF-8 bytes
                    unsigned char utf8[4];
                    int len = 0;
                    if (codepoint < 0x80) {
                        utf8[0] = (unsigned char)codepoint;
                        len = 1;
                    } else if (codepoint < 0x800) {
                        utf8[0] = (unsigned char)(0xC0 | (codepoint >> 6));
                        utf8[1] = (unsigned char)(0x80 | (codepoint & 0x3F));
                        len = 2;
                    } else if (codepoint < 0x10000) {
                        utf8[0] = (unsigned char)(0xE0 | (codepoint >> 12));
                        utf8[1] = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3F));
                        utf8[2] = (unsigned char)(0x80 | (codepoint & 0x3F));
                        len = 3;
                    }
                    for (int j = 0; j < len; j++) {
                        bytes1.push_back(utf8[j]);
                    }
                }
            } else {
                bytes1.push_back(c);
                i++;
            }
        }

        // Same for bpe2
        for (size_t i = 0; i < bpe2.size(); ) {
            unsigned char c = (unsigned char)bpe2[i];
            if (c >= 0x80) {
                int codepoint = 0;
                if ((c & 0xF0) == 0xF0) {
                    codepoint = c & 0x07;
                    if (i + 1 < bpe2.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe2[i+1] & 0x3F);
                    if (i + 2 < bpe2.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe2[i+2] & 0x3F);
                    if (i + 3 < bpe2.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe2[i+3] & 0x3F);
                    i += 4;
                } else if ((c & 0xE0) == 0xE0) {
                    codepoint = c & 0x0F;
                    if (i + 1 < bpe2.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe2[i+1] & 0x3F);
                    if (i + 2 < bpe2.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe2[i+2] & 0x3F);
                    i += 3;
                } else if ((c & 0xC0) == 0xC0) {
                    codepoint = c & 0x1F;
                    if (i + 1 < bpe2.size()) codepoint = (codepoint << 6) | ((unsigned char)bpe2[i+1] & 0x3F);
                    i += 2;
                }

                if (codepoint >= 256 && codepoint < 432) {
                    bytes2.push_back((unsigned char)(codepoint - 256));
                } else {
                    unsigned char utf8[4];
                    int len = 0;
                    if (codepoint < 0x80) {
                        utf8[0] = (unsigned char)codepoint;
                        len = 1;
                    } else if (codepoint < 0x800) {
                        utf8[0] = (unsigned char)(0xC0 | (codepoint >> 6));
                        utf8[1] = (unsigned char)(0x80 | (codepoint & 0x3F));
                        len = 2;
                    } else if (codepoint < 0x10000) {
                        utf8[0] = (unsigned char)(0xE0 | (codepoint >> 12));
                        utf8[1] = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3F));
                        utf8[2] = (unsigned char)(0x80 | (codepoint & 0x3F));
                        len = 3;
                    }
                    for (int j = 0; j < len; j++) {
                        bytes2.push_back(utf8[j]);
                    }
                }
            } else {
                bytes2.push_back(c);
                i++;
            }
        }

        // Create merge pair
        if (bytes1.size() > 0 && bytes2.size() > 0) {
            // Compute a hash to use as key
            int key1 = bytes1[0];
            for (size_t i = 1; i < bytes1.size(); i++) {
                key1 = (key1 * 256) ^ bytes1[i];
                key1 = key1 % 1000003;  // Prime for hashing
            }
            key1 = (key1 * 31) + bytes2[0];
            for (size_t i = 1; i < bytes2.size(); i++) {
                key1 = (key1 * 256) ^ bytes2[i];
                key1 = key1 % 1000003;
            }

            // For simplicity, use rank-based indexing
            merges_[{id1, id2}] = merge_rank;
            merge_order_.push_back({id1, id2});
            merge_rank++;
        }
    }

    std::cout << "Loaded " << merges_.size() << " BPE merges" << std::endl;
    return true;
}

std::vector<int> GPT2Tokenizer::encode(const std::string& text) {
    std::vector<int> result;

    // Step 1: Regex chunking
    std::sregex_iterator it(text.begin(), text.end(), pat_);
    std::sregex_iterator end;

    while (it != end) {
        std::string chunk = it->str();

        // Step 2: UTF-8 encode each character to bytes
        std::vector<unsigned char> bytes;
        for (size_t i = 0; i < chunk.size(); ) {
            unsigned char c = (unsigned char)chunk[i];
            if (c < 0x80) {
                // ASCII
                bytes.push_back(c);
                i++;
            } else if ((c & 0xE0) == 0xC0) {
                // 2-byte UTF-8
                if (i + 1 < chunk.size()) {
                    bytes.push_back(c);
                    bytes.push_back((unsigned char)chunk[i + 1]);
                }
                i += 2;
            } else if ((c & 0xF0) == 0xE0) {
                // 3-byte UTF-8
                if (i + 2 < chunk.size()) {
                    bytes.push_back(c);
                    bytes.push_back((unsigned char)chunk[i + 1]);
                    bytes.push_back((unsigned char)chunk[i + 2]);
                }
                i += 3;
            } else if ((c & 0xF8) == 0xF0) {
                // 4-byte UTF-8
                if (i + 3 < chunk.size()) {
                    bytes.push_back(c);
                    bytes.push_back((unsigned char)chunk[i + 1]);
                    bytes.push_back((unsigned char)chunk[i + 2]);
                    bytes.push_back((unsigned char)chunk[i + 3]);
                }
                i += 4;
            } else {
                bytes.push_back(c);
                i++;
            }
        }

        // Step 3: Convert to token IDs (base bytes become their byte values)
        // This is a simplification - real GPT-2 vocab has 50257 tokens
        // Base tokens 0-255 are raw bytes, 256-289 are special (GPT-2 encoding bytes)
        // Then 290+ are merged tokens

        // For now, just use byte values directly
        for (unsigned char b : bytes) {
            result.push_back((int)b);
        }

        ++it;
    }

    // Step 4: Apply BPE merges (simplified)
    // Real BPE would iteratively merge pairs based on merge order
    // This is where the complexity lies

    return result;
}

std::string GPT2Tokenizer::decode(const std::vector<int>& tokens) {
    std::string result;

    for (int token : tokens) {
        // Convert token ID back to bytes
        if (token < 256) {
            // Raw byte
            result += (char)token;
        } else if (token >= 256 && token < 432) {
            // GPT-2 encoded byte
            result += (char)(token - 256);
        }
        // Note: merged tokens would need reverse lookup
    }

    return result;
}

// Utility: Read file contents
std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Simplified vocab parser
std::unordered_map<int, std::string> parse_vocab(const std::string& path) {
    std::unordered_map<int, std::string> vocab;
    // This would parse the GPT-2 vocab.json
    return vocab;
}

// Simplified merges parser
std::vector<std::pair<std::string, std::string>> parse_merges(const std::string& path) {
    std::vector<std::pair<std::string, std::string>> merges;
    return merges;
}

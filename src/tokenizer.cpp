#include "tokenizer.hpp"
#include <climits>
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

    // Initialize byte to id mapping (will be populated fully in load)
    // Fallback: This maps raw bytes 0-255 to token IDs 0-255 if not in vocab
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

    // Parse vocab JSON string (extracts keys and IDs)
    std::string json_str((std::istreambuf_iterator<char>(vocab_file)),
                          std::istreambuf_iterator<char>());

    // Parse JSON object: iterate through to find "key": value pairs
    size_t pos = 0;
    while (pos < json_str.size()) {
        // Find opening quote of key
        pos = json_str.find('"', pos);
        if (pos == std::string::npos) break;
        pos++; // skip opening quote

        // Extract key (handle escaped characters)
        std::string raw_key;
        while (pos < json_str.size() && json_str[pos] != '"') {
            if (json_str[pos] == '\\' && pos + 1 < json_str.size()) {
                raw_key += json_str[pos];
                raw_key += json_str[pos + 1];
                pos += 2;
            } else {
                raw_key += json_str[pos];
                pos++;
            }
        }
        if (pos >= json_str.size()) break;
        pos++; // skip closing quote

        // Find colon
        pos = json_str.find(':', pos);
        if (pos == std::string::npos) break;
        pos++; // skip colon

        // Skip whitespace
        while (pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t')) pos++;

        // Parse integer value
        size_t val_start = pos;
        while (pos < json_str.size() && json_str[pos] >= '0' && json_str[pos] <= '9') pos++;
        if (pos == val_start) continue; // no number found
        int id = std::stoi(json_str.substr(val_start, pos - val_start));

        // Unescape the JSON key
        std::string token;
        for (size_t i = 0; i < raw_key.size(); i++) {
            if (raw_key[i] == '\\' && i + 1 < raw_key.size()) {
                switch (raw_key[i + 1]) {
                    case 'n': token += '\n'; i++; break;
                    case 't': token += '\t'; i++; break;
                    case 'r': token += '\r'; i++; break;
                    case '"': token += '"'; i++; break;
                    case '\\': token += '\\'; i++; break;
                    case '/': token += '/'; i++; break;
                    case 'u': {
                        if (i + 5 < raw_key.size()) {
                            std::string hex = raw_key.substr(i + 2, 4);
                            int cp = std::stoi(hex, nullptr, 16);
                            if (cp < 0x80) {
                                token += (char)cp;
                            } else if (cp < 0x800) {
                                token += (char)(0xC0 | (cp >> 6));
                                token += (char)(0x80 | (cp & 0x3F));
                            } else {
                                token += (char)(0xE0 | (cp >> 12));
                                token += (char)(0x80 | ((cp >> 6) & 0x3F));
                                token += (char)(0x80 | (cp & 0x3F));
                            }
                            i += 5;
                        }
                        break;
                    }
                    default: token += raw_key[i]; break;
                }
            } else {
                token += raw_key[i];
            }
        }

        // Decode GPT-2 byte encoder mapping back to raw bytes
        std::string decoded;
        for (size_t i = 0; i < token.size(); ) {
            unsigned char c = (unsigned char)token[i];
            int cp = 0;
            int nbytes = 1;
            if (c < 0x80) {
                cp = c; nbytes = 1;
            } else if ((c & 0xE0) == 0xC0) {
                cp = c & 0x1F;
                if (i + 1 < token.size()) cp = (cp << 6) | ((unsigned char)token[i+1] & 0x3F);
                nbytes = 2;
            } else if ((c & 0xF0) == 0xE0) {
                cp = c & 0x0F;
                if (i + 1 < token.size()) cp = (cp << 6) | ((unsigned char)token[i+1] & 0x3F);
                if (i + 2 < token.size()) cp = (cp << 6) | ((unsigned char)token[i+2] & 0x3F);
                nbytes = 3;
            } else {
                cp = c & 0x07;
                if (i + 1 < token.size()) cp = (cp << 6) | ((unsigned char)token[i+1] & 0x3F);
                if (i + 2 < token.size()) cp = (cp << 6) | ((unsigned char)token[i+2] & 0x3F);
                if (i + 3 < token.size()) cp = (cp << 6) | ((unsigned char)token[i+3] & 0x3F);
                nbytes = 4;
            }
            i += nbytes;

            if (cp >= 0 && cp <= 255) {
                decoded += (char)cp;
            } else if (cp >= 256 && cp <= 511) {
                decoded += (char)(cp - 256);
            } else {
                if (cp < 0x80) decoded += (char)cp;
                else if (cp < 0x800) {
                    decoded += (char)(0xC0 | (cp >> 6));
                    decoded += (char)(0x80 | (cp & 0x3F));
                } else {
                    decoded += (char)(0xE0 | (cp >> 12));
                    decoded += (char)(0x80 | ((cp >> 6) & 0x3F));
                    decoded += (char)(0x80 | (cp & 0x3F));
                }
            }
        }

        id_to_token_[id] = decoded;
        
        // Correct base byte mapping
        // If decoded is exactly 1 byte long, it means this token ID directly maps to a raw byte natively
        if (decoded.size() == 1) {
            byte_to_id_[(unsigned char)decoded[0]] = id;
        }
    }
    std::cout << "Loaded " << id_to_token_.size() << " vocab entries" << std::endl;

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

        std::string decoded1, decoded2;
        
        auto decode_bpe = [](const std::string& bpe) {
            std::string dec;
            for (size_t i = 0; i < bpe.size(); ) {
                unsigned char c = (unsigned char)bpe[i];
                int cp = 0, nbytes = 1;
                if (c < 0x80) { cp = c; nbytes = 1; }
                else if ((c & 0xE0) == 0xC0) {
                    cp = c & 0x1F;
                    if (i + 1 < bpe.size()) cp = (cp << 6) | ((unsigned char)bpe[i+1] & 0x3F);
                    nbytes = 2;
                } else if ((c & 0xF0) == 0xE0) {
                    cp = c & 0x0F;
                    if (i + 1 < bpe.size()) cp = (cp << 6) | ((unsigned char)bpe[i+1] & 0x3F);
                    if (i + 2 < bpe.size()) cp = (cp << 6) | ((unsigned char)bpe[i+2] & 0x3F);
                    nbytes = 3;
                } else {
                    cp = c & 0x07;
                    if (i + 1 < bpe.size()) cp = (cp << 6) | ((unsigned char)bpe[i+1] & 0x3F);
                    if (i + 2 < bpe.size()) cp = (cp << 6) | ((unsigned char)bpe[i+2] & 0x3F);
                    if (i + 3 < bpe.size()) cp = (cp << 6) | ((unsigned char)bpe[i+3] & 0x3F);
                    nbytes = 4;
                }
                i += nbytes;
                if (cp >= 0 && cp <= 255) dec += (char)cp;
                else if (cp >= 256 && cp <= 511) dec += (char)(cp - 256);
                else {
                    if (cp < 0x80) dec += (char)cp;
                    else if (cp < 0x800) { dec += (char)(0xC0 | (cp >> 6)); dec += (char)(0x80 | (cp & 0x3F)); }
                    else { dec += (char)(0xE0 | (cp >> 12)); dec += (char)(0x80 | ((cp >> 6) & 0x3F)); dec += (char)(0x80 | (cp & 0x3F)); }
                }
            }
            return dec;
        };
        
        decoded1 = decode_bpe(bpe1);
        decoded2 = decode_bpe(bpe2);
        
        int id1 = -1, id2 = -1;
        for (const auto& pair : id_to_token_) {
            if (id1 == -1 && pair.second == decoded1) id1 = pair.first;
            if (id2 == -1 && pair.second == decoded2) id2 = pair.first;
            if (id1 != -1 && id2 != -1) break;
        }

        if (id1 != -1 && id2 != -1) {
            merges_[{id1, id2}] = merge_rank;
            // Compute the merged string and look up its token ID
            std::string merged_str = decoded1 + decoded2;
            // Look up the merged string in id_to_token_ to find its token ID
            int merged_id = -1;
            for (const auto& pair : id_to_token_) {
                if (pair.second == merged_str) {
                    merged_id = pair.first;
                    break;
                }
            }
            if (merged_id != -1) {
                merge_to_token_[{id1, id2}] = merged_id;
            }
            merge_order_.push_back({id1, id2});
            merge_rank++;
        }
    }

    std::cout << "Loaded " << merges_.size() << " BPE merges" << std::endl;
    return true;
}

std::vector<int> GPT2Tokenizer::encode(const std::string& text) {
    std::vector<int> result;

    // Step 1: Regex chunking - split text into tokens matching the GPT-2 pattern
    std::sregex_iterator it(text.begin(), text.end(), pat_);
    std::sregex_iterator end;

    while (it != end) {
        std::string chunk = it->str();

        // Step 2: UTF-8 encode to raw bytes
        std::vector<unsigned char> bytes;
        for (size_t i = 0; i < chunk.size(); ) {
            unsigned char c = (unsigned char)chunk[i];
            if (c < 0x80) {
                bytes.push_back(c);
                i++;
            } else if ((c & 0xE0) == 0xC0) {
                if (i + 1 < chunk.size()) {
                    bytes.push_back(c);
                    bytes.push_back((unsigned char)chunk[i + 1]);
                }
                i += 2;
            } else if ((c & 0xF0) == 0xE0) {
                if (i + 2 < chunk.size()) {
                    bytes.push_back(c);
                    bytes.push_back((unsigned char)chunk[i + 1]);
                    bytes.push_back((unsigned char)chunk[i + 2]);
                }
                i += 3;
            } else if ((c & 0xF8) == 0xF0) {
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
        
        // Step 3: Convert bytes to token IDs (base tokens 0-255)
        // In GPT-2's byte-level BPE, bytes 0-255 map directly to token IDs 0-255
        std::vector<int> token_ids;
        token_ids.reserve(bytes.size());
        for (unsigned char b : bytes) {
            auto it_id = byte_to_id_.find((int)b);
            if (it_id != byte_to_id_.end()) {
                token_ids.push_back(it_id->second);
            } else {
                // Fallback: use byte value directly
                std::cerr << "[WARNING] Unmapped byte: " << (int)b << std::endl;
                token_ids.push_back((int)b);
            }
        }
        
        // Step 4: Apply BPE merges iteratively
        // Standard BPE: repeatedly find the highest-priority adjacent pair and merge
        while (token_ids.size() >= 2) {
            // Find the pair with the lowest merge rank (highest priority)
            int best_rank = INT_MAX;
            int best_pos = -1;

            for (size_t i = 0; i < token_ids.size() - 1; i++) {
                std::pair<int, int> pair = {token_ids[i], token_ids[i + 1]};
                auto merge_it = merges_.find(pair);
                if (merge_it != merges_.end()) {
                    if (merge_it->second < best_rank) {
                        best_rank = merge_it->second;
                        best_pos = (int)i;
                    }
                }
            }

            // If no valid merge exists, stop merging this chunk
            if (best_pos == -1) break;

            // Look up the actual merged token ID
            std::pair<int, int> pair = {token_ids[best_pos], token_ids[best_pos + 1]};
            auto merge_it = merge_to_token_.find(pair);
            if (merge_it == merge_to_token_.end()) break;  // No valid merged token found

            int new_token = merge_it->second;
            token_ids.erase(token_ids.begin() + best_pos);
            token_ids[best_pos] = new_token;
        }

        // Append the final token IDs for this chunk
        result.insert(result.end(), token_ids.begin(), token_ids.end());

        ++it;
    }

    return result;
}

std::string GPT2Tokenizer::decode(const std::vector<int>& tokens) {
    std::string result;

    for (int token : tokens) {
        auto it = id_to_token_.find(token);
        if (it != id_to_token_.end()) {
            result += it->second;
        } else if (token < 256) {
            // Fallback: raw byte
            result += (char)token;
        }
        // Skip unknown tokens silently
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

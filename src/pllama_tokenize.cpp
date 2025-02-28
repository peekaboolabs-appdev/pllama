#include "pllama_tokenize.h"

// Add these headers at the top
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include <atomic>
#include <shared_mutex>  // Remove this if not supported

#include "llama.h"
#include "ggml.h"

// Modify the logging macro to use a more compatible approach
#define PLLAMA_LOG(level, message) \
    do { \
        std::cerr << "[pllama-tokenize:" << level << "] " << message << std::endl; \
    } while(0)

class TokenizerManager {
  public:
      enum class LogLevel { DEBUG, INFO, WARNING, ERROR };
  
      // Public logging method
      static void log(LogLevel level, const std::string& message) {
          switch(level) {
              case LogLevel::ERROR:
                  PLLAMA_LOG("ERROR", message);
                  break;
              case LogLevel::WARNING:
                  PLLAMA_LOG("WARNING", message);
                  break;
              case LogLevel::INFO:
                  PLLAMA_LOG("INFO", message);
                  break;
              case LogLevel::DEBUG:
                  PLLAMA_LOG("DEBUG", message);
                  break;
          }
      }
  
      // Singleton instance retrieval
      static TokenizerManager& getInstance() {
          static TokenizerManager instance;
          return instance;
      }
  
      // Improved model caching with more compatible locking
      std::shared_ptr<llama_model> getOrLoadModel(const std::string& model_path) {
          // Use standard mutex instead of shared_mutex
          std::unique_lock<std::mutex> lock(cache_mutex);
          auto now = std::chrono::steady_clock::now();
  
          // Check cache hit and validate entry
          auto it = model_cache.find(model_path);
          if (it != model_cache.end()) {
              auto& entry = it->second;
              if (isModelCacheValid(entry, now)) {
                  entry.last_access = now;
                  return entry.model;
              }
          }
  
          // Load model with cache
          return loadModelWithCache(model_path, now);
      }
  
  private:
      struct ModelCacheEntry {
          std::shared_ptr<llama_model> model;
          std::chrono::steady_clock::time_point last_access;
          std::chrono::steady_clock::time_point created_at;
      };
  
      std::unordered_map<std::string, ModelCacheEntry> model_cache;
      std::mutex cache_mutex;
      std::atomic<size_t> total_cached_models{0};
      static constexpr size_t MAX_CACHED_MODELS = 5;
      static constexpr auto MODEL_CACHE_DURATION = std::chrono::minutes(30);
  
      bool isModelCacheValid(const ModelCacheEntry& entry, 
                              const std::chrono::steady_clock::time_point& now) {
          auto age = std::chrono::duration_cast<std::chrono::minutes>(
              now - entry.created_at);
          return age < MODEL_CACHE_DURATION;
      }
  
      std::shared_ptr<llama_model> loadModelWithCache(
          const std::string& model_path, 
          const std::chrono::steady_clock::time_point& now) {
          
          // Validate model file
          std::ifstream model_file(model_path, std::ios::binary);
          if (!model_file.good()) {
              log(LogLevel::ERROR, "Invalid model file: " + model_path);
              return nullptr;
          }
  
          // Prepare model loading parameters
          llama_model_params mparams = llama_model_default_params();
          mparams.vocab_only = true;
          mparams.use_mmap = true;
          mparams.n_gpu_layers = 0;
  
          // Suppress extensive logging
          llama_log_set(
              [](enum ggml_log_level level, const char* text, void*) {
                  if (level >= GGML_LOG_LEVEL_ERROR) {
                      std::cerr << "[llama-internal] " << text;
                  }
              }, 
              nullptr
          );
  
          // Initialize backend
          llama_backend_init();
  
          // Load model
          llama_model* raw_model = llama_model_load_from_file(model_path.c_str(), mparams);
          
          if (!raw_model) {
              log(LogLevel::ERROR, "Failed to load model: " + model_path);
              llama_backend_free();
              return nullptr;
          }
  
          // Create shared_ptr with custom deleter
          auto model = std::shared_ptr<llama_model>(
              raw_model, 
              [](llama_model* ptr) { 
                  llama_model_free(ptr); 
              }
          );
  
          // Manage cache size
          if (total_cached_models >= MAX_CACHED_MODELS) {
              evictOldestModel();
          }
  
          // Insert into cache
          model_cache[model_path] = {
              model, 
              now,  // last access 
              now   // created at
          };
          total_cached_models++;
  
          llama_backend_free();
          return model;
      }
  
      void evictOldestModel() {
          auto oldest_it = std::min_element(
              model_cache.begin(), model_cache.end(),
              [](const auto& a, const auto& b) {
                  return a.second.last_access < b.second.last_access;
              }
          );
  
          if (oldest_it != model_cache.end()) {
              model_cache.erase(oldest_it);
              total_cached_models--;
          }
      }
  };

extern "C" {
EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT 
size_t pllama_tokenize(struct pllama_tokenize_request request) {
    auto& manager = TokenizerManager::getInstance();

    // Validate input
    if (!request.input || !request.model_path) {
        manager.log(TokenizerManager::LogLevel::ERROR, 
                    "Invalid tokenization request: missing input or model path");
        return 0;
    }

    try {
        // Get or load model
        auto model = manager.getOrLoadModel(request.model_path);
        if (!model) {
            manager.log(TokenizerManager::LogLevel::ERROR, 
                        "Failed to load model for tokenization");
            return 0;
        }

        // Get vocabulary
        const llama_vocab* vocab = llama_model_get_vocab(model.get());
        if (!vocab) {
            manager.log(TokenizerManager::LogLevel::ERROR, 
                        "Failed to retrieve vocabulary from model");
            return 0;
        }

        // Validate input length
        const size_t input_len = strlen(request.input);
        if (input_len == 0) {
            manager.log(TokenizerManager::LogLevel::INFO, 
                        "Empty input provided for tokenization");
            return 0;
        }

        // Allocate token buffer with safe sizing
        const size_t max_possible_tokens = input_len * 2 + 16;
        std::vector<llama_token> tokens(max_possible_tokens);

        // First pass: determine required token count
        const int token_count_needed = -llama_tokenize(
            vocab, 
            request.input, 
            input_len, 
            nullptr, 
            0, 
            llama_vocab_get_add_bos(vocab), 
            true
        );

        if (token_count_needed <= 0) {
            manager.log(TokenizerManager::LogLevel::WARNING, 
                        "Tokenization count determination failed");
            return 0;
        }

        // Resize buffer if needed
        if (static_cast<size_t>(token_count_needed) > tokens.size()) {
            tokens.resize(token_count_needed + 8);
        }

        // Perform actual tokenization
        const int n_tokens = llama_tokenize(
            vocab, 
            request.input, 
            input_len, 
            tokens.data(), 
            tokens.size(), 
            llama_vocab_get_add_bos(vocab), 
            true
        );

        if (n_tokens < 0) {
            manager.log(TokenizerManager::LogLevel::ERROR, 
                        "Tokenization failed with error code: " + 
                        std::to_string(n_tokens));
            return 0;
        }

        // Log successful tokenization
        manager.log(TokenizerManager::LogLevel::INFO, 
                    "Successful tokenization: " + 
                    std::to_string(n_tokens) + " tokens");

        return n_tokens;
    }
    catch (const std::exception& e) {
        manager.log(TokenizerManager::LogLevel::ERROR, 
                    "Unexpected error during tokenization: " + 
                    std::string(e.what()));
        return 0;
    }
    catch (...) {
        manager.log(TokenizerManager::LogLevel::ERROR, 
                    "Unknown critical error during tokenization");
        return 0;
    }
}
} // extern "C"
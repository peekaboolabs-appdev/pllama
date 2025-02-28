#include "pllama.h"
#include "clip.h"
#include "pllama_chat_template.h"
#include "pllama_eos.h"
#include "pllama_inference_queue.h"
#include "pllama_llava.h"
#include "llava.h"

// LLaMA.cpp cross-platform support
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#if TARGET_OS_IOS
// iOS-specific includes
#include "../ios/llama.cpp/common/base64.hpp"
#include "../ios/llama.cpp/common/common.h"
#include "../ios/llama.cpp/common/sampling.h"
#include "../ios/llama.cpp/ggml/include/ggml.h"
#include "../ios/llama.cpp/include/llama.h"

#elif TARGET_OS_OSX
// macOS-specific includes
#include "../macos/llama.cpp/common/base64.hpp"
#include "../macos/llama.cpp/common/common.h"
#include "../macos/llama.cpp/common/sampling.h"
#include "../macos/llama.cpp/ggml/include/ggml.h"
#include "../macos/llama.cpp/include/llama.h"
#else
// Other platforms
#include "llama.cpp/common/base64.hpp"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/sampling.h"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"
#endif

#include <atomic>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits.h>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif
#include "ggml-backend.h"
#include "llama.cpp/src/llama-sampling.h"

// Global atomic for tracking model loading state
static std::atomic<bool> model_loading_in_progress(false);

// Forward declare logging functions
static void log_message(const char *message, pllama_log_callback dart_logger = nullptr);
static void log_message(const std::string &message, pllama_log_callback dart_logger = nullptr);

// Memory utilities
static void force_memory_release() {
  // Attempt to free memory by forcing system calls
#ifdef _WIN32
  // Windows: call EmptyWorkingSet
  EmptyWorkingSet(GetCurrentProcess());
#else
  // Linux/Mac: malloc trim if available
#ifdef __GLIBC__
  malloc_trim(0);
#endif
#endif
  // Allow some time for memory to be released
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Implement logging functions
static void log_message(const char *message, pllama_log_callback dart_logger) {
    if (dart_logger == nullptr) {
        fprintf(stderr, "%s\n", message);
    } else {
        dart_logger(message);
    }
}

static void log_message(const std::string &message, pllama_log_callback dart_logger) {
    log_message(message.c_str(), dart_logger);
}

static InferenceQueue global_inference_queue;

extern "C" {

EMSCRIPTEN_KEEPALIVE void pllama_inference(pllama_inference_request request,
                                           pllama_inference_callback callback) {
  std::cout << "[pllama] Hello from pllama.cpp! Queueing your request."
            << std::endl;
  global_inference_queue.enqueue(request, callback);
}

EMSCRIPTEN_KEEPALIVE FFI_PLUGIN_EXPORT void
pllama_inference_cancel(int request_id) {
  global_inference_queue.cancel(request_id);
}

static bool add_tokens_to_context(struct llama_context *ctx_llama,
                                  const std::vector<llama_token>& tokens, int n_batch,
                                  int *n_past, pllama_log_callback logger) {
    log_message("[DEBUG] add_tokens_to_context start", logger);
    const int N = (int)tokens.size();
    log_message("[DEBUG] token count: " + std::to_string(N), logger);
    if (N == 0) return true;

    // Keep tokens data alive until we're done with the batch
    std::vector<llama_token> tokens_data = tokens;
    log_message("[DEBUG] about to call llama_batch_get_one", logger);
    
    // Safety check for nullptr
    if (!ctx_llama) {
        log_message("[ERROR] Context is null in add_tokens_to_context", logger);
        return false;
    }
    
    llama_batch batch = llama_batch_get_one(tokens_data.data(), tokens_data.size());
    log_message("[DEBUG] got batch with " + std::to_string(batch.n_tokens) + " tokens", logger);
    
    // Check context space
    int n_ctx = llama_n_ctx(ctx_llama);
    int n_ctx_used = llama_get_kv_cache_used_cells(ctx_llama);
    log_message("[DEBUG] ctx space: used=" + std::to_string(n_ctx_used) + ", total=" + std::to_string(n_ctx), logger);
    
    if (n_ctx_used + batch.n_tokens > n_ctx) {
        log_message("context size exceeded", logger);
        return false;
    }
    
    log_message("[DEBUG] about to decode batch", logger);
    if (llama_decode(ctx_llama, batch)) {
        log_message("failed to decode", logger);
        return false;
    }
    log_message("[DEBUG] decode successful", logger);
    
    // Update past token count
    *n_past = llama_get_kv_cache_used_cells(ctx_llama);
    log_message("[DEBUG] updated n_past to " + std::to_string(*n_past), logger);
    return true;
}

static bool add_token_to_context(struct llama_context *ctx_llama,
                                 llama_token id, int *n_past, pllama_log_callback logger) {
    log_message("[DEBUG] adding token " + std::to_string(id) + " to context", logger);
    log_message("[DEBUG] add_token_to_context start, token id: " + std::to_string(id), logger);
    
    // Safety check for nullptr
    if (!ctx_llama) {
        log_message("[ERROR] Context is null in add_token_to_context", logger);
        return false;
    }
    
    // Check context space first
    int n_ctx = llama_n_ctx(ctx_llama);
    int n_ctx_used = llama_get_kv_cache_used_cells(ctx_llama);
    log_message("[DEBUG] ctx space: used=" + std::to_string(n_ctx_used) + ", total=" + std::to_string(n_ctx), logger);
    
    if (n_ctx_used + 1 > n_ctx) {
        log_message("context size exceeded", logger);
        return false;
    }

    // Create batch with a single token, following simple-chat.cpp
    llama_batch batch = llama_batch_get_one(&id, 1);
    log_message("[DEBUG] created batch with token " + std::to_string(id), logger);

    // No need to manually manage logits - llama_batch_get_one handles this
    
    log_message("[DEBUG] about to decode", logger);
    if (llama_decode(ctx_llama, batch)) {
        log_message("failed to decode", logger);
        
        return false;
    }
    log_message("[DEBUG] decode successful", logger);

    *n_past = llama_get_kv_cache_used_cells(ctx_llama);
    log_message("[DEBUG] add_token_to_context complete, n_past: " + std::to_string(*n_past), logger);
    return true;
}

static bool add_string_to_context(struct llama_context *ctx_llama,
                                  const char *str, int n_batch, int *n_past,
                                  bool add_bos, pllama_log_callback logger) {
  // Safety check for null pointers
  if (!ctx_llama || !str) {
    log_message("[ERROR] Null pointer passed to add_string_to_context", logger);
    return false;
  }

  std::string str2 = str;
  const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_llama));
  if (!vocab) {
    log_message("[ERROR] Failed to get vocabulary from model", logger);
    return false;
  }

  const int n_prompt_tokens = -llama_tokenize(
      vocab, str2.c_str(), str2.length(), NULL, 0, add_bos, true);
  std::vector<llama_token> embd_inp(n_prompt_tokens);
  if (llama_tokenize(vocab, str2.c_str(), str2.length(), embd_inp.data(),
                     embd_inp.size(), add_bos, true) < 0) {
    log_message("tokenization failed", logger);
    return false;
  }
  return add_tokens_to_context(ctx_llama, embd_inp, n_batch, n_past, logger);
}

static void log_callback_wrapper(enum ggml_log_level level, const char *text,
                                 void *user_data) {
  std::cout << "[llama] " << text;
}

// Verifies model file header to ensure it's a valid GGUF file
static bool verify_model_file(const char* path, bool detailed_check = false) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[pllama] Cannot open model file: " << path << std::endl;
    return false;
  }
  
  // Check file size
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  if (size < 32) { // Very small files can't be valid models
    std::cerr << "[pllama] File too small to be a valid model: " << size << " bytes" << std::endl;
    return false;
  }
  
  // Check GGUF magic
  char header[4];
  file.read(header, 4);
  bool valid_header = header[0] == 'G' && header[1] == 'G' && 
                     header[2] == 'U' && header[3] == 'F';
  
  if (!valid_header) {
    std::cerr << "[pllama] Invalid model file format (not a GGUF file)" << std::endl;
    return false;
  }
  
  // For detailed validation, we could check file version, tensor counts, etc.
  if (detailed_check) {
    // Read GGUF version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    // More detailed validation as needed
    std::cout << "[pllama] GGUF version: " << version << std::endl;
  }
  
  return true;
}

EMSCRIPTEN_KEEPALIVE void
pllama_inference_sync(pllama_inference_request request,
                      pllama_inference_callback callback) {
  // Prevent concurrent model loading
  if (model_loading_in_progress.exchange(true)) {
    // Another loading operation is already in progress
    if (callback != NULL) {
      callback("Error: Another model loading operation is already in progress", true);
    }
    model_loading_in_progress.store(false);
    return;
  }
  
  // Reset the flag when we exit this function
  auto reset_loading_flag = [&]() {
    model_loading_in_progress.store(false);
  };
  
  // Setup parameters, then load the model and create a context.
  int64_t start = ggml_time_ms();
  std::cout << "[pllama] Inference thread start" << std::endl;
  
  // Validate input parameters before proceeding
  if (!request.model_path || !request.input) {
    if (callback != NULL) {
      callback("Error: Missing required input parameters (model_path and input are required)", true);
    }
    std::cerr << "[pllama] Missing required input parameters" << std::endl;
    reset_loading_flag();
    return;
  }
  
  // Verify model file header
  if (!verify_model_file(request.model_path)) {
    if (callback != NULL) {
      callback("Error: Invalid or inaccessible model file", true);
    }
    reset_loading_flag();
    return;
  }
  
  try {
    // Release memory before loading
    force_memory_release();
    
    ggml_backend_load_all();
    std::cout << "[pllama] Backend initialized." << std::endl;

    // Create model parameters with optimized settings for better loading performance
    llama_model_params model_params = llama_model_default_params();
    
    // Optimize for memory-constrained environments
    model_params.n_gpu_layers = request.num_gpu_layers;
    model_params.use_mmap = true;  // Use memory mapping for efficiency
    model_params.use_mlock = false; // Don't lock memory
    model_params.progress_callback = NULL; // No progress callback for cleaner loading
    
    // Optimize context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = request.context_size;
    ctx_params.n_batch = request.context_size;
    
    // Enforce safe limits for mobile
    #if defined(__ANDROID__) || (defined(__APPLE__) && (TARGET_OS_IOS || TARGET_IPHONE_SIMULATOR))
      // Mobile device - force even more conservative settings
      model_params.n_gpu_layers = 0; // CPU only on mobile
      model_params.use_mmap = true;
      
      // Limit thread count on mobile
      if (request.num_threads > 2) {
        ctx_params.n_threads = 2;
        if (request.dart_logger) {
          request.dart_logger("[pllama] Mobile detected: limiting to 2 threads for stability");
        }
      } else {
        ctx_params.n_threads = request.num_threads;
      }
    #else
      // Desktop environment
      ctx_params.n_threads = request.num_threads;
    #endif
    
    // ctx_params.seed = LLAMA_DEFAULT_SEED; // 이 라인은 오류 발생으로 제거
    ctx_params.flash_attn = false; // Disable flash attention for compatibility
    
    std::cout << "[pllama] Context size: " << ctx_params.n_ctx << std::endl;
    std::cout << "[pllama] Batch size: " << ctx_params.n_batch << std::endl;
    std::cout << "[pllama] Threads: " << ctx_params.n_threads << std::endl;
    std::cout << "[pllama] GPU layers: " << model_params.n_gpu_layers << std::endl;

    // Configure sampling
    llama_sampler *smpl =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(
        smpl, llama_sampler_init_min_p((1.0f - request.top_p), 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(request.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Configure logging
    if (request.dart_logger != NULL) {
      std::cout << "[pllama] Using custom logger" << std::endl;
      llama_log_set(
          [](enum ggml_log_level level, const char *text, void *user_data) {
            pllama_log_callback dart_logger =
                reinterpret_cast<pllama_log_callback>(user_data);
            dart_logger(text);
          },
          reinterpret_cast<void *>(request.dart_logger));
    } else {
      llama_log_set(log_callback_wrapper, NULL);
    }
    
    // Multimodal handling
    bool prompt_contains_img = prompt_contains_image(request.input);
    bool should_load_clip = false;
    if (prompt_contains_img) {
      log_message("Prompt contains images, will process them later.",
                 request.dart_logger);
      std::string mmproj =
          request.model_mmproj_path == NULL ? "" : request.model_mmproj_path;
      if (mmproj.empty()) {
        log_message(
              "Warning: prompt contains images, but inference request doesn't "
              "specify model_mmproj_path. Multimodal model requires a .mmproj "
              "file.",
              request.dart_logger);
      } else {
        should_load_clip = true;
      }
    }

    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    std::vector<llava_image_embed *> image_embeddings;
    char *c_result = nullptr;

    auto cleanup = [&]() {
      // Proper resource cleanup in order
      if (ctx)
        llama_free(ctx);
      if (model)
        llama_model_free(model);
      if (smpl)
        llama_sampler_free(smpl);
      llama_backend_free();
      if (c_result) free(c_result);
      reset_loading_flag();
    };

    // Progressive model loading approach for better memory management
    log_message("Starting progressive model loading...", request.dart_logger);
    
    // Step 1: Load vocabulary only first (much faster and lower memory)
    std::cout << "[pllama] Phase 1: Loading model vocabulary..." << std::endl;
    model_params.vocab_only = true;
    model = llama_model_load_from_file(request.model_path, model_params);
    
    if (model == NULL) {
      std::cout << "[pllama] Unable to load model vocabulary." << std::endl;
      if (callback != NULL) {
        callback("Error: Unable to load model vocabulary", true);
      }
      cleanup();
      return;
    }
    
    // Release memory again after vocabulary load
    force_memory_release();
    
    // Step 2: Now load the full model
    std::cout << "[pllama] Phase 2: Loading full model..." << std::endl;
    llama_model_free(model);
    model = nullptr;
    
    model_params.vocab_only = false;
    
    // More detailed progress logging for full model load
    if (request.dart_logger) {
      request.dart_logger("[pllama] Loading full model - this may take some time...");
    }
    
    model = llama_model_load_from_file(request.model_path, model_params);
    if (model == NULL) {
      std::cout << "[pllama] Unable to load full model." << std::endl;
      if (callback != NULL) {
        callback("Error: Unable to load full model", true);
      }
      cleanup();
      return;
    }

    log_message("Model loaded successfully", request.dart_logger);
    
    // Create context with the loaded model
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
      std::cout << "[pllama] Unable to create context." << std::endl;
      if (callback != NULL) {
        callback("Error: Unable to create context", true);
      }
      cleanup();
      return;
    }

    std::string final_request_input = request.input;
    
    // Handle multimodal/image content if present
    if (should_load_clip) {
      std::string mmproj_path_std_str =
          request.model_mmproj_path == NULL ? "" : request.model_mmproj_path;
      log_message("Loading multimodal model...", request.dart_logger);
      const char *mmproj_path = mmproj_path_std_str.c_str();
      
      // Validate CLIP model path
      std::ifstream clip_file_check(mmproj_path);
      if (!clip_file_check.good()) {
        std::cout << "[pllama] Unable to load CLIP model." << std::endl;
        if (callback != NULL) {
          callback("Error: Unable to load CLIP model", true);
        }
        cleanup();
        return;
      }
      clip_file_check.close();
      
      auto ctx_clip = clip_model_load(mmproj_path, /*verbosity=*/1);
      if (!ctx_clip) {
        std::cout << "[pllama] Failed to load CLIP model." << std::endl;
        if (callback != NULL) {
          callback("Error: Failed to load CLIP model", true);
        }
        cleanup();
        return;
      }
      
      std::cout << "Loaded CLIP model successfully" << std::endl;
      image_embeddings = llava_image_embed_make_with_prompt_base64(
          ctx_clip, ctx_params.n_threads, final_request_input);
      clip_free(ctx_clip);
    }

    // Process and clean up prompt if it contains images
    if (prompt_contains_img) {
      if (image_embeddings.empty()) {
        std::cout
            << "[pllama] Unable to create image embeddings, removing image "
               "data from prompt."
            << std::endl;
      } else {
        std::cout << "[pllama] Images loaded, replacing image data in prompt "
                     "with clip output"
                  << std::endl;
      }
      final_request_input = remove_all_images_from_prompt(request.input, "");
    }

    int64_t model_load_end = ggml_time_ms();
    int64_t model_load_duration_ms = model_load_end - start;
    log_message("Model loaded in " + std::to_string(model_load_duration_ms) +
                   " ms.",
               request.dart_logger);

    // Tokenize the prompt
    const int n_ctx = llama_n_ctx(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    if (!vocab) {
      std::cout << "[pllama] Failed to get vocabulary." << std::endl;
      if (callback != NULL) {
        callback("Error: Failed to get vocabulary", true);
      }
      cleanup();
      return;
    }

    const int n_prompt_tokens =
        -llama_tokenize(vocab, final_request_input.c_str(),
                        final_request_input.length(), NULL, 0, true, true);
    
    if (n_prompt_tokens <= 0) {
      std::cout << "[pllama] Tokenization failed." << std::endl;
      if (callback != NULL) {
        callback("Error: Tokenization failed", true);
      }
      cleanup();
      return;
    }
    
    std::vector<llama_token> tokens_list(n_prompt_tokens);
    if (llama_tokenize(vocab, final_request_input.c_str(),
                       final_request_input.length(), tokens_list.data(),
                       tokens_list.size(), true, true) < 0) {
      fprintf(stderr, "%s: tokenization failed\n", __func__);
      if (callback != NULL) {
        callback("Error: Unable to tokenize input", true);
      }
      cleanup();
      return;
    }
    
    log_message("Input token count: " + std::to_string(tokens_list.size()),
               request.dart_logger);
    log_message("Output token count: " + std::to_string(request.max_tokens),
               request.dart_logger);
    
    const int n_max_tokens = request.max_tokens;
    const int n_batch = ctx_params.n_batch;
    
    // Validate context capacity
    if (tokens_list.size() > static_cast<size_t>(n_ctx - n_max_tokens)) {
      std::cout << "[pllama] Input too large for context size." << std::endl;
      if (callback != NULL) {
        callback("Error: Input too large for context size", true);
      }
      cleanup();
      return;
    }

    // Process images embeddings first if they exist
    int n_past = 0;
    bool add_bos = llama_vocab_get_add_bos(vocab);
    int idx_embedding = 0;
    for (auto *embedding : image_embeddings) {
      if (embedding != NULL) {
        if (image_embeddings.size() > 1) {
          const std::string image_prompt =
              "Attached Image #" + std::to_string(idx_embedding + 1) + ":\n";
          add_string_to_context(ctx, image_prompt.c_str(), n_batch, &n_past,
                                add_bos, request.dart_logger);
          idx_embedding++;
        }
        log_message("Adding image #" + std::to_string(idx_embedding + 1) +
                       " to context.",
                   request.dart_logger);
        auto success =
            add_image_embed_to_context(ctx, embedding, n_batch, &n_past);
        if (!success) {
          log_message(
              "Unable to add image to context. Continuing to run inference "
              "anyway.",
              request.dart_logger);
        }
        llava_image_embed_free(embedding);
        log_message("Added image #" + std::to_string(idx_embedding + 1) +
                       " to context.",
                   request.dart_logger);
      }
    }

    log_message("Adding input to context...", request.dart_logger);
    
    // Add text tokens to context
    if (!add_tokens_to_context(ctx, tokens_list, n_batch, &n_past, request.dart_logger)) {
      std::cout << "[pllama] Failed to add tokens to context." << std::endl;
      if (callback != NULL) {
        callback("Error: Failed to add tokens to context", true);
      }
      cleanup();
      return;
    }
    
    log_message("Input added to context successfully", request.dart_logger);
    
    // Get EOS token for generation
    const char *eos_token_chars =
        request.eos_token != NULL ? request.eos_token
                                  : pllama_get_eos_token(request.model_path);
    
    // Check if EOS token retrieval failed
    if (!eos_token_chars) {
      std::cout << "[pllama] Failed to get EOS token." << std::endl;
      if (callback != NULL) {
        callback("Error: Failed to get EOS token", true);
      }
      cleanup();
      return;
    }
    
    const std::string eos_token_as_string = std::string(eos_token_chars);
    free((void *)eos_token_chars);
    
    const int64_t context_setup_complete = ggml_time_ms();
    log_message("Context setup complete in " +
                   std::to_string(context_setup_complete - start) + " ms.",
               request.dart_logger);

    // Check for cancellation before starting generation
    int request_id = request.request_id;
    if (global_inference_queue.is_cancelled(request_id)) {
      log_message("Request cancelled before generation started",
                 request.dart_logger);
      if (callback != NULL) {
        callback("", true);
      }
      cleanup();
      return;
    }

    // Signal that we're starting the generation phase
    if (callback != NULL) {
      callback("", false);
    }
    
    // Allocate result buffer with safety checks
    const auto estimated_total_size = n_max_tokens * 10;
    std::string result;
    result.reserve(estimated_total_size);
    c_result = (char *)malloc(estimated_total_size);
    if (!c_result) {
      std::cout << "[pllama] Failed to allocate memory for result." << std::endl;
      if (callback != NULL) {
        callback("Error: Memory allocation failure", true);
      }
      cleanup();
      return;
    }
    c_result[0] = '\0'; // Initialize to empty string

    // Generation loop with improved error handling and stability
    log_message("[DEBUG] starting token generation loop", request.dart_logger);
    
    // Safely sample first token
    if (!ctx) {
      std::cout << "[pllama] Context is null before token generation." << std::endl;
      if (callback != NULL) {
        callback("Error: Context is null", true);
      }
      cleanup();
      return;
    }
    
    // Start token generation
    llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
    if (new_token_id == -1) {
      std::cout << "[pllama] Failed to sample first token." << std::endl;
      if (callback != NULL) {
        callback("Error: Token sampling failed", true);
      }
      cleanup();
      return;
    }
    
    int n_gen = 0;
    const auto model_eos_token = llama_vocab_eos(vocab);
    const int64_t start_t = ggml_time_ms();
    int64_t t_last = start_t;
    
    std::vector<std::string> eos_tokens = {
        eos_token_as_string, // The original EOS token
        "<|end|>",           // Phi 3 24-04-30
        "<|eot_id|>"         // Llama 3 24-04-30
    };
    
    // Main token generation loop with batch processing
    llama_batch batch = llama_batch_get_one(&new_token_id, 1);
    bool generation_complete = false;
    
    while (!generation_complete) {
        // Check context space
        int n_ctx = llama_n_ctx(ctx);
        int n_ctx_used = llama_get_kv_cache_used_cells(ctx);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            log_message("[DEBUG] context size exceeded", request.dart_logger);
            break;
        }
    
        // Convert current token to text
        char token_text[256] = {0}; // Initialize to zeros for safety
        int token_len = llama_token_to_piece(vocab, new_token_id, token_text, sizeof(token_text) - 1, 0, true);
        if (token_len < 0) {
            log_message("[DEBUG] failed to convert token to text", request.dart_logger);
            break;
        }
        
        // Ensure null termination
        token_text[token_len < 255 ? token_len : 255] = '\0';
        
        // Add to result and send update
        std::string piece(token_text);
        result += piece;
        n_gen++;
        
        // Send intermediate result to callback
        if (callback != NULL) {
            if (result.length() < estimated_total_size) {
                std::strncpy(c_result, result.c_str(), estimated_total_size - 1);
                c_result[estimated_total_size - 1] = '\0'; // Ensure null termination
                callback(c_result, false);
            } else {
                log_message("[WARNING] Result exceeded estimated size", request.dart_logger);
            }
        }
    
        // Process the batch
        if (llama_decode(ctx, batch)) {
            log_message("[DEBUG] decode failed", request.dart_logger);
            break;
        }
        
        // Sample next token
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
    
        // Check end conditions
        if (new_token_id == model_eos_token || llama_vocab_is_eog(vocab, new_token_id)) {
            log_message("[DEBUG] end of generation detected", request.dart_logger);
            generation_complete = true;
            break;
        }
        
        if (n_gen >= n_max_tokens) {
            log_message("[DEBUG] reached max tokens: " + std::to_string(n_max_tokens), request.dart_logger);
            generation_complete = true;
            break;
        }
        
        if (global_inference_queue.is_cancelled(request_id)) {
            log_message("[DEBUG] generation cancelled", request.dart_logger);
            generation_complete = true;
            break;
        }
    
        // Prepare next batch
        batch = llama_batch_get_one(&new_token_id, 1);
        
        // Log generation speed periodically
        const auto t_now = ggml_time_ms();
        if (t_now - t_last > 1000) {
            float speed = n_gen / ((t_now - start_t) / 1000.0f);
            log_message("[pllama] generated " + std::to_string(n_gen) + 
                      " tokens at " + std::to_string(speed) + " tokens/sec",
                      request.dart_logger);
            t_last = t_now;
        }
    }
    
    log_message("[DEBUG] token generation loop complete", request.dart_logger);
    
    // Send final result
    if (result.length() < estimated_total_size) {
        std::strncpy(c_result, result.c_str(), estimated_total_size - 1);
        c_result[estimated_total_size - 1] = '\0'; // Ensure null termination
    } else {
        log_message("[WARNING] Result exceeded estimated size", request.dart_logger);
        std::strncpy(c_result, result.c_str(), estimated_total_size - 1);
        c_result[estimated_total_size - 1] = '\0'; // Ensure null termination
    }
    
    if (callback != NULL) {
        log_message("[DEBUG] Invoking final callback", request.dart_logger);
        callback(c_result, true);
        log_message("[DEBUG] Final callback invoked", request.dart_logger);
    } else {
        log_message("WARNING: callback is NULL. Output: " + result,
                   request.dart_logger);
    }

    // Log final performance statistics
    const auto t_now = ggml_time_ms();
    const auto total_time_ms = t_now - start_t;
    const auto speed_tokens_per_sec = n_gen / (total_time_ms / 1000.0f);
    
    const auto speed_string =
        "Generated " + std::to_string(n_gen) + " tokens in " +
        std::to_string(total_time_ms / 1000.0f) + " seconds, speed: " +
        std::to_string(speed_tokens_per_sec) + " tokens/sec";

    log_message(speed_string, request.dart_logger);
    
    // Clean up resources
    log_message("Cleaning up resources...", request.dart_logger);
    cleanup();
    log_message("Resources cleaned up successfully", request.dart_logger);
  } catch (const std::exception &e) {
    std::string error_msg = "Unhandled error: " + std::string(e.what());
    if (callback != NULL) {
      callback(error_msg.c_str(), true);
    }
    std::cerr << error_msg << std::endl;
    model_loading_in_progress.store(false);
  } catch (...) {
    std::string error_msg = "Unknown unhandled error occurred";
    if (callback != NULL) {
      callback(error_msg.c_str(), true);
    }
    std::cerr << error_msg << std::endl;
    model_loading_in_progress.store(false);
  }
}

} // extern "C"
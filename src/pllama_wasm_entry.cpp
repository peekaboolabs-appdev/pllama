#include "pllama.h"
#include "pllama_chat_template.h"
#include "pllama_eos.h"
#include "pllama_tokenize.h"
#include <stdio.h>

extern "C" {
const char *pllama_get_bos_token_export(const char *fname) {
  return pllama_get_bos_token(fname);
}

const char *pllama_get_eos_token_export(const char *fname) {
  return pllama_get_eos_token(fname);
}

size_t pllama_tokenize_export(const char *fname, const char *input) {
  pllama_tokenize_request request;
  request.input = const_cast<char *>(
      input); // Since input is 'const char*' and request.input is 'char*'
  request.model_path = const_cast<char *>(
      fname); // Since fname is 'const char*' and request.model_path is 'char*'

  size_t result = pllama_tokenize(request);
  return result;
}

const char *pllama_get_chat_template_export(const char *fname) {
  return pllama_get_chat_template(fname);
}

void pllama_cancel_inference_export(int request_id) {
  pllama_inference_cancel(request_id);
}

// Wrapper function to be called from JavaScript
void pllama_inference_export(
    int request_id, int context_size, char *input, int max_tokens,
    char *model_path, char *model_mmproj_path, int num_gpu_layers,
    int num_threads, float temperature, float top_p, float penalty_freq,
    float penalty_repeat, char *grammar, char *eos_token,
    void (*inference_callback_js)(const char *, uint8_t),
    void (*log_callback_js)(const char *)) {
  struct pllama_inference_request request;
  request.request_id = request_id;
  request.context_size = context_size;
  request.input = input;
  request.max_tokens = max_tokens;
  request.model_path = model_path;
  request.model_mmproj_path = model_mmproj_path;
  request.num_gpu_layers = num_gpu_layers;
  request.num_threads = num_threads;
  request.temperature = temperature;
  request.top_p = top_p;
  request.penalty_freq = penalty_freq;
  request.penalty_repeat = penalty_repeat;
  request.grammar = grammar;
  request.eos_token = eos_token;
  request.dart_logger = log_callback_js;
  pllama_inference_sync(request, inference_callback_js);
}
}

int main() {
  // This might remain empty if you're primarily calling functions from
  // JavaScript. Or perform any initialization needed.
}
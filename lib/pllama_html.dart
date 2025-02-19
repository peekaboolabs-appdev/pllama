import 'dart:async';
import 'dart:js_interop';
import 'dart:convert';

import 'package:pllama/pllama.dart';

@JS('pllamaInferenceJs')
external JSPromise<JSNumber> pllamaInferenceJs(
    JSAny request, JSFunction callback);

typedef FllamaInferenceCallback = void Function(String response, bool done);

// Keep in sync with pllama_inference_request.dart to pass correctly from Dart to JS
extension type _JSFllamaInferenceRequest._(JSObject _)  implements JSObject {
  external factory _JSFllamaInferenceRequest({
    required int contextSize,
    required String input,
    required int maxTokens,
    required String modelPath,
    String? modelMmprojPath,
    required int numGpuLayers,
    required int numThreads,
    required double temperature,
    required double penaltyFrequency,
    required double penaltyRepeat,
    required double topP,
    String? grammar,
    String? eosToken,
    // INTENTIONALLY MISSING: logger
  });
}


/// Runs standard LLM inference. The future returns immediately after being
/// called. [callback] is called on each new output token with the response and
/// a boolean indicating whether the response is the final response.
///
/// This is *not* what most people want to use. LLMs post-ChatGPT use a chat
/// template and an EOS token. Use [pllamaChat] instead if you expect this
/// sort of interface, i.e. an OpenAI-like API.
Future<int> pllamaInference(FllamaInferenceRequest dartRequest,
    FllamaInferenceCallback callback) async {
  final jsRequest = _JSFllamaInferenceRequest(
    contextSize: dartRequest.contextSize,
    input: dartRequest.input,
    maxTokens: dartRequest.maxTokens,
    modelPath: dartRequest.modelPath,
    modelMmprojPath: dartRequest.modelMmprojPath,
    numGpuLayers: dartRequest.numGpuLayers,
    numThreads: dartRequest.numThreads,
    temperature: dartRequest.temperature,
    penaltyFrequency: dartRequest.penaltyFrequency,
    penaltyRepeat: dartRequest.penaltyRepeat,
    topP: dartRequest.topP,
    grammar: dartRequest.grammar,
    eosToken: dartRequest.eosToken,
  );

  final completer = Completer<int>();
  callbackFn(String response, bool done) {
    callback(response, done);
  }
  pllamaInferenceJs(jsRequest, callbackFn.toJS).toDart.then((value) {
    completer.complete(value.toDartInt);
  });
  return completer.future;
}

// JSAny used to be void. JSVoid does not work.
@JS('pllamaMlcWebModelDeleteJs')
external JSPromise<JSAny> pllamaMlcWebModelDeleteJs(String modelId);

@JS('pllamaMlcIsWebModelDownloadedJs')
external JSPromise<JSBoolean> pllamaMlcIsWebModelDownloadedJs(String modelId);

@JS('pllamaChatMlcWebJs')
external JSPromise<JSNumber> pllamaChatMlcWebJs(
    // ignore: library_private_types_in_public_api
    _JSFllamaMlcInferenceRequest request, JSFunction loadCallback, JSFunction callback);

extension type _JSFllamaMlcInferenceRequest._(JSObject _)  implements JSObject {
   external _JSFllamaMlcInferenceRequest({
    required String messagesAsJsonString,
    required String toolsAsJsonString,
    required int maxTokens,
    // Must match a model_id in [prebuiltAppConfig] in https://github.com/mlc-ai/web-llm/blob/main/src/config.ts
    required String modelId,
    required double temperature,
    required double penaltyFrequency,
    required double penaltyRepeat,
    required double topP,
  });
}

typedef FllamaMlcLoadCallback = void Function(
    double downloadProgress, double loadProgress);

Future<bool> pllamaMlcIsWebModelDownloaded(String modelId) async {
  return pllamaMlcIsWebModelDownloadedJs(modelId).toDart.then((value) {
    return value.toDart;
  });
}

/// Use MLC's web JS SDK to do chat inference.
/// If not on web, this will fallback to using [pllamaChat].
///
/// llama.cpp converted to WASM is very slow compared to native inference on the
/// same platform, because it does not use the GPU.
///
/// MLC uses WebGPU to achieve ~native inference speeds.
Future<int> pllamaChatMlcWeb(
    OpenAiRequest request,
    FllamaMlcLoadCallback loadCallback,
    FllamaInferenceCallback callback) async {
  final messagesAsMaps = request.messages
      .map((e) => {
            'role': e.role.openAiName,
            'content': e.text,
          })
      .toList();
  final toolsAsMaps = request.tools
      .map((e) => {
            'type': 'function',
            'function': {
              'name': e.name,
              'description': e.description,
              'parameters': e.jsonSchema,
            }
          })
      .toList();
  final jsRequest = _JSFllamaMlcInferenceRequest(
    toolsAsJsonString: jsonEncode(toolsAsMaps),
    messagesAsJsonString: jsonEncode(messagesAsMaps),
    maxTokens: request.maxTokens,
    modelId: request.modelPath,
    temperature: request.temperature,
    penaltyFrequency: request.frequencyPenalty,
    penaltyRepeat: request.presencePenalty,
    topP: request.topP,
  );
  final completer = Completer<int>();
  firstCallback(double downloadProgress, double loadProgress) {
    loadCallback(downloadProgress, loadProgress);
  } 
  secondCallback(String response, bool done) {
    callback(response, done);
  }
  pllamaChatMlcWebJs(jsRequest, firstCallback.toJS, secondCallback.toJS).toDart.then((value) {
    completer.complete(value.toDartInt);
  });
  return completer.future;
}

Future<void> pllamaMlcWebModelDelete(String modelId) async {
  await pllamaMlcWebModelDeleteJs(modelId).toDart;
}

// Tokenize
@JS('pllamaTokenizeJs')
external JSPromise<JSNumber> pllamaTokenizeJs(String modelPath, String input);

/// Returns the number of tokens in [request.input].
///
/// Useful for identifying what messages will be in context when the LLM is run.
Future<int> pllamaTokenize(FllamaTokenizeRequest request) async {
  try {
    final completer = Completer<int>();
    // print('[pllama_html] calling pllamaTokenizeJs at ${DateTime.now()}');

    pllamaTokenizeJs(request.modelPath, request.input).toDart.then((value) {
      // print(
      // '[pllama_html] pllamaTokenizeAsync finished with $value at ${DateTime.now()}');
      completer.complete(value.toDartInt);
    });
    // print('[pllama_html] called pllamaTokenizeJs at ${DateTime.now()}');
    return completer.future;
  } catch (e) {
    // ignore: avoid_print
    print('[pllama_html] pllamaTokenizeAsync caught error: $e');
    rethrow;
  }
}

// Chat template
@JS('pllamaChatTemplateGetJs')
external JSPromise<JSString> pllamaChatTemplateGetJs(String modelPath);

/// Returns the chat template embedded in the .gguf file.
/// If none is found, returns an empty string.
///
/// See [pllamaSanitizeChatTemplate] for using sensible fallbacks for gguf
/// files that don't have a chat template or have incorrect chat templates.
Future<String> pllamaChatTemplateGet(String modelPath) async {
  try {
    final completer = Completer<String>();
    // print('[pllama_html] calling pllamaChatTemplateGetJs at ${DateTime.now()}');
    pllamaChatTemplateGetJs(modelPath).toDart.then((value) {
      // print(
      // '[pllama_html] pllamaChatTemplateGetJs finished with $value at ${DateTime.now()}');
      completer.complete(value.toDart);
    });
    // print('[pllama_html] called pllamaChatTemplateGetJs at ${DateTime.now()}');
    return completer.future;
  } catch (e) {
    // ignore: avoid_print
    print('[pllama_html] pllamaChatTemplateGetJs caught error: $e');
    rethrow;
  }
}

@JS('pllamaBosTokenGetJs')
external JSPromise<JSString?> pllamaBosTokenGetJs(String modelPath);

/// Returns the EOS token embedded in the .gguf file.
/// If none is found, returns an empty string.
///
/// See [pllamaApplyChatTemplate] for using sensible fallbacks for gguf
/// files that don't have an EOS token or have incorrect EOS tokens.
Future<String> pllamaBosTokenGet(String modelPath) {
  try {
    final completer = Completer<String>();
    // print('[pllama_html] calling pllamaEosTokenGet at ${DateTime.now()}');
    pllamaBosTokenGetJs(modelPath).toDart.then((value) {
      // print(
      // '[pllama_html] pllamaEosTokenGet finished with $value at ${DateTime.now()}');
      completer.complete(value?.toDart ?? '');
    });
    // print('[pllama_html] called pllamaEosTokenGet at ${DateTime.now()}');
    return completer.future;
  } catch (e) {
    // ignore: avoid_print
    print('[pllama_html] pllamaBosTokenGet caught error: $e');
    rethrow;
  }
}

@JS('pllamaEosTokenGetJs')
external JSPromise<JSString> pllamaEosTokenGetJs(String modelPath);

/// Returns the EOS token embedded in the .gguf file.
/// If none is found, returns an empty string.
///
/// See [pllamaApplyChatTemplate] for using sensible fallbacks for gguf
/// files that don't have an EOS token or have incorrect EOS tokens.
Future<String> pllamaEosTokenGet(String modelPath) {
  try {
    final completer = Completer<String>();
    // print('[pllama_html] calling pllamaEosTokenGet at ${DateTime.now()}');
    pllamaEosTokenGetJs(modelPath).toDart.then((value) {
      // print(
      // '[pllama_html] pllamaEosTokenGet finished with $value at ${DateTime.now()}');
      completer.complete(value.toDart);
    });
    // print('[pllama_html] called pllamaEosTokenGet at ${DateTime.now()}');
    return completer.future;
  } catch (e) {
    // ignore: avoid_print
    print('[pllama_html] pllamaEosTokenGet caught error: $e');
    rethrow;
  }
}

@JS('pllamaCancelInferenceJs')
external void pllamaCancelInferenceJs(int requestId);

/// Cancels the inference with the given [requestId].
///
/// It is recommended you do _not_ update your state based on this.
/// Use the callbacks, like you would generally.
///
/// This is supported via:
/// - Inferences that have not yet started will call their callback with `done` set
///  to `true` and an empty string.
/// - Inferences that have started will call their callback with `done` set to
/// `true` and the final output of the inference.
void pllamaCancelInference(int requestId) {
  pllamaCancelInferenceJs(requestId);
}

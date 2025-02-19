export 'io/pllama_io_inference.dart';
export 'io/pllama_io_tokenize.dart';

import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:pllama/pllama_universal.dart';
import 'package:pllama/io/pllama_bindings_generated.dart';
import 'package:pllama/io/pllama_io_helpers.dart';
import 'package:pllama/misc/openai.dart';

typedef FllamaInferenceCallback = void Function(String response, bool done);
typedef FllamaMlcLoadCallback = void Function(
    double downloadProgress, double loadProgress);

/// The dynamic library in which the symbols for [FllamaBindings] can be found.
final DynamicLibrary pllamaDylib = () {
  const String pllamaLibName = 'pllama';
  if (Platform.isMacOS || Platform.isIOS) {
    return DynamicLibrary.open('$pllamaLibName.framework/$pllamaLibName');
  }
  if (Platform.isAndroid || Platform.isLinux) {
    return DynamicLibrary.open('lib$pllamaLibName.so');
  }
  if (Platform.isWindows) {
    return DynamicLibrary.open('$pllamaLibName.dll');
  }
  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

/// The bindings to the native functions in [pllamaDylib].
final FllamaBindings pllamaBindings = FllamaBindings(pllamaDylib);

/// Returns the chat template embedded in the .gguf file.
/// If none is found, returns an empty string.
///
/// See [pllamaSanitizeChatTemplate] for using sensible fallbacks for gguf
/// files that don't have a chat template or have incorrect chat templates.
Future<String> pllamaChatTemplateGet(String modelPath) {
  // Example models without a chat template:
  // - Phi 2 has no template, either intended or in the model.
  // - Mistral 7B via OpenHermes has no template and intends ChatML.
  final filenamePointer = stringToPointerChar(modelPath);
  final templatePointer =
      pllamaBindings.pllama_get_chat_template(filenamePointer);
  calloc.free(filenamePointer);
  if (templatePointer == nullptr) {
    return Future.value('');
  }
  final builtInChatTemplate = pointerCharToString(templatePointer);
  return Future.value(builtInChatTemplate);
}

/// Returns the EOS token embedded in the .gguf file.
/// If none is found, returns an empty string.
///
/// See [pllamaApplyChatTemplate] for using sensible fallbacks for gguf
/// files that don't have an EOS token or have incorrect EOS tokens.
Future<String> pllamaEosTokenGet(String modelPath) async {
  final filenamePointer = stringToPointerChar(modelPath);
  final eosTokenPointer = pllamaBindings.pllama_get_eos_token(filenamePointer);
  calloc.free(filenamePointer);
  if (eosTokenPointer == nullptr) {
    return '';
  }
  return pointerCharToString(eosTokenPointer);
}

/// Returns the EOS token embedded in the .gguf file.
/// If none is found, returns an empty string.
///
/// See [pllamaApplyChatTemplate] for using sensible fallbacks for gguf
/// files that don't have an EOS token or have incorrect EOS tokens.
Future<String> pllamaBosTokenGet(String modelPath) async {
  final filenamePointer = stringToPointerChar(modelPath);
  final eosTokenPointer = pllamaBindings.pllama_get_bos_token(filenamePointer);
  calloc.free(filenamePointer);
  if (eosTokenPointer == nullptr) {
    return '';
  }
  return pointerCharToString(eosTokenPointer);
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
  // ignore: avoid_print
  print(
      'WARNING: called pllamaChatMlcWeb on native platform. Using pllamaChat instead.');
  return pllamaChat(request, callback);
}

Future<void> pllamaMlcWebModelDelete(String modelId) async {
  // ignore: avoid_print
  print(
      'WARNING: called pllamaMlcWebModelDelete on native platform. Ignoring.');
}

Future<bool> pllamaMlcIsWebModelDownloaded(String modelId) async {
  // ignore: avoid_print
  print(
      'WARNING: called pllamaMlcIsWebModelDownloaded on native platform. Returning false.');
  return false;
}

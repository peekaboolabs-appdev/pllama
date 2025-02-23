// ignore_for_file: always_specify_types
// ignore_for_file: camel_case_types
// ignore_for_file: non_constant_identifier_names

// AUTO GENERATED FILE, DO NOT EDIT.
//
// Generated by `package:ffigen`.
// ignore_for_file: type=lint
import 'dart:ffi' as ffi;

/// Bindings for `src/pllama.h`.
///
/// Regenerate bindings with `flutter pub run ffigen --config ffigen.yaml`.
///
class FllamaBindings {
  /// Holds the symbol lookup function.
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
      _lookup;

  /// The symbols are looked up in [dynamicLibrary].
  FllamaBindings(ffi.DynamicLibrary dynamicLibrary)
      : _lookup = dynamicLibrary.lookup;

  /// The symbols are looked up with [lookup].
  FllamaBindings.fromLookup(
      ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
          lookup)
      : _lookup = lookup;

  void pllama_inference(
    pllama_inference_request request,
    pllama_inference_callback callback,
  ) {
    return _pllama_inference(
      request,
      callback,
    );
  }

  late final _pllama_inferencePtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(pllama_inference_request,
              pllama_inference_callback)>>('pllama_inference');
  late final _pllama_inference = _pllama_inferencePtr.asFunction<
      void Function(pllama_inference_request, pllama_inference_callback)>();

  void pllama_inference_sync(
    pllama_inference_request request,
    pllama_inference_callback callback,
  ) {
    return _pllama_inference_sync(
      request,
      callback,
    );
  }

  late final _pllama_inference_syncPtr = _lookup<
      ffi.NativeFunction<
          ffi.Void Function(pllama_inference_request,
              pllama_inference_callback)>>('pllama_inference_sync');
  late final _pllama_inference_sync = _pllama_inference_syncPtr.asFunction<
      void Function(pllama_inference_request, pllama_inference_callback)>();

  void pllama_inference_cancel(
    int request_id,
  ) {
    return _pllama_inference_cancel(
      request_id,
    );
  }

  late final _pllama_inference_cancelPtr =
      _lookup<ffi.NativeFunction<ffi.Void Function(ffi.Int)>>(
          'pllama_inference_cancel');
  late final _pllama_inference_cancel =
      _pllama_inference_cancelPtr.asFunction<void Function(int)>();

  /// Chat template functions
  ffi.Pointer<ffi.Char> pllama_get_chat_template(
    ffi.Pointer<ffi.Char> fname,
  ) {
    return _pllama_get_chat_template(
      fname,
    );
  }

  late final _pllama_get_chat_templatePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<ffi.Char> Function(
              ffi.Pointer<ffi.Char>)>>('pllama_get_chat_template');
  late final _pllama_get_chat_template = _pllama_get_chat_templatePtr
      .asFunction<ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Char>)>();

  ffi.Pointer<ffi.Char> pllama_get_bos_token(
    ffi.Pointer<ffi.Char> fname,
  ) {
    return _pllama_get_bos_token(
      fname,
    );
  }

  late final _pllama_get_bos_tokenPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<ffi.Char> Function(
              ffi.Pointer<ffi.Char>)>>('pllama_get_bos_token');
  late final _pllama_get_bos_token = _pllama_get_bos_tokenPtr
      .asFunction<ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Char>)>();

  ffi.Pointer<ffi.Char> pllama_get_eos_token(
    ffi.Pointer<ffi.Char> fname,
  ) {
    return _pllama_get_eos_token(
      fname,
    );
  }

  late final _pllama_get_eos_tokenPtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<ffi.Char> Function(
              ffi.Pointer<ffi.Char>)>>('pllama_get_eos_token');
  late final _pllama_get_eos_token = _pllama_get_eos_tokenPtr
      .asFunction<ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Char>)>();

  int pllama_tokenize(
    pllama_tokenize_request request,
  ) {
    return _pllama_tokenize(
      request,
    );
  }

  late final _pllama_tokenizePtr =
      _lookup<ffi.NativeFunction<ffi.Size Function(pllama_tokenize_request)>>(
          'pllama_tokenize');
  late final _pllama_tokenize =
      _pllama_tokenizePtr.asFunction<int Function(pllama_tokenize_request)>();
}

final class pllama_inference_request extends ffi.Struct {
  /// Required: unique ID for the request. Used for cancellation.
  @ffi.Int()
  external int request_id;

  /// Required: context size
  @ffi.Int()
  external int context_size;

  /// Required: input text
  external ffi.Pointer<ffi.Char> input;

  /// Required: max tokens to generate
  @ffi.Int()
  external int max_tokens;

  /// Required: .ggml model file path
  external ffi.Pointer<ffi.Char> model_path;

  /// Optional: .mmproj file for multimodal models.
  external ffi.Pointer<ffi.Char> model_mmproj_path;

  /// Required: number of GPU layers. 0 for CPU only. 99 for
  /// all layers. Automatically 0 on iOS simulator.
  @ffi.Int()
  external int num_gpu_layers;

  /// Required: 2 recommended. Platforms can be highly sensitive
  /// to this, ex. Android stopped working with 4 suddenly.
  @ffi.Int()
  external int num_threads;

  /// Optional: temperature. Defaults to 0. (llama.cpp behavior)
  @ffi.Float()
  external double temperature;

  /// Optional: 0 < top_p <= 1. Defaults to 1. (llama.cpp behavior)
  @ffi.Float()
  external double top_p;

  /// Optional: 0 <= penalty_freq <= 1. Defaults to 0.0,
  /// which means disabled. (llama.cpp behavior)
  @ffi.Float()
  external double penalty_freq;

  /// Optional: 0 <= penalty_repeat <= 1. Defaults to 1.0,
  /// which means disabled. (llama.cpp behavior)
  @ffi.Float()
  external double penalty_repeat;

  /// Optional: BNF-like grammar to constrain sampling. Defaults to
  /// "" (llama.cpp behavior). See
  /// https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
  external ffi.Pointer<ffi.Char> grammar;

  /// Optional: end of sequence token. Defaults to one in model file. (llama.cpp behavior)
  /// For example, in ChatML / OpenAI, <|im_end|> means the message is complete.
  /// Often times GGUF files were created incorrectly, and this should be overridden.
  /// Using pllamaChat from Dart handles this automatically.
  external ffi.Pointer<ffi.Char> eos_token;

  /// Optional: Dart caller logger. Defaults to NULL.
  external pllama_log_callback dart_logger;
}

typedef pllama_log_callback
    = ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Char>)>>;
typedef pllama_inference_callback = ffi.Pointer<
    ffi.NativeFunction<
        ffi.Void Function(ffi.Pointer<ffi.Char> response, ffi.Uint8 done)>>;

final class pllama_tokenize_request extends ffi.Struct {
  /// Required: input text
  external ffi.Pointer<ffi.Char> input;

  /// Required: .ggml model file path
  external ffi.Pointer<ffi.Char> model_path;
}

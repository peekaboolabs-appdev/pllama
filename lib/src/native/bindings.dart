import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

// Native type definitions
typedef NativeInferenceCallback = Void Function(Pointer<Char> response, Int8 done);
typedef DartInferenceCallback = void Function(Pointer<Char> response, int done);

// Native function type definitions
typedef InferenceNative = Void Function(
    Pointer<pllama_inference_request>,
    Pointer<NativeFunction<NativeInferenceCallback>>);
typedef InferenceDart = void Function(
    Pointer<pllama_inference_request>,
    Pointer<NativeFunction<NativeInferenceCallback>>);

typedef TokenizeNative = Size Function(Pointer<pllama_tokenize_request>);
typedef TokenizeDart = int Function(Pointer<pllama_tokenize_request>);

/// FFI bindings to the native library
class PllamaBindings {
  late final DynamicLibrary _lib;
  late final InferenceDart pllama_inference;
  late final TokenizeDart pllama_tokenize;
  
  PllamaBindings() {
    _lib = _loadLibrary();
    pllama_inference = _lib
        .lookupFunction<InferenceNative, InferenceDart>('pllama_inference_sync');
    pllama_tokenize = _lib
        .lookupFunction<TokenizeNative, TokenizeDart>('pllama_tokenize');
  }

  // Helper to load the appropriate library for the current platform
  DynamicLibrary _loadLibrary() {
    const libName = 'pllama';
    if (Platform.isMacOS || Platform.isIOS) {
      return DynamicLibrary.open('$libName.framework/$libName');
    }
    if (Platform.isAndroid || Platform.isLinux) {
      return DynamicLibrary.open('lib$libName.so');
    }
    if (Platform.isWindows) {
      return DynamicLibrary.open('$libName.dll');
    }
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }
}

// Native struct definitions
final class pllama_inference_request extends Struct {
  external Pointer<Utf8> input;
  external Pointer<Utf8> model_path;
  external Pointer<Utf8> grammar;
  external Pointer<Utf8> eos_token;
  
  @Int32()
  external int context_size;
  
  @Int32()
  external int max_tokens;
  
  @Int32()
  external int num_gpu_layers;
  
  @Int32()
  external int num_threads;
  
  @Float()
  external double temperature;
  
  @Float()
  external double top_p;
  
  @Float()
  external double penalty_freq;
  
  @Float()
  external double penalty_repeat;
}

final class pllama_tokenize_request extends Struct {
  external Pointer<Utf8> input;
  external Pointer<Utf8> model_path;
}

// Global instance
final pllamaBindings = PllamaBindings();
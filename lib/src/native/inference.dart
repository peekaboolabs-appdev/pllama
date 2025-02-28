import 'dart:async';
import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import '../types.dart';
import '../constants.dart';
import '../errors.dart';

// 콜백 타입 정의 - C++ 시그니처와 일치
typedef NativeInferenceCallback = Void Function(Pointer<Char> response, Uint8 done);
typedef DartInferenceCallback = void Function(Pointer<Char> response, int done);

// 네이티브 함수 타입 정의
typedef InferenceNativeFunc = Void Function(
    Pointer<pllama_inference_request>,
    Pointer<NativeFunction<NativeInferenceCallback>>);
typedef InferenceDartFunc = void Function(
    Pointer<pllama_inference_request>,
    Pointer<NativeFunction<NativeInferenceCallback>>);

typedef TokenizeNativeFunc = Int32 Function(Pointer<pllama_tokenize_request>);
typedef TokenizeDartFunc = int Function(Pointer<pllama_tokenize_request>);

/// Native struct definitions
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

// 전역 콜백 인스턴스
Completer<void>? _activeCompleter;
void Function(String, bool)? _activeTokenCallback;

// 정적 네이티브 콜백 핸들러
void _nativeCallbackHandler(Pointer<Char> response, int done) {
  final responseStr = response.cast<Utf8>().toDartString();
  
  if (_activeTokenCallback != null) {
    _activeTokenCallback!(responseStr, done == 1);
  }
  
  if (done == 1 && _activeCompleter != null && !_activeCompleter!.isCompleted) {
    _activeCompleter!.complete();
  }
}

/// 네이티브 라이브러리 인터페이스
class NativeLibrary {
  static final NativeLibrary instance = NativeLibrary._();
  late final DynamicLibrary _lib;
  late final InferenceDartFunc _pllama_inference;
  late final TokenizeDartFunc _pllama_tokenize;

  NativeLibrary._() {
    try {
      _lib = _loadLibrary();
      _pllama_inference = _lib
          .lookupFunction<InferenceNativeFunc, InferenceDartFunc>('pllama_inference_sync');
      _pllama_tokenize = _lib
          .lookupFunction<TokenizeNativeFunc, TokenizeDartFunc>('pllama_tokenize');
    } catch (e) {
      print("Failed to initialize native bindings: $e");
      rethrow;
    }
  }

  // 라이브러리 로드 헬퍼
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

  void pllama_inference_sync(Pointer<pllama_inference_request> request) {
    // 정적 콜백 함수 등록 - exceptionalReturn 제거
    final callbackPtr = Pointer.fromFunction<NativeInferenceCallback>(
      _nativeCallbackHandler,
    );
    
    _pllama_inference(request, callbackPtr);
  }

  int pllama_tokenize(Pointer<pllama_tokenize_request> request) {
    return _pllama_tokenize(request);
  }
}

/// Native inference and tokenization handler
class InferenceNative {
  /// Runs inference using the native library
  static Future<void> runInference({
    required InferenceConfig config,
    required String modelPath,
    required void Function(String response, bool done) onToken,
  }) async {
    // Validate inputs
    if (modelPath.isEmpty) {
      throw ModelLoadError(
        message: 'Model path cannot be empty', 
        modelPath: modelPath
      );
    }

    // Check model file exists
    final modelFile = File(modelPath);
    if (!await modelFile.exists()) {
      throw ModelLoadError(
        message: 'Model file does not exist', 
        modelPath: modelPath
      );
    }

    // 이전 콜백이 있으면 제거
    _activeCompleter = Completer<void>();
    
    // 네이티브 요청 준비
    final request = calloc<pllama_inference_request>();
    
    // Input과 model path를 네이티브 메모리로 변환
    final inputPtr = config.prompt.toNativeUtf8();
    final modelPathPtr = modelPath.toNativeUtf8();
    
    // 요청 파라미터 설정
    request.ref.input = inputPtr;
    request.ref.model_path = modelPathPtr;
    request.ref.context_size = config.contextSize;
    request.ref.max_tokens = config.maxTokens;
    request.ref.temperature = config.temperature;
    request.ref.top_p = config.topP;
    request.ref.num_threads = config.numThreads;
    request.ref.num_gpu_layers = config.numGpuLayers;

    try {
      // 토큰 콜백 설정
      _activeTokenCallback = onToken;

      // Perform native inference
      final nativeLib = NativeLibrary.instance;
      nativeLib.pllama_inference_sync(request);

      // Wait for inference to complete
      await _activeCompleter!.future.timeout(
        const Duration(minutes: 5),
        onTimeout: () {
          throw InferenceError(
            'Inference timed out after 5 minutes', 
            maxTokens: config.maxTokens
          );
        }
      );
    } catch (e) {
      if (_activeCompleter != null && !_activeCompleter!.isCompleted) {
        _activeCompleter!.completeError(
          InferenceError(
            'Native inference failed: ${e.toString()}', 
            maxTokens: config.maxTokens
          )
        );
      }
      rethrow;
    } finally {
      // 정리
      _activeTokenCallback = null;
      _activeCompleter = null;
      
      // 네이티브 메모리 해제
      calloc.free(inputPtr);
      calloc.free(modelPathPtr);
      calloc.free(request);
    }
  }

  /// Runs tokenization using the native library
  static Future<int> runTokenize(TokenizeRequest request) async {
    // Validate inputs
    if (request.input.isEmpty) {
      throw TokenizationError(
        message: 'Input text cannot be empty',
        input: request.input,
        modelPath: request.modelPath
      );
    }

    // Check model file exists
    final modelFile = File(request.modelPath);
    if (!await modelFile.exists()) {
      throw TokenizationError(
        message: 'Model file does not exist',
        input: request.input,
        modelPath: request.modelPath
      );
    }

    final completer = Completer<int>();

    try {
      // 네이티브 토큰화 요청 준비
      final nativeRequest = calloc<pllama_tokenize_request>();
      final inputPtr = request.input.toNativeUtf8();
      final modelPathPtr = request.modelPath.toNativeUtf8();
      
      nativeRequest.ref.input = inputPtr;
      nativeRequest.ref.model_path = modelPathPtr;
      
      // 네이티브 라이브러리 호출
      final nativeLib = NativeLibrary.instance;
      final tokenCount = nativeLib.pllama_tokenize(nativeRequest);

      if (tokenCount < 0) {
        throw TokenizationError(
          message: 'Tokenization failed with error code: $tokenCount',
          input: request.input,
          modelPath: request.modelPath
        );
      }

      // 네이티브 메모리 해제
      calloc.free(inputPtr);
      calloc.free(modelPathPtr);
      calloc.free(nativeRequest);

      completer.complete(tokenCount);
    } catch (e) {
      completer.completeError(
        TokenizationError(
          message: 'Native tokenization failed: ${e.toString()}',
          input: request.input,
          modelPath: request.modelPath
        )
      );
    }

    return completer.future.timeout(
      const Duration(seconds: 30),
      onTimeout: () {
        throw TokenizationError(
          message: 'Tokenization timed out',
          input: request.input,
          modelPath: request.modelPath
        );
      }
    );
  }
}
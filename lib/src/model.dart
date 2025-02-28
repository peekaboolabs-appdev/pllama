/// High-level Dart bindings for LLaMA models
library pllama;

import 'dart:async';
import 'dart:math' as math;
import 'dart:io';

import 'types.dart';
import 'constants.dart';
import 'errors.dart';
import 'native/inference.dart';

/// High-level interface for interacting with LLaMA models
class PllamaModel {
  final String _modelPath;
  bool _isDisposed = false;
  
  PllamaModel._(this._modelPath);

  /// Creates a PllamaModel instance from a model file
  static Future<PllamaModel> fromFile(String modelPath, {
    int? numThreads,
    int? contextSize,
    int? numGpuLayers,
  }) async {
    // Validate model path
    final modelFile = File(modelPath);
    if (!await modelFile.exists()) {
      throw ModelLoadError(
        message: 'Model file does not exist: $modelPath', 
        modelPath: modelPath
      );
    }

    final model = PllamaModel._(modelPath);
    
    try {
      // Enhanced model loading with multiple strategies
      await Future.any([
        _loadModelWithOptimizedSettings(
          model, 
          fileSize: await modelFile.length(),
          numThreads: numThreads,
          contextSize: contextSize,
          numGpuLayers: numGpuLayers,
        ),
        Future.delayed(
          Duration(minutes: 2), 
          () => throw TimeoutException('Model loading timed out')
        )
      ]);
      
      return model;
    } catch (e) {
      // Comprehensive error handling
      if (e is TimeoutException) {
        throw ModelLoadError(
          message: 'Model loading took too long', 
          modelPath: modelPath
        );
      } else if (e is ModelLoadError) {
        rethrow;
      } else {
        throw ModelLoadError(
          message: 'Unexpected error during model loading', 
          modelPath: modelPath
        );
      }
    }
  }

  /// Internal method for loading model with optimized settings
  static Future<void> _loadModelWithOptimizedSettings(
    PllamaModel model, {
    required int fileSize,
    int? numThreads,
    int? contextSize,
    int? numGpuLayers,
  }) async {
    // Platform-specific optimizations
    if (Platform.isAndroid) {
      numThreads = math.min(2, Platform.numberOfProcessors);
      contextSize ??= 512;
      numGpuLayers = 0;
    }

    // Large model handling
    if (fileSize > 1 * 1024 * 1024 * 1024) { // 1GB+
      contextSize = 256;
      numThreads = 1;
      numGpuLayers = 0;
    }

    // Multiple fallback loading attempts
    final loadAttempts = [
      () => model._testGenerate(
        prompt: 'test',
        maxTokens: 1,
        temperature: 0.0,
        topP: 1.0,
        contextSize: contextSize ?? PllamaDefaults.contextSize,
        numThreads: numThreads ?? PllamaDefaults.numThreads,
        numGpuLayers: numGpuLayers ?? PllamaDefaults.numGpuLayers,
      ),
      () => model._testGenerate(
        prompt: 'test minimal',
        maxTokens: 1,
        temperature: 0.1,
        topP: 1.0,
        contextSize: 128,
        numThreads: 1,
        numGpuLayers: 0,
      )
    ];

    for (var attempt in loadAttempts) {
      try {
        await attempt();
        return;
      } catch (e) {
        // Log or handle specific loading failures
        print('Model loading attempt failed: $e');
      }
    }

    throw ModelLoadError(
      message: 'All model loading attempts failed', 
      modelPath: model._modelPath
    );
  }

  /// Internal test generation method
  Future<String> _testGenerate({
    required String prompt,
    int? maxTokens,
    double? temperature,
    double? topP,
    int? contextSize,
    int? numThreads,
    int? numGpuLayers,
  }) async {
    final config = InferenceConfig(
      prompt: prompt,
      maxTokens: maxTokens ?? PllamaDefaults.maxTokens,
      temperature: temperature ?? PllamaDefaults.temperature,
      topP: topP ?? PllamaDefaults.topP,
      contextSize: contextSize ?? PllamaDefaults.contextSize,
      numThreads: numThreads ?? PllamaDefaults.numThreads,
      numGpuLayers: numGpuLayers ?? PllamaDefaults.numGpuLayers,
    );

    final completer = Completer<String>();
    String fullResponse = '';

    try {
      await InferenceNative.runInference(
        config: config,
        modelPath: _modelPath,
        onToken: (response, done) {
          fullResponse += response;
          if (done) completer.complete(fullResponse);
        },
      ).timeout(
        Duration(seconds: 120),  // Extended timeout
        onTimeout: () {
          if (!completer.isCompleted) {
            completer.completeError(
              InferenceError('Test generation timed out after 120 seconds')
            );
          }
        },
      );
    } catch (e) {
      completer.completeError(
        InferenceError('Test generation failed: $e')
      );
    }

    return completer.future;
  }

  /// Generates text completion
  Future<String> generate({
    required String prompt,
    int? maxTokens,
    double? temperature,
    double? topP,
    int? contextSize,
    int? numThreads,
    int? numGpuLayers,
  }) async {
    _checkDisposed();

    final config = InferenceConfig(
      prompt: prompt,
      maxTokens: maxTokens ?? PllamaDefaults.maxTokens,
      temperature: temperature ?? PllamaDefaults.temperature,
      topP: topP ?? PllamaDefaults.topP,
      contextSize: contextSize ?? PllamaDefaults.contextSize,
      numThreads: numThreads ?? PllamaDefaults.numThreads,
      numGpuLayers: numGpuLayers ?? PllamaDefaults.numGpuLayers,
    );

    final completer = Completer<String>();
    String fullResponse = '';

    try {
      await InferenceNative.runInference(
        config: config,
        modelPath: _modelPath,
        onToken: (response, done) {
          fullResponse += response;
          if (done) completer.complete(fullResponse);
        },
      ).timeout(
        Duration(seconds: 120),  // Extended timeout
        onTimeout: () {
          if (!completer.isCompleted) {
            completer.completeError(
              InferenceError('Generation timed out after 120 seconds')
            );
          }
        },
      );
    } catch (e) {
      completer.completeError(
        InferenceError('Generation failed: $e')
      );
    }

    return completer.future;
  }

  /// Generates text completion in a streaming manner
  Stream<String> generateStream({
    required String prompt,
    int? maxTokens,
    double? temperature,
    double? topP,
    int? contextSize,
    int? numThreads,
    int? numGpuLayers,
  }) async* {
    _checkDisposed();

    final config = InferenceConfig(
      prompt: prompt,
      maxTokens: maxTokens ?? PllamaDefaults.maxTokens,
      temperature: temperature ?? PllamaDefaults.temperature,
      topP: topP ?? PllamaDefaults.topP,
      contextSize: contextSize ?? PllamaDefaults.contextSize,
      numThreads: numThreads ?? PllamaDefaults.numThreads,
      numGpuLayers: numGpuLayers ?? PllamaDefaults.numGpuLayers,
    );

    final controller = StreamController<String>();
    
    try {
      await InferenceNative.runInference(
        config: config,
        modelPath: _modelPath,
        onToken: (response, done) {
          if (!done) {
            controller.add(response);
          } else {
            controller.close();
          }
        },
      ).timeout(
        Duration(seconds: 120),  // Extended timeout
        onTimeout: () {
          controller.addError(
            InferenceError('Streaming generation timed out after 120 seconds')
          );
          controller.close();
        },
      );
    } catch (e) {
      controller.addError(
        InferenceError('Streaming generation failed: $e')
      );
      await controller.close();
    }

    yield* controller.stream;
  }

  /// Returns the number of tokens in the input text
  Future<int> tokenCount(String text) async {
    _checkDisposed();
    final request = TokenizeRequest(
      input: text,
      modelPath: _modelPath,
    );
    
    try {
      return await InferenceNative.runTokenize(request).timeout(
        Duration(seconds: 10),
        onTimeout: () => throw TimeoutException('Tokenization timed out'),
      );
    } catch (e) {
      throw TokenizationError(
        message: 'Tokenization failed: $e', 
        input: text, 
        modelPath: _modelPath
      );
    }
  }

  /// Cleans up resources associated with the model
  void dispose() {
    _isDisposed = true;
  }

  void _checkDisposed() {
    if (_isDisposed) {
      throw ModelDisposedError();
    }
  }
}
/// Base class for all pllama-specific errors
abstract class PllamaError extends Error {
  final String message;
  final dynamic cause;
  
  PllamaError(this.message, [this.cause]);
  
  @override
  String toString() {
    if (cause != null) {
      return 'PllamaError: $message\nCaused by: $cause';
    }
    return 'PllamaError: $message';
  }
}

/// Thrown when there are issues loading or initializing a model
class ModelLoadError extends PllamaError {
  final String modelPath;

  ModelLoadError({
    required String message, 
    required this.modelPath, 
    dynamic cause,
  }) : super(message, cause);
}

/// Thrown when trying to use a disposed model
class ModelDisposedError extends PllamaError {
  ModelDisposedError() : super('Model has been disposed');
}

/// Thrown when inference fails
class InferenceError extends PllamaError {
  final double? temperature;
  final int? maxTokens;

  InferenceError(
    String message, {
    this.temperature,
    this.maxTokens,
    dynamic cause,
  }) : super(message, cause);
}

/// Thrown when tokenization fails
class TokenizationError extends PllamaError {
  final String? input;
  final String? modelPath;

  TokenizationError({
    required String message,
    this.input,
    this.modelPath,
    dynamic cause,
  }) : super(message, cause);
}

/// Thrown when the model file is invalid or corrupted
class InvalidModelError extends PllamaError {
  final String modelPath;

  InvalidModelError({
    required String message, 
    required this.modelPath, 
    dynamic cause,
  }) : super(message, cause);
}

/// Thrown when platform-specific operations fail
class PlatformError extends PllamaError {
  final String? platform;

  PlatformError({
    required String message,
    this.platform,
    dynamic cause,
  }) : super(message, cause);
}

/// Helper methods for error handling
extension PllamaErrorHelpers on PllamaError {
  /// Whether this error is related to model loading
  bool get isModelError => 
    this is ModelLoadError || this is InvalidModelError;
  
  /// Whether this error is related to inference
  bool get isInferenceError => this is InferenceError;
  
  /// Whether this error is related to resource management
  bool get isResourceError => this is ModelDisposedError;
  
  /// Whether this error is platform-specific
  bool get isPlatformError => this is PlatformError;
}
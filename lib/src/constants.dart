/// Default values and constants for the pllama package
class PllamaDefaults {
  /// Default temperature for generation (0.7 balances creativity and coherence)
  static const double temperature = 0.7;
  
  /// Default top-p for nucleus sampling
  static const double topP = 0.95;
  
  /// Default maximum tokens to generate (~250 words)
  static const int maxTokens = 333;
  
  /// Default context size in tokens (safe for most devices)
  static const int contextSize = 2048;
  
  /// Default number of threads for inference
  static const int numThreads = 2;
  
  /// Default number of GPU layers (0 means CPU-only)
  static const int numGpuLayers = 0;
  
  /// Default frequency penalty
  static const double frequencyPenalty = 0.0;
  
  /// Default presence penalty
  static const double presencePenalty = 1.1;
  
  /// Minimum model file size (1MB)
  static const int minModelSize = 1024 * 1024;
  
  /// Default end of sequence token
  static const String eosToken = "</s>";
  
  /// Default beginning of sequence token
  static const String bosToken = "<s>";

  /// Platform-specific configuration hints
  static Map<String, dynamic> getPlatformOptimizedConfig(String platform) {
    switch (platform) {
      case 'android':
        return {
          'numThreads': 2,
          'contextSize': 512,
          'numGpuLayers': 0,
        };
      case 'ios':
        return {
          'numThreads': 2,
          'contextSize': 512,
          'numGpuLayers': 0,
        };
      case 'windows':
        return {
          'numThreads': 4,
          'contextSize': 2048,
          'numGpuLayers': 4,
        };
      case 'macos':
        return {
          'numThreads': 4,
          'contextSize': 2048,
          'numGpuLayers': 8,
        };
      default:
        return {
          'numThreads': numThreads,
          'contextSize': contextSize,
          'numGpuLayers': numGpuLayers,
        };
    }
  }

  /// Validates generation parameters
  static bool validateGenerationParams({
    double? temperature,
    double? topP,
    int? maxTokens,
  }) {
    if (temperature != null && (temperature < 0.0 || temperature > 2.0)) {
      return false;
    }
    if (topP != null && (topP <= 0.0 || topP > 1.0)) {
      return false;
    }
    if (maxTokens != null && maxTokens <= 0) {
      return false;
    }
    return true;
  }
}

/// Internal constants used by the package
class PllamaInternals {
  /// Maximum time to wait for inference to start
  static const Duration inferenceTimeout = Duration(seconds: 30);
  
  /// Maximum time to wait for model loading
  static const Duration loadTimeout = Duration(minutes: 2);
  
  /// Buffer size for native string operations
  static const int stringBufferSize = 4096;

  /// Logging configuration
  static const bool verboseLogging = false;
}
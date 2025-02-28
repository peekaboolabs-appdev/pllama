/// Configuration for model inference
class InferenceConfig {
  final String prompt;
  final double temperature;
  final int maxTokens;
  final double topP;
  final int contextSize;
  final int numThreads;
  final int numGpuLayers;
  final bool optimizeForLargeModel;
  final int loadTimeoutSeconds;

  InferenceConfig({
    required this.prompt,
    this.temperature = 0.7,
    this.maxTokens = 333,
    this.topP = 0.95,
    this.contextSize = 2048,
    this.numThreads = 2,
    this.numGpuLayers = 0,
    this.optimizeForLargeModel = true,
    this.loadTimeoutSeconds = 240,
  });
}

/// Request for tokenization
class TokenizeRequest {
  final String input;
  final String modelPath;

  TokenizeRequest({
    required this.input,
    required this.modelPath,
  });
}
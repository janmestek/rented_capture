import 'dart:typed_data';

/// Web: TFLite není k dispozici, capture běží bez embeddingů.
class Embedder {
  int size = 224;
  int outLen = 1024;
  bool ready = false;
  String status = 'model není na webu k dispozici';

  Future<void> load() async {}

  List<double> run(Float32List flat) => const [];

  void close() {}
}

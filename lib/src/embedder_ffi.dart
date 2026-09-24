import 'dart:math';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';

const _modelPath = 'packages/rented_capture/assets/mobilenet.tflite';

class Embedder {
  Interpreter? _it;
  int size = 224;
  int outLen = 1024;
  bool ready = false;
  String status = 'načítám model…';

  Future<void> load() async {
    try {
      final opts = InterpreterOptions()
        ..threads = 2
        ..addDelegate(XNNPackDelegate());
      _it = await Interpreter.fromAsset(_modelPath, options: opts);
    } catch (_) {
      try {
        _it = await Interpreter.fromAsset(_modelPath,
            options: InterpreterOptions()..threads = 2);
      } catch (e) {
        status = 'model SELHAL: $e';
        return;
      }
    }
    size = _it!.getInputTensor(0).shape[1];
    outLen = _it!.getOutputTensor(0).shape.last;
    ready = true;
    status = 'model OK';
  }

  List<double> run(Float32List flat) {
    final it = _it;
    if (it == null || !ready) return const [];
    final input = flat.reshape([1, size, size, 3]);
    final output = List.generate(1, (_) => List<double>.filled(outLen, 0));
    try {
      it.run(input, output);
    } catch (_) {
      return const [];
    }
    final v = List<double>.from(output[0]);
    double norm = 0;
    for (final x in v) {
      norm += x * x;
    }
    norm = sqrt(norm);
    if (norm < 1e-6) return const [];
    for (int i = 0; i < v.length; i++) {
      v[i] /= norm;
    }
    return v;
  }

  void close() => _it?.close();
}

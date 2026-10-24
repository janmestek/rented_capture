import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:rented_capture/rented_capture.dart';

void main() {
  // Raw BGRA 40×30, R = x, G = y. Telefon na výšku: senzor otočený o 90°.
  const w = 40, h = 30;
  final raw = Uint8List(w * h * 4);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final i = (y * w + x) * 4;
      raw[i + 2] = x;
      raw[i + 1] = y;
    }
  }
  final port = ReceivePort().sendPort;

  test('rot 90: box ve vzpřímených souřadnicích, výstup vzpřímený', () {
    // vzpřímený obraz má 30×40, box přes celý obraz
    final out = prepCrop(
        PrepJob(raw, w, h, 30, true, 0, 0, 29.9, 39.9, port, 90, w * 4))!;
    double r(int x, int y) => out[(y * 30 + x) * 3] * 255;
    double g(int x, int y) => out[(y * 30 + x) * 3 + 1] * 255;
    // vzpřímený levý horní roh = raw levý dolní (x=0, y=h-1)
    expect(r(0, 0).round(), 0);
    expect(g(0, 0).round(), h - 1);
    // posun doprava ve vzpřímeném = nahoru v raw
    expect(g(10, 0).round(), h - 1 - 9);
    // posun dolů ve vzpřímeném = doprava v raw
    expect(r(0, 10).round(), 13);
  });

  test('rot 0 beze změny', () {
    final out =
        prepCrop(PrepJob(raw, w, h, 30, true, 0, 0, 39.9, 29.9, port, 0, w * 4))!;
    expect((out[(0 * 30 + 29) * 3] * 255).round(), 39 * 29 ~/ 30);
  });

  test('NV21 neutrální chroma = šedá, ne jen jas', () {
    final nv = Uint8List(w * h * 3 ~/ 2)..fillRange(0, w * h * 3 ~/ 2, 128);
    for (int i = w * h; i < nv.length; i += 2) {
      nv[i] = 200; // V -> víc červené
    }
    final out =
        prepCrop(PrepJob(nv, w, h, 30, false, 0, 0, 39.9, 29.9, port, 0, w))!;
    expect(out[0], greaterThan(out[2])); // R > B
  });
}

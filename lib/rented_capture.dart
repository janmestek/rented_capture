import 'dart:async';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/scheduler.dart';
import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:camerawesome/pigeon.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
// tflite_flutter používá dart:ffi, na webu nejde zkompilovat -> stub.
import 'src/embedder_stub.dart' if (dart.library.ffi) 'src/embedder_ffi.dart';

const kBrand = Color(0xFFFF881A);
const kOk = Color(0xFF34E58B);
const kDim = Color(0x66000000);

double gCalX = 0.0;
double gCalY = 0.0;
double gCalScale = 1.0;

// ================= veřejné API =================

/// Jeden pohled, který má uživatel nasnímat.
class ViewSpec {
  final String id, label, instruction;
  final double minQuality;

  /// Referenční vektory (golden + learned dohromady).
  /// Prázdné = pohled se nerozpoznává, odškrtne se podle kvality záběru.
  final List<List<double>> refs;

  const ViewSpec({
    required this.id,
    required this.label,
    required this.instruction,
    this.minQuality = 0.45,
    this.refs = const [],
  });

  /// Vytvoření z Firestore mapy + vektorů ze Storage.
  factory ViewSpec.fromMap(Map<String, dynamic> m, List<List<double>> refs) {
    return ViewSpec(
      id: (m['view_id'] ?? m['id'] ?? '').toString(),
      label: (m['label'] ?? '').toString(),
      instruction: (m['instruction'] ?? '').toString(),
      minQuality: (m['min_quality'] as num?)?.toDouble() ?? 0.45,
      refs: refs,
    );
  }
}

/// Jeden pořízený snímek.
class Shot {
  final String? viewId; // null = průběžný
  final String path;
  final int atMs;
  final double quality;
  final double similarity;

  /// Vektor snímku – kandidát na novou referenci.
  final List<double> vector;

  Shot(this.viewId, this.path, this.atMs, this.quality, this.similarity,
      this.vector);

  Map<String, dynamic> toMap() => {
        'viewId': viewId,
        'path': path,
        'atMs': atMs,
        'quality': quality,
        'similarity': similarity,
        'vector': vector,
      };
}

/// Výsledek celé session.
class CaptureResult {
  final List<Shot> shots;
  final List<String> lockedViews;
  final int durationMs;
  final bool skipped;
  CaptureResult(this.shots, this.lockedViews, this.durationMs, this.skipped);

  List<String> get photoPaths => shots.map((s) => s.path).toList();
}

class PaintState {
  final Rect? box;
  final Size? imgSize;
  final double heat, hold, lockFlash, pulse;
  const PaintState(
      this.box, this.imgSize, this.heat, this.hold, this.lockFlash, this.pulse);
}

double cosine(List<double> a, List<double> b) {
  if (a.length != b.length || a.isEmpty) return 0;
  double d = 0;
  for (int i = 0; i < a.length; i++) {
    d += a[i] * b[i];
  }
  return d.clamp(-1.0, 1.0);
}

double bestAgainst(List<double> v, List<List<double>>? bank) {
  if (bank == null || bank.isEmpty || v.isEmpty) return -1;
  double best = -1;
  for (final r in bank) {
    final s = cosine(v, r);
    if (s > best) best = s;
  }
  return best;
}

// ================= isolate =================

class PrepJob {
  final Uint8List bytes;
  // imgW/imgH = buffer v orientaci senzoru, rot = otočení do vzpřímené polohy
  // (stupně po směru), stride = bajtů na řádek.
  final int imgW, imgH, size, rot, stride;
  final bool isBgra;
  // box od ML Kitu, ve vzpřímených souřadnicích
  final double l, t, r, b;
  final SendPort reply;
  PrepJob(this.bytes, this.imgW, this.imgH, this.size, this.isBgra, this.l,
      this.t, this.r, this.b, this.reply, this.rot, this.stride);
}

/// Vzpřímený bod (u, v) -> bod (x, y) v bufferu senzoru.
(int, int) uprightToRaw(int u, int v, int rot, int imgW, int imgH) {
  switch (rot) {
    case 90:
      return (v, imgH - 1 - u);
    case 180:
      return (imgW - 1 - u, imgH - 1 - v);
    case 270:
      return (imgW - 1 - v, u);
    default:
      return (u, v);
  }
}

/// Ořízne box, otočí ho do vzpřímené polohy a zmenší na size×size RGB 0..1.
/// Vrací null, když je box moc malý.
Float32List? prepCrop(PrepJob j) {
  final swap = j.rot == 90 || j.rot == 270;
  final uw = swap ? j.imgH : j.imgW, uh = swap ? j.imgW : j.imgH;
  final x0 = j.l.clamp(0, uw - 1).toInt();
  final x1 = j.r.clamp(0, uw - 1).toInt();
  final y0 = j.t.clamp(0, uh - 1).toInt();
  final y1 = j.b.clamp(0, uh - 1).toInt();
  final bw = x1 - x0, bh = y1 - y0;
  if (bw < 24 || bh < 24) return null;

  final bytes = j.bytes, len = bytes.length;
  // NV21: Y rovina, za ní prokládané V,U v poloviční velikosti
  final ySize = j.stride * j.imgH;
  final hasUv = !j.isBgra && len >= ySize + j.stride * (j.imgH ~/ 2);

  final out = Float32List(j.size * j.size * 3);
  int k = 0;
  for (int y = 0; y < j.size; y++) {
    final v = y0 + (bh * y) ~/ j.size;
    for (int x = 0; x < j.size; x++) {
      final u = x0 + (bw * x) ~/ j.size;
      final (sx, sy) = uprightToRaw(u, v, j.rot, j.imgW, j.imgH);
      if (j.isBgra) {
        final i = sy * j.stride + sx * 4;
        if (i < 0 || i + 2 >= len) {
          k += 3;
          continue;
        }
        out[k++] = bytes[i + 2] / 255.0;
        out[k++] = bytes[i + 1] / 255.0;
        out[k++] = bytes[i] / 255.0;
      } else {
        final i = sy * j.stride + sx;
        if (i < 0 || i >= len) {
          k += 3;
          continue;
        }
        final yv = bytes[i].toDouble();
        if (hasUv) {
          final c = ySize + (sy >> 1) * j.stride + (sx & ~1);
          final cv = bytes[c] - 128.0, cu = bytes[c + 1] - 128.0;
          out[k++] = (yv + 1.402 * cv).clamp(0, 255) / 255.0;
          out[k++] = (yv - 0.344 * cu - 0.714 * cv).clamp(0, 255) / 255.0;
          out[k++] = (yv + 1.772 * cu).clamp(0, 255) / 255.0;
        } else {
          out[k++] = yv / 255.0;
          out[k++] = yv / 255.0;
          out[k++] = yv / 255.0;
        }
      }
    }
  }
  return out;
}

void prepIsolate(SendPort main) {
  final port = ReceivePort();
  main.send(port.sendPort);

  port.listen((msg) {
    final j = msg as PrepJob;
    final out = prepCrop(j);
    if (out == null) {
      j.reply.send(null);
      return;
    }

    double sum = 0;
    int cnt = 0;
    for (int y = 0; y < j.size; y += 2) {
      for (int x = 0; x < j.size - 2; x += 2) {
        sum += (out[(y * j.size + x) * 3] - out[(y * j.size + x + 2) * 3]).abs();
        cnt++;
      }
    }

    j.reply.send([out, cnt == 0 ? 0.0 : sum / cnt * 255]);
  });
}

// ================= model =================

// ================= ML Kit =================

InputImageRotation _rot(InputAnalysisImageRotation r) {
  switch (r) {
    case InputAnalysisImageRotation.rotation0deg:
      return InputImageRotation.rotation0deg;
    case InputAnalysisImageRotation.rotation90deg:
      return InputImageRotation.rotation90deg;
    case InputAnalysisImageRotation.rotation180deg:
      return InputImageRotation.rotation180deg;
    case InputAnalysisImageRotation.rotation270deg:
      return InputImageRotation.rotation270deg;
  }
}

extension MlkitConvert on AnalysisImage {
  InputImage? toInput() => when(
        nv21: (img) => InputImage.fromBytes(
          bytes: img.bytes,
          metadata: InputImageMetadata(
            rotation: _rot(img.rotation),
            format: InputImageFormat.nv21,
            size: img.size,
            bytesPerRow: img.planes.first.bytesPerRow,
          ),
        ),
        bgra8888: (img) => InputImage.fromBytes(
          bytes: img.bytes,
          metadata: InputImageMetadata(
            rotation: _rot(img.rotation),
            format: InputImageFormat.bgra8888,
            size: img.size,
            bytesPerRow: img.planes.first.bytesPerRow,
          ),
        ),
      );

  Uint8List? get rawBytes =>
      when(nv21: (i) => i.bytes, bgra8888: (i) => i.bytes);

  bool get isBgra => format == InputAnalysisImageFormat.bgra8888;

  int get rotDeg => switch (rotation) {
        InputAnalysisImageRotation.rotation0deg => 0,
        InputAnalysisImageRotation.rotation90deg => 90,
        InputAnalysisImageRotation.rotation180deg => 180,
        InputAnalysisImageRotation.rotation270deg => 270,
      };

  // NV21 jde z camerawesome bez zarovnání řádků, BGRA má bytesPerRow.
  int get stride =>
      when(nv21: (i) => i.width, bgra8888: (i) => i.planes.first.bytesPerRow) ??
      width;

  /// Rozměr vzpřímeného obrazu, ve kterém ML Kit vrací boxy.
  Size get upright => rotDeg == 90 || rotDeg == 270
      ? Size(height.toDouble(), width.toDouble())
      : size;
}

// ================= widget =================

enum _Mode { loading, scan, done }

enum _Phase { searching, framing, holding, locked }

class CaptureSession extends StatefulWidget {
  final String? rentalId;
  final String? skuId;

  /// Pohledy včetně referenčních vektorů.
  final List<ViewSpec> views;

  /// Zavolá se po dokončení.
  final void Function(CaptureResult result)? onComplete;

  /// Práh podobnosti. Začni nízko, zvyšuj s rostoucí bankou.
  final double simThreshold;

  const CaptureSession({
    super.key,
    required this.views,
    this.rentalId,
    this.skuId,
    this.onComplete,
    this.simThreshold = 0.65,
  });

  @override
  State<CaptureSession> createState() => _CaptureSessionState();
}

class _CaptureSessionState extends State<CaptureSession>
    with SingleTickerProviderStateMixin {
  final detector = ObjectDetector(
    options: ObjectDetectorOptions(
      mode: DetectionMode.stream,
      classifyObjects: false,
      multipleObjects: false,
    ),
  );
  final embedder = Embedder();

  static const holdMs = 1000;
  static const cooldownMs = 1800;
  static const skipAfterMs = 12000;
  static const warmupMs = 1500;
  static const embedEveryMs = 220;
  static const idleShotEveryMs = 6000;

  Isolate? _iso;
  SendPort? _isoPort;

  late final Ticker ticker;
  final paint =
      ValueNotifier<PaintState>(const PaintState(null, null, 0, 0, 0, 0));

  CameraState? camState;
  DateTime? sessionStart;
  Size? screenSize;
  bool torchOn = false, showDebug = false, shooting = false;
  DateTime? lastIdleShot;

  _Mode mode = _Mode.loading;
  _Phase phase = _Phase.searching;

  bool busy = false;
  Rect? targetBox, shownBox;
  Size? imgSize;
  double quality = 0, holdProgress = 0;
  List<double> lastVec = const [];
  int lastMs = 0;
  DateTime? lastEmbed;

  double heat = 0, heatTarget = 0, lockFlash = 0, pulseT = 0;
  int lastTickMs = 0;
  DateTime? lastBuzz;

  String? matchedId;
  double matchedSim = 0, runnerUpSim = 0;

  final shots = <Shot>[];
  final done = <String>{};
  DateTime? holdSince, cooldownUntil;
  bool hadLock = false;
  int lockGuard = 0;
  final history = <Offset>[];
  double sharpMax = 1;

  DateTime? lastFocus;
  Offset? lastFocusPt;

  bool showSkip = false, skipped = false;
  int durationMs = 0;

  List<ViewSpec> get views => widget.views;

  /// Máme vůbec nějaké reference? Když ne, jedeme sekvenčně.
  bool get hasRefs => views.any((v) => v.refs.isNotEmpty);

  /// Při sekvenčním režimu: aktuální pohled.
  ViewSpec get seqView =>
      views.firstWhere((v) => !done.contains(v.id), orElse: () => views.last);

  @override
  void initState() {
    super.initState();
    ticker = createTicker(_tick)..start();
    _init();
  }

  Future<void> _init() async {
    await embedder.load();
    final rp = ReceivePort();
    _iso = await Isolate.spawn(prepIsolate, rp.sendPort);
    _isoPort = await rp.first as SendPort;
    if (!mounted) return;
    sessionStart = DateTime.now();
    lastIdleShot = DateTime.now();
    setState(() {
      mode = _Mode.scan;
      phase = _Phase.searching;
    });
  }

  void _toggleTorch() {
    final s = camState;
    if (s == null) return;
    try {
      torchOn = !torchOn;
      s.sensorConfig.setFlashMode(torchOn ? FlashMode.always : FlashMode.none);
      setState(() {});
    } catch (_) {}
  }

  void _autoFocus(Rect box, Size img) {
    final s = camState;
    final scr = screenSize;
    if (s == null || scr == null) return;
    final pt = _mapRect(box, img, scr).center;

    final now = DateTime.now();
    if (lastFocus != null && now.difference(lastFocus!).inMilliseconds < 1600) {
      return;
    }
    if (lastFocusPt != null && (pt - lastFocusPt!).distance < scr.width * 0.14) {
      return;
    }
    lastFocus = now;
    lastFocusPt = pt;

    var iw = img.width, ih = img.height;
    if ((iw > ih) != (scr.width > scr.height)) {
      final t = iw;
      iw = ih;
      ih = t;
    }
    final pixel = PreviewSize(width: iw, height: ih);
    final fl = PreviewSize(width: scr.width, height: scr.height);
    try {
      s.when(
        onPhotoMode: (v) => v.focusOnPoint(
            flutterPosition: pt,
            pixelPreviewSize: pixel,
            flutterPreviewSize: fl),
        onPreviewMode: (v) => v.focusOnPoint(
            flutterPosition: pt,
            pixelPreviewSize: pixel,
            flutterPreviewSize: fl),
      );
    } catch (_) {}
  }

  Future<String?> _shoot({int retries = 3}) async {
    for (int i = 0; i < retries; i++) {
      if (!shooting) break;
      await Future.delayed(const Duration(milliseconds: 250));
    }
    final s = camState;
    if (s is! PhotoCameraState || shooting) return null;
    shooting = true;
    try {
      final req = await s.takePhoto();
      return req.path;
    } catch (e) {
      debugPrint('>>> foto CHYBA: $e');
      return null;
    } finally {
      shooting = false;
    }
  }

  int get _ms => sessionStart == null
      ? 0
      : DateTime.now().difference(sessionStart!).inMilliseconds;

  void _finish({required bool wasSkipped}) {
    setState(() {
      skipped = wasSkipped;
      durationMs = _ms;
      mode = _Mode.done;
    });
    HapticFeedback.heavyImpact();
    widget.onComplete
        ?.call(CaptureResult(shots, done.toList(), durationMs, wasSkipped));
  }

  void _tick(Duration d) {
    if (mode != _Mode.scan) return;

    final ms = d.inMilliseconds;
    final dt = lastTickMs == 0 ? 16 : (ms - lastTickMs).clamp(1, 100);
    lastTickMs = ms;
    pulseT += dt / 1000.0;

    shownBox = targetBox == null
        ? null
        : (shownBox == null
            ? targetBox
            : Rect.lerp(shownBox!, targetBox!, 0.30));

    heat += (heatTarget - heat) * 0.12;
    if (lockFlash > 0) lockFlash = max(0, lockFlash - dt / 450.0);

    if (!showSkip && _ms > skipAfterMs) {
      showSkip = true;
      if (mounted) setState(() {});
    }

    if (phase == _Phase.holding && holdSince != null) {
      final held = DateTime.now().difference(holdSince!).inMilliseconds;
      holdProgress = (held / holdMs).clamp(0.0, 1.0);
      if (held >= holdMs) {
        _lock();
        return;
      }
    }

    _heatBuzz();
    paint.value =
        PaintState(shownBox, imgSize, heat, holdProgress, lockFlash, pulseT);
  }

  void _heatBuzz() {
    if (phase == _Phase.locked || heat < 0.3) return;
    final interval = (620 - 530 * ((heat - 0.3) / 0.7)).clamp(90, 620).toInt();
    final now = DateTime.now();
    if (lastBuzz == null ||
        now.difference(lastBuzz!).inMilliseconds >= interval) {
      lastBuzz = now;
      if (heat > 0.8) {
        HapticFeedback.mediumImpact();
      } else {
        HapticFeedback.selectionClick();
      }
    }
  }

  Future<void> analyze(AnalysisImage img) async {
    if (busy || mode != _Mode.scan || phase == _Phase.locked) return;
    if (cooldownUntil != null && DateTime.now().isBefore(cooldownUntil!)) return;
    busy = true;
    try {
      final input = img.toInput();
      if (input != null) {
        final objects = await detector.processImage(input);
        await _evaluate(img, objects.isEmpty ? null : objects.first.boundingBox);
      }
    } catch (_) {}
    busy = false;
  }

  Future<void> _evaluate(AnalysisImage img, Rect? b) async {
    imgSize = img.size;

    if (b == null) {
      history.clear();
      holdSince = null;
      hadLock = false;
      holdProgress = 0;
      targetBox = null;
      quality = 0;
      heatTarget = 0;
      lastVec = const [];
      matchedId = null;
      matchedSim = 0;
      _setPhase(_Phase.searching);
      return;
    }

    _autoFocus(b, img.size);

    final iw = img.upright.width, ih = img.upright.height;
    final area = (b.width * b.height) / (iw * ih);
    double s;
    if (area < 0.05) {
      s = 0;
    } else if (area < 0.25) {
      s = (area - 0.05) / 0.20;
    } else if (area <= 0.70) {
      s = 1;
    } else {
      s = max(0, 1 - (area - 0.70) / 0.20);
    }

    final c = b.center;
    final diag = sqrt(iw * iw + ih * ih);
    final ctr =
        max(0.0, 1 - (c - Offset(iw / 2, ih / 2)).distance / (0.35 * diag));

    history.add(c);
    if (history.length > 5) history.removeAt(0);
    bool st = history.length >= 5;
    if (st) {
      for (final p in history) {
        if ((p - c).distance > iw * 0.06) st = false;
      }
    }

    targetBox = b;

    final now = DateTime.now();
    final canEmbed = lastEmbed == null ||
        now.difference(lastEmbed!).inMilliseconds > embedEveryMs;

    double shp = 0;

    if (canEmbed && _isoPort != null) {
      final bytes = img.rawBytes;
      if (bytes != null) {
        final t0 = DateTime.now();
        final rp = ReceivePort();
        _isoPort!.send(PrepJob(bytes, img.width, img.height, embedder.size,
            img.isBgra, b.left, b.top, b.right, b.bottom, rp.sendPort,
            img.rotDeg, img.stride));
        final res = await rp.first;
        rp.close();

        if (res != null) {
          final flat = res[0] as Float32List;
          final rawSharp = res[1] as double;
          if (rawSharp > sharpMax) sharpMax = rawSharp;
          sharpMax *= 0.995;
          shp = (rawSharp / max(sharpMax, 1)).clamp(0.0, 1.0);

          lastVec = embedder.run(flat);
          lastMs = DateTime.now().difference(t0).inMilliseconds;
          lastEmbed = now;
        }
      }
    }

    final q = st ? (0.35 * s + 0.25 * ctr + 0.40 * max(shp, quality)) : 0.0;
    quality = q;

    // průběžný snímek
    if (lastIdleShot != null &&
        now.difference(lastIdleShot!).inMilliseconds > idleShotEveryMs &&
        !shooting &&
        heat < 0.5 &&
        q > 0.4) {
      lastIdleShot = now;
      final vec = List<double>.from(lastVec);
      _shoot().then((p) {
        if (p != null) shots.add(Shot(null, p, _ms, q, 0, vec));
      });
    }

    if (_ms < warmupMs) {
      matchedId = null;
      matchedSim = 0;
      heatTarget = 0;
      _setPhase(_Phase.framing);
      return;
    }

    // ---- rozpoznání pohledu ----
    if (hasRefs) {
      String? bestId;
      double bestSim = -1, secondSim = -1;
      for (final v in views) {
        if (done.contains(v.id) || v.refs.isEmpty) continue;
        final sim = bestAgainst(lastVec, v.refs);
        if (sim < 0) continue;
        if (sim > bestSim) {
          secondSim = bestSim;
          bestSim = sim;
          bestId = v.id;
        } else if (sim > secondSim) {
          secondSim = sim;
        }
      }
      matchedId = bestId;
      matchedSim = bestSim < 0 ? 0 : bestSim;
      runnerUpSim = secondSim < 0 ? 0 : secondSim;

      final simHeat = ((matchedSim - 0.5) / 0.4).clamp(0.0, 1.0);
      heatTarget = (0.7 * simHeat + 0.3 * q).clamp(0.0, 1.0);

      final spec =
          views.firstWhere((v) => v.id == (bestId ?? views.first.id));
      final qT = hadLock ? spec.minQuality - 0.15 : spec.minQuality;
      final th = hadLock ? widget.simThreshold - 0.04 : widget.simThreshold;
      final decisive = runnerUpSim <= 0 || (matchedSim - runnerUpSim) > 0.03;

      if (q >= qT && matchedSim >= th && decisive && bestId != null) {
        hadLock = true;
        holdSince ??= DateTime.now();
        _setPhase(_Phase.holding);
      } else {
        holdSince = null;
        hadLock = false;
        holdProgress = 0;
        _setPhase(_Phase.framing);
      }
    } else {
      // ---- sekvenční režim: jen kvalita záběru ----
      final spec = seqView;
      matchedId = spec.id;
      matchedSim = 0;
      runnerUpSim = 0;
      heatTarget = q.clamp(0.0, 1.0);

      final qT = hadLock ? spec.minQuality - 0.15 : spec.minQuality;
      if (q >= qT) {
        hadLock = true;
        holdSince ??= DateTime.now();
        _setPhase(_Phase.holding);
      } else {
        holdSince = null;
        hadLock = false;
        holdProgress = 0;
        _setPhase(_Phase.framing);
      }
    }
  }

  void _setPhase(_Phase p) {
    if (phase == p) return;
    phase = p;
    if (mounted) setState(() {});
  }

  void _lock() {
    final id = matchedId;
    if (id == null || done.contains(id) || lockGuard != 0) return;
    lockGuard = 1;

    phase = _Phase.locked;
    holdSince = null;
    lockFlash = 1.0;
    cooldownUntil = DateTime.now().add(const Duration(milliseconds: cooldownMs));

    HapticFeedback.heavyImpact();
    Future.delayed(const Duration(milliseconds: 55),
        () => HapticFeedback.heavyImpact());
    Future.delayed(const Duration(milliseconds: 120),
        () => HapticFeedback.mediumImpact());

    final q = quality, sim = matchedSim, at = _ms;
    final vec = List<double>.from(lastVec);
    done.add(id);
    holdProgress = 1;
    setState(() {});

    _shoot().then((p) {
      if (p != null) shots.add(Shot(id, p, at, q, sim, vec));
      debugPrint('>>> LOCK $id @ ${(at / 1000).toStringAsFixed(1)}s '
          'q=${(q * 100).round()}% sim=${(sim * 100).round()}%');
    });

    Future.delayed(const Duration(milliseconds: 850), () {
      if (!mounted) return;
      history.clear();
      holdSince = null;
      hadLock = false;
      targetBox = null;
      shownBox = null;
      quality = 0;
      holdProgress = 0;
      heat = 0;
      heatTarget = 0;
      matchedId = null;
      matchedSim = 0;
      lastVec = const [];
      lockGuard = 0;
      if (done.length == views.length) {
        _finish(wasSkipped: false);
      } else {
        setState(() => phase = _Phase.searching);
      }
    });
  }

  ViewSpec? get _currentSpec {
    if (matchedId == null) return hasRefs ? null : seqView;
    final i = views.indexWhere((v) => v.id == matchedId);
    return i < 0 ? null : views[i];
  }

  String get hint {
    switch (phase) {
      case _Phase.searching:
        return 'Namiř na předmět';
      case _Phase.framing:
        if (!hasRefs) return seqView.instruction;
        if (heat > 0.75) return 'Ještě kousek…';
        if (heat > 0.4) return 'Přihořívá';
        return 'Otáčej předmětem';
      case _Phase.holding:
        return 'Drž…';
      case _Phase.locked:
        return 'Máme to';
    }
  }

  // ---- UI ----

  @override
  Widget build(BuildContext context) {
    screenSize = MediaQuery.of(context).size;

    if (mode == _Mode.loading) {
      return Container(
        color: Colors.black,
        child: Center(
          child: Text(embedder.status,
              style: const TextStyle(color: Colors.white54, fontSize: 14)),
        ),
      );
    }

    return Stack(
      children: [
        CameraAwesomeBuilder.awesome(
          saveConfig: SaveConfig.photo(),
          onImageForAnalysis: analyze,
          imageAnalysisConfig: AnalysisConfig(maxFramesPerSecond: 10),
          previewFit: CameraPreviewFit.cover,
          bottomActionsBuilder: (state) => const SizedBox.shrink(),
          topActionsBuilder: (state) => const SizedBox.shrink(),
          middleContentBuilder: (state) => const SizedBox.shrink(),
          previewDecoratorBuilder: (state, preview) {
            if (camState != state) {
              camState = state;
              WidgetsBinding.instance.addPostFrameCallback((_) {
                if (mounted) setState(() {});
              });
            }
            return IgnorePointer(
              child: RepaintBoundary(
                child: CustomPaint(
                  painter: ScanOverlay(paint),
                  child: const SizedBox.expand(),
                ),
              ),
            );
          },
        ),
        Positioned.fill(child: _ui()),
      ],
    );
  }

  Widget _ui() {
    if (mode == _Mode.done) return _resultView();
    return GestureDetector(
      behavior: HitTestBehavior.translucent,
      onDoubleTap: () => setState(() => showDebug = !showDebug),
      child: Stack(children: [
        _steps(),
        _torch(),
        if (showDebug) _debug(),
        _bottom(),
      ]),
    );
  }

  Widget _steps() {
    return Positioned(
      top: 62,
      left: 24,
      right: 24,
      child: IgnorePointer(
        child: Row(
          children: views.map((v) {
            final scanned = done.contains(v.id);
            final isNow = v.id == (matchedId ?? (hasRefs ? null : seqView.id));
            final c = scanned ? kOk : (isNow ? kBrand : Colors.white24);
            return Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 4),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 320),
                  curve: Curves.easeOutCubic,
                  height: scanned ? 6 : 4,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(4),
                    color: c,
                    boxShadow: scanned
                        ? [
                            BoxShadow(
                                color: c.withValues(alpha: 0.5),
                                blurRadius: 10,
                                spreadRadius: 1)
                          ]
                        : null,
                  ),
                ),
              ),
            );
          }).toList(),
        ),
      ),
    );
  }

  Widget _torch() => Positioned(
        top: 100,
        right: 24,
        child: GestureDetector(
          onTap: _toggleTorch,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            padding: const EdgeInsets.all(13),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: torchOn ? kBrand : Colors.black38,
              border: Border.all(
                  color: torchOn ? kBrand : Colors.white24, width: 1.2),
            ),
            child: Icon(
                torchOn
                    ? Icons.flashlight_on_rounded
                    : Icons.flashlight_off_rounded,
                color: torchOn ? Colors.black : Colors.white70,
                size: 22),
          ),
        ),
      );

  Widget _debug() => Positioned(
        top: 100,
        left: 24,
        right: 100,
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(
            '${hasRefs ? "AI" : "sekvence"}  '
            'sim ${(matchedSim * 100).round()}% → ${matchedId ?? "—"} '
            'Δ${((matchedSim - runnerUpSim) * 100).round()}%\n'
            'q ${(quality * 100).round()}%  ${lastMs}ms  fotek ${shots.length}\n'
            'X ${gCalX.toStringAsFixed(2)} Y ${gCalY.toStringAsFixed(2)} '
            'S ${gCalScale.toStringAsFixed(2)}',
            style: const TextStyle(
                color: Colors.white70, fontSize: 11, fontFamily: 'Courier'),
          ),
          const SizedBox(height: 8),
          Row(children: [
            _cal('◀', () => gCalX -= 0.01),
            _cal('▶', () => gCalX += 0.01),
            _cal('▲', () => gCalY -= 0.01),
            _cal('▼', () => gCalY += 0.01),
            _cal('−', () => gCalScale -= 0.02),
            _cal('+', () => gCalScale += 0.02),
          ]),
        ]),
      );

  Widget _cal(String t, VoidCallback f) => GestureDetector(
        onTap: () => setState(f),
        child: Container(
          margin: const EdgeInsets.only(right: 5),
          padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 7),
          decoration: BoxDecoration(
              color: Colors.white24, borderRadius: BorderRadius.circular(8)),
          child:
              Text(t, style: const TextStyle(color: Colors.white, fontSize: 14)),
        ),
      );

  Widget _bottom() {
    final spec = _currentSpec;
    return Positioned(
      bottom: 44,
      left: 0,
      right: 0,
      child: Column(children: [
        IgnorePointer(
          child: Column(children: [
            Text(
              spec?.label.toUpperCase() ?? '',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 15,
                fontWeight: FontWeight.w800,
                letterSpacing: 2.5,
                shadows: [Shadow(color: Colors.black87, blurRadius: 12)],
              ),
            ),
            const SizedBox(height: 10),
            Text(hint,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 21,
                  fontWeight: FontWeight.w600,
                  shadows: [Shadow(color: Colors.black87, blurRadius: 14)],
                )),
          ]),
        ),
        const SizedBox(height: 20),
        if (showSkip)
          GestureDetector(
            onTap: () => _finish(wasSkipped: true),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
              child: const Text('Nedaří se? Pokračovat bez záznamu',
                  style: TextStyle(color: Colors.white54, fontSize: 13)),
            ),
          ),
      ]),
    );
  }

  Widget _btn(String text, Color color, VoidCallback onTap) => GestureDetector(
        onTap: onTap,
        child: Container(
          padding:
              const EdgeInsets.symmetric(horizontal: 44, vertical: 19),
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(40),
            boxShadow: [
              BoxShadow(
                  color: color.withValues(alpha: 0.35),
                  blurRadius: 22,
                  spreadRadius: 1)
            ],
          ),
          child: Text(text,
              style: const TextStyle(
                  fontSize: 17,
                  color: Colors.black,
                  fontWeight: FontWeight.w800)),
        ),
      );

  Widget _resultView() {
    final locked = shots.where((s) => s.viewId != null).toList()
      ..sort((a, b) => a.atMs.compareTo(b.atMs));
    final idle = shots.where((s) => s.viewId == null).length;
    return Container(
      color: const Color(0xF00A0A0A),
      padding: const EdgeInsets.all(30),
      child: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 40),
              Icon(skipped ? Icons.info_outline : Icons.check_circle_rounded,
                  color: skipped ? kBrand : kOk, size: 54),
              const SizedBox(height: 16),
              Text(skipped ? 'Přeskočeno' : 'Hotovo',
                  style: TextStyle(
                      color: skipped ? kBrand : kOk,
                      fontSize: 34,
                      fontWeight: FontWeight.w800)),
              const SizedBox(height: 8),
              Text(
                  '${(durationMs / 1000).toStringAsFixed(1)} s   ·   '
                  '${done.length}/${views.length} pohledů   ·   '
                  '${shots.length} fotek',
                  style: const TextStyle(color: Colors.white54, fontSize: 15)),
              const SizedBox(height: 26),
              ...locked.map((l) => Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: Row(children: [
                      const Icon(Icons.check_rounded, color: kOk, size: 17),
                      const SizedBox(width: 10),
                      Text(
                          views
                              .firstWhere((v) => v.id == l.viewId,
                                  orElse: () => views.first)
                              .label,
                          style: const TextStyle(
                              color: Colors.white, fontSize: 15)),
                      const Spacer(),
                      Text('${(l.atMs / 1000).toStringAsFixed(1)}s',
                          style: const TextStyle(
                              color: Colors.white38,
                              fontSize: 13,
                              fontFamily: 'Courier')),
                    ]),
                  )),
              if (idle > 0)
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text('+ $idle průběžných snímků',
                      style:
                          const TextStyle(color: Colors.white38, fontSize: 13)),
                ),
              const SizedBox(height: 34),
              Center(
                  child: _btn('Zavřít', Colors.white,
                      () => Navigator.of(context).maybePop())),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    ticker.dispose();
    paint.dispose();
    detector.close();
    embedder.close();
    _iso?.kill(priority: Isolate.immediate);
    super.dispose();
  }
}

Rect _mapRect(Rect box, Size img, Size scr) {
  var iw = img.width, ih = img.height;
  if ((iw > ih) != (scr.width > scr.height)) {
    final t = iw;
    iw = ih;
    ih = t;
  }
  final scale = max(scr.width / iw, scr.height / ih) * gCalScale;
  final dx = (scr.width - iw * scale) / 2 + gCalX * scr.width;
  final dy = (scr.height - ih * scale) / 2 + gCalY * scr.height;
  return Rect.fromLTRB(box.left * scale + dx, box.top * scale + dy,
      box.right * scale + dx, box.bottom * scale + dy);
}

class ScanOverlay extends CustomPainter {
  final ValueNotifier<PaintState> state;
  ScanOverlay(this.state) : super(repaint: state);

  @override
  void paint(Canvas canvas, Size size) {
    final s = state.value;
    if (s.box == null || s.imgSize == null) {
      final t = (sin(s.pulse * 2.2) + 1) / 2;
      canvas.drawCircle(
        Offset(size.width / 2, size.height * 0.45),
        92.0 + t * 10,
        Paint()
          ..color = Colors.white.withValues(alpha: 0.10 + t * 0.10)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
      return;
    }

    final r = _mapRect(s.box!, s.imgSize!, size);
    final hole =
        RRect.fromRectAndRadius(r.inflate(26), const Radius.circular(30));
    canvas.saveLayer(Offset.zero & size, Paint());
    canvas.drawRect(Offset.zero & size, Paint()..color = kDim);
    canvas.drawRRect(hole, Paint()..blendMode = BlendMode.clear);
    canvas.restore();

    _frame(canvas, r, s);
  }

  void _frame(Canvas canvas, Rect r, PaintState s) {
    final heat = s.heat.clamp(0.0, 1.0);
    final locked = s.lockFlash > 0;
    final color = locked
        ? kOk
        : Color.lerp(
            Color.lerp(Colors.white70, kBrand, (heat * 1.6).clamp(0.0, 1.0))!,
            kOk,
            ((heat - 0.72) / 0.28).clamp(0.0, 1.0))!;

    final box = r.inflate(26 - s.lockFlash * 8);
    final len = min(box.width, box.height) * 0.22;
    final p = Paint()
      ..color = color
      ..strokeWidth = 3.0 + heat * 2.5 + s.lockFlash * 3
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;
    if (heat > 0.35 || locked) {
      p.maskFilter =
          MaskFilter.blur(BlurStyle.solid, 2 + heat * 5 + s.lockFlash * 10);
    }

    void corner(Offset o, double sx, double sy) {
      canvas.drawPath(
        Path()
          ..moveTo(o.dx + sx * len, o.dy)
          ..lineTo(o.dx + sx * 16, o.dy)
          ..quadraticBezierTo(o.dx, o.dy, o.dx, o.dy + sy * 16)
          ..lineTo(o.dx, o.dy + sy * len),
        p,
      );
    }

    corner(box.topLeft, 1, 1);
    corner(box.topRight, -1, 1);
    corner(box.bottomLeft, 1, -1);
    corner(box.bottomRight, -1, -1);

    final radius = min(box.width, box.height) * 0.30;
    final arc = Rect.fromCircle(center: box.center, radius: radius);

    if (s.hold > 0.005 || locked) {
      canvas.drawArc(arc, -pi / 2, 2 * pi, false,
          Paint()
            ..color = Colors.white24
            ..style = PaintingStyle.stroke
            ..strokeWidth = 5);
      canvas.drawArc(arc, -pi / 2, 2 * pi * (locked ? 1.0 : s.hold), false,
          Paint()
            ..color = color
            ..style = PaintingStyle.stroke
            ..strokeWidth = 5.5
            ..strokeCap = StrokeCap.round
            ..maskFilter = const MaskFilter.blur(BlurStyle.solid, 4));
    } else if (heat > 0.25) {
      final n = (6 + heat * 18).round();
      final dot = Paint()..color = color.withValues(alpha: 0.5 + heat * 0.5);
      for (int i = 0; i < n; i++) {
        final a = -pi / 2 + (2 * pi * i / n) + s.pulse * (0.4 + heat);
        canvas.drawCircle(box.center + Offset(cos(a), sin(a)) * radius,
            1.4 + heat * 1.8, dot);
      }
    }

    if (s.lockFlash > 0) {
      canvas.drawRRect(
        RRect.fromRectAndRadius(
            box.inflate(s.lockFlash * 22), const Radius.circular(34)),
        Paint()
          ..color = kOk.withValues(alpha: s.lockFlash * 0.55)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2 + s.lockFlash * 5
          ..maskFilter = MaskFilter.blur(BlurStyle.normal, s.lockFlash * 16),
      );
    }
  }

  @override
  bool shouldRepaint(covariant ScanOverlay old) => false;
}
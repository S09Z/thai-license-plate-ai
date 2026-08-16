# Face detection latency — Phase 10 (YuNet)

Reproduce with:

```bash
make fetch-face-model
poetry run python docs/benchmark/bench_face.py --repeats 25
```

Measured 2026-08-07 on macOS 26.5.2, arm64 (Apple silicon), CPU only.
Model: `models/face/face_detection_yunet_2023mar.onnx` (227 KB, OpenCV Zoo),
score threshold 0.6, frame 1280×720.

## Results

| Stage | Median | Min | Max | n |
|---|---|---|---|---|
| Face, flat frame | **16.6 ms** | 15.2 | 121.9 | 25 |
| Face, textured frame | **17.6 ms** | 15.1 | 24.6 | 25 |
| Plate, textured frame | 23.0 ms | 19.7 | 33.0 | 25 |
| Camera tick, concurrent | **23.0 ms** | | | |
| Camera tick, if issued serially | 40.6 ms | | | |

Cold start (first call, loads the ONNX graph): **203 ms**, paid once per process.

**Budget: `<25 ms` — met, at 17.6 ms worst-case median (0.70× of budget).**

The 121.9 ms max on the flat frame is a first-iteration outlier, not a
content effect — the textured run, which follows it with the graph already
warm, never exceeds 24.6 ms. Medians are reported for exactly this reason.

## Why YuNet rather than a Haar cascade

The choice was made on measurement, not reputation. Haar was timed first, on
the same machine, at 720p:

| Haar cascade input | Median |
|---|---|
| Blank frame | 6.1 ms |
| Textured frame | **56.0 ms** (52–78) |

Haar's cost tracks image content — a **9× swing** between a flat and a busy
frame — and 56 ms on a realistic frame blows the `<25 ms` budget on its own,
before the plate pass runs. A "Haar is fast" claim measured only on a blank
frame would have been wrong in exactly the case that matters.

YuNet was expected not to show that spread, being a fixed-input-size CNN whose
work is set by the input dimensions rather than by scene complexity. **The
flat-vs-textured pair in this benchmark exists to check that, not to assume
it**, and it holds: 16.6 ms vs 17.6 ms, a 6% difference against Haar's 9×.

## Effect on the camera tick

Phase 9's tracking loop runs at a 200 ms interval, so a 23.0 ms tick leaves
ample headroom whether or not faces are enabled. The loop issues `/detect` and
`/detect/faces` concurrently from a single captured frame, so the tick costs
the **slower** of the two stages (23.0 ms) rather than their sum (40.6 ms).
With "Show faces" unticked — the default — the face request is not issued at
all and the tick is plate-only.

These are in-process figures. The browser round trip adds HTTP and multipart
overhead on top; Phase 9 measured that at 21–37 ms warm for `/detect` alone.

## Accuracy — mostly unmeasured, with one real data point

The benchmark frames contain **no faces**, so the numbers above bound the
model's *speed only*. They say nothing about how well it finds faces.

One real check was run outside the benchmark, since no face had ever been put
through this pipeline before: a 500×624 public-domain photograph posted to
`POST /detect/faces` returned exactly one box, `(184, 43)–(302, 212)` at
confidence **0.946**, and the box was confirmed visually to land on the face.
That validates the `x, y, w, h → x1, y1, x2, y2` conversion and the score
column against live model output rather than against the unit tests' fakes.

That is a single still photograph. It is **not** evidence of accuracy under
the conditions this feature actually runs in: there is still no camera device
in this environment, so no face has been tracked in a live video stream, at an
angle, in motion, or under poor lighting. Treat detection quality as unproven.

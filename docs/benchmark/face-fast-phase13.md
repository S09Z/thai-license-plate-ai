# Fast face detection latency — Phase 13 (server-side downscale)

Reproduce with:

```bash
make fetch-face-model
poetry run python docs/benchmark/bench_face_fast.py --repeats 30
```

Measured 2026-08-07 on macOS 26.5.2, arm64 (Apple silicon), CPU only.
Model: `models/face/face_detection_yunet_2023mar.onnx`, confidence threshold
0.6, source frame 1280×720 downscaled to a **480px longest edge** (480×270),
per the `APP_FACE_FAST_MAX_SIZE` default.

## Results

| Stage | Median | Min | Max | n |
|---|---|---|---|---|
| Detect, 720p full | **18.46 ms** | 16.33 | 52.02 | 30 |
| Detect, fast | **3.22 ms** | 2.74 | 4.74 | 30 |
| **Speedup** | **5.7×** | | | |
| Fast detection alone, ceiling | **310 fps** | | | |

YuNet is input-size-bound, so cutting the input area by ~7× (921,600 → 129,600
pixels) cuts the run time by 5.7×. The measured 3.22 ms leaves the detection
stage well inside even a single 60 fps frame's 16.7 ms budget, before network
and encode are counted.

## What the number does and does not buy

- **It is a cost bound, not a frame-rate promise.** 310 fps is only the
  detector running on one image in a loop. The realtime browser loop that uses
  this path (`/detect/faces?fast=true`) must also capture a frame, JPEG-encode
  it, round-trip over HTTP and draw into the overlay — all inside one tick. The
  detector no longer being the bottleneck (3 ms near 18 ms) is what lets the
  loop *approach* the camera's native cadence instead of being pinned to the
  ~200 ms plate loop.
- **Coordinates are source-pixel, unchanged.** The downscale and every result
  (boxes and any landmark points) are rescaled back on the server, so the
  client keeps its 1:1 overlay invariant. A `?fast=true` response is
  coordinate-compatible with a plain one; only precision is coarser.
- **Precision cost is real but bounded by the loop's use.** The path exists for
  *boxes* (cheap, follow a moving face). Feature/mesh modes stay full
  resolution because their user-visible quality depends on it.

## Caveats

- **No camera device in this environment** — carried forward unchanged from
  Phases 10, 11 and 12. The 3.22 ms figure is the detector on a synthetic
  texture frame; the achieved browser frame rate is unmeasured.
- **Synthetic frame, no faces.** As before, this bounds speed only — the
  textured frame has no face in it, so accuracy at this downscale is untested.
- **The 52.02 ms max is a scheduling outlier** on a laptop, not a regression;
  the median is the number that matters and it is consistent.
# Face mesh latency — Phase 12 (Delaunay triangulation of the 68 points)

Reproduce with:

```bash
make fetch-face-landmark-model
poetry run python docs/benchmark/bench_face_landmarks.py --repeats 40
poetry run python docs/benchmark/bench_face_landmarks.py --repeats 40 --image face.jpg
```

Measured 2026-08-07 on macOS 26.5.2, arm64 (Apple silicon), CPU only.
Model: `models/face/lbfmodel.yaml` (54 MB), frame 1280×720. No new model and no
new dependency: the triangulation is `cv2.Subdiv2D`, already in the pinned
OpenCV 4.10.0.

## Results

| Stage | Median | Min | Max | n |
|---|---|---|---|---|
| Fit, 1 face | 1.5 ms | 1.2 | 2.0 | 40 |
| **Fit + mesh, 1 face** | **1.9 ms** | 1.5 | 2.4 | 40 |
| **Mesh overhead** | **0.4 ms** | | | |
| Fit + mesh, real photograph (500×624) | 3.8 ms | 2.9 | 6.8 | 40 |
| Detect, same photograph | 16.3 ms | 9.0 | 57.3 | 40 |
| Detect + mesh, serial | **20.1 ms** | | | |

**Budget: `<25 ms` — met, at 20.1 ms for detect + fit + mesh serially.**

The mesh is the cheapest stage in the pipeline. Triangulating 68 points is
0.4 ms on top of a fit that was already 1.5 ms, so opting into the mesh costs
about a quarter again on a stage that was never the bottleneck. Detection still
dominates, as it has since Phase 10.

## Where the time actually goes

The first implementation measured **1.75 ms** for the triangulation alone —
4× the final number — and the breakdown is worth recording because the cost was
not where it looked:

| Step | Median |
|---|---|
| `cv2.Subdiv2D` construction | 0.001 ms |
| 68 × `insert()` | 0.036 ms |
| `getTriangleList()` | 0.002 ms |
| **Python filtering loop** | **~1.7 ms** |

OpenCV's share is 0.04 ms. Everything else was the loop that maps triangle
corners back to point indices: `getTriangleList()` returns a numpy array, so
iterating it directly yields **`numpy.float32` scalars**, and hashing those as
dict keys costs roughly 7× what native floats do. A single `.tolist()` before
the loop took the whole function from **1.75 ms to 0.23 ms** with byte-identical
output (113 triangles either way, verified equal).

That call is load-bearing rather than cosmetic, and `face/landmarks.py` says so
at the line, because it looks exactly like the kind of thing a later cleanup
would remove.

## Triangulation shape, on the real photograph

| | |
|---|---|
| Points in | 68 |
| Triangles out | **113** |
| Index range | 0–67, all valid |
| Distinct indices per triangle | 3, always |
| Points used by at least one triangle | 68 of 68 |
| Duplicate coordinates | none |
| Response size, one meshed face | **~2.8 KB** |

Triangles are sent as **index triples into a flat 68-point array**, not as
coordinate triples. Coordinates would be roughly 4 KB per face for the triangle
list alone; indices make it ~1.5 KB of that 2.8 KB total, and they let a client
check the topology against points it already holds.

## Two hazards this measurement exposed

1. **Fitted points fall outside the detection box.** One of 68 did on this
   photograph. `Subdiv2D` raises on any insert outside its rectangle, so the
   rectangle is derived from the points' own min/max with padding — never from
   the face box that seeded them.
2. **`getTriangleList()` returns a bare `()`**, not an empty array, when nothing
   triangulates. Collinear or coincident fits hit this, and the unguarded
   `.tolist()` above raised `AttributeError` until the result was normalised
   through `np.asarray(...).reshape(-1, 6)`. Both degenerate cases now return
   an empty triangle list rather than a 500.

## Caveats

- **The mesh stops at the eyebrow line.** iBUG-68 has no forehead or scalp
  points, so "whole face" means jaw-to-eyebrow. The convex hull of the 68 is the
  boundary; the forehead is outside it and is not covered.
- **Still no camera device in this environment.** Every number here comes from
  one frontal, well-lit portrait and synthetic frames. Nothing has been measured
  on a live stream, at an angle, in motion, or in poor light — carried forward
  unchanged from Phases 10 and 11.
- **The 57.3 ms detection max** is a scheduling outlier on a laptop, not a
  regression; the median of 16.3 ms is consistent with Phase 10's 17.6 ms.
- Triangulation cost is per face and linear, like the fit it follows.

# Facial landmark latency — Phase 11 (OpenCV Facemark LBF)

Reproduce with:

```bash
make fetch-face-landmark-model
poetry run python docs/benchmark/bench_face_landmarks.py --repeats 25
poetry run python docs/benchmark/bench_face_landmarks.py --repeats 25 --image face.jpg
```

Measured 2026-08-07 on macOS 26.5.2, arm64 (Apple silicon), CPU only.
Model: `models/face/lbfmodel.yaml` (54 MB, OpenCV GSOC2017), frame 1280×720.

## Results

| Stage | Median | Min | Max | n |
|---|---|---|---|---|
| Fit, 1 face | **1.1 ms** | 1.0 | 1.9 | 25 |
| Fit, 3 faces | 3.3 ms | 3.1 | 11.1 | 25 |
| Per extra face | 1.1 ms | | | |
| Fit, real photograph (500×624) | **1.0 ms** | 1.0 | 1.1 | 25 |
| Detect, same photograph | 6.6 ms | 6.0 | 8.1 | 25 |
| Detect + fit, serial | 7.6 ms | | | |

Cold start (first call, loads the 54 MB model): **495 ms**, paid once per
process — the largest cold start in the project, and the reason the model loads
lazily on first fit rather than at startup.

**Budget: `<25 ms` — met, at 1.1 ms per face (0.04× of budget).**

Unlike detection, this is **per-face** work: cost scales linearly at 1.1 ms per
additional face, so the budget question is how many faces a tick can carry, not
whether one fit fits. At 1.1 ms a face, a 200 ms camera tick would need roughly
**17 faces** before the landmark stage alone reached the 25 ms detection budget,
and ~180 before it reached the tick interval. In practice YuNet's detection pass
dominates long before that.

The 11.1 ms max on the three-face run is a first-iteration outlier; the median
of 3.3 ms is consistent with 3 × 1.1 ms.

## Effect on the camera tick

Landmarks ride along on the **existing** face request as `?landmarks=true`, so
the *Facial features* mode costs one extra ~1 ms of server work, not a third
round trip. The tick still issues `/detect` and `/detect/faces` concurrently and
costs the slower of the two.

Measured on the real photograph, the whole face path — detect then fit — is
7.6 ms serial. These are in-process figures; the browser round trip adds HTTP
and multipart overhead on top (Phase 9 measured 21–37 ms warm for `/detect`).

## Accuracy — one real data point, and it is a good one

The synthetic part of the benchmark fits fabricated boxes on a noise frame. That
bounds the **cost** of a fit (the regression does the same work wherever it is
pointed) and says nothing about whether the points land on features.

So the fit was run on a real photograph — the same 500×624 public-domain
portrait Phase 10 used — and the result rendered and inspected:

| Group | Points | x range | y range |
|---|---|---|---|
| right_eyebrow | 5 | 197–241 | 91–105 |
| left_eyebrow | 5 | 255–293 | 89–103 |
| right_eye | 6 | 212–232 | 104–109 |
| left_eye | 6 | 261–281 | 104–109 |
| nose | 9 | 235–261 | 104–147 |
| mouth | 20 | 218–274 | 154–178 |

Every group landed on its feature under visual inspection: eyebrow polylines
follow the eyebrows, the eye loops outline the eyes, the nose renders as a
bridge plus a nostril line, and the mouth renders as an outer and an inner lip
loop. The numbers above corroborate it independently — eyebrows sit above eyes,
the mouth is lowest, and every `right_*` group has smaller x than its `left_*`
counterpart, which is the iBUG-68 subject-relative convention holding on real
model output rather than on the unit tests' fakes.

**What this is not.** One frontal, well-lit, high-quality portrait. There is
still no camera device in this environment, so no face has been tracked in a
live stream, at an angle, in motion, or under poor lighting — and LBF is an
older, cheaper model than the alternatives, so degradation off-frontal is the
expected failure mode rather than a surprise. Treat landmark accuracy as
demonstrated for the easy case and unproven for the hard one.

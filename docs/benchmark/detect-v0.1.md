# Benchmark: detection latency (detector v0.1)

Reproduce with:

```bash
poetry run python docs/benchmark/bench_detect.py
```

## What was measured

`PlateDetector.detect()` — the YOLOv8n inference call `POST /detect` and `POST /recognize` both
run per uploaded image — against the real trained weights recorded in
`docs/experiments/detector-v0.1.md`. Input is the same synthetic 640×480 scene
`bench_recognize.py` uses (one bordered plate rendered into a flat background), not a
photograph.

## Results

Machine: macOS 26.5.2, arm64 (Apple M3, CPU-only). `n=15` warm iterations.

| Stage | Median | Budget (`CLAUDE.md`) | |
|---|---|---|---|
| Detection | 25 ms (min 23, max 27) | <25 ms | at the budget line |

Cold start (first call, loads detector weights): ~1.8 s.

Found: 0 boxes. The rendered scene is synthetic text on a flat background, out of distribution
for a detector trained on real photographs — expected, not a defect. `bench_detect.py` reports
latency regardless of whether the known box is actually recovered.

## Conclusion

Detection sits right at the 25 ms budget line, not comfortably under it. On a `yolov8n` CPU
inference this is close enough that run-to-run scheduling noise decides whether an individual
run reads "within budget" or "slightly over" — the min/max spread (23–27 ms) already straddles
the line. This is a real, disclosed result, not a defect to paper over: unlike OCR (~411 ms,
~10× over budget, see `recognize-phase6.md`), detection is close enough that GPU inference or a
smaller/quantized export would likely be sufficient if the budget needs to be met with margin,
rather than requiring an architectural change.

## Caveats

- **The input is rendered text, not a photograph**, same caveat as every other benchmark in this
  set — this measures latency, not detection accuracy on real plates.
- Latency was measured on the trained v0.1 weights (18.4 MB, `yolov8n` base). A different model
  size or export target would change this number.
- Single-image inference only. Batched inference (multiple plates in one frame, or multiple
  frames) was not measured.

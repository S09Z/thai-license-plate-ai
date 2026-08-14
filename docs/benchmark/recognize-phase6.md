# Benchmark: full recognition pipeline (Phase 6)

Reproduce with:

```bash
poetry run python docs/benchmark/bench_recognize.py
```

## What was measured

The stages `POST /recognize` runs per detected plate: perspective correction → OCR →
post-processing → RAG validation. Input is a synthetic 640×480 scene with one bordered
two-line Thai plate rendered into it (Thonburi), plate number above province.

**Detection is not measured.** No plate weights have ever been trained, so
`models/detector/best.pt` does not exist and the benchmark substitutes a fixed known box.
Every total below is therefore a **lower bound** on the real end-to-end figure, not the
figure itself.

## Results

Machine: macOS 26.5.2, arm64. Crop size `(256, 128)`. `n=15` warm iterations.

| Stage | Median | Budget (`CLAUDE.md`) | |
|---|---|---|---|
| Detection | **unmeasured** | <25 ms | no trained weights exist |
| Perspective | 0.5 ms | — | not budgeted separately |
| OCR | 411.4 ms | <40 ms | **~10× over** |
| Post-processing + RAG | 0.6 ms | <15 ms (RAG) | met |
| **Total (lower bound)** | **412.5 ms** | **<100 ms** | **4.1× over** |

Cold start (first recognition, loads OCR weights): ~5.1 s.

A second run of the same script measured OCR at 394.9 ms and a total of 396.0 ms. Run-to-run
variance of roughly ±20 ms on an unloaded machine does not change the conclusion.

## Accuracy

| Field | Raw OCR | Final | |
|---|---|---|---|
| Plate number | `'กข 1234'` | `'กข 1234'`, well formed | MATCH |
| Province | `'ชลบูรดี 9'` | `'ชลบุรี'` @ 0.667 | MATCH |

The province result is the more interesting one. The recognizer damaged `ชลบุรี` into
`'ชลบูรดี 9'` — it dropped a mark, hallucinated a trailing `ดี`, and picked up a stray `9`.
Phase 4's deterministic matcher correctly refuses that string. Phase 5's fuzzy corrector
recovered it, on a damage pattern it was never tuned against (Phase 5 was tuned on `ชลบรดี`,
from the Phase 3 crop). This is the first evidence the RAG stage does useful work on input it
did not see during development.

The 0.667 score sits comfortably above `MIN_SCORE = 0.6` but is not a wide margin. One more
character of damage would have pushed it into abstention.

## Conclusion

**The performance budget is not met and is not close.** OCR owns effectively all of the
runtime: perspective, post-processing and RAG together account for 1.1 ms, about 0.3% of the
total. Adding a trained detector (<25 ms budget) would not change the shape of the problem.

Untried levers, largest first:

1. **Skip PaddleOCR's text-detection pass.** A rectified crop has known line positions, so the
   detection sub-model is doing work the pipeline has already done. Likely the single biggest win.
2. ONNX / OpenVINO export, or quantization of the recognizer.
3. GPU inference, which moves the problem rather than solving it for CPU deployment.

Optimization is deliberately out of scope for Phase 6; this document is the baseline to beat.

## Caveats

- **The input is rendered text, not a photograph.** Every accuracy claim here is provisional
  until a real Thai plate dataset exists.
- The `(256, 128)` crop size remains a placeholder ratio, not a measured one. It is now
  configurable via `APP_PLATE_CROP_WIDTH` / `APP_PLATE_CROP_HEIGHT` so it can be calibrated
  when real photographs arrive.
- One plate per scene. A scene with N plates costs roughly N × the OCR figure, since each
  detection is recognized independently.

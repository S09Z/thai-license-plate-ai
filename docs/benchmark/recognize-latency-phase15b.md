# Recognition latency on real plates — Phase 15b

First **latency** measurement of the full `/recognize` pipeline against real
Thai plate photographs. Earlier latency benchmarks (`bench_recognize.py`,
`bench_ocr.py`) render synthetic text and use a fixed box, so they never
exercise the real detector; this runs the shipped pipeline over the same 22
hand-verified images as the accuracy benchmark. Reproduce with:

```
make bench-recognize-latency
```

Machine: macOS 26.5.2, arm64 (Apple Silicon), CPU only. PaddleOCR 3.7.0 /
paddlepaddle 3.3.1, `lang="th"` → `th_PP-OCRv5_mobile_rec`. Numbers are medians
over the eval set; expect ±10% run to run on a warm CPU.

## Per-stage latency (shipped config, `text_det_limit=max/192`)

| Stage | Median (ms) |
|---|---|
| detect (YOLO) | 25.2 |
| perspective | 0.6 |
| **OCR** | **260.4** |
| postprocess + RAG | 0.0 |
| **TOTAL** | **286.2** |

**2.9× over the `<100 ms` budget.** OCR is ~91% of it; every other stage is
already within budget. So Phase 15b is entirely an OCR-latency question.

## The finding that reframed the phase

`docs/benchmark/ocr-phase3.md` proposed the largest win as *"skipping text
detection entirely — the rectified crop has known line positions, so only the
recognition head is strictly needed."* Measured on real plates, that is false:
**PaddleOCR's learned text detector is doing accuracy-critical work.**

Decomposing the OCR stage (recognition head alone vs the rest):

| OCR sub-stage | Median (ms) | Share |
|---|---|---|
| recognition head (`TextRecognition`) | 54.5 | ~21% |
| text detection (DBNet) | 205.9 | ~79% |

Detection dominates — but replacing it with hand-rolled line segmentation
regresses accuracy badly, because the detector localizes low-contrast, colored
province text and separates rows in ways a projection profile cannot:

| OCR approach | exact | CER | province | OCR ms |
|---|---|---|---|---|
| **full det + rec (baseline)** | **77.3%** | **0.063** | **95.5%** | ~389 |
| rec-only + Otsu projection split | 54.5% | 0.332 | 54.5% | ~133 |
| rec-only + fixed geometric bands | 59.1% | 0.109 | 77.3% | ~112 |

Province accuracy — which Phase 15a worked to raise to 95.5% — collapses under
both segmenters (Otsu misses colored province glyphs; geometric bands clip
variable layouts). Skipping detection is therefore **rejected**: it trades away
15a's accuracy for latency the budget still would not meet.

## The safe win that shipped: cap the detector's input size

The rectified crop is only 256 px wide, yet PaddleOCR still resizes it before
running DBNet. Capping the detector's long edge below the crop width downscales
its input and cuts detection cost with no change to the recognized text. Swept
on the real crops (accuracy is deterministic per config):

| `text_det_limit` | exact | CER | province | OCR ms |
|---|---|---|---|---|
| max/256 (= default, no-op) | 77.3% | 0.063 | 95.5% | ~389 |
| max/224 | 81.8% | 0.056 | 95.5% | ~348 |
| **max/192 (shipped)** | **77.3%** | **0.063** | **95.5%** | **~280** |
| max/176 | 77.3% | 0.063 | 95.5% | ~295 |
| max/160 | 86.4% | 0.105 | **90.9%** | ~228 |

`max/192` is the shipped default: it keeps **all three accuracy metrics
identical to the baseline** while cutting OCR ~28% (~389 → ~280 ms per crop).
`max/224` happens to nudge exact-match up on this set but wins less latency;
`max/160` is faster still but **regresses province accuracy** (95.5 → 90.9%), so
it fails the zero-accuracy-loss bar. The knob is `Settings.ocr_det_limit_side_len`
/ `ocr_det_limit_type` — tunable per deployment without code changes.

## What it would take to actually meet `<100 ms`

Even with the safe knob, OCR is ~260 ms and the pipeline ~286 ms — still ~2.9×
over budget, and the remaining cost is the DBNet detector, which is inherently
expensive on CPU. Closing the gap needs a **runtime or model change**, not a
config tweak:

- **ONNX Runtime / OpenVINO** export of the det + rec models (same weights → no
  accuracy change), which can materially speed CPU inference. Cost: two new
  dependencies (`onnxruntime`, `paddle2onnx`), provenance verification, and
  export work; the speedup is unproven on this stack.
- **GPU execution**, unavailable here (no CUDA; Apple Silicon).
- **A lighter detector** or **int8 quantization**, guided by this benchmark.

These are deferred as a scoped future slice; Phase 15b delivers the honest
measurement, the accuracy-preserving knob, and this decision record.

## Caveats — read before quoting numbers

1. **CPU- and machine-specific.** Apple Silicon, CPU only; absolute ms move with
   hardware and warm state. The *ratios* (detection ≫ recognition; OCR ≫ rest)
   are the durable finding.
2. **Eval set is small (22) and dataset-specific** — all from one watermarked,
   mostly-Bangkok Roboflow set (same caveat as 15a). "Identical accuracy at
   max/192" is exact on this set but not a generalization guarantee; re-run the
   sweep before lowering the cap for a new camera population.
3. **Skip-detection numbers are from throwaway experiments**, not shipped code;
   they exist to justify *not* taking that path.

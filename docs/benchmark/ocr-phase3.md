# Benchmark — OCR stage (Phase 3)

Reproduce with:

```bash
poetry run python docs/benchmark/bench_ocr.py
```

## Result — 2026-07-31

| | |
|---|---|
| Machine | macOS 26.5.2, arm64 (Apple Silicon), CPU only |
| Engine | PaddleOCR 3.7.0 / paddlepaddle 3.3.1, `lang="th"` → `th_PP-OCRv5_mobile_rec` |
| Input | Synthetic 256×128 two-line Thai plate, Thonburi font |
| Git commit | `3a899e6` |

| Metric | Value | Budget (`CLAUDE.md`) | Verdict |
|---|---|---|---|
| `plate_text` | `กข 1234` @ 1.000 | exact | ✅ match |
| province candidate | `ชลบรดี` | `ชลบุรี` | ❌ mismatch |
| Warm median (n=15) | **394 ms** | <40 ms | ❌ **~10× over** |
| Warm min / max | 385 / 444 ms | — | — |
| Cold start | 4729 ms | — | one-off weight load |

Raw fragments: `[('1234', 1.0), ('กข', 1.0), ('ชลบรดี', 0.902)]` — note the engine
returns one box per *fragment*, so `กข` and `1234` arrive separately and are rejoined by
`group_into_rows`.

## Configuration finding

Disabling the preprocessing submodels a pre-rectified crop does not need
(`use_doc_orientation_classify`, `use_doc_unwarping`, `use_textline_orientation`) roughly
**halved latency and improved accuracy**. Measured before shipping:

| | default pipeline | submodels disabled (shipped) |
|---|---|---|
| `plate_text` | `กข 1234` @ 0.995 | `กข 1234` @ 1.000 |
| province | `ชลบร 9` | `ชลบรดี` |
| warm median | 915 ms | 468 ms |
| at 3× upscale | collapses to `VEZL` | still correct |

The orientation classifier actively *corrupted* upscaled crops, so this is a win on both axes.

## Interpretation

**The <40 ms OCR budget is not met, and nothing here should be read as meeting it.** At 394 ms
the OCR stage alone exceeds the whole-pipeline `<100 ms` budget by ~4×.

Untried levers, deferred to Phase 6 when the full pipeline is assembled:
- ONNX / OpenVINO export, or quantization of the recognizer
- GPU execution
- **Skipping text detection entirely** — a rectified crop from Phase 2 has known line positions,
  so only the recognition head is strictly needed. Likely the largest single win.

## Caveats — read before quoting these numbers

1. **The input is rendered text, not a photograph.** These numbers bound engine speed. They say
   almost nothing about accuracy on real plates: no motion blur, glare, dirt, embossing, tilt,
   or the actual Thai plate typeface.
2. **Thai vowel and tone marks are dropped** — `ชลบุรี` → `ชลบรดี`. The consonant skeleton
   survives, which is why Phase 5 (RAG) resolves provinces by fuzzy-matching against the
   77-province list rather than trusting OCR output directly.
3. **Font choice silently changes results.** Two earlier benchmark runs were invalid: one used
   NotoSansBhaiksuki (tofu boxes, not Thai) and one used Ayuthaya, whose glyphs the recognizer
   read as Latin (`กข` → `VEZT`). The script hardcodes Thonburi and asserts the render has ink.
4. **Never pass a `*_model_name` kwarg alongside `lang`.** Doing so makes PaddleOCR silently
   ignore `lang` and fall back to a non-Thai recognizer — it turned `กข` into `∩U` mid-benchmark.
5. An earlier figure of **589 ms** appears in the Phase 3 commit message and PR description. That
   run used n=5 immediately after a heavy comparison benchmark and was inflated by CPU
   contention; **394 ms (n=15) supersedes it**. The conclusion is unchanged.

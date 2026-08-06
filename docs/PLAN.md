# Plan: Pipeline Roadmap + Detection Slice

> Exported from the planning session on 2026-07-30. Living document — update as phases land.

## Context
The first slice is **complete and verified**: a runnable FastAPI app with `/health`, structured
logging, pydantic settings, and a green test/lint/type-check suite. That established the project
conventions from `CLAUDE.md`. This document is (A) a phased roadmap for the whole pipeline and
(B) the next slice (Detection) detailed to code-ready depth.

Guiding constraints from `CLAUDE.md`: Simplicity First, surgical changes, type hints + Google
docstrings, Ruff/Black/MyPy/Pytest, structured logging, **validate uploaded images / never execute
OCR output**, per-stage latency budget (Detection <25ms, OCR <40ms, RAG <15ms, total <100ms), and
model/experiment tracking. Python target stays **3.12** (installed venv).

---

## Part A — Full Pipeline Roadmap (high level)

Each phase is one shippable slice: endpoint/module + tests + docs, green on all four tools.
Reuse the established patterns: `create_app()` factory (`app/main.py`), `APIRouter` per feature
(`app/api/health.py`), pydantic schemas (`app/schemas/`), `Settings`/`get_settings()`
(`app/core/config.py`), structured logging (`app/core/logging.py`), `client` fixture
(`tests/conftest.py`).

| Phase | Stage | Key modules | New deps | Acceptance / budget |
|------|-------|-------------|----------|---------------------|
| 0 ✅ | API skeleton + health | `app/` | fastapi, pydantic | done, suite green |
| 1 ✅ | Detection | `detector/`, `app/api/detection.py` | ultralytics(+torch), opencv, numpy, pillow | done — `POST /detect` returns boxes; uploads validated; latency unmeasured (no trained weights yet) |
| 2 ✅ | Perspective correction | `detector/pipelines/perspective.py` | (opencv) | done — box → deskewed crop; synthetic-warp tested; 0.32ms median |
| 3 ✅ | OCR | `ocr/`, `app/api/ocr.py` | paddleocr, paddlepaddle | done — `POST /ocr` returns plate text + province candidates; **394ms vs <40ms budget (see Part D)** |
| 4 ✅ | Post-processing | `postprocess/` | — | done — plate normalized, province mapped deterministically; 14 pure-function tests |
| 5 ✅ | RAG validation | `rag/` | ~~chromadb, sentence-transformers~~ **none** (see Part F) | done — `ชลบรดี` → `ชลบุรี`; 0 mis-attributions over 231 degraded candidates; **0.44ms vs <15ms budget — met** |
| 6 ✅ | Full `/recognize` | `app/api/recognize.py`, `app/services/recognize_service.py` | — | done — chains 1→5, one JSON response per plate; **412ms lower bound vs <100ms budget — 4.1× over (see Part G)** |
| 7 ✅ | Web UI | `web/static/`, `app/api/web.py` | ~~jinja2/gradio~~ **none** (see Part H) | done — upload → boxed canvas + results table; abstention shown as "Unknown" |
| 8 ✅ | UI mode switch | `web/static/` | — | done — Upload / Live camera toggle, one panel at a time (see Part I.1) |
| 9 ✅ | Realtime tracking | `web/static/` | — | done — `/detect` at 200 ms strokes boxes over the live video; `/recognize` stays at 1.5 s for the table (see Part I.2) |

Cross-cutting (fold in as stages land, not upfront): model registry under `models/` with
version/dataset/metrics/git-commit (CLAUDE.md Model Rules) — **still deferred**, though detector
v0.1 weights now exist at `models/detector/best.pt` (gitignored); experiment log under
`docs/experiments/` — **✅ landed** (`detector-v0.1.md`); latency benchmark under
`docs/benchmark/` — **✅ landed in Phase 3** (`bench_ocr.py` + `ocr-phase3.md`), extended in Phase 5 (`bench_rag.py` + `rag-phase5.md`), the pattern to
follow when later stages need numbers.

---

## Part B — Detection Slice (✅ shipped 2026-07-31)

Built test-first. Deviations from the plan as written, all deliberate:
- **No `detector/pipelines/__init__.py` or `detector/models/__init__.py`** — they would be empty
  packages with no members yet (Simplicity First). Add them when Phase 2 needs them.
- **Added a `503` mapping** for missing detector weights (`FileNotFoundError`). Not in the original
  plan, but it is the state the app is in today, and 500 would have been wrong.
- **Optional integration test skipped** — there are no weights to point it at yet.
- `_load_yolo()` is a module-level function so tests patch it instead of touching private state;
  this is what keeps the suite weight-free.


**Goal:** `POST /detect` accepts an image upload, validates it, runs YOLO plate detection, and
returns bounding boxes as JSON. Model weights are **configurable** and inference is **mockable** so
tests stay fast and need no downloaded weights or trained plate model (none exists yet).

### B1. Dependencies — `pyproject.toml`
Add runtime: `ultralytics` (pulls `torch`/`torchvision` — gigabytes, expected here per CLAUDE.md
deferral), `opencv-python-headless`, `numpy`, `pillow`. No new dev deps.

### B2. Config — `app/core/config.py` (extend existing `Settings`)
Add fields: `detector_model_path: str` (default e.g. `models/detector/best.pt`),
`detector_conf_threshold: float = 0.25`, `max_upload_bytes: int = 10 * 1024 * 1024`,
`allowed_image_types: tuple[str, ...] = ("image/jpeg", "image/png")`.

### B3. Image validation util — `app/utils/image.py` (new)
`load_image(data: bytes, content_type: str, *, max_bytes, allowed_types) -> np.ndarray`.
Enforces content-type allowlist, size cap, and that bytes actually **decode** to an image
(reject on failure with a domain error). Satisfies CLAUDE.md "Validate uploaded images."
Unit-tested directly.

### B4. Detector package — `detector/`
- `detector/__init__.py`, `detector/pipelines/__init__.py`, `detector/models/__init__.py`
- `detector/detector.py` — `PlateDetector`:
  - `__init__(self, model_path: str, conf_threshold: float)` — lazy-loads the Ultralytics `YOLO`
    model on first `detect()` (keeps import cheap, model absent = clear error only when used).
  - `detect(self, image: np.ndarray) -> list[Detection]` — returns dataclass/pydantic boxes.
  - Keep OpenCV/torch confined here; the API layer never imports ultralytics directly.

### B5. Schemas — `app/schemas/detection.py` (new)
`BoundingBox` (`x1,y1,x2,y2: int`, `confidence: float`) and
`DetectionResponse` (`count: int`, `boxes: list[BoundingBox]`).

### B6. Service — `app/services/detection_service.py` (new)
`detect_plates(data, content_type) -> DetectionResponse`: calls `load_image` (B3) then a
process-cached `PlateDetector` (built from `get_settings()`), maps model output → schema.

### B7. Route — `app/api/detection.py` (new)
`POST /detect` with `UploadFile`; reads bytes, delegates to the service, returns
`DetectionResponse`. Maps validation errors → `HTTPException 400/415/413`. Register the router in
`create_app()` (`app/main.py`) alongside the health router.

### B8. Tests — `tests/`
- `tests/unit/test_image_util.py` — good decode; rejects bad content-type, oversize, undecodable bytes.
- `tests/unit/test_detection_api.py` — **mock `PlateDetector.detect`**; assert `POST /detect` with a
  tiny generated PNG (pillow) → 200 + expected boxes; bad/oversize upload → 4xx.
- Optional `tests/integration/test_detection_model.py` — real weights, `pytest.mark.skipif` when
  `detector_model_path` missing.

---

## Verification (Detection slice)
1. `poetry install` — resolves the new ML deps (slow/large; expected).
2. `poetry run pytest -q` — unit tests green **without** any model file (detector mocked).
3. `poetry run ruff check .` / `poetry run black --check .` / `poetry run mypy app detector` — clean.
4. Manual: `poetry run uvicorn app.main:app` then `curl -F "file=@sample.png" localhost:8000/detect`
   → JSON boxes (needs a real weights file); bad upload → 415/400.

---

## Part C — Perspective Correction Slice (✅ shipped 2026-07-31)

**Goal:** a detection box is axis-aligned, so a plate shot at an angle stays skewed inside it.
Recover the plate's quadrilateral and warp it flat, ready for OCR.

**Module:** `detector/pipelines/perspective.py` — library only, no endpoint. Phase 3 (OCR) is the
first consumer; Phase 6 chains it into `/recognize`.

| Function | Role |
|---|---|
| `order_corners(points)` | 4 points → TL, TR, BR, BL via coordinate sum/difference. Pure. |
| `find_plate_quad(crop)` | Canny → contours → `approxPolyDP`; largest 4-sided contour covering ≥20% of the crop. `None` when absent. |
| `warp_plate(image, quad, size)` | `getPerspectiveTransform` + `warpPerspective` onto a fixed rectangle. |
| `correct_perspective(image, box, ...)` | Orchestrates: pad 4% → clamp to image → crop → find quad → warp. |

### Decisions
- **Fallback, not failure.** When no quadrilateral is found, the padded crop is returned resized.
  A skewed crop still reaches OCR; refusing to produce one would drop the plate entirely.
- **No new `Settings` fields.** `output_size` and `padding` are keyword arguments with defaults.
  Nothing in the API surface consumes them yet, so config plumbing would be speculative — add it
  in Phase 6 when `/recognize` needs to tune them.
- **`output_size` default `(256, 128)` is a placeholder**, not a measured Thai-plate ratio. Thai
  plates carry two lines (number above, province below); calibrate against real data in Phase 3.

### Measured
- 8 unit tests, suite total 22 green; ruff/black/mypy clean.
- Synthetic-warp recovery: MAE **8.5** vs true plate, against **68.9** for an uncorrected crop.
- Latency on a 720p frame with a 435×176 box: median **0.32 ms**, p95 0.52 ms — negligible
  against the 100ms total budget.

---

## Part D — OCR Slice (✅ shipped 2026-07-31)

**Goal:** a rectified plate crop → plate number text + province candidates, exposed as `POST /ocr`.

**Modules:** `ocr/reading.py` (engine-free data + row logic), `ocr/recognizer.py` (PaddleOCR
adapter), `app/schemas/ocr.py`, `app/services/ocr_service.py`, `app/api/ocr.py`.

Province *resolution* is deliberately out of scope: `/ocr` reports candidates verbatim and
Phase 5 (RAG) matches them against the province list. Recognized text is never interpreted or
executed, per the security rule in `CLAUDE.md`.

### Decisions
- **`ocr/reading.py` holds no engine import.** All row/line logic is pure data, so the whole
  suite runs with no model weights and no network. `_load_paddleocr` is a module-level function
  purely so tests can patch it.
- **Fragments, not lines.** PaddleOCR returns one box per *fragment*, not per visual line — it
  split `กข 1234` into two boxes. `TextLine` therefore carries `top`/`bottom`/`left`, and
  `group_into_rows` reassembles rows by vertical overlap (≥50% of the shorter box), ordering
  fragments left-to-right within a row. Discovered by running the real engine, not by design.
- **Topmost row = plate number**, every row below = province candidate.
- **Row confidence is the weakest fragment's**, not the mean — a joined row is only as
  trustworthy as its least certain part.
- **Preprocessing submodels disabled** (`use_doc_orientation_classify`, `use_doc_unwarping`,
  `use_textline_orientation`). Crops arriving here are already detected and rectified.
- **No `*_model_name` override.** Passing one makes PaddleOCR ignore `lang` and silently fall
  back to a non-Thai recognizer — it turned `กข` into `∩U` during benchmarking.
- **`lines` stays at fragment granularity in the response**, so a client can see what the engine
  actually saw before rows were joined.

### Measured (synthetic 256×128 Thai plate, Thonburi, CPU / Apple Silicon)
| | default pipeline | submodels disabled (shipped) |
|---|---|---|
| `plate_text` | `กข 1234` @ 0.995 | `กข 1234` @ 1.000 |
| province | `ชลบร 9` | `ชลบรดี` |
| warm median | 915 ms | **394 ms** (n=15, `docs/benchmark/`) |
| at 3× upscale | collapses to `VEZL` | still correct |

- 14 unit tests for this slice, suite total **42 green**; ruff/black/mypy clean.
- Cold start (first recognition, weights load): ~4.7 s.
- Reproducible: `poetry run python docs/benchmark/bench_ocr.py`; result recorded in
  `docs/benchmark/ocr-phase3.md`. An earlier figure of 589 ms (n=5, measured under load) appears
  in the Phase 3 commit message and PR body and is superseded by 394 ms (n=15).

### ⚠️ Open issues carried forward
1. **Latency misses the budget by ~10×** — 394 ms measured against `<40 ms` in `CLAUDE.md`.
   The OCR stage alone exceeds the whole-pipeline `<100 ms` budget by ~4×.
   Untried levers: ONNX/OpenVINO export, quantization, batching, GPU, or skipping text
   *detection* entirely (a rectified crop has known line positions — likely the largest single
   win). **The performance budget is currently aspirational, not met.** Revisit in Phase 6 when
   the full pipeline is assembled.
2. **Thai vowel/tone marks are dropped** — `ชลบุรี` read as `ชลบร`/`ชลบรดี`. The consonant
   skeleton survives, so Phase 5 fuzzy-matching against the 77-province list should recover it;
   this is the main reason province resolution is deferred rather than attempted here.
3. **Evidence is synthetic.** All numbers above come from rendered text, not photographs. Nothing
   here is validated against a real Thai plate; treat accuracy claims as provisional until a real
   dataset exists.

---

## Part E — Post-Processing Slice (✅ shipped 2026-07-31)

Turns raw OCR text into canonical fields. Pure functions: no engine, no I/O, no settings.

### Modules
- `postprocess/thai.py` — `strip_thai_marks`, the shared primitive. Removes Thai combining vowel
  and tone marks; deliberately keeps base and spacing vowels (เ แ โ ใ ไ ำ), which survive
  recognition and carry real identity.
- `postprocess/plate.py` — `normalize_plate_text` → `NormalizedPlate(text, letters, digits,
  is_well_formed)`. Strips separators and stray marks, then matches
  `^(\d?[ก-ฮ]{1,3})(\d{1,4})$` — an optional leading digit (post-2012 plates), 1–3 consonants,
  1–4 digits.
- `postprocess/provinces.py` — `THAI_PROVINCES` (all 77) and `match_province`.

### Decisions
- **Library only, no endpoint.** Follows the Phase 2 precedent. Phase 6 is the first consumer;
  adding a `/postprocess` route or a service wrapper now would be speculative. This is a
  deliberate deviation from the Part A row, which named `app/services/`, `app/utils/`.
- **New `postprocess/` package rather than `app/utils/`.** Domain logic and a 77-row data table
  belong beside `ocr/` and `detector/`, not in the web layer next to upload validation.
- **Never coerce a misread.** Text failing the pattern is returned verbatim with
  `is_well_formed=False`. A confidently wrong plate number is worse than an admitted failure.
- **Matching is mark-insensitive but otherwise exact.** `match_province` returns `None` rather
  than guessing at a close candidate. Fuzzy recovery is Phase 5's job, by design.

### Verified
| Behaviour | Result |
|---|---|
| `'กข1234'` → canonical | `'กข 1234'`, well formed |
| `'1กข2345'` → modern prefix kept | `'1กข 2345'`, well formed |
| `'VEZL'` (real Phase 3 misread) | returned verbatim, **not** well formed |
| `'ชลบร'` (marks dropped) | → `'ชลบุรี'` |
| `'ชลบุรี'` (clean) | → `'ชลบุรี'` |
| Mark-stripping collisions across all 77 | **none** — asserted by test |

14 new tests; suite now 56 green. Ruff, Black, MyPy clean (`mypy app detector ocr postprocess`).

### ⚠️ Known gap — ✅ closed by Phase 5
`'ชลบรดี'` — what Phase 3 OCR *actually* produces for ชลบุรี — still returns `None`. The
recognizer both drops marks and hallucinates a trailing `ดี`, and deterministic matching
correctly refuses it. **Province resolution on real OCR output does not work yet**; that is
precisely the case Phase 5 exists to handle. Do not report province mapping as working
end-to-end until then.

*(Phase 5 closes this: `rag.validator.correct_province('ชลบรดี')` → `ชลบุรี`. The Phase 4
behaviour above is unchanged and still correct — deterministic matching still refuses to
guess; the guessing is now done deliberately, elsewhere, with guards. See Part F.)*

---

## Part F — RAG Validation Slice (✅ shipped 2026-07-31)

**Goal:** recover a province name the recognizer damaged, without ever inventing one.

**Modules:** `rag/similarity.py` (edit distance, pure), `rag/validator.py` (knowledge base +
thresholds + `correct_province` / `resolve_province`). Library only, no endpoint — Phase 2 and
Phase 4 precedent; Phase 6 is the first consumer.

### The dependency decision — deviation from Part A

Part A named **chromadb + sentence-transformers**. Both were dropped. This is the largest
deviation from the plan so far, so the reasoning is recorded in full:

- **The knowledge base is 77 static proper nouns.** A vector store exists to make similarity
  search over large corpora tractable. At 77 rows a linear scan is 0.44 ms. ChromaDB would add
  a persistence layer, a schema and a migration story to index a tuple that fits on one screen.
- **The damage is character-level, not semantic.** The recognizer drops combining marks and
  hallucinates a syllable. Edit distance models exactly that. A sentence embedding answers a
  different question — whether two names *mean* the same thing — and proper nouns are precisely
  where that signal is weakest.
- **The budget forbids it.** `CLAUDE.md` allows RAG 15 ms. Sentence-transformer inference on
  CPU is tens of milliseconds before the ~500 MB of dependencies and the model load. The
  shipped approach uses 3% of the budget.
- **It is still retrieval-augmented.** Retrieval against a knowledge base, used to correct
  generated output. Lexical retrieval is retrieval; BM25 did not stop being RAG.

If the knowledge base ever grows a real plate registry (millions of rows, fuzzy semantic
queries), revisit. Nothing here forecloses that — `rag/embeddings/` and `rag/retriever/` are
still empty scaffold directories, deliberately left in place.

**Plate correction was scoped out, not forgotten.** Part A said "province/plate KB". There is
no plate registry and no legal way to invent one; Phase 4 already validates plate *format*.
Correcting a plate number against a fabricated KB would manufacture authority the data does not
have. Out of scope until a real registry exists.

### Decisions
- **Deterministic first, fuzzy second.** `correct_province` calls Phase 4's `match_province`
  before scoring anything. This is load-bearing, not an optimization: a mark-stripped `เพชรบุรี`
  scores 1.000 but sits only 0.143 from `เพชรบูรณ์`, so the margin guard would *abstain* on a
  perfectly clean candidate. Exact matching catches it first.
- **A margin, not just a floor, decides confidence.** `MIN_SCORE = 0.6`, `MIN_MARGIN = 0.15`.
  The sweep in `docs/benchmark/rag-phase5.md` shows the floor contributes nothing to safety —
  non-province text scores ~0.17 — while the margin takes mis-attributions from 4 to 0.
- **Abstain over guess, again.** Same rule as Phase 4, now applied to a component whose whole
  purpose is guessing. 5 of 231 degraded candidates return `None` rather than a close call.
- **No shared helper with `postprocess`.** `rag.validator._lookup_key` duplicates one line of
  `postprocess.provinces._lookup_key`. Extracting it would mean editing shipped Phase 4 code
  for no behavioural gain; the duplication is one regex substitution and both paths are covered
  by tests.

### Verified
| Behaviour | Result |
|---|---|
| `'ชลบรดี'` (the real Phase 3 misread) | → `ชลบุรี` @ 0.800 |
| `'ชลบร 9'` (the other real misread) | → `ชลบุรี` @ 0.800 |
| `'เพชรบรดี'` (ties two real provinces) | → `None` — abstains |
| `'กข 1234'`, `'1กข 2345'`, `'VEZL'`, `''` | → `None` — no false positives |
| All 77 under 3 observed degradations | 226 recovered, 5 abstained, **0 wrong** |
| Latency, worst case (77-way scan) | median **0.44 ms**, p95 0.48 ms vs `<15 ms` — **met** |

24 new tests; suite now **80 green**. Ruff, Black, MyPy clean
(`mypy app detector ocr postprocess rag`). Reproducible:
`poetry run python docs/benchmark/bench_rag.py`.

**This is the first stage to meet its stated latency budget.** That says more about the budget's
distribution than about this stage: detection is unmeasured, OCR misses `<40 ms` by ~10×.

### ⚠️ Known gaps
1. **One province pair is irrecoverable by construction.** Truncating `เพชรบูรณ์` by one
   character yields `เพชรบร`, which *is* the mark-free skeleton of `เพชรบุรี`. The corrector
   returns `เพชรบุรี` at full confidence and is right to — the distinguishing character no
   longer exists. Only such pair in the 77; asserted by
   `test_truncating_phetchabun_aliases_onto_phetchaburi`.
2. **The degradation model comes from one synthetic plate.** Every input except two was damaged
   programmatically, and those two came from OCR on *rendered text, not a photograph*. The
   thresholds are tuned to failures we have seen, which is not the same as the failures that
   exist. Expect to retune once real plate photos land.
3. **Abstention needs handling upstream.** 5 of 231 return `None`. Phase 6 must render a
   missing province as unknown, not as a failure.

---

## Part G — Full `/recognize` Slice (✅ shipped 2026-08-05)

**Goal:** run the whole pipeline in one request. The first time detection, perspective, OCR,
post-processing and RAG have executed together in the running application.

**Modules:** `app/schemas/recognize.py`, `app/services/recognize_service.py`,
`app/api/recognize.py`. Pure wiring — no new dependencies, no new algorithms.

### Decisions
- **Every plate, not the best one.** A scene can hold several plates, so the response is
  `{count, plates: [...]}`, mirroring `/detect`'s shape. Reporting only the highest-confidence
  detection would silently discard plates the detector actually found. The cost is honest and
  linear: N plates ≈ N × the OCR figure.
- **Reuse the singletons, do not rebuild them.** The service imports `get_detector` and
  `get_recognizer` from the Phase 1 and Phase 3 services rather than constructing its own.
  Duplicating the `lru_cache` would put two YOLO models and two PaddleOCR engines in one process.
- **Abstention renders as `null`, never as an error.** `province: null` with
  `province_candidates` still populated means *unknown*, which is what Phases 4 and 5 built
  toward. Likewise a misread number is returned verbatim with `is_well_formed: false`. Nothing in
  this layer guesses; that was the whole point of the two stages below it.
- **One bad box does not fail the request.** `correct_perspective` raises `ValueError` for a box
  with no area inside the image. That plate is skipped with a warning and the rest are returned.
- **Crop size moved into `Settings`.** Phase 2 deferred `output_size`/`padding` until a caller
  needed them; Phase 6 is that caller. `plate_crop_width`/`plate_crop_height`/`plate_crop_padding`
  default to exactly the previous keyword values, so behaviour is unchanged — but the known
  placeholder ratio is now tunable by env var when real photographs arrive.
- **Timings are logged, not returned.** Per-stage latency goes to the structured log; the
  benchmark is the authority on numbers. Keeps the response contract to what a client needs.

### Deviation: a Phase 0 logging bug had to be fixed
`JsonFormatter` built its payload from four fixed fields and silently discarded everything passed
via `extra=` — including the `path` field that `detector/detector.py` had been logging since
Phase 1. Timings would have vanished the same way. The formatter now merges caller-supplied
fields, filtered against `LogRecord`'s reserved attribute names. Out of the strict Phase 6 scope,
but the chosen timings mechanism does not work without it.

### Verified
| Behaviour | Result |
|---|---|
| Two detections | `count: 2`, both plates fully populated |
| `'ชลบรดี'` (real Phase 3 misread) | → `ชลบุรี` @ 0.800 |
| `'เพชรบรดี'` (ambiguous) | → `province: null`, candidates preserved |
| `'VEZL'` (real Phase 3 misread) | verbatim, `is_well_formed: false` |
| No detections | `200`, `count: 0` |
| Box outside the image | skipped; neighbouring plate still returned |
| Bad type / oversize / undecodable / no weights | `415` / `413` / `400` / `503` |

13 new tests; suite now **93 green**. Ruff, Black, MyPy clean
(`mypy app detector ocr postprocess rag`).

### Measured — end to end, for the first time
`poetry run python docs/benchmark/bench_recognize.py`, recorded in
`docs/benchmark/recognize-phase6.md`:

| Stage | Median | Budget |
|---|---|---|
| Detection | **unmeasured at the time** — since measured at 25 ms, see below | <25 ms |
| Perspective | 0.5 ms | — |
| OCR | 411 ms | <40 ms — **~10× over** |
| Post-processing + RAG | 0.6 ms | <15 ms — met |
| **Total (lower bound)** | **412 ms** | **<100 ms — 4.1× over** |

OCR owns 99.7% of the runtime. Adding a trained detector would not change the shape of the
problem.

One genuinely new result: the recognizer damaged `ชลบุรี` into `'ชลบูรดี 9'` and Phase 5's
corrector recovered it at 0.667 — a damage pattern it was never tuned against. First evidence the
RAG stage works on input it did not see during development.

### ⚠️ Known gaps carried forward
1. **The `<100 ms` budget is not met and is not close.** Optimization is deliberately its own
   phase; this benchmark is the baseline to beat. Largest untried lever remains skipping
   PaddleOCR's text-detection pass on an already-rectified crop.
2. ~~**Detection has still never been timed.**~~ **Resolved after this benchmark was written.**
   Detector v0.1 was trained (`docs/experiments/detector-v0.1.md`) and timed at 25 ms
   (`docs/benchmark/detect-v0.1.md`), and `models/detector/best.pt` now exists, so `/recognize`
   no longer answers `503`. The end-to-end totals above still predate those weights and should be
   re-run before being quoted as the current cost of a full request.
3. **Evidence is still synthetic.** Rendered text, not photographs. All accuracy claims stay
   provisional until a real Thai plate dataset exists.

---

## Part H — Web UI Slice (✅ shipped 2026-08-05)

The last slice on the roadmap. Phase 6 made the pipeline answer in one request, but the only way
to exercise it was `curl`. Phase 7 puts it in a browser: upload an image, see the plates boxed on
it, read the number and province beside it.

**A pure client of the existing JSON API.** No pipeline code changed, no new dependencies —
`StaticFiles` ships with starlette, and Phase 6 already returned everything the page needs.

### Decision: no template engine

Part A had pencilled in `web/templates` with jinja2 or gradio. Neither is used. The page is
static HTML/CSS/JS served from `web/static/`, calling `POST /recognize` with `fetch`. A template
engine would render values the API already returns, making a second contract to keep in step with
the first. The dead `web/templates/.gitkeep` scaffolding was removed with this phase.

### Files

| File | Role |
|---|---|
| `app/api/web.py` | `GET /` → `FileResponse(index.html)`, `include_in_schema=False` |
| `web/static/index.html` | upload control, canvas, results table |
| `web/static/app.js` | submit, draw boxes, render rows, map errors |
| `web/static/style.css` | minimal layout |
| `app/main.py` | registers `web_router`, mounts `/static` |
| `tests/unit/test_web_ui.py` | 5 route tests |

`WEB_ROOT` resolves from the module location, not the process working directory, so `uvicorn`
started from anywhere works.

### Two things this UI is careful about

**1. It never executes OCR output.** CLAUDE.md's "Never execute OCR output" becomes a concrete
browser rule: every recognized string reaches the DOM through `textContent`, never `innerHTML`,
and is never evaluated. The canvas draws only the row *number* over each box — never the
recognized text — which also keeps Thai glyph rendering out of the canvas.

**2. It shows abstention as abstention.** `province: null` renders as "Unknown", muted and
italic, with the raw `province_candidates` shown beside it. `is_well_formed: false` shows the
number exactly as read and marks it. Neither is cleaned up into something plausible — that is
the whole point of Phases 4 and 5, and the UI is where it would have been easiest to throw away.

Box coordinates are source-image pixels, so the canvas is drawn at `naturalWidth/naturalHeight`
and scaled down by CSS. No coordinate maths, nothing to get subtly wrong.

### Verification

`pytest` 98 green (93 from Phase 6 + 5 new); ruff, black, mypy clean. Route tests deliberately assert nothing
about markup or JS internals — those tests break on every edit and prove nothing. Interactive
behaviour was checked in a real browser instead:

| Check | Result |
|---|---|
| **Submit a real photo → `503`** | ✅ **"Detector model is not installed…"** — the correct answer today, and the main thing this phase had to get right |
| Happy path (2 plates) | ✅ verified **against a stubbed detector/OCR only** — no weights exist. Boxes drew at correct coordinates, numbered to match rows; `กข 1234`/`ชลบุรี` clean, `VEZL`/`Unknown` muted |
| `.txt` renamed `.png` | ✅ caught **twice**: the client's preview rejects it before upload, and submitting anyway returns `400 "Payload does not decode to an image"`, surfaced verbatim |
| `/health`, `/docs`, `/openapi.json` | ✅ unshadowed by the `/static` mount; `GET /` absent from the schema |

### Addendum — live camera capture (same PR, same branch)

Extends the same page with a "Start camera" / "Stop" panel: `getUserMedia` shows the laptop's
camera live, and every 1.5s a frame is captured to an in-memory canvas and sent through the exact
same `recognize()` → `drawScene()` → `renderRows()` path the upload flow already uses — no new
endpoint, no pipeline change. Two existing functions in `app.js` were generalized rather than
duplicated: `recognize()` now attaches the HTTP status to the thrown error, and `drawScene()`
reads `source.naturalWidth ?? source.width` so it accepts a captured `<canvas>` frame the same way
it already accepted an uploaded `<img>`.

> **Superseded by Part I.** Camera mode no longer draws through `drawScene()` at all, so the
> `?? source.width` fallback described above has been removed and `drawScene()` is once again
> `<img>`-only. `recognize()` keeping the HTTP status on the error stands, and Part I's
> `detectOnly()` relies on it.

Because no detector weights existed at the time, every capture answered `503`. Retrying on the same
schedule would just hammer a known-broken endpoint, so the loop **auto-stops on the first `503`**
and re-enables Start; any other error (one garbled frame, a network blip) is shown but the loop
keeps going, since the next frame is likely fine.

Verified live in a browser: the first run surfaced a real bug the design hadn't anticipated — the
very first capture fired before the video's `loadedmetadata` event, so `videoWidth`/`videoHeight`
were still `0` and every camera session opened with a guaranteed, misleading `400` before the
`503`. Fixed by awaiting `loadedmetadata` (or checking `readyState`) before the first capture.
Re-verified: exactly one request, a clean `503`, Start/Stop/video state all reset correctly. The
upload flow was re-checked afterward against the two edited shared functions — unchanged
behaviour, no regression.

**Known gap:** permission-denial and manual mid-capture Stop were not independently exercised —
the automated session's `getUserMedia` was auto-granted with no fake device, and the loop
auto-stopped on `503` faster than a manual Stop click could race it. `stopCamera()` is the same
function both paths call, and it was verified via the auto-stop path, so this is a coverage gap
in the *test*, not unverified code — but it should be said plainly rather than implied covered.

### ⚠️ Known gaps carried forward
1. **The happy path has never run against real weights.** Every box and every string seen in a
   browser so far came from a stub. The `503` is the only end-to-end-honest browser result.
2. **All Phase 3–6 gaps still stand**: 412 ms vs the `<100 ms` budget, detection untimed,
   evidence entirely synthetic.
3. ~~No UI affordance for slow responses beyond a disabled button.~~ Partially addressed by the
   camera loop's own auto-stop-on-503, but the upload flow's disabled-button-only feedback during
   a slow request is still unchanged.

---

## Part I — Mode Switch + Realtime Tracking (✅ shipped 2026-08-06)

Two UI slices on top of Part H, both frontend-only — no Python, no schema, no new route.
Branches `feature/phase-8` (PR #8) and `feature/phase-9`.

### I.1 — Upload / camera mode switch (PR #8, `d8c3fe9`)

A segmented **Upload photo** / **Live camera** control (`#mode-switch`) so the two input panels are
never visible at once. `setMode()` toggles `hidden` on the form and `#camera-panel`, mirrors state
onto `aria-pressed`, and calls the existing `stopCamera()` when leaving camera mode so the device is
released rather than left streaming behind a hidden panel. Upload is the default.

**Bug found during verification, fixed in the same commit.** `form` and `#camera-panel` each declare
their own author-origin `display`, which beats the browser's default `[hidden] { display: none }` UA
rule regardless of specificity — so toggling the `hidden` attribute alone had no visual effect even
though the JS and ARIA state were correct. Fixed with explicit `[hidden] { display: none; }`
overrides for both.

### I.2 — Realtime plate tracking in camera mode (`5d5e76d`, `ca1b579`)

Before: camera mode captured a frame every 1.5 s, posted it to `/recognize`, and painted the
*returned frame* into `#preview-panel` with boxes on it. A moving plate therefore got a new frozen
snapshot every 1.5 s, never a box that follows it.

After: two independent self-rescheduling loops run over the **still-playing** video.

| Loop | Endpoint | Cadence | Job |
|---|---|---|---|
| `trackLoop()` | `POST /detect` | 200 ms | stroke boxes onto a transparent canvas over the video |
| `captureAndRecognize()` | `POST /recognize` | 1500 ms | refresh the results table (text, province) |

This is only possible because `/detect` skips OCR entirely. Measured **21–37 ms warm** browser
round trip (70 ms first call) against the real v0.1 weights, versus ~400 ms for the full pipeline —
so 200 ms is a conservative interval, not an aspirational one. Each loop schedules its next tick
only after the current one resolves, so a slow response stretches the cadence instead of stacking
requests.

**Markup.** The `<video>` is wrapped in `#camera-stage` (`position: relative`) alongside a
`#tracking-overlay` canvas (`position: absolute; inset: 0`). The video sets the rendered size; the
overlay's *internal* resolution is set to `videoWidth`/`videoHeight`, so API box coordinates map 1:1
and CSS handles the downscale — no manual coordinate math on resize. Visibility is toggled on the
wrapper so video and overlay can never desync.

**Consequences for existing code.** Camera mode no longer touches `#preview-panel` or
`drawScene()`; both are now upload-only, and `drawScene()`'s `?? source.width` canvas fallback was
removed as dead code (see the superseded note in Part H). `detectOnly()` deliberately mirrors
`recognize()`'s error handling, including carrying `.status`, so a `503` stops the whole camera
session exactly as before.

**Deliberately not built:** no row numbers on tracked boxes (correlating a 200 ms detect tick to a
1.5 s-old OCR row would need cross-frame IoU tracking; without it a number points at the wrong plate
more often than the right one), and no UI control for the interval — one named `TRACK_INTERVAL_MS`,
matching the existing `CAPTURE_INTERVAL_MS`.

**Second cascade bug, same class as I.1 (`ca1b579`).** The overlay inherited the shared
`canvas { background: var(--surface); border; border-radius }` rule written for the upload preview,
so the "transparent" overlay was an **opaque white sheet over the video** — the camera started, the
permission was granted, and the panel rendered blank white. Reported by the user, confirmed by
reading the computed style (`rgb(255, 255, 255)`), fixed by resetting `background`, `border` and
`border-radius` on `#tracking-overlay` only. `#camera-feed` still carries the same inherited white
background; harmless because it sits *behind* the video, but it is the same latent pattern.

### Verification

| Check | Result |
|---|---|
| `/detect` warm round trip, from the browser via `detectOnly()` | ✅ 21, 23, 26, 32, 37 ms (70 ms cold) |
| Overlay pixel-aligned with the video, before and after the border reset | ✅ exact match at 640×360 with a 1280×720 bitmap |
| `drawTrackingBoxes()` paints, `clearRect` wipes | ✅ 2400 px stroked for one box; 0 px remaining after clear (no smearing) |
| Overlay transparent where there is no box | ✅ off-stroke `[0,0,0,0]`, on-stroke `[224,36,94,255]` |
| Upload flow regression after the `drawScene()` simplification | ✅ preview canvas still sizes to the source image, panel still revealed |
| `pytest` / `ruff` / `black` / `mypy` | ✅ 98 passed, all clean (no backend touched) |

### ⚠️ Known gaps

1. **No camera device in this environment.** `getUserMedia`, real video playback, and an actual box
   following an actual moving plate have **never been observed**. The geometry `startCamera()`
   produces was *simulated* to test alignment and compositing. The loops and the overlay are proven;
   their integration with a live stream is not.
2. **Detection returned 0 boxes on the synthetic test plate**, matching the out-of-distribution
   caveat already in `docs/benchmark/detect-v0.1.md`. The plumbing is proven end to end; that boxes
   appear around *real* plates is not — that needs a photograph, or the camera.
3. **The 200 ms interval is measured against `127.0.0.1`.** Any real deployment adds network
   latency; the loop degrades gracefully (cadence stretches, no stacking) but the number would not
   hold.
4. **No `beforeunload`/`visibilitychange` release** — carried forward from Part H unchanged.

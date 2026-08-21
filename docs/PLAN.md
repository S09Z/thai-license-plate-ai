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
| 10 ✅ | Face detection boxes | `face/`, `app/api/face.py`, `web/static/` | — (opencv's bundled YuNet) | done — `POST /detect/faces`, opt-in overlay beside plate boxes; **17.6ms vs <25ms budget — met (see Part J)** |
| 11 ✅ | Facial landmarks | `face/landmarks.py`, `app/schemas/face.py`, `web/static/` | — (opencv contrib's Facemark LBF) | done — `?landmarks=true` fits eyebrows/eyes/nose/mouth; **1.1ms per face vs <25ms budget — met (see Part K)** |
| 12 ✅ | Whole-face mesh | `face/landmarks.py`, `web/static/` | — (opencv's `cv2.Subdiv2D`) | done — `?mesh=true` adds the jaw and a Delaunay wireframe over all 68 points; **0.4ms on top of the fit, 20.1ms serial vs <25ms — met (see Part L)** |
| 13 ✅ | Fast realtime face boxes | `app/api/face.py`, `app/services/face_service.py`, `web/static/` | — | done — `?fast=true` downsizes server-side before YuNet (720p 18.5ms → 480px 3.2ms, **5.7×**); the camera loop runs plain face boxes at a 16ms (~60fps target) cadence on their own overlay, decoupled from the 200ms plate loop (see Part M) |
| 14 ✅ | Face attributes (expression + apparent gender) | `face/attributes.py`, `app/services/face_service.py`, `web/static/` | — (opencv's `cv2.dnn`) | done — `?attributes=true` labels each face via Levi-Hassner gender (Caffe) + OpenCV Zoo expression (ONNX); infer-render-discard, weights hash-pinned in the Makefile; **~18.6ms/face inference vs <25ms — met, but off the fast path (see Part N)** |

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

---

## Part J — Face Detection Slice (✅ shipped 2026-08-07)

Phase 9 put plate boxes on the live video. This puts **face boxes** beside them, in a distinct
colour, behind an opt-in checkbox that is **off by default** — face inference should not be paid for
unless it is asked for.

### Scope boundary, deliberate and load-bearing

This is face **detection** — locating a region in a frame — and explicitly **not** face
**recognition**. Nothing here identifies a person, computes an embedding, matches against a gallery,
or persists a frame. YuNet returns five landmarks per face and they are **discarded** at the wrapper
boundary; the endpoint reports coordinates and a score, nothing else.

If identification is ever wanted it is a different feature with different consent and legal
questions — Thailand's PDPA treats biometric identifiers as sensitive personal data — and should be
designed as such, not bolted onto this endpoint.

### Decisions

- **YuNet ONNX over a Haar cascade**, on measurement rather than reputation. Haar was timed first at
  720p: 6.1 ms on a blank frame but **56.0 ms on a textured one**, a 9× content-dependent swing that
  breaks the `<25 ms` budget on its own. YuNet does not show that spread (16.6 vs 17.6 ms), and the
  benchmark times both frame kinds specifically to *check* that rather than assume it. Full numbers
  in `docs/benchmark/face-phase10.md`.
- **A separate `POST /detect/faces` rather than a flag on `/detect`.** Different model, different
  failure mode, different opt-in. It also lets the camera tick request faces only when wanted.
- **`DetectionResponse` reused unchanged.** The shape (`count`, `boxes`) and the meaning are
  identical to a plate box; a `FaceResponse` clone would be a distinction without a difference.
  `face/detector.py` likewise returns `detector.detector.Detection` — importing that frozen dataclass
  is a one-way dependency on a pure value type, cheaper than a duplicate plus mapping code.
- **Structure copied from the plate detector deliberately**: lazy load on first `detect()`,
  module-level `_load_yunet` so tests patch it exactly as `_load_yolo` is patched, `FileNotFoundError`
  → 503, `@lru_cache` service singleton, identical 415/413/400/503 exception mapping.

### Deviation from the plan: a missing face model no longer stops the camera

The plan specified that `detectFaces()` mirror `detectOnly()` so that a 503 "stops the session the
same way". Implemented literally, that meant a missing *face* model would kill *plate* tracking too.

The rationale for stopping was "don't hammer a known-broken endpoint" — and unticking the checkbox
achieves that completely, since the face request is then not issued at all. So a face 503 now
unticks "Show faces" and reports why, while plate tracking continues; a **plate** 503 still calls
`stopCamera()` unchanged. Verified live: with the face model absent, `/detect/faces` returns 503 and
`/detect` on the same server returns 200.

### Measured — `docs/benchmark/face-phase10.md`

| | Median | Budget |
|---|---|---|
| Face, flat 720p frame | 16.6 ms | |
| Face, textured 720p frame | **17.6 ms** | `<25 ms` ✅ **met** |
| Camera tick, both requests concurrent | 23.0 ms | |
| Camera tick, if issued serially | 40.6 ms | |

The tick issues both requests from a **single captured frame** via `Promise.all`, so it costs the
slower stage rather than the sum. Cold start is 203 ms, paid once per process.

This is the **second stage to meet its budget**, after RAG. Note it is not on the `/recognize` path
at all, so it does not affect the 4.1× total-pipeline miss that OCR still dominates.

### Verified

| Check | Result |
|---|---|
| `POST /detect/faces` on a **real photograph** | ✅ 1 box, conf **0.946**, visually confirmed on the face |
| w/h → x1,y1,x2,y2 conversion against live model output | ✅ correct (the real-photo box lands on the face) |
| 415 / 413 / 400 / 503 mapping | ✅ all four, against a live server |
| Face model absent → 503, plates unaffected | ✅ `/detect/faces` 503, `/detect` 200 on the same process |
| Checkbox **off** → one request per tick | ✅ `/detect` only |
| Checkbox **on** → both, concurrently | ✅ `/detect` + `/detect/faces`, issued 0 ms apart, **one** frame built |
| Single `clearRect` per tick | ✅ `clearRect → strokeRect(#e0245e) → strokeRect(#00b8d4)`; faces do not erase plates |
| `.toggle` did not leak to the upload form | ✅ `#file-input` still `display: block`, `gap: normal` |
| `/detect`, `/recognize`, `/health` regression | ✅ 200 each; `/detect/faces` present in the OpenAPI schema |
| `pytest` / `ruff` / `black` / `mypy` | ✅ **112 passed** (98 + 14 new), all clean |

Both browser passes force-reloaded the static assets before checking anything, per the stale-cache
lesson in Part I. **A stale server nearly invalidated this one too**: two `uvicorn` processes from
earlier sessions were still bound to port 8000 and answered the first round of endpoint checks.
Those results were discarded and every check re-run against a single, verified-fresh process.

### Deliberately not built

- No recognition, embeddings, identity matching, or frame persistence (see the scope boundary).
- No redaction/blur — `POST /redact` stays a separate future feature.
- No face boxes in the **upload** flow; the request was camera mode, and `/recognize` has its own
  response shape that faces would not fit without changing a shipped schema.
- No configurable face-track interval — `TRACK_INTERVAL_MS` governs both loops.
- No landmark rendering — that is Phase 11, which builds on this phase's boxes.

### ⚠️ Known gaps

1. **Detection accuracy is essentially unmeasured.** One public-domain still photograph was detected
   correctly at 0.946. That is a single frontal, well-lit portrait — it is *not* evidence for the
   conditions this feature runs in. There is still **no camera device in this environment**, so no
   face has ever been tracked in a live stream, at an angle, in motion, or in poor light.
2. **The benchmark frames contain no faces**, so the latency figures bound speed only. YuNet's cost
   is input-size-bound, so faces being present should not change them — but that is reasoning, not a
   measurement.
3. **The model is gitignored** (`.gitignore:47`, `*.onnx`), so a fresh clone must run
   `make fetch-face-model`. Absent, the endpoint answers 503 by design and the checkbox unticks
   itself rather than failing silently.
4. **The 200 ms tick is measured against `127.0.0.1`** — carried forward from Part I unchanged.

### Superseded by Phase 11

Two claims above no longer describe the shipped code:

- **The "Show faces" checkbox is gone**, replaced by a three-state `<select>` (Off / Face boxes /
  Facial features). Where this part says the checkbox "unticks itself" on a 503, the control now
  **steps down one level** instead — see Part K.
- **`DetectionResponse` is no longer the face response shape.** `POST /detect/faces` returns
  `FaceResponse` (`count`, `faces[].box`, `faces[].landmarks`). The reasoning in this part was
  sound while a face was only a box; a face is now a box *plus* optional landmarks, which is the
  distinction that was missing. Part K records the break.

---

## Part K — Facial Landmarks Slice (✅ shipped 2026-08-07)

Phase 10 put a **box** around each face. This puts **points inside it**: eyebrows, eyes, nose and
mouth, fitted with OpenCV contrib's 68-point Facemark LBF regressor and stroked over the live video
as polylines. The face control becomes three-state — Off / Face boxes / Facial features — with the
extra work opt-in at each step.

### Scope boundary, tightened rather than relaxed

Still **not** face recognition: nothing identifies a person, computes an embedding, matches a
gallery, or persists a frame. Phase 10's boundary carries forward unchanged, and this phase makes it
sharper rather than softer, because 68 landmarks are closer to a biometric template than a box is —
Thailand's PDPA treats biometric identifiers as sensitive personal data.

Concretely: **jaw points 0–16 are dropped at the wrapper boundary and never leave the process.** The
jaw contour is the most identity-bearing part of the 68 — face shape is what a naive geometric
matcher keys on — and it is not one of the four features the phase asked for. Discarding it costs
nothing here and removes the most obviously misusable output.

> **⚠️ Reversed by Phase 12.** The jaw is no longer dropped unconditionally: `?mesh=true` reports
> it, because a whole-face surface has no boundary without it. The reasoning above still stands and
> is why the mesh is a *fourth, opt-in mode* rather than a widening of `?landmarks=true`, which
> still returns no jaw. See Part L.

### Decisions

- **Facemark LBF over dlib or MediaPipe.** LBF ships in the `opencv-contrib-python` already in the
  lockfile, so the feature adds **no new dependency** — same reasoning that picked YuNet in Phase 10.
  Measured at 1.1 ms per face it is comfortably inside budget, so the accuracy/size tradeoff against
  a heavier model was never forced.
- **An explicit `Path.is_file()` check before loading.** This is not defensive padding.
  `cv2.face.createFacemarkLBF()` **constructs happily without a model** and only fails deep inside
  `fit()` with a bare `cv2.error` — which the route would surface as a **500**. The check turns that
  into `FileNotFoundError` → **503**, keeping the same contract the plate detector and YuNet already
  honour. A unit test pins it (`test_fit_raises_file_not_found_when_the_model_is_missing`).
- **A new `FaceResponse` — deliberately breaking Phase 10's "reuse `DetectionResponse`" call.** That
  call was right when a face was just a box. A face is now a box *plus* optionally six point groups,
  which a `DetectionResponse` cannot carry. The response nests rather than parallels
  (`faces[].box` + `faces[].landmarks`) so a landmark set cannot drift away from the box it belongs
  to. **This changes an already-shipped response shape**; Phase 10 is one unmerged PR away and has
  no external consumers, so it was cheaper to fix now than to add a second endpoint.
- **`?landmarks=true` on the existing endpoint, not a new one.** The model, the upload validation and
  the failure modes are shared; only the depth of the result changes. Default `false` means the
  54 MB model never loads for callers that do not ask.
- **Groups named by iBUG-68 convention, i.e. the *subject's* right and left.** So `right_eyebrow`
  renders on the **left** of the image. Renaming to viewer-relative would be friendlier to the
  drawing code and wrong against every reference; a unit test and the real-photograph coordinates
  both pin the convention instead.

### The 503 problem, and a self-correcting answer

Two models can now be missing independently, and the server deliberately does not say which — the
503 detail is "A face model required by this request is not available". Leaking model filenames to
the browser is not worth the diagnostic convenience, and parsing server prose in the client would be
worse.

So the client **steps down one level** on a face 503: *Facial features* → *Face boxes* → *Off*. If
only the landmark model is missing, the next tick succeeds at *Face boxes* and stops there. If both
are missing, one more tick steps to *Off*. The control settles on the highest mode that actually
works without the client ever knowing which file is absent. A **plate** 503 still calls
`stopCamera()`, unchanged from Phase 9.

### Measured — `docs/benchmark/face-landmarks-phase11.md`

| | Median | Budget |
|---|---|---|
| Fit, 1 face | **1.1 ms** | `<25 ms` ✅ **met** |
| Fit, 3 faces | 3.3 ms | |
| Per extra face | 1.1 ms | |
| Detect + fit on a real photograph, serial | 7.6 ms | |

Cold start is **495 ms** — the largest in the project, loading a 54 MB model — paid once per process,
which is why the load is lazy and gated on the query parameter. The cost is **per-face** and linear;
at 1.1 ms a face it would take ~17 faces in one frame for the landmark stage alone to reach the
25 ms detection budget.

Third stage to meet its budget, after RAG and face detection. Not on the `/recognize` path, so the
4.1× total-pipeline miss that OCR dominates is unaffected.

### Verified

| Check | Result |
|---|---|
| Landmarks on a **real photograph** | ✅ all six groups land on their features, visually confirmed on a rendered overlay |
| Subject-right convention on **real model output** | ✅ `right_eyebrow` x 197–241 < `left_eyebrow` x 255–293; `right_eye` 212–232 < `left_eye` 261–281 |
| Point counts per group | ✅ 5 / 5 / 9 / 6 / 6 / 20 (eyebrows, nose, eyes, mouth) — jaw's 17 absent |
| Default request omits landmarks | ✅ `landmarks: null`, landmark model never loaded |
| Missing landmark model + `?landmarks=true` | ✅ **503, not 500** — against a live server with the path pointed at a nonexistent file |
| 415 / 413 / 400 mapping | ✅ unchanged |
| Per-mode request sets in the browser | ✅ Off → `/detect` only; Face boxes → both; Facial features → both, with `?landmarks=true` |
| 503 step-down | ✅ all three transitions observed live (features → boxes → off) |
| Single `clearRect` per tick | ✅ landmarks draw inside the existing clear; boxes not erased |
| `.toggle select` did not leak to `#file-input` | ✅ computed styles unchanged |
| `/health`, `/detect`, `/recognize` regression + OpenAPI shape | ✅ 200 each, `landmarks` parameter present |
| `pytest` / `ruff` / `black` / `mypy` | ✅ **124 passed** (112 + 12 new), all clean |

Both browser passes force-reloaded the static assets first (Part I lesson) and `lsof` confirmed
exactly one listener on the port before any live result was trusted (Part J lesson).

### Deliberately not built

- No jaw contour — dropped on purpose, see the scope boundary.
- No recognition, embeddings, identity matching, or frame persistence — carried forward from Part J.
- No landmarks in the **upload** flow, for the same reason Phase 10 kept faces out of it.
- No per-group toggles; the three-state control is one axis of cost, not six.
- No blink/gaze/expression derivation from the points — that is inference about a person, which is a
  different feature with different consent questions.
- No smoothing across frames. Landmarks jitter more than boxes do, and a temporal filter needs frame
  history, which this phase deliberately does not keep.

### ⚠️ Known gaps

1. **Accuracy is one photograph.** A single frontal, well-lit, high-quality portrait fitted
   correctly and was inspected visually. There is still **no camera device in this environment**, so
   no face has been tracked in a live stream, at an angle, in motion, or in poor light. LBF is an
   older, cheaper regressor, so off-frontal degradation is the *expected* failure mode and is
   entirely unmeasured here.
2. **The synthetic benchmark fits fabricated boxes on a noise frame.** That bounds the cost of a fit
   — the regression does the same work wherever it is pointed — and says nothing about accuracy. The
   real-photograph run is the only accuracy evidence.
3. **Jitter is unquantified.** Point 6 above declines to build smoothing partly because the amount of
   jitter has never been observed, there being no live stream to observe it in.
4. **The model is gitignored** (`.gitignore:50`, `models/face/*.yaml`), so a fresh clone must run
   `make fetch-face-landmark-model`. A blanket `*.yaml` was rejected — this repo has config YAML that
   must stay tracked.
5. **The 200 ms tick is measured against `127.0.0.1`** — carried forward from Part I unchanged.

---

## Part L — Whole-Face Mesh Slice (✅ shipped 2026-08-07)

Phase 11 put **points** inside the face box. This joins them into a **surface**: a Delaunay
triangulation over all 68 landmarks, stroked as a wireframe. The face control becomes four-state —
Off / Face boxes / Facial features / Face mesh — and each step is still opt-in.

### The Part K jaw decision is deliberately reversed

Part K dropped jaw points 0–16 at the wrapper boundary and argued that discarding them "costs
nothing here". For a mesh it costs everything: the jaw *is* the face boundary, and without it the
triangulation covers only the middle of the face.

So the jaw is back — but the reasoning that removed it was sound and is preserved structurally
rather than discarded:

- **`?landmarks=true` is byte-for-byte unchanged.** It still returns six feature groups and no jaw.
  The API test asserts the response dict **exactly**, so the three new keys appear there as `null`;
  that exact comparison is what proves the default did not quietly widen.
- **The jaw is reachable only through `?mesh=true`.** One flag, one decision, visible in the request.
- **`test_fit_omits_the_jaw_contour_by_default`** was kept, not deleted, and narrowed to the default
  path. It is now the regression test for the privacy default rather than a statement that the jaw
  is never computed.

This is a real reversal of a documented decision, recorded as one. Part K carries a pointer here.

### Decisions

- **`cv2.Subdiv2D` over a dense 468-point FaceMesh.** Subdiv2D is in the pinned OpenCV 4.10.0, so
  the mesh adds **no new dependency and no new model** — the same reasoning that picked YuNet in
  Phase 10 and Facemark LBF in Phase 11. MediaPipe FaceMesh has Apache-2.0 weights, but its ONNX
  exports fuse custom ops `cv2.dnn` will not load, so it would pull in `onnxruntime`. Deferred to
  its own phase, with evidence, rather than smuggled in here.
- **Triangles are index triples into a flat 68-point array, not coordinate triples.** A quarter of
  the payload, and the topology can be checked against points the client already has. `points` is
  sent explicitly instead of asking clients to concatenate the seven groups in iBUG order — that
  ordering would be an invisible contract no test could catch if it drifted.
- **Triangulated server-side.** It is testable in pytest; the repo has no JS test infrastructure,
  so the same logic in the browser would ship unverified.
- **The Subdiv2D rectangle comes from the points, not the face box.** Not a style choice: **1 of 68
  fitted points fell outside the detection box** on the reference photograph, and Subdiv2D raises on
  any insert outside its rectangle.
- **The renderer is chosen by the data, not the dropdown** — `landmarks.triangles ? drawMesh :
  drawLandmarks`. The mode that produced a response is already encoded in it, and reading the
  control at draw time could disagree with the frame in hand.

### Measured — `docs/benchmark/face-mesh-phase12.md`

| | Median | Budget |
|---|---|---|
| Fit, 1 face | 1.5 ms | |
| Fit + mesh, 1 face | 1.9 ms | |
| **Mesh overhead** | **0.4 ms** | |
| Detect + fit + mesh, serial | **20.1 ms** | `<25 ms` ✅ **met** |

**A 4× win came out of the benchmark, not the design.** The first implementation triangulated in
1.75 ms; profiling showed OpenCV's share was 0.04 ms and the rest was the index-mapping loop.
`getTriangleList()` returns a numpy array, so iterating it directly yields `numpy.float32` scalars,
and hashing those as dict keys costs ~7× native floats. One `.tolist()` took it to 0.23 ms with
byte-identical output. The call is commented as load-bearing, because it looks removable.

### Verified

| Check | Result |
|---|---|
| `?mesh=true` on the **real photograph** | ✅ 17 jaw points, 68 flat points, **113 triangles** |
| Triangle indices | ✅ 0–67, all valid, 3 distinct per triangle, all 68 points used |
| iBUG order of `points` | ✅ `points[0:17] == jaw`, `points[48:68] == mouth` |
| `?landmarks=true` privacy gate | ✅ `jaw`, `points`, `triangles` all `null`; mouth still present |
| No flag | ✅ `landmarks: null`, unchanged |
| 415 / 413 / 400 mapping with `?mesh=true` | ✅ all three |
| Missing LBF model → 503, path not leaked | ✅ `?mesh=true` **and** `?landmarks=true` 503; boxes-only 200 |
| OpenAPI exposes both params | ✅ `['landmarks', 'mesh']` |
| Mesh rendered on the portrait and **looked at** | ✅ boundary traces the jaw, surface is connected |
| Browser: four modes → correct URLs | ✅ `/detect/faces`, `?landmarks=true`, `?mesh=true` |
| Browser: data-driven dispatch | ✅ features response draws 0 mesh-coloured pixels; mesh response draws them |
| Single `clearRect` per tick | ✅ 1 call; all 6 feature colours drawn, **0 survive** into the next (mesh) tick |
| Plate boxes still visible under the mesh | ✅ `#e0245e` present alongside the wireframe |
| `pytest` / `ruff` / `black` / `mypy` | ✅ **133 passed** (124 + 9 new), all clean |

The browser pass force-reloaded the static assets first, per the Part I lesson — the first
navigation showed 1 console error and 0 after. **Two stale `uvicorn` processes were again bound to
port 8000**; rather than kill what may be the user's own server, every check ran against a
purpose-started process on port 8010, confirmed to be the only listener there.

### Deliberately not built

- **No dense 468-point surface.** 68 points is a coarse mesh; a smooth one is a separate phase with
  a real dependency decision (see Decisions).
- **No filled/shaded triangles.** A wireframe shows the topology; a filled surface would hide the
  video it is drawn over.
- **No mesh in the upload flow**, matching Phases 10–11: `/recognize` has its own response shape.
- **No temporal smoothing.** Carried forward from Part K unchanged.
- **No expression/attribute inference.** Researched (FER+ is the candidate, MIT, 3.7 ms per face)
  but not built, and out of scope here.

### ⚠️ Known gaps

1. **"Whole face" means jaw-to-eyebrow.** iBUG-68 has **no forehead or scalp points**, so the convex
   hull of the 68 is the mesh boundary and the forehead is not covered. Visible in the rendered
   check. This is a property of the landmark set, not a bug, and it is the main reason a dense
   FaceMesh might still be wanted.
2. **Topology may flicker on live video.** Delaunay is recomputed per frame, so near-cocircular
   points can flip triangles between ticks. Landmark jitter was already unquantified in Part K, and
   with no camera device here this remains **unobserved rather than ruled out**. The fallback, if it
   looks bad, is to freeze the topology from a reference shape and reuse the index list.
3. **Still no camera device in this environment** — carried forward from Parts J and K. Every check
   is a still photograph or an injected payload; nothing has run against a live stream.
4. **Accuracy is one photograph.** 113 triangles on one frontal, well-lit portrait. The mesh is only
   as good as the fit under it, and off-frontal LBF degradation remains unmeasured.
5. **Payload grows to ~2.8 KB per meshed face**, versus a few hundred bytes for boxes. Fine at the
   200 ms tick over loopback; unmeasured over a real network, like every latency figure since
   Part I.

---

## Part M — Fast Realtime Face Boxes (✅ shipped 2026-08-07)

Phase 12's face overlay updated at the **200 ms plate cadence** — a box took 1/5 of a second to
follow a moving face. This phase decouples face boxes from that loop and runs them toward the
camera's 60 fps rate, by making the per-tick detection cheap enough to run that fast.

**Target: 60 fps.** The 16.7 ms/frame budget is the design goal. Detection on a full 720p frame
cost 18.5 ms — already over — so the first, load-bearing change is to stop detecting at full size.

### Decisions

- **`?fast=true` downsizes server-side, and every result is scaled back.** The route decodes the
  upload, shrinks it to a 480px longest edge (`APP_FACE_FAST_MAX_SIZE`), runs YuNet on that, and
  rescales every box (and any landmark points) into **source-frame pixels** before responding. The
  client's 1:1 overlay invariant is untouched — no coordinate math moved into the browser, which is
  the one thing every phase since Part H has deliberately avoided.
- **A 5.7× speedup that costs nothing to opt out of.** 18.46 ms → **3.22 ms** median on 720p,
  measured with the real model (`docs/benchmark/face-fast-phase13.md`). The flag is off by default;
  every existing caller is byte-for-byte unchanged.
- **Fast is for boxes; features and mesh stay full-resolution.** Landmark and mesh precision is the
  product there, so the rich modes keep the 200 ms cadence and full-res fit. The control's cadence
  is per-mode: `boxes` → `FACE_FAST_MS` (16), `features`/`mesh` → `TRACK_INTERVAL_MS` (200).
- **A separate overlay, so the two cadences never fight.** Faces move to a second transparent
  canvas (`#face-overlay`) stacked over `#tracking-overlay`. Each loop clears only its own sheet —
  the face loop redrawing every 16 ms can never erase plate boxes the slower loop just drew. This
  replaces Phase 10's "one clearRect per tick" invariant with one per overlay per loop.
- **One face loop, not three.** `faceLoop()` reads the control each tick and picks both the request
  shape (`?fast` / `?landmarks` / `?mesh`) and the cadence from the mode, so the overlay is never
  drawn from two loops that could disagree. `trackLoop()` is now plates-only, and its 503 handling
  is back to the simple "a plate 503 stops the session" — the face step-down moved to `faceLoop()`.
- **Fast mode does not promise 60 fps, it buys headroom.** 3.22 ms is the detector alone. The
  browser must still capture, JPEG-encode, round-trip and draw per tick, so the honest claim is "the
  detector is no longer the bottleneck"; the achieved browser rate is unmeasured (no camera device,
  carried forward from Parts J–L).

### Measured — `docs/benchmark/face-fast-phase13.md`

| | Median | Budget |
|---|---|---|
| Detect, 720p full | 18.46 ms | |
| **Detect, fast (480px)** | **3.22 ms** | **5.7× speedup** |
| Fast detection ceiling | 310 fps | 60 fps target |

Reproducible with `make bench-face-fast`.

### Verified

| Check | Result |
|---|---|
| Box rescale with `?fast=true` (16×8 upload, 8px cap) | ✅ (3,4,13,16) → (6,8,26,32) — back in source pixels |
| Landmark points rescaled too | ✅ mouth (7,13) → (14,26) |
| Below the cap, fast is byte-identical to non-fast | ✅ asserted in tests |
| Default request unchanged | ✅ existing 11 face-API tests still pass untouched |
| 415 / 413 / 400 / 503 mapping with `?fast=true` | ✅ (400 against a live server) |
| OpenAPI exposes `fast` | ✅ `['landmarks', 'mesh', 'fast']` |
| Live server, real YuNet, `?fast=true` parity | ✅ same empty result as non-fast on a real photo; no crash |
| Browser: boxes mode → `?fast=true` at ~16ms, no other mode's URL | ✅ static JS review; `node --check` clean |
| Browser: plates still drawn on the lower overlay | ✅ separate canvases, plate loop untouched |
| `pytest` / `ruff` / `black` / `mypy` (changed modules) | ✅ **149 passed** (5 new in `test_face_api.py`, 1 restored), all clean |

### Deliberately not built

- **No 60 fps guarantee and no measured browser frame rate.** The 16 ms cadence is the design
  target; a real number needs a camera, which this environment does not have. The loop
  self-schedules after each tick, so a slow response stretches the cadence instead of stacking.
- **No client-side downscaling.** Keeping the full frame in the browser and letting the server
  downscale preserves the no-coordinate-math invariant. Client-side resize would need the client to
  rescale boxes back — the exact class of bug every phase since Part H avoided.
- **No landmark/mesh fast mode.** Downscaled feature points would be visibly coarser where the
  feature's whole point is precision. The fast flag *works* with them (tests pin the rescale), but
  no UI path requests that combination.
- **No change to the upload flow.** `?fast` is a camera-loop optimisation; `/recognize` keeps its
  own shape.

### ⚠️ Known gaps

1. **No camera device in this environment** — carried forward unchanged from Parts J–L. The fast
   detection number is the detector on a synthetic frame; achieved browser fps is unmeasured.
2. **Accuracy at the downscaled size is untested.** The benchmark frame has no face in it. YuNet at
   480px on a small/distant face may miss — the trade of the fast path, and entirely unmeasured.
3. **No real face photograph exercised live here.** Unit tests pin the rescale math and the live
   server check pinned the empty-result parity, but no real face box was produced in this session.
4. **Frontend verified by static review only.** The split of drawing and the 16 ms cadence were not
   observed in a browser; the JS is syntax-checked and the control flow reviewed, but live-box
   behaviour is unverified, consistent with the project's no-JS-test-infra stance.

## Part N — Face Attributes: Expression + Apparent Gender (✅ shipped 2026-08-10)

Phases 10–12 report only *where* a face's features sit. This phase widens the face pipeline to
infer *something about the person* — an expression and an apparent gender per box — behind a new
`?attributes=true` flag and an "Expression & gender" camera mode. The pattern is deliberately
**infer, render, discard**: nothing is stored, no frame persisted, and the result is never linked to
a plate number. The two pipelines share a frame, not an output.

**Naming is a decision, not a detail.** The response says `expression` (not "emotion" — a face does
not reliably reveal an internal state; Barrett et al., 2019) and `apparent_gender` (a
visual-presentation classifier over two labels, not a determination of sex). Both are carried on a
new `FaceAttributesModel`, separate from `BoundingBox`, so face concepts never leak into `/detect`.

### Decisions

- **Two `cv2.dnn` nets, no new dependency.** Levi-Hassner gender (Caffe, 227² BGR, fixed mean,
  softmax head → 2 classes) and OpenCV Zoo's MobileFaceNet expression (ONNX, 112² eye-aligned,
  RGB → 7 FER labels). `cv2.dnn` already ships with the contrib OpenCV the LBF landmarker needs.
- **Attributes imply fitting — for the eyes only.** Expression alignment warps the face onto the
  ArcFace two-eye template, so `?attributes=true` runs the landmarker to get eye centers even when
  landmark *geometry* isn't requested (`landmarks` stays `null` in that response). Gender needs only
  the box, so it still runs when a fit doesn't converge; expression abstains on that face.
- **Each label is independently gated, but the score is always reported.** Below
  `APP_FACE_ATTRIBUTE_MIN_CONFIDENCE` (0.5) the label is `null` while the winning score stays
  visible, so a near-call is never silently hidden or rounded into a guess.
- **Weights are hash-pinned, not trusted blindly.** `make fetch-face-attribute-models` downloads
  each file and checks it against a SHA-256 pinned in the Makefile; a changed or tampered mirror
  fails the target loudly. The gender caffemodel's hash was verified **identical across three
  independent mirrors including Gil Levi's original repo**; the expression ONNX URL is pinned to an
  `opencv_zoo` commit and its hash reproduced from that commit. `*.caffemodel` and
  `models/face/*.prototxt` were added to `.gitignore` (the 45 MB caffemodel must never be committed).
- **The UI degrades one step, with its own message.** A 503 in attributes mode steps down to plain
  `features` (a missing gender/expression model is the likely cause, and features still work) and
  shows a message naming `make fetch-face-attribute-models` — not the landmark-model wording the
  other modes borrow. If the landmark model is the one missing, the next tick steps features → boxes,
  so the chain self-heals. Labels are English tokens drawn on the canvas; Thai plate glyphs are still
  kept off it, as before.

### Trust / verification

- Both real nets load under the project's cv2 4.10 and produce the expected shapes — gender `(1, 2)`
  summing to 1.0 (softmax head), expression `(1, 7)`. (Note: bare-Python cv2 **5.0** *removed*
  `readNetFromCaffe`; the project runs on contrib 4.10, where it exists.)
- The real inference path — configured reader → `cv2.dnn` forward → softmax → threshold — was
  exercised end-to-end through the service on a neutral crop (not mocked); it returns a structured,
  thresholded `FaceAttributes`. Full gate green: 153 tests, ruff, black, mypy.

### Deliberately not built

- **No age.** The Levi-Hassner release also ships an age net; it was left out to keep the slice to
  what was asked and avoid a second research-licensed model.
- **No fast (downscaled) attributes path.** The gender CaffeNet dominates cost (~18.6 ms/face), so
  attributes stay on the 200 ms cadence rather than the 16 ms fast-box path.
- **No plate linkage.** By design — the whole point is that this output touches nothing else.

### ⚠️ Known gaps

1. **No real face photograph exercised live here.** The real weights were run through the full code
   path on a neutral crop, but no actual face was detected-then-labelled in this session; a live
   check needs the camera UI (`make run` → camera → "Expression & gender").
2. **Accuracy is entirely unmeasured** — no labelled face set, no confusion matrix. The confidence
   gate is a guess at 0.5.
3. **Research-use licensing.** The Levi-Hassner Adience weights are licensed for research use; this
   is noted in the Makefile and must be resolved before any non-research deployment.
4. **Fairness is unaudited.** A two-label apparent-gender classifier trained on one dataset will have
   uneven error across faces it under-represents. Shipped as opt-in and label-abstaining, but not
   evaluated for bias.
5. **Frontend verified by static review only** — the label plate and camera mode were reviewed and
   the JS syntax-checked, but not observed in a browser, consistent with the no-JS-test-infra stance.

### Optimization backlog (proposed 2026-08-10 — not committed, decide per item later)

Ideas for improving the **expression** classifier, captured for future consideration. None are
scheduled; each is a separate go/no-go. Ordered by impact ÷ effort. Constraints assumed: `cv2.dnn`
only, no `torch`, camera-only, infer/render/discard, no plate linkage.

**Scope caveat first.** This detects **facial expression** (7 FER labels), *not* depression or any
mood/clinical condition. Inferring depression from a face is scientifically unreliable and ethically
fraught — it stays permanently out of scope; no code here should ever label a person that way.

**Tier 1 — cheap, high value (no new deps)**

1. **Measure before optimizing.** No accuracy number exists today (see Known gap #2). Build a small
   labelled eval set (~100–200 crops across the 7 classes; FER2013 test or own) and add
   `docs/benchmark/bench_expression.py` → confusion matrix + per-class accuracy + latency. This is
   the prerequisite scoreboard that makes every model swap below decidable instead of guesswork.
2. **Temporal smoothing (camera).** The face loop runs ~200 ms/tick; per-frame predictions flicker.
   Smooth the last N frames per tracked face (EMA on the probability vector or majority vote).
   Largest perceived-stability win for near-zero cost.
3. **Calibration + honest confidence.** FER softmax saturates (see [[expression-model-research]]).
   Show top-1 vs top-2 margin, or fit one temperature scalar on the eval set; tune
   `APP_FACE_ATTRIBUTE_MIN_CONFIDENCE` off the eval curve instead of the guessed 0.5.

**Tier 2 — robustness (medium effort)**

4. **Pose / quality gating.** Off-frontal faces break the 2-point alignment. Estimate yaw from
   landmark symmetry; abstain (`?`) when too rotated or the face is smaller than N px, rather than
   guessing. Cheap geometry check that removes the worst errors.
5. **Better alignment.** Add nose/mouth from the existing 68-pt fit → full 5-point ArcFace template
   → a proper affine, less sensitive to eye-localization noise than the current 2-eye similarity.

**Tier 3 — model swaps (measure the payoff via #1)**

6. **A/B the model.** Wire the already-researched **FER+** (MIT, ~3.7 ms, `cv2.dnn`) behind a config
   flag and let #1's benchmark pick the winner on our data — don't assume MobileFaceNet is best.
7. **Small ensemble.** Average MobileFaceNet + FER+ probabilities; often +2–4% for near-zero code,
   still no new deps. Only if the budget allows two forwards.

**Tier 4 — performance (only if latency bites, after accuracy work)**

8. Batch faces into one `blobFromImage`, use an **int8** ONNX variant, downscale the alignment input.
   Guided by the benchmark, not assumption.

**Suggested first slice if pursued:** #1 + #2 + #3 as one small phase — low-risk, no new deps, and #1
unblocks #6/#7. (Geometry-only expression via the mesh polygon was considered and rejected: it
discards the skin-texture signal a CNN reads and generally *loses* accuracy — see the 2026-08-10
discussion; only a learned hybrid, not pure geometry, would help.)

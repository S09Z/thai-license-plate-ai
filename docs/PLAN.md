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
| **3** | **OCR** (next) | `ocr/`, `app/api/` | paddleocr | crop → raw text + province candidates; <40ms |
| 4 | Post-processing | `app/services/`, `app/utils/` | — | normalize plate format, map province; pure-function tests |
| 5 | RAG validation | `rag/`, `app/services/` | chromadb, sentence-transformers | correct OCR against province/plate KB; <15ms |
| 6 | Full `/recognize` | `app/api/recognize.py` | — | chains 1→5, one JSON response; total <100ms budget checked |
| 7 | Web UI | `web/templates`, `web/static` | (jinja2/gradio TBD) | upload image → view boxed result + text |

Cross-cutting (fold in as stages land, not upfront): model registry under `models/` with
version/dataset/metrics/git-commit (CLAUDE.md Model Rules); experiment log under `docs/experiments/`;
latency benchmark under `docs/benchmark/`. Deferred until Phase 3+ to avoid speculative infrastructure.

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

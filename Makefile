.PHONY: help install test lint format format-check typecheck check run bench bench-ocr bench-rag bench-recognize bench-detect bench-face fetch-face-model fetch-face-landmark-model clean

help:
	@echo "install         Install dependencies via Poetry"
	@echo "test            Run the pytest suite"
	@echo "lint            Run ruff"
	@echo "format          Apply black formatting"
	@echo "format-check    Check formatting without writing changes"
	@echo "typecheck       Run mypy over app, detector, face, ocr, postprocess, rag"
	@echo "check           test + lint + format-check + typecheck (the full gate)"
	@echo "run             Start the dev server at localhost:8000"
	@echo "bench           Run all reproducible benchmarks (ocr, rag, recognize, detect, face)"
	@echo "bench-ocr       Run docs/benchmark/bench_ocr.py"
	@echo "bench-rag       Run docs/benchmark/bench_rag.py"
	@echo "bench-recognize Run docs/benchmark/bench_recognize.py"
	@echo "bench-detect    Run docs/benchmark/bench_detect.py"
	@echo "bench-face      Run docs/benchmark/bench_face.py"
	@echo "fetch-face-model  Download the YuNet ONNX face detection model"
	@echo "fetch-face-landmark-model  Download the LBF 68-point landmark model"
	@echo "clean           Remove Python and pytest caches"

install:
	poetry install

test:
	poetry run pytest -q

lint:
	poetry run ruff check .

format:
	poetry run black .

format-check:
	poetry run black --check .

typecheck:
	poetry run mypy app detector face ocr postprocess rag

check: test lint format-check typecheck

run:
	poetry run uvicorn app.main:app --reload

bench: bench-ocr bench-rag bench-recognize bench-detect bench-face

bench-ocr:
	poetry run python docs/benchmark/bench_ocr.py

bench-rag:
	poetry run python docs/benchmark/bench_rag.py

bench-recognize:
	poetry run python docs/benchmark/bench_recognize.py

bench-detect:
	poetry run python docs/benchmark/bench_detect.py

bench-face:
	poetry run python docs/benchmark/bench_face.py

# The model is gitignored (*.onnx), so a fresh clone fetches it here.
fetch-face-model:
	mkdir -p models/face
	curl -fsSL -o models/face/face_detection_yunet_2023mar.onnx \
		https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# 54MB, and only needed for landmarks, so it is a separate target from the
# detector model above rather than a second line inside it.
fetch-face-landmark-model:
	mkdir -p models/face
	curl -fsSL -o models/face/lbfmodel.yaml \
		https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -not -path "./.venv/*" -exec rm -rf {} +

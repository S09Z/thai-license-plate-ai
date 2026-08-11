.PHONY: help install test lint format format-check typecheck check run bench bench-ocr bench-rag bench-recognize bench-recognize-accuracy bench-detect bench-face bench-face-fast fetch-face-model fetch-face-landmark-model fetch-face-attribute-models clean

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
	@echo "bench-recognize-accuracy  Grade /recognize on the real-plate eval set"
	@echo "bench-detect    Run docs/benchmark/bench_detect.py"
	@echo "bench-face      Run docs/benchmark/bench_face.py"
	@echo "bench-face-fast Run docs/benchmark/bench_face_fast.py"
	@echo "fetch-face-model  Download the YuNet ONNX face detection model"
	@echo "fetch-face-landmark-model  Download the LBF 68-point landmark model"
	@echo "fetch-face-attribute-models  Download the gender + expression models (hash-checked)"
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
	poetry run mypy app detector face ocr postprocess rag eval

check: test lint format-check typecheck

run:
	poetry run uvicorn app.main:app --reload

bench: bench-ocr bench-rag bench-recognize bench-detect bench-face bench-face-fast
bench-ocr:
	poetry run python docs/benchmark/bench_ocr.py

bench-rag:
	poetry run python docs/benchmark/bench_rag.py

bench-recognize:
	poetry run python docs/benchmark/bench_recognize.py

bench-recognize-accuracy:
	poetry run python docs/benchmark/bench_recognize_accuracy.py

bench-recognize-latency:
	poetry run python docs/benchmark/bench_recognize_latency.py

bench-detect:
	poetry run python docs/benchmark/bench_detect.py

bench-face:
	poetry run python docs/benchmark/bench_face.py

bench-face-fast:
	poetry run python docs/benchmark/bench_face_fast.py

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

# Apparent-gender (Levi-Hassner Caffe) and expression (OpenCV Zoo ONNX) models.
# Each download is checked against a SHA-256 pinned below, so a changed or
# tampered mirror fails the target loudly instead of installing silently. The
# gender hash was verified identical across three mirrors including the original
# author's repo; the expression URL is pinned to an opencv_zoo commit. Verified
# 2026-08-10. Note: the Levi-Hassner weights are licensed for research use.
GENDER_CAFFEMODEL_SHA = ac7571b281ae078817764b645a20541bd6aa1babeac20a45e6d8de7d61ba0e50
GENDER_PROTOTXT_SHA   = c1961acc32e6e9ce855f6ec4973e9a939cc2d49089a8aaefeafa0e100fb110cc
EXPRESSION_ONNX_SHA   = 4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9

fetch-face-attribute-models:
	mkdir -p models/face
	curl -fsSL -o models/face/gender_net.caffemodel \
		https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel
	curl -fsSL -o models/face/gender_deploy.prototxt \
		https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_deploy.prototxt
	curl -fsSL -o models/face/expression.onnx \
		https://github.com/opencv/opencv_zoo/raw/3c4f8c9308075d22f148f74b9306f0222a9aeb30/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx
	@echo "$(GENDER_CAFFEMODEL_SHA)  models/face/gender_net.caffemodel" | shasum -a 256 -c -
	@echo "$(GENDER_PROTOTXT_SHA)  models/face/gender_deploy.prototxt" | shasum -a 256 -c -
	@echo "$(EXPRESSION_ONNX_SHA)  models/face/expression.onnx" | shasum -a 256 -c -

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -not -path "./.venv/*" -exec rm -rf {} +

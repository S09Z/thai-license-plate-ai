.PHONY: help install test lint format format-check typecheck check run bench bench-ocr bench-rag bench-recognize clean

help:
	@echo "install         Install dependencies via Poetry"
	@echo "test            Run the pytest suite"
	@echo "lint            Run ruff"
	@echo "format          Apply black formatting"
	@echo "format-check    Check formatting without writing changes"
	@echo "typecheck       Run mypy over app, detector, ocr, postprocess, rag"
	@echo "check           test + lint + format-check + typecheck (the full gate)"
	@echo "run             Start the dev server at localhost:8000"
	@echo "bench           Run all reproducible benchmarks (ocr, rag, recognize)"
	@echo "bench-ocr       Run docs/benchmark/bench_ocr.py"
	@echo "bench-rag       Run docs/benchmark/bench_rag.py"
	@echo "bench-recognize Run docs/benchmark/bench_recognize.py"
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
	poetry run mypy app detector ocr postprocess rag

check: test lint format-check typecheck

run:
	poetry run uvicorn app.main:app --reload

bench: bench-ocr bench-rag bench-recognize

bench-ocr:
	poetry run python docs/benchmark/bench_ocr.py

bench-rag:
	poetry run python docs/benchmark/bench_rag.py

bench-recognize:
	poetry run python docs/benchmark/bench_recognize.py

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -not -path "./.venv/*" -exec rm -rf {} +

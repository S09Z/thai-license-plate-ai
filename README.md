# Thai License Plate AI

Detect Thai vehicle license plates from a camera or PNG, recognize the plate number and
province, and (in future) apply RAG-based correction. Results are served as JSON via a
FastAPI API with a web UI.

## Pipeline

```
Image -> Detection -> Perspective Correction -> OCR -> Post Processing -> RAG Validation -> JSON API -> Web UI
```

> **Status:** early scaffold. Only the FastAPI service skeleton and a `/health` endpoint
> are implemented so far. The detection / OCR / RAG stages are not yet built.

## Requirements

- Python 3.12
- [Poetry](https://python-poetry.org/) 2.x

## Quickstart

```bash
# Install dependencies
poetry install

# Run the API (http://127.0.0.1:8000)
poetry run uvicorn app.main:app --reload

# Check it is alive
curl http://127.0.0.1:8000/health
# -> {"status":"ok","service":"thai-license-plate-ai","version":"0.1.0"}
```

## Development

```bash
poetry run pytest        # tests
poetry run ruff check .  # lint
poetry run black .       # format
poetry run mypy app      # type check
```

Configuration is read from environment variables (prefix `APP_`); see `.env.example`.

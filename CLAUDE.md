# CLAUDE.md — AI Red Team Lab (Evidence-Driven, Read-Only)

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed.

## 1. Think Before Coding
- State assumptions explicitly.
- If uncertain, ask.
- Present tradeoffs.
- Prefer simpler solutions.

## 2. Simplicity First
- Minimum code.
- No speculative abstractions.
- No unnecessary flexibility.

## 3. Surgical Changes
- Only modify what is requested.
- Do not refactor unrelated code.
- Remove only dead code created by your own changes.

## 4. Goal-Driven Execution
- Define success criteria.
- Verify each implementation.

----------------------------------------------------------------------------

# Thailand Vehicle License Plate AI Extension

## Project
Detect Thai vehicle license plates from camera or PNG, recognize plate number and province, and support future RAG-based correction.

## Technology
Python 3.13
Poetry
FastAPI
OpenCV
Ultralytics YOLO
PaddleOCR
PyTorch
Albumentations
ChromaDB
SentenceTransformers
Jupyter Notebook

## Pipeline
Image
-> Detection
-> Perspective Correction
-> OCR
-> Post Processing
-> RAG Validation
-> JSON API
-> Web UI

## Coding Standards 
- Python type hints required.
- Google-style docstrings.
- Ruff, Black, MyPy, Pytest.
- Structured logging.
- No wildcard imports.

## Dataset Rules
datasets/
  raw/
  processed/
  augmented/
Raw data is immutable.

## Model Rules
Store release models only.
Track version, dataset, metrics, git commit.

## Experiment Tracking
Record:
- learning rate
- epochs
- optimizer
- mAP
- OCR accuracy
- latency
- FPS

## Performance Budget
Detection <25ms
OCR <40ms
RAG <15ms
Total <100ms

## Security
Validate uploaded images.
Never execute OCR output.

## AI Review Checklist
- Correct?
- Simple?
- Testable?
- Performant?
- Secure?
- Maintainable?

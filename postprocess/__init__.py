"""Post-processing of raw OCR output into canonical plate fields.

Pure functions only: no OCR engine, no I/O, no configuration. Phase 6 chains
this between :mod:`ocr` and the API response.
"""

"""Offline accuracy evaluation for the recognition pipeline.

Holds a small hand-verified ground-truth set (``plates.jsonl``) and the pure
scoring functions (``scoring``) that grade a prediction against it. The runner
that drives the real pipeline lives in ``docs/benchmark/bench_recognize_accuracy.py``.
"""

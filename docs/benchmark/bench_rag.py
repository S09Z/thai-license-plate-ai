"""Benchmark the shipped RAG validation stage on degraded province names.

Measures what the application actually runs — :func:`rag.validator.correct_province`
over the 77-province knowledge base — and reports recovery rate, safety and
latency against the ``<15ms`` RAG budget in ``CLAUDE.md``.

Unlike the OCR benchmark this needs no rendered image: the input is province
text degraded the way the recognizer was *observed* to degrade it in Phase 3.
That makes the numbers reproducible, but it also means they measure the
corrector, not the recognizer feeding it.

Run from the repository root::

    poetry run python docs/benchmark/bench_rag.py
"""

import argparse
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from postprocess.provinces import THAI_PROVINCES  # noqa: E402
from postprocess.thai import strip_thai_marks  # noqa: E402
from rag.validator import correct_province  # noqa: E402

RAG_BUDGET_MS = 15.0

# The real Phase 3 reads for a ชลบุรี plate, recorded in docs/benchmark/ocr-phase3.md.
OBSERVED_MISREADS = ("ชลบรดี", "ชลบร 9")

# Text that reaches this stage but names no province: plate numbers, a Latin
# misread, and empty rows. None of these may resolve to a province.
NON_PROVINCE = ("กข 1234", "1กข 2345", "VEZL", "", "   ")

# Degradations the recognizer was observed to apply. Truncation is deliberately
# excluded: it aliases เพชรบูรณ์ onto เพชรบุรี irrecoverably, which is a
# property of the province list, not of the corrector.
DEGRADATIONS: dict[str, Callable[[str], str]] = {
    "marks stripped": lambda name: strip_thai_marks(name),
    "marks + trailing junk": lambda name: strip_thai_marks(name) + "ดี",
    "marks + digit bleed": lambda name: strip_thai_marks(name) + " 9",
}


def measure_accuracy() -> None:
    """Report recovery and mis-attribution counts over every degradation."""
    print("Recovery over all 77 provinces")
    print(f"{'degradation':<24} {'recovered':>10} {'abstained':>10} {'WRONG':>7}")

    total_wrong = 0
    for label, degrade in DEGRADATIONS.items():
        recovered = abstained = wrong = 0
        for province in THAI_PROVINCES:
            match = correct_province(degrade(province))
            if match is None:
                abstained += 1
            elif match.province == province:
                recovered += 1
            else:
                wrong += 1
        total_wrong += wrong
        print(f"{label:<24} {recovered:>10} {abstained:>10} {wrong:>7}")

    print(f"\nMis-attributions across all degradations: {total_wrong}")

    print("\nObserved Phase 3 misreads")
    for text in OBSERVED_MISREADS:
        match = correct_province(text)
        result = (
            f"{match.province} @ {match.score:.3f}" if match else "None (abstained)"
        )
        print(f"  {text!r:<14} -> {result}")

    print("\nMust not resolve to any province")
    for text in NON_PROVINCE:
        match = correct_province(text)
        verdict = "ok" if match is None else f"FALSE POSITIVE -> {match.province}"
        print(f"  {text!r:<14} -> {verdict}")


def measure_latency(repeats: int) -> None:
    """Time the worst case: a candidate that misses the deterministic path.

    Args:
        repeats: How many timed corrections to run.
    """
    # An exact candidate returns from a dict lookup; a damaged one is scored
    # against all 77. Only the latter is worth timing.
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        correct_province("ชลบรดี")
        samples.append((time.perf_counter() - start) * 1000)

    samples.sort()
    median = statistics.median(samples)
    p95 = samples[int(len(samples) * 0.95)]

    print(f"\nLatency over {repeats} runs (full 77-way scan)")
    print(f"  median {median:.4f} ms")
    print(f"  p95    {p95:.4f} ms")
    verdict = "MET" if p95 < RAG_BUDGET_MS else "MISSED"
    print(f"  budget {RAG_BUDGET_MS:.1f} ms -> {verdict}")


def main() -> None:
    """Run the accuracy and latency benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=500, help="timed corrections to run"
    )
    arguments = parser.parse_args()

    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(
        f"Knowledge base: {len(THAI_PROVINCES)} provinces, "
        "no model, no vector store\n"
    )

    measure_accuracy()
    measure_latency(arguments.repeats)


if __name__ == "__main__":
    main()

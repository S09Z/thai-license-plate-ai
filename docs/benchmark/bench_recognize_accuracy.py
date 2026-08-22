"""Grade the full ``/recognize`` pipeline on the real-plate eval set.

Unlike ``bench_recognize.py`` (which renders a synthetic scene to bound
*latency*), this runs the actual detector + OCR + post-processing + RAG over the
hand-verified photographs in ``eval/plates.jsonl`` and reports *accuracy*:
plate exact-match rate, mean character error rate, and province accuracy, plus a
list of every miss so failures are diagnosable.

When several plates are detected in one image, the largest box is scored — the
eval set records one plate per image.

Run from the repository root::

    poetry run python docs/benchmark/bench_recognize_accuracy.py
    poetry run python docs/benchmark/bench_recognize_accuracy.py --limit 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.recognize_service import recognize_plates  # noqa: E402
from eval.scoring import PlateOutcome, aggregate, score_one  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_FILE = REPO_ROOT / "eval" / "plates.jsonl"
IMAGE_DIRS = [
    REPO_ROOT / "datasets/raw/thailand-license-plates-v1/valid/images",
    REPO_ROOT / "datasets/raw/thailand-license-plates-v1/test/images",
]


def _find_image(name: str) -> Path | None:
    """Locate an eval image by file name across the dataset splits."""
    for directory in IMAGE_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _largest_plate(data: bytes):  # type: ignore[no-untyped-def]
    """Return the largest recognized plate in an image, or ``None``."""
    response = recognize_plates(data, "image/jpeg")
    if not response.plates:
        return None
    return max(
        response.plates,
        key=lambda p: (p.box.x2 - p.box.x1) * (p.box.y2 - p.box.y1),
    )


def _load_eval(limit: int | None) -> list[dict[str, str]]:
    """Read the JSONL ground-truth set, optionally truncated."""
    rows = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def main() -> None:
    """Score the pipeline over the eval set and print a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Score only the first N images."
    )
    args = parser.parse_args()

    rows = _load_eval(args.limit)
    outcomes: list[PlateOutcome] = []
    missing: list[str] = []
    undetected: list[str] = []

    for row in rows:
        image_path = _find_image(row["img"])
        if image_path is None:
            missing.append(row["img"])
            continue
        plate = _largest_plate(image_path.read_bytes())
        if plate is None:
            undetected.append(row["img"])
            continue
        outcomes.append(
            score_one(
                img=row["img"],
                plate_pred=plate.plate_text,
                plate_truth=row["plate"],
                province_pred=plate.province,
                province_truth=row["province"],
            )
        )

    agg = aggregate(outcomes)
    print("\n=== Recognition accuracy (real plates) ===")
    print(f"scored images       : {agg.count}")
    print(f"plate exact-match   : {agg.plate_exact_rate:.1%}")
    print(f"mean char error rate: {agg.mean_cer:.3f}")
    print(f"province accuracy   : {agg.province_accuracy:.1%}")
    if missing:
        print(f"\nimages not found ({len(missing)}): {', '.join(missing)}")
    if undetected:
        print(f"\nno plate detected ({len(undetected)}): {', '.join(undetected)}")

    misses = [o for o in outcomes if not (o.plate_correct and o.province_correct)]
    if misses:
        print(f"\n--- misses ({len(misses)}) ---")
        for o in misses:
            plate_mark = "ok " if o.plate_correct else "PL "
            prov_mark = "ok" if o.province_correct else "PR"
            print(
                f"[{plate_mark}{prov_mark}] {o.img}\n"
                f"    plate    pred={o.plate_pred!r} truth={o.plate_truth!r} "
                f"cer={o.plate_cer:.2f}\n"
                f"    province pred={o.province_pred!r} truth={o.province_truth!r}"
            )


if __name__ == "__main__":
    main()

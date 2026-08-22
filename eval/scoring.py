"""Pure scoring for plate-recognition accuracy.

Every function here is deterministic and free of I/O or model calls, so the
metrics can be unit-tested without loading OCR. The runner supplies predictions
and ground truth; this module only compares them.

Plate strings are compared with whitespace removed, because the canonical
``"กข 1234"`` form and a spaceless ``"กข1234"`` name the same plate — the space
is presentation, not identity. Province strings are compared exactly, since the
knowledge base already canonicalizes them.
"""

from collections.abc import Sequence
from dataclasses import dataclass


def normalize_plate(text: str) -> str:
    """Strip all whitespace so spacing differences don't count as errors.

    Args:
        text: A plate string in any spacing, e.g. ``"กข 1234"``.

    Returns:
        The same characters with every whitespace run removed.
    """
    return "".join(text.split())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (insert/delete/substitute = 1).

    Args:
        a: First string.
        b: Second string.

    Returns:
        The minimum number of single-character edits turning ``a`` into ``b``.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def char_error_rate(pred: str, truth: str) -> float:
    """Character error rate between a prediction and ground truth.

    Both strings are normalized (whitespace removed) before comparison.

    Args:
        pred: Predicted plate string.
        truth: Ground-truth plate string.

    Returns:
        ``edit_distance / len(truth)``. Returns ``0.0`` when both are empty and
        ``1.0`` when only the truth is empty but a prediction was produced.
    """
    p = normalize_plate(pred)
    t = normalize_plate(truth)
    if not t:
        return 0.0 if not p else 1.0
    return edit_distance(p, t) / len(t)


def plate_exact_match(pred: str, truth: str) -> bool:
    """Whether two plate strings match after whitespace normalization.

    Args:
        pred: Predicted plate string.
        truth: Ground-truth plate string.

    Returns:
        ``True`` when the normalized strings are identical.
    """
    return normalize_plate(pred) == normalize_plate(truth)


@dataclass(frozen=True)
class PlateOutcome:
    """The graded result for one image.

    Attributes:
        img: The image file name the prediction came from.
        plate_pred: The predicted plate string.
        plate_truth: The ground-truth plate string.
        plate_correct: Whether the plate matched exactly (normalized).
        plate_cer: Character error rate for the plate.
        province_pred: The predicted province, or ``None`` when unresolved.
        province_truth: The ground-truth province.
        province_correct: Whether the predicted province equals the truth.
    """

    img: str
    plate_pred: str
    plate_truth: str
    plate_correct: bool
    plate_cer: float
    province_pred: str | None
    province_truth: str
    province_correct: bool


def score_one(
    img: str,
    plate_pred: str,
    plate_truth: str,
    province_pred: str | None,
    province_truth: str,
) -> PlateOutcome:
    """Grade a single prediction against its ground truth.

    Args:
        img: Image file name, carried through for the failure report.
        plate_pred: Predicted plate string.
        plate_truth: Ground-truth plate string.
        province_pred: Predicted province, or ``None`` when unresolved.
        province_truth: Ground-truth province.

    Returns:
        A :class:`PlateOutcome` with per-field correctness.
    """
    return PlateOutcome(
        img=img,
        plate_pred=plate_pred,
        plate_truth=plate_truth,
        plate_correct=plate_exact_match(plate_pred, plate_truth),
        plate_cer=char_error_rate(plate_pred, plate_truth),
        province_pred=province_pred,
        province_truth=province_truth,
        province_correct=province_pred == province_truth,
    )


@dataclass(frozen=True)
class Aggregate:
    """Corpus-level metrics over a set of :class:`PlateOutcome`.

    Attributes:
        count: Number of graded images.
        plate_exact_rate: Fraction with an exact plate match.
        mean_cer: Mean character error rate across images.
        province_accuracy: Fraction with the correct province.
    """

    count: int
    plate_exact_rate: float
    mean_cer: float
    province_accuracy: float


def aggregate(outcomes: Sequence[PlateOutcome]) -> Aggregate:
    """Reduce per-image outcomes to corpus metrics.

    Args:
        outcomes: The graded results.

    Returns:
        An :class:`Aggregate`. An empty input yields all-zero metrics rather
        than raising, so a runner can report "nothing scored" uniformly.
    """
    n = len(outcomes)
    if n == 0:
        return Aggregate(0, 0.0, 0.0, 0.0)
    return Aggregate(
        count=n,
        plate_exact_rate=sum(o.plate_correct for o in outcomes) / n,
        mean_cer=sum(o.plate_cer for o in outcomes) / n,
        province_accuracy=sum(o.province_correct for o in outcomes) / n,
    )

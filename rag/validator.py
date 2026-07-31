"""Correct a degraded province candidate against the province knowledge base.

The knowledge base is the 77 canonical province names. Phase 4 already resolves
candidates that survived recognition intact; this module handles the ones it
refuses, and refuses them in turn whenever the evidence is thin.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from postprocess.provinces import THAI_PROVINCES, match_province
from postprocess.thai import strip_thai_marks
from rag.similarity import similarity_ratio

# A candidate must reach this similarity before any province is considered.
# Text that is not a province name at all falls far below it: plate numbers and
# Latin misreads score around 0.17 against the list.
MIN_SCORE = 0.6

# The best province must also beat the runner-up by this margin. This, not the
# score floor, is what makes fuzzy matching safe. เพชรบุรี and เพชรบูรณ์ differ
# only in marks the recognizer drops, so a damaged candidate can tie against
# both; without a margin the tie is broken arbitrarily and the answer is
# confidently wrong. Measured over 308 degraded candidates, 0.15 is the
# smallest margin that never returns the wrong province.
MIN_MARGIN = 0.15

_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class ProvinceMatch:
    """A province the knowledge base vouches for.

    Attributes:
        province: The canonical province name.
        score: Similarity to the candidate, in ``[0, 1]``.
        is_exact: Whether the deterministic matcher resolved it, meaning no
            fuzzy correction was applied.
    """

    province: str
    score: float
    is_exact: bool


# Built once at import: each province beside the mark-free skeleton that a
# damaged candidate is compared against.
_SKELETONS: tuple[tuple[str, str], ...] = tuple(
    (province, strip_thai_marks(province)) for province in THAI_PROVINCES
)


def _lookup_key(text: str) -> str:
    """Reduce a candidate to the form compared against province skeletons.

    Args:
        text: A province candidate, exactly as recognized.

    Returns:
        The text without whitespace or Thai combining marks.
    """
    return strip_thai_marks(_WHITESPACE_PATTERN.sub("", text))


def correct_province(candidate: str) -> ProvinceMatch | None:
    """Map a recognized province candidate onto a real province.

    The deterministic matcher runs first, so an undamaged candidate never
    reaches fuzzy scoring. Fuzzy retrieval then handles what it rejected, and
    returns a province only when one wins clearly — an ambiguous candidate
    yields ``None``, because a confidently wrong province is worse than an
    admitted failure.

    Args:
        candidate: A province candidate, exactly as recognized.

    Returns:
        The matched province, or ``None`` when nothing matches confidently.
    """
    exact = match_province(candidate)
    if exact is not None:
        return ProvinceMatch(province=exact, score=1.0, is_exact=True)

    key = _lookup_key(candidate)
    if not key:
        return None

    ranked = sorted(
        (
            (similarity_ratio(key, skeleton), province)
            for province, skeleton in _SKELETONS
        ),
        reverse=True,
    )
    best_score, best_province = ranked[0]
    runner_up_score = ranked[1][0]

    if best_score < MIN_SCORE or best_score - runner_up_score < MIN_MARGIN:
        return None

    return ProvinceMatch(province=best_province, score=best_score, is_exact=False)


def resolve_province(candidates: Sequence[str]) -> ProvinceMatch | None:
    """Pick the best province among the candidates of one plate reading.

    Args:
        candidates: Province candidates, as :mod:`ocr.reading` reports them —
            every row below the plate number, none of them yet interpreted.

    Returns:
        The highest-scoring confident match, or ``None`` when there is none.
    """
    matches = [
        match for match in map(correct_province, candidates) if match is not None
    ]
    if not matches:
        return None

    return max(matches, key=lambda match: match.score)

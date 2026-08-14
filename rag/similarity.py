"""Character-level similarity between short Thai strings.

The recognizer damages province names one character at a time: it drops
combining marks, appends a hallucinated syllable, or loses a trailing letter.
Edit distance models that damage directly, which is why retrieval in this
package is lexical. A dense embedding would answer a question nobody asked —
whether two names *mean* the same thing — at a cost the latency budget in
``CLAUDE.md`` cannot absorb.
"""


def levenshtein_distance(left: str, right: str) -> int:
    """Count the single-character edits that turn one string into the other.

    Args:
        left: The first string.
        right: The second string.

    Returns:
        The number of insertions, deletions and substitutions required.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    # Only the previous row of the edit matrix is ever read, so keep one row.
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current

    return previous[-1]


def similarity_ratio(left: str, right: str) -> float:
    """Score how alike two strings are, on a 0.0 to 1.0 scale.

    Normalizing by the longer string keeps a one-character error costly on a
    short province name and cheap on a long one, which matches how much a
    single character actually narrows the candidates.

    Args:
        left: The first string.
        right: The second string.

    Returns:
        ``1.0`` for identical strings, ``0.0`` when nothing is shared.
    """
    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0

    return 1.0 - levenshtein_distance(left, right) / longest

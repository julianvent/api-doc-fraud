from __future__ import annotations

from typing import List

from ...models import Word


def reconstruct_text(words: List[Word], line_threshold: float) -> str:
    """Rebuild coherent text from spatially-positioned words.

    Groups words whose vertical centers are within `line_threshold` (in
    normalized 0-1 coordinates) into the same line, then orders each line
    left-to-right.
    """
    if not words:
        return ""

    sorted_words = sorted(words, key=_y_center)
    lines: List[List[Word]] = []
    current: List[Word] = [sorted_words[0]]

    for word in sorted_words[1:]:
        if abs(_y_center(word) - _y_center(current[-1])) <= line_threshold:
            current.append(word)
        else:
            lines.append(sorted(current, key=_x_start))
            current = [word]
    lines.append(sorted(current, key=_x_start))

    return "\n".join(" ".join(w.text for w in line) for line in lines)


def _y_center(w: Word) -> float:
    (_, y1), (_, y2) = w.geometry
    return (y1 + y2) / 2


def _x_start(w: Word) -> float:
    (x1, _), _ = w.geometry
    return x1

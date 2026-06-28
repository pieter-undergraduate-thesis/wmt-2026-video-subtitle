"""Subtitle-format constraints + cue splitting.

A perfect translation that doesn't fit on screen still fails, so format QC is
separate from translation quality.
"""
from __future__ import annotations

from .subtitle_io import Cue

# ponytail: Netflix-style industry defaults (Karamitroglou-derived). PLACEHOLDER.
# Swap to WMT26's official spec when it ships with the test data on Jul 1 2026.
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CPS = 17.0


def check_constraints(
    cue_text: str,
    duration_s: float,
    max_chars_per_line: int = MAX_CHARS_PER_LINE,
    max_lines: int = MAX_LINES,
    max_cps: float = MAX_CPS,
) -> list[str]:
    """Return a list of violation strings (empty list = compliant)."""
    violations: list[str] = []
    lines = cue_text.split("\n")
    if len(lines) > max_lines:
        violations.append(f"too many lines: {len(lines)}")
    for ln in lines:
        if len(ln) > max_chars_per_line:
            violations.append(f"line too long: {len(ln)} chars")
    cps = len(cue_text.replace("\n", "")) / max(duration_s, 0.01)
    if cps > max_cps:
        violations.append(f"reading speed too high: {cps:.1f} CPS")
    return violations


def split_cue_if_needed(
    cue: Cue,
    max_chars_per_line: int = MAX_CHARS_PER_LINE,
    max_lines: int = MAX_LINES,
) -> list[Cue]:
    """Split an over-long cue into two timed cues rather than truncating.

    Splitting preserves meaning; truncation loses it. Original start stays on
    the first half, original end on the second, with no gap/overlap at the
    split point (matches professional subtitle tooling).

    ponytail: naive midpoint + nearest word boundary. Known ceiling — real
    splitting should prefer grammatical boundaries (after punctuation, before
    conjunctions). Upgrade here when the word-alignment module lands.
    """
    if not check_constraints(cue.text, cue.duration_s, max_chars_per_line, max_lines):
        return [cue]

    mid = len(cue.text) // 2
    space = cue.text.rfind(" ", 0, mid)
    split_point = space if space > 0 else mid
    first_half = cue.text[:split_point].strip()
    second_half = cue.text[split_point:].strip()
    midpoint_ms = (cue.start_ms + cue.end_ms) // 2
    return [
        Cue(cue.index, cue.start_ms, midpoint_ms, first_half),
        Cue(cue.index + 1, midpoint_ms, cue.end_ms, second_half),
    ]


def violation_rate(cues: list[Cue]) -> float:
    """Fraction of cues with >=1 constraint violation. Phase A/B eval metric."""
    if not cues:
        return 0.0
    bad = sum(1 for c in cues if check_constraints(c.text, c.duration_s))
    return bad / len(cues)

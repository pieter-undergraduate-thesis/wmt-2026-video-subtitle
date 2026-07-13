"""Context assembly for sequential subtitle translation.

Two kinds of context, kept separate:
- synopsis-level (genre/plot summary, optionally from Zahra's SLM layer)
- short-term: a sliding window of recently *translated* cues, for consistency
  over a sequence (challenge #1: long-sentence / cross-cue coherence).
"""
from __future__ import annotations

from collections import deque
from typing import Callable


class RecentContext:
    """Sliding window of the last N (source, translation) cue pairs.

    Pairs feed few-shot ICL: the video's own already-translated cues become the
    examples, so terminology/style stay consistent down the episode.
    """

    def __init__(self, window: int = 4):
        self._buf: deque[tuple[str, str]] = deque(maxlen=window)

    def add(self, source_text: str, translated_text: str) -> None:
        self._buf.append((source_text, translated_text))

    def pairs(self) -> list[tuple[str, str]]:
        return list(self._buf)

    def block(self) -> str:
        """Prompt fragment, or '' if empty."""
        if not self._buf:
            return ""
        joined = "\n".join(f"{s} -> {t}" for s, t in self._buf)
        return (
            "Recent translated context (for consistency only, do not "
            f"re-translate):\n{joined}"
        )


def build_context(
    synopsis: str | None,
    recent: RecentContext | None,
    *,
    prior_src: list[str] | None = None,
    synopsis_provider: Callable[[], str] | None = None,
    timeout_s: float = 5.0,
) -> str | None:
    """Combine synopsis + prior source lines + recent-cue context into one block.

    Fallback switch: if a synopsis_provider (e.g. Zahra's SLM layer) is given
    but errors or hangs, fall through to whatever static context we have rather
    than blocking the whole pipeline. A None return makes translate_cue use the
    plain no-context template.

    ponytail: timeout via a thread+join, not async — single call per cue, the
    thread overhead is irrelevant next to a 7B forward pass.
    """
    parts: list[str] = []

    if synopsis_provider is not None:
        try:
            synopsis = _call_with_timeout(synopsis_provider, timeout_s) or synopsis
        except Exception:
            pass  # keep the static synopsis (possibly None); never block

    if synopsis:
        parts.append(synopsis.strip())
    if prior_src:
        joined = "\n".join(prior_src)
        parts.append(
            f"Prior source lines (context only, do not translate):\n{joined}"
        )
    if recent is not None:
        rb = recent.block()
        if rb:
            parts.append(rb)

    return "\n\n".join(parts) if parts else None


def _call_with_timeout(fn: Callable[[], str], timeout_s: float) -> str | None:
    import threading

    result: list[str | None] = [None]

    def _run():
        result[0] = fn()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError("synopsis_provider timed out")
    return result[0]

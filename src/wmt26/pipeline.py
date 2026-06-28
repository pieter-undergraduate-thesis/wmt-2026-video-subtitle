"""End-to-end Phase C pipeline: srt -> translate -> constrain -> srt.

flow per cue:
  synopsis + recent-cue context  ->  prompt assembly
                                  ->  Hy-MT2 inference (vLLM)
                                  ->  constraint post-processor (split if needed)
  then: re-index all cues and write the SRT.
"""
from __future__ import annotations

from typing import Callable

from . import constraints, translate
from .context import RecentContext, build_context
from .subtitle_io import Cue, parse_srt, write_srt


def translate_cues(
    cues: list[Cue],
    target: str,
    *,
    synopsis: str | None = None,
    synopsis_provider: Callable[[], str] | None = None,
    window: int = 4,
    model: str = translate.DEFAULT_MODEL,
    base_url: str = translate.DEFAULT_BASE_URL,
    translate_fn: Callable[..., str] = translate.translate_cue,
) -> list[Cue]:
    """Translate cues sequentially with rolling context, then split + re-index.

    `translate_fn` is injectable so tests can run the whole pipeline without a
    live vLLM server.
    """
    recent = RecentContext(window=window)
    out: list[Cue] = []

    for cue in cues:
        ctx = build_context(synopsis, recent, synopsis_provider=synopsis_provider)
        translated = translate_fn(cue.text, target, ctx, model=model, base_url=base_url)
        recent.add(translated)
        new_cue = Cue(cue.index, cue.start_ms, cue.end_ms, translated)
        out.extend(constraints.split_cue_if_needed(new_cue))

    # Re-index after splits so the SRT numbering is contiguous.
    for i, c in enumerate(out, start=1):
        c.index = i
    return out


def translate_srt(
    in_path: str,
    out_path: str,
    target: str,
    *,
    synopsis: str | None = None,
    synopsis_provider: Callable[[], str] | None = None,
    model: str = translate.DEFAULT_MODEL,
    base_url: str = translate.DEFAULT_BASE_URL,
) -> list[Cue]:
    cues = parse_srt(in_path)
    out = translate_cues(
        cues,
        target,
        synopsis=synopsis,
        synopsis_provider=synopsis_provider,
        model=model,
        base_url=base_url,
    )
    write_srt(out, out_path)
    return out

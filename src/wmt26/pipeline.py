"""End-to-end Phase C pipeline: srt -> translate -> srt.

flow per cue:
  synopsis context + recent-cue ICL examples  ->  prompt assembly
                                              ->  Hy-MT2 inference (vLLM)
  cues stay 1:1 with the source (no re-segmentation); then write the SRT.
"""
from __future__ import annotations

from typing import Callable

from . import translate
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

    # Synopsis as background; recent cue pairs as few-shot ICL examples. The
    # synopsis is fixed for the whole video, so resolve it once.
    ctx = build_context(synopsis, None, synopsis_provider=synopsis_provider)

    for i, cue in enumerate(cues, start=1):
        translated = translate_fn(
            cue.text, target, ctx, examples=recent.pairs(), model=model, base_url=base_url
        )
        recent.add(cue.text, translated)
        out.append(Cue(i, cue.start_ms, cue.end_ms, translated))

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

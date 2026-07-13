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
    src_window: int = 6,
    register: str | None = None,
    model: str = translate.DEFAULT_MODEL,
    base_url: str = translate.DEFAULT_BASE_URL,
    translate_fn: Callable[..., str] = translate.translate_cue,
) -> list[Cue]:
    """Translate cues sequentially with rolling context, then split + re-index.

    Context per cue = synopsis + the prior `src_window` source lines + the
    rolling window of already-translated cues. `register` (a formality note) is
    forwarded to `translate_fn`. `translate_fn` is injectable so tests can run
    the whole pipeline without a live vLLM server.
    """
    recent = RecentContext(window=window)
    out: list[Cue] = []

    for i, cue in enumerate(cues):
        prior_src = [c.text for c in cues[max(0, i - src_window):i]] or None
        ctx = build_context(
            synopsis, recent, prior_src=prior_src, synopsis_provider=synopsis_provider
        )
        translated = translate_fn(
            cue.text, target, ctx, examples=recent.pairs(), model=model, base_url=base_url, register=register
        )
        recent.add(translated)
        new_cue = Cue(cue.index, cue.start_ms, cue.end_ms, translated)
        out.extend(constraints.split_cue_if_needed(new_cue))

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

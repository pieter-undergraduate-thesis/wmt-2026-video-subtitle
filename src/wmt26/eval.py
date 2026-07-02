"""Evaluation: translation quality + subtitle-format QC.

COMET is the primary number (correlates best with human judgment on loose,
idiom-heavy subtitle text); BLEU/chrF are secondary sanity checks.
"""
from __future__ import annotations

from dataclasses import dataclass

from .constraints import violation_rate
from .subtitle_io import Cue


@dataclass
class Scores:
    bleu: float
    chrf: float
    comet: float | None
    violation_rate: float


def score_bleu_chrf(hyps: list[str], refs: list[str]) -> tuple[float, float]:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return bleu, chrf


def score_comet(
    srcs: list[str], hyps: list[str], refs: list[str], model: str = "Unbabel/wmt22-comet-da"
) -> float:
    """Lazy import — COMET pulls torch and downloads a model on first use."""
    from comet import download_model, load_from_checkpoint

    ckpt = load_from_checkpoint(download_model(model))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    return ckpt.predict(data, batch_size=8, progress_bar=False).system_score


def score_comet_qe(
    srcs: list[str], hyps: list[str], model: str = "Unbabel/wmt22-cometkiwi-da"
) -> float:
    """Reference-free QE (CometKiwi) — scores src+mt only, so it runs on the blind test set."""
    from comet import download_model, load_from_checkpoint

    ckpt = load_from_checkpoint(download_model(model))
    data = [{"src": s, "mt": h} for s, h in zip(srcs, hyps)]
    return ckpt.predict(data, batch_size=8, progress_bar=False).system_score


def evaluate(
    srcs: list[str],
    hyp_cues: list[Cue],
    refs: list[str],
    *,
    with_comet: bool = True,
) -> Scores:
    hyps = [c.text for c in hyp_cues]
    bleu, chrf = score_bleu_chrf(hyps, refs)
    comet = score_comet(srcs, hyps, refs) if with_comet else None
    return Scores(bleu=bleu, chrf=chrf, comet=comet, violation_rate=violation_rate(hyp_cues))

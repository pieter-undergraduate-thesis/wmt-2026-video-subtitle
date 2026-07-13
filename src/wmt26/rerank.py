"""QE-rerank: pick the best of N candidates with CometKiwi (reference-free).

Stays in the parameter budget (no second generator model). Off-target filtering
first, because docs/research.md flags CometKiwi being fooled by code-switching
as the #1 zh->id risk — a candidate with leaked Chinese can still score high.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

# CJK Unified Ideographs; a non-Chinese target containing these is off-target.
_HAN_RE = re.compile(r"[一-鿿]")


@lru_cache(maxsize=2)
def _load_kiwi(model: str):
    """Load the CometKiwi checkpoint once and keep it hot across cues.

    ponytail: lru_cache is the whole cache. eval.score_comet_qe reloads the
    checkpoint every call, which is O(N*cues) fatal here. Swap for an explicit
    scorer object only if we ever need >1 QE model resident at once.
    """
    from comet import download_model, load_from_checkpoint

    return load_from_checkpoint(download_model(model))


def qe_scores(
    srcs: list[str], mts: list[str], model: str = "Unbabel/wmt22-cometkiwi-da"
) -> list[float]:
    """Per-segment CometKiwi scores for aligned (src, mt) pairs."""
    ckpt = _load_kiwi(model)
    data = [{"src": s, "mt": m} for s, m in zip(srcs, mts)]
    return list(ckpt.predict(data, batch_size=8, progress_bar=False).scores)


def offtarget_filter(cands: list[str], target: str) -> list[str]:
    """Drop candidates with leaked Chinese for non-Chinese targets.

    No-op for Chinese targets (zh-TW legitimately contains Han). If filtering
    would empty the set, keep all — a bad candidate beats no candidate.
    """
    if str(target).startswith("zh"):
        return cands
    kept = [c for c in cands if not _HAN_RE.search(c)]
    return kept or cands


def pick_best(
    src: str,
    cands: list[str],
    target: str,
    *,
    score_fn: Callable[[list[str], list[str]], list[float]] = qe_scores,
) -> str:
    """Off-target filter -> QE score -> argmax. `score_fn` injectable for tests."""
    cands = offtarget_filter(cands, target)
    if len(cands) == 1:
        return cands[0]
    scores = score_fn([src] * len(cands), cands)
    return max(zip(scores, cands), key=lambda sc: sc[0])[1]

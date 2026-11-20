"""Post-editing: one refine pass, applied selectively.

Two entry points:
- `postedit` — TEaR/Proofreader style single refinement of one draft.
- `postedit_cues` — the retrieval-augmented pass over a whole episode
  (docs/postedit-research.md): flag only the cues whose source carries a term-DB
  idiom/honorific/wuxia term, edit those, and accept the edit only if it beats
  the draft on the checks that are free and deterministic.

ONE iteration only — docs/research.md warns over-refinement degrades dev COMET,
so there is deliberately no loop here.

The accept gate is what makes this safe to ship: Hy-MT2 is an MT model, not an
instruction-tuned editor, so "return it unchanged" may not land. A rejected edit
falls back to the draft, which floors the worst case at baseline parity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from . import prompts, rerank, translate, termdb
from .constraints import check_constraints
from .subtitle_io import Cue

if TYPE_CHECKING:
    from openai import OpenAI

    from .termdb import Entry, TermDB


@dataclass
class PostEditStats:
    """Gain-to-edit / over-edit instrumentation (research §4's ablation numbers)."""

    cues: int = 0
    flagged: int = 0    # source line matched >=1 DB entry -> sent to the model
    edited: int = 0     # model came back with something different from the draft
    accepted: int = 0   # edit survived the gate and made it into the output


def postedit(
    src: str,
    hyp: str,
    target: str,
    *,
    register: str | None = None,
    terms: "list[Entry] | None" = None,
    client: "OpenAI | None" = None,
    model: str = translate.DEFAULT_MODEL,
    complete_fn: Callable[..., str] = translate._complete,
) -> str:
    """Return one refined translation. `complete_fn` injectable for CPU tests."""
    client = client or translate._client()
    prompt = prompts.postedit_prompt(src, hyp, target, register=register, terms=terms)
    return complete_fn(client, prompt, model)


def _accept(
    draft: str, edit: str, entries: "list[Entry]", target: str, duration_s: float
) -> bool:
    """Keep the edit only if it doesn't regress on any free, deterministic check.

    QE is deliberately not here: research finding 3 reports reference-free QE
    correlating near-zero/negative *on idiom lines* — exactly the lines this
    gate judges. It's opt-in via postedit_cues(qe_gate=True) instead.
    """
    if rerank.is_offtarget(edit, target):
        return False  # leaked Han into a non-Chinese target
    if termdb.hits(edit, entries, target) < termdb.hits(draft, entries, target):
        return False  # dropped a required rendering the draft already had
    if len(check_constraints(edit, duration_s)) > len(
        check_constraints(draft, duration_s)
    ):
        return False  # more CPL/line/CPS violations than we started with
    return True


def _qe_filter(
    src_cues: list[Cue],
    hyp_cues: list[Cue],
    pending: list[tuple[int, str]],
    score_fn: Callable[..., list[float]],
) -> list[tuple[int, str]]:
    """Drop edits scoring below their draft. One batched call for the episode.

    Batched because rerank._load_kiwi caches the *checkpoint*, not the forward
    pass — scoring two strings at a time per cue would still be N round trips.
    """
    srcs = [src_cues[i].text for i, _ in pending]
    scores = score_fn(
        srcs + srcs,
        [hyp_cues[i].text for i, _ in pending] + [e for _, e in pending],
    )
    n = len(pending)
    return [
        p for p, drafted, edited in zip(pending, scores[:n], scores[n:])
        if edited >= drafted
    ]


def postedit_cues(
    src_cues: list[Cue],
    hyp_cues: list[Cue],
    target: str,
    db: "TermDB",
    *,
    register: str | None = None,
    qe_gate: bool = False,
    client: "OpenAI | None" = None,
    model: str = translate.DEFAULT_MODEL,
    complete_fn: Callable[..., str] = translate._complete,
    score_fn: Callable[..., list[float]] = rerank.qe_scores,
) -> tuple[list[Cue], PostEditStats]:
    """Selectively post-edit `hyp_cues` using terms matched in `src_cues`.

    Cues whose source matches no DB entry are passed through untouched and never
    reach the model — that selectivity is the whole point (APE overcorrects;
    research §2.1's gain-to-edit heuristic). Output keeps the draft's index and
    timing, 1:1.

    `complete_fn` / `score_fn` injectable so the orchestration tests run on CPU.
    """
    if len(src_cues) != len(hyp_cues):
        raise ValueError(
            f"cue count mismatch: {len(src_cues)} source vs {len(hyp_cues)} draft. "
            "Positional pairing needs a 1:1 draft (pipeline.translate_cues keeps "
            "cues 1:1; zero_shot_translate.py splits them). Re-run the draft "
            "through run_pipeline.py, or align by timestamp overlap "
            "(see run_benchmark_eval.py::align_to_src)."
        )

    client = client or translate._client()
    stats = PostEditStats(cues=len(hyp_cues))
    pending: list[tuple[int, str]] = []

    for i, (s, h) in enumerate(zip(src_cues, hyp_cues)):
        entries = db.match(s.text)
        if not entries:
            continue
        stats.flagged += 1
        edit = (
            postedit(
                s.text, h.text, target,
                register=register, terms=entries,
                client=client, model=model, complete_fn=complete_fn,
            )
            or ""
        ).strip()
        if not edit or edit == h.text:
            continue
        stats.edited += 1
        if _accept(h.text, edit, entries, target, h.duration_s):
            pending.append((i, edit))

    if qe_gate and pending:
        pending = _qe_filter(src_cues, hyp_cues, pending, score_fn)

    out = [Cue(h.index, h.start_ms, h.end_ms, h.text) for h in hyp_cues]
    for i, edit in pending:
        out[i].text = edit
    stats.accepted = len(pending)
    return out, stats

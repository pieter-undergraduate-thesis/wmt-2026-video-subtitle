"""N-candidate generation — the Chimera recipe: one prompt, varied temperature.

The highest-ROI training-free win (see docs/research.md): generate several
diverse translations of the same cue, then let rerank.pick_best keep the best.
Reuses translate._complete (one shared client) and prompts.advanced_prompt.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from . import prompts, translate

if TYPE_CHECKING:
    from openai import OpenAI

# Temperature spread from the Chimera recipe; widen if candidates aren't diverse.
DEFAULT_TEMPS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)


def generate_candidates(
    text: str,
    target: str,
    *,
    context: str | None = None,
    glossary: dict[str, str] | None = None,
    examples: list[tuple[str, str]] | None = None,
    register: str | None = None,
    n: int = 6,
    temps: tuple[float, ...] = DEFAULT_TEMPS,
    client: "OpenAI | None" = None,
    model: str = translate.DEFAULT_MODEL,
    complete_fn: Callable[..., str] = translate._complete,
) -> list[str]:
    """Return up to `n` distinct candidate translations, order preserved.

    `complete_fn(client, prompt, model, temperature)` is injectable so the
    orchestration can be tested without a live server. Identical outputs across
    temperatures are deduped — fewer than `n` results is expected and fine.
    """
    client = client or translate._client()
    prompt = prompts.advanced_prompt(
        text, target, context=context, glossary=glossary, examples=examples, register=register
    )
    cands = [complete_fn(client, prompt, model, t) for t in temps[:n]]
    return list(dict.fromkeys(c for c in cands if c))  # order-preserving dedup, drop empties

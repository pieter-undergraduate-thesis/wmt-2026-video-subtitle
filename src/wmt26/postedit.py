"""Single post-edit / proofreader pass over a chosen translation.

TEaR/Proofreader style: one refinement for adequacy, register, and brevity.
ONE iteration only — docs/research.md warns over-refinement degrades dev COMET,
so there is deliberately no loop here. Opt-in via the pipeline's --postedit flag.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from . import prompts, translate

if TYPE_CHECKING:
    from openai import OpenAI


def postedit(
    src: str,
    hyp: str,
    target: str,
    *,
    register: str | None = None,
    client: "OpenAI | None" = None,
    model: str = translate.DEFAULT_MODEL,
    complete_fn: Callable[..., str] = translate._complete,
) -> str:
    """Return one refined translation. `complete_fn` injectable for CPU tests."""
    client = client or translate._client()
    prompt = prompts.postedit_prompt(src, hyp, target, register=register)
    return complete_fn(client, prompt, model)

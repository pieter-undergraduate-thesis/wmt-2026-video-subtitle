"""Episode glossary: one LLM pass to fix character/sect/title renderings.

Consistency across an episode is the dominant quality lever for xianxia/wuxia
dramas and for zh->id (docs/research.md). The glossary is built once per
(episode, target), injected per window via prompts.advanced_prompt(glossary=...),
and `enforce` is the final safety net for any source term that leaked untranslated.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from . import prompts, translate
from .subtitle_io import Cue

if TYPE_CHECKING:
    from openai import OpenAI


def build_glossary(
    cues: list[Cue],
    synopsis: str | None,
    target: str,
    *,
    client: "OpenAI | None" = None,
    model: str = translate.DEFAULT_MODEL,
    complete_fn: Callable[..., str] = translate._complete,
) -> dict[str, str]:
    """Extract `源词 -> rendering` pairs from the whole episode. One LLM call.

    Malformed lines are skipped. `complete_fn` injectable for CPU tests.
    """
    client = client or translate._client()
    episode = "\n".join(c.text for c in cues)
    prompt = prompts.glossary_prompt(episode, target, synopsis)
    raw = complete_fn(client, prompt, model)
    return _parse_glossary(raw)


def _parse_glossary(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if "->" not in line:
            continue
        src, _, tgt = line.partition("->")
        src, tgt = src.strip(), tgt.strip()
        if src and tgt:
            out[src] = tgt
    return out


def enforce(text: str, glossary: dict[str, str]) -> str:
    """Replace any source term that leaked untranslated with its agreed rendering.

    Verified exact find/replace — safe and deterministic (fixes the zh->id
    code-switching failure where a name stays in Chinese).
    """
    for src, tgt in glossary.items():
        text = text.replace(src, tgt)
    return text

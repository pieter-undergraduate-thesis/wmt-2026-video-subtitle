"""Hy-MT2 inference via vLLM's OpenAI-compatible server.

Start the server with scripts/serve.sh, then call translate_cue / translate_batch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import prompts

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_MODEL = "tencent/Hy-MT2-1.8B"
DEFAULT_BASE_URL = "http://localhost:8000/v1"

# Greedy decoding — deterministic and stronger for MT than the card's sampling.
SAMPLING = dict(temperature=0.0, top_p=0.6, max_tokens=512)
EXTRA_BODY = dict(top_k=20, repetition_penalty=1.05)


def _client(base_url: str = DEFAULT_BASE_URL) -> "OpenAI":
    from openai import OpenAI  # lazy: keep openai off the import path for io/eval/tests

    return OpenAI(base_url=base_url, api_key="not-needed")


def _complete(
    client: "OpenAI", prompt: str, model: str, temperature: float | None = None
) -> str:
    sampling = {**SAMPLING, "temperature": temperature} if temperature is not None else SAMPLING
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body=EXTRA_BODY,
        **sampling,
    )
    return resp.choices[0].message.content.strip()


def translate_cue(
    text: str,
    target: str,
    context: str | None = None,
    *,
    examples: list[tuple[str, str]] | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    client: OpenAI | None = None,
    temperature: float | None = None,
    register: str | None = None,
) -> str:
    """Translate one cue's text.

    Pass `context` for the synopsis-aware prompt, `examples` for few-shot ICL.
    """
    client = client or _client(base_url)
    if register:
        # combined builder composes context + examples + register; keeps the
        # plain default/context/few-shot prompts byte-stable when register unset.
        prompt = prompts.advanced_prompt(
            text, target, context=context, examples=examples, register=register
        )
    elif examples:
        prompt = prompts.few_shot_prompt(text, target, examples, context)
    elif context:
        prompt = prompts.context_prompt(text, target, context)
    else:
        prompt = prompts.default_prompt(text, target)
    return _complete(client, prompt, model, temperature)


def translate_batch(
    texts: list[str],
    target: str,
    contexts: list[str | None] | None = None,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> list[str]:
    """Sequential calls; vLLM's continuous batching overlaps them server-side.

    ponytail: one client, plain loop. Add asyncio fan-out only if throughput
    on the GPU measurably bottlenecks here.
    """
    client = _client(base_url)
    contexts = contexts or [None] * len(texts)
    return [
        translate_cue(t, target, c, model=model, client=client)
        for t, c in zip(texts, contexts)
    ]

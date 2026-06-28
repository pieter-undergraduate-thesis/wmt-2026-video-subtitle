"""Hy-MT2 inference via vLLM's OpenAI-compatible server.

Start the server with scripts/serve.sh, then call translate_cue / translate_batch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import prompts

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_MODEL = "tencent/Hy-MT2-7B"
DEFAULT_BASE_URL = "http://localhost:8000/v1"

# Sampling params from the Hy-MT2 model card.
SAMPLING = dict(temperature=0.7, top_p=0.6, max_tokens=512)
EXTRA_BODY = dict(top_k=20, repetition_penalty=1.05)


def _client(base_url: str = DEFAULT_BASE_URL) -> "OpenAI":
    from openai import OpenAI  # lazy: keep openai off the import path for io/eval/tests

    return OpenAI(base_url=base_url, api_key="not-needed")


def _complete(client: "OpenAI", prompt: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body=EXTRA_BODY,
        **SAMPLING,
    )
    return resp.choices[0].message.content.strip()


def translate_cue(
    text: str,
    target: str,
    context: str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    client: OpenAI | None = None,
) -> str:
    """Translate one cue's text. Pass `context` to use the synopsis-aware prompt."""
    client = client or _client(base_url)
    prompt = (
        prompts.context_prompt(text, target, context)
        if context
        else prompts.default_prompt(text, target)
    )
    return _complete(client, prompt, model)


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

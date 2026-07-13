"""Extra evaluation metrics beyond eval.py: chrF++, XCOMET-XXL, GEMBA.

eval.py already covers BLEU, chrF, COMET, and CometKiwi (reference-free QE) —
this module only adds what's missing, and reuses those where the benchmark
script needs them. Heavy deps (comet, openai) are lazy-imported, matching the
rest of the package so io/tests stay importable CPU-side.
"""
from __future__ import annotations

import re


def score_chrf_pp(hyps: list[str], refs: list[str]) -> float:
    """chrF++ = chrF with word bigrams (word_order=2).

    eval.py::score_bleu_chrf uses default chrF (word_order=0); this is the ++
    variant, which correlates better with human judgment on subtitle text.
    """
    import sacrebleu

    return sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score


def score_xcomet(
    srcs: list[str],
    hyps: list[str],
    refs: list[str],
    model: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
) -> dict:
    """Reference-based XCOMET-XXL. Returns {'system', 'segments'}.

    Gated ~10.7B HF model — needs a GPU and an accepted HF licence + token.
    """
    from comet import download_model, load_from_checkpoint

    ckpt = load_from_checkpoint(download_model(model))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    out = ckpt.predict(data, batch_size=batch_size, progress_bar=False)
    return {"system": out.system_score, "segments": list(out.scores)}


# --- GEMBA (LLM-as-judge, reference-free DA) ---------------------------------

_GEMBA_PROMPT = (
    "Score the following machine translation from {src_lang} to {tgt_lang} on a "
    "continuous 0-100 scale, where 0 is worthless and 100 is a perfect, fluent, "
    "fully accurate translation. Consider adequacy and fluency. Respond with ONLY "
    "the integer score, nothing else.\n\n"
    "{src_lang} source: {src}\n"
    "{tgt_lang} translation: {hyp}\n\n"
    "Score (0-100):"
)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _gemba_client(backend: str, base_url: str | None):
    """Return (openai_client, model_default) for the chosen backend."""
    if backend == "vllm":
        from . import translate

        from openai import OpenAI

        return OpenAI(base_url=base_url or translate.DEFAULT_BASE_URL, api_key="not-needed")
    if backend == "openai":
        from openai import OpenAI  # reads OPENAI_API_KEY from env

        return OpenAI(base_url=base_url) if base_url else OpenAI()
    raise ValueError(f"unknown gemba backend: {backend!r}")


def _parse_score(text: str) -> float | None:
    m = _NUM_RE.search(text)
    if not m:
        return None
    return max(0.0, min(100.0, float(m.group())))


def score_gemba(
    srcs: list[str],
    hyps: list[str],
    *,
    src_lang: str = "Chinese",
    tgt_lang: str = "English",
    backend: str = "openai",
    model: str = "gpt-4o",
    base_url: str | None = None,
) -> dict:
    """GEMBA-DA style 0-100 QE via an LLM judge. Returns {'system', 'segments'}.

    One LLM call per (src, hyp) pair — cap the pair count upstream for cost.
    Unparseable replies are dropped from the average and recorded as None.
    """
    client = _gemba_client(backend, base_url)
    segs: list[float | None] = []
    for s, h in zip(srcs, hyps):
        prompt = _GEMBA_PROMPT.format(src_lang=src_lang, tgt_lang=tgt_lang, src=s, hyp=h)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
        )
        segs.append(_parse_score(resp.choices[0].message.content or ""))

    valid = [x for x in segs if x is not None]
    system = sum(valid) / len(valid) if valid else float("nan")
    return {"system": system, "segments": segs}

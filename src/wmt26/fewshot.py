"""Few-shot example retrieval from a parallel pool (especially for zh->id).

Retrieved (src, tgt) pairs go into prompts.advanced_prompt(examples=...).
Fuzzy string match via stdlib difflib — docs/research.md names edit-distance
retrieval a strong, cheap baseline that gets a generic open model near GPT-4o-mini.

ponytail: difflib SequenceMatcher, O(pool) per line. Upgrade to FAISS + sentence
embeddings over OpenSubtitles only if pool size / recall demands it — and account
the embedder's params against the 20B budget then.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path


def load_pool(tsv_path: str | Path) -> list[tuple[str, str]]:
    """Load a zh<TAB>target parallel pool. Blank/malformed lines skipped."""
    pairs: list[tuple[str, str]] = []
    for line in Path(tsv_path).read_text(encoding="utf-8").splitlines():
        src, sep, tgt = line.partition("\t")
        if sep and src.strip() and tgt.strip():
            pairs.append((src.strip(), tgt.strip()))
    return pairs


def retrieve(
    src_text: str, pool: list[tuple[str, str]], k: int = 5
) -> list[tuple[str, str]]:
    """Top-k pool pairs whose source is most similar to `src_text`."""
    ranked = sorted(
        pool, key=lambda p: SequenceMatcher(None, src_text, p[0]).ratio(), reverse=True
    )
    return ranked[:k]

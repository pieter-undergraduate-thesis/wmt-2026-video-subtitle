"""Idiom/terminology database: load, and match source lines against it.

The retrieval half of retrieval-augmented post-editing (docs/postedit-research.md).
An entry pairs a Chinese headword with its *figurative* meaning, the preferred
rendering per target language, and the literal renderings to avoid — the three
signals that move idiom accuracy (IdiomKB, Li et al. AAAI 2024).

Detection and retrieval are the same call: the DB is a dict keyed by headword,
so `match` both flags a line and returns what to inject into the prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    zh: str                                   # headword, simplified — the match key
    meaning_en: str = ""                      # figurative meaning
    equivalents: dict[str, list[str]] = field(default_factory=dict)  # {"en": [...], "id": [...]}
    avoid: list[str] = field(default_factory=list)
    register: str = "neutral"                 # historical | modern | internet_slang | neutral
    category: str = ""                        # chengyu | title_honorific | martial_arts | ...
    provenance: list[str] = field(default_factory=list)
    license: str = ""


def _entry(d: dict) -> Entry:
    """Build an Entry from a raw dict. Unknown keys (build bookkeeping such as
    `confidence`) are ignored on purpose — the build's records shouldn't leak
    into the runtime schema."""
    return Entry(
        zh=d["zh"],
        meaning_en=d.get("meaning_en", ""),
        equivalents={k: list(v) for k, v in (d.get("equivalents") or {}).items()},
        avoid=list(d.get("avoid") or []),
        register=d.get("register") or "neutral",
        category=d.get("category", ""),
        provenance=list(d.get("provenance") or []),
        license=d.get("license", ""),
    )


class TermDB:
    def __init__(self, entries: list[Entry]):
        self._by_zh = {e.zh: e for e in entries if e.zh}
        self._maxlen = max((len(z) for z in self._by_zh), default=0)

    def __len__(self) -> int:
        return len(self._by_zh)

    def match(self, text: str) -> list[Entry]:
        """Entries whose headword occurs in `text`, longest-match-wins, deduped.

        Chinese has no spaces, so substring matching *is* the right primitive for
        fixed multi-character expressions — word segmentation (jieba) would only
        lose recall here. A headword dict + this sliding window is a trie in a
        few lines.

        ponytail: O(len(text) * maxlen) dict lookups per line, no dependency.
        Swap in pyahocorasick only if the DB or the longest headword makes that
        measurable next to a 7B forward pass (it won't at a few thousand entries).
        """
        found: dict[str, Entry] = {}  # keyed by headword = order-preserving dedup
        i, n = 0, len(text)
        while i < n:
            for size in range(min(self._maxlen, n - i), 0, -1):
                e = self._by_zh.get(text[i : i + size])
                if e is not None:
                    found.setdefault(e.zh, e)
                    i += size  # consume the span; no overlapping matches
                    break
            else:
                i += 1
        return list(found.values())


def load(path: str | Path) -> TermDB:
    """Load a JSONL term DB (one entry per line; blank and `#` lines skipped)."""
    entries: list[Entry] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(_entry(json.loads(line)))
    return TermDB(entries)


def required(entries: list[Entry], lang: str) -> int:
    """How many of `entries` actually constrain `lang` (i.e. have an equivalent)."""
    return sum(1 for e in entries if e.equivalents.get(lang))


def hits(text: str, entries: list[Entry], lang: str) -> int:
    """How many of `entries` have one of their `lang` equivalents present in `text`.

    Shared by the post-edit accept gate and metrics.term_recall so both agree on
    what "the term was used" means.
    """
    low = text.lower()
    return sum(
        1
        for e in entries
        if any(q.lower() in low for q in e.equivalents.get(lang, []))
    )

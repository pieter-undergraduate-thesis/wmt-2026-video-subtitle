"""Build the term DB from the multi-source plan in docs/postedit-research.md §5.

Stages: parse -> merge -> filter -> enrich -> qc -> emit. Logic lives here (so
it's importable and testable, like pipeline.py); scripts/build_termdb.py is the
thin CLI over it, mirroring run_pipeline.py.

Two things carry the design:

- **Order matters: `filter(attested=...)` runs before `enrich`.** Enriching all
  ~125k CC-CEDICT entries is 125k LLM calls; enriching only the few thousand that
  actually fire on the test corpus is the same final quality for ~40x less
  compute, and keeps the committed artifact at MBs rather than ~50MB.
- **Provenance and license travel with every entry.** PETCI and IdiomTranslate30
  are CC-BY-NC-SA; WMT26 needs a permissively-licensed release, so `emit` can
  drop NC-derived entries (`exclude_nc=True`) without rebuilding.

Entries are plain dicts here — the same shape termdb.load reads. Build-only keys
(`confidence`) ride along; termdb._entry ignores what it doesn't know.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Callable, Iterable

from . import termdb, translate
from .subtitle_io import parse_srt

# `Traditional Simplified [pin1 yin1] /gloss/gloss/`
_CEDICT_RE = re.compile(r"^(\S+)\s+(\S+)\s+\[([^\]]*)\]\s+/(.+)/\s*$")

# CC-CEDICT marks set phrases in the gloss itself -- free chengyu detection.
_IDIOM_MARKER = "(idiom"

LIST_SEP = ";"  # multi-value cells in manual TSVs / LLM replies

# Most-restrictive-wins when sources merge.
_LICENSE_RANK = {"": 0, "CC-BY-SA-4.0": 1, "CC-BY-NC-SA-4.0": 2, "unknown": 3}


def _split(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(LIST_SEP) if x.strip()]


# --- parse -------------------------------------------------------------------


def parse_cedict(
    path: str | Path, lang: str, source: str, license: str
) -> list[dict]:
    """Parse a CC-CEDICT-format file into entries, glosses landing in `lang`.

    CC-CIDICT (zh-id, cidict.org) is a direct CC-CEDICT derivative and shares the
    format exactly, so it reads through here too with lang="id" -- the single
    biggest reuse win in the build, and the only non-pivot zh->id source we have.
    """
    out: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        m = _CEDICT_RE.match(line)
        if not m:
            continue
        _trad, simp, _pinyin, defs = m.groups()
        glosses = [g.strip() for g in defs.split("/") if g.strip()]
        if not glosses:
            continue
        entry = {
            "zh": simp,
            "equivalents": {lang: glosses},
            "provenance": [source],
            "license": license,
        }
        if any(_IDIOM_MARKER in g for g in glosses):
            entry["category"] = "chengyu"
        out.append(entry)
    return out


def parse_idiomkb(path: str | Path) -> list[dict]:
    """IdiomKB's zh_idiom_meaning.json -> figurative meanings.

    Meanings are GPT-3.5-generated and the repo itself warns they may contain
    errors, so they land as confidence="llm" and QC can weed them.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.values() if isinstance(data, dict) else data
    out: list[dict] = []
    for r in rows:
        zh = (r.get("idiom") or r.get("zh") or "").strip()
        meaning = (r.get("en_meaning") or "").strip()
        if not zh:
            continue
        out.append({
            "zh": zh,
            "meaning_en": meaning,
            "category": "chengyu",
            "provenance": ["IdiomKB"],
            "license": "CC-BY-SA-4.0",
            "confidence": "llm",
        })
    return out


def parse_petci(path: str | Path) -> list[dict]:
    """PETCI: chengyu<TAB>english[;english...]. CC-BY-NC-SA -- flagged as such."""
    out: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        zh, sep, en = line.partition("\t")
        if not sep or not zh.strip():
            continue
        eqs = _split(en)
        if not eqs:
            continue
        out.append({
            "zh": zh.strip(),
            "equivalents": {"en": eqs},
            "category": "chengyu",
            "provenance": ["PETCI"],
            "license": "CC-BY-NC-SA-4.0",
        })
    return out


def parse_manual(path: str | Path) -> list[dict]:
    """Hand-curated TSV with a header row. All columns but `zh` are optional:

        zh	en	id	category	register	meaning_en	avoid
        江湖	jianghu;the martial world	dunia persilatan	wuxia_term	historical	the world of...	rivers and lakes

    This is the drop-in path for the fan glossaries (Immortal Mountain,
    Wuxiaworld, honorific/imperial-title references). Deliberately not scraped:
    those are unversioned community sites with murky licensing, and a scraper
    against them would rot faster than it earns.
    """
    out: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            zh = (row.get("zh") or "").strip()
            if not zh:
                continue
            eqs = {
                lang: _split(row.get(lang, ""))
                for lang in ("en", "id")
                if _split(row.get(lang, ""))
            }
            out.append({
                "zh": zh,
                "meaning_en": (row.get("meaning_en") or "").strip(),
                "equivalents": eqs,
                "avoid": _split(row.get("avoid", "")),
                "register": (row.get("register") or "").strip() or "neutral",
                "category": (row.get("category") or "").strip(),
                "provenance": [f"manual:{Path(path).stem}"],
                "license": "CC-BY-SA-4.0",
                "confidence": "verified",
            })
    return out


# --- merge -------------------------------------------------------------------


def merge(*groups: Iterable[dict]) -> list[dict]:
    """Union entries by headword: equivalents unioned, provenance accumulated,
    license set to the most restrictive contributor."""
    out: dict[str, dict] = {}
    for group in groups:
        for e in group:
            cur = out.get(e["zh"])
            if cur is None:
                out[e["zh"]] = {
                    "zh": e["zh"],
                    "meaning_en": e.get("meaning_en", ""),
                    "equivalents": {k: list(v) for k, v in (e.get("equivalents") or {}).items()},
                    "avoid": list(e.get("avoid") or []),
                    "register": e.get("register") or "neutral",
                    "category": e.get("category", ""),
                    "provenance": list(e.get("provenance") or []),
                    "license": e.get("license", ""),
                    "confidence": e.get("confidence", "source"),
                }
                continue
            for lang, eqs in (e.get("equivalents") or {}).items():
                merged = cur["equivalents"].setdefault(lang, [])
                merged.extend(q for q in eqs if q not in merged)
            for a in e.get("avoid") or []:
                if a not in cur["avoid"]:
                    cur["avoid"].append(a)
            cur["meaning_en"] = cur["meaning_en"] or e.get("meaning_en", "")
            cur["category"] = cur["category"] or e.get("category", "")
            if cur["register"] == "neutral":
                cur["register"] = e.get("register") or "neutral"
            cur["provenance"].extend(
                p for p in (e.get("provenance") or []) if p not in cur["provenance"]
            )
            if _LICENSE_RANK.get(e.get("license", ""), 3) > _LICENSE_RANK.get(cur["license"], 3):
                cur["license"] = e.get("license", "")
    return list(out.values())


# --- filter ------------------------------------------------------------------


def filter_entries(
    entries: list[dict], *, min_len: int = 2, tests_dir: str | Path | None = None
) -> list[dict]:
    """Cut CC-CEDICT's bulk down to what can actually help.

    Two cuts:
    - **Categorised only.** CC-CEDICT's 125k is mostly ordinary vocabulary (的,
      你好). Matching those would fire on nearly every line and drive the
      over-editing APE is notorious for. Keep entries carrying a category --
      i.e. the `(idiom)`-marked chengyu plus everything hand-curated.
    - **min_len 2.** Single-character headwords would match constantly.

    `tests_dir` additionally keeps only entries that fire on the real source
    corpus, reusing the runtime matcher (termdb.TermDB) so the build and
    inference agree on what "matches" means.
    """
    kept = [e for e in entries if e.get("category") and len(e["zh"]) >= min_len]
    if tests_dir is None:
        return kept

    db = termdb.TermDB([termdb._entry(e) for e in kept])
    fired: set[str] = set()
    for p in sorted(Path(tests_dir).glob("*_zh.srt")):
        for cue in parse_srt(str(p)):
            fired.update(e.zh for e in db.match(cue.text))
    return [e for e in kept if e["zh"] in fired]


# --- enrich ------------------------------------------------------------------

# Build-time only, so it stays out of prompts.py -- nothing at inference reads it.
_ENRICH_PROMPT = (
    "You are compiling a translation glossary for Chinese historical drama "
    "(wuxia/xianxia/palace) subtitles.\n\n"
    "Expression: {zh}\n"
    "Known English renderings: {en}\n"
    "Known meaning: {meaning}\n\n"
    "Reply with exactly these lines, nothing else:\n"
    "meaning: <one-line figurative meaning in English>\n"
    "register: <historical|modern|internet_slang|neutral>\n"
    "category: <chengyu|title_honorific|wuxia_term|martial_arts|cultivation|culture_item|theatre|internet_slang>\n"
    "indonesian: <Indonesian rendering(s), semicolon-separated>\n"
    "avoid: <misleading literal rendering(s), semicolon-separated, or none>"
)


def _parse_kv(raw: str) -> dict[str, str]:
    """`key: value` lines -> dict. Mirrors glossary._parse_glossary's tolerance:
    skip anything malformed rather than fail the batch."""
    out: dict[str, str] = {}
    for line in raw.splitlines():
        key, sep, val = line.partition(":")
        if sep and key.strip() and val.strip():
            out[key.strip().lower()] = val.strip()
    return out


def enrich(
    entries: list[dict],
    *,
    client=None,
    model: str = translate.DEFAULT_MODEL,
    complete_fn: Callable[..., str] = translate._complete,
) -> list[dict]:
    """Fill missing meaning / register / category / Indonesian, one LLM call each.

    Indonesian is generated by **English pivot** (research §5.2) -- only where
    CC-CIDICT had no entry, since a real lexical source beats a pivot. Anything
    the model produced is marked confidence="llm"; QC and the human spot-check
    key off that.

    Only fields that are missing get written -- source data always wins.
    """
    client = client or translate._client()
    for e in entries:
        needs_id = not e.get("equivalents", {}).get("id")
        if e.get("meaning_en") and e.get("category") and not needs_id:
            continue
        raw = complete_fn(
            client,
            _ENRICH_PROMPT.format(
                zh=e["zh"],
                en=" / ".join(e.get("equivalents", {}).get("en", [])) or "(none)",
                meaning=e.get("meaning_en") or "(unknown)",
            ),
            model,
        )
        kv = _parse_kv(raw or "")
        if not kv:
            continue
        if not e.get("meaning_en") and kv.get("meaning"):
            e["meaning_en"] = kv["meaning"]
            e["confidence"] = "llm"
        if not e.get("category") and kv.get("category"):
            e["category"] = kv["category"]
        if e.get("register", "neutral") == "neutral" and kv.get("register"):
            e["register"] = kv["register"]
        if needs_id and kv.get("indonesian"):
            e.setdefault("equivalents", {})["id"] = _split(kv["indonesian"])
            e["confidence"] = "llm"
        avoid = _split(kv.get("avoid", ""))
        if avoid and avoid[0].lower() != "none":
            e["avoid"] = list(dict.fromkeys(e.get("avoid", []) + avoid))
    return entries


# --- qc / emit ---------------------------------------------------------------


def qc(entries: list[dict], *, langs: tuple[str, ...] = ("en", "id")) -> tuple[list[dict], dict]:
    """Drop unusable entries; return (kept, report-counts).

    Unusable = no headword, headword under 2 chars, or no rendering in any target
    language (nothing to inject and nothing term_recall could ever score).
    """
    kept: list[dict] = []
    dropped = 0
    for e in entries:
        eqs = e.get("equivalents") or {}
        if len(e.get("zh", "")) < 2 or not any(eqs.get(l) for l in langs):
            dropped += 1
            continue
        kept.append(e)

    report: dict = {"kept": len(kept), "dropped": dropped}
    for axis in ("category", "license", "confidence"):
        counts: dict[str, int] = {}
        for e in kept:
            counts[e.get(axis) or "(unset)"] = counts.get(e.get(axis) or "(unset)", 0) + 1
        report[axis] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return kept, report


def emit(entries: list[dict], path: str | Path, *, exclude_nc: bool = False) -> int:
    """Write the JSONL termdb.load reads. `exclude_nc` drops NC-derived entries
    so a shippable, permissively-licensed DB is one flag, not a rebuild."""
    if exclude_nc:
        entries = [e for e in entries if "NC" not in (e.get("license") or "")]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in sorted(entries, key=lambda e: e["zh"]):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(entries)

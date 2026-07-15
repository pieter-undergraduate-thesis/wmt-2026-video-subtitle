#!/usr/bin/env python
"""Build resources/termdb.jsonl from the sources in docs/postedit-research.md §5.

Thin CLI over wmt26.termdb_build (which holds the parsers, so they're testable).

Two source dirs, split by what they are:

  --raw     downloaded dictionaries: big, licensed, regenerable, so they live in
            data/termdb_build/raw (data/ is already gitignored).
  --manual  hand-curated glossaries: small and authored, so they're committed
            under resources/manual/ like any other source file.

Everything is optional -- run with whatever is present and re-run when more
lands. Expected files in --raw:

  cedict.txt        CC-CEDICT, CC-BY-SA-4.0  -- auto-downloadable via --fetch
                    https://www.mdbg.net/chinese/dictionary?page=cc-cedict
  cidict.txt        CC-CIDICT (zh-id), CC-BY-SA-4.0, same format -- cidict.org
  idiomkb.json      IdiomKB zh_idiom_meaning.json -- github.com/lishuang-w/IdiomKB
  petci.tsv         PETCI chengyu<TAB>en;en -- CC-BY-NC-SA (see --exclude-nc)

and any *.tsv in --manual, header row: zh en id category register meaning_en
avoid (only `zh` is required).

Only CC-CEDICT is auto-fetched -- the rest need a manual drop rather than a
guessed URL or a scraper against an unversioned community site.

Usage:
  python scripts/build_termdb.py --fetch                       # get CC-CEDICT
  python scripts/build_termdb.py --attested data/tests         # build (no LLM)
  python scripts/build_termdb.py --attested data/tests --enrich --exclude-nc
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

from wmt26 import termdb_build as tb
from wmt26 import translate

CEDICT_URL = (
    "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"
)


def fetch_cedict(raw: Path) -> None:
    dest = raw / "cedict.txt"
    if dest.exists():
        print(f"{dest} exists, skipping fetch")
        return
    raw.mkdir(parents=True, exist_ok=True)
    gz = raw / "cedict.txt.gz"
    print(f"downloading {CEDICT_URL}")
    urllib.request.urlretrieve(CEDICT_URL, gz)
    with gzip.open(gz, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz.unlink()
    print(f"wrote {dest}")


def collect_sources(raw: Path, manual: Path) -> list[list[dict]]:
    groups: list[list[dict]] = []

    def add(name: str, fn, *args) -> None:
        p = raw / name
        if not p.exists():
            print(f"  - {name}: absent, skipping")
            return
        rows = fn(p, *args)
        print(f"  - {name}: {len(rows)} entries")
        groups.append(rows)

    print("parsing sources:")
    add("cedict.txt", tb.parse_cedict, "en", "CC-CEDICT", "CC-BY-SA-4.0")
    add("cidict.txt", tb.parse_cedict, "id", "CC-CIDICT", "CC-BY-SA-4.0")
    add("idiomkb.json", tb.parse_idiomkb)
    add("petci.tsv", tb.parse_petci)

    for tsv in sorted(manual.glob("*.tsv")):
        rows = tb.parse_manual(tsv)
        print(f"  - {manual.name}/{tsv.name}: {len(rows)} entries")
        groups.append(rows)

    if not groups:
        raise SystemExit(
            f"no sources found in {raw} or {manual}. Run --fetch for CC-CEDICT, or "
            "drop files by hand (see this script's docstring for names/URLs)."
        )
    return groups


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/termdb_build/raw"),
                   help="downloaded dictionaries (gitignored)")
    p.add_argument("--manual", type=Path, default=Path("resources/manual"),
                   help="hand-curated glossary TSVs (committed)")
    p.add_argument("--out", type=Path, default=Path("resources/termdb.jsonl"))
    p.add_argument("--fetch", action="store_true", help="download CC-CEDICT first")
    p.add_argument("--attested", type=Path, default=None, metavar="TESTS_DIR",
                   help="keep only entries that fire on the source corpus "
                        "(e.g. data/tests) -- run this before --enrich")
    p.add_argument("--enrich", action="store_true",
                   help="LLM pass for missing meaning/register/category and "
                        "English-pivot Indonesian (needs scripts/serve.sh)")
    p.add_argument("--exclude-nc", dest="exclude_nc", action="store_true",
                   help="drop NC-licensed entries (PETCI) for a shippable DB")
    p.add_argument("--model", default=translate.DEFAULT_MODEL)
    p.add_argument("--base-url", dest="base_url", default=translate.DEFAULT_BASE_URL)
    a = p.parse_args()

    if a.fetch:
        fetch_cedict(a.raw)

    entries = tb.merge(*collect_sources(a.raw, a.manual))
    print(f"merged: {len(entries)} unique headwords")

    entries = tb.filter_entries(entries, tests_dir=a.attested)
    print(f"filtered: {len(entries)} entries"
          + (f" attested in {a.attested}" if a.attested else " (categorised, len>=2)"))

    if a.enrich:
        if not a.attested:
            # 125k LLM calls vs a few thousand, for the same final quality.
            print("WARNING: --enrich without --attested will hit the model once per "
                  "entry. Filter to the test corpus first unless you mean it.")
        print(f"enriching {len(entries)} entries via {a.model} ...")
        entries = tb.enrich(
            entries, client=translate._client(a.base_url), model=a.model
        )

    entries, report = tb.qc(entries)
    print(f"\nqc: kept {report['kept']}, dropped {report['dropped']}")
    for axis in ("category", "license", "confidence"):
        print(f"  {axis}: {report[axis]}")

    n = tb.emit(entries, a.out, exclude_nc=a.exclude_nc)
    print(f"\nwrote {a.out} ({n} entries)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Retrieval-augmented post-edit pass over an already-translated output dir.

Reads source <vid>_zh.srt + a draft <vid>_<lang>.srt, flags the cues whose
source carries a term-DB idiom/honorific/wuxia term, re-writes only those, and
writes a new <vid>_<lang>.srt. Runs as a second pass rather than inline in the
pipeline so an A/B costs no re-translation: only the ~10-20% of lines that flag
ever reach the model.

Needs a running vLLM server (scripts/serve.sh). --qe-gate additionally needs
CometKiwi weights (GPU) and is opt-in on purpose: docs/postedit-research.md
finding 3 reports reference-free QE correlating near-zero on idiom lines.

Score the result against the draft with the same harness:
  python scripts/run_benchmark_eval.py --hyp results/hymt2-7b-out --termdb resources/termdb.jsonl
  python scripts/run_benchmark_eval.py --hyp result/out_pe        --termdb resources/termdb.jsonl

Usage:
  python scripts/run_postedit.py --src data/tests --hyp results/hymt2-7b-out \
      --out result/out_pe --langs en id --limit 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wmt26 import termdb, translate
from wmt26.postedit import postedit_cues
from wmt26.prompts import REGISTER_NOTES
from wmt26.subtitle_io import parse_srt, write_srt


def run(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    db = termdb.load(args.termdb)
    srt_files = sorted(args.src_dir.glob("*_zh.srt"))
    if args.limit:
        srt_files = srt_files[: args.limit]
    if not srt_files:
        raise SystemExit(f"no *_zh.srt files in {args.src_dir}")
    print(f"loaded {len(db)} term entries from {args.termdb}")

    client = translate._client(args.base_url)
    totals = {"cues": 0, "flagged": 0, "edited": 0, "accepted": 0}

    for srt_path in srt_files:
        vid = srt_path.stem.rsplit("_", 1)[0]  # a0046jl5d6d_zh -> a0046jl5d6d
        src_cues = parse_srt(str(srt_path))

        for lang in args.langs:
            hyp_path = args.hyp_dir / f"{vid}_{lang}.srt"
            if not hyp_path.exists():
                continue
            out, st = postedit_cues(
                src_cues,
                parse_srt(str(hyp_path)),
                lang,
                db,
                register=REGISTER_NOTES.get(lang),
                qe_gate=args.qe_gate,
                client=client,
                model=args.model,
            )
            out_path = args.out_dir / f"{vid}_{lang}.srt"
            write_srt(out, str(out_path))
            print(
                f"wrote {out_path}: {st.cues} cues, {st.flagged} flagged, "
                f"{st.edited} edited, {st.accepted} accepted"
            )
            for k in totals:
                totals[k] += getattr(st, k)

    # Over-edit / gain-to-edit instrumentation (research §4).
    if totals["cues"]:
        print(
            f"\ntotal: {totals['cues']} cues | "
            f"flagged {totals['flagged']} ({totals['flagged'] / totals['cues']:.1%}) | "
            f"edited {totals['edited']} | accepted {totals['accepted']}"
            + (
                f" ({totals['accepted'] / totals['edited']:.1%} of edits)"
                if totals["edited"]
                else ""
            )
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", dest="src_dir", default=Path("data/tests"), type=Path)
    p.add_argument("--hyp", dest="hyp_dir", required=True, type=Path,
                   help="dir of draft <vid>_<lang>.srt to post-edit")
    p.add_argument("--out", dest="out_dir", default=Path("result/out_pe"), type=Path)
    p.add_argument("--langs", nargs="+", default=["en", "id"])
    p.add_argument("--termdb", default=Path("resources/termdb.jsonl"), type=Path)
    p.add_argument("--limit", type=int, default=0, help="only the first N source files (0 = all)")
    p.add_argument("--qe-gate", dest="qe_gate", action="store_true",
                   help="also require CometKiwi non-regression (GPU)")
    p.add_argument("--model", default=translate.DEFAULT_MODEL)
    p.add_argument("--base-url", dest="base_url", default=translate.DEFAULT_BASE_URL)
    run(p.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Advanced training-free pipeline: candidates -> QE-rerank -> post-edit -> glossary.

Layers the docs/research.md quality stages on top of the existing per-cue
orchestration (wmt26.pipeline.translate_cues) via an injected translate_fn, so
splitting / re-indexing / rolling context are reused unchanged. Output naming
matches zero_shot_translate.py so run_benchmark_eval.py scores it directly.

Needs a running vLLM server (scripts/serve.sh) and CometKiwi weights for the
reranker. Use --limit to try the first N source files before a full run.

Resumable: a (video, lang) whose output .srt already exists is skipped, so an
interrupted run picks up where it stopped by re-issuing the same command. Pass
--overwrite to force a re-translate.

Usage:
  python scripts/run_advanced_pipeline.py --in data/tests --out result/out_p1 --langs en id --limit 20
  python scripts/run_advanced_pipeline.py --in data/tests --out result/out_p1 --postedit --fewshot-pool data/zh_id.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wmt26 import translate
from wmt26.candidates import DEFAULT_TEMPS, generate_candidates
from wmt26.fewshot import load_pool, retrieve
from wmt26.glossary import build_glossary, enforce as enforce_glossary
from wmt26.metadata import synopsis_for_srt
from wmt26.pipeline import translate_cues
from wmt26.postedit import postedit
from wmt26.prompts import REGISTER_NOTES
from wmt26.rerank import pick_best
from wmt26.subtitle_io import parse_srt, write_srt


def run(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    srt_files = sorted(args.in_dir.glob("*.srt"))
    if args.limit:
        srt_files = srt_files[: args.limit]
    if not srt_files:
        raise SystemExit(f"no .srt files in {args.in_dir}")

    client = translate._client(args.base_url)
    pool = load_pool(args.fewshot_pool) if args.fewshot_pool else []

    for srt_path in srt_files:
        vid = srt_path.stem.rsplit("_", 1)[0]  # a0046jl5d6d_zh -> a0046jl5d6d
        cues = parse_srt(str(srt_path))
        synopsis = args.synopsis or synopsis_for_srt(srt_path)

        for lang in args.langs:
            out_path = args.out_dir / f"{vid}_{lang}.srt"
            # Resume: skip before build_glossary — that's an LLM pass per (video, lang),
            # so checking any later would still pay for work we're throwing away.
            # ponytail: existence, not validity. A run killed mid-write leaves a
            # partial .srt this will happily skip — delete it or pass --overwrite.
            if out_path.exists() and not args.overwrite:
                print(f"skip {out_path} (exists)")
                continue

            register = REGISTER_NOTES.get(lang)
            gloss = (
                {} if args.no_glossary
                else build_glossary(cues, synopsis, lang, client=client, model=args.model)
            )

            def cand_translate(text, target, context=None, *, register=None, **kw):
                examples = retrieve(text, pool, k=args.shots) if pool else None
                cands = generate_candidates(
                    text, target, context=context, glossary=gloss or None,
                    examples=examples, register=register, n=args.n,
                    temps=DEFAULT_TEMPS, client=client, model=args.model,
                )
                best = pick_best(text, cands, target)
                if args.postedit:
                    best = postedit(text, best, target, register=register,
                                    client=client, model=args.model)
                return enforce_glossary(best, gloss) if gloss else best

            out = translate_cues(
                cues, lang, synopsis=synopsis, register=register,
                translate_fn=cand_translate, model=args.model, base_url=args.base_url,
            )
            write_srt(out, str(out_path))
            print(f"wrote {out_path} ({len(out)} cues)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", default=Path("data/tests"), type=Path)
    p.add_argument("--out", dest="out_dir", default=Path("result/out_p1"), type=Path)
    p.add_argument("--langs", nargs="+", default=["en", "id"])
    p.add_argument("--limit", type=int, default=0, help="only the first N source files (0 = all)")
    p.add_argument("--n", type=int, default=6, help="candidates per cue (capped by temp spread)")
    p.add_argument("--shots", type=int, default=5, help="few-shot examples per cue")
    p.add_argument("--postedit", action="store_true", help="add one post-edit pass")
    p.add_argument("--no-glossary", action="store_true", help="skip episode glossary")
    p.add_argument("--overwrite", action="store_true",
                   help="re-translate even if the output .srt exists (default: skip)")
    p.add_argument("--fewshot-pool", dest="fewshot_pool", type=Path,
                   help="zh<TAB>target TSV pool for few-shot retrieval")
    p.add_argument("--synopsis", default=None, help="override metadata synopsis")
    p.add_argument("--model", default=translate.DEFAULT_MODEL)
    p.add_argument("--base-url", dest="base_url", default=translate.DEFAULT_BASE_URL)
    run(p.parse_args())


if __name__ == "__main__":
    main()

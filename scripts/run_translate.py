#!/usr/bin/env python
"""Translate every .srt in a dir into the target language(s).

--window 0  = zero-shot (the Phase A baseline: synopsis context only)
--window N  = few-shot ICL, where the examples are the video's own last N
              already-translated cues (see wmt26.context.RecentContext)

Baseline and ICL runs go through the same code path, so a score delta is
attributable to the window and not to a second implementation.

Output naming follows the WMT26 submission format: <vid>_<lang>.srt.
Needs a running vLLM server (scripts/serve.sh). Existing outputs are skipped,
so an interrupted run resumes.

Usage:
  python scripts/run_translate.py --in test_data/... --out runs/w0 --langs en id --window 0
  python scripts/run_translate.py --in test_data/... --out runs/w4 --langs en id --window 4 --videos 10
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from wmt26 import translate
from wmt26.metadata import synopsis_for_srt
from wmt26.pipeline import translate_cues
from wmt26.subtitle_io import parse_srt, write_srt


def run_one(
    srt_path: Path,
    out_dir: Path,
    lang: str,
    synopsis: str | None,
    window: int,
    model: str,
    base_url: str,
) -> Path:
    vid = srt_path.stem.rsplit("_", 1)[0]  # a0046jl5d6d_zh -> a0046jl5d6d
    out_path = out_dir / f"{vid}_{lang}.srt"
    if out_path.exists():
        logger.info(f"skip {out_path.name} (exists)")
        return out_path

    cues = parse_srt(str(srt_path))
    # Cues stay 1:1 with the source — professional subs don't re-segment, and
    # splitting can't fix CPS anyway.
    out = translate_cues(
        cues,
        lang,
        synopsis=synopsis or synopsis_for_srt(srt_path),
        window=window,
        model=model,
        base_url=base_url,
    )
    write_srt(out, str(out_path))
    logger.info(f"wrote {out_path.name} ({len(out)} cues)")
    return out_path


def run(
    in_dir: Path,
    out_dir: Path,
    langs: list[str],
    synopsis: str | None,
    window: int,
    videos: int | None,
    workers: int,
    model: str,
    base_url: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_files = sorted(in_dir.glob("*.srt"))
    if not srt_files:
        raise SystemExit(f"no .srt files in {in_dir}")
    if videos:
        srt_files = srt_files[:videos]

    # With window > 0 a video's cues are strictly sequential (cue N's prompt carries
    # cue N-1's output), so parallelism has to come from running separate
    # (video, lang) jobs at once and letting vLLM batch across them.
    jobs = [(f, lang) for f in srt_files for lang in langs]
    logger.info(f"{len(jobs)} jobs | window={window} | workers={workers} | out={out_dir}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, f, out_dir, lang, synopsis, window, model, base_url): (f, lang)
            for f, lang in jobs
        }
        for fut in as_completed(futures):
            f, lang = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.error(f"failed {f.name} -> {lang}: {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", default=Path("data"), type=Path)
    p.add_argument("--out", dest="out_dir", default=Path("out"), type=Path)
    p.add_argument("--langs", nargs="+", default=["en", "th", "id", "ms", "zh-TW"])
    p.add_argument("--window", type=int, default=0, help="0 = zero-shot; N = N ICL examples")
    p.add_argument("--videos", type=int, default=None, help="only the first N videos (dev subset)")
    p.add_argument("--workers", type=int, default=8, help="concurrent (video, lang) jobs")
    p.add_argument("--model", default=translate.DEFAULT_MODEL)
    p.add_argument("--base-url", default=translate.DEFAULT_BASE_URL)
    p.add_argument("--synopsis", default=None, help="override the per-video metadata synopsis")
    a = p.parse_args()
    run(
        a.in_dir, a.out_dir, a.langs, a.synopsis, a.window,
        a.videos, a.workers, a.model, a.base_url,
    )


if __name__ == "__main__":
    main()

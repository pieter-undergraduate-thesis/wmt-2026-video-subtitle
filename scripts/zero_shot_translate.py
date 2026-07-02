#!/usr/bin/env python
"""Zero-shot baseline: translate every .srt in a dir into the target language(s).

Output naming follows the WMT26 submission format: vid_langshort.srt
(langshort = en/th/id/ms/zh-TW). Needs a running vLLM server (scripts/serve.sh).

Usage:
  python scripts/zero_shot_translate.py --in data/test --out out --langs en th id ms zh-TW
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wmt26 import translate
from wmt26.constraints import split_cue_if_needed
from wmt26.metadata import synopsis_for_srt
from wmt26.subtitle_io import Cue, parse_srt, write_srt


def run(in_dir: Path, out_dir: Path, langs: list[str], synopsis: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    srt_files = sorted(in_dir.glob("*.srt"))
    if not srt_files:
        raise SystemExit(f"no .srt files in {in_dir}")

    for srt_path in srt_files:
        vid = srt_path.stem.rsplit("_", 1)[0]  # a0046jl5d6d_zh -> a0046jl5d6d
        cues = parse_srt(str(srt_path))
        # Manual --synopsis overrides; otherwise use the video's own metadata.
        syn = synopsis or synopsis_for_srt(srt_path)
        for lang in langs:
            translated: list[Cue] = []
            for c in cues:
                txt = translate.translate_cue(c.text, lang, syn)
                translated.extend(
                    split_cue_if_needed(Cue(c.index, c.start_ms, c.end_ms, txt))
                )
            for i, c in enumerate(translated, start=1):
                c.index = i
            out_path = out_dir / f"{vid}_{lang}.srt"
            write_srt(translated, str(out_path))
            print(f"wrote {out_path} ({len(translated)} cues)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", default=Path("data"), type=Path)
    p.add_argument("--out", dest="out_dir", default=Path("out"), type=Path)
    p.add_argument("--langs", nargs="+", default=["en", "th", "id", "ms", "zh-TW"])
    p.add_argument("--synopsis", default=None, help="optional shared context string")
    a = p.parse_args()
    run(a.in_dir, a.out_dir, a.langs, a.synopsis)


if __name__ == "__main__":
    main()

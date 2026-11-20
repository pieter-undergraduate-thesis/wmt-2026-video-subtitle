#!/usr/bin/env python
"""Full Phase C pipeline on one .srt: rolling context + translate + constrain.

Usage:
  python scripts/run_pipeline.py --in vid.srt --out vid_en.srt --lang en \
      --synopsis "A courtroom drama set in 1990s Hong Kong."
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wmt26 import translate
from wmt26.metadata import synopsis_for_srt
from wmt26.pipeline import translate_srt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    p.add_argument("--lang", required=True)
    p.add_argument("--synopsis", default=None)
    p.add_argument("--base-url", default=translate.DEFAULT_BASE_URL)
    a = p.parse_args()

    # Manual --synopsis overrides; otherwise use the video's own metadata.
    synopsis = a.synopsis or synopsis_for_srt(a.in_path)
    out = translate_srt(
        str(a.in_path), str(a.out_path), a.lang, synopsis=synopsis, base_url=a.base_url
    )
    print(f"wrote {a.out_path} ({len(out)} cues)")


if __name__ == "__main__":
    main()

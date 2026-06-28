#!/usr/bin/env python
"""Full Phase C pipeline on one .srt: rolling context + translate + constrain.

Usage:
  python scripts/run_pipeline.py --in vid.srt --out vid_en.srt --lang en \
      --synopsis "A courtroom drama set in 1990s Hong Kong."
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wmt26.pipeline import translate_srt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    p.add_argument("--lang", required=True)
    p.add_argument("--synopsis", default=None)
    a = p.parse_args()

    out = translate_srt(str(a.in_path), str(a.out_path), a.lang, synopsis=a.synopsis)
    print(f"wrote {a.out_path} ({len(out)} cues)")


if __name__ == "__main__":
    main()

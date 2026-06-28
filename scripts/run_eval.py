#!/usr/bin/env python
"""Score hypothesis SRTs against reference SRTs (+ source for COMET).

Expects matching filenames across the three dirs (cue order must align).
Prints BLEU / chrF / COMET / constraint-violation-rate per file.

Usage:
  python scripts/run_eval.py --src data/src --hyp out --ref data/ref [--no-comet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wmt26.eval import evaluate
from wmt26.subtitle_io import parse_srt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--hyp", required=True, type=Path)
    p.add_argument("--ref", required=True, type=Path)
    p.add_argument("--no-comet", action="store_true")
    a = p.parse_args()

    for hyp_path in sorted(a.hyp.glob("*.srt")):
        name = hyp_path.name
        src_path, ref_path = a.src / name, a.ref / name
        if not (src_path.exists() and ref_path.exists()):
            print(f"skip {name}: missing src/ref")
            continue
        src = [c.text for c in parse_srt(str(src_path))]
        hyp = parse_srt(str(hyp_path))
        ref = [c.text for c in parse_srt(str(ref_path))]
        s = evaluate(src, hyp, ref, with_comet=not a.no_comet)
        comet = f"{s.comet:.4f}" if s.comet is not None else "n/a"
        print(
            f"{name}: BLEU={s.bleu:.2f} chrF={s.chrf:.2f} "
            f"COMET={comet} viol={s.violation_rate:.1%}"
        )


if __name__ == "__main__":
    main()

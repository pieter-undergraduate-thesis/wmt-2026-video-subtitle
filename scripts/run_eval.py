#!/usr/bin/env python
"""Score hypothesis SRTs. Reference-based, or --qe (CometKiwi, reference-free).

Expects matching filenames across dirs (cue order must align).
Prints BLEU / chrF / COMET / violation-rate per file, or CometKiwi / violation-rate with --qe.

Usage:
  python scripts/run_eval.py --src data/src --hyp out --ref data/ref [--no-comet]
  python scripts/run_eval.py --src data/src --hyp out --qe        # blind set, no --ref
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wmt26.constraints import violation_rate
from wmt26.eval import evaluate, score_comet_qe
from wmt26.subtitle_io import parse_srt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--hyp", required=True, type=Path)
    p.add_argument("--ref", type=Path, help="reference SRTs; omit with --qe")
    p.add_argument("--qe", action="store_true", help="reference-free CometKiwi QE")
    p.add_argument("--no-comet", action="store_true")
    a = p.parse_args()
    if not a.qe and a.ref is None:
        p.error("--ref is required unless --qe is set")

    for hyp_path in sorted(a.hyp.glob("*.srt")):
        name = hyp_path.name
        vid = name.rsplit("_", 1)[0]  # <vid>_en.srt -> <vid>
        src_path = a.src / f"{vid}_zh.srt" if a.qe else a.src / name
        if not src_path.exists():
            print(f"skip {name}: missing src")
            continue
        src = [c.text for c in parse_srt(str(src_path))]
        hyp = parse_srt(str(hyp_path))

        if a.qe:
            qe = score_comet_qe(src, [c.text for c in hyp])
            print(f"{name}: CometKiwi={qe:.4f} viol={violation_rate(hyp):.1%}")
            continue

        ref_path = a.ref / name
        if not ref_path.exists():
            print(f"skip {name}: missing ref")
            continue
        ref = [c.text for c in parse_srt(str(ref_path))]
        s = evaluate(src, hyp, ref, with_comet=not a.no_comet)
        comet = f"{s.comet:.4f}" if s.comet is not None else "n/a"
        print(
            f"{name}: BLEU={s.bleu:.2f} chrF={s.chrf:.2f} "
            f"COMET={comet} viol={s.violation_rate:.1%}"
        )


if __name__ == "__main__":
    main()

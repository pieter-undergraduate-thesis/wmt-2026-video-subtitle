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
import csv
from bisect import bisect_right
from pathlib import Path

from wmt26.constraints import violation_rate
from wmt26.eval import evaluate, score_comet_qe
from wmt26.subtitle_io import parse_srt


def regroup_to_src(src_cues, hyp_cues) -> list[str]:
    """Merge split hyp cues back to one string per src cue, aligned by time.

    split_cue_if_needed keeps halves inside the parent's [start, end] window, so
    each hyp cue is assigned to the src cue whose start it falls at or after.
    """
    starts = [c.start_ms for c in src_cues]
    merged = [[] for _ in src_cues]
    for h in hyp_cues:
        i = bisect_right(starts, h.start_ms) - 1
        if i < 0:
            i = 0
        merged[i].append(h.text)
    return [" ".join(parts) for parts in merged]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--hyp", required=True, type=Path)
    p.add_argument("--ref", type=Path, help="reference SRTs; omit with --qe")
    p.add_argument("--qe", action="store_true", help="reference-free CometKiwi QE")
    p.add_argument("--no-comet", action="store_true")
    p.add_argument("--out", type=Path, help="write per-file scores to this CSV")
    a = p.parse_args()
    if not a.qe and a.ref is None:
        p.error("--ref is required unless --qe is set")

    rows = []
    for hyp_path in sorted(a.hyp.glob("*.srt")):
        name = hyp_path.name
        vid = name.rsplit("_", 1)[0]  # <vid>_en.srt -> <vid>
        src_path = a.src / f"{vid}_zh.srt" if a.qe else a.src / name
        if not src_path.exists():
            print(f"skip {name}: missing src")
            continue
        src_cues = parse_srt(str(src_path))
        src = [c.text for c in src_cues]
        hyp = parse_srt(str(hyp_path))

        if a.qe:
            # Translation may split one src cue into several (split_cue_if_needed),
            # so hyp has more cues than src. Regroup hyp back into src time windows
            # to restore 1:1 alignment before positional QE scoring.
            hyp_text = regroup_to_src(src_cues, hyp)
            qe = score_comet_qe(src, hyp_text)
            viol = violation_rate(hyp)
            print(f"{name}: CometKiwi={qe:.4f} viol={viol:.1%}")
            rows.append({"file": name, "cometkiwi": f"{qe:.4f}", "viol": f"{viol:.4f}"})
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
        rows.append({
            "file": name, "bleu": f"{s.bleu:.2f}", "chrf": f"{s.chrf:.2f}",
            "comet": comet, "viol": f"{s.violation_rate:.4f}",
        })

    if a.out and rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

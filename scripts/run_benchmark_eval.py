#!/usr/bin/env python
"""WMT26 benchmark eval: zh -> en/id, hypothesis vs ground-truth subtitles.

Joins three independently-named data sources:
  data/tests/<vid>_zh.srt      (source)      -- vid is the filename stem
  data/tests/<vid>_metadata.json               video_title links to the CSV
  data/gt/data.csv             (map)         episode == video_title
  data/gt/[English|Indonesian] ... .srt (reference, named in the CSV)
  <hyp>/<vid>_<lang>.srt       (hypothesis)

Only ~89 of the 100 test videos have a ground-truth row; the rest are skipped.
Hypothesis shares the source's cue timing (1:1), but the ground-truth subtitles
are independently segmented (sometimes far finer or coarser), so hyp and ref are
aligned onto the source cue timeline by timestamp OVERLAP before scoring.

Metrics: BLEU, chrF, chrF++ (always, CPU). XCOMET-XXL + CometKiwi behind
--neural (GPU + HF token). GEMBA behind --gemba (LLM judge).

Usage:
  python scripts/run_benchmark_eval.py --hyp result/out_fixed --langs en id
  python scripts/run_benchmark_eval.py --hyp result/out_fixed --neural --out results.csv
  python scripts/run_benchmark_eval.py --hyp result/out_fixed --gemba --gemba-backend openai
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path

from wmt26.eval import score_bleu_chrf, score_comet, score_comet_qe
from wmt26.metrics import score_chrf_pp, score_gemba, score_xcomet
from wmt26.subtitle_io import parse_srt

# lang code -> (gt filename prefix, full name for GEMBA)
LANG_INFO = {
    "en": ("[English]", "English"),
    "id": ("[Indonesian]", "Indonesian"),
}


def normalize_key(s: str) -> str:
    """Join key: fold full-width punctuation to ASCII, drop .mp4 and whitespace.

    Catches e.g. video_title '..._，...' vs CSV episode '..._,...' (one row)
    on top of the exact matches.
    """
    s = unicodedata.normalize("NFKC", s)  # full-width -> ASCII
    s = s.replace(".mp4", "")
    return "".join(s.split())  # strip all whitespace


def load_gt_map(csv_path: Path) -> dict[str, dict[str, str]]:
    """normalized episode -> {'en': gtfile, 'id': gtfile} (only langs present)."""
    out: dict[str, dict[str, str]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep = normalize_key(row["episode"].strip())
            entry: dict[str, str] = {}
            for col in ("EN Subtitle", "ID Subtitle"):
                fn = (row.get(col) or "").strip()
                if not fn:
                    continue
                for lang, (prefix, _name) in LANG_INFO.items():
                    if fn.startswith(prefix):
                        entry[lang] = fn
            if entry:
                out[ep] = entry
    return out


def iter_test_videos(tests_dir: Path):
    """Yield (vid, normalized_video_title) for each metadata file."""
    for meta in sorted(tests_dir.glob("*_metadata.json")):
        data = json.loads(meta.read_text(encoding="utf-8"))
        obj = data[0] if isinstance(data, list) else data
        yield obj["vid"], normalize_key((obj.get("video_title") or "").strip())


def align_to_src(src_cues, other_cues) -> list[str]:
    """Project other_cues onto the source cue timeline by time overlap.

    Returns one string per source cue: the texts of every other-cue whose
    [start, end) overlaps that source window, space-joined. Handles GT that is
    finer than source (many ref cues merge into one window) and coarser (one
    long ref cue repeats across the windows it spans). Hyp shares src timing, so
    this is effectively 1:1 for hyp.
    """
    out = []
    for c in src_cues:
        parts = [o.text for o in other_cues if o.start_ms < c.end_ms and o.end_ms > c.start_ms]
        out.append(" ".join(parts))
    return out


def collect(tests: Path, gt: Path, hyp: Path, langs: list[str]):
    """Per lang -> list of dicts {vid, src, hyp, ref, empty_ref}."""
    gt_map = load_gt_map(gt / "data.csv")
    per_lang: dict[str, list[dict]] = {l: [] for l in langs}
    skipped = 0

    for vid, vt in iter_test_videos(tests):
        entry = gt_map.get(vt)
        if not entry:
            skipped += 1
            continue
        src_path = tests / f"{vid}_zh.srt"
        if not src_path.exists():
            continue
        src_cues = parse_srt(str(src_path))
        src = [c.text for c in src_cues]
        for lang in langs:
            ref_name = entry.get(lang)
            hyp_path = hyp / f"{vid}_{lang}.srt"
            if not ref_name or not hyp_path.exists():
                continue
            hy = align_to_src(src_cues, parse_srt(str(hyp_path)))
            ref = align_to_src(src_cues, parse_srt(str(gt / ref_name)))
            empty_ref = sum(1 for r in ref if not r.strip())
            per_lang[lang].append(
                {"vid": vid, "src": src, "hyp": hy, "ref": ref, "empty_ref": empty_ref}
            )
    return per_lang, skipped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tests", type=Path, default=Path("data/tests"))
    p.add_argument("--gt", type=Path, default=Path("data/gt"))
    p.add_argument("--hyp", type=Path, required=True)
    p.add_argument("--langs", nargs="+", default=["en", "id"])
    p.add_argument("--neural", action="store_true", help="XCOMET-XXL + CometKiwi (GPU)")
    p.add_argument("--gemba", action="store_true", help="GEMBA LLM-judge QE")
    p.add_argument("--gemba-backend", default="openai", choices=["openai", "vllm"])
    p.add_argument("--gemba-model", default="gpt-4o")
    p.add_argument("--gemba-sample", type=int, default=0,
                   help="cap cues scored by GEMBA per lang (0 = all)")
    p.add_argument("--out", type=Path, help="write per-file + aggregate rows to CSV")
    a = p.parse_args()

    bad = [l for l in a.langs if l not in LANG_INFO]
    if bad:
        p.error(f"unsupported langs {bad}; known: {list(LANG_INFO)}")

    per_lang, skipped = collect(a.tests, a.gt, a.hyp, a.langs)
    rows: list[dict] = []

    for lang in a.langs:
        items = per_lang[lang]
        print(f"\n=== {lang}: {len(items)} episodes ===")
        if not items:
            continue

        for it in items:
            bleu, chrf = score_bleu_chrf(it["hyp"], it["ref"])
            chrfpp = score_chrf_pp(it["hyp"], it["ref"])
            print(f"  {it['vid']}: BLEU={bleu:.2f} chrF={chrf:.2f} chrF++={chrfpp:.2f}"
                  + (f" ({it['empty_ref']} empty ref)" if it["empty_ref"] else ""))
            rows.append({"lang": lang, "file": it["vid"], "cues": len(it["hyp"]),
                         "bleu": f"{bleu:.2f}", "chrf": f"{chrf:.2f}",
                         "chrfpp": f"{chrfpp:.2f}", "empty_ref": it["empty_ref"]})

        srcs = [t for it in items for t in it["src"]]
        hyps = [t for it in items for t in it["hyp"]]
        refs = [t for it in items for t in it["ref"]]
        bleu, chrf = score_bleu_chrf(hyps, refs)
        chrfpp = score_chrf_pp(hyps, refs)
        agg = {"lang": lang, "file": "__ALL__", "cues": len(hyps),
               "bleu": f"{bleu:.2f}", "chrf": f"{chrf:.2f}", "chrfpp": f"{chrfpp:.2f}",
               "empty_ref": sum(it["empty_ref"] for it in items)}
        print(f"  [corpus] BLEU={bleu:.2f} chrF={chrf:.2f} chrF++={chrfpp:.2f} ({len(hyps)} cues)")

        if a.neural:
            xc = score_xcomet(srcs, hyps, refs)["system"]
            kiwi = score_comet_qe(srcs, hyps)
            comet = score_comet(srcs, hyps, refs)
            agg |= {"comet": f"{comet:.4f}", "cometkiwi": f"{kiwi:.4f}", "xcomet": f"{xc:.4f}"}
            print(f"  [corpus] COMET={comet:.4f} CometKiwi={kiwi:.4f} XCOMET={xc:.4f}")

        if a.gemba:
            gs, gh = srcs, hyps
            if a.gemba_sample and a.gemba_sample < len(gs):
                gs, gh = gs[: a.gemba_sample], gh[: a.gemba_sample]
            g = score_gemba(gs, gh, tgt_lang=LANG_INFO[lang][1],
                            backend=a.gemba_backend, model=a.gemba_model)["system"]
            agg["gemba"] = f"{g:.2f}"
            print(f"  [corpus] GEMBA={g:.2f} (n={len(gh)})")

        rows.append(agg)

    print(f"\nskipped {skipped} videos without ground truth")

    if a.out and rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())

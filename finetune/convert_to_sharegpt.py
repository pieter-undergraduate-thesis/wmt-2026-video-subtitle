#!/usr/bin/env python
"""Convert a zh->target parallel corpus to LLaMA-Factory sharegpt format.

Bakes the subtitle format constraints into the instruction string so the model
learns length-awareness end-to-end (not just post-processing). Idiom-tagged rows
can be upweighted via --idiom-repeat.

Input: TSV with columns  source<TAB>target  (optionally a 3rd col "idiom" flag).
Output: a sharegpt JSON list ready to drop in LLaMA-Factory/data/.

Usage:
  python finetune/convert_to_sharegpt.py --in corpus.tsv --out data/wmt26_sft.json \
      --lang English --idiom-repeat 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmt26.constraints import MAX_CHARS_PER_LINE, MAX_LINES


def instruction(source: str, lang: str) -> str:
    return (
        f"Translate the following text into {lang}. Keep each line under "
        f"{MAX_CHARS_PER_LINE} characters, max {MAX_LINES} lines, output only "
        f"the translation:\n{source}"
    )


def row_to_sample(source: str, target: str, lang: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": instruction(source, lang)},
            {"role": "assistant", "content": target},
        ]
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    p.add_argument("--lang", required=True, help="full target language name")
    p.add_argument("--idiom-repeat", type=int, default=1,
                   help="duplicate idiom-flagged rows N times to upweight them")
    a = p.parse_args()

    samples: list[dict] = []
    with open(a.in_path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            source, target = cols[0], cols[1]
            is_idiom = len(cols) >= 3 and cols[2].strip().lower() in {"1", "idiom", "true"}
            sample = row_to_sample(source, target, a.lang)
            samples.extend([sample] * (a.idiom_repeat if is_idiom else 1))

    a.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"wrote {a.out_path} ({len(samples)} samples)")


if __name__ == "__main__":
    main()

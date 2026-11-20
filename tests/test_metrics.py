"""CPU-only checks for the benchmark eval — no GPU/model/network deps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from wmt26.metrics import score_chrf_pp

import run_benchmark_eval as rbe


def test_chrf_pp_identical_is_100():
    hyps = ["Hello there, general.", "A second line."]
    assert score_chrf_pp(hyps, list(hyps)) == 100.0


def test_normalize_folds_fullwidth_comma():
    # video_title uses a full-width comma; CSV episode uses ASCII -> must match.
    assert rbe.normalize_key("第3集：对话，但对") == rbe.normalize_key("第3集:对话,但对")


def test_gt_join_resolves_89_episodes():
    root = Path(__file__).resolve().parents[1]
    gt_map = rbe.load_gt_map(root / "data" / "gt" / "data.csv")
    matched = sum(1 for _vid, vt in rbe.iter_test_videos(root / "data" / "tests") if vt in gt_map)
    assert matched == 89, matched

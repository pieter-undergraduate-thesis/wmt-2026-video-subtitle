from wmt26.constraints import check_constraints, split_cue_if_needed, violation_rate
from wmt26.subtitle_io import Cue


def test_compliant_cue_passes():
    assert check_constraints("short line", duration_s=3.0) == []


def test_long_line_flagged():
    long_text = "x" * 60
    v = check_constraints(long_text, duration_s=10.0)
    assert any("line too long" in s for s in v)


def test_too_many_lines_flagged():
    v = check_constraints("a\nb\nc", duration_s=10.0)
    assert any("too many lines" in s for s in v)


def test_reading_speed_flagged():
    # 40 chars in 1s = 40 CPS, well over the 17 CPS cap
    v = check_constraints("x" * 40, duration_s=1.0)
    assert any("CPS" in s for s in v)


def test_split_compliant_returns_one():
    cue = Cue(1, 0, 3000, "short")
    assert split_cue_if_needed(cue) == [cue]


def test_split_returns_two_contiguous_cues():
    long_text = "word " * 20  # 100 chars, one line -> too long
    cue = Cue(5, 1000, 5000, long_text.strip())
    out = split_cue_if_needed(cue)
    assert len(out) == 2
    first, second = out
    # contiguous timing: no gap, no overlap
    assert first.end_ms == second.start_ms
    assert first.start_ms == cue.start_ms
    assert second.end_ms == cue.end_ms
    # text preserved across the split (modulo the boundary space)
    assert (first.text + " " + second.text).split() == cue.text.split()


def test_violation_rate():
    cues = [Cue(1, 0, 3000, "ok"), Cue(2, 0, 1000, "x" * 50)]
    assert violation_rate(cues) == 0.5
    assert violation_rate([]) == 0.0

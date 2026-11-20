from pathlib import Path

import pytest

from wmt26 import termdb
from wmt26.metrics import term_recall
from wmt26.postedit import postedit_cues
from wmt26.subtitle_io import Cue

DB = termdb.load(Path(__file__).parent / "data" / "termdb_sample.jsonl")


def fixed(reply):
    """complete_fn that always returns `reply`, recording every prompt it saw."""
    seen = []

    def fn(client, prompt, model, temperature=None):
        seen.append(prompt)
        return reply

    return fn, seen


def test_unflagged_cues_pass_through_and_never_reach_the_model():
    src = [Cue(1, 0, 3000, "今天天气很好"), Cue(2, 3000, 6000, "我们走吧")]
    hyp = [Cue(1, 0, 3000, "Nice weather."), Cue(2, 3000, 6000, "Let's go.")]
    fn, seen = fixed("SHOULD NOT BE USED")

    out, st = postedit_cues(src, hyp, "en", DB, client=object(), complete_fn=fn)

    assert seen == []  # selectivity: no DB match -> no LLM call at all
    assert [c.text for c in out] == ["Nice weather.", "Let's go."]
    assert (st.cues, st.flagged, st.edited, st.accepted) == (2, 0, 0, 0)


def test_flagged_cue_is_edited_and_keeps_index_and_timing():
    src = [Cue(7, 1000, 4000, "他重出江湖了")]
    hyp = [Cue(7, 1000, 4000, "He came out of rivers and lakes again.")]
    fn, seen = fixed("He returned to the martial world.")

    out, st = postedit_cues(src, hyp, "en", DB, client=object(), complete_fn=fn)

    assert out[0].text == "He returned to the martial world."
    assert (out[0].index, out[0].start_ms, out[0].end_ms) == (7, 1000, 4000)
    assert (st.flagged, st.edited, st.accepted) == (1, 1, 1)


def test_matched_terms_reach_the_prompt():
    src = [Cue(1, 0, 3000, "陛下驾到")]
    hyp = [Cue(1, 0, 3000, "The emperor arrives.")]
    fn, seen = fixed("His Majesty arrives.")

    postedit_cues(src, hyp, "id", DB, client=object(), complete_fn=fn)

    prompt = seen[0]
    assert "陛下" in prompt
    assert "Yang Mulia" in prompt  # target-lang equivalent, given exactly
    assert "under the steps" in prompt  # the avoid-list literal
    assert "Your Majesty" not in prompt  # other langs' equivalents stay out


def test_gate_rejects_an_edit_that_drops_a_term_the_draft_had():
    src = [Cue(1, 0, 5000, "江湖险恶")]
    hyp = [Cue(1, 0, 5000, "The martial world is treacherous.")]
    fn, _ = fixed("Rivers and lakes are dangerous.")  # loses the required rendering

    out, st = postedit_cues(src, hyp, "en", DB, client=object(), complete_fn=fn)

    assert out[0].text == "The martial world is treacherous."  # draft kept
    assert (st.edited, st.accepted) == (1, 0)


def test_gate_rejects_an_edit_that_leaks_han():
    src = [Cue(1, 0, 5000, "江湖险恶")]
    hyp = [Cue(1, 0, 5000, "The martial world is treacherous.")]
    fn, _ = fixed("The 江湖 is treacherous.")

    out, st = postedit_cues(src, hyp, "en", DB, client=object(), complete_fn=fn)

    assert out[0].text == "The martial world is treacherous."
    assert (st.edited, st.accepted) == (1, 0)


def test_gate_rejects_an_edit_that_adds_a_length_violation():
    src = [Cue(1, 0, 5000, "江湖险恶")]
    hyp = [Cue(1, 0, 5000, "The martial world is treacherous.")]
    # keeps the term, but blows past MAX_CHARS_PER_LINE (42)
    fn, _ = fixed("The martial world is treacherous " + "and perilous " * 8)

    out, st = postedit_cues(src, hyp, "en", DB, client=object(), complete_fn=fn)

    assert out[0].text == "The martial world is treacherous."
    assert (st.edited, st.accepted) == (1, 0)


def test_identical_reply_is_not_counted_as_an_edit():
    src = [Cue(1, 0, 5000, "江湖险恶")]
    hyp = [Cue(1, 0, 5000, "The martial world is treacherous.")]
    fn, _ = fixed("The martial world is treacherous.")

    out, st = postedit_cues(src, hyp, "en", DB, client=object(), complete_fn=fn)

    assert (st.flagged, st.edited, st.accepted) == (1, 0, 0)


def test_qe_gate_drops_edits_that_score_below_the_draft():
    src = [Cue(1, 0, 5000, "江湖险恶")]
    hyp = [Cue(1, 0, 5000, "The martial world is treacherous.")]
    fn, _ = fixed("Jianghu is treacherous.")  # passes the deterministic gate

    # _qe_filter asks for [draft..., edit...]; score the edit worse than the draft.
    def score_fn(srcs, mts):
        return [0.9, 0.1]

    out, st = postedit_cues(
        src, hyp, "en", DB, client=object(), complete_fn=fn,
        qe_gate=True, score_fn=score_fn,
    )

    assert out[0].text == "The martial world is treacherous."
    assert (st.edited, st.accepted) == (1, 0)


def test_cue_count_mismatch_is_a_clear_error():
    src = [Cue(1, 0, 1000, "江湖"), Cue(2, 1000, 2000, "陛下")]
    hyp = [Cue(1, 0, 1000, "jianghu")]
    with pytest.raises(ValueError, match="cue count mismatch"):
        postedit_cues(src, hyp, "en", DB, client=object(), complete_fn=fixed("x")[0])


def test_term_recall_counts_required_renderings():
    srcs = ["江湖险恶", "陛下驾到", "今天天气很好"]
    hyps = ["The martial world is treacherous.", "The emperor arrives.", "Nice weather."]
    # 2 required (江湖, 陛下); 1 hit (the martial world). Clean line constrains nothing.
    assert term_recall(srcs, hyps, DB, "en") == pytest.approx(0.5)


def test_term_recall_is_zero_when_nothing_is_required():
    assert term_recall(["今天天气很好"], ["Nice weather."], DB, "en") == 0.0

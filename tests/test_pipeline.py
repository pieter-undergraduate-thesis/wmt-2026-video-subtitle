from wmt26.pipeline import translate_cues
from wmt26.subtitle_io import Cue


def fake_translate(text, target, context=None, *, examples=None, **kw):
    return f"[{target}] {text}"


def test_pipeline_preserves_cues_and_reindexes():
    cues = [
        Cue(1, 0, 2000, "hello"),
        Cue(2, 2000, 4000, "sentence here"),
        Cue(3, 4000, 6000, "bye"),
    ]
    out = translate_cues(cues, "en", translate_fn=fake_translate)

    # cues stay 1:1 with source, timings and indices preserved
    assert [c.index for c in out] == [1, 2, 3]
    assert [(c.start_ms, c.end_ms) for c in out] == [(0, 2000), (2000, 4000), (4000, 6000)]


def test_pipeline_uses_target():
    cues = [Cue(1, 0, 2000, "hi")]
    out = translate_cues(cues, "th", translate_fn=fake_translate)
    assert out[0].text == "[th] hi"


def test_pipeline_feeds_recent_cues_as_examples():
    seen = []

    def spy(text, target, context=None, *, examples=None, **kw):
        seen.append(list(examples or []))
        return f"T:{text}"

    cues = [Cue(1, 0, 1000, "a"), Cue(2, 1000, 2000, "b"), Cue(3, 2000, 3000, "c")]
    translate_cues(cues, "en", translate_fn=spy, window=4)

    # cue 1: no examples; cue 2: sees (a->T:a); cue 3: sees a and b
    assert seen[0] == []
    assert seen[1] == [("a", "T:a")]
    assert seen[2] == [("a", "T:a"), ("b", "T:b")]

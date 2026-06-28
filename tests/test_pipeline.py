from wmt26.pipeline import translate_cues
from wmt26.subtitle_io import Cue


def fake_translate(text, target, context=None, **kw):
    # echo a long string for one cue to force a split, short otherwise
    return "x" * 60 if "long" in text else f"[{target}] {text}"


def test_pipeline_reindexes_after_split():
    cues = [
        Cue(1, 0, 2000, "hello"),
        Cue(2, 2000, 4000, "long sentence here"),  # -> splits into 2
        Cue(3, 4000, 6000, "bye"),
    ]
    out = translate_cues(cues, "en", translate_fn=fake_translate)

    # 3 in, one split -> 4 out, indices contiguous 1..4
    assert [c.index for c in out] == [1, 2, 3, 4]
    # timings monotonic non-decreasing
    starts = [c.start_ms for c in out]
    assert starts == sorted(starts)


def test_pipeline_uses_target():
    cues = [Cue(1, 0, 2000, "hi")]
    out = translate_cues(cues, "th", translate_fn=fake_translate)
    assert out[0].text == "[th] hi"

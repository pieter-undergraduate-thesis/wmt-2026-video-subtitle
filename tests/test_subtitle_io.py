from wmt26.subtitle_io import Cue, parse_srt, write_srt

SAMPLE = """1
00:00:01,000 --> 00:00:03,000
你怎麼可以這樣對我

2
00:00:03,500 --> 00:00:05,000
line one
line two
"""


def test_roundtrip(tmp_path):
    src = tmp_path / "in.srt"
    src.write_text(SAMPLE, encoding="utf-8")

    cues = parse_srt(str(src))
    assert len(cues) == 2
    assert cues[0].start_ms == 1000 and cues[0].end_ms == 3000
    assert cues[0].text == "你怎麼可以這樣對我"
    assert cues[1].line_count == 2

    out = tmp_path / "out.srt"
    write_srt(cues, str(out))
    again = parse_srt(str(out))

    assert len(again) == len(cues)
    for a, b in zip(cues, again):
        assert (a.start_ms, a.end_ms, a.text) == (b.start_ms, b.end_ms, b.text)


def test_bom_is_stripped(tmp_path):
    src = tmp_path / "bom.srt"
    src.write_bytes(b"\xef\xbb\xbf" + SAMPLE.encode("utf-8"))
    cues = parse_srt(str(src))
    assert cues[0].index == 1  # BOM didn't corrupt the first index

import json

from wmt26.metadata import synopsis_for_srt


def _write_pair(tmp_path, vid, video_title):
    (tmp_path / f"{vid}_zh.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")
    (tmp_path / f"{vid}_metadata.json").write_text(
        json.dumps([{"album_title": "影后的复仇", "album_desc": "沈晚意复仇的故事", "video_title": video_title}]),
        encoding="utf-8",
    )
    return tmp_path / f"{vid}_zh.srt"


def test_synopsis_includes_title_episode_desc(tmp_path):
    srt = _write_pair(tmp_path, "b0046em36cy", "影后的复仇_16")
    out = synopsis_for_srt(srt)
    assert "影后的复仇" in out
    assert "第16集" in out
    assert "沈晚意复仇的故事" in out


def test_variety_title_omits_episode(tmp_path):
    srt = _write_pair(tmp_path, "k0046nulhyw", "陪你看半熟第9期：宝儿老孟复盘")
    out = synopsis_for_srt(srt)
    assert "第" not in out.split("\n")[0].replace("剧名", "")  # no （第N集）
    assert "影后的复仇" in out  # title still present


def test_missing_metadata_returns_none(tmp_path):
    (tmp_path / "x_zh.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")
    assert synopsis_for_srt(tmp_path / "x_zh.srt") is None

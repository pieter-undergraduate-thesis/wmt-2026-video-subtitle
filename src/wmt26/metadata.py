"""Per-video metadata paired with each source .srt, as a translation-context block.

Pairing convention (see data/): <vid>_zh.srt  <->  <vid>_metadata.json in the same dir.
Each metadata file is a 1-element JSON array; album_desc is the real per-video synopsis.
This is the WMT26 metadata hook prompts.context_prompt was written for.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _meta_path(srt_path: str | Path) -> Path:
    p = Path(srt_path)
    vid = p.stem.rsplit("_", 1)[0]  # "a0046jl5d6d_zh" -> "a0046jl5d6d"
    return p.with_name(f"{vid}_metadata.json")


def load_metadata(srt_path: str | Path) -> dict | None:
    """The metadata object paired with an .srt, or None if there's no file."""
    mp = _meta_path(srt_path)
    if not mp.exists():
        return None
    data = json.loads(mp.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data[0] if data else None
    return data or None


def synopsis_for_srt(srt_path: str | Path) -> str | None:
    """Context block (title + episode + desc) from metadata, or None if absent.

    ponytail: episode parsed as a trailing _<digits> in video_title. Variety-show
    titles embed it differently (第9期) and simply omit the episode line — upgrade
    the regex only if those need episode awareness.
    """
    meta = load_metadata(srt_path)
    if not meta:
        return None
    title = (meta.get("album_title") or "").strip()
    desc = (meta.get("album_desc") or "").strip()
    ep = re.search(r"_(\d+)$", meta.get("video_title") or "")
    if title and ep:
        head = f"剧名：{title}（第{ep.group(1)}集）"
    elif title:
        head = f"剧名：{title}"
    else:
        head = ""
    parts = [x for x in (head, f"简介：{desc}" if desc else "") if x]
    return "\n".join(parts) or None

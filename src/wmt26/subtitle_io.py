"""Shared subtitle I/O contract.

Every pipeline stage is a `list[Cue] -> list[Cue]` transform. Both the
benchmarking track and the inference pipeline read/write through here so the
data structures never drift apart.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

import srt


@dataclass
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str
    line_count: int = 0

    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0


def parse_srt(path: str) -> list[Cue]:
    """Parse an .srt file into Cue objects.

    Uses the `srt` package rather than a hand-rolled regex — it already handles
    timestamp-format edge cases and BOM/encoding (utf-8-sig below strips a BOM).
    """
    with open(path, encoding="utf-8-sig") as f:
        subs = list(srt.parse(f.read()))
    return [
        Cue(
            index=s.index,
            start_ms=int(s.start.total_seconds() * 1000),
            end_ms=int(s.end.total_seconds() * 1000),
            text=s.content,
            line_count=s.content.count("\n") + 1,
        )
        for s in subs
    ]


def write_srt(cues: list[Cue], path: str) -> None:
    subs = [
        srt.Subtitle(
            index=c.index,
            start=datetime.timedelta(milliseconds=c.start_ms),
            end=datetime.timedelta(milliseconds=c.end_ms),
            content=c.text,
        )
        for c in cues
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

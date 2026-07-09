"""Hy-MT2 zero-shot prompt templates.

Mirrors Hy-MT2's documented instruction scenarios. Rules from the model card:
- No system prompt; the instruction goes in the user turn.
- Use full language names ("Indonesian", not "id") — abbreviations hurt
  instruction-following.
"""
from __future__ import annotations

# WMT26 target languages -> full names for the prompt.
LANG_NAMES = {
    "en": "English",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "zh-TW": "Traditional Chinese",
}


def lang_name(target: str) -> str:
    """Accept either a short code or an already-full name."""
    return LANG_NAMES.get(target, target)


def default_prompt(source_text: str, target: str) -> str:
    return (
        f"Translate the following text into {lang_name(target)}. Output only the "
        f"translated result, with no additional explanation:\n{source_text}"
    )


def context_prompt(source_text: str, target: str, context: str) -> str:
    """Genre/synopsis-aware translation — the WMT26 metadata hook."""
    return (
        "Background information for context only — do not translate this part:\n"
        f"{context}\n\n"
        f"Using the background above, translate the following text into "
        f"{lang_name(target)}. Output only the translation, no extra "
        f"explanation:\n{source_text}"
    )


def few_shot_prompt(
    source_text: str,
    target: str,
    examples: list[tuple[str, str]],
    context: str | None = None,
) -> str:
    """In-context learning: show src->target example pairs before the real cue.

    The pipeline feeds the video's own already-translated cues here, so style
    and terminology stay consistent down the episode.
    """
    shots = "\n".join(f"{src} -> {tgt}" for src, tgt in examples)
    ctx = f"Background information for context only:\n{context}\n\n" if context else ""
    return (
        f"{ctx}Here are example translations into {lang_name(target)}:\n{shots}\n\n"
        f"Now translate the following text into {lang_name(target)} in the same "
        f"style. Output only the translated result, with no additional "
        f"explanation:\n{source_text}"
    )


def terminology_prompt(source_text: str, target: str, terms: dict[str, str]) -> str:
    """Character names / recurring proper nouns pinned to fixed translations."""
    glossary = "\n".join(f"{k} -> {v}" for k, v in terms.items())
    return (
        "Reference terminology (use these exact translations when these terms "
        f"appear):\n{glossary}\n\n"
        f"Translate the following text into {lang_name(target)}. Output only the "
        f"translated result, with no additional explanation:\n{source_text}"
    )

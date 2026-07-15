"""Hy-MT2 zero-shot prompt templates.

Mirrors Hy-MT2's documented instruction scenarios. Rules from the model card:
- No system prompt; the instruction goes in the user turn.
- Use full language names ("Indonesian", not "id") — abbreviations hurt
  instruction-following.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .termdb import Entry

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


# Register/formality note per target. zh->id is diglossic and register is the
# #1 quality lever there (see docs/research.md); other targets need none yet.
REGISTER_NOTES = {
    "id": (
        "Use colloquial-but-clean Indonesian: kamu/aku among peers, Anda or "
        "Bapak/Ibu for elders/superiors; capitalize Anda."
    ),
}


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


def advanced_prompt(
    source_text: str,
    target: str,
    *,
    context: str | None = None,
    glossary: dict[str, str] | None = None,
    examples: list[tuple[str, str]] | None = None,
    register: str | None = None,
) -> str:
    """One user turn combining every optional signal the advanced pipeline has.

    Reduces to `default_prompt` when all extras are None. Kept separate from the
    single-purpose prompts above so zero-shot output stays byte-stable.

    ponytail: additive builder; don't refactor the working prompts into it.
    """
    parts: list[str] = []
    if examples:
        shots = "\n".join(f"{src} -> {tgt}" for src, tgt in examples)
        parts.append(f"Example translations (for style reference):\n{shots}")
    if glossary:
        terms = "\n".join(f"{k} -> {v}" for k, v in glossary.items())
        parts.append(
            "Reference terminology (use these exact translations when these "
            f"terms appear):\n{terms}"
        )
    if context:
        parts.append(
            "Background information for context only — do not translate this "
            f"part:\n{context}"
        )
    instruction = f"Translate the following text into {lang_name(target)}."
    if register:
        instruction += f" {register}"
    instruction += " Output only the translation, no extra explanation:"
    parts.append(f"{instruction}\n{source_text}")
    return "\n\n".join(parts)


def glossary_prompt(source_text: str, target: str, synopsis: str | None = None) -> str:
    """Ask the model to extract an episode glossary as `源词 -> rendering` lines."""
    syn = f"Synopsis:\n{synopsis}\n\n" if synopsis else ""
    return (
        f"{syn}Below is the full subtitle text of one episode. List the recurring "
        "proper nouns that must stay consistent across the episode — character "
        "names, sect/organization names, titles, place names, and recurring "
        f"idioms (chengyu) — with one fixed {lang_name(target)} rendering each. "
        "Output only lines in the form `源词 -> rendering`, one per line, no "
        f"other text:\n{source_text}"
    )


def _terms_block(terms: list["Entry"], target: str) -> str:
    """Format matched term-DB entries for the post-edit prompt.

    Only the target language's equivalents are shown — the other languages are
    noise the model would have to ignore. Entries with no equivalent for this
    target still earn their place if they carry a figurative meaning: meaning
    injection alone is IdiomKB's core lever.
    """
    lines: list[str] = []
    for t in terms:
        eqs = t.equivalents.get(target) or []
        if not eqs and not t.meaning_en:
            continue
        head = f"- {t.zh}"
        if t.meaning_en:
            head += f" [{t.meaning_en}]"
        meta: list[str] = []
        if t.category:
            meta.append(f"category={t.category}")
        if eqs:
            meta.append(f'{lang_name(target)}="{" / ".join(eqs)}"')
        if t.avoid:
            meta.append("avoid: " + ", ".join(f'"{a}"' for a in t.avoid))
        lines.append(f"{head} {'; '.join(meta)}" if meta else head)
    return "\n".join(lines)


def postedit_prompt(
    source_text: str,
    draft: str,
    target: str,
    *,
    register: str | None = None,
    terms: list["Entry"] | None = None,
) -> str:
    """Single refine pass: adequacy, register, glossary compliance, brevity.

    `terms` are the term-DB entries matched in *this* source line — per-segment
    injection only, so the prompt stays clean. The exact required rendering is
    given rather than asked for: per Translate-and-Revise (arXiv:2407.13164),
    feeding rule-detected constraints is what drives constraint completion up.
    Output is byte-stable with the no-terms version when `terms` is empty/None.
    """
    reg = f" {register}" if register else ""
    block = _terms_block(terms, target) if terms else ""
    rule = (
        " Use the GLOSSARY below for the terms it lists: prefer the given "
        "rendering and avoid the literal ones. If the draft is already correct, "
        "return it unchanged."
        if block
        else ""
    )
    gloss = f"GLOSSARY (terms found in this line):\n{block}\n\n" if block else ""
    return (
        f"You are proofreading a {lang_name(target)} subtitle translation. Given "
        "the source and a draft, produce an improved translation: fix any "
        "inaccuracy or omission, keep it concise for on-screen reading, and "
        f"ensure natural {lang_name(target)}.{reg}{rule} Output only the revised "
        f"translation, no explanation:\n\n"
        f"{gloss}"
        f"Source: {source_text}\n"
        f"Draft: {draft}"
    )

# WMT26 Video Subtitle Translation — Task Brief

One-page explanation of what we're competing in, why the model/approach was
chosen, and what "done" means. For the module-level implementation, see
[PIPELINE.md](PIPELINE.md); for who owns what, see [TEAM.md](TEAM.md).

## The task, in one paragraph

WMT26 is running a shared task on translating video subtitles for five
Chinese-source language pairs. We're given Chinese subtitle files plus each
video's synopsis/metadata, and must produce translated `.srt` files that are
both linguistically correct *and* fit on-screen (line length, line count,
reading speed). It's a constrained track: only certain kinds of resources and
model sizes are allowed, and the organizer is Tencent's own Hunyuan MT team —
which is why they recommend their own model as the base.

## Rules that shape every decision below

| Constraint | Detail | Why it matters here |
|---|---|---|
| Language pairs | zh (Simplified) → en, th, id, ms, zh (Traditional) | One source language, five targets — a multilingual base model is a much better fit than five bilingual ones |
| Track | Constrained only | No arbitrary external data/models — has to be justifiable |
| Parameter cap | **≤20B total params**, final submission. MoE counts *total* params (not active); an ensemble sums all members' weights | Rules out Hy-MT2's 30B MoE variant outright; the dense 7B has huge headroom under the cap even combined with the 1.8B |
| Weight release | Required, under a license permitting unrestricted non-commercial use | **Open risk** — Hy-MT2's default license has regional restrictions; needs confirming before we can ship a fine-tuned derivative (see TEAM.md open items) |
| Test data release | Jul 1, 2026 | Already past — pipeline had to be baseline-ready before this date |
| Submission deadline | Jul 15, 2026 (AoE) | Hard deadline for `.srt` output, not the paper |
| Submission format | `.srt`, named `vid_langshort.srt` | Directly what `write_srt` + the script naming convention produce |
| Available at test time | Video synopsis/description, source subtitle text | This is the `<vid>_metadata.json` file paired with each `<vid>_zh.srt` — see `metadata.py` |

## Why Hy-MT2-7B

- **Organizer-recommended and prize-eligible** — Tencent Hunyuan is co-running
  the task and explicitly points at their own model.
- **Already covers all 5 target languages** from pretraining (33-language
  coverage), so this is genuinely zero-shot multilingual, not a gap we have
  to train around.
- **Comfortably under the 20B cap** (7B, or 8.8B combined with the 1.8B
  fallback as an ensemble) — leaves room to fine-tune without touching the
  ceiling.
- Ships with a documented LoRA/full-fine-tune recipe (LLaMA-Factory +
  DeepSpeed) and full serving support (`transformers`, vLLM, SGLang,
  llama.cpp/GGUF) — no custom training or serving code needed, just wrapper
  scripts (see `finetune/`, `scripts/serve.sh`).
- Has a **native "context-based translation" instruction mode** — i.e. the
  model already knows how to use a synopsis/genre block if you hand it one in
  the prompt. That's the direct hook for the video metadata WMT26 gives us,
  and for Zahra's SLM context-reasoning layer upstream of it. We may not need
  heavy fine-tuning just to get context injection working.

Hy-MT2-1.8B (GGUF-quantized) is kept as a fallback: CPU-runnable, cheap
insurance if the 7B server falls over near the deadline, and a legitimate
lightweight ensemble member if that turns out to help.

## The three phases

**Phase A — zero-shot baseline.** Get Hy-MT2-7B talking to our subtitle
pipeline with no fine-tuning: parse `.srt`, build a prompt (with or without
metadata context), call the model, write `.srt` back out, score it
(BLEU/chrF/COMET) and check it for format violations. This is the safety net
— a submittable system that exists *before* trying anything riskier.

**Phase B — fine-tuning.** LoRA (not full fine-tune — cheaper, faster to
iterate, and the model card's LoRA recipe is single-GPU feasible) on
subtitle-domain data, reformatted so the *instruction itself* teaches length
constraints, not just post-processing. zh↔en subtitle corpora exist; th/id/ms
don't have an equivalent parallel corpus yet, so those three pairs are left
zero-shot unless the measured gap (from Phase A eval) proves large enough to
justify building synthetic/pivot data. Idiomatic translation is specifically
upweighted — it's the team's identified biggest quality differentiator and
weakest point for automatic metrics.

**Phase C — inference & integration pipeline.** Everything documented in
[PIPELINE.md](PIPELINE.md): serving, per-cue context assembly (metadata +
rolling translated-cue window), constraint checking and splitting, SRT
writing. Designed so a slow/broken upstream context source (Zahra's SLM
layer) degrades gracefully to the no-context prompt instead of blocking
output.

## What "done" looks like

A `.srt` file per video per target language, translated by Hy-MT2 (zero-shot
and/or LoRA-adapted depending on language pair), passing format QC, submitted
by Jul 15. Model weights released under a compliant license. Everything
upstream of the deadline is in service of de-risking that one submission —
which is why Phase A exists as a standalone, always-working fallback rather
than something the later phases replace outright.

## Open risks (tracked in TEAM.md)

1. Hy-MT2 license vs. WMT26's release requirement — unconfirmed.
2. th/id/ms data gap — deliberately unaddressed until Phase A measures
   whether it's actually a problem.
3. Official subtitle constraint spec (CPL/lines/CPS) not yet published —
   pipeline runs on Netflix-style placeholders in `constraints.py` until then.

# Implementation Plan: Training-Free Quality Pipeline for WMT26 Subtitle MT (zh→en/id)

*Derived from [`docs/research.md`](./research.md), mapped onto the current codebase. Planning document only — nothing here is executed yet.*

## Context
`docs/research.md` concludes the biggest training-free wins over the current zero-shot single-pass baseline are, in priority order:

1. **N-candidate generation + fusion/rerank** (≈ +2–5% XCOMET) — the highest-ROI move.
2. **Context-aware windowed prompting + metadata injection** (≈ +0.5–1 COMET; already half-built in the repo).
3. **Episode glossary enforcement** — large qualitative gain for xianxia/wuxia and zh→id consistency.
4. **Few-shot retrieval** (especially zh→id) and **a single post-edit pass**.

Hard constraint: the submitted system must be **open-weight, < 20B total parameters**, so the fusion/rerank scorer stays an open QE model (CometKiwi) — no GPT/Claude/Gemini inside the pipeline.

We already have the measurement loop: `scripts/run_benchmark_eval.py` scores any output directory against the ground truth over 89 episodes (BLEU / chrF / chrF++, `--neural` for XCOMET + CometKiwi). That is the dev harness every stage is validated on. Baseline to beat = `result/out_fixed` (en corpus BLEU 16.49 / chrF 39.11).

## Current code to build on (reuse, don't duplicate)
- `src/wmt26/translate.py` — `_client` / `_complete`, `translate_cue`, `SAMPLING` / `EXTRA_BODY`, `DEFAULT_MODEL` / `DEFAULT_BASE_URL`. The injectable `client=` seam lets one OpenAI/vLLM client be reused across N candidate calls.
- `src/wmt26/prompts.py` — `default_prompt`, `context_prompt`, and an **already-written-but-unused `terminology_prompt`** (the glossary injection hook).
- `src/wmt26/context.py` — `RecentContext` (window of *translated* cues) + `build_context` (synopsis + recent, with timeout fallback).
- `src/wmt26/metadata.py` — `synopsis_for_srt` (title + episode + `album_desc`); metadata also carries character info per the task.
- `src/wmt26/eval.py::score_comet_qe` — CometKiwi QE, reused as the reranker utility.
- `scripts/zero_shot_translate.py` — output naming `<vid>_<lang>.srt`; the new pipeline mirrors it so the eval harness scores it unchanged.

## Recommended approach — phased

### Phase 1 (core, highest ROI): candidates + QE-rerank + windowed metadata context
Default fusion = **QE-rerank with CometKiwi** (no second model, stays in the parameter budget). Leave a clean seam to swap in Hunyuan-MT-Chimera-7B fusion later for an A/B on the harness.

**New `src/wmt26/candidates.py`**
- `generate_candidates(text, target, context, *, n=6, temps=(0.6,0.7,0.8,0.9,1.0,1.1), model, client) -> list[str]` — the Chimera recipe: same prompt, varied temperature via a per-call `SAMPLING` override; reuse `translate._complete`. Dedup identical strings. One shared `client`.

**New `src/wmt26/rerank.py`**
- `QEScorer` — lazy-loads CometKiwi **once**. (The current `score_comet_qe` reloads the checkpoint on every call; a cached scorer is required for O(N×cues) cost.) `score(srcs, mts) -> list[float]`.
- `offtarget_filter(cands, target) -> list[str]` — language-ID guard (drop candidates containing Han characters for zh→id; the doc flags CometKiwi being fooled by code-switching as the #1 zh→id risk). Keep all if the filter would empty the set.
- `pick_best(src, cands, scorer, target) -> str` — filter → QE-score → argmax.

**Extend `src/wmt26/context.py`**
- Add a prior **source**-line window (research wants source context; current code only keeps translated cues) plus a character/register block: `build_window_context(synopsis, prior_src_lines, glossary_terms, register_note)`. For zh→id, add a register instruction keyed to the character list (informal *kamu/aku* vs *Anda/Bapak/Ibu*) in `prompts.py`.

**New `scripts/run_advanced_pipeline.py`** — Stage 0→3 orchestration
- Parse SRT (keep timecodes out of the model, reuse `subtitle_io`), sliding window (~15 lines + 5–10 prior source lines), inject `synopsis_for_srt` + glossary terms, generate N candidates, QE-rerank, re-attach timecodes, write `<vid>_<lang>.srt`. Then score with `run_benchmark_eval.py`.

### Phase 2 (stretch): episode glossary
**New `src/wmt26/glossary.py`** — `build_glossary(cues, synopsis, target, *, client) -> dict` (one LLM pass extracting character names / sects / titles / recurring chengyu, one agreed rendering each; seed from metadata) and `enforce(text, glossary)` verified find/replace as the Stage-5 consistency pass. Inject per-window via the existing `terminology_prompt`.

### Phase 3 (stretch): post-edit + few-shot
- **`src/wmt26/postedit.py`** — a single TEaR/Proofreader-style refine pass (adequacy, register, glossary, brevity). **One iteration only** (the doc warns over-refinement degrades dev COMET). Disable if dev COMET drops.
- **zh→id few-shot** — FAISS index over OpenSubtitles zh–id, retrieve top-5 per line. Heaviest item; do last, only if Phase 1–2 dev gains plateau.

## Compliance / anti-gaming guardrails (from the research doc)
- Every pipeline model open-weight, sum < 20B. QE-rerank + base 7B is within budget; adding Chimera means accounting all three models.
- Rerank utility (CometKiwi) ≠ the metric validated on (XCOMET / chrF via the harness), to avoid overfitting one QE family. Always check chrF + off-target rate on the dev slice.
- Cap N at 8; stop post-edit after one pass; never let the model touch timecodes.

## Verification (dev loop on the 89-episode harness)
1. Baseline already scored (`result/out_fixed`). Record en/id XCOMET + chrF as the number to beat (`run_benchmark_eval.py --neural`).
2. After Phase 1: `python scripts/run_advanced_pipeline.py --in data/tests --out result/out_p1 --langs en id`, then `run_benchmark_eval.py --hyp result/out_p1 --neural` — expect XCOMET up; watch chrF and the zh→id off-target rate. Per the doc: < 0.5 XCOMET gain ⇒ widen the temperature spread / add a second base model.
3. Each later phase is re-scored the same way; keep a stage only if it beats the previous on XCOMET without collapsing chrF.
4. Self-checks (CPU, no GPU): `candidates.generate_candidates` with an injected fake `translate_fn` returns N deduped strings; `rerank.offtarget_filter` drops a Han-containing zh→id candidate; `glossary.enforce` does exact replacement. Mirror the existing injected-`translate_fn` test style in `tests/test_pipeline.py`.

## Files
- **New (Phase 1):** `src/wmt26/candidates.py`, `src/wmt26/rerank.py`, `scripts/run_advanced_pipeline.py`.
- **New (Phase 2):** `src/wmt26/glossary.py`.
- **New (Phase 3):** `src/wmt26/postedit.py`.
- **Edit:** `src/wmt26/context.py` (+ source window), `src/wmt26/prompts.py` (+ register note, + fusion / glossary / post-edit prompts).
- **New tests:** `tests/test_candidates.py`, `tests/test_rerank.py`.
- **Unchanged (reused as-is):** `eval.py`, `run_eval.py`, `run_benchmark_eval.py`, `metrics.py`.

## Open decisions (defaults chosen; easy to flip)
- Fusion = QE-rerank first, Hunyuan-MT-Chimera-7B as a later A/B (vs. building Chimera now).
- Assumes vLLM + Hy-MT2-7B GPU and CometKiwi weights are available; Chimera + OpenSubtitles are treated as optional.
- Scope defaults to Phase 1 as the committed build; Phases 2–3 are stretch, subject to the 15-Jul window.

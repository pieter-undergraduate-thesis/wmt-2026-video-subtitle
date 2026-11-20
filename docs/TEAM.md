# Team Scaffold

Who owns what, the contracts between our pieces, and the open items nobody's
closed yet. See [TASK_BRIEF.md](TASK_BRIEF.md) for why we're doing any of
this, [PIPELINE.md](PIPELINE.md) for how the code works.

## Roles

| Owner | Track | Codebase surface |
|---|---|---|
| **Dzaki** | Phase A (zero-shot baseline, eval) + Phase B (LoRA fine-tuning) | `src/wmt26/eval.py`, `finetune/`, benchmark runs via `scripts/zero_shot_translate.py` / `scripts/run_eval.py` |
| **Pieter** | Shared infra + Phase C (inference/pipeline) | `src/wmt26/{subtitle_io,prompts,translate,constraints,context,pipeline,metadata}.py`, `scripts/serve.sh`, `scripts/run_pipeline.py` |
| *(unassigned)* | Advanced training-free quality pipeline (candidates/rerank/glossary/post-edit/few-shot) | `src/wmt26/{candidates,rerank,glossary,postedit,fewshot}.py`, `scripts/run_advanced_pipeline.py` — follows Pieter's Phase C surface, not formally assigned |
| **Zahra** | SLM context-reasoning layer (upstream input) | Plugs into `context.build_context` via `synopsis_provider` — see contract below. Not yet integrated. |

Nobody blocks anybody by default: Dzaki's fine-tuned checkpoint is just a
different `--model` arg to `serve.sh` / `translate.py`; Zahra's layer is an
optional callable that degrades to the static/no-context path if absent or
slow.

## Handoff contracts

These are the interfaces to code against — change them and you break someone
else's work, so treat edits here as cross-team, not local.

**`Cue` / `list[Cue] -> list[Cue]`** (`subtitle_io.py`)
Every stage — parsing, translating, writing — takes and returns `list[Cue]`.
This is the one structure Dzaki's eval scripts and Pieter's pipeline both
read/write. Don't add a second subtitle representation anywhere; extend `Cue`
instead if a new field is needed. **Note:** `pipeline.translate_cues` keeps
cues 1:1 with the source (no constraint-splitting, no re-indexing) — don't
assume downstream that its output count/indices can differ from the input.
`constraints.split_cue_if_needed` still exists and is still called, just from
`scripts/zero_shot_translate.py` directly, one layer above `pipeline.py`.

**`<vid>_zh.srt` + `<vid>_metadata.json` pairing** (`metadata.py`)
Source and metadata files share a video-id prefix in the same directory.
Metadata JSON is a 1-element array with `album_title`, `album_desc`,
`video_title` keys. `synopsis_for_srt(srt_path)` is the one place this
pairing convention is implemented — if the test-data layout differs from
this on Jul 1, fix it here, not in every script that calls it.

**`synopsis_provider` callable** (`context.py`)
Zahra's integration point: a zero-arg callable returning `str` (or raising).
`build_context(synopsis, recent, synopsis_provider=fn, timeout_s=5.0)` calls
it on a background thread with a hard timeout; any exception or timeout
falls through to the static `synopsis` arg (from `metadata.py`) or `None`.
**Whatever Zahra's layer becomes, it must satisfy this exact signature** —
zero args in, a string out, safe to call repeatedly per-cue. No shared
mutable state assumptions across calls.

**`translate_fn` injection** (`pipeline.py`)
`translate_cues(..., translate_fn=translate.translate_cue)` — anyone testing
pipeline orchestration (context threading, splitting, re-indexing) swaps this
for a fake and never needs a live GPU/server. Used by `tests/test_pipeline.py`
already; reuse the same pattern for new orchestration tests instead of
spinning up vLLM in CI.

## Milestone timeline

| Window | Focus | Owner |
|---|---|---|
| through Jul 1, 2026 | Phase A: zero-shot baseline working end-to-end, banked fallback | Dzaki (lead), Pieter (infra) |
| Jul 1, 2026 | Test data drops — run baseline on it immediately | Dzaki + Pieter |
| Jul 1 – Jul 7 | Phase B: LoRA fine-tuning, idiom upweighting | Dzaki |
| Jul 1 – Jul 7 (parallel) | Phase C: serving, constraint post-processor, metadata + rolling context | Pieter |
| Jul 7 – Jul 10 | Integration: fine-tuned checkpoint + Zahra's context layer + full pipeline, end-to-end test | All three |
| Jul 10 – Jul 13 | Full eval, error analysis, decide LoRA vs. zero-shot per language pair | All three |
| Jul 13 – Jul 15 | Final run on official test set, SRT QC, weight packaging, submission | All three |

Submission deadline: **Jul 15, 2026 (AoE)**.

## Open action items

| Item | Owner | Status |
|---|---|---|
| Confirm Hy-MT2 derivative weights can be released under WMT26's unrestricted-non-commercial license requirement | Pieter | **Blocker** — unconfirmed, contact `hunyuan@tencent.com` or read license text directly |
| Measure zero-shot COMET gap on th/id/ms from Phase A eval; only build pivot/synthetic data if the gap is real | Dzaki | Pending Phase A results |
| Swap `constraints.py` placeholder constants (42 CPL / 2 lines / 17 CPS) for WMT26's official spec | Pieter | Pending official spec publication |
| If ensembling 7B + 1.8B, confirm summed param count stays under the 20B cap (8.8B — should be fine, but verify at submission time) | Pieter | Not started |
| Wire `synopsis_provider` to Zahra's SLM layer once it exists | Zahra + Pieter | Not started |

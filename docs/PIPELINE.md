# Pipeline Documentation

Technical reference for `src/wmt26`. Read this before modifying any module — it
defines the contracts other modules (and the other two people on the team)
depend on.

## Data flow

```
                 ┌─────────────────────────┐
                 │  <vid>_zh.srt            │  source cues
                 │  <vid>_metadata.json     │  album_title / album_desc / video_title
                 └───────────┬──────────────┘
                             │ parse_srt()            metadata.synopsis_for_srt()
                             ▼                                   │
                        list[Cue]                                │
                             │                                   ▼
                             │              ┌────────────────────────────────┐
                             │              │ synopsis string (or None)       │
                             │              └───────────────┬────────────────┘
                             ▼                               │
                 ┌───────────────────────┐                   │
                 │  per cue, in order:    │◄──────────────────┘
                 │  build_context()       │  synopsis + last 3-5 translated cues
                 │  prompts.*_prompt()    │  default / context / terminology template
                 │  translate.translate_cue()  → vLLM (Hy-MT2-7B, OpenAI-compat API)
                 │  split_cue_if_needed() │  CPL/line/CPS check, split on overflow
                 └───────────┬─────────────┘
                             ▼
                        list[Cue]  (re-indexed)
                             │ write_srt()
                             ▼
                 ┌─────────────────────────┐
                 │  <vid>_<lang>.srt        │  submission format
                 └─────────────────────────┘
```

Everything upstream of `translate.translate_cue` and downstream of it is
model-agnostic — swapping Hy-MT2-7B for the 1.8B fallback or a LoRA adapter
touches only `translate.py` / `serve.sh`, nothing else.

## Module reference

### `subtitle_io.py` — the shared contract
`Cue(index, start_ms, end_ms, text, line_count)`. Every stage in the pipeline
is a `list[Cue] -> list[Cue]` transform; this is the one data structure both
the benchmarking track and the inference pipeline agree on. `parse_srt` /
`write_srt` wrap the `srt` package (handles BOM + timestamp edge cases —
don't hand-roll a parser here).

### `metadata.py` — per-video context lookup
WMT26 test data ships as `<vid>_zh.srt` + `<vid>_metadata.json` pairs in the
same directory. `synopsis_for_srt(srt_path)` finds the paired JSON, and
formats `album_title` (+ parsed episode number, when the title ends in
`_<digits>`) and `album_desc` into a short Chinese-language context block:

```
剧名：影后的复仇（第16集）
简介：沈晚意复仇的故事
```

Returns `None` if no metadata file exists — callers must handle that (they
do; see `pipeline.py` below). Variety-show titles (`陪你看半熟第9期：...`)
don't match the `_<digits>` suffix pattern, so they get the title line with
no episode — this is a known gap, not a bug, until a variety-show-specific
parser is needed.

### `prompts.py` — prompt templates
Three templates matching Hy-MT2's documented instruction scenarios:
`default_prompt` (no context), `context_prompt` (synopsis-aware — the WMT26
metadata hook), `terminology_prompt` (glossary pinning for character names).
No system prompt is used anywhere (Hy-MT2 has none by convention); the
instruction always goes in the user turn. Target languages are always
expanded to full names (`"Indonesian"`, not `"id"`) via `LANG_NAMES` —
abbreviations measurably hurt the model's instruction-following.

### `context.py` — rolling + synopsis context, with a fallback switch
`RecentContext` is a bounded deque of the last N *translated* cues, formatted
as a "for consistency only, do not re-translate" block — this is what gives
the model cross-cue coherence on long dialogue exchanges.

`build_context(synopsis, recent, synopsis_provider=None, timeout_s=5.0)` is
the single place context assembly happens. If a `synopsis_provider` callable
is passed (the hook point for Zahra's SLM context layer), it's called with a
hard timeout on a background thread; any exception or timeout falls straight
through to the static `synopsis` argument (or `None`) rather than blocking
the cue. **This is the fallback switch called out in the task doc** — it is
a literal `try/except`, not a retry/circuit-breaker framework, by design.

### `translate.py` — model inference
Thin OpenAI-client wrapper around vLLM's OpenAI-compatible server.
`translate_cue(text, target, context=None)` builds the right prompt and
calls the model; `translate_batch` is a sequential convenience wrapper (vLLM's
continuous batching overlaps the requests server-side, so no client-side
async fan-out is implemented — add it only if GPU utilization is measurably
bottlenecked by request issuance, not before). The `openai` import is lazy
(inside `_client()`), so `subtitle_io` / `constraints` / `pipeline` (with an
injected `translate_fn`) can be imported and tested without the `openai`
package installed.

Sampling params are fixed from the Hy-MT2 model card:
`temperature=0.7, top_p=0.6, top_k=20, repetition_penalty=1.05, max_tokens=512`.

### `constraints.py` — subtitle format QC
`check_constraints(text, duration_s)` returns violation strings for: too many
lines (>`MAX_LINES`), any line over `MAX_CHARS_PER_LINE`, or reading speed
over `MAX_CPS`. Current defaults (42 CPL / 2 lines / 17 CPS) are Netflix-style
industry convention — **placeholder until WMT26 publishes its own spec**; the
three constants at the top of the file are the single edit point when that
happens.

`split_cue_if_needed(cue)` splits an overflowing cue into two time-contiguous
cues (first keeps the original start, second keeps the original end, midpoint
in between, no gap/overlap) rather than truncating — truncation loses
meaning. The split point is the nearest word boundary to the character
midpoint. Known ceiling: this doesn't look at grammar (punctuation,
conjunctions) — upgrade only if a word-alignment/grammar-aware splitter
becomes necessary.

### `pipeline.py` — orchestration
`translate_cues(cues, target, synopsis=None, synopsis_provider=None, window=4,
translate_fn=translate.translate_cue)` is the loop: for each cue, assemble
context from what's been translated so far, translate, split if needed,
accumulate. After the loop, all output cues are re-indexed 1..N (splits break
the original 1:1 index mapping). `translate_fn` is injectable — this is what
lets `tests/test_pipeline.py` exercise the full orchestration logic (context
threading, splitting, re-indexing) with zero GPU / network dependency.

`translate_srt(in_path, out_path, target, ...)` is the file-level wrapper
used by `scripts/run_pipeline.py`.

### `eval.py` — scoring
`evaluate(srcs, hyp_cues, refs)` returns `Scores(bleu, chrf, comet,
violation_rate)`. BLEU/chrF via `sacrebleu`; COMET (`wmt22-comet-da`) is
lazy-imported (pulls torch + downloads a checkpoint on first call) and is the
**primary quality number** — it correlates better with human judgment than
BLEU on loose, idiom-heavy subtitle text. `violation_rate` reuses
`constraints.check_constraints` so format QC and translation-quality eval
always agree on what counts as a violation.

## Scripts (CLI entry points)

| Script | Purpose |
|---|---|
| `scripts/serve.sh` | Starts vLLM. `MAX_MODEL_LEN` / `GPU_MEMORY_UTILIZATION` env vars override defaults (8192 / 0.85). Currently `--tensor-parallel-size 2` — tuned for the team's 2-GPU box; the model's native 256K context is intentionally capped to 8192 since cues + rolling context never need more, and the full context needs ~8 GiB of KV cache that OOMs at higher GPU utilization. |
| `scripts/zero_shot_translate.py` | Batch baseline: every `<vid>_zh.srt` in `--in` (defaults to `data/`) → one `<vid>_<lang>.srt` per `--langs` entry. Auto-loads each video's own metadata for context unless `--synopsis` is passed manually. |
| `scripts/run_pipeline.py` | Full Phase C pipeline on a single file: rolling context + metadata + constraint splitting. Same metadata auto-load / manual-override behavior. |
| `scripts/run_eval.py` | Scores everything in `--hyp` against matching filenames in `--src`/`--ref`. |

## Fine-tuning (`finetune/`)

Not part of the runtime pipeline — a data converter (`convert_to_sharegpt.py`,
bakes format constraints into the instruction string) plus a thin
`llamafactory-cli` wrapper (`train_lora.sh`). LLaMA-Factory owns the actual
training loop; nothing here re-implements it. Output plugs back into
`serve.sh` via the commented `--enable-lora` lines.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

All 14 tests run without a GPU or a live model server:
- `test_subtitle_io.py` — parse/write roundtrip, BOM handling
- `test_constraints.py` — violation detection, split contiguity
- `test_pipeline.py` — orchestration with a fake `translate_fn`
- `test_metadata.py` — synopsis formatting, episode parsing, missing-file case

GPU-dependent paths (`translate.py` against a live server, `eval.py`'s COMET
scoring) are exercised manually via the scripts above, not in the test suite.

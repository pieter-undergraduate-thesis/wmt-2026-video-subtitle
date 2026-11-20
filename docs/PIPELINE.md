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
                 │  build_context()       │  synopsis + prior source-line window
                 │                        │  + recent (source, translation) pairs
                 │  translate.translate_cue()  → vLLM (Hy-MT2, OpenAI-compat API)
                 └───────────┬─────────────┘
                             ▼
                        list[Cue]  (1:1 with source, same index/timing)
                             │ write_srt()
                             ▼
                 ┌─────────────────────────┐
                 │  <vid>_<lang>.srt        │  submission format
                 └─────────────────────────┘
```

`translate_cues` (in `pipeline.py`) keeps cues 1:1 with the source — no
constraint-driven splitting or re-indexing. `constraints.split_cue_if_needed`
still exists and is still used, just one layer up: `scripts/zero_shot_translate.py`
calls it directly per translated line, outside the `pipeline.py` loop.

Everything upstream of `translate.translate_cue` and downstream of it is
model-agnostic — swapping models or a LoRA adapter touches only
`translate.py` / `serve.sh`, nothing else.

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
Base templates matching Hy-MT2's documented instruction scenarios:
`default_prompt` (no context), `context_prompt` (synopsis-aware — the WMT26
metadata hook), `few_shot_prompt` (src -> target example pairs, fed by
`RecentContext.pairs()`), `terminology_prompt` (glossary pinning for character
names). No system prompt is used anywhere (Hy-MT2 has none by convention); the
instruction always goes in the user turn. Target languages are always
expanded to full names (`"Indonesian"`, not `"id"`) via `LANG_NAMES` —
abbreviations measurably hurt the model's instruction-following.

`REGISTER_NOTES` is a per-target formality instruction dict (currently just
`id` — Indonesian's diglossia is the #1 zh->id quality lever per
`docs/research.md`). `advanced_prompt(source_text, target, *, context=None,
glossary=None, examples=None, register=None)` composes any subset of
few-shot examples / glossary terms / background context / register into one
prompt — `translate_cue` switches to it whenever `register` is set, and the
advanced pipeline (below) always uses it. `glossary_prompt` and
`postedit_prompt` are single-purpose prompts for `glossary.py` / `postedit.py`.

### `context.py` — rolling + synopsis context, with a fallback switch
`RecentContext` is a bounded deque of the last N *(source, translation)* cue
pairs (`.add(source_text, translated_text)`), not just translated text. Two
things read it: `.block()` formats a "for consistency only, do not
re-translate" string for `build_context`'s background block, and `.pairs()`
hands the same pairs to `translate.translate_cue(examples=...)` as few-shot
ICL demonstrations — the video's own already-translated cues teach the model
its own established style/terminology as it goes.

`build_context(synopsis, recent, prior_src=None, synopsis_provider=None,
timeout_s=5.0)` is the single place background-context assembly happens.
`prior_src` is the prior `src_window` *source* lines (set by `pipeline.py`,
not translated yet) — the model sees upcoming dialogue it hasn't translated,
for anaphora/consistency, distinct from `recent`'s already-translated pairs.
If a `synopsis_provider` callable is passed (the hook point for Zahra's SLM
context layer), it's called with a hard timeout on a background thread; any
exception or timeout falls straight through to the static `synopsis` argument
(or `None`) rather than blocking the cue. **This is the fallback switch
called out in the task doc** — it is a literal `try/except`, not a
retry/circuit-breaker framework, by design.

### `translate.py` — model inference
Thin OpenAI-client wrapper around vLLM's OpenAI-compatible server.
`DEFAULT_MODEL` is `tencent/Hy-MT2-1.8B` (the CPU/low-RAM fallback, not the
7B) with **greedy decoding** (`temperature=0.0`) — deterministic and, per the
code comment, stronger for MT than the model card's sampling defaults; `top_k`
/ `repetition_penalty` are still taken from the card.

`translate_cue(text, target, context=None, *, examples=None, temperature=None,
register=None, ...)` picks the prompt template by what's passed: `register`
set → `prompts.advanced_prompt` (composes context+examples+register);
otherwise `examples` → `few_shot_prompt`; otherwise `context` → `context_prompt`;
otherwise `default_prompt`. `temperature` overrides `SAMPLING` per-call — the
hook `candidates.generate_candidates` uses to sample the same prompt at
several temperatures. `translate_batch` is a sequential convenience wrapper
(vLLM's continuous batching overlaps the requests server-side, so no
client-side async fan-out is implemented — add it only if GPU utilization is
measurably bottlenecked by request issuance, not before). The `openai` import
is lazy (inside `_client()`), so `subtitle_io` / `constraints` / `pipeline`
(with an injected `translate_fn`) can be imported and tested without the
`openai` package installed.

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

**Not called from `pipeline.py` anymore** — `translate_cues` keeps cues 1:1
with the source (see below). `scripts/zero_shot_translate.py` still calls
`split_cue_if_needed` directly, per translated line, outside the pipeline loop.

### `pipeline.py` — orchestration
`translate_cues(cues, target, *, synopsis=None, synopsis_provider=None,
window=4, src_window=6, register=None, translate_fn=translate.translate_cue)`
is the loop: for each cue, assemble context (synopsis + the prior
`src_window` source lines + the rolling window of already-translated
`(source, translation)` pairs), call `translate_fn` with that context and
`examples=recent.pairs()` and `register`, accumulate. Cues stay 1:1 with the
source — same `index`/`start_ms`/`end_ms`, no splitting, no re-index.
`translate_fn` is injectable — this is what lets `tests/test_pipeline.py`
exercise the full orchestration logic (context threading, few-shot example
accumulation) with zero GPU / network dependency.

`translate_srt(in_path, out_path, target, ...)` is the file-level wrapper
used by `scripts/run_pipeline.py`.

### Advanced training-free pipeline (`candidates.py`, `rerank.py`, `glossary.py`, `postedit.py`, `fewshot.py`)
A quality-focused inference stack layered on top of `translate_cues` via an
injected `translate_fn` closure — `scripts/run_advanced_pipeline.py` is the
only place that wires all of it together; `pipeline.py` itself is unaware of
it. Every model in this stack is open-weight and the < 20B total-parameter
budget accounts for the base model + the CometKiwi reranker (see
`docs/research.md` for the compliance rationale).

- **`candidates.py::generate_candidates(text, target, *, context=None,
  glossary=None, examples=None, register=None, n=6, temps=DEFAULT_TEMPS,
  client=None, model=..., complete_fn=translate._complete)`** — the Chimera
  recipe: one `prompts.advanced_prompt`, sampled at `n` different
  temperatures, order-preserving-deduped. `complete_fn` is injectable for tests.
- **`rerank.py::qe_scores` / `offtarget_filter` / `pick_best(src, cands,
  target, *, score_fn=qe_scores)`** — `pick_best` filters out candidates that
  leaked Han characters into a non-Chinese target (the documented zh->id
  code-switching risk), then argmaxes the CometKiwi QE score. The CometKiwi
  checkpoint is loaded once via `@lru_cache` (`_load_kiwi`) and kept resident —
  `eval.py::score_comet_qe` reloads it every call, which would be
  O(N candidates × cues) fatal here.
- **`glossary.py::build_glossary(cues, synopsis, target, *, client, ...)
  -> dict[str, str]`** — one LLM pass over the whole episode extracting a
  `源词 -> rendering` mapping for character names / sects / titles / recurring
  idioms, seeded from the synopsis. `enforce(text, glossary)` is a verified
  exact find/replace — the safety net for any source term that still leaked
  into the translated output.
- **`postedit.py::postedit(src, hyp, target, *, register=None, client, ...)`**
  — a single TEaR/proofreader refine pass via `prompts.postedit_prompt`.
  Deliberately **one iteration only**: `docs/research.md` warns further
  refinement passes degrade dev COMET.
- **`fewshot.py::load_pool(tsv_path)` / `retrieve(src_text, pool, k=5)`** —
  fuzzy (stdlib `difflib.SequenceMatcher`) retrieval of similar `(src, tgt)`
  pairs from a zh-target parallel pool, fed into `advanced_prompt(examples=...)`.
  No FAISS/embeddings — deliberately, to stay out of the parameter budget and
  avoid an extra heavy dependency; upgrade only if pool size/recall demands it.

`scripts/run_advanced_pipeline.py` builds one shared client, optionally runs
`build_glossary` once per (video, lang) before the cue loop, and defines a
`cand_translate` closure — `generate_candidates` → `pick_best` → optional
`postedit` → `enforce` (glossary) — passed to `translate_cues` as `translate_fn`.
`REGISTER_NOTES.get(lang)` supplies the register instruction automatically.

### Retrieval-augmented post-editing (`termdb.py`, `postedit.py`, `termdb_build.py`)
A second pass over an *already-translated* output dir, per
[`docs/postedit-research.md`](./postedit-research.md): flag the cues whose
source carries a known idiom / honorific / wuxia term, re-write only those, and
accept the rewrite only if it measurably beats the draft. `scripts/run_postedit.py`
is the entry point; nothing in `pipeline.py` knows about it. Running as a
separate pass (rather than inline) is what makes an A/B free — it re-uses the
existing draft SRTs instead of re-translating 100 videos on the GPU.

- **`termdb.py`** — `load(path) -> TermDB`; `TermDB.match(text) -> list[Entry]`.
  An `Entry` pairs a Chinese headword with its *figurative* meaning, the
  preferred rendering per target language, and the literal renderings to
  `avoid` — the three signals that move idiom accuracy (IdiomKB, AAAI 2024).
  Detection and retrieval are the same call: the DB is a dict keyed by headword,
  so `match` both flags a line and returns what to inject. Matching is a sliding
  n-gram window with longest-match-wins — **no jieba, no Aho-Corasick**: Chinese
  has no spaces, so substring matching *is* the right primitive for fixed
  expressions, and word segmentation would only lose recall. `required()` /
  `hits()` are the shared primitives the accept gate and `term_recall` both
  count with, so they can't drift apart.
- **`postedit.py::postedit_cues(src_cues, hyp_cues, target, db, ...)`** — the
  selective pass; returns `(list[Cue], PostEditStats)` 1:1 with the draft's
  indices/timings. Cues matching no entry never reach the model at all; that
  selectivity is the point (APE is notorious for overcorrecting). `_accept`
  rejects an edit that leaks Han (`rerank.is_offtarget`), drops a rendering the
  draft already had, or adds a `check_constraints` violation — all free and
  deterministic. **The gate is what makes this safe to ship:** Hy-MT2 is an MT
  model, not an instruction-tuned editor, so a rejected edit falling back to the
  draft floors the worst case at baseline parity.
- **The QE gate is opt-in** (`--qe-gate`), not default. The research doc's own
  finding 3 reports reference-free QE correlating near-zero/negative *on idiom
  lines* — exactly the lines being judged. When on, `_qe_filter` batches the
  whole episode into one `rerank.qe_scores` call (`_load_kiwi` caches the
  checkpoint, not the forward pass).
- **`metrics.py::term_recall(srcs, hyps, db, lang)`** — the constraint-conformance
  number the terminology-APE literature actually moves (36.67%→72.88%; zh→en
  28.33%→82.65%, Moslem et al. WMT23). Judge post-editing on this, **not BLEU**:
  IdiomKB saw sacreBLEU fall 14.26→9.64 *while* idiom accuracy rose.
- **`termdb_build.py`** + `scripts/build_termdb.py` — the DB build
  (parse → merge → filter → enrich → qc → emit). Logic sits in the module so the
  parsers are testable; the script is a thin CLI, same split as
  `pipeline.py`/`run_pipeline.py`. Two things carry the design: **`--attested`
  runs before `--enrich`** (enriching all ~125k CC-CEDICT entries is 125k LLM
  calls; enriching only what fires on the test corpus is a few thousand, for the
  same final quality), and **provenance/license ride on every entry** so
  `--exclude-nc` can drop CC-BY-NC-SA-derived data (PETCI) for a
  release-compliant DB without a rebuild. CC-CIDICT is a direct CC-CEDICT
  derivative, so `parse_cedict` reads both — it's the only non-pivot zh→id
  lexical source available.

Sources split by kind: downloaded dictionaries in `data/termdb_build/raw/`
(gitignored, regenerable), hand-curated glossaries in `resources/manual/*.tsv`
(committed — they're authored, not fetched). Only CC-CEDICT is auto-fetched;
the rest need a manual drop rather than a guessed URL or a scraper against an
unversioned community site.

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
| `scripts/zero_shot_translate.py` | Batch baseline: every `<vid>_zh.srt` in `--in` (defaults to `data/`) → one `<vid>_<lang>.srt` per `--langs` entry. Auto-loads each video's own metadata for context unless `--synopsis` is passed manually. `--limit N` caps it to the first N source files. |
| `scripts/run_pipeline.py` | Full Phase C pipeline on a single file: rolling context + metadata. Same metadata auto-load / manual-override behavior. |
| `scripts/run_advanced_pipeline.py` | Batch, quality-focused: candidates → QE-rerank → optional post-edit → glossary-enforce per video/lang (see "Advanced training-free pipeline" above). Flags: `--in --out --langs --limit --n --shots --postedit --no-glossary --fewshot-pool --synopsis --model --base-url`. `--limit N` caps it to the first N source files, for trying 20/30 files before a full run. |
| `scripts/run_postedit.py` | Retrieval-augmented post-edit pass over an existing output dir: `--src` source + `--hyp` draft → `--out`. Flags: `--src --hyp --out --langs --termdb --limit --qe-gate --model --base-url`. Prints flagged/edited/accepted counts (the gain-to-edit and over-edit instrumentation). Requires the draft to be 1:1 with the source. |
| `scripts/build_termdb.py` | Builds `resources/termdb.jsonl`. `--fetch` grabs CC-CEDICT; `--attested data/tests` keeps only entries that fire on the corpus (**run before `--enrich`**); `--enrich` is the LLM pass for missing meanings/register/category and English-pivot Indonesian; `--exclude-nc` drops NC-licensed entries. |
| `scripts/run_eval.py` | Scores everything in `--hyp` against matching filenames in `--src`/`--ref`. |
| `scripts/run_benchmark_eval.py` | Scores a hyp dir against the ground-truth corpus (`data/gt`) instead of paired references — BLEU/chrF/chrF++ always, `--termdb` for term recall, `--neural` for XCOMET-XXL/CometKiwi, `--gemba` for an LLM-judge score. |

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

All of these run without a GPU or a live model server:
- `test_subtitle_io.py` — parse/write roundtrip, BOM handling
- `test_constraints.py` — violation detection, split contiguity
- `test_pipeline.py` — orchestration with a fake `translate_fn` (1:1 preservation, recent pairs fed as examples)
- `test_metadata.py` — synopsis formatting, episode parsing, missing-file case
- `test_candidates.py` — varied-temperature sampling, order-preserving dedup, `n` cap
- `test_rerank.py` — off-target Han filtering, argmax selection, single-candidate short-circuit
- `test_glossary.py` — glossary parsing, exact find/replace enforcement
- `test_fewshot.py` — fuzzy-similarity ranking, pool loading, single-pass post-edit
- `test_termdb.py` — longest-match-wins, dedup, `hits`/`required` counting
- `test_postedit.py` — selective pass with a fake `complete_fn`: unflagged cues never reach the model, each gate rejection, 1:1 index/timing preservation, term recall
- `test_termdb_build.py` — CC-CEDICT/CC-CIDICT/manual-TSV parsers, merge (license precedence), `--attested` filtering, enrich-fills-only-missing, `--exclude-nc`

Fixtures live in `tests/data/` — note `.gitignore`'s bare `data/` matches at any
depth, so that dir needs its `!tests/data/` negation to stay tracked.

GPU-dependent paths (`translate.py` against a live server, `eval.py`'s COMET
scoring, `rerank.py`'s CometKiwi, `build_termdb.py --enrich`) are exercised
manually via the scripts above, not in the test suite.

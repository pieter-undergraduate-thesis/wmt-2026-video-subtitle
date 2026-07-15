# WMT26 Video Subtitle Translation

Constrained-track system for WMT26 video subtitle translation: **Chinese (Simplified) →
English, Thai, Indonesian, Malay, Chinese (Traditional)**. Base model **Hy-MT2-7B** (≤20B
param cap), served via vLLM. Output is `.srt` named `vid_langshort.srt`.

## Docs

- [docs/TASK_BRIEF.md](docs/TASK_BRIEF.md) — start here: what the task is, why Hy-MT2, the 3-phase plan
- [docs/PIPELINE.md](docs/PIPELINE.md) — module-by-module implementation reference
- [docs/TEAM.md](docs/TEAM.md) — who owns what, handoff contracts, milestones, open items

## Layout

| Path | What |
|---|---|
| `src/wmt26/subtitle_io.py` | `Cue` + `parse_srt`/`write_srt` — the shared `list[Cue] -> list[Cue]` contract |
| `src/wmt26/prompts.py` | default / context / terminology prompt templates |
| `src/wmt26/translate.py` | vLLM OpenAI-client inference (`translate_cue`, `translate_batch`) |
| `src/wmt26/constraints.py` | CPL/line/CPS checks + cue splitting (Phase C) |
| `src/wmt26/context.py` | rolling translated-cue context + synopsis injection w/ fallback |
| `src/wmt26/pipeline.py` | end-to-end srt → translate → constrain → srt |
| `src/wmt26/eval.py` | BLEU/chrF (sacrebleu) + COMET + violation rate |
| `src/wmt26/metadata.py` | loads `<vid>_metadata.json` synopsis paired with `<vid>_zh.srt` |
| `scripts/` | `serve.sh`, `zero_shot_translate.py`, `run_pipeline.py`, `run_eval.py` |
| `finetune/` | sharegpt converter + LoRA train wrapper (Phase B, LLaMA-Factory) |

## Install

```bash
pip install -e ".[dev]"        # io/eval/pipeline (CPU ok)
pip install -e ".[serve]"      # + vLLM (GPU)
```

## Quickstart

```bash
# 1. serve the model (GPU)
bash scripts/serve.sh                      # vLLM on :8000

# 2. zero-shot baseline over a dir of source .srt, all 5 langs
#    (auto-loads each video's paired <vid>_metadata.json for context)
python scripts/zero_shot_translate.py --in data/test --out out

# 3. full Phase C pipeline on one file (rolling context + constraints)
#    --synopsis overrides metadata auto-load; omit to use the paired metadata file
python scripts/run_pipeline.py --in data/test/vid.srt --out out/vid_en.srt --lang en

# 4. score against references
python scripts/run_eval.py --src data/src --hyp out --ref data/ref
```

## Fine-tuning (Phase B)

```bash
python finetune/convert_to_sharegpt.py --in corpus.tsv --out data/wmt26_subtitle_sft.json \
    --lang English --idiom-repeat 3
# place JSON in LLaMA-Factory/data/, merge finetune/dataset_info_snippet.json, then:
bash finetune/train_lora.sh
```

Serve the adapter by uncommenting the `--enable-lora` lines in `scripts/serve.sh`.

## Tests

```bash
pytest          # io roundtrip, constraint split, pipeline orchestration (no GPU needed)
```

## Notes / open items

- **Constraint spec is a placeholder** (Netflix-style 42 CPL / 2 lines / 17 CPS in
  `constraints.py`). Swap for WMT26's official spec when it ships with the test data (Jul 1 2026).
- **th/id/ms fine-tuning deferred** — Hy-MT2 covers them zero-shot; measure the gap first.
- **License blocker**: confirm Hy-MT2 derivative weights can be released under WMT26's
  unrestricted-non-commercial requirement before relying on a fine-tuned submission.


<!-- 1. Install + serve the model
pip install -e ".[serve]"        # base + vLLM (GPU)
bash scripts/serve.sh            # vLLM OpenAI server on :8000
Every translate script talks to that server, so it must be up first.

2. Sample run first (the --limit flag you asked for)
# advanced pipeline (candidates -> QE-rerank -> post-edit -> glossary), first 20 files
python scripts/run_advanced_pipeline.py --in data/tests --out result/out_p1 --langs en id --limit 20
- --limit 20 = only the first 20 source .srt files. Bump to --limit 30, then drop it entirely for the full run.
- Stages default on: candidates (--n 6) + QE-rerank + episode glossary. Off by default: --postedit and --fewshot-pool data/zh_id.tsv — add each as a separate A/B.
- Same --limit also works on the baseline: python scripts/zero_shot_translate.py --in data/tests --out out --limit 20

3. Score it
python scripts/run_benchmark_eval.py --hyp result/out_p1 --neural --langs en id
A 20-file output dir just scores those 20 (the rest are skipped as missing hyp — no eval change needed). --neural adds XCOMET + CometKiwi (GPU + HF token); drop it for BLEU/chrF only. Compare XCOMET up / chrF not
collapsing vs the baseline results/hymt2-7b-out. -->

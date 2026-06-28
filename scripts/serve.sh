#!/usr/bin/env bash
# Serve Hy-MT2-7B via vLLM (OpenAI-compatible API on :8000).
# Usage: bash scripts/serve.sh [MODEL]
# Once a LoRA checkpoint exists, append the --enable-lora lines below.
set -euo pipefail

MODEL="${1:-tencent/Hy-MT2-7B}"

vllm serve "$MODEL" \
  --tensor-parallel-size 1
  # --enable-lora \
  # --lora-modules subtitle-lora=./checkpoints/hy-mt2-7b-subtitle-lora

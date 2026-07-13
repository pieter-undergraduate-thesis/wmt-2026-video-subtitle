#!/usr/bin/env bash
# Serve Hy-MT2-1.8B via vLLM (OpenAI-compatible API on :8000).
# Usage: bash scripts/serve.sh [MODEL]
#   MAX_MODEL_LEN / GPU_MEMORY_UTILIZATION env vars override the defaults below.
# Once a LoRA checkpoint exists, append the --enable-lora lines below.
set -euo pipefail

MODEL="${1:-tencent/Hy-MT2-1.8B}"
# The model's native 262144 (256K) context needs ~8 GiB KV cache and OOMs. Subtitle
# translation only sends one short cue + small rolling context, so 8192 is ample.
MAX_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"

vllm serve "$MODEL" \
  --tensor-parallel-size 2 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --swap-space 0
  # --enable-lora \
  # --lora-modules subtitle-lora=./checkpoints/hy-mt2-1.8b-subtitle-lora

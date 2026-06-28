#!/usr/bin/env bash
# LoRA fine-tune Hy-MT2-7B on the WMT26 subtitle SFT set via LLaMA-Factory.
#
# Prereqs:
#   git clone https://github.com/hiyouga/LLaMA-Factory.git
#   cd LLaMA-Factory && pip install -e ".[torch,metrics]"
#   - put data/wmt26_subtitle_sft.json in LLaMA-Factory/data/
#   - merge finetune/dataset_info_snippet.json into LLaMA-Factory/data/dataset_info.json
#
# Settings from the Hy-MT2 training guide (LoRA column).
set -euo pipefail

MODEL="${MODEL:-tencent/Hy-MT2-7B}"
OUT="${OUT:-./checkpoints/hy-mt2-7b-subtitle-lora}"

llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "$MODEL" \
  --dataset wmt26_subtitle_sft \
  --template hy_dense_7b \
  --finetuning_type lora \
  --lora_target all \
  --cutoff_len 8192 \
  --learning_rate 2.0e-4 \
  --num_train_epochs 3 \
  --output_dir "$OUT"

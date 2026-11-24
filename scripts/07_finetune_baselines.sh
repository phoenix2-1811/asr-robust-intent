#!/usr/bin/env bash
# Fine-tuning / LoRA baselines with a small learning-rate grid, DEV only. Test evaluation happens in 09_final_eval.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
DS=${DS:-slurp}
SEEDS=${SEEDS:-"13 42 123"}
FULL_LRS=${FULL_LRS:-"1e-5 2e-5 5e-5"}
LORA_LRS=${LORA_LRS:-"3e-4 1e-3 3e-3"}
for seed in $SEEDS; do
  for lr in $FULL_LRS; do python3 -m src.baselines.finetune --dataset "$DS" --mode full --lr_encoder "$lr" --seed "$seed"; done
  for lr in $LORA_LRS; do python3 -m src.baselines.finetune --dataset "$DS" --mode lora --lr_head "$lr" --seed "$seed"; done
done
python3 -m src.baselines.select_finetune --dataset "$DS"

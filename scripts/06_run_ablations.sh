#!/usr/bin/env bash
# Ablations on the main encoder pair (override with A=... T=...). DEV ONLY.
# To report an ablation on test, run: python3 -m src.eval_final --dataset $DS --run <run name>  (do this once, at the end).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
DS=${DS:-slurp}
A=${A:-microsoft/wavlm-base-plus}
T=${T:-roberta-base}
SEEDS=${SEEDS:-"13 42 123 777 2024"}
for seed in $SEEDS; do
  common="--dataset $DS --fusion reliability --audio_encoder $A --text_encoder $T --seed $seed"
  python3 -m src.train_head $common --audio_layers last --audio_stats mean --tag abl-lastmean
  python3 -m src.train_head $common --audio_layers last --audio_stats mean,std --tag abl-laststd
  python3 -m src.train_head $common --audio_layers weighted --audio_stats mean --tag abl-wmean
  python3 -m src.train_head $common --stage joint --tag abl-joint
  python3 -m src.train_head $common --stage joint --loss_mode learned_convex --tag abl-legacyalpha
  python3 -m src.train_head $common --modality_dropout 0 --tag abl-nomoddrop
  python3 -m src.train_head $common --train_pairs clean:whisper-small@clean --tag abl-asrtrainonly
  python3 -m src.train_head $common --train_pairs clean:gold --tag abl-goldtrainonly
  python3 -m src.train_head $common --train_pairs clean:gold,clean:whisper-small@clean,snr10:whisper-small@snr10 --tag abl-noisetrain
  python3 -m src.train_head $common --route hard --tag abl-hardroute
  for fusion in prob logit_product emb gated avg; do
    python3 -m src.train_head --dataset $DS --fusion $fusion --audio_encoder $A --text_encoder $T --seed $seed
  done
  if [ "${FRAMES:-0}" = "1" ]; then
    for arch in attn bilstm transformer; do
      python3 -m src.train_head $common --audio_mode frames --frame_arch $arch
      python3 -m src.train_head --dataset $DS --fusion xattn --audio_encoder $A --text_encoder $T --seed $seed --audio_mode frames --frame_arch $arch
    done
  fi
done

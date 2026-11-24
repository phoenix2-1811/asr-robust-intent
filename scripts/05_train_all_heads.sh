#!/usr/bin/env bash
# Baselines + every fusion variant, 5 seeds, over cached features. DEV ONLY (no test evaluation here).
# GRID=1 gives EVERY method the same hyper-parameter grid (lr x dropout); selection happens later on dev.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p logs
FAIL_LOG="logs/failed_head_runs.log"; : > "$FAIL_LOG"
DS=${DS:-"slurp"}
SEEDS=${SEEDS:-"13 42 123 777 2024"}
AUDIO_ENCODERS=${AUDIO_ENCODERS:-"facebook/wav2vec2-base facebook/hubert-base-ls960 microsoft/wavlm-base-plus openai/whisper-small"}
TEXT_ENCODERS=${TEXT_ENCODERS:-"distilbert-base-uncased roberta-base"}
FUSIONS=${FUSIONS:-"avg prob logit_product emb gated reliability"}   # xattn needs frames (see FRAME_ARCHS)
FRAME_ARCHS=${FRAME_ARCHS:-""}   # e.g. "attn bilstm transformer" (requires FRAMES=1 extraction)
FRAME_FUSIONS=${FRAME_FUSIONS:-"audio_only xattn reliability"}
if [ "${GRID:-0}" = "1" ]; then HP_LRS=${HP_LRS:-"3e-4 1e-3 3e-3"}; HP_DROPOUTS=${HP_DROPOUTS:-"0.1 0.3"}; else HP_LRS="1e-3"; HP_DROPOUTS="0.1"; fi
run_job() { echo ">> $*"; "$@" || echo "FAILED: $*" >> "$FAIL_LOG"; }

for ds in $DS; do
 for lr in $HP_LRS; do
  for do in $HP_DROPOUTS; do
   HP="--lr $lr --dropout $do"
   TAG=""; if [ "${GRID:-0}" = "1" ]; then TAG="--tag hp-lr$lr-do$do"; fi
   for seed in $SEEDS; do
    for t in $TEXT_ENCODERS; do
      run_job python3 -m src.train_head --dataset "$ds" --fusion text_only --text_encoder "$t" --seed "$seed" $HP $TAG
    done
    for a in $AUDIO_ENCODERS; do
      run_job python3 -m src.train_head --dataset "$ds" --fusion audio_only --audio_encoder "$a" --seed "$seed" $HP $TAG
      for t in $TEXT_ENCODERS; do
        for f in $FUSIONS; do
          run_job python3 -m src.train_head --dataset "$ds" --fusion "$f" --audio_encoder "$a" --text_encoder "$t" --seed "$seed" $HP $TAG
        done
      done
    done
    for arch in $FRAME_ARCHS; do
      for a in $AUDIO_ENCODERS; do
        for t in $TEXT_ENCODERS; do
          for f in $FRAME_FUSIONS; do
            if [ "$f" = "audio_only" ]; then
              run_job python3 -m src.train_head --dataset "$ds" --fusion audio_only --audio_encoder "$a" --seed "$seed" --audio_mode frames --frame_arch "$arch" $HP $TAG
            else
              run_job python3 -m src.train_head --dataset "$ds" --fusion "$f" --audio_encoder "$a" --text_encoder "$t" --seed "$seed" --audio_mode frames --frame_arch "$arch" $HP $TAG
            fi
          done
        done
      done
    done
   done
  done
 done
done
echo "done; failures: $(wc -l < "$FAIL_LOG")"

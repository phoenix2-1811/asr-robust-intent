#!/usr/bin/env bash
# Whisper transcripts + confidence for every (model x audio condition x split), then synthetic corruption.
# Multi-GPU: NGPU=4 bash scripts/03_transcribe_asr.sh  (shards each job across GPUs, then merges)
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
NGPU=${NGPU:-1}
DATASETS=${DATASETS:-"slurp"}
SPLITS=${SPLITS:-"train dev test"}
MODELS=${MODELS:-"tiny base small medium large-v3"}
CTC_MODEL=${CTC_MODEL:-"facebook/wav2vec2-base-960h"}   # set SKIP_CTC=1 for non-English datasets
SKIP_CTC=${SKIP_CTC:-0}
TRAIN_CONDS=${TRAIN_CONDS:-"clean"}
EVAL_CONDS=${EVAL_CONDS:-"clean snr20 snr10 snr5 snr0 rev400 rev400+snr10"}
mkdir -p logs

run_one() {
  local ds=$1 split=$2 model=$3 cond=$4
  if [ "$NGPU" -le 1 ]; then
    python3 -m src.asr.transcribe --dataset "$ds" --split "$split" --model "$model" --audio_cond "$cond" --skip_existing
  else
    for ((g=0; g<NGPU; g++)); do
      CUDA_VISIBLE_DEVICES=$g python3 -m src.asr.transcribe --dataset "$ds" --split "$split" --model "$model" \
        --audio_cond "$cond" --shard "$g/$NGPU" &
    done
    wait
    python3 -m src.asr.transcribe --dataset "$ds" --split "$split" --model "$model" --audio_cond "$cond" --shard "0/$NGPU" --merge
  fi
}

for ds in $DATASETS; do
  for model in $MODELS; do
    for cond in $TRAIN_CONDS; do
      run_one "$ds" train "$model" "$cond"
      run_one "$ds" dev "$model" "$cond"
    done
    for cond in $EVAL_CONDS; do
      run_one "$ds" test "$model" "$cond"
      run_one "$ds" dev "$model" "$cond"
    done
  done
  if [ "$SKIP_CTC" != "1" ]; then
    for cond in $TRAIN_CONDS; do
      python3 -m src.asr.transcribe_ctc --dataset "$ds" --split train --model "$CTC_MODEL" --audio_cond "$cond" --skip_existing
      python3 -m src.asr.transcribe_ctc --dataset "$ds" --split dev --model "$CTC_MODEL" --audio_cond "$cond" --skip_existing
    done
    for cond in $EVAL_CONDS; do
      python3 -m src.asr.transcribe_ctc --dataset "$ds" --split test --model "$CTC_MODEL" --audio_cond "$cond" --skip_existing
      python3 -m src.asr.transcribe_ctc --dataset "$ds" --split dev --model "$CTC_MODEL" --audio_cond "$cond" --skip_existing
    done
  fi
  python3 -m src.asr.synthetic_corruption --dataset "$ds"
  # predicted-WER feature for the reliability gate (cross-fitted on train@clean)
  PRED_MODELS=""
  for m in $MODELS; do PRED_MODELS="$PRED_MODELS whisper-$m"; done
  if [ "$SKIP_CTC" != "1" ]; then PRED_MODELS="$PRED_MODELS w2v2ctc-$(basename $CTC_MODEL | sed 's/wav2vec2-//')"; fi
  python3 -m src.asr.wer_predictor --dataset "$ds" --models $PRED_MODELS
done

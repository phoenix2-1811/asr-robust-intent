#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p features logs
FAIL_LOG="logs/failed_extractions.log"; : > "$FAIL_LOG"
DATASETS=${DATASETS:-"slurp"}
AUDIO_ENCODERS=${AUDIO_ENCODERS:-"facebook/wav2vec2-base facebook/hubert-base-ls960 microsoft/wavlm-base-plus openai/whisper-small"}
TEXT_ENCODERS=${TEXT_ENCODERS:-"distilbert-base-uncased roberta-base"}
AUDIO_CONDS_TRAIN=${AUDIO_CONDS_TRAIN:-"clean"}
AUDIO_CONDS_EVAL=${AUDIO_CONDS_EVAL:-"clean snr20 snr10 snr5 snr0 rev400 rev400+snr10"}
TEXT_MODELS=${TEXT_MODELS:-"whisper-tiny whisper-small whisper-large-v3 w2v2ctc-base-960h"}
FRAMES=${FRAMES:-0}      # FRAMES=1 also caches frame-level features (needed for attentive/BiLSTM/transformer/xattn heads)
FR=""; if [ "$FRAMES" = "1" ]; then FR="--save_frames"; fi
SYNTH=${SYNTH:-"synth-wer10 synth-wer20 synth-wer30"}

run_job() { echo ">> $*"; "$@" || echo "FAILED: $*" >> "$FAIL_LOG"; }

for ds in $DATASETS; do
  for enc in $AUDIO_ENCODERS; do
    for c in $AUDIO_CONDS_TRAIN; do
      run_job python3 -m src.features.extract_embeddings --dataset "$ds" --encoder_type audio --encoder_name "$enc" --split train --audio_cond "$c" $FR
    done
    for c in $AUDIO_CONDS_EVAL; do
      for sp in dev test; do
        run_job python3 -m src.features.extract_embeddings --dataset "$ds" --encoder_type audio --encoder_name "$enc" --split "$sp" --audio_cond "$c" $FR
      done
    done
  done
  for enc in $TEXT_ENCODERS; do
    for sp in train dev test; do
      run_job python3 -m src.features.extract_embeddings --dataset "$ds" --encoder_type text --encoder_name "$enc" --split "$sp" --text_source gold
    done
    for m in $TEXT_MODELS; do
      for c in $AUDIO_CONDS_TRAIN; do
        run_job python3 -m src.features.extract_embeddings --dataset "$ds" --encoder_type text --encoder_name "$enc" --split train --text_source "$m@$c"
      done
      for c in $AUDIO_CONDS_EVAL; do
        for sp in dev test; do
          run_job python3 -m src.features.extract_embeddings --dataset "$ds" --encoder_type text --encoder_name "$enc" --split "$sp" --text_source "$m@$c"
        done
      done
    done
    for s in $SYNTH; do
      for sp in dev test; do
        run_job python3 -m src.features.extract_embeddings --dataset "$ds" --encoder_type text --encoder_name "$enc" --split "$sp" --text_source "$s"
      done
    done
  done
done
N=$(wc -l < "$FAIL_LOG"); echo "extraction done, $N failed job(s) (see $FAIL_LOG)"

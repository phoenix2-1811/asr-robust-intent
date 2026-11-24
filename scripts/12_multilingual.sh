#!/usr/bin/env bash
# Multilingual extension on Speech-MASSIVE (German by default; edit configs/experiment.yaml for other locales).
# Requires multilingual encoders: text xlm-roberta-base, audio XLS-R or Whisper encoder. wav2vec2-CTC is English-only (skipped).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
DS=${DS:-speechmassive_de}
AUDIO=${AUDIO:-"openai/whisper-small facebook/wav2vec2-xls-r-300m"}
TEXT=${TEXT:-"xlm-roberta-base"}
python3 -m src.data.prepare_speechmassive --dataset "$DS"
DATASETS="$DS" SKIP_CTC=1 MODELS="${MODELS:-small}" EVAL_CONDS="clean snr10 snr0" bash scripts/03_transcribe_asr.sh
DATASETS="$DS" AUDIO_ENCODERS="$AUDIO" TEXT_ENCODERS="$TEXT" AUDIO_CONDS_EVAL="clean snr10 snr0" \
  TEXT_MODELS="whisper-small" bash scripts/04_extract_all_features.sh
DS="$DS" AUDIO_ENCODERS="$AUDIO" TEXT_ENCODERS="$TEXT" bash scripts/05_train_all_heads.sh

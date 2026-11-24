#!/usr/bin/env bash
# Development -> selection -> single final test evaluation -> analysis. See PROTOCOL.md.
set -uo pipefail
cd "$(dirname "$0")/.."
bash scripts/03_transcribe_asr.sh
bash scripts/04_extract_all_features.sh
GRID=1 bash scripts/05_train_all_heads.sh
bash scripts/07_finetune_baselines.sh
bash scripts/08_select.sh
bash scripts/09_final_eval.sh
bash scripts/10_analysis.sh

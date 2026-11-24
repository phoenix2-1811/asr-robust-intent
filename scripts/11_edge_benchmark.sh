#!/usr/bin/env bash
# Run ON the target device (Jetson Nano etc.). Small encoders recommended there.
set -uo pipefail
cd "$(dirname "$0")/.."
python3 -m src.edge.benchmark --audio_encoder "${A:-facebook/wav2vec2-base}" --text_encoder "${T:-distilbert-base-uncased}" \
  --device "${DEVICE:-auto}" --runs 30 --out "results/edge_$(hostname).json"

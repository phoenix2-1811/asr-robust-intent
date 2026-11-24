#!/usr/bin/env bash
# Pre-specified model selection on DEV only (see PROTOCOL.md). Writes results/selection_<ds>.json/.env.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
DS=${DS:-slurp}
python3 -m src.analysis.aggregate --split dev
python3 -m src.analysis.select_best --dataset "$DS" --min_seeds "${MIN_SEEDS:-5}"

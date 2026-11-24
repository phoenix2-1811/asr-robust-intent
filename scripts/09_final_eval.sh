#!/usr/bin/env bash
# The ONLY place the test split is evaluated: once, for the runs chosen on dev. Re-runs are refused unless forced.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
DS=${DS:-slurp}
source "results/selection_${DS}.env"
RUN_ARGS=""; for r in $RUNS_FOR_FINAL; do RUN_ARGS="$RUN_ARGS --run $r"; done
python3 -m src.eval_final --dataset "$DS" $RUN_ARGS
if [ -f "results/selection_finetune_${DS}.env" ]; then
  source "results/selection_finetune_${DS}.env"
  for d in $FT_DIRS; do python3 -m src.baselines.finetune --eval_only "$d"; done
fi
python3 -m src.analysis.aggregate --split test

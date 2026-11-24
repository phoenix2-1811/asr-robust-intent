#!/usr/bin/env bash
# CPU-only end-to-end check of the head-training + evaluation + analysis path on synthetic features.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-/tmp/asr_smoke}"
PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO"
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
$PY "$REPO/tests/test_units.py"
$PY "$REPO/tests/make_synthetic.py" "$WORK"
$PY -m src.asr.wer_predictor --config test_config.yaml --dataset synth --models whisper-small whisper-tiny
for seed in 1 2; do
  $PY -m src.train_head --config test_config.yaml --dataset synth --fusion text_only --text_encoder faketext --seed $seed
  $PY -m src.train_head --config test_config.yaml --dataset synth --fusion audio_only --audio_encoder fakeaudio --seed $seed
  for f in avg prob logit_product emb gated reliability; do
    $PY -m src.train_head --config test_config.yaml --dataset synth --fusion $f --audio_encoder fakeaudio --text_encoder faketext --seed $seed
  done
  $PY -m src.train_head --config test_config.yaml --dataset synth --fusion reliability --audio_encoder fakeaudio --text_encoder faketext --seed $seed --route hard --epochs 10
  for arch in attn bilstm transformer; do
    $PY -m src.train_head --config test_config.yaml --dataset synth --fusion xattn --audio_encoder fakeaudio --text_encoder faketext --seed $seed --audio_mode frames --frame_arch $arch --epochs 10
  done
  $PY -m src.train_head --config test_config.yaml --dataset synth --fusion reliability --audio_encoder fakeaudio --text_encoder faketext --seed $seed --audio_mode frames --frame_arch bilstm --epochs 10
done
$PY -m src.data.audit_splits --config test_config.yaml --dataset synth
# development phase: no test evaluation yet
test ! -e checkpoints/synth/text_only__faketext__tx-gold+whisper-small-clean/seed1/preds_test_clean_gold.npz
$PY -m src.analysis.aggregate --config test_config.yaml --split dev
$PY -m src.analysis.select_best --config test_config.yaml --dataset synth --min_seeds 2
source results/selection_synth.env
RUN_ARGS=""; for r in $RUNS_FOR_FINAL; do RUN_ARGS="$RUN_ARGS --run $r"; done
$PY -m src.eval_final --config test_config.yaml --dataset synth $RUN_ARGS
$PY -m src.eval_final --config test_config.yaml --dataset synth $RUN_ARGS | grep -c skip
$PY -m src.analysis.aggregate --config test_config.yaml --split test
BASE=checkpoints/synth
COND="clean_whisper-small snr10_whisper-small snr10_whisper-tiny"
$PY -m src.analysis.compare_all --config test_config.yaml --dataset synth --pair "$FUSED:$TEXT" --pair "$FUSED:$AUDIO" --conditions $COND --out results/stats/primary.csv
$PY -m src.analysis.bootstrap --config test_config.yaml --a "$BASE/$FUSED/seed*/preds_test_snr10_whisper-small.npz" --b "$BASE/$TEXT/seed*/preds_test_snr10_whisper-small.npz" > /dev/null
$PY -m src.analysis.complementarity --config test_config.yaml --dataset synth --fused "$BASE/$FUSED/seed*/preds_test_snr10_whisper-small.npz" --text_only "$BASE/$TEXT/seed*/preds_test_snr10_whisper-small.npz" --asr_source whisper-small@snr10 --out_prefix results/stats/comp > /dev/null
$PY -m src.analysis.independence --preds "$BASE/$FUSED/seed1/preds_test_clean_whisper-small.npz" --n_boot 30 --n_perm 100 > /dev/null
$PY -m src.analysis.error_taxonomy --config test_config.yaml --dataset synth --asr_source whisper-small@snr10 --fused "$BASE/$FUSED/seed*/preds_test_snr10_whisper-small.npz" --text_only "$BASE/$TEXT/seed*/preds_test_snr10_whisper-small.npz" --out_prefix results/stats/tax > /dev/null
$PY -m src.analysis.gain_bound --config test_config.yaml --dataset synth --fused_tpl "$BASE/$FUSED/seed*/preds_test_{cond}.npz" --text_tpl "$BASE/$TEXT/seed*/preds_test_{cond}.npz" --conditions $COND --out results/stats/gain.csv > /dev/null
$PY -m src.analysis.seen_unseen --config test_config.yaml --dataset synth --run "$TEXT" --run "$AUDIO" --run "$FUSED" --condition clean_gold --out results/stats/seen_unseen.csv
$PY -m src.analysis.layer_weights --config test_config.yaml --dataset synth > /dev/null
$PY -m src.reporting.build_tables --config test_config.yaml --dataset synth --conditions clean_gold clean_whisper-small snr10_whisper-small
echo "SMOKE TEST PASSED (synthetic data: numbers are meaningless, this only checks the plumbing)"

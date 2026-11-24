#!/usr/bin/env bash
# All statistics and analyses on the TEST predictions of dev-selected runs.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
DS=${DS:-slurp}
source "results/selection_${DS}.env"
FT_FULL=""; FT_LORA=""
if [ -f "results/selection_finetune_${DS}.env" ]; then source "results/selection_finetune_${DS}.env"; fi
BASE=checkpoints/$DS
mkdir -p results/stats
COND_MAIN=${COND_MAIN:-"clean_whisper-small snr20_whisper-small snr10_whisper-small snr5_whisper-small snr0_whisper-small rev400+snr10_whisper-small clean_whisper-tiny clean_whisper-large-v3 clean_w2v2ctc-base-960h"}

# Primary (pre-specified): selected fused system vs selected text-only baseline, all main conditions, Holm-corrected.
PAIRS="--pair $FUSED:$TEXT"
python3 -m src.analysis.compare_all --dataset "$DS" $PAIRS --conditions $COND_MAIN --out "results/stats/primary_${DS}.csv"

# Secondary: every fusion family's best vs text-only, vs average ensemble, vs fine-tuning (non-inferiority at 1 point).
SEC="--pair $FUSED:$AUDIO"
for r in $RUNS_FOR_FINAL; do if [ "$r" != "$TEXT" ] && [ "$r" != "$AUDIO" ] && [ "$r" != "$FUSED" ]; then SEC="$SEC --pair $r:$TEXT"; fi; done
if [ -n "${FT_FULL:-}" ]; then SEC="$SEC --pair $FUSED:$FT_FULL"; fi
if [ -n "${FT_LORA:-}" ]; then SEC="$SEC --pair $FUSED:$FT_LORA"; fi
python3 -m src.analysis.compare_all --dataset "$DS" $SEC --conditions $COND_MAIN --out "results/stats/secondary_${DS}.csv"

python3 -m src.analysis.complementarity --dataset "$DS" --fused "$BASE/$FUSED/seed*/preds_test_snr10_whisper-small.npz" \
  --text_only "$BASE/$TEXT/seed*/preds_test_snr10_whisper-small.npz" --asr_source "whisper-small@snr10" \
  --out_prefix "results/stats/complementarity_${DS}_snr10"
python3 -m src.analysis.independence --preds "$BASE/$FUSED/seed42/preds_test_clean_whisper-small.npz" --out "results/stats/independence_${DS}.json"
for cond in clean_whisper-small snr10_whisper-small snr0_whisper-small; do
  src="${cond#*_}@${cond%%_*}"
  python3 -m src.analysis.error_taxonomy --dataset "$DS" --asr_source "$src" \
    --fused "$BASE/$FUSED/seed*/preds_test_${cond}.npz" --text_only "$BASE/$TEXT/seed*/preds_test_${cond}.npz" \
    --out_prefix "results/stats/taxonomy_${DS}_${cond}"
done
python3 -m src.analysis.gain_bound --dataset "$DS" --fused_tpl "$BASE/$FUSED/seed*/preds_test_{cond}.npz" \
  --text_tpl "$BASE/$TEXT/seed*/preds_test_{cond}.npz" \
  --conditions clean_whisper-small snr20_whisper-small snr10_whisper-small snr5_whisper-small snr0_whisper-small \
               clean_whisper-tiny snr10_whisper-tiny clean_whisper-large-v3 snr10_whisper-large-v3 \
               clean_w2v2ctc-base-960h snr10_w2v2ctc-base-960h --out "results/stats/gain_bound_${DS}.csv"
for cond in clean_gold clean_whisper-small snr10_whisper-small; do
  python3 -m src.analysis.seen_unseen --dataset "$DS" --run "$TEXT" --run "$AUDIO" --run "$FUSED" --condition "$cond" \
    --out "results/stats/seen_unseen_${DS}_${cond}.csv"
done
python3 -m src.analysis.layer_weights --dataset "$DS"
python3 -m src.reporting.build_tables --dataset "$DS"

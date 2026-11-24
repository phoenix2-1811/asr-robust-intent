#!/usr/bin/env bash
# CPU-only check of extraction, ASR plumbing, fine-tuning/LoRA and benchmark using tiny RANDOM Hugging Face models.
# Whisper decoding uses a stub module (tests/stubs/whisper.py), so the real openai-whisper calls are not exercised.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-/tmp/asr_extract_smoke}"
PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO"
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
$PY "$REPO/tests/make_tiny_models.py" "$WORK/models" "$WORK/data/tiny" >/dev/null 2>&1
REPO_DIR="$REPO" $PY - <<'PYEOF'
import os, yaml
cfg = yaml.safe_load(open(os.environ["REPO_DIR"] + "/configs/experiment.yaml"))
cfg["datasets"] = {"tiny": {"type": "jsonl", "root": "data/tiny", "max_audio_seconds": 5, "language": "en"}}
cfg["audio_features"].update({"batch_size": 4, "whisper_batch_size": 4})
cfg["conditions"].update({"audio": ["clean", "snr10", "rev300+snr5"], "eval_asr_models": ["whisper-tiny", "w2v2ctc-ctc-tiny"],
                          "asr_models": ["whisper-tiny"], "synthetic_wer": [0.1, 0.3]})
cfg["train"].update({"epochs": 8, "batch_size": 16, "train_pairs": [{"audio": "clean", "text": "gold"}],
                     "select_on": {"audio": "clean", "text": "gold"}})
yaml.safe_dump(cfg, open("cfg.yaml", "w"))
PYEOF
for split in train dev test; do
  for cond in clean snr10 rev300+snr5; do
    for enc in models/wav2vec2-tiny models/wav2vec2-tiny-ln models/whisper-tiny-rand; do
      $PY -m src.features.extract_embeddings --config cfg.yaml --dataset tiny --encoder_type audio --encoder_name $enc --split $split --audio_cond $cond --save_frames >/dev/null
    done
  done
  $PY -m src.features.extract_embeddings --config cfg.yaml --dataset tiny --encoder_type text --encoder_name models/distilbert-tiny --split $split --text_source gold >/dev/null
done
export PYTHONPATH="$REPO:$REPO/tests/stubs"
for cond in clean snr10; do
  $PY -m src.asr.transcribe --config cfg.yaml --dataset tiny --split test --model tiny --audio_cond $cond --batch_size 3 >/dev/null
  $PY -m src.asr.transcribe_ctc --config cfg.yaml --dataset tiny --split test --model models/wav2vec2-ctc-tiny --audio_cond $cond >/dev/null
done
$PY -m src.asr.synthetic_corruption --config cfg.yaml --dataset tiny >/dev/null
export PYTHONPATH="$REPO"
for mode in pooled frames; do
  $PY -m src.train_head --config cfg.yaml --dataset tiny --fusion reliability --audio_encoder models/wav2vec2-tiny --text_encoder models/distilbert-tiny --seed 1 --audio_mode $mode --epochs 3
done
for mode in full lora; do
  $PY -m src.baselines.finetune --config cfg.yaml --dataset tiny --audio_encoder models/wav2vec2-tiny --text_encoder models/distilbert-tiny --mode $mode --seed 1 --epochs 2 --num_workers 0 --train_pairs clean:gold
done
$PY -m src.edge.benchmark --audio_encoder models/wav2vec2-tiny --text_encoder models/distilbert-tiny --runs 3 --n_labels 4 >/dev/null
echo "EXTRACTION SMOKE TEST PASSED (random tiny models: numbers are meaningless)"

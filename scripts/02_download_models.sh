#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
python3 - <<'PYEOF'
import whisper
from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer
for size in ["tiny", "base", "small", "medium", "large-v3"]:
    print("whisper", size)
    whisper.load_model(size)
for n in ["facebook/wav2vec2-base", "facebook/hubert-base-ls960", "microsoft/wavlm-base-plus", "openai/whisper-small"]:
    AutoModel.from_pretrained(n); AutoFeatureExtractor.from_pretrained(n)
for n in ["distilbert-base-uncased", "roberta-base"]:
    AutoModel.from_pretrained(n); AutoTokenizer.from_pretrained(n)
print("models cached")
PYEOF

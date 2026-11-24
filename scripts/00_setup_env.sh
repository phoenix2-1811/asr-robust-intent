#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Create an environment, install torch matching your CUDA, then:"
echo "  pip install -r requirements.txt"
echo "  huggingface-cli login"

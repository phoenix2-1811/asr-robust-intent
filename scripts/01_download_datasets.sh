#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p data/slurp data/fsc

if [ ! -d "data/slurp/raw" ]; then
    git clone https://github.com/pswietojanski/slurp.git data/slurp/raw
    if [ -f "data/slurp/raw/scripts/download_audio.sh" ]; then
        (cd data/slurp/raw/scripts && bash download_audio.sh)
        if [ -d "data/slurp/raw/scripts/audio" ] && [ ! -d "data/slurp/raw/audio" ]; then
            mv data/slurp/raw/scripts/audio data/slurp/raw/audio
        fi
    fi
fi

if [ ! -f "data/fsc/fluent_speech_commands_dataset.tar.gz" ] && [ ! -f "data/fsc/fluent-speech-corpus.zip" ] && [ ! -d "data/fsc/fluent_speech_commands_dataset" ]; then
    echo "FSC: download from https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/"
    echo "     place it at data/fsc/fluent_speech_commands_dataset.tar.gz, then re-run."
else
    if [ -f "data/fsc/fluent_speech_commands_dataset.tar.gz" ]; then
        tar -xzf data/fsc/fluent_speech_commands_dataset.tar.gz -C data/fsc/
    elif [ -f "data/fsc/fluent-speech-corpus.zip" ]; then
        (cd data/fsc && unzip -o fluent-speech-corpus.zip)
        if [ -f "data/fsc/fluent_speech_commands_dataset/data/train_data.csv" ]; then
            mv data/fsc/fluent_speech_commands_dataset/data/*.csv data/fsc/fluent_speech_commands_dataset/
        fi
    fi
fi

echo "Noise corpus (optional but recommended): MUSAN from https://www.openslr.org/17/"
echo "  extract it, then set noise.noise_dir in configs/experiment.yaml to <musan>/noise"
echo "  If noise_dir is null, synthetic white/pink/brown noise is used instead."

python3 -m src.data.prepare_slurp --config configs/experiment.yaml
python3 -m src.data.prepare_fsc --config configs/experiment.yaml
python3 -m src.data.derive_max_len --config configs/experiment.yaml
python3 -m src.data.derive_audio_len --config configs/experiment.yaml
python3 -m src.data.audit_splits

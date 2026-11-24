# What you need before running

## Hardware
- One or more CUDA GPUs, 24 GB+ recommended (full fine-tuning of WavLM + RoBERTa and Whisper-large-v3 decoding are the
  heaviest jobs; head training and all analysis run on CPU or a small GPU).
- Disk: SLURP real audio (several GB); cached features about 1 to 2 GB per audio encoder per split per condition
  (layer-wise mean+std in fp16); ASR transcripts a few hundred MB.
- Edge device (e.g. Jetson) only for `scripts/09_edge_benchmark.sh`.

## Accounts and data
- Hugging Face access for the pretrained encoders; `openai-whisper` downloads its own weights.
- SLURP: cloned and downloaded by `01_download_datasets.sh`. FSC: free form on the Fluent.ai page.
- Optional noise corpus: MUSAN (openslr.org/17); without it synthetic white/pink/brown noise is used.

## Software
`pip install -r requirements.txt` after installing a CUDA build of PyTorch.

## Rough compute order (single GPU, to be measured, not promised)
1. Whisper decoding over all splits and conditions dominates: shard with `NGPU=<n>`, start with `MODELS="tiny small"`.
2. Feature extraction scales with (encoders × conditions × splits); start with one encoder and the clean condition.
3. Head training takes seconds to minutes per run; fine-tuning baselines take hours per run.
Run a reduced version first (one dataset, `MODELS="small"`, `AUDIO_ENCODERS=microsoft/wavlm-base-plus`) before the full grid.

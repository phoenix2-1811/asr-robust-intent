# ASR-Robust Multimodal Intent Recognition (v2)

Frozen self-supervised audio encoders + frozen text encoders + small trained heads, evaluated under **real ASR errors**
(Whisper at several sizes, additive noise, reverberation) and synthetic word-level corruption. v2 adds a
reliability-aware fusion head, a much stronger audio branch, fine-tuning/LoRA baselines, and a full statistics suite.
See `CHANGES.md` for what changed relative to the submitted version and why.

## Pipeline and protocol

Read `PROTOCOL.md` first: it fixes hypotheses, the selection rule, the statistics and the claims each outcome allows, and
should be committed before the full experiments. The code enforces it: training is **dev-only**, selection uses **dev
only**, and the **test split is evaluated once**, for the selected runs.

```
raw audio ──► [03 Whisper + CTC transcripts, confidence, predicted WER] ─┐
   │                                                                      ├─► [04 cached frozen features]
   └─► [audio condition: clean / snrN / revMS / combos] ──────────────────┘            │
                                                                                        ▼
  [05 train heads, dev only, equal grid for every method] ─► [07 fine-tune/LoRA, dev only] ─► [08 select on dev]
                                                                                        ▼
                    [09 final test evaluation, once] ─► [10 statistics and analysis]  (+ 11 edge, 12 multilingual)
```

| Step | Script | What it does |
|---|---|---|
| 01 | `scripts/01_download_datasets.sh` | SLURP, FSC and JSONL splits (`scenario_action` = 69 classes, `scenario` = 18, FSC official and `fsc_unseen`); **split audit** (sentence overlap, majority-class and sentence-lookup baselines) |
| 02 | `scripts/02_download_models.sh` | Caches Whisper and encoder weights |
| 03 | `scripts/03_transcribe_asr.sh` | Batched Whisper and wav2vec2-CTC decoding per (model × audio condition × split) with confidence and utterance WER; synthetic corruption; cross-fitted predicted-WER feature. `NGPU=4` shards Whisper |
| 04 | `scripts/04_extract_all_features.sh` | Length-aware masked layer-wise mean+std pooling (wav2vec 2.0, HuBERT, WavLM Base/Large, Whisper encoder), text CLS+mean per transcript source; `FRAMES=1` adds frame caches |
| 05 | `scripts/05_train_all_heads.sh` | Text-only, audio-only and every fusion variant × encoders × 5 seeds; `GRID=1` gives every method the same lr × dropout grid; `FRAME_ARCHS=...` adds frame heads. **Dev only** |
| 06 | `scripts/06_run_ablations.sh` | Pooling, layers, two-stage vs joint, loss weights, modality dropout, hard routing, training sources, noise-augmented training. Dev only |
| 07 | `scripts/07_finetune_baselines.sh` | Full fine-tuning and LoRA with a learning-rate grid, dev only; picks the best rate on dev |
| 08 | `scripts/08_select.sh` | Pre-specified selection on dev (mean dev accuracy over all real-ASR conditions); writes `results/selection_<ds>.json/.env` |
| 09 | `scripts/09_final_eval.sh` | **The only test evaluation**: selected runs, all seeds; repeats are refused and counted |
| 10 | `scripts/10_analysis.sh` | Holm-corrected primary and secondary comparisons with non-inferiority vs fine-tuning, complementarity, independence + CMI, error taxonomy, gain bound, seen-vs-unseen sentence slice, layer weights, tables |
| 11 | `scripts/11_edge_benchmark.sh` | Parameters, latency and memory on the target device |
| 12 | `scripts/12_multilingual.sh` | Speech-MASSIVE (German by default) |

Quick start (after installing `requirements.txt` and downloading data):

```bash
bash scripts/01_download_datasets.sh && bash scripts/02_download_models.sh
python3 -m src.data.derive_audio_len        # then edit max_audio_seconds in configs/experiment.yaml if needed
bash scripts/run_pipeline.sh                # or run the steps one at a time
```

Check the install first (CPU only, a few minutes):

```bash
bash tests/run_smoke.sh              # synthetic features: training, selection, final eval, all analyses
bash tests/run_extraction_smoke.sh   # tiny random HF models: extraction, ASR plumbing, fine-tune/LoRA, benchmark
```

## Fusion variants (`--fusion`)

`text_only`, `audio_only`, `avg` (average log-probabilities), `prob` (the original: concatenated probabilities → MLP),
`logit_product` (calibrated product of experts with learned per-branch weights and prior), `emb` (embedding concat → MLP),
`gated` (scalar gate from embeddings), `xattn` (text embedding attends over audio frames, needs `--audio_mode frames`),
`reliability` (proposed: a gate driven by ASR confidence, predicted WER, cross-ASR disagreement and branch entropies
mixes text, audio and joint experts; `--route hard` makes it a straight-through hard router).

Audio branch (`--audio_mode`): `pooled` (layer-weighted mean+std statistics, default) or `frames` (cached frame-level features
of one layer with `--frame_arch attn|bilstm|transformer`). Pick the layer from `python -m src.analysis.layer_weights`
and set `audio_features.frame_layer`. The frame cache is large (SLURP train, stride 2: roughly 6 GB per encoder and
condition), and frame heads have far more parameters (0.1 to 1.2 M) than the pooled heads, so report parameter counts.

Training: `--stage two_stage` (default; branches trained and temperature-calibrated on dev, then frozen, then the fusion
module is trained) or `--stage joint`. Loss weights are fixed by default; `--loss_mode learned_convex` reproduces the v1 behaviour.
`--modality_dropout` randomly hides one modality during fusion training.

## Theory

`docs/theory.md` states an exact identity for the gain available to any rule that picks one branch's prediction,
`G* = q(1-p) - phi*sqrt(p(1-p)q(1-q))`, and what it predicts; `src.analysis.gain_bound` measures every term and the
ROC-AUC of the reliability features for the event "text wrong and audio right".

## External baselines

ESPnet-SLU, SpeechBrain or any other system: run it with its own recipe, write `utt_id,pred` (label string or id), then
`python -m src.baselines.import_external --dataset slurp --pred_file preds.csv --name espnet-slu`. It then appears in
aggregation, bootstrap and tables like any other system. `src/baselines/speech_llm.py` is an untested zero-shot
Qwen2-Audio wrapper that produces such a file.

## Adding a dataset

Write `data/<name>/{train,dev,test}.jsonl` with `audio_path, transcript, label, utt_id, group` (`group` identifies
repeated sentences and is used for clustered statistics) plus `label_map.json`, then add
`<name>: {type: jsonl, root: data/<name>, max_audio_seconds: 10}` under `datasets:` in the config.

## Outputs

`checkpoints/<dataset>/<run>/seed<N>/` holds `best.pt`, `meta.json`, `metrics.json` (accuracy per condition) and
`preds_test_<condition>.npz` (per-utterance predictions of every branch; all analysis scripts read these).
`results/stats/` holds aggregated tables and analysis outputs. `results/v1_reference/` keeps the submitted version's numbers.

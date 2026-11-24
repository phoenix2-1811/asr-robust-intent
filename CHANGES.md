# Changes from the submitted version (v1)

## Problems found in the v1 code

1. **Audio pooling included padding.** Audio was padded to `max_audio_seconds` and mean-pooled over every frame,
   including the padded silence (`extract_embeddings.py` v1). Short utterances were averaged mostly over padding,
   which likely explains the 45.5% audio-only accuracy on SLURP.
2. **Last layer only.** SSL encoders carry intent information in middle layers; v1 used only the last.
3. **Learned convex loss weights collapse.** `AlphaWeights` could (and did: α₂ = 0.008 on SLURP) put almost all weight
   on the easiest loss, effectively not training the audio branch.
4. **Temperature scaling was applied after fusion training**, to the audio branch only, which changes the input
   distribution the fusion MLP was trained on.
5. **Corruption was independent of the audio, single-seed, test-only**, and the Whisper pass was skipped.
6. `results/paper_tables.md` in the repo contained a stale SLURP audio-only floor (0.093, from a mismatched encoder)
   that disagrees with the paper (0.455). It is kept only under `results/v1_reference/`.
7. `FSC` official split: every unique test sentence is also in training (the new `prepare_fsc` prints this overlap, and
   `fsc_unseen` builds a split with zero overlap).

## What v2 changes

- **Audio branch:** length-aware masked pooling, per-layer mean+std features, learned layer weighting, Whisper encoder.
- **Real ASR errors:** batched Whisper decoding (tiny to large-v3) with confidence, additive noise at set SNRs,
  synthetic reverberation, combined conditions; measured WER reported per condition. Conditions are deterministic per
  utterance, so audio features and ASR transcripts always see the same degraded waveform.
- **Fusion:** six variants including the proposed reliability-aware gate, two-stage training with branch calibration,
  fixed or uncertainty-based loss weights, modality dropout, training on gold + ASR transcripts (+ noisy pairs).
- **Baselines:** text-only on ASR transcripts, average ensemble, full fine-tuning, LoRA.
- **Statistics:** 5 seeds everywhere, 95% CIs, sentence-clustered paired bootstrap, McNemar, oracle complementarity by
  utterance WER and ASR confidence, class-conditional (Cochran–Mantel–Haenszel) independence test with cluster-bootstrap CI.
- **Datasets:** SLURP `scenario_action`, optional SLURP real+synthetic audio (`slurp_real_synth`), `fsc_unseen`, a
  Speech-MASSIVE converter (multilingual; Whisper language and unicode-aware normalization are configurable per
  dataset), and a generic JSONL adapter for anything else.
- **Frame-level audio (round 2):** ragged frame cache of one layer; attentive pooling, BiLSTM and transformer audio heads;
  text-to-audio cross-attention fusion (`xattn`).
- **Second ASR system:** wav2vec2-CTC decoding with confidence (blank fraction, mean max-probability).
- **Reliability signals:** cross-fitted gradient-boosted predictor of utterance WER (`pred_wer`) added to the gate inputs;
  hard (straight-through) routing option.
- **Layer-norm encoders:** WavLM Large / XLS-R style models now get an attention mask during extraction.
- **Analysis:** error taxonomy by ASR error type (numbers, negation, function and content words) with intent confusions
  recovered by fusion; conditional mutual information with a within-class permutation test; gain-bound check and AUC of
  reliability features (`docs/theory.md`); learned-layer-weight inspector; external-prediction importer; GPU-hours logged.
- **Theory:** `docs/theory.md` replaces the earlier lemma/proposition with a measurable identity and falsification criteria.
- **Data format:** every analysis reads `preds_*.npz` files, so no analysis needs the checkpoints or GPUs.
- Removed: `src/models/fusion.py` (old end-to-end model), `inspect_alpha_weights.py`, `mcnemar_test_head.py`
  (superseded by `analysis/bootstrap.py`), `ablate_*_head.py` (superseded by the config-driven ablations).
- Old checkpoints and cached features are **not compatible** with v2 and must be regenerated.

## Round 3: removing methodological liabilities

- **Test-set hygiene:** `train_head.py` and `finetune.py` are dev-only unless `--final`; `src/eval_final.py` evaluates
  selected runs on test once and refuses repeats (`test_evaluations` is recorded).
- **No hand-picked winners:** `src/analysis/select_best.py` chooses runs by mean dev accuracy over all real-ASR
  conditions with a rule fixed in `PROTOCOL.md`; analysis scripts read the selection instead of hard-coded run names.
- **Equal tuning budget:** `GRID=1` applies the same lr × dropout grid to every head-based method; fine-tuning/LoRA get an
  lr grid selected on dev (`select_finetune.py`).
- **Multiple comparisons:** `compare_all.py` reports Holm-adjusted p-values, Bonferroni intervals and
  non-inferiority bounds (default margin 1 point) across a pre-specified family.
- **Memorisation:** `audit_splits.py` (majority-class and sentence-lookup baselines, split overlaps) and
  `seen_unseen.py` (accuracy on test sentences seen vs unseen in training, cluster-bootstrap CIs).
- **More metrics:** macro-F1 and ECE for every condition; parameter counts and GPU-hours per run.
- **Reproducibility:** every run records config hash, git commit, library versions and GPU.
- **Pre-registration:** `PROTOCOL.md` (hypotheses, selection, statistics, allowed claims per outcome, threats to validity).
- Robustness fix: the independence test no longer crashes when a branch is always right.

## What has and has not been verified

Verified on CPU with synthetic features and tiny randomly initialised Hugging Face models (`tests/`, run both
`tests/run_smoke.sh` and `tests/run_extraction_smoke.sh`):
unit tests (Holm correction, ECE, config hash, padding invariance of every frame encoder and of cross-attention, hard routing, loss/stage logic, noise SNR,
corruption hitting its target WER, bootstrap, FSC split, gain identity, CMI, importer, unicode normalization);
training and evaluation of every fusion variant, loss mode, stage, audio mode and frame architecture; extraction for
wav2vec 2.0-style (group-norm and layer-norm), Whisper-encoder and DistilBERT models under noise/reverb, with pooled and
frame caches; wav2vec2-CTC decoding; Whisper transcript plumbing with a stubbed `whisper` module (sharding, merging,
schema); WER-predictor cross-fitting; full fine-tuning and LoRA loops; all analysis scripts; the latency benchmark;
the unseen-utterance FSC split; SLURP preparation on a fake layout; the Speech-MASSIVE row converter.

**Partly verified:** the real `openai-whisper` API (signature, batched `decode`, result fields, language option) and
my `transcribe.py` were run against the installed package with a tiny random-weight model on CPU; the fp16/GPU path and
real weights were not.

**Not verified** (no access here to model hubs, datasets or GPUs): real Whisper weights and the GPU fp16 path;
loading `FBK-MT/Speech-MASSIVE` (split names per locale were taken from its dataset card, the train split may be
`train_115` for some locales and the test repo uses a loading script); real SLURP/FSC files; `speech_llm.py`
(untested by design); GPU speed and memory; and, importantly, **any accuracy result**. No v2 accuracy numbers exist yet.

## Still not done

- n-best disagreement (openai-whisper does not expose n-best lists); cross-ASR disagreement and the predicted-WER
  regressor are used instead.
- Token-level text cross-attention (text tokens are not cached; `xattn` uses the text CLS embedding as the query).
- STOP and MInDS-14 converters (formats could not be verified; use the JSONL adapter).
- Running ESPnet-SLU / SpeechBrain themselves (use the importer).
- The paper rewrite itself.

Notes: HF base audio models use GroupNorm, so pooled features depend slightly on the padding in a batch; batches are
sorted by length to minimise this and `audio_features.batch_size: 1` removes it. Layer-norm models (WavLM Large,
XLS-R) are batch-invariant with the attention mask (verified on a tiny layer-norm model).

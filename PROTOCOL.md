# Evaluation protocol (commit this file before running the full experiments)

Writing the protocol down in advance is what separates a confirmatory result from a search over results. Edit it only
before the final test evaluation, and note any change in the paper.

## Hypotheses

- **H1 (main).** With real ASR errors, the dev-selected fused system has higher intent accuracy than the dev-selected
  text-only system, and the advantage grows as transcripts degrade.
- **H2.** Reliability-aware fusion is at least as accurate as the best simple fusion (average, product of experts,
  concatenation) and strictly better in the conditions where audio can help (high WER).
- **H3.** The frozen-encoder system is non-inferior (margin 1 accuracy point) to fully fine-tuned and LoRA systems on
  the real-ASR conditions, at a small fraction of the trainable parameters and GPU time.
- **H4 (mechanism).** The gain is explained by the measured quantities in `docs/theory.md`: oracle router gain, branch
  correlation and the AUC of reliability features for "text wrong, audio right".

## Data rules

- Train, dev and test splits are fixed by `configs/experiment.yaml` and audited by `python -m src.data.audit_splits`.
  The audit reports majority-class and sentence-lookup baselines; if sentence lookup alone is high, headline results
  use the unseen-sentence slice (`src.analysis.seen_unseen`).
- ASR transcripts, the predicted-WER regressor, temperatures and every hyper-parameter are fitted on train and dev only.
- The test split is evaluated exactly once per selected run through `src/eval_final.py` / `scripts/09_final_eval.sh`.
  `train_head.py` is dev-only unless `--final` is passed; re-evaluation is refused without `--force` and counted in
  `metrics.json` (`test_evaluations`).

## Selection rule (all methods get the same budget)

- Every method (text-only, audio-only, every fusion variant) is trained with the same grid
  (`GRID=1`: lr in {3e-4, 1e-3, 3e-3} x dropout in {0.1, 0.3}) and 5 seeds; fine-tuning/LoRA get a 3-value learning-rate
  grid and 3 seeds.
- A run is scored by its mean **dev** accuracy over all real-ASR conditions (clean and degraded, every ASR system),
  averaged over seeds. The top run per family is chosen by `src/analysis/select_best.py`. Ablation runs (`abl-` tags)
  are never candidates.

## Primary and secondary analyses

- Primary: selected fused run vs selected text-only run on each condition in `COND_MAIN`, sentence-clustered paired
  bootstrap (10,000 resamples, seed-averaged correctness), Holm-adjusted p-values across all comparisons, plus
  Bonferroni-adjusted intervals. One pre-specified comparison family only.
- Secondary (reported as secondary, with the same correction): every fusion family vs text-only, fused vs audio-only,
  fused vs average ensemble, fused vs fine-tuning/LoRA (non-inferiority: one-sided 95% lower bound above -1 point).
- Descriptive: macro-F1 and ECE per condition, error taxonomy, oracle complementarity by WER and by ASR confidence,
  class-conditional independence and conditional mutual information, gain-bound table.

## Reporting rules

- Report every condition, system and seed that was run, including ones that favour the baseline.
- Report measured WER per condition, not target WER. Say which transcripts were synthetic.
- Give parameter counts and GPU-hours for every system.

## What the paper may claim (decide before looking at test results)

| Outcome on the primary family (Holm-adjusted) | Claim allowed |
|---|---|
| Fused better than text-only in most degraded conditions, not worse on clean | "Audio improves robustness to ASR errors" with the mechanism evidence |
| Better only under acoustic noise or the highest WER | "Audio helps when transcripts are unreliable", with the conditions named; no general claim |
| No significant difference anywhere | Analysis paper: when and why audio does not help (use H4 evidence); no method claim |
| Fused worse than text-only | Report as a negative result; do not present the method as an improvement |
| Non-inferior to fine-tuning at 1 point | "Comparable accuracy at a fraction of the cost"; otherwise report the measured gap |

## Known threats to validity (state them in the paper)

- Whisper and the text encoders may have seen related data during pre-training.
- Synthetic noise and reverberation are not field recordings; MUSAN noise is a stand-in for real environments.
- Branch accuracy correlations are measured on one test set; clustering by sentence handles repeated sentences but not
  speaker effects.
- Frame-level heads add parameters; compare systems at matched parameter counts where possible.

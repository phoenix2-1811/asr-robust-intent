# Theory: when can adding audio help a text intent classifier?

This replaces the earlier Lemma 1 / Proposition 1 (extra information cannot hurt the Bayes error; the product rule is
optimal under conditional independence). Those are classical and say nothing about a trained system. The result below
is an exact identity about two trained branches, uses no independence assumption, and every quantity in it is measured
by `src/analysis/gain_bound.py`.

## Setup

Fix an evaluation condition (e.g. Whisper-small transcripts at SNR 10 dB). For a test utterance let
`C_T` and `C_A` be the indicators that the text branch and the audio branch predict the true intent, and write

- `p = P(C_T)`, `q = P(C_A)` (branch accuracies),
- `phi` = the correlation (phi coefficient) between `C_T` and `C_A`,
- `s = sqrt(p(1-p) q(1-q))`.

## Proposition 1 (router gain)

For any rule that outputs the prediction of one of the two branches, accuracy is at most that of the ideal router,
`P(C_T or C_A)`, and its gain over the text branch is at most

```
G* = P(not C_T and C_A) = q(1-p) - phi * s .
```

*Proof.* `P(C_T and C_A) = pq + phi*s` by the definition of the phi coefficient, so
`P(C_T or C_A) = p + q - pq - phi*s = p + q(1-p) - phi*s`. A rule choosing one branch's prediction is correct on an
utterance only if that branch is correct, hence its accuracy is at most `P(C_T or C_A)`. ∎

## Consequences (each is checked by the script)

1. **Independence is optimistic.** With `phi = 0` the gain is `q(1-p)`; every positive correlation between branch errors
   removes `phi*s` from the available gain. Column `phi_marginal` and the class-conditional versions in
   `analysis/independence.py` measure it; `gain_if_independent` vs `oracle_router_gain` shows the loss.
2. **Gain grows as transcripts degrade, only if audio stays reliable.** Under clean audio (`q` fixed) and `phi` fixed,
   `G*` increases as `p` falls. Under acoustic noise `q` also falls, so `dG*/d(noise) = q'(1-p) - q p'` (independent
   case) and the gain can shrink. This is the prediction to compare with the SNR conditions.
3. **Joint experts can exceed the router.** A fusion model that also uses joint evidence can be correct when both
   branches are wrong. That headroom is `R = P(not C_T and not C_A) = (1-p)(1-q) + phi*s`
   (`both_wrong_headroom`). `capture_of_router_gain` above 1 means the model is using it.
4. **What a reliability gate must predict.** A gate reaches `G*` exactly when it can tell the event
   `E = (not C_T and C_A)` from its complement. Reliability features are therefore useful if and only if they
   discriminate `E`: `gain_bound.py --dataset ...` reports the ROC-AUC of `-avg_logprob`, the predicted WER and the true
   utterance WER (an upper reference) for `E`. An AUC near 0.5 for every feature means the gate has nothing to work
   with and the method should not be expected to beat plain fusion.

## What would falsify the account

- `oracle_router_gain` not equal to the formula (a bug in the pipeline; the two columns must agree to numerical precision).
- Fused gain below the text-only baseline in conditions where `E` has substantial rate and a reliability feature has AUC
  clearly above 0.5: the gate is failing to use available information.
- A flat or falling gain as WER rises under clean audio.

## Relation to the earlier proposition

Kittler et al. (1998) give the product rule under class-conditional independence of the two sources given the label.
That assumption is not needed above; it is replaced by the measured `phi`. The product-of-experts variant
(`--fusion logit_product`) remains in the code as the calibrated-probability reading of that rule and is compared
empirically rather than assumed optimal.

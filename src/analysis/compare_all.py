import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.bootstrap import cluster_sums, load_correct, mcnemar_exact
from src.common.utils import load_config


def boot_diffs(ca, cb, groups, n_boot, seed=0):
    sa, n = cluster_sums(ca, groups)
    sb, _ = cluster_sums(cb, groups)
    rng = np.random.default_rng(seed)
    K = len(n)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, K, K)
        d[b] = (sa[idx].sum() - sb[idx].sum()) / n[idx].sum()
    return d, ca.mean() - cb.mean()


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(description="Pre-specified comparisons with Holm correction and non-inferiority margins.")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pair", action="append", required=True, help="A_run:B_run (repeat); tests A minus B")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--delta", type=float, default=0.01, help="non-inferiority margin (accuracy points as fraction)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["checkpoints"]) / args.dataset
    nb = cfg["analysis"]["bootstrap_samples"]
    rows, diffs = [], []
    for pair in args.pair:
        a, b = pair.split(":")
        for cond in args.conditions:
            pa = sorted(glob.glob(str(root / a / "seed*" / f"preds_test_{cond}.npz")))
            pb = sorted(glob.glob(str(root / b / "seed*" / f"preds_test_{cond}.npz")))
            if not pa or not pb:
                continue
            ca, ids, groups = load_correct(pa, "fused_pred")
            cb, ids_b, _ = load_correct(pb, "fused_pred")
            if not np.array_equal(ids, ids_b):
                continue
            d, obs = boot_diffs(ca, cb, groups, nb)
            diffs.append(d)
            p = max(float(((d - d.mean()) >= abs(obs)).mean() + ((d - d.mean()) <= -abs(obs)).mean()), 1.0 / nb)
            rows.append({"A": a, "B": b, "condition": cond, "acc_A": float(ca.mean()), "acc_B": float(cb.mean()),
                         "diff_A_minus_B": float(obs), "ci95_lo": float(np.percentile(d, 2.5)),
                         "ci95_hi": float(np.percentile(d, 97.5)), "p_boot": min(p, 1.0),
                         "lower_bound_one_sided_95": float(np.percentile(d, 5)),
                         "noninferior_at_delta": bool(np.percentile(d, 5) > -args.delta),
                         "n_seeds_A": len(pa), "n_seeds_B": len(pb), "n_clusters": int(len(np.unique(groups))),
                         **mcnemar_exact(ca, cb)})
    if not rows:
        raise SystemExit("no comparable prediction files found")
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_boot"].values)
    df["significant_holm"] = df["p_holm"] < args.alpha
    m = len(df)
    df["ci_bonferroni_lo"] = [float(np.percentile(d, 100 * args.alpha / (2 * m))) for d in diffs]
    df["ci_bonferroni_hi"] = [float(np.percentile(d, 100 - 100 * args.alpha / (2 * m))) for d in diffs]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    show = ["A", "B", "condition", "acc_A", "acc_B", "diff_A_minus_B", "ci95_lo", "ci95_hi", "p_holm", "significant_holm",
            "noninferior_at_delta"]
    print(df[show].round(4).to_string(index=False))
    print(f"\n{m} comparisons; p-values Holm-adjusted across all of them; CIs are cluster bootstrap over sentences.")


if __name__ == "__main__":
    main()

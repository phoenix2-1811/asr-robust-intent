import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.data import load_examples
from src.common.text_norm import normalize_text
from src.common.utils import load_config


def cluster_mean_ci(values, groups, mask, n_boot=2000, seed=0):
    v, g = values[mask], groups[mask]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq, inv = np.unique(g, return_inverse=True)
    s = np.zeros(len(uniq))
    n = np.zeros(len(uniq))
    np.add.at(s, inv, v)
    np.add.at(n, inv, 1.0)
    rng = np.random.default_rng(seed)
    K = len(uniq)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, K, K)
        boots[b] = s[idx].sum() / n[idx].sum()
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser(description="Accuracy on test sentences seen vs unseen in training (memorisation check).")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--condition", default="clean_gold")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_sents = {normalize_text(ex["transcript"]) for ex in load_examples(cfg, args.dataset, "train")}
    test = {ex["utt_id"]: normalize_text(ex["transcript"]) for ex in load_examples(cfg, args.dataset, "test")}
    root = Path(cfg["paths"]["checkpoints"]) / args.dataset
    rows = []
    for run in args.run:
        paths = sorted(glob.glob(str(root / run / "seed*" / f"preds_test_{args.condition}.npz")))
        if not paths:
            continue
        ds = [np.load(p, allow_pickle=False) for p in paths]
        y = ds[0]["labels"]
        correct = np.mean([d["fused_pred"] == y for d in ds], axis=0)
        groups = ds[0]["groups"]
        seen = np.array([test[str(u)] in train_sents for u in ds[0]["utt_ids"]])
        r = {"run": run, "condition": args.condition, "n_seen": int(seen.sum()), "n_unseen": int((~seen).sum())}
        for name, m in (("seen", seen), ("unseen", ~seen), ("all", np.ones_like(seen))):
            mean, lo, hi = cluster_mean_ci(correct, groups, m)
            r.update({f"acc_{name}": mean, f"acc_{name}_ci_lo": lo, f"acc_{name}_ci_hi": hi})
        rows.append(r)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

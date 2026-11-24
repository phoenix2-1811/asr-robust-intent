import argparse
import glob
import json
from pathlib import Path

import numpy as np

from src.common.utils import load_config, write_json


def load_correct(paths, key):
    ids = groups = None
    acc = []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        if ids is None:
            ids, groups = d["utt_ids"], d["groups"]
        elif not np.array_equal(ids, d["utt_ids"]):
            raise ValueError(f"utterance order differs in {p}")
        acc.append((d[key] == d["labels"]).astype(np.float64))
    return np.mean(acc, axis=0), ids, groups


def cluster_sums(values, groups):
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, values)
    sizes = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    return sums, sizes


def cluster_bootstrap_diff(ca, cb, groups, n_boot=10000, seed=0):
    sa, n = cluster_sums(ca, groups)
    sb, _ = cluster_sums(cb, groups)
    rng = np.random.default_rng(seed)
    K = len(n)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, K, K)
        tot = n[idx].sum()
        diffs[b] = (sa[idx].sum() - sb[idx].sum()) / tot
    obs = ca.mean() - cb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    centered = diffs - diffs.mean()
    p = float((np.abs(centered) >= abs(obs)).mean())
    return {"acc_a": float(ca.mean()), "acc_b": float(cb.mean()), "diff": float(obs),
            "ci95": [float(lo), float(hi)], "p_boot": max(p, 1.0 / n_boot), "n_clusters": int(K),
            "n_utts": int(len(ca))}


def mcnemar_exact(ca, cb):
    from scipy.stats import binomtest
    a_only = int(((ca >= 0.5) & (cb < 0.5)).sum())
    b_only = int(((ca < 0.5) & (cb >= 0.5)).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "p_mcnemar_exact": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--a", required=True, help="glob of preds npz for system A (one per seed)")
    ap.add_argument("--b", required=True, help="glob of preds npz for system B (one per seed)")
    ap.add_argument("--key_a", default="fused_pred")
    ap.add_argument("--key_b", default="fused_pred")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pa, pb = sorted(glob.glob(args.a)), sorted(glob.glob(args.b))
    if not pa or not pb:
        raise SystemExit("no files matched")
    ca, ids, groups = load_correct(pa, args.key_a)
    cb, ids_b, _ = load_correct(pb, args.key_b)
    if not np.array_equal(ids, ids_b):
        raise SystemExit("A and B cover different utterances")
    res = cluster_bootstrap_diff(ca, cb, groups, cfg["analysis"]["bootstrap_samples"])
    res.update(mcnemar_exact(ca, cb))
    res.update({"n_seeds_a": len(pa), "n_seeds_b": len(pb), "files_a": pa[:3], "files_b": pb[:3]})
    print(json.dumps(res, indent=2))
    if args.out:
        write_json(args.out, res)


if __name__ == "__main__":
    main()

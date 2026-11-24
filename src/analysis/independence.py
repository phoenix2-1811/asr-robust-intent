import argparse
import glob
import json

import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import StratifiedTable

from src.common.utils import write_json


def phi_from_counts(a, b, c, d):
    num = a * d - b * c
    den = np.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
    return 0.0 if den == 0 else num / den


def class_tables(y, tc, ac, min_n=20):
    tables, weights, phis = [], [], []
    for k in np.unique(y):
        m = y == k
        if m.sum() < min_n:
            continue
        t, a = tc[m], ac[m]
        both = int((t & a).sum())
        t_only = int((t & ~a).sum())
        a_only = int((~t & a).sum())
        neither = int((~t & ~a).sum())
        tables.append(np.array([[both, t_only], [a_only, neither]], dtype=float))
        weights.append(int(m.sum()))
        phis.append(phi_from_counts(both, t_only, a_only, neither))
    return tables, np.array(weights, dtype=float), np.array(phis)


def weighted_phi(y, tc, ac, min_n=20):
    _, w, p = class_tables(y, tc, ac, min_n)
    return float((w * p).sum() / max(w.sum(), 1)) if len(w) else float("nan")


def cluster_boot_phi(y, tc, ac, groups, n_boot=1000, seed=0, min_n=20):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        vals.append(weighted_phi(y[idx], tc[idx], ac[idx], min_n))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def cond_mutual_info(y, tc, ac, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    n = len(y)

    def mi(t, a):
        tot = 0.0
        for k in classes:
            m = y == k
            nk = int(m.sum())
            if nk == 0:
                continue
            tt, aa = t[m], a[m]
            val = 0.0
            for i in (0, 1):
                for j in (0, 1):
                    pij = np.mean((tt == i) & (aa == j))
                    pi, pj = np.mean(tt == i), np.mean(aa == j)
                    if pij > 0:
                        val += pij * np.log(pij / (pi * pj))
            tot += (nk / n) * val
        return tot

    obs = mi(tc, ac)
    bias = 0.0
    for k in classes:
        m = y == k
        nk = int(m.sum())
        if nk > 0:
            bias += (nk / n) * (1.0 / (2.0 * nk))
    null = []
    idx_by_class = [np.where(y == k)[0] for k in classes]
    for _ in range(n_perm):
        a2 = ac.copy()
        for ii in idx_by_class:
            a2[ii] = a2[rng.permutation(ii)]
        null.append(mi(tc, a2))
    null = np.array(null)
    return {"cmi_nats": float(obs), "cmi_plugin_minus_expected_null": float(obs - null.mean()),
            "permutation_null_mean": float(null.mean()), "permutation_p": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
            "note": "I(correct_text; correct_audio | true intent), permuting audio correctness within class"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="glob of fused-run preds npz; first match is used")
    ap.add_argument("--min_class_n", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = sorted(glob.glob(args.preds))[0]
    d = np.load(path, allow_pickle=False)
    y = d["labels"]
    tc = d["text_pred"] == y
    ac = d["audio_pred"] == y
    both = int((tc & ac).sum())
    t_only = int((tc & ~ac).sum())
    a_only = int((~tc & ac).sum())
    neither = int((~tc & ~ac).sum())
    n = len(y)
    try:
        chi2, p, _, _ = chi2_contingency(np.array([[both, t_only], [a_only, neither]]), correction=False)
    except ValueError:
        chi2, p = float("nan"), float("nan")
    tables, w, phis = class_tables(y, tc, ac, args.min_class_n)
    cmh_stat, cmh_p, pooled_or = float("nan"), float("nan"), float("nan")
    try:
        st = StratifiedTable(tables)
        cmh = st.test_null_odds(correction=True)
        cmh_stat, cmh_p, pooled_or = float(cmh.statistic), float(cmh.pvalue), float(st.oddsratio_pooled)
    except Exception:
        pass
    res = {
        "file": path, "n_utts": n, "n_unique_groups": int(len(np.unique(d["groups"]))),
        "marginal": {"table": {"both": both, "text_only": t_only, "audio_only": a_only, "neither": neither},
                     "chi2": float(chi2), "p": float(p), "phi": float(np.sqrt(chi2 / n)) if chi2 == chi2 else float("nan")},
        "class_conditional": {
            "n_strata": len(tables), "cmh_statistic": cmh_stat, "cmh_p": cmh_p,
            "pooled_odds_ratio": pooled_or,
            "weighted_mean_phi": float((w * phis).sum() / w.sum()),
            "weighted_mean_phi_ci95_cluster_bootstrap": cluster_boot_phi(y, tc, ac, d["groups"], args.n_boot,
                                                                          0, args.min_class_n),
        },
        "conditional_mutual_information": cond_mutual_info(y, tc.astype(int), ac.astype(int), n_perm=args.n_perm),
        "oracle_acc": float((tc | ac).mean()), "text_branch_acc": float(tc.mean()),
        "audio_branch_acc": float(ac.mean()),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        write_json(args.out, res)


if __name__ == "__main__":
    main()

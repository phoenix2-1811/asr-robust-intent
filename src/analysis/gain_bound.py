from pathlib import Path
import argparse
import glob
import math

import numpy as np
import pandas as pd

from src.common.data import load_asr
from src.common.utils import load_config


def stats_for(paths_fused, paths_text):
    fused = [np.load(p, allow_pickle=False) for p in paths_fused]
    y = fused[0]["labels"]
    tc = np.mean([d["text_pred"] == y for d in fused], axis=0)
    ac = np.mean([d["audio_pred"] == y for d in fused], axis=0)
    fc = np.mean([d["fused_pred"] == y for d in fused], axis=0)
    p, q = float(tc.mean()), float(ac.mean())
    both = float((tc * ac).mean())
    s = math.sqrt(max(p * (1 - p) * q * (1 - q), 1e-12))
    phi = (both - p * q) / s
    row = {"text_branch_acc": p, "audio_branch_acc": q, "phi_marginal": phi,
           "gain_if_independent": q * (1 - p),
           "oracle_router_gain": float(((1 - tc) * ac).mean()),
           "predicted_from_p_q_phi": q * (1 - p) - phi * s,
           "both_wrong_headroom": float(((1 - tc) * (1 - ac)).mean()),
           "fused_acc": float(fc.mean())}
    if paths_text:
        base = float(np.mean([np.load(p, allow_pickle=False)["fused_pred"] == y for p in paths_text]))
        row["text_only_baseline"] = base
        row["fused_gain_over_text_only"] = row["fused_acc"] - base
    else:
        row["fused_gain_over_text_only"] = row["fused_acc"] - p
    og = row["oracle_router_gain"]
    row["capture_of_router_gain"] = row["fused_gain_over_text_only"] / og if og > 1e-9 else float("nan")
    return row


def cond_to_source(cond):
    a, rest = cond.split("_", 1)
    return f"{rest}@{a}" if (rest.startswith("whisper") or rest.startswith("w2v2ctc")) else None


def event_aucs(cfg, dataset, cond, paths_fused):
    from sklearn.metrics import roc_auc_score
    src = cond_to_source(cond)
    if src is None or cfg is None:
        return {}
    try:
        asr = load_asr(cfg, dataset, "test", src)["utts"]
    except Exception:
        return {}
    fused = [np.load(p, allow_pickle=False) for p in paths_fused]
    y = fused[0]["labels"]
    tc = np.mean([d["text_pred"] == y for d in fused], axis=0) >= 0.5
    ac = np.mean([d["audio_pred"] == y for d in fused], axis=0) >= 0.5
    event = (~tc) & ac
    if event.sum() < 5 or (~event).sum() < 5:
        return {}
    ids = [str(u) for u in fused[0]["utt_ids"]]
    out = {"event_rate_text_wrong_audio_right": float(event.mean())}
    feats = {"auc_true_utt_wer": [asr[u]["wer"] for u in ids]}
    if "avg_logprob" in asr[ids[0]]:
        feats["auc_neg_avg_logprob"] = [-asr[u]["avg_logprob"] for u in ids]
    if "pred_wer" in asr[ids[0]]:
        feats["auc_pred_wer"] = [asr[u]["pred_wer"] for u in ids]
    for k, v in feats.items():
        out[k] = float(roc_auc_score(event, v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", default=None, help="enables the AUC check of reliability features")
    ap.add_argument("--fused_tpl", required=True, help='glob with {cond}, e.g. ".../seed*/preds_test_{cond}.npz"')
    ap.add_argument("--text_tpl", default=None)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in args.conditions:
        pf = sorted(glob.glob(args.fused_tpl.format(cond=c)))
        pt = sorted(glob.glob(args.text_tpl.format(cond=c))) if args.text_tpl else []
        if not pf:
            continue
        row = {"condition": c, **stats_for(pf, pt)}
        if args.dataset:
            row.update(event_aucs(load_config(args.config), args.dataset, c, pf))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.round(4).to_string(index=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["text_branch_acc"], df["oracle_router_gain"], "o-", label="oracle router gain (measured)")
        ax.plot(df["text_branch_acc"], df["predicted_from_p_q_phi"], "x--", label="q(1-p) - phi*sqrt(p(1-p)q(1-q))")
        ax.plot(df["text_branch_acc"], df["fused_gain_over_text_only"], "s-", label="fused gain over text-only")
        ax.set_xlabel("text branch accuracy p")
        ax.set_ylabel("accuracy gain")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out.replace(".csv", ".png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()

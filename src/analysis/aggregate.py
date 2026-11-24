import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.common.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["checkpoints"])
    acc = defaultdict(list)
    extra = defaultdict(list)
    meta = {}
    for mp in sorted(root.glob("*/*/seed*/metrics.json")):
        m = json.load(open(mp))
        key = "results" if args.split == "test" else "dev_results"
        for cond, r in m.get(key, {}).items():
            acc[(m["dataset"], m["run"], cond)].append(r["fused_acc"])
            extra[(m["dataset"], m["run"], cond)].append((r.get("fused_f1"), r.get("fused_ece")))
        meta.setdefault((m["dataset"], m["run"]), []).append(
            {"params": m.get("trainable_params"), "secs": m.get("train_seconds")})
    rows = []
    for (ds, run, cond), v in sorted(acc.items()):
        v = np.array(v)
        n = len(v)
        mean = float(v.mean())
        sd = float(v.std(ddof=1)) if n > 1 else float("nan")
        if n > 1:
            h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        else:
            h = float("nan")
        ex = [e for e in extra[(ds, run, cond)] if e[0] is not None]
        f1 = float(np.mean([e[0] for e in ex])) if ex else float("nan")
        ece = float(np.mean([e[1] for e in ex])) if ex else float("nan")
        rows.append({"dataset": ds, "run": run, "condition": cond, "n_seeds": n, "mean_f1": f1, "mean_ece": ece,
                     "mean_acc": mean,
                     "std_acc": sd, "ci95_low": max(0.0, mean - h), "ci95_high": min(1.0, mean + h)})
    df = pd.DataFrame(rows)
    out = Path(cfg["paths"]["results"]) / "stats"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.split == "test" else "_dev"
    df.to_csv(out / f"accuracy_by_condition{suffix}.csv", index=False)
    cost = []
    for (ds, run), v in meta.items():
        cost.append({"dataset": ds, "run": run, "n_seeds": len(v),
                     "head_params": v[0]["params"], "mean_train_seconds": float(np.mean([x["secs"] for x in v if x["secs"]]))})
    pd.DataFrame(cost).to_csv(out / "cost_by_run.csv", index=False)
    if len(df):
        print(df[df["condition"].isin(["clean_gold"])].round(4).to_string(index=False))
    print(f"\nSaved -> {out / ('accuracy_by_condition' + suffix + '.csv')} and cost_by_run.csv")
    if len(df) and (df["n_seeds"] < 5).any():
        print("NOTE: some runs have fewer than 5 seeds.")


if __name__ == "__main__":
    main()

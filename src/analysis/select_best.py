import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.utils import load_config, write_json


def main():
    ap = argparse.ArgumentParser(description="Pre-specified model selection on DEV only.")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--min_seeds", type=int, default=5)
    ap.add_argument("--condition_regex", default=r"_(whisper|w2v2ctc)",
                    help="dev conditions averaged for the selection score (default: all real-ASR conditions)")
    ap.add_argument("--exclude_regex", default=r"abl-", help="runs whose name matches are ablations, not candidates")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["checkpoints"]) / args.dataset
    per_run = defaultdict(list)
    for mp in sorted(root.glob("*/seed*/metrics.json")):
        m = json.load(open(mp))
        if m.get("dev_results") and not re.search(args.exclude_regex, m["run"]) and not m["run"].startswith("external-") \
                and not m["run"].startswith("finetune-"):
            per_run[m["run"]].append(m["dev_results"])
    runs = {r: v for r, v in per_run.items() if len(v) >= args.min_seeds}
    if not runs:
        raise SystemExit(f"no run has >= {args.min_seeds} seeds with dev results")
    names = set.intersection(*[set.intersection(*[set(d) for d in v]) for v in runs.values()])
    conds = sorted(c for c in names if re.search(args.condition_regex, c))
    if not conds:
        raise SystemExit("no common dev conditions match --condition_regex")
    all_conds = set().union(*[set().union(*[set(d) for d in v]) for v in runs.values()])
    dropped = sorted(c for c in all_conds - names if re.search(args.condition_regex, c))
    print(f"selection uses {len(conds)} dev conditions common to all {len(runs)} candidate runs")
    if dropped:
        print(f"WARNING: {len(dropped)} matching conditions are missing for some runs and are NOT used: {dropped}")
        print("         (extract the missing dev features for those runs, or the selection score ignores them)")
    rows = []
    for r, v in runs.items():
        per_seed = [np.mean([d[c]["fused_acc"] for c in conds]) for d in v]
        rows.append({"run": r, "fusion": r.split("__")[0], "n_seeds": len(v), "dev_score": float(np.mean(per_seed)),
                     "dev_score_std": float(np.std(per_seed))})
    df = pd.DataFrame(rows).sort_values("dev_score", ascending=False)

    def best(mask):
        sub = df[mask]
        return None if sub.empty else sub.iloc[0]["run"]

    fused = df["fusion"].apply(lambda f: f not in ("text_only", "audio_only"))
    chosen = {"text_only": best(df["fusion"] == "text_only"), "audio_only": best(df["fusion"] == "audio_only"),
              "fused_best": best(fused),
              "per_fusion": {f: best(df["fusion"] == f) for f in sorted(df["fusion"].unique())
                             if f not in ("text_only", "audio_only")}}
    out = {"dataset": args.dataset, "criterion": f"mean dev accuracy over {len(conds)} conditions matching "
           f"'{args.condition_regex}' and averaged over >= {args.min_seeds} seeds", "conditions": conds,
           "dropped_conditions": dropped, "ranking": df.to_dict("records"), "chosen": chosen}
    res = Path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    write_json(res / f"selection_{args.dataset}.json", out)
    df.to_csv(res / f"selection_{args.dataset}_ranking.csv", index=False)
    with open(res / f"selection_{args.dataset}.env", "w") as f:
        f.write(f'TEXT="{chosen["text_only"]}"\nAUDIO="{chosen["audio_only"]}"\nFUSED="{chosen["fused_best"]}"\n')
        f.write('RUNS_FOR_FINAL="' + " ".join(r for r in [chosen["text_only"], chosen["audio_only"], chosen["fused_best"],
                                                         *chosen["per_fusion"].values()] if r) + '"\n')
    print(df.round(4).to_string(index=False))
    print("\nchosen:", json.dumps(chosen, indent=2))
    print(f"\nwrote results/selection_{args.dataset}.json / .env (selection used DEV only)")


if __name__ == "__main__":
    main()

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.utils import load_config


def main():
    ap = argparse.ArgumentParser(description="Pick the best fine-tuning/LoRA learning rate per mode using DEV accuracy only.")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["checkpoints"]) / args.dataset
    accs, dirs = defaultdict(list), defaultdict(list)
    for mp in sorted(root.glob("finetune-*/seed*/metrics.json")):
        m = json.load(open(mp))
        accs[m["run"]].append(m["best_dev_acc"])
        dirs[m["run"]].append(str(mp.parent))
    best = {}
    for run, v in accs.items():
        mode = run.split("__")[0]
        if mode not in best or np.mean(v) > best[mode][1]:
            best[mode] = (run, float(np.mean(v)))
    for mode, (run, a) in sorted(best.items()):
        print(f"{mode}: {run} mean dev acc {a:.4f} ({len(accs[run])} seeds)")
    with open(Path(cfg["paths"]["results"]) / f"selection_finetune_{args.dataset}.env", "w") as f:
        f.write('FT_RUNS="' + " ".join(r for r, _ in best.values()) + '"\n')
        f.write('FT_DIRS="' + " ".join(d for r, _ in best.values() for d in dirs[r]) + '"\n')
        for mode, (run, _) in best.items():
            f.write(f'FT_{mode.split("-")[1].upper()}="{run}"\n')


if __name__ == "__main__":
    main()

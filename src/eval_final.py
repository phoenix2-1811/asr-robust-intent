import argparse
from pathlib import Path

import torch

from src.common.evalutil import evaluate_conditions
from src.common.loadmodel import load_run
from src.common.repro import config_hash
from src.common.utils import load_config, read_json, write_json


def main():
    ap = argparse.ArgumentParser(description="Evaluate already-trained runs on the TEST split (do this once, after selection).")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", action="append", required=True, help="run name; repeat for several runs")
    ap.add_argument("--force", action="store_true", help="re-evaluate runs that already have test results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in args.run:
        root = Path(cfg["paths"]["checkpoints"]) / args.dataset / run
        seeds = sorted(root.glob("seed*"))
        if not seeds:
            raise SystemExit(f"no seeds found for run {run}")
        for sd in seeds:
            mp = sd / "metrics.json"
            m = read_json(mp)
            if m.get("results") and not args.force:
                print(f"[skip] {run}/{sd.name}: test already evaluated {m.get('test_evaluations', 1)}x (use --force to repeat)")
                continue
            model, meta = load_run(sd, device)
            a = meta["arch"]
            res = evaluate_conditions(cfg, model, args.dataset, "test", a["audio_encoder"], a["text_encoder"],
                                      a["audio_stats"], a["audio_pool"], a["text_feature"], device, sd,
                                      save_preds=True, audio_mode=a.get("audio_mode", "pooled"))
            m["results"] = res
            m["test_evaluations"] = int(m.get("test_evaluations", 0)) + 1
            m["test_eval_config_hash"] = config_hash(cfg)
            write_json(mp, m)
            print(f"[test] {run}/{sd.name}: clean_gold={res.get('clean_gold', {}).get('fused_acc')} "
                  f"({len(res)} conditions)")


if __name__ == "__main__":
    main()

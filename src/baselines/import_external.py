import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.common.data import load_examples, load_label_map
from src.common.utils import load_config, write_json


def read_predictions(path):
    path = Path(path)
    preds = {}
    if path.suffix == ".jsonl":
        for line in open(path):
            r = json.loads(line)
            preds[str(r["utt_id"])] = r["pred"]
    else:
        with open(path) as f:
            for r in csv.DictReader(f):
                preds[str(r["utt_id"])] = r["pred"]
    return preds


def import_predictions(cfg, dataset, preds, name, condition="clean_gold", seed=0, split="test"):
    examples = load_examples(cfg, dataset, split)
    label_map = load_label_map(cfg, dataset)
    missing = [ex["utt_id"] for ex in examples if ex["utt_id"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} utterances have no prediction (e.g. {missing[:3]})")
    y = np.array([label_map[ex["label"]] for ex in examples])
    pred = np.array([label_map.get(str(preds[ex["utt_id"]]), -1) if not str(preds[ex["utt_id"]]).isdigit()
                     else int(preds[ex["utt_id"]]) for ex in examples])
    unknown = int((pred < 0).sum())
    run = f"external-{name}"
    out_dir = Path(cfg["paths"]["checkpoints"]) / dataset / run / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(y)
    np.savez_compressed(out_dir / f"preds_{split}_{condition}.npz", utt_ids=np.array([ex["utt_id"] for ex in examples]),
                        groups=np.array([str(ex.get("group", ex["utt_id"])) for ex in examples]), labels=y,
                        fused_pred=pred, text_pred=np.full(n, -1), audio_pred=np.full(n, -1),
                        fused_conf=np.zeros(n), text_conf=np.zeros(n), audio_conf=np.zeros(n))
    mp = out_dir / "metrics.json"
    m = json.load(open(mp)) if mp.exists() else {"run": run, "dataset": dataset, "seed": seed, "results": {}}
    m["results"][condition] = {"fused_acc": float((pred == y).mean()), "n": int(n), "text_acc": None,
                               "audio_acc": None, "unmapped_predictions": unknown}
    write_json(mp, m)
    print(f"{run} [{condition}] acc={(pred == y).mean():.4f} n={n} unmapped={unknown} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred_file", required=True, help="CSV (utt_id,pred) or JSONL; pred is a label string or id")
    ap.add_argument("--name", required=True)
    ap.add_argument("--condition", default="clean_gold")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    import_predictions(cfg, args.dataset, read_predictions(args.pred_file), args.name, args.condition, args.seed)


if __name__ == "__main__":
    main()

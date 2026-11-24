import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.common.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", default="slurp")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["checkpoints"]) / args.dataset
    acc = defaultdict(list)
    for p in sorted(root.glob("*/seed*/best.pt")):
        sd = torch.load(p, map_location="cpu")
        k = "audio_pool.layer_logits"
        if k in sd:
            acc[p.parent.parent.name].append(torch.softmax(sd[k], 0).numpy())
    for run, ws in acc.items():
        w = np.mean(ws, axis=0)
        print(f"{run}\n  layer weights: {np.round(w, 3).tolist()}\n  argmax layer = {int(w.argmax())} (use as audio_features.frame_layer)")
    if not acc:
        print("no weighted-layer checkpoints found")


if __name__ == "__main__":
    main()

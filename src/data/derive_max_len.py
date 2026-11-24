import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.common.utils import load_config, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--tokenizer", default="roberta-base")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths = []
    for key, d in cfg["datasets"].items():
        p = Path(d["root"]) / "train.jsonl"
        if not p.exists():
            continue
        for ex in read_jsonl(p):
            lengths.append(len(tok.encode(ex["transcript"], add_special_tokens=True)))
    ceiling = cfg.get("max_text_len_ceiling", 128)
    derived = min(int(np.percentile(lengths, 99)) + 16, ceiling) if lengths else ceiling
    Path("data").mkdir(exist_ok=True)
    with open("data/derived_max_len.json", "w") as f:
        json.dump({"max_text_len": derived}, f, indent=2)
    print(f"max_text_len={derived}")


if __name__ == "__main__":
    main()

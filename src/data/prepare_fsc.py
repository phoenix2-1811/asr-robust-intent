import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from src.common.text_norm import normalize_text
from src.common.utils import load_config


def read_rows(root, csv_name):
    p = root / csv_name
    if not p.exists():
        print(f"!! Missing {p}")
        return []
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            rows.append({
                "audio_path": str(root / r["path"]),
                "transcript": r["transcription"],
                "label": f"{r['action']}_{r['object']}_{r['location']}",
                "utt_id": Path(r["path"]).stem,
                "group": normalize_text(r["transcription"]),
            })
    return rows


def unseen_utterance_split(rows, seed, fracs=(0.7, 0.1, 0.2)):
    rng = random.Random(seed)
    by_label = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_label[r["label"]][r["group"]].append(r)
    train, dev, test = [], [], []
    for label in sorted(by_label):
        groups = sorted(by_label[label])
        rng.shuffle(groups)
        n = len(groups)
        if n < 3:
            for g in groups:
                train += by_label[label][g]
            continue
        n_dev = max(1, int(round(n * fracs[1])))
        n_test = max(1, int(round(n * fracs[2])))
        n_train = n - n_dev - n_test
        if n_train < 1:
            n_train, n_dev, n_test = n - 2, 1, 1
        for i, g in enumerate(groups):
            tgt = train if i < n_train else (dev if i < n_train + n_dev else test)
            tgt += by_label[label][g]
    return train, dev, test


def write_split(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    keys = [args.dataset] if args.dataset else [k for k, v in cfg["datasets"].items() if v.get("type") == "fsc"]
    for key in keys:
        dcfg = cfg["datasets"][key]
        src_root = Path(dcfg.get("source_root", dcfg["root"]))
        root = src_root / "fluent_speech_commands_dataset"
        out_root = Path(dcfg["root"])
        out_root.mkdir(parents=True, exist_ok=True)
        tr = read_rows(root, "train_data.csv")
        dv = read_rows(root, "valid_data.csv")
        te = read_rows(root, "test_data.csv")
        if dcfg.get("split_mode", "official") == "unseen_utterance":
            tr, dv, te = unseen_utterance_split(tr + dv + te, dcfg.get("split_seed", 0))
        for name, rows in (("train", tr), ("dev", dv), ("test", te)):
            write_split(out_root / f"{name}.jsonl", rows)
        train_sent = {r["group"] for r in tr}
        test_sent = {r["group"] for r in te}
        overlap = len(test_sent & train_sent)
        print(f"[{key}] train={len(tr)} dev={len(dv)} test={len(te)} | unique test sentences={len(test_sent)} "
              f"seen in train={overlap}")
        labels = sorted({r["label"] for r in tr + dv + te})
        with open(out_root / "label_map.json", "w") as f:
            json.dump({lab: i for i, lab in enumerate(labels)}, f, indent=2)
        print(f"[{key}] {len(labels)} classes")


if __name__ == "__main__":
    main()

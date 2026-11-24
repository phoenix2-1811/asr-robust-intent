import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.common.utils import load_config


def resolve_label(entry, granularity):
    scenario = entry.get("scenario", "unknown")
    action = entry.get("action", "unknown")
    if granularity == "scenario":
        return scenario
    if granularity == "action":
        return action
    if granularity == "scenario_action":
        return f"{scenario}_{action}"
    raise ValueError(f"Unknown granularity: {granularity}")


def find_audio_path(audio_root, recording_file, subs):
    for sub in subs:
        cand = audio_root / sub / recording_file
        if cand.exists():
            return str(cand)
    return None


def process_split(split_file, audio_root, out_path, granularity, split_name, subs):
    n_written, n_skipped = 0, 0
    with open(split_file) as fin, open(out_path, "w") as fout:
        for line in tqdm(fin, desc=f"SLURP {split_name}"):
            entry = json.loads(line)
            label = resolve_label(entry, granularity)
            for rec in entry.get("recordings", []):
                audio_path = find_audio_path(audio_root, rec["file"], subs)
                if audio_path is None:
                    n_skipped += 1
                    continue
                out = {
                    "audio_path": audio_path,
                    "transcript": entry.get("sentence", ""),
                    "label": label,
                    "utt_id": rec["file"],
                    "group": str(entry.get("slurp_id", entry.get("sentence", ""))),
                }
                fout.write(json.dumps(out) + "\n")
                n_written += 1
    print(f"[{split_name}] wrote {n_written}, skipped {n_skipped} (missing audio)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", default=None, help="config key; default: all datasets of type slurp")
    ap.add_argument("--audio_subsets", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    keys = [args.dataset] if args.dataset else [k for k, v in cfg["datasets"].items() if v.get("type") == "slurp"]
    for key in keys:
        dcfg = cfg["datasets"][key]
        raw = Path(dcfg["raw_root"])
        ann_root = raw / "dataset" / "slurp"
        audio_root = raw / "audio"
        out_root = Path(dcfg["root"])
        out_root.mkdir(parents=True, exist_ok=True)
        if not audio_root.exists():
            print(f"!! {audio_root} does not exist")
        split_map = {"train": "train.jsonl", "dev": "devel.jsonl", "test": "test.jsonl"}
        labels = set()
        for split_name, fname in split_map.items():
            sf = ann_root / fname
            if not sf.exists():
                print(f"!! Missing {sf}, skipping {split_name}")
                continue
            out_path = out_root / f"{split_name}.jsonl"
            process_split(sf, audio_root, out_path, dcfg["granularity"], split_name,
                          args.audio_subsets or dcfg.get("audio_subsets", ["slurp_real"]))
            with open(out_path) as f:
                for line in f:
                    labels.add(json.loads(line)["label"])
        label_map = {lab: i for i, lab in enumerate(sorted(labels))}
        with open(out_root / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)
        print(f"[{key}] {dcfg['granularity']}: {len(label_map)} classes -> {out_root / 'label_map.json'}")


if __name__ == "__main__":
    main()

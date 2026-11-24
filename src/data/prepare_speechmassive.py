import argparse
import json
from pathlib import Path

from src.common.text_norm import normalize_text
from src.common.utils import load_config


def convert_rows(rows, out_root, split_name, label_field):
    wav_dir = Path(out_root) / "wavs" / split_name
    wav_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for row in rows:
        uid = f"{split_name}_{row['id']}"
        audio = row["audio"]
        path = wav_dir / f"{uid}.wav"
        if audio.get("bytes") is not None:
            path.write_bytes(audio["bytes"])
        else:
            import soundfile as sf
            sf.write(path, audio["array"], audio["sampling_rate"])
        out.append({"audio_path": str(path), "transcript": row["utt"], "label": row[label_field], "utt_id": uid,
                    "group": normalize_text(row["utt"])})
    return out


def load_split(repo, locale, split):
    from datasets import Audio, load_dataset
    try:
        ds = load_dataset(repo, locale, split=split)
    except Exception:
        ds = load_dataset(repo, locale, split=split, trust_remote_code=True)
    return ds.cast_column("audio", Audio(decode=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True, help="config key of type speechmassive")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["datasets"][args.dataset]
    root = Path(d["root"])
    root.mkdir(parents=True, exist_ok=True)
    plan = [("train", d["hf_repo"], d["train_split"]), ("dev", d["hf_repo"], d["dev_split"]),
            ("test", d["hf_test_repo"], d["test_split"])]
    labels = set()
    for name, repo, split in plan:
        try:
            ds = load_split(repo, d["locale"], split)
        except Exception as e:
            raise SystemExit(f"Could not load {repo} [{d['locale']}] split '{split}': {e}\n"
                             f"Check which splits exist for this locale on the dataset page "
                             f"(e.g. use train_split: train_115) and edit configs/experiment.yaml.")
        rows = convert_rows(ds, root, name, d["label_field"])
        with open(root / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
                labels.add(r["label"])
        print(f"[{args.dataset}] {name}: {len(rows)} utterances")
    with open(root / "label_map.json", "w") as f:
        json.dump({lab: i for i, lab in enumerate(sorted(labels))}, f, indent=2)
    print(f"[{args.dataset}] {len(labels)} classes")


if __name__ == "__main__":
    main()

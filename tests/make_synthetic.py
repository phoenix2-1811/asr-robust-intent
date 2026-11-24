import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.text_norm import utterance_wer

K, DT, DA, L = 12, 48, 32, 13
rng = np.random.default_rng(0)
proto_t = rng.standard_normal((K, DT)).astype(np.float32) * 1.0
proto_a = rng.standard_normal((K, L, DA)).astype(np.float32)
vocab = [f"w{i}" for i in range(60)]


def make_split(name, n_sent_per_class, recs):
    rows = []
    sents = []
    for k in range(K):
        for s in range(n_sent_per_class):
            words = [vocab[(k * 5 + s * 3 + j) % len(vocab)] for j in range(4 + (s % 3))]
            sents.append((k, " ".join(words), f"{name}-{k}-{s}"))
    for k, text, gid in sents:
        for r in range(recs):
            rows.append({"audio_path": "x.wav", "transcript": text, "label": f"c{k}", "utt_id": f"{gid}-r{r}", "group": gid})
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(root):
    root = Path(root)
    import os
    os.chdir(root)
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"))
    cfg["datasets"] = {"synth": {"type": "jsonl", "root": "data/synth", "max_audio_seconds": 5}}
    cfg["conditions"]["audio"] = ["clean", "snr10"]
    cfg["conditions"]["eval_asr_models"] = ["whisper-tiny", "whisper-small"]
    cfg["conditions"]["asr_models"] = ["whisper-tiny", "whisper-small"]
    cfg["train"]["train_pairs"] = [{"audio": "clean", "text": "gold"}, {"audio": "clean", "text": "whisper-small@clean"}]
    cfg["train"]["epochs"] = 40
    cfg["train"]["patience"] = 5
    cfg["train"]["batch_size"] = 128
    cfg["model"]["audio_stats"] = ["mean", "std"]
    cfg["analysis"]["bootstrap_samples"] = 500
    yaml.safe_dump(cfg, open("test_config.yaml", "w"))
    json.dump({f"c{k}": k for k in range(K)}, open("label_map_tmp.json", "w"))
    splits = {"train": make_split("tr", 10, 4), "dev": make_split("dv", 3, 4), "test": make_split("te", 6, 5)}
    for name, rows in splits.items():
        write_jsonl(Path("data/synth") / f"{name}.jsonl", rows)
    json.dump({f"c{k}": k for k in range(K)}, open("data/synth/label_map.json", "w"))
    err_rate = {("whisper-small", "clean"): 0.08, ("whisper-tiny", "clean"): 0.18,
                ("whisper-small", "snr10"): 0.30, ("whisper-tiny", "snr10"): 0.45}
    aud_noise = {"clean": 0.9, "snr10": 1.3}
    gold_err = 0.0
    for name, rows in splits.items():
        n = len(rows)
        y = np.array([int(r["label"][1:]) for r in rows])
        r = np.random.default_rng(hash(name) % 1000)
        for cond in ("clean", "snr10"):
            a = proto_a[y] + r.standard_normal((n, L, DA)).astype(np.float32) * aud_noise[cond]
            a_std = np.abs(r.standard_normal((n, L, DA)).astype(np.float32)) * 0.3
            d = Path("features/synth") / name
            d.mkdir(parents=True, exist_ok=True)
            lens = r.integers(5, 13, n)
            fr = [torch.from_numpy(proto_a[y[i], 8][None, :] + r.standard_normal((lens[i], DA)).astype(np.float32) * aud_noise[cond] * 1.5).half() for i in range(n)]
            offs = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(torch.tensor(lens), 0)])
            torch.save({"utt_ids": [x["utt_id"] for x in rows], "frames": torch.cat(fr), "offsets": offs},
                       Path("features/synth") / name / f"frames__fakeaudio__{cond}.pt")
            torch.save({"utt_ids": [x["utt_id"] for x in rows], "mean": torch.from_numpy(a).half(),
                        "std": torch.from_numpy(a_std).half(), "n_frames": torch.ones(n)},
                       d / f"audio__fakeaudio__{cond}.pt")
        def text_feats(err_flags):
            lab = np.where(err_flags, r.integers(0, K, n), y)
            return proto_t[lab] + r.standard_normal((n, DT)).astype(np.float32) * 0.7
        t = text_feats(np.zeros(n, dtype=bool))
        torch.save({"utt_ids": [x["utt_id"] for x in rows], "cls": torch.from_numpy(t), "mean": torch.from_numpy(t)},
                   Path("features/synth") / name / "text__faketext__gold.pt")
        for (m, cond), p in err_rate.items():
            err = r.random(n) < p
            t = text_feats(err)
            src = f"{m}@{cond}"
            torch.save({"utt_ids": [x["utt_id"] for x in rows], "cls": torch.from_numpy(t), "mean": torch.from_numpy(t)},
                       Path("features/synth") / name / f"text__faketext__{src.replace('@', '-')}.pt")
            utts = {}
            for i, row in enumerate(rows):
                words = row["transcript"].split()
                hyp = words if not err[i] else words[:-1] + ["zzz"]
                lp = (-0.15 - 0.1 * r.random()) if not err[i] else (-0.9 - 0.4 * r.random())
                utts[row["utt_id"]] = {"text": " ".join(hyp), "avg_logprob": float(lp),
                                       "no_speech_prob": float(0.02 + 0.03 * r.random()), "compression_ratio": 1.5,
                                       "wer": utterance_wer(row["transcript"], " ".join(hyp)),
                                       "n_ref_words": len(words)}
            Path(f"asr/synth/{name}").mkdir(parents=True, exist_ok=True)
            json.dump({"source": src, "model": m, "audio_cond": cond, "wer": float(np.mean([u["wer"] for u in utts.values()])),
                       "utts": utts}, open(f"asr/synth/{name}/{src}.json", "w"))
    print("synthetic data ready in", root)


if __name__ == "__main__":
    main(sys.argv[1])

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.asr.transcribe import asr_out_path
from src.common.data import _hyp_disagreement
from src.common.utils import load_config, read_jsonl, read_json, write_json


def fold_of(group, k):
    return int(hashlib.sha1(str(group).encode()).hexdigest()[:8], 16) % k


def build_features(cfg, dataset, split, source, examples, ref_model):
    data = read_json(asr_out_path(cfg, dataset, split, source))
    cond = source.split("@", 1)[1]
    ref_path = asr_out_path(cfg, dataset, split, f"{ref_model}@{cond}")
    ref = read_json(ref_path) if ref_path.exists() and ref_model != source.split("@")[0] else None
    X = np.zeros((len(examples), 7), dtype=np.float32)
    y = np.full(len(examples), np.nan, dtype=np.float32)
    for i, ex in enumerate(examples):
        u = data["utts"].get(ex["utt_id"])
        if u is None or "avg_logprob" not in u:
            continue
        xd = _hyp_disagreement(u["text"], ref["utts"][ex["utt_id"]]["text"]) if ref is not None and ex["utt_id"] in ref["utts"] else 0.0
        words = u["text"].split()
        X[i] = [u["avg_logprob"], u["no_speech_prob"], u["compression_ratio"], np.log1p(len(words)), xd,
                len(u["text"]) / max(len(words), 1), float(len(u["text"]))]
        if "wer" in u:
            y[i] = u["wer"]
    return X, y, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="ASR model names, e.g. whisper-small w2v2ctc-base-960h")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--max_iter", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ref_model = cfg["reliability"]["reference_model"]
    fallback = cfg["reliability"]["fallback_reference_model"]
    root = Path(cfg["paths"]["asr"]) / args.dataset
    ex_cache = {}

    def examples(split):
        if split not in ex_cache:
            ex_cache[split] = read_jsonl(Path(cfg["datasets"][args.dataset]["root"]) / f"{split}.jsonl")
        return ex_cache[split]

    for model in args.models:
        src_train = f"{model}@clean"
        if not asr_out_path(cfg, args.dataset, "train", src_train).exists():
            print(f"!! {src_train} train transcripts missing, skipping {model}")
            continue
        rm = fallback if model == ref_model else ref_model
        ex_tr = examples("train")
        X, y, _ = build_features(cfg, args.dataset, "train", src_train, ex_tr, rm)
        ok = ~np.isnan(y)
        folds = np.array([fold_of(ex.get("group", ex["utt_id"]), args.n_folds) for ex in ex_tr])
        models = []
        for k in range(args.n_folds):
            tr = ok & (folds != k)
            m = HistGradientBoostingRegressor(max_iter=args.max_iter, learning_rate=0.1, max_depth=4, random_state=k)
            m.fit(X[tr], y[tr])
            models.append(m)
        oof = np.zeros(len(ex_tr))
        for k in range(args.n_folds):
            sel = folds == k
            if sel.any():
                oof[sel] = models[k].predict(X[sel])
        r = np.corrcoef(oof[ok], y[ok])[0, 1] if ok.sum() > 2 else float("nan")
        print(f"[{model}] out-of-fold Pearson r(pred, true utterance WER) on train@clean = {r:.3f}")
        for split in ("train", "dev", "test"):
            for path in sorted((root / split).glob(f"{model}@*.json")):
                source = path.stem
                if ".shard" in source:
                    continue
                exs = examples(split)
                Xs, _, data = build_features(cfg, args.dataset, split, source, exs, rm)
                if split == "train":
                    fo = np.array([fold_of(ex.get("group", ex["utt_id"]), args.n_folds) for ex in exs])
                    pred = np.zeros(len(exs))
                    for k in range(args.n_folds):
                        sel = fo == k
                        if sel.any():
                            pred[sel] = models[k].predict(Xs[sel])
                else:
                    pred = np.mean([m.predict(Xs) for m in models], axis=0)
                pred = np.clip(pred, 0.0, 1.5)
                for i, ex in enumerate(exs):
                    u = data["utts"].get(ex["utt_id"])
                    if u is not None:
                        u["pred_wer"] = float(pred[i])
                write_json(path, data)
                print(f"  wrote pred_wer -> {path}")


if __name__ == "__main__":
    main()

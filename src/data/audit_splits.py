import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.common.data import load_examples, load_label_map
from src.common.text_norm import normalize_text
from src.common.utils import load_config, write_json


def audit(cfg, key):
    ex = {s: load_examples(cfg, key, s) for s in ("train", "dev", "test")}
    sents = {s: {normalize_text(e["transcript"]) for e in v} for s, v in ex.items()}
    ids = {s: {e["utt_id"] for e in v} for s, v in ex.items()}
    lm = load_label_map(cfg, key)
    train_labels = Counter(e["label"] for e in ex["train"])
    majority = train_labels.most_common(1)[0][0]
    by_sent = defaultdict(Counter)
    for e in ex["train"]:
        by_sent[normalize_text(e["transcript"])][e["label"]] += 1
    n_test = len(ex["test"])
    hit_major = sum(e["label"] == majority for e in ex["test"])
    seen = [normalize_text(e["transcript"]) in by_sent for e in ex["test"]]
    lookup_correct = [(by_sent[normalize_text(e["transcript"])].most_common(1)[0][0] if s else majority) == e["label"]
                      for e, s in zip(ex["test"], seen)]
    seen_arr = np.array(seen)
    lc = np.array(lookup_correct)
    out = {
        "dataset": key, "n_utts": {s: len(v) for s, v in ex.items()},
        "n_unique_sentences": {s: len(v) for s, v in sents.items()}, "n_classes": len(lm),
        "test_sentences_seen_in_train_frac": len(sents["test"] & sents["train"]) / max(len(sents["test"]), 1),
        "test_utterances_with_seen_sentence_frac": float(seen_arr.mean()),
        "dev_sentences_seen_in_train_frac": len(sents["dev"] & sents["train"]) / max(len(sents["dev"]), 1),
        "utt_id_overlap": {"train_test": len(ids["train"] & ids["test"]), "train_dev": len(ids["train"] & ids["dev"]),
                           "dev_test": len(ids["dev"] & ids["test"])},
        "majority_class_baseline_test_acc": hit_major / n_test,
        "sentence_lookup_baseline_test_acc": float(lc.mean()),
        "sentence_lookup_acc_on_seen": float(lc[seen_arr].mean()) if seen_arr.any() else None,
        "interpretation": "A high sentence_lookup_baseline_test_acc means text matching alone solves the benchmark; "
                          "report results on the unseen-sentence slice (src.analysis.seen_unseen) in that case.",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    keys = [args.dataset] if args.dataset else [k for k in cfg["datasets"] if (Path(cfg["datasets"][k]["root"]) / "test.jsonl").exists()]
    for k in keys:
        out = audit(cfg, k)
        write_json(Path(cfg["paths"]["results"]) / f"audit_{k}.json", out)
        print(f"[{k}] test sentences seen in train: {100 * out['test_sentences_seen_in_train_frac']:.1f}% | "
              f"majority-class acc {100 * out['majority_class_baseline_test_acc']:.1f}% | "
              f"sentence-lookup acc {100 * out['sentence_lookup_baseline_test_acc']:.1f}% | utt-id overlaps {out['utt_id_overlap']}")


if __name__ == "__main__":
    main()

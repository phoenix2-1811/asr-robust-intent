import argparse
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.data import load_asr, load_examples, load_label_map
from src.common.text_norm import normalize_text, word_edit_ops
from src.common.utils import load_config, write_json

NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                   "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
                   "hundred thousand first second third fourth fifth half quarter noon midnight".split())
NEGATIONS = {"no", "not", "never", "don't", "dont", "doesn't", "didn't", "won't", "cannot", "can't", "stop", "cancel",
             "off", "without", "nothing", "none", "neither", "nor", "isn't", "aren't", "wasn't"}
FUNCTION = set("a an the of to in on at for from by with and or but is are was were be been it its this that these "
               "those my me i you your we our they them he she his her as if so than then do does did can could "
               "will would shall should may might have has had about up down out into over under".split())


def word_category(w):
    if w is None:
        return "none"
    if re.search(r"\d", w) or w in NUMBER_WORDS:
        return "number/time"
    if w in NEGATIONS:
        return "negation/polarity"
    if w in FUNCTION:
        return "function word"
    return "content word"


def error_categories(ref, hyp):
    cats = []
    for op, r, h in word_edit_ops(ref, hyp):
        if op == "ok":
            continue
        cats.append((op, word_category(r if r is not None else h)))
    return cats


def mean_correct(paths, key, y):
    return np.mean([(np.load(p, allow_pickle=False)[key] == y) for p in paths], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--asr_source", required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--text_only", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    pf, pt = sorted(glob.glob(args.fused)), sorted(glob.glob(args.text_only))
    ref = np.load(pf[0], allow_pickle=False)
    y, ids = ref["labels"], [str(u) for u in ref["utt_ids"]]
    f_ok = mean_correct(pf, "fused_pred", y)
    t_ok = mean_correct(pt, "fused_pred", y)
    a_ok = mean_correct(pf, "audio_pred", y)
    examples = {ex["utt_id"]: ex for ex in load_examples(cfg, args.dataset, "test")}
    asr = load_asr(cfg, args.dataset, "test", args.asr_source)["utts"]
    inv = {v: k for k, v in load_label_map(cfg, args.dataset).items()}

    cat_utts = defaultdict(list)
    n_err_words = Counter()
    for i, u in enumerate(ids):
        gold = normalize_text(examples[u]["transcript"])
        cats = error_categories(gold, asr[u]["text"])
        for op, c in cats:
            n_err_words[(op, c)] += 1
        for c in {c for _, c in cats}:
            cat_utts[c].append(i)
        if not cats:
            cat_utts["no ASR error"].append(i)

    rows = []
    for c, idx in sorted(cat_utts.items()):
        idx = np.array(idx)
        wrong_t = 1 - t_ok[idx]
        rows.append({"utterances_with": c, "n_utts": len(idx), "text_only_acc": float(t_ok[idx].mean()),
                     "fused_acc": float(f_ok[idx].mean()), "audio_branch_acc": float(a_ok[idx].mean()),
                     "recovered_rate": float(((1 - t_ok[idx]) * f_ok[idx]).sum() / max(wrong_t.sum(), 1e-9)),
                     "lost_rate": float((t_ok[idx] * (1 - f_ok[idx])).sum() / max(t_ok[idx].sum(), 1e-9))})
    df = pd.DataFrame(rows)
    df.to_csv(args.out_prefix + "_by_error_type.csv", index=False)

    conf = Counter()
    rec = Counter()
    tp = np.mean([np.load(p, allow_pickle=False)["fused_pred"] for p in pt], axis=0)
    for i in range(len(y)):
        if t_ok[i] < 0.5:
            key = (inv[int(y[i])], inv.get(int(round(tp[i])), "?"))
            conf[key] += 1
            rec[key] += float(f_ok[i])
    conf_rows = [{"true_intent": k[0], "text_only_predicted": k[1], "count": c, "fused_recovered": rec[k]}
                 for k, c in conf.most_common(args.top_k)]
    pd.DataFrame(conf_rows).to_csv(args.out_prefix + "_top_confusions.csv", index=False)
    pd.DataFrame([{"op": k[0], "category": k[1], "count": v} for k, v in sorted(n_err_words.items())]).to_csv(
        args.out_prefix + "_asr_error_words.csv", index=False)
    print(df.round(3).to_string(index=False))
    print("\nTop text-only confusions and how many the fused model recovers:")
    print(pd.DataFrame(conf_rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.data import load_asr
from src.common.utils import load_config, write_json


def per_seed_stack(paths):
    arrs = []
    ref = None
    for p in paths:
        d = np.load(p, allow_pickle=False)
        if ref is None:
            ref = d
        arrs.append(d)
    return arrs, ref


def bucket_table(mask_defs, text_c, audio_c, fused_c, oracle_c, textonly_c=None):
    rows = []
    for name, m in mask_defs:
        if m.sum() == 0:
            continue
        r = {"bucket": name, "n": int(m.sum()), "text_branch": float(text_c[m].mean()),
             "audio_branch": float(audio_c[m].mean()), "oracle_text_or_audio": float(oracle_c[m].mean()),
             "fused": float(fused_c[m].mean())}
        if textonly_c is not None:
            r["text_only_baseline"] = float(textonly_c[m].mean())
            r["fused_minus_text_only"] = r["fused"] - r["text_only_baseline"]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fused", required=True, help="glob of fused-run preds npz (one per seed)")
    ap.add_argument("--text_only", default=None, help="glob of text-only-run preds npz (one per seed)")
    ap.add_argument("--asr_source", default=None, help="e.g. whisper-small@clean (for WER / confidence buckets)")
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    fused, ref = per_seed_stack(sorted(glob.glob(args.fused)))
    y = ref["labels"]
    tc = np.mean([(d["text_pred"] == y) for d in fused], axis=0)
    ac = np.mean([(d["audio_pred"] == y) for d in fused], axis=0)
    fc = np.mean([(d["fused_pred"] == y) for d in fused], axis=0)
    oc = np.mean([((d["text_pred"] == y) | (d["audio_pred"] == y)) for d in fused], axis=0)
    toc = None
    if args.text_only:
        to, ref2 = per_seed_stack(sorted(glob.glob(args.text_only)))
        if not np.array_equal(ref["utt_ids"], ref2["utt_ids"]):
            raise SystemExit("text-only and fused preds cover different utterances")
        toc = np.mean([(d["fused_pred"] == y) for d in to], axis=0)

    overall = bucket_table([("all", np.ones(len(y), dtype=bool))], tc, ac, fc, oc, toc)
    out = {"overall": overall}

    if args.asr_source:
        split = "test"
        asr = load_asr(cfg, args.dataset, split, args.asr_source)
        ids = [str(u) for u in ref["utt_ids"]]
        wer = np.array([asr["utts"][u]["wer"] for u in ids])
        edges = cfg["analysis"]["wer_buckets"]
        defs = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (wer >= lo) & (wer < hi)
            defs.append((f"utt_WER[{lo:g},{hi:g})", m))
        out["by_wer"] = bucket_table(defs, tc, ac, fc, oc, toc)
        if "avg_logprob" in next(iter(asr["utts"].values())):
            lp = np.array([asr["utts"][u]["avg_logprob"] for u in ids])
            q = cfg["analysis"]["confidence_quantiles"]
            qs = np.quantile(lp, np.linspace(0, 1, q + 1))
            defs = []
            for i in range(q):
                m = (lp >= qs[i]) & ((lp <= qs[i + 1]) if i == q - 1 else (lp < qs[i + 1]))
                defs.append((f"conf_q{i + 1}_of_{q}(avg_logprob {qs[i]:.2f}..{qs[i + 1]:.2f})", m))
            out["by_asr_confidence"] = bucket_table(defs, tc, ac, fc, oc, toc)

    write_json(args.out_prefix + ".json", out)
    for k in ("overall", "by_wer", "by_asr_confidence"):
        if k in out:
            df = pd.DataFrame(out[k])
            df.to_csv(f"{args.out_prefix}_{k}.csv", index=False)
            print(f"\n== {k} ==")
            print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

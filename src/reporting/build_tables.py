import argparse
from pathlib import Path

import pandas as pd

from src.asr.transcribe import asr_out_path
from src.common.utils import load_config, read_json


def fmt(r):
    if r is None:
        return "PENDING"
    if pd.isna(r["std_acc"]):
        return f"{100 * r['mean_acc']:.1f} (n={int(r['n_seeds'])})"
    return f"{100 * r['mean_acc']:.1f} ± {100 * r['std_acc']:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", default="slurp")
    ap.add_argument("--conditions", nargs="+", default=[
        "clean_gold", "clean_whisper-small", "snr10_whisper-small", "snr0_whisper-small",
        "rev400+snr10_whisper-small", "clean_synth-wer20"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir = Path(cfg["paths"]["results"])
    acc = pd.read_csv(res_dir / "stats" / "accuracy_by_condition.csv")
    acc = acc[acc["dataset"] == args.dataset]
    lines = [f"## Accuracy (%), mean ± std over seeds — {args.dataset}\n"]
    lines.append("| System | " + " | ".join(args.conditions) + " |")
    lines.append("|---|" + "---|" * len(args.conditions))
    for run in sorted(acc["run"].unique()):
        cells = []
        for c in args.conditions:
            m = acc[(acc["run"] == run) & (acc["condition"] == c)]
            cells.append(fmt(m.iloc[0] if len(m) else None))
        lines.append(f"| {run} | " + " | ".join(cells) + " |")

    lines.append("\n## Measured corpus WER of each transcript condition (test split)\n")
    lines.append("| Condition | WER (%) |")
    lines.append("|---|---|")
    for c in args.conditions:
        if c == "clean_gold":
            lines.append("| clean_gold | 0.0 |")
            continue
        ac, model_src = c.split("_", 1)
        src = f"{model_src}@{ac}" if model_src.startswith("whisper") else model_src
        p = asr_out_path(cfg, args.dataset, "test", src)
        lines.append(f"| {c} | {100 * read_json(p)['wer']:.1f} |" if p.exists() else f"| {c} | PENDING |")

    cost_p = res_dir / "stats" / "cost_by_run.csv"
    if cost_p.exists():
        cost = pd.read_csv(cost_p)
        cost = cost[cost["dataset"] == args.dataset]
        lines.append("\n## Training cost\n")
        lines.append("| System | Params | Mean train time (s) | Seeds |")
        lines.append("|---|---|---|---|")
        for _, r in cost.iterrows():
            prm = "n/a" if pd.isna(r["head_params"]) else f"{int(r['head_params']):,}"
            sec = "n/a" if pd.isna(r["mean_train_seconds"]) else f"{r['mean_train_seconds']:.0f}"
            lines.append(f"| {r['run']} | {prm} | {sec} | {int(r['n_seeds'])} |")

    text = "\n".join(lines)
    out = Path(args.out) if args.out else res_dir / f"tables_{args.dataset}.md"
    out.write_text(text)
    print(text)
    print(f"\nWritten -> {out}")


if __name__ == "__main__":
    main()

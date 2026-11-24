"""Zero-shot speech-LLM intent classification baseline (Qwen2-Audio style).

NOT TESTED in the development environment (no GPU/model access); transformers' audio-LLM APIs change between
versions, so expect to adjust the processor call. Output is a predictions CSV and an imported checkpoint dir, so
all downstream analysis works unchanged.
"""
import argparse
import csv
import difflib
from pathlib import Path

import torch
from tqdm import tqdm

from src.baselines.import_external import import_predictions
from src.common.data import load_examples, load_label_map
from src.common.utils import load_config
from src.data.audio_conditions import apply_condition, load_wav


def closest_label(text, labels):
    t = text.strip().strip(".").lower()
    for l in labels:
        if t == l.lower():
            return l
    for l in labels:
        if l.lower() in t:
            return l
    m = difflib.get_close_matches(t, [l.lower() for l in labels], n=1, cutoff=0.0)
    return next(l for l in labels if l.lower() == m[0]) if m else labels[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--audio_cond", default="clean")
    ap.add_argument("--max_utts", type=int, default=None)
    ap.add_argument("--name", default="qwen2-audio-zeroshot")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                               device_map="auto").eval()
    labels = sorted(load_label_map(cfg, args.dataset))
    examples = load_examples(cfg, args.dataset, "test")
    if args.max_utts:
        examples = examples[: args.max_utts]
    question = ("Classify the intent of this spoken command. Choose exactly one label from this list: "
                + ", ".join(labels) + ". Answer with the label only.")
    conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    preds = {}
    for ex in tqdm(examples):
        wav = apply_condition(load_wav(ex["audio_path"], 16000, 30), 16000, args.audio_cond, ex["utt_id"],
                              cfg["noise"]["noise_dir"])
        try:
            inputs = proc(text=prompt, audio=[wav], sampling_rate=16000, return_tensors="pt", padding=True)
        except TypeError:
            inputs = proc(text=prompt, audios=[wav], sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        preds[ex["utt_id"]] = closest_label(text, labels)
    cond_name = "clean_gold" if args.audio_cond == "clean" else f"{args.audio_cond}_audio-only"
    out_csv = Path(cfg["paths"]["results"]) / "external" / f"{args.name}_{args.dataset}_{args.audio_cond}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "pred"])
        for k, v in preds.items():
            w.writerow([k, v])
    if args.max_utts is None:
        import_predictions(cfg, args.dataset, preds, args.name, cond_name)


if __name__ == "__main__":
    main()

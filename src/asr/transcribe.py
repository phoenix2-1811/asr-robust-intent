import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.common.text_norm import corpus_wer, normalize_text, utterance_wer
from src.common.utils import load_config, read_jsonl, write_json
from src.data.audio_conditions import apply_condition, load_wav


def asr_out_path(cfg, dataset, split, source, shard=None):
    d = Path(cfg["paths"]["asr"]) / dataset / split
    d.mkdir(parents=True, exist_ok=True)
    if shard is None:
        return d / f"{source}.json"
    return d / f"{source}.shard{shard[0]}of{shard[1]}.json"


def build_output(source, dataset, split, model, cond, examples, results):
    refs, hyps, utts = [], [], {}
    for ex in examples:
        r = results.get(ex["utt_id"])
        if r is None:
            continue
        ref = normalize_text(ex["transcript"])
        hyp = r["text"]
        refs.append(ref)
        hyps.append(hyp)
        utts[ex["utt_id"]] = {**r, "wer": utterance_wer(ref, hyp), "n_ref_words": len(ref.split())}
    return {"source": source, "dataset": dataset, "split": split, "model": model, "audio_cond": cond,
            "n_utts": len(utts), "wer": corpus_wer(refs, hyps), "utts": utts}


def decode_batches(model, whisper, examples, cond, noise_dir, batch_size, device, max_seconds=30, language="en"):
    fp16 = device == "cuda"
    options = whisper.DecodingOptions(language=language, task="transcribe", without_timestamps=True, fp16=fp16)
    n_mels = getattr(model.dims, "n_mels", 80)
    results = {}
    for i in tqdm(range(0, len(examples), batch_size), desc="transcribe"):
        chunk = examples[i:i + batch_size]
        mels = []
        for ex in chunk:
            wav = load_wav(ex["audio_path"], 16000, max_seconds)
            wav = apply_condition(wav, 16000, cond, ex["utt_id"], noise_dir)
            audio = whisper.pad_or_trim(wav)
            mels.append(whisper.log_mel_spectrogram(audio, n_mels=n_mels))
        import torch
        mel = torch.stack(mels).to(device)
        if fp16:
            mel = mel.half()
        outs = whisper.decode(model, mel, options)
        for ex, o in zip(chunk, outs):
            results[ex["utt_id"]] = {
                "text": normalize_text(o.text),
                "avg_logprob": float(o.avg_logprob),
                "no_speech_prob": float(o.no_speech_prob),
                "compression_ratio": float(o.compression_ratio),
            }
    return results


def merge_shards(cfg, dataset, split, source, n_shards, examples, model_name, cond):
    results = {}
    for s in range(n_shards):
        p = asr_out_path(cfg, dataset, split, source, (s, n_shards))
        data = json.load(open(p))
        for uid, r in data["utts"].items():
            results[uid] = {k: r[k] for k in ("text", "avg_logprob", "no_speech_prob", "compression_ratio")}
    out = build_output(source, dataset, split, model_name, cond, examples, results)
    write_json(asr_out_path(cfg, dataset, split, source), out)
    print(f"merged -> {asr_out_path(cfg, dataset, split, source)} | WER={out['wer']:.4f} n={out['n_utts']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--model", default="small", help="tiny|base|small|medium|large-v3")
    ap.add_argument("--audio_cond", default="clean")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--shard", default="0/1", help="i/n for multi-GPU or multi-job runs")
    ap.add_argument("--merge", action="store_true", help="merge shard files into the final json")
    ap.add_argument("--max_utts", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["datasets"][args.dataset]
    examples = read_jsonl(Path(ds["root"]) / f"{args.split}.jsonl")
    if args.max_utts:
        examples = examples[: args.max_utts]
    source = f"whisper-{args.model}@{args.audio_cond}"
    shard_i, shard_n = [int(x) for x in args.shard.split("/")]

    if args.merge:
        merge_shards(cfg, args.dataset, args.split, source, shard_n, examples, f"whisper-{args.model}", args.audio_cond)
        return

    final_path = asr_out_path(cfg, args.dataset, args.split, source)
    if args.skip_existing and shard_n == 1 and final_path.exists():
        print(f"exists, skipping: {final_path}")
        return

    import torch
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(args.model, device=device)
    my = examples[shard_i::shard_n]
    results = decode_batches(model, whisper, my, args.audio_cond, cfg["noise"]["noise_dir"],
                             args.batch_size, device, language=ds.get("language", "en"))
    out = build_output(source, args.dataset, args.split, f"whisper-{args.model}", args.audio_cond, my, results)
    path = asr_out_path(cfg, args.dataset, args.split, source, None if shard_n == 1 else (shard_i, shard_n))
    write_json(path, out)
    print(f"{source} [{args.dataset}/{args.split}] shard {shard_i}/{shard_n}: WER={out['wer']:.4f} n={out['n_utts']} -> {path}")


if __name__ == "__main__":
    main()

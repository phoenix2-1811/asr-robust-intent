import argparse
import json
import os
import statistics
import time

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from src.common.utils import write_json
from src.models.heads import REL_DIM, IntentHeadModel


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1e3
    return float("nan")


def timed(fn, warmup, runs, device):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"median_ms": statistics.median(ts), "p95_ms": ts[min(len(ts) - 1, int(0.95 * len(ts)))],
            "mean_ms": float(np.mean(ts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_encoder", required=True)
    ap.add_argument("--text_encoder", required=True)
    ap.add_argument("--head_ckpt", default=None, help="best.pt of a trained head (optional; random head if absent)")
    ap.add_argument("--head_meta", default=None, help="meta.json of that run (needed with --head_ckpt)")
    ap.add_argument("--n_labels", type=int, default=69)
    ap.add_argument("--audio_seconds", type=float, default=3.0)
    ap.add_argument("--text", default="wake me up at nine am on friday")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    base_rss = rss_mb()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    is_whisper = "whisper" in args.audio_encoder.lower()
    fe = AutoFeatureExtractor.from_pretrained(args.audio_encoder)
    a_model = AutoModel.from_pretrained(args.audio_encoder)
    a_enc = (a_model.encoder if is_whisper and hasattr(a_model, "encoder") else a_model).to(device, dtype).eval()
    tok = AutoTokenizer.from_pretrained(args.text_encoder)
    t_enc = AutoModel.from_pretrained(args.text_encoder).to(device, dtype).eval()

    arch = None
    if args.head_meta:
        arch = json.load(open(args.head_meta))["arch"]
        head = IntentHeadModel(arch["fusion"], arch["n_labels"], arch["text_dim"], arch["audio_dim"],
                               arch["audio_layers_n"], arch["audio_pool"], arch["branch_hidden"],
                               arch["fusion_hidden"], arch["gate_hidden"], arch["dropout"], arch.get("loss_mode", "fixed"),
                               tuple(arch.get("loss_weights", (0.25, 0.25, 0.5))), arch.get("audio_mode", "pooled"),
                               arch.get("frame_arch", "attn"), arch.get("frame_dim", 256), arch.get("frame_layers", 1),
                               arch.get("route", "soft"))
        if args.head_ckpt:
            head.load_state_dict(torch.load(args.head_ckpt, map_location="cpu"))
    else:
        n_layers = a_enc.config.num_hidden_layers + 1 if hasattr(a_enc.config, "num_hidden_layers") else 13
        if is_whisper:
            n_layers = a_enc.config.encoder_layers + 1
        d = a_enc.config.hidden_size * 2
        head = IntentHeadModel("reliability", args.n_labels, t_enc.config.hidden_size, d, n_layers, "weighted")
    head = head.to(device).eval()

    wav = (np.random.default_rng(0).standard_normal(int(16000 * args.audio_seconds)) * 0.05).astype(np.float32)
    if is_whisper:
        feats = fe(wav, sampling_rate=16000, return_tensors="pt")["input_features"].to(device, dtype)
    else:
        feats = fe(wav, sampling_rate=16000, return_tensors="pt")["input_values"].to(device, dtype)
    enc_t = tok([args.text], return_tensors="pt").to(device)
    n_layers_hs = head.audio_pool.layer_logits.numel() if head.audio_pool.layer_logits is not None else 1

    @torch.no_grad()
    def audio_fwd():
        out = a_enc(feats, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 1).float()
        if not is_whisper:
            return hs.mean(2), hs.std(2)
        n = min(int(np.ceil(args.audio_seconds * 50)), hs.shape[2])
        return hs[:, :, :n].mean(2), hs[:, :, :n].std(2)

    @torch.no_grad()
    def text_fwd():
        return t_enc(**enc_t).last_hidden_state[:, 0].float()

    @torch.no_grad()
    def head_fwd(a, t):
        m, s = a
        x = torch.cat([m, s], -1) if head.audio_pool.ln.normalized_shape[0] == 2 * m.shape[-1] else m
        rel = torch.zeros(1, REL_DIM, device=device)
        return head(t, x, rel)

    a_stat, t_cls = audio_fwd(), text_fwd()
    res = {
        "device": str(device), "dtype": str(dtype), "audio_encoder": args.audio_encoder,
        "text_encoder": args.text_encoder, "audio_seconds": args.audio_seconds,
        "params_audio_encoder": sum(p.numel() for p in a_enc.parameters()),
        "params_text_encoder": sum(p.numel() for p in t_enc.parameters()),
        "params_head": head.n_total(),
        "latency_audio_encoder": timed(audio_fwd, args.warmup, args.runs, device),
        "latency_text_encoder": timed(text_fwd, args.warmup, args.runs, device),
        "latency_head": timed(lambda: head_fwd(a_stat, t_cls), args.warmup, args.runs, device),
        "rss_mb_after_load": rss_mb(), "rss_mb_before_load": base_rss,
    }
    res["latency_end_to_end_median_ms"] = (res["latency_audio_encoder"]["median_ms"] +
                                           res["latency_text_encoder"]["median_ms"] + res["latency_head"]["median_ms"])
    if device.type == "cuda":
        res["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
    print(json.dumps(res, indent=2))
    if args.out:
        write_json(args.out, res)


if __name__ == "__main__":
    main()

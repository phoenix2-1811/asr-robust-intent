import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from src.common.data import (audio_feat_path, frames_feat_path, load_examples, source_available, text_feat_path,
                             texts_for_source)
from src.common.utils import load_config
from src.data.audio_conditions import apply_condition, load_wav


def conv_out_lengths(config, n_samples):
    kernels = getattr(config, "conv_kernel", None)
    strides = getattr(config, "conv_stride", None)
    L = n_samples
    for k, s in zip(kernels, strides):
        L = (L - k) // s + 1
    return max(L, 1)


def frame_count(encoder, n_samples):
    if hasattr(encoder, "_get_feat_extract_output_lengths"):
        v = encoder._get_feat_extract_output_lengths(torch.tensor(n_samples))
        return int(v)
    return conv_out_lengths(encoder.config, n_samples)


def masked_stats(hidden, lengths, want_std):
    B, L, T, D = hidden.shape
    ar = torch.arange(T, device=hidden.device)[None, :]
    mask = (ar < lengths[:, None]).float()[:, None, :, None]
    n = lengths.float().clamp(min=1)[:, None, None]
    hs = hidden.float()
    mean = (hs * mask).sum(2) / n
    std = None
    if want_std:
        var = (((hs - mean[:, :, None, :]) ** 2) * mask).sum(2) / n
        std = torch.sqrt(var + 1e-8)
    return mean, std


def extract_audio(cfg, dataset, split, enc_name, cond, device, max_utts=None, overwrite=False, save_frames=False):
    out_path = audio_feat_path(cfg, dataset, split, enc_name, cond)
    fr_path = frames_feat_path(cfg, dataset, split, enc_name, cond)
    if out_path.exists() and not overwrite and (not save_frames or fr_path.exists()):
        print(f"exists, skipping: {out_path}")
        return
    examples = load_examples(cfg, dataset, split)
    if max_utts:
        examples = examples[:max_utts]
    max_sec = cfg["datasets"][dataset]["max_audio_seconds"]
    af = cfg["audio_features"]
    want_std = "std" in af["stats"]
    is_whisper = "whisper" in enc_name.lower()
    fe = AutoFeatureExtractor.from_pretrained(enc_name)
    model = AutoModel.from_pretrained(enc_name)
    encoder = model.encoder if is_whisper and hasattr(model, "encoder") else model
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    use_amp = bool(af.get("fp16_encoder", True)) and device.type == "cuda"
    bs = af["whisper_batch_size"] if is_whisper else af["batch_size"]
    sr = 16000
    noise_dir = cfg["noise"]["noise_dir"]

    dt = torch.float16 if af["store_dtype"] == "float16" else torch.float32
    order = list(range(len(examples)))
    if not is_whisper:
        try:
            import soundfile as sf
            infos = [sf.info(ex["audio_path"]) for ex in examples]
            order = list(np.argsort([f.frames / f.samplerate for f in infos]))
        except Exception:
            pass

    means, stds, nfr, ids = {}, {}, {}, {}
    frames_store = {}
    f_layer = int(af.get("frame_layer", 8))
    f_stride = int(af.get("frame_stride", 2))
    f_max = int(af.get("max_frames", 400))
    use_mask = getattr(getattr(encoder, "config", None), "feat_extract_norm", "group") == "layer" and not is_whisper
    with torch.no_grad():
        for i in tqdm(range(0, len(order), bs), desc=f"audio:{enc_name}:{dataset}:{split}:{cond}"):
            idxs = order[i:i + bs]
            wavs = []
            for j in idxs:
                w = load_wav(examples[j]["audio_path"], sr, max_sec)
                wavs.append(apply_condition(w, sr, cond, examples[j]["utt_id"], noise_dir))
            lens_samples = [len(w) for w in wavs]
            if is_whisper:
                inp = fe(wavs, sampling_rate=sr, return_tensors="pt")
                feats = inp["input_features"].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    out = encoder(feats, output_hidden_states=True)
                frames = torch.tensor([min(max(int(math.ceil(n / sr * 50)), 1), 1500) for n in lens_samples],
                                      device=device)
            else:
                norm = [fe(w, sampling_rate=sr, return_tensors="np")["input_values"][0] for w in wavs]
                T = max(len(x) for x in norm)
                batch = np.zeros((len(norm), T), dtype=np.float32)
                for k, x in enumerate(norm):
                    batch[k, :len(x)] = x
                x = torch.from_numpy(batch).to(device)
                kw = {}
                if use_mask:
                    am = torch.zeros(len(norm), T, dtype=torch.long)
                    for k, n in enumerate(lens_samples):
                        am[k, :min(n, T)] = 1
                    kw["attention_mask"] = am.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    out = encoder(x, output_hidden_states=True, **kw)
                frames = torch.tensor([frame_count(encoder, n) for n in lens_samples], device=device)
            hs = torch.stack(out.hidden_states, dim=1)
            T_out = hs.shape[2]
            frames = frames.clamp(max=T_out)
            m, s = masked_stats(hs, frames, want_std)
            for k, j in enumerate(idxs):
                uid = examples[j]["utt_id"]
                means[uid] = m[k].cpu().to(dt)
                if want_std:
                    stds[uid] = s[k].cpu().to(dt)
                nfr[uid] = int(frames[k])
                if save_frames:
                    L_idx = min(f_layer, hs.shape[1] - 1)
                    fx = hs[k, L_idx, :int(frames[k])].float()
                    if f_stride > 1:
                        fx = torch.nn.functional.avg_pool1d(fx.t().unsqueeze(0), f_stride, f_stride, ceil_mode=True)[0].t()
                    frames_store[uid] = fx[:f_max].cpu().to(dt)
    ids_out = [ex["utt_id"] for ex in examples]
    payload = {
        "utt_ids": ids_out,
        "mean": torch.stack([means[u] for u in ids_out]),
        "std": torch.stack([stds[u] for u in ids_out]) if want_std else None,
        "n_frames": torch.tensor([nfr[u] for u in ids_out]),
        "encoder": enc_name, "cond": cond,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"Saved mean {tuple(payload['mean'].shape)} -> {out_path}")
    if save_frames:
        lens = torch.tensor([frames_store[u].shape[0] for u in ids_out])
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lens, 0)])
        torch.save({"utt_ids": ids_out, "frames": torch.cat([frames_store[u] for u in ids_out]), "offsets": offsets,
                    "layer": f_layer, "stride": f_stride, "encoder": enc_name, "cond": cond}, fr_path)
        print(f"Saved frames {int(offsets[-1])} x {frames_store[ids_out[0]].shape[1]} -> {fr_path}")


def extract_text(cfg, dataset, split, enc_name, source, device, max_utts=None, overwrite=False):
    out_path = text_feat_path(cfg, dataset, split, enc_name, source)
    if out_path.exists() and not overwrite:
        print(f"exists, skipping: {out_path}")
        return
    if not source_available(cfg, dataset, split, source):
        print(f"!! transcript source {source} not available for {dataset}/{split}, skipping")
        return
    examples = load_examples(cfg, dataset, split)
    if max_utts:
        examples = examples[:max_utts]
    texts = texts_for_source(cfg, dataset, split, source, examples)
    tok = AutoTokenizer.from_pretrained(enc_name)
    mlp = Path("data/derived_max_len.json")
    max_len = json.load(open(mlp))["max_text_len"] if mlp.exists() else cfg.get("max_text_len_ceiling", 128)
    encoder = AutoModel.from_pretrained(enc_name).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    cls_l, mean_l = [], []
    bs = 128
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), bs), desc=f"text:{enc_name}:{dataset}:{split}:{source}"):
            batch = texts[i:i + bs]
            enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            h = encoder(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            cls_l.append(h[:, 0].float().cpu())
            mean_l.append(((h.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    payload = {"utt_ids": [ex["utt_id"] for ex in examples], "cls": torch.cat(cls_l), "mean": torch.cat(mean_l),
               "encoder": enc_name, "source": source}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"Saved cls {tuple(payload['cls'].shape)} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder_type", required=True, choices=["audio", "text"])
    ap.add_argument("--encoder_name", required=True)
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--audio_cond", default="clean")
    ap.add_argument("--text_source", default="gold", help="gold | whisper-small@clean | synth-wer10 | ...")
    ap.add_argument("--max_utts", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save_frames", action="store_true",
                    help="also cache frame-level features of audio_features.frame_layer (ragged, strided)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.encoder_type == "audio":
        extract_audio(cfg, args.dataset, args.split, args.encoder_name, args.audio_cond, device,
                      args.max_utts, args.overwrite, args.save_frames)
    else:
        extract_text(cfg, args.dataset, args.split, args.encoder_name, args.text_source, device,
                     args.max_utts, args.overwrite)


if __name__ == "__main__":
    main()

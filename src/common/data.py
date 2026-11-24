from pathlib import Path

import numpy as np
import torch

from src.asr.transcribe import asr_out_path
from src.common.text_norm import normalize_text, word_edit_distance
from src.common.utils import read_json, read_jsonl, short, split_source

REL_DIM = 7


def load_examples(cfg, dataset, split):
    return read_jsonl(Path(cfg["datasets"][dataset]["root"]) / f"{split}.jsonl")


def load_label_map(cfg, dataset):
    return read_json(Path(cfg["datasets"][dataset]["root"]) / "label_map.json")


def feat_dir(cfg, dataset, split):
    return Path(cfg["paths"]["features"]) / dataset / split


def audio_feat_path(cfg, dataset, split, enc, cond):
    return feat_dir(cfg, dataset, split) / f"audio__{short(enc)}__{cond}.pt"


def frames_feat_path(cfg, dataset, split, enc, cond):
    return feat_dir(cfg, dataset, split) / f"frames__{short(enc)}__{cond}.pt"


def text_feat_path(cfg, dataset, split, enc, source):
    return feat_dir(cfg, dataset, split) / f"text__{short(enc)}__{source.replace('@', '-')}.pt"


def load_asr(cfg, dataset, split, source):
    return read_json(asr_out_path(cfg, dataset, split, source))


def source_available(cfg, dataset, split, source):
    if source == "gold":
        return True
    return asr_out_path(cfg, dataset, split, source).exists()


def texts_for_source(cfg, dataset, split, source, examples):
    if source == "gold":
        return [normalize_text(ex["transcript"]) for ex in examples]
    data = load_asr(cfg, dataset, split, source)
    return [data["utts"][ex["utt_id"]]["text"] for ex in examples]


def _hyp_disagreement(a, b):
    e, n = word_edit_distance(a, b)
    return e / max(n, 1)


def reliability_features(cfg, dataset, split, source, examples):
    N = len(examples)
    feats = np.zeros((N, REL_DIM), dtype=np.float32)
    if source == "gold" or source.startswith("synth"):
        return torch.from_numpy(feats)
    data = load_asr(cfg, dataset, split, source)
    model, cond = split_source(source)
    ref_model = cfg["reliability"]["reference_model"]
    if model == ref_model:
        ref_model = cfg["reliability"]["fallback_reference_model"]
    ref_src = f"{ref_model}@{cond}"
    ref = load_asr(cfg, dataset, split, ref_src) if asr_out_path(cfg, dataset, split, ref_src).exists() else None
    for i, ex in enumerate(examples):
        u = data["utts"][ex["utt_id"]]
        if "avg_logprob" not in u:
            continue
        xd = 0.0
        if ref is not None:
            xd = _hyp_disagreement(u["text"], ref["utts"][ex["utt_id"]]["text"])
        feats[i] = [u["avg_logprob"], u["no_speech_prob"], u["compression_ratio"],
                    np.log1p(len(u["text"].split())), xd, u.get("pred_wer", 0.0), 1.0]
    return torch.from_numpy(feats)


def _align(saved_ids, examples, what):
    index = {u: i for i, u in enumerate(saved_ids)}
    missing = [ex["utt_id"] for ex in examples if ex["utt_id"] not in index]
    if missing:
        raise RuntimeError(f"{what}: {len(missing)} utterances missing from cached features (e.g. {missing[:3]}). "
                           f"Re-run feature extraction for this split.")
    return torch.tensor([index[ex["utt_id"]] for ex in examples], dtype=torch.long)


def load_audio_feats(cfg, dataset, split, enc, cond, examples, stats, layers):
    d = torch.load(audio_feat_path(cfg, dataset, split, enc, cond), map_location="cpu")
    order = _align(d["utt_ids"], examples, f"audio {short(enc)}@{cond}")
    parts = []
    for s in stats:
        if d.get(s) is None:
            raise RuntimeError(f"stat '{s}' not stored in {audio_feat_path(cfg, dataset, split, enc, cond)}")
        parts.append(d[s][order])
    x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    if layers == "last":
        x = x[:, -1:, :]
    return x


def load_frames(cfg, dataset, split, enc, cond, examples):
    d = torch.load(frames_feat_path(cfg, dataset, split, enc, cond), map_location="cpu")
    order = _align(d["utt_ids"], examples, f"frames {short(enc)}@{cond}")
    offs = d["offsets"]
    starts = offs[order]
    lens = offs[order + 1] - starts
    return d["frames"], starts, lens


def load_text_feats(cfg, dataset, split, enc, source, examples, kind):
    d = torch.load(text_feat_path(cfg, dataset, split, enc, source), map_location="cpu")
    order = _align(d["utt_ids"], examples, f"text {short(enc)}@{source}")
    if kind == "cls+mean":
        return torch.cat([d["cls"][order], d["mean"][order]], dim=-1).float()
    return d[kind][order].float()


class SplitData:
    def __init__(self, cfg, dataset, split, pairs, audio_encoder, text_encoder, need_text, need_audio,
                 audio_stats, audio_layers, text_feature, device, audio_mode="pooled"):
        self.audio_mode = audio_mode
        self.fr = {}
        self.examples = load_examples(cfg, dataset, split)
        label_map = load_label_map(cfg, dataset)
        self.labels = torch.tensor([label_map[ex["label"]] for ex in self.examples], dtype=torch.long, device=device)
        self.utt_ids = [ex["utt_id"] for ex in self.examples]
        self.groups = [ex.get("group", ex["utt_id"]) for ex in self.examples]
        self.pairs = pairs
        self.device = device
        self.N = len(self.examples)
        self.audio, self.text, self.rel = {}, {}, {}
        for p in pairs:
            if need_audio and audio_mode == "pooled" and p["audio"] not in self.audio:
                self.audio[p["audio"]] = load_audio_feats(cfg, dataset, split, audio_encoder, p["audio"],
                                                          self.examples, audio_stats, audio_layers).to(device)
            if need_audio and audio_mode == "frames" and p["audio"] not in self.fr:
                self.fr[p["audio"]] = load_frames(cfg, dataset, split, audio_encoder, p["audio"], self.examples)
            if need_text and p["text"] not in self.text:
                self.text[p["text"]] = load_text_feats(cfg, dataset, split, text_encoder, p["text"],
                                                       self.examples, text_feature).to(device)
            if p["text"] not in self.rel:
                self.rel[p["text"]] = reliability_features(cfg, dataset, split, p["text"], self.examples).to(device)
        self.need_text, self.need_audio = need_text, need_audio
        self.text_dim = next(iter(self.text.values())).shape[1] if self.text else None
        if audio_mode == "pooled" and self.audio:
            a0 = next(iter(self.audio.values()))
            self.audio_n_layers, self.audio_dim = a0.shape[1], a0.shape[2]
        elif audio_mode == "frames" and self.fr:
            self.audio_n_layers, self.audio_dim = 1, next(iter(self.fr.values()))[0].shape[1]
        else:
            self.audio_n_layers, self.audio_dim = 1, None

    def gather(self, idx, pair_choice):
        B = idx.numel()
        text = audio = amask = None
        rel = torch.zeros(B, REL_DIM, device=self.device)
        if self.need_text:
            text = torch.zeros(B, self.text_dim, device=self.device)
        if self.need_audio and self.audio_mode == "pooled":
            a0 = next(iter(self.audio.values()))
            audio = torch.zeros(B, a0.shape[1], a0.shape[2], device=self.device)
        frame_rows = [None] * B
        idx_cpu = idx.cpu()
        for k, p in enumerate(self.pairs):
            m = pair_choice == k
            if not bool(m.any()):
                continue
            ii = idx[m]
            if self.need_text:
                text[m] = self.text[p["text"]][ii]
            if self.need_audio and self.audio_mode == "pooled":
                audio[m] = self.audio[p["audio"]][ii].float()
            rel[m] = self.rel[p["text"]][ii]
            if self.need_audio and self.audio_mode == "frames":
                frames, starts, lens = self.fr[p["audio"]]
                rows = torch.nonzero(m.cpu()).flatten().tolist()
                for r in rows:
                    j = int(idx_cpu[r])
                    st, ln = int(starts[j]), int(lens[j])
                    frame_rows[r] = frames[st:st + ln]
        if self.need_audio and self.audio_mode == "frames":
            Tm = max(int(x.shape[0]) for x in frame_rows)
            D = frame_rows[0].shape[1]
            audio = torch.zeros(B, Tm, D)
            amask = torch.zeros(B, Tm, dtype=torch.bool)
            for r, x in enumerate(frame_rows):
                audio[r, :x.shape[0]] = x.float()
                amask[r, :x.shape[0]] = True
            audio, amask = audio.to(self.device), amask.to(self.device)
        return {"text": text, "audio": audio, "amask": amask, "rel": rel, "label": self.labels[idx]}

    def iterate(self, batch_size, shuffle, generator=None):
        if shuffle:
            order = torch.randperm(self.N, generator=generator).to(self.device)
        else:
            order = torch.arange(self.N, device=self.device)
        for i in range(0, self.N, batch_size):
            idx = order[i:i + batch_size]
            if shuffle and len(self.pairs) > 1:
                choice = torch.randint(0, len(self.pairs), (idx.numel(),), device=self.device, generator=None)
            else:
                choice = torch.zeros(idx.numel(), dtype=torch.long, device=self.device)
            yield idx, self.gather(idx, choice)

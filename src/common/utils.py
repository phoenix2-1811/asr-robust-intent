import json
import os
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path="configs/experiment.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def short(name):
    return name.rstrip("/").split("/")[-1]


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def split_source(source):
    if source == "gold":
        return "gold", None
    if "@" in source:
        m, c = source.split("@", 1)
        return m, c
    return source, None


def run_name(fusion, audio_encoder, text_encoder, train_pairs, stage, loss_mode, audio_layers, audio_stats, tag=None,
             audio_mode="pooled", frame_arch="attn", route="soft"):
    parts = [fusion]
    if fusion != "text_only" and audio_encoder:
        parts.append(short(audio_encoder))
    if fusion != "audio_only" and text_encoder:
        parts.append(short(text_encoder))
    srcs = sorted({p["text"] for p in train_pairs}) if fusion != "audio_only" else []
    auds = sorted({p["audio"] for p in train_pairs}) if fusion != "text_only" else []
    if srcs:
        parts.append("tx-" + "+".join(s.replace("@", "-") for s in srcs))
    if auds and auds != ["clean"]:
        parts.append("au-" + "+".join(auds))
    if fusion not in ("text_only", "audio_only"):
        parts.append(stage)
        if loss_mode != "fixed":
            parts.append(loss_mode)
    if fusion != "text_only":
        if audio_mode == "frames":
            parts.append(f"F{frame_arch}")
        else:
            parts.append(f"L{audio_layers}")
            parts.append("S" + "".join(s[0] for s in audio_stats))
    if fusion == "reliability" and route == "hard":
        parts.append("hard")
    if tag:
        parts.append(tag)
    return "__".join(parts)


def build_conditions(cfg, only_available_fn=None):
    c = cfg["conditions"]
    conds = [{"name": "clean_gold", "audio": "clean", "text": "gold"}]
    for ac in c["audio"]:
        for m in c["eval_asr_models"]:
            conds.append({"name": f"{ac}_{m}", "audio": ac, "text": f"{m}@{ac}"})
    for w in c["synthetic_wer"]:
        conds.append({"name": f"clean_synth-wer{int(round(w * 100))}", "audio": "clean",
                      "text": f"synth-wer{int(round(w * 100))}"})
    if only_available_fn is not None:
        conds = [x for x in conds if only_available_fn(x)]
    return conds

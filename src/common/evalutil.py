from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.common.data import (SplitData, audio_feat_path, frames_feat_path, source_available, text_feat_path)
from src.common.utils import build_conditions, write_json


def ece_from(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def f1_macro(labels, preds):
    from sklearn.metrics import f1_score
    return float(f1_score(labels, preds, average="macro", labels=np.unique(labels), zero_division=0))


@torch.no_grad()
def predict(model, data, batch_size=512):
    model.eval()
    cols = {k: [] for k in ("text_pred", "audio_pred", "fused_pred", "text_conf", "audio_conf", "fused_conf",
                            "text_ent", "audio_ent", "gate")}
    logits_f = []
    for idx, b in data.iterate(batch_size, shuffle=False):
        out = model(b["text"], b["audio"], b["rel"], None, None, b["amask"])
        pf = F.softmax(out["fused_logits"], -1)
        cols["fused_pred"].append(pf.argmax(-1).cpu())
        cols["fused_conf"].append(pf.max(-1).values.cpu())
        logits_f.append(out["fused_logits"].cpu())
        for name, key in (("text", "text_logits"), ("audio", "audio_logits")):
            if key in out:
                p = F.softmax(out[key], -1)
                cols[f"{name}_pred"].append(p.argmax(-1).cpu())
                cols[f"{name}_conf"].append(p.max(-1).values.cpu())
                cols[f"{name}_ent"].append((-(p * torch.log(p + 1e-12)).sum(-1)).cpu())
            else:
                n = idx.numel()
                cols[f"{name}_pred"].append(torch.full((n,), -1, dtype=torch.long))
                cols[f"{name}_conf"].append(torch.zeros(n))
                cols[f"{name}_ent"].append(torch.zeros(n))
        if "gate" in out:
            g = out["gate"]
            cols["gate"].append(g.cpu() if g.dim() == 2 else g.cpu().reshape(-1, 1))
    res = {k: torch.cat(v).numpy() for k, v in cols.items() if v}
    res["labels"] = data.labels.cpu().numpy()
    res["utt_ids"] = np.array(data.utt_ids)
    res["groups"] = np.array([str(g) for g in data.groups])
    return res


def acc_of(res, key="fused_pred"):
    return float((res[key] == res["labels"]).mean())


def condition_available(cfg, dataset, split, cond, audio_encoder, text_encoder, need_audio, need_text,
                        audio_mode="pooled"):
    if need_audio:
        path = (frames_feat_path if audio_mode == "frames" else audio_feat_path)(cfg, dataset, split, audio_encoder, cond["audio"])
        if not path.exists():
            return False
    if need_text:
        if cond["text"] != "gold" and not source_available(cfg, dataset, split, cond["text"]):
            return False
        if not text_feat_path(cfg, dataset, split, text_encoder, cond["text"]).exists():
            return False
    return True


def evaluate_conditions(cfg, model, dataset, split, audio_encoder, text_encoder, audio_stats, audio_layers,
                        text_feature, device, out_dir, conditions=None, save_preds=True, audio_mode="pooled"):
    need_t, need_a = model.has_text, model.has_audio
    conds = conditions or build_conditions(
        cfg, lambda c: condition_available(cfg, dataset, split, c, audio_encoder, text_encoder, need_a, need_t, audio_mode))
    results = {}
    out_dir = Path(out_dir)
    for c in conds:
        d = SplitData(cfg, dataset, split, [{"audio": c["audio"], "text": c["text"]}], audio_encoder, text_encoder,
                      need_t, need_a, audio_stats, audio_layers, text_feature, device, audio_mode)
        res = predict(model, d)
        results[c["name"]] = {"fused_acc": acc_of(res), "n": int(len(res["labels"])),
                              "fused_f1": f1_macro(res["labels"], res["fused_pred"]),
                              "fused_ece": ece_from(res["fused_conf"], (res["fused_pred"] == res["labels"]).astype(float)),
                              "text_acc": acc_of(res, "text_pred") if need_t else None,
                              "audio_acc": acc_of(res, "audio_pred") if need_a else None}
        if save_preds:
            np.savez_compressed(out_dir / f"preds_{split}_{c['name']}.npz", **res)
    return results

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.common.data import load_examples, load_label_map, source_available, texts_for_source
from src.common.repro import env_info
from src.common.utils import build_conditions, load_config, read_json, set_seed, short, write_json
from src.data.audio_conditions import apply_condition, load_wav

LORA_TARGETS = {
    "roberta": ["query", "value"], "bert": ["query", "value"], "distilbert": ["q_lin", "v_lin"],
    "wav2vec2": ["q_proj", "v_proj"], "hubert": ["q_proj", "v_proj"], "wavlm": ["q_proj", "v_proj"],
}


class FTDataset(Dataset):
    def __init__(self, cfg, dataset, split, pairs, label_map, train):
        self.cfg = cfg
        self.examples = load_examples(cfg, dataset, split)
        self.pairs = pairs
        self.train = train
        self.label_map = label_map
        self.max_sec = cfg["datasets"][dataset]["max_audio_seconds"]
        self.noise_dir = cfg["noise"]["noise_dir"]
        self.texts = {}
        for p in pairs:
            if p["text"] not in self.texts:
                self.texts[p["text"]] = texts_for_source(cfg, dataset, split, p["text"], self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        p = random.choice(self.pairs) if self.train else self.pairs[0]
        wav = load_wav(ex["audio_path"], 16000, self.max_sec)
        wav = apply_condition(wav, 16000, p["audio"], ex["utt_id"], self.noise_dir)
        return {"wav": wav, "text": self.texts[p["text"]][i], "label": self.label_map[ex["label"]],
                "utt_id": ex["utt_id"], "group": str(ex.get("group", ex["utt_id"]))}


def make_collate(fe, tok, max_len):
    def collate(items):
        norm = [fe(it["wav"], sampling_rate=16000, return_tensors="np")["input_values"][0] for it in items]
        T = max(len(x) for x in norm)
        wav = np.zeros((len(norm), T), dtype=np.float32)
        for k, x in enumerate(norm):
            wav[k, :len(x)] = x
        enc = tok([it["text"] for it in items], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {"wav": torch.from_numpy(wav), "lens": torch.tensor([len(x) for x in norm]),
                "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "label": torch.tensor([it["label"] for it in items]),
                "utt_ids": [it["utt_id"] for it in items], "groups": [it["group"] for it in items]}
    return collate


def add_lora(module, r, alpha, dropout):
    from peft import LoraConfig, inject_adapter_in_model
    mt = module.config.model_type
    targets = LORA_TARGETS.get(mt)
    if targets is None:
        raise SystemExit(f"No LoRA targets known for model_type={mt}")
    for p in module.parameters():
        p.requires_grad = False
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, bias="none")
    module = inject_adapter_in_model(cfg, module)
    for n, p in module.named_parameters():
        p.requires_grad = "lora_" in n
    return module


class FTModel(nn.Module):
    def __init__(self, audio_name, text_name, n_labels, mode, lora_cfg, fusion_hidden=512, dropout=0.1):
        super().__init__()
        self.audio = AutoModel.from_pretrained(audio_name)
        self.text = AutoModel.from_pretrained(text_name)
        if mode == "lora":
            self.audio = add_lora(self.audio, **lora_cfg)
            self.text = add_lora(self.text, **lora_cfg)
        elif mode == "frozen":
            for p in list(self.audio.parameters()) + list(self.text.parameters()):
                p.requires_grad = False
        da, dt = self.audio.config.hidden_size, self.text.config.hidden_size
        self.audio_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(da, n_labels))
        self.text_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dt, n_labels))
        self.fuse = nn.Sequential(nn.Linear(2 * n_labels, fusion_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(fusion_hidden, n_labels))

    def frames(self, lens):
        if hasattr(self.audio, "_get_feat_extract_output_lengths"):
            return self.audio._get_feat_extract_output_lengths(lens)
        raise RuntimeError("audio encoder has no _get_feat_extract_output_lengths")

    def forward(self, wav, lens, input_ids, attention_mask):
        h = self.audio(wav).last_hidden_state
        fr = self.frames(lens.to(h.device)).clamp(max=h.shape[1])
        mask = (torch.arange(h.shape[1], device=h.device)[None, :] < fr[:, None]).float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        t = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        la, lt = self.audio_head(pooled.float()), self.text_head(t.float())
        lf = self.fuse(torch.cat([F.softmax(lt, -1), F.softmax(la, -1)], -1))
        return lt, la, lf


def loss_fn(outs, y, w):
    lt, la, lf = outs
    return w[0] * F.cross_entropy(lt, y) + w[1] * F.cross_entropy(la, y) + w[2] * F.cross_entropy(lf, y)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    rec = {k: [] for k in ("text_pred", "audio_pred", "fused_pred", "text_conf", "audio_conf", "fused_conf",
                           "labels")}
    ids, groups = [], []
    for b in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            lt, la, lf = model(b["wav"].to(device), b["lens"], b["input_ids"].to(device),
                               b["attention_mask"].to(device))
        for name, lg in (("text", lt), ("audio", la), ("fused", lf)):
            p = F.softmax(lg.float(), -1)
            rec[f"{name}_pred"].append(p.argmax(-1).cpu())
            rec[f"{name}_conf"].append(p.max(-1).values.cpu())
        rec["labels"].append(b["label"])
        ids += b["utt_ids"]
        groups += b["groups"]
    out = {k: torch.cat(v).numpy() for k, v in rec.items()}
    out["utt_ids"], out["groups"] = np.array(ids), np.array(groups)
    return out


def test_conditions_eval(cfg, model, dataset, label_map, collate, bs, device, amp, ckpt_dir, num_workers):
    conds = build_conditions(cfg, lambda c: c["text"] == "gold" or source_available(cfg, dataset, "test", c["text"]))
    results = {}
    for c in conds:
        ds = FTDataset(cfg, dataset, "test", [{"audio": c["audio"], "text": c["text"]}], label_map, False)
        loader = DataLoader(ds, batch_size=bs * 2, shuffle=False, collate_fn=collate, num_workers=num_workers)
        res = evaluate(model, loader, device, amp)
        np.savez_compressed(Path(ckpt_dir) / f"preds_test_{c['name']}.npz", **res)
        results[c["name"]] = {"fused_acc": float((res["fused_pred"] == res["labels"]).mean()),
                              "text_acc": float((res["text_pred"] == res["labels"]).mean()),
                              "audio_acc": float((res["audio_pred"] == res["labels"]).mean()),
                              "n": int(len(res["labels"]))}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--audio_encoder", default=None)
    ap.add_argument("--text_encoder", default=None)
    ap.add_argument("--mode", default=None, choices=["full", "lora", "frozen"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train_pairs", default=None)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--lr_encoder", type=float, default=None)
    ap.add_argument("--lr_head", type=float, default=None, help="for LoRA runs this is the LoRA/head learning rate")
    ap.add_argument("--final", action="store_true", help="also evaluate TEST after training (else dev only)")
    ap.add_argument("--eval_only", default=None, help="run dir of a trained model: evaluate TEST from its saved weights")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ft = cfg["finetune"]
    if args.eval_only:
        rd = Path(args.eval_only)
        m = read_json(rd / "metrics.json")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        label_map = load_label_map(cfg, m["dataset"])
        tok = AutoTokenizer.from_pretrained(m["text_encoder"])
        fe = AutoFeatureExtractor.from_pretrained(m["audio_encoder"])
        collate = make_collate(fe, tok, cfg["max_text_len_ceiling"])
        model = FTModel(m["audio_encoder"], m["text_encoder"], len(label_map), m["mode"], m["lora_cfg"]).to(device)
        model.load_state_dict(torch.load(rd / "trainable.pt", map_location="cpu"), strict=False)
        m["results"] = test_conditions_eval(cfg, model, m["dataset"], label_map, collate, ft["batch_size"], device,
                                            device.type == "cuda", rd, args.num_workers)
        m["test_evaluations"] = int(m.get("test_evaluations", 0)) + 1
        write_json(rd / "metrics.json", m)
        print(f"[test] {rd}: clean_gold={m['results'].get('clean_gold', {}).get('fused_acc')}")
        return
    if not args.dataset:
        raise SystemExit("--dataset is required unless --eval_only is used")
    if args.lr_encoder:
        ft["lr_encoder"] = args.lr_encoder
    if args.lr_head:
        ft["lr_head"] = args.lr_head
    audio_name = args.audio_encoder or ft["audio_encoder"]
    text_name = args.text_encoder or ft["text_encoder"]
    mode = args.mode or ft["mode"]
    epochs = args.epochs or ft["epochs"]
    if "whisper" in audio_name.lower():
        raise SystemExit("finetune baseline supports wav2vec2/hubert/wavlm audio encoders only")
    pairs = []
    if args.train_pairs:
        for item in args.train_pairs.split(","):
            a, t = item.split(":", 1)
            pairs.append({"audio": a, "text": t})
    else:
        pairs = ft["train_pairs"]
    sel = cfg["train"]["select_on"]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    label_map = load_label_map(cfg, args.dataset)
    tok = AutoTokenizer.from_pretrained(text_name)
    fe = AutoFeatureExtractor.from_pretrained(audio_name)
    collate = make_collate(fe, tok, cfg["max_text_len_ceiling"])
    train_ds = FTDataset(cfg, args.dataset, "train", pairs, label_map, True)
    if args.max_train:
        train_ds.examples = train_ds.examples[: args.max_train]
        train_ds.texts = {k: v[: args.max_train] for k, v in train_ds.texts.items()}
    dev_ds = FTDataset(cfg, args.dataset, "dev", [sel], label_map, False)
    bs = ft["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate, num_workers=args.num_workers,
                              drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=bs * 2, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    lora_cfg = {"r": ft["lora_r"], "alpha": ft["lora_alpha"], "dropout": ft["lora_dropout"]}
    model = FTModel(audio_name, text_name, len(label_map), mode, lora_cfg).to(device)
    enc_params = [p for n, p in list(model.audio.named_parameters()) + list(model.text.named_parameters())
                  if p.requires_grad]
    head_params = [p for m in (model.audio_head, model.text_head, model.fuse) for p in m.parameters()]
    groups = [{"params": head_params, "lr": ft["lr_head"]}]
    if enc_params:
        groups.append({"params": enc_params, "lr": ft["lr_encoder"] if mode == "full" else ft["lr_head"]})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    total_steps = max(1, epochs * int(np.ceil(len(train_loader) / ft["grad_accum"])))
    sched = get_linear_schedule_with_warmup(opt, int(ft["warmup_ratio"] * total_steps), total_steps)
    w = cfg["train"]["loss_weights"]
    weights = (w["text"], w["audio"], w["fused"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"mode={mode} trainable={n_train:,} total={n_total:,}")

    name = f"finetune-{mode}__{short(audio_name)}__{short(text_name)}__tx-" + "+".join(
        sorted({p['text'].replace('@', '-') for p in pairs})) + (f"__lr{ft['lr_encoder']:g}" if mode == "full" else f"__lr{ft['lr_head']:g}" if mode == "lora" else "")
    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / args.dataset / name / f"seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        for step, b in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                outs = model(b["wav"].to(device), b["lens"], b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = loss_fn(outs, b["label"].to(device), weights) / ft["grad_accum"]
            loss.backward()
            if (step + 1) % ft["grad_accum"] == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        dres = evaluate(model, dev_loader, device, amp)
        acc = float((dres["fused_pred"] == dres["labels"]).mean())
        print(f"epoch {epoch}: dev fused acc={acc:.4f}")
        if acc > best:
            best = acc
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable}
    train_seconds = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    torch.save(best_state, ckpt_dir / "trainable.pt")

    results = {}
    if args.final and not args.no_eval:
        results = test_conditions_eval(cfg, model, args.dataset, label_map, collate, bs, device, amp, ckpt_dir,
                                       args.num_workers)
    write_json(ckpt_dir / "metrics.json", {
        "run": name, "dataset": args.dataset, "seed": args.seed, "mode": mode, "trainable_params": n_train,
        "total_params": n_total, "train_seconds": train_seconds, "peak_gpu_mem_gb": peak, "best_dev_acc": best,
        "gpu_hours": train_seconds / 3600.0 if device.type == "cuda" else None,
        "train_pairs": pairs, "results": results, "test_evaluations": 1 if results else 0,
        "dev_results": {"select_on": {"fused_acc": best, "n": len(dev_ds)}},
        "audio_encoder": audio_name, "text_encoder": text_name, "lora_cfg": lora_cfg, "env": env_info(cfg)})
    print(f"[{name} seed{args.seed}] dev={best:.4f} test_evaluated={bool(results)}")


if __name__ == "__main__":
    main()

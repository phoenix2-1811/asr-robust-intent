import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.common.data import SplitData
from src.common.evalutil import acc_of, evaluate_conditions, predict
from src.common.repro import env_info
from src.common.utils import load_config, run_name, set_seed, write_json
from src.models.heads import FUSION_TYPES, IntentHeadModel, fit_temperature


def parse_pairs(s):
    pairs = []
    for item in s.split(","):
        a, t = item.split(":", 1)
        pairs.append({"audio": a, "text": t})
    return pairs


def rel_stats(data):
    rows = []
    for rel in data.rel.values():
        m = rel[:, -1] > 0
        if m.any():
            rows.append(rel[m, :-1])
    if not rows:
        return torch.zeros(rel.shape[1] - 1), torch.ones(rel.shape[1] - 1)
    x = torch.cat(rows)
    return x.mean(0).cpu(), x.std(0).cpu()


@torch.no_grad()
def dev_scores(model, dev):
    res = predict(model, dev)
    s = {"fused": acc_of(res)}
    if model.has_text:
        s["text"] = acc_of(res, "text_pred")
    if model.has_audio:
        s["audio"] = acc_of(res, "audio_pred")
    return s


@torch.no_grad()
def branch_logits(model, dev):
    model.eval()
    t, a, ys = [], [], []
    for idx, b in dev.iterate(1024, shuffle=False):
        zt = model.text_in(b["text"]) if model.has_text else None
        if model.has_audio:
            za = model.audio_pool(b["audio"], b["amask"])[0] if model.audio_mode == "frames" else model.audio_pool(b["audio"])
        else:
            za = None
        if zt is not None:
            t.append(model.text_head(zt))
        if za is not None:
            a.append(model.audio_head(za))
        ys.append(b["label"])
    return (torch.cat(t) if t else None), (torch.cat(a) if a else None), torch.cat(ys)


def run_epochs(model, params, train, dev, args, stage, metric_fn, seed, tag):
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, bad = -1.0, None, 0
    gen = torch.Generator().manual_seed(seed)
    use_drop = args.modality_dropout > 0 and model.fusion not in ("text_only", "audio_only", "avg") \
        and stage in ("joint", "fusion")
    for epoch in range(args.epochs):
        model.train()
        gen.manual_seed(seed * 1000 + epoch)
        for idx, b in train.iterate(args.batch_size, True, gen):
            drop_t = drop_a = None
            if use_drop:
                r = torch.rand(idx.numel(), device=idx.device)
                drop_a = r < args.modality_dropout
                drop_t = (r >= args.modality_dropout) & (r < 2 * args.modality_dropout)
            out = model(b["text"], b["audio"], b["rel"], drop_t, drop_a, b["amask"])
            loss, _ = model.loss(out, b["label"], drop_t, drop_a, stage)
            opt.zero_grad()
            loss.backward()
            opt.step()
        s = dev_scores(model, dev)
        m = metric_fn(s)
        if m > best:
            best, bad = m, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                break
        if epoch % 10 == 0:
            print(f"[{tag} seed{seed}] {stage} epoch {epoch}: dev={ {k: round(v, 4) for k, v in s.items()} }")
    model.load_state_dict(best_state)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fusion", default=None, choices=FUSION_TYPES)
    ap.add_argument("--audio_encoder", default=None)
    ap.add_argument("--text_encoder", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", default=None, choices=["joint", "two_stage"])
    ap.add_argument("--loss_mode", default=None, choices=["fixed", "uncertainty", "learned_convex"])
    ap.add_argument("--audio_layers", default=None, choices=["weighted", "last"])
    ap.add_argument("--audio_stats", default=None, help="comma list from mean,std")
    ap.add_argument("--text_feature", default=None, choices=["cls", "mean", "cls+mean"])
    ap.add_argument("--audio_mode", default=None, choices=["pooled", "frames"])
    ap.add_argument("--frame_arch", default=None, choices=["attn", "bilstm", "transformer"])
    ap.add_argument("--route", default=None, choices=["soft", "hard"])
    ap.add_argument("--train_pairs", default=None, help="audio:text,audio:text e.g. clean:gold,clean:whisper-small@clean")
    ap.add_argument("--select_on", default=None, help="audio:text pair used for dev early stopping")
    ap.add_argument("--modality_dropout", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--final", action="store_true",
                    help="also evaluate the TEST split. Leave off during development; use src.eval_final after selection.")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--no_save_preds", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr, mo = cfg["train"], cfg["model"]
    args.fusion = args.fusion or mo["default_fusion"]
    args.stage = args.stage or tr["stage"]
    args.loss_mode = args.loss_mode or tr["loss_mode"]
    args.audio_layers = args.audio_layers or mo["audio_layers"]
    args.audio_stats = args.audio_stats.split(",") if args.audio_stats else mo["audio_stats"]
    args.text_feature = args.text_feature or mo["text_feature"]
    args.audio_mode = args.audio_mode or mo.get("audio_mode", "pooled")
    args.frame_arch = args.frame_arch or mo.get("frame_arch", "attn")
    args.route = args.route or mo.get("route", "soft")
    pairs = parse_pairs(args.train_pairs) if args.train_pairs else tr["train_pairs"]
    sel = parse_pairs(args.select_on)[0] if args.select_on else tr["select_on"]
    args.modality_dropout = tr["modality_dropout"] if args.modality_dropout is None else args.modality_dropout
    args.epochs = args.epochs or tr["epochs"]
    args.patience = args.patience or tr["patience"]
    args.lr = args.lr or tr["lr"]
    args.weight_decay = tr["weight_decay"]
    args.batch_size = tr["batch_size"]
    args.dropout = mo["dropout"] if args.dropout is None else args.dropout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    need_t = args.fusion != "audio_only"
    need_a = args.fusion != "text_only"
    if need_a and not args.audio_encoder:
        raise SystemExit("--audio_encoder required for this fusion type")
    if need_t and not args.text_encoder:
        raise SystemExit("--text_encoder required for this fusion type")

    name = run_name(args.fusion, args.audio_encoder if need_a else None, args.text_encoder if need_t else None,
                    pairs, args.stage, args.loss_mode, args.audio_layers, args.audio_stats, args.tag,
                    args.audio_mode, args.frame_arch, args.route)
    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / args.dataset / name / f"seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train = SplitData(cfg, args.dataset, "train", pairs, args.audio_encoder, args.text_encoder, need_t, need_a,
                      args.audio_stats, args.audio_layers, args.text_feature, device, args.audio_mode)
    dev = SplitData(cfg, args.dataset, "dev", [sel], args.audio_encoder, args.text_encoder, need_t, need_a,
                    args.audio_stats, args.audio_layers, args.text_feature, device, args.audio_mode)
    from src.common.data import load_label_map
    n_labels = len(load_label_map(cfg, args.dataset))
    text_dim = train.text_dim if need_t else None
    audio_dim = train.audio_dim if need_a else None
    n_layers = train.audio_n_layers if need_a else 1

    model = IntentHeadModel(args.fusion, n_labels, text_dim, audio_dim, n_layers, args.audio_layers,
                            mo["branch_hidden"], mo["fusion_hidden"], mo["gate_hidden"], args.dropout,
                            args.loss_mode, (tr["loss_weights"]["text"], tr["loss_weights"]["audio"],
                                             tr["loss_weights"]["fused"]),
                            args.audio_mode, args.frame_arch, mo.get("frame_dim", 256), mo.get("frame_layers", 1),
                            args.route).to(device)
    mean, std = rel_stats(train)
    model.set_rel_stats(mean.to(device), std.to(device))

    t0 = time.time()
    info = {}
    two = args.stage == "two_stage" and args.fusion not in ("text_only", "audio_only")
    if args.fusion == "avg":
        two = True
    if two:
        def metric_branches(s):
            vals = [s[k] for k in ("text", "audio") if k in s]
            return float(np.mean(vals))
        run_epochs(model, model.branch_params(), train, dev, args, "branches", metric_branches, args.seed, name)
        lt, la, ys = branch_logits(model, dev)
        if lt is not None:
            model.T_text.fill_(fit_temperature(lt, ys))
        if la is not None:
            model.T_audio.fill_(fit_temperature(la, ys))
        info["T_text"] = float(model.T_text) if model.has_text else None
        info["T_audio"] = float(model.T_audio) if model.has_audio else None
        model.freeze_branches()
        if args.fusion != "avg" and model.fusion_params():
            run_epochs(model, model.fusion_params(), train, dev, args, "fusion", lambda s: s["fused"],
                       args.seed, name)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        run_epochs(model, params, train, dev, args, "joint", lambda s: s["fused"], args.seed, name)
    train_seconds = time.time() - t0

    torch.save(model.state_dict(), ckpt_dir / "best.pt")
    final_dev = dev_scores(model, dev)
    arch = {"fusion": args.fusion, "n_labels": n_labels, "text_dim": text_dim, "audio_dim": audio_dim,
            "audio_layers_n": n_layers, "audio_pool": args.audio_layers, "branch_hidden": mo["branch_hidden"],
            "fusion_hidden": mo["fusion_hidden"], "gate_hidden": mo["gate_hidden"], "dropout": args.dropout,
            "loss_mode": args.loss_mode,
            "loss_weights": [tr["loss_weights"]["text"], tr["loss_weights"]["audio"], tr["loss_weights"]["fused"]],
            "audio_encoder": args.audio_encoder, "text_encoder": args.text_encoder,
            "audio_stats": args.audio_stats, "text_feature": args.text_feature, "audio_mode": args.audio_mode,
            "frame_arch": args.frame_arch, "frame_dim": mo.get("frame_dim", 256),
            "frame_layers": mo.get("frame_layers", 1), "route": args.route}
    meta = {"run": name, "dataset": args.dataset, "seed": args.seed, "stage": args.stage, "arch": arch,
            "train_pairs": pairs, "select_on": sel, "modality_dropout": args.modality_dropout,
            "trainable_params": model.n_total(), "train_seconds": train_seconds,
            "gpu_hours": train_seconds / 3600.0 if device.type == "cuda" else None, "dev": final_dev, **info}
    if two and model.fusion != "avg":
        meta["fusion_trainable_params"] = sum(p.numel() for p in model.fusion_params())
    write_json(ckpt_dir / "meta.json", meta)

    dev_results, results = {}, {}
    if not args.no_eval:
        kw = dict(save_preds=not args.no_save_preds, audio_mode=args.audio_mode)
        dev_results = evaluate_conditions(cfg, model, args.dataset, "dev", args.audio_encoder, args.text_encoder,
                                          args.audio_stats, args.audio_layers, args.text_feature, device, ckpt_dir, **kw)
        if args.final:
            results = evaluate_conditions(cfg, model, args.dataset, "test", args.audio_encoder, args.text_encoder,
                                          args.audio_stats, args.audio_layers, args.text_feature, device, ckpt_dir, **kw)
    metrics = {**meta, "dev_results": dev_results, "results": results, "test_evaluations": 1 if results else 0,
               "env": env_info(cfg)}
    write_json(ckpt_dir / "metrics.json", metrics)
    headline = dev_results.get("clean_gold", {}).get("fused_acc")
    print(f"[{name} seed{args.seed}] dev(select)={final_dev['fused']:.4f} dev clean_gold={headline} "
          f"params={meta['trainable_params']} test_evaluated={bool(results)}")


if __name__ == "__main__":
    main()

from pathlib import Path

import torch

from src.common.utils import read_json
from src.models.heads import IntentHeadModel


def build_from_arch(arch):
    return IntentHeadModel(
        arch["fusion"], arch["n_labels"], arch["text_dim"], arch["audio_dim"], arch["audio_layers_n"],
        arch["audio_pool"], arch["branch_hidden"], arch["fusion_hidden"], arch["gate_hidden"], arch["dropout"],
        arch["loss_mode"], tuple(arch["loss_weights"]), arch.get("audio_mode", "pooled"),
        arch.get("frame_arch", "attn"), arch.get("frame_dim", 256), arch.get("frame_layers", 1),
        arch.get("route", "soft"))


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "meta.json")
    model = build_from_arch(meta["arch"])
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return model.to(device).eval(), meta

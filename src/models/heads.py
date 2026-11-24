import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSION_TYPES = ["text_only", "audio_only", "avg", "prob", "logit_product", "emb", "gated", "xattn", "reliability"]
NEEDS_TEXT = {"text_only", "avg", "prob", "logit_product", "emb", "gated", "xattn", "reliability"}
NEEDS_AUDIO = {"audio_only", "avg", "prob", "logit_product", "emb", "gated", "xattn", "reliability"}
BRANCH_MODULES = ("text_in", "audio_pool", "text_head", "audio_head")
REL_DIM = 7
STAT_DIM = 5
FRAME_ARCHS = ["attn", "bilstm", "transformer"]


def entropy_from_logits(logits):
    lp = F.log_softmax(logits, dim=-1)
    return -(lp.exp() * lp).sum(-1)


class AudioPool(nn.Module):
    def __init__(self, n_layers, dim, mode, dropout):
        super().__init__()
        self.mode = mode
        self.layer_logits = nn.Parameter(torch.zeros(n_layers)) if mode == "weighted" and n_layers > 1 else None
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.layer_logits is None:
            z = x[:, -1]
        else:
            w = torch.softmax(self.layer_logits, 0)
            z = (x * w[None, :, None]).sum(1)
        return self.drop(self.ln(z))


def sinusoid(T, d, device):
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class FrameEncoder(nn.Module):
    def __init__(self, in_dim, arch, d=256, dropout=0.1, n_layers=1, n_heads=4):
        super().__init__()
        assert arch in FRAME_ARCHS, arch
        self.arch, self.d = arch, d
        self.ln = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d)
        self.drop = nn.Dropout(dropout)
        if arch == "bilstm":
            self.rnn = nn.LSTM(d, d // 2, num_layers=n_layers, batch_first=True, bidirectional=True)
        elif arch == "transformer":
            layer = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout, batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, x, mask):
        h = self.drop(self.proj(self.ln(x)))
        if self.arch == "bilstm":
            lens = mask.sum(1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        elif self.arch == "transformer":
            h = h + sinusoid(h.shape[1], self.d, h.device)[None]
            h = self.tr(h, src_key_padding_mask=~mask)
        s = self.att(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)
        pooled = (a.unsqueeze(-1) * h).sum(1)
        return self.drop(pooled), h


class TextIn(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.ln(x))


def make_head(in_dim, n_labels, hidden, dropout):
    if hidden and hidden > 0:
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_labels))
    return nn.Linear(in_dim, n_labels)


def mlp(in_dim, hidden, out_dim, dropout):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))


class IntentHeadModel(nn.Module):
    def __init__(self, fusion, n_labels, text_dim=None, audio_dim=None, audio_layers=13, audio_pool="weighted",
                 branch_hidden=0, fusion_hidden=512, gate_hidden=64, dropout=0.1, loss_mode="fixed",
                 loss_weights=(0.25, 0.25, 0.5), audio_mode="pooled", frame_arch="attn", frame_dim=256,
                 frame_layers=1, route="soft"):
        super().__init__()
        assert fusion in FUSION_TYPES, fusion
        self.fusion = fusion
        self.n_labels = n_labels
        self.loss_mode = loss_mode
        self.audio_mode = audio_mode
        self.route = route
        self.register_buffer("loss_w", torch.tensor(loss_weights, dtype=torch.float32))
        self.has_text = fusion in NEEDS_TEXT
        self.has_audio = fusion in NEEDS_AUDIO
        self.branches_frozen = False
        if fusion == "xattn" and audio_mode != "frames":
            raise ValueError("fusion 'xattn' needs --audio_mode frames")
        ad = frame_dim if audio_mode == "frames" else audio_dim
        self.ad = ad
        if self.has_text:
            self.text_in = TextIn(text_dim, dropout)
            self.text_head = make_head(text_dim, n_labels, branch_hidden, dropout)
            self.register_buffer("T_text", torch.tensor(1.0))
        if self.has_audio:
            if audio_mode == "frames":
                self.audio_pool = FrameEncoder(audio_dim, frame_arch, frame_dim, dropout, frame_layers)
            else:
                self.audio_pool = AudioPool(audio_layers, audio_dim, audio_pool, dropout)
            self.audio_head = make_head(ad, n_labels, branch_hidden, dropout)
            self.register_buffer("T_audio", torch.tensor(1.0))
        self.register_buffer("rel_mean", torch.zeros(REL_DIM - 1))
        self.register_buffer("rel_std", torch.ones(REL_DIM - 1))
        K = n_labels
        if fusion == "prob":
            self.fuse = mlp(2 * K, fusion_hidden, K, dropout)
        elif fusion == "logit_product":
            self.w_t = nn.Parameter(torch.tensor(1.0))
            self.w_a = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.zeros(K))
        elif fusion == "emb":
            self.fuse = mlp(text_dim + ad, fusion_hidden, K, dropout)
        elif fusion == "gated":
            self.gate = nn.Linear(text_dim + ad, 1)
        elif fusion == "xattn":
            self.xq = nn.Linear(text_dim, ad)
            self.xattn = nn.MultiheadAttention(ad, 4, dropout=dropout, batch_first=True)
            self.fuse = mlp(text_dim + 2 * ad, fusion_hidden, K, dropout)
        elif fusion == "reliability":
            self.joint = mlp(text_dim + ad, fusion_hidden, K, dropout)
            self.gate = nn.Sequential(nn.Linear(REL_DIM + STAT_DIM, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 3))
        if loss_mode == "uncertainty":
            self.log_var = nn.Parameter(torch.zeros(3))
        elif loss_mode == "learned_convex":
            self.alpha_logits = nn.Parameter(torch.log(torch.tensor([0.3, 0.3, 0.4])))

    def set_rel_stats(self, mean, std):
        self.rel_mean.copy_(mean)
        self.rel_std.copy_(std.clamp(min=1e-6))

    def freeze_branches(self):
        self.branches_frozen = True
        for name in BRANCH_MODULES:
            if hasattr(self, name):
                for p in getattr(self, name).parameters():
                    p.requires_grad = False
        self.eval_branches()

    def eval_branches(self):
        for name in BRANCH_MODULES:
            if hasattr(self, name):
                getattr(self, name).eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and self.branches_frozen:
            self.eval_branches()
        return self

    def branch_params(self):
        out = []
        for name in BRANCH_MODULES:
            if hasattr(self, name):
                out += list(getattr(self, name).parameters())
        return out

    def fusion_params(self):
        ids = {id(p) for p in self.branch_params()}
        return [p for p in self.parameters() if id(p) not in ids and p.requires_grad]

    def rel_features(self, rel):
        has = rel[:, -1:]
        x = (rel[:, :-1] - self.rel_mean) / self.rel_std
        return torch.cat([x * has, has], dim=-1)

    def forward(self, text=None, audio=None, rel=None, drop_t=None, drop_a=None, amask=None):
        out = {}
        B = (text if text is not None else audio).shape[0]
        zt = za = lt = la = seq = None
        if self.has_text:
            if drop_t is not None:
                text = text * (~drop_t).float()[:, None]
            zt = self.text_in(text)
            lt = self.text_head(zt) / self.T_text
            if drop_t is not None:
                lt = torch.where(drop_t[:, None], torch.zeros_like(lt), lt)
            out["text_logits"] = lt
        if self.has_audio:
            if drop_a is not None:
                audio = audio * (~drop_a).float().view(-1, *([1] * (audio.dim() - 1)))
            if self.audio_mode == "frames":
                za, seq = self.audio_pool(audio, amask)
            else:
                za = self.audio_pool(audio)
            la = self.audio_head(za) / self.T_audio
            if drop_a is not None:
                la = torch.where(drop_a[:, None], torch.zeros_like(la), la)
            out["audio_logits"] = la
        f = self.fusion
        if f == "text_only":
            fused = lt
        elif f == "audio_only":
            fused = la
        elif f == "avg":
            fused = 0.5 * (F.log_softmax(lt, -1) + F.log_softmax(la, -1))
        elif f == "prob":
            fused = self.fuse(torch.cat([F.softmax(lt, -1), F.softmax(la, -1)], -1))
        elif f == "logit_product":
            fused = self.w_t * F.log_softmax(lt, -1) + self.w_a * F.log_softmax(la, -1) + self.bias
        elif f == "emb":
            fused = self.fuse(torch.cat([zt, za], -1))
        elif f == "gated":
            g = torch.sigmoid(self.gate(torch.cat([zt, za], -1)))
            fused = g * F.log_softmax(lt, -1) + (1 - g) * F.log_softmax(la, -1)
            out["gate"] = g
        elif f == "xattn":
            q = self.xq(zt).unsqueeze(1)
            ctx, attn = self.xattn(q, seq, seq, key_padding_mask=~amask)
            fused = self.fuse(torch.cat([zt, ctx.squeeze(1), za], -1))
        elif f == "reliability":
            relf = self.rel_features(rel if rel is not None else torch.zeros(B, REL_DIM, device=zt.device))
            pt, pa = F.softmax(lt, -1), F.softmax(la, -1)
            stats = torch.stack([
                entropy_from_logits(lt), entropy_from_logits(la), pt.max(-1).values, pa.max(-1).values,
                (pt.argmax(-1) == pa.argmax(-1)).float()], dim=-1)
            g = F.softmax(self.gate(torch.cat([relf, stats], -1)), dim=-1)
            if self.route == "hard":
                hard = F.one_hot(g.argmax(-1), 3).float()
                g = hard + (g - g.detach()) if self.training else hard
            joint = self.joint(torch.cat([zt, za], -1))
            experts = torch.stack([F.log_softmax(lt, -1), F.log_softmax(la, -1), F.log_softmax(joint, -1)], dim=1)
            fused = (g.unsqueeze(-1) * experts).sum(1)
            out["gate"] = g
        out["fused_logits"] = fused
        return out

    def _weights(self):
        if self.loss_mode == "fixed":
            return self.loss_w[0], self.loss_w[1], self.loss_w[2]
        if self.loss_mode == "learned_convex":
            w = F.softmax(self.alpha_logits, 0)
            return w[0], w[1], w[2]
        return None

    @staticmethod
    def _ce(logits, y, present):
        if present is None:
            return F.cross_entropy(logits, y)
        if not bool(present.any()):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[present], y[present])

    def loss(self, out, y, drop_t=None, drop_a=None, stage="joint"):
        pt = None if drop_t is None else ~drop_t
        pa = None if drop_a is None else ~drop_a
        zero = out["fused_logits"].sum() * 0.0
        lt = self._ce(out["text_logits"], y, pt) if "text_logits" in out else zero
        la = self._ce(out["audio_logits"], y, pa) if "audio_logits" in out else zero
        if self.fusion in ("text_only", "audio_only"):
            l = lt if self.fusion == "text_only" else la
            return l, {"loss": float(l.detach())}
        lf = F.cross_entropy(out["fused_logits"], y)
        if stage == "branches":
            w = self.loss_w
            total = w[0] * lt + w[1] * la
        elif stage == "fusion":
            total = lf
        elif self.loss_mode == "uncertainty":
            s = self.log_var
            total = torch.exp(-s[0]) * lt + s[0] + torch.exp(-s[1]) * la + s[1] + torch.exp(-s[2]) * lf + s[2]
        else:
            w = self._weights()
            total = w[0] * lt + w[1] * la + w[2] * lf
        return total, {"lt": float(lt.detach()), "la": float(la.detach()), "lf": float(lf.detach())}

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_total(self):
        return sum(p.numel() for p in self.parameters())


def fit_temperature(logits, labels, max_iter=100):
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())


def expected_calibration_error(probs, labels, n_bins=15):
    conf, pred = probs.max(-1)
    acc = pred.eq(labels).float()
    ece = torch.zeros((), device=probs.device)
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return float(ece)

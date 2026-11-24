import numpy as np, torch
from types import SimpleNamespace
class _M:
    dims = SimpleNamespace(n_mels=80)
def load_model(name, device="cpu"):
    return _M()
class DecodingOptions:
    def __init__(self, **kw): self.kw = kw
def pad_or_trim(a, length=480000):
    a = np.asarray(a)
    return a[:length] if len(a) >= length else np.pad(a, (0, length - len(a)))
def log_mel_spectrogram(audio, n_mels=80):
    return torch.zeros(n_mels, 3000) + float(np.abs(audio).mean())
def decode(model, mel, options):
    out = []
    for m in mel:
        v = float(m[0, 0])
        out.append(SimpleNamespace(text=" Hello, World! w1 w2.", avg_logprob=-0.2 - v, no_speech_prob=0.01, compression_ratio=1.4))
    return out

import glob
import hashlib
import os
import re

import numpy as np

_NOISE_CACHE = {}
_NOISE_LIST = {}


def _seed(utt_id, cond):
    h = hashlib.sha1(f"{utt_id}|{cond}".encode()).hexdigest()
    return int(h[:8], 16)


def parse_condition(cond):
    out = {"snr": None, "rt60": None}
    if cond in (None, "", "clean"):
        return out
    for part in cond.split("+"):
        m = re.fullmatch(r"snr(-?\d+(?:\.\d+)?)", part)
        if m:
            out["snr"] = float(m.group(1))
            continue
        m = re.fullmatch(r"rev(\d+)", part)
        if m:
            out["rt60"] = int(m.group(1)) / 1000.0
            continue
        raise ValueError(f"Unknown audio condition part: {part}")
    return out


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _colored_noise(n, kind, rng):
    w = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return w
    spec = np.fft.rfft(w)
    f = np.arange(len(spec), dtype=np.float32)
    f[0] = 1.0
    if kind == "pink":
        spec = spec / np.sqrt(f)
    else:
        spec = spec / f
    x = np.fft.irfft(spec, n).astype(np.float32)
    return x / (np.std(x) + 1e-9)


def _list_noise(noise_dir):
    if noise_dir not in _NOISE_LIST:
        files = []
        for ext in ("wav", "flac"):
            files += glob.glob(os.path.join(noise_dir, "**", f"*.{ext}"), recursive=True)
        _NOISE_LIST[noise_dir] = sorted(files)
    return _NOISE_LIST[noise_dir]


def _load_noise_file(path, sr):
    if path in _NOISE_CACHE:
        return _NOISE_CACHE[path]
    import soundfile as sf
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        import librosa
        x = librosa.resample(x, orig_sr=fs, target_sr=sr)
    if len(_NOISE_CACHE) > 64:
        _NOISE_CACHE.pop(next(iter(_NOISE_CACHE)))
    _NOISE_CACHE[path] = x
    return x


def _get_noise(n, rng, sr, noise_dir):
    if noise_dir:
        files = _list_noise(noise_dir)
        if files:
            path = files[int(rng.integers(len(files)))]
            x = _load_noise_file(path, sr)
            if len(x) < 2:
                return _colored_noise(n, "white", rng)
            if len(x) < n:
                x = np.tile(x, int(np.ceil(n / len(x))))
            start = int(rng.integers(0, len(x) - n + 1))
            return x[start:start + n].astype(np.float32)
    kind = ["white", "pink", "brown"][int(rng.integers(3))]
    return _colored_noise(n, kind, rng)


def _reverb(wav, sr, rt60, rng):
    from scipy.signal import fftconvolve
    n = int(rt60 * sr)
    t = np.arange(n, dtype=np.float32) / sr
    h = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9078 * t / rt60)
    h[0] = 1.0
    h = h / (np.sqrt(np.sum(h ** 2)) + 1e-9)
    y = fftconvolve(wav, h)[: len(wav)].astype(np.float32)
    return y * (_rms(wav) / _rms(y))


def apply_condition(wav, sr, cond, utt_id, noise_dir=None):
    p = parse_condition(cond)
    if p["snr"] is None and p["rt60"] is None:
        return wav
    rng = np.random.default_rng(_seed(utt_id, cond))
    y = wav.astype(np.float32)
    if p["rt60"] is not None:
        y = _reverb(y, sr, p["rt60"], rng)
    if p["snr"] is not None:
        noise = _get_noise(len(y), rng, sr, noise_dir)
        ps = _rms(y) ** 2
        pn = _rms(noise) ** 2
        scale = np.sqrt(ps / (pn * (10.0 ** (p["snr"] / 10.0))))
        y = y + scale * noise
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def load_wav(path, sr=16000, max_seconds=None):
    import soundfile as sf
    try:
        x, fs = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if fs != sr:
            import librosa
            x = librosa.resample(x, orig_sr=fs, target_sr=sr)
    except Exception:
        import librosa
        x, _ = librosa.load(path, sr=sr)
    if max_seconds is not None:
        x = x[: int(max_seconds * sr)]
    if len(x) < 400:
        x = np.pad(x, (0, 400 - len(x)))
    return x.astype(np.float32)

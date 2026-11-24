import hashlib
import json
import platform
import subprocess
import sys
import time


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def env_info(cfg):
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "config_hash": config_hash(cfg),
            "git_commit": git_commit(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import numpy
        import torch
        info.update({"torch": torch.__version__, "numpy": numpy.__version__, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info

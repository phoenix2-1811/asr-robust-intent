import argparse
import hashlib
import random
from pathlib import Path

from src.asr.transcribe import asr_out_path, build_output
from src.common.text_norm import normalize_text
from src.common.utils import load_config, read_jsonl, write_json

CONFUSIONS = {
    "to": ["too", "two"], "too": ["to"], "two": ["to"], "for": ["four", "fore"], "four": ["for"],
    "there": ["their"], "their": ["there"], "your": ["you're"], "you're": ["your"],
    "write": ["right"], "right": ["write"], "no": ["know"], "know": ["no"], "new": ["knew"],
    "one": ["won"], "by": ["buy"], "buy": ["by"], "hear": ["here"], "here": ["hear"],
    "whether": ["weather"], "weather": ["whether"], "off": ["of"], "of": ["off"], "in": ["on"],
    "on": ["in"], "the": ["a"], "a": ["the"], "play": ["pray"], "turn": ["torn"], "call": ["cal"],
    "set": ["sit"], "light": ["like"], "lights": ["likes"], "alarm": ["alarms"], "music": ["muse"],
    "email": ["mail"], "mail": ["email"],
}


def build_vocab(examples):
    vocab = {}
    for ex in examples:
        for w in normalize_text(ex["transcript"]).split():
            vocab[w] = vocab.get(w, 0) + 1
    return sorted(vocab)


def corrupt(text, target_wer, rng, vocab, ops):
    words = text.split()
    if not words:
        return text
    expected = len(words) * target_wer
    n_err = int(expected) + (1 if rng.random() < (expected - int(expected)) else 0)
    n_err = min(n_err, len(words))
    idxs = rng.sample(range(len(words)), n_err)
    kinds = ["sub", "del", "ins"]
    weights = [ops["sub"], ops["del"], ops["ins"]]
    out = list(words)
    for i in idxs:
        op = rng.choices(kinds, weights)[0]
        w = out[i]
        if op == "sub":
            cands = CONFUSIONS.get(w)
            out[i] = rng.choice(cands) if cands and rng.random() < 0.5 else rng.choice(vocab)
        elif op == "del":
            out[i] = ""
        else:
            out[i] = w + " " + rng.choice(vocab)
    return " ".join(x for x in " ".join(out).split(" ") if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = cfg["datasets"][args.dataset]
    c = cfg["conditions"]
    vocab = build_vocab(read_jsonl(Path(ds["root"]) / "train.jsonl"))
    for split in args.splits:
        p = Path(ds["root"]) / f"{split}.jsonl"
        if not p.exists():
            continue
        examples = read_jsonl(p)
        for target in c["synthetic_wer"]:
            tag = f"synth-wer{int(round(target * 100))}"
            results = {}
            for ex in examples:
                h = hashlib.sha1(f"{ex['utt_id']}|{tag}|{c['synthetic_seed']}".encode()).hexdigest()
                rng = random.Random(int(h[:12], 16))
                results[ex["utt_id"]] = {"text": corrupt(normalize_text(ex["transcript"]), target, rng, vocab,
                                                         c["synthetic_ops"])}
            out = build_output(tag, args.dataset, split, "synthetic", "clean", examples, results)
            write_json(asr_out_path(cfg, args.dataset, split, tag), out)
            print(f"[{args.dataset}/{split}] {tag}: target={target:.2f} measured corpus WER={out['wer']:.4f}")


if __name__ == "__main__":
    main()

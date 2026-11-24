import argparse
import zlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoTokenizer, Wav2Vec2ForCTC

from src.asr.transcribe import asr_out_path, build_output
from src.common.text_norm import normalize_text
from src.common.utils import load_config, read_jsonl, short, write_json
from src.data.audio_conditions import apply_condition, load_wav


def ctc_source_name(model_name, cond):
    return f"w2v2ctc-{short(model_name).replace('wav2vec2-', '')}@{cond}"


@torch.no_grad()
def decode_ctc(model, fe, tok, examples, cond, noise_dir, max_seconds, batch_size, device):
    blank = model.config.pad_token_id
    use_mask = getattr(model.config, "feat_extract_norm", "group") == "layer"
    results = {}
    wavs = {}
    order = sorted(range(len(examples)), key=lambda i: examples[i].get("_len", 0))
    for i in tqdm(range(0, len(order), batch_size), desc="ctc"):
        idxs = order[i:i + batch_size]
        batch_w = []
        for j in idxs:
            ex = examples[j]
            w = load_wav(ex["audio_path"], 16000, max_seconds)
            batch_w.append(apply_condition(w, 16000, cond, ex["utt_id"], noise_dir))
        norm = [fe(w, sampling_rate=16000, return_tensors="np")["input_values"][0] for w in batch_w]
        T = max(len(x) for x in norm)
        x = np.zeros((len(norm), T), dtype=np.float32)
        am = torch.zeros(len(norm), T, dtype=torch.long)
        for k, a in enumerate(norm):
            x[k, :len(a)] = a
            am[k, :len(a)] = 1
        kw = {"attention_mask": am.to(device)} if use_mask else {}
        logits = model(torch.from_numpy(x).to(device), **kw).logits.float()
        lp = torch.log_softmax(logits, -1)
        frames = [int(model._get_feat_extract_output_lengths(torch.tensor(len(a)))) for a in norm]
        for k, j in enumerate(idxs):
            n = min(frames[k], lp.shape[1])
            l = lp[k, :n]
            mx, am_ = l.max(-1)
            ids = am_.cpu().tolist()
            nonblank = [t != blank for t in ids]
            collapsed, prev = [], None
            for t in ids:
                if t != prev and t != blank:
                    collapsed.append(t)
                prev = t
            text = normalize_text(tok.decode(collapsed))
            nb = torch.tensor(nonblank, device=l.device)
            avg_lp = float(mx[nb].mean()) if bool(nb.any()) else -5.0
            raw = text.encode("utf-8")
            results[examples[j]["utt_id"]] = {
                "text": text, "avg_logprob": avg_lp, "no_speech_prob": float(1.0 - nb.float().mean()),
                "compression_ratio": len(raw) / max(len(zlib.compress(raw)), 1) if raw else 0.0}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--model", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--audio_cond", default="clean")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_utts", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = cfg["datasets"][args.dataset]
    examples = read_jsonl(Path(ds["root"]) / f"{args.split}.jsonl")
    if args.max_utts:
        examples = examples[: args.max_utts]
    source = ctc_source_name(args.model, args.audio_cond)
    out_path = asr_out_path(cfg, args.dataset, args.split, source)
    if args.skip_existing and out_path.exists():
        print(f"exists, skipping: {out_path}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    res = decode_ctc(model, fe, tok, examples, args.audio_cond, cfg["noise"]["noise_dir"],
                     ds["max_audio_seconds"], args.batch_size, device)
    out = build_output(source, args.dataset, args.split, short(args.model), args.audio_cond, examples, res)
    write_json(out_path, out)
    print(f"{source} [{args.dataset}/{args.split}] WER={out['wer']:.4f} n={out['n_utts']} -> {out_path}")


if __name__ == "__main__":
    main()

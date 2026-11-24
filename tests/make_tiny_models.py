import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


def main(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    from transformers import (BertTokenizerFast, DistilBertConfig, DistilBertModel, Wav2Vec2Config,
                              Wav2Vec2FeatureExtractor, Wav2Vec2Model, WhisperConfig, WhisperFeatureExtractor,
                              WhisperModel)
    w2v = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(16, 16, 16), conv_stride=(5, 2, 2), conv_kernel=(10, 3, 3),
                         num_feat_extract_layers=3, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                         feat_extract_norm="group")
    m = Wav2Vec2Model(w2v)
    m.save_pretrained(root / "wav2vec2-tiny")
    Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True, return_attention_mask=False).save_pretrained(
        root / "wav2vec2-tiny")
    w2v_ln = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                            conv_dim=(16, 16, 16), conv_stride=(5, 2, 2), conv_kernel=(10, 3, 3),
                            num_feat_extract_layers=3, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                            feat_extract_norm="layer", do_stable_layer_norm=True)
    Wav2Vec2Model(w2v_ln).save_pretrained(root / "wav2vec2-tiny-ln")
    Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True, return_attention_mask=True).save_pretrained(
        root / "wav2vec2-tiny-ln")
    wc = WhisperConfig(d_model=32, encoder_layers=2, decoder_layers=1, encoder_attention_heads=2,
                       decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=80,
                       vocab_size=120, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=3)
    WhisperModel(wc).save_pretrained(root / "whisper-tiny-rand")
    WhisperFeatureExtractor(feature_size=80).save_pretrained(root / "whisper-tiny-rand")
    words = [f"w{i}" for i in range(40)]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (root / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(root / "vocab.txt"))
    dc = DistilBertConfig(vocab_size=len(vocab), dim=32, n_layers=2, n_heads=2, hidden_dim=64)
    DistilBertModel(dc).save_pretrained(root / "distilbert-tiny")
    tok.save_pretrained(root / "distilbert-tiny")
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
    chars = ["<pad>", "<s>", "</s>", "<unk>", "|"] + list("abcdefghijklmnopqrstuvwxyz0123456789")
    (root / "ctc_vocab.json").write_text(json.dumps({c: i for i, c in enumerate(chars)}))
    ctc_tok = Wav2Vec2CTCTokenizer(str(root / "ctc_vocab.json"), unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    ctc_cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                             conv_dim=(16, 16, 16), conv_stride=(5, 2, 2), conv_kernel=(10, 3, 3),
                             num_feat_extract_layers=3, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                             feat_extract_norm="group", vocab_size=len(chars), pad_token_id=0)
    Wav2Vec2ForCTC(ctc_cfg).save_pretrained(root / "wav2vec2-ctc-tiny")
    fe = Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True, return_attention_mask=False)
    Wav2Vec2Processor(feature_extractor=fe, tokenizer=ctc_tok).save_pretrained(root / "wav2vec2-ctc-tiny")
    print("tiny models ->", root)


def make_dataset(root, ds_root):
    ds_root = Path(ds_root)
    (ds_root).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    K = 4
    for split, n in (("train", 24), ("dev", 8), ("test", 8)):
        rows = []
        for i in range(n):
            k = i % K
            dur = 1.0 + 0.4 * (i % 4)
            t = np.arange(int(16000 * dur)) / 16000
            wav = (0.3 * np.sin(2 * np.pi * (200 + 80 * k) * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
            p = ds_root / "wavs" / f"{split}_{i}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, wav, 16000)
            text = " ".join(f"w{(k * 3 + j) % 40}" for j in range(3 + (i % 3)))
            rows.append({"audio_path": str(p), "transcript": text, "label": f"c{k}", "utt_id": f"{split}{i}",
                         "group": f"{split}-g{i // 2}"})
        with open(ds_root / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    json.dump({f"c{k}": k for k in range(K)}, open(ds_root / "label_map.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
    if len(sys.argv) > 2:
        make_dataset(sys.argv[1], sys.argv[2])

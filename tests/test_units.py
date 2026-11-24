import csv
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.bootstrap import cluster_bootstrap_diff, mcnemar_exact
from src.asr.synthetic_corruption import corrupt
from src.common.text_norm import corpus_wer, normalize_text, utterance_wer
from src.data.audio_conditions import apply_condition, parse_condition
from src.data.prepare_fsc import unseen_utterance_split
from src.models.heads import FRAME_ARCHS, FUSION_TYPES, IntentHeadModel, REL_DIM, fit_temperature


def test_text_norm_and_wer():
    assert normalize_text("What's the Weather, TODAY?!") == "what's the weather today"
    assert utterance_wer("a b c d", "a x c") == 0.5
    assert abs(corpus_wer(["a b", "c d"], ["a b", "c"]) - 0.25) < 1e-9


def test_conditions():
    x = (0.3 * np.sin(np.arange(16000) / 10)).astype(np.float32)
    for snr in (20, 10, 0):
        y = apply_condition(x, 16000, f"snr{snr}", "u")
        alpha = float(np.dot(y, x) / np.dot(x, x))
        n = y - alpha * x
        assert abs(10 * np.log10(np.mean((alpha * x) ** 2) / np.mean(n ** 2)) - snr) < 0.3
    assert np.array_equal(apply_condition(x, 16000, "clean", "u"), x)
    assert parse_condition("rev400+snr10") == {"snr": 10.0, "rt60": 0.4}


def test_synthetic_corruption_hits_target():
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(50)]
    refs = [" ".join(f"w{(i + j) % 50}" for j in range(5)) for i in range(2000)]
    for target in (0.1, 0.2, 0.3):
        hyps = [corrupt(r, target, rng, vocab, {"sub": 0.6, "del": 0.2, "ins": 0.2}) for r in refs]
        assert abs(corpus_wer(refs, hyps) - target) < 0.03, target


def test_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 5).astype(str)
    a = (rng.random(1000) < 0.9).astype(float)
    b = (rng.random(1000) < 0.8).astype(float)
    r = cluster_bootstrap_diff(a, b, groups, 2000)
    assert r["ci95"][0] > 0 and r["p_boot"] < 0.05
    assert "p_mcnemar_exact" in mcnemar_exact(a, b)


def test_fsc_unseen_split_has_no_sentence_overlap():
    rows = []
    for k in range(5):
        for s in range(10):
            for r in range(3):
                rows.append({"label": f"c{k}", "group": f"s{k}-{s}", "utt_id": f"{k}-{s}-{r}"})
    tr, dv, te = unseen_utterance_split(rows, 0)
    g = lambda rs: {r["group"] for r in rs}
    assert not (g(tr) & g(te)) and not (g(tr) & g(dv)) and not (g(dv) & g(te))
    assert {r["label"] for r in te} == {f"c{k}" for k in range(5)}


def test_all_fusions_forward_backward_and_two_stage_freeze():
    B, K, Dt, Da, L, R = 8, 6, 16, 20, 5, REL_DIM
    for f in FUSION_TYPES:
        if f == "xattn":
            continue
        m = IntentHeadModel(f, K, Dt, Da, L, "weighted")
        text, audio, rel = torch.randn(B, Dt), torch.randn(B, L, Da), torch.randn(B, R)
        rel[:, -1] = 1.0
        y = torch.randint(0, K, (B,))
        dt = torch.tensor([True] + [False] * (B - 1))
        out = m(text if m.has_text else None, audio if m.has_audio else None, rel, dt if m.has_text else None, None)
        loss, _ = m.loss(out, y, dt if m.has_text else None, None, "joint")
        loss.backward()
        assert out["fused_logits"].shape == (B, K)
    m = IntentHeadModel("reliability", K, Dt, Da, L, "weighted")
    m.freeze_branches()
    assert all(not p.requires_grad for p in m.text_head.parameters())
    assert all(p.requires_grad for p in m.gate.parameters())
    m.train()
    assert not m.text_in.training and m.gate.training


def test_frame_encoders_and_cross_attention_respect_padding():
    B, T, K, Dt, Da = 4, 9, 5, 16, 24
    lens = torch.tensor([9, 5, 3, 7])
    mask = torch.arange(T)[None] < lens[:, None]
    for arch in FRAME_ARCHS:
        for fusion in ("audio_only", "xattn", "reliability"):
            m = IntentHeadModel(fusion, K, Dt, Da, 1, "weighted", audio_mode="frames", frame_arch=arch, frame_dim=32,
                                dropout=0.0).eval()
            x = torch.randn(B, T, Da)
            text = torch.randn(B, Dt)
            rel = torch.zeros(B, REL_DIM)
            o1 = m(text if m.has_text else None, x, rel, None, None, mask)["fused_logits"]
            x2 = x.clone()
            x2[~mask] = torch.randn(int((~mask).sum()), Da) * 50
            o2 = m(text if m.has_text else None, x2, rel, None, None, mask)["fused_logits"]
            assert torch.allclose(o1, o2, atol=1e-4), (arch, fusion)
            y = torch.randint(0, K, (B,))
            loss, _ = m.loss(m(text if m.has_text else None, x, rel, None, None, mask), y)
            loss.backward()


def test_hard_routing_is_one_hot_at_eval():
    m = IntentHeadModel("reliability", 5, 8, 8, 3, "weighted", route="hard").eval()
    out = m(torch.randn(6, 8), torch.randn(6, 3, 8), torch.zeros(6, REL_DIM))
    g = out["gate"]
    assert torch.all((g == 0) | (g == 1)) and torch.allclose(g.sum(-1), torch.ones(6))


def test_temperature_fit_reduces_nll_on_overconfident_logits():
    torch.manual_seed(0)
    y = torch.randint(0, 5, (2000,))
    logits = torch.randn(2000, 5)
    logits[torch.arange(2000), y] += 1.0
    logits = logits * 6.0
    t = fit_temperature(logits, y)
    assert t > 1.5
    assert torch.nn.functional.cross_entropy(logits / t, y) < torch.nn.functional.cross_entropy(logits, y)


def test_unicode_normalization_and_alignment_ops():
    assert normalize_text("Wie spät ist es? Привет, мир!") == "wie spät ist es привет мир"
    from src.common.text_norm import word_edit_ops
    ops = word_edit_ops("turn on the light", "turn off light")
    kinds = [o[0] for o in ops]
    assert kinds.count("sub") == 1 and kinds.count("del") == 1


def test_error_categories():
    from src.analysis.error_taxonomy import error_categories
    cats = dict((c, op) for op, c in error_categories("set alarm for nine am", "set alarm for five am"))
    assert "number/time" in cats
    assert ("sub", "negation/polarity") in error_categories("do not call", "do now call")


def test_gain_identity_and_cmi():
    import os
    from src.analysis.gain_bound import stats_for
    from src.analysis.independence import cond_mutual_info
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.integers(0, 5, n)
    shared = rng.random(n)
    tc = (shared + 0.3 * rng.random(n)) < 0.8
    ac = (shared + 0.3 * rng.random(n)) < 0.6
    with tempfile.TemporaryDirectory() as d:
        pf = Path(d) / "f.npz"
        tp = np.where(tc, y, (y + 1) % 5)
        ap = np.where(ac, y, (y + 2) % 5)
        np.savez(pf, labels=y, text_pred=tp, audio_pred=ap, fused_pred=np.where(tc | ac, y, 0))
        row = stats_for([str(pf)], [])
    assert abs(row["oracle_router_gain"] - row["predicted_from_p_q_phi"]) < 1e-9
    assert row["phi_marginal"] > 0.1
    r = cond_mutual_info(y, tc.astype(int), ac.astype(int), n_perm=50)
    assert r["cmi_nats"] > r["permutation_null_mean"]
    r0 = cond_mutual_info(y, tc.astype(int), rng.permutation(ac).astype(int), n_perm=50)
    assert r0["permutation_p"] > 0.05


def test_import_external_roundtrip():
    from src.baselines.import_external import import_predictions
    from src.common.utils import write_json
    with tempfile.TemporaryDirectory() as d:
        old = Path.cwd()
        import os
        os.chdir(d)
        try:
            Path("data/x").mkdir(parents=True)
            rows = [{"audio_path": "a", "transcript": "t", "label": f"c{i % 3}", "utt_id": f"u{i}", "group": f"g{i}"} for i in range(30)]
            with open("data/x/test.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            write_json("data/x/label_map.json", {"c0": 0, "c1": 1, "c2": 2})
            cfg = {"datasets": {"x": {"root": "data/x"}}, "paths": {"checkpoints": "ck"}}
            import_predictions(cfg, "x", {r["utt_id"]: r["label"] for r in rows}, "sys", "clean_gold", 0)
            m = json.load(open("ck/x/external-sys/seed0/metrics.json"))
            assert m["results"]["clean_gold"]["fused_acc"] == 1.0
        finally:
            os.chdir(old)


def test_holm_and_ece_and_repro():
    from src.analysis.compare_all import holm
    from src.common.evalutil import ece_from, f1_macro
    from src.common.repro import config_hash
    adj = holm(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    conf = np.array([0.9] * 100)
    assert abs(ece_from(conf, np.array([1.0] * 90 + [0.0] * 10)) - 0.0) < 1e-9
    assert ece_from(conf, np.zeros(100)) > 0.85
    assert f1_macro(np.array([0, 1, 1, 2]), np.array([0, 1, 1, 2])) == 1.0
    assert config_hash({"a": 1}) == config_hash({"a": 1}) != config_hash({"a": 2})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

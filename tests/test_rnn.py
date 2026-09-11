"""Tests for the MDN-RNN and the latent sequence dataset.

Runnable two ways (pytest is not guaranteed to be installed here)::

    python -m tests.test_rnn
    pytest tests/test_rnn.py

Every test is a plain zero-argument function named ``test_*``.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from wm.rnn import MDNRNN, RNNConfig, load_rnn, rnn_loss, save_rnn
from wm.seq_data import LatentSequenceDataset

torch.manual_seed(0)


# ---------------------------------------------------------------- shapes


def test_forward_shapes() -> None:
    cfg = RNNConfig(z_dim=4, n_actions=3, hidden=16, n_gauss=5)
    m = MDNRNN(cfg)
    B, T = 7, 11
    z = torch.randn(B, T, 4)
    a = torch.eye(3)[torch.randint(0, 3, (B, T))]
    parts, h = m(z, a)

    assert parts["logits"].shape == (B, T, 5)
    assert parts["mean"].shape == (B, T, 5, 4)
    assert parts["logstd"].shape == (B, T, 5, 4)
    assert parts["hit_logit"].shape == (B, T, 1)
    assert parts["reward"].shape == (B, T, 1)
    assert h[0].shape == (1, B, 16) and h[1].shape == (1, B, 16)

    # log-std must be inside the clamp no matter what the linear head emits.
    assert parts["logstd"].min() >= -7.0 - 1e-6
    assert parts["logstd"].max() <= 2.0 + 1e-6

    # step() is forward() with T=1, and carrying h manually must match a
    # single full-sequence pass.
    h = m.init_hidden(B)
    outs = []
    for t in range(T):
        p, h = m.step(z[:, t], a[:, t], h)
        outs.append(p["logits"])
    seq, _ = m(z, a)
    assert torch.allclose(torch.cat(outs, 1), seq["logits"], atol=1e-5)


def test_mdn_nll_equals_gaussian_for_k1() -> None:
    """With one component the mixture NLL is exactly the diagonal Gaussian NLL."""
    cfg = RNNConfig(z_dim=3, n_actions=3, hidden=8, n_gauss=1, predict_delta=False)
    m = MDNRNN(cfg)
    B, T, Z = 2, 4, 3
    z = torch.randn(B, T, Z)
    a = torch.eye(3)[torch.randint(0, 3, (B, T))]
    target = torch.randn(B, T, Z)
    parts, _ = m(z, a)

    mean = parts["mean"][:, :, 0, :]
    logstd = parts["logstd"][:, :, 0, :]
    # -log N(y|mu,sigma) summed over dims, meaned over (B,T), by hand.
    hand = (
        0.5 * ((target - mean) / torch.exp(logstd)) ** 2
        + logstd
        + 0.5 * math.log(2 * math.pi)
    ).sum(-1).mean()

    got = m.mdn_nll(parts, target)
    assert torch.allclose(got, hand, atol=1e-6), (got.item(), hand.item())

    # And a fully hand-computed scalar, so the test does not merely check the
    # implementation against a paraphrase of itself.
    y, mu_, sd = 1.5, 0.5, 2.0
    logstd_c = torch.full((1, 1, 1, 1), math.log(sd))
    p = {
        "logits": torch.zeros(1, 1, 1),
        "mean": torch.full((1, 1, 1, 1), mu_),
        "logstd": logstd_c,
        "z_in": torch.zeros(1, 1, 1),
    }
    expected = 0.5 * ((y - mu_) / sd) ** 2 + math.log(sd) + 0.5 * math.log(2 * math.pi)
    m2 = MDNRNN(RNNConfig(z_dim=1, n_gauss=1, hidden=4, predict_delta=False))
    got2 = m2.mdn_nll(p, torch.full((1, 1, 1), y))
    assert abs(got2.item() - expected) < 1e-6, (got2.item(), expected)


def test_temperature_zero_returns_mode_mean() -> None:
    """tau -> 0 must be deterministic and equal the top component's mean."""
    m = MDNRNN(RNNConfig(z_dim=4, hidden=8, n_gauss=1, predict_delta=False))
    z = torch.randn(3, 5, 4)
    a = torch.eye(3)[torch.randint(0, 3, (3, 5))]
    parts, _ = m(z, a)

    s0 = m.sample_next(parts, temperature=0.0)
    assert torch.allclose(s0, parts["mean"][:, :, 0, :], atol=1e-7)
    assert torch.allclose(s0, m.sample_next(parts, temperature=0.0))  # deterministic

    # Multi-component: still the argmax component's mean, not the mixture mean.
    mm = MDNRNN(RNNConfig(z_dim=4, hidden=8, n_gauss=5, predict_delta=False))
    parts, _ = mm(z, a)
    k = parts["logits"].argmax(-1)
    ref = torch.stack(
        [torch.stack([parts["mean"][b, t, k[b, t]] for t in range(5)]) for b in range(3)]
    )
    assert torch.allclose(mm.sample_next(parts, temperature=0.0), ref, atol=1e-7)

    # tau=1 must actually be stochastic (otherwise the dream is a lie).
    s_a = mm.sample_next(parts, temperature=1.0)
    s_b = mm.sample_next(parts, temperature=1.0)
    assert not torch.allclose(s_a, s_b)


def test_predict_delta_roundtrip() -> None:
    """Residual mode: absolute outputs = raw head output + z_t, and the NLL
    target is the delta. Both directions checked."""
    m = MDNRNN(RNNConfig(z_dim=4, hidden=8, n_gauss=3, predict_delta=True))
    z = torch.randn(2, 6, 4)
    a = torch.eye(3)[torch.randint(0, 3, (2, 6))]
    parts, _ = m(z, a)

    abs_mean = m._abs_mean(parts)
    assert torch.allclose(abs_mean - z.unsqueeze(2), parts["mean"], atol=1e-6)

    z_next = z + 0.1 * torch.randn_like(z)
    assert torch.allclose(m._target_in_model_space(parts, z_next), z_next - z, atol=1e-6)

    # A delta model whose head outputs ~0 (our init) predicts z_{t+1} ~= z_t.
    s0 = m.sample_next(parts, temperature=0.0)
    assert (s0 - z).abs().max() < 0.5, "freshly-initialised delta model should be ~identity"

    # And with predict_delta=False the head output IS the absolute prediction.
    m2 = MDNRNN(RNNConfig(z_dim=4, hidden=8, n_gauss=3, predict_delta=False))
    p2, _ = m2(z, a)
    assert torch.allclose(m2._abs_mean(p2), p2["mean"], atol=1e-7)


def test_save_load_roundtrip() -> None:
    m = MDNRNN(RNNConfig(z_dim=4, hidden=8, n_gauss=3))
    with tempfile.TemporaryDirectory() as d:
        p = save_rnn(Path(d) / "rnn.pt", m, {"lr": 1e-3})
        m2, cfg = load_rnn(p)
    assert cfg.hidden == 8 and cfg.n_gauss == 3 and cfg.z_dim == 4
    z = torch.randn(2, 3, 4)
    a = torch.eye(3)[torch.randint(0, 3, (2, 3))]
    m.eval()
    assert torch.allclose(m(z, a)[0]["logits"], m2(z, a)[0]["logits"], atol=1e-6)


# ------------------------------------------------------------- learning test


def test_overfit_linear_system() -> None:
    """The model must actually learn a 2-D linear dynamical system.

    The system is a rotation, so ``z_{t+1}`` is a deterministic function of
    ``z_t`` alone and a competent model should drive the NLL strongly negative
    (a tight Gaussian on a deterministic target has arbitrarily low NLL). We
    only require a large drop from the untrained value, which is a weak enough
    bar to be robust and a strong enough one to catch a broken loss.
    """
    torch.manual_seed(0)
    theta = 0.2
    R = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    B, T = 64, 24
    z0 = torch.randn(B, 2) * 0.5
    zs = [z0]
    for _ in range(T):
        zs.append(zs[-1] @ R.T)
    seq = torch.stack(zs, 1)                    # (B, T+1, 2)
    z, z_next = seq[:, :-1], seq[:, 1:]
    a = torch.zeros(B, T, 3)
    a[:, :, 1] = 1.0
    batch = {
        "z_next": z_next,
        "hit": torch.zeros(B, T),
        "reward": torch.zeros(B, T),
    }

    m = MDNRNN(RNNConfig(z_dim=2, hidden=32, n_gauss=3, predict_delta=True))
    with torch.no_grad():
        parts, _ = m(z, a)
        start = m.mdn_nll(parts, z_next).item()

    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(400):
        parts, _ = m(z, a)
        loss, d = rnn_loss(m, parts, batch, pos_weight=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    end = d["nll"].item()

    assert math.isfinite(end), "NLL went non-finite -- check the logsumexp path"
    assert end < start - 3.0, f"barely learned: {start:.2f} -> {end:.2f}"

    # Sanity: a deterministic (tau=0) one-step prediction should be close.
    with torch.no_grad():
        parts, _ = m(z, a)
        pred = m.sample_next(parts, temperature=0.0)
    err = (pred - z_next).abs().mean().item()
    assert err < 0.05, f"one-step error {err:.4f} too large"


# ------------------------------------------------------------------ dataset


def _fake_root(d: Path, E: int = 3, T: int = 10, Z: int = 4) -> Path:
    """A dataset whose mu encodes (episode, timestep) so leaks are detectable."""
    d.mkdir(parents=True, exist_ok=True)
    mu = np.zeros((E, T + 1, Z), np.float32)
    for e in range(E):
        for t in range(T + 1):
            mu[e, t, 0] = e * 100 + t          # unique fingerprint
    np.save(d / "mu.npy", mu)
    np.save(d / "logvar.npy", np.full((E, T + 1, Z), -20.0, np.float32))
    acts = np.arange(E * T).reshape(E, T) % 3
    np.save(d / "actions.npy", acts.astype(np.int8))
    ev = np.zeros((E, T), np.uint8)
    ev[0, 3] = 4          # a paddle contact
    ev[1, 5] = 4 | 1      # contact plus a wall bounce -- masking must isolate bit 4
    np.save(d / "events.npy", ev)
    states = np.zeros((E, T + 1, 6), np.float32)
    states[..., 0] = 0.7   # ball_x
    states[..., 4] = 0.2   # paddle_x
    np.save(d / "states.npy", states)
    (d / "meta.json").write_text(json.dumps({"episodes": E, "steps": T}))
    return d


def test_windows_never_cross_episodes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_root(Path(tmp) / "ds", E=3, T=10)
        L = 4
        ds = LatentSequenceDataset(root, seq_len=L, use_mean=True, seed=0)
        # T - L + 1 = 7 windows per episode, 3 episodes.
        assert len(ds) == 3 * (10 - L + 1)

        for i in range(len(ds)):
            item = ds[i]
            fp = item["z"][:, 0].numpy()
            nxt = item["z_next"][:, 0].numpy()
            ep_ids = np.floor(fp / 100).astype(int)
            assert len(set(ep_ids.tolist())) == 1, "window spans two episodes"
            ts = fp % 100
            assert np.allclose(np.diff(ts), 1.0), "window is not contiguous in time"
            # z_next is the window shifted by exactly one step, same episode.
            assert np.allclose(nxt, fp + 1.0)
            assert np.floor(nxt / 100).astype(int).tolist() == ep_ids.tolist()


def test_onehot_actions_and_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_root(Path(tmp) / "ds", E=3, T=10)
        ds = LatentSequenceDataset(root, seq_len=10, use_mean=True, seed=0)
        raw_actions = np.load(Path(root) / "actions.npy")

        for e in range(3):
            item = ds[e * 1]  # seq_len == T so there is exactly one window / ep
            a = item["a"].numpy()
            assert a.shape == (10, 3)
            assert np.allclose(a.sum(1), 1.0), "not one-hot"
            assert np.array_equal(a.argmax(1), raw_actions[e])
            assert np.array_equal(item["a_idx"].numpy(), raw_actions[e])

        # hit flag isolates bit 4 even when other event bits are set.
        assert ds[0]["hit"].numpy()[3] == 1.0
        assert ds[0]["hit"].numpy().sum() == 1.0
        assert ds[1]["hit"].numpy()[5] == 1.0
        assert ds[1]["hit"].numpy().sum() == 1.0

        # reward = 1 - |ball_x - paddle_x| = 1 - |0.7 - 0.2| = 0.5
        assert abs(float(ds[0]["reward"][0]) - 0.5) < 1e-6

        # pos_weight from a 2 / 30 positive rate.
        assert abs(ds.hit_rate() - 2 / 30) < 1e-9
        assert abs(ds.pos_weight() - (1 - 2 / 30) / (2 / 30)) < 1e-6


def test_sampling_vs_mean() -> None:
    """use_mean=False must give a different z each access; True must not."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_root(Path(tmp) / "ds", E=2, T=10)
        # Override logvar so sampling actually moves.
        np.save(Path(root) / "logvar.npy", np.zeros((2, 11, 4), np.float32))
        ds_s = LatentSequenceDataset(root, seq_len=4, use_mean=False, seed=0)
        ds_m = LatentSequenceDataset(root, seq_len=4, use_mean=True, seed=0)
        assert not torch.allclose(ds_s[0]["z"], ds_s[0]["z"])
        assert torch.allclose(ds_m[0]["z"], ds_m[0]["z"])


def test_multiple_roots_union() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        r1 = _fake_root(Path(tmp) / "a", E=2, T=10)
        r2 = _fake_root(Path(tmp) / "b", E=3, T=10)
        ds = LatentSequenceDataset([r1, r2], seq_len=5, use_mean=True)
        assert len(ds) == (2 + 3) * (10 - 5 + 1)
        assert ds.n_transitions() == (2 + 3) * 10


# ---------------------------------------------------------------- runner


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

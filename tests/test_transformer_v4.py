"""v4: the causal transformer M, and the promise that it is a drop-in.

Two families of claim are pinned here.

**The transformer is a transformer.** Causality (an output at t is unaffected by
anything after t), the equivalence of the incremental and parallel paths (the
one property every open-loop evaluation silently assumes), and the sliding
window actually sliding.

**Nothing else changed.** The heads, the likelihood and the checkpoint format
are shared with ``MDNRNN`` after the v4 refactor, which is only safe if the
LSTM's own behaviour is bit-identical to what it was -- so the K=1 likelihood is
still exactly a Gaussian, the feed-forward control still builds, and every
checkpoint written before v4 still loads as an LSTM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm.rnn import (  # noqa: E402
    LOG_SQRT_2PI, MDNRNN, RNNConfig, load_rnn, rnn_loss, save_rnn,
)
from wm.transformer import TransformerConfig, TransformerDynamics  # noqa: E402


def tiny(**kw) -> TransformerConfig:
    """A model small enough to be exact and fast, with the same shape as v4's."""
    base = dict(z_dim=4, n_actions=3, d_model=16, n_layers=2, n_heads=2,
                context=8, n_gauss=3, dropout=0.0)
    base.update(kw)
    return TransformerConfig(**base)


def make(seed: int = 0, **kw) -> TransformerDynamics:
    torch.manual_seed(seed)
    m = TransformerDynamics(tiny(**kw))
    m.eval()
    return m


def seq(B: int, T: int, cfg: TransformerConfig, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, T, cfg.z_dim, generator=g)
    a = torch.eye(cfg.n_actions)[
        torch.randint(cfg.n_actions, (B, T), generator=g)]
    return z, a


# ------------------------------------------------------------------ causality


def test_causal_mask_blocks_the_future():
    """Changing the inputs at times > t must not move the output at t.

    This is the property that makes a "prediction" a prediction. It is also the
    one that a single transposed mask silently breaks while every loss curve
    keeps looking healthy -- teacher-forced NLL simply becomes trivially low
    because the model can read the answer -- so it is worth an explicit test.
    """
    m = make()
    z, a = seq(2, 6, m.cfg)
    out, _ = m(z, a)
    h = out["h"]

    z2 = z.clone()
    z2[:, 3:] = torch.randn_like(z2[:, 3:]) * 5.0
    a2 = a.clone()
    a2[:, 3:] = a2[:, 3:].flip(-1)
    h2 = m(z2, a2)[0]["h"]

    assert torch.allclose(h[:, :3], h2[:, :3], atol=1e-6)
    # ... and the model is not simply ignoring its input.
    assert not torch.allclose(h[:, 3:], h2[:, 3:], atol=1e-4)


# ------------------------------------------------- step == forward, exactly


@pytest.mark.parametrize("T", [1, 5, 8])
def test_step_matches_forward_within_context(T):
    """The incremental path is the same function as the parallel one.

    Training uses ``forward``; dreaming uses ``step``. If they disagree, every
    open-loop number in v4 is measured on a model that was never trained.
    """
    m = make()
    z, a = seq(3, T, m.cfg, seed=2)
    parts, _ = m(z, a)

    h = m.init_hidden(3)
    outs = []
    for t in range(T):
        p, h = m.step(z[:, t], a[:, t], h)
        outs.append(p["h"][:, 0])
    stepped = torch.stack(outs, 1)
    assert torch.allclose(parts["h"], stepped, atol=1e-5)


def test_step_matches_forward_past_the_context():
    """Beyond ``context``, ``forward`` falls back to looping and stays exact."""
    m = make()
    C = m.cfg.context
    z, a = seq(2, C + 5, m.cfg, seed=3)
    parts, _ = m(z, a)

    h = m.init_hidden(2)
    outs = []
    for t in range(C + 5):
        p, h = m.step(z[:, t], a[:, t], h)
        outs.append(p["h"][:, 0])
    assert torch.allclose(parts["h"], torch.stack(outs, 1), atol=1e-5)


def test_warmup_then_step_matches_one_pass():
    """The mixed regime every dream uses: forward for the warm-up, then step."""
    m = make()
    z, a = seq(2, 7, m.cfg, seed=4)
    full = m(z, a)[0]["h"]

    parts, h = m(z[:, :3], a[:, :3])
    outs = [parts["h"][:, i] for i in range(3)]
    for t in range(3, 7):
        p, h = m.step(z[:, t], a[:, t], h)
        outs.append(p["h"][:, 0])
    assert torch.allclose(full, torch.stack(outs, 1), atol=1e-5)


# ------------------------------------------------------------- the buffer


def test_buffer_truncates_at_context():
    """The carried buffer stops growing, and holds the LAST ``context`` inputs."""
    m = make()
    C = m.cfg.context
    z, a = seq(2, C + 6, m.cfg, seed=5)

    h = m.init_hidden(2)
    lens = []
    for t in range(C + 6):
        _, h = m.step(z[:, t], a[:, t], h)
        lens.append(h[2].shape[1])

    assert lens[:C] == list(range(1, C + 1))
    assert all(n == C for n in lens[C:])
    # The window really is the most recent inputs, not the oldest.
    want = torch.cat([z[:, -C:], a[:, -C:]], dim=-1)
    assert torch.allclose(h[2], want, atol=1e-6)


def test_hidden_tuple_looks_like_an_lstms():
    """``h[0][0]`` and ``h[1][0]`` are (B, hidden), as every call site assumes.

    ``wm.eval_rnn.collect_hidden``, ``wm.dream_env`` and the controller stage all
    reach into the carried state positionally. Keeping the shapes means none of
    them needs to know which model it is holding -- which is the condition the
    v4 brief put on this whole exercise.
    """
    m = make()
    h = m.init_hidden(5)
    assert h[0][0].shape == (5, m.cfg.hidden) == h[1][0].shape
    z, a = seq(5, 3, m.cfg, seed=6)
    parts, h = m(z, a)
    assert h[0][0].shape == (5, m.cfg.hidden) == h[1][0].shape
    # h[0][0] is the last timestep's output, exactly as h_n is for an LSTM.
    assert torch.allclose(h[0][0], parts["h"][:, -1], atol=1e-6)


def test_context_shorter_than_the_sequence_forgets():
    """A ctx-N model's output at t genuinely cannot see time t-N.

    This is what makes ``runs/tf_v4_ctx32`` a control rather than a synonym: if
    changing an input outside the window moved the output, the "transformer that
    cannot see back to the flip" would not be one.
    """
    m = make(context=4)
    z, a = seq(2, 10, m.cfg, seed=7)
    h_full = m(z, a)[0]["h"]
    z2 = z.clone()
    z2[:, :5] = torch.randn_like(z2[:, :5]) * 3.0
    h_pert = m(z2, a)[0]["h"]
    # Output at t=9 attends over inputs 6..9 only.
    assert torch.allclose(h_full[:, 9], h_pert[:, 9], atol=1e-5)
    assert not torch.allclose(h_full[:, 4], h_pert[:, 4], atol=1e-4)


# --------------------------------------------------------- the shared head


def test_k1_nll_is_exactly_a_gaussian():
    """With one mixture component the MDN likelihood must reduce to a Gaussian.

    Run on the transformer because the head is now shared code: this asserts the
    sharing did not break the arithmetic, and it is the same check
    ``tests/test_rnn.py`` makes for the LSTM.
    """
    m = make(n_gauss=1)
    z, a = seq(2, 5, m.cfg, seed=8)
    parts, _ = m(z, a)
    z_next = z + 0.1 * torch.randn_like(z)

    got = m.mdn_nll_per_step(parts, z_next)
    y = m._target_in_model_space(parts, z_next)
    mean = parts["mean"][:, :, 0]
    logstd = parts["logstd"][:, :, 0]
    want = (0.5 * ((y - mean) / logstd.exp()) ** 2 + logstd + LOG_SQRT_2PI).sum(-1)
    assert torch.allclose(got, want, atol=1e-5)


def test_rnn_loss_runs_on_a_transformer():
    """The shared loss takes either model with no branch."""
    m = make()
    z, a = seq(4, 6, m.cfg, seed=9)
    parts, _ = m(z, a)
    batch = {
        "z_next": z + 0.05 * torch.randn_like(z),
        "hit": (torch.rand(4, 6) < 0.1).float(),
        "reward": torch.rand(4, 6),
    }
    loss, d = rnn_loss(m, parts, batch, pos_weight=9.0)
    assert torch.isfinite(loss) and set(d) >= {"nll", "hit_bce", "reward_mse"}


def test_sampling_at_zero_temperature_is_the_top_component():
    m = make()
    z, a = seq(2, 4, m.cfg, seed=10)
    parts, _ = m(z, a)
    assert torch.allclose(m.sample_next(parts, temperature=0.0),
                          m.most_likely_mean(parts))


# ------------------------------------------------------------- checkpoints


def test_transformer_round_trips(tmp_path):
    m = make(seed=11)
    p = save_rnn(tmp_path / "tf.pt", m, {"arch": "transformer"})
    back, cfg = load_rnn(p)
    assert isinstance(back, TransformerDynamics)
    assert cfg.context == m.cfg.context and cfg.d_model == m.cfg.d_model
    z, a = seq(2, 5, m.cfg, seed=12)
    assert torch.allclose(m(z, a)[0]["h"], back(z, a)[0]["h"], atol=1e-6)


def test_lstm_checkpoints_are_untouched(tmp_path):
    """An LSTM saved with no ``arch`` key -- i.e. every pre-v4 checkpoint."""
    torch.manual_seed(13)
    m = MDNRNN(RNNConfig(z_dim=4, hidden=16, n_gauss=3))
    m.eval()
    p = save_rnn(tmp_path / "lstm.pt", m)
    payload = torch.load(p, weights_only=False)
    payload["cfg"].pop("arch")          # simulate a checkpoint from v1..v3.1
    torch.save(payload, p)

    back, cfg = load_rnn(p)
    assert isinstance(back, MDNRNN) and cfg.arch == "lstm"
    z, a = seq(2, 5, m.cfg, seed=14)
    assert torch.allclose(m(z, a)[0]["h"], back(z, a)[0]["h"], atol=1e-6)


def test_feedforward_still_builds_and_loads(tmp_path):
    m = MDNRNN(RNNConfig(z_dim=4, hidden=16, n_gauss=3, feedforward=True))
    m.eval()
    back, cfg = load_rnn(save_rnn(tmp_path / "ff.pt", m))
    assert cfg.feedforward and back.lstm is None
    z, a = seq(2, 5, m.cfg, seed=15)
    assert torch.allclose(m(z, a)[0]["h"], back(z, a)[0]["h"], atol=1e-6)


def test_action_ablation_zeroes_the_action_input():
    m = make(ablate_actions=True)
    z, a = seq(2, 5, m.cfg, seed=16)
    other = torch.eye(m.cfg.n_actions)[torch.randint(m.cfg.n_actions, (2, 5))]
    assert torch.allclose(m(z, a)[0]["h"], m(z, other)[0]["h"], atol=1e-6)


def test_parameter_count_is_in_the_lstms_league():
    """The v4 comparison is only readable if the two models are the same size.

    d_model 96 matches the 327k LSTM almost exactly; the briefed d_model 128 is
    1.7x larger, which is recorded here so the number cannot drift unnoticed.
    """
    n = lambda m: sum(p.numel() for p in m.parameters())
    lstm = n(MDNRNN(RNNConfig()))
    assert 320_000 < lstm < 335_000
    assert 0.95 < n(TransformerDynamics(TransformerConfig(d_model=96))) / lstm < 1.05
    assert 1.5 < n(TransformerDynamics(TransformerConfig())) / lstm < 2.0

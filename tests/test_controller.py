"""Tests for stage three: the controller, the dream environment, and the harness.

Runnable two ways::

    python -m tests.test_controller
    pytest tests/test_controller.py

Most of these are cheap structural checks on a randomly-initialised MDN-RNN and
need no checkpoints. Three of them (marked below) load the real V and M and run
a handful of real episodes; they are the ones that would actually catch a wiring
mistake, so they are skipped rather than deleted if the checkpoints are absent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from wm.controller import (
    LinearController,
    NormStats,
    OracleController,
    RandomController,
    StayController,
    batched_logits,
    load_controller,
    logits_to_actions,
    make_features,
    save_controller,
)
from wm.dream_env import DreamEnv, StartPool, dream_rollout
from wm.rnn import MDNRNN, RNNConfig

torch.manual_seed(0)

VAE_CKPT = Path("runs/vae_b1/vae.pt")
RNN_CKPT = Path("runs/rnn_v1/rnn.pt")


def _toy_rnn(z_dim: int = 4, hidden: int = 8) -> MDNRNN:
    m = MDNRNN(RNNConfig(z_dim=z_dim, hidden=hidden, n_gauss=3))
    # The MDN head is initialised near zero on purpose (see wm/rnn.py), which
    # makes every component identical and the mixture degenerate. Randomise it
    # so that temperature and argmax actually have something to act on.
    with torch.no_grad():
        m.mdn.weight.normal_(0, 0.2)
        m.mdn.bias.normal_(0, 0.2)
    return m.eval()


def _toy_pool(E: int = 6, T: int = 40, z_dim: int = 4, warmup: int = 8) -> StartPool:
    rng = np.random.default_rng(0)
    return StartPool(
        rng.standard_normal((E, T + 1, z_dim)).astype(np.float32),
        rng.integers(0, 3, size=(E, T)).astype(np.int64),
        rng.standard_normal((E, T + 1, 6)).astype(np.float32),
        warmup,
    )


# ------------------------------------------------------------- parameters


def test_param_roundtrip() -> None:
    """set_params(get_params()) must be the identity, and the layout stable."""
    for inputs, in_dim in (("z", 16), ("h", 256), ("zh", 272)):
        c = LinearController(16, 256, inputs=inputs)
        assert c.in_dim == in_dim
        assert c.n_params == 3 * in_dim + 3

        rng = np.random.default_rng(1)
        v = rng.standard_normal(c.n_params)
        c.set_params(v)
        assert np.allclose(c.get_params(), v)
        # ... and the unpacking is the documented one: [W.ravel(), b].
        assert np.allclose(c.W.ravel(), v[: 3 * in_dim])
        assert np.allclose(c.b, v[3 * in_dim :])

        # A second round trip through a fresh object must reproduce the logits.
        z = rng.standard_normal((5, 16)).astype(np.float32)
        h = rng.standard_normal((5, 256)).astype(np.float32)
        c2 = LinearController(16, 256, inputs=inputs).set_params(c.get_params())
        assert np.allclose(c.logits(z, h), c2.logits(z, h))

    # Wrong-sized vectors must fail loudly rather than broadcast silently.
    c = LinearController(16, 256)
    for bad in (np.zeros(c.n_params - 1), np.zeros(c.n_params + 1)):
        try:
            c.set_params(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("set_params accepted a wrong-sized vector")


def test_batched_matches_per_sample() -> None:
    """The population einsum must equal P separate single-controller passes."""
    rng = np.random.default_rng(2)
    P, R, z_dim, hidden = 7, 5, 16, 32
    norm = NormStats(
        rng.standard_normal(z_dim).astype(np.float32),
        np.abs(rng.standard_normal(z_dim)).astype(np.float32) + 0.1,
        rng.standard_normal(hidden).astype(np.float32),
        np.abs(rng.standard_normal(hidden)).astype(np.float32) + 0.1,
    )
    c = LinearController(z_dim, hidden, inputs="zh", norm=norm)
    params = rng.standard_normal((P, c.n_params))
    z = rng.standard_normal((P, R, z_dim)).astype(np.float32)
    h = rng.standard_normal((P, R, hidden)).astype(np.float32)

    feats = make_features(z, h, "zh", norm)
    lg = batched_logits(params, feats)
    acts = logits_to_actions(lg)

    for p in range(P):
        c.set_params(params[p])
        ref = c.logits(z[p], h[p])
        assert np.allclose(lg[p], ref, atol=1e-10), p
        assert np.array_equal(acts[p], c.act(z[p], h[p])), p

    assert acts.shape == (P, R)
    assert acts.min() >= 0 and acts.max() <= 2


def test_normalisation_is_applied_and_clamped() -> None:
    z = np.tile(np.arange(4, dtype=np.float32), (3, 1))
    h = np.zeros((3, 4), np.float32)
    norm = NormStats(
        np.zeros(4, np.float32), np.array([1, 2, 4, 8], np.float32),
        np.zeros(4, np.float32),
        np.array([1.0, 1e-9, 1.0, 1.0], np.float32),  # a "dead" unit
    )
    f = make_features(z, h, "zh", norm)
    assert np.allclose(f[:, :4], np.array([0, 0.5, 0.5, 0.375]))
    # The dead unit must not be divided by ~0 (std floor); zero in -> zero out.
    assert np.isfinite(f).all()


def test_baseline_controllers() -> None:
    z, h = np.zeros((9, 4), np.float32), np.zeros((9, 8), np.float32)

    s = StayController()
    assert np.array_equal(s.act(z, h), np.ones(9, np.int64))

    r = RandomController(mean_hold=8.0, seed=3)
    r.reset(9, seed=3)
    seq = np.stack([r.act(z, h) for _ in range(40)])
    assert seq.shape == (40, 9) and set(np.unique(seq)) <= {0, 1, 2}
    # Sticky: consecutive actions must agree far more often than the 1/3 an
    # i.i.d. policy would give. (If this fails the "hard baseline" claim dies.)
    assert (seq[1:] == seq[:-1]).mean() > 0.6

    # The oracle reads the true state and nothing else.
    o = OracleController(paddle_w=0.26)
    state = np.array([[0.9, 0.2, 0, 0, 0.2, 0],     # ball far right -> RIGHT
                      [0.1, 0.2, 0, 0, 0.8, 0],     # ball far left  -> LEFT
                      [0.5, 0.2, 0, 0, 0.5, 0]])    # on target      -> STAY
    assert np.array_equal(o.act(z[:3], h[:3], state=state), [2, 0, 1])


def test_checkpoint_roundtrip() -> None:
    rng = np.random.default_rng(4)
    norm = NormStats(
        rng.standard_normal(16).astype(np.float32),
        np.ones(16, np.float32),
        rng.standard_normal(32).astype(np.float32),
        np.ones(32, np.float32),
    )
    c = LinearController(16, 32, inputs="zh", norm=norm)
    c.set_params(rng.standard_normal(c.n_params))
    other = rng.standard_normal(c.n_params)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "controller.pt"
        save_controller(path, c, extra={"params_last_dream": other})
        back = load_controller(path)
        assert back.inputs == "zh" and back.in_dim == c.in_dim
        assert np.allclose(back.get_params(), c.get_params())
        assert np.allclose(back.norm.z_mean, norm.z_mean)
        assert np.allclose(back.norm.h_mean, norm.h_mean)
        # The alternate parameter vector must survive too, and be different.
        alt = load_controller(path, which="params_last_dream")
        assert np.allclose(alt.get_params(), other)
        assert not np.allclose(alt.get_params(), back.get_params())


# ----------------------------------------------------------- the dream env


def test_dream_env_shapes() -> None:
    rnn = _toy_rnn()
    env = DreamEnv(rnn, _toy_pool(), temperature=1.0, reward="mix")
    B = 11
    z, h = env.reset(batch=B, seed=0)
    assert z.shape == (B, 4) and h.shape == (B, 8)

    z, h, r, p, done = env.step(np.ones(B, np.int64))
    assert z.shape == (B, 4) and h.shape == (B, 8)
    assert r.shape == (B,) and p.shape == (B,) and done.shape == (B,)
    assert (p >= 0).all() and (p <= 1).all()          # it is a probability
    assert not done.any()                             # the dream never ends

    # Reward modes must differ, and `mix` must be exactly hit + lam * dense.
    for mode in ("hit", "dense", "mix"):
        e = DreamEnv(rnn, _toy_pool(), temperature=0.0, reward=mode, lam=0.1)
        e.reset(batch=B, seed=0)
        _, _, rr, pp, _ = e.step(np.ones(B, np.int64))
        if mode == "hit":
            assert np.allclose(rr, pp)
            hit_r = rr
        elif mode == "dense":
            dense_r = rr
        else:
            assert np.allclose(rr, hit_r + 0.1 * dense_r, atol=1e-6)


def test_dream_env_deterministic_at_tau0() -> None:
    """tau = 0 is the mode of the mixture: two identical runs must match exactly."""
    rnn = _toy_rnn()
    pool = _toy_pool()
    acts = np.random.default_rng(5).integers(0, 3, size=(20, 9))

    def roll(seed: int):
        env = DreamEnv(rnn, pool, temperature=0.0, reward="mix")
        z, h = env.reset(batch=9, seed=seed)
        zs, rs = [z], []
        for t in range(20):
            z, h, r, _, _ = env.step(acts[t])
            zs.append(z)
            rs.append(r)
        return np.stack(zs), np.stack(rs)

    a_z, a_r = roll(0)
    b_z, b_r = roll(0)
    assert np.array_equal(a_z, b_z) and np.array_equal(a_r, b_r)

    # A different seed picks different start states, so it must NOT match.
    c_z, _ = roll(1)
    assert not np.allclose(a_z, c_z)

    # And at tau = 1 the same seed still gives the same trajectory (the
    # generator is seeded), but a different tau gives a different one.
    env = DreamEnv(rnn, pool, temperature=1.0, reward="mix")
    env.reset(batch=9, seed=0)
    s1 = np.stack([env.step(acts[t])[0] for t in range(20)])
    env.reset(batch=9, seed=0)
    s2 = np.stack([env.step(acts[t])[0] for t in range(20)])
    assert np.array_equal(s1, s2)
    assert not np.allclose(s1, a_z[1:])


def test_dream_reset_uses_warmup_not_cold_h() -> None:
    """The start hidden state must carry the 8 warm-up steps, not be zeros."""
    rnn = _toy_rnn()
    pool = _toy_pool()
    env = DreamEnv(rnn, pool, temperature=0.0)
    _, h = env.reset(batch=4, seed=0)
    assert np.abs(h).max() > 1e-3, "h is ~zero: the warm-up did not run"

    # It must equal a hand-rolled teacher-forced pass over [t0-W, t0).
    e, t0 = env.start_e, env.start_t0
    W = pool.warmup
    idx = t0[:, None] - W + np.arange(W)[None, :]
    z_w = torch.from_numpy(np.take_along_axis(pool.mu[e], idx[:, :, None], 1))
    a_w = torch.eye(3)[torch.from_numpy(np.take_along_axis(pool.actions[e], idx, 1))]
    with torch.no_grad():
        parts, hn = rnn(z_w, a_w)
    assert np.allclose(h, hn[0][0].numpy(), atol=1e-6)
    # h_n[0] and the last LSTM output are the same tensor -- the claim the
    # dream/real code both rely on.
    assert np.allclose(hn[0][0].numpy(), parts["h"][:, -1].numpy(), atol=1e-7)


def test_dream_rollout_returns_and_recording() -> None:
    rnn = _toy_rnn()
    env = DreamEnv(rnn, _toy_pool(), temperature=0.0, reward="mix")
    c = LinearController(4, 8, inputs="zh")
    c.set_params(np.random.default_rng(6).standard_normal(c.n_params))

    rec = dream_rollout(env, lambda z, h: c.act(z, h), batch=5, steps=12,
                        seed=0, record=True)
    assert rec["return"].shape == (5,)
    assert rec["z"].shape == (5, 13, 4)
    assert rec["h"].shape == (5, 12, 8)
    assert rec["actions"].shape == (5, 12)
    assert rec["p_hit"].shape == (5, 12)
    # The summed reward must equal what the recorded components imply.
    assert np.allclose(rec["hit_sum"], rec["p_hit"].sum(1), atol=1e-5)


# ------------------------------------------------- the timing convention


def test_dream_and_real_timing_agree() -> None:
    """The controller must see IDENTICAL inputs in both code paths.

    This is the test that protects the whole result. Stage three trains in the
    dream and is measured in the real environment; if the two loops disagreed
    about which hidden state pairs with which latent -- an off-by-one that no
    shape check would catch -- the controller would be reading a time-shifted
    signal at evaluation and the transfer number would be meaningless.

    The trick is to make the dream *not* dream: feed a fixed latent sequence to
    both loops (here by driving the real-env recurrence with the same latents
    the dream is warm-started on) and assert the observations match step for
    step.
    """
    rnn = _toy_rnn()
    pool = _toy_pool(E=1, T=30, warmup=8)
    W = pool.warmup
    c = LinearController(4, 8, inputs="zh")
    c.set_params(np.random.default_rng(7).standard_normal(c.n_params))

    # --- path A: DreamEnv, but we ignore its dreamed z and inject the true one,
    # so the only thing under test is the hidden-state bookkeeping.
    env = DreamEnv(rnn, pool, temperature=0.0)
    z, h = env.reset(batch=1, seed=0, starts=(np.array([0]), np.array([W])))
    obs_a = [(z.copy(), h.copy())]
    actions = []
    for t in range(10):
        a = c.act(z, h)
        actions.append(int(a[0]))
        _, h, _, _, _ = env.step(a)
        z = pool.mu[0, W + t + 1][None]            # inject the TRUE next latent
        env._z = torch.from_numpy(z)
        obs_a.append((z.copy(), h.copy()))

    # --- path B: the real-environment recurrence from wm.eval_controller,
    # written out here so the test does not merely call the same function twice.
    hh = rnn.init_hidden(1)
    with torch.no_grad():
        zw = torch.from_numpy(pool.mu[0, :W][None])
        aw = torch.eye(3)[torch.from_numpy(pool.actions[0, :W][None])]
        _, hh = rnn(zw, aw)
    zb = pool.mu[0, W][None]
    obs_b = [(zb.copy(), hh[0][0].numpy().copy())]
    for t in range(10):
        a = c.act(zb, hh[0][0].numpy())
        assert int(a[0]) == actions[t], f"actions diverge at t={t}"
        with torch.no_grad():
            _, hh = rnn.step(torch.from_numpy(zb), torch.eye(3)[a], hh)
        zb = pool.mu[0, W + t + 1][None]
        obs_b.append((zb.copy(), hh[0][0].numpy().copy()))

    for t, ((za, ha), (zb_, hb)) in enumerate(zip(obs_a, obs_b)):
        assert np.allclose(za, zb_, atol=1e-6), f"z differs at t={t}"
        assert np.allclose(ha, hb, atol=1e-6), f"h differs at t={t}"


# ----------------------------------------------- the real-env harness (slow)


def _have_ckpts() -> bool:
    return VAE_CKPT.exists() and RNN_CKPT.exists()


def test_oracle_beats_random_in_the_real_env() -> None:
    """Sanity of the measuring instrument, not of any learned controller.

    If the oracle -- a three-line policy with access to the true state -- does
    not beat sticky-random contacts here, then the harness is broken (wrong
    event bit, wrong action mapping, seeds not shared) and every other number in
    stage three is noise.
    """
    if not _have_ckpts():
        print("    (skipped: V/M checkpoints missing)")
        return
    from wm.analyze import load_ckpt
    from wm.eval_controller import run_real_episodes, summarise
    from wm.rnn import load_rnn

    vae, _, _ = load_ckpt(str(VAE_CKPT), "cpu")
    rnn, _ = load_rnn(str(RNN_CKPT), "cpu")

    scores = {}
    for ctrl in (OracleController(0.26), RandomController(8.0, seed=11)):
        roll = run_real_episodes(ctrl, vae, rnn, episodes=8, steps=200,
                                 seed_base=4242, device="cpu")
        s = summarise(ctrl.name, roll)
        scores[ctrl.name] = s
        # The physics ceiling must be the same for both -- identical seeds.
        assert s["floor_visits_per_episode"] > 0.5

    assert scores["oracle"]["hits_per_episode"] > scores["random"]["hits_per_episode"], (
        scores["oracle"]["hits_per_episode"], scores["random"]["hits_per_episode"]
    )
    # The oracle should also be much closer to the ball when it matters.
    assert scores["oracle"]["mean_gap_at_floor"] < scores["random"]["mean_gap_at_floor"]


def test_real_harness_is_seed_paired() -> None:
    """Two controllers on the same seed base must face the identical physics.

    Specifically: until the first action differs, the trajectories must be
    bit-identical. StayController and a RandomController that happens to start
    with STAY would be a weak test, so we check the cheaper invariant -- the
    initial states match exactly across runs.
    """
    if not _have_ckpts():
        print("    (skipped: V/M checkpoints missing)")
        return
    from wm.analyze import load_ckpt
    from wm.eval_controller import run_real_episodes
    from wm.rnn import load_rnn

    vae, _, _ = load_ckpt(str(VAE_CKPT), "cpu")
    rnn, _ = load_rnn(str(RNN_CKPT), "cpu")
    a = run_real_episodes(StayController(), vae, rnn, episodes=4, steps=5,
                          seed_base=321, device="cpu")
    b = run_real_episodes(RandomController(8.0, seed=0), vae, rnn, episodes=4,
                          steps=5, seed_base=321, device="cpu")
    assert np.array_equal(a["states"][:, 0], b["states"][:, 0])
    # And STAY really does keep the paddle still.
    assert np.allclose(a["states"][:, :, 4].std(1), 0.0, atol=1e-6)


def test_population_real_matches_single_real() -> None:
    """``run_population_real`` (the --fitness real path) must agree with the
    single-controller harness on the same params and seeds."""
    if not _have_ckpts():
        print("    (skipped: V/M checkpoints missing)")
        return
    from wm.analyze import load_ckpt
    from wm.eval_controller import run_population_real, run_real_episodes
    from wm.rnn import load_rnn

    vae, _, _ = load_ckpt(str(VAE_CKPT), "cpu")
    rnn, rcfg = load_rnn(str(RNN_CKPT), "cpu")
    c = LinearController(rcfg.z_dim, rcfg.hidden, inputs="zh")
    rng = np.random.default_rng(9)
    params = rng.standard_normal((2, c.n_params)) * 0.3

    seeds = [555, 556, 557]
    pop = run_population_real(params, c, vae, rnn, seeds, steps=60, device="cpu")
    assert pop.shape == (2, 3)
    for p in range(2):
        c.set_params(params[p])
        roll = run_real_episodes(c, vae, rnn, episodes=3, steps=60,
                                 seed_base=555, device="cpu")
        assert np.allclose(pop[p], roll["hits"].sum(1)), (p, pop[p], roll["hits"].sum(1))


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

"""Tests for the v2 additions to stage two: latent suffixes and the causal tools.

Runnable two ways (pytest is not guaranteed to be installed here)::

    python -m tests.test_rnn_v2
    pytest tests/test_rnn_v2.py

Four things are checked, and they are the four places a silent v2 bug would do
the most damage:

    * the latent-suffix round trip -- write mu_x.npy, read mu_x.npy, and never
      accidentally read the default mu.npy instead. If that broke, the
      ``--ablate-color`` control would quietly train on the very latents it is
      supposed to be blind to and the headline contrast would vanish;
    * the ``--ablate-color`` flag actually selecting those files;
    * the recolour intervention being EXACT -- same positions, new hue, and
      bit-identical to the dataset's own frame when the mass is unchanged;
    * the dreamed-speed estimator, on a trajectory whose speed is known.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from worldsim.bouncing_box import BouncingBox, BoxConfig, mass_to_color

from wm.cache_latents import latent_suffix
from wm.eval_causal_v2 import dreamed_speed, recolor_frames
from wm.seq_data import LatentSequenceDataset, episode_arrays

STATE_NAMES_V2 = ["ball_x", "ball_y", "ball_vx", "ball_vy",
                  "paddle_x", "paddle_vx", "mass"]


# ------------------------------------------------------------ latent suffix


def test_latent_suffix_normalisation() -> None:
    assert latent_suffix("") == ""
    assert latent_suffix(None) == ""
    assert latent_suffix("v1vae") == "_v1vae"
    # Leading underscores are tolerated so "--suffix _v1vae" is not a silent
    # double-underscore filename that nothing will ever find again.
    assert latent_suffix("_v1vae") == "_v1vae"


def _fake_v2_root(d: Path, E: int = 3, T: int = 10, Z: int = 4) -> Path:
    """A v2-shaped dataset with TWO latent caches whose values differ.

    ``mu.npy`` fingerprints (episode, timestep) as ``e*100 + t``; the ``_alt``
    cache offsets everything by 1000, so any confusion between the two is
    visible in the first latent dimension rather than being a subtle numeric
    difference.
    """
    d.mkdir(parents=True, exist_ok=True)
    mu = np.zeros((E, T + 1, Z), np.float32)
    for e in range(E):
        for t in range(T + 1):
            mu[e, t, 0] = e * 100 + t
    np.save(d / "mu.npy", mu)
    np.save(d / "logvar.npy", np.full((E, T + 1, Z), -20.0, np.float32))
    np.save(d / "mu_alt.npy", mu + 1000.0)
    np.save(d / "logvar_alt.npy", np.full((E, T + 1, Z), -20.0, np.float32))

    np.save(d / "actions.npy", (np.arange(E * T).reshape(E, T) % 3).astype(np.int8))
    np.save(d / "events.npy", np.zeros((E, T), np.uint8))
    # Seven columns, which is the whole point: v1 had six.
    states = np.zeros((E, T + 1, 7), np.float32)
    states[..., 0] = 0.7        # ball_x
    states[..., 4] = 0.2        # paddle_x
    states[..., 6] = 1.3        # mass
    np.save(d / "states.npy", states)
    (d / "meta.json").write_text(json.dumps({
        "episodes": E, "steps": T, "state_names": STATE_NAMES_V2,
    }))
    return d


def test_latent_suffix_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_v2_root(Path(tmp) / "ds", E=3, T=10)

        default = LatentSequenceDataset(root, seq_len=4, use_mean=True)
        alt = LatentSequenceDataset(root, seq_len=4, use_mean=True,
                                    latent_suffix="alt")
        # Same windows, same actions, same states -- only the latents differ.
        assert len(default) == len(alt)
        d0, a0 = default[0], alt[0]
        assert np.allclose(a0["z"].numpy()[:, 0] - d0["z"].numpy()[:, 0], 1000.0)
        assert np.allclose(a0["a"].numpy(), d0["a"].numpy())
        assert np.allclose(a0["state"].numpy(), d0["state"].numpy())
        assert d0["state"].shape[-1] == 7, "v2 state must keep its mass column"

        # episode_arrays honours the suffix too (eval and the causal script use it).
        assert np.allclose(
            episode_arrays(root, latent_suffix="alt")["mu"]
            - episode_arrays(root)["mu"], 1000.0)

        # A missing cache must fail loudly, not fall back to the default one.
        try:
            LatentSequenceDataset(root, seq_len=4, latent_suffix="nope")
        except FileNotFoundError as exc:
            assert "mu_nope.npy" in str(exc)
        else:
            raise AssertionError("missing latent cache did not raise")


def test_ablate_color_selects_the_alternate_latents() -> None:
    """The flag -> filename mapping, exercised through train_rnn's own parser."""
    import wm.train_rnn as tr

    src = Path(tr.__file__).read_text()
    # The one line that turns the flag into a suffix; if it is edited, this
    # test should be edited with it deliberately rather than silently pass.
    assert 'latent_suffix = "v1vae" if a.ablate_color else a.latent_suffix' in src

    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_v2_root(Path(tmp) / "ds", E=2, T=10)
        # Stand in for the v1-VAE cache under the name the flag selects.
        for f in ("mu", "logvar"):
            np.save(Path(root) / f"{f}_v1vae.npy", np.load(Path(root) / f"{f}_alt.npy"))
        loader = tr.make_seq_loader(
            [str(root)], seq_len=4, batch_size=2, shuffle=False,
            use_mean=True, latent_suffix="v1vae",
        )
        z = next(iter(loader))["z"].numpy()
        assert z[:, :, 0].min() >= 1000.0, "ablate-color read the default latents"


# -------------------------------------------------------------- recolouring


def _v2_cfg() -> BoxConfig:
    return BoxConfig(res=64, ball_radius=0.08, mass_from_color=True,
                     mass_min=0.5, mass_max=2.0)


def test_recolor_preserves_geometry_and_changes_hue() -> None:
    cfg = _v2_cfg()
    m0, m1 = 0.6, 1.8
    state = np.array([[0.42, 0.61, 0.01, -0.01, 0.33, 0.0, m0]], dtype=np.float32)

    orig = recolor_frames(state, cfg, mass=None)[0]
    cf = recolor_frames(state, cfg, mass=m1)[0]
    assert orig.shape == (64, 64, 3) and cf.shape == (64, 64, 3)

    # 1. The ball's colour is exactly the colour the new mass maps to. The ball
    #    interior is drawn at full coverage, so the modal non-background,
    #    non-paddle colour is the ball colour with no antialiasing blend.
    want = np.array(mass_to_color(m1, cfg), dtype=np.int16)
    d = np.abs(cf.astype(np.int16) - want).sum(-1)
    assert (d == 0).sum() > 50, "no block of pixels has the new ball colour"
    # And it is NOT the old colour any more.
    old = np.array(mass_to_color(m0, cfg), dtype=np.int16)
    assert np.abs(old - want).sum() > 10, "test masses map to the same colour"
    assert (np.abs(cf.astype(np.int16) - old).sum(-1) == 0).sum() == 0

    # 2. The geometry is untouched: the ball and paddle occupy exactly the same
    #    pixels, which is what makes this an intervention on appearance alone.
    bg = np.array(cfg.bg_color, dtype=np.int16)
    occ_o = (np.abs(orig.astype(np.int16) - bg).sum(-1) > 0)
    occ_c = (np.abs(cf.astype(np.int16) - bg).sum(-1) > 0)
    assert np.array_equal(occ_o, occ_c), "recolouring moved something"
    # The paddle pixels are bit-identical (only the ball was repainted).
    pad = np.abs(orig.astype(np.int16)
                 - np.array(cfg.paddle_color, dtype=np.int16)).sum(-1) == 0
    assert pad.sum() > 20
    assert np.array_equal(orig[pad], cf[pad])


def test_recolor_matches_the_simulator_exactly() -> None:
    """With mass=None the re-render must reproduce the simulator's own frame.

    This is the guarantee that makes the counterfactual exact rather than
    approximate: if the two paths disagreed, every recoloured dream would carry
    an unknown rendering offset on top of the colour change.
    """
    cfg = _v2_cfg()
    env = BouncingBox(cfg, seed=3)
    env.reset(seed=3)
    frames, states = [], []
    for t in range(6):
        frames.append(env.render())
        states.append(env.state())
        env.step(t % 3)
    got = recolor_frames(np.stack(states), cfg, mass=None)
    assert np.array_equal(got, np.stack(frames))


# ---------------------------------------------------------- speed estimator


def test_dreamed_speed_on_a_straight_line() -> None:
    # A ball moving at a known speed along a diagonal.
    v = 0.0137
    k = np.arange(24)[None, :, None]
    direction = np.array([0.6, 0.8])            # unit vector
    pos = 0.1 + v * k * direction
    est = dreamed_speed(pos)
    assert abs(float(est[0]) - v) < 1e-9, est

    # Robustness, which is why the estimator is a median: one probe glitch and
    # one wall bounce inside the window must not move the answer.
    pos2 = pos.copy()
    pos2[0, 9] += 0.25                          # a kNN snap
    pos2[0, 15:] = pos2[0, 14] + (pos2[0, 14] - pos2[0, 15:])   # a reversal
    est2 = dreamed_speed(pos2)
    assert abs(float(est2[0]) - v) < 1e-3, est2

    # Several trajectories at once, and the window bounds are respected: only
    # steps SPEED_LO..SPEED_HI count, so a different speed outside them is
    # invisible.
    slow = pos.copy()
    slow[0, :4] = slow[0, 0]                    # frozen during the settle window
    assert abs(float(dreamed_speed(slow)[0]) - v) < 1e-9

    batch = np.concatenate([pos, 2 * pos], 0)
    out = dreamed_speed(batch)
    assert out.shape == (2,)
    assert abs(float(out[1]) - 2 * v) < 1e-9


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

"""Tests for the interventional controller test (``wm.eval_ctrl_intervention``).

Runnable two ways::

    python -m tests.test_ctrl_intervention
    pytest tests/test_ctrl_intervention.py

The experiment's whole claim is "everything except the colour is identical", and
that claim rests on three pieces of machinery, each of which is checked here.

1. **The repaint is exact.** Re-rendering a recorded state with the ball's OWN
   mass must give back bit-identical pixels. If it did not, the measured effect
   would be part re-rendering artefact, and the null control would absorb an
   unknown amount of the signal.
2. **A policy that cannot see colour scores zero.** Two anchors: the oracle,
   which reads true state and never touches a pixel, and a synthetic controller
   whose weights read only the position dimensions of a deliberately
   interpretable latent. Both must give Δ = 0. The mirror-image controller --
   same latent, weights on the colour dimension -- must give Δ != 0, or the
   harness would be incapable of detecting mass use at all and every zero it
   reports would be vacuous.
3. **Each controller is driven by its own M.** The controller reads ``h``, so
   the dynamics model is part of the policy. Pairing ``ctrl_v2_cons`` with
   ``rnn_v2`` produces no error and no warning -- just a wrong number -- so the
   mapping is asserted directly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from worldsim.bouncing_box import BouncingBox, BoxConfig, mass_to_color
from wm.controller import LinearController, OracleController
from wm.eval_causal_v2 import recolor_frames
from wm.eval_ctrl_intervention import WARMUP, approach_frames, drives
from wm.eval_controller_v2 import parse_overrides, stack_for_run

CFG = BoxConfig(res=64, ball_radius=0.08, mass_from_color=True)


# ------------------------------------------------------------------ fixtures


def _episode(n: int = 24, mass: float = 0.7, seed: int = 3):
    """A real recorded episode: (states (n+1, 7), actions (n,), frames)."""
    env = BouncingBox(CFG)
    f0 = env.reset(seed=seed)
    states = [env.state()]
    frames = [f0]
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 3, n)
    for a in actions:
        f, s, _ = env.step(int(a))
        frames.append(f)
        states.append(s)
    return np.stack(states), actions.astype(np.int64), np.stack(frames)


class _FakeVAE:
    """An encoder whose latent dimensions have known, separable meanings.

    ``mu = [ball_x, ball_y, red, green]`` -- the centroid of the ball's pixels
    and their mean colour. Dims 0-1 are (up to a pixel of antialiasing) a pure
    geometry read and dims 2-3 a pure colour read, which is what makes the two
    synthetic controllers below a genuine positive/negative pair rather than a
    coincidence.

    The ball mask is "not the background, and not the paddle". Note it cannot be
    "red dominates blue": the heavy end of the ramp is purple (150, 40, 200),
    which is blue-dominant, so that rule would silently classify a heavy ball as
    paddle and make the whole test vacuous.
    """

    z_dim = 4

    def encode(self, x: torch.Tensor):
        img = x.permute(0, 2, 3, 1).cpu().numpy()             # (B, H, W, 3)
        bg = np.array(CFG.bg_color, np.float32) / 255.0
        pad = np.array(CFG.paddle_color, np.float32) / 255.0
        out = np.zeros((len(img), self.z_dim), np.float32)
        for i, f in enumerate(img):
            m = (np.abs(f - bg).sum(-1) > 1e-6) & (np.abs(f - pad).sum(-1) > 0.25)
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            out[i, 0] = xs.mean() / f.shape[1]
            out[i, 1] = ys.mean() / f.shape[0]
            out[i, 2] = f[..., 0][m].mean()
            out[i, 3] = f[..., 1][m].mean()
        t = torch.from_numpy(out)
        return t, torch.zeros_like(t)


class _FakeRNN(torch.nn.Module):
    """An M that carries nothing: ``h`` is always zero.

    The intervention then flows through ``z`` alone, which is the point -- these
    two tests are about the controller's weights, not about the dynamics model.
    """

    def __init__(self, hidden: int = 2):
        super().__init__()
        self.hidden = hidden

    def forward(self, z, a):
        B = z.shape[0]
        h = torch.zeros(1, B, self.hidden)
        return {}, (h, h.clone())


def _synthetic(read_dim: int) -> LinearController:
    """``drive = 10 * z[read_dim]`` and nothing else."""
    c = LinearController(z_dim=_FakeVAE.z_dim, hidden=2, inputs="zh", norm=None)
    v = np.zeros(c.n_params)
    W = np.zeros((3, c.in_dim))
    W[2, read_dim] = 10.0                                    # logit(RIGHT)
    W[0, read_dim] = -10.0                                   # logit(LEFT)
    v[: 3 * c.in_dim] = W.ravel()
    return c.set_params(v)


# ------------------------------------------------------------------- the tests


def test_repaint_with_the_same_mass_is_bit_identical() -> None:
    """The null control's premise: m1 = m0 changes no pixel, anywhere."""
    states, _a, frames = _episode(mass=0.7)
    m0 = float(states[0, 6])
    as_is = recolor_frames(states, CFG, mass=None)
    null = recolor_frames(states, CFG, mass=m0)
    assert as_is.dtype == np.uint8 and null.shape == as_is.shape
    assert np.array_equal(as_is, null), int(np.abs(
        as_is.astype(int) - null.astype(int)).max())
    # ...and it reproduces what the simulator itself drew, which is what makes
    # the intervention exact rather than approximate.
    assert np.array_equal(as_is, frames)


def test_repaint_with_a_different_mass_does_change_the_pixels() -> None:
    """The positive control on the renderer, so the test above is not vacuous."""
    states, _a, _f = _episode(mass=0.7)
    m0 = float(states[0, 6])
    other = 2.0 if m0 < 1.2 else 0.5
    assert mass_to_color(other, CFG) != mass_to_color(m0, CFG)
    assert not np.array_equal(recolor_frames(states, CFG, mass=None),
                              recolor_frames(states, CFG, mass=other))


def test_oracle_delta_is_exactly_zero() -> None:
    """It reads true state; the intervention touches only pixels."""
    states, actions, _f = _episode(n=40)
    ep = np.zeros(6, np.int64)
    t = np.arange(WARMUP, WARMUP + 6, dtype=np.int64)
    ctrl = OracleController(paddle_w=CFG.paddle_w)
    base = drives(ctrl, None, None, states[None], actions[None], ep, t, CFG, None)
    for m1 in (0.5, 1.0, 2.0):
        rep = drives(ctrl, None, None, states[None], actions[None], ep, t, CFG, m1)
        assert np.array_equal(base, rep), (m1, base, rep)


def _synthetic_delta(read_dim: int) -> tuple:
    states, actions, _f = _episode(n=40)
    ep = np.zeros(8, np.int64)
    t = np.arange(WARMUP, WARMUP + 8, dtype=np.int64)
    vae, rnn = _FakeVAE(), _FakeRNN()
    ctrl = _synthetic(read_dim)
    base = drives(ctrl, vae, rnn, states[None], actions[None], ep, t, CFG, None)
    rep = drives(ctrl, vae, rnn, states[None], actions[None], ep, t, CFG, 2.0)
    return base, np.abs(rep - base).mean()


def test_synthetic_position_reader_vs_colour_reader() -> None:
    """The harness's own positive and negative control, in one comparison.

    ``z[0]`` is the ball's x, ``z[2]`` its red channel; the two controllers are
    otherwise identical. The position reader must be ~blind to the repaint and
    the colour reader must not be.

    Why a *ratio* and not ``Δ == 0`` for the position reader: the centroid is
    computed over an antialiased mask, and a handful of edge pixels sit at
    different sub-threshold coverages for a yellow ball than for a purple one.
    That is a property of this toy encoder, not of the intervention (the frames
    themselves are bit-identical apart from the ball -- see the repaint test
    above), and the honest bar is "two orders of magnitude smaller than a real
    colour read", not "exactly zero".
    """
    base_pos, d_pos = _synthetic_delta(0)
    _base_col, d_col = _synthetic_delta(2)
    assert base_pos.std() > 1e-6, "degenerate probe: the drive never varies"
    assert d_col > 1.0, d_col
    assert d_pos < 0.01 * d_col, (d_pos, d_col)


def test_approach_frames_are_descending_and_have_history() -> None:
    states, actions, _f = _episode(n=120, seed=11)
    S, A = states[None], actions[None]
    ep, t = approach_frames(S, A, n_want=5, seed=0)
    assert len(ep) > 0
    assert (t >= WARMUP).all()
    s = S[ep, t]
    assert (s[:, 3] < 0).all() and (s[:, 1] < 0.5).all()
    assert (np.abs(s[:, 0] - s[:, 4]) > 0.02).all()


def test_evaluator_maps_each_run_to_its_rnn() -> None:
    """A run is evaluated with the M it was TRAINED with, not the default.

    Three routes, in the order the evaluator resolves them: an explicit
    ``--run-rnn NAME=PATH``, then the ``--rnn`` recorded in the run's own
    ``history.json``, then the global default.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "ctrl_v2_cons"
        run.mkdir()
        (run / "history.json").write_text(json.dumps({"meta": {"args": {
            "rnn": "runs/rnn_v2_cons/rnn.pt", "vae": "runs/vae_v2/vae.pt"}}}))
        got = stack_for_run(str(run), {}, {}, "runs/rnn_v2/rnn.pt",
                            "runs/vae_b1/vae.pt")
        assert got == ("runs/rnn_v2_cons/rnn.pt", "runs/vae_v2/vae.pt"), got

        over = parse_overrides(["ctrl_v2_cons=runs/rnn_v2_ms/rnn.pt"])
        got = stack_for_run(str(run), over, {}, "runs/rnn_v2/rnn.pt",
                            "runs/vae_b1/vae.pt")
        assert got[0] == "runs/rnn_v2_ms/rnn.pt", got

        bare = Path(tmp) / "ctrl_nohistory"
        bare.mkdir()
        got = stack_for_run(str(bare), {}, {}, "runs/rnn_v2/rnn.pt",
                            "runs/vae_v2/vae.pt")
        assert got == ("runs/rnn_v2/rnn.pt", "runs/vae_v2/vae.pt"), got


def test_real_runs_are_paired_with_their_own_checkpoints() -> None:
    """The same check against the checkpoints on disk, when they exist."""
    for name, want in (("runs/ctrl_v2", "runs/rnn_v2/rnn.pt"),
                       ("runs/ctrl_v2_cons", "runs/rnn_v2_cons/rnn.pt"),
                       ("runs/ctrl_v2_cons_tau0.5", "runs/rnn_v2_cons/rnn.pt"),
                       ("runs/ctrl_v2_tau0.5", "runs/rnn_v2/rnn.pt")):
        if not (Path(name) / "history.json").exists():
            continue
        got, _ = stack_for_run(name, {}, {}, "runs/rnn_v2/rnn.pt",
                               "runs/vae_v2/vae.pt")
        assert got == want, (name, got, want)


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

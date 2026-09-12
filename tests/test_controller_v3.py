"""v3 stage three: the occlusion bookkeeping, and the pieces that can be
silently wrong.

Every number in `README_C3.md` is a function of four things that no loss curve
would catch if they were broken:

1. where the "required move" is measured (at the frame the ball VANISHES, not
   at the frame it lands),
2. what counts as the paddle moving toward the landing point while blind,
3. that the wait-and-see reference is genuinely blind during an occlusion --
   if it peeked, the memoryless bound would be the ceiling and the whole
   comparison would collapse,
4. that the feed-forward M -- the memory FLOOR -- actually runs through the
   dream and the real harness, and that each controller is evaluated with the
   dynamics model it was trained in.

So those four are the tests, on hand-built inputs whose right answer can be
worked out on paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wm.controller import WaitAndSeeOracleController
from wm.dream_env import DreamEnv, StartPool, dream_rollout
from wm.eval_controller import floor_visit_stats, run_real_episodes
from wm.eval_controller_v2 import parse_overrides, stack_for_run
from wm.eval_controller_v3 import (
    MOVE_BINS,
    annotate_occlusion,
    floor_visits,
    occlusion_table,
    required_move_bin,
    summarise_rollout,
    visible_column,
)
from wm.rnn import MDNRNN, RNNConfig

LEFT, STAY, RIGHT = 0, 1, 2
BAND = (0.36, 0.50)


# --------------------------------------------------------------- a trajectory


def build_episode(
    ball_x: float = 0.8,
    paddle_x0: float = 0.2,
    paddle_step: float = 0.0,
    action: int = STAY,
    n: int = 45,
):
    """One descending ball, hidden for exactly eight frames. Answers on paper.

    ``y`` falls by 0.02 a frame from 0.9, so the ball is inside ``BAND`` for
    frames 20-27 and crosses the floor-visit threshold (0.147) at frame 38. The
    paddle starts at ``paddle_x0`` and, from the frame the ball vanishes,
    marches at ``paddle_step`` per frame while taking ``action``.

    Returns ``(states, actions, hits)`` shaped exactly like a rollout of one
    episode, so the production functions are exercised on their real input type.
    """
    t = np.arange(n, dtype=np.float64)
    y = np.maximum(0.9 - 0.02 * t, 0.02)
    vis = np.where((y >= BAND[0]) & (y <= BAND[1]), 0.0, 1.0)
    hide = int(np.flatnonzero(vis < 0.5)[0])

    paddle = np.full(n, paddle_x0)
    moving = np.clip(t - hide, 0, None) * paddle_step
    paddle = paddle + moving

    s = np.zeros((1, n, 7))
    s[0, :, 0] = ball_x
    s[0, :, 1] = y
    s[0, :, 2] = 0.0
    s[0, :, 3] = -0.02                    # descending the whole way
    s[0, :, 4] = paddle
    s[0, :, 5] = paddle_step
    s[0, :, 6] = vis
    a = np.full((1, n - 1), action, np.int64)
    hits = np.zeros((1, n - 1), np.float32)
    return s, a, hits


def test_visible_column_is_the_last_one_and_defaults_to_visible() -> None:
    s, _, _ = build_episode()
    assert visible_column(s).shape == (1, 45)
    assert visible_column(s)[0, 0] == 1.0 and visible_column(s)[0, 22] == 0.0
    # A v1/v2-shaped state has no band, so nothing is ever hidden.
    assert visible_column(s[..., :6]).min() == 1.0


def test_floor_visits_agree_with_the_v1_visit_count() -> None:
    """The two implementations must count the same chances, or no table lines up."""
    rng = np.random.default_rng(0)
    s = rng.random((5, 60, 7))
    s[..., 1] = np.abs(np.sin(np.linspace(0, 9, 60)))[None, :] * 0.9
    n_here = np.zeros(5)
    for v in floor_visits(s):
        n_here[v["episode"]] += 1
    assert np.array_equal(n_here, floor_visit_stats(s)["floor_visits"])


# --------------------------------------------------------- the required move


def test_required_move_is_measured_when_the_ball_vanishes() -> None:
    s, a, _ = build_episode(ball_x=0.8, paddle_x0=0.2)
    v = annotate_occlusion(floor_visits(s)[0], s, a)
    assert v["hidden"] is True
    assert v["hidden_frames"] == 8
    assert v["t_hide"] == 20 and v["t_emerge"] == 27
    # The paddle stood at 0.2 and the ball came down at 0.8.
    assert v["required_move"] == pytest.approx(0.6, abs=1e-9)
    assert required_move_bin(v["required_move"]) == MOVE_BINS.index("long")

    # Same ball, paddle already underneath it: a short move, not a long one.
    s2, a2, _ = build_episode(ball_x=0.8, paddle_x0=0.75)
    v2 = annotate_occlusion(floor_visits(s2)[0], s2, a2)
    assert v2["required_move"] == pytest.approx(0.05, abs=1e-9)
    assert required_move_bin(v2["required_move"]) == MOVE_BINS.index("short")


def test_bin_edges() -> None:
    assert required_move_bin(0.149) == 0
    assert required_move_bin(0.15) == 1 and required_move_bin(0.349) == 1
    assert required_move_bin(0.351) == 2


def test_a_visit_with_no_occlusion_on_the_descent_is_excluded() -> None:
    s, a, _ = build_episode()
    s[..., 6] = 1.0                       # never hidden
    v = annotate_occlusion(floor_visits(s)[0], s, a)
    assert v["hidden"] is False and "required_move" not in v


# ------------------------------------------------- paddle motion in the dark


def test_paddle_motion_all_toward_is_one_and_all_stay_is_zero() -> None:
    """The defining calibration of the occlusion statistic.

    The ball lands at 0.8 and the paddle sits at 0.2, so RIGHT is toward and
    LEFT is away. Chance for a uniform random policy over three actions is 1/3,
    which is only a meaningful reference because STAY scores 0.
    """
    s, a, h = build_episode(action=RIGHT, paddle_step=0.03)
    row = summarise_rollout("all_toward", {"states": s, "actions": a, "hits": h})
    assert row["occlusion"]["toward_fraction"] == pytest.approx(1.0)
    # Eight hidden frames at 0.03 each, against a required move of 0.6.
    assert row["occlusion"]["displacement_fraction"] == pytest.approx(
        8 * 0.03 / 0.6, abs=1e-6
    )

    s, a, h = build_episode(action=STAY)
    row = summarise_rollout("all_stay", {"states": s, "actions": a, "hits": h})
    assert row["occlusion"]["toward_fraction"] == pytest.approx(0.0)
    assert row["occlusion"]["displacement_fraction"] == pytest.approx(0.0)

    # Moving the WRONG way is worse than standing still, and the displacement
    # fraction says so with a sign.
    s, a, h = build_episode(action=LEFT, paddle_step=-0.03)
    row = summarise_rollout("all_away", {"states": s, "actions": a, "hits": h})
    assert row["occlusion"]["toward_fraction"] == pytest.approx(0.0)
    assert row["occlusion"]["displacement_fraction"] < 0


def test_interceptions_land_in_the_right_required_move_bin() -> None:
    s, a, h = build_episode(ball_x=0.8, paddle_x0=0.2)
    h[0, 40] = 1.0                        # a contact inside the floor visit
    row = summarise_rollout("c", {"states": s, "actions": a, "hits": h})
    b = row["by_required_move"]
    assert b["long"]["n_visits"] == 1
    assert b["long"]["interceptions_per_visit"] == pytest.approx(1.0)
    assert b["short"]["n_visits"] == 0 and b["medium"]["n_visits"] == 0
    assert np.isnan(b["short"]["interceptions_per_visit"])


def test_occlusion_table_is_one_record_per_visit() -> None:
    s, a, h = build_episode()
    tbl = occlusion_table({"states": s, "actions": a, "hits": h})
    assert len(tbl) == 1 and tbl[0]["interceptions"] == 0


# ------------------------------------------------------- the wait-and-see ref


def test_wait_and_see_stays_while_the_ball_is_hidden() -> None:
    """Perfect vision, zero memory: the moment the ball goes behind the band it
    must stop, whatever the true state says."""
    ctrl = WaitAndSeeOracleController()
    s, _, _ = build_episode(ball_x=0.9, paddle_x0=0.1)
    z = np.zeros((1, 4))
    for t in range(s.shape[1]):
        a = int(ctrl.act(z, z, state=s[:, t])[0])
        if s[0, t, 6] < 0.5:
            assert a == STAY, f"peeked at t={t}"
        else:
            assert a == RIGHT       # the ball is far to the right of the paddle

    # And with no ball_visible column at all it degrades into the plain oracle.
    assert int(ctrl.act(z, z, state=s[:, 22, :6])[0]) == RIGHT


# ----------------------------------------------------- the feed-forward floor


def _tiny_rnn(feedforward: bool) -> MDNRNN:
    torch.manual_seed(0)
    return MDNRNN(RNNConfig(z_dim=4, hidden=8, n_gauss=2,
                            feedforward=feedforward)).eval()


def _tiny_pool(n_ep: int = 3, T: int = 12, z: int = 4) -> StartPool:
    rng = np.random.default_rng(0)
    return StartPool(
        rng.standard_normal((n_ep, T + 1, z)).astype(np.float32),
        rng.integers(0, 3, size=(n_ep, T)).astype(np.int64),
        np.zeros((n_ep, T + 1, 7), np.float32),
        warmup=4,
    )


def test_feedforward_runs_through_the_dream_with_the_right_shapes() -> None:
    pool = _tiny_pool()
    for ff in (True, False):
        env = DreamEnv(_tiny_rnn(ff), pool, temperature=0.5, reward="dense")
        z, h = env.reset(batch=5, seed=0)
        assert z.shape == (5, 4) and h.shape == (5, 8)
        rec = dream_rollout(env, lambda z, h: np.ones(len(z), np.int64),
                            batch=5, steps=3, seed=0, record=True)
        assert rec["z"].shape == (5, 4, 4) and rec["h"].shape == (5, 3, 8)


def test_feedforward_h_is_the_mlp_layer_not_a_zero_dummy() -> None:
    """The floor has to be the FLOOR, not an accidental z-only controller.

    ``h_pre`` is read as ``h[0][0]`` by both the dream and the real harness. If
    the feed-forward model returned the incoming zero dummy there, its
    controller would see a constant vector and ``ctrl_v3_ff`` would silently
    become a second ``ctrl_v3_z_only`` -- a different experiment with the same
    name. So: it must equal the MLP's own hidden layer for the PREVIOUS input,
    and it must carry no information from further back.
    """
    m = _tiny_rnn(feedforward=True)
    pool = _tiny_pool()
    env = DreamEnv(m, pool, temperature=0.0, reward="dense")
    _, h = env.reset(batch=2, seed=0)
    assert np.abs(h).max() > 1e-6

    # h after a step is exactly ff([z_t, a_t]) for the z and a just consumed.
    z0, h0 = env.reset(batch=2, seed=0)
    a = np.array([2, 0])
    _, h1, _, _, _ = env.step(a)
    with torch.no_grad():
        want = m.ff(torch.cat([torch.from_numpy(z0),
                               torch.eye(3)[torch.from_numpy(a)]], -1))
    assert np.allclose(h1, want.numpy(), atol=1e-6)


class _FakeVAE:
    """A frozen random projection standing in for V, so the harness test is fast.

    ``run_real_episodes`` only ever calls ``encode``; the physics, the timing
    convention and the state bookkeeping it is being tested for do not depend on
    what the encoder computes, only on its shape.
    """

    def __init__(self, z_dim: int = 4, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(z_dim, 3 * 64 * 64, generator=g) * 0.01

    def encode(self, x):
        mu = x.reshape(len(x), -1) @ self.W.T
        return mu, torch.zeros_like(mu)


def test_feedforward_runs_through_the_real_harness_with_an_occluder() -> None:
    from wm.controller import LinearController

    m = _tiny_rnn(feedforward=True)
    ctrl = LinearController(z_dim=4, hidden=8, inputs="zh")
    roll = run_real_episodes(
        ctrl, _FakeVAE(), m, episodes=2, steps=6, seed_base=5000,
        occluder=True, record_logits=True,
    )
    assert roll["states"].shape == (2, 7, 7)      # the 7th column is ball_visible
    assert roll["actions"].shape == (2, 6)
    assert roll["logits"].shape == (2, 6, 3)
    assert "mass" not in roll                     # v3 has no mass column
    assert visible_column(roll["states"]).max() <= 1.0

    # A taller band really is taller: more hidden frames, same seeds.
    def hidden_frac(band):
        r = run_real_episodes(ctrl, _FakeVAE(), m, episodes=4, steps=60,
                              seed_base=5000, occluder=True, occluder_y=band)
        return float((visible_column(r["states"]) < 1e-3).mean())

    assert hidden_frac((0.16, 0.70)) > hidden_frac((0.28, 0.58))


def test_mass_is_not_read_off_the_ball_visible_column() -> None:
    """The v3 trap: column 6 is ``ball_visible``, and a "speed" of
    0.022 / visibility would be infinite on a hidden frame."""
    from wm.eval_controller import episode_masses

    s = np.zeros((2, 5, 7))
    assert episode_masses(s, mass_from_color=False) is None
    assert episode_masses(s, mass_from_color=True) is not None


# ------------------------------------------------------- the (V, M) pairing


V3_RUNS = {
    "runs/ctrl_v3": "runs/rnn_v3/rnn.pt",
    "runs/ctrl_v3_tau1": "runs/rnn_v3/rnn.pt",
    "runs/ctrl_v3_ff": "runs/rnn_v3_ff/rnn.pt",
    "runs/ctrl_v3_poshead": "runs/rnn_v3_poshead/rnn.pt",
    "runs/ctrl_v3_emerge": "runs/rnn_v3_emerge/rnn.pt",
    "runs/ctrl_v3_z_only": "runs/rnn_v3/rnn.pt",
}


def test_each_v3_run_is_paired_with_the_rnn_it_was_trained_in() -> None:
    """A wrong pairing is silent -- the shapes match and the number is garbage.

    ``ctrl_v3_ff`` is the one that matters: evaluate it with the recurrent model
    and it is handed a hidden state carrying exactly the memory the experiment
    claims it does not have.
    """
    present = {r: m for r, m in V3_RUNS.items()
               if (Path(r) / "history.json").exists()}
    if not present:
        pytest.skip("v3 controllers not trained in this checkout")
    for run, want in present.items():
        got, vae = stack_for_run(run, {}, {}, "runs/rnn_v3/rnn.pt",
                                 "runs/vae_v3/vae.pt")
        assert got == want, f"{run} would be evaluated with {got}"
        assert vae == "runs/vae_v3/vae.pt"
        # And it is read off the run's own record, not from this test's table.
        args = json.loads((Path(run) / "history.json").read_text())["meta"]["args"]
        assert args["rnn"] == want and args["occluder"] is True


def test_an_explicit_override_beats_the_recorded_pairing() -> None:
    over = parse_overrides(["ctrl_v3=runs/rnn_v3_emerge/rnn.pt"])
    got, _ = stack_for_run("runs/ctrl_v3", over, {}, "runs/rnn_v3/rnn.pt",
                           "runs/vae_v3/vae.pt")
    assert got == "runs/rnn_v3_emerge/rnn.pt"

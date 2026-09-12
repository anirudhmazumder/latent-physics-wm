"""v3.1 stage three: the four things that would be silently wrong.

v3.1 re-runs v3's controller stage on a re-geometried world, and the whole point
of the re-geometry is a number -- the memoryless bound falls from 0.99 to 0.48 --
that depends on two config fields being right everywhere. So:

1. **the dream-alive ball detector.** The verdict at the top of
   `README_C31.md` is a pixel measurement, and a detector that counted the
   paddle, or counted the band's antialiased edge, or missed a ball sitting
   just below the band, would produce that verdict out of nothing. Tested on
   synthetic frames rendered by the simulator itself, where the answer is known
   by construction: ball clearly outside the band, ball fully inside it, no
   ball at all, ball below it.
2. **the evaluator's world.** `paddle_w` 0.16 and band (0.13, 0.63) have to
   reach `BoxConfig`, and a v3-era call with neither flag has to still build
   v3's world -- otherwise v3's published table stops reproducing.
3. **the memoryless bound is genuinely blind on the v3.1 band.** If
   `wait_and_see` peeked for even one frame of a 21-frame occlusion the bound
   would not be a bound.
4. **the (V, M) pairing for the v3.1 runs.** A controller reads `h`, so M is
   part of the policy; `ctrl_v31_ff` evaluated with the recurrent model would
   be handed exactly the memory the experiment claims it does not have.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from worldsim.bouncing_box import BouncingBox, BoxConfig, visible_fraction
from wm.controller import WaitAndSeeOracleController
from wm.eval_controller import make_box_cfg
from wm.eval_controller_v2 import parse_overrides, stack_for_run
from wm.eval_controller_v3 import build_references, plot_skill_vs_bound
from wm.eval_dream_alive import (
    detect_ball,
    detect_paddle,
    paired_starts,
    pick_tau,
    presence_stats,
)

LEFT, STAY, RIGHT = 0, 1, 2

# The v3.1 world, spelled out once. Every assertion below about a band edge or
# a paddle width is against these two lines and nothing else.
V31_BAND = (0.13, 0.63)
V31_PADDLE_W = 0.16
V31_CFG = BoxConfig(res=64, ball_radius=0.08, occluder=True,
                    occluder_y=V31_BAND, paddle_w=V31_PADDLE_W)


# ------------------------------------------------------- 1. the ball detector


def _frame(ball_y: float, ball_x: float = 0.5, paddle_x: float = 0.5,
           cfg: BoxConfig = V31_CFG) -> np.ndarray:
    """One simulator-rendered frame with the ball at a chosen height.

    Rendered by ``BouncingBox`` itself rather than drawn by hand, so the
    detector is tested against the exact pixels -- antialiasing, band colour
    and all -- that it will meet on real and decoded frames. The env is reset
    and then its state is overwritten: the physics is irrelevant here, the
    geometry is the whole test.
    """
    env = BouncingBox(cfg, seed=0)
    env.reset(seed=0)
    env.ball[:] = (ball_x, ball_y)
    env.paddle_x = paddle_x
    return (np.asarray(env.render(), np.float32) / 255.0)[None, ...]


def test_ball_detector_fires_above_the_band_and_not_inside_it() -> None:
    """Present when the ball is clear of the band, absent when it is behind it."""
    above = detect_ball(_frame(0.85), V31_CFG)
    assert above["present"][0]
    assert above["mass"][0] == pytest.approx(1.0, abs=0.15)
    assert not above["below"][0]
    assert above["y"][0] == pytest.approx(0.85, abs=0.02)

    # Deep inside the band (its middle is 0.38), nothing sticks out at all.
    inside = detect_ball(_frame(0.38), V31_CFG)
    assert not inside["present"][0]
    assert inside["mass"][0] < 0.05
    assert np.isnan(inside["x"][0]) and np.isnan(inside["y"][0])


def test_ball_detector_calls_a_ball_below_the_band_below_it() -> None:
    """The measurement the v3.1 verdict rests on: is the ball where the paddle
    can act? The band's bottom edge is 0.13, so a ball at 0.10 is below it and
    a ball at 0.30 (inside the band) is not."""
    low = detect_ball(_frame(0.10, ball_x=0.3, paddle_x=0.8), V31_CFG)
    assert low["present"][0] and low["below"][0]
    assert low["x"][0] == pytest.approx(0.3, abs=0.03)

    assert not detect_ball(_frame(0.30), V31_CFG)["below"][0]
    assert not detect_ball(_frame(0.85), V31_CFG)["below"][0]


def test_ball_detector_ignores_the_paddle_and_the_band_edge() -> None:
    """An empty frame must read zero.

    Both distractors are in it: the blue paddle sits on the floor below the
    band and the band's own antialiased edges run across the middle. Either one
    leaking in would put a floor under every "no ball" measurement and the
    dream would look alive by construction.
    """
    empty = detect_ball(_frame(0.38, paddle_x=0.25), V31_CFG)   # ball hidden
    assert empty["mass"][0] < 0.05 and empty["mass_below"][0] < 0.05
    assert not empty["present"][0]


def test_paddle_detector_reads_the_paddle_x() -> None:
    """The reward-head correlation is against |decoded ball x - decoded paddle
    x|, so the paddle estimate has to be right to a fraction of a paddle."""
    for px in (0.15, 0.5, 0.85):
        got = detect_paddle(_frame(0.38, paddle_x=px), V31_CFG)[0]
        assert got == pytest.approx(px, abs=0.02)


def test_presence_stats_distinguishes_a_parked_ball_from_a_bouncing_one() -> None:
    """Arrivals, not presence, is what the temperature is chosen on.

    Two runs with the SAME fraction of frames below the band: one arrives three
    times, one arrives once and parks. The fractions cannot tell them apart;
    ``arrivals_per_run`` and ``mean_y_sd`` must.
    """
    steps = 60
    bouncing = np.zeros((1, steps), bool)
    for k in range(3):
        bouncing[0, 10 + 15 * k: 16 + 15 * k] = True
    parked = np.zeros((1, steps), bool)
    parked[0, 10:28] = True
    present = np.ones((1, steps), bool)

    y_bounce = np.where(bouncing[0], 0.10, 0.60)[None, :]
    y_park = np.full((1, steps), 0.12)

    b = presence_stats(present, bouncing, y_bounce)
    p = presence_stats(present, parked, y_park)
    assert b["frac_ball_below_overall"] == pytest.approx(18 / steps)
    assert p["frac_ball_below_overall"] == pytest.approx(18 / steps)
    assert b["arrivals_per_run"] == 3.0
    assert p["arrivals_per_run"] == 1.0
    assert b["mean_y_sd"] > 10 * p["mean_y_sd"]


def test_pick_tau_prefers_the_arrival_rate_closest_to_the_world() -> None:
    """Neither the quietest nor the busiest dream: the one that matches.

    This is the rule that, measured on ``frac_reemerged`` instead, picked a
    dream that never returns the ball over one that always does, by 0.1 of a
    percentage point.
    """
    cells = {
        "0.0": {"arrivals_per_run": 0.03},
        "0.5": {"arrivals_per_run": 0.75},
        "1.0": {"arrivals_per_run": 4.80},
    }
    assert pick_tau(cells, {"arrivals_per_run": 0.80}) == 0.5
    # A world that barely ever produces one picks the quiet dream.
    assert pick_tau(cells, {"arrivals_per_run": 0.05}) == 0.0


def test_paired_starts_leave_room_for_the_real_reference() -> None:
    """The dream is graded against the real continuation of the same starts, so
    every start must have a full window of real frames after it. Measured with
    the unconstrained draw, half the reference windows were the last frame
    repeated and the real re-emergence rate came out at 48.8 % instead of 70 %.
    """
    class _Pool:
        warmup = 8
        actions = np.zeros((5, 200), np.int64)

    e, t0 = paired_starts(_Pool(), 64, 150, np.random.default_rng(0))
    assert len(e) == len(t0) == 64
    assert t0.min() >= 8 and t0.max() + 150 <= 200
    with pytest.raises(SystemExit):
        paired_starts(_Pool(), 8, 199, np.random.default_rng(0))


# ---------------------------------------------- 2. the evaluator's v3.1 world


def test_make_box_cfg_builds_the_v31_world() -> None:
    cfg = make_box_cfg(ball_radius=0.08, occluder=True,
                       occluder_y=V31_BAND, paddle_w=V31_PADDLE_W)
    assert cfg.occluder is True
    assert cfg.occluder_y == V31_BAND
    assert cfg.paddle_w == V31_PADDLE_W
    assert cfg.ball_radius == 0.08
    assert cfg.res == 64


def test_make_box_cfg_with_no_v31_flags_is_still_v3() -> None:
    """v3's published table has to keep reproducing: with neither new flag the
    config is byte-for-byte the one v3 evaluated in."""
    cfg = make_box_cfg(ball_radius=0.08, occluder=True)
    assert cfg.occluder_y == (0.28, 0.58)
    assert cfg.paddle_w == 0.26


def test_the_references_get_the_v31_paddle_width() -> None:
    """Both oracles' dead zones are a fraction of the paddle width, so a 0.26
    oracle in a 0.16 world would park while the ball was still outside the
    paddle and the ceiling would come out below 0.99."""
    refs = {c.name: c for c in build_references(V31_PADDLE_W)}
    assert set(refs) == {"stay", "random", "oracle", "wait_and_see"}
    assert refs["oracle"].paddle_w == V31_PADDLE_W
    assert refs["wait_and_see"].paddle_w == V31_PADDLE_W
    # And the default is still v1-v3's, so no existing caller changes meaning.
    assert {c.name: getattr(c, "paddle_w", None)
            for c in build_references()}["oracle"] == 0.26


# --------------------------------- 3. the memoryless bound is blind, on v3.1


def _v31_descent(n: int = 60, ball_x: float = 0.9, paddle_x: float = 0.1):
    """A ball falling through the v3.1 band, with its ``ball_visible`` column.

    ``ball_visible`` is the fraction of the ball's area outside the band, which
    is how ``worldsim`` computes it; all the reference needs is that it be < 0.5
    while the ball is behind the band. v3.1's band is 0.50 tall against a 0.16
    ball, so the fully hidden stretch is ~17 frames at 0.02/frame -- twice v3's,
    which is exactly the point of the re-design.
    """
    y = np.maximum(0.95 - 0.02 * np.arange(n), 0.02)
    r = V31_CFG.ball_radius
    vis = np.array([visible_fraction(float(v), r, V31_BAND) for v in y])
    s = np.zeros((1, n, 7))
    s[0, :, 0] = ball_x
    s[0, :, 1] = y
    s[0, :, 3] = -0.02
    s[0, :, 4] = paddle_x
    s[0, :, 6] = vis
    return s


def test_wait_and_see_is_blind_for_the_whole_v31_occlusion() -> None:
    ctrl = WaitAndSeeOracleController(paddle_w=V31_PADDLE_W)
    s = _v31_descent()
    z = np.zeros((1, 4))
    hidden = 0
    for t in range(s.shape[1]):
        a = int(ctrl.act(z, z, state=s[:, t])[0])
        if s[0, t, 6] < 0.5:
            hidden += 1
            assert a == STAY, f"the memoryless bound peeked at t={t}"
        else:
            assert a == RIGHT      # the ball is far right of the paddle
    # The band has to actually hide the ball for a long stretch, or the test
    # above is vacuous. v3's band managed ~9 frames; v3.1's must manage 15+.
    assert hidden >= 15, f"only {hidden} hidden frames -- wrong band?"


def test_the_v31_band_hides_the_ball_right_down_to_contact_height() -> None:
    """The design's central claim about the geometry: the band's bottom edge is
    at contact height, so there are no visible frames of run-up left. If this
    drifted, the memoryless bound would rise back toward v3's 0.99."""
    s = _v31_descent()
    vis = s[0, :, 6]
    y = s[0, :, 1]
    emerge = float(y[np.flatnonzero(vis > 0.5)[-1]]) if (vis > 0.5).any() else 0.0
    contact = V31_CFG.paddle_h + V31_CFG.ball_radius
    assert emerge <= contact + 0.03, (
        f"the ball is half visible again at y={emerge:.3f}, well above contact "
        f"height {contact:.3f} -- that is v3's geometry, not v3.1's"
    )


# ----------------------------------------------- 4. the (V, M) pairing for v3.1

V31_RUNS = {
    "runs/ctrl_v31": "runs/rnn_v31/rnn.pt",
    "runs/ctrl_v31_tau1": "runs/rnn_v31/rnn.pt",
    "runs/ctrl_v31_ff": "runs/rnn_v31_ff/rnn.pt",
    "runs/ctrl_v31_poshead": "runs/rnn_v31_poshead/rnn.pt",
    "runs/ctrl_v31_allbands": "runs/rnn_v31_allbands/rnn.pt",
    "runs/ctrl_v31_z_only": "runs/rnn_v31/rnn.pt",
    "runs/ctrl_v31_real": "runs/rnn_v31/rnn.pt",
}


def test_each_v31_run_is_paired_with_the_rnn_it_was_trained_in() -> None:
    """A wrong pairing is silent: the shapes match and the number is garbage.

    And the pairing is read off each run's own ``history.json``, not off this
    test's table -- the table only says what the answer should be.
    """
    present = {r: m for r, m in V31_RUNS.items()
               if (Path(r) / "history.json").exists()}
    if not present:
        pytest.skip("v3.1 controllers not trained in this checkout")
    for run, want in present.items():
        got, vae = stack_for_run(run, {}, {}, "runs/rnn_v31/rnn.pt",
                                 "runs/vae_v31/vae.pt")
        assert got == want, f"{run} would be evaluated with {got}"
        assert vae == "runs/vae_v31/vae.pt"
        args = json.loads((Path(run) / "history.json").read_text())["meta"]["args"]
        assert args["rnn"] == want
        # The world every one of them was trained in, checked from the record.
        assert args["occluder"] is True
        assert tuple(args["occluder_y"]) == V31_BAND
        assert args["paddle_w"] == V31_PADDLE_W


def test_an_explicit_override_still_beats_the_recorded_pairing() -> None:
    over = parse_overrides(["ctrl_v31=runs/rnn_v31_emerge/rnn.pt"])
    got, _ = stack_for_run("runs/ctrl_v31", over, {}, "runs/rnn_v31/rnn.pt",
                           "runs/vae_v31/vae.pt")
    assert got == "runs/rnn_v31_emerge/rnn.pt"


# ------------------------------------------------------------ the new figure


def test_skill_vs_bound_plot_draws_one_bar_per_non_reference_row(tmp_path) -> None:
    """The headline figure. Cheap to draw and easy to break by renaming a row,
    so it is exercised rather than trusted."""
    def row(name, v):
        return {"name": name, "interceptions_per_visit": v,
                "interceptions_per_visit_ci95": [v - 0.05, v + 0.05]}

    rows = [row("oracle", 0.99), row("wait_and_see", 0.48), row("stay", 0.38),
            row("ctrl_v31", 0.55), row("ctrl_v31_ff", 0.40)]
    out = plot_skill_vs_bound(rows, tmp_path / "bound.png")
    assert out.exists() and out.stat().st_size > 5000

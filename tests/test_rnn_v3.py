"""Tests for the v3 additions to stage two: the memory control and the graders.

Runnable two ways (pytest is not guaranteed to be installed here)::

    python -m tests.test_rnn_v3
    pytest tests/test_rnn_v3.py

Four things are checked, and they are the four places a silent v3 bug would do
the most damage:

    * the **feed-forward control** really has no memory. Shapes, checkpoint
      round trip, and -- the test that matters -- permuting the time axis
      leaves every per-step output unchanged. If that were false the control
      would have some memory, and the whole object-permanence argument (which
      is a gap between the recurrent model and this one) would be measuring a
      smaller gap than it claims;
    * **hidden-run extraction from events**, against a hand-built example.
      ``events[e, t]`` describes the transition into frame ``t + 1``, so a run
      of EVENT_HIDDEN at event indices i..j is a run of hidden STATE frames at
      i+1..j+1. An off-by-one here shifts every exit time in the stage by
      exactly one frame and nothing else would notice;
    * the **exit-time detector**, on a synthetic trajectory whose crossing
      frame is known by construction, including the two-frame hysteresis and
      the single-frame spike it is there to reject;
    * the **no-memory baseline**, which has to be exactly "the ball is where it
      was when it vanished, and it comes back when it really did" -- a baseline
      that accidentally saw one frame of the future would be unbeatable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from worldsim.bouncing_box import EVENT_HIDDEN, EVENT_PADDLE, EVENT_WALL_X

from wm.permanence import (
    first_exit, hidden_age, hidden_runs_from_events, linear_exit,
    no_memory_exit, reflect_x,
)
from wm.rnn import MDNRNN, RNNConfig, load_rnn, save_rnn

BAND = (0.28, 0.58)
R = 0.08          # band interior is therefore [0.36, 0.50]


# --------------------------------------------------- the feed-forward control


def _batch(B=3, T=7, z=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    zz = torch.randn(B, T, z, generator=g)
    a = torch.eye(3)[torch.randint(0, 3, (B, T), generator=g)]
    return zz, a


def test_feedforward_shapes_match_the_lstm() -> None:
    z, a = _batch()
    outs = {}
    for ff in (False, True):
        m = MDNRNN(RNNConfig(feedforward=ff))
        parts, h = m(z, a)
        outs[ff] = parts
        assert parts["h"].shape == (3, 7, 256)
        assert parts["mean"].shape == (3, 7, 5, 16)
        assert parts["logits"].shape == (3, 7, 5)
        assert parts["hit_logit"].shape == (3, 7, 1)
        # init_hidden / the carried state keep their shapes so that every
        # caller written for the LSTM works unchanged.
        assert h[0].shape == (1, 3, 256) and h[1].shape == (1, 3, 256)
        h0 = m.init_hidden(3)
        assert h0[0].shape == (1, 3, 256)
        p1, _ = m.step(z[:, 0], a[:, 0], h0)
        assert p1["mean"].shape == (3, 1, 5, 16)
    # And the control is genuinely smaller: no gates, no recurrent matrices.
    assert (sum(p.numel() for p in MDNRNN(RNNConfig(feedforward=True)).parameters())
            < sum(p.numel() for p in MDNRNN(RNNConfig()).parameters()))


def test_feedforward_has_no_memory() -> None:
    """Permute the time order; every per-step output must follow its own step.

    This is the defining property of the control, and it is the one thing that
    cannot be checked by looking at the loss curve.
    """
    torch.manual_seed(0)
    m = MDNRNN(RNNConfig(feedforward=True)).eval()
    z, a = _batch(seed=1)
    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    with torch.no_grad():
        p, _ = m(z, a)
        q, _ = m(z[:, perm], a[:, perm])
    for key in ("h", "mean", "logstd", "logits", "hit_logit", "reward"):
        assert torch.allclose(p[key][:, perm], q[key], atol=1e-6), key

    # The same permutation must NOT leave the LSTM unchanged -- otherwise the
    # test above would be passing for a trivial reason.
    torch.manual_seed(0)
    r = MDNRNN(RNNConfig(feedforward=False)).eval()
    with torch.no_grad():
        p, _ = r(z, a)
        q, _ = r(z[:, perm], a[:, perm])
    assert not torch.allclose(p["h"][:, perm], q["h"], atol=1e-6)


def test_feedforward_checkpoint_round_trip() -> None:
    torch.manual_seed(0)
    m = MDNRNN(RNNConfig(feedforward=True, hidden=64)).eval()
    z, a = _batch(z=16, seed=2)
    with torch.no_grad():
        before = m(z, a)[0]["mean"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rnn.pt"
        save_rnn(p, m)
        m2, cfg = load_rnn(p)
    assert cfg.feedforward is True and cfg.hidden == 64
    with torch.no_grad():
        after = m2(z, a)[0]["mean"]
    assert torch.allclose(before, after, atol=1e-6)


def test_default_config_is_still_recurrent() -> None:
    """Every checkpoint written before `feedforward` existed must still load."""
    cfg = RNNConfig()
    assert cfg.feedforward is False
    m = MDNRNN(cfg)
    assert m.lstm is not None and m.ff is None
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rnn.pt"
        save_rnn(p, m)
        ck = torch.load(p, weights_only=False)
        del ck["cfg"]["feedforward"]          # simulate a pre-v3 checkpoint
        torch.save(ck, p)
        m2, cfg2 = load_rnn(p)
    assert cfg2.feedforward is False and m2.lstm is not None


# -------------------------------------------------------- hidden-run extraction


def test_hidden_runs_from_events_hand_built() -> None:
    # One episode, 12 transitions. Hidden at event indices 2,3,4 and 8,9.
    # By the frame convention that is hidden STATE frames 3,4,5 and 9,10.
    ev = np.zeros((1, 12), dtype=np.int64)
    ev[0, 2:5] = EVENT_HIDDEN
    ev[0, 8:10] = EVENT_HIDDEN
    ev[0, 3] |= EVENT_WALL_X          # a wall bounce inside the first run
    ev[0, 7] |= EVENT_WALL_X          # ... and one just OUTSIDE the second
    ev[0, 6] |= EVENT_PADDLE          # an unrelated flag must not confuse it

    runs = hidden_runs_from_events(ev, min_len=1)
    assert [(r.episode, r.t0, r.t1, r.length) for r in runs] == [
        (0, 3, 5, 3), (0, 9, 10, 2)]
    assert [r.wall_x for r in runs] == [True, False]
    # entry is the last partially-visible frame, exit the first one after.
    assert (runs[0].entry, runs[0].exit) == (2, 6)

    # min_len filters on the run length, not on anything else.
    assert len(hidden_runs_from_events(ev, min_len=3)) == 1
    assert hidden_runs_from_events(ev, min_len=4) == []


def test_hidden_runs_drop_incomplete() -> None:
    """A run touching either end of the episode has no entry or no exit."""
    ev = np.zeros((2, 6), dtype=np.int64)
    ev[0, 0:2] = EVENT_HIDDEN          # starts at event 0 -> no entry frame
    ev[1, 4:6] = EVENT_HIDDEN          # ends at the last event -> no exit frame
    assert hidden_runs_from_events(ev, min_len=1) == []
    assert len(hidden_runs_from_events(ev, min_len=1, drop_incomplete=False)) == 2


def test_hidden_age_counts_consecutive_frames() -> None:
    vis = np.array([[1.0, 0.5, 0.0, 0.0, 0.0, 0.4, 1.0, 0.0, 0.0]])
    assert hidden_age(vis).tolist() == [[0, 0, 1, 2, 3, 0, 0, 1, 2]]


# --------------------------------------------------------- the exit detector


def test_first_exit_on_a_synthetic_trajectory() -> None:
    """A ball entering the band from above at a known constant speed.

    y goes 0.63, 0.58, 0.53, 0.48, ... in steps of -0.05. The interior is
    [0.36, 0.50], so the ball is fully hidden at 0.48, 0.43 and 0.38 (indices
    3-5) and is out again at 0.33 (index 6). The values are chosen to sit
    clear of both edges: 0.50 lands exactly on the interior's top edge in
    exact arithmetic but 0.58 - 0.08 is a hair below 0.50 in float64, and a
    test that depended on which side of that hair a value fell would be
    testing IEEE754 rather than the detector.
    """
    y = np.round(0.63 - 0.05 * np.arange(12), 10)
    # The evaluator always calls this on a trajectory that STARTS on the first
    # fully hidden frame (dream index 0 is the first hidden frame), so slice
    # the same way: y[3:] is 0.48, 0.43, 0.38, 0.33, ...
    k, side = first_exit(y[3:], BAND, R, consec=2)
    assert (k, side) == (3, -1)
    assert round(float(y[3 + k]), 10) == 0.33

    # Upward through the band: 0.38, 0.43, 0.48, 0.53, 0.58.
    y_up = np.round(0.38 + 0.05 * np.arange(5), 10)
    assert first_exit(y_up, BAND, R, consec=2) == (3, 1)

    # `start` skips a prefix, which is how a trajectory that begins OUTSIDE the
    # band is handled: without it the detector fires on frame 0.
    assert first_exit(y, BAND, R, consec=2)[0] == 0
    assert first_exit(y, BAND, R, consec=2, start=3)[0] == 6

    # Never leaving -> (-1, 0).
    assert first_exit(np.full(9, 0.43), BAND, R, consec=2) == (-1, 0)


def test_first_exit_hysteresis_rejects_a_single_frame_spike() -> None:
    """One frame of probe noise poking out of the band is not an exit."""
    y = np.array([0.44, 0.43, 0.20, 0.43, 0.42, 0.41, 0.30, 0.25, 0.20])
    assert first_exit(y, BAND, R, consec=1)[0] == 2      # the spike fires
    assert first_exit(y, BAND, R, consec=2)[0] == 6      # the real exit
    assert first_exit(y, BAND, R, consec=2)[1] == -1


# ----------------------------------------------------------- the baselines


def _fake_state(T: int = 20) -> np.ndarray:
    """A ball falling straight through the band while drifting right.

    x: 0.20 + 0.03 t, y: 0.70 - 0.05 t. Columns are the v3 seven.
    """
    t = np.arange(T + 1)
    s = np.zeros((T + 1, 7), dtype=np.float64)
    s[:, 0] = 0.20 + 0.03 * t
    s[:, 1] = 0.70 - 0.05 * t
    s[:, 2] = 0.03
    s[:, 3] = -0.05
    return s


def test_no_memory_baseline_is_the_last_visible_position() -> None:
    from wm.permanence import HiddenRun

    s = _fake_state()
    # y is 0.50 at t = 4 and 0.40 at t = 6, so hidden frames are 4, 5, 6.
    run = HiddenRun(episode=0, t0=4, t1=6, wall_x=False)
    b = no_memory_exit(s, run, BAND, R)
    # The entry frame is t = 3, where x = 0.29.
    assert abs(b["exit_x"] - (0.20 + 0.03 * 3)) < 1e-12
    # It is handed the true duration: 3 hidden frames -> exit 4 frames later.
    assert b["exit_time"] == 4
    # ... and it names the side the ball went IN on, which is the wrong one.
    assert b["side"] == 1
    # The true exit x is 0.03 * 4 further right; that gap IS the baseline's
    # error, and it must grow linearly with the occlusion length.
    assert abs(s[run.exit, 0] - b["exit_x"] - 0.12) < 1e-12


def test_linear_baseline_matches_the_straight_line() -> None:
    from wm.permanence import HiddenRun

    s = _fake_state()
    run = HiddenRun(episode=0, t0=4, t1=6, wall_x=False)
    b = linear_exit(s, run, BAND, R)
    # No wall is hit here, so straight-line extrapolation is EXACT.
    assert abs(b["exit_x"] - s[run.exit, 0]) < 1e-12
    assert b["side"] == -1
    # Time to fall from y(entry) = 0.55 to the lower interior edge 0.36.
    assert abs(b["exit_time"] - (0.36 - 0.55) / -0.05) < 1e-9


def test_reflect_x_folds_at_the_walls() -> None:
    assert abs(reflect_x(0.5, 0.08) - 0.5) < 1e-12
    assert abs(reflect_x(1.00, 0.08) - 0.84) < 1e-12     # 0.92 + 0.08 overshoot
    assert abs(reflect_x(0.00, 0.08) - 0.16) < 1e-12
    # Inside the box it is the identity.
    for x in (0.08, 0.3, 0.92):
        assert abs(reflect_x(x, 0.08) - x) < 1e-12


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"\n{len(fns)} tests passed")
    sys.exit(0)

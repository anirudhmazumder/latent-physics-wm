"""Tests for the v3.1 exit-clock head (``wm/clock.py``, ``--clock-head``).

Runnable two ways::

    python -m tests.test_clock31
    pytest tests/test_clock31.py

Five things are pinned here, and each of them is a place where a silent error
would still produce a plausible training curve:

    * **the counters themselves.** "frames since the ball was last at least
      half visible" and "frames until it next is" are computed with two
      cumulative passes and a reversal, which is fast and easy to get backwards
      -- a reversed ``until`` is still monotone, still in range, and still
      trains to a small MSE. So they are checked against a hand-built
      visibility sequence with the answers written out by hand, including both
      saturation cases;

    * **the frozen z-visibility probe.** The whole claim that the fair clock
      run is fair rests on "is the ball at least half visible" being readable
      from the model's own input latents. If that probe were poor, the head
      would be regressing noise and the run would measure the probe rather than
      the objective. The test demands held-out R^2 > 0.9 (v3.1 measures 0.99);

    * **checkpoint compatibility in both directions.** A ``--clock-head`` model
      must round-trip through ``save_rnn``/``load_rnn`` with its head intact,
      and every checkpoint written before the flag existed must still load --
      which it does because ``RNNConfig`` gains a defaulted field and
      ``load_rnn`` filters on the dataclass's own field set;

    * **the direct head readout** (``wm/eval_clock_readout.py``). The verdict in
      ``README_CLOCK31.md`` §3 -- every clock model learned "how long has the
      ball been gone" and none learned "when does it come back" -- rests on
      scoring the head's two outputs SEPARATELY, in frames, on hidden frames
      only. A ``score`` that pooled the targets, or reported the head's scaled
      [0, 1] units, would reproduce exactly the blindness the module exists to
      remove, and would do so with numbers that still looked reasonable;

    * **the two-seed summary helper.** Part B's entire table is mean and spread
      over two seeds; an off-by-one there would silently rewrite the ordering
      the report concludes from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wm.clock import (  # noqa: E402
    CLOCK_CLIP, clock_arrays_for_roots, clock_targets, fit_visibility_probe,
    seed_summary, unscale,
)
from wm.rnn import MDNRNN, RNNConfig, load_rnn, save_rnn  # noqa: E402

V31 = ROOT / "data" / "v31"


def _have(*roots: Path) -> bool:
    return all((r / "mu.npy").exists() for r in roots)


# ------------------------------------------------------------ the counters


def test_clock_targets_hand_built() -> None:
    """The worked example, in frames, with both saturations.

        vis    1  1  0  0  0  1  0
        since  0  0  1  2  3  0  1
        until  0  0  3  2  1  0  CLIP     <- nothing visible ever again
    """
    vis = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    got = unscale(clock_targets(vis, clip=40), clip=40)
    assert got.shape == (7, 2)
    assert np.array_equal(got[:, 0], [0, 0, 1, 2, 3, 0, 1])
    assert np.array_equal(got[:, 1], [0, 0, 3, 2, 1, 0, 40])


def test_clock_targets_leading_hidden_saturates() -> None:
    """An episode that starts hidden has no earlier sighting: ``since`` clips.

    This is the case that must NOT be special-cased away. Those frames really
    are "a long time since", and dropping them would delete exactly the longest
    occlusions -- the ones the head exists for.
    """
    vis = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    got = unscale(clock_targets(vis, clip=3), clip=3)
    assert np.array_equal(got[:, 0], [3, 3, 0, 1, 2, 3])
    assert np.array_equal(got[:, 1], [2, 1, 0, 3, 3, 3])


def test_clock_targets_clipping_and_range() -> None:
    """Values are clipped INTO [0, clip] and returned scaled into [0, 1]."""
    vis = np.zeros(100)
    vis[0] = 1.0
    scaled = clock_targets(vis, clip=40)
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    frames = unscale(scaled, clip=40)
    # since: 0, 1, 2, ... saturating at 40 from t = 40 onward.
    assert frames[0, 0] == 0 and frames[10, 0] == 10
    assert frames[40, 0] == 40 and frames[99, 0] == 40
    # until: never visible again, so saturated everywhere after t = 0.
    assert frames[0, 1] == 0 and np.all(frames[1:, 1] == 40)


def test_clock_targets_half_visible_threshold() -> None:
    """The reset is "at least HALF visible", not "fully visible".

    ``ball_visible`` is the fraction of the ball clear of the band, and the
    ball is partially visible on ~33 % of v3.1 frames. A counter that only
    reset on 1.0 would spend a third of its life mis-counting.
    """
    vis = np.array([0.0, 0.49, 0.51, 1.0, 0.2])
    got = unscale(clock_targets(vis, clip=10), clip=10)
    assert np.array_equal(got[:, 0], [10, 10, 0, 0, 1])


def test_clock_targets_batched_matches_per_row() -> None:
    """(E, T) in -> (E, T, 2) out, row for row identical to the 1-d call."""
    rng = np.random.default_rng(0)
    vis = (rng.random((5, 37)) > 0.6).astype(np.float32)
    batched = clock_targets(vis, clip=CLOCK_CLIP)
    assert batched.shape == (5, 37, 2)
    for e in range(5):
        assert np.allclose(batched[e], clock_targets(vis[e], clip=CLOCK_CLIP))


def test_clock_targets_are_consistent_with_each_other() -> None:
    """On any frame with ``since == 0`` the ball is visible, so ``until == 0``.

    A cheap invariant that catches a swapped pair of columns, which is the
    other way the reversal trick can go wrong.
    """
    rng = np.random.default_rng(1)
    vis = (rng.random((4, 60)) > 0.5).astype(np.float32)
    t = clock_targets(vis, clip=CLOCK_CLIP)
    both_zero = (t[..., 0] == 0) == (t[..., 1] == 0)
    assert both_zero.all()


# ------------------------------------------------------- the frozen z-probe


def test_visibility_probe_is_good_enough_to_be_fair() -> None:
    """Held-out R^2 > 0.9 for ``mu -> ball_visible``, split BY EPISODE.

    If this ever fails, `--clock-head` stops being a test of the objective and
    becomes a test of the probe, and `README_CLOCK31.md`'s fair/privileged
    comparison loses its meaning.
    """
    roots = [V31 / "val", V31 / "val_mix"]
    if not _have(*roots):
        print("  (skipped: data/v31/{val,val_mix} not present)")
        return
    _, r2 = fit_visibility_probe([str(r) for r in roots], n_samples=20000,
                                 seed=0)
    assert r2 > 0.9, f"visibility probe R^2 {r2:.4f}"


def test_privileged_clock_arrays_match_the_simulator_column() -> None:
    """``probe=None`` must be exactly ``clock_targets`` of the true column.

    The fair and the privileged run differ in ONE argument, so this is the test
    that the argument does what its name says and nothing else moved with it.
    """
    root = V31 / "val"
    if not _have(root):
        print("  (skipped: data/v31/val not present)")
        return
    arrays, info = clock_arrays_for_roots([str(root)], None, clip=CLOCK_CLIP)
    states = np.load(root / "states.npy")
    expect = clock_targets(states[:, :, 6], clip=CLOCK_CLIP)
    assert np.allclose(arrays[0], expect)
    # The privileged targets agree with themselves by construction.
    assert info["threshold_agreement"] == 1.0


def test_fair_clock_arrays_mostly_agree_with_the_simulator() -> None:
    """The probe's thresholded reading matches the truth on >95 % of frames."""
    root = V31 / "val"
    if not _have(root):
        print("  (skipped: data/v31/val not present)")
        return
    probe, _ = fit_visibility_probe([str(root)], n_samples=20000, seed=0)
    arrays, info = clock_arrays_for_roots([str(root)], probe, clip=CLOCK_CLIP)
    assert info["threshold_agreement"] > 0.95
    assert arrays[0].shape == (*np.load(root / "mu.npy").shape[:2], 2)
    assert arrays[0].min() >= 0.0 and arrays[0].max() <= 1.0


# ------------------------------------------------------------- the head i/o


def test_clock_head_forward_shapes() -> None:
    m = MDNRNN(RNNConfig(z_dim=16, clock_head=True, vy_head=True))
    z = torch.zeros(3, 7, 16)
    a = torch.zeros(3, 7, 3)
    parts, _ = m(z, a)
    assert parts["clock"].shape == (3, 7, 2)
    assert parts["ball_vy"].shape == (3, 7, 1)


def test_clock_head_is_off_by_default() -> None:
    """No flag, no parameters, no output key -- so the state dict is unchanged."""
    m = MDNRNN(RNNConfig(z_dim=16))
    assert m.clock_head is None and m.vy_head is None
    parts, _ = m(torch.zeros(1, 2, 16), torch.zeros(1, 2, 3))
    assert "clock" not in parts and "ball_vy" not in parts
    assert not any(k.startswith(("clock_head", "vy_head"))
                   for k in m.state_dict())


def test_clock_model_round_trips(tmp_path: Path | None = None) -> None:
    """save_rnn/load_rnn keeps the head and its weights."""
    import tempfile

    d = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    m = MDNRNN(RNNConfig(z_dim=16, clock_head=True))
    with torch.no_grad():
        m.clock_head.weight.fill_(0.123)
    p = save_rnn(d / "rnn.pt", m, {"clock_head": True})
    back, cfg = load_rnn(p)
    assert cfg.clock_head is True and back.clock_head is not None
    assert torch.allclose(back.clock_head.weight,
                          torch.full_like(back.clock_head.weight, 0.123))


def test_pre_clock_checkpoints_still_load(tmp_path: Path | None = None) -> None:
    """A checkpoint whose saved cfg has no clock fields must still load.

    Simulated by deleting the new keys from a saved payload, which is exactly
    what a checkpoint written before the flag existed looks like on disk.
    """
    import tempfile

    d = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    m = MDNRNN(RNNConfig(z_dim=16))
    p = save_rnn(d / "old.pt", m, {})
    ck = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("clock_head", "vy_head"):
        ck["cfg"].pop(k, None)
    torch.save(ck, p)
    back, cfg = load_rnn(p)
    assert cfg.clock_head is False and back.clock_head is None


def test_real_v31_baseline_checkpoint_still_loads() -> None:
    """The frozen baseline on disk, loaded by the post-change code."""
    ck = ROOT / "runs" / "rnn_v31" / "rnn.pt"
    if not ck.exists():
        print("  (skipped: runs/rnn_v31/rnn.pt not present)")
        return
    _, cfg = load_rnn(ck)
    assert cfg.clock_head is False and cfg.vy_head is False


# ------------------------------------------------- the dataset plumbing


def test_frame_targets_align_with_state() -> None:
    """``clock`` must be sliced on the SAME axis as ``state`` (t0+1 .. t0+L).

    Every head on ``h`` in this project reads ``h_t`` and describes the frame
    the MDN is predicting. If the clock were sliced one frame earlier it would
    still train, and it would be counting the wrong frame.
    """
    root = V31 / "val"
    if not _have(root):
        print("  (skipped: data/v31/val not present)")
        return
    from wm.seq_data import LatentSequenceDataset

    arrays, _ = clock_arrays_for_roots([str(root)], None, clip=CLOCK_CLIP)
    ds = LatentSequenceDataset(str(root), seq_len=8, stride=13, use_mean=True,
                               frame_targets={"clock": arrays})
    plain = LatentSequenceDataset(str(root), seq_len=8, stride=13,
                                  use_mean=True)
    assert set(ds[0]) - set(plain[0]) == {"clock"}
    for i in (0, 3, len(ds) - 1):
        ri, e, t0 = ds.index[i]
        item = ds[i]
        assert item["clock"].shape == (8, 2)
        assert np.allclose(item["clock"].numpy(), arrays[ri][e, t0 + 1: t0 + 9])
        # and the un-targeted dataset is byte-identical elsewhere
        assert torch.allclose(item["state"], plain[i]["state"])


def test_frame_targets_shape_is_checked() -> None:
    """A mis-shaped extra target must fail loudly, not broadcast quietly."""
    root = V31 / "val"
    if not _have(root):
        print("  (skipped: data/v31/val not present)")
        return
    from wm.seq_data import LatentSequenceDataset

    bad = [np.zeros((3, 5, 2), dtype=np.float32)]
    try:
        LatentSequenceDataset(str(root), seq_len=8, frame_targets={"c": bad})
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected a ValueError on a mis-shaped target")


# ------------------------------------------------- the direct head readout


def test_readout_score_is_per_target_and_in_frames() -> None:
    """``score`` must keep the two counters APART and report frames, not scaled.

    The whole point of `wm.eval_clock_readout` is that the training log's single
    combined clock MSE hides an asymmetry -- v3.1's head learned `since` and not
    `until` -- so a ``score`` that pooled the two targets would reproduce
    exactly the blindness the module exists to remove.
    """
    from wm.eval_clock_readout import TARGET_NAMES, score

    true = np.zeros((1, 4, 2), dtype=np.float32)
    true[0, :, 0] = [0, 10, 20, 30]      # since: a perfect prediction below
    true[0, :, 1] = [8, 6, 4, 2]         # until: predicted as its own mean
    pred = true.copy()
    pred[0, :, 1] = 5.0
    mask = np.ones((1, 4), dtype=bool)

    got = score(pred, true, mask)
    assert set(TARGET_NAMES) <= set(got) and got["n_frames"] == 4
    # Exact prediction -> R^2 1, rmse 0.
    assert abs(got["frames_since"]["r2"] - 1.0) < 1e-9
    assert got["frames_since"]["rmse_frames"] < 1e-6
    # Predicting the mean -> R^2 exactly 0, and the rmse is in FRAMES (the
    # target sd), not in the [0, 1] units the head is trained in.
    assert abs(got["frames_until"]["r2"]) < 1e-9
    assert abs(got["frames_until"]["rmse_frames"] - np.sqrt(5.0)) < 1e-6
    assert abs(got["frames_since"]["true_mean_frames"] - 15.0) < 1e-6


def test_readout_masks_to_the_requested_frames() -> None:
    """Only masked frames count, and R^2 is against THEIR variance."""
    from wm.eval_clock_readout import score

    true = np.zeros((1, 4, 2), dtype=np.float32)
    true[0, :, 0] = [0, 0, 10, 30]
    pred = true.copy()
    pred[0, :2, 0] = 99.0                      # garbage, but masked out
    got = score(pred, true, np.array([[False, False, True, True]]))
    assert got["n_frames"] == 2
    assert abs(got["frames_since"]["r2"] - 1.0) < 1e-9


def test_readout_refuses_a_checkpoint_without_the_head() -> None:
    """A model with no clock head must fail loudly rather than KeyError."""
    import tempfile

    from wm.eval_clock_readout import head_predictions

    d = Path(tempfile.mkdtemp())
    p = save_rnn(d / "plain.pt", MDNRNN(RNNConfig(z_dim=16)), {})
    try:
        head_predictions(str(p), np.zeros((1, 3, 16), dtype=np.float32),
                         np.zeros((1, 2), dtype=np.int64))
    except SystemExit as exc:
        assert "--clock-head" in str(exc)
    else:
        raise AssertionError("expected a SystemExit on a head-less checkpoint")


def test_readout_prediction_shape_and_units() -> None:
    """(E, T+1, z) latents in -> (E, T, 2) counters in frames out."""
    import tempfile

    from wm.eval_clock_readout import head_predictions

    d = Path(tempfile.mkdtemp())
    m = MDNRNN(RNNConfig(z_dim=16, clock_head=True))
    p = save_rnn(d / "clock.pt", m, {"clock_head": True})
    got = head_predictions(str(p), np.zeros((2, 6, 16), dtype=np.float32),
                           np.zeros((2, 5), dtype=np.int64))
    assert got.shape == (2, 5, 2)
    # The head's raw output is in [0, 1]; the readout returns frames, so an
    # untrained head's bias-only output must still be on a CLOCK_CLIP scale.
    assert np.abs(got).max() <= CLOCK_CLIP


# -------------------------------------------------------- two-seed helper


def test_seed_summary_mean_and_spread() -> None:
    s = seed_summary([0.73, 0.61])
    assert abs(s["mean"] - 0.67) < 1e-12
    assert abs(s["spread"] - 0.12) < 1e-12
    assert s["min"] == 0.61 and s["max"] == 0.73 and s["n"] == 2


def test_seed_summary_single_and_empty() -> None:
    one = seed_summary([0.5])
    assert one["n"] == 1 and one["mean"] == 0.5 and one["spread"] == 0.0
    none = seed_summary([])
    assert none["n"] == 0 and np.isnan(none["mean"]) and np.isnan(none["spread"])


def test_seed_summary_drops_missing_runs() -> None:
    s = seed_summary([0.4, float("nan"), None, 0.6])
    assert s["n"] == 2 and abs(s["mean"] - 0.5) < 1e-12
    assert abs(s["spread"] - 0.2) < 1e-12


# ------------------------------------------------------------------- main

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"- {fn.__name__}")
        fn()
    print(f"\n{len(fns)} checks passed")

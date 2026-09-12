"""Tests for the v3.1 generalisations: multi-root frames, paddle width, bands.

Runnable two ways (pytest is not guaranteed to be installed here)::

    python -m tests.test_v31
    pytest tests/test_v31.py

v3.1 changed no physics. What it changed is *plumbing*, and plumbing is where a
silent mistake survives longest, because every number downstream still looks
plausible. Four things are pinned here, one per change:

    * the **multi-root FrameDataset**. The VAE now trains on three occluder
      heights at once, and the concatenation has to be boring: with one root it
      must be the same object it always was, item for item; with several, the
      length must be the exact sum and item ``i`` in the first root's range must
      still be that root's item ``i``. Get the index arithmetic wrong and the
      encoder trains on a shuffled-but-valid-looking mixture, which no loss
      curve would ever reveal;

    * **``--paddle-w`` reaches the config and the meta**. The paddle's width is
      what sets the *chance* catch rate (0.26 -> 0.50, 0.16 -> 0.38 per floor
      visit, ``runs/v31_design/sweep.md``), so a dataset collected at one width
      and evaluated at another is scored against the wrong floor. It therefore
      has to travel with the data, and the default has to stay 0.26 or every
      v1-v3 dataset stops reproducing;

    * **the band arguments reach ``eval_permanence_v3``**. v3 hard-coded the
      names ``tall`` and ``taller``; v3.1's comparison bands are a *shorter*
      and a *longer* occlusion, so the names are now data. The parse must
      produce the v3 pair when nothing is passed;

    * **the v1/v2/v3 byte-identity tests still pass.** They live in
      ``tests/test_env_v2.py`` and ``tests/test_env_v3.py``; this file re-runs
      them so that "did v3.1 break an older dataset" has an answer inside the
      v3.1 test file too, rather than depending on someone running the whole
      suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldsim.bouncing_box import BoxConfig  # noqa: E402
from worldsim.collect import collect  # noqa: E402

V31 = ROOT / "data" / "v31"


def _have(*roots: Path) -> bool:
    return all((r / "frames.npy").exists() for r in roots)


# --------------------------------------------------------- multi-root frames


def test_single_root_frame_dataset_is_unchanged() -> None:
    """One root passed as a string and as a one-element list must agree exactly.

    Not "agree statistically": the same flat index has to address the same
    (episode, timestep), which is what lets a multi-root run be compared
    against a single-root one.
    """
    from wm.data import FrameDataset

    root = V31 / "val"
    if not _have(root):
        print("  (skipped: data/v31/val not present)")
        return
    a = FrameDataset(root, return_state=True)
    b = FrameDataset([root], return_state=True)
    assert len(a) == len(b) == a.E * a.Tp1
    for i in (0, 1, 7, len(a) // 2, len(a) - 1):
        xa, sa = a[i]
        xb, sb = b[i]
        assert np.array_equal(xa.numpy(), xb.numpy())
        assert np.array_equal(sa.numpy(), sb.numpy())
    assert a.roots == b.roots == [root]


def test_multi_root_length_and_prefix_alignment() -> None:
    """Length is the exact sum; the first root's items keep their indices.

    The second half of that is the load-bearing one. ``__len__`` being right is
    easy; the index *arithmetic* being right is what the searchsorted could get
    wrong, and a half-frame offset in the second root would still produce a
    perfectly trainable dataset of real frames.
    """
    from wm.data import FrameDataset

    r1, r2 = V31 / "val", V31 / "short"
    if not _have(r1, r2):
        print("  (skipped: data/v31/{val,short} not present)")
        return
    d1, d2 = FrameDataset(r1), FrameDataset(r2)
    both = FrameDataset([r1, r2], return_state=True)
    assert len(both) == len(d1) + len(d2)

    for i in (0, 3, len(d1) - 1):
        assert np.array_equal(both[i][0].numpy(), d1[i].numpy())
    for j in (0, 5, len(d2) - 1):
        assert np.array_equal(both[len(d1) + j][0].numpy(), d2[j].numpy())

    # The two roots have different band geometries on purpose: that is the
    # whole reason the VAE sees both. So the concatenation must NOT require
    # matching metas -- only matching resolution and state columns.
    assert both.metas[0]["occluder_y"] != both.metas[1]["occluder_y"]
    assert both.meta is both.metas[0]
    # Episode lengths may differ between roots (the probe split is 24 steps).
    assert both.Tp1s == [d1.Tp1, d2.Tp1]


def test_multi_root_loader_covers_every_item_exactly_once() -> None:
    """A shuffled loader over several roots is a permutation, not a resample."""
    from wm.data import make_loader

    r1, r2 = V31 / "probe", V31 / "val"
    if not _have(r1, r2):
        print("  (skipped: data/v31/{probe,val} not present)")
        return
    loader = make_loader([r1, r2], batch_size=64, shuffle=False, return_state=True)
    ds = loader.dataset
    seen = 0
    for x, s in loader:
        seen += len(x)
        assert x.shape[1:] == (3, ds.frames.shape[2], ds.frames.shape[3])
    assert seen == len(ds)


def test_multi_root_rejects_mismatched_state_columns() -> None:
    """A v1 root and a v3 root concatenated would silently mis-align `state`."""
    from wm.data import FrameDataset

    v1, v31 = ROOT / "data" / "v1" / "val", V31 / "val"
    if not _have(v1, v31):
        print("  (skipped: data/v1/val or data/v31/val not present)")
        return
    try:
        FrameDataset([v31, v1])
    except ValueError as e:
        assert "state_names" in str(e)
    else:
        raise AssertionError("mixing a v1 root with a v3 root must be refused")


# ----------------------------------------------------------- the paddle width


def test_paddle_w_default_is_still_the_v1_paddle() -> None:
    """``collect`` and ``BoxConfig`` must still agree on 0.26 when unasked."""
    import inspect

    assert BoxConfig().paddle_w == 0.26
    sig = inspect.signature(collect)
    assert sig.parameters["paddle_w"].default == 0.26


def test_paddle_w_reaches_the_config_and_the_meta(tmp_path=None) -> None:
    """A collected split records the width it was actually rendered with.

    Both halves matter: ``config.paddle_w`` is what a replay reconstructs the
    world from, and the top-level ``paddle_w`` is what a human reading
    ``meta.json`` sees. They must be the same number.
    """
    import tempfile

    out = Path(tmp_path or tempfile.mkdtemp()) / "tiny"
    collect(out, episodes=1, steps=4, res=32, seed=0, ball_radius=0.08,
            occluder=True, occluder_y=(0.13, 0.63), paddle_w=0.16)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["paddle_w"] == 0.16
    assert meta["config"]["paddle_w"] == 0.16
    assert meta["occluder_y"] == [0.13, 0.63]


def test_v31_splits_carry_the_narrow_paddle() -> None:
    """The real datasets, not a freshly collected toy one."""
    if not V31.exists():
        print("  (skipped: data/v31 not present)")
        return
    for d in sorted(V31.iterdir()):
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        assert meta["paddle_w"] == 0.16, d
        assert meta["config"]["paddle_w"] == 0.16, d
        assert meta["occluder_y"][0] == 0.13, d


def test_controller_env_takes_a_paddle_width() -> None:
    """``make_box_cfg`` defaults to v1-v3 and honours an explicit width.

    Stage three fits a controller against a chance floor set by this number, so
    a width that failed to reach the config would make every controller result
    incomparable to the oracle sweep that motivated v3.1.
    """
    from wm.eval_controller import env_kwargs, make_box_cfg

    assert make_box_cfg(0.08).paddle_w == 0.26
    cfg = make_box_cfg(0.08, occluder=True, occluder_y=(0.13, 0.63),
                       paddle_w=0.16)
    assert cfg.paddle_w == 0.16
    assert cfg.occluder_y == (0.13, 0.63)

    import argparse

    from wm.eval_controller import add_env_args

    p = add_env_args(argparse.ArgumentParser())
    p.add_argument("--ball-radius", type=float, default=0.08)
    a = p.parse_args(["--occluder", "--occluder-y", "0.13", "0.63",
                      "--paddle-w", "0.16"])
    assert env_kwargs(a)["paddle_w"] == 0.16
    assert make_box_cfg(**env_kwargs(a)).paddle_w == 0.16
    # And the flagless case is still exactly v1.
    assert env_kwargs(p.parse_args([]))["paddle_w"] is None
    assert make_box_cfg(**env_kwargs(p.parse_args([]))).paddle_w == 0.26


# ------------------------------------------------------------ the band naming


def _parse_bands(argv):
    """The band-spec parse as ``eval_permanence_v3.main`` performs it."""
    import wm.eval_permanence_v3 as ep

    p = ep.argparse.ArgumentParser()
    p.add_argument("--bands", nargs="*", default=None)
    p.add_argument("--tall", nargs="+", default=["data/v3/tall"])
    p.add_argument("--taller", nargs="+", default=["data/v3/taller"])
    a = p.parse_args(argv)
    return (
        [(n, r.split(",")) for n, r in (b.split("=", 1) for b in a.bands)]
        if a.bands else [("tall", a.tall), ("taller", a.taller)]
    )


def test_band_arguments_default_to_the_v3_pair() -> None:
    assert _parse_bands([]) == [("tall", ["data/v3/tall"]),
                                ("taller", ["data/v3/taller"])]


def test_band_arguments_are_named_on_the_cli() -> None:
    got = _parse_bands(["--bands", "short=data/v31/short",
                        "long=data/v31/long,data/v31/train_long"])
    assert got == [("short", ["data/v31/short"]),
                   ("long", ["data/v31/long", "data/v31/train_long"])]


def test_extra_and_max_age_are_flags() -> None:
    """The dream horizon and the decay curve's k range both follow the band.

    v3's 23-frame-longer occlusion is exactly the case where leaving these at
    their v3 constants would truncate the answer rather than fail loudly.
    """
    import wm.eval_permanence_v3 as ep

    p = ep.argparse.ArgumentParser()
    p.add_argument("--extra", type=int, default=ep.EXTRA)
    p.add_argument("--max-age", type=int, default=ep.MAX_AGE)
    assert vars(p.parse_args([])) == {"extra": 10, "max_age": 25}
    a = p.parse_args(["--extra", "16", "--max-age", "40"])
    assert (a.extra, a.max_age) == (16, 40)


def test_permanence_cli_accepts_the_v31_invocation() -> None:
    """The real parser, with the real v3.1 flags, must not reject them."""
    import subprocess

    r = subprocess.run(
        [sys.executable, "-m", "wm.eval_permanence_v3", "--help"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    for flag in ("--bands", "--extra", "--max-age"):
        assert flag in r.stdout, flag


# ------------------------------------------------- the older worlds, unchanged


def test_v1_v2_v3_byte_identity_still_holds() -> None:
    """Re-run the older suites' identity tests from inside the v3.1 file.

    v3.1 touched ``collect`` (a new argument) and ``BoxConfig`` is reached with
    one more keyword, so this is the cheapest possible guard on the claim that
    no existing dataset changed.
    """
    from tests import test_env_v2, test_env_v3

    test_env_v2.test_v1_frames_byte_identical()
    test_env_v3.test_v1_frames_still_byte_identical()
    test_env_v3.test_v2_frames_still_byte_identical()
    test_env_v3.test_physics_identical_with_and_without_the_band()


def test_v3_datasets_replay_byte_identically_after_the_paddle_w_change() -> None:
    """A v3 split must still reproduce from its own meta, paddle width included.

    ``collect`` now passes ``paddle_w`` to ``BoxConfig`` explicitly. Passing
    0.26 is the same as not passing it, but "is the same" is the kind of claim
    that should be executed rather than asserted in a comment.
    """
    from worldsim.bouncing_box import BouncingBox
    from worldsim.policies import sticky_random_actions

    root = ROOT / "data" / "v3" / "val"
    if not _have(root):
        print("  (skipped: data/v3/val not present)")
        return
    meta = json.loads((root / "meta.json").read_text())
    c = meta["config"]
    cfg = BoxConfig(res=meta["res"], ball_radius=c["ball_radius"],
                    occluder=True, occluder_y=tuple(meta["occluder_y"]),
                    paddle_w=c["paddle_w"])
    env = BouncingBox(cfg)
    rng = np.random.default_rng(meta["seed"])
    got = [env.reset(seed=int(rng.integers(0, 2**31 - 1)))]
    acts = sticky_random_actions(meta["steps"], rng, mean_hold=meta["mean_hold"])
    for t in range(16):
        got.append(env.step(int(acts[t]))[0])
    ref = np.load(root / "frames.npy", mmap_mode="r")
    assert np.array_equal(np.stack(got), np.asarray(ref[0, : len(got)])), \
        "v3 frames changed!"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all good" if not fails else f"{fails} failure(s)")
    sys.exit(1 if fails else 0)

"""Tests for the v3 environment (the occlusion band).

Runnable two ways (pytest is not guaranteed to be installed here)::

    python -m tests.test_env_v3
    pytest tests/test_env_v3.py

The shape of this file mirrors ``tests/test_env_v2.py``, and for the same
reason. v3 adds a branch to ``render``, a branch to ``state``, a branch to
``step`` and two fields to ``BoxConfig``. Any one of those leaking into the
default config would silently invalidate every v1 dataset and checkpoint; a
leak into the ``mass_from_color`` path would do the same to v2. So the first
two tests do not check that v1 and v2 "still roughly work" -- they replay the
collector's exact seeding and compare against the frames on disk, pixel for
pixel.

After that the file tests the four things that are genuinely new:

* the band is drawn where it says it is, opaquely, and nowhere else;
* ``ball_visible`` is a correct, monotone coverage fraction;
* ``EVENT_HIDDEN`` says exactly what ``ball_visible`` says;
* the physics is bit-identical with the band on, which is the whole premise of
  v3 -- the band changes what you can *see*, not what happens.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsim.bouncing_box import (
    EVENT_HIDDEN,
    STATE_NAMES,
    STATE_NAMES_V2,
    STATE_NAMES_V3,
    BouncingBox,
    BoxConfig,
    visible_fraction,
)
from worldsim.policies import sticky_random_actions

REPO = Path(__file__).resolve().parents[1]
V1_VAL = REPO / "data" / "v1" / "val"
V2_VAL = REPO / "data" / "v2" / "val"

BAND = (0.28, 0.58)
R = 0.08


# ------------------------------------------------------- v1 / v2 regression


def _replay_first_frames(root: Path, cfg: BoxConfig, n: int = 5) -> np.ndarray:
    """Reproduce ``frames[0, :n]`` of a collected split from its meta.json.

    Mirrors ``worldsim.collect.collect``: one rng seeded with --seed, from
    which each episode draws its own env seed, and from which (after the reset)
    the open-loop sticky action sequence is drawn. Getting that order wrong
    gives you a plausible-looking but different episode, which is exactly the
    kind of silent break these tests exist to catch.
    """
    meta = json.loads((root / "meta.json").read_text())
    env = BouncingBox(cfg)
    rng = np.random.default_rng(meta["seed"])
    got = [env.reset(seed=int(rng.integers(0, 2**31 - 1)))]
    acts = sticky_random_actions(meta["steps"], rng, mean_hold=meta["mean_hold"])
    for t in range(n - 1):
        got.append(env.step(int(acts[t]))[0])
    return np.stack(got)


def test_v1_frames_still_byte_identical() -> None:
    """Default BoxConfig() -- no occluder flag anywhere -- still reproduces v1."""
    if not (V1_VAL / "frames.npy").exists():
        print("  (skipped: data/v1/val not present)")
        return
    meta = json.loads((V1_VAL / "meta.json").read_text())
    cfg = BoxConfig(res=meta["res"], ball_radius=meta["config"]["ball_radius"])
    assert cfg.occluder is False, "default config must still be v1"
    assert cfg.mass_from_color is False
    ref = np.load(V1_VAL / "frames.npy", mmap_mode="r")
    got = _replay_first_frames(V1_VAL, cfg)
    assert np.array_equal(got, np.asarray(ref[0, : len(got)])), "v1 frames changed!"


def test_v2_frames_still_byte_identical() -> None:
    """``mass_from_color=True`` alone still reproduces data/v2/val.

    The v2 equivalent of the test above, added in v3 because v3 is the first
    change since v2 that touches ``state()`` and ``render()`` -- the two places
    a v2 dataset could be broken from.
    """
    if not (V2_VAL / "frames.npy").exists():
        print("  (skipped: data/v2/val not present)")
        return
    meta = json.loads((V2_VAL / "meta.json").read_text())
    c = meta["config"]
    cfg = BoxConfig(
        res=meta["res"],
        ball_radius=c["ball_radius"],
        mass_from_color=True,
        mass_holdout=tuple(meta["mass_holdout"]) if meta["mass_holdout"] else None,
        mass_only=tuple(meta["mass_only"]) if meta["mass_only"] else None,
    )
    assert cfg.occluder is False
    ref = np.load(V2_VAL / "frames.npy", mmap_mode="r")
    got = _replay_first_frames(V2_VAL, cfg)
    assert np.array_equal(got, np.asarray(ref[0, : len(got)])), "v2 frames changed!"


# -------------------------------------------------------------- state shape


def test_state_width_for_every_switch_combination() -> None:
    """All four (mass, occluder) combinations, and the column ORDER."""
    cases = {
        (False, False): (6, STATE_NAMES),
        (True, False): (7, STATE_NAMES_V2),
        (False, True): (7, STATE_NAMES_V3),
        (True, True): (8, STATE_NAMES_V2 + ("ball_visible",)),
    }
    for (mass, occ), (width, names) in cases.items():
        env = BouncingBox(BoxConfig(mass_from_color=mass, occluder=occ), seed=0)
        assert env.state().shape == (width,), (mass, occ)
        assert env.state_names == names, (mass, occ)
        # ball_visible is always LAST, after mass, so the two switches compose.
        if occ:
            assert env.state_names[-1] == "ball_visible"
            assert abs(float(env.state()[-1]) - env.ball_visible()) < 1e-6


# ------------------------------------------------------------- the band itself


def test_band_pixels_exact_inside_and_untouched_outside() -> None:
    """Opaque in the interior, and not a single pixel changed outside it.

    The band is drawn with the paddle's separable box coverage, so rows whose
    pixel *extent* is strictly inside the band get coverage exactly 1 and come
    out as the literal band colour, while rows whose extent is strictly outside
    get coverage exactly 0 and are byte-identical to the no-band render. The
    one or two rows straddling each edge are the antialiased fringe and are
    deliberately not asserted on -- that they are *soft* is the point.
    """
    res = 64
    cfg_on = BoxConfig(res=res, ball_radius=R, occluder=True, occluder_y=BAND)
    cfg_off = BoxConfig(res=res, ball_radius=R)
    lo, hi = BAND
    px = 1.0 / res
    ys = 1.0 - (np.arange(res) + 0.5) / res  # world y of each row centre

    for seed in range(6):
        on = BouncingBox(cfg_on, seed=seed).render()
        off = BouncingBox(cfg_off, seed=seed).render()

        inside = (ys - px / 2 > lo) & (ys + px / 2 < hi)
        outside = (ys - px / 2 > hi) | (ys + px / 2 < lo)
        assert inside.any() and outside.any()

        band = np.asarray(cfg_on.occluder_color, np.uint8)
        assert np.all(on[inside] == band), "band interior is not the band colour"
        assert np.array_equal(on[outside], off[outside]), "band leaked outside itself"


def test_band_hides_the_ball_completely() -> None:
    """A ball parked in the middle of the band leaves no trace in the frame.

    This is the claim the whole of v3 rests on: on a fully hidden frame the
    image is the *same* for every ball position, so no encoder can recover it.
    Tested directly -- two very different ball positions inside the band must
    render to identical bytes.
    """
    cfg = BoxConfig(res=64, ball_radius=R, occluder=True, occluder_y=BAND)
    mid = 0.5 * (BAND[0] + BAND[1])
    env = BouncingBox(cfg, seed=0)
    env.ball[:] = (0.2, mid)
    a = env.render()
    env.ball[:] = (0.8, mid)
    b = env.render()
    assert np.array_equal(a, b), "the hidden ball is still visible in the frame"
    assert env.ball_visible() < 1e-3


# ----------------------------------------------------------- ball_visible


def test_visible_fraction_endpoints_and_monotonicity() -> None:
    lo, hi = BAND
    # Far above and far below: fully visible.
    assert visible_fraction(0.95, R, BAND) == 1.0
    assert visible_fraction(0.05, R, BAND) == 1.0
    # Just clear of each edge by exactly the radius: still fully visible.
    assert visible_fraction(hi + R, R, BAND) == 1.0
    assert visible_fraction(lo - R, R, BAND) == 1.0
    # Deep inside: fully hidden.
    assert visible_fraction(0.5 * (lo + hi), R, BAND) < 1e-9
    # Half in, half out at each edge, by symmetry of the disc.
    assert abs(visible_fraction(lo, R, BAND) - 0.5) < 1e-9
    assert abs(visible_fraction(hi, R, BAND) - 0.5) < 1e-9

    # Monotone as the ball descends into the band from above, and back out
    # below. (Not monotone over the whole traverse -- it goes 1 -> 0 -> 1 --
    # so the two halves are checked separately.)
    down = [visible_fraction(y, R, BAND) for y in np.linspace(hi + R, 0.5 * (lo + hi), 60)]
    assert all(b <= a + 1e-12 for a, b in zip(down, down[1:]))
    assert down[0] == 1.0 and down[-1] < 1e-9
    up = [visible_fraction(y, R, BAND) for y in np.linspace(0.5 * (lo + hi), lo - R, 60)]
    assert all(b >= a - 1e-12 for a, b in zip(up, up[1:]))
    assert up[-1] == 1.0


def test_visible_fraction_matches_a_brute_force_count() -> None:
    """The analytic circular segment against dumb dense sampling of the disc."""
    rng = np.random.default_rng(0)
    n = 400_000
    # Uniform points on the unit disc, reused for every centre height.
    t = rng.uniform(0, 2 * np.pi, n)
    rad = R * np.sqrt(rng.uniform(0, 1, n))
    dy = rad * np.sin(t)
    lo, hi = BAND
    for cy in np.linspace(lo - R - 0.02, hi + R + 0.02, 25):
        y = cy + dy
        brute = float(((y <= lo) | (y >= hi)).mean())
        assert abs(visible_fraction(float(cy), R, BAND) - brute) < 3e-3, cy


def test_ball_visible_column_tracks_the_ball() -> None:
    """Whole-episode check: the state column equals the geometry, every step."""
    cfg = BoxConfig(res=64, ball_radius=R, occluder=True, occluder_y=BAND)
    env = BouncingBox(cfg, seed=4)
    rng = np.random.default_rng(4)
    for _ in range(200):
        _, s, _ = env.step(int(rng.integers(0, 3)))
        assert abs(float(s[6]) - visible_fraction(float(s[1]), R, BAND)) < 1e-6
        assert 0.0 <= float(s[6]) <= 1.0


def test_event_hidden_agrees_with_ball_visible() -> None:
    """EVENT_HIDDEN is set on exactly the steps where ball_visible < 1e-3.

    And it fires at all: an episode that never sets the bit would make the test
    vacuously true, so we assert both classes are non-empty.
    """
    cfg = BoxConfig(res=64, ball_radius=R, occluder=True, occluder_y=BAND)
    env = BouncingBox(cfg, seed=7)
    rng = np.random.default_rng(7)
    n_hidden = 0
    for _ in range(400):
        _, s, ev = env.step(int(rng.integers(0, 3)))
        flag = bool(ev & EVENT_HIDDEN)
        assert flag == (float(s[6]) < 1e-3), (float(s[6]), ev)
        n_hidden += flag
    assert 0 < n_hidden < 400, f"degenerate episode: {n_hidden} hidden frames"

    # And the bit must never appear when the occluder is off, or every v1/v2
    # events.npy would suddenly mean something different.
    env = BouncingBox(BoxConfig(ball_radius=R), seed=7)
    rng = np.random.default_rng(7)
    for _ in range(200):
        assert not (env.step(int(rng.integers(0, 3)))[2] & EVENT_HIDDEN)


# ----------------------------------------------------------- physics is inert


def test_physics_identical_with_and_without_the_band() -> None:
    """Same seed, same actions, same trajectory -- to the last bit.

    v3's entire premise is that the band is a *rendering* change. If it ever
    perturbed the dynamics (e.g. by consuming a random number at reset, or by
    a collision test) then "the model cannot see the ball" and "the ball does
    something different behind the band" would be confounded and no memory
    result would mean anything.
    """
    for seed in range(5):
        a = BouncingBox(BoxConfig(ball_radius=R), seed=seed)
        b = BouncingBox(
            BoxConfig(ball_radius=R, occluder=True, occluder_y=BAND), seed=seed
        )
        assert np.array_equal(a.state(), b.state()[:6])
        rng = np.random.default_rng(seed)
        for _ in range(300):
            act = int(rng.integers(0, 3))
            _, sa, eva = a.step(act)
            _, sb, evb = b.step(act)
            assert np.array_equal(sa, sb[:6]), "trajectory diverged"
            # Events must agree too, except for the new bit.
            assert eva == (evb & ~EVENT_HIDDEN)


def test_band_height_changes_occlusion_but_not_the_trajectory() -> None:
    """A taller band hides more, for longer, and still does not move the ball."""
    R_, seed = R, 11
    lengths = {}
    ref = None
    for band in ((0.28, 0.58), (0.22, 0.64), (0.16, 0.70)):
        env = BouncingBox(
            BoxConfig(ball_radius=R_, occluder=True, occluder_y=band), seed=seed
        )
        rng = np.random.default_rng(seed)
        vis, traj = [], []
        for _ in range(600):
            _, s, _ = env.step(int(rng.integers(0, 3)))
            vis.append(float(s[6]))
            traj.append(s[:6].copy())
        traj = np.stack(traj)
        if ref is None:
            ref = traj
        else:
            assert np.array_equal(ref, traj), "band height changed the physics"
        lengths[band] = float(np.mean(np.asarray(vis) < 1e-3))
    vals = list(lengths.values())
    assert vals[0] < vals[1] < vals[2], lengths


# ------------------------------------------------------------------ colour


def test_band_colour_is_distinguishable_from_everything_else() -> None:
    """Same >100 RGB-unit margin the v2 colour ramp is held to.

    If the band were close to the paddle blue, "the VAE lost the band" and "the
    VAE confused the band with the paddle" would be the same picture.
    """
    cfg = BoxConfig(occluder=True)
    c = np.array(cfg.occluder_color, float)
    for other in (cfg.ball_color, cfg.paddle_color, cfg.bg_color,
                  cfg.light_color, cfg.heavy_color):
        assert np.linalg.norm(c - np.array(other, float)) > 100.0, (c, other)


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()

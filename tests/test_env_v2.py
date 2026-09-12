"""Tests for the v2 environment (mass from colour).

Runnable two ways (pytest is not guaranteed to be installed here)::

    python -m tests.test_env_v2
    pytest tests/test_env_v2.py

Every test is a plain zero-argument function named ``test_*``.

The most important test in this file is the first one. v2 adds a branch to the
environment's hottest code path (reset, renormalise, the paddle impulse), and if
any of those branches leaks into the default config then every v1 dataset,
checkpoint and result silently stops matching the code that produced it. So we
do not test "v1 still roughly works" -- we replay the collector's exact seeding
and assert the frames are equal to the ones on disk, pixel for pixel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsim.bouncing_box import (
    EVENT_PADDLE,
    STATE_NAMES,
    STATE_NAMES_V2,
    BouncingBox,
    BoxConfig,
    color_to_mass,
    mass_to_color,
    mass_to_u,
    u_to_mass,
)
from worldsim.policies import sticky_random_actions

REPO = Path(__file__).resolve().parents[1]
V1_VAL = REPO / "data" / "v1" / "val"


# ------------------------------------------------------------ v1 regression


def test_v1_frames_byte_identical() -> None:
    """Default BoxConfig() reproduces data/v1/val frame-for-frame.

    Mirrors ``worldsim.collect.collect``: one rng seeded with --seed, from
    which each episode draws its own env seed, and from which (after the reset)
    the open-loop sticky action sequence is drawn. Getting that order wrong
    gives you a plausible-looking but different episode, which is exactly the
    kind of silent break this test exists to catch.
    """
    if not (V1_VAL / "frames.npy").exists():
        print("  (skipped: data/v1/val not present)")
        return

    meta = json.loads((V1_VAL / "meta.json").read_text())
    ref = np.load(V1_VAL / "frames.npy", mmap_mode="r")

    cfg = BoxConfig(res=meta["res"], ball_radius=meta["config"]["ball_radius"])
    assert cfg.mass_from_color is False, "default config must still be v1"

    env = BouncingBox(cfg)
    rng = np.random.default_rng(meta["seed"])

    n = 5  # frames[0, :5] is enough: any divergence shows up in frame 1
    e = 0
    frame0 = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    acts = sticky_random_actions(meta["steps"], rng, mean_hold=meta["mean_hold"])
    got = [frame0]
    for t in range(n - 1):
        f, _, _ = env.step(int(acts[t]))
        got.append(f)
    got = np.stack(got)

    assert np.array_equal(got, np.asarray(ref[e, :n])), "v1 frames changed!"
    assert env.state().shape == (6,)
    assert env.mass == 1.0 and env.speed == cfg.ball_speed


def test_state_width_matches_flag() -> None:
    assert BouncingBox(BoxConfig()).state().shape == (6,)
    assert BouncingBox(BoxConfig(mass_from_color=True)).state().shape == (7,)
    assert BouncingBox(BoxConfig()).state_names == STATE_NAMES
    assert BouncingBox(BoxConfig(mass_from_color=True)).state_names == STATE_NAMES_V2


# ---------------------------------------------------------- colour <-> mass


def test_color_mass_round_trip() -> None:
    cfg = BoxConfig(mass_from_color=True)
    for m in np.exp(np.linspace(np.log(cfg.mass_min), np.log(cfg.mass_max), 41)):
        c = mass_to_color(float(m), cfg)
        assert all(0 <= v <= 255 for v in c)
        back = color_to_mass(c, cfg)
        # Colours are stored as integer RGB, so the round trip is limited by
        # quantisation, not by the maths. ~0.5% is what 8-bit channels buy you.
        assert abs(back - m) / m < 0.01, (m, c, back)


def test_color_map_endpoints_and_monotone() -> None:
    cfg = BoxConfig(mass_from_color=True)
    assert mass_to_color(cfg.mass_min, cfg) == cfg.light_color
    assert mass_to_color(cfg.mass_max, cfg) == cfg.heavy_color
    # The geometric midpoint is m = 1 and must be exactly the v1 ball colour.
    assert abs(mass_to_u(1.0, cfg) - 0.5) < 1e-12
    assert mass_to_color(1.0, cfg) == cfg.ball_color

    # Monotone: the red channel decreases all the way along the ramp, so
    # "redder = lighter" holds globally and the map has no fold.
    reds = [mass_to_color(u_to_mass(u, cfg), cfg)[0] for u in np.linspace(0, 1, 50)]
    assert all(b <= a for a, b in zip(reds, reds[1:]))


def test_ball_colour_far_from_paddle_and_background() -> None:
    """Every ball on the ramp must be easy to tell from the paddle and the bg.

    Not pedantry: if a heavy ball were nearly paddle-coloured, a "the VAE lost
    the colour" failure would be indistinguishable from "the VAE confused the
    ball with the paddle", and the v2 result would be unreadable.
    """
    cfg = BoxConfig(mass_from_color=True)
    for u in np.linspace(0, 1, 50):
        c = np.array(mass_to_color(u_to_mass(float(u), cfg), cfg), float)
        for other in (cfg.paddle_color, cfg.bg_color):
            assert np.linalg.norm(c - np.array(other, float)) > 100.0, (u, c, other)


# --------------------------------------------------------------- v2 physics


def test_effective_speed_conserved_per_episode() -> None:
    cfg = BoxConfig(mass_from_color=True, ball_radius=0.08)
    for seed in range(8):
        env = BouncingBox(cfg, seed=seed)
        m = env.mass
        assert cfg.mass_min <= m <= cfg.mass_max
        rng = np.random.default_rng(seed + 100)
        for _ in range(150):
            _, s, _ = env.step(int(rng.integers(0, 3)))
            speed = float(np.hypot(s[2], s[3]))
            # The invariant is speed * mass == ball_speed, i.e. the same
            # impulse budget for every ball.
            assert abs(speed * m - cfg.ball_speed) < 1e-7
            assert abs(s[6] - m) < 1e-7  # mass column is constant in-episode


def test_colour_in_frame_decodes_to_the_sampled_mass() -> None:
    """Closing the loop: mass -> colour -> pixels -> colour -> mass.

    Reads the pixel nearest the ball centre, which is fully covered by the ball
    and therefore carries the exact ramp colour with no antialiasing blend.
    """
    cfg = BoxConfig(mass_from_color=True, ball_radius=0.08, res=64)
    for seed in range(10):
        env = BouncingBox(cfg, seed=seed)
        f = env.render()
        j = int(env.ball[0] * cfg.res)
        i = int((1.0 - env.ball[1]) * cfg.res)
        px = f[min(i, cfg.res - 1), min(j, cfg.res - 1)]
        assert abs(color_to_mass(px, cfg) - env.mass) / env.mass < 0.02


def test_english_scales_as_one_over_mass() -> None:
    """Construct a real paddle contact and check the impulse is divided by m.

    Rather than reaching into ``_collide_paddle``, we put a light, a medium and
    a heavy ball in the *same* geometric situation -- same position, same
    direction, same paddle sweep -- and compare the resulting sideways velocity.

    There is a genuine subtlety here, and it is worth stating because it is the
    kind of thing you only notice by writing the test. Both v2 effects carry a
    1/m, so in the post-bounce *direction* they cancel exactly:

        tan(theta) = vx / vy = (english * paddle_vx / m) / (ball_speed / m)
                   = english * paddle_vx / ball_speed          [no m]

    and renormalisation rescales both components, preserving the ratio. So a
    heavy ball leaves the paddle at the SAME ANGLE as a light one; what differs
    is that it leaves slower, so the absolute sideways velocity -- the thing
    that decides where it actually is fifty frames later -- still scales as
    1/m. That is the invariant asserted below.
    """
    cfg = BoxConfig(mass_from_color=True, ball_radius=0.08)

    def kick(mass: float):
        env = BouncingBox(cfg, seed=0)
        # Override the sampled episode: same geometry, different mass only.
        env.mass = mass
        env.speed = cfg.ball_speed / mass
        env.paddle_x = 0.5
        # Drop the ball straight down onto the middle of the paddle, close
        # enough that the contact lands within this step's substeps.
        env.ball[:] = (0.5, cfg.paddle_y + cfg.paddle_h / 2.0 + cfg.ball_radius + 0.004)
        env.ball_v[:] = (0.0, -env.speed)
        _, s, ev = env.step(2)  # RIGHT: paddle sweeps right under the ball
        assert ev & EVENT_PADDLE, "no contact -- test geometry is wrong"
        return float(s[2]), float(s[2] / abs(s[3]))  # vx, tan(theta)

    vx_l, tan_l = kick(0.5)
    vx_1, tan_1 = kick(1.0)
    vx_h, tan_h = kick(2.0)

    assert vx_l > vx_1 > vx_h > 0.0             # lighter ball is kicked harder
    assert abs(vx_l * 0.5 - vx_1 * 1.0) < 1e-6  # vx * m is the same impulse
    assert abs(vx_h * 2.0 - vx_1 * 1.0) < 1e-6
    # ...and the angle is mass-independent, as derived above.
    assert abs(tan_l - tan_1) < 1e-6 and abs(tan_h - tan_1) < 1e-6


# ------------------------------------------------------------- mass sampling


def test_mass_distribution_is_log_uniform() -> None:
    cfg = BoxConfig(mass_from_color=True)
    env = BouncingBox(cfg, seed=0)
    ms = np.array([BouncingBox(cfg, seed=s).mass for s in range(1200)])
    assert ms.min() >= cfg.mass_min and ms.max() <= cfg.mass_max
    u = np.log(ms / cfg.mass_min) / np.log(cfg.mass_max / cfg.mass_min)
    # u should be ~Uniform(0,1): mean 0.5, and roughly flat deciles.
    assert abs(u.mean() - 0.5) < 0.03
    counts, _ = np.histogram(u, bins=10, range=(0, 1))
    assert counts.min() > 0.6 * counts.mean()
    assert env.mass > 0


def test_holdout_and_only_bands() -> None:
    lo, hi = 0.85, 1.2
    cfg_h = BoxConfig(mass_from_color=True, mass_holdout=(lo, hi))
    ms = np.array([BouncingBox(cfg_h, seed=s).mass for s in range(400)])
    assert not ((ms >= lo) & (ms <= hi)).any()

    cfg_o = BoxConfig(mass_from_color=True, mass_only=(lo, hi))
    ms = np.array([BouncingBox(cfg_o, seed=s).mass for s in range(400)])
    assert ((ms >= lo) & (ms <= hi)).all()

    # The bands must be inert in v1 mode, so a stray flag cannot alter v1 data.
    assert BouncingBox(BoxConfig(mass_holdout=(lo, hi)), seed=0).mass == 1.0


def test_mass_flags_ignored_without_the_v2_flag() -> None:
    a = BouncingBox(BoxConfig(ball_radius=0.08), seed=3).render()
    b = BouncingBox(BoxConfig(ball_radius=0.08, mass_holdout=(0.85, 1.2)), seed=3).render()
    assert np.array_equal(a, b)


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _main()

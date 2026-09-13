"""Tests for the v4 environment (the gravity switch).

Runnable two ways (pytest is not guaranteed to be installed here)::

    python -m tests.test_env_v4
    pytest tests/test_env_v4.py

The shape of this file mirrors ``tests/test_env_v3.py``, and for the same
reason. v4 adds a branch to ``reset``, a branch to ``step``, a branch to
``state`` and a branch to ``_collide_paddle``, plus four fields on
``BoxConfig``. Any one of them leaking into the default config silently
invalidates every v1 dataset and checkpoint, and a leak into the
``mass_from_color`` or ``occluder`` path does the same to v2/v3/v3.1. So the
first test does not check that the earlier worlds "still roughly work" -- it
replays the collector's exact seeding for all four and compares against the
frames on disk, pixel for pixel.

There is one new and easily-missed way to break them, which is why
``test_v1_launch_rule_is_reproduced_exactly`` exists separately from the
frame replay. The launch-angle guard became a parameter, and the parameter's
default (14.5°) does not have ``sin(14.5°) == 0.25``. Since the guard is a
*rejection sampler*, a threshold 4e-4 too high does not perturb an angle -- it
rejects a draw v1 accepted, shifts the rng stream by one, and produces a
completely different, completely plausible episode. That happens on about one
reset in 2,500, which is often enough to corrupt a dataset and rare enough to
survive a five-frame spot check.

After that the file tests what is genuinely new:

* free flight accelerates by exactly ``gravity`` per frame, in the right
  direction;
* a paddle contact flips the sign, sets ``EVENT_FLIP``, and does both exactly
  once per frame;
* the speed stays bounded with no renormalisation in sight;
* the state gains one column, last, for every combination of switches;
* the sign-blind oracle and the ballistic oracle agree exactly when the true
  sign is the one the blind oracle assumes -- which is what makes the gap
  between them in the design sweep attributable to the sign and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from unittest import SkipTest  # noqa: E402
from tests.conftest import require_ckpt, require_data  # noqa: E402
from worldsim.bouncing_box import (
    EVENT_FLIP,
    EVENT_PADDLE,
    LAUNCH_MIN_SIN_V1,
    STATE_NAMES,
    STATE_NAMES_V2,
    STATE_NAMES_V3,
    STATE_NAMES_V4,
    BouncingBox,
    BoxConfig,
    launch_min_sin,
)
from worldsim.policies import sticky_random_actions
from wm.controller import (
    BallisticOracleController,
    SignBlindOracleController,
    ballistic_landing_x,
)

REPO = Path(__file__).resolve().parents[1]
G = 1e-4
R = 0.08


# ------------------------------------------------- v1 / v2 / v3 regression


def _replay(root: Path, cfg: BoxConfig, n_episodes: int = 4, n: int = 10) -> None:
    """Reproduce ``frames[:n_episodes, :n]`` of a collected split from meta.json.

    Mirrors ``worldsim.collect.collect``: one rng seeded with --seed, from
    which each episode draws its own env seed, and from which (after the reset)
    the open-loop sticky action sequence is drawn. Several episodes rather than
    one because the failure this guards against is a *stream* shift, which by
    construction does not show up until some episode rejects a draw differently.
    """
    meta = json.loads((root / "meta.json").read_text())
    env = BouncingBox(cfg)
    rng = np.random.default_rng(meta["seed"])
    ref = np.load(root / "frames.npy", mmap_mode="r")
    for e in range(min(n_episodes, ref.shape[0])):
        got = [env.reset(seed=int(rng.integers(0, 2**31 - 1)))]
        acts = sticky_random_actions(meta["steps"], rng, mean_hold=meta["mean_hold"])
        for t in range(n - 1):
            got.append(env.step(int(acts[t]))[0])
        assert np.array_equal(np.stack(got), np.asarray(ref[e, :n])), (
            f"{root.parent.name}/{root.name} episode {e} frames changed!"
        )


def test_earlier_versions_still_byte_identical() -> None:
    """v1, v2, v3 and v3.1 all still reproduce their val sets frame-for-frame."""
    cases = {
        "data/v1/val": {},
        "data/v2/val": dict(mass_from_color=True),
        "data/v3/val": dict(occluder=True),
        "data/v31/val": dict(occluder=True, occluder_y=(0.13, 0.63), paddle_w=0.16),
    }
    ran = 0
    for rel, kw in cases.items():
        root = REPO / rel
        if not (root / "frames.npy").exists():
            print(f"  (skipped: {rel} not present)")
            continue
        meta = json.loads((root / "meta.json").read_text())
        if meta.get("mass_holdout"):
            kw = {**kw, "mass_holdout": tuple(meta["mass_holdout"])}
        if meta.get("mass_only"):
            kw = {**kw, "mass_only": tuple(meta["mass_only"])}
        cfg = BoxConfig(
            res=meta["res"], ball_radius=meta["config"]["ball_radius"], **kw
        )
        assert cfg.gravity == 0.0, "default config must still be gravity-free"
        _replay(root, cfg)
        ran += 1
    if ran == 0:
        require_data(REPO / "data" / "v1" / "val" / "frames.npy",
                     "at least one earlier val split is needed to replay")
    print(f"  ({ran} earlier splits replayed byte-identically)")


def test_v1_launch_rule_is_reproduced_exactly() -> None:
    """The default launch guard IS ``|sin| > 0.25``, not ``|sin| > sin(14.5°)``.

    Checked two ways, because the cheap way is not the convincing one. First
    the threshold itself is the v1 literal. Then 400 resets are replayed
    against a hand-written copy of the v1 rejection loop, sharing one rng --
    so any disagreement about whether to accept a draw shows up as a different
    launch velocity, immediately and loudly.
    """
    assert launch_min_sin(14.5) == LAUNCH_MIN_SIN_V1 == 0.25
    assert launch_min_sin(40.0) == float(np.sin(np.deg2rad(40.0)))

    cfg = BoxConfig(res=64, ball_radius=R)
    env = BouncingBox(cfg)
    for seed in range(400):
        env.reset(seed=seed)
        # The v1 loop, verbatim, on an rng seeded identically and advanced
        # through the same two uniform draws reset() makes before the angle.
        rng = np.random.default_rng(seed)
        rng.uniform(R, 1.0 - R)
        rng.uniform(cfg.paddle_y + cfg.paddle_h / 2.0 + R + 0.05, 1.0 - R)
        while True:
            theta = rng.uniform(0.0, 2.0 * np.pi)
            if abs(np.sin(theta)) > 0.25 and abs(np.cos(theta)) > 0.25:
                break
        want = cfg.ball_speed * np.array([np.cos(theta), np.sin(theta)])
        assert np.array_equal(env.ball_v, want), f"launch changed at seed {seed}"


def test_v4_launch_angle_is_honoured() -> None:
    """At 40° every launch is between 40° and 75.5° from horizontal.

    The upper bound is not a v4 choice -- it is v1's ``|cos| > 0.25`` rule,
    left where it was. The test pins both so that a future edit tying the two
    bounds to one constant fails here rather than hanging forever (at 50° a
    symmetric rule is unsatisfiable: no angle has |sin| and |cos| both above
    0.766).
    """
    cfg = BoxConfig(res=64, ball_radius=R, gravity=G, launch_min_angle_deg=40.0)
    env = BouncingBox(cfg)
    angles = []
    for seed in range(200):
        env.reset(seed=seed)
        v = env.ball_v / np.linalg.norm(env.ball_v)
        angles.append(np.degrees(np.arcsin(abs(v[1]))))
    angles = np.asarray(angles)
    assert angles.min() > 40.0, angles.min()
    assert angles.max() < 75.53, angles.max()
    # And the energy claim the angle was chosen for: |vy| must exceed the
    # speed needed to climb the box against the pull.
    vy_needed = np.sqrt(2.0 * G * (1.0 - 2.0 * R))
    assert cfg.ball_speed * np.sin(np.deg2rad(40.0)) > vy_needed


# ------------------------------------------------------------- the physics


def test_free_flight_accelerates_by_exactly_gravity() -> None:
    """``dvy`` per frame is ``-gravity`` with the sign down, ``+gravity`` up.

    Read off ``env.ball_v`` rather than ``env.state()``: the state array is
    float32, which cannot represent a 1e-4 change to a 2e-2 velocity to better
    than about 2%, and a test that tolerated 2% would not notice a missing
    substep (a 25% error).
    """
    for init, want in (("down", -G), ("up", +G)):
        cfg = BoxConfig(res=64, ball_radius=R, gravity=G,
                        gravity_sign_init=init, launch_min_angle_deg=40.0)
        env = BouncingBox(cfg)
        env.reset(seed=11)
        assert env.gravity_sign == np.sign(want)
        prev = float(env.ball_v[1])
        n = 0
        for _t in range(60):
            _f, _s, ev = env.step(1)
            cur = float(env.ball_v[1])
            if ev == 0:                       # free flight: no wall, no paddle
                assert abs((cur - prev) - want) < 1e-15, (cur - prev, want)
                n += 1
            prev = cur
        assert n > 20, f"only {n} free-flight frames -- test has no power"


def test_paddle_contact_flips_the_sign_and_sets_event_flip() -> None:
    """The event and the latent change together, once per frame, always.

    Three claims in one loop, and they are separable failures: the sign could
    flip without the bit (analysis code slicing on ``events.npy`` would see
    nothing), the bit could fire without the flip (the reverse), or a contact
    spanning two substeps could flip twice and so not flip at all.
    """
    cfg = BoxConfig(res=64, ball_radius=R, gravity=G, launch_min_angle_deg=40.0)
    env = BouncingBox(cfg)
    seen = 0
    for seed in range(40):
        env.reset(seed=seed)
        prev_sign = env.gravity_sign
        for _t in range(200):
            # Chase the ball, so contacts are frequent rather than accidental.
            a = 2 if env.ball[0] > env.paddle_x else 0
            _f, s, ev = env.step(a)
            flipped = env.gravity_sign != prev_sign
            assert flipped == bool(ev & EVENT_FLIP), "EVENT_FLIP disagrees with the sign"
            assert bool(ev & EVENT_FLIP) <= bool(ev & EVENT_PADDLE), \
                "a flip without a paddle contact"
            if flipped:
                assert env.gravity_sign == -prev_sign, "sign did not simply invert"
                assert s[-1] == env.gravity_sign, "state column lags the flip"
                seen += 1
            prev_sign = env.gravity_sign
    assert seen > 30, f"only {seen} flips observed -- test has no power"


def test_no_flip_bit_when_gravity_is_off() -> None:
    """EVENT_FLIP must never appear in a v1/v2/v3 dataset's event mask."""
    env = BouncingBox(BoxConfig(res=64, ball_radius=R))
    env.reset(seed=5)
    mask = 0
    for _t in range(400):
        mask |= env.step(2 if env.ball[0] > env.paddle_x else 0)[2]
    assert not (mask & EVENT_FLIP)


def test_speed_stays_bounded_without_renormalisation() -> None:
    """Speed varies (it must) but never exceeds ``max_speed``.

    The upper bound is the assertion; the *lower* claim -- that speed really
    does vary -- matters just as much, because a v4 that quietly kept
    renormalising would pass a bound check trivially and have no gravity in it.
    """
    cfg = BoxConfig(res=64, ball_radius=R, gravity=G, launch_min_angle_deg=40.0)
    env = BouncingBox(cfg)
    lo, hi = np.inf, 0.0
    for seed in range(20):
        env.reset(seed=seed)
        for _t in range(200):
            env.step(2 if env.ball[0] > env.paddle_x else 0)
            sp = float(np.linalg.norm(env.ball_v))
            lo, hi = min(lo, sp), max(hi, sp)
    assert hi <= cfg.max_speed + 1e-12, hi
    assert lo > 0.0
    assert hi / lo > 1.2, f"speed barely varied ({lo:.4f}..{hi:.4f})"


# -------------------------------------------------------------- state shape


def test_state_width_for_every_switch_combination() -> None:
    """All eight (mass, occluder, gravity) combinations, and the column ORDER."""
    for mass in (False, True):
        for occ in (False, True):
            for grav in (0.0, G):
                cfg = BoxConfig(res=64, ball_radius=R, mass_from_color=mass,
                                occluder=occ, gravity=grav,
                                launch_min_angle_deg=40.0 if grav else 14.5)
                env = BouncingBox(cfg)
                env.reset(seed=0)
                want = list(STATE_NAMES)
                if mass:
                    want.append("mass")
                if occ:
                    want.append("ball_visible")
                if grav:
                    want.append("gravity_sign")
                assert list(env.state_names) == want, (mass, occ, grav)
                assert env.state().shape == (len(want),)
                if grav:
                    assert abs(env.state()[-1]) == 1.0
    # The published name tuples, so a consumer can key off them directly.
    assert STATE_NAMES_V2 == STATE_NAMES + ("mass",)
    assert STATE_NAMES_V3 == STATE_NAMES + ("ball_visible",)
    assert STATE_NAMES_V4 == STATE_NAMES + ("gravity_sign",)


def test_gravity_is_inert_at_zero() -> None:
    """``gravity=0`` with a v4-shaped config is bit-identical to plain v1."""
    a = BouncingBox(BoxConfig(res=64, ball_radius=R))
    b = BouncingBox(BoxConfig(res=64, ball_radius=R, gravity=0.0,
                              gravity_sign_init="down", max_speed=0.05))
    fa, fb = a.reset(seed=7), b.reset(seed=7)
    assert np.array_equal(fa, fb)
    for t in range(120):
        act = t % 3
        ra, rb = a.step(act), b.step(act)
        assert np.array_equal(ra[0], rb[0]) and np.array_equal(ra[1], rb[1])
        assert ra[2] == rb[2]


# -------------------------------------------------------------- the oracles


def test_sign_blind_oracle_equals_ballistic_oracle_when_the_sign_is_down() -> None:
    """The two differ in exactly one thing, so when that thing agrees, they do.

    This is what licenses reading the sweep's oracle-to-blind gap as the price
    of the hidden bit: same solver, same dead zone, same geometry. If this test
    failed, part of the gap would be an implementation difference and the whole
    design argument would be unfounded.
    """
    kw = dict(gravity=G, paddle_w=0.26, ball_radius=R, paddle_h=0.045)
    full = BallisticOracleController(**kw)
    blind = SignBlindOracleController(**kw)
    rng = np.random.default_rng(0)
    n_down = 0
    states = []
    for _ in range(500):
        sign = -1.0 if rng.random() < 0.5 else 1.0
        n_down += sign < 0
        states.append([
            rng.uniform(R, 1 - R), rng.uniform(R, 1 - R),
            rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03),
            rng.uniform(0.13, 0.87), 0.0, sign,
        ])
    s = np.asarray(states)
    a_full, a_blind = full.act(None, None, state=s), blind.act(None, None, state=s)
    down = s[:, -1] < 0
    assert np.array_equal(a_full[down], a_blind[down]), \
        "the two oracles disagree where they cannot"
    # ... and they must genuinely disagree somewhere, or the "gap" the sweep
    # measures would be zero for a trivial reason.
    assert (a_full[~down] != a_blind[~down]).any(), "the sign never mattered at all"
    assert 150 < n_down < 350


def test_ballistic_landing_matches_the_simulator() -> None:
    """The analytic landing x is the one the simulator actually reaches.

    Rolled forward with STAY and a paddle parked in a corner, so nothing
    intervenes. The tolerance is one ball radius: the simulator resolves wall
    bounces at quarter-frame granularity and the solver does not, and over a
    60-frame flight with two wall bounces that is worth a few hundredths.
    """
    cfg = BoxConfig(res=64, ball_radius=R, gravity=G,
                    gravity_sign_init="down", launch_min_angle_deg=40.0)
    y_land = cfg.paddle_h + R
    errs = []
    for seed in range(60):
        env = BouncingBox(cfg)
        env.reset(seed=seed)
        # Park the paddle hard left and keep it there, so the ball cannot be
        # deflected before it lands.
        for _ in range(40):
            env.step(0)
        s = np.concatenate([env.state()[:6], [env.gravity_sign]])
        pred = ballistic_landing_x(s, G, env.gravity_sign, R, cfg.paddle_h)
        prev_y = float(env.ball[1])
        for _t in range(400):
            env.step(0)
            y = float(env.ball[1])
            if prev_y > y_land >= y:            # the downward crossing
                errs.append(abs(float(env.ball[0]) - pred))
                break
            prev_y = y
    errs = np.asarray(errs)
    assert len(errs) > 40, f"only {len(errs)} landings found"
    assert np.median(errs) < R, f"median landing error {np.median(errs):.3f}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except SkipTest as exc:
                print(f"skip {name}: {exc}")
    print("all v4 environment tests passed or skipped")

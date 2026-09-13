"""Tests for the v4.1 side-wind (``BoxConfig.gravity_axis``).

    python -m tests.test_env_v41
    pytest tests/test_env_v41.py

``gravity_axis`` adds a branch to the one line of ``step`` that the whole v4
tier rests on, and a whole second code path to ``ballistic_landing_x``. The
risk is entirely in the *default*: if ``gravity_axis="y"`` is not literally the
v4 arithmetic, then every v4 dataset, every v4 number in ``runs/v4_design`` and
every checkpoint trained on them silently stops reproducing. So the first test
here is not "axis y still works" but "axis y is bit-identical to a config that
has never heard of the axis", checked on frames, states and events together.

(``tests/test_env_v4.py`` keeps the v1/v2/v3 byte-identity replays. This file
does not repeat them; it pins the v4-to-v4.1 seam, which is the new one.)

After that, the genuinely new claims:

* under axis x the wind accelerates ``vx`` by exactly ``gravity`` per frame of
  free flight and leaves ``vy`` untouched -- the second half matters more than
  the first, because "vertical motion is exactly v1's" is the entire reason the
  side-wind was proposed (it is what removes the per-traverse energy cue);
* the sign still flips on paddle contact and still sets ``EVENT_FLIP``;
* the axis-x ballistic solver really does predict where the ball lands;
* the sign-blind oracle is the ballistic oracle when the true sign is -1, under
  axis x as under axis y -- which is what makes the sweep's gap attributable to
  the bit.
"""

from __future__ import annotations

import numpy as np

from worldsim.bouncing_box import (
    EVENT_FLIP,
    EVENT_PADDLE,
    BouncingBox,
    BoxConfig,
)
from wm.controller import (
    BallisticOracleController,
    SignBlindOracleController,
    ballistic_landing_x,
)

G = 1e-4
R = 0.08


# ------------------------------------------------------- the axis-y seam


def test_axis_y_is_the_v4_environment_bit_for_bit() -> None:
    """The default axis must be the v4 world, arithmetic included.

    Two configs: one written the way every existing v4 call site writes it
    (no ``gravity_axis`` at all) and one naming the default explicitly. They
    have to agree on frames, states and events for a long episode under a
    ball-chasing policy -- chasing, so that paddle contacts and therefore
    flips actually happen, since the flip is where a wrong axis would show up
    second.
    """
    kw = dict(res=64, ball_radius=R, gravity=G, launch_min_angle_deg=40.0)
    a = BouncingBox(BoxConfig(**kw))
    b = BouncingBox(BoxConfig(gravity_axis="y", **kw))
    flips = 0
    for seed in (0, 3, 17):
        fa, fb = a.reset(seed=seed), b.reset(seed=seed)
        assert np.array_equal(fa, fb)
        for _t in range(300):
            act = 2 if a.ball[0] > a.paddle_x else 0
            ra, rb = a.step(act), b.step(act)
            assert np.array_equal(ra[0], rb[0]), "frames diverged"
            assert np.array_equal(ra[1], rb[1]), "states diverged"
            assert ra[2] == rb[2], "events diverged"
            flips += bool(ra[2] & EVENT_FLIP)
    assert flips > 5, f"only {flips} flips -- the test never exercised the seam"


def test_axis_x_actually_changes_the_world() -> None:
    """...and the two axes are not accidentally the same thing."""
    kw = dict(res=64, ball_radius=R, gravity=G)
    a = BouncingBox(BoxConfig(gravity_axis="y", **kw))
    b = BouncingBox(BoxConfig(gravity_axis="x", **kw))
    a.reset(seed=1)
    b.reset(seed=1)
    assert np.array_equal(a.ball_v, b.ball_v), "the launch should be identical"
    for _t in range(80):
        a.step(1)
        b.step(1)
    assert not np.allclose(a.ball, b.ball)


def test_bad_axis_is_rejected() -> None:
    for bad in ("z", "Y", "", "xy"):
        try:
            BoxConfig(gravity_axis=bad)
        except ValueError:
            continue
        raise AssertionError(f"gravity_axis={bad!r} was accepted")


# ------------------------------------------------------------ the physics


def test_side_wind_accelerates_vx_and_leaves_vy_alone() -> None:
    """``dvx`` is exactly ``+/-gravity`` per free-flight frame; ``dvy`` is 0.

    ``dvy == 0`` exactly (not approximately) is the claim the whole v4.1
    revision is built on: it is what makes traverse durations, stall rates and
    the ball's height distribution independent of the hidden sign, and so what
    removes the energy-budget cue the v4 sweep found. Read off ``ball_v``
    rather than ``state()``, which is float32 and cannot see a 1e-4 change.
    """
    for init, want in (("down", -G), ("up", +G)):
        cfg = BoxConfig(res=64, ball_radius=R, gravity=G, gravity_axis="x",
                        gravity_sign_init=init)
        env = BouncingBox(cfg)
        env.reset(seed=11)
        assert env.gravity_sign == np.sign(want)
        prev = np.array(env.ball_v, dtype=np.float64)
        n = 0
        for _t in range(120):
            _f, _s, ev = env.step(1)
            cur = np.array(env.ball_v, dtype=np.float64)
            if ev == 0:                       # free flight: no wall, no paddle
                assert abs((cur[0] - prev[0]) - want) < 1e-15, (cur[0] - prev[0])
                assert cur[1] == prev[1], "the side-wind touched vy"
                n += 1
            prev = cur
        assert n > 40, f"only {n} free-flight frames -- test has no power"


def test_vertical_motion_is_independent_of_the_sign() -> None:
    """The same seed under either wind gives the SAME ``ball_y`` trace.

    The stronger form of the test above, and the one that states the design
    claim directly. Runs until the first paddle contact, because a contact
    reflects the ball off a paddle that the wind has moved it onto differently
    -- after that the two worlds legitimately diverge in y as well.
    """
    kw = dict(res=64, ball_radius=R, gravity=2e-4, gravity_axis="x")
    a = BouncingBox(BoxConfig(gravity_sign_init="down", **kw))
    b = BouncingBox(BoxConfig(gravity_sign_init="up", **kw))
    checked = 0
    for seed in range(25):
        a.reset(seed=seed)
        b.reset(seed=seed)
        for _t in range(200):
            _fa, _sa, ea = a.step(1)
            _fb, _sb, eb = b.step(1)
            if (ea | eb) & EVENT_PADDLE:
                break
            assert a.ball[1] == b.ball[1], "vertical motion depends on the wind"
            assert a.ball_v[1] == b.ball_v[1]
            checked += 1
    assert checked > 2000, checked


def test_side_wind_flips_on_contact_and_sets_event_flip() -> None:
    """Same contract as axis y: the bit and the latent change together, once."""
    cfg = BoxConfig(res=64, ball_radius=R, gravity=2e-4, gravity_axis="x")
    env = BouncingBox(cfg)
    seen = 0
    for seed in range(40):
        env.reset(seed=seed)
        prev_sign = env.gravity_sign
        for _t in range(200):
            a = 2 if env.ball[0] > env.paddle_x else 0
            _f, s, ev = env.step(a)
            flipped = env.gravity_sign != prev_sign
            assert flipped == bool(ev & EVENT_FLIP), "EVENT_FLIP disagrees"
            assert bool(ev & EVENT_FLIP) <= bool(ev & EVENT_PADDLE)
            if flipped:
                assert env.gravity_sign == -prev_sign
                assert s[-1] == env.gravity_sign, "state column lags the flip"
                seen += 1
            prev_sign = env.gravity_sign
    assert seen > 30, f"only {seen} flips observed"


def test_side_wind_speed_stays_bounded() -> None:
    """Bounded by the ENERGY cap, which is above ``max_speed``.

    The same subtlety as v4's, moved through ninety degrees: the clip fires at
    the moment of contact, and the wind then does up to ``g (1 - 2r)`` of work
    on the ball before it next touches the paddle. So the true bound is
    ``sqrt(max_speed² + 2 g (1 - 2r))``, which is what
    ``worldsim.collect._sanity_check`` asserts (that formula was already
    axis-agnostic, which is why it needed no change). Asserting ``max_speed``
    itself here would fail on a perfectly correct dataset -- as it did.
    """
    g = 2e-4
    cfg = BoxConfig(res=64, ball_radius=R, gravity=g, gravity_axis="x")
    env = BouncingBox(cfg)
    hi, lo = 0.0, np.inf
    for seed in range(20):
        env.reset(seed=seed)
        for _t in range(200):
            env.step(2 if env.ball[0] > env.paddle_x else 0)
            sp = float(np.linalg.norm(env.ball_v))
            hi, lo = max(hi, sp), min(lo, sp)
    cap = float(np.sqrt(cfg.max_speed**2 + 2.0 * g * (1.0 - 2.0 * R)))
    assert hi <= cap + 1e-9, (hi, cap)
    assert lo > 0.0
    assert hi / lo > 1.2, f"speed barely varied ({lo:.4f}..{hi:.4f})"


# -------------------------------------------------------------- the oracle


def test_side_wind_ballistic_oracle_lands_the_ball() -> None:
    """The axis-x solver predicts the simulator's own landing x to <= 0.01.

    Two things about the measurement, both of which are the difference between
    a meaningful tolerance and a meaningless one.

    *The crossing is interpolated.* The ball moves ~0.02 in x per frame, so
    reading ``ball_x`` at the first frame BELOW contact height compares the
    solver's exact crossing against a position up to a full frame late -- a 0.02
    "error" that is entirely an artefact of sampling. Interpolating the true
    crossing between the bracketing frames removes it, and what is left
    (median 1e-5) is the solver.

    *Episodes with a paddle contact are discarded.* A contact flips the wind,
    so the trajectory the solver extrapolated no longer exists. That is not an
    inaccuracy, it is a different world.
    """
    for g in (1e-4, 2e-4):
        cfg = BoxConfig(res=64, ball_radius=R, gravity=g, gravity_axis="x")
        y_land = cfg.paddle_h + R
        errs = []
        for seed in range(300):
            if len(errs) >= 50:
                break
            env = BouncingBox(cfg)
            env.reset(seed=seed)
            for _ in range(40):           # let the wind build up a real vx
                env.step(1)
            s = np.concatenate([env.state()[:6], [env.gravity_sign]])
            pred = ballistic_landing_x(s, g, env.gravity_sign, R, cfg.paddle_h,
                                       gravity_axis="x")
            py, px = float(env.ball[1]), float(env.ball[0])
            for _t in range(500):
                _f, _st, ev = env.step(1)
                if ev & EVENT_PADDLE:
                    break
                y, x = float(env.ball[1]), float(env.ball[0])
                if py > y_land >= y:
                    f = (py - y_land) / (py - y)
                    errs.append(abs(px + f * (x - px) - pred))
                    break
                py, px = y, x
        errs = np.asarray(errs)
        assert len(errs) >= 50, f"only {len(errs)} clean landings at g={g}"
        assert errs.max() < 0.01, f"g={g}: worst landing error {errs.max():.4f}"


def test_side_wind_sign_blind_equals_ballistic_when_the_sign_is_down() -> None:
    """Under axis x too, the two oracles differ in exactly one thing."""
    kw = dict(gravity=2e-4, paddle_w=0.26, ball_radius=R, paddle_h=0.045,
              gravity_axis="x")
    full = BallisticOracleController(**kw)
    blind = SignBlindOracleController(**kw)
    rng = np.random.default_rng(0)
    states = []
    for _ in range(400):
        sign = -1.0 if rng.random() < 0.5 else 1.0
        states.append([
            rng.uniform(R, 1 - R), rng.uniform(R, 1 - R),
            rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03),
            rng.uniform(0.13, 0.87), 0.0, sign,
        ])
    s = np.asarray(states)
    a_full = full.act(None, None, state=s)
    a_blind = blind.act(None, None, state=s)
    down = s[:, -1] < 0
    assert np.array_equal(a_full[down], a_blind[down]), \
        "the two oracles disagree where they cannot"
    assert (a_full[~down] != a_blind[~down]).any(), "the sign never mattered"


def test_axis_y_landing_solver_is_untouched() -> None:
    """``ballistic_landing_x`` with the default axis is the v4 function.

    Cheap and worth having explicitly: the axis-x branch is an early return, so
    a mistake in where it is placed would silently reroute axis y through the
    substepping solver -- which gives *nearly* the same answer, which is the
    worst kind of regression. Compared against a hand-rolled copy of the v4
    closed form.
    """
    rng = np.random.default_rng(3)
    for _ in range(200):
        st = np.array([rng.uniform(R, 1 - R), rng.uniform(R, 1 - R),
                       rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03),
                       0.5, 0.0, 1.0])
        sign = -1.0 if rng.random() < 0.5 else 1.0
        got = ballistic_landing_x(st, G, sign, R, 0.045)
        want = _v4_landing_x_reference(st, G, sign, R, 0.045)
        assert abs(got - want) < 1e-12, (got, want)


def _v4_landing_x_reference(state, gravity, sign, r, paddle_h, max_bounces=4):
    """The v4 closed form, transcribed, so the test does not call the code
    it is testing. Parabolic y, linear x, one modulo for every side bounce."""
    x, y, vx, vy = (float(state[0]), float(state[1]),
                    float(state[2]), float(state[3]))
    a = float(sign) * float(gravity)
    y_land, y_top = paddle_h + r, 1.0 - r

    def root(y0, vy0, target):
        c = y0 - target
        if abs(a) < 1e-15:
            return None if abs(vy0) < 1e-15 or -c / vy0 <= 1e-9 else -c / vy0
        disc = vy0 * vy0 - 2.0 * a * c
        if disc < 0.0:
            return None
        sq = float(np.sqrt(disc))
        ts = [t for t in ((-vy0 + sq) / a, (-vy0 - sq) / a) if t > 1e-9]
        return min(ts) if ts else None

    def fold(xx):
        w = 1.0 - 2.0 * r
        u = (xx - r) % (2.0 * w)
        return r + (2.0 * w - u if u > w else u)

    for _ in range(max_bounces):
        tl, tc = root(y, vy, y_land), root(y, vy, y_top)
        if tl is not None and (tc is None or tl <= tc):
            return fold(x + vx * tl)
        if tc is None:
            break
        x += vx * tc
        vy = -(vy + a * tc)
        y = y_top
    return fold(x)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all v4.1 side-wind tests passed")

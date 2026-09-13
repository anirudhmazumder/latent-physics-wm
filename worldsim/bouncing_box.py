"""v1/v2: a ball bouncing in a unit box, with a paddle the agent moves left/right.

v2 in one line: the ball's COLOUR tells you its MASS, and mass changes how the
ball moves. See "v2: mass from colour" below. With ``mass_from_color=False``
(the default) this file is bit-for-bit the v1 environment.

Design notes
------------
* World coordinates are the unit square [0, 1] x [0, 1]. Rendering resolution is
  a separate concern, so you can train at 64x64 and inspect at 256x256 without
  touching the physics.
* The integrator is hand-written and substepped. No physics engine, because the
  point is to have exact control over (and exact access to) the generative
  process.
* Ball speed is held constant. The paddle imparts sideways "english" on contact,
  which changes direction but not speed. This keeps the data distribution
  stationary -- no slow energy drift for the dynamics model to chase -- while
  still making the action causally relevant to the ball's future.
* The paddle sits flush on the floor, so the ball can never get wedged
  underneath it. That removes the one genuinely fiddly collision case.
* Rendering is antialiased by analytic pixel coverage. This matters more than it
  looks: hard pixel edges quantise position, and a VAE trained on quantised
  positions gives you a latent space where sub-pixel ball position is simply not
  recoverable. Antialiasing leaves the information in the image.

Coordinate convention for frames: ``img[i, j]`` has x = (j + 0.5) / res and
y = 1 - (i + 0.5) / res. So y = 0 is the bottom of the image, matching physics.

v2: mass from colour
--------------------
At ``reset()`` we draw a mass ``m`` and paint the ball a colour that is a
deterministic, monotone function of ``m``. Mass then feeds back into the
physics in two places, both of the form "same impulse, different mass":

  * speed      ``|v| = ball_speed / m``   (a heavy ball is slow)
  * english    ``dvx = english * paddle_vx / m``  (a heavy ball is hard to nudge)

Why this is the right second environment. In v1 every dynamically relevant
variable was either visible in the frame (positions) or invisible in *any*
single frame (velocities). v2 adds a third kind: a variable that is *visible*
in a single frame but only as **appearance**, and whose consequences are purely
*dynamical*. That is an appearance -> dynamics causal edge. It is the smallest
honest version of "the world looks a certain way and therefore behaves a
certain way", and each stage of the V-M-C stack has a different job with it:

  V  should notice the colour at all (a new active latent dimension).
  M  must learn that this latent multiplies the step size -- a *multiplicative*
     interaction between two latents, which a linear dynamics model cannot do.
  C  can in principle read the colour and lead a fast ball more than a slow one.

Mass is sampled LOG-uniformly in ``[mass_min, mass_max] = [0.5, 2.0]``. Log
rather than linear because mass enters the physics as a ratio (1/m): a linear
prior would make "twice as heavy" and "half as heavy" occur at very different
rates, and the median ball would not be m = 1. Log-uniform makes the
distribution symmetric under m -> 1/m, so u = 0.5 is exactly m = 1 -- the v1
ball -- and the colour ramp is symmetric about the v1 red.

Colour ramp (see ``mass_to_color``): a TWO-SEGMENT path in RGB,
    light yellow (250,220,80) -> v1 red (235,90,70) -> heavy purple (150,40,200)
with the v1 red pinned at u = 0.5, i.e. m = 1. A single straight lerp from
yellow to purple passes through a desaturated dusty pink around the midpoint
(200,130,140), which is muddy on a dark background and is the region where most
of the probability mass sits -- exactly where you least want the colour signal
to be weak. Routing through the v1 red keeps every ball vivid, keeps the whole
ramp far from the blue paddle (70,150,235) and the near-black background, and
has the pleasing property that the median v2 ball is literally the v1 ball.

v3: the occlusion band
----------------------
With ``occluder=True`` an opaque horizontal band is painted across the full
width of the frame, *after* the ball and the paddle, between world heights
``occluder_y = (lo, hi)``. The physics is not touched at all: the ball flies
straight through, bounces off walls behind it, and is simply not drawn.

Why that is the right third environment. v1 hid *velocity* (invisible in any
single frame, inferable from two). v2 hid *mass* behind an appearance cue. v3
hides **position itself**, for a stretch of every vertical traverse. While the
ball is behind the band the frame is pixel-identical for every ball position,
so no encoder can recover the ball from the image -- the information has to be
carried in a recurrent state or not at all. That is object permanence.

Two diagnostics come with it, both for grading only (the model never sees them):

``ball_visible``
    The fraction of the ball's disc area NOT covered by the band, computed
    analytically from the circular-segment formula (no sampling, no pixel
    counting, so it is exact and resolution-independent). 1 far from the band,
    0 when the ball is entirely inside it, and a smooth monotone ramp in
    between as the ball slides in or out.

``EVENT_HIDDEN``
    A bit in the per-step event mask, set when ``ball_visible < 1e-3`` at the
    end of the step. Redundant with the state column by construction, which is
    why one of the tests asserts they agree: it exists so that analysis code
    can slice hidden stretches straight out of ``events.npy`` without loading
    the (much larger) states array or re-deriving the geometry.

v4: the gravity switch
----------------------
With ``gravity > 0`` the ball feels a constant vertical acceleration whose
DIRECTION is a hidden binary latent, ``gravity_sign`` (+1 pulls toward the
ceiling, −1 toward the paddle). It is drawn at reset and **multiplied by −1 on
every paddle contact**, which also sets ``EVENT_FLIP``.

Why that is the right fourth environment. v1 hid velocity (invisible in one
frame, inferable from two). v2 hid mass behind an appearance cue. v3 hid
position for a bounded stretch. In all three the information needed to predict
the next frame was recoverable from a *short window* of recent frames. v4
removes that: at 1e-4 per frame² the acceleration moves the ball by less than
the renderer's 1/64 quantisation over ~15 frames, so no bounded window reveals
the sign — but it holds for the 100+ frames until the next contact and bends
every trajectory in it. The only way to predict is to notice the event and
remember its consequence. That is the regime where "carry a bit in a recurrent
state" and "attend back to the frame where the bit was set" come apart.

Three physics consequences, each of which had to be decided rather than
inherited:

* **Speed is no longer conserved**, so ``_renormalise_velocity`` is OFF when
  gravity is on. Gravity is a conservative force and the walls are elastic, so
  the speed stays bounded anyway (it is a function of height); the only thing
  that can pump energy in is the paddle's english, so the speed is clipped to
  ``max_speed`` after contact. In practice the clip essentially never fires --
  the collector counts and reports how often it does.
* **The ``min_vy_frac`` guard goes with it.** It lives inside
  ``_renormalise_velocity``, so it is inert under gravity by construction. That
  is the right call and not just convenient: the guard exists to stop a ball
  skimming horizontally forever, and under gravity a horizontal ball is pulled
  off the horizontal within a few dozen frames on its own.
* **The launch angle has to be steep enough to cross the box against gravity.**
  ``launch_min_angle_deg`` generalises v1's ``|sin θ| > 0.25`` rejection rule.
  With ``vy² > 2 g Δh`` and Δh = 1 − 2r = 0.84, g = 1e-4 needs |vy| > 0.0130,
  i.e. 36.2° at speed 0.022 -- so v4 launches above 40°, with a little margin.
  The default 14.5° reproduces the v1 rule exactly (see ``launch_min_sin``).

v4.1: the side-wind (``gravity_axis``)
--------------------------------------
The v4 design sweep (``runs/v4_design/sweep.md``) priced the hidden sign with
privileged controllers and found it was worth *nothing*: a wrong sign moves the
predicted landing x by at most 0.035 -- a quarter of a paddle -- and that error
shrinks to zero as the ball arrives, so a bang-bang tracker absorbs it without
ever committing. The reason is geometric, not a matter of the magnitude: a
vertical acceleration perturbs the landing *height*, and height converts into
*x* only through the ball's slow horizontal speed.

``gravity_axis="x"`` turns the acceleration ninety degrees. The flipping term
now acts on ``ball_v[0]``: a **side-wind**, +1 blowing toward +x. Two things
follow, and both are the point:

* the sign's effect on the landing x is first-order rather than second-order.
  It is ``a t² / 2`` in x directly -- ~0.18 over a 60-frame fall at 1e-4, and
  the two hypotheses differ by twice that -- so a wrong sign misses by a whole
  paddle, not a quarter of one.
* **vertical motion is exactly v1's.** ``vy`` is untouched, so traverse
  durations, the stall rate and the ball's height distribution do not depend on
  the sign at all. That removes the one leak the v4 sweep found by accident:
  under vertical gravity the sign also set the ball's energy budget, so a
  single traverse's duration betrayed it (1.20x) without any memory of the
  flip. Under a side-wind the two signs have *identical* vertical dynamics, and
  the sign is only in the horizontal curvature.

Its own new failure mode, which the collector measures: a side-wind can pin the
ball against a side wall (bounce, decelerate, get blown back, bounce again).
``worldsim.collect.wall_pinned_fraction`` reports how much of the time the ball
spends within one radius of a side wall; treat >20% in a cell the way you would
treat stalls.

Under axis x the launch guard goes back to v1's 14.5 degrees: the energy
argument that forced 40 degrees was about climbing against a vertical pull, and
there is no vertical pull any more. ``gravity_sign_init="down"`` reads as
"toward -x" and ``"up"`` as "toward +x"; the state column is still called
``gravity_sign`` under both axes so that every v4 consumer works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

LEFT, STAY, RIGHT = 0, 1, 2
ACTIONS: Tuple[int, int, int] = (LEFT, STAY, RIGHT)
ACTION_NAMES = ("left", "stay", "right")

STATE_NAMES = ("ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x", "paddle_vx")
# v2 appends mass as a 7th column. ``state()`` returns the 6-vector when
# mass_from_color is off, so v1 datasets and every v1 consumer are untouched;
# downstream code should read ``meta["state_names"]`` rather than assume 6.
STATE_NAMES_V2 = STATE_NAMES + ("mass",)
# v3 appends ball_visible LAST, after the (optional) mass column, so the two
# switches compose: the column order is always
#   [ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_vx, (mass), (ball_visible)]
# and a consumer that reads meta["state_names"] never has to know which
# combination of flags produced the file.
STATE_NAMES_V3 = STATE_NAMES + ("ball_visible",)
# v4 appends gravity_sign LAST, after any v2/v3 column, so all three switches
# compose and the column order is always
#   [ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_vx,
#    (mass), (ball_visible), (gravity_sign)]
STATE_NAMES_V4 = STATE_NAMES + ("gravity_sign",)

# Bit flags for the per-step event mask.
EVENT_WALL_X = 1
EVENT_WALL_Y = 2
EVENT_PADDLE = 4
EVENT_HIDDEN = 8  # v3: the ball was fully behind the occluder at the end of the step
EVENT_FLIP = 16   # v4: gravity_sign was multiplied by -1 during this step

# The v1 launch guard, as a literal. ``asin(0.25) = 14.4775°``, which is what
# ``LAUNCH_MIN_ANGLE_V1_DEG`` names (rounded to the 0.1° the flag is specified
# in). See ``launch_min_sin`` for why the constant is pinned rather than
# recomputed from the degrees.
LAUNCH_MIN_SIN_V1 = 0.25
LAUNCH_MIN_ANGLE_V1_DEG = 14.5


def launch_min_sin(deg: float) -> float:
    """``|sin θ|`` threshold for the launch-angle rejection rule.

    Pinned to exactly 0.25 at the v1 default instead of recomputing
    ``sin(14.5°) = 0.250380``. The difference looks negligible and is not: the
    rule is a *rejection sampler*, so a threshold 0.00038 higher rejects a
    sliver of angles v1 accepted, consuming one extra draw from the episode's
    rng and producing a completely different (but equally valid-looking)
    episode. That happens on ~0.04% of resets -- often enough to break one
    episode in a few hundred, rarely enough that you would never find it. Every
    v1/v2/v3 dataset on disk has to replay byte-for-byte, so the v1 constant is
    the v1 constant.
    """
    if abs(float(deg) - LAUNCH_MIN_ANGLE_V1_DEG) < 1e-9:
        return LAUNCH_MIN_SIN_V1
    return float(np.sin(np.deg2rad(float(deg))))


@dataclass
class BoxConfig:
    res: int = 64

    ball_radius: float = 0.055
    ball_speed: float = 0.022  # world units per frame; held constant
    min_vy_frac: float = 0.15  # floor on |vy| / speed, to avoid endless skimming

    paddle_w: float = 0.26
    paddle_h: float = 0.045
    paddle_y: Optional[float] = None  # default: flush on the floor
    paddle_speed: float = 0.030  # world units per frame at full deflection
    english: float = 0.35  # fraction of paddle vx added to ball vx on contact

    substeps: int = 4

    bg_color: Tuple[int, int, int] = (18, 18, 22)
    ball_color: Tuple[int, int, int] = (235, 90, 70)
    paddle_color: Tuple[int, int, int] = (70, 150, 235)

    # ------------------------------------------------------------------ v2
    # All of this is inert while mass_from_color is False, which is why the
    # default BoxConfig() still *is* v1.
    mass_from_color: bool = False
    mass_min: float = 0.5
    mass_max: float = 2.0

    # Endpoints of the colour ramp. The midpoint of the ramp (u = 0.5, m = 1)
    # is ``ball_color``, so the v1 ball is the median v2 ball.
    light_color: Tuple[int, int, int] = (250, 220, 80)   # u = 0, m = mass_min
    heavy_color: Tuple[int, int, int] = (150, 40, 200)   # u = 1, m = mass_max

    # Generalisation knobs, mutually exclusive in practice:
    #   mass_holdout = (lo, hi)  -> never sample a mass inside [lo, hi]
    #   mass_only    = (lo, hi)  -> sample ONLY inside [lo, hi]
    # Train with the hold-out, test with the "only" set, and you have a clean
    # *interpolation* test: the model has seen lighter and heavier balls but
    # never one of exactly this colour. Rejection sampling rather than a
    # re-parameterised distribution because it keeps the accepted samples
    # exactly log-uniform-conditioned-on-the-band, with no arithmetic to get
    # subtly wrong.
    mass_holdout: Optional[Tuple[float, float]] = None
    mass_only: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------ v3
    # Inert while occluder is False, so BoxConfig() is still exactly v1 and
    # BoxConfig(mass_from_color=True) is still exactly v2.
    occluder: bool = False
    occluder_y: Tuple[float, float] = (0.28, 0.58)  # (bottom, top) in world coords
    # A mid grey-blue. Chosen to sit >100 RGB units from all three existing
    # colours (ball, paddle, background) -- the same margin the v2 colour ramp
    # is held to -- while being desaturated and dim enough to read as *scenery*
    # rather than as a second moving object. The nearest neighbour is the
    # paddle blue at ~115 units, but the two differ in saturation and
    # brightness, not just hue, so they are not confusable by eye either.
    occluder_color: Tuple[int, int, int] = (95, 110, 130)

    # ------------------------------------------------------------------ v4
    # Inert while gravity == 0.0, so BoxConfig() is still exactly v1 and every
    # existing (mass, occluder) combination is untouched.
    gravity: float = 0.0            # world units per frame^2; v4 uses 1e-4
    gravity_sign_init: str = "random"   # "random" | "down" | "up"
    # v4.1: WHICH AXIS the flipping acceleration acts on.
    #   "y" -- the original vertical gravity. +1 pulls toward the ceiling,
    #          -1 toward the paddle. Byte-identical to the v4 environment.
    #   "x" -- a horizontal SIDE-WIND. +1 pushes toward +x (right), -1 toward
    #          -x (left). Vertical motion is then exactly v1's, so traverse
    #          times, stall rates and the per-traverse energy cue are v1's too,
    #          while the sign moves the LANDING X by an order of magnitude
    #          more than vertical gravity ever did. See runs/v4_design/
    #          sweep_v41.md for why the axis had to move.
    # The state column is called ``gravity_sign`` under both axes; under axis x
    # it is the WIND's sign. The name is kept so that every v4 consumer, probe
    # and meta.json key works unchanged.
    gravity_axis: str = "y"
    # The launch-angle guard, in degrees from horizontal. 14.5 is the v1 rule
    # (see launch_min_sin); v4 uses 40, which is the shallowest launch whose
    # vertical kinetic energy still carries the ball across the box against
    # gravity pointing the other way.
    launch_min_angle_deg: float = LAUNCH_MIN_ANGLE_V1_DEG
    # Only consulted when gravity > 0, where speed is no longer renormalised.
    # Gravity itself cannot push the ball past ~0.026; this bounds the slow
    # accumulation of english over many paddle contacts. See ``_clip_speed``.
    max_speed: float = 0.05

    def __post_init__(self) -> None:
        if self.paddle_y is None:
            self.paddle_y = self.paddle_h / 2.0
        if self.mass_min <= 0.0 or self.mass_max <= self.mass_min:
            raise ValueError("need 0 < mass_min < mass_max")
        # Tuples, not lists, so a config round-tripped through JSON compares
        # equal and so the dataclass stays hashable-ish in spirit.
        if self.mass_holdout is not None:
            self.mass_holdout = (float(self.mass_holdout[0]), float(self.mass_holdout[1]))
        if self.mass_only is not None:
            self.mass_only = (float(self.mass_only[0]), float(self.mass_only[1]))
        lo, hi = float(self.occluder_y[0]), float(self.occluder_y[1])
        if hi <= lo:
            raise ValueError(
                "occluder_y must be (bottom, top) with top > bottom, got "
                f"{self.occluder_y}"
            )
        self.occluder_y = (lo, hi)
        if self.gravity < 0.0:
            raise ValueError(
                "gravity is a MAGNITUDE (>= 0); the direction lives in "
                f"gravity_sign_init, got gravity={self.gravity}"
            )
        if self.gravity_sign_init not in ("random", "down", "up"):
            raise ValueError(
                'gravity_sign_init must be "random", "down" or "up", got '
                f"{self.gravity_sign_init!r}"
            )
        if self.gravity_axis not in ("x", "y"):
            raise ValueError(
                f'gravity_axis must be "x" or "y", got {self.gravity_axis!r}'
            )


# --------------------------------------------------------------- mass <-> colour
#
# Kept as free functions taking a config, not methods, because analysis code
# wants to go from a *pixel* back to a mass without instantiating an
# environment (e.g. "what mass did the VAE just draw?").


def _ramp_points(cfg: "BoxConfig") -> np.ndarray:
    """The (3, 3) polyline light -> v1 red -> heavy, as float RGB."""
    return np.array(
        [cfg.light_color, cfg.ball_color, cfg.heavy_color], dtype=np.float64
    )


def mass_to_u(m: float, cfg: "BoxConfig") -> float:
    """Position along the ramp: u = log(m/mass_min) / log(mass_max/mass_min)."""
    return float(
        np.log(m / cfg.mass_min) / np.log(cfg.mass_max / cfg.mass_min)
    )


def u_to_mass(u: float, cfg: "BoxConfig") -> float:
    return float(cfg.mass_min * (cfg.mass_max / cfg.mass_min) ** u)


def mass_to_color(m: float, cfg: Optional["BoxConfig"] = None) -> Tuple[int, int, int]:
    """Deterministic, monotone colour for a mass. u = 0.5 gives the v1 ball.

    Rounded to integer RGB on purpose: that is what actually lands in the
    frame, so the inverse map below is inverting the real thing rather than an
    idealised continuous one.
    """
    cfg = cfg or BoxConfig()
    u = float(np.clip(mass_to_u(m, cfg), 0.0, 1.0))
    pts = _ramp_points(cfg)
    # Two equal halves of u: [0, .5] light->mid, [.5, 1] mid->heavy.
    if u <= 0.5:
        c = pts[0] + (pts[1] - pts[0]) * (u / 0.5)
    else:
        c = pts[1] + (pts[2] - pts[1]) * ((u - 0.5) / 0.5)
    return tuple(int(round(v)) for v in c)  # type: ignore[return-value]


def color_to_mass(color, cfg: Optional["BoxConfig"] = None) -> float:
    """Inverse of ``mass_to_color``: nearest point on the ramp, then u -> m.

    Implemented as a projection onto the polyline rather than by inverting one
    channel. Two reasons: it is robust to the integer rounding above, and it
    degrades gracefully on colours that are near but not on the ramp -- e.g. an
    antialiased edge pixel, or a VAE reconstruction. Uses all three channels,
    which matters because the red channel alone changes by only 15 units over
    the whole light->red segment.
    """
    cfg = cfg or BoxConfig()
    c = np.asarray(color, dtype=np.float64)
    pts = _ramp_points(cfg)

    best = (np.inf, 0.0)
    for k in (0, 1):  # segment k covers u in [k/2, (k+1)/2]
        a, b = pts[k], pts[k + 1]
        d = b - a
        denom = float(d @ d)
        t = 0.0 if denom < 1e-12 else float(np.clip((c - a) @ d / denom, 0.0, 1.0))
        resid = float(np.sum((a + t * d - c) ** 2))
        if resid < best[0]:
            best = (resid, 0.5 * (k + t))
    return u_to_mass(best[1], cfg)


# ---------------------------------------------------------------- v3 geometry
#
# Free function for the same reason as the colour maps: analysis code wants to
# ask "how much of this ball was visible" about a ball position it read out of
# a states array, or predicted, without instantiating an environment.


def _disc_area_below(cy: float, r: float, line_y: float) -> float:
    """Area of the disc (centre ``cy``, radius ``r``) lying below ``line_y``.

    Circular-segment formula, written so the two degenerate cases fall out of
    the clamp rather than needing branches::

        A(d) = pi r^2 / 2 + r^2 asin(d/r) + d sqrt(r^2 - d^2),   d = line_y - cy

    with A(-r) = 0 and A(+r) = pi r^2 exactly.
    """
    d = float(np.clip(line_y - cy, -r, r))
    return float(
        0.5 * np.pi * r * r + r * r * np.arcsin(d / r) + d * np.sqrt(max(r * r - d * d, 0.0))
    )


def visible_fraction(ball_y: float, r: float, band: Tuple[float, float]) -> float:
    """Fraction of a ball's disc area NOT covered by a full-width band.

    Exact and analytic, so it does not depend on the render resolution -- which
    matters, because ``ball_visible`` is a *grading* variable and we do not want
    the grading to move when someone renders at 128 instead of 64. The only
    approximation left is that the renderer antialiases the ball's edge, so at
    the extreme ends of the ramp a pixel or two of fringe can survive at
    ``ball_visible`` slightly below 1e-3; that is what makes the EVENT_HIDDEN
    threshold a threshold rather than an exact zero (see the note there).
    """
    lo, hi = band
    # Short-circuit the no-overlap case rather than letting it fall out of the
    # arithmetic. The segment formula does give ~1 there, but only to within a
    # few ulps -- the two halves of A(-r) = pi r^2 / 2 + r^2 asin(-1) associate
    # their multiplications differently -- and "exactly 1.0 when the ball is
    # clear of the band" is a property downstream code and the tests rely on.
    if ball_y - r >= hi or ball_y + r <= lo:
        return 1.0
    covered = _disc_area_below(ball_y, r, hi) - _disc_area_below(ball_y, r, lo)
    return float(np.clip(1.0 - covered / (np.pi * r * r), 0.0, 1.0))


class BouncingBox:
    """Deterministic given a seed. ``step`` advances one frame."""

    def __init__(self, config: Optional[BoxConfig] = None, seed: Optional[int] = None):
        self.cfg = config or BoxConfig()
        self.rng = np.random.default_rng(seed)
        self._grid: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._grid_res: Optional[int] = None
        # v4.1: 0 = horizontal side-wind, 1 = vertical gravity (the v4 world).
        self._grav_axis = 0 if self.cfg.gravity_axis == "x" else 1
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        r = cfg.ball_radius

        # v2: mass first, because it decides both the colour and the speed.
        # Drawn BEFORE any other rng use only when mass_from_color is on, so
        # the v1 stream of random numbers (position, angle, paddle) is
        # byte-identical to before when the flag is off.
        self.mass = self._sample_mass()
        self.ball_color = (
            mass_to_color(self.mass, cfg) if cfg.mass_from_color else cfg.ball_color
        )
        # The *effective* speed. Every place that used cfg.ball_speed in v1 now
        # uses this; with m = 1 it is exactly cfg.ball_speed (division by 1.0
        # is exact in IEEE754, so v1 reproduces bit-for-bit).
        self.speed = cfg.ball_speed / self.mass

        # v4: the hidden latent. Drawn here, immediately after the mass, and --
        # like the mass -- consuming NO randomness when the switch is off, so
        # the v1/v2/v3 rng streams are untouched. ``speed_clips`` is a
        # diagnostic counter that survives resets (the collector reports the
        # total over a whole split).
        self.gravity_sign = self._sample_gravity_sign()
        self.speed_clips = getattr(self, "speed_clips", 0)

        y_lo = cfg.paddle_y + cfg.paddle_h / 2.0 + r + 0.05
        self.ball = np.array(
            [self.rng.uniform(r, 1.0 - r), self.rng.uniform(y_lo, 1.0 - r)],
            dtype=np.float64,
        )

        # Reject launch angles too close to an axis -- those trajectories are
        # degenerate and take a long time to explore the box.
        #
        # v4 raises the floor on |sin| only. The two bounds do different jobs
        # and only one of them is a v4 concern: |sin θ| keeps the launch off
        # HORIZONTAL, which under gravity is also what guarantees the ball can
        # cross the box against the pull; |cos θ| keeps it off VERTICAL, which
        # is purely about exploring x and stays at the v1 value. Tying both to
        # one raised constant would be worse than cosmetic -- at 50° the two
        # bounds become |sin| > 0.766 AND |cos| > 0.766, which no angle
        # satisfies, and the sampler would spin forever.
        min_sin = launch_min_sin(cfg.launch_min_angle_deg)
        while True:
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            if abs(np.sin(theta)) > min_sin and abs(np.cos(theta)) > LAUNCH_MIN_SIN_V1:
                break
        self.ball_v = self.speed * np.array(
            [np.cos(theta), np.sin(theta)], dtype=np.float64
        )

        hw = cfg.paddle_w / 2.0
        self.paddle_x = float(self.rng.uniform(hw, 1.0 - hw))
        self.paddle_vx = 0.0
        self.t = 0
        self._events = 0
        return self.render()

    def _sample_mass(self) -> float:
        """Log-uniform in [mass_min, mass_max], honouring holdout / only bands.

        Returns exactly 1.0 (and consumes no randomness) in v1 mode.
        """
        cfg = self.cfg
        if not cfg.mass_from_color:
            return 1.0
        lo, hi = np.log(cfg.mass_min), np.log(cfg.mass_max)
        for _ in range(10_000):
            m = float(np.exp(self.rng.uniform(lo, hi)))
            if cfg.mass_holdout is not None and cfg.mass_holdout[0] <= m <= cfg.mass_holdout[1]:
                continue
            if cfg.mass_only is not None and not (cfg.mass_only[0] <= m <= cfg.mass_only[1]):
                continue
            return m
        # Only reachable if the band is empty or absurdly narrow; better a loud
        # failure at collection time than a dataset with a silent bias.
        raise RuntimeError(
            "mass rejection sampling failed -- check mass_holdout / mass_only "
            "against [mass_min, mass_max]"
        )

    def _sample_gravity_sign(self) -> float:
        """v4: +1 (pulls up) / -1 (pulls down). Returns 1.0 and draws nothing
        in v1/v2/v3 mode, which is what keeps those rng streams identical.

        "down" and "up" exist for the tests and for ``worldsim.play``: a fixed
        sign turns the flip experiment into an ordinary constant-gravity world,
        which is the control you want when you are checking the integrator
        rather than the memory.
        """
        cfg = self.cfg
        if cfg.gravity <= 0.0:
            return 1.0
        if cfg.gravity_sign_init == "down":
            return -1.0
        if cfg.gravity_sign_init == "up":
            return 1.0
        return -1.0 if self.rng.random() < 0.5 else 1.0

    @property
    def state_names(self) -> Tuple[str, ...]:
        """Column names matching ``state()``. Write these into meta.json."""
        names = STATE_NAMES_V2 if self.cfg.mass_from_color else STATE_NAMES
        if self.cfg.occluder:
            names = names + ("ball_visible",)
        if self.cfg.gravity > 0.0:
            names = names + ("gravity_sign",)
        return names

    def ball_visible(self) -> float:
        """v3: fraction of the ball's area not hidden by the band. 1.0 in v1/v2."""
        if not self.cfg.occluder:
            return 1.0
        return visible_fraction(
            float(self.ball[1]), self.cfg.ball_radius, self.cfg.occluder_y
        )

    # ------------------------------------------------------------------- step

    def step(self, action: int):
        """Advance one frame. Returns ``(frame, state, events)``."""
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")

        cfg = self.cfg
        dt = 1.0 / cfg.substeps
        drive = (action - 1) * cfg.paddle_speed  # LEFT->-1, STAY->0, RIGHT->+1
        hw = cfg.paddle_w / 2.0

        self._events = 0
        for _ in range(cfg.substeps):
            # Paddle first, so the ball sees this substep's paddle position.
            new_x = float(np.clip(self.paddle_x + drive * dt, hw, 1.0 - hw))
            # Derive velocity from the realised motion so that clamping at the
            # walls is reflected in the english, rather than pretending the
            # paddle kept moving.
            self.paddle_vx = (new_x - self.paddle_x) / dt
            self.paddle_x = new_x

            # v4: gravity acts BEFORE the move (semi-implicit Euler), once per
            # substep, so over a whole frame the velocity change is exactly
            # ``substeps * gravity * dt = gravity``. Guarded rather than
            # multiplied by a zero sign, so the v1 arithmetic is literally
            # unchanged when the switch is off.
            if cfg.gravity > 0.0:
                # v4.1: the axis is a config choice. Indexing with a stored
                # integer rather than branching keeps the two axes literally
                # the same line of arithmetic, so nothing can drift between
                # them; ``_grav_axis`` is 1 for "y", which is the v4 line.
                self.ball_v[self._grav_axis] += (
                    self.gravity_sign * cfg.gravity * dt
                )

            self.ball += self.ball_v * dt
            self._collide_walls()
            self._collide_paddle()

        # v3: a *state* flag rather than a collision, so it is evaluated once
        # at the end of the step (after every substep has moved the ball)
        # rather than inside the substep loop. Computed from the same
        # ``ball_visible`` the state column reports, so the two can never
        # disagree -- asserted in tests/test_env_v3.py.
        if self.cfg.occluder and self.ball_visible() < 1e-3:
            self._events |= EVENT_HIDDEN

        self.t += 1
        return self.render(), self.state(), self._events

    def state(self) -> np.ndarray:
        """6 columns in v1 mode; +mass for v2, +ball_visible for v3, in that order.

        A variable-width state vector is a little impolite, but the alternative
        -- always emitting a constant mass=1 column -- would change the shape of
        every v1 dataset and break every v1 consumer, which is worse.
        """
        s = [
            self.ball[0],
            self.ball[1],
            self.ball_v[0],
            self.ball_v[1],
            self.paddle_x,
            self.paddle_vx,
        ]
        if self.cfg.mass_from_color:
            s.append(self.mass)
        if self.cfg.occluder:
            s.append(self.ball_visible())
        if self.cfg.gravity > 0.0:
            s.append(self.gravity_sign)
        return np.array(s, dtype=np.float32)

    # -------------------------------------------------------------- collisions

    def _collide_walls(self) -> None:
        r = self.cfg.ball_radius
        for axis, flag in ((0, EVENT_WALL_X), (1, EVENT_WALL_Y)):
            if self.ball[axis] < r:
                self.ball[axis] = 2.0 * r - self.ball[axis]  # mirror position
                self.ball_v[axis] = -self.ball_v[axis]
                self._events |= flag
            elif self.ball[axis] > 1.0 - r:
                self.ball[axis] = 2.0 * (1.0 - r) - self.ball[axis]
                self.ball_v[axis] = -self.ball_v[axis]
                self._events |= flag

    def _collide_paddle(self) -> None:
        cfg = self.cfg
        r = cfg.ball_radius
        hw, hh = cfg.paddle_w / 2.0, cfg.paddle_h / 2.0
        px, py = self.paddle_x, cfg.paddle_y
        cx, cy = float(self.ball[0]), float(self.ball[1])

        # Closest point on the paddle AABB to the ball centre.
        qx = min(max(cx, px - hw), px + hw)
        qy = min(max(cy, py - hh), py + hh)
        dx, dy = cx - qx, cy - qy
        d2 = dx * dx + dy * dy
        if d2 >= r * r:
            return

        if d2 > 1e-16:
            d = np.sqrt(d2)
            nx, ny = dx / d, dy / d
            self.ball[0], self.ball[1] = qx + nx * r, qy + ny * r
        else:
            # Ball centre is inside the paddle. Eject along the axis of least
            # penetration. Reachable only in pathological configs, but cheap to
            # handle and it keeps the sim from ever exploding.
            pens = ((px + hw) - cx, cx - (px - hw), (py + hh) - cy, cy - (py - hh))
            k = int(np.argmin(pens))
            nx, ny = ((1, 0), (-1, 0), (0, 1), (0, -1))[k]
            if k == 0:
                self.ball[0] = px + hw + r
            elif k == 1:
                self.ball[0] = px - hw - r
            elif k == 2:
                self.ball[1] = py + hh + r
            else:
                self.ball[1] = py - hh - r

        vn = self.ball_v[0] * nx + self.ball_v[1] * ny
        if vn < 0.0:  # only reflect if actually moving into the paddle
            self.ball_v[0] -= 2.0 * vn * nx
            self.ball_v[1] -= 2.0 * vn * ny

        # v2 effect 2 of 2: the paddle delivers the same sideways IMPULSE to
        # every ball, so the velocity change it produces is impulse / mass. A
        # heavy (purple) ball barely swerves; a light (yellow) one is flicked
        # hard. With m = 1 this is the v1 line unchanged.
        self.ball_v[0] += cfg.english * self.paddle_vx / self.mass
        if cfg.gravity > 0.0:
            # v4: no renormalisation -- speed is a real, varying quantity now,
            # and pinning it would destroy the very curvature the tier is
            # about. (The min-|vy| guard lives inside ``_renormalise_velocity``
            # and therefore goes with it; see the module docstring.) Only the
            # english can pump energy in, so only the english is bounded.
            self._clip_speed()
            # THE EVENT. Once per FRAME, not once per substep: a contact that
            # spans two substeps is one contact, and flipping twice would
            # silently mean not flipping at all. ``EVENT_PADDLE`` is already
            # set if an earlier substep in this frame touched the paddle, which
            # makes it exactly the "have we flipped yet this frame" flag.
            if not (self._events & EVENT_PADDLE):
                self.gravity_sign = -self.gravity_sign
                self._events |= EVENT_FLIP
        else:
            self._renormalise_velocity()
        self._events |= EVENT_PADDLE

        # English can push the ball back into a wall; re-resolve.
        self._collide_walls()
        self.ball[0] = float(np.clip(self.ball[0], r, 1.0 - r))
        self.ball[1] = float(np.clip(self.ball[1], r, 1.0 - r))

    def _clip_speed(self) -> None:
        """v4: bound |v| after a paddle contact. Counts how often it fires.

        Gravity alone cannot run the speed away -- it is conservative and the
        box is 1 unit tall, so |v| is pinned to ``sqrt(v_floor² ± 2 g Δh)``,
        at most 0.026 for the v4 numbers. The english is the one term that adds
        energy from outside, up to 0.0105 per contact, and it does NOT average
        out: a tracking paddle is usually moving toward the ball when it hits,
        so successive impulses correlate rather than cancelling.

        This guard was expected to be a formality and it is not, which is
        precisely why the counter exists. Under the tracking half of the
        ``mix`` behaviour policy the ball is struck often enough to reach the
        cap, and ``worldsim.collect._gravity_report`` prints how often per
        split. Read that number before trusting any claim that speed is
        "roughly constant" in a v4 dataset: it is not, by a factor of about
        seven between the slowest and fastest frames.
        """
        speed = float(np.linalg.norm(self.ball_v))
        if speed > self.cfg.max_speed:
            self.ball_v *= self.cfg.max_speed / speed
            self.speed_clips += 1

    def _renormalise_velocity(self) -> None:
        # v2 effect 1 of 2: the conserved quantity is now the *effective* speed
        # ball_speed / m, fixed for the whole episode. So speed is still
        # conserved within an episode (the data distribution stays stationary,
        # which was the v1 design goal) but varies across episodes, and the
        # colour tells you which one you are in.
        cfg = self.cfg
        target = self.speed
        speed = float(np.linalg.norm(self.ball_v))
        if speed < 1e-12:
            self.ball_v[:] = (0.0, target)
            return
        self.ball_v *= target / speed

        # Keep the trajectory off horizontal, or a well-timed paddle hit can
        # leave the ball skimming a wall for hundreds of frames.
        min_vy = cfg.min_vy_frac * target
        if abs(self.ball_v[1]) < min_vy:
            sign = 1.0 if self.ball_v[1] >= 0.0 else -1.0
            self.ball_v[1] = sign * min_vy
            vx_mag2 = target**2 - self.ball_v[1] ** 2
            vx_sign = 1.0 if self.ball_v[0] >= 0.0 else -1.0
            self.ball_v[0] = vx_sign * np.sqrt(max(vx_mag2, 0.0))

    # --------------------------------------------------------------- rendering

    def _ensure_grid(self, res: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._grid is None or self._grid_res != res:
            xs = (np.arange(res) + 0.5) / res
            ys = 1.0 - (np.arange(res) + 0.5) / res  # row 0 is the top
            self._grid = np.meshgrid(xs, ys)
            self._grid_res = res
        return self._grid

    def render(self, res: Optional[int] = None) -> np.ndarray:
        """Render the current state to a ``(res, res, 3)`` uint8 array."""
        cfg = self.cfg
        res = res or cfg.res
        gx, gy = self._ensure_grid(res)
        px_size = 1.0 / res

        img = np.empty((res, res, 3), dtype=np.float64)
        img[:] = np.asarray(cfg.bg_color, dtype=np.float64)

        # Paddle: separable box coverage.
        hw, hh = cfg.paddle_w / 2.0, cfg.paddle_h / 2.0
        ax = np.clip(0.5 + (hw - np.abs(gx - self.paddle_x)) / px_size, 0.0, 1.0)
        ay = np.clip(0.5 + (hh - np.abs(gy - cfg.paddle_y)) / px_size, 0.0, 1.0)
        a_paddle = (ax * ay)[..., None]
        img += (np.asarray(cfg.paddle_color, dtype=np.float64) - img) * a_paddle

        # Ball on top: signed-distance coverage.
        d = np.sqrt((gx - self.ball[0]) ** 2 + (gy - self.ball[1]) ** 2)
        a_ball = np.clip(0.5 + (cfg.ball_radius - d) / px_size, 0.0, 1.0)[..., None]
        # self.ball_color, not cfg.ball_color: in v2 the colour is per-episode
        # state, set at reset from the mass. That one-word change is the entire
        # rendering diff for v2 -- the physics/appearance coupling lives in
        # reset() and the collision code, not here.
        img += (np.asarray(self.ball_color, dtype=np.float64) - img) * a_ball

        # v3: the occlusion band, LAST, so it covers both ball and paddle.
        # Full width, so only the vertical coverage term is non-trivial -- the
        # same separable box coverage the paddle uses, which keeps the band's
        # edges antialiased like everything else in the frame. That matters:
        # a hard-edged band would be the only quantised object in the image and
        # the VAE would find its edge easier to code than the ball.
        if cfg.occluder:
            lo, hi = cfg.occluder_y
            cy, hh_b = 0.5 * (lo + hi), 0.5 * (hi - lo)
            a_band = np.clip(0.5 + (hh_b - np.abs(gy - cy)) / px_size, 0.0, 1.0)[..., None]
            img += (np.asarray(cfg.occluder_color, dtype=np.float64) - img) * a_band

        return np.clip(img + 0.5, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------- misc

    def config_dict(self) -> dict:
        return asdict(self.cfg)
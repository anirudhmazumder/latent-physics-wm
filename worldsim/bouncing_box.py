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

# Bit flags for the per-step event mask.
EVENT_WALL_X = 1
EVENT_WALL_Y = 2
EVENT_PADDLE = 4


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


class BouncingBox:
    """Deterministic given a seed. ``step`` advances one frame."""

    def __init__(self, config: Optional[BoxConfig] = None, seed: Optional[int] = None):
        self.cfg = config or BoxConfig()
        self.rng = np.random.default_rng(seed)
        self._grid: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._grid_res: Optional[int] = None
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

        y_lo = cfg.paddle_y + cfg.paddle_h / 2.0 + r + 0.05
        self.ball = np.array(
            [self.rng.uniform(r, 1.0 - r), self.rng.uniform(y_lo, 1.0 - r)],
            dtype=np.float64,
        )

        # Reject launch angles too close to an axis -- those trajectories are
        # degenerate and take a long time to explore the box.
        while True:
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            if abs(np.sin(theta)) > 0.25 and abs(np.cos(theta)) > 0.25:
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

    @property
    def state_names(self) -> Tuple[str, ...]:
        """Column names matching ``state()``. Write these into meta.json."""
        return STATE_NAMES_V2 if self.cfg.mass_from_color else STATE_NAMES

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

            self.ball += self.ball_v * dt
            self._collide_walls()
            self._collide_paddle()

        self.t += 1
        return self.render(), self.state(), self._events

    def state(self) -> np.ndarray:
        """6 columns in v1 mode, 7 (with mass) when mass_from_color is on.

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
        self._renormalise_velocity()
        self._events |= EVENT_PADDLE

        # English can push the ball back into a wall; re-resolve.
        self._collide_walls()
        self.ball[0] = float(np.clip(self.ball[0], r, 1.0 - r))
        self.ball[1] = float(np.clip(self.ball[1], r, 1.0 - r))

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

        return np.clip(img + 0.5, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------- misc

    def config_dict(self) -> dict:
        return asdict(self.cfg)
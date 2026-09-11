"""v1: a ball bouncing in a unit box, with a paddle the agent moves left/right.

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
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

LEFT, STAY, RIGHT = 0, 1, 2
ACTIONS: Tuple[int, int, int] = (LEFT, STAY, RIGHT)
ACTION_NAMES = ("left", "stay", "right")

STATE_NAMES = ("ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x", "paddle_vx")

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

    def __post_init__(self) -> None:
        if self.paddle_y is None:
            self.paddle_y = self.paddle_h / 2.0


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
        self.ball_v = cfg.ball_speed * np.array(
            [np.cos(theta), np.sin(theta)], dtype=np.float64
        )

        hw = cfg.paddle_w / 2.0
        self.paddle_x = float(self.rng.uniform(hw, 1.0 - hw))
        self.paddle_vx = 0.0
        self.t = 0
        self._events = 0
        return self.render()

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
        return np.array(
            [
                self.ball[0],
                self.ball[1],
                self.ball_v[0],
                self.ball_v[1],
                self.paddle_x,
                self.paddle_vx,
            ],
            dtype=np.float32,
        )

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

        self.ball_v[0] += cfg.english * self.paddle_vx
        self._renormalise_velocity()
        self._events |= EVENT_PADDLE

        # English can push the ball back into a wall; re-resolve.
        self._collide_walls()
        self.ball[0] = float(np.clip(self.ball[0], r, 1.0 - r))
        self.ball[1] = float(np.clip(self.ball[1], r, 1.0 - r))

    def _renormalise_velocity(self) -> None:
        cfg = self.cfg
        speed = float(np.linalg.norm(self.ball_v))
        if speed < 1e-12:
            self.ball_v[:] = (0.0, cfg.ball_speed)
            return
        self.ball_v *= cfg.ball_speed / speed

        # Keep the trajectory off horizontal, or a well-timed paddle hit can
        # leave the ball skimming a wall for hundreds of frames.
        min_vy = cfg.min_vy_frac * cfg.ball_speed
        if abs(self.ball_v[1]) < min_vy:
            sign = 1.0 if self.ball_v[1] >= 0.0 else -1.0
            self.ball_v[1] = sign * min_vy
            vx_mag2 = cfg.ball_speed**2 - self.ball_v[1] ** 2
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
        img += (np.asarray(cfg.ball_color, dtype=np.float64) - img) * a_ball

        return np.clip(img + 0.5, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------- misc

    def config_dict(self) -> dict:
        return asdict(self.cfg)
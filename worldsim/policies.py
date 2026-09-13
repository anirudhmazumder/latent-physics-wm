"""Action sequences for data collection.

The behaviour policies used to collect every dataset (v1-v4).
Written up in `docs/01_environment_and_data.md`; see also `data/README.md`.

Uniform i.i.d. random actions look like the obvious choice and are a trap: the
paddle random-walks with tiny steps and spends nearly all its time near where it
started. The dynamics model then never sees sustained paddle motion, so it never
learns that actions do anything much.

Sticky actions -- hold a choice for a geometrically distributed number of frames
-- give you long sweeps across the box and a much wider marginal distribution
over paddle position and velocity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def sticky_random_actions(
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
    mean_hold: float = 8.0,
) -> np.ndarray:
    """Return an ``(n_steps,)`` int8 array of actions in {0, 1, 2}."""
    rng = rng or np.random.default_rng()
    out = np.empty(n_steps, dtype=np.int8)
    t = 0
    p = 1.0 / max(mean_hold, 1.0)
    while t < n_steps:
        a = int(rng.integers(0, 3))
        hold = int(rng.geometric(p))
        out[t : t + hold] = a
        t += hold
    return out


def uniform_random_actions(
    n_steps: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """i.i.d. uniform actions. Kept for comparison -- see the module docstring."""
    rng = rng or np.random.default_rng()
    return rng.integers(0, 3, size=n_steps).astype(np.int8)

# --------------------------------------------------------------- state-aware
#
# Why anything beyond sticky random actions is needed at all.
#
# Under random actions the paddle and the ball are statistically independent, so
# contact happens only by coincidence: about 0.5% of frames, roughly one hit per
# 200-step episode. A dynamics model trained on that data can reach a very good
# loss while treating the action input as noise, because for 99.5% of
# transitions the action genuinely does not affect the ball. The action -> ball
# causal channel is real but it is buried under the sampling noise of a rare
# event.
#
# The fix is to bias the *behaviour policy* toward states where the action
# matters. A paddle that tracks the ball collides with it constantly, so the
# dataset now contains many examples of "paddle was here, ball arrived, ball
# left with english applied". This is a deliberate, documented distribution
# shift: the dynamics model is trained on a mixture of random and tracking
# behaviour so that it sees both the wide paddle-position marginal of the random
# policy and the dense contact statistics of the tracking policy.
#
# Keeping BOTH in the mixture matters. Pure tracking would put the paddle almost
# always underneath the ball, and the model would learn the shortcut
# "paddle_x ~= ball_x" instead of learning what the actions do.


def tracking_action(
    state: np.ndarray,
    paddle_w: float = 0.26,
    dead_zone_frac: float = 0.25,
) -> int:
    """Move the paddle toward ``ball_x``. Returns an action in {0, 1, 2}.

    ``state`` is the 6-vector ``[ball_x, ball_y, ball_vx, ball_vy, paddle_x,
    paddle_vx]`` in world coordinates.

    The dead zone (default a quarter of the paddle width) is not cosmetic. A
    bang-bang controller with no dead zone oscillates left/right every frame
    once it is on target, which produces a high-frequency action sequence that
    (a) looks like noise to the model and (b) averages to zero paddle motion,
    so the interesting "sustained sweep" behaviour disappears. With the dead
    zone the paddle parks when it is close enough and the action sequence stays
    piecewise constant, like the sticky policy.
    """
    ball_x = float(state[0])
    paddle_x = float(state[4])
    err = ball_x - paddle_x
    if abs(err) < dead_zone_frac * paddle_w:
        return 1  # STAY
    return 2 if err > 0.0 else 0  # RIGHT if the ball is to the right


class MixedPolicy:
    """Alternating segments of ball-tracking and sticky-random behaviour.

    At the start of every segment we flip a coin: with probability ``p_track``
    the next segment tracks the ball, otherwise it holds one fixed random
    action. Segment lengths are geometric with mean ``mean_hold`` in both cases,
    matching ``sticky_random_actions`` so the two modes have the same temporal
    statistics and the only difference is *what* the action depends on.

    Stateful by necessity: a tracking action is a function of the current
    environment state, so unlike the open-loop policies above this cannot be
    pre-sampled as an array. Call ``act(state)`` once per step.
    """

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        p_track: float = 0.5,
        mean_hold: float = 8.0,
        paddle_w: float = 0.26,
        dead_zone_frac: float = 0.25,
    ):
        self.rng = rng or np.random.default_rng()
        self.p_track = float(p_track)
        self.p_end = 1.0 / max(mean_hold, 1.0)
        self.paddle_w = float(paddle_w)
        self.dead_zone_frac = float(dead_zone_frac)
        self._left = 0          # frames remaining in the current segment
        self._tracking = False
        self._fixed_action = 1

    def _new_segment(self) -> None:
        self._tracking = bool(self.rng.random() < self.p_track)
        self._left = int(self.rng.geometric(self.p_end))
        if not self._tracking:
            self._fixed_action = int(self.rng.integers(0, 3))

    def act(self, state: np.ndarray) -> int:
        if self._left <= 0:
            self._new_segment()
        self._left -= 1
        if self._tracking:
            return tracking_action(
                state, paddle_w=self.paddle_w, dead_zone_frac=self.dead_zone_frac
            )
        return self._fixed_action

    # Convenience so callers can treat it like the open-loop policies.
    __call__ = act

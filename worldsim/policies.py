"""Action sequences for data collection.

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

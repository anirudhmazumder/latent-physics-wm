"""Primitives for the v3 object-permanence experiments.

Separated from ``wm.eval_permanence_v3`` for one reason: these four functions
are the ones that can be *wrong in a way no plot would reveal*, so they are
unit-tested against hand-built examples in ``tests/test_rnn_v3.py``. The
evaluator is the part that dreams and draws; this is the part that decides what
counts as a hidden run, when a dreamed ball has come out, and what "no memory"
would have predicted.

The frame convention, which everything here depends on
------------------------------------------------------
``frames[e, t]`` with ``actions[e, t]`` produces ``frames[e, t + 1]``, so

    events[e, t]  describes the transition INTO frame t + 1
    states[e, t]  is the state AT frame t   (there are T + 1 of them)

which means ``EVENT_HIDDEN`` set in ``events[e, t]`` is a statement about
``states[e, t + 1]``. That off-by-one is checked directly in the tests, because
getting it wrong would shift every hidden run by one frame and quietly bias
every exit-time number in this stage by exactly one.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from worldsim.bouncing_box import EVENT_HIDDEN, EVENT_WALL_X

# A ball is "fully hidden" when its whole disc is inside the band, i.e. its
# centre is at least one radius inside each edge. The same quantity the
# simulator reports as ball_visible < 1e-3; the geometric form is what we need
# for a *dreamed* ball, which has no simulator to ask.
def band_interior(band: Tuple[float, float], radius: float) -> Tuple[float, float]:
    return band[0] + radius, band[1] - radius


class HiddenRun(NamedTuple):
    """One maximal stretch of fully-hidden frames, in STATE time.

    ``t0``/``t1`` are inclusive state indices: ``states[e, t0 .. t1]`` all have
    ``ball_visible < 1e-3``. ``length`` is the number of hidden frames, and the
    ball is partially visible again at ``t1 + 1`` (the *exit* frame) and was
    still partially visible at ``t0 - 1`` (the *entry* frame).
    """

    episode: int
    t0: int
    t1: int
    wall_x: bool      # did a side-wall bounce happen while hidden?

    @property
    def length(self) -> int:
        return self.t1 - self.t0 + 1

    @property
    def entry(self) -> int:
        return self.t0 - 1

    @property
    def exit(self) -> int:
        return self.t1 + 1


def hidden_runs_from_events(
    events: np.ndarray, min_len: int = 1, drop_incomplete: bool = True
) -> List[HiddenRun]:
    """Maximal ``EVENT_HIDDEN`` runs, converted to state time.

    ``events`` is (E, T) int. A run of set bits at event indices ``i0..i1``
    means the ball was fully hidden at state times ``i0+1 .. i1+1``.

    ``drop_incomplete`` discards runs that touch either end of the episode,
    because those have no entry frame (nothing to warm up on) or no exit frame
    (no ground truth to score against). That is a real selection -- the
    reported counts say how many went -- but the alternative is scoring an
    emergence that never happened.

    A run is flagged ``wall_x`` if any transition *inside* it (event indices
    i0..i1, i.e. the steps that moved the ball while it was hidden) carried
    ``EVENT_WALL_X``.
    """
    events = np.asarray(events)
    runs: List[HiddenRun] = []
    E, T = events.shape
    hid = (events & EVENT_HIDDEN) != 0
    wall = (events & EVENT_WALL_X) != 0
    for e in range(E):
        row = hid[e]
        i = 0
        while i < T:
            if not row[i]:
                i += 1
                continue
            j = i
            while j + 1 < T and row[j + 1]:
                j += 1
            t0, t1 = i + 1, j + 1
            ok = (t1 - t0 + 1) >= min_len
            if drop_incomplete and (i == 0 or j == T - 1):
                ok = False
            if ok:
                runs.append(HiddenRun(e, t0, t1, bool(wall[e, i : j + 1].any())))
            i = j + 1
    return runs


def hidden_age(visible: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    """How many consecutive frames the ball has been fully hidden, inclusive.

    ``visible`` is (E, T+1) ``ball_visible``. Returns the same shape: 0 where
    the ball is not fully hidden, else k >= 1 counting from the first hidden
    frame of the current run. This is the x-axis of the memory-decay curve.
    """
    hid = np.asarray(visible) < thresh
    age = np.zeros(hid.shape, dtype=np.int64)
    for e in range(hid.shape[0]):
        k = 0
        for t in range(hid.shape[1]):
            k = k + 1 if hid[e, t] else 0
            age[e, t] = k
    return age


def first_exit(
    y: np.ndarray,
    band: Tuple[float, float],
    radius: float,
    consec: int = 2,
    start: int = 0,
) -> Tuple[int, int]:
    """First index at which a trajectory's ball is no longer fully hidden.

    ``y`` is a 1-d sequence of ball-centre heights (true or decoded from a
    dream). Returns ``(k, side)``: ``k`` is the first index >= ``start`` that
    begins ``consec`` consecutive frames outside the band interior, and
    ``side`` is +1 if it left through the top and -1 through the bottom. If it
    never does, returns ``(-1, 0)``.

    Why ``consec = 2`` by default. The decoded position comes from a kNN probe
    whose nearest neighbour can be one frame's worth of motion away, so a
    single frame poking over the edge is within the instrument's noise. Two in
    a row is not. The cost is that a true exit is reported one frame late only
    if the trajectory re-enters immediately, which cannot happen: nothing
    inside the band reverses a ball's vertical motion.
    """
    lo, hi = band_interior(band, radius)
    y = np.asarray(y, dtype=float)
    out = (y < lo) | (y > hi)
    n = len(y)
    for k in range(start, n - consec + 1):
        if out[k : k + consec].all():
            return k, (1 if y[k] > hi else -1)
    return -1, 0


def no_memory_exit(
    state: np.ndarray, run: HiddenRun, band: Tuple[float, float], radius: float
) -> dict:
    """The floor: "the ball is wherever it was when it disappeared".

    A model with no memory at all can still see the band and can still have
    learned the *typical* duration of an occlusion. What it cannot do is track
    x. So the baseline is given the true exit time for free and predicts the
    exit x as the ball's x on the last frame it was visible; its exit side is
    the side the ball went *in* on, which is the opposite of the true exit side
    whenever the ball crosses the band (i.e. almost always), so we report it as
    the entry side and let the table show what that costs.

    ``state`` is that episode's (T+1, S) true state array.
    """
    entry = state[run.entry]
    lo, hi = band_interior(band, radius)
    return {
        "exit_x": float(entry[0]),
        "exit_time": int(run.length + 1),          # handed the true duration
        "side": 1 if entry[1] > hi else -1,        # the side it went IN on
    }


def linear_exit(
    state: np.ndarray, run: HiddenRun, band: Tuple[float, float], radius: float
) -> dict:
    """Straight-line extrapolation from the entry state. No walls, no bounces.

    The interesting baseline for part (c): it uses the entry *velocity*, so it
    gets the exit x right whenever nothing happened behind the band, and is
    wrong by about twice the overshoot whenever the ball hit a side wall. The
    gap between this and the model on the wall-bounce subset is the "simulates
    rather than extrapolates" measurement.

    The predicted exit *time* comes from the same straight line: the number of
    frames to travel from the entry y to whichever band edge the ball is
    heading for.
    """
    x0, y0, vx0, vy0 = (float(v) for v in state[run.entry][:4])
    lo, hi = band_interior(band, radius)
    edge = hi if vy0 > 0 else lo
    dt = (edge - y0) / vy0 if vy0 != 0 else np.inf
    dt = float(max(dt, 0.0))
    return {
        "exit_x": x0 + vx0 * (run.length + 1),
        "exit_time": dt,
        "side": 1 if vy0 > 0 else -1,
    }


def reflect_x(x: float, radius: float) -> float:
    """Fold a straight-line x back into the box, the way a wall would.

    Only used to *describe* what the linear baseline is missing -- it is never
    fed to a model. The box is [radius, 1 - radius]; reflecting repeatedly is
    the unrolled-billiard map.
    """
    lo, hi = radius, 1.0 - radius
    if hi <= lo:
        return float(np.clip(x, lo, hi))
    period = 2.0 * (hi - lo)
    u = (x - lo) % period
    return lo + (u if u <= hi - lo else period - u)

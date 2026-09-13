# v4 design sweep — what does forgetting the gravity sign cost?

Run BEFORE any training (the v3 lesson, now a habit). 60 episodes × 200 steps
per cell, seeds 5000+, **interceptions per floor visit**. Script:
`sweep_v4.py`, run with
`PYTHONPATH=. /opt/miniconda3/envs/NN/bin/python runs/v4_design/sweep_v4.py`
(498 s, one core). Raw numbers in `sweep_v4.json`, console output in
`sweep_v4.log`.

The five reference policies, all privileged (true state, no encoder):

| policy | what it sees | what it is |
|---|---|---|
| `oracle` | true ball x | `tracking_action` — the v1–v3.1 ceiling |
| `ballistic_oracle` | true state **and the true sign** | aims at the predicted landing x; the anticipation ceiling |
| `sign_blind_oracle` | true state, sign **assumed DOWN** | the same solver minus one bit — **the memoryless bound** |
| `stay` | — | never moves |
| `random` | — | sticky random actions, the behaviour policy |

`sign_blind_oracle` is `ballistic_oracle` with `assumed_sign = -1`. They share
the solver, the dead zone and the geometry, and `tests/test_env_v4.py` asserts
they agree *exactly* wherever the true sign is down — so any gap between them is
attributable to the sign and to nothing else.

## The table

| gravity | launch min angle | oracle | ballistic | sign-blind | **gap** | stay | random | floor visits/ep | flips/ep | episodes stalled |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.00005 | 40° | 0.98 | 0.98 | 0.98 | **0.00** | 0.50 | 0.43 | 2.27 | 2.22 | 0% |
| 0.00005 | 50° | 0.99 | 0.99 | 0.99 | **0.00** | 0.46 | 0.39 | 2.50 | 2.47 | 0% |
| **0.0001** | **40°** | **0.98** | **0.98** | **0.98** | **0.00** | 0.41 | 0.41 | 2.12 | 2.08 | 5% |
| 0.0001 | 50° | 0.99 | 0.99 | 0.99 | **0.00** | 0.43 | 0.40 | 2.38 | 2.37 | 2% |
| 0.00015 | 40° | 0.98 | 0.98 | 0.98 | **0.00** | 0.40 | 0.41 | 2.02 | 1.98 | 20% |
| 0.00015 | 50° | 0.99 | 0.98 | 0.98 | **0.00** | 0.40 | 0.36 | 2.23 | 2.20 | 18% |
| 0.0002 | 40° | 0.98 | 0.98 | 0.98 | **0.00** | 0.46 | 0.39 | 2.02 | 1.98 | 43% |
| 0.0002 | 50° | 1.00 | 0.99 | 1.00 | **0.00** | 0.46 | 0.38 | 2.28 | 2.28 | 33% |

"Episodes stalled" = the ball failed to reach either end of the box for 100+
consecutive frames at some point in the episode, under the oracle's own play.

## The finding, stated plainly

**The tracking task does not need the gravity sign, at any gravity in the
range.** The gap is 0.00 in all eight cells, and it is not a near-miss that a
bigger number would fix: raising gravity from 5e-5 to 2e-4 does not move the
sign-blind oracle by a single point, while it takes the stalled-episode rate
from 0% to 43%. There is no setting of this knob at which the criterion
(gap ≥ 0.15, no stalls) is met. Raising it further is not an option either —
the stalls are the same physics as the gap, and they arrive first.

Note also that `oracle` and `ballistic_oracle` are identical to two decimal
places. Under gravity, aiming at where the ball *is* is still as good as aiming
at where it *will land*; anticipation buys nothing either.

## Why — and this is the part that says what a v4.1 would have to change

The sweep measures it directly (bottom of `sweep_v4.log`, `recoverability` in
the script). At every frame of a descending approach, with g = 1e-4 and 40°:

| frames to landing | n | mean \|Δx\| between the two predictions | fraction > half a paddle | mean slack (frames) | fraction with slack < 0 |
|---|---|---|---|---|---|
| 1–10 | 861 | 0.001 | 0.0% | 4.9 | **0.00%** |
| 10–20 | 883 | 0.006 | 0.0% | 14.2 | **0.00%** |
| 20–40 | 1562 | 0.024 | 0.0% | 28.1 | **0.00%** |
| 40–80 | 494 | 0.031 | 6.9% | 48.8 | **0.00%** |
| 80–120 | 368 | 0.035 | 8.4% | 95.9 | **0.00%** |

`slack` = `frames_until_landing − |Δx| / paddle_speed`: how many frames of
paddle travel the blind oracle has spare to undo its own error. **It is never
negative, on any of 4,171 approach frames.**

Two things conspire.

1. **The prediction error from a wrong sign is small in absolute terms.** It
   peaks at 0.035 world units — a quarter of a paddle width — and only 8% of
   the earliest frames exceed half a paddle. Gravity's contribution to the
   landing point is `a t²/2 ≈ 1e-4 · 60² / 2 = 0.18` at most, and the two
   hypotheses differ by twice that in *height*, which converts to far less in
   *x* because the ball's horizontal speed is only ~0.015.
2. **The error shrinks to zero exactly as the deadline approaches**, while the
   paddle's speed (0.030/frame) does not. Being wrong about the sign is a
   *large error far away and no error up close*, which is the one error profile
   a fast bang-bang tracker is perfectly built to absorb. It never has to
   commit.

That is the structural difference from v3.1, where the memoryless oracle was
genuinely beaten: there, the information was *withheld until too late* (the ball
emerged from the band at contact height, leaving no time to cross), so the
memoryless policy had to commit early or lose. Here the information is never
withheld — it just is not needed.

**A v4.1 that wanted a C-stage question would have to remove the slack, not
raise the gravity.** The levers that would work are the ones that set a
deadline: a slower paddle (`paddle_speed` well below 0.015 so the box takes
longer to cross than the ball takes to fall), a wider box, or the v3.1 trick of
hiding the ball on the final approach (`occluder` composes with `gravity` — all
three switches do). Gravity is the wrong dial.

## What was chosen, and why anyway

**gravity = 0.0001, launch_min_angle = 40°** — the design document's default,
kept unchanged.

The selection rule in the brief (smallest gravity with gap ≥ 0.15 and no
stalls) has no solution, so it does not select. Given that, the design's own
numbers are the right thing to keep, for three reasons:

* v4's headline questions are **M-stage** questions — is the sign in `h`, does
  the dream curve the right way, does the flip counterfactual work, LSTM vs
  transformer — and none of them depend on the controller gap. They need a
  sign that is invisible per frame, holds for ~100 frames, and measurably bends
  the trajectory. All three hold at 1e-4/40°.
* It is the *only* cell that is both clean (5% stalled under oracle play, 0–2.4%
  of frames under the behaviour policies that actually make the data) and at
  the design's stated SNR working point.
* Larger gravity buys nothing and costs stalls.

50° is the one defensible alternative: it halves the stall rate (2% vs 5%) at
the same gravity and makes the two signs slightly more similar in traverse
duration (see below). It was not taken, because the design document specifies
40° and nothing in the sweep argues for changing it.

## One thing the design document did not anticipate

The sign is invisible *per frame*, as designed. But it is not as invisible
*per traverse* as the curvature argument implies, because it also sets the
ball's energy budget. Measured on `data/v4/long`
(`runs/v4_env/traverse_stats.json`):

| sign in force | full traverses | median frames | mean \|vy\| |
|---|---|---|---|
| pulls DOWN | 177 | **48.0** | 0.0192 |
| pulls UP | 186 | **40.0** | 0.0224 |

A 1.20× difference in crossing time, and a 17% difference in mean vertical
speed. That is a much coarser cue than a quadratic curvature fit, and it is
available from a *single traverse* rather than from a memory of the flip. It
does not break the tier — it is a property of the trajectory, not of a frame,
so `z` still cannot carry it — but stage two must treat "M knows the sign" and
"M remembers the flip" as separable claims, and the flip counterfactual (which
holds the trajectory-so-far fixed) is the test that separates them. It also
explains the 55/45 frame imbalance between the signs in every collected split:
down-traverses simply take longer.

---

Next: [`wm/README_V4.md`](../../wm/README_V4.md) — the datasets and the v4 VAE.

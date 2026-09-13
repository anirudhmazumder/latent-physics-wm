# v4.1 design sweep — the side-wind. Does the sign matter now?

**Headline: no cell qualifies, and the reason is not the wind's magnitude, the
axis, or the paddle's width. It is the paddle's *speed*, exactly as
[`sweep.md`](sweep.md) predicted. Nothing was collected.**

Run BEFORE any training, same as v4. 60 episodes × 200 steps per cell, seeds
5000+, **interceptions per floor visit**. Script: `sweep_v41.py`, run with
`PYTHONPATH=. /opt/miniconda3/envs/NN/bin/python runs/v4_design/sweep_v41.py`
(431 s, one core). Raw numbers in `sweep_v41.json`, console output in
`sweep_v41.log`.

## What was changed and why

The v4 sweep found the gap between the full and the sign-blind oracle was 0.00
at every gravity in the range, and diagnosed it: a *vertical* acceleration
perturbs the landing **height**, and height converts into **x** only through the
ball's slow horizontal speed, so a wrong sign was worth at most 0.035 in x — a
quarter of a paddle — and even that shrank to zero as the ball arrived.

v4.1 turns the acceleration ninety degrees. `BoxConfig.gravity_axis="x"` makes
the flipping term act on `ball_v[0]`: a **side-wind**, +1 blowing toward +x,
still flipped by every paddle contact, still reported in the `gravity_sign`
state column and `EVENT_FLIP`. The hypothesis was that the sign's effect on the
landing point becomes first-order (`a t²` directly in x, ~0.36 over a 60-frame
fall at 1e-4) rather than second-order, and as a bonus that the *vertical*
dynamics become sign-independent — which removes the energy-budget cue the v4
sweep found by accident (a 1.20× difference in traverse duration between the
signs, available from one traverse with no memory of the flip at all).

**The second half of that worked perfectly. The first half did not.**

## The table

`pinned` = fraction of episodes spending >20 % of their frames within one ball
radius of a side wall; `pin_fr` = mean fraction of frames so spent. Read both
against the baseline at the bottom of the section — they are much less
informative than they look.

| axis | wind | paddle_w | launch | oracle | ballistic | sign-blind | **gap** | stay | random | visits/ep | flips/ep | stalled | pinned | pin_fr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x | 0.00005 | 0.26 | 14.5° | 0.98 | 0.98 | 0.98 | **0.00** | 0.52 | 0.39 | 1.72 | 1.68 | 13% | 33% | 19.3% |
| x | 0.00005 | 0.16 | 14.5° | 0.98 | 0.98 | 0.98 | **0.00** | 0.38 | 0.30 | 1.73 | 1.70 | 13% | 32% | 18.7% |
| x | 0.0001 | 0.26 | 14.5° | 0.98 | 0.98 | 0.98 | **0.00** | 0.50 | 0.41 | 1.72 | 1.68 | 13% | 42% | 21.7% |
| x | 0.0001 | 0.16 | 14.5° | 0.98 | 0.98 | 0.98 | **0.00** | 0.38 | 0.35 | 1.72 | 1.68 | 13% | 37% | 20.5% |
| x | 0.0002 | 0.26 | 14.5° | 0.98 | 0.98 | 0.98 | **0.00** | 0.46 | 0.42 | 1.72 | 1.68 | 13% | 33% | 18.6% |
| x | 0.0002 | 0.16 | 14.5° | 0.99 | 0.98 | 0.98 | **0.00** | 0.35 | 0.31 | 1.72 | 1.72 | 13% | 38% | 19.0% |
| *y (v4 ref)* | *0.0001* | *0.26* | *40°* | *0.98* | *0.98* | *0.98* | ***0.00*** | *0.41* | *0.41* | *2.12* | *2.08* | *5%* | *28%* | *19.0%* |

Criterion: smallest wind with gap ≥ 0.15 and ≤ 5 % stalled-or-pinned episodes,
preferring paddle_w 0.26. **No cell meets it.** The gap is 0.00 in all six new
cells, as it was in all eight of v4's.

### The two hazard columns are not the story, and both are false alarms

They were in the brief as guards, so they were measured; neither is what blocks
a cell. Measured over 60 episodes × 200 steps under a ball-chasing policy:

| world | stalled episodes | frames stuck | wall-pinned frames | min \|vy\| |
|---|---|---|---|---|
| plain v1 (no gravity, 14.5°) | 17% | 1.9% | **19.1%** | 0.0050 |
| v4.1 axis x, 1e-4, 14.5° | 13% | 0.7% | 22.2% | 0.0056 |
| v4.1 axis x, 1e-4, 40° | **0%** | 0.0% | 21.4% | 0.0142 |
| v4 axis y, 1e-4, 40° | 5% | 1.8% | 20.8% | 0.0000 |

* **Stalls.** The 13 % in the table is *lower than gravity-free v1's 17 %*. It
  is not a side-wind pathology at all: it is `stuck_fraction`'s 100-frame
  window against v1's shallow launch guard, which admits `|vy|` as low as
  0.005 and therefore honest traverses of 150+ frames. Raising the launch angle
  to 40° removes it entirely (0 %), which is available for free under axis x —
  the energy argument that *forced* 40° in v4 does not apply when there is no
  vertical pull. **And the min-|vy| skimming guard does not bite.** It lives
  inside `_renormalise_velocity`, which is off whenever `gravity > 0`, so the
  worry was that a paddle contact could leave a ball skimming horizontally with
  nothing to restore it. Observed minimum `|vy|` under axis x is 0.0056, *above*
  v1's own floor of 0.0050 and far above axis y's 0.0000 — the side-wind never
  touches `vy`, so a contact is the only thing that can flatten a trajectory and
  the launch guard is the only floor needed. Axis y is the axis with the
  vy problem, not axis x.
* **Pinning.** A ball uniform in x spends `2r / (1 − 2r)` ≈ 19 % of its life
  within one radius of a side wall *by geometry alone* — and gravity-free v1
  measures 19.1 %, right on it. So the 20 % threshold in the brief sits
  essentially *at* the baseline and flags a third to a half of the episodes in
  every cell, including the gravity-free ones. The honest reading is the
  **excess over v1's 19.1 %, which peaks at +2.6 points** (1e-4/0.26). The
  side-wind at these magnitudes does not pin the ball in any meaningful sense.

So the environment v4.1 asks for is *cleaner* than v4's: no stalls, no pinning,
and — the design goal — no per-traverse energy cue. It simply does not make the
sign matter to the controller.

## Why not. The recoverability table, and a closed form

`slack = frames_until_landing − |Δx| / paddle_speed`: how many frames of paddle
travel the blind oracle has spare to undo its own error. In v4 it was never
negative on any of 4,171 approach frames. Under the side-wind:

| cell | frames to landing | n | mean \|Δx\| | frac > half a paddle | mean slack | frac slack < 0 |
|---|---|---|---|---|---|---|
| x, 1e-4, w=0.26 | 20–40 | 1322 | 0.020 | 1.1% | 28.6 | **0.00%** |
| | 40–80 | 1007 | 0.069 | 26.3% | 51.4 | **0.00%** |
| | 80–120 | 444 | 0.062 | 16.2% | 95.3 | **0.00%** |
| x, 2e-4, w=0.16 | 20–40 | 1431 | 0.056 | 31.4% | 27.4 | **0.00%** |
| | 40–80 | 1069 | 0.094 | 35.1% | 50.5 | **0.00%** |
| | 80–120 | 330 | **0.105** | 38.5% | 93.2 | **0.00%** |
| *y, 1e-4, w=0.26 (v4)* | *40–80* | *494* | *0.031* | *6.9%* | *48.8* | ***0.00%*** |

The axis change **did** do its job on `|Δx|`: 0.105 against v4's 0.035, a 3×
improvement, and 38.5 % of far-out frames now differ by more than half a
paddle where v4 managed 8.4 %. Slack is still negative **zero times out of
4,338**.

### Two things cap it, and the second is fatal

**1. The box folds the parabola.** The predicted 0.3 assumed a free parabola in
x. There is no free parabola in x — there are side walls every 0.84 units, and
the ball hits one every ~40 frames. The wind does not translate the ball, it
*shifts the phase* of a bounded oscillation, and the resulting separation
saturates. Measured directly (two identical episodes, opposite fixed wind, STAY
policy, divergence of the true landing x):

| wind | mean true \|Δx\| at landing | max |
|---|---|---|
| 0.0001 | 0.132 | 0.459 |
| 0.0002 | 0.180 | 0.759 |
| 0.0005 | 0.154 | 0.663 |
| 0.001 | 0.189 | 0.748 |

**It stops growing after 2e-4.** A 10× wind buys 5 % more separation. There is
no magnitude at which this knob keeps paying.

**2. The error still vanishes at the deadline — necessarily.** This is the
part the axis change could never fix, and it has a closed form. Slack is
negative only when

    |Δx| > paddle_speed · t_land

Against that, `|Δx|` is bounded twice: it grows as `a · t_land²` (the two
hypotheses separating), and it is capped by the box at `1 − 2r`. So a frame
where the blind oracle is *forced* to lose needs **both**

    a · t² > v_p · t      ⟹  t > v_p / a          (the error must have grown)
    v_p · t < 1 − 2r      ⟹  t < (1 − 2r) / v_p   (the paddle must not have time to cross anyway)

which is a non-empty window **only if `v_p² < a · (1 − 2r)`**. With v4's paddle
(`v_p` = 0.030) and the box (`1 − 2r` = 0.84) that demands

    a > 0.030² / 0.84 = 1.07e-3

But the *same* physics that makes the ball's x-motion bounded caps the usable
wind from the other side: the ball can only cross the box against the wind if
`vx² > 2a(1 − 2r)`, and the fastest a v4 ball is ever launched horizontally is
0.0213, giving

    a < 0.0213² / (2 · 0.84) = 2.7e-4

**1.07e-3 > 2.7e-4.** The two constraints do not overlap, by a factor of four.
No side-wind exists that both makes the sign binding and leaves a playable box,
at this paddle speed. That is not a tuning failure; it is an arithmetic one.

## The appendix: what the lever actually is

`sweep.md` said the fix was to remove the slack — slow the paddle, widen the
box, or hide the ball — and not to touch gravity. The closed form above says
the same thing (`v_p²` is the term to attack). So the sweep tests it: hold the
side-wind fixed and vary only `paddle_speed`. 40 episodes/cell.

| axis | wind | paddle_w | paddle_speed | ballistic | sign-blind | **gap** | stay | frac slack < 0 |
|---|---|---|---|---|---|---|---|---|
| x | 0.0002 | 0.16 | 0.030 (v1) | 0.97 | 0.97 | 0.00 | 0.35 | 0.0% |
| x | 0.0002 | 0.16 | 0.012 | 0.96 | 0.96 | 0.00 | 0.35 | 0.0% |
| **x** | **0.0002** | **0.16** | **0.006** | **0.91** | **0.88** | **+0.03** | 0.35 | **5.9%** |
| y | 0.0001 | 0.16 | 0.006 | 0.86 | 0.87 | −0.01 | 0.35 | 0.3% |

Two readings, and the second is the one worth keeping:

* At `paddle_speed` = 0.006 the criterion `v_p² < a(1 − 2r)` finally holds
  (3.6e-5 < 1.68e-4), slack goes negative on 5.9 % of approach frames, and the
  gap becomes **the first non-zero gap in either sweep**. The mechanism is real
  and the closed form predicts where it turns on.
* **The side-wind is what makes the slow paddle work.** At the *same* slow
  paddle, vertical gravity gives slack < 0 on 0.3 % of frames and a gap of
  −0.01 — nothing. The axis change is necessary. It is just not sufficient.

+0.03 is still far short of 0.15, so this is a direction, not a cell. A v4.2
would have to combine the side-wind with a paddle slow enough (or a box wide
enough, or an occluder late enough) that the criterion holds with margin, and
then re-run this sweep. The occluder composes with `gravity_axis`, and v3.1
already demonstrated that hiding the ball on the final approach is the one
lever that has beaten a memoryless oracle in this codebase.

## What was chosen

**Nothing.** The brief's rule — stop before collecting if no cell qualifies —
applies. `data/v4` is untouched (still the vertical-gravity datasets,
1.4 GB), no datasets were re-collected, no figures regenerated, and no VAE
trained.

What *is* in the tree is the mechanism, tested and inert by default:
`BoxConfig.gravity_axis` (default `"y"`, byte-identical to v4 —
`tests/test_env_v41.py` asserts it on frames, states and events together),
the axis-x ballistic solver, `--gravity-axis` on the collector, and
`worldsim.collect.wall_pinned_fraction`. If a v4.2 gets a paddle-speed or
occluder lever, the side-wind is ready for it.

## For the record: the cue that the side-wind *does* remove

The one v4 finding that a v4.1 would have fixed, had a cell qualified. Under
vertical gravity the sign also set the ball's energy budget, so a single
traverse betrayed it — 48.0 vs 40.0 median frames, a 1.20× cue needing no
memory of the flip at all. Under a side-wind `vy` is never touched, so the
two signs have **identical** vertical dynamics: same traverse durations, same
`|vy|`, same height distribution, by construction rather than by tuning.
`tests/test_env_v41.py::test_vertical_motion_is_independent_of_the_sign`
asserts it exactly (`==`, not `allclose`) over 2,000+ frames. The cue is not
reduced; it is zero. That property survives into any v4.2 built on this axis.

---

Back to: [`sweep.md`](sweep.md) (the v4 sweep this revises),
[`../../docs/v4/00_v4_design.md`](../../docs/v4/00_v4_design.md) §2b.

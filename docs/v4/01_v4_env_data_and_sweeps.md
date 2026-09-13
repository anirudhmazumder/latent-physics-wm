# v4 — 01: The environment, two oracle sweeps, and the data

*The gravity switch as built, the two sweeps that killed its controller
question before a single model was trained, and what that leaves.*

---

## 1. The environment

`BoxConfig(gravity=1e-4, launch_min_angle_deg=40)`, everything else v1. The
ball feels a constant vertical acceleration whose direction, `gravity_sign`, is
±1 at reset and **multiplies by −1 on every paddle contact** (`EVENT_FLIP`).
Constant-speed renormalisation is off (energy is conserved by the elastic walls;
a speed cap at 0.05 fired on 5 frames in 60,000). The default configuration
stays byte-identical to v1, v2 and v3, as the regression tests check.

Why 1e-4 and 40°: at 1/64 pixel quantisation the acceleration is undetectable
in any window under ~15 frames (quadratic-fit signal-to-noise 0.25 at 10
frames, 1.4 at 20, 8 at 40) and changes the vertical speed by 27% over a
traverse; 40° keeps enough vertical energy that the ball still crosses the box
against either gravity, so both phases stay playable.

## 2. Sweep one: does the sign matter for play? — No

The v3 lesson, applied before training: run privileged policies across
candidate worlds and see whether the latent is worth anything to a controller.
Three oracles share one ballistic solver and differ in exactly one thing —
whether they know the sign:

| policy | knows |
|---|---|
| tracking oracle | the true ball x (the v1–v3.1 ceiling) |
| ballistic oracle | true state **and** the true sign; aims at the predicted landing point |
| **sign-blind oracle** | true state, sign assumed always down — the memoryless-for-this-latent bound |

Interceptions per floor visit, 60 episodes per cell:

| gravity | angle | tracking | ballistic | sign-blind | gap | stalled episodes |
|---|---|---|---|---|---|---|
| 5e-5 | 40° / 50° | 0.98 / 0.99 | 0.98 / 0.99 | 0.98 / 0.99 | **0.00** | 0% |
| **1e-4** | **40°** | 0.98 | 0.98 | 0.98 | **0.00** | 5% |
| 1.5e-4 | 40° | 0.98 | 0.98 | 0.98 | **0.00** | 20% |
| 2e-4 | 40° | 0.98 | 0.98 | 0.98 | **0.00** | 43% |

The gap is zero in every cell, and raising gravity only adds stalled episodes
(balls that no longer reach the far wall). The reason is measured on 4,171
approach frames: a wrong sign mis-predicts the landing point by at most 0.035
of the box — a quarter of a paddle — and the error shrinks to zero as the ball
approaches, while the paddle's speed does not. The blind oracle always has
slack to correct itself; it never has to commit. That is the structural
opposite of v3.1, where the band *withheld* the information until it was too
late. Here the information is never needed.

## 3. Sweep two: a side-wind instead — also no, with a closed form

If the problem is that the sign barely moves the landing point, make the
acceleration **horizontal**: a wind whose sign flips on contact. It would move
the landing x by ~0.3 in a free parabola, leave vertical motion untouched (so
no stalls and no traverse-duration cue), and keep everything else. The option
(`gravity_axis="x"`) was implemented, the oracles generalised, and the sweep
repeated across three wind strengths and two paddle widths.

| wind | paddle | ballistic | sign-blind | gap |
|---|---|---|---|---|
| 5e-5 … 2e-4 | 0.26 / 0.16 | 0.98 | 0.98 | **0.00** in all six cells |

The side-wind tripled the sign's effect (mean landing-x difference 0.035 →
0.105) and it still never bound: slack negative on 0 of 4,338 frames. Two
caps. The **box walls fold the parabola** — a side wall arrives every ~40
frames, so the wind shifts the phase of a bounded oscillation rather than
translating the ball; a 10× stronger wind moved the divergence 5%. And a
closed form: a frame where the blind oracle cannot recover needs
`paddle_speed² < a·(1 − 2r)`, i.e. `a > 1.07e-3` at the v1 paddle speed, while
the ball can only cross the box against the wind if `a < 2.7e-4`. **The two
constraints miss by 4×.** No wind is both binding and playable.

The only lever that produced any gap was a paddle five times slower
(`paddle_speed` 0.006): +0.03. Not a controller question worth a tier.

## 4. What that leaves: v4 is a dynamics-model tier

The sign is invisible per frame, holds for ~75 frames between flips, is set by
a visible event, and bends every trajectory — all of which the **M-stage**
questions need and none of which the controller needs. So v4 asks only:

- Is the sign in `h`, and for how long after the flip?
- Does the model *remember the flip* or *infer the sign from the trajectory*?
  (Under vertical gravity the sign also sets the traverse duration — 48 vs 40
  frames — a per-traverse cue the sweep found and the design had not
  anticipated. A model can know the sign without remembering anything. The
  flip counterfactual separates the two.)
- Does an LSTM that must *carry* the bit lose to a transformer that can *look
  back* at the contact frame?

The controller stage is dropped, and recorded here as a negative design
result rather than skipped silently.

## 5. The data

Vertical gravity, g = 1e-4, launch ≥ 40°, six splits under `data/v4/`
(1.4 GB): `train` (150×200, sticky), `train_mix` (300×200, mix), `val`,
`val_mix`, `probe` (120×24), and `long` (30×600 steps) for memory curves.

| | train | train_mix | long |
|---|---|---|---|
| flips per episode | 1.05 | 1.68 | ~2 per 200 frames |
| episodes with 0 / 1 / 2 / 3+ flips | 26 / 51 / 17 / 7% | 7 / 38 / 38 / 16% | |
| mean frames between flips | 74 | 81 | 78 |
| frames under gravity-down | 55% | 55% | 61% |
| stalled frames (no wall reached in 100) | 1.3% | 2.4% | 0% |

Half the training episodes contain at least one flip and a quarter contain none
— so the model sees sign-constant episodes as well as switches. Under
gravity-down a full traverse takes a median 48 frames; under gravity-up, 40.

Next: [02 — M: LSTM versus transformer on a remembered bit](02_v4_dynamics_lstm_vs_transformer.md).

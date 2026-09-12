# v3 — 02: M, and the object-permanence tests

*Does the world model know where the ball is while it cannot see it? Partly,
and the shape of "partly" names the mechanism.*

---

## 1. Setup and the control that matters

The v3 dynamics model is the same MDN-RNN, trained teacher-forced on the v3
latents:

```bash
python -m wm.train_rnn --data data/v3/train data/v3/train_mix \
    --val data/v3/val data/v3/val_mix --out runs/rnn_v3 --epochs 35
```

Three controls, same data and budget:

| run | change | what it is for |
|---|---|---|
| `rnn_v3_noact` | action input zeroed | as in v1/v2 |
| **`rnn_v3_ff`** | **no recurrence**: an MLP on `[z_t, a_t]`, same heads, its last hidden layer standing in for `h` | by construction it cannot carry anything through the gap; it is the floor for every memory test |
| `rnn_v3_ms` | 8-step open-loop rollout loss (the v2 fix machinery) | the obvious candidate cure, tested up front |

| run | val NLL | open-loop latent MSE @ 32 |
|---|---|---|
| `rnn_v3` | **3.97** | 0.45 |
| `rnn_v3_ms` | 4.69 | 0.53 |
| `rnn_v3_noact` | 5.01 | 0.63 |
| `rnn_v3_ff` | 6.24 | 0.93 |

Recurrence is worth 2.3 nats of likelihood here, more than the action input
(1.0). That already says memory is being *used*; the question is what for.

The feed-forward control is the key methodological point of this stage.
Doc 01 showed `z` carries no ball position on hidden frames, but half the
training episodes use a tracking policy, so the always-visible *paddle* is a
weak proxy for the ball's x and a probe on `z` pooled over hidden frames reads
R² ≈ 0.2, not 0. The feed-forward model has that same cue and no memory. So
"better than ff", not "better than zero", is the bar.

## 2. What is in `h` while the ball is hidden

Linear-probe R² on fully hidden frames:

| from | ball_x | ball_y | ball_vx | ball_vy | frames hidden |
|---|---|---|---|---|---|
| `z` | −0.07 | −0.10 | — | — | — |
| `h`, feed-forward control | −0.21 | −0.07 | — | — | — |
| **`h`, `rnn_v3`** | **0.17** | **0.51** | **−0.07** | **0.67** | 0.54 |

![position from h by hidden time](../../runs/rnn_v3/permanence/position_from_h_by_hidden_time.png)

Left: position error read from `h` as the occlusion goes on, against the
no-memory baseline (assume the ball is where it vanished; error grows with
speed × time). Right: R² for `ball_x`. The recurrent model is clearly above the
feed-forward floor and holds a roughly constant error for ~20 frames while the
no-memory baseline runs away — so **there is memory**. But the error never gets
below 0.17 box widths, twice the ball's diameter, and the baseline is *better*
for the first ten hidden frames, which is most of them.

The table says why. The model carries the ball's **vertical** state — `y` at
0.51, `vy` at 0.67 — and it **counts** (frames hidden at 0.54). It carries
essentially nothing horizontal: `x` at 0.17, `vx` at −0.07. The model has
learned "the ball is falling behind the band and will come out at the bottom in
about *n* frames". It has not learned *where along the band*.

**The mechanism.** Teacher forcing pays for exactly one thing: predicting the
next latent. While the ball is hidden, the next latent is the blank-band frame
*regardless of the ball's x*. Tracking `y` and the clock pays immediately — they
determine *when* the ball reappears, and getting that frame right is worth a
lot of likelihood. Tracking `x` pays nothing until the exit frame, ten steps
away, and the gradient from that one frame in ten is diluted against nine
frames where x is irrelevant. The model learned precisely what its loss paid
for. This is the v2 lesson in a new form: a quantity the world conserves but the
one-step loss does not reward is not carried.

## 3. Emergence: does the ball come out where and when it should?

For each of 109 hidden runs in validation, warm the model up to the last frame
the ball was partly visible, dream through the occlusion with the true actions,
and compare the dreamed re-emergence to the real one.

| | exit-time error (frames) | within ±2 frames | exit side correct | exit-x error (box widths) |
|---|---|---|---|---|
| `rnn_v3` | **3.2** | **60%** | **92%** | 0.159 |
| `rnn_v3_ms` | 3.1 | 66% | 95% | 0.193 |
| feed-forward | 9.5 | 7% | 41% | 0.221 (and never re-emerges a ball on 49% of runs) |
| no-memory baseline (reappears where it vanished, at the true time) | — | — | — | **0.146** |
| linear extrapolation from entry velocity, no walls | 0.6 | — | 100% | **0.034** |

Time and side: the recurrent model gets them, the feed-forward one does not —
consistent with `y`, `vy` and the clock being in `h`. Horizontal position: the
model is *worse than assuming the ball did not move*, and far worse than a
three-line physics extrapolation from the entry velocity. The information is
fully available at entry; the model does not use it.

**Hidden wall bounces** (the case where extrapolation must fail and simulation
is needed): on 49 runs with a side-wall bounce behind the band, the model's exit
x is nearer the reflected than the unreflected position 88% of the time, against
69% for the no-memory baseline and 50% chance. Weak but real — and the first
version of this test scored every model *and the baseline* at 100%, because the
alternative hypothesis ("the ball went through the wall") was implausible a
priori; it was rewritten against a clipped-at-the-wall alternative. A test on
which the null cannot fail is not a test.

**Memory horizon** (pooled over the three band heights):

![memory horizon](../../runs/rnn_v3/permanence/memory_horizon.png)

The model's exit-x error is nearly flat in occlusion length (0.13 → 0.20 from
3–8 to 26+ hidden frames) where the no-memory baseline grows linearly
(0.07 → 0.36). The two cross at about **13 hidden frames**: longer than the
default band's typical 9.5, which is exactly why the default-band number is a
loss and the taller-band numbers are wins. Caveat: on the longest occlusions the
model failed to re-emerge a ball at all on ~half the runs, and those are not
random, so the 26+ point is optimistic.

**Counterfactual entry** (re-simulate with the entry velocity mirrored, dream
both): the dreamed exit moves the right way on 55–63% of cases. Chance is 50%.
Consistent with everything above: `vx` is not in `h`.

## 4. Standard checks

The unconditional useful-dream horizon is 0 — a ruler artefact: the position
probe cannot read hidden frames, so its floor on an occluded world is 0.03
rather than 0.004. Conditioned on frames where the ball is visible, the horizon
is 14 (v1: 35, v2: 26). Velocity from `h` on visible frames and the action
counterfactual are in line with v1/v2. The v2 conservation check, applied to the
hidden ball's velocity *direction* across a dreamed occlusion, is poor: it
survives 60% of the time at τ = 0 and 50% at τ = 1 — another face of the same
missing `vx`.

## 5. What was achieved, and what to remember

**Achieved.** A clean, mechanistic answer to "does the world model have object
permanence": *vertically yes, horizontally no*. It knows the hidden ball is
falling, how long it has been hidden, and when and on which side it will
reappear (92% side, 60% of exits within two frames) — none of which a
memoryless model can do — but not where along the band, because the one-step
training loss never pays for that until the exit frame. The multi-step loss
that fixed v2's drift did nothing here, because during an occlusion the
open-loop target is the easy blank-band latent.

**Remember.**

- **A model learns what its loss pays for, when it pays for it.** Delayed
  consequences (exit x, ten frames later) are under-learned by one-step
  prediction even when the information is trivially available. This is a
  general fact about teacher-forced sequence models, and occlusion is the
  cleanest possible demonstration of it.
- **Anisotropic memory is a diagnosis, not a failure.** Which variables are
  carried tells you which ones the loss rewarded.
- **Use the right floor.** On this world "zero" is not the floor for a probe on
  hidden frames; a memoryless model with the same cues is.
- **A null that cannot fail is not a control.** The first wall-bounce test
  proved it by passing the no-memory baseline.
- **Baselines can be embarrassing, and that is the point.** Linear
  extrapolation from the entry velocity beats the model by 5× on exit x. If a
  three-line baseline beats your world model, say so; it tells you exactly
  what the model is not doing.

The fix attempt — a rollout loss long enough to put the exit frame inside the
loss, an emergence-weighted variant, and a privileged position head as a
ceiling — is in [03 — fixing horizontal permanence](03_v3_fixing_permanence.md).

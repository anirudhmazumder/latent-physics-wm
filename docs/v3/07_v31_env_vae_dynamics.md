# v3.1 — 07: Environment, V, and M on the harder band

*The same pipeline as v3 on a band that a memoryless policy cannot beat. The
encoder confound is gone, horizontal memory arrives — and the dream stops
letting the ball out.*

---

## 1. The environment and the data

`BoxConfig(occluder=True, occluder_y=(0.13, 0.63), paddle_w=0.16)`, chosen by
the oracle sweep in doc 06. Nine splits under `data/v31/` (1.9 GB): the usual
train / train_mix / val / val_mix / probe on the default band, plus training
and test splits on a shorter band (0.13, 0.45) and a longer one (0.13, 0.78).

| band | fully hidden | partial | visible | hidden run: mean / median / max | hidden runs with a wall bounce |
|---|---|---|---|---|---|
| **(0.13, 0.63)** default | **42%** | 33% | 24% | **21 / 19 / 104** | **36%** |
| (0.13, 0.45) short | 20% | — | — | 11 | 19% |
| (0.13, 0.78) long | 61% | — | — | 30 | 47% |

Against v3: hidden runs are twice as long (21 vs 9.5 frames), twice as many
contain an out-of-sight wall bounce (36% vs 18%), and the ball is out of view
for three quarters of all frames one way or another. Paddle contacts per step
fell by 7–15% with the narrower paddle, less than the width alone implies,
because the ball now re-emerges at contact height.

## 2. V — trained on all three bands, confound removed

`runs/vae_v31`, 60 epochs on 130k frames from the three bands (110 minutes):

| | default band | short | long |
|---|---|---|---|
| reconstruction MSE, v3.1 encoder | **0.00006** | 0.00010 | 0.00006 |
| reconstruction MSE, v3's encoder on the same frames | 0.028 | 0.038 | 0.048 |
| ball_x R² from a frame, visible / partial / hidden | 0.99 / 0.72 / ≤ 0 | 0.996 / 0.85 / +0.19 | 0.995 / 0.79 / ≤ 0 |

The 260× encoder confound that spoiled v3's taller-band controller test is
gone: one encoder, three bands, equal quality. The band's top edge — a
three-valued nuisance factor the code must now carry — is decodable at R²
0.997, at no cost: total KL fell again (11.9 → 11.3 nats). The `short` band's
hidden bin reads +0.19 rather than 0 because the tracking-policy paddle leaks
the ball's x, as in v3; all hidden-frame claims below are stated against the
feed-forward floor, never against zero.

## 3. M — what the longer occlusion did to memory

Same four-model design as v3: fair baseline, feed-forward floor, privileged
position-head ceiling, and the emergence-weighted fair fix; plus one model
trained on all three bands.

| model | val NLL | hidden-frame `x` R² from `h` (linear) | hidden `vy` | frames-hidden |
|---|---|---|---|---|
| **baseline `rnn_v31`** | **5.29** | **0.52** | −0.32 | −0.23 |
| feed-forward floor | 6.90 | 0.01 | — | — |
| *privileged position head* | *5.34* | *0.48* | | |
| emergence-weighted fix | 6.12 | 0.35 | | |
| all-bands | 5.21 | 0.34 (kNN 0.50) | | |
| v3 baseline, for reference | 3.97 | 0.17 | 0.67 | 0.54 |

![comparison](../../runs/rnn_v31_permanence_comparison.png)

**Horizontal permanence arrived, and closed the whole gap to the ceiling.** The
fair model's hidden-frame `x` went from 0.17 in v3 to **0.52**, level with the
privileged model's 0.48, and stays at 0.40 after 24 hidden frames where the
no-memory baseline is at −1.4. The memory-horizon crossover — the hidden
duration at which reading `h` beats assuming the ball never moved — is now 11
frames, *below* the default occlusion's 21-frame mean, where v3's ~13 was above
its 9.4. The mechanism is the one doc 02 named, running the other way: with
occlusions twice as long and twice as many hidden bounces, the loss now pays
for `x` often enough for the model to learn it. Nothing about the architecture
changed.

**Vertical permanence went away.** `vy` from `h` on hidden frames fell from
0.67 to −0.32, and the frames-hidden counter from 0.54 to −0.23. The model now
knows *where* the hidden ball is but not *when* it will come out — the exact
inverse of v3. A plausible reading: with the exit 21 frames away instead of 9,
the clock's payoff is too delayed for one-step teacher forcing, in the same way
`x`'s was in v3. Whichever variable the exit frame is closest to being
predictable from gets learned; the other does not.

**And the dream stops letting the ball out.** Dreaming through an occlusion at
τ = 0, the ball re-emerged in **5 of 99** runs for the fair model, 2 for the
ceiling, 0 for the floor. The dreamed `y` flattens at about 0.40 — inside the
band — and stays there:

![dream examples](../../runs/rnn_v31_permanence/dream_examples.png)

This kills four of the six v3 permanence experiments (emergence, hidden
bounces, memory horizon, counterfactual entry) on this geometry — the table's
exit-x and exit-time columns are five runs and must not be quoted. Two causes
compound. Geometrically, v3's band interior was a 0.14-wide strip that noise
pushed a prediction out of; v3.1's is 0.34 wide and absorbs it. Dynamically, a
model that has lost the exit clock assigns each hidden step a small exit
probability, and a deterministic (argmax) rollout never takes a step whose
probability is below one half — so the ball never leaves. Whether sampling at
τ > 0 revives the dream, and at what rate, is the first thing the controller
stage measures before training anything inside it.

Other numbers: the visible-frame dream horizon is 16 (v3: 14); velocity from
`h` on visible frames 0.46 for `vx` (v3: 0.67), contact prediction PR-AUC 0.55
(v3: 0.89) — the model is worse at everything that depends on knowing when the
ball will be where, and better at the one thing the longer occlusion trains.
Band diversity in training helped the encoder, not the dynamics.

## 4. What to carry into the controller stage

- **The environment now tests memory** (memoryless oracle 0.48 vs oracle 0.99).
- **The fair dynamics model carries the hidden ball's x** as well as the
  privileged one does. If a controller can use that, it should beat 0.48.
- **The dream may be dead as a training environment** at τ = 0. The reward head
  — trained on `h`, where `x` lives — may still be informative even when the
  decoded frame shows no ball. The controller stage must establish which
  before interpreting anything.
- **The exit-based dream metrics need a new detector** for wide bands (decoded
  ball *mass* reappearing, not decoded `y` leaving an interval). Not done.

## 5. Lessons

- **Design the loss's payoff schedule, not just the world.** The same
  architecture learned `x` and forgot the clock when the occlusion doubled.
  One-step prediction learns whichever hidden variable the nearest informative
  frame depends on.
- **Remove confounds before re-running a comparison.** Training the encoder on
  all bands cost nothing and turned v3's uninterpretable taller-band table into
  a usable one.
- **Argmax rollouts cannot cross rare events.** A deterministic dream will
  never take a step whose per-step probability is below 0.5, so it can never
  end a long occlusion. Sampled dreams can; whether they do so at the right
  time is a separate question.
- **Metrics have geometric validity conditions.** An exit detector that worked
  on a thin band silently loses its sample on a thick one.

Next: [08 — v3.1 controller: does memory buy play?](08_v31_controller.md).

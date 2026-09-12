# V2 stage two ("M") — the dynamics model, and the causal experiments

Technical log. Hardware: Apple M1, 8 GB. Interpreter
`/opt/miniconda3/envs/NN/bin/python`, all commands from the repo root. The v2
VAE (`runs/vae_v2/vae.pt`, z_dim 16) and the v1 VAE (`runs/vae_b1/vae.pt`) were
both frozen; no dataset was modified. Read
[`docs/v2/00_v2_design.md`](../docs/v2/00_v2_design.md) §3 first, then
[`README_V2.md`](README_V2.md) for what stage one established.

**The one-sentence question for stage two of v2:** M can predict the ball's
speed either by *watching it move* (the v1 skill) or by *looking at its colour*
(the new causal law) — which does it do?

**The answer, up front.** Both, and the colour route is real and measurable.
Warm the model up on a **single frame**, where no motion information exists at
all, and the dreamed ball's speed still tracks the true speed at r = **0.56**
(slope 0.30 against a probe ceiling of 0.94); the no-colour control scores
r = **0.12**, slope 0.06. Repaint a real ball a different mass's colour and
re-dream it with the same actions and the dreamed speed follows the law with a
log-log slope of **−0.60** where the truth is −1.00; the control gives
**+0.01**. `log_mass` — a quantity knowable *only* through colour — is readable
out of the recurrent state at R² **0.987**. And all of it holds inside the
held-out mass band the model never saw move.

The honest other half: the cold-start dream **systematically under-moves**
(dreamed/true speed ≈ 0.51 at K=1), so the effect is present in *direction*
much more strongly than in *magnitude*; and the useful dream horizon dropped
from v1's 35 frames to 26.

### Scoring the M-stage predictions from [`docs/v2/00_v2_design.md`](../docs/v2/00_v2_design.md)

| prediction | outcome |
|---|---|
| cold-start dreamed speed tracks true speed across masses | **yes** — r 0.563 vs control 0.120; slope 0.296 vs 0.060 |
| recolouring a dream changes its speed in the right direction | **yes** — log-log slope −0.595 vs control +0.011 (law −1) |
| speed probe error not concentrated in one mass range | **yes** — speed RMSE from `h` is 0.0014 / 0.0010 / 0.0009 across light/medium/heavy |
| the law generalises into the held-out band | **yes, on the metrics that have power there** — horizon 30.4 vs 26.3 frames, bias 0.536 vs 0.508; the *correlation* inside the band is uninformative (see §8) |
| english by mass | **not decidable** — 122 contacts, and the TRUTH itself only correlates at r 0.136 |
| the extra factor costs dream horizon | **yes** — 26 frames vs v1's 35 |

---

## 1. Files added or changed

| file | what |
|---|---|
| `wm/cache_latents.py` | `--suffix` / `latent_suffix()` — a second encoder's latents (`mu_v1vae.npy`) can live beside the default ones in the same dataset root |
| `wm/seq_data.py` | `latent_suffix=` on `_load_root`, `LatentSequenceDataset`, `make_seq_loader`, `episode_arrays`; state width no longer assumed to be 6 |
| `wm/train_rnn.py` | `--latent-suffix`, `--ablate-color` (the v2 control) |
| `wm/eval_rnn.py` | state names from `meta["state_names"]`; derived `speed` / `log_mass` targets; poly-2 added to part (c) via `_reduce_for_poly`; `horizon_by_mass()` |
| **`wm/eval_causal_v2.py`** | new — the six causal experiments |
| `tests/test_rnn_v2.py` | new — 6 tests (suite is now 40, all passing) |

Artifacts: `runs/rnn_v2/`, `runs/rnn_v2_noact/`, `runs/rnn_v2_nocolor/`,
`runs/rnn_v2/eval/`, `runs/rnn_v2/causal/`.

## 2. Commands, in order, with wall clock

```bash
# 1. the no-colour control's latents: the same v2 frames, the V1 encoder.  ~2 min
for d in train train_mix val val_mix probe holdout; do
  python -m wm.cache_latents --ckpt runs/vae_b1/vae.pt --data data/v2/$d \
      --suffix v1vae --device mps
done

# 2. three models, run CONCURRENTLY with OMP_NUM_THREADS=3.  1438 s each (~24 min)
D="--data data/v2/train data/v2/train_mix --val data/v2/val data/v2/val_mix"
python -m wm.train_rnn $D --out runs/rnn_v2         --epochs 35 --eval-every 2
python -m wm.train_rnn $D --out runs/rnn_v2_noact   --epochs 35 --eval-every 2 --ablate-actions
python -m wm.train_rnn $D --out runs/rnn_v2_nocolor --epochs 35 --eval-every 2 --ablate-color

# 3. the v1 diagnostics, generalised.  26 s
python -m wm.eval_rnn --ckpt runs/rnn_v2/rnn.pt --vae runs/vae_v2/vae.pt \
    --val data/v2/val data/v2/val_mix \
    --probe-data data/v2/probe data/v2/val_mix \
    --ablate-ckpt runs/rnn_v2_noact/rnn.pt --out runs/rnn_v2/eval

# 4. the causal experiments (both models, in one run).  16 s
python -m wm.eval_causal_v2 --out runs/rnn_v2/causal

# 5. tests.  26 s
python -m pytest tests/ -q          # 40 passed (6 of them the new test_rnn_v2)
```

Run alone, each training is ~24 s/epoch as in v1; three at once on four
performance cores it is ~41 s/epoch, hence 24 min wall clock for all three.

## 3. The no-colour control, and why it is built this way

`--ablate-color` does **not** mask a colour subspace of the v2 latent. There
isn't one: stage one found the colour code is *distributed* — no single latent
dimension carries it, and `log_mass` needs a degree-2 probe (R² 0.97) because
kNN cannot find it at all (0.08). Anything you could zero would take position
with it.

So the ablation is at the **data** level: encode the same v2 frames with the
**v1 VAE**, which was trained on a world where every ball was the same colour
and so had no reason to spend capacity on hue. Verified with `probe_suite`
before any training, on `data/v2/probe` (6000 frames, episode-level split):

| factor | v2 VAE `mu` (linear / poly2 / knn) | **v1 VAE `mu_v1vae`** |
|---|---|---|
| `ball_x` | 0.491 / 0.876 / 0.916 | 0.132 / 0.943 / **0.995** |
| `ball_y` | 0.342 / 0.796 / 0.950 | 0.314 / 0.936 / **0.995** |
| `paddle_x` | 0.662 / 0.962 / 0.773 | 0.743 / 0.985 / 0.872 |
| `mass` | −0.385 / **0.910** / 0.088 | −0.339 / **−0.766** / −0.844 |
| `log_mass` | −0.430 / **0.972** / 0.083 | −0.380 / **−0.798** / −0.886 |
| `speed` | −0.451 / **0.975** / 0.065 | −0.394 / **−0.755** / −0.876 |

**The control is clean: speed R² = −0.76 and log_mass R² = −0.80 from the v1
latents** (negative, i.e. the probe does worse than predicting the mean — which
is what "no information, plus a fitted probe that overfits slightly" looks
like). No accidental brightness channel; the colour genuinely is not there.

**Two things this buys, and one it costs.** It buys an exact counterfactual
("the same frames, minus the colour") and a model with an identical
architecture and budget. It costs comparability of two numbers: the v1 latent
space is a *different and, for position, better* code (kNN 0.995 vs 0.916, and
the poly-2 gap flips too), so **the two models' val NLLs are not comparable and
neither are their dream horizons** — they are measured in different latent
spaces with different measuring probes. Everything in §5–§10 that contrasts the
two models is a contrast of *colour-dependent* quantities, where the control's
score is the null by construction. §8's horizon comparison is the one place this
confound bites, and it is flagged there.

## 4. Training results

Best val MDN NLL (lower is better; val targets are posterior *means*, train
targets are samples — the two are not comparable, as in v1).

| run | config | best val NLL | val hit F1 @0.5 | val reward MSE | open-loop latent MSE @ h=32 |
|---|---|---|---|---|---|
| `runs/rnn_v2` | K=5 MDN, delta, 256 h | **1.394** | 0.397 | 0.00380 | 0.64 |
| `runs/rnn_v2_noact` | actions zeroed | 2.660 | 0.342 | 0.00464 | 0.89 |
| `runs/rnn_v2_nocolor` | v1-VAE latents | 3.259 † | 0.433 | 0.00362 | 0.62 |

† **not comparable to the rows above** — a different latent space, so a
different target distribution. See §3. The comparison that *is* valid is
`rnn_v2` vs `rnn_v2_noact`: zeroing the actions costs 1.27 nats, so the model
is genuinely using them.

327k params in all three (LSTM 19→256, plus MDN/hit/reward heads). Training hit
rate 0.747 %, `pos_weight = 132.9`. All three still improving slowly at epoch
35; the v1 observation that teacher-forced NLL keeps falling after the
open-loop rollout metric plateaus holds again here (`runs/rnn_v2/history.json`:
val NLL 2.53 → 1.39 from epoch 10 to 35 while roll[32] bounces between 0.37 and
0.64 with no trend).

## 5. The v1 diagnostics on v2 (`runs/rnn_v2/eval/report.json`)

35 val episodes, warm-up K=8, horizon H=64, true actions. v1's numbers in the
last column for comparison.

| diagnostic | v2 | v1 |
|---|---|---|
| (a) VAE pixel floor | 0.000187 | 0.000142 |
| (a) dream pixel MSE τ=0, h=1 / 16 / 64 | 0.00032 / 0.0039 / 0.0114 | 0.00025 / 0.0025 / 0.0085 |
| (b) **useful dream horizon** (τ=0) | **26 steps** | 35 |
| (b) probe floor ball_x / ball_y / paddle_x | 0.0062 / 0.0059 / 0.0204 | 0.0040 / 0.0032 / 0.0141 |
| (c) **velocity R² from `h`** (linear, vx / vy) | **0.892 / 0.834** | 0.918 / 0.867 |
| (c) velocity R² from `z` (knn) | −0.133 / −0.421 | −0.037 / −0.531 |
| (d) **action counterfactual separation** @H=30 | **0.573** | 0.660 |
| (d) the same, `--ablate-actions` control | 0.000 | 0.000 |
| (e) contact base rate / **PR-AUC** / best F1 | 0.84 % / **0.698** / 0.678 | 0.74 % / 0.727 / 0.759 |
| (e) P(contact) at lag −10 / −3 / 0 | 0.004 / 0.218 / 0.808 | 0.002 / 0.348 / 0.98 |
| (f) **wall bounces predicted ±2 steps** | **54.5 %** (chance 25.0 %) | 36.5 % (chance 15.5 %) |

Everything is a little worse except the walls, which are markedly better. The
uniform ~10–25 % degradation is the price of the extra factor: the model now
has to represent a family of speeds rather than one, and the *measuring
instruments* got worse too (the kNN probe's own floor is 1.4–1.9× v1's, because
the v2 latent metric is partly spent on colour — exactly the effect stage one
found when kNN failed on mass). Part of the horizon drop is therefore
instrument, not model.

The wall result is the one genuine improvement, and the likely reason is the
detector rather than the model: the reversal test requires a sustained
displacement above `0.4 × 2 × 0.022`, and in v2 two thirds of the balls move
*faster* than the v1 ball, so real reversals clear the threshold more easily.
Chance also rose (15.5 → 25.0 %), so the effect size grew by less than the
headline: 2.2× chance in v2, 2.4× in v1. **Read it as "unchanged", not
"improved".**

**New in (c): the v2 rows of the probe table** (6000 timesteps, episode-level
split, held-out R²):

| features | probe | ball_vx | ball_vy | `mass` | `log_mass` | `speed` |
|---|---|---|---|---|---|---|
| z | linear | 0.030 | −0.046 | 0.089 | 0.101 | 0.104 |
| z | **poly2** | −0.055 | −0.372 | **0.934** | **0.980** | **0.981** |
| z | knn | −0.133 | −0.421 | 0.488 | 0.517 | 0.512 |
| h | linear | 0.892 | 0.834 | 0.970 | **0.987** | **0.989** |
| h | poly2 | 0.912 | 0.909 | 0.971 | 0.987 | 0.989 |

Read the first and second rows together: **`speed` is decodable from a single
latent at R² 0.98 while its own components `vx, vy` are at R² ≈ 0.** That is
the appearance → dynamics edge stated as compactly as it can be. The frame does
not show motion; it shows colour, and colour determines the magnitude.

Note also that in `h` the colour information becomes *linearly* available
(0.101 → 0.987), the same re-coding v1 saw for position: the LSTM converts the
VAE's curved code into a flat one.

## 6. Cold start — the headline experiment (`causal/cold_start_speed.png`)

Warm up on **one** frame, dream 24 steps with the true actions at τ=0, decode
positions with the frozen kNN probe, take the **median per-step displacement
over steps 4–20** (median because a wall bounce or a probe snap inside the
window is an outlier among ~16 otherwise identical displacements).

| model / split | n | pearson r | slope | d log(speed)/d log(m) | dreamed speed | true speed |
|---|---|---|---|---|---|---|
| `rnn_v2` / val | 175 | **+0.563** | **+0.296** | **−0.587** | 0.0125 ± 0.0058 | 0.0246 ± 0.0111 |
| `rnn_v2` / holdout | 150 | +0.138 | +0.324 | −0.338 | 0.0117 ± 0.0040 | 0.0219 ± 0.0017 |
| `rnn_v2_nocolor` / val | 175 | +0.120 | +0.060 | −0.023 | 0.0113 ± 0.0055 | 0.0246 |
| `rnn_v2_nocolor` / holdout | 150 | −0.004 | −0.016 | +0.375 | 0.0104 ± 0.0063 | 0.0219 |
| *probe ceiling* (same estimator on TRUE latents) | | +0.981 | +0.941 | −0.936 | | |

A model that could not see colour would sit on the control's row. It does not.

**The failure inside the success: the K=1 dream under-moves by about half**
(0.0125 dreamed vs 0.0246 true; relative bias 0.51). With an empty hidden state
the model hedges toward a slow ball, and the colour shifts that hedge in the
right direction without restoring its scale. So the cold-start result is strong
evidence of *ordering* and weak evidence of *calibration*. The log-log slope
(−0.59 against a law of −1 and a probe ceiling of −0.94) is the number that
states this honestly.

**Speed vs warm-up length** (`speed_vs_warmup.png`), which is the cleanest
single picture of the whole stage:

| K | `rnn_v2` r / slope | `rnn_v2_nocolor` r / slope |
|---|---|---|
| 1 | **+0.459 / +0.302** | **−0.065 / −0.031** |
| 2 | +0.751 / +0.631 | +0.653 / +0.405 |
| 4 | +0.956 / +0.936 | +0.916 / +0.849 |
| 8 | +0.937 / +0.971 | +0.958 / +0.972 |

The two models are indistinguishable by K=8 and separated completely at K=1.
**That gap is the colour channel, isolated.** It also explains why teacher-forced
NLL could never have found this: after two frames the colour is redundant, and
almost every transition the loss ever sees has more than two frames of history
behind it.

## 7. Recolour — the intervention (`causal/recolor_speed.png`, `recolor_counterfactual.png/.gif`)

Take a real val episode at t₀ = T/3, re-render its warm-up frames from the
recorded state with the ball repainted as mass m1 (exact: the renderer is a pure
function of position + colour, and `tests/test_rnn_v2.py` checks that
re-rendering with the *original* mass reproduces the dataset's frames
bit-for-bit), re-encode, dream forward with the same actions.

There is no ground truth here — a repainted ball never existed and its real
trajectory would have been different — so the comparison is **dream against
law**, which is what an interventional test is.

Mean dreamed speed at K=8, 12 source episodes:

| m1 | 0.5 | 0.7 | 1.0 † | 1.4 | 2.0 |
|---|---|---|---|---|---|
| true law 0.022/m1 | 0.0440 | 0.0314 | 0.0220 | 0.0157 | 0.0110 |
| **`rnn_v2`** | 0.0292 | 0.0267 | 0.0230 | 0.0163 | 0.0130 |
| `rnn_v2_nocolor` | 0.0262 | 0.0246 | 0.0225 | 0.0246 | 0.0256 |

† inside the held-out band [0.85, 1.2] — a colour whose dynamics M never saw.
It is the closest point to the law in the whole table (0.0230 vs 0.0220).

| model / warm-up | d log(speed) / d log(m1) | per-source median (IQR) | r |
|---|---|---|---|
| `rnn_v2` K=8 | **−0.595** | −0.557 (0.454) | −0.707 |
| `rnn_v2_nocolor` K=8 | **+0.011** | −0.006 (0.118) | +0.012 |
| `rnn_v2` K=1 | −0.346 | −0.351 (0.332) | −0.222 |
| `rnn_v2_nocolor` K=1 | −0.087 | −0.070 (0.099) | −0.105 |

**K=8 is the usable number, and it is the *harder* test**, which surprised me.
At K=8 the eight warm-up frames show the ball moving at its *original* speed
while wearing the *new* colour — colour and motion contradict each other — and
the model still follows the colour to within a factor of ~1.7 of the law. At
K=1 there is no contradiction but also no scale (§6's under-movement compresses
every speed toward 0.009–0.015), so the K=1 slope is weaker purely as a
measurement artefact. Both are reported; the second panel of
`recolor_speed.png` shows the compression plainly.

**Does the repaint survive the dream?** (`recolor_color_persistence.png`.)
Decode every dreamed frame, find the most ball-like pixel, project it back onto
the ramp with `worldsim.color_to_mass`:

| repainted as m1 | 0.5 | 0.7 | 1.0 | 1.4 | 2.0 |
|---|---|---|---|---|---|
| mass read back, dream step 0 | 0.54 | 0.64 | 1.00 | 1.41 | 1.75 |
| mass read back, dream step 23 | 0.58 | 0.67 | 1.17 | 1.42 | 1.66 |

The colour is carried faithfully through 24 steps of the model's own dreaming —
so M is not merely reacting to colour at the boundary, it is *propagating* it as
part of the state. The two extremes are compressed (2.0 reads as ~1.7), which
is a VAE round-trip limitation on repaints far from the source colour, not a
dynamics one; the intervention is therefore slightly *weaker* than intended at
the ends, which makes the measured slope a lower bound.

`recolor_counterfactual.png` is the figure to look at: one source episode, six
rows (original + five repaints), six dream steps across. Same start, same
actions, only the hue differs, and the m1=0.5 ball has crossed most of the box
by step 23 while the m1=2.0 ball has barely moved.

## 8. Interpolation into the held-out band

| metric | in-distribution | held-out band [0.85, 1.2] |
|---|---|---|
| useful dream horizon, `rnn_v2` | 26.3 frames | **30.4** |
| useful dream horizon, `rnn_v2_nocolor` | 30.1 | 31.2 |
| cold-start r | +0.563 | +0.138 |
| cold-start relative bias (dreamed/true) | 0.508 | **0.536** |
| recolour at m1 = 1.0 (inside the band) | — | 0.0230 vs law 0.0220 |

**The held-out band is handled as well as anything else**, and by the horizon
metric slightly better. There is no sign of a lookup table.

**But the cold-start correlation inside the band is not evidence of anything,
in either direction.** The band spans masses 0.85–1.2, i.e. speeds 0.0183–0.0259
— a ±8 % range. The *measuring probe's own* r on true latents inside the band
is only +0.387 (against +0.981 outside it), so the instrument cannot resolve
what is being asked. Range restriction, not model failure. The scale-free bias
and the recolour point at m1=1.0 are the metrics with power there, and both say
the interpolation is fine.

**The confound to keep in mind** (§3): `rnn_v2_nocolor` shows a *longer* horizon
than `rnn_v2` (30.1 vs 26.3). That is very unlikely to mean "colour hurts". The
v1 latent space codes position better (kNN 0.995 vs 0.916) and the measuring
probe fitted in it is correspondingly sharper, so both the dynamics and the
ruler are favoured. Horizon numbers should only be compared *within* a model.

## 9. Horizon by mass (`causal/horizon_by_mass.png`)

`rnn_v2`, in-distribution, K=8, per-episode horizons averaged:

| tercile | mass | speed | horizon (frames) | horizon (ball diameters travelled) |
|---|---|---|---|---|
| light | 0.51–0.69 | 0.038 | 23.9 | **5.53** |
| medium | 0.69–1.48 | 0.023 | 25.9 | 3.62 |
| heavy | 1.49–1.94 | 0.013 | 29.0 | **2.32** |

Light (fast) balls do drift out of tolerance sooner **in frames**, as predicted
— but the frame counts differ by only 21 % while the speeds differ by 3×, so in
**ball-diameters travelled the ordering reverses and the spread is 2.4×**: the
model tracks a fast ball for more than twice as far as a slow one before losing
it. The naive expectation ("same accuracy per unit distance, so fewer frames for
fast balls") is wrong in the other direction. Plausible reason: a fast ball's
per-step latent delta is large compared with the VAE's posterior noise, so its
dynamics are a higher-SNR signal; a slow ball's motion is closer to the noise
floor and its dreamed position random-walks. Not verified — a follow-up would
re-run with `--use-mean` latents, which would remove that noise.

The same table on the `eval_rnn` pooled run (`runs/rnn_v2/eval`, 30 episodes,
different subsample) gives 26.9 / 28.5 / 26.8 frames and 6.00 / 3.68 / 2.09
diameters — the diameter ordering is robust, the frame ordering is not. **Read
the frame column as "roughly flat".**

## 10. English by mass (`causal/english_by_mass.png`) — best effort, not decidable

122 real paddle contacts across val + val_mix + holdout, which clears the
"< 40 and say so" bar. Warm up on 6 true frames ending 5 steps before the
contact, dream 14 steps across it, finite-difference the probed ball position
either side.

| | r(abs Δvx, 1/m) | r(dream Δvx, true Δvx) |
|---|---|---|
| **truth** | **+0.136** | — |
| `rnn_v2` dream | +0.129 | +0.552 |
| `rnn_v2_nocolor` dream | +0.084 | +0.528 |

The dream reproduces the true correlation almost exactly (0.129 vs 0.136) and
tracks the per-contact deflection at r = 0.55. But **the truth itself only
correlates at 0.136**, because `Δvx = 0.35 · paddle_vx / m` and `paddle_vx` is
uncontrolled and varies far more than `1/m` does across these 122 contacts. The
experiment as designed has almost no power; the `rnn_v2` − `rnn_v2_nocolor` gap
(0.129 vs 0.084) is in the right direction and nowhere near significant at
n = 122. **Reported as inconclusive.** A real version would need contacts
stratified by `paddle_vx`, which means collecting data with a policy that drives
the paddle hard through the ball — a data-collection change, not an analysis one.

## 11. Surprises, failure modes, and things I changed

1. **A poly-2 probe on a 256-unit hidden state is 33,000 features.** The first
   version of the generalised part (c) tried it and hung the machine. Fixed with
   `_reduce_for_poly`: PCA to 32 components before the polynomial expansion,
   run as a *separate* `probe_suite` call with the same seed and group ids so it
   is the same split. The PCA is unsupervised but is fitted on both halves of
   the split — a small, target-blind leak, recorded rather than hidden.
2. **The nocolor control's NLL is not comparable to the main model's** (3.26 vs
   1.39). Different encoder, different latent space, different target
   distribution. This is the single easiest number in this document to misread
   and it is why the training table carries a dagger. Same for dream horizons.
3. **K=8 beats K=1 on the recolour test**, the opposite of what I expected when
   I added the K=1 variant to "remove the confounding motion cue". The reason is
   that K=1 dreams are globally too slow (§6), which compresses the very
   quantity being measured. The K=1 panel is kept because it shows the
   compression, but K=8 is the number.
4. **The cold-start dream under-moves by ~2×.** The direction is right, the
   magnitude is not. I considered calibrating the estimator against the probe
   ceiling and decided against it — a corrected number would hide a real
   property of the model. Both the raw slope and the ceiling are reported.
5. **The correlation metric is useless inside the held-out band** and I nearly
   reported it as a failure of interpolation. The band is ±8 % in speed and the
   *probe's own* correlation there is 0.387. Added the log-log slope and the
   scale-free relative bias, which do have power, plus the m1 = 1.0 recolour
   point, which is a within-episode design and the strongest interpolation
   evidence in the document.
6. **Fast balls are dreamt for more than twice as far as slow ones** in
   ball-diameters, which is backwards from the prediction in the design doc.
   See §9. Not chased down.
7. **The repainted colour survives 24 steps of dreaming** — I expected it to
   drift back toward the source colour within a few steps, and it does not
   (§7). The residual compression at the extremes is a VAE round-trip effect,
   which makes the measured recolour slope a lower bound.
8. **The v1 VAE encodes v2 positions *better* than the v2 VAE does** (kNN 0.995
   vs 0.916). Not spending ~1 nat on colour leaves more of a 16-d code for
   position. A nice, concrete illustration that latent capacity is a budget.
9. **The GIF has no captions**, so `recolor_counterfactual.gif` alone is
   unreadable as evidence (which column is which?). Added a labelled matplotlib
   contact sheet, `recolor_counterfactual.png`, which is the figure that should
   actually go in a report.
10. The `eval_rnn` bounce test still reports "better than chance", not "knows
    about walls" — the v1 caveat is unchanged and the improvement from 36.5 % to
    54.5 % is mostly a threshold effect (§5).

## 12. Choices made without asking

* `--ablate-color` is implemented as a data-level encoder swap, not a latent
  mask; `--latent-suffix v1vae` is the general mechanism and the flag is
  shorthand for it. Reasons in §3 and in the `train_rnn.py` docstring.
* The dreamed-speed estimator is the **median** per-step displacement over dream
  steps 4–20 of a 24-step dream (skip while `h` settles, stop before the dream
  drifts, median for robustness to bounces and probe snaps).
* Recolouring is done by re-rendering from recorded state rather than
  re-simulating, so it is exact and testable; verified bit-identical to the
  dataset's own frames when the mass is unchanged.
* Part (b) is run at two warm-up lengths (8 as specified, and 1) because they
  ask different questions; §7 says which to read.
* Per-episode useful horizons are averaged, rather than reading one horizon off
  the tercile-mean error curve — a mean curve is dominated by whichever episode
  diverged first and is systematically pessimistic.
* Mass terciles are computed per split, so the held-out band's "light/medium/
  heavy" are 0.89/1.00/1.10, not comparable to val's 0.6/1.0/1.7. They are there
  to show the band is uniform inside itself, not to compare across splits.
* The state probe and the decoder are always taken from the *same* latent space
  as the model being measured (the `Setup` dataclass exists to make that
  impossible to get wrong).
* `--starts-per-ep 5`, `--n-recolor-src 12`, probe fitted on
  `data/v2/probe + data/v2/val_mix` (12k frames), frozen before any dream.

## 13. What stage three inherits

`runs/rnn_v2/rnn.pt` is the dream environment for C. Two properties matter for
it. The reward head is as good as v1's (val MSE 0.0038) and the contact head is
slightly worse (PR-AUC 0.698 vs 0.727). And the model **does** carry mass and
speed in `h` at R² 0.987 linearly — so a *linear* controller reading `[z, h]`
has the mass available to it as a linear feature, which is exactly what the
design doc's "interaction term" prediction needs. The useful dream horizon of 26
frames is the budget the controller has to work inside; a light ball crosses the
box vertically in ~12 frames, so that is still 2 traversals.

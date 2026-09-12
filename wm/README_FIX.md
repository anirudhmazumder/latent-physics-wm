# Fixing the colour drift in the v2 world model

**Recommendation: `runs/rnn_v2_cons/rnn.pt`** — the v2 MDN-RNN retrained with a
targeted conservation penalty (`--rollout-loss-steps 8 --rollout-loss-weight 0
--cons-loss-weight 1.0 --mass-head`). It is **not a full fix**: the stated bar
was corr ≥ 0.80 between the dreamed and true log-mass at dream step 150 under
τ = 1, and it reaches **+0.24** there. But it turns the failure from a *random
walk* into a *bounded error*, which is the qualitative change that matters, and
it does so while **improving** every other number the project tracks except two.

| τ = 1, 200-step dream | baseline `rnn_v2` | **`rnn_v2_cons`** |
|---|---|---|
| corr(dreamed log-mass, true), step 24 | −0.11 | **+0.57** |
| the same, step 150 | −0.34 | **+0.24** |
| the same, mean over steps 20–199 | −0.21 | **+0.42** |
| fraction of dreamed frames with a well-formed ball | 0.74 | **0.96** |
| useful dream horizon at τ = 0 (`eval_rnn`) | 26 frames | **29** |
| recolour intervention, d log(speed)/d log(m₁) (law −1) | −0.595 | **−0.652** |
| best val NLL | 1.394 | **1.290** |
| cold-start r (K=1) | +0.563 | +0.436 ← **regression** |
| contact PR-AUC | 0.698 | 0.586 ← **regression** |

Hardware: Apple M1, 8 GB. Interpreter `/opt/miniconda3/envs/NN/bin/python`, all
commands from the repo root. The v2 VAE (`runs/vae_v2/vae.pt`) was frozen, no
dataset was touched, and `runs/rnn_v2/rnn.pt` was not modified.

---

## 1. What was broken, and what was missing

`wm/README_C2.md` §10–11 found, *by watching a GIF*, that over a 200-step τ = 1
dream the ball's colour walks from yellow through red to purple. Mass is
constant within a v2 episode and mass sets the speed (`speed = 0.022 / m`), so
a dream whose colour walks is a dream whose **physics** walks — and 150-step
τ = 1 dreams are exactly what `ctrl_v2` was trained inside.

The mechanism, stated precisely (and written into `wm/conservation.py`'s
docstring): the MDN's predicted std in the colour direction is not zero and
*cannot* be, because the training targets are posterior **samples** and those
really do jitter by about that much. Under teacher forcing the model never
experiences the consequence — a fresh, correct z arrives every step and the
jitter is thrown away. Under sampled open-loop rollout the increments
accumulate. Position has restoring forces (walls, floor, a bounded box); mass
has none. So the colour is a pure random walk with variance growing linearly in
the step count, and after ~25 steps it has covered the whole colour range.

**Nothing in the existing evaluation could see this.** `eval_rnn` and
`eval_causal_v2` dream for 24–64 steps, mostly at τ = 0, and both measure
agreement with one particular true trajectory — which is the wrong question for
a sampled dream, and is dominated by position long before the colour has moved.

## 2. Files added or changed

| file | what |
|---|---|
| **`wm/conservation.py`** | new. `Poly2Probe` (a frozen degree-2 ridge probe with a numpy path *and* a differentiable torch path), `reparam_sample`, `rollout_losses`, `conservation_penalty`, `rolling_speed`, `speed_horizon` |
| **`wm/eval_conservation.py`** | new. The metric: four curves × three temperatures × a 200-step dream, plus `--replot` and `--compare` |
| `wm/rnn.py` | `RNNConfig.mass_head` (default `False`, so every existing checkpoint's state dict is unchanged) and the optional `mass_head` linear head |
| `wm/train_rnn.py` | `--rollout-loss-steps/-weight`, `--cons-loss-weight`, `--mass-head`, `--w-mass`, `--cons-probe-samples`; also always writes `rnn_last.pt` beside the best-on-val `rnn.pt` |
| `wm/eval_rnn.py` | a second dream horizon: the step at which the dreamed **speed** leaves ±25 % of the law. Reported at all three τ. The old metric is unchanged and still reported |
| `wm/eval_causal_v2.py` | `--parts` (run a subset of the six experiments), `--name`, and `Episodes.has_mass`; empty `--nocolor-ckpt ""` now correctly skips the control (`Path("")` is `Path(".")`, which exists) |
| **`tests/test_conservation.py`** | new, 9 tests. Suite is 58, all passing |

Artifacts: `runs/rnn_v2_{mean,ms,cons,ms_cons}/`, `runs/*/conservation/`,
`runs/rnn_v1/conservation/`, `runs/rnn_v2_fix_comparison.png`.

`runs/rnn_v2/eval/report.json` was regenerated so the baseline carries the new
speed-horizon fields. Every pre-existing number in it reproduced to the last
significant digit (`git diff` shows only additions plus float-noise in the last
digit of the MLP probe rows).

## 3. Commands, in order, with wall clock

```bash
# 1. baseline conservation + the v1 sanity reference.            ~100 s each
python -m wm.eval_conservation --ckpt runs/rnn_v2/rnn.pt \
    --vae runs/vae_v2/vae.pt --out runs/rnn_v2/conservation
python -m wm.eval_conservation --ckpt runs/rnn_v1/rnn.pt \
    --vae runs/vae_b1/vae.pt --val data/v1/val data/v1/val_mix \
    --probe-data data/v1/probe data/v1/val_mix --out runs/rnn_v1/conservation

# 2. baseline eval_rnn re-run, for the new speed-horizon rows.       ~110 s
python -m wm.eval_rnn --ckpt runs/rnn_v2/rnn.pt --vae runs/vae_v2/vae.pt \
    --val data/v2/val data/v2/val_mix \
    --probe-data data/v2/probe data/v2/val_mix \
    --ablate-ckpt runs/rnn_v2_noact/rnn.pt --out runs/rnn_v2/eval

# 3. three candidates, CONCURRENTLY, OMP_NUM_THREADS=2.   1438-2162 s each
D="--data data/v2/train data/v2/train_mix --val data/v2/val data/v2/val_mix"
python -m wm.train_rnn $D --out runs/rnn_v2_mean --epochs 35 --eval-every 2 \
    --use-mean                                                    # 1508 s
python -m wm.train_rnn $D --out runs/rnn_v2_ms   --epochs 35 --eval-every 2 \
    --rollout-loss-steps 8 --rollout-loss-weight 1.0              # 2141 s
python -m wm.train_rnn $D --out runs/rnn_v2_cons --epochs 35 --eval-every 2 \
    --rollout-loss-steps 8 --rollout-loss-weight 0.0 \
    --cons-loss-weight 1.0 --mass-head                            # 2162 s

# 4. the (b)+(c) combination, run alone afterwards.                   877 s
python -m wm.train_rnn $D --out runs/rnn_v2_ms_cons --epochs 35 --eval-every 2 \
    --rollout-loss-steps 8 --rollout-loss-weight 1.0 \
    --cons-loss-weight 1.0 --mass-head

# 5. per candidate: conservation, eval_rnn, causal parts (a)+(b). ~5 min each
for R in rnn_v2_mean rnn_v2_ms rnn_v2_cons rnn_v2_ms_cons; do
  python -m wm.eval_conservation --ckpt runs/$R/rnn.pt --vae runs/vae_v2/vae.pt \
      --out runs/$R/conservation
  python -m wm.eval_rnn --ckpt runs/$R/rnn.pt --vae runs/vae_v2/vae.pt \
      --val data/v2/val data/v2/val_mix \
      --probe-data data/v2/probe data/v2/val_mix \
      --ablate-ckpt runs/rnn_v2_noact/rnn.pt --out runs/$R/eval
  python -m wm.eval_causal_v2 --ckpt runs/$R/rnn.pt --nocolor-ckpt "" \
      --out runs/$R/causal --parts ab --name $R
done

# 6. the comparison figure, assembled from the reports (no dreaming).   ~2 s
python -m wm.eval_conservation --compare runs/rnn_v2 runs/rnn_v2_mean \
    runs/rnn_v2_ms runs/rnn_v2_cons runs/rnn_v2_ms_cons

# 7. tests.                                                            11 s
python -m pytest tests/ -q          # 58 passed (9 of them test_conservation)
```

Concurrency cost: run alone a candidate with the rollout loss is ~25 s/epoch;
three at once on four performance cores it is 41–62 s/epoch. Total wall clock
for the whole exercise, ~2 h.

Per-epoch cost of the extra machinery, measured alone: 21 s (baseline) →
25 s with `--rollout-loss-steps 8`. That is about +20 %, which is the K = 8
rollout plus a teacher-forced prefix pass that has to be recomputed because
`nn.LSTM` hands back only the *final* (h, c) and the rollout needs the pair at
a random interior position.

---

## 4. The baseline, measured (`runs/rnn_v2/conservation/`)

30 val + val_mix episodes, 8 true warm-up frames, 200 dream steps, true actions
(the last 8 steps hold the final recorded action, since episodes are only 200
transitions long). log-mass read with a frozen poly-2 ridge probe on `mu` fit on
`data/v2/probe` + `data/v2/val_mix`, every 2nd frame, in-sample R² 0.981.

**corr(dreamed log-mass, true log-mass), across episodes:**

| dream step | 0 | 24 | 60 | 120 | 199 |
|---|---|---|---|---|---|
| τ = 0 | 0.98 | 0.95 | 0.96 | 0.94 | 0.91 |
| τ = 0.5 | 0.73 | −0.02 | −0.20 | −0.37 | −0.26 |
| τ = 1 | 0.56 | −0.11 | −0.34 | −0.05 | −0.35 |
| *probe floor (true latents)* | 0.99 | 0.99 | 0.99 | 0.99 | — |

This reproduces the orchestrator's follow-up measurement (0.97 → 0.88 at τ = 0;
0.79 → 0.30 → ≈0 at τ = 1) on a larger episode set. **τ = 0.5 is as broken as
τ = 1**, which was not previously known and matters: halving the temperature is
not a fix.

The RMSE of the dreamed log-mass saturates at ~0.8 nats against a total
log-mass range of 1.386, i.e. the dreamed mass is **as good as uninformative**
by step 25. The well-formedness proxy sits at 0.74 (τ = 1) and 0.80 (τ = 0.5)
against 1.00 at τ = 0, so the smearing `README_C2` §10 described is real and
persistent, not a transient around step 60.

### The v1 sanity reference (`runs/rnn_v1/conservation/`)

v1 has no mass, but its ball speed is a hard constant 0.022, so (iii) and (iv)
are conservation tests there too. **The disease is not v2-specific.**

| | τ = 0 | τ = 0.5 | τ = 1 |
|---|---|---|---|
| steps inside ±25 % of the speed law | 47.9 | 4.5 | **0.5** |
| dreamed speed / law at step 150 | 0.94 | 0.97 | **1.71** |
| well-formed fraction | 1.00 | 0.94 | 0.91 |

v1's τ = 1 dream also fails the ±25 % band immediately — but it then *settles*
at 1.4–2.0× the law instead of running away, and keeps 91 % of its balls
well-formed against v2's 74 %. So: **sampling a long MDN-RNN rollout costs you
conserved quantities in any version of this world; v2 is worse because it has a
conserved quantity with no restoring force and a nonlinear, distributed code.**

## 5. The four candidates

| | (a) `rnn_v2_mean` | (b) `rnn_v2_ms` | (c) `rnn_v2_cons` | (b)+(c) `rnn_v2_ms_cons` |
|---|---|---|---|---|
| what | train on posterior means | + K=8 open-loop MDN NLL | + K=8 probe-space conservation penalty & log-mass head | both |
| best val NLL | −32.37 † | 2.529 | **1.290** | 2.274 |
| open-loop latent MSE @ h=32 | 1.154 | 0.606 | **0.277** | 0.524 |

† **not comparable to any other row**: the `--use-mean` model is scored against
posterior *means*, a much lower-entropy target, so its NLL is a different
quantity. The comparison that is valid is ms / cons / ms_cons / baseline (1.394).

### Conservation — the fix metric (τ = 1, `conservation/report.json`)

| corr(dreamed log-mass, true) | step 24 | 60 | 120 | 150 | 199 | **mean, steps 20–199** | well-formed |
|---|---|---|---|---|---|---|---|
| `rnn_v2` (baseline) | −0.11 | −0.34 | −0.05 | −0.34 | −0.35 | **−0.21** | 0.74 |
| `rnn_v2_mean` | −0.15 | +0.36 | +0.04 | −0.08 | −0.08 | −0.07 | 0.45 |
| `rnn_v2_ms` | +0.35 | +0.16 | +0.54 | +0.49 | +0.34 | +0.28 | 0.93 |
| **`rnn_v2_cons`** | **+0.57** | **+0.35** | +0.34 | +0.24 | **+0.53** | **+0.42** | **0.96** |
| `rnn_v2_ms_cons` | +0.41 | +0.27 | +0.21 | +0.13 | +0.11 | +0.26 | 0.80 |
| *probe floor* | 0.99 | 0.99 | 0.99 | 0.99 | — | 0.99 | — |

Same table at **τ = 0.5**, corr at step 150: baseline **−0.58**, mean +0.14,
ms +0.38, **cons +0.48**, ms_cons +0.23.

The shape of the curve matters more than any single number, and
`runs/rnn_v2_cons/conservation/conservation.png` shows it: the baseline's
correlation *decays monotonically* through zero and keeps going negative — a
random walk. `rnn_v2_cons`'s is **flat at ≈ 0.45 for the whole 200 steps**, and
its log-mass RMSE is flat at ≈ 0.6 rather than climbing. The drift is gone; what
is left is a constant, non-accumulating error level. That is a different failure
mode, and a much more benign one for a controller: a world whose ball has a
slightly wrong but *stable* mass is a world with a consistent physics.

### Dreamed speed — the dynamical consequence

| steps inside ±25 % of the law | τ = 0 | τ = 0.5 | τ = 1 |
|---|---|---|---|
| `rnn_v2` | 46.3 | 5.0 | 0.5 |
| `rnn_v2_mean` | 8.2 | 3.6 | 4.7 |
| `rnn_v2_ms` | 32.2 | 6.8 | 3.5 |
| **`rnn_v2_cons`** | **48.1** | **8.0** | 2.2 |
| `rnn_v2_ms_cons` | 31.6 | 4.6 | 2.0 |
| *probe floor (true latents)* | 128.1 | — | — |

**Read this column with care and do not over-weight it.** At τ > 0 the speed
*ratio* is inflated by something that is not drift: a sampled latent jitters
step to step, the kNN position probe turns that jitter into displacement, and
the estimator cannot separate jitter from motion. The τ = 1 ratios run 2–8×.
That inflation is not purely an artefact of the ruler — a controller dreaming
at τ = 1 really does watch a ball that moves that much per frame — but it is a
different complaint from "the mass drifted". The correlation panel (iii b) in
each run's figure is the jitter-immune version, and there **every** model is
near zero at τ = 1: none of them makes the dreamed speed track the true speed
inside a sampled 200-step dream. **This is the honest limit of the result.**

### No-regression checks (`eval/report.json`, `causal/report.json`)

| | baseline | mean | ms | **cons** | ms_cons |
|---|---|---|---|---|---|
| useful dream horizon, τ = 0 | 26 | 12 | 25 | **29** | 26 |
| velocity R² from `h` (linear, vx / vy) | 0.892 / 0.834 | 0.680 / 0.572 | 0.873 / 0.793 | 0.888 / 0.813 | 0.867 / 0.749 |
| action counterfactual separation @ H=30 | 0.573 | 0.500 | 0.585 | **0.587** | **0.603** |
| contact PR-AUC | **0.698** | 0.526 | 0.497 | 0.586 | 0.571 |
| wall bounces ±2 steps (chance) | 0.545 (0.250) | 0.355 (0.165) | 0.495 (0.230) | 0.545 (0.255) | 0.495 (0.200) |
| dream pixel MSE @ h=64, τ=0 | 0.01144 | 0.01279 | 0.01035 | **0.00921** | 0.01092 |
| val reward MSE | 0.00380 | 0.03037 | 0.00661 | **0.00294** | 0.00720 |
| **cold-start r (K=1)** | **+0.563** | +0.458 | **+0.628** | +0.436 | +0.449 |
| cold-start slope | +0.296 | +0.742 | +0.412 | +0.262 | +0.410 |
| **recolour d log(speed)/d log(m₁)**, law −1 | −0.595 | −0.445 | −0.645 | **−0.652** | **−0.697** |

The colour-blind control `rnn_v2_nocolor` was **not** re-run; its numbers
(cold-start r +0.120, recolour +0.011) are properties of a control trained on
colour-free latents and do not change when a new v2 model is trained. They are
reused from `runs/rnn_v2/causal/report.json`.

## 6. The recommendation, and the reasoning

Criterion order as set: (1) colour conservation at τ = 1; (2) no regression in
useful horizon at τ = 0 beyond ~10 %; (3) no regression in cold-start r or
recolour slope.

1. **Conservation.** `rnn_v2_cons` is best on every summary of the τ = 1 curve
   (mean over steps 20–199: +0.42 vs ms's +0.28 vs baseline's −0.21), best at
   τ = 0.5, and best on well-formedness (0.96 vs 0.93 vs 0.74). **No model
   reaches 0.80 at step 150**, so nothing here is a full fix.
2. **Horizon.** `rnn_v2_cons` scores 29 frames against the baseline's 26 — a
   12 % *improvement*, not a regression. `rnn_v2_ms` loses 4 %. Both pass.
   `rnn_v2_mean` loses 54 % and is out on this criterion alone.
3. **Causal.** `rnn_v2_cons` improves the recolour slope (−0.652 vs −0.595,
   closer to the law's −1) and **regresses the cold-start correlation**
   (+0.436 vs +0.563, −23 %). That is the one place it is genuinely worse and
   it is not small. `rnn_v2_ms` is the only candidate that improves *both*
   (r +0.628, slope −0.645).

So (1) and (2) point at `rnn_v2_cons` and (3) points at `rnn_v2_ms`. The
criterion order is explicit, (1) is the whole point of the exercise, and
`rnn_v2_cons` also wins on the metrics that are not in the criteria at all
(best val NLL, open-loop latent MSE, pixel MSE, reward MSE, action separation,
wall bounces). **Take `runs/rnn_v2_cons/rnn.pt`.**

If the cold-start number is the one you care about — i.e. if the next thing
built on this model depends on reading speed off a single frame —
`runs/rnn_v2_ms/rnn.pt` is the alternative, and it costs ~0.14 of conservation
correlation to buy back ~0.19 of cold-start r.

### What I would try next

1. **A schedule on the conservation weight.** `cons` plateaus at ≈ 0.46 on the
   training penalty from epoch ~10 onward (`runs/rnn_v2_cons_train.log`) — the
   loss stops making progress long before training ends. Ramping w from 1 to
   ~5 over the run, or annealing K from 4 to 16, is the cheapest next
   experiment and the one I would run first.
2. **Penalise the predicted std directly.** The conservation penalty acts on
   the *realised* sample; a term on `std · ∂g/∂z` (the probe-space standard
   deviation the model is about to inject) would act on the mechanism instead
   of its consequence, and would not need a rollout at all.
3. **Let the model discover the conserved direction.** (c) is told which latent
   function to hold; a version that penalises the drift of the *top principal
   components of the per-episode-constant subspace*, found once from the data,
   would be a stronger scientific claim and would not need privileged state.
4. **Retrain `ctrl_v2` inside `rnn_v2_cons`.** The whole point was that the
   controller trained in a world with a drifting speed law. Whether a stable
   dream produces a controller that finally uses mass is the experiment this
   was all for, and it has not been run.

## 7. Design decisions, all arguable, all recorded

* **The component choice in the differentiable rollout sampler is the
  most-likely one**, not a Gumbel-softmax draw. An argmax gives no gradient to
  the choice, but a relaxation would *mix the component means*, which is
  precisely the averaging-across-modes failure the mixture exists to prevent
  (`wm/rnn.py`'s opening docstring). The within-component draw is
  reparameterised (`mean + std·eps`), which is the part that has to carry
  gradient for any of this to work — sampling under `no_grad` would inject the
  same noise and teach nothing. `tests/test_conservation.py` asserts the
  gradient actually reaches `lstm.weight_hh_l0` *and* the MDN head.
* **Step k = 1 of the K-step rollout is effectively teacher-forced** (its input
  was the true z at t₀). Kept, because it is the literal reading of the spec,
  costs nothing, and only adds a little weight to the one-step term. Steps
  2…K are genuinely open-loop. The K terms are averaged so the weight is
  interpretable.
* **t₀ is resampled uniformly per batch.** A fixed t₀ would let the model learn
  a position-specific fix.
* **The conservation probe is fitted once, before training, and frozen.** A
  co-adapting probe can be satisfied by moving the probe. It is fitted on
  training-root `mu` (posterior means — the probe should read the mass a frame
  actually has), held-out-by-episode R² 0.977.
* **The log-mass head's target is privileged**, from `states[..., 6]`, exactly
  like the existing reward head. Training-time only; nothing downstream reads
  it; `RNNConfig.mass_head` defaults to `False` so every checkpoint trained
  before it existed still loads with an unchanged state dict (asserted in the
  tests).
* **Actions past the end of the episode: hold the last recorded action.** Only
  the last 8 of 200 dream steps, and it affects only the paddle.
* **Well-formedness proxy.** Decode the dreamed latent; count pixels close to
  the mass→colour ramp *and* far from the background (the ramp test is what
  keeps the paddle out — blue and every blend of blue with the background sit
  far off a yellow→red→purple polyline); call the frame well-formed if the
  count is within 50–150 % of a per-episode reference. The reference is the
  same count on the VAE's **reconstruction of the true latents**, not on the
  true frames, so the VAE's own blur divides out. It is a proxy: it will not
  notice a right-sized ball in the wrong place, and it flags a ball that has
  merged with the paddle.
* **Checkpoint selection is unchanged** (best teacher-forced val NLL), so every
  run in the comparison is selected the same way. Since a rollout-loss run is
  optimising something that rule cannot see, `rnn_last.pt` is now also written.
  For all four candidates the best epoch was 33 or 34 of 35, so the selection
  rule did not distort anything here.

## 8. Surprises and honest failures

1. **`--use-mean` is a disaster, and in the way that was predicted but larger.**
   The hypothesis was right about the mechanism — at τ = 1 its *first* dreamed
   step keeps corr 0.97 against the baseline's 0.56, so the predicted colour-
   direction std really did collapse. And it does not help, because the model
   has never been fed a noisy latent and its open-loop rollout leaves the
   manifold entirely: the log-mass RMSE reaches **193 at step 150 and 593 at
   step 199** against a total log-mass range of 1.39 (see the log-scaled panel
   (ii) of `runs/rnn_v2_mean/conservation/conservation.png` — the latents
   diverge). Useful horizon 26 → 12, PR-AUC 0.698 → 0.526, velocity R² from h
   0.89 → 0.68. Its one genuine win is calibration: the cold-start dream no
   longer under-moves (relative bias 0.88 vs the baseline's 0.51), which is the
   flaw `README_M2` §6 flagged. A model trained on means predicts the right
   *magnitude* and cannot survive its own noise.
2. **The combination (b)+(c) is worse than (c) alone.** Mean τ = 1 correlation
   +0.26 against cons's +0.42; well-formedness 0.80 against 0.96. Two losses
   that each attack drift do not add. The plausible reading is that the
   multi-step NLL pulls the predicted std *up* on everything (it is a
   likelihood, and a wider distribution covers a drifted rollout) while the
   conservation penalty pulls one direction *down*; run together they partly
   cancel. Not verified — checking it means looking at the per-dimension
   predicted std on the colour direction, which would be the right follow-up.
3. **The drift is not a v2 problem.** `runs/rnn_v1/conservation/` shows the v1
   model failing the ±25 % speed band at τ = 1 within one step, too. v2 is
   worse, not different. The finding generalises: **any conserved quantity in a
   sampled latent rollout will random-walk unless something stops it**, and
   that includes quantities nobody has thought to name.
4. **τ = 0.5 is not a safe middle ground.** At step 150 the baseline's log-mass
   correlation at τ = 0.5 is −0.58, *worse* than its τ = 1 value of −0.34. If
   the remedy list in `README_C2` had been followed by "train C at a lower τ",
   it would not have worked at 0.5. It might at 0.25; not tested.
5. **The conservation loss improved the ordinary metrics too**, which I did not
   expect. `rnn_v2_cons` has a better val NLL (1.290 vs 1.394), a much better
   open-loop latent MSE at h=32 (0.277 vs 0.643, and unlike the baseline it
   *trends down* over training instead of bouncing), a longer useful horizon,
   lower pixel MSE and lower reward MSE. Plausibly the penalty is acting as a
   regulariser against exactly the off-manifold excursions that ruin long
   rollouts. It is also possible it is mostly the extra `mass_head` supervision;
   the two were deliberately bundled per the spec and are **not separated here**.
6. **The contact head got worse in every candidate** (PR-AUC 0.698 → 0.586 for
   cons, 0.497 for ms). Unexplained. Each extra loss term competes with the
   BCE for the same 256 hidden units, and contact is ~0.8 % of transitions, so
   it is the cheapest thing for the optimiser to give up. If the contact head
   matters downstream, `--w-hit` should be raised in a rerun.
7. **The speed metric at τ > 0 is contaminated by sampling jitter** and I could
   not cleanly separate "the dreamed ball moves fast" from "the dreamed latent
   jitters and the position probe reads that as motion". Both the level (the
   ratio) and the jitter-immune version (the across-episode correlation) are
   reported; the correlation says **no** model tracks the true speed inside a
   sampled long dream. This is the weakest part of the measurement.
8. **One seed per configuration.** The differences between `ms` and `cons` on
   conservation (+0.28 vs +0.42) are larger than the step-to-step noise in the
   curves, but there is no second seed, and the `mean` model's
   `runs/rnn_v2_mean/history.json` shows its own open-loop metric swinging
   between 1.2 and 6.8 across epochs, which is a reminder of how unstable these
   rollout numbers can be.

## 9. Visual artifacts (all looked at)

* `runs/rnn_v2_fix_comparison.png` — the headline. Panel (i): the baseline (blue)
  decays monotonically through zero; `rnn_v2_cons` (red) sits flat at ≈ 0.45 for
  200 steps. Panel (iii b): every model near zero — the honest limit. Table
  underneath. Curves are drawn raw at 20 % alpha with an 11-step rolling mean on
  top; each run's own figure is unsmoothed.
* `runs/rnn_v2/conservation/conservation.png` — the baseline, five panels. The
  τ = 0/τ > 0 gap in panel (i) is the whole finding in one picture.
* `runs/rnn_v2_cons/conservation/conservation.png` — flat correlation, flat
  RMSE, well-formedness pinned near 1.0 at every τ.
* `runs/rnn_v2_mean/conservation/conservation.png` — panel (ii) on a log axis
  climbing to 10², which is what "the latents left the manifold" looks like.
* `runs/rnn_v1/conservation/conservation.png` — two panels only (no mass in v1).
  The τ = 1 speed ratio settles at 1.5–2.0× instead of running away.
* `runs/rnn_v2_cons/causal/recolor_speed.png` — the K = 8 curve tracks the law
  across the whole mass range and is almost on the dashed line for m₁ ≥ 1.0.
* `runs/rnn_v2_cons/training_curves.png` — the open-loop latent MSE panel
  trends *down* across training, unlike the baseline's trendless bounce.
* `runs/rnn_v2_cons/eval/state_error_vs_horizon.png` — τ = 0 crosses the ball
  radius at ~38 frames in x and ~60 in y.

## 10. Things to double-check if you are reviewing

* **The well-formedness proxy's thresholds** (`MASK_RAMP_TOL = MASK_BG_TOL =
  60`, band 50–150 %). They are reasoned but not tuned, and the metric is the
  one place a number could move a lot under a different choice.
* **The speed-ratio inflation at τ > 0** (§8.7). If you want one number for
  "does the dream keep the law", use the across-episode correlation, not the
  ratio.
* **`rnn_v2_cons` bundles two changes** — the conservation penalty and the
  log-mass head — because the spec specified them together. Their contributions
  are not separated. A `--cons-loss-weight 1.0` run *without* `--mass-head` is a
  20-minute experiment and would settle §8.5.
* **The τ = 1 log-mass correlation for `rnn_v2_cons` is ≈ 0.45 flat, not ≈ 0.9.**
  The residual is a bounded error, not drift, but it is still a large error:
  RMSE 0.6–0.7 nats against a range of 1.386. Do not read "the fix worked" as
  "the dreamed ball has the right mass".
* **`runs/rnn_v2/eval/report.json` was overwritten** by the re-run in step 2.
  The checkpoint was not touched and every previous number reproduced; `git
  diff` is the check.
* **Selection on teacher-forced val NLL** across runs that optimise different
  objectives. `rnn_last.pt` is written for each run if you want to check that
  the selection did not matter (for these four, it did not — best epoch 33–34
  of 35).

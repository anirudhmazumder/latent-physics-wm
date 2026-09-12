# v2 stage three, re-run inside the fixed dream — and the test that was missing

Technical log. Hardware: Apple M1, 8 GB. Interpreter
`/opt/miniconda3/envs/NN/bin/python`, every command from the repo root, CPU
only. **No checkpoint and no dataset was modified**: `runs/vae_v2/vae.pt`,
`runs/rnn_v2/rnn.pt`, `runs/rnn_v2_cons/rnn.pt` and `runs/ctrl_v2/` were all
read-only. Read [`README_C2.md`](README_C2.md) for what stage three found the
first time and [`README_FIX.md`](README_FIX.md) for what `rnn_v2_cons` is.

**Two answers, up front.**

1. **Fixing the colour drift changed the controller's skill, by a lot.**
   Retraining the same controller, with the same CMA-ES settings, inside
   `rnn_v2_cons`'s dream instead of `rnn_v2`'s takes interceptions per floor
   visit from **0.79 to 0.94–1.00** against an oracle ceiling of **0.99** and a
   do-nothing floor of 0.45. It is at the ceiling in every mass tercile and on
   the held-out colour band. This was the single open experiment at the end of
   `README_FIX.md` §6 and it came out larger than expected.
2. **The intervention test says yes, the controller's decision does depend on
   the ball's colour** — and it says so with a null control that is *exactly*
   zero, an oracle row that is exactly zero, and a colour-blind-encoder row at
   roughly half the effect with no coherent direction. Repainting the ball
   changes the drive by 0.24–0.34 of its own standard deviation and flips
   7–14 % of actions, in the direction the speed law predicts.

**The honest failures.** (1) Both effects come from **one seed per
configuration**; the 2×2 over (M, τ) is clean but each cell is n = 1. (2) The
gain cannot be attributed to the conservation *penalty* specifically — `cons`
bundles the penalty with a `mass_head`, and `README_FIX.md` §10 already flagged
that. (3) `ctrl_v1_on_v2`'s repaint effect is **not** zero (0.150 against
`ctrl_v2`'s 0.244), so a colour-blind encoder still leaks some colour; the
interpretation is comparative, not absolute. (4) The old "interaction term"
test is confirmed dead: it gives ΔR² ≈ +0.014 for *every* row, and its single
largest value (+0.0192) belongs to the v1 controller that cannot see colour.

---

## 1. Files added or changed

| file | what |
|---|---|
| **`wm/eval_ctrl_intervention.py`** | new. The interventional controller test: `approach_frames`, `drives`, `intervention_stats`, `_boot_slope`, `plot_intervention`, `--replot` |
| **`tests/test_ctrl_intervention.py`** | new, 7 tests. Suite is **65**, all passing (was 58) |
| `wm/eval_controller_v2.py` | `StackCache`, `parse_overrides`, `stack_for_run`; `--run-rnn` / `--run-vae` / `--params` / `--holdout-names`; the decision analysis now covers every linear controller; the (V, M) pairing is written into `summary.json` and into the report |

Artifacts: `runs/ctrl_v2_cons/`, `runs/ctrl_v2_cons_tau0.5/`,
`runs/ctrl_v2_tau0.5/`, `runs/ctrl_eval_v2_fixed/`, `runs/ctrl_intervention/`,
`runs/ctrl_intervention_valmix/`.

### The change that mattered most in the evaluator

**The controller reads `h`, so M is part of the policy.** `ctrl_v2_cons` must be
driven by `rnn_v2_cons` at evaluation time; pairing it with `rnn_v2` produces no
error, no warning and no shape mismatch — just a wrong number. The evaluator now
resolves each run's M and V from the `--rnn` / `--vae` recorded in that run's own
`history.json` (overridable with `--run-rnn NAME=PATH`), prints the mapping
before it starts, writes it into the report, and
`tests/test_ctrl_intervention.py::test_evaluator_maps_each_run_to_its_rnn`
asserts all three resolution routes. The harness already did this for
`ctrl_v1_on_v2`; it did it by a hard-coded special case, which would not have
extended.

## 2. Commands, in order, with wall clock

```bash
# 1. three controllers, CONCURRENTLY, OMP_NUM_THREADS=2.        649-701 s each
C="--vae runs/vae_v2/vae.pt --data data/v2/train data/v2/train_mix \
   --inputs zh --reward dense --popsize 32 --rollouts 16 --dream-steps 150 \
   --sigma0 0.5 --generations 200 --real-eval-every 5 --real-eval-episodes 24 \
   --real-eval-seed-base 7000 --real-eval-metric interceptions \
   --mass-from-color --mass-holdout 0.85 1.2"
python -m wm.train_controller --rnn runs/rnn_v2_cons/rnn.pt --temperature 1.0 \
    --out runs/ctrl_v2_cons          $C                              # 701 s
python -m wm.train_controller --rnn runs/rnn_v2_cons/rnn.pt --temperature 0.5 \
    --out runs/ctrl_v2_cons_tau0.5   $C                              # 649 s
python -m wm.train_controller --rnn runs/rnn_v2/rnn.pt      --temperature 0.5 \
    --out runs/ctrl_v2_tau0.5        $C                              # 700 s

# 2. the by-mass evaluation: 14 rows in-distribution, 11 on the hold-out band.
python -m wm.eval_controller_v2 \
    --runs runs/ctrl_v2 runs/ctrl_v2_cons runs/ctrl_v2_cons_tau0.5 \
           runs/ctrl_v2_tau0.5 \
    --params params_best_real params_last_dream \
    --episodes 150 --holdout-episodes 60 --out runs/ctrl_eval_v2_fixed  # 466 s

# 3. the intervention, on-policy frames and then a shared off-policy set.
python -m wm.eval_ctrl_intervention --out runs/ctrl_intervention        # 142 s
python -m wm.eval_ctrl_intervention --frames val_mix \
    --out runs/ctrl_intervention_valmix                                 #  87 s

# 4. tests.                                                              21 s
python -m pytest tests/ -q          # 65 passed
```

Total wall clock ≈ 25 minutes. Concurrency cost: a controller run alone is 956 s
at these settings (`ctrl_v2`, measured previously); three at once with
`OMP_NUM_THREADS=2` ran in 649–701 s each, i.e. **faster than the solo
baseline** — CMA-ES here is memory-bandwidth-bound on a batch-512 LSTM, and
capping threads per process turned out to help each process individually.

### Sample budgets

| | per controller | this exercise |
|---|---|---|
| dream env-steps (training) | 15,360,000 | 46,080,000 |
| real env-steps (periodic model-selection check, 41 × 24 × 200) | 196,800 | 590,400 |
| real env-steps (final evaluation) | — | 552,000 (14 × 150 × 200 in-dist + 11 × 60 × 200 hold-out) |
| real env-steps (intervention rollouts) | 8,000 | 56,000 |
| re-renders + encodes (intervention) | 13,500 | 162,000 (6 policies x 2 frame sources) |

The ratio the world model exists to buy is unchanged: **15.4 M dream steps for
0.20 M real ones per controller**, ~78:1.

## 3. The by-tercile table, before and after the fix

150 episodes × 200 steps, seeds 5000–5149, identical starts for every row.
Masses log-uniform in [0.5, 2.0] **excluding the held-out band [0.85, 1.2]**;
observed range [0.50, 2.00], tercile cuts at m = 0.70 and m = 1.41. Light =
fast. **Interceptions per floor visit**, 95 % bootstrap CI over episodes.

`runs/ctrl_eval_v2` (before) and `runs/ctrl_eval_v2_fixed` (after) share seeds,
so the rows are paired and `stay`/`random`/`oracle`/`ctrl_v2`/`ctrl_v1_on_v2`
reproduce to the last digit — which is the check that nothing in the evaluator
changed underneath the comparison.

| controller | M it was trained in | τ | overall | light (fast) | medium | heavy (slow) |
|---|---|---|---|---|---|---|
| `stay` | — | — | 0.45 [0.39, 0.51] | 0.42 | 0.47 | 0.50 |
| `random` | — | — | 0.45 [0.40, 0.50] | 0.43 | 0.43 | 0.54 |
| **`oracle`** (ceiling) | — | — | **0.99 [0.98, 1.01]** | 0.99 | 0.99 | 1.00 |
| `ctrl_v2` *(existing)* | `rnn_v2` | 1.0 | 0.79 [0.74, 0.84] | 0.77 | 0.83 | 0.78 |
| `ctrl_v2_lastdream` | `rnn_v2` | 1.0 | 0.77 [0.72, 0.82] | 0.76 | 0.77 | 0.79 |
| `ctrl_v2_tau0.5` | `rnn_v2` | 0.5 | 0.72 [0.67, 0.78] | 0.61 | 0.81 | 0.90 |
| `ctrl_v2_tau0.5_lastdream` | `rnn_v2` | 0.5 | 0.75 [0.70, 0.80] | 0.73 | 0.78 | 0.76 |
| **`ctrl_v2_cons`** | `rnn_v2_cons` | 1.0 | **0.94 [0.89, 0.97]** | 0.90 | 0.97 | 0.98 |
| **`ctrl_v2_cons_lastdream`** | `rnn_v2_cons` | 1.0 | **1.00 [0.97, 1.03]** | 0.99 | 1.04 | 0.98 |
| **`ctrl_v2_cons_tau0.5`** | `rnn_v2_cons` | 0.5 | **1.00 [0.96, 1.07]** | 0.96 | 0.98 | 1.19 |
| **`ctrl_v2_cons_tau0.5_lastdream`** | `rnn_v2_cons` | 0.5 | **1.03 [0.98, 1.09]** | 0.96 | 1.07 | 1.17 |
| `ctrl_v1_on_v2` *(existing)* | `rnn_v1` (v1 stack) | — | 0.79 [0.74, 0.84] | 0.71 | 0.87 | 0.90 |

**The 2×2 is the result.** Holding everything else fixed:

| interceptions / floor visit | τ = 1.0 | τ = 0.5 |
|---|---|---|
| M = `rnn_v2` (drifting colour) | 0.79 | 0.72 |
| M = `rnn_v2_cons` (fixed) | **0.94** | **1.00** |

The dynamics model is the factor. Temperature is not: halving it **hurts**
inside the drifting dream (0.79 → 0.72) and **helps** inside the fixed one
(0.94 → 1.00), which is what an interaction looks like and is why the third run
was worth its 700 seconds. `README_FIX.md` §8.4 had already measured that
τ = 0.5 does not fix conservation in `rnn_v2`; this says it does not fix the
controller either.

**Ratios above 1.00 are not a bug and not a ceiling violation.** A single floor
visit can yield two interceptions when the first contact fails to clear the
floor zone. `ctrl_v2_cons_tau0.5` scores 2.03 interceptions per episode against
2.02 floor visits and the oracle's 1.89 / 1.91 — so it genuinely intercepts
slightly more often than the oracle does, on the same 150 seeds. The heavy
tercile's 1.19 [1.00, 1.51] is the noisiest cell in the table (a slow ball
offers ~1.3 visits per episode, so the denominator is small).

### The design doc's bar

"≥ 85 % of oracle in every tercile" was **missed** by `ctrl_v2` (77 / 84 / 78 %).
It is now **met comfortably**: `ctrl_v2_cons` scores 91 / 98 / 98 % and
`ctrl_v2_cons_tau0.5` 96 / 99 / 119 %.

### `params_best_real` vs `params_last_dream`

| controller | best-real | last-dream | Δ |
|---|---|---|---|
| `ctrl_v2` | 0.789 | 0.770 | −0.019 |
| `ctrl_v2_tau0.5` | 0.722 | 0.749 | +0.027 |
| `ctrl_v2_cons` | 0.935 | 1.000 | **+0.065** |
| `ctrl_v2_cons_tau0.5` | 1.003 | 1.029 | +0.026 |

**In the fixed dream, CMA-ES's own final mean is the better policy.** That is
the cleanest single statement of what the fix bought: the dream stopped being
exploitable, so the periodic real check stopped being needed as a selector — and
in three of four runs it now *costs* a little skill, because it selects on 24
episodes of a noisy metric and the last-dream mean is averaged over 200
generations. The selected `params_best_real` remains the honest default (it is
the only unbiased signal), but the gap has reversed sign relative to v1's
finding, and that is a result in itself.

### Other metrics, overall (150 episodes)

| controller | interceptions/ep | floor visits/ep | gap at floor ↓ | sign agree. | position lead |
|---|---|---|---|---|---|
| `oracle` | 1.89 | 1.91 | 0.069 | 1.000 | 8.8 |
| `ctrl_v2` | 1.45 | 1.83 | 0.140 | 0.721 | 11.7 |
| `ctrl_v2_tau0.5` | 1.42 | 1.97 | 0.174 | 0.650 | 7.7 |
| `ctrl_v2_cons` | 1.82 | 1.95 | 0.103 | 0.651 | 7.8 |
| `ctrl_v2_cons_tau0.5` | **2.03** | 2.02 | **0.086** | 0.686 | 10.2 |
| `ctrl_v1_on_v2` | 1.64 | 2.07 | 0.141 | 0.580 | 9.0 |
| `stay` | 0.95 | 2.11 | 0.278 | — | 31.3 |

**A surprise worth staring at: sign agreement went DOWN while skill went up.**
`ctrl_v2_cons` intercepts 26 % more often than `ctrl_v2` and halves the distance
between paddle and ball at the moment that matters (gap 0.103 vs 0.140), yet
agrees with `sign(x_err)` on *fewer* approach frames (0.651 vs 0.721). The
reading: once the paddle is already under the ball, "move toward the ball" is
the wrong thing to do, and a better controller spends more of the approach
window parked and dithering — which the per-frame sign statistic scores as
disagreement. `README_C2.md` called sign agreement "the most seed-stable
statistic in the whole stage"; stable it may be, but **it is not monotone in
skill**, and the by-mass sign-agreement panel should not be read as a skill
plot. The gap-at-floor column is the one that tracks the headline metric.

The "reacts earlier for light balls" prediction remains **unsupported**: every
controller's position lead is *shortest* on light balls (6–10 frames) and
longest on heavy ones (10–21), and so is the oracle's (8.1 / 6.5 / 14.7). A fast
ball simply does not offer many frames between "committed" and "arrived"; the
statistic is dominated by the physics, not the policy.

## 4. The held-out colour band

60 episodes, seeds 6000–6059, `mass_only` on [0.85, 1.2] — colours no model in
the stack has ever seen move. The comparison is against each row's own **medium**
tercile above.

| controller | interceptions/visit | vs its own medium | gap at floor | position lead |
|---|---|---|---|---|
| `oracle` | 1.00 [1.00, 1.00] | 0.99 | 0.058 | 9.9 |
| `ctrl_v2` | 0.79 [0.72, 0.85] | 0.83 | 0.111 | 13.5 |
| `ctrl_v2_tau0.5` | 0.77 [0.71, 0.84] | 0.81 | 0.156 | 12.1 |
| **`ctrl_v2_cons`** | **0.99 [0.96, 1.02]** | 0.97 | 0.081 | 11.0 |
| **`ctrl_v2_cons_tau0.5`** | **1.00 [0.98, 1.02]** | 0.98 | 0.076 | 13.6 |
| `ctrl_v1_on_v2` | 0.85 [0.78, 0.91] | 0.87 | 0.107 | 9.9 |
| `stay` | 0.55 [0.46, 0.64] | 0.47 | 0.222 | 33.6 |

Every row plays the never-seen colours as well as it plays a trained colour of
similar speed, CIs overlapping. **Interpolation in colour space is not where any
of these controllers fail** — which was already true at 0.79 and is now true at
1.00.

## 5. Transfer (`dream_vs_real.png`, all four looked at)

| run | Pearson r (smoothed / raw) | best real during training | final dream fitness |
|---|---|---|---|
| `ctrl_v2` | 0.74 / 0.70 | 1.833 | 136.91 |
| `ctrl_v2_cons` | 0.58 / 0.54 | 2.083 | 138.54 |
| `ctrl_v2_cons_tau0.5` | 0.68 / 0.64 | 2.125 | 139.23 |
| `ctrl_v2_tau0.5` | 0.77 / 0.70 | 1.833 | 137.68 |

All four curves rise together and none shows the "dream keeps climbing while
real flattens" signature of an exploited model. **The correlation is *lowest*
for the run with the best real score**, which is a ceiling effect and not a
warning: `ctrl_v2_cons`'s real curve saturates near 2.0 interceptions/episode by
generation ~15 and then only jitters, so there is almost no variance left for
the dream curve to correlate with. Reading r as "transfer quality" across rows
with different real ceilings is a mistake; the level of the orange curve is the
thing.

## 6. The intervention: does the controller's decision use the colour?

`runs/ctrl_intervention/` (on-policy frames) and
`runs/ctrl_intervention_valmix/` (one shared frame set for every row).

300 approach frames per policy, drawn round-robin across episodes so no single
mass dominates; the ball descending in the lower half with |x_err| > 0.02. For
each frame the inputs are rebuilt twice — as recorded, and with all 9 frames of
the window re-rendered in the colour of a different mass. Positions, paddle and
actions are bit-identical between the arms.

### Headline (on-policy frames)

| controller | \|Δdrive\|/sd | flips | **null** \|Δ\|/sd | **null** flips | drive sd |
|---|---|---|---|---|---|
| `ctrl_v2` | 0.244 | 0.071 | **0.000** | **0.000** | 39.3 |
| `ctrl_v2_cons` | 0.292 | 0.129 | **0.000** | **0.000** | 25.2 |
| **`ctrl_v2_cons_tau0.5`** | **0.342** | **0.144** | **0.000** | **0.000** | 24.1 |
| `ctrl_v2_tau0.5` | 0.324 | 0.120 | **0.000** | **0.000** | 31.0 |
| `ctrl_v2_z_only` | 0.263 | 0.083 | **0.000** | **0.000** | 7.4 |
| `ctrl_v1_on_v2` (colour-blind V) | 0.150 | 0.074 | **0.000** | **0.000** | 33.5 |
| `oracle` (true state) | **0.000** | **0.000** | 0.000 | 0.000 | 0.70 |

### Direction — the coefficient that carries the claim

β is the slope of (Δdrive / sd) on (log m₁ − log m₀)·sign(x_err), CI from a
bootstrap **over episodes** (frames inside an episode are not independent).
**The speed law predicts β < 0**: repainting the ball lighter, hence faster,
should push the paddle harder toward the side the ball is on.

| controller | m₁ = 0.5 | m₁ = 1.0 *(held out)* | m₁ = 2.0 |
|---|---|---|---|
| `ctrl_v2` | −0.078 [−0.20, +0.03] | **−0.201 [−0.31, −0.10]** | **−0.268 [−0.37, −0.18]** |
| `ctrl_v2_cons` | −0.091 [−0.22, +0.05] | −0.086 [−0.19, +0.02] | −0.127 [−0.25, +0.00] |
| **`ctrl_v2_cons_tau0.5`** | **−0.173 [−0.29, −0.07]** | **−0.360 [−0.48, −0.23]** | **−0.482 [−0.59, −0.37]** |
| `ctrl_v2_tau0.5` | −0.054 [−0.20, +0.10] | **−0.186 [−0.33, −0.04]** | **−0.244 [−0.36, −0.11]** |
| `ctrl_v2_z_only` | **−0.152 [−0.28, −0.04]** | **−0.209 [−0.31, −0.11]** | **−0.132 [−0.21, −0.06]** |
| `ctrl_v1_on_v2` | **+0.042 [+0.02, +0.07]** | **−0.124 [−0.20, −0.04]** | −0.004 [−0.04, +0.03] |
| `oracle` | 0.000 [0, 0] | 0.000 [0, 0] | 0.000 [0, 0] |

**How to read this.**

* **The null control is exactly 0.000 for every row and every statistic**, not
  approximately. The renderer is a pure function of (ball position, paddle
  position, colour), so re-rendering with m₁ = m₀ reproduces the recorded frames
  bit-for-bit — asserted in
  `tests/test_ctrl_intervention.py::test_repaint_with_the_same_mass_is_bit_identical`.
  The null therefore measures the *plumbing* (that the as-is and repaint arms
  are the same code path reached two ways), not stochastic round-trip noise;
  there is none to measure. Nothing in the table can be blamed on a re-rendering
  artefact.
* **The oracle is exactly 0**, as it must be: it reads true state, which the
  intervention does not touch.
* **Every v2 controller shows a real effect with a consistent negative
  direction.** 10 of 15 v2 CIs exclude zero, and all 15 point negative. The
  effect is also **smallest at m₁ = 1.0 for every v2 row** — the repaint that
  moves the colour least, since m₀ is log-uniform on [0.5, 2.0] minus that very
  band — which is what a *graded* read of the colour ramp looks like and not
  what a threshold or an artefact would look like. For `ctrl_v1_on_v2` the
  m₁ = 1.0 column is instead the **largest** of the three (0.193 against 0.131
  and 0.126): whatever the v1 stack responds to, it is not the size of the
  colour change.
* **`ctrl_v1_on_v2` is the leak measurement, and it behaves differently in
  kind.** Its magnitude is real but smallest (0.150), and its direction is
  **incoherent**: significantly *positive* at m₁ = 0.5, significantly negative
  at m₁ = 1.0, and zero at m₁ = 2.0. A policy using the speed law cannot produce
  that pattern. A colour-blind encoder whose latents shift a little when the
  ball's luminance changes can, and does. **This is the row that makes the
  others mean something** — it shows the test is not simply reporting "pixels
  changed, therefore z changed, therefore the drive changed".
* **`ctrl_v2_cons_tau0.5` — the best controller in the skill table — has the
  largest effect and the steepest, tightest β at every mass.** The two results
  line up: the controller that plays best is the one whose decision depends most
  on the colour, in the direction the law prescribes.
* **`ctrl_v2_cons` is the awkward row.** It is the second-best policy but has
  the *flattest* β of the four v2 runs, and none of its three CIs quite excludes
  zero. Its effect size and flip fraction are healthy (0.292 / 0.129), so it
  reacts to colour; it just does not convert that into a consistently signed
  push. It may be reading colour and using it for something other than lateral
  gain (timing, say), or it may be a one-seed accident. Not resolved.
* **`ctrl_v2_z_only` shows the effect too**, which is expected rather than
  surprising: colour is a *single-frame* property and `z` is a single frame, so
  z-only is precisely the input mode that *should* be able to read mass. It is a
  worse controller (0.57 interceptions/visit in `README_C2.md`) because it has
  no velocity, not because it is colour-blind.
* **The m₁ = 1.0 column is inside the held-out band** [0.85, 1.2]. Every v2 row
  responds to it, and four of the five have a *steeper* β there than at
  m₁ = 0.5 even though it is the smaller colour change — so the controllers'
  colour reading interpolates into a band the VAE and the RNN never saw.

### The shared-frame control

`--frames val_mix` reruns everything on one fixed, off-policy frame set. The
levels drop (fewer episodes contribute: 19 vs 38, and the frames are not the
ones each policy actually visits) but the **ordering and the signs are
preserved**: all five v2 rows negative at every mass, `ctrl_v1_on_v2` at
+0.033 / +0.008 for m₁ = 1 and 2 with only its m₁ = 0.5 cell slightly negative
and CI straddling zero, oracle exactly 0. The conclusion does not depend on
whose states are used.

### Reconstruction fidelity

The intervention rebuilds `h_pre` from an 8-frame warm start, while the real
rollout carries `h` from a cold start through the whole episode. Pearson r
between the reconstructed drive and the drive the rollout actually used:
`ctrl_v2` 0.987, `ctrl_v2_cons` 0.984, `ctrl_v2_cons_tau0.5` 0.982,
`ctrl_v2_tau0.5` 0.976, `ctrl_v2_z_only` 1.000 (it has no h term, so the
reconstruction is exact), `ctrl_v1_on_v2` 0.961. Not load-bearing — both arms
share the reconstruction, so the *contrast* is exact regardless — but it says
the probe is measuring the policy as played and not a caricature of it.

## 7. The decision-analysis table, kept, and why it should be retired

`runs/ctrl_eval_v2_fixed/summary.md` still carries the observational regression.
Now that it is run for every row instead of a hand-picked three, it is obvious
that it separates nothing:

| controller | R² base | R² + interactions | ΔR² |
|---|---|---|---|
| `ctrl_v2` | 0.4635 | 0.4774 | +0.0139 |
| `ctrl_v2_cons` | 0.3975 | 0.4141 | +0.0166 |
| `ctrl_v2_cons_tau0.5` | 0.5583 | 0.5738 | +0.0155 |
| `ctrl_v2_tau0.5` | 0.5587 | 0.5662 | +0.0074 |
| **`ctrl_v1_on_v2`** (cannot see colour) | 0.3107 | 0.3299 | **+0.0192** |

Every value lands in +0.007 … +0.020, and the **largest belongs to the
controller whose encoder is colour-blind**. Two free parameters added to a
regression on 7,000 correlated frames buy about that much R² regardless of what
they mean. The observational test is confounded — `speed` predicts which
*situations* a controller is in, so a purely positional policy shows an
interaction — and the intervention is the replacement, not a supplement.

## 8. Design decisions, all arguable, all recorded

* **Nine frames per probe, not eight.** The controller acts on
  `[z_t, h_pre_t]`, and `h_pre_t` is M's output after consuming `(z_{t-1},
  a_{t-1})`, so the window is `t-8 … t-1` for the warm-up plus frame `t` for
  `z_t`. This is `DreamEnv.reset`'s convention exactly; using eight total would
  have shifted the probe one frame relative to the policy.
* **The whole window is repainted, not just the last frame.** Repainting only
  frame `t` would leave M's hidden state carrying the old colour, and the
  measured Δ would be the response of `z` alone — a strictly weaker question
  than "does the decision depend on the colour".
* **Frames are drawn round-robin across episodes.** A light ball crosses the box
  four times as often as a heavy one, so a uniform draw over approach frames
  would be a draw over *light* balls, and every "mean effect" would be a
  statement about one tercile.
* **Δ is normalised by the controller's own drive std before any regression.**
  A linear policy's logits have no scale: multiplying W by 10 multiplies every
  drive and every Δ by 10 and changes no action. Raw slopes are not comparable
  across rows; `drive_std` is reported so the raw numbers can be recovered.
* **The bootstrap resamples episodes, not frames.** 300 frames from 38 episodes
  carry nothing like 300 independent observations. A frame bootstrap would have
  produced CIs roughly 3× too narrow and made `ctrl_v1_on_v2`'s incoherent
  pattern look like three significant findings.
* **`sign(x_err)` rather than `x_err` in the regressor**, and frames with
  |x_err| < 0.02 dropped. The hypothesis is about which *side* the push goes,
  and multiplying by a magnitude would let a handful of far-from-the-ball frames
  dominate the fit.
* **`m1 = 1.0` is deliberately in the held-out band.** It costs nothing to
  include, it is flagged in every output, and it converts the intervention into
  a generalisation test for free.
* **The oracle row is included even though its answer is known.** A test whose
  negative control is not actually run is not a test.
* **The null control is per row, not once.** A different encoder could in
  principle have a different round-trip error; asserting it globally would have
  hidden that.
* **`--replot`.** Changing how a result is drawn should never require
  re-measuring it. Both figures were redrawn from `summary.json` after the
  direction panel was added.

## 9. Surprises and honest failures

1. **The effect of the M fix on the controller is much larger than the effect
   of the M fix on M.** `rnn_v2_cons` improved the dreamed-log-mass correlation
   from −0.21 to +0.42 and the useful horizon from 26 to 29 frames — real but
   modest. Downstream it moved the controller from 0.79 to 1.00 interceptions
   per floor visit, i.e. from 79 % of oracle to 101 %. A partial fix to a
   conserved quantity in the dream bought a near-total fix in the policy. That
   is the strongest evidence in this project that **long-horizon conservation,
   not one-step accuracy, is what a dream-trained controller actually needs**;
   `rnn_v2_cons` is also the model with the *worse* contact PR-AUC (0.586 vs
   0.698), and it did not matter.
2. **CMA-ES's last-dream mean is now better than the real-selected
   parameters** in three of four runs, by up to +0.065. In v1 and in `ctrl_v2`
   the gap ran the other way. The periodic real check was invented to catch a
   dream being exploited; inside a conserving dream there is less to catch, and
   the check's own 24-episode noise starts to cost more than it saves.
3. **Sign agreement fell as skill rose** (§3). A statistic that `README_C2.md`
   promoted for its stability turns out to be non-monotone in the thing it was
   standing in for. I did not expect to have to retire a metric in the same
   document that vindicates a fix.
4. **Lowering τ interacts with M rather than helping on its own** — worse in
   `rnn_v2` (0.79 → 0.72), better in `rnn_v2_cons` (0.94 → 1.00). The natural
   story ("a cooler dream drifts less, so it should always help") is wrong; in
   the drifting model a cooler dream is a *more confidently wrong* dream, and
   CMA-ES exploits it harder.
5. **`ctrl_v2_cons`'s direction coefficient is the weakest of the four v2 runs**
   despite being the second-best policy (§6). Unexplained, and the one place the
   "skill tracks colour use" story does not hold cleanly.
6. **The v1 controller's colour leak is not negligible** (0.150 against
   `ctrl_v2`'s 0.244). A colour-blind VAE is not a colour-invariant one. The
   argument that separates it from the v2 rows rests on the *direction* being
   incoherent, not on the magnitude being small — which is a subtler claim than
   I expected to have to make, and it is the part of §6 a reviewer should push
   on hardest.
7. **One seed per cell.** Four controllers, four seeds of 0. The 2×2 over (M, τ)
   is internally consistent and the gaps (0.79/0.72 vs 0.94/1.00) are far larger
   than the bootstrap CIs, but bootstrap CIs are over *episodes*, not over
   *training runs*, and the latter is the variance that matters for the claim.
8. **The gain is attributed to "training inside `rnn_v2_cons`", not to the
   conservation penalty.** `rnn_v2_cons` bundles the penalty with a `mass_head`
   and, being a separate training run, differs from `rnn_v2` in every weight. A
   controller trained inside `rnn_v2_ms` (the rollout-NLL variant, which fixes
   conservation less well but keeps a better cold-start r) would separate "a
   conserving dream" from "a different dream", and was not run.
9. **Interceptions per floor visit is saturating.** Three of the six new rows
   sit at or above the oracle line, and the heavy tercile's CI reaches 1.51. The
   metric was designed to make terciles comparable, not to resolve differences
   near 1.0. If the next controller is better again, this metric will not show
   it; gap-at-floor still has headroom (0.080 vs the oracle's 0.069).

## 10. Visual artifacts (all looked at)

* `runs/ctrl_eval_v2_fixed/interceptions_per_visit_by_mass.png` — the headline.
  Twelve bars per cluster is at the edge of legible; the four `cons` bars clear
  the green oracle segment in the medium and heavy clusters, and only `light`
  keeps any of them visibly below it.
* `runs/ctrl_eval_v2_fixed/sign_agreement_by_mass.png` — every trained row
  between 0.58 and 0.76, essentially flat across mass, and **not** ordered like
  the skill plot. Read §3 before reading this figure.
* `runs/ctrl_intervention/intervention_effect.png` — three panels. Left and
  middle: the null bars have literally zero height, so the panel is annotated
  "null = 0.000 for every row" rather than leaving the reader hunting. Right:
  the direction panel, where `ctrl_v2_cons_tau0.5`'s three bars run
  −0.17 / −0.36 / −0.48 with CIs clear of zero and `ctrl_v1_on_v2`'s straddle
  it in both directions.
* `runs/ctrl_intervention_valmix/intervention_effect.png` — the same picture at
  smaller amplitude and wider CIs. Same ordering.
* `runs/ctrl_v2_cons/dream_vs_real.png`, `runs/ctrl_v2_cons_tau0.5/…`,
  `runs/ctrl_v2_tau0.5/…` — all three rise together; the two `cons` runs
  saturate their real curve near 2.0 interceptions/episode by generation ~15.
* `runs/ctrl_eval_v2_fixed/real_play_ctrl_v2_cons_tau0.5_light.gif`
  (seed 5057, m = 0.50, 3 interceptions) and `…_heavy.gif` (seed 5146, m = 1.99,
  1 interception) — rendered for this log with the winning controller on the
  same episodes the `ctrl_v2` GIFs use. The paddle is under the ball in every
  sampled frame of both.
* `runs/ctrl_eval_v2_fixed/real_play_ctrl_v2_{light,heavy}.gif` and
  `dream_play_ctrl_v2.gif` — regenerated identically to `runs/ctrl_eval_v2`.

## 11. Things to double-check if you are reviewing

* **The (V, M) pairing**, printed at the top of `runs/ctrl_eval_v2_fixed.log`
  and stored under `"stacks"` in `summary.json`. Every conclusion here dies if
  `ctrl_v2_cons` was scored with `rnn_v2`. It was not, but the failure mode is
  silent.
* **`ctrl_v1_on_v2`'s repaint effect is 0.150, not 0**, and the argument that it
  is qualitatively different rests on the *sign incoherence* across m₁, not on
  the magnitude (§9.6).
* **Interceptions per floor visit above 1.0** (§3). Check
  `interceptions_per_episode` against `floor_visits_per_episode` if it looks
  wrong; the numbers are 2.03 vs 2.02 for the best row.
* **One seed per configuration** (§9.7). The cheapest strengthening of this
  document is `--seed 1` on `ctrl_v2` and `ctrl_v2_cons`, ~23 minutes for both
  concurrently.
* **`ctrl_v2_cons` vs `ctrl_v2_cons_tau0.5` on the direction coefficient**
  (§9.5) — the one internal inconsistency.
* **The intervention repaints the whole 9-frame window**, which makes the
  warm-up internally consistent (colour and motion agree about nothing, since
  the motion is the *old* mass's). The same ambiguity `eval_causal_v2` §(b)
  documents applies: a controller could in principle be responding to the
  colour/motion *mismatch* rather than to the colour. The K = 1 trick that
  resolves it there is not available here, because the controller needs h.

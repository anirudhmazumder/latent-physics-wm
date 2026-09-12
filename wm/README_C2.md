# v2 stage three ("C") — the controller, by mass

Technical log. Hardware: Apple M1, 8 GB. Interpreter
`/opt/miniconda3/envs/NN/bin/python`, all commands from the repo root, all on
CPU. **The v2 VAE (`runs/vae_v2/vae.pt`) and the v2 MDN-RNN
(`runs/rnn_v2/rnn.pt`) were frozen; no dataset and no v1 artefact was
modified.** Read [`docs/v2/00_v2_design.md`](../docs/v2/00_v2_design.md) §3 "C"
first, then [`README_C.md`](README_C.md) for what v1 established and
[`README_M2.md`](README_M2.md) §13 for what stage two handed over.

**The question.** v2's ball wears its mass as a colour, and mass sets its speed
(`0.022/m`, so a light ball is up to 4× faster than a heavy one). A controller
that has learned that law should handle both regimes — and, per the design doc,
should **react earlier for light balls**.

**The answer, up front.** The dream-trained controller works: **0.79
interceptions per floor visit overall against an oracle ceiling of 0.99 and a
do-nothing floor of 0.45**, and — the part that matters — it is **flat across
mass**: 0.76 / 0.83 / 0.78 on light / medium / heavy, with overlapping CIs. It
plays never-seen hold-out colours at **0.79**, identical to its in-distribution
medium tercile (0.83, CIs overlapping). The design doc's "≥ 85 % of oracle in
every tercile" bar is **missed** (77 % / 84 % / 78 %), but only just, and by the
same margin everywhere rather than by failing on fast balls.

**The two honest failures.** (1) The **v1 controller, dropped into the v2 world
with the v1 V and M, ties it** — 0.79 overall, and it actually *beats* ctrl_v2 on
medium and heavy while losing on light (0.71 vs 0.76). So the transfer cost of
not being able to read colour is real but small, and it sits exactly where
theory says it should (fast balls); it is not the large effect the design doc
anticipated. (2) The **"reacts earlier for light balls" prediction is not
supported**: ctrl_v2 gets into position 6.8 frames ahead of contact on light
balls and 21.1 on heavy — later for fast balls, like everything else, and
slightly later than the oracle (8.1 light). The interaction term *is* present
(β = +0.34 on `speed·x_err`, the largest single coefficient after `ball_vx`) but
the same test fires on ctrl_v1_on_v2, so **the interaction test alone does not
establish mass-awareness** — see §8.

---

## 1. Files added or changed

| file | what |
|---|---|
| `wm/eval_controller.py` | `make_box_cfg` / `add_env_args` / `env_kwargs` — one definition of the v2 environment flags, shared by both entry points; `episode_masses`; `run_real_episodes` records `mass` and `speed`; `floor_visit_stats(speed=…)`; `gap_episode` so gaps can be grouped; `run_population_real(count=…)`; `interceptions_per_floor_visit` in `summarise`; `dream_play_gif(starts=…)`; v2 flags on the CLI |
| `wm/train_controller.py` | v2 env flags passed through to every real-env call; `--real-eval-metric`; `--real-fitness-count`; `RealFitness(env=…, count=…)`; honest axis labels on `dream_vs_real.png` |
| `wm/dream_env.py` | `StartPool.mass`, `StartPool.sample_mass_range` (pick a light ball to dream about); state width no longer assumed to be 6 |
| **`wm/eval_controller_v2.py`** | new — the by-mass analysis: terciles, ratio bootstrap, per-episode sign agreement, reaction/position lead, the hold-out run, the interaction regression, the two PNGs, the GIFs, the report |
| `tests/test_controller_v2.py` | new — 9 tests (suite is now **49**, all passing) |

Artifacts: `runs/ctrl_v2/`, `runs/ctrl_v2_mix/`, `runs/ctrl_v2_z_only/`,
`runs/ctrl_v2_real/`, `runs/ctrl_eval_v2/`.

Nothing in `worldsim/`, `wm/vae.py`, `wm/rnn.py`, `wm/seq_data.py` was touched.
No checkpoint or dataset was modified, and nothing was committed.

### Every new argument defaults to v1

`make_box_cfg()` with no v2 argument returns the v1 `BoxConfig`;
`floor_visit_stats(speed=None)` uses the v1 constant;
`run_population_real(count="frames")` counts contact frames as v1 did;
`--real-eval-metric hits` keeps v1's model-selection criterion. Verified two
ways: `tests/test_controller_v2.py::test_v1_code_path_is_unchanged` pins the v1
controller's hits in the v1 world against a value recorded from the pre-change
code (obtained by `git stash`, re-running, and popping), and the v1 CLI still
runs unchanged:

```bash
python -m wm.eval_controller --ctrl runs/ctrl_v1/controller.pt --episodes 5 \
    --no-gifs --out <scratch>                                        # 10 s, OK
python -m pytest tests/ -q                                           # 49 passed
```

### The floor-visit threshold (asked for explicitly)

v1's band was `paddle_h + ball_radius + ball_speed = 0.045 + 0.08 + 0.022 =
0.147` — contact height plus **one frame of travel**, the smallest margin that
makes the band policy-independent. In v2 one frame of travel is `0.022/m`, i.e.
0.044 for the lightest ball and 0.011 for the heaviest. So
`floor_visit_stats` now takes a **per-episode `speed`** (the harness supplies
`0.022 / mass` from the recorded 7th state column) and the band becomes
`0.125 + speed`, evaluated per episode and broadcast over time. `speed=None`
falls back to 0.022 and reproduces the v1 numbers exactly
(`test_floor_visit_stats_speed_none_is_the_v1_number`). The bias this removes is
the worst kind available here: a fixed 0.147 band under-counts *chances* for
fast balls, which would inflate interceptions-per-visit on precisely the light
tercile the whole experiment is about.

## 2. Commands, in order, with wall clock

```bash
# 1. the three dream-trained controllers + the real-env baseline, all four
#    concurrently with OMP_NUM_THREADS=2 on a 4-core M1.
COMMON="--rnn runs/rnn_v2/rnn.pt --vae runs/vae_v2/vae.pt \
  --data data/v2/train data/v2/train_mix \
  --mass-from-color --mass-holdout 0.85 1.2 --real-eval-metric interceptions \
  --real-eval-every 5 --real-eval-episodes 24 --real-eval-seed-base 7000"
DREAM="--temperature 1.0 --popsize 32 --rollouts 16 --dream-steps 150 \
  --sigma0 0.5 --generations 200"

python -m wm.train_controller --out runs/ctrl_v2        --inputs zh --reward dense $COMMON $DREAM   #  956 s
python -m wm.train_controller --out runs/ctrl_v2_mix    --inputs zh --reward mix   $COMMON $DREAM   # 1050 s
python -m wm.train_controller --out runs/ctrl_v2_z_only --inputs z  --reward dense $COMMON $DREAM   # 1024 s
python -m wm.train_controller --out runs/ctrl_v2_real   --inputs zh --fitness real \
    --real-fitness-count interceptions --popsize 16 --rollouts 8 --real-steps 200 \
    --generations 40 $COMMON                                                                        # 1928 s

# 2. the evaluation: 150 in-distribution episodes (seeds 5000+) and 60 hold-out
#    episodes (seeds 6000+), by mass tercile, plus plots, GIFs and diagnostics.
python -m wm.eval_controller_v2 --out runs/ctrl_eval_v2                                             #  430 s
```

`ctrl_v1_on_v2` needs **no training**: it is `runs/ctrl_v1/controller.pt` driven
by the v1 VAE (`runs/vae_b1/vae.pt`) and the v1 RNN (`runs/rnn_v1/rnn.pt`),
loaded by `eval_controller_v2` as its own `Spec(name, ctrl, vae, rnn)` triple so
each policy reads the world through the stack it was trained on.

Per-generation cost, with all four jobs sharing the machine: a dream generation
(32 × 16 × 150 = 76,800 batched RNN steps) is **~5 s**; a real-fitness
generation (16 × 8 × 200 = 25,600 rendered-and-encoded real frames) is
**~48 s**. Same ~10× gap as v1.

## 3. What was configured, and why (decisions made without asking)

* **`--reward dense` is the headline run.** v1's lesson: at λ = 0.1 the `mix`
  reward is numerically dominated by the dense term anyway, and the small hit
  component is the exploitable one (the hit head is trained with
  `pos_weight ≈ 115`). `ctrl_v2_mix` is kept for continuity and duly reproduces
  the v1 pathology — see §5.
* **The periodic real check counts interceptions, not contact frames**
  (`--real-eval-metric interceptions`), because that is the number v1 concluded
  was honest and it is what selects `params_best_real`. The flag defaults to
  `hits` so the v1 runs stay exactly reproducible.
* **`--fitness real` maximises interceptions too** (`--real-fitness-count
  interceptions`), closing v1's loophole: v1's `ctrl_real` learned to pin the
  ball against the paddle, scoring 2.00 hits/ep on 1.27 real interceptions.
* **In-distribution evaluation uses `mass_holdout=(0.85, 1.2)`**, so the
  reported masses are drawn from the same distribution the models were trained
  on; the hold-out run uses `mass_only=(0.85, 1.2)`.
* **Terciles are data-defined** (33rd/67th centile of the observed episode
  masses: cuts at m = 0.70 and 1.41 over an observed range [0.50, 1.99]), so the
  groups are equal-sized. Every policy sees the same 150 seeds and therefore the
  same masses, so the table is paired across rows.
* **Ratio CIs bootstrap the ratio of sums** over resampled *episodes*, rather
  than averaging per-episode ratios, so episodes that offered no chance neither
  have to be dropped nor counted as zero.
* **Baseline sign agreement is computed from the action**, not from logits
  (`stay`/`random`/`oracle` have no logits), with STAY frames dropped from the
  denominator. This makes `oracle` score 1.000 *by construction* — it is defined
  as "move toward the ball" — so that row is a tautology and not evidence. The
  load-bearing comparisons in that table are controller vs controller.

## 4. Results — 150 episodes × 200 steps, seeds 5000–5149, identical starts

Masses log-uniform in [0.5, 2.0] **excluding** [0.85, 1.2]. Full tables:
`runs/ctrl_eval_v2/summary.md`; raw numbers `summary.json`.

### Interceptions per floor visit — the mass-fair metric

Light balls reach the floor **~3× more often** than heavy ones (3.2 vs 1.0
visits/episode at the `stay` baseline), so per-episode counts mostly measure
physics. This is the table to read.

| controller | overall | light (fast) | medium | heavy (slow) |
|---|---|---|---|---|
| `stay` | 0.45 [0.39, 0.51] | 0.42 [0.33, 0.51] | 0.47 [0.37, 0.56] | 0.50 [0.37, 0.63] |
| `random` (sticky) | 0.45 [0.40, 0.50] | 0.43 [0.36, 0.50] | 0.43 [0.34, 0.51] | 0.54 [0.40, 0.67] |
| `oracle` (true state) | **0.99** [0.98, 1.01] | 0.99 [0.96, 1.02] | 0.99 [0.96, 1.00] | 1.00 [1.00, 1.00] |
| **`ctrl_v2`** (zh, dense) | **0.79** [0.74, 0.84] | 0.76 [0.68, 0.85] | 0.83 [0.76, 0.91] | 0.78 [0.66, 0.88] |
| `ctrl_v2_lastdream` | 0.77 [0.72, 0.82] | 0.76 [0.68, 0.84] | 0.77 [0.67, 0.86] | 0.79 [0.69, 0.89] |
| `ctrl_v2_mix` (zh, mix) | 0.62 [0.55, 0.68] | 0.51 [0.42, 0.60] | 0.66 [0.55, 0.78] | 0.84 [0.73, 0.94] |
| `ctrl_v2_z_only` (z) | 0.57 [0.51, 0.64] | 0.57 [0.47, 0.66] | 0.62 [0.50, 0.74] | 0.51 [0.35, 0.67] |
| `ctrl_v2_real` (no WM) | 0.64 [0.58, 0.69] | 0.66 [0.59, 0.74] | 0.63 [0.55, 0.72] | 0.57 [0.43, 0.71] |
| `ctrl_v1_on_v2` (transfer) | **0.79** [0.74, 0.84] | 0.71 [0.64, 0.78] | 0.87 [0.77, 0.95] | 0.90 [0.79, 1.02] |

Plot: `runs/ctrl_eval_v2/interceptions_per_visit_by_mass.png` (grouped bars,
oracle drawn as a per-tercile reference segment rather than a competing bar).

As a fraction of the oracle: `ctrl_v2` is **77 % / 84 % / 78 %** across light /
medium / heavy. The design doc's success bar was ≥ 85 % in every tercile; that
is missed, but uniformly, not by collapsing on fast balls.

### The supporting numbers

| controller | floor visits/ep | interceptions/ep | hits (frames)/ep | gap at floor | approach sign agreement |
|---|---|---|---|---|---|
| `stay` | 2.11 | 0.95 | 0.95 | 0.278 | — |
| `random` | 2.08 | 0.93 | 1.03 | 0.306 | 0.338 |
| `oracle` | 1.91 | 1.89 | 1.95 | **0.069** | 1.000 (tautological) |
| `ctrl_v2` | 1.83 | 1.45 | 1.48 | **0.140** | **0.721** |
| `ctrl_v2_lastdream` | 1.85 | 1.43 | 1.51 | 0.149 | 0.701 |
| `ctrl_v2_mix` | 2.01 | 1.24 | **1.97** | 0.205 | 0.575 |
| `ctrl_v2_z_only` | 1.93 | 1.11 | 1.13 | 0.219 | 0.649 |
| `ctrl_v2_real` | 2.05 | 1.31 | 1.33 | 0.187 | 0.587 |
| `ctrl_v1_on_v2` | 2.07 | 1.64 | 1.84 | 0.141 | 0.580 |

Sign agreement by tercile (chance 0.50; plot
`runs/ctrl_eval_v2/sign_agreement_by_mass.png`):

| controller | light | medium | heavy |
|---|---|---|---|
| `ctrl_v2` | **0.757** | 0.709 | 0.694 |
| `ctrl_v2_lastdream` | 0.740 | 0.676 | 0.684 |
| `ctrl_v2_mix` | 0.604 | 0.591 | 0.533 |
| `ctrl_v2_z_only` | 0.653 | 0.653 | 0.642 |
| `ctrl_v2_real` | 0.664 | 0.614 | **0.491** |
| `ctrl_v1_on_v2` | 0.658 | 0.579 | **0.504** |

### How to read it

* **`ctrl_v2` is the best controller on the honest metric and the only one above
  0.69 sign agreement in every tercile.** It beats `stay`/`random` with
  non-overlapping CIs everywhere. Its mean paddle–ball gap at closest approach
  is 0.140 against the oracle's 0.069 and `stay`'s 0.278, so it is tracking
  rather than getting lucky.
* **`ctrl_v2_lastdream` ≈ `ctrl_v2`** (0.77 vs 0.79, CIs overlapping heavily).
  The periodic real check therefore bought essentially nothing, and the
  zero-real-sample claim can be made for the full 0.77.
* **`ctrl_v2_z_only` is the predicted failure, but a milder one than v1's.** v1's
  z-only controller sat at 0.527 approach-sign agreement — chance. v2's sits at
  **0.649**, clearly above chance, while still being the *worst* trained
  controller on interceptions-per-visit (0.57). That is exactly the twist the
  design anticipated: in v2, `z` carries the colour and therefore *how fast* the
  ball is, but still nothing about *which way* it is going. Knowing the speed
  without the direction is worth something (it beats v1's z-only relative to its
  own baselines: 0.57 vs a 0.45 stay floor and a 0.99 ceiling, i.e. 58 % of
  oracle, against v1's z-only at (1.03/1.70)/(1.63/1.64) = 61 % of oracle on interceptions per visit —
  — 58 % vs 61 %, i.e. a wash — and the honest statement is that the extra
  colour information did not rescue it) but it is not enough to act on.
* **`ctrl_v2_real` (1,024,000 real environment steps) loses to `ctrl_v2`
  (0 real training steps)**: 0.64 vs 0.79. Constraining its fitness to
  interceptions did close v1's pinning loophole — its hits/ep (1.33) and
  interceptions/ep (1.31) now agree — but with popsize 16 / 8 rollouts and
  40 generations it is a much weaker optimiser, so this is not a like-for-like
  comparison and should not be read as "the world model wins" on quality alone.
  It is a like-for-like comparison on *cost*.
* **`ctrl_v2_mix` reproduces v1's `mix` pathology and then some.** 1.97 contact
  frames per episode, the highest in the table — and only 1.24 interceptions,
  the second-lowest. Its heavy tercile shows 2.30 hits/ep against 0.92
  interceptions/ep: it is pinning slow balls against the paddle. Its `dense`
  sibling `ctrl_v2` shows 1.48 hits on 1.45 interceptions, i.e. essentially one
  contact frame per interception. `--reward dense` was the right default.

## 5. `ctrl_v1_on_v2` — the transfer baseline

The v1 controller has never seen a coloured ball, a fast ball, or a ball whose
speed differs from 0.022. Its encoder was trained on red balls only and its
dynamics model on a single speed. Dropped into v2 it scores **0.79** overall —
tied with `ctrl_v2`.

The interesting part is the shape, which is the predicted one:

| tercile | `ctrl_v2` | `ctrl_v1_on_v2` | difference |
|---|---|---|---|
| light (fast) | **0.76** | 0.71 | +0.05 for ctrl_v2 |
| medium | 0.83 | **0.87** | −0.04 |
| heavy (slow) | 0.78 | **0.90** | −0.12 |

So **the v1 policy loses only on light balls** — the regime that did not exist
in v1 — and is better than the v2-trained controller everywhere else. Its
approach-sign agreement falls from 0.658 on light to **0.504 (chance) on
heavy**, which is the more diagnostic reading: on slow balls it is barely
steering yet still intercepts, because a slow ball gives a wandering paddle
plenty of time. Two caveats before this is taken as "mass does not matter":

1. The comparison is confounded by optimiser luck. `ctrl_v1` was one of six v1
   runs; `ctrl_v2` is one run. A 0.04–0.12 gap is within the seed spread v1
   itself documented (its own re-check moved `ctrl_dense`'s interceptions by
   0.13).
2. **The v1 stack is not guaranteed to be colour-blind.** The v1 VAE is being
   fed out-of-distribution colours, and its `mu` will move with them — not in a
   way it was trained to make meaningful, but not necessarily at zero
   information either. "It cannot read mass" is an assumption here, not a
   measurement. See §8, where its interaction term misbehaves.

## 6. Hold-out colours — does the law interpolate for C?

60 episodes, seeds 6000–6059, `mass_only = (0.85, 1.2)`: colours no model in
the stack has ever seen move.

| controller | interceptions/visit (hold-out) | `ctrl_v2`'s in-dist **medium** tercile | overall in-dist |
|---|---|---|---|
| `stay` | 0.55 [0.46, 0.64] | 0.47 | 0.45 |
| `oracle` | 1.00 [1.00, 1.00] | 0.99 | 0.99 |
| **`ctrl_v2`** | **0.79** [0.72, 0.85] | **0.83** [0.76, 0.91] | 0.79 |
| `ctrl_v1_on_v2` | 0.85 [0.78, 0.91] | 0.87 | 0.79 |

**`ctrl_v2` plays never-seen colours as well as it plays trained colours of
similar speed**: 0.79 [0.72, 0.85] against 0.83 [0.76, 0.91], CIs almost fully
overlapping, and the remaining gap (0.04) is smaller than the spread between its
own three in-distribution terciles (0.76–0.83). Its gap at the floor is 0.111 in
the band against 0.119 in the medium tercile, and its sign agreement 0.677
against 0.709. No interpolation failure is visible.

The weak form of this claim is worth stating: the band is narrow (m ∈ [0.85,
1.2] is speed 0.018–0.026, essentially the v1 speed) so it is not a demanding
extrapolation for a *controller* even if it was for M. The strong version of the
interpolation question was answered at stage M.

## 7. Dream-vs-real transfer

Pearson r between the dream-return curve and the periodic real interception
score (41 real checks per dream run, 9 for `ctrl_v2_real`; plots `runs/*/dream_vs_real.png`).

| run | r (smoothed) | r (raw) | dream return first → last | real interceptions first → last |
|---|---|---|---|---|
| `ctrl_v2` (zh, dense) | **+0.74** | +0.70 | 123.0 → 136.9 | 0.96 → 1.38 |
| `ctrl_v2_mix` (zh, mix) | +0.24 | +0.12 | 13.8 → 17.4 | 1.54 → 1.38 |
| `ctrl_v2_z_only` (z) | +0.38 | +0.41 | 126.8 → 133.8 | 0.96 → 1.12 |
| `ctrl_v2_real` (no WM) | −0.18 | −0.18 | 1.38 → 1.63 (real fitness) | 1.67 → 1.50 |

* **`ctrl_v2` is the cleanest transfer plot in either version of this project**
  (r = +0.74 smoothed). Both curves rise together and both flatten together.
  v1's best dream-vs-real correlation for a `dense` run was r ≈ +0.03, so
  switching the real-side metric from contact frames to interceptions did
  a great deal for the diagnostic — most of v1's "no correlation" was the
  *measurement* being noisy, not the dream failing to transfer.
* **`ctrl_v2_mix`'s +0.24 with a falling real score** is the mix pathology again:
  dream return climbs 13.8 → 17.4 while real interceptions drift 1.54 → 1.38.
* `ctrl_v2_real`'s r is negative and meaningless — both of its curves are
  measurements of the same noisy quantity over 40 generations, and neither moves
  much. Note its left-hand axis is labelled "real training fitness", not "dream
  return"; `--fitness real` has no dream in it and the plot now says so.

## 8. Decision analysis — the interaction term, and why it is not enough

Drive `logit(RIGHT) − logit(LEFT)` on approach frames (ball descending, lower
half), regressed on standardised features. Base model: `x_err, ball_vx,
ball_vy, paddle_vx, speed`. Full model adds `speed·x_err` and `speed·ball_vx`.

| controller | R² base | R² + interactions | ΔR² | frames |
|---|---|---|---|---|
| `ctrl_v2` | 0.4635 | 0.4774 | **+0.0139** | 6,644 |
| `ctrl_v2_z_only` | 0.1981 | 0.1989 | +0.0008 | 7,066 |
| `ctrl_v1_on_v2` | 0.3107 | 0.3299 | **+0.0192** | 7,175 |

Standardised coefficients of the full model:

| controller | x_err | ball_vx | ball_vy | paddle_vx | speed | **speed·x_err** | speed·ball_vx |
|---|---|---|---|---|---|---|---|
| `ctrl_v2` | +0.194 | +0.571 | −0.039 | −0.031 | +0.065 | **+0.341** | −0.240 |
| `ctrl_v2_z_only` | +0.136 | +0.190 | −0.198 | +0.325 | −0.184 | +0.049 | −0.083 |
| `ctrl_v1_on_v2` | −0.121 | +0.093 | +0.060 | +0.421 | +0.046 | **+0.385** | −0.026 |

Read on its own, `ctrl_v2` looks like the design doc's prediction confirmed:
`speed·x_err` (+0.341) is a bigger standardised coefficient than `x_err` itself
(+0.194), i.e. its gain on positional error genuinely scales with how fast the
ball is coming, and the pure `speed` main effect is near zero. And
`ctrl_v2_z_only` — which can see the colour but not the direction of travel —
shows essentially **no** interaction (ΔR² +0.0008, β +0.049), which is the clean
negative control.

**But `ctrl_v1_on_v2` shows a *larger* interaction than `ctrl_v2`**, and it is
driven by the v1 stack, which was never trained on a coloured or fast ball. Two
readings, and I cannot separate them with what is here:

* the interaction is partly **confounded**: `speed` is correlated with the whole
  distribution of approach states (a fast ball spends fewer frames descending,
  arrives at steeper angles, and produces larger `|x_err|` swings), so
  `speed·x_err` can pick up curvature in the drive that has nothing to do with
  the controller reading colour; or
* the **v1 VAE is not colour-blind** on out-of-distribution colours, and its
  `mu` leaks some of the ramp into the v1 controller's input.

Either way the honest conclusion is that **the interaction term is present for
`ctrl_v2` and absent for `ctrl_v2_z_only`, but the test does not by itself
license "C learned the causal law"**, because it also fires on a policy that
should not have it. The design doc listed "interaction term present" as
sufficient evidence; on this data it is not. What would settle it: re-encode the
same real trajectories with the ball recoloured (the stage-M recolour
counterfactual, but scoring the *controller's* drive rather than the dream) and
check the drive changes. That is a one-file experiment and it is not done here.

### The behavioural test: "reacts earlier for light balls"

Two statistics per interception, both counting backwards from the first contact
frame, both defined in `wm/eval_controller_v2.reaction_lead`:

* **A — motion lead**: consecutive frames on which the paddle's velocity pointed
  toward the spot where the interception ended up happening (or it was parked
  within 0.03 of it).
* **B — position lead**: consecutive frames on which the paddle was already
  within half a paddle-width (0.13) of that spot.

A is **dominated by dithering** and should be ignored: a bang-bang paddle flips
`paddle_vx` in place, which resets the run, so `ctrl_v2` scores 1.5 frames and
`stay` scores 5.3. Reported in `summary.md` for completeness only. B is the
number:

| controller | light | medium | heavy |
|---|---|---|---|
| `oracle` | 8.1 | 6.5 | 14.7 |
| `ctrl_v2` | **6.8** | 13.8 | 21.1 |
| `ctrl_v2_z_only` | 10.7 | 15.2 | 24.3 |
| `ctrl_v1_on_v2` | 8.2 | 9.1 | 10.7 |
| `stay` | 28.5 | 32.1 | 37.0 |

**The prediction is not supported.** `ctrl_v2` gets into position 6.8 frames
ahead of contact on light balls and 21.1 on heavy — i.e. *later* for fast balls,
and slightly later than the oracle on the light tercile (6.8 vs 8.1). Every
policy shows the same monotone pattern, because it is mostly a property of the
physics: a light ball's whole descent is ~12 frames, so there is simply less
time in which to be in position. To test "reacts earlier" properly the metric
would have to be normalised by the available descent time, and even then
`stay`'s 28–37 frames show that the statistic rewards *not moving* (a stationary
paddle is trivially "in position" for every hit it happens to get). Read the
table as a sanity check on the physics, not as evidence about the policy.

## 9. Sample budget

| run | dream steps | REAL steps, training | REAL steps, diagnostic | REAL total |
|---|---|---|---|---|
| `ctrl_v2` / `ctrl_v2_mix` / `ctrl_v2_z_only` | 15,360,000 each | **0** | 196,800 each | 196,800 |
| `ctrl_v2_real` | 0 | **1,024,000** | 43,200 | 1,067,200 |
| `ctrl_v1_on_v2` | 0 | 0 | 0 | 0 |

Evaluation cost on top: 9 policies × 150 × 200 = 270,000 in-distribution steps,
4 × 60 × 200 = 48,000 hold-out steps, 400 for the GIFs.

The 196,800 diagnostic steps per dream run buy only the transfer plot:
`ctrl_v2_lastdream` (which consulted zero real episodes) scores 0.77 against the
selected candidate's 0.79. Set `--real-eval-every 0` and the run is genuinely
zero-real-sample, against the baseline's 1,024,000 — and it scores higher.

## 10. Visual artifacts (all looked at, frame by frame)

* `runs/ctrl_eval_v2/real_play_ctrl_v2_light.gif` — seed 5057, m = 0.50, speed
  0.044/frame, **4 interceptions in 200 steps**. A bright yellow ball ricocheting
  fast; the paddle keeps arriving under it.
* `runs/ctrl_eval_v2/real_play_ctrl_v2_heavy.gif` — seed 5146, m = 1.99, speed
  0.011/frame, 1 interception. A purple ball on a long lazy arc; one floor visit
  in the whole episode, which is exactly why per-episode counts are useless here.
* `runs/ctrl_eval_v2/dream_play_ctrl_v2.gif` — the controller inside its own
  dream, decoded by the frozen V. **Cherry-picked**, and labelled as such in the
  code: best of 16 dreams by predicted contact, with the starts restricted to
  light balls (`StartPool.sample_mass_range(…, 0.5, 0.75)`) so the speed is
  visible. **This one contains the most interesting negative result in the
  stage:** over 200 frames the dreamed ball's colour **drifts monotonically from
  yellow through the v1 red to purple**, and around frame 60 it smears and
  briefly fragments. The dream does not conserve mass. Stage M measured a useful
  horizon of 26 frames; this is what that looks like in the colour channel, and
  it means the 150-step dreams the controller trained in were teaching it a world
  whose ball gets heavier as the episode goes on. That the resulting policy is
  flat across mass in the real world is, in hindsight, slightly surprising.
* `runs/ctrl_eval_v2/interceptions_per_visit_by_mass.png`,
  `sign_agreement_by_mass.png`.
* `runs/ctrl_v2*/fitness.png`, `runs/ctrl_v2*/dream_vs_real.png`.

## 11. Surprises and honest failures

1. **The dream does not conserve the ball's colour** (§10). This is the single
   most important thing found at stage three, and it was found by looking at a
   GIF rather than by any metric in the report. It caps how much of the causal
   law a 150-step dream can possibly teach C.
2. **The v1 controller transfers almost perfectly** (§5). The design doc framed
   "how much does a policy that cannot read mass lose on light balls?" expecting
   a large answer. The answer is ~0.05 interceptions per visit, and it is
   *positive* for ctrl_v2 only on the light tercile.
3. **The interaction test has a false positive** (§8). `ctrl_v1_on_v2` shows a
   bigger `speed·x_err` coefficient than `ctrl_v2`. Reported rather than dropped;
   it means the design doc's stated success criterion is too weak.
4. **"Reacts earlier for light balls" is not supported** (§8). Everything reacts
   *later* on light balls because there is less time.
5. **The `mix` reward is worse in v2 than it was in v1**, and in the same
   direction: 1.97 contact frames on 1.24 interceptions, with the pinning
   concentrated on heavy (slow) balls, where the paddle has time to trap the
   ball. v1's advice to use `dense` holds.
6. **z-only did not benefit much from being able to see the speed.** The
   prediction was that v2's z-only controller would do relatively better than
   v1's, because `z` now carries speed. It does move off chance on sign agreement
   (0.649 vs v1's 0.527) but its skill relative to the oracle is about the same
   (58 % vs 61 %). Knowing how fast the ball is, without knowing which way it is
   going, is nearly worthless.
7. **`ctrl_v2`'s dream fitness saturates by generation ~25** (fitness.png):
   123 → 134 in 25 generations, then 134 → 137 over the remaining 175. Same as
   v1. Twenty-five generations would have given ~90 % of the result.
8. **`sigma` barely contracts** (0.50 → 0.38). As in v1, this search has not
   converged in the CMA-ES sense.
9. **The oracle is *better* in v2 than in v1** — 0.99 interceptions per visit
   against v1's 1.63/1.64 = 0.99. Identical, in fact, which is a good sign that
   the speed-scaled floor-visit band is doing its job: if the band were wrong the
   oracle's ratio would drift away from 1.0 in the light tercile, and it does not
   (0.99 light, 1.00 heavy).

## 12. Things to double-check if you are reviewing

* **The speed-scaled floor band.** It sets every denominator in the headline
  table. `floor_zone_height(0.08, 0.045, speed)` and the per-episode
  broadcasting in `floor_visit_stats`; the oracle's 0.99–1.00 across terciles is
  the strongest evidence it is right, since the oracle should intercept every
  chance regardless of mass.
* **The interaction regression's confound** (§8). `ctrl_v1_on_v2`'s +0.385 on
  `speed·x_err` is either a confound or a leak through the v1 encoder, and I did
  not separate them. The recolour-the-controller's-input experiment is the fix.
* **`reaction_lead` table A** is reported but should not be used; if you want a
  commitment-time statistic, normalise table B by the ball's remaining descent
  time.
* **Baseline sign agreement uses the action, not logits**, and the oracle's
  1.000 is therefore a definition, not a measurement.
* **`ctrl_v2_real` used popsize 16 / 8 rollouts / 40 generations** against the
  dream runs' 32 / 16 / 200 (v1's budget, kept for comparability). It is a weaker
  optimiser and its 0.64 should be read as a cost comparison, not a quality one.
* **One seed per configuration.** v1's own re-check moved interceptions by up to
  0.13 between seed sets; the 0.79 vs 0.79 tie between `ctrl_v2` and
  `ctrl_v1_on_v2` is well inside that. A second seed base (e.g. 9000+) would be
  the cheapest way to firm up §5.

## Orchestrator's follow-up on the colour-drift finding (added after review)

Checked whether the dreamed ball's mass (read back with a poly-2 probe on
`log_mass`) is conserved over a 200-step dream from 15 val episodes, at the two
temperatures that matter. Correlation between the dreamed and true log-mass:

| dream step | 0 | 24 | 60 | 120 | 199 |
|---|---|---|---|---|---|
| τ = 0 (deterministic) | 0.97 | 0.95 | 0.96 | 0.93 | 0.88 |
| τ = 1 (as the controller was trained) | 0.79 | 0.30 | −0.24 | −0.12 | −0.73 |

So the model *has* learned that mass is a constant — the deterministic dream
holds it for 200 steps — but under τ = 1 sampling the colour factor random-walks
and is uncorrelated with the true mass within ~25 steps. Mass is a latent with
no restoring force (nothing in the data ever pulls it back to a value), so
per-step sampling noise integrates. This is a general property to expect of any
conserved quantity in a stochastic latent rollout, and it directly affects stage
three: `ctrl_v2` was trained inside 150-step τ = 1 dreams in which the ball's
speed law drifted under it, which is a plausible reason its mass-interaction
term is weak and it does not beat the mass-blind v1 controller. Remedies to try:
train C at a lower τ, or pin the colour subspace during the dream, or add a
consistency loss to M for factors that are constant within an episode.

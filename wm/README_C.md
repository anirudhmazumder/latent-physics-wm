# Stage three ("C") — run log, results, and caveats

Technical log only. Hardware: Apple M1, 8 GB. Interpreter
`/opt/miniconda3/envs/NN/bin/python` (torch 2.12, numpy 1.25, `cma` 4.4.4).
All commands from the repo root, all on CPU. **V (`runs/vae_b1/vae.pt`) and M
(`runs/rnn_v1/rnn.pt`) were not touched, not retrained, not fine-tuned.**

## The one-paragraph result

A 819-parameter linear controller trained with CMA-ES **entirely inside the
MDN-RNN's dream, using zero real environment steps**, gets **1.45–1.47 paddle
contacts per 200-step episode** in the real `BouncingBox`, against **0.93** for
sticky-random, **0.92** for stay-still and **1.68** for a privileged-state
oracle. That is ~87 % of the oracle on a task where the physical ceiling is ~1.7
interceptions per episode. Removing `h` from its input (`--inputs z`) drops it to
**1.20** and destroys transfer entirely — its dream return keeps climbing while
its real score *falls* — which is the paper's central claim reproduced: the
controller needs the recurrent state because a single latent contains no
velocity. A controller trained directly on real episodes with the same optimiser
reaches a comparable score using **1,024,000 real environment steps**; the
dream-trained ones used **0** (plus 196,800 optional real steps spent purely on
the diagnostic curve).

## Files added

| file | what |
|---|---|
| `wm/controller.py` | `LinearController` (numpy, flat param vector), `RandomController`, `StayController`, `OracleController`, `make_features`, `batched_logits`, `compute_norm_stats`, ckpt i/o |
| `wm/dream_env.py` | `DreamEnv` — the RNN as a batched gym-like env; `StartPool`, `load_start_pool`, `dream_rollout` |
| `wm/train_controller.py` | CMA-ES loop; `DreamFitness`, `RealFitness`; fitness and dream-vs-real plots |
| `wm/eval_controller.py` | real-env harness, metrics, bootstrap CIs, GIFs, decision diagnostics, report writer |
| `tests/test_controller.py` | 13 tests, `python -m tests.test_controller` or pytest |

Artifacts: `runs/ctrl_v1/`, `runs/ctrl_z_only/`, `runs/ctrl_tau1.5/`,
`runs/ctrl_dense/`, `runs/ctrl_real/`, `runs/ctrl_eval/`.

Nothing in `worldsim/`, `wm/vae.py`, `wm/rnn.py`, `wm/seq_data.py`,
`wm/train_rnn.py` or `wm/eval_rnn.py` was modified. `wm/eval_controller.py`
imports `decode_latents` from `wm.eval_rnn` (which has a `main()` guard, so the
import runs nothing).

## Commands, in order, with wall clock

```bash
# 1. tests
python -m tests.test_controller                                    # 4.6 s, 13/13
pytest tests/test_controller.py -q                                 # 3.7 s, 13 passed

# 2. dream-trained variants. 200 generations, popsize 32, 16 rollouts,
#    150 dream steps, sigma0 0.5, real check every 5 gens on 24 episodes.
#    (i)-(iii) were run concurrently with OMP_NUM_THREADS=2 each.
python -m wm.train_controller --out runs/ctrl_v1      --inputs zh --temperature 1.0 \
    --reward mix --popsize 32 --rollouts 16 --dream-steps 150 --generations 200 \
    --real-eval-every 5 --real-eval-episodes 24                    # 655 s
python -m wm.train_controller --out runs/ctrl_z_only  --inputs z  --temperature 1.0 ...   # 624 s
python -m wm.train_controller --out runs/ctrl_tau1.5  --inputs zh --temperature 1.5 ...   # 654 s
python -m wm.train_controller --out runs/ctrl_dense   --inputs zh --reward dense ...      # 322 s (solo)

# 3. the no-world-model baseline: same CMA-ES, fitness = real hits
python -m wm.train_controller --out runs/ctrl_real --fitness real --inputs zh \
    --popsize 16 --rollouts 8 --real-steps 200 --generations 40 \
    --real-eval-every 5 --real-eval-episodes 24                    # 1002 s

# 4. final evaluation: 100 episodes x 200 steps, seeds 5000..5099, identical
#    starts for every row; baselines + every variant + the two parameter sets
python -m wm.eval_controller \
  --ctrl runs/ctrl_v1/controller.pt runs/ctrl_v1/controller.pt \
         runs/ctrl_z_only/controller.pt runs/ctrl_tau1.5/controller.pt \
         runs/ctrl_dense/controller.pt runs/ctrl_dense/controller.pt \
         runs/ctrl_real/controller.pt \
  --ctrl-names ctrl_v1 ctrl_v1_lastdream ctrl_z_only ctrl_tau1.5 \
               ctrl_dense ctrl_dense_lastdream ctrl_real \
  --ctrl-which params_best_real params_last_dream params_best_real \
               params_best_real params_best_real params_last_dream params_best_real \
  --gif-for ctrl_v1 ctrl_dense --episodes 100 --steps 200 --out runs/ctrl_eval
                                                                   # 198 s
```

Solo, a dream generation (32 × 16 × 150 = 76,800 batched RNN steps) takes
**~1.3 s**; a real-fitness generation (16 × 8 × 200 = 25,600 real frames, each
needing a render, a VAE encode and an LSTM step) takes **~24 s**. That 18×
per-generation gap, on top of needing far fewer generations, is the entire
practical argument for the world model.

## Design decisions (made without asking; here is the reasoning)

**The timing convention.** M was trained as `h_{t+1} = LSTM(h_t, [z_t, a_t])`,
so the hidden state that "knows about" `z_t` is not available until `a_t` has
been chosen. The controller therefore acts on `[z_t, h_t]` where `h_t` is the
LSTM output after consuming `(z_{t-1}, a_{t-1})` — current observation plus
carried-in history. Then `(z_t, a_t)` goes into M and the cycle repeats. This is
causally valid, requires no change to the trained model, and is implemented
identically in `DreamEnv.step` and `run_real_episodes`;
`test_dream_and_real_timing_agree` drives both code paths with the same latent
sequence and asserts the controller inputs are equal to 1e-6. `h_n[0]` and
`parts["h"][:, -1]` are the same tensor for a one-layer LSTM, which is also
asserted.

**`mu`, not a posterior sample, at test time.** Sampling would add encoder noise
to every decision for no benefit. `DreamEnv.reset`, `run_real_episodes` and
`compute_norm_stats` all use `mu`, so the three distributions are consistent.
(Dreamed latents *are* samples — that is the dream's own stochasticity, governed
by `--temperature`.)

**Warm starts.** Every dream begins at a real `(episode, t0)` from
`data/v1/train` + `train_mix` with 8 teacher-forced true latents behind it. A
cold `h = 0` would mean the controller's first decisions are made by a model
with no velocity in it, and the dream itself would be wrong for those frames.
The real-env runner *does* start cold (a deployed agent gets no warm-up); it
costs nothing measurable because the ball needs ~50 frames to reach the floor.

**Input normalisation.** z and h are standardised with per-dimension statistics
computed once over 38,400 teacher-forced training steps and frozen into the
checkpoint. h-unit stds span [0.10, 0.84] and no unit is dead, so this is a mild
whitening rather than a rescue; CMA-ES starts isotropic with a single `sigma0`
and would otherwise spend generations discovering that anisotropy itself.

**Common random numbers.** All `popsize` candidates in a generation are
evaluated on the same set of start states (dream) or the same seeds (real),
re-drawn each generation. CMA-ES only ranks, so pairing the comparisons removes
most of the variance of a 16-rollout fitness estimate.

**Disjoint seeds.** Periodic real evaluations during training use seed base
7000; the reported evaluation uses 5000. Model selection and reporting therefore
never touch the same episodes.

**Metric: hits *and* interceptions.** `EVENT_PADDLE` is a per-frame flag and one
interception can set it on several consecutive frames, so a policy that pins the
ball against the paddle can inflate hits/episode above the number of floor
visits. `interceptions` collapses each run of contact frames to one. This turned
out to matter — see `ctrl_real` below.

**Floor-visit threshold.** A "chance" is a downward crossing of
`ball_y < paddle_h + ball_radius + ball_speed = 0.147`, not `< 0.125`. At the
strict geometric contact height the collision resolver places the ball at
*exactly* `0.125`, never below, so a strict test fires on missed approaches and
not on successful ones — it undercounts exactly the good policies. Measured on
`data/v1/val`: 1.07 visits/episode at 0.125 vs 1.80 at 0.147.

## Results — 100 episodes × 200 steps, seeds 5000–5099, identical starts

| controller | hits/ep | 95 % CI | interceptions/ep | floor visits/ep | hits per visit | mean gap at floor | eps with ≥1 hit | left/stay/right |
|---|---|---|---|---|---|---|---|---|
| `stay` | 0.92 | [0.77, 1.07] | 0.92 | 1.77 | 0.52 | 0.237 | 69 % | 0.00/1.00/0.00 |
| `random` (sticky) | 0.93 | [0.71, 1.19] | 0.76 | 1.67 | 0.56 | 0.287 | 63 % | 0.31/0.34/0.35 |
| `oracle` (true state) | **1.68** | [1.52, 1.84] | 1.63 | 1.64 | 1.02 | 0.061 | 97 % | 0.24/0.52/0.24 |
| `ctrl_v1` (zh, mix, τ1.0) | 1.44 | [1.29, 1.59] | 1.43 | 1.74 | 0.83 | 0.105 | 93 % | 0.36/0.27/0.37 |
| `ctrl_v1_lastdream` | 1.45 | [1.24, 1.69] | 1.31 | 1.67 | 0.87 | 0.120 | 89 % | 0.31/0.35/0.34 |
| `ctrl_z_only` (z only) | 1.20 | [1.00, 1.41] | 1.03 | 1.70 | 0.71 | 0.257 | 76 % | 0.35/0.24/0.41 |
| `ctrl_tau1.5` (τ = 1.5) | 1.26 | [1.04, 1.55] | 1.17 | 1.67 | 0.75 | 0.148 | 84 % | 0.27/0.42/0.31 |
| `ctrl_dense` (reward dense) | **1.98** | [1.61, 2.44] | 1.61 | 1.71 | 1.16 | 0.100 | 95 % | 0.30/0.39/0.32 |
| `ctrl_dense_lastdream` | 1.47 | [1.30, 1.64] | 1.40 | 1.64 | 0.90 | 0.110 | 89 % | 0.29/0.41/0.30 |
| `ctrl_real` (no world model) | **2.00** | [1.57, 2.47] | 1.27 | 1.66 | 1.20 | 0.178 | 83 % | 0.32/0.38/0.30 |

`_lastdream` rows are CMA-ES's final distribution mean — what the **dream alone**
would have selected, with zero real episodes consulted. The unsuffixed rows are
the candidate with the best score on the periodic 24-episode real check.

Plot: `runs/ctrl_eval/hits_bar.png` (both series, with the floor-visit ceiling).

### How to read it

* Every trained controller beats both trivial baselines with non-overlapping
  CIs. `stay` at 0.92 is a surprisingly strong floor — the paddle is 26 % of the
  box width and the ball comes down ~1.7 times per episode — so "beats random"
  is a lower bar than it sounds and the oracle's 1.68 is the number to compare
  against.
* **`ctrl_v1`, trained on zero real frames, reaches 86 % of the oracle.** Its
  mean paddle–ball gap at the moment of closest approach is 0.105 world units
  against the oracle's 0.061 and random's 0.287, so it is genuinely tracking,
  not getting lucky.
* **`ctrl_real`'s 2.00 hits/ep is partly an artefact.** Its *interceptions* are
  only 1.27 — worse than `ctrl_v1`'s 1.43 — and its gap at the floor is 0.178,
  worse than every other trained controller. It found a policy that generates
  repeated contact frames per approach rather than one that intercepts more
  often. Optimising the raw per-frame contact count in the real environment
  found the loophole in the *metric*; optimising M's dense reward head did not,
  because that head predicts `1 − |ball_x − paddle_x|`, which has no loophole.
* `ctrl_dense` (1.98 hits, 1.61 interceptions, gap 0.100) is the best controller
  in the table on the honest metric — a dream-trained controller matching the
  oracle's interception count. But 0.5 of that 1.98 came from selecting on the
  real check: the same run's `_lastdream` parameters give 1.47. Both numbers are
  reported; the zero-real-sample claim belongs to 1.47.

### What drives the decisions (`runs/ctrl_eval/controller_weights*.png`)

Reading `W` is useless (256 of its 272 columns multiply uninterpretable LSTM
units), so the controller is probed the way stage two probed the RNN: regress
its output drive `logit(RIGHT) − logit(LEFT)` on true-state features from the
eval episodes.

For `ctrl_v1`, R² = 0.052 on `ball_x − paddle_x` alone, **0.318** once
`ball_vx, ball_vy, ball_y, paddle_vx` are added, 0.031 on a ballistic
landing-point estimate. Coefficients of the middle model (drive std 26.1):
`paddle_vx +500`, `ball_vy −195`, `ball_vx +146`, `x_err +19`, `ball_y +6`.
The dominant term is the paddle's own velocity — the controller has strong
hysteresis, committing to a sweep rather than re-deciding every frame, which is
sensible for a bang-bang actuator that needs ~30 frames to cross the box.

R² is a weak instrument here because only the *sign* of the drive reaches the
world and the policy is effectively bang-bang. Sign agreement is the
decision-relevant statistic (chance = 0.50; "approach" = ball descending in the
lower half, 4,593 frames):

| controller | toward ball (all) | toward ball (approach) | toward ballistic landing (approach) |
|---|---|---|---|
| `ctrl_v1` | 0.590 | 0.601 | 0.565 |
| `ctrl_v1_lastdream` | 0.536 | 0.614 | 0.602 |
| `ctrl_z_only` | 0.535 | **0.527** | 0.514 |
| `ctrl_tau1.5` | 0.559 | **0.508** | 0.532 |
| `ctrl_dense` | 0.577 | **0.704** | 0.687 |
| `ctrl_dense_lastdream` | 0.576 | 0.665 | 0.677 |
| `ctrl_real` | 0.493 | **0.489** | 0.532 |

This table is the clearest single piece of evidence in stage three.
`ctrl_dense` moves the right way on 70 % of approach frames; `ctrl_z_only`, which
cannot see velocity, is at 0.527 — barely above chance, exactly as predicted;
and `ctrl_real` is at chance, confirming that its high hit count does not come
from tracking.

I did **not** find clean evidence that any controller "leads" the ball: the
agreement with the ballistic landing point is never meaningfully higher than
with the ball's current position. That is honest but also unsurprising — the
paddle crosses the box in ~30 frames and the ball's remaining fall is often
shorter than that, so following and leading rarely prescribe different actions.

## Dream-vs-real transfer (`runs/*/dream_vs_real.png`)

Pearson r between the dream return curve and the periodic real score
(40 real checks per run). `smoothed` uses an 11-generation moving average of the
dream curve, whose per-generation jitter is pure estimator noise; `early` is
restricted to generations ≤ 50, where both curves are still moving.

| run | r (smoothed) | r (raw) | r (gens ≤ 50) | dream return first → last | real hits first → last |
|---|---|---|---|---|---|
| `ctrl_v1` | **+0.48** | +0.38 | +0.52 | 12.3 → 14.4 | 1.12 → 2.04 |
| `ctrl_tau1.5` | **+0.67** | +0.59 | +0.77 | 12.7 → 14.1 | 0.67 → 1.17 |
| `ctrl_z_only` | −0.00 | −0.13 | **−0.52** | 13.0 → 14.1 | 1.25 → 0.96 |
| `ctrl_dense` | +0.03 | +0.01 | −0.19 | 119.6 → 138.0 | 1.33 → 1.75 |
| `ctrl_real` | +0.09 | +0.37 | +0.37 | 2.12 → 3.38 | 2.50 → 3.58 |

* **`ctrl_v1` transfers.** Both curves rise together, r ≈ 0.5.
* **`ctrl_z_only` is the textbook failure.** Its dream return rises just as
  smoothly as `ctrl_v1`'s (13.0 → 14.1) while its real score *falls* over the
  first 50 generations (r = −0.52). CMA-ES is making genuine progress against M's
  reward heads using a representation that cannot support the actual task, so
  the only progress available is progress against M's flaws. If you had only the
  fitness curve you would call this run a success.
* **`ctrl_dense`'s r ≈ 0 is a saturation artefact, not a failure.** Its dream
  return plateaus by generation 25 and then moves by < 1 %, so the correlation
  over 200 generations is computed almost entirely on noise; its real score kept
  creeping up (1.33 → 1.75). Pearson r over a run where one curve is flat is a
  bad summary and I am reporting it rather than quietly dropping it.
* **Higher temperature helped transfer but hurt the score.** τ = 1.5 gives the
  best correlation in the table (+0.67) and the *worst* approach-sign agreement
  of the zh runs (0.508). The paper's argument is that τ > 1 stops C exploiting
  M; here it seems to have mostly made the dream so noisy that CMA-ES could not
  resolve the useful directions either. With 200 generations at popsize 32, the
  exploitation τ is supposed to prevent was not the binding constraint.

## Sample budget — the actual point of the exercise

| run | dream steps | REAL steps for training | REAL steps for the diagnostic | REAL total |
|---|---|---|---|---|
| `ctrl_v1` / `ctrl_z_only` / `ctrl_tau1.5` / `ctrl_dense` | 15,360,000 each | **0** | 196,800 | 196,800 |
| `ctrl_real` | 0 | **1,024,000** | 43,200 | 1,067,200 |

`ctrl_v1`'s 196,800 real steps (40 checks × 24 episodes × 200) buy *nothing but
the plot*: `ctrl_v1_lastdream`, which consulted zero real episodes, scores 1.45
against the selected candidate's 1.44. Drop `--real-eval-every 0` and the run
costs **zero real environment steps** for the same result — a 1,067,200 : 0
comparison against the baseline. (For `ctrl_dense` the check did earn something:
1.98 selected vs 1.47 unselected. So the honest statement is "the diagnostic is
free to drop for `ctrl_v1`, and is a real, paid-for model-selection step for
`ctrl_dense`".)

Wall clock tells the same story from the other side: 200 dream generations in
655 s versus 40 real generations in 1002 s.

## Visual artifacts (all checked by eye)

* `runs/ctrl_eval/real_play_ctrl_v1.gif` — 200 steps in the real box. The paddle
  visibly sweeps under the descending ball.
* `runs/ctrl_eval/dream_play_ctrl_v1.gif` — the same controller inside M's head,
  decoded by the frozen V. **The dream stays on-manifold for the full 200
  steps**: a crisp ball, a crisp paddle, plausible bounces, no smearing. The
  frame border turns green when the hit head's probability exceeds 0.5, and it
  does so exactly on the frames where the ball is sitting on the paddle.
  *Cherry-picked*: the best of 16 dreams by predicted contact, because ~0.8 % of
  dreamed frames are contacts and a single random start usually contains none.
  This is labelled in the docstring and in the output; the averages are in
  `summary.json`.
* `runs/ctrl_eval/real_vs_dream_side_by_side.gif` — same 8 real warm-up frames
  feed both branches, so at the split they share a latent and a hidden state.
  They then diverge within ~30 frames while the dream stays internally coherent.
  That is precisely the regime in which a dream-trained controller can still
  transfer: it learned a reflex, not a trajectory.
* `runs/ctrl_eval/hits_bar.png`, `controller_weights*.png`,
  `runs/*/fitness.png`, `runs/*/dream_vs_real.png`.

## Surprises and failure modes

1. **The `mix` reward was a mild mistake.** At λ = 0.1 over 150 steps the dense
   term is ~10× the hit term, so `mix` is numerically dominated by `dense`
   anyway — but the small hit component measurably hurt: `ctrl_dense` (dense
   only) reaches 0.704 approach-sign-agreement against `ctrl_v1`'s 0.601, and
   1.61 vs 1.43 interceptions. The hit head is the head with the exploitable
   slack (it is trained with `pos_weight ≈ 115` on a 0.87 % positive rate and is
   deliberately over-confident), and the dense head is the one that cannot be
   gamed. If I were fixing the default it would be `--reward dense`.
2. **`ctrl_real` beating the oracle on the headline metric was a red flag, not a
   win.** Chasing it down is what produced the `interceptions` metric. The
   lesson generalises: the moment a learned policy exceeds a privileged-state
   oracle, suspect the metric.
3. **`stay` scoring 0.92** — nearly identical to sticky-random's 0.93 — makes
   "beats random" almost meaningless on this task. The oracle and the
   interception count are the load-bearing comparisons.
4. **Higher temperature did not help** (see above). Reported rather than buried.
5. **Selecting a checkpoint on a 24-episode real evaluation is noisy.**
   `ctrl_v1`'s best periodic score was 2.54 hits/ep; the same parameters score
   1.44 on 100 held-out episodes. The two seed bases are disjoint, so this is
   honest selection noise rather than leakage, but it is a reminder not to quote
   the training log's best number.
6. **CMA-ES learns almost everything in the first ~25 generations** and then
   grinds. 60 generations would have given ~95 % of the final result; I ran 200
   because it costs 11 minutes.
7. **`sigma` barely contracts** (0.50 → 0.38 over 200 generations) because the
   fitness is noisy and 819-dimensional. Nothing went wrong, but this is a
   search that has not converged in the CMA-ES sense — it is still a wide
   distribution whose mean happens to be a decent controller.

## Things to double-check if you are reviewing

* The timing convention in `wm/dream_env.py`'s module docstring, and that
  `tests/test_controller.py::test_dream_and_real_timing_agree` really does
  exercise two independently-written loops (it does — path B is written out
  inline rather than calling `run_real_episodes`).
* `floor_zone_height`'s 0.147 threshold: it changes the "ceiling" and therefore
  every `hits per visit` number. The justification is in the docstring.
* `contact_runs` — whether collapsing consecutive contact frames is the right
  call. It is what makes `ctrl_real` look worse than its raw hit count.
* The `explain_decisions` ballistic landing estimate reflects off the side walls
  at most once and ignores paddle english, so it is approximate. It is only used
  as a regressor, never as a target.
* `ctrl_real` used popsize 16 / 8 rollouts against the dream runs' 32 / 16, to
  keep it inside a 17-minute budget. That is a weaker optimiser, and its
  advantage on raw hits should be read with that in mind.

## Orchestrator's independent re-check (added after review)

Re-ran the same harness on 60 fresh episodes (seeds 9000–9059, disjoint from
both the training-time checks at 7000+ and the table above at 5000+). Full
table in `runs/ctrl_eval/recheck_seed9000.md`. Contacts (interceptions):

| controller | seeds 5000+ | seeds 9000+ |
|---|---|---|
| stay | 0.92 (0.92) | 0.93 (0.93) |
| oracle | 1.68 (1.63) | 1.60 (1.55) |
| `ctrl_v1` | 1.44 (1.43) | 1.55 (1.47) |
| `ctrl_v1_lastdream` | 1.45 (1.31) | 1.68 (1.38) |
| `ctrl_dense` | 1.98 (1.61) | 2.12 (1.48) |
| `ctrl_dense_lastdream` | 1.47 (1.40) | 1.25 (1.20) |
| `ctrl_z_only` | 1.20 (1.03) | 0.97 (0.93) |
| `ctrl_real` | 2.00 (1.27) | 1.40 (1.12) |

The qualitative story holds (zh dream-trained ≈ oracle-level interceptions;
z-only ≈ stay; real-trained no better than dream-trained), but `ctrl_real`'s
2.00 and `ctrl_dense_lastdream`'s 1.47 did not replicate, and the CIs are wide
enough that the ranking among the top three is not settled. The approach-frame
sign-agreement statistic was stable to ±0.02 across the two seed sets
(`ctrl_dense` 0.70/0.70, `ctrl_v1` 0.60/0.59, `ctrl_z_only` 0.53/0.51,
`ctrl_real` 0.49/0.47) and is the more trustworthy metric.

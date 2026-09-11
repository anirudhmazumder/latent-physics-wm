# Stage two ("M") — run log, results, and caveats

Technical log only. Hardware: Apple M1, 8 GB. Interpreter:
`/opt/miniconda3/envs/NN/bin/python` (torch 2.12, numpy 1.25). All commands run
from the repo root. The VAE (`runs/vae_b1/vae.pt`, z_dim 16) was **not** touched.

## Files added

| file | what |
|---|---|
| `worldsim/policies.py` (extended) | `tracking_action()`, `MixedPolicy` — state-aware ball-tracking mixed with sticky-random segments |
| `worldsim/collect.py` (extended) | `--policy mix`, `--p-track`; closed-loop step path; records `p_track` in meta |
| `worldsim/__init__.py` (extended) | exports `MixedPolicy`, `tracking_action` |
| `wm/seq_data.py` | latent-window dataset (`LatentSequenceDataset`, `make_seq_loader`, `episode_arrays`) |
| `wm/rnn.py` | `RNNConfig`, `MDNRNN`, `rnn_loss`, `save_rnn` / `load_rnn` |
| `wm/train_rnn.py` | training loop + per-epoch open-loop latent rollout metric |
| `wm/eval_rnn.py` | the six diagnostics (a)–(f) |
| `tests/test_rnn.py` | 10 tests, runnable as `python -m tests.test_rnn` or under pytest |

Artifacts: `runs/rnn_v1/` (main), `runs/rnn_g1/` (Gaussian, `--n-gauss 1`),
`runs/rnn_noact/` (`--ablate-actions`), `runs/rnn_v1/eval/`.

## Commands, in order, with wall clock

```bash
# 1. extra data with more paddle contacts
python -m worldsim.collect --out data/v1/train_mix --episodes 300 --steps 200 \
    --seed 10 --ball-radius 0.08 --policy mix --p-track 0.5          # 24 s, 741 MB
python -m worldsim.collect --out data/v1/val_mix --episodes 20 --steps 200 \
    --seed 11 --ball-radius 0.08 --policy mix --p-track 0.5          #  2 s,  49 MB
python -m wm.cache_latents --ckpt runs/vae_b1/vae.pt --data data/v1/train_mix --device cpu   # 80 s
python -m wm.cache_latents --ckpt runs/vae_b1/vae.pt --data data/v1/val_mix   --device cpu   #  5 s

# 2. tests
python -m tests.test_rnn                                             # <20 s, 10/10 pass

# 3. main model (35 epochs)
python -m wm.train_rnn --data data/v1/train data/v1/train_mix \
    --val data/v1/val data/v1/val_mix --out runs/rnn_v1 --epochs 35 --eval-every 2
                                                                     # 685 s (~20 s/epoch)
# 4. controls (35 epochs each, run concurrently with OMP_NUM_THREADS=4)
python -m wm.train_rnn ... --out runs/rnn_g1    --epochs 35 --n-gauss 1       # 1117 s
python -m wm.train_rnn ... --out runs/rnn_noact --epochs 35 --ablate-actions  # 1127 s

# 5. evaluation
python -m wm.eval_rnn --ckpt runs/rnn_v1/rnn.pt --vae runs/vae_b1/vae.pt \
    --val data/v1/val data/v1/val_mix --out runs/rnn_v1/eval         # 26 s
```

Defaults chosen and worth knowing: `--device cpu` (see below), `--stride 4`
(window stride; 19,350 training windows of 32 steps ≈ 619k transitions/epoch),
val loader uses non-overlapping windows and `use_mean=True`.

## Data

* `data/v1/train_mix`: 300 × 200, seed 10, paddle-contact rate **0.868 %**.
* `data/v1/val_mix`: 20 × 200, seed 11, contact rate **1.025 %**.
* Baseline for comparison: pure sticky-random gives **0.625 %**.

**This is well short of the "several %" that was hoped for, and the reason is
physical, not a policy bug.** Contact can only happen when the ball is at the
floor. With `ball_speed = 0.022` and a mean |vy| around 0.015, a full vertical
traverse takes ~55 frames, so the ball visits the floor roughly every 110
frames and a contact lasts 1–2 frames. The ceiling on the contact rate under
*perfect* tracking is therefore ~1.8 %. Measured over 20 episodes:

| p_track | contact rate | episodes with ≥1 contact |
|---|---|---|
| 0.0 (sticky) | 0.625 % | 8 / 20 |
| 0.5 | 1.025 % | 19 / 20 |
| 0.8 | 0.750 % | 20 / 20 |
| 1.0 | 0.925 % | 20 / 20 |

The number that actually improved is coverage: nearly every episode now contains
a contact instead of fewer than half. `p_track = 0.5` was kept (it is also the
requested default, and pure tracking would collapse the paddle-position
marginal onto `paddle_x ≈ ball_x`, teaching the model a shortcut instead of the
dynamics).

Open-loop `sticky`/`uniform` collection still pre-samples the whole action array
from the same rng draws in the same order, so old datasets reproduce from the
same seed. `mix` queries the policy per step, since a tracking action depends on
the current state.

## Training results

Best val MDN NLL (lower is better; val targets are posterior *means*):

| run | config | best val NLL | val hit F1 @0.5 | val reward MSE | open-loop latent MSE @ h=32 |
|---|---|---|---|---|---|
| `runs/rnn_v1` | K=5 MDN, delta, 256 h | **2.210** | 0.44–0.49 | 0.0030 | 0.36 |
| `runs/rnn_g1` | K=1 Gaussian | 2.251 | 0.50 | 0.0031 | 0.19 |
| `runs/rnn_noact` | actions zeroed | 3.232 | 0.45 | 0.0034 | 0.64 |

Model: 327k params, single-layer LSTM(19 → 256), heads as described in `rnn.py`.
Training hit rate 0.759 %, so `pos_weight = 130.8`.

## Evaluation numbers (`runs/rnn_v1/eval/report.json`)

30 val episodes, warm-up K=8 true latents, horizon H=64, true actions.

**(a) pixels.** VAE reconstruction floor 0.000142. Dream MSE at τ=0:
0.00025 (h=1) → 0.0025 (h=16) → 0.0085 (h=64). τ=0.5 / τ=1.0 sit at
~0.0093 mean. The one-step dream is within 2× of the VAE floor, so at short
horizons the decoder and the dynamics contribute comparably; by h=16 the
dynamics error dominates by an order of magnitude.

*(Corrected by the orchestrator after review: the first version of
`eval_rnn.py` compared dream step k against true index `warmup+1+k` instead
of `warmup+k`, charging the model one frame of ball motion at every horizon.
The original numbers were h=1 0.00165, useful horizon 33, |Δball_x|@16 0.024.)*

**(b) world coordinates** (frozen kNN probe, k=10, fit on `data/v1/probe` +
`data/v1/val_mix`, 12k frames). Probe's own floor: ball_x 0.0040, ball_y 0.0032,
paddle_x 0.0141.

| τ | useful dream horizon | \|Δball_x\|@16 | \|Δball_y\|@16 | \|Δpaddle_x\|@16 |
|---|---|---|---|---|
| 0.0 | **35 steps** | 0.019 | 0.019 | 0.027 |
| 0.5 | 0 | 0.229 | 0.186 | 0.103 |
| 1.0 | 0 | 0.229 | 0.212 | 0.157 |

"Useful dream horizon" = first step at which the mean Euclidean ball-position
error exceeds one ball radius (0.08).

**(c) where velocity lives** (6000 timesteps, episode-level split, held-out R²):

| features | probe | ball_x | ball_y | ball_vx | ball_vy | paddle_x | paddle_vx |
|---|---|---|---|---|---|---|---|
| z | linear | 0.401 | 0.233 | 0.024 | −0.109 | 0.756 | 0.017 |
| z | knn | 0.997 | 0.996 | −0.037 | −0.531 | 0.893 | −0.093 |
| h | linear | 0.995 | 0.994 | **0.918** | **0.867** | 0.994 | 0.760 |
| h | knn | 0.996 | 0.996 | 0.879 | 0.739 | 0.536 | 0.119 |
| z+h | linear | 0.995 | 0.994 | 0.922 | 0.875 | 0.994 | 0.769 |

The headline: velocity R² ≈ 0 (and negative) from z, ≈ 0.87–0.92 from h. The
cell state c is similar to h and is stored in `velocity_probe.json` under
`c_linear`. Also note h beats z *linearly* on position too — the LSTM has
re-coded the VAE's nonlinear positional code into a linear one.

**(d) action counterfactuals.** Mean |paddle_x(all-left) − paddle_x(all-right)|
at H=30: **0.660** world units (drift −0.359 left / +0.300 right). The
`--ablate-actions` control: **0.0000**, identical by construction. In the plot
the dreamed paddle saturates at ≈0.13 and ≈0.87, which are exactly the true
clamp limits `paddle_w/2` and `1 − paddle_w/2` — the model learned the paddle's
walls, not just its velocity.

**(e) paddle contact.** Base rate 0.743 %. At threshold 0.5: P 0.347, R 1.000,
F1 0.515 (the `pos_weight` makes 0.5 a bad operating point). Threshold-free:
**PR-AUC 0.727**, best-F1 point **F1 0.759** (P 0.688, R 0.846 at thr 0.994).
Event-triggered average over 51 contacts: P(contact) is 0.0016 at lag −10,
0.348 at lag −3, 0.98 at lag 0 — the model anticipates several frames ahead,
which is the point of the plot.

**(f) walls.** Dreamed ball reverses within ±2 steps of a true bounce in
**36.5 %** of 200 cases, versus a **15.5 %** chance level from applying the same
detector at a random time in the same dream window. Real but weak; see caveats.

## Surprises, failure modes, and things I changed

1. **CPU is ~4× faster than MPS here** (and bit-identical). 20 s/epoch on cpu vs
   ~75 s on mps. A 256-unit LSTM stepped 32 times is a chain of tiny matmuls,
   so it is dispatch-latency-bound, not FLOP-bound. `--device` therefore
   defaults to `cpu` in `train_rnn.py`, unlike `train_vae.py`. `"auto"` still
   works.
2. **Train NLL and val NLL are not comparable** (9.74 vs 2.21). Train targets
   are posterior *samples*, val targets are posterior *means*; the sampled
   target carries the VAE's posterior noise and is genuinely harder. This looked
   alarming until it was traced. Compare like with like across runs only.
3. **Teacher-forced NLL kept improving after the open-loop rollout error
   stopped.** See `runs/rnn_v1/training_curves.png`: val NLL falls from 2.9 to
   2.21 between epochs 12 and 35 while open-loop latent MSE @ h=32 bounces
   around 0.2–0.4 with no trend. This is exactly the pathology the rollout
   metric was added to catch, and it is the reason the metric is in the training
   loop rather than only in eval. Model selection is still on val NLL (it is
   much lower-variance), but the honest statement is that the last ~20 epochs
   bought little dream quality.
4. **The MDN barely beats the Gaussian on NLL (2.210 vs 2.251), and the Gaussian
   is better on open-loop latent MSE (0.19 vs 0.36).** I expected a clearer win
   for the mixture. Two plausible reasons, not disentangled here: (i) val targets
   are posterior means, which strips out much of the genuine multimodality the
   mixture exists to model; (ii) at τ=0 the MDN commits to its argmax component,
   and component-switching between consecutive steps makes the deterministic
   rollout less smooth than a unimodal model's. Both are worth a follow-up.
   Reported as-is rather than dressed up.
5. **τ > 0 dreams score 0 on "useful dream horizon".** This is a metric
   artefact as much as a model property: a sampled dream is a *different*
   plausible future, and the metric measures agreement with *this* episode's
   truth. The τ=1 contact sheet shows the ball still moving and bouncing
   coherently, just elsewhere. The summary text says so. The underlying cause of
   the width is that the model was trained on posterior *samples*, so its
   predictive distribution correctly includes the VAE's own posterior noise —
   which is large in 16 dimensions.
6. **The first version of the bounce test scored 93 %, which was wrong.** It
   counted any sign change of the raw finite-differenced probe output, and the
   probe is noisy at the ~0.005 level, so a straight-line model would also have
   scored ~90 %. Rewritten to require a *sustained* reversal (2-step
   displacement before vs after, both above 0.4 × 2 × ball_speed) and to report a
   chance level from the same detector at a random time. The honest numbers are
   36.5 % vs 15.5 % chance — a real effect, but this is the weakest of the six
   diagnostics and should be read as "better than chance", not "the model knows
   about walls". A cleaner version would condition on which wall and check the
   correct axis sign, not just any reversal.
7. **First contact sheet layout was unreadable** (3-panel strips tiled into a
   grid). Changed to 3 rows (truth / VAE recon / dream) × time along columns.
8. Only **51 paddle contacts** land in the event-triggered average, because val
   is 35 episodes. The ±1 sd band on `hit_prob_around_hits.png` is wide for that
   reason. More val episodes would tighten it.
9. `--ablate-actions` zeroes the action input inside `forward()` rather than
   changing the input width, so parameter count and every shape stay identical
   and the NLL comparison is apples to apples.

## Choices made without asking

* Reward target is evaluated on the **post-transition** state `states[t+1]`,
  matching `(frames[t], a[t]) -> frames[t+1]`.
* Window stride defaults to 4 (stride 1 gives a 4× larger, highly redundant
  epoch for no measurable benefit here).
* Validation uses `use_mean=True` and non-overlapping windows, so val NLL is
  deterministic across epochs and each transition is counted once.
* Checkpoint selection on val MDN NLL, not on the rollout metric (too noisy).
* `mdn_nll` sums over z dims and means over (batch, time), matching the
  convention in `vae_loss`.
* Log-std clamped to [−7, 2].
* The state probe for (b)/(d) is fit on `data/v1/probe` + `data/v1/val_mix`
  (12k frames subsampled) and frozen before any dream is measured.

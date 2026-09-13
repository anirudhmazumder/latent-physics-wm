# v4 stage two — the gravity switch, and LSTM vs transformer

Technical log for v4's V and M stages: the encoder for the gravity-switch world,
a causal transformer added as a second dynamics backbone, and the five
experiments the tier was designed to run. The design is
[`docs/v4/00_v4_design.md`](../docs/v4/00_v4_design.md) (read §2b first — the
controller stage was retired before any training, on the evidence of
[`runs/v4_design/sweep.md`](../runs/v4_design/sweep.md) and
[`sweep_v41.md`](../runs/v4_design/sweep_v41.md)). Hardware: Apple M1, 8 GB;
`python`; every command from the repo root. No
existing dataset or checkpoint was modified.

**The result, in three sentences.** The gravity sign is invisible in a single
frame, as designed: a probe on `mu` scores **0.36–0.40** against a
shuffled-label null of **0.57**, and a position-matched probe scores lower
still. It *is* recoverable from every recurrent model's hidden state — up to
**0.79 balanced accuracy** — but only from about 25 frames after a flip onwards,
and at 0–10 frames after a flip every model sits at chance. So what M holds is
not a memory of the event; it is an **inference from the motion**, riding the
per-traverse energy cue `sweep.md` found by accident. On the one measurement
that isolates the event — the flip counterfactual — **no model, LSTM or
transformer, is above chance.** The tier's headline comparison therefore has no
winner: the thing it was built to discriminate never appeared in either arm.

---

## 1. Commands, in order, with wall clock

```bash
# ---- V. The VAE (started by the orchestrator; 60 epochs, beta 1.0)   18 min
python -m wm.train_vae --data data/v4/train ... --out runs/vae_v4     # 1073 s

# ---- latents for all six splits                                      ~2 min
for s in train train_mix val val_mix probe long; do
  python -m wm.cache_latents --ckpt runs/vae_v4/vae.pt \
      --data data/v4/$s --device mps
done

# ---- V analysis                                                      ~3 min
python -m wm.analyze    --ckpt runs/vae_v4/vae.pt --data data/v4/probe \
    --out runs/vae_v4/analysis --device mps
python -m wm.analyze_v4 --ckpt runs/vae_v4/vae.pt --data data/v4/probe \
    --out runs/vae_v4/analysis --device mps

# ---- M. Five models, 35 epochs each, three at a time (runs/_train_v4.sh)
runs/_train_v4.sh rnn      # runs/rnn_v4        LSTM baseline      1075 s
runs/_train_v4.sh tf       # runs/tf_v4         transformer, ctx 128 549 s (MPS)
runs/_train_v4.sh ff       # runs/rnn_v4_ff     feed-forward floor   472 s
runs/_train_v4.sh noact    # runs/rnn_v4_noact  action ablation      869 s
runs/_train_v4.sh ctx32    # runs/tf_v4_ctx32   transformer, ctx 32  840 s (MPS)

# ---- standard evals, per model (runs/_eval_v4.sh <run> <rnn|cons|alive>)
runs/_eval_v4.sh rnn_v4 rnn        # ~6 min each
runs/_eval_v4.sh rnn_v4 cons       # ~7 min each
runs/_eval_v4.sh rnn_v4 alive      # one call covers all five models, 218 s

# ---- the v4 experiments (a)-(e), all five models, one comparison figure
python -m wm.eval_switch_v4 \
    --models lstm=runs/rnn_v4/rnn.pt transformer=runs/tf_v4/rnn.pt \
             ff=runs/rnn_v4_ff/rnn.pt noact=runs/rnn_v4_noact/rnn.pt \
             tf_ctx32=runs/tf_v4_ctx32/rnn.pt \
    --vae runs/vae_v4/vae.pt --n-pairs 80 \
    --comparison runs/v4_switch_comparison.png       # ~25 min for all five
```

Every wall clock above was measured with two or three jobs sharing the machine
(`OMP_NUM_THREADS=3`), so the per-epoch numbers in §3 are contended and are
comparable to each other but not to a clean single-job run.

**Device.** CPU for the LSTMs (MPS was slower for them in v1–v3.1 and was
re-checked here), MPS for the transformers. Measured on one training batch
(B=64, T=128) with everything else running: transformer 484 ms CPU vs 228 ms
MPS; LSTM 201 ms CPU vs 120 ms MPS. MPS is faster for both, but the GPU is a
single shared device and three MPS jobs contend badly, so splitting the two
kinds across the two devices was the fastest way to run five models on one
laptop.

---

## 2. What is new in the code

| file | what |
|---|---|
| `wm/rnn.py` | refactored: `MDNDynamics` now holds the MDN head, the hit and reward heads, `mdn_nll*`, `most_likely_mean` and `sample_next`. `MDNRNN` supplies only the LSTM. Parameters are still direct attributes, so **every pre-v4 checkpoint loads with a byte-identical state dict**; `RNNConfig` gains `arch` (default `"lstm"`) and `load_rnn` dispatches on it |
| `wm/transformer.py` | **new.** `TransformerConfig` / `TransformerDynamics`: causal pre-norm transformer over the last `context` `(z, a)` pairs, the same heads, the same loss, the same sampler |
| `wm/train_rnn.py` | `--arch lstm\|transformer`, `--context`, `--d-model`, `--n-layers`, `--n-heads`, `--d-ff`, `--dropout` |
| `wm/eval_switch_v4.py` | **new.** Experiments (a)–(e), the per-model `switch/` reports and the comparison figure |
| `wm/probes.py` | `shuffled_label_null` and `classification_suite(..., n_shuffles=)` |
| `wm/analyze_v4.py` | finished: the sign probe is now scored against the shuffled-label null as well as the base rate |
| `wm/eval_conservation.py` | `--sign-metric` / `--sign-probe-data`: metric **(vi)**, probe-decoded sign consistency along the dream |
| `wm/eval_dream_alive.py` | two `None` guards so a world with **no** occluder can be measured (v4 pushes the band above the ceiling) |
| `tests/test_transformer_v4.py` | **new**, 17 tests |

### 2.1 The transformer, and the one design decision in it

`h` is a 3-tuple `(h_last, c_dummy, buf)`. The first two entries exist purely so
that every call site written for an LSTM — `h[0][0]` for "the carried state",
`h[1][0]` for "the cell state" — keeps working with no branch; `buf` is the real
state, the raw inputs the next step will attend over. That is what lets
`eval_rnn`, `eval_dream_alive`, `eval_conservation` and `dream_env` run on a
transformer unmodified, which they do.

Positions are embedded by their offset **from the start of the visible window**,
not by absolute time in the episode. An episode is 200 or 600 frames and the
table has `context` rows, so absolute time is not available; and a
window-relative index is the only scheme under which the incremental `step` path
and the parallel `forward` path compute the *same function*. When a `forward`
call would need more than `context` inputs to serve all of its output positions,
it falls back to looping `step` — slower, and exactly correct. That is why
`test_step_matches_forward_past_the_context` can assert agreement to 1e-5
without qualification, and it matters: training uses `forward` and every dream
uses `step`.

### 2.2 Training window: 128 frames, not v1's 32

All five models train on **`--seq-len 128`**. This is not a tuning choice. v4's
latent is set by an event and must survive ~100 frames; a 32-frame window cannot
contain a flip *and* its consequences, so neither architecture would have a
gradient reason to carry the bit and the comparison would measure nothing. 128
is also the transformer's context, so its position table is fully exercised
during training rather than only its first 32 rows. `--stride 8` keeps the
number of transitions per epoch in the same range as v1's `32 / stride 4`
(576k vs 619k).

### 2.3 Parameter counts, and the one place the brief could not be met

| model | params | note |
|---|---|---|
| LSTM (hidden 256) | **327k** | the v1 architecture, unchanged |
| transformer, d_model 128, 4 layers, `d_ff = 2*d_model` | **571k** | the briefed config |
| transformer, d_model 96, 4 layers | 330k | an exact size match, **not trained** (no budget) |
| transformer, d_model 128, `d_ff = 4*d_model` | 787k | the textbook default, not used |

With `d_model = 128` and 4 layers the attention blocks alone cost 66k per layer,
so **no** feed-forward width brings the briefed config to 327k. `d_ff` was
halved to 2× to close some of the gap and the rest is reported rather than
hidden. The transformer is therefore the **larger** model by 1.7×, which biases
the comparison *in its favour* — so the transformer failing to beat the LSTM (§5)
is the robust direction of this result, and a transformer win would have carried
a size caveat. `tests/test_transformer_v4.py::test_parameter_count_is_in_the_lstms_league`
pins all three numbers so they cannot drift unnoticed.

---

## 3. V — is the sign invisible in one frame?

`runs/vae_v4/analysis/`, 3000 frames / 120 episodes of `data/v4/probe`,
episode-level split. 10 of 16 units active.

**The sign probe and its null** (`report_v4.json`, `sign_probe.png`):

| probe | accuracy | |
|---|---|---|
| majority-class baseline | 0.292 | the test episodes happen to be dominated by the *other* class; see the note below |
| logistic on raw `mu` | **0.360** | |
| kNN on raw `mu` | **0.403** | |
| logistic, position-matched | 0.329 | 2340 of 3000 frames, signs balanced inside every 8×8 position bin |
| kNN, position-matched | 0.333 | |
| **shuffled-label null** | **0.501 / 0.540 mean, 0.572 worst of 5** | same features, same split, same classifier, episode-level label permutation |

Every real number is *below* the null. The sign is not in the frame.

Two things about that table are worth being explicit about, because both look
like bugs and neither is. First, the majority baseline is 0.292 rather than
≥0.5: with only 24 held-out episodes the class that dominates training can be
the minority in test, and the majority-class rule then scores badly. That is
precisely why the shuffled-label null was added — it is the only baseline here
that accounts for the fact that the *effective* sample size is a few dozen sign
runs, not 3000 frames. Second, the probes scoring *below* chance is the same
phenomenon, not evidence of an anti-correlation: 0.36 and 0.50 are one noisy
draw apart at this sample size.

**The size of the available leak.** Under gravity-down the ball sits lower;
measured on the same frames, the `ball_y` histograms differ by a total-variation
distance of **0.126**, so a classifier seeing only `ball_y` could not exceed
**0.563**. `ball_x` differs by 0.055 (0.24 sd). The leak is real, small, and
entirely a position-distribution effect — which is exactly what the design
document predicted and asked to be measured. It returns in §4.

**The v1 sanity check** — positions decode, velocities do not, nothing changed:

| factor | linear | poly2 | poly3 | kNN | MLP |
|---|---|---|---|---|---|
| ball_x | −0.187 | 0.927 | **0.993** | 0.985 | 0.977 |
| ball_y | 0.137 | 0.971 | **0.988** | 0.966 | 0.981 |
| ball_vx | −0.091 | −0.427 | −3.662 | −0.573 | −0.893 |
| ball_vy | −0.070 | −0.297 | −4.568 | −0.251 | −0.198 |
| paddle_x | 0.881 | **0.990** | 0.988 | 0.615 | 0.984 |
| gravity_sign | −0.597 | −0.970 | −5.328 | −1.024 | −1.709 |

MCC 0.221 (pearson) / 0.240 (spearman); `paddle_x` is the one axis-aligned
factor (`z[6]`, |r| 0.71), as in v1–v3.1.

---

## 4. M — training

`--data data/v4/train data/v4/train_mix --val data/v4/val data/v4/val_mix`,
35 epochs, seq-len 128, stride 8, batch 64, lr 1e-3, K=5, residual.

| run | arch | params | best val NLL | final train NLL | s/epoch* | open-loop MSE @1 | @32 | hit F1 | reward-head R² |
|---|---|---|---|---|---|---|---|---|---|
| `rnn_v4` | LSTM 256 | 327k | 2.498 | 9.76 | 30.7 | 0.0726 | 0.818 | 0.378 | 0.902 |
| `tf_v4` | transformer, ctx 128 | 571k | **1.870** | 9.54 | 15.7 | 0.0572 | 0.768 | 0.464 | 0.948 |
| `tf_v4_ctx32` | transformer, ctx 32 | 558k | **1.779** | 9.38 | 24.0 | 0.0589 | **0.615** | 0.470 | **0.963** |
| `rnn_v4_ff` | feed-forward | 114k | 5.385 | 12.29 | 13.5 | 0.1399 | 1.063 | 0.230 | 0.825 |
| `rnn_v4_noact` | LSTM, actions zeroed | 327k | 3.292 | 10.33 | 24.8 | 0.0903 | 0.951 | 0.362 | 0.886 |

\* contended; see §1.

**The first surprise.** `tf_v4_ctx32` — the control built to be *unable* to see
back to the flip — has the **best** val NLL of the five, and the best open-loop
latent MSE at 32 steps. One-step prediction in this world is dominated by
short-range physics (where is the ball going, will it bounce), and a 32-frame
window is both sufficient for that and much easier to learn than a 128-frame
one. The one-step objective simply does not pay for long context. That is the
same mechanism v3 found for object permanence, in its most extreme form: the
flip's payoff is spread over hundreds of frames at 1e-4 per frame, and the loss
never notices.

### 4.1 The v1-style table (`wm.eval_rnn`, `runs/<run>/eval/`)

| run | useful dream horizon (τ=0) | kNN R² vx \| vy **from h** | from z | pixel MSE @16 | action separation @30 | contact PR-AUC |
|---|---|---|---|---|---|---|
| `rnn_v4` | 22 | 0.792 / 0.882 | −0.041 / −0.301 | 0.0042 | 0.520 | 0.545 |
| `tf_v4` | 19 | 0.735 / 0.888 | −0.041 / −0.301 | 0.0050 | 0.517 | 0.585 |
| `tf_v4_ctx32` | 24 | 0.748 / 0.846 | −0.041 / −0.301 | 0.0038 | 0.562 | **0.627** |
| `rnn_v4_ff` | **3** | −0.017 / −0.303 | −0.041 / −0.301 | 0.0110 | 0.581 | 0.375 |
| `rnn_v4_noact` | 20 | 0.802 / 0.893 | −0.041 / −0.301 | 0.0081 | **0.000** | 0.491 |

v1's two central claims replicate on the new world and on the new architecture:
velocity is not in `z` (R² ≤ 0) and is in `h` (0.74–0.89), and the feed-forward
control, which has no recurrence, cannot recover it (−0.02 / −0.30) and dreams
for 3 steps. `gravity_sign` from `h` is ≤ 0 under the *regression* probe for
every model — which is why §5 uses a classifier instead. The VAE pixel floor is
0.00016.

### 4.2 The dream is alive (`runs/v4_dream_alive/`)

v4 has no occluder, so `--occluder-y 1.01 1.02` pushes the band above the
ceiling: "outside the band" becomes the whole frame, "below the band" becomes
identical to "present", and the paddle detector searches every row. Only the
presence column and the reward correlation carry information; arrivals and
re-emergence are artefacts of a band that is not there and are ignored.

At the chosen τ = 0, **every** model dreams a ball on 99–100% of 150 frames
(the real continuation: 100%), with decoded-`ball_y` spread 0.226 / 0.228 /
0.233 / **0.034** / 0.225 against the world's 0.232 — i.e. the LSTM, both
transformers and the action-ablated model dream a ball that moves like a real
one, and the feed-forward floor dreams a ball that essentially sits still.
`r(reward head, decoded |ball_x − paddle_x|)` inside the dream is 0.95 / 0.97 /
0.98 / 0.91 / 0.97. Nothing here blocks the rest of the evaluation.

### 4.3 Conservation (`runs/<run>/conservation/`)

Metrics (i)–(ii) are inapplicable (no mass) and **(iii) is meaningless in v4** —
gravity does work on the ball, so speed is *not* conserved and the "law says
1.000" row is comparing the dream against a law that does not hold. It is left
in the output rather than suppressed, and should be read as noise. (iv) says
88–96% of dreamed frames at τ=1 still contain a well-formed ball.

**(vi), the new one.** Between flips the sign *is* a conserved quantity, and
nothing in the model's loss restores it once a dream lets it drift. A linear
probe on `h`, fitted on real teacher-forced passes over `data/v4/train` and then
frozen, is read along a 190-step dream:

| run | probe accuracy on real data | sign consistency τ=0 | τ=0.5 | τ=1 |
|---|---|---|---|---|
| `rnn_v4` | 0.594 | 0.511 | 0.528 | 0.506 |
| `tf_v4` | 0.537 | 0.519 | 0.508 | 0.511 |
| `tf_v4_ctx32` | 0.587 | 0.498 | 0.483 | 0.462 |
| `rnn_v4_ff` | 0.525 | 0.445 | 0.475 | 0.492 |
| `rnn_v4_noact` | 0.603 | 0.470 | 0.437 | 0.467 |

Chance, at every temperature. Whatever the recurrent state holds on *real*
latents does not survive one step of feeding on its own output. Note that this
probe is deliberately fitted on a much larger pool than `--val`: on 35 val
episodes it comes out at chance itself (0.47–0.51) and the consistency number is
then unreadable rather than negative — an early version of this table made
exactly that mistake.

---

## 5. The v4 experiments

`runs/<run>/switch/report.json`, figure `runs/v4_switch_comparison.png`.
Pool: `val` + `val_mix` + `long` + `data/v4/train` = 215 episodes; the probe is
fitted on 19,000 frames from 150 episodes and scored on 17,000 frames from the
65 held out, of which 8,724 fall after a flip and are what the bins below count
(the rest precede the episode's first flip and are reported separately in
`report.json` under `bins_h_no_flip_yet`). Adding `train` to the pool
is the difference between a weak probe and a real result: on 65 episodes alone
every model came out at the null, and on 215 the structure below appears.

### (a) The sign in `h`, versus frames since the last flip

Balanced accuracy (mean of the two per-class recalls), because the base rate is
0.601 and *moves between bins* — down-traverses take longer, so the slow sign is
over-represented, unevenly with distance from a flip. Plain accuracy is in
`report.json`; the shuffled-label null is 0.53 balanced / 0.55 plain.

| model | 0–10 | 10–25 | 25–50 | 50–100 | 100–200 | 200+ † | overall |
|---|---|---|---|---|---|---|---|
| `lstm` | 0.509 | 0.577 | **0.790** | 0.681 | 0.637 | 0.705 | 0.594 |
| `transformer` | 0.483 | 0.546 | 0.605 | 0.626 | 0.592 | 0.554 | 0.552 |
| `tf_ctx32` | 0.570 | 0.549 | 0.713 | 0.710 | **0.753** | 0.188 | 0.585 |
| `ff` | 0.488 | 0.472 | 0.512 | 0.553 | 0.511 | 0.661 | 0.506 |
| `noact` | 0.513 | 0.567 | **0.802** | 0.658 | 0.710 | 0.661 | **0.606** |
| **`z` (one frame)** | 0.548 | 0.518 | 0.513 | 0.571 | 0.533 | 0.920 | — |
| test frames in bin | 894 | 1276 | 1975 | 2731 | 1736 | **112** | 8724 |

† the 200+ bin has 112 test frames from a handful of `long` episodes and is
noise: it is where `tf_ctx32` reads 0.188 and `z` reads 0.920. Ringed in the
figure; ignore it.

Read the middle panel of `v4_switch_comparison.png`, which plots `h` minus `z`
in the same bin, because the single-frame null is not flat: the further you are
from a flip, the longer the ball has gone without touching the paddle, which
means it has been living high in the box — and how much time it spends high is
what the sign sets. The position leak from §3 therefore *grows* with
frames-since-flip. Subtracting it, the recurrent models gain **+0.28 (lstm,
noact), +0.20 (ctx32), +0.09 (transformer)** at 25–50 frames and **~0.00
(ff)** everywhere.

**Half-lives (e): none, for any model.** Not because recall is flat but because
the curve has the wrong shape for the question — it *rises* from chance at the
flip to its maximum 25–50 frames later. There is no decay from a high initial
value to halve.

### (b) Memory of the event, or inference from the motion?

| model | (i) memory: 0–10 frames after a flip | (ii) inference: 50+ frames, hidden state started *after* the flip |
|---|---|---|
| `lstm` | 0.521 | **0.733** |
| `transformer` | 0.523 | 0.699 |
| `tf_ctx32` | 0.583 | **0.764** |
| `ff` | 0.516 | 0.639 |
| `noact` | 0.519 | 0.723 |
| *`z` alone, same frames* | *0.603* | *0.75* |

(plain accuracy here, to be comparable with the `z` reference row.)

This is the tier's central measurement and it is unambiguous. Column (i) — the
only column in this document that requires a memory of the contact — is at
chance for every model, and **below what a single frame gives on the same
frames**. Column (ii), which forbids memory by construction, is 0.64–0.76. The
models know the sign; not one of them remembers the flip.

That `noact` is the best model overall in (a) is the same finding from another
angle: a flip is *caused* by a paddle contact, so knowing the actions should
help notice one — and removing the actions entirely costs nothing, because
nothing was noticing.

### (c) Dream curvature

Warm up on 16 true frames chosen so no flip falls in the window, dream 60 steps
at τ=0, decode positions with the v1 kNN probe, fit `y = c₀ + c₁t + c₂t²/2`
between bounces on segments of ≥18 frames, compare `sign(c₂)` with the truth.
102 windows per model. Figures: `runs/<run>/switch/dream_curvature.png`.

| what | sign agreement | mean \|acc\| | n fitted |
|---|---|---|---|
| true simulator state — the ceiling | **1.000** | 1.00e-4 | 102 |
| decoded true latents — probe noise only | 0.879 | 1.21e-4 | 99 |
| dreamed, `lstm` | 0.602 | 1.35e-4 | 98 |
| dreamed, `transformer` | 0.594 | 2.07e-4 | 101 |
| dreamed, `tf_ctx32` | 0.596 | 1.83e-4 | 99 |
| dreamed, `noact` | **0.667** | 1.38e-4 | 99 |
| dreamed, `ff` | 0.403 | 1.57e-4 | 62 |
| linear extrapolation (baseline) | 0 by construction | 0 | — |

The instrument works: on the true trajectory the fit recovers the right sign
100% of the time and the right magnitude (1.00e-4 against a true 1e-4), and the
decoder costs 12 points. The dreams land at 0.59–0.67, clearly above a straight
line but far below the 0.879 that a *perfect* dream would score. Magnitudes are
1.3–2.1× too large, i.e. the dreams over-curve. `ff` is at chance and only 62 of
its 102 dreams even contained a segment long enough to fit — its ball barely
moves (§4.2).

### (d) The flip counterfactual — the interventional test

Two arms re-simulated from the **same** recorded state with the **same** action
stream, differing only in the paddle's starting offset, so the model's action
input is identical and only the collision differs. The factual arm is asserted
to reproduce the recording to 1e-4 before any pair is kept; the counterfactual
is asserted to have (or not have) `EVENT_PADDLE` as intended. The true signs
differ between the arms on 95% of pairs. Both directions are built:
`hit → miss` (move the paddle away) and `miss → hit` (move it underneath), the
second because a model that merely responds to *any* collision would pass the
first. Two variants of the warm-up, per the brief: 4 post-contact frames, and 1.

| model | variant | pairs fitted | **opposite signs** (chance 0.50) | hit→miss | miss→hit | factual arm matches its truth |
|---|---|---|---|---|---|---|
| `lstm` | +4 frames | 92 | 0.467 | 0.591 (22) | 0.429 (70) | 0.424 |
| `lstm` | +1 frame | 96 | 0.500 | 0.375 (24) | 0.542 (72) | 0.385 |
| `transformer` | +4 | 95 | 0.411 | 0.435 (23) | 0.403 (72) | 0.432 |
| `transformer` | +1 | 96 | 0.458 | 0.500 (24) | 0.444 (72) | 0.417 |
| `tf_ctx32` | +4 | 94 | **0.245** | 0.087 (23) | 0.296 (71) | 0.404 |
| `tf_ctx32` | +1 | 99 | 0.293 | 0.333 (24) | 0.280 (75) | 0.434 |
| `ff` | +4 | 44 | 0.432 | 0.615 (13) | 0.355 (31) | 0.295 |
| `noact` | +4 | 94 | 0.457 | 0.636 (22) | 0.403 (72) | 0.383 |

**Nothing is above chance, in either variant, in either direction, for any
model.** `tf_ctx32` is significantly *below* chance, which means its dreamed
curvature is more similar across the intervention than two random dreams would
be — its dreams have a strong preferred curvature that the intervention does not
touch. The "factual matches truth" column sits at 0.38–0.47 for all of them,
i.e. also chance, while "counterfactual matches truth" reads 0.54–0.77 — the
asymmetry is not evidence of anything about flips, it is the same preferred
curvature meeting a column whose truth happens to agree with it more often.

The one-frame variant is the strict version: at `t_c + 1` essentially the only
thing separating the two warm-ups is the collision itself, and there is no
post-contact motion from which the new sign could be read. The models score the
same as at +4 frames. They are not reading the sign off the post-contact speed;
they are not doing anything.

### (e) Half-life, and the long dream

Half-life: **none for any model**, for the reason in (a). Long-dream sign
consistency (200 steps, the probe from (a), steps where the model's own hit head
is quiet):

| model | τ=0 | τ=1 |
|---|---|---|
| `lstm` | 0.519 | 0.454 |
| `transformer` | 0.538 | 0.516 |
| `tf_ctx32` | 0.552 | 0.535 |
| `ff` | 0.440 | 0.511 |
| `noact` | 0.494 | 0.479 |

Chance everywhere, matching §4.3's independent measurement through
`eval_conservation --sign-metric`.

---

## 6. The verdict: LSTM vs transformer

**There is no winner, because the question never got asked.** The comparison was
designed to discriminate "carries a bit through hundreds of updates" from "looks
back at the frame where the bit was set", and *neither model does either*. Both
arrive at the sign the same way — by watching the ball move for 25–50 frames —
and on that route the LSTM is slightly ahead (balanced 0.594 vs 0.552 overall;
+0.28 vs +0.09 over `z` at 25–50 frames), while on the interventional test both
are at chance.

What the transformer *did* win, clearly:

* **one-step likelihood** — val NLL 1.870 vs 2.498, and 1.779 at context 32;
* **contact prediction** — PR-AUC 0.585 / 0.627 vs 0.545;
* **reward-head honesty** — R² 0.948 / 0.963 vs 0.902;
* **wall clock** — 549 s vs 1075 s for the same 35 epochs.

And it lost, mildly, on every measurement about the hidden latent. The most
telling single number in this document is that `tf_v4_ctx32` — a transformer
that *cannot see* more than 32 frames back, less than a third of the typical
interval between flips — is the **best** model on val NLL, on open-loop MSE, on
contact PR-AUC and on the reward head, and is at or above the full-context
transformer on the sign probe too. Long context bought nothing at all.

**Caveats, stated plainly.**

1. **One seed per model.** Every gap in §5 is a single draw. The (a) curves'
   shape is consistent across five independently trained models, which is worth
   more than any one gap, but no individual difference between `lstm` and
   `transformer` should be treated as established.
2. **The transformer is 1.7× the LSTM's size** (§2.3), which favours it. The
   size-matched `d_model = 96` variant was not trained.
3. **The per-traverse cue** (`sweep.md`): the sign also sets the ball's energy
   budget, so a traverse takes 48 frames under one sign and 40 under the other.
   That is what (a) and (b)(ii) are measuring. v4's data cannot separate
   "remembers the flip" from "infers the sign" by probing alone, which is why (d)
   exists — and (d) is at chance for everyone.
4. **`h` widths differ** (256 for the LSTMs, 128 for the transformers). A linear
   probe on a wider state has more room, which favours the LSTMs on (a). The
   in-sample accuracies (0.63 lstm, 0.58 transformer) suggest this is small, but
   it is not zero.
5. **`tf_v4_ctx32` differs from `tf_v4` in two ways**, not one: context 32 *and*
   seq-len 32 (a 128-frame window cannot be trained against a 32-frame context
   without falling back to the slow per-step path). Transitions per epoch were
   matched (619k vs 576k), but it is not a clean single-variable control.
6. The 200+ bin in (a) and the `hit → miss` column in (d) are both
   small-n (112 frames; 22–24 pairs).

---

## 7. Surprises, and honest failures

**Surprises.**

* Shorter context wins on every standard metric (§4). The one-step objective has
  no use for 128 frames of history in this world.
* The action ablation is the *best* model on the sign probe (§5a). A flip is
  caused by a contact, so actions ought to help notice one; they cost nothing,
  because nothing is noticing.
* The single-frame null is not flat in frames-since-flip (§5a). Frames long after
  a flip are frames on which the ball has been living high in the box, and the
  position leak §3 measured at TV distance 0.126 grows with that. Any version of
  this experiment that compared `h` against 0.5 instead of against `z`
  bin-by-bin would have reported a memory curve that is mostly a position
  histogram.
* The probe's training-set size changes the headline. On 65 episodes every model
  was at the null; on 215 the +0.28 structure at 25–50 frames appears. A negative
  result about representations is only as strong as the probe that produced it,
  and the first version of this run was not strong enough to make the claim.

**Honest failures.**

* **v4's central claim is not established.** "The latent is in memory" was to be
  shown by sign accuracy from `h` well above chance *long after the flip*; what
  is shown instead is accuracy well above chance *because of the motion since*
  the flip. The design document's expected failure mode — "the one-step loss
  never pays for remembering a bit whose effect per frame is 1e-4" — is what
  happened, to all five models.
* **The flip counterfactual fails for everyone**, so the claim "the model knows
  contact flips gravity" is negative, and the architecture comparison that was
  supposed to sit on top of it has nothing to compare.
* **`eval_conservation`'s metric (iii)** reports a speed law that does not hold
  under gravity. It is left visible rather than removed, and flagged in §4.3, but
  it should have been suppressed for this world.
* **No size-matched transformer** and **no second seed**, both for time.
* The dreamed curvature magnitudes are 1.3–2.1× too large (§5c) and nothing here
  explains why. It is consistent with the dreams over-committing to whichever
  curvature they read out of the warm-up, but that is a guess.

---

## 8. The figures

| file | what to look at |
|---|---|
| `runs/v4_switch_comparison.png` | **the headline.** Left: balanced sign accuracy vs frames since the flip, five models plus the `z` null. Middle: `h` minus `z` in the same bin — the only panel that isolates what M's state adds. Right: memory vs inference, against the `z` reference lines |
| `runs/vae_v4/analysis/sign_probe.png` | the size of the position leak (left) and the four frame-level sign probes against the shuffled-label band (right) |
| `runs/vae_v4/analysis/{recon,traversal,tuning_maps,prior_samples}.png` | the ordinary v1 VAE plates; nothing unusual |
| `runs/<run>/switch/dream_curvature.png` | (c): the three-bar ladder (true state → decoded → dreamed), and two example dreamed `ball_y` traces against the truth. The `ff` panel is the one to look at to see what "the dream is alive but does not move" means |
| `runs/<run>/switch/flip_counterfactual.png` | (d): factual vs counterfactual dreamed acceleration, one point per pair, circles for `hit → miss` and triangles for `miss → hit`. The green quadrants are the ones the model would have to land in. `tf_v4_ctx32`'s panel shows the preferred-curvature artefact clearly — almost every counterfactual is negative |
| `runs/v4_dream_alive/dream_alive.png` | five rows, presence over 150 dream steps. The middle column duplicates the left by construction in this world |
| `runs/<run>/conservation/conservation.png` | panel (iii) is the speed law that does not hold under gravity (§4.3); panel (iv) is well-formedness |
| `runs/<run>/eval/*.png`, `*.gif` | the v1 plates: pixel MSE vs horizon, state error vs horizon, `where_is_velocity.png`, the action counterfactual and the dream GIFs |
| `runs/<run>/training_curves.png` | NLL, the contact head, and the open-loop MSE per epoch |

---

## 9. Where this leaves v4

The tier's V-stage claim holds (the sign is invisible per frame, §3) and the
M-stage claim does not (§5b, §5d). The design document already retired the
C stage on the oracle sweep's evidence; §5 now says the M stage's own headline
question needs a world change too, not just a controller change. Concretely: a
world where the sign does **not** also set the energy budget — `gravity_axis="x"`,
the side-wind of `docs/v4/00_v4_design.md` §2b, which by construction leaves `vy`
untouched and kills the traverse cue — would make (a) and (b)(ii) impossible to
pass by inference, so any above-chance reading there would have to be memory.
That is a one-flag change to `BoxConfig` and a new data collection, and it is the
single highest-value next experiment. Whether *anything* would then be above
chance is exactly the open question; the evidence here is that it would not be,
and that a loss term which rewards noticing the event would be needed before the
LSTM-vs-transformer comparison has two arms worth comparing.

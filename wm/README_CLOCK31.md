# v3.1 follow-up A — the exit clock, and why it did not revive the dream

Technical log. Hardware: Apple M1, 8 GB, CPU. Interpreter
`python`, every command from the repo root. The v3.1
VAE (`runs/vae_v31/vae.pt`) and the frozen baseline `runs/rnn_v31/rnn.pt` were
not retrained and no dataset was modified. Read [`README_V31.md`](README_V31.md)
§4.5 and [`README_C31.md`](README_C31.md) §0 first — they establish the defect
this follow-up tries to fix.

Commands: [`runs/_train_rnn_clock31.sh`](../runs/_train_rnn_clock31.sh) (three
models, ~2.2 h each, three concurrent) and
[`runs/_eval_clock31.sh`](../runs/_eval_clock31.sh) (~20 min).

---

## 0. The verdict, first

> **NO. There is no temperature at which a fair clock model's dream is alive,
> and the clock head made the dream *deader*, not livelier.** The bar set in
> advance was: arrivals below the band within 2× of the real 0.80 per 150
> frames (i.e. 0.40–1.60), decoded-height sd > 0.10, and re-emergence > 70 % of
> the dreams that start hidden. The best fair cell (`clock`, τ = 1.0) scores
> **0.03 arrivals, sd 0.070, 2.6 % re-emergence**. The privileged ceiling
> (`clock_priv`, τ = 1.0) scores **0.03 / 0.076 / 2.6 %**. Neither is close, and
> the privileged row is not better than the fair one, so the failure is not the
> probe's fault.

The one thing that did move is worth stating precisely, because it is the
mechanism:

> The clock head **learned "how long has the ball been gone" and learned nothing
> about "when will it come back."** On truly hidden validation frames the
> trained head reads *frames-since* off `h` at **R² 0.76 / 6.1 frames rmse**,
> and *frames-until* at **R² −0.06 / 12.9 frames rmse** — i.e. at chance. A
> model with no representation of the exit time has nothing to condition the
> re-emergence on, which is exactly the symptom the dream shows.

So v3.1's headline stands unchanged: **the v3.1 dream is not a usable training
environment for this task**, and counting frames does not make it one. No
`ctrl_v31_clock` controller was trained, because there is no alive τ to train it
at (follow-up B, [`README_C31_SEEDS.md`](README_C31_SEEDS.md), is therefore the
five seed-1 rows and nothing more).

---

## 1. What was trained, and what "fair" means here

Three models, identical data (`data/v31/{train,train_mix}`), identical budget
(35 epochs, `--eval-every 2`) and identical everything else to `runs/rnn_v31` —
the only difference is the flag being tested.

| run | flags | what it is | best val NLL |
|---|---|---|---|
| `runs/rnn_v31` | — | the frozen baseline, **not retrained** | 5.292 |
| `runs/rnn_v31_clock` | `--clock-head --w-clock 1` | **FAIR.** A linear head on `h` regressing (frames since the ball was last ≥ half visible, frames until it next is), with the visibility sequence read by a **frozen poly-2 probe on the model's own input latents** | **5.279** |
| `runs/rnn_v31_clock_priv` | `+ --clock-privileged --vy-head` | **PRIVILEGED CEILING.** The same two targets built from the simulator's `ball_visible` column, plus the true `ball_vy` | **5.276** |
| `runs/rnn_v31_clock_emerge` | `+ --rollout-loss-steps 24 --emerge-weight 5` | fair clock **and** v3's emergence fix | 5.968 |

**The fairness is measured, not asserted.** `runs/rnn_v31_clock/clock_info.json`
records the frozen probe's held-out **R² = 0.9948**, and its thresholded reading
("is the ball at least half visible") agrees with the simulator's column on
**99.1 %** of train frames and **99.1 %** of val frames. That is why the fair and
the privileged runs land within 0.003 NLL of each other and within a tenth of a
frame of each other on the clock loss: the probe is not the bottleneck, and the
`clock_priv` row exists precisely so that sentence can be a measurement.

Note the sign of the NLL column. The clock head is a *free lunch on the density*
— both clock runs beat the baseline's 5.292, by about as much as run-to-run
noise. Whatever went wrong downstream did not go wrong because the auxiliary
loss hurt the generative model.

---

## 2. The decisive test — is the dream alive?  (`eval_dream_alive`)

`runs/rnn_v31_clock_dream_alive/` —
[`summary.md`](../runs/rnn_v31_clock_dream_alive/summary.md),
`dream_alive.png`, 242 s. 64 dreams × 150 steps per cell, warm-started on 8 real
frames, driven by one fixed sticky-random action stream; every model and every τ
sees the identical starts and the identical actions. The baseline row is also a
determinism check on the whole pipeline — it reproduces
`runs/rnn_v31_dream_alive` cell for cell.

**Reference (the real continuation of the dreams' own 64 starts):** ball present
35.5 %, ball below the band 2.5 %, **0.80 arrivals / 150 frames**, **height sd
0.164**, **70.0 % re-emergence** of the 40 runs that start hidden.

| model | τ | ball present | below band | **arrivals / run** (real 0.80) | **height sd** (real 0.164) | mean decoded y | **re-emergence** (real 70 %) | r(reward head, −\|gap\|) | alive? |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` ← chosen | 0.0 | 6.1 % | 0.1 % | 0.03 | 0.053 | 0.762 | 0.0 % | 0.97 | no |
| `baseline` | 0.5 | 11.6 % | 10.4 % | 3.62 | 0.059 | 0.121 | **100 %** | 0.96 | no (4.5× too many, sd) |
| `baseline` | 1.0 | 14.9 % | 12.8 % | 4.78 | **0.116** | 0.140 | **100 %** | 0.92 | no (6.0× too many) |
| **`clock`** ← chosen | 0.0 | 5.9 % | 0.1 % | 0.03 | 0.053 | 0.762 | 0.0 % | 0.98 | **no** |
| **`clock`** | 0.5 | 45.9 % | 0.0 % | 0.02 | 0.068 | 0.774 | 0.0 % | 0.98 | **no** |
| **`clock`** | 1.0 | 47.2 % | 0.0 % | 0.03 | 0.070 | 0.770 | 2.6 % | 0.94 | **no** |
| `clock_priv` ← chosen | 0.0 | 6.2 % | 0.1 % | 0.03 | 0.053 | 0.766 | 0.0 % | 0.97 | no |
| `clock_priv` | 0.5 | 40.1 % | 0.0 % | 0.02 | 0.071 | 0.765 | 0.0 % | 0.95 | no |
| `clock_priv` | 1.0 | 36.9 % | 0.0 % | 0.03 | 0.076 | 0.758 | 2.6 % | 0.92 | no |
| `clock_emerge` | 0.0 | 4.1 % | 0.1 % | 0.02 | 0.044 | 0.762 | 0.0 % | 0.96 | no |
| `clock_emerge` | 0.5 | 2.0 % | 0.0 % | 0.02 | 0.034 | 0.777 | 0.0 % | 0.93 | no |
| `clock_emerge` ← chosen | 1.0 | 22.5 % | 0.1 % | **0.11** | 0.082 | 0.760 | 10.3 % | **0.20** | no |

Four readings, in order of how much they change the conclusion.

**1. The clock head traded re-emergence for stability, and that is the opposite
of the intended effect.** Compare `baseline` and `clock` at the same τ. At
τ = 0.5 the baseline brings a ball back in **100 %** of the dreams that start
hidden (far too often — 3.62 arrivals against 0.80, and the "ball" it brings is
a low-variance blob at y ≈ 0.12); `clock` brings one back in **0 %**. At τ = 1.0
it is 100 % against 2.6 %. The clock head painted **more ball** (47 % of frames
present, against the baseline's 15 % and the real 36 %) and put **none of it
below the band** — it made the dream a better-looking still life. Counting how
long the ball has been gone apparently gives the MDN a cheap way to be confident
about a hidden state, and a confident hidden state is a fixed point.

**2. The privileged ceiling does not rescue it,** which is the whole reason that
row was trained. `clock_priv` had the simulator's own visibility column and the
true `ball_vy`, and it is within noise of the fair run on every column of this
table. So the failure is a property of the objective — a per-frame regression
head on `h` — and not of the frozen probe, and a better probe would not help.

**3. `clock_emerge` is the only row that moved the arrival count at all, and it
paid for it with the reward head.** 0.11 arrivals is 3.7× the baseline's 0.03
and still 7× short of the world's 0.80; meanwhile its dream-time reward
correlation collapses from ~0.95 to **0.20**. That number is the one that
matters for a controller: it is the correlation, over frames where a ball was
actually decoded, between what CMA-ES maximises and where the ball really is.
A dream that arrives slightly more often but whose reward signal has stopped
tracking the ball is strictly worse as a training environment.

**4. The closest thing to a live dream in this table is the *baseline* at
τ = 1.0, and it fails too** — 100 % re-emergence and a height sd of 0.116 (above
the 0.10 bar), but 4.78 arrivals per run against 0.80. That is not a ball
falling through the band, it is a ball flickering in and out of existence six
times too often. It is the reason the arrivals criterion is stated as a
two-sided band rather than a floor.

---

## 3. What *is* in `h` — the counters, read directly

Permanence part (f) probes `frames_hidden` off `h` with a **frozen linear probe
refit on fully-hidden frames**, and it reports a *negative* R² for every model
including the privileged one (§4). That reads like "the model did not learn the
counter", and it would be the wrong conclusion, so the head was asked directly.

[`wm/eval_clock_readout.py`](eval_clock_readout.py) →
`runs/rnn_v31_clock_readout/`. `h_t` → the model's own trained clock head →
frames, scored on the 3,130 fully hidden validation frames of
`data/v31/{val,val_mix}` (44.7 % of them). Nothing is refit, and both counters
are scored against the **simulator's** column for the fair and the privileged
run alike, so the two rows are comparable:

| model | *frames-since* R² | rmse (frames) | *frames-until* R² | rmse (frames) |
|---|---|---|---|---|
| `clock` | **0.761** | 6.05 | **−0.063** | 12.89 |
| `clock_priv` | **0.715** | 6.60 | **−0.117** | 13.21 |
| `clock_emerge` | **0.671** | 7.09 | **−0.049** | 12.80 |

Both training logs report a combined clock rmse of ~9.3 frames; this table says
where that 9.3 comes from. **Every one of these models learned target (a) and
none of them learned target (b).** `frames-until` is at or below chance —
predicting the mean would score 0.

This is the mechanism behind §2, and it is the honest reason the hypothesis
failed. The exit clock was proposed on the theory that a model which *knows* the
ball is 14 frames from reappearing has a reason to reappear it. Target (a) is
learnable by integration — `h` already carries "is the ball visible", so a
counter is a running sum and an LSTM does that for free. Target (b) requires
predicting the future of a hidden object, which is the very capability the head
was supposed to induce; asking for it as a regression on `h` does not supply it,
it only *measures* that it is absent. The head reports a state the model already
had and provides no gradient towards the state it did not.

**A caveat on the two tables.** Part (f)'s `frames_hidden` and the clock's
*frames-since* are not the same quantity: (f) counts from the first
**fully** hidden frame and is unclipped, the clock counts from the last
**at-least-half-visible** frame and saturates at 40. On top of that (f)'s probe
is refit on the hidden bin alone with episode-grouped held-out folds, where the
target's variance is small. So the two disagree partly by construction, and the
direct readout above is the one that answers "did the model learn the counter".
The part-(f) row is still worth reading as what it is: a *refit* probe cannot
recover the counter from `h`, even though a head trained jointly with the model
can.

---

## 4. Object permanence — parts (a) and (f)

`runs/rnn_v31_clock_permanence/` — [`report.json`](../runs/rnn_v31_clock_permanence/report.json),
`position_from_h_by_hidden_time.png`,
[`log`](../runs/rnn_v31_clock_permanence.log). All four models in one pass, so
every one is scored against the same frozen position probe, the same episode
split (E = 35, T = 200, 101 hidden runs) and the same baselines. `--skip b e`:
(b)/(d) dream through real occlusions and (e) re-simulates, and neither is the
question here.

**(a) ball position from a frozen probe, hidden frames only** (linear probe;
R² / rmse in world units, `ball_x` then `ball_y`):

| feature | hidden `ball_x` | hidden `ball_y` |
|---|---|---|
| `z` (no memory at all) | 0.11 / 0.222 | −0.02 / 0.102 |
| `h[baseline]` | **0.52** / 0.163 | 0.09 / 0.096 |
| `h[clock]` | 0.31 / 0.196 | **0.21** / 0.090 |
| `h[clock_priv]` | 0.42 / 0.180 | 0.14 / 0.094 |
| `h[clock_emerge]` | 0.28 / 0.200 | **0.22** / 0.089 |

The visible and partial bins are unchanged across all four models (0.99–1.00 and
0.87–0.89), so nothing was lost where the ball is in view. In the hidden bin the
clock models trade **`ball_x` away for `ball_y`**: the baseline's 0.52 on `x`
drops to 0.31, while `y` rises from 0.09 to 0.21. That trade is legible given
§3 — a model optimised to know *how long* the ball has been behind the band is
being pushed towards representing the occlusion's geometry (a hidden ball is
somewhere in the band, so `y` is nearly determined) rather than the ball's
lateral trajectory (`x`, the only coordinate the paddle cares about). The kNN
probe tells a milder version of the same story (`h[clock]` 0.51, `h[clock_priv]`
0.56 against the baseline's 0.50 on `x`), so the linear drop is partly a
linearity effect and partly real; either way **no clock model beats the baseline
on the coordinate the task needs**.

The decay curve (`position_from_h_by_hidden_time.png`) is the same picture: at
k = 1…24 frames hidden every `h` row stays under the "no memory" line — that is
v3.1's permanence result and it survives — and no clock row is below
`h[baseline]` except `h[clock_priv]`, by ~0.02 world units.

**(f) what else is in `h` on fully hidden frames** (frozen linear probe, R²):

| feature | `ball_vx` | `ball_vy` | `frames_hidden` |
|---|---|---|---|
| `z` | −0.005 | −0.183 | −0.577 |
| `h[baseline]` | 0.005 | −0.324 | −0.232 |
| `h[clock]` | 0.076 | −0.063 | −0.336 |
| `h[clock_priv]` | −0.011 | 0.039 | −0.220 |
| `h[clock_emerge]` | **−0.113** | **0.460** | −0.295 |

`clock_emerge`'s `ball_vy` at **0.46** is the single largest improvement anywhere
in this follow-up, and it comes from the emergence fix (the 24-step open-loop
rollout), not from the clock: it is the only model that has to keep a hidden
ball moving over a long rollout and it is the only model that keeps its vertical
velocity. Note that `clock_priv` had `--vy-head` *and the true `ball_vy` as a
target* and still only reaches 0.04 — a per-frame regression head does not put
the quantity into `h`; a multi-step objective does. Every `frames_hidden` entry
is negative, which §3 explains.

---

## 5. Did the clock cost anything elsewhere?  (`eval_rnn`)

`runs/rnn_v31_{clock,clock_priv,clock_emerge}/eval/`. τ = 0.0, warm-up 8,
horizon 64, 30 metric episodes; `h`-linear probes.

| model | useful visible-dream horizon | ball err @ h16 (visible) | contact PR-AUC | contact F1 | `ball_vx` from `h` | `ball_vy` from `h` | `paddle_vx` from `h` |
|---|---|---|---|---|---|---|---|
| `baseline` | 16 | 0.0382 | 0.551 | 0.490 | 0.463 | 0.343 | 0.928 |
| `clock` | 16 | **0.0283** | **0.641** | 0.474 | 0.416 | 0.368 | 0.922 |
| `clock_priv` | **17** | **0.0263** | 0.589 | **0.497** | **0.467** | 0.457 | 0.921 |
| `clock_emerge` | 16 | 0.0618 | 0.524 | 0.428 | 0.436 | **0.645** | **0.953** |

Read this as the "it cost nothing" check that it is. Both clock runs are level
with or slightly better than the baseline on every visible-frame quantity — a
26 % lower ball error at 16 steps for `clock`, a higher contact PR-AUC, the same
16-step useful horizon. `clock_emerge` is the one that paid (0.0618 ball error,
PR-AUC 0.524), which is the usual price of the 24-step rollout loss and matches
its NLL of 5.968.

So the clock head is **cheap and harmless**. It is simply not the thing that was
missing.

---

## 6. What this rules out, and what it leaves

**Ruled out.** "The dream is dead because the model does not track how long the
ball has been hidden" — it does track that (R² 0.76 off `h`), and tracking it
harder makes the dream *more* static. Also ruled out: "a better visibility
signal would fix it" — the privileged run had the simulator's own column and
changed nothing.

**Left standing.** The defect is in **target (b)**: nothing in a teacher-forced
per-frame objective asks the model to commit to *where and when* a hidden ball
reappears. Two directions remain untested, and `clock_emerge`'s `ball_vy` = 0.46
is the evidence for both:

* a **multi-step** objective is the only thing in this project that has ever put
  a hidden-state quantity into `h` that a per-frame head could not;
* the exit should be **predicted as an event** (a distribution over the exit
  frame and exit `x`), not regressed as a scalar countdown — a countdown has no
  way to express "in 8 frames, at x ≈ 0.3, or possibly not at all", which is the
  actual posterior.

Neither is attempted here. Every downstream controller number in
[`README_C31.md`](README_C31.md) and
[`README_C31_SEEDS.md`](README_C31_SEEDS.md) must still be read under the
standing caveat: those policies were fitted to what a reward head says about
latents with no reachable ball in them.

---

## 7. Files

| what | where |
|---|---|
| training script, three models | [`runs/_train_rnn_clock31.sh`](../runs/_train_rnn_clock31.sh), logs `runs/rnn_v31_clock*_train.log` |
| the clock targets, the frozen probe, the two-seed helper | [`wm/clock.py`](clock.py) |
| the `--clock-head` / `--clock-privileged` / `--vy-head` flags | [`wm/train_rnn.py`](train_rnn.py), [`wm/rnn.py`](rnn.py) |
| evaluation script | [`runs/_eval_clock31.sh`](../runs/_eval_clock31.sh) |
| dream-alive (§2) | `runs/rnn_v31_clock_dream_alive/{summary.md,summary.json,dream_alive.png}` |
| permanence (a)+(f) (§4) | `runs/rnn_v31_clock_permanence/report.json`, `runs/rnn_v31_clock_permanence.log` |
| counter readout (§3) | [`wm/eval_clock_readout.py`](eval_clock_readout.py), `runs/rnn_v31_clock_readout/clock_readout.{md,json}` |
| visible-frame eval (§5) | `runs/rnn_v31_{clock,clock_priv,clock_emerge}/eval/report.json` |
| probe fairness numbers | `runs/rnn_v31_clock*/clock_info.json` |
| tests | [`tests/test_clock31.py`](../tests/test_clock31.py) |

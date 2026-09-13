# v3.1 follow-up B — a second training seed for every controller row

Technical log. Hardware: Apple M1, 8 GB, CPU. Interpreter
`python`, every command from the repo root. The v3.1
VAE and all five dynamics checkpoints were frozen, no dataset was modified, and
nothing under an existing `runs/ctrl_v31*/` directory was touched — the seed-1
runs are new directories with an `_s1` suffix.

Read [`README_C31.md`](README_C31.md) §4 first. This document exists to ask
whether that section's ordering is a property of the settings or of the draw.

Commands: [`runs/_train_ctrl_v31_s1.sh`](https://github.com/anirudhmazumder/latent-physics-wm/blob/main/runs/_train_ctrl_v31_s1.sh) and
[`runs/_train_ctrl_v31_real_s1.sh`](https://github.com/anirudhmazumder/latent-physics-wm/blob/main/runs/_train_ctrl_v31_real_s1.sh)
(~12 min each for the dream rows, 2.4 h for the real-fitness row),
[`runs/_eval_ctrl_v31_seeds.sh`](https://github.com/anirudhmazumder/latent-physics-wm/blob/main/runs/_eval_ctrl_v31_seeds.sh) (~21 min).

---

## 0. The verdict, first

> **Not one adjacent ordering in `README_C31.md` §4 survives two seeds.** All
> five neighbouring pairs are unresolved: on the headline metric every gap is
> smaller than the spread of the rows it separates, and four of the five have
> overlapping seed ranges. CMA-ES on 819 parameters against 16 noisy dream
> rollouts moves a row by up to **0.24 interceptions per visit** — larger than
> any gap §4 read off.

Three things do survive, and they are the three the stage actually needed:

> **1. The fair τ = 1 row is above the memoryless bound on both seeds**
> (0.73 and 0.97 against 0.51), and it is above the privileged ceiling, the
> τ = 0 row, the feed-forward row and the z-only control **on both seeds** with
> non-overlapping ranges. The permanence claim is not a draw.
>
> **2. On long required moves — the bin where a memoryless policy collapses —
> `ctrl_v31_tau1` is above *every other row including the real-trained one* on
> both seeds** (0.66–0.72 against real's 0.37–0.51), and it has the *smallest*
> spread of any row in that bin (0.06). This is the single most robust result
> in v3.1's controller stage.
>
> **3. The z-only negative control is bottom on displacement-while-blind on
> both seeds** (0.02 and 0.03, against 0.09–0.34 for every row that reads `h`),
> separated from all five others. The control works.

And one published ordering is now **wrong**:

> **The feed-forward row is not a floor.** `ctrl_v31_ff_s1` scores **0.63**,
> above the bound and above both seeds of `ctrl_v31` (τ = 0). §4's
> "z-only ≈ the bound > feed-forward floor" tail does not reproduce.

No clock controller was trained: follow-up A
([`README_CLOCK31.md`](README_CLOCK31.md) §0) found no temperature at which any
clock model's dream is alive, so there was no τ to train one at.

---

## 1. What was retrained, and the one thing that changed

Five dream-trained rows plus the real-fitness control, each a copy of its
seed-0 command with **exactly one flag changed: `--seed 1`**. In
`wm.train_controller` that single number seeds torch, numpy, the
input-normalisation sample, CMA-ES itself (`seed + 1`) and the per-generation
rollout starts (`seed * 10000 + gen`), so it is a genuine retrain and not a
re-reading of the same search. Temperatures are unchanged and deliberately so —
each was chosen by `wm.eval_dream_alive` from the *dynamics* model, which is
frozen here; only the controller's seed moves.

| row | M | τ | inputs | fitness | seed-1 dir | best real hits/ep during training |
|---|---|---|---|---|---|---|
| `ctrl_v31` | `rnn_v31` | 0.0 | `zh` | dream | `runs/ctrl_v31_s1` | 0.875 (seed 0: 1.333) |
| `ctrl_v31_tau1` | `rnn_v31` | 1.0 | `zh` | dream | `runs/ctrl_v31_tau1_s1` | **2.250** (seed 0: 1.583) |
| `ctrl_v31_ff` | `rnn_v31_ff` | 1.0 | `zh` | dream | `runs/ctrl_v31_ff_s1` | 1.167 (seed 0: 0.958) |
| `ctrl_v31_poshead` | `rnn_v31_poshead` | 0.0 | `zh` | dream | `runs/ctrl_v31_poshead_s1` | 1.292 (seed 0: 1.583) |
| `ctrl_v31_z_only_tau1` | `rnn_v31` | 1.0 | `z` | dream | `runs/ctrl_v31_z_only_tau1_s1` | 1.083 (seed 0: 0.958) |
| `ctrl_v31_real` | `rnn_v31` | — | `zh` | **real**, 1.07 M env steps | `runs/ctrl_v31_real_s1` | 1.417 (seed 0: 1.500) |

Note the last column already tells the story: the training-time real-eval on
24 episodes moves by 0.4–0.7 hits per episode between seeds on four of the six
rows. That is the quantity CMA-ES's model selection is reading.

---

## 2. The evaluation, and the check that makes it meaningful

All **15** v3.1 controllers — six seed-0/seed-1 pairs plus `ctrl_v31_tau0.5`,
`ctrl_v31_allbands` and `ctrl_v31_z_only`, which have no twin — on **one** set
of episodes: seeds 5000–5149, 200 steps, band (0.13, 0.63), paddle 0.16.
Identical starts for every row. `runs/ctrl_eval_v31_seeds/` —
[`summary.md`](../runs/ctrl_eval_v31_seeds/summary.md),
[`two_seeds.md`](../runs/ctrl_eval_v31_seeds/two_seeds.md).

Per-run (V, M) pairing is read from each run's own `history.json` and **printed
by the evaluator** ([`log`](../runs/ctrl_eval_v31_seeds.log)) — checked: every
`_s1` row is driven by the same dynamics model and encoder as its seed-0 twin
(`ff_s1` → `rnn_v31_ff`, `poshead_s1` → `rnn_v31_poshead`, the other four →
`rnn_v31`). The controller reads `h`, so M is part of the policy; a mis-pairing
here would silently turn a seed comparison into a model comparison.

**The reproduction check passed.** Every seed-0 row lands on its published
number to two decimals, CI for CI: `ctrl_v31_tau1` 0.73 [0.66, 0.80],
`ctrl_v31_poshead` 0.68 [0.60, 0.75], `ctrl_v31_real` 0.65 [0.58, 0.71],
`ctrl_v31` 0.57 [0.51, 0.63], `ctrl_v31_z_only` 0.51 [0.45, 0.58],
`ctrl_v31_allbands` 0.47, `ctrl_v31_ff` 0.46, plus the references oracle 0.99,
wait-and-see 0.51, stay 0.38, random 0.31 and (from
`runs/ctrl_eval_v31_extra/`) `ctrl_v31_tau0.5` 0.61 and `ctrl_v31_z_only_tau1`
0.51. Nothing about the evaluation moved between the two runs, so the two-seed
table below is comparable to §4 line for line.

Only `params_best_real` is scored. The `_lastdream` variants are a second
*reading* of the same CMA-ES run, not a second run, so they say nothing about
seed variance; the other bands and the demo GIFs are unchanged facts about
seed 0 and are already published.

---

## 3. The two-seed table

![two seeds](../runs/ctrl_eval_v31_seeds/skill_vs_bound_two_seeds.png)

References on these 150 episodes: **oracle 0.99**, **wait-and-see (the
memoryless bound) 0.51**, stay 0.38, random 0.31.

`spread` is |seed 0 − seed 1|, reported raw. It is **not** a standard deviation
and must not be read as an error bar — two draws say almost nothing about the
distribution they came from. Its honest use is comparative: an ordering whose
gap is smaller than the spread of either row it separates is unresolved.

### (a) interceptions / floor visit — the headline

| controller | seed 0 | seed 1 | mean | spread | above the bound on |
|---|---|---|---|---|---|
| **`ctrl_v31_tau1`** (fair, τ=1) | 0.73 | **0.97** | **0.85** | 0.24 | **both seeds** |
| `ctrl_v31_real` (1.07 M real steps) | 0.65 | 0.75 | **0.70** | 0.10 | **both seeds** |
| *`ctrl_v31_poshead`* (**PRIVILEGED**) | *0.68* | *0.65* | ***0.66*** | *0.02* | ***both seeds*** |
| `ctrl_v31_ff` (the supposed floor) | 0.46 | **0.63** | **0.55** | 0.17 | ONE SEED ONLY |
| `ctrl_v31` (fair, τ=0) | 0.57 | 0.50 | **0.53** | 0.07 | ONE SEED ONLY |
| `ctrl_v31_z_only_tau1` (**negative control**) | 0.51 | 0.53 | **0.52** | 0.03 | ONE SEED ONLY |

### (b) interceptions / visit on long required moves (> 0.35)

The bin where a memoryless policy collapses — `wait_and_see` scores **0.04**
here — and therefore the bin the permanence claim rests on.

| controller | seed 0 | seed 1 | mean | spread |
|---|---|---|---|---|
| **`ctrl_v31_tau1`** | **0.72** | **0.66** | **0.69** | **0.06** |
| *`ctrl_v31_poshead`* | *0.71* | *0.49* | ***0.60*** | *0.23* |
| `ctrl_v31_real` | 0.37 | 0.51 | **0.44** | 0.14 |
| `ctrl_v31` | 0.45 | 0.18 | **0.32** | 0.28 |
| `ctrl_v31_ff` | 0.20 | 0.38 | **0.29** | 0.19 |
| `ctrl_v31_z_only_tau1` | 0.06 | 0.25 | **0.15** | 0.20 |
| — `wait_and_see` (reference) | 0.04 | — | — | — |

### (c) displacement while blind / required move

The outcome-free version of "did it commit in the dark": how far the paddle
actually moved toward the landing point while the ball was hidden. Chance for
the *direction* column is 1/3; `wait_and_see` is 0.000 by construction and
`oracle` scores 0.79 but is **not** a ceiling here (it chases the ball's current
x, not its landing x).

| controller | seed 0 | seed 1 | mean | spread |
|---|---|---|---|---|
| *`ctrl_v31_poshead`* | *0.30* | *0.27* | ***0.28*** | *0.03* |
| `ctrl_v31` | 0.28 | 0.24 | **0.26** | 0.04 |
| `ctrl_v31_real` | 0.34 | 0.17 | **0.25** | 0.17 |
| `ctrl_v31_tau1` | 0.22 | 0.10 | **0.16** | 0.12 |
| `ctrl_v31_ff` | 0.13 | 0.09 | **0.11** | 0.04 |
| `ctrl_v31_z_only_tau1` | **0.02** | **0.03** | **0.02** | **0.01** |

---

## 4. Which orderings survive — plainly

The rule is strict and stated in advance: an ordering is **resolved** only if the
two rows' [min, max] seed ranges do not overlap *and* the gap between their means
exceeds the larger of the two spreads. Adjacent pairs first, then every
non-adjacent pair whose ranges separate — because a row can be unambiguously
above one three places below it while every rung of the ladder between them is
unresolved, and reporting only neighbours would hide that.

### Adjacent pairs: **0 of 5 resolved on the headline metric**

| above | below | gap | larger spread | resolved |
|---|---|---|---|---|
| `ctrl_v31_tau1` | `ctrl_v31_real` | 0.15 | 0.24 | no |
| `ctrl_v31_real` | `ctrl_v31_poshead` | 0.04 | 0.10 | no |
| `ctrl_v31_poshead` | `ctrl_v31_ff` | 0.12 | 0.17 | no |
| `ctrl_v31_ff` | `ctrl_v31` | 0.02 | 0.17 | no |
| `ctrl_v31` | `ctrl_v31_z_only_tau1` | 0.02 | 0.07 | no |

Long-move bin: 0 of 5 resolved. Displacement: 1 of 5 (`ctrl_v31_ff` above
`ctrl_v31_z_only_tau1`, 0.09–0.13 against 0.02–0.03).

### The separations that DO hold on both seeds

**Fair τ = 1 vs everything.** `ctrl_v31_tau1` [0.73, 0.97] sits entirely above
`ctrl_v31_poshead` [0.65, 0.68], `ctrl_v31_ff` [0.46, 0.63], `ctrl_v31`
[0.50, 0.57] and `ctrl_v31_z_only_tau1` [0.51, 0.53]. **Survives.** On long
moves it also clears `ctrl_v31_real`: [0.66, 0.72] against [0.37, 0.51].

**Fair τ = 1 vs real-trained, overall.** [0.73, 0.97] against [0.65, 0.75] —
they overlap by 0.02. **Unresolved overall**, resolved on long moves. So the
claim "197 k dream steps beat 1.07 M real steps" holds only in the form that
matters for permanence (long moves) and not as a headline.

**Real-trained vs the rest.** `ctrl_v31_real` [0.65, 0.75] is above
`ctrl_v31_ff`, `ctrl_v31` and `ctrl_v31_z_only_tau1` on both seeds. **Survives.**

**Privileged vs the rest.** `ctrl_v31_poshead` [0.65, 0.68] is above
`ctrl_v31_ff`, `ctrl_v31` and `ctrl_v31_z_only_tau1` on both seeds, and it has
the tightest spread of any row (0.02). Note the direction: the privileged
ceiling is *below* the fair τ = 1 row on both seeds, overall and on long moves.
**Survives, and it is not a ceiling.**

**vs the memoryless bound (0.51).** Above on **both** seeds: `ctrl_v31_tau1`,
`ctrl_v31_real`, `ctrl_v31_poshead`. **One seed only**: `ctrl_v31`,
`ctrl_v31_ff`, `ctrl_v31_z_only_tau1`. So of the six rows, three are using
memory and three are indistinguishable from a policy that has none.

**The z-only control.** Overall it is at the bound on both seeds (0.51, 0.53) —
as designed. On displacement-while-blind it is at **0.02 / 0.03**, separated
from every other row including the feed-forward one. That is the control
working exactly as intended: strip `h` and the paddle stops moving in the dark.

**The feed-forward "floor".** Does **not** survive, in either direction.
`ctrl_v31_ff_s1` = 0.63 is above the bound, above both seeds of `ctrl_v31`, and
its range [0.46, 0.63] overlaps four of the five other rows. Its long-move and
displacement numbers stay low (0.29, 0.11), so the row is still doing something
memoryless — it just is not a floor on the headline number.

**τ = 0 vs the negative control.** `ctrl_v31` [0.50, 0.57] against
`ctrl_v31_z_only_tau1` [0.51, 0.53]. **Unresolved** — the τ = 0 fair row is not
distinguishable from a controller that cannot see `h` at all. On
displacement-while-blind it clearly is (0.24–0.28 against 0.02–0.03), which is
the honest reading: τ = 0 commits in the dark and does not profit from it.

---

## 5. What this does and does not do to `README_C31.md` §4

**Stands.** "Fair τ = 1 is above the memoryless bound, and above the privileged
ceiling, the τ = 0 row, the feed-forward row and the z-only control." Both
seeds, non-overlapping, and strongest in the long-move bin where it matters.
That is v3.1's result and it is not a draw.

**Stands, narrowed.** "Fair beats real-trained." True on long moves with clear
separation; overall the two seed ranges overlap by 0.02 and the claim should be
stated as the long-move one.

**Falls.** The ordering of the tail — "`z_only` ≈ the bound > `ff` floor" — and
every adjacent gap of 0.02–0.15 that §4 read as an ordering. Five of the seven
trained rows being "above the bound" is a seed-0 fact: on two seeds it is three
of six on both, three of six on one.

**The general lesson, stated once.** The within-pair spread here (up to 0.24) is
of the same size as the between-pair gaps §4 ranked. A single CMA-ES run on a
819-parameter policy scored by 16 noisy dream rollouts is not a deterministic
map from settings to skill, and a per-row bootstrap CI does not capture this at
all — `ctrl_v31_poshead` has a CI of ±0.08 and a spread of 0.02, while
`ctrl_v31_tau1` has a CI of ±0.07 and a spread of 0.24. The CI is the
uncertainty in scoring *one* policy on 150 episodes; the spread is the variation
between two policies the *same recipe* produced, and only the second one answers
"is this ordering a property of the settings".

**The standing caveat is unchanged and applies to every number here.** Follow-up
A confirmed the v3.1 dream is not a live training environment at any temperature
for any model, clock heads included. These policies were fitted to what a reward
head says about latents containing no reachable ball. Whatever they learned, they
did not learn it by watching a simulated ball fall.

---

## 6. Files

| what | where |
|---|---|
| seed-1 training, five dream rows | [`runs/_train_ctrl_v31_s1.sh`](https://github.com/anirudhmazumder/latent-physics-wm/blob/main/runs/_train_ctrl_v31_s1.sh), logs `runs/ctrl_v31_*_s1_train.log` |
| seed-1 training, the real-fitness row | [`runs/_train_ctrl_v31_real_s1.sh`](https://github.com/anirudhmazumder/latent-physics-wm/blob/main/runs/_train_ctrl_v31_real_s1.sh) |
| checkpoints | `runs/ctrl_v31{,_tau1,_ff,_poshead,_z_only_tau1,_real}_s1/controller.pt` |
| the one evaluation | [`runs/_eval_ctrl_v31_seeds.sh`](https://github.com/anirudhmazumder/latent-physics-wm/blob/main/runs/_eval_ctrl_v31_seeds.sh), [`log`](../runs/ctrl_eval_v31_seeds.log) |
| all 15 rows + references | `runs/ctrl_eval_v31_seeds/summary.{md,json}` |
| the two-seed table and orderings | [`wm/two_seeds.py`](two_seeds.py) → `runs/ctrl_eval_v31_seeds/two_seeds.{md,json}` |
| the figure | `runs/ctrl_eval_v31_seeds/skill_vs_bound_two_seeds.png` |
| tests | [`tests/test_clock31.py`](../tests/test_clock31.py) (`seed_summary` and the pairing helper) |

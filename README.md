# latent-physics-wm

Can a neural network watch a simple physical world through 64×64 pixels, learn a
model of it good enough to *imagine* the world forward, and then learn to act
using nothing but its own imagination? This repository answers that four times,
on four versions of the same paddle-and-ball game, each adding one hidden
variable the agent can only get at by understanding something: nothing (v1),
mass encoded in the ball's colour (v2), an occlusion band that hides the ball
(v3 / v3.1), and a gravity direction flipped by an invisible event (v4). Each
tier is a designed experiment with controls, ablations and a stated null — and
the negative results are kept, because two of them are the most interesting
things here.

## The four tiers

**v1 — the recipe works.** A complete VAE → MDN-RNN → CMA-ES controller learned
from pixels. Velocity is not in a single frame (linear R² **0.02** from `z`) but
is in the dynamics model's hidden state (**0.92** from `h`); the dream stays
useful for **35 frames**. *A policy trained on zero real frames plays the real
game at 85–90% of a privileged-state oracle* — 1.4–1.5 interceptions per episode
against the oracle's 1.55–1.63, where standing still scores 0.93. A policy
trained on 1,024,000 *real* frames was not better.
→ [`docs/05_results_and_lessons.md`](docs/05_results_and_lessons.md)

**v2 — an appearance → dynamics causal edge.** The ball's colour sets its mass,
which sets its speed. M reads speed off colour from a single cold-start frame
(r **0.56**, against **0.12** for a colour-blind control trained on the same
frames through the v1 encoder), and repainting the ball inside a dream changes
its dreamed speed with the right sign (slope **−0.60** vs **+0.01** for the
control; the true law is −1.00). The first dream-trained controller reached only
0.79 of the oracle because colour *diffused* in stochastic dreams; a conservation
penalty fixed the drift and the retrained policy reaches **0.94–1.00
interceptions per visit against an oracle's 0.99**, in every mass tercile and on
colours it never saw, still on zero real frames.
→ [`docs/v2/06_v2_controller_fixed.md`](docs/v2/06_v2_controller_fixed.md)

**v3 / v3.1 — memory, and an honest negative.** An opaque band hides the ball.
In v3, M carried the hidden ball's *vertical* state but not its horizontal
position — and then a two-line memoryless oracle turned out to catch **99%** of
balls, so the tier never needed the thing it was measuring. That negative is
kept and it drove v3.1, where an oracle sweep re-designed the band until a
memoryless policy catches only about half. M then does carry the hidden ball's
horizontal position (R² **0.17 → 0.52**, equal to a privileged ceiling of 0.48),
and a controller trained inside its dream scores **0.73 against a memoryless
bound of 0.51**, moving the paddle on 52% of hidden frames. The `h`-ablation
control lands *exactly* on the bound, which is what makes the number mean
something. What transferred was a dense reward on a memory-carrying state, not a
simulation: the dream barely lets the ball out at all.
→ [`docs/v3/09_v31_results_and_lessons.md`](docs/v3/09_v31_results_and_lessons.md)

**v4 — inference beats memory.** A hidden gravity sign, flipped by paddle
contact, invisible in any single frame. Two oracle sweeps (with a closed form)
showed the sign never matters for play — wrong-sign landing error **≤ 0.035** —
so the controller stage was retired before any training and v4 became a dynamics
tier. An LSTM and a causal transformer both learn to *infer* the sign from the
trajectory's curvature, and neither *remembers* the flip: **the interventional
flip counterfactual is at chance (0.25–0.50) for every model**, and the
context-32 transformer, which cannot see back as far as the flip, is the best
predictor of all (NLL **1.78–1.87** vs the LSTM's **2.50**). The objective, not
the architecture, set the ceiling.
→ [`docs/v4/03_v4_results_and_retrospective.md`](docs/v4/03_v4_results_and_retrospective.md)

## How to read this repo in 30 seconds

Start at [`docs/README.md`](docs/README.md) — it has a "10 minutes / 1 hour"
reading guide and a [glossary](docs/glossary.md). The documents in `docs/` are
the narrative: what was built, why, what was measured, what it means. The
`README_*.md` files in `wm/` are the technical run logs: exact commands,
wall-clock times, every number. [`wm/README.md`](wm/README.md) indexes every
module and maps every `runs/<name>` directory to what it is.

## Setup

Python 3.10. Everything is pinned to the versions every result was produced
with.

```bash
conda env create -f environment.yml && conda activate latent-physics-wm
# or, in a venv you already have:
#   pip install -r requirements.txt
```

Install **torch first and separately** if you are not on Apple Silicon —
`requirements.txt` pins 2.12.0, but which wheel you need depends on your
platform, so follow https://pytorch.org/get-started/locally/. `pygame` is only
needed for the interactive viewer; everything else, including GIF recording,
runs headless.

Then use plain `python` for every command in this repository; all of them run
from the repository root.

**Data.** `data/` is gitignored (~7 GB) and fully deterministic, so regenerating
it *is* downloading it:

```bash
bash scripts/collect_all.sh              # all five versions, ~30 min
bash scripts/collect_all.sh --only v1    # just the v1 tier, ~3 min
```

See [`data/README.md`](data/README.md) for the exact command behind every split.
You need the data only to **retrain** or to run evaluations that read a
validation split — the checkpoints ship in `runs/`.

**Expected runtimes** (Apple M1, 8 GB — the machine everything here ran on):
collecting a version 30 s – 2 min; caching its latents 20 s – 2 min; training a
VAE ~54 min on the M1 GPU; training a dynamics model ~11 min; CMA-ES on a
controller ~11 min. Note that **CPU beat GPU by 4× for the small LSTM** —
dispatch latency, not FLOPs, was the bottleneck.

## Quickstart

```bash
python -m pytest -q          # ~45 s; tests that need data skip cleanly without it
```

On a fresh clone with no data this passes with a handful of skips; after
`scripts/collect_all.sh` everything runs. Then watch the v1 world model think —
this needs **no data at all**, only the checkpoints in the repo:

```bash
python -m wm.live --record demo.gif --steps 300 --autopilot   # headless
python -m wm.live                                             # interactive, needs pygame
```

Four panels, same instant: the real frame, the VAE's reconstruction of it, what
M predicted one frame ago this frame would look like, and a free-running dream
that drifts until you re-sync. `A` hands control to the dream-trained
controller; arrow keys play it yourself. An example output is
[`runs/live_demo_autopilot.gif`](runs/live_demo_autopilot.gif).

## Repository layout

```
worldsim/    the environment: physics, rendering, behaviour policies, collection
wm/          V (vae), M (rnn/transformer), C (controller) — models, trainers, evaluations
docs/        the narrative write-up, four tiers
wm/README_*  the technical run logs: exact commands and every number
runs/        checkpoints, figures and reports for every run the docs cite
runs/_*.sh   the per-stage driver scripts (interpreter from $PY, default python)
scripts/     collect_all.sh (regenerate the data), check_links.py
tests/       206 unit tests
data/        datasets — not tracked; see data/README.md
```

## Reproducing a tier end to end

Each tier's run log has the full command sequence with wall-clock times, in
order: the data, the VAE, the latent cache, the dynamics models, the
evaluations, the controllers. Start there rather than from the docs.

| tier | run logs |
|---|---|
| v1 | [`wm/README_M.md`](wm/README_M.md), [`wm/README_C.md`](wm/README_C.md) |
| v2 | [`wm/README_V2.md`](wm/README_V2.md), [`wm/README_M2.md`](wm/README_M2.md), [`wm/README_C2.md`](wm/README_C2.md), [`wm/README_FIX.md`](wm/README_FIX.md), [`wm/README_C2_FIX.md`](wm/README_C2_FIX.md) |
| v3 | [`wm/README_V3.md`](wm/README_V3.md), [`wm/README_M3.md`](wm/README_M3.md), [`wm/README_FIX3.md`](wm/README_FIX3.md), [`wm/README_C3.md`](wm/README_C3.md) |
| v3.1 | [`wm/README_V31.md`](wm/README_V31.md), [`wm/README_C31.md`](wm/README_C31.md), [`wm/README_CLOCK31.md`](wm/README_CLOCK31.md), [`wm/README_C31_SEEDS.md`](wm/README_C31_SEEDS.md) |
| v4 | [`wm/README_M4.md`](wm/README_M4.md) |

## Status and limitations

All four planned tiers are complete. What a reader should hold against the
numbers:

- **One seed per row in most tables.** v3.1 added a second CMA-ES seed per
  controller row and it kept the headline while dissolving *every* adjacent
  ordering — a seed spread of 0.1–0.25 against gaps of 0.02–0.08. That is a
  warning for every other table here, and the first thing to fix.
  ([`docs/v3/10_v31_clock_and_seeds.md`](docs/v3/10_v31_clock_and_seeds.md))
- **M1-only timings**, and no multi-machine or multi-platform check. Nothing
  here has been run on CUDA.
- **Small evaluation sets** in places — v1's validation split contains only 51
  real paddle contacts, and contact counts on 100 episodes carry ±0.3
  confidence intervals.
- **`runs/` is large** (~370 MB) because it carries the checkpoints and the
  figures the documents embed. 66 unreferenced GIFs were pruned; they are
  listed with their regeneration commands in
  [`runs/PRUNED.md`](runs/PRUNED.md). Console logs are written next to each run
  but are not tracked.
- v4's controller stage was **retired by design**, not attempted and failed —
  the sweeps in [`runs/v4_design/sweep.md`](runs/v4_design/sweep.md) show why.

## Credit

The recipe is Ha & Schmidhuber's *World Models* (2018) — the V-M-C
decomposition, training the controller inside the dynamics model's dream, and
the sampling temperature that keeps the dream from being exploited. What is
added here is the per-tier experimental design: a hidden variable, a privileged
ceiling, a memoryless bound, and an ablation that has to land on that bound
before a positive result counts.

**How this was built.** A human set the goals, the tiered plan and the
direction at each step, and reviewed the results. An AI assistant (Claude)
designed the experiments within that plan, implemented the code, ran the
experiments, verified them — re-running tests, re-measuring headline numbers on
fresh seeds, and auditing the code its own sub-agents wrote — and drafted the
documentation. Where a claim rests on a single seed or a single re-check, the
documents say so.

License: to be chosen by the author.

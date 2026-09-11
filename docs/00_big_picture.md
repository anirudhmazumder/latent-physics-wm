# 00 — The big picture: what a world model is and what we built

*Start here. This document explains the idea, the recipe we followed, and how
the pieces in this repo fit together. The numbered documents that follow go
deep on each piece.*

---

## 1. The idea in plain words

When you play a video game you do not think in pixels. Within a few seconds you
have a small mental model: *there is a ball, it moves in straight lines, it
bounces off walls, my paddle is here and it moves when I press a key.* You can
then **imagine** what will happen next — "if I move left now, I'll catch it" —
and act on that imagination, without waiting to see the real outcome.

A **world model** is a learned version of that. It is a neural network (or a
few) that:

1. **compresses** what it sees into a compact description of the situation,
2. **predicts** how that description will change over time, given its actions,
3. and (optionally) is used to **choose actions**, either by planning inside the
   prediction or by training a policy entirely on imagined experience.

This project is about learning to build such models for worlds with *physics*
— objects, motion, collisions, cause and effect — starting from the smallest
possible case and growing it.

## 2. The recipe: V, M, C

We follow the structure of Ha & Schmidhuber's *World Models* (2018), which
splits the system into three parts trained one after another:

```
                 pixels x_t
                     │
        ┌────────────▼────────────┐
        │  V  (Vision)  — a VAE   │   x_t  ──►  z_t          "what is in this frame"
        └────────────┬────────────┘
                     │ z_t
        ┌────────────▼────────────┐
        │  M  (Memory) — an RNN   │   (z_t, a_t, h_t) ──► h_{t+1}, p(z_{t+1})
        └────────────┬────────────┘                       "what happens next"
                     │ [z_t, h_t]
        ┌────────────▼────────────┐
        │  C  (Controller) — tiny │   [z_t, h_t] ──► a_t   "what should I do"
        └─────────────────────────┘
```

- **V** is a variational autoencoder. It squeezes each 64×64 frame into a
  16-number code `z` and can draw the frame back from it. It is trained on
  shuffled single frames and knows nothing about time.
- **M** is a recurrent network with a mixture-density output (an "MDN-RNN"). At
  every step it takes the current code `z_t`, the action `a_t`, and its own
  hidden state `h_t`, and outputs a *probability distribution* over the next
  code `z_{t+1}`. The hidden state `h` is where anything that is *not visible
  in a single frame* — velocity, most importantly — has to live.
- **C** is a deliberately tiny policy (a single linear layer) that maps
  `[z_t, h_t]` to an action. Because it is tiny it can be trained with a
  black-box evolutionary optimiser, and — this is the striking part of the
  paper — it can be trained **entirely inside M's imagination**, never touching
  the real environment, and then transferred back.

Why the split? Because each piece can be trained with a *dense, cheap,
unsupervised* signal (reconstruct the frame; predict the next code) while only
the tiny C needs the sparse, expensive reward signal. Most of the "learning
about the world" happens without any reward at all.

## 3. Our world: a paddle and a ball

The environment (`worldsim/`) is a ball bouncing at constant speed in a box with
a paddle on the floor that the agent slides left or right. Three actions. The
model sees only 64×64 images. The true state is six numbers — ball position and
velocity, paddle position and velocity — which we keep on the side purely to
*grade* the model with.

It is the smallest world that has the three things we want to learn about:
hidden state (velocity), nonlinear dynamics (bounces), and a sparse causal
action channel (the paddle only affects the ball on contact). Details in
[01 — environment and data](01_environment_and_data.md).

## 4. What "understanding physics" means here, operationally

"The model understands physics" is not a measurable statement. These are:

| question | how we test it | where |
|---|---|---|
| Does V know *where* things are? | regress true positions from `z`; R² near 1 | doc 02 |
| Does V know *velocity*? | same regression; it should be **near 0** — a single frame cannot contain velocity | doc 02 |
| Does M recover velocity? | regress velocity from M's hidden state `h`; should be high where `z` was ~0 | doc 03 |
| Does M know about walls? | dream past a wall bounce; does the imagined ball turn around? | doc 03 |
| Does M know actions matter? | dream the same start with all-left vs all-right; does the imagined paddle diverge? | doc 03 |
| How far ahead can M imagine usefully? | decode dreamed frames, measure how the position error grows with horizon | doc 03 |
| Can a policy learned *only in the dream* play the real game? | train C in imagination, then count real paddle hits | doc 04 |

Every one of these is a concrete number or picture produced by a script in
`wm/`, and every document links to the artefact it discusses.

## 5. Map of the repository

```
worldsim/                the environment and data collection (numpy only)
  bouncing_box.py          physics + antialiased renderer
  policies.py              sticky-random and ball-tracking data-collection policies
  collect.py               writes frames/actions/states/events as memory-mappable .npy
  render.py                GIFs, side-by-side rollouts, frame grids
  play.py                  play it yourself (pygame)

wm/                      the world model
  vae.py, train_vae.py     V: the convolutional VAE and its trainer
  diagnostics.py, probes.py, analyze.py, calibrate_probes.py
                           V diagnostics: reconstructions, traversals, probes, tuning maps
  cache_latents.py         freeze V, encode every frame once → (mu, logvar) on disk
  seq_data.py              latent sequence windows for M
  rnn.py, train_rnn.py     M: the MDN-RNN and its trainer
  eval_rnn.py              M diagnostics: dream rollouts, horizon curves, velocity probes,
                           action counterfactuals, bounce test, hit prediction
  controller.py, train_controller.py, eval_controller.py
                           C: linear policy, CMA-ES training in the dream, real-env evaluation
  live.py                  play the game while watching V, M and the dream side by side

data/v1/                 datasets (see doc 01)
runs/                    checkpoints, histories and every figure referenced in the docs
docs/                    you are here
tests/                   unit tests for M and C
```

## 6. Reading order

1. [01 — The environment and the data](01_environment_and_data.md)
2. [02 — V: the vision model](02_vae_the_vision_model.md)
3. [03 — M: the dynamics model](03_dynamics_model.md)
4. [04 — C: the controller, trained in a dream](04_controller.md)
5. [05 — Results, lessons, and what comes next](05_results_and_lessons.md)
6. [Glossary](glossary.md)

## 7. Reproducing everything

All commands use the project's Python environment (see
[05](05_results_and_lessons.md#reproducing) for the exact interpreter). The
whole pipeline, in order:

```bash
# data
python -m worldsim.collect --out data/v1/train --episodes 150 --steps 200 --ball-radius 0.08
python -m worldsim.collect --out data/v1/val   --episodes 15  --steps 200 --ball-radius 0.08 --seed 1
python -m worldsim.collect --out data/v1/probe --episodes 120 --steps 24  --ball-radius 0.08 --seed 777
python -m worldsim.collect --out data/v1/train_mix --episodes 300 --steps 200 --ball-radius 0.08 --seed 10 --policy mix
python -m worldsim.collect --out data/v1/val_mix   --episodes 20  --steps 200 --ball-radius 0.08 --seed 11 --policy mix

# V
python -m wm.train_vae --data data/v1/train --val data/v1/val --out runs/vae_b1 --epochs 80 --free-bits 0.5
python -m wm.analyze --ckpt runs/vae_b1/vae.pt --data data/v1/probe
for s in train val probe train_mix val_mix; do python -m wm.cache_latents --ckpt runs/vae_b1/vae.pt --data data/v1/$s; done

# M
python -m wm.train_rnn --data data/v1/train data/v1/train_mix --val data/v1/val data/v1/val_mix \
    --out runs/rnn_v1 --epochs 35 --eval-every 2
python -m wm.eval_rnn --ckpt runs/rnn_v1/rnn.pt --vae runs/vae_b1/vae.pt \
    --val data/v1/val data/v1/val_mix --out runs/rnn_v1/eval

# C
python -m wm.train_controller --out runs/ctrl_v1 --inputs zh --reward mix --temperature 1.0 \
    --popsize 32 --rollouts 16 --dream-steps 150 --generations 200 --real-eval-every 5
python -m wm.eval_controller --ctrl runs/ctrl_v1/controller.pt --episodes 100 --out runs/ctrl_eval

# tests
python -m tests.test_rnn && python -m tests.test_controller
```

(Full command lines with every flag and wall-clock time: `wm/README_M.md` and
`wm/README_C.md`.)

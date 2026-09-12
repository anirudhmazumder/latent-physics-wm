# latent-physics-wm

Learning to build **world models that understand physics and cause and effect**,
one small environment at a time.

**v1 (this repo's current state):** a complete Ha & Schmidhuber-style world model
— VAE → MDN-RNN → CMA-ES controller — for a paddle-and-ball game, learned from
64×64 pixels. The dynamics model discovers velocity in its hidden state, dreams
~35 frames ahead, and responds correctly to counterfactual actions; a controller
trained on **zero real frames** inside the dream plays the real game at 85–90% of
a privileged-state oracle.

**v2 (mass from colour):** the ball's colour encodes its mass, which sets its
speed. The dynamics model reads speed off colour from a single frame (r 0.56 vs
0.12 for a colour-blind control), and repainting the ball in a dream changes its
dreamed speed with the right sign (slope −0.6 vs +0.01) — an appearance →
dynamics causal edge learned from pixels. The first dream-trained
controller reached only 79% of the oracle, because the dreamed ball's colour
diffused in stochastic dreams; a conservation penalty on M fixed the drift, and
a controller retrained in the fixed dream plays at **oracle level across all
masses and on never-seen colours**, still on zero real frames. Repainting the
ball in a real history shifts its decisions in the direction the physics
predicts. Docs in `docs/v2/`.

**v3 (occlusion band):** an opaque band hides the ball for ~10 frames per
traverse. The dynamics model carries the hidden ball's *vertical* state and the
time since it vanished (exit side 92% right) but not its horizontal position,
because one-step prediction never pays for it until the ball reappears; a
privileged position head proves the LSTM could hold it. A memoryless oracle
then catches 99% of balls, so the default band never required object
permanence for play — the honest result, and the reason the next step is a
harder band. Docs in `docs/v3/`.

**v3.1:** the band re-designed by an oracle sweep so a memoryless policy
catches only half the balls. The dynamics model now carries the hidden ball's
horizontal position (R² 0.52, equal to a privileged ceiling) but loses the exit
clock, and its dream never lets the ball out — yet a controller trained inside
it beats the memoryless bound by 0.22, moves the paddle while the ball is
hidden, and collapses to the bound when its memory input is removed. What
transferred was a dense reward on a memory-carrying state, not a simulation.
Docs in `docs/v3/06–09`.

**Start with the docs:** [`docs/README.md`](docs/README.md) — written so that
someone new to world models can follow what was built, why, what was measured,
and what it means.

```
worldsim/   the environment (numpy), data collection, rendering
wm/         V (vae), M (rnn), C (controller): models, trainers, evaluations
data/v1, v2 datasets (not tracked; regenerate with worldsim.collect)
runs/       checkpoints and every figure the docs reference
docs/       the write-up
tests/      unit tests: python -m tests.test_rnn && python -m tests.test_controller
```

Python environment: conda `NN` (`/opt/miniconda3/envs/NN/bin/python`), PyTorch
2.12, numpy, scikit-learn, matplotlib, Pillow, scipy, `cma`.

Roadmap: revive the v3.1 dream (exit clock) → v4 gravity switch (transformer vs LSTM) (see
`worldsim/worldsim.md` and `docs/05_results_and_lessons.md`).

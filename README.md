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

Roadmap: v3 occlusion → v4 gravity switch (see
`worldsim/worldsim.md` and `docs/05_results_and_lessons.md`).

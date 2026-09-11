# latent-physics-wm

Learning to build **world models that understand physics and cause and effect**,
one small environment at a time.

**v1 (this repo's current state):** a complete Ha & Schmidhuber-style world model
— VAE → MDN-RNN → CMA-ES controller — for a paddle-and-ball game, learned from
64×64 pixels. The dynamics model discovers velocity in its hidden state, dreams
~35 frames ahead, and responds correctly to counterfactual actions; a controller
trained on **zero real frames** inside the dream plays the real game at 85–90% of
a privileged-state oracle.

**Start with the docs:** [`docs/README.md`](docs/README.md) — written so that
someone new to world models can follow what was built, why, what was measured,
and what it means.

```
worldsim/   the environment (numpy), data collection, rendering
wm/         V (vae), M (rnn), C (controller): models, trainers, evaluations
data/v1/    datasets (not tracked; regenerate with worldsim.collect)
runs/       checkpoints and every figure the docs reference
docs/       the write-up
tests/      unit tests: python -m tests.test_rnn && python -m tests.test_controller
```

Python environment: conda `NN` (`/opt/miniconda3/envs/NN/bin/python`), PyTorch
2.12, numpy, scikit-learn, matplotlib, Pillow, scipy, `cma`.

Roadmap: v2 mass-from-colour → v3 occlusion → v4 gravity switch (see
`worldsim/worldsim.md` and `docs/05_results_and_lessons.md`).

# Independent re-check on fresh seeds (orchestrator)

90 episodes, seeds 9000+, both training seeds of the fair tau=1, ff and real-trained controllers. Each scored with its own (V, M).

# stage three (C) — real-environment evaluation by occlusion

90 episodes x 200 steps, seeds 9000..9089, identical starts for every row. Band (0.13, 0.63), paddle width 0.16.

`oracle` has perfect vision and perfect memory; **`wait_and_see` has perfect vision and NO memory** (it tracks the true ball only while `ball_visible > 0.5` and STAYs otherwise). Every trained row should be read as a position between those two, and the single most important fact about any row is which side of `wait_and_see` it falls on: **above it is memory, below it is not**. Any row whose name contains `poshead` is a **PRIVILEGED CEILING** — its dynamics model was trained with a supervised head on the simulator's true ball position — and is not a world-model result.

**The two numbers everything is graded against, measured here on these 90 episodes: oracle 0.98, memoryless (wait-and-see) bound **0.53**, stand still 0.41.**

Each controller is driven by the (V, M) it was trained with; the policy reads `h`, so M is part of the policy:

| controller | M (dynamics) | V (encoder) |
|---|---|---|
| `ctrl_v31_ff` | `runs/rnn_v31_ff/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_ff_s1` | `runs/rnn_v31_ff/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_real` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_real_s1` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_tau1` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_tau1_s1` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |

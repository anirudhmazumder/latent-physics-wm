# Documentation — v1 world model for the paddle game

Read in order. Each document is self-contained enough to be read alone, but
they build on one another.

| # | document | one line |
|---|---|---|
| 00 | [The big picture](00_big_picture.md) | What a world model is, the V-M-C recipe, how this repo maps onto it |
| 01 | [The environment and the data](01_environment_and_data.md) | The game, why it looks the way it does, how data was collected |
| 02 | [V: the vision model](02_vae_the_vision_model.md) | The VAE, posterior collapse, what the latent space actually encodes |
| 03 | [M: the dynamics model](03_dynamics_model.md) | The MDN-RNN, dreams, where velocity lives, counterfactuals |
| 04 | [C: the controller](04_controller.md) | Training a policy inside the dream, transfer to the real game |
| 05 | [Results, lessons, next steps](05_results_and_lessons.md) | Everything in one place, and the plan for v1.1 / v2 |
| — | [Glossary](glossary.md) | Every term used above |

## v2 — mass from colour (an appearance → dynamics causal edge)

| # | document | one line |
|---|---|---|
| v2-00 | [Design](v2/00_v2_design.md) | The hypothesis, the world change, the planned experiments |
| v2-01 | [Environment, data, V](v2/01_v2_env_data_vae.md) | Colour encodes mass; speed becomes decodable from a single frame |
| v2-02 | [M and the causal tests](v2/02_v2_dynamics_and_causal_tests.md) | Cold-start speed from colour, repaint-and-redream, interpolation — against a colour-blind control |
| v2-03 | [C by mass](v2/03_v2_controller.md) | Uniform skill across a 4× speed range; the mass-blind v1 policy ties it; colour drifts in stochastic dreams |
| v2-04 | [Results and lessons](v2/04_v2_results_and_lessons.md) | v2 in one page, and the plan for v3 |

**See it live.** `python -m wm.live` opens a window with the real game (arrow
keys) next to what the VAE sees, what the RNN predicted this frame would look
like, and a free-running dream that drifts until you re-sync it. `A` hands
control to the dream-trained controller. `python -m wm.live --record out.gif
--autopilot` produces the same thing as a GIF without pygame; an example is
`runs/live_demo_autopilot.gif`.

Technical run logs with exact commands and every number, written by the
implementing agents and reviewed: [`../wm/README_M.md`](../wm/README_M.md),
[`../wm/README_C.md`](../wm/README_C.md); for v2: [`../wm/README_V2.md`](../wm/README_V2.md), [`../wm/README_M2.md`](../wm/README_M2.md), [`../wm/README_C2.md`](../wm/README_C2.md). The environment's own notes:
[`../worldsim/worldsim.md`](../worldsim/worldsim.md).

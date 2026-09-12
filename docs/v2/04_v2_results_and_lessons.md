# v2 — 04: Results and lessons

*The v2 tier in one page.*

---

## 1. What v2 asked

v1 showed a world model can *track* hidden state (velocity in `h`). v2 asked
whether it can learn a **causal link from appearance to dynamics**: the ball's
colour encodes its mass, mass sets its speed (`0.022 / m`) and how much the
paddle deflects it, and a band of masses was hidden from training to test
interpolation. Same V-M-C recipe, same tooling, new questions.

## 2. What came out, stage by stage

| stage | question | answer |
|---|---|---|
| **V** | Is colour in `z`? | Yes: log-mass R² 0.97 (poly-2), distributed across dimensions, no single "colour axis". Reconstructions keep the colour (6% median mass error). |
| V | Anything new? | **Speed is decodable from a single frame (R² 0.98) while velocity components are not.** Colour determines the magnitude of motion; "decodable" and "visible" have come apart. |
| V | Interpolates? | Coarsely: unseen colours placed within ±0.1 mass on the 0.5–2 scale, but not resolved within the band. |
| **M** | Does it read speed off colour? | Yes. From **one frame**, dreamed speed correlates with true speed at r = 0.56 vs 0.12 for a colour-blind control trained on the same frames. The advantage is gone by 8 frames of warm-up: colour is a prior, used when motion evidence is scarce. |
| M | Does colour *cause* dreamed speed? | Yes. **Repaint the ball, re-dream**: slope of log speed vs log mass −0.60 (law −1.00) vs +0.01 for the control. Repainted colour persists through the dream. |
| M | Interpolates? | Yes: the repaint at the held-out mass sits closest to the law; held-out dream horizon 30 frames vs 26 in-distribution. |
| M | Cost of the extra factor? | v1 diagnostics 10–25% worse (horizon 26 vs 35), partly instrument. Cold-start dreams under-move by 2×. |
| **C** (first pass) | Does the agent cope with a 4× speed range? | Partly: 76 / 83 / 78% of oracle, dream-trained on zero real frames; a mass-blind v1 controller tied it. |
| C | Why not better? | **At τ = 1 the dream did not conserve the ball's colour**: dreamed mass uncorrelated with the truth within 25 steps (τ = 0 holds it for 200). The controller trained in dreams whose speed law drifted under it. |
| **M, fixed** | Can the drift be removed? | Mostly: a conservation penalty on a frozen mass probe during 8-step open-loop rollouts (plus a log-mass head) turns the random walk into a flat, bounded error (corr ≈ 0.45 for 200 steps vs decay to −0.4), with a *better* horizon (29 vs 26). Training on posterior means was much worse. |
| **C, retrained** | In the fixed dream? | **Oracle level**: 0.94–1.00 interceptions per visit vs oracle 0.99, in every mass tercile, on held-out colours, on zero real frames; confirmed on fresh seeds. Temperature alone did nothing; the model was the factor. |
| C | Does it *use* mass? | **Yes.** Repainting the ball in a real history and re-deciding shifts the controller's drive by 0.3 sd, flips 14% of actions, in the direction the physics predicts (all coefficients negative, most significant); null control exactly zero; oracle unaffected. |

## 3. The six lessons of v2

1. **A paired control is the whole argument.** "M uses colour" was only
   meaningful against `rnn_v2_nocolor`, trained on identical frames encoded by a
   colour-blind VAE. A redundant cue can be argued either way without it.
2. **Correlational and interventional tests are different tests.** Cold-start
   scatter (correlation) and repaint-and-redream (intervention) agreed here;
   the regression test for the controller (correlation) was confounded and
   would have needed an intervention to fix. Prefer interventions whenever the
   world model lets you make them — and a world model is precisely the thing
   that lets you make them cheaply.
3. **Constants diffuse in stochastic dreams.** Any latent factor without a
   restoring force random-walks under τ > 0 sampling. Test long dreams for the
   constancy of things that should be constant before training anything inside
   them. This was found by eye in a GIF, after the fact; it should have been a
   metric from the start.
4. **Instruments degrade with the model.** kNN probes failed on the new factor
   for geometric reasons; position probes got worse because the latent metric
   was partly spent on colour. Always report the probe's floor next to the
   model's number, and never compare likelihoods across latent spaces.
5. **The value of a causal cue depends on the task's tolerance — but so does
   the quality of the dream.** With a drifting dream the mass-aware controller
   barely beat a mass-blind one; with a conserving dream it reached the oracle
   on the fast balls where the mass-blind one loses. Before concluding a cue
   does not matter, make sure the dream presents it consistently.
6. **Fix the model, not the temperature.** A 2 × 2 over {original, fixed
   model} × {τ = 1, 0.5} showed the model was the whole effect. Cheap knobs are
   worth trying, but only inside a design that can tell them apart.

## 4. Numbers to remember

| | v1 | v2 |
|---|---|---|
| VAE active units / KL | 9 / 14.4 nats | 8 / 15.5 nats |
| speed from one frame (poly-2 R²) | — | 0.98 |
| useful dream horizon (τ = 0) | 35 | 26 |
| cold-start speed r, colour-seeing vs colour-blind | — | 0.56 vs 0.12 |
| repaint slope (law −1), colour-seeing vs colour-blind | — | −0.60 vs +0.01 |
| controller vs oracle, interceptions per visit | ~0.9 | 0.79 (drifting dream) → **0.94–1.00** (fixed dream) |
| repaint intervention on the controller's drive, effect / flips | — | 0.3 sd / 14% of actions, correctly signed |
| colour conservation in a 200-step τ = 1 dream, corr | — | −0.4 (baseline) → 0.45 flat (fixed) |
| held-out colours | — | 0.79, same as in-distribution |
| real frames used to train the controller | 0 | 0 |

## 5. Next, in order of value per hour

1. ~~Fix colour drift and retrain C~~ — done, docs 05–06. Remaining: separate
   the conservation penalty from the log-mass head (train C in `rnn_v2_ms`),
   and a second training seed per cell.
2. ~~Interventional controller test~~ — done, doc 06.
3. **Make speed matter**: a task variant where a missed ball ends the episode
   (or a narrower paddle), so mass-awareness has a large payoff.
4. **Cold-start calibration**: why do one-frame dreams under-move by 2×?
5. **v3 — the occlusion band**: an opaque strip the ball passes behind. Object
   permanence, and the first test of whether `h` carries *position* through a
   gap in observation, not just velocity. The v2 tooling (probes by factor,
   counterfactual dreams, paired controls, per-opportunity grading) carries
   over.

## 6. Reproducing

Exact commands with wall-clock times: [`wm/README_V2.md`](../../wm/README_V2.md)
(environment, data, VAE), [`wm/README_M2.md`](../../wm/README_M2.md)
(dynamics and causal tests), [`wm/README_C2.md`](../../wm/README_C2.md)
(controller), [`wm/README_FIX.md`](../../wm/README_FIX.md) (colour drift fix), [`wm/README_C2_FIX.md`](../../wm/README_C2_FIX.md) (controller in the fixed dream, intervention test). Tests: `python -m pytest tests/ -q` (65). Live viewer for the v2
world: `python -m wm.live --v2`.

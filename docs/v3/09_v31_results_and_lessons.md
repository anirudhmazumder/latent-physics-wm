# v3.1 — 09: Results and lessons

*The occlusion tier, second attempt, in one page — and the state of the whole
project after three tiers.*

---

## 1. What v3.1 changed and why

v3's controller stage found that its band did not require memory: a memoryless
oracle caught 99% of balls. v3.1 fixed the design with an oracle sweep run
*before* any training: band lowered to (0.13, 0.63) and the paddle narrowed to
0.16, giving oracle 0.99, memoryless oracle 0.48–0.53, stand-still 0.38. The
VAE was trained on all three band heights so taller-band tests would not be
confounded by the encoder.

## 2. What came out

| stage | question | answer |
|---|---|---|
| **V** | One encoder for all bands? | Yes: reconstruction error within 1.7× across bands (v3: 262×); positions decode equally on all; band top decodable at R² 0.997 at no cost in KL. |
| **M** | Horizontal permanence? | **Yes, now.** Hidden-frame `x` from `h`: 0.17 (v3) → **0.52**, level with the privileged ceiling (0.48); still 0.40 after 24 hidden frames. The longer occlusion made the one-step loss pay for `x`. |
| M | Vertical permanence? | **Lost.** `vy` 0.67 → −0.32, frames-hidden 0.54 → −0.23. The model knows where the hidden ball is but not when it comes out — the inverse of v3. |
| M | Is the dream alive? | **No.** Balls re-emerge in 5 of 99 deterministic dreams; at τ ≥ 0.5 a ball appears below the band but parked and flickering, 4–6× too often. Three of four models paint balls only *above* the band, where the agent cannot act. |
| **C** | Does a dream-trained controller beat the memoryless bound? | **Yes: 0.73 vs 0.51** (fresh seeds 0.73 vs 0.53), non-overlapping intervals, flat across required moves (0.76 / 0.71 / 0.72) where the bound is 0.04 on long moves. It moves on 52% of hidden frames and covers 28% of the required move while blind (v3: 3%). |
| C | Is it `h`? | Yes. The matched `z`-only control (same V, M, τ, optimiser, `h` removed) scores exactly the bound, 0.51, and 0.06–0.11 on long moves. |
| C | Does the privileged ceiling help? | No (0.68–0.70 vs 0.73), and it collapses on the other bands. The fair model's `x`-memory had already caught up. |
| C | Does 1M real frames help? | No (0.65): a reaction policy, best on medium moves, worst on long. The dream's contribution is a *dense* per-step reward on a memory-carrying state. |
| C | What was the dream, then? | A feature-conditioned reward model, not a simulator: the reward head reads the true ball–paddle gap at R² 0.72 from `h`, and CMA-ES learned "move toward where `h` says the ball is" without ever seeing a dreamed ball come out of the band. |

## 3. The lessons of v3.1

1. **Measure the environment before training in it — including the dream.**
   v3 taught "measure the memoryless bound first"; v3.1 adds "measure whether
   the dream is alive first". Both cost minutes and both change what every
   later number means.
2. **"Object present" is not "object where the agent can act".** A dream can be
   75% balls and contain no reachable ball.
3. **A world model can be useful while being a bad world.** What transferred was
   a dense objective on a memory-carrying state, not a simulation of the
   occlusion. Say the weaker thing the evidence supports.
4. **The loss's payoff schedule picks which hidden variable gets learned.** The
   same architecture learned `x` and forgot the exit clock when the occlusion
   doubled. This is the single most important fact about teacher-forced
   world models that this project has found, and it appeared in v2 (colour
   drift), v3 (no `x`), and v3.1 (no clock) in three different guises.
5. **Argmax rollouts cannot cross rare events.** A deterministic dream never
   takes a step with per-step probability below one half, so it can never end
   a long occlusion. Pre-registered temperature selection by "closest arrival
   rate" then picked a dead dream over a lively one; skill turned out to be
   monotone in temperature the other way.
6. **Matched negative controls beat privileged ceilings.** `z`-only settled the
   attribution; the position-head ceiling has failed to be informative three
   times running.
7. **Metric validity has geometric conditions.** The exit detector that worked
   on a thin band lost its sample on a thick one; the long-move bin credits
   late arrival as well as early commitment. The displacement-while-blind
   statistic is the one that carries the behavioural claim.

## 4. Where the project stands after three tiers

| capability | demonstrated | how |
|---|---|---|
| tracking hidden state (velocity) | v1 | velocity in `h`, R² 0.92, from single-frame codes that contain none |
| an appearance → dynamics causal law | v2 | speed read off colour from one frame; repaint-and-redream changes dreamed speed with the right sign; agent's decisions shift under the intervention |
| conserving a constant in a stochastic dream | v2 (partial) | conservation penalty turns colour drift into bounded error |
| object permanence in the model | v3.1 (horizontal) | hidden-`x` from `h` 0.52, equal to a privileged ceiling; vertical lost |
| an agent acting on memory | v3.1 | +0.22 over the memoryless bound; moves while blind; `h`-ablation kills it |
| a dream that simulates an occlusion | **not yet** | balls do not re-emerge |

## 5. Next

1. **Revive the dream.** The missing exit clock is the cause; candidates are an
   emergence-timed auxiliary target (privileged, as a ceiling), a rollout loss
   scored on re-emergence frames specifically, or the v4 architecture change.
2. **Second seeds** for the fair / privileged / real rows; the 0.73 / 0.68 /
   0.65 ordering is within one seed's spread.
3. **A causal test that the controller reads hidden `x`**: perturb `h`'s
   `x`-direction (found by the probe) during occlusion and watch the drive.
4. **v4 — the gravity switch.** A latent flipped on paddle contact and never
   visible afterwards; the dependency length is the whole episode. This is where
   a transformer over the latent history should beat an LSTM that has to *carry*
   the bit — and the toolkit is now complete: oracle sweeps, dream-alive checks,
   conservation and permanence metrics, matched controls, per-run (V, M)
   pairing.

## 6. Reproducing

Run logs: [`wm/README_V31.md`](../../wm/README_V31.md) (environment, VAE,
dynamics, permanence), [`wm/README_C31.md`](../../wm/README_C31.md) (dream-alive
check, controllers, evaluation). Design sweep: `runs/v31_design/sweep.md`. Tests:
`python -m pytest tests/ -q` (145). Live viewer for the v3.1 world:
`python -m wm.live --occluder --occluder-y 0.13 0.63 --vae runs/vae_v31/vae.pt --rnn runs/rnn_v31/rnn.pt --ctrl runs/ctrl_v31_tau1/controller.pt`
(the paddle width is not yet a `live.py` flag; add `--paddle-w` if you want the
exact v3.1 board).

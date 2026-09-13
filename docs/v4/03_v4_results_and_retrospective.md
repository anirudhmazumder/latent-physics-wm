# v4 — 03: Results, lessons, and the project after four tiers

---

## 1. v4 in one page

| stage | question | answer |
|---|---|---|
| design | Does the gravity sign matter for play? | **No**, at any gravity or with a side-wind: a wrong sign shifts the landing point by ≤ 0.035 (≤ 0.13 with wind), the error vanishes near the deadline, and the paddle absorbs it. A closed form shows no acceleration is both binding and playable at this paddle speed. Controller stage dropped. |
| **V** | Is the sign in a frame? | No (0.36–0.40 vs a 0.50 null). Correct. |
| **M** | Is the sign in `h`? | Partly: the LSTM adds +0.28 over a single frame at 25–50 frames after a flip, the transformer +0.09, the feed-forward floor 0. |
| M | Memory or inference? | **Inference.** Recall is at chance in the first 10 frames after a flip and rises as the trajectory bends. The flip counterfactual is at chance for every model (0.25–0.50). |
| M | LSTM vs transformer? | Unresolved as posed: neither remembers. The transformer is the better predictor (NLL 1.78–1.87 vs 2.50; PR-AUC 0.63 vs 0.55); the LSTM reads curvature slightly better; the context-32 transformer that *cannot* see the flip is the best model. |

## 2. What the four tiers established, and what they did not

| capability | verdict | tier | evidence |
|---|---|---|---|
| hidden state inferable from two frames (velocity) | **yes** | v1 | velocity in `h` at R² 0.92 from codes that contain none; controller uses it |
| an appearance → dynamics law | **yes** | v2 | speed read off colour from one frame; repainting a dream changes its speed with the right sign; agent's decisions shift under the intervention |
| a constant conserved through a stochastic dream | partial | v2 | conservation penalty turns drift into bounded error |
| position carried through a gap (object permanence) | **partial** | v3 / v3.1 | vertical then horizontal permanence, never both; the dream never re-emerges the ball on the wide band |
| an agent acting on memory | **yes** | v3.1 | +0.22 over the memoryless bound on both seeds; the `h`-ablation sits on the bound |
| a bit set by a rare event and held indefinitely | **no** | v4 | flip counterfactual at chance for LSTM and transformer alike |
| a dream that *simulates* a hidden process | **no** | v3.1, v4 | balls do not re-emerge; dreamed curvature barely above chance |

Read as a whole: the V-M-C recipe with a one-step teacher-forced dynamics loss
learns whatever the next frame pays for — velocity (paid every frame), colour
→ speed (every frame), hidden height and the exit clock (paid at re-emergence,
~10 frames away), hidden x (paid when occlusions were long enough) — and does
not learn what the next frame does not pay for: a bit whose per-frame effect is
1e-4, an exit clock 21 frames away, a constant that nothing restores. **The
objective, not the architecture, set the ceiling in every tier where a ceiling
was hit**, and the privileged-head experiments proved it each time (the model
could hold the state when told to).

## 3. The lessons that survived every tier

1. **Measure before training.** The memoryless oracle (v3), the oracle sweep
   (v3.1, v4), and the dream-alive check (v3.1) each cost minutes and each
   changed what every later number meant. Twice they showed the tier's
   premise was wrong before a model existed.
2. **Pair every claim with a matched control.** The colour-blind twin (v2),
   the feed-forward floor and privileged ceiling (v3), the `h`-ablation (v3.1),
   the context-32 transformer (v4). The controls settled more than the main
   runs did.
3. **Intervene; don't regress.** Repaint-and-redream (v2), repaint the ball for
   the controller (v2), re-simulate a miss (v4). The regression tests for
   "does it use X" gave false positives twice.
4. **The loss's payoff schedule picks what gets learned.** Colour drift, no
   `x`, no clock, no flip — four faces of one fact.
5. **Report the fraction scored, the seed spread, and the null next to the
   number.** Censoring (v3.1), one-seed orderings (v3.1), and a non-flat null
   (v4) would each have misled without them.
6. **A world model can be useful while being a bad world.** The v3.1
   controller learned from a dead dream because the reward head read a
   memory-carrying state. Say the weaker thing the evidence supports.

## 4. What would actually move the remaining "no"s

- **An objective with long-range credit.** Multi-step losses helped once (v2)
  and hurt twice (v3, v3.1); the fix for a rare, diffuse bit is more likely a
  *predictive-coding* or *contrastive* target on far-future latents, or an
  auxiliary that pays for the event's consequences directly.
- **A dream that re-emerges hidden objects.** The counter head learned the
  past and not the future (v3.1); attention did not help (v4). A latent
  designed with an explicit object slot, or a decoder-side detector-driven
  loss on re-emergence frames, are the untried options.
- **A second seed everywhere.** The one-seed orderings that dissolved in
  v3.1 are a warning for every other table in these documents.

## 5. Reproducing

Logs: [`wm/README_M4.md`](../../wm/README_M4.md); sweeps
`runs/v4_design/sweep.md`, `sweep_v41.md`. Tests: `python -m pytest tests/ -q`
(206). The transformer: `python -m wm.train_rnn --arch transformer --context 128 ...`.

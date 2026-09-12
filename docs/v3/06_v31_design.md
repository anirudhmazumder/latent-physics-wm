# v3.1 — 06: Design. Making memory matter

*Fixing the design flaw that v3's controller stage exposed, with the cheapest
possible experiment run first this time.*

---

## 1. What went wrong in v3, in one sentence

The default band was placed by an argument about frames that ignored the
paddle's width and the ball's *partial* visibility, and a two-line memoryless
oracle then caught 99% of balls — so v3 never tested whether an agent uses
object permanence, because the task did not require it.

## 2. The rule this time: sweep with oracles before training anything

Before collecting a single frame of v3.1 data, two privileged policies were run
across candidate environments: the full oracle (tracks the true ball always)
and the **wait-and-see oracle** (tracks it only while at least half visible,
otherwise stands still). The gap between them is, by construction, the entire
value of object permanence for play in that environment. Sixty episodes per
cell, a minute each; full table in `runs/v31_design/sweep.md`.

| band (bottom, top) | paddle width | oracle | memoryless oracle | stay |
|---|---|---|---|---|
| (0.28, 0.58) — the v3 default | 0.26 | 0.99 | **1.00** | ~0.5 |
| (0.20, 0.50) | 0.26 | 0.99 | 0.83 | |
| (0.16, 0.64) | 0.16 | 0.99 | 0.61 | 0.38 |
| **(0.13, 0.63)** | **0.16** | **0.99** | **0.48** | **0.38** |
| (0.13, 0.63) | 0.12 | 0.99 | 0.43 | 0.33 |

Two things the sweep taught that the v3 design argument missed:

- **The band's bottom edge is what matters, not its height.** The memoryless
  policy pre-positions under the ball while it is still visible *above* the
  band; only the ball's drift during the hidden stretch can defeat it. Lowering
  the bottom edge to 0.13 (the ball re-emerges essentially at contact height)
  removed its last visible frames before the floor. Raising the top edge beyond
  ~0.5 did nothing further: the memoryless floor is bounded below by chance,
  which is set by the paddle's width.
- **Paddle width sets the chance floor.** At 0.26, standing still catches half
  the balls; at 0.16, 0.38.

## 3. The v3.1 environment

| field | v3 | **v3.1** |
|---|---|---|
| `occluder_y` | (0.28, 0.58) | **(0.13, 0.63)** |
| `paddle_w` | 0.26 | **0.16** |
| everything else | v1 physics | unchanged |

The ball is now fully hidden for roughly 23 frames per crossing (band height
0.50 minus the ball's 0.16, over a typical vertical speed), about a third of
hidden runs contain a side-wall bounce that happens out of sight, and the ball
re-emerges at the moment it can touch the paddle. An agent that catches more
than 48% of balls per floor visit is using memory.

Two further bands are collected for the memory-horizon test, **and this time
the VAE is trained on all three**, so that taller-band results are not
confounded by an out-of-distribution encoder as v3's were:

| band | purpose |
|---|---|
| (0.13, 0.63) | the default; all training and main evaluation |
| (0.13, 0.45) | shorter occlusion |
| (0.13, 0.78) | longer occlusion |

The band's top edge therefore varies across the training data, which is one
more thing the frame code must represent (three discrete values). The band
bottom, and everything else, is fixed.

## 4. What is re-asked, and what is new

Everything from v3's design (doc 00 §3–4) is re-run on the new environment:
V's by-visibility probes; M's hidden-frame position from `h`, emergence
prediction, hidden wall bounces, memory horizon — now with a 23-frame default
occlusion that the v3 model's ~13-frame crossover would not survive; the
feed-forward floor and the privileged position-head ceiling; and C's
by-required-move table, paddle motion during occlusion, and taller-band
generalisation, with the memoryless oracle now at 0.48 rather than 0.99.

New questions v3.1 can ask that v3 could not:

- **Does horizontal permanence buy play?** The privileged-ceiling controller
  versus the fair one, in a world where the memoryless bound is 0.48. If the
  ceiling controller is not clearly above the fair one here, permanence is not
  what limits a linear controller even when the task demands it.
- **Does the fair fix (`emerge`) buy play?** It doubled hidden-`x` information
  in `h` without improving the dream's exit; the controller is a different
  consumer of `h` than the decoder, and may be able to use it.
- **How does skill fall off with occlusion length** when the encoder is *not*
  the confound?

## 5. What would count as success

| claim | evidence |
|---|---|
| the environment tests memory | memoryless oracle ≈ 0.5, oracle ≈ 1.0 (already established) |
| M carries the ball | hidden-frame `x` R² from `h` well above the feed-forward floor over 20+ hidden frames; dreamed exit x below the no-memory baseline |
| permanence is worth something for play | the privileged-ceiling controller clearly above the fair one, which is clearly above the feed-forward floor, all on the same 150 episodes |
| the fair model gets some of it | fair controller above the memoryless oracle's 0.48 |

Failure modes to expect: the fair model's `x`-memory decays before the 23-frame
crossing (v3's crossover was ~13 frames), so the fair controller sits at or
below the memoryless bound; the ceiling controller fails to exploit `x` even
though its dream carries it, because 819 linear parameters cannot turn a
remembered position into a timed sweep; and the varying band top costs the VAE
positional precision.

Next: [07 — v3.1: environment, V, M](07_v31_env_vae_dynamics.md).

# stage three (C) — real-environment evaluation by occlusion

150 episodes x 200 steps, seeds 5000..5149, identical starts for every row. Band (0.13, 0.63), paddle width 0.16.

`oracle` has perfect vision and perfect memory; **`wait_and_see` has perfect vision and NO memory** (it tracks the true ball only while `ball_visible > 0.5` and STAYs otherwise). Every trained row should be read as a position between those two, and the single most important fact about any row is which side of `wait_and_see` it falls on: **above it is memory, below it is not**. Any row whose name contains `poshead` is a **PRIVILEGED CEILING** — its dynamics model was trained with a supervised head on the simulator's true ball position — and is not a world-model result.

**The two numbers everything is graded against, measured here on these 150 episodes: oracle 0.99, memoryless (wait-and-see) bound **0.51**, stand still 0.38.**

Each controller is driven by the (V, M) it was trained with; the policy reads `h`, so M is part of the policy:

| controller | M (dynamics) | V (encoder) |
|---|---|---|
| `ctrl_v31` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_allbands` | `runs/rnn_v31_allbands/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_ff` | `runs/rnn_v31_ff/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_ff_s1` | `runs/rnn_v31_ff/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_poshead` | `runs/rnn_v31_poshead/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_poshead_s1` | `runs/rnn_v31_poshead/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_real` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_real_s1` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_s1` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_tau0.5` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_tau1` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_tau1_s1` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_z_only` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_z_only_tau1` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |
| `ctrl_v31_z_only_tau1_s1` | `runs/rnn_v31/rnn.pt` | `runs/vae_v31/vae.pt` |

## The headline: every row against the memoryless bound

`wait_and_see` = **0.51** is the best a policy with perfect vision and no memory can do in this world. A row above it is using memory; a row below it is not, whatever else it is doing well.

| controller | interceptions/visit | vs the bound | above it? |
|---|---|---|---|
| `oracle` | 0.99 [0.98, 1.00] | +0.48 | **yes** |
| `ctrl_v31_tau1_s1` | 0.97 [0.87, 1.07] | +0.46 | **yes** |
| `ctrl_v31_real_s1` | 0.75 [0.69, 0.82] | +0.24 | **yes** |
| `ctrl_v31_tau1` | 0.73 [0.66, 0.80] | +0.22 | **yes** |
| `ctrl_v31_poshead` | 0.68 [0.60, 0.75] | +0.17 | **yes** |
| `ctrl_v31_poshead_s1` | 0.65 [0.57, 0.74] | +0.14 | **yes** |
| `ctrl_v31_real` | 0.65 [0.58, 0.71] | +0.14 | **yes** |
| `ctrl_v31_ff_s1` | 0.63 [0.57, 0.69] | +0.12 | **yes** |
| `ctrl_v31_tau0.5` | 0.61 [0.55, 0.68] | +0.10 | **yes** |
| `ctrl_v31` | 0.57 [0.51, 0.63] | +0.06 | **yes** |
| `ctrl_v31_z_only_tau1_s1` | 0.53 [0.46, 0.60] | +0.02 | **yes** |
| `ctrl_v31_z_only` | 0.51 [0.45, 0.58] | +0.00 | **yes** |
| `wait_and_see` | 0.51 [0.44, 0.58] | +0.00 | no |
| `ctrl_v31_z_only_tau1` | 0.51 [0.44, 0.57] | -0.00 | no |
| `ctrl_v31_s1` | 0.50 [0.43, 0.57] | -0.01 | no |
| `ctrl_v31_allbands` | 0.47 [0.41, 0.53] | -0.04 | no |
| `ctrl_v31_ff` | 0.46 [0.40, 0.53] | -0.05 | no |
| `stay` | 0.38 [0.33, 0.44] | -0.13 | no |
| `random` | 0.31 [0.25, 0.37] | -0.20 | no |

The CI is a 4000-sample bootstrap over EPISODES, the unit of independence; "above the bound" is read off the point estimate, so a row whose interval straddles the bound is not a claim.


## Interceptions per floor visit, by required move

The **required move** is the distance between the paddle's x at the moment the ball became fully hidden on its way down and the ball's landing x. Flat across the bins = object permanence; collapsing on `long` = memoryless.

| controller | overall | short | medium | long |
|---|---|---|---|---|
| `stay` | 0.38 [0.33, 0.44] | 0.99 [0.96, 1.00] | 0.13 [0.06, 0.21] | 0.00 [0.00, 0.00] |
| `random` | 0.31 [0.25, 0.37] | 0.54 [0.43, 0.65] | 0.30 [0.20, 0.42] | 0.18 [0.11, 0.26] |
| `oracle` | 0.99 [0.98, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.97 [0.93, 1.00] |
| `wait_and_see` | 0.51 [0.44, 0.58] | 1.00 [1.00, 1.00] | 0.57 [0.47, 0.68] | 0.04 [0.00, 0.09] |
| `ctrl_v31` | 0.57 [0.51, 0.63] | 0.67 [0.57, 0.76] | 0.56 [0.46, 0.65] | 0.45 [0.33, 0.58] |
| `ctrl_v31_s1` | 0.50 [0.43, 0.57] | 0.72 [0.60, 0.85] | 0.52 [0.42, 0.63] | 0.18 [0.07, 0.31] |
| `ctrl_v31_tau1` | 0.73 [0.66, 0.80] | 0.76 [0.67, 0.86] | 0.71 [0.62, 0.79] | 0.72 [0.53, 0.92] |
| `ctrl_v31_tau1_s1` | 0.97 [0.87, 1.07] | 1.10 [0.99, 1.22] | 1.00 [0.83, 1.18] | 0.66 [0.43, 0.91] |
| `ctrl_v31_ff` | 0.46 [0.40, 0.53] | 0.85 [0.77, 0.91] | 0.27 [0.19, 0.37] | 0.20 [0.10, 0.31] |
| `ctrl_v31_ff_s1` | 0.63 [0.57, 0.69] | 0.88 [0.81, 0.94] | 0.54 [0.44, 0.64] | 0.38 [0.25, 0.52] |
| `ctrl_v31_poshead` | 0.68 [0.60, 0.75] | 0.76 [0.66, 0.86] | 0.59 [0.48, 0.69] | 0.71 [0.53, 0.90] |
| `ctrl_v31_poshead_s1` | 0.65 [0.57, 0.74] | 0.89 [0.78, 1.03] | 0.59 [0.48, 0.69] | 0.49 [0.32, 0.68] |
| `ctrl_v31_z_only_tau1` | 0.51 [0.44, 0.57] | 0.93 [0.88, 0.98] | 0.41 [0.30, 0.52] | 0.06 [0.00, 0.13] |
| `ctrl_v31_z_only_tau1_s1` | 0.53 [0.46, 0.60] | 0.94 [0.87, 1.00] | 0.41 [0.30, 0.53] | 0.25 [0.14, 0.38] |
| `ctrl_v31_real` | 0.65 [0.58, 0.71] | 0.86 [0.78, 0.94] | 0.80 [0.70, 0.89] | 0.37 [0.26, 0.50] |
| `ctrl_v31_real_s1` | 0.75 [0.69, 0.82] | 0.93 [0.85, 1.00] | 0.77 [0.68, 0.87] | 0.51 [0.36, 0.68] |
| `ctrl_v31_tau0.5` | 0.61 [0.55, 0.68] | 0.91 [0.85, 0.98] | 0.50 [0.40, 0.61] | 0.37 [0.22, 0.54] |
| `ctrl_v31_allbands` | 0.47 [0.41, 0.53] | 0.75 [0.65, 0.84] | 0.33 [0.24, 0.43] | 0.28 [0.17, 0.39] |
| `ctrl_v31_z_only` | 0.51 [0.45, 0.58] | 0.92 [0.86, 0.97] | 0.45 [0.33, 0.58] | 0.11 [0.04, 0.19] |

**The bins are policy-dependent and that is not a bug to hide.** The required move is measured from *this policy's own* paddle position, so a policy that already tends to stand near the ball generates few long visits and a policy that parks in a corner generates many. The counts are therefore reported per row, and a row with a handful of long visits should be read as noise. The episodes and seeds are identical throughout; only the paddle differs.

| controller | visits (short / medium / long) | mean required move (short / medium / long) | visits with no occlusion on the descent |
|---|---|---|---|
| `stay` | 91 / 91 / 82 | 0.069 / 0.239 / 0.496 | 2 |
| `random` | 68 / 79 / 114 | 0.084 / 0.242 / 0.541 | 2 |
| `oracle` | 72 / 102 / 76 | 0.075 / 0.234 / 0.486 | 2 |
| `wait_and_see` | 63 / 112 / 78 | 0.072 / 0.244 / 0.500 | 2 |
| `ctrl_v31` | 90 / 107 / 64 | 0.075 / 0.236 / 0.459 | 2 |
| `ctrl_v31_s1` | 79 / 113 / 62 | 0.068 / 0.244 / 0.455 | 2 |
| `ctrl_v31_tau1` | 89 / 107 / 54 | 0.071 / 0.237 / 0.437 | 2 |
| `ctrl_v31_tau1_s1` | 108 / 95 / 53 | 0.070 / 0.244 / 0.458 | 2 |
| `ctrl_v31_ff` | 97 / 103 / 61 | 0.076 / 0.251 / 0.427 | 2 |
| `ctrl_v31_ff_s1` | 94 / 113 / 55 | 0.079 / 0.254 / 0.430 | 2 |
| `ctrl_v31_poshead` | 83 / 94 / 84 | 0.084 / 0.243 / 0.509 | 2 |
| `ctrl_v31_poshead_s1` | 84 / 87 / 80 | 0.071 / 0.248 / 0.483 | 2 |
| `ctrl_v31_z_only_tau1` | 88 / 123 / 54 | 0.066 / 0.243 / 0.471 | 2 |
| `ctrl_v31_z_only_tau1_s1` | 79 / 112 / 67 | 0.073 / 0.240 / 0.459 | 2 |
| `ctrl_v31_real` | 80 / 80 / 102 | 0.068 / 0.247 / 0.532 | 2 |
| `ctrl_v31_real_s1` | 81 / 109 / 70 | 0.073 / 0.244 / 0.454 | 2 |
| `ctrl_v31_tau0.5` | 91 / 101 / 63 | 0.075 / 0.242 / 0.471 | 2 |
| `ctrl_v31_allbands` | 95 / 97 / 69 | 0.076 / 0.251 / 0.490 | 2 |
| `ctrl_v31_z_only` | 89 / 105 / 71 | 0.073 / 0.237 / 0.485 | 2 |

## The reach curve (interceptions per visit at 0.1 resolution)

A required move is covered once the paddle's centre is within half a paddle-width (0.08) of the landing x, plus whatever the visible frames between the band's bottom edge and contact height buy. On v3's geometry that came to ~0.43 and `wait_and_see` was at the ceiling out to 0.6; v3.1 halves both terms, which is the whole point of the re-design. This table says where `wait_and_see` actually falls off, and therefore which bins above carry any information about object permanence.

| controller | <0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | >0.6 |
|---|---|---|---|---|---|---|---|
| `stay` | 1.00 | 0.67 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| `random` | 0.58 | 0.43 | 0.25 | 0.26 | 0.20 | 0.04 | 0.25 |
| `oracle` | 1.00 | 1.00 | 1.00 | 0.97 | 0.96 | 1.00 | 1.00 |
| `wait_and_see` | 1.00 | 0.96 | 0.50 | 0.24 | 0.00 | 0.00 | 0.00 |
| `ctrl_v31` | 0.67 | 0.67 | 0.50 | 0.57 | 0.41 | 0.22 | 1.00 |
| `ctrl_v31_s1` | 0.78 | 0.60 | 0.50 | 0.35 | 0.23 | 0.00 | 0.00 |
| `ctrl_v31_tau1` | 0.82 | 0.72 | 0.65 | 0.64 | 0.69 | 1.17 | 1.50 |
| `ctrl_v31_tau1_s1` | 1.08 | 1.19 | 0.96 | 0.68 | 0.65 | 0.79 | 2.00 |
| `ctrl_v31_ff` | 0.82 | 0.73 | 0.20 | 0.19 | 0.16 | 0.00 | 0.00 |
| `ctrl_v31_ff_s1` | 0.90 | 0.75 | 0.60 | 0.49 | 0.22 | 0.33 | 0.00 |
| `ctrl_v31_poshead` | 0.81 | 0.64 | 0.74 | 0.36 | 0.74 | 0.79 | 0.79 |
| `ctrl_v31_poshead_s1` | 0.84 | 0.82 | 0.62 | 0.44 | 0.44 | 0.40 | 0.86 |
| `ctrl_v31_z_only_tau1` | 0.96 | 0.69 | 0.39 | 0.21 | 0.05 | 0.00 | 0.00 |
| `ctrl_v31_z_only_tau1_s1` | 1.00 | 0.70 | 0.34 | 0.34 | 0.14 | 0.23 | 0.00 |
| `ctrl_v31_real` | 0.87 | 0.82 | 0.79 | 0.73 | 0.52 | 0.16 | 0.27 |
| `ctrl_v31_real_s1` | 1.00 | 0.88 | 0.82 | 0.51 | 0.41 | 0.85 | 0.00 |
| `ctrl_v31_tau0.5` | 0.95 | 0.71 | 0.45 | 0.43 | 0.21 | 0.69 | 0.25 |
| `ctrl_v31_allbands` | 0.80 | 0.61 | 0.32 | 0.26 | 0.48 | 0.10 | 0.00 |
| `ctrl_v31_z_only` | 0.94 | 0.80 | 0.35 | 0.19 | 0.08 | 0.12 | 0.00 |

| controller | visits per slice |
|---|---|
| `stay` | 62 / 58 / 44 / 38 / 28 / 18 / 16 |
| `random` | 38 / 58 / 32 / 43 / 30 / 24 / 36 |
| `oracle` | 48 / 64 / 39 / 40 / 26 / 22 / 11 |
| `wait_and_see` | 48 / 46 / 58 / 37 / 36 / 11 / 17 |
| `ctrl_v31` | 61 / 60 / 58 / 37 / 22 / 18 / 5 |
| `ctrl_v31_s1` | 58 / 52 / 58 / 43 / 30 / 8 / 5 |
| `ctrl_v31_tau1` | 61 / 65 / 48 / 42 / 26 / 6 / 2 |
| `ctrl_v31_tau1_s1` | 79 / 54 / 51 / 34 / 23 / 14 / 1 |
| `ctrl_v31_ff` | 66 / 60 / 44 / 54 / 31 / 5 / 1 |
| `ctrl_v31_ff_s1` | 61 / 53 / 63 / 49 / 32 / 3 / 1 |
| `ctrl_v31_poshead` | 47 / 66 / 43 / 36 / 31 / 19 / 19 |
| `ctrl_v31_poshead_s1` | 58 / 44 / 52 / 36 / 32 / 15 / 14 |
| `ctrl_v31_z_only_tau1` | 67 / 51 / 72 / 34 / 21 / 19 / 1 |
| `ctrl_v31_z_only_tau1_s1` | 56 / 54 / 62 / 41 / 22 / 22 / 1 |
| `ctrl_v31_real` | 61 / 40 / 43 / 30 / 33 / 25 / 30 |
| `ctrl_v31_real_s1` | 53 / 59 / 50 / 53 / 27 / 13 / 5 |
| `ctrl_v31_tau0.5` | 64 / 58 / 49 / 35 / 29 / 16 / 4 |
| `ctrl_v31_allbands` | 64 / 51 / 53 / 39 / 23 / 21 / 10 |
| `ctrl_v31_z_only` | 63 / 60 / 54 / 32 / 25 / 25 / 6 |

## Paddle motion during occlusion

Over the descending hidden runs: the fraction of hidden frames whose ACTION moved the paddle toward the eventual landing x (chance **1/3** — three actions, STAY counts as not-toward), and the displacement the paddle actually achieved toward the landing point while blind, as a fraction of the required move.

`oracle` is **not** the ceiling for these two columns and should not be read as one: it chases the ball's CURRENT x rather than its landing x, and it is usually already inside its dead zone when the ball vanishes, so it STAYs. The references that mean something here are chance (1/3) and `wait_and_see` (0.000 by construction).

| controller | hidden runs | mean hidden frames | mean required move | moved on (frac. of hidden frames) | toward / all hidden frames (chance 1/3) | toward / moving frames (chance 1/2) | displacement fraction |
|---|---|---|---|---|---|---|---|
| `stay` | 264 | 21.8 | 0.260 | 0.000 | 0.000 [0.000, 0.000] | -- [--, --] | 0.000 [0.000, 0.000] |
| `random` | 261 | 21.7 | 0.331 | 0.671 | 0.278 [0.234, 0.322] | 0.414 [0.355, 0.476] | 0.202 [0.125, 0.284] |
| `oracle` | 250 | 22.3 | 0.265 | 0.509 | 0.456 [0.434, 0.480] | 0.897 [0.868, 0.926] | 0.793 [0.747, 0.843] |
| `wait_and_see` | 253 | 21.9 | 0.280 | 0.000 | 0.000 [0.000, 0.000] | -- [--, --] | 0.000 [0.000, 0.000] |
| `ctrl_v31` | 261 | 21.6 | 0.235 | 0.519 | 0.326 [0.306, 0.347] | 0.628 [0.593, 0.662] | 0.280 [0.209, 0.351] |
| `ctrl_v31_s1` | 254 | 21.8 | 0.241 | 0.559 | 0.336 [0.314, 0.359] | 0.602 [0.567, 0.637] | 0.235 [0.162, 0.312] |
| `ctrl_v31_tau1` | 250 | 21.6 | 0.221 | 0.283 | 0.193 [0.158, 0.227] | 0.682 [0.631, 0.731] | 0.215 [0.151, 0.279] |
| `ctrl_v31_tau1_s1` | 256 | 21.4 | 0.215 | 0.352 | 0.200 [0.170, 0.234] | 0.569 [0.526, 0.614] | 0.098 [0.039, 0.163] |
| `ctrl_v31_ff` | 261 | 21.7 | 0.227 | 0.109 | 0.086 [0.063, 0.111] | 0.791 [0.704, 0.864] | 0.125 [0.078, 0.172] |
| `ctrl_v31_ff_s1` | 262 | 21.6 | 0.228 | 0.256 | 0.150 [0.121, 0.178] | 0.584 [0.549, 0.624] | 0.088 [0.053, 0.124] |
| `ctrl_v31_poshead` | 261 | 21.6 | 0.278 | 0.508 | 0.338 [0.315, 0.360] | 0.665 [0.621, 0.708] | 0.296 [0.216, 0.375] |
| `ctrl_v31_poshead_s1` | 251 | 21.8 | 0.264 | 0.573 | 0.360 [0.339, 0.381] | 0.629 [0.587, 0.674] | 0.268 [0.184, 0.357] |
| `ctrl_v31_z_only_tau1` | 265 | 22.0 | 0.231 | 0.091 | 0.046 [0.027, 0.068] | 0.513 [0.438, 0.579] | 0.018 [0.005, 0.033] |
| `ctrl_v31_z_only_tau1_s1` | 258 | 21.7 | 0.246 | 0.092 | 0.042 [0.025, 0.062] | 0.457 [0.325, 0.598] | 0.025 [0.012, 0.040] |
| `ctrl_v31_real` | 262 | 21.4 | 0.304 | 0.460 | 0.332 [0.304, 0.362] | 0.722 [0.684, 0.761] | 0.337 [0.294, 0.380] |
| `ctrl_v31_real_s1` | 260 | 21.2 | 0.247 | 0.258 | 0.164 [0.135, 0.196] | 0.636 [0.578, 0.698] | 0.168 [0.124, 0.216] |
| `ctrl_v31_tau0.5` | 255 | 21.6 | 0.239 | 0.113 | 0.075 [0.061, 0.091] | 0.663 [0.591, 0.734] | 0.074 [0.040, 0.110] |
| `ctrl_v31_allbands` | 261 | 21.6 | 0.251 | 0.225 | 0.165 [0.132, 0.201] | 0.730 [0.656, 0.801] | 0.195 [0.126, 0.268] |
| `ctrl_v31_z_only` | 265 | 21.7 | 0.248 | 0.149 | 0.092 [0.058, 0.130] | 0.617 [0.561, 0.679] | 0.072 [0.038, 0.110] |

## Overall

| controller | interceptions/visit | interceptions/ep | floor visits/ep | gap at floor | approach sign agreement | left/stay/right |
|---|---|---|---|---|---|---|
| `stay` | 0.38 [0.33, 0.44] | 0.68 | 1.77 | 0.261 | -- | 0.00/1.00/0.00 |
| `random` | 0.31 [0.25, 0.37] | 0.55 | 1.75 | 0.327 | 0.337 | 0.35/0.32/0.33 |
| `oracle` | 0.99 [0.98, 1.00] | 1.67 | 1.68 | 0.039 | 1.000 | 0.26/0.48/0.26 |
| `wait_and_see` | 0.51 [0.44, 0.58] | 0.87 | 1.70 | 0.222 | 1.000 | 0.11/0.77/0.12 |
| `ctrl_v31` | 0.57 [0.51, 0.63] | 1.00 | 1.75 | 0.184 | 0.492 | 0.30/0.41/0.30 |
| `ctrl_v31_s1` | 0.50 [0.43, 0.57] | 0.85 | 1.71 | 0.215 | 0.486 | 0.32/0.38/0.31 |
| `ctrl_v31_tau1` | 0.73 [0.66, 0.80] | 1.23 | 1.68 | 0.149 | 0.575 | 0.25/0.50/0.25 |
| `ctrl_v31_tau1_s1` | 0.97 [0.87, 1.07] | 1.67 | 1.72 | 0.149 | 0.551 | 0.26/0.48/0.25 |
| `ctrl_v31_ff` | 0.46 [0.40, 0.53] | 0.81 | 1.75 | 0.208 | 0.556 | 0.10/0.80/0.10 |
| `ctrl_v31_ff_s1` | 0.63 [0.57, 0.69] | 1.11 | 1.76 | 0.153 | 0.589 | 0.18/0.63/0.18 |
| `ctrl_v31_poshead` | 0.68 [0.60, 0.75] | 1.19 | 1.75 | 0.189 | 0.573 | 0.30/0.38/0.32 |
| `ctrl_v31_poshead_s1` | 0.65 [0.57, 0.74] | 1.10 | 1.69 | 0.190 | 0.544 | 0.32/0.32/0.35 |
| `ctrl_v31_z_only_tau1` | 0.51 [0.44, 0.57] | 0.90 | 1.78 | 0.209 | 0.597 | 0.11/0.81/0.08 |
| `ctrl_v31_z_only_tau1_s1` | 0.53 [0.46, 0.60] | 0.92 | 1.73 | 0.204 | 0.553 | 0.13/0.76/0.11 |
| `ctrl_v31_real` | 0.65 [0.58, 0.71] | 1.14 | 1.76 | 0.204 | 0.570 | 0.24/0.52/0.23 |
| `ctrl_v31_real_s1` | 0.75 [0.69, 0.82] | 1.31 | 1.75 | 0.164 | 0.598 | 0.18/0.63/0.19 |
| `ctrl_v31_tau0.5` | 0.61 [0.55, 0.68] | 1.05 | 1.71 | 0.174 | 0.516 | 0.23/0.55/0.23 |
| `ctrl_v31_allbands` | 0.47 [0.41, 0.53] | 0.82 | 1.75 | 0.208 | 0.442 | 0.20/0.61/0.18 |
| `ctrl_v31_z_only` | 0.51 [0.45, 0.58] | 0.91 | 1.78 | 0.212 | 0.474 | 0.15/0.72/0.13 |

## VAE reconstruction on each band

The frozen encoder+decoder on frames from datasets collected at each band height, so an other-band drop can be attributed to V or to C.

| dataset | band | recon MSE | mean hidden run (frames) |
|---|---|---|---|
| `data/v31/val` | [0.13, 0.63] | 0.00006 | 24.6 |

## Dream-vs-real transfer

Pearson r between the smoothed best-of-generation dream fitness and the periodic real interception count, per training run (`dream_vs_real_all.png`, and `dream_vs_real.png` in each run directory).

| run | r (smoothed) | r (raw) | best real int./ep | final dream fitness | dream env steps | real env steps | wall clock (s) |
|---|---|---|---|---|---|---|---|
| `ctrl_v31` | 0.62 | 0.58 | 1.33 | 165.3 | 15,360,000 | 196,800 | 742 |
| `ctrl_v31_s1` | -0.10 | -0.13 | 0.88 | 166.1 | 15,360,000 | 196,800 | 899 |
| `ctrl_v31_tau1` | -0.16 | -0.07 | 1.58 | 148.5 | 15,360,000 | 196,800 | 721 |
| `ctrl_v31_tau1_s1` | -0.31 | -0.30 | 2.25 | 147.5 | 15,360,000 | 196,800 | 891 |
| `ctrl_v31_ff` | -0.33 | -0.32 | 0.96 | 131.9 | 15,360,000 | 196,800 | 693 |
| `ctrl_v31_ff_s1` | -0.23 | -0.22 | 1.17 | 131.7 | 15,360,000 | 196,800 | 687 |
| `ctrl_v31_poshead` | 0.73 | 0.70 | 1.58 | 159.0 | 15,360,000 | 196,800 | 737 |
| `ctrl_v31_poshead_s1` | 0.52 | 0.43 | 1.29 | 159.4 | 15,360,000 | 196,800 | 749 |
| `ctrl_v31_z_only_tau1` | 0.35 | 0.31 | 0.96 | 136.2 | 15,360,000 | 196,800 | 708 |
| `ctrl_v31_z_only_tau1_s1` | 0.13 | 0.09 | 1.08 | 136.1 | 15,360,000 | 196,800 | 416 |
| `ctrl_v31_real` | 0.93 | 0.65 | 1.54 | 1.1 | 0 | 1,067,200 | 1997 |
| `ctrl_v31_real_s1` | 0.79 | 0.65 | 1.42 | 1.5 | 0 | 1,067,200 | 8695 |
| `ctrl_v31_tau0.5` | -0.47 | -0.41 | 1.29 | 152.6 | 15,360,000 | 196,800 | 727 |
| `ctrl_v31_allbands` | -0.27 | -0.26 | 1.08 | 143.4 | 15,360,000 | 196,800 | 722 |
| `ctrl_v31_z_only` | 0.45 | 0.38 | 1.12 | 161.6 | 15,360,000 | 196,800 | 682 |

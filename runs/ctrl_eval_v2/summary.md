# v2 stage three (C) — real-environment evaluation by mass

150 episodes x 200 steps, seeds 5000..5149, identical starts for every row. Masses log-uniform in [0.5, 2.0] **excluding the held-out band [0.85, 1.2]**; observed range [0.50, 1.99], tercile cuts at m = 0.70 and 1.41.

Light = fast (speed 0.022/m, up to 0.044/frame); heavy = slow. A light ball reaches the floor several times more often than a heavy one, so per-episode counts are not comparable across terciles and **interceptions per floor visit** is the metric to read.

## Interceptions per floor visit (the mass-fair skill metric)

| controller | overall | light | medium | heavy |
|---|---|---|---|---|
| `stay` | 0.45 [0.39, 0.51] | 0.42 [0.33, 0.51] | 0.47 [0.37, 0.56] | 0.50 [0.37, 0.63] |
| `random` | 0.45 [0.40, 0.50] | 0.43 [0.36, 0.50] | 0.43 [0.34, 0.51] | 0.54 [0.40, 0.67] |
| `oracle` | 0.99 [0.98, 1.01] | 0.99 [0.96, 1.02] | 0.99 [0.96, 1.00] | 1.00 [1.00, 1.00] |
| `ctrl_v2` | 0.79 [0.74, 0.84] | 0.76 [0.68, 0.85] | 0.83 [0.76, 0.91] | 0.78 [0.66, 0.88] |
| `ctrl_v2_lastdream` | 0.77 [0.72, 0.82] | 0.76 [0.68, 0.84] | 0.77 [0.67, 0.86] | 0.79 [0.69, 0.89] |
| `ctrl_v2_mix` | 0.62 [0.55, 0.68] | 0.51 [0.42, 0.60] | 0.66 [0.55, 0.78] | 0.84 [0.73, 0.94] |
| `ctrl_v2_z_only` | 0.57 [0.51, 0.64] | 0.57 [0.47, 0.66] | 0.62 [0.50, 0.74] | 0.51 [0.35, 0.67] |
| `ctrl_v2_real` | 0.64 [0.58, 0.69] | 0.66 [0.59, 0.74] | 0.63 [0.55, 0.72] | 0.57 [0.43, 0.71] |
| `ctrl_v1_on_v2` | 0.79 [0.74, 0.84] | 0.71 [0.64, 0.78] | 0.87 [0.77, 0.95] | 0.90 [0.79, 1.02] |

## Floor visits per episode (the number of chances physics offered)

| controller | overall | light | medium | heavy |
|---|---|---|---|---|
| `stay` | 2.11 | 3.24 | 2.02 | 1.08 |
| `random` | 2.08 | 3.34 | 1.82 | 1.08 |
| `oracle` | 1.91 | 2.96 | 1.72 | 1.04 |
| `ctrl_v2` | 1.83 | 2.72 | 1.80 | 0.98 |
| `ctrl_v2_lastdream` | 1.85 | 2.70 | 1.80 | 1.06 |
| `ctrl_v2_mix` | 2.01 | 3.16 | 1.78 | 1.10 |
| `ctrl_v2_z_only` | 1.93 | 2.94 | 1.82 | 1.02 |
| `ctrl_v2_real` | 2.05 | 3.34 | 1.78 | 1.02 |
| `ctrl_v1_on_v2` | 2.07 | 3.22 | 1.96 | 1.04 |

## Interceptions per episode

| controller | overall | light | medium | heavy |
|---|---|---|---|---|
| `stay` | 0.95 | 1.36 | 0.94 | 0.54 |
| `random` | 0.93 | 1.42 | 0.78 | 0.58 |
| `oracle` | 1.89 | 2.94 | 1.70 | 1.04 |
| `ctrl_v2` | 1.45 | 2.08 | 1.50 | 0.76 |
| `ctrl_v2_lastdream` | 1.43 | 2.06 | 1.38 | 0.84 |
| `ctrl_v2_mix` | 1.24 | 1.62 | 1.18 | 0.92 |
| `ctrl_v2_z_only` | 1.11 | 1.68 | 1.12 | 0.52 |
| `ctrl_v2_real` | 1.31 | 2.22 | 1.12 | 0.58 |
| `ctrl_v1_on_v2` | 1.64 | 2.28 | 1.70 | 0.94 |

## Hits (contact frames) per episode

| controller | overall | light | medium | heavy |
|---|---|---|---|---|
| `stay` | 0.95 | 1.36 | 0.94 | 0.54 |
| `random` | 1.03 | 1.52 | 0.80 | 0.76 |
| `oracle` | 1.95 | 3.08 | 1.70 | 1.06 |
| `ctrl_v2` | 1.48 | 2.12 | 1.56 | 0.76 |
| `ctrl_v2_lastdream` | 1.51 | 2.12 | 1.56 | 0.84 |
| `ctrl_v2_mix` | 1.97 | 1.64 | 1.98 | 2.30 |
| `ctrl_v2_z_only` | 1.13 | 1.70 | 1.14 | 0.56 |
| `ctrl_v2_real` | 1.33 | 2.26 | 1.12 | 0.60 |
| `ctrl_v1_on_v2` | 1.84 | 2.38 | 1.98 | 1.16 |

## Mean gap at closest approach (world units; lower is better)

| controller | overall | light | medium | heavy |
|---|---|---|---|---|
| `stay` | 0.278 | 0.300 | 0.261 | 0.245 |
| `random` | 0.306 | 0.313 | 0.308 | 0.279 |
| `oracle` | 0.069 | 0.078 | 0.063 | 0.054 |
| `ctrl_v2` | 0.140 | 0.157 | 0.119 | 0.130 |
| `ctrl_v2_lastdream` | 0.149 | 0.165 | 0.134 | 0.135 |
| `ctrl_v2_mix` | 0.205 | 0.210 | 0.201 | 0.196 |
| `ctrl_v2_z_only` | 0.219 | 0.216 | 0.215 | 0.235 |
| `ctrl_v2_real` | 0.187 | 0.179 | 0.192 | 0.206 |
| `ctrl_v1_on_v2` | 0.141 | 0.152 | 0.130 | 0.126 |

## Approach-frame sign agreement (chance 0.50)

| controller | overall | light | medium | heavy |
|---|---|---|---|---|
| `stay` | -- | -- | -- | -- |
| `random` | 0.338 | 0.359 | 0.308 | 0.349 |
| `oracle` | 1.000 | 1.000 | 1.000 | 1.000 |
| `ctrl_v2` | 0.721 | 0.757 | 0.709 | 0.694 |
| `ctrl_v2_lastdream` | 0.701 | 0.740 | 0.676 | 0.684 |
| `ctrl_v2_mix` | 0.575 | 0.604 | 0.591 | 0.533 |
| `ctrl_v2_z_only` | 0.649 | 0.653 | 0.653 | 0.642 |
| `ctrl_v2_real` | 0.587 | 0.664 | 0.614 | 0.491 |
| `ctrl_v1_on_v2` | 0.580 | 0.658 | 0.579 | 0.504 |

## Reaction lead A: consecutive frames of correctly-directed paddle motion before each interception

Dominated by dithering (a bang-bang paddle flips sign in place and resets the run). Reported for completeness; read table B.

| controller | overall | light | medium | heavy |
|---|---|---|---|---|
| `stay` | 5.3 | 5.6 | 2.1 | 10.3 |
| `random` | 5.1 | 4.7 | 6.6 | 3.9 |
| `oracle` | 5.3 | 8.2 | 2.1 | 2.2 |
| `ctrl_v2` | 1.5 | 1.5 | 1.4 | 1.3 |
| `ctrl_v2_lastdream` | 1.6 | 1.7 | 2.0 | 0.5 |
| `ctrl_v2_mix` | 2.6 | 2.8 | 3.2 | 1.3 |
| `ctrl_v2_z_only` | 3.5 | 3.2 | 4.1 | 3.2 |
| `ctrl_v2_real` | 3.3 | 2.9 | 4.3 | 2.7 |
| `ctrl_v1_on_v2` | 3.6 | 4.0 | 4.0 | 2.1 |

## Reaction lead B: frames the paddle was already within half a paddle-width of the interception point

The dither-proof version of "how far ahead of the ball's landing did the paddle get".

| controller | overall | light | medium | heavy |
|---|---|---|---|---|
| `stay` | 31.3 | 28.5 | 32.1 | 37.0 |
| `random` | 12.2 | 12.2 | 11.9 | 12.8 |
| `oracle` | 8.8 | 8.1 | 6.5 | 14.7 |
| `ctrl_v2` | 11.7 | 6.8 | 13.8 | 21.1 |
| `ctrl_v2_lastdream` | 11.7 | 6.8 | 14.2 | 19.3 |
| `ctrl_v2_mix` | 4.7 | 6.0 | 5.2 | 2.0 |
| `ctrl_v2_z_only` | 14.3 | 10.7 | 15.2 | 24.3 |
| `ctrl_v2_real` | 8.3 | 7.8 | 7.5 | 11.9 |
| `ctrl_v1_on_v2` | 9.0 | 8.2 | 9.1 | 10.7 |

## Held-out colour band [0.85, 1.2]

60 episodes, seeds 6000.., `mass_only` on the band no model in the stack has ever seen move. The right comparison is against the **medium** tercile above, whose speeds bracket this band — not against the overall average.

| controller | interceptions/visit | floor visits/ep | gap at floor | sign agreement | position lead |
|---|---|---|---|---|---|
| `stay` | 0.55 [0.46, 0.64] | 1.98 | 0.222 | -- | 33.6 |
| `oracle` | 1.00 [1.00, 1.00] | 1.82 | 0.058 | 1.000 | 9.9 |
| `ctrl_v2` | 0.79 [0.72, 0.85] | 1.78 | 0.111 | 0.677 | 13.5 |
| `ctrl_v1_on_v2` | 0.85 [0.78, 0.91] | 1.88 | 0.107 | 0.595 | 9.9 |

## Decision analysis: is there a speed interaction?

`logit(RIGHT) − logit(LEFT)` on approach frames, regressed on standardised features. The base model is `x_err, ball_vx, ball_vy, paddle_vx, speed`; the second adds `speed·x_err` and `speed·ball_vx`. A mass-aware controller should gain R² from them.

| controller | R² base | R² + interactions | ΔR² | frames |
|---|---|---|---|---|
| `ctrl_v2` | 0.4635 | 0.4774 | +0.0139 | 6644 |
| `ctrl_v2_z_only` | 0.1981 | 0.1989 | +0.0008 | 7066 |
| `ctrl_v1_on_v2` | 0.3107 | 0.3299 | +0.0192 | 7175 |

Standardised coefficients of the full model:

| controller | x_err | ball_vx | ball_vy | paddle_vx | speed | speed*x_err | speed*ball_vx |
|---|---|---|---|---|---|---|---|
| `ctrl_v2` | +0.194 | +0.571 | -0.039 | -0.031 | +0.065 | +0.341 | -0.240 |
| `ctrl_v2_z_only` | +0.136 | +0.190 | -0.198 | +0.325 | -0.184 | +0.049 | -0.083 |
| `ctrl_v1_on_v2` | -0.121 | +0.093 | +0.060 | +0.421 | +0.046 | +0.385 | -0.026 |

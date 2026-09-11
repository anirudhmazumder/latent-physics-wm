# Stage three (C) — real-environment evaluation

100 episodes x 200 steps, fixed seeds 5000..5099 (identical starts for every row).

`hits` counts contact FRAMES; `interceptions` collapses a run of consecutive contact frames to one, so a policy that pins the ball against the paddle cannot inflate it.

| controller | hits/ep | 95% CI | interceptions/ep | floor visits/ep | hits per visit | mean gap at floor | eps with ≥1 hit | left/stay/right |
|---|---|---|---|---|---|---|---|---|
| `stay` | **0.92** | [0.77, 1.07] | 0.92 | 1.77 | 0.52 | 0.237 | 69% | 0.00/1.00/0.00 |
| `random` | **0.93** | [0.71, 1.19] | 0.76 | 1.67 | 0.56 | 0.287 | 63% | 0.31/0.34/0.35 |
| `oracle` | **1.68** | [1.52, 1.84] | 1.63 | 1.64 | 1.02 | 0.061 | 97% | 0.24/0.52/0.24 |
| `ctrl_v1` | **1.44** | [1.29, 1.59] | 1.43 | 1.74 | 0.83 | 0.105 | 93% | 0.36/0.27/0.37 |
| `ctrl_v1_lastdream` | **1.45** | [1.24, 1.69] | 1.31 | 1.67 | 0.87 | 0.120 | 89% | 0.31/0.35/0.34 |
| `ctrl_z_only` | **1.20** | [1.00, 1.41] | 1.03 | 1.70 | 0.71 | 0.257 | 76% | 0.35/0.24/0.41 |
| `ctrl_tau1.5` | **1.26** | [1.04, 1.55] | 1.17 | 1.67 | 0.75 | 0.148 | 84% | 0.27/0.42/0.31 |
| `ctrl_dense` | **1.98** | [1.61, 2.44] | 1.61 | 1.71 | 1.16 | 0.100 | 95% | 0.30/0.39/0.32 |
| `ctrl_dense_lastdream` | **1.47** | [1.30, 1.64] | 1.40 | 1.64 | 0.90 | 0.110 | 89% | 0.29/0.41/0.30 |
| `ctrl_real` | **2.00** | [1.57, 2.47] | 1.27 | 1.66 | 1.20 | 0.178 | 83% | 0.32/0.38/0.30 |

## Does each controller move the right way?

Sign agreement between `logit(RIGHT) − logit(LEFT)` and the direction of the true error (chance = 0.50). 'approach' = ball descending in the lower half of the box.

| controller | toward ball (all) | toward ball (approach) | toward ballistic landing (approach) |
|---|---|---|---|
| `ctrl_v1` | 0.590 | 0.601 | 0.565 |
| `ctrl_v1_lastdream` | 0.536 | 0.614 | 0.602 |
| `ctrl_z_only` | 0.535 | 0.527 | 0.514 |
| `ctrl_tau1.5` | 0.559 | 0.508 | 0.532 |
| `ctrl_dense` | 0.577 | 0.704 | 0.687 |
| `ctrl_dense_lastdream` | 0.576 | 0.665 | 0.677 |
| `ctrl_real` | 0.493 | 0.489 | 0.532 |

## What drives `ctrl_v1`'s decisions

R² of a linear fit of `logit(RIGHT) − logit(LEFT)` on true-state features:

| features | R² |
|---|---|
| ball_x − paddle_x | 0.052 |
| + ball_vx, ball_vy, ball_y, paddle_vx | 0.318 |
| predicted landing x − paddle_x | 0.031 |

Coefficients of the middle model (drive std = 26.08):

| feature | coefficient |
|---|---|
| bias | -1.637 |
| x_err | +19.467 |
| ball_vx | +145.946 |
| ball_vy | -194.864 |
| ball_y | +6.387 |
| paddle_vx | +499.556 |

Approach frames used for the sign test: 4593.

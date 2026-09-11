# Independent re-check on fresh seeds (orchestrator)

Same harness, 60 episodes x 200 steps, seeds 9000..9059 (disjoint from both the training checks at 7000+ and the agent's report at 5000+). Run after the agent finished, to test how much of the headline table was seed luck.

| controller | hits/ep | 95% CI | interceptions/ep | floor visits/ep | hits per visit | mean gap at floor | eps with ≥1 hit | left/stay/right |
|---|---|---|---|---|---|---|---|---|
| `stay` | **0.93** | [0.77, 1.08] | 0.93 | 1.65 | 0.57 | 0.258 | 77% | 0.00/1.00/0.00 |
| `random` | **0.73** | [0.57, 0.93] | 0.67 | 1.62 | 0.45 | 0.315 | 62% | 0.35/0.33/0.31 |
| `oracle` | **1.60** | [1.40, 1.83] | 1.55 | 1.58 | 1.01 | 0.062 | 97% | 0.25/0.52/0.23 |
| `ctrl_v1` | **1.55** | [1.28, 1.85] | 1.47 | 1.72 | 0.90 | 0.115 | 88% | 0.37/0.27/0.36 |
| `ctrl_v1_lastdream` | **1.68** | [1.25, 2.23] | 1.38 | 1.58 | 1.06 | 0.118 | 90% | 0.32/0.33/0.35 |
| `ctrl_dense` | **2.12** | [1.62, 2.77] | 1.48 | 1.55 | 1.37 | 0.121 | 93% | 0.30/0.39/0.31 |
| `ctrl_dense_lastdream` | **1.25** | [1.08, 1.42] | 1.20 | 1.50 | 0.83 | 0.126 | 90% | 0.29/0.41/0.30 |
| `ctrl_real` | **1.40** | [1.10, 1.75] | 1.12 | 1.57 | 0.89 | 0.196 | 87% | 0.31/0.39/0.29 |
| `ctrl_z_only` | **0.97** | [0.78, 1.18] | 0.93 | 1.67 | 0.58 | 0.270 | 77% | 0.35/0.24/0.41 |


## Does each controller move the right way?

Sign agreement between `logit(RIGHT) − logit(LEFT)` and the direction of the true error (chance = 0.50). 'approach' = ball descending in the lower half of the box.

| controller | toward ball (all) | toward ball (approach) | toward ballistic landing (approach) |
|---|---|---|---|
| `ctrl_v1` | 0.590 | 0.590 | 0.571 |
| `ctrl_v1_lastdream` | 0.564 | 0.618 | 0.603 |
| `ctrl_dense` | 0.597 | 0.703 | 0.695 |
| `ctrl_dense_lastdream` | 0.566 | 0.627 | 0.665 |
| `ctrl_real` | 0.494 | 0.465 | 0.496 |
| `ctrl_z_only` | 0.533 | 0.507 | 0.489 |


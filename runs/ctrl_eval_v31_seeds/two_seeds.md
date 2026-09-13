# Two training seeds per controller row

References: oracle **0.99**, wait-and-see (the memoryless bound) **0.51**, stay 0.38, random 0.31.

![two seeds](skill_vs_bound_two_seeds.png)

## interceptions / floor visit

| controller | seed 0 | seed 1 | mean | spread |
|---|---|---|---|---|
| `ctrl_v31_tau1` | 0.73 | 0.97 | **0.85** | 0.24 |
| `ctrl_v31_real` | 0.65 | 0.75 | **0.70** | 0.10 |
| `ctrl_v31_poshead` | 0.68 | 0.65 | **0.66** | 0.02 |
| `ctrl_v31_ff` | 0.46 | 0.63 | **0.55** | 0.17 |
| `ctrl_v31` | 0.57 | 0.50 | **0.53** | 0.07 |
| `ctrl_v31_z_only_tau1` | 0.51 | 0.53 | **0.52** | 0.03 |

## interceptions / visit, long moves (>0.35)

| controller | seed 0 | seed 1 | mean | spread |
|---|---|---|---|---|
| `ctrl_v31_tau1` | 0.72 | 0.66 | **0.69** | 0.06 |
| `ctrl_v31_poshead` | 0.71 | 0.49 | **0.60** | 0.23 |
| `ctrl_v31_real` | 0.37 | 0.51 | **0.44** | 0.14 |
| `ctrl_v31` | 0.45 | 0.18 | **0.32** | 0.28 |
| `ctrl_v31_ff` | 0.20 | 0.38 | **0.29** | 0.19 |
| `ctrl_v31_z_only_tau1` | 0.06 | 0.25 | **0.15** | 0.20 |

## displacement while blind / required move

| controller | seed 0 | seed 1 | mean | spread |
|---|---|---|---|---|
| `ctrl_v31_poshead` | 0.30 | 0.27 | **0.28** | 0.03 |
| `ctrl_v31` | 0.28 | 0.24 | **0.26** | 0.04 |
| `ctrl_v31_real` | 0.34 | 0.17 | **0.25** | 0.17 |
| `ctrl_v31_tau1` | 0.22 | 0.10 | **0.16** | 0.12 |
| `ctrl_v31_ff` | 0.13 | 0.09 | **0.11** | 0.04 |
| `ctrl_v31_z_only_tau1` | 0.02 | 0.03 | **0.02** | 0.01 |

## Above the memoryless bound?

| controller | two-seed mean | above the bound on |
|---|---|---|
| `ctrl_v31` | 0.53 | ONE SEED ONLY |
| `ctrl_v31_tau1` | 0.85 | both seeds |
| `ctrl_v31_ff` | 0.55 | ONE SEED ONLY |
| `ctrl_v31_poshead` | 0.66 | both seeds |
| `ctrl_v31_z_only_tau1` | 0.52 | ONE SEED ONLY |
| `ctrl_v31_real` | 0.70 | both seeds |

## Which adjacent orderings survive two seeds -- interceptions / floor visit

| above | below | gap of means | larger spread | seeds overlap | resolved |
|---|---|---|---|---|---|
| `ctrl_v31_tau1` | `ctrl_v31_real` | 0.15 | 0.24 | yes | no |
| `ctrl_v31_real` | `ctrl_v31_poshead` | 0.04 | 0.10 | yes | no |
| `ctrl_v31_poshead` | `ctrl_v31_ff` | 0.12 | 0.17 | no | no |
| `ctrl_v31_ff` | `ctrl_v31` | 0.02 | 0.17 | yes | no |
| `ctrl_v31` | `ctrl_v31_z_only_tau1` | 0.02 | 0.07 | yes | no |

## Which adjacent orderings survive two seeds -- interceptions / visit, long moves (>0.35)

| above | below | gap of means | larger spread | seeds overlap | resolved |
|---|---|---|---|---|---|
| `ctrl_v31_tau1` | `ctrl_v31_poshead` | 0.09 | 0.23 | yes | no |
| `ctrl_v31_poshead` | `ctrl_v31_real` | 0.16 | 0.23 | yes | no |
| `ctrl_v31_real` | `ctrl_v31` | 0.13 | 0.28 | yes | no |
| `ctrl_v31` | `ctrl_v31_ff` | 0.03 | 0.28 | yes | no |
| `ctrl_v31_ff` | `ctrl_v31_z_only_tau1` | 0.13 | 0.20 | yes | no |

## Which adjacent orderings survive two seeds -- displacement while blind / required move

| above | below | gap of means | larger spread | seeds overlap | resolved |
|---|---|---|---|---|---|
| `ctrl_v31_poshead` | `ctrl_v31` | 0.02 | 0.04 | yes | no |
| `ctrl_v31` | `ctrl_v31_real` | 0.01 | 0.17 | yes | no |
| `ctrl_v31_real` | `ctrl_v31_tau1` | 0.10 | 0.17 | yes | no |
| `ctrl_v31_tau1` | `ctrl_v31_ff` | 0.05 | 0.12 | yes | no |
| `ctrl_v31_ff` | `ctrl_v31_z_only_tau1` | 0.08 | 0.04 | no | **yes** |

## Non-adjacent separations that hold on both seeds

Every pair (not only neighbours) whose two seed ranges do not overlap at all. These are the orderings this experiment actually established.

| metric | above | below | above's seeds | below's seeds |
|---|---|---|---|---|
| interceptions / floor visit | `ctrl_v31_tau1` | `ctrl_v31_poshead` | [0.73, 0.97] | [0.65, 0.68] |
| interceptions / floor visit | `ctrl_v31_tau1` | `ctrl_v31_ff` | [0.73, 0.97] | [0.46, 0.63] |
| interceptions / floor visit | `ctrl_v31_tau1` | `ctrl_v31` | [0.73, 0.97] | [0.50, 0.57] |
| interceptions / floor visit | `ctrl_v31_tau1` | `ctrl_v31_z_only_tau1` | [0.73, 0.97] | [0.51, 0.53] |
| interceptions / floor visit | `ctrl_v31_real` | `ctrl_v31_ff` | [0.65, 0.75] | [0.46, 0.63] |
| interceptions / floor visit | `ctrl_v31_real` | `ctrl_v31` | [0.65, 0.75] | [0.50, 0.57] |
| interceptions / floor visit | `ctrl_v31_real` | `ctrl_v31_z_only_tau1` | [0.65, 0.75] | [0.51, 0.53] |
| interceptions / floor visit | `ctrl_v31_poshead` | `ctrl_v31_ff` | [0.65, 0.68] | [0.46, 0.63] |
| interceptions / floor visit | `ctrl_v31_poshead` | `ctrl_v31` | [0.65, 0.68] | [0.50, 0.57] |
| interceptions / floor visit | `ctrl_v31_poshead` | `ctrl_v31_z_only_tau1` | [0.65, 0.68] | [0.51, 0.53] |
| interceptions / visit, long moves (>0.35) | `ctrl_v31_tau1` | `ctrl_v31_real` | [0.66, 0.72] | [0.37, 0.51] |
| interceptions / visit, long moves (>0.35) | `ctrl_v31_tau1` | `ctrl_v31` | [0.66, 0.72] | [0.18, 0.45] |
| interceptions / visit, long moves (>0.35) | `ctrl_v31_tau1` | `ctrl_v31_ff` | [0.66, 0.72] | [0.20, 0.38] |
| interceptions / visit, long moves (>0.35) | `ctrl_v31_tau1` | `ctrl_v31_z_only_tau1` | [0.66, 0.72] | [0.06, 0.25] |
| interceptions / visit, long moves (>0.35) | `ctrl_v31_poshead` | `ctrl_v31` | [0.49, 0.71] | [0.18, 0.45] |
| interceptions / visit, long moves (>0.35) | `ctrl_v31_poshead` | `ctrl_v31_ff` | [0.49, 0.71] | [0.20, 0.38] |
| interceptions / visit, long moves (>0.35) | `ctrl_v31_poshead` | `ctrl_v31_z_only_tau1` | [0.49, 0.71] | [0.06, 0.25] |
| interceptions / visit, long moves (>0.35) | `ctrl_v31_real` | `ctrl_v31_z_only_tau1` | [0.37, 0.51] | [0.06, 0.25] |
| displacement while blind / required move | `ctrl_v31_poshead` | `ctrl_v31_tau1` | [0.27, 0.30] | [0.10, 0.22] |
| displacement while blind / required move | `ctrl_v31_poshead` | `ctrl_v31_ff` | [0.27, 0.30] | [0.09, 0.13] |
| displacement while blind / required move | `ctrl_v31_poshead` | `ctrl_v31_z_only_tau1` | [0.27, 0.30] | [0.02, 0.03] |
| displacement while blind / required move | `ctrl_v31` | `ctrl_v31_tau1` | [0.24, 0.28] | [0.10, 0.22] |
| displacement while blind / required move | `ctrl_v31` | `ctrl_v31_ff` | [0.24, 0.28] | [0.09, 0.13] |
| displacement while blind / required move | `ctrl_v31` | `ctrl_v31_z_only_tau1` | [0.24, 0.28] | [0.02, 0.03] |
| displacement while blind / required move | `ctrl_v31_real` | `ctrl_v31_ff` | [0.17, 0.34] | [0.09, 0.13] |
| displacement while blind / required move | `ctrl_v31_real` | `ctrl_v31_z_only_tau1` | [0.17, 0.34] | [0.02, 0.03] |
| displacement while blind / required move | `ctrl_v31_tau1` | `ctrl_v31_z_only_tau1` | [0.10, 0.22] | [0.02, 0.03] |
| displacement while blind / required move | `ctrl_v31_ff` | `ctrl_v31_z_only_tau1` | [0.09, 0.13] | [0.02, 0.03] |

# Independent re-check on fresh seeds (orchestrator)

90 in-distribution episodes x 200 steps, seeds 9000..9089 (disjoint from training checks at 7000+ and the report at 5000+). Interceptions per floor visit, overall / light / medium / heavy, with 95% bootstrap CIs. Each controller scored with its own (V, M) stack.

| `stay` | 0.52 [0.45, 0.59] | 0.58 [0.47, 0.68] | 0.49 [0.38, 0.60] | 0.42 [0.27, 0.58] |
| `random` | 0.48 [0.39, 0.56] | 0.47 [0.34, 0.60] | 0.54 [0.40, 0.67] | 0.41 [0.23, 0.60] |
| `oracle` | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| `ctrl_v2` | 0.81 [0.74, 0.87] | 0.86 [0.77, 0.94] | 0.82 [0.68, 0.94] | 0.67 [0.52, 0.81] |
| `ctrl_v2_cons` | 0.97 [0.92, 1.02] | 0.94 [0.83, 1.02] | 1.02 [0.95, 1.10] | 1.00 [1.00, 1.00] |
| `ctrl_v2_cons_tau0.5` | 0.98 [0.95, 1.01] | 0.99 [0.94, 1.03] | 0.96 [0.90, 1.00] | 1.00 [1.00, 1.00] |
| `ctrl_v1_on_v2` | 0.80 [0.71, 0.89] | 0.78 [0.64, 0.93] | 0.83 [0.69, 0.94] | 0.81 [0.63, 0.97] |

# Does the controller's decision depend on the ball's colour?

300 approach frames per policy, 200-step episodes, warm-up 8 frames. Repaint masses [0.5, 1.0, 2.0] (m₁ = 1.0 is **inside the held-out band** [0.85, 1.2]). Positions, paddle and actions are bit-identical between the two arms; only the hue changes.

## Headline

| controller | |Δdrive|/sd | flips | null |Δ|/sd | null flips | drive sd | frames (eps) |
|---|---|---|---|---|---|---|
| `ctrl_v2` | 0.244 | 0.071 | 0.000 | 0.000 | 39.26 | 300 (38) |
| `ctrl_v2_cons` | 0.292 | 0.129 | 0.000 | 0.000 | 25.19 | 300 (38) |
| `ctrl_v2_cons_tau0.5` | 0.342 | 0.144 | 0.000 | 0.000 | 24.09 | 300 (38) |
| `ctrl_v2_tau0.5` | 0.324 | 0.120 | 0.000 | 0.000 | 31.04 | 300 (38) |
| `ctrl_v2_z_only` | 0.263 | 0.083 | 0.000 | 0.000 | 7.36 | 300 (38) |
| `ctrl_v1_on_v2` | 0.150 | 0.074 | 0.000 | 0.000 | 33.50 | 300 (38) |
| `oracle` | 0.000 | 0.000 | 0.000 | 0.000 | 0.70 | 300 (38) |

## Per repaint mass

`β` is the coefficient of **Δdrive / std(drive)** on (log m₁ − log m₀)·sign(x_err), with a 95% CI from a bootstrap over episodes. **The law predicts β < 0**: repainting the ball lighter (faster) should push the paddle harder toward the side the ball is on.

| controller | m₁ | |Δ|/sd | flips | β | 95% CI |
|---|---|---|---|---|---|
| `ctrl_v2` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v2` | 0.5 | 0.210 | 0.067 | -0.078 | [-0.196, 0.025] |
| `ctrl_v2` | 1 *(held out)* | 0.178 | 0.033 | -0.201 | [-0.307, -0.097] |
| `ctrl_v2` | 2 | 0.343 | 0.113 | -0.268 | [-0.366, -0.176] |
| `ctrl_v2_cons` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v2_cons` | 0.5 | 0.287 | 0.100 | -0.091 | [-0.218, 0.046] |
| `ctrl_v2_cons` | 1 *(held out)* | 0.195 | 0.090 | -0.086 | [-0.189, 0.023] |
| `ctrl_v2_cons` | 2 | 0.393 | 0.197 | -0.127 | [-0.254, 0.002] |
| `ctrl_v2_cons_tau0.5` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v2_cons_tau0.5` | 0.5 | 0.284 | 0.110 | -0.173 | [-0.287, -0.073] |
| `ctrl_v2_cons_tau0.5` | 1 *(held out)* | 0.241 | 0.093 | -0.360 | [-0.484, -0.234] |
| `ctrl_v2_cons_tau0.5` | 2 | 0.502 | 0.230 | -0.482 | [-0.587, -0.366] |
| `ctrl_v2_tau0.5` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v2_tau0.5` | 0.5 | 0.297 | 0.127 | -0.054 | [-0.202, 0.097] |
| `ctrl_v2_tau0.5` | 1 *(held out)* | 0.235 | 0.090 | -0.186 | [-0.328, -0.040] |
| `ctrl_v2_tau0.5` | 2 | 0.440 | 0.143 | -0.244 | [-0.364, -0.106] |
| `ctrl_v2_z_only` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v2_z_only` | 0.5 | 0.261 | 0.057 | -0.152 | [-0.284, -0.038] |
| `ctrl_v2_z_only` | 1 *(held out)* | 0.181 | 0.040 | -0.209 | [-0.313, -0.110] |
| `ctrl_v2_z_only` | 2 | 0.347 | 0.153 | -0.132 | [-0.207, -0.061] |
| `ctrl_v1_on_v2` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v1_on_v2` | 0.5 | 0.131 | 0.050 | 0.042 | [0.015, 0.071] |
| `ctrl_v1_on_v2` | 1 *(held out)* | 0.193 | 0.120 | -0.124 | [-0.202, -0.038] |
| `ctrl_v1_on_v2` | 2 | 0.126 | 0.053 | -0.004 | [-0.037, 0.027] |
| `oracle` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `oracle` | 0.5 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] |
| `oracle` | 1 *(held out)* | 0.000 | 0.000 | 0.000 | [0.000, 0.000] |
| `oracle` | 2 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] |

## Reconstruction check

Pearson r between the drive the rollout actually used (cold-start h, full episode) and the drive reconstructed here (8-frame warm-start). Not load-bearing — both arms of the intervention share the reconstruction — but it says how close the probe is to the policy as played.

| controller | r |
|---|---|
| `ctrl_v2` | 0.987 |
| `ctrl_v2_cons` | 0.984 |
| `ctrl_v2_cons_tau0.5` | 0.982 |
| `ctrl_v2_tau0.5` | 0.976 |
| `ctrl_v2_z_only` | 1.000 |
| `ctrl_v1_on_v2` | 0.961 |

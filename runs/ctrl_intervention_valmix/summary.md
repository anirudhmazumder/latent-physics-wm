# Does the controller's decision depend on the ball's colour?

300 approach frames per policy, 200-step episodes, warm-up 8 frames. Repaint masses [0.5, 1.0, 2.0] (m₁ = 1.0 is **inside the held-out band** [0.85, 1.2]). Positions, paddle and actions are bit-identical between the two arms; only the hue changes.

## Headline

| controller | |Δdrive|/sd | flips | null |Δ|/sd | null flips | drive sd | frames (eps) |
|---|---|---|---|---|---|---|
| `ctrl_v2` | 0.169 | 0.058 | 0.000 | 0.000 | 58.61 | 300 (19) |
| `ctrl_v2_cons` | 0.247 | 0.111 | 0.000 | 0.000 | 31.43 | 300 (19) |
| `ctrl_v2_cons_tau0.5` | 0.214 | 0.071 | 0.000 | 0.000 | 37.58 | 300 (19) |
| `ctrl_v2_tau0.5` | 0.279 | 0.099 | 0.000 | 0.000 | 37.38 | 300 (19) |
| `ctrl_v2_z_only` | 0.170 | 0.078 | 0.000 | 0.000 | 11.35 | 300 (19) |
| `ctrl_v1_on_v2` | 0.162 | 0.063 | 0.000 | 0.000 | 36.68 | 300 (19) |
| `oracle` | 0.000 | 0.000 | 0.000 | 0.000 | 0.78 | 300 (19) |

## Per repaint mass

`β` is the coefficient of **Δdrive / std(drive)** on (log m₁ − log m₀)·sign(x_err), with a 95% CI from a bootstrap over episodes. **The law predicts β < 0**: repainting the ball lighter (faster) should push the paddle harder toward the side the ball is on.

| controller | m₁ | |Δ|/sd | flips | β | 95% CI |
|---|---|---|---|---|---|
| `ctrl_v2` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v2` | 0.5 | 0.159 | 0.060 | -0.063 | [-0.130, 0.003] |
| `ctrl_v2` | 1 *(held out)* | 0.134 | 0.047 | -0.121 | [-0.194, -0.052] |
| `ctrl_v2` | 2 | 0.215 | 0.067 | -0.178 | [-0.253, -0.097] |
| `ctrl_v2_cons` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v2_cons` | 0.5 | 0.240 | 0.127 | -0.041 | [-0.161, 0.058] |
| `ctrl_v2_cons` | 1 *(held out)* | 0.199 | 0.083 | -0.079 | [-0.151, -0.008] |
| `ctrl_v2_cons` | 2 | 0.301 | 0.123 | -0.099 | [-0.182, -0.017] |
| `ctrl_v2_cons_tau0.5` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v2_cons_tau0.5` | 0.5 | 0.168 | 0.057 | -0.098 | [-0.165, -0.014] |
| `ctrl_v2_cons_tau0.5` | 1 *(held out)* | 0.182 | 0.067 | -0.144 | [-0.214, -0.072] |
| `ctrl_v2_cons_tau0.5` | 2 | 0.294 | 0.090 | -0.185 | [-0.304, -0.066] |
| `ctrl_v2_tau0.5` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v2_tau0.5` | 0.5 | 0.280 | 0.097 | -0.010 | [-0.161, 0.134] |
| `ctrl_v2_tau0.5` | 1 *(held out)* | 0.223 | 0.067 | -0.029 | [-0.154, 0.080] |
| `ctrl_v2_tau0.5` | 2 | 0.336 | 0.133 | -0.148 | [-0.304, 0.006] |
| `ctrl_v2_z_only` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v2_z_only` | 0.5 | 0.185 | 0.093 | -0.092 | [-0.145, -0.039] |
| `ctrl_v2_z_only` | 1 *(held out)* | 0.122 | 0.047 | -0.148 | [-0.206, -0.082] |
| `ctrl_v2_z_only` | 2 | 0.203 | 0.093 | -0.078 | [-0.162, 0.002] |
| `ctrl_v1_on_v2` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `ctrl_v1_on_v2` | 0.5 | 0.126 | 0.040 | -0.031 | [-0.069, 0.008] |
| `ctrl_v1_on_v2` | 1 *(held out)* | 0.237 | 0.113 | 0.033 | [-0.079, 0.148] |
| `ctrl_v1_on_v2` | 2 | 0.123 | 0.037 | 0.008 | [-0.024, 0.048] |
| `oracle` | m₀ (null) | 0.000 | 0.000 | -- | [--, --] |
| `oracle` | 0.5 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] |
| `oracle` | 1 *(held out)* | 0.000 | 0.000 | 0.000 | [0.000, 0.000] |
| `oracle` | 2 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] |

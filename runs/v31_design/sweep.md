# v3.1 design sweep — where does a memoryless oracle fail?

Run BEFORE any training (the v3 lesson). 60 episodes x 200 steps per cell,
seeds 5000+, interceptions per floor visit. `oracle` = tracks the true ball
always; `wait&see` = tracks it only while `ball_visible > 0.5`, else STAY
(the memoryless upper bound); `stay` = never moves. Script: `sweep_v31.py`
(run with `PYTHONPATH=. python runs/v31_design/sweep_v31.py`).

| band (lo, hi) | paddle_w | oracle | wait&see | stay |
|---|---|---|---|---|
| (0.28, 0.58) v3 default | 0.26 | 0.99 | 1.00 | — |
| (0.28, 0.58) | 0.16 | 0.99 | 0.99 | — |
| (0.28, 0.58) | 0.12 | 0.99 | 0.91 | — |
| (0.20, 0.50) | 0.26 / 0.16 / 0.12 | 0.99 | 0.83 / 0.75 / 0.76 | — |
| (0.16, 0.50) | 0.26 / 0.16 / 0.12 | 0.99 | 0.71 / 0.65 / 0.64 | — |
| (0.16, 0.56) | 0.26 / 0.16 / 0.12 | 0.99 | 0.74 / 0.65 / 0.69 | — |
| (0.20, 0.60) | 0.26 / 0.16 / 0.12 | 0.99 | 0.83 / 0.79 / 0.71 | — |
| (0.16, 0.64) | 0.26 / 0.16 | 0.99 | 0.78 / 0.61 | — / 0.38 |
| (0.16, 0.70) | 0.26 / 0.16 | 0.99 | 0.81 / 0.66 | — |
| (0.16, 0.76) | 0.26 / 0.16 | 0.99 | 0.79 / 0.67 | — |
| (0.16, 0.82) | 0.26 / 0.16 | 0.99 | 0.79 / 0.65 | — |
| **(0.13, 0.63)** | **0.16** | **0.99** | **0.48** | **0.38** |
| (0.13, 0.63) | 0.12 | 0.99 | 0.43 | 0.33 |
| (0.13, 0.55) | 0.16 / 0.12 | 0.99 | 0.46 / 0.40 | 0.38 / 0.33 |

Reading: lowering the band's bottom edge matters far more than raising its top
(the memoryless policy pre-positions while the ball is still visible above the
band, and only the drift during the hidden stretch defeats it); making the band
taller beyond ~0.5 does nothing further because the wait&see floor is bounded
below by chance (≈ stay + a bit). Paddle width sets `stay`. Chosen: band
(0.13, 0.63), paddle 0.16 — the oracle-to-memoryless gap is 0.51, vs 0.01 on the
v3 default.

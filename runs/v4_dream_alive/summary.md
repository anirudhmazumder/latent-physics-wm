# v3.1 stage three, step zero — is the dream alive?

The dream puts a ball below the band at a rate comparable to the world's for `lstm`, `transformer`, `tf_ctx32`, `ff`, `noact`; the other models: .

64 dreams x 150 steps per cell, warm-started on 8 real frames from `['data/v4/train_mix']`, driven by a fixed sticky-random action stream. Every model and every temperature sees the identical starts and the identical actions, so the cells are paired.

## The real data, measured with the same detector

| reference | frames | ball present | ball below the band | arrivals below the band per 150 frames | ball-height sd | re-emergence (of runs starting hidden) |
|---|---|---|---|---|---|---|
| `data/v4/train_mix`, 4096 random frames | 4096 | 100.0% | 100.0% | — | — | — |
| **the real continuation of the dreams' own 64 starts** | 64x150 | 100.0% | 100.0% | **1.00** | 0.232 | nan% (0) |

This world has no occluder, so the band was pushed above the ceiling: `ball present` is therefore measured over the whole frame and `ball below the band` is identical to it by construction. Only the presence column and the reward-head correlation carry information here; the arrivals and re-emergence columns are artefacts of a band that is not there.

## Every cell

| model | tau | ball present | ball below band | arrivals / run (real 1.00) | ball-height sd (real 0.232) | mean decoded ball y | re-emergence (of dreams starting hidden) | r(reward head, -abs decoded gap) | n pairs |
|---|---|---|---|---|---|---|---|---|---|
| `lstm` **<-- chosen** | 0.0 | 99.3% | 99.3% | 1.25 | 0.226 | 0.510 | nan% (0) | 0.95 | 9536 |
| `lstm` | 0.5 | 89.7% | 89.7% | 6.09 | 0.100 | 0.571 | nan% (0) | 0.81 | 8614 |
| `lstm` | 1.0 | 92.0% | 92.0% | 7.44 | 0.120 | 0.594 | nan% (0) | 0.87 | 8830 |
| `transformer` **<-- chosen** | 0.0 | 100.0% | 100.0% | 1.00 | 0.228 | 0.518 | nan% (0) | 0.97 | 9600 |
| `transformer` | 0.5 | 98.1% | 98.1% | 2.30 | 0.118 | 0.316 | nan% (0) | 0.95 | 9418 |
| `transformer` | 1.0 | 96.8% | 96.8% | 4.45 | 0.120 | 0.329 | nan% (0) | 0.93 | 9288 |
| `tf_ctx32` **<-- chosen** | 0.0 | 100.0% | 100.0% | 1.00 | 0.233 | 0.533 | nan% (0) | 0.98 | 9600 |
| `tf_ctx32` | 0.5 | 98.9% | 98.9% | 1.73 | 0.093 | 0.622 | nan% (0) | 0.97 | 9497 |
| `tf_ctx32` | 1.0 | 97.5% | 97.5% | 3.36 | 0.096 | 0.612 | nan% (0) | 0.96 | 9357 |
| `ff` **<-- chosen** | 0.0 | 100.0% | 100.0% | 1.00 | 0.034 | 0.531 | nan% (0) | 0.91 | 9600 |
| `ff` | 0.5 | 99.7% | 99.7% | 1.31 | 0.042 | 0.658 | nan% (0) | 0.94 | 9572 |
| `ff` | 1.0 | 95.2% | 95.2% | 6.09 | 0.074 | 0.676 | nan% (0) | 0.86 | 9137 |
| `noact` **<-- chosen** | 0.0 | 100.0% | 100.0% | 1.02 | 0.225 | 0.506 | nan% (0) | 0.97 | 9598 |
| `noact` | 0.5 | 92.4% | 92.4% | 5.53 | 0.100 | 0.690 | nan% (0) | 0.90 | 8866 |
| `noact` | 1.0 | 90.5% | 90.5% | 8.75 | 0.108 | 0.690 | nan% (0) | 0.85 | 8687 |

## The reward head, teacher-forced on real latents

M's dense reward head against the true `1 - |ball_x - paddle_x|` on real latents and real actions. This is the ceiling on what the head can contribute inside a dream.

| model | R2 | Pearson r | rmse | pred mean | true mean | n |
|---|---|---|---|---|---|---|
| `lstm` | 0.902 | 0.951 | 0.0620 | 0.774 | 0.770 | 6,965 |
| `transformer` | 0.948 | 0.976 | 0.0451 | 0.780 | 0.770 | 6,965 |
| `tf_ctx32` | 0.963 | 0.982 | 0.0378 | 0.774 | 0.770 | 6,965 |
| `ff` | 0.825 | 0.909 | 0.0828 | 0.779 | 0.770 | 6,965 |
| `noact` | 0.886 | 0.942 | 0.0666 | 0.767 | 0.770 | 6,965 |

## Chosen training temperature per model

The tau whose count of arrivals below the band is closest to the real continuation's 1.00 per 150 frames, ties going to the lower tau (stage two: a sampled v3.1 dream does not conserve the hidden ball's direction of travel).

| model | chosen tau | its arrivals / run | real |
|---|---|---|---|
| `lstm` | 0.0 | 1.25 | 1.00 |
| `transformer` | 0.0 | 1.00 | 1.00 |
| `tf_ctx32` | 0.0 | 1.00 | 1.00 |
| `ff` | 0.0 | 1.00 | 1.00 |
| `noact` | 0.0 | 1.02 | 1.00 |

Wall clock 245 s.


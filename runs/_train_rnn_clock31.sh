#!/bin/bash
# v3.1 follow-up A: reviving the exit clock. Three models, concurrently,
# OMP_NUM_THREADS=2, on exactly the data and budget `runs/_train_rnn_v31.sh`
# used -- so the only difference from `runs/rnn_v31` is the flag being tested.
#
#   rnn_v31_clock         --clock-head                    FAIR
#                         a linear head on h regressing "frames since the ball
#                         was last at least half visible" and "frames until it
#                         next is", with the visibility sequence read by a
#                         FROZEN poly-2 probe on the model's own input latents
#                         (held-out R^2 0.99). Nothing privileged enters; see
#                         wm/clock.py for why hindsight on target (b) is the
#                         same arrangement the reward head has always had.
#   rnn_v31_clock_priv    --clock-privileged --vy-head    PRIVILEGED CEILING
#                         the identical two targets built from the simulator's
#                         ball_visible column, plus the true ball_vy. It
#                         measures how much of any failure of the fair run is
#                         the probe's fault rather than the objective's.
#   rnn_v31_clock_emerge  --clock-head + v3's emergence fix (open-loop rollout
#                         of 24 steps, exit frames up-weighted x5). The "both
#                         at once" arm, in case counting alone is not enough to
#                         make the MDN place the exit.
#
# `runs/rnn_v31` is NOT retrained and NOT touched.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
COMMON="--data data/v31/train data/v31/train_mix --val data/v31/val data/v31/val_mix --epochs 35 --eval-every 2"
export OMP_NUM_THREADS=2

$PY -m wm.train_rnn $COMMON --out runs/rnn_v31_clock \
    --clock-head --w-clock 1.0            > runs/rnn_v31_clock_train.log 2>&1 &
$PY -m wm.train_rnn $COMMON --out runs/rnn_v31_clock_priv \
    --clock-head --clock-privileged --w-clock 1.0 \
    --vy-head --w-vy 1.0                  > runs/rnn_v31_clock_priv_train.log 2>&1 &
$PY -m wm.train_rnn $COMMON --out runs/rnn_v31_clock_emerge \
    --clock-head --w-clock 1.0 \
    --rollout-loss-steps 24 --rollout-loss-weight 1.0 \
    --emerge-weight 5                     > runs/rnn_v31_clock_emerge_train.log 2>&1 &
wait
echo "=== all three done"

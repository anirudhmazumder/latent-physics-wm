#!/bin/bash
# v3.1 stage two: the four models, trained concurrently with OMP_NUM_THREADS=2.
#
# The argument is the ff/poshead pair: identical data, budget, heads and latent
# space, so whatever the recurrent model scores, the gap up from ff is memory
# and the gap down from poshead is what a 256-unit state COULD hold if it were
# told what to remember.
#
#   rnn_v31          the fair model
#   rnn_v31_ff       the floor (no recurrence at all)
#   rnn_v31_poshead  the PRIVILEGED ceiling (a head on the true ball position)
#   rnn_v31_emerge   the best fair fix from v3 (long open-loop rollout +
#                    emergence frames up-weighted x5). K = 24 reaches the exit
#                    frame for 72% of v3.1's hidden runs (median run 19 frames,
#                    mean 21.6), so it is kept at v3's value rather than raised
#                    to 32, which would also have meant raising --seq-len and
#                    so changing the window the baseline sees.
set -e
cd "$(dirname "$0")/.."
PY=/opt/miniconda3/envs/NN/bin/python
COMMON="--data data/v31/train data/v31/train_mix --val data/v31/val data/v31/val_mix --epochs 35 --eval-every 2"
export OMP_NUM_THREADS=2

$PY -m wm.train_rnn $COMMON --out runs/rnn_v31                         > runs/rnn_v31_train.log 2>&1 &
$PY -m wm.train_rnn $COMMON --out runs/rnn_v31_ff --feedforward        > runs/rnn_v31_ff_train.log 2>&1 &
$PY -m wm.train_rnn $COMMON --out runs/rnn_v31_poshead --pos-head --w-pos 1.0 \
                                                                       > runs/rnn_v31_poshead_train.log 2>&1 &
$PY -m wm.train_rnn $COMMON --out runs/rnn_v31_emerge \
    --rollout-loss-steps 24 --rollout-loss-weight 1.0 --emerge-weight 5 > runs/rnn_v31_emerge_train.log 2>&1 &
wait
echo "=== all four done"

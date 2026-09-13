#!/bin/zsh
# v4 stage two. All models see exactly the same data, the same number of
# epochs and the same window length; the ONLY difference between the first two
# is the sequence backbone.
#
# --seq-len 128 (v1-v3 used 32) and it is not a detail. v4's latent is set by an
# event and must survive ~100 frames; a 32-frame training window cannot contain
# a flip and its consequences, so NEITHER model would have a gradient reason to
# carry the bit and the comparison would measure nothing. 128 is also the
# transformer's context, so its position table is fully exercised in training.
cd "$(dirname "$0")/.."
PY=/opt/miniconda3/envs/NN/bin/python
D="--data data/v4/train data/v4/train_mix --val data/v4/val data/v4/val_mix"
COMMON="${=D} --epochs 35 --seq-len 128 --stride 8 --eval-every 5"
case "$1" in
  rnn)    OMP_NUM_THREADS=3 $PY -m wm.train_rnn ${=COMMON} --out runs/rnn_v4 ;;
  tf)     OMP_NUM_THREADS=3 $PY -m wm.train_rnn ${=COMMON} --arch transformer --context 128 --out runs/tf_v4 --device mps ;;
  ff)     OMP_NUM_THREADS=3 $PY -m wm.train_rnn ${=COMMON} --feedforward --out runs/rnn_v4_ff ;;
  noact)  OMP_NUM_THREADS=3 $PY -m wm.train_rnn ${=COMMON} --ablate-actions --out runs/rnn_v4_noact ;;
  ctx32)  OMP_NUM_THREADS=3 $PY -m wm.train_rnn ${=COMMON} --arch transformer --context 32 --seq-len 32 --stride 4 --out runs/tf_v4_ctx32 --device mps ;;
  match)  OMP_NUM_THREADS=3 $PY -m wm.train_rnn ${=COMMON} --arch transformer --context 128 --d-model 96 --out runs/tf_v4_match --device mps ;;
esac

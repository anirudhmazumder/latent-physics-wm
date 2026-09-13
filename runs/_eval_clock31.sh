#!/bin/bash
# v3.1 follow-up A: scoring the clock models against the frozen baseline.
#
# Three evaluations, in the order they can falsify the hypothesis:
#
#   1. eval_permanence_v3 parts (a) + (f), ALL FOUR MODELS IN ONE PASS, so
#      every one is scored against the same frozen position probe, the same
#      episode split and the same baselines. --skip b e: (b)/(d) dream through
#      real occlusions and (e) re-simulates and re-encodes, both of which are
#      expensive and neither of which is the question here. (f) is the
#      question: does `frames_hidden` come back off h?
#   2. eval_dream_alive, the decisive test. Same 64 starts, same fixed
#      sticky-random action stream and the same seed as the run that produced
#      runs/rnn_v31_dream_alive, with the baseline included -- so its row is
#      also a determinism check on the whole pipeline.
#   3. eval_rnn, so that "the clock head cost nothing elsewhere" is measured
#      rather than assumed: the visible-frame dream horizon and the paddle-
#      contact head's PR-AUC.
set -e
cd "$(dirname "$0")/.."
PY=/opt/miniconda3/envs/NN/bin/python
export OMP_NUM_THREADS=3
VAE="--vae runs/vae_v31/vae.pt --val data/v31/val data/v31/val_mix \
     --probe-data data/v31/probe data/v31/val_mix"

echo "=== (a)+(f) permanence probes, one pass"
$PY -m wm.eval_permanence_v3 --out runs/rnn_v31_clock_permanence \
    --vae runs/vae_v31/vae.pt \
    --val data/v31/val data/v31/val_mix \
    --bands short=data/v31/short long=data/v31/long \
    --probe-data data/v31/probe data/v31/val_mix \
    --extra 16 --max-age 40 --skip b e \
    --models baseline=runs/rnn_v31/rnn.pt \
             clock=runs/rnn_v31_clock/rnn.pt \
             clock_priv=runs/rnn_v31_clock_priv/rnn.pt \
             clock_emerge=runs/rnn_v31_clock_emerge/rnn.pt \
    > runs/rnn_v31_clock_permanence.log 2>&1

echo "=== is the dream alive?  tau in {0, 0.5, 1}"
$PY -m wm.eval_dream_alive --out runs/rnn_v31_clock_dream_alive \
    --models baseline=runs/rnn_v31/rnn.pt \
             clock=runs/rnn_v31_clock/rnn.pt \
             clock_priv=runs/rnn_v31_clock_priv/rnn.pt \
             clock_emerge=runs/rnn_v31_clock_emerge/rnn.pt \
    > runs/rnn_v31_clock_dream_alive.log 2>&1

echo "=== which of the two counters did the head actually learn?"
# The combined clock MSE in the training log averages over BOTH targets and
# over visible frames (where both are 0 and predicting 0 is free), so it cannot
# tell "learned one" from "learned both". This splits it, on hidden frames only.
$PY -m wm.eval_clock_readout --out runs/rnn_v31_clock_readout \
    --models clock=runs/rnn_v31_clock/rnn.pt \
             clock_priv=runs/rnn_v31_clock_priv/rnn.pt \
             clock_emerge=runs/rnn_v31_clock_emerge/rnn.pt \
    > runs/rnn_v31_clock_readout.log 2>&1

for M in _clock _clock_priv _clock_emerge; do
  echo "=== eval_rnn rnn_v31$M"
  $PY -m wm.eval_rnn --ckpt runs/rnn_v31$M/rnn.pt $VAE --ablate-ckpt "" \
      --out runs/rnn_v31$M/eval > runs/rnn_v31${M}_eval.log 2>&1
done
echo "=== done"

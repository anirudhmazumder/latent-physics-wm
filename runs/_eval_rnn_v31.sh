#!/bin/bash
# v3.1 stage-two evaluation. Everything is scored against the same frozen
# position probe, the same episode split and the same baselines, which is why
# the permanence suite runs all four models in ONE pass.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
export OMP_NUM_THREADS=3
VAE="--vae runs/vae_v31/vae.pt --val data/v31/val data/v31/val_mix \
     --probe-data data/v31/probe data/v31/val_mix"

for M in "" _ff _poshead _emerge; do
  echo "=== eval_rnn rnn_v31$M"
  $PY -m wm.eval_rnn --ckpt runs/rnn_v31$M/rnn.pt $VAE --ablate-ckpt "" \
      --out runs/rnn_v31$M/eval > runs/rnn_v31${M}_eval.log 2>&1
  echo "=== eval_conservation rnn_v31$M"
  $PY -m wm.eval_conservation --ckpt runs/rnn_v31$M/rnn.pt $VAE \
      --out runs/rnn_v31$M/conservation --name "v31$M" \
      > runs/rnn_v31${M}_conservation.log 2>&1
done

# The permanence suite. --extra 16 and --max-age 40 because v3.1's default band
# hides the ball for a mean of 21 frames (v3: 9.4) and its long band for 31, so
# v3's constants (10 and 25) would truncate the answer rather than fail loudly.
echo "=== eval_permanence_v3 (all four models, one pass)"
$PY -m wm.eval_permanence_v3 --out runs/rnn_v31_permanence \
    --vae runs/vae_v31/vae.pt \
    --val data/v31/val data/v31/val_mix \
    --bands short=data/v31/short long=data/v31/long \
    --probe-data data/v31/probe data/v31/val_mix \
    --extra 16 --max-age 40 \
    --models baseline=runs/rnn_v31/rnn.pt ff=runs/rnn_v31_ff/rnn.pt \
             poshead=runs/rnn_v31_poshead/rnn.pt emerge=runs/rnn_v31_emerge/rnn.pt \
    > runs/rnn_v31_permanence.log 2>&1

echo "=== comparison figure"
$PY -m wm.compare_fix_v3 \
    --permanence runs/rnn_v31_permanence/report.json \
    --out runs/rnn_v31_permanence_comparison.png \
    --title "v3.1 object permanence: baseline vs the feed-forward floor and the privileged ceiling" \
    --runs baseline=runs/rnn_v31 ff=runs/rnn_v31_ff \
           poshead=runs/rnn_v31_poshead emerge=runs/rnn_v31_emerge
echo "=== done"

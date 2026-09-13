#!/bin/zsh
# v4 stage two evaluation. One model per invocation so at most three run at once.
#
#   --occluder-y 1.01 1.02 in the dream-alive call is the v4 adaptation of a
#   v3.1 instrument. That script measures "is there a ball, and is it BELOW the
#   band"; v4 has no band, so the band is pushed above the ceiling, which makes
#   "outside the band" the whole frame and "below the band" identical to "present".
#   Nothing else about the measurement changes, and the paddle detector (which
#   searches the rows below the band) then searches the whole frame, as it should.
cd "$(dirname "$0")/.."
PY=${PY:-python}
R=runs/$1
VAL="data/v4/val data/v4/val_mix"
case "$2" in
  rnn)
    OMP_NUM_THREADS=3 $PY -m wm.eval_rnn --ckpt $R/rnn.pt --vae runs/vae_v4/vae.pt \
      --val ${=VAL} --probe-data data/v4/probe --ablate-ckpt runs/rnn_v4_noact/rnn.pt \
      --out $R/eval --horizon 64 ;;
  cons)
    OMP_NUM_THREADS=3 $PY -m wm.eval_conservation --ckpt $R/rnn.pt --vae runs/vae_v4/vae.pt \
      --val ${=VAL} --probe-data data/v4/probe --taus 0.0 0.5 1.0 \
      --n-episodes 30 --horizon 190 --sign-metric \
      --sign-probe-data data/v4/train --out $R/conservation ;;
      # The sign probe is fitted on data/v4/train, not on --val. 35 val episodes
      # is far too few: the sign is constant over long runs inside an episode, so
      # the effective sample size of an episode-level split over them is a few
      # dozen, and the probe comes out at chance -- which makes the consistency
      # number below unreadable rather than negative.
  alive)
    # One invocation for ALL models: the script draws a single set of 64 starts
    # and one action stream, so every row is paired only if they share a run.
    OMP_NUM_THREADS=3 $PY -m wm.eval_dream_alive --vae runs/vae_v4/vae.pt \
      --models lstm=runs/rnn_v4/rnn.pt transformer=runs/tf_v4/rnn.pt \
               tf_ctx32=runs/tf_v4_ctx32/rnn.pt ff=runs/rnn_v4_ff/rnn.pt \
               noact=runs/rnn_v4_noact/rnn.pt --temperatures 0.0 0.5 1.0 \
      --start-roots data/v4/train_mix --val ${=VAL} --real-root data/v4/train_mix \
      --occluder-y 1.01 1.02 --paddle-w 0.26 --ball-radius 0.08 \
      --steps 150 --out runs/v4_dream_alive ;;
esac

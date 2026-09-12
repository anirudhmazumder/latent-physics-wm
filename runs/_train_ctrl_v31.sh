#!/bin/zsh
# Stage three of v3.1: seven controllers, three concurrent processes at a time.
#
# The temperatures are NOT a free choice -- each one is the tau
# `wm.eval_dream_alive` selected for that dynamics model (the tau whose rate of
# bringing a ball below the band is closest to the real continuation's 0.80
# arrivals per 150 frames; runs/rnn_v31_dream_alive/summary.md):
#
#   baseline  tau 0.0     poshead  tau 0.0     ff  tau 1.0     allbands  tau 1.0
#
# `ctrl_v31_tau1` exists because the baseline's chosen tau is not 1.0, so the
# temperature it would have been given by default is run as well and the two
# are comparable on the same 150 evaluation episodes.
#
# The world is v3.1's and every flag of it is spelled out: --paddle-w is not
# optional (it sets the chance catch rate, and therefore the memoryless bound
# of 0.48 that every result is graded against) and neither is --occluder-y.
cd "$(dirname "$0")/.."
PY=/opt/miniconda3/envs/NN/bin/python
ENV=(--occluder --occluder-y 0.13 0.63 --paddle-w 0.16 --ball-radius 0.08)
COMMON=(--vae runs/vae_v31/vae.pt --data data/v31/train data/v31/train_mix
        --inputs zh --reward dense "${ENV[@]}"
        --dream-steps 150 --popsize 32 --rollouts 16 --generations 200 --sigma0 0.5
        --real-eval-every 5 --real-eval-episodes 24 --real-eval-seed-base 7000
        --real-eval-metric interceptions)
run () {  # run OUT RNN TAU [extra...]
  local out=$1 rnn=$2 tau=$3; shift 3
  OMP_NUM_THREADS=2 $PY -m wm.train_controller --out "$out" --rnn "$rnn" \
      --temperature "$tau" "${COMMON[@]}" "$@" > "${out}_train.log" 2>&1
}

run runs/ctrl_v31          runs/rnn_v31/rnn.pt          0.0 &
run runs/ctrl_v31_tau1     runs/rnn_v31/rnn.pt          1.0 &
run runs/ctrl_v31_ff       runs/rnn_v31_ff/rnn.pt       1.0 &
wait
run runs/ctrl_v31_poshead  runs/rnn_v31_poshead/rnn.pt  0.0 &
run runs/ctrl_v31_allbands runs/rnn_v31_allbands/rnn.pt 1.0 &
run runs/ctrl_v31_z_only   runs/rnn_v31/rnn.pt          0.0 --inputs z &
wait

# The no-world-model control: the SAME optimiser, the same 819-parameter linear
# policy and the same inputs [mu_t, h_pre_t] -- M is still there as a feature
# extractor, so `h` is available -- but fitness is interceptions counted in the
# real BouncingBox rather than in a dream. v1/v2's budget: popsize 16, 8
# rollouts, 40 generations = 1.02M real environment steps against the dream
# runs' 197k. If this row wins, the limit is the dream; if it does not, the
# limit is the policy class.
OMP_NUM_THREADS=6 $PY -m wm.train_controller --out runs/ctrl_v31_real \
    --rnn runs/rnn_v31/rnn.pt --vae runs/vae_v31/vae.pt \
    --data data/v31/train data/v31/train_mix \
    --inputs zh --fitness real --real-fitness-count interceptions \
    "${ENV[@]}" --popsize 16 --rollouts 8 --generations 40 --sigma0 0.5 \
    --real-eval-every 5 --real-eval-episodes 24 --real-eval-seed-base 7000 \
    --real-eval-metric interceptions > runs/ctrl_v31_real_train.log 2>&1
echo DONE

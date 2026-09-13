#!/bin/zsh
# Stage three of v3: six controllers, three concurrent processes at a time.
cd "$(dirname "$0")/.."
PY=${PY:-python}
COMMON=(--vae runs/vae_v3/vae.pt --data data/v3/train data/v3/train_mix
        --occluder --inputs zh --reward dense
        --dream-steps 150 --popsize 32 --rollouts 16 --generations 200 --sigma0 0.5
        --real-eval-every 5 --real-eval-episodes 24 --real-eval-seed-base 7000
        --real-eval-metric interceptions)
run () {  # run OUT RNN TEMP [extra...]
  local out=$1 rnn=$2 tau=$3; shift 3
  OMP_NUM_THREADS=2 $PY -m wm.train_controller --out "$out" --rnn "$rnn" \
      --temperature "$tau" "${COMMON[@]}" "$@" > "${out}_train.log" 2>&1
}
run runs/ctrl_v3         runs/rnn_v3/rnn.pt         0.5 &
run runs/ctrl_v3_tau1    runs/rnn_v3/rnn.pt         1.0 &
run runs/ctrl_v3_ff      runs/rnn_v3_ff/rnn.pt      0.5 &
wait
run runs/ctrl_v3_poshead runs/rnn_v3_poshead/rnn.pt 0.5 &
run runs/ctrl_v3_emerge  runs/rnn_v3_emerge/rnn.pt  0.5 &
run runs/ctrl_v3_z_only  runs/rnn_v3/rnn.pt         0.5 --inputs z &
wait
echo DONE

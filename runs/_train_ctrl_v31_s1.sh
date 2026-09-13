#!/bin/zsh
# v3.1 follow-up B: a SECOND TRAINING SEED for every controller row.
#
# Every flag is copied from `runs/_train_ctrl_v31.sh` (and, for the z-only
# tau = 1 row, from the extra run that followed it) and exactly one thing is
# changed: `--seed 1`. In `wm.train_controller` that one number seeds torch,
# numpy, the input-normalisation sample, CMA-ES itself (`seed + 1`) and the
# per-generation rollout starts (`seed * 10000 + gen`), so it is a genuine
# retrain and not a re-evaluation of the same search.
#
# Why this exists. README_C31 §4 reports an ordering -- fair > privileged >
# real > z-only ~ the memoryless bound > feed-forward floor -- off ONE CMA-ES
# run per row. CMA-ES on a 819-parameter policy scored by 16 noisy rollouts is
# not a deterministic map from settings to skill, and several of those gaps are
# 0.05-0.08 interceptions per visit. A second seed is the cheapest test of
# whether the ordering is a property of the settings or of the draw.
#
# The temperatures are unchanged, and deliberately so: they were chosen by
# `wm.eval_dream_alive` from the DYNAMICS model, which is frozen here. Only the
# controller's seed moves.
#
# Outputs go to runs/<name>_s1/. Nothing under runs/<name>/ is touched.
cd "$(dirname "$0")/.."
PY=/opt/miniconda3/envs/NN/bin/python
ENV=(--occluder --occluder-y 0.13 0.63 --paddle-w 0.16 --ball-radius 0.08)
COMMON=(--vae runs/vae_v31/vae.pt --data data/v31/train data/v31/train_mix
        --inputs zh --reward dense "${ENV[@]}"
        --dream-steps 150 --popsize 32 --rollouts 16 --generations 200 --sigma0 0.5
        --real-eval-every 5 --real-eval-episodes 24 --real-eval-seed-base 7000
        --real-eval-metric interceptions --seed 1)
run () {  # run OUT RNN TAU [extra...]
  local out=$1 rnn=$2 tau=$3; shift 3
  OMP_NUM_THREADS=2 $PY -m wm.train_controller --out "$out" --rnn "$rnn" \
      --temperature "$tau" "${COMMON[@]}" "$@" > "${out}_train.log" 2>&1
}

run runs/ctrl_v31_tau1_s1        runs/rnn_v31/rnn.pt         1.0 &
run runs/ctrl_v31_s1             runs/rnn_v31/rnn.pt         0.0 &
run runs/ctrl_v31_ff_s1          runs/rnn_v31_ff/rnn.pt      1.0 &
wait
run runs/ctrl_v31_poshead_s1     runs/rnn_v31_poshead/rnn.pt 0.0 &
run runs/ctrl_v31_z_only_tau1_s1 runs/rnn_v31/rnn.pt         1.0 --inputs z &
wait
echo DONE

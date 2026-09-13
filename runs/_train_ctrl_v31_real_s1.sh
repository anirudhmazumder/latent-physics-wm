#!/bin/zsh
# The no-world-model control, seed 1. Identical to the tail of
# runs/_train_ctrl_v31.sh except for --seed 1; OMP_NUM_THREADS=2 rather than 6
# because it is run alongside the Part A dynamics training.
cd "$(dirname "$0")/.."
PY=/opt/miniconda3/envs/NN/bin/python
OMP_NUM_THREADS=2 $PY -m wm.train_controller --out runs/ctrl_v31_real_s1 \
    --rnn runs/rnn_v31/rnn.pt --vae runs/vae_v31/vae.pt \
    --data data/v31/train data/v31/train_mix \
    --inputs zh --fitness real --real-fitness-count interceptions \
    --occluder --occluder-y 0.13 0.63 --paddle-w 0.16 --ball-radius 0.08 \
    --popsize 16 --rollouts 8 --generations 40 --sigma0 0.5 \
    --real-eval-every 5 --real-eval-episodes 24 --real-eval-seed-base 7000 \
    --real-eval-metric interceptions --seed 1 > runs/ctrl_v31_real_s1_train.log 2>&1
echo DONE

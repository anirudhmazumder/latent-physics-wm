#!/bin/zsh
# Stage three of v3.1: the one evaluation call.
#
# Everything the v3 evaluator hard-coded about its world is a flag now, and all
# three of them are named here: the band, the paddle width, and the two other
# bands. `--paddle-w 0.16` is the load-bearing one -- it sets the stand-still
# catch rate and both oracles' dead zones, so it is what makes the memoryless
# bound come out at 0.48 rather than v3's 0.99.
#
# The other bands are (0.13, 0.45) and (0.13, 0.78), the two the v3.1 VAE was
# ALSO trained on. That is the difference from v3, where the same test ran on
# bands the encoder had never seen and cost it 133-262x reconstruction error;
# `--recon-roots` prints the reconstruction on each of them so the claim can be
# checked rather than asserted.
cd "$(dirname "$0")/.."
PY=/opt/miniconda3/envs/NN/bin/python
OMP_NUM_THREADS=6 $PY -m wm.eval_controller_v3 \
    --out runs/ctrl_eval_v31 \
    --vae runs/vae_v31/vae.pt --rnn runs/rnn_v31/rnn.pt \
    --runs runs/ctrl_v31 runs/ctrl_v31_tau1 runs/ctrl_v31_ff \
           runs/ctrl_v31_poshead runs/ctrl_v31_allbands runs/ctrl_v31_z_only \
           runs/ctrl_v31_real \
    --occluder-y 0.13 0.63 --paddle-w 0.16 --ball-radius 0.08 \
    --episodes 150 --steps 200 --seed-base 5000 \
    --band-names ctrl_v31 ctrl_v31_ff ctrl_v31_poshead oracle wait_and_see \
    --band-episodes 60 --bands 0.13,0.45 0.13,0.78 --band-seed-base 6000 \
    --recon-roots data/v31/val data/v31/short data/v31/long \
    --dream-roots data/v31/train data/v31/train_mix \
    --gif-run ctrl_v31 --dream-temperature 0.0

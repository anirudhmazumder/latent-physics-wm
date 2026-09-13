#!/bin/zsh
# v3.1 follow-up B: BOTH SEEDS of every controller row, on ONE set of episodes.
#
# Seeds 5000-5149, 200 steps, band (0.13, 0.63), paddle 0.16 -- the identical
# world and the identical starts `runs/_eval_ctrl_v31.sh` used, so a seed-0 row
# here must reproduce its number in README_C31 §4 exactly. That reproduction is
# the check that nothing about the evaluation moved between the two runs; if a
# seed-0 row disagrees, the two-seed table is meaningless and the disagreement
# is the finding.
#
# Differences from `_eval_ctrl_v31.sh`, all of them narrowings:
#   --params params_best_real   only the selected policy. The `_lastdream`
#                               variant is a second reading of the same CMA-ES
#                               run, not a second run, so it has nothing to say
#                               about seed variance and would double the cost.
#   --bands (empty) --no-gifs   the other bands and the demo GIFs are unchanged
#                               facts about seed 0 and are already published.
#
# The last three rows -- `ctrl_v31_tau0.5`, `ctrl_v31_allbands`, `ctrl_v31_z_only`
# -- have no `_s1` twin and so contribute nothing to the two-seed table
# (`wm.two_seeds` pairs on the `_s1` suffix and skips them, deliberately: a
# one-seed row in a two-seed table invites exactly the reading this exercise
# exists to prevent). They are here so that EVERY v3.1 controller sits on the
# same 150 episodes with the same references, which is what makes the
# single-seed ordering in README_C31 section 4 and the two-seed verdict below
# comparable line for line rather than across two evaluation runs.
#
# CLOCK_RUNS is left empty: follow-up A found no temperature at which any clock
# model's dream is alive (README_CLOCK31.md section 0), so no `ctrl_v31_clock`
# was trained and there is nothing to add here.
#
# Per-run (V, M) pairing is automatic: every run records its own --rnn and
# --vae in history.json and `stack_for_run` reads them, so the `_s1` runs are
# paired with the same dynamics model their seed-0 twin used. The script prints
# the pairing it chose for every row -- read it.
cd "$(dirname "$0")/.."
PY=${PY:-python}
OMP_NUM_THREADS=6 $PY -m wm.eval_controller_v3 \
    --out runs/ctrl_eval_v31_seeds \
    --vae runs/vae_v31/vae.pt --rnn runs/rnn_v31/rnn.pt \
    --runs runs/ctrl_v31            runs/ctrl_v31_s1 \
           runs/ctrl_v31_tau1       runs/ctrl_v31_tau1_s1 \
           runs/ctrl_v31_ff         runs/ctrl_v31_ff_s1 \
           runs/ctrl_v31_poshead    runs/ctrl_v31_poshead_s1 \
           runs/ctrl_v31_z_only_tau1 runs/ctrl_v31_z_only_tau1_s1 \
           runs/ctrl_v31_real       runs/ctrl_v31_real_s1 \
           runs/ctrl_v31_tau0.5     runs/ctrl_v31_allbands \
           runs/ctrl_v31_z_only \
           ${=CLOCK_RUNS} \
    --params params_best_real \
    --occluder-y 0.13 0.63 --paddle-w 0.16 --ball-radius 0.08 \
    --episodes 150 --steps 200 --seed-base 5000 \
    --bands --band-names --no-gifs \
    --recon-roots data/v31/val \
    --dream-roots data/v31/train data/v31/train_mix

$PY -m wm.two_seeds --summary runs/ctrl_eval_v31_seeds/summary.json \
    --out runs/ctrl_eval_v31_seeds

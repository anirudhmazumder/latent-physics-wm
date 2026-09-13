#!/bin/bash
# v4 data collection. gravity 1e-4 with a 40-degree minimum launch angle, the
# pair chosen in runs/v4_design/sweep.md -- read that first, because the sweep's
# headline finding (a sign-blind oracle is NOT punished on the tracking task)
# changes what the stage-three question can be.
#
# No occluder and no mass: the gravity sign is the only hidden variable, so any
# memory result is unambiguously about it.
set -e
PY=/opt/miniconda3/envs/NN/bin/python
cd "$(dirname "$0")/../.."
C="$PY -m worldsim.collect --ball-radius 0.08 --res 64 --gravity 0.0001 --launch-min-angle 40"

echo "=== train (150, sticky)"
$C --out data/v4/train      --episodes 150 --steps 200 --seed 0   --policy sticky
echo "=== train_mix (300, mix 0.5)"
$C --out data/v4/train_mix  --episodes 300 --steps 200 --seed 10  --policy mix --p-track 0.5
echo "=== val (15, sticky)"
$C --out data/v4/val        --episodes 15  --steps 200 --seed 1   --policy sticky
echo "=== val_mix (20, mix)"
$C --out data/v4/val_mix    --episodes 20  --steps 200 --seed 11  --policy mix --p-track 0.5
echo "=== probe (120 x 24, sticky) -- many short episodes, for the probes"
$C --out data/v4/probe      --episodes 120 --steps 24  --seed 777 --policy sticky
# 600-step episodes: the memory curve stage two measures needs episodes several
# flip-intervals long (flips land ~90 frames apart), and a 200-step episode can
# only ever show two. 30 x 601 frames is 221 MB, well inside the memmap budget.
echo "=== long (30 x 600, mix)"
$C --out data/v4/long       --episodes 30  --steps 600 --seed 50  --policy mix --p-track 0.5
echo "=== done"

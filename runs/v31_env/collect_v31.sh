#!/bin/bash
# v3.1 data collection. Band (0.13, 0.63) is the default; (0.13, 0.45) and
# (0.13, 0.78) are the shorter/longer-occlusion bands, collected for the
# memory-horizon test AND fed to the VAE so that the encoder is in-distribution
# on all three (the confound that made v3's taller-band numbers unreadable).
set -e
PY=/opt/miniconda3/envs/NN/bin/python
cd "$(dirname "$0")/../.."
C="$PY -m worldsim.collect --ball-radius 0.08 --steps 200 --res 64 --occluder --paddle-w 0.16"

echo "=== train (150, sticky, default band)"
$C --out data/v31/train       --episodes 150 --seed 0  --policy sticky --occluder-y 0.13 0.63
echo "=== train_mix (300, mix)"
$C --out data/v31/train_mix   --episodes 300 --seed 10 --policy mix --p-track 0.5 --occluder-y 0.13 0.63
echo "=== train_short (100, mix, short band)"
$C --out data/v31/train_short --episodes 100 --seed 40 --policy mix --p-track 0.5 --occluder-y 0.13 0.45
echo "=== train_long (100, mix, long band)"
$C --out data/v31/train_long  --episodes 100 --seed 41 --policy mix --p-track 0.5 --occluder-y 0.13 0.78
echo "=== val (15, sticky)"
$C --out data/v31/val         --episodes 15  --seed 1  --policy sticky --occluder-y 0.13 0.63
echo "=== val_mix (20, mix)"
$C --out data/v31/val_mix     --episodes 20  --seed 11 --policy mix --p-track 0.5 --occluder-y 0.13 0.63
echo "=== probe (120 x 24, sticky)"
$PY -m worldsim.collect --ball-radius 0.08 --res 64 --occluder --paddle-w 0.16 \
    --out data/v31/probe --episodes 120 --steps 24 --seed 777 --policy sticky --occluder-y 0.13 0.63
echo "=== short (30, mix, short band)"
$C --out data/v31/short       --episodes 30  --seed 42 --policy mix --p-track 0.5 --occluder-y 0.13 0.45
echo "=== long (30, mix, long band)"
$C --out data/v31/long        --episodes 30  --seed 43 --policy mix --p-track 0.5 --occluder-y 0.13 0.78
echo "=== done"

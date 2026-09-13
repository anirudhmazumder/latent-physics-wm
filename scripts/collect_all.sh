#!/usr/bin/env bash
#
# Regenerate every dataset this repository's results were produced from, and
# cache the VAE latents for them.
#
#   bash scripts/collect_all.sh              # all five versions (~30 min, ~7 GB)
#   bash scripts/collect_all.sh --only v1    # one version
#   bash scripts/collect_all.sh --only v31
#   bash scripts/collect_all.sh --no-latents # frames only, skip cache_latents
#
# Versions: v1 v2 v3 v31 v4.
#
# The environment is deterministic: every split is seeded, so a fresh run
# reproduces the frames byte-for-byte. `tests/test_env_v4.py` asserts exactly
# that, which is why the seeds below must not be edited.
#
# Latent caching needs the VAE checkpoints, which ship in the repo
# (runs/vae_b1, runs/vae_v2, runs/vae_v3, runs/vae_v31, runs/vae_v4). Override
# the accelerator with DEVICE=cpu if `mps` is unavailable on your machine.
#
# Every command below is quoted from the run logs under wm/ (README_V2.md,
# README_V3.md, README_M2.md, README_M4.md) or from the collection scripts
# under runs/*_env/. See data/README.md for the same commands with commentary.

set -euo pipefail

PY=${PY:-python}
DEVICE=${DEVICE:-mps}
cd "$(dirname "$0")/.."

ONLY=""
LATENTS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --no-latents) LATENTS=0; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
banner() { echo; echo "=============== $*"; }

# --------------------------------------------------------------------- v1
if want v1; then
  banner "v1 — the plain paddle game (5 splits, ~30 s, 1.1 GB)"
  C="$PY -m worldsim.collect --ball-radius 0.08 --res 64"
  $C --out data/v1/train     --episodes 150 --steps 200 --seed 0
  $C --out data/v1/val       --episodes 15  --steps 200 --seed 1
  $C --out data/v1/probe     --episodes 120 --steps 24  --seed 777
  $C --out data/v1/train_mix --episodes 300 --steps 200 --seed 10 --policy mix --p-track 0.5
  $C --out data/v1/val_mix   --episodes 20  --steps 200 --seed 11 --policy mix --p-track 0.5

  if [ "$LATENTS" = 1 ]; then
    banner "v1 latents (encoder runs/vae_b1/vae.pt)"
    for s in train val probe train_mix val_mix; do
      $PY -m wm.cache_latents --ckpt runs/vae_b1/vae.pt --data "data/v1/$s" --device "$DEVICE"
    done
  fi
fi

# --------------------------------------------------------------------- v2
if want v2; then
  banner "v2 — mass from colour (6 splits, ~40 s, 1.3 GB)"
  C="$PY -m worldsim.collect --ball-radius 0.08 --steps 200 --res 64 --mass-from-color"
  HO="--mass-holdout 0.85 1.2"
  # shellcheck disable=SC2086
  $C --out data/v2/train     --episodes 150 --seed 0   --policy sticky $HO
  # shellcheck disable=SC2086
  $C --out data/v2/train_mix --episodes 300 --seed 10  --policy mix --p-track 0.5 $HO
  # shellcheck disable=SC2086
  $C --out data/v2/val       --episodes 15  --seed 1   --policy sticky $HO
  # shellcheck disable=SC2086
  $C --out data/v2/val_mix   --episodes 20  --seed 11  --policy mix $HO
  # `probe` overrides --steps, so it is spelled out rather than reusing $C.
  # shellcheck disable=SC2086
  $PY -m worldsim.collect --out data/v2/probe --episodes 120 --seed 777 --policy sticky \
      --steps 24 --ball-radius 0.08 --res 64 --mass-from-color $HO
  # `holdout` is the generalisation set: ONLY the colours training never saw.
  $C --out data/v2/holdout   --episodes 30  --seed 21  --policy mix --mass-only 0.85 1.2

  if [ "$LATENTS" = 1 ]; then
    banner "v2 latents (encoder runs/vae_v2/vae.pt)"
    for d in train train_mix val val_mix probe holdout; do
      $PY -m wm.cache_latents --ckpt runs/vae_v2/vae.pt --data "data/v2/$d" --device "$DEVICE"
    done
    # The colour-blind control: the same v2 frames through the v1 encoder,
    # written to mu_v1vae.npy so `--latent-suffix v1vae` can select them.
    banner "v2 colour-blind control latents (encoder runs/vae_b1/vae.pt, suffix v1vae)"
    for d in train train_mix val val_mix probe holdout; do
      $PY -m wm.cache_latents --ckpt runs/vae_b1/vae.pt --data "data/v2/$d" \
          --suffix v1vae --device "$DEVICE"
    done
  fi
fi

# --------------------------------------------------------------------- v3
if want v3; then
  banner "v3 — the occlusion band (7 splits, ~63 s, 1.4 GB)"
  C="$PY -m worldsim.collect --ball-radius 0.08 --steps 200 --res 64 --occluder"
  $C --out data/v3/train     --episodes 150 --seed 0   --policy sticky
  $C --out data/v3/train_mix --episodes 300 --seed 10  --policy mix --p-track 0.5
  $C --out data/v3/val       --episodes 15  --seed 1   --policy sticky
  $C --out data/v3/val_mix   --episodes 20  --seed 11  --policy mix
  $PY -m worldsim.collect --out data/v3/probe --episodes 120 --seed 777 --policy sticky \
      --steps 24 --ball-radius 0.08 --res 64 --occluder
  # Two taller bands, for the memory-horizon test.
  $C --out data/v3/tall      --episodes 30  --seed 31  --policy mix --occluder-y 0.22 0.64
  $C --out data/v3/taller    --episodes 30  --seed 32  --policy mix --occluder-y 0.16 0.70

  if [ "$LATENTS" = 1 ]; then
    banner "v3 latents (encoder runs/vae_v3/vae.pt)"
    for d in train train_mix val val_mix probe tall taller; do
      $PY -m wm.cache_latents --ckpt runs/vae_v3/vae.pt --data "data/v3/$d" --device "$DEVICE"
    done
  fi
fi

# -------------------------------------------------------------------- v3.1
if want v31; then
  banner "v3.1 — the re-designed band (9 splits, ~75 s, 1.9 GB)"
  # The nine calls live in the script that produced the published data.
  bash runs/v31_env/collect_v31.sh

  if [ "$LATENTS" = 1 ]; then
    banner "v3.1 latents (encoder runs/vae_v31/vae.pt)"
    for d in train train_mix train_short train_long val val_mix probe short long; do
      $PY -m wm.cache_latents --ckpt runs/vae_v31/vae.pt --data "data/v31/$d" --device "$DEVICE"
    done
  fi
fi

# --------------------------------------------------------------------- v4
if want v4; then
  banner "v4 — the gravity switch (6 splits, ~2 min)"
  bash runs/v4_env/collect_v4.sh

  if [ "$LATENTS" = 1 ]; then
    banner "v4 latents (encoder runs/vae_v4/vae.pt)"
    for d in train train_mix val val_mix probe long; do
      $PY -m wm.cache_latents --ckpt runs/vae_v4/vae.pt --data "data/v4/$d" --device "$DEVICE"
    done
  fi
fi

banner "done"
echo "Check with: python -m pytest -q   (the byte-identity replays now run)"

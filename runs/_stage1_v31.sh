#!/bin/bash
# v3.1 stage one, after the VAE: analysis, the by-band probes, latent caching.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
CK=runs/vae_v31/vae.pt

# The version-independent half (active units, per-dim KL, tuning, MCC, the
# unconditional probe table) on the default-band probe split.
$PY -m wm.analyze --ckpt $CK --data data/v31/probe --device mps \
    --out runs/vae_v31/analysis

# The occlusion-conditioned half, once per band. v3 could only do this for the
# band the encoder was trained on; v3.1's encoder saw all three, so the three
# reports are directly comparable -- which is the point.
$PY -m wm.analyze_v3 --ckpt $CK --data data/v31/probe --device mps \
    --out runs/vae_v31/analysis \
    --band-roots data/v31/probe data/v31/short data/v31/long
$PY -m wm.analyze_v3 --ckpt $CK --data data/v31/short --device mps \
    --out runs/vae_v31/analysis_short
$PY -m wm.analyze_v3 --ckpt $CK --data data/v31/long  --device mps \
    --out runs/vae_v31/analysis_long

for d in train train_mix train_short train_long val val_mix probe short long; do
  echo "=== caching $d"
  $PY -m wm.cache_latents --ckpt $CK --data data/v31/$d --device mps
done
echo "=== stage one done"

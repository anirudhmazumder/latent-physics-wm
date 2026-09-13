#!/bin/bash
# v4 stage one, after the VAE: analysis, the sign probe, latent caching.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
CK=runs/vae_v4/vae.pt

# The version-independent half (active units, per-dim KL, tuning maps, prior
# samples, MCC, the unconditional probe table -- gravity_sign is just another
# state column there, so it gets an R^2 for free).
$PY -m wm.analyze --ckpt $CK --data data/v4/probe --device mps \
    --out runs/vae_v4/analysis

# The v4 half: the sign as a BIT rather than a number. Logistic + kNN accuracy
# against the majority-class baseline, the position-distribution shift that is
# the likeliest leak, and the same probes on a position-matched subsample that
# closes it. Run on `probe` (120 short episodes -> 120 near-independent labels)
# and again on `val_mix`, whose 2 flips/episode mean the sign varies WITHIN an
# episode -- a harder and more honest test of the episode-level split.
$PY -m wm.analyze_v4 --ckpt $CK --data data/v4/probe --device mps \
    --out runs/vae_v4/analysis
$PY -m wm.analyze_v4 --ckpt $CK --data data/v4/val_mix --device mps \
    --out runs/vae_v4/analysis_valmix

for d in train train_mix val val_mix probe long; do
  echo "=== caching $d"
  $PY -m wm.cache_latents --ckpt $CK --data data/v4/$d --device mps
done
echo "=== stage one done"

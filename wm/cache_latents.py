"""Freeze the VAE and cache its latents. The handoff to stage two.

    python -m wm.cache_latents --ckpt runs/vae/vae.pt --data data/v1/train

Writes ``mu.npy`` and ``logvar.npy`` of shape (E, T+1, z_dim) into the dataset
directory.

This is the single best practical decision in the pipeline. A 20k-frame dataset
is 245 MB of pixels and about 2.5 MB of latents -- roughly 100x smaller. Once
cached, the dynamics model trains entirely on latents and never runs the
encoder, which means a GRU-vs-transformer comparison costs seconds per epoch
instead of minutes. On 8 GB of RAM the entire cached dataset sits comfortably in
memory with room for the model.

Two things to get right at consumption time:

1. SAMPLE, do not use the mean. During dynamics training draw
   ``z = mu + eps * exp(0.5 * logvar)`` fresh on every access from the cached
   parameters. Two reasons. It is free stochastic regularisation, and more
   importantly it matches dream time: during a rollout the dynamics model
   receives latents it produced itself, which are draws from a distribution, not
   clean posterior means. Train on means and you get a model that has never seen
   the input distribution it will face at inference.

2. Do NOT fine-tune the VAE jointly with the dynamics model. It is tempting and
   it is where this gets confusing -- the dynamics loss starts reshaping the
   latent space, and a bad rollout becomes impossible to attribute. Was it bad
   dynamics or a degraded encoder? Keeping the encoder frozen means every
   stage-two result is attributable to stage two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from .analyze import load_ckpt


@torch.no_grad()
def cache(
    ckpt: str | Path,
    data: str | Path,
    device: str = "cpu",
    batch_size: int = 256,
) -> Path:
    model, cfg, _ = load_ckpt(ckpt, device)
    root = Path(data)
    frames = np.load(root / "frames.npy", mmap_mode="r")
    E, Tp1 = frames.shape[0], frames.shape[1]

    mu_out = open_memmap(
        root / "mu.npy", mode="w+", dtype=np.float32, shape=(E, Tp1, cfg.z_dim)
    )
    lv_out = open_memmap(
        root / "logvar.npy", mode="w+", dtype=np.float32, shape=(E, Tp1, cfg.z_dim)
    )

    for e in range(E):
        # One episode at a time, chunked. Keeps peak memory at a few hundred
        # frames regardless of dataset size.
        for t0 in range(0, Tp1, batch_size):
            t1 = min(t0 + batch_size, Tp1)
            x = np.array(frames[e, t0:t1], dtype=np.float32) / 255.0
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
            mu, logvar = model.encode(x)
            mu_out[e, t0:t1] = mu.cpu().numpy()
            lv_out[e, t0:t1] = logvar.cpu().numpy()
        if (e + 1) % max(1, E // 10) == 0:
            print(f"  {e + 1}/{E} episodes", flush=True)

    mu_out.flush()
    lv_out.flush()

    meta_path = root / "latent_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "ckpt": str(Path(ckpt).resolve()),
                "z_dim": cfg.z_dim,
                "shape": [E, Tp1, cfg.z_dim],
                "note": "sample z = mu + eps*exp(0.5*logvar) at train time",
            },
            indent=2,
        )
    )
    mb = mu_out.nbytes / 1e6
    print(f"cached ({E}, {Tp1}, {cfg.z_dim}) latents -> {root}  ({2 * mb:.1f} MB)")
    return root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    a = p.parse_args()
    cache(a.ckpt, a.data, device=a.device, batch_size=a.batch_size)


if __name__ == "__main__":
    main()

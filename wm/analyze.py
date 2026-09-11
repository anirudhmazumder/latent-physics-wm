"""Analyse a trained VAE: what did the latent space actually learn?

    python -m wm.analyze --ckpt runs/vae/vae.pt --data data/v1/val --out runs/vae/analysis

Writes recon grid, traversals, tuning maps, prior samples, and a JSON of every
probe/MCC number.

Run this on a val set with MANY SHORT episodes rather than few long ones. Frames
within an episode are heavily autocorrelated, so 8 episodes of 150 frames gives
you far less statistical power than 60 episodes of 20 frames, at identical cost:

    python -m worldsim.collect --out data/v1/probe --episodes 120 --steps 24 \
        --ball-radius 0.08 --seed 777
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import FrameDataset, make_loader
from .diagnostics import (
    active_units,
    encode_dataset,
    latent_traversal,
    reconstruction_grid,
    save_png,
)
from .probes import mcc, probe_suite, save_tuning_grid, spearman_mcc, tuning_maps
from .vae import ConvVAE, VAEConfig


def load_ckpt(path: str | Path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = VAEConfig(**{k: v for k, v in ck["cfg"].items() if k in VAEConfig.__annotations__})
    model = ConvVAE(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    free_bits = float(ck.get("args", {}).get("free_bits", 0.0))
    return model, cfg, free_bits


@torch.no_grad()
def prior_samples(model, n: int = 16, device: str = "cpu", pad: int = 2) -> np.ndarray:
    """Decode z ~ N(0, I).

    Why this matters for what comes next: your dynamics model will predict
    latents that are only approximately right, and the decoder has to render
    them into something sensible. Prior samples test exactly that -- whether the
    decoder behaves on points drawn from the prior rather than from the
    aggregate posterior. If prior samples look like plausible frames, dream
    rollouts have room to be slightly wrong and survive. If they look like
    noise, the aggregate posterior has not matched the prior and your rollouts
    will fall apart within a few dozen steps no matter how good the dynamics
    model is.
    """
    z = torch.randn(n, model.cfg.z_dim, device=device)
    imgs = model.decode(z).cpu().clamp(0, 1).numpy()
    imgs = (imgs.transpose(0, 2, 3, 1) * 255 + 0.5).astype(np.uint8)
    h = w = imgs.shape[1]
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    out = np.full((nrows * (h + pad) - pad, ncols * (w + pad) - pad, 3), 40, np.uint8)
    for i in range(n):
        r, c = divmod(i, ncols)
        out[r * (h + pad) : r * (h + pad) + h, c * (w + pad) : c * (w + pad) + w] = imgs[i]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="dataset to probe on (use a val/probe set)")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out or (Path(a.ckpt).parent / "analysis"))
    out.mkdir(parents=True, exist_ok=True)

    model, cfg, free_bits = load_ckpt(a.ckpt, a.device)
    ds = FrameDataset(a.data)
    loader = make_loader(a.data, 256, shuffle=False, return_state=True)

    print(f"z_dim={cfg.z_dim}  free_bits={free_bits}  frames={len(loader.dataset)}")

    enc = encode_dataset(model, loader, device=a.device, limit=a.limit)
    mu, logvar, state = enc["mu"], enc["logvar"], enc["state"]
    names = list(ds.state_names)

    # Episode id per frame, so probe splits are made at the episode level and
    # autocorrelated neighbours cannot straddle the train/test boundary.
    Tp1 = ds.Tp1
    group_ids = np.arange(len(mu)) // Tp1

    # ---------------------------------------------------------- active units
    au = active_units(logvar, mu, free_bits=free_bits)
    print(f"\nactive units: {au['n_active']}/{cfg.z_dim}")
    print("  per-dim KL (sorted): " +
          " ".join(f"{au['kl_per_dim'][d]:.2f}" for d in au["order"]))
    dims = [int(d) for d in au["order"][: max(au["n_active"], 6)]]

    # ---------------------------------------------------------------- probes
    print("\nheld-out R^2 (episode-level split):")
    suite = probe_suite(mu, state, names, group_ids=group_ids, seed=a.seed)
    hdr = f"{'factor':10s}" + "".join(f"{k:>9s}" for k in suite)
    print("  " + hdr)
    for name in names:
        row = f"  {name:10s}" + "".join(f"{suite[k][name]:9.3f}" for k in suite)
        print(row)

    # ------------------------------------------------------------------- MCC
    m_p = mcc(mu, state, names)
    m_s = spearman_mcc(mu, state, names)
    print(f"\nMCC pearson={m_p['mcc']:.3f}  spearman={m_s['mcc']:.3f}")
    for name in names:
        d, c = m_p["matched"][name]
        ds_, cs = m_s["matched"][name]
        print(f"  {name:10s} -> z[{d:2d}] |r|={c:.3f}   (rank: z[{ds_:2d}] |rho|={cs:.3f})")

    # --------------------------------------------------------------- visuals
    x_fixed, _ = next(iter(loader))
    x_fixed = x_fixed[:8]
    save_png(reconstruction_grid(model, x_fixed, device=a.device), out / "recon.png")
    save_png(
        latent_traversal(model, x_fixed, dims, device=a.device), out / "traversal.png",
        scale=2,
    )
    save_png(prior_samples(model, 16, device=a.device), out / "prior_samples.png", scale=2)

    maps, counts = tuning_maps(mu, state, dims, bins=a.bins)
    try:
        save_tuning_grid(maps, dims, out / "tuning_maps.png")
    except ImportError:
        print("  (matplotlib not installed; skipping tuning map render)")
    np.save(out / "tuning_maps.npy", maps)

    # Coverage check: an empty corner in the histogram means the probe never saw
    # that region, and every R^2 above is an extrapolation there.
    frac_covered = float((counts >= 3).mean())
    print(f"\nball-position bin coverage: {frac_covered:.1%} of {a.bins}x{a.bins} bins")

    (out / "report.json").write_text(
        json.dumps(
            {
                "z_dim": cfg.z_dim,
                "free_bits": free_bits,
                "n_frames": int(len(mu)),
                "n_episodes": int(group_ids.max() + 1),
                "active_units": au["n_active"],
                "kl_per_dim": au["kl_per_dim"].tolist(),
                "probes": suite,
                "mcc_pearson": m_p["mcc"],
                "mcc_spearman": m_s["mcc"],
                "matched_pearson": {k: [v[0], v[1]] for k, v in m_p["matched"].items()},
                "bin_coverage": frac_covered,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

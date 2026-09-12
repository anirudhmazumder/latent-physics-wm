"""Train the conv VAE.

    python -m wm.train_vae --data data/v1/train --val data/v1/val --epochs 30

Runs the diagnostics from wm.diagnostics every few epochs and writes PNGs, so
you are watching reconstructions and probe R^2 rather than a loss curve. On this
data the loss curve is close to uninformative.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import FrameDataset, make_loader
from .diagnostics import (
    active_units,
    encode_dataset,
    latent_traversal,
    linear_probe,
    reconstruction_grid,
    save_png,
)
from .vae import ConvVAE, VAEConfig, masked_recon_error, vae_loss


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ball_mask(states: torch.Tensor, res: int, radius: float, slack: float = 1.6):
    """Boolean (B, res, res) box around the true ball position.

    Uses ground-truth state, which is legitimate here: this is a diagnostic, not
    a training signal.
    """
    bx, by = states[:, 0], states[:, 1]
    js = (torch.arange(res, device=states.device) + 0.5) / res           # x
    is_ = 1.0 - (torch.arange(res, device=states.device) + 0.5) / res    # y, row 0 top
    dx = (js[None, None, :] - bx[:, None, None]).abs()
    dy = (is_[None, :, None] - by[:, None, None]).abs()
    return (dx < radius * slack) & (dy < radius * slack)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # Several roots are allowed and are concatenated (see wm/data.py). v3.1
    # trains one encoder on all three occlusion-band heights so that the
    # taller/shorter-band results downstream are not confounded by an
    # out-of-distribution encoder, which is what spoiled v3's.
    p.add_argument("--data", required=True, nargs="+")
    p.add_argument("--val", default=None, nargs="+")
    p.add_argument("--out", default="runs/vae")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--base-ch", type=int, default=16)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=float, default=5.0)
    p.add_argument("--free-bits", type=float, default=0.5,
                   help="nats per latent dim exempt from KL pressure. 0 "
                        "reliably posterior-collapses on this data.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        a.data, a.batch_size, shuffle=True, num_workers=a.num_workers, return_state=True
    )
    val_root = a.val or a.data
    val_loader = make_loader(
        val_root, a.batch_size, shuffle=False, num_workers=0, return_state=True
    )
    ds = FrameDataset(a.data)
    res = ds.frames.shape[2]
    ball_r = ds.meta["config"]["ball_radius"]
    print(f"device={device}  train_frames={len(train_loader.dataset)}  res={res}")
    if len(ds.roots) > 1:
        print("  train roots: " + "  ".join(
            f"{r.name}({e}x{t})" for r, e, t in zip(ds.roots, ds.Es, ds.Tp1s)))

    model = ConvVAE(VAEConfig(z_dim=a.z_dim, base_ch=a.base_ch, res=res)).to(device)
    n_par = sum(p_.numel() for p_ in model.parameters())
    print(f"params={n_par/1e3:.0f}k")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    # A fixed validation batch, so every reconstruction grid you look at shows
    # the same frames and you can see progress rather than resampling noise.
    fixed_x, fixed_s = next(iter(val_loader))
    fixed_x, fixed_s = fixed_x[:8], fixed_s[:8]

    steps_per_epoch = max(1, len(train_loader))
    history = []
    t0 = time.time()

    for epoch in range(a.epochs):
        model.train()
        agg = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "ball_mse": 0.0, "n": 0}

        for it, (x, s) in enumerate(train_loader):
            x, s = x.to(device), s.to(device)

            # Linear KL warmup. Optional but reliably prevents the early
            # posterior collapse where the encoder gives up before the decoder
            # is good enough for latent information to pay for its KL cost.
            frac = (epoch + it / steps_per_epoch) / max(a.warmup_epochs, 1e-9)
            beta = a.beta * min(1.0, frac)

            x_hat, mu, logvar = model(x)
            loss, parts = vae_loss(
                x_hat, x, mu, logvar, beta=beta, free_bits=a.free_bits
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Not usually needed here, but free, and it turns a rare NaN into a
            # slightly bad step instead of a dead run.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()

            with torch.no_grad():
                bm = ball_mask(s, res, ball_r)
                agg["ball_mse"] += masked_recon_error(x_hat, x, bm).mean().item() * len(x)
            agg["loss"] += parts["loss"].item() * len(x)
            agg["recon"] += parts["recon"].item() * len(x)
            agg["kl"] += parts["kl"].item() * len(x)
            agg["n"] += len(x)

        n = agg["n"]
        # Global per-pixel MSE for comparison against the ball-region MSE. The
        # gap between these two numbers is the whole point of tracking both.
        global_mse = agg["recon"] / n / (3 * res * res)
        row = {
            "epoch": epoch,
            "beta": beta,
            "loss": agg["loss"] / n,
            "recon_sum": agg["recon"] / n,
            "global_mse": global_mse,
            "ball_mse": agg["ball_mse"] / n,
            "kl": agg["kl"] / n,
            "secs": time.time() - t0,
        }
        history.append(row)
        print(
            f"ep {epoch:3d}  beta {beta:.2f}  loss {row['loss']:8.2f}  "
            f"kl {row['kl']:6.2f}  global_mse {global_mse:.5f}  "
            f"ball_mse {row['ball_mse']:.5f}  ({row['secs']:.0f}s)",
            flush=True,
        )

        if (epoch + 1) % a.eval_every == 0 or epoch == a.epochs - 1:
            enc = encode_dataset(model, val_loader, device=device, limit=4000)
            au = active_units(
                enc["logvar"], enc["mu"], free_bits=a.free_bits
            )
            r2 = linear_probe(enc["mu"], enc["state"], ds.state_names)
            print(f"    active_units {au['n_active']}/{a.z_dim}   probe R^2: " +
                  "  ".join(f"{k}={v:.3f}" for k, v in r2.items()))

            save_png(
                reconstruction_grid(model, fixed_x, device=device),
                out / f"recon_ep{epoch:03d}.png",
            )
            top = [int(d) for d in au["order"][: max(au["n_active"], 4)]]
            save_png(
                latent_traversal(model, fixed_x, top, device=device),
                out / f"traversal_ep{epoch:03d}.png",
                scale=2,
            )
            row["active_units"] = au["n_active"]
            row["probe_r2"] = r2
            torch.save(
                {"model": model.state_dict(), "cfg": vars(model.cfg),
                 "args": vars(a), "epoch": epoch},
                out / "vae.pt",
            )

    torch.save(
        {"model": model.state_dict(), "cfg": vars(model.cfg), "args": vars(a)},
        out / "vae.pt",
    )
    (out / "history.json").write_text(json.dumps(history, indent=2, default=float))
    print(f"\nsaved {out/'vae.pt'}")


if __name__ == "__main__":
    main()

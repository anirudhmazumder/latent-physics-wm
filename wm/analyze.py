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
from .probes import (
    mass_tuning,
    mcc,
    probe_suite,
    probe_transfer,
    save_mass_tuning,
    save_tuning_grid,
    spearman_mcc,
    tuning_maps,
)
from .vae import ConvVAE, VAEConfig


def add_derived_targets(state: np.ndarray, names):
    """Append derived probe targets to the state matrix. v2 only.

    Returns ``(state_ext, names_ext)``. Two extra columns, both only defined
    when the dataset has a ``mass`` column:

    ``log_mass``
        The colour ramp is linear in u = log(m) rescaled, so log_mass is the
        variable the *pixels* are actually linear in. Probing raw mass as well
        as log mass separates "the model coded the colour" (log_mass decodes)
        from "the model coded the physical mass" (mass decodes) -- they differ
        by a monotone warp, and a linear probe can tell them apart even though
        kNN cannot.

    ``speed``
        |v|, which in v2 is exactly ball_speed / mass. In v1 speed was constant
        and probing it was meaningless. In v2 it is the first *dynamical*
        quantity that is decodable from a single frame -- not because the frame
        shows motion (it does not; ball_vx and ball_vy stay undecodable) but
        because the frame shows COLOUR and colour determines speed. That gap --
        speed decodable, its components not -- is the appearance -> dynamics
        edge showing up already at stage V, and it is the single most
        interesting number in the v2 probe table.
    """
    names = list(names)
    if "mass" not in names:
        return np.asarray(state), names
    m = np.asarray(state)[:, names.index("mass")]
    vx, vy = np.asarray(state)[:, 2], np.asarray(state)[:, 3]
    extra = np.stack([np.log(m), np.hypot(vx, vy)], axis=1)
    return np.concatenate([state, extra], axis=1), names + ["log_mass", "speed"]


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


@torch.no_grad()
def color_fidelity(model, ds, names, device: str = "cpu", n: int = 16, pad: int = 2):
    """v2: does the decoder put the ball back in the RIGHT COLOUR?

    The routine reconstruction grid cannot answer this, because its eight frames
    come from one episode and therefore share one mass. So here we take ``n``
    episodes spanning the whole mass range, reconstruct one frame from each, and
    read the mass straight back out of the decoded pixels with
    ``worldsim.color_to_mass``.

    Reading the colour needs a little care: the reconstruction is blurry, so the
    ball's pixels are blends of ball colour and background and the *average*
    ball pixel is systematically dark. We take the single pixel furthest from
    the background colour -- the most ball-like pixel -- and let
    ``color_to_mass`` project it onto the ramp. That projection is exactly why
    the inverse map was written as a projection rather than a channel inversion.

    Returns ``(grid_image, true_mass, decoded_mass)``.
    """
    from worldsim.bouncing_box import BoxConfig, color_to_mass

    j_mass = names.index("mass")
    cfg = BoxConfig(**{k: v for k, v in ds.meta["config"].items()
                       if k in BoxConfig.__annotations__})
    mass_all = np.asarray(ds.states[:, 0, j_mass])
    order = np.argsort(mass_all)
    eps = order[np.linspace(0, len(order) - 1, min(n, len(order))).astype(int)]
    t = ds.Tp1 // 2

    x = np.stack([np.asarray(ds.frames[e, t], np.float32) / 255.0 for e in eps])
    xb = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
    mu, _ = model.encode(xb)
    xh = model.decode(mu).cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()

    bg = np.asarray(cfg.bg_color, np.float64) / 255.0
    res = x.shape[1]
    true_m, dec_m = [], []
    for k, e in enumerate(eps):
        bx, by = ds.states[e, t, 0], ds.states[e, t, 1]
        j0, i0 = int(bx * res), int((1.0 - by) * res)
        r = max(2, int(cfg.ball_radius * res))
        sl = (slice(max(i0 - r, 0), i0 + r + 1), slice(max(j0 - r, 0), j0 + r + 1))
        patch = xh[k][sl].reshape(-1, 3)
        px = patch[np.argmax(((patch - bg) ** 2).sum(-1))] * 255.0
        true_m.append(float(mass_all[e]))
        dec_m.append(color_to_mass(px, cfg))

    imgs = (np.concatenate([x, xh], 0) * 255 + 0.5).astype(np.uint8)
    m = len(eps)
    grid = np.full((2 * res + pad, m * (res + pad) - pad, 3), 40, np.uint8)
    for k in range(m):
        x0 = k * (res + pad)
        grid[:res, x0 : x0 + res] = imgs[k]
        grid[res + pad :, x0 : x0 + res] = imgs[m + k]
    return grid, np.array(true_m), np.array(dec_m)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="dataset to probe on (use a val/probe set)")
    p.add_argument("--eval-data", default=None,
                   help="second dataset: probes are FIT on --data and SCORED "
                        "here. Use it for the v2 mass hold-out (fit on "
                        "data/v2/probe, score on data/v2/holdout) to ask "
                        "whether the latent code generalises to unseen colours.")
    p.add_argument("--tuning-data", default=None,
                   help="v2: compute mass_tuning.png on this split instead of "
                        "--data. Long episodes are best here -- averaging mu "
                        "within a mass bin only works once position has "
                        "averaged out.")
    p.add_argument("--tuning-limit", type=int, default=40000)
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
    # state_names comes from meta.json, so a v2 dataset (7 columns, mass last)
    # flows through here with no special-casing beyond the derived targets.
    names_raw = list(ds.state_names)
    state, names = add_derived_targets(state, names_raw)
    has_mass = "mass" in names_raw

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
    # MCC is computed on the RAW state columns only. It matches factors to
    # latents one-to-one, so feeding it redundant derived columns (log_mass is
    # a monotone function of mass, speed is 1/mass up to a constant) would
    # force three different latents to claim what is really one factor and
    # would deflate the score for reasons that have nothing to do with the model.
    n_raw = len(names_raw)
    m_p = mcc(mu, state[:, :n_raw], names_raw)
    m_s = spearman_mcc(mu, state[:, :n_raw], names_raw)
    print(f"\nMCC pearson={m_p['mcc']:.3f}  spearman={m_s['mcc']:.3f}")
    for name in names_raw:
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

    # ------------------------------------------------- v2: tuning to colour
    mass_curves = None
    colour_stats = None
    if has_mass:
        grid, m_true, m_dec = color_fidelity(model, ds, names_raw, device=a.device)
        save_png(grid, out / "recon_by_mass.png", scale=3)
        rel = np.abs(m_dec - m_true) / m_true
        # Spearman rather than Pearson: the question is whether the ORDER of the
        # colours survived the bottleneck. A decoder that renders every ball
        # slightly too pale still gets the ordering right and is fine; one that
        # renders them all the same colour scores ~0 and is a v2 failure.
        from scipy.stats import spearmanr

        rho = float(spearmanr(m_true, m_dec).correlation)
        colour_stats = {
            "n": int(len(m_true)),
            "median_rel_err": float(np.median(rel)),
            "max_rel_err": float(rel.max()),
            "spearman": rho,
            "true": m_true.tolist(),
            "decoded": m_dec.tolist(),
        }
        print(f"\ncolour fidelity (mass read back out of the reconstruction, "
              f"{len(m_true)} episodes): median |dm|/m = {np.median(rel):.3f}  "
              f"max {rel.max():.3f}  spearman {rho:.3f}")
        # The tuning curve is a mean of mu within a mass bin, so everything mu
        # encodes that is NOT mass -- ball and paddle position, which carry far
        # more variance -- has to average out. On the probe set (120 episodes x
        # 24 frames) it does not average out nearly enough and the curves come
        # out visibly noisy. --tuning-data computes this one plot on a larger
        # split; LONG episodes are ideal here, which is the exact opposite of
        # what you want for the probes, so it is a separate dataset argument.
        mu_tune = mu
        mcol = state[:, names_raw.index("mass")]
        if a.tuning_data:
            enc_t = encode_dataset(
                model,
                make_loader(a.tuning_data, 256, shuffle=False, return_state=True),
                device=a.device,
                limit=a.tuning_limit,
            )
            names_t = list(FrameDataset(a.tuning_data).state_names)
            mu_tune = enc_t["mu"]
            mcol = enc_t["state"][:, names_t.index("mass")]
            print(f"\n(mass tuning computed on {a.tuning_data}: "
                  f"{len(mu_tune)} frames)")
        centres, mass_curves, mcounts = mass_tuning(mu_tune, mcol, dims, bins=12)
        band = ds.meta.get("mass_holdout")
        try:
            save_mass_tuning(centres, mass_curves, dims, out / "mass_tuning.png",
                             holdout=band)
        except ImportError:
            print("  (matplotlib not installed; skipping mass tuning render)")
        # Which dimension moves most across the mass range? That is the
        # candidate "colour dimension", and the number to compare against the
        # same dimension's spread over positions.
        # Which dimensions actually CARRY the colour? A tuning curve can look
        # like it slopes for the wrong reason (z[7] codes paddle_x here, and
        # its bin means wobble because paddle position has not averaged out),
        # so the decisive measure is a probe restricted to one dimension at a
        # time: held-out R^2 for log_mass from z[d] alone. Read it against the
        # all-dimensions number: if no single dim comes close, the colour code
        # is distributed, not axis-aligned.
        j_lm = names.index("log_mass")
        per_dim = {}
        for d in dims:
            r = probe_suite(mu[:, [d]], state[:, [j_lm]], ["log_mass"],
                            group_ids=group_ids, seed=a.seed,
                            which=("poly2",))["poly2"]["log_mass"]
            per_dim[int(d)] = float(r)
        rank = sorted(per_dim.items(), key=lambda kv: -kv[1])
        print("  log_mass from ONE dim (poly2 R^2): " +
              "  ".join(f"z[{d}]={v:.3f}" for d, v in rank[:5]))

        spread = np.nanmax(mass_curves, axis=1) - np.nanmin(mass_curves, axis=1)
        k = int(np.argmax(spread))
        print(f"\nmass tuning: strongest dim is z[{dims[k]}] "
              f"(mean mu swings {spread[k]:.2f} across the mass range); "
              "next: " + ", ".join(
                  f"z[{dims[i]}]={spread[i]:.2f}"
                  for i in np.argsort(-spread)[1:4]))

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

    # --------------------------------------------------- transfer / hold-out
    transfer = None
    if a.eval_data:
        ds2 = FrameDataset(a.eval_data)
        loader2 = make_loader(a.eval_data, 256, shuffle=False, return_state=True)
        enc2 = encode_dataset(model, loader2, device=a.device, limit=a.limit)
        s2, names2 = add_derived_targets(enc2["state"], list(ds2.state_names))
        if names2 != names:
            raise SystemExit("--eval-data has different state_names than --data")
        print(f"\nprobe transfer: fit on {a.data} ({len(mu)} frames), "
              f"evaluated on {a.eval_data} ({len(enc2['mu'])} frames)")
        if has_mass:
            print(f"  fit masses  {state[:, names_raw.index('mass')].min():.2f}"
                  f"..{state[:, names_raw.index('mass')].max():.2f}   "
                  f"eval masses {s2[:, names_raw.index('mass')].min():.2f}"
                  f"..{s2[:, names_raw.index('mass')].max():.2f}")
        transfer = probe_transfer(mu, state, enc2["mu"], s2, names, seed=a.seed)

        # R^2 on the eval set is measured against the EVAL set's own variance,
        # and for the mass hold-out that variance is tiny by construction (the
        # band is 0.85..1.2 out of a 0.5..2.0 range). So a probe that puts every
        # held-out ball in roughly the right part of the mass range but cannot
        # resolve *within* the band scores hugely negative, which reads as
        # "total failure" and is not what happened. The two extra columns fix
        # the framing: ``rmse`` is the error in the factor's own units, and
        # ``r2_wide`` rescores that same error against the spread of the FIT
        # set -- i.e. "is this prediction useful on the scale of the whole
        # world" rather than "on the scale of this narrow slice of it".
        cols = list(transfer)
        print("  " + f"{'factor':10s}" + "".join(f"{k:>9s}" for k in cols)
              + f"{'rmse':>10s}{'r2_wide':>9s}")
        extra = {}
        for j, name in enumerate(names):
            sd_eval = float(np.std(s2[:, j]))
            sd_fit = float(np.std(state[:, j])) or 1.0
            # r2 = 1 - mse/var_eval  =>  rmse = sd_eval * sqrt(1 - r2)
            r2_best = max(transfer[k][name] for k in cols)
            rmse = sd_eval * float(np.sqrt(max(0.0, 1.0 - r2_best)))
            r2_wide = 1.0 - (rmse / sd_fit) ** 2
            extra[name] = {"rmse_best": rmse, "r2_wide": r2_wide,
                           "sd_eval": sd_eval, "sd_fit": sd_fit}
            print(f"  {name:10s}" + "".join(f"{transfer[k][name]:9.3f}" for k in cols)
                  + f"{rmse:10.3f}{r2_wide:9.3f}")
        print("  (rmse / r2_wide use the BEST probe per row; r2_wide rescores "
              "that error against the fit set's spread)")
        transfer = {**transfer, "_abs": extra}

    (out / "report.json").write_text(
        json.dumps(
            {
                "z_dim": cfg.z_dim,
                "free_bits": free_bits,
                "n_frames": int(len(mu)),
                "n_episodes": int(group_ids.max() + 1),
                "active_units": au["n_active"],
                "kl_per_dim": au["kl_per_dim"].tolist(),
                "state_names": names,
                "probes": suite,
                "probes_transfer": transfer,
                "eval_data": a.eval_data,
                "mass_tuning": None if mass_curves is None else mass_curves.tolist(),
                "colour_fidelity": colour_stats,
                "log_mass_per_dim_poly2": None if mass_curves is None else per_dim,
                "tuning_dims": dims,
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

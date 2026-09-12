"""v3-specific VAE analysis: what does `z` know while the ball is hidden?

    python -m wm.analyze_v3 --ckpt runs/vae_v3/vae.pt --data data/v3/probe \
        --out runs/vae_v3/analysis --device mps

Run this AFTER ``wm.analyze``, which produces the version-independent half of
the picture (active units, per-dim KL, tuning maps, prior samples, MCC, and the
unconditional probe table -- note that ``ball_visible`` is just another state
column there, so it gets probed for free). This file adds the three things that
are only meaningful once part of the data is unobservable.

**1. Probes conditioned on visibility.** The headline. Split the frames into
fully visible (``ball_visible > 0.99``), partially occluded (0.01-0.99) and
fully hidden (< 0.01), and probe ``mu`` for ``ball_x`` / ``ball_y`` separately
in each. The expected answer is R^2 ~= 0.99 visible and ~= 0 hidden, and that
is a *success*, not a failure: on a hidden frame the image is pixel-identical
for every ball position (``tests/test_env_v3.py::test_band_hides_the_ball_completely``
asserts exactly that), so an encoder that scored well there would have to be
reading something it should not. Which is why the hidden-frame number is worth
computing rather than assuming: if it comes out clearly above 0, the likely
culprit is leakage at the threshold -- balls counted as "hidden" whose
antialiased fringe still pokes a pixel out from under the band edge -- and the
``ball_visible`` distribution inside the hidden bin tells you immediately.

**2. Hallucination rate.** Decode ``mu`` on fully hidden frames and measure how
much ball-coloured pixel mass appears *outside* the band. The failure mode this
catches is a decoder that, having learned "frames usually contain a ball",
paints one somewhere plausible when the evidence says there is none. Reported
against the same measure on fully visible frames (where it should be ~1 ball)
and against the REAL frames (which calibrates the measure itself).

**3. recon_by_visibility.png.** The same question by eye, for the case the
numbers cannot summarise: is a *partially* occluded ball reconstructed as a
partial ball in the right place, or is it smeared across the band edge?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from worldsim.bouncing_box import BoxConfig

from .analyze import load_ckpt
from .data import FrameDataset, make_loader
from .diagnostics import active_units, encode_dataset, save_png
from .probes import make_split, probe_suite

WHICH = ("linear", "poly2", "knn")
# The three visibility bins. Deliberately NOT a partition at 0/1: the 0.01 and
# 0.99 cuts leave a sliver of "barely occluded" and "barely visible" frames out
# of the visible and hidden bins, which is what keeps those two bins clean.
BINS = {
    "visible": lambda v: v > 0.99,
    "partial": lambda v: (v >= 0.01) & (v <= 0.99),
    "hidden": lambda v: v < 0.01,
}


# ------------------------------------------------------- the ball-mass measure


def ball_mass(imgs: np.ndarray, cfg: BoxConfig, resid_scale: float = 60.0,
              color: Optional[Sequence[float]] = None) -> np.ndarray:
    """"How many balls' worth of ball-coloured pixels is in each image."

    ``imgs`` is (N, H, W, 3) float in [0, 1]. Returns (N, H, W) "ballness" per
    pixel, in units where a pixel fully covered by the ball scores 1.

    The measure has to survive three distractors -- the near-black background,
    the grey-blue band and the blue paddle -- and one complication: VAE
    reconstructions are blurry, so a rendered ball is a *blend* of ball colour
    and whatever is behind it rather than the pure colour. So we model each
    pixel as exactly that blend,

        p ~= b + alpha * (ball - b)

    where ``b`` is the local background (the band colour inside the band, the
    background colour outside it), solve for alpha by projection, and then
    *reject the projection if the residual is large*. That second half is what
    stops the paddle counting: the paddle projects onto the bg->ball direction
    at alpha ~= 0.57, but leaves a residual of ~220 RGB units, so the Gaussian
    rejection term kills it. A blurry ball, by contrast, lies almost exactly on
    the line and keeps its alpha.

    ``color`` overrides which object is being looked for; None means the ball,
    which is every caller before v3.1's stage three. The same projection works
    for the paddle (``cfg.paddle_color``) because the argument above is
    symmetric in the two colours: the ball projects onto the bg->paddle
    direction at alpha ~= 0.47 and leaves a ~200-unit residual, so it is
    rejected by exactly the same Gaussian.
    """
    x = np.asarray(imgs, np.float64) * 255.0
    n, h, w, _ = x.shape
    ball = np.asarray(cfg.ball_color if color is None else color, np.float64)
    lo, hi = cfg.occluder_y
    ys = 1.0 - (np.arange(h) + 0.5) / h
    in_band = (ys >= lo) & (ys <= hi)

    b = np.where(
        in_band[None, :, None, None],
        np.asarray(cfg.occluder_color, np.float64),
        np.asarray(cfg.bg_color, np.float64),
    )
    d = ball - b                                   # (1, H, 1, 3), two distinct rows
    denom = (d * d).sum(-1)
    alpha = np.clip(((x - b) * d).sum(-1) / denom, 0.0, 1.0)
    resid = x - b - alpha[..., None] * d
    return alpha * np.exp(-(resid * resid).sum(-1) / resid_scale**2)


def outside_band_mass(imgs: np.ndarray, cfg: BoxConfig):
    """Ball-coloured pixel mass strictly outside the band, in units of one ball.

    Returns ``(total, peak)``: the summed mass per image, and the single
    brightest ball-like pixel outside the band. Both are needed, because they
    fail differently and only the pair distinguishes the two ways a hidden
    frame can score above zero:

    * a faint haze spread over hundreds of pixels (a blurry decoder) gives a
      small total and a *tiny* peak -- harmless;
    * an actual hallucinated ball gives a similar total concentrated in ~20
      pixels, and therefore a peak near 1.

    Rows whose pixel extent touches the band are dropped entirely rather than
    counted: the band's own edge is antialiased, so those rows are a blend and
    would contribute a small spurious signal on every single frame -- including
    frames with no ball in them at all, which is precisely the baseline this
    number is supposed to establish.
    """
    res = imgs.shape[1]
    lo, hi = cfg.occluder_y
    px = 1.0 / res
    ys = 1.0 - (np.arange(res) + 0.5) / res
    outside = (ys - px / 2 > hi) | (ys + px / 2 < lo)
    one_ball = np.pi * (cfg.ball_radius * res) ** 2
    m = ball_mass(imgs, cfg)[:, outside, :]
    return m.sum(axis=(1, 2)) / one_ball, m.reshape(len(m), -1).max(axis=1)


# ------------------------------------------------------------------ probing


def probe_by_visibility(
    mu: np.ndarray,
    state: np.ndarray,
    names: Sequence[str],
    group_ids: np.ndarray,
    targets=("ball_x", "ball_y"),
    seed: int = 0,
) -> Dict[str, dict]:
    """``probe_suite`` per visibility bin, with rmse in world units alongside R^2.

    Each bin is probed on its own frames with its own episode-level split, so
    the hidden-frame number is not propped up by visible frames leaking in
    through a shared training set. Bins with fewer than two distinct episodes
    are reported as unprobeable rather than silently scored on a degenerate
    split.

    ``rmse`` is recovered from R^2 as ``sd_test * sqrt(1 - R^2)``, using the
    *same* split ``probe_suite`` used -- ``make_split`` is a pure function of
    (n, seed, group_ids), so replaying it here reproduces the identical test
    indices rather than approximating them. Worth having: an R^2 of 0.0 on the
    hidden bin means "no better than predicting the mean", and only the rmse
    tells you how big an error that actually is (~0.26 in x, i.e. a quarter of
    the box).
    """
    j_vis = list(names).index("ball_visible")
    vis = state[:, j_vis]
    cols = [list(names).index(t) for t in targets]
    out: Dict[str, dict] = {}
    for label, test in BINS.items():
        m = test(vis)
        n_eps = len(np.unique(group_ids[m]))
        rec: dict = {
            "n_frames": int(m.sum()),
            "n_episodes": int(n_eps),
            "ball_visible_mean": float(vis[m].mean()) if m.any() else float("nan"),
            "ball_visible_max": float(vis[m].max()) if m.any() else float("nan"),
        }
        if n_eps < 4:
            rec["note"] = "too few episodes to split; not probed"
            out[label] = rec
            continue
        sub_groups = group_ids[m]
        suite = probe_suite(
            mu[m], state[m][:, cols], list(targets),
            group_ids=sub_groups, seed=seed, which=WHICH,
        )
        _, te = make_split(int(m.sum()), seed=seed, group_ids=sub_groups)
        rec["r2"] = {k: dict(v) for k, v in suite.items()}
        rec["rmse"] = {
            k: {
                t: float(
                    np.std(state[m][te, c]) * np.sqrt(max(0.0, 1.0 - suite[k][t]))
                )
                for t, c in zip(targets, cols)
            }
            for k in suite
        }
        rec["sd_test"] = {
            t: float(np.std(state[m][te, c])) for t, c in zip(targets, cols)
        }
        out[label] = rec
    return out


# ------------------------------------------------------------------ visuals


@torch.no_grad()
def recon_by_visibility(model, ds, mu, state, names, device="cpu", k=8, pad=2, seed=0):
    """Six rows: real/recon for k fully visible, k partial, k fully hidden frames.

    Frames are re-encoded from the dataset rather than decoded from the cached
    ``mu`` alone, so the top row of each pair is the literal input and the
    comparison is honest. The partial row is sorted by coverage so you can read
    the ramp left to right -- that ordering is the whole point of the panel:
    a correct decoder shows a ball being progressively eaten by the band edge,
    a confused one shows a ball that fades in place or jumps.
    """
    j_vis = list(names).index("ball_visible")
    vis = state[:, j_vis]
    rng = np.random.default_rng(seed)
    Tp1 = ds.Tp1

    rows = []
    labels = []
    for label, test in BINS.items():
        idx = np.where(test(vis))[0]
        if not len(idx):
            continue
        sel = rng.choice(idx, size=min(k, len(idx)), replace=False)
        sel = sel[np.argsort(-vis[sel])]
        x = np.stack([
            np.asarray(ds.frames[i // Tp1, i % Tp1], np.float32) / 255.0 for i in sel
        ])
        xb = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        m, _ = model.encode(xb)
        xh = model.decode(m).cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
        rows.append(x)
        rows.append(xh)
        labels.append((label, vis[sel]))

    res = rows[0].shape[1]
    ncol = max(len(r) for r in rows)
    H = len(rows) * (res + pad) - pad
    W = ncol * (res + pad) - pad
    grid = np.full((H, W, 3), 40, np.uint8)
    for r, imgs in enumerate(rows):
        for c in range(len(imgs)):
            tile = (imgs[c] * 255 + 0.5).astype(np.uint8)
            grid[r * (res + pad) : r * (res + pad) + res,
                 c * (res + pad) : c * (res + pad) + res] = tile
    return grid, labels


# ----------------------------------------------------- the band as a nuisance


def probe_band_top(
    model, roots: Sequence[str], device: str = "cpu", limit: int = 12000,
    seed: int = 0,
) -> Dict:
    """Can ``mu`` tell you where the top of the occluder is?

    New in v3.1, and it exists because v3.1's encoder is trained on THREE band
    heights instead of one. The band's top edge is therefore a genuine latent
    factor of the data -- a nuisance one: it is not the ball, the model is never
    asked about it, and nothing downstream uses it, but it is 65 rows of pixels
    that move between episodes, so the VAE has to spend code on it or eat the
    reconstruction cost. Two numbers:

    * **R^2**, treating the top edge as continuous. This is the "is it in there
      at all" number.
    * **3-way accuracy**, snapping the prediction to the nearest of the three
      values actually collected. This is the number that matters for the claim
      the design makes -- that the encoder can tell the bands apart -- because
      R^2 on three discrete levels is dominated by which level, not by how
      precisely each is placed.

    The split is by EPISODE and episodes are numbered across roots, so a probe
    cannot pass by memorising an episode it also trained on. Frames are taken
    in root-major order (``shuffle=False``), which is what lets the label be
    reconstructed from the flat index.
    """
    from .data import FrameDataset, make_loader

    ds = FrameDataset(roots, return_state=True)
    tops = np.array([m["occluder_y"][1] for m in ds.metas], float)
    if len(np.unique(tops)) < 2:
        return {"note": "all roots share one band; nothing to probe"}

    loader = make_loader(roots, 256, shuffle=False, return_state=True)
    enc = encode_dataset(model, loader, device=device, limit=limit)
    mu = enc["mu"]
    n = len(mu)
    loc = np.array([ds._locate(i)[:2] for i in range(n)])   # (root, episode)
    y = tops[loc[:, 0]]
    # Episode ids unique across roots, so the group split is honest.
    groups = loc[:, 0] * 10_000 + loc[:, 1]

    suite = probe_suite(mu, y[:, None], ["band_top"], group_ids=groups,
                        seed=seed, which=WHICH)
    tr, te = make_split(n, seed=seed, group_ids=groups)
    # Nearest-level accuracy needs the predictions themselves, and
    # ``probe_suite`` returns only scores -- so the two readouts are refit here
    # with a plain ridge on the SAME split, which is what makes the accuracy
    # and the R^2 two views of one fit rather than two experiments.
    acc = {}
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    levels = np.unique(tops)
    for k, feat in (("linear", None), ("poly2", 2)):
        X = mu
        if feat:
            X = PolynomialFeatures(feat, include_bias=False).fit_transform(mu)
        sc = StandardScaler().fit(X[tr])
        r = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        pred = r.predict(sc.transform(X[te]))
        snap = levels[np.abs(pred[:, None] - levels[None, :]).argmin(1)]
        acc[k] = float((snap == y[te]).mean())
    return {
        "roots": list(map(str, roots)),
        "levels": levels.tolist(),
        "n_frames": int(n),
        "n_test": int(len(te)),
        "majority_class_acc": float(
            max((y[te] == L).mean() for L in levels)),
        "r2": {k: suite[k]["band_top"] for k in WHICH},
        "nearest_level_acc": acc,
    }


# --------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="a v3 probe/val split")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--band-roots", nargs="*", default=None,
                   help="v3.1: two or more roots with DIFFERENT occluder "
                        "heights. Probes mu for the band's top edge -- the "
                        "nuisance factor an encoder trained on several bands "
                        "has to carry. Skipped when not given.")
    p.add_argument("--band-limit", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out or (Path(a.ckpt).parent / "analysis"))
    out.mkdir(parents=True, exist_ok=True)

    model, mcfg, free_bits = load_ckpt(a.ckpt, a.device)
    ds = FrameDataset(a.data)
    names = list(ds.state_names)
    if "ball_visible" not in names:
        raise SystemExit(f"{a.data} is not a v3 dataset (no ball_visible column)")
    cfg = BoxConfig(**{k: v for k, v in ds.meta["config"].items()
                       if k in BoxConfig.__annotations__})

    loader = make_loader(a.data, 256, shuffle=False, return_state=True)
    enc = encode_dataset(model, loader, device=a.device, limit=a.limit)
    mu, logvar, state = enc["mu"], enc["logvar"], enc["state"]
    group_ids = np.arange(len(mu)) // ds.Tp1
    vis = state[:, names.index("ball_visible")]

    print(f"z_dim={mcfg.z_dim}  free_bits={free_bits}  frames={len(mu)}  "
          f"band y={cfg.occluder_y[0]:.2f}..{cfg.occluder_y[1]:.2f}")
    for label, test in BINS.items():
        m = test(vis)
        print(f"  {label:8s} {m.sum():6d} frames ({m.mean():5.1%})  "
              f"{len(np.unique(group_ids[m])):3d} episodes")

    # ------------------------------------------------- active units / KL
    # Repeated from wm.analyze so this report stands on its own; same numbers.
    au = active_units(logvar, mu, free_bits=free_bits)
    print(f"\nactive units: {au['n_active']}/{mcfg.z_dim}")
    print("  per-dim KL (sorted): " +
          " ".join(f"{au['kl_per_dim'][d]:.2f}" for d in au["order"]))
    dims = [int(d) for d in au["order"][: max(au["n_active"], 6)]]

    # --------------------------------------------- probes by visibility
    by_vis = probe_by_visibility(mu, state, names, group_ids, seed=a.seed)
    print("\nball position from mu, conditioned on visibility "
          "(held-out R^2 / rmse in world units):")
    print(f"  {'bin':9s}{'frames':>7s}{'eps':>5s}" +
          "".join(f"{k + '.' + t:>16s}" for t in ("x", "y") for k in WHICH))
    for label, rec in by_vis.items():
        row = f"  {label:9s}{rec['n_frames']:7d}{rec['n_episodes']:5d}"
        if "r2" not in rec:
            print(row + "   " + rec["note"])
            continue
        for t in ("ball_x", "ball_y"):
            for k in WHICH:
                row += f"{rec['r2'][k][t]:9.3f}/{rec['rmse'][k][t]:.3f}"
        print(row)
    for label, rec in by_vis.items():
        if "r2" not in rec:
            continue
        best = max(rec["r2"][k][t] for k in WHICH for t in ("ball_x", "ball_y"))
        if label == "hidden" and best > 0.2:
            print(f"\n  !! hidden-frame R^2 is {best:.3f}, which should be ~0. "
                  f"ball_visible in this bin runs up to "
                  f"{rec['ball_visible_max']:.4f} (mean "
                  f"{rec['ball_visible_mean']:.4f}) -- if that is not ~0 the "
                  "bin is contaminated by barely-occluded balls; if it IS ~0 "
                  "the encoder is reading the ball's antialiased fringe from "
                  "under the band edge, which is a real finding.")

    # ------------------------------------- does any latent encode visibility?
    # ball_visible is a function of ball_y alone, so a model that codes ball_y
    # well gets it for free from the full mu -- the interesting question is
    # whether any SINGLE dimension tracks it, i.e. whether the VAE allocated an
    # axis to "is there a ball on screen".
    j_v = names.index("ball_visible")
    full = probe_suite(mu, state[:, [j_v]], ["ball_visible"],
                       group_ids=group_ids, seed=a.seed, which=WHICH)
    per_dim = {
        int(d): float(
            probe_suite(mu[:, [d]], state[:, [j_v]], ["ball_visible"],
                        group_ids=group_ids, seed=a.seed,
                        which=("poly2",))["poly2"]["ball_visible"]
        )
        for d in dims
    }
    rank = sorted(per_dim.items(), key=lambda kv: -kv[1])
    print("\nball_visible from mu: " +
          "  ".join(f"{k}={full[k]['ball_visible']:.3f}" for k in WHICH))
    print("  from ONE dim (poly2): " +
          "  ".join(f"z[{d}]={v:.3f}" for d, v in rank[:5]))

    # ------------------------------------------------ the band's own height
    band_probe = None
    if a.band_roots:
        band_probe = probe_band_top(model, a.band_roots, device=a.device,
                                    limit=a.band_limit, seed=a.seed)
        print("\nband top edge from mu (the nuisance factor v3.1 added)")
        if "note" in band_probe:
            print("  " + band_probe["note"])
        else:
            print(f"  levels {band_probe['levels']}  "
                  f"{band_probe['n_frames']} frames, "
                  f"{band_probe['n_test']} held out")
            print("  R^2: " + "  ".join(
                f"{k}={band_probe['r2'][k]:.3f}" for k in WHICH))
            print("  nearest-of-three accuracy: " + "  ".join(
                f"{k}={v:.3f}" for k, v in band_probe["nearest_level_acc"].items())
                + f"   (majority class {band_probe['majority_class_acc']:.3f})")

    # -------------------------------------------------- hallucination rate
    halluc = {}
    for label, test in BINS.items():
        m = test(vis)
        if not m.any():
            continue
        idx = np.where(m)[0]
        rng = np.random.default_rng(a.seed)
        idx = rng.choice(idx, size=min(512, len(idx)), replace=False)
        x = np.stack([
            np.asarray(ds.frames[i // ds.Tp1, i % ds.Tp1], np.float32) / 255.0
            for i in idx
        ])
        with torch.no_grad():
            xb = torch.from_numpy(x).permute(0, 3, 1, 2).to(a.device)
            m_, _ = model.encode(xb)
            xh = model.decode(m_).cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
        tot_h, peak_h = outside_band_mass(xh, cfg)
        tot_r, peak_r = outside_band_mass(x, cfg)
        halluc[label] = {
            "n": int(len(idx)),
            "recon": float(tot_h.mean()),
            "real": float(tot_r.mean()),
            "recon_peak_median": float(np.median(peak_h)),
            "recon_peak_max": float(peak_h.max()),
            "real_peak_median": float(np.median(peak_r)),
        }
    print("\nball-coloured pixel mass OUTSIDE the band, in units of one ball")
    print("  (the 'real' columns calibrate the measure: they are what the "
          "actual frames score)")
    print(f"  {'bin':9s}{'real':>9s}{'recon':>9s}{'real pk':>9s}"
          f"{'recon pk':>10s}{'pk max':>9s}")
    for label, h in halluc.items():
        print(f"  {label:9s}{h['real']:9.3f}{h['recon']:9.3f}"
              f"{h['real_peak_median']:9.3f}{h['recon_peak_median']:10.3f}"
              f"{h['recon_peak_max']:9.3f}")
    if "hidden" in halluc:
        hh = halluc["hidden"]
        print(f"\n  hallucination rate = {hh['recon']:.4f} balls of spurious "
              f"ball mass on fully hidden frames, against "
              f"{halluc.get('visible', {}).get('recon', float('nan')):.4f} "
              "on fully visible ones.")
        print(f"  brightest ball-like pixel on a hidden frame: "
              f"{hh['recon_peak_max']:.3f} (median {hh['recon_peak_median']:.3f}). "
              + ("Well under 1, so the residual is diffuse haze, not a ball "
                 "drawn in the wrong place."
                 if hh["recon_peak_max"] < 0.4 else
                 "Near 1: the decoder IS painting a ball on frames that have "
                 "none. Investigate before building M on this encoder."))

    # ------------------------------------------------------------- visuals
    grid, labels = recon_by_visibility(
        model, ds, mu, state, names, device=a.device, seed=a.seed
    )
    save_png(grid, out / "recon_by_visibility.png", scale=3)
    print(f"\nwrote {out/'recon_by_visibility.png'}  (row pairs real/recon: " +
          ", ".join(f"{lab} vis {v.min():.2f}-{v.max():.2f}" for lab, v in labels)
          + ")")

    (out / "report_v3.json").write_text(json.dumps(
        {
            "data": a.data,
            "z_dim": mcfg.z_dim,
            "free_bits": free_bits,
            "n_frames": int(len(mu)),
            "occluder_y": list(cfg.occluder_y),
            "active_units": au["n_active"],
            "kl_per_dim": au["kl_per_dim"].tolist(),
            "probe_by_visibility": by_vis,
            "ball_visible_probe": {k: full[k]["ball_visible"] for k in WHICH},
            "ball_visible_per_dim_poly2": per_dim,
            "hallucination": halluc,
            "band_top_probe": band_probe,
        },
        indent=2,
    ))
    print(f"wrote {out/'report_v3.json'}")


if __name__ == "__main__":
    main()

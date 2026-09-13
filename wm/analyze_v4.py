"""v4-specific VAE analysis: is the gravity sign REALLY invisible in one frame?

    python -m wm.analyze_v4 --ckpt runs/vae_v4/vae.pt --data data/v4/probe \
        --out runs/vae_v4/analysis --device mps

Run this AFTER ``wm.analyze``, which produces the version-independent half of
the picture (active units, per-dim KL, tuning maps, prior samples, MCC, and the
unconditional probe table -- ``gravity_sign`` is just another state column
there, so it gets an R² for free). This file exists because for v4 the R² is
the wrong instrument and the number it produces is the *most important number
in stage one*, so it deserves the right one.

**Why a classifier and not R².** ``gravity_sign`` is a single bit. A regression
probe's R² on a ±1 target has no readable null: 0 means "predicts the mean",
which for an unbalanced target is already better than guessing, and the value
you should compare against moves with the class balance. Accuracy against the
majority-class rate says both things at once and is the quantity the v4 design
document actually commits to ("accuracy ≈ 50%").

**What the expected answer is, and why a negative result is the point.** The
whole premise of v4 is that no single frame contains the sign. The ball is one
colour, there is no sprite, nothing in the image changes when the sign flips.
So V should look exactly like v1's, and sign accuracy from ``mu`` should sit at
chance. That is the v4 analogue of v1's "velocity is not in z" -- the negative
result that *defines the problem* and licenses every memory claim stage two
will make. If it fails, stage two cannot distinguish "M remembered the flip"
from "M re-read the sign off the current frame", and the tier is dead.

**The leak that is most likely, and how this file rules it in or out.** A
hidden *dynamical* variable can still shift the *distribution* of visible ones.
Under gravity pulling down the ball is slow near the ceiling and fast near the
floor, so it spends more of its time high; under gravity pulling up, the
reverse. The marginal of ``ball_y`` therefore differs by sign even though no
frame shows the sign -- and an encoder doing its job (coding position well)
hands a probe a free, entirely uninteresting route to above-chance accuracy.

So this file measures three things in order:

1. **the raw probe** -- logistic and kNN accuracy for the sign from ``mu``,
   split at the EPISODE level (the sign is constant within an episode between
   flips, so a per-frame split is not a weak test, it is no test at all),
   against TWO nulls: the majority-class rate, and the same two probes run on
   labels shuffled between whole episodes. The second is the one that matters.
   "Chance" for a probe on 20,000 autocorrelated frames is not 0.500 with a
   binomial error bar around it -- the effective sample size is the number of
   sign RUNS, a few hundred -- so a raw accuracy of 0.55 may be nothing at all.
   The shuffled-label null measures what nothing looks like here rather than
   assuming it (``wm.probes.shuffled_label_null``);
2. **the position distribution by sign** -- how big the shift actually is,
   in ``ball_y``, ``ball_x``, ``|v|`` and their histograms, so the size of the
   available leak is a measured quantity;
3. **the position-matched probe** -- the same two probes on a subsample in
   which the two signs are equally frequent inside every 8x8 position bin. A
   probe reading only position can do no better than chance there. If accuracy
   collapses, the leak is explained; if it survives, something else carries the
   bit and that is a genuine and unexpected finding.

Plus, for completeness against the v1 baseline: velocities should stay
undecodable (nothing has changed that) and positions should decode as well as
they ever did.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .analyze import load_ckpt
from .data import FrameDataset, make_loader
from .diagnostics import active_units, encode_dataset
from .probes import (
    balance_within_position_bins,
    classification_suite,
    probe_suite,
)

WHICH_R2 = ("linear", "poly2", "knn")
WHICH_CLF = ("logistic", "knn")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def position_shift(state: np.ndarray, names: Sequence[str]) -> Dict:
    """How differently does the ball sit under each sign? The size of the leak.

    Reported as per-column means plus a total-variation distance between the
    two ``ball_y`` histograms. TV distance rather than a t-test because the
    question is not "is there a difference" (with 100k autocorrelated frames
    there always is) but "how much of one" -- and TV distance has a direct
    reading: it is an upper bound on how often ANY classifier looking only at
    ``ball_y`` can beat chance.
    """
    j = list(names).index("gravity_sign")
    sign = np.asarray(state)[:, j]
    down, up = sign < 0, sign > 0
    out: Dict[str, object] = {
        "n_down": int(down.sum()),
        "n_up": int(up.sum()),
        "frac_down": float(down.mean()),
    }
    speed = np.hypot(np.asarray(state)[:, 2], np.asarray(state)[:, 3])
    cols = {
        "ball_x": np.asarray(state)[:, 0],
        "ball_y": np.asarray(state)[:, 1],
        "speed": speed,
    }
    out["means"] = {
        k: {"down": float(v[down].mean()), "up": float(v[up].mean()),
            "diff": float(v[down].mean() - v[up].mean()),
            "pooled_sd": float(v.std())}
        for k, v in cols.items()
    }
    edges = np.linspace(0.0, 1.0, 21)
    hd, _ = np.histogram(cols["ball_y"][down], bins=edges, density=False)
    hu, _ = np.histogram(cols["ball_y"][up], bins=edges, density=False)
    hd = hd / max(hd.sum(), 1)
    hu = hu / max(hu.sum(), 1)
    out["ball_y_tv_distance"] = float(0.5 * np.abs(hd - hu).sum())
    out["ball_y_hist_down"] = hd.tolist()
    out["ball_y_hist_up"] = hu.tolist()
    out["ball_y_hist_edges"] = edges.tolist()
    # The bound that makes the TV distance actionable: the best possible
    # accuracy of a classifier that sees only ball_y.
    out["best_possible_acc_from_ball_y"] = float(0.5 + 0.5 * out["ball_y_tv_distance"])
    return out


def sign_figure(report: Dict, path: Path) -> Path:
    """Two panels: the size of the available leak, and what the probes got."""
    plt = _plt()
    ps = report["position_shift"]
    edges = np.asarray(ps["ball_y_hist_edges"])
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].step(centres, ps["ball_y_hist_down"], where="mid",
               color="#c2543a", lw=1.6, label="gravity pulls DOWN")
    ax[0].step(centres, ps["ball_y_hist_up"], where="mid",
               color="#3a7bd5", lw=1.6, label="gravity pulls UP")
    ax[0].set_xlabel("ball_y")
    ax[0].set_ylabel("fraction of frames")
    ax[0].set_title(
        f"where the ball sits, by sign\n"
        f"total-variation distance {ps['ball_y_tv_distance']:.3f} → a ball_y-only "
        f"classifier could reach {ps['best_possible_acc_from_ball_y']:.3f}",
        fontsize=9)
    ax[0].legend(fontsize=8)

    groups = [("raw mu", report["sign_probe"]),
              ("position-matched", report["sign_probe_position_matched"])]
    labels, vals, colors = [], [], []
    for gname, rec in groups:
        for k in WHICH_CLF:
            if k in rec:
                labels.append(f"{k}\n({gname})")
                vals.append(rec[k])
                colors.append("#3a7bd5" if gname == "raw mu" else "#2a9d4a")
    ax[1].bar(range(len(vals)), vals, color=colors)
    ax[1].axhline(0.5, ls="--", c="k", lw=1)
    ax[1].text(len(vals) - 0.4, 0.508, "chance", ha="right", fontsize=8)
    ax[1].axhline(report["sign_probe"]["majority"], ls=":", c="0.4", lw=1)
    # The null the numbers are actually judged against: the worst shuffled-label
    # accuracy any of the probes reached. Anything under this line is nothing.
    nulls = [report["sign_probe"].get(f"null_{k}_max") for k in WHICH_CLF]
    nulls = [v for v in nulls if v is not None]
    if nulls:
        ax[1].axhspan(0.0, max(nulls), color="0.85", zorder=0)
        ax[1].text(-0.4, max(nulls) + 0.01,
                   f"shuffled-label null (worst {max(nulls):.3f})",
                   fontsize=8, color="0.35")
    ax[1].set_xticks(range(len(vals)))
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("held-out accuracy (episode-level split)")
    ax[1].set_title("can one frame's mu tell you which way gravity pulls?",
                    fontsize=9)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True,
                   help="a v4 split with MANY SHORT episodes (data/v4/probe)")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--bins", type=int, default=8,
                   help="position bins per axis for the matched probe. Finer "
                        "bins match position more tightly but throw away more "
                        "frames; 8x8 keeps ~2/3 of them at 0.08 ball radius")
    p.add_argument("--shuffles", type=int, default=5,
                   help="how many shuffled-label refits to average for the "
                        "null. 0 disables it and leaves only the base rate, "
                        "which is not enough -- see the module docstring.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out or (Path(a.ckpt).parent / "analysis"))
    out.mkdir(parents=True, exist_ok=True)

    model, mcfg, free_bits = load_ckpt(a.ckpt, a.device)
    ds = FrameDataset(a.data)
    names = list(ds.state_names)
    if "gravity_sign" not in names:
        raise SystemExit(f"{a.data} is not a v4 dataset (no gravity_sign column)")

    enc = encode_dataset(
        model, make_loader(a.data, 256, shuffle=False, return_state=True),
        device=a.device, limit=a.limit,
    )
    mu, logvar, state = enc["mu"], enc["logvar"], enc["state"]
    group_ids = np.arange(len(mu)) // ds.Tp1
    j_sign = names.index("gravity_sign")
    y = np.sign(state[:, j_sign])

    print(f"{a.data}: {len(mu)} frames, {int(group_ids.max()) + 1} episodes, "
          f"z_dim {mcfg.z_dim}")
    au = active_units(logvar, mu, free_bits=free_bits)
    print(f"active units: {au['n_active']}/{mcfg.z_dim}")

    # ------------------------------------------------------- the raw probe
    clf = classification_suite(mu, y, group_ids=group_ids, seed=a.seed,
                               which=WHICH_CLF, n_shuffles=a.shuffles)
    print("\ngravity_sign from ONE frame's mu (held-out, episode-level split):")
    print(f"  majority-class baseline  {clf['majority']:.3f}"
          f"   ({clf['n_train']} train / {clf['n_test']} test frames)")
    for k in WHICH_CLF:
        if k in clf:
            null = clf.get(f"null_{k}_mean")
            tail = ("" if null is None else
                    f"   (shuffled-label null {null:.3f}, "
                    f"worst of {clf['null_n_shuffles']} "
                    f"{clf[f'null_{k}_max']:.3f})")
            print(f"  {k:9s}                {clf[k]:.3f}{tail}")

    # ------------------------------------------------- how big is the leak
    shift = position_shift(state, names)
    print(f"\nposition distribution by sign "
          f"({shift['n_down']} down / {shift['n_up']} up frames)")
    for k, v in shift["means"].items():
        print(f"  {k:8s} down {v['down']:.4f}  up {v['up']:.4f}  "
              f"diff {v['diff']:+.4f}  ({v['diff'] / max(v['pooled_sd'], 1e-9):+.3f} sd)")
    print(f"  ball_y histogram total-variation distance "
          f"{shift['ball_y_tv_distance']:.3f}  -> a classifier seeing ONLY "
          f"ball_y could reach at most "
          f"{shift['best_possible_acc_from_ball_y']:.3f}")

    # ------------------------------------------------ the matched probe
    idx = balance_within_position_bins(state, y, bins=a.bins, seed=a.seed)
    if len(idx) < 200:
        matched = {"note": f"only {len(idx)} matched frames; probe skipped"}
        print(f"\nposition-matched probe skipped ({len(idx)} frames)")
    else:
        matched = classification_suite(
            mu[idx], y[idx], group_ids=group_ids[idx], seed=a.seed,
            which=WHICH_CLF, n_shuffles=a.shuffles,
        )
        print(f"\nposition-matched probe ({len(idx)} of {len(mu)} frames kept; "
              f"the two signs are equally frequent inside every "
              f"{a.bins}x{a.bins} position bin)")
        print(f"  majority-class baseline  {matched['majority']:.3f}")
        for k in WHICH_CLF:
            if k in matched:
                print(f"  {k:9s}                {matched[k]:.3f}")

    # ------------------------------- the v1 sanity table, on v4's encoder
    # Positions must still decode and velocities must still not. Nothing in v4
    # was supposed to change either, so a change here is a bug in the encoder
    # or the data, not a finding about gravity.
    base = probe_suite(mu, state, names, group_ids=group_ids, seed=a.seed,
                       which=WHICH_R2)
    print("\nheld-out R^2 for the ordinary factors (the v1 sanity check):")
    print("  " + f"{'factor':12s}" + "".join(f"{k:>9s}" for k in WHICH_R2))
    for nm in names:
        print(f"  {nm:12s}" + "".join(f"{base[k][nm]:9.3f}" for k in WHICH_R2))

    # The line every accuracy above is judged against. Taken as the WORST of
    # the shuffled refits rather than their mean: with a handful of shuffles the
    # mean understates the spread, and the conservative reading is the one that
    # protects the negative result this whole file is trying to establish.
    null_worst = max(
        [clf.get(f"null_{k}_max", 0.5) for k in WHICH_CLF]
        + [matched.get(f"null_{k}_max", 0.5) for k in WHICH_CLF]
        + [0.5])

    report = {
        "data": a.data,
        "ckpt": a.ckpt,
        "n_frames": int(len(mu)),
        "n_episodes": int(group_ids.max() + 1),
        "z_dim": mcfg.z_dim,
        "active_units": au["n_active"],
        "kl_per_dim": au["kl_per_dim"].tolist(),
        "sign_probe": clf,
        "sign_probe_position_matched": matched,
        "null_worst": null_worst,
        "n_matched_frames": int(len(idx)),
        "position_shift": shift,
        "probes_r2": base,
        "state_names": names,
    }
    try:
        sign_figure(report, out / "sign_probe.png")
        print(f"\nwrote {out/'sign_probe.png'}")
    except ImportError:
        print("  (matplotlib not installed; skipping sign_probe.png)")
    (out / "report_v4.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out/'report_v4.json'}")

    best = max([clf.get(k, 0.0) for k in WHICH_CLF] +
               [matched.get(k, 0.0) for k in WHICH_CLF])
    null = null_worst
    print(f"\nthe null: shuffled labels reach {null:.3f} at worst, so anything "
          f"at or below that is indistinguishable from no information.")
    if best > max(0.60, null + 0.05):
        print(f"\n  !! sign accuracy reaches {best:.3f}, well above chance. "
              "Compare it against best_possible_acc_from_ball_y above: if the "
              "raw probe is high and the position-matched one is not, the "
              "route is the position-distribution shift and stage two's memory "
              "claims need the matched control. If BOTH are high, something in "
              "the frame carries the sign and the v4 premise is broken -- find "
              "it before training M.")
    else:
        print(f"\n  sign accuracy tops out at {best:.3f} against a null of "
              f"{null:.3f}: the frames do not carry the bit, which is what v4 "
              "needs to be true.")


if __name__ == "__main__":
    main()

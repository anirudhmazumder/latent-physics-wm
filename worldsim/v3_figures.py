"""Eyeball figures for the v3 datasets.

    python -m worldsim.v3_figures --data data/v3/train --out runs/v3_env

Three things, and all three are worth actually looking at before training:

``sample_grid.png``
    32 frames chosen to span the visibility range, annotated (by ordering)
    from fully visible through partial to fully hidden. If you cannot tell a
    partially occluded ball from a fully occluded one by eye, the VAE's job on
    partial frames is harder than you think.

``occlusion_episode.gif``
    One episode in which the ball goes behind the band AND bounces off a side
    wall while hidden -- found by intersecting ``EVENT_HIDDEN`` runs with
    ``EVENT_WALL_X`` in ``events.npy``. This is the picture of the v3 claim:
    the exit x depends on a collision that never appeared in a single frame.

``hidden_run_lengths.png``
    Histogram of consecutive-hidden-frame run lengths for all three band
    heights on one axis. This is the memory-horizon x-axis: it says how many
    steps ``h`` actually has to bridge, and how much the tall/taller splits
    stretch that.

Output lives under runs/ rather than data/ because data/ is gitignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .bouncing_box import EVENT_HIDDEN, EVENT_WALL_X
from .collect import hidden_runs
from .render import frame_grid, save_gif, upscale


def _load(root: str | Path):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    if "ball_visible" not in meta["state_names"]:
        raise SystemExit(f"{root} is not a v3 dataset (no ball_visible column)")
    return (
        meta,
        np.load(root / "frames.npy", mmap_mode="r"),
        np.load(root / "states.npy", mmap_mode="r"),
        np.load(root / "events.npy", mmap_mode="r"),
    )


# --------------------------------------------------------------- sample grid


def sample_grid(frames, states, meta, n: int = 32, seed: int = 0) -> np.ndarray:
    """``n`` frames spanning the visibility range, ordered visible -> hidden.

    Sampled by *stratifying on ``ball_visible``* rather than by taking one
    frame per episode: a uniform sample of v3 frames is ~44% fully visible and
    ~18% fully hidden, so a plain sample would show you a grid of ordinary
    frames and almost no partial occlusions -- which are the interesting case
    and the one the VAE is most likely to get wrong.
    """
    j = meta["state_names"].index("ball_visible")
    vis = np.asarray(states[..., j])
    rng = np.random.default_rng(seed)

    # Half the tiles from the partial band, a quarter each fully visible and
    # fully hidden, and the partial half spread evenly across coverage.
    edges = np.concatenate([[-0.01, 1e-3], np.linspace(1e-3, 1 - 1e-3, n // 2 + 1)[1:], [1.01]])
    picks = []
    for k in range(len(edges) - 1):
        cand = np.argwhere((vis > edges[k]) & (vis <= edges[k + 1]))
        if not len(cand):
            continue
        take = (n // 4) if k in (0, len(edges) - 2) else 1
        sel = cand[rng.choice(len(cand), size=min(take, len(cand)), replace=False)]
        picks.extend([(int(e), int(t)) for e, t in sel])
    picks = picks[:n]
    picks.sort(key=lambda et: -vis[et])
    tiles = np.stack([np.asarray(frames[e, t]) for e, t in picks])
    return frame_grid(upscale(tiles, 2), ncols=8), np.array([vis[p] for p in picks])


# ------------------------------------------------------------ the episode gif


def find_bounce_behind_band(
    events, max_len: int = 0
) -> Optional[Tuple[int, int, int]]:
    """An (episode, start, length) hidden run containing a wall-x bounce.

    Returns the LONGEST such run, because the longer the ball is out of sight
    the more striking the figure -- and the harder the corresponding prediction
    problem is for M later.

    ``max_len`` caps that: 0 (the v3 default) means no cap, but in v3.1 the
    tallest band admits runs of 100+ frames in which the ball is skimming
    almost horizontally inside the band, and those make a boring, atypical
    picture. Capping at roughly the mean run length picks a crossing that looks
    like the ones the model is actually graded on.
    """
    ev = np.asarray(events)
    runs = hidden_runs((ev & EVENT_HIDDEN).astype(bool))
    wall = (ev & EVENT_WALL_X).astype(bool)
    hits = [r for r in runs if wall[r[0], r[1] : r[1] + r[2]].any()]
    if max_len > 0:
        hits = [r for r in hits if r[2] <= max_len] or hits
    return max(hits, key=lambda r: r[2]) if hits else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", default=["data/v3/train"],
                   help="split(s) for the grid. Several roots give one band of "
                        "rows each, stacked -- which is how v3.1 shows that the "
                        "three occluder heights all look like the same world "
                        "with the band moved.")
    p.add_argument("--gif-data", default="data/v3/val",
                   help="split to search for a wall bounce behind the band")
    p.add_argument("--hist-data", nargs="*",
                   default=["data/v3/train_mix", "data/v3/tall", "data/v3/taller"],
                   help="one split per band height for hidden_run_lengths.png")
    p.add_argument("--out", default="runs/v3_env")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--gif-max-len", type=int, default=0,
                   help="ignore hidden runs longer than this when choosing the "
                        "gif (0 = no cap). Use ~the mean run length to avoid "
                        "picking a freak near-horizontal skim inside the band.")
    p.add_argument("--gif-pad", type=int, default=25,
                   help="frames of context to show either side of the hidden run")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    # ------------------------------------------------------------ grid
    # One grid per root, stacked vertically with a bright divider. With a
    # single root this is byte-identical to what v3 produced.
    panels, lines = [], []
    for root in a.data:
        meta, frames, states, events = _load(root)
        grid, gv = sample_grid(frames, states, meta, n=a.n, seed=a.seed)
        panels.append(grid)
        band = meta["occluder_y"]
        lines.append(
            f"  {Path(root).name:16s} band {band[0]:.2f}-{band[1]:.2f}  "
            f"ball_visible {gv.min():.2f}..{gv.max():.2f}  "
            f"({int((gv > 0.999).sum())} visible / "
            f"{int(((gv > 1e-3) & (gv <= 0.999)).sum())} partial / "
            f"{int((gv <= 1e-3).sum())} hidden)")
    if len(panels) == 1:
        grid = panels[0]
    else:
        w = max(g.shape[1] for g in panels)
        rows = []
        for k, g in enumerate(panels):
            if g.shape[1] < w:
                g = np.pad(g, ((0, 0), (0, w - g.shape[1]), (0, 0)),
                           constant_values=40)
            if k:
                rows.append(np.full((6, w, 3), 200, dtype=g.dtype))
            rows.append(g)
        grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(out / "sample_grid.png")
    print(f"wrote {out/'sample_grid.png'}")
    for ln in lines:
        print(ln)

    # ------------------------------------------------------------- gif
    gmeta, gframes, gstates, gevents = _load(a.gif_data)
    hit = find_bounce_behind_band(gevents, a.gif_max_len)
    if hit is None:
        print(f"  no hidden run in {a.gif_data} contains a wall-x bounce; "
              "falling back to the longest hidden run")
        runs = hidden_runs((np.asarray(gevents) & EVENT_HIDDEN).astype(bool))
        hit = max(runs, key=lambda r: r[2])
    e, t0, n = hit
    # events[e, t] is the transition INTO states[e, t + 1], so the hidden run
    # covers frames t0 + 1 .. t0 + n; pad either side for context.
    lo = max(0, t0 + 1 - a.gif_pad)
    hi = min(gframes.shape[1], t0 + n + 1 + a.gif_pad)
    save_gif(np.asarray(gframes[e, lo:hi]), out / "occlusion_episode.gif", fps=12, scale=4)
    j = gmeta["state_names"].index("ball_visible")
    xs = np.asarray(gstates[e, t0 + 1 : t0 + n + 1, 0])
    print(f"wrote {out/'occlusion_episode.gif'}  episode {e}, frames {lo}..{hi}  "
          f"hidden for {n} frames (t={t0+1}..{t0+n}), "
          f"ball_x {xs[0]:.2f} -> {xs[-1]:.2f} while hidden, "
          f"entered at y={float(gstates[e, t0, 1]):.2f}")

    # ---------------------------------------------------------- histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping hidden_run_lengths.png)")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.arange(0, 72, 2)
    summary = []
    for path in a.hist_data:
        m, _, _, ev = _load(path)
        runs = hidden_runs((np.asarray(ev) & EVENT_HIDDEN).astype(bool))
        L = np.array([r[2] for r in runs])
        band = m["occluder_y"]
        label = (f"{Path(path).name}  band {band[0]:.2f}-{band[1]:.2f} "
                 f"(h={band[1]-band[0]:.2f})  mean {L.mean():.1f}")
        # density=True, because the splits have very different episode counts
        # and the question is the SHAPE of the distribution, not how many runs
        # each split happens to contain.
        n_, _, patches = ax.hist(
            L, bins=bins, density=True, histtype="step", lw=2, label=label
        )
        ax.axvline(L.mean(), ls=":", lw=1.2, alpha=0.6,
                   color=patches[0].get_edgecolor())
        summary.append((path, band, len(L), L.mean(), np.median(L), L.max()))
    ax.set_xlabel("consecutive fully-hidden frames in one run")
    ax.set_ylabel("density")
    ax.set_title("How long the ball is out of sight, by band height\n"
                 "(this is the number of steps h has to bridge)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "hidden_run_lengths.png", dpi=140)
    print(f"wrote {out/'hidden_run_lengths.png'}")
    for path, band, n_runs, mean, med, mx in summary:
        print(f"  {path:22s} band {band[0]:.2f}-{band[1]:.2f}  "
              f"{n_runs:4d} runs  mean {mean:5.1f}  median {med:5.1f}  max {mx}")


if __name__ == "__main__":
    main()

"""Eyeball figures for the v4 datasets.

    python -m worldsim.v4_figures --data data/v4/train --gif-data data/v4/train_mix \
        --out runs/v4_env

Three figures, and they are deliberately a matched pair plus a counterexample:
the first two say "there is nothing to see", the third says "and yet there is".

``sample_grid.png``
    32 frames in two stacked panels, the top one sampled from frames where
    gravity points DOWN and the bottom one from frames where it points UP,
    matched on ball height so the two panels have the same marginal. They
    should be indistinguishable. That is the whole v4 premise as a picture, and
    it is the thing the sign probe on ``mu`` will shortly confirm numerically:
    if you *can* tell the panels apart, the latent leaks through appearance and
    every downstream memory claim is contaminated.

``gravity_episode.gif``
    One episode containing a paddle contact, padded either side, chosen so that
    the ball's arc visibly changes curvature at the flip -- decelerating into
    the ceiling before and accelerating away from it after, or vice versa. The
    per-frame difference is 1e-4 and invisible; the difference across 50 frames
    is not.

``trajectories_by_sign.png``
    ``ball_y`` against time for a handful of episodes, coloured by the sign in
    force and with flips marked. This is the same information as the GIF but
    legible at a glance: under one sign the traverses bulge toward the ceiling,
    under the other toward the floor, and at a flip the curvature reverses.

Output lives under runs/ rather than data/ because data/ is gitignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .bouncing_box import EVENT_FLIP
from .render import frame_grid, save_gif, upscale


def _load(root: str | Path):
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    if "gravity_sign" not in meta["state_names"]:
        raise SystemExit(f"{root} is not a v4 dataset (no gravity_sign column)")
    return (
        meta,
        np.load(root / "frames.npy", mmap_mode="r"),
        np.load(root / "states.npy", mmap_mode="r"),
        np.load(root / "events.npy", mmap_mode="r"),
    )


# --------------------------------------------------------------- sample grid


def sample_grid_by_sign(frames, states, meta, n: int = 32, seed: int = 0):
    """``n`` frames, half per sign, MATCHED on ball height.

    The matching is the point and it is not decoration. Under gravity-down the
    ball genuinely spends more of its time low in the box (it moves slowest at
    the top of its arc, so a uniform sample over frames is biased toward the
    apex, which sits at a different height under each sign). A naive sample
    would therefore show a visibly lower ball in one panel -- and you would be
    looking at the *sampling*, not at the world. Drawing a height for each tile
    and then finding the nearest frame of each sign at that height removes the
    confound, so anything left that distinguishes the panels is real.

    Returns ``(grid, heights)``.
    """
    j = meta["state_names"].index("gravity_sign")
    sign = np.asarray(states[..., j])
    y = np.asarray(states[..., 1])
    rng = np.random.default_rng(seed)

    m = n // 2
    r = meta["config"]["ball_radius"]
    targets = np.linspace(r + 0.02, 1.0 - r - 0.02, m)

    panels, picked = [], []
    for want in (-1.0, 1.0):
        idx = np.argwhere(np.sign(sign) == want)
        if not len(idx):
            raise SystemExit("split contains only one gravity sign")
        ys = y[idx[:, 0], idx[:, 1]]
        tiles, hs = [], []
        used = set()
        for t in targets:
            # Nearest unused frame at this height, jittered so repeated runs do
            # not always pick the same episode.
            order = np.argsort(np.abs(ys - t) + rng.uniform(0, 1e-3, len(ys)))
            for k in order[:64]:
                key = (int(idx[k, 0]), int(idx[k, 1]))
                if key not in used:
                    used.add(key)
                    tiles.append(np.asarray(frames[key[0], key[1]]))
                    hs.append(float(ys[k]))
                    break
        panels.append(frame_grid(upscale(np.stack(tiles), 2), ncols=8))
        picked.append(np.asarray(hs))

    w = max(p.shape[1] for p in panels)
    rows = []
    for k, g in enumerate(panels):
        if g.shape[1] < w:
            g = np.pad(g, ((0, 0), (0, w - g.shape[1]), (0, 0)), constant_values=40)
        if k:
            rows.append(np.full((6, w, 3), 200, dtype=g.dtype))
        rows.append(g)
    return np.concatenate(rows, axis=0), picked


# ------------------------------------------------------------ the episode gif


def find_flip_episode(
    events, states, meta, pad: int = 45
) -> Optional[Tuple[int, int]]:
    """An ``(episode, flip_t)`` whose flip has room to show on both sides.

    "Room" means ``pad`` frames of free flight before and after with no second
    flip in them, because two flips 20 frames apart is a picture of nothing:
    the curvature needs a whole traverse on each side to become visible at all.
    Among the candidates we take the one whose ball covers the most vertical
    distance around the flip, which is the one where the arc is most legible.
    """
    ev = np.asarray(events)
    y = np.asarray(states[..., 1])
    flips = np.argwhere((ev & EVENT_FLIP).astype(bool))
    best = None
    for e, t in flips:
        e, t = int(e), int(t)
        lo, hi = t - pad, t + pad
        if lo < 0 or hi >= y.shape[1]:
            continue
        others = np.flatnonzero((ev[e] & EVENT_FLIP).astype(bool))
        if np.any((others >= lo) & (others <= hi) & (others != t)):
            continue
        span = float(y[e, lo:hi].max() - y[e, lo:hi].min())
        if best is None or span > best[0]:
            best = (span, e, t)
    return None if best is None else (best[1], best[2])


# ------------------------------------------------------- trajectories by sign


def trajectories_figure(states, events, meta, path, n_episodes: int = 6, seed: int = 0):
    """``ball_y`` vs t, one row per episode, coloured by the sign in force."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j = meta["state_names"].index("gravity_sign")
    sign = np.asarray(states[..., j])
    y = np.asarray(states[..., 1])
    ev = np.asarray(events)
    flip = (ev & EVENT_FLIP).astype(bool)

    # Prefer episodes that actually contain a flip -- an episode with none is a
    # correct picture of a constant-gravity world and shows nothing about the
    # switch. Fill up with flipless ones only if there are not enough.
    n_flips = flip.sum(1)
    order = list(np.flatnonzero(n_flips >= 1)) + list(np.flatnonzero(n_flips == 0))
    eps = order[:n_episodes]

    fig, axes = plt.subplots(len(eps), 1, figsize=(9, 1.55 * len(eps)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    C = {-1.0: "#c2543a", 1.0: "#3a7bd5"}
    for ax, e in zip(axes, eps):
        t = np.arange(y.shape[1])
        s = np.sign(sign[e])
        # One line per constant-sign stretch, so a colour change is exactly a
        # flip and the eye is not asked to interpolate across one.
        edges = [0] + (np.flatnonzero(np.diff(s)) + 1).tolist() + [len(t)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            sl = slice(lo, min(hi + 1, len(t)))
            ax.plot(t[sl], y[e, sl], lw=1.4, color=C[float(s[lo])])
        for ft in np.flatnonzero(flip[e]):
            ax.axvline(ft + 1, color="k", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"ep {e}", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_title(
        "ball height over time — red = gravity pulls DOWN, blue = pulls UP, "
        "dashed = a paddle contact flips it", fontsize=9)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def traverse_stats(states, meta) -> dict:
    """How long a floor-to-ceiling crossing takes, split by the sign in force.

    The number that turned out to matter most in v4 stage one, and it is not in
    the design document. The design argued that the sign is invisible because
    the *acceleration* is small against the renderer's 1/64 quantisation -- true
    as far as it goes. But the sign also sets the ball's ENERGY BUDGET, and at a
    40° launch that budget is marginal: pulling down, the ball only just reaches
    the ceiling and so crawls the last third of the way; pulling up, it
    accelerates the whole climb. The two regimes therefore differ in traverse
    duration by a large factor, which is a much coarser and much more available
    cue than a curvature fit -- available, in principle, from one traverse
    rather than from a memory of the flip.

    A traverse is the stretch between consecutive sign changes of ``ball_vy``
    (a wall bounce or a paddle contact), counted only where the gravity sign
    did not change inside it -- a traverse straddling a flip belongs to neither
    regime.
    """
    j = meta["state_names"].index("gravity_sign")
    S = np.asarray(states)
    sign, vy, y = S[..., j], S[..., 3], S[..., 1]
    out = {}
    for want, label in ((-1.0, "down"), (1.0, "up")):
        durs, spans, vys = [], [], []
        for e in range(S.shape[0]):
            turns = np.flatnonzero(np.diff(np.sign(vy[e])) != 0) + 1
            for t0, t1 in zip(turns[:-1], turns[1:]):
                seg_sign = np.sign(sign[e, t0:t1])
                if len(seg_sign) < 3 or not np.all(seg_sign == want):
                    continue
                durs.append(t1 - t0)
                spans.append(float(y[e, t0:t1].max() - y[e, t0:t1].min()))
                vys.append(float(np.abs(vy[e, t0:t1]).mean()))
        if not durs:
            continue
        d = np.asarray(durs, float)
        # Full traverses only: a 5-frame wobble at the floor is not a crossing
        # and would drag the median down by an amount that depends on the
        # policy rather than on the physics.
        m = np.asarray(spans) > 0.5
        out[label] = {
            "n_traverses": int(len(d)),
            "n_full": int(m.sum()),
            "median_frames": float(np.median(d[m])) if m.any() else float("nan"),
            "mean_frames": float(d[m].mean()) if m.any() else float("nan"),
            "mean_abs_vy": float(np.asarray(vys)[m].mean()) if m.any() else float("nan"),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/v4/train",
                   help="split for the sample grid")
    p.add_argument("--gif-data", default="data/v4/train_mix",
                   help="split to search for a well-isolated flip")
    p.add_argument("--traj-data", default=None,
                   help="split for trajectories_by_sign.png (default --gif-data)")
    p.add_argument("--out", default="runs/v4_env")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--gif-pad", type=int, default=45,
                   help="frames of context either side of the flip. Roughly one "
                        "traverse: less and the curvature has not had time to "
                        "show, more and the ball has flipped again")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    # ------------------------------------------------------------ grid
    meta, frames, states, events = _load(a.data)
    grid, heights = sample_grid_by_sign(frames, states, meta, n=a.n, seed=a.seed)
    Image.fromarray(grid).save(out / "sample_grid.png")
    print(f"wrote {out/'sample_grid.png'}")
    for nm, hs in zip(("gravity down (top panel)", "gravity up (bottom panel)"),
                      heights):
        print(f"  {nm:26s} {len(hs)} frames, ball_y {hs.min():.2f}..{hs.max():.2f}"
              f"  mean {hs.mean():.3f}")

    # ------------------------------------------------------------- gif
    gmeta, gframes, gstates, gevents = _load(a.gif_data)
    hit = find_flip_episode(gevents, gstates, gmeta, pad=a.gif_pad)
    if hit is None:
        raise SystemExit(
            f"no flip in {a.gif_data} has {a.gif_pad} clear frames either side")
    e, t = hit
    lo, hi = t - a.gif_pad, t + a.gif_pad + 1
    save_gif(np.asarray(gframes[e, lo:hi]), out / "gravity_episode.gif",
             fps=14, scale=4)
    js = gmeta["state_names"].index("gravity_sign")
    before = np.asarray(gstates[e, lo:t + 1, 1])
    after = np.asarray(gstates[e, t + 1:hi, 1])
    print(f"wrote {out/'gravity_episode.gif'}  episode {e}, frames {lo}..{hi}, "
          f"flip at t={t + 1}  "
          f"sign {float(gstates[e, t, js]):+.0f} -> {float(gstates[e, t + 1, js]):+.0f}")
    # A quadratic fit either side: the headline of the whole figure is that
    # these two numbers have opposite signs and magnitude ~= the gravity.
    for nm, seg in (("before", before), ("after", after)):
        tt = np.arange(len(seg))
        acc = 2.0 * np.polyfit(tt, seg, 2)[0]
        print(f"    fitted vertical acceleration {nm:6s}: {acc:+.2e} "
              f"(gravity = {gmeta['gravity']:.0e})")

    # ---------------------------------------------------- trajectories
    troot = a.traj_data or a.gif_data
    tmeta, _tf, tstates, tevents = _load(troot)
    try:
        trajectories_figure(tstates, tevents, tmeta,
                            out / "trajectories_by_sign.png",
                            n_episodes=a.episodes, seed=a.seed)
        print(f"wrote {out/'trajectories_by_sign.png'}  (from {troot})")
    except ImportError:
        print("  (matplotlib not installed; skipping trajectories figure)")

    ts = traverse_stats(tstates, tmeta)
    print(f"\ntraverse duration by sign (full crossings only, {troot}):")
    for label, rec in ts.items():
        print(f"  gravity {label:5s} n={rec['n_full']:4d}  "
              f"median {rec['median_frames']:6.1f} frames  "
              f"mean {rec['mean_frames']:6.1f}  "
              f"mean |vy| {rec['mean_abs_vy']:.4f}")
    if len(ts) == 2:
        a, b = ts["down"]["median_frames"], ts["up"]["median_frames"]
        print(f"  ratio {max(a, b) / max(min(a, b), 1e-9):.2f}x  -- read this "
              "before claiming the sign needs to be REMEMBERED: a cue this "
              "coarse is re-derivable from one traverse.")
    (out / "traverse_stats.json").write_text(json.dumps(ts, indent=2))


if __name__ == "__main__":
    main()

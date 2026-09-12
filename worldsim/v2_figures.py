"""Eyeball figures for the v2 datasets.

    python -m worldsim.v2_figures --data data/v2/train --out runs/v2_env

Produces two things, and both are worth actually looking at before training
anything on this data:

``sample_grid.png``
    32 frames, one per episode, sorted by mass. If the colour ramp is not
    obviously a ramp to your eye, it will not be obvious to a VAE either -- the
    encoder has to spend KL nats on colour, and it only will if colour buys back
    reconstruction error.

``light_vs_heavy.gif``
    The lightest and the heaviest episode in the split, played side by side at
    the same frame rate. This is the picture of the *causal claim*: same box,
    same rules, and the yellow ball covers about four times the distance per
    frame that the purple one does. Output lives under runs/ rather than data/
    because data/ is gitignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .render import frame_grid, save_gif, side_by_side, upscale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/v2/train")
    p.add_argument("--gif-data", default="data/v2/val_mix",
                   help="split to pull the light/heavy pair from")
    p.add_argument("--out", default="runs/v2_env")
    p.add_argument("--n", type=int, default=32, help="tiles in the grid")
    p.add_argument("--gif-steps", type=int, default=120)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(a.data)
    meta = json.loads((root / "meta.json").read_text())
    names = meta["state_names"]
    if "mass" not in names:
        raise SystemExit(f"{root} is not a v2 dataset (no mass column)")
    j_mass = names.index("mass")

    # mmap: these arrays are hundreds of MB and we touch ~32 frames of them.
    frames = np.load(root / "frames.npy", mmap_mode="r")
    states = np.load(root / "states.npy", mmap_mode="r")
    mass = np.asarray(states[:, 0, j_mass])

    # One frame per episode, episodes ordered light -> heavy, and a different
    # timestep for each so the grid also shows a spread of ball positions
    # rather than 32 balls frozen at t=0.
    order = np.argsort(mass)
    pick = order[np.linspace(0, len(order) - 1, min(a.n, len(order))).astype(int)]
    ts = (np.arange(len(pick)) * 7) % frames.shape[1]
    tiles = np.stack([np.asarray(frames[e, t]) for e, t in zip(pick, ts)])
    grid = frame_grid(upscale(tiles, 2), ncols=8)
    from PIL import Image

    Image.fromarray(grid).save(out / "sample_grid.png")
    print(f"wrote {out/'sample_grid.png'}   masses "
          f"{mass[pick].min():.2f}..{mass[pick].max():.2f}")

    # ------------------------------------------------------------------ gif
    groot = Path(a.gif_data)
    gf = np.load(groot / "frames.npy", mmap_mode="r")
    gs = np.load(groot / "states.npy", mmap_mode="r")
    gm = np.asarray(gs[:, 0, j_mass])
    e_light, e_heavy = int(np.argmin(gm)), int(np.argmax(gm))
    T = min(a.gif_steps, gf.shape[1])
    pair = side_by_side([np.asarray(gf[e_light, :T]), np.asarray(gf[e_heavy, :T])])
    save_gif(pair, out / "light_vs_heavy.gif", fps=20, scale=3)
    print(f"wrote {out/'light_vs_heavy.gif'}   left m={gm[e_light]:.2f} "
          f"(speed {0.022/gm[e_light]:.4f}/frame), right m={gm[e_heavy]:.2f} "
          f"(speed {0.022/gm[e_heavy]:.4f}/frame)")


if __name__ == "__main__":
    main()

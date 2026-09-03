"""Visualisation helpers.

``side_by_side`` exists for later: when you have a dynamics model, the way you
will actually judge it is by putting the true rollout next to the imagined one
and watching where they diverge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _to_uint8(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255.0 if frames.max() <= 1.0 else frames, 0, 255)
        frames = frames.astype(np.uint8)
    return frames


def upscale(frames: np.ndarray, factor: int = 4) -> np.ndarray:
    """Nearest-neighbour upscale, so 64x64 is actually visible."""
    frames = _to_uint8(frames)
    return np.repeat(np.repeat(frames, factor, axis=-3), factor, axis=-2)


def save_gif(
    frames: np.ndarray,
    path: str | Path,
    fps: int = 20,
    scale: int = 4,
    loop: int = 0,
) -> Path:
    """Write a (T, H, W, 3) stack to an animated GIF. Requires Pillow."""
    from PIL import Image

    frames = upscale(frames, scale) if scale > 1 else _to_uint8(frames)
    imgs = [Image.fromarray(f) for f in frames]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / fps),
        loop=loop,
        optimize=False,
    )
    return path


def frame_grid(
    frames: np.ndarray,
    ncols: int = 8,
    pad: int = 2,
    pad_value: int = 40,
) -> np.ndarray:
    """Tile a (T, H, W, 3) stack into one image. Good for eyeballing a dataset."""
    frames = _to_uint8(frames)
    t, h, w, c = frames.shape
    nrows = int(np.ceil(t / ncols))
    out = np.full(
        (nrows * (h + pad) - pad, ncols * (w + pad) - pad, c), pad_value, np.uint8
    )
    for i in range(t):
        r, col = divmod(i, ncols)
        y0, x0 = r * (h + pad), col * (w + pad)
        out[y0 : y0 + h, x0 : x0 + w] = frames[i]
    return out


def side_by_side(
    stacks: Sequence[np.ndarray],
    pad: int = 4,
    pad_value: int = 40,
    labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Concatenate several (T, H, W, 3) stacks horizontally into one stack.

    Use this for true-vs-dreamed rollouts. ``labels`` is accepted and ignored
    here; draw text at save time if you want it.
    """
    stacks = [_to_uint8(s) for s in stacks]
    t = min(s.shape[0] for s in stacks)
    h = max(s.shape[1] for s in stacks)
    pieces = []
    for i, s in enumerate(stacks):
        s = s[:t]
        if s.shape[1] != h:
            raise ValueError("stacks must share height")
        pieces.append(s)
        if i != len(stacks) - 1:
            pieces.append(np.full((t, h, pad, 3), pad_value, np.uint8))
    return np.concatenate(pieces, axis=2)

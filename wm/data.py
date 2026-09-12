"""Dataset that streams frames from memory-mapped .npy without loading them.

For stage one the VAE treats frames as i.i.d. images -- temporal structure is
irrelevant here, and shuffling across all (episode, timestep) pairs gives better
gradient estimates than sampling contiguous chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def as_roots(root: str | Path | Sequence[str | Path]) -> list:
    """Normalise "one root or several" into a list of Paths.

    A bare string is a path, not a sequence of characters -- which is the one
    thing that makes ``isinstance(x, Sequence)`` the wrong test here.
    """
    if isinstance(root, (str, Path)):
        return [Path(root)]
    return [Path(r) for r in root]


class FrameDataset(Dataset):
    """Flat view over (E, T+1, H, W, 3) uint8 frames as individual images.

    Accepts ONE root or SEVERAL. Several is what v3.1 needs: the encoder has to
    see all three occlusion-band heights, or every taller-band result later on
    is confounded by an out-of-distribution encoder (the v3 mistake, README_V3
    Section 5). Roots are concatenated in the order given, so index i < len(ds0)
    addresses exactly the item single-root ``FrameDataset(roots[0])`` would --
    the single-root path is the multi-root path with one root, not a special
    case beside it.

    The per-root arrays must agree on resolution and on ``state_names``; they
    are free to differ in episode count, episode length, and (the point) band
    geometry. ``meta`` is the FIRST root's meta, with ``roots`` and ``metas``
    alongside for code that needs to know the rest.
    """

    def __init__(
        self,
        root: str | Path | Sequence[str | Path],
        return_state: bool = False,
    ):
        self.roots = as_roots(root)
        if not self.roots:
            raise ValueError("FrameDataset needs at least one root")
        self.root = self.roots[0]
        self.metas = [
            json.loads((r / "meta.json").read_text()) for r in self.roots
        ]
        self.meta = self.metas[0]

        # mmap_mode="r" means the array is never fully resident. With 8 GB of
        # RAM this is what lets you train on a dataset bigger than memory: the
        # OS pages in the 4 KB blocks you actually touch.
        self._frames = [np.load(r / "frames.npy", mmap_mode="r") for r in self.roots]
        self._states = [np.load(r / "states.npy", mmap_mode="r") for r in self.roots]
        self.return_state = return_state

        # ``frames`` / ``states`` stay bound to the first root, so every
        # existing consumer that reaches in for ``ds.frames.shape[2]`` (the
        # resolution) keeps working unchanged.
        self.frames = self._frames[0]
        self.states = self._states[0]

        for r, f in zip(self.roots[1:], self._frames[1:]):
            if f.shape[2:] != self.frames.shape[2:]:
                raise ValueError(
                    f"{r} has frame shape {f.shape[2:]}, expected "
                    f"{self.frames.shape[2:]} (from {self.root})"
                )
        names = {tuple(m["state_names"]) for m in self.metas}
        if len(names) != 1:
            raise ValueError(f"roots disagree on state_names: {names}")

        self.Tp1s = [f.shape[1] for f in self._frames]
        self.Es = [f.shape[0] for f in self._frames]
        # Per-root item counts and their cumulative sum: the index arithmetic
        # for "which root, and where inside it" is a single searchsorted.
        self._counts = np.array(
            [e * t for e, t in zip(self.Es, self.Tp1s)], dtype=np.int64
        )
        self._offsets = np.concatenate([[0], np.cumsum(self._counts)])

        # Single-root names for the attributes v1-v3 code reads directly.
        self.E, self.Tp1 = self.Es[0], self.Tp1s[0]
        self.state_names = self.meta["state_names"]

    def __len__(self) -> int:
        return int(self._counts.sum())

    def _unravel(self, i: int) -> Tuple[int, int]:
        """(episode, timestep) within the FIRST root. Kept for single-root use."""
        return divmod(i, self.Tp1)

    def _locate(self, i: int) -> Tuple[int, int, int]:
        """(root, episode, timestep) for a flat index across all roots."""
        if i < 0:
            i += len(self)
        k = int(np.searchsorted(self._offsets, i, side="right") - 1)
        e, t = divmod(i - int(self._offsets[k]), self.Tp1s[k])
        return k, e, t

    def __getitem__(self, i: int):
        k, e, t = self._locate(i)
        # np.array() forces a copy out of the memmap; without it the tensor
        # would alias a page-cache buffer that can move under you.
        frame = np.array(self._frames[k][e, t], dtype=np.float32) / 255.0
        # HWC -> CHW. Conv2d expects channels first.
        x = torch.from_numpy(frame).permute(2, 0, 1)
        if not self.return_state:
            return x
        s = torch.from_numpy(np.array(self._states[k][e, t], dtype=np.float32))
        return x, s


def make_loader(
    root: str | Path | Sequence[str | Path],
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    return_state: bool = False,
):
    ds = FrameDataset(root, return_state=return_state)
    # num_workers=0 is usually right on an M1 for this: the per-item work is a
    # 12 KB memcpy, so worker startup and IPC cost more than they save. Try 2 if
    # you see the GPU idling.
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )

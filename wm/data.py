"""Dataset that streams frames from memory-mapped .npy without loading them.

For stage one the VAE treats frames as i.i.d. images -- temporal structure is
irrelevant here, and shuffling across all (episode, timestep) pairs gives better
gradient estimates than sampling contiguous chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    """Flat view over (E, T+1, H, W, 3) uint8 frames as individual images."""

    def __init__(self, root: str | Path, return_state: bool = False):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())

        # mmap_mode="r" means the array is never fully resident. With 8 GB of
        # RAM this is what lets you train on a dataset bigger than memory: the
        # OS pages in the 4 KB blocks you actually touch.
        self.frames = np.load(self.root / "frames.npy", mmap_mode="r")
        self.states = np.load(self.root / "states.npy", mmap_mode="r")
        self.return_state = return_state

        self.E, self.Tp1 = self.frames.shape[0], self.frames.shape[1]
        self.state_names = self.meta["state_names"]

    def __len__(self) -> int:
        return self.E * self.Tp1

    def _unravel(self, i: int) -> Tuple[int, int]:
        return divmod(i, self.Tp1)

    def __getitem__(self, i: int):
        e, t = self._unravel(i)
        # np.array() forces a copy out of the memmap; without it the tensor
        # would alias a page-cache buffer that can move under you.
        frame = np.array(self.frames[e, t], dtype=np.float32) / 255.0
        # HWC -> CHW. Conv2d expects channels first.
        x = torch.from_numpy(frame).permute(2, 0, 1)
        if not self.return_state:
            return x
        s = torch.from_numpy(np.array(self.states[e, t], dtype=np.float32))
        return x, s


def make_loader(
    root: str | Path,
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

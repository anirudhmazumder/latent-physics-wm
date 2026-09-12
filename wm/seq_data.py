"""Windows of cached latents, for training the dynamics model (stage two, "M").

Stage one treated frames as i.i.d. images. Stage two is the opposite: the ONLY
thing that matters is temporal structure, and the pixels are gone entirely --
we train on the (E, T+1, z_dim) arrays that ``wm.cache_latents`` wrote. The
whole of ``data/v1/train`` is 2.5 MB of latents instead of 245 MB of pixels, so
everything here loads fully into RAM and there is no memmap juggling.

What one item is
----------------
A window of ``seq_len`` consecutive transitions from a single episode::

    z       (L, z_dim)   latent at times t0 .. t0+L-1      -- model input
    a       (L, 3)       one-hot action at the same times  -- model input
    z_next  (L, z_dim)   latent at times t0+1 .. t0+L      -- MDN target
    hit     (L,)         1.0 if that transition was a paddle contact
    reward  (L,)         dense shaping reward for that transition
    state   (L, S)       TRUE state at times t0+1 .. t0+L  -- diagnostics only
                         S = 6 in v1, 7 in v2 (mass is the extra column).

Three points that are easy to get wrong.

1. **Windows never cross episode boundaries.** A window is indexed as
   (episode, offset) and the offset is capped so that ``t0 + L <= T``. Letting a
   window straddle a reset would teach the model a teleport that does not exist
   in the world.

2. **z is resampled on every access.** We store ``mu`` and ``logvar`` and draw
   ``z = mu + eps * exp(0.5 * logvar)`` fresh each time the window is fetched.
   The long version of why is in ``wm.cache_latents``'s docstring; the short
   version is that at dream time the RNN is fed latents it produced itself,
   which are draws from a distribution and not clean posterior means. Training
   on means gives a model whose input distribution at inference has never been
   seen. ``use_mean=True`` switches this off so the effect can be measured.

3. **``state`` is never a model input.** It is used for exactly two things: the
   dense reward target (the environment computing a reward from privileged
   state, which is what environments do) and offline diagnostics. The RNN's
   input is always and only ``[z_t, onehot(a_t)]``.

The reward
----------
``r_t = 1 - |ball_x - paddle_x|``, evaluated on the state *after* the
transition, i.e. ``states[t+1]``. It is in [0, 1]: 1 when the paddle is directly
under the ball, falling off linearly. This is deliberately a dense shaping
reward rather than the sparse hit indicator, because stage three (the
controller) needs a gradient of information everywhere, not just on the ~1% of
frames where contact happens. We predict both, from two separate heads, so the
controller can use either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from worldsim.bouncing_box import EVENT_PADDLE

from .cache_latents import latent_suffix as _suffix_of

N_ACTIONS = 3


@dataclass
class _Episodes:
    """One dataset root's arrays, all resident in memory."""

    mu: np.ndarray        # (E, T+1, z)
    logvar: np.ndarray    # (E, T+1, z)
    actions: np.ndarray   # (E, T)   int64
    hit: np.ndarray       # (E, T)   float32
    reward: np.ndarray    # (E, T)   float32
    state: np.ndarray     # (E, T+1, S) float32; S = 6 (v1) or 7 (v2, +mass)
    root: str


def _load_root(root: str | Path, suffix: str = "") -> _Episodes:
    """Load one dataset root. ``suffix`` selects a non-default latent cache.

    The frames, actions, events and states are properties of the *world* and are
    shared; only ``mu``/``logvar`` depend on which encoder produced them. So the
    v2 ``--ablate-color`` control is a one-word change here rather than a
    duplicated dataset.
    """
    root = Path(root)
    sfx = _suffix_of(suffix)
    mu_path = root / f"mu{sfx}.npy"
    if not mu_path.exists():
        raise FileNotFoundError(
            f"{mu_path} missing -- run `python -m wm.cache_latents "
            f"--ckpt runs/vae_b1/vae.pt --data {root}"
            + (f" --suffix {sfx.lstrip('_')}" if sfx else "")
            + "` first"
        )
    mu = np.load(mu_path).astype(np.float32)
    logvar = np.load(root / f"logvar{sfx}.npy").astype(np.float32)
    actions = np.load(root / "actions.npy").astype(np.int64)
    events = np.load(root / "events.npy")
    states = np.load(root / "states.npy").astype(np.float32)

    hit = ((events & EVENT_PADDLE) != 0).astype(np.float32)

    # Reward from the POST-transition state: states[:, 1:] lines up with
    # actions[:, :] under the dataset's convention
    # (frames[t], actions[t]) -> frames[t+1]. Columns 0 and 4 are ball_x and
    # paddle_x in both v1 (6 columns) and v2 (7 -- mass is appended last), so
    # this is version-agnostic.
    post = states[:, 1:, :]                                   # (E, T, S)
    reward = (1.0 - np.abs(post[..., 0] - post[..., 4])).astype(np.float32)

    T = actions.shape[1]
    assert mu.shape[1] == T + 1, f"{root}: mu has {mu.shape[1]} steps, expected {T+1}"
    return _Episodes(mu, logvar, actions, hit, reward, states, str(root))


class LatentSequenceDataset(Dataset):
    """Fixed-length windows over one or more cached-latent dataset roots."""

    def __init__(
        self,
        roots: str | Path | Sequence[str | Path],
        seq_len: int = 32,
        stride: int = 1,
        use_mean: bool = False,
        seed: int | None = None,
        latent_suffix: str = "",
    ):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.roots = [str(r) for r in roots]
        self.seq_len = int(seq_len)
        self.use_mean = bool(use_mean)
        self.latent_suffix = latent_suffix
        self.eps = [_load_root(r, latent_suffix) for r in self.roots]

        z_dims = {e.mu.shape[-1] for e in self.eps}
        if len(z_dims) != 1:
            raise ValueError(f"roots disagree on z_dim: {z_dims}")
        self.z_dim = z_dims.pop()

        # Flat window index: (root_i, episode, t0). Built once; the cap
        # t0 + seq_len <= T is what keeps a window inside one episode.
        self.index: List[tuple] = []
        for ri, ep in enumerate(self.eps):
            E, T = ep.actions.shape
            if T < self.seq_len:
                raise ValueError(
                    f"{ep.root}: episodes have {T} transitions < seq_len {self.seq_len}"
                )
            for e in range(E):
                for t0 in range(0, T - self.seq_len + 1, stride):
                    self.index.append((ri, e, t0))
        self.index_arr = np.asarray(self.index, dtype=np.int64)

        # Our own generator so that sampling z is reproducible and independent
        # of whatever else is drawing from the global torch/numpy streams.
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ stats

    def n_transitions(self) -> int:
        return sum(int(e.actions.size) for e in self.eps)

    def hit_rate(self) -> float:
        """Fraction of transitions that are paddle contacts.

        Used to set ``pos_weight`` for the hit head's BCE: with a ~1% positive
        rate an unweighted BCE is minimised close to "always predict no", which
        looks like a fine loss and is a useless classifier.
        """
        num = sum(float(e.hit.sum()) for e in self.eps)
        return num / max(self.n_transitions(), 1)

    def pos_weight(self) -> float:
        r = self.hit_rate()
        return (1.0 - r) / max(r, 1e-8)

    # ------------------------------------------------------------------ items

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ri, e, t0 = self.index[i]
        ep = self.eps[ri]
        L = self.seq_len
        t1 = t0 + L

        mu = ep.mu[e, t0 : t1 + 1]            # (L+1, z) -- inputs AND targets
        if self.use_mean:
            zz = mu
        else:
            lv = ep.logvar[e, t0 : t1 + 1]
            # One eps draw per (time, dim). Note we sample the input z_t and the
            # target z_{t+1} independently even though they are the same cached
            # frame in consecutive windows -- that is correct: each is an
            # independent draw from that frame's posterior.
            noise = self._rng.standard_normal(mu.shape).astype(np.float32)
            zz = mu + noise * np.exp(0.5 * lv)

        a = ep.actions[e, t0:t1]              # (L,)
        a_onehot = np.zeros((L, N_ACTIONS), dtype=np.float32)
        a_onehot[np.arange(L), a] = 1.0

        return {
            "z": torch.from_numpy(np.ascontiguousarray(zz[:-1])),
            "a": torch.from_numpy(a_onehot),
            "a_idx": torch.from_numpy(np.ascontiguousarray(a)),
            "z_next": torch.from_numpy(np.ascontiguousarray(zz[1:])),
            "hit": torch.from_numpy(np.ascontiguousarray(ep.hit[e, t0:t1])),
            "reward": torch.from_numpy(np.ascontiguousarray(ep.reward[e, t0:t1])),
            # Diagnostics only. Never reaches the model.
            "state": torch.from_numpy(
                np.ascontiguousarray(ep.state[e, t0 + 1 : t1 + 1])
            ),
        }


def make_seq_loader(
    roots,
    seq_len: int = 32,
    batch_size: int = 64,
    shuffle: bool = True,
    stride: int = 1,
    use_mean: bool = False,
    seed: int | None = None,
    num_workers: int = 0,
    latent_suffix: str = "",
) -> DataLoader:
    ds = LatentSequenceDataset(
        roots, seq_len=seq_len, stride=stride, use_mean=use_mean, seed=seed,
        latent_suffix=latent_suffix,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    return loader


# ------------------------------------------------------------------ rollouts


def episode_arrays(root: str | Path, latent_suffix: str = "") -> Dict[str, np.ndarray]:
    """Whole-episode arrays for evaluation (dream rollouts, probes).

    Deliberately separate from the Dataset: evaluation wants whole episodes and
    the posterior MEAN (that is the best single estimate of the true latent, and
    for a rollout warm-up we want the cleanest possible starting point), whereas
    training wants short windows and samples.
    """
    ep = _load_root(root, latent_suffix)
    return {
        "mu": ep.mu,
        "logvar": ep.logvar,
        "actions": ep.actions,
        "hit": ep.hit,
        "reward": ep.reward,
        "state": ep.state,
        "meta": json.loads((Path(root) / "meta.json").read_text()),
    }

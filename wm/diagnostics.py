"""Diagnostics. These matter more than the architecture.

Global loss is nearly useless on this problem (see masked_recon_error), so the
question "is my VAE working" has to be answered by these four instead:

  1. reconstruction_grid  -- did it keep the ball at all
  2. linear_probe         -- are the true factors recoverable from mu
  3. latent_traversal     -- are they axis-aligned and interpretable
  4. active_units         -- did it find the right latent dimensionality
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch


# --------------------------------------------------------------- encoding


@torch.no_grad()
def encode_dataset(
    model, loader, device: str = "cpu", limit: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Run the encoder over a loader that yields (x, state). Returns mu/logvar/state."""
    model.eval()
    mus, logvars, states = [], [], []
    seen = 0
    for x, s in loader:
        mu, logvar = model.encode(x.to(device))
        mus.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
        states.append(s.numpy())
        seen += x.shape[0]
        if limit is not None and seen >= limit:
            break
    return {
        "mu": np.concatenate(mus)[:limit],
        "logvar": np.concatenate(logvars)[:limit],
        "state": np.concatenate(states)[:limit],
    }


# ------------------------------------------------------------ active units


def active_units(
    logvar: np.ndarray, mu: np.ndarray, thresh: float = 0.01, free_bits: float = 0.0
):
    """Per-dimension KL, and how many dimensions carry information.

    A dimension the KL has driven to the prior has mu ~ 0 and logvar ~ 0, and
    contributes ~0 nats. Counting the rest measures the latent dimensionality
    the model actually settled on -- which for this data should land near 3.

    IMPORTANT with free bits: the clamp holds every dimension at >= free_bits
    nats, so a threshold below that reports 16/16 active no matter what. The
    threshold has to sit above the floor to mean anything.
    """
    thresh = max(thresh, free_bits * 1.5)
    kl = (-0.5 * (1.0 + logvar - mu**2 - np.exp(logvar))).mean(axis=0)
    order = np.argsort(-kl)
    return {
        "kl_per_dim": kl,
        "n_active": int((kl > thresh).sum()),
        "order": order,  # dims sorted most- to least-informative
    }


# ------------------------------------------------------------- linear probe


def linear_probe(
    mu: np.ndarray,
    state: np.ndarray,
    state_names: Sequence[str],
    ridge: float = 1e-6,
) -> Dict[str, float]:
    """OLS from mu to each ground-truth variable, reported as R^2.

    This is the real success metric. Note what we expect:
      ball_x, ball_y, paddle_x  -> R^2 > 0.98   (present in a single frame)
      ball_vx, ball_vy          -> R^2 ~ 0      (NOT present in a single frame)

    The near-zero velocity R^2 is the correct answer, not a failure. A static
    frame is pixel-identical whether the ball moves up or down, so velocity is
    the dynamics model's job. Getting a high velocity R^2 here would actually
    mean something was leaking -- e.g. positional bias correlated with speed.
    """
    X = np.concatenate([mu, np.ones((len(mu), 1), dtype=mu.dtype)], axis=1)
    XtX = X.T @ X + ridge * np.eye(X.shape[1], dtype=X.dtype)
    W = np.linalg.solve(XtX, X.T @ state)
    pred = X @ W
    ss_res = ((state - pred) ** 2).sum(axis=0)
    ss_tot = ((state - state.mean(0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {name: float(v) for name, v in zip(state_names, r2)}


def nonlinear_probe(
    mu: np.ndarray,
    state: np.ndarray,
    state_names: Sequence[str],
    hidden: int = 64,
    seed: int = 0,
) -> Dict[str, float]:
    """Same thing with a small MLP. If nonlinear >> linear, the factor is in mu
    but encoded on a curved manifold rather than a linear subspace."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    n = len(mu)
    idx = np.random.default_rng(seed).permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n) :]
    sx = StandardScaler().fit(mu[tr])
    out = {}
    for j, name in enumerate(state_names):
        sy = StandardScaler().fit(state[tr, j : j + 1])
        m = MLPRegressor(
            hidden_layer_sizes=(hidden, hidden), max_iter=400, random_state=seed
        )
        m.fit(sx.transform(mu[tr]), sy.transform(state[tr, j : j + 1]).ravel())
        p = m.predict(sx.transform(mu[te]))
        y = sy.transform(state[te, j : j + 1]).ravel()
        out[name] = float(1.0 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    return out


# ----------------------------------------------------------------- visuals


def _to_img(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) float in [0,1] -> (H, W, C) uint8."""
    a = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (a * 255 + 0.5).astype(np.uint8)


@torch.no_grad()
def reconstruction_grid(
    model, x: torch.Tensor, device: str = "cpu", pad: int = 2, use_mean: bool = True
) -> np.ndarray:
    """Originals on the top row, reconstructions below. Look at this often."""
    model.eval()
    x = x.to(device)
    mu, logvar = model.encode(x)
    # Decode mu rather than a sample: we want to see what the model believes,
    # not what one noisy draw looks like.
    z = mu if use_mean else model.reparameterize(mu, logvar)
    xh = model.decode(z)

    n = x.shape[0]
    h = w = x.shape[-1]
    out = np.full((2 * h + pad, n * (w + pad) - pad, 3), 40, np.uint8)
    for i in range(n):
        x0 = i * (w + pad)
        out[:h, x0 : x0 + w] = _to_img(x[i])
        out[h + pad :, x0 : x0 + w] = _to_img(xh[i])
    return out


@torch.no_grad()
def latent_traversal(
    model,
    x: torch.Tensor,
    dims: Sequence[int],
    span: float = 3.0,
    steps: int = 9,
    device: str = "cpu",
    pad: int = 2,
) -> np.ndarray:
    """One row per latent dim, sweeping it from -span to +span.

    This is the visual form of the identifiability question: did the model
    recover axis-aligned interpretable factors (one dim moves the ball
    horizontally, another vertically, another slides the paddle), or an
    entangled rotation of them where every dim moves everything at once?
    """
    model.eval()
    mu, _ = model.encode(x[:1].to(device))
    vals = torch.linspace(-span, span, steps, device=mu.device)

    res = x.shape[-1]
    rows = []
    for d in dims:
        zs = mu.repeat(steps, 1)
        zs[:, d] = vals
        imgs = model.decode(zs)
        row = np.full((res, steps * (res + pad) - pad, 3), 40, np.uint8)
        for i in range(steps):
            row[:, i * (res + pad) : i * (res + pad) + res] = _to_img(imgs[i])
        rows.append(row)

    W = rows[0].shape[1]
    out = np.full((len(rows) * (res + pad) - pad, W, 3), 40, np.uint8)
    for i, r in enumerate(rows):
        out[i * (res + pad) : i * (res + pad) + res] = r
    return out


def save_png(arr: np.ndarray, path: str | Path, scale: int = 3) -> Path:
    from PIL import Image

    if scale > 1:
        arr = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path

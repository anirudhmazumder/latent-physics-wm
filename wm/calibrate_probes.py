"""Calibrate the probe suite against synthetic codes of KNOWN functional form.

Shared evaluation machinery: the calibration table that makes every probe
number in `docs/02_vae_the_vision_model.md` and the per-tier V documents
readable. Not tied to one tier.

    python -m wm.calibrate_probes

Why this exists. A probe number on its own is uninterpretable. "Linear R^2 = 0.31"
does not tell you what the encoder is computing, because you have no idea what
0.31 corresponds to. So build codes whose form you already know, run the exact
same probes on them, and read your real result off the resulting table.

Reference output (3000 samples, episode-level split, 16 latent dims):

    code                      linear   poly3    knn
    place fields  w=0.30        1.00    1.00    1.00
    place fields  w=0.18        0.96    1.00    1.00
    place fields  w=0.10        0.84    0.99    1.00
    place fields  w=0.05        0.41    0.71    0.95
    fourier  k=1                0.99    1.00    1.00
    fourier  k=2                0.61    0.82    0.93
    fourier  k=3                0.01    0.07   -0.02

Two things to take from that table.

First, broad place fields are almost LINEARLY decodable. This is unintuitive and
it kills the lazy inference. A tiled set of wide Gaussian bumps supports a
near-perfect linear readout of position, because a weighted sum of overlapping
bumps is a smooth ramp. So a low linear R^2 does NOT imply "place-field code" --
it implies place fields that are *narrow relative to the box*, roughly w < 0.06,
which at 64x64 is about 3 pixels.

Second, the k=3 row is the interesting failure: every probe including the
nonparametric one reads ~0. That is a NON-INJECTIVE code. cos(3*pi*x) takes the
same value at several distinct x, so no function of it can recover x -- the
information is destroyed, not hidden. kNN reading zero while reconstruction
looks fine would mean position is encoded jointly across dimensions in a way
that no single-target regression can unpick, or that your probe set is missing
the dimensions that carry the phase.

So the decisive discriminator is kNN, not linear:

    linear low + knn high  -> information present, geometry localised/curved
    linear low + knn low   -> information folded, or spread across dims such
                              that marginal regression cannot find it
"""

from __future__ import annotations

import argparse

import numpy as np

from .probes import probe_suite


def place_field_code(x, y, n_side: int = 4, width: float = 0.18) -> np.ndarray:
    cx, cy = np.meshgrid(
        np.linspace(0.15, 0.85, n_side), np.linspace(0.15, 0.85, n_side)
    )
    return np.stack(
        [
            np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * width**2))
            for a, b in zip(cx.ravel(), cy.ravel())
        ],
        axis=1,
    )


def fourier_code(x, y, k: int = 1) -> np.ndarray:
    return np.stack(
        [
            np.cos(k * np.pi * x),
            np.sin(k * np.pi * x),
            np.cos(k * np.pi * y),
            np.sin(k * np.pi * y),
        ],
        axis=1,
    )


def linear_rotated_code(x, y, z_dim: int = 16, seed: int = 0) -> np.ndarray:
    """Position under a random linear embedding into z_dim dimensions.

    The null hypothesis worth having on the table: a model can be perfectly
    "entangled" in the sense that no single dimension means anything, while
    still being trivially LINEARLY decodable. Entanglement and nonlinearity are
    different properties and this code separates them -- expect linear ~1.00 and
    MCC well below 1.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(2, z_dim))
    return np.stack([x, y], 1) @ A


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--episode-len", type=int, default=25)
    p.add_argument("--noise", type=float, default=0.0,
                   help="std of gaussian noise added to every code dim")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    x = rng.uniform(0, 1, a.n)
    y = rng.uniform(0, 1, a.n)
    state = np.stack([x, y], 1)
    names = ["ball_x", "ball_y"]
    groups = np.arange(a.n) // a.episode_len

    codes = {}
    for w in (0.30, 0.18, 0.10, 0.05, 0.03):
        codes[f"place fields  w={w:.2f}"] = place_field_code(x, y, 4, w)
    for k in (1, 2, 3):
        codes[f"fourier  k={k}"] = fourier_code(x, y, k)
    codes["linear rotated"] = linear_rotated_code(x, y)

    which = ("linear", "poly2", "poly3", "knn")
    print(f"{'code':26s}" + "".join(f"{k:>8s}" for k in which))
    for tag, mu in codes.items():
        if a.noise > 0:
            mu = mu + rng.normal(0, a.noise, mu.shape)
        s = probe_suite(mu, state, names, group_ids=groups, seed=a.seed, which=which)
        # average the two factors, they are symmetric by construction
        cells = "".join(
            f"{0.5 * (s[k]['ball_x'] + s[k]['ball_y']):8.2f}" for k in which
        )
        print(f"{tag:26s}{cells}")


if __name__ == "__main__":
    main()

"""Probing a frozen latent space for ground-truth factors.

The central question this file answers is not "is the information there" but
"in what form is it there". Those are different questions and they need
different probes:

    linear      is the factor a linear functional of mu?
    poly2/3     is it a low-order polynomial (a smooth curve/fold)?
    knn         is it recoverable at all, by any smooth function? (nonparametric)
    mlp         is it recoverable by a flexible learned function?

Reading the pattern:
    linear high                      -> axis-aligned-ish linear code
    linear low,  poly high           -> smooth low-order curvature
    linear low,  poly low, knn high  -> strongly nonlinear / localised code
                                        (place-field-like, or a high-frequency
                                        multiplexed code)
    all low                          -> the information genuinely is not in mu

METHODOLOGICAL POINT, and the reason this file exists separately from
diagnostics.linear_probe: every probe here is fit on a train split and scored on
a held-out split, using THE SAME split for all probes. The quick OLS in
diagnostics.py reports in-sample R^2, which is optimistically biased. Comparing
an in-sample OLS number against an out-of-sample MLP number is not a valid
comparison -- it stacks the deck in favour of the linear probe and therefore
*understates* any nonlinearity you find. Use this module for any conclusion you
intend to act on.

A second methodological point about sample size: frames within an episode are
heavily autocorrelated, so 1200 frames from 8 episodes is nowhere near 1200
independent samples. The effective sample size is closer to the number of
episodes times the number of independent excursions across the box. Probe on a
val set with many short episodes rather than few long ones, and treat R^2
differences under ~0.03 as noise unless you have run several seeds.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------- split logic


def make_split(
    n: int, frac_train: float = 0.8, seed: int = 0, group_ids: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx).

    If ``group_ids`` is given (e.g. episode index per frame), the split is made
    at the GROUP level so no episode appears in both halves. With autocorrelated
    frames a random per-frame split leaks: the test frame at t=51 is nearly
    identical to the train frame at t=50, and every probe -- especially kNN --
    scores far above its true generalisation. Always pass group_ids if you have
    them.
    """
    rng = np.random.default_rng(seed)
    if group_ids is None:
        idx = rng.permutation(n)
        k = int(frac_train * n)
        return idx[:k], idx[k:]

    groups = np.unique(group_ids)
    rng.shuffle(groups)
    k = max(1, int(frac_train * len(groups)))
    tr_groups = set(groups[:k].tolist())
    mask = np.array([g in tr_groups for g in group_ids])
    return np.where(mask)[0], np.where(~mask)[0]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# -------------------------------------------------------------- the probes


def _fit_linear(Xtr, ytr, Xte, ridge: float = 1e-6):
    X = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ ytr)
    Xt = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    return Xt @ w


def probe_suite(
    mu: np.ndarray,
    state: np.ndarray,
    state_names: Sequence[str],
    group_ids: Optional[np.ndarray] = None,
    seed: int = 0,
    which: Sequence[str] = ("linear", "poly2", "poly3", "knn", "mlp"),
) -> Dict[str, Dict[str, float]]:
    """Held-out R^2 for each (probe, factor) pair. Returns probe -> factor -> R^2."""
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    mu = np.asarray(mu, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    tr, te = make_split(len(mu), seed=seed, group_ids=group_ids)

    # Standardise on TRAIN statistics only. Fitting the scaler on the full set
    # is a small but real leak, and it is the kind that silently flatters every
    # probe equally so you never notice it.
    sx = StandardScaler().fit(mu[tr])
    Xtr, Xte = sx.transform(mu[tr]), sx.transform(mu[te])

    out: Dict[str, Dict[str, float]] = {k: {} for k in which}

    poly_cache = {}
    for deg in (2, 3):
        if f"poly{deg}" in which:
            pf = PolynomialFeatures(deg, include_bias=False).fit(Xtr)
            poly_cache[deg] = (pf.transform(Xtr), pf.transform(Xte))

    for j, name in enumerate(state_names):
        ytr, yte = state[tr, j], state[te, j]

        if "linear" in which:
            out["linear"][name] = _r2(yte, _fit_linear(Xtr, ytr, Xte))

        for deg in (2, 3):
            key = f"poly{deg}"
            if key in which:
                Ptr, Pte = poly_cache[deg]
                # Ridge, not OLS: degree 3 on 16 inputs is 968 features, which
                # will happily interpolate noise without regularisation.
                out[key][name] = _r2(yte, _fit_linear(Ptr, ytr, Pte, ridge=1.0))

        if "knn" in which:
            from sklearn.neighbors import KNeighborsRegressor

            # Nonparametric, so it answers "is the information present under ANY
            # smooth map" without assuming a functional form. This is your
            # upper bound. If knn is high and linear is low, the information is
            # definitely there and the question is purely about geometry.
            m = KNeighborsRegressor(n_neighbors=10, weights="distance").fit(Xtr, ytr)
            out["knn"][name] = _r2(yte, m.predict(Xte))

        if "mlp" in which:
            from sklearn.neural_network import MLPRegressor

            sy_mean, sy_std = ytr.mean(), ytr.std() + 1e-12
            m = MLPRegressor(
                hidden_layer_sizes=(64, 64), max_iter=600,
                random_state=seed, early_stopping=True,
            ).fit(Xtr, (ytr - sy_mean) / sy_std)
            out["mlp"][name] = _r2(yte, m.predict(Xte) * sy_std + sy_mean)

    return out


# --------------------------------------------------------------------- MCC


def mcc(
    mu: np.ndarray, state: np.ndarray, state_names: Sequence[str]
) -> Dict[str, object]:
    """Mean Correlation Coefficient with optimal one-to-one matching.

    The standard summary metric in the identifiability literature (TCL, iVAE,
    CITRIS all report a variant). Procedure: correlate every latent dimension
    with every true factor, then use the Hungarian algorithm to find the
    assignment of latents to factors maximising total |correlation|, then
    average.

    What it measures, precisely: whether the model recovered the true factors up
    to *permutation*. That is the identifiability guarantee those papers prove --
    recovery up to permutation and elementwise transformation, never up to the
    identity.

    What it does NOT measure, and the limitation to keep in mind: Pearson
    correlation is a linear statistic, so a dimension that encodes ball_x
    perfectly but as cos(pi * x) scores ~0 here. MCC therefore conflates "did
    not recover the factor" with "recovered it under a non-monotone transform".
    Read MCC alongside the probe suite, never alone. If knn R^2 is high and MCC
    is low, you have recovery up to a nonlinear transform -- which is exactly
    what the theory permits and what MCC is blind to.
    """
    from scipy.optimize import linear_sum_assignment

    mu = np.asarray(mu, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    # (z_dim, n_factors) absolute correlation matrix
    C = np.zeros((mu.shape[1], state.shape[1]))
    for i in range(mu.shape[1]):
        for j in range(state.shape[1]):
            sd = mu[:, i].std() * state[:, j].std()
            if sd < 1e-12:
                continue
            C[i, j] = abs(np.corrcoef(mu[:, i], state[:, j])[0, 1])

    rows, cols = linear_sum_assignment(-C)
    matched = {state_names[j]: (int(i), float(C[i, j])) for i, j in zip(rows, cols)}
    return {
        "mcc": float(C[rows, cols].mean()),
        "corr_matrix": C,
        "matched": matched,  # factor -> (latent dim, |corr|)
    }


def spearman_mcc(mu: np.ndarray, state: np.ndarray, state_names: Sequence[str]):
    """MCC on ranks instead of values.

    Worth running next to the Pearson version. Spearman is invariant to any
    monotone transform, so a large gap (spearman >> pearson) means the code is
    monotone but curved -- a squashing nonlinearity. If BOTH are low while knn
    R^2 is high, the map is non-monotone: folded, periodic, or localised.
    That single comparison narrows the hypothesis space a lot.
    """
    from scipy.stats import rankdata

    mu_r = np.apply_along_axis(rankdata, 0, mu)
    st_r = np.apply_along_axis(rankdata, 0, state)
    return mcc(mu_r, st_r, state_names)


# ------------------------------------------------------------- tuning maps


def tuning_maps(
    mu: np.ndarray,
    state: np.ndarray,
    dims: Sequence[int],
    bins: int = 16,
    x_col: int = 0,
    y_col: int = 1,
    min_count: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean value of each latent dim as a function of true ball (x, y).

    Returns ``(maps, counts)`` with maps of shape ``(len(dims), bins, bins)``,
    NaN where a bin has fewer than ``min_count`` samples.

    This is the decisive plot for the low-linear/high-knn case, because the
    hypotheses look completely different here:

      linear code      a smooth left-right or top-bottom gradient
      curved monotone  a gradient that compresses at one end
      place field      a single localised blob of activation
      periodic         stripes or a checkerboard
      radial           concentric rings

    A probe tells you a linear map failed. This tells you what the encoder is
    actually computing instead, which is the thing you want to know.
    """
    mu = np.asarray(mu)
    x, y = np.asarray(state)[:, x_col], np.asarray(state)[:, y_col]
    xi = np.clip((x * bins).astype(int), 0, bins - 1)
    yi = np.clip((y * bins).astype(int), 0, bins - 1)

    counts = np.zeros((bins, bins))
    np.add.at(counts, (yi, xi), 1.0)

    maps = np.zeros((len(dims), bins, bins))
    for k, d in enumerate(dims):
        acc = np.zeros((bins, bins))
        np.add.at(acc, (yi, xi), mu[:, d])
        with np.errstate(invalid="ignore", divide="ignore"):
            m = acc / counts
        m[counts < min_count] = np.nan
        # Row 0 is the top of the image, so flip to match how you view frames.
        maps[k] = m[::-1]
    return maps, counts[::-1]


def save_tuning_grid(
    maps: np.ndarray,
    dims: Sequence[int],
    path,
    ncols: int = 4,
    titles: Optional[Sequence[str]] = None,
):
    """Render tuning maps as a labelled grid. Needs matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(dims)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for k in range(len(axes)):
        ax = axes[k]
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= n:
            ax.axis("off")
            continue
        # Symmetric colour limits about zero so sign is readable.
        v = np.nanmax(np.abs(maps[k])) or 1.0
        ax.imshow(maps[k], cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(titles[k] if titles else f"z[{dims[k]}]", fontsize=9)
    fig.suptitle("latent value vs true ball position (x right, y up)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

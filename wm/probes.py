"""Probing a frozen latent space for ground-truth factors.

Shared evaluation machinery, used by every tier (v1-v4). The results it
produces are read in `docs/02_vae_the_vision_model.md` and each tier's
`0*_..._env_data_vae.md`.

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
    mu = np.asarray(mu, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    tr, te = make_split(len(mu), seed=seed, group_ids=group_ids)
    return _fit_probes(mu[tr], state[tr], mu[te], state[te], state_names, which, seed)


def probe_transfer(
    mu_fit: np.ndarray,
    state_fit: np.ndarray,
    mu_eval: np.ndarray,
    state_eval: np.ndarray,
    state_names: Sequence[str],
    seed: int = 0,
    which: Sequence[str] = ("linear", "poly2", "poly3", "knn", "mlp"),
) -> Dict[str, Dict[str, float]]:
    """Fit probes on one dataset, score them on a DIFFERENT one.

    Same machinery as ``probe_suite``, but the train/test boundary is a
    distribution shift you chose rather than a random split. This is how you ask
    a generalisation question instead of a decodability question.

    v2 uses it for the mass hold-out: probes are fit on frames whose ball mass
    was never in [0.85, 1.2] and evaluated on frames where it always is. A high
    R^2 means the latent code for colour is a genuine continuum that interpolates
    into a band it never saw; a low R^2 with high in-distribution R^2 means the
    model memorised the colours it was shown, which is a much weaker claim and
    would be a real limitation to know about before building M on top of it.

    R^2 here is computed against the EVAL set's own mean, so a probe that is
    perfect in-distribution but predicts a constant on the new band scores ~0,
    and one that is systematically biased on the new band scores below 0.
    """
    return _fit_probes(
        np.asarray(mu_fit, dtype=np.float64),
        np.asarray(state_fit, dtype=np.float64),
        np.asarray(mu_eval, dtype=np.float64),
        np.asarray(state_eval, dtype=np.float64),
        state_names,
        which,
        seed,
    )


def _fit_probes(
    mu_tr: np.ndarray,
    state_tr: np.ndarray,
    mu_te: np.ndarray,
    state_te: np.ndarray,
    state_names: Sequence[str],
    which: Sequence[str],
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Shared body of probe_suite / probe_transfer: fit on tr, score on te."""
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    # Standardise on TRAIN statistics only. Fitting the scaler on the full set
    # is a small but real leak, and it is the kind that silently flatters every
    # probe equally so you never notice it.
    sx = StandardScaler().fit(mu_tr)
    Xtr, Xte = sx.transform(mu_tr), sx.transform(mu_te)

    out: Dict[str, Dict[str, float]] = {k: {} for k in which}

    poly_cache = {}
    for deg in (2, 3):
        if f"poly{deg}" in which:
            pf = PolynomialFeatures(deg, include_bias=False).fit(Xtr)
            poly_cache[deg] = (pf.transform(Xtr), pf.transform(Xte))

    for j, name in enumerate(state_names):
        ytr, yte = state_tr[:, j], state_te[:, j]

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


# ------------------------------------------------------- classification probes


def classification_suite(
    mu: np.ndarray,
    y: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    seed: int = 0,
    which: Sequence[str] = ("logistic", "knn"),
    n_neighbors: int = 10,
    n_shuffles: int = 0,
) -> Dict[str, float]:
    """Held-out ACCURACY for a discrete factor, with the majority baseline.

    Introduced for v4, where the factor of interest -- the direction of gravity
    -- is a single bit rather than a real number. R² is the wrong lens for a
    bit: it is not bounded below in a way anyone can read, it has no natural
    null value, and a probe that predicts the majority class everywhere scores
    exactly 0, which looks like "no information" and is in fact "no information
    *beyond the base rate*" -- a distinction that matters enormously when the
    base rate is not 50/50. Accuracy against ``majority`` says both things at
    once.

    The two probes answer different questions, the same way the regression
    suite's do:

        ``logistic``  is the bit a linear functional of mu? (a hyperplane)
        ``knn``       is it recoverable AT ALL, by any local map? The upper
                      bound, and the one that matters for a negative result --
                      "not linearly decodable" is a much weaker claim than
                      "not decodable", and only the second one is evidence that
                      the frames genuinely do not contain the bit.

    ``group_ids`` should be episode indices. Without them the split leaks
    catastrophically here: ``gravity_sign`` is *constant within an episode
    between flips*, so a per-frame split puts near-duplicate frames carrying the
    same label on both sides and kNN scores near 1.0 on pure autocorrelation.
    That is not a subtle bias, it is the whole answer.
    """
    from sklearn.preprocessing import StandardScaler

    mu = np.asarray(mu, dtype=np.float64)
    y = np.asarray(y).ravel()
    tr, te = make_split(len(mu), seed=seed, group_ids=group_ids)
    sx = StandardScaler().fit(mu[tr])
    Xtr, Xte = sx.transform(mu[tr]), sx.transform(mu[te])
    ytr, yte = y[tr], y[te]

    vals, counts = np.unique(ytr, return_counts=True)
    majority = float((yte == vals[np.argmax(counts)]).mean())
    out: Dict[str, float] = {
        "majority": majority,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "test_base_rate": float((yte == vals[np.argmax(counts)]).mean()),
    }
    if len(vals) < 2:
        out["note"] = "only one class present; no probe is meaningful"
        return out

    if "logistic" in which:
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        out["logistic"] = float(m.score(Xte, yte))
    if "knn" in which:
        from sklearn.neighbors import KNeighborsClassifier

        m = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights="distance"
        ).fit(Xtr, ytr)
        out["knn"] = float(m.score(Xte, yte))

    if n_shuffles > 0:
        null = shuffled_label_null(
            mu, y, group_ids=group_ids, seed=seed, which=which,
            n_neighbors=n_neighbors, n_shuffles=n_shuffles,
        )
        out.update(null)
    return out


def shuffled_label_null(
    mu: np.ndarray,
    y: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    seed: int = 0,
    which: Sequence[str] = ("logistic", "knn"),
    n_neighbors: int = 10,
    n_shuffles: int = 5,
) -> Dict[str, float]:
    """The same probes, on labels that have been detached from the features.

    Why the majority-class rate is not enough on its own. A held-out accuracy is
    a random variable, and the two things that inflate it here are not visible
    in the base rate:

    * the probe is fitted on ~100 features and can overfit a finite training
      set in ways that happen to transfer if the split is small;
    * the frames are not independent. An episode's labels come in long runs, so
      the *effective* sample size is the number of runs, not the number of
      frames, and the sampling spread of a held-out accuracy around 0.5 is far
      wider than a binomial on 20,000 frames would suggest.

    Shuffling the labels kills any real relationship while keeping both effects
    exactly: same feature matrix, same split, same classifier, same class
    balance, same run-length structure. Whatever accuracy comes back is what
    "nothing" looks like on this data, and a measured accuracy is only evidence
    if it clears it.

    The shuffle is done at the GROUP level when groups are available: whole
    episodes' label sequences are permuted between episodes rather than frames
    being shuffled individually. A per-frame shuffle would destroy the run
    structure and produce an optimistically tight null -- the very thing the
    test is supposed to account for. (Episodes of unequal length fall back to a
    per-frame permutation, and the returned dict says so.)
    """
    mu = np.asarray(mu, dtype=np.float64)
    y = np.asarray(y).ravel()
    rng = np.random.default_rng(seed + 991)

    blocks: Optional[list] = None
    if group_ids is not None:
        order = [np.where(group_ids == g)[0] for g in np.unique(group_ids)]
        if len({len(o) for o in order}) == 1 and len(order) > 2:
            blocks = order

    accs: Dict[str, list] = {k: [] for k in which}
    for s in range(n_shuffles):
        if blocks is not None:
            perm = rng.permutation(len(blocks))
            y_s = y.copy()
            for dst, src in enumerate(perm):
                y_s[blocks[dst]] = y[blocks[src]]
        else:
            y_s = y[rng.permutation(len(y))]
        rec = classification_suite(
            mu, y_s, group_ids=group_ids, seed=seed + s, which=which,
            n_neighbors=n_neighbors, n_shuffles=0,
        )
        for k in which:
            if k in rec:
                accs[k].append(rec[k])

    out: Dict[str, float] = {
        "null_shuffle_blocks": "episode" if blocks is not None else "frame",
        "null_n_shuffles": int(n_shuffles),
    }
    for k, v in accs.items():
        if v:
            out[f"null_{k}_mean"] = float(np.mean(v))
            out[f"null_{k}_max"] = float(np.max(v))
    return out


def balance_within_position_bins(
    state: np.ndarray,
    y: np.ndarray,
    bins: int = 8,
    seed: int = 0,
    x_col: int = 0,
    y_col: int = 1,
) -> np.ndarray:
    """Indices of a subsample in which the two classes are equally frequent
    *inside every position bin*.

    The diagnostic for a suspected leak THROUGH position. A hidden dynamical
    variable can be perfectly invisible per frame and still be decodable from
    one, because it changes where the ball tends to BE -- under gravity pulling
    down the ball dawdles near the ceiling and hurries past the floor, so the
    marginal distribution of ``ball_y`` differs by sign even though no single
    frame shows the sign. An encoder that codes position (which is its job)
    then hands a probe a legitimate, and completely uninteresting, route to
    above-chance accuracy.

    Matching on position closes that route: within a bin the two classes are
    equinumerous by construction, so a probe reading only position can do no
    better than chance. If accuracy survives the matching, something other
    than position carries the bit; if it collapses to chance, the leak is
    explained and the frames really are uninformative.

    Bins with only one class present contribute nothing and are dropped, which
    is correct -- there is no matched comparison to be made there.
    """
    rng = np.random.default_rng(seed)
    state = np.asarray(state)
    y = np.asarray(y).ravel()
    xi = np.clip((state[:, x_col] * bins).astype(int), 0, bins - 1)
    yi = np.clip((state[:, y_col] * bins).astype(int), 0, bins - 1)
    key = yi * bins + xi

    keep = []
    classes = np.unique(y)
    for b in np.unique(key):
        m = np.flatnonzero(key == b)
        per = [m[y[m] == c] for c in classes]
        n = min(len(g) for g in per)
        if n == 0:
            continue
        for g in per:
            keep.append(rng.choice(g, size=n, replace=False))
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, np.int64)


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


def mass_tuning(
    mu: np.ndarray,
    mass: np.ndarray,
    dims: Sequence[int],
    bins: int = 12,
    min_count: int = 3,
):
    """Mean of each latent dim as a function of log-mass. v2's version of a tuning map.

    Binned in LOG mass because that is the variable the colour ramp is linear
    in (u = log m rescaled), so a dimension that codes colour linearly shows up
    here as a straight line. A dimension that codes it with a saturating or
    folded shape shows up as a curve or a hump, which is the same
    linear-vs-nonlinear question the probe suite asks, but visible.

    Returns ``(centres, curves, counts)`` with curves of shape
    ``(len(dims), bins)``, NaN where a bin is too sparse to trust.
    """
    mu = np.asarray(mu, dtype=np.float64)
    lm = np.log(np.asarray(mass, dtype=np.float64))
    edges = np.linspace(lm.min(), lm.max() + 1e-12, bins + 1)
    idx = np.clip(np.digitize(lm, edges) - 1, 0, bins - 1)

    counts = np.zeros(bins)
    np.add.at(counts, idx, 1.0)
    centres = 0.5 * (edges[:-1] + edges[1:])

    curves = np.full((len(dims), bins), np.nan)
    for k, d in enumerate(dims):
        acc = np.zeros(bins)
        np.add.at(acc, idx, mu[:, d])
        with np.errstate(invalid="ignore", divide="ignore"):
            c = acc / counts
        c[counts < min_count] = np.nan
        curves[k] = c
    return centres, curves, counts


def save_mass_tuning(centres, curves, dims: Sequence[int], path, holdout=None):
    """Render mass_tuning as one line per latent dim. Needs matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, d in enumerate(dims):
        ax.plot(np.exp(centres), curves[k], marker="o", ms=3, label=f"z[{d}]")
    if holdout is not None:
        # Shade the band the training data never contained, so you can see at a
        # glance whether the curves are interpolating across a gap or across
        # data they actually saw.
        ax.axvspan(holdout[0], holdout[1], color="0.85", zorder=0,
                   label="held-out band")
    ax.set_xscale("log")
    ax.set_xlabel("ball mass (log scale) -- yellow/light on the left, purple/heavy on the right")
    ax.set_ylabel("mean mu in bin")
    ax.set_title("latent tuning to mass (i.e. to ball colour)")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


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

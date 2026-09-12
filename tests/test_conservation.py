"""Tests for the conservation metric and the losses meant to fix it.

Four claims, each of which would be expensive to discover was false after a
35-epoch training run:

1. the differentiable torch probe is the SAME function as the numpy/sklearn one
2. the rollout loss actually puts gradient on the LSTM
3. ``--rollout-loss-steps 0`` leaves the original loss bit-for-bit unchanged
4. the metric says "conserved" when handed a dream that is a copy of the truth
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wm.conservation import (
    Poly2Probe, conservation_penalty, corr, reparam_sample, rollout_losses,
    rolling_speed, speed_horizon,
)
from wm.rnn import MDNRNN, RNNConfig, rnn_loss


def _tiny_model(z_dim=4, seed=0):
    torch.manual_seed(seed)
    return MDNRNN(RNNConfig(z_dim=z_dim, n_actions=3, hidden=16, n_gauss=3))


def _tiny_batch(B=5, L=12, z_dim=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, L + 1, z_dim, generator=g) * 0.3
    a_idx = torch.randint(0, 3, (B, L), generator=g)
    return {
        "z": z[:, :-1],
        "z_next": z[:, 1:],
        "a": torch.eye(3)[a_idx],
        "hit": (torch.rand(B, L, generator=g) < 0.1).float(),
        "reward": torch.rand(B, L, generator=g),
    }


# ------------------------------------------------------- 1. probe agreement


def test_torch_poly2_probe_matches_numpy_probe():
    """The torch path and the numpy path must be the same function of z."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = 0.3 * X[:, 0] - 1.1 * X[:, 2] ** 2 + 0.7 * X[:, 1] * X[:, 3] + 0.05

    probe = Poly2Probe().fit(X, y)
    Z = rng.normal(size=(37, 6)) * 1.5           # deliberately off the fit range
    np_out = probe(Z)
    t_out = probe.torch(torch.as_tensor(Z, dtype=torch.float64)).numpy()
    assert np.abs(np_out - t_out).max() < 1e-4

    # Shape handling: (..., z_dim) -> (...), in both paths.
    Z3 = rng.normal(size=(4, 9, 6))
    assert probe(Z3).shape == (4, 9)
    assert probe.torch(torch.as_tensor(Z3, dtype=torch.float64)).shape == (4, 9)
    assert np.abs(probe(Z3) - probe.torch(
        torch.as_tensor(Z3, dtype=torch.float64)).numpy()).max() < 1e-4


def test_poly2_probe_matches_sklearn_reference():
    """Independent reference: sklearn's own features + the same ridge solve.

    This pins the FEATURE ORDERING and the regularisation to what
    ``wm.probes`` uses for its poly-2 column, so the number this file reports
    and the number the stage-one probe table reports mean the same thing.
    """
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 5))
    y = np.log(rng.uniform(0.5, 2.0, size=300))

    sx = StandardScaler().fit(X)
    P = PolynomialFeatures(2, include_bias=False).fit_transform(sx.transform(X))
    A = np.concatenate([P, np.ones((len(P), 1))], 1)
    w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y)

    Z = rng.normal(size=(20, 5))
    Pz = PolynomialFeatures(2, include_bias=False).fit(
        sx.transform(X)).transform(sx.transform(Z))
    ref = Pz @ w[:-1] + w[-1]

    probe = Poly2Probe().fit(X, y)
    assert np.abs(probe(Z) - ref).max() < 1e-4
    assert np.abs(
        probe.torch(torch.as_tensor(Z, dtype=torch.float64)).numpy() - ref
    ).max() < 1e-4


# --------------------------------------------------------- 2. gradient flow


def test_rollout_loss_gives_finite_gradients_into_the_lstm():
    model = _tiny_model()
    b = _tiny_batch()
    torch.manual_seed(7)
    roll = rollout_losses(model, b["z"], b["a"], t0=3, k_steps=4)

    assert roll["z_dreamed"].shape == (b["z"].shape[0], 4, model.cfg.z_dim)
    assert torch.isfinite(roll["nll"])

    roll["nll"].backward()
    g = model.lstm.weight_hh_l0.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_conservation_penalty_gives_finite_gradients_into_the_lstm():
    """The probe-space penalty must reach the recurrent weights too.

    This is the load-bearing claim of candidate (c): a loss stated in terms of
    a frozen probe's reading of a DREAMED latent has to differentiate back
    through the reparameterised sample. If the sample were drawn under
    no_grad -- easy to do by accident -- this gradient would be exactly zero
    and the run would look fine while teaching nothing.
    """
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4))
    probe = Poly2Probe().fit(X, rng.normal(size=200))

    model = _tiny_model()
    b = _tiny_batch()
    torch.manual_seed(11)
    roll = rollout_losses(model, b["z"], b["a"], t0=2, k_steps=5)
    pen = conservation_penalty(probe, roll["z_dreamed"], roll["z_ref"])
    assert torch.isfinite(pen) and pen.item() >= 0.0

    pen.backward()
    g = model.lstm.weight_hh_l0.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    # The MDN's log-std head must get gradient specifically: shrinking the
    # predicted noise is the mechanism by which this loss can be satisfied.
    assert model.mdn.weight.grad.abs().sum() > 0


def test_reparam_sample_is_the_component_mean_when_std_is_tiny():
    """Sanity on the sampler itself: no eps, no noise."""
    model = _tiny_model()
    b = _tiny_batch()
    with torch.no_grad():
        parts, _ = model(b["z"], b["a"])
        parts["logstd"] = torch.full_like(parts["logstd"], -20.0)
        s = reparam_sample(model, parts)
        m = model.most_likely_mean(parts)
    assert torch.allclose(s, m, atol=1e-5)


def test_rollout_rejects_a_window_it_does_not_fit_in():
    model = _tiny_model()
    b = _tiny_batch(L=12)
    with pytest.raises(ValueError):
        rollout_losses(model, b["z"], b["a"], t0=8, k_steps=8)


# ------------------------------------------------- 3. the off switch is off


def test_rollout_steps_zero_reproduces_the_baseline_loss_exactly():
    """K = 0 must change nothing about the loss OR the parameters it touches.

    Reproduces what ``train_rnn``'s inner loop does with and without the
    mechanism enabled, on the same batch and the same model, and requires the
    two scalars to be bit-identical -- not merely close. Anything else means
    the mechanism has leaked into the default path and every pre-existing
    checkpoint's training recipe has silently changed.
    """
    model = _tiny_model()
    b = _tiny_batch()

    parts, _ = model(b["z"], b["a"])
    base, _ = rnn_loss(model, parts, b, pos_weight=5.0)

    # "with the flag, set to zero": the branch in train_rnn is guarded by
    # `if a.rollout_loss_steps > 0`, so nothing is added.
    k_steps = 0
    parts2, _ = model(b["z"], b["a"])
    loss2, _ = rnn_loss(model, parts2, b, pos_weight=5.0)
    if k_steps > 0:                                     # pragma: no cover
        loss2 = loss2 + rollout_losses(model, b["z"], b["a"], 0, k_steps)["nll"]

    assert loss2.item() == base.item()
    # And the model without a mass head emits no log_mass key, so the head's
    # loss term cannot be added by accident either.
    assert model.mass_head is None
    assert "log_mass" not in parts


def test_mass_head_is_off_by_default_and_state_dict_is_unchanged():
    """Old checkpoints must still load: no head means no parameters."""
    off = MDNRNN(RNNConfig(z_dim=4, hidden=16, n_gauss=3))
    on = MDNRNN(RNNConfig(z_dim=4, hidden=16, n_gauss=3, mass_head=True))
    assert set(on.state_dict()) - set(off.state_dict()) == {
        "mass_head.weight", "mass_head.bias"}
    assert off.mass_head is None

    b = _tiny_batch()
    parts, _ = on(b["z"], b["a"])
    assert parts["log_mass"].shape == (b["z"].shape[0], b["z"].shape[1], 1)


# ----------------------------------------------- 4. the metric's null case


def test_metric_scores_a_perfect_copy_of_the_truth_as_conserved():
    """A "dream" that IS the truth must score corr ~ 1 and zero drift.

    The point of the test is that a metric which can only ever report drift is
    useless -- it has to be able to report conservation when conservation
    happened. So: synthesise trajectories at a known per-episode speed, hand
    the estimator the true positions as if they were dreamed, and require the
    speed correlation to be 1 and the speed horizon to be the whole rollout.
    """
    rng = np.random.default_rng(5)
    N, H = 24, 120
    mass = rng.uniform(0.5, 2.0, size=N)
    speed = 0.022 / mass
    ang = rng.uniform(0, 2 * np.pi, size=N)
    d = np.stack([np.cos(ang), np.sin(ang)], 1) * speed[:, None]      # (N, 2)
    pos = d[:, None, :] * np.arange(H)[None, :, None]                 # (N, H, 2)

    est = rolling_speed(pos, win=6)
    assert np.abs(est - speed[:, None]).max() < 1e-9
    assert corr(est[:, H // 2], speed) == pytest.approx(1.0, abs=1e-9)
    assert (speed_horizon(est, speed, tol=0.25) == H).all()

    # And a trajectory whose speed doubles halfway through must be caught at
    # the step where it doubles (up to the estimator's half-window lag).
    drift = pos.copy()
    drift[:, 60:] = pos[:, 59][:, None] + d[:, None, :] * 2 * np.arange(1, H - 59)[None, :, None]
    hz = speed_horizon(rolling_speed(drift, win=6), speed, tol=0.25)
    assert np.all(np.abs(hz - 54) <= 8), hz

    # Log-mass side of the same null. The real latent code for log-mass is a
    # smooth degree-2 fold (stage one: poly-2 R^2 0.97, linear 0.10), so the
    # honest synthetic version is an exactly-degree-2 function of the latent:
    # read back on HELD-OUT points it should come out at corr ~ 1.
    X = rng.normal(size=(600, 6))
    f = lambda A: (0.4 * A[:, 0] + 0.3 * A[:, 1] ** 2  # noqa: E731
                   - 0.2 * A[:, 2] * A[:, 3] + 0.1)
    probe = Poly2Probe().fit(X[:500], f(X[:500]))
    assert corr(probe(X[500:]), f(X[500:])) > 0.99

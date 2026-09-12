"""Conservation: the machinery for measuring, and for teaching, "this stays put".

Stage three found it by watching a GIF: over a 200-step dream at tau = 1 the
ball's colour drifts from yellow through red to purple. Mass is constant within
a v2 episode and mass sets the speed law, so a dream whose colour walks is a
dream whose *physics* walks -- and that is the world the controller was trained
in.

Why it happens, stated precisely. The MDN predicts a distribution over
z_{t+1}; at tau = 1 we draw from it. Whatever direction of the latent carries
colour has a predicted std that is not exactly zero (it cannot be -- the
training targets are posterior SAMPLES, which jitter around the true code by
the VAE's own posterior noise), so every dreamed step adds an independent
increment along it. Position has a restoring force in the data (the walls, the
floor, the bounded box) and colour has none, so the colour increments simply
integrate: a random walk, variance growing linearly in the number of steps.
After ~25 steps the accumulated walk is the size of the whole colour range and
the correlation with the truth is gone. Nothing in the teacher-forced loss can
see this, because teacher forcing hands the model a fresh, correct z at every
step and the walk never gets a chance to accumulate.

This module holds the three pieces that the diagnosis implies:

``Poly2Probe``
    A frozen degree-2 ridge probe from VAE ``mu`` to a scalar factor (here
    ``log(mass)``), with a numpy path for measurement and a **torch** path for
    training. The torch path is what makes a conservation *loss* possible:
    gradients flow from ``g(z_dreamed)`` back into the dreamed latent and
    thence into the LSTM. The two paths share one set of fitted coefficients
    and one feature ordering, and ``tests/test_conservation.py`` pins them
    together to 1e-4.

``rollout_losses``
    A K-step OPEN-LOOP rollout inside training, fed by the model's own
    reparameterised samples, returning (a) the MDN NLL of the TRUE latents
    along it and (b) the dreamed latents themselves. (a) is the fix of the form
    "penalise drift of everything, because the truth does not drift"; (b) is
    what the targeted conservation penalty is computed from.

``rolling_speed`` / ``speed_horizon``
    The dynamical consequence, measured. A drifting colour is only interesting
    because it changes how fast the dreamed ball moves, and per-episode
    position agreement is the wrong yardstick for a sampled dream (a tau = 1
    dream is a different plausible future, not a failed copy of this one).
    Speed is the right yardstick: it is a scalar the law pins exactly, and it
    is well defined in a dream that has diverged in position.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .rnn import MDNRNN

# Default per-step speed estimator: a centred window of +-SPEED_WIN steps, from
# which we take the MEDIAN per-step displacement. Median for the same reason
# eval_causal_v2.dreamed_speed uses it -- a wall bounce inside the window
# shortens one step and the kNN position probe occasionally snaps a dreamed
# latent to a neighbour some distance away, and both are single outliers among
# ~2*WIN otherwise-identical displacements.
SPEED_WIN = 6


# ------------------------------------------------------------------- probe


def _triu_pairs(d: int):
    """Index arrays for the degree-2 cross terms, in sklearn's ordering."""
    return np.triu_indices(d)


class Poly2Probe:
    """Frozen degree-2 ridge probe: standardised ``mu`` -> one scalar factor.

    Degree 2 and not linear, and not kNN, because stage one measured exactly
    which of those works: ``log_mass`` from the v2 VAE's ``mu`` scores R^2 0.97
    under a poly-2 probe, 0.10 linear and 0.08 kNN. The colour code is a smooth
    low-order *fold* in the latent, so a linear probe cannot see it and a
    nearest neighbour lands on a frame at the same place with a different
    colour. Any measurement of "did the dream conserve mass" therefore has to
    use the probe that can read mass in the first place.

    Fit is numpy/closed form (the same ridge solve as ``wm.probes._fit_linear``
    with ``ridge=1.0``); evaluation has two paths:

        ``probe(z)``          numpy, for measurement
        ``probe.torch(z)``    torch and differentiable, for the training loss

    The torch path builds the same features from the same fitted constants, so
    the two agree to floating-point noise. That equality is not a nicety: the
    training penalty and the eval metric are only the same statement about the
    world if the function ``g`` is literally the same function.
    """

    def __init__(self, ridge: float = 1.0):
        self.ridge = float(ridge)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None   # (n_features + 1,), bias last

    # -------------------------------------------------------------- fitting

    def fit(self, mu: np.ndarray, y: np.ndarray) -> "Poly2Probe":
        X = np.asarray(mu, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.mean_ = X.mean(0)
        # Guard a constant column: a zero scale would divide by zero and a
        # constant feature carries nothing anyway.
        self.scale_ = np.where(X.std(0) > 1e-12, X.std(0), 1.0)
        self._iu, self._ju = _triu_pairs(X.shape[1])
        P = self._features_np(X)
        A = np.concatenate([P, np.ones((len(P), 1))], 1)
        G = A.T @ A + self.ridge * np.eye(A.shape[1])
        self.coef_ = np.linalg.solve(G, A.T @ y)
        return self

    # ------------------------------------------------------------ numpy path

    def _features_np(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean_) / self.scale_
        return np.concatenate([Z, Z[:, self._iu] * Z[:, self._ju]], 1)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """(..., z_dim) -> (...,). Shape-preserving, like ``StateProbe``."""
        z = np.asarray(z, dtype=np.float64)
        shape = z.shape[:-1]
        flat = z.reshape(-1, z.shape[-1])
        P = self._features_np(flat)
        out = P @ self.coef_[:-1] + self.coef_[-1]
        return out.reshape(shape)

    # ------------------------------------------------------------ torch path

    def torch_constants(self, device="cpu", dtype=torch.float32):
        """Cache the fitted constants as tensors on ``device``."""
        key = (str(device), dtype)
        cache = getattr(self, "_tcache", {})
        if key not in cache:
            t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
            cache[key] = (
                t(self.mean_), t(self.scale_),
                torch.as_tensor(self._iu, dtype=torch.long, device=device),
                torch.as_tensor(self._ju, dtype=torch.long, device=device),
                t(self.coef_[:-1]), t(np.asarray(self.coef_[-1])),
            )
            self._tcache = cache
        return cache[key]

    def torch(self, z: torch.Tensor) -> torch.Tensor:
        """Differentiable evaluation. ``(..., z_dim) -> (...,)``.

        Everything here is a differentiable function of ``z``: a shift, a
        scale, an elementwise product and a matrix-vector product. So
        ``g(z).backward()`` puts a gradient on ``z``, which is the whole point
        -- it is how a loss stated in *probe* space ("the decoded log-mass must
        not change") reaches the parameters that produced the dreamed latent.
        """
        mean, scale, iu, ju, w, b = self.torch_constants(z.device, z.dtype)
        zz = (z - mean) / scale
        feats = torch.cat([zz, zz[..., iu] * zz[..., ju]], dim=-1)
        return feats @ w + b


def fit_log_mass_probe(
    mu: np.ndarray, mass: np.ndarray, ridge: float = 1.0
) -> Poly2Probe:
    """Convenience: fit ``mu -> log(mass)``. ``mu`` (N, z), ``mass`` (N,)."""
    return Poly2Probe(ridge=ridge).fit(mu, np.log(np.asarray(mass, float)))


# ------------------------------------------------------- open-loop rollout


def reparam_sample(
    model: MDNRNN,
    parts: Dict[str, torch.Tensor],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """A tau=1 draw from the MDN that gradients can flow through. (B, T, z).

    Two decisions, both deliberate.

    **The component choice is the most-likely one**, not a categorical draw.
    An argmax is not differentiable and a Gumbel-softmax relaxation would mix
    the components' means, which is precisely the averaging-across-modes
    failure the mixture exists to avoid (``wm.rnn``'s opening docstring). Using
    ``argmax`` keeps the rollout on a single mode and keeps the gradient exact
    for the parameters of that mode; the price is that the *choice* itself gets
    no gradient signal, which is acceptable because it is the within-component
    noise, not the component choice, that drives the colour random walk this
    module exists to stop.

    **The within-component draw is reparameterised**: ``mean + std * eps`` with
    ``eps`` a fixed sample. So the noise that a tau = 1 dream actually injects
    is present in the graph, and a loss placed downstream of it can push the
    predicted ``std`` in the colour direction toward zero. That is the
    mechanism by which any of this can work at all: sampling with
    ``torch.no_grad`` or with a detached std would inject the same noise and
    teach the model nothing about it.
    """
    k = parts["logits"].argmax(-1)                                  # (B, T)
    mean = model._abs_mean(parts)                                   # (B,T,K,z)
    std = torch.exp(parts["logstd"])
    idx = k[..., None, None].expand(-1, -1, 1, mean.shape[-1])
    m = mean.gather(2, idx).squeeze(2)
    s = std.gather(2, idx).squeeze(2)
    eps = torch.randn(m.shape, device=m.device, dtype=m.dtype, generator=generator)
    return m + eps * s


def rollout_losses(
    model: MDNRNN,
    z: torch.Tensor,             # (B, L, z) window of TRUE latents
    a_onehot: torch.Tensor,      # (B, L, n_actions)
    t0: int,
    k_steps: int,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """K steps of open-loop dreaming from window position ``t0``, with grads.

    Why teacher forcing alone cannot teach conservation
    ---------------------------------------------------
    Under teacher forcing the model is handed the true ``z_t`` at every step.
    Its colour is therefore always exactly right when it makes its next
    prediction, and the *only* thing the NLL asks of it is that the one-step
    predictive distribution be well calibrated. A predicted std of 0.05 in the
    colour direction is not merely tolerated by that loss, it is *required* by
    it: the training targets are posterior samples, which really do jitter by
    about that much, so a model that predicted zero colour variance would take
    a large NLL penalty. Teacher forcing then throws the sample away and hands
    back the truth, so the model never experiences the consequence -- eight
    steps of its own 0.05 jitter compounding into a ball of a different
    colour. The gradient signal for "this quantity must not drift" simply is
    not in the teacher-forced loss.

    Why this can
    ------------
    Here the model eats its own samples for K steps and is then asked for the
    likelihood of the TRUE latent at each of those steps. The truth has not
    drifted -- the real ball is still the same colour -- so any accumulated
    walk shows up directly as a low likelihood, and the gradient flows back
    through the reparameterised samples to the std that produced them. The
    model can lower that loss in exactly one way that generalises: shrink the
    predicted noise along directions that should not move, while leaving it
    where real uncertainty lives (which mode the bounce takes). It is
    scheduled-sampling / professor-forcing logic, specialised to the quantity
    we care about.

    Returns ``{"nll": scalar, "z_dreamed": (B, K, z), "z_ref": (B, z)}`` where
    ``z_ref`` is the true latent at ``t0`` -- the anchor a conservation penalty
    compares against.

    Indexing. After the teacher-forced pass over ``[0 .. t0]`` the LSTM state
    has consumed ``(z_t0, a_t0)``, and the output at that position predicts
    ``z_{t0+1}``. So step ``k = 1`` is still effectively teacher-forced (its
    input was the true ``z_t0``) and only ``k >= 2`` are genuinely open-loop;
    ``k = 1`` is kept because it costs nothing, it is the literal reading of
    "the NLL of the true z_{t0+k} at each of the K steps", and its presence
    simply puts a little extra weight on the one-step term.

    The teacher-forced prefix is recomputed here rather than reused from the
    caller's full-window pass, because ``nn.LSTM`` returns only the FINAL
    (h, c) and we need the pair at ``t0``. The extra cost is one LSTM pass over
    ``t0`` steps, i.e. about half a window on average.
    """
    B, L, _ = z.shape
    if not (0 <= t0 and t0 + k_steps <= L - 1):
        raise ValueError(
            f"t0={t0}, k_steps={k_steps} does not fit a window of length {L}; "
            f"need t0 + k_steps <= L - 1"
        )

    parts, h = model(z[:, : t0 + 1], a_onehot[:, : t0 + 1])
    # Keep only the last position's prediction; every entry of ``parts`` is
    # (B, T, ...) so one slice does all of them, ``z_in`` included, which keeps
    # the residual bookkeeping in ``mdn_nll`` correct.
    p = {k: v[:, -1:] for k, v in parts.items()}

    nlls: List[torch.Tensor] = []
    dreamed: List[torch.Tensor] = []
    for k in range(1, k_steps + 1):
        nlls.append(model.mdn_nll(p, z[:, t0 + k : t0 + k + 1]))
        z_s = reparam_sample(model, p, generator=generator)          # (B, 1, z)
        dreamed.append(z_s[:, 0])
        if k < k_steps:
            p, h = model(z_s, a_onehot[:, t0 + k : t0 + k + 1], h)

    return {
        "nll": torch.stack(nlls).mean(),      # mean over k, so w is interpretable
        "nll_per_step": torch.stack(nlls).detach(),
        "z_dreamed": torch.stack(dreamed, 1),                        # (B, K, z)
        "z_ref": z[:, t0],                                           # (B, z)
    }


def conservation_penalty(
    probe: Poly2Probe, z_dreamed: torch.Tensor, z_ref: torch.Tensor
) -> torch.Tensor:
    """``mean_k ( g(z_dreamed_k) - g(z_ref) )^2`` for a frozen probe ``g``.

    The targeted version of the fix. The multi-step NLL says "do not drift,
    anything"; this says "do not drift *along this one direction*", naming the
    quantity with a function we already trust to read it (the same poly-2 probe
    the eval metric uses). The anchor ``g(z_ref)`` is detached: it is the true
    latent at the start of the rollout, so it is a constant target and not
    something the model gets to move toward the dream.

    A caveat worth stating: this hands the model privileged information about
    which latent function matters, so it is a weaker scientific claim than the
    multi-step NLL, which discovers the constant on its own. It is included
    because it is the sharpest possible test of the *diagnosis* -- if naming
    the quantity does not fix the drift, the diagnosis is wrong.
    """
    g_d = probe.torch(z_dreamed)                                     # (B, K)
    g_r = probe.torch(z_ref).detach().unsqueeze(1)                   # (B, 1)
    return ((g_d - g_r) ** 2).mean()


# ------------------------------------------------------------ speed metrics


def rolling_speed(pos: np.ndarray, win: int = SPEED_WIN) -> np.ndarray:
    """Per-step speed of a trajectory, as a function of step. (N, H) -> (N, H).

    ``pos`` is (N, H, 2). At each step the estimate is the MEDIAN per-step
    displacement over a centred window of +-``win`` steps, so it is a local
    version of ``eval_causal_v2.dreamed_speed`` that can be plotted against
    dream step instead of collapsing the whole rollout to one number. Windows
    are clipped at the ends rather than padded, so the first and last few
    estimates are noisier; they are still honest.
    """
    step = np.linalg.norm(np.diff(pos, axis=1), axis=-1)              # (N, H-1)
    H = pos.shape[1]
    out = np.empty((pos.shape[0], H), dtype=float)
    for t in range(H):
        lo, hi = max(0, t - win), min(step.shape[1], t + win)
        seg = step[:, lo:hi] if hi > lo else step[:, :1]
        out[:, t] = np.median(seg, axis=1)
    return out


def speed_horizon(
    speed: np.ndarray, true_speed: np.ndarray, tol: float = 0.25
) -> np.ndarray:
    """First step at which the estimated speed is off the law by > ``tol``.

    ``speed`` is (N, H), ``true_speed`` is (N,). Returns (N,) of first-crossing
    indices, H if it never crosses.

    Why this and not position error, for tau > 0: a sampled dream is a
    DIFFERENT plausible future, so its ball is not supposed to be where the
    real ball is, and the usual "useful horizon" (first step the dreamed ball
    stops overlapping the true one) measures divergence that is partly correct
    behaviour. Speed is the part the law does pin down regardless of which
    future you are in, so a speed horizon at tau = 1 is a statement about the
    model and not about the sampling.
    """
    H = speed.shape[1]
    rel = np.abs(speed / np.asarray(true_speed, float)[:, None] - 1.0)
    bad = rel > tol
    out = np.full(speed.shape[0], float(H))
    for i in range(speed.shape[0]):
        w = np.where(bad[i])[0]
        if len(w):
            out[i] = float(w[0])
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, NaN-safe and degenerate-safe."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

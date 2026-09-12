"""The controller: stage three ("C") of the V-M-C world model.

C is the smallest piece of the whole system by a factor of about a thousand.
V has ~1.5M parameters, M has ~600k, and C -- the only part that actually
*decides* anything -- has 3 x (16 + 256) + 3 = 819. That ratio is the thesis of
Ha & Schmidhuber (2018): if V compresses the pixels and M compresses time, then
the policy left over is close to linear, and a linear policy is small enough to
train with a derivative-free optimiser (CMA-ES) directly on episode return. No
backprop through time, no credit assignment, no value function, no replay
buffer. You just try 32 weight vectors, keep the good ones, and repeat.

Why a linear map is enough HERE
-------------------------------
The task is "put the paddle where the ball is about to land". In world
coordinates that is roughly ``sign(ball_x + ball_vx * t_to_floor - paddle_x)``,
which is linear in a state that contains position and velocity. It is *not*
linear in raw pixels, and it is *not* linear in z alone -- a single frame has no
velocity in it (stage one measured velocity R^2 ~ 0 from mu, stage two measured
R^2 ~ 0.9 from h). So the representation is doing the work, and the controller
is only reading it off. That is exactly the claim this file lets us test, by
flipping ``--inputs`` between ``z``, ``h`` and ``zh``.

The parameter-vector interface
------------------------------
CMA-ES optimises a flat real vector. Everything here therefore round-trips
through ``get_params() / set_params()`` and the actual arithmetic happens in
numpy, not torch: there are no gradients anywhere in stage three, and a
(P, 3, 272) einsum over a population of 32 candidates is microseconds. Keeping C
in numpy also means one single code path is shared by the dream and the real
environment -- see ``make_features``, which both call -- so the two cannot drift
apart silently. That mattered enough to be a unit test.

Input normalisation
-------------------
h is an LSTM output, so its entries are bounded in (-1, 1) but with wildly
different per-unit means and scales (some units saturate, some barely move).
z has unit-ish scale by construction but its active dims are far from
standardised. CMA-ES starts from an isotropic Gaussian with a single sigma0, so
if one input has std 0.02 and another has std 0.9, the search has to discover
that anisotropy itself, spending generations on what a one-line whitening does
for free. We therefore standardise both blocks with statistics computed ONCE
from the training set (``compute_norm_stats``) and freeze them into the
checkpoint. This is not strictly necessary -- CMA-ES adapts a full covariance
and would get there -- but it is free and it makes short runs work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

N_ACTIONS = 3
INPUT_MODES = ("z", "h", "zh")


# ------------------------------------------------------------- normalisation


@dataclass
class NormStats:
    """Per-dimension mean/std for the two input blocks. Frozen at train time."""

    z_mean: np.ndarray  # (z_dim,)
    z_std: np.ndarray   # (z_dim,)
    h_mean: np.ndarray  # (hidden,)
    h_std: np.ndarray   # (hidden,)

    @staticmethod
    def identity(z_dim: int, hidden: int) -> "NormStats":
        return NormStats(
            np.zeros(z_dim, np.float32), np.ones(z_dim, np.float32),
            np.zeros(hidden, np.float32), np.ones(hidden, np.float32),
        )

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v, np.float32) for k, v in asdict(self).items()}

    @staticmethod
    def from_dict(d: Dict[str, np.ndarray]) -> "NormStats":
        return NormStats(**{k: np.asarray(v, np.float32) for k, v in d.items()})


# The floor on std is not cosmetic. Several of the 256 LSTM units are
# effectively dead (std ~ 1e-4 over the whole training set); dividing by that
# turns numerical noise into an input with std 1, and CMA-ES will happily fit
# weights to pure noise. Clamping leaves dead units near zero, where they
# belong.
_STD_FLOOR = 1e-3


def make_features(
    z: np.ndarray,          # (..., z_dim)
    h: np.ndarray,          # (..., hidden)
    inputs: str,
    norm: Optional[NormStats] = None,
) -> np.ndarray:
    """Build the controller's input vector. THE single definition of "obs".

    Both the dream loop and the real-environment loop call exactly this, which
    is the only reason we can claim the two agree; ``tests.test_controller``
    asserts it numerically.
    """
    if inputs not in INPUT_MODES:
        raise ValueError(f"inputs must be one of {INPUT_MODES}, got {inputs!r}")
    z = np.asarray(z, np.float32)
    h = np.asarray(h, np.float32)
    if norm is not None:
        z = (z - norm.z_mean) / np.maximum(norm.z_std, _STD_FLOOR)
        h = (h - norm.h_mean) / np.maximum(norm.h_std, _STD_FLOOR)
    if inputs == "z":
        return z
    if inputs == "h":
        return h
    return np.concatenate([z, h], axis=-1)


def feature_dim(z_dim: int, hidden: int, inputs: str) -> int:
    return {"z": z_dim, "h": hidden, "zh": z_dim + hidden}[inputs]


# ------------------------------------------------------- population-level ops


def batched_logits(params: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """Evaluate P different controllers on R observations each, in one einsum.

    ``params`` is ``(P, 3 * in_dim + 3)`` -- one flat CMA-ES vector per row --
    and ``feats`` is ``(P, R, in_dim)``. Returns ``(P, R, 3)``.

    The whole population shares one forward pass through M (the RNN sees a
    batch of P*R dreams), so the controller must be evaluated the same way or it
    becomes the bottleneck. Rolling out 32 candidates x 16 dreams x 150 steps
    one-at-a-time is 76,800 sequential LSTM calls; batched it is 150.
    """
    P, R, in_dim = feats.shape
    assert params.shape == (P, N_ACTIONS * in_dim + N_ACTIONS), params.shape
    W = params[:, : N_ACTIONS * in_dim].reshape(P, N_ACTIONS, in_dim)
    b = params[:, N_ACTIONS * in_dim :]                      # (P, 3)
    return np.einsum("pai,pri->pra", W, feats) + b[:, None, :]


def logits_to_actions(
    logits: np.ndarray,
    temperature: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """argmax (temperature 0, the default) or a softmax draw.

    Deterministic argmax is the right default for *evaluation*: we want to
    measure the policy CMA-ES actually optimised, and CMA-ES optimised the
    argmax policy. The softmax path exists only to check that the controller is
    not sitting on a razor-thin decision boundary where a 1e-6 logit change
    flips the action.
    """
    if temperature <= 0.0:
        return logits.argmax(-1).astype(np.int64)
    rng = rng or np.random.default_rng()
    x = logits / temperature
    x = x - x.max(-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(-1, keepdims=True)
    # Inverse-CDF sampling, vectorised over every leading axis.
    u = rng.random(p.shape[:-1] + (1,))
    return (u > np.cumsum(p, -1)).sum(-1).astype(np.int64)


# ------------------------------------------------------------- the controller


class BaseController:
    """Common interface so the eval harness can treat every policy alike.

    ``act`` takes the world-model observation ``(z, h)`` plus, optionally, the
    TRUE state -- which only the oracle is allowed to look at, and which is
    passed to everything else purely so the call site has no special cases.
    """

    name: str = "base"
    uses_true_state: bool = False

    def reset(self, batch: int, seed: Optional[int] = None) -> None:  # noqa: D102
        pass

    def act(
        self,
        z: np.ndarray,
        h: np.ndarray,
        state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError


class LinearController(BaseController):
    """``a = argmax(W @ [z, h] + b)``. 819 parameters at the default sizes."""

    def __init__(
        self,
        z_dim: int = 16,
        hidden: int = 256,
        inputs: str = "zh",
        norm: Optional[NormStats] = None,
        name: str = "linear",
    ):
        if inputs not in INPUT_MODES:
            raise ValueError(f"inputs must be one of {INPUT_MODES}, got {inputs!r}")
        self.z_dim, self.hidden, self.inputs = int(z_dim), int(hidden), str(inputs)
        self.in_dim = feature_dim(self.z_dim, self.hidden, self.inputs)
        self.norm = norm
        self.name = name
        # Zero init means "all logits equal" -> argmax picks action 0 (LEFT)
        # everywhere. That is a deliberate, boring starting point: CMA-ES
        # explores from x0 anyway, and a random init would make generation 0 of
        # different seeds incomparable.
        self.W = np.zeros((N_ACTIONS, self.in_dim), np.float64)
        self.b = np.zeros(N_ACTIONS, np.float64)
        self.temperature = 0.0

    # ------------------------------------------------------------- params

    @property
    def n_params(self) -> int:
        return N_ACTIONS * self.in_dim + N_ACTIONS

    def get_params(self) -> np.ndarray:
        """Flat vector, layout ``[W.ravel(), b]``. What CMA-ES searches over."""
        return np.concatenate([self.W.ravel(), self.b])

    def set_params(self, vec: np.ndarray) -> "LinearController":
        vec = np.asarray(vec, np.float64).ravel()
        if vec.size != self.n_params:
            raise ValueError(f"expected {self.n_params} params, got {vec.size}")
        self.W = vec[: N_ACTIONS * self.in_dim].reshape(N_ACTIONS, self.in_dim).copy()
        self.b = vec[N_ACTIONS * self.in_dim :].copy()
        return self

    # --------------------------------------------------------------- act

    def features(self, z: np.ndarray, h: np.ndarray) -> np.ndarray:
        return make_features(z, h, self.inputs, self.norm)

    def logits(self, z: np.ndarray, h: np.ndarray) -> np.ndarray:
        return self.features(z, h) @ self.W.T + self.b

    def act(self, z, h, state=None) -> np.ndarray:
        # Routed through the same two helpers the population path uses, so a
        # single controller and a population of one give bit-identical actions.
        f = self.features(z, h)[None]                        # (1, B, in_dim)
        lg = batched_logits(self.get_params()[None], f)[0]   # (B, 3)
        return logits_to_actions(lg, self.temperature, self._rng)

    _rng: Optional[np.random.Generator] = None

    def reset(self, batch: int, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)


class RandomController(BaseController):
    """Sticky random actions -- the same behaviour policy that made the data.

    Sticky rather than i.i.d. uniform because i.i.d. uniform is a *weaker*
    baseline than it looks: the paddle random-walks and barely leaves its start,
    so it is close to "stay still" plus jitter. Sticky actions sweep the box and
    therefore blunder into the ball more often. Beating the harder baseline is
    the claim worth making.
    """

    name = "random"

    def __init__(self, mean_hold: float = 8.0, seed: int = 0):
        self.mean_hold = float(mean_hold)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._left = np.zeros(0, np.int64)
        self._cur = np.zeros(0, np.int64)

    def reset(self, batch: int, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(self.seed if seed is None else seed)
        self._left = np.zeros(batch, np.int64)
        self._cur = np.full(batch, 1, np.int64)

    def act(self, z, h, state=None) -> np.ndarray:
        n = len(np.atleast_2d(z))
        if len(self._left) != n:
            self.reset(n)
        new = self._left <= 0
        if new.any():
            k = int(new.sum())
            self._cur[new] = self._rng.integers(0, N_ACTIONS, size=k)
            self._left[new] = self._rng.geometric(1.0 / self.mean_hold, size=k)
        self._left -= 1
        return self._cur.copy()


class StayController(BaseController):
    """Always STAY. The do-nothing floor: any hits it gets are pure luck."""

    name = "stay"

    def act(self, z, h, state=None) -> np.ndarray:
        return np.full(len(np.atleast_2d(z)), 1, np.int64)


class OracleController(BaseController):
    """``worldsim.policies.tracking_action`` on the TRUE state. The ceiling.

    It cheats twice over: it reads privileged state that never passes through
    the VAE, and that state is exact rather than inferred. It is here to convert
    "0.9 hits per episode" into a fraction of what is physically achievable --
    without it, the absolute numbers mean nothing, because the number of
    opportunities per episode is a property of the physics (~2 floor visits per
    200 steps), not of the policy.
    """

    name = "oracle"
    uses_true_state = True

    def __init__(self, paddle_w: float = 0.26, dead_zone_frac: float = 0.25):
        self.paddle_w = float(paddle_w)
        self.dead_zone_frac = float(dead_zone_frac)

    def act(self, z, h, state=None) -> np.ndarray:
        from worldsim.policies import tracking_action

        if state is None:
            raise ValueError("OracleController needs the true state")
        state = np.atleast_2d(np.asarray(state))
        return np.array(
            [
                tracking_action(s, self.paddle_w, self.dead_zone_frac)
                for s in state
            ],
            dtype=np.int64,
        )


# The six columns every version of the world has. Anything past them is a v2 or
# v3 diagnostic, and in an occluded world the last one is ``ball_visible``.
_BASE_STATE_NAMES = (
    "ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x", "paddle_vx",
)
STAY_ACTION = 1


class WaitAndSeeOracleController(BaseController):
    """v3: the oracle, blinded whenever the ball is behind the band.

    The MEMORYLESS UPPER BOUND, and the single most informative reference in
    stage three of v3. It is handed the same privileged true state as
    ``OracleController`` -- exact position, no encoder, no inference -- but it is
    only allowed to look at it while ``ball_visible > threshold``; on every other
    frame it STAYs. So it has perfect vision and zero memory, which is exactly
    the policy class a controller with no object permanence is confined to, at
    its theoretical best.

    Read it against ``OracleController`` (perfect vision AND perfect memory):
    the gap between the two is the *entire* value of object permanence for this
    task, measured behaviourally and with no world model in the loop at all. If
    that gap is small, no controller -- fair, privileged or otherwise -- can gain
    much from permanence here, and the whole C-stage question is answered before
    a single trained policy is scored.

    ``ball_visible`` is the LAST state column when the environment has an
    occluder (``[..., paddle_vx, (mass), ball_visible]``), so a state vector
    with no extra column means "never occluded" and this degrades exactly into
    the plain oracle.
    """

    name = "wait_and_see"
    uses_true_state = True

    def __init__(
        self,
        paddle_w: float = 0.26,
        dead_zone_frac: float = 0.25,
        threshold: float = 0.5,
        name: str = "wait_and_see",
    ):
        self.paddle_w = float(paddle_w)
        self.dead_zone_frac = float(dead_zone_frac)
        self.threshold = float(threshold)
        self.name = name

    def act(self, z, h, state=None) -> np.ndarray:
        from worldsim.policies import tracking_action

        if state is None:
            raise ValueError("WaitAndSeeOracleController needs the true state")
        state = np.atleast_2d(np.asarray(state))
        visible = (
            state[:, -1] if state.shape[-1] > len(_BASE_STATE_NAMES)
            else np.ones(len(state))
        )
        return np.array(
            [
                tracking_action(s, self.paddle_w, self.dead_zone_frac)
                if v > self.threshold else STAY_ACTION
                for s, v in zip(state, visible)
            ],
            dtype=np.int64,
        )


# --------------------------------------------------- normalisation statistics


def compute_norm_stats(
    rnn,
    roots: Sequence[str],
    device: str = "cpu",
    n_episodes: int = 96,
    seed: int = 0,
) -> Tuple[NormStats, Dict[str, float]]:
    """Mean/std of z and of teacher-forced h, over a sample of training episodes.

    "Teacher-forced" matters: h is collected by feeding the RNN the TRUE latent
    sequence with the TRUE actions, i.e. the distribution of hidden states the
    model visits when it is right. A controller trained in the dream will visit
    a slightly different h distribution (the dream drifts), and in the real
    environment it visits yet another one (real z, its own actions). We
    standardise with the training-set statistics because they are the only ones
    available before any controller exists -- and because all three
    distributions have to stay comparable, which they would not if each code
    path whitened with its own running statistics.

    We use ``mu`` rather than posterior samples, matching what the controller is
    fed at test time (see ``eval_controller``).
    """
    import torch

    from .seq_data import episode_arrays

    rng = np.random.default_rng(seed)
    zs, hs = [], []
    for root in roots:
        d = episode_arrays(root)
        mu, actions = d["mu"], d["actions"]
        E, T = actions.shape
        take = min(n_episodes, E)
        idx = rng.choice(E, size=take, replace=False)
        z = torch.from_numpy(mu[idx, :T].astype(np.float32)).to(device)
        a = torch.eye(N_ACTIONS, device=device)[
            torch.from_numpy(actions[idx]).long().to(device)
        ]
        with torch.no_grad():
            parts, _ = rnn(z, a)
        zs.append(mu[idx, :T].reshape(-1, mu.shape[-1]))
        hs.append(parts["h"].reshape(-1, parts["h"].shape[-1]).cpu().numpy())

    Z = np.concatenate(zs, 0).astype(np.float32)
    H = np.concatenate(hs, 0).astype(np.float32)
    stats = NormStats(
        Z.mean(0), Z.std(0) + 1e-6, H.mean(0), H.std(0) + 1e-6
    )
    info = {
        "n_samples": int(len(Z)),
        "h_std_min": float(H.std(0).min()),
        "h_std_max": float(H.std(0).max()),
        "h_dead_units": int((H.std(0) < _STD_FLOOR).sum()),
        "z_std_min": float(Z.std(0).min()),
        "z_std_max": float(Z.std(0).max()),
    }
    return stats, info


# ------------------------------------------------------------------ ckpt i/o


def save_controller(
    path: str | Path,
    ctrl: LinearController,
    params: Optional[np.ndarray] = None,
    args: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Pickle the params + normalisation + config. No torch module involved.

    C has no ``state_dict``; it is a 819-vector. We still use ``torch.save`` for
    consistency with the V and M checkpoints (one loader habit, not three).
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": np.asarray(
            ctrl.get_params() if params is None else params, np.float64
        ),
        "cfg": {
            "z_dim": ctrl.z_dim,
            "hidden": ctrl.hidden,
            "inputs": ctrl.inputs,
            "in_dim": ctrl.in_dim,
        },
        "norm": None if ctrl.norm is None else ctrl.norm.to_dict(),
        "args": args or {},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_controller(
    path: str | Path,
    name: Optional[str] = None,
    which: str = "params",
) -> LinearController:
    """Rebuild a ``LinearController`` from ``save_controller``'s payload.

    ``which`` selects among the parameter vectors a training run stores:
    ``params`` (the default / headline set), ``params_best_real`` (the candidate
    with the best periodically-measured REAL score) or ``params_last_dream``
    (CMA-ES's final mean, i.e. the best the DREAM thinks it has). Comparing the
    last two is itself the transfer diagnostic.
    """
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if which not in ck:
        raise KeyError(f"{path} has no '{which}' (keys: {sorted(ck)})")
    norm = None if ck["norm"] is None else NormStats.from_dict(ck["norm"])
    ctrl = LinearController(
        z_dim=cfg["z_dim"],
        hidden=cfg["hidden"],
        inputs=cfg["inputs"],
        norm=norm,
        name=name or Path(path).parent.name,
    )
    ctrl.set_params(ck[which])
    return ctrl


def load_controller_payload(path: str | Path) -> dict:
    """The raw checkpoint dict (params_best_real / params_last / history / ...)."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)

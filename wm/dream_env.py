"""The MDN-RNN wrapped as a batched, gym-like environment: the *dream*.

This is the one idea that makes stage three interesting. Once M can predict
``z_{t+1}`` from ``(z_t, a_t)`` and can also predict the reward and the paddle
contact, it *is* an environment -- one that needs no renderer, no physics, and
no pixels, and that runs 512 parallel episodes on a laptop CPU faster than the
real simulator runs one. The controller is then trained entirely inside it and
only ever meets the real world at evaluation time.

Three design decisions that are easy to get wrong, in order of how much damage
they do.

1. THE TIMING CONVENTION
------------------------
M was trained as ``h_{t+1} = LSTM(h_t, [z_t, a_t])``, and the heads read off
``h_{t+1}`` to predict ``z_{t+1}``, the reward, and whether a contact happens
during ``t -> t+1``. So the hidden state that "knows about" z_t is the one
produced *after* consuming z_t -- which is not available when a_t must be
chosen, because producing it requires a_t.

The controller therefore acts on

    obs_t = [ z_t , h_t ]        where   h_t = LSTM_out after (z_{t-1}, a_{t-1})

i.e. the CURRENT latent (fresh information: where things are now) together with
the hidden state carried in from the previous step (accumulated information:
velocity, which z cannot contain). Then ``(z_t, a_t)`` is fed to M, which
returns z_{t+1} and h_{t+1}, and the cycle repeats. Nothing from the future
leaks in, and nothing about the trained model is bent to make it work. We call
this hidden state ``h_pre`` throughout -- "the hidden state before this step's
input is consumed".

The real-environment runner in ``wm.eval_controller`` implements the identical
recurrence with ``z_t = encode(real frame_t).mu``. If those two ever disagreed,
a controller trained in the dream would be reading a differently-shifted signal
at test time and the transfer number would be meaningless.
``tests/test_controller.py::test_dream_and_real_timing_agree`` pins it down.

2. WHERE A DREAM STARTS
-----------------------
Not from ``h = 0``. A zero hidden state is a claim that nothing has happened
yet, and since velocity lives only in h (stage two: R^2 ~ 0.9 from h, ~0 from
z), a cold start means the controller's first several decisions are made by a
model that does not know which way the ball is moving. Worse, the dream itself
is wrong during that period -- M cannot extrapolate a trajectory it has not
seen moving -- so early dream reward is noise.

Instead every dream begins at a real ``(episode, t0)`` drawn from the training
sets, with M warm-started by teacher-forcing the ``warmup`` (default 8) TRUE
latents and TRUE actions immediately before ``t0``. Eight steps is comfortably
enough: velocity is a two-frame quantity and the probe saturates within a few.

3. THE REWARD IS PREDICTED, NOT MEASURED
----------------------------------------
There is no simulator in here, so "reward" means *M's reward head* and *M's hit
head*, evaluated on M's own dreamed latents. That is the whole gamble of the
method: if the heads are wrong in some exploitable direction, CMA-ES will find
that direction, because finding exploitable directions in a fixed scalar
objective is precisely what it does. This is why training logs a real-env score
alongside the dream score, and why ``--temperature`` exists (Ha & Schmidhuber
raise tau so the dream is too uncertain to exploit).

    hit    sum_t sigmoid(hit_logit_t)       sparse, ~1.5 for a good policy
    dense  sum_t reward_head_t              dense,  ~100-130 over 150 steps
    mix    hit + lambda * dense             default, lambda = 0.1

Note the arithmetic: at lambda = 0.1 over 150 steps the dense term is ~10x the
hit term, so ``mix`` is dominated by dense. That is intentional rather than a
misconfiguration -- dense (``1 - |ball_x - paddle_x|``) is the only term with a
usable gradient of information at every step, and "maximise it" literally means
"keep the paddle under the ball", which is the task. The hit term then breaks
ties between policies that track equally well but intercept differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .rnn import MDNRNN
from .seq_data import episode_arrays

N_ACTIONS = 3
REWARD_MODES = ("hit", "dense", "mix")


# ------------------------------------------------------------- start states


@dataclass
class StartPool:
    """Real (latent, action) trajectories to warm-start dreams from."""

    mu: np.ndarray        # (E, T+1, z)
    actions: np.ndarray   # (E, T)   int64
    state: np.ndarray     # (E, T+1, 6)  diagnostics only -- never an input
    warmup: int

    @property
    def z_dim(self) -> int:
        return int(self.mu.shape[-1])

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` (episode, t0) pairs with a full warm-up window behind them."""
        E, T = self.actions.shape
        e = rng.integers(0, E, size=n)
        # t0 must leave `warmup` steps of history behind it and at least one
        # transition ahead of it.
        t0 = rng.integers(self.warmup, T, size=n)
        return e, t0


def load_start_pool(
    roots: Sequence[str], warmup: int = 8, max_episodes: Optional[int] = None
) -> StartPool:
    """Pool several dataset roots into one bank of possible dream starts.

    Both ``train`` (sticky-random) and ``train_mix`` (half ball-tracking) are
    used on purpose. train_mix alone would start most dreams with the paddle
    already under the ball -- flattering initial conditions that make every
    candidate look good and give CMA-ES nothing to rank on.
    """
    mus, acts, sts = [], [], []
    for r in roots:
        d = episode_arrays(r)
        mus.append(d["mu"])
        acts.append(d["actions"])
        sts.append(d["state"])
    T = min(a.shape[1] for a in acts)
    mu = np.concatenate([m[:, : T + 1] for m in mus], 0).astype(np.float32)
    a = np.concatenate([x[:, :T] for x in acts], 0).astype(np.int64)
    s = np.concatenate([x[:, : T + 1] for x in sts], 0).astype(np.float32)
    if max_episodes is not None:
        mu, a, s = mu[:max_episodes], a[:max_episodes], s[:max_episodes]
    return StartPool(mu, a, s, int(warmup))


# ----------------------------------------------------------------- the env


class DreamEnv:
    """A batch of ``B`` independent dreams rolled out in lockstep.

    Usage mirrors gym, minus the parts that make no sense here (there is no
    ``done``: the dream never terminates, because the underlying task never
    terminates -- the ball bounces off the floor rather than dying)::

        env = DreamEnv(rnn, pool, temperature=1.0)
        z, h = env.reset(batch=512, seed=0)
        for t in range(150):
            a = controller(z, h)
            z, h, r, p_hit = env.step(a)
    """

    def __init__(
        self,
        rnn: MDNRNN,
        pool: StartPool,
        temperature: float = 1.0,
        reward: str = "mix",
        lam: float = 0.1,
        device: str = "cpu",
    ):
        if reward not in REWARD_MODES:
            raise ValueError(f"reward must be one of {REWARD_MODES}, got {reward!r}")
        self.rnn = rnn.eval()
        self.pool = pool
        self.temperature = float(temperature)
        self.reward_mode = reward
        self.lam = float(lam)
        self.device = device
        self.z_dim = rnn.cfg.z_dim
        self.hidden = rnn.cfg.hidden
        self._eye = torch.eye(N_ACTIONS, device=device)

        self._z: Optional[torch.Tensor] = None
        self._h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._gen: Optional[torch.Generator] = None
        self.t = 0

    # ------------------------------------------------------------- reset

    @torch.no_grad()
    def reset(
        self,
        batch: int = 64,
        seed: int = 0,
        starts: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warm-start ``batch`` dreams. Returns ``(z_t0, h_pre_t0)`` as numpy.

        ``starts`` lets the caller pin the (episode, t0) pairs. CMA-ES uses that
        to give every candidate in a generation the *same* initial conditions:
        the fitness comparison is then a paired one, which removes most of the
        variance from a 16-rollout estimate. Without it, a candidate can win a
        generation purely by having been handed easier starts.
        """
        rng = np.random.default_rng(seed)
        e, t0 = self.pool.sample(batch, rng) if starts is None else starts
        e = np.asarray(e)
        t0 = np.asarray(t0)
        W = self.pool.warmup

        # Gather the warm-up windows [t0 - W, t0): true latents, true actions.
        idx = t0[:, None] - W + np.arange(W)[None, :]             # (B, W)
        z_w = np.take_along_axis(self.pool.mu[e], idx[:, :, None], axis=1)
        a_w = np.take_along_axis(self.pool.actions[e], idx, axis=1)

        z_t = torch.from_numpy(z_w).to(self.device)
        a_t = self._eye[torch.from_numpy(a_w).long().to(self.device)]
        _, h = self.rnn(z_t, a_t)          # h is (h_n, c_n) after step t0 - 1

        self._h = h
        # The current latent at t0 -- the posterior MEAN, not a sample. It is
        # the best point estimate of the true latent, and the real-env runner
        # also feeds mu, so the two agree at t = 0 as well as thereafter.
        self._z = torch.from_numpy(self.pool.mu[e, t0]).to(self.device)
        self._gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        self.t = 0
        self.start_e, self.start_t0 = e, t0
        return self._obs()

    def _obs(self) -> Tuple[np.ndarray, np.ndarray]:
        # h_n[0] for a single-layer LSTM is exactly the last timestep's output,
        # i.e. the same tensor as parts["h"][:, -1]. Using h_n keeps the code
        # honest about the fact that it is the *carried* state.
        return (
            self._z.detach().cpu().numpy(),
            self._h[0][0].detach().cpu().numpy(),
        )

    # -------------------------------------------------------------- step

    @torch.no_grad()
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance every dream one frame.

        Returns ``(z_next, h_pre_next, reward, p_hit, done)``. ``done`` is
        always False and is returned only so the signature reads like a gym
        step; see the class docstring.
        """
        a = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        a_oh = self._eye[a]
        parts, h = self.rnn.step(self._z, a_oh, self._h)

        p_hit = torch.sigmoid(parts["hit_logit"][:, 0, 0])
        r_dense = parts["reward"][:, 0, 0]
        if self.reward_mode == "hit":
            r = p_hit
        elif self.reward_mode == "dense":
            r = r_dense
        else:
            r = p_hit + self.lam * r_dense

        z_next = self.rnn.sample_next(
            parts, temperature=self.temperature, generator=self._gen
        )[:, 0]

        self._z, self._h = z_next, h
        self.t += 1
        z_np, h_np = self._obs()
        return (
            z_np,
            h_np,
            r.cpu().numpy(),
            p_hit.cpu().numpy(),
            np.zeros(len(z_np), bool),
        )


# -------------------------------------------------------------- rollout loop


def dream_rollout(
    env: DreamEnv,
    act_fn,
    batch: int,
    steps: int,
    seed: int = 0,
    starts: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    record: bool = False,
) -> Dict[str, np.ndarray]:
    """Roll ``batch`` dreams for ``steps`` frames under ``act_fn(z, h) -> (B,)``.

    Returns per-episode summed reward plus, if ``record``, the whole latent and
    action trace (used to render the "controller playing inside its own dream"
    GIF). Recording 512 x 150 x 16 floats is 5 MB, but decoding them all is not
    free, so it is off by default.
    """
    z, h = env.reset(batch=batch, seed=seed, starts=starts)
    total_r = np.zeros(batch, np.float64)
    total_hit = np.zeros(batch, np.float64)
    zs: List[np.ndarray] = [z]
    hs: List[np.ndarray] = []
    acts: List[np.ndarray] = []
    hits: List[np.ndarray] = []

    for _ in range(steps):
        a = act_fn(z, h)
        if record:
            acts.append(a)
            hs.append(h)
        z, h, r, p_hit, _ = env.step(a)
        total_r += r
        total_hit += p_hit
        if record:
            zs.append(z)
            hits.append(p_hit)

    out = {"return": total_r, "hit_sum": total_hit}
    if record:
        out.update(
            z=np.stack(zs, 1),
            h=np.stack(hs, 1),
            actions=np.stack(acts, 1),
            p_hit=np.stack(hits, 1),
        )
    return out

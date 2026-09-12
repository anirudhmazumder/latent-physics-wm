"""Does the dreamed controller work in the REAL world? The only test that counts.

    python -m wm.eval_controller --vae runs/vae_b1/vae.pt --rnn runs/rnn_v1/rnn.pt \
        --ctrl runs/ctrl_v1/controller.pt runs/ctrl_z_only/controller.pt \
        --episodes 100 --steps 200 --out runs/ctrl_eval

Everything upstream of here can look healthy while the system is useless: the
VAE reconstructs, the RNN's NLL is low, CMA-ES's fitness curve rises smoothly.
None of that is evidence, because all three quantities are measured inside the
model's own head. This file puts the controller in ``BouncingBox`` -- real
physics, real pixels, the frozen encoder, no privileged state -- and counts
paddle contacts.

What the numbers mean
---------------------
Hits per episode is not interpretable on its own, because the number of
*chances* is fixed by physics rather than by policy. The ball takes ~55 frames
to cross the box vertically, so in 200 steps it reaches the floor about twice,
and no policy can score more than that. We therefore report

    floor visits / episode      the ceiling: how many chances existed
    hits / floor visit          the actual skill measure, in [0, 1]
    |ball_x - paddle_x| at the
        lowest point of a visit  how badly it missed, in world units

and the oracle (privileged true state) is run through the identical harness so
that "skill" has a top end to be a fraction of.

The timing convention is the one from ``wm.dream_env``, restated in code: at
step t the controller sees ``[mu_t, h_pre_t]``, where ``h_pre_t`` is the LSTM
state carried in from ``(mu_{t-1}, a_{t-1})``. The two implementations are
asserted equal in ``tests/test_controller.py``.

Two honest caveats, both visible in the numbers.

* ``h_pre`` starts at ZERO here, whereas a dream is warm-started from 8 true
  latents. For the first ~8 of 200 frames the controller is therefore reading a
  hidden state with no velocity in it. We leave it that way rather than
  hand-feeding warm-up frames, because a deployed agent does not get a warm-up;
  the ball needs ~50 frames to reach the floor anyway, so it costs nothing
  measurable.
* We feed the encoder's ``mu``, not a posterior sample. At test time the point
  estimate is what you want (sampling only adds encoder noise to every
  decision), and it matches ``compute_norm_stats`` and ``DreamEnv.reset``.

v2: mass enters the harness, not the policy
-------------------------------------------
With ``--mass-from-color`` the environment draws a mass per episode and the
ball's speed becomes ``ball_speed / m``. Nothing about the controller changes --
it still sees only ``[mu_t, h_pre_t]`` -- but two things about the *measurement*
have to.

* Every rollout now records the episode's mass (the 7th state column), so any
  metric can be split by mass. ``run_real_episodes`` returns ``mass`` and the
  derived ``speed`` alongside ``states``.
* The floor-visit band is ``contact_height + one frame of travel``, and one
  frame of travel is now mass-dependent. A light ball (m = 0.5) moves 0.044 per
  frame, twice the v1 step, so a fixed 0.147 threshold would let it jump the
  band and the visit would never be counted -- undercounting chances exactly on
  the episodes where chances are most frequent. ``floor_visit_stats`` therefore
  accepts a per-episode ``speed``. With ``speed=None`` it falls back to the v1
  constant and every v1 number is bit-identical.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from worldsim.bouncing_box import EVENT_PADDLE, BouncingBox, BoxConfig
from worldsim.render import save_gif, side_by_side

from .analyze import load_ckpt
from .controller import (
    BaseController,
    LinearController,
    OracleController,
    RandomController,
    StayController,
    batched_logits,
    load_controller,
    logits_to_actions,
    make_features,
)
from .eval_rnn import decode_latents
from .rnn import MDNRNN, load_rnn

N_ACTIONS = 3
ACTION_NAMES = ("left", "stay", "right")
DEFAULT_SEED_BASE = 5000


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ------------------------------------------------------------- env construction


def make_box_cfg(
    ball_radius: float = 0.08,
    mass_from_color: bool = False,
    mass_holdout: Optional[Sequence[float]] = None,
    mass_only: Optional[Sequence[float]] = None,
) -> BoxConfig:
    """The ONE place the evaluation environment is configured.

    Every rollout entry point (``run_real_episodes``, ``run_population_real``,
    ``real_vs_dream_gif``) routes through here so a v2 flag cannot reach one of
    them and not the others. With all the v2 arguments at their defaults this
    returns exactly the v1 config the harness has always built.
    """
    return BoxConfig(
        res=64,
        ball_radius=ball_radius,
        mass_from_color=bool(mass_from_color),
        mass_holdout=None if mass_holdout is None else tuple(mass_holdout),
        mass_only=None if mass_only is None else tuple(mass_only),
    )


def episode_masses(states: np.ndarray) -> Optional[np.ndarray]:
    """``(E,)`` mass per episode, or None for a v1 (6-column) rollout.

    Mass is constant within an episode, so the first frame's column is the
    episode's mass; we read column 6 rather than carrying a separate channel
    because that keeps the harness's single source of truth the environment's
    own ``state()``.
    """
    if states.shape[-1] < 7:
        return None
    return np.asarray(states[:, 0, 6], np.float64)


# ------------------------------------------------------------------ encoding


@torch.no_grad()
def encode_frames(vae, frames: np.ndarray, device: str = "cpu") -> np.ndarray:
    """``(B, 64, 64, 3) uint8 -> (B, z_dim)`` posterior means, frozen encoder."""
    x = torch.from_numpy(frames.astype(np.float32) / 255.0)
    x = x.permute(0, 3, 1, 2).to(device)
    mu, _ = vae.encode(x)
    return mu.cpu().numpy()


# --------------------------------------------------------------- the rollout


@torch.no_grad()
def run_real_episodes(
    ctrl: BaseController,
    vae,
    rnn: MDNRNN,
    episodes: int = 100,
    steps: int = 200,
    seed_base: int = DEFAULT_SEED_BASE,
    device: str = "cpu",
    ball_radius: float = 0.08,
    record_frames: int = 0,
    record_logits: bool = False,
    mass_from_color: bool = False,
    mass_holdout: Optional[Sequence[float]] = None,
    mass_only: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Run ``episodes`` real episodes in lockstep. Seeds are ``seed_base + i``.

    Lockstep rather than one-at-a-time because the expensive parts -- the VAE
    encode and the LSTM step -- are batched over episodes, turning 100 x 200
    forward passes into 200 forward passes of batch 100. The physics itself is
    still a Python loop; it is ~0.3 ms per env-step and not worth vectorising.

    Fixed seeds are the point of ``seed_base``: every controller and baseline
    sees the identical 100 initial conditions, so the comparison is paired and a
    difference of 0.1 hits/episode is not just a different draw of starts.
    """
    cfg = make_box_cfg(ball_radius, mass_from_color, mass_holdout, mass_only)
    envs = [BouncingBox(cfg) for _ in range(episodes)]
    frames = np.stack([e.reset(seed=seed_base + i) for i, e in enumerate(envs)])
    states = np.stack([e.state() for e in envs])                  # (E, 6) or (E, 7)

    ctrl.reset(episodes, seed=seed_base)
    h = rnn.init_hidden(episodes, device=device)                  # cold; see docstring
    eye = torch.eye(N_ACTIONS, device=device)

    all_states = [states.copy()]
    all_actions: List[np.ndarray] = []
    all_hits: List[np.ndarray] = []
    logits: List[np.ndarray] = []
    kept: List[np.ndarray] = [frames[:record_frames].copy()] if record_frames else []

    mu = encode_frames(vae, frames, device)                       # (E, z)

    for _t in range(steps):
        h_pre = h[0][0].cpu().numpy()                             # (E, hidden)
        if record_logits and isinstance(ctrl, LinearController):
            logits.append(ctrl.logits(mu, h_pre))
        a = np.asarray(ctrl.act(mu, h_pre, state=states), np.int64)

        nxt, sts, evs = [], [], []
        for i, env in enumerate(envs):
            f, s, ev = env.step(int(a[i]))
            nxt.append(f)
            sts.append(s)
            evs.append(ev)
        frames = np.stack(nxt)
        states = np.stack(sts)

        all_actions.append(a)
        all_hits.append(((np.asarray(evs) & EVENT_PADDLE) != 0).astype(np.float32))
        all_states.append(states.copy())
        if record_frames:
            kept.append(frames[:record_frames].copy())

        # Advance M with the latent the controller actually acted on and the
        # action it actually took -- this is the h_pre for the NEXT step.
        z_t = torch.from_numpy(mu.astype(np.float32)).to(device)
        _, h = rnn.step(z_t, eye[torch.from_numpy(a).to(device)], h)
        mu = encode_frames(vae, frames, device)

    out = {
        "states": np.stack(all_states, 1),        # (E, steps+1, 6 or 7)
        "actions": np.stack(all_actions, 1),      # (E, steps)
        "hits": np.stack(all_hits, 1),            # (E, steps)
    }
    # v2 bookkeeping. Recorded here, at the only place that owns the
    # environment, rather than re-derived by each analysis: the mass is a
    # property of the episode the harness ran, not of the states array's shape.
    mass = episode_masses(out["states"])
    if mass is not None:
        out["mass"] = mass
        out["speed"] = cfg.ball_speed / mass
    if record_frames:
        out["frames"] = np.stack(kept, 1)         # (record_frames, steps+1, 64,64,3)
    if logits:
        out["logits"] = np.stack(logits, 1)       # (E, steps, 3)
    return out


@torch.no_grad()
def run_population_real(
    params: np.ndarray,
    template: LinearController,
    vae,
    rnn: MDNRNN,
    seeds: Sequence[int],
    steps: int = 200,
    device: str = "cpu",
    ball_radius: float = 0.08,
    mass_from_color: bool = False,
    mass_holdout: Optional[Sequence[float]] = None,
    mass_only: Optional[Sequence[float]] = None,
    count: str = "frames",
) -> np.ndarray:
    """Evaluate ``P`` candidates on ``R`` real episodes each. Returns ``(P, R)`` hits.

    This is the fitness function of the no-world-model baseline
    (``train_controller --fitness real``), and it is structured exactly like
    ``DreamFitness``: ``P * R`` episodes in lockstep, one batched VAE encode and
    one batched LSTM step per frame, and the same seeds ``R`` for every
    candidate (common random numbers).

    It is also the cost centre of the whole comparison. ``P * R * steps`` real
    frames have to be *rendered and encoded* every generation, which the dream
    gets for free -- that asymmetry is the point of the experiment, so it is
    counted explicitly in ``RealFitness.env_steps_used``.

    ``count`` picks what a "hit" is worth. ``frames`` (the v1 default) adds one
    per contact FRAME, which is what v1's ``ctrl_real`` optimised -- and it
    found the loophole: pin the ball against the paddle and collect contact
    frames without ever really intercepting. ``interceptions`` collapses each
    run of consecutive contact frames to one, closing it. The v1 default is kept
    so the v1 baseline reproduces; v2 uses ``interceptions``.
    """
    P, R = len(params), len(seeds)
    B = P * R
    cfg = make_box_cfg(ball_radius, mass_from_color, mass_holdout, mass_only)
    envs = [BouncingBox(cfg) for _ in range(B)]
    frames = np.stack([
        envs[p * R + r].reset(seed=int(seeds[r])) for p in range(P) for r in range(R)
    ])
    h = rnn.init_hidden(B, device=device)
    eye = torch.eye(N_ACTIONS, device=device)
    hits = np.zeros((P, R), np.float64)
    # For `count="interceptions"` we need to know whether the PREVIOUS frame was
    # also a contact, so a run scores once. One bool per environment is all the
    # state that requires.
    was_contact = np.zeros(B, bool)
    mu = encode_frames(vae, frames, device)

    for _t in range(steps):
        h_pre = h[0][0].cpu().numpy()
        f = make_features(mu, h_pre, template.inputs, template.norm)
        a = logits_to_actions(batched_logits(params, f.reshape(P, R, -1)))
        a = a.reshape(B)

        nxt = []
        for i, env in enumerate(envs):
            fr, _s, ev = env.step(int(a[i]))
            nxt.append(fr)
            contact = bool(ev & EVENT_PADDLE)
            if contact and not (count == "interceptions" and was_contact[i]):
                hits[i // R, i % R] += 1.0
            was_contact[i] = contact
        frames = np.stack(nxt)

        z_t = torch.from_numpy(mu.astype(np.float32)).to(device)
        _, h = rnn.step(z_t, eye[torch.from_numpy(a).to(device)], h)
        mu = encode_frames(vae, frames, device)
    return hits


# ----------------------------------------------------------------- metrics


def _bootstrap_ci(
    x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap CI of the MEAN over episodes.

    Hits per episode is a small count (0, 1, 2, occasionally 3) over ~2
    opportunities, so it is nowhere near Gaussian and a t-interval is the wrong
    shape. The bootstrap resamples episodes, which is also the right unit of
    independence -- steps within an episode are heavily correlated.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def contact_height(ball_radius: float = 0.08, paddle_h: float = 0.045) -> float:
    """The ball_y below which a paddle contact is geometrically possible.

    The paddle sits flush on the floor, so ``paddle_y = paddle_h / 2`` and its
    top edge is at ``paddle_h``. Add one ball radius: 0.125 at the defaults.
    """
    return paddle_h + ball_radius


def floor_zone_height(
    ball_radius: float = 0.08, paddle_h: float = 0.045, ball_speed: float = 0.022
) -> float:
    """Contact height plus one frame of travel: the "a chance is happening" band.

    Using ``contact_height`` itself as the threshold silently undercounts, and
    it undercounts *exactly the good policies*, which is the worst possible bias
    here. When the paddle intercepts, the collision resolver places the ball at
    ``paddle_top + r`` -- i.e. exactly AT the contact height, never below it --
    so a strict ``y < contact_height`` test fires on missed approaches and not
    on successful ones. Measured on ``data/v1/val``: 1.07 visits/episode at
    0.125 versus 1.80 at 0.147, and 1.80 is the number that matches "the ball
    crosses the box every ~55 frames". One ball_speed of headroom is the
    smallest margin that makes the band policy-independent.
    """
    return contact_height(ball_radius, paddle_h) + ball_speed


def floor_visit_stats(
    states: np.ndarray,
    ball_radius: float = 0.08,
    paddle_h: float = 0.045,
    speed: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Count the CHANCES each episode offered, and how close the paddle was.

    A "floor visit" is a downward crossing of ``ball_y`` below
    ``paddle_top + ball_radius`` -- the height at which contact first becomes
    geometrically possible. Counting crossings rather than frames-below matters:
    the ball spends several frames in the zone per approach, and we want one
    chance per approach.

    ``speed`` is the per-episode ``|v|`` (scalar or ``(E,)``). It sets the
    one-frame headroom on the band. In v1 it is a constant 0.022 and omitting it
    reproduces the v1 threshold of 0.147 exactly. In v2 it is ``0.022 / m``,
    which for a light ball is 0.044: a light ball can cross a 0.022-wide band in
    a single frame, so a fixed threshold would silently drop its visits and
    inflate its interceptions-per-visit. Getting this wrong biases exactly the
    comparison the whole stage is about.
    """
    y = states[..., 1]
    if speed is None:
        thr = floor_zone_height(ball_radius, paddle_h)
    else:
        sp = np.asarray(speed, np.float64)
        thr = floor_zone_height(ball_radius, paddle_h, 0.0) + sp
        if sp.ndim == 1:
            thr = thr[:, None]                # broadcast over time
    below = y < thr
    # A crossing is a False -> True transition along time.
    entering = below[:, 1:] & ~below[:, :-1]       # (E, T)
    n_visits = entering.sum(1).astype(np.float64)

    # Closest approach: |ball_x - paddle_x| at the lowest frame of each visit.
    gaps: List[float] = []
    owner: List[int] = []          # which episode each gap came from
    dx = np.abs(states[..., 0] - states[..., 4])
    for e in range(states.shape[0]):
        idx = np.flatnonzero(entering[e]) + 1
        for t0 in idx:
            t1 = t0
            while t1 + 1 < below.shape[1] and below[e, t1 + 1]:
                t1 += 1
            seg = slice(t0, t1 + 1)
            gaps.append(float(dx[e, seg][np.argmin(y[e, seg])]))
            owner.append(e)
    return {
        "floor_visits": n_visits,
        "gap_at_floor": np.asarray(gaps, np.float64),
        # The episode index behind each gap, so gaps can be grouped by mass
        # without re-running the scan.
        "gap_episode": np.asarray(owner, np.int64),
    }


def contact_runs(hits: np.ndarray) -> np.ndarray:
    """Count *interceptions* rather than contact frames, per episode.

    ``EVENT_PADDLE`` is a per-frame flag, and one interception can set it on
    several consecutive frames -- or a policy can pin the ball against the
    paddle and rack up contacts without ever really "returning" it. Since
    hits/episode is the headline number, it needs a companion that cannot be
    inflated that way: a run of consecutive contact frames counts once.
    """
    h = hits > 0
    starts = h[:, 1:] & ~h[:, :-1]
    return (h[:, :1].astype(np.float64).squeeze(-1) + starts.sum(1)).astype(np.float64)


def summarise(name: str, roll: Dict[str, np.ndarray], seed: int = 0) -> Dict:
    hits = roll["hits"].sum(1)                                   # (E,)
    runs = contact_runs(roll["hits"])
    lo, hi = _bootstrap_ci(hits, seed=seed)
    rlo, rhi = _bootstrap_ci(runs, seed=seed)
    fv = floor_visit_stats(roll["states"], speed=roll.get("speed"))
    visits = fv["floor_visits"]
    a = roll["actions"].ravel()
    return {
        "name": name,
        "episodes": int(len(hits)),
        "steps": int(roll["actions"].shape[1]),
        "hits_per_episode": float(hits.mean()),
        "hits_ci95": [lo, hi],
        "hits_std": float(hits.std()),
        "interceptions_per_episode": float(runs.mean()),
        "interceptions_ci95": [rlo, rhi],
        "episodes_with_a_hit": float((hits > 0).mean()),
        "floor_visits_per_episode": float(visits.mean()),
        "hit_rate_per_floor_visit": float(hits.sum() / max(visits.sum(), 1)),
        # The mass-fair skill metric. Interceptions rather than contact frames
        # (no pinning loophole) and per CHANCE rather than per episode -- a
        # light ball reaches the floor several times more often than a heavy
        # one, so per-episode counts mostly measure the physics, not the policy.
        "interceptions_per_floor_visit": float(runs.sum() / max(visits.sum(), 1)),
        "mean_gap_at_floor": float(fv["gap_at_floor"].mean()),
        "action_dist": {
            ACTION_NAMES[k]: float((a == k).mean()) for k in range(N_ACTIONS)
        },
        # Underscored keys are stripped before JSON; they carry the per-episode
        # vectors that the plots and the by-mass analysis need.
        "_hits": hits,
        "_interceptions": runs,
        "_visits": visits,
        "_gaps": fv["gap_at_floor"],
        "_gap_episode": fv["gap_episode"],
        "_mass": roll.get("mass"),
    }


# ------------------------------------------------------------------- render


def _tint_border(frames: np.ndarray, mask: np.ndarray, color, width: int = 2):
    """Paint a coloured frame around timesteps where ``mask`` is True.

    Used to show the hit head's belief inside the dream GIF. A border is the
    right channel for this: it cannot be confused with the ball or the paddle,
    and it needs no text rendering (the GIF is 64x64; text would be illegible).
    """
    out = frames.copy()
    c = np.asarray(color, np.uint8)
    for t in np.flatnonzero(mask):
        out[t, :width, :] = c
        out[t, -width:, :] = c
        out[t, :, :width] = c
        out[t, :, -width:] = c
    return out


@torch.no_grad()
def dream_play_gif(
    ctrl: BaseController,
    rnn: MDNRNN,
    vae,
    pool,
    out_path: Path,
    steps: int = 200,
    temperature: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
    candidates: int = 16,
    starts: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Path:
    """The controller playing inside M's head, decoded by V for our benefit.

    The decoder is a *microscope*, not part of the loop: the controller never
    sees a pixel here. If the dream looks like plausible physics, M's latent
    trajectory is on-manifold; if it dissolves into a smear, the controller has
    been optimising against something that is not a world.

    We roll ``candidates`` dreams and render the one with the most predicted
    contact, because a single random start very often contains *none*: the ball
    reaches the floor roughly twice per 200 frames and the hit head fires on 1-2
    frames per interception, so ~0.8% of dreamed frames are contacts. This is a
    cherry-pick and is labelled as one -- it is a demo of the green hit-head
    indicator, not evidence about the average dream. The averages are in
    ``summary.json``.

    ``starts`` pins the ``(episode, t0)`` pairs the dreams begin from, which is
    how v2 asks for a *light* ball: speed is only visible in a GIF if the ball
    actually moves, and a random draw from a log-uniform mass gives a slow one
    half the time.
    """
    from .dream_env import DreamEnv, dream_rollout

    env = DreamEnv(rnn, pool, temperature=temperature, reward="mix", device=device)
    rec = dream_rollout(
        env, lambda z, h: ctrl.act(z, h), batch=candidates, steps=steps,
        seed=seed, record=True, starts=starts,
    )
    pick = int(rec["hit_sum"].argmax())
    z = rec["z"][pick]                                        # (steps+1, 16)
    frames = decode_latents(vae, z, device=device)            # (steps+1, 64,64,3)
    p_hit = np.concatenate([[0.0], rec["p_hit"][pick]])
    frames = _tint_border(frames, p_hit > 0.5, (90, 235, 120))
    return save_gif(frames, out_path, fps=20, scale=4)


@torch.no_grad()
def real_vs_dream_gif(
    ctrl: BaseController,
    rnn: MDNRNN,
    vae,
    out_path: Path,
    steps: int = 150,
    warmup: int = 8,
    temperature: float = 1.0,
    seed: int = DEFAULT_SEED_BASE,
    device: str = "cpu",
    ball_radius: float = 0.08,
    mass_from_color: bool = False,
    mass_holdout: Optional[Sequence[float]] = None,
    mass_only: Optional[Sequence[float]] = None,
) -> Path:
    """Left: the controller in the real box. Right: the same controller dreaming.

    Both branches share the first ``warmup`` real frames, so at the split they
    hold the *same* latent and the *same* hidden state. Everything after that is
    divergence, and divergence is the thing worth looking at: the dream stays
    self-consistent (M's physics does not fall apart) while drifting away from
    the truth, which is exactly the regime in which a dream-trained controller
    can still transfer -- it learned a reflex, not a trajectory.
    """
    from .dream_env import DreamEnv, StartPool

    cfg = make_box_cfg(ball_radius, mass_from_color, mass_holdout, mass_only)
    env = BouncingBox(cfg)
    f = env.reset(seed=seed)
    ctrl.reset(1, seed=seed)

    h = rnn.init_hidden(1, device=device)
    eye = torch.eye(N_ACTIONS, device=device)
    real_frames, mus, acts = [f], [], []
    state = env.state()

    for t in range(warmup + steps):
        mu = encode_frames(vae, f[None], device)
        a = int(ctrl.act(mu, h[0][0].cpu().numpy(), state=state[None])[0])
        mus.append(mu[0])
        acts.append(a)
        f, state, _ = env.step(a)
        real_frames.append(f)
        _, h = rnn.step(
            torch.from_numpy(mu.astype(np.float32)).to(device),
            eye[torch.tensor([a], device=device)],
            h,
        )
        if t == warmup - 1:
            # Freeze a StartPool of exactly this one trajectory prefix so the
            # dream re-derives the identical warm-up hidden state.
            mu_hist = np.stack(mus + [encode_frames(vae, f[None], device)[0]])
            a_hist = np.asarray(acts, np.int64)

    pool = StartPool(
        mu_hist[None].astype(np.float32),
        a_hist[:warmup][None],
        np.zeros((1, warmup + 1, state.shape[-1]), np.float32),
        warmup,
    )
    denv = DreamEnv(rnn, pool, temperature=temperature, reward="mix", device=device)
    z, hh = denv.reset(batch=1, seed=seed, starts=(np.array([0]), np.array([warmup])))
    ctrl.reset(1, seed=seed)
    zs = [z[0]]
    for _ in range(steps):
        a = ctrl.act(z, hh)
        z, hh, _, _, _ = denv.step(a)
        zs.append(z[0])

    dream_frames = decode_latents(vae, np.stack(zs), device=device)
    real = np.stack(real_frames[warmup : warmup + steps + 1])
    stack = side_by_side([real, dream_frames[: len(real)]])
    return save_gif(stack, out_path, fps=20, scale=4)


# --------------------------------------------------------------- diagnostics


def explain_decisions(
    roll: Dict[str, np.ndarray], out: Path, ball_radius: float = 0.08
) -> Dict:
    """What does the controller's decision actually depend on?

    Reading ``W`` directly is useless: 256 of its 272 columns multiply LSTM
    units with no individual meaning. So we probe the controller the way stage
    two probed the RNN -- regress its *output* on interpretable true-state
    features and see which ones carry it.

    The target is the signed drive ``logit(RIGHT) - logit(LEFT)``, because that
    is the quantity whose sign decides which way the paddle goes (STAY wins only
    when both are beaten, which is a second-order effect).

    The three nested models are the actual question:

        (1) x_err = ball_x - paddle_x            "move toward where the ball IS"
        (2) (1) + ball velocities and ball_y     enough information to extrapolate
        (3) x_pred_err = predicted landing spot  "move toward where it WILL BE"

    A controller that only has (1) is a follower and will always arrive late,
    because the ball crosses the box in ~45 frames while the paddle needs ~30 to
    cross. If (3) explains much more variance than (1), C learned to lead the
    ball -- which it can only have done by reading velocity out of h.
    """
    if "logits" not in roll:
        return {}
    lg = roll["logits"]                                   # (E, T, 3)
    s = roll["states"][:, : lg.shape[1]]                  # (E, T, 6) at decision time
    drive = (lg[..., 2] - lg[..., 0]).ravel()

    ball_x, ball_y, vx, vy, pad_x, pad_vx = [s[..., i].ravel() for i in range(6)]
    x_err = ball_x - pad_x

    # Ballistic landing estimate, ignoring wall bounces: where the ball will be
    # when it reaches contact height. Only defined while descending.
    thr = contact_height(ball_radius)
    t_fall = np.where(vy < -1e-6, (ball_y - thr) / np.maximum(-vy, 1e-6), 0.0)
    t_fall = np.clip(t_fall, 0.0, 120.0)
    x_land = ball_x + vx * t_fall
    # Reflect off the side walls once, which is the common case.
    x_land = np.abs(x_land)
    x_land = np.where(x_land > 1.0, 2.0 - x_land, x_land)
    x_pred_err = np.clip(x_land, 0.0, 1.0) - pad_x

    def fit(names: Sequence[str], X: np.ndarray) -> Dict:
        A = np.concatenate([np.ones((len(X), 1)), X], 1)
        coef, *_ = np.linalg.lstsq(A, drive, rcond=None)
        resid = drive - A @ coef
        r2 = 1.0 - resid.var() / max(drive.var(), 1e-12)
        return {
            "r2": float(r2),
            "coef": {n: float(c) for n, c in zip(("bias",) + tuple(names), coef)},
        }

    res = {
        "m1_position_only": fit(("x_err",), x_err[:, None]),
        "m2_position_and_velocity": fit(
            ("x_err", "ball_vx", "ball_vy", "ball_y", "paddle_vx"),
            np.stack([x_err, vx, vy, ball_y, pad_vx], 1),
        ),
        "m3_predicted_landing": fit(("x_pred_err",), x_pred_err[:, None]),
        "drive_std": float(drive.std()),
    }

    # R^2 on the raw drive understates the story, because the policy is
    # effectively bang-bang: only the SIGN of the drive reaches the world, and a
    # saturating, near-binary target is badly served by a linear fit. The
    # decision-relevant question is simply how often the controller moves the
    # right way, so we also report sign agreement -- overall, and restricted to
    # the regime that actually matters (ball descending in the lower half,
    # i.e. the approach where the interception is decided).
    approach = (vy < 0) & (ball_y < 0.5)

    def agree(target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        m = np.ones_like(target, bool) if mask is None else mask
        m = m & (np.abs(target) > 0.02)      # ignore ties inside the dead zone
        if m.sum() == 0:
            return float("nan")
        return float((np.sign(drive[m]) == np.sign(target[m])).mean())

    res["sign_agreement"] = {
        "toward_ball_all_frames": agree(x_err),
        "toward_ball_on_approach": agree(x_err, approach),
        "toward_predicted_landing_on_approach": agree(x_pred_err, approach),
        "approach_frames": int(approach.sum()),
    }

    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    keys = ["m1_position_only", "m2_position_and_velocity", "m3_predicted_landing"]
    ax[0].bar(range(3), [res[k]["r2"] for k in keys], color="#3a7bd5")
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(["ball_x\n- paddle_x", "+ velocity,\nball_y", "predicted\nlanding x"])
    ax[0].set_ylabel("R² explaining logit(RIGHT) − logit(LEFT)")
    ax[0].set_title("what drives the controller's decision?")
    ax[0].set_ylim(0, 1)
    for i, k in enumerate(keys):
        ax[0].text(i, res[k]["r2"] + 0.02, f"{res[k]['r2']:.2f}", ha="center")

    sa = res["sign_agreement"]
    keys2 = ["toward_ball_all_frames", "toward_ball_on_approach",
             "toward_predicted_landing_on_approach"]
    ax[1].bar(range(3), [sa[k] for k in keys2], color="#d55e3a")
    ax[1].axhline(0.5, ls="--", c="k", lw=1)
    ax[1].text(2.4, 0.51, "chance", ha="right", fontsize=8)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(["toward ball\n(all frames)", "toward ball\n(approach)",
                           "toward predicted\nlanding (approach)"], fontsize=8)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("fraction of steps where the paddle moves that way")
    ax[1].set_title("does it move the right way?")
    for i, k in enumerate(keys2):
        ax[1].text(i, sa[k] + 0.02, f"{sa[k]:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return res


def hits_bar(rows: Sequence[Dict], out: Path) -> Path:
    plt = _plt()
    order = list(rows)
    names = [r["name"] for r in order]
    vals = [r["hits_per_episode"] for r in order]
    err = np.array(
        [[v - r["hits_ci95"][0], r["hits_ci95"][1] - v] for r, v in zip(order, vals)]
    ).T
    ceil = np.mean([r["floor_visits_per_episode"] for r in order])

    icp = [r["interceptions_per_episode"] for r in order]
    ierr = np.array(
        [[v - r["interceptions_ci95"][0], r["interceptions_ci95"][1] - v]
         for r, v in zip(order, icp)]
    ).T

    fig, ax = plt.subplots(figsize=(2.0 + 1.2 * len(names), 4.4))
    colors = ["#888" if n in ("random", "stay") else
              "#2a9d4a" if n == "oracle" else "#3a7bd5" for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, vals, width=0.4, yerr=err, capsize=3, color=colors,
           label="contact frames")
    # The paired bar is not decoration: hits/episode can exceed the number of
    # floor visits (a policy can touch the ball on several consecutive frames),
    # so the dashed ceiling only bounds the hatched series.
    ax.bar(x + 0.2, icp, width=0.4, yerr=ierr, capsize=3, color=colors,
           alpha=0.55, hatch="//", edgecolor="white", label="interceptions")
    ax.axhline(ceil, ls="--", c="k", lw=1)
    ax.text(len(names) - 0.4, ceil + 0.03, "floor visits (ceiling)", ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("paddle contacts per 200-step episode")
    ax.set_title("real-environment performance (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ------------------------------------------------------------------- report


def write_report(rows: List[Dict], extra: Dict, out: Path) -> None:
    clean = []
    for r in rows:
        r = {k: v for k, v in r.items() if not k.startswith("_")}
        clean.append(r)
    (out / "summary.json").write_text(
        json.dumps({"controllers": clean, **extra}, indent=2)
    )

    lines = [
        "# Stage three (C) — real-environment evaluation",
        "",
        f"{clean[0]['episodes']} episodes x {clean[0]['steps']} steps, "
        f"fixed seeds {DEFAULT_SEED_BASE}..{DEFAULT_SEED_BASE + clean[0]['episodes'] - 1} "
        "(identical starts for every row).",
        "",
        "`hits` counts contact FRAMES; `interceptions` collapses a run of "
        "consecutive contact frames to one, so a policy that pins the ball "
        "against the paddle cannot inflate it.",
        "",
        "| controller | hits/ep | 95% CI | interceptions/ep | floor visits/ep | "
        "hits per visit | interceptions per visit | mean gap at floor | "
        "eps with ≥1 hit | left/stay/right |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in clean:
        d = r["action_dist"]
        lines.append(
            f"| `{r['name']}` | **{r['hits_per_episode']:.2f}** | "
            f"[{r['hits_ci95'][0]:.2f}, {r['hits_ci95'][1]:.2f}] | "
            f"{r['interceptions_per_episode']:.2f} | "
            f"{r['floor_visits_per_episode']:.2f} | "
            f"{r['hit_rate_per_floor_visit']:.2f} | "
            f"{r['interceptions_per_floor_visit']:.2f} | "
            f"{r['mean_gap_at_floor']:.3f} | "
            f"{r['episodes_with_a_hit']:.0%} | "
            f"{d['left']:.2f}/{d['stay']:.2f}/{d['right']:.2f} |"
        )
    dec = extra.get("decisions") or {}
    if dec:
        lines += [
            "",
            "## Does each controller move the right way?",
            "",
            "Sign agreement between `logit(RIGHT) − logit(LEFT)` and the "
            "direction of the true error (chance = 0.50). "
            "'approach' = ball descending in the lower half of the box.",
            "",
            "| controller | toward ball (all) | toward ball (approach) | "
            "toward ballistic landing (approach) |",
            "|---|---|---|---|",
        ]
        for nm, d in dec.items():
            s = d["sign_agreement"]
            lines.append(
                f"| `{nm}` | {s['toward_ball_all_frames']:.3f} | "
                f"{s['toward_ball_on_approach']:.3f} | "
                f"{s['toward_predicted_landing_on_approach']:.3f} |"
            )

    if extra.get("headline") in dec:
        d = dec[extra["headline"]]
        sa = d["sign_agreement"]
        lines += [
            "",
            f"## What drives `{extra['headline']}`'s decisions",
            "",
            "R² of a linear fit of `logit(RIGHT) − logit(LEFT)` on true-state features:",
            "",
            "| features | R² |",
            "|---|---|",
            f"| ball_x − paddle_x | {d['m1_position_only']['r2']:.3f} |",
            f"| + ball_vx, ball_vy, ball_y, paddle_vx | {d['m2_position_and_velocity']['r2']:.3f} |",
            f"| predicted landing x − paddle_x | {d['m3_predicted_landing']['r2']:.3f} |",
            "",
            "Coefficients of the middle model "
            "(drive std = " + f"{d['drive_std']:.2f}" + "):",
            "",
            "| feature | coefficient |",
            "|---|---|",
        ] + [
            f"| {k} | {v:+.3f} |"
            for k, v in d["m2_position_and_velocity"]["coef"].items()
        ] + [
            "",
            f"Approach frames used for the sign test: {sa['approach_frames']}.",
        ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# --------------------------------------------------------------------- main


def add_env_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The v2 environment flags, defined once and shared by both entry points.

    ``wm.train_controller`` and ``wm.eval_controller`` must agree about what
    world they are talking about; the cheapest way to guarantee that is for the
    flags to have exactly one definition. All three default to the v1 world.
    """
    p.add_argument("--mass-from-color", action="store_true",
                   help="v2: the ball's colour sets its mass, and mass sets its "
                        "speed (0.022/m) and how much english the paddle imparts")
    p.add_argument("--mass-holdout", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="never sample a mass inside [LO, HI]. Use (0.85, 1.2) to "
                        "evaluate on the masses the models were trained on")
    p.add_argument("--mass-only", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="sample ONLY inside [LO, HI]. Use (0.85, 1.2) for the "
                        "held-out-colour generalisation test")
    return p


def env_kwargs(a: argparse.Namespace) -> Dict:
    """``argparse.Namespace -> the kwargs every rollout entry point takes``."""
    return {
        "ball_radius": a.ball_radius,
        "mass_from_color": bool(getattr(a, "mass_from_color", False)),
        "mass_holdout": getattr(a, "mass_holdout", None),
        "mass_only": getattr(a, "mass_only", None),
    }


def build_baselines() -> List[BaseController]:
    return [
        StayController(),
        RandomController(mean_hold=8.0, seed=1234),
        OracleController(paddle_w=0.26),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae", default="runs/vae_b1/vae.pt")
    p.add_argument("--rnn", default="runs/rnn_v1/rnn.pt")
    p.add_argument("--ctrl", nargs="*", default=[],
                   help="controller.pt checkpoints to evaluate, in order")
    p.add_argument("--ctrl-names", nargs="*", default=None,
                   help="display names; defaults to each checkpoint's parent dir")
    p.add_argument("--ctrl-which", nargs="*", default=None,
                   help="per-checkpoint parameter set: 'params_best_real' (the "
                        "default, selected on periodic real evaluations) or "
                        "'params_last_dream' (CMA-ES's final mean -- what the "
                        "DREAM alone would have chosen). Comparing the two "
                        "measures how much of the headline number is selection "
                        "on the real-eval noise.")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    p.add_argument("--ball-radius", type=float, default=0.08)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="runs/ctrl_eval")
    p.add_argument("--gif-for", nargs="*", default=None,
                   help="names of controllers to render GIFs for "
                        "(default: the first --ctrl). The first one is the "
                        "'headline' controller used for controller_weights.png "
                        "and real_vs_dream_side_by_side.gif")
    p.add_argument("--gif-steps", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="dream temperature used for the dream GIFs only")
    p.add_argument("--no-gifs", action="store_true")
    p.add_argument("--dream-roots", nargs="*",
                   default=["data/v1/train", "data/v1/train_mix"])
    add_env_args(p)
    a = p.parse_args()
    ekw = env_kwargs(a)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    vae, _, _ = load_ckpt(a.vae, a.device)
    rnn, _ = load_rnn(a.rnn, a.device)

    ctrls: List[BaseController] = build_baselines()
    names = a.ctrl_names or [Path(c).parent.name for c in a.ctrl]
    which = a.ctrl_which or ["params"] * len(a.ctrl)
    trained: List[LinearController] = []
    for path, nm, wh in zip(a.ctrl, names, which):
        trained.append(load_controller(path, name=nm, which=wh))
    ctrls += trained

    gif_for: List[str] = list(a.gif_for) if a.gif_for else names[:1]
    rows: List[Dict] = []
    rolls: Dict[str, Dict[str, np.ndarray]] = {}
    for c in ctrls:
        t = time.time()
        roll = run_real_episodes(
            c, vae, rnn,
            episodes=a.episodes, steps=a.steps, seed_base=a.seed_base,
            device=a.device, **ekw,
            # Logits are only defined for a linear controller, and they are
            # cheap (100 x 200 x 3 floats), so record them for all of them: the
            # decision diagnostic is far more interesting *compared across*
            # variants than in isolation.
            record_logits=True,
        )
        rows.append(summarise(c.name, roll))
        rolls[c.name] = roll
        print(f"  {c.name:<14} hits/ep {rows[-1]['hits_per_episode']:.2f}  "
              f"({time.time() - t:.0f}s)")

    hits_bar(rows, out / "hits_bar.png")

    decisions: Dict[str, Dict] = {}
    for nm in names:
        if nm in rolls and "logits" in rolls[nm]:
            decisions[nm] = explain_decisions(
                rolls[nm], out / f"controller_weights_{nm}.png", a.ball_radius
            )
    headline = gif_for[0] if gif_for else (names[0] if names else None)
    if headline in decisions:
        # The spec'd filename, kept as a copy of the headline controller's plot.
        import shutil

        shutil.copyfile(out / f"controller_weights_{headline}.png",
                        out / "controller_weights.png")

    if not a.no_gifs and gif_for:
        from .dream_env import load_start_pool

        pool = load_start_pool(a.dream_roots, warmup=8)
        for i, nm in enumerate(gif_for):
            c = next(c for c in ctrls if c.name == nm)
            print(f"\nrendering GIFs for {nm}")
            roll = run_real_episodes(
                c, vae, rnn, episodes=1, steps=a.gif_steps, seed_base=a.seed_base,
                device=a.device, record_frames=1, **ekw,
            )
            save_gif(roll["frames"][0], out / f"real_play_{nm}.gif", fps=20, scale=4)
            dream_play_gif(c, rnn, vae, pool, out / f"dream_play_{nm}.gif",
                           steps=a.gif_steps, temperature=a.temperature,
                           seed=a.seed_base, device=a.device)
            suffix = "" if i == 0 else f"_{nm}"
            real_vs_dream_gif(
                c, rnn, vae, out / f"real_vs_dream_side_by_side{suffix}.gif",
                steps=min(a.gif_steps, 150), temperature=a.temperature,
                seed=a.seed_base, device=a.device, **ekw,
            )

    write_report(
        rows,
        {
            "decisions": decisions,
            "headline": headline,
            "vae": a.vae,
            "rnn": a.rnn,
            "controllers_evaluated": dict(zip(names, a.ctrl)),
            "seed_base": a.seed_base,
            "env": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                    for k, v in ekw.items()},
            "wall_clock_s": round(time.time() - t0, 1),
        },
        out,
    )
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

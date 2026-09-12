"""Did M learn the causal edge from APPEARANCE to DYNAMICS? Six experiments.

    python -m wm.eval_causal_v2 --ckpt runs/rnn_v2/rnn.pt \
        --nocolor-ckpt runs/rnn_v2_nocolor/rnn.pt \
        --vae runs/vae_v2/vae.pt --nocolor-vae runs/vae_b1/vae.pt \
        --out runs/rnn_v2/causal

This is the point of v2. In v1 everything that mattered about the ball's motion
could be inferred FROM motion: two frames give you velocity. A dynamics model
that scores well on v1 has shown it can integrate; it has not shown it can learn
that *what a thing looks like* tells you *how it will behave*.

v2 puts a second route in the world. The ball's colour encodes its mass, and
mass sets the speed (``speed = 0.022 / m``) and the paddle's english
(``dvx = 0.35 * paddle_vx / m``). So M can succeed two ways:

    (1) watch a few frames and integrate  -- the v1 skill, nothing new
    (2) look at the colour and know       -- the causal law

Teacher-forced NLL cannot tell these apart, because after a handful of frames
route (1) is sufficient and colour is redundant. Every experiment here is
designed to remove route (1) and see whether anything is left.

    (a) cold start      one frame of warm-up. No motion exists yet. Does the
                        dreamed ball move at the speed its colour implies?
    (b) recolour        an INTERVENTION: repaint a real ball and re-dream. Does
                        the dream speed up? There is no ground-truth trajectory
                        to compare against -- the physics would have been
                        different -- so the comparison is dream against LAW.
    (c) speed in h      probe z and h for speed and log_mass, by mass tercile.
    (d) interpolation   (a) restricted to the held-out mass band [0.85, 1.2],
                        colours whose dynamics M has never been shown.
    (e) english         the sparse effect, around real paddle contacts.
    (f) horizon by mass does the extra factor cost dream quality unevenly?

THE CONTROL IS THE RESULT. Every experiment is run for ``rnn_v2`` and for
``rnn_v2_nocolor`` -- the same architecture, the same budget, trained on the
same frames encoded by the *v1* VAE, which never saw a coloured ball and (probe:
speed R^2 = -0.76) carries no colour. The nocolor model is the model that can
only do (1). A gap between the two is the causal edge, measured.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from worldsim.bouncing_box import (
    BouncingBox, BoxConfig, color_to_mass, mass_to_color,
)
from worldsim.render import save_gif, side_by_side

from .analyze import load_ckpt
from .eval_rnn import (
    BALL_RADIUS,
    BASE_SPEED,
    StateProbe,
    _plt,
    _reduce_for_poly,
    decode_latents,
    dream,
    horizon_by_mass,
)
from .probes import make_split, probe_suite
from .rnn import MDNRNN, load_rnn
from .seq_data import episode_arrays
from .train_vae import pick_device

# Dream-speed estimator window. The first few steps are skipped because h is
# still settling after a short warm-up (at K=1 it is settling from zero), and
# the tail is dropped because a long dream drifts. Steps 4..20 of a 24-step
# dream is a compromise between "h has settled" and "the dream is still real".
SPEED_LO, SPEED_HI = 4, 20


# ----------------------------------------------------------------- helpers


def dreamed_speed(pos: np.ndarray, lo: int = SPEED_LO, hi: int = SPEED_HI) -> np.ndarray:
    """Per-step speed of a dreamed trajectory. ``pos`` is (N, H, 2) -> (N,).

    The MEDIAN per-step displacement, not the mean, for two reasons that both
    bite here. A wall bounce inside the window shortens one step (the ball
    reverses mid-step), and the kNN position probe occasionally snaps a dreamed
    latent to a neighbour some distance away, which adds a large one-off jump.
    Both are outliers in a list of ~16 otherwise-identical displacements, and
    the median ignores them where a mean would not.

    Note this measures SPEED (a magnitude), which is exactly the quantity the
    colour determines; it says nothing about direction, which colour does not.
    """
    hi = min(hi, pos.shape[1] - 1)
    step = np.linalg.norm(np.diff(pos[:, lo : hi + 1], axis=1), axis=-1)  # (N, W)
    return np.median(step, axis=1)


def recolor_frames(
    states: np.ndarray, cfg: BoxConfig, mass: Optional[float] = None
) -> np.ndarray:
    """Re-render true states, optionally repainting the ball for a NEW mass.

    ``states`` is (T, 7) rows of the v2 state vector. The renderer is a pure
    function of (ball position, paddle position, ball colour), so we can drive
    it directly from recorded state and get frames that are bit-identical to the
    dataset's own when ``mass`` is None -- which the test suite checks, and which
    is what makes the intervention exact rather than approximate.

    With ``mass=m1`` the positions are still the ones the ball with mass m0
    actually visited, and only the hue changes. That is the intervention we
    want: appearance alone, physics untouched. It also means the recoloured
    warm-up is internally INCONSISTENT (the colour says one speed, the motion
    across those frames says another) -- see the K=1 variant in part (b), which
    removes the motion cue and leaves only the colour.
    """
    env = BouncingBox(cfg)
    env.reset(seed=0)
    out = []
    for s in np.asarray(states):
        env.ball = np.array([float(s[0]), float(s[1])], dtype=np.float64)
        env.paddle_x = float(s[4])
        m = float(s[6]) if mass is None else float(mass)
        env.ball_color = mass_to_color(m, cfg)
        out.append(env.render())
    return np.stack(out)


@torch.no_grad()
def encode_frames(vae, frames: np.ndarray, device: str = "cpu") -> np.ndarray:
    """(..., 64, 64, 3) uint8 -> (..., z_dim) posterior means."""
    shape = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:]).astype(np.float32) / 255.0
    x = torch.from_numpy(flat).permute(0, 3, 1, 2).to(device)
    mu, _ = vae.encode(x)
    return mu.cpu().numpy().reshape(*shape, -1)


def _ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """(slope, intercept, pearson r) for y ~ x. NaN-safe on degenerate input."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(slope), float(intercept), r


def _terciles(mass: np.ndarray) -> Dict[str, np.ndarray]:
    q = np.quantile(mass, [1 / 3, 2 / 3])
    return {
        "light": mass <= q[0],
        "medium": (mass > q[0]) & (mass <= q[1]),
        "heavy": mass > q[1],
    }


def _r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1.0 - float(((y - p) ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)


# ------------------------------------------------------------------ setups


@dataclass
class Setup:
    """One (dynamics model, latent space, decoder, position probe) quadruple.

    The nocolor control is not just a different checkpoint: it lives in a
    DIFFERENT latent space (the v1 VAE's). So its decoder and its measuring
    probe have to be the v1 VAE's too, or every number would be measured with
    the wrong ruler. Bundling the four together makes that impossible to forget.
    """

    name: str
    model: MDNRNN
    suffix: str
    vae: object
    probe: StateProbe = field(default=None)


@dataclass
class Episodes:
    """One stacked dataset (val / holdout), in one latent space."""

    mu: np.ndarray
    actions: np.ndarray
    state: np.ndarray
    hit: np.ndarray
    names: List[str]

    @property
    def mass(self) -> np.ndarray:
        return self.state[:, 0, self.names.index("mass")]

    @property
    def true_speed(self) -> np.ndarray:
        return BASE_SPEED / self.mass


def load_episodes(roots: Sequence[str], suffix: str) -> Episodes:
    parts = [episode_arrays(r, latent_suffix=suffix) for r in roots]
    return Episodes(
        mu=np.concatenate([p["mu"] for p in parts], 0),
        actions=np.concatenate([p["actions"] for p in parts], 0),
        state=np.concatenate([p["state"] for p in parts], 0),
        hit=np.concatenate([p["hit"] for p in parts], 0),
        names=list(parts[0]["meta"]["state_names"]),
    )


def fit_position_probe(
    roots: Sequence[str], suffix: str, n: int, seed: int
) -> StateProbe:
    mus, sts = [], []
    for r in roots:
        d = episode_arrays(r, latent_suffix=suffix)
        mus.append(d["mu"].reshape(-1, d["mu"].shape[-1]))
        sts.append(d["state"].reshape(-1, d["state"].shape[-1]))
    M, S = np.concatenate(mus), np.concatenate(sts)
    rng = np.random.default_rng(seed)
    if len(M) > n:
        sel = rng.choice(len(M), n, replace=False)
        M, S = M[sel], S[sel]
    return StateProbe().fit(M, S)


def _dream_positions(
    setup: Setup, ep: Episodes, starts: np.ndarray, eps: np.ndarray,
    warmup: int, horizon: int, device: str, seed: int,
    mu_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dream and read the ball position back out. Returns (N, horizon, 2).

    ``mu_override`` replaces the warm-up latents (used by the recolour
    intervention); it is (N, warmup, z) and is spliced into a copy of the real
    latent slab so the action bookkeeping in ``dream`` is unchanged.
    """
    mu = ep.mu[eps]
    if mu_override is not None:
        mu = mu.copy()
        for i in range(len(eps)):
            mu[i, starts[i] : starts[i] + warmup] = mu_override[i]
    z_d = dream(setup.model, mu, ep.actions[eps], starts, warmup=warmup,
                horizon=horizon, temperature=0.0, device=device, seed=seed)
    return setup.probe(z_d)[..., :2]


# ---------------------------------------------------------------- part (a)


def part_a_cold_start(
    setups: Sequence[Setup], data: Dict[str, Dict[str, Episodes]], out: Path,
    device: str, horizon: int, starts_per_ep: int, seed: int,
) -> Dict:
    """Warm up on ONE frame and see how fast the dreamed ball goes.

    A single frame has no velocity in it -- stage one measured ball_vx/vy at
    R^2 < 0 from mu, in v1 and v2 alike. So at K=1 the LSTM's hidden state
    contains exactly what one latent affords: position, and (in v2) colour.
    Anything the dream then knows about SPEED it got from the colour.

    Predicted outcomes, written down before running it: rnn_v2 should show
    dreamed speed rising as true speed rises (r > 0, slope > 0), and
    rnn_v2_nocolor should dream one average speed for every ball (r ~ 0,
    slope ~ 0), because in its latent space the colour is simply not there.
    """
    print("\n(a) cold-start speed inference (K=1: no motion information)")
    res: Dict[str, Dict] = {}
    scatter: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    for setup in setups:
        for split in ("val", "holdout"):
            ep = data[split][setup.suffix]
            E, T = ep.actions.shape
            # Several start times per episode, spread over the episode, so the
            # sample is not dominated by whatever the ball happened to be doing
            # at t=0 and so each mass contributes more than one measurement.
            ts = np.linspace(0, T - horizon - 8, starts_per_ep).astype(int)
            eps = np.repeat(np.arange(E), len(ts))
            starts = np.tile(ts, E)

            pos = _dream_positions(setup, ep, starts, eps, 1, horizon, device, seed)
            sp = dreamed_speed(pos)
            true_sp = ep.true_speed[eps]
            m = ep.mass[eps]

            # The probe floor: the SAME estimator applied to true latents. Any
            # shortfall in slope below this line is the probe's, not the
            # model's -- kNN smoothing shrinks displacements slightly.
            idx = starts[:, None] + 1 + np.arange(horizon)[None, :]
            true_pos = setup.probe(
                np.stack([ep.mu[e, idx[i]] for i, e in enumerate(eps)])
            )[..., :2]
            sp_floor = dreamed_speed(true_pos)

            sl, ic, r = _ols(true_sp, sp)
            sl_f, _, r_f = _ols(true_sp, sp_floor)
            # log-log slope against MASS, directly comparable with part (b)'s
            # number and with the true law's -1; and a scale-free bias, which
            # is the metric that survives the holdout band's narrow mass range
            # (0.85-1.2 spans only +-20% in speed, so a correlation there is
            # mostly measuring probe noise however good the model is).
            log_sl, _, log_r = _ols(np.log(m), np.log(np.maximum(sp, 1e-6)))
            log_sl_f, _, _ = _ols(np.log(m), np.log(np.maximum(sp_floor, 1e-6)))
            key = f"{setup.name}/{split}"
            res[key] = {
                "n": int(len(sp)), "pearson_r": r, "slope": sl, "intercept": ic,
                "log_slope_vs_mass": log_sl, "log_r_vs_mass": log_r,
                "probe_floor_log_slope": log_sl_f,
                "relative_bias": float(sp.mean() / true_sp.mean()),
                "dreamed_speed_mean": float(sp.mean()),
                "dreamed_speed_std": float(sp.std()),
                "true_speed_mean": float(true_sp.mean()),
                "true_speed_std": float(true_sp.std()),
                "probe_floor_slope": sl_f, "probe_floor_r": r_f,
            }
            scatter[(setup.name, split)] = {
                "true": true_sp, "dream": sp, "mass": m, "floor": sp_floor,
            }
            print(f"    {key:28s} n={len(sp):4d}  r={r:+.3f}  slope={sl:+.3f}  "
                  f"dlog(speed)/dlog(m)={log_sl:+.3f} (law -1, floor "
                  f"{log_sl_f:+.3f})  probe floor r={r_f:+.3f}  "
                  f"dreamed speed {sp.mean():.4f} +- {sp.std():.4f} "
                  f"(true {true_sp.mean():.4f})")

    _plot_cold_start(scatter, setups, out / "cold_start_speed.png")

    # ------- dreamed speed vs warm-up length ------------------------------
    print("\n    dreamed speed vs warm-up length K")
    ks = (1, 2, 4, 8)
    vs_k: Dict[str, Dict[int, Dict[str, float]]] = {}
    for setup in setups:
        ep = data["val"][setup.suffix]
        E, T = ep.actions.shape
        ts = np.linspace(0, T - horizon - 12, starts_per_ep).astype(int) + 8
        eps = np.repeat(np.arange(E), len(ts))
        vs_k[setup.name] = {}
        for K in ks:
            starts = np.tile(ts, E) - K
            pos = _dream_positions(setup, ep, starts, eps, K, horizon, device, seed)
            sp = dreamed_speed(pos)
            sl, _, r = _ols(ep.true_speed[eps], sp)
            vs_k[setup.name][K] = {"pearson_r": r, "slope": sl,
                                   "mean": float(sp.mean())}
            print(f"      {setup.name:18s} K={K}  r={r:+.3f}  slope={sl:+.3f}")
    _plot_speed_vs_warmup(vs_k, ks, out / "speed_vs_warmup.png")
    res["vs_warmup"] = vs_k
    return res


def _plot_cold_start(scatter, setups, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(setups), figsize=(5.6 * len(setups), 4.8),
                             sharex=True, sharey=True, squeeze=False)
    lim = None
    for ax, setup in zip(axes[0], setups):
        for split, mk, lab in (("val", "o", "in-distribution"),
                               ("holdout", "^", "held-out mass band")):
            d = scatter[(setup.name, split)]
            ax.scatter(d["true"], d["dream"], s=14, marker=mk, alpha=0.45,
                       label=f"{lab} (n={len(d['true'])})")
        d = scatter[(setup.name, "val")]
        lo = min(d["true"].min(), d["dream"].min()) * 0.9
        hi = max(d["true"].max(), d["dream"].max()) * 1.05
        lim = (min(lim[0], lo), max(lim[1], hi)) if lim else (lo, hi)
        ax.plot([0, 0.06], [0, 0.06], "k--", lw=1, label="identity (the law)")
        sl, ic, r = _ols(d["true"], d["dream"])
        xs = np.linspace(d["true"].min(), d["true"].max(), 2)
        ax.plot(xs, sl * xs + ic, "r-", lw=1.6,
                label=f"fit: slope {sl:+.2f}, r {r:+.2f}")
        ax.set_title(setup.name)
        ax.set_xlabel("true speed = 0.022 / mass")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0][0].set_ylabel("dreamed speed after a ONE-frame warm-up")
    for ax in axes[0]:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    fig.suptitle("cold start: can the model read speed off the colour?")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_speed_vs_warmup(vs_k, ks, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, d in vs_k.items():
        axes[0].plot(ks, [d[k]["pearson_r"] for k in ks], marker="o", label=name)
        axes[1].plot(ks, [d[k]["slope"] for k in ks], marker="o", label=name)
    for ax, t, yl in ((axes[0], "correlation", "pearson r (dreamed vs true speed)"),
                      (axes[1], "slope", "slope of dreamed on true speed")):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("warm-up length K (true frames)")
        ax.set_xticks(ks)
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1].axhline(1.0, color="grey", ls=":", lw=1)
    fig.suptitle("the gap at K=1 is the colour channel; by K>=2 motion suffices")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- part (b)


RECOLOR_MASSES = (0.5, 0.7, 1.0, 1.4, 2.0)


def part_b_recolor(
    setups: Sequence[Setup], val_roots: Sequence[str], out: Path, device: str,
    warmup: int, horizon: int, n_src: int, seed: int, cfg: BoxConfig,
    holdout_band: Tuple[float, float],
) -> Dict:
    """The intervention. Repaint a real ball, re-dream, measure the speed.

    Read the design point carefully, because it is what makes this different
    from part (a): there IS no ground truth here. A ball with mass m0 that we
    have repainted as m1 never existed, and the trajectory it would have taken
    is not the one in the dataset. So the comparison is dream vs LAW: the true
    law says ``speed = 0.022 / m1``, i.e. slope -1 in log-log, and the question
    is whether the dream moves that way when nothing but hue has changed.

    Two warm-up lengths, because they ask different questions:

    K=8   the specified test. Eight frames of motion at the ORIGINAL speed,
          painted the new colour. Colour and motion now CONTRADICT each other,
          and the slope measures how much the model lets colour override the
          evidence of its own eyes.
    K=1   no motion at all, so the only cue is the new colour. This is the clean
          measurement of the causal edge; the K=8 number is the interesting one
          about how the model arbitrates.
    """
    print("\n(b) recolour counterfactual")
    src = [episode_arrays(r) for r in val_roots]
    states = np.concatenate([p["state"] for p in src], 0)
    actions = np.concatenate([p["actions"] for p in src], 0)
    T = actions.shape[1]

    rng = np.random.default_rng(seed)
    E = states.shape[0]
    eps = rng.choice(E, min(n_src, E), replace=False)
    t0 = int(T // 3)                      # mid-episode: ball well away from reset
    assert t0 + warmup + horizon < T

    res: Dict[str, Dict] = {}
    curves: Dict[Tuple[str, int], np.ndarray] = {}
    gif_latents: Dict[str, np.ndarray] = {}
    gif_pos: Dict[str, np.ndarray] = {}

    for K in (warmup, 1):
        # Re-render the warm-up window once per target mass; this is the
        # intervention and it is exact (the renderer is a pure function of
        # position + colour, verified against the dataset's own frames).
        frames = {
            m1: np.stack([
                recolor_frames(states[e, t0 : t0 + K], cfg, mass=m1) for e in eps
            ])
            for m1 in RECOLOR_MASSES
        }
        frames["orig"] = np.stack([
            recolor_frames(states[e, t0 : t0 + K], cfg, mass=None) for e in eps
        ])

        for setup in setups:
            ep = load_episodes(val_roots, setup.suffix)
            starts = np.full(len(eps), t0)
            sp_by_m = []
            for m1 in RECOLOR_MASSES:
                mu_cf = encode_frames(setup.vae, frames[m1], device=device)
                pos = _dream_positions(setup, ep, starts, eps, K, horizon,
                                       device, seed, mu_override=mu_cf)
                sp_by_m.append(dreamed_speed(pos))
                if K == warmup and setup.name == setups[0].name:
                    gif_latents[f"m{m1}"] = dream(
                        setup.model, _spliced(ep.mu[eps], starts, K, mu_cf),
                        ep.actions[eps], starts, warmup=K, horizon=horizon,
                        temperature=0.0, device=device, seed=seed)
                    gif_pos[f"m{m1}"] = pos
            if K == warmup and setup.name == setups[0].name:
                mu_o = encode_frames(setup.vae, frames["orig"], device=device)
                gif_latents["orig"] = dream(
                    setup.model, _spliced(ep.mu[eps], starts, K, mu_o),
                    ep.actions[eps], starts, warmup=K, horizon=horizon,
                    temperature=0.0, device=device, seed=seed)
                gif_pos["orig"] = _dream_positions(
                    setup, ep, starts, eps, K, horizon, device, seed,
                    mu_override=mu_o)

            sp = np.stack(sp_by_m)                          # (n_masses, n_src)
            curves[(setup.name, K)] = sp
            ms = np.asarray(RECOLOR_MASSES, float)
            # log-log slope, pooled over source episodes. The true law is -1.
            xs = np.repeat(np.log(ms), sp.shape[1])
            ys = np.log(np.maximum(sp.ravel(), 1e-6))
            slope, _, r = _ols(xs, ys)
            # Per-source slope too: a pooled slope can be dragged by one episode.
            per_src = [_ols(np.log(ms), np.log(np.maximum(sp[:, j], 1e-6)))[0]
                       for j in range(sp.shape[1])]
            key = f"{setup.name}/K{K}"
            res[key] = {
                "log_slope": slope, "log_r": r,
                "log_slope_per_source_median": float(np.nanmedian(per_src)),
                "log_slope_per_source_iqr": float(
                    np.nanpercentile(per_src, 75) - np.nanpercentile(per_src, 25)),
                "n_sources": int(sp.shape[1]),
                "dreamed_speed_by_mass": {
                    str(m): float(v) for m, v in zip(RECOLOR_MASSES, sp.mean(1))},
                "true_law_by_mass": {
                    str(m): BASE_SPEED / m for m in RECOLOR_MASSES},
                "masses_in_holdout_band": [
                    m for m in RECOLOR_MASSES
                    if holdout_band[0] <= m <= holdout_band[1]],
            }
            print(f"    {key:26s} d log(speed) / d log(m1) = {slope:+.3f} "
                  f"(true law -1.000, r={r:+.3f}); per-source median "
                  f"{np.nanmedian(per_src):+.3f}")

    _plot_recolor(curves, setups, warmup, holdout_band, out / "recolor_speed.png")
    _recolor_gif(setups[0], gif_latents, out / "recolor_counterfactual.gif", device)
    res["color_persistence"] = _color_persistence(
        setups[0], gif_latents, gif_pos, cfg, device,
        out / "recolor_color_persistence.png", out / "recolor_counterfactual.png")
    return res


def _read_ball_mass(frames: np.ndarray, pos: np.ndarray, cfg: BoxConfig) -> np.ndarray:
    """Read the mass back OUT of decoded dream frames, via the colour ramp.

    A dreamed latent is the model's own invention, so there is no guarantee it
    still carries the colour we pasted into the warm-up. This measures whether
    it does: locate the ball with the position probe, take the most ball-like
    pixel in that patch (the reconstruction is blurry, so the average ball pixel
    is a dark blend and systematically reads as the wrong mass), and project it
    onto the ramp with ``worldsim.color_to_mass``. Same trick as
    ``analyze.color_fidelity``; the difference is that this is applied to a
    DREAM rather than to a reconstruction.
    """
    res = frames.shape[-2]
    bg = np.asarray(cfg.bg_color, np.float64)
    N, H = frames.shape[0], frames.shape[1]
    out = np.zeros((N, H))
    r = max(2, int(cfg.ball_radius * res))
    for i in range(N):
        for t in range(H):
            j0, i0 = int(pos[i, t, 0] * res), int((1.0 - pos[i, t, 1]) * res)
            sl = (slice(max(i0 - r, 0), i0 + r + 1), slice(max(j0 - r, 0), j0 + r + 1))
            patch = frames[i, t][sl].reshape(-1, 3).astype(np.float64)
            if not len(patch):
                out[i, t] = np.nan
                continue
            px = patch[np.argmax(((patch - bg) ** 2).sum(-1))]
            out[i, t] = color_to_mass(px, cfg)
    return out


def _color_persistence(setup: Setup, lat, pos, cfg, device, path: Path,
                       sheet_path: Path) -> Dict:
    """Does the repainted hue survive the dream, or does the model repaint it?"""
    plt = _plt()
    keys = [f"m{m}" for m in RECOLOR_MASSES]
    if not set(keys) <= set(lat):
        return {}
    res: Dict[str, Dict[str, float]] = {}
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    decoded_cache = {}
    for m1, k in zip(RECOLOR_MASSES, keys):
        dec = decode_latents(setup.vae, lat[k], device=device)
        decoded_cache[k] = dec
        mm = _read_ball_mass(dec, pos[k], cfg)
        res[str(m1)] = {
            "decoded_mass_step0": float(np.nanmedian(mm[:, 0])),
            "decoded_mass_step_last": float(np.nanmedian(mm[:, -1])),
            "target_mass": m1,
        }
        ax.plot(np.arange(mm.shape[1]), np.nanmedian(mm, 0), marker=".",
                label=f"repainted as m1={m1}")
        ax.axhline(m1, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("dream step")
    ax.set_ylabel("mass read back out of the dreamed ball's colour")
    ax.set_yscale("log")
    ax.set_title("does the repainted colour persist through the dream?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

    # A labelled contact sheet -- the GIF has no captions, and without them you
    # cannot tell which column is which.
    cols = ["orig"] + keys
    n_t = 6
    ts = np.linspace(0, lat[keys[0]].shape[1] - 1, n_t).astype(int)
    fig, axes = plt.subplots(len(cols), n_t, figsize=(1.5 * n_t, 1.5 * len(cols)))
    for i, k in enumerate(cols):
        dec = decoded_cache.get(k)
        if dec is None:
            dec = decode_latents(setup.vae, lat[k], device=device)
        for j, t in enumerate(ts):
            axes[i, j].imshow(dec[0, t])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f"step {t}", fontsize=8)
        axes[i, 0].set_ylabel("original" if k == "orig" else k.replace("m", "m1="),
                              fontsize=8)
    fig.suptitle("one source episode, repainted and re-dreamt", fontsize=10)
    fig.tight_layout()
    fig.savefig(sheet_path, dpi=130)
    plt.close(fig)
    return res


def _spliced(mu: np.ndarray, starts: np.ndarray, K: int, mu_cf: np.ndarray):
    mu = mu.copy()
    for i in range(len(starts)):
        mu[i, starts[i] : starts[i] + K] = mu_cf[i]
    return mu


def _plot_recolor(curves, setups, warmup, band, path: Path) -> None:
    plt = _plt()
    ms = np.asarray(RECOLOR_MASSES, float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, K in zip(axes, (warmup, 1)):
        for setup, col in zip(setups, ("tab:blue", "tab:orange", "tab:green")):
            sp = curves[(setup.name, K)]
            for j in range(min(sp.shape[1], 8)):
                ax.plot(ms, sp[:, j], color=col, alpha=0.22, lw=1)
            # geometric mean, to match the log-space fit that produced the slope
            ax.plot(ms, np.exp(np.log(np.maximum(sp, 1e-6)).mean(1)),
                    color=col, marker="o", lw=2, label=setup.name)
        ax.plot(ms, BASE_SPEED / ms, "k--", lw=1.8, label="true law 0.022 / m")
        ax.axvspan(band[0], band[1], color="red", alpha=0.10,
                   label=f"held-out band {band}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(ms)
        ax.set_xticklabels([str(m) for m in RECOLOR_MASSES])
        ax.set_xlabel("mass the ball was REPAINTED as, m1")
        ax.set_title(f"warm-up K={K}"
                     + (" (colour vs motion)" if K != 1 else " (colour only)"))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("dreamed speed")
    fig.suptitle("recolour intervention: repaint the ball, re-dream, measure the speed")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _recolor_gif(setup: Setup, lat: Dict[str, np.ndarray], path: Path, device: str) -> None:
    if not {"orig", "m0.5", "m2.0"} <= set(lat):
        return
    n = min(3, lat["orig"].shape[0])
    cols = [decode_latents(setup.vae, lat[k][:n], device=device)
            for k in ("orig", "m0.5", "m2.0")]
    strip = np.concatenate(
        [side_by_side([c[e] for c in cols]) for e in range(n)], axis=1)
    save_gif(strip, path, fps=10, scale=3)


# ---------------------------------------------------------------- part (c)


@torch.no_grad()
def _hidden(model: MDNRNN, mu: np.ndarray, actions: np.ndarray, device: str):
    E, T = actions.shape
    z = torch.from_numpy(mu[:, :T].astype(np.float32)).to(device)
    a = torch.eye(model.cfg.n_actions, device=device)[
        torch.from_numpy(actions).long().to(device)]
    parts, _ = model(z, a)
    return mu[:, :T], parts["h"].cpu().numpy()


def part_c_speed_in_h(
    setups: Sequence[Setup], data: Dict[str, Dict[str, Episodes]], out: Path,
    device: str, n_samples: int, seed: int,
) -> Dict:
    """Is speed (and mass) linearly available in the recurrent state?

    Part (c) of ``eval_rnn`` asks where VELOCITY lives and finds it in h. Here
    the question is narrower and newer: SPEED is a function of colour, so a
    model that read it off the colour should have it in h from the first step,
    while a model that integrated motion should have it too -- both routes end
    in h. The discriminating number is not the pooled R^2 but the contrast with
    ``log_mass``: mass is *only* available through colour, so log_mass in h is a
    direct readout of whether the colour survived into the recurrent state.
    """
    print("\n(c) speed and mass in the recurrent state")
    res: Dict[str, Dict] = {}
    for setup in setups:
        ep = data["val"][setup.suffix]
        Z, H = _hidden(setup.model, ep.mu, ep.actions, device)
        E, T = H.shape[0], H.shape[1]
        m = np.repeat(ep.mass, T)
        targets = np.stack([BASE_SPEED / m, np.log(m)], 1)      # speed, log_mass
        names = ["speed", "log_mass"]
        g = np.repeat(np.arange(E), T)
        keep = np.where(np.tile(np.arange(T), E) >= 4)[0]
        rng = np.random.default_rng(seed)
        if len(keep) > n_samples:
            keep = rng.choice(keep, n_samples, replace=False)

        Zf, Hf = Z.reshape(E * T, -1)[keep], H.reshape(E * T, -1)[keep]
        Y, G, M = targets[keep], g[keep], m[keep]
        feats = {"z": Zf, "h": Hf, "z+h": np.concatenate([Zf, Hf], 1)}

        per_feat = {}
        for fname, X in feats.items():
            per_feat[fname] = probe_suite(X, Y, names, group_ids=G, seed=seed,
                                          which=("linear",))
            # Degree 2 on the raw 256-unit h would be 33k features; see
            # _reduce_for_poly. Same seed and groups, so the same split.
            per_feat[fname]["poly2"] = probe_suite(
                _reduce_for_poly(X, seed), Y, names, group_ids=G, seed=seed,
                which=("poly2",))["poly2"]
        # By-tercile breakdown: fit once on the train episodes, then score the
        # test episodes separately inside each mass tercile. R^2 inside a
        # tercile is against that tercile's OWN (much smaller) variance, so it
        # is a harsh number by construction -- the RMSE column is the one to
        # compare across terciles.
        tr, te = make_split(len(Y), seed=seed, group_ids=G)
        by_ter = _tercile_table(feats["h"], Y, names, tr, te, M)

        res[setup.name] = {"pooled": per_feat, "by_mass_tercile_from_h": by_ter}
        print(f"    {setup.name}")
        hdr = "      " + f"{'feature':6s} {'probe':7s}" + "".join(f"{n:>11s}" for n in names)
        print(hdr)
        for fname in feats:
            for pr in ("linear", "poly2"):
                print("      " + f"{fname:6s} {pr:7s}"
                      + "".join(f"{per_feat[fname][pr][n]:11.3f}" for n in names))
        for ter, row in by_ter.items():
            print(f"      tercile {ter:7s} m {row['mass_lo']:.2f}-{row['mass_hi']:.2f}  "
                  f"speed rmse {row['speed_rmse']:.5f} (R2 {row['speed_r2']:+.3f})  "
                  f"log_mass rmse {row['log_mass_rmse']:.3f} "
                  f"(R2 {row['log_mass_r2']:+.3f})")

    (out / "speed_in_h.json").write_text(json.dumps(res, indent=2, default=float))
    _plot_speed_in_h(res, out / "speed_in_h.png")
    return res


def _tercile_table(X, Y, names, tr, te, mass) -> Dict[str, Dict[str, float]]:
    from sklearn.preprocessing import StandardScaler

    sx = StandardScaler().fit(X[tr])
    Xtr, Xte = sx.transform(X[tr]), sx.transform(X[te])
    A = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    W = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Y[tr])
    P = np.concatenate([Xte, np.ones((len(Xte), 1))], 1) @ W
    mte = mass[te]
    out = {}
    for ter, sel in _terciles(mte).items():
        if sel.sum() < 20:
            continue
        row = {"n": int(sel.sum()), "mass_lo": float(mte[sel].min()),
               "mass_hi": float(mte[sel].max())}
        for j, nm in enumerate(names):
            row[f"{nm}_rmse"] = float(np.sqrt(((Y[te][sel, j] - P[sel, j]) ** 2).mean()))
            row[f"{nm}_r2"] = _r2(Y[te][sel, j], P[sel, j])
        out[ter] = row
    return out


def _plot_speed_in_h(res, path: Path) -> None:
    plt = _plt()
    names = ["speed", "log_mass"]
    feats = ["z", "h", "z+h"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    xs = np.arange(len(feats))
    width = 0.38
    for ax, nm in zip(axes, names):
        for i, (mname, d) in enumerate(res.items()):
            vals = [max(d["pooled"][f]["poly2"][nm], -0.1) for f in feats]
            ax.bar(xs + (i - 0.5) * width, vals, width, label=mname)
        ax.set_xticks(xs)
        ax.set_xticklabels(feats)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{nm} (poly-2 probe)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("held-out R^2")
    axes[0].legend(fontsize=8)
    fig.suptitle("is the colour -> speed information in the model's state?")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- part (e)


def part_e_english(
    setups: Sequence[Setup], data: Dict[str, Dict[str, Episodes]], out: Path,
    device: str, seed: int, pre: int = 6, post: int = 8,
) -> Dict:
    """The sparse effect: does the dream deflect light balls more?

    ``dvx = 0.35 * paddle_vx / m``, so the change in vx across a contact should
    scale with 1/m -- for a given paddle motion. That last clause is the problem
    and the reason this is labelled best-effort: paddle_vx varies from contact
    to contact and is not controlled for, so even the TRUE dvx correlates with
    1/m only weakly. Contacts are also ~1% of transitions. Both scatters are
    plotted, truth and dream, so the dream can be judged against how much signal
    is there to find rather than against the ideal law.

    Geometry: warm up on ``pre`` true frames ending 5 steps before the contact,
    then dream across it, so the dream contains the approach AND the rebound and
    a velocity can be finite-differenced on both sides.
    """
    print("\n(e) english (paddle deflection) by mass -- best effort, sparse")
    res: Dict[str, Dict] = {}
    lead, horizon = 5, 14
    for setup in setups:
        ep_all = {s: data[s][setup.suffix] for s in ("val", "holdout")}
        rows_d, rows_t, rows_m, rows_split = [], [], [], []
        for split, ep in ep_all.items():
            E, T = ep.actions.shape
            ee, tt = np.where(ep.hit > 0.5)
            cases = [(int(e), int(t)) for e, t in zip(ee, tt)
                     if t - lead - pre >= 0 and t + post + 2 < T]
            if not cases:
                continue
            eps = np.array([c[0] for c in cases])
            tc = np.array([c[1] for c in cases])
            starts = tc - lead - pre
            pos = _dream_positions(setup, ep, starts, eps, pre, horizon, device, seed)
            # dream step k <-> true index starts + pre + k = tc - lead + k, so
            # the contact sits at k = lead = 5.
            vx_before = (pos[:, lead - 1, 0] - pos[:, 0, 0]) / max(lead - 1, 1)
            vx_after = (pos[:, horizon - 1, 0] - pos[:, lead + 1, 0]) / (horizon - lead - 2)
            ix = ep.names.index("ball_vx")
            true_before = ep.state[eps, tc - 1, ix]
            true_after = ep.state[eps, np.minimum(tc + 2, T), ix]
            rows_d.append(vx_after - vx_before)
            rows_t.append(true_after - true_before)
            rows_m.append(ep.mass[eps])
            rows_split.append(np.full(len(eps), split == "holdout"))
        if not rows_d:
            res[setup.name] = {"n_contacts": 0}
            continue
        dv_d = np.concatenate(rows_d)
        dv_t = np.concatenate(rows_t)
        m = np.concatenate(rows_m)
        inv = 1.0 / m
        _, _, r_d = _ols(inv, np.abs(dv_d))
        _, _, r_t = _ols(inv, np.abs(dv_t))
        _, _, r_dt = _ols(dv_t, dv_d)
        res[setup.name] = {
            "n_contacts": int(len(dv_d)),
            "pearson_r_true_dvx_vs_inv_mass": r_t,
            "pearson_r_dream_dvx_vs_inv_mass": r_d,
            "pearson_r_dream_vs_true_dvx": r_dt,
            "enough_contacts": bool(len(dv_d) >= 40),
        }
        print(f"    {setup.name:18s} n={len(dv_d)}  r(|dvx_true|, 1/m)={r_t:+.3f}  "
              f"r(|dvx_dream|, 1/m)={r_d:+.3f}  r(dream, true)={r_dt:+.3f}")
        if setup.name == setups[0].name:
            _plot_english(inv, dv_t, dv_d, len(dv_d), out / "english_by_mass.png")
    return res


def _plot_english(inv_m, dv_t, dv_d, n, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.scatter(inv_m, np.abs(dv_t), s=22, alpha=0.6, label="true |$\\Delta v_x$|")
    ax.scatter(inv_m, np.abs(dv_d), s=22, alpha=0.6, marker="^",
               label="dreamed |$\\Delta v_x$| (via probe)")
    for y, c in ((np.abs(dv_t), "tab:blue"), (np.abs(dv_d), "tab:orange")):
        sl, ic, _ = _ols(inv_m, y)
        xs = np.linspace(inv_m.min(), inv_m.max(), 2)
        ax.plot(xs, sl * xs + ic, color=c, lw=1.5)
    ax.set_xlabel("1 / mass")
    ax.set_ylabel("|change in ball vx across the contact|")
    ax.set_title(f"english by mass, {n} real paddle contacts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------- parts (d) and (f)


def part_df_horizon(
    setups: Sequence[Setup], data: Dict[str, Dict[str, Episodes]], out: Path,
    device: str, warmup: int, horizon: int, seed: int,
) -> Dict:
    """Useful dream horizon: in-distribution vs held-out band, and by mass.

    Two questions in one dream. (d) asks whether the held-out colours are dreamt
    as well as the trained ones -- a lookup-table model has nothing to look up
    there. (f) asks whether the pooled horizon was hiding a mass dependence:
    a light ball covers more ground per frame, so it should exhaust a fixed
    error budget in fewer FRAMES even if the model is equally good at it. The
    ball-diameters-travelled column is the distance-normalised version.
    """
    print("\n(d)/(f) useful dream horizon: in-distribution vs held-out, by mass")
    res: Dict[str, Dict] = {}
    for setup in setups:
        for split in ("val", "holdout"):
            ep = data[split][setup.suffix]
            E = ep.actions.shape[0]
            eps = np.arange(E)
            starts = np.zeros(E, dtype=int)
            pos = _dream_positions(setup, ep, starts, eps, warmup, horizon,
                                   device, seed)
            idx = warmup + np.arange(horizon)
            truth = ep.state[:E][:, idx, :2]
            err = np.linalg.norm(pos - truth, axis=-1)          # (E, H)
            pooled = float(np.mean([
                np.argmax(err[i] > BALL_RADIUS) if (err[i] > BALL_RADIUS).any()
                else horizon for i in range(E)]))
            key = f"{setup.name}/{split}"
            res[key] = {
                "useful_dream_horizon_mean": pooled,
                "by_mass_tercile": horizon_by_mass(err, ep.mass, horizon),
                "n_episodes": int(E),
            }
            print(f"    {key:28s} horizon {pooled:5.1f} frames")
            for ter, row in res[key]["by_mass_tercile"].items():
                print(f"      {ter:7s} m {row['mass_lo']:.2f}-{row['mass_hi']:.2f}  "
                      f"{row['useful_dream_horizon']:5.1f} frames  "
                      f"{row['horizon_ball_diameters']:5.2f} ball diameters")
    _plot_horizon(res, out / "horizon_by_mass.png")
    return res


def _plot_horizon(res, path: Path) -> None:
    plt = _plt()
    ters = ["light", "medium", "heavy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.arange(len(ters))
    keys = list(res)
    width = 0.8 / max(len(keys), 1)
    for i, k in enumerate(keys):
        t = res[k]["by_mass_tercile"]
        axes[0].bar(xs + (i - (len(keys) - 1) / 2) * width,
                    [t.get(c, {}).get("useful_dream_horizon", 0) for c in ters],
                    width, label=k)
        axes[1].bar(xs + (i - (len(keys) - 1) / 2) * width,
                    [t.get(c, {}).get("horizon_ball_diameters", 0) for c in ters],
                    width, label=k)
    for ax, t in ((axes[0], "useful dream horizon (frames)"),
                  (axes[1], "the same, in ball diameters travelled")):
        ax.set_xticks(xs)
        ax.set_xticklabels(ters)
        ax.set_title(t)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7)
    fig.suptitle("does the dream degrade faster for fast (light) balls?")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="runs/rnn_v2/rnn.pt")
    p.add_argument("--nocolor-ckpt", default="runs/rnn_v2_nocolor/rnn.pt")
    p.add_argument("--vae", default="runs/vae_v2/vae.pt")
    p.add_argument("--nocolor-vae", default="runs/vae_b1/vae.pt")
    p.add_argument("--nocolor-suffix", default="v1vae")
    p.add_argument("--val", nargs="+", default=["data/v2/val", "data/v2/val_mix"])
    p.add_argument("--holdout", nargs="+", default=["data/v2/holdout"])
    p.add_argument("--probe-data", nargs="+",
                   default=["data/v2/probe", "data/v2/val_mix"])
    p.add_argument("--out", default="runs/rnn_v2/causal")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--cold-horizon", type=int, default=24)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--starts-per-ep", type=int, default=5)
    p.add_argument("--n-recolor-src", type=int, default=12)
    p.add_argument("--probe-samples", type=int, default=12000)
    p.add_argument("--probe-samples-h", type=int, default=6000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = pick_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    meta = json.loads((Path(a.val[0]) / "meta.json").read_text())
    cfg = BoxConfig(**meta["config"])
    band = tuple(meta.get("mass_holdout") or (0.85, 1.2))

    setups: List[Setup] = []
    for name, ck, vae_path, sfx in (
        ("rnn_v2", a.ckpt, a.vae, ""),
        ("rnn_v2_nocolor", a.nocolor_ckpt, a.nocolor_vae, a.nocolor_suffix),
    ):
        if not Path(ck).exists():
            print(f"skipping {name}: {ck} not found")
            continue
        model, _ = load_rnn(ck, device)
        vae, _, _ = load_ckpt(vae_path, device)
        probe = fit_position_probe(a.probe_data, sfx, a.probe_samples, a.seed)
        setups.append(Setup(name, model, sfx, vae, probe))
        print(f"{name:16s} {ck}  latents=mu{'_' + sfx if sfx else ''}.npy  vae={vae_path}")

    suffixes = sorted({s.suffix for s in setups})
    data = {
        "val": {sfx: load_episodes(a.val, sfx) for sfx in suffixes},
        "holdout": {sfx: load_episodes(a.holdout, sfx) for sfx in suffixes},
    }
    print(f"val {data['val'][suffixes[0]].mu.shape[0]} episodes, "
          f"holdout {data['holdout'][suffixes[0]].mu.shape[0]} episodes "
          f"(masses only in {band})")

    report: Dict[str, object] = {
        "ckpts": {s.name: s.suffix for s in setups},
        "holdout_band": list(band),
        "warmup": a.warmup, "cold_horizon": a.cold_horizon, "horizon": a.horizon,
    }
    report["a_cold_start"] = part_a_cold_start(
        setups, data, out, device, a.cold_horizon, a.starts_per_ep, a.seed)
    report["b_recolor"] = part_b_recolor(
        setups, a.val, out, device, a.warmup, a.cold_horizon,
        a.n_recolor_src, a.seed, cfg, band)
    report["c_speed_in_h"] = part_c_speed_in_h(
        setups, data, out, device, a.probe_samples_h, a.seed)
    report["df_horizon"] = part_df_horizon(
        setups, data, out, device, a.warmup, a.horizon, a.seed)
    report["e_english"] = part_e_english(setups, data, out, device, a.seed)

    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    _summary(report, out, band)


def _summary(r: Dict, out: Path, band) -> None:
    def g(d, *ks, default=float("nan")):
        for k in ks:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d if not isinstance(d, dict) else default

    a = r["a_cold_start"]
    b = r["b_recolor"]
    c = r["c_speed_in_h"]
    h = r["df_horizon"]
    e = r["e_english"]
    main, ctl = "rnn_v2", "rnn_v2_nocolor"
    print("\n" + "=" * 74)
    print("SUMMARY -- did M learn that colour causes speed?")
    print("=" * 74)

    print(
        "\n1. COLD START (one frame of warm-up, so no motion information exists).\n"
        f"   rnn_v2         : r = {g(a, main + '/val', 'pearson_r'):+.3f}, "
        f"slope = {g(a, main + '/val', 'slope'):+.3f}\n"
        f"   rnn_v2_nocolor : r = {g(a, ctl + '/val', 'pearson_r'):+.3f}, "
        f"slope = {g(a, ctl + '/val', 'slope'):+.3f}\n"
        f"   The measuring probe's own ceiling on this slope is "
        f"{g(a, main + '/val', 'probe_floor_slope'):+.3f}.\n"
        f"   As a log-log slope against mass (comparable with 2 below, law -1):\n"
        f"     rnn_v2 {g(a, main + '/val', 'log_slope_vs_mass'):+.3f}  "
        f"nocolor {g(a, ctl + '/val', 'log_slope_vs_mass'):+.3f}  "
        f"(probe floor {g(a, main + '/val', 'probe_floor_log_slope'):+.3f})\n"
        "   The control sees the same frames through an encoder with no colour in\n"
        "   it, so whatever separates the two rows came from the colour."
    )
    print(
        "\n2. RECOLOUR INTERVENTION (repaint a real ball as mass m1, re-dream).\n"
        "   d log(dreamed speed) / d log(m1); the true law is exactly -1.\n"
        f"   rnn_v2         : {g(b, main + '/K8', 'log_slope'):+.3f} with 8 frames of "
        f"contradicting motion, {g(b, main + '/K1', 'log_slope'):+.3f} with colour alone\n"
        f"   rnn_v2_nocolor : {g(b, ctl + '/K8', 'log_slope'):+.3f} / "
        f"{g(b, ctl + '/K1', 'log_slope'):+.3f}\n"
        "   There is no ground-truth trajectory for a repainted ball -- the\n"
        "   comparison is dream against law, which is what an interventional test is."
    )
    print(
        "\n3. WHERE THE INFORMATION SITS (poly-2 probe, held-out episodes).\n"
        f"   rnn_v2  speed: z {g(c, main, 'pooled', 'z', 'poly2', 'speed'):+.3f}  "
        f"h {g(c, main, 'pooled', 'h', 'poly2', 'speed'):+.3f}   "
        f"log_mass: z {g(c, main, 'pooled', 'z', 'poly2', 'log_mass'):+.3f}  "
        f"h {g(c, main, 'pooled', 'h', 'poly2', 'log_mass'):+.3f}\n"
        f"   nocolor speed: z {g(c, ctl, 'pooled', 'z', 'poly2', 'speed'):+.3f}  "
        f"h {g(c, ctl, 'pooled', 'h', 'poly2', 'speed'):+.3f}   "
        f"log_mass: z {g(c, ctl, 'pooled', 'z', 'poly2', 'log_mass'):+.3f}  "
        f"h {g(c, ctl, 'pooled', 'h', 'poly2', 'log_mass'):+.3f}\n"
        "   log_mass is the discriminating row: mass is knowable ONLY through\n"
        "   colour, so a high R^2 in h means the colour survived into the state."
    )
    print(
        f"\n4. INTERPOLATION into the held-out band {tuple(band)} -- colours whose\n"
        "   dynamics M was never shown.\n"
        f"   cold-start r : in-distribution {g(a, main + '/val', 'pearson_r'):+.3f}  "
        f"held-out {g(a, main + '/holdout', 'pearson_r'):+.3f}\n"
        "   -- but the band spans only +-20% in speed, so a correlation inside it\n"
        "   is mostly probe noise (the probe's OWN r there is "
        f"{g(a, main + '/holdout', 'probe_floor_r'):+.3f}). The scale-free number\n"
        f"   is the bias: dreamed / true speed = "
        f"{g(a, main + '/holdout', 'relative_bias'):.3f} in the band vs "
        f"{g(a, main + '/val', 'relative_bias'):.3f} outside it, and the recolour\n"
        "   test at m1 = 1.0 (inside the band) is the powerful within-episode version.\n"
        f"   useful dream horizon : in-distribution "
        f"{g(h, main + '/val', 'useful_dream_horizon_mean'):.1f} frames  "
        f"held-out {g(h, main + '/holdout', 'useful_dream_horizon_mean'):.1f}\n"
        "   A model that memorised a lookup table of colours has nothing to look\n"
        "   up inside the gap; a model that learned the monotone law does."
    )
    ter = h.get(main + "/val", {}).get("by_mass_tercile", {})
    if ter:
        print("\n5. HORIZON BY MASS (in-distribution).")
        for k in ("light", "medium", "heavy"):
            t = ter.get(k)
            if t:
                print(f"   {k:7s} m {t['mass_lo']:.2f}-{t['mass_hi']:.2f}: "
                      f"{t['useful_dream_horizon']:5.1f} frames = "
                      f"{t['horizon_ball_diameters']:.2f} ball diameters travelled")
        print("   If the frame counts differ but the diameters agree, the model is\n"
              "   equally good at every mass and light balls simply run out the\n"
              "   error budget sooner because they cover more ground per frame.")
    em = e.get(main, {})
    print(
        f"\n6. ENGLISH (sparse, best effort): {em.get('n_contacts', 0)} real contacts. "
        f"r(|true dvx|, 1/m) = {em.get('pearson_r_true_dvx_vs_inv_mass', float('nan')):+.3f}, "
        f"r(|dreamed dvx|, 1/m) = "
        f"{em.get('pearson_r_dream_dvx_vs_inv_mass', float('nan')):+.3f}.\n"
        "   paddle_vx is not controlled for, so even the TRUTH correlates only\n"
        "   weakly with 1/m -- judge the dream against the truth row, not the law."
    )
    print(f"\nartifacts in {out}")


if __name__ == "__main__":
    main()

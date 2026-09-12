"""v3.1 stage three, step zero: is the dream a usable training environment?

    python -m wm.eval_dream_alive --out runs/rnn_v31_dream_alive

v3 trained six controllers in a dream and only afterwards discovered (README_C3
§8) that the dreamed ball was present on 36 % of frames, which is almost
certainly why no controller ever committed during an occlusion. v3.1's stage two
found the same defect in a much sharper form: in a tau = 0 dream the ball, once
hidden, essentially never comes back (5 of 99 runs; README_V31 §4.5), and the
dreamed ``ball_y`` flattens at y ~ 0.40, the middle of the band.

So before a single generation of CMA-ES is spent, this script asks the question
the controller's objective actually depends on:

    over 150 dream steps, on what fraction of frames does the DECODED dream
    contain a ball at all, and on what fraction is that ball BELOW the band --
    i.e. in the only region of the box where the paddle can do anything?

and compares every answer to the SAME measurement on the real continuation of
the same 64 starts. A dream that never brings a ball below the band is not a
world for this task: a controller trained in it is being fitted to whatever the
reward head says about a latent with no reachable ball in it, and every
downstream number has to be read that way. (v3.1's answer, measured: none of
the four models does, at any temperature.)

The quantities, and why each one is here
----------------------------------------
``frac_ball_present``
    a decoded frame contains a ball when more than half a ball's worth of
    ball-coloured pixel mass lies strictly outside the band. Reused from
    ``analyze_v3.ball_mass``, the same estimator stage one used to show the VAE
    does *not* hallucinate balls -- so a ball detected here is a ball the
    dynamics model predicted, not a decoder artefact.
``frac_ball_below_band``
    of that mass, how much is below the band's bottom edge. This is the strict
    version of the question: a ball dreamed *above* the band is scenery as far
    as the paddle is concerned. On real v3.1 frames the ball is below the band
    on only a few per cent of frames, so the dream does not have far to go --
    which is exactly why the comparison has to be against the real fraction and
    not against 100 %.
``arrivals_per_run`` and ``mean_y_sd``
    how many SEPARATE times a ball came below the band in 150 frames, and how
    much the decoded ball's height varied. These two exist because the first
    two are not enough: a dream that drops one ball to the band's lower edge
    and parks it there scores well on both fractions above and is not a world.
    The world arrives 0.80 times per 150 frames with a height sd of 0.164, and
    those are the two numbers every dream is read against.
    ``arrivals_per_run`` is what the training temperature is chosen on.
``reward_head_r2`` (teacher-forced val)
    M's dense reward head against the true ``1 - |ball_x - paddle_x|``, on real
    latents. If this is poor, the objective is broken before the dream is.
``reward_vs_decoded_gap`` (the dream)
    the correlation, over dream frames where a ball was decoded, between the
    reward head's output and ``-|decoded ball_x - decoded paddle_x|``. This is
    the only measurement that asks whether the number CMA-ES maximises means,
    inside the dream, what it means in the real world. A dream can be visually
    dead and still train a controller if the reward head is honest about the
    latents it is fed; it can also be visually alive and score nothing, if the
    head has drifted off the manifold the dream walks on.

The actions
-----------
There is no controller yet, so the dream has to be driven by something. It is
driven by a sticky-random policy (mean hold 8 frames) with a fixed seed -- the
same policy that collected ``data/v31/train``, so the action sequence is in
distribution and the paddle actually moves, which the reward correlation needs.
Every model and every temperature gets the identical action stream and the
identical 64 starts, so the rows are paired.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from worldsim.bouncing_box import BoxConfig

from .analyze import load_ckpt
from .analyze_v3 import ball_mass
from .dream_env import DreamEnv, load_start_pool
from .rnn import load_rnn
from .seq_data import episode_arrays

N_ACTIONS = 3


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# --------------------------------------------------------- the ball detector


def detect_ball(imgs: np.ndarray, cfg: BoxConfig,
                threshold: float = 0.5) -> Dict[str, np.ndarray]:
    """"Is there a ball in this frame, and if so where" -- from pixels only.

    ``imgs`` is (N, H, W, 3) float in [0, 1], real or decoded. Returns per-frame
    arrays:

    ``mass``        ball-coloured pixel mass strictly outside the band, in units
                    of one ball (a fully visible ball scores ~1).
    ``mass_below``  the part of it below the band's bottom edge.
    ``present``     ``mass > threshold``.
    ``below``       ``mass_below > threshold`` -- operationally "the decoded
                    ball's centre is below the band", because a ball centred at
                    the band's bottom edge has exactly half its area below it.
    ``x``, ``y``    mass-weighted centroid in world coordinates, NaN when no
                    ball is present.

    Rows whose pixel extent *touches* the band are dropped from every sum rather
    than counted, exactly as ``analyze_v3.outside_band_mass`` does: the band's
    edge is antialiased, so those rows are a colour blend and would score a
    small positive mass on frames containing no ball at all -- including the
    ball-free frames that are the whole baseline here.
    """
    res = imgs.shape[1]
    lo, hi = cfg.occluder_y
    px = 1.0 / res
    ys = 1.0 - (np.arange(res) + 0.5) / res
    xs = (np.arange(imgs.shape[2]) + 0.5) / imgs.shape[2]
    above = ys - px / 2 > hi
    below = ys + px / 2 < lo
    outside = above | below
    one_ball = np.pi * (cfg.ball_radius * res) ** 2

    m = ball_mass(imgs, cfg)                                   # (N, H, W)
    m_out = m[:, outside, :]
    total = m_out.sum(axis=(1, 2)) / one_ball
    total_below = m[:, below, :].sum(axis=(1, 2)) / one_ball

    w = m_out.sum(axis=(1, 2))
    safe = np.where(w > 1e-9, w, 1.0)
    cy = (m_out.sum(2) * ys[outside][None, :]).sum(1) / safe
    cx = (m_out.sum(1) * xs[None, :]).sum(1) / safe
    present = total > threshold
    return {
        "mass": total,
        "mass_below": total_below,
        "present": present,
        "below": total_below > threshold,
        "x": np.where(present, cx, np.nan),
        "y": np.where(present, cy, np.nan),
    }


def detect_paddle(imgs: np.ndarray, cfg: BoxConfig) -> np.ndarray:
    """The decoded paddle's centre x, from paddle-coloured pixel mass.

    The paddle is flush on the floor (centre y = 0.0225) and v3.1's band starts
    at 0.13, so it is never occluded and never has to be remembered -- which is
    what makes it usable as the reference point for the reward head's
    ``|ball_x - paddle_x|``. Only the rows below the band are searched, so a
    stray ball-coloured pixel up in the band cannot pull the estimate.
    """
    res = imgs.shape[1]
    lo, _ = cfg.occluder_y
    ys = 1.0 - (np.arange(res) + 0.5) / res
    rows = ys < lo
    xs = (np.arange(imgs.shape[2]) + 0.5) / imgs.shape[2]
    m = ball_mass(imgs, cfg, color=cfg.paddle_color)[:, rows, :]
    w = m.sum(axis=(1, 2))
    safe = np.where(w > 1e-9, w, 1.0)
    cx = (m.sum(1) * xs[None, :]).sum(1) / safe
    return np.where(w > 1e-9, cx, np.nan)


# -------------------------------------------------------------- real frames


def real_fractions(root: str, cfg: BoxConfig, n_frames: int = 4096,
                   seed: int = 0, chunk: int = 256) -> Dict[str, float]:
    """The same detector on real frames: the target the dream is compared to.

    Two numbers come out of this and both are needed. ``frac_ball_present`` is
    partly a check on the detector itself: it must land between the dataset's
    own ``frac_frames_visible`` (24.9 % on ``train_mix``, frames where the ball
    is *entirely* out of the band) and ``visible + partial`` (57 %), because the
    threshold is half a ball and a third of the frames have part of one
    sticking out. It comes out at ~38 %, i.e. essentially the fraction of frames
    with ``ball_visible > 0.5``, which is the right answer.
    ``frac_ball_below_band`` is the harder reference: the fraction of real
    frames on which the ball is in the strip where the paddle can act.
    """
    frames = np.load(Path(root) / "frames.npy", mmap_mode="r")
    flat = frames.reshape(-1, *frames.shape[-3:])
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(flat), size=min(n_frames, len(flat)),
                             replace=False))
    pres, bel, mass = [], [], []
    for i in range(0, len(idx), chunk):
        x = np.asarray(flat[idx[i:i + chunk]], np.float32) / 255.0
        d = detect_ball(x, cfg)
        pres.append(d["present"])
        bel.append(d["below"])
        mass.append(d["mass"])
    pres = np.concatenate(pres)
    bel = np.concatenate(bel)
    meta = json.loads((Path(root) / "meta.json").read_text())
    return {
        "root": root,
        "n_frames": int(len(idx)),
        "frac_ball_present": float(pres.mean()),
        "frac_ball_below_band": float(bel.mean()),
        "mean_mass": float(np.concatenate(mass).mean()),
        "meta_frac_frames_visible":
            meta.get("occlusion", {}).get("frac_frames_visible"),
    }


def presence_stats(present: np.ndarray, below: np.ndarray,
                   ys: np.ndarray) -> Dict:
    """The presence summary, computed identically for a dream and for the world.

    ``present``/``below`` are (B, steps) booleans and ``ys`` the (B, steps)
    decoded ball height with NaN where no ball was found. Four statistics, and
    the last two exist because the first two turned out not to be enough:

    ``frac_reemerged``  of the runs that began with the ball hidden, how many
                        ever showed one below the band. "Did the ball come
                        back at all."
    ``arrivals_per_run``  how many SEPARATE times it came below the band --
                        rising edges of ``below``. This is the statistic that
                        separates a live dream from a dream that has parked a
                        ball at the band's lower edge and left it there: both
                        score well on ``frac_below``, the parked one arrives
                        once and the world arrives about three times in 150
                        frames.
    ``mean_y_sd``       the mean over runs of the decoded ball's height
                        standard deviation. A collapsed dream is flat; the
                        world traverses the box.
    """
    started_hidden = ~present[:, 0]
    reemerged = (below & started_hidden[:, None]).any(1)
    edges = below[:, 1:] & ~below[:, :-1]
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(np.where(present, ys, np.nan), axis=1)
    return {
        "frac_ball_present_overall": float(present.mean()),
        "frac_ball_below_overall": float(below.mean()),
        "frac_present_by_step": present.mean(0).tolist(),
        "frac_below_by_step": below.mean(0).tolist(),
        "n_started_hidden": int(started_hidden.sum()),
        "frac_reemerged": float(reemerged[started_hidden].mean())
        if started_hidden.any() else float("nan"),
        "arrivals_per_run": float((below[:, 0].sum() + edges.sum()) / len(below)),
        "mean_y_sd": float(np.nanmean(sd)) if np.isfinite(sd).any()
        else float("nan"),
        "mean_decoded_y": float(np.nanmean(ys)) if np.isfinite(ys).any()
        else float("nan"),
    }


def paired_starts(pool, batch: int, steps: int, rng: np.random.Generator,
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """``(episode, t0)`` pairs that have a FULL real continuation behind them.

    ``StartPool.sample`` draws ``t0`` anywhere in ``[warmup, T)``, which is the
    right thing for training -- every real state is a legitimate place to start
    dreaming from. It is the wrong thing here, because the reference this script
    grades the dream against is the real episode's own next 150 frames, and an
    episode that ends at t0 + 10 does not have them. Measured with the
    unconstrained draw, the real re-emergence rate came out at 48.8 % purely
    because half the reference windows were the final frame repeated; with the
    constraint it is what the physics actually does.

    So ``t0`` is confined to ``[warmup, T - steps]``. The world is stationary --
    the ball bounces for ever and the episode has no phases -- so this costs
    nothing but the position in the episode, and it buys an exactly paired
    comparison.
    """
    E, T = pool.actions.shape
    hi = T - steps
    if hi <= pool.warmup:
        raise SystemExit(
            f"--steps {steps} leaves no room for a real reference window in "
            f"{T}-step episodes (need t0 in [{pool.warmup}, {hi}]); lower "
            "--steps or use longer episodes"
        )
    return rng.integers(0, E, size=batch), rng.integers(pool.warmup, hi + 1,
                                                        size=batch)


def real_from_starts(root: str, cfg: BoxConfig, e: np.ndarray, t0: np.ndarray,
                     steps: int) -> Dict:
    """The detector on the REAL continuations of the dreams' own start states.

    This is the reference that matters, and it is paired: the same 64
    ``(episode, t0)`` pairs the dreams are warm-started from, followed forward
    through the real ``frames.npy`` for the same 150 steps. So
    ``frac_below_by_step`` and ``frac_reemerged`` mean exactly the same thing
    for the world and for the dream, and the comparison carries no selection
    effect from "which frames happen to be in the dataset".

    Only defined for a single root, which is why ``--start-roots`` defaults to
    one: with several roots pooled, ``e`` indexes the concatenation and there is
    no single ``frames.npy`` to walk. ``paired_starts`` guarantees the window
    fits; the clamp below is belt-and-braces and would silently flatten the
    reference if it ever fired, so it is asserted instead.
    """
    frames = np.load(Path(root) / "frames.npy", mmap_mode="r")
    T = frames.shape[1]
    if int(t0.max()) + steps > T - 1:
        raise ValueError(
            f"start t0 max {t0.max()} + {steps} steps overruns the {T}-frame "
            "episodes; use paired_starts to draw the starts"
        )
    pres, bel, yy = [], [], []
    for t in range(steps):
        x = np.asarray(frames[e, t0 + t], np.float32) / 255.0
        d = detect_ball(x, cfg)
        pres.append(d["present"])
        bel.append(d["below"])
        yy.append(d["y"])
    return presence_stats(np.stack(pres, 1), np.stack(bel, 1), np.stack(yy, 1))


# ------------------------------------------------------------- the reward head


@torch.no_grad()
def reward_head_r2(rnn, roots: Sequence[str], device: str = "cpu",
                   max_episodes: int = 40, steps: int = 199) -> Dict[str, float]:
    """R^2 of M's dense reward head against the truth, TEACHER-FORCED.

    Teacher-forced means M is fed the real latents and real actions, so this
    measures the head alone with the dream's own error taken out of the
    question. It is the ceiling for anything the head can contribute inside a
    dream, and if it is low there is no point looking at the dream at all.
    """
    pred, true = [], []
    for root in roots:
        d = episode_arrays(root)
        mu = d["mu"][:max_episodes, : steps + 1]
        a = d["actions"][:max_episodes, :steps]
        r = d["reward"][:max_episodes, :steps]
        z = torch.from_numpy(np.asarray(mu[:, :steps], np.float32)).to(device)
        eye = torch.eye(N_ACTIONS, device=device)
        a_oh = eye[torch.from_numpy(np.asarray(a, np.int64)).to(device)]
        parts, _ = rnn(z, a_oh)
        pred.append(parts["reward"][..., 0].cpu().numpy().ravel())
        true.append(np.asarray(r, np.float64).ravel())
    p, t = np.concatenate(pred), np.concatenate(true)
    ss_res = float(((p - t) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return {
        "n": int(len(t)),
        "r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "pearson_r": float(np.corrcoef(p, t)[0, 1]),
        "rmse": float(np.sqrt(((p - t) ** 2).mean())),
        "pred_mean": float(p.mean()),
        "true_mean": float(t.mean()),
    }


# ------------------------------------------------------------------ the dream


def sticky_actions(batch: int, steps: int, mean_hold: float = 8.0,
                   seed: int = 0) -> np.ndarray:
    """``(batch, steps)`` sticky-random actions -- ``data/v31/train``'s policy.

    Identical for every model and every temperature, so the twelve cells differ
    only in the model and the sampling temperature. A dream driven by STAY would
    freeze the paddle and leave the reward correlation below with no variance in
    one of its two terms.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((batch, steps), np.int64)
    a = rng.integers(0, N_ACTIONS, size=batch)
    hold = rng.geometric(1.0 / mean_hold, size=batch)
    for t in range(steps):
        switch = hold <= 0
        a = np.where(switch, rng.integers(0, N_ACTIONS, size=batch), a)
        hold = np.where(switch, rng.geometric(1.0 / mean_hold, size=batch), hold)
        out[:, t] = a
        hold -= 1
    return out


@torch.no_grad()
def dream_alive(rnn, vae, pool, cfg: BoxConfig, temperature: float,
                starts: Tuple[np.ndarray, np.ndarray],
                steps: int = 150, seed: int = 0,
                device: str = "cpu") -> Dict:
    """Dream, decode every frame, and measure whether there is a ball in it.

    Written as its own loop rather than via ``dream_rollout`` because two things
    are needed that the training loop has no use for: the reward head's output
    *per step* (``DreamEnv`` is built with ``reward="dense"``, so its returned
    reward IS the head) and the decoded frame *per step*.
    """
    env = DreamEnv(rnn, pool, temperature=temperature, reward="dense",
                   device=device)
    batch = len(starts[0])
    z, _ = env.reset(batch=batch, seed=seed, starts=starts)
    acts = sticky_actions(batch, steps, seed=seed)

    per_step: List[Dict[str, np.ndarray]] = []
    rewards, gaps = [], []
    for t in range(steps):
        # Decode the CURRENT latent, then step: frame t is what the controller
        # would have been looking at when it chose action t.
        x = torch.from_numpy(np.asarray(z, np.float32)).to(device)
        img = vae.decode(x).permute(0, 2, 3, 1).cpu().numpy()
        d = detect_ball(img, cfg)
        px = detect_paddle(img, cfg)
        z, _, r, _, _ = env.step(acts[:, t])
        per_step.append({"present": d["present"], "below": d["below"],
                         "y": d["y"], "mass": d["mass"]})
        ok = d["present"] & np.isfinite(px)
        rewards.append(r[ok])
        gaps.append(-np.abs(d["x"][ok] - px[ok]))

    present = np.stack([p["present"] for p in per_step], 1)     # (B, steps)
    below = np.stack([p["below"] for p in per_step], 1)
    ys = np.stack([p["y"] for p in per_step], 1)
    mass = np.stack([p["mass"] for p in per_step], 1)

    rw = np.concatenate(rewards) if rewards else np.zeros(0)
    gp = np.concatenate(gaps) if gaps else np.zeros(0)
    corr = (float(np.corrcoef(rw, gp)[0, 1])
            if len(rw) > 30 and rw.std() > 1e-9 and gp.std() > 1e-9
            else float("nan"))

    out = {
        "temperature": float(temperature),
        "batch": int(batch),
        "steps": int(steps),
        "mean_mass_by_step": mass.mean(0).tolist(),
        "reward_vs_decoded_gap_r": corr,
        "n_reward_pairs": int(len(rw)),
        "reward_mean": float(rw.mean()) if len(rw) else float("nan"),
    }
    out.update(presence_stats(present, below, ys))
    return out


# ------------------------------------------------------------------- the plot


def plot_dream_alive(report: Dict, out: Path) -> Path:
    """One row per model: ball present, ball below the band, reward honesty.

    The real-data fractions are the dashed black lines, and they are the point
    of the figure: a dream curve that starts near them and decays to zero is a
    dream that loses the ball, which is a different failure from one that never
    had it.
    """
    plt = _plt()
    models = list(report["models"])
    real = report["real_from_starts"]
    taus = report["temperatures"]
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.82, len(taus)))
    fig, axes = plt.subplots(len(models), 3, figsize=(13.5, 2.9 * len(models)),
                             squeeze=False)
    for i, name in enumerate(models):
        cells = report["models"][name]["cells"]
        for j, (key, label) in enumerate((
            ("frac_present_by_step", "fraction of dreams with a decoded ball"),
            ("frac_below_by_step", "fraction with the ball BELOW the band"),
        )):
            ax = axes[i][j]
            for c, tau in zip(colors, taus):
                cell = cells[str(tau)]
                ax.plot(cell[key], c=c, lw=1.4, label=f"tau={tau}")
            # The real continuation of the SAME 64 starts, step for step: the
            # only honest reference, because a dream step and a real step are
            # then the same step of the same episode.
            ax.plot(real[key], c="k", ls="--", lw=1.4,
                    label="real continuation" if i == 0 and j == 0 else None)
            # The left panel needs the full [0, 1]: the feed-forward dream
            # paints a ball on three quarters of its frames and a clipped axis
            # would hide the most surprising curve in the figure. The right
            # panel is zoomed to the region the world actually occupies (a few
            # per cent), since that is what the dream has to match.
            top = 1.02 if j == 0 else max(0.30, max(real[key]) * 1.3)
            ax.set_ylim(-0.02, top)
            ax.set_xlabel("dream step", fontsize=8)
            ax.set_ylabel(label, fontsize=7)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
        ax = axes[i][2]
        vals = [cells[str(t)]["reward_vs_decoded_gap_r"] for t in taus]
        ns = [cells[str(t)]["n_reward_pairs"] for t in taus]
        ax.bar(np.arange(len(taus)), vals,
               color=[c for c in colors])
        ax.axhline(0, c="k", lw=0.8)
        ax.set_xticks(np.arange(len(taus)))
        ax.set_xticklabels([f"tau={t}\n(n={n})" for t, n in zip(taus, ns)],
                           fontsize=7)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("r(reward head, -|decoded gap|)", fontsize=7)
        r2 = report["models"][name]["reward_head_r2"]["r2"]
        ax.set_title(f"reward head R2 (teacher-forced) = {r2:.2f}", fontsize=8)
        axes[i][0].set_title(f"{name}", fontsize=10, loc="left",
                             fontweight="bold")
    fig.suptitle("is the v3.1 dream alive? decoded ball presence over 150 dream "
                 "steps, against the real data", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ reporting


def pick_tau(cells: Dict[str, Dict], real: Dict) -> float:
    """The temperature whose ball re-emerges at the rate closest to the world's.

    The statistic is ``arrivals_per_run`` -- how many separate times, in 150
    frames, a ball decodes below the band -- against the real continuation of
    the same starts. It is preferred to ``frac_reemerged`` for two reasons, both
    of which came out of the measurement rather than out of taste:

    * ``frac_reemerged`` is a once-per-run indicator, so with 40 runs that
      began hidden its resolution is 2.5 percentage points, and the first pass
      of this script picked tau = 0 over tau = 0.5 by **0.1** of a point -- a
      tie decided by rounding, between a dream that never brings the ball back
      and one that always does.
    * a dream can bring a ball below the band once and then leave it parked
      there for 140 frames. That scores a perfect ``frac_reemerged`` and is not
      a world. Arrivals counts it once against the world's ~3.

    "Closest to" rather than "most": a dream that arrives far more often than
    the world is hallucinating, and a controller fitted to it would learn to
    expect a ball that is not coming. Ties go to the LOWER temperature, since
    stage two measured that a sampled v3.1 dream does not conserve the hidden
    ball's direction of travel (README_V31 §4.8).
    """
    target = real["arrivals_per_run"]
    best, best_err = None, np.inf
    for tau in sorted(cells, key=float):
        v = cells[tau]["arrivals_per_run"]
        err = np.inf if not np.isfinite(v) else abs(v - target)
        if err < best_err - 1e-9:
            best, best_err = float(tau), err
    return float(min(cells, key=float)) if best is None else float(best)


def verdict(report: Dict) -> Dict:
    """The up-front paragraph, computed rather than written.

    Two independent ways for the dream to be unusable, and they have to be
    reported separately because on v3.1 they disagree:

    * **no ball at all.** Every cell hides the ball on more than 80 % of dream
      frames. That is the failure v3's README_C3 §8 described.
    * **no ball where it matters.** A ball is decoded, but never below the
      band, so it is never in the strip where the paddle can act. This is the
      one that actually binds on v3.1, and a presence-only criterion would
      declare those dreams healthy: the feed-forward model paints a ball on
      three quarters of its frames and it is parked near the top of the box on
      every one of them.

    "Where it matters" is measured as arrivals below the band against the real
    continuation's rate. Half the world's rate is the "almost never" cut and
    twice it is the "too often" cut -- and the second cut earns its place, not
    as symmetry: at tau >= 0.5 the baseline's dream drops a ball to the band's
    lower edge and leaves it there, flickering across the detector's threshold
    four to six times more often than the world produces a real arrival, with a
    decoded height that barely varies. More arrivals than the world is not a
    more alive dream.
    """
    rs = report["real_from_starts"]
    real_arr, real_sd = rs["arrivals_per_run"], rs["mean_y_sd"]
    per_model: Dict[str, Dict] = {}
    for name, m in report["models"].items():
        best = max(c["frac_ball_present_overall"] for c in m["cells"].values())
        # Classified on the CHOSEN cell, because that is the dream the
        # controller will actually be trained in. The best cell over tau is
        # reported alongside so the reader can see whether a different
        # temperature would have been better and the selection rule missed it.
        cell = m["cells"][str(m["chosen_tau"])]
        arr, sd = cell["arrivals_per_run"], cell["mean_y_sd"]
        best_arr = max(c["arrivals_per_run"] for c in m["cells"].values())
        if arr < 0.5 * real_arr:
            regime = "almost never a ball below the band"
        elif arr > 2.0 * real_arr:
            regime = ("a ball below the band %.0fx too often, height sd %.2fx "
                      "the world's — parked at the band edge and flickering "
                      "across the detector, not a trajectory"
                      % (arr / real_arr, sd / max(real_sd, 1e-9)))
        else:
            regime = "comparable to the world"
        per_model[name] = {
            "best_frac_present": best,
            "chosen_arrivals_per_run": arr,
            "best_arrivals_per_run": best_arr,
            "arrivals_vs_real": arr / real_arr if real_arr else float("nan"),
            "y_sd_vs_real": sd / real_sd if real_sd else float("nan"),
            "regime": regime,
            "chosen_tau": m["chosen_tau"],
        }
    usable = [n for n, d in per_model.items()
              if d["regime"] == "comparable to the world"]
    # Is ANY cell in the comparable range, chosen or not? Asked separately, so
    # that "the dream is unusable" is never a claim about a temperature the
    # selection rule happened not to pick.
    any_cell = [
        (n, tau) for n, m in report["models"].items() for tau, c in m["cells"].items()
        if 0.5 * real_arr <= c["arrivals_per_run"] <= 2.0 * real_arr
    ]
    dead_presence = all(d["best_frac_present"] <= 0.20
                        for d in per_model.values())
    lines = []
    if dead_presence:
        lines.append(
            "At every model and every temperature the decoded ball is absent "
            "from more than 80 % of dream frames."
        )
    if not usable:
        lines.append(
            "**THE DREAM IS NOT A USABLE TRAINING ENVIRONMENT FOR THIS TASK.** "
            "At its chosen temperature no model puts a ball below the band at "
            f"a rate comparable to the world's ({real_arr:.2f} arrivals per "
            f"150 frames, decoded-height sd {real_sd:.3f}) — "
            + "; ".join(f"`{n}` {d['regime']}" for n, d in per_model.items())
            + (". No other (model, temperature) cell is in that range either."
               if not any_cell else
               ". The cells that ARE in that range were not the ones selected: "
               + ", ".join(f"`{n}` at tau={t}" for n, t in any_cell) + ".")
            + " A ball decoded above the band is scenery — the paddle cannot "
            "reach it and no action changes it; a ball parked at the band's "
            "lower edge is a fixed point, not a trajectory. So every "
            "controller trained below is fitted to what its M's REWARD HEAD "
            "says about latents that contain no reachable ball, not to a "
            "simulated interception task, and every controller number must be "
            "read that way."
        )
    else:
        lines.append(
            "The dream puts a ball below the band at a rate comparable to the "
            "world's for " + ", ".join(f"`{n}`" for n in usable)
            + "; the other models: "
            + "; ".join(f"`{n}` {d['regime']}" for n, d in per_model.items()
                        if n not in usable) + "."
        )
    return {
        "dream_is_dead_by_presence": bool(dead_presence),
        "models_with_a_usable_dream": usable,
        "other_cells_in_the_comparable_range": [
            {"model": n, "tau": t} for n, t in any_cell],
        "real_arrivals_per_run": float(real_arr),
        "real_y_sd": float(real_sd),
        "per_model": per_model,
        "statement": " ".join(lines),
    }


def write_report(report: Dict, out: Path) -> None:
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    v = report["verdict"]
    real = report["real"]
    rs = report["real_from_starts"]
    L = [
        "# v3.1 stage three, step zero — is the dream alive?",
        "",
        v["statement"],
        "",
        f"{report['batch']} dreams x {report['steps']} steps per cell, "
        f"warm-started on {report['warmup']} real frames from "
        f"`{report['start_roots']}`, driven by a fixed sticky-random action "
        "stream. Every model and every temperature sees the identical starts "
        "and the identical actions, so the cells are paired.",
        "",
        "## The real data, measured with the same detector",
        "",
        "| reference | frames | ball present | ball below the band | "
        "arrivals below the band per 150 frames | ball-height sd | "
        "re-emergence (of runs starting hidden) |",
        "|---|---|---|---|---|---|---|",
        f"| `{real['root']}`, {real['n_frames']} random frames | "
        f"{real['n_frames']} | {real['frac_ball_present']:.1%} | "
        f"{real['frac_ball_below_band']:.1%} | — | — | — |",
        f"| **the real continuation of the dreams' own "
        f"{report['batch']} starts** | "
        f"{report['batch']}x{report['steps']} | "
        f"{rs['frac_ball_present_overall']:.1%} | "
        f"{rs['frac_ball_below_overall']:.1%} | "
        f"**{rs['arrivals_per_run']:.2f}** | {rs['mean_y_sd']:.3f} | "
        f"{rs['frac_reemerged']:.1%} ({rs['n_started_hidden']}) |",
        "",
        "The detector's `ball present` is a half-a-ball-outside-the-band "
        "threshold, so it should land between the dataset's "
        "`frac_frames_visible` "
        f"({real['meta_frac_frames_visible']:.1%}, the ball entirely clear of "
        "the band) and `visible + partial` (57 %) — i.e. at roughly the "
        "fraction of frames with `ball_visible > 0.5`. It does. `ball below "
        "the band` is the reference the dream is graded against: the fraction "
        "of frames on which the ball is in the strip where the paddle can act "
        "at all, which in this world is only a few per cent.",
        "",
        "## Every cell",
        "",
        "| model | tau | ball present | ball below band | arrivals / run "
        f"(real {rs['arrivals_per_run']:.2f}) | ball-height sd "
        f"(real {rs['mean_y_sd']:.3f}) | mean decoded ball y | "
        "re-emergence (of dreams starting hidden) | r(reward head, "
        "-abs decoded gap) | n pairs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report["models"].items():
        for tau in report["temperatures"]:
            c = m["cells"][str(tau)]
            mark = " **<-- chosen**" if float(tau) == m["chosen_tau"] else ""
            L.append(
                f"| `{name}`{mark} | {tau} | "
                f"{c['frac_ball_present_overall']:.1%} | "
                f"{c['frac_ball_below_overall']:.1%} | "
                f"{c['arrivals_per_run']:.2f} | {c['mean_y_sd']:.3f} | "
                f"{c['mean_decoded_y']:.3f} | "
                f"{c['frac_reemerged']:.1%} ({c['n_started_hidden']}) | "
                f"{c['reward_vs_decoded_gap_r']:.2f} | {c['n_reward_pairs']} |"
            )
    L += ["", "## The reward head, teacher-forced on real latents", "",
          "M's dense reward head against the true `1 - |ball_x - paddle_x|` on "
          "real latents and real actions. This is the ceiling on what the head "
          "can contribute inside a dream.", "",
          "| model | R2 | Pearson r | rmse | pred mean | true mean | n |",
          "|---|---|---|---|---|---|---|"]
    for name, m in report["models"].items():
        r = m["reward_head_r2"]
        L.append(f"| `{name}` | {r['r2']:.3f} | {r['pearson_r']:.3f} | "
                 f"{r['rmse']:.4f} | {r['pred_mean']:.3f} | "
                 f"{r['true_mean']:.3f} | {r['n']:,} |")
    L += ["", "## Chosen training temperature per model", "",
          "The tau whose count of arrivals below the band is closest to the "
          f"real continuation's {rs['arrivals_per_run']:.2f} per 150 frames, "
          "ties going to the lower tau (stage two: a sampled v3.1 dream does "
          "not conserve the hidden ball's direction of travel).", "",
          "| model | chosen tau | its arrivals / run | real |",
          "|---|---|---|---|"]
    for name, m in report["models"].items():
        c = m["cells"][str(m["chosen_tau"])]
        L.append(f"| `{name}` | {m['chosen_tau']} | "
                 f"{c['arrivals_per_run']:.2f} | "
                 f"{rs['arrivals_per_run']:.2f} |")
    L += ["", f"Wall clock {report['wall_clock_s']:.0f} s.", ""]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# --------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vae", default="runs/vae_v31/vae.pt")
    p.add_argument("--models", nargs="*", default=[
        "baseline=runs/rnn_v31/rnn.pt",
        "poshead=runs/rnn_v31_poshead/rnn.pt",
        "ff=runs/rnn_v31_ff/rnn.pt",
        "allbands=runs/rnn_v31_allbands/rnn.pt",
    ], metavar="NAME=PATH")
    p.add_argument("--temperatures", type=float, nargs="*",
                   default=[0.0, 0.5, 1.0])
    p.add_argument("--start-roots", nargs="*", default=["data/v31/train_mix"])
    p.add_argument("--val", nargs="*",
                   default=["data/v31/val", "data/v31/val_mix"])
    p.add_argument("--real-root", default="data/v31/train_mix")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--real-frames", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ball-radius", type=float, default=0.08)
    p.add_argument("--occluder-y", type=float, nargs=2, default=[0.13, 0.63])
    p.add_argument("--paddle-w", type=float, default=0.16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="runs/rnn_v31_dream_alive")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = BoxConfig(res=64, ball_radius=a.ball_radius, occluder=True,
                    occluder_y=(a.occluder_y[0], a.occluder_y[1]),
                    paddle_w=a.paddle_w)
    vae, _, _ = load_ckpt(a.vae, a.device)
    pool = load_start_pool(a.start_roots, warmup=a.warmup)
    print(f"start pool: {pool.mu.shape[0]} episodes x {pool.actions.shape[1]} "
          f"steps, warmup {a.warmup}")

    real = real_fractions(a.real_root, cfg, n_frames=a.real_frames, seed=a.seed)
    print(f"real frames ({real['root']}): ball present "
          f"{real['frac_ball_present']:.1%} (meta says visible "
          f"{real['meta_frac_frames_visible']:.1%}), below the band "
          f"{real['frac_ball_below_band']:.1%}")

    # One set of starts for the whole script. Drawn here rather than inside
    # each cell so that all twelve dreams AND the real reference below are the
    # same 64 (episode, t0) pairs.
    starts = paired_starts(pool, a.batch, a.steps,
                           np.random.default_rng(a.seed))
    rs = real_from_starts(a.start_roots[0], cfg, starts[0], starts[1], a.steps)
    print(f"real continuation of those {a.batch} starts: present "
          f"{rs['frac_ball_present_overall']:.1%}, below the band "
          f"{rs['frac_ball_below_overall']:.1%}, arrivals/run "
          f"{rs['arrivals_per_run']:.2f}, re-emergence "
          f"{rs['frac_reemerged']:.1%} of {rs['n_started_hidden']} that began "
          "hidden")

    models: Dict[str, Dict] = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        rnn, _ = load_rnn(path, a.device)
        rh = reward_head_r2(rnn, a.val, device=a.device)
        print(f"\n{name} ({path}): reward head R2 {rh['r2']:.3f}")
        cells: Dict[str, Dict] = {}
        for tau in a.temperatures:
            t = time.time()
            cells[str(tau)] = dream_alive(
                rnn, vae, pool, cfg, tau, starts, steps=a.steps,
                seed=a.seed, device=a.device,
            )
            c = cells[str(tau)]
            print(f"  tau {tau:<4} present {c['frac_ball_present_overall']:.1%} "
                  f"below {c['frac_ball_below_overall']:.1%}  arrivals "
                  f"{c['arrivals_per_run']:.2f}  r(reward, gap) "
                  f"{c['reward_vs_decoded_gap_r']:.2f}  ({time.time() - t:.0f}s)")
        models[name] = {
            "path": path,
            "reward_head_r2": rh,
            "cells": cells,
            "chosen_tau": pick_tau(cells, rs),
        }
        print(f"  -> chosen tau {models[name]['chosen_tau']}")

    report = {
        "real": real,
        "real_from_starts": rs,
        "models": models,
        "temperatures": a.temperatures,
        "start_roots": a.start_roots,
        "batch": a.batch,
        "steps": a.steps,
        "warmup": a.warmup,
        "band": list(a.occluder_y),
        "paddle_w": a.paddle_w,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    report["verdict"] = verdict(report)
    plot_dream_alive(report, out / "dream_alive.png")
    write_report(report, out)
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

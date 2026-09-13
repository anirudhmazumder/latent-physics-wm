"""Does a long dream conserve what the world conserves? The metric that was missing.

    python -m wm.eval_conservation --ckpt runs/rnn_v2/rnn.pt \
        --vae runs/vae_v2/vae.pt --out runs/rnn_v2/conservation

    # v1 sanity reference: no mass there, but the speed IS a constant
    python -m wm.eval_conservation --ckpt runs/rnn_v1/rnn.pt \
        --vae runs/vae_b1/vae.pt --val data/v1/val data/v1/val_mix \
        --probe-data data/v1/probe data/v1/val_mix --out runs/rnn_v1/conservation

Why this file exists
--------------------
Stage three's most important finding was made by watching a GIF: over a 200-step
tau = 1 dream the ball's colour walks from yellow through red to purple, so the
speed law the controller was training against drifted under it. Nothing in
``eval_rnn`` or ``eval_causal_v2`` could have caught that. Both of them dream
for 24-64 steps, mostly at tau = 0, and both measure *agreement with a
particular true trajectory* -- which is the wrong question for a sampled dream
(a tau = 1 dream is a different plausible future, not a failed copy of this
one) and is in any case dominated by position long before the colour has walked
anywhere.

The missing question is narrower and answerable: **the world has quantities
that do not change, so does the dream hold them?** Mass is constant within a v2
episode. It follows that anything derived from mass is constant too -- most
importantly the speed. So:

    (i)   corr(dreamed log-mass, true log-mass) vs dream step
    (ii)  RMSE of the dreamed log-mass vs dream step
    (iii) the SPEED implied by the dreamed positions, against the law
          ``0.022 / mass`` -- the dynamical consequence of (i), and the thing
          that actually damages a controller
    (iv)  the fraction of dreamed frames whose decoded ball is still
          well-formed -- the drift's other visible symptom was smearing and
          fragmentation around step 60, which is not a colour statement

all at tau in {0, 0.5, 1}, over a 200-step horizon, which is the horizon the
controller actually trained in.

(iii) is the point of the whole exercise and is why this is not just "plot the
colour". A colour that random-walks is only interesting because the colour is
*load-bearing*: it sets the speed. Measuring the dreamed speed against the law
turns an aesthetic complaint into a physics one.

v1 has no mass, so (i) and (ii) do not exist there. But v1's ball speed is a
hard constant (0.022 for every episode, every frame), so (iii) and (iv) are
still exactly conservation tests, and running them on ``runs/rnn_v1`` answers
the obvious sceptical question: is stochastic drift a v2 problem, or is it what
any MDN-RNN does when you sample it for 200 steps?

Measurement choices, all arguable, all recorded
-----------------------------------------------
* **Warm-up is 8 true frames**, matching ``eval_rnn``. One frame would leave
  the model with no velocity and confound the speed metric with the cold-start
  under-movement documented in ``README_M2`` §6.
* **Actions run out.** Episodes are 200 transitions and we dream 200 steps
  after an 8-frame warm-up, so the last 8 steps have no recorded action. We
  **hold the last action** for those steps. It affects only the paddle, only at
  the very end of the horizon, and only for 4 % of the rollout.
* **log-mass is read with the frozen poly-2 ridge probe** on ``mu``, fit on
  ``data/v2/probe`` + ``data/v2/val_mix`` at every 2nd frame. Poly-2 and not
  linear or kNN because that is the only probe family that can read mass at all
  out of this latent space (stage one: R^2 0.97 poly-2, 0.10 linear, 0.08 kNN).
* **Position is read with the same frozen kNN probe** the rest of the eval uses.
* **Well-formedness is a cheap proxy**: decode the dreamed latent, count pixels
  that are near the colour ramp and far from the background, and call the frame
  well-formed if that count is within 50-150 % of a per-episode reference. The
  reference is the same count on the VAE's *reconstruction of the true
  latents*, not on the true frames, so the VAE's own blur is divided out and
  what is left is the dynamics model's contribution. This is a proxy, not a
  segmentation: it will not notice a ball that is the right size in the wrong
  place, and it will flag a ball that has merged with the paddle.
* **The probe floor is plotted for every metric.** Everything here is measured
  through two frozen probes with error of their own; the floor is the same
  estimator applied to TRUE latents, and it only exists for the first 192 steps
  because that is where true latents exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from worldsim.bouncing_box import BoxConfig, _ramp_points

from .analyze import load_ckpt
from .conservation import Poly2Probe, corr, rolling_speed, speed_horizon
from .eval_causal_v2 import Episodes, Setup, fit_position_probe, load_episodes
from .eval_rnn import BASE_SPEED, _plt, decode_latents, dream
from .rnn import load_rnn
from .seq_data import episode_arrays
from .train_vae import pick_device

# Steps at which the headline table is printed. 199 is the end of the horizon
# the controller trained in; 24 is roughly where README_C2's follow-up found
# the correlation had already gone.
REPORT_STEPS = (0, 24, 60, 120, 199)

# Well-formedness: RGB distance thresholds for the ball mask. 60 is about a
# quarter of the distance between the background and the paddle colour, so a
# half-blended paddle edge pixel (the obvious false positive) is excluded, and
# so is a half-blended ball edge pixel (a consistent, small undercount that the
# per-episode reference divides out).
MASK_RAMP_TOL = 60.0
MASK_BG_TOL = 60.0
AREA_LO, AREA_HI = 0.5, 1.5


# ----------------------------------------------------------------- dreaming


def dream_long(
    setup: Setup, ep: Episodes, eps: np.ndarray, warmup: int, horizon: int,
    tau: float, device: str, seed: int,
) -> np.ndarray:
    """``eval_rnn.dream`` with the action/latent slabs padded past the episode.

    ``dream`` gathers a ``warmup + horizon`` slice of both ``mu`` and
    ``actions`` for bookkeeping, but only ever READS the first ``warmup``
    latents -- everything after that is the model's own. So padding ``mu`` by
    repeating its last frame is invisible, while padding ``actions`` by
    repeating the last one is the documented "hold the last action" choice.
    """
    T = ep.actions.shape[1]
    need = warmup + horizon
    mu, act = ep.mu[eps], ep.actions[eps]
    if need > T:
        pad = need - T
        mu = np.concatenate([mu, np.repeat(mu[:, -1:], pad, axis=1)], 1)
        act = np.concatenate([act, np.repeat(act[:, -1:], pad, axis=1)], 1)
    starts = np.zeros(len(eps), dtype=int)
    return dream(setup.model, mu, act, starts, warmup=warmup, horizon=horizon,
                 temperature=tau, device=device, seed=seed)


# ------------------------------------------------------------ ball-mask proxy


def ball_mask_area(frames: np.ndarray, cfg: BoxConfig) -> np.ndarray:
    """Count ball-ish pixels per frame. ``(..., 64, 64, 3)`` uint8 -> ``(...)``.

    "Ball-ish" = close to the mass->colour ramp AND far from the background.
    The ramp test is what keeps the paddle out: the paddle's blue and every
    blend of it with the background sit a long way off a polyline that runs
    yellow -> red -> purple. Vectorised projection onto the two ramp segments,
    the same geometry ``worldsim.color_to_mass`` uses one pixel at a time.
    """
    pts = _ramp_points(cfg)
    x = frames.astype(np.float64).reshape(-1, 3)
    resid = np.full(len(x), np.inf)
    for k in (0, 1):
        a, b = pts[k], pts[k + 1]
        d = b - a
        denom = float(d @ d)
        t = np.clip((x - a) @ d / max(denom, 1e-12), 0.0, 1.0)[:, None]
        resid = np.minimum(resid, np.sqrt(((a + t * d - x) ** 2).sum(-1)))
    d_bg = np.sqrt(((x - np.asarray(cfg.bg_color, float)) ** 2).sum(-1))
    mask = (resid < MASK_RAMP_TOL) & (d_bg > MASK_BG_TOL)
    return mask.reshape(frames.shape[:-3] + (-1,)).sum(-1).astype(float)


# ------------------------------------------------- v3: the hidden ball's velocity


def hidden_velocity_conservation(
    pos: np.ndarray, band: Sequence[float], radius: float, win: int = 3,
    min_len: int = 2,
) -> Dict[str, float]:
    """Does a dream hold the ball's velocity across a stretch it cannot see?

    v3's conserved quantity. The band hides the ball but changes no physics:
    there is no horizontal surface inside it, so a ball that goes in heading
    down MUST come out heading down, at the same speed, and with |vx|
    unchanged whether or not it bounced off a side wall on the way. Those are
    three statements the dream can violate independently.

    Unlike every other metric in this file this one needs no ground truth --
    it compares the dream against ITSELF, before and after the gap. That is
    what makes it usable at tau = 1, where the dream is a different plausible
    future and agreement with the recorded trajectory is the wrong question.

    ``pos`` is (N, H, 2) decoded dream positions. A "hidden stretch" is a
    maximal run of >= ``min_len`` steps whose decoded y is inside the band
    interior, with ``win`` clean steps either side to estimate velocity from.
    Returns aggregate statistics over every such stretch found in the batch.
    """
    lo, hi = band[0] + radius, band[1] - radius
    inside = (pos[..., 1] >= lo) & (pos[..., 1] <= hi)
    dvy_sign_ok, dvx_flip, speed_ratio, absvx_ratio = [], [], [], []
    N, H = inside.shape
    for n in range(N):
        t = 0
        while t < H:
            if not inside[n, t]:
                t += 1
                continue
            e = t
            while e + 1 < H and inside[n, e + 1]:
                e += 1
            if (e - t + 1) >= min_len and t - win - 1 >= 0 and e + 1 + win < H:
                v_in = (pos[n, t - 1] - pos[n, t - 1 - win]) / win
                v_out = (pos[n, e + 1 + win] - pos[n, e + 1]) / win
                dvy_sign_ok.append(float(np.sign(v_in[1]) == np.sign(v_out[1])))
                dvx_flip.append(float(np.sign(v_in[0]) != np.sign(v_out[0])))
                s_in = float(np.hypot(*v_in)) or 1e-9
                speed_ratio.append(float(np.hypot(*v_out)) / s_in)
                absvx_ratio.append(abs(float(v_out[0])) / max(abs(float(v_in[0])), 1e-9))
            t = e + 1
    n = len(dvy_sign_ok)
    if n == 0:
        return {"n_stretches": 0}
    return {
        "n_stretches": n,
        "vy_sign_preserved": float(np.mean(dvy_sign_ok)),
        "vx_sign_flipped": float(np.mean(dvx_flip)),
        "speed_ratio_median": float(np.median(speed_ratio)),
        "abs_vx_ratio_median": float(np.median(absvx_ratio)),
    }


# ------------------------------------------------------------------ metrics


def _per_step_corr(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """corr across EPISODES at each dream step. est (N, H), truth (N,)."""
    return np.array([corr(est[:, t], truth) for t in range(est.shape[1])])


def evaluate_tau(
    setup: Setup, ep: Episodes, eps: np.ndarray, tau: float, cfg: BoxConfig,
    mass_probe: Optional[Poly2Probe], warmup: int, horizon: int,
    decode_every: int, device: str, seed: int, ref_area: np.ndarray,
    speed_win: int, speed_tol: float,
    band: Optional[Sequence[float]] = None, radius: float = 0.08,
) -> Dict[str, object]:
    """All four metrics for one temperature. Returns arrays over dream steps."""
    z_d = dream_long(setup, ep, eps, warmup, horizon, tau, device, seed)
    pos = setup.probe(z_d)[..., :2]                                  # (N, H, 2)

    out: Dict[str, object] = {}

    # (iii) speed, and its horizon.
    true_speed = ep.true_speed[eps] if ep.has_mass else np.full(
        len(eps), BASE_SPEED)
    sp = rolling_speed(pos, win=speed_win)                           # (N, H)
    rel = sp / true_speed[:, None]
    out["speed_ratio_median"] = np.median(rel, 0)
    out["speed_ratio_q25"] = np.percentile(rel, 25, axis=0)
    out["speed_ratio_q75"] = np.percentile(rel, 75, axis=0)
    out["speed_abs_rel_err_mean"] = np.abs(rel - 1.0).mean(0)
    out["speed_horizon_mean"] = float(
        speed_horizon(sp, true_speed, tol=speed_tol).mean())
    # The ratio above is a LEVEL and at tau > 0 it is inflated by something
    # other than drift: a sampled latent jitters from step to step, the
    # position probe turns that jitter into displacement, and the estimator
    # cannot tell jitter from motion. (That inflation is not an artefact of the
    # ruler alone -- a controller dreaming at tau = 1 really does watch a ball
    # that moves that much per frame -- but it is not what this file is about.)
    # The correlation ACROSS episodes is the jitter-insensitive companion: a
    # common multiplicative inflation leaves it untouched, while a colour that
    # random-walks independently per episode destroys it. Read the two together.
    out["speed_corr"] = _per_step_corr(sp, true_speed)

    # (v) v3 only: the hidden ball's velocity, across the band.
    if band is not None:
        out["hidden_velocity"] = hidden_velocity_conservation(pos, band, radius)

    # (i) and (ii): only defined where a mass exists.
    if mass_probe is not None and ep.has_mass:
        est = mass_probe(z_d)                                        # (N, H)
        truth = np.log(ep.mass[eps])
        out["logmass_corr"] = _per_step_corr(est, truth)
        out["logmass_rmse"] = np.sqrt(((est - truth[:, None]) ** 2).mean(0))
        out["logmass_bias"] = (est - truth[:, None]).mean(0)

    # (iv) well-formedness, on a subsample of steps (decoding every step of
    # every episode at three temperatures is the expensive part of this file
    # and buys nothing -- the quantity is smooth in the step index).
    steps = np.arange(0, horizon, decode_every)
    dec = decode_latents(setup.vae, z_d[:, steps], device=device)
    area = ball_mask_area(dec, cfg)                                  # (N, S)
    ok = (area >= AREA_LO * ref_area[:, None]) & (area <= AREA_HI * ref_area[:, None])
    out["wellformed_steps"] = steps
    out["wellformed_frac"] = ok.mean(0)
    out["ball_area_median"] = np.median(area, 0)
    return out


def probe_floor(
    setup: Setup, ep: Episodes, eps: np.ndarray, cfg: BoxConfig,
    mass_probe: Optional[Poly2Probe], warmup: int, horizon: int,
    decode_every: int, device: str, speed_win: int, speed_tol: float,
    band: Optional[Sequence[float]] = None, radius: float = 0.08,
) -> Dict[str, object]:
    """The same estimators applied to TRUE latents. The instrument's own score.

    Only defined for the steps where a true latent exists (dream step k is true
    index ``warmup + k``, and the episode has ``T + 1`` latents), so this stops
    short of the full horizon. Without it a corr of 0.93 is unreadable: you
    cannot tell whether the remaining 0.07 is the model or the probe.
    """
    T = ep.mu.shape[1] - 1
    n = int(min(horizon, T + 1 - warmup))
    idx = warmup + np.arange(n)
    mu = ep.mu[eps][:, idx]
    pos = setup.probe(mu)[..., :2]
    true_speed = ep.true_speed[eps] if ep.has_mass else np.full(len(eps), BASE_SPEED)
    sp = rolling_speed(pos, win=speed_win)
    out: Dict[str, object] = {
        "n_steps": n,
        "speed_ratio_median": np.median(sp / true_speed[:, None], 0),
        "speed_corr": _per_step_corr(sp, true_speed),
        "speed_horizon_mean": float(
            speed_horizon(sp, true_speed, tol=speed_tol).mean()),
    }
    if band is not None:
        out["hidden_velocity"] = hidden_velocity_conservation(pos, band, radius)
    if mass_probe is not None and ep.has_mass:
        est = mass_probe(mu)
        truth = np.log(ep.mass[eps])
        out["logmass_corr"] = _per_step_corr(est, truth)
        out["logmass_rmse"] = np.sqrt(((est - truth[:, None]) ** 2).mean(0))
    steps = np.arange(0, n, decode_every)
    dec = decode_latents(setup.vae, mu[:, steps], device=device)
    out["ref_area"] = np.median(ball_mask_area(dec, cfg), 1)         # (N,)
    out["wellformed_steps"] = steps
    return out


# --------------------------------------------------------------------- plot


def plot_conservation(res: Dict, floor: Dict, taus, path: Path, title: str) -> None:
    plt = _plt()
    has_mass = "logmass_corr" in next(iter(res.values()))
    rows = (["logmass_corr", "logmass_rmse", "speed", "speed_corr", "wellformed"]
            if has_mass else ["speed", "wellformed"])
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.0, 2.9 * len(rows)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    colours = {0.0: "tab:blue", 0.5: "tab:orange", 1.0: "tab:red"}

    for ax, row in zip(axes, rows):
        for tau in taus:
            d = res[tau]
            c = colours.get(tau)
            if row == "speed":
                y = d["speed_ratio_median"]
                ax.plot(np.arange(len(y)), y, color=c, label=f"tau={tau}")
                ax.fill_between(np.arange(len(y)), d["speed_ratio_q25"],
                                d["speed_ratio_q75"], color=c, alpha=0.12)
            elif row == "wellformed":
                ax.plot(d["wellformed_steps"], d["wellformed_frac"], color=c,
                        marker=".", ms=3, label=f"tau={tau}")
            else:
                y = d[row]
                ax.plot(np.arange(len(y)), y, color=c, label=f"tau={tau}")
        if row in floor and row != "speed":
            f = floor[row]
            ax.plot(np.arange(len(f)), f, "k--", lw=1,
                    label="probe floor (true latents)")
        if row == "speed":
            ax.axhline(1.0, color="k", ls="--", lw=1, label="the law")
            ax.axhspan(0.75, 1.25, color="grey", alpha=0.12,
                       label="+-25% (the speed-horizon band)")
            # Log scale: at tau = 1 the ratio runs to 5-10x and a linear axis
            # simply clips the most interesting line off the top of the panel.
            ax.set_yscale("log")
            ax.set_ylim(0.3, 20)
            ax.set_ylabel("dreamed speed / true speed")
            ax.set_title("(iii) the dynamical consequence: does the dream keep "
                         "the speed the law implies?")
        elif row == "wellformed":
            ax.set_ylim(-0.03, 1.03)
            ax.set_ylabel("fraction of frames")
            ax.set_title("(iv) fraction of dreamed frames whose decoded ball is "
                         "the right size (50-150% of the VAE reference)")
        elif row in ("logmass_corr", "speed_corr"):
            ax.axhline(0, color="k", lw=0.8)
            ax.axhline(0.8, color="green", ls=":", lw=1, label="0.8 (a full fix)")
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel("corr across episodes")
            ax.set_title("(i) dreamed log-mass vs true log-mass"
                         if row == "logmass_corr" else
                         "(iii b) corr(dreamed speed, true speed) -- the same "
                         "question, immune to the jitter inflation above")
        else:
            # Log scale for the same reason as (iii): a model whose latents
            # actually diverge puts this in the hundreds, and on a linear axis
            # that squashes every well-behaved curve onto the zero line.
            ax.set_yscale("log")
            ax.set_ylim(0.02, None)
            ax.set_ylabel("RMSE (nats of log mass)")
            ax.set_title("(ii) error in the dreamed log-mass "
                         "(the full range of log mass is 1.39)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)
    axes[-1].set_xlabel("dream step (after an 8-frame warm-up)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- main


def _at(arr, step: int):
    a = np.asarray(arr)
    return float(a[step]) if step < len(a) else float("nan")


# ---------------------------------------------------- the comparison figure


def compare(runs: Sequence[str], tau: float, path: Path) -> None:
    """One figure for several runs: conservation curves + a horizons table.

    Reads each run's ``conservation/report.json`` and, when it exists, its
    ``eval/report.json``, so the figure is assembled from measurements rather
    than recomputed -- there is no way for the plot and the tables in the
    README to disagree.
    """
    plt = _plt()
    rows: List[Dict] = []
    for r in runs:
        cons = json.loads((Path(r) / "conservation" / "report.json").read_text())
        ev_path = Path(r) / "eval" / "report.json"
        ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}
        rows.append({"run": Path(r).name, "cons": cons, "eval": ev})

    fig = plt.figure(figsize=(12.5, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.33, wspace=0.22)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, :])
    ax_t.axis("off")

    def smooth(y, w=11):
        """Centred rolling mean. The per-step correlations are computed from
        30 episodes and bounce by +-0.15 from step to step; the comparison is
        about the LEVEL and the TREND, so the raw curve is drawn faint and a
        smoothed one on top. Nothing is hidden -- each run's own figure shows
        the unsmoothed version."""
        k = np.ones(w) / w
        pad = np.pad(y, (w // 2, w // 2), mode="edge")
        return np.convolve(pad, k, mode="valid")[: len(y)]

    key = str(tau)
    for row in rows:
        per = row["cons"]["per_tau"].get(key)
        if per is None:
            continue
        for ax, k in ((ax_c, "logmass_corr"), (ax_s, "speed_corr")):
            y = np.asarray(per[k], float)
            line, = ax.plot(np.arange(len(y)), smooth(y), lw=1.8,
                            label=row["run"])
            ax.plot(np.arange(len(y)), y, lw=0.6, alpha=0.20,
                    color=line.get_color())
    floor = rows[0]["cons"]["probe_floor"]
    for ax, k in ((ax_c, "logmass_corr"), (ax_s, "speed_corr")):
        f = np.asarray(floor[k], float)
        ax.plot(np.arange(len(f)), f, "k--", lw=1, label="probe floor")
        ax.axhline(0.8, color="green", ls=":", lw=1)
        ax.axhline(0.0, color="k", lw=0.7)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("dream step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    ax_c.set_ylabel("corr across episodes")
    ax_c.set_title(f"(i) dreamed log-mass vs true log-mass, tau={tau}")
    ax_s.set_title(f"(iii b) dreamed speed vs the law, tau={tau}")

    hdr = ["run", "corr@24", "corr@60", "corr@120", "corr@150", "corr@199",
           "speed horizon\n(tau=1)", "useful horizon\n(tau=0, eval_rnn)"]
    table = []
    for row in rows:
        per = row["cons"]["per_tau"].get(key, {})
        cc = np.asarray(per.get("logmass_corr", []), float)
        uh = row["eval"].get("b_state", {}).get("tau0.0", {}).get(
            "useful_dream_horizon", float("nan"))
        table.append([
            row["run"],
            *[f"{_at(cc, s):+.2f}" for s in (24, 60, 120, 150, 199)],
            f"{row['cons']['summary'][key]['speed_horizon_mean']:.1f}",
            f"{uh:.0f}" if uh == uh else "-",
        ])
    t = ax_t.table(cellText=table, colLabels=hdr, loc="upper center",
                   cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1.0, 1.7)
    for j in range(len(hdr)):
        t[0, j].set_facecolor("#dddddd")
    ax_t.set_title(f"conservation of the ball's mass at tau={tau}, and the cost "
                   "in ordinary dream quality", fontsize=10, pad=2)

    fig.suptitle("candidate fixes for the colour drift in the v2 world model")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare", nargs="+", default=None,
                   help="run directories to put in one comparison figure "
                        "(reads their conservation/ and eval/ reports; does not "
                        "dream anything itself)")
    p.add_argument("--replot", default=None,
                   help="regenerate conservation.png for a run directory from "
                        "its existing conservation/report.json, without "
                        "dreaming anything again (the report carries every "
                        "per-step array the figure draws)")
    p.add_argument("--compare-out", default="runs/rnn_v2_fix_comparison.png")
    p.add_argument("--compare-tau", type=float, default=1.0)
    p.add_argument("--ckpt", default="runs/rnn_v2/rnn.pt")
    p.add_argument("--vae", default="runs/vae_v2/vae.pt")
    p.add_argument("--latent-suffix", default="")
    p.add_argument("--val", nargs="+", default=["data/v2/val", "data/v2/val_mix"])
    p.add_argument("--probe-data", nargs="+",
                   default=["data/v2/probe", "data/v2/val_mix"])
    p.add_argument("--out", default=None)
    p.add_argument("--name", default=None, help="label for the plots/report")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--taus", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument("--decode-every", type=int, default=4)
    p.add_argument("--speed-win", type=int, default=6)
    p.add_argument("--speed-tol", type=float, default=0.25)
    p.add_argument("--probe-samples", type=int, default=12000)
    p.add_argument("--mass-probe-stride", type=int, default=2,
                   help="use every Nth frame when fitting the log-mass probe")
    # v4. The conserved quantity in the gravity-switch world is the SIGN: it is
    # constant between paddle contacts and nothing in the model's loss restores
    # it once a dream lets it drift. Off unless asked for, so v1-v3.1 reruns are
    # byte-identical.
    p.add_argument("--sign-metric", action="store_true",
                   help="v4: probe-decoded gravity-sign consistency along the "
                        "dream. Needs a dataset with a gravity_sign column.")
    p.add_argument("--sign-probe-data", nargs="+", default=None,
                   help="roots to fit the sign probe on (default: --val)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.compare:
        compare(a.compare, a.compare_tau, Path(a.compare_out))
        return

    if a.replot:
        d = Path(a.replot)
        d = d if d.name == "conservation" else d / "conservation"
        rep = json.loads((d / "report.json").read_text())
        res = {float(k): {kk: np.asarray(vv) if isinstance(vv, list) else vv
                          for kk, vv in v.items()}
               for k, v in rep["per_tau"].items()}
        fl = {k: np.asarray(v) if isinstance(v, list) else v
              for k, v in rep["probe_floor"].items()}
        plot_conservation(res, fl, sorted(res), d / "conservation.png",
                          f"conservation over a {rep['horizon']}-step dream -- "
                          f"{rep['name']}")
        print(f"replotted {d / 'conservation.png'}")
        return

    device = pick_device(a.device)
    out = Path(a.out or (Path(a.ckpt).parent / "conservation"))
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or Path(a.ckpt).parent.name
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    vmeta = json.loads((Path(a.val[0]) / "meta.json").read_text())
    cfg = BoxConfig(**vmeta["config"])
    model, mcfg = load_rnn(a.ckpt, device)
    vae, vcfg, _ = load_ckpt(a.vae, device)
    pos_probe = fit_position_probe(a.probe_data, a.latent_suffix,
                                   a.probe_samples, a.seed)
    setup = Setup(name, model, a.latent_suffix, vae, pos_probe)

    ep = load_episodes(a.val, a.latent_suffix)
    n = min(a.n_episodes, ep.mu.shape[0])
    eps = np.arange(n)
    print(f"model {a.ckpt}\nvae   {a.vae}  z_dim={vcfg.z_dim}\n"
          f"{n} episodes x {a.horizon} dream steps, warm-up {a.warmup}, "
          f"taus {a.taus}, device={device}")

    mass_probe = _fit_mass_probe(a.probe_data, a.latent_suffix,
                                 a.mass_probe_stride) if ep.has_mass else None
    if mass_probe is None:
        print("no mass column in this dataset -- running (iii) and (iv) only; "
              "the speed is a hard constant here, so those ARE the conservation "
              "tests for v1")

    # v3: the band, if this world has one. Its presence switches on metric
    # (v) -- conservation of the HIDDEN ball's velocity direction, which is the
    # v3 analogue of v2's constant mass and is the thing a controller that has
    # to commit while the ball is invisible actually depends on.
    band = tuple(vmeta["occluder_y"]) if vmeta.get("occluder") else None
    if band is not None:
        print(f"occluder band {band}; metric (v) -- the hidden ball's velocity "
              f"-- is on")

    floor = probe_floor(setup, ep, eps, cfg, mass_probe, a.warmup, a.horizon,
                        a.decode_every, device, a.speed_win, a.speed_tol,
                        band=band, radius=cfg.ball_radius)
    ref_area = floor.pop("ref_area")

    res: Dict[float, Dict] = {}
    for tau in a.taus:
        print(f"\n  dreaming at tau={tau} ...", flush=True)
        res[tau] = evaluate_tau(setup, ep, eps, tau, cfg, mass_probe, a.warmup,
                                a.horizon, a.decode_every, device, a.seed,
                                ref_area, a.speed_win, a.speed_tol,
                                band=band, radius=cfg.ball_radius)

    plot_conservation(res, floor, a.taus, out / "conservation.png",
                      f"conservation over a {a.horizon}-step dream -- {name}")

    sign_metric = sign_conservation(
        model, a, device, name) if a.sign_metric else None

    report = {
        "name": name, "ckpt": str(a.ckpt), "vae": str(a.vae),
        "val_roots": list(a.val), "n_episodes": int(n),
        "warmup": a.warmup, "horizon": a.horizon, "taus": list(a.taus),
        "speed_tol": a.speed_tol, "speed_win": a.speed_win,
        "report_steps": list(REPORT_STEPS),
        "has_mass": bool(ep.has_mass),
        "band": list(band) if band else None,
        "action_policy_past_episode_end": "hold the last recorded action",
        "sign_conservation": sign_metric,
        "probe_floor": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in floor.items()},
        "per_tau": {
            str(tau): {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in d.items()}
            for tau, d in res.items()
        },
        "summary": {
            str(tau): {
                **({f"logmass_corr@{s}": _at(d["logmass_corr"], s)
                    for s in REPORT_STEPS} if "logmass_corr" in d else {}),
                **({f"logmass_rmse@{s}": _at(d["logmass_rmse"], s)
                    for s in REPORT_STEPS} if "logmass_rmse" in d else {}),
                **{f"speed_ratio@{s}": _at(d["speed_ratio_median"], s)
                   for s in REPORT_STEPS},
                **{f"speed_corr@{s}": _at(d["speed_corr"], s)
                   for s in REPORT_STEPS},
                "speed_horizon_mean": d["speed_horizon_mean"],
                **({f"hidden_{k}": v for k, v in d["hidden_velocity"].items()}
                   if "hidden_velocity" in d else {}),
                "wellformed_frac_last": float(d["wellformed_frac"][-1]),
                "wellformed_frac_mean": float(d["wellformed_frac"].mean()),
            }
            for tau, d in res.items()
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    _summary(report, res, floor, a, out)


def sign_conservation(model, a, device: str, name: str) -> Dict:
    """(vi) v4: does the dreamed world keep the gravity sign it started with?

    The v2 lesson said the ball's mass drifts in a sampled dream because nothing
    in the one-step loss punishes losing it. v4's sign is the same failure mode
    with the difficulty turned up: mass was at least *visible* in every frame,
    so a drifting dream contradicted its own pixels, whereas the sign is visible
    in no frame at all. There is literally nothing in a dreamed image for the
    model to check itself against.

    Measured with the same instrument as ``wm.eval_switch_v4`` (a): a linear
    probe from h to the sign, fitted on REAL teacher-forced passes and then
    frozen, read along the dream. A step at which the model's own hit head says
    a contact is happening is excluded, because at such a step the sign is
    *supposed* to change and holding it would be the error.

    Reusing that module rather than reimplementing is deliberate: two different
    sign probes would make the number here and the number in the memory curve
    incomparable, and comparing them is the point.
    """
    from .eval_switch_v4 import (
        SignProbe, experiment_e, hidden_states, load_split,
    )
    from .probes import make_split as _split

    roots = a.sign_probe_data or a.val
    sp = load_split(roots)
    if "gravity_sign" not in sp.names:
        print("  (--sign-metric asked for, but this world has no gravity_sign "
              "column; skipping)")
        return None

    H = hidden_states(model, sp.mu, sp.actions, device)
    E, T, _ = H.shape
    X = H.reshape(E * T, -1)
    y = sp.sign[:, :T].reshape(-1)
    grp = np.repeat(np.arange(E), T)
    tr, te = _split(len(X), frac_train=0.7, seed=a.seed, group_ids=grp)
    probe = SignProbe().fit(X, y, tr[::2])
    held = probe.accuracy(X[te], y[te])
    print(f"\n  sign probe on real teacher-forced h: held-out accuracy "
          f"{held:.3f} ({len(roots)} root(s), {E} episodes)")

    rec = experiment_e(model, sp, probe, device, a.seed, warm=a.warmup,
                       horizon=min(a.horizon, T - a.warmup - 1),
                       taus=tuple(a.taus), n_ep=min(a.n_episodes, E))
    rec["teacher_forced_probe_accuracy"] = held
    return rec


def _fit_mass_probe(roots: Sequence[str], suffix: str, stride: int) -> Poly2Probe:
    mus, ms = [], []
    for r in roots:
        d = episode_arrays(r, latent_suffix=suffix)
        mu, st = d["mu"][:, ::stride], d["state"][:, ::stride]
        mus.append(mu.reshape(-1, mu.shape[-1]))
        ms.append(st[..., 6].reshape(-1))
    M, Y = np.concatenate(mus), np.log(np.concatenate(ms))
    probe = Poly2Probe().fit(M, Y)
    pred = probe(M)
    r2 = 1.0 - ((Y - pred) ** 2).sum() / max(((Y - Y.mean()) ** 2).sum(), 1e-12)
    print(f"log-mass probe: poly-2 ridge on {len(M)} frames from {list(roots)}, "
          f"in-sample R^2 {r2:.4f}")
    return probe


def _summary(report: Dict, res: Dict, floor: Dict, a, out: Path) -> None:
    print("\n" + "=" * 78)
    print(f"CONSERVATION -- {report['name']}")
    print("=" * 78)

    if report["has_mass"]:
        print("\n(i) corr(dreamed log-mass, true log-mass), across episodes, "
              "by dream step")
        hdr = "    " + f"{'tau':>6s}" + "".join(f"{s:>9d}" for s in REPORT_STEPS)
        print(hdr)
        for tau in a.taus:
            d = res[tau]
            print("    " + f"{tau:6.2f}"
                  + "".join(f"{_at(d['logmass_corr'], s):9.3f}" for s in REPORT_STEPS))
        f = floor.get("logmass_corr")
        if f is not None:
            print("    " + f"{'floor':>6s}"
                  + "".join(f"{_at(f, s):9.3f}" for s in REPORT_STEPS)
                  + "   (the probe on TRUE latents; blank past step "
                  + f"{floor['n_steps'] - 1})")

        print("\n(ii) RMSE of the dreamed log-mass (the whole range of log-mass "
              "is log(2/0.5) = 1.386)")
        print(hdr)
        for tau in a.taus:
            d = res[tau]
            print("    " + f"{tau:6.2f}"
                  + "".join(f"{_at(d['logmass_rmse'], s):9.3f}" for s in REPORT_STEPS))

    print("\n(iii) dreamed speed / true speed (the law says 1.000)")
    print("    " + f"{'tau':>6s}" + "".join(f"{s:>9d}" for s in REPORT_STEPS)
          + f"{'horizon':>10s}")
    for tau in a.taus:
        d = res[tau]
        print("    " + f"{tau:6.2f}"
              + "".join(f"{_at(d['speed_ratio_median'], s):9.3f}" for s in REPORT_STEPS)
              + f"{d['speed_horizon_mean']:10.1f}")
    print("    " + f"{'floor':>6s}"
          + "".join(f"{_at(floor['speed_ratio_median'], s):9.3f}" for s in REPORT_STEPS)
          + f"{floor['speed_horizon_mean']:10.1f}")
    print(f"    'horizon' = mean first step at which the dreamed speed is off "
          f"the law by > {a.speed_tol:.0%} (max {a.horizon})")
    print("    At tau > 0 the RATIO is inflated by per-step sampling jitter that "
          "the position\n    probe cannot separate from motion.", end=" ")
    if report["has_mass"]:
        # In v1 every episode has the same true speed, so a correlation across
        # episodes has no variance to work with and is undefined by
        # construction -- not a failure, just a question v1 cannot be asked.
        print("The correlation below is the clean version:")
        print("    " + f"{'tau':>6s}" + "".join(f"{s:>9d}" for s in REPORT_STEPS))
        for tau in a.taus:
            print("    " + f"{tau:6.2f}"
                  + "".join(f"{_at(res[tau]['speed_corr'], s):9.3f}"
                            for s in REPORT_STEPS))
        print("    " + f"{'floor':>6s}"
              + "".join(f"{_at(floor['speed_corr'], s):9.3f}" for s in REPORT_STEPS))
    else:
        print("\n    (the across-episode speed correlation is undefined here: in v1 "
              "every\n    episode has the SAME true speed, so there is no variance "
              "to correlate with.)")

    if report.get("band"):
        print("\n(v) v3: the HIDDEN ball's velocity across a dreamed occlusion.")
        print("    Nothing inside the band can turn a ball around, so vy must keep "
              "its sign and\n    |v| and |vx| must be unchanged. This compares the "
              "dream with ITSELF, so it is\n    the one metric here that is "
              "meaningful at tau = 1.")
        print("    " + f"{'tau':>6s}{'stretches':>11s}{'vy sign kept':>14s}"
              f"{'|v| out/in':>12s}{'|vx| out/in':>13s}{'vx flipped':>12s}")
        rows = [(f"{t:6.2f}", res[t].get("hidden_velocity", {})) for t in a.taus]
        rows.append((f"{'floor':>6s}", floor.get("hidden_velocity", {})))
        for lab, hv in rows:
            if not hv.get("n_stretches"):
                print(f"    {lab}       (no complete hidden stretch found)")
                continue
            print(f"    {lab}{hv['n_stretches']:11d}{hv['vy_sign_preserved']:14.3f}"
                  f"{hv['speed_ratio_median']:12.3f}"
                  f"{hv['abs_vx_ratio_median']:13.3f}"
                  f"{hv['vx_sign_flipped']:12.3f}")
        print("    ('floor' is the same estimator on TRUE latents: the instrument's "
              "own score.)")

    sc = report.get("sign_conservation")
    if sc:
        print("\n(vi) v4: the gravity SIGN along the dream -- the conserved "
              "quantity here.")
        print(f"    A frozen linear probe on h, {sc['teacher_forced_probe_accuracy']:.3f} "
              f"accurate on real teacher-forced passes, read along a "
              f"{sc['horizon']}-step dream.")
        print("    " + f"{'tau':>6s}{'consistency':>13s}{'(all steps)':>13s}"
              f"{'steps the model calls a contact':>34s}")
        for t in sc["taus"]:
            print(f"    {t['tau']:6.2f}{t['sign_consistency']:13.3f}"
                  f"{t['sign_consistency_all_steps']:13.3f}"
                  f"{t['frac_steps_model_predicts_contact']:34.3f}")
        print("    'consistency' = fraction of dreamed steps whose decoded sign "
              "still matches the\n    one in force when the dream began, over "
              "steps at which the model's own hit head\n    does not think a "
              "contact is happening.")

    print("\n(iv) fraction of dreamed frames with a well-formed ball "
          "(area within 50-150% of the VAE reference)")
    for tau in a.taus:
        d = res[tau]
        print(f"    tau={tau:4.2f}  mean over the dream {d['wellformed_frac'].mean():.3f}"
              f"   at the end {d['wellformed_frac'][-1]:.3f}")

    # ---- plain English ---------------------------------------------------
    print("\nIn plain English:")
    hot = max(a.taus)
    d = res[hot]
    late = min(150, a.horizon - 1)          # the step the criterion is stated at
    if report["has_mass"]:
        c0, c24, cl = (_at(d["logmass_corr"], 0), _at(d["logmass_corr"], 24),
                       _at(d["logmass_corr"], late))
        gone = next((t for t in range(len(d["logmass_corr"]))
                     if d["logmass_corr"][t] < 0.5), None)
        print(
            f"  At tau={hot} the dreamed ball's mass starts at corr {c0:.2f} with the\n"
            f"  truth, is {c24:.2f} by step 24 and {cl:.2f} by step {late}."
            + (f" It falls below 0.5\n  at step {gone}."
               if gone is not None else " It never falls below 0.5.")
        )
    corr_line = (
        f"  It correlates with the speed its colour implies at r = "
        f"{_at(d['speed_corr'], late):.2f} (probe floor\n"
        f"  {_at(floor['speed_corr'], late):.2f} where that is defined).\n"
        if report["has_mass"] else ""
    )
    print(
        f"  At step {late} the dreamed ball moves at "
        f"{_at(d['speed_ratio_median'], late):.2f} x the speed the law gives it.\n"
        + corr_line +
        f"  The dream stays inside +-{a.speed_tol:.0%} of the law for "
        f"{d['speed_horizon_mean']:.0f} steps on average (the\n"
        f"  measuring probe's own floor is {floor['speed_horizon_mean']:.0f}).  "
        f"{d['wellformed_frac'].mean():.0%} of the dreamed frames still\n"
        f"  contain a ball of roughly the right size."
    )
    d0 = res[min(a.taus)]
    print(
        f"  At tau={min(a.taus)} (deterministic) the same numbers are "
        + (f"log-mass corr {_at(d0['logmass_corr'], late):.2f}, "
           if report["has_mass"] else "")
        + f"speed ratio {_at(d0['speed_ratio_median'], late):.2f}, "
        f"speed horizon {d0['speed_horizon_mean']:.0f}.\n"
        "  The gap between the two rows is the cost of sampling, which is the whole\n"
        "  finding: the model KNOWS the quantity is constant, and loses it anyway\n"
        "  as soon as it has to feed on its own draws."
    )
    print(f"\nartifacts in {out}")


if __name__ == "__main__":
    main()

"""v2 stage three: does the controller's skill depend on the ball's mass?

    python -m wm.eval_controller_v2 --out runs/ctrl_eval_v2

Stage three of v1 asked one question -- "does a controller trained entirely
inside M's dream work in the real box?" -- and answered it with one number per
policy. v2 asks a sharper one. The ball's colour now sets its mass, mass sets
its speed (``0.022 / m``, so a light ball is up to 4x faster than a heavy one),
and the design document's prediction is that a controller which has learned that
law should **react earlier for light balls**. A single averaged score cannot see
that: it mixes together two regimes with different difficulty *and* different
opportunity counts.

So everything here is computed **by mass tercile**, and the headline metric is
interceptions per floor visit.

Why that metric and not hits per episode
----------------------------------------
Two separate corrections, and both matter more in v2 than they did in v1.

* *Interceptions*, not contact frames. ``EVENT_PADDLE`` is a per-frame flag, so
  a policy that pins the ball against the paddle racks up hits without ever
  returning it. v1 caught its real-env-trained baseline doing exactly that.
* *Per floor visit*, not per episode. A light ball crosses the box in ~12
  frames, a heavy one in ~50, so in a fixed 200-step episode the light ball
  offers roughly four times as many chances. Per-episode counts would therefore
  make every policy look far better on light balls than on heavy ones purely
  because of the physics. Dividing by the number of chances is what makes the
  terciles comparable at all.

The three analyses beyond the tables
------------------------------------
1. **Hold-out colours.** Masses in ``[0.85, 1.2]`` were excluded from every
   dataset V and M ever saw. Re-running the evaluation with ``mass_only`` on
   that band asks whether C plays a never-seen colour as well as it plays a
   trained colour of similar speed -- which is why the comparison is against its
   own *medium* tercile, not against its overall average.
2. **The interaction term.** Regress the controller's drive
   ``logit(R) - logit(L)`` on approach frames against ``x_err``, the ball and
   paddle velocities and ``speed``, then add ``speed*x_err`` and
   ``speed*ball_vx``. If the controller merely tracks, the interactions add
   nothing; if its gain on positional error scales with how fast the ball is
   coming, they do. Reported as delta-R^2 plus standardised coefficients, so the
   size of a term is comparable across features with wildly different units.
3. **Reaction lead.** A direct behavioural version of the same question, with no
   regression in it: for each interception, count backwards from the contact
   frame the number of consecutive frames during which the paddle was already
   moving toward (or parked on) the spot where the contact ended up happening.
   That is "how many frames ahead of the ball the paddle had committed", and it
   is reported per tercile for the controller and for the oracle.

Nothing in here re-implements the rollout: it calls ``wm.eval_controller``'s
harness, which now records each episode's mass and scales the floor-visit band
by that episode's speed. The only thing this file owns is the slicing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from worldsim.render import save_gif

from .analyze import load_ckpt
from .controller import BaseController, load_controller
from .eval_controller import (
    DEFAULT_SEED_BASE,
    build_baselines,
    contact_runs,
    dream_play_gif,
    floor_visit_stats,
    run_real_episodes,
    _bootstrap_ci,
    _plt,
)
from .rnn import load_rnn

HOLDOUT_BAND = (0.85, 1.2)
TERCILES = ("light", "medium", "heavy")
N_BOOT = 4000


# ------------------------------------------------------------------- slicing


def mass_terciles(mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split episodes into light / medium / heavy by the 33rd and 67th centile.

    Data-defined rather than fixed cut points, so the three groups have equal
    size whatever the sampler produced. Every policy is evaluated on the SAME
    seeds and therefore the same masses, so one assignment serves the whole
    table and the comparison across rows stays paired.
    """
    q = np.percentile(mass, [100 / 3, 200 / 3])
    idx = np.digitize(mass, q)                       # 0, 1, 2
    return idx.astype(np.int64), q


def _ratio_ci(
    num: np.ndarray, den: np.ndarray, seed: int = 0
) -> Tuple[float, float, float]:
    """Bootstrap CI of ``sum(num) / sum(den)`` resampling EPISODES.

    The episode is the unit of independence; frames inside one are heavily
    correlated. Resampling the ratio of sums (rather than the mean of per-
    episode ratios) keeps episodes that offered no chance at all from having to
    be dropped or counted as zero, both of which bias the number.
    """
    if len(num) == 0 or den.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(N_BOOT, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-9)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def sign_agreement_per_episode(
    roll: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-episode approach-frame sign agreement, and the frame count behind it.

    v1 found this the most seed-stable statistic in the whole stage, so it is
    the one we split by mass. "Approach" is the ball descending in the lower
    half of the box -- the window in which the interception is actually decided.
    Frames where the ball is within 0.02 of the paddle are dropped as ties.

    The drive is ``logit(RIGHT) - logit(LEFT)`` for a linear controller, exactly
    as in v1. The baselines (stay / random / oracle) have no logits, so for them
    the drive is the ACTION itself (+1 right, -1 left) with STAY frames dropped
    from the denominator -- a STAY is not a directional decision, and counting
    it as a wrong one would make ``stay`` score 0.00 rather than "undefined".
    The two definitions are not perfectly interchangeable; the baseline rows are
    there as orientation, and the load-bearing comparisons are controller vs
    controller.

    Returned per episode (rather than pooled over frames) so it can be
    bootstrapped over the same unit as every other metric. Episodes with no
    approach frames come back as NaN and are dropped by the aggregator.
    """
    if "logits" in roll:
        lg = roll["logits"]                              # (E, T, 3)
        drive = lg[..., 2] - lg[..., 0]
        T = lg.shape[1]
    else:
        drive = roll["actions"].astype(np.float64) - 1.0
        T = roll["actions"].shape[1]
    s = roll["states"][:, :T]
    x_err = s[..., 0] - s[..., 4]
    mask = (
        (s[..., 3] < 0) & (s[..., 1] < 0.5)
        & (np.abs(x_err) > 0.02) & (drive != 0.0)
    )
    agree = (np.sign(drive) == np.sign(x_err)) & mask
    n = mask.sum(1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n > 0, agree.sum(1) / np.maximum(n, 1), np.nan)
    return frac, n


def reaction_lead(
    roll: Dict[str, np.ndarray], max_lead: int = 60, dead_zone: float = 0.03,
    paddle_w: float = 0.26,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frames of commitment before each interception.

    Returns ``(motion_lead, position_lead, episode_index)`` -- two answers to
    "how far ahead of the ball did the paddle get", because the obvious one
    turns out to be dominated by an artefact.

    Definition, kept deliberately crude so it cannot be argued with: take the
    first frame ``t_c`` of each run of contact frames and the ball's x there --
    that is where the interception happened. Walk backwards while the paddle was
    *already doing the right thing*, meaning either its velocity pointed toward
    that x, or it was within ``dead_zone`` of it (a paddle parked on target is
    committed, not idle). The length of that unbroken run is the lead.

    This is "how far ahead the paddle had committed", not a reaction time in the
    psychophysical sense: it says nothing about when the decision was made, only
    about how long the correct behaviour had been sustained. It is capped at
    ``max_lead`` because a paddle that never moves would otherwise score the
    whole episode.

    **The artefact.** That definition breaks the run on any frame where the
    paddle's velocity flips sign, and a bang-bang controller dithers: it
    alternates LEFT/RIGHT while sitting roughly on target. Measured, every
    trained controller scores 1-4 frames and ``stay`` scores 5, which says
    nothing about anticipation and everything about dither. So a second,
    dither-proof statistic is reported alongside it:

    ``position_lead`` -- the number of consecutive frames before contact during
    which the paddle was already **within half a paddle-width of where the
    interception ended up happening**. It asks about the paddle's *position*
    rather than its velocity, so oscillating in place does not reset it. This is
    the "how far ahead of the ball's landing did the paddle get" half of the
    question, and it is the one to read.
    """
    hits = roll["hits"] > 0
    s = roll["states"]
    motion: List[float] = []
    position: List[float] = []
    owner: List[int] = []
    half = paddle_w / 2.0
    for e in range(hits.shape[0]):
        t_contacts = np.flatnonzero(hits[e] & ~np.r_[False, hits[e][:-1]])
        for tc in t_contacts:
            x_land = s[e, tc, 0]
            k = 0
            while k < max_lead and tc - 1 - k >= 0:
                t = tc - 1 - k
                d = x_land - s[e, t, 4]
                v = s[e, t, 5]
                ok = abs(d) < dead_zone or (np.sign(v) == np.sign(d) and v != 0)
                if not ok:
                    break
                k += 1
            j = 0
            while j < max_lead and tc - 1 - j >= 0:
                if abs(x_land - s[e, tc - 1 - j, 4]) >= half:
                    break
                j += 1
            motion.append(float(k))
            position.append(float(j))
            owner.append(e)
    return (
        np.asarray(motion, np.float64),
        np.asarray(position, np.float64),
        np.asarray(owner, np.int64),
    )


# ----------------------------------------------------------- the by-mass table


def by_mass_rows(name: str, roll: Dict[str, np.ndarray], tercile: np.ndarray) -> Dict:
    """Every metric, overall and for each tercile, with bootstrap CIs."""
    hits = roll["hits"].sum(1)
    runs = contact_runs(roll["hits"])
    fv = floor_visit_stats(roll["states"], speed=roll.get("speed"))
    visits, gaps, gap_ep = fv["floor_visits"], fv["gap_at_floor"], fv["gap_episode"]
    sa, sa_n = sign_agreement_per_episode(roll)
    lead, pos_lead, lead_ep = reaction_lead(roll)

    def block(sel: np.ndarray) -> Dict:
        g = np.isin(gap_ep, np.flatnonzero(sel))
        ipv, lo, hi = _ratio_ci(runs[sel], visits[sel])
        hpv, _, _ = _ratio_ci(hits[sel], visits[sel])
        good = sel & ~np.isnan(sa)
        # Frame-weighted, so one episode with three approach frames does not
        # count as much as one with forty.
        sa_w = (
            float((sa[good] * sa_n[good]).sum() / max(sa_n[good].sum(), 1))
            if good.any() else float("nan")
        )
        sa_mean, sa_lo, sa_hi = _ratio_ci(sa[good] * sa_n[good], sa_n[good])
        lsel = np.isin(lead_ep, np.flatnonzero(sel))
        ll, pl = lead[lsel], pos_lead[lsel]
        return {
            "n_episodes": int(sel.sum()),
            "hits_per_episode": float(hits[sel].mean()),
            "interceptions_per_episode": float(runs[sel].mean()),
            "interceptions_ci95": list(_bootstrap_ci(runs[sel], n_boot=N_BOOT)),
            "floor_visits_per_episode": float(visits[sel].mean()),
            "hits_per_visit": hpv,
            "interceptions_per_visit": ipv,
            "interceptions_per_visit_ci95": [lo, hi],
            "mean_gap_at_floor": float(gaps[g].mean()) if g.any() else float("nan"),
            "sign_agreement_approach": sa_w,
            "sign_agreement_approach_ci95": [sa_lo, sa_hi],
            "approach_frames": int(sa_n[sel].sum()),
            "reaction_lead_frames": float(ll.mean()) if len(ll) else float("nan"),
            "position_lead_frames": float(pl.mean()) if len(pl) else float("nan"),
            "n_interceptions": int(len(ll)),
        }

    out = {"name": name, "overall": block(np.ones(len(hits), bool))}
    for k, t in enumerate(TERCILES):
        out[t] = block(tercile == k)
    return out


# --------------------------------------------------------- decision analysis


def decision_analysis(roll: Dict[str, np.ndarray]) -> Dict:
    """Drive ~ features, with and without the two speed interactions.

    Restricted to approach frames: everywhere else the controller's output is
    not deciding anything we care about, and including those frames just dilutes
    whatever structure exists. All regressors are standardised before the fit so
    the coefficients are directly comparable (a coefficient is then "standard
    deviations of drive per standard deviation of feature"); ``speed`` spans
    0.011 to 0.044 while ``x_err`` spans about a unit, so raw coefficients would
    be meaningless side by side.
    """
    if "logits" not in roll:
        return {}
    lg = roll["logits"]
    s = roll["states"][:, : lg.shape[1]]
    drive = lg[..., 2] - lg[..., 0]
    ball_x, ball_y, vx, vy, pad_x, pad_vx = [s[..., i] for i in range(6)]
    speed = np.broadcast_to(roll["speed"][:, None], ball_x.shape)
    x_err = ball_x - pad_x
    m = (vy < 0) & (ball_y < 0.5)
    if m.sum() < 50:
        return {}

    y = drive[m]
    base_names = ("x_err", "ball_vx", "ball_vy", "paddle_vx", "speed")
    base = np.stack([x_err[m], vx[m], vy[m], pad_vx[m], speed[m]], 1)
    inter_names = ("speed*x_err", "speed*ball_vx")
    inter = np.stack([(speed * x_err)[m], (speed * vx)[m]], 1)

    def fit(names: Sequence[str], X: np.ndarray) -> Dict:
        Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
        ys = (y - y.mean()) / max(y.std(), 1e-12)
        A = np.concatenate([np.ones((len(Xs), 1)), Xs], 1)
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        r2 = 1.0 - (ys - A @ coef).var() / max(ys.var(), 1e-12)
        return {
            "r2": float(r2),
            "beta": {n: float(c) for n, c in zip(names, coef[1:])},
        }

    no_i = fit(base_names, base)
    with_i = fit(base_names + inter_names, np.concatenate([base, inter], 1))
    return {
        "n_frames": int(m.sum()),
        "without_interactions": no_i,
        "with_interactions": with_i,
        "delta_r2": float(with_i["r2"] - no_i["r2"]),
        "drive_std": float(y.std()),
    }


# --------------------------------------------------------------------- plots


def plot_by_mass(
    rows: Sequence[Dict], key: str, ylabel: str, title: str, out: Path,
    oracle_name: str = "oracle", chance: Optional[float] = None,
) -> Path:
    """Grouped bars: one cluster per tercile, one bar per controller.

    The oracle is drawn as a per-tercile horizontal reference segment rather
    than as another bar, because it is a ceiling, not a competitor -- and its
    ceiling is genuinely different in each tercile, which a single global line
    would hide.
    """
    plt = _plt()
    oracle = next((r for r in rows if r["name"] == oracle_name), None)
    bars = [r for r in rows if r["name"] != oracle_name]
    n = len(bars)
    x = np.arange(len(TERCILES))
    w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(2.5 + 1.4 * n, 4.6))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(bars):
        vals = [r[t][key] for t in TERCILES]
        ci = [r[t].get(key + "_ci95") for t in TERCILES]
        err = None
        if all(c is not None and not np.isnan(c[0]) for c in ci):
            err = np.array([[v - c[0], c[1] - v] for v, c in zip(vals, ci)]).T
        color = {"stay": "#aaaaaa", "random": "#777777"}.get(
            r["name"], cmap(0.12 + 0.72 * i / max(n - 1, 1))
        )
        ax.bar(x - 0.4 + w * (i + 0.5), vals, width=w * 0.92, yerr=err,
               capsize=2, color=color, label=r["name"])
    if oracle is not None:
        for k, t in enumerate(TERCILES):
            v = oracle[t][key]
            ax.plot([k - 0.42, k + 0.42], [v, v], c="#2a9d4a", lw=2.2,
                    label="oracle (ceiling)" if k == 0 else None)
    if chance is not None:
        ax.axhline(chance, ls="--", c="k", lw=1)
        ax.text(len(TERCILES) - 0.55, chance + 0.005, "chance", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(fast)" if t == "light" else
                        f"{t}\n(slow)" if t == "heavy" else t for t in TERCILES])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Headroom so the legend never sits on top of a bar it is labelling.
    top = max(ax.get_ylim()[1], 1e-6)
    ax.set_ylim(ax.get_ylim()[0], top * 1.32)
    ax.legend(fontsize=7, ncol=3, loc="upper center", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ the runs


class Spec:
    """A policy plus the V and M it reads the world through.

    ``ctrl_v1_on_v2`` is the whole reason this is not just a list of
    controllers: it must be driven by the *v1* encoder and the *v1* dynamics
    model, because that is the stack it was trained on. Pairing each policy with
    its own (V, M) is the only way the transfer baseline is honest.
    """

    def __init__(self, name: str, ctrl: BaseController, vae, rnn):
        self.name, self.ctrl, self.vae, self.rnn = name, ctrl, vae, rnn


def run_all(specs: Sequence[Spec], episodes: int, steps: int, seed_base: int,
            env: Dict, device: str) -> Dict[str, Dict[str, np.ndarray]]:
    rolls = {}
    for sp in specs:
        t = time.time()
        rolls[sp.name] = run_real_episodes(
            sp.ctrl, sp.vae, sp.rnn, episodes=episodes, steps=steps,
            seed_base=seed_base, device=device, record_logits=True, **env,
        )
        print(f"  {sp.name:<22} "
              f"hits/ep {rolls[sp.name]['hits'].sum(1).mean():.2f}  "
              f"({time.time() - t:.0f}s)", flush=True)
    return rolls


# -------------------------------------------------------------------- report


def _fmt(v: float, nd: int = 2) -> str:
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def table(rows: Sequence[Dict], key: str, nd: int = 2, ci: bool = False) -> List[str]:
    head = "| controller | overall | " + " | ".join(TERCILES) + " |"
    lines = [head, "|---" * (len(TERCILES) + 2) + "|"]
    for r in rows:
        cells = []
        for t in ("overall",) + TERCILES:
            v = _fmt(r[t][key], nd)
            c = r[t].get(key + "_ci95")
            if ci and c is not None and not np.isnan(c[0]):
                v += f" [{c[0]:.2f}, {c[1]:.2f}]"
            cells.append(v)
        lines.append(f"| `{r['name']}` | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vae", default="runs/vae_v2/vae.pt")
    p.add_argument("--rnn", default="runs/rnn_v2/rnn.pt")
    p.add_argument("--vae-v1", default="runs/vae_b1/vae.pt")
    p.add_argument("--rnn-v1", default="runs/rnn_v1/rnn.pt")
    p.add_argument("--ctrl-v1", default="runs/ctrl_v1/controller.pt")
    p.add_argument("--runs", nargs="*",
                   default=["runs/ctrl_v2", "runs/ctrl_v2_mix",
                            "runs/ctrl_v2_z_only", "runs/ctrl_v2_real"])
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--holdout-episodes", type=int, default=60)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    p.add_argument("--holdout-seed-base", type=int, default=6000)
    p.add_argument("--ball-radius", type=float, default=0.08)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="runs/ctrl_eval_v2")
    p.add_argument("--dream-roots", nargs="*",
                   default=["data/v2/train", "data/v2/train_mix"])
    p.add_argument("--no-gifs", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    vae, _, _ = load_ckpt(a.vae, a.device)
    rnn, _ = load_rnn(a.rnn, a.device)

    specs: List[Spec] = [Spec(c.name, c, vae, rnn) for c in build_baselines()]
    for r in a.runs:
        ck = Path(r) / "controller.pt"
        if not ck.exists():
            print(f"  (skipping {r}: no controller.pt)")
            continue
        nm = Path(r).name
        specs.append(Spec(nm, load_controller(ck, name=nm), vae, rnn))
        if nm == "ctrl_v2":
            # CMA-ES's final distribution mean: what the DREAM alone would have
            # chosen, with zero real episodes consulted. The gap between this
            # row and the selected one is the price of the periodic real check.
            specs.append(Spec(
                "ctrl_v2_lastdream",
                load_controller(ck, name="ctrl_v2_lastdream",
                                which="params_last_dream"),
                vae, rnn))

    # The transfer baseline: v1's controller, on v1's V and M, dropped into the
    # v2 world. It has never seen a coloured ball or a fast one, and its
    # encoder cannot even represent the colour.
    if Path(a.ctrl_v1).exists():
        v1_vae, _, _ = load_ckpt(a.vae_v1, a.device)
        v1_rnn, _ = load_rnn(a.rnn_v1, a.device)
        specs.append(Spec("ctrl_v1_on_v2",
                          load_controller(a.ctrl_v1, name="ctrl_v1_on_v2"),
                          v1_vae, v1_rnn))

    base_env = {"ball_radius": a.ball_radius, "mass_from_color": True}
    in_env = {**base_env, "mass_holdout": HOLDOUT_BAND}
    ho_env = {**base_env, "mass_only": HOLDOUT_BAND}

    print(f"in-distribution: {a.episodes} episodes x {a.steps} steps, "
          f"seeds {a.seed_base}.. , masses outside {HOLDOUT_BAND}")
    rolls = run_all(specs, a.episodes, a.steps, a.seed_base, in_env, a.device)
    mass = rolls[specs[0].name]["mass"]
    terc, cuts = mass_terciles(mass)
    rows = [by_mass_rows(sp.name, rolls[sp.name], terc) for sp in specs]

    print(f"\nheld-out band {HOLDOUT_BAND}: {a.holdout_episodes} episodes, "
          f"seeds {a.holdout_seed_base}..")
    ho_specs = [sp for sp in specs
                if sp.name in ("stay", "oracle", "ctrl_v2", "ctrl_v1_on_v2")]
    ho_rolls = run_all(ho_specs, a.holdout_episodes, a.steps,
                       a.holdout_seed_base, ho_env, a.device)
    # One "tercile" only -- the band is narrow by construction, so splitting it
    # would just be noise. Everything lands in `overall`.
    ho_terc = np.zeros(a.holdout_episodes, np.int64)
    ho_rows = [by_mass_rows(sp.name, ho_rolls[sp.name], ho_terc) for sp in ho_specs]

    decisions = {
        nm: decision_analysis(rolls[nm])
        for nm in ("ctrl_v2", "ctrl_v2_z_only", "ctrl_v1_on_v2")
        if nm in rolls
    }

    plot_by_mass(rows, "interceptions_per_visit",
                 "interceptions per floor visit",
                 "skill by mass tercile (95% bootstrap CI)",
                 out / "interceptions_per_visit_by_mass.png")
    plot_by_mass(rows, "sign_agreement_approach",
                 "fraction of approach frames moving toward the ball",
                 "does it move the right way, by mass?",
                 out / "sign_agreement_by_mass.png", chance=0.5)

    if not a.no_gifs and "ctrl_v2" in rolls:
        render_gifs(rolls, specs, vae, rnn, out, a)

    write_report(rows, ho_rows, decisions, cuts, mass, a, out,
                 round(time.time() - t0, 1))
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------- gifs


def render_gifs(rolls, specs, vae, rnn, out: Path, a) -> None:
    """One light episode, one heavy episode, and one dream -- all of ctrl_v2.

    The episode seeds are picked out of the evaluation set itself rather than
    drawn fresh, so the GIF is literally one of the episodes in the table above
    it.
    """
    from .dream_env import load_start_pool

    ctrl = next(sp for sp in specs if sp.name == "ctrl_v2").ctrl
    mass = rolls["ctrl_v2"]["mass"]
    env = {"ball_radius": a.ball_radius, "mass_from_color": True,
           "mass_holdout": HOLDOUT_BAND}
    for tag, ep in (("light", int(mass.argmin())), ("heavy", int(mass.argmax()))):
        roll = run_real_episodes(
            ctrl, vae, rnn, episodes=1, steps=a.steps,
            seed_base=a.seed_base + ep, device=a.device, record_frames=1, **env,
        )
        save_gif(roll["frames"][0], out / f"real_play_ctrl_v2_{tag}.gif",
                 fps=20, scale=4)
        print(f"  {tag}: seed {a.seed_base + ep}, m = {roll['mass'][0]:.2f}, "
              f"speed {roll['speed'][0]:.3f}, "
              f"{int(contact_runs(roll['hits'])[0])} interceptions")

    pool = load_start_pool(a.dream_roots, warmup=8)
    rng = np.random.default_rng(0)
    # Light balls only, so the dream's speed is visible at all.
    starts = pool.sample_mass_range(16, rng, 0.5, 0.75)
    dream_play_gif(ctrl, rnn, vae, pool, out / "dream_play_ctrl_v2.gif",
                   steps=a.steps, temperature=1.0, seed=a.seed_base,
                   device=a.device, starts=starts)


# ------------------------------------------------------------------- writing


def write_report(rows, ho_rows, decisions, cuts, mass, a, out: Path,
                 wall: float) -> None:
    payload = {
        "in_distribution": rows,
        "holdout": ho_rows,
        "decisions": decisions,
        "tercile_cuts": [float(c) for c in cuts],
        "mass_range": [float(mass.min()), float(mass.max())],
        "holdout_band": list(HOLDOUT_BAND),
        "episodes": a.episodes,
        "holdout_episodes": a.holdout_episodes,
        "steps": a.steps,
        "seed_base": a.seed_base,
        "holdout_seed_base": a.holdout_seed_base,
        "wall_clock_s": wall,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2))

    L = [
        "# v2 stage three (C) — real-environment evaluation by mass",
        "",
        f"{a.episodes} episodes x {a.steps} steps, seeds "
        f"{a.seed_base}..{a.seed_base + a.episodes - 1}, identical starts for "
        f"every row. Masses log-uniform in [0.5, 2.0] **excluding the held-out "
        f"band {list(HOLDOUT_BAND)}**; observed range "
        f"[{mass.min():.2f}, {mass.max():.2f}], tercile cuts at "
        f"m = {cuts[0]:.2f} and {cuts[1]:.2f}.",
        "",
        "Light = fast (speed 0.022/m, up to 0.044/frame); heavy = slow. A light "
        "ball reaches the floor several times more often than a heavy one, so "
        "per-episode counts are not comparable across terciles and "
        "**interceptions per floor visit** is the metric to read.",
        "",
        "## Interceptions per floor visit (the mass-fair skill metric)",
        "",
    ]
    L += table(rows, "interceptions_per_visit", ci=True)
    L += ["", "## Floor visits per episode (the number of chances physics offered)", ""]
    L += table(rows, "floor_visits_per_episode")
    L += ["", "## Interceptions per episode", ""]
    L += table(rows, "interceptions_per_episode")
    L += ["", "## Hits (contact frames) per episode", ""]
    L += table(rows, "hits_per_episode")
    L += ["", "## Mean gap at closest approach (world units; lower is better)", ""]
    L += table(rows, "mean_gap_at_floor", nd=3)
    L += ["", "## Approach-frame sign agreement (chance 0.50)", ""]
    L += table(rows, "sign_agreement_approach", nd=3, ci=True)
    L += ["", "## Reaction lead A: consecutive frames of correctly-directed "
          "paddle motion before each interception", "",
          "Dominated by dithering (a bang-bang paddle flips sign in place and "
          "resets the run). Reported for completeness; read table B.", ""]
    L += table(rows, "reaction_lead_frames", nd=1)
    L += ["", "## Reaction lead B: frames the paddle was already within half a "
          "paddle-width of the interception point", "",
          "The dither-proof version of \"how far ahead of the ball's landing did "
          "the paddle get\".", ""]
    L += table(rows, "position_lead_frames", nd=1)

    L += [
        "",
        f"## Held-out colour band {list(HOLDOUT_BAND)}",
        "",
        f"{a.holdout_episodes} episodes, seeds {a.holdout_seed_base}.., "
        "`mass_only` on the band no model in the stack has ever seen move. The "
        "right comparison is against the **medium** tercile above, whose speeds "
        "bracket this band — not against the overall average.",
        "",
        "| controller | interceptions/visit | floor visits/ep | gap at floor | "
        "sign agreement | position lead |",
        "|---|---|---|---|---|---|",
    ]
    for r in ho_rows:
        o = r["overall"]
        L.append(
            f"| `{r['name']}` | {_fmt(o['interceptions_per_visit'])} "
            f"[{_fmt(o['interceptions_per_visit_ci95'][0])}, "
            f"{_fmt(o['interceptions_per_visit_ci95'][1])}] | "
            f"{_fmt(o['floor_visits_per_episode'])} | "
            f"{_fmt(o['mean_gap_at_floor'], 3)} | "
            f"{_fmt(o['sign_agreement_approach'], 3)} | "
            f"{_fmt(o['position_lead_frames'], 1)} |"
        )

    if decisions:
        L += [
            "",
            "## Decision analysis: is there a speed interaction?",
            "",
            "`logit(RIGHT) − logit(LEFT)` on approach frames, regressed on "
            "standardised features. The base model is `x_err, ball_vx, "
            "ball_vy, paddle_vx, speed`; the second adds `speed·x_err` and "
            "`speed·ball_vx`. A mass-aware controller should gain R² from them.",
            "",
            "| controller | R² base | R² + interactions | ΔR² | frames |",
            "|---|---|---|---|---|",
        ]
        for nm, d in decisions.items():
            if not d:
                continue
            L.append(
                f"| `{nm}` | {d['without_interactions']['r2']:.4f} | "
                f"{d['with_interactions']['r2']:.4f} | "
                f"{d['delta_r2']:+.4f} | {d['n_frames']} |"
            )
        L += ["", "Standardised coefficients of the full model:", "",
              "| controller | " + " | ".join(
                  list(next(iter(decisions.values()))["with_interactions"]["beta"])
              ) + " |",
              "|---" * (1 + len(next(iter(decisions.values()))["with_interactions"]["beta"])) + "|"]
        for nm, d in decisions.items():
            if not d:
                continue
            b = d["with_interactions"]["beta"]
            L.append(f"| `{nm}` | " + " | ".join(f"{v:+.3f}" for v in b.values()) + " |")

    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

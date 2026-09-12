"""Does the controller's DECISION depend on the ball's colour? An intervention.

    python -m wm.eval_ctrl_intervention --out runs/ctrl_intervention

Stage three's design document asks "does C use mass?" and `README_C2.md` tried
to answer it with a regression: add ``speed*x_err`` to a linear model of the
controller's drive and see whether R^2 goes up. That test **gave a false
positive on the v1 controller** -- a policy whose encoder cannot represent
colour at all scored a non-zero interaction -- so it cannot be evidence of
anything. The reason is the same one that motivates every other experiment in
this project: in an observational regression, ``speed`` is confounded with
everything else about the episode. Fast balls arrive at different angles, from
different heights, with the paddle in different places. A controller that reads
only position and velocity will still show a speed interaction, because speed
*predicts* the situations it is in.

The fix is to intervene rather than observe.

THE EXPERIMENT
--------------
Take a frame on which the controller is actually deciding something -- the ball
descending in the lower half of the box -- and reconstruct its inputs twice:

  (i)  **as-is**: re-render the 9 recorded frames ``t-8 .. t`` from state,
       encode them, teacher-force M over the first 8 with the recorded actions
       to get ``h_pre`` at ``t``, and read the drive ``logit(R) - logit(L)``.
  (ii) **repainted**: exactly the same, except every one of those 9 frames is
       re-rendered with the colour of a DIFFERENT mass ``m1``. Ball positions,
       paddle positions and actions are bit-identical; only the hue moves.

The difference between the two drives is caused by the colour and by nothing
else, because nothing else changed. That is the whole point: this is a
``do(colour = c)`` operator, not a conditional.

Four numbers per controller:

  (a) **effect size** ``mean |Δdrive| / std(drive)``. Normalised because a
      linear controller's logits have no natural scale -- CMA-ES is free to
      multiply every weight by 10 and change nothing about the policy, so a raw
      Δ is uninterpretable across rows.
  (b) **flip fraction**: how often ``argmax`` changes. This is the behavioural
      version -- a large Δdrive that never crosses a decision boundary changes
      no actions.
  (c) **direction**: is the change in the *useful* direction? A light ball is
      fast, so a controller that has learned the law should, on seeing the ball
      repainted lighter, push harder toward the side the ball is on. Regress
      ``Δdrive`` on ``(log m1 - log m0) * sign(x_err)``; the law predicts a
      **negative** coefficient (lighter = smaller log m1 = more drive toward the
      ball). Bootstrap CI resampling EPISODES, since frames within an episode
      are not independent.
  (d) **null control**: the same pipeline with ``m1 = m0``. The renderer is a
      pure function of (ball, paddle, colour), so this must come back exactly
      zero; it is a check on the plumbing (that the as-is and repaint paths are
      the same code), not on stochastic noise, and it is reported so a
      non-zero effect cannot be blamed on a round-trip artefact.

THE ROWS THAT MAKE IT READABLE
------------------------------
* ``oracle`` reads the true state, which repainting does not touch, so its Δ is
  identically 0 by construction. It is the "a policy that provably ignores the
  pixels scores 0" anchor.
* ``ctrl_v1_on_v2`` runs on the v1 VAE, which never saw a coloured ball. Its Δ
  is the **leak**: whatever survives of a colour change through a colour-blind
  encoder. It is not expected to be exactly zero (the ramp changes luminance,
  and the encoder is not literally invariant), and measuring it is what makes
  the other rows interpretable.
* ``m1 = 1.0`` lands inside the held-out band [0.85, 1.2] -- a colour no model
  in the stack has ever seen. Flagged in the output.

WHERE THE FRAMES COME FROM
--------------------------
By default, from real episodes played by the controller under test, so each row
is measured on its own on-policy state distribution -- the states where its
decisions actually matter. ``--frames val_mix`` instead uses one shared set of
frames from ``data/v2/val_mix`` for every row, which makes the numbers directly
comparable at the cost of being off-policy. Both are run in the log; they agree
on the ordering, which is the thing worth knowing.

A CAVEAT ON h
-------------
The real rollout starts M from a cold hidden state at t=0 and runs it forward
for the whole episode; here it is warm-started from 8 frames. So the
reconstructed ``h_pre`` is not bit-identical to the one the rollout used, and
the reconstructed as-is drive is not bit-identical to the recorded logit either.
That is fine and is measured (``drive_reconstruction_r`` in the report): both
arms of the intervention use the identical procedure, so the *contrast* is
exact even where the level is not.
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

from .controller import BaseController, load_controller
from .eval_causal_v2 import encode_frames, recolor_frames
from .eval_controller import DEFAULT_SEED_BASE, make_box_cfg, run_real_episodes
from .eval_controller_v2 import (
    HOLDOUT_BAND,
    StackCache,
    parse_overrides,
    stack_for_run,
    _plt,
)
from .seq_data import episode_arrays

WARMUP = 8                       # frames of teacher forcing behind each probe frame
REPAINT_MASSES = (0.5, 1.0, 2.0)
N_BOOT = 4000


# --------------------------------------------------------------- frame picking


def approach_frames(
    states: np.ndarray, actions: np.ndarray, n_want: int, seed: int = 0,
    warmup: int = WARMUP, min_x_err: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick ``(episode, t)`` pairs where the controller is deciding something.

    "Approach" is the same window every other stage-three statistic uses: the
    ball descending (``vy < 0``) in the lower half of the box (``y < 0.5``).
    Frames whose ``|x_err|`` is below ``min_x_err`` are dropped, because
    ``sign(x_err)`` is the axis the direction regression projects onto and it is
    meaningless when the ball is directly above the paddle.

    Sampling is **stratified by episode** -- the same number of frames from each
    eligible episode, as far as supply allows -- so one long episode with a slow
    ball cannot contribute a third of the sample and drag every average toward
    its own mass.
    """
    rng = np.random.default_rng(seed)
    E, T = actions.shape
    ok_t = np.arange(warmup, T)
    per_ep: List[np.ndarray] = []
    for e in range(E):
        s = states[e, ok_t]
        m = (s[:, 3] < 0) & (s[:, 1] < 0.5) & (np.abs(s[:, 0] - s[:, 4]) > min_x_err)
        per_ep.append(ok_t[m])
    have = np.array([len(x) for x in per_ep])
    live = np.flatnonzero(have > 0)
    if live.size == 0:
        raise ValueError("no approach frames found")
    # Round-robin over episodes, each in its own shuffled order, until we have
    # enough. Deterministic given the seed.
    order = [rng.permutation(per_ep[e]) for e in live]
    picks: List[Tuple[int, int]] = []
    k = 0
    while len(picks) < n_want and any(k < len(o) for o in order):
        for e, o in zip(live, order):
            if k < len(o):
                picks.append((int(e), int(o[k])))
                if len(picks) >= n_want:
                    break
        k += 1
    arr = np.asarray(picks, np.int64)
    return arr[:, 0], arr[:, 1]


# ------------------------------------------------------------ the drive itself


@torch.no_grad()
def drives(
    ctrl: BaseController, vae, rnn, states: np.ndarray, actions: np.ndarray,
    ep: np.ndarray, t: np.ndarray, cfg: BoxConfig,
    mass: Optional[float | np.ndarray] = None,
    device: str = "cpu", batch: int = 64,
) -> np.ndarray:
    """The controller's drive on each probe frame, with the ball repainted.

    ``mass=None`` renders each frame with the mass it really had (the as-is
    arm); a float repaints every frame of every window with that one mass; an
    ``(n,)`` array gives each window its own repaint mass, which is how the null
    control is built (window *i* repainted with its own ``m0``).

    For a ``LinearController`` the drive is ``logit(RIGHT) - logit(LEFT)``, the
    same scalar the rest of stage three calls "drive". For the oracle -- which
    reads the true state and never looks at a pixel -- it is the chosen action
    as +-1, which is the only drive it has; repainting cannot move it, and the
    row exists to show that the harness agrees.
    """
    n = len(ep)
    out = np.zeros(n, np.float64)
    per_window = None if mass is None or np.isscalar(mass) \
        else np.asarray(mass, np.float64)
    eye = torch.eye(3, device=device)
    for i0 in range(0, n, batch):
        sl = slice(i0, min(i0 + batch, n))
        e, tt = ep[sl], t[sl]
        # (B, WARMUP + 1, S): the window t-WARMUP .. t inclusive.
        idx = tt[:, None] - WARMUP + np.arange(WARMUP + 1)[None, :]
        win = states[e[:, None], idx]                         # (B, W+1, S)
        if isinstance(ctrl, BaseController) and ctrl.uses_true_state:
            a = np.asarray(ctrl.act(np.zeros((len(e), 1)), np.zeros((len(e), 1)),
                                    state=win[:, -1]), np.int64)
            out[sl] = a.astype(np.float64) - 1.0
            continue
        ms = ([mass] * len(win) if per_window is None
              else list(per_window[sl]))
        frames = np.stack([recolor_frames(w, cfg, mass=m)
                           for w, m in zip(win, ms)])
        mu = encode_frames(vae, frames, device)               # (B, W+1, z)
        a_w = actions[e[:, None], idx[:, :WARMUP]]            # (B, W)
        z_t = torch.from_numpy(mu[:, :WARMUP].astype(np.float32)).to(device)
        a_t = eye[torch.from_numpy(a_w).long().to(device)]
        _, h = rnn(z_t, a_t)                                  # h_pre at time t
        h_pre = h[0][0].cpu().numpy()
        lg = ctrl.logits(mu[:, WARMUP], h_pre)                # (B, 3)
        out[sl] = lg[:, 2] - lg[:, 0]
    return out


# ------------------------------------------------------------------ statistics


def _boot_slope(x: np.ndarray, y: np.ndarray, group: np.ndarray,
                seed: int = 0) -> Tuple[float, float, float]:
    """OLS slope of ``y ~ x`` with a CI from resampling ``group`` (episodes).

    A per-frame bootstrap would be wrong by an order of magnitude here: 300
    frames drawn from 40 episodes carry nothing like 300 independent pieces of
    information, and the cluster bootstrap is the cheapest honest correction.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, group = x[ok], y[ok], group[ok]
    if len(x) < 8 or x.std() < 1e-12:
        return float("nan"), float("nan"), float("nan")

    def slope_of(xb: np.ndarray, yb: np.ndarray) -> float:
        # The closed form rather than ``polyfit``: 4000 resamples x 4 conditions
        # x 6 policies is 100k fits, and polyfit's Vandermonde solve is both
        # slower and noisy about conditioning on degenerate clusters.
        v = xb.var()
        return float(((xb - xb.mean()) * (yb - yb.mean())).mean() / v) \
            if v > 1e-24 else np.nan

    slope = slope_of(x, y)
    gs = np.unique(group)
    members = [np.flatnonzero(group == g) for g in gs]
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([members[j]
                              for j in rng.integers(0, len(gs), len(gs))])
        draws[b] = slope_of(x[idx], y[idx])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return slope, float(lo), float(hi)


def intervention_stats(
    base: np.ndarray, rep: np.ndarray, m0: np.ndarray, m1: float,
    x_err: np.ndarray, ep: np.ndarray, seed: int = 0,
) -> Dict:
    """Effect size, flip fraction and direction for one repaint mass."""
    d = rep - base
    sd = float(base.std())
    flips = np.sign(np.where(np.abs(base) < 1e-12, 0.0, base)) != np.sign(
        np.where(np.abs(rep) < 1e-12, 0.0, rep))
    # `x` is the regressor the direction hypothesis names: how much lighter the
    # repaint made the ball, signed by which side the ball is on.
    x = (np.log(m1) - np.log(m0)) * np.sign(x_err)
    # Δ is expressed in units of the controller's own drive std before the fit.
    # A linear policy's logits have no scale -- doubling W doubles every drive
    # and changes no action -- so a raw slope is not comparable across rows.
    slope, lo, hi = _boot_slope(x, d / max(sd, 1e-12), ep, seed=seed)
    return {
        "m1": float(m1),
        "in_holdout_band": bool(HOLDOUT_BAND[0] <= m1 <= HOLDOUT_BAND[1]),
        "n_frames": int(len(d)),
        "drive_std": sd,
        "mean_abs_delta": float(np.abs(d).mean()),
        "effect_size": float(np.abs(d).mean() / max(sd, 1e-12)),
        "mean_delta": float(d.mean()),
        "flip_fraction": float(flips.mean()),
        "direction_beta": slope,
        "direction_ci95": [lo, hi],
    }


# --------------------------------------------------------------------- one row


def run_controller(
    name: str, ctrl: BaseController, vae, rnn, cfg: BoxConfig, a,
    shared: Optional[Dict] = None, device: str = "cpu",
) -> Dict:
    """Every statistic for one policy. ``shared`` pins a common frame set."""
    t0 = time.time()
    if shared is None:
        roll = run_real_episodes(
            ctrl, vae, rnn, episodes=a.episodes, steps=a.steps,
            seed_base=a.seed_base, device=device, record_logits=True,
            ball_radius=a.ball_radius, mass_from_color=True,
            mass_holdout=HOLDOUT_BAND,
        )
        states, actions = roll["states"], roll["actions"]
        recorded = roll.get("logits")
        source = f"on-policy, {a.episodes} eps, seeds {a.seed_base}.."
    else:
        states, actions, recorded = shared["states"], shared["actions"], None
        source = shared["source"]

    ep, t = approach_frames(states, actions, a.n_frames, seed=a.seed)
    m0 = states[ep, 0, 6]
    x_err = states[ep, t, 0] - states[ep, t, 4]

    base = drives(ctrl, vae, rnn, states, actions, ep, t, cfg, None, device)
    row: Dict = {
        "name": name,
        "source": source,
        "n_frames": int(len(ep)),
        "n_episodes": int(len(np.unique(ep))),
        "mass_range": [float(m0.min()), float(m0.max())],
        "drive_std": float(base.std()),
        "repaints": {},
    }
    if recorded is not None:
        # How faithful is the 8-frame warm-start reconstruction of h_pre? Not
        # load-bearing (both arms share it) but the reader should see it.
        rec = recorded[ep, t, 2] - recorded[ep, t, 0]
        row["drive_reconstruction_r"] = (
            float(np.corrcoef(rec, base)[0, 1]) if base.std() > 1e-12 else 1.0
        )

    # The null control first, so a broken pipeline shows up before any of the
    # real conditions are interpreted.
    for tag, m1 in [("null", None)] + [(f"m{m:g}", m) for m in REPAINT_MASSES]:
        if tag == "null":
            # m1 = m0, window by window. ``recolor_frames`` reads ``s[6]`` when
            # mass is None and the explicit value when it is given, so this is
            # the same render reached by a *different code path* -- which is
            # exactly the plumbing the null control is there to check.
            rep = drives(ctrl, vae, rnn, states, actions, ep, t, cfg, m0, device)
            st = intervention_stats(base, rep, m0, 1.0, x_err, ep, seed=a.seed)
            st.update(m1=None, in_holdout_band=False,
                      direction_beta=float("nan"), direction_ci95=[float("nan")] * 2)
        else:
            rep = drives(ctrl, vae, rnn, states, actions, ep, t, cfg, m1, device)
            st = intervention_stats(base, rep, m0, m1, x_err, ep, seed=a.seed)
        row["repaints"][tag] = st
        print(f"    {tag:<6} |Δ|/sd {st['effect_size']:.4f}  "
              f"flips {st['flip_fraction']:.3f}  "
              f"β {st['direction_beta']:+.3f}", flush=True)

    # The headline "does colour matter at all" number pools the three real
    # repaints; the per-mass rows are kept because the answer is not the same
    # for a mass inside the held-out band as for one outside it.
    real = [v for k, v in row["repaints"].items() if k != "null"]
    row["effect_size_mean"] = float(np.mean([v["effect_size"] for v in real]))
    row["flip_fraction_mean"] = float(np.mean([v["flip_fraction"] for v in real]))
    row["null_effect_size"] = row["repaints"]["null"]["effect_size"]
    row["null_flip_fraction"] = row["repaints"]["null"]["flip_fraction"]
    row["wall_s"] = round(time.time() - t0, 1)
    return row


# ------------------------------------------------------------------------ plot


def plot_intervention(rows: Sequence[Dict], path: Path) -> Path:
    """Three panels: effect size, flip fraction, and the direction coefficient.

    The null bar is drawn for every row rather than once, because "the null is
    zero" is a claim about each pipeline separately -- a different encoder could
    in principle have a different round-trip error. In practice they all come
    back *exactly* zero, so the grey bars have no height and the panel is
    annotated to say so rather than leaving the reader hunting for them.
    """
    plt = _plt()
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(5.5 + 1.5 * len(rows), 4.6))
    for ax, key, nkey, lab in (
        (axes[0], "effect_size_mean", "null_effect_size",
         "mean |Δdrive| / std(drive)"),
        (axes[1], "flip_fraction_mean", "null_flip_fraction",
         "fraction of frames whose action flips"),
    ):
        ax.bar(x - 0.19, [r[key] for r in rows], width=0.36, color="#3a7bd5",
               label="repaint (m₁ ∈ {0.5, 1, 2})")
        ax.bar(x + 0.19, [r[nkey] for r in rows], width=0.36, color="#cccccc",
               edgecolor="#888888", label="null control (m₁ = m₀)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(lab, fontsize=9)
        ax.legend(fontsize=7)
        top = max(max(r[key] for r in rows) * 1.32, 1e-6)
        ax.set_ylim(0, top)
        if max(r[nkey] for r in rows) == 0.0:
            ax.text(len(rows) - 0.5, top * 0.02,
                    "null = 0.000 for every row", fontsize=7, ha="right",
                    color="#666666")

    # Panel three: the direction. This is the panel that carries the finding --
    # a large |Δ| that points nowhere in particular is a controller reacting to
    # colour, not one using the speed law.
    ax = axes[2]
    tags = [t for t in rows[0]["repaints"] if t != "null"]
    w = 0.8 / len(tags)
    cmap = plt.get_cmap("viridis")
    for i, tag in enumerate(tags):
        b = np.array([r["repaints"][tag]["direction_beta"] for r in rows])
        ci = np.array([r["repaints"][tag]["direction_ci95"] for r in rows])
        err = np.abs(np.stack([b - ci[:, 0], ci[:, 1] - b]))
        ax.bar(x - 0.4 + w * (i + 0.5), b, width=w * 0.9,
               yerr=err, capsize=2, color=cmap(0.15 + 0.7 * i / max(len(tags) - 1, 1)),
               label=f"m₁ = {rows[0]['repaints'][tag]['m1']:g}")
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("β:  Δdrive/sd  per  (log m₁ − log m₀)·sign(x_err)", fontsize=9)
    ax.set_title("direction — the law predicts β < 0", fontsize=10)
    ax.legend(fontsize=7)

    axes[0].set_title("effect of repainting the ball on the drive", fontsize=10)
    axes[1].set_title("...and on the action actually taken", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------------- reporting


def _fmt(v, nd=3) -> str:
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:.{nd}f}"


def write_report(rows: Sequence[Dict], a, out: Path, wall: float) -> None:
    (out / "summary.json").write_text(json.dumps(
        {"rows": rows, "args": vars(a), "repaint_masses": list(REPAINT_MASSES),
         "holdout_band": list(HOLDOUT_BAND), "wall_clock_s": wall}, indent=2))

    L = [
        "# Does the controller's decision depend on the ball's colour?",
        "",
        f"{a.n_frames} approach frames per policy, {a.steps}-step episodes, "
        f"warm-up {WARMUP} frames. Repaint masses {list(REPAINT_MASSES)} "
        f"(m₁ = 1.0 is **inside the held-out band** {list(HOLDOUT_BAND)}). "
        "Positions, paddle and actions are bit-identical between the two arms; "
        "only the hue changes.",
        "",
        "## Headline",
        "",
        "| controller | |Δdrive|/sd | flips | null |Δ|/sd | null flips | "
        "drive sd | frames (eps) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| `{r['name']}` | {_fmt(r['effect_size_mean'])} | "
            f"{_fmt(r['flip_fraction_mean'])} | {_fmt(r['null_effect_size'])} | "
            f"{_fmt(r['null_flip_fraction'])} | {_fmt(r['drive_std'], 2)} | "
            f"{r['n_frames']} ({r['n_episodes']}) |"
        )
    L += [
        "",
        "## Per repaint mass",
        "",
        "`β` is the coefficient of **Δdrive / std(drive)** on (log m₁ − log m₀)·sign(x_err), with "
        "a 95% CI from a bootstrap over episodes. **The law predicts β < 0**: "
        "repainting the ball lighter (faster) should push the paddle harder "
        "toward the side the ball is on.",
        "",
        "| controller | m₁ | |Δ|/sd | flips | β | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        for tag, s in r["repaints"].items():
            band = " *(held out)*" if s.get("in_holdout_band") else ""
            m1 = "m₀ (null)" if s["m1"] is None and tag == "null" else \
                f"{s['m1']:g}{band}"
            ci = s["direction_ci95"]
            L.append(
                f"| `{r['name']}` | {m1} | {_fmt(s['effect_size'])} | "
                f"{_fmt(s['flip_fraction'])} | {_fmt(s['direction_beta'])} | "
                f"[{_fmt(ci[0])}, {_fmt(ci[1])}] |"
            )
    rec = [r for r in rows if "drive_reconstruction_r" in r]
    if rec:
        L += ["", "## Reconstruction check", "",
              "Pearson r between the drive the rollout actually used (cold-start "
              "h, full episode) and the drive reconstructed here (8-frame "
              "warm-start). Not load-bearing — both arms of the intervention "
              "share the reconstruction — but it says how close the probe is to "
              "the policy as played.", "",
              "| controller | r |", "|---|---|"]
        L += [f"| `{r['name']}` | {_fmt(r['drive_reconstruction_r'])} |"
              for r in rec]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------------ main


def build_specs(a, stacks: StackCache) -> List[Tuple[str, BaseController, object, object]]:
    """(name, controller, V, M) for every policy, each with ITS OWN stack."""
    from .controller import OracleController

    rnn_over, vae_over = parse_overrides(a.run_rnn), parse_overrides(a.run_vae)
    specs: List[Tuple[str, BaseController, object, object]] = []
    for r in a.runs:
        ck = Path(r) / "controller.pt"
        if not ck.exists():
            print(f"  (skipping {r}: no controller.pt)")
            continue
        nm = Path(r).name
        rp, vp = stack_for_run(r, rnn_over, vae_over, a.rnn, a.vae)
        print(f"  {nm:<22} M={rp}  V={vp}")
        specs.append((nm, load_controller(ck, name=nm), stacks.vae(vp),
                      stacks.rnn(rp)))
    if a.ctrl_v1 and Path(a.ctrl_v1).exists():
        print(f"  {'ctrl_v1_on_v2':<22} M={a.rnn_v1}  V={a.vae_v1}")
        specs.append(("ctrl_v1_on_v2",
                      load_controller(a.ctrl_v1, name="ctrl_v1_on_v2"),
                      stacks.vae(a.vae_v1), stacks.rnn(a.rnn_v1)))
    if not a.no_oracle:
        specs.append(("oracle", OracleController(paddle_w=0.26),
                      stacks.vae(a.vae), stacks.rnn(a.rnn)))
    return specs


def shared_frames(roots: Sequence[str], n_episodes: int) -> Dict:
    """One frame set for every policy: recorded states from a held-out root."""
    parts = [episode_arrays(r) for r in roots]
    st = np.concatenate([p["state"] for p in parts], 0)
    ac = np.concatenate([p["actions"] for p in parts], 0)
    T = ac.shape[1]
    return {"states": st[:n_episodes, : T + 1], "actions": ac[:n_episodes],
            "source": f"{list(roots)}, first {n_episodes} episodes"}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae", default="runs/vae_v2/vae.pt")
    p.add_argument("--rnn", default="runs/rnn_v2/rnn.pt")
    p.add_argument("--vae-v1", default="runs/vae_b1/vae.pt")
    p.add_argument("--rnn-v1", default="runs/rnn_v1/rnn.pt")
    p.add_argument("--ctrl-v1", default="runs/ctrl_v1/controller.pt")
    p.add_argument("--runs", nargs="*", default=[
        "runs/ctrl_v2", "runs/ctrl_v2_cons", "runs/ctrl_v2_cons_tau0.5",
        "runs/ctrl_v2_tau0.5", "runs/ctrl_v2_z_only"])
    p.add_argument("--run-rnn", nargs="*", default=[], metavar="NAME=PATH")
    p.add_argument("--run-vae", nargs="*", default=[], metavar="NAME=PATH")
    p.add_argument("--frames", choices=["onpolicy", "val_mix"], default="onpolicy")
    p.add_argument("--shared-roots", nargs="*", default=["data/v2/val_mix"])
    p.add_argument("--n-frames", type=int, default=300)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    p.add_argument("--ball-radius", type=float, default=0.08)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-oracle", action="store_true")
    p.add_argument("--out", default="runs/ctrl_intervention")
    p.add_argument("--replot", action="store_true",
                   help="redraw the figure from an existing summary.json and "
                        "exit; changing how a result is drawn should never "
                        "require re-measuring it")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.replot:
        rows = json.loads((out / "summary.json").read_text())["rows"]
        print(plot_intervention(rows, out / "intervention_effect.png"))
        return
    t0 = time.time()

    # The environment config the repaint renders through. It must be the v2 one
    # whatever stack decodes the frames: the *world* is v2 even when the
    # encoder looking at it is v1's.
    cfg = make_box_cfg(a.ball_radius, mass_from_color=True,
                       mass_holdout=HOLDOUT_BAND)

    stacks = StackCache(a.device)
    specs = build_specs(a, stacks)
    shared = (shared_frames(a.shared_roots, a.episodes)
              if a.frames == "val_mix" else None)

    rows = []
    for name, ctrl, vae, rnn in specs:
        print(f"\n  {name}", flush=True)
        rows.append(run_controller(name, ctrl, vae, rnn, cfg, a, shared,
                                   a.device))

    plot_intervention(rows, out / "intervention_effect.png")
    write_report(rows, a, out, round(time.time() - t0, 1))
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

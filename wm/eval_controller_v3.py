"""v3 stage three: does the controller act on MEMORY while the ball is hidden?

    python -m wm.eval_controller_v3 --out runs/ctrl_eval_v3

v2 asked whether skill depended on a property of the ball (its mass) and sliced
every metric by mass tercile. v3 asks a behavioural question that a single
averaged score cannot see at all, because the average is dominated by the easy
case.

The geometry is the whole argument. The band's bottom edge sits at y = 0.28, so
a descending ball reappears about **10 frames** before it can reach the paddle,
and the paddle crosses the box in about **30**. A policy that waits until it can
see the ball can therefore still cover a *short* move and cannot possibly cover
a long one -- it is not a matter of skill, it is a matter of frames. So:

    for each floor visit, measure the REQUIRED MOVE -- the distance between
    where the paddle was standing at the moment the ball vanished on its way
    down, and where the ball actually came down -- and report interceptions per
    floor visit in three bins of it.

A controller with object permanence is **flat** across those bins. A memoryless
one **collapses on long moves**. That is a shape, not a number, and it cannot be
faked by a policy that is simply better at everything.

The two references that make the shape readable
-----------------------------------------------
``oracle``
    true state every frame: perfect vision *and* perfect memory. The ceiling.
``wait_and_see``
    the same true state, but only on frames where ``ball_visible > 0.5``; STAY
    otherwise. Perfect vision, **zero memory** -- the best a policy without
    object permanence can possibly do, with no world model in the loop.

The gap between those two is the entire value of object permanence for this
task, measured behaviourally. If it is small, then permanence is not the binding
constraint here and no controller -- fair or privileged -- can gain much from it.
That comparison is worth more than any of the trained rows and it is why it is
computed first.

The second measurement: what the paddle DOES while the ball is invisible
------------------------------------------------------------------------
Interceptions are an outcome and outcomes are noisy. The direct version of the
question reads the actions:

* **toward fraction** -- over the hidden frames of each descending occlusion,
  the fraction on which the action moved the paddle toward where the ball would
  eventually come down. Chance is **1/3** (three actions, and STAY counts as
  not-toward). A memoryless policy cannot beat chance on these frames except
  through whatever correlation its last-seen information still carries.
* **displacement fraction** -- how much of the required move the paddle actually
  covered *while the ball was hidden*, as a fraction. This is the outcome-free
  version of "did it commit".

Nothing here re-implements the rollout: it calls ``wm.eval_controller``'s
harness (now carrying the ``--occluder`` flags), and the only thing this file
owns is the slicing. The occlusion bookkeeping -- which is the part that can be
silently wrong -- lives in small functions at the top and is unit-tested in
``tests/test_controller_v3.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from worldsim.render import save_gif

from .controller import (
    BaseController,
    OracleController,
    RandomController,
    StayController,
    WaitAndSeeOracleController,
    load_controller,
)
from .eval_controller import (
    DEFAULT_SEED_BASE,
    _bootstrap_ci,
    _plt,
    contact_runs,
    dream_play_gif,
    floor_zone_height,
    run_real_episodes,
)
from .eval_controller_v2 import (
    StackCache,
    Spec,
    _fmt,
    _ratio_ci,
    parse_overrides,
    sign_agreement_per_episode,
    stack_for_run,
)

LEFT, STAY, RIGHT = 0, 1, 2

# The bin edges the design document asks for. "long" starts at 0.35 because the
# paddle moves 0.030/frame and a descending ball is visible for ~10 frames
# before contact: anything past ~0.30 is unreachable from a standing start once
# the ball is already out of the band, so the long bin is exactly the set of
# chances that REQUIRE having moved early.
REQUIRED_MOVE_EDGES = (0.15, 0.35)
MOVE_BINS = ("short", "medium", "long")
# A finer slicing of the same axis, used only for the "reach curve" -- where
# exactly does the memoryless bound break? The three headline bins turn out to
# be too coarse to see it, because the paddle is 0.26 WIDE: a required move is
# covered once the paddle's centre is within half a paddle-width (0.13) of the
# landing x, and the ~10 visible frames before contact buy another ~0.30 of
# travel. So a wait-and-see policy can still catch a "long" 0.40 move, and the
# interesting question is what happens past ~0.43.
FINE_EDGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
FINE_BINS = ("<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">0.6")
HIDDEN_EPS = 1e-3          # matches worldsim's EVENT_HIDDEN threshold
N_BOOT = 4000


# ------------------------------------------------------------ the bookkeeping


def visible_column(states: np.ndarray) -> np.ndarray:
    """``ball_visible`` per frame: the LAST state column in an occluded world.

    v3 appends it after the optional mass column, so its index depends on which
    flags built the dataset and "the last one" is the only description that is
    always right. A 6-column state means there is no band and everything is
    visible, which keeps every function below defined on v1/v2 rollouts.
    """
    if states.shape[-1] <= 6:
        return np.ones(states.shape[:-1], np.float64)
    return np.asarray(states[..., -1], np.float64)


def floor_visits(
    states: np.ndarray,
    ball_radius: float = 0.08,
    paddle_h: float = 0.045,
) -> List[Dict]:
    """One record per CHANCE the physics offered, with where the ball came down.

    The visit definition is character-for-character the one in
    ``eval_controller.floor_visit_stats`` -- a downward crossing of
    ``paddle_top + ball_radius + one frame of travel`` -- so the counts in this
    file and the counts in the v1/v2 tables are the same counts. A test asserts
    it rather than trusting the comment.

    ``landing_x`` is the ball's x at the LOWEST frame of the visit, i.e. where
    the interception either happened or failed to. That is also the x the gap
    metric measures against, so "required move" and "gap at floor" are answers
    about the same point in the same episode.
    """
    y = states[..., 1]
    thr = floor_zone_height(ball_radius, paddle_h)
    below = y < thr
    entering = below[:, 1:] & ~below[:, :-1]
    out: List[Dict] = []
    for e in range(states.shape[0]):
        for t0 in np.flatnonzero(entering[e]) + 1:
            t1 = t0
            while t1 + 1 < below.shape[1] and below[e, t1 + 1]:
                t1 += 1
            seg = slice(t0, t1 + 1)
            t_low = int(t0 + np.argmin(y[e, seg]))
            out.append({
                "episode": int(e),
                "t_enter": int(t0),
                "t_low": t_low,
                "t_exit": int(t1),
                "landing_x": float(states[e, t_low, 0]),
                "gap": float(abs(states[e, t_low, 0] - states[e, t_low, 4])),
            })
    return out


def annotate_occlusion(
    visit: Dict, states: np.ndarray, actions: np.ndarray
) -> Dict:
    """Add "how far did the paddle have to go, and what did it do in the dark".

    Walks back from the lowest frame of the visit through the ball's DESCENT
    (the unbroken stretch of ``ball_vy < 0`` that ends there) and finds the last
    stretch of fully hidden frames inside it -- that is the occlusion the
    controller had to act through on this approach. Then:

    ``required_move``   |landing_x - paddle_x| at the frame the ball vanished.
                        The distance the policy had to cover, measured at the
                        last instant it could still see the ball.
    ``toward_frames``   hidden frames on which the ACTION moved the paddle
                        toward ``landing_x``. STAY is not toward -- a policy
                        that stands still has not committed -- so chance for a
                        uniform random policy is 1/3, not 1/2.
    ``moved``           how much of ``required_move`` the paddle covered while
                        the ball was hidden, signed toward the landing point
                        (negative means it went the wrong way).

    A visit whose descent contains no fully hidden frame at all (the ball
    entered the box below the band, or the band was crossed while ascending)
    gets ``hidden=False`` and is excluded from every by-required-move statistic
    -- there was no occlusion to be memoryless about.
    """
    e, t_low = visit["episode"], visit["t_low"]
    vy = states[e, :, 3]
    vis = visible_column(states)[e]
    paddle = states[e, :, 4]
    T = actions.shape[1]

    # The descent: back from the lowest frame while the ball is still falling.
    t_desc = t_low
    while t_desc - 1 >= 0 and vy[t_desc - 1] < 0:
        t_desc -= 1

    hidden = vis[t_desc : t_low + 1] < HIDDEN_EPS
    v = dict(visit, hidden=False)
    if not hidden.any():
        return v

    # The LAST hidden stretch of the descent: if the ball crossed the band
    # twice on one descent (possible only with a bounce, but cheap to be right
    # about) it is the most recent disappearance that the decision rides on.
    idx = np.flatnonzero(hidden)
    end = int(idx[-1])
    start = end
    while start - 1 >= 0 and hidden[start - 1]:
        start -= 1
    t_hide, t_emerge = t_desc + start, t_desc + end

    req = float(abs(visit["landing_x"] - paddle[t_hide]))
    direction = np.sign(visit["landing_x"] - paddle[t_hide])
    acts = actions[e, t_hide : min(t_emerge + 1, T)]
    step = np.where(acts == RIGHT, 1.0, np.where(acts == LEFT, -1.0, 0.0))
    toward = int(((step != 0) & (step == direction)).sum())
    moving = int((step != 0).sum())
    t_after = min(t_emerge + 1, states.shape[1] - 1)
    moved = float((paddle[t_after] - paddle[t_hide]) * direction)

    v.update(
        hidden=True,
        t_hide=int(t_hide),
        t_emerge=int(t_emerge),
        hidden_frames=int(len(acts)),
        required_move=req,
        paddle_at_hide=float(paddle[t_hide]),
        toward_frames=toward,
        moving_frames=moving,
        moved_toward=moved,
    )
    return v


def required_move_bin(d: float) -> int:
    """0 = short (< 0.15), 1 = medium (0.15-0.35), 2 = long (> 0.35)."""
    return int(np.digitize(d, REQUIRED_MOVE_EDGES))


def visit_interceptions(visit: Dict, hits: np.ndarray) -> int:
    """How many interceptions happened during this visit.

    ``hits[t]`` flags a contact during the transition ``states[t] -> states[t+1]``,
    so a contact belonging to a visit spanning state indices ``[t_enter, t_exit]``
    has hit-index in ``[t_enter - 1, t_exit - 1]``. Runs of consecutive contact
    frames collapse to one, exactly as ``contact_runs`` does globally, so a
    policy that pins the ball cannot inflate a bin.
    """
    e = visit["episode"]
    h = hits[e] > 0
    starts = np.flatnonzero(h & ~np.r_[False, h[:-1]])
    lo, hi = visit["t_enter"] - 1, visit["t_exit"] - 1
    return int(((starts >= lo) & (starts <= hi)).sum())


def occlusion_table(roll: Dict[str, np.ndarray]) -> List[Dict]:
    """Every floor visit of a rollout, annotated and scored. The raw material."""
    states, actions, hits = roll["states"], roll["actions"], roll["hits"]
    out = []
    for v in floor_visits(states):
        v = annotate_occlusion(v, states, actions)
        v["interceptions"] = visit_interceptions(v, hits)
        out.append(v)
    return out


# ------------------------------------------------------------- aggregation


def _per_episode(values: Sequence[float], episodes: Sequence[int], n: int) -> np.ndarray:
    """Sum per-visit (or per-run) quantities into a per-episode vector.

    Everything is bootstrapped over EPISODES -- the unit of independence -- so
    every statistic has to be expressible as a ratio of two per-episode sums
    first. Episodes that contributed nothing come out as 0/0, which the ratio
    bootstrap handles by construction.
    """
    out = np.zeros(n, np.float64)
    for v, e in zip(values, episodes):
        out[int(e)] += float(v)
    return out


def summarise_rollout(name: str, roll: Dict[str, np.ndarray]) -> Dict:
    """Overall metrics, the by-required-move split, and the occlusion behaviour."""
    n_ep = roll["states"].shape[0]
    visits = occlusion_table(roll)
    runs = contact_runs(roll["hits"])
    sa, sa_n = sign_agreement_per_episode(roll)
    a = roll["actions"].ravel()

    ep = [v["episode"] for v in visits]
    inter = _per_episode([v["interceptions"] for v in visits], ep, n_ep)
    n_visits = _per_episode(np.ones(len(visits)), ep, n_ep)
    ipv, lo, hi = _ratio_ci(inter, n_visits)

    good = ~np.isnan(sa)
    sa_mean, sa_lo, sa_hi = _ratio_ci(np.where(good, sa, 0.0) * sa_n, sa_n)

    row: Dict = {
        "name": name,
        "episodes": int(n_ep),
        "steps": int(roll["actions"].shape[1]),
        "floor_visits_per_episode": float(n_visits.mean()),
        "interceptions_per_episode": float(runs.mean()),
        "interceptions_ci95": list(_bootstrap_ci(runs, n_boot=N_BOOT)),
        "interceptions_per_visit": ipv,
        "interceptions_per_visit_ci95": [lo, hi],
        "mean_gap_at_floor": float(np.mean([v["gap"] for v in visits]))
        if visits else float("nan"),
        "sign_agreement_approach": sa_mean,
        "sign_agreement_approach_ci95": [sa_lo, sa_hi],
        "action_dist": {n: float((a == k).mean())
                        for k, n in enumerate(("left", "stay", "right"))},
        "n_visits": int(len(visits)),
        "n_visits_hidden": int(sum(v["hidden"] for v in visits)),
    }

    # ---- by required move
    hid = [v for v in visits if v["hidden"]]
    by_bin: Dict[str, Dict] = {}
    for b, bname in enumerate(MOVE_BINS):
        sel = [v for v in hid if required_move_bin(v["required_move"]) == b]
        e_sel = [v["episode"] for v in sel]
        num = _per_episode([v["interceptions"] for v in sel], e_sel, n_ep)
        den = _per_episode(np.ones(len(sel)), e_sel, n_ep)
        val, blo, bhi = _ratio_ci(num, den)
        by_bin[bname] = {
            "n_visits": len(sel),
            "interceptions_per_visit": val,
            "interceptions_per_visit_ci95": [blo, bhi],
            "mean_required_move": float(np.mean([v["required_move"] for v in sel]))
            if sel else float("nan"),
            "mean_hidden_frames": float(np.mean([v["hidden_frames"] for v in sel]))
            if sel else float("nan"),
        }
    row["by_required_move"] = by_bin

    # The same split at 0.1 resolution: the reach curve.
    fine: Dict[str, Dict] = {}
    for b, bname in enumerate(FINE_BINS):
        sel = [v for v in hid
               if int(np.digitize(v["required_move"], FINE_EDGES)) == b]
        e_sel = [v["episode"] for v in sel]
        num = _per_episode([v["interceptions"] for v in sel], e_sel, n_ep)
        den = _per_episode(np.ones(len(sel)), e_sel, n_ep)
        val, blo, bhi = _ratio_ci(num, den)
        fine[bname] = {"n_visits": len(sel), "interceptions_per_visit": val,
                       "interceptions_per_visit_ci95": [blo, bhi]}
    row["reach_curve"] = fine

    # ---- what the paddle did in the dark
    row["occlusion"] = _occlusion_behaviour(hid, n_ep)
    # The same thing restricted to the chances that actually demanded a move.
    # On a short required move "toward" is barely defined -- the paddle is
    # already there -- so the pooled fraction is diluted by visits where the
    # right answer is to stand still.
    row["occlusion_long"] = _occlusion_behaviour(
        [v for v in hid if v["required_move"] > REQUIRED_MOVE_EDGES[0]], n_ep
    )
    return row


def _occlusion_behaviour(visits: Sequence[Dict], n_ep: int) -> Dict:
    """Toward-fraction and displacement fraction over a set of hidden runs."""
    if not visits:
        return {"n_runs": 0, "toward_fraction": float("nan"),
                "toward_fraction_ci95": [float("nan")] * 2,
                "toward_when_moving": float("nan"),
                "toward_when_moving_ci95": [float("nan")] * 2,
                "moving_fraction": float("nan"),
                "displacement_fraction": float("nan"),
                "displacement_fraction_ci95": [float("nan")] * 2,
                "mean_hidden_frames": float("nan"),
                "mean_required_move": float("nan")}
    ep = [v["episode"] for v in visits]
    tw = _per_episode([v["toward_frames"] for v in visits], ep, n_ep)
    fr = _per_episode([v["hidden_frames"] for v in visits], ep, n_ep)
    t_val, t_lo, t_hi = _ratio_ci(tw, fr)
    # The same fraction with STAY frames removed from the denominator. It has
    # to be reported next to the first one, because the two answer different
    # questions and the raw one is confounded by how twitchy a policy is: a
    # bang-bang policy that never stops racks up "toward" frames simply by
    # being in motion half the time, while a policy that parks scores near zero
    # however well it is parked. Chance here is 1/2.
    mvf = _per_episode([v["moving_frames"] for v in visits], ep, n_ep)
    m_val, m_lo, m_hi = _ratio_ci(tw, mvf)
    # Displacement as a fraction of the required move is a ratio of two
    # distances, so it too is aggregated as a ratio of sums rather than as a
    # mean of per-visit ratios: a visit whose required move is 0.01 would
    # otherwise dominate with a ratio of 30.
    mv = _per_episode([v["moved_toward"] for v in visits], ep, n_ep)
    rq = _per_episode([v["required_move"] for v in visits], ep, n_ep)
    d_val, d_lo, d_hi = _ratio_ci(mv, rq)
    return {
        "n_runs": len(visits),
        "toward_fraction": t_val,
        "toward_fraction_ci95": [t_lo, t_hi],
        "toward_when_moving": m_val,
        "toward_when_moving_ci95": [m_lo, m_hi],
        "moving_fraction": float(mvf.sum() / max(fr.sum(), 1e-9)),
        "displacement_fraction": d_val,
        "displacement_fraction_ci95": [d_lo, d_hi],
        "mean_hidden_frames": float(np.mean([v["hidden_frames"] for v in visits])),
        "mean_required_move": float(np.mean([v["required_move"] for v in visits])),
    }


# ------------------------------------------------------------------- plots


def plot_by_required_move(
    rows: Sequence[Dict], out: Path,
    refs: Sequence[str] = ("oracle", "wait_and_see"),
) -> Path:
    """Grouped bars, one cluster per required-move bin.

    The two references are drawn as per-bin horizontal segments rather than as
    bars, because they are not competitors: ``oracle`` is the ceiling and
    ``wait_and_see`` is the memoryless bound, and what the reader has to be able
    to see is where each trained bar falls BETWEEN them, bin by bin.
    """
    plt = _plt()
    ref_rows = [r for r in rows if r["name"] in refs]
    bars = [r for r in rows if r["name"] not in refs]
    n = len(bars)
    x = np.arange(len(MOVE_BINS))
    w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(3.0 + 1.15 * n, 5.0))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(bars):
        vals = [r["by_required_move"][b]["interceptions_per_visit"] for b in MOVE_BINS]
        ci = [r["by_required_move"][b]["interceptions_per_visit_ci95"] for b in MOVE_BINS]
        err = None
        if all(c is not None and not np.isnan(c[0]) for c in ci):
            err = np.array([[max(v - c[0], 0), max(c[1] - v, 0)]
                            for v, c in zip(vals, ci)]).T
        color = {"stay": "#bbbbbb", "random": "#888888"}.get(
            r["name"], cmap(0.10 + 0.75 * i / max(n - 1, 1))
        )
        ax.bar(x - 0.4 + w * (i + 0.5), vals, width=w * 0.92, yerr=err,
               capsize=2, color=color, label=r["name"])
    styles = {"oracle": ("#2a9d4a", "-"), "wait_and_see": ("#d55e3a", "--")}
    for r in ref_rows:
        c, ls = styles.get(r["name"], ("#333333", ":"))
        for k, b in enumerate(MOVE_BINS):
            v = r["by_required_move"][b]["interceptions_per_visit"]
            ax.plot([k - 0.45, k + 0.45], [v, v], c=c, ls=ls, lw=2.2,
                    label=f"{r['name']} (reference)" if k == 0 else None)
    # No visit counts on the axis: the required move is measured from each
    # policy's OWN paddle, so the three bins hold a different number of visits
    # for every row. The per-row counts are in summary.md.
    ax.set_xticks(x)
    ax.set_xticklabels(["short\n< 0.15", "medium\n0.15 - 0.35", "long\n> 0.35"])
    ax.set_xlabel("required move: |paddle_x when the ball vanished − landing x|")
    ax.set_ylabel("interceptions per floor visit")
    ax.set_title("skill by how far the paddle had to go while it could not see\n"
                 "(flat = object permanence; collapsing = memoryless)")
    ax.set_ylim(0, max(ax.get_ylim()[1], 1e-6) * 1.38)
    ax.legend(fontsize=7, ncol=3, loc="upper center", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_reach_curve(rows: Sequence[Dict], out: Path) -> Path:
    """Interceptions per visit against the required move, at 0.1 resolution.

    The three headline bins answer "is the controller flat"; this answers the
    prior question "where does a memoryless policy actually break, given a
    paddle 0.26 wide and ten visible frames of run-up". If ``wait_and_see``
    stays at the ceiling out to 0.4, then every bin below 0.4 is uninformative
    about object permanence no matter what the trained rows do in it.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(FINE_BINS))
    cmap = plt.get_cmap("viridis")
    trained = [r for r in rows if r["name"] not in
               ("oracle", "wait_and_see", "stay", "random")]
    fixed = {"oracle": dict(c="#2a9d4a", lw=2.4),
             "wait_and_see": dict(c="#d55e3a", lw=2.4, ls="--"),
             "stay": dict(c="#cccccc", lw=1.2),
             "random": dict(c="#999999", lw=1.2)}
    for r in rows:
        v = [r["reach_curve"][b]["interceptions_per_visit"] for b in FINE_BINS]
        if r["name"] in fixed:
            style = fixed[r["name"]]
        else:
            i = trained.index(r)
            style = dict(c=cmap(0.1 + 0.75 * i / max(len(trained) - 1, 1)),
                         lw=1.4, marker="o", ms=3)
        ax.plot(x, v, label=r["name"], **style)
    n = [rows[0]["reach_curve"][b]["n_visits"] for b in FINE_BINS]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={k})" for b, k in zip(FINE_BINS, n)], fontsize=7)
    ax.set_xlabel("required move (world units) — first row's visit counts shown")
    ax.set_ylabel("interceptions per floor visit")
    ax.set_title("the reach curve: where does a memoryless policy break?")
    ax.set_ylim(-0.02, 1.15)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_occlusion_behaviour(rows: Sequence[Dict], out: Path) -> Path:
    """Two panels: does the paddle move the right way in the dark, and how far.

    Left is the per-frame action statistic with its 1/3 chance line; right is
    the distance actually covered as a fraction of the distance needed, with 1.0
    meaning "arrived before the ball reappeared" and 0 meaning "did not move".
    Both are restricted to descending occlusions with a required move above
    0.15, because on shorter ones standing still is the correct behaviour and a
    low score would mean nothing.
    """
    plt = _plt()
    names = [r["name"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(4.0 + 1.2 * len(names), 4.8))
    for ax, key, label, chance in (
        (axes[0], "toward_fraction",
         "toward / ALL hidden frames", 1 / 3),
        (axes[1], "toward_when_moving",
         "toward / MOVING hidden frames", 0.5),
        (axes[2], "displacement_fraction",
         "paddle displacement toward landing / required move", 0.0),
    ):
        vals = [r["occlusion_long"][key] for r in rows]
        ci = [r["occlusion_long"][key + "_ci95"] for r in rows]
        err = np.array([[max(v - c[0], 0), max(c[1] - v, 0)]
                        for v, c in zip(vals, ci)]).T
        colors = ["#2a9d4a" if n == "oracle" else
                  "#d55e3a" if n == "wait_and_see" else
                  "#bbbbbb" if n in ("stay", "random") else "#3a7bd5"
                  for n in names]
        ax.bar(np.arange(len(names)), vals, yerr=err, capsize=3, color=colors)
        ax.axhline(chance, ls="--", c="k", lw=1)
        ax.text(len(names) - 0.4, chance + 0.01,
                "no motion" if chance == 0 else f"chance ({chance:.2f})",
                ha="right", fontsize=8)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(label, fontsize=8)
    axes[0].set_title("does it move the right way while blind?", fontsize=9)
    axes[1].set_title("...when it moves at all?", fontsize=9)
    axes[2].set_title("how much of the move does it make while blind?", fontsize=9)
    fig.suptitle("paddle motion during occlusion (required move > 0.15)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_transfer(runs: Sequence[str], out: Path) -> Dict[str, Dict]:
    """One panel per training run: dream fitness against the real check.

    ``train_controller`` already wrote this per run; the grid exists so the six
    curves can be read against each other, which is where the interesting thing
    is -- whether the dreams that are *worse* world models are also the ones
    whose fitness curve stops predicting the real score.
    """
    plt = _plt()
    from .train_controller import _smooth

    have = [r for r in runs if (Path(r) / "history.json").exists()]
    if not have:
        return {}
    ncol = min(3, len(have))
    nrow = int(np.ceil(len(have) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                             squeeze=False)
    corr: Dict[str, Dict] = {}
    for ax, run in zip(axes.ravel(), have):
        h = json.loads((Path(run) / "history.json").read_text())
        hist, meta = h["history"], h["meta"]
        ax.plot(hist["gen"], hist["best"], c="#3a7bd5", alpha=0.25, lw=0.8)
        ax.plot(hist["gen"], _smooth(hist["best"]), c="#3a7bd5")
        ax.set_ylabel("dream return", color="#3a7bd5", fontsize=8)
        ax2 = ax.twinx()
        rg, rh = hist["real_gen"], np.asarray(hist["real_hits"])
        ax2.plot(rg, rh, c="#d55e3a", marker="o", ms=3)
        ax2.set_ylabel("real interceptions/ep", color="#d55e3a", fontsize=8)
        tr = meta.get("transfer", {})
        corr[Path(run).name] = {
            "pearson_r_smoothed": tr.get("pearson_r_smoothed"),
            "pearson_r_raw": tr.get("pearson_r_raw"),
            "best_real": meta.get("best_real_hits"),
            "final_dream": meta.get("final_dream_fitness"),
            "wall_clock_s": meta.get("wall_clock_s"),
            "dream_env_steps": meta.get("dream_env_steps"),
            "real_env_steps_total": meta.get("real_env_steps_total"),
        }
        ax.set_title(f"{Path(run).name}   r = "
                     f"{tr.get('pearson_r_smoothed', float('nan')):.2f} "
                     f"(smoothed)", fontsize=9)
        ax.set_xlabel("generation", fontsize=8)
    for ax in axes.ravel()[len(have):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return corr


# --------------------------------------------------------------- VAE check


def vae_recon_mse(vae, roots: Sequence[str], n_frames: int = 512,
                  device: str = "cpu", seed: int = 0) -> Dict[str, Dict]:
    """Reconstruction MSE of the frozen v3 encoder on each dataset root.

    The taller-band evaluation moves the band to heights the VAE has never seen,
    and a drop in skill there could be *either* the controller failing to
    generalise *or* V simply not encoding the frame any more. Measuring the
    reconstruction separates the two. ``data/v3/tall`` and ``data/v3/taller``
    were collected at exactly the two bands the generalisation test uses, so
    this needs no new rollouts.
    """
    import torch

    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}
    for root in roots:
        frames = np.load(Path(root) / "frames.npy", mmap_mode="r")
        flat = frames.reshape(-1, *frames.shape[-3:])
        idx = np.sort(rng.choice(len(flat), size=min(n_frames, len(flat)),
                                 replace=False))
        x = torch.from_numpy(np.asarray(flat[idx], np.float32) / 255.0)
        x = x.permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(x)
            xh = vae.decode(mu)
        mse = float(((xh - x) ** 2).mean())
        meta = json.loads((Path(root) / "meta.json").read_text())
        out[root] = {
            "recon_mse": mse,
            "occluder_y": meta.get("occluder_y"),
            "n_frames": int(len(idx)),
            "mean_hidden_run_frames":
                meta.get("occlusion", {}).get("mean_hidden_run_frames"),
        }
    return out


# --------------------------------------------------------------------- gifs


def pick_demo_episode(
    table: Sequence[Dict], min_move: float = 0.35, min_toward: float = 0.5,
    min_hidden: int = 6,
) -> Optional[Dict]:
    """A floor visit with a long required move where the paddle moved in the dark.

    This is an explicit cherry-pick and the report says so: it is a *demo* of the
    behaviour the statistics measure, not evidence about the average episode.
    Returns the most extreme qualifying visit (longest required move) or None.
    """
    ok = [v for v in table
          if v.get("hidden") and v["required_move"] >= min_move
          and v["hidden_frames"] >= min_hidden
          and v["toward_frames"] / v["hidden_frames"] >= min_toward]
    if not ok:
        return None
    return max(ok, key=lambda v: v["required_move"])


# ------------------------------------------------------------------ reports


def by_move_table(rows: Sequence[Dict], key: str, nd: int = 2,
                  ci: bool = False) -> List[str]:
    """Markdown: one row per controller, one column per required-move bin."""
    L = ["| controller | overall | " + " | ".join(MOVE_BINS) + " |",
         "|---" * (len(MOVE_BINS) + 2) + "|"]
    for r in rows:
        cells = []
        for c in ("overall",) + MOVE_BINS:
            d = r if c == "overall" else r["by_required_move"][c]
            v = _fmt(d[key], nd)
            cc = d.get(key + "_ci95")
            if ci and cc is not None and not np.isnan(cc[0]):
                v += f" [{cc[0]:.2f}, {cc[1]:.2f}]"
            cells.append(v)
        L.append(f"| `{r['name']}` | " + " | ".join(cells) + " |")
    return L


def occlusion_table_md(rows: Sequence[Dict]) -> List[str]:
    L = [
        "| controller | hidden runs | mean hidden frames | mean required move | "
        "moved on (frac. of hidden frames) | toward / all hidden frames "
        "(chance 1/3) | toward / moving frames (chance 1/2) | "
        "displacement fraction |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        o, ol = r["occlusion"], r["occlusion_long"]
        L.append(
            f"| `{r['name']}` | {o['n_runs']} | {_fmt(o['mean_hidden_frames'], 1)} | "
            f"{_fmt(o['mean_required_move'], 3)} | "
            f"{_fmt(ol['moving_fraction'], 3)} | "
            f"{_fmt(ol['toward_fraction'], 3)} "
            f"[{_fmt(ol['toward_fraction_ci95'][0], 3)}, "
            f"{_fmt(ol['toward_fraction_ci95'][1], 3)}] | "
            f"{_fmt(ol['toward_when_moving'], 3)} "
            f"[{_fmt(ol['toward_when_moving_ci95'][0], 3)}, "
            f"{_fmt(ol['toward_when_moving_ci95'][1], 3)}] | "
            f"{_fmt(ol['displacement_fraction'], 3)} "
            f"[{_fmt(ol['displacement_fraction_ci95'][0], 3)}, "
            f"{_fmt(ol['displacement_fraction_ci95'][1], 3)}] |"
        )
    return L


# --------------------------------------------------------------------- main


def build_references(paddle_w: float = 0.26) -> List[BaseController]:
    """stay, random, oracle and the memoryless upper bound."""
    return [
        StayController(),
        RandomController(mean_hold=8.0, seed=1234),
        OracleController(paddle_w=paddle_w),
        WaitAndSeeOracleController(paddle_w=paddle_w),
    ]


def run_all(specs: Sequence[Spec], episodes: int, steps: int, seed_base: int,
            env: Dict, device: str) -> Dict[str, Dict[str, np.ndarray]]:
    rolls = {}
    for sp in specs:
        t = time.time()
        rolls[sp.name] = run_real_episodes(
            sp.ctrl, sp.vae, sp.rnn, episodes=episodes, steps=steps,
            seed_base=seed_base, device=device, record_logits=True, **env,
        )
        print(f"  {sp.name:<24} interceptions/ep "
              f"{contact_runs(rolls[sp.name]['hits']).mean():.2f}  "
              f"({time.time() - t:.0f}s)", flush=True)
    return rolls


def draw_plots(rows: Sequence[Dict], out: Path) -> None:
    """The three figures, from the ``params_best_real`` rows only.

    The ``_lastdream`` rows stay in every table -- the gap between the two is
    the price of the periodic real check and it is worth reading -- but putting
    fourteen bars in one cluster makes the figure unreadable and the figure's
    job is to show a SHAPE.
    """
    shown = [r for r in rows if not r["name"].endswith("_lastdream")]
    plot_by_required_move(shown, out / "interceptions_by_required_move.png")
    plot_occlusion_behaviour(shown, out / "paddle_motion_during_occlusion.png")
    plot_reach_curve(shown, out / "reach_curve.png")


def replot(out: Path) -> None:
    """Redraw the figures from an existing ``summary.json``.

    Every rollout in this evaluation costs ~20 minutes, and a figure is a
    presentation choice that should never be a reason to re-run one.
    """
    rows = json.loads((out / "summary.json").read_text())["in_distribution"]
    draw_plots(rows, out)
    print(f"redrew the figures in {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vae", default="runs/vae_v3/vae.pt")
    p.add_argument("--rnn", default="runs/rnn_v3/rnn.pt")
    p.add_argument("--runs", nargs="*", default=[
        "runs/ctrl_v3", "runs/ctrl_v3_tau1", "runs/ctrl_v3_ff",
        "runs/ctrl_v3_poshead", "runs/ctrl_v3_emerge", "runs/ctrl_v3_z_only",
    ])
    p.add_argument("--run-rnn", nargs="*", default=[], metavar="NAME=PATH")
    p.add_argument("--run-vae", nargs="*", default=[], metavar="NAME=PATH")
    p.add_argument("--params", nargs="*",
                   default=["params_best_real", "params_last_dream"],
                   choices=["params_best_real", "params_last_dream",
                            "params_best_dream_sample"])
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    p.add_argument("--band-names", nargs="*",
                   default=["ctrl_v3", "ctrl_v3_ff", "ctrl_v3_poshead",
                            "oracle", "wait_and_see"],
                   help="rows re-run on the taller bands")
    p.add_argument("--band-episodes", type=int, default=60)
    p.add_argument("--bands", nargs="*", default=["0.22,0.64", "0.16,0.70"],
                   help="BOTTOM,TOP pairs; seeds 6000, 6100, ...")
    p.add_argument("--band-seed-base", type=int, default=6000)
    p.add_argument("--recon-roots", nargs="*",
                   default=["data/v3/val", "data/v3/tall", "data/v3/taller"])
    p.add_argument("--dream-roots", nargs="*",
                   default=["data/v3/train", "data/v3/train_mix"])
    p.add_argument("--gif-run", default="ctrl_v3")
    p.add_argument("--no-gifs", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="runs/ctrl_eval_v3")
    p.add_argument("--replot", action="store_true",
                   help="redraw the figures from an existing summary.json and "
                        "exit -- no rollouts")
    a = p.parse_args()
    if a.replot:
        replot(Path(a.out))
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    stacks = StackCache(a.device)
    vae, rnn = stacks.vae(a.vae), stacks.rnn(a.rnn)
    rnn_over, vae_over = parse_overrides(a.run_rnn), parse_overrides(a.run_vae)

    # The references read the true state, so which (V, M) they are handed
    # changes nothing -- it only has to exist.
    specs: List[Spec] = [Spec(c.name, c, vae, rnn) for c in build_references()]
    stack_of: Dict[str, Tuple[str, str]] = {}
    print("per-run (V, M) pairing -- the controller reads h, so M is part of "
          "the policy:")
    for r in a.runs:
        ck = Path(r) / "controller.pt"
        if not ck.exists():
            raise SystemExit(f"--runs {r}: no controller.pt")
        nm = Path(r).name
        r_path, v_path = stack_for_run(r, rnn_over, vae_over, a.rnn, a.vae)
        print(f"  {nm:<24} M={r_path}  V={v_path}")
        for which in a.params:
            suffix = "" if which == "params_best_real" else "_lastdream"
            name = nm + suffix
            try:
                c = load_controller(ck, name=name, which=which)
            except KeyError:
                print(f"    (no {which} in {ck})")
                continue
            specs.append(Spec(name, c, stacks.vae(v_path), stacks.rnn(r_path)))
            stack_of[name] = (r_path, v_path)

    env = {"ball_radius": 0.08, "occluder": True}
    print(f"\nin-distribution: {a.episodes} episodes x {a.steps} steps, seeds "
          f"{a.seed_base}.., band (0.28, 0.58)")
    rolls = run_all(specs, a.episodes, a.steps, a.seed_base, env, a.device)
    rows = [summarise_rollout(sp.name, rolls[sp.name]) for sp in specs]

    # ---- taller bands
    band_rows: Dict[str, List[Dict]] = {}
    for i, spec in enumerate(a.bands):
        band = [float(x) for x in spec.replace(",", " ").split()]
        seed = a.band_seed_base + 100 * i
        sel = [sp for sp in specs if sp.name in set(a.band_names)]
        print(f"\ntaller band {tuple(band)}: {a.band_episodes} episodes, "
              f"seeds {seed}..")
        br = run_all(sel, a.band_episodes, a.steps, seed,
                     {**env, "occluder_y": band}, a.device)
        band_rows[str(tuple(band))] = [
            summarise_rollout(sp.name, br[sp.name]) for sp in sel
        ]

    recon = vae_recon_mse(vae, a.recon_roots, device=a.device)
    transfer = plot_transfer(a.runs, out / "dream_vs_real_all.png")

    draw_plots(rows, out)

    gifs: Dict[str, str] = {}
    if not a.no_gifs and a.gif_run in rolls:
        gifs = render_gifs(rolls, specs, vae, rnn, out, a)

    write_report(rows, band_rows, recon, transfer, gifs, stack_of, a, out,
                 round(time.time() - t0, 1))
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")


def render_gifs(rolls, specs, vae, rnn, out: Path, a) -> Dict[str, str]:
    """A real episode with a long required move, and the same policy dreaming."""
    from .dream_env import load_start_pool

    ctrl = next(sp for sp in specs if sp.name == a.gif_run).ctrl
    tbl = occlusion_table(rolls[a.gif_run])
    pick = pick_demo_episode(tbl)
    info: Dict[str, str] = {}
    if pick is None:
        # Fall back to the longest required move regardless of what the paddle
        # did: an honest GIF of the failure is better than no GIF.
        hid = [v for v in tbl if v.get("hidden") and v["hidden_frames"] >= 6]
        pick = max(hid, key=lambda v: v["required_move"]) if hid else None
        print("  no episode met the demo criteria; falling back to the longest "
              "required move regardless of what the paddle did")
    if pick is not None:
        seed = a.seed_base + pick["episode"]
        roll = run_real_episodes(
            ctrl, vae, rnn, episodes=1, steps=a.steps, seed_base=seed,
            device=a.device, record_frames=1, ball_radius=0.08, occluder=True,
        )
        save_gif(roll["frames"][0], out / f"real_play_{a.gif_run}.gif",
                 fps=20, scale=4)
        info["real"] = (
            f"seed {seed}: required move {pick['required_move']:.2f}, "
            f"{pick['hidden_frames']} hidden frames, "
            f"{pick['toward_frames']} of them moving toward the landing x, "
            f"{int(contact_runs(roll['hits'])[0])} interceptions in the episode. "
            "CHERRY-PICKED: chosen out of the evaluation set as the longest "
            "required move the policy moved through, so it is a demo of the "
            "behaviour, not a sample of it."
        )
        print("  real GIF:", info["real"])
    pool = load_start_pool(a.dream_roots, warmup=8)
    dream_play_gif(ctrl, rnn, vae, pool, out / f"dream_play_{a.gif_run}.gif",
                   steps=a.steps, temperature=1.0, seed=a.seed_base,
                   device=a.device)
    info["dream"] = ("the controller inside its own dream, decoded by V. "
                     "Best of 16 dreams by predicted contact (a cherry-pick, "
                     "as in v1/v2). Watch whether a ball re-emerges below the "
                     "band at all.")
    return info


def write_report(rows, band_rows, recon, transfer, gifs, stack_of, a,
                 out: Path, wall: float) -> None:
    payload = {
        "stacks": {k: {"rnn": v[0], "vae": v[1]} for k, v in stack_of.items()},
        "in_distribution": rows,
        "bands": band_rows,
        "vae_recon": recon,
        "transfer": transfer,
        "gifs": gifs,
        "episodes": a.episodes,
        "steps": a.steps,
        "seed_base": a.seed_base,
        "required_move_edges": list(REQUIRED_MOVE_EDGES),
        "wall_clock_s": wall,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2))

    L = [
        "# v3 stage three (C) — real-environment evaluation by occlusion",
        "",
        f"{a.episodes} episodes x {a.steps} steps, seeds "
        f"{a.seed_base}..{a.seed_base + a.episodes - 1}, identical starts for "
        "every row. Band (0.28, 0.58): the ball is fully hidden on ~18 % of "
        "frames and partly hidden on ~39 %.",
        "",
        "`oracle` has perfect vision and perfect memory; **`wait_and_see` has "
        "perfect vision and NO memory** (it tracks the true ball only while "
        "`ball_visible > 0.5` and STAYs otherwise). Every trained row should be "
        "read as a position between those two. `ctrl_v3_poshead` is a "
        "**PRIVILEGED CEILING** — its dynamics model was trained with a "
        "supervised head on the simulator's true ball position — and is not a "
        "world-model result.",
        "",
        "Each controller is driven by the (V, M) it was trained with; the "
        "policy reads `h`, so M is part of the policy:",
        "",
        "| controller | M (dynamics) | V (encoder) |",
        "|---|---|---|",
    ] + [f"| `{k}` | `{v[0]}` | `{v[1]}` |" for k, v in sorted(stack_of.items())]

    L += ["", "## Interceptions per floor visit, by required move", "",
          "The **required move** is the distance between the paddle's x at the "
          "moment the ball became fully hidden on its way down and the ball's "
          "landing x. Flat across the bins = object permanence; collapsing on "
          "`long` = memoryless.", ""]
    L += by_move_table(rows, "interceptions_per_visit", ci=True)
    L += ["",
          "**The bins are policy-dependent and that is not a bug to hide.** The "
          "required move is measured from *this policy's own* paddle position, "
          "so a policy that already tends to stand near the ball generates few "
          "long visits and a policy that parks in a corner generates many. The "
          "counts are therefore reported per row, and a row with a handful of "
          "long visits should be read as noise. The episodes and seeds are "
          "identical throughout; only the paddle differs.",
          "",
          "| controller | visits (short / medium / long) | mean required move "
          "(short / medium / long) | visits with no occlusion on the descent |",
          "|---|---|---|---|"]
    for r in rows:
        b = r["by_required_move"]
        L.append(
            f"| `{r['name']}` | "
            + " / ".join(str(b[k]["n_visits"]) for k in MOVE_BINS) + " | "
            + " / ".join(_fmt(b[k]["mean_required_move"], 3) for k in MOVE_BINS)
            + f" | {r['n_visits'] - r['n_visits_hidden']} |"
        )

    L += ["", "## The reach curve (interceptions per visit at 0.1 resolution)",
          "",
          "The paddle is **0.26 wide** and a descending ball is visible for "
          "~10 frames before it can be touched, which buys ~0.30 of travel. So "
          "a required move is covered whenever it is under roughly "
          "`0.13 + 0.30 = 0.43` even with no memory at all. This table says "
          "where `wait_and_see` actually falls off, and therefore which bins "
          "above carry any information about object permanence.", "",
          "| controller | " + " | ".join(FINE_BINS) + " |",
          "|---" * (len(FINE_BINS) + 1) + "|"]
    for r in rows:
        c = r["reach_curve"]
        L.append(f"| `{r['name']}` | " + " | ".join(
            _fmt(c[b]["interceptions_per_visit"]) for b in FINE_BINS) + " |")
    L += ["", "| controller | visits per slice |", "|---|---|"]
    for r in rows:
        c = r["reach_curve"]
        L.append(f"| `{r['name']}` | "
                 + " / ".join(str(c[b]["n_visits"]) for b in FINE_BINS) + " |")

    L += ["", "## Paddle motion during occlusion", "",
          "Over the descending hidden runs: the fraction of hidden frames whose "
          "ACTION moved the paddle toward the eventual landing x (chance "
          "**1/3** — three actions, STAY counts as not-toward), and the "
          "displacement the paddle actually achieved toward the landing point "
          "while blind, as a fraction of the required move.",
          "",
          "`oracle` is **not** the ceiling for these two columns and should not "
          "be read as one: it chases the ball's CURRENT x rather than its "
          "landing x, and it is usually already inside its dead zone when the "
          "ball vanishes, so it STAYs. The references that mean something here "
          "are chance (1/3) and `wait_and_see` (0.000 by construction).", ""]
    L += occlusion_table_md(rows)

    L += ["", "## Overall", ""]
    L += ["| controller | interceptions/visit | interceptions/ep | floor "
          "visits/ep | gap at floor | approach sign agreement | left/stay/right |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r["action_dist"]
        L.append(
            f"| `{r['name']}` | {_fmt(r['interceptions_per_visit'])} "
            f"[{_fmt(r['interceptions_per_visit_ci95'][0])}, "
            f"{_fmt(r['interceptions_per_visit_ci95'][1])}] | "
            f"{_fmt(r['interceptions_per_episode'])} | "
            f"{_fmt(r['floor_visits_per_episode'])} | "
            f"{_fmt(r['mean_gap_at_floor'], 3)} | "
            f"{_fmt(r['sign_agreement_approach'], 3)} | "
            f"{d['left']:.2f}/{d['stay']:.2f}/{d['right']:.2f} |"
        )

    for band, brows in band_rows.items():
        L += ["", f"## Taller band {band}", "",
              f"{a.band_episodes} episodes. Neither V nor M nor C has ever seen "
              "this band. Read the VAE reconstruction table below before "
              "attributing a drop to the controller.", "",
              "| controller | interceptions/visit | floor visits/ep | "
              "toward fraction (move > 0.15) | displacement fraction | "
              "mean hidden frames |", "|---|---|---|---|---|---|"]
        for r in brows:
            ol = r["occlusion_long"]
            L.append(
                f"| `{r['name']}` | {_fmt(r['interceptions_per_visit'])} "
                f"[{_fmt(r['interceptions_per_visit_ci95'][0])}, "
                f"{_fmt(r['interceptions_per_visit_ci95'][1])}] | "
                f"{_fmt(r['floor_visits_per_episode'])} | "
                f"{_fmt(ol['toward_fraction'], 3)} | "
                f"{_fmt(ol['displacement_fraction'], 3)} | "
                f"{_fmt(r['occlusion']['mean_hidden_frames'], 1)} |"
            )

    L += ["", "## VAE reconstruction on each band", "",
          "The frozen v3 encoder+decoder on frames from datasets collected at "
          "the three band heights, so a taller-band drop can be attributed to V "
          "or to C.", "",
          "| dataset | band | recon MSE | mean hidden run (frames) |",
          "|---|---|---|---|"]
    for root, d in recon.items():
        L.append(f"| `{root}` | {d['occluder_y']} | {d['recon_mse']:.5f} | "
                 f"{_fmt(d.get('mean_hidden_run_frames'), 1)} |")

    if transfer:
        L += ["", "## Dream-vs-real transfer", "",
              "Pearson r between the smoothed best-of-generation dream fitness "
              "and the periodic real interception count, per training run "
              "(`dream_vs_real_all.png`, and `dream_vs_real.png` in each run "
              "directory).", "",
              "| run | r (smoothed) | r (raw) | best real int./ep | final dream "
              "fitness | dream env steps | real env steps | wall clock (s) |",
              "|---|---|---|---|---|---|---|---|"]
        for nm, d in transfer.items():
            L.append(
                f"| `{nm}` | {_fmt(d['pearson_r_smoothed'])} | "
                f"{_fmt(d['pearson_r_raw'])} | {_fmt(d['best_real'])} | "
                f"{_fmt(d['final_dream'], 1)} | {d['dream_env_steps']:,} | "
                f"{d['real_env_steps_total']:,} | {d['wall_clock_s']:.0f} |"
            )

    if gifs:
        L += ["", "## GIFs", ""] + [f"* **{k}** — {v}" for k, v in gifs.items()]

    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

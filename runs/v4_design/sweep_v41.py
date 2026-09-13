"""Design sweep for v4.1: the side-wind. Does the sign matter NOW?

``sweep_v4.py`` priced the hidden gravity sign and found it was worth exactly
nothing to the controller, at any magnitude in the range: gap 0.00 in all eight
cells, while the stall rate went from 0% to 43%. Its own diagnosis was that the
knob was the wrong one. A vertical acceleration moves the landing *height*, and
height converts into *x* only through the ball's slow horizontal speed, so a
wrong sign was worth at most 0.035 in x -- a quarter of a paddle -- and even
that shrank to zero as the ball arrived. The paddle never had to commit.

v4.1 turns the acceleration ninety degrees: ``gravity_axis="x"``, a side-wind
whose direction is the hidden latent and still flips on every paddle contact.
Now the sign's effect on the landing point is ``a t² / 2`` **in x directly**,
which is an order of magnitude larger; and as a bonus the vertical dynamics
become sign-independent, which removes the per-traverse energy cue the v4 sweep
found by accident (a 1.20x difference in crossing time between the signs).

Same five reference policies, same 60 fixed seeds, same headline number
(interceptions per floor visit), so the two sweeps' tables are comparable line
for line. Two things are added:

``wall-pinned``   the side-wind's own failure mode, and it is invisible to the
                  stall detector: vertical motion is untouched, so a pinned ball
                  still visits floor and ceiling on schedule while grazing one
                  side wall for the whole episode. Measured as the fraction of
                  frames with the ball within one radius of a side wall. Note
                  the geometric floor: a ball uniform in x is inside that band
                  ~19% of the time at r = 0.08, which is why the axis-y
                  reference row is in the table.
``paddle_w``      v3.1's lever. A narrower paddle cannot absorb as large a
                  landing error, so it raises the price of a wrong sign without
                  touching the physics. 0.26 is v1's; 0.16 is v3.1's.

    PYTHONPATH=. /opt/miniconda3/envs/NN/bin/python runs/v4_design/sweep_v41.py

Criterion: the smallest wind with gap(ballistic - sign_blind) >= 0.15 and at
most 5% of episodes stalled or pinned, preferring paddle_w 0.26.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldsim.bouncing_box import (
    EVENT_FLIP,
    EVENT_PADDLE,
    LAUNCH_MIN_ANGLE_V1_DEG,
    BouncingBox,
    BoxConfig,
)
from worldsim.collect import stuck_fraction, wall_pinned_fraction
from wm.controller import (
    BallisticOracleController,
    ballistic_landing_x,
    OracleController,
    RandomController,
    SignBlindOracleController,
    StayController,
)
from wm.eval_controller import contact_runs, floor_visit_stats

EPISODES = 60
STEPS = 200
SEED_BASE = 5000
BALL_R = 0.08

# (gravity_axis, wind, paddle_w, launch_min_angle_deg)
#
# The axis-y row is the v4 default, carried over verbatim as a reference line:
# it is what every number in sweep.md was measured at, and it is the control
# for the two new columns (a wall-pinned fraction means nothing without the
# unpinned baseline to compare it to).
CELLS = [
    ("x", 0.00005, 0.26, LAUNCH_MIN_ANGLE_V1_DEG),
    ("x", 0.00005, 0.16, LAUNCH_MIN_ANGLE_V1_DEG),
    ("x", 0.0001, 0.26, LAUNCH_MIN_ANGLE_V1_DEG),
    ("x", 0.0001, 0.16, LAUNCH_MIN_ANGLE_V1_DEG),
    ("x", 0.0002, 0.26, LAUNCH_MIN_ANGLE_V1_DEG),
    ("x", 0.0002, 0.16, LAUNCH_MIN_ANGLE_V1_DEG),
    ("y", 0.0001, 0.26, 40.0),
]

# Appendix. Not candidate cells -- a test of the v4 sweep's own diagnosis.
# sweep.md said the blocker is SLACK (``t_land - |dx| / paddle_speed``), which
# is set by how fast the paddle crosses the box, and that gravity was the wrong
# dial. If that is right, then holding the side-wind fixed and only slowing the
# paddle should open the gap the wind could not. If it is wrong, nothing here
# moves either and the whole v4 C-stage question is dead rather than merely
# mis-parameterised. Either answer is worth 90 seconds.
# (axis, wind, paddle_w, angle, paddle_speed)
APPENDIX_CELLS = [
    ("x", 0.0002, 0.16, LAUNCH_MIN_ANGLE_V1_DEG, 0.030),   # the sweep's cell
    ("x", 0.0002, 0.16, LAUNCH_MIN_ANGLE_V1_DEG, 0.012),
    ("x", 0.0002, 0.16, LAUNCH_MIN_ANGLE_V1_DEG, 0.006),
    ("y", 0.0001, 0.16, 40.0, 0.006),                      # vertical, for contrast
]

GAP_WANTED = 0.15
BAD_WANTED = 0.05
PINNED_LIMIT = 0.20   # an episode is "pinned" if it spends more than this
                      # fraction of its frames within one radius of a side wall

Z0 = np.zeros((1, 16))
H0 = np.zeros((1, 256))


def run(cfg: BoxConfig, ctrl, episodes: int = EPISODES, steps: int = STEPS) -> dict:
    """One (config, controller) cell. Identical to ``sweep_v4.run`` plus pinning."""
    inter = visits = 0.0
    flips = 0
    stuck_eps = pinned_eps = bad_eps = 0
    pinned_frac_total = 0.0
    speeds = []
    env = BouncingBox(cfg)
    for i in range(episodes):
        env.reset(seed=SEED_BASE + i)
        if hasattr(ctrl, "reset"):
            ctrl.reset(1, seed=SEED_BASE + i)
        st = [env.state()]
        hits = []
        for _t in range(steps):
            a = int(np.asarray(ctrl.act(Z0, H0, state=st[-1][None]))[0])
            _f, s, ev = env.step(a)
            st.append(s)
            hits.append(float((ev & EVENT_PADDLE) != 0))
            flips += int((ev & EVENT_FLIP) != 0)
        states = np.stack(st)[None]                       # (1, T+1, S)
        speed = np.linalg.norm(states[..., 2:4], axis=-1)
        fv = floor_visit_stats(
            states, ball_radius=cfg.ball_radius, paddle_h=cfg.paddle_h,
            speed=np.array([speed.max()]),
        )
        visits += float(fv["floor_visits"].sum())
        inter += float(contact_runs(np.asarray(hits)[None])[0])
        speeds.append(float(speed.max()))
        stalled = stuck_fraction(states, cfg, window=100) > 0.0
        pf = wall_pinned_fraction(states, cfg)
        pinned_frac_total += pf
        stuck_eps += int(stalled)
        pinned_eps += int(pf > PINNED_LIMIT)
        # The selection criterion is on episodes that are unusable for EITHER
        # reason, not on the two rates separately -- an episode can be both.
        bad_eps += int(stalled or pf > PINNED_LIMIT)
    return {
        "inter_per_visit": inter / max(visits, 1.0),
        "visits_per_episode": visits / episodes,
        "flips_per_episode": flips / episodes,
        "frac_episodes_stuck": stuck_eps / episodes,
        "frac_episodes_pinned": pinned_eps / episodes,
        "frac_episodes_bad": bad_eps / episodes,
        "mean_frac_frames_wall_pinned": pinned_frac_total / episodes,
        "max_speed_seen": max(speeds),
    }


def recoverability(cfg: BoxConfig, episodes: int = 60, steps: int = STEPS) -> dict:
    """How much does the sign move the landing point, and is there time to fix it?

    Identical in structure to ``sweep_v4.recoverability`` -- the whole argument
    of the v4 sweep lives in this table, so v4.1 has to be judged on the same
    one. ``slack = frames_until_landing - |dx| / paddle_speed`` is how many
    frames of paddle travel the blind oracle has spare to undo its own error.
    In v4 it was never negative on any of 4,171 approach frames, which is
    exactly why the gap was zero. A side-wind that works is a side-wind that
    makes it negative often.
    """
    env = BouncingBox(cfg)
    y_land = cfg.paddle_h + cfg.ball_radius
    recs = []
    for i in range(episodes):
        env.reset(seed=SEED_BASE + i)
        st = [env.state()]
        for _t in range(steps):
            _f, s, _ev = env.step(1)          # STAY: nothing perturbs the fall
            st.append(s)
        S = np.stack(st)
        y = S[:, 1]
        cross = np.flatnonzero((y[:-1] > y_land) & (y[1:] <= y_land)) + 1
        for t in range(len(S) - 1):
            nxt = cross[cross > t]
            if not len(nxt) or S[t, 3] >= 0.0:      # only descending approaches
                continue
            t_land = int(nxt[0]) - t
            if t_land > 120:
                continue
            true_sign = float(S[t, -1])
            kw = dict(gravity_axis=cfg.gravity_axis)
            xt = ballistic_landing_x(S[t], cfg.gravity, true_sign,
                                     cfg.ball_radius, cfg.paddle_h, **kw)
            xb = ballistic_landing_x(S[t], cfg.gravity, -1.0,
                                     cfg.ball_radius, cfg.paddle_h, **kw)
            d = abs(xt - xb)
            recs.append((t_land, d, t_land - d / cfg.paddle_speed))
    a = np.asarray(recs)
    buckets = [(1, 10), (10, 20), (20, 40), (40, 80), (80, 120)]
    rows = []
    for lo, hi in buckets:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if not m.any():
            continue
        rows.append({
            "frames_to_landing": f"{lo}-{hi}",
            "n": int(m.sum()),
            "mean_pred_gap": float(a[m, 1].mean()),
            "frac_gap_over_half_paddle": float(
                (a[m, 1] > 0.5 * cfg.paddle_w).mean()),
            "mean_slack_frames": float(a[m, 2].mean()),
            "frac_slack_negative": float((a[m, 2] < 0).mean()),
        })
    return {
        "paddle_w": float(cfg.paddle_w),
        "n_approach_frames": int(len(a)),
        "frac_slack_negative_overall": float((a[:, 2] < 0).mean()),
        "by_frames_to_landing": rows,
    }


def print_recoverability(tag: str, cfg: BoxConfig, rec: dict) -> None:
    print(f"\nwhy: how much slack does a wrong sign leave?  [{tag}] "
          f"({rec['n_approach_frames']} descending frames, "
          f"paddle_w={cfg.paddle_w})")
    print(f"  {'frames to landing':>18} {'n':>7} {'|dx| pred':>10} "
          f"{'> w/2':>7} {'slack (fr)':>11} {'slack<0':>8}")
    for r in rec["by_frames_to_landing"]:
        print(f"  {r['frames_to_landing']:>18} {r['n']:7d} "
              f"{r['mean_pred_gap']:10.3f} "
              f"{r['frac_gap_over_half_paddle']:7.1%} "
              f"{r['mean_slack_frames']:11.1f} "
              f"{r['frac_slack_negative']:8.2%}")
    print(f"  overall frames with negative slack: "
          f"{rec['frac_slack_negative_overall']:.3%}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = []
    hdr = (f"{'axis':>5} {'wind':>9} {'pad_w':>6} {'ang':>5} {'oracle':>7} "
           f"{'ballistic':>10} {'signblind':>10} {'gap':>6} {'stay':>6} "
           f"{'random':>7} {'visits':>7} {'flips':>6} {'stall':>6} "
           f"{'pinned':>7} {'pin_fr':>7}")
    print(hdr)
    print("-" * len(hdr))
    t0 = time.time()
    for axis, g, pw, ang in CELLS:
        cfg = BoxConfig(res=64, ball_radius=BALL_R, gravity=g,
                        gravity_axis=axis, launch_min_angle_deg=ang,
                        paddle_w=pw)
        common = dict(gravity=g, paddle_w=pw, ball_radius=BALL_R,
                      paddle_h=cfg.paddle_h, gravity_axis=axis)
        o = run(cfg, OracleController(paddle_w=pw))
        b = run(cfg, BallisticOracleController(**common))
        sb = run(cfg, SignBlindOracleController(**common))
        stc = run(cfg, StayController())
        rd = run(cfg, RandomController(mean_hold=8.0, seed=1234))
        # The gap is measured from the BALLISTIC oracle, not from the best of
        # the two oracles: the sign-blind controller is the ballistic one minus
        # one bit, and the tracker is a different algorithm. (In v4 the two
        # oracles tied, so this made no difference there; under a side-wind it
        # can, and the honest comparison is the matched one.)
        gap = b["inter_per_visit"] - sb["inter_per_visit"]
        row = {
            "gravity_axis": axis, "gravity": g, "paddle_w": pw,
            "launch_min_angle_deg": ang,
            "oracle": o["inter_per_visit"],
            "ballistic_oracle": b["inter_per_visit"],
            "sign_blind_oracle": sb["inter_per_visit"],
            "gap_ballistic_minus_sign_blind": gap,
            "gap_best_oracle_minus_sign_blind":
                max(o["inter_per_visit"], b["inter_per_visit"])
                - sb["inter_per_visit"],
            "stay": stc["inter_per_visit"],
            "random": rd["inter_per_visit"],
            "visits_per_episode": o["visits_per_episode"],
            "flips_per_episode_oracle": o["flips_per_episode"],
            "frac_episodes_stuck": o["frac_episodes_stuck"],
            "frac_episodes_pinned": o["frac_episodes_pinned"],
            "frac_episodes_bad": o["frac_episodes_bad"],
            "mean_frac_frames_wall_pinned": o["mean_frac_frames_wall_pinned"],
            "max_speed_seen": o["max_speed_seen"],
            # The behaviour-policy view of the same two hazards: the datasets
            # are collected under random/mix, not under the oracle, so a cell
            # that is only clean when a perfect player is at the controls is
            # not clean.
            "frac_episodes_bad_random": rd["frac_episodes_bad"],
            "mean_frac_frames_wall_pinned_random":
                rd["mean_frac_frames_wall_pinned"],
        }
        rows.append(row)
        print(f"{axis:>5} {g:9.5f} {pw:6.2f} {ang:5.1f} {row['oracle']:7.2f} "
              f"{row['ballistic_oracle']:10.2f} "
              f"{row['sign_blind_oracle']:10.2f} {gap:6.2f} "
              f"{row['stay']:6.2f} {row['random']:7.2f} "
              f"{row['visits_per_episode']:7.2f} "
              f"{row['flips_per_episode_oracle']:6.2f} "
              f"{row['frac_episodes_stuck']:6.0%} "
              f"{row['frac_episodes_pinned']:7.0%} "
              f"{row['mean_frac_frames_wall_pinned']:7.1%}", flush=True)

    # ---------------------------------------------------------- the choice
    ok = [r for r in rows
          if r["gravity_axis"] == "x"
          and r["gap_ballistic_minus_sign_blind"] >= GAP_WANTED
          and r["frac_episodes_bad"] <= BAD_WANTED
          and r["frac_episodes_bad_random"] <= BAD_WANTED]
    # Smallest wind first, then v1's wide paddle before v3.1's narrow one.
    ok.sort(key=lambda r: (r["gravity"], -r["paddle_w"]))
    chosen = ok[0] if ok else None
    print()
    if chosen is None:
        print("NO CELL QUALIFIES "
              f"(need gap >= {GAP_WANTED}, bad episodes <= {BAD_WANTED:.0%})")
    else:
        print(f"CHOSEN: axis={chosen['gravity_axis']} "
              f"wind={chosen['gravity']:g} paddle_w={chosen['paddle_w']} "
              f"gap={chosen['gap_ballistic_minus_sign_blind']:.2f} "
              f"bad={chosen['frac_episodes_bad']:.0%}  "
              f"({len(ok)} cells qualified)")

    # ------------------------------------------------- the recoverability table
    recs = {}
    tags = []
    if chosen is not None:
        tags.append(("chosen", chosen["gravity_axis"], chosen["gravity"],
                     chosen["paddle_w"], chosen["launch_min_angle_deg"]))
    else:
        for r in rows:
            if r["gravity_axis"] == "x":
                tags.append((f"x_g{r['gravity']:g}_w{r['paddle_w']}",
                             "x", r["gravity"], r["paddle_w"],
                             r["launch_min_angle_deg"]))
    tags.append(("v4_reference_axis_y", "y", 1e-4, 0.26, 40.0))
    for tag, axis, g, pw, ang in tags:
        cfg = BoxConfig(res=64, ball_radius=BALL_R, gravity=g,
                        gravity_axis=axis, launch_min_angle_deg=ang,
                        paddle_w=pw)
        rec = recoverability(cfg)
        recs[tag] = {"gravity_axis": axis, "gravity": g, **rec}
        print_recoverability(tag, cfg, rec)

    # ------------------------------------------------------------- appendix
    print("\nappendix: is the blocker the WIND or the PADDLE's speed?")
    ahdr = (f"{'axis':>5} {'wind':>9} {'pad_w':>6} {'pad_spd':>8} "
            f"{'ballistic':>10} {'signblind':>10} {'gap':>6} {'stay':>6} "
            f"{'slack<0':>8}")
    print(ahdr)
    print("-" * len(ahdr))
    app = []
    for axis, g, pw, ang, ps in APPENDIX_CELLS:
        cfg = BoxConfig(res=64, ball_radius=BALL_R, gravity=g,
                        gravity_axis=axis, launch_min_angle_deg=ang,
                        paddle_w=pw, paddle_speed=ps)
        common = dict(gravity=g, paddle_w=pw, ball_radius=BALL_R,
                      paddle_h=cfg.paddle_h, gravity_axis=axis)
        b = run(cfg, BallisticOracleController(**common), episodes=40)
        sb = run(cfg, SignBlindOracleController(**common), episodes=40)
        stc = run(cfg, StayController(), episodes=40)
        rec = recoverability(cfg, episodes=30)
        rowa = {
            "gravity_axis": axis, "gravity": g, "paddle_w": pw,
            "paddle_speed": ps,
            "ballistic_oracle": b["inter_per_visit"],
            "sign_blind_oracle": sb["inter_per_visit"],
            "gap": b["inter_per_visit"] - sb["inter_per_visit"],
            "stay": stc["inter_per_visit"],
            "frac_slack_negative": rec["frac_slack_negative_overall"],
        }
        app.append(rowa)
        print(f"{axis:>5} {g:9.5f} {pw:6.2f} {ps:8.3f} "
              f"{rowa['ballistic_oracle']:10.2f} "
              f"{rowa['sign_blind_oracle']:10.2f} {rowa['gap']:6.2f} "
              f"{rowa['stay']:6.2f} {rowa['frac_slack_negative']:8.1%}",
              flush=True)

    (out_dir / "sweep_v41.json").write_text(json.dumps(
        {"cells": rows, "chosen": chosen, "recoverability": recs,
         "appendix_paddle_speed": app}, indent=2))
    print(f"\nwrote {out_dir/'sweep_v41.json'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

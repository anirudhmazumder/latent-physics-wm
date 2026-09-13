"""Design sweep for v4: pick the gravity at which forgetting the sign costs you.

Run BEFORE anything is trained. This is the v3 lesson as a habit: v3 built a
whole environment around object permanence and only afterwards discovered that
a two-line memoryless oracle caught 99% of the balls, so nothing in the tier
ever needed the thing it was measuring. The way not to repeat that is to price
the hidden variable with privileged controllers first, while changing a number
is still free.

Five reference policies, all on the same 60 fixed seeds per cell:

``oracle``            ``tracking_action`` on the true state -- aim at where the
                      ball IS. The v1-v3.1 ceiling. Under gravity it is no
                      longer obviously a ceiling, which is the first thing the
                      sweep has to find out.
``ballistic_oracle``  aim at where the ball WILL LAND, solved with the TRUE
                      gravity sign. The anticipation ceiling.
``sign_blind_oracle`` the same solver, always assuming gravity points down.
                      Everything the ballistic oracle has except the one bit.
                      THE MEMORYLESS BOUND.
``stay``              never move. What luck alone scores.
``random``            sticky random actions -- the behaviour policy, which is a
                      harder floor than "stay" because it sweeps the box.

The headline number is interceptions per floor visit: interceptions rather than
contact frames (no pinning loophole) and per CHANCE rather than per episode,
because under gravity the number of chances is itself a function of the cell.

    PYTHONPATH=. /opt/miniconda3/envs/NN/bin/python runs/v4_design/sweep_v4.py

Every cell also reports the fraction of episodes that STALL -- the ball never
reaching either end of the box for 100+ consecutive frames. A cell can have a
beautiful oracle gap and be unusable because half of its episodes are a ball
hovering in the middle of the box, and that failure mode grows with gravity for
exactly the reason the gap does.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worldsim.bouncing_box import EVENT_FLIP, EVENT_PADDLE, BouncingBox, BoxConfig
from worldsim.collect import stuck_fraction
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
GRAVITIES = [0.00005, 0.0001, 0.00015, 0.0002]
ANGLES = [40.0, 50.0]

# The controllers never see z or h, but ``BaseController.act`` takes them, so
# the harness hands over correctly-shaped zeros. Allocated once: a fresh
# (1, 256) array per step is 12,000 allocations per cell for nothing.
Z0 = np.zeros((1, 16))
H0 = np.zeros((1, 256))


def run(cfg: BoxConfig, ctrl, episodes: int = EPISODES, steps: int = STEPS) -> dict:
    """One (config, controller) cell. Returns the interception statistics.

    Episode-at-a-time rather than in lockstep: there is no encoder and no RNN
    here, so the whole thing is numpy physics and batching would only add
    bookkeeping. 60 x 200 steps is about two seconds.
    """
    inter = visits = 0.0
    flips = 0
    stuck_eps = 0
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
        # Per-episode MAX speed sets the floor-visit band; see the note in
        # wm.eval_controller.run_real_episodes for why the max and not the mean.
        fv = floor_visit_stats(
            states, ball_radius=cfg.ball_radius, paddle_h=cfg.paddle_h,
            speed=np.array([speed.max()]),
        )
        visits += float(fv["floor_visits"].sum())
        inter += float(contact_runs(np.asarray(hits)[None])[0])
        speeds.append(float(speed.max()))
        # "Stalled" at the episode level: any frame whose preceding 100 frames
        # contain no visit to either end of the box.
        if stuck_fraction(states, cfg, window=100) > 0.0:
            stuck_eps += 1
    return {
        "inter_per_visit": inter / max(visits, 1.0),
        "visits_per_episode": visits / episodes,
        "flips_per_episode": flips / episodes,
        "frac_episodes_stuck": stuck_eps / episodes,
        "max_speed_seen": max(speeds),
    }


def recoverability(cfg: BoxConfig, episodes: int = 60, steps: int = STEPS) -> dict:
    """WHY the sign-blind oracle is not punished. Run on the chosen cell.

    A zero gap in the table above is a result, but on its own it is not an
    explanation, and without the explanation you cannot tell whether a
    different number would have fixed it. So: at every frame of a descending
    approach, measure

        ``d``     how far apart the two landing predictions are, and
        ``slack`` ``frames_until_landing - d / paddle_speed``,

    i.e. how many frames of paddle travel the blind oracle has *spare* to undo
    its own error once the error becomes apparent. Slack is the quantity that
    decides everything: the wrong sign produces a real, large prediction error
    early in the approach, but it shrinks continuously as the ball falls (the
    gravity term is ``a t²/2``, and t is going to zero), while the paddle moves
    at a flat 0.030 per frame. If slack is comfortably positive everywhere,
    "commit early" was never required and a memoryless policy simply waits.

    A negative slack fraction near zero therefore says, precisely: the task is
    not that the sign is unknowable, it is that the sign does not have to be
    known IN TIME. That is a statement about the paddle's speed and the box's
    height, not about gravity, and it is what a v4.1 would have to change.
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
        # Frames-to-landing, looking forward to the next downward crossing.
        cross = np.flatnonzero((y[:-1] > y_land) & (y[1:] <= y_land)) + 1
        for t in range(len(S) - 1):
            nxt = cross[cross > t]
            if not len(nxt) or S[t, 3] >= 0.0:      # only descending approaches
                continue
            t_land = int(nxt[0]) - t
            if t_land > 120:
                continue
            true_sign = float(S[t, -1])
            xt = ballistic_landing_x(S[t], cfg.gravity, true_sign,
                                     cfg.ball_radius, cfg.paddle_h)
            xb = ballistic_landing_x(S[t], cfg.gravity, -1.0,
                                     cfg.ball_radius, cfg.paddle_h)
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
        "n_approach_frames": int(len(a)),
        "frac_slack_negative_overall": float((a[:, 2] < 0).mean()),
        "by_frames_to_landing": rows,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = []
    hdr = (f"{'gravity':>9} {'angle':>6} {'oracle':>7} {'ballistic':>10} "
           f"{'signblind':>10} {'gap':>6} {'stay':>6} {'random':>7} "
           f"{'visits':>7} {'flips':>6} {'stuck':>6}")
    print(hdr)
    print("-" * len(hdr))
    t0 = time.time()
    for g in GRAVITIES:
        for ang in ANGLES:
            cfg = BoxConfig(res=64, ball_radius=BALL_R,
                            gravity=g, launch_min_angle_deg=ang)
            common = dict(gravity=g, paddle_w=cfg.paddle_w,
                          ball_radius=BALL_R, paddle_h=cfg.paddle_h)
            o = run(cfg, OracleController(paddle_w=cfg.paddle_w))
            b = run(cfg, BallisticOracleController(**common))
            sb = run(cfg, SignBlindOracleController(**common))
            st = run(cfg, StayController())
            rd = run(cfg, RandomController(mean_hold=8.0, seed=1234))
            best = max(o["inter_per_visit"], b["inter_per_visit"])
            row = {
                "gravity": g, "launch_min_angle_deg": ang,
                "oracle": o["inter_per_visit"],
                "ballistic_oracle": b["inter_per_visit"],
                "sign_blind_oracle": sb["inter_per_visit"],
                "gap_best_oracle_minus_sign_blind": best - sb["inter_per_visit"],
                "stay": st["inter_per_visit"],
                "random": rd["inter_per_visit"],
                "visits_per_episode": o["visits_per_episode"],
                "flips_per_episode_oracle": o["flips_per_episode"],
                "frac_episodes_stuck": o["frac_episodes_stuck"],
                "max_speed_seen": o["max_speed_seen"],
            }
            rows.append(row)
            print(f"{g:9.5f} {ang:6.0f} {row['oracle']:7.2f} "
                  f"{row['ballistic_oracle']:10.2f} "
                  f"{row['sign_blind_oracle']:10.2f} "
                  f"{row['gap_best_oracle_minus_sign_blind']:6.2f} "
                  f"{row['stay']:6.2f} {row['random']:7.2f} "
                  f"{row['visits_per_episode']:7.2f} "
                  f"{row['flips_per_episode_oracle']:6.2f} "
                  f"{row['frac_episodes_stuck']:6.0%}", flush=True)
    # The explanation, on the design's default cell.
    cfg = BoxConfig(res=64, ball_radius=BALL_R, gravity=1e-4,
                    launch_min_angle_deg=40.0)
    rec = recoverability(cfg)
    print(f"\nwhy: how much slack does a wrong sign leave? "
          f"(g=1e-4, 40deg, {rec['n_approach_frames']} descending frames)")
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

    (out_dir / "sweep_v4.json").write_text(
        json.dumps({"cells": rows, "recoverability_g1e-4_40deg": rec}, indent=2))
    print(f"\nwrote {out_dir/'sweep_v4.json'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

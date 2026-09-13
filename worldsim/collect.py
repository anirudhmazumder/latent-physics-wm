"""Collect a dataset of (frames, actions, states, events) from BouncingBox.

Written as separate ``.npy`` files rather than a single ``.npz`` on purpose:
``.npy`` supports ``np.load(..., mmap_mode="r")``, so with 8 GB of RAM you can
train on a dataset larger than memory and let the OS page it in. An ``.npz``
would force the whole thing into RAM on load.

Arrays are also written incrementally via ``open_memmap``, so collection itself
never holds more than one episode in memory.

Layout (E episodes of T steps):
    frames.npy   uint8    (E, T + 1, res, res, 3)
    actions.npy  int8     (E, T)
    states.npy   float32  (E, T + 1, S)   S = 6 (v1), +1 for v2 (mass),
                                          +1 for v3 (ball_visible),
                                          +1 for v4 (gravity_sign)
    events.npy   uint8    (E, T)
    meta.json                             ``state_names`` tells you S

frames[e, t] and states[e, t] are the observation *before* actions[e, t] is
applied; frames[e, t + 1] is the result. So a transition is
(frames[e, t], actions[e, t]) -> frames[e, t + 1], and there are T transitions
from T + 1 frames.

Usage:
    python -m worldsim.collect --out data/v1/train --episodes 100 --steps 200
    python -m worldsim.collect --out data/v1/val   --episodes 10  --steps 200 --seed 1

v2 (mass from colour), with a held-out band of masses reserved for testing:

    python -m worldsim.collect --out data/v2/train --episodes 150 --steps 200 \
        --ball-radius 0.08 --mass-from-color --mass-holdout 0.85 1.2
    python -m worldsim.collect --out data/v2/holdout --episodes 30 --steps 200 \
        --ball-radius 0.08 --mass-from-color --mass-only 0.85 1.2 --policy mix --seed 21

v3 (occlusion band). Mass is left off so the two effects are not compounded:

    python -m worldsim.collect --out data/v3/train --episodes 150 --steps 200 \
        --ball-radius 0.08 --res 64 --occluder
    python -m worldsim.collect --out data/v3/tall --episodes 30 --steps 200 \
        --ball-radius 0.08 --res 64 --occluder --occluder-y 0.22 0.64 --policy mix --seed 31

v3.1 (same occluder switch, re-geometried so that a memoryless policy fails).
The band reaches down to where the ball can touch the paddle and the paddle is
narrower, which together drop the wait-and-see oracle from 1.00 to 0.48:

    python -m worldsim.collect --out data/v31/train --episodes 150 --steps 200 \
        --ball-radius 0.08 --res 64 --occluder --occluder-y 0.13 0.63 --paddle-w 0.16

v4 (the gravity switch). No occluder and no mass, so the hidden latent is the
only thing that is hidden. ``--launch-min-angle 40`` is not optional: at the
v1 angle the ball cannot cross the box against gravity and the episode
degenerates into floor-skimming.

    python -m worldsim.collect --out data/v4/train --episodes 150 --steps 200 \
        --ball-radius 0.08 --res 64 --gravity 0.0001 --launch-min-angle 40

v4.1 (the same switch, turned ninety degrees: a horizontal SIDE-WIND). Vertical
motion is then exactly v1's, so ``--launch-min-angle`` goes back to its default
-- the 40 degree rule existed only to let the ball climb against a vertical
pull. No dataset has been collected in this mode: ``runs/v4_design/sweep_v41.md``
found the controller gap is still 0.00 and stopped before collecting.

    python -m worldsim.collect --out data/v41/train --episodes 150 --steps 200 \
        --ball-radius 0.08 --res 64 --gravity 0.0002 --gravity-axis x
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.lib.format import open_memmap

from .bouncing_box import (
    EVENT_FLIP,
    EVENT_HIDDEN,
    EVENT_WALL_X,
    BouncingBox,
    BoxConfig,
    LAUNCH_MIN_ANGLE_V1_DEG,
)
from .policies import MixedPolicy, sticky_random_actions, uniform_random_actions


def collect(
    out_dir: str | Path,
    episodes: int = 100,
    steps: int = 200,
    res: int = 64,
    seed: int = 0,
    policy: str = "sticky",
    mean_hold: float = 8.0,
    ball_radius: float = 0.055,
    p_track: float = 0.5,
    mass_from_color: bool = False,
    mass_holdout: Optional[Tuple[float, float]] = None,
    mass_only: Optional[Tuple[float, float]] = None,
    occluder: bool = False,
    occluder_y: Tuple[float, float] = (0.28, 0.58),
    paddle_w: float = 0.26,
    gravity: float = 0.0,
    gravity_sign_init: str = "random",
    launch_min_angle: float = LAUNCH_MIN_ANGLE_V1_DEG,
    gravity_axis: str = "y",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = BoxConfig(
        res=res,
        ball_radius=ball_radius,
        mass_from_color=mass_from_color,
        mass_holdout=mass_holdout,
        mass_only=mass_only,
        occluder=occluder,
        occluder_y=tuple(occluder_y),
        # v3.1: the paddle's width sets the *chance* catch rate, so it is the
        # other half of "does this environment require memory" alongside the
        # band geometry (runs/v31_design/sweep.md). Recorded in meta because
        # every controller number is conditioned on it.
        paddle_w=paddle_w,
        # v4. Both inert at their defaults, so every earlier `collect` call
        # reproduces byte-for-byte.
        gravity=gravity,
        gravity_sign_init=gravity_sign_init,
        launch_min_angle_deg=launch_min_angle,
        gravity_axis=gravity_axis,
    )
    env = BouncingBox(cfg)
    state_names = env.state_names   # 6 in v1; +mass (v2), +ball_visible (v3),
                                    # +gravity_sign (v4), appended in that order
    rng = np.random.default_rng(seed)

    frames = open_memmap(
        out / "frames.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(episodes, steps + 1, res, res, 3),
    )
    actions = open_memmap(
        out / "actions.npy", mode="w+", dtype=np.int8, shape=(episodes, steps)
    )
    states = open_memmap(
        out / "states.npy",
        mode="w+",
        dtype=np.float32,
        shape=(episodes, steps + 1, len(state_names)),
    )
    events = open_memmap(
        out / "events.npy", mode="w+", dtype=np.uint8, shape=(episodes, steps)
    )

    t_start = time.time()

    for e in range(episodes):
        frames[e, 0] = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        states[e, 0] = env.state()

        # Two kinds of policy, and the difference is structural rather than
        # cosmetic. ``sticky``/``uniform`` are OPEN LOOP: the action sequence is
        # a function of the rng alone, so the whole episode can be sampled up
        # front. ``mix`` is CLOSED LOOP: a tracking action depends on where the
        # ball currently is, so it has to be queried inside the step loop.
        # We keep the pre-sampled path for the open-loop policies so that old
        # datasets reproduce byte-for-byte from the same seed.
        acts = None
        mixed = None
        if policy == "sticky":
            acts = sticky_random_actions(steps, rng, mean_hold=mean_hold)
        elif policy == "uniform":
            acts = uniform_random_actions(steps, rng)
        else:
            mixed = MixedPolicy(
                rng, p_track=p_track, mean_hold=mean_hold, paddle_w=cfg.paddle_w
            )
        if acts is not None:
            actions[e] = acts

        cur_state = env.state()
        for t in range(steps):
            a = int(acts[t]) if acts is not None else int(mixed.act(cur_state))
            if acts is None:
                actions[e, t] = a
            frame, state, ev = env.step(a)
            frames[e, t + 1] = frame
            states[e, t + 1] = state
            events[e, t] = ev
            cur_state = state

        if (e + 1) % max(1, episodes // 20) == 0 or e == episodes - 1:
            done = e + 1
            elapsed = time.time() - t_start
            rate = done * (steps + 1) / max(elapsed, 1e-9)
            print(
                f"  episode {done}/{episodes}  "
                f"({elapsed:.1f}s, {rate:.0f} frames/s)",
                flush=True,
            )

    _sanity_check(states, events, cfg, state_names)
    occ_stats = _occlusion_report(states, events, cfg, state_names) if occluder else None
    grav_stats = (
        _gravity_report(states, events, cfg, state_names, env.speed_clips)
        if gravity > 0.0 else None
    )

    for arr in (frames, actions, states, events):
        arr.flush()

    meta = {
        # The version string is what downstream code keys off to know whether a
        # mass column exists; keep it in sync with state_names.
        "version": (
            "v4_gravity" if gravity > 0.0
            else "v3_occluder" if occluder
            else ("v2_mass_from_color" if mass_from_color else "v1_bouncing_box")
        ),
        "episodes": episodes,
        "steps": steps,
        "res": res,
        "seed": seed,
        "policy": policy,
        "mean_hold": mean_hold if policy in ("sticky", "mix") else None,
        "p_track": p_track if policy == "mix" else None,
        "state_names": list(state_names),
        "mass_from_color": mass_from_color,
        "mass_holdout": list(mass_holdout) if mass_holdout else None,
        "mass_only": list(mass_only) if mass_only else None,
        "occluder": occluder,
        "occluder_y": list(cfg.occluder_y) if occluder else None,
        "paddle_w": float(cfg.paddle_w),
        # The occlusion statistics are cheap to recompute but expensive to
        # remember to recompute, and every v3 result is conditioned on them, so
        # they travel with the dataset.
        "occlusion": occ_stats,
        # v4, same contract as "occlusion": the statistics every downstream
        # claim is conditioned on travel with the dataset.
        "gravity": float(gravity),
        "gravity_axis": cfg.gravity_axis if gravity > 0.0 else None,
        "gravity_sign_init": gravity_sign_init if gravity > 0.0 else None,
        "launch_min_angle_deg": float(cfg.launch_min_angle_deg),
        "flips": grav_stats,
        "action_names": ["left", "stay", "right"],
        "config": env.config_dict(),
        "frame_convention": (
            "frames[e, t] with actions[e, t] produces frames[e, t + 1]"
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    total_frames = episodes * (steps + 1)
    mb = frames.nbytes / 1e6
    print(f"\nwrote {total_frames} frames to {out}  ({mb:.0f} MB)")
    return out


def _sanity_check(states, events, cfg: BoxConfig, state_names) -> None:
    """Cheap invariants. Catching these here beats debugging a VAE later."""
    s = np.asarray(states)
    assert np.isfinite(s).all(), "non-finite values in states"

    r = cfg.ball_radius
    x, y = s[..., 0], s[..., 1]
    slack = 1e-4
    assert x.min() >= r - slack and x.max() <= 1 - r + slack, "ball x out of bounds"
    assert y.min() >= r - slack and y.max() <= 1 - r + slack, "ball y out of bounds"

    speed = np.linalg.norm(s[..., 2:4], axis=-1)
    if cfg.gravity > 0.0:
        # v4: speed is a genuine variable now (that is the whole point), so the
        # invariant to check is not conservation but BOUNDEDNESS -- gravity is
        # conservative, the walls are elastic, and the one term that can pump
        # energy in is clipped at cfg.max_speed.
        # The bound is NOT ``max_speed``: the clip is applied at the moment of
        # contact, and the ball then falls, so gravity adds the box's worth of
        # potential energy on top. That headroom is ``sqrt(v^2 + 2 g dh)`` with
        # dh = 1 - 2r, about 0.0017 at the v4 numbers -- small, but the
        # assertion has to be the physics rather than the knob, or a correct
        # dataset trips it (and this one did).
        cap = float(np.sqrt(
            cfg.max_speed**2 + 2.0 * cfg.gravity * (1.0 - 2.0 * cfg.ball_radius)
        ))
        assert speed.max() <= cap + 1e-6, (
            f"ball reached {speed.max():.5f}, above the energy cap {cap:.5f}"
        )
        assert speed.min() > 0.0, "ball came to a dead stop"
    elif "mass" in state_names:
        # v2: speed is conserved *within* an episode but is ball_speed / m, so
        # the invariant to check is speed * mass == ball_speed. Comparing
        # against the constant here would fire on every correct v2 dataset.
        mass = s[..., state_names.index("mass")]
        assert np.allclose(speed * mass, cfg.ball_speed, atol=1e-6), \
            "effective ball speed not conserved"
        # And the mass really must be constant within an episode -- it is drawn
        # once at reset, so a per-episode std of anything but 0 means state()
        # is being filled from the wrong place.
        # max - min, not std: std subtracts a float32 mean whose summation
        # rounding makes it ~1e-8 off even for 201 identical values, so an
        # "== 0" on the std fails on perfectly correct data.
        assert float((mass.max(axis=1) - mass.min(axis=1)).max()) == 0.0, \
            "mass varies within an episode"
        _mass_report(mass[:, 0], cfg)
    else:
        assert np.allclose(speed, cfg.ball_speed, atol=1e-6), "ball speed not conserved"

    hit_rate = float((np.asarray(events) & 4).astype(bool).mean())
    speed_claim = "speed bounded" if cfg.gravity > 0.0 else "speed conserved"
    print(f"  sanity: bounds ok, {speed_claim}, paddle-contact rate {hit_rate:.3%}")
    if hit_rate < 1e-4:
        print("  warning: the ball almost never touches the paddle -- actions will "
              "look nearly irrelevant to the dynamics model")


def _mass_report(masses: np.ndarray, cfg: BoxConfig) -> None:
    """Per-split mass summary. Log-spaced bins, because the prior is log-uniform:
    on a log axis a correct sample is flat, and the held-out band shows up as a
    clean empty gap rather than as a vague dip."""
    m = np.asarray(masses, dtype=np.float64)
    edges = np.exp(np.linspace(np.log(cfg.mass_min), np.log(cfg.mass_max), 13))
    hist, _ = np.histogram(m, bins=edges)
    print(
        f"  mass: n={len(m)}  min={m.min():.3f}  median={np.median(m):.3f}  "
        f"max={m.max():.3f}  mean_eff_speed={np.mean(cfg.ball_speed / m):.4f}"
    )
    print("  mass histogram (log-spaced bins):")
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        bar = "#" * int(round(40 * c / max(hist.max(), 1)))
        print(f"    [{lo:5.3f},{hi:5.3f})  {c:4d} {bar}")


def hidden_runs(hidden: np.ndarray) -> list:
    """Lengths of maximal runs of consecutive True along the last axis.

    Takes the whole (E, T) boolean array rather than one episode at a time so
    that run detection is a single vectorised diff: pad each row with False on
    both sides, and a run is a rising edge followed by a falling edge. Returns
    a list of ``(episode, start, length)`` triples so callers can go back and
    ask what else happened during a run -- which is exactly what the "did a
    wall bounce happen while hidden" statistic needs.
    """
    h = np.asarray(hidden, dtype=bool)
    if h.ndim == 1:
        h = h[None]
    pad = np.zeros((h.shape[0], 1), dtype=bool)
    d = np.diff(np.concatenate([pad, h, pad], axis=1).astype(np.int8), axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    return [
        (int(e), int(t0), int(t1 - t0))
        for (e, t0), (_, t1) in zip(starts, ends)
    ]


def _occlusion_report(states, events, cfg: BoxConfig, state_names) -> dict:
    """v3 per-split summary. Printed AND written into meta.json.

    Three numbers decide whether a v3 split is usable, and they are not
    interchangeable:

    * how much of the data is fully hidden -- the frames where ``z`` provably
      cannot carry the ball, i.e. the ones every memory claim rests on;
    * how long a typical hidden stretch is -- the number of steps ``h`` has to
      bridge, which is what the "memory horizon" experiment varies;
    * how often the ball bounces off a side wall *while* hidden -- those are
      the runs where "keep extrapolating the last seen velocity" gives the
      wrong exit x, so they are the ones that separate a simulator from an
      extrapolator. If this is ~0 the split cannot test that claim at all.
    """
    s = np.asarray(states)
    ev = np.asarray(events)
    j = list(state_names).index("ball_visible")
    vis = s[..., j]

    hidden_f = vis < 1e-3
    full_f = vis > 1.0 - 1e-3
    part_f = ~hidden_f & ~full_f

    # Runs are detected on the EVENT array (one entry per transition, length T)
    # rather than on the states (length T+1), because the wall-bounce flag we
    # want to intersect them with lives there. events[e, t] and
    # states[e, t + 1] describe the same instant.
    hid_ev = (ev & EVENT_HIDDEN).astype(bool)
    runs = hidden_runs(hid_ev)
    lengths = np.array([n for _, _, n in runs], dtype=np.int64)
    wall_x = (ev & EVENT_WALL_X).astype(bool)
    with_bounce = sum(
        1 for e, t0, n in runs if wall_x[e, t0 : t0 + n].any()
    )

    stats = {
        "frac_frames_hidden": float(hidden_f.mean()),
        "frac_frames_partial": float(part_f.mean()),
        "frac_frames_visible": float(full_f.mean()),
        "n_hidden_runs": int(len(runs)),
        "mean_hidden_run_frames": float(lengths.mean()) if len(lengths) else 0.0,
        "median_hidden_run_frames": float(np.median(lengths)) if len(lengths) else 0.0,
        "max_hidden_run_frames": int(lengths.max()) if len(lengths) else 0,
        "frac_hidden_runs_with_wall_x": (
            float(with_bounce / len(runs)) if runs else 0.0
        ),
        "runs_per_episode": float(len(runs) / s.shape[0]),
        "occluder_y": list(cfg.occluder_y),
    }
    print(
        f"  occlusion: band y={cfg.occluder_y[0]:.2f}..{cfg.occluder_y[1]:.2f}  "
        f"frames hidden {stats['frac_frames_hidden']:.1%} / "
        f"partial {stats['frac_frames_partial']:.1%} / "
        f"visible {stats['frac_frames_visible']:.1%}"
    )
    print(
        f"    hidden runs: {stats['n_hidden_runs']} "
        f"({stats['runs_per_episode']:.1f}/episode)  "
        f"mean {stats['mean_hidden_run_frames']:.1f} frames  "
        f"median {stats['median_hidden_run_frames']:.0f}  "
        f"max {stats['max_hidden_run_frames']}  "
        f"with a wall-x bounce while hidden: "
        f"{stats['frac_hidden_runs_with_wall_x']:.1%}"
    )
    if stats["frac_frames_hidden"] < 0.02:
        print("  warning: almost nothing is ever hidden -- check --occluder-y "
              "against the ball radius")
    return stats


def stuck_fraction(
    states: np.ndarray, cfg: BoxConfig, window: int = 100
) -> float:
    """Fraction of frames at which the ball has not reached EITHER end recently.

    The one failure mode gravity can introduce that the bounds assertion would
    not catch: a ball whose vertical kinetic energy is too small to climb
    against the pull just oscillates in a shallow band, never visits the floor,
    and never gives the paddle a chance. That is not "out of bounds" and it is
    not "stopped" -- it is a *boring* episode, and a cell of the design sweep
    full of them is unusable however good its oracle gap looks.

    "Reached an end" is deliberately generous -- within ~a ball radius of the
    ceiling, or inside the band where a paddle contact is geometrically
    possible -- so this fires only on a genuinely stalled traverse and not on a
    ball that merely turns around slightly early. Frames before the first full
    window are excluded (there is nothing to look back at).
    """
    s = np.asarray(states)
    y = s[..., 1]
    r = cfg.ball_radius
    end = (y > 1.0 - r - 0.02) | (y < cfg.paddle_h + r + 0.02)
    T = end.shape[1]
    if T <= window:
        return 0.0
    # Cumulative count of "reached an end" frames, so the look-back is O(1).
    c = np.cumsum(end.astype(np.int64), axis=1)
    recent = c[:, window:] - c[:, :-window]
    return float((recent == 0).mean())


def wall_pinned_fraction(states: np.ndarray, cfg: BoxConfig) -> float:
    """v4.1: fraction of frames with the ball within one radius of a side wall.

    The side-wind's own version of ``stuck_fraction``, and it needed its own
    metric because it is invisible to that one: a pinned ball is still visiting
    the floor and the ceiling on schedule (vertical motion is untouched by a
    horizontal wind), so the stall detector sees a perfectly healthy episode
    while the ball grazes the same wall for two hundred frames. What goes wrong
    is horizontal: if the wind can reverse the ball's ``vx`` before it has
    crossed the box (``vx² < 2 a (1 - 2r)``), the ball is confined to one side,
    bouncing, being blown back, and bouncing again.

    "Within one radius" means the ball's surface is closer to the wall than its
    own width, i.e. ``x < 2r`` or ``x > 1 - 2r``. A healthy episode spends
    roughly ``2r / (1 - 2r)`` of its time there by geometry alone -- about 19%
    at r = 0.08 for a uniform x -- so the useful reading is the EXCESS over a
    matched axis-y cell, which is why the sweep prints the axis-y row too.
    """
    x = np.asarray(states)[..., 0]
    r = cfg.ball_radius
    return float(((x < 2.0 * r) | (x > 1.0 - 2.0 * r)).mean())


def _gravity_report(
    states, events, cfg: BoxConfig, state_names, speed_clips: int = 0
) -> dict:
    """v4 per-split summary. Printed AND written into meta.json.

    What each number is for:

    * **speed min/mean/max** -- the invariant v1-v3 could assert is gone, so its
      replacement is a distribution. A max near ``max_speed`` would mean the
      english is running away; a min near zero would mean balls are hanging at
      the top of a ballistic arc, which is where the renderer's quantisation
      would make the physics look frozen.
    * **flips per episode, and the 0/1/2/3+ histogram** -- the flip is the
      event whose consequence has to be remembered, so its RATE is the tier's
      difficulty knob. Episodes with zero flips are not waste: they are the
      control condition (the sign held from reset), and a split with too few of
      them cannot support the "sign at episode start" baseline.
    * **mean frames between flips** -- the memory horizon, i.e. the x-axis of
      the decay curve stage two measures.
    * **fraction of frames under each sign** -- a balanced split is what makes
      50% the right chance level for the sign probe. An imbalance would let a
      probe score above chance by predicting the majority sign.
    * **stuck fraction** -- see ``stuck_fraction``.
    * **speed clips** -- how often ``BouncingBox._clip_speed`` actually fired.
      Expected zero, observed not-zero on the tracking-heavy splits; see the
      note in ``BouncingBox._clip_speed``. It is the number that says how much
      of the speed distribution's top end is physics and how much is a knob.
    """
    s = np.asarray(states)
    ev = np.asarray(events)
    j = list(state_names).index("gravity_sign")
    sign = s[..., j]
    speed = np.linalg.norm(s[..., 2:4], axis=-1)

    flip = (ev & EVENT_FLIP).astype(bool)
    per_ep = flip.sum(1)
    # Frames between flips: gaps between consecutive flip times within an
    # episode. Episodes with fewer than two flips contribute nothing, which is
    # correct -- "the interval" is undefined there, and padding it with the
    # episode length would bias the mean by exactly the episodes that have no
    # interval to report.
    gaps = []
    for e in range(flip.shape[0]):
        t = np.flatnonzero(flip[e])
        if len(t) > 1:
            gaps.extend(np.diff(t).tolist())
    gaps = np.asarray(gaps, dtype=np.float64)

    stats = {
        "gravity": float(cfg.gravity),
        "gravity_axis": cfg.gravity_axis,
        "launch_min_angle_deg": float(cfg.launch_min_angle_deg),
        "speed_min": float(speed.min()),
        "speed_mean": float(speed.mean()),
        "speed_max": float(speed.max()),
        "speed_p01": float(np.percentile(speed, 1)),
        "speed_p99": float(np.percentile(speed, 99)),
        "flips_per_episode": float(per_ep.mean()),
        "frac_episodes_0_flips": float((per_ep == 0).mean()),
        "frac_episodes_1_flip": float((per_ep == 1).mean()),
        "frac_episodes_2_flips": float((per_ep == 2).mean()),
        "frac_episodes_3plus_flips": float((per_ep >= 3).mean()),
        "max_flips_in_an_episode": int(per_ep.max()),
        "mean_frames_between_flips": float(gaps.mean()) if len(gaps) else float("nan"),
        "median_frames_between_flips": (
            float(np.median(gaps)) if len(gaps) else float("nan")
        ),
        "n_flip_intervals": int(len(gaps)),
        "frac_frames_sign_down": float((sign < 0).mean()),
        "frac_frames_sign_up": float((sign > 0).mean()),
        "frac_episodes_start_down": float((sign[:, 0] < 0).mean()),
        "frac_frames_stuck_100": stuck_fraction(s, cfg, window=100),
        "frac_frames_wall_pinned": wall_pinned_fraction(s, cfg),
        "speed_clips": int(speed_clips),
        "speed_clips_per_episode": float(speed_clips / s.shape[0]),
    }
    print(
        f"  gravity: g={cfg.gravity:g} axis={cfg.gravity_axis}  "
        f"launch>={cfg.launch_min_angle_deg:g}deg  "
        f"speed {stats['speed_min']:.4f}/{stats['speed_mean']:.4f}/"
        f"{stats['speed_max']:.4f} (min/mean/max)  "
        f"speed clips {speed_clips} ({stats['speed_clips_per_episode']:.2f}/ep)"
    )
    print(
        f"    flips: {stats['flips_per_episode']:.2f}/episode  "
        f"0/1/2/3+ = {stats['frac_episodes_0_flips']:.0%}/"
        f"{stats['frac_episodes_1_flip']:.0%}/"
        f"{stats['frac_episodes_2_flips']:.0%}/"
        f"{stats['frac_episodes_3plus_flips']:.0%}  "
        f"mean {stats['mean_frames_between_flips']:.1f} frames apart "
        f"(median {stats['median_frames_between_flips']:.0f}, "
        f"n={stats['n_flip_intervals']})"
    )
    print(
        f"    sign: down on {stats['frac_frames_sign_down']:.1%} of frames / "
        f"up on {stats['frac_frames_sign_up']:.1%}  "
        f"stuck (no end reached in 100 frames): "
        f"{stats['frac_frames_stuck_100']:.2%}  "
        f"wall-pinned: {stats['frac_frames_wall_pinned']:.1%}"
    )
    if stats["frac_frames_stuck_100"] > 0.02:
        print("  warning: balls are stalling mid-box -- raise --launch-min-angle "
              "or lower --gravity")
    # Against the GEOMETRIC baseline, not against a flat 20%. A ball uniform in
    # x is inside the band 2r/(1-2r) of the time -- 19% at r = 0.08 -- so a flat
    # threshold near 20% fires on gravity-free v1 (measured: 19.1%) and says
    # nothing. Only the excess is evidence of a wind holding the ball against a
    # wall. See runs/v4_design/sweep_v41.md, "the two hazard columns".
    pin_floor = 2.0 * cfg.ball_radius / (1.0 - 2.0 * cfg.ball_radius)
    if stats["frac_frames_wall_pinned"] > pin_floor + 0.05:
        print(f"  warning: the ball is spending its life against a side wall "
              f"({stats['frac_frames_wall_pinned']:.1%} of frames vs a "
              f"geometric baseline of {pin_floor:.1%}) -- lower --gravity: the "
              f"wind is reversing vx before the ball can cross the box")
    if abs(stats["frac_frames_sign_down"] - 0.5) > 0.15:
        print("  warning: the two gravity signs are unbalanced -- chance level "
              "for the sign probe is no longer 50%")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", choices=["sticky", "uniform", "mix"], default="sticky")
    p.add_argument("--p-track", type=float, default=0.5,
                   help="for --policy mix: probability that a hold segment "
                        "tracks the ball instead of holding a random action")
    p.add_argument("--mean-hold", type=float, default=8.0)
    p.add_argument("--mass-from-color", action="store_true",
                   help="v2: sample a mass per episode and paint the ball a "
                        "colour that encodes it; speed = ball_speed / mass")
    p.add_argument("--mass-holdout", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="v2: never sample a mass inside [LO, HI] (training "
                        "sets, so the band stays unseen)")
    p.add_argument("--mass-only", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="v2: sample ONLY inside [LO, HI] -- the interpolation "
                        "test set matching --mass-holdout")
    p.add_argument("--occluder", action="store_true",
                   help="v3: draw an opaque band across the frame, hiding the "
                        "ball for part of every vertical traverse. Physics "
                        "unchanged; states gain a ball_visible column and "
                        "events an EVENT_HIDDEN bit.")
    p.add_argument("--occluder-y", type=float, nargs=2, default=(0.28, 0.58),
                   metavar=("LO", "HI"),
                   help="v3: bottom and top edge of the band in world "
                        "coordinates (y up). A taller band means longer "
                        "occlusions -- that is the memory-horizon knob.")
    p.add_argument("--paddle-w", type=float, default=0.26,
                   help="paddle width in world units. 0.26 (default) is the "
                        "v1-v3 paddle; v3.1 uses 0.16, which drops the "
                        "stand-still catch rate from ~0.50 to ~0.38 and so "
                        "lowers the memoryless ceiling the agent has to beat.")
    p.add_argument("--gravity", type=float, default=0.0,
                   help="v4: constant vertical acceleration MAGNITUDE in world "
                        "units per frame^2 (0 = v1). The direction is a hidden "
                        "per-episode latent that flips on every paddle "
                        "contact. 0.0001 is the v4 default.")
    p.add_argument("--gravity-axis", choices=["x", "y"], default="y",
                   help="v4.1: which axis the flipping acceleration acts on. "
                        "'y' (default) is the original vertical gravity; 'x' "
                        "is a horizontal SIDE-WIND, which leaves vertical "
                        "motion exactly v1's and moves the landing x by ~10x "
                        "more per sign. See runs/v4_design/sweep_v41.md.")
    p.add_argument("--gravity-sign-init", choices=["random", "down", "up"],
                   default="random",
                   help="v4: the sign at reset. 'random' (default) is the real "
                        "environment; the fixed options make a constant-gravity "
                        "control world for debugging.")
    p.add_argument("--launch-min-angle", type=float,
                   default=LAUNCH_MIN_ANGLE_V1_DEG,
                   help="minimum launch angle from horizontal, in degrees. "
                        "14.5 (default) is the v1-v3 rule exactly; v4 uses 40, "
                        "the shallowest launch whose vertical kinetic energy "
                        "still carries the ball across the box against "
                        "gravity pulling the other way.")
    p.add_argument("--ball-radius", type=float, default=0.055,
                   help="0.08 makes the ball ~2x more of the loss; "
                        "recommended for your first VAE")
    a = p.parse_args()
    collect(
        a.out,
        episodes=a.episodes,
        steps=a.steps,
        res=a.res,
        seed=a.seed,
        policy=a.policy,
        mean_hold=a.mean_hold,
        ball_radius=a.ball_radius,
        p_track=a.p_track,
        mass_from_color=a.mass_from_color,
        mass_holdout=tuple(a.mass_holdout) if a.mass_holdout else None,
        mass_only=tuple(a.mass_only) if a.mass_only else None,
        occluder=a.occluder,
        occluder_y=tuple(a.occluder_y),
        paddle_w=a.paddle_w,
        gravity=a.gravity,
        gravity_sign_init=a.gravity_sign_init,
        launch_min_angle=a.launch_min_angle,
        gravity_axis=a.gravity_axis,
    )


if __name__ == "__main__":
    main()
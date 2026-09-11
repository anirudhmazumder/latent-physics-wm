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
    states.npy   float32  (E, T + 1, 6)
    events.npy   uint8    (E, T)
    meta.json

frames[e, t] and states[e, t] are the observation *before* actions[e, t] is
applied; frames[e, t + 1] is the result. So a transition is
(frames[e, t], actions[e, t]) -> frames[e, t + 1], and there are T transitions
from T + 1 frames.

Usage:
    python -m worldsim.collect --out data/v1/train --episodes 100 --steps 200
    python -m worldsim.collect --out data/v1/val   --episodes 10  --steps 200 --seed 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .bouncing_box import STATE_NAMES, BouncingBox, BoxConfig
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
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = BoxConfig(res=res, ball_radius=ball_radius)
    env = BouncingBox(cfg)
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
        shape=(episodes, steps + 1, len(STATE_NAMES)),
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

    _sanity_check(states, events, cfg)

    for arr in (frames, actions, states, events):
        arr.flush()

    meta = {
        "version": "v1_bouncing_box",
        "episodes": episodes,
        "steps": steps,
        "res": res,
        "seed": seed,
        "policy": policy,
        "mean_hold": mean_hold if policy in ("sticky", "mix") else None,
        "p_track": p_track if policy == "mix" else None,
        "state_names": list(STATE_NAMES),
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


def _sanity_check(states, events, cfg: BoxConfig) -> None:
    """Cheap invariants. Catching these here beats debugging a VAE later."""
    s = np.asarray(states)
    assert np.isfinite(s).all(), "non-finite values in states"

    r = cfg.ball_radius
    x, y = s[..., 0], s[..., 1]
    slack = 1e-4
    assert x.min() >= r - slack and x.max() <= 1 - r + slack, "ball x out of bounds"
    assert y.min() >= r - slack and y.max() <= 1 - r + slack, "ball y out of bounds"

    speed = np.linalg.norm(s[..., 2:4], axis=-1)
    assert np.allclose(speed, cfg.ball_speed, atol=1e-6), "ball speed not conserved"

    hit_rate = float((np.asarray(events) & 4).astype(bool).mean())
    print(f"  sanity: bounds ok, speed conserved, paddle-contact rate {hit_rate:.3%}")
    if hit_rate < 1e-4:
        print("  warning: the ball almost never touches the paddle -- actions will "
              "look nearly irrelevant to the dynamics model")


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
    )


if __name__ == "__main__":
    main()
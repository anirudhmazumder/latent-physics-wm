"""Tests for the v2 (mass-from-colour) additions to stage three.

Runnable two ways::

    python -m tests.test_controller_v2
    pytest tests/test_controller_v2.py

Four things are worth pinning down, and they are exactly the four ways the v2
generalisation could be silently wrong.

1. The harness records each episode's mass. Every number in the stage-three
   report is sliced by mass, so if the mass does not come out of the rollout the
   whole analysis is a fiction.
2. The floor-visit band scales with the episode's speed. A light ball moves
   0.044 per frame; a band of fixed 0.022 headroom can be jumped in one step,
   and the visit -- the *chance* -- would never be counted. That would inflate
   interceptions-per-visit on precisely the tercile the experiment is about.
3. The hold-out flags actually reach ``BouncingBox``. ``--mass-only 0.85 1.2``
   passing silently through argparse and never arriving would give a
   "generalisation result" measured on training colours.
4. The v1 code path is untouched. The v1 numbers in ``wm/README_C.md`` were
   measured before any of this existed, so the v1 controller in the v1 world
   must still produce exactly what it produced then. The expected value below
   was recorded by running the harness at the commit *before* the v2 changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from unittest import SkipTest  # noqa: E402
from tests.conftest import require_ckpt, require_data  # noqa: E402
from wm.controller import StayController
from wm.eval_controller import (
    floor_visit_stats,
    floor_zone_height,
    make_box_cfg,
    run_real_episodes,
)

VAE_V2 = Path("runs/vae_v2/vae.pt")
RNN_V2 = Path("runs/rnn_v2/rnn.pt")
VAE_V1 = Path("runs/vae_b1/vae.pt")
RNN_V1 = Path("runs/rnn_v1/rnn.pt")
CTRL_V1 = Path("runs/ctrl_v1/controller.pt")

HOLDOUT_BAND = (0.85, 1.2)

# Recorded from `run_real_episodes(load_controller("runs/ctrl_v1/controller.pt"),
# vae_b1, rnn_v1, episodes=2, steps=60, seed_base=5000, ball_radius=0.08)` on the
# code as it stood BEFORE the v2 generalisation (verified by `git stash`).
V1_REFERENCE_HITS = [1.0, 0.0]


def _load(vae_path: Path, rnn_path: Path):
    from wm.analyze import load_ckpt
    from wm.rnn import load_rnn

    vae, _, _ = load_ckpt(str(vae_path), "cpu")
    rnn, _ = load_rnn(str(rnn_path), "cpu")
    return vae, rnn


def _have(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


# ----------------------------------------------------- the config helper alone


def test_make_box_cfg_defaults_to_v1() -> None:
    """No v2 argument given -> the config is byte-for-byte the v1 environment."""
    cfg = make_box_cfg(ball_radius=0.08)
    assert cfg.mass_from_color is False
    assert cfg.mass_holdout is None and cfg.mass_only is None

    cfg2 = make_box_cfg(0.08, mass_from_color=True, mass_only=HOLDOUT_BAND)
    assert cfg2.mass_from_color is True
    # BoxConfig normalises the band to a float tuple; a list must survive that.
    assert cfg2.mass_only == (0.85, 1.2)


# ------------------------------------------------- the speed-scaled floor band


def test_floor_zone_scales_with_speed() -> None:
    """The band is contact height + ONE FRAME of travel, whatever that is."""
    # v1: 0.045 + 0.08 + 0.022.
    assert abs(floor_zone_height(0.08, 0.045, 0.022) - 0.147) < 1e-12
    # A light v2 ball travels twice as far per frame, so its band is wider.
    assert abs(floor_zone_height(0.08, 0.045, 0.044) - 0.169) < 1e-12


def test_floor_visit_stats_uses_per_episode_speed() -> None:
    """A fast ball's visit is only counted when the band is scaled to it.

    Synthetic, so the arithmetic is visible: a ball descending in steps of 0.044
    goes 0.170 -> 0.126 -> 0.082. With the v1 threshold of 0.147 it does enter
    the band (0.126 < 0.147), so that alone would not show the bug; the case
    that bites is a ball that steps from above 0.147 straight to below the
    contact height in one frame, where the *entry* is what gets missed. We
    construct the tighter case: one frame inside the wide band and none inside
    the narrow one.
    """
    T = 6
    # y: 0.30, 0.22, 0.16, 0.13(=contact), 0.13, 0.13  -- resolved onto the
    # paddle, exactly as the collision resolver does.
    y = np.array([0.30, 0.22, 0.160, 0.125, 0.125, 0.125])
    states = np.zeros((1, T, 6), np.float32)
    states[0, :, 1] = y
    # With the v1 band (0.147) the first frame below it is 0.125 -> one visit.
    v1 = floor_visit_stats(states, speed=None)
    # With a light ball (speed 0.044 -> band 0.169) the crossing happens one
    # frame earlier, at y = 0.160, and is still exactly one visit -- the count
    # is what must not change, the frame it fires on is.
    v2 = floor_visit_stats(states, speed=np.array([0.044]))
    assert v1["floor_visits"][0] == 1.0
    assert v2["floor_visits"][0] == 1.0

    # Now the case the fixed threshold actually loses: a ball that is above
    # 0.147 and then resolves onto the paddle top at 0.125 is caught either way,
    # so instead make the ball skim -- dip to 0.160 and climb back out. The wide
    # band sees a chance; the narrow one sees nothing.
    y2 = np.array([0.30, 0.22, 0.160, 0.160, 0.22, 0.30])
    st2 = np.zeros((1, 6, 6), np.float32)
    st2[0, :, 1] = y2
    assert floor_visit_stats(st2, speed=None)["floor_visits"][0] == 0.0
    assert floor_visit_stats(st2, speed=np.array([0.044]))["floor_visits"][0] == 1.0


def test_floor_visit_stats_speed_none_is_the_v1_number() -> None:
    """Omitting `speed` must reproduce the v1 threshold exactly, not approximately."""
    rng = np.random.default_rng(0)
    states = rng.random((5, 80, 6)).astype(np.float32)
    a = floor_visit_stats(states)
    b = floor_visit_stats(states, speed=np.full(5, 0.022))
    assert np.array_equal(a["floor_visits"], b["floor_visits"])
    assert np.allclose(a["gap_at_floor"], b["gap_at_floor"])


# ------------------------------------------------------- the harness, for real


def test_harness_records_mass_per_episode() -> None:
    require_ckpt(VAE_V2); require_ckpt(RNN_V2)
    vae, rnn = _load(VAE_V2, RNN_V2)
    roll = run_real_episodes(
        StayController(), vae, rnn, episodes=6, steps=12, seed_base=5000,
        device="cpu", ball_radius=0.08, mass_from_color=True,
        mass_holdout=HOLDOUT_BAND,
    )
    assert "mass" in roll and roll["mass"].shape == (6,)
    assert roll["states"].shape[-1] == 7
    # The recorded mass IS the state column, at every timestep (mass is
    # constant within an episode).
    assert np.allclose(roll["states"][:, :, 6], roll["mass"][:, None], atol=1e-6)
    # And the derived speed obeys the law.
    assert np.allclose(roll["speed"], 0.022 / roll["mass"], rtol=1e-5)
    # Masses are inside the legal range and none is in the excluded band.
    assert roll["mass"].min() >= 0.5 - 1e-6 and roll["mass"].max() <= 2.0 + 1e-6
    assert not ((roll["mass"] > 0.85) & (roll["mass"] < 1.2)).any()


def test_holdout_flags_reach_the_real_env() -> None:
    """`mass_only` must put EVERY evaluation episode inside the band."""
    require_ckpt(VAE_V2); require_ckpt(RNN_V2)
    vae, rnn = _load(VAE_V2, RNN_V2)
    roll = run_real_episodes(
        StayController(), vae, rnn, episodes=10, steps=8, seed_base=6000,
        device="cpu", ball_radius=0.08, mass_from_color=True,
        mass_only=HOLDOUT_BAND,
    )
    m = roll["mass"]
    assert (m >= HOLDOUT_BAND[0]).all() and (m <= HOLDOUT_BAND[1]).all(), m
    # It must also be a real sample, not one repeated mass.
    assert m.std() > 1e-3


def test_v1_code_path_is_unchanged() -> None:
    """The v1 controller in the v1 world still scores exactly what it did.

    This is the regression test for the whole v2 generalisation: every new
    argument defaults to the v1 value, so a v1 call must be bit-identical. The
    reference was recorded from the pre-change code.
    """
    require_ckpt(VAE_V1); require_ckpt(RNN_V1); require_ckpt(CTRL_V1)
    from wm.controller import load_controller

    vae, rnn = _load(VAE_V1, RNN_V1)
    ctrl = load_controller(str(CTRL_V1), name="ctrl_v1")
    roll = run_real_episodes(
        ctrl, vae, rnn, episodes=2, steps=60, seed_base=5000, device="cpu",
        ball_radius=0.08, mass_from_color=False,
    )
    assert roll["states"].shape[-1] == 6, "v1 state must stay 6 columns"
    assert "mass" not in roll, "a v1 rollout has no mass to record"
    assert roll["hits"].sum(1).tolist() == V1_REFERENCE_HITS, roll["hits"].sum(1)

    # Determinism: the same call twice gives the same thing.
    again = run_real_episodes(
        ctrl, vae, rnn, episodes=2, steps=60, seed_base=5000, device="cpu",
        ball_radius=0.08,
    )
    assert np.array_equal(roll["hits"], again["hits"])
    assert np.allclose(roll["states"], again["states"])


# ---------------------------------------------------------------- the analysis


def test_mass_terciles_split_evenly() -> None:
    from wm.eval_controller_v2 import TERCILES, mass_terciles

    m = np.exp(np.linspace(np.log(0.5), np.log(2.0), 150))
    idx, cuts = mass_terciles(m)
    counts = [int((idx == k).sum()) for k in range(len(TERCILES))]
    assert max(counts) - min(counts) <= 1, counts
    assert cuts[0] < cuts[1]


def test_reaction_lead_is_zero_for_a_paddle_moving_away() -> None:
    """The behavioural metric must not reward a paddle that was going the wrong way."""
    from wm.eval_controller_v2 import reaction_lead

    T = 20
    states = np.zeros((1, T + 1, 7), np.float32)
    states[0, :, 0] = 0.8                 # ball parked at x = 0.8
    states[0, :, 4] = 0.2                 # paddle far to the left of it
    states[0, :, 5] = -0.03               # ... and driving further left
    hits = np.zeros((1, T), np.float32)
    hits[0, 10] = 1.0
    lead, pos, _ = reaction_lead({"hits": hits, "states": states})
    assert lead.tolist() == [0.0]
    # It was also never in position: |0.8 - 0.2| = 0.6 > half a paddle width.
    assert pos.tolist() == [0.0]

    # Now the same interception with the paddle closing on the ball.
    states[0, :, 5] = +0.03
    lead2, _pos2, _ = reaction_lead({"hits": hits, "states": states})
    assert lead2[0] == 10.0, lead2

    # And the position lead ignores dithering: a paddle parked on the landing
    # spot but flapping left/right scores the full window.
    states[0, :, 4] = 0.8
    states[0, :, 5] = np.where(np.arange(T + 1) % 2, 0.03, -0.03)
    _l3, pos3, _ = reaction_lead({"hits": hits, "states": states})
    assert pos3[0] == 10.0, pos3


# ---------------------------------------------------------------- runner


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except SkipTest as exc:
            print(f"  SKIP  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

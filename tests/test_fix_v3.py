"""Tests for the v3 permanence fixes: emergence weighting and the position head.

Runnable two ways (pytest is not guaranteed to be installed here)::

    python -m tests.test_fix_v3
    pytest tests/test_fix_v3.py

Four things are checked, and they are the four places one of these fixes could
be silently wrong and still produce a plausible-looking training curve:

    * the **emergence mask**, against hand-built visibility sequences. The mask
      is what fix (b) up-weights; if it were shifted by one it would up-weight
      the last hidden frame (whose target carries no information about the
      ball's x at all) instead of the first visible one, and the run would look
      like a failed hypothesis rather than a bug. The alignment convention --
      entry 0 of ``visible`` is the frame BEFORE the first target -- is pinned
      here because two different arrays feed it in ``train_rnn``;
    * the **position head is invisible to everything downstream**. Every eval
      reads position out of ``h`` with its own external probe, which is a fair
      measurement; reading the head's own output would not be. So the head must
      appear under exactly one new key and change nothing else;
    * the **long rollout** (``--rollout-loss-steps 24``) produces finite
      gradients. A 24-step chain of reparameterised MDN samples is the longest
      graph in this project and the obvious place for a NaN;
    * **every old checkpoint still loads.** ``RNNConfig`` gained a field, and a
      field with a default is only free if ``load_rnn`` really ignores what it
      does not know.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from wm.conservation import rollout_losses
from wm.rnn import MDNRNN, RNNConfig, load_rnn, rnn_loss, save_rnn
from wm.train_rnn import emergence_weight_mask

REPO = Path(__file__).resolve().parents[1]


def _vis(seq) -> torch.Tensor:
    return torch.tensor([seq], dtype=torch.float32)


# ------------------------------------------------------- the emergence mask


def test_emergence_mask_matches_a_hand_built_sequence() -> None:
    """visible[0] is the frame before target 0, so the mask is one shorter."""
    #                   t0  |------------- the 9 target frames -------------|
    v = _vis([1, 1, 0, 0, 0, 1, 1, 1, 1, 1])
    #                         ^ frame 5 is the first visible one after a run

    # Strict reading: exactly the frame whose predecessor was hidden.
    assert emergence_weight_mask(v, 1).tolist() == [
        [False, False, False, False, True, False, False, False, False]]
    # v3's reading: that frame and the two after it.
    assert emergence_weight_mask(v, 3).tolist() == [
        [False, False, False, False, True, True, True, False, False]]


def test_emergence_mask_ignores_a_run_already_in_progress() -> None:
    """A window that opens mid-flight cannot know when that flight began."""
    v = _vis([1, 1, 1, 1, 0, 0, 1, 1])
    m = emergence_weight_mask(v, 3).tolist()[0]
    # Targets 0-2 are visible but their run started before the window: unmarked.
    assert m[:3] == [False, False, False]
    # The run that starts at index 6 is fully observed, so it IS marked.
    assert m[5:] == [True, True]


def test_emergence_mask_marks_nothing_without_an_occlusion() -> None:
    for seq in ([1] * 8, [0] * 8):
        assert not emergence_weight_mask(_vis(seq), 3).any()


def test_emergence_mask_handles_several_runs_and_a_batch() -> None:
    v = torch.tensor(
        [[0, 1, 1, 1, 1, 0, 0, 1, 1, 0],      # two runs, both fully observed
         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],     # alternating: every visible frame
        dtype=torch.float32)
    m = emergence_weight_mask(v, 3)
    assert m.shape == (2, 9)
    assert m[0].tolist() == [True, True, True, False, False,
                             False, True, True, False]
    assert m[1].tolist() == [False, True, False, True, False,
                             True, False, True, False]


def test_emergence_weight_changes_only_the_weighted_frames() -> None:
    """The weighted NLL is a re-weighting, not a different likelihood.

    Two checks that together pin the semantics: all-ones weights reproduce the
    unweighted loss exactly, and the weighted loss is the mask-weighted average
    of the same per-step numbers.
    """
    torch.manual_seed(0)
    model = MDNRNN(RNNConfig(z_dim=4, hidden=16, n_gauss=2))
    z = torch.randn(2, 6, 4)
    a = torch.eye(3)[torch.randint(0, 3, (2, 6))]
    zn = torch.randn(2, 6, 4)
    parts, _ = model(z, a)

    plain = model.mdn_nll(parts, zn)
    per_step = model.mdn_nll_per_step(parts, zn)
    assert torch.allclose(plain, per_step.mean(), atol=1e-6)
    assert torch.allclose(model.mdn_nll(parts, zn, torch.ones(2, 6)), plain,
                          atol=1e-6)

    w = torch.ones(2, 6)
    w[:, 3] = 5.0
    got = model.mdn_nll(parts, zn, w)
    want = (per_step * w).mean() / w.mean()
    assert torch.allclose(got, want, atol=1e-6)


# ------------------------------------------------------- the position head


def _forward_keys(pos_head: bool) -> set:
    torch.manual_seed(0)
    model = MDNRNN(RNNConfig(z_dim=4, hidden=16, n_gauss=2, pos_head=pos_head))
    parts, _ = model(torch.randn(2, 5, 4), torch.eye(3)[torch.zeros(2, 5).long()])
    return set(parts)


def test_pos_head_adds_exactly_one_key_and_nothing_else() -> None:
    base, with_head = _forward_keys(False), _forward_keys(True)
    assert with_head - base == {"ball_pos"}
    assert base - with_head == set()


def test_pos_head_output_is_not_used_by_the_shared_loss() -> None:
    """``rnn_loss`` -- what every eval and the val metric call -- must not see it.

    The head is trained by an explicit extra term in ``train_rnn``; if it ever
    leaked into ``rnn_loss`` the reported val NLL would stop being comparable
    with every other run in the project.
    """
    torch.manual_seed(0)
    cfg = RNNConfig(z_dim=4, hidden=16, n_gauss=2, pos_head=True)
    model = MDNRNN(cfg)
    z = torch.randn(2, 5, 4)
    a = torch.eye(3)[torch.zeros(2, 5).long()]
    batch = {"z": z, "a": a, "z_next": torch.randn(2, 5, 4),
             "hit": torch.zeros(2, 5), "reward": torch.zeros(2, 5)}
    parts, _ = model(z, a)
    loss, _ = rnn_loss(model, parts, batch)
    loss.backward()
    # No gradient reached the head: it is not on this loss's graph at all.
    assert model.pos_head.weight.grad is None


def test_pos_head_shape_and_checkpoint_round_trip() -> None:
    cfg = RNNConfig(z_dim=4, hidden=16, n_gauss=2, pos_head=True)
    model = MDNRNN(cfg)
    parts, _ = model(torch.randn(2, 5, 4), torch.eye(3)[torch.zeros(2, 5).long()])
    assert parts["ball_pos"].shape == (2, 5, 2)
    with tempfile.TemporaryDirectory() as d:
        p = save_rnn(Path(d) / "rnn.pt", model)
        back, back_cfg = load_rnn(p)
    assert back_cfg.pos_head is True
    assert back.pos_head is not None


# ----------------------------------------------------------- the long rollout


def test_rollout_24_gives_finite_gradients() -> None:
    """The longest graph in the project, on a model small enough to run here."""
    torch.manual_seed(0)
    model = MDNRNN(RNNConfig(z_dim=4, hidden=16, n_gauss=2))
    L, K = 32, 24
    z = torch.randn(3, L, 4) * 0.1
    a = torch.eye(3)[torch.randint(0, 3, (3, L))]

    roll = rollout_losses(model, z, a, t0=0, k_steps=K)
    assert roll["z_dreamed"].shape == (3, K, 4)
    assert torch.isfinite(roll["nll"])
    roll["nll"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    # The recurrent weights specifically: a rollout that had silently detached
    # its own samples would still train the heads and teach the model nothing.
    assert model.lstm.weight_hh_l0.grad.abs().sum() > 0


def test_rollout_weights_are_indexed_at_the_right_step() -> None:
    """Weighting step k must reach ``weights[:, t0 + k]``, not ``[:, k]``."""
    torch.manual_seed(0)
    model = MDNRNN(RNNConfig(z_dim=4, hidden=16, n_gauss=2))
    L, K, t0 = 16, 4, 5
    z = torch.randn(2, L, 4) * 0.1
    a = torch.eye(3)[torch.randint(0, 3, (2, L))]

    g = lambda: torch.Generator().manual_seed(7)  # noqa: E731
    plain = rollout_losses(model, z, a, t0, K, generator=g())["nll"]
    ones = rollout_losses(model, z, a, t0, K, generator=g(),
                          weights=torch.ones(2, L))["nll"]
    assert torch.allclose(plain, ones, atol=1e-6)

    w = torch.ones(2, L)
    w[:, t0 + 2] = 9.0            # the second scored step
    got = rollout_losses(model, z, a, t0, K, generator=g(), weights=w)["nll"]
    per = rollout_losses(model, z, a, t0, K, generator=g())["nll_per_step"]
    ww = torch.tensor([1.0, 9.0, 1.0, 1.0])
    assert torch.allclose(got, (per * ww).sum() / ww.sum(), atol=1e-5)


def test_rollout_refuses_a_window_it_does_not_fit_in() -> None:
    model = MDNRNN(RNNConfig(z_dim=4, hidden=16, n_gauss=2))
    z, a = torch.randn(2, 16, 4), torch.eye(3)[torch.zeros(2, 16).long()]
    with pytest.raises(ValueError):
        rollout_losses(model, z, a, t0=0, k_steps=16)


# ----------------------------------------------------- old checkpoints load


OLD_CKPTS = ["runs/rnn_v1/rnn.pt", "runs/rnn_v2/rnn.pt",
             "runs/rnn_v2_cons/rnn.pt", "runs/rnn_v3/rnn.pt",
             "runs/rnn_v3_ff/rnn.pt"]


def test_old_checkpoints_still_load() -> None:
    """A new ``RNNConfig`` field must not invalidate anything already trained."""
    seen = 0
    for rel in OLD_CKPTS:
        path = REPO / rel
        if not path.exists():
            continue
        model, cfg = load_rnn(path)
        assert cfg.pos_head is False, rel
        assert model.pos_head is None, rel
        z = torch.randn(1, 4, cfg.z_dim)
        a = torch.eye(cfg.n_actions)[torch.zeros(1, 4).long()]
        parts, _ = model(z, a)
        assert "ball_pos" not in parts, rel
        assert torch.isfinite(parts["h"]).all(), rel
        seen += 1
    if seen == 0:
        pytest.skip("no pre-existing checkpoints in this working copy")


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"\n{len(fns)} tests passed")
    sys.exit(0)

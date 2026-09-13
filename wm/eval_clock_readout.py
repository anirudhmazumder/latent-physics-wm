"""Did the clock head learn the counter -- and WHICH of its two counters?

    python -m wm.eval_clock_readout \
        --models clock=runs/rnn_v31_clock/rnn.pt \
                 clock_priv=runs/rnn_v31_clock_priv/rnn.pt \
                 clock_emerge=runs/rnn_v31_clock_emerge/rnn.pt \
        --out runs/rnn_v31_clock_readout

Why this file exists
--------------------
Two other measurements disagree about whether ``--clock-head`` worked, and
neither of them answers the question that matters.

* The **training log** prints one combined clock MSE (v3.1: ~9.3 frames rmse on
  val) over BOTH targets and over EVERY frame, visible ones included. On a
  visible frame both counters are 0 and predicting 0 is free, so that number is
  dominated by the easy two-thirds of the data and cannot distinguish "learned
  one target" from "learned both".
* **``eval_permanence_v3`` part (f)** refits a FROZEN linear probe on ``h``,
  on fully hidden frames only, and reports a negative R^2 for ``frames_hidden``
  for every model -- including the privileged one. Read alone that says the
  counter is not in ``h`` at all, which turns out to be the wrong conclusion.

So this script asks the model's OWN trained head, on the frames the question is
about. It is deliberately the least clever measurement available: run the model
teacher-forced over real validation latents, take the head's two outputs, and
score each one separately against the true counter on fully hidden frames.
Nothing is refit, so nothing can be blamed on a probe.

The v3.1 answer, and the reason follow-up A failed
--------------------------------------------------
``frames-since`` comes back at R^2 ~0.7 (about 6 frames rmse); ``frames-until``
comes back at R^2 ~0 (about 13 frames rmse), i.e. no better than predicting the
mean. Every clock model learned "how long has the ball been gone" and none of
them learned "when does it come back".

That asymmetry is not surprising once stated. ``h`` already carries "is the ball
visible" (R^2 0.99, README_V31 §3), so ``since`` is a running sum and an LSTM
integrates for free. ``until`` requires predicting the future of an object the
model cannot see, which is the capability the head was supposed to INDUCE --
and a regression head does not supply it, it only measures that it is missing.
See README_CLOCK31.md §3.

A note on the two thresholds
----------------------------
This script's ``hidden`` mask is ``ball_visible < 0.01`` (fully hidden), to
match part (f)'s bin, while the counters themselves reset at
``VISIBLE_THRESHOLD`` = 0.5 (at least half visible). The two are deliberately
different -- the mask selects the frames we are asking about, the threshold
defines the quantity -- and conflating them is the easy mistake here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .clock import CLOCK_CLIP, clock_targets, unscale
from .rnn import load_rnn
from .seq_data import episode_arrays

# Column 6 of ``states.npy`` is ``ball_visible``, the fraction of the ball's
# area clear of the band. "Fully hidden" is the same cut part (f) uses.
VISIBLE_COL = 6
HIDDEN_CUT = 0.01
TARGET_NAMES = ("frames_since", "frames_until")


def load_val(roots: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate the validation roots into (mu, actions, ball_visible)."""
    parts = [episode_arrays(r) for r in roots]
    mu = np.concatenate([p["mu"] for p in parts], 0)          # (E, T+1, z)
    actions = np.concatenate([p["actions"] for p in parts], 0)  # (E, T)
    vis = np.concatenate([p["state"] for p in parts], 0)[..., VISIBLE_COL]
    return mu, actions, vis


def head_predictions(ckpt: str, mu: np.ndarray, actions: np.ndarray,
                     device: str = "cpu") -> np.ndarray:
    """(E, T, 2) counters in FRAMES, as the model's own head reports them.

    The model consumes ``mu[:, :-1]`` and ``actions``; every head on ``h`` in
    this project describes the frame the MDN is predicting, so ``h_t`` is about
    frame ``t + 1`` and the caller must align the targets the same way. That
    alignment is the one thing a mistake here would hide, and
    ``tests/test_clock31.py::test_frame_targets_align_with_state`` pins the
    training side of it.
    """
    model, cfg = load_rnn(ckpt, device=device)
    if getattr(cfg, "clock_head", False) is not True:
        raise SystemExit(f"{ckpt} was not trained with --clock-head")
    model.eval()
    a1h = torch.nn.functional.one_hot(
        torch.as_tensor(actions, dtype=torch.long), cfg.n_actions).float()
    with torch.no_grad():
        parts, _ = model(torch.as_tensor(mu[:, :-1], dtype=torch.float32),
                         a1h.to(device))
    return unscale(parts["clock"].cpu().numpy(), clip=CLOCK_CLIP)


def score(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> Dict:
    """Per-target R^2 and rmse (in frames) on the masked frames.

    R^2 is against the variance of the target WITHIN the mask, so 0 means "no
    better than predicting the mean of the hidden frames" and a negative value
    means worse than that. With ``until`` nearly flat inside long occlusions
    that denominator is small, which is exactly why the rmse column is printed
    next to it rather than instead of it.
    """
    out: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(TARGET_NAMES):
        p, y = pred[..., j][mask], true[..., j][mask]
        ss_res = float(((p - y) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out[name] = {
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "rmse_frames": float(np.sqrt(((p - y) ** 2).mean())),
            "true_mean_frames": float(y.mean()),
            "pred_mean_frames": float(p.mean()),
        }
    out["n_frames"] = int(mask.sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", metavar="NAME=PATH", default=[
        "clock=runs/rnn_v31_clock/rnn.pt",
        "clock_priv=runs/rnn_v31_clock_priv/rnn.pt",
        "clock_emerge=runs/rnn_v31_clock_emerge/rnn.pt",
    ])
    p.add_argument("--val", nargs="+",
                   default=["data/v31/val", "data/v31/val_mix"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="runs/rnn_v31_clock_readout")
    a = p.parse_args()

    mu, actions, vis = load_val(a.val)
    # Targets are built from the SIMULATOR's column for every model, fair and
    # privileged alike. The fair run was TRAINED against a probe's reading, but
    # the question being scored is "does it know the truth", so the truth is
    # what it is scored against -- otherwise the fair run would be graded on a
    # curve and the two rows would not be comparable.
    true_all = unscale(clock_targets(vis, clip=CLOCK_CLIP), clip=CLOCK_CLIP)
    T = mu.shape[1] - 1
    true = true_all[:, 1:T + 1]
    hidden = vis[:, 1:T + 1] < HIDDEN_CUT
    print(f"val: {mu.shape[0]} episodes x {T} steps, "
          f"{int(hidden.sum())} fully hidden frames "
          f"({hidden.mean():.1%})")

    rows: Dict[str, Dict] = {}
    for spec in a.models:
        name, _, path = spec.partition("=")
        rows[name] = score(head_predictions(path, mu, actions, a.device),
                           true, hidden)
        rows[name]["ckpt"] = path

    lines: List[str] = [
        "# The clock head, read directly off `h` on fully hidden frames", "",
        f"`{' '.join(a.val)}`, {int(hidden.sum())} fully hidden frames "
        f"(`ball_visible` < {HIDDEN_CUT}). Nothing is refit: these are the "
        "models' own trained heads, teacher-forced on real latents.", "",
        "| model | *frames-since* R2 | rmse (frames) | *frames-until* R2 | rmse (frames) |",
        "|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        s, u = r["frames_since"], r["frames_until"]
        lines.append(f"| `{name}` | {s['r2']:.3f} | {s['rmse_frames']:.2f} "
                     f"| {u['r2']:.3f} | {u['rmse_frames']:.2f} |")
    lines += ["",
              "An R2 of 0 is 'no better than predicting the mean of the hidden "
              "frames'. See README_CLOCK31.md section 3."]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "clock_readout.md").write_text("\n".join(lines) + "\n")
    (out / "clock_readout.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    print(f"\n-> {out / 'clock_readout.md'}")


if __name__ == "__main__":
    main()

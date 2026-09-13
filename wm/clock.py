"""The exit clock: teaching ``h`` to count frames through an occlusion.

The problem this file exists to solve, stated exactly
-----------------------------------------------------
v3.1's band hides the ball for a mean of 21 frames. One-step teacher forcing
pays the model only for whatever the *next* latent depends on, and for 20 of
those 21 frames the next latent is the same blank band whatever the ball is
doing. The only frame that pays for having counted is the exit, and it is 20
steps away -- far past anything a one-step gradient can reach. So the v3.1
baseline learned to hold the hidden ball's ``x`` (R^2 0.52 from ``h``) and lost
the clock entirely (``frames_hidden`` R^2 **-0.23**, ``ball_vy`` -0.32), and a
tau = 0 dream therefore never brings the ball back out: it knows *where* the
ball will reappear and has no idea *when*, so it never commits to the exit.
In v3, whose occlusions were 9.5 frames long, the same architecture learned the
clock (0.54) and not ``x`` -- the nearest informative frame was close enough for
the gradient to reach, and it was the exit.

The fix is to pay for counting directly, with a **self-supervised** auxiliary
target::

    (a) frames since the ball was last at least half visible
    (b) frames until it is next  at least half visible

Both clipped at ``CLOCK_CLIP`` frames and divided by it, so the head's two
outputs live in [0, 1] and an MSE on them weights a 20-frame error twenty times
a 1-frame one, which is the ordering we want.

Why this is FAIR and ``--pos-head`` is not
------------------------------------------
The visibility sequence the targets are built from does **not** come from the
simulator's ``ball_visible`` column. It comes from a **frozen probe on the
model's own input stream**: a degree-2 ridge from VAE ``mu`` to ``ball_visible``,
fitted once on training latents before a single gradient step and then never
touched (held-out R^2 ~ 0.995 on v3.1 -- stage one already established that
"is there a ball in this frame" is nearly perfectly decodable from ``z``, which
is what makes the fair version possible at all). Every bit of information in
the target is therefore information the dynamics model can read off its own
input at that frame. Nothing privileged enters.

The one thing target (b) uses that the model does not have *at that instant* is
the FUTURE: "frames until next visible" is computed by looking ahead along the
episode at training time. That is **hindsight**, and it is exactly the
arrangement the reward head has already been trained under since v1 -- a label
computed offline from the recorded episode, available at training time and
never at dream time. A hindsight target is not a privileged one: it asks the
model to predict something it will itself observe, which is the whole business
of a predictive model. The head's output is never read by anything downstream;
it exists solely to put a gradient on ``h``.

``--clock-privileged`` swaps the probe for the simulator's ``ball_visible``
column. That run is the CEILING: it measures how much of any failure is the
probe's fault rather than the objective's, and it is labelled as privileged
everywhere it appears.

Alignment
---------
The targets are emitted on the same time axis as ``state`` in
``LatentSequenceDataset`` -- the POST-transition frames t0+1 .. t0+L -- so the
clock head reads ``h_t`` and describes the frame the MDN is predicting, exactly
like ``--pos-head`` and the reward head. One convention for every head on
``h``; no new alignment to get wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .conservation import Poly2Probe
from .probes import make_split
from .seq_data import episode_arrays

# How far the counters are allowed to count before they saturate, in frames.
# 40 because v3.1's default band hides the ball for a mean of 21 frames and a
# median of 19, its long band for 30, and the tail runs to 104. A clip of 40
# covers ~95 % of hidden runs exactly and turns the remaining tail into "a long
# time", which is all a controller can act on anyway. It is also the value of
# --max-age the v3.1 permanence decay curve already uses, so the clock the head
# is asked to keep and the clock the evaluation probes for are the same clock.
CLOCK_CLIP = 40

# "At least half visible". ``ball_visible`` is the FRACTION of the ball's area
# clear of the band, so 0.5 is "the ball's centre is outside the band" -- the
# same threshold ``wm.eval_dream_alive.detect_ball`` uses on pixels. Using a
# fraction threshold rather than "fully visible" matters: the ball spends ~33 %
# of frames partially visible, and a counter that only reset on a fully clear
# ball would spend a third of its life mis-counting.
VISIBLE_THRESHOLD = 0.5


# ------------------------------------------------------------------ targets


def clock_targets(visible: np.ndarray, clip: int = CLOCK_CLIP) -> np.ndarray:
    """(..., T) visibility -> (..., T, 2) scaled [since, until] counters.

    ``visible`` is anything that compares sensibly against
    ``VISIBLE_THRESHOLD``: a float fraction, a bool, or a probe's real-valued
    reading. Leading/trailing axes are preserved, so an (E, T) episode array in
    gives an (E, T, 2) array out.

    Semantics, with ``v`` the thresholded boolean sequence:

        since[t] = t - max{s <= t : v[s]}     (0 where v[t]; clip if never)
        until[t] = min{s >= t : v[s]} - t     (0 where v[t]; clip if never)

    Both are clipped INTO [0, clip] and then divided by ``clip``, so the
    returned values are in [0, 1]. The "never" cases -- an episode that starts
    hidden and has no earlier sighting, or ends hidden with no later one --
    saturate at 1.0 rather than being masked out. Saturating is the right
    answer and not a fudge: from inside the window those frames genuinely are
    "a long time since / until", and masking them would delete precisely the
    longest occlusions, which are the ones the clock is for.

    Computed with two cumulative passes rather than a python loop over t, so
    the whole training set is a few milliseconds.
    """
    v = np.asarray(visible)
    vis = v > VISIBLE_THRESHOLD
    T = vis.shape[-1]
    idx = np.arange(T)
    big = T + clip + 1

    # Running index of the most recent visible frame, via a max-accumulate over
    # ``idx`` masked to -big where hidden. Where nothing has been seen yet the
    # accumulator is still -big, which makes ``since`` enormous and the clip
    # below turns it into "clip".
    last = np.maximum.accumulate(np.where(vis, idx, -big), axis=-1)
    since = idx - last

    # The mirror image. A running MINIMUM over the frames AHEAD is a running
    # maximum of the negated indices over the reversed sequence, so the same
    # one-pass accumulate does both directions -- note the minus sign inside
    # ``where``, which is the whole trick and the easy thing to get wrong.
    rev = np.maximum.accumulate(np.where(vis, -idx, -big)[..., ::-1], axis=-1)
    nxt = -rev[..., ::-1]          # first visible index at or after t
    until = nxt - idx

    out = np.stack([since, until], axis=-1).astype(np.float32)
    np.clip(out, 0, clip, out=out)
    return out / float(clip)


def unscale(targets: np.ndarray, clip: int = CLOCK_CLIP) -> np.ndarray:
    """[0, 1] -> frames. The inverse of the scaling, for readable reports."""
    return np.asarray(targets) * float(clip)


# ------------------------------------------------------- the frozen z-probe


def fit_visibility_probe(
    roots: Sequence[str | Path],
    latent_suffix: str = "",
    n_samples: int = 40000,
    seed: int = 0,
) -> Tuple[Poly2Probe, float]:
    """Fit ``mu -> ball_visible`` on TRAINING latents. Returns (probe, R^2).

    Same shape of thing as ``train_rnn._fit_cons_probe``, and deliberately the
    same class: degree-2 ridge, fitted in closed form on standardised ``mu``,
    frozen thereafter. The reported R^2 is held out BY EPISODE -- frames inside
    an episode are near-duplicates of each other, so a per-frame split would
    report a number that is mostly memorisation.

    Fitted on ``mu`` and not on posterior samples: the probe is meant to read
    the visibility a frame actually has, and the mean is the best estimate of
    that. At use time it is also applied to ``mu``, so the two agree.

    On v3.1 this scores R^2 0.995 and 99.2 % binary agreement at the 0.5
    threshold. If it ever comes back low, the fair clock run is measuring the
    probe and not the objective -- which is why the number is printed before
    training starts and why ``--clock-privileged`` exists as the comparison.
    """
    col = _visible_column(roots)
    mus: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    for ri, r in enumerate(roots):
        d = episode_arrays(r, latent_suffix=latent_suffix)
        mu, st = d["mu"], d["state"]
        E, T = mu.shape[0], mu.shape[1]
        mus.append(mu.reshape(E * T, -1))
        ys.append(st[:, :, col].reshape(E * T))
        groups.append(np.repeat(np.arange(E) + 1000 * ri, T))

    M = np.concatenate(mus)
    Y = np.concatenate(ys)
    G = np.concatenate(groups)

    rng = np.random.default_rng(seed)
    if len(M) > n_samples:
        sel = rng.choice(len(M), n_samples, replace=False)
        M, Y, G = M[sel], Y[sel], G[sel]

    tr, te = make_split(len(M), seed=seed, group_ids=G)
    held = Poly2Probe().fit(M[tr], Y[tr])
    pred = held(M[te])
    r2 = 1.0 - float(((Y[te] - pred) ** 2).sum()) / max(
        float(((Y[te] - Y[te].mean()) ** 2).sum()), 1e-12)
    # Refit on everything for the probe actually used; the split above existed
    # only to produce an honest quality number.
    return Poly2Probe().fit(M, Y), r2


def _visible_column(roots: Sequence[str | Path]) -> int:
    """Index of ``ball_visible``, by NAME. See ``train_rnn._state_column``."""
    names = json.loads((Path(roots[0]) / "meta.json").read_text()).get(
        "state_names", [])
    if "ball_visible" not in names:
        raise SystemExit("this dataset has no ball_visible column; the clock "
                         "head is v3+ only")
    return names.index("ball_visible")


# ------------------------------------------------- per-root target arrays


def clock_arrays_for_roots(
    roots: Sequence[str | Path],
    probe: Poly2Probe | None,
    latent_suffix: str = "",
    clip: int = CLOCK_CLIP,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """One (E, T+1, 2) target array per root, plus agreement diagnostics.

    ``probe`` is the frozen z-probe for the FAIR run; pass ``None`` for the
    PRIVILEGED run, which reads the simulator's ``ball_visible`` column
    instead. That is the only difference between the two, and it is one
    argument, so there is no chance of the two runs differing in anything else.

    The counters are computed over the WHOLE episode and only then sliced into
    windows by the dataset. That is the point of doing it here: a 32-frame
    window cannot see the exit of a 21-frame occlusion that began 15 frames
    before it, so a window-local computation would clip most of the long
    occlusions to "still hidden at the end of the window" and destroy exactly
    the signal the head is for.
    """
    col = _visible_column(roots)
    arrays: List[np.ndarray] = []
    n_agree = n_total = 0
    for r in roots:
        d = episode_arrays(r, latent_suffix=latent_suffix)
        true_vis = d["state"][:, :, col]                      # (E, T+1)
        if probe is None:
            vis = true_vis
        else:
            vis = np.asarray(probe(d["mu"]), dtype=np.float32)  # (E, T+1)
        n_agree += int(((vis > VISIBLE_THRESHOLD)
                        == (true_vis > VISIBLE_THRESHOLD)).sum())
        n_total += int(true_vis.size)
        arrays.append(clock_targets(vis, clip=clip))          # (E, T+1, 2)
    return arrays, {
        "threshold_agreement": n_agree / max(n_total, 1),
        "n_frames": n_total,
    }


# ------------------------------------------------------------ two-seed help


def seed_summary(values: Sequence[float]) -> Dict[str, float]:
    """mean / spread over the seeds of one row. Used by the Part B table.

    ``spread`` is max - min, NOT a standard deviation: with two seeds the
    sample sd is just |a - b| / sqrt(2), which is the same information wearing
    a costume that invites people to read it as a confidence interval. The
    honest statement about two numbers is how far apart they are, so that is
    what is reported, and any ordering whose gap is smaller than the spreads it
    is separating is called unresolved.

    ``nan`` entries are dropped (a run that did not finish), and an empty list
    gives all-nan rather than raising, so a half-finished table still prints.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "spread": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "spread": float(v.max() - v.min()),
        "min": float(v.min()),
        "max": float(v.max()),
    }

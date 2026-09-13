"""v4 stage two: is the gravity sign in memory, and does the dream obey it?

    python -m wm.eval_switch_v4 \
        --models lstm=runs/rnn_v4/rnn.pt transformer=runs/tf_v4/rnn.pt \
                 ff=runs/rnn_v4_ff/rnn.pt \
        --vae runs/vae_v4/vae.pt \
        --val data/v4/val data/v4/val_mix --long data/v4/long \
        --comparison runs/v4_switch_comparison.png

Five experiments, in the order the claims depend on each other. Each writes into
``runs/<model>/switch/``; the cross-model figure is written once.

(a) **Is the sign in h, and for how long?** A logistic probe from h_t to the
    sign in force at frame t, split at the episode level, reported as a function
    of *frames since the last flip*. The same probe from z_t is the per-frame
    null and should sit at chance -- that is stage one's result, re-measured
    with stage two's instrument so the two curves can be drawn on one axis.

(b) **Knowing the sign is not the same as remembering the flip**, and v4's data
    forces the distinction. `runs/v4_design/sweep.md` found late that under
    vertical gravity the sign also sets the ball's energy budget, so one
    floor-to-ceiling traverse takes 48 frames under one sign and 40 under the
    other. That is a *trajectory* cue, available to any model watching the last
    40 frames, with no memory of the contact whatsoever. So the headline number
    in (a) is an upper bound on memory, and two sub-measurements separate it:

        (i)  accuracy in the first 10 frames after a flip. No traverse has
             happened yet, so nothing about the new sign is in the trajectory.
             Only a model that saw the contact and drew the conclusion can be
             right here. This is MEMORY.
        (ii) accuracy 50+ frames after a flip, on a pass whose hidden state was
             started AFTER the flip -- the model never saw the contact and has
             only the motion since. This is INFERENCE.

    A model can be good at one and bad at the other, and which one it is good at
    is the actually interesting result.

(c) **Does the dream curve the right way?** Dream 60 steps from a flip-free
    warm-up, decode positions with a kNN probe, fit the vertical acceleration of
    the dreamed path between bounces, compare its sign to the truth. The
    baseline is a straight line (acceleration exactly 0), and the ceiling is the
    same fit on the TRUE trajectory, which says whether 60 dreamed frames can
    carry a 1e-4 acceleration at all.

(d) **The flip counterfactual** -- the interventional test, and the only one
    that cannot be passed by reading the trajectory. Two warm-ups from the SAME
    recorded state and the SAME action stream, differing only in where the
    paddle started: in one the ball is struck (and gravity flips), in the other
    it sails past into the floor (and gravity does not). Dream both; the dreamed
    accelerations should differ in sign.

    Run in two variants, because of the traverse cue again. With four frames of
    post-contact motion in the warm-up, a model could in principle read the new
    sign off the ball's speed rather than off the contact. The one-frame variant
    removes almost all of that: at ``t_c + 1`` the only thing distinguishing the
    two warm-ups is the collision itself.

(e) **Half-life.** The frames-since-flip at which (a)'s curve has fallen halfway
    from its 0-10 value to chance, plus the sign-consistency of a long sampled
    dream (the conservation reading: between flips the sign is a conserved
    quantity, and nothing in the model restores it once it drifts).

A note on what "the sign at time t" means. ``h_t`` is produced by consuming
``(z_t, a_t)``; a contact that happens *during* step t is not visible in ``z_t``,
so the target is the sign recorded at frame t (``states[e, t, -1]``), not the one
after the step. A "flip at t" here means the sign recorded at t differs from the
one recorded at t-1 -- i.e. the event happened in the previous step and its
consequence is now in force.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from worldsim.bouncing_box import EVENT_PADDLE, BouncingBox, BoxConfig

from .eval_rnn import StateProbe
from .probes import make_split
from .rnn import load_rnn
from .seq_data import episode_arrays

# Frames-since-flip bins for the memory curve. Fine where the interesting decay
# happens and coarse in the tail, which only the 600-frame ``long`` split fills.
BINS: Tuple[Tuple[int, int], ...] = (
    (0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 10 ** 9),
)
BIN_LABELS = ("0-10", "10-25", "25-50", "50-100", "100-200", "200+")

COLORS = {
    "lstm": "#c2543a", "transformer": "#3a7bd5", "ff": "#7a7a7a",
    "noact": "#2a9d4a", "ctx32": "#9a6bd0", "match": "#d99a2b",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _color(name: str) -> str:
    for k, v in COLORS.items():
        if k in name:
            return v
    return "#444444"


# --------------------------------------------------------------- the data


@dataclass
class Split:
    """One or more dataset roots, stacked. Episodes must share a length."""

    mu: np.ndarray          # (E, T+1, z)
    actions: np.ndarray     # (E, T)
    state: np.ndarray       # (E, T+1, S)
    hit: np.ndarray         # (E, T)
    names: List[str]
    cfg: BoxConfig
    root: str

    @property
    def j_sign(self) -> int:
        # By NAME, never by position: v2's mass and v3's ball_visible also live
        # at the end of the state vector and the column order differs by world.
        return self.names.index("gravity_sign")

    @property
    def sign(self) -> np.ndarray:
        """(E, T+1) in {-1, +1}: the sign in force at each frame."""
        return np.sign(self.state[..., self.j_sign])


def load_split(roots: Sequence[str]) -> Split:
    mus, acts, sts, hits, names, cfg, tag = [], [], [], [], None, None, []
    for r in roots:
        d = episode_arrays(r)
        mus.append(d["mu"])
        acts.append(d["actions"])
        sts.append(d["state"])
        hits.append(d["hit"])
        meta = d["meta"]
        names = list(meta["state_names"])
        cfg = BoxConfig(**meta["config"])
        tag.append(Path(r).name)
    T = min(a.shape[1] for a in acts)
    return Split(
        mu=np.concatenate([m[:, : T + 1] for m in mus], 0),
        actions=np.concatenate([a[:, :T] for a in acts], 0),
        state=np.concatenate([s[:, : T + 1] for s in sts], 0),
        hit=np.concatenate([h[:, :T] for h in hits], 0),
        names=names, cfg=cfg, root="+".join(tag),
    )


def flip_structure(sign: np.ndarray) -> Dict[str, np.ndarray]:
    """Per frame: frames since the last flip, and whether a flip has happened.

    ``since[e, t]`` counts from the frame at which the new sign first applies.
    Before an episode's first flip it counts from frame 0 instead and
    ``had_flip`` is False there -- those frames are a different question ("do you
    know the sign you were born with") and are reported separately rather than
    mixed in.
    """
    E, Tp1 = sign.shape
    changed = np.zeros_like(sign, dtype=bool)
    changed[:, 1:] = sign[:, 1:] != sign[:, :-1]
    since = np.zeros((E, Tp1), dtype=np.int64)
    had = np.zeros((E, Tp1), dtype=bool)
    for e in range(E):
        last, seen = 0, False
        for t in range(Tp1):
            if changed[e, t]:
                last, seen = t, True
            since[e, t] = t - last
            had[e, t] = seen
    return {"since": since, "had_flip": had, "flip_at": changed}


# ------------------------------------------------------- hidden states


@torch.no_grad()
def hidden_states(model, mu, actions, device: str = "cpu",
                  t0: int = 0, batch: int = 16) -> np.ndarray:
    """Teacher-forced pass from ``t0`` with a FRESH state. (E, T-t0, hidden).

    ``t0 > 0`` is not a slice of the t0 = 0 pass: the hidden state (or, for the
    transformer, the context buffer) starts empty, so the model has genuinely
    not seen anything before ``t0``. That is what experiment (b)(ii) needs --
    a model that must infer the sign from the motion because it was not present
    at the contact.

    Batched over episodes only to keep peak memory bounded on 8 GB; the
    transformer's own ``forward`` handles windows longer than its context.
    """
    model.eval()
    E, T = actions.shape
    eye = torch.eye(model.cfg.n_actions, device=device)
    out = []
    for i in range(0, E, batch):
        z = torch.from_numpy(mu[i : i + batch, t0:T].astype(np.float32)).to(device)
        a = eye[torch.from_numpy(actions[i : i + batch, t0:]).long().to(device)]
        parts, _ = model(z, a)
        out.append(parts["h"].float().cpu().numpy())
    return np.concatenate(out, 0)


# ------------------------------------------------------------ the probe


@dataclass
class SignProbe:
    """A frozen logistic map from a feature vector to the sign. (a)'s ruler.

    Linear on purpose. The question is whether the sign is *represented* -- i.e.
    whether a downstream reader as simple as the model's own linear heads could
    use it -- not whether it is recoverable by an arbitrarily flexible function
    of a 256-d state, which on autocorrelated data mostly measures the probe.
    """

    scaler: object = None
    clf: object = None
    train_groups: np.ndarray = field(default=None)

    def fit(self, X: np.ndarray, y: np.ndarray, tr: np.ndarray) -> "SignProbe":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(X[tr])
        self.clf = LogisticRegression(max_iter=3000, C=1.0).fit(
            self.scaler.transform(X[tr]), y[tr])
        return self

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(X) == 0:
            return float("nan")
        return float((self.clf.predict(self.scaler.transform(X)) == y).mean())

    def balanced_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean of the two per-class recalls.

        The base rate here is 0.537 and it moves from bin to bin (down-traverses
        take longer, so slow-sign frames are over-represented, and differently
        so at each distance from a flip). Plain accuracy therefore mixes "how
        much does the probe know" with "how skewed is this bin", and a bin-to-bin
        comparison of plain accuracies is not quite a like-for-like. Balanced
        accuracy has its null pinned at 0.5 whatever the skew.
        """
        if len(X) == 0:
            return float("nan")
        pred = self.clf.predict(self.scaler.transform(X))
        rec = [float((pred[y == c] == c).mean()) for c in (-1.0, 1.0)
               if (y == c).any()]
        return float(np.mean(rec)) if rec else float("nan")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.scaler.transform(X))


def _binned_accuracy(probe: SignProbe, X, y, since, mask) -> List[Dict]:
    rows = []
    for (lo, hi), lab in zip(BINS, BIN_LABELS):
        sel = mask & (since >= lo) & (since < hi)
        rows.append({
            "bin": lab, "lo": lo, "n": int(sel.sum()),
            "acc": probe.accuracy(X[sel], y[sel]) if sel.sum() else float("nan"),
            "balanced_acc": probe.balanced_accuracy(X[sel], y[sel])
            if sel.sum() else float("nan"),
            "base_rate": float(max((y[sel] > 0).mean(), (y[sel] < 0).mean()))
            if sel.sum() else float("nan"),
        })
    return rows


def experiment_a(model, splits: Sequence[Split], device: str, seed: int,
                 stride: int = 2) -> Dict:
    """The memory curve: sign accuracy from h, versus frames since the flip."""
    Xs, Zs, ys, sinces, hads, groups = [], [], [], [], [], []
    g0 = 0
    for sp in splits:
        H = hidden_states(model, sp.mu, sp.actions, device)      # (E, T, D)
        E, T, _ = H.shape
        fs = flip_structure(sp.sign)
        # h_t aligns with frames 0..T-1 of mu/state; drop the last state frame.
        Xs.append(H.reshape(E * T, -1))
        Zs.append(sp.mu[:, :T].reshape(E * T, -1))
        ys.append(sp.sign[:, :T].reshape(-1))
        sinces.append(fs["since"][:, :T].reshape(-1))
        hads.append(fs["had_flip"][:, :T].reshape(-1))
        groups.append(np.repeat(np.arange(E) + g0, T))
        g0 += E
    X = np.concatenate(Xs, 0)
    Z = np.concatenate(Zs, 0)
    y = np.concatenate(ys, 0)
    since = np.concatenate(sinces, 0)
    had = np.concatenate(hads, 0)
    grp = np.concatenate(groups, 0)

    # Sub-sample the TRAINING frames only; the test set stays whole so the
    # per-bin counts are honest. Consecutive frames are near-duplicates, so
    # every other one loses almost nothing and halves the fit time.
    tr, te = make_split(len(X), frac_train=0.7, seed=seed, group_ids=grp)
    tr = tr[::stride]

    ph = SignProbe().fit(X, y, tr)
    pz = SignProbe().fit(Z, y, tr)

    # The null. Same features, same split, same classifier, labels permuted
    # between whole episodes -- so the run structure (and hence the effective
    # sample size, which is what actually sets the spread) is preserved.
    rng = np.random.default_rng(seed + 17)
    y_null = y.copy()
    gs = np.unique(grp)
    perm = rng.permutation(len(gs))
    idx_by_g = {g: np.where(grp == g)[0] for g in gs}
    for dst, src in enumerate(perm):
        a, b = idx_by_g[gs[dst]], idx_by_g[gs[src]]
        n = min(len(a), len(b))
        y_null[a[:n]] = y[b[:n]]
    p_null = SignProbe().fit(X, y_null, tr)

    test = np.zeros(len(X), dtype=bool)
    test[te] = True
    return {
        "n_frames": int(len(X)), "n_test_frames": int(test.sum()),
        "hidden_dim": int(X.shape[1]),
        "overall_h": ph.accuracy(X[te], y[te]),
        "overall_h_balanced": ph.balanced_accuracy(X[te], y[te]),
        "overall_z": pz.accuracy(Z[te], y[te]),
        "overall_null": p_null.accuracy(X[te], y_null[te]),
        "overall_null_balanced": p_null.balanced_accuracy(X[te], y_null[te]),
        # The overfitting diagnostic. A probe scoring 0.99 in-sample and 0.55
        # held out is measuring its own capacity, not the model's memory; one
        # scoring 0.56 in-sample has simply found that the bit is not there.
        "train_h": ph.accuracy(X[tr], y[tr]),
        "n_train_frames": int(len(tr)), "n_train_episodes": int(len(np.unique(grp[tr]))),
        "n_test_episodes": int(len(np.unique(grp[te]))),
        "base_rate": float(max((y[te] > 0).mean(), (y[te] < 0).mean())),
        "bins_h": _binned_accuracy(ph, X, y, since, test & had),
        "bins_z": _binned_accuracy(pz, Z, y, since, test & had),
        "bins_h_no_flip_yet": _binned_accuracy(ph, X, y, since, test & ~had),
        "_probe": ph, "_test": test, "_groups": grp,
    }


def half_life(bins: Sequence[Dict], floor: float = 0.5,
              margin: float = 0.03) -> Optional[float]:
    """Frames-since-flip at which accuracy falls halfway from bin 0 to chance.

    Read off the piecewise-linear curve through the bins' left edges. Returns
    None if the curve never crosses -- which for a flat transformer curve is the
    *result*, not a failure, and is reported as "no decay measured".
    """
    xs = [b["lo"] for b in bins if np.isfinite(b["acc"]) and b["n"] > 30]
    ys = [b["acc"] for b in bins if np.isfinite(b["acc"]) and b["n"] > 30]
    if len(xs) < 2:
        return None
    # Nothing to measure the decay of unless the curve STARTS clearly above
    # the floor. Without the margin a model whose first bin is a hair above the
    # null reports a half-life of half a frame, which reads as "forgets
    # instantly" when the truth is "never knew".
    if ys[0] <= floor + margin:
        return None
    target = floor + 0.5 * (ys[0] - floor)
    for i in range(1, len(xs)):
        if ys[i] <= target:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if abs(y1 - y0) < 1e-9:
                return float(x1)
            return float(x0 + (y0 - target) * (x1 - x0) / (y0 - y1))
    return None


# ------------------------------------------------- (b) memory vs inference


def experiment_b(model, splits: Sequence[Split], probe: SignProbe,
                 test_groups: set, device: str, min_gap: int = 60) -> Dict:
    """Separate "saw the contact" from "read it off the motion".

    (i) is a slice of (a): the first 10 frames after a flip, on held-out
    episodes. (ii) needs a different pass entirely -- one started after the
    flip, so the contact is outside the model's history.
    """
    # ---- (i) memory: read straight off the same probe, restricted to flips.
    acc_i, n_i = [], 0
    # ---- (ii) inference: fresh passes starting one frame after a flip.
    acc_ii, n_ii = [], 0
    g0 = 0
    for sp in splits:
        E, T = sp.actions.shape
        fs = flip_structure(sp.sign)
        H = hidden_states(model, sp.mu, sp.actions, device)
        for e in range(E):
            if (e + g0) not in test_groups:
                continue
            sel = fs["had_flip"][e, :T] & (fs["since"][e, :T] < 10)
            if sel.sum():
                pred = probe.predict(H[e][sel])
                acc_i.append(float((pred == sp.sign[e, :T][sel]).mean()))
                n_i += int(sel.sum())

            # The longest flip-to-next-flip run in this episode, so that the
            # 50+ frames we score are all under one sign.
            flips = np.where(fs["flip_at"][e, :T])[0]
            best = None
            for f in flips:
                nxt = flips[flips > f]
                end = int(nxt[0]) if len(nxt) else T
                if end - f > (best[1] - best[0] if best else min_gap):
                    best = (int(f), end)
            if best is None:
                continue
            f, end = best
            Hf = hidden_states(model, sp.mu[e : e + 1], sp.actions[e : e + 1],
                               device, t0=f)                      # (1, T-f, D)
            off = np.arange(Hf.shape[1])
            keep = (off >= 50) & (off + f < end)
            if keep.sum():
                pred = probe.predict(Hf[0][keep])
                acc_ii.append(float((pred == sp.sign[e, f:T][keep]).mean()))
                n_ii += int(keep.sum())
        g0 += E
    return {
        "memory_first_10_after_flip": float(np.mean(acc_i)) if acc_i else float("nan"),
        "memory_n_frames": n_i, "memory_n_episodes": len(acc_i),
        "inference_50plus_cold_start": float(np.mean(acc_ii)) if acc_ii else float("nan"),
        "inference_n_frames": n_ii, "inference_n_episodes": len(acc_ii),
    }


# ------------------------------------------------------- (c) dream curvature


def fit_vertical_acceleration(y: np.ndarray, min_seg: int = 18
                              ) -> Tuple[float, int]:
    """Vertical acceleration of a trajectory, fitted between bounces.

    A bounce reverses vy discontinuously, so a single quadratic over a path that
    contains one is meaningless -- it fits the corner, not the curvature. Split
    at the turning points, fit ``y = c0 + c1 t + c2 t^2 / 2`` on each long
    enough piece and return the length-weighted mean of ``c2``, plus the number
    of frames that contributed.

    Returns (nan, 0) when no segment is long enough; at g = 1e-4 a segment
    shorter than ~18 frames has a signal-to-noise ratio below 1 even on the true
    trajectory, so a number from one would be noise wearing a decimal point.
    """
    y = np.asarray(y, dtype=float).ravel()
    if len(y) < min_seg:
        return float("nan"), 0
    d = np.diff(y)
    # Turning points: where the first difference changes sign. Smoothed with a
    # 3-frame median first, or decoder jitter invents a bounce every few frames.
    if len(d) >= 3:
        d = np.array([np.median(d[max(0, i - 1): i + 2]) for i in range(len(d))])
    turns = [0] + [i + 1 for i in range(1, len(d)) if d[i] * d[i - 1] < 0] + [len(y)]
    accs, ws = [], []
    for a, b in zip(turns[:-1], turns[1:]):
        if b - a < min_seg:
            continue
        t = np.arange(b - a, dtype=float)
        c = np.polyfit(t, y[a:b], 2)         # c[0] * t^2 + ...
        accs.append(2.0 * c[0])
        ws.append(b - a)
    if not accs:
        return float("nan"), 0
    w = np.asarray(ws, float)
    return float(np.average(accs, weights=w)), int(w.sum())


@torch.no_grad()
def dream(model, mu_warm: np.ndarray, a_warm: np.ndarray, a_dream: np.ndarray,
          device: str, temperature: float = 0.0, seed: int = 0) -> np.ndarray:
    """Warm up teacher-forced, then roll open-loop on TRUE actions. (N, H, z).

    Actions stay real throughout: the question is whether the model's *physics*
    carries the sign, not whether it can also guess the behaviour policy.
    """
    model.eval()
    eye = torch.eye(model.cfg.n_actions, device=device)
    z = torch.from_numpy(mu_warm.astype(np.float32)).to(device)
    aw = eye[torch.from_numpy(a_warm).long().to(device)]
    ad = eye[torch.from_numpy(a_dream).long().to(device)]
    gen = torch.Generator(device="cpu").manual_seed(seed)

    parts, h = model(z, aw)
    zt = model.sample_next(parts, temperature=temperature)[:, -1]
    out = [zt]
    for k in range(1, ad.shape[1]):
        parts, h = model.step(zt, ad[:, k], h)
        if temperature <= 0.0:
            zt = model.most_likely_mean(parts)[:, 0]
        else:
            zt = model.sample_next(
                parts, temperature=temperature,
                generator=gen if device == "cpu" else None)[:, 0]
        out.append(zt)
    return torch.stack(out, 1).cpu().numpy()


def _flip_free_starts(sp: Split, warm: int, horizon: int, per_ep: int,
                      rng) -> List[Tuple[int, int]]:
    """(episode, t0) pairs whose whole warm-up AND dream are under one sign."""
    fs = flip_structure(sp.sign)
    T = sp.actions.shape[1]
    out = []
    for e in range(sp.mu.shape[0]):
        ok = [t0 for t0 in range(0, T - warm - horizon - 1)
              if not fs["flip_at"][e, t0 : t0 + warm + horizon + 1].any()]
        if ok:
            out += [(e, int(t)) for t in rng.choice(
                ok, size=min(per_ep, len(ok)), replace=False)]
    return out


def experiment_c(model, sp: Split, probe: StateProbe, device: str, seed: int,
                 warm: int = 16, horizon: int = 60, per_ep: int = 3) -> Dict:
    """Fit the dreamed trajectory's vertical acceleration; check its sign."""
    rng = np.random.default_rng(seed)
    starts = _flip_free_starts(sp, warm, horizon, per_ep, rng)
    if not starts:
        return {"note": "no flip-free windows"}
    ee = np.array([e for e, _ in starts])
    tt = np.array([t for _, t in starts])

    mu_warm = np.stack([sp.mu[e, t : t + warm] for e, t in starts])
    a_warm = np.stack([sp.actions[e, t : t + warm] for e, t in starts])
    a_dream = np.stack([sp.actions[e, t + warm : t + warm + horizon]
                        for e, t in starts])
    zd = dream(model, mu_warm, a_warm, a_dream, device, 0.0, seed)

    # Positions: the kNN probe, the same instrument v1 used, fitted on TRUE
    # latents. Its job here is only to turn a latent path into a y(t).
    y_dream = probe(zd)[..., 1]                                  # (N, H)
    idx = tt[:, None] + warm + np.arange(horizon)[None, :]
    y_true = sp.state[ee[:, None], idx, 1]                       # simulator
    y_dec = probe(sp.mu[ee[:, None], idx])[..., 1]               # decoded truth
    truth = sp.sign[ee, tt + warm]
    g = float(sp.cfg.gravity)

    def table(Y, label):
        fits = [fit_vertical_acceleration(Y[i]) for i in range(len(Y))]
        acc = np.array([f[0] for f in fits])
        ok = np.isfinite(acc)
        return {
            "what": label, "n": int(ok.sum()), "n_attempted": int(len(Y)),
            "sign_agreement": float((np.sign(acc[ok]) == truth[ok]).mean())
            if ok.any() else float("nan"),
            "mean_abs_acc": float(np.abs(acc[ok]).mean()) if ok.any() else float("nan"),
            "median_signed_acc_when_up": float(np.median(acc[ok & (truth > 0)]))
            if (ok & (truth > 0)).any() else float("nan"),
            "median_signed_acc_when_down": float(np.median(acc[ok & (truth < 0)]))
            if (ok & (truth < 0)).any() else float("nan"),
        }

    return {
        "warm": warm, "horizon": horizon, "n_windows": len(starts),
        "true_gravity": g,
        "rows": [table(y_true, "true state (the ceiling)"),
                 table(y_dec, "decoded true latents (probe noise only)"),
                 table(y_dream, "dreamed (tau=0)")],
        # A straight line has acceleration 0 by construction, so its sign
        # agreement is undefined rather than 0.5; recorded explicitly so the
        # comparison figure can say so instead of drawing a bar at nothing.
        "linear_baseline_acc": 0.0,
        "_y_dream": y_dream, "_y_true": y_true, "_truth": truth,
    }


# -------------------------------------------------- (d) the flip counterfactual


def env_from_state(cfg: BoxConfig, s: np.ndarray, names: Sequence[str],
                   paddle_dx: float = 0.0) -> BouncingBox:
    """A simulator wound to a recorded state, optionally with the paddle moved.

    Everything the dynamics depends on is in the state vector, so this is an
    exact restore and not an approximation -- ``resimulate`` asserts it by
    replaying a recorded window and comparing.

    ``paddle_dx`` is the intervention: it shifts the paddle's starting position
    and nothing else. The action stream is left alone, so the paddle's *motion*
    over the window is identical in both arms and the only difference is where
    it is standing. That keeps the model's action input identical between the
    factual and the counterfactual, which a "move the paddle by changing the
    actions" intervention would not.
    """
    env = BouncingBox(cfg)
    env.reset(seed=0)
    hw = cfg.paddle_w / 2.0
    env.ball = np.array([float(s[0]), float(s[1])], dtype=np.float64)
    env.ball_v = np.array([float(s[2]), float(s[3])], dtype=np.float64)
    env.paddle_x = float(np.clip(float(s[4]) + paddle_dx, hw, 1.0 - hw))
    env.paddle_vx = float(s[5])
    env.gravity_sign = float(s[list(names).index("gravity_sign")])
    env.t = 0
    return env


def resimulate(cfg: BoxConfig, s0: np.ndarray, names: Sequence[str],
               actions: np.ndarray, paddle_dx: float = 0.0):
    """Replay a window from a recorded state. Returns (frames, states, events)."""
    env = env_from_state(cfg, s0, names, paddle_dx)
    frames, states, events = [], [], []
    for a in actions:
        f, st, ev = env.step(int(a))
        frames.append(f)
        states.append(st)
        events.append(ev)
    return np.stack(frames), np.stack(states), np.asarray(events)


@torch.no_grad()
def encode(vae, frames: np.ndarray, device: str) -> np.ndarray:
    shape = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:]).astype(np.float32) / 255.0
    x = torch.from_numpy(flat).permute(0, 3, 1, 2).to(device)
    mu, _ = vae.encode(x)
    return mu.cpu().numpy().reshape(*shape, -1)


def build_counterfactual_pairs(sp: Split, pre: int, post: int, horizon: int,
                               n_pairs: int, seed: int,
                               offsets=(0.45, -0.45, 0.6, -0.6, 0.3, -0.3),
                               ) -> List[Dict]:
    """Real contacts, each paired with a re-simulated near-identical miss.

    And the reverse where one can be built: a real floor bounce with the paddle
    moved underneath so that it becomes a contact. The reverse direction matters
    because in the forward one the counterfactual is always the "nothing
    happened" arm, and a model that simply responds to *any* collision would
    pass. It has to be the paddle specifically.
    """
    rng = np.random.default_rng(seed)
    T = sp.actions.shape[1]
    names = sp.names
    out: List[Dict] = []

    hits = [(e, t) for e in range(sp.mu.shape[0])
            for t in np.where(sp.hit[e] > 0.5)[0]
            if pre <= t and t + post + horizon + 1 < T]
    rng.shuffle(hits)
    for e, t_c in hits:
        if sum(1 for r in out if r["direction"] == "hit_to_miss") >= n_pairs:
            break
        t0 = t_c - pre
        acts = sp.actions[e, t0 : t_c + post + 1]
        s0 = sp.state[e, t0]
        fr_f, st_f, ev_f = resimulate(sp.cfg, s0, names, acts, 0.0)
        # The factual arm must reproduce the recording, or the restore is wrong
        # and every counterfactual built on it is meaningless.
        if np.abs(st_f[:, :2] - sp.state[e, t0 + 1 : t_c + post + 2, :2]).max() > 1e-4:
            continue
        for dx in offsets:
            fr_c, st_c, ev_c = resimulate(sp.cfg, s0, names, acts, dx)
            if not (ev_c & EVENT_PADDLE).any():
                out.append({
                    "direction": "hit_to_miss", "e": int(e), "t_c": int(t_c),
                    "paddle_dx": float(dx),
                    "frames_f": fr_f, "frames_c": fr_c,
                    "states_f": st_f, "states_c": st_c,
                    "actions_dream": sp.actions[e, t_c + post + 1:
                                                t_c + post + 1 + horizon],
                })
                break

    # ---- the reverse. A ball low in the box with no contact in the window,
    # with the paddle moved underneath so that the counterfactual DOES connect.
    misses = []
    for e in range(sp.mu.shape[0]):
        below = sp.state[e, :, 1] < (sp.cfg.paddle_y + 0.12)
        for t in np.where(below[: T - post - horizon - 1])[0]:
            if t >= pre and not sp.hit[e, t - pre : t + post + 1].any():
                misses.append((int(e), int(t)))
    rng.shuffle(misses)
    for e, t_c in misses:
        if sum(1 for r in out if r["direction"] == "miss_to_hit") >= n_pairs:
            break
        t0 = t_c - pre
        acts = sp.actions[e, t0 : t_c + post + 1]
        s0 = sp.state[e, t0]
        fr_f, st_f, ev_f = resimulate(sp.cfg, s0, names, acts, 0.0)
        if (ev_f & EVENT_PADDLE).any():
            continue
        # Put the paddle under where the ball actually went.
        want = float(sp.state[e, t_c, 0]) - float(sp.state[e, t0, 4])
        for dx in (want, want + 0.05, want - 0.05):
            fr_c, st_c, ev_c = resimulate(sp.cfg, s0, names, acts, dx)
            if (ev_c & EVENT_PADDLE).any():
                out.append({
                    "direction": "miss_to_hit", "e": int(e), "t_c": int(t_c),
                    "paddle_dx": float(dx),
                    "frames_f": fr_f, "frames_c": fr_c,
                    "states_f": st_f, "states_c": st_c,
                    "actions_dream": sp.actions[e, t_c + post + 1:
                                                t_c + post + 1 + horizon],
                })
                break
    return out


def experiment_d(model, vae, sp: Split, probe: StateProbe, device: str,
                 seed: int, pre: int = 16, posts=(4, 1), horizon: int = 60,
                 n_pairs: int = 60) -> Dict:
    """Dream from a warm-up that contains a contact, and from one that does not."""
    res: Dict[str, object] = {"pre": pre, "horizon": horizon, "variants": []}
    j_sign = sp.j_sign
    for post in posts:
        pairs = build_counterfactual_pairs(sp, pre, post, horizon, n_pairs, seed)
        if not pairs:
            res["variants"].append({"post_frames": post, "note": "no pairs built"})
            continue
        # Encode both arms of every pair. The warm-up is the re-simulated window
        # in BOTH arms, so the encoder path is identical and cannot explain any
        # difference.
        fr = np.stack([p["frames_f"] for p in pairs] +
                      [p["frames_c"] for p in pairs])
        mu_w = encode(vae, fr, device)
        n = len(pairs)
        a_w = np.stack([sp.actions[p["e"], p["t_c"] - pre: p["t_c"] + post + 1]
                        for p in pairs] * 2)
        a_d = np.stack([p["actions_dream"] for p in pairs] * 2)
        zd = dream(model, mu_w, a_w, a_d, device, 0.0, seed)
        y = probe(zd)[..., 1]
        acc = np.array([fit_vertical_acceleration(y[i])[0] for i in range(len(y))])
        af, ac = acc[:n], acc[n:]

        # The truth. In the factual arm the sign in force after the contact is
        # the recorded post-contact one; in the counterfactual no flip happened,
        # so it is the pre-contact one. (For the reverse direction, swap.)
        sign_f = np.array([p["states_f"][-1][j_sign] for p in pairs])
        sign_c = np.array([p["states_c"][-1][j_sign] for p in pairs])
        ok = np.isfinite(af) & np.isfinite(ac)
        dirs = np.array([p["direction"] for p in pairs])
        by_dir = {}
        for d in ("hit_to_miss", "miss_to_hit"):
            m = ok & (dirs == d)
            by_dir[d] = {
                "n": int(m.sum()),
                "frac_opposite_sign": float(
                    (np.sign(af[m]) != np.sign(ac[m])).mean()) if m.any() else float("nan"),
                "arm_with_contact_matches_truth": float(
                    (np.sign(af[m] if d == "hit_to_miss" else ac[m])
                     == (sign_f[m] if d == "hit_to_miss" else sign_c[m])).mean())
                if m.any() else float("nan"),
            }
        res["variants"].append({
            "by_direction": by_dir,
            "post_frames": post,
            "n_pairs": int(n), "n_fitted": int(ok.sum()),
            "n_hit_to_miss": sum(1 for p in pairs if p["direction"] == "hit_to_miss"),
            "n_miss_to_hit": sum(1 for p in pairs if p["direction"] == "miss_to_hit"),
            "true_signs_differ": float((sign_f != sign_c).mean()),
            "frac_opposite_sign": float(
                (np.sign(af[ok]) != np.sign(ac[ok])).mean()) if ok.any() else float("nan"),
            "factual_matches_truth": float(
                (np.sign(af[ok]) == sign_f[ok]).mean()) if ok.any() else float("nan"),
            "counterfactual_matches_truth": float(
                (np.sign(ac[ok]) == sign_c[ok]).mean()) if ok.any() else float("nan"),
            "mean_acc_factual": float(np.nanmean(af[ok])) if ok.any() else float("nan"),
            "mean_acc_counterfactual": float(np.nanmean(ac[ok])) if ok.any() else float("nan"),
            # Kept in the report (not underscore-prefixed) so the scatter can
            # be redrawn without re-dreaming: two floats per pair is nothing.
            "acc_factual": af.tolist(), "acc_counterfactual": ac.tolist(),
            "direction": dirs.tolist(),
            "_af": af, "_ac": ac, "_sf": sign_f, "_sc": sign_c,
        })
    return res


# ------------------------------------------------ (e) the long sampled dream


@torch.no_grad()
def experiment_e(model, sp: Split, probe: SignProbe, device: str, seed: int,
                 warm: int = 16, horizon: int = 200, taus=(0.0, 1.0),
                 n_ep: int = 20) -> Dict:
    """Does the sign SURVIVE a long dream? The conservation reading.

    Between flips the sign is a constant of the motion, and nothing in the
    model's loss restores it once it drifts -- the v2 lesson about the ball's
    mass, applied to a quantity that is not even visible. Scored as: the
    fraction of dreamed steps at which the probe reads the sign the dream
    started with, given that the dream contains no contact the model itself
    predicted.
    """
    model.eval()
    E = min(n_ep, sp.mu.shape[0])
    T = sp.actions.shape[1]
    if warm + horizon >= T:
        horizon = T - warm - 1
    mu_w = sp.mu[:E, :warm]
    a_w = sp.actions[:E, :warm]
    a_d = sp.actions[:E, warm : warm + horizon]
    start_sign = sp.sign[:E, warm]

    out = {"warm": warm, "horizon": horizon, "n_episodes": int(E), "taus": []}
    eye = torch.eye(model.cfg.n_actions, device=device)
    for tau in taus:
        # Re-run the dream keeping h at every step, so the sign probe (which
        # reads h, not z) has something to read.
        z = torch.from_numpy(mu_w.astype(np.float32)).to(device)
        aw = eye[torch.from_numpy(a_w).long().to(device)]
        ad = eye[torch.from_numpy(a_d).long().to(device)]
        gen = torch.Generator(device="cpu").manual_seed(seed)
        parts, h = model(z, aw)
        zt = model.sample_next(parts, temperature=tau)[:, -1]
        hs, hits = [], []
        for k in range(1, horizon):
            parts, h = model.step(zt, ad[:, k], h)
            hs.append(parts["h"][:, 0].float().cpu().numpy())
            hits.append(torch.sigmoid(parts["hit_logit"][:, 0, 0]).cpu().numpy())
            if tau <= 0.0:
                zt = model.most_likely_mean(parts)[:, 0]
            else:
                zt = model.sample_next(
                    parts, temperature=tau,
                    generator=gen if device == "cpu" else None)[:, 0]
        H = np.stack(hs, 1)                                   # (E, H-1, D)
        P = np.stack(hits, 1)
        read = probe.predict(H.reshape(-1, H.shape[-1])).reshape(H.shape[:2])
        agree = read == start_sign[:, None]
        # A step the model itself thinks is a contact is a step at which the
        # sign is *supposed* to change, so it does not count against it.
        quiet = P < 0.5
        out["taus"].append({
            "tau": float(tau),
            "sign_consistency": float(agree[quiet].mean()) if quiet.any() else float("nan"),
            "sign_consistency_all_steps": float(agree.mean()),
            "frac_steps_model_predicts_contact": float((~quiet).mean()),
            "consistency_by_step": [float(agree[:, k].mean())
                                    for k in range(agree.shape[1])],
        })
    return out


# ---------------------------------------------------------------- figures


def fig_memory_curve(reports: Dict[str, Dict], path: Path) -> Path:
    """The headline: sign recall versus frames since the flip, per model.

    Three panels, and the middle one is the one that matters. The left panel's
    obvious reading -- "recall climbs with distance from the flip" -- is true and
    is the opposite of a memory curve, but it is not by itself evidence about M,
    because the SINGLE-FRAME null climbs too. Frames long after a flip are frames
    on which the ball has gone a long time without touching the paddle, which
    means it has been living high in the box, and how much time it spends high is
    exactly what the sign sets. So position leaks the sign more the further you
    are from a flip, and a probe on z rides that leak.

    The middle panel subtracts it: ``h`` accuracy minus ``z`` accuracy in the same
    bin, on the same frames. That is what M's recurrent state adds over the frame
    it is currently looking at, which is the only thing stage two can claim.
    """
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    xs = np.arange(len(BIN_LABELS))
    any_rep = next(iter(reports.values()))["a"]
    # BALANCED accuracy in the two left panels. The sign's base rate is 0.60 and
    # it moves from bin to bin (down-traverses are slower, so the slow sign is
    # over-represented, and unevenly so with distance from a flip). Plain
    # accuracy therefore climbs partly because the bins get more skewed -- most
    # of the z curve's apparent rise is exactly that -- and balanced accuracy,
    # whose null is 0.5 whatever the skew, takes it back out.
    z_acc = np.array([b["balanced_acc"] for b in any_rep["bins_z"]], dtype=float)
    z_plain = np.array([b["acc"] for b in any_rep["bins_z"]], dtype=float)

    for name, rep in reports.items():
        a = rep["a"]
        ys = np.array([b["balanced_acc"] for b in a["bins_h"]], dtype=float)
        ns = [b["n"] for b in a["bins_h"]]
        ax[0].plot(xs, ys, "-o", color=_color(name), lw=1.8, ms=5,
                   label=f"{name}  (h, {a['hidden_dim']}-d)")
        ax[1].plot(xs, ys - z_acc, "-o", color=_color(name), lw=1.8, ms=5,
                   label=name)
        for k, (x, y, n) in enumerate(zip(xs, ys, ns)):
            if n and n < 400 and np.isfinite(y):
                ax[0].plot([x], [y], "o", mfc="none", mec="k", ms=10)
                ax[1].plot([x], [y - z_acc[k]], "o", mfc="none", mec="k", ms=10)

    ax[0].plot(xs, z_acc, "--s", color="k", lw=1.4, ms=4,
               label="z (one frame: the null AND the leak)")
    ax[0].axhline(0.5, ls=":", c="0.4")
    nulls = [r["a"]["overall_null_balanced"] for r in reports.values()]
    if nulls:
        ax[0].axhspan(1 - max(nulls), max(nulls), color="0.88", zorder=0)
        ax[0].text(-0.4, max(nulls) + 0.012,
                   f"shuffled-label null (up to {max(nulls):.3f})",
                   fontsize=8, color="0.35", ha="left")
    ax[0].set_ylim(0.15, 1.02)
    ax[0].set_yticks(np.arange(0.2, 1.01, 0.1))
    for a_ in ax[:2]:
        a_.set_xticks(xs)
        a_.set_xticklabels(BIN_LABELS)
        a_.set_xlabel("frames since the last flip")
    ax[0].set_ylabel("held-out BALANCED sign accuracy")
    ax[0].set_title("(a) sign accuracy vs frames since the flip\n"
                    "hollow rings: fewer than 400 test frames in the bin",
                    fontsize=9)
    ax[0].legend(fontsize=7.5, loc="lower right")

    ax[1].axhline(0.0, c="k", lw=1)
    ax[1].set_ylabel("balanced accuracy from h  minus  from z")
    ax[1].set_title("what the recurrent/attentional state ADDS\n"
                    "over the single frame it is looking at", fontsize=9)
    ax[1].legend(fontsize=8)

    # (b): the two halves of (a)'s headline number, against the z baselines.
    names = list(reports)
    w = 0.38
    for i, key in enumerate(("memory_first_10_after_flip",
                             "inference_50plus_cold_start")):
        vals = [reports[n]["b"].get(key, float("nan")) for n in names]
        ax[2].bar(np.arange(len(names)) + (i - 0.5) * w, vals, w,
                  color=["#c2543a", "#3a7bd5"][i],
                  label=["(i) memory: 0-10 after the flip",
                         "(ii) inference: 50+, cold start"][i])
    # The single-frame reference for each: what a probe on z scores on the same
    # frames. The memory bar has to beat the FIRST of these to mean anything.
    ax[2].axhline(z_plain[0], ls="--", c="#c2543a", lw=1.2)
    ax[2].text(-0.45, z_plain[0] + 0.014, f"z alone, 0-10 frames ({z_plain[0]:.2f})",
               fontsize=7.5, color="#c2543a", ha="left")
    z_late = float(np.nanmean(z_plain[3:5]))
    ax[2].axhline(z_late, ls="--", c="#3a7bd5", lw=1.2)
    ax[2].text(-0.45, z_late + 0.014, f"z alone, 50-200 frames ({z_late:.2f})",
               fontsize=7.5, color="#3a7bd5", ha="left")
    ax[2].axhline(0.5, ls=":", c="0.4")
    ax[2].set_xticks(np.arange(len(names)))
    ax[2].set_xticklabels(names, fontsize=8)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("sign accuracy")
    ax[2].set_title("(b) did it remember the event, or read the motion?",
                    fontsize=9)
    ax[2].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_dream_curvature(rep: Dict, name: str, path: Path) -> Path:
    plt = _plt()
    c = rep["c"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    rows = c["rows"]
    ax[0].bar(range(len(rows)), [r["sign_agreement"] for r in rows],
              color=["#7a7a7a", "#2a9d4a", _color(name)])
    ax[0].axhline(0.5, ls="--", c="k", lw=1)
    ax[0].set_xticks(range(len(rows)))
    ax[0].set_xticklabels([r["what"].replace(" (", "\n(") for r in rows],
                          fontsize=8)
    ax[0].set_ylim(0, 1.25)
    ax[0].set_ylabel("fraction with the right sign of curvature")
    ax[0].set_title(f"(c) {name}: does the dream curve the right way?\n"
                    f"a straight line would score 0 (no curvature at all)",
                    fontsize=9)
    for i, r in enumerate(rows):
        ax[0].text(i, r["sign_agreement"] + 0.02,
                   f"{r['sign_agreement']:.2f}\n(n={r['n']}/{r['n_attempted']})",
                   ha="center", fontsize=8)

    yd, yt, tr = rep["c"]["_y_dream"], rep["c"]["_y_true"], rep["c"]["_truth"]
    for k, (i, lab) in enumerate([(int(np.argmax(tr > 0)), "sign +1 (pulls up)"),
                                  (int(np.argmax(tr < 0)), "sign -1 (pulls down)")]):
        ax[1].plot(yt[i], color=["#3a7bd5", "#c2543a"][k], lw=1.4,
                   label=f"true, {lab}")
        ax[1].plot(yd[i], color=["#3a7bd5", "#c2543a"][k], lw=1.4, ls="--",
                   label=f"dreamed, {lab}")
    ax[1].set_xlabel("dream step")
    ax[1].set_ylabel("ball_y")
    ax[1].set_title("two example dreams against the truth", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_counterfactual(rep: Dict, name: str, path: Path) -> Path:
    plt = _plt()
    # "_af" in a live report, "acc_factual" in one reloaded from JSON (the
    # underscore keys are stripped before writing). Accept either, so --replot
    # can redraw this figure without re-dreaming.
    variants = [v for v in rep["d"]["variants"]
                if "_af" in v or "acc_factual" in v]
    if not variants:
        return path
    fig, ax = plt.subplots(1, len(variants), figsize=(5.5 * len(variants), 4.2),
                           squeeze=False)
    for j, v in enumerate(variants):
        a = ax[0][j]
        af = np.asarray(v.get("_af", v.get("acc_factual")), dtype=float)
        ac = np.asarray(v.get("_ac", v.get("acc_counterfactual")), dtype=float)
        dirs = np.asarray(v["direction"])
        ok = np.isfinite(af) & np.isfinite(ac)
        for d, mk, col in (("hit_to_miss", "o", "#c2543a"),
                           ("miss_to_hit", "^", "#3a7bd5")):
            m = ok & (dirs == d)
            a.scatter(af[m] * 1e4, ac[m] * 1e4, s=20, alpha=0.75, marker=mk,
                      color=col, label=f"{d.replace('_', ' ')} (n={int(m.sum())})")
        a.legend(fontsize=7, loc="upper left")
        lim = np.nanpercentile(np.abs(np.concatenate([af[ok], ac[ok]])) * 1e4, 98)
        lim = float(max(lim, 1.0))
        a.axhline(0, c="k", lw=0.8)
        a.axvline(0, c="k", lw=0.8)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        # "Factual" is the arm re-simulated with the paddle where it really
        # was, whichever way that went; "counterfactual" is the arm with the
        # paddle displaced. In the hit->miss pairs the contact is in the
        # factual arm, in the miss->hit pairs it is in the counterfactual one,
        # which is why both markers are drawn.
        a.set_xlabel("dreamed vertical acceleration,\nFACTUAL (paddle where it was)  [1e-4]")
        a.set_ylabel("dreamed vertical acceleration,\nCOUNTERFACTUAL (paddle moved)  [1e-4]")
        a.set_title(
            f"(d) {name}: {v['post_frames']} post-contact warm-up frame"
            f"{'s' if v['post_frames'] != 1 else ''}\n"
            f"opposite signs on {v['frac_opposite_sign']:.2f} of "
            f"{v['n_fitted']} pairs (chance 0.50)", fontsize=9)
        # The two quadrants where the intervention changed the sign.
        a.fill_between([-lim, 0], 0, lim, color="#2a9d4a", alpha=0.07)
        a.fill_between([0, lim], -lim, 0, color="#2a9d4a", alpha=0.07)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------------- main


def _strip(obj):
    """Drop the underscore-prefixed arrays before writing JSON."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_strip(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True,
                   help="name=path/to/rnn.pt, one per model to compare")
    p.add_argument("--vae", default="runs/vae_v4/vae.pt")
    p.add_argument("--val", nargs="+",
                   default=["data/v4/val", "data/v4/val_mix"])
    p.add_argument("--long", default="data/v4/long",
                   help="the long split, which is what fills the tail bins")
    p.add_argument("--probe-pool", nargs="*", default=["data/v4/train"],
                   help="EXTRA roots added to (a) and (b)'s pool. val+val_mix+"
                        "long is only 65 episodes, and the sign is constant over "
                        "long runs inside an episode, so the effective sample "
                        "size of an episode-level split over them is tiny. These "
                        "roots are M's own training data, which is fine for a "
                        "representation probe -- the probe is what is being "
                        "fitted, and it is scored on held-out EPISODES -- and it "
                        "is the difference between a weak probe and a real "
                        "negative result. Pass nothing to switch it off.")
    p.add_argument("--probe-data", nargs="+", default=["data/v4/probe"],
                   help="where the mu -> position kNN probe is fitted")
    p.add_argument("--probe-samples", type=int, default=12000)
    p.add_argument("--comparison", default="runs/v4_switch_comparison.png")
    p.add_argument("--warm", type=int, default=16)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--n-pairs", type=int, default=60)
    p.add_argument("--cons-horizon", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replot", action="store_true",
                   help="rebuild the comparison figure from the report.json "
                        "files already on disk, without re-running anything")
    p.add_argument("--skip", nargs="*", default=[],
                   help="experiment letters to skip, e.g. --skip d")
    a = p.parse_args()

    if a.replot:
        reports = {}
        for spec in a.models:
            name, _, path = spec.partition("=")
            out = Path(path).parent / "switch"
            reports[name] = json.loads((out / "report.json").read_text())
            if "d" in reports[name]:
                fig_counterfactual(reports[name], name,
                                   out / "flip_counterfactual.png")
        fig_memory_curve(reports, Path(a.comparison))
        print(f"replotted {a.comparison}")
        return

    from .analyze import load_ckpt

    val = load_split(a.val)
    lng = load_split([a.long])
    pool = [val, lng] + ([load_split(a.probe_pool)] if a.probe_pool else [])
    print(f"val   {val.mu.shape[0]} episodes x {val.actions.shape[1]}  ({val.root})")
    print(f"long  {lng.mu.shape[0]} episodes x {lng.actions.shape[1]}")
    fs = flip_structure(val.sign)
    print(f"flips per episode: {fs['flip_at'].sum() / val.mu.shape[0]:.2f}  "
          f"(long: {flip_structure(lng.sign)['flip_at'].sum() / lng.mu.shape[0]:.2f})")

    vae, _, _ = load_ckpt(a.vae, a.device)
    # The position probe: mu -> true state, kNN, fitted on the probe split.
    pr = load_split(a.probe_data)
    rng = np.random.default_rng(a.seed)
    flat_mu = pr.mu.reshape(-1, pr.mu.shape[-1])
    flat_st = pr.state.reshape(-1, pr.state.shape[-1])
    sel = rng.choice(len(flat_mu), min(a.probe_samples, len(flat_mu)), replace=False)
    pos_probe = StateProbe().fit(flat_mu[sel], flat_st[sel])
    print(f"position probe fitted on {len(sel)} frames of {pr.root}")

    reports: Dict[str, Dict] = {}
    for spec in a.models:
        name, _, path = spec.partition("=")
        model, cfg = load_rnn(path, a.device)
        nparam = sum(q.numel() for q in model.parameters())
        out = Path(path).parent / "switch"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name}  {path}  arch={getattr(cfg, 'arch', 'lstm')}  "
              f"hidden={cfg.hidden}  params={nparam/1e3:.0f}k")
        rep: Dict[str, object] = {
            "name": name, "ckpt": path, "params": int(nparam),
            "arch": getattr(cfg, "arch", "lstm"), "hidden": int(cfg.hidden),
            "context": int(getattr(cfg, "context", 0)),
        }

        rep["a"] = experiment_a(model, pool, a.device, a.seed)
        print(f"    probe fitted on {rep['a']['n_train_frames']} frames from "
              f"{rep['a']['n_train_episodes']} episodes, tested on "
              f"{rep['a']['n_test_episodes']}; in-sample accuracy "
              f"{rep['a']['train_h']:.3f}")
        print(f"(a) sign from h: {rep['a']['overall_h']:.3f} overall  "
              f"| from z {rep['a']['overall_z']:.3f}  "
              f"| shuffled-label null {rep['a']['overall_null']:.3f}  "
              f"| base rate {rep['a']['base_rate']:.3f}")
        for b in rep["a"]["bins_h"]:
            print(f"      since flip {b['bin']:>8s}  n={b['n']:6d}  "
                  f"acc {b['acc']:.3f}  balanced {b['balanced_acc']:.3f}")
        # Measured against the SHUFFLED-LABEL null rather than 0.5. A curve
        # that never clears the null has no half-life to report -- the decay of
        # something that was never there is not a number.
        rep["half_life"] = half_life(
            rep["a"]["bins_h"], floor=max(0.5, rep["a"]["overall_null"]))
        hl = rep["half_life"]
        print("(e) half-life: " + (f"{hl:.0f} frames" if hl is not None else
                                    "none -- the curve never clears the null "
                                    "by enough to have one"))

        test_groups = set(np.unique(
            rep["a"]["_groups"][rep["a"]["_test"]]).tolist())
        rep["b"] = experiment_b(model, pool, rep["a"]["_probe"],
                                test_groups, a.device)
        print(f"(b) memory (0-10 after flip) {rep['b']['memory_first_10_after_flip']:.3f}"
              f"  |  inference (50+, cold start) "
              f"{rep['b']['inference_50plus_cold_start']:.3f}")

        if "c" not in a.skip:
            rep["c"] = experiment_c(model, val, pos_probe, a.device, a.seed,
                                    warm=a.warm, horizon=a.horizon)
            for r in rep["c"].get("rows", []):
                print(f"(c) {r['what']:38s} sign agreement {r['sign_agreement']:.3f}"
                      f"  |mean acc| {r['mean_abs_acc']:.2e}  (n={r['n']})")
            fig_dream_curvature(rep, name, out / "dream_curvature.png")

        if "d" not in a.skip:
            rep["d"] = experiment_d(model, vae, val, pos_probe, a.device,
                                    a.seed, pre=a.warm, horizon=a.horizon,
                                    n_pairs=a.n_pairs)
            for v in rep["d"]["variants"]:
                if "frac_opposite_sign" in v:
                    print(f"(d) {v['post_frames']} post-contact frame(s): "
                          f"opposite signs {v['frac_opposite_sign']:.3f} "
                          f"(n={v['n_fitted']} of {v['n_pairs']}: "
                          f"{v['n_hit_to_miss']} hit->miss, "
                          f"{v['n_miss_to_hit']} miss->hit), factual matches "
                          f"truth {v['factual_matches_truth']:.3f}, "
                          f"counterfactual {v['counterfactual_matches_truth']:.3f}")
                    for d, r in v["by_direction"].items():
                        print(f"        {d:12s} n={r['n']:3d}  opposite "
                              f"{r['frac_opposite_sign']:.3f}  the struck arm "
                              f"matches truth {r['arm_with_contact_matches_truth']:.3f}")
            fig_counterfactual(rep, name, out / "flip_counterfactual.png")

        if "e" not in a.skip:
            rep["e"] = experiment_e(model, val, rep["a"]["_probe"], a.device,
                                    a.seed, horizon=a.cons_horizon)
            for t in rep["e"]["taus"]:
                print(f"(e) tau={t['tau']:.1f} long-dream sign consistency "
                      f"{t['sign_consistency']:.3f}")

        (out / "report.json").write_text(json.dumps(_strip(rep), indent=2))
        print(f"wrote {out/'report.json'}")
        reports[name] = rep

    fig_memory_curve(reports, Path(a.comparison))
    print(f"\nwrote {a.comparison}")

    # ------------------------------------------------------ plain English
    print("\n" + "=" * 72)
    print("what v4 stage two found, in plain English")
    print("=" * 72)
    for name, rep in reports.items():
        aa, bb = rep["a"], rep["b"]
        hl = rep["half_life"]
        first = aa["bins_h"][0]["acc"]
        last = [b for b in aa["bins_h"] if b["n"] > 30][-1]
        print(f"\n{name} ({rep['arch']}, {rep['params']/1e3:.0f}k):")
        print(f"  Right after a flip the sign is readable from h at "
              f"{first:.0%}; by {last['bin']} frames later it is "
              f"{last['acc']:.0%}. One frame of z gives {aa['overall_z']:.0%} "
              f"and shuffled labels give {aa['overall_null']:.0%}.")
        print(f"  Of that, {bb['memory_first_10_after_flip']:.0%} is memory of "
              f"the contact itself and {bb['inference_50plus_cold_start']:.0%} "
              f"is what a cold-started pass can infer from the motion alone.")
        if hl is not None:
            print(f"  Half-life: about {hl:.0f} frames.")
        elif first <= aa["overall_null"] + 0.03:
            print("  There is no memory to decay: even immediately after the "
                  "flip the sign is not readable above the shuffled-label null.")
        else:
            print("  The recall does not measurably decay over the range "
                  "covered -- no half-life to report.")
        if "d" in rep:
            v = [x for x in rep["d"]["variants"] if "frac_opposite_sign" in x]
            if v:
                print(f"  Intervening on the contact flips the dreamed "
                      f"curvature on {v[0]['frac_opposite_sign']:.0%} of pairs "
                      f"(chance 50%).")


if __name__ == "__main__":
    main()

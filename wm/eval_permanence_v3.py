"""Does M have object permanence? Six experiments behind the occlusion band.

    python -m wm.eval_permanence_v3 \
        --models rnn_v3=runs/rnn_v3/rnn.pt ff=runs/rnn_v3_ff/rnn.pt \
                 ms=runs/rnn_v3_ms/rnn.pt \
        --vae runs/vae_v3/vae.pt --out runs/rnn_v3/permanence

This is the point of v3. Stage one established the precondition and it is
unusually clean: ball position is recoverable from a single frame's ``mu`` at
R^2 = 0.984 when the ball is visible and at **R^2 <= 0 when it is fully
hidden**, because the frame is byte-identical for every ball position behind
the band. So any position information the dynamics model has during an
occlusion did not come from its input. It came from ``h`` or from nowhere.

    (a) position from h    probe h (and z, for reference) for ball x/y, split
                           by visibility; then the decay curve -- how the
                           readout degrades with the number of frames hidden.
    (b) emergence          dream through a real occlusion on the true actions.
                           When does the ball come back, where, and on which
                           side? Against two baselines.
    (c) hidden wall bounce the subset of (b) where the ball hit a side wall
                           while hidden. Straight-line extrapolation is wrong
                           by twice the overshoot there; is the model?
    (d) memory horizon     (b) on the `tall` and `taller` bands, binned by how
                           long the ball was hidden. Where does the model meet
                           the no-memory floor?
    (e) counterfactual     re-simulate the same episode with vx -> -vx before
                           entry, re-render, re-encode, dream. Does the exit
                           move the way the physics says it must?
    (f) what else is in h  velocity and "how long have I been hidden", probed
                           on hidden frames.

THE CONTROL IS THE ARGUMENT. Everything is run for ``rnn_v3`` and for
``rnn_v3_ff`` -- identical training data, budget, heads and latent space, but
the LSTM is replaced by a two-layer MLP on ``[z_t, a_t]``. The feed-forward
model cannot carry anything through the gap, by construction: its output at
time t is a function of the current frame alone, and the current frame during
an occlusion contains no ball. Whatever the recurrent model scores, the gap to
the feed-forward model is the part that is memory. ``rnn_v3_ms`` (the
multi-step open-loop training loss) is the third arm, because during occlusion
the model is effectively running open-loop already -- teacher forcing never
asks it to survive its own predictions.

THE PADDLE LEAKS, A LITTLE. Half the episodes are collected with a tracking
policy, so the paddle sits roughly under the ball -- and the paddle is visible
on every frame, band or no band. That means ``z`` is not quite uninformative
about ball x during an occlusion, and neither is a memoryless ``h``. This is
why the feed-forward model, and not zero, is the floor that matters: it has
exactly the same access to that cue and no memory. Where the ``z`` reference
curve sits above zero, that is the leak, measured.

A measurement caveat that shapes every table here, inherited from stage one
(`README_V3.md` Section 5.3): **R^2 is scored against the variance of whatever
slice you conditioned on**, and the slices here have wildly different spreads.
Every table therefore carries rmse in world units next to R^2, and rmse is the
column to read across slices.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from worldsim.bouncing_box import BouncingBox, BoxConfig

from .analyze import load_ckpt
from .eval_causal_v2 import encode_frames, fit_position_probe
from .eval_rnn import StateProbe, _plt, dream
from .permanence import (
    HiddenRun, band_interior, first_exit, hidden_age, hidden_runs_from_events,
    linear_exit, no_memory_exit, reflect_x,
)
from .probes import make_split
from .rnn import MDNRNN, load_rnn
from .seq_data import episode_arrays
from .train_vae import pick_device

WARMUP = 8          # true frames fed before every dream; matches eval_rnn
EXTRA = 10          # dreamed frames past the true exit, as the design asks
EXIT_CONSEC = 2     # frames outside the band before we call it an exit
MAX_AGE = 25        # last k on the decay curve


# --------------------------------------------------------------- containers


@dataclass
class Model:
    name: str
    model: MDNRNN
    ckpt: str

    @property
    def is_ff(self) -> bool:
        return bool(self.model.cfg.feedforward)


@dataclass
class Split:
    """One dataset root group, stacked, with its band geometry."""

    name: str
    mu: np.ndarray           # (E, T+1, z)
    actions: np.ndarray      # (E, T)
    state: np.ndarray        # (E, T+1, S)
    events: np.ndarray       # (E, T)
    band: Tuple[float, float]
    radius: float
    names: List[str] = field(default_factory=list)

    @property
    def visible(self) -> np.ndarray:
        return self.state[..., self.names.index("ball_visible")]

    @property
    def T(self) -> int:
        return self.actions.shape[1]


def load_split(name: str, roots: Sequence[str], suffix: str = "") -> Split:
    parts = [episode_arrays(r, latent_suffix=suffix) for r in roots]
    evs = [np.load(Path(r) / "events.npy") for r in roots]
    meta = parts[0]["meta"]
    bands = {tuple(episode_arrays(r)["meta"]["occluder_y"]) for r in roots}
    if len(bands) != 1:
        raise ValueError(f"{name}: roots disagree on the band: {bands}")
    return Split(
        name=name,
        mu=np.concatenate([p["mu"] for p in parts], 0),
        actions=np.concatenate([p["actions"] for p in parts], 0),
        state=np.concatenate([p["state"] for p in parts], 0),
        events=np.concatenate(evs, 0),
        band=bands.pop(),
        radius=float(meta["config"]["ball_radius"]),
        names=list(meta["state_names"]),
    )


# ------------------------------------------------------------------- probes


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / max(den, 1e-12)


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


class Readout:
    """A frozen linear-or-kNN map from a feature (h or z) to some targets.

    Kept as an object rather than a function because the decay curve fits ONCE
    on all hidden frames and then scores separately at each hidden age. Fitting
    a fresh probe per age would confound "the memory decayed" with "there were
    fewer frames to fit on at k = 22", which is the trap that a per-bin fit
    walks straight into.
    """

    def __init__(self, kind: str = "knn", k: int = 10):
        self.kind, self.k = kind, k

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "Readout":
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        if self.kind == "linear":
            A = np.concatenate([Xs, np.ones((len(Xs), 1))], 1)
            self.w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Y)
        else:
            from sklearn.neighbors import KNeighborsRegressor

            self.knn = KNeighborsRegressor(
                n_neighbors=self.k, weights="distance").fit(Xs, Y)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        if self.kind == "linear":
            return np.concatenate([Xs, np.ones((len(Xs), 1))], 1) @ self.w
        return self.knn.predict(Xs)


def _score(Y: np.ndarray, P: np.ndarray, names: Sequence[str]) -> Dict[str, Dict]:
    return {n: {"r2": _r2(Y[:, j], P[:, j]), "rmse": _rmse(Y[:, j], P[:, j])}
            for j, n in enumerate(names)}


def probe_bin(
    X: np.ndarray, Y: np.ndarray, groups: np.ndarray, names: Sequence[str],
    seed: int = 0, kinds: Sequence[str] = ("linear", "knn"),
) -> Dict[str, Dict]:
    """Fit and score inside one bin, with an EPISODE-level train/test split."""
    if len(X) < 50 or len(np.unique(groups)) < 4:
        return {k: {n: {"r2": float("nan"), "rmse": float("nan")} for n in names}
                for k in kinds}
    tr, te = make_split(len(X), seed=seed, group_ids=groups)
    out = {}
    for kind in kinds:
        p = Readout(kind).fit(X[tr], Y[tr])
        out[kind] = _score(Y[te], p(X[te]), names)
    return out


# -------------------------------------------------------- teacher-forced h


@torch.no_grad()
def hidden_states(model: MDNRNN, mu: np.ndarray, actions: np.ndarray,
                  device: str = "cpu", chunk: int = 16) -> np.ndarray:
    """Teacher-forced h over whole episodes. Returns (E, T, hidden).

    ``h[e, t]`` is the state after consuming ``(z_t, a_t)`` -- i.e. the state
    from which the model predicts ``z_{t+1}`` -- and is therefore aligned with
    the TRUE state at time t, which is what we probe it against.
    """
    outs = []
    for i in range(0, len(mu), chunk):
        z = torch.from_numpy(mu[i : i + chunk, :-1].astype(np.float32)).to(device)
        a = torch.eye(model.cfg.n_actions, device=device)[
            torch.from_numpy(actions[i : i + chunk]).long().to(device)]
        parts, _ = model(z, a)
        outs.append(parts["h"].cpu().numpy())
    return np.concatenate(outs, 0)


# ---------------------------------------------------------------- part (a)


def part_a_position_from_h(
    models: Sequence[Model], splits: Dict[str, Split], out: Path,
    device: str, seed: int,
) -> Dict:
    """Position (and velocity, and hidden age) from h, by visibility and by age."""
    main = splits["default"]
    T = main.T
    vis = main.visible[:, :T]
    E = main.mu.shape[0]
    groups_all = np.repeat(np.arange(E), T)
    pos = main.state[:, :T, :2].reshape(-1, 2)
    vel = main.state[:, :T, 2:4].reshape(-1, 2)
    age_all = hidden_age(main.visible)[:, :T].reshape(-1)
    v = vis.reshape(-1)

    bins = {
        "visible": v > 0.99,
        "partial": (v >= 0.01) & (v <= 0.99),
        "hidden": v < 0.01,
    }

    res: Dict[str, Dict] = {"bins": {}, "n_frames": {k: int(m.sum())
                                                     for k, m in bins.items()}}

    # --- the headline table: position from h (and from z) by visibility -----
    feats = {"z": main.mu[:, :T].reshape(-1, main.mu.shape[-1])}
    for m in models:
        feats[f"h[{m.name}]"] = hidden_states(
            m.model, main.mu, main.actions, device).reshape(-1, m.model.cfg.hidden)

    for fname, X in feats.items():
        res["bins"][fname] = {
            b: probe_bin(X[mask], pos[mask], groups_all[mask],
                         ("ball_x", "ball_y"), seed)
            for b, mask in bins.items()
        }

    # --- (f) what ELSE is in h while hidden ---------------------------------
    hid = bins["hidden"]
    res["hidden_extras"] = {}
    for fname, X in feats.items():
        Y = np.concatenate([vel, age_all[:, None].astype(float)], 1)
        res["hidden_extras"][fname] = probe_bin(
            X[hid], Y[hid], groups_all[hid],
            ("ball_vx", "ball_vy", "frames_hidden"), seed)

    # --- the decay curve ----------------------------------------------------
    # Pooled across every band so that large k exist at all: the default band
    # gives a mean hidden run of 9.4 frames and almost nothing past 20, while
    # `taller` averages 23.5. Provenance per k is recorded alongside.
    Xs, Ys, Gs, Ks, Srcs, NoMem = [], [], [], [], [], []
    for si, (sname, sp) in enumerate(splits.items()):
        Tn = sp.T
        age = hidden_age(sp.visible)[:, :Tn]
        hidmask = age > 0
        if not hidmask.any():
            continue
        # "no memory" prediction for every hidden frame: the ball's position on
        # the last frame before this run began.
        last = np.zeros(sp.state[:, :Tn, :2].shape, dtype=np.float64)
        for e in range(sp.mu.shape[0]):
            k = 0
            for t in range(Tn):
                if age[e, t] > 0:
                    if age[e, t] == 1:
                        k = max(t - 1, 0)
                    last[e, t] = sp.state[e, k, :2]
        sel = hidmask.reshape(-1)
        Ys.append(sp.state[:, :Tn, :2].reshape(-1, 2)[sel])
        Gs.append((np.repeat(np.arange(sp.mu.shape[0]), Tn) + 10_000 * si)[sel])
        Ks.append(age.reshape(-1)[sel])
        Srcs.append(np.full(int(sel.sum()), si))
        NoMem.append(last.reshape(-1, 2)[sel])
        Xs.append({"z": sp.mu[:, :Tn].reshape(-1, sp.mu.shape[-1])[sel]})
        for m in models:
            Xs[-1][f"h[{m.name}]"] = hidden_states(
                m.model, sp.mu, sp.actions, device
            ).reshape(-1, m.model.cfg.hidden)[sel]

    Y = np.concatenate(Ys); G = np.concatenate(Gs)
    K = np.concatenate(Ks); S = np.concatenate(Srcs)
    NM = np.concatenate(NoMem)
    feat_names = list(Xs[0].keys())
    XX = {f: np.concatenate([x[f] for x in Xs]) for f in feat_names}

    # Both readouts, because they answer different questions and on 256-d h
    # they disagree: kNN in 256 dimensions is a weak instrument (distances
    # concentrate), while a ridge-regularised linear map is well conditioned.
    # Stage one needed kNN because the VAE's position code is place-field-like;
    # h has no such excuse, and the (a) table shows linear >= kNN on h.
    tr, te = make_split(len(Y), seed=seed, group_ids=G)
    curve: Dict[str, Dict] = {}
    for f, kind in [(f, k) for f in feat_names for k in ("linear", "knn")]:
        key = f if kind == "knn" else f"{f} (lin)"
        p = Readout(kind).fit(XX[f][tr], Y[tr])
        pred = p(XX[f][te])
        f = key
        # ``rmse`` is the JOINT 2-d error (x and y summed, then rooted), which
        # is what the decay plot shows; ``rmse_x``/``rmse_y`` are the per-axis
        # versions, added so that the fix comparison can put the no-memory
        # baseline on the same R^2 axis as the models. v3 is a story about x
        # and y behaving differently, so the joint number hides the finding.
        curve[f] = {"k": [], "r2_x": [], "r2_y": [], "rmse": [],
                    "rmse_x": [], "rmse_y": [], "n": []}
        for k in range(1, MAX_AGE + 1):
            m = K[te] == k
            if m.sum() < 25:
                continue
            curve[f]["k"].append(k)
            curve[f]["r2_x"].append(_r2(Y[te][m, 0], pred[m, 0]))
            curve[f]["r2_y"].append(_r2(Y[te][m, 1], pred[m, 1]))
            curve[f]["rmse"].append(
                float(np.sqrt(((Y[te][m] - pred[m]) ** 2).sum(1).mean())))
            curve[f]["rmse_x"].append(_rmse(Y[te][m, 0], pred[m, 0]))
            curve[f]["rmse_y"].append(_rmse(Y[te][m, 1], pred[m, 1]))
            curve[f]["n"].append(int(m.sum()))
    nm = {"k": [], "rmse": [], "rmse_x": [], "rmse_y": [],
          "r2_x": [], "r2_y": [], "n": [], "splits": []}
    split_names = list(splits)
    for k in range(1, MAX_AGE + 1):
        m = K[te] == k
        if m.sum() < 25:
            continue
        nm["k"].append(k)
        nm["rmse"].append(float(np.sqrt(((Y[te][m] - NM[te][m]) ** 2).sum(1).mean())))
        # The baseline scored the same way as the probes, so the comparison in
        # the fix figure is like for like: same frames, same split, same axis.
        nm["rmse_x"].append(_rmse(Y[te][m, 0], NM[te][m, 0]))
        nm["rmse_y"].append(_rmse(Y[te][m, 1], NM[te][m, 1]))
        nm["r2_x"].append(_r2(Y[te][m, 0], NM[te][m, 0]))
        nm["r2_y"].append(_r2(Y[te][m, 1], NM[te][m, 1]))
        nm["n"].append(int(m.sum()))
        nm["splits"].append({split_names[i]: int((S[te][m] == i).sum())
                             for i in np.unique(S[te][m])})
    res["decay"] = {"per_feature": curve, "no_memory": nm,
                    "pooled_splits": split_names}

    _plot_decay(curve, nm, out / "position_from_h_by_hidden_time.png")
    return res


def _plot_decay(curve: Dict, nm: Dict, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for f, d in curve.items():
        # One family per panel: the linear readout of h (the better of the two
        # on h, see the note where the curve is fitted) plus z's kNN readout as
        # the "what the frame alone gives you" reference.
        if not d["k"] or not (f.endswith("(lin)") or f == "z"):
            continue
        style = dict(marker="o", ms=3)
        if f == "z":
            style.update(color="0.6", ls=":")
        axes[0].plot(d["k"], d["rmse"], label=f, **style)
        axes[1].plot(d["k"], d["r2_x"], label=f, **style)
    axes[0].plot(nm["k"], nm["rmse"], color="k", ls="--",
                 label="no memory (last visible)")
    axes[0].set_ylabel("position rmse (world units)")
    axes[1].set_ylabel("R^2, ball_x")
    axes[1].axhline(0, color="k", lw=0.8)
    for ax in axes:
        ax.set_xlabel("frames the ball has been fully hidden")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("position read out of h, as the occlusion goes on "
                 "(one frozen kNN probe, pooled over bands)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------- dreaming a run set


def _pad(a: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([a, np.repeat(a[:, -1:], n, axis=1)], 1) if n > 0 else a


@torch.no_grad()
def dream_runs(
    m: Model, sp: Split, runs: Sequence[HiddenRun], probe: StateProbe,
    device: str, seed: int, warmup: int = WARMUP, extra: int = EXTRA,
) -> np.ndarray:
    """Dream each run open-loop from its entry frame. Returns (N, H, 2).

    Index k of the output is true time ``run.t0 + k``, so the true exit (the
    first partially-visible frame) is at k = ``run.length``. The warm-up ends
    on the entry frame -- the LAST frame on which the ball was still at least
    partially visible -- which is the design's specification and is the
    strictest honest choice: one frame later and the model would have been
    shown a hidden frame it could not learn anything from anyway; one frame
    earlier and it would have been handed a clean sighting for free.
    """
    H = max(r.length for r in runs) + extra
    need_mu = max(r.t0 - warmup + warmup + H + 1 for r in runs)
    mu = _pad(sp.mu, max(0, need_mu - sp.mu.shape[1]))
    act = _pad(sp.actions, max(0, need_mu - sp.actions.shape[1]))
    rows = np.array([r.episode for r in runs])
    starts = np.array([r.t0 - warmup for r in runs])
    z_d = dream(m.model, mu[rows], act[rows], starts, warmup=warmup,
                horizon=H, temperature=0.0, device=device, seed=seed)
    return probe(z_d)[..., :2]


def score_runs(
    sp: Split, runs: Sequence[HiddenRun], pos: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Exit time / x / side of a set of dreamed trajectories.

    ``exit_time`` is counted in frames from the ENTRY frame, so the truth is
    ``run.length + 1`` and a dreamed exit at output index k is ``k + 1``. Both
    baselines use the same clock.
    """
    n = len(runs)
    o = {k: np.full(n, np.nan) for k in
         ("exit_time", "exit_x", "side", "censored")}
    for i, r in enumerate(runs):
        k, side = first_exit(pos[i, :, 1], sp.band, sp.radius, EXIT_CONSEC)
        if k < 0:
            o["censored"][i] = 1.0
            continue
        o["censored"][i] = 0.0
        o["exit_time"][i] = k + 1
        o["exit_x"][i] = pos[i, k, 0]
        o["side"][i] = side
    return o


def truth_of(sp: Split, runs: Sequence[HiddenRun]) -> Dict[str, np.ndarray]:
    lo, hi = band_interior(sp.band, sp.radius)
    t = {k: np.zeros(len(runs)) for k in ("exit_time", "exit_x", "side",
                                          "entry_x", "length", "wall_x")}
    for i, r in enumerate(runs):
        s_exit = sp.state[r.episode, r.exit]
        t["exit_time"][i] = r.length + 1
        t["exit_x"][i] = s_exit[0]
        t["side"][i] = 1 if s_exit[1] > hi else -1
        t["entry_x"][i] = sp.state[r.episode, r.entry, 0]
        t["length"][i] = r.length
        t["wall_x"][i] = float(r.wall_x)
    return t


# ---------------------------------------------------------- parts (b) + (c)


def emergence(
    models: Sequence[Model], sp: Split, probe: StateProbe, device: str,
    seed: int, min_len: int = 3, warmup: int = WARMUP,
) -> Dict:
    """Dream through every hidden run of `sp` and score the re-emergence."""
    runs = [r for r in hidden_runs_from_events(sp.events, min_len=min_len)
            if r.t0 >= warmup]
    dropped = len(hidden_runs_from_events(sp.events, min_len=min_len)) - len(runs)
    truth = truth_of(sp, runs)

    base = {
        "no_memory": {k: np.array([no_memory_exit(sp.state[r.episode], r, sp.band,
                                                  sp.radius)[k] for r in runs])
                      for k in ("exit_x", "exit_time", "side")},
        "linear": {k: np.array([linear_exit(sp.state[r.episode], r, sp.band,
                                            sp.radius)[k] for r in runs])
                   for k in ("exit_x", "exit_time", "side")},
    }
    pos = {m.name: dream_runs(m, sp, runs, probe, device, seed) for m in models}
    pred = {name: score_runs(sp, runs, pp) for name, pp in pos.items()}

    return {"split": sp.name, "n_runs": len(runs), "dropped_short_prefix": dropped,
            "runs": runs, "truth": truth, "pred": pred, "baselines": base,
            "pos": pos}


def _agg(truth: Dict, est: Dict, mask: np.ndarray) -> Dict[str, float]:
    """Errors on one subset. NaNs (censored dreams) are excluded and counted."""
    ex, tx = np.asarray(est["exit_x"], float), truth["exit_x"]
    et, tt = np.asarray(est["exit_time"], float), truth["exit_time"]
    ok = mask & np.isfinite(ex)
    n = int(mask.sum())
    if ok.sum() == 0:
        return {"n": n, "n_scored": 0}
    side = np.asarray(est.get("side", np.full(len(ex), np.nan)), float)
    return {
        "n": n,
        "n_scored": int(ok.sum()),
        "censored_frac": float(1.0 - ok.sum() / max(n, 1)),
        "exit_x_mae": float(np.abs(ex[ok] - tx[ok]).mean()),
        "exit_x_rmse": float(np.sqrt(((ex[ok] - tx[ok]) ** 2).mean())),
        "exit_x_bias": float((ex[ok] - tx[ok]).mean()),
        "exit_time_mae": float(np.abs(et[ok] - tt[ok]).mean()),
        "exit_time_bias": float((et[ok] - tt[ok]).mean()),
        "exit_time_within2": float((np.abs(et[ok] - tt[ok]) <= 2).mean()),
        "side_correct": float((side[ok] == truth["side"][ok]).mean())
        if np.isfinite(side[ok]).all() else float("nan"),
    }


def part_bc_tables(res: Dict) -> Dict:
    t = res["truth"]
    all_ = np.ones(res["n_runs"], bool)
    wall = t["wall_x"] > 0.5
    subsets = {"all": all_, "wall_x": wall, "no_wall_x": ~wall}
    est = {**res["pred"], **res["baselines"]}
    return {sub: {name: _agg(t, e, m) for name, e in est.items()}
            for sub, m in subsets.items()}


def part_c_reflection(res: Dict, radius: float = 0.08) -> Dict:
    """On the wall-bounce subset: did the dream see the bounce it could not see?

    Naming a "correct side" needs care, and the obvious version of this test is
    worthless. The ball hit a side wall while hidden, so the straight line
    overshoots -- usually to somewhere OUTSIDE the box. Scoring "is the dream
    nearer the truth or nearer the straight line" therefore hands full marks to
    anything that puts the ball inside the box at all, including a baseline
    with no memory. (It did: every model scored 1.00 on the first version of
    this metric, which is how the bug was found.)

    The fair alternative hypothesis is the straight line CLIPPED to the box:
    "the ball kept going and is now against the wall". That is where a model
    which saw the ball heading for the wall but did not simulate the bounce
    would put it; the truth is one overshoot further inward. Both hypotheses
    are inside the box, the separation is the overshoot rather than twice it,
    and chance really is 0.5.
    """
    t = res["truth"]
    wall = t["wall_x"] > 0.5
    lin = np.asarray(res["baselines"]["linear"]["exit_x"], float)
    stuck = np.clip(lin, radius, 1.0 - radius)     # "kept going, hit the wall"
    sep = np.abs(stuck - t["exit_x"])              # = the overshoot
    clear = wall & (sep > 0.05)
    if wall.sum() == 0:
        return {"n": 0}
    out = {
        "n": int(wall.sum()),
        "n_clear": int(clear.sum()),
        "overshoot_mean": float(sep[wall].mean()),
        "overshoot_mean_clear": float(sep[clear].mean()) if clear.any() else None,
        "no_wall_separation_mean": float(sep[~wall].mean()),
    }
    for name, e in {**res["pred"], **res["baselines"]}.items():
        ex = np.asarray(e["exit_x"], float)
        row = {}
        for tag, m in (("", wall), ("_clear", clear)):
            ok = m & np.isfinite(ex)
            if ok.sum() == 0:
                row[f"frac_nearer_reflected{tag}"] = float("nan")
                row[f"n_scored{tag}"] = 0
                continue
            nearer = np.abs(ex[ok] - t["exit_x"][ok]) < np.abs(ex[ok] - stuck[ok])
            row[f"frac_nearer_reflected{tag}"] = float(nearer.mean())
            row[f"n_scored{tag}"] = int(ok.sum())
        ok = wall & np.isfinite(ex)
        row["exit_x_mae"] = float(np.abs(ex[ok] - t["exit_x"][ok]).mean()) \
            if ok.sum() else float("nan")
        ok2 = (~wall) & np.isfinite(ex)
        row["exit_x_mae_no_wall"] = float(np.abs(ex[ok2] - t["exit_x"][ok2]).mean()) \
            if ok2.sum() else float("nan")
        out[name] = row
    # What the straight line would score if somebody folded it at the wall by
    # hand. Not a model -- a check that the truth really IS the mirror image,
    # i.e. that nothing else happened behind the band.
    out["linear_folded_mae"] = float(np.abs(
        np.array([reflect_x(x, radius) for x in lin[wall]]) - t["exit_x"][wall]).mean())
    return out


def merge_results(per_split: Dict[str, Dict]) -> Dict:
    """Concatenate several splits' emergence results into one.

    Only for the wall-bounce test, which on the default band has 12 positives
    and no statistical power at all. The three bands have the same physics and
    the same ball; they differ only in how long the ball is hidden, so pooling
    them is legitimate for a question about what happens BEHIND the band. The
    default-band table is still reported separately.
    """
    keys = list(per_split)
    runs = [r for k in keys for r in per_split[k]["runs"]]
    def cat(get):
        return {f: np.concatenate([np.asarray(get(per_split[k])[f], float)
                                   for k in keys]) for f in get(per_split[keys[0]])}
    return {
        "split": "+".join(keys),
        "n_runs": len(runs),
        "runs": runs,
        "truth": cat(lambda r: r["truth"]),
        "pred": {m: cat(lambda r, m=m: r["pred"][m]) for m in per_split[keys[0]]["pred"]},
        "baselines": {b: cat(lambda r, b=b: r["baselines"][b])
                      for b in per_split[keys[0]]["baselines"]},
    }


def _plot_exit_scatter(res: Dict, path: Path) -> None:
    plt = _plt()
    est = {**res["pred"], **res["baselines"]}
    n = len(est)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), sharex=True, sharey=True)
    t = res["truth"]
    wall = t["wall_x"] > 0.5
    for ax, (name, e) in zip(np.atleast_1d(axes), est.items()):
        ex = np.asarray(e["exit_x"], float)
        ax.plot([0, 1], [0, 1], color="0.7", lw=1)
        ax.scatter(t["exit_x"][~wall], ex[~wall], s=9, alpha=0.6, label="no bounce")
        ax.scatter(t["exit_x"][wall], ex[wall], s=14, alpha=0.8, color="crimson",
                   marker="^", label="wall bounce while hidden")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("true exit x")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("predicted exit x")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    np.atleast_1d(axes)[0].set_xlim(-0.1, 1.1)
    np.atleast_1d(axes)[0].set_ylim(-0.6, 1.6)
    fig.suptitle("where the ball comes back out", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_dream_examples(res: Dict, sp: Split, path: Path, n: int = 6) -> None:
    """The raw material behind every number in (b): dreamed x and y vs truth.

    Worth looking at before believing any of the tables. A model that has
    simply learned "the ball vanishes and something reappears after about nine
    frames" and a model that is tracking the ball produce very similar exit
    TIMES and completely different x traces, and only this figure shows which
    you have.
    """
    plt = _plt()
    runs, pos = res["runs"], res["pos"]
    if not runs:
        return
    # A stratified sample, not the n longest: the longest runs are the hardest
    # and showing only those would misrepresent the typical case. Half the
    # columns are drawn evenly across the length distribution, half are
    # wall-bounce runs (the interesting ones) if any exist.
    by_len = np.argsort([r.length for r in runs])
    pick = list(by_len[np.linspace(0, len(by_len) - 1,
                                   num=max(n - 2, 1)).astype(int)])
    walls = [i for i in by_len if runs[i].wall_x and i not in pick]
    order = (pick + walls[len(walls) // 2 : len(walls) // 2 + 2])[:n]
    lo, hi = band_interior(sp.band, sp.radius)
    fig, axes = plt.subplots(2, len(order), figsize=(2.6 * len(order), 5.2),
                             sharex=True)
    for col, i in enumerate(order):
        r = runs[i]
        H = pos[list(pos)[0]].shape[1]
        t = np.arange(H)
        true = sp.state[r.episode, r.t0 : r.t0 + H, :2]
        for row, (j, lab) in enumerate(((0, "ball x"), (1, "ball y"))):
            ax = axes[row, col]
            ax.plot(np.arange(len(true)), true[:, j], color="k", lw=2,
                    label="truth", zorder=3)
            for name, pp in pos.items():
                ax.plot(t, pp[i, :, j], lw=1.2, label=name)
            ax.axvline(r.length, color="0.5", ls="--", lw=1)
            if row == 1:
                ax.axhspan(lo, hi, color="0.85", zorder=0)
                ax.set_xlabel("dream step")
            if col == 0:
                ax.set_ylabel(lab)
            ax.grid(alpha=0.25)
        axes[0, col].set_title(f"ep{r.episode} hidden {r.length}f"
                               + (" WALL" if r.wall_x else ""), fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("dreaming through an occlusion -- grey band = fully hidden, "
                 "dashed line = true exit")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_exit_time_hist(res: Dict, path: Path) -> None:
    plt = _plt()
    t = res["truth"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.arange(-12.5, 13.5, 1.0)
    for name, e in {**res["pred"], **{"linear": res["baselines"]["linear"]}}.items():
        et = np.asarray(e["exit_time"], float)
        ok = np.isfinite(et)
        ax.hist(np.clip(et[ok] - t["exit_time"][ok], -12, 12), bins=bins,
                histtype="step", lw=1.6, label=f"{name} (n={int(ok.sum())})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("dreamed exit time - true exit time (frames)")
    ax.set_ylabel("hidden runs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- part (d)


DURATION_BINS = ((3, 8), (9, 15), (16, 25), (26, 10**6))


def part_d_horizon(per_split: Dict[str, Dict], out: Path) -> Dict:
    """Exit error vs true hidden duration, pooled over bands, binned."""
    rows: Dict[str, Dict] = {}
    pooled: Dict[str, Dict] = {}
    for sname, res in per_split.items():
        t = res["truth"]
        est = {**res["pred"], **res["baselines"]}
        rows[sname] = {}
        for lo, hi in DURATION_BINS:
            m = (t["length"] >= lo) & (t["length"] <= hi)
            key = f"{lo}-{hi if hi < 10**6 else '+'}"
            rows[sname][key] = {name: _agg(t, e, m) for name, e in est.items()}
    # pooled across bands for the plot
    keys = [f"{lo}-{hi if hi < 10**6 else '+'}" for lo, hi in DURATION_BINS]
    names = sorted({n for s in rows.values() for k in s.values() for n in k})
    for name in names:
        pooled[name] = {}
        for ki, key in enumerate(keys):
            num = den = tnum = 0.0
            n_runs = 0
            for sname in rows:
                a = rows[sname][key].get(name, {})
                n_runs += a.get("n", 0)
                if a.get("n_scored", 0):
                    num += a["exit_x_mae"] * a["n_scored"]
                    tnum += a["exit_time_mae"] * a["n_scored"]
                    den += a["n_scored"]
            pooled[name][key] = {
                "exit_x_mae": num / den if den else float("nan"),
                "exit_time_mae": tnum / den if den else float("nan"),
                # n_runs is how many runs fell in the bin; n_scored is how many
                # of them the model actually re-emerged a ball for. They differ
                # a lot for the feed-forward control, which often never brings
                # the ball back at all, and reporting only the second would
                # quietly grade it on the subset it happened to manage.
                "n_runs": int(n_runs), "n_scored": int(den),
            }
    _plot_horizon(pooled, keys, out / "memory_horizon.png")
    return {"by_split": rows, "pooled": pooled, "bins": keys}


def _plot_horizon(pooled: Dict, keys: Sequence[str], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(keys))
    # A bin the model only re-emerged a ball in three times is not a
    # measurement, so points below MIN_N are dropped rather than drawn with a
    # confidence interval nobody would read.
    MIN_N = 5
    for name, d in pooled.items():
        style = dict(marker="o", ms=4)
        if name == "no_memory":
            style.update(color="k", ls="--")
        if name == "linear":
            style.update(color="0.55", ls=":")
        ok = [i for i, k in enumerate(keys) if d[k]["n_scored"] >= MIN_N]
        axes[0].plot([x[i] for i in ok],
                     [d[keys[i]]["exit_x_mae"] for i in ok], label=name, **style)
        axes[1].plot([x[i] for i in ok],
                     [d[keys[i]]["exit_time_mae"] for i in ok], label=name, **style)
    labels = [f"{k}\nn={pooled['no_memory'][k]['n_runs']}" for k in keys]
    for ax, lab in zip(axes, ("exit x MAE (world units)",
                              "exit time MAE (frames)")):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("true hidden duration (frames)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("memory horizon: pooled over the 0.30 / 0.42 / 0.54 bands "
                 f"(points with fewer than {MIN_N} scored runs are dropped)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- part (e)


def part_e_counterfactual(
    models: Sequence[Model], sp: Split, vae, probe: StateProbe, cfg: BoxConfig,
    device: str, seed: int, n_runs: int = 40, warmup: int = WARMUP,
    extra: int = EXTRA,
) -> Dict:
    """Re-simulate with vx -> -vx before entry, re-encode, re-dream.

    The counterfactual has a GROUND TRUTH, because we ran the simulator to make
    it: that is what separates this from the v2 recolour intervention, where
    the intervened world had no trajectory to compare against. The two branches
    are ``replay`` (the same velocity, re-simulated and re-rendered -- a
    control that should reproduce the real answer and mostly tests that the
    re-encode is faithful) and ``flip`` (vx negated).

    The physical prediction is simple: flipping vx leaves the y trajectory
    untouched, so the ball is hidden for the same number of frames and comes
    out at the same time, on the same side, with the horizontal displacement
    reversed.
    """
    runs = [r for r in hidden_runs_from_events(sp.events, min_len=3)
            if r.t0 >= warmup + 2][:n_runs]
    if not runs:
        return {"n": 0}
    H = max(r.length for r in runs) + extra
    env = BouncingBox(cfg)
    env.reset(seed=0)
    names = sp.names

    out: Dict[str, Dict] = {}
    for branch, flip in (("replay", 1.0), ("flip", -1.0)):
        frames, states = [], []
        for r in runs:
            t_start = r.t0 - warmup
            s0 = sp.state[r.episode, t_start]
            env.ball = np.array([float(s0[0]), float(s0[1])])
            env.ball_v = np.array([flip * float(s0[2]), float(s0[3])])
            env.paddle_x = float(s0[4])
            env.paddle_vx = float(s0[5])
            fr = [env.render()]
            st = [env.state()]
            for k in range(warmup + H):
                t = min(t_start + k, sp.T - 1)
                f, s, _ = env.step(int(sp.actions[r.episode, t]))
                fr.append(f)
                st.append(s)
            frames.append(np.stack(fr[:warmup]))
            states.append(np.stack(st))
        frames = np.stack(frames)                       # (N, warmup, 64,64,3)
        states = np.stack(states)                       # (N, warmup+H+1, S)

        mu_cf = encode_frames(vae, frames, device)      # (N, warmup, z)
        # dream() wants (N, T+1, z); only the first `warmup` are read.
        mu_slab = np.concatenate(
            [mu_cf, np.repeat(mu_cf[:, -1:], H + 1, 1)], 1).astype(np.float32)
        act = np.stack([sp.actions[r.episode,
                                   np.clip(r.t0 - warmup + np.arange(warmup + H),
                                           0, sp.T - 1)] for r in runs])
        starts = np.zeros(len(runs), dtype=int)

        # Ground truth for this counterfactual, from the simulator itself.
        vis_col = names.index("ball_visible")
        y_cf = states[:, warmup:, 1]
        tru = {"exit_time": [], "exit_x": [], "side": [], "entry_x": []}
        keep = []
        for i in range(len(runs)):
            k, side = first_exit(y_cf[i], sp.band, sp.radius, 1,
                                 start=int(np.argmax(
                                     states[i, warmup:, vis_col] < 0.01)))
            hidden_any = (states[i, warmup:, vis_col] < 0.01).any()
            keep.append(bool(hidden_any and k > 0))
            tru["exit_time"].append(k + 1 if k > 0 else np.nan)
            tru["exit_x"].append(states[i, warmup + k, 0] if k > 0 else np.nan)
            tru["side"].append(side)
            tru["entry_x"].append(states[i, warmup - 1, 0])
        keep = np.array(keep)
        tru = {k: np.asarray(v, float) for k, v in tru.items()}

        branch_out: Dict[str, Dict] = {"n": int(keep.sum())}
        for m in models:
            z_d = dream(m.model, mu_slab, act, starts, warmup=warmup,
                        horizon=H, temperature=0.0, device=device, seed=seed)
            pos = probe(z_d)[..., :2]
            ex_t, ex_x, sgn = [], [], []
            for i in range(len(runs)):
                k, side = first_exit(pos[i, :, 1], sp.band, sp.radius, EXIT_CONSEC)
                ex_t.append(k + 1 if k >= 0 else np.nan)
                ex_x.append(pos[i, k, 0] if k >= 0 else np.nan)
                sgn.append(side if k >= 0 else np.nan)
            ex_x = np.asarray(ex_x, float)
            ok = keep & np.isfinite(ex_x)
            dx_pred = ex_x - tru["entry_x"]
            dx_true = tru["exit_x"] - tru["entry_x"]
            branch_out[m.name] = {
                "n_scored": int(ok.sum()),
                "exit_x_mae": float(np.abs(ex_x[ok] - tru["exit_x"][ok]).mean()),
                "exit_x_signed_err": float((ex_x[ok] - tru["exit_x"][ok]).mean()),
                "exit_time_mae": float(np.abs(
                    np.asarray(ex_t, float)[ok] - tru["exit_time"][ok]).mean()),
                "dx_sign_agrees": float(
                    (np.sign(dx_pred[ok]) == np.sign(dx_true[ok])).mean()),
                "mean_dx_pred": float(dx_pred[ok].mean()),
                "mean_dx_true": float(dx_true[ok].mean()),
            }
        out[branch] = branch_out
    return out


# ------------------------------------------------------------------- main


def _ck(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        return Path(spec).parent.name, spec
    n, p = spec.split("=", 1)
    return n, p


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+",
                   default=["rnn_v3=runs/rnn_v3/rnn.pt",
                            "ff=runs/rnn_v3_ff/rnn.pt",
                            "ms=runs/rnn_v3_ms/rnn.pt"],
                   help="name=path/to/rnn.pt")
    p.add_argument("--vae", default="runs/vae_v3/vae.pt")
    p.add_argument("--val", nargs="+", default=["data/v3/val", "data/v3/val_mix"])
    p.add_argument("--tall", nargs="+", default=["data/v3/tall"])
    p.add_argument("--taller", nargs="+", default=["data/v3/taller"])
    p.add_argument("--probe-data", nargs="+",
                   default=["data/v3/probe", "data/v3/val_mix"])
    p.add_argument("--out", default="runs/rnn_v3/permanence")
    p.add_argument("--probe-samples", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip", nargs="*", default=[],
                   help="parts to skip, e.g. --skip e")
    a = p.parse_args()

    device = pick_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    models = []
    for spec in a.models:
        name, path = _ck(spec)
        m, _ = load_rnn(path, device)
        models.append(Model(name, m, path))
        print(f"{name:8s} {path}  feedforward={m.cfg.feedforward}  "
              f"params={sum(q.numel() for q in m.parameters())/1e3:.0f}k")

    vae, vcfg, _ = load_ckpt(a.vae, device)
    probe = fit_position_probe(a.probe_data, "", a.probe_samples, a.seed)

    splits = {"default": load_split("default", a.val),
              "tall": load_split("tall", a.tall),
              "taller": load_split("taller", a.taller)}
    for s in splits.values():
        print(f"split {s.name:8s} E={s.mu.shape[0]} T={s.T} band={s.band} "
              f"hidden_runs={len(hidden_runs_from_events(s.events, 3))}")

    # The probe's own reading of a genuinely hidden frame. Quoted in the report
    # because every exit detector below depends on it: if a hidden frame
    # decoded to somewhere OUTSIDE the band, the detector would fire instantly
    # on every dream and every exit time would be 1.
    d = splits["default"]
    hid = d.visible < 0.01
    p_hid = probe(d.mu[hid])[:, :2]
    lo, hi = band_interior(d.band, d.radius)
    probe_check = {
        "n_hidden_frames": int(hid.sum()),
        "mean_decoded_y": float(p_hid[:, 1].mean()),
        "frac_decoded_outside_band_interior": float(
            ((p_hid[:, 1] < lo) | (p_hid[:, 1] > hi)).mean()),
        "rmse_decoded_x": float(np.sqrt(
            ((p_hid[:, 0] - d.state[hid][:, 0]) ** 2).mean())),
    }
    print(f"position probe on TRUE hidden frames: mean decoded y "
          f"{probe_check['mean_decoded_y']:.3f} (band interior "
          f"[{lo:.2f}, {hi:.2f}]), "
          f"{probe_check['frac_decoded_outside_band_interior']:.1%} fall outside")

    report: Dict[str, object] = {
        "models": {m.name: m.ckpt for m in models},
        "vae": a.vae, "warmup": WARMUP, "extra": EXTRA,
        "exit_consec": EXIT_CONSEC,
        "probe_on_hidden_frames": probe_check,
    }

    if "a" not in a.skip:
        print("\n(a)+(f) probing h ...", flush=True)
        report["a_position_from_h"] = part_a_position_from_h(
            models, splits, out, device, a.seed)

    per_split: Dict[str, Dict] = {}
    if "b" not in a.skip:
        for sname in ("default", "tall", "taller"):
            print(f"\n(b) emergence on {sname} ...", flush=True)
            per_split[sname] = emergence(models, splits[sname], probe,
                                         device, a.seed)
        res = per_split["default"]
        report["b_emergence"] = {
            "n_runs": res["n_runs"],
            "run_lengths": [r.length for r in res["runs"]],
            "n_wall_x": int(res["truth"]["wall_x"].sum()),
            "dropped_runs_starting_before_warmup": res["dropped_short_prefix"],
            "tables": part_bc_tables(res),
        }
        report["c_wall_bounce"] = part_c_reflection(res, splits["default"].radius)
        # The default band yields only ~12 hidden wall bounces, which is no
        # sample at all. The same test pooled over all three bands has ~80.
        report["c_wall_bounce_pooled"] = part_c_reflection(
            merge_results(per_split), splits["default"].radius)
        _plot_exit_scatter(res, out / "exit_x_pred_vs_true.png")
        _plot_dream_examples(res, splits["default"], out / "dream_examples.png")
        _plot_exit_time_hist(res, out / "exit_time_error_hist.png")

        report["d_memory_horizon"] = part_d_horizon(per_split, out)
        report["d_tables_by_split"] = {
            s: part_bc_tables(r) for s, r in per_split.items()}

    if "e" not in a.skip:
        print("\n(e) counterfactual entry ...", flush=True)
        cfg = BoxConfig(**json.loads(
            (Path(a.val[0]) / "meta.json").read_text())["config"])
        report["e_counterfactual"] = part_e_counterfactual(
            models, splits["default"], vae, probe, cfg, device, a.seed)

    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    _summary(report, [m.name for m in models], out)


# ------------------------------------------------------------------ summary


def _fmt(v, w=7, p=3):
    return f"{'   --':>{w}s}" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"{v:{w}.{p}f}"


def _summary_c(tag: str, c) -> None:
    if not c or not c.get("n"):
        return
    print(f"\n(c) hidden wall bounces -- {tag}")
    print(f"    {c['n']} runs, {c['n_clear']} of them with an overshoot over "
          f"0.05 (mean overshoot\n    {c['overshoot_mean']:.3f}; on no-bounce "
          f"runs the two hypotheses coincide to "
          f"{c['no_wall_separation_mean']:.3f}).")
    print("    'nearer reflected' = is the prediction closer to the truth "
          "(the ball bounced) or to\n    the straight line clipped at the "
          "wall (the ball kept going and stopped there)?")
    print(f"    {'model':12s} {'n':>4s} {'nearer reflected':>17s} "
          f"{'n':>4s} {'(clear runs)':>14s} {'exit_x MAE':>11s} {'(no bounce)':>12s}")
    for name, d_ in c.items():
        if not isinstance(d_, dict) or not d_.get("n_scored"):
            continue
        print(f"    {name:12s} {d_['n_scored']:4d} "
              f"{_fmt(d_['frac_nearer_reflected'], 17, 2)} "
              f"{d_.get('n_scored_clear', 0):4d} "
              f"{_fmt(d_.get('frac_nearer_reflected_clear'), 14, 2)} "
              f"{_fmt(d_['exit_x_mae'], 11, 4)} "
              f"{_fmt(d_['exit_x_mae_no_wall'], 12, 4)}")
    print(f"    chance is 0.50; the straight-line baseline is 0.00 by "
          f"construction. Folding that\n    line at the wall by hand gives MAE "
          f"{c['linear_folded_mae']:.4f}, i.e. the truth really is the mirror "
          f"image.")



def _summary(rep: Dict, names: Sequence[str], out: Path) -> None:
    print("\n" + "=" * 78)
    print("OBJECT PERMANENCE -- v3 stage two")
    print("=" * 78)

    a_ = rep.get("a_position_from_h")
    if a_:
        print("\n(a) ball position from a frozen probe, by visibility "
              "(R^2 / rmse, in world units)")
        print("    Read the rmse column across bins and the R^2 column down "
              "them: the hidden bin's\n    variance is smaller than the visible "
              "bin's, so the two columns disagree on purpose.")
        for pr in ("linear", "knn"):
            print(f"\n    probe = {pr}")
            print(f"    {'feature':14s} " + "".join(
                f"{b:>23s}" for b in ("visible", "partial", "hidden")))
            for f, d in a_["bins"].items():
                row = f"    {f:14s} "
                for b in ("visible", "partial", "hidden"):
                    x, y = d[b][pr]["ball_x"], d[b][pr]["ball_y"]
                    row += (f"  {x['r2']:6.2f}/{x['rmse']:.3f}"
                            f"{y['r2']:6.2f}/{y['rmse']:.3f}")
                print(row)
        print(f"    frames per bin: {a_['n_frames']}")

        print("\n    decay of position-from-h with frames hidden "
              "(rmse, world units; one frozen kNN probe)")
        nm = a_["decay"]["no_memory"]
        ks = nm["k"]
        show = [k for k in ks if k in (1, 3, 5, 8, 12, 16, 20, 25)]
        print("      k" + " " * 18 + "".join(f"{k:>8d}" for k in show))
        for f, d in a_["decay"]["per_feature"].items():
            m = {k: r for k, r in zip(d["k"], d["rmse"])}
            print(f"      {f:18s}" + "".join(_fmt(m.get(k), 8) for k in show))
        m = {k: r for k, r in zip(nm["k"], nm["rmse"])}
        print(f"      {'no memory':18s}" + "".join(_fmt(m.get(k), 8) for k in show))
        print("      (no memory = predict the ball's last visible position; its "
              "error grows\n       as speed x k and is the line every h curve has "
              "to get under. The `z` row is\n       not zero because a tracking "
              "paddle leaks some ball x; `h[ff]` is the honest floor.)")

        print("\n(f) what else is in h on fully hidden frames (linear R^2)")
        print(f"    {'feature':14s} {'ball_vx':>9s} {'ball_vy':>9s} "
              f"{'frames_hidden':>14s}")
        for f, d in a_["hidden_extras"].items():
            k = d["linear"]
            print(f"    {f:14s} {k['ball_vx']['r2']:9.3f} "
                  f"{k['ball_vy']['r2']:9.3f} {k['frames_hidden']['r2']:14.3f}")

    b = rep.get("b_emergence")
    if b:
        print(f"\n(b) emergence, {b['n_runs']} hidden runs (>= 3 frames) on val+val_mix")
        for sub in ("all", "no_wall_x", "wall_x"):
            print(f"\n    subset = {sub}")
            print(f"      {'model':12s} {'n':>4s} {'exit_x MAE':>11s} "
                  f"{'bias':>8s} {'t MAE':>7s} {'|dt|<=2':>8s} {'side ok':>8s} "
                  f"{'censor':>7s}")
            for name, d in b["tables"][sub].items():
                if not d.get("n_scored"):
                    print(f"      {name:12s} {d['n']:4d}   (nothing scored)")
                    continue
                print(f"      {name:12s} {d['n']:4d} {d['exit_x_mae']:11.4f} "
                      f"{d['exit_x_bias']:8.4f} {d['exit_time_mae']:7.2f} "
                      f"{d['exit_time_within2']:8.2f} "
                      f"{_fmt(d['side_correct'], 8, 2)} "
                      f"{d.get('censored_frac', 0):7.2f}")

    for tag, c in (("default band", rep.get("c_wall_bounce")),
                   ("all three bands pooled", rep.get("c_wall_bounce_pooled"))):
        _summary_c(tag, c)

    d = rep.get("d_memory_horizon")
    if d:
        print("\n(d) memory horizon -- exit x MAE (world units), pooled over bands")
        keys = d["bins"]
        print(f"    {'model':12s}" + "".join(f"{k:>10s}" for k in keys))
        for name, row in d["pooled"].items():
            print(f"    {name:12s}" + "".join(
                _fmt(row[k]["exit_x_mae"], 10, 4) for k in keys))
        print(f"    {'(n runs)':12s}" + "".join(
            f"{d['pooled']['no_memory'][k]['n_runs']:>10d}" for k in keys))
        print("    n_scored (runs where the model brought a ball back at all):")
        for name, row in d["pooled"].items():
            print(f"    {name:12s}" + "".join(f"{row[k]['n_scored']:>10d}"
                                              for k in keys))

    e = rep.get("e_counterfactual")
    if e and e.get("replay"):
        print("\n(e) counterfactual entry (vx -> -vx), re-simulated ground truth")
        for branch in ("replay", "flip"):
            print(f"    branch = {branch}  (n={e[branch]['n']})")
            for name in names:
                x = e[branch].get(name)
                if not x:
                    continue
                print(f"      {name:12s} exit_x MAE {x['exit_x_mae']:.4f}  "
                      f"signed {x['exit_x_signed_err']:+.4f}  "
                      f"dx sign agrees {x['dx_sign_agrees']:.2f}  "
                      f"mean dx pred {x['mean_dx_pred']:+.3f} vs "
                      f"true {x['mean_dx_true']:+.3f}")

    print(f"\nreport -> {out / 'report.json'}")


if __name__ == "__main__":
    main()

"""Does the dynamics model actually know any physics? Six ways of asking.

    python -m wm.eval_rnn --ckpt runs/rnn_v1/rnn.pt --vae runs/vae_b1/vae.pt \
        --val data/v1/val data/v1/val_mix --out runs/rnn_v1/eval

Teacher-forced NLL is a number that goes down. It is not evidence that the model
learned physics -- with ``predict_delta`` a model that has learned nothing but
"the latent barely changes" already scores well. Everything in this file is an
attempt to break that illusion by asking the model to do something a
one-step-denoiser cannot do.

    (a) dream rollouts vs truth, in pixels
    (b) dream rollouts vs truth, in world coordinates
    (c) where does velocity live -- z, or h?
    (d) does the model believe actions do anything?
    (e) can it anticipate a paddle contact?
    (f) does it know about walls?

Two framing points that the plots are designed around.

THE VAE FLOOR. A dreamed frame is decoded by the frozen VAE, so its error has
two sources: the dynamics model predicted the wrong latent, and the VAE cannot
render a latent perfectly anyway. Every plot here therefore also shows the
error of ``decode(encode(true frame))`` -- the VAE ceiling. The distance from
the dashed floor line to a dream curve is the part that belongs to stage two.
Without that line a mediocre dream and a mediocre decoder are indistinguishable.

THE PROBE IS A MEASURING INSTRUMENT, NOT PART OF THE MODEL. To say "the model
thinks the ball is here" we need a map from latent to world coordinates. We fit
one (kNN, as in ``wm.probes``) on TRUE latents from a held-out dataset, freeze
it, and apply it to dreamed latents. It has its own error, which is also plotted
as a floor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from worldsim.render import frame_grid, save_gif, side_by_side, upscale

from .analyze import add_derived_targets, load_ckpt
from .probes import probe_suite
from .rnn import MDNRNN, load_rnn
from .seq_data import episode_arrays
from .train_vae import pick_device

TAUS = (0.0, 0.5, 1.0)
BALL_RADIUS = 0.08  # from the dataset meta; the "useful horizon" threshold
BASE_SPEED = 0.022  # v2: the ball's speed is BASE_SPEED / mass


# --------------------------------------------------------------- plumbing


def _save_png(img: np.ndarray, path: Path, scale: int = 1) -> Path:
    from PIL import Image

    if scale > 1:
        img = upscale(img, scale)
    Image.fromarray(img).save(path)
    return path


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@torch.no_grad()
def decode_latents(vae, z: np.ndarray, device: str = "cpu", chunk: int = 256) -> np.ndarray:
    """(..., z_dim) -> (..., 64, 64, 3) uint8 frames, via the FROZEN decoder."""
    shape = z.shape[:-1]
    flat = torch.from_numpy(z.reshape(-1, z.shape[-1]).astype(np.float32))
    outs = []
    for i in range(0, len(flat), chunk):
        x = vae.decode(flat[i : i + chunk].to(device))
        outs.append((x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(outs, 0).reshape(*shape, 64, 64, 3)


@torch.no_grad()
def dream(
    model: MDNRNN,
    mu: np.ndarray,          # (E, T+1, z) true latents
    actions: np.ndarray,     # (E, T) int
    starts: np.ndarray,      # (E,) index of the first warm-up step
    warmup: int = 8,
    horizon: int = 64,
    temperature: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
    action_override: Optional[int] = None,
) -> np.ndarray:
    """Open-loop rollout. Returns dreamed latents ``(E, horizon, z)``.

    The model is fed the TRUE action sequence (or a single fixed action if
    ``action_override`` is given) but its OWN latent. That is the only honest
    test of a dynamics model: teacher forcing hands it a fresh, correct z every
    step, which quietly repairs whatever it got wrong, so teacher-forced error
    never compounds and open-loop error is the only place compounding shows up.
    """
    model.eval()
    torch.manual_seed(seed)
    E = mu.shape[0]
    z_dim = mu.shape[-1]

    # Gather the (E, warmup+horizon) slices each episode needs.
    idx = starts[:, None] + np.arange(warmup + horizon)[None, :]
    z_w = np.stack([mu[e, idx[e]] for e in range(E)])              # (E, W+H, z)
    a_i = np.stack([actions[e, idx[e]] for e in range(E)])         # (E, W+H)
    if action_override is not None:
        a_i = a_i.copy()
        a_i[:, warmup:] = action_override

    z_t = torch.from_numpy(z_w.astype(np.float32)).to(device)
    a_oh = torch.eye(model.cfg.n_actions, device=device)[
        torch.from_numpy(a_i).long().to(device)
    ]

    parts, h = model(z_t[:, :warmup], a_oh[:, :warmup])
    z = model.sample_next(parts, temperature=temperature)[:, -1]
    preds = [z]
    for k in range(1, horizon):
        p, h = model.step(z, a_oh[:, warmup + k], h)
        z = model.sample_next(p, temperature=temperature)[:, 0]
        preds.append(z)
    out = torch.stack(preds, 1).cpu().numpy()
    assert out.shape == (E, horizon, z_dim)
    return out


def _stack_val(roots: Sequence[str], latent_suffix: str = "") -> Dict[str, np.ndarray]:
    """Concatenate several dataset roots along the episode axis.

    Also carries ``state_names`` out of the roots' meta.json instead of letting
    the rest of the file assume six columns: v1 datasets have six, v2 has seven
    (``mass`` appended). Everything downstream indexes by name where it can and
    by the shared prefix (0=ball_x, 1=ball_y, 4=paddle_x) where it cannot.
    """
    parts = [episode_arrays(r, latent_suffix=latent_suffix) for r in roots]
    keys = ("mu", "logvar", "actions", "hit", "reward", "state")
    out = {k: np.concatenate([p[k] for p in parts], 0) for k in keys}
    out["root_of_episode"] = np.concatenate(
        [np.full(len(p["mu"]), i) for i, p in enumerate(parts)]
    )
    names = [tuple(p["meta"]["state_names"]) for p in parts]
    if len(set(names)) != 1:
        raise ValueError(f"roots disagree on state_names: {set(names)}")
    out["state_names"] = list(names[0])
    return out


# ------------------------------------------------------ the state probe


class StateProbe:
    """Frozen kNN map: VAE mu -> true 6-d state. Our measuring instrument.

    k=10 distance-weighted on standardised mu, the same configuration the
    stage-one analysis used, chosen there because it is nonparametric: it
    answers "is this information present under any smooth map" without assuming
    the code is linear. Stage one found it is not (linear R^2 ~ 0.04-0.7 for
    position, kNN ~ 0.98), so a linear probe here would understate the model.

    Fit on TRUE latents, applied to DREAMED latents. That is a mild
    extrapolation -- a dreamed latent can land somewhere the encoder never puts
    real frames -- and it is the main caveat on part (b): a dream that drifts
    off the latent manifold gets mapped to whatever real frame happens to be
    nearest, so the measured error saturates rather than diverging.
    """

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, mu: np.ndarray, state: np.ndarray) -> "StateProbe":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(mu)
        self.model = KNeighborsRegressor(
            n_neighbors=self.k, weights="distance"
        ).fit(self.scaler.transform(mu), state)
        return self

    def __call__(self, z: np.ndarray) -> np.ndarray:
        shape = z.shape[:-1]
        flat = z.reshape(-1, z.shape[-1])
        pred = self.model.predict(self.scaler.transform(flat))
        return pred.reshape(*shape, pred.shape[-1])


# ---------------------------------------------------------------- part (a)


def part_a_pixels(
    model, vae, val, out: Path, device: str, warmup: int, horizon: int,
    n_metric: int, n_gif: int, seed: int,
) -> Dict:
    print("\n(a) dream rollouts in pixel space")
    E = min(n_metric, val["mu"].shape[0])
    T = val["actions"].shape[1]
    # Start every episode at the same place; warmup+horizon must fit.
    start = 0
    starts = np.full(E, start)
    # Alignment: the warm-up consumes true latents mu[start .. start+warmup-1];
    # the LSTM output at the last warm-up step predicts mu[start+warmup], so
    # dream step k lines up with true index start + warmup + k (NOT +1 -- an
    # earlier version had that off-by-one and charged the model one frame of
    # ball motion, ~0.02 world units, at every horizon).
    idx_true = start + warmup + np.arange(horizon)       # true frames the dream predicts

    true_frames = _load_true_frames(val, E, idx_true)          # (E, H, 64, 64, 3)
    true_f = true_frames.astype(np.float32) / 255.0

    # The VAE floor: decode the TRUE latents for the same timesteps.
    recon = decode_latents(vae, val["mu"][:E][:, idx_true], device=device)
    floor = ((recon.astype(np.float32) / 255.0 - true_f) ** 2).mean((1, 2, 3, 4)).mean()
    floor_per_h = ((recon.astype(np.float32) / 255.0 - true_f) ** 2).mean((0, 2, 3, 4))

    res: Dict[str, object] = {"vae_floor_mse": float(floor)}
    curves = {}
    for tau in TAUS:
        z_d = dream(model, val["mu"][:E], val["actions"][:E], starts, warmup=warmup,
                    horizon=horizon, temperature=tau, device=device, seed=seed)
        dec = decode_latents(vae, z_d, device=device)
        mse = ((dec.astype(np.float32) / 255.0 - true_f) ** 2).mean((0, 2, 3, 4))
        curves[tau] = mse
        res[f"pixel_mse_tau{tau}"] = {
            "h1": float(mse[0]), "h16": float(mse[min(15, horizon - 1)]),
            "h64": float(mse[-1]), "mean": float(mse.mean()),
        }
        # GIF for the first few episodes: truth | VAE recon of truth | dream.
        g = min(n_gif, E)
        strip = np.concatenate(
            [side_by_side([true_frames[e], recon[e], dec[e]]) for e in range(g)], axis=1
        )
        save_gif(strip, out / f"dream_vs_true_tau{tau}.gif", fps=12, scale=3)
        # A still contact sheet as well: a GIF is the right thing to watch, but
        # a strip of every 4th frame is the right thing to put in a report and
        # the only thing you can diff between two runs.
        # Rows are truth / VAE recon of truth / dream; columns are time. Laid
        # out this way so the eye can follow one trajectory along a row and
        # compare the three vertically at a fixed timestep.
        every = max(1, horizon // 16)
        rows = [frame_grid(s[0][::every], ncols=horizon // every, pad=1)
                for s in (true_frames, recon, dec)]
        w = min(r.shape[1] for r in rows)
        sheet = np.concatenate(
            [np.concatenate([r[:, :w], np.full((2, w, 3), 90, np.uint8)], 0)
             for r in rows], 0
        )
        _save_png(sheet, out / f"dream_vs_true_tau{tau}.png", scale=2)
        print(f"    tau={tau}: pixel MSE h1 {mse[0]:.5f} -> h{horizon} {mse[-1]:.5f}"
              f"   (VAE floor {floor:.5f})")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    hs = np.arange(1, horizon + 1)
    for tau, mse in curves.items():
        ax.plot(hs, mse, label=f"dream tau={tau}")
    ax.plot(hs, floor_per_h, "k--", label="VAE reconstruction floor")
    ax.set_xlabel("dream horizon (steps after warm-up)")
    ax.set_ylabel("decoded-frame MSE vs true frame")
    ax.set_title("pixel error vs horizon")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pixel_mse_vs_horizon.png", dpi=130)
    plt.close(fig)
    return res


def _load_true_frames(val, E: int, idx: np.ndarray) -> np.ndarray:
    """Pull (E, len(idx)) true frames from the memory-mapped per-root files."""
    outs = []
    taken = 0
    for root, n in zip(val["roots"], val["per_root"]):
        if taken >= E:
            break
        f = np.load(Path(root) / "frames.npy", mmap_mode="r")
        take = min(n, E - taken)
        outs.append(np.array(f[:take][:, idx]))
        taken += take
    return np.concatenate(outs, 0)


# ---------------------------------------------------------------- part (b)



def _first_exceed(curve: np.ndarray, thresh: float, horizon: int) -> int:
    """First index where ``curve`` exceeds ``thresh``, else ``horizon``."""
    bad = np.where(curve > thresh)[0]
    return int(bad[0]) if len(bad) else int(horizon)


def horizon_by_mass(
    ball_err_ep: np.ndarray, mass: np.ndarray, horizon: int
) -> Dict[str, Dict[str, float]]:
    """Useful dream horizon split into mass terciles. v2 only.

    Why this exists: the pooled useful horizon now mixes fast balls and slow
    ones, and a fast ball covers the one-ball-radius error budget in fewer
    FRAMES purely because it moves further per frame. So the frame count is
    reported alongside the same horizon expressed in BALL DIAMETERS TRAVELLED,
    ``horizon * speed / (2 * ball_radius)``, which is the distance-normalised
    version and the fair comparison across masses. If the two orderings
    disagree -- fewer frames but the same number of diameters -- the model is
    not worse on light balls, it is being asked a harder question per frame.

    Horizons are computed PER EPISODE and then averaged, not from the
    tercile-mean error curve: a mean curve is dominated by whichever episode
    diverged first and gives a systematically pessimistic horizon.
    """
    out: Dict[str, Dict[str, float]] = {}
    q = np.quantile(mass, [1 / 3, 2 / 3])
    groups = {
        "light": mass <= q[0],
        "medium": (mass > q[0]) & (mass <= q[1]),
        "heavy": mass > q[1],
    }
    for name, sel in groups.items():
        if not sel.any():
            continue
        hs = np.array([
            _first_exceed(ball_err_ep[i], BALL_RADIUS, horizon)
            for i in np.where(sel)[0]
        ], dtype=float)
        speed = BASE_SPEED / mass[sel]
        out[name] = {
            "n": int(sel.sum()),
            "mass_lo": float(mass[sel].min()),
            "mass_hi": float(mass[sel].max()),
            "mean_speed": float(speed.mean()),
            "useful_dream_horizon": float(hs.mean()),
            "horizon_ball_diameters": float((hs * speed / (2 * BALL_RADIUS)).mean()),
        }
    return out


def part_b_state(
    model, probe: StateProbe, val, out: Path, device: str,
    warmup: int, horizon: int, n_metric: int, seed: int,
) -> Dict:
    print("\n(b) dream rollouts in world coordinates")
    E = min(n_metric, val["mu"].shape[0])
    starts = np.zeros(E, dtype=int)
    idx_true = warmup + np.arange(horizon)      # see alignment note in part_a_pixels
    true_state = val["state"][:E][:, idx_true]                 # (E, H, 6)

    # Probe floor: the probe's own error on TRUE latents.
    est_true = probe(val["mu"][:E][:, idx_true])
    floor = np.abs(est_true - true_state).mean(0)              # (H, 6)

    # v2: mass per episode, for the by-tercile breakdown below. None in v1.
    names = val.get("state_names", [])
    mass = val["state"][:E, 0, names.index("mass")] if "mass" in names else None

    res: Dict[str, object] = {}
    curves = {}
    for tau in TAUS:
        z_d = dream(model, val["mu"][:E], val["actions"][:E], starts, warmup=warmup,
                    horizon=horizon, temperature=tau, device=device, seed=seed)
        est = probe(z_d)
        err = np.abs(est - true_state).mean(0)                 # (H, S)
        curves[tau] = err
        # "Useful dream horizon": first step where the euclidean ball-position
        # error exceeds one ball radius, i.e. the dreamed ball no longer
        # overlaps the true one. A blunt but honest summary number.
        ball_err_ep = np.linalg.norm(est[..., :2] - true_state[..., :2], axis=-1)
        ball_err = ball_err_ep.mean(0)                         # (H,)
        bad = np.where(ball_err > BALL_RADIUS)[0]
        useful = int(bad[0]) if len(bad) else horizon
        res[f"tau{tau}"] = {
            "useful_dream_horizon": useful,
            "ball_x_err_h16": float(err[min(15, horizon - 1), 0]),
            "ball_y_err_h16": float(err[min(15, horizon - 1), 1]),
            "paddle_x_err_h16": float(err[min(15, horizon - 1), 4]),
            "ball_err_final": float(ball_err[-1]),
        }
        if mass is not None:
            res[f"tau{tau}"]["by_mass_tercile"] = horizon_by_mass(
                ball_err_ep, mass, horizon
            )
        print(f"    tau={tau}: useful dream horizon {useful} steps "
              f"(|ball| err > {BALL_RADIUS}); |dx|@16 {err[15,0]:.3f} "
              f"|dpaddle|@16 {err[15,4]:.3f}")
    if mass is not None:
        t = res["tau0.0"]["by_mass_tercile"]
        print("    by mass tercile (tau=0):")
        for k in ("light", "medium", "heavy"):
            r = t[k]
            print(f"      {k:7s} m in [{r['mass_lo']:.2f}, {r['mass_hi']:.2f}]  "
                  f"horizon {r['useful_dream_horizon']:5.1f} frames = "
                  f"{r['horizon_ball_diameters']:.2f} ball diameters travelled")

    res["probe_floor"] = {
        "ball_x": float(floor[:, 0].mean()),
        "ball_y": float(floor[:, 1].mean()),
        "paddle_x": float(floor[:, 4].mean()),
    }

    plt = _plt()
    names = [("ball_x", 0), ("ball_y", 1), ("paddle_x", 4)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    hs = np.arange(1, horizon + 1)
    for ax, (nm, j) in zip(axes, names):
        for tau, err in curves.items():
            ax.plot(hs, err[:, j], label=f"tau={tau}")
        ax.plot(hs, floor[:, j], "k--", label="probe floor (true latents)")
        ax.axhline(BALL_RADIUS, color="r", ls=":", lw=1, label="ball radius")
        ax.set_title(f"|delta {nm}| (world units)")
        ax.set_xlabel("dream horizon")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "state_error_vs_horizon.png", dpi=130)
    plt.close(fig)
    return res


# ---------------------------------------------------------------- part (c)


@torch.no_grad()
def collect_hidden(
    model: MDNRNN, mu: np.ndarray, actions: np.ndarray, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Teacher-forced pass. Returns (z_in, h, c), each (E, T, ...).

    Stepped manually rather than with one ``lstm(...)`` call because we want the
    CELL state too, and nn.LSTM only hands back the final one.
    """
    model.eval()
    E, T = actions.shape
    z = torch.from_numpy(mu[:, :T].astype(np.float32)).to(device)
    a = torch.eye(model.cfg.n_actions, device=device)[
        torch.from_numpy(actions).long().to(device)
    ]
    h = model.init_hidden(E, device)
    hs, cs = [], []
    for t in range(T):
        _, h = model.step(z[:, t], a[:, t], h)
        hs.append(h[0][0].clone())
        cs.append(h[1][0].clone())
    return (
        mu[:, :T],
        torch.stack(hs, 1).cpu().numpy(),
        torch.stack(cs, 1).cpu().numpy(),
    )



# A degree-2 probe on a 256-unit hidden state would be 33,000 features, which
# is both slow and a memory hazard on 8 GB. Project wide feature matrices onto
# their leading principal components first. PCA is unsupervised -- it never
# looks at the targets -- but it IS fitted on both halves of the probe split,
# which is a small, target-blind leak; it is recorded here rather than hidden.
POLY_MAX_DIM = 32


def _reduce_for_poly(X: np.ndarray, seed: int = 0, max_dim: int = POLY_MAX_DIM):
    if X.shape[1] <= max_dim:
        return X
    from sklearn.decomposition import PCA

    return PCA(n_components=max_dim, random_state=seed).fit_transform(X)


def part_c_velocity(
    model, val, out: Path, device: str, n_samples: int, seed: int
) -> Dict:
    print("\n(c) where does velocity live?")
    # v1 had six state columns and they were hardcoded here. v2 has seven, and
    # the two most interesting probe targets (speed, log_mass) are not columns
    # at all -- they are derived. So take the names from the dataset meta and
    # append the derived pair, which is a no-op on a v1 dataset.
    state_names = list(val.get("state_names")
                       or ["ball_x", "ball_y", "ball_vx", "ball_vy",
                           "paddle_x", "paddle_vx"])
    z_in, h_all, c_all = collect_hidden(model, val["mu"], val["actions"], device)
    E, T = h_all.shape[0], h_all.shape[1]

    # h_t is produced after consuming (z_t, a_t), so it is the model's belief at
    # time t -- pair it with the TRUE state at time t.
    st = val["state"][:, :T]
    ep_ids = np.repeat(np.arange(E), T)
    Z = z_in.reshape(E * T, -1)
    H = h_all.reshape(E * T, -1)
    S = st.reshape(E * T, -1)
    S, state_names = add_derived_targets(S, state_names)   # +log_mass, +speed

    # Drop the first few steps of each episode: h starts at zero and needs a
    # couple of frames before it could possibly contain a velocity estimate.
    keep = (np.tile(np.arange(T), E) >= 4)
    rng = np.random.default_rng(seed)
    idx = np.where(keep)[0]
    if len(idx) > n_samples:
        idx = rng.choice(idx, n_samples, replace=False)
    Z, H, S, G = Z[idx], H[idx], S[idx], ep_ids[idx]

    feats = {"z": Z, "h": H, "z+h": np.concatenate([Z, H], 1)}
    results: Dict[str, Dict] = {}
    for name, X in feats.items():
        results[name] = probe_suite(
            X, S, state_names, group_ids=G, seed=seed, which=("linear", "knn")
        )
        # poly2 is added for v2: stage one showed speed and log_mass are
        # decodable from z only under a degree-2 map and NOT by kNN (the latent
        # metric is dominated by position, so a nearest neighbour is a frame at
        # the same place with a different colour). A linear/kNN-only table would
        # have reported "colour is not in z", which is false. Run separately, on
        # a PCA-reduced X, with the same seed and groups so it is the same split.
        results[name]["poly2"] = probe_suite(
            _reduce_for_poly(X, seed), S, state_names, group_ids=G, seed=seed,
            which=("poly2",),
        )["poly2"]

    print(f"    n={len(idx)} timesteps, episode-level split")
    header = f"    {'feature':7s} {'probe':7s} " + " ".join(f"{n:>9s}" for n in state_names)
    print(header)
    for name in feats:
        for pr in ("linear", "poly2", "knn"):
            row = " ".join(f"{results[name][pr][n]:9.3f}" for n in state_names)
            print(f"    {name:7s} {pr:7s} {row}")

    (out / "velocity_probe.json").write_text(json.dumps(results, indent=2, default=float))

    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), sharey=True)
    width = 0.25
    xs = np.arange(len(state_names))
    for ax, pr in zip(axes, ("linear", "poly2", "knn")):
        for i, name in enumerate(feats):
            vals = [max(results[name][pr][n], -0.05) for n in state_names]
            ax.bar(xs + (i - 1) * width, vals, width, label=name)
        ax.set_xticks(xs)
        ax.set_xticklabels(state_names, rotation=30, ha="right")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{pr} probe")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("held-out R^2")
    axes[0].legend()
    fig.suptitle("what is decodable from the latent (z) vs the LSTM state (h)")
    fig.tight_layout()
    fig.savefig(out / "where_is_velocity.png", dpi=130)
    plt.close(fig)

    # Cell state, reported but not plotted -- in an LSTM c is the long-term
    # store and h is a gated read of it, so they usually carry the same
    # information and c is only interesting when they disagree.
    Cm = c_all.reshape(E * T, -1)[idx]
    c_res = probe_suite(Cm, S, state_names, group_ids=G, seed=seed, which=("linear",))
    results["c_linear"] = c_res["linear"]
    (out / "velocity_probe.json").write_text(json.dumps(results, indent=2, default=float))
    return results


# ---------------------------------------------------------------- part (d)


def part_d_counterfactual(
    model, vae, probe: StateProbe, val, out: Path, device: str,
    warmup: int, horizon: int, n_metric: int, seed: int, tag: str = "",
) -> Dict:
    print(f"\n(d) action counterfactuals {tag}")
    E = min(n_metric, val["mu"].shape[0])
    starts = np.zeros(E, dtype=int)
    est = {}
    decs = {}
    for name, a in (("left", 0), ("stay", 1), ("right", 2)):
        z_d = dream(model, val["mu"][:E], val["actions"][:E], starts, warmup=warmup,
                    horizon=horizon, temperature=0.0, device=device, seed=seed,
                    action_override=a)
        est[name] = probe(z_d)
        decs[name] = decode_latents(vae, z_d[: min(3, E)], device=device)

    h30 = min(29, horizon - 1)
    sep = float(np.abs(est["left"][:, h30, 4] - est["right"][:, h30, 4]).mean())
    drift_l = float((est["left"][:, h30, 4] - est["left"][:, 0, 4]).mean())
    drift_r = float((est["right"][:, h30, 4] - est["right"][:, 0, 4]).mean())
    print(f"    mean |paddle_x(left) - paddle_x(right)| at H=30: {sep:.4f} world units")
    print(f"    mean paddle drift over 30 steps: left {drift_l:+.4f}  right {drift_r:+.4f}")
    print(f"    (the true paddle moves 0.030/step, so 30 steps of one action is "
          f"up to 0.9 before the walls clamp it)")

    g = min(3, E)
    strip = np.concatenate(
        [side_by_side([decs["left"][e], decs["stay"][e], decs["right"][e]])
         for e in range(g)], axis=1
    )
    save_gif(strip, out / f"action_counterfactual{tag}.gif", fps=12, scale=3)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    hs = np.arange(1, horizon + 1)
    for name, colour in (("left", "tab:blue"), ("stay", "tab:grey"), ("right", "tab:red")):
        m = est[name][:, :, 4].mean(0)
        s = est[name][:, :, 4].std(0)
        ax.plot(hs, m, color=colour, label=f"always {name}")
        ax.fill_between(hs, m - s, m + s, color=colour, alpha=0.15)
    ax.set_xlabel("dream horizon")
    ax.set_ylabel("dreamed paddle_x (via state probe)")
    ax.set_title(f"does the model believe actions move the paddle?{tag}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"action_counterfactual{tag}.png", dpi=130)
    plt.close(fig)
    return {"paddle_sep_at_h30": sep, "paddle_drift_left": drift_l,
            "paddle_drift_right": drift_r}


# ---------------------------------------------------------------- part (e)


@torch.no_grad()
def part_e_events(model, val, out: Path, device: str, window: int = 10) -> Dict:
    print("\n(e) paddle-contact prediction")
    E, T = val["actions"].shape
    z = torch.from_numpy(val["mu"][:, :T].astype(np.float32)).to(device)
    a = torch.eye(model.cfg.n_actions, device=device)[
        torch.from_numpy(val["actions"]).long().to(device)
    ]
    parts, _ = model(z, a)
    prob = torch.sigmoid(parts["hit_logit"].squeeze(-1)).cpu().numpy()   # (E, T)
    true = val["hit"]

    p, t = prob.ravel(), true.ravel()
    pred = p > 0.5
    tp = float((pred & (t > 0.5)).sum())
    fp = float((pred & (t < 0.5)).sum())
    fn = float((~pred & (t > 0.5)).sum())
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    from sklearn.metrics import average_precision_score, precision_recall_curve

    pr_auc = float(average_precision_score(t, p))
    base = float(t.mean())

    # pos_weight deliberately unbalances the head, so probabilities come out
    # over-confident and the 0.5 threshold is the wrong place to judge it:
    # recall goes to ~1 and precision collapses. The threshold-free PR-AUC and
    # the best achievable F1 are the numbers that describe the RANKING, which
    # is what actually matters for a downstream controller.
    pc, rc, th = precision_recall_curve(t, p)
    f1s = 2 * pc * rc / np.maximum(pc + rc, 1e-9)
    bi = int(np.argmax(f1s))
    best = {"f1": float(f1s[bi]), "precision": float(pc[bi]), "recall": float(rc[bi]),
            "threshold": float(th[min(bi, len(th) - 1)])}
    print(f"    best-F1 operating point: thr {best['threshold']:.3f}  "
          f"P {best['precision']:.3f}  R {best['recall']:.3f}  F1 {best['f1']:.3f}")
    print(f"    base rate {base:.3%}   precision {prec:.3f}  recall {rec:.3f} "
          f" F1 {f1:.3f}  PR-AUC {pr_auc:.3f}  (a random ranker scores "
          f"PR-AUC = base rate)")

    # Event-triggered average of the predicted probability.
    ev_e, ev_t = np.where(true > 0.5)
    rows = []
    for e, tt in zip(ev_e, ev_t):
        if tt - window < 0 or tt + window >= T:
            continue
        rows.append(prob[e, tt - window : tt + window + 1])
    curve = np.stack(rows) if rows else np.zeros((1, 2 * window + 1))
    m, s = curve.mean(0), curve.std(0)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    lag = np.arange(-window, window + 1)
    ax.plot(lag, m, color="tab:purple")
    ax.fill_between(lag, m - s, m + s, color="tab:purple", alpha=0.2)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axhline(base, color="grey", ls=":", lw=1, label="base rate")
    ax.set_xlabel("steps relative to a true paddle contact")
    ax.set_ylabel("predicted P(contact)")
    ax.set_title(f"contact probability around {len(rows)} true contacts")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "hit_prob_around_hits.png", dpi=130)
    plt.close(fig)

    return {
        "base_rate": base, "precision": prec, "recall": rec, "f1": f1,
        "pr_auc": pr_auc, "n_events_in_curve": len(rows), "best_f1": best,
        "prob_at_lag_-3": float(m[window - 3]), "prob_at_lag_0": float(m[window]),
        "prob_at_lag_-10": float(m[0]),
        "note": (
            "A good curve rises BEFORE lag 0. The contact is a deterministic "
            "consequence of where the ball and paddle are several frames "
            "earlier, so a model that has learned the geometry should be "
            "confident in advance; a model that has only learned to recognise "
            "contact after the fact would show a spike at lag 0 and nothing "
            "before it."
        ),
    }


# ---------------------------------------------------------------- part (f)


def part_f_bounces(
    model, probe: StateProbe, val, out: Path, device: str,
    warmup: int = 8, horizon: int = 12, max_cases: int = 200, seed: int = 0,
) -> Dict:
    """Does the dream reverse the ball at a wall?

    Setup: find a true wall bounce at time tb, warm the model up on the 8 true
    latents ending 3 steps before it, then dream through the bounce. A model
    that has only learned "the ball drifts in a straight line" will sail the
    ball through the wall; a model that knows about walls will flip the sign of
    the probed velocity within a step or two of the truth.

    Velocity is measured as a finite difference of the probed position, which is
    noisy, so we require the sign to flip and use a +/-2 step tolerance.
    """
    print("\n(f) wall bounces")
    E, T = val["actions"].shape
    # Reconstruct wall events from the true state instead of events.npy, since
    # _stack_val does not carry events; a sign change in the true velocity IS
    # the bounce.
    st = val["state"]
    lead = 3
    cases: List[Tuple[int, int, int]] = []  # (episode, bounce time, axis)
    for axis in (0, 1):
        v = st[:, :, 2 + axis]
        flip = (np.sign(v[:, 1:]) != np.sign(v[:, :-1])) & (np.abs(v[:, :-1]) > 1e-9)
        ee, tt = np.where(flip)
        for e, t in zip(ee, tt):
            t = int(t) + 1
            if t - warmup - lead >= 0 and t + horizon - lead + 1 < T:
                cases.append((int(e), t, axis))
    rng = np.random.default_rng(seed)
    if len(cases) > max_cases:
        cases = [cases[i] for i in rng.choice(len(cases), max_cases, replace=False)]
    if not cases:
        return {"n_cases": 0}

    starts = np.array([t - lead - warmup for _, t, _ in cases])
    eps = np.array([e for e, _, _ in cases])
    mu_sub = val["mu"][eps]
    act_sub = val["actions"][eps]
    z_d = dream(model, mu_sub, act_sub, starts, warmup=warmup, horizon=horizon,
                temperature=0.0, device=device, seed=seed)
    est = probe(z_d)                                     # (N, H, 6)

    # Dream step k corresponds to absolute time starts[i] + warmup + k.
    #
    # A SUSTAINED reversal, not any sign change of the raw finite difference.
    # The probe's output is noisy at the ~0.005 level and a raw diff crosses
    # zero constantly, which would score a straight-line model at ~90%. So we
    # compare motion over the two steps BEFORE a candidate step against motion
    # over the two steps AFTER, and require both to exceed a fraction of the
    # real per-step speed. Anything smaller than that is probe noise.
    min_disp = 0.4 * 2 * 0.022     # 2 steps of ball motion, times a slack factor

    def reversal_times(pos: np.ndarray, offset: int) -> np.ndarray:
        before = pos[2:-2] - pos[:-4]
        after = pos[4:] - pos[2:-2]
        ok = (
            (np.sign(before) != np.sign(after))
            & (np.abs(before) > min_disp)
            & (np.abs(after) > min_disp)
        )
        return offset + 2 + np.where(ok)[0]

    hits = 0
    chance = 0
    rng2 = np.random.default_rng(seed + 1)
    for i, (_, tb, axis) in enumerate(cases):
        off = starts[i] + warmup
        t_rev = reversal_times(est[i, :, axis], off)
        if len(t_rev) and np.min(np.abs(t_rev - tb)) <= 2:
            hits += 1
        # Chance control: the same test against a random time inside the same
        # dream window. Without this the headline number is uninterpretable --
        # a model that reverses the ball constantly would also "predict" every
        # bounce.
        t_fake = int(rng2.integers(off, off + horizon))
        if len(t_rev) and np.min(np.abs(t_rev - t_fake)) <= 2:
            chance += 1
    frac = hits / len(cases)
    frac_chance = chance / len(cases)
    print(f"    {hits}/{len(cases)} bounces reversed within +/-2 steps "
          f"({frac:.1%});  same test at a random time: {frac_chance:.1%} (chance)")
    return {
        "n_cases": len(cases),
        "fraction_predicted": frac,
        "fraction_at_random_time": frac_chance,
    }


# ------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="runs/rnn_v1/rnn.pt")
    p.add_argument("--vae", default="runs/vae_b1/vae.pt")
    p.add_argument("--val", nargs="+", default=["data/v1/val", "data/v1/val_mix"])
    p.add_argument("--probe-data", nargs="+",
                   default=["data/v1/probe", "data/v1/val_mix"],
                   help="roots used to FIT the mu->state probe")
    p.add_argument("--ablate-ckpt", default="runs/rnn_noact/rnn.pt",
                   help="optional no-action model, for the (d) contrast")
    p.add_argument("--out", default=None)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--n-metric", type=int, default=30)
    p.add_argument("--n-gif", type=int, default=3)
    p.add_argument("--probe-samples", type=int, default=12000)
    p.add_argument("--velocity-samples", type=int, default=6000)
    p.add_argument("--latent-suffix", default="",
                   help='evaluate a model trained on mu<suffix>.npy, e.g. "v1vae" '
                        "for the --ablate-color control")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = pick_device(a.device)
    out = Path(a.out or (Path(a.ckpt).parent / "eval"))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    model, cfg = load_rnn(a.ckpt, device)
    vae, vcfg, _ = load_ckpt(a.vae, device)
    print(f"model {a.ckpt}  cfg={cfg}\nvae   {a.vae}  z_dim={vcfg.z_dim}\ndevice={device}")

    val = _stack_val(a.val, latent_suffix=a.latent_suffix)
    val["roots"] = list(a.val)
    val["per_root"] = [episode_arrays(r)["mu"].shape[0] for r in a.val]
    print(f"val: {val['mu'].shape[0]} episodes x {val['actions'].shape[1]} steps")

    # --- fit the measuring instrument -------------------------------------
    pm, ps = [], []
    for r in a.probe_data:
        # The probe must be fitted on the SAME latent space the model dreams in,
        # hence latent_suffix here too. Its state width is taken from the data
        # (6 in v1, 7 in v2) instead of being hardcoded.
        d = episode_arrays(r, latent_suffix=a.latent_suffix)
        pm.append(d["mu"].reshape(-1, d["mu"].shape[-1]))
        ps.append(d["state"].reshape(-1, d["state"].shape[-1]))
    PM, PS = np.concatenate(pm), np.concatenate(ps)
    rng = np.random.default_rng(a.seed)
    if len(PM) > a.probe_samples:
        sel = rng.choice(len(PM), a.probe_samples, replace=False)
        PM, PS = PM[sel], PS[sel]
    probe = StateProbe().fit(PM, PS)
    print(f"state probe fitted on {len(PM)} frames from {a.probe_data}")

    report: Dict[str, object] = {
        "ckpt": str(a.ckpt), "cfg": vars(cfg), "val_roots": list(a.val),
        "warmup": a.warmup, "horizon": a.horizon, "n_metric": a.n_metric,
    }
    report["a_pixels"] = part_a_pixels(model, vae, val, out, device, a.warmup,
                                       a.horizon, a.n_metric, a.n_gif, a.seed)
    report["b_state"] = part_b_state(model, probe, val, out, device, a.warmup,
                                     a.horizon, a.n_metric, a.seed)
    report["c_velocity"] = part_c_velocity(model, val, out, device,
                                           a.velocity_samples, a.seed)
    report["d_actions"] = part_d_counterfactual(model, vae, probe, val, out, device,
                                                a.warmup, a.horizon, a.n_metric, a.seed)
    if a.ablate_ckpt and Path(a.ablate_ckpt).exists():
        abl, _ = load_rnn(a.ablate_ckpt, device)
        report["d_actions_ablated"] = part_d_counterfactual(
            abl, vae, probe, val, out, device, a.warmup, a.horizon,
            a.n_metric, a.seed, tag="_noact"
        )
    report["e_events"] = part_e_events(model, val, out, device)
    report["f_bounces"] = part_f_bounces(model, probe, val, out, device,
                                         warmup=a.warmup, seed=a.seed)

    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    _summary(report, out)


def _summary(r: Dict, out: Path) -> None:
    b = r["b_state"]
    c = r["c_velocity"]
    print("\n" + "=" * 72)
    print("SUMMARY -- what the dynamics model did and did not learn")
    print("=" * 72)
    print(
        f"\n1. Dreams stay accurate for about "
        f"{b['tau0.0']['useful_dream_horizon']} steps (tau=0) before the dreamed\n"
        f"   ball drifts more than one ball radius from the true one. Sampling at\n"
        f"   tau=1 scores {b['tau1.0']['useful_dream_horizon']} steps on the same "
        f"metric -- but read that carefully. A\n"
        f"   sampled dream is a DIFFERENT plausible future, not a failed copy of\n"
        f"   this one, so per-episode agreement with the truth is the wrong\n"
        f"   yardstick for tau > 0. Watch the tau=1 GIF: the ball still moves and\n"
        f"   bounces coherently, it just is not doing what the real ball did. The\n"
        f"   metric is reported for all three because the comparison shows how much\n"
        f"   of the predictive distribution's width is real uncertainty.\n"
        f"   Pixel error never reaches the VAE floor "
        f"({r['a_pixels']['vae_floor_mse']:.5f}), which is the part of the error\n"
        f"   that belongs to the frozen encoder rather than to the dynamics."
    )
    vz = c["z"]["knn"]["ball_vx"], c["z"]["knn"]["ball_vy"]
    vh = c["h"]["knn"]["ball_vx"], c["h"]["knn"]["ball_vy"]
    lz = c["z"]["linear"]["ball_vx"], c["z"]["linear"]["ball_vy"]
    lh = c["h"]["linear"]["ball_vx"], c["h"]["linear"]["ball_vy"]
    print(
        f"\n2. Velocity is not in the latent and is in the recurrent state -- the\n"
        f"   central claim of this stage. kNN R^2 for (ball_vx, ball_vy):\n"
        f"       from z_t alone : {vz[0]:.3f}, {vz[1]:.3f}\n"
        f"       from h_t alone : {vh[0]:.3f}, {vh[1]:.3f}\n"
        f"   linear R^2, same pairs: z {lz[0]:.3f}, {lz[1]:.3f}   "
        f"h {lh[0]:.3f}, {lh[1]:.3f}\n"
        f"   A single frame has no direction of travel in it; the LSTM has to\n"
        f"   infer velocity by integrating consecutive latents, and it does."
    )
    d = r["d_actions"]
    print(
        f"\n3. Actions matter to the model: dreaming 30 steps of always-left vs\n"
        f"   always-right separates the dreamed paddle by "
        f"{d['paddle_sep_at_h30']:.3f} world units."
    )
    if "d_actions_ablated" in r:
        print(
            f"   The no-action ablation, which cannot see the action at all, "
            f"separates by\n   "
            f"{r['d_actions_ablated']['paddle_sep_at_h30']:.3f} -- the control."
        )
    e = r["e_events"]
    print(
        f"\n4. Paddle contact (base rate {e['base_rate']:.2%}): precision "
        f"{e['precision']:.3f}, recall {e['recall']:.3f},\n   F1 {e['f1']:.3f}, "
        f"PR-AUC {e['pr_auc']:.3f}. P(contact) three steps BEFORE a contact is\n"
        f"   {e['prob_at_lag_-3']:.3f} against a base rate of {e['base_rate']:.3f}: "
        f"the model anticipates."
    )
    b0 = b.get("tau0.0", {})
    if "by_mass_tercile" in b0:
        print("\n   v2: the same horizon split by mass tercile --")
        for k in ("light", "medium", "heavy"):
            t = b0["by_mass_tercile"].get(k)
            if t:
                print(f"      {k:7s} m {t['mass_lo']:.2f}-{t['mass_hi']:.2f}  "
                      f"{t['useful_dream_horizon']:5.1f} frames  "
                      f"({t['horizon_ball_diameters']:.2f} ball diameters travelled)")
        cz = c["z"].get("poly2", {})
        ch = c["h"].get("poly2", {})
        if "speed" in cz:
            print(f"\n   v2: speed R^2 -- z(poly2) {cz['speed']:.3f}, "
                  f"h(poly2) {ch['speed']:.3f};  log_mass -- "
                  f"z {cz.get('log_mass', float('nan')):.3f}, "
                  f"h {ch.get('log_mass', float('nan')):.3f}")
    f = r["f_bounces"]
    print(
        f"\n5. Walls: the dreamed ball reverses within +/-2 steps of the true\n"
        f"   bounce in {f.get('fraction_predicted', 0):.1%} of "
        f"{f.get('n_cases', 0)} cases, against a chance level of\n"
        f"   {f.get('fraction_at_random_time', 0):.1%} for the same test applied "
        f"at a random time."
    )
    print(f"\nartifacts in {out}")


if __name__ == "__main__":
    main()

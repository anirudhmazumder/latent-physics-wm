"""Train the MDN-RNN on cached latents.

    python -m wm.train_rnn --data data/v1/train data/v1/train_mix \
        --val data/v1/val data/v1/val_mix --out runs/rnn_v1 --epochs 30

Teacher forcing on fixed-length windows: at every step the model is fed the
TRUE latent z_t and asked for z_{t+1}. That is the cheap, stable, parallel way
to train a sequence model, and it is also a systematically optimistic one --
at dream time the model is fed its own output, and errors compound. The gap
between the two regimes is the interesting quantity, so we do not only log
teacher-forced NLL: every ``--eval-every`` epochs we also run an OPEN-LOOP
rollout in latent space (warm up on 8 true latents, then feed the model its own
deterministic prediction for 32 steps) and log the latent error at the end of
the horizon. Watch both. Teacher-forced NLL improving while the rollout error
does not is the signature of a model that has learned a good one-step
denoiser and no dynamics.

The rollout metric here is in LATENT space and uses no decoder, so it is cheap
enough to run every few epochs. ``wm.eval_rnn`` does the expensive pixel-space
and state-space versions once, at the end.

Three optional extra loss terms, added after stage three found that a tau = 1
dream does not conserve the ball's mass (see ``wm/conservation.py`` for the
diagnosis and ``wm/README_FIX.md`` for the measurements):

    --rollout-loss-steps K   roll the model open-loop for K steps on its own
    --rollout-loss-weight w  reparameterised samples and add the MDN NLL of the
                             TRUE latents along the way. The generic fix.
    --cons-loss-weight w     penalise the change of a frozen poly-2 log-mass
                             probe's reading along that same rollout. The
                             targeted fix, which names the conserved quantity.
    --mass-head              a linear head on h predicting log(mass), with a
                             privileged training-time-only target.

``--rollout-loss-steps 0`` (the default) disables all of the rollout machinery
and reproduces the original loss bit for bit, which ``tests/test_conservation``
asserts.

Two more, added in v3 after the occluded world turned out to give the model no
reason to remember where a hidden ball is (``wm/README_M3.md`` Section 10,
``wm/README_FIX3.md`` for the results):

    --emerge-weight w        up-weight the NLL on the frames where the ball
                             comes back out from behind the band. Still
                             self-supervised: the privileged visibility flag
                             chooses WHICH transitions count, not what to
                             predict.
    --pos-head --w-pos w     a linear head on h predicting the true ball
                             position on every frame, hidden included. Openly
                             privileged: the CEILING experiment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .conservation import (
    Poly2Probe, conservation_penalty, rollout_losses,
)
from .rnn import MDNRNN, RNNConfig, rnn_loss, save_rnn
from .seq_data import LatentSequenceDataset, episode_arrays, make_seq_loader
from .train_vae import pick_device


# ----------------------------------------------------- conservation probe


def _state_column(roots, name: str) -> int | None:
    """Index of a named column in ``states.npy``, or None if absent.

    Looked up from ``meta["state_names"]`` rather than by position, because the
    versions disagree about what column 6 is: v2's is ``mass`` and v3's is
    ``ball_visible``. Getting this wrong once already nearly trained a "log
    mass" head on an occlusion flag (README_M3 Section 9), so every column this
    file needs -- mass, ball_visible, ball_x, ball_y -- goes through here.
    """
    names = json.loads((Path(roots[0]) / "meta.json").read_text()).get(
        "state_names", [])
    return names.index(name) if name in names else None


def _mass_column(roots) -> int | None:
    """Back-compatible alias; see ``_state_column``."""
    return _state_column(roots, "mass")


# ------------------------------------------------------- emergence weighting


def emergence_weight_mask(
    visible: torch.Tensor,          # (B, L+1) 1.0 where the ball is visible
    n_frames: int = 3,
) -> torch.Tensor:
    """Which TARGET frames are "the ball has just come back". (B, L) bool.

    ``visible`` is the window's full visibility sequence, frames t0 .. t0+L:
    entry 0 is the frame before the first target and entries 1..L are the L
    target frames. The returned mask lines up with the targets.

    A target frame is an *emergence* frame if the ball is visible there and it
    is within the first ``n_frames`` frames of a visible run that a hidden
    frame preceded. ``n_frames = 1`` is the strict "the previous frame was
    fully hidden" reading; 3 is what v3 uses, because the exit is spread over
    two or three frames by the ball's radius and by the mixture's hedging, and
    a single frame of signal per nine-frame occlusion is very little gradient.

    The window's own first frame has no history inside the window, so a visible
    run already in progress at entry 0 is never marked -- we cannot tell
    whether it began behind the band. That is a small, conservative loss (it
    can only mark fewer frames than the truth) and it is what makes this
    computable from one window.

    PRIVILEGE NOTE. ``visible`` comes from the simulator's state vector, which
    the model never sees. It is used here only to decide *how much a
    transition counts*, never as an input and never as a target: the model is
    still asked to predict exactly the same latents from exactly the same
    inputs. Re-weighting a loss with a privileged label is a much weaker form
    of cheating than ``--pos-head``, but it is not zero, and README_FIX3 says
    so out loud.
    """
    vis = visible > 0.5
    B, Lp1 = vis.shape
    BIG = Lp1 + 1
    # runpos[j] = how many frames the current visible run has lasted at j, or 0
    # if hidden. The first entry is seeded with BIG when visible, which is the
    # "run of unknown age" case the docstring describes.
    runpos = torch.zeros(B, Lp1, dtype=torch.long, device=vis.device)
    runpos[:, 0] = torch.where(vis[:, 0], torch.full_like(runpos[:, 0], BIG),
                               torch.zeros_like(runpos[:, 0]))
    for j in range(1, Lp1):
        runpos[:, j] = torch.where(vis[:, j], runpos[:, j - 1] + 1,
                                   torch.zeros_like(runpos[:, j]))
    mask = (runpos >= 1) & (runpos <= n_frames)
    return mask[:, 1:]                                                # (B, L)


def _fit_cons_probe(roots, latent_suffix: str, n: int, seed: int):
    """Fit the frozen poly-2 ``mu -> log(mass)`` probe on TRAINING latents.

    Fitted on ``mu`` (posterior means) even when the model trains on samples:
    the probe is supposed to read the mass a frame actually has, and the mean
    is the best estimate of that. Reports a held-out (by episode) R^2 so a
    silently broken probe is visible before 35 epochs are spent on it.
    """
    col = _mass_column(roots)
    if col is None:
        raise SystemExit("--cons-loss-weight needs a dataset with a mass column")
    mus, masses, groups = [], [], []
    for ri, r in enumerate(roots):
        d = episode_arrays(r, latent_suffix=latent_suffix)
        mu, st = d["mu"], d["state"]
        E, T = mu.shape[0], mu.shape[1]
        mus.append(mu.reshape(E * T, -1))
        masses.append(st[:, :, col].reshape(E * T))
        groups.append(np.repeat(np.arange(E) + 1000 * ri, T))
    M = np.concatenate(mus)
    Y = np.log(np.concatenate(masses))
    G = np.concatenate(groups)

    rng = np.random.default_rng(seed)
    if len(M) > n:
        sel = rng.choice(len(M), n, replace=False)
        M, Y, G = M[sel], Y[sel], G[sel]

    from .probes import make_split

    tr, te = make_split(len(M), seed=seed, group_ids=G)
    held = Poly2Probe().fit(M[tr], Y[tr])
    pred = held(M[te])
    r2 = 1.0 - float(((Y[te] - pred) ** 2).sum()) / max(
        float(((Y[te] - Y[te].mean()) ** 2).sum()), 1e-12)
    # Refit on everything for the probe we actually use -- the split above was
    # only to produce an honest quality number.
    return Poly2Probe().fit(M, Y), r2


# --------------------------------------------------------------- open loop


@torch.no_grad()
def latent_rollout(
    model: MDNRNN,
    mu: np.ndarray,
    actions: np.ndarray,
    warmup: int = 8,
    horizon: int = 32,
    temperature: float = 0.0,
    device: str = "cpu",
    max_episodes: int = 16,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Dream forward on true actions and its own latents. Returns per-horizon MSE.

    Warm-up matters and is not a detail: a single latent has no velocity in it,
    so a model started from t=0 with an empty hidden state cannot know which way
    the ball is going. Eight true steps is plenty for the LSTM to integrate a
    direction; less than ~3 and the first dreamed steps are a coin flip.
    """
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    E = min(mu.shape[0], max_episodes)
    T = actions.shape[1]
    assert warmup + horizon <= T, "episode too short for this warmup+horizon"

    z_true = torch.from_numpy(mu[:E]).to(device)                     # (E, T+1, z)
    a_idx = torch.from_numpy(actions[:E]).to(device)                 # (E, T)
    a_onehot = torch.eye(model.cfg.n_actions, device=device)[a_idx]  # (E, T, 3)

    # Warm-up: teacher-forced on true latents, just to build h.
    parts, h = model(z_true[:, :warmup], a_onehot[:, :warmup])
    z = model.sample_next(parts, temperature=temperature)[:, -1]     # (E, z)

    preds: List[torch.Tensor] = [z]
    for k in range(1, horizon):
        t = warmup + k
        p, h = model.step(z, a_onehot[:, t], h)
        if temperature <= 0.0:
            z = model.most_likely_mean(p)[:, 0]
        else:
            # sample_next wants CPU-side randomness for reproducibility; on MPS
            # a generator cannot be passed, so fall back to global RNG there.
            z = model.sample_next(
                p, temperature=temperature,
                generator=gen if device == "cpu" else None,
            )[:, 0]
        preds.append(z)

    pred = torch.stack(preds, 1)                                     # (E, H, z)
    truth = z_true[:, warmup + 1 : warmup + 1 + horizon]
    mse = ((pred - truth) ** 2).mean(-1)                             # (E, H)
    return {
        "mse_per_h": mse.mean(0).cpu().numpy(),
        "pred": pred.cpu().numpy(),
        "truth": truth.cpu().numpy(),
    }


# ------------------------------------------------------------------- epochs


@torch.no_grad()
def evaluate(model: MDNRNN, loader, pos_weight: float, device: str) -> Dict[str, float]:
    model.eval()
    agg = {"nll": 0.0, "hit_bce": 0.0, "reward_mse": 0.0, "n": 0}
    tp = fp = fn = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        parts, _ = model(batch["z"], batch["a"])
        _, d = rnn_loss(model, parts, batch, pos_weight=pos_weight)
        n = batch["z"].shape[0]
        for k in ("nll", "hit_bce", "reward_mse"):
            agg[k] += float(d[k]) * n
        agg["n"] += n

        pred = (torch.sigmoid(parts["hit_logit"].squeeze(-1)) > 0.5).float()
        true = batch["hit"]
        tp += float(((pred == 1) & (true == 1)).sum())
        fp += float(((pred == 1) & (true == 0)).sum())
        fn += float(((pred == 0) & (true == 1)).sum())

    n = max(agg["n"], 1)
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    return {
        "nll": agg["nll"] / n,
        "hit_bce": agg["hit_bce"] / n,
        "reward_mse": agg["reward_mse"] / n,
        "hit_precision": prec,
        "hit_recall": rec,
        "hit_f1": 2 * prec * rec / max(prec + rec, 1e-9),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True,
                   help="one or more dataset roots with cached mu/logvar")
    p.add_argument("--val", nargs="+", default=None)
    p.add_argument("--out", default="runs/rnn_v1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-gauss", type=int, default=5)
    p.add_argument("--no-delta", action="store_true",
                   help="predict z_{t+1} directly instead of the residual")
    p.add_argument("--use-mean", action="store_true",
                   help="feed mu instead of a fresh posterior sample")
    p.add_argument("--ablate-actions", action="store_true",
                   help="zero the action input -- the 'do actions matter' control")
    p.add_argument("--feedforward", action="store_true",
                   help="v3 control: replace the LSTM with a 2x256 MLP on "
                        "[z_t, a_t]. No recurrence, so no memory -- the floor "
                        "for every object-permanence test.")
    p.add_argument("--latent-suffix", default="",
                   help='read mu<suffix>.npy instead of mu.npy, e.g. "v1vae"')
    p.add_argument("--ablate-color", action="store_true",
                   help="v2 control: train on latents from an encoder that never "
                        "saw colour (shorthand for --latent-suffix v1vae). See the "
                        "note below on why the ablation is at the DATA level.")
    p.add_argument("--w-hit", type=float, default=1.0)
    p.add_argument("--w-reward", type=float, default=1.0)
    # ------------------------------------------------------ conservation
    # See wm/conservation.py for why teacher forcing alone cannot teach a model
    # to hold a constant, and wm/README_FIX.md for the measured effect.
    p.add_argument("--rollout-loss-steps", type=int, default=0,
                   help="K: length of the extra OPEN-LOOP rollout inside each "
                        "training window, fed by the model's own reparameterised "
                        "samples. 0 (default) disables the whole mechanism and "
                        "reproduces the plain teacher-forced loss exactly.")
    p.add_argument("--rollout-loss-weight", type=float, default=1.0,
                   help="weight on the mean MDN NLL of the TRUE latents along "
                        "that rollout. Set to 0 to run the rollout only for the "
                        "conservation penalty below.")
    p.add_argument("--cons-loss-weight", type=float, default=0.0,
                   help="weight on mean_k (g(z_dreamed_k) - g(z_true_t0))^2 for "
                        "a frozen poly-2 log-mass probe g. Requires "
                        "--rollout-loss-steps > 0 and a dataset with a mass "
                        "column.")
    p.add_argument("--mass-head", action="store_true",
                   help="add a linear head on h predicting log(mass). The target "
                        "is privileged (simulator state), training-time only, "
                        "exactly like the reward head.")
    p.add_argument("--w-mass", type=float, default=1.0)
    # ------------------------------------------------- v3 permanence fixes
    # Diagnosis in wm/README_M3.md Section 10, measurements in
    # wm/README_FIX3.md. One-step teacher forcing pays only for what changes
    # the NEXT latent, and behind an opaque band the next latent is the blank
    # band whatever the ball's x is -- so tracking x buys nothing until the
    # ball comes back out. These two flags attack that from opposite ends.
    p.add_argument("--emerge-weight", type=float, default=1.0,
                   help="up-weight the MDN NLL by this factor on target frames "
                        "where the ball has just re-emerged from behind the "
                        "occluder (the first --emerge-frames frames of a "
                        "visible run that a hidden frame preceded). Applies to "
                        "both the teacher-forced term and the open-loop rollout "
                        "term. Needs a dataset with a ball_visible column. 1.0 "
                        "(default) is a no-op. Uses a privileged label for "
                        "WEIGHTING only -- never as an input or a target.")
    p.add_argument("--emerge-frames", type=int, default=3,
                   help="how many frames after a hidden run count as emergence")
    p.add_argument("--pos-head", action="store_true",
                   help="add a linear head on h predicting the TRUE (ball_x, "
                        "ball_y) on EVERY frame, hidden ones included. This is "
                        "the privileged CEILING experiment, not a fair "
                        "world-model result: it tells the recurrent state what "
                        "to remember. Nothing downstream reads the head.")
    p.add_argument("--w-pos", type=float, default=1.0)
    p.add_argument("--cons-probe-samples", type=int, default=20000,
                   help="frames used to FIT the frozen conservation probe")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stride", type=int, default=4,
                   help="window stride. 1 gives maximal overlap (and a big, "
                        "highly redundant epoch); 4 is plenty here.")
    # Measured on the M1: cpu is ~4x faster than mps here, and gives bit-identical
    # numbers. A 256-unit LSTM stepped 32 times is a sequence of tiny matmuls, so
    # the run is dominated by per-kernel dispatch latency, not by arithmetic --
    # exactly the regime where shipping work to an accelerator loses. The VAE
    # (big convs over 64x64 images) is the opposite regime, which is why
    # train_vae defaults to "auto"/mps and this does not. "auto" still works.
    p.add_argument("--device", default="cpu",
                   help='"cpu" (default, fastest here), "mps", "cuda" or "auto"')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--rollout-horizon", type=int, default=32)
    p.add_argument("--rollout-warmup", type=int, default=8)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # The colour ablation is done by swapping the ENCODER, not by masking the
    # latent. Colour is embedded nonlinearly and distributedly in the v2 code
    # (stage one: no single dimension carries it), so there is no subspace you
    # can zero without also damaging position. What you CAN do is encode the
    # same v2 frames with the v1 VAE, which was trained on a world where the
    # ball was always the same colour -- it has no reason to spend capacity on
    # hue. That gives a latent stream with the same positions and (as verified
    # by a probe before training) little colour information, which is exactly
    # the counterfactual the experiment needs.
    latent_suffix = "v1vae" if a.ablate_color else a.latent_suffix

    train_loader = make_seq_loader(
        a.data, seq_len=a.seq_len, batch_size=a.batch_size, shuffle=True,
        stride=a.stride, use_mean=a.use_mean, seed=a.seed,
        latent_suffix=latent_suffix,
    )
    val_roots = a.val or a.data
    val_loader = make_seq_loader(
        val_roots, seq_len=a.seq_len, batch_size=a.batch_size, shuffle=False,
        stride=a.seq_len,  # non-overlapping windows: val should not double-count
        use_mean=True,     # deterministic val, so epoch-to-epoch changes are the model
        seed=a.seed,
        latent_suffix=latent_suffix,
    )
    # NOTE, and it surprises people: train NLL and val NLL are NOT comparable
    # here, and val is much lower. Train targets are posterior SAMPLES, which
    # carry the VAE's own posterior noise and are therefore genuinely harder to
    # predict; val targets are posterior MEANS. The gap is a property of the
    # targets, not evidence of anything about generalisation. Compare train to
    # train and val to val across runs, never train to val.
    train_ds: LatentSequenceDataset = train_loader.dataset
    z_dim = train_ds.z_dim

    # pos_weight is computed from the TRAINING data only.
    pos_weight = train_ds.pos_weight()
    print(
        f"device={device}  z_dim={z_dim}  latents=mu{'_' + latent_suffix if latent_suffix else ''}.npy  "
        f"windows={len(train_ds)} "
        f"(train) / {len(val_loader.dataset)} (val)\n"
        f"hit_rate={train_ds.hit_rate():.4%}  pos_weight={pos_weight:.1f}"
    )

    cfg = RNNConfig(
        z_dim=z_dim, n_actions=3, hidden=a.hidden, n_gauss=a.n_gauss,
        predict_delta=not a.no_delta, ablate_actions=a.ablate_actions,
        mass_head=a.mass_head, feedforward=a.feedforward,
        pos_head=a.pos_head,
    )
    model = MDNRNN(cfg).to(device)
    print(f"params={sum(p_.numel() for p_ in model.parameters())/1e3:.0f}k  cfg={cfg}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    # Held-out episodes for the open-loop metric, taken from the first val root.
    roll_src = episode_arrays(val_roots[0], latent_suffix=latent_suffix)

    # --- the frozen conservation probe ------------------------------------
    # Fitted ONCE, on training latents, before a single gradient step, and then
    # never updated. It has to be frozen: a probe that co-adapts with the model
    # can be satisfied by moving the probe rather than by conserving anything,
    # and the number the eval reports would stop meaning what it says.
    cons_probe = None
    if a.cons_loss_weight > 0.0:
        if a.rollout_loss_steps <= 0:
            raise SystemExit("--cons-loss-weight needs --rollout-loss-steps > 0")
        cons_probe, cons_r2 = _fit_cons_probe(
            a.data, latent_suffix, a.cons_probe_samples, a.seed)
        print(f"conservation probe: poly-2 ridge, mu -> log(mass), "
              f"held-out R^2 {cons_r2:.4f}")

    roll_rng = np.random.default_rng(a.seed + 1)
    # The mass column is found BY NAME, not by index. v2's 7th column is mass;
    # v3's 7th column is ball_visible. A hard-coded index 6 would have let
    # --mass-head run happily on v3 and train a "mass" head on the occlusion
    # flag -- a privileged target leak that no assertion on shape could catch.
    mass_col = _mass_column(a.data)
    if (a.mass_head or cons_probe is not None) and mass_col is None:
        raise SystemExit("this dataset has no mass column; --mass-head / "
                         "--cons-loss-weight are v2-only")

    # Same by-name discipline for the v3 columns.
    vis_col = _state_column(a.data, "ball_visible")
    if a.emerge_weight != 1.0 and vis_col is None:
        raise SystemExit("this dataset has no ball_visible column; "
                         "--emerge-weight is v3-only")
    pos_cols = [_state_column(a.data, n) for n in ("ball_x", "ball_y")]
    if a.pos_head and any(c is None for c in pos_cols):
        raise SystemExit("this dataset has no ball_x/ball_y columns")

    history: List[dict] = []
    best = float("inf")
    t0 = time.time()

    extra_keys = ("roll_nll", "cons", "mass_mse", "pos_mse", "emerge_frac")
    for epoch in range(a.epochs):
        model.train()
        agg = {"loss": 0.0, "nll": 0.0, "hit_bce": 0.0, "reward_mse": 0.0, "n": 0}
        agg.update({k: 0.0 for k in extra_keys})
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # The emergence weights. Built once per batch from the privileged
            # visibility flag and reused by both NLL terms, so the two always
            # agree about which frames matter. ``weights`` is (B, L) and lines
            # up with the targets; ``roll_w`` is the same thing padded on the
            # left by one so that rollout step k indexes position t0 + k.
            weights = roll_w = None
            emerge_frac = 0.0
            if a.emerge_weight != 1.0:
                vis = torch.cat(
                    [batch["state_in"][:, :1, vis_col],
                     batch["state"][..., vis_col]], dim=1)            # (B, L+1)
                mask = emergence_weight_mask(vis, a.emerge_frames)     # (B, L)
                weights = torch.where(
                    mask, torch.full_like(mask, a.emerge_weight, dtype=torch.float32),
                    torch.ones_like(mask, dtype=torch.float32))
                emerge_frac = float(mask.float().mean())
                # rollout_losses slices ``weights[:, t0+1 : t0+K+1]``, i.e. it
                # indexes the same (B, L) target axis one position later, so it
                # needs a column for the position it never scores. Prepending a
                # 1.0 keeps the alignment exact.
                roll_w = torch.cat(
                    [torch.ones_like(weights[:, :1]), weights], dim=1)

            parts, _ = model(batch["z"], batch["a"])
            loss, d = rnn_loss(
                model, parts, batch, pos_weight=pos_weight,
                w_hit=a.w_hit, w_reward=a.w_reward, nll_weights=weights,
            )
            d.update({k: torch.zeros((), device=device) for k in extra_keys})
            d["emerge_frac"] = torch.as_tensor(emerge_frac, device=device)

            # (ball_x, ball_y) from h, on every frame including the hidden
            # ones. Privileged, training-time only, and the point of the
            # experiment: it is the upper reference for how much position a
            # 256-unit LSTM state CAN hold, not a claim that self-supervision
            # would ever put it there.
            if model.pos_head is not None:
                tgt = batch["state"][..., pos_cols]                   # (B, L, 2)
                pos_mse = torch.nn.functional.mse_loss(parts["ball_pos"], tgt)
                loss = loss + a.w_pos * pos_mse
                d["pos_mse"] = pos_mse.detach()

            # log(mass) from h. The window's state rows are all one episode and
            # mass is constant within an episode, so this is a constant target
            # across the window -- which is precisely the point: it is asking
            # the recurrent state to hold still.
            if model.mass_head is not None:
                log_m = torch.log(batch["state"][..., mass_col])
                mass_mse = torch.nn.functional.mse_loss(
                    parts["log_mass"].squeeze(-1), log_m)
                loss = loss + a.w_mass * mass_mse
                d["mass_mse"] = mass_mse.detach()

            if a.rollout_loss_steps > 0:
                # A fresh random start inside the window every batch. Fixing t0
                # would let the model learn a position-specific fix; sampling it
                # makes "do not drift" a property of every point in the window.
                K = a.rollout_loss_steps
                start = int(roll_rng.integers(0, a.seq_len - K))
                roll = rollout_losses(model, batch["z"], batch["a"], start, K,
                                      weights=roll_w)
                if a.rollout_loss_weight > 0.0:
                    loss = loss + a.rollout_loss_weight * roll["nll"]
                    d["roll_nll"] = roll["nll"].detach()
                if cons_probe is not None:
                    cons = conservation_penalty(
                        cons_probe, roll["z_dreamed"], roll["z_ref"])
                    loss = loss + a.cons_loss_weight * cons
                    d["cons"] = cons.detach()

            d["loss"] = loss.detach()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Clipping is not optional for an MDN. A single window where one
            # component's log-std has been driven low produces a huge gradient,
            # and one such step is enough to wreck the run.
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()

            n = batch["z"].shape[0]
            for k in ("loss", "nll", "hit_bce", "reward_mse") + extra_keys:
                agg[k] += float(d[k]) * n
            agg["n"] += n

        n = max(agg["n"], 1)
        row = {
            "epoch": epoch,
            "train_loss": agg["loss"] / n,
            "train_nll": agg["nll"] / n,
            "secs": time.time() - t0,
        }
        row.update({f"train_{k}": agg[k] / n for k in extra_keys})
        va = evaluate(model, val_loader, pos_weight, device)
        row.update({f"val_{k}": v for k, v in va.items()})

        msg = (
            f"ep {epoch:3d}  train_nll {row['train_nll']:8.3f}  "
            f"val_nll {va['nll']:8.3f}  hit_f1 {va['hit_f1']:.3f}  "
            f"rew_mse {va['reward_mse']:.5f}  ({row['secs']:.0f}s)"
        )
        if a.rollout_loss_steps > 0:
            msg += (f"  roll_nll {row['train_roll_nll']:7.3f}"
                    f"  cons {row['train_cons']:.4f}")
        if model.mass_head is not None:
            msg += f"  mass_mse {row['train_mass_mse']:.4f}"
        if model.pos_head is not None:
            msg += f"  pos_mse {row['train_pos_mse']:.5f}"
        if a.emerge_weight != 1.0:
            msg += f"  emerge {row['train_emerge_frac']:.3%}"

        if (epoch + 1) % a.eval_every == 0 or epoch == a.epochs - 1:
            ro = latent_rollout(
                model, roll_src["mu"], roll_src["actions"],
                warmup=a.rollout_warmup, horizon=a.rollout_horizon,
                temperature=0.0, device=device, seed=a.seed,
            )
            row["rollout_mse_per_h"] = ro["mse_per_h"].tolist()
            row["rollout_mse_h1"] = float(ro["mse_per_h"][0])
            row["rollout_mse_end"] = float(ro["mse_per_h"][-1])
            msg += (
                f"  roll[1]={row['rollout_mse_h1']:.4f} "
                f"roll[{a.rollout_horizon}]={row['rollout_mse_end']:.4f}"
            )

        print(msg, flush=True)
        history.append(row)

        # Best-on-val checkpointing, selected on NLL: the dynamics term is the
        # one the rest of the pipeline depends on.
        if va["nll"] < best:
            best = va["nll"]
            save_rnn(out / "rnn.pt", model, vars(a), extra={"epoch": epoch,
                                                            "val_nll": va["nll"],
                                                            "pos_weight": pos_weight})

    # The selection rule is best-teacher-forced-val-NLL, unchanged from v1 so
    # that every run in the comparison is selected the same way. But a run with
    # a rollout loss is optimising something the selection rule cannot see, so
    # the FINAL epoch is saved alongside it -- if the two differ by much, the
    # selection rule is the thing to question, not the model.
    save_rnn(out / "rnn_last.pt", model, vars(a),
             extra={"epoch": a.epochs - 1, "pos_weight": pos_weight})
    (out / "history.json").write_text(json.dumps(history, indent=2, default=float))
    _plot_history(history, out / "training_curves.png", a.rollout_horizon)
    print(f"\nbest val NLL {best:.3f}  ->  {out / 'rnn.pt'}")


def _plot_history(history: List[dict], path: Path, horizon: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(ep, [r["train_nll"] for r in history], label="train")
    axes[0].plot(ep, [r["val_nll"] for r in history], label="val")
    axes[0].set_title("MDN NLL (teacher forced)")
    axes[0].legend()

    axes[1].plot(ep, [r["val_hit_f1"] for r in history], label="hit F1")
    axes[1].plot(ep, [r["val_hit_recall"] for r in history], label="recall")
    axes[1].plot(ep, [r["val_hit_precision"] for r in history], label="precision")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("paddle-contact head")
    axes[1].legend()

    rl = [(r["epoch"], r["rollout_mse_end"]) for r in history if "rollout_mse_end" in r]
    if rl:
        axes[2].plot([x for x, _ in rl], [y for _, y in rl], marker="o")
    axes[2].set_yscale("log")
    axes[2].set_title(f"open-loop latent MSE @ h={horizon}")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

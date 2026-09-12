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
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .rnn import MDNRNN, RNNConfig, rnn_loss, save_rnn
from .seq_data import LatentSequenceDataset, episode_arrays, make_seq_loader
from .train_vae import pick_device


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
    p.add_argument("--latent-suffix", default="",
                   help='read mu<suffix>.npy instead of mu.npy, e.g. "v1vae"')
    p.add_argument("--ablate-color", action="store_true",
                   help="v2 control: train on latents from an encoder that never "
                        "saw colour (shorthand for --latent-suffix v1vae). See the "
                        "note below on why the ablation is at the DATA level.")
    p.add_argument("--w-hit", type=float, default=1.0)
    p.add_argument("--w-reward", type=float, default=1.0)
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
    )
    model = MDNRNN(cfg).to(device)
    print(f"params={sum(p_.numel() for p_ in model.parameters())/1e3:.0f}k  cfg={cfg}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    # Held-out episodes for the open-loop metric, taken from the first val root.
    roll_src = episode_arrays(val_roots[0], latent_suffix=latent_suffix)

    history: List[dict] = []
    best = float("inf")
    t0 = time.time()

    for epoch in range(a.epochs):
        model.train()
        agg = {"loss": 0.0, "nll": 0.0, "hit_bce": 0.0, "reward_mse": 0.0, "n": 0}
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            parts, _ = model(batch["z"], batch["a"])
            loss, d = rnn_loss(
                model, parts, batch, pos_weight=pos_weight,
                w_hit=a.w_hit, w_reward=a.w_reward,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Clipping is not optional for an MDN. A single window where one
            # component's log-std has been driven low produces a huge gradient,
            # and one such step is enough to wreck the run.
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()

            n = batch["z"].shape[0]
            for k in ("loss", "nll", "hit_bce", "reward_mse"):
                agg[k] += float(d[k]) * n
            agg["n"] += n

        n = max(agg["n"], 1)
        row = {
            "epoch": epoch,
            "train_loss": agg["loss"] / n,
            "train_nll": agg["nll"] / n,
            "secs": time.time() - t0,
        }
        va = evaluate(model, val_loader, pos_weight, device)
        row.update({f"val_{k}": v for k, v in va.items()})

        msg = (
            f"ep {epoch:3d}  train_nll {row['train_nll']:8.3f}  "
            f"val_nll {va['nll']:8.3f}  hit_f1 {va['hit_f1']:.3f}  "
            f"rew_mse {va['reward_mse']:.5f}  ({row['secs']:.0f}s)"
        )

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

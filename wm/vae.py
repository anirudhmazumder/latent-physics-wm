"""Convolutional VAE for 64x64 frames.

Stage one ("V") of the V-M-C world model; used by every tier (v1-v4).
Written up in `docs/02_vae_the_vision_model.md`.

The whole model is ~30 lines of actual architecture. Every non-obvious choice is
annotated inline, because on this problem the architecture is easy and the
*decisions* are where things go wrong.

Shapes, encoder:
      3 x 64 x 64
  ->  c x 32 x 32     conv 4x4 s2 p1
  -> 2c x 16 x 16
  -> 4c x  8 x  8
  -> 8c x  4 x  4
  ->  flatten (8c * 16)
  ->  Linear -> 2 * z_dim      (mu, logvar)

Decoder mirrors it. Default c=16 gives ~570k params total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    z_dim: int = 16          # deliberately overcomplete; true manifold is 3-d
    base_ch: int = 16        # channel width multiplier
    res: int = 64
    logvar_clamp: Tuple[float, float] = (-6.0, 2.0)


class ConvVAE(nn.Module):
    def __init__(self, cfg: VAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or VAEConfig()
        c = self.cfg.base_ch
        z = self.cfg.z_dim

        # ---------------------------------------------------------- encoder
        # kernel 4 / stride 2 / pad 1 halves the resolution exactly. Kernel
        # divisible by stride means every output pixel sees the same number of
        # input pixels -- uneven overlap is what produces checkerboard texture.
        self.enc = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.ReLU(inplace=True),        # 32
            nn.Conv2d(c, 2 * c, 4, 2, 1), nn.ReLU(inplace=True),    # 16
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1), nn.ReLU(inplace=True),#  8
            nn.Conv2d(4 * c, 8 * c, 4, 2, 1), nn.ReLU(inplace=True),#  4
        )
        self.flat_dim = 8 * c * 4 * 4

        # Flatten + Linear, NOT global average pooling. A conv feature map
        # encodes "ball here" as an activation at a spatial location; pooling
        # averages over space and destroys exactly the variable we need. This
        # linear layer is what converts an equivariant spatial map into an
        # explicit positional code, and it is where most of the encoder's
        # parameters live. That is the correct place for them.
        self.to_latent = nn.Linear(self.flat_dim, 2 * z)

        # ---------------------------------------------------------- decoder
        self.from_latent = nn.Linear(z, self.flat_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(8 * c, 4 * c, 4, 2, 1), nn.ReLU(inplace=True),  #  8
            nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1), nn.ReLU(inplace=True),  # 16
            nn.ConvTranspose2d(2 * c, c, 4, 2, 1), nn.ReLU(inplace=True),      # 32
            nn.ConvTranspose2d(c, 3, 4, 2, 1),                                 # 64
        )
        # No skip connections from encoder to decoder, on purpose. They would
        # let information route around the bottleneck and z would learn nothing.
        # The bottleneck IS the architecture here.

    # -------------------------------------------------------------- pieces

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        mu, logvar = self.to_latent(h).chunk(2, dim=-1)
        # Cheap insurance: an early large logvar makes exp() explode and the
        # run dies in the first few hundred steps.
        lo, hi = self.cfg.logvar_clamp
        return mu, logvar.clamp(lo, hi)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        c = self.cfg.base_ch
        h = self.from_latent(z).view(-1, 8 * c, 4, 4)
        # sigmoid because pixels live in [0, 1]. Paired with summed MSE below
        # this is a fixed-variance Gaussian likelihood. We do NOT learn a
        # per-pixel output variance -- that lets the model lower its loss by
        # declaring pixels unpredictable, which is a known instability.
        return torch.sigmoid(self.dec(h))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + eps * sigma. Sampling this way keeps the randomness in `eps`,
        # which has no parameters, so gradients flow through mu and logvar.
        # Sampling from N(mu, sigma) directly would give no gradient path.
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ------------------------------------------------------------------- losses


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.0,
):
    """Returns (loss, parts_dict).

    Detail one: sum over dimensions, mean over the batch only.

    Writing ``F.mse_loss(x_hat, x)`` takes the mean over all 12288 pixel values,
    which silently divides the reconstruction term by 12288 while leaving the KL
    untouched. Effective beta becomes ~12000, the model outputs a uniform grey
    blob, and it presents as "my VAE doesn't work" rather than as a loss
    misweighting. This is the most common VAE bug there is.

    Detail two: ``free_bits``. Each latent dimension gets a budget of `free_bits`
    nats that costs it nothing, implemented as ``clamp(kl_i, min=free_bits)``.
    Below the budget the clamp has zero gradient, so the KL cannot crush that
    dimension to the prior.

    Why this is needed here and not merely nice: at initialisation the decoder
    cannot render a ball, so latent information buys no reconstruction
    improvement yet, while the KL gradient pulling mu -> 0 is immediate and
    strong. Once mu -> 0 the decoder learns to ignore z, and then dL/dz ~= 0
    forever -- there is no gradient left to revive the latent. It is a
    self-reinforcing trap, and on this dataset (a 2%-of-frame object) it is not
    an edge case: plain beta=1 collapses to the mean image reliably.

    Note this is an optimisation failure, not the true optimum. Encoding
    position costs roughly 9 nats and saves well over 70 in reconstruction. The
    model just cannot find its way there unaided.
    """
    recon = F.mse_loss(x_hat, x, reduction="none").flatten(1).sum(-1)  # (B,)

    # Analytic KL between N(mu, sigma^2) and N(0, I), per dimension.
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())      # (B, z)
    kl_pd_mean = kl_per_dim.mean(0)                                    # (z,)

    if free_bits > 0.0:
        # Applied to the batch-averaged per-dimension KL, following Kingma et
        # al. (2016). Doing it per-sample instead would let the clamp fire on
        # individual examples and is a different (weaker) objective.
        kl_objective = kl_pd_mean.clamp(min=free_bits).sum()
    else:
        kl_objective = kl_pd_mean.sum()

    loss = recon.mean() + beta * kl_objective
    parts = {
        "loss": loss.detach(),
        "recon": recon.mean().detach(),
        "kl": kl_pd_mean.sum().detach(),          # true KL, for reporting
        "kl_objective": kl_objective.detach(),    # what we optimise
        "kl_per_dim": kl_pd_mean.detach(),        # (z,) -- the active-units plot
    }
    return loss, parts


def masked_recon_error(
    x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error restricted to a boolean mask, per sample.

    Why this exists: the ball occupies ~1-2% of the frame. A model that renders
    background and paddle perfectly and omits the ball entirely still gets a
    tiny global MSE. So global MSE cannot tell you whether the model kept the
    object. Evaluating inside a box around the true ball position can.
    """
    err = (x_hat - x).pow(2).mean(1)              # (B, H, W), mean over channels
    m = mask.float()
    denom = m.flatten(1).sum(-1).clamp(min=1.0)
    return (err * m).flatten(1).sum(-1) / denom

"""The MDN-RNN: stage two ("M") of the V-M-C world model.

The job is one line: given the latent now and the action now, predict the
latent next. What makes it interesting is the word *predict*.

Why a MIXTURE density and not a regression
------------------------------------------
The obvious model is ``z_{t+1} = f(z_t, a_t)`` trained with MSE. That is a
unimodal Gaussian with fixed variance, and it is wrong here for a concrete
reason: the transition is genuinely multimodal. Two frames before the ball
reaches the floor, whether it bounces off the paddle or off the floor depends on
sub-pixel detail the encoder may not have resolved. The true predictive
distribution has two lumps. An MSE model cannot represent two lumps, so it
predicts the average of them -- a ball in between, which is a place the ball can
never be. In a one-step metric this looks fine (small MSE!) and in a rollout it
is fatal, because the averaged state is off-manifold and the next step is
garbage. Averaging futures is the single classic failure of deterministic
dynamics models, and the mixture is the fix: it can put mass on "bounces" and
mass on "does not" and commit to one when sampled.

``n_gauss=1`` collapses this to exactly the plain-Gaussian baseline (with a
learned per-dim variance), which is why it is worth keeping as a flag: it makes
the MDN-vs-Gaussian comparison a one-character change.

Why RESIDUAL (``predict_delta``)
--------------------------------
z_{t+1} is very close to z_t: the ball moves 0.022 world units per frame, which
is a small perturbation of a 16-d code. Predicting z_{t+1} directly means the
network must spend most of its capacity re-emitting its own input -- learning
the identity map to high precision -- before any of its capacity goes to the
physics. Predicting the *delta* hands it the identity for free and leaves the
network with only the interesting part: the change. In practice this is worth a
large chunk of NLL and it makes early training far better behaved.

Why an LSTM and not a feedforward net
-------------------------------------
Because a single latent does not contain velocity. The VAE sees one frame; a
still image of a ball has a position and no direction of travel (the stage-one
probes confirm this: velocity R^2 ~ 0 from mu). The transition is therefore NOT
Markov in z. The recurrent state is where velocity has to live -- the model must
infer it by integrating consecutive latents -- and ``wm.eval_rnn`` part (c)
tests exactly that claim by probing h_t for velocity.

Heads
-----
    MDN over z_{t+1}    the dynamics
    hit logit           will this transition be a paddle contact?
    reward scalar       dense shaping reward for this transition

The extra heads are cheap (a linear map off the same hidden state) and they are
what makes the model usable as an environment for a controller in stage three:
a controller needs to dream forward AND be told how well it is doing, without
ever touching the real simulator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# log-std is clamped to this range. Lower bound: a mixture component can
# otherwise shrink its std toward zero on a single point and send the NLL to
# -inf (the classic degenerate-mixture blow-up). Upper bound: keeps exp() sane.
# exp(-7) ~ 9e-4, which is far below the scale of any real latent motion, and
# exp(2) ~ 7.4, wider than the latent space itself.
LOGSTD_MIN, LOGSTD_MAX = -7.0, 2.0

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass
class RNNConfig:
    z_dim: int = 16
    n_actions: int = 3
    hidden: int = 256
    n_gauss: int = 5
    predict_delta: bool = True
    ablate_actions: bool = False  # zero the action input; see eval part (d)
    # Optional auxiliary head predicting log(mass) from h. Off by default, so
    # every checkpoint trained before it existed still loads: the field simply
    # takes its default and no parameter is created, leaving the state dict
    # identical. See the note on privileged targets in the Heads section above.
    mass_head: bool = False


class MDNRNN(nn.Module):
    """Single-layer LSTM + mixture-density head, as in Ha & Schmidhuber (2018)."""

    def __init__(self, cfg: Optional[RNNConfig] = None):
        super().__init__()
        self.cfg = cfg or RNNConfig()
        c = self.cfg
        in_dim = c.z_dim + c.n_actions

        # One layer, exactly as in the paper. Depth is not the bottleneck on
        # this problem -- the thing that is hard is carrying velocity through
        # time, which is a recurrence property, not a depth property.
        self.lstm = nn.LSTM(in_dim, c.hidden, num_layers=1, batch_first=True)

        # One linear head emitting all mixture parameters at once:
        #   K logits + K*z means + K*z log-stds
        self.mdn = nn.Linear(c.hidden, c.n_gauss * (1 + 2 * c.z_dim))
        self.hit_head = nn.Linear(c.hidden, 1)
        self.reward_head = nn.Linear(c.hidden, 1)
        # Privileged at TRAINING time only, exactly like the reward head: the
        # target comes from the simulator's state vector, never from anything
        # the model can see at dream time, and nothing downstream reads it.
        # Its job is to force the colour -> mass factor to stay explicitly
        # represented in h rather than being smeared through whatever mixture
        # of latent directions happens to minimise the one-step NLL.
        self.mass_head = nn.Linear(c.hidden, 1) if c.mass_head else None

        # Start with small MDN outputs so the initial predicted delta is ~0,
        # i.e. the model starts life as the identity map. With predict_delta
        # that is already a decent predictor, so training begins from a sane
        # place instead of from random jumps.
        nn.init.zeros_(self.mdn.bias)
        nn.init.normal_(self.mdn.weight, std=1e-3)

    # ------------------------------------------------------------------ core

    def init_hidden(
        self, batch: int, device: str | torch.device = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, self.cfg.hidden, device=device)
        return h, h.clone()

    def _split(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(B, T, K*(1+2z)) -> logits (B,T,K), mean/logstd (B,T,K,z)."""
        B, T, _ = raw.shape
        K, z = self.cfg.n_gauss, self.cfg.z_dim
        logits, mean, logstd = torch.split(raw, [K, K * z, K * z], dim=-1)
        return {
            "logits": logits,
            "mean": mean.view(B, T, K, z),
            "logstd": logstd.view(B, T, K, z).clamp(LOGSTD_MIN, LOGSTD_MAX),
        }

    def forward(
        self,
        z: torch.Tensor,              # (B, T, z_dim)
        a_onehot: torch.Tensor,       # (B, T, n_actions)
        h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if self.cfg.ablate_actions:
            # The ablation is done HERE rather than by not building the input,
            # so the parameter count and every shape stay identical to the main
            # model and the NLL comparison is apples to apples.
            a_onehot = torch.zeros_like(a_onehot)

        x = torch.cat([z, a_onehot], dim=-1)
        out, h = self.lstm(x, h)
        parts = self._split(self.mdn(out))
        parts["hit_logit"] = self.hit_head(out)          # (B, T, 1)
        parts["reward"] = self.reward_head(out)          # (B, T, 1)
        if self.mass_head is not None:
            parts["log_mass"] = self.mass_head(out)      # (B, T, 1)
        parts["h"] = out                                 # (B, T, hidden)
        # Stash z so the delta bookkeeping lives in one place.
        parts["z_in"] = z
        return parts, h

    def step(
        self,
        z_t: torch.Tensor,            # (B, z_dim)
        a_t: torch.Tensor,            # (B, n_actions) one-hot
        h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """One timestep. Returns (parts with T=1 squeezed to T dim kept, h)."""
        parts, h = self.forward(z_t.unsqueeze(1), a_t.unsqueeze(1), h)
        return parts, h

    # -------------------------------------------------------- delta plumbing

    def _abs_mean(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Component means in absolute z space: (B, T, K, z)."""
        if not self.cfg.predict_delta:
            return parts["mean"]
        # The MDN models z_{t+1} - z_t, so add the input back. Note the std is
        # unchanged: a shift does not change the spread.
        return parts["mean"] + parts["z_in"].unsqueeze(2)

    def _target_in_model_space(
        self, parts: Dict[str, torch.Tensor], z_next: torch.Tensor
    ) -> torch.Tensor:
        """Map the target into whatever space the MDN is parameterising."""
        if not self.cfg.predict_delta:
            return z_next
        return z_next - parts["z_in"]

    # ------------------------------------------------------------------ loss

    def mdn_nll(
        self, parts: Dict[str, torch.Tensor], z_next: torch.Tensor
    ) -> torch.Tensor:
        """Negative log likelihood of ``z_next``. Scalar (mean over B and T).

        Per (b, t) the log-likelihood of a diagonal mixture is

            log sum_k pi_k prod_d N(y_d | mu_kd, sigma_kd)
          = logsumexp_k [ log pi_k + sum_d log N(y_d | mu_kd, sigma_kd) ]

        Everything stays in the log domain and the sum over k goes through
        ``logsumexp``. Doing it naively -- exponentiating the per-component
        densities and summing -- underflows to exactly 0 as soon as one
        component is a poor fit in 16 dimensions, and then log(0) = -inf ends
        the run. This is *the* numerical trap in MDNs.

        Convention: sum over the z dimensions, mean over batch and time. Same
        reasoning as ``vae_loss``: a mean over dimensions would silently divide
        the dynamics term by z_dim relative to the auxiliary heads.
        """
        y = self._target_in_model_space(parts, z_next).unsqueeze(2)  # (B,T,1,z)
        mean, logstd = parts["mean"], parts["logstd"]
        var = torch.exp(2.0 * logstd)

        # (B, T, K, z) elementwise Gaussian log-density.
        log_p = -0.5 * (y - mean) ** 2 / var - logstd - LOG_SQRT_2PI
        log_p = log_p.sum(-1)                                         # (B,T,K)

        log_pi = F.log_softmax(parts["logits"], dim=-1)                # (B,T,K)
        ll = torch.logsumexp(log_pi + log_p, dim=-1)                   # (B,T)
        return -ll.mean()

    # ---------------------------------------------------------- sampling etc.

    def most_likely_mean(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean of the highest-weight component, in absolute z. (B, T, z).

        NOT the mixture mean sum_k pi_k mu_k -- that is the averaging failure
        this whole file exists to avoid. For a deterministic rollout you want a
        point the model considers likely, and the mixture mean can sit in a
        valley between two modes.
        """
        k = parts["logits"].argmax(-1)                                  # (B,T)
        mean = self._abs_mean(parts)                                    # (B,T,K,z)
        idx = k[..., None, None].expand(-1, -1, 1, mean.shape[-1])
        return mean.gather(2, idx).squeeze(2)

    def sample_next(
        self,
        parts: Dict[str, torch.Tensor],
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw z_{t+1} ~ mixture, with the paper's temperature. (B, T, z).

        tau scales the component log-stds by tau and the mixture logits by
        1/tau. tau = 1 is the model's honest predictive distribution; tau < 1
        sharpens both the component choice and the within-component spread;
        tau -> 0 degenerates to "the mean of the most likely component", which
        is what ``temperature=0`` returns exactly (no sampling at all, so the
        rollout is deterministic and reproducible).

        Why you would ever want tau != 1: a dream rolled out at tau = 1
        accumulates the model's own noise and drifts off-manifold; lowering tau
        gives a cleaner but less diverse dream. Ha & Schmidhuber use tau > 1 for
        the opposite reason -- a more uncertain dream makes a controller trained
        inside it less able to exploit the model's flaws.
        """
        if temperature <= 0.0:
            return self.most_likely_mean(parts)

        logits = parts["logits"] / temperature
        # Gumbel-max: argmax(logits + Gumbel noise) is an exact categorical draw
        # and needs no loop over the batch.
        u = torch.rand(logits.shape, device=logits.device, generator=generator)
        g = -torch.log(-torch.log(u.clamp_min(1e-20)).clamp_min(1e-20))
        k = (logits + g).argmax(-1)                                    # (B,T)

        mean = self._abs_mean(parts)                                   # (B,T,K,z)
        std = torch.exp(parts["logstd"]) * temperature
        idx = k[..., None, None].expand(-1, -1, 1, mean.shape[-1])
        m = mean.gather(2, idx).squeeze(2)
        s = std.gather(2, idx).squeeze(2)
        eps = torch.randn(m.shape, device=m.device, generator=generator)
        return m + eps * s


# ------------------------------------------------------------- combined loss


def rnn_loss(
    model: MDNRNN,
    parts: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    pos_weight: float = 1.0,
    w_hit: float = 1.0,
    w_reward: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """NLL + weighted BCE(hit) + MSE(reward). Returns (loss, parts_dict).

    ``pos_weight`` matters more than it looks. Paddle contact is ~1% of
    transitions, so the unweighted BCE optimum is very close to "always say no",
    which scores a fine loss and a recall of zero. ``pos_weight = (1-p)/p``
    re-balances the two classes so the head is actually forced to find the
    positives; the cost is that the raw probabilities come out over-confident,
    which is why eval reports precision/recall/PR-AUC rather than accuracy.
    """
    nll = model.mdn_nll(parts, batch["z_next"])

    hit_logit = parts["hit_logit"].squeeze(-1)
    bce = F.binary_cross_entropy_with_logits(
        hit_logit,
        batch["hit"],
        pos_weight=torch.as_tensor(pos_weight, device=hit_logit.device),
    )

    rew = parts["reward"].squeeze(-1)
    mse = F.mse_loss(rew, batch["reward"])

    loss = nll + w_hit * bce + w_reward * mse
    return loss, {
        "loss": loss.detach(),
        "nll": nll.detach(),
        "hit_bce": bce.detach(),
        "reward_mse": mse.detach(),
    }


# ------------------------------------------------------------------ ckpt i/o


def save_rnn(path, model: MDNRNN, args: Optional[dict] = None, extra: Optional[dict] = None):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "cfg": asdict(model.cfg),
        "args": args or {},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_rnn(path, device: str = "cpu") -> Tuple[MDNRNN, RNNConfig]:
    ck = torch.load(path, map_location=device, weights_only=False)
    fields = set(RNNConfig.__dataclass_fields__)
    cfg = RNNConfig(**{k: v for k, v in ck["cfg"].items() if k in fields})
    model = MDNRNN(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg

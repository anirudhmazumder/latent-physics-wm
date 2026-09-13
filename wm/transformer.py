"""A causal transformer as the dynamics model M. The v4 comparison.

Everything about this file exists to make one sentence testable:

    the LSTM has to *carry* the gravity sign forward through every update
    between the flip and now; the transformer can *look back* at the frame
    where the flip happened.

That is the standard story about recurrence versus attention, and v4 is built
so that it can be measured rather than asserted: the hidden latent is set by a
single visible event (a paddle contact), it is invisible in any one frame
afterwards, and it stays in force for 100+ frames. A memory that decays with
distance from the event and one that does not should look different here if
they ever do.

What is shared, and why that matters
------------------------------------
The MDN head, the hit head, the reward head, the mixture likelihood, the delta
parameterisation, the temperature sampler -- all of it comes from
``wm.rnn.MDNDynamics``, unchanged and not reimplemented. A comparison between
two architectures is worth nothing if the two also differ in their output
distribution, their loss or their sampler, and the cheapest way to guarantee
they do not is to have exactly one copy of that code. The only thing this file
supplies is the map from a history of (z, a) pairs to a vector per timestep.

The sliding window, and the one design decision in here
-------------------------------------------------------
The model attends over the last ``context`` inputs. Positions are embedded by
their offset *from the start of the visible window*, not by absolute time in
the episode -- an episode is 200 or 600 frames long and the table has only
``context`` rows, so absolute time is not an option, and a window-relative
index is the only scheme under which the incremental ``step`` path and the
parallel ``forward`` path compute the same function.

Keeping those two paths identical is worth some care, because they are used for
different things and the whole evaluation assumes they agree:

    ``forward``  teacher-forced training and teacher-forced probing. One
                 parallel pass with a causal mask; O(T^2) attention, O(1)
                 passes.
    ``step``     open-loop dreaming, where z_{t+1} is the model's own output
                 and there is nothing to parallelise. Appends to the buffer,
                 drops the oldest input if the buffer is full, re-attends.

When a call to ``forward`` would need more than ``context`` inputs to serve all
of its output positions (a warm-up buffer plus a long window, or a 600-frame
episode in one go), it cannot do it in one parallel pass without either
lengthening the position table or changing what each position sees. Rather than
quietly approximating, it falls back to looping ``step``. That is slower and
exactly correct, and it is why ``tests/test_transformer_v4.py`` can assert the
two paths agree to 1e-5 without qualification.

The carried state
-----------------
``h`` is a 3-tuple::

    (h_last, c_dummy, buf)
     (1,B,D)  (1,B,D)   (B, L<=context, z+a)

The first two entries exist so that every call site written for an LSTM --
``h[0][0]`` for "the carried state", ``h[1][0]`` for "the cell state" -- keeps
working with no branch. There is no cell state in a transformer, so the second
entry is zeros; a probe fitted on it reports exactly the null, which is the
honest answer. ``buf`` is the actual state: the raw inputs the next step will
attend over.

Parameter count
---------------
The default ``d_ff = 2 * d_model`` rather than the usual 4x, to bring the total
nearer the LSTM's 327k. It does not get all the way there (see
``wm/README_M4.md`` for both numbers); the transformer is the larger model, so
a transformer *loss* on the memory question is the robust direction of the
comparison and a transformer *win* carries a size caveat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn import MDNDynamics


@dataclass
class TransformerConfig:
    z_dim: int = 16
    n_actions: int = 3
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    context: int = 128
    n_gauss: int = 5
    predict_delta: bool = True
    dropout: float = 0.0
    # None means ``2 * d_model``; see the note on parameter count above.
    d_ff: Optional[int] = None
    ablate_actions: bool = False
    # The auxiliary heads. Present so that ``MDNDynamics._build_heads`` can read
    # the same field names off either config; none of them is used in v4.
    mass_head: bool = False
    pos_head: bool = False
    clock_head: bool = False
    vy_head: bool = False
    # Unused here, but ``wm.eval_permanence_v3`` asks every model whether it is
    # the feed-forward control. A transformer is not.
    feedforward: bool = False
    arch: str = "transformer"

    @property
    def hidden(self) -> int:
        """The width of ``parts["h"]``, under the name every caller uses.

        A property and not a field on purpose: ``asdict`` skips properties, so
        the saved config has exactly one place where the width is recorded
        (``d_model``) and the two can never disagree.
        """
        return self.d_model

    @property
    def ff_dim(self) -> int:
        return self.d_ff if self.d_ff is not None else 2 * self.d_model


class _Block(nn.Module):
    """Pre-norm self-attention + MLP, the standard modern arrangement.

    Pre-norm (normalise *before* each sublayer, add the residual after) rather
    than the original post-norm, because pre-norm trains without a learning-rate
    warm-up schedule. With 4 layers this is not a make-or-break choice; it is
    simply one fewer hyper-parameter to get wrong, and this run has no budget
    for a schedule sweep.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        if self.d_head * cfg.n_heads != cfg.d_model:
            raise ValueError(
                f"d_model {cfg.d_model} not divisible by n_heads {cfg.n_heads}")
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_dim),
            nn.GELU(),
            nn.Linear(cfg.ff_dim, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        # (B, heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(mask, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TransformerDynamics(MDNDynamics):
    """Causal transformer over the last ``context`` (z, a) pairs."""

    def __init__(self, cfg: Optional[TransformerConfig] = None):
        super().__init__()
        self.cfg = cfg or TransformerConfig()
        c = self.cfg
        self.in_proj = nn.Linear(c.z_dim + c.n_actions, c.d_model)
        # Learned absolute position *within the visible window*. See the module
        # docstring: this is what makes step and forward the same function.
        self.pos = nn.Parameter(torch.zeros(c.context, c.d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(_Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self._build_heads()

        # A single upper-triangular boolean, sliced per call. Registered as a
        # buffer so it moves with ``.to(device)`` and is excluded from the saved
        # state dict (``persistent=False``) -- it is derived, not learned, and
        # keeping it out means a checkpoint stays loadable if ``context`` ever
        # changes shape.
        mask = torch.triu(torch.ones(c.context, c.context, dtype=torch.bool), 1)
        self.register_buffer("causal_mask", mask, persistent=False)

    # ------------------------------------------------------------------ core

    def init_hidden(self, batch: int, device: str | torch.device = "cpu"):
        """An empty context buffer, dressed as an LSTM's (h, c).

        See the module docstring for why there are three entries. The buffer
        starts with zero inputs in it, not with ``context`` zeros: a zero input
        is a *statement* ("the latent was 0 and the action was none"), and
        padding the window with it at the start of an episode would feed the
        model 128 frames of a fictional motionless world.
        """
        c = self.cfg
        d = torch.zeros(1, batch, c.d_model, device=device)
        buf = torch.zeros(batch, 0, c.z_dim + c.n_actions, device=device)
        return d, d.clone(), buf

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L<=context, in_dim) raw inputs -> (B, L, d_model) outputs."""
        L = x.shape[1]
        if L > self.cfg.context:
            raise ValueError(f"window of {L} > context {self.cfg.context}")
        y = self.drop(self.in_proj(x) + self.pos[:L])
        mask = self.causal_mask[:L, :L]
        for blk in self.blocks:
            y = blk(y, mask)
        return self.ln_f(y)

    def _pack(self, out_last: torch.Tensor, buf: torch.Tensor):
        d = out_last.unsqueeze(0)
        return d, torch.zeros_like(d), buf

    def forward(
        self,
        z: torch.Tensor,              # (B, T, z_dim)
        a_onehot: torch.Tensor,       # (B, T, n_actions)
        h=None,                       # (h_last, c_dummy, buf) or None
    ):
        if self.cfg.ablate_actions:
            # Zeroed rather than removed, exactly as in MDNRNN: the parameter
            # count and every shape stay identical, so the NLL comparison is
            # apples to apples.
            a_onehot = torch.zeros_like(a_onehot)
        x = torch.cat([z, a_onehot], dim=-1)                    # (B, T, in_dim)
        prev = h[2] if h is not None else None
        n_prev = 0 if prev is None else int(prev.shape[1])
        T = x.shape[1]

        if n_prev + T <= self.cfg.context:
            # The fast path, and the one training always takes.
            full = x if n_prev == 0 else torch.cat([prev, x], dim=1)
            out = self._backbone(full)[:, n_prev:]              # (B, T, d_model)
            return self._heads(out, z), self._pack(out[:, -1], full)

        # The slow, exactly-equivalent path: some output position needs a window
        # that starts after the buffer's own start, so there is no single set of
        # window-relative positions that serves every step at once. Loop.
        outs, hh = [], h
        for t in range(T):
            xt = x[:, t : t + 1]
            prev_t = hh[2] if hh is not None else None
            full = xt if prev_t is None else torch.cat([prev_t, xt], dim=1)
            full = full[:, -self.cfg.context :]
            o = self._backbone(full)[:, -1]                     # (B, d_model)
            outs.append(o)
            hh = self._pack(o, full)
        out = torch.stack(outs, dim=1)
        return self._heads(out, z), hh

    def step(self, z_t: torch.Tensor, a_t: torch.Tensor, h=None):
        """One timestep, with the buffer truncated to the last ``context``.

        Not inherited from ``MDNDynamics``: the generic version would route
        through ``forward``'s fast path, which does not truncate, and the buffer
        would grow without bound over a long dream until it ran off the end of
        the position table.
        """
        if self.cfg.ablate_actions:
            a_t = torch.zeros_like(a_t)
        x = torch.cat([z_t, a_t], dim=-1).unsqueeze(1)          # (B, 1, in_dim)
        prev = h[2] if h is not None else None
        full = x if prev is None else torch.cat([prev, x], dim=1)
        full = full[:, -self.cfg.context :]
        out = self._backbone(full)[:, -1:]                      # (B, 1, d_model)
        return self._heads(out, z_t.unsqueeze(1)), self._pack(out[:, -1], full)

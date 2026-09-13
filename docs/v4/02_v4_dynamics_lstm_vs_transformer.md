# v4 — 02: M. Does anything remember the flip? LSTM versus transformer

*The tier's question, asked properly, with a null and an intervention — and a
clear negative for both architectures.*

---

## 1. The setup

Same recipe as every tier, plus one new model. `wm/transformer.py` is a causal
transformer over the last 128 latents and actions (4 layers, width 128, 4
heads) with the **same** mixture-density, contact and reward heads as the
LSTM — the heads were factored out into a shared class so the two models
differ in nothing but the sequence module. It is a drop-in: `step` and the
parallel `forward` are the same function (tested to 1e-5, including across the
context boundary), and every existing evaluation runs on it unchanged.

| model | parameters | val NLL | contact PR-AUC | reward-head R² |
|---|---|---|---|---|
| LSTM (`rnn_v4`) | 327k | 2.50 | 0.55 | 0.90 |
| transformer, context 128 (`tf_v4`) | 571k | 1.87 | 0.59 | 0.95 |
| **transformer, context 32** (`tf_v4_ctx32`) | 558k | **1.78** | **0.63** | **0.96** |
| feed-forward floor | 114k | 5.39 | 0.38 | 0.83 |
| LSTM, actions ablated | 327k | 3.29 | 0.49 | 0.89 |

Two things before the sign experiments. The transformer is the better
one-step predictor by every ordinary measure. And the context-32 variant —
built as the control that *cannot* see back to a flip ~75 frames ago — is the
best model of all, which is already a hint about what follows.

**The VAE's negative result held.** The sign is not decodable from a single
frame: logistic 0.36, kNN 0.40, both *below* a shuffled-label null of 0.50
(episode-level split; small-sample nulls are noisy, hence the below-chance
values). Position-matched probes cap at 0.56 via the ball's height distribution
alone. Whatever a model knows about the sign, it did not read it off a frame.

## 2. Experiment (a): is the sign in `h`, and for how long?

Linear probe from `h` to the current sign, held-out episodes, as a function of
frames since the last flip.

![switch comparison](../../runs/v4_switch_comparison.png)

Two features of the left panel decide the tier.

**The curve rises.** A memory of the flip would be perfect at 0–10 frames and
decay. Every model is at chance for the first ten frames and *climbs* to a peak
around 25–50 frames. That is the signature of **inference from motion**: the
sign becomes readable once the trajectory has bent enough (doc 00's SNR
argument put that at 20–40 frames), and the models read it then.

**The single-frame null is not flat.** `z` alone rises with frames-since-flip
too, because frames far from a flip are frames the ball spent high, and height
correlates weakly with sign. So the honest quantity is what `h` adds over `z`
(middle panel): about +0.28 for the LSTM at 25–50 frames, +0.20 for the
context-32 transformer, +0.09 for the context-128 transformer, ~0 for the
feed-forward floor. The recurrent and attentional states *do* carry the sign,
and the LSTM carries it best.

## 3. Experiment (b): memory of the event, or inference from the motion?

The right panel splits it: accuracy in the first ten frames after a flip
(only memory of the contact could give this) versus accuracy 50+ frames later
on episodes whose warm-up began *after* the flip (only the trajectory could).

| | memory (0–10 frames after the flip) | inference (50+ frames, never saw the contact) |
|---|---|---|
| LSTM | 0.52 | 0.73 |
| transformer | 0.52 | 0.70 |
| transformer, context 32 | 0.58 | 0.76 |
| feed-forward | 0.52 | 0.64 |
| one frame alone | 0.60 | 0.75 |

The memory column is at chance for every model, and **below what one frame
gives**. No model retains the contact as a bit. The inference column is high
for all — including, at 0.75, the single frame — so the "recall" seen in (a) is
mostly trajectory reading, with the recurrent state adding a modest amount of
integration on top.

## 4. Experiments (c)–(e): dreams, the intervention, and half-lives

**Dream curvature.** Dream 60 steps from a flip-free warm-up, fit the vertical
acceleration of the decoded path. The instrument is sound (true states 1.00
sign agreement; decoded true latents 0.88). The models: LSTM 0.60,
transformer 0.59, action-ablated 0.67, feed-forward 0.40 — barely above
chance, with magnitudes 1.3–2× too large. The dreams bend, but not reliably
the right way.

**The flip counterfactual** — the interventional test. Take a real contact,
build a twin warm-up re-simulated with the paddle moved so the ball *misses*
(and the reverse), dream both, compare the dreamed accelerations' signs. A
model that knows contact flips gravity produces opposite curvatures.

| model | opposite-sign fraction, dream from 4 frames after contact | from 1 frame after |
|---|---|---|
| LSTM | 0.47 | 0.50 |
| transformer | 0.41 | 0.46 |
| transformer, ctx 32 | 0.25 | 0.29 |
| feed-forward | 0.43 | 0.41 |
| chance | 0.50 | 0.50 |

**Nothing is above chance.** The one-frame variant removes the confound that a
model might read the post-contact *speed* rather than the contact: it makes no
difference. The context-32 transformer sits *below* chance because its dreams
have a strong preferred curvature the intervention does not move.

**Half-lives.** Undefined for every model — not because recall is flat, but
because the curve has the wrong shape (chance → peak → plateau). Sign
consistency along a 200-step sampled dream is 0.44–0.55 at every temperature:
the dreamed sign is a coin flip from the start.

## 5. Verdict

**The tier's question was not answered because it was never reached.** The
question was "does a transformer that can attend to the contact frame remember
the flip where an LSTM that must carry the bit forgets it". Neither remembers
the flip. Both infer the current sign from the trajectory's curvature, and on
that route the LSTM is slightly ahead, while the transformer is the better
predictor of everything else. The context-32 control — the one that could not
possibly see back to the flip — is the best model in the table, which says
directly that seeing back to the flip was never used.

Why, in the terms every earlier tier has taught: the one-step loss pays for
the sign only through its effect on the next latent, which is 1e-4 — the
smallest per-frame payoff in the project, spread over hundreds of frames. The
cheapest way to get most of that payoff is to read the curvature from the
recent window, which both architectures can do and both did. Storing a bit at
a rare event and holding it indefinitely is a strictly harder solution to the
same objective, and nothing in the objective rewards the difference. This is
the v3 mechanism (delayed credit is under-learned) at its limit: not delayed,
but *diffuse*.

## 6. Caveats

One seed per model. The transformer has 1.7× the LSTM's parameters (a
width-96 variant would match; not trained). The context-32 run also shortened
the training window to 32, so it varies two things. `h` is 256-wide for the
LSTM and 128 for the transformer, which favours the LSTM on a linear probe. The
sign probe needed `data/v4/train` in its pool to have enough flips to show
structure — that is the dynamics model's own training data, a defensible
choice for a representation probe but worth knowing. Bins beyond 200 frames
hold 112 frames and the hit→miss counterfactual ~23 pairs.

Next: [03 — v4 results, and the project after four tiers](03_v4_results_and_retrospective.md).

# 05 — Results, lessons, and what comes next

*The whole v1 in one place: what was built, what each stage demonstrated, what
surprised us, and the concrete next experiments.*

---

## 1. What was built

A complete Ha & Schmidhuber-style world model of a paddle-and-ball game,
learned from 64×64 pixels, in three frozen-in-sequence stages:

| stage | model | parameters | trained on | wall clock (M1 laptop) |
|---|---|---|---|---|
| **V** | convolutional VAE, 16-d latent | ~570k | 30k shuffled frames | 54 min (GPU) |
| **M** | LSTM-256 + 5-component mixture head + contact & reward heads | ~330k | 90k latent transitions, 35 epochs | 11 min (CPU) |
| **C** | one linear layer, 819 params | 819 | 15M dreamed steps, CMA-ES | 11 min (CPU) |

Plus: an environment with ground-truth state for grading, two data-collection
policies, latent caching, a probe suite, dream-rollout evaluation with
world-coordinate decoding, counterfactual dreams, a batched dream environment,
a real-environment evaluation harness with baselines and an oracle, 23 unit
tests, and these documents.

## 2. What each stage demonstrated

### V — the frame code
- Ball and paddle **positions** are recoverable from the 16-d code to ~1% of
  the box (kNN R² 0.99).
- **Velocity is not** (R² ≈ 0), and that is the correct answer: a single frame
  contains no direction of travel. This is the reason the recipe has a memory
  stage at all.
- The code is a **distributed, place-field-like** representation, not a set of
  coordinates: linear probes fail (R² 0.04–0.2), nonlinear ones succeed, and the
  tuning maps show blobs and stripes rather than ramps.
- Posterior collapse is the default failure on small-object worlds; free bits
  plus KL warm-up prevented it.

### M — the dynamics
- **Velocity appears in the hidden state.** Linear R² for ball velocity: 0.02
  from `z`, **0.92** from `h`. Nobody told the LSTM about velocity; storing it
  is simply what it takes to predict the next code.
- The LSTM also **linearised position**: from `h`, a linear probe recovers
  ball position at R² 0.995, where from `z` it needed a nonlinear probe.
  Representations are shaped by what they are used for.
- **Actions have effects in the dream.** Always-left vs always-right dreams
  separate the imagined paddle by 0.66 of the box and saturate at exactly the
  simulator's clamp limits. The action-ablated control separates by 0.00.
- **Contacts are anticipated**: predicted probability rises from a 0.7% base
  rate to 35% three frames before a real contact and 92% one frame before.
- **Useful dream horizon: 35 frames** (deterministic rollout) before the
  imagined ball drifts more than one radius from the real one. Error grows
  linearly, as a small velocity error integrated over time should.
- Walls: better than chance (36% vs 16%) but not mastered, on a crude test.

### C — the agent
- **A policy trained on zero real frames plays the real game at 85–90% of a
  privileged-state oracle** (1.4–1.5 interceptions per episode vs 1.55–1.63;
  standing still gets 0.93).
- **Without `h` the policy is no better than standing still**, and its dream
  score climbs just as smoothly — the canonical exploitation failure, caught
  only because real return was logged alongside dream return.
- A policy trained on **1,024,000 real frames** with the same optimiser was not
  better than the dream-trained one trained on **0**.
- The dense reward head (`1 − |ball_x − paddle_x|`) trained better policies
  than the sparse, class-reweighted contact head; the contact head is the
  exploitable one.

## 3. How the pieces fit — the causal chain the model learned

```
pixels ──V──► z  (where things are)
                │
   z_t, a_t ────M──► h  (where things are AND how they move; paddle limits;
                │        "a contact is coming")
                │
   [z, h] ──────C──► action  (sweep the paddle toward where the ball is)
```

Read bottom-up, the agent's competence is *explained*: C is linear, so it can
only act on information already present in `[z, h]`; that information is
velocity, which M put there because it needed it to predict frames, from codes
that V produced knowing nothing about time. Each stage's contribution is
separately measured and separately attributable — which is the practical payoff
of the staged, frozen recipe.

## 4. Surprises and lessons (the things worth carrying to v2)

1. **Global pixel loss lies.** On a world where the object is 2% of the frame,
   a model that drops the object entirely has a tiny loss. Track masked error;
   grade in world coordinates via a probe; look at the pictures.
2. **The mixture head did not measurably help.** Textbook reasoning (averaging
   multimodal futures is fatal) was correct in principle and did not matter in
   practice: the Gaussian baseline matched the mixture on likelihood and
   rolled out *more* accurately. Run the ablation before believing the story.
3. **Teacher-forced loss keeps improving after dreams stop improving.** Val NLL
   fell for 20 epochs during which open-loop rollout error was flat. Put the
   rollout metric in the training loop.
4. **Sequence-evaluation alignment bugs are silent.** An off-by-one in the
   dream evaluation inflated every horizon error by one frame of motion. It
   was caught only because the one-step error was implausibly far above a
   floor we trusted. Always compare to a floor.
5. **Dream return is not a success metric.** The `z`-only controller's dream
   return rose exactly like the good controller's while its real performance
   fell. Plot both curves, every time.
6. **A learned policy beating an oracle means the metric is broken.** The
   real-trained controller "beat" the oracle by registering several contact
   frames per catch. Count interceptions.
7. **Sample-size honesty.** Frames within an episode are near-duplicates:
   probe on many short episodes and split by episode. Contact counts on 100
   episodes carry ±0.3 confidence intervals; the decision-level agreement
   statistic was stable to ±0.02 and is the better primary metric.
8. **Higher dream temperature is a trade-off, not a fix.** τ = 1.5 improved
   dream-to-real correlation and worsened the final policy.
9. **Report the selection procedure with the number.** "Best of 40 real
   checks" and "final CMA-ES mean" differ by up to 0.9 contacts and by
   200,000 real frames of cost.
10. **CPU beat GPU** for the small LSTM by 4×: dispatch latency, not FLOPs,
    was the bottleneck. Measure before assuming.

## 5. Known limitations of v1

- The VAE's latent space is nonlinear (place-field-like). M coped, but a
  coordinate-like latent might dream further and be easier to inspect.
- The wall-bounce test is crude (any-axis reversal). The dream visibly
  bounces off the floor; the number under-sells it.
- The sampled (τ > 0) dream diverges from the real episode within a few
  frames because M was trained on posterior samples and correctly models that
  spread. Whether training on posterior means narrows the dream without
  hurting the controller is untested.
- Only 51 real paddle contacts in the validation set; the contact-anticipation
  curve has wide error bands.
- `ctrl_real` was given a smaller population than the dream runs to fit a time
  budget, so the real-vs-dream training comparison is directional, not exact.
- The controller is linear and does not lead the ball; whether a nonlinear C
  would is untested.

## 6. Next experiments, in order of value per hour

1. **Iterate the loop** (the paper's own suggestion): collect data with the
   trained controller, retrain M, retrain C. Does the useful horizon grow? Does
   transfer improve? This is v1.1 and needs no new code beyond a `--policy ctrl`
   in the collector.
2. **Gaussian vs mixture, properly.** Train both to convergence, evaluate on
   posterior *samples* as well as means, and dream at several τ. Settle
   whether the mixture earns its place here.
3. **A per-wall bounce test.** Condition on which wall, check the correct
   axis's sign. Cheap and it would sharpen the weakest result.
4. **`--use-mean` training for M**, then compare τ = 1 dream horizons.
5. **A disentangled V** (β > 1 or a total-correlation penalty) and re-run every
   probe: does a coordinate-like `z` make M's job easier?
6. **Then v2**: mass-from-colour. A ball colour sampled at reset and mapped to
   mass, so an *appearance* variable has a *dynamical* consequence that cannot
   be read from a single frame's motion. The whole v1 tooling — probes,
   counterfactual dreams, transfer curves — carries over unchanged; the new
   question is whether M learns the colour → bounce causal edge, which
   `eval_rnn`'s counterfactual machinery can test directly by recolouring the
   ball in a dream.

## 7. Reproducing

<a name="reproducing"></a>

Interpreter: the conda environment `NN`
(`/opt/miniconda3/envs/NN/bin/python`; Python 3.10, PyTorch 2.12, numpy,
scikit-learn, matplotlib, Pillow, scipy, `cma`). All commands from the repo
root. The exact sequence with wall-clock times is in
[`wm/README_M.md`](../wm/README_M.md) and [`wm/README_C.md`](../wm/README_C.md);
doc 00 §7 has the condensed version. Tests:

```bash
python -m tests.test_rnn && python -m tests.test_controller
```

Everything referenced in these documents lives under `runs/`; nothing was
edited by hand.

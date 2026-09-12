# v3 — 03: Trying to fix horizontal permanence

*Three ways to make the model carry the hidden ball's x, one privileged ceiling
that shows it can be carried, and an honest negative result about the fair
ones.*

---

## 1. The target

Doc 02 diagnosed the defect: the recurrent state holds the hidden ball's `y`,
`vy` and time-since-hidden, and not its `x` or `vx`, because one-step teacher
forcing pays nothing for `x` until the exit frame. The bar for a fix, set in
advance: hidden-frame `ball_x` R² from `h` clearly above 0.17, **and** dreamed
exit-x error below the no-memory baseline (0.146 box widths, "the ball is where
it vanished"), **and** below linear extrapolation on runs with a hidden wall
bounce, without regressing the ordinary dream quality by more than 10%.

## 2. The candidates

All trained from scratch, same data and budget as `rnn_v3`, VAE frozen. Code
in `wm/train_rnn.py` (`--rollout-loss-steps 24`, `--emerge-weight`,
`--pos-head`) and `wm/conservation.py`.

| candidate | mechanism | the bet |
|---|---|---|
| **(a) `ms24`** | 24-step open-loop rollout loss (the v2 machinery, tripled in length) | the exit frame now lies inside the rollout window for most occlusions, so carrying `x` finally pays |
| **(b) `emerge`** | (a) plus a 5× weight on the likelihood of the first three frames after a re-emergence | those frames are the only ones that reveal hidden `x`, and they are one in ten; up-weight them so their gradient competes. Uses the simulator's visibility flag to *weight* the loss — a label, not an input |
| **(c) `poshead`** | a linear head on `h` supervised with the true ball position on every frame, hidden ones included | **privileged.** Not a fair world-model result; the ceiling that tells us whether the recurrent state *can* hold `x` if simply told to |
| (d) `ms24 + poshead` | both | — |
| (extra) `emerge_only` | the emergence weighting without the long rollout | to decompose (b) |

## 3. Results

![comparison](../../runs/rnn_v3_fix_comparison.png)

| model | hidden `x` R² from `h` | hidden `vx` R² | exit-x MAE (default band) | runs with no re-emergence | exit-time MAE | side correct | contact PR-AUC | val NLL |
|---|---|---|---|---|---|---|---|---|
| `rnn_v3` (baseline) | 0.17 | −0.07 | 0.159 | **2%** | **3.2** | **0.92** | **0.89** | **3.97** |
| (a) `ms24` | 0.42 | −1.47 | 0.176 | 20% | 5.9 | 0.83 | 0.80 | 4.98 |
| (extra) `emerge_only` | 0.11 | −0.13 | 0.179 | 31% | 5.0 | 0.81 | 0.88 | 4.07 |
| **(b) `emerge`** | **0.44** | **+0.30** | 0.152 | 24% | 5.0 | 0.87 | 0.73 | 4.96 |
| *(c) `poshead`, privileged* | *0.71* | *0.53* | *0.115* | *33%* | *4.0* | *0.79* | *0.90* | *3.96* |
| no-memory baseline | | | 0.146 | | | | | |
| linear extrapolation from entry | | | 0.034 | | | | | |

**The privileged ceiling is the headline.** With 514 extra parameters and a
supervised position target, the recurrent state holds the hidden ball's `x` at
R² 0.71 and `vx` at 0.53, the dreamed exit lands within 0.115 — the only model
that beats the no-memory baseline — and the ordinary likelihood is *better*
than the baseline's. The LSTM can carry the ball through the band perfectly
well. It was never asked to. **This is an objective problem, not a capacity or
architecture problem.**

**The best fair fix gets halfway and pays for it.** `emerge` more than doubles
the horizontal position information in `h` (0.17 → 0.44, half the gap to the
ceiling) and is the only fair model with any horizontal *velocity* in `h`
(−0.07 → +0.30, 62% of the gap). But its dreamed exit position does not beat
the no-memory baseline once censoring is accounted for (it fails to re-emerge a
ball on a quarter of runs, and the runs it does score are the easier ones), its
contact prediction drops 18%, and its likelihood costs a full nat. The
decomposition is clean: weighting alone does nothing (`emerge_only`, 0.11), the
long rollout alone gets `x` but wrecks `vx` (`ms24`, −1.47 — a real number,
checked twice), and only the combination produces a usable `vx`.

**The 24-step rollout, on its own, is a bad trade** here: it triples the
per-epoch cost, costs a nat of likelihood, and degrades exit timing, side,
contact prediction and vertical memory, for a gain in `x` that does not reach
the dreamed exit. This is the second time the multi-step loss has disappointed
on this world (doc 02 tried 8 steps), and the reason is now clearer: rolling
open-loop through a stretch whose target is the blank band teaches nothing
about the ball, and lengthening the stretch mostly adds noise.

## 4. The verdict for stage C

The checkpoint handed to the controller stage is still the baseline,
`runs/rnn_v3/rnn.pt`: it wins exit timing, side, contact prediction,
conservation and completeness, which are what a controller needs most. The
feed-forward model is the floor. And the privileged `poshead` model goes along
**as a labelled ceiling**: a controller trained inside its dream tells us what
horizontal permanence would be *worth* for play, which is a question the fair
models cannot yet answer.

## 5. Lessons

- **Ceilings are worth building.** One privileged run settled in twenty
  minutes what three fair runs could not: the model *can* hold the state; the
  loss never asks. Without the ceiling, the fair failures would have been
  ambiguous between "impossible" and "unrewarded".
- **Delayed credit is the enemy of one-step prediction.** Information that
  pays off ten frames later is systematically under-learned. Long rollouts are
  the textbook remedy and they were, on this world, mostly noise; weighting
  the frames that carry the credit worked better, and only in combination.
- **Censoring is a metric trap.** A model that declines to re-emerge a ball on
  the hard runs looks good on the runs it scores. Always report the fraction
  scored next to the error.
- **Second time round, the general fix stopped being general.** The multi-step
  loss was v2's clean win and v3's clean loss. The difference is what the
  open-loop target looks like during the stretch that matters.
- **A cheap next step exists.** `ball_visible` is readable from `z` at R² 0.98,
  so the emergence weighting could come from a frozen probe on the latent
  rather than the simulator's flag, making (b) fully self-supervised. Not done.

Next: [04 — the v3 controller: acting on memory](04_v3_controller.md).

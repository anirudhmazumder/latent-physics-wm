# v3 — 05: Results and lessons

*The occlusion tier in one page.*

---

## 1. What v3 asked

Can a world model carry an object's state through a gap in observation — and
does an agent need it to? An opaque band hides the ball for ~10 frames on every
vertical traverse; nothing else changes.

## 2. What came out

| stage | question | answer |
|---|---|---|
| **V** | Behaves correctly? | Yes. Position from a frame: R² 0.98 when the ball is visible, ≤ 0 when hidden (as it must be); partial balls reconstructed in place; no hallucinated balls (0.02 of a ball's pixel mass on hidden frames). Total KL *fell* 2.5 nats — occlusion removes information. |
| **M** | Object permanence? | **Vertically yes, horizontally no.** On hidden frames `h` carries `y` (R² 0.51), `vy` (0.67) and time-since-hidden (0.54), not `x` (0.17) or `vx` (−0.07). Exit side 92% right, exit time within 2 frames 60%; exit *position* worse than "the ball is where it vanished" and 5× worse than linear extrapolation. A memoryless feed-forward control fails all of these. |
| M | Why? | One-step teacher forcing pays for `y` and the clock (they set *when* the ball reappears) and pays nothing for `x` until the exit frame ten steps later. A model learns what its loss pays for, when it pays for it. |
| M | Can it be fixed? | A privileged position head proves the LSTM *can* hold `x` (R² 0.71, at better likelihood): the problem is the objective, not capacity. The best fair fix (24-step rollout + 5× weight on re-emergence frames) closes half the gap in `h` but does not improve the dreamed exit and costs a nat. |
| **C** | Does the agent need memory? | **Not on this band.** A memoryless wait-and-see oracle catches 99% overall and 97% of long-move balls, because the paddle is wide and the ball is half-visible for ~25 frames of run-up. The fair controller reaches 85% of the oracle; its shortfall is reaction to visible balls, not memory. |
| C | Does the ceiling exploit permanence? | No: the privileged-dream controller scores 0.69, below the fair 0.85. Permanence is not binding. |
| C | Where would it matter? | On the tallest band, where the memoryless bound drops to 0.65 — but there the VAE is out of distribution (recon error 260×), so the test is confounded. |

## 3. The lessons of v3

1. **Check the premise with the cheapest possible policy before training
   anything.** A two-line wait-and-see oracle would have shown on day one that
   the default band does not require memory. It was built last.
2. **Design arithmetic must include the actuator's tolerance and all partial
   cues.** Half a paddle width of slack and fifteen frames of partial
   visibility turned "impossible without memory" into "trivial without
   memory".
3. **Anisotropic memory names the mechanism.** Which variables the recurrent
   state carries through the gap is a readout of which ones the loss rewarded,
   and when. Delayed credit is systematically under-learned by one-step
   prediction.
4. **Build the ceiling.** One privileged run separated "cannot" from "was not
   asked to" faster than three fair fixes.
5. **The general fix is not general.** The multi-step rollout loss fixed v2's
   drift and did nothing for v3's permanence, because during an occlusion the
   open-loop target is the uninformative blank band.
6. **Censoring hides failure.** A dream that declines to re-emerge a ball on
   the hard cases scores well on the cases it answers. Report the fraction
   scored.
7. **A null that cannot fail is not a control.** The first hidden-wall-bounce
   test passed the no-memory baseline at 100%.
8. **Use the right floor.** On this world a probe of `z` on hidden frames is
   not zero (the tracking paddle leaks ball x); the memoryless model with the
   same cues is the floor.

## 4. Numbers to remember

| | v1 | v2 | v3 |
|---|---|---|---|
| VAE KL (nats) / active units | 14.4 / 9 | 15.5 / 8 | 11.9 / 6 |
| position R² from a frame, ball visible / hidden | 0.99 / — | 0.98 / — | 0.98 / ≤ 0 |
| hidden-frame `y` / `x` R² from `h` | — | — | 0.51 / 0.17 (privileged ceiling 0.71 for `x`) |
| exit side / time-within-2 | — | — | 92% / 60% |
| useful dream horizon (visible frames) | 35 | 26 | 14 |
| controller vs oracle, interceptions per visit | ~0.9 → 1.0 | 0.79 → 1.0 | 0.85 |
| memoryless oracle vs oracle | — | — | 0.99 |

## 5. Next, in order of value per hour

1. **Make memory matter, then re-test.** Train the VAE on all three band
   heights; narrow the paddle or lower the band so the memoryless oracle drops
   well below 1 on the default band; re-run the controller comparison with the
   floor and ceiling. This is the experiment v3 was meant to be.
2. **A fully self-supervised emergence weighting** from a frozen `z`-probe of
   visibility (R² 0.98) instead of the simulator flag.
3. **Fix `x` for real**: the privileged head shows it is an objective problem.
   Candidates: a contrastive or predictive loss on the exit frame specifically;
   or the v4 plan — a transformer with attention back to the entry frames,
   which sidesteps the need to *carry* `x` at all.
4. **v4 — the gravity switch**: a latent variable that is never visible in the
   current frame and flips on paddle contact. The long-range dependency where a
   transformer should beat the LSTM, and where the tools built here (paired
   controls, ceilings, conservation and permanence metrics, memoryless
   oracles) all apply.

## 6. Reproducing

Run logs with every command and number: [`wm/README_V3.md`](../../wm/README_V3.md),
[`wm/README_M3.md`](../../wm/README_M3.md), [`wm/README_FIX3.md`](../../wm/README_FIX3.md),
[`wm/README_C3.md`](../../wm/README_C3.md). Tests: `python -m pytest tests/ -q`
(116). Live viewer: `python -m wm.live --v3`.

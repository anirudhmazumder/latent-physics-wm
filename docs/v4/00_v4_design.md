# v4 — 00: Design. The gravity switch: a latent that must be remembered

*The last tier of the plan. A variable that is set by an event, is never visible
afterwards, and shapes every frame from then on.*

---

## 1. What the earlier tiers could not ask

- v1: hidden state inferable from *two* frames (velocity).
- v2: dynamics readable from *one* frame (colour → speed).
- v3/v3.1: state carried through a *gap* of ~10–20 frames (position behind a
  band).

In every case the information needed to predict the future was recoverable
from a bounded, short window of recent frames. v4 removes that: a binary
latent — the **direction of gravity** — is flipped by an event (paddle
contact), and after the event nothing in any short window of frames reveals it.
It shows up only as a slow curvature of the ball's path, detectable over ~40
frames, and it stays in force for the rest of the episode or until the next
flip, typically 100+ frames later. To predict well, the world model has to
**notice the event and remember its consequence indefinitely**. This is the
regime where an LSTM that must *carry* a bit through hundreds of updates is
expected to lose to a transformer that can *look back* at the frame where the
bit was set — and where that comparison can be made cleanly.

## 2. The change to the world

> **Superseded in part.** The parameters in this section are still what
> `data/v4` was collected at and what every v4 number on disk refers to, so
> they stay. But the *claim* made for them — that they price the hidden latent
> for the controller — did not survive the oracle sweep. See **§2b** below
> before designing anything on top of this section.

| field | value | meaning |
|---|---|---|
| `gravity` | 0.0001 (world units per frame²) | a constant vertical acceleration on the ball |
| `gravity_sign` | ±1, sampled at reset | +1 = pulls up (toward the ceiling), −1 = pulls down (toward the paddle) |
| flip rule | every paddle contact multiplies `gravity_sign` by −1 | the event that sets the latent |
| `launch_min_angle` | 40° from horizontal | so the ball still reaches both floor and ceiling under either gravity |

Everything else is v1 (single ball colour, no band, paddle width 0.26,
speed 0.022 at launch). With gravity the ball's speed is no longer constant, so
the constant-speed renormalisation is **off** in v4; energy is conserved by the
elastic walls, so speed stays bounded (the collector asserts a cap). The
paddle's english still applies at contact, adding a little horizontal velocity;
its cumulative effect over ~2 contacts per episode is small and is clipped.

**Why these numbers.** Position is quantised to 1/64 by the renderer, so an
acceleration of 1e-4 per frame² is invisible in any window shorter than ~15
frames (the noiseless quadratic-fit signal-to-noise ratio is 0.25 at 10 frames
and 1.4 at 20) and unmistakable over 40 (SNR 8). Over one floor-to-ceiling
traverse (~60 frames) it changes the vertical speed by 27%, so trajectories
under the two signs are visibly different *as trajectories* while being
indistinguishable *frame to frame*. With launch angles above 40° the ball's
vertical kinetic energy exceeds the potential energy of the box height, so it
still bounces floor ↔ ceiling under both signs and the game stays playable in
both phases.

The state vector gains `gravity_sign` (grading only) and the event mask gains
`EVENT_FLIP`. Paddle contacts occur on ~1% of frames under the mixed policy, so
episodes contain 0–4 flips; some episodes never flip, which is also
informative.

## 2b. Revision after the oracle sweep — the side-wind, and why it is not enough

*Written after [`runs/v4_design/sweep.md`](../../runs/v4_design/sweep.md) and
[`sweep_v41.md`](../../runs/v4_design/sweep_v41.md). Two design sweeps, both
run before any training, both negative on the C-stage question. This section
records what changed in the code and what the arithmetic now says.*

**What the v4 sweep found.** The gap between the full oracle and the sign-blind
oracle is 0.00 at every gravity from 5e-5 to 2e-4, while the stalled-episode
rate goes from 0 % to 43 %. §2's argument for 1e-4 was about *visibility* (the
acceleration is below the renderer's quantisation over a short window) and it
holds. Its unstated second half — that an invisible-but-consequential latent is
therefore worth something to an agent — does not. A wrong sign moves the
predicted landing x by at most 0.035, and the error *shrinks to zero as the
deadline approaches*, which is the one error profile a fast bang-bang tracker
absorbs for free. It never has to commit.

**The change: `gravity_axis`.** `BoxConfig` gains `gravity_axis: "y" | "x"`.
At `"y"` (the default, and byte-identical to §2's world) the flipping
acceleration is the vertical gravity described above. At `"x"` it is a
horizontal **side-wind** on `ball_v[0]`, +1 blowing toward +x — same magnitude,
same flip-on-contact rule, same `EVENT_FLIP`, same `gravity_sign` state column
(under axis x it names the wind's direction). The reasoning: vertical gravity
perturbs the landing *height*, which converts into *x* only through the ball's
slow horizontal speed, so the sign's effect on the thing the paddle cares about
was second-order. A side-wind makes it first-order.

**What it bought, and it is not nothing.** Under a side-wind `vy` is never
touched, so the two signs have *identical* vertical dynamics. That kills, by
construction, the leak §2 did not anticipate and the v4 sweep found by accident:
under vertical gravity the sign also sets the ball's energy budget, so one
traverse's duration betrays it (48 vs 40 median frames, 1.20×) with no memory of
the flip at all. It also removes the stall–magnitude tradeoff entirely (0 %
stalled at 40°, against v4's 5 %), because nothing is climbing against anything,
and it lets the launch guard go back to v1's 14.5° since the energy argument
that forced 40° no longer applies. The separation between the two sign
hypotheses at landing roughly tripled, 0.035 → 0.105.

**What it did not buy: the gap is still 0.00**, in all six new cells. Two caps,
and the second one closes the question:

1. *The box folds the parabola.* There is no free parabola in x — side walls
   arrive every ~40 frames, so the wind shifts the phase of a bounded
   oscillation rather than translating the ball. The true divergence saturates
   near 0.18 and stops responding to the magnitude entirely above 2e-4.
2. *The deadline argument is axis-independent.* A frame where a blind oracle is
   forced to lose needs `|Δx| > v_p · t`. Since `|Δx|` grows as `a t²` and is
   capped by the box at `1 − 2r`, such a frame exists only when
   **`v_p² < a (1 − 2r)`** — i.e. only when the paddle *cannot* cross the box
   in the time the error takes to appear. At v4's `paddle_speed` = 0.030 that
   demands `a > 1.07e-3`, while a ball can only cross the box against the wind
   at all if `a < 2.7e-4`. The two constraints miss each other by 4×.

**The consequence for the design.** The lever was never the acceleration — not
its magnitude and not its axis. It is `paddle_speed`, and the sweep's appendix
confirms it directly: at `paddle_speed` = 0.006 the criterion holds, slack goes
negative on 5.9 % of approach frames, and a non-zero gap (+0.03) appears for the
first time in either sweep — *and only under the side-wind* (vertical gravity at
the same slow paddle gives −0.01). So the axis change is **necessary but not
sufficient**. A v4.2 that wants the §3 "C (the controller)" claims would combine
`gravity_axis="x"` with a paddle slow enough (or a box wide enough, or v3.1's
occluder placed late on the approach — all three switches compose) to satisfy
`v_p² < a (1 − 2r)` with margin, and then re-run `sweep_v41.py`.

**What this does *not* change.** §3's M-stage questions — is the sign in `h`,
does the dream curve the right way, does the flip counterfactual work,
LSTM vs transformer — are untouched by all of this. They need a latent that is
invisible per frame, holds for ~100 frames, and measurably bends the
trajectory, and §2's parameters deliver all three. What is now known is that the
last row of §4's success table ("the agent uses it") cannot be earned at these
settings on either axis, and should be read as an open v4.2 question rather than
a v4 deliverable. `data/v4` is unchanged and remains the vertical-gravity world
of §2.

## 3. Where the information has to go, stage by stage

### V (the frame code)
Nothing new is visible in a frame: no sprite, no colour change. V should look
like v1's. Probes must confirm `gravity_sign` is **not** decodable from `z`
(R² ≈ 0 / accuracy ≈ 50%) — the negative result that defines the problem, as
"velocity not in z" did in v1. A leak here (e.g. through a position-distribution
shift between signs) must be measured and reported.

### M (the dynamics) — the tier's question
- **Is the sign in `h`?** Probe `h` for `gravity_sign` (accuracy) as a function
  of *frames since the last flip* (or since the episode start if none). The
  memory curve: perfect recall right after a contact, decay over 50–150 frames
  for a model that cannot hold it.
- **Does the dream curve the right way?** Dream 60 steps from a warm-up that
  includes a flip; decode; fit the dreamed trajectory's vertical acceleration;
  compare its sign to the truth. Against a no-gravity extrapolation and a
  "sign at episode start" baseline.
- **The flip counterfactual.** Two warm-ups identical except that one contains a
  paddle contact (re-simulated from recorded state with the paddle moved so the
  ball misses / hits); dream both; the dreamed acceleration should differ in
  sign. This is the direct interventional test of "the model knows contact
  flips gravity".
- **How long can it hold it?** The memory curve's half-life, against the
  interval between flips.
- **LSTM vs transformer.** Train a causal transformer over the last N latents
  and actions (N = 128, the same MDN heads, comparable parameter count) as an
  alternative M. Same probes, same dreams, same counterfactual. The prediction
  from the literature: the LSTM's recall decays with distance from the flip;
  the transformer's does not, because attention can attend to the contact
  frame directly. Measure it.
- **Controls:** the feed-forward model (floor); the LSTM with actions ablated;
  and — the v2/v3 lesson — `eval_conservation` on the sign (a constant between
  flips) in long sampled dreams, and `eval_dream_alive`, before any controller
  is trained.

### C (the controller)
A controller must know where the ball will land, and under ±gravity the same
visible position and velocity land in different places. Before training:
**an oracle sweep** — the full oracle (true state), a **sign-blind oracle** that
assumes gravity always points one way (the memoryless-for-this-latent bound),
and stand-still — to confirm the task actually rewards knowing the sign. If the
sign-blind oracle is within a few percent of the full oracle, adjust `gravity`
upward before training anything (the v3 lesson, applied in advance). Then the
usual: dream-trained controllers on each M (LSTM, transformer, floor), skill
split by frames-since-flip, and the matched `h`-ablation.

## 4. What would count as success

| claim | evidence |
|---|---|
| the latent is invisible per frame | sign accuracy from `z` ≈ 50% |
| the latent is in memory | sign accuracy from `h` well above 50% long after the flip |
| the model knows contact flips gravity | the flip counterfactual changes the dreamed acceleration's sign |
| architecture matters for long memory | transformer recall flat in frames-since-flip where the LSTM's decays; same on the counterfactual |
| the agent uses it | dream-trained controller above the sign-blind oracle, and the `h`-ablation below it |

Expected failure modes, from the earlier tiers: the one-step loss never pays
for remembering a bit whose effect per frame is 1e-4 (the v3 mechanism at its
most extreme — the flip's payoff is spread thinly over hundreds of frames);
the sign diffusing in sampled dreams (v2's drift) since nothing restores it;
and the dream being alive but curving the wrong way, which is the one failure
that would be *interesting* rather than merely bad.

## 5. What carries over

Everything, plus the v3.1 additions: the oracle sweep as the first step, the
dream-alive check before controller training, per-run (V, M) pairing, matched
`h`-ablation controls, conservation and memory metrics. The only genuinely new
code is the transformer dynamics model and the flip counterfactual.

Next: [01 — the v4 environment, data and sweeps](01_v4_env_data_and_sweeps.md).

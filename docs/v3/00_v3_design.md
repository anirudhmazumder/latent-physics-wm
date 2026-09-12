# v3 — 00: Design. The occlusion band: object permanence

*Why v3 exists, what changes in the world, where the information has to go,
and the experiments that decide whether the model has it.*

---

## 1. What v1 and v2 could not ask

In v1 the model had to *infer* something a frame does not show (velocity) from
consecutive frames. In v2 it had to *read* a dynamical property (speed) off an
appearance cue (colour). In both, the ball was always visible: at every step
the current frame pinned down where it was. The hidden state `h` only ever had
to carry *derived* quantities.

v3 removes the ball from view. For a stretch of every vertical traverse the
frame contains no information about the ball's position whatsoever. If the
world model is to predict where the ball re-emerges, or the agent is to be
under it when it does, **position itself must be carried in memory**, updated
by the model's own dynamics while nothing is observed. That is object
permanence — the thing infants acquire around eight months — and it is the
first test in this project of whether `h` is a *state estimator* rather than a
velocity buffer.

## 2. The change to the world

One addition, rendered on top of everything after the ball is drawn, with the
physics untouched:

| field | value | meaning |
|---|---|---|
| `occluder` | on | draw an opaque horizontal band across the full width |
| `occluder_y` | (0.28, 0.58) | the band's bottom and top edge in world coordinates |
| `occluder_color` | a mid grey-blue, distinct from ball, paddle and background | static scenery the VAE must learn once |

Why those numbers. The ball's diameter is 0.16, so a band 0.30 tall hides it
*completely* for `(0.30 − 0.16) / |v_y|` frames — about 9 frames at a typical
vertical speed, with roughly 20 frames of partial occlusion around them. The
band's bottom edge at 0.28 means a descending ball reappears only about **10
frames before it reaches the paddle**; the paddle needs ~30 frames to cross the
box. So a controller that waits to see the ball cannot catch it from far away:
**it has to commit while the ball is hidden**, on the strength of the model's
memory. That is what makes occlusion matter for the agent and not just for the
dynamics model.

Everything else is v1: constant speed, one ball colour, the v1 paddle. Mass
from colour is left *off* so the occlusion question is not compounded (the two
switches are independent and can be combined later).

The state vector gains a diagnostic column, `ball_visible` — the fraction of the
ball's area not covered by the band — and the event mask gains a flag for
"fully hidden this frame". Both are for grading only; the model never sees
them.

## 3. Where the information has to go, stage by stage

### V (the frame code)

When the ball is fully hidden the frame is pixel-identical for every ball
position behind the band. So **V cannot encode position on hidden frames**, by
construction, and a probe of `z` for ball position *conditioned on visibility*
must show: high R² when visible, near zero when hidden, and something in
between when partially covered. That is the correct answer, not a failure — and
it is the clean statement of why M has to do more than in v1 and v2.

Two things V does have to get right: render the band as static scenery without
spending latent capacity on it, and reconstruct a *partially* occluded ball
correctly (the visible sliver, in the right place).

### M (the dynamics)

`h` must carry the ball's position through the gap. Experiments:

- **Position from `h` on hidden frames.** Probe `h` for `ball_x, ball_y`
  separately on visible, partially hidden and fully hidden frames. The
  headline: R² on fully hidden frames, where `z` gives nothing. Also as a
  function of *how many frames the ball has been hidden* — the memory's decay
  curve.
- **Emergence prediction.** Dream through an occlusion with the true actions:
  does the ball re-emerge at the right time, on the right side, at the right
  x? Error in exit time (frames) and exit position, against the no-memory
  baseline of "the ball is wherever it was when it disappeared".
- **The wall bounce behind the band.** Some balls hit a side wall *while
  hidden*. The exit x then depends on a collision the model never saw. Does it
  get those right? This is the strongest test of an internal simulation.
- **Counterfactual entry velocity.** Same entry point, different (dreamed)
  pre-occlusion trajectories → different exits; check the mapping.
- **Memory horizon.** Evaluate on bands of several heights (collected as
  separate test sets with a taller band): how many hidden frames can `h`
  bridge before the exit prediction degrades to the no-memory baseline?
- **Conservation.** The v2 lesson applied: run `eval_conservation` — here the
  conserved quantity during occlusion is the *hidden ball's velocity* — before
  any controller is trained.
- **A colour-blind-style control**: the same model trained on **v1 frames**
  (no band) then evaluated on v3 frames would see the band as an out-of-
  distribution object; more useful is a *feed-forward* control (an MLP on
  `[z_t, a_t]` with no recurrence), which by construction cannot carry position
  through the gap and gives the floor for every memory test.

### C (the controller)

The band sits so that a descending ball is invisible during most of the window
in which the paddle must move. Tests:

- Interceptions per floor visit, split by whether the ball was **hidden at the
  moment the paddle would have had to start moving** (i.e., by how far the
  paddle had to travel). A controller with object permanence is flat across
  that split; one without collapses on long moves.
- **Paddle motion during occlusion**: does the paddle start moving toward the
  ball's eventual landing point *while the ball is hidden*? Measured directly
  from actions, against the oracle (which always knows) and a "wait-and-see"
  policy (tracks only visible balls).
- The `z`-only controller is now a sharp negative control: it sees nothing
  while the ball is hidden and must stall.
- Held-out band height (taller band) for the controller: does skill degrade
  gracefully with longer occlusions?

## 4. What would count as success

| claim | evidence |
|---|---|
| V behaves correctly | position R² from `z` ≈ 0.99 visible, ≈ 0 hidden; partial occlusion reconstructed |
| M has object permanence | position R² from `h` on fully hidden frames > 0.9; exit-time error < 2 frames; exit-x error well below the no-memory baseline; hidden wall bounces predicted above chance |
| M simulates, not extrapolates | wall bounces behind the band handled; memory horizon > the band's natural occlusion length |
| C uses the memory | flat interceptions per visit across "had to start moving while hidden"; paddle moves toward the landing point during occlusion; `z`-only controller collapses |

Failure modes to watch: M learns "ball vanishes, ball appears" as two events
with a learned typical delay but no positional carry-over (exit x uncorrelated
with entry trajectory); the VAE encodes the *band edges* as ball position on
partial frames; the dream forgets the ball entirely and re-emerges it at a
prior location; and the v2 lesson — velocity drifting during the hidden stretch
under sampling.

## 5. What carries over

Everything: the collector and probe suite (state names from meta), the MDN-RNN
and its trainer, `eval_rnn`, `eval_conservation`, the dream environment, CMA-ES
training with per-controller (V, M) pairing, the by-condition evaluator, and the
live viewer (`--v3` to be wired). v3 is new questions on the same
infrastructure.

Next: [01 — the v3 environment, data and VAE](01_v3_env_data_vae.md).

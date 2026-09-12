# v2 — 00: Design. Mass from colour: an appearance → dynamics causal edge

*Why v2 exists, what changes in the world, what we expect each stage to do,
and the experiments that decide whether it did.*

---

## 1. What v1 could not ask

In v1 everything that mattered about the ball's motion could be *inferred from
motion*: two frames give velocity, and velocity plus walls give the future. The
frame's appearance carried no information about the dynamics beyond position.
So v1 could show that a world model learns to *track* hidden state (velocity in
`h`), but it could not show that a world model learns a **causal link from what
a thing looks like to how it behaves**. That is the next rung.

Physics is full of such links: colour → temperature, size → mass, material →
bounciness. A model that has "understood" them can predict how a *new* object
will move before it has seen it move. That is exactly what we will test.

## 2. The change to the world

At each reset the ball is given a **mass** `m`, drawn log-uniformly from
`[0.5, 2.0]` (median 1, which is the v1 ball). The mass is not shown as a
number; it is shown as **colour**: a monotone interpolation from yellow (light)
to purple (heavy). Mass has two consequences, both of the form "same impulse,
different mass":

| effect | rule | consequence |
|---|---|---|
| speed | `speed = 0.022 / m` | a light ball moves up to 4× faster than a heavy one |
| english | `Δvx = 0.35 · paddle_vx / m` | the paddle deflects a light ball much more |

The speed effect is dense (visible in every pair of frames); the english effect
is sparse (paddle contacts only). Both are deterministic functions of colour.
Everything else — walls, paddle, rendering, actions — is v1. With
`mass_from_color` off, the environment *is* v1, byte for byte.

A **held-out band** of masses, `[0.85, 1.2]`, is excluded from all training
data and collected separately. That gives us a clean generalisation question:
can the model *interpolate* the colour → dynamics law to colours it has never
seen move?

Code: `worldsim/bouncing_box.py` (`BoxConfig.mass_from_color`, `mass_min`,
`mass_max`, `mass_holdout`, `mass_to_color`). Data: `data/v2/*`.

## 3. Where the information has to go, stage by stage

The V-M-C split makes a sharp prediction about *where* the new information
should appear, and each stage has a test.

### V (the frame code)

Colour is a property of a single frame, so **V must encode it**. Expectations:

- One more active latent dimension than v1 (v1 had 9; expect ~10).
- `mass` (equivalently `log m`) decodable from `z` by a probe.
- **A new phenomenon**: the ball's *speed* `|v|` becomes decodable from a
  single frame — not because V sees motion (it cannot) but because speed is a
  function of colour. The velocity *components* `vx, vy` stay at R² ≈ 0. This
  is the appearance → dynamics link already visible at stage V, and it is a
  useful reminder that "decodable from a frame" does not mean "visible in a
  frame".
- Does mass decoding **interpolate** into the held-out colour band? A probe fit
  on training colours, evaluated on held-out colours.

### M (the dynamics)

M sees `z` (position + colour) and must predict the next `z`. Two ways it could
succeed:

1. **Infer speed from motion** (as in v1): after a few frames, `h` carries the
   velocity and colour is redundant.
2. **Read speed from colour**: from the very first frame, before any motion has
   been observed.

A model that only does (1) has learned nothing new. A model that does (2) has
learned the causal law. The experiments separate them:

- **Cold-start prediction.** Dream from a warm-up of *one* frame (no motion
  information). Does the dreamed ball's speed match the true speed, as a
  function of the true mass? A model that ignores colour will dream one
  average speed for every colour.
- **Recolour counterfactual.** Take a real trajectory, replace the ball's
  colour in the *latent* (or in the frame, then re-encode) with a different
  mass's colour, dream forward with the same actions. Does the dreamed ball
  speed up or slow down accordingly? This is the direct interventional test
  of the causal edge.
- **Speed in `h` by mass.** Probe `h` for `|v|`; check the probe's error is not
  concentrated in one mass range.
- **Interpolation.** Repeat the above on the held-out band. A model that
  memorised colour → speed as a lookup table fails; one that learned the
  monotone law passes.
- **English by mass** (best-effort, sparse): around real paddle contacts,
  compare the dreamed post-contact `vx` change for light vs heavy balls.
- All v1 diagnostics (useful horizon, velocity probe, action counterfactual,
  contact anticipation) re-run so we can see what the extra factor cost.

### C (the controller)

A fast ball and a slow ball demand different behaviour: the paddle has ~30
frames to cross the box, a light ball can cross the box vertically in ~12. A
controller that has learned the law should **react earlier for light balls**.

- Train the same 819-parameter linear controller in the v2 dream.
- Evaluate in the real v2 game **by mass tercile** (light / medium / heavy):
  interceptions per floor visit for each.
- Baselines by tercile: stay, random, oracle. The oracle needs no mass; it
  reads the true position. The gap between controller and oracle *as a
  function of mass* is the measure of how well the controller handles speed.
- Held-out band: does the controller play interpolated masses as well as
  trained ones?
- Decision analysis: regress the controller's drive on `x_err`, `vx`, `vy`,
  and `speed × x_err`. A mass-aware controller should show an interaction
  term.

## 4. What would count as success

| claim | evidence that would support it |
|---|---|
| V encodes the new factor | mass R² > 0.9 from `z`; one extra active unit; speed decodable from a single frame |
| M learned colour → speed | cold-start dreamed speed tracks true speed across masses; recolouring a dream changes its speed in the right direction |
| The law generalises | both of the above hold in the held-out band |
| C exploits it | controller ≥ 85% of oracle in every tercile; interaction term present |

And the failure modes worth watching for: V encodes colour but M ignores it
(velocity inferred from motion suffices for the loss, so colour is never
used — cold-start test fails, recolour test fails); M uses colour only as a
lookup table (fails interpolation); the extra factor eats M's capacity and the
useful dream horizon collapses.

## 5. What carries over from v1 unchanged

The entire toolchain: latent caching, the probe suite, dream rollouts graded in
world coordinates, counterfactual dreams, the batched dream environment, CMA-ES
training with the dream-vs-real transfer plot, the real-environment harness,
and the live viewer (`python -m wm.live --mass-from-color` once wired). That is
the point of having built v1 carefully: v2 is new *questions*, not new
infrastructure.

Next: [01 — the v2 environment, data and VAE](01_v2_env_data_vae.md).

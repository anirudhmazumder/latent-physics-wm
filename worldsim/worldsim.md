# worldsim — v1, v2 and v3

A ball bouncing in a unit box with a paddle you move left/right/stay. Renders to
64×64 RGB, hands you ground-truth latent state alongside every frame.

Numpy only. Pillow for GIFs, pygame only for the interactive mode.

## Quickstart

```bash
python -m worldsim.collect --out data/v1/train --episodes 100 --steps 200
python -m worldsim.collect --out data/v1/val   --episodes 10  --steps 200 --seed 1
python -m worldsim.play          # arrow keys, needs pygame
```

100×200 is ~20k frames, 245 MB, about a minute to generate.

## Data format

| file | dtype | shape |
|---|---|---|
| `frames.npy` | uint8 | `(E, T+1, 64, 64, 3)` |
| `actions.npy` | int8 | `(E, T)` — 0 left, 1 stay, 2 right |
| `states.npy` | float32 | `(E, T+1, S)` — S = 6, +1 for v2, +1 for v3 |
| `events.npy` | uint8 | `(E, T)` — bitmask: 1 wall-x, 2 wall-y, 4 paddle, 8 hidden (v3) |
| `meta.json` | | config + conventions |

`(frames[e,t], actions[e,t]) -> frames[e,t+1]`. T+1 frames give T transitions.

State columns: `ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_vx`, then
`mass` if v2 and `ball_visible` if v3, in that order. Read
`meta["state_names"]` rather than assuming 6. World coords are `[0,1]²` with
y=0 at the *bottom* of the image.

Load with `mmap_mode="r"` — these are separate `.npy` files rather than one
`.npz` specifically so you can train on data larger than 8 GB of RAM:

```python
frames = np.load("data/v1/train/frames.npy", mmap_mode="r")  # no RAM cost
batch = torch.from_numpy(np.array(frames[eps, ts]))          # copy just the batch
```

## Things worth knowing before you train

**States are for probing, not training.** Don't feed `states.npy` to the VAE.
It's there so that after training you can ask which of the six true variables
are linearly decodable from `z`, which need a nonlinear probe, and which just
aren't in there. That's the concrete version of "did it learn the right
variables," and it's a couple of hours of work on top of a model you already
have.

**Rendering is antialiased on purpose.** Hard pixel edges quantise position to
1/64, and a VAE trained on quantised positions gives you a latent space where
sub-pixel ball position isn't recoverable even in principle. The analytic
coverage rendering keeps that information in the image.

**Ball speed is conserved.** The paddle changes direction via sideways
"english," not energy. This keeps the data distribution stationary, so the
dynamics model isn't chasing a slow drift.

**Actions affect the ball rarely.** Measured paddle-contact rate is ~0.2% of
frames, i.e. the paddle intercepts roughly a third of floor bounces. Action →
paddle position is dense and easy; action → ball trajectory is a sparse, delayed,
contact-mediated effect. That's the interesting part, but if your dynamics model
seems to ignore actions entirely, that sparsity is the first thing to suspect.
Knobs: `paddle_w` (0.26), `english` (0.35), `ball_speed` (0.022).

**Sticky actions, not i.i.d.** Uniform random actions make the paddle
random-walk in tiny steps and never leave the middle. `mean_hold=8` gives full
sweeps across the box, so the model sees real paddle velocity.

## Suggested sequencing

1. Play it by hand once. Check the ball is trackable at 64×64.
2. Collect, then look at `sample_grid.png`-style tiles of your own data.
3. Train the conv VAE to convincing reconstructions **before** touching
   dynamics. If reconstructions are mush, every downstream result is
   uninterpretable and you'll spend days blaming the GRU.
4. Then dynamics on `(z_t, a_t) -> z_{t+1}`, trained separately from the VAE.
5. Then `render.side_by_side(true_rollout, dreamed_rollout)` → `save_gif`. This
   is the payoff: watching *where* and *how* it drifts.

---

# v2 — mass from colour

Everything above still describes the environment exactly when
`mass_from_color` is `False`, which is the default. `BoxConfig()` with no
arguments *is* v1, bit for bit — `tests/test_env_v2.py::test_v1_frames_byte_identical`
replays the collector's seeding and compares against `data/v1/val/frames.npy`
pixel for pixel, so if that ever stops being true you find out immediately.

## The idea

At `reset()` the ball is given a **mass** `m`, drawn log-uniformly from
`[0.5, 2.0]`, and painted a colour that encodes it. Mass then feeds back into
the physics in exactly two places, both the same statement — *same impulse,
different mass*:

| effect | v1 | v2 |
|---|---|---|
| speed (conserved within an episode) | `ball_speed` | `ball_speed / m` |
| english on paddle contact | `english · paddle_vx` | `english · paddle_vx / m` |

So a yellow ball is light: it moves ~0.044 world units/frame (≈2.8 px at 64²)
and the paddle flicks it hard. A purple ball is heavy: ~0.011/frame, and the
paddle barely bends its path.

**Why this is the right second world.** v1 had two kinds of variable: visible in
a frame (positions) and invisible in *any* frame (velocities). v2 adds a third:
visible in a frame, but only as *appearance*, with purely *dynamical*
consequences. That is an appearance→dynamics causal edge, and each stage of
V–M–C has a different job with it — V has to notice the colour at all, M has to
learn that this latent *multiplies* the step size (a multiplicative interaction
between latents, which a linear dynamics model cannot represent), and C can in
principle lead a fast ball more than a slow one.

## Why log-uniform mass

Mass enters as a ratio (`1/m`). Under a *linear* prior on `[0.5, 2]`, "twice as
heavy" and "half as heavy" would occur at very different rates and the median
ball would be `m = 1.25`, not `1`. Log-uniform makes the distribution symmetric
under `m → 1/m`, puts the median exactly at `m = 1` — the v1 ball — and makes
the colour ramp linear in the variable the pixels actually vary with.

## The colour ramp

`u = log(m/0.5) / log(2/0.5) ∈ [0,1]`, then a **two-segment** path in RGB:

```
u=0.0  (250,220,80)  yellow   m = 0.5   fast, easily deflected
u=0.5  (235,90,70)   v1 red   m = 1.0   exactly the v1 ball
u=1.0  (150,40,200)  purple   m = 2.0   slow, stubborn
```

A single straight lerp from yellow to purple was the first choice and it is
worse: its midpoint is `(200,130,140)`, a desaturated dusty pink, and the
midpoint is exactly where most of the probability mass sits. Routing through the
v1 red keeps every ball vivid, keeps the whole ramp ≥100 RGB units away from the
blue paddle and the near-black background (asserted in the tests), and makes the
median v2 ball literally the v1 ball.

`mass_to_color(m, cfg)` and `color_to_mass(rgb, cfg)` are exported. The inverse
projects onto the polyline using all three channels rather than inverting one —
robust to the 8-bit rounding, to antialiased edge pixels, and to VAE
reconstructions that are near the ramp but not on it. Round trip is accurate to
<0.5% in mass.

## State vector

`states.npy` gains a 7th column, `mass`, **only** when `mass_from_color` is on:

```
v1: ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_vx          (6)
v2: ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_vx, mass    (7)
```

Read `meta["state_names"]` rather than assuming 6. `meta["version"]` is
`"v2_mass_from_color"` for v2 splits, and `meta` also records `mass_from_color`,
`mass_holdout` and `mass_only`.

## Held-out masses: the generalisation test

Two mutually exclusive collector flags, both rejection-sampled at `reset()`:

- `--mass-holdout LO HI` — never sample a mass inside `[LO, HI]`. Used for every
  *training* split.
- `--mass-only LO HI` — sample only inside `[LO, HI]`. Used for the matching
  *test* split.

With `0.85 1.2` this gives a clean **interpolation** test: the model has seen
lighter balls and heavier balls, but never one of that colour. Fit a probe on
the seen colours, score it on the unseen band, and you learn whether the latent
code for colour is a genuine continuum or a lookup table of the colours it was
shown. (`wm/analyze.py --eval-data` does exactly this.)

## Collecting v2

```bash
COMMON="--ball-radius 0.08 --steps 200 --res 64 --mass-from-color"
HO="--mass-holdout 0.85 1.2"
python -m worldsim.collect --out data/v2/train     --episodes 150 --seed 0   --policy sticky $COMMON $HO
python -m worldsim.collect --out data/v2/train_mix --episodes 300 --seed 10  --policy mix --p-track 0.5 $COMMON $HO
python -m worldsim.collect --out data/v2/val       --episodes 15  --seed 1   --policy sticky $COMMON $HO
python -m worldsim.collect --out data/v2/val_mix   --episodes 20  --seed 11  --policy mix $COMMON $HO
python -m worldsim.collect --out data/v2/probe     --episodes 120 --seed 777 --policy sticky --steps 24 --ball-radius 0.08 --res 64 --mass-from-color $HO
python -m worldsim.collect --out data/v2/holdout   --episodes 30  --seed 21  --policy mix $COMMON --mass-only 0.85 1.2
python -m worldsim.v2_figures --data data/v2/train --gif-data data/v2/val_mix --out runs/v2_env
python -m worldsim.play --mass-from-color     # press R for a new mass
```

Whole set: ~1.3 GB, about 40 s of wall clock. The collector prints a log-binned
mass histogram per split — a correct training split is flat with an empty gap at
`[0.891, 1.122)`, and the hold-out split is the gap and nothing else.

Measured (see `runs/v2_env/collect.log`):

| split | eps | policy | paddle-contact rate | mean effective speed |
|---|---|---|---|---|
| train | 150 | sticky | 0.553% | 0.0230 |
| train_mix | 300 | mix p=0.5 | 0.843% | 0.0253 |
| val | 15 | sticky | 1.067% | 0.0252 |
| val_mix | 20 | mix | 0.675% | 0.0242 |
| probe | 120×24 | sticky | 0.521% | 0.0258 |
| holdout | 30 | mix | 1.133% | 0.0219 |

Contact rates are 2–5× the v1 figure (~0.2%), for two reasons worth separating:
the bigger ball (`--ball-radius 0.08`, also used for the v1 VAE data) and the
fact that light balls reach the floor far more often per episode.

![sample grid](../runs/v2_env/sample_grid.png)

32 episodes sorted by mass. The red band in the middle of the ramp is missing
because that is precisely the held-out mass band — the gap in the colours *is*
the generalisation test, made visible.

`runs/v2_env/light_vs_heavy.gif` plays the lightest and heaviest episode of
`val_mix` side by side (m=0.51 vs m=1.94) at the same frame rate.

## One surprise, worth knowing before you build M

Both v2 effects carry a `1/m`, and in the post-bounce *direction* they cancel
exactly:

```
tan θ = vx / vy = (english·paddle_vx / m) / (ball_speed / m) = english·paddle_vx / ball_speed
```

and the renormalisation rescales both components, preserving the ratio. So a
heavy ball leaves the paddle at the **same angle** as a light one; what differs
is that it leaves *slower*, so the absolute sideways velocity — the thing that
decides where the ball actually is fifty frames later — still scales as `1/m`.
This is asserted in `tests/test_env_v2.py::test_english_scales_as_one_over_mass`.
If you later want mass to change the bounce *geometry* as well, decouple the two
knobs (e.g. `english_mass_exponent`) rather than changing the speed law, which
would break the within-episode speed conservation v1 was built around.

## Tests

```bash
python -m tests.test_env_v2      # 11 tests, includes the v1 byte-identity check
python -m tests.test_rnn
python -m tests.test_controller
```

---

# v3 — the occlusion band

Everything above still describes the environment exactly when `occluder` is
`False`, which is the default. `BoxConfig()` is still bit-for-bit v1 and
`BoxConfig(mass_from_color=True)` is still bit-for-bit v2 —
`tests/test_env_v3.py` replays the collector's seeding for *both* and compares
against `data/v1/val/frames.npy` and `data/v2/val/frames.npy` pixel for pixel.

## The idea

With `occluder=True`, an opaque horizontal band is painted across the full
width of the frame, **after** the ball and the paddle, between world heights
`occluder_y = (0.28, 0.58)`. That is the entire change. The physics is not
touched: the ball flies straight through the band, bounces off the side walls
behind it, and is simply not drawn.

| field | default | meaning |
|---|---|---|
| `occluder` | `False` | draw the band at all |
| `occluder_y` | `(0.28, 0.58)` | bottom and top edge, world coords, y up |
| `occluder_color` | `(95, 110, 130)` | a mid grey-blue |

**Why this is the right third world.** v1 had two kinds of variable: visible in
a frame (positions) and invisible in *any* frame (velocities). v2 added a
third: visible as *appearance*, with purely dynamical consequences. v3 takes
away the first kind. For a stretch of every vertical traverse the frame
contains **no information whatsoever** about where the ball is — the image is
byte-identical for every ball position behind the band, which
`test_band_hides_the_ball_completely` asserts directly. If anything downstream
is to know where the ball is during that stretch, it has to be carrying it.
That is object permanence.

**Why those numbers.** The ball's diameter is 0.16 and the band is 0.30 tall,
so the ball is *completely* hidden for `(0.30 − 0.16)/|v_y|` frames — measured
at **9.7 frames on average**, with about 39% of all frames partially occluded
around them. The band's bottom edge at 0.28 means a descending ball reappears
only ~10 frames before it reaches the paddle, while the paddle needs ~30 frames
to cross the box. A controller that waits until it can see the ball therefore
cannot catch it from far away: it has to commit while the ball is hidden. That
is what makes the band matter for C and not only for M.

Mass-from-colour is left **off** for the v3 datasets, so the two questions are
not compounded. The switches are independent and compose — with both on the
state has 8 columns.

## The band colour

`(95, 110, 130)`, a desaturated grey-blue. It has to be distinguishable from
all three existing colours, for the same reason the v2 ramp did: if the band
were near the paddle blue, "the VAE lost the band" and "the VAE confused the
band with the paddle" would be the same picture. Distances in RGB:

| against | distance |
|---|---|
| background `(18,18,22)` | 160 |
| ball `(235,90,70)` | 153 |
| paddle `(70,150,235)` | **115** |

The paddle is the nearest neighbour and 115 is the tightest margin in the
project, but it clears the same >100 threshold the v2 ramp is held to
(`test_band_colour_is_distinguishable_from_everything_else`), and the two
differ in *saturation and brightness* as well as hue — a dim grey slab versus a
small vivid blue bar. Looking at `runs/v3_env/sample_grid.png` settles it: they
do not read as the same object. So the grey-blue stayed; no grey-green was
needed.

The band is drawn with the **same antialiased separable box coverage the paddle
uses**, so its edges are soft like everything else in the frame. This is not
cosmetic. A hard-edged band would be the only quantised object in the image,
and a VAE will happily spend capacity on the one crisp edge available to it.

## `ball_visible`

`states.npy` gains a column, **last**, only when `occluder` is on:

```
v1:    ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_vx                        (6)
v2:    ... , mass                                                                    (7)
v3:    ... , ball_visible                                                            (7)
v2+v3: ... , mass, ball_visible                                                      (8)
```

`ball_visible` is the fraction of the ball's **disc area** not covered by the
band, computed analytically from the circular-segment formula
(`worldsim.visible_fraction`, exported):

```
A_below(c) = πr²/2 + r² asin(d/r) + d √(r² − d²),   d = clamp(c − y, −r, r)
visible    = 1 − [A_below(hi) − A_below(lo)] / πr²
```

Analytic rather than by sampling pixels, deliberately: `ball_visible` is a
*grading* variable, and grading must not move when someone renders at 128
instead of 64. It is 1 when the ball is clear of the band, 0.5 when the centre
sits exactly on an edge, 0 when the ball is entirely inside, and monotone in
between. `test_visible_fraction_matches_a_brute_force_count` checks it against
400k-point Monte Carlo on the disc to 3e-3.

It is a diagnostic only — the model never sees it. The same goes for the new
event bit.

## `EVENT_HIDDEN = 8`

A fourth bit in the per-step event mask, set when `ball_visible < 1e-3` at the
end of the step. It is redundant with the state column by construction (and
`test_event_hidden_agrees_with_ball_visible` asserts they never disagree); it
exists so analysis code can slice hidden stretches straight out of the small
`events.npy` without loading the states, and — the reason it earns its place —
so that "was the ball hidden" and "did it hit a side wall" live in the *same*
array and can be intersected in one line. `worldsim.collect.hidden_runs` does
exactly that, and it is how the wall-bounce statistic below is computed.

Why a threshold and not `== 0`: the renderer antialiases the ball, so a ball
whose analytic coverage is a few parts in ten thousand can still tint a pixel
under the band edge. `1e-3` of a disc is about 0.02 px² at 64×64 — safely below
one pixel, and comfortably above the float noise.

## Collecting v3

```bash
COMMON="--ball-radius 0.08 --steps 200 --res 64 --occluder"
python -m worldsim.collect --out data/v3/train     --episodes 150 --seed 0   --policy sticky $COMMON
python -m worldsim.collect --out data/v3/train_mix --episodes 300 --seed 10  --policy mix --p-track 0.5 $COMMON
python -m worldsim.collect --out data/v3/val       --episodes 15  --seed 1   --policy sticky $COMMON
python -m worldsim.collect --out data/v3/val_mix   --episodes 20  --seed 11  --policy mix $COMMON
python -m worldsim.collect --out data/v3/probe     --episodes 120 --seed 777 --policy sticky --steps 24 --ball-radius 0.08 --res 64 --occluder
python -m worldsim.collect --out data/v3/tall      --episodes 30  --seed 31  --policy mix $COMMON --occluder-y 0.22 0.64
python -m worldsim.collect --out data/v3/taller    --episodes 30  --seed 32  --policy mix $COMMON --occluder-y 0.16 0.70
python -m worldsim.v3_figures --data data/v3/train --gif-data data/v3/val --out runs/v3_env
python -m worldsim.play --occluder      # try catching it yourself; it is hard
```

Whole set: ~1.4 GB, **63 s** of wall clock. The collector prints an occlusion
summary per split and writes it into `meta["occlusion"]`, so every v3 result
carries the occlusion statistics it was conditioned on.

Measured (`runs/v3_env/collect.log`):

| split | eps | band | frames hidden | partial | visible | hidden runs | mean run | median | max | runs with a wall-x bounce |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 150 | 0.28–0.58 | 17.6% | 38.8% | 43.7% | 560 | 9.4 | 8 | 43 | 19.1% |
| train_mix | 300 | 0.28–0.58 | 17.6% | 39.0% | 43.4% | 1093 | 9.7 | 8 | 44 | 15.5% |
| val | 15 | 0.28–0.58 | 18.1% | 38.1% | 43.8% | 57 | 9.5 | 8 | 27 | 14.0% |
| val_mix | 20 | 0.28–0.58 | 16.9% | 39.4% | 43.8% | 66 | 10.2 | 9 | 34 | 13.6% |
| probe | 120×24 | 0.28–0.58 | 17.6% | 40.3% | 42.1% | 72 | 6.9 | 7 | 22 | 15.3% |
| tall | 30 | 0.22–0.64 | **32.0%** | 39.8% | 28.2% | 120 | **16.0** | 14.5 | 47 | **31.7%** |
| taller | 30 | 0.16–0.70 | **46.6%** | 36.0% | 17.4% | 119 | **23.5** | 21 | 66 | **42.0%** |

Four things worth reading off that table.

1. **The design estimate was right.** §2 of `docs/v3/00_v3_design.md` predicted
   ~9 hidden frames and ~20 partial frames per traverse; the measurement is 9.7
   and 39% of frames partial. Nothing needed retuning.
2. **A sixth of the data is provably uninformative about ball position.** That
   is the fraction on which every memory claim in v3 will be scored.
3. **Hidden runs are short but heavy-tailed** — median 8, max 44 on the default
   band. The tail is the ball entering the band at a shallow angle and crawling
   through it, and it matters because those are the runs that need the longest
   memory. See `runs/v3_env/hidden_run_lengths.png`.
4. **15–19% of hidden runs contain a side-wall bounce**, rising to 42% on the
   `taller` band. Those are the runs where "keep extrapolating the last seen
   velocity" gives the *wrong* exit x, so they are the ones that separate an
   internal simulator from an extrapolator. There are ~170 of them in
   `train` + `train_mix`, which is enough to learn from and enough to test on.

The `probe` split's runs are shorter (6.9) only because its episodes are 24
steps long, so runs straddling the episode boundary are truncated.

![sample grid](../runs/v3_env/sample_grid.png)

32 frames stratified on `ball_visible` and sorted visible → hidden (a uniform
sample would have shown almost no partial occlusions, which are the interesting
case). The last row is fully hidden: those eight frames differ only in the
paddle.

`runs/v3_env/occlusion_episode.gif` is `data/v3/val` episode 0, frames 63–140.
The ball enters the band from below at y = 0.36, is hidden for **27 frames**,
hits the **left wall at x = 0.082 while completely out of sight** (the event
mask for that step is `9` = `EVENT_WALL_X | EVENT_HIDDEN`), and re-emerges
travelling right. The exit x depends entirely on a collision that never
appeared in a single pixel. That one clip is the whole of v3.

## Tests

```bash
python -m tests.test_env_v3      # 12 tests, includes the v1 AND v2 byte-identity checks
python -m pytest tests/ -q       # 77 total
```

The four that matter: v1 and v2 frames unchanged; the band is exactly its own
colour inside and touches nothing outside; `ball_visible` matches brute-force
Monte Carlo and `EVENT_HIDDEN` matches `ball_visible`; and **the trajectory is
bit-identical with and without the band**, for the same seed and actions. That
last one is v3's premise — if the band ever perturbed the dynamics, "the model
cannot see the ball" and "the ball does something different back there" would
be confounded and no memory result would mean anything.

---

## Extending to v4

`BoxConfig` / `BouncingBox` are structured so this stays additive (v2 and v3
above both were):

- **v4 gravity switch** — add a switch sprite and a `gravity_sign` field, flip it
  on paddle contact, apply in the substep loop. A latent variable that is *not*
  visible in the current frame at all — the long-range dependency where a
  transformer dynamics model should beat a GRU, and where you can measure the
  crossover cleanly.

Keep `events.npy` in mind for v4 — you'll want an `EVENT_SWITCH` flag (16, the
next free bit after `EVENT_HIDDEN = 8`) so you can condition analysis on "how
many frames since the switch flipped." `worldsim.collect.hidden_runs` is
written against a generic boolean mask and will find switch intervals too.

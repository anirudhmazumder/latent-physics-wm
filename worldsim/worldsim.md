# worldsim — v1

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
| `states.npy` | float32 | `(E, T+1, 6)` |
| `events.npy` | uint8 | `(E, T)` — bitmask: 1 wall-x, 2 wall-y, 4 paddle |
| `meta.json` | | config + conventions |

`(frames[e,t], actions[e,t]) -> frames[e,t+1]`. T+1 frames give T transitions.

State columns: `ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_vx`. World
coords are `[0,1]²` with y=0 at the *bottom* of the image.

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

## Extending to v2–v4

`BoxConfig` / `BouncingBox` are structured so these are additive:

- **v2 mass-from-colour** — sample a ball colour at reset, map it to a mass, use
  it in `_renormalise_velocity` / the paddle reflection. Adds an appearance→
  dynamics causal edge that can't be read off a static frame.
- **v3 occlusion band** — draw an opaque rect in `render` after the ball, leave
  physics untouched. Object permanence, and a real test of whether `z` carries
  state through occlusion.
- **v4 gravity switch** — add a switch sprite and a `gravity_sign` field, flip it
  on paddle contact, apply in the substep loop. A latent variable that is *not*
  visible in the current frame at all — the long-range dependency where a
  transformer dynamics model should beat a GRU, and where you can measure the
  crossover cleanly.

Keep `events.npy` in mind for v4 — you'll want a `EVENT_SWITCH` flag so you can
condition analysis on "how many frames since the switch flipped."

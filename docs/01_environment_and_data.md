# 01 — The environment and the data

*The world we are trying to model, why it looks the way it does, and how the
training data was generated.*

---

## 1. The game

A single red ball moves at constant speed inside a unit square. It bounces
elastically off all four walls. A blue paddle sits flush on the floor and can be
driven left or right by the agent (three actions: `left`, `stay`, `right`). When
the ball hits the paddle it reflects, and the paddle's sideways velocity adds a
bit of "english" to the ball's horizontal velocity.

Everything is rendered to a 64×64 RGB image. That image is the *only* thing the
world model ever sees. The simulator also hands out the true underlying state at
every step —

```
state = [ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_vx]
```

— but this is used **only for evaluation** (asking "did the model learn where
the ball is?"), never as a training input. Keeping that boundary strict is what
makes the later probe results meaningful.

Code: [`worldsim/bouncing_box.py`](../worldsim/bouncing_box.py). Numpy only,
hand-written integrator, four physics substeps per rendered frame.

## 2. Why this game, and why these specific choices

It is deliberately the *smallest* world that has the three ingredients we care
about:

1. **Unobservable state.** Velocity is not visible in a single frame. A model
   must integrate over time to know it. This is the minimal version of "hidden
   physical state."
2. **Deterministic but nonlinear dynamics.** Straight-line motion is trivial;
   wall bounces are a discontinuity (velocity flips sign). A model that only
   learns "ball keeps going" will look fine for 20 frames and then be wrong.
3. **A causal action channel that is sparse and delayed.** Actions move the
   paddle immediately (dense, easy), but only affect the ball when contact
   happens (rare, delayed). That is the interesting bit: does the model learn
   that the paddle *causes* the ball's trajectory to change?

Several simulator details exist purely to make the learning problem clean:

| choice | reason |
|---|---|
| **constant ball speed** (paddle changes direction, not energy) | keeps the data distribution stationary; no slow energy drift for the dynamics model to chase |
| **paddle flush on the floor** | removes the one fiddly collision case (ball wedged under the paddle) |
| **minimum vertical velocity** after a paddle hit | prevents the ball skimming a wall for hundreds of frames |
| **antialiased rendering** (analytic pixel coverage) | hard pixel edges quantise position to 1/64; antialiasing keeps sub-pixel position *in the image*, so the VAE can in principle recover it |
| **ball radius 0.08** (the default is 0.055) | a bigger ball is a bigger share of the pixel loss, which makes the first VAE far easier to train (see doc 02 on posterior collapse) |
| **substepping** (4 physics steps per frame) | so a fast ball cannot tunnel through the paddle |

The coordinate convention is `[0,1]²` with `y = 0` at the **bottom** of the
image, matching physics intuition rather than image-row order.

## 3. The action policy used to collect data — a small trap

Ha & Schmidhuber collect their training data with a random policy. We do too,
but *which* random policy matters more than it sounds.

Uniform i.i.d. random actions make the paddle random-walk in tiny steps and
spend almost all its time near where it started. The dynamics model then never
sees sustained paddle motion and never learns that actions do much.

We use **sticky random actions**: pick an action, hold it for a geometrically
distributed number of frames (mean 8), repeat. That produces long paddle sweeps
across the box, a wide distribution over paddle position, and real paddle
velocity in the data. Code: [`worldsim/policies.py`](../worldsim/policies.py).

Even so, **paddle contact is rare**: about 0.5% of frames, roughly one hit per
200-step episode, under random play. Around a third of the training episodes
contain no hit at all. This sparsity is the single biggest difficulty for the
"does the model understand the action→ball causal link" question, and it is
why a second, mixed-policy dataset was added for the dynamics stage (§5).

## 4. The datasets

Collected with `python -m worldsim.collect`. Each split is a directory of
separate `.npy` files (not one `.npz`), so they can be memory-mapped and never
fully loaded — the frames alone are 425 MB and the machine has 8 GB.

| split | episodes × steps | seed | policy | purpose |
|---|---|---|---|---|
| `data/v1/train` | 150 × 200 | 0 | sticky random | VAE training; dynamics training |
| `data/v1/val` | 15 × 200 | 1 | sticky random | held-out monitoring |
| `data/v1/probe` | 120 × 24 | 777 | sticky random | probing the latent space (many *short* episodes — consecutive frames are near-duplicates, so 120 short episodes carry far more independent information than 15 long ones) |

Per split:

| file | dtype | shape | meaning |
|---|---|---|---|
| `frames.npy` | uint8 | `(E, T+1, 64, 64, 3)` | what the model sees |
| `actions.npy` | int8 | `(E, T)` | 0 left, 1 stay, 2 right |
| `states.npy` | float32 | `(E, T+1, 6)` | ground truth, evaluation only |
| `events.npy` | uint8 | `(E, T)` | bitmask: 1 wall-x, 2 wall-y, 4 paddle contact |
| `meta.json` | | | config and conventions |
| `mu.npy`, `logvar.npy` | float32 | `(E, T+1, 16)` | cached VAE posteriors (added after stage V) |

The convention is `(frames[e,t], actions[e,t]) → frames[e,t+1]`, and
`events[e,t]` is whatever happened during that transition. `T+1` frames give
`T` transitions.

The collector runs cheap invariant checks on every dataset: ball inside the
box, speed exactly conserved, no NaNs, and it prints the paddle-contact rate so
you notice sparsity *before* training anything.

## 5. Extra data for the dynamics stage: a mixed tracking/random policy

Added during stage M. With ~1 hit per episode the dynamics model sees only a
few hundred examples of the paddle actually deflecting the ball, and the
controller stage needs a *reward model* that can predict hits. So a second
policy was added: at each sticky "segment" the paddle either **tracks the
ball** (moves toward `ball_x`, with a small dead-zone so it doesn't jitter) or
takes a random action, chosen with probability `p_track = 0.5`.

| split | episodes × steps | seed | policy |
|---|---|---|---|
| `data/v1/train_mix` | 300 × 200 | 10 | mix, p_track 0.5 |
| `data/v1/val_mix` | 20 × 200 | 11 | mix, p_track 0.5 |

The paddle-contact rate under the mixed policy is reported in
[03 — the dynamics model](03_dynamics_model.md). The VAE was **not** retrained
on this data: the frames are the same kind of frames, and the frozen encoder
generalises to them. (Ha & Schmidhuber discuss exactly this kind of iterative
data collection — collect with a better policy, retrain M — as the natural
extension of their method.)

## 6. Things to try yourself

- `python -m worldsim.play` (needs pygame) — drive the paddle by hand. It is
  the fastest way to feel how rarely the ball reaches the paddle.
- Open `data/v1/train/frames.npy` with `np.load(..., mmap_mode="r")` and tile a
  few frames with `worldsim.render.frame_grid` to see what the model sees.
- Change `english` in `BoxConfig` and re-collect: it directly controls how
  *much* the action can affect the ball, and hence how detectable the causal
  link is.

Next: [02 — V: the vision model](02_vae_the_vision_model.md).

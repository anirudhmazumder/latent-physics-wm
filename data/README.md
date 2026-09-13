# Datasets

**Nothing in this directory except this file is tracked in git.** The five
dataset versions total ~7 GB of `float`/`uint8` frames, which does not belong in
a repository. They are fully deterministic, so regenerating them is the same
thing as downloading them:

```bash
bash scripts/collect_all.sh              # all five versions, ~30 min, ~7 GB
bash scripts/collect_all.sh --only v1    # just what the v1 tier needs
bash scripts/collect_all.sh --no-latents # frames only, skip the VAE encoding
```

Determinism is not an aspiration here, it is tested: `tests/test_env_v4.py`
replays each earlier version's collector seeding and asserts the frames match
the ones on disk pixel for pixel. So the seeds in the commands below must not be
edited — changing one silently invalidates every checkpoint trained on it.

You only need the data to **retrain** or to run the evaluations that read a
validation split. The checkpoints under `runs/` are tracked, so
`python -m wm.live` and the other checkpoint-only tools work on a fresh clone
with no data at all.

## Layout

Each split is a directory holding `frames.npy` (`uint8`, `[E, T+1, res, res, 3]`),
`actions.npy`, `states.npy`, `events.npy` and `meta.json` (the full config, the
seed, and the exact collector arguments). After latent caching it also holds
`mu.npy`, `logvar.npy` and `latent_meta.json` — the frozen VAE's encoding of
every frame, which is what the dynamics models actually train on.

| version | splits | what is new | size |
|---|---|---|---|
| `v1` | train, train_mix, val, val_mix, probe | the plain paddle-and-ball game | ~1.1 GB |
| `v2` | + holdout | the ball's colour encodes its mass | ~1.3 GB |
| `v3` | + tall, taller | an opaque occlusion band | ~1.4 GB |
| `v31` | + train_short, train_long, short, long | the band re-geometried so memory matters | ~1.9 GB |
| `v4` | + long (600-step episodes) | a hidden gravity sign flipped by paddle contact | ~1.5 GB |

`*_mix` splits use the `mix` behaviour policy (half the hold segments track the
ball) so the data contains useful paddle behaviour as well as random flailing.
`probe` splits are many short episodes, collected for the linear probes.

## Collector flags

`python -m worldsim.collect --help`. The ones that matter:

| flag | default | meaning |
|---|---|---|
| `--out` | required | the split directory to write |
| `--episodes` / `--steps` | 100 / 200 | `T` transitions gives `T+1` frames |
| `--res` / `--ball-radius` | 64 / 0.055 | every split here uses `--ball-radius 0.08` |
| `--seed` | 0 | the only thing that distinguishes two splits of the same config |
| `--policy` | `sticky` | `sticky`, `uniform` or `mix` |
| `--p-track` | 0.5 | for `mix`: probability a hold segment tracks the ball |
| `--mass-from-color` | off | **v2**: per-episode mass, colour-coded; speed = `ball_speed / mass` |
| `--mass-holdout LO HI` | none | **v2**: never sample a mass inside the band |
| `--mass-only LO HI` | none | **v2**: sample only inside the band (the generalisation set) |
| `--occluder` | off | **v3**: the opaque band, plus a `ball_visible` state column |
| `--occluder-y LO HI` | 0.28 0.58 | band edges; **v3.1** uses 0.13 0.63 |
| `--paddle-w` | 0.26 | **v3.1** uses 0.16 |
| `--gravity` | 0.0 | **v4**: magnitude; the *direction* is a hidden per-episode bit that flips on paddle contact |
| `--launch-min-angle` | 14.5 | **v4** uses 40, so episodes carry enough vertical energy |

`--gravity-axis {x,y}` (the v4.1 side-wind) exists in the parser and is used by
`runs/v4_design/sweep_v41.py`, but **no dataset on disk was collected with it**.

---

## The exact commands

These are quoted from the run logs (`wm/README_V2.md`, `wm/README_V3.md`,
`wm/README_V31.md`, `wm/README_M2.md`, `wm/README_M4.md`) and the collection
scripts under `runs/*_env/`. `scripts/collect_all.sh` runs all of them.

### v1 — 5 splits, ~30 s

```bash
python -m worldsim.collect --out data/v1/train     --episodes 150 --steps 200 --ball-radius 0.08
python -m worldsim.collect --out data/v1/val       --episodes 15  --steps 200 --ball-radius 0.08 --seed 1
python -m worldsim.collect --out data/v1/probe     --episodes 120 --steps 24  --ball-radius 0.08 --seed 777
python -m worldsim.collect --out data/v1/train_mix --episodes 300 --steps 200 --ball-radius 0.08 --seed 10 --policy mix --p-track 0.5
python -m worldsim.collect --out data/v1/val_mix   --episodes 20  --steps 200 --ball-radius 0.08 --seed 11 --policy mix --p-track 0.5
```

### v2 — 6 splits, ~40 s, 1.3 GB

```bash
COMMON="--ball-radius 0.08 --steps 200 --res 64 --mass-from-color"
HO="--mass-holdout 0.85 1.2"
python -m worldsim.collect --out data/v2/train     --episodes 150 --seed 0   --policy sticky $COMMON $HO
python -m worldsim.collect --out data/v2/train_mix --episodes 300 --seed 10  --policy mix --p-track 0.5 $COMMON $HO
python -m worldsim.collect --out data/v2/val       --episodes 15  --seed 1   --policy sticky $COMMON $HO
python -m worldsim.collect --out data/v2/val_mix   --episodes 20  --seed 11  --policy mix $COMMON $HO
python -m worldsim.collect --out data/v2/probe     --episodes 120 --seed 777 --policy sticky \
    --steps 24 --ball-radius 0.08 --res 64 --mass-from-color $HO
python -m worldsim.collect --out data/v2/holdout   --episodes 30  --seed 21  --policy mix $COMMON --mass-only 0.85 1.2
```

`holdout` is the generalisation set: only the masses (and therefore colours) the
training splits were forbidden to sample.

### v3 — 7 splits, ~63 s, 1.4 GB

```bash
COMMON="--ball-radius 0.08 --steps 200 --res 64 --occluder"
python -m worldsim.collect --out data/v3/train     --episodes 150 --seed 0   --policy sticky $COMMON
python -m worldsim.collect --out data/v3/train_mix --episodes 300 --seed 10  --policy mix --p-track 0.5 $COMMON
python -m worldsim.collect --out data/v3/val       --episodes 15  --seed 1   --policy sticky $COMMON
python -m worldsim.collect --out data/v3/val_mix   --episodes 20  --seed 11  --policy mix $COMMON
python -m worldsim.collect --out data/v3/probe     --episodes 120 --seed 777 --policy sticky \
    --steps 24 --ball-radius 0.08 --res 64 --occluder
python -m worldsim.collect --out data/v3/tall      --episodes 30  --seed 31  --policy mix $COMMON --occluder-y 0.22 0.64
python -m worldsim.collect --out data/v3/taller    --episodes 30  --seed 32  --policy mix $COMMON --occluder-y 0.16 0.70
```

### v3.1 — 9 splits, ~75 s, 1.9 GB

The nine calls are in [`runs/v31_env/collect_v31.sh`](../runs/v31_env/collect_v31.sh),
which is what produced the published data:

```bash
bash runs/v31_env/collect_v31.sh
```

Three bands are collected — the default `(0.13, 0.63)`, a short `(0.13, 0.45)`
and a long `(0.13, 0.78)` — and all three are fed to the VAE, which is the fix
for the encoder confound that made v3's taller-band numbers unreadable.
Everything uses `--paddle-w 0.16`.

### v4 — 6 splits, ~2 min

```bash
bash runs/v4_env/collect_v4.sh
```

`--gravity 0.0001 --launch-min-angle 40`, no occluder and no mass, so the
gravity sign is the only hidden variable. The `long` split is 30 episodes of
600 steps rather than 200: flips land about 90 frames apart, and the memory
curve needs episodes several flip-intervals long.

---

## Caching latents

Every dynamics model trains on the frozen VAE's `mu`, not on pixels. Cache them
after collecting, with the encoder that belongs to that tier:

```bash
for s in train val probe train_mix val_mix; do
  python -m wm.cache_latents --ckpt runs/vae_b1/vae.pt  --data data/v1/$s  --device cpu
done
for d in train train_mix val val_mix probe holdout; do
  python -m wm.cache_latents --ckpt runs/vae_v2/vae.pt  --data data/v2/$d  --device mps
done
for d in train train_mix val val_mix probe tall taller; do
  python -m wm.cache_latents --ckpt runs/vae_v3/vae.pt  --data data/v3/$d  --device mps
done
for d in train train_mix train_short train_long val val_mix probe short long; do
  python -m wm.cache_latents --ckpt runs/vae_v31/vae.pt --data data/v31/$d --device mps
done
for d in train train_mix val val_mix probe long; do
  python -m wm.cache_latents --ckpt runs/vae_v4/vae.pt  --data data/v4/$d  --device mps
done
```

Use `--device cpu` if `mps` is not available. Takes 20 s to 2 min per version.

**One extra pass, for v2.** The colour-blind control in
[`docs/v2/02_v2_dynamics_and_causal_tests.md`](../docs/v2/02_v2_dynamics_and_causal_tests.md)
trains on the *same v2 frames seen through the v1 encoder*, written under a
suffix so `--latent-suffix v1vae` can select them:

```bash
for d in train train_mix val val_mix probe holdout; do
  python -m wm.cache_latents --ckpt runs/vae_b1/vae.pt --data data/v2/$d --suffix v1vae --device mps
done
```

This is the only non-default latent suffix used anywhere in the project.

## Wall clock (Apple M1, 8 GB)

| step | time |
|---|---|
| v1 collect (5 splits) | ~30 s |
| v2 collect (6 splits) | 40 s |
| v3 collect (7 splits) | 63 s |
| v3.1 collect (9 splits) | 75 s |
| v4 collect (6 splits) | ~2 min |
| latent caching, per version | 20 s – 2 min |
| v2 `v1vae`-suffix caching | ~2 min |

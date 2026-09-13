# `wm/` — the world model, module by module

The narrative documentation is [`../docs/README.md`](../docs/README.md). This
file is the index: what each module is, and which tier and document it belongs
to. The `README_*.md` files in this same directory are the **technical run
logs** — the exact commands, wall-clock times and every number, written as each
stage was run.

Everything is a module, so everything runs as `python -m wm.<name> --help` from
the repository root.

## Run logs in this directory

| log | stage | doc it backs |
|---|---|---|
| [`README_M.md`](README_M.md) | v1 dynamics | [`docs/03_dynamics_model.md`](../docs/03_dynamics_model.md) |
| [`README_C.md`](README_C.md) | v1 controller | [`docs/04_controller.md`](../docs/04_controller.md) |
| [`README_V2.md`](README_V2.md) | v2 environment, data, VAE | [`docs/v2/01_v2_env_data_vae.md`](../docs/v2/01_v2_env_data_vae.md) |
| [`README_M2.md`](README_M2.md) | v2 dynamics and causal tests | [`docs/v2/02_v2_dynamics_and_causal_tests.md`](../docs/v2/02_v2_dynamics_and_causal_tests.md) |
| [`README_C2.md`](README_C2.md) | v2 controller | [`docs/v2/03_v2_controller.md`](../docs/v2/03_v2_controller.md) |
| [`README_FIX.md`](README_FIX.md) | v2 colour-drift fix | [`docs/v2/05_v2_fixing_colour_drift.md`](../docs/v2/05_v2_fixing_colour_drift.md) |
| [`README_C2_FIX.md`](README_C2_FIX.md) | v2 controller in the fixed dream | [`docs/v2/06_v2_controller_fixed.md`](../docs/v2/06_v2_controller_fixed.md) |
| [`README_V3.md`](README_V3.md) | v3 environment, data, VAE | [`docs/v3/01_v3_env_data_vae.md`](../docs/v3/01_v3_env_data_vae.md) |
| [`README_M3.md`](README_M3.md) | v3 dynamics and permanence | [`docs/v3/02_v3_dynamics_and_permanence.md`](../docs/v3/02_v3_dynamics_and_permanence.md) |
| [`README_FIX3.md`](README_FIX3.md) | v3 permanence fixes | [`docs/v3/03_v3_fixing_permanence.md`](../docs/v3/03_v3_fixing_permanence.md) |
| [`README_C3.md`](README_C3.md) | v3 controller | [`docs/v3/04_v3_controller.md`](../docs/v3/04_v3_controller.md) |
| [`README_V31.md`](README_V31.md) | v3.1 environment, VAE, dynamics | [`docs/v3/07_v31_env_vae_dynamics.md`](../docs/v3/07_v31_env_vae_dynamics.md) |
| [`README_C31.md`](README_C31.md) | v3.1 controller | [`docs/v3/08_v31_controller.md`](../docs/v3/08_v31_controller.md) |
| [`README_CLOCK31.md`](README_CLOCK31.md) | v3.1 exit-clock head | [`docs/v3/10_v31_clock_and_seeds.md`](../docs/v3/10_v31_clock_and_seeds.md) |
| [`README_C31_SEEDS.md`](README_C31_SEEDS.md) | v3.1 second training seeds | [`docs/v3/10_v31_clock_and_seeds.md`](../docs/v3/10_v31_clock_and_seeds.md) |
| [`README_M4.md`](README_M4.md) | v4 VAE and dynamics | [`docs/v4/02_v4_dynamics_lstm_vs_transformer.md`](../docs/v4/02_v4_dynamics_lstm_vs_transformer.md) |

---

## V — the vision model (stage one)

All tiers. Written up in [`docs/02_vae_the_vision_model.md`](../docs/02_vae_the_vision_model.md).

| module | what it is |
|---|---|
| [`vae.py`](vae.py) | The convolutional VAE itself: 64×64×3 → 16-d latent → 64×64×3, ~570k params |
| [`train_vae.py`](train_vae.py) | Trains it. One run per tier (`vae_b1`, `vae_v2`, `vae_v3`, `vae_v31`, `vae_v4`) |
| [`data.py`](data.py) | Streams frames from memory-mapped `.npy` without loading a dataset into RAM |
| [`cache_latents.py`](cache_latents.py) | Freezes the trained VAE and writes `mu.npy` per split — the handoff to stage two |
| [`diagnostics.py`](diagnostics.py) | Reconstruction grids, traversals, active units. Global loss is nearly useless here; these are what answer "is it working" |
| [`probes.py`](probes.py) | Linear / poly / kNN / MLP probes on a frozen latent space, train-test split. The tool every "is the information there, and in what form" claim uses |
| [`calibrate_probes.py`](calibrate_probes.py) | Runs the probe suite on synthetic codes of known form, so a probe number can be interpreted rather than just reported |
| [`analyze.py`](analyze.py) | The v1 VAE analysis that ties the above together |

## M — the dynamics model (stage two)

| module | what it is | tier |
|---|---|---|
| [`rnn.py`](rnn.py) | The MDN-RNN: LSTM-256 + a 5-component mixture head, plus contact and reward heads | all |
| [`seq_data.py`](seq_data.py) | Windows of cached latents, the training set for M | all |
| [`train_rnn.py`](train_rnn.py) | Trains M. Also hosts every variant flag: `--n-gauss`, `--no-action`, `--feedforward`, `--rollout-loss-steps`, `--emerge-weight`, `--pos-head`, `--clock-head` | all |
| [`transformer.py`](transformer.py) | A causal transformer as a drop-in second backbone for M — the v4 comparison | v4 |
| [`dream_env.py`](dream_env.py) | M wrapped as a batched gym-like environment: **the dream**. The thing the controller trains inside | all |
| [`conservation.py`](conservation.py) | Measuring, and penalising, the drift of a quantity the world conserves | v2, v3 |
| [`permanence.py`](permanence.py) | Primitives for the object-permanence experiments behind the band | v3, v3.1 |
| [`clock.py`](clock.py) | The exit clock: teaching `h` to count frames through an occlusion | v3.1 |

## C — the controller (stage three)

| module | what it is |
|---|---|
| [`controller.py`](controller.py) | The policy — one linear layer, 819 params — and the reference controllers it is scored against: oracle, ballistic oracle, sign-blind oracle, wait-and-see, stay, random |
| [`train_controller.py`](train_controller.py) | CMA-ES on those 819 parameters, inside the dream (or, for the baseline, in the real game) |

## Evaluation

| module | what it asks | tier |
|---|---|---|
| [`eval_rnn.py`](eval_rnn.py) | Does M know any physics? Dream horizon, velocity in `h`, contact anticipation, action counterfactuals | all |
| [`eval_controller.py`](eval_controller.py) | Does the dream-trained policy work in the *real* world? The only test that counts | v1 |
| [`eval_conservation.py`](eval_conservation.py) | Does a long dream conserve what the world conserves? | v2, v3 |
| [`eval_dream_alive.py`](eval_dream_alive.py) | Is the dream a usable training environment at all — does the ball still exist in it? | v3.1, v4 |
| [`analyze_v3.py`](analyze_v3.py) | What does `z` know while the ball is hidden? | v3 |
| [`analyze_v4.py`](analyze_v4.py) | Is the gravity sign really invisible in a single frame? | v4 |
| [`two_seeds.py`](two_seeds.py) | Pairs two CMA-ES seeds per row and labels which orderings a seed spread leaves unresolved | v3.1 |
| [`live.py`](live.py) | The interactive viewer: real game, what V sees, what M predicted, and a free-running dream, side by side | all |

## v2 — mass from colour

Docs: [`docs/v2/`](../docs/v2/).

| module | what it is |
|---|---|
| [`eval_causal_v2.py`](eval_causal_v2.py) | The six causal experiments: cold-start speed from colour, repaint-and-redream, interpolation, each against a colour-blind control |
| [`eval_controller_v2.py`](eval_controller_v2.py) | Does the controller's skill depend on the ball's mass? Sliced by mass tercile and on held-out colours |
| [`eval_ctrl_intervention.py`](eval_ctrl_intervention.py) | Repaint the ball in a *real* history: does the policy's decision move the way physics says it should? |
| [`worldsim/v2_figures.py`](../worldsim/v2_figures.py) | Eyeball figures for the v2 datasets |

## v3 / v3.1 — the occlusion band

Docs: [`docs/v3/`](../docs/v3/).

| module | what it is |
|---|---|
| [`eval_permanence_v3.py`](eval_permanence_v3.py) | Does M have object permanence? Six experiments behind the band |
| [`compare_fix_v3.py`](compare_fix_v3.py) | One figure and one table comparing the three permanence fixes against the privileged ceiling |
| [`eval_controller_v3.py`](eval_controller_v3.py) | Does the controller act on memory while the ball is hidden? Also the v3.1 evaluator |
| [`eval_clock_readout.py`](eval_clock_readout.py) | Did the clock head learn the counter — and *which* of its two counters? |
| [`worldsim/v3_figures.py`](../worldsim/v3_figures.py) | Eyeball figures for the v3 / v3.1 datasets |

## v4 — the gravity switch

Docs: [`docs/v4/`](../docs/v4/).

| module | what it is |
|---|---|
| [`eval_switch_v4.py`](eval_switch_v4.py) | Is the gravity sign in memory, and does the dream obey it? The flip counterfactual |
| [`transformer.py`](transformer.py) | (above) the second backbone the tier compares against the LSTM |
| [`worldsim/v4_figures.py`](../worldsim/v4_figures.py) | Eyeball figures for the v4 datasets |

---

# `runs/` — what each run directory is

Naming is systematic: `vae_*` is a stage-one run, `rnn_*` / `tf_*` a stage-two
run, `ctrl_*` a trained controller, `ctrl_eval*` an evaluation of several
controllers together, `*_env` / `*_design` the dataset figures and design
sweeps. A `_s1` suffix is the second training seed of the row without it.

Every directory keeps its checkpoint, its `history.json`, and the figures the
documents embed. Console logs (`*.log`) are written next to them but are not
tracked — see [`../runs/PRUNED.md`](../runs/PRUNED.md) for the GIFs that were
pruned and how to regenerate them.

### Stage one — VAEs

| run | what it is | doc |
|---|---|---|
| `vae_b1` | the v1 encoder, β=1 — used by every v1 result | `docs/02` |
| `vae_v2` | the v2 encoder (colour carries mass) | `docs/v2/01` |
| `vae_v3` | the v3 encoder (occlusion band) | `docs/v3/01` |
| `vae_v31` | the v3.1 encoder, trained on three bands at once | `docs/v3/07` |
| `vae_v4` | the v4 encoder (gravity switch) | `docs/v4/02` |

### Stage two — dynamics models

| run | what it is | doc |
|---|---|---|
| `rnn_v1` | the v1 MDN-RNN. The main model | `docs/03` |
| `rnn_g1` | v1 ablation: a single Gaussian instead of a 5-component mixture (`--n-gauss 1`) | `docs/03` |
| `rnn_noact` | v1 ablation: the action input removed. The counterfactual control | `docs/03` |
| `rnn_v2` | the v2 baseline M | `docs/v2/02` |
| `rnn_v2_nocolor` | v2 **colour-blind control**: same frames, v1 encoder (`--latent-suffix v1vae`) | `docs/v2/02` |
| `rnn_v2_noact` | v2 action ablation | `docs/v2/02` |
| `rnn_v2_mean`, `rnn_v2_ms`, `rnn_v2_cons`, `rnn_v2_ms_cons` | the four colour-drift fixes compared: mean-mode dreaming, multi-step rollout loss, the conservation penalty, and both | `docs/v2/05` |
| `rnn_v3` | the v3 baseline M | `docs/v3/02` |
| `rnn_v3_ff` | v3 **memory floor**: a feed-forward M with no recurrence at all | `docs/v3/02` |
| `rnn_v3_noact` | v3 action ablation | `docs/v3/02` |
| `rnn_v3_ms`, `rnn_v3_ms24` | multi-step rollout loss (24-step) | `docs/v3/03` |
| `rnn_v3_emerge`, `rnn_v3_emerge_only` | emergence-weighted loss: up-weight the frame the ball reappears | `docs/v3/03` |
| `rnn_v3_poshead`, `rnn_v3_ms24_poshead` | the **privileged position head** — the ceiling proving the LSTM *could* hold the hidden `x` | `docs/v3/03` |
| `rnn_v3_fix` | the side-by-side comparison of those fixes | `docs/v3/03` |
| `rnn_v31` | the v3.1 baseline M — the tier's main model | `docs/v3/07` |
| `rnn_v31_ff`, `rnn_v31_emerge`, `rnn_v31_poshead` | the v3.1 re-runs of the floor, the emergence weighting and the privileged ceiling | `docs/v3/07` |
| `rnn_v31_allbands` | v3.1 M trained on all three band heights | `docs/v3/07` |
| `rnn_v31_clock`, `rnn_v31_clock_priv`, `rnn_v31_clock_emerge` | the exit-clock head: fair, privileged, and emergence-weighted | `docs/v3/10` |
| `rnn_v31_permanence*`, `rnn_v31_clock_permanence`, `rnn_v31_clock_readout`, `rnn_v31_dream_alive`, `rnn_v31_clock_dream_alive` | evaluation outputs for the above (permanence probes, clock readout, dream-alive check) | `docs/v3/07`, `docs/v3/10` |
| `rnn_v4` | the v4 LSTM baseline | `docs/v4/02` |
| `rnn_v4_ff` | v4 feed-forward floor | `docs/v4/02` |
| `rnn_v4_noact` | v4 action ablation | `docs/v4/02` |
| `tf_v4` | the v4 **causal transformer**, context 128 | `docs/v4/02` |
| `tf_v4_ctx32` | the same transformer with context 32 — it cannot see back to the flip, and is the best predictor | `docs/v4/02` |
| `v4_dream_alive` | the v4 dream-alive evaluation, all five models | `docs/v4/02` |

### Stage three — controllers

| run | what it is | doc |
|---|---|---|
| `ctrl_v1` | the headline v1 policy: trained on **zero real frames** | `docs/04` |
| `ctrl_dense` | v1 variant: dense reward only | `docs/04` |
| `ctrl_tau1.5` | v1 variant: dream sampling temperature 1.5 | `docs/04` |
| `ctrl_z_only` | v1 **memory ablation**: the policy sees `z` but not `h` | `docs/04` |
| `ctrl_real` | v1 baseline trained in the *real* game — 1,024,000 real frames | `docs/04` |
| `ctrl_eval` | the evaluation that scores all of the above against the oracle | `docs/04` |
| `ctrl_v2`, `ctrl_v2_tau0.5`, `ctrl_v2_mix`, `ctrl_v2_z_only`, `ctrl_v2_real` | the v2 controller row and its ablations, trained in the *drifting* dream | `docs/v2/03` |
| `ctrl_v2_cons`, `ctrl_v2_cons_tau0.5` | retrained in the **fixed** (conservation-penalised) dream — the oracle-level result | `docs/v2/06` |
| `ctrl_eval_v2`, `ctrl_eval_v2_fixed` | the two v2 evaluations, before and after the fix | `docs/v2/03`, `docs/v2/06` |
| `ctrl_intervention`, `ctrl_intervention_valmix` | the repaint-a-real-history intervention on the trained policy | `docs/v2/06` |
| `ctrl_v3`, `ctrl_v3_tau1`, `ctrl_v3_ff`, `ctrl_v3_poshead`, `ctrl_v3_emerge`, `ctrl_v3_z_only` | the v3 controller row: fair, privileged, the memory floor and the `z`-only control | `docs/v3/04` |
| `ctrl_eval_v3` | the v3 evaluation, against the memoryless oracle that catches 99% | `docs/v3/04` |
| `ctrl_v31`, `ctrl_v31_tau1`, `ctrl_v31_tau0.5`, `ctrl_v31_ff`, `ctrl_v31_poshead`, `ctrl_v31_allbands`, `ctrl_v31_real` | the v3.1 controller row on the re-designed band | `docs/v3/08` |
| `ctrl_v31_z_only`, `ctrl_v31_z_only_tau1` | the v3.1 **h-ablation** — it lands exactly on the memoryless bound, which is the control that makes the headline mean something | `docs/v3/08` |
| `ctrl_v31_*_s1` | the second CMA-ES training seed of each v3.1 row | `docs/v3/10` |
| `ctrl_eval_v31`, `ctrl_eval_v31_extra`, `ctrl_eval_v31_seeds` | the v3.1 evaluations; `_seeds` scores both seeds on the same 150 episodes | `docs/v3/08`, `docs/v3/10` |

(v4 has no controller runs: two oracle sweeps showed the gravity sign never
matters for play, so the stage was retired before any training. See
`runs/v4_design/sweep.md`.)

### Environment figures and design sweeps

| run | what it is | doc |
|---|---|---|
| `v2_env`, `v3_env`, `v31_env`, `v4_env` | dataset eyeball figures, and the collection scripts for v3.1 and v4 | each tier's `01` / `07` doc |
| `v31_design` | the oracle sweep that chose the v3.1 band and paddle width | `docs/v3/06` |
| `v4_design` | the two oracle sweeps that retired the v4 controller stage, with the closed form | `docs/v4/01` |

### Driver scripts

`runs/_*.sh` are the per-stage driver scripts — the exact training and
evaluation sequences, quoted from the run logs that reference them by path
(`README_C3.md`, `README_C31.md`, `README_CLOCK31.md`, `README_V31.md`,
`README_M4.md`, `README_C31_SEEDS.md`). They take the interpreter from
`PY`, defaulting to `python`:

```bash
bash runs/_stage1_v31.sh        # v3.1: analyze + cache latents
bash runs/_train_rnn_v31.sh     # v3.1: the dynamics models
bash runs/_eval_rnn_v31.sh      # v3.1: the standard M evaluations
```

They stay under `runs/` rather than moving to `scripts/` because the run logs
record them by that path as the record of what was executed.

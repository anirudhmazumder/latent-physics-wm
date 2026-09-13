# GIFs pruned from the repository

`runs/` held 140 tracked GIFs totalling 454 MB, which is most of the repository.
66 of them (233 MB) were not referenced by any Markdown file in the repo, so
they were removed from git to make the clone a reasonable size. **Nothing that a
document points at was removed**; the 74 GIFs still tracked are exactly the ones
cited in `README.md`, `docs/**/*.md`, `wm/README_*.md`, `worldsim/*.md` or
`runs/**/*.md`.

No checkpoint (`.pt`), figure (`.png`), report (`.json`), run log (`.md`) or
array (`.npy`) was touched. Every pruned GIF is an output of an evaluation
script and can be regenerated from the checkpoints that *are* in the repo, plus
the datasets (`bash scripts/collect_all.sh`, see [`../data/README.md`](../data/README.md)).

All commands below are run from the repository root with the project's
environment active (see the setup section of the top-level
[`README.md`](../README.md)).

---

## Class 1 — `dream_vs_true_tau0.5.gif` and `dream_vs_true_tau1.0.gif` (50 files)

The open-loop dream-vs-truth strips at sampling temperatures 0.5 and 1.0. The
`tau0.0` strip of the same evaluation is the one the docs cite, and it is still
tracked. `wm.eval_rnn` writes all three temperatures (`TAUS = (0.0, 0.5, 1.0)`
in `wm/eval_rnn.py`) in a single pass, so re-running the evaluation for a run
restores both files:

```bash
python -m wm.eval_rnn --ckpt runs/<RUN>/rnn.pt --vae runs/<VAE>/vae.pt \
    --val <val roots> --out runs/<RUN>/eval --n-gif 3
```

Affected runs, with the `--vae` and `--val` they belong to:

| runs | `--vae` | `--val` |
|---|---|---|
| `rnn_v1` | `runs/vae_b1/vae.pt` | `data/v1/val data/v1/val_mix` |
| `rnn_v2`, `rnn_v2_cons`, `rnn_v2_mean`, `rnn_v2_ms`, `rnn_v2_ms_cons` | `runs/vae_v2/vae.pt` | `data/v2/val data/v2/val_mix` |
| `rnn_v3`, `rnn_v3_emerge`, `rnn_v3_emerge_only`, `rnn_v3_ms`, `rnn_v3_ms24`, `rnn_v3_ms24_poshead`, `rnn_v3_poshead` | `runs/vae_v3/vae.pt` | `data/v3/val data/v3/val_mix` |
| `rnn_v31`, `rnn_v31_clock`, `rnn_v31_clock_emerge`, `rnn_v31_clock_priv`, `rnn_v31_emerge`, `rnn_v31_ff`, `rnn_v31_poshead` | `runs/vae_v31/vae.pt` | `data/v31/val data/v31/val_mix` |
| `rnn_v4`, `rnn_v4_ff`, `rnn_v4_noact`, `tf_v4`, `tf_v4_ctx32` | `runs/vae_v4/vae.pt` | `data/v4/val data/v4/val_mix` |

The exact invocation used for each run is quoted in the corresponding run log:
`wm/README_M.md` (v1), `wm/README_M2.md` (v2), `wm/README_M3.md` /
`wm/README_FIX3.md` (v3), `wm/README_V31.md` / `wm/README_CLOCK31.md` (v3.1),
`wm/README_M4.md` (v4).

## Class 2 — `action_counterfactual_noact.gif` (12 files)

The action-counterfactual strip for the *action-blind ablation* model. The
docs cite the non-ablated `action_counterfactual.gif`, which is still tracked.
The `_noact` suffix is the tag `wm.eval_rnn` gives the run named by
`--ablate-ckpt`, so the same command as class 1 regenerates it provided the
ablation checkpoint is passed:

```bash
python -m wm.eval_rnn --ckpt runs/<RUN>/rnn.pt --vae runs/<VAE>/vae.pt \
    --ablate-ckpt runs/<RUN>_noact/rnn.pt --out runs/<RUN>/eval
```

Affected: `rnn_v1`, `rnn_v2`, `rnn_v2_cons`, `rnn_v2_mean`, `rnn_v2_ms`,
`rnn_v2_ms_cons`, `rnn_v3`, `rnn_v4`, `rnn_v4_ff`, `rnn_v4_noact`, `tf_v4`,
`tf_v4_ctx32`. (The corresponding `*_noact` checkpoints — `runs/rnn_noact`,
`runs/rnn_v2_noact`, `runs/rnn_v3_noact`, `runs/rnn_v4_noact` — are all still in
the repo.)

## Class 3 — the `ctrl_dense` playback GIFs (3 files)

`runs/ctrl_eval/dream_play_ctrl_dense.gif`,
`runs/ctrl_eval/real_play_ctrl_dense.gif`,
`runs/ctrl_eval/real_vs_dream_side_by_side_ctrl_dense.gif`.

`ctrl_dense` appears in the v1 controller tables (`docs/04_controller.md`) but
its GIFs are never embedded; the headline GIFs for that evaluation are
`ctrl_v1`'s, which are still tracked. Regenerate by naming `ctrl_dense` in
`--gif-for`:

```bash
python -m wm.eval_controller --vae runs/vae_b1/vae.pt --rnn runs/rnn_v1/rnn.pt \
    --ctrl runs/ctrl_v1/controller.pt runs/ctrl_dense/controller.pt \
    --ctrl-names ctrl_v1 ctrl_dense \
    --gif-for ctrl_dense --out runs/ctrl_eval
```

## Class 4 — `real_play_ctrl_v2_cons_tau0.5_heavy.gif` (1 file)

`runs/ctrl_eval_v2_fixed/real_play_ctrl_v2_cons_tau0.5_heavy.gif`. The *light*
ball counterpart of this same run is cited in
`docs/v2/06_v2_controller_fixed.md` and is still tracked. Regenerate with the
v2-fixed controller evaluation, naming the run in `--gif-for` and asking for the
heavy-mass band:

```bash
python -m wm.eval_controller_v2 --vae runs/vae_v2/vae.pt \
    --rnn runs/rnn_v2_cons/rnn.pt \
    --ctrl runs/ctrl_v2_cons_tau0.5/controller.pt \
    --ctrl-names ctrl_v2_cons_tau0.5 \
    --mass-from-color --gif-for ctrl_v2_cons_tau0.5 \
    --out runs/ctrl_eval_v2_fixed
```

The exact published invocation is in `wm/README_C2_FIX.md`.

---

## Full list of pruned files

```
runs/ctrl_eval/dream_play_ctrl_dense.gif
runs/ctrl_eval/real_play_ctrl_dense.gif
runs/ctrl_eval/real_vs_dream_side_by_side_ctrl_dense.gif
runs/ctrl_eval_v2_fixed/real_play_ctrl_v2_cons_tau0.5_heavy.gif
runs/rnn_v1/eval/action_counterfactual_noact.gif
runs/rnn_v1/eval/dream_vs_true_tau0.5.gif
runs/rnn_v1/eval/dream_vs_true_tau1.0.gif
runs/rnn_v2/eval/action_counterfactual_noact.gif
runs/rnn_v2/eval/dream_vs_true_tau0.5.gif
runs/rnn_v2/eval/dream_vs_true_tau1.0.gif
runs/rnn_v2_cons/eval/action_counterfactual_noact.gif
runs/rnn_v2_cons/eval/dream_vs_true_tau0.5.gif
runs/rnn_v2_cons/eval/dream_vs_true_tau1.0.gif
runs/rnn_v2_mean/eval/action_counterfactual_noact.gif
runs/rnn_v2_mean/eval/dream_vs_true_tau0.5.gif
runs/rnn_v2_mean/eval/dream_vs_true_tau1.0.gif
runs/rnn_v2_ms/eval/action_counterfactual_noact.gif
runs/rnn_v2_ms/eval/dream_vs_true_tau0.5.gif
runs/rnn_v2_ms/eval/dream_vs_true_tau1.0.gif
runs/rnn_v2_ms_cons/eval/action_counterfactual_noact.gif
runs/rnn_v2_ms_cons/eval/dream_vs_true_tau0.5.gif
runs/rnn_v2_ms_cons/eval/dream_vs_true_tau1.0.gif
runs/rnn_v3/eval/action_counterfactual_noact.gif
runs/rnn_v3/eval/dream_vs_true_tau0.5.gif
runs/rnn_v3/eval/dream_vs_true_tau1.0.gif
runs/rnn_v31/eval/dream_vs_true_tau0.5.gif
runs/rnn_v31/eval/dream_vs_true_tau1.0.gif
runs/rnn_v31_clock/eval/dream_vs_true_tau0.5.gif
runs/rnn_v31_clock/eval/dream_vs_true_tau1.0.gif
runs/rnn_v31_clock_emerge/eval/dream_vs_true_tau0.5.gif
runs/rnn_v31_clock_emerge/eval/dream_vs_true_tau1.0.gif
runs/rnn_v31_clock_priv/eval/dream_vs_true_tau0.5.gif
runs/rnn_v31_clock_priv/eval/dream_vs_true_tau1.0.gif
runs/rnn_v31_emerge/eval/dream_vs_true_tau0.5.gif
runs/rnn_v31_emerge/eval/dream_vs_true_tau1.0.gif
runs/rnn_v31_ff/eval/dream_vs_true_tau0.5.gif
runs/rnn_v31_ff/eval/dream_vs_true_tau1.0.gif
runs/rnn_v31_poshead/eval/dream_vs_true_tau0.5.gif
runs/rnn_v31_poshead/eval/dream_vs_true_tau1.0.gif
runs/rnn_v3_emerge/eval/dream_vs_true_tau0.5.gif
runs/rnn_v3_emerge/eval/dream_vs_true_tau1.0.gif
runs/rnn_v3_emerge_only/eval/dream_vs_true_tau0.5.gif
runs/rnn_v3_emerge_only/eval/dream_vs_true_tau1.0.gif
runs/rnn_v3_ms/eval/dream_vs_true_tau0.5.gif
runs/rnn_v3_ms/eval/dream_vs_true_tau1.0.gif
runs/rnn_v3_ms24/eval/dream_vs_true_tau0.5.gif
runs/rnn_v3_ms24/eval/dream_vs_true_tau1.0.gif
runs/rnn_v3_ms24_poshead/eval/dream_vs_true_tau0.5.gif
runs/rnn_v3_ms24_poshead/eval/dream_vs_true_tau1.0.gif
runs/rnn_v3_poshead/eval/dream_vs_true_tau0.5.gif
runs/rnn_v3_poshead/eval/dream_vs_true_tau1.0.gif
runs/rnn_v4/eval/action_counterfactual_noact.gif
runs/rnn_v4/eval/dream_vs_true_tau0.5.gif
runs/rnn_v4/eval/dream_vs_true_tau1.0.gif
runs/rnn_v4_ff/eval/action_counterfactual_noact.gif
runs/rnn_v4_ff/eval/dream_vs_true_tau0.5.gif
runs/rnn_v4_ff/eval/dream_vs_true_tau1.0.gif
runs/rnn_v4_noact/eval/action_counterfactual_noact.gif
runs/rnn_v4_noact/eval/dream_vs_true_tau0.5.gif
runs/rnn_v4_noact/eval/dream_vs_true_tau1.0.gif
runs/tf_v4/eval/action_counterfactual_noact.gif
runs/tf_v4/eval/dream_vs_true_tau0.5.gif
runs/tf_v4/eval/dream_vs_true_tau1.0.gif
runs/tf_v4_ctx32/eval/action_counterfactual_noact.gif
runs/tf_v4_ctx32/eval/dream_vs_true_tau0.5.gif
runs/tf_v4_ctx32/eval/dream_vs_true_tau1.0.gif
```

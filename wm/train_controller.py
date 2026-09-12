"""CMA-ES on the controller, inside the dream (or, for the baseline, outside it).

    # the main run: train entirely in M's head, check the real world every 5 gens
    python -m wm.train_controller --out runs/ctrl_v1 --inputs zh \
        --reward mix --temperature 1.0 --generations 60

    # the "no world model" control: identical optimiser, real episodes as fitness
    python -m wm.train_controller --out runs/ctrl_real --fitness real \
        --popsize 16 --rollouts 8 --generations 20

Why CMA-ES and not policy gradients
-----------------------------------
The objective here -- "number of paddle contacts" -- is a count of rare events
produced by a 150-step interaction between a discrete policy and a stochastic
sequence model. It has no usable gradient: ``argmax`` is not differentiable, the
events are sparse, and backpropagating through 150 LSTM steps of a *sampled*
rollout is both expensive and high-variance. CMA-ES ignores all of that. It only
needs to RANK candidates, so it never touches the objective's shape, and with
819 parameters it is comfortably inside the regime (up to a few thousand
parameters) where it is competitive. The price is sample complexity -- tens of
thousands of episodes -- which is exactly the price the world model is there to
make affordable: dream episodes cost nothing.

The three things that make this work rather than thrash
-------------------------------------------------------
1. COMMON RANDOM NUMBERS. Every candidate in a generation is evaluated on the
   SAME set of initial states. Fitness from 16 rollouts is a noisy estimate; if
   each candidate also got different starts, a large part of the ranking would
   be luck and CMA-ES would chase noise. Pairing the comparisons removes that
   term. (The starts are re-drawn every generation, so there is nothing to
   overfit to.)
2. ONE BATCH FOR THE WHOLE POPULATION. popsize x rollouts dreams are stepped in
   lockstep through one LSTM, so a generation is ``dream_steps`` forward passes
   of batch 512, not 512 x 150 sequential ones. This is the difference between
   seconds and an hour per generation.
3. A REAL-WORLD MEASUREMENT ON THE SIDE. Dream fitness is the thing being
   optimised, so it is guaranteed to go up and is therefore worthless as
   evidence. Every ``--real-eval-every`` generations we take the current best
   candidate into ``BouncingBox`` and count actual contacts. The gap between the
   two curves in ``dream_vs_real.png`` IS the result of stage three: if dream
   return keeps climbing while real hits plateau or fall, the controller has
   stopped learning physics and started exploiting M.

Sign convention: ``cma`` minimises, we maximise, so every fitness handed to
``es.tell`` is negated. Only that one line knows about it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .controller import (
    LinearController,
    batched_logits,
    compute_norm_stats,
    logits_to_actions,
    make_features,
    save_controller,
)
from .dream_env import DreamEnv, dream_rollout, load_start_pool
from .rnn import load_rnn


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------- dream fitness


class DreamFitness:
    """Evaluate a whole CMA-ES population in one batched dream.

    ``__call__(params (P, n_params), seed) -> (P,)`` mean return per candidate.
    """

    def __init__(
        self,
        env: DreamEnv,
        ctrl: LinearController,
        rollouts: int = 16,
        steps: int = 150,
    ):
        self.env = env
        self.ctrl = ctrl
        self.rollouts = int(rollouts)
        self.steps = int(steps)

    def __call__(self, params: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
        P, R = len(params), self.rollouts
        rng = np.random.default_rng(seed)
        # Common random numbers: one set of R starts, tiled across all P.
        e, t0 = self.env.pool.sample(R, rng)
        starts = (np.tile(e, P), np.tile(t0, P))

        def act(z, h):
            f = make_features(z, h, self.ctrl.inputs, self.ctrl.norm)
            lg = batched_logits(params, f.reshape(P, R, -1))
            return logits_to_actions(lg).reshape(P * R)

        rec = dream_rollout(
            self.env, act, batch=P * R, steps=self.steps, seed=seed, starts=starts
        )
        return {
            "fitness": rec["return"].reshape(P, R).mean(1),
            "hit_sum": rec["hit_sum"].reshape(P, R).mean(1),
        }


# ----------------------------------------------------------- real fitness


class RealFitness:
    """The no-world-model baseline: fitness is contacts counted in BouncingBox.

    Same optimiser, same controller, same inputs ``[mu_t, h_pre_t]`` -- the RNN
    is still used, but only as a *feature extractor* on real frames, never as a
    simulator. What changes is where the episodes come from, and therefore what
    they cost: one generation here is ``popsize * rollouts`` real episodes, each
    of which needs 200 physics steps, 200 VAE encodes and 200 LSTM steps.

    Candidates are evaluated on a shared block of seeds per generation (common
    random numbers again), re-drawn each generation.
    """

    def __init__(self, vae, rnn, rollouts: int = 8, steps: int = 200,
                 device: str = "cpu", seed_base: int = 900_000,
                 count: str = "frames", env: Optional[Dict] = None):
        self.vae, self.rnn = vae, rnn
        self.rollouts, self.steps = int(rollouts), int(steps)
        self.device = device
        self.seed_base = int(seed_base)
        # `count` is the v1-vs-v2 difference that actually changed the answer:
        # v1's ctrl_real optimised contact FRAMES and learned to pin the ball
        # against the paddle, scoring 2.00 hits on 1.27 real interceptions. v2
        # optimises interceptions, so the loophole is not on the table.
        self.count = str(count)
        self.env = dict(env or {"ball_radius": 0.08})
        self.env_steps_used = 0

    def __call__(self, params: np.ndarray, seed: int, ctrl: LinearController):
        from .eval_controller import run_population_real

        P, R = len(params), self.rollouts
        seeds = self.seed_base + (seed % 997) * R + np.arange(R)
        hits = run_population_real(
            params, ctrl, self.vae, self.rnn, seeds,
            steps=self.steps, device=self.device, count=self.count, **self.env,
        )                                                   # (P, R)
        self.env_steps_used += P * R * self.steps
        return {"fitness": hits.mean(1), "hit_sum": hits.mean(1)}


# ------------------------------------------------------------------- train


def train(a: argparse.Namespace) -> Dict:
    import cma

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # One dict describing the world, built by the same helper the evaluation
    # harness uses, and handed to every real-environment call below. Training
    # and evaluation cannot disagree about the world by construction.
    from .eval_controller import env_kwargs

    ekw = env_kwargs(a)
    print(f"real env: {ekw}")

    rnn, rcfg = load_rnn(a.rnn, a.device)
    print(f"M: {a.rnn}  z_dim {rcfg.z_dim}  hidden {rcfg.hidden}  K {rcfg.n_gauss}")

    print("computing input normalisation from the training set ...")
    norm, ninfo = compute_norm_stats(rnn, a.data, device=a.device, seed=a.seed)
    print(f"  {ninfo['n_samples']} samples; h std in "
          f"[{ninfo['h_std_min']:.4f}, {ninfo['h_std_max']:.4f}]; "
          f"{ninfo['h_dead_units']} dead LSTM units")

    ctrl = LinearController(rcfg.z_dim, rcfg.hidden, inputs=a.inputs, norm=norm)
    print(f"C: inputs={a.inputs}  in_dim={ctrl.in_dim}  n_params={ctrl.n_params}")

    pool = load_start_pool(a.data, warmup=a.warmup)
    print(f"start pool: {pool.mu.shape[0]} episodes x {pool.actions.shape[1]} steps")

    # The evaluator that produces the number CMA-ES optimises.
    vae = None
    if a.fitness == "real" or a.real_eval_every > 0:
        from .analyze import load_ckpt

        vae, _, _ = load_ckpt(a.vae, a.device)

    if a.fitness == "dream":
        env = DreamEnv(rnn, pool, temperature=a.temperature, reward=a.reward,
                       lam=a.lam, device=a.device)
        fit = DreamFitness(env, ctrl, rollouts=a.rollouts, steps=a.dream_steps)
        evaluate = lambda X, s: fit(X, s)                      # noqa: E731
        dream_env_steps_per_gen = a.popsize * a.rollouts * a.dream_steps
        real_env_steps_train = 0
    else:
        rf = RealFitness(vae, rnn, rollouts=a.rollouts, steps=a.real_steps,
                         device=a.device, count=a.real_fitness_count, env=ekw)
        evaluate = lambda X, s: rf(X, s, ctrl)                 # noqa: E731
        dream_env_steps_per_gen = 0
        real_env_steps_train = None  # filled from rf at the end

    es = cma.CMAEvolutionStrategy(
        np.zeros(ctrl.n_params),
        a.sigma0,
        {"popsize": a.popsize, "seed": a.seed + 1, "verbose": -9},
    )

    hist: Dict[str, list] = {
        "gen": [], "best": [], "mean": [], "worst": [], "sigma": [],
        "dream_hit_sum": [], "wall_s": [],
        "real_gen": [], "real_hits": [], "real_ci": [], "real_of_dream_best": [],
    }
    real_eval_steps = 0
    best_real, best_real_params = -np.inf, None
    best_dream, best_dream_params = -np.inf, None

    for gen in range(a.generations):
        X = np.asarray(es.ask())                               # (P, n_params)
        res = evaluate(X, a.seed * 10_000 + gen)
        f = res["fitness"]
        es.tell(list(X), list(-f))                             # cma minimises

        hist["gen"].append(gen)
        hist["best"].append(float(f.max()))
        hist["mean"].append(float(f.mean()))
        hist["worst"].append(float(f.min()))
        hist["sigma"].append(float(es.sigma))
        hist["dream_hit_sum"].append(float(res["hit_sum"][int(f.argmax())]))
        hist["wall_s"].append(round(time.time() - t_start, 1))

        # CMA-ES's own recommendation is the distribution MEAN (``es.result.xfavorite``),
        # not the best sample: the mean is where it believes the optimum is,
        # while the best sample is partly a lucky draw from a 16-rollout estimate.
        cur = np.asarray(es.result.xfavorite)
        if f.max() > best_dream:
            best_dream, best_dream_params = float(f.max()), X[int(f.argmax())].copy()

        line = (f"gen {gen:3d}  best {f.max():8.3f}  mean {f.mean():8.3f}  "
                f"sigma {es.sigma:.3f}  ({time.time() - t_start:5.0f}s)")

        if a.real_eval_every > 0 and (
            gen % a.real_eval_every == 0 or gen == a.generations - 1
        ):
            from .eval_controller import _bootstrap_ci, run_real_episodes

            ctrl.set_params(cur)
            roll = run_real_episodes(
                ctrl, vae, rnn, episodes=a.real_eval_episodes, steps=a.real_steps,
                seed_base=a.real_eval_seed_base, device=a.device, **ekw,
            )
            # This number selects `params_best_real`, so what it counts
            # matters. v1 counted contact FRAMES; v1's own conclusion was that
            # interceptions (runs of contact frames collapsed) is the honest
            # metric, so v2 selects on that. The default stays `hits` purely so
            # the v1 runs remain exactly reproducible.
            from .eval_controller import contact_runs

            hits = (
                contact_runs(roll["hits"])
                if a.real_eval_metric == "interceptions"
                else roll["hits"].sum(1)
            )
            real_eval_steps += a.real_eval_episodes * a.real_steps
            hist["real_gen"].append(gen)
            hist["real_hits"].append(float(hits.mean()))
            hist["real_ci"].append(list(_bootstrap_ci(hits, n_boot=2000)))
            hist["real_of_dream_best"].append(float(f.max()))
            if hits.mean() > best_real:
                best_real, best_real_params = float(hits.mean()), cur.copy()
            line += f"   REAL hits/ep {hits.mean():.2f}"

        print(line, flush=True)

    last = np.asarray(es.result.xfavorite)
    if best_real_params is None:
        best_real_params = last

    if a.fitness == "real":
        real_env_steps_train = rf.env_steps_used

    meta = {
        "args": vars(a),
        "env": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in ekw.items()},
        "n_params": ctrl.n_params,
        "norm_info": ninfo,
        "best_real_hits": best_real,
        "best_dream_fitness": best_dream,
        "final_dream_fitness": hist["best"][-1] if hist["best"] else None,
        "real_env_steps_train": real_env_steps_train,
        "real_env_steps_eval": real_eval_steps,
        "real_env_steps_total": (real_env_steps_train or 0) + real_eval_steps,
        "dream_env_steps": dream_env_steps_per_gen * a.generations,
        "wall_clock_s": round(time.time() - t_start, 1),
    }

    # Plots first: the transfer correlation they compute belongs in `meta`, and
    # `meta` is serialised into the checkpoint below.
    plot_fitness(hist, out / "fitness.png", a.reward)
    if hist["real_gen"]:
        # With --fitness real the left-hand curve is not a dream at all; say
        # so on the axis rather than shipping a plot that claims otherwise.
        meta["transfer"] = plot_dream_vs_real(
            hist, out / "dream_vs_real.png", a.real_eval_metric,
            "dream return" if a.fitness == "dream"
            else f"real training fitness ({a.real_fitness_count})",
        )

    save_controller(
        out / "controller.pt",
        ctrl,
        params=best_real_params,
        args=vars(a),
        extra={
            # `params` (what load_controller returns by default) is the
            # BEST-BY-REAL-SCORE candidate -- the honest choice, since real
            # score is the only unbiased signal we have. `params_last_dream` is
            # CMA-ES's final mean, i.e. the best the DREAM believes in. When the
            # two differ a lot, the dream was being exploited.
            "params_best_real": best_real_params,
            "params_last_dream": last,
            "params_best_dream_sample": best_dream_params,
            "history": hist,
            "meta": meta,
        },
    )
    (out / "history.json").write_text(json.dumps({"history": hist, "meta": meta}, indent=2))

    print(f"\nbest real hits/ep {best_real:.3f} | final dream fitness "
          f"{hist['best'][-1]:.3f} | {meta['wall_clock_s']:.0f}s -> {out}")
    return meta


# -------------------------------------------------------------------- plots


def _smooth(y: Sequence[float], w: int = 11) -> np.ndarray:
    """Centred moving average, for reading a trend through per-generation noise.

    Best-of-generation is the max of ``popsize`` estimates each built from
    ``rollouts`` stochastic dreams, so it jitters by ~0.5 even when nothing is
    improving. The raw trace is still drawn faintly -- smoothing a curve you
    then make claims about is only honest if the reader can see what was
    smoothed.
    """
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    pad = np.r_[np.full(w // 2, y[0]), y, np.full(w // 2, y[-1])]
    return np.convolve(pad, np.ones(w) / w, mode="valid")[: len(y)]


def plot_fitness(hist: Dict, path: Path, reward: str) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    g = hist["gen"]
    ax.plot(g, hist["best"], c="#3a7bd5", alpha=0.3, lw=0.8)
    ax.plot(g, _smooth(hist["best"]), label="best of generation", c="#3a7bd5")
    ax.plot(g, hist["mean"], label="population mean", c="#8ab", ls="--")
    ax.fill_between(g, hist["worst"], hist["best"], color="#3a7bd5", alpha=0.12)
    ax.set_xlabel("CMA-ES generation")
    ax.set_ylabel(f"dream return ({reward})")
    ax.set_title("fitness inside the dream")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_dream_vs_real(hist: Dict, path: Path, metric: str = "hits",
                       fitness_label: str = "dream return") -> Path:
    """The transfer plot. Two y-axes, because the units are incomparable.

    What to look for: the two curves should rise TOGETHER. If the blue (dream)
    curve keeps climbing after the orange (real) one flattens, CMA-ES has found
    slack in M -- states M scores highly and the world does not produce.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.plot(hist["gen"], hist["best"], c="#3a7bd5", alpha=0.25, lw=0.8)
    ax.plot(hist["gen"], _smooth(hist["best"]), c="#3a7bd5",
            label=f"{fitness_label} (best, smoothed)")
    ax.set_xlabel("CMA-ES generation")
    ax.set_ylabel(fitness_label, color="#3a7bd5")
    ax.tick_params(axis="y", labelcolor="#3a7bd5")

    ax2 = ax.twinx()
    rg, rh = hist["real_gen"], np.asarray(hist["real_hits"])
    ci = np.asarray(hist["real_ci"])
    ax2.errorbar(rg, rh, yerr=[rh - ci[:, 0], ci[:, 1] - rh],
                 c="#d55e3a", marker="o", ms=4, capsize=3,
                 label=f"REAL {metric}/episode")
    ax2.set_ylabel(f"real {metric} per 200-step episode", color="#d55e3a")
    ax2.tick_params(axis="y", labelcolor="#d55e3a")

    # Two correlations, because the raw dream curve's per-generation jitter is
    # pure estimator noise and drags any correlation toward zero. The smoothed
    # one is the fair measure of "do the two trends move together"; the raw one
    # is reported alongside so the reader can see the cost of the noise.
    raw = np.interp(rg, hist["gen"], hist["best"])
    sm = np.interp(rg, hist["gen"], _smooth(hist["best"]))
    r_raw = float(np.corrcoef(raw, rh)[0, 1])
    r_sm = float(np.corrcoef(sm, rh)[0, 1])
    ax.set_title(f"does the dream transfer?   Pearson r = {r_sm:.2f} "
                 f"(smoothed) / {r_raw:.2f} (raw)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"path": str(path), "pearson_r_smoothed": r_sm, "pearson_r_raw": r_raw}


# --------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rnn", default="runs/rnn_v1/rnn.pt")
    p.add_argument("--vae", default="runs/vae_b1/vae.pt",
                   help="only needed for real-env evaluation / --fitness real")
    p.add_argument("--data", nargs="*",
                   default=["data/v1/train", "data/v1/train_mix"],
                   help="roots supplying dream start states and norm statistics")
    p.add_argument("--out", default="runs/ctrl_v1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--inputs", choices=["z", "h", "zh"], default="zh",
                   help="what the controller sees. 'z' has no velocity in it")
    p.add_argument("--fitness", choices=["dream", "real"], default="dream",
                   help="'real' is the no-world-model baseline")

    p.add_argument("--reward", choices=["hit", "dense", "mix"], default="mix")
    p.add_argument("--lam", type=float, default=0.1,
                   help="weight on the dense term in --reward mix")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="M's sampling temperature during the dream; >1 makes the "
                        "dream harder to exploit")
    p.add_argument("--dream-steps", type=int, default=150)
    p.add_argument("--warmup", type=int, default=8)

    p.add_argument("--popsize", type=int, default=32)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--sigma0", type=float, default=0.5)

    p.add_argument("--real-eval-every", type=int, default=5,
                   help="0 disables the periodic real-environment check")
    p.add_argument("--real-eval-episodes", type=int, default=16)
    p.add_argument("--real-eval-seed-base", type=int, default=7000,
                   help="deliberately NOT the final-eval seed base (5000), so the "
                        "model-selection episodes and the reported episodes are "
                        "disjoint")
    p.add_argument("--real-steps", type=int, default=200)
    p.add_argument("--ball-radius", type=float, default=0.08)
    p.add_argument("--real-eval-metric", choices=["hits", "interceptions"],
                   default="hits",
                   help="what the periodic real check reports and selects on. "
                        "'hits' is v1's contact-frame count (kept as the default "
                        "so v1 runs reproduce); 'interceptions' is the honest one")
    p.add_argument("--real-fitness-count", choices=["frames", "interceptions"],
                   default="frames",
                   help="what --fitness real maximises. 'frames' is v1's "
                        "(exploitable) contact-frame count; 'interceptions' "
                        "collapses each run of contact frames to one")
    from .eval_controller import add_env_args

    add_env_args(p)
    a = p.parse_args()
    train(a)


if __name__ == "__main__":
    main()

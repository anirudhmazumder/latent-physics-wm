#!/usr/bin/env python
"""Extract the series behind the index.html figures into assets/figdata.json.

The page used to embed matplotlib PNGs. assets/charts.js now draws the same
series as SVG in the browser; this script is the bridge. It reads only the
evaluation reports already on disk under runs/ -- nothing is recomputed and no
model is loaded -- so the interactive charts cannot disagree with the PNGs or
with the numbers in the READMEs.

Rerun with:

    /opt/miniconda3/envs/NN/bin/python scripts/build_figdata.py

One key per figure. Where each comes from, and which matplotlib function drew
the original:

  where_is_velocity   runs/rnn_v1/eval/velocity_probe.json
                      wm/eval_rnn.py, part (c). Two panels (linear, kNN), three
                      feature sets (z, h, z+h), six state variables. The
                      original clamps bars at -0.05; we keep the true value and
                      clamp only when drawing.

  cold_start_speed    runs/rnn_v2/causal/report.json -> a_cold_start
                      wm/eval_causal_v2.py, _plot_cold_start + _plot_speed_vs_warmup.
                      NOTE: the original left panel is a SCATTER of 325 per-start
                      points and those points were never saved to disk -- only
                      the OLS fit, the correlation and the marginal means/sds
                      survive in report.json. Panel A therefore draws the fit
                      lines, the identity line and a mean +/- sd marker per
                      split rather than the cloud. Panel B is the companion
                      warm-up sweep (also in report.json, and the source of the
                      caption's "by K>=2 motion suffices" claim).

  fix_comparison      runs/rnn_v2{,_mean,_ms,_cons,_ms_cons}/conservation/report.json
                      wm/eval_conservation.py, compare() -> runs/rnn_v2_fix_comparison.png.
                      per_tau["1.0"]["logmass_corr"] and ["speed_corr"] for each
                      run, plus probe_floor from the first run. The smoothing
                      (centred rolling mean, w=11, edge-padded) is done in JS so
                      only the raw curves ship. The summary table under the
                      original figure is not reproduced -- it is a table, and
                      the corr@k values it holds are points on these curves.

  skill_by_mass       runs/ctrl_eval_v2_fixed/summary.json -> in_distribution
                      wm/eval_controller_v2.py, plot_by_mass("interceptions_per_visit").
                      One bar per controller per mass tercile with its 95%
                      bootstrap CI; the oracle is a per-tercile reference segment.

  permanence_decay    runs/rnn_v3/permanence/report.json -> a_position_from_h.decay
                      wm/eval_permanence_v3.py, _plot_decay. The original draws
                      only the "(lin)" readouts plus the raw "z" kNN reference
                      and the no-memory baseline; we ship exactly those.

  skill_vs_bound      runs/ctrl_eval_v31/summary.json -> in_distribution
                      wm/eval_controller_v3.py, plot_skill_vs_bound. Bars sorted
                      ascending; oracle and wait_and_see become reference lines.

  two_seeds           runs/ctrl_eval_v31_seeds/two_seeds.json
                      wm/two_seeds.py, plot_two_seeds (metric "overall"). Pairs
                      sorted by the two-seed mean.

  v4_switch           runs/{rnn_v4,tf_v4,rnn_v4_ff,rnn_v4_noact,tf_v4_ctx32}/switch/report.json
                      wm/eval_switch_v4.py, fig_memory_curve. Three panels. The
                      z reference in panels (a)/(b) comes from the FIRST model's
                      report, as in the original (`any_rep`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figdata.json"

SIGFIG = 4


def rd(x: Any) -> Any:
    """Round floats to SIGFIG significant figures; NaN/inf -> None."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int,)):
        return x
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        if x == 0.0:
            return 0.0
        d = SIGFIG - 1 - math.floor(math.log10(abs(x)))
        v = round(x, max(d, 0) if d >= 0 else d)
        return int(v) if d <= 0 and v == int(v) else v
    if isinstance(x, list):
        return [rd(v) for v in x]
    if isinstance(x, dict):
        return {k: rd(v) for k, v in x.items()}
    return x


def load(p: str) -> dict:
    return json.loads((ROOT / p).read_text())


# ----------------------------------------------------------------- figures


def where_is_velocity() -> dict:
    d = load("runs/rnn_v1/eval/velocity_probe.json")
    vars_ = ["ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x", "paddle_vx"]
    feats = ["z", "h", "z+h"]
    panels = [("linear", "linear probe"), ("knn", "kNN probe")]
    return {
        "vars": vars_,
        "series": feats,
        "panels": [{"key": k, "label": lab} for k, lab in panels],
        "values": {k: {f: rd([d[f][k][v] for v in vars_]) for f in feats}
                   for k, _ in panels},
        "clip": -0.05,
        "ylabel": "held-out R²",
    }


def cold_start_speed() -> dict:
    r = load("runs/rnn_v2/causal/report.json")
    a = r["a_cold_start"]
    models = ["rnn_v2", "rnn_v2_nocolor"]
    labels = {"rnn_v2": "colour-seeing", "rnn_v2_nocolor": "colour-blind control"}
    fits = {}
    for m in models:
        fits[m] = {
            s: rd({k: a[f"{m}/{s}"][k] for k in
                   ("n", "pearson_r", "slope", "intercept",
                    "dreamed_speed_mean", "dreamed_speed_std",
                    "true_speed_mean", "true_speed_std")})
            for s in ("val", "holdout")
        }
    ks = [1, 2, 4, 8]
    warmup = {m: {"k": ks,
                  "r": rd([a["vs_warmup"][m][str(k)]["pearson_r"] for k in ks]),
                  "slope": rd([a["vs_warmup"][m][str(k)]["slope"] for k in ks])}
              for m in models}
    # The true law: speed = 0.022 / mass, mass log-uniform on [0.5, 2.0]; the
    # held-out band is report["holdout_band"].
    band = r["holdout_band"]
    return {
        "models": models,
        "labels": labels,
        "fits": fits,
        "warmup": warmup,
        "cold_horizon": r["cold_horizon"],
        "base_speed": 0.022,
        "mass_range": [0.5, 2.0],
        "true_speed_range": rd([0.022 / 2.0, 0.022 / 0.5]),
        "holdout_speed_range": rd([0.022 / band[1], 0.022 / band[0]]),
        "holdout_band": band,
    }


def fix_comparison() -> dict:
    runs = ["rnn_v2", "rnn_v2_mean", "rnn_v2_ms", "rnn_v2_cons", "rnn_v2_ms_cons"]
    tau = "1.0"
    series, floor = {}, {}
    for i, run in enumerate(runs):
        c = load(f"runs/{run}/conservation/report.json")
        per = c["per_tau"][tau]
        series[run] = {"logmass_corr": rd(per["logmass_corr"]),
                       "speed_corr": rd(per["speed_corr"])}
        if i == 0:
            floor = {"logmass_corr": rd(c["probe_floor"]["logmass_corr"]),
                     "speed_corr": rd(c["probe_floor"]["speed_corr"])}
    return {
        "runs": runs,
        "tau": 1.0,
        "smooth_window": 11,
        "series": series,
        "probe_floor": floor,
        "panels": [
            {"key": "logmass_corr",
             "label": "(i) dreamed log-mass vs true log-mass"},
            {"key": "speed_corr", "label": "(iii b) dreamed speed vs the law"},
        ],
        "good_line": 0.8,
        "ylabel": "corr across episodes",
        "xlabel": "dream step",
    }


def skill_by_mass() -> dict:
    s = load("runs/ctrl_eval_v2_fixed/summary.json")
    terciles = ["light", "medium", "heavy"]
    rows = s["in_distribution"]
    key = "interceptions_per_visit"
    oracle = next(r for r in rows if r["name"] == "oracle")
    bars = []
    for r in rows:
        if r["name"] == "oracle":
            continue
        bars.append({
            "name": r["name"],
            "values": rd([r[t][key] for t in terciles]),
            "ci": rd([r[t].get(key + "_ci95") for t in terciles]),
        })
    return {
        "terciles": terciles,
        "tercile_labels": ["light\n(fast)", "medium", "heavy\n(slow)"],
        "bars": bars,
        "oracle": rd([oracle[t][key] for t in terciles]),
        "baselines": ["stay", "random"],
        "ylabel": "interceptions per floor visit",
        "tercile_cuts": rd(s["tercile_cuts"]),
    }


def permanence_decay() -> dict:
    d = load("runs/rnn_v3/permanence/report.json")["a_position_from_h"]["decay"]
    keep = [f for f in d["per_feature"] if f.endswith("(lin)") or f == "z"]
    series = {f: {"k": d["per_feature"][f]["k"],
                  "rmse": rd(d["per_feature"][f]["rmse"]),
                  "r2_x": rd(d["per_feature"][f]["r2_x"])}
              for f in keep}
    return {
        "series": series,
        "order": keep,
        "no_memory": {"k": d["no_memory"]["k"], "rmse": rd(d["no_memory"]["rmse"])},
        "panels": [
            {"key": "rmse", "label": "position rmse (world units)"},
            {"key": "r2_x", "label": "R², ball_x"},
        ],
        "xlabel": "frames the ball has been fully hidden",
    }


def _bound_rows(path: str, key: str = "interceptions_per_visit") -> dict:
    s = load(path)
    rows = s["in_distribution"]
    refs = {r["name"]: r[key] for r in rows
            if r["name"] in ("oracle", "wait_and_see")}
    bars = sorted([r for r in rows if r["name"] not in ("oracle", "wait_and_see")],
                  key=lambda r: r[key])
    return {
        "bars": [{"name": r["name"], "value": rd(r[key]),
                  "ci": rd(r.get(key + "_ci95"))} for r in bars],
        "refs": rd(refs),
    }


def skill_vs_bound() -> dict:
    d = _bound_rows("runs/ctrl_eval_v31/summary.json")
    d["baselines"] = ["stay", "random"]
    d["ylabel"] = "interceptions per floor visit"
    return d


def two_seeds() -> dict:
    d = load("runs/ctrl_eval_v31_seeds/two_seeds.json")
    metric = "overall"
    ranked = sorted(d["pairs"], key=lambda p: p[metric]["mean"])
    return {
        "pairs": [{"name": p["name"],
                   "seed0": rd(p[metric]["seed0"]),
                   "seed1": rd(p[metric]["seed1"]),
                   "mean": rd(p[metric]["mean"]),
                   "spread": rd(p[metric]["spread"])} for p in ranked],
        "refs": rd({k: v for k, v in d["references"].items()
                    if k in ("oracle", "wait_and_see")}),
        "ylabel": "interceptions / floor visit",
    }


V4_MODELS = [
    ("lstm", "runs/rnn_v4"),
    ("transformer", "runs/tf_v4"),
    ("ff", "runs/rnn_v4_ff"),
    ("noact", "runs/rnn_v4_noact"),
    ("tf_ctx32", "runs/tf_v4_ctx32"),
]


def v4_switch() -> dict:
    reports = {n: load(f"{p}/switch/report.json") for n, p in V4_MODELS}
    first = reports[V4_MODELS[0][0]]["a"]
    bins = [b["bin"] for b in first["bins_h"]]
    z_bal = rd([b["balanced_acc"] for b in first["bins_z"]])
    z_plain = [b["acc"] for b in first["bins_z"]]
    models = {}
    for name, _ in V4_MODELS:
        a = reports[name]["a"]
        models[name] = {
            "hidden_dim": a["hidden_dim"],
            "balanced": rd([b["balanced_acc"] for b in a["bins_h"]]),
            "n": [b["n"] for b in a["bins_h"]],
            "memory": rd(reports[name]["b"].get("memory_first_10_after_flip")),
            "inference": rd(reports[name]["b"].get("inference_50plus_cold_start")),
        }
    nulls = [reports[n]["a"]["overall_null_balanced"] for n, _ in V4_MODELS]
    return {
        "bins": bins,
        "order": [n for n, _ in V4_MODELS],
        "models": models,
        "z_balanced": z_bal,
        "z_ref_memory": rd(z_plain[0]),
        "z_ref_inference": rd(sum(z_plain[3:5]) / 2.0),
        "null_balanced": rd(max(nulls)),
        "small_bin": 400,
        "xlabel": "frames since the last flip",
    }


FIGURES = {
    "where_is_velocity": where_is_velocity,
    "cold_start_speed": cold_start_speed,
    "fix_comparison": fix_comparison,
    "skill_by_mass": skill_by_mass,
    "permanence_decay": permanence_decay,
    "skill_vs_bound": skill_vs_bound,
    "two_seeds": two_seeds,
    "v4_switch": v4_switch,
}


def main() -> None:
    data = {name: fn() for name, fn in FIGURES.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), sort_keys=False))
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  {kb:.1f} KB  ({len(data)} figures)")
    for k in data:
        print(f"  {k}")


if __name__ == "__main__":
    main()

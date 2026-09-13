"""Two training seeds per controller row: the table and the paired-bar figure.

    python -m wm.two_seeds --summary runs/ctrl_eval_v31_seeds/summary.json \
        --out runs/ctrl_eval_v31_seeds

Why this file exists
--------------------
`README_C31.md` §4 concludes an ORDERING -- fair > privileged > real-trained >
z-only ~ the memoryless bound > the feed-forward floor -- and every row in it is
one CMA-ES run. CMA-ES searching 819 parameters against a fitness estimated
from 16 noisy dream rollouts is not a deterministic map from settings to skill,
and several of the gaps being read are 0.05-0.08 interceptions per floor visit,
which is inside the width of a single row's own confidence interval. So the
ordering might be a property of the settings or a property of the draw, and one
seed cannot tell the two apart.

This module does the reading-off, once the evaluator has scored both seeds on
the same 150 episodes. It owns no rollout and no metric: it takes
``summary.json`` from ``wm.eval_controller_v3`` and pairs rows by name, on the
convention that seed 1 of ``X`` is called ``X_s1``.

What "spread" means here, and what it does not
----------------------------------------------
``spread`` is |seed0 - seed1|, reported raw. With two samples a standard
deviation is just that number over sqrt(2) and invites being read as an error
bar, which it is not -- two draws say almost nothing about the shape of the
distribution they came from. The honest use is a comparison: an ordering whose
gap is smaller than the spread of either row it separates is **unresolved**,
and this module labels it so rather than leaving the reader to do the
arithmetic. Note that the per-row CI printed by the evaluator and the spread
measured here are different quantities: the CI is the uncertainty in scoring
ONE policy on 150 episodes, the spread is the variation between two policies
the same recipe produced. A row can have a tight CI and a large spread, and
that combination is the interesting one -- it means the recipe, not the
measurement, is what is noisy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .clock import seed_summary

# The three numbers Part B tracks, and where each lives in a summary row.
# "overall" is the headline; "long" is the bin where a memoryless policy
# collapses and therefore the one the permanence claim rests on;
# "displacement" is the outcome-free version of "did it commit in the dark".
METRICS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "overall": (("interceptions_per_visit",), "interceptions / floor visit"),
    "long": (("by_required_move", "long", "interceptions_per_visit"),
             "interceptions / visit, long moves (>0.35)"),
    "displacement": (("occlusion_long", "displacement_fraction"),
                     "displacement while blind / required move"),
}
SEED_SUFFIX = "_s1"


def _dig(row: Dict, path: Sequence[str]) -> float:
    cur: object = row
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return float("nan")
        cur = cur[k]
    return float(cur) if isinstance(cur, (int, float)) else float("nan")


def pair_rows(rows: Sequence[Dict],
              bases: Optional[Sequence[str]] = None) -> List[Dict]:
    """Group ``X`` with ``X_s1`` and compute mean/spread for every metric.

    ``bases`` fixes the ORDER of the table; when omitted, every row that has an
    ``_s1`` partner is paired, in the order the evaluator produced them. Rows
    with no partner (the references, and any ``_lastdream`` variant that was
    only trained once) are skipped -- they are reported by the evaluator's own
    table and adding a one-seed row to a two-seed table would invite exactly
    the reading this file exists to prevent.
    """
    by_name = {r["name"]: r for r in rows}
    if bases is None:
        bases = [n for n in by_name if n + SEED_SUFFIX in by_name]
    out: List[Dict] = []
    for base in bases:
        s0, s1 = by_name.get(base), by_name.get(base + SEED_SUFFIX)
        if s0 is None or s1 is None:
            print(f"  (skipping {base}: "
                  f"{'seed 0' if s0 is None else 'seed 1'} not in the summary)")
            continue
        entry: Dict[str, object] = {"name": base}
        for key, (path, _) in METRICS.items():
            v0, v1 = _dig(s0, path), _dig(s1, path)
            entry[key] = {"seed0": v0, "seed1": v1, **seed_summary([v0, v1])}
        entry["n_visits"] = [s0.get("n_visits"), s1.get("n_visits")]
        out.append(entry)
    return out


def references(rows: Sequence[Dict]) -> Dict[str, float]:
    """The two lines every row is graded against, if they were scored."""
    return {r["name"]: float(r["interceptions_per_visit"]) for r in rows
            if r["name"] in ("oracle", "wait_and_see", "stay", "random")}


# ------------------------------------------------------------- what survives


def surviving_orderings(pairs: Sequence[Dict], metric: str = "overall",
                        ) -> List[Dict]:
    """Every adjacent pair in the mean-sorted ranking, labelled resolved or not.

    "Resolved" here is deliberately strict and deliberately simple: the gap
    between the two rows' MEANS must exceed the larger of the two rows'
    spreads, AND the two rows' [min, max] seed ranges must not overlap. The
    second condition is the one that does the work -- two rows whose seeds
    interleave are not ordered by anything this experiment measured, whatever
    their means do.
    """
    ranked = sorted(pairs, key=lambda p: -p[metric]["mean"])
    out = []
    for hi, lo in zip(ranked, ranked[1:]):
        a, b = hi[metric], lo[metric]
        gap = a["mean"] - b["mean"]
        overlap = a["min"] <= b["max"]
        out.append({
            "above": hi["name"], "below": lo["name"], "gap": gap,
            "spread_above": a["spread"], "spread_below": b["spread"],
            "seeds_overlap": bool(overlap),
            "resolved": bool(not overlap
                             and gap > max(a["spread"], b["spread"])),
        })
    return out


def vs_bound(pairs: Sequence[Dict], bound: float,
             metric: str = "overall") -> List[Dict]:
    """Is this row above the memoryless bound on BOTH seeds, or only on one?"""
    out = []
    for p in pairs:
        m = p[metric]
        both = m["min"] > bound
        neither = m["max"] <= bound
        out.append({
            "name": p["name"], "mean": m["mean"],
            "verdict": "both seeds" if both
            else "neither seed" if neither else "ONE SEED ONLY",
        })
    return out


# -------------------------------------------------------------------- output


def _f(v: float, nd: int = 2) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def table(pairs: Sequence[Dict], metric: str) -> List[str]:
    lines = [f"| controller | seed 0 | seed 1 | mean | spread |",
             "|---|---|---|---|---|"]
    for p in sorted(pairs, key=lambda q: -q[metric]["mean"]):
        m = p[metric]
        lines.append(f"| `{p['name']}` | {_f(m['seed0'])} | {_f(m['seed1'])} "
                     f"| **{_f(m['mean'])}** | {_f(m['spread'])} |")
    return lines


def plot_two_seeds(pairs: Sequence[Dict], refs: Dict[str, float], out: Path,
                   metric: str = "overall") -> Path:
    """Paired bars, one pair per row, sorted by the two-seed mean.

    The point of the figure is the DISTANCE BETWEEN THE TWO BARS of a pair
    relative to the distance between pairs. If the within-pair gaps are as big
    as the between-pair gaps, the ranking in `README_C31` §4 is a ranking of
    draws. Drawn as two solid bars rather than a mean with an error bar
    precisely so that the reader cannot mistake two points for a distribution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(pairs, key=lambda p: p[metric]["mean"])
    x = np.arange(len(ranked))
    w = 0.38
    v0 = [p[metric]["seed0"] for p in ranked]
    v1 = [p[metric]["seed1"] for p in ranked]
    bound = refs.get("wait_and_see", float("nan"))

    fig, ax = plt.subplots(figsize=(3.6 + 0.95 * len(ranked), 5.2))
    ax.bar(x - w / 2, v0, w, label="seed 0", color="#3d6fb0")
    ax.bar(x + w / 2, v1, w, label="seed 1", color="#8fb4dd")
    # A thin connector makes the within-pair spread readable at a glance, which
    # is the whole question.
    for xi, a, b in zip(x, v0, v1):
        ax.plot([xi - w / 2, xi + w / 2], [a, b], color="#333333", lw=1.0,
                marker="_", ms=6, zorder=5)
    for name, style, label in (
        ("oracle", dict(c="#2a9d4a", ls="-", lw=2.0), "oracle (vision + memory)"),
        ("wait_and_see", dict(c="#d55e3a", ls="--", lw=2.0),
         "wait-and-see (vision, NO memory)"),
    ):
        if name in refs:
            ax.axhline(refs[name], **style, label=f"{label} = {refs[name]:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([p["name"] for p in ranked], rotation=35, ha="right",
                       fontsize=7)
    ax.set_ylabel(METRICS[metric][1])
    ax.set_ylim(0, max(1.12, np.nanmax(v0 + v1) * 1.15))
    ax.set_title("two training seeds per row\n"
                 "(the gap WITHIN a pair is what the single-seed table could "
                 "not see)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    if np.isfinite(bound):
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default="runs/ctrl_eval_v31_seeds/summary.json",
                   help="summary.json written by wm.eval_controller_v3")
    p.add_argument("--rows", nargs="*", default=None,
                   help="base run names, in table order. Default: every row "
                        "that has an _s1 partner.")
    p.add_argument("--out", default="runs/ctrl_eval_v31_seeds")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(a.summary).read_text())["in_distribution"]
    pairs = pair_rows(rows, a.rows)
    refs = references(rows)
    if not pairs:
        raise SystemExit("no seed-paired rows found in " + a.summary)

    fig = plot_two_seeds(pairs, refs, out / "skill_vs_bound_two_seeds.png")
    order = surviving_orderings(pairs)
    bound = refs.get("wait_and_see", float("nan"))

    md: List[str] = ["# Two training seeds per controller row", ""]
    md.append(f"References: oracle **{_f(refs.get('oracle', float('nan')))}**, "
              f"wait-and-see (the memoryless bound) **{_f(bound)}**, "
              f"stay {_f(refs.get('stay', float('nan')))}, "
              f"random {_f(refs.get('random', float('nan')))}.")
    md.append("")
    md.append(f"![two seeds]({fig.name})")
    for key, (_, label) in METRICS.items():
        md += ["", f"## {label}", ""] + table(pairs, key)
    md += ["", "## Above the memoryless bound?", "",
           "| controller | two-seed mean | above the bound on |",
           "|---|---|---|"]
    for r in vs_bound(pairs, bound):
        md.append(f"| `{r['name']}` | {_f(r['mean'])} | {r['verdict']} |")
    # The ordering test is run on ALL THREE metrics, not just the headline.
    # That is not thoroughness for its own sake: "interceptions per visit" is
    # an outcome, and an outcome is the noisiest thing here -- it mixes how
    # well the policy committed in the dark with how lucky the 150 episodes'
    # ball trajectories were. The long-move bin and the displacement fraction
    # are closer to the behaviour, and if an ordering survives anywhere it is
    # likelier to survive there. Reporting only the headline would let a
    # resolved ordering go unnoticed.
    orders = {k: surviving_orderings(pairs, k) for k in METRICS}
    for key, (_, label) in METRICS.items():
        md += ["", f"## Which adjacent orderings survive two seeds -- {label}",
               "",
               "| above | below | gap of means | larger spread | seeds overlap | resolved |",
               "|---|---|---|---|---|---|"]
        for o in orders[key]:
            md.append(
                f"| `{o['above']}` | `{o['below']}` | {_f(o['gap'])} | "
                f"{_f(max(o['spread_above'], o['spread_below']))} | "
                f"{'yes' if o['seeds_overlap'] else 'no'} | "
                f"{'**yes**' if o['resolved'] else 'no'} |")

    # Adjacent pairs are the strict reading, but they under-report: a row can
    # be unambiguously above one three places below it while every step of the
    # ladder between them is unresolved. So the non-adjacent separations are
    # listed too, on the same seeds-do-not-overlap rule.
    md += ["", "## Non-adjacent separations that hold on both seeds", "",
           "Every pair (not only neighbours) whose two seed ranges do not "
           "overlap at all. These are the orderings this experiment actually "
           "established.", "",
           "| metric | above | below | above's seeds | below's seeds |",
           "|---|---|---|---|---|"]
    n_sep = 0
    for key, (_, label) in METRICS.items():
        ranked = sorted(pairs, key=lambda p: -p[key]["mean"])
        for i, hi in enumerate(ranked):
            for lo in ranked[i + 1:]:
                a, b = hi[key], lo[key]
                if a["min"] > b["max"]:
                    n_sep += 1
                    md.append(
                        f"| {label} | `{hi['name']}` | `{lo['name']}` | "
                        f"[{_f(a['min'])}, {_f(a['max'])}] | "
                        f"[{_f(b['min'])}, {_f(b['max'])}] |")
    if n_sep == 0:
        md.append("| — | — | — | — | — |")

    (out / "two_seeds.md").write_text("\n".join(md) + "\n")
    (out / "two_seeds.json").write_text(json.dumps(
        {"pairs": pairs, "references": refs, "orderings": order,
         "orderings_by_metric": orders,
         "vs_bound": {k: vs_bound(pairs, bound, k) for k in METRICS}},
        indent=2, default=float))
    print("\n".join(md))
    print(f"\n-> {out / 'two_seeds.md'}")


if __name__ == "__main__":
    main()

"""One figure and one table comparing the v3 permanence fixes.

    python -m wm.compare_fix_v3 \
        --permanence runs/rnn_v3_fix/permanence/report.json \
        --runs rnn_v3=runs/rnn_v3 ms24=runs/rnn_v3_ms24 \
               emerge=runs/rnn_v3_emerge poshead=runs/rnn_v3_poshead \
        --out runs/rnn_v3_fix_comparison.png

Nothing here computes anything about a model: every number is read back out of
the reports that ``eval_permanence_v3``, ``eval_rnn`` and ``eval_conservation``
already wrote, so the figure cannot disagree with the tables it summarises.

The two panels are the two halves of the defect (``wm/README_M3.md`` Section 10):

  left   how much of the hidden ball's **x** a linear probe can read out of
         ``h``, as a function of how long it has been hidden, against the
         no-memory baseline ("the ball is where it vanished");
  right  the consequence -- the error in the dreamed exit position, binned by
         how long the occlusion lasted, against that baseline and against
         straight-line extrapolation from the entry velocity.

The no-memory baseline is scored on the same frames, the same episode-level
split and the same axis as the probes (``eval_permanence_v3`` writes its
per-axis R^2 alongside theirs), so the left panel is a like-for-like
comparison and the crossing point on it is the number that decides the fix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BINS = ["3-8", "9-15", "16-25", "26-+"]


def _load(path) -> Optional[dict]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# --------------------------------------------------------------- the numbers


def table_rows(perm: dict, runs: Dict[str, Path], names: List[str]) -> List[dict]:
    """One dict per model, pulling each number from the report that owns it."""
    rows = []
    for n in names:
        bins = _get(perm, "a_position_from_h", "bins", f"h[{n}]", "hidden",
                    default={})
        extras = _get(perm, "a_position_from_h", "hidden_extras", f"h[{n}]",
                      default={})
        emg = _get(perm, "b_emergence", "tables", "all", n, default={})
        bounce = _get(perm, "c_wall_bounce_pooled", n, default={})

        run = runs.get(n)
        ev = _load(run / "eval" / "report.json") if run else None
        cons = _load(run / "conservation" / "report.json") if run else None
        hist = _load(run / "history.json") if run else None

        rows.append({
            "model": n,
            "hidden_r2_x": _get(bins, "linear", "ball_x", "r2"),
            "hidden_r2_y": _get(bins, "linear", "ball_y", "r2"),
            "hidden_r2_vx": _get(extras, "linear", "ball_vx", "r2"),
            "hidden_r2_vy": _get(extras, "linear", "ball_vy", "r2"),
            "exit_x_mae": emg.get("exit_x_mae"),
            "exit_t_mae": emg.get("exit_time_mae"),
            "side_correct": emg.get("side_correct"),
            "censored": emg.get("censored_frac"),
            "bounce_frac": bounce.get("frac_nearer_reflected"),
            "bounce_n": bounce.get("n_scored"),
            "bounce_exit_x_mae": bounce.get("exit_x_mae"),
            "visible_horizon": _get(ev, "b_state", "tau0.0",
                                    "useful_dream_horizon_visible"),
            "vel_r2_vx": _get(ev, "c_velocity", "h", "linear", "ball_vx"),
            "paddle_sep": _get(ev, "d_actions", "paddle_sep_at_h30"),
            "contact_pr_auc": _get(ev, "e_events", "pr_auc"),
            "hidden_vy_kept": _get(cons, "summary", "0.0",
                                   "hidden_vy_sign_preserved"),
            "val_nll": min((r["val_nll"] for r in hist), default=None) if hist
            else None,
        })
    return rows


def baseline_row(perm: dict, which: str) -> dict:
    emg = _get(perm, "b_emergence", "tables", "all", which, default={})
    bounce = _get(perm, "c_wall_bounce_pooled", which, default={})
    # The no-memory baseline is HANDED the true exit time -- that is what makes
    # it a pure test of x -- so its exit-time and side columns are not a score
    # and are blanked rather than printed as a perfect 0.
    given = which == "no_memory"
    return {
        "model": which,
        "exit_x_mae": emg.get("exit_x_mae"),
        "exit_t_mae": None if given else emg.get("exit_time_mae"),
        "side_correct": None if given else emg.get("side_correct"),
        "bounce_frac": bounce.get("frac_nearer_reflected"),
        "bounce_n": bounce.get("n_scored"),
        "bounce_exit_x_mae": bounce.get("exit_x_mae"),
    }


# ---------------------------------------------------------------- rendering


def _fmt(v, spec=".3f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return format(v, spec) if isinstance(v, float) else str(v)


COLUMNS = [
    ("model", "model", "s"),
    ("hidden_r2_x", "R2 x|hid", ".2f"),
    ("hidden_r2_vx", "R2 vx|hid", ".2f"),
    ("exit_x_mae", "exit-x MAE", ".3f"),
    ("exit_t_mae", "exit-t MAE", ".2f"),
    ("side_correct", "side ok", ".2f"),
    ("censored", "censor", ".2f"),
    ("bounce_frac", "bounce ok", ".2f"),
    ("visible_horizon", "vis horizon", "d"),
    ("vel_r2_vx", "R2 vx (all)", ".2f"),
    ("contact_pr_auc", "PR-AUC", ".3f"),
    ("hidden_vy_kept", "vy kept", ".2f"),
    ("val_nll", "val NLL", ".3f"),
]


def markdown_table(rows: List[dict]) -> str:
    head = "| " + " | ".join(h for _, h, _ in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    out = [head, rule]
    for r in rows:
        cells = []
        for key, _, spec in COLUMNS:
            v = r.get(key)
            cells.append(v if spec == "s" else _fmt(v, spec if spec != "d" else ""))
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(out)


def make_figure(perm: dict, rows: List[dict], names: List[str], path: Path,
                baselines: List[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.38,
                          wspace=0.22)
    ax_decay = fig.add_subplot(gs[0, 0])
    ax_exit = fig.add_subplot(gs[0, 1])
    ax_tab = fig.add_subplot(gs[1, :])

    decay = _get(perm, "a_position_from_h", "decay", default={})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # -------------------------------------------------- (1) decay of x from h
    for i, n in enumerate(names):
        row = _get(decay, "per_feature", f"h[{n}] (lin)")
        if row is None:
            continue
        ax_decay.plot(row["k"], row["r2_x"], marker="o", ms=3,
                      color=colors[i % 10], label=f"h[{n}]")
    ff = _get(decay, "per_feature", "h[ff] (lin)")
    if ff is not None:
        ax_decay.plot(ff["k"], ff["r2_x"], color="0.45", ls="--", lw=1.6,
                      label="h[ff] (no-memory floor)")

    nm = _get(decay, "no_memory", default={})
    if nm and "r2_x" in nm:
        ax_decay.plot(nm["k"], nm["r2_x"], color="k", ls=":", lw=2.0,
                      label="no-memory baseline")
    ax_decay.axhline(0.0, color="0.7", lw=0.8)
    ax_decay.set_xlabel("frames the ball has been hidden, k")
    ax_decay.set_ylabel(r"linear probe $R^2$ for ball_x from $h$")
    ax_decay.set_title("(1) how much of the hidden ball's x is in h\n"
                       "pooled over default / tall / taller bands")
    ax_decay.set_ylim(-1.0, 1.0)
    ax_decay.grid(alpha=0.3)
    ax_decay.legend(fontsize=7.5, loc="lower left", ncol=2)

    # ------------------------------------------- (2) exit x vs hidden duration
    pooled = _get(perm, "d_memory_horizon", "pooled", default={})
    xs = np.arange(len(BINS))
    for i, n in enumerate(names):
        p = pooled.get(n)
        if p is None:
            continue
        y = [p[b]["exit_x_mae"] if p[b]["n_scored"] >= 5 else np.nan for b in BINS]
        ax_exit.plot(xs, y, marker="o", color=colors[i % 10], label=n)
    for n, style in (("ff", dict(color="0.45", ls="--")),
                     ("no_memory", dict(color="k", ls=":", lw=2.0)),
                     ("linear", dict(color="tab:red", ls="-.", lw=1.4))):
        p = pooled.get(n)
        if p is None:
            continue
        y = [p[b]["exit_x_mae"] if p[b]["n_scored"] >= 5 else np.nan for b in BINS]
        ax_exit.plot(xs, y, marker="s", ms=3, label=n, **style)
    ax_exit.set_xticks(xs)
    ax_exit.set_xticklabels(BINS)
    ax_exit.set_xlabel("true hidden duration (frames)")
    ax_exit.set_ylabel("exit-x MAE (world units)")
    ax_exit.set_title("(2) dreamed exit position vs occlusion length\n"
                      "cells with < 5 scored runs are dropped")
    ax_exit.grid(alpha=0.3)
    ax_exit.legend(fontsize=7.5, ncol=2)

    # ----------------------------------------------------------- (3) the table
    ax_tab.axis("off")
    body = []
    for r in rows + baselines:
        body.append([r.get("model")] + [
            _fmt(r.get(k), spec if spec != "d" else "")
            for k, _, spec in COLUMNS[1:]])
    tab = ax_tab.table(cellText=body,
                       colLabels=[h for _, h, _ in COLUMNS],
                       loc="upper center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    tab.scale(1.0, 1.28)
    for j in range(len(COLUMNS)):
        tab[0, j].set_facecolor("#dddddd")
    for i, r in enumerate(rows + baselines, start=1):
        if r.get("model") in ("no_memory", "linear", "ff"):
            for j in range(len(COLUMNS)):
                tab[i, j].set_facecolor("#f2f2f2")
    ax_tab.set_title(
        "(3) every headline number, default band unless stated.  "
        "'bounce ok' is pooled over the three bands.  "
        "poshead uses a PRIVILEGED position target: it is a ceiling, not a result.",
        fontsize=9, pad=14)

    fig.suptitle("v3 object permanence: the horizontal component, and the "
                 "attempts to fix it", fontsize=13)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--permanence", default="runs/rnn_v3_fix/permanence/report.json")
    p.add_argument("--runs", nargs="+", required=True, help="name=runs/dir")
    p.add_argument("--out", default="runs/rnn_v3_fix_comparison.png")
    a = p.parse_args()

    perm = _load(a.permanence)
    if perm is None:
        raise SystemExit(f"{a.permanence} not found")
    runs = {}
    for spec in a.runs:
        n, d = spec.split("=", 1)
        runs[n] = Path(d)
    names = list(runs)

    rows = table_rows(perm, runs, names)
    baselines = [baseline_row(perm, "no_memory"), baseline_row(perm, "linear")]
    make_figure(perm, rows, names, Path(a.out), baselines)

    md = markdown_table(rows + baselines)
    print(md)
    Path(a.out).with_suffix(".md").write_text(md + "\n")
    Path(a.out).with_suffix(".json").write_text(
        json.dumps({"rows": rows, "baselines": baselines}, indent=2,
                   default=float))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()

"""Shared test fixtures and skip helpers.

A fresh clone of this repository has **no datasets** — `data/` is gitignored and
is regenerated with `bash scripts/collect_all.sh` (see `data/README.md`). It
does ship every checkpoint under `runs/`, but a checkout that has had `runs/`
pruned, or one where a particular run was never trained, will not have all of
them.

Most tests here are self-contained: they build a random model or a hand-made
array and check a property. A minority load a real dataset or a real
checkpoint, and those are exactly the tests that would catch a wiring mistake,
so they are skipped rather than deleted when their input is missing.

    from tests.conftest import require_data, require_ckpt

    def test_something() -> None:
        root = require_data(ROOT / "data" / "v1" / "val" / "frames.npy")
        ...

Both helpers raise :class:`unittest.SkipTest`, which pytest reports as a skip
and which the ``python -m tests.test_*`` runners at the bottom of each test file
catch and report as ``skip``. So the two ways of running a test file agree.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve(path: "str | Path") -> Path:
    """Interpret a relative path against the repo root, not the shell's cwd.

    Test files use both conventions (``ROOT / "data" / ...`` and the bare
    ``"runs/vae_b1/vae.pt"``), and pytest does not chdir, so a run started from
    a subdirectory would otherwise resolve the bare ones to nothing.
    """
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def require_data(path: "str | Path", what: str | None = None) -> Path:
    """Return ``path``, or skip the current test because the dataset is absent.

    `path` is a file inside a dataset split (usually ``frames.npy``,
    ``mu.npy`` or ``meta.json``) or the split directory itself.
    """
    p = _resolve(path)
    if not p.exists():
        rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        raise unittest.SkipTest(
            f"dataset not present: {rel}"
            + (f" ({what})" if what else "")
            + " — regenerate with `bash scripts/collect_all.sh`"
              " (see data/README.md)"
        )
    return p


def require_ckpt(path: "str | Path", what: str | None = None) -> Path:
    """Return ``path``, or skip the current test because the checkpoint is absent."""
    p = _resolve(path)
    if not p.exists():
        rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        raise unittest.SkipTest(
            f"checkpoint not present: {rel}"
            + (f" ({what})" if what else "")
            + " — it ships in `runs/`; see the run logs in wm/ to retrain it"
        )
    return p


def have(*paths: "str | Path") -> bool:
    """True when every path exists. For tests that want to branch, not skip."""
    return all(_resolve(p).exists() for p in paths)

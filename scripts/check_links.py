#!/usr/bin/env python3
"""Check that every relative link in every Markdown file in the repo resolves.

    python scripts/check_links.py            # whole repo, exit 1 on any breakage
    python scripts/check_links.py docs        # one subtree

Checks inline links and images (``[text](target)`` and ``![alt](target)``) plus
reference definitions (``[label]: target``). Absolute URLs (``http:``, ``https:``,
``mailto:``) and bare in-page anchors (``#section``) are left alone; a
``file.md#anchor`` target is checked for the file only.

Skips fenced code blocks, so a command line that happens to contain brackets is
not mistaken for a link. Targets that are gitignored by design — anything under
`data/`, and the per-run `*.log` console output — are reported as a separate,
non-fatal "not shipped" category, because a fresh clone legitimately does not
have them.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# [text](target)  /  ![alt](target) -- target stops at whitespace or ')'
INLINE = re.compile(r"!?\[[^\]]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
# [label]: target
REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
FENCE = re.compile(r"^\s*(```|~~~)")

EXTERNAL = ("http://", "https://", "mailto:", "ftp://", "//")
# Gitignored by design: a link to one of these is not a repo defect, it just
# will not resolve on a fresh clone. Kept visible, but non-fatal.
IGNORED_TREES = ("data/",)
IGNORED_SUFFIXES = (".log",)


def strip_code_fences(text: str) -> str:
    out, in_fence = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return "\n".join(out)


def targets(text: str):
    body = strip_code_fences(text)
    for m in INLINE.finditer(body):
        yield m.group(1)
    for m in REFDEF.finditer(body):
        yield m.group(1)


def markdown_files(subtree: Path):
    try:
        listed = subprocess.check_output(
            ["git", "ls-files", "-z", "*.md"], cwd=ROOT, text=True
        ).split("\0")
        files = [ROOT / p for p in listed if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = sorted(ROOT.rglob("*.md"))
    return [f for f in files if subtree in f.parents or f == subtree]


def main() -> int:
    subtree = (ROOT / sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
    files = markdown_files(subtree)

    broken: list[tuple[Path, str]] = []
    ignored: list[tuple[Path, str]] = []
    checked = 0

    for md in files:
        text = md.read_text(encoding="utf-8", errors="replace")
        for raw in targets(text):
            if raw.startswith(EXTERNAL) or raw.startswith("#"):
                continue
            target = raw.split("#", 1)[0]
            if not target:          # pure in-page anchor
                continue
            checked += 1
            resolved = (md.parent / target).resolve()
            rel_from_root = str(target).lstrip("./")
            if resolved.exists():
                continue
            if (target.endswith(IGNORED_SUFFIXES)
                    or any(rel_from_root.startswith(t) or f"/{t}" in str(resolved)
                           for t in IGNORED_TREES)):
                ignored.append((md, raw))
            else:
                broken.append((md, raw))

    print(f"checked {checked} relative links in {len(files)} Markdown files")
    if ignored:
        print(f"\n{len(ignored)} link(s) to files not shipped in git "
              f"(not an error — datasets are regenerated with "
              f"scripts/collect_all.sh; *.log is each run's console output):")
        for md, raw in ignored:
            print(f"  {md.relative_to(ROOT)} -> {raw}")
    if broken:
        print(f"\n{len(broken)} BROKEN link(s):")
        for md, raw in broken:
            print(f"  {md.relative_to(ROOT)} -> {raw}")
        return 1
    print("\nall relative links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

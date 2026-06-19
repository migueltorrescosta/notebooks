#!/usr/bin/env python3
"""Check for old-style ``plt.cm.`` / ``matplotlib.cm`` imports.

Migrated API: use ``plt.colormaps["..."]`` instead of ``plt.cm.NAME``.

Usage::

    python scripts/check_plt_cm.py          # check all .py files
    python scripts/check_plt_cm.py --fix     # not implemented (informational)

Exit codes:
    0  — no violations found
    1  — at least one violation found
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Patterns that indicate the old-style API
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("plt.cm.NAME (attribute access)", re.compile(r"plt\.cm\.")),
    (
        "import matplotlib.cm (direct import)",
        re.compile(r"^\s*import\s+matplotlib\.cm\b", re.MULTILINE),
    ),
    (
        "from matplotlib import cm (from-import)",
        re.compile(r"^\s*from\s+matplotlib\s+import\s+cm\b", re.MULTILINE),
    ),
]

# Directories to skip entirely
SKIP_DIRS = {
    ".venv",
    ".git",
    ".ruff_cache",
    "__pycache__",
    "jupyter",
    "mathematica",
    ".opencode",
}

_IGNORE_SUFFIXES = {".py"}


def _skip_dir(name: str) -> bool:
    return name.startswith(".") or name in SKIP_DIRS


def find_violations(root: Path) -> list[tuple[Path, int, str, str]]:
    """Scan *root* for old-style colormap access.

    Returns a list of ``(file, lineno, line, pattern_label)`` tuples.
    """
    violations: list[tuple[Path, int, str, str]] = []
    for py_file in sorted(root.rglob("*.py")):
        # Skip files in excluded directories
        if any(_skip_dir(part) for part in py_file.relative_to(root).parts):
            continue
        # Skip self (docstring / pattern labels contain literal "plt.cm.")
        if py_file.samefile(__file__):
            continue
        try:
            text = py_file.read_text(encoding="utf-8")
        except Exception:
            continue  # skip unreadable files

        lines = text.splitlines()
        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                # Compute 1-based line number
                lineno = text[: match.start()].count("\n") + 1
                line = lines[lineno - 1].strip() if lineno <= len(lines) else ""
                violations.append((py_file, lineno, line, label))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Check for old-style plt.cm usage.")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Not implemented — placeholder for future auto-fix",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1

    if args.fix:
        print("--fix is not yet implemented.", file=sys.stderr)
        return 1

    violations = find_violations(root)

    if not violations:
        print("✅ No plt.cm / matplotlib.cm violations found.")
        return 0

    print(f"❌ Found {len(violations)} plt.cm / matplotlib.cm violation(s):\n")
    for file_path, lineno, line, label in violations:
        rel = file_path.relative_to(root)
        print(f"  {rel}:{lineno}  [{label}]")
        print(f"    {line}")
        print('    → Use plt.colormaps["..."] instead.\n')

    return 1


if __name__ == "__main__":
    sys.exit(main())

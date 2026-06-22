"""Check for placeholder/stub implementations in Python code.

Detects:
  1. Functions whose body consists only of ``pass`` (not ``@abstractmethod``).
  2. Functions whose body consists only of ``...`` (not ``@abstractmethod`` or
     ``@overload``).
  3. Functions or methods containing ``[stub]`` inside a string literal.

This is an AST-based check, so it cannot detect dynamically generated
stubs.  Intentional no-ops (callbacks, abstract methods, overload stubs)
are excluded via decorator inspection.

Usage::

    uv run python scripts/check_placeholders.py    # check all .py files

Exit codes:
    0  — no violations found
    1  — at least one violation found
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories whose Python files are scanned.
_INCLUDE_DIRS = ["src", "pages", "reports", "tests", "scripts"]

# Substrings that exclude a full path from scanning.
_EXCLUDE_PATTERNS = [
    ".opencode/",
    "jupyter/",
    "mathematica/",
    "__pycache__/",
    ".venv/",
    ".egg-info/",
    ".git/",
    "scripts/check_placeholders.py",  # self-exempt — own message strings contain the marker
]


def _is_excluded(path: Path) -> bool:
    """Return True if *path* should be skipped."""
    posix = path.as_posix()
    return any(p in posix for p in _EXCLUDE_PATTERNS)


def _get_py_files() -> list[Path]:
    """Return all ``.py`` files under ``_INCLUDE_DIRS``."""
    files: list[Path] = []
    for d in _INCLUDE_DIRS:
        base = _PROJECT_ROOT / d
        if not base.is_dir():
            continue
        files.extend(
            py_path
            for py_path in sorted(base.rglob("*.py"))
            if not _is_excluded(py_path)
        )
    return files


def _has_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """True if *node* has a decorator named *name* (e.g. ``abstractmethod``)."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == name:
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == name:
            return True
    return False


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    """Return *body* with the leading docstring expression removed."""
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        val = body[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return body[1:]
    return body


def _has_stub_marker(body: list[ast.stmt]) -> bool:
    """True if any string literal in *body* contains a ``[stub]`` marker."""
    # Note: Python 3.12+ uses ast.Constant for all string literals.
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "[stub]" in node.value  # type: ignore[operator]
        ):
            return True
    return False


def _check_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    filename: str,
) -> list[str]:
    """Check *node* for placeholder patterns. Returns list of violation descriptions."""
    violations: list[str] = []

    # Skip abstract methods and overload stubs.
    if _has_decorator(node, "abstractmethod"):
        return violations
    if _has_decorator(node, "overload"):
        return violations

    body = _strip_docstring(node.body)

    # Check 1 & 2: body is only `pass` or `...`
    if len(body) == 1:
        only_stmt = body[0]
        if isinstance(only_stmt, ast.Pass):
            violations.append(
                f"{filename}:{node.lineno}: function `{node.name}` body is "
                "only `pass` (placeholder — implement or remove)"
            )
        elif (
            isinstance(only_stmt, ast.Expr)
            and isinstance(only_stmt.value, ast.Constant)
            and only_stmt.value.value is ...
        ):
            violations.append(
                f"{filename}:{node.lineno}: function `{node.name}` body is "
                "only `...` (placeholder — implement or remove)"
            )

    # Check 3: explicit [stub] marker.
    if _has_stub_marker(node.body):
        violations.append(
            f"{filename}:{node.lineno}: function `{node.name}` contains "
            "`[stub]` marker (placeholder — implement or remove)"
        )

    return violations


def _check_class(node: ast.ClassDef, filename: str) -> list[str]:
    """Check *node* for placeholder patterns. Returns list of violation descriptions."""
    violations: list[str] = []

    body = _strip_docstring(node.body)

    # Class body is only `pass` or `...` → placeholder class.
    if len(body) == 1:
        only_stmt = body[0]
        if isinstance(only_stmt, ast.Pass):
            violations.append(
                f"{filename}:{node.lineno}: class `{node.name}` body is "
                "only `pass` (placeholder — implement or remove)"
            )
        elif (
            isinstance(only_stmt, ast.Expr)
            and isinstance(only_stmt.value, ast.Constant)
            and only_stmt.value.value is ...
        ):
            violations.append(
                f"{filename}:{node.lineno}: class `{node.name}` body is "
                "only `...` (placeholder — implement or remove)"
            )

    return violations


def check_file(py_path: Path) -> list[str]:
    """Check a single Python file for placeholder patterns."""
    rel = py_path.relative_to(_PROJECT_ROOT).as_posix()
    source = py_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_path))
    except SyntaxError:
        return []  # skip files with syntax errors

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_check_function(node, rel))
        elif isinstance(node, ast.ClassDef):
            violations.extend(_check_class(node, rel))
    return violations


def main() -> int:
    py_files = _get_py_files()
    all_violations: list[str] = []
    for py_path in py_files:
        all_violations.extend(check_file(py_path))

    if all_violations:
        print(
            f"Found {len(all_violations)} placeholder violation(s):\n",
            file=sys.stderr,
        )
        for v in all_violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

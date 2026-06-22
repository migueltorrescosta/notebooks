"""Lint-style tests for source-code architecture constraints.

All tests scan ``src/``, ``pages/``, and ``tests/`` for violations of
project-wide rules that were previously checked only during manual audits.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_PAGES_DIR = _PROJECT_ROOT / "pages"
_TESTS_DIR = _PROJECT_ROOT / "tests"


def _py_files_under(directory: Path, *, exclude_tests: bool = False) -> list[Path]:
    """Recursively list ``.py`` files under *directory*.

    When *exclude_tests* is ``True``, files matching ``test_*.py`` are skipped.
    """
    files: list[Path] = []
    for f in sorted(directory.rglob("*.py")):
        if exclude_tests and f.name.startswith("test_"):
            continue
        if f.name == "__init__.py":
            continue
        files.append(f)
    return files


# ── Module-level constants ─────────────────────────────────────────────────


class TestNoModuleLevelConstantsInSrc:
    """Global Constraint 6: No module-level constants in ``src/``.

    Shared modules must not define module-level ``UPPER_CASE`` constants
    for default parameters, bounds, or reference values.  Use function-level
    defaults or ``@dataclass`` config objects instead.
    """

    # Whitelist for modules that intentionally define constants
    _WHITELIST: frozenset[str] = frozenset(
        {
            "constants.py",  # This IS the constants module
            "optimization.py",  # TEST_FUNCTIONS / MINIMIZERS are benchmark definitions
        }
    )

    @pytest.mark.parametrize(
        "py_path",
        _py_files_under(_SRC_DIR, exclude_tests=True),
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_no_module_level_constants(self, py_path: Path) -> None:
        """No ``UPPER_CASE = ...`` assignments at module level."""
        if py_path.name in self._WHITELIST:
            pytest.skip(f"{py_path.name} is whitelisted (intentional constants module)")

        source = py_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_path))
        except SyntaxError:
            pytest.fail(f"Syntax error in {py_path.relative_to(_PROJECT_ROOT)}")

        constants = _find_module_level_constants(tree)
        if constants:
            rel = py_path.relative_to(_PROJECT_ROOT)
            details = "; ".join(f"{name} (L{lineno})" for name, lineno in constants)
            pytest.fail(
                f"{rel}: module-level constant(s) found: {details}."
                " Use function-level defaults or @dataclass config objects instead."
            )


def _find_module_level_constants(tree: ast.Module) -> list[tuple[str, int]]:
    """Return ``(name, lineno)`` for every ``UPPER_CASE`` module-level assignment."""
    constants: list[tuple[str, int]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets: list[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]
            constants.extend(
                (t.id, node.lineno)
                for t in targets
                if isinstance(t, ast.Name) and t.id.isupper()
            )
    return constants


# ── np.linalg.inv ──────────────────────────────────────────────────────────


class TestNoLinalgInvInSrc:
    """Numerical Guidelines (Stability): Never invert matrices directly.

    ``np.linalg.inv`` is banned; use ``np.linalg.solve`` for linear systems
    or ``scipy.linalg.expm`` / ``scipy.linalg.eig`` for matrix functions.
    """

    @pytest.mark.parametrize(
        "py_path",
        _py_files_under(_SRC_DIR),
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_no_linalg_inv(self, py_path: Path) -> None:
        """No ``np.linalg.inv`` or ``numpy.linalg.inv`` usage."""
        source = py_path.read_text(encoding="utf-8")
        # Check only actual code calls, not comments/docstrings
        patterns = [
            r"np\.linalg\.inv\s*\(",
            r"numpy\.linalg\.inv\s*\(",
            r"from numpy\.linalg import.*\binv\b",
        ]
        violations: list[tuple[int, str]] = []
        for line_idx, line in enumerate(source.splitlines(), start=1):
            # Strip comments
            code_part = line.split("#")[0]
            for pat in patterns:
                if re.search(pat, code_part):
                    violations.append((line_idx, line.strip()))
                    break

        if violations:
            rel = py_path.relative_to(_PROJECT_ROOT)
            msg = f"{rel}: np.linalg.inv or alias found:\n"
            for lineno, text in violations:
                msg += f"  L{lineno}: {text}\n"
            msg += "Use np.linalg.solve or scipy.linalg.expm instead."
            pytest.fail(msg)


# ── Imports from local.py ──────────────────────────────────────────────────


def _find_imports_from_local_py(source: str) -> list[tuple[int, str]]:
    """Return ``(line_number, line_text)`` for each import from ``local.py``.

    Only matches actual Python import statements, not comments or docstrings.
    """
    patterns = [
        r"^\s*from\s+reports\..*local\s+import\s+",
        r"^\s*import\s+reports\..*local",
    ]
    violations: list[tuple[int, str]] = []
    for line_idx, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", '"""')):
            continue
        if stripped.startswith(("r", "f")):
            continue
        for pat in patterns:
            if re.search(pat, line):
                violations.append((line_idx, line.strip()))
                break
    return violations


class TestNoImportsFromLocalPy:
    """Global Constraint 7: No imports from ``local.py`` outside reports.

    Code in ``src/``, ``pages/``, and ``tests/`` must never import from a
    report's ``local.py``.  If the function is needed externally, promote it
    to ``src/`` first.
    """

    @pytest.mark.parametrize(
        "py_path",
        _py_files_under(_SRC_DIR)
        + _py_files_under(_PAGES_DIR)
        + _py_files_under(_TESTS_DIR),
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_no_imports_from_local_py(self, py_path: Path) -> None:
        """No ``from reports.*local import`` or ``import reports.*local``."""
        source = py_path.read_text(encoding="utf-8")
        violations = _find_imports_from_local_py(source)
        if violations:
            rel = py_path.relative_to(_PROJECT_ROOT)
            msg = f"{rel}: import from report's local.py found:\n"
            for lineno, text in violations:
                msg += f"  L{lineno}: {text}\n"
            msg += "Promote the needed code to src/ first."
            pytest.fail(msg)


def _all_project_py_files() -> list[Path]:
    """Return sorted list of all ``.py`` files in the project (excluding ``__init__.py``).

    Covers ``src/``, ``pages/``, ``tests/``, root-level files (``Home.py``,
    ``conftest.py``), and all ``reports/*/`` files.
    """
    files: list[Path] = []
    for directory in (_SRC_DIR, _PAGES_DIR, _TESTS_DIR):
        files.extend(_py_files_under(directory))
    # Root-level .py files
    files.extend(f for f in _PROJECT_ROOT.glob("*.py") if f.name != "__init__.py")
    # Report directories (local.py, test_local.py, runner scripts, etc.)
    for report_dir in sorted(_PROJECT_ROOT.joinpath("reports").iterdir()):
        if (
            report_dir.is_dir()
            and report_dir.name.isdigit()
            and len(report_dir.name) == 8
        ):
            files.extend(
                f for f in sorted(report_dir.glob("*.py")) if f.name != "__init__.py"
            )
    return sorted(set(files))


def _find_alias_imports_from_src(tree: ast.Module) -> list[tuple[str, str, str, int]]:
    """Return ``(module, name, alias, lineno)`` for every alias import from ``src.``.

    Identity re-exports (``from src.foo import bar as bar``) are not flagged.
    """
    violations: list[tuple[str, str, str, int]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("src.")
        ):
            violations.extend(
                (node.module, alias.name, alias.asname, node.lineno)
                for alias in node.names
                if alias.asname and alias.asname != alias.name
            )
    return violations


# ── Alias imports from src/ ────────────────────────────────────────────────


class TestNoAliasImportsFromSrc:
    """Global Constraint: No alias imports of in-repo-defined functions.

    All imports of functions/classes/variables defined in ``src/`` modules
    must use the canonical name, not an alias.

    Correct: ``from src.foo import bar``
    Wrong:   ``from src.foo import bar as baz``

    Identity re-exports in ``__init__.py`` (``from src.foo import bar as bar``)
    are exempt and the helper already excludes ``__init__.py`` from the scan.
    """

    @pytest.mark.parametrize(
        "py_path",
        _all_project_py_files(),
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_no_alias_imports_from_src(self, py_path: Path) -> None:
        """No ``from src.xxx import yyy as zzz`` with ``zzz != yyy``."""
        source = py_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_path))
        except SyntaxError:
            pytest.fail(f"Syntax error in {py_path.relative_to(_PROJECT_ROOT)}")

        violations = _find_alias_imports_from_src(tree)
        if violations:
            rel = py_path.relative_to(_PROJECT_ROOT)
            details = "; ".join(
                f"L{lineno}: from {module} import {name} as {asname}"
                for module, name, asname, lineno in violations
            )
            pytest.fail(
                f"{rel}: alias import(s) of src functions: {details}. "
                "Use the canonical name instead of the alias."
            )


# ── Radon cyclomatic complexity ─────────────────────────────────────────────


class TestRadonCyclomaticComplexity:
    """Cyclomatic complexity must stay ≤ B (grade A or B) in every block.

    Blocks (functions, methods, classes) with grade C, D, E, or F
    (cyclomatic complexity ≥ 11) are flagged.  The test is skipped by default
    because 43 files currently violate the threshold.  Unskip only after every
    violating block has been refactored below the threshold.
    """

    @pytest.mark.parametrize(
        "py_path",
        _all_project_py_files(),
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_cyclomatic_complexity_max_B(self, py_path: Path) -> None:
        """No block (function/method/class) shall have grade C or worse."""
        from radon.complexity import cc_rank, cc_visit

        source = py_path.read_text(encoding="utf-8")
        try:
            blocks = cc_visit(source)
        except SyntaxError:
            pytest.fail(f"Syntax error in {py_path.relative_to(_PROJECT_ROOT)}")

        violations: list[tuple[str, str, int, str, int]] = []
        _GRADE_THRESHOLD = "C"  # fail on C, D, E, F

        for block in blocks:
            grade = cc_rank(block.complexity)
            if grade >= _GRADE_THRESHOLD:
                violations.append(
                    (
                        type(block).__name__,
                        block.name,
                        block.complexity,
                        grade,
                        block.lineno,
                    )
                )

        if violations:
            rel = py_path.relative_to(_PROJECT_ROOT)
            lines = "\n".join(
                f"  L{lineno}: {btype} {name} (cc={cc}, grade={grade})"
                for btype, name, cc, grade, lineno in violations
            )
            pytest.fail(
                f"{rel}: {len(violations)} block(s) with grade ≥ C:\n"
                f"{lines}\n"
                "Refactor to reduce cyclomatic complexity below 11 (grade B)."
            )


# ── Function length ─────────────────────────────────────────────────────────


class TestFunctionLength:
    """Functions must not exceed length limits.

    Hard limit: 200 LOC (all lines counted — docstrings, blanks, comments).
    Soft limit: 175 LOC (advisory, still fails the test).

    Counts physical lines (``end_lineno - lineno + 1``) via AST, so blank
    readability spacing and multi-line docstrings are included.  The limits
    are deliberately generous to accommodate physics-verbose code (inline
    assertions, operator construction, multi-line docstrings).
    """

    _SOFT_LIMIT = 175
    _HARD_LIMIT = 200

    @pytest.mark.parametrize(
        "py_path",
        _all_project_py_files(),
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_function_length_within_limits(self, py_path: Path) -> None:
        """No function shall exceed ``_HARD_LIMIT`` lines."""
        source = py_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_path))
        except SyntaxError:
            pytest.fail(f"Syntax error in {py_path.relative_to(_PROJECT_ROOT)}")

        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.end_lineno is None:
                    continue  # malformed node (should not happen in valid code)
                n_lines = node.end_lineno - node.lineno + 1
                if n_lines > self._HARD_LIMIT:
                    violations.append(
                        f"L{node.lineno}: HARD — {node.name} ({n_lines} lines, "
                        f"exceeds hard limit of {self._HARD_LIMIT})"
                    )
                elif n_lines > self._SOFT_LIMIT:
                    violations.append(
                        f"L{node.lineno}: SOFT — {node.name} ({n_lines} lines, "
                        f"exceeds soft limit of {self._SOFT_LIMIT})"
                    )

        if violations:
            rel = py_path.relative_to(_PROJECT_ROOT)
            detail = "\n".join(f"  {v}" for v in violations)
            pytest.fail(
                f"{rel}: {len(violations)} function(s) exceed length limits:\n"
                f"{detail}\n"
                "Refactor by extracting helper functions or shortening docstrings."
            )


# ── File length ────────────────────────────────────────────────────────────


class TestFileLength:
    """Files must not exceed length limits.

    Hard limit: 1500 LOC — any file exceeding this (and not in the whitelist)
    is a test failure.  Soft limit: 500 LOC — advisory warning only (printed to
    stderr, test passes).

    Known hard violators are whitelisted in ``_HARD_OVERRIDES`` and tracked in
    the CHANGELOG backlog under "File Length Reduction Campaign".  Any file NOT
    in the override set that exceeds ``_HARD_LIMIT`` is a regression that must
    be fixed or explicitly added to the backlog.
    """

    _HARD_LIMIT = 1500
    _SOFT_LIMIT = 500

    # Known hard violations (≥_HARD_LIMIT) — tracked in CHANGELOG backlog.
    # New files exceeding the hard limit must be added here with a backlog entry.
    _HARD_OVERRIDES: frozenset[str] = frozenset(
        {
            "src/analysis/scaling_survey.py",
            "src/analysis/ancilla_optimization.py",
            "src/analysis/ancilla_drive_metrology.py",
            "reports/20260518/local.py",
            "reports/20260524/local.py",
            "reports/20260616/local.py",
            "reports/20260528/local.py",
            "reports/20260525/local.py",
            "reports/20260619/local.py",
            "reports/20260610/local.py",
            "reports/20260523/local.py",
            "reports/20260620/local.py",
            "reports/20260615/local.py",
            "reports/20260519/local.py",
            "reports/20260621/local.py",
        }
    )

    @pytest.mark.parametrize(
        "py_path",
        _all_project_py_files(),
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_file_length_within_limits(self, py_path: Path) -> None:
        """No file shall exceed ``_HARD_LIMIT`` lines (unless whitelisted)."""
        source = py_path.read_text(encoding="utf-8")
        n_lines = len(source.splitlines())
        rel = py_path.relative_to(_PROJECT_ROOT).as_posix()

        in_override = rel in self._HARD_OVERRIDES

        # Hard violation → fail unless whitelisted
        if n_lines > self._HARD_LIMIT:
            if in_override:
                # Acknowledged — let test pass (backlog entry exists)
                pass
            else:
                pytest.fail(
                    f"{rel}: {n_lines} lines exceeds hard limit "
                    f"({self._HARD_LIMIT}). "
                    "Refactor or add to _HARD_OVERRIDES with a backlog entry."
                )

        # Soft violation → warning only (always reported, even for overrides)
        if n_lines > self._SOFT_LIMIT:
            msg = (
                f"SOFT: {rel}: {n_lines} lines exceeds soft limit ({self._SOFT_LIMIT})."
            )
            if in_override:
                msg += " (whitelisted hard violation)"
            warnings.warn(msg, stacklevel=2)

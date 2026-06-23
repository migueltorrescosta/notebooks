"""Lint-style tests for report markdown files.

All tests scan ``reports/`` and ``CHANGELOG.md`` for formatting violations
and content issues that the skill workflow previously checked via manual
``rg`` / ``grep`` commands.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parents[1]
_REPORTS_DIR = _PROJECT_ROOT / "reports"
_CHANGELOG = _PROJECT_ROOT / "CHANGELOG.md"

# All markdown files under reports/
_REPORT_MD_FILES: list[Path] = sorted(_REPORTS_DIR.rglob("*.md"))

# YYYYMMDD-prefixed report directories only (excludes reports/findings/ etc.)
_REPORT_DIRS: list[Path] = sorted(
    d
    for d in _REPORTS_DIR.iterdir()
    if d.is_dir() and d.name.isdigit() and len(d.name) == 8
)

# Section emojis in prescribed order with mandatory flag
_SECTION_ORDER: list[tuple[str, bool]] = [
    ("\U0001f9ea", True),  # 🧪 Hypothesis
    ("\u269b\ufe0f", True),  # ⚛️ Theoretical Model
    ("\U0001f4ca", False),  # 📊 Models Survey (optional)
    ("\U0001f4bb", True),  # 💻 Numerical Simulation
    ("\u26a0\ufe0f", True),  # ⚠️ Expected Failure Conditions
    ("\U0001f52c", True),  # 🔬 Results
    ("\u2705", True),  # ✅ Success Criteria
    ("\u2696\ufe0f", False),  # ⚖️ Physical Invariants / Analytical Bounds (optional)
    ("\U0001f3c1", True),  # 🏁 Conclusions
]
_MANDATORY = {emoji for emoji, _mandatory in _SECTION_ORDER if _mandatory}

# Reports with known section-order violations (all resolved)
_KNOWN_SECTION_ORDER_ISSUES: set[str] = set()

# ── Helpers ──────────────────────────────────────────────────────────────────


def _report_md_files() -> list[Path]:
    """Return all report markdown files in YYYYMMDD directories (excludes findings/)."""
    return sorted(
        p
        for p in _REPORT_MD_FILES
        if p.parent.name.isdigit() and len(p.parent.name) == 8
    )


def _find_bare_pipes_in_table_math(line: str) -> list[int]:
    """Return column positions of bare ``|`` inside ``$...$`` in a table row.

    A bare ``|`` is one not preceded by ``\\`` (i.e., ``|`` vs ``\\|``).
    Only checks lines that start with ``|`` (table rows).
    """
    if not line.startswith("|"):
        return []

    violations: list[int] = []
    in_math = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "$":
            in_math = not in_math
        elif ch == "|" and in_math and (i == 0 or line[i - 1] != "\\"):
            violations.append(i)
        i += 1
    return violations


def _extract_section_emojis(text: str) -> list[str]:
    """Return section-header emojis found in *text* in order of appearance.

    Matches ``## <emoji> <title>`` lines only.
    """
    section_emojis: list[str] = []
    for line in text.splitlines():
        for emoji, _ in _SECTION_ORDER:
            if line.startswith(f"## {emoji} "):
                section_emojis.append(emoji)
                break
    return section_emojis


def _check_section_order_violations(
    section_emojis: list[str],
) -> list[str]:
    """Return list of violation descriptions, or empty if all OK.

    Checks ordering (emojis appear in prescribed order) and that all
    mandatory sections are present.
    """
    violations: list[str] = []
    if not section_emojis:
        violations.append("No section headers with emojis found")
        return violations

    order_map: dict[str, int] = {}
    for idx, (emoji, _) in enumerate(_SECTION_ORDER):
        order_map[emoji] = idx

    prev_idx = -1
    for emoji in section_emojis:
        curr_idx = order_map[emoji]
        if curr_idx <= prev_idx:
            violations.append(f"  {emoji} appears after position {prev_idx}")
        prev_idx = curr_idx

    found_set = set(section_emojis)
    missing = _MANDATORY - found_set
    if missing:
        sorted_missing = sorted(missing, key=lambda e: order_map[e])
        violations.append(f"  Missing mandatory sections: {''.join(sorted_missing)}")

    return violations


def _parse_figure_references(md_path: Path) -> list[tuple[str, Path]]:
    """Return list of (alt_text, expected_figure_abspath) from a report.

    Handles both ``figures/name.svg`` (relative to report dir) and
    ``reports/YYYYMMDD/figures/name.svg`` (relative to project root).
    """
    refs: list[tuple[str, Path]] = []
    text = md_path.read_text(encoding="utf-8")
    for match in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", text):
        alt, target = match.group(1), match.group(2)
        if target.startswith(("reports/", "/reports/")):
            # Repo-root-relative path
            figure_path = (_PROJECT_ROOT / target.lstrip("/")).resolve()
        else:
            # Relative to the report directory
            figure_path = (md_path.parent / target).resolve()
        refs.append((alt, figure_path))
    return refs


# ── Content checks (all report .md files + CHANGELOG) ───────────────────────


class TestReportContent:
    """Structural and formatting checks on report markdown content."""

    # ── Bare pipe in LaTeX math (table rows only) ────────────────────────

    def test_no_bare_pipe_in_latex(self) -> None:
        """No bare ``|`` inside ``$...$`` in any table row (all reports + CHANGELOG).

        Single test to keep output quiet on success.  On failure, lists every
        violation across all files.
        """
        md_files: list[Path] = [*list(_REPORT_MD_FILES), _CHANGELOG]
        all_failures: list[str] = []

        for md_path in md_files:
            violations: list[tuple[int, int]] = []
            for line_idx, line in enumerate(
                md_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                cols = _find_bare_pipes_in_table_math(line)
                violations.extend((line_idx, col) for col in cols)

            if violations:
                rel = md_path.relative_to(_PROJECT_ROOT)
                msg = f"{rel}:\n"
                for line_idx, col in violations:
                    msg += f"  Line {line_idx}, col {col}: bare | inside $...$ \u2014 use \\vert\n"
                all_failures.append(msg)

        if all_failures:
            pytest.fail("\n".join(all_failures))

    # ── Display math ─────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "md_path",
        [*list(_REPORT_MD_FILES), _CHANGELOG],
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_report_no_display_math(self, md_path: Path) -> None:
        """No ``$$`` display math in reports \u2014 use ``$...$`` inline only."""
        text = md_path.read_text(encoding="utf-8")
        matches = [(i + 1) for i, line in enumerate(text.splitlines()) if "$$" in line]
        if matches:
            rel = md_path.relative_to(_PROJECT_ROOT)
            pytest.fail(
                f"{rel}: display math ($$) found at lines {matches} \u2014"
                " use $...$ inline math instead"
            )

    # ── \ket ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "md_path",
        [*list(_REPORT_MD_FILES), _CHANGELOG],
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_report_no_ket_usage(self, md_path: Path) -> None:
        r"""No ``\ket`` in reports \u2014 not recognised by the Markdown engine."""
        text = md_path.read_text(encoding="utf-8")
        matches = [
            (i + 1, line)
            for i, line in enumerate(text.splitlines())
            # Only flag actual \ket, not \\ket (escaped backslash)
            if re.search(r"(?<!\\)\\ket", line)
        ]
        if matches:
            rel = md_path.relative_to(_PROJECT_ROOT)
            lines_str = ", ".join(f"L{line_no}" for line_no, _ in matches[:10])
            pytest.fail(
                f"{rel}: '\\ket' found at {lines_str} \u2014"
                " use |...\u27e9 or \\vert...\\rangle instead"
            )


# ── Structure checks (YYYYMMDD report dirs only) ────────────────────────────


class TestReportStructure:
    """Section ordering, emoji placement, and figure references."""

    # ── Section order ────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "md_path",
        _report_md_files(),
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_report_section_order_conforms(self, md_path: Path) -> None:
        """Section headers appear in the prescribed order.

        Mandatory sections: \U0001f9ea \u269b\ufe0f \U0001f4bb \u26a0\ufe0f
                           \U0001f52c \u2705 \U0001f3c1
        Optional sections : \U0001f4ca (after \u269b\ufe0f),
                            \u2696\ufe0f (after \u2705)
        """
        report_date = md_path.parent.name
        text = md_path.read_text(encoding="utf-8")
        section_emojis = _extract_section_emojis(text)
        violations = _check_section_order_violations(section_emojis)
        if violations:
            if report_date in _KNOWN_SECTION_ORDER_ISSUES:
                pytest.skip(f"Known issue: section-order violations in {report_date}")
            rel = md_path.relative_to(_PROJECT_ROOT)
            pytest.fail(f"{rel} section order violations:\n" + "\n".join(violations))

    # ── Emoji placement ──────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "md_path",
        _report_md_files(),
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_report_emoji_placement(self, md_path: Path) -> None:
        """Section-title emoji goes after ``## `` and before the title text.

        Correct: ``## \U0001f9ea Hypothesis``
        Wrong  : ``## Hypothesis \U0001f9ea``
        """
        all_emojis = {e for e, _ in _SECTION_ORDER}
        text = md_path.read_text(encoding="utf-8")
        issues: list[str] = []
        for line_idx, line in enumerate(text.splitlines(), start=1):
            if not line.startswith("## "):
                continue
            rest = line[3:]
            for emoji in all_emojis:
                if emoji not in rest:
                    continue
                # Correct: "## 🧪 Hypothesis"
                if rest.startswith(emoji):
                    break
                # Wrong: emoji is not at the start of the header text
                issues.append(f"  L{line_idx}: emoji not at start of header: '{line}'")
                break

        if issues:
            rel = md_path.relative_to(_PROJECT_ROOT)
            pytest.fail(f"{rel} emoji placement issues:\n" + "\n".join(issues))

    # ── Figure references ────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "md_path",
        _report_md_files(),
        ids=lambda p: p.relative_to(_PROJECT_ROOT).as_posix(),
    )
    def test_report_figure_references_resolve(self, md_path: Path) -> None:
        """All ``![...](...)`` figure references point to existing files."""
        refs = _parse_figure_references(md_path)
        broken: list[tuple[str, Path]] = []
        for alt, fig_path in refs:
            if not fig_path.exists():
                broken.append((alt, fig_path))

        if broken:
            rel = md_path.relative_to(_PROJECT_ROOT)
            msg = f"{rel}: {len(broken)} figure(s) not found:\n"
            for alt, fig_path in broken:
                try:
                    rel_path = fig_path.relative_to(_PROJECT_ROOT)
                except ValueError:
                    rel_path = fig_path
                msg += f"  [{alt}]({rel_path})\n"
            pytest.fail(msg)


# ── Duplication checks (reports/*/<slug>.py) ───────────────────────────────


class TestReportDuplication:
    """Detect functions and constants duplicated across >=2 reports.

    These tests flag code that should be promoted to ``src/`` shared modules.
    The test currently fails due to pre-existing duplications; run the
    ``audit-code`` or ``align-project`` skill to promote the shared code.
    """

    FUNCTION_BLACKLIST = frozenset(
        {
            # Standard entry points / patterns that legitimately appear everywhere
            "_fig_path",
            "_parquet_path",
            "fig_path",
            "parquet_path",
            "main",
        }
    )

    CONSTANT_BLACKLIST = frozenset(
        {
            "RTOL",
            "ATOL",
            "SEED",
            "REPORT_DATE",
            "N_POINTS",
            "REPORTS_DIR",
            "DATE_TAG",
            "_REPORT_DIR",
        }
    )

    @staticmethod
    def _report_code_files() -> list[Path]:
        """Return sorted list of experiment code files (was ``local.py``)."""
        return sorted(
            p
            for p in _REPORTS_DIR.rglob("*.py")
            if not p.name.startswith("test_") and not p.name.startswith("_")
        )

    # ── Function duplication ─────────────────────────────────────────────

    def test_functions_duplicated_across_reports(self) -> None:
        """No function defined in >=4 reports without being promoted to ``src/``.

        Replaces the manual ``grep -rn "^def " reports/*/<slug>.py`` check.
        """
        func_to_reports: dict[str, set[str]] = {}
        for py_file in self._report_code_files():
            report_name = py_file.parent.name
            text = py_file.read_text(encoding="utf-8")
            for match in re.finditer(r"^def (\w+)", text, re.MULTILINE):
                name = match.group(1)
                if name not in self.FUNCTION_BLACKLIST:
                    func_to_reports.setdefault(name, set()).add(report_name)

        duplicates = {
            name: sorted(reports)
            for name, reports in func_to_reports.items()
            if len(reports) >= 4
        }

        if duplicates:
            msg = f"{len(duplicates)} function(s) defined in >=4 reports:\n"
            for name in sorted(duplicates):
                reports = duplicates[name]
                msg += f"  {name}: {', '.join(reports)}\n"
            msg += "Consider promoting to src/ via audit-code / align-project skills."
            pytest.fail(msg)

    # ── Constant duplication ─────────────────────────────────────────────

    def test_constants_duplicated_across_reports(self) -> None:
        """No module-level constant defined in >=2 reports without promotion.

        Replaces the manual ``grep -rn "^[A-Z_][A-Z_0-9]* *="`` check.
        """
        const_to_reports: dict[str, list[str]] = {}
        for py_file in self._report_code_files():
            report_name = py_file.parent.name
            text = py_file.read_text(encoding="utf-8")
            for match in re.finditer(r"^([A-Z_][A-Z_0-9]*) *=", text, re.MULTILINE):
                name = match.group(1)
                if name not in self.CONSTANT_BLACKLIST:
                    const_to_reports.setdefault(name, []).append(report_name)

        duplicates = {
            name: reports
            for name, reports in const_to_reports.items()
            if len(reports) >= 2
        }

        if duplicates:
            msg = f"{len(duplicates)} constant(s) defined in >=2 reports:\n"
            for name in sorted(duplicates):
                reports = duplicates[name]
                msg += f"  {name}: {', '.join(sorted(set(reports)))}\n"
            msg += "Consider promoting to src/ via audit-code / align-project skills."
            pytest.fail(msg)

"""Root conftest — slow-test enforcement.

``src`` is importable via the editable install (``uv sync`` / ``uv pip install -e .``).
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Slow-test enforcement: warn on any test exceeding 5 s without
# @pytest.mark.slow.  Collect call-phase durations during the run and
# report them in pytest_terminal_summary.
# ---------------------------------------------------------------------------
_nonslow_durations: dict[str, float] = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(  # type: ignore[no-untyped-def]  # hookwrapper-style generator
    item: pytest.Item,
    call: pytest.CallInfo,
):
    """Record call-phase duration for tests not marked ``slow``."""
    yield
    if call.when == "call":
        has_slow = item.get_closest_marker("slow") is not None
        if not has_slow:
            _nonslow_durations[item.nodeid] = call.duration


def pytest_terminal_summary(
    terminalreporter: Any,  # pytest.terminal.TerminalReporter; typed as Any to avoid pyright false-positive
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Warn on any test exceeding 5 s that lacks ``@pytest.mark.slow``."""
    slow_unmarked: list[tuple[str, float]] = [
        (nodeid, dur) for nodeid, dur in _nonslow_durations.items() if dur > 5.0
    ]
    if not slow_unmarked:
        return

    slow_unmarked.sort(key=lambda x: -x[1])
    terminalreporter.write_sep("-", "Slow test warnings", bold=True)
    terminalreporter.write_line(
        "The following tests took >5 s without @pytest.mark.slow:"
    )
    for nodeid, duration in slow_unmarked:
        terminalreporter.write_line(f"  {duration:.2f}s  {nodeid}")
    terminalreporter.write_line(
        f"\n{len(slow_unmarked)} test(s) may need @pytest.mark.slow"
    )

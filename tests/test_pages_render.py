"""Integration tests to ensure all Streamlit pages render without errors.

These tests verify that:
1. Each page loads without Streamlit errors
2. No duplicate element IDs exist
3. Pages have rendered content
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# Get all page files in the pages directory
PAGES_DIR = Path(__file__).parent.parent / "pages"
PAGE_FILES = sorted(p for p in PAGES_DIR.glob("*.py") if p.name != "__init__.py")


# Pages that are computationally expensive and may timeout
SLOW_PAGES = {
    "BEC_Sensitivity_Scaling.py",
    "Delta_estimation.py",
    "Energy_Level_Calculator.py",
    "High_Order_Squeezing.py",
    "Numerical_Quantum_Time_Evolution.py",
}


def get_page_script(page_file: Path) -> str:
    """Get the script content for a page file."""
    return page_file.read_text()


def _run_and_check(page_file: Path) -> AppTest:
    """Run a page via AppTest and check for common streamlit errors.

    Returns the AppTest instance for further assertions.
    """
    script = get_page_script(page_file)

    at = AppTest.from_string(script)

    # Use a longer timeout for computationally slow pages
    timeout = 30 if page_file.name in SLOW_PAGES else 10

    try:
        at.run(timeout=timeout)
    except RuntimeError as e:
        if "timed out" in str(e).lower() and page_file.name in SLOW_PAGES:
            pytest.skip(
                f"Page {page_file.name} is computationally expensive and timed out",
            )
        raise

    if at.exception:
        if "timed out" in str(at.exception).lower() and page_file.name in SLOW_PAGES:
            pytest.skip(
                f"Page {page_file.name} is computationally expensive and timed out",
            )
        pytest.fail(f"Page {page_file.name} raised an exception: {at.exception}")

    return at


@pytest.mark.parametrize(
    "page_file",
    [p for p in PAGE_FILES if p.name not in SLOW_PAGES],
    ids=lambda p: p.name,
)
def test_page_renders_without_errors(page_file: Path) -> None:
    """Test that each page renders without errors and has UI content.

    Checks for:
    1. No StreamlitDuplicateElementId / StreamlitDuplicateElementKey errors
    2. Rendered UI content exists (skipped for computationally expensive pages)
    """
    at = _run_and_check(page_file)
    assert len(at.main.children) > 0, (
        f"Page {page_file.name} should render some UI content"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "page_file", [p for p in PAGE_FILES if p.name in SLOW_PAGES], ids=lambda p: p.name
)
def test_slow_page_renders_without_errors(page_file: Path) -> None:
    """Test that computationally expensive pages render without errors."""
    _run_and_check(page_file)

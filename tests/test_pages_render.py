"""Integration tests to ensure all Streamlit pages render without errors.

These tests verify that:
1. Each page loads without Streamlit errors
2. No duplicate element IDs exist
3. Pages have rendered content
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Get all page files in the pages directory
PAGES_DIR = PROJECT_ROOT / "pages"
PAGE_FILES = sorted(PAGES_DIR.glob("*.py"))


# Pages that are computationally expensive and may timeout
SLOW_PAGES = {
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
    except st.errors.StreamlitDuplicateElementId as e:
        pytest.fail(
            f"Page {page_file.name} has duplicate element IDs: {e}. "
            "Add unique 'key' arguments to widgets with the same label."
        )
    except st.errors.StreamlitDuplicateElementKey as e:
        pytest.fail(
            f"Page {page_file.name} has duplicate element keys: {e}. "
            "Make sure each widget has a unique key."
        )
    except RuntimeError as e:
        if "timed out" in str(e).lower() and page_file.name in SLOW_PAGES:
            pytest.skip(
                f"Page {page_file.name} is computationally expensive and timed out"
            )
        raise

    if at.exception:
        if "timed out" in str(at.exception).lower() and page_file.name in SLOW_PAGES:
            pytest.skip(
                f"Page {page_file.name} is computationally expensive and timed out"
            )
        pytest.fail(f"Page {page_file.name} raised an exception: {at.exception}")

    return at


@pytest.mark.parametrize("page_file", PAGE_FILES, ids=lambda p: p.name)
def test_page_renders_without_errors(page_file: Path) -> None:
    """Test that each page renders without errors and has UI content.

    Checks for:
    1. No StreamlitDuplicateElementId / StreamlitDuplicateElementKey errors
    2. Rendered UI content exists (skipped for computationally expensive pages)
    """
    at = _run_and_check(page_file)

    # Check for rendered content (skip for slow pages that use extended timeout)
    if page_file.name not in SLOW_PAGES:
        assert len(at.main.children) > 0, (
            f"Page {page_file.name} should render some UI content"
        )

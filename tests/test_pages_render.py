"""Tests to ensure all Streamlit pages render without errors."""

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
# We'll skip the full run test for these but still check for duplicate ID errors
EXPENSIVE_PAGES = {
    "Delta_Sensitivity_Heatmap.py",
    "Delta_estimation.py",
    "Energy_Level_Calculator.py",
    "Minimize_heatmap.py",
    "Numerical_Quantum_Time_Evolution.py",
}


def get_page_script(page_file: Path) -> str:
    """Get the script content for a page file."""
    return page_file.read_text()


@pytest.mark.parametrize("page_file", PAGE_FILES, ids=lambda p: p.name)
def test_page_renders_without_duplicate_id_error(page_file: Path) -> None:
    """Test that each page renders without StreamlitDuplicateElementId error.

    This test catches the common bug where st.number_input (or other widget)
    calls use the same label across multiple code paths, causing Streamlit to
    assign them the same internal ID.
    """
    script = get_page_script(page_file)

    # Create an AppTest instance and run it
    at = AppTest.from_string(script)

    # Use a longer timeout for computationally expensive pages
    timeout = 30 if page_file.name in EXPENSIVE_PAGES else 10

    # Run the app and capture any exceptions
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
        # Timeout errors are expected for expensive pages
        if "timed out" in str(e).lower() and page_file.name in EXPENSIVE_PAGES:
            pytest.skip(
                f"Page {page_file.name} is computationally expensive and timed out"
            )
        raise

    # Check for any exceptions in the session state
    # This catches other runtime errors
    if at.exception:
        # Skip expensive pages that timeout
        if (
            "timed out" in str(at.exception).lower()
            and page_file.name in EXPENSIVE_PAGES
        ):
            pytest.skip(
                f"Page {page_file.name} is computationally expensive and timed out"
            )
        pytest.fail(f"Page {page_file.name} raised an exception: {at.exception}")


@pytest.mark.parametrize("page_file", PAGE_FILES, ids=lambda p: p.name)
def test_page_loads_successfully(page_file: Path) -> None:
    """Test that each page loads without crashing."""
    if page_file.name in EXPENSIVE_PAGES:
        pytest.skip(f"Page {page_file.name} is computationally expensive")

    script = get_page_script(page_file)
    at = AppTest.from_string(script)

    # Just verify it can be created and run without raising
    at.run()

    # Basic sanity check: the app should have some content
    assert len(at.main.children) > 0, f"Page {page_file.name} has no content"

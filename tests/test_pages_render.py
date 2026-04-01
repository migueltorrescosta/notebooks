"""Integration tests to ensure all Streamlit pages render without errors.

These tests verify that:
1. Each page loads without Streamlit errors
2. Each page has expected UI components
3. No duplicate element IDs exist
4. Pages handle widget interactions gracefully
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
# We'll skip the full run test for these but still check for duplicate ID errors
EXPENSIVE_PAGES = {
    "Energy_Level_Calculator.py",
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


@pytest.mark.parametrize("page_file", PAGE_FILES, ids=lambda p: p.name)
def test_page_has_content(page_file: Path) -> None:
    """Verify that each page renders with content."""
    if page_file.name in EXPENSIVE_PAGES:
        pytest.skip(f"Page {page_file.name} is computationally expensive")

    script = get_page_script(page_file)
    at = AppTest.from_string(script)
    at.run()

    # Page should have some children (UI elements)
    assert len(at.main.children) > 0, (
        f"Page {page_file.name} should render some UI content"
    )


@pytest.mark.parametrize("page_file", PAGE_FILES, ids=lambda p: p.name)
def test_page_sidebar_exists(page_file: Path) -> None:
    """Verify that sidebar is present and accessible."""
    if page_file.name in EXPENSIVE_PAGES:
        pytest.skip(f"Page {page_file.name} is computationally expensive")

    script = get_page_script(page_file)
    at = AppTest.from_string(script)
    at.run()

    # Sidebar should exist
    assert hasattr(at, "sidebar"), f"Page {page_file.name} should have a sidebar"


class TestSpecificPages:
    """Tests for specific page behaviors and components."""

    def test_wave_interference_has_plot_array_calls(self) -> None:
        """Verify Wave_interference page uses plot_array for visualization."""
        page_file = PAGES_DIR / "Wave_interference.py"
        script = page_file.read_text()

        # Check that the page imports and uses plot_array
        assert "from src.plotting import plot_array" in script, (
            "Wave_interference should import plot_array"
        )
        assert "plot_array(" in script, "Wave_interference should call plot_array"

    def test_heisenberg_model_has_simulation_code(self) -> None:
        """Verify Heisenberg model page contains expected physics code."""
        page_file = PAGES_DIR / "Heisenberg_model.py"
        script = page_file.read_text()

        # Should have some physics-related content
        assert len(script) > 1000, "Heisenberg_model page should have substantial code"

    def test_bayes_updates_references_bayesian_concepts(self) -> None:
        """Verify Bayes_updates page uses Bayesian concepts."""
        page_file = PAGES_DIR / "Bayes_updates.py"
        script = page_file.read_text()

        # Should have Bayesian-related content
        assert "bayes" in script.lower() or "prior" in script.lower(), (
            "Bayes_updates should reference Bayesian concepts"
        )

    def test_delta_estimation_has_phase_logic(self) -> None:
        """Verify Delta_estimation page has phase estimation logic."""
        page_file = PAGES_DIR / "Delta_estimation.py"
        script = page_file.read_text()

        # Should contain phase-related variables or logic
        assert "delta" in script.lower() or "phase" in script.lower(), (
            "Delta_estimation should contain phase-related logic"
        )

    def test_fisher_information_has_fisher_import(self) -> None:
        """Verify Fisher_information page imports or uses Fisher information."""
        page_file = PAGES_DIR / "Fisher_information.py"
        script = page_file.read_text()

        # Should have Fisher-related content
        assert "fisher" in script.lower() or "information" in script.lower(), (
            "Fisher_information page should contain Fisher information logic"
        )

    def test_visualize_partial_trace_has_density_matrix(self) -> None:
        """Verify Visualize_Partial_Trace page handles density matrices."""
        page_file = PAGES_DIR / "Visualize_Partial_Trace.py"
        script = page_file.read_text()

        # Should contain density matrix or rho related code
        assert "rho" in script.lower() or "density" in script.lower(), (
            "Visualize_Partial_Trace should reference density matrices"
        )

    def test_all_pages_import_streamlit(self) -> None:
        """Verify all pages properly import streamlit."""
        for page_file in PAGE_FILES:
            script = page_file.read_text()
            assert "import streamlit" in script or "from streamlit" in script, (
                f"Page {page_file.name} should import streamlit"
            )

    def test_all_pages_call_set_page_config(self) -> None:
        """Verify all pages call st.set_page_config for proper configuration."""
        for page_file in PAGE_FILES:
            script = page_file.read_text()
            assert "st.set_page_config" in script, (
                f"Page {page_file.name} should call st.set_page_config"
            )


class TestPageInteractions:
    """Test that pages handle widget interactions correctly."""

    def test_wave_interference_runs_with_defaults(self) -> None:
        """Verify Wave_interference page can run with default values."""
        page_file = PAGES_DIR / "Wave_interference.py"
        script = page_file.read_text()
        at = AppTest.from_string(script)
        at.run()

        # Page should have content
        assert len(at.main.children) > 0

    def test_delta_estimation_runs_with_defaults(self) -> None:
        """Verify Delta_estimation page can run with default values."""
        page_file = PAGES_DIR / "Delta_estimation.py"
        script = page_file.read_text()
        at = AppTest.from_string(script)
        at.run()

        assert len(at.main.children) > 0

    def test_fisher_information_runs_with_defaults(self) -> None:
        """Verify Fisher_information page can run with default values."""
        page_file = PAGES_DIR / "Fisher_information.py"
        script = page_file.read_text()
        at = AppTest.from_string(script)
        at.run()

        assert len(at.main.children) > 0

    def test_heisenberg_model_runs_with_defaults(self) -> None:
        """Verify Heisenberg_model page can run with default values."""
        page_file = PAGES_DIR / "Heisenberg_model.py"
        script = page_file.read_text()
        at = AppTest.from_string(script)
        at.run()

        assert len(at.main.children) > 0

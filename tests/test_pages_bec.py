"""Integration tests for BEC sensitivity and ancilla pages.

Tests verify:
1. Each page loads without Streamlit errors
2. Simulations complete within timeout
3. Physics assertions hold
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page files
PAGES_DIR = PROJECT_ROOT / "pages"
BEC_SENSITIVITY_PAGE = PAGES_DIR / "BEC_Sensitivity_Scaling.py"
BEC_ANCILLA_PAGE = PAGES_DIR / "BEC_Ancilla.py"


def get_page_script(page_file: Path) -> str:
    """Get the script content for a page file."""
    return page_file.read_text()


# =============================================================================
# BEC_Sensitivity_Scaling page
# =============================================================================


class TestBECSensitivityScalingPage:
    """Tests for BEC_Sensitivity_Scaling page."""

    def test_page_renders_without_errors(self) -> None:
        """Test page loads without Streamlit errors."""
        script = get_page_script(BEC_SENSITIVITY_PAGE)
        at = AppTest.from_string(script)
        at.run(timeout=30)
        if at.exception:
            pytest.fail(f"Page raised exception: {at.exception}")

    def test_page_has_content(self) -> None:
        """Verify page renders with content."""
        script = get_page_script(BEC_SENSITIVITY_PAGE)
        at = AppTest.from_string(script)
        at.run()
        assert len(at.main.children) > 0, "Page should render content"

    def test_twa_simulation_completes_quickly(self) -> None:
        """Test that TWA simulation with small N completes within timeout."""
        from src.physics.truncated_wigner import run_twa_simulation
        import time

        start = time.perf_counter()

        result = run_twa_simulation(
            N=20,
            state_type="CSS",
            chi=1.0,
            T=0.1,
            N_traj=50,
            rng_seed=42,
        )

        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Simulation took {elapsed:.2f}s, expected < 2s"
        assert "delta_phi" in result
        assert "delta_phi_sql" in result
        assert "delta_phi_hl" in result


# =============================================================================
# BEC_Ancilla page
# =============================================================================


class TestBECAncillaPage:
    """Tests for BEC_Ancilla page."""

    def test_page_renders_without_errors(self) -> None:
        """Test page loads without Streamlit errors."""
        script = get_page_script(BEC_ANCILLA_PAGE)
        at = AppTest.from_string(script)
        at.run(timeout=30)
        if at.exception:
            pytest.fail(f"Page raised exception: {at.exception}")

    def test_page_has_content(self) -> None:
        """Verify page renders with content."""
        script = get_page_script(BEC_ANCILLA_PAGE)
        at = AppTest.from_string(script)
        at.run()
        assert len(at.main.children) > 0

    def test_ancilla_evolution_produces_finite_expectations(self) -> None:
        """Test ancilla evolution produces finite Jz expectation."""
        from src.algorithms.spin_squeezing import coherent_spin_state
        from src.evolution.lindblad_solver import (
            compute_expectation,
            evolve_lindblad,
            ket_to_density,
            LindbladConfig,
        )
        from src.physics.dicke_basis import jz_operator

        N = 10
        state = coherent_spin_state(N)
        chi = 1.0
        T = 0.1

        rho0 = ket_to_density(state)
        config = LindbladConfig(N=N, chi=chi)
        rho = evolve_lindblad(rho0, config, T, 0.01)
        J_z = jz_operator(N)

        Jz_mean = np.real(compute_expectation(rho, J_z))
        assert np.isfinite(Jz_mean), "Jz expectation should be finite"

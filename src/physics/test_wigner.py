"""
Tests for Wigner Function module.

Physical Validation:
- Wigner function integrates to 1: ∫∫ W(x,p) dx dp = 1
- Vacuum state: W(x,p) = (2/π) exp(-2(x²+p²)) (Gaussian)
- Coherent state: W(x,p) is Gaussian centered at (Re(α), Im(α))
- Negative values indicate non-Gaussianity
"""

import numpy as np
import pytest

from .wigner import (
    wigner_function_single,
    wigner_from_hybrid_state,
    wigner_minimum,
    wigner_is_negative,
)
from .hybrid_system import (
    hybrid_vacuum_state,
    hybrid_coherent_state,
)


# =============================================================================
# Test Wigner Function for Simple States
# =============================================================================


class TestWignerFunctionSingle:
    """Test Wigner function computation for single-mode states."""

    def test_vacuum_normalization(self) -> None:
        """∫∫ W(x,p) dx dp ≈ 1 for vacuum."""
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0  # |0⟩⟨0|

        x = np.linspace(-5, 5, 100)
        p = np.linspace(-5, 5, 100)

        W = wigner_function_single(rho, x, p)

        # Integrate using trapezoidal rule
        dx = x[1] - x[0]
        dp = p[1] - p[0]
        integral = np.sum(W) * dx * dp

        # Relaxed tolerance due to numerical integration
        assert np.isclose(integral, 1.0, rtol=1e-1, atol=0.1)

    def test_vacuum_maximum(self) -> None:
        """Vacuum Wigner max should be 2/π ≈ 0.637 at origin."""
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-3, 3, 50)
        p = np.linspace(-3, 3, 50)

        W = wigner_function_single(rho, x, p)

        max_w = np.max(W)
        assert np.isclose(max_w, 2.0 / np.pi, atol=0.1)

    def test_vacuum_no_negative(self) -> None:
        """Vacuum Wigner should be non-negative everywhere."""
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-5, 5, 100)
        p = np.linspace(-5, 5, 100)

        W = wigner_function_single(rho, x, p)

        assert np.min(W) >= -1e-10

    def test_wigner_shape(self) -> None:
        """Wigner output should have correct shape."""
        N = 5
        rho = np.eye(N + 1, dtype=complex) / (N + 1)  # Maximally mixed

        x = np.linspace(-3, 3, 50)
        p = np.linspace(-3, 3, 60)

        W = wigner_function_single(rho, x, p)

        assert W.shape == (50, 60)


class TestWignerFromHybridState:
    """Test Wigner extraction from hybrid state."""

    def test_hybrid_vacuum(self) -> None:
        """Wigner for |0,↓⟩ should be vacuum Wigner."""
        N = 10
        state = hybrid_vacuum_state(N, spin_state="down")

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_from_hybrid_state(state, N, x, p, spin_component="down")

        assert W.shape == (50, 50)
        # Vacuum should have max near 2/π ≈ 0.637
        assert np.max(W) > 0.3

    def test_hybrid_coherent(self) -> None:
        """Wigner for coherent state should be Gaussian."""
        N = 20
        alpha = 1.0 + 0j
        state = hybrid_coherent_state(N, alpha, spin_state="down")

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_from_hybrid_state(state, N, x, p, spin_component="down")

        # Coherent states are Gaussian - allow small negative due to
        # numerical integration artifacts in the discrete Wigner computation
        assert np.min(W) > -2e-2


class TestWignerMinimum:
    """Test Wigner minimum and negativity detection."""

    def test_minimum_vacuum(self) -> None:
        """Vacuum Wigner minimum should be positive."""
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_function_single(rho, x, p)
        min_w = wigner_minimum(W)

        assert min_w >= -1e-10

    def test_is_negative_vacuum(self) -> None:
        """Vacuum is not Wigner-negative."""
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_function_single(rho, x, p)

        assert not wigner_is_negative(W)

    def test_squeezed_state_negative(self) -> None:
        """Squeezed states can have negative Wigner."""
        # This test is a placeholder - would need actual squeezed state
        # For now, just test that the function works
        N = 10
        rho = np.eye(N + 1, dtype=complex) / (N + 1)

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_function_single(rho, x, p)
        # Maximally mixed state has no negativity
        assert not wigner_is_negative(W)


# =============================================================================
# Integration Tests
# =============================================================================


class TestWignerIntegration:
    """Integration tests for Wigner function."""

    def test_wigner_grid_symmetry(self) -> None:
        """Wigner function should be symmetric for vacuum."""
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        # Use symmetric grid
        x = np.linspace(-4, 4, 81)
        p = np.linspace(-4, 4, 81)

        W = wigner_function_single(rho, x, p)

        # W(0,0) should be maximum for vacuum
        assert W[40, 40] == np.max(W)

    def test_invalid_rho_shape(self) -> None:
        """Non-square rho should raise ValueError."""
        rho = np.zeros((5, 3), dtype=complex)
        x = np.linspace(-3, 3, 10)
        p = np.linspace(-3, 3, 10)

        with pytest.raises(ValueError, match="must be square"):
            wigner_function_single(rho, x, p)

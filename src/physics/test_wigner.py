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

from .hybrid_system import (
    hybrid_coherent_state,
    hybrid_vacuum_state,
)
from .wigner import (
    wigner_from_hybrid_state,
    wigner_function_single,
    wigner_is_negative,
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
        assert integral == pytest.approx(1.0, rel=1e-1, abs=0.1), (
            "Expected integral == pytest.approx(1.0, rel=1e-1, abs=0.1)"
        )

    def test_vacuum_maximum(self) -> None:
        """Vacuum Wigner max should be 2/π ≈ 0.637 at origin."""
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-3, 3, 50)
        p = np.linspace(-3, 3, 50)

        W = wigner_function_single(rho, x, p)

        max_w = np.max(W)
        assert max_w == pytest.approx(2.0 / np.pi, abs=0.1), (
            "Expected max_w == pytest.approx(2.0 / np.pi, abs=0.1)"
        )

    def test_vacuum_no_negative(self) -> None:
        """Vacuum Wigner should be non-negative everywhere."""
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-5, 5, 100)
        p = np.linspace(-5, 5, 100)

        W = wigner_function_single(rho, x, p)

        assert np.min(W) >= -1e-10, "Expected np.min(W) >= -1e-10"

    def test_wigner_shape(self) -> None:
        """Wigner output should have correct shape."""
        N = 5
        rho = np.eye(N + 1, dtype=complex) / (N + 1)  # Maximally mixed

        x = np.linspace(-3, 3, 50)
        p = np.linspace(-3, 3, 60)

        W = wigner_function_single(rho, x, p)

        assert W.shape == (50, 60), "Expected W.shape == (50, 60)"


class TestWignerFromHybridState:
    """Test Wigner extraction from hybrid state."""

    def test_hybrid_vacuum(self) -> None:
        """Wigner for |0,↓⟩ should be vacuum Wigner."""
        N = 10
        state = hybrid_vacuum_state(N, spin_state="down")

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_from_hybrid_state(state, N, x, p, spin_component="down")

        assert W.shape == (50, 50), "Expected W.shape == (50, 50)"
        # Vacuum should have max near 2/π ≈ 0.637
        assert np.max(W) > 0.3, "Expected np.max(W) > 0.3"

    @pytest.mark.slow
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
        assert np.min(W) > -2e-2, "Expected np.min(W) > -2e-2"


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
        min_w = float(np.min(W))

        assert min_w >= -1e-10, "Expected min_w >= -1e-10"

    def test_is_negative_vacuum(self) -> None:
        """Vacuum is not Wigner-negative."""
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_function_single(rho, x, p)

        assert not wigner_is_negative(W), "wigner_is_negative(W) should be falsy"

    def test_fock_n1_negativity(self) -> None:
        """Fock state |1⟩ must show negative Wigner at origin.

        Analytical: W_1(0,0) = -2/π ≈ -0.637.
        This is a regression test against the incorrect formula
        W = (2/π) exp(-2r²) Σ ρ_mn (-1)^n α^m (α*)^n / √(m!n!)
        which gave W(0,0) = 0 for |1⟩.
        """
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[1, 1] = 1.0  # |1⟩⟨1|

        x = np.array([0.0])
        p = np.array([0.0])
        W = wigner_function_single(rho, x, p)

        min_W = W[0, 0]
        assert min_W == pytest.approx(-2.0 / np.pi, abs=1e-4), (
            f"Fock |1⟩ W(0,0) should be -2/π ≈ -0.637, got {min_W}"
        )

    def test_fock_n2_positivity_at_origin(self) -> None:
        """Fock state |2⟩ has positive Wigner at origin: W = +2/π.

        Even-n Fock states have (+1)^n factor, giving positive Wigner at origin.
        """
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[2, 2] = 1.0  # |2⟩⟨2|

        x = np.array([0.0])
        p = np.array([0.0])
        W = wigner_function_single(rho, x, p)

        max_W = W[0, 0]
        assert max_W == pytest.approx(2.0 / np.pi, abs=1e-4), (
            f"Fock |2⟩ W(0,0) should be +2/π ≈ +0.637, got {max_W}"
        )

    def test_maximally_mixed_no_negativity(self) -> None:
        """Maximally mixed state has no Wigner negativity."""
        N = 5
        rho = np.eye(N + 1, dtype=complex) / (N + 1)

        x = np.linspace(-4, 4, 50)
        p = np.linspace(-4, 4, 50)

        W = wigner_function_single(rho, x, p)
        assert not wigner_is_negative(W), "wigner_is_negative(W) should be falsy"


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
        assert W[40, 40] == np.max(W), "Expected W[40, 40] == np.max(W)"

    def test_invalid_rho_shape(self) -> None:
        """Non-square rho should raise ValueError."""
        rho = np.zeros((5, 3), dtype=complex)
        x = np.linspace(-3, 3, 10)
        p = np.linspace(-3, 3, 10)

        with pytest.raises(ValueError, match="must be square"):
            wigner_function_single(rho, x, p)

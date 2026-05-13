"""
Tests for Kerr-nonlinear Mach-Zehnder Interferometer module.
"""

import numpy as np
import pytest

from src.utils.validators import validate_unitary

from .kerr_mzi import (
    compute_kerr_interference_fringe,
    compute_kerr_output_probabilities,
    compute_kerr_phase_sensitivity,
    kerr_mzi,
    kerr_phase_shift_unitary,
)
from .mzi_simulation import (
    fock_state,
    noon_state,
    phase_shift_unitary,
    vacuum_state,
)


class TestKerrPhaseShiftUnitary:
    """Test combined phase + Kerr unitary construction."""

    def test_diagonal_structure(self) -> None:
        """Kerr-phase unitary should be diagonal in Fock basis."""
        U = kerr_phase_shift_unitary(1.0, 0.1, 1.0, max_photons=2)
        assert pytest.approx(np.diag(np.diag(U))) == U, (
            "Expected U == pytest.approx(np.diag(np.diag(U)))"
        )

    def test_unitarity(self) -> None:
        """U^dagger U = I."""
        U = kerr_phase_shift_unitary(1.0, 0.1, 1.0, max_photons=2)
        assert validate_unitary(U, tol=1e-10), (
            "Condition failed: validate_unitary(U, tol=1e-10)"
        )

    def test_dimensions(self) -> None:
        """Matrix dimension should be (max_photons+1)^2."""
        for mp in [0, 1, 2, 3]:
            U = kerr_phase_shift_unitary(0.5, 0.1, 1.0, max_photons=mp)
            expected_dim = (mp + 1) ** 2
            assert U.shape == (expected_dim, expected_dim), (
                "Expected U.shape == (expected_dim, expected_dim)"
            )

    def test_reduces_to_standard_phase_shift(self) -> None:
        """When chi=0, should match phase_shift_unitary."""
        for phi in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            U_kerr = kerr_phase_shift_unitary(phi, 0.0, 1.0, max_photons=2)
            U_std = phase_shift_unitary(phi, max_photons=2)
            assert U_kerr == pytest.approx(U_std), (
                "Expected U_kerr == pytest.approx(U_std)"
            )

    def test_chi_zero_independent_of_T(self) -> None:
        """When chi=0, T should not affect the unitary."""
        U_T1 = kerr_phase_shift_unitary(1.0, 0.0, 0.0, max_photons=2)
        U_T2 = kerr_phase_shift_unitary(1.0, 0.0, 10.0, max_photons=2)
        assert pytest.approx(U_T2) == U_T1, "Expected U_T1 == pytest.approx(U_T2)"

    def test_diagonal_element_formula(self) -> None:
        """Each diagonal element should be exp(i * [phi*n2 + chi*T*(n1^2 + n2^2)])."""
        phi, chi, T = 0.5, 0.3, 2.0
        max_photons = 3
        U = kerr_phase_shift_unitary(phi, chi, T, max_photons=max_photons)
        dim = max_photons + 1
        kerr_coeff = chi * T
        for n1 in range(dim):
            for n2 in range(dim):
                idx = n1 * dim + n2
                expected = np.exp(1j * (phi * n2 + kerr_coeff * (n1**2 + n2**2)))
                assert U[idx, idx] == pytest.approx(expected), (
                    "Expected U[idx, idx] == pytest.approx(expected)"
                )

    def test_chi_T_scaling(self) -> None:
        """Same chi*T product should give same unitary."""
        U_a = kerr_phase_shift_unitary(0.5, 0.2, 3.0, max_photons=2)
        U_b = kerr_phase_shift_unitary(0.5, 0.6, 1.0, max_photons=2)
        assert U_a == pytest.approx(U_b), "Expected U_a == pytest.approx(U_b)"

    def test_negative_chi_raises(self) -> None:
        """Negative chi should raise ValueError."""
        try:
            kerr_phase_shift_unitary(1.0, -0.1, 1.0, 2)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_negative_T_raises(self) -> None:
        """Negative T should raise ValueError."""
        try:
            kerr_phase_shift_unitary(1.0, 0.1, -1.0, 2)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_negative_max_photons_raises(self) -> None:
        """Negative max_photons should raise ValueError."""
        try:
            kerr_phase_shift_unitary(1.0, 0.1, 1.0, -1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_phi_zero_gives_identity_at_chi_zero(self) -> None:
        """When phi=0 and chi=0, unitary should be identity."""
        U = kerr_phase_shift_unitary(0.0, 0.0, 1.0, max_photons=3)
        assert pytest.approx(np.eye((3 + 1) ** 2)) == U, (
            "Expected U == pytest.approx(np.eye((3 + 1) ** 2))"
        )

    def test_phi_zero_with_kerr_not_identity(self) -> None:
        """When phi=0 but chi>0, unitary should not be identity."""
        U = kerr_phase_shift_unitary(0.0, 0.5, 1.0, max_photons=2)
        assert pytest.approx(np.eye(9)) != U, "Expected U != pytest.approx(np.eye(9))"


class TestKerrMzi:
    """Test full Kerr MZI circuit."""

    def test_norm_preserved_no_kerr(self) -> None:
        """Norm preserved without Kerr."""
        state = noon_state(3, max_photons=3)
        final = kerr_mzi(state, phi=1.0, chi=0.0, T=0.0, max_photons=3)
        assert np.sum(np.abs(final) ** 2) == pytest.approx(1.0), (
            "Expected np.sum(np.abs(final) ** 2) == pytest.approx(1.0)"
        )

    def test_norm_preserved_with_kerr(self) -> None:
        """Norm preserved with Kerr."""
        state = noon_state(3, max_photons=3)
        final = kerr_mzi(state, phi=1.0, chi=0.5, T=1.0, max_photons=3)
        assert np.sum(np.abs(final) ** 2) == pytest.approx(1.0), (
            "Expected np.sum(np.abs(final) ** 2) == pytest.approx(1.0)"
        )

    def test_norm_preserved_fock_input(self) -> None:
        """Norm preserved for Fock state input."""
        state = fock_state(2, 0, 3)
        final = kerr_mzi(state, phi=0.5, chi=0.3, T=2.0, max_photons=3)
        assert np.sum(np.abs(final) ** 2) == pytest.approx(1.0), (
            "Expected np.sum(np.abs(final) ** 2) == pytest.approx(1.0)"
        )

    def test_kerr_modifies_output(self) -> None:
        """Kerr should change the output probabilities."""
        state = noon_state(3, max_photons=3)
        final_lin = kerr_mzi(state, phi=1.0, chi=0.0, T=0.0, max_photons=3)
        final_kerr = kerr_mzi(state, phi=1.0, chi=0.5, T=1.0, max_photons=3)
        P0_lin, _ = compute_kerr_output_probabilities(final_lin, 3)
        P0_kerr, _ = compute_kerr_output_probabilities(final_kerr, 3)
        assert P0_lin != pytest.approx(P0_kerr, abs=1e-6), (
            "Expected P0_lin != pytest.approx(P0_kerr, abs=1e-6)"
        )

    def test_identity_bs_at_theta_zero(self) -> None:
        """With theta=0, BS is identity, MZI reduces to phase+Kerr."""
        state = noon_state(3, max_photons=3)
        final = kerr_mzi(state, phi=0.5, chi=0.3, T=1.0, max_photons=3, theta=0.0)
        # Phase + Kerr only (no mode mixing)
        U = kerr_phase_shift_unitary(0.5, 0.3, 1.0, 3)
        expected = U @ state
        assert final == pytest.approx(expected), (
            "Expected final == pytest.approx(expected)"
        )

    def test_zero_phase_gives_no_change_without_kerr(self) -> None:
        """With phi=0, chi=0 and theta=pi/4, MZI should output same as input BS2(BS1|psi>)."""
        state = noon_state(2, max_photons=2)
        final = kerr_mzi(state, phi=0.0, chi=0.0, T=0.0, max_photons=2)
        # With phi=chi=0, the state just goes BS1 -> BS2 = identity
        P0, _P1 = compute_kerr_output_probabilities(final, 2)
        # Should produce same output distribution as NOON state (balanced)
        assert pytest.approx(0.5, abs=1e-10) == P0, (
            "Expected P0 == pytest.approx(0.5, abs=1e-10)"
        )

    def test_vacuum_balanced_output(self) -> None:
        """Vacuum input should always give 0.5/0.5 output."""
        vac = vacuum_state(2)
        for phi, chi, T in [(0.0, 0.0, 0.0), (0.5, 0.1, 1.0), (np.pi, 2.0, 0.5)]:
            final = kerr_mzi(vac, phi=phi, chi=chi, T=T, max_photons=2)
            P0, P1 = compute_kerr_output_probabilities(final, 2)
            assert pytest.approx(0.5) == P0 and np.isclose(P1, 0.5), (
                "Expected P0 == pytest.approx(0.5) and np.isclose(P1, 0.5)"
            )

    def test_invalid_state_raises(self) -> None:
        """Non-normalized state should raise ValueError."""
        bad_state = np.array([0.5, 0.5, 0.0, 0.0])
        try:
            kerr_mzi(bad_state, phi=0.0, chi=0.0, T=0.0, max_photons=1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_wrong_dimension_raises(self) -> None:
        """State with wrong dimension should raise ValueError."""
        bad_state = np.array([1.0, 0.0])  # dim=2, expected 4 for max_photons=1
        try:
            kerr_mzi(bad_state, phi=0.0, chi=0.0, T=0.0, max_photons=1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


class TestKerrOutputProbabilities:
    """Test output probability computation."""

    def test_probabilities_sum_to_one(self) -> None:
        """Probabilities should sum to 1 for any state."""
        state = noon_state(2, max_photons=2)
        final = kerr_mzi(state, phi=0.5, chi=0.2, T=1.0, max_photons=2)
        P0, P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(1.0) == P0 + P1, "Expected P0 + P1 == pytest.approx(1.0)"

    def test_probabilities_nonnegative(self) -> None:
        """Probabilities should be non-negative."""
        state = noon_state(3, max_photons=3)
        for phi in np.linspace(0, 2 * np.pi, 8):
            final = kerr_mzi(state, phi=phi, chi=0.3, T=1.0, max_photons=3)
            P0, P1 = compute_kerr_output_probabilities(final, 3)
            assert P0 >= 0 and P1 >= 0, "Expected P0 >= 0; Expected P1 >= 0"

    def test_vacuum_gives_half(self) -> None:
        """Vacuum input gives P0=P1=0.5."""
        vac = vacuum_state(2)
        final = kerr_mzi(vac, phi=1.0, chi=0.5, T=1.0, max_photons=2)
        P0, P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(0.5) == P0 and np.isclose(P1, 0.5), (
            "Expected P0 == pytest.approx(0.5) and np.isclose(P1, 0.5)"
        )

    def test_single_photon_output(self) -> None:
        """Single photon input should give valid probabilities."""
        state = fock_state(1, 0, 2)
        final = kerr_mzi(state, phi=0.5, chi=0.1, T=1.0, max_photons=2)
        P0, P1 = compute_kerr_output_probabilities(final, 2)
        assert pytest.approx(1.0) == P0 + P1, "Expected P0 + P1 == pytest.approx(1.0)"


class TestKerrPhaseSensitivity:
    """Test QFI computation for Kerr MZI."""

    def test_no_kerr_heisenberg_limit(self) -> None:
        """Without Kerr, NOON QFI = N^2 (Heisenberg limit)."""
        for N in [1, 2, 3, 4, 5]:
            F = compute_kerr_phase_sensitivity(N, 0.0, 0.0)
            assert pytest.approx(N**2) == F, f"N={N}: F_Q={F}, expected {N**2}"

    def test_with_kerr_same_qfi(self) -> None:
        """Kerr preserves QFI since generator n2 is diagonal."""
        for N in [1, 2, 3]:
            F_no_kerr = compute_kerr_phase_sensitivity(N, 0.0, 0.0)
            F_kerr = compute_kerr_phase_sensitivity(N, 0.5, 1.0)
            assert F_no_kerr == pytest.approx(F_kerr), "QFI should be same with Kerr"

    def test_qfi_independent_of_chi(self) -> None:
        """QFI should not depend on chi for different values."""
        F_base = compute_kerr_phase_sensitivity(4, 0.0, 0.0)
        for chi in [0.1, 0.5, 1.0, 2.0]:
            F = compute_kerr_phase_sensitivity(4, chi, 1.0)
            assert pytest.approx(F_base) == F, "Expected F == pytest.approx(F_base)"

    def test_default_max_photons(self) -> None:
        """When max_photons=None, should default to N."""
        F_explicit = compute_kerr_phase_sensitivity(3, 0.0, 0.0, max_photons=3)
        F_default = compute_kerr_phase_sensitivity(3, 0.0, 0.0)
        assert F_explicit == pytest.approx(F_default), (
            "Expected F_explicit == pytest.approx(F_default)"
        )

    def test_qfi_scaling(self) -> None:
        """Verify F_Q = N^2 for NOON states."""
        F_1 = compute_kerr_phase_sensitivity(1, 0.0, 0.0)
        F_2 = compute_kerr_phase_sensitivity(2, 0.0, 0.0)
        F_10 = compute_kerr_phase_sensitivity(10, 0.0, 0.0)
        assert pytest.approx(1.0) == F_1, "Expected F_1 == pytest.approx(1.0)"
        assert pytest.approx(4.0) == F_2, "Expected F_2 == pytest.approx(4.0)"
        assert pytest.approx(100.0) == F_10, "Expected F_10 == pytest.approx(100.0)"


class TestKerrInterferenceFringe:
    """Test interference fringe computation."""

    def test_fringe_shape(self) -> None:
        """Fringe should return array matching phase_range."""
        phases = np.linspace(0, 2 * np.pi, 50)
        fringe = compute_kerr_interference_fringe(
            phases,
            chi=0.1,
            T=1.0,
            max_photons=3,
            N=3,
        )
        assert fringe.shape == (50,), "Expected fringe.shape == (50,)"

    def test_fringe_values_in_range(self) -> None:
        """Probabilities should be in [0, 1]."""
        phases = np.linspace(0, 2 * np.pi, 50)
        fringe = compute_kerr_interference_fringe(
            phases,
            chi=0.1,
            T=1.0,
            max_photons=3,
            N=3,
        )
        assert np.all(fringe >= 0) and np.all(fringe <= 1), (
            "Expected np.all(fringe >= 0) and np.all(fringe <= 1)"
        )

    def test_fringe_default_initial_state(self) -> None:
        """Should work with default NOON state."""
        phases = np.linspace(0, np.pi, 20)
        fringe_default = compute_kerr_interference_fringe(
            phases,
            chi=0.0,
            T=1.0,
            max_photons=3,
            N=3,
        )
        fringe_explicit = compute_kerr_interference_fringe(
            phases,
            chi=0.0,
            T=1.0,
            max_photons=3,
            initial_state=noon_state(3, max_photons=3),
        )
        assert fringe_default == pytest.approx(fringe_explicit), (
            "Expected fringe_default == pytest.approx(fringe_explicit)"
        )

    def test_fringe_periodic(self) -> None:
        """Interference fringe should be 2pi periodic without Kerr."""
        phases = np.linspace(0, 2 * np.pi, 100)
        fringe = compute_kerr_interference_fringe(
            phases,
            chi=0.0,
            T=1.0,
            max_photons=2,
            N=2,
        )
        # Check periodicity: should be close at 0 and 2pi
        assert fringe[0] == pytest.approx(fringe[-1], abs=1e-10), (
            "Expected fringe[0] == pytest.approx(fringe[-1], abs=1e-10)"
        )

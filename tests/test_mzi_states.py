"""
Tests for MZI input states module.

Tests:
- Twin-Fock state normalization
- NOON state equal overlap with |N,0⟩ and |0,N⟩
- Orthogonality to incorrect Fock states
- ⟨J_z⟩ = 0 for Twin-Fock
- Var(J_z) = N²/4 for NOON
- Fisher information for NOON = N² (Heisenberg limit)
"""

import numpy as np
import pytest

from src.physics.mzi_simulation import noon_state
from src.physics.mzi_states import (
    twin_fock_state,
    coherent_state_two_mode,
    input_state_factory,
    create_jz_operator,
    compute_jz_expectation,
    compute_jz_variance,
    compute_fisher_information,
    single_photon_split_state,
    validate_twin_fock,
    validate_noon,
)
from src.utils.validators import validate_state_mzi


class TestTwinFockState:
    """Test Twin-Fock state creation and properties."""

    def test_twin_fock_normalization(self) -> None:
        """Twin-Fock state must be normalized."""
        for N in [2, 4, 6, 8, 10]:
            state = twin_fock_state(N)
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0), f"Twin-Fock N={N} not normalized"

    def test_twin_fock_requires_even_n(self) -> None:
        """Twin-Fock requires even N."""
        with pytest.raises(ValueError):
            twin_fock_state(N=3)

    def test_twin_fock_jz_expectation_zero(self) -> None:
        """⟨J_z⟩ = 0 for Twin-Fock (symmetric)."""
        for N in [2, 4, 6, 8]:
            state = twin_fock_state(N)
            jz_mean = np.real(compute_jz_expectation(state, N))
            assert np.isclose(jz_mean, 0.0, atol=1e-10), f"Twin-Fock N={N}: ⟨J_z⟩ ≠ 0"

    def test_twin_fock_variance(self) -> None:
        """Twin-Fock variance Var(J_z) = N(N+2)/12."""
        for N in [2, 4, 6, 8]:
            state = twin_fock_state(N)
            jz_var = compute_jz_variance(state, N)
            # Var(J_z) = N(N+2)/12
            expected_var = N * (N + 2) / 12
            assert np.isclose(jz_var, expected_var, rtol=1e-6), (
                f"Twin-Fock N={N}: Var(J_z) mismatch"
            )

    def test_twin_fock_fisher_information(self) -> None:
        """Twin-Fock Fisher information: F_Q = N(N+2)/3."""
        for N in [2, 4, 6, 8]:
            state = twin_fock_state(N)
            f_q = compute_fisher_information(state, N)
            # F_Q = 4 * Var = N(N+2)/3
            expected_fq = N * (N + 2) / 3
            assert np.isclose(f_q, expected_fq, rtol=1e-6), (
                f"Twin-Fock F_Q mismatch for N={N}"
            )

    def test_twin_fock_has_all_permutations(self) -> None:
        """Twin-Fock includes all N+1 permutations."""
        N = 4
        state = twin_fock_state(N)

        # Check that there are N+1 non-zero amplitudes (each with norm² = 1/(N+1))
        nonzero_amplitudes = np.sum(np.abs(state) > 1e-10)
        assert nonzero_amplitudes == N + 1


class TestNOONState:
    """Test NOON state creation and properties."""

    def test_noon_normalization(self) -> None:
        """NOON state must be normalized."""
        for N in [1, 2, 3, 4, 5]:
            state = noon_state(N, max_photons=N)
            norm = np.sum(np.abs(state) ** 2)
            assert np.isclose(norm, 1.0), f"NOON N={N} not normalized"

    def test_noon_equal_overlap(self) -> None:
        """NOON has equal overlap with |N,0⟩ and |0,N⟩."""
        for N in [1, 2, 3, 4, 5]:
            state = noon_state(N, max_photons=N)

            # Build |N, 0⟩
            fock_n0 = np.zeros((N + 1) ** 2, dtype=complex)
            fock_n0[N * (N + 1)] = 1.0

            # Build |0, N⟩
            fock_0n = np.zeros((N + 1) ** 2, dtype=complex)
            fock_0n[N] = 1.0

            # Check overlaps
            overlap_n0 = np.abs(np.conj(state) @ fock_n0) ** 2
            overlap_0n = np.abs(np.conj(state) @ fock_0n) ** 2

            assert np.isclose(overlap_n0, 0.5, atol=1e-10), (
                f"NOON N={N}: overlap with |N,0⟩ ≠ 0.5"
            )
            assert np.isclose(overlap_0n, 0.5, atol=1e-10), (
                f"NOON N={N}: overlap with |0,N⟩ ≠ 0.5"
            )

    def test_noon_jz_expectation_zero(self) -> None:
        """⟨J_z⟩ = 0 for NOON."""
        for N in [1, 2, 3, 4, 5]:
            state = noon_state(N, max_photons=N)
            jz_mean = np.real(compute_jz_expectation(state, N))
            assert np.isclose(jz_mean, 0.0, atol=1e-10), f"NOON N={N}: ⟨J_z⟩ ≠ 0"

    def test_noon_variance(self) -> None:
        """Var(J_z) = N²/4 for NOON."""
        for N in [1, 2, 3, 4, 5]:
            state = noon_state(N, max_photons=N)
            jz_var = compute_jz_variance(state, N)
            expected_var = N**2 / 4
            assert np.isclose(jz_var, expected_var, rtol=1e-6), (
                f"NOON N={N}: Var(J_z) mismatch"
            )

    def test_noon_fisher_information(self) -> None:
        """NOON achieves Heisenberg limit F_Q = N²."""
        for N in [1, 2, 3, 4, 5]:
            state = noon_state(N, max_photons=N)
            f_q = compute_fisher_information(state, N)
            expected_fq = N**2
            assert np.isclose(f_q, expected_fq, rtol=1e-6), (
                f"NOON N={N}: F_Q = {f_q} ≠ N² = {expected_fq}"
            )


class TestStateOrthogonality:
    """Test orthogonality properties."""

    def test_noon_orthogonal_to_middle_fock(self) -> None:
        """NOON should be orthogonal to intermediate Fock states."""
        N = 4
        state = noon_state(N, max_photons=N)

        # |N-1, 1⟩ should have zero overlap
        middle_fock = np.zeros((N + 1) ** 2, dtype=complex)
        middle_fock[(N - 1) * (N + 1) + 1] = 1.0

        overlap = np.abs(np.conj(state) @ middle_fock) ** 2
        assert np.isclose(overlap, 0.0, atol=1e-10)

    def test_coherent_state_normalized(self) -> None:
        """Two-mode coherent state should be normalized when max_photons is adequate."""
        for alpha1 in [0.5, 1.0, 1.5]:
            for alpha2 in [0.0, 0.5, 1.0]:
                # Use sufficiently large max_photons to avoid truncation
                state = coherent_state_two_mode(alpha1, alpha2, max_photons=15)
                norm = np.sum(np.abs(state) ** 2)
                assert np.isclose(norm, 1.0, rtol=1e-6), (
                    f"Coherent ({alpha1}, {alpha2}): norm = {norm}"
                )


class TestInputStateFactory:
    """Test input state factory function."""

    def test_factory_twin_fock(self) -> None:
        """Factory creates Twin-Fock correctly."""
        state = input_state_factory("twin_fock", N=4)
        assert validate_state_mzi(state)

    def test_factory_noon(self) -> None:
        """Factory creates NOON correctly."""
        state = input_state_factory("noon", N=3)
        assert validate_state_mzi(state)

    def test_factory_coherent(self) -> None:
        """Factory creates coherent state correctly."""
        state = input_state_factory("coherent", N=0, alpha1=1.0, alpha2=0.0)
        assert validate_state_mzi(state)

    def test_factory_fock(self) -> None:
        """Factory creates Fock state correctly."""
        state = input_state_factory("fock", N=3)
        assert validate_state_mzi(state)

    def test_factory_single_photon_split(self) -> None:
        """Factory creates single-photon split state correctly."""
        state = input_state_factory("single_photon_split", N=2)
        assert validate_state_mzi(state)

    def test_factory_css(self) -> None:
        """Factory creates CSS (coherent state split) correctly."""
        # CSS is like coherent state with one mode
        state = input_state_factory("css", N=1)
        assert validate_state_mzi(state)

    def test_factory_unknown_raises(self) -> None:
        """Factory raises for unknown state type."""
        with pytest.raises(ValueError):
            input_state_factory("unknown_type", N=1)


class TestValidation:
    """Test validation functions."""

    def test_validate_twin_fock(self) -> None:
        """Twin-Fock validation passes."""
        N = 4
        state = twin_fock_state(N)
        assert validate_twin_fock(N, state, N)

    def test_validate_noon(self) -> None:
        """NOON validation passes."""
        N = 3
        state = noon_state(N, max_photons=N)
        assert validate_noon(N, state, N)


class TestJ_z_Operator:
    """Test J_z operator creation."""

    def test_jz_hermitian(self) -> None:
        """J_z operator should be Hermitian."""
        jz = create_jz_operator(max_photons=3)
        assert np.allclose(jz, jz.conj().T)

    def test_jz_eigenvalues(self) -> None:
        """J_z eigenvalues should be (n1-n2)/2."""
        jz = create_jz_operator(max_photons=2)

        # Check diagonal
        for n1 in range(3):
            for n2 in range(3):
                idx = n1 * 3 + n2
                expected = (n1 - n2) / 2
                assert np.isclose(jz[idx, idx], expected)


class TestSinglePhotonSplit:
    """Test single-photon split state."""

    def test_sps_normalization(self) -> None:
        """Single-photon split must be normalized."""
        state = single_photon_split_state(N=3)
        norm = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm, 1.0)

    def test_sps_requires_n_ge_2(self) -> None:
        """Single-photon split requires N >= 2."""
        with pytest.raises(ValueError):
            single_photon_split_state(N=1)

    def test_sps_symmetric(self) -> None:
        """Single-photon split should be symmetric."""
        state = single_photon_split_state(N=4)
        jz_mean = np.real(compute_jz_expectation(state, 4))
        assert np.isclose(jz_mean, 0.0, atol=1e-10)

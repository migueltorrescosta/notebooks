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
import qutip

from src.physics.mzi_simulation import noon_state
from src.utils.validators import validate_state_mzi

from .mzi_states import (
    compute_fisher_information,
    compute_jz_expectation,
    compute_jz_variance,
    input_state_factory,
    single_photon_split_state,
    twin_fock_state,
    two_mode_jz_operator,
    validate_noon,
    validate_twin_fock,
)


class TestTwinFockState:
    """Test Twin-Fock state creation and properties."""

    def test_twin_fock_state_must_be_normalized(self) -> None:
        for N in [2, 4, 6, 8, 10]:
            state = twin_fock_state(N)
            norm = np.sum(np.abs(state) ** 2)
            assert norm == pytest.approx(1.0), f"Twin-Fock N={N} not normalized"

    def test_twin_fock_requires_even_n(self) -> None:
        with pytest.raises(ValueError):
            twin_fock_state(N=3)

    def test_j_z_0_for_twin_fock_symmetric(self) -> None:
        for N in [2, 4, 6, 8]:
            state = twin_fock_state(N)
            jz_mean = np.real(compute_jz_expectation(state, N))
            assert jz_mean == pytest.approx(0.0, abs=1e-10), (
                f"Twin-Fock N={N}: ⟨J_z⟩ ≠ 0"
            )

    def test_twin_fock_variance_var_j_z_n_n_2_12(self) -> None:
        for N in [2, 4, 6, 8]:
            state = twin_fock_state(N)
            jz_var = compute_jz_variance(state, N)
            # Var(J_z) = N(N+2)/12
            expected_var = N * (N + 2) / 12
            assert jz_var == pytest.approx(expected_var, rel=1e-6), (
                f"Twin-Fock N={N}: Var(J_z) mismatch"
            )

    def test_twin_fock_fisher_information_f_q_n_n_2_3(self) -> None:
        for N in [2, 4, 6, 8]:
            state = twin_fock_state(N)
            f_q = compute_fisher_information(state, N)
            # F_Q = 4 * Var = N(N+2)/3
            expected_fq = N * (N + 2) / 3
            assert f_q == pytest.approx(expected_fq, rel=1e-6), (
                f"Twin-Fock F_Q mismatch for N={N}"
            )

    def test_twin_fock_includes_all_n_1_permutations(self) -> None:
        N = 4
        state = twin_fock_state(N)

        # Check that there are N+1 non-zero amplitudes (each with norm² = 1/(N+1))
        nonzero_amplitudes = np.sum(np.abs(state) > 1e-10)
        assert nonzero_amplitudes == N + 1, "Expected nonzero_amplitudes == N + 1"


class TestNOONState:
    """Test NOON state creation and properties."""

    def test_noon_state_must_be_normalized(self) -> None:
        for N in [1, 2, 3, 4, 5]:
            state = noon_state(N, max_photons=N)
            norm = np.sum(np.abs(state) ** 2)
            assert norm == pytest.approx(1.0), f"NOON N={N} not normalized"

    def test_noon_has_equal_overlap_with_n_0_and_0_n(self) -> None:
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

            assert overlap_n0 == pytest.approx(0.5, abs=1e-10), (
                f"NOON N={N}: overlap with |N,0⟩ ≠ 0.5"
            )
            assert overlap_0n == pytest.approx(0.5, abs=1e-10), (
                f"NOON N={N}: overlap with |0,N⟩ ≠ 0.5"
            )

    def test_j_z_0_for_noon(self) -> None:
        for N in [1, 2, 3, 4, 5]:
            state = noon_state(N, max_photons=N)
            jz_mean = np.real(compute_jz_expectation(state, N))
            assert jz_mean == pytest.approx(0.0, abs=1e-10), f"NOON N={N}: ⟨J_z⟩ ≠ 0"

    def test_var_j_z_n_4_for_noon(self) -> None:
        for N in [1, 2, 3, 4, 5]:
            state = noon_state(N, max_photons=N)
            jz_var = compute_jz_variance(state, N)
            expected_var = N**2 / 4
            assert jz_var == pytest.approx(expected_var, rel=1e-6), (
                f"NOON N={N}: Var(J_z) mismatch"
            )

    def test_noon_achieves_heisenberg_limit_f_q_n(self) -> None:
        for N in [1, 2, 3, 4, 5]:
            state = noon_state(N, max_photons=N)
            f_q = compute_fisher_information(state, N)
            expected_fq = N**2
            assert f_q == pytest.approx(expected_fq, rel=1e-6), (
                f"NOON N={N}: F_Q = {f_q} ≠ N² = {expected_fq}"
            )


class TestStateOrthogonality:
    """Test orthogonality properties."""

    def test_noon_should_be_orthogonal_to_intermediate_fock_states(self) -> None:
        N = 4
        state = noon_state(N, max_photons=N)

        # |N-1, 1⟩ should have zero overlap
        middle_fock = np.zeros((N + 1) ** 2, dtype=complex)
        middle_fock[(N - 1) * (N + 1) + 1] = 1.0

        overlap = np.abs(np.conj(state) @ middle_fock) ** 2
        assert overlap == pytest.approx(0.0, abs=1e-10), (
            "Expected overlap == pytest.approx(0.0, abs=1e-10)"
        )

    def test_two_mode_coherent_state_should_be_normalized_when_max_photons_is_adequate(
        self,
    ) -> None:
        for alpha1 in [0.5, 1.0, 1.5]:
            for alpha2 in [0.0, 0.5, 1.0]:
                # Use sufficiently large max_photons to avoid truncation
                dim = 15 + 1
                state = (
                    qutip.tensor(
                        qutip.coherent(dim, alpha1), qutip.coherent(dim, alpha2)
                    )
                    .full()
                    .ravel()
                )
                norm = np.sum(np.abs(state) ** 2)
                assert norm == pytest.approx(1.0, rel=1e-6), (
                    f"Coherent ({alpha1}, {alpha2}): norm = {norm}"
                )


class TestInputStateFactory:
    """Test input state factory function."""

    def test_factory_creates_twin_fock_correctly(self) -> None:
        state = input_state_factory("twin_fock", N=4)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_factory_creates_noon_correctly(self) -> None:
        state = input_state_factory("noon", N=3)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_factory_creates_coherent_state_correctly(self) -> None:
        state = input_state_factory("coherent", N=0, alpha1=1.0, alpha2=0.0)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_factory_creates_fock_state_correctly(self) -> None:
        state = input_state_factory("fock", N=3)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_factory_creates_single_photon_split_state_correctly(self) -> None:
        state = input_state_factory("single_photon_split", N=2)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_factory_creates_css_coherent_state_split_correctly(self) -> None:
        # CSS is like coherent state with one mode
        state = input_state_factory("css", N=1)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_factory_raises_for_unknown_state_type(self) -> None:
        with pytest.raises(ValueError):
            input_state_factory("unknown_type", N=1)


class TestValidation:
    """Test validation functions."""

    def test_twin_fock_validation_passes(self) -> None:
        N = 4
        state = twin_fock_state(N)
        assert validate_twin_fock(N, state, N), (
            "Condition failed: validate_twin_fock(N, state, N)"
        )

    def test_noon_validation_passes(self) -> None:
        N = 3
        state = noon_state(N, max_photons=N)
        assert validate_noon(N, state, N), (
            "Condition failed: validate_noon(N, state, N)"
        )


class TestJ_z_Operator:
    """Test J_z operator creation."""

    def test_j_z_operator_should_be_hermitian(self) -> None:
        jz = two_mode_jz_operator(max_photons=3)
        assert jz == pytest.approx(jz.conj().T), (
            "Expected jz == pytest.approx(jz.conj().T)"
        )

    def test_j_z_eigenvalues_should_be_n1_n2_2(self) -> None:
        jz = two_mode_jz_operator(max_photons=2)

        # Check diagonal
        for n1 in range(3):
            for n2 in range(3):
                idx = n1 * 3 + n2
                expected = (n1 - n2) / 2
                assert jz[idx, idx] == pytest.approx(expected), (
                    "Expected jz[idx, idx] == pytest.approx(expected)"
                )


class TestSinglePhotonSplit:
    """Test single-photon split state."""

    def test_single_photon_split_must_be_normalized(self) -> None:
        state = single_photon_split_state(N=3)
        norm = np.sum(np.abs(state) ** 2)
        assert norm == pytest.approx(1.0), "Expected norm == pytest.approx(1.0)"

    def test_single_photon_split_requires_n_2(self) -> None:
        with pytest.raises(ValueError):
            single_photon_split_state(N=1)

    def test_single_photon_split_should_be_symmetric(self) -> None:
        state = single_photon_split_state(N=4)
        jz_mean = np.real(compute_jz_expectation(state, 4))
        assert jz_mean == pytest.approx(0.0, abs=1e-10), (
            "Expected jz_mean == pytest.approx(0.0, abs=1e-10)"
        )

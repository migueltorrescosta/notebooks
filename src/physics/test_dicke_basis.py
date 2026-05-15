"""Tests for dicke_basis.py - Dicke basis representation for N-atom systems."""

from __future__ import annotations

import numpy as np
import pytest

from .dicke_basis import (
    basis_transformation_matrix,
    dicke_states,
    from_dicke_basis,
    jx_operator,
    jy_operator,
    jz_operator,
    to_dicke_basis,
)


class TestDickeStates:
    """Test suite for dicke_states function."""

    def test_test_that_dictionary_keys_are_correct_magnetic_quantum_numbers(
        self,
    ) -> None:
        for N in [1, 2, 3, 4, 5, 10]:
            J = N / 2.0
            basis = dicke_states(N)
            expected_m_values = set(np.arange(J, -J - 1, -1))
            assert set(basis.keys()) == expected_m_values, (
                "Expected set(basis.keys()) == expected_m_values"
            )

    def test_test_that_output_has_n_1_entries(self) -> None:
        for N in [0, 1, 2, 5, 10, 50]:
            basis = dicke_states(N)
            assert len(basis) == N + 1, "Expected len(basis) == N + 1"

    def test_test_that_indices_increase_correctly(self) -> None:
        N = 4
        basis = dicke_states(N)
        # m=N/2 should have lowest index (0), m=-N/2 highest (N)
        J = N / 2.0
        assert basis[float(J)] == 0  # m=2.0 at idx 0
        assert basis[float(-J)] == N  # m=-2.0 at idx 4

    def test_test_that_negative_n_raises_valueerror(self) -> None:
        with pytest.raises(ValueError):
            dicke_states(-1)


class TestBasisConversions:
    """Test suite for to_dicke_basis and from_dicke_basis."""

    def test_test_conversion_roundtrip_for_dicke_states_with_n_2(self) -> None:
        N = 2
        # For Dicke states with total N=2, states are |n1,n2⟩ with n1+n2=2
        # n1=2, n2=0 → m=1; n1=1, n2=1 → m=0; n1=0, n2=2 → m=-1

        # |2,0⟩ → m = 1 → idx 0 in Dicke basis
        fock = np.zeros((N + 1) ** 2, dtype=complex)
        idx = 2 * (N + 1) + 0  # |2,0⟩ at index 6
        fock[idx] = 1.0

        dicke = to_dicke_basis(fock, N)
        fock_back = from_dicke_basis(dicke, N)

        assert fock == pytest.approx(fock_back), (
            "Expected fock == pytest.approx(fock_back)"
        )

    def test_test_conversion_roundtrip_for_dicke_states(self) -> None:
        for N in [2, 3, 5]:
            # |N,0⟩ → m = N/2 (max m)
            fock = np.zeros((N + 1) ** 2, dtype=complex)
            idx = N * (N + 1)  # |N,0⟩ at index N*(N+1)
            fock[idx] = 1.0

            dicke = to_dicke_basis(fock, N)
            fock_back = from_dicke_basis(dicke, N)
            assert fock == pytest.approx(fock_back), (
                "Expected fock == pytest.approx(fock_back)"
            )

            # |0,N⟩ → m = -N/2 (min m)
            fock = np.zeros((N + 1) ** 2, dtype=complex)
            fock[N] = 1.0  # |0,N⟩ at index N
            dicke = to_dicke_basis(fock, N)
            fock_back = from_dicke_basis(dicke, N)
            assert fock == pytest.approx(fock_back), (
                "Expected fock == pytest.approx(fock_back)"
            )

            # |N/2,N/2⟩ → m = 0 (middle state, if N is even)
            if N % 2 == 0:
                half = N // 2
                fock = np.zeros((N + 1) ** 2, dtype=complex)
                idx = half * (N + 1) + half  # |N/2,N/2⟩
                fock[idx] = 1.0
                dicke = to_dicke_basis(fock, N)
                fock_back = from_dicke_basis(dicke, N)
                assert fock == pytest.approx(fock_back), (
                    "Expected fock == pytest.approx(fock_back)"
                )

    def test_test_conversion_roundtrip_for_arbitrary_dicke_state(self) -> None:
        N = 4
        # Create a random state in the symmetric subspace only
        rng = np.random.default_rng(42)
        # Only fill symmetric subspace (n1 + n2 = N)
        fock = np.zeros((N + 1) ** 2, dtype=complex)
        for n1 in range(N + 1):
            n2 = N - n1
            idx = n1 * (N + 1) + n2
            fock[idx] = rng.random() + 1j * rng.random()
        fock = fock / np.linalg.norm(fock)

        dicke = to_dicke_basis(fock, N)
        fock_back = from_dicke_basis(dicke, N)

        assert fock == pytest.approx(fock_back), (
            "Expected fock == pytest.approx(fock_back)"
        )

    def test_test_that_dicke_basis_dimension_is_n_1(self) -> None:
        for N in [1, 2, 3, 5, 10]:
            # Create Fock state in symmetric subspace
            fock = np.zeros((N + 1) ** 2, dtype=complex)
            fock[N * (N + 1)] = 1.0  # |N,0⟩
            dicke = to_dicke_basis(fock, N)
            assert len(dicke) == N + 1, "Expected len(dicke) == N + 1"

    def test_test_that_basis_transformation_is_unitary(self) -> None:
        for N in [1, 2, 3, 4, 5]:
            T = basis_transformation_matrix(N)
            assert T.shape == ((N + 1) ** 2, N + 1), (
                "Expected T.shape == ((N + 1) ** 2, N + 1)"
            )
            # T^\dagger T should be identity in the symmetric subspace
            T_dag_T = T.conj().T @ T
            assert T_dag_T == pytest.approx(np.eye(N + 1), abs=1e-10), (
                "Expected T_dag_T == pytest.approx(np.eye(N + 1), abs=1e-10)"
            )

    def test_test_that_wrong_dimensions_raise_valueerror(self) -> None:
        N = 2
        wrong_dim = np.zeros(5)  # Wrong dimension
        with pytest.raises(ValueError):
            to_dicke_basis(wrong_dim, N)

    def test_test_that_negative_n_raises_valueerror(self) -> None:
        fock = np.zeros(9, dtype=complex)
        fock[0] = 1.0
        with pytest.raises(ValueError):
            to_dicke_basis(fock, -1)

    def test_test_that_fock_to_dicke_mapping_is_correct_for_specific_states(
        self,
    ) -> None:
        for N in [2, 3, 4]:
            # |N,0⟩ → m = N/2 → idx 0
            fock = np.zeros((N + 1) ** 2, dtype=complex)
            fock[N * (N + 1)] = 1.0
            dicke = to_dicke_basis(fock, N)
            assert np.abs(dicke[0]) == 1.0  # Should be at idx 0

            # |0,N⟩ → m = -N/2 → idx N
            fock = np.zeros((N + 1) ** 2, dtype=complex)
            fock[N] = 1.0
            dicke = to_dicke_basis(fock, N)
            assert np.abs(dicke[N]) == 1.0  # Should be at idx N


class TestJzOperator:
    """Test suite for jz_operator function."""

    def test_test_that_j_z_is_diagonal(self) -> None:
        for N in [1, 2, 3, 5, 10]:
            J_z = jz_operator(N)
            off_diagonal = J_z - np.diag(np.diag(J_z))
            assert off_diagonal == pytest.approx(0), (
                "Expected off_diagonal == pytest.approx(0)"
            )

    def test_test_that_diagonal_matches_expected_eigenvalues(self) -> None:
        for N in [1, 2, 5, 10]:
            J_z = jz_operator(N)
            eigenvalues = np.arange(N / 2.0, -N / 2.0 - 1, -1)
            assert J_z.diagonal() == pytest.approx(eigenvalues), (
                "Expected J_z.diagonal() == pytest.approx(eigenvalues)"
            )

    def test_test_that_j_z_is_hermitian(self) -> None:
        for N in [1, 2, 5]:
            J_z = jz_operator(N)
            assert J_z == pytest.approx(J_z.T.conj()), (
                "Expected J_z == pytest.approx(J_z.T.conj())"
            )

    def test_test_that_dimension_is_n_1(self) -> None:
        for N in [1, 2, 5, 10]:
            J_z = jz_operator(N)
            assert J_z.shape == (N + 1, N + 1), "Expected J_z.shape == (N + 1, N + 1)"


class TestJxOperator:
    """Test suite for jx_operator function."""

    def test_test_that_j_x_is_symmetric_real(self) -> None:
        for N in [1, 2, 3, 5, 10]:
            J_x = jx_operator(N)
            assert J_x == pytest.approx(J_x.T), "Expected J_x == pytest.approx(J_x.T)"

    def test_test_j_x_has_non_zero_only_on_super_sub_diagonals(self) -> None:
        N = 5
        J_x = jx_operator(N)
        for i in range(N + 1):
            for j in range(N + 1):
                if abs(i - j) > 1:
                    assert J_x[i, j] == pytest.approx(0), (
                        "Expected J_x[i, j] == pytest.approx(0)"
                    )

    def test_test_j_x_matrix_elements_match_analytical_formula(self) -> None:
        N = 4
        J = N / 2.0
        J_x = jx_operator(N)

        for i in range(N):
            m = J - i
            expected = 0.5 * np.sqrt((J + m) * (J - m + 1))
            assert J_x[i + 1, i] == pytest.approx(expected), (
                "Expected J_x[i + 1, i] == pytest.approx(expected)"
            )
            assert J_x[i, i + 1] == pytest.approx(expected), (
                "Expected J_x[i, i + 1] == pytest.approx(expected)"
            )

    def test_test_that_j_x_is_hermitian(self) -> None:
        for N in [1, 2, 5]:
            J_x = jx_operator(N)
            assert J_x == pytest.approx(J_x.conj().T), (
                "Expected J_x == pytest.approx(J_x.conj().T)"
            )

    def test_test_that_dimension_is_n_1(self) -> None:
        for N in [1, 2, 5, 10]:
            J_x = jx_operator(N)
            assert J_x.shape == (N + 1, N + 1), "Expected J_x.shape == (N + 1, N + 1)"


class TestCommutationRelations:
    """Test SU(2) commutation relations [J_i, J_j] = i ε_ijk J_k."""

    def test_test_j_x_j_y_i_j_z(self) -> None:
        for N in [1, 2, 3, 5]:
            J_x = jx_operator(N)
            J_y = jy_operator(N)
            J_z = jz_operator(N)

            commutator = J_x @ J_y - J_y @ J_x
            expected = 1j * J_z
            assert commutator == pytest.approx(expected, abs=1e-10), (
                "Expected commutator == pytest.approx(expected, abs=1e-10)"
            )

    def test_test_j_y_j_z_i_j_x(self) -> None:
        for N in [1, 2, 3, 5]:
            J_x = jx_operator(N)
            J_y = jy_operator(N)
            J_z = jz_operator(N)

            commutator = J_y @ J_z - J_z @ J_y
            expected = 1j * J_x
            assert commutator == pytest.approx(expected, abs=1e-10), (
                "Expected commutator == pytest.approx(expected, abs=1e-10)"
            )

    def test_test_j_z_j_x_i_j_y(self) -> None:
        for N in [1, 2, 3, 5]:
            J_x = jx_operator(N)
            J_y = jy_operator(N)
            J_z = jz_operator(N)

            commutator = J_z @ J_x - J_x @ J_z
            expected = 1j * J_y
            assert commutator == pytest.approx(expected, abs=1e-10), (
                "Expected commutator == pytest.approx(expected, abs=1e-10)"
            )


class TestPhysicalInvariants:
    """Test physical invariants that must hold."""

    def test_test_tr_j_z_0_for_half_integer_spin_any_n(self) -> None:
        for N in [1, 2, 3, 4, 5]:
            J_z = jz_operator(N)
            trace = np.trace(J_z)
            assert trace == pytest.approx(0, abs=1e-10), (
                "Expected trace == pytest.approx(0, abs=1e-10)"
            )

    def test_test_j_x_has_no_imaginary_part(self) -> None:
        for N in [1, 2, 5]:
            J_x = jx_operator(N)
            assert np.isrealobj(J_x), "Condition failed: np.isrealobj(J_x)"

    def test_test_j_y_is_hermitian_j_y_j_y_dagger(self) -> None:
        for N in [1, 2, 3, 5]:
            J_y = jy_operator(N)
            # Check Hermitian: J_y = J_y^\dagger
            assert J_y == pytest.approx(J_y.T.conj()), (
                "Expected J_y == pytest.approx(J_y.T.conj())"
            )


class TestEdgeCases:
    """Test edge cases and special values."""

    def test_test_n_0_single_atom(self) -> None:
        J_z = jz_operator(0)
        assert J_z.shape == (1, 1), "Expected J_z.shape == (1, 1)"
        assert J_z[0, 0] == pytest.approx(0.0), (
            "Expected J_z[0, 0] == pytest.approx(0.0)"
        )

    def test_test_n_1_spin_1_2(self) -> None:
        J_z = jz_operator(1)
        J_x = jx_operator(1)

        # J_z = diag(0.5, -0.5)
        assert np.allclose(J_z, [[0.5, 0], [0, -0.5]]), "J_z should be diag(0.5, -0.5)"
        # J_x = [[0, 0.5], [0.5, 0]]
        assert np.allclose(J_x, [[0, 0.5], [0.5, 0]]), (
            "J_x should be [[0, 0.5], [0.5, 0]]"
        )

    def test_test_n_2_spin_1(self) -> None:
        J_z = jz_operator(2)
        J_x = jx_operator(2)

        # J_z = diag(1, 0, -1)
        assert np.allclose(J_z, [[1, 0, 0], [0, 0, 0], [0, 0, -1]]), (
            "J_z should be diag(1, 0, -1)"
        )
        # J_x[1,0] = J_x[0,1] = J_x[2,1] = J_x[1,2] = 1/sqrt(2)
        sqrt2 = np.sqrt(2)
        expected_jx = [[0, sqrt2 / 2, 0], [sqrt2 / 2, 0, sqrt2 / 2], [0, sqrt2 / 2, 0]]
        assert np.allclose(J_x, expected_jx), "J_x should match expected_jx"


class TestNumericalStability:
    """Test numerical stability for larger N."""

    def test_test_that_operations_complete_for_n_100(self) -> None:
        N = 100
        J_z = jz_operator(N)
        J_x = jx_operator(N)
        J_y = jy_operator(N)

        assert J_z.shape == (N + 1, N + 1), "Expected J_z.shape == (N + 1, N + 1)"
        assert J_x.shape == (N + 1, N + 1), "Expected J_x.shape == (N + 1, N + 1)"
        assert J_y.shape == (N + 1, N + 1), "Expected J_y.shape == (N + 1, N + 1)"

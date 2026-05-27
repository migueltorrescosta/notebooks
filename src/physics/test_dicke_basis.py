"""Tests for dicke_basis.py - Dicke basis representation for N-atom systems."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.enums import OperatorBasis

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
    @pytest.mark.parametrize(
        "N", [1, 2, 3, 4, 5, 10], ids=["N=1", "N=2", "N=3", "N=4", "N=5", "N=10"]
    )
    def test_keys_correspond_to_magnetic_quantum_numbers(self, N: int) -> None:
        J = N / 2.0
        basis = dicke_states(N)
        expected_m_values = set(np.arange(J, -J - 1, -1))
        assert set(basis.keys()) == expected_m_values

    @pytest.mark.parametrize(
        "N", [0, 1, 2, 5, 10, 50], ids=["N=0", "N=1", "N=2", "N=5", "N=10", "N=50"]
    )
    def test_output_has_n_plus_one_entries(self, N: int) -> None:
        basis = dicke_states(N)
        assert len(basis) == N + 1

    def test_indices_decrease_with_magnetic_quantum_number(self) -> None:
        N = 4
        basis = dicke_states(N)
        J = N / 2.0
        assert basis[float(J)] == 0  # m=+J at idx 0
        assert basis[float(-J)] == N  # m=-J at idx N

    def test_given_negative_n_then_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            dicke_states(-1)


class TestBasisConversions:
    def test_given_n2_then_roundtrip(self) -> None:
        N = 2
        fock = np.zeros((N + 1) ** 2, dtype=complex)
        idx = 2 * (N + 1) + 0  # |2,0⟩ at index 6
        fock[idx] = 1.0

        dicke = to_dicke_basis(fock, N)
        fock_back = from_dicke_basis(dicke, N)

        assert fock == pytest.approx(fock_back)

    @pytest.mark.parametrize("N", [2, 3, 5], ids=["N=2", "N=3", "N=5"])
    def test_given_symmetric_fock_states_then_roundtrip(self, N: int) -> None:
        # |N,0⟩ → m = N/2 (max m)
        fock = np.zeros((N + 1) ** 2, dtype=complex)
        idx = N * (N + 1)
        fock[idx] = 1.0

        dicke = to_dicke_basis(fock, N)
        fock_back = from_dicke_basis(dicke, N)
        assert fock == pytest.approx(fock_back)

        # |0,N⟩ → m = -N/2 (min m)
        fock = np.zeros((N + 1) ** 2, dtype=complex)
        fock[N] = 1.0
        dicke = to_dicke_basis(fock, N)
        fock_back = from_dicke_basis(dicke, N)
        assert fock == pytest.approx(fock_back)

        # |N/2,N/2⟩ → m = 0 (middle state, if N is even)
        if N % 2 == 0:
            half = N // 2
            fock = np.zeros((N + 1) ** 2, dtype=complex)
            idx = half * (N + 1) + half
            fock[idx] = 1.0
            dicke = to_dicke_basis(fock, N)
            fock_back = from_dicke_basis(dicke, N)
            assert fock == pytest.approx(fock_back)

    def test_given_arbitrary_symmetric_state_then_roundtrip(self) -> None:
        N = 4
        rng = np.random.default_rng(42)
        fock = np.zeros((N + 1) ** 2, dtype=complex)
        for n1 in range(N + 1):
            n2 = N - n1
            idx = n1 * (N + 1) + n2
            fock[idx] = rng.random() + 1j * rng.random()
        fock = fock / np.linalg.norm(fock)

        dicke = to_dicke_basis(fock, N)
        fock_back = from_dicke_basis(dicke, N)

        assert fock == pytest.approx(fock_back)

    @pytest.mark.parametrize(
        "N", [1, 2, 3, 5, 10], ids=["N=1", "N=2", "N=3", "N=5", "N=10"]
    )
    def test_basis_dimension_is_n_plus_one(self, N: int) -> None:
        fock = np.zeros((N + 1) ** 2, dtype=complex)
        fock[N * (N + 1)] = 1.0  # |N,0⟩
        dicke = to_dicke_basis(fock, N)
        assert len(dicke) == N + 1

    @pytest.mark.parametrize(
        "N", [1, 2, 3, 4, 5], ids=["N=1", "N=2", "N=3", "N=4", "N=5"]
    )
    def test_basis_transformation_is_isometry(self, N: int) -> None:
        T = basis_transformation_matrix(N)
        assert T.shape == ((N + 1) ** 2, N + 1)
        T_dag_T = T.conj().T @ T
        assert T_dag_T == pytest.approx(np.eye(N + 1), abs=1e-10)

    def test_given_wrong_input_dimension_then_raises_value_error(self) -> None:
        N = 2
        wrong_dim = np.zeros(5)
        with pytest.raises(ValueError):
            to_dicke_basis(wrong_dim, N)

    def test_given_negative_n_then_raises_value_error(self) -> None:
        fock = np.zeros(9, dtype=complex)
        fock[0] = 1.0
        with pytest.raises(ValueError):
            to_dicke_basis(fock, -1)

    @pytest.mark.parametrize("N", [2, 3, 4], ids=["N=2", "N=3", "N=4"])
    def test_fock_to_dicke_maps_extremal_states_correctly(self, N: int) -> None:
        # |N,0⟩ → m = N/2 → idx 0
        fock = np.zeros((N + 1) ** 2, dtype=complex)
        fock[N * (N + 1)] = 1.0
        dicke = to_dicke_basis(fock, N)
        assert np.abs(dicke[0]) == 1.0

        # |0,N⟩ → m = -N/2 → idx N
        fock = np.zeros((N + 1) ** 2, dtype=complex)
        fock[N] = 1.0
        dicke = to_dicke_basis(fock, N)
        assert np.abs(dicke[N]) == 1.0


class TestJzOperator:
    @pytest.mark.parametrize(
        "N", [1, 2, 3, 5, 10], ids=["N=1", "N=2", "N=3", "N=5", "N=10"]
    )
    def test_jz_is_diagonal(self, N: int) -> None:
        J_z = jz_operator(N)
        off_diagonal = J_z - np.diag(np.diag(J_z))
        assert off_diagonal == pytest.approx(0)

    @pytest.mark.parametrize("N", [1, 2, 5, 10], ids=["N=1", "N=2", "N=5", "N=10"])
    def test_jz_diagonal_matches_eigenvalues(self, N: int) -> None:
        J_z = jz_operator(N)
        eigenvalues = np.arange(N / 2.0, -N / 2.0 - 1, -1)
        assert J_z.diagonal() == pytest.approx(eigenvalues)

    @pytest.mark.parametrize("N", [1, 2, 5], ids=["N=1", "N=2", "N=5"])
    def test_jz_is_hermitian(self, N: int) -> None:
        J_z = jz_operator(N)
        assert J_z == pytest.approx(J_z.T.conj())

    @pytest.mark.parametrize("N", [1, 2, 5, 10], ids=["N=1", "N=2", "N=5", "N=10"])
    def test_jz_dimension_is_n_plus_one(self, N: int) -> None:
        J_z = jz_operator(N)
        assert J_z.shape == (N + 1, N + 1)


class TestJzBasisConventions:
    """Verify jz_operator basis conventions are consistent."""

    @pytest.mark.parametrize("N", [2, 4, 10])
    def test_given_dicke_basis_explicit_vs_default_then_match(self, N: int) -> None:
        mat_default = jz_operator(N)
        mat_explicit = jz_operator(N, basis=OperatorBasis.DICKE)
        np.testing.assert_allclose(
            mat_default,
            mat_explicit,
            rtol=1e-12,
            err_msg=f"Default basis does not match explicit DICKE for N={N}",
        )

    @pytest.mark.parametrize("N", [2, 4, 10])
    def test_given_fock_basis_is_reversed_dicke_then_have_reversed_eigenvalues(
        self, N: int
    ) -> None:
        mat_dicke = jz_operator(N, basis=OperatorBasis.DICKE)
        mat_fock = jz_operator(N, basis=OperatorBasis.FOCK)

        dicke_diag = np.diag(mat_dicke)
        fock_diag = np.diag(mat_fock)
        np.testing.assert_allclose(
            fock_diag,
            dicke_diag[::-1],
            rtol=1e-12,
            err_msg=f"FOCK J_z eigenvalues are not reversed DICKE for N={N}",
        )


class TestJxOperator:
    @pytest.mark.parametrize(
        "N", [1, 2, 3, 5, 10], ids=["N=1", "N=2", "N=3", "N=5", "N=10"]
    )
    def test_jx_is_real_symmetric(self, N: int) -> None:
        J_x = jx_operator(N)
        assert J_x == pytest.approx(J_x.T)

    def test_jx_nonzero_only_on_first_off_diagonals(self) -> None:
        N = 5
        J_x = jx_operator(N)
        for i in range(N + 1):
            for j in range(N + 1):
                if abs(i - j) > 1:
                    assert J_x[i, j] == pytest.approx(0)

    def test_jx_matrix_elements_match_analytical_formula(self) -> None:
        N = 4
        J = N / 2.0
        J_x = jx_operator(N)

        for i in range(N):
            m = J - i
            expected = 0.5 * np.sqrt((J + m) * (J - m + 1))
            assert J_x[i + 1, i] == pytest.approx(expected)
            assert J_x[i, i + 1] == pytest.approx(expected)

    @pytest.mark.parametrize("N", [1, 2, 5], ids=["N=1", "N=2", "N=5"])
    def test_jx_is_hermitian(self, N: int) -> None:
        J_x = jx_operator(N)
        assert J_x == pytest.approx(J_x.conj().T)

    @pytest.mark.parametrize("N", [1, 2, 5, 10], ids=["N=1", "N=2", "N=5", "N=10"])
    def test_jx_dimension_is_n_plus_one(self, N: int) -> None:
        J_x = jx_operator(N)
        assert J_x.shape == (N + 1, N + 1)


class TestCommutationRelations:
    @pytest.mark.parametrize("N", [1, 2, 3, 5], ids=["N=1", "N=2", "N=3", "N=5"])
    def test_commutator_jx_jy_equals_i_jz(self, N: int) -> None:
        J_x = jx_operator(N)
        J_y = jy_operator(N)
        J_z = jz_operator(N)

        commutator = J_x @ J_y - J_y @ J_x
        expected = 1j * J_z
        assert commutator == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 3, 5], ids=["N=1", "N=2", "N=3", "N=5"])
    def test_commutator_jy_jz_equals_i_jx(self, N: int) -> None:
        J_x = jx_operator(N)
        J_y = jy_operator(N)
        J_z = jz_operator(N)

        commutator = J_y @ J_z - J_z @ J_y
        expected = 1j * J_x
        assert commutator == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 3, 5], ids=["N=1", "N=2", "N=3", "N=5"])
    def test_commutator_jz_jx_equals_i_jy(self, N: int) -> None:
        J_x = jx_operator(N)
        J_y = jy_operator(N)
        J_z = jz_operator(N)

        commutator = J_z @ J_x - J_x @ J_z
        expected = 1j * J_y
        assert commutator == pytest.approx(expected, abs=1e-10)


class TestPhysicalInvariants:
    @pytest.mark.parametrize(
        "N", [1, 2, 3, 4, 5], ids=["N=1", "N=2", "N=3", "N=4", "N=5"]
    )
    def test_jz_trace_is_zero(self, N: int) -> None:
        J_z = jz_operator(N)
        trace = np.trace(J_z)
        assert trace == pytest.approx(0, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 5], ids=["N=1", "N=2", "N=5"])
    def test_jx_is_real_valued(self, N: int) -> None:
        J_x = jx_operator(N)
        assert np.isrealobj(J_x)

    @pytest.mark.parametrize("N", [1, 2, 3, 5], ids=["N=1", "N=2", "N=3", "N=5"])
    def test_jy_is_hermitian(self, N: int) -> None:
        J_y = jy_operator(N)
        assert J_y == pytest.approx(J_y.T.conj())


class TestEdgeCases:
    def test_given_n0_then_jz_is_1x1_zero(self) -> None:
        J_z = jz_operator(0)
        assert J_z.shape == (1, 1)
        assert J_z[0, 0] == pytest.approx(0.0)

    def test_given_n1_then_operators_match_spin_half(self) -> None:
        J_z = jz_operator(1)
        J_x = jx_operator(1)

        assert np.allclose(J_z, [[0.5, 0], [0, -0.5]])
        assert np.allclose(J_x, [[0, 0.5], [0.5, 0]])

    def test_given_n2_then_operators_match_spin_one(self) -> None:
        J_z = jz_operator(2)
        J_x = jx_operator(2)

        assert np.allclose(J_z, [[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        sqrt2 = np.sqrt(2)
        expected_jx = [
            [0, sqrt2 / 2, 0],
            [sqrt2 / 2, 0, sqrt2 / 2],
            [0, sqrt2 / 2, 0],
        ]
        assert np.allclose(J_x, expected_jx)


class TestNumericalStability:
    def test_given_n100_then_operators_have_correct_shape(self) -> None:
        N = 100
        J_z = jz_operator(N)
        J_x = jx_operator(N)
        J_y = jy_operator(N)

        assert J_z.shape == (N + 1, N + 1)
        assert J_x.shape == (N + 1, N + 1)
        assert J_y.shape == (N + 1, N + 1)

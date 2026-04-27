"""Tests for dicke_basis.py - Dicke basis representation for N-atom systems."""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.dicke_basis import (
    basis_transformation_matrix,
    dicke_states,
    from_dicke_basis,
    j_raising_operator,
    j_squared_operator,
    j_lowering_operator,
    jx_operator,
    jy_operator,
    jz_eigenvalues,
    jz_operator,
    to_dicke_basis,
)


class TestDickeStates:
    """Test suite for dicke_states function."""

    def test_output_keys(self) -> None:
        """Test that dictionary keys are correct magnetic quantum numbers."""
        for N in [1, 2, 3, 4, 5, 10]:
            J = N / 2.0
            basis = dicke_states(N)
            expected_m_values = set(np.arange(J, -J - 1, -1))
            assert set(basis.keys()) == expected_m_values

    def test_output_length(self) -> None:
        """Test that output has N+1 entries."""
        for N in [0, 1, 2, 5, 10, 50]:
            basis = dicke_states(N)
            assert len(basis) == N + 1

    def test_index_mapping(self) -> None:
        """Test that indices increase correctly."""
        N = 4
        basis = dicke_states(N)
        # m=N/2 should have lowest index (0), m=-N/2 highest (N)
        J = N / 2.0
        assert basis[float(J)] == 0  # m=2.0 at idx 0
        assert basis[float(-J)] == N  # m=-2.0 at idx 4

    def test_negative_N_raises(self) -> None:
        """Test that negative N raises ValueError."""
        with pytest.raises(ValueError):
            dicke_states(-1)


class TestBasisConversions:
    """Test suite for to_dicke_basis and from_dicke_basis."""

    def test_roundtrip_vacuum(self) -> None:
        """Test conversion roundtrip for Dicke states with N=2."""
        N = 2
        # For Dicke states with total N=2, states are |n1,n2⟩ with n1+n2=2
        # n1=2, n2=0 → m=1; n1=1, n2=1 → m=0; n1=0, n2=2 → m=-1

        # |2,0⟩ → m = 1 → idx 0 in Dicke basis
        fock = np.zeros((N + 1) ** 2, dtype=complex)
        idx = 2 * (N + 1) + 0  # |2,0⟩ at index 6
        fock[idx] = 1.0

        dicke = to_dicke_basis(fock, N)
        fock_back = from_dicke_basis(dicke, N)

        assert np.allclose(fock, fock_back)

    def test_roundtrip_single_photon(self) -> None:
        """Test conversion roundtrip for Dicke states."""
        for N in [2, 3, 5]:
            # |N,0⟩ → m = N/2 (max m)
            fock = np.zeros((N + 1) ** 2, dtype=complex)
            idx = N * (N + 1)  # |N,0⟩ at index N*(N+1)
            fock[idx] = 1.0

            dicke = to_dicke_basis(fock, N)
            fock_back = from_dicke_basis(dicke, N)
            assert np.allclose(fock, fock_back)

            # |0,N⟩ → m = -N/2 (min m)
            fock = np.zeros((N + 1) ** 2, dtype=complex)
            fock[N] = 1.0  # |0,N⟩ at index N
            dicke = to_dicke_basis(fock, N)
            fock_back = from_dicke_basis(dicke, N)
            assert np.allclose(fock, fock_back)

            # |N/2,N/2⟩ → m = 0 (middle state, if N is even)
            if N % 2 == 0:
                half = N // 2
                fock = np.zeros((N + 1) ** 2, dtype=complex)
                idx = half * (N + 1) + half  # |N/2,N/2⟩
                fock[idx] = 1.0
                dicke = to_dicke_basis(fock, N)
                fock_back = from_dicke_basis(dicke, N)
                assert np.allclose(fock, fock_back)

    def test_roundtrip_arbitrary_state(self) -> None:
        """Test conversion roundtrip for arbitrary Dicke state."""
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

        assert np.allclose(fock, fock_back)

    def test_basis_dimension_N_plus_one(self) -> None:
        """Test that Dicke basis dimension is N+1."""
        for N in [1, 2, 3, 5, 10]:
            # Create Fock state in symmetric subspace
            fock = np.zeros((N + 1) ** 2, dtype=complex)
            fock[N * (N + 1)] = 1.0  # |N,0⟩
            dicke = to_dicke_basis(fock, N)
            assert len(dicke) == N + 1

    def test_unitarity_of_transformation(self) -> None:
        """Test that basis transformation is unitary."""
        for N in [1, 2, 3, 4, 5]:
            T = basis_transformation_matrix(N)
            assert T.shape == ((N + 1) ** 2, N + 1)
            # T^\dagger T should be identity in the symmetric subspace
            T_dag_T = T.conj().T @ T
            assert np.allclose(T_dag_T, np.eye(N + 1), atol=1e-10)

    def test_wrong_dimension_raises(self) -> None:
        """Test that wrong dimensions raise ValueError."""
        N = 2
        wrong_dim = np.zeros(5)  # Wrong dimension
        with pytest.raises(ValueError):
            to_dicke_basis(wrong_dim, N)

    def test_negative_N_raises(self) -> None:
        """Test that negative N raises ValueError."""
        fock = np.zeros(9, dtype=complex)
        fock[0] = 1.0
        with pytest.raises(ValueError):
            to_dicke_basis(fock, -1)

    def test_mapping_correctness(self) -> None:
        """Test that Fock to Dicke mapping is correct for specific states."""
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


class TestJzEigenvalues:
    """Test suite for jz_eigenvalues function."""

    def test_eigenvalue_range(self) -> None:
        """Test eigenvalues are in correct range."""
        for N in [1, 2, 3, 4, 5]:
            eigenvalues = jz_eigenvalues(N)
            assert eigenvalues[0] == N / 2.0  # Max m = N/2
            assert eigenvalues[-1] == -N / 2.0  # Min m = -N/2

    def test_step_is_one(self) -> None:
        """Test that eigenvalue spacing is 1."""
        for N in [1, 2, 5, 10]:
            eigenvalues = jz_eigenvalues(N)
            diffs = np.diff(eigenvalues)
            assert np.allclose(diffs, -1.0)  # Decreasing by 1

    def test_correct_count(self) -> None:
        """Test that there are N+1 eigenvalues."""
        for N in [0, 1, 5, 10]:
            eigenvalues = jz_eigenvalues(N)
            assert len(eigenvalues) == N + 1

    def test_sum_is_zero_for_even_N(self) -> None:
        """Test sum is zero for half-integer J."""
        for N in [2, 4, 6, 10]:
            eigenvalues = jz_eigenvalues(N)
            assert np.isclose(np.sum(eigenvalues), 0.0)


class TestJzOperator:
    """Test suite for jz_operator function."""

    def test_is_diagonal(self) -> None:
        """Test that J_z is diagonal."""
        for N in [1, 2, 3, 5, 10]:
            J_z = jz_operator(N)
            off_diagonal = J_z - np.diag(np.diag(J_z))
            assert np.allclose(off_diagonal, 0)

    def test_eigenvalues_match(self) -> None:
        """Test that diagonal matches jz_eigenvalues."""
        for N in [1, 2, 5, 10]:
            J_z = jz_operator(N)
            eigenvalues = jz_eigenvalues(N)
            assert np.allclose(J_z.diagonal(), eigenvalues)

    def test_hermitian(self) -> None:
        """Test that J_z is Hermitian."""
        for N in [1, 2, 5]:
            J_z = jz_operator(N)
            assert np.allclose(J_z, J_z.T.conj())

    def test_dimension(self) -> None:
        """Test that dimension is N+1."""
        for N in [1, 2, 5, 10]:
            J_z = jz_operator(N)
            assert J_z.shape == (N + 1, N + 1)


class TestJxOperator:
    """Test suite for jx_operator function."""

    def test_is_symmetric(self) -> None:
        """Test that J_x is symmetric (real)."""
        for N in [1, 2, 3, 5, 10]:
            J_x = jx_operator(N)
            assert np.allclose(J_x, J_x.T)

    def test_off_diagonal_only_adjacent(self) -> None:
        """Test J_x has non-zero only on super/sub-diagonals."""
        N = 5
        J_x = jx_operator(N)
        for i in range(N + 1):
            for j in range(N + 1):
                if abs(i - j) > 1:
                    assert np.isclose(J_x[i, j], 0)

    def test_correct_matrix_elements(self) -> None:
        """Test J_x matrix elements match analytical formula."""
        N = 4
        J = N / 2.0
        J_x = jx_operator(N)

        for i in range(N):
            m = J - i
            expected = 0.5 * np.sqrt((J + m) * (J - m + 1))
            assert np.isclose(J_x[i + 1, i], expected)
            assert np.isclose(J_x[i, i + 1], expected)

    def test_hermitian(self) -> None:
        """Test that J_x is Hermitian."""
        for N in [1, 2, 5]:
            J_x = jx_operator(N)
            assert np.allclose(J_x, J_x.conj().T)

    def test_dimension(self) -> None:
        """Test that dimension is N+1."""
        for N in [1, 2, 5, 10]:
            J_x = jx_operator(N)
            assert J_x.shape == (N + 1, N + 1)


class TestCommutationRelations:
    """Test SU(2) commutation relations [J_i, J_j] = i ε_ijk J_k."""

    def test_xy_commutator(self) -> None:
        """Test [J_x, J_y] = i J_z."""
        for N in [1, 2, 3, 5]:
            J_x = jx_operator(N)
            J_y = jy_operator(N)
            J_z = jz_operator(N)

            commutator = J_x @ J_y - J_y @ J_x
            expected = 1j * J_z
            assert np.allclose(commutator, expected, atol=1e-10)

    def test_yz_commutator(self) -> None:
        """Test [J_y, J_z] = i J_x."""
        for N in [1, 2, 3, 5]:
            J_x = jx_operator(N)
            J_y = jy_operator(N)
            J_z = jz_operator(N)

            commutator = J_y @ J_z - J_z @ J_y
            expected = 1j * J_x
            assert np.allclose(commutator, expected, atol=1e-10)

    def test_zx_commutator(self) -> None:
        """Test [J_z, J_x] = i J_y."""
        for N in [1, 2, 3, 5]:
            J_x = jx_operator(N)
            J_y = jy_operator(N)
            J_z = jz_operator(N)

            commutator = J_z @ J_x - J_x @ J_z
            expected = 1j * J_y
            assert np.allclose(commutator, expected, atol=1e-10)


class TestJ2Operator:
    """Test J²|m⟩ = J(J+1)|m⟩ relation."""

    def test_diagonal_with_correct_eigenvalue(self) -> None:
        """Test that J² is diagonal with eigenvalue J(J+1)."""
        for N in [1, 2, 3, 5, 10]:
            J2 = j_squared_operator(N)
            J = N / 2.0
            expected_eigenvalue = J * (J + 1)

            assert np.allclose(J2, expected_eigenvalue * np.eye(N + 1))

    def test_equality_to_sum_of_squares(self) -> None:
        """Test J² = J_x² + J_y² + J_z²."""
        for N in [1, 2, 3, 5]:
            J_x = jx_operator(N)
            J_y = jy_operator(N)
            J_z = jz_operator(N)
            J2_constructed = J_x @ J_x + J_y @ J_y + J_z @ J_z
            J2_direct = j_squared_operator(N)

            assert np.allclose(J2_constructed, J2_direct, atol=1e-10)


class TestLadderOperators:
    """Test raising and lowering operators."""

    def test_raising_raises_m(self) -> None:
        """Test J_+ increases m."""
        N = 4
        J_plus = j_raising_operator(N)
        # J_+ should be above diagonal (connecting to higher m)
        for i in range(N):
            assert not np.isclose(J_plus[i, i + 1], 0)

    def test_lowering_is_hermitian_conjugate(self) -> None:
        r"""Test J_- = (J_+)^\dagger."""
        for N in [1, 2, 3, 5]:
            J_plus = j_raising_operator(N)
            J_minus = j_lowering_operator(N)
            assert np.allclose(J_minus, J_plus.T.conj())

    def test_anticommutation_sum(self) -> None:
        """Test J_x = (J_+ + J_-) / 2."""
        for N in [1, 2, 3, 5]:
            J_plus = j_raising_operator(N)
            J_minus = j_lowering_operator(N)
            J_x_direct = jx_operator(N)
            J_x_from_ladders = (J_plus + J_minus) / 2
            assert np.allclose(J_x_direct, J_x_from_ladders, atol=1e-10)

    def test_ladder_consistency(self) -> None:
        """Test that J_+ and J_- have correct matrix elements."""
        N = 4
        J_plus = j_raising_operator(N)
        J_minus = j_lowering_operator(N)
        J = N / 2.0

        # Check that J_+ is above diagonal
        for i in range(1, N + 1):
            m = J - i  # m for state i
            expected_plus = np.sqrt((J - m) * (J + m + 1))
            # J_+[i-1, i] connects |m⟩ to |m+1⟩
            assert np.isclose(J_plus[i - 1, i], expected_plus)

        # Check that J_- is below diagonal (Hermitian conjugate)
        assert np.allclose(J_minus, J_plus.T.conj())


class TestPhysicalInvariants:
    """Test physical invariants that must hold."""

    def test_trace_of_jz_for_half_integer_spin(self) -> None:
        """Test Tr(J_z) = 0 for half-integer spin (any N)."""
        for N in [1, 2, 3, 4, 5]:
            J_z = jz_operator(N)
            trace = np.trace(J_z)
            assert np.isclose(trace, 0, atol=1e-10)

    def test_jx_is_real(self) -> None:
        """Test J_x has no imaginary part."""
        for N in [1, 2, 5]:
            J_x = jx_operator(N)
            assert np.isrealobj(J_x)

    def test_jy_is_hermitian(self) -> None:
        r"""Test J_y is Hermitian: J_y = J_y^\dagger."""
        for N in [1, 2, 3, 5]:
            J_y = jy_operator(N)
            # Check Hermitian: J_y = J_y^\dagger
            assert np.allclose(J_y, J_y.T.conj())


class TestEdgeCases:
    """Test edge cases and special values."""

    def test_N_equals_zero(self) -> None:
        """Test N=0 (single atom)."""
        J_z = jz_operator(0)
        assert J_z.shape == (1, 1)
        assert np.isclose(J_z[0, 0], 0.0)

    def test_N_equals_one(self) -> None:
        """Test N=1 (spin-1/2)."""
        J_z = jz_operator(1)
        J_x = jx_operator(1)

        # J_z = diag(0.5, -0.5)
        assert np.allclose(J_z, [[0.5, 0], [0, -0.5]])
        # J_x = [[0, 0.5], [0.5, 0]]
        assert np.allclose(J_x, [[0, 0.5], [0.5, 0]])

    def test_N_equals_two(self) -> None:
        """Test N=2 (spin-1)."""
        J_z = jz_operator(2)
        J_x = jx_operator(2)

        # J_z = diag(1, 0, -1)
        assert np.allclose(J_z, [[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        # J_x[1,0] = J_x[0,1] = J_x[2,1] = J_x[1,2] = 1/sqrt(2)
        sqrt2 = np.sqrt(2)
        expected_jx = [[0, sqrt2 / 2, 0], [sqrt2 / 2, 0, sqrt2 / 2], [0, sqrt2 / 2, 0]]
        assert np.allclose(J_x, expected_jx)


class TestNumericalStability:
    """Test numerical stability for larger N."""

    def test_large_N_dimensions(self) -> None:
        """Test that operations complete for N=100."""
        N = 100
        J_z = jz_operator(N)
        J_x = jx_operator(N)
        J_y = jy_operator(N)
        J2 = j_squared_operator(N)

        assert J_z.shape == (N + 1, N + 1)
        assert J_x.shape == (N + 1, N + 1)
        assert J_y.shape == (N + 1, N + 1)
        assert J2.shape == (N + 1, N + 1)

    def test_j2_from_sum_stability(self) -> None:
        """Test J² = Jx² + Jy² + Jz² for larger N."""
        N = 50
        J_x = jx_operator(N)
        J_y = jy_operator(N)
        J_z = jz_operator(N)
        J2 = j_squared_operator(N)
        J2_constructed = J_x @ J_x + J_y @ J_y + J_z @ J_z

        assert np.allclose(J2, J2_constructed, atol=1e-8)

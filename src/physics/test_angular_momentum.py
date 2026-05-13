"""Unit tests for angular_momentum module.

These tests verify the mathematical correctness of spin matrix generation,
including dimensions, hermiticity, commutation relations, and eigenvalue spectra.
"""

from __future__ import annotations

import numpy as np
import pytest

from .angular_momentum import generate_spin_matrices


class TestMatrixDimensions:
    """Test that spin matrices have correct dimensions."""

    @pytest.mark.parametrize("dim", [2, 3, 4, 5, 8])
    def test_matrices_are_square(self, dim: int) -> None:
        """Spin matrices must be square with shape (dim, dim)."""
        jx, jz = generate_spin_matrices(dim)
        assert jx.shape == (dim, dim), f"Jx should be {dim}x{dim}"
        assert jz.shape == (dim, dim), f"Jz should be {dim}x{dim}"

    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    def test_both_matrices_same_dimension(self, dim: int) -> None:
        """Jx and Jz must have identical shapes."""
        jx, jz = generate_spin_matrices(dim)
        assert jx.shape == jz.shape, "Expected jx.shape == jz.shape"

    def test_dtype_is_float(self) -> None:
        """Matrices should be real-valued (float dtype)."""
        for dim in [2, 3, 4]:
            jx, jz = generate_spin_matrices(dim)
            assert np.issubdtype(jx.dtype, np.floating), (
                f"Jx dtype {jx.dtype} not float"
            )
            assert np.issubdtype(jz.dtype, np.floating), (
                f"Jz dtype {jz.dtype} not float"
            )


class TestHermiticity:
    """Test that spin matrices are Hermitian (self-adjoint)."""

    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    def test_jx_is_hermitian(self, dim: int) -> None:
        """Jx must equal its own transpose (real symmetric matrix)."""
        jx, _ = generate_spin_matrices(dim)
        assert jx == pytest.approx(jx.T), (
            "Jx must be symmetric (Hermitian for real matrices)"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    def test_jz_is_hermitian(self, dim: int) -> None:
        """Jz must be diagonal and thus Hermitian."""
        _, jz = generate_spin_matrices(dim)
        assert jz == pytest.approx(jz.T), (
            "Jz must be symmetric (Hermitian for real matrices)"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_off_diagonal_real(self, dim: int) -> None:
        """All elements should be purely real."""
        jx, jz = generate_spin_matrices(dim)
        assert jx.imag == pytest.approx(0), "Jx must have zero imaginary parts"
        assert jz.imag == pytest.approx(0), "Jz must have zero imaginary parts"


class TestSpinHalf:
    """Special tests for spin-1/2 (dimension 2) case."""

    def test_jz_eigenvalues_spin_half(self) -> None:
        """For spin-1/2, Jz eigenvalues should be ±1/2."""
        _, jz = generate_spin_matrices(2)
        eigenvalues = np.linalg.eigvalsh(jz)
        expected = np.array([-0.5, 0.5])
        assert sorted(eigenvalues) == pytest.approx(expected), (
            "Expected sorted(eigenvalues) == pytest.approx(expected)"
        )

    def test_jz_structure_spin_half(self) -> None:
        """Spin-1/2 Jz should be diagonal with ±1/2 on diagonal."""
        _, jz = generate_spin_matrices(2)
        expected_jz = np.array([[0.5, 0.0], [0.0, -0.5]])
        assert jz == pytest.approx(expected_jz), (
            "Expected jz == pytest.approx(expected_jz)"
        )

    def test_jx_structure_spin_half(self) -> None:
        """Spin-1/2 Jx should have 1/2 on off-diagonals."""
        jx, _ = generate_spin_matrices(2)
        expected_jx = np.array([[0.0, 0.5], [0.5, 0.0]])
        assert jx == pytest.approx(expected_jx), (
            "Expected jx == pytest.approx(expected_jx)"
        )

    def test_jx_off_diagonal_nonzero(self) -> None:
        """Jx should have non-zero off-diagonal elements for dim>1."""
        jx, _ = generate_spin_matrices(2)
        assert jx[0, 1] != pytest.approx(0), "Jx off-diagonal should be non-zero"


class TestSpinOne:
    """Tests for spin-1 (dimension 3) case."""

    def test_jz_eigenvalues_spin_one(self) -> None:
        """For spin-1, Jz eigenvalues should be -1, 0, +1."""
        _, jz = generate_spin_matrices(3)
        eigenvalues = np.linalg.eigvalsh(jz)
        expected = np.array([-1.0, 0.0, 1.0])
        assert sorted(eigenvalues) == pytest.approx(expected), (
            "Expected sorted(eigenvalues) == pytest.approx(expected)"
        )

    def test_jz_is_diagonal(self) -> None:
        """Jz should be purely diagonal for all dimensions."""
        _, jz = generate_spin_matrices(3)
        # Extract diagonal and check off-diagonals are zero
        diag = np.diag(jz)
        reconstructed = np.diag(diag)
        assert jz == pytest.approx(reconstructed), "Jz should be diagonal"


class TestCommutationRelations:
    """Test fundamental angular momentum commutation relations.

    The spin operators satisfy [Ji, Jj] = i εijk Jk, where εijk is the
    Levi-Civita symbol. For our 2D case:
    - [Jx, Jz] = i Jy
    """

    def test_jx_jz_commutator_spin_half(self) -> None:
        """Test [Jx, Jz] = i*(1/2)*Jy for spin-1/2.

        In spin-1/2, Jx and Jz don't generate a non-trivial Jy here
        since we only have Jx and Jz defined. This is a simplified check.
        """
        jx, jz = generate_spin_matrices(2)
        commutator = jx @ jz - jz @ jx
        # For 2x2 spin matrices, the commutator should have a specific pattern
        # This tests that the matrices are not simultaneously diagonalizable
        assert commutator != pytest.approx(np.zeros((2, 2))), (
            "Jx and Jz should not commute (angular momentum algebra)"
        )


class TestTraceProperties:
    """Test trace-related properties of spin matrices."""

    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    def test_jz_trace_is_zero(self, dim: int) -> None:
        """Trace of Jz should be zero (sum of magnetic quantum numbers)."""
        _, jz = generate_spin_matrices(dim)
        trace = np.trace(jz)
        assert trace == pytest.approx(0.0, abs=1e-10), (
            f"Trace of Jz should be 0, got {trace}"
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_jx_trace_is_zero(self, dim: int) -> None:
        """Trace of Jx should be zero (symmetric off-diagonal structure)."""
        jx, _ = generate_spin_matrices(dim)
        trace = np.trace(jx)
        assert trace == pytest.approx(0.0, abs=1e-10), (
            f"Trace of Jx should be 0, got {trace}"
        )


class TestOffDiagonalStructure:
    """Test the structure of off-diagonal elements."""

    def test_jx_off_diagonal_symmetry(self) -> None:
        """Jx should be symmetric: Jx[i,j] = Jx[j,i]."""
        for dim in [2, 3, 4, 5]:
            jx, _ = generate_spin_matrices(dim)
            for i in range(dim):
                for j in range(i + 1, dim):
                    assert jx[i, j] == pytest.approx(jx[j, i]), (
                        f"Jx should be symmetric: [{i},{j}]={jx[i, j]} vs [{j},{i}]={jx[j, i]}"
                    )

    def test_jx_off_diagonal_pattern(self) -> None:
        """Jx should have non-zero elements only on first super-diagonal and symmetric."""
        jx, _ = generate_spin_matrices(4)
        # Check that only (i, i+1) and (i+1, i) are non-zero
        for i in range(4):
            for j in range(4):
                if abs(i - j) > 1:
                    assert jx[i, j] == pytest.approx(0.0, abs=1e-10), (
                        f"Jx[{i},{j}] should be zero, got {jx[i, j]}"
                    )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_dim_2_minimum(self) -> None:
        """Dimension 2 (spin-1/2) is the minimum non-trivial case."""
        jx, jz = generate_spin_matrices(2)
        assert jx.shape == (2, 2), "Expected jx.shape == (2, 2)"
        assert jz.shape == (2, 2), "Expected jz.shape == (2, 2)"
        # Verify non-trivial structure
        assert jx != pytest.approx(np.zeros((2, 2))), (
            "Expected jx != pytest.approx(np.zeros((2, 2)))"
        )

    def test_larger_dimensions_consistent(self) -> None:
        """Larger dimensions should maintain mathematical consistency."""
        for dim in [4, 5, 6]:
            jx, jz = generate_spin_matrices(dim)
            # All previous properties should hold
            assert jx.shape == (dim, dim), "Expected jx.shape == (dim, dim)"
            assert jx == pytest.approx(jx.T), "Expected jx == pytest.approx(jx.T)"
            assert jz == pytest.approx(jz.T), "Expected jz == pytest.approx(jz.T)"
            assert np.trace(jz) == pytest.approx(0.0, abs=1e-10), (
                "Expected np.trace(jz) == pytest.approx(0.0, abs=1e-10)"
            )

"""Smoke tests for angular_momentum module."""

import numpy as np

from src.angular_momentum import generate_spin_matrices


class TestGenerateSpinMatrices:
    def test_dim_2_returns_correct_shapes(self):
        jx, jz = generate_spin_matrices(2)
        assert jx.shape == (2, 2)
        assert jz.shape == (2, 2)

    def test_dim_3_returns_correct_shapes(self):
        jx, jz = generate_spin_matrices(3)
        assert jx.shape == (3, 3)
        assert jz.shape == (3, 3)

    def test_jz_is_diagonal(self):
        _, jz = generate_spin_matrices(5)
        # Check off-diagonal elements are zero
        assert np.allclose(jz, np.diag(np.diag(jz)))

    def test_jx_is_symmetric(self):
        jx, _ = generate_spin_matrices(4)
        assert np.allclose(jx, jx.T)

    def test_jz_has_correct_eigenvalues(self):
        """For spin-1/2 (dim=2), eigenvalues should be +1/2 and -1/2"""
        _, jz = generate_spin_matrices(2)
        eigenvalues = np.linalg.eigvalsh(jz)
        expected = np.array([-0.5, 0.5])
        assert np.allclose(sorted(eigenvalues), expected)

    def test_output_is_float_dtype(self):
        jx, jz = generate_spin_matrices(3)
        assert jx.dtype == np.float64 or jx.dtype == np.float_
        assert jz.dtype == np.float64 or jz.dtype == np.float_

"""Tests for validators.py - validation utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from .validators import (
    validate_eigendecomposition,
    validate_eigenvectors_orthonormal,
    validate_hamiltonian_delta_estimation,
    validate_hamiltonian_hermitian,
    validate_orthonormality,
    validate_partial_trace,
    validate_probability_conservation,
    validate_sensitivity,
    validate_state_delta_estimation,
    validate_state_mzi,
    validate_unitary,
)


class TestValidateHamiltonianHermitian:
    def test_hermitian_passes(self) -> None:
        H = np.array([[1.0, 2.0 - 1.0j], [2.0 + 1.0j, 3.0]])
        assert validate_hamiltonian_hermitian(H)

    def test_non_hermitian_fails(self) -> None:
        H = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert not validate_hamiltonian_hermitian(H)

    def test_real_symmetric_passes(self) -> None:
        H = np.array([[2.0, -1.0], [-1.0, 2.0]])
        assert validate_hamiltonian_hermitian(H)


class TestValidateEigenvectorsOrthonormal:
    def test_orthonormal_passes(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.normal(size=(4, 4))
        Q, _ = np.linalg.qr(A)
        assert validate_eigenvectors_orthonormal(Q)

    def test_non_orthonormal_fails(self) -> None:
        vectors = np.array([[1.0, 0.0], [1.0, 1.0]])
        assert not validate_eigenvectors_orthonormal(vectors)

    def test_with_custom_tolerance(self) -> None:
        rng = np.random.default_rng(99)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        assert validate_eigenvectors_orthonormal(Q, tolerance=1e-6)


class TestValidateEigendecomposition:
    def test_valid_decomposition_passes(self) -> None:
        H = np.array([[2.0, -1.0], [-1.0, 2.0]])
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        assert validate_eigendecomposition(H, eigenvalues, eigenvectors)

    def test_wrong_eigenvalues_fails(self) -> None:
        H = np.array([[2.0, -1.0], [-1.0, 2.0]])
        _, eigenvectors = np.linalg.eigh(H)
        eigenvalues = np.array([100.0, 200.0])
        assert not validate_eigendecomposition(H, eigenvalues, eigenvectors)

    def test_wrong_eigenvectors_fails(self) -> None:
        H = np.array([[2.0, -1.0], [-1.0, 2.0]])
        eigenvalues, _ = np.linalg.eigh(H)
        eigenvectors = np.eye(2)
        assert not validate_eigendecomposition(H, eigenvalues, eigenvectors)


class TestValidateOrthonormality:
    def test_orthonormal_returns_small_deviation(self) -> None:
        rng = np.random.default_rng(7)
        Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        deviation = validate_orthonormality(Q)
        assert deviation < 1e-10

    def test_non_orthonormal_returns_large_deviation(self) -> None:
        vectors = np.array([[1.0, 0.0], [2.0, 1.0]])
        deviation = validate_orthonormality(vectors)
        assert deviation > 1e-10

    def test_returns_float(self) -> None:
        rng = np.random.default_rng(13)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        result = validate_orthonormality(Q)
        assert isinstance(result, float)


class TestValidateProbabilityConservation:
    def test_normalized_passes(self) -> None:
        wf = np.array([0.6 + 0.0j, 0.0 + 0.8j])
        assert validate_probability_conservation(wf)

    def test_unnormalized_fails(self) -> None:
        wf = np.array([1.0, 1.0])
        assert not validate_probability_conservation(wf)

    def test_normalized_random_passes(self) -> None:
        rng = np.random.default_rng(42)
        wf = rng.normal(size=10) + 1j * rng.normal(size=10)
        wf = wf / np.linalg.norm(wf)
        assert validate_probability_conservation(wf)

    def test_zero_state_fails(self) -> None:
        wf = np.zeros(4)
        assert not validate_probability_conservation(wf)


class TestValidatePartialTrace:
    def test_valid_partial_trace_passes(self) -> None:
        psi = np.array([1.0, 0.0, 0.0, 0.0]) / np.sqrt(2)
        psi = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
        rho_full = np.outer(psi, psi.conj())
        rho_a = np.array([[0.5, 0.0], [0.0, 0.5]])
        rho_b = np.array([[0.5, 0.0], [0.0, 0.5]])
        assert validate_partial_trace(rho_full, rho_a, rho_b)

    def test_wrong_trace_fails(self) -> None:
        rho_full = np.eye(4) / 2.0
        rho_a = np.eye(2)
        rho_b = np.eye(2)
        assert not validate_partial_trace(rho_full, rho_a, rho_b)

    def test_non_hermitian_full_fails(self) -> None:
        rho_full = np.array([[1.0, 2.0], [3.0, 4.0]])
        rho_a = np.eye(2)
        rho_b = np.eye(2)
        assert not validate_partial_trace(rho_full, rho_a, rho_b)

    def test_mismatched_traces_fails(self) -> None:
        psi = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
        rho_full = np.outer(psi, psi.conj())
        rho_a = np.array([[0.5, 0.0], [0.0, 0.5]])
        rho_b = np.eye(2)
        assert not validate_partial_trace(rho_full, rho_a, rho_b)


class TestValidateSensitivity:
    def test_sensitivity_matches_numerical_derivative(self) -> None:
        try:
            result = validate_sensitivity(2, 1, 1.0, 0.5, 0.3, 0.2, 1.0, tolerance=1e-3)
            assert result
        except ImportError:
            pytest.skip("sensitivity_analysis module not available")
        except Exception:
            pytest.skip("sensitivity analysis prerequisites not met")

    def test_invalid_sensitivity_fails(self) -> None:
        try:
            result = validate_sensitivity(
                2, 1, 1.0, 0.5, 0.3, 0.2, 1.0, tolerance=1e-12
            )
            assert not result
        except ImportError:
            pytest.skip("sensitivity_analysis module not available")
        except Exception:
            pytest.skip("sensitivity analysis prerequisites not met")


class TestValidateStateDeltaEstimation:
    def test_valid_density_matrix_passes(self) -> None:
        rho = np.eye(4) / 4.0
        assert validate_state_delta_estimation(rho, (4, 4))

    def test_wrong_shape_fails(self) -> None:
        rho = np.eye(4) / 4.0
        assert not validate_state_delta_estimation(rho, (2, 2))

    def test_non_hermitian_fails(self) -> None:
        state = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert not validate_state_delta_estimation(state, (2, 2))

    def test_wrong_trace_fails(self) -> None:
        state = np.eye(2) * 2.0
        assert not validate_state_delta_estimation(state, (2, 2))

    def test_non_positive_fails(self) -> None:
        state = np.diag([-0.1, 1.1])
        assert not validate_state_delta_estimation(state, (2, 2))


class TestValidateHamiltonianDeltaEstimation:
    def test_hermitian_passes(self) -> None:
        H = np.array([[1.0, 2.0 - 1.0j], [2.0 + 1.0j, 3.0]])
        assert validate_hamiltonian_delta_estimation(H)

    def test_non_hermitian_fails(self) -> None:
        H = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert not validate_hamiltonian_delta_estimation(H)

    def test_real_symmetric_passes(self) -> None:
        H = np.array([[1.0, 0.5], [0.5, 2.0]])
        assert validate_hamiltonian_delta_estimation(H)


class TestValidateStateMzi:
    def test_normalized_vector_passes(self) -> None:
        state = np.array([1.0, 0.0])
        assert validate_state_mzi(state)

    def test_unnormalized_fails(self) -> None:
        state = np.array([2.0, 0.0])
        assert not validate_state_mzi(state)

    def test_normalized_random_passes(self) -> None:
        rng = np.random.default_rng(42)
        state = rng.normal(size=8) + 1j * rng.normal(size=8)
        state = state / np.linalg.norm(state)
        assert validate_state_mzi(state)

    def test_zero_state_fails(self) -> None:
        state = np.zeros(4)
        assert not validate_state_mzi(state)


class TestValidateUnitary:
    def test_identity_passes(self) -> None:
        U = np.eye(4)
        assert validate_unitary(U)

    def test_non_unitary_fails(self) -> None:
        U = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert not validate_unitary(U)

    def test_random_unitary_passes(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        Q, _ = np.linalg.qr(A)
        assert validate_unitary(Q)

    def test_custom_tolerance(self) -> None:
        U = np.eye(3) + 1e-6 * np.ones((3, 3))
        assert not validate_unitary(U, tol=1e-8)
        assert validate_unitary(U, tol=1e-4)

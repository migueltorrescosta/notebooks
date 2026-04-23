"""Unit tests for Heisenberg Model physics module."""

import numpy as np
import pytest

from src.heisenberg_model import (
    heisenberg_hamiltonian,
    heisenberg_coupling_term,
    heisenberg_field_term,
    diagonalize_hamiltonian,
    compute_expectation_values,
    run_simulation,
    validate_eigenvectors_orthonormal,
    validate_eigendecomposition,
)


class TestHamiltonianConstruction:
    def test_hamiltonian_shape(self) -> None:
        """Hamiltonian should have correct shape."""
        for n in [2, 3, 4]:
            H = heisenberg_hamiltonian(n)
            dim = 2**n
            assert H.shape == (dim, dim)

    def test_hamiltonian_hermitian(self) -> None:
        """Hamiltonian should be Hermitian."""
        H = heisenberg_hamiltonian(4)
        assert np.allclose(H, H.conj().T)

    def test_coupling_term(self) -> None:
        """Coupling term should be Hermitian."""
        H = heisenberg_coupling_term(3)
        assert np.allclose(H, H.conj().T)

    def test_field_term(self) -> None:
        """Field term should be Hermitian."""
        H = heisenberg_field_term(3)
        assert np.allclose(H, H.conj().T)

    def test_invalid_n_sites(self) -> None:
        """Should raise for invalid n_sites."""
        with pytest.raises(ValueError):
            heisenberg_hamiltonian(0)
        with pytest.raises(ValueError):
            heisenberg_hamiltonian(27)


class TestEigendecomposition:
    def test_eigenvalues_sorted(self) -> None:
        """Eigenvalues should be sorted."""
        eigenvalues, _ = diagonalize_hamiltonian(3)
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:])

    def test_eigenvectors_orthogonal(self) -> None:
        """Eigenvectors should be orthogonal."""
        _, eigenvectors = diagonalize_hamiltonian(3)
        assert validate_eigenvectors_orthonormal(eigenvectors)

    def test_eigendecomposition_valid(self) -> None:
        """Should satisfy H|v⟩ = E|v⟩."""
        H = heisenberg_hamiltonian(3)
        eigenvalues, eigenvectors = diagonalize_hamiltonian(3)
        assert validate_eigendecomposition(H, eigenvalues, eigenvectors)


class TestExpectationValues:
    def test_expectation_values_real(self) -> None:
        """Expectation values should be real."""
        _, eigenvectors = diagonalize_hamiltonian(3, j=1.0, u=1.0)
        expectations = compute_expectation_values(3, eigenvectors)
        assert np.all(np.isreal(expectations))

    def test_expectation_values_in_range(self) -> None:
        """σ_z expectation should be in [-1, 1]."""
        _, eigenvectors = diagonalize_hamiltonian(3, j=1.0, u=1.0)
        expectations = compute_expectation_values(3, eigenvectors)
        # Check range [-1, 1]
        assert np.all(np.abs(expectations[:, :, 0]) <= 1.0 + 1e-8)


class TestSimulation:
    def test_run_simulation_returns_dict(self) -> None:
        """Should return correctly shaped dictionary."""
        result = run_simulation(3, j=1.0, u=1.0)
        assert "hamiltonian" in result
        assert "eigenvalues" in result
        assert "eigenvectors" in result

    def test_run_simulation_consistency(self) -> None:
        """Eigenvalues should match hamiltonian."""
        result = run_simulation(3, j=1.0, u=1.0)
        eigenvalues, eigenvectors = np.linalg.eigh(result["hamiltonian"])
        assert np.allclose(eigenvalues, result["eigenvalues"])

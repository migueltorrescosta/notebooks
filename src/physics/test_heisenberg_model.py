"""Tests for Heisenberg model quantum spin chain (src.physics.heisenberg_model).

Tests validate Hamiltonian construction, eigendecomposition,
expectation values, physical constraints, and Hilbert-space scaling.
"""

from __future__ import annotations

import numpy as np
import pytest

from .heisenberg_model import (
    compute_expectation_values,
    diagonalize_hamiltonian,
    heisenberg_coupling_term,
    heisenberg_field_term,
    heisenberg_hamiltonian,
    run_simulation,
    validate_eigendecomposition,
    validate_eigenvectors_orthonormal,
)


# =============================================================================
# Hamiltonian construction
# =============================================================================


class TestHamiltonianConstruction:
    """Tests for Hamiltonian dimension, Hermiticity, and parameter sensitivity."""

    def test_hamiltonian_shape(self) -> None:
        """Hamiltonian should have dimension 2^n x 2^n."""
        for n in [2, 3, 4]:
            H = heisenberg_hamiltonian(n)
            dim = 2**n
            assert H.shape == (dim, dim)

    def test_hamiltonian_hermitian(self) -> None:
        """Hamiltonian should be Hermitian."""
        H = heisenberg_hamiltonian(4)
        assert np.allclose(H, H.conj().T)

    def test_coupling_term_hermitian(self) -> None:
        """Coupling term should be Hermitian."""
        H = heisenberg_coupling_term(3)
        assert np.allclose(H, H.conj().T)

    def test_field_term_hermitian(self) -> None:
        """Field term should be Hermitian."""
        H = heisenberg_field_term(3)
        assert np.allclose(H, H.conj().T)

    def test_invalid_n_sites_raises(self) -> None:
        """Should raise ValueError for invalid n_sites."""
        with pytest.raises(ValueError):
            heisenberg_hamiltonian(0)
        with pytest.raises(ValueError):
            heisenberg_hamiltonian(27)

    def test_varying_j_yields_different_hamiltonians(self) -> None:
        """Different J values should yield different Hamiltonians."""
        H_pos = heisenberg_hamiltonian(n_sites=3, j=1.0)
        H_neg = heisenberg_hamiltonian(n_sites=3, j=-1.0)
        assert not np.allclose(H_pos, H_neg, atol=1e-10)

    def test_varying_u_yields_different_hamiltonians(self) -> None:
        """Different U values should yield different Hamiltonians."""
        H_low = heisenberg_hamiltonian(n_sites=3, u=0.1)
        H_high = heisenberg_hamiltonian(n_sites=3, u=10.0)
        assert not np.allclose(H_low, H_high, atol=1e-10)

    def test_hamiltonian_has_nonzero_structure(self) -> None:
        """Hamiltonian should have a non-trivial fraction of non-zero elements."""
        H = heisenberg_hamiltonian(n_sites=3)
        non_zero_count = np.count_nonzero(H)
        total_elements = H.shape[0] ** 2
        assert non_zero_count > total_elements * 0.1, (
            f"Only {non_zero_count}/{total_elements} non-zero elements"
        )


# =============================================================================
# Eigendecomposition
# =============================================================================


class TestEigendecomposition:
    """Tests for eigenvalues and eigenvectors of the Hamiltonian."""

    def test_eigenvalues_sorted(self) -> None:
        """Eigenvalues should be returned in ascending order."""
        eigenvalues, _ = diagonalize_hamiltonian(3)
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:])

    def test_eigenvectors_orthonormal(self) -> None:
        """Eigenvectors should form an orthonormal basis."""
        _, eigenvectors = diagonalize_hamiltonian(3)
        assert validate_eigenvectors_orthonormal(eigenvectors)

    def test_eigendecomposition_valid(self) -> None:
        """Should satisfy H|v⟩ = E|v⟩ for all eigenpairs."""
        H = heisenberg_hamiltonian(3)
        eigenvalues, eigenvectors = diagonalize_hamiltonian(3)
        assert validate_eigendecomposition(H, eigenvalues, eigenvectors)

    def test_spectral_reconstruction(self) -> None:
        """H should be exactly recoverable from its spectral decomposition."""
        for n_sites in [2, 3, 4]:
            H = heisenberg_hamiltonian(n_sites)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
            assert np.allclose(H, reconstructed, atol=1e-10)

    def test_hilbert_space_grows_exponentially(self) -> None:
        """Hilbert-space dimension should double with each added site."""
        dimensions = []
        for n_sites in [2, 3, 4, 5]:
            H = heisenberg_hamiltonian(n_sites)
            dimensions.append(H.shape[0])

        for i in range(1, len(dimensions)):
            assert dimensions[i] == 2 * dimensions[i - 1], (
                f"Dimension should double: {dimensions[i]} vs {dimensions[i - 1]}"
            )


# =============================================================================
# Expectation values
# =============================================================================


class TestExpectationValues:
    """Tests for expectation value calculations on eigenstates."""

    def test_expectation_values_real(self) -> None:
        """Expectation values should be real."""
        _, eigenvectors = diagonalize_hamiltonian(3, j=1.0, u=1.0)
        expectations = compute_expectation_values(3, eigenvectors)
        assert np.all(np.isreal(expectations))

    def test_expectation_values_in_range(self) -> None:
        """σ_z expectation values should be in [-1, 1]."""
        _, eigenvectors = diagonalize_hamiltonian(3, j=1.0, u=1.0)
        expectations = compute_expectation_values(3, eigenvectors)
        assert np.all(np.abs(expectations[:, :, 0]) <= 1.0 + 1e-8)

    def test_sum_rule_for_expectation_values(self) -> None:
        """Sum of σ_z expectation values should be physically bounded."""
        _, eigenvectors = diagonalize_hamiltonian(3, j=1.0, u=1.0)
        expectations = compute_expectation_values(3, eigenvectors)
        n_sites = 3
        for state in range(expectations.shape[1]):
            sum_sz = sum(
                expectations[site, state, 0]
                if expectations[site, state, 0] != 0
                else -expectations[site, state, 1]
                for site in range(n_sites)
            )
            assert -n_sites <= sum_sz <= n_sites


# =============================================================================
# Physical constraints
# =============================================================================


class TestPhysicalConstraints:
    """Tests for energy bounds and Hilbert-space structure."""

    def test_energy_levels_are_bounded(self) -> None:
        """Energy levels should be within reasonable bounds for a spin-1/2 chain."""
        for n_sites in [2, 3, 4]:
            H = heisenberg_hamiltonian(n_sites)
            eigenvalues = np.linalg.eigvalsh(H)

            max_energy = 2 * n_sites
            min_energy = -2 * n_sites

            assert np.all(eigenvalues >= min_energy - 1e-10), (
                f"Minimum energy too low: {min(eigenvalues)}"
            )
            assert np.all(eigenvalues <= max_energy + 1e-10), (
                f"Maximum energy too high: {max(eigenvalues)}"
            )


# =============================================================================
# Simulation runner
# =============================================================================


class TestSimulationRunner:
    """Tests for the run_simulation function."""

    def test_run_simulation_returns_dict(self) -> None:
        """Should return correctly shaped dictionary."""
        result = run_simulation(3, j=1.0, u=1.0)
        assert "hamiltonian" in result
        assert "eigenvalues" in result
        assert "eigenvectors" in result

    def test_run_simulation_consistency(self) -> None:
        """Eigenvalues should match hamiltonian."""
        result = run_simulation(3, j=1.0, u=1.0)
        eigenvalues, _ = np.linalg.eigh(result["hamiltonian"])
        assert np.allclose(eigenvalues, result["eigenvalues"])

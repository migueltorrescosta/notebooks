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

    def test_hamiltonian_should_have_dimension_2_n_x_2_n(self) -> None:
        for n in [2, 3, 4]:
            H = heisenberg_hamiltonian(n)
            dim = 2**n
            assert H.shape == (dim, dim), "Expected H.shape == (dim, dim)"

    def test_hamiltonian_should_be_hermitian(self) -> None:
        H = heisenberg_hamiltonian(4)
        assert pytest.approx(H.conj().T) == H, "Expected H == pytest.approx(H.conj().T)"

    def test_coupling_term_should_be_hermitian(self) -> None:
        H = heisenberg_coupling_term(3)
        assert pytest.approx(H.conj().T) == H, "Expected H == pytest.approx(H.conj().T)"

    def test_field_term_should_be_hermitian(self) -> None:
        H = heisenberg_field_term(3)
        assert pytest.approx(H.conj().T) == H, "Expected H == pytest.approx(H.conj().T)"

    def test_should_raise_valueerror_for_invalid_n_sites(self) -> None:
        with pytest.raises(ValueError):
            heisenberg_hamiltonian(0)
        with pytest.raises(ValueError):
            heisenberg_hamiltonian(27)

    def test_different_j_values_should_yield_different_hamiltonians(self) -> None:
        H_pos = heisenberg_hamiltonian(n_sites=3, j=1.0)
        H_neg = heisenberg_hamiltonian(n_sites=3, j=-1.0)
        assert H_pos != pytest.approx(H_neg, abs=1e-10), (
            "Expected H_pos != pytest.approx(H_neg, abs=1e-10)"
        )

    def test_different_u_values_should_yield_different_hamiltonians(self) -> None:
        H_low = heisenberg_hamiltonian(n_sites=3, u=0.1)
        H_high = heisenberg_hamiltonian(n_sites=3, u=10.0)
        assert H_low != pytest.approx(H_high, abs=1e-10), (
            "Expected H_low != pytest.approx(H_high, abs=1e-10)"
        )

    def test_hamiltonian_should_have_a_non_trivial_fraction_of_non_zero_elements(
        self,
    ) -> None:
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

    def test_eigenvalues_should_be_returned_in_ascending_order(self) -> None:
        eigenvalues, _ = diagonalize_hamiltonian(3)
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:]), (
            "Expected np.all(eigenvalues[:-1] <= eigenvalues[1:])"
        )

    def test_eigenvectors_should_form_an_orthonormal_basis(self) -> None:
        _, eigenvectors = diagonalize_hamiltonian(3)
        assert validate_eigenvectors_orthonormal(eigenvectors), (
            "Condition failed: validate_eigenvectors_orthonormal(eigenvectors)"
        )

    def test_should_satisfy_h_v_e_v_for_all_eigenpairs(self) -> None:
        H = heisenberg_hamiltonian(3)
        eigenvalues, eigenvectors = diagonalize_hamiltonian(3)
        assert validate_eigendecomposition(H, eigenvalues, eigenvectors), (
            "Condition failed: validate_eigendecomposition(H, eigenvalues, eigenvectors)"
        )

    def test_h_should_be_exactly_recoverable_from_its_spectral_decomposition(
        self,
    ) -> None:
        for n_sites in [2, 3, 4]:
            H = heisenberg_hamiltonian(n_sites)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
            assert pytest.approx(reconstructed, abs=1e-10) == H, (
                "Expected H == pytest.approx(reconstructed, abs=1e-10)"
            )

    def test_hilbert_space_dimension_should_double_with_each_added_site(self) -> None:
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

    def test_expectation_values_should_be_real(self) -> None:
        _, eigenvectors = diagonalize_hamiltonian(3, j=1.0, u=1.0)
        expectations = compute_expectation_values(3, eigenvectors)
        assert np.all(np.isreal(expectations)), (
            "All values should satisfy np.isreal(expectations)"
        )

    def test_z_expectation_values_should_be_in_1_1(self) -> None:
        _, eigenvectors = diagonalize_hamiltonian(3, j=1.0, u=1.0)
        expectations = compute_expectation_values(3, eigenvectors)
        assert np.all(np.abs(expectations[:, :, 0]) <= 1.0 + 1e-8), (
            "Expected np.all(np.abs(expectations[:, :, 0]) <= 1.0 + 1e-8)"
        )

    def test_sum_of_z_expectation_values_should_be_physically_bounded(self) -> None:
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
            assert -n_sites <= sum_sz <= n_sites, (
                "Expected -n_sites <= sum_sz <= n_sites"
            )


# =============================================================================
# Physical constraints
# =============================================================================


class TestPhysicalConstraints:
    """Tests for energy bounds and Hilbert-space structure."""

    def test_energy_levels_should_be_within_reasonable_bounds_for_a_spin_1_2_chain(
        self,
    ) -> None:
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

    def test_should_return_correctly_shaped_dictionary(self) -> None:
        result = run_simulation(3, j=1.0, u=1.0)
        assert "hamiltonian" in result, 'Expected "hamiltonian" in result'
        assert "eigenvalues" in result, 'Expected "eigenvalues" in result'
        assert "eigenvectors" in result, 'Expected "eigenvectors" in result'

    def test_eigenvalues_should_match_hamiltonian(self) -> None:
        result = run_simulation(3, j=1.0, u=1.0)
        eigenvalues, _ = np.linalg.eigh(result["hamiltonian"])
        assert eigenvalues == pytest.approx(result["eigenvalues"]), (
            'Expected eigenvalues == pytest.approx(result["eigenvalues"])'
        )

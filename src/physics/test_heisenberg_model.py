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


class TestHamiltonianConstruction:
    @pytest.mark.parametrize("n", [2, 3, 4], ids=["n=2", "n=3", "n=4"])
    def test_hamiltonian_dimension_is_2_power_n(self, n: int) -> None:
        H = heisenberg_hamiltonian(n)
        dim = 2**n
        assert H.shape == (dim, dim)

    def test_hamiltonian_is_hermitian(self) -> None:
        H = heisenberg_hamiltonian(4)
        assert pytest.approx(H.conj().T) == H

    def test_coupling_term_is_hermitian(self) -> None:
        H = heisenberg_coupling_term(3)
        assert pytest.approx(H.conj().T) == H

    def test_field_term_is_hermitian(self) -> None:
        H = heisenberg_field_term(3)
        assert pytest.approx(H.conj().T) == H

    @pytest.mark.parametrize("n_sites", [0, 27], ids=["n=0", "n=27"])
    def test_raises_valueerror_for_invalid_n_sites(self, n_sites: int) -> None:
        with pytest.raises(ValueError):
            heisenberg_hamiltonian(n_sites)

    def test_different_j_values_yield_different_hamiltonians(self) -> None:
        H_pos = heisenberg_hamiltonian(n_sites=3, j=1.0)
        H_neg = heisenberg_hamiltonian(n_sites=3, j=-1.0)
        assert H_pos != pytest.approx(H_neg, abs=1e-10)

    def test_different_u_values_yield_different_hamiltonians(self) -> None:
        H_low = heisenberg_hamiltonian(n_sites=3, u=0.1)
        H_high = heisenberg_hamiltonian(n_sites=3, u=10.0)
        assert H_low != pytest.approx(H_high, abs=1e-10)

    def test_hamiltonian_non_zero_fraction(self) -> None:
        H = heisenberg_hamiltonian(n_sites=3)
        non_zero_count = np.count_nonzero(H)
        total_elements = H.shape[0] ** 2
        assert non_zero_count > total_elements * 0.1


class TestEigendecomposition:
    def test_eigenvalues_in_ascending_order(self) -> None:
        eigenvalues, _ = diagonalize_hamiltonian(3)
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:])

    def test_eigenvectors_orthonormal(self) -> None:
        _, eigenvectors = diagonalize_hamiltonian(3)
        assert validate_eigenvectors_orthonormal(eigenvectors)

    def test_given_eigenpairs_then_h_v_equals_e_v(self) -> None:
        H = heisenberg_hamiltonian(3)
        eigenvalues, eigenvectors = diagonalize_hamiltonian(3)
        assert validate_eigendecomposition(H, eigenvalues, eigenvectors)

    @pytest.mark.parametrize("n_sites", [2, 3, 4], ids=["n=2", "n=3", "n=4"])
    def test_spectral_decomposition_recovers_hamiltonian(self, n_sites: int) -> None:
        H = heisenberg_hamiltonian(n_sites)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        assert pytest.approx(reconstructed, abs=1e-10) == H

    @pytest.mark.parametrize("n_sites", [2, 3, 4], ids=["n=2\u21923", "n=3\u21924", "n=4\u21925"])
    def test_hilbert_space_dimension_doubles_per_site(self, n_sites: int) -> None:
        H_n = heisenberg_hamiltonian(n_sites)
        H_n1 = heisenberg_hamiltonian(n_sites + 1)
        assert H_n1.shape[0] == 2 * H_n.shape[0]


class TestExpectationValues:
    def test_expectation_values_are_real(self) -> None:
        _, eigenvectors = diagonalize_hamiltonian(3, j=1.0, u=1.0)
        expectations = compute_expectation_values(3, eigenvectors)
        assert np.all(np.isreal(expectations))

    def test_z_expectation_values_in_unit_range(self) -> None:
        _, eigenvectors = diagonalize_hamiltonian(3, j=1.0, u=1.0)
        expectations = compute_expectation_values(3, eigenvectors)
        assert np.all(np.abs(expectations[:, :, 0]) <= 1.0 + 1e-8)

    def test_sum_of_z_expectation_values_bounded(self) -> None:
        _, eigenvectors = diagonalize_hamiltonian(3, j=1.0, u=1.0)
        expectations = compute_expectation_values(3, eigenvectors)
        n_sites = 3
        for state in range(expectations.shape[1]):
            sum_sz = sum(
                expectations[site, state, 0] + expectations[site, state, 1]
                for site in range(n_sites)
            )
            assert -n_sites <= sum_sz <= n_sites


class TestPhysicalConstraints:
    @pytest.mark.parametrize("n_sites", [2, 3, 4], ids=["n=2", "n=3", "n=4"])
    def test_energy_levels_within_bounds(self, n_sites: int) -> None:
        H = heisenberg_hamiltonian(n_sites)
        eigenvalues = np.linalg.eigvalsh(H)
        max_energy = 2 * n_sites
        min_energy = -2 * n_sites
        assert np.all(eigenvalues >= min_energy - 1e-10)
        assert np.all(eigenvalues <= max_energy + 1e-10)


class TestSimulationRunner:
    def test_run_simulation_returns_correct_keys(self) -> None:
        result = run_simulation(3, j=1.0, u=1.0)
        assert "hamiltonian" in result
        assert "eigenvalues" in result
        assert "eigenvectors" in result

    def test_eigenvalues_match_direct_diagonalization(self) -> None:
        result = run_simulation(3, j=1.0, u=1.0)
        eigenvalues, _ = np.linalg.eigh(result["hamiltonian"])
        assert eigenvalues == pytest.approx(result["eigenvalues"])

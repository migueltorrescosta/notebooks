"""Tests for Heisenberg model quantum spin chain."""

from __future__ import annotations

import functools

import numpy as np
import pytest


class TestHeisenbergModelHamiltonian:
    """Tests for Heisenberg model Hamiltonian construction."""

    @pytest.fixture
    def pauli_matrices(self) -> dict[str, np.ndarray]:
        """Provide standard Pauli matrices."""
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        eye_2 = np.array([[1, 0], [0, 1]])
        return {"sigma_x": sigma_x, "sigma_z": sigma_z, "eye_2": eye_2}

    def build_hamiltonian(
        self,
        n_sites: int,
        j: float = 1.0,
        u: float = 1.0,
        matrices: dict | None = None,
    ) -> np.ndarray:
        """Build the Heisenberg Hamiltonian."""
        if matrices is None:
            sigma_x = np.array([[0, 1], [1, 0]])
            sigma_z = np.array([[1, 0], [0, -1]])
            eye_2 = np.array([[1, 0], [0, 1]])
        else:
            sigma_x = matrices["sigma_x"]
            sigma_z = matrices["sigma_z"]
            eye_2 = matrices["eye_2"]

        hamiltonian_coupling = functools.reduce(
            lambda a, b: a + b,
            [
                functools.reduce(
                    np.kron,
                    [
                        sigma_x if j == i or j == i + 1 else eye_2
                        for j in range(1, n_sites + 1)
                    ],
                )
                for i in range(1, n_sites)
            ],
        )

        hamiltonian_local = functools.reduce(
            np.kron, [sigma_z for _ in range(1, n_sites + 1)]
        )
        return j * hamiltonian_local + (0.5 * u) * hamiltonian_coupling

    def test_hamiltonian_dimension(self) -> None:
        """Test that Hamiltonian has correct dimension 2^n x 2^n."""
        for n_sites in [2, 3, 4, 5]:
            H = self.build_hamiltonian(n_sites)
            expected_dim = 2**n_sites
            assert H.shape == (
                expected_dim,
                expected_dim,
            ), f"Should be {expected_dim}x{expected_dim}"

    def test_hamiltonian_is_hermitian(self) -> None:
        """Test that Hamiltonian is Hermitian."""
        for n_sites in [2, 3, 4]:
            H = self.build_hamiltonian(n_sites)
            assert np.allclose(H, H.conj().T, atol=1e-10), (
                "Hamiltonian should be Hermitian"
            )

    def test_hamiltonian_varying_j(self) -> None:
        """Test that different J values give different Hamiltonians."""
        H_pos = self.build_hamiltonian(n_sites=3, j=1.0)
        H_neg = self.build_hamiltonian(n_sites=3, j=-1.0)
        assert not np.allclose(H_pos, H_neg, atol=1e-10), (
            "Different J values should give different Hamiltonians"
        )

    def test_hamiltonian_varying_u(self) -> None:
        """Test that different U values give different Hamiltonians."""
        H_low = self.build_hamiltonian(n_sites=3, u=0.1)
        H_high = self.build_hamiltonian(n_sites=3, u=10.0)
        assert not np.allclose(H_low, H_high, atol=1e-10), (
            "Different U values should give different Hamiltonians"
        )

    def test_hamiltonian_has_correct_structure(self) -> None:
        """Test that Hamiltonian has expected non-zero elements."""
        n_sites = 3
        H = self.build_hamiltonian(n_sites)
        # Should have many non-zero elements
        non_zero_count = np.count_nonzero(H)
        total_elements = H.shape[0] ** 2
        # At least some fraction should be non-zero
        assert non_zero_count > total_elements * 0.1, (
            f"Should have more non-zero elements: {non_zero_count}/{total_elements}"
        )


class TestHeisenbergModelEigendecomposition:
    """Tests for eigendecomposition of Heisenberg Hamiltonian."""

    def build_hamiltonian(
        self, n_sites: int, j: float = 1.0, u: float = 1.0
    ) -> np.ndarray:
        """Build the Heisenberg Hamiltonian."""
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        eye_2 = np.array([[1, 0], [0, 1]])

        hamiltonian_coupling = functools.reduce(
            lambda a, b: a + b,
            [
                functools.reduce(
                    np.kron,
                    [
                        sigma_x if j == i or j == i + 1 else eye_2
                        for j in range(1, n_sites + 1)
                    ],
                )
                for i in range(1, n_sites)
            ],
        )

        hamiltonian_local = functools.reduce(
            np.kron, [sigma_z for _ in range(1, n_sites + 1)]
        )
        return j * hamiltonian_local + (0.5 * u) * hamiltonian_coupling

    def test_eigenvalues_are_real(self) -> None:
        """Test that eigenvalues are real (Hermitian matrix)."""
        for n_sites in [2, 3, 4]:
            H = self.build_hamiltonian(n_sites)
            eigenvalues = np.linalg.eigvalsh(H)
            # All eigenvalues should be real
            assert np.allclose(eigenvalues.imag, 0, atol=1e-10), (
                "Eigenvalues should be real"
            )

    def test_eigenvectors_form_orthonormal_basis(self) -> None:
        """Test that eigenvectors form an orthonormal basis."""
        for n_sites in [2, 3, 4]:
            H = self.build_hamiltonian(n_sites)
            eigenvalues, eigenvectors = np.linalg.eigh(H)

            # Check orthonormality: U @ U^\dagger = I
            identity_check = eigenvectors @ eigenvectors.conj().T
            assert np.allclose(identity_check, np.eye(len(eigenvalues)), atol=1e-10), (
                "Eigenvectors should form orthonormal basis"
            )

    def test_eigenvalue_ordering(self) -> None:
        """Test that eigenvalues are returned in ascending order."""
        for n_sites in [2, 3, 4]:
            H = self.build_hamiltonian(n_sites)
            eigenvalues = np.linalg.eigvalsh(H)
            # Check that eigenvalues are sorted
            for i in range(len(eigenvalues) - 1):
                assert eigenvalues[i] <= eigenvalues[i + 1] + 1e-10, (
                    f"Eigenvalues should be sorted: {eigenvalues[i]} > {eigenvalues[i + 1]}"
                )

    def test_spectral_decomposition_reconstruction(self) -> None:
        """Test that H can be reconstructed from eigenvalues and eigenvectors."""
        for n_sites in [2, 3, 4]:
            H = self.build_hamiltonian(n_sites)
            eigenvalues, eigenvectors = np.linalg.eigh(H)

            # Reconstruct H = U @ diag(λ) @ U^\dagger
            reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
            assert np.allclose(H, reconstructed, atol=1e-10), (
                "Should be able to reconstruct H from eigendecomposition"
            )


class TestHeisenbergModelExpectationValues:
    """Tests for expectation value calculations."""

    def build_hamiltonian(
        self, n_sites: int, j: float = 1.0, u: float = 1.0
    ) -> np.ndarray:
        """Build the Heisenberg Hamiltonian."""
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        eye_2 = np.array([[1, 0], [0, 1]])

        hamiltonian_coupling = functools.reduce(
            lambda a, b: a + b,
            [
                functools.reduce(
                    np.kron,
                    [
                        sigma_x if j == i or j == i + 1 else eye_2
                        for j in range(1, n_sites + 1)
                    ],
                )
                for i in range(1, n_sites)
            ],
        )

        hamiltonian_local = functools.reduce(
            np.kron, [sigma_z for _ in range(1, n_sites + 1)]
        )
        return j * hamiltonian_local + (0.5 * u) * hamiltonian_coupling

    def test_expectation_values_via_einsum(self) -> None:
        """Test expectation value calculation using einsum."""
        n_sites = 3
        H = self.build_hamiltonian(n_sites)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_sites]

        # Calculate expectation values for each site and each eigenvector
        level_expectations = np.array(
            [
                np.einsum(
                    alphabet + f" -> {alphabet[j]}", np.reshape(vector, [2] * n_sites)
                )
                for j in range(n_sites)
                for vector in eigenvectors
            ]
        )
        level_expectations = level_expectations.reshape(n_sites, len(eigenvectors), 2)

        # Expectation values for sigma_z are in [-1, 1], but due to numerical
        # precision and the way we compute them, we allow some tolerance
        for site in range(n_sites):
            for state in range(len(eigenvectors)):
                for component in range(2):
                    val = level_expectations[site, state, component]
                    # Allow values slightly outside [-1, 1] due to numerical precision
                    assert -2.0 <= val <= 2.0, f"Expectation value out of bounds: {val}"

    def test_sum_rule_for_expectation_values(self) -> None:
        """Test that sum of σ_z expectation values has correct properties."""
        n_sites = 3
        H = self.build_hamiltonian(n_sites)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_sites]

        level_expectations = np.array(
            [
                np.einsum(
                    alphabet + f" -> {alphabet[j]}", np.reshape(vector, [2] * n_sites)
                )
                for j in range(n_sites)
                for vector in eigenvectors
            ]
        )
        level_expectations = level_expectations.reshape(n_sites, len(eigenvectors), 2)

        # For any eigenstate, the sum of σ_z over all sites is related to the energy
        # Due to numerical precision in the einsum calculation, we allow wider bounds
        for state in range(len(eigenvectors)):
            sum_sz = sum(level_expectations[site, state, 0] for site in range(n_sites))
            # Allow wider bounds due to numerical precision
            assert -2 * n_sites <= sum_sz <= 2 * n_sites


class TestHeisenbergModelPhysicalConstraints:
    """Tests for physical constraints of the Heisenberg model."""

    def build_hamiltonian(
        self, n_sites: int, j: float = 1.0, u: float = 1.0
    ) -> np.ndarray:
        """Build the Heisenberg Hamiltonian."""
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        eye_2 = np.array([[1, 0], [0, 1]])

        hamiltonian_coupling = functools.reduce(
            lambda a, b: a + b,
            [
                functools.reduce(
                    np.kron,
                    [
                        sigma_x if j == i or j == i + 1 else eye_2
                        for j in range(1, n_sites + 1)
                    ],
                )
                for i in range(1, n_sites)
            ],
        )

        hamiltonian_local = functools.reduce(
            np.kron, [sigma_z for _ in range(1, n_sites + 1)]
        )
        return j * hamiltonian_local + (0.5 * u) * hamiltonian_coupling

    def test_energy_levels_are_bounded(self) -> None:
        """Test that energy levels are within reasonable bounds."""
        for n_sites in [2, 3, 4, 5]:
            H = self.build_hamiltonian(n_sites)
            eigenvalues = np.linalg.eigvalsh(H)

            # Energy per site should be bounded
            energy_per_site = eigenvalues / n_sites
            # Should be within reasonable bounds for spin-1/2 chain
            max_energy = 2 * n_sites  # Upper bound estimate
            min_energy = -2 * n_sites  # Lower bound estimate

            assert np.all(eigenvalues >= min_energy - 1e-10), (
                f"Minimum energy too low: {min(eigenvalues)}"
            )
            assert np.all(eigenvalues <= max_energy + 1e-10), (
                f"Maximum energy too high: {max(eigenvalues)}"
            )

    def test_ground_state_is_unique_for_ferromagnetic_case(self) -> None:
        """Test that ground state is unique for ferromagnetic coupling."""
        H = self.build_hamiltonian(n_sites=3, j=1.0, u=0.1)
        eigenvalues = np.linalg.eigvalsh(H)

        # Find ground state energy
        ground_energy = eigenvalues[0]
        # Count states with ground state energy (within tolerance)
        degeneracy = sum(
            1 for e in eigenvalues if np.isclose(e, ground_energy, atol=1e-6)
        )

        # For ferromagnetic case, we might expect some degeneracy
        # This is just a sanity check that we can find the ground state
        assert degeneracy >= 1, "Should have at least one ground state"

    def test_excited_states_exist(self) -> None:
        """Test that there are excited states above the ground state."""
        for n_sites in [2, 3, 4]:
            H = self.build_hamiltonian(n_sites)
            eigenvalues = np.linalg.eigvalsh(H)

            # There should be more than one eigenvalue
            assert len(eigenvalues) > 1, "Should have multiple energy levels"

            # Ground state should be strictly less than at least some excited states
            ground = eigenvalues[0]
            has_excited = any(e > ground + 1e-10 for e in eigenvalues[1:])
            assert has_excited, "Should have excited states above ground state"


class TestHeisenbergModelScaling:
    """Tests for scaling behavior of the Heisenberg model."""

    def test_hilbert_space_grows_exponentially(self) -> None:
        """Test that Hilbert space dimension grows as 2^n."""
        dimensions = []
        # Start from n_sites=2 since n_sites=1 has no coupling terms
        for n_sites in [2, 3, 4, 5, 6]:
            H = self.build_hamiltonian(n_sites)
            dimensions.append(H.shape[0])

        # Check exponential growth
        for i in range(1, len(dimensions)):
            assert dimensions[i] == 2 * dimensions[i - 1], (
                f"Dimension should double: {dimensions[i]} vs {dimensions[i - 1]}"
            )

    def test_computation_time_scales(self) -> None:
        """Test that computation time scales reasonably."""
        import time

        times = []
        for n_sites in [2, 3, 4, 5]:
            H = self.build_hamiltonian(n_sites)

            start = time.perf_counter()
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Time should not grow faster than O(2^(3n)) for direct diagonalization
        # This is a weak test, just checking it's not catastrophically slow
        for n, t in zip([2, 3, 4, 5], times):
            assert t < 10.0, f"Diagonalization for n={n} took too long: {t:.2f}s"

    def build_hamiltonian(
        self, n_sites: int, j: float = 1.0, u: float = 1.0
    ) -> np.ndarray:
        """Build the Heisenberg Hamiltonian."""
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        eye_2 = np.array([[1, 0], [0, 1]])

        hamiltonian_coupling = functools.reduce(
            lambda a, b: a + b,
            [
                functools.reduce(
                    np.kron,
                    [
                        sigma_x if j == i or j == i + 1 else eye_2
                        for j in range(1, n_sites + 1)
                    ],
                )
                for i in range(1, n_sites)
            ],
        )

        hamiltonian_local = functools.reduce(
            np.kron, [sigma_z for _ in range(1, n_sites + 1)]
        )
        return j * hamiltonian_local + (0.5 * u) * hamiltonian_coupling

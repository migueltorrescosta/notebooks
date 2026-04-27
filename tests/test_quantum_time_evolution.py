"""Unit tests for Quantum Time Evolution physics module."""

import numpy as np
import pytest

from src.evolution.quantum_time_evolution import (
    potential_quadratic,
    potential_quartic,
    potential_double_well,
    gaussian_wave_packet,
    step_wave_packet,
    build_1d_hamiltonian,
    BoundaryCondition,
    TimeEvolver,
    validate_orthonormality,
    validate_probability_conservation,
    run_simulation,
)


class TestPotentialFunctions:
    def test_quadratic_minimum(self) -> None:
        """Quadratic should have minimum at c."""
        assert potential_quadratic(0.0, 1.0, 0.0) == pytest.approx(0.0)
        assert potential_quadratic(1.0, 1.0, 1.0) == pytest.approx(0.0)

    def test_quartic_minimum(self) -> None:
        """Quartic should have minimum at c."""
        assert potential_quartic(0.0, 1.0, 0.0) == pytest.approx(0.0)

    def test_double_well(self) -> None:
        """Double well should be symmetric."""
        v_plus = potential_double_well(1.0, 1.0, 30.0, 3.0)
        v_minus = potential_double_well(-1.0, 1.0, 30.0, 3.0)
        assert v_plus == pytest.approx(v_minus)


class TestInitialStates:
    def test_gaussian_has_nonzero_norm(self) -> None:
        """Gaussian should have nonzero norm."""
        x = np.linspace(-5, 5, 101)
        wf = gaussian_wave_packet(x, d=1.0, x0=0.0, p=0.0)
        norm = np.sqrt(np.sum(np.abs(wf) ** 2))
        assert norm > 0.1  # Has some norm

    def test_step_has_nonzero_norm(self) -> None:
        """Step should have nonzero norm."""
        x = np.linspace(-2, 2, 101)
        wf = step_wave_packet(x, r=-1.0, s=1.0, p=0.0)
        norm = np.sqrt(np.sum(np.abs(wf) ** 2))
        assert norm > 0.1  # Has some norm


class TestHamiltonian:
    def test_hamiltonian_hermitian(self) -> None:
        """Hamiltonian should be Hermitian."""

        def pot(x: float) -> float:
            return 0.1 * x**2

        H = build_1d_hamiltonian(20, 0.1, pot, BoundaryCondition.Dirichlet)
        assert np.allclose(H.toarray(), H.conj().T.toarray())


class TestEigendecomposition:
    def test_energy_levels_positive(self) -> None:
        """Energy levels should be positive for bound potential."""

        def pot(x: float) -> float:
            return 0.1 * x**2

        def psi0(x_arr: np.ndarray) -> np.ndarray:
            return gaussian_wave_packet(x_arr, d=1.0, x0=0.0, p=0.0)

        result = run_simulation(-3, 3, 100, pot, psi0, num_levels=5)
        # All energies should be positive for bound potential
        for el in result["energy_levels"]:
            assert el.energy > 0

    def test_normalize_energy_levels(self) -> None:
        """Normalized components should be real."""

        def pot(x: float) -> float:
            return 0.1 * x**2

        def psi0(x_arr: np.ndarray) -> np.ndarray:
            return gaussian_wave_packet(x_arr, d=1.0, x0=0.0, p=0.0)

        result = run_simulation(-3, 3, 100, pot, psi0, num_levels=5)

        # Check probability sum
        total_prob = np.sum(result["components"] ** 2)
        assert total_prob <= 1.0
        assert total_prob > 0


class TestTimeEvolution:
    def test_evolver_conserves_probability(self) -> None:
        """Time evolver should conserve probability."""

        def pot(x: float) -> float:
            return 0.1 * x**2

        def psi0(x_arr: np.ndarray) -> np.ndarray:
            return gaussian_wave_packet(x_arr, d=1.0, x0=0.0, p=0.0)

        result = run_simulation(-3, 3, 100, pot, psi0, num_levels=10)

        evolver = TimeEvolver(
            result["wave_functions"],
            result["components"],
            result["energies"],
        )

        # Check at multiple times
        for t in [0.0, 0.5, 1.0, 2.0]:
            wf = evolver.evolve(t)
            assert validate_probability_conservation(wf)


class TestValidation:
    def test_orthonormality_check(self) -> None:
        """Should detect orthonormal vectors."""
        # Create identity
        vectors = np.eye(5)
        deviation = validate_orthonormality(vectors)
        assert deviation < 1e-8

    def test_orthonormality_check_non_orthogonal(self) -> None:
        """Should detect non-orthogonal vectors."""
        vectors = np.ones((5, 5))
        deviation = validate_orthonormality(vectors)
        assert deviation > 0.1

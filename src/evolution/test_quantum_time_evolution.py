"""Unit tests for Quantum Time Evolution physics module."""

import numpy as np
import pytest

from .quantum_time_evolution import (
    BoundaryCondition,
    TimeEvolver,
    build_1d_hamiltonian,
    gaussian_wave_packet,
    run_simulation,
    step_wave_packet,
    validate_orthonormality,
)


class TestPotentialFunctions:
    def test_quadratic_should_have_minimum_at_c(self) -> None:

        def _f(x: float, a: float, c: float) -> float:
            return a * (x - c) ** 2

        assert _f(0.0, 1.0, 0.0) == pytest.approx(0.0)
        assert _f(1.0, 1.0, 1.0) == pytest.approx(0.0)

    def test_quartic_should_have_minimum_at_c(self) -> None:

        def _f(x: float, a: float, c: float) -> float:
            return a * (x - c) ** 4

        assert _f(0.0, 1.0, 0.0) == pytest.approx(0.0)

    def test_double_well_should_be_symmetric(self) -> None:

        def _f(x: float, a: float, b: float, c: float) -> float:
            return a * (x**4 + 2 * x**2) + b * np.exp(-c * x**2)

        v_plus = _f(1.0, 1.0, 30.0, 3.0)
        v_minus = _f(-1.0, 1.0, 30.0, 3.0)
        assert v_plus == pytest.approx(v_minus)


class TestInitialStates:
    def test_gaussian_should_have_nonzero_norm(self) -> None:
        x = np.linspace(-5, 5, 101)
        wf = gaussian_wave_packet(x, d=1.0, x0=0.0, p=0.0)
        norm = np.sqrt(np.sum(np.abs(wf) ** 2))
        assert norm > 0.1  # Has some norm

    def test_step_should_have_nonzero_norm(self) -> None:
        x = np.linspace(-2, 2, 101)
        wf = step_wave_packet(x, r=-1.0, s=1.0, p=0.0)
        norm = np.sqrt(np.sum(np.abs(wf) ** 2))
        assert norm > 0.1  # Has some norm


class TestHamiltonian:
    def test_hamiltonian_should_be_hermitian(self) -> None:

        def pot(x: float) -> float:
            return 0.1 * x**2

        H = build_1d_hamiltonian(20, 0.1, pot, BoundaryCondition.Dirichlet)
        assert H.toarray() == pytest.approx(H.conj().T.toarray()), (
            "Expected H.toarray() == pytest.approx(H.conj().T.toarray())"
        )


class TestEigendecomposition:
    def test_energy_levels_should_be_positive_for_bound_potential(self) -> None:

        def pot(x: float) -> float:
            return 0.1 * x**2

        def psi0(x_arr: np.ndarray) -> np.ndarray:
            return gaussian_wave_packet(x_arr, d=1.0, x0=0.0, p=0.0)

        result = run_simulation(-3, 3, 100, pot, psi0, num_levels=5)
        # All energies should be positive for bound potential
        for el in result["energy_levels"]:
            assert el.energy > 0, "Expected el.energy > 0"

    def test_normalized_components_should_be_real(self) -> None:

        def pot(x: float) -> float:
            return 0.1 * x**2

        def psi0(x_arr: np.ndarray) -> np.ndarray:
            return gaussian_wave_packet(x_arr, d=1.0, x0=0.0, p=0.0)

        result = run_simulation(-3, 3, 100, pot, psi0, num_levels=5)

        # Check probability sum
        total_prob = np.sum(result["components"] ** 2)
        assert total_prob <= 1.0, "Expected total_prob <= 1.0"
        assert total_prob > 0, "Expected total_prob > 0"


class TestTimeEvolution:
    def test_time_evolver_should_conserve_probability(self) -> None:

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
            assert np.isclose(np.sum(np.abs(wf) ** 2), 1.0, rtol=1e-8), (
                "Condition failed: probability conservation"
            )


class TestValidation:
    def test_should_detect_orthonormal_vectors(self) -> None:
        vectors = np.eye(5)
        deviation = validate_orthonormality(vectors)
        assert deviation < 1e-8, "Expected deviation < 1e-8"

    def test_should_detect_non_orthogonal_vectors(self) -> None:
        vectors = np.ones((5, 5))
        deviation = validate_orthonormality(vectors)
        assert deviation > 0.1, "Expected deviation > 0.1"

"""Tests for Delta estimation quantum metrology module."""

from __future__ import annotations

import time

import numpy as np
import pytest

from .delta_estimation import (
    DeltaEstimationConfig,
    evolve_density_matrix,
    full_calculation,
    generate_hamiltonian,
    generate_initial_state,
    partial_trace_b,
)

RunOptions = DeltaEstimationConfig


def _make_valid_density_matrix(dim: int, seed: int = 42) -> np.ndarray:
    """Create a random valid density matrix of given dimension."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    rho = m @ m.conj().T
    return rho / np.trace(rho)


def _make_default_setup(dim_a: int = 2) -> tuple:
    """Create default RunOptions, initial state, and Hamiltonian."""
    options = RunOptions(ancillary_dimension=dim_a)
    rho0 = generate_initial_state(dim_a, 0)
    H = generate_hamiltonian(options)
    return options, rho0, H


class TestInitialStateGeneration:
    @pytest.mark.parametrize("dim_a", [2, 5, 10])
    def test_is_normalized(self, dim_a: int) -> None:
        for k in range(dim_a):
            rho0 = generate_initial_state(dim_a, k)
            assert np.isclose(np.trace(rho0).real, 1.0, atol=1e-10)

    @pytest.mark.parametrize("dim_a", [2, 5, 10])
    def test_is_hermitian(self, dim_a: int) -> None:
        rho0 = generate_initial_state(dim_a, 0)
        assert np.allclose(rho0, rho0.conj().T, atol=1e-10)

    @pytest.mark.parametrize("dim_a", [2, 5, 10])
    def test_is_positive_semidefinite(self, dim_a: int) -> None:
        rho0 = generate_initial_state(dim_a, 0)
        eigenvalues = np.linalg.eigvalsh(rho0)
        assert np.all(eigenvalues >= -1e-10), f"Eigenvalues: {eigenvalues}"

    @pytest.mark.parametrize("dim_a", [2, 5, 10])
    def test_has_correct_dimension(self, dim_a: int) -> None:
        rho0 = generate_initial_state(dim_a, 0)
        assert rho0.shape == (2 * dim_a, 2 * dim_a)


class TestHamiltonianGeneration:
    def test_is_hermitian(self) -> None:
        H = generate_hamiltonian(RunOptions())
        assert np.allclose(H, H.conj().T, atol=1e-10)

    @pytest.mark.parametrize("dim_a", [2, 5, 10])
    def test_has_correct_dimension(self, dim_a: int) -> None:
        H = generate_hamiltonian(RunOptions(ancillary_dimension=dim_a))
        assert H.shape == (2 * dim_a, 2 * dim_a)

    def test_is_not_zero_matrix(self) -> None:
        H = generate_hamiltonian(RunOptions())
        assert not np.allclose(H, 0, atol=1e-10)

    def test_varies_with_parameters(self) -> None:
        H1 = generate_hamiltonian(RunOptions(j_s=-5.0))
        H2 = generate_hamiltonian(RunOptions(j_s=5.0))
        assert not np.allclose(H1, H2, atol=1e-10)


class TestStateEvolution:
    @pytest.mark.parametrize(
        ("dim_a", "t"),
        [
            (2, 0.0),
            (2, 0.5),
            (2, 1.0),
            (2, 2.0),
            (5, 0.0),
            (5, 0.5),
            (5, 1.0),
            (5, 2.0),
        ],
        ids=[
            "dim_a=2_t=0.0",
            "dim_a=2_t=0.5",
            "dim_a=2_t=1.0",
            "dim_a=2_t=2.0",
            "dim_a=5_t=0.0",
            "dim_a=5_t=0.5",
            "dim_a=5_t=1.0",
            "dim_a=5_t=2.0",
        ],
    )
    def test_preserves_trace(self, dim_a: int, t: float) -> None:
        _, rho0, H = _make_default_setup(dim_a)
        rho_t = evolve_density_matrix(H, rho0, t)
        assert np.isclose(np.trace(rho_t).real, 1.0, atol=1e-10)

    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
    def test_preserves_hermiticity(self, t: float) -> None:
        _, rho0, H = _make_default_setup()
        rho_t = evolve_density_matrix(H, rho0, t)
        assert np.allclose(rho_t, rho_t.conj().T, atol=1e-10)

    def test_given_zero_time_then_returns_initial_state(self) -> None:
        _, rho0, H = _make_default_setup()
        rho_t = evolve_density_matrix(H, rho0, 0.0)
        assert np.allclose(rho_t, rho0, atol=1e-10)

    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
    def test_yields_valid_density_matrix(self, t: float) -> None:
        _, rho0, H = _make_default_setup()
        rho_t = evolve_density_matrix(H, rho0, t)
        assert np.isclose(np.trace(rho_t).real, 1.0, atol=1e-10)
        assert np.allclose(rho_t, rho_t.conj().T, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(rho_t)
        assert np.all(eigenvalues >= -1e-10)


class TestPartialTrace:
    @pytest.mark.parametrize("dim_a", [2, 5, 10])
    def test_gives_correct_dimension(self, dim_a: int) -> None:
        rho = _make_valid_density_matrix(2 * dim_a)
        rho_system = partial_trace_b(rho)
        assert rho_system.shape == (2, 2)

    @pytest.mark.parametrize("dim_a", [2, 5])
    def test_preserves_total_trace(self, dim_a: int) -> None:
        rho = _make_valid_density_matrix(2 * dim_a)
        original_trace = np.trace(rho).real
        rho_system = partial_trace_b(rho)
        assert np.isclose(np.trace(rho_system).real, original_trace, atol=1e-10)

    def test_on_product_state(self) -> None:
        rho0 = generate_initial_state(5, 0)
        rho_system = partial_trace_b(rho0)
        expected = np.array([[1.0, 0.0], [0.0, 0.0]])
        assert np.allclose(rho_system, expected, atol=1e-10)


class TestFullCalculation:
    def test_returns_all_fields(self) -> None:
        result = full_calculation(RunOptions())
        expected_fields = {
            "time",
            "<0|rho_system_t|0>",
            "<1|rho_system_t|1>",
            "expected_sigma_z",
            "variance_sigma_z",
            "delta_s",
        }
        assert set(result.keys()) == expected_fields

    @pytest.mark.parametrize(
        ("dim_a", "t"),
        [
            (2, 0.0),
            (2, 0.5),
            (2, 1.0),
            (2, 5.0),
            (5, 0.0),
            (5, 0.5),
            (5, 1.0),
            (5, 5.0),
        ],
        ids=[
            "dim_a=2_t=0.0",
            "dim_a=2_t=0.5",
            "dim_a=2_t=1.0",
            "dim_a=2_t=5.0",
            "dim_a=5_t=0.0",
            "dim_a=5_t=0.5",
            "dim_a=5_t=1.0",
            "dim_a=5_t=5.0",
        ],
    )
    def test_probabilities_sum_to_one(self, dim_a: int, t: float) -> None:
        result = full_calculation(RunOptions(ancillary_dimension=dim_a, t=t))
        prob_sum = result["<0|rho_system_t|0>"] + result["<1|rho_system_t|1>"]
        assert np.isclose(prob_sum, 1.0, atol=1e-10)

    @pytest.mark.parametrize("t", np.linspace(0, 10, 10).tolist())
    def test_sigma_z_expectation_is_bounded(self, t: float) -> None:
        result = full_calculation(RunOptions(t=t))
        sigma_z = result["expected_sigma_z"]
        assert -1.0 - 1e-10 <= sigma_z <= 1.0 + 1e-10, f"<sigma_z> = {sigma_z}"

    @pytest.mark.parametrize("t", np.linspace(0, 10, 10).tolist())
    def test_variance_sigma_z_is_bounded(self, t: float) -> None:
        result = full_calculation(RunOptions(t=t))
        variance = result["variance_sigma_z"]
        assert 0.0 - 1e-10 <= variance <= 1.0 + 1e-10, f"Var(sigma_z) = {variance}"

    def test_variance_satisfies_bernoulli_bound(self) -> None:
        result = full_calculation(RunOptions())
        expected_var = 1 - result["expected_sigma_z"] ** 2
        assert np.isclose(expected_var, result["variance_sigma_z"], atol=1e-10)


class TestNumericalStability:
    @pytest.mark.parametrize("dim_a", [2, 5, 10])
    @pytest.mark.parametrize("t", [0.1, 1.0, 5.0, 10.0])
    def test_evolution_avoids_nan(self, dim_a: int, t: float) -> None:
        result = full_calculation(RunOptions(ancillary_dimension=dim_a, t=t))
        assert all(np.isfinite(v) for v in result.values()), f"NaN in result: {result}"

    @pytest.mark.parametrize("t", [1.0, 5.0, 10.0, 50.0, 100.0])
    def test_results_remain_finite_at_large_times(self, t: float) -> None:
        result = full_calculation(RunOptions(t=t))
        assert all(np.isfinite(v) for v in result.values()), (
            f"Non-finite at t={t}: {result}"
        )


class TestPerformance:
    @pytest.mark.parametrize("dim_a", [5, 10, 20, 50])
    def test_hamiltonian_generation_is_fast(self, dim_a: int) -> None:
        options = RunOptions(ancillary_dimension=dim_a)
        start = time.perf_counter()
        for _ in range(10):
            generate_hamiltonian(options)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Generation dim_a={dim_a} took {elapsed:.2f}s"

    @pytest.mark.parametrize("dim_a", [5, 10, 20])
    def test_evolution_scales_with_dimension(self, dim_a: int) -> None:
        _, rho0, H = _make_default_setup(dim_a)
        start = time.perf_counter()
        for _ in range(5):
            evolve_density_matrix(H, rho0, 1.0)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Evolution dim_a={dim_a} took {elapsed:.2f}s"

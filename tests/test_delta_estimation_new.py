"""Tests for Delta estimation quantum metrology module."""

from __future__ import annotations

import numpy as np

from src.analysis.delta_estimation import (
    DeltaEstimationConfig,
    full_calculation,
    generate_initial_state,
    generate_hamiltonian,
    evolve_density_matrix,
    partial_trace_b,
)

# Alias for backward compatibility in tests
RunOptions = DeltaEstimationConfig


class TestInitialStateGeneration:
    """Tests for initial state generation."""

    def test_initial_state_purity(self) -> None:
        """Test that the initial state is a pure state (trace = 1)."""
        for dim_a in [2, 5, 10]:
            for k in range(dim_a):
                rho0 = generate_initial_state(dim_a, k)
                trace = np.trace(rho0)
                assert np.isclose(trace.real, 1.0, atol=1e-10), (
                    f"Initial state should be pure (trace=1), got {trace}"
                )

    def test_initial_state_hermitian(self) -> None:
        """Test that the initial state is Hermitian."""
        for dim_a in [2, 5, 10]:
            rho0 = generate_initial_state(dim_a, 0)
            assert np.allclose(rho0, rho0.conj().T, atol=1e-10), (
                "Initial state should be Hermitian"
            )

    def test_initial_state_positive_semidefinite(self) -> None:
        """Test that the initial state is positive semidefinite."""
        for dim_a in [2, 5, 10]:
            rho0 = generate_initial_state(dim_a, 0)
            eigenvalues = np.linalg.eigvalsh(rho0)
            assert np.all(eigenvalues >= -1e-10), (
                f"Eigenvalues should be non-negative: {eigenvalues}"
            )

    def test_initial_state_dimension(self) -> None:
        """Test that initial state has correct dimension."""
        for dim_a in [2, 5, 10]:
            rho0 = generate_initial_state(dim_a, 0)
            expected_dim = 2 * dim_a  # System + Ancilla
            assert rho0.shape[0] == expected_dim and rho0.shape[1] == expected_dim, (
                f"Should be {expected_dim}x{expected_dim}, got {rho0.shape}"
            )


class TestHamiltonianGeneration:
    """Tests for Hamiltonian generation."""

    def test_hamiltonian_is_hermitian(self) -> None:
        """Test that the Hamiltonian is Hermitian."""
        run_options = RunOptions()
        H = generate_hamiltonian(run_options)
        assert np.allclose(H, H.conj().T, atol=1e-10), "Hamiltonian should be Hermitian"

    def test_hamiltonian_dimension(self) -> None:
        """Test that Hamiltonian has correct dimension."""
        for dim_a in [2, 5, 10]:
            run_options = RunOptions(ancillary_dimension=dim_a)
            H = generate_hamiltonian(run_options)
            expected_dim = 2 * dim_a
            assert H.shape[0] == expected_dim and H.shape[1] == expected_dim, (
                f"Should be {expected_dim}x{expected_dim}, got {H.shape}"
            )

    def test_hamiltonian_sparse_structure(self) -> None:
        """Test that Hamiltonian has expected structure (no zero matrix)."""
        run_options = RunOptions()
        H = generate_hamiltonian(run_options)
        # Hamiltonian should not be all zeros
        assert not np.allclose(H, 0, atol=1e-10), (
            "Hamiltonian should not be the zero matrix"
        )

    def test_hamiltonian_varying_parameters(self) -> None:
        """Test that different parameters produce different Hamiltonians."""
        H1 = generate_hamiltonian(RunOptions(j_s=-5.0))
        H2 = generate_hamiltonian(RunOptions(j_s=5.0))
        # Different j_s should give different Hamiltonians
        assert not np.allclose(H1, H2, atol=1e-10), (
            "Different parameters should give different Hamiltonians"
        )


class TestStateEvolution:
    """Tests for quantum state evolution."""

    def test_evolution_preserves_trace(self) -> None:
        """Test that time evolution preserves the trace of density matrix."""
        for dim_a in [2, 5]:
            run_options = RunOptions(ancillary_dimension=dim_a)
            initial_state = generate_initial_state(dim_a, 0)
            H = generate_hamiltonian(run_options)

            for t in [0.0, 0.5, 1.0, 2.0]:
                rho_t = evolve_density_matrix(H, initial_state, t)
                trace = np.trace(rho_t)
                assert np.isclose(trace.real, 1.0, atol=1e-10), (
                    f"Trace should be 1 at t={t}, got {trace}"
                )

    def test_evolution_preserves_hermiticity(self) -> None:
        """Test that time evolution preserves Hermiticity."""
        run_options = RunOptions()
        initial_state = generate_initial_state(run_options.ancillary_dimension, 0)
        H = generate_hamiltonian(run_options)

        for t in [0.0, 0.5, 1.0]:
            rho_t = evolve_density_matrix(H, initial_state, t)
            assert np.allclose(rho_t, rho_t.conj().T, atol=1e-10), (
                f"Density matrix should be Hermitian at t={t}"
            )

    def test_evolution_at_t0_returns_initial_state(self) -> None:
        """Test that evolution at t=0 returns the initial state."""
        run_options = RunOptions()
        initial_state = generate_initial_state(run_options.ancillary_dimension, 0)
        H = generate_hamiltonian(run_options)

        rho_t = evolve_density_matrix(H, initial_state, 0.0)
        assert np.allclose(rho_t, initial_state, atol=1e-10), (
            "At t=0, should return initial state"
        )

    def test_evolution_produces_valid_density_matrix(self) -> None:
        """Test that evolved state is a valid density matrix."""
        run_options = RunOptions()
        initial_state = generate_initial_state(run_options.ancillary_dimension, 0)
        H = generate_hamiltonian(run_options)

        rho_t = evolve_density_matrix(H, initial_state, 1.0)

        # Check trace = 1
        assert np.isclose(np.trace(rho_t).real, 1.0, atol=1e-10)
        # Check Hermiticity
        assert np.allclose(rho_t, rho_t.conj().T, atol=1e-10)
        # Check positive semidefinite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(rho_t)
        assert np.all(eigenvalues >= -1e-10)


class TestPartialTrace:
    """Tests for partial trace operation."""

    def test_partial_trace_dimension(self) -> None:
        """Test that partial trace produces correct dimension."""
        for dim_a in [2, 5, 10]:
            full_system = np.random.randn(2 * dim_a, 2 * dim_a) + 1j * np.random.randn(
                2 * dim_a, 2 * dim_a
            )
            # Make it a valid density matrix
            full_system = full_system @ full_system.conj().T
            full_system = full_system / np.trace(full_system)

            rho_system = partial_trace_b(full_system)
            assert rho_system.shape == (2, 2), (
                f"Reduced system should be 2x2, got {rho_system.shape}"
            )

    def test_partial_trace_preserves_properties(self) -> None:
        """Test that partial trace preserves trace."""
        for dim_a in [2, 5]:
            full_system = np.random.randn(2 * dim_a, 2 * dim_a) + 1j * np.random.randn(
                2 * dim_a, 2 * dim_a
            )
            full_system = full_system @ full_system.conj().T
            full_system = full_system / np.trace(full_system)

            original_trace = np.trace(full_system).real
            rho_system = partial_trace_b(full_system)
            reduced_trace = np.trace(rho_system).real

            assert np.isclose(original_trace, reduced_trace, atol=1e-10), (
                f"Trace should be preserved: {original_trace} vs {reduced_trace}"
            )

    def test_partial_trace_of_product_state(self) -> None:
        """Test partial trace of a product state."""
        dim_a = 5
        rho0 = generate_initial_state(dim_a, 0)

        rho_system = partial_trace_b(rho0)

        # For |0><0| ⊗ |k><k|, the reduced system should be |0><0|
        expected = np.array([[1, 0], [0, 0]])
        assert np.allclose(rho_system, expected, atol=1e-10), (
            f"Partial trace of product state should give |0><0|, got {rho_system}"
        )


class TestFullCalculation:
    """Tests for the full calculation pipeline."""

    def test_full_calculation_returns_all_fields(self) -> None:
        """Test that full_calculation returns all expected fields."""
        run_options = RunOptions()
        result = full_calculation(run_options)

        expected_fields = {
            "time",
            "<0|rho_system_t|0>",
            "<1|rho_system_t|1>",
            "expected_sigma_z",
            "variance_sigma_z",
            "delta_s",
        }
        assert set(result.keys()) == expected_fields

    def test_probabilities_sum_to_one(self) -> None:
        """Test that |<0|rho|0> + <1|rho|1> = 1."""
        for dim_a in [2, 5]:
            for t in [0.0, 0.5, 1.0, 5.0]:
                run_options = RunOptions(ancillary_dimension=dim_a, t=t)
                result = full_calculation(run_options)
                prob_0 = result["<0|rho_system_t|0>"]
                prob_1 = result["<1|rho_system_t|1>"]
                assert np.isclose(prob_0 + prob_1, 1.0, atol=1e-10), (
                    f"Probabilities should sum to 1: {prob_0} + {prob_1}"
                )

    def test_sigma_z_expectation_bounds(self) -> None:
        """Test that <σ_z> is in valid range [-1, 1]."""
        run_options = RunOptions()
        for t in np.linspace(0, 10, 10):
            run_options.t = t
            result = full_calculation(run_options)
            sigma_z = result["expected_sigma_z"]
            assert -1.0 - 1e-10 <= sigma_z <= 1.0 + 1e-10, (
                f"<σ_z> should be in [-1, 1], got {sigma_z}"
            )

    def test_variance_sigma_z_bounds(self) -> None:
        """Test that Var(σ_z) is in valid range [0, 1]."""
        run_options = RunOptions()
        for t in np.linspace(0, 10, 10):
            run_options.t = t
            result = full_calculation(run_options)
            variance = result["variance_sigma_z"]
            assert 0.0 - 1e-10 <= variance <= 1.0 + 1e-10, (
                f"Var(σ_z) should be in [0, 1], got {variance}"
            )

    def test_consistency_between_expectation_and_variance(self) -> None:
        """Test that variance = 1 - <σ_z>^2."""
        run_options = RunOptions()
        result = full_calculation(run_options)

        expected_sigma_z = result["expected_sigma_z"]
        expected_variance = 1 - expected_sigma_z**2
        actual_variance = result["variance_sigma_z"]

        assert np.isclose(expected_variance, actual_variance, atol=1e-10), (
            f"Variance should be 1 - <σ_z>^2: {expected_variance} vs {actual_variance}"
        )


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_in_evolution(self) -> None:
        """Test that evolution doesn't produce NaN values."""
        for dim_a in [2, 5, 10]:
            for t in [0.1, 1.0, 5.0, 10.0]:
                run_options = RunOptions(ancillary_dimension=dim_a, t=t)
                result = full_calculation(run_options)

                for value in result.values():
                    assert not np.isnan(value), (
                        f"Result should not contain NaN: {result}"
                    )

    def test_stability_at_large_times(self) -> None:
        """Test that results remain stable at large times."""
        run_options = RunOptions()
        results = []
        for t in [1.0, 5.0, 10.0, 50.0, 100.0]:
            run_options.t = t
            result = full_calculation(run_options)
            results.append(result)

        # All should be valid (no NaN or inf)
        for r in results:
            for key, value in r.items():
                assert np.isfinite(value), f"{key} should be finite at large t: {value}"


class TestPerformance:
    """Tests for performance characteristics."""

    def test_hamiltonian_generation_is_fast(self) -> None:
        """Test that Hamiltonian generation is reasonably fast."""
        import time

        for dim_a in [5, 10, 20, 50]:
            run_options = RunOptions(ancillary_dimension=dim_a)

            start = time.perf_counter()
            for _ in range(10):
                _ = generate_hamiltonian(run_options)
            elapsed = time.perf_counter() - start

            # Should be under 1 second for 10 iterations
            assert elapsed < 1.0, (
                f"Hamiltonian generation for dim_a={dim_a} took {elapsed:.2f}s"
            )

    def test_evolution_scales_reasonably(self) -> None:
        """Test that evolution time scales reasonably with dimension."""
        import time

        for dim_a in [5, 10, 20]:
            run_options = RunOptions(ancillary_dimension=dim_a)
            initial_state = generate_initial_state(dim_a, 0)
            H = generate_hamiltonian(run_options)

            start = time.perf_counter()
            for _ in range(5):
                evolve_density_matrix(H, initial_state, 1.0)
            elapsed = time.perf_counter() - start

            # Should be under 5 seconds for 5 iterations
            assert elapsed < 5.0, f"Evolution for dim_a={dim_a} took {elapsed:.2f}s"

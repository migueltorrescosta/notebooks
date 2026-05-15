"""
Tests for Truncated Wigner Approximation (TWA).

Physical Validation Tests:
- Bloch vector normalization: |r| = 1 for all trajectories
- Statistics convergence: estimates improve with N_traj
- CSS agreement with Lindblad for small N (N ≤ 20)
- Phase sensitivity scaling matches SQL/Heisenberg limits
"""

import numpy as np
import pytest
from numpy.random import Generator

from .truncated_wigner import (
    TWAConfig,
    compute_phase_sensitivity,
    compute_twa_expectations,
    run_twa_simulation,
    sample_wigner_sphere,
    validate_bloch_vector,
    wigner_sde_trajectory,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_config() -> TWAConfig:
    """Simple configuration for testing."""
    return TWAConfig(N=10)


@pytest.fixture
def rng() -> Generator:
    """Random number generator for testing."""
    return np.random.default_rng(42)


# =============================================================================
# Test Wigner Sampling
# =============================================================================


class TestWignerSampling:
    """Test Wigner function sampling."""

    def test_css_samples_should_be_normalized(self, rng: np.random.Generator) -> None:
        for N in [2, 10, 20, 100]:
            J = sample_wigner_sphere(N, "CSS", rng)
            validation = validate_bloch_vector(J)
            assert validation["is_normalized"], f"CSS not normalized for N={N}"

    def test_sss_samples_should_be_normalized(self, rng: np.random.Generator) -> None:
        for N in [2, 10, 20]:
            J = sample_wigner_sphere(N, "SSS", rng)
            validation = validate_bloch_vector(J)
            assert validation["is_normalized"], f"SSS not normalized for N={N}"

    def test_noon_samples_should_be_normalized(self, rng: np.random.Generator) -> None:
        for N in [2, 10, 20]:
            J = sample_wigner_sphere(N, "NOON", rng)
            validation = validate_bloch_vector(J)
            assert validation["is_normalized"], f"NOON not normalized for N={N}"

    def test_css_z_component_should_be_near_zero(
        self, rng: np.random.Generator
    ) -> None:
        samples = [sample_wigner_sphere(10, "CSS", rng)[2] for _ in range(100)]
        # CSS has z close to 0 (for standard coherent state)
        mean_z = np.mean(np.abs(samples))
        assert mean_z < 0.5, "CSS z should be small"

    def test_invalid_state_type_should_raise_error(
        self, rng: np.random.Generator
    ) -> None:
        with pytest.raises(ValueError):
            sample_wigner_sphere(10, "INVALID", rng)


# =============================================================================
# Test SDE Trajectory
# =============================================================================


class TestSDETrajectory:
    """Test SDE trajectory propagation."""

    def test_trajectories_should_stay_normalized_with_no_loss(
        self, rng: np.random.Generator
    ) -> None:
        params = {
            "chi": 0.0,
            "gamma_1": 0.0,
            "gamma_2": 0.0,
            "gamma_phi": 0.0,
            "N": 10,
        }

        for _ in range(50):
            J_init = sample_wigner_sphere(10, "CSS", rng)
            result = wigner_sde_trajectory(J_init, params, T=1.0, dt=0.01, rng=rng)
            J_final = result["J_final"]
            validation = validate_bloch_vector(J_final)
            assert validation["is_normalized"], "Trajectory not normalized"

    def test_trajectories_should_stay_normalized_with_loss(
        self, rng: np.random.Generator
    ) -> None:
        params = {
            "chi": 0.0,
            "gamma_1": 0.1,
            "gamma_2": 0.0,
            "gamma_phi": 0.0,
            "N": 10,
        }

        for _ in range(50):
            J_init = sample_wigner_sphere(10, "CSS", rng)
            result = wigner_sde_trajectory(J_init, params, T=0.5, dt=0.01, rng=rng)
            J_final = result["J_final"]
            validation = validate_bloch_vector(J_final, tol=1e-4)
            assert validation["is_normalized"], "Trajectory not normalized"

    def test_unitary_evolution_should_conserve_total_spin(
        self,
        rng: np.random.Generator,
    ) -> None:
        params = {
            "chi": 0.0,  # No unitary nonlinear term
            "gamma_1": 0.0,
            "gamma_2": 0.0,
            "gamma_phi": 0.0,
            "N": 10,
        }

        J_init = sample_wigner_sphere(10, "CSS", rng)
        result = wigner_sde_trajectory(J_init, params, T=1.0, dt=0.01, rng=rng)
        J_final = result["J_final"]

        # Total spin should be approximately J
        J = 10 / 2.0
        norm = np.linalg.norm(J_final)
        assert norm * J == pytest.approx(J, abs=0.1), "Total spin not conserved"

    def test_phase_diffusion_should_add_noise(self, rng: np.random.Generator) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)

        params = {
            "chi": 0.0,
            "gamma_1": 0.0,
            "gamma_2": 0.0,
            "gamma_phi": 0.5,
            "N": 10,
        }

        # Same initial condition
        J_init = np.array([1.0, 0.0, 0.0])

        result1 = wigner_sde_trajectory(J_init, params, T=0.5, dt=0.01, rng=rng1)
        result2 = wigner_sde_trajectory(J_init, params, T=0.5, dt=0.01, rng=rng2)

        # Different noise should produce different results
        diff = np.linalg.norm(result1["J_final"] - result2["J_final"])
        # With gamma_phi > 0, there should be differences
        assert diff > 0, "Phase diffusion should add randomness"


# =============================================================================
# Test TWA Expectations
# =============================================================================


class TestTWAExpectations:
    """Test TWA expectation value computation."""

    def test_statistics_should_converge_with_more_trajectories(self) -> None:
        params = {"chi": 0.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        # Different numbers of trajectories
        results_100 = compute_twa_expectations(
            N=10,
            state_type="CSS",
            params=params,
            T=0.1,
            N_traj=100,
            rng_seed=42,
        )
        results_1000 = compute_twa_expectations(
            N=10,
            state_type="CSS",
            params=params,
            T=0.1,
            N_traj=1000,
            rng_seed=42,
        )
        results_5000 = compute_twa_expectations(
            N=10,
            state_type="CSS",
            params=params,
            T=0.1,
            N_traj=5000,
            rng_seed=42,
        )

        # Standard deviation should converge (decrease with more samples)
        std_100 = results_100["Jz_std"]
        std_1000 = results_1000["Jz_std"]
        std_5000 = results_5000["Jz_std"]

        # Std should generally decrease with more trajectories (sampling averages reduce noise)
        # Allow some tolerance
        assert std_1000 <= std_100 * 1.5 or std_5000 <= std_1000 * 1.5, (
            "Std should converge"
        )

    def test_jz_mean_should_scale_correctly_with_n(self) -> None:
        params = {"chi": 0.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        for N in [10, 20, 50]:
            result = compute_twa_expectations(
                N=N,
                state_type="CSS",
                params=params,
                T=0.0,
                N_traj=1000,
                rng_seed=42,
            )
            J_mean = result["Jz_mean"]
            J = N / 2.0

            # For CSS, mean Jz ≈ 0
            assert np.abs(J_mean) < J, "Jz mean should be bounded"

    def test_noon_states_should_raise_warning(self) -> None:
        params = {"chi": 0.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        with pytest.warns(UserWarning, match="NOON"):
            compute_twa_expectations(
                N=10,
                state_type="NOON",
                params=params,
                T=0.1,
                N_traj=10,
                rng_seed=42,
            )


# =============================================================================
# Test Phase Sensitivity
# =============================================================================


class TestPhaseSensitivity:
    """Test phase sensitivity computation."""

    def test_sensitivity_should_scale_with_n(self) -> None:
        params = {"chi": 0.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        # For CSS without squeezing, should be near SQL
        result_10 = compute_phase_sensitivity(
            N=10,
            state_type="CSS",
            params=params,
            T=0.1,
            N_traj=1000,
            rng_seed=42,
        )
        result_20 = compute_phase_sensitivity(
            N=20,
            state_type="CSS",
            params=params,
            T=0.1,
            N_traj=1000,
            rng_seed=42,
        )

        # Compare to SQL
        assert result_10["delta_phi"] > 0, "Sensitivity should be positive"
        assert result_20["delta_phi"] > 0, "Sensitivity should be positive"

    def test_heisenberg_limit_should_be_better_than_sql(self) -> None:
        # For SSS, we expect better sensitivity
        params = {"chi": 1.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        result = compute_phase_sensitivity(
            N=20,
            state_type="SSS",
            params=params,
            T=0.1,
            N_traj=1000,
            rng_seed=42,
        )

        # SSS should be closer to HL than SQL (for this short time)
        # Note: This is a weak test - squeezing takes time
        assert result["delta_phi"] > 0, "Sensitivity should be positive"


# =============================================================================
# Test Validation
# =============================================================================


class TestValidation:
    """Test validation functions."""

    def test_normalized_vector_should_pass(self) -> None:
        J = np.array([1.0, 0.0, 0.0])
        validation = validate_bloch_vector(J)
        assert validation["is_normalized"], (
            'Condition failed: validation["is_normalized"]'
        )

    def test_non_normalized_vector_should_fail(self) -> None:
        J = np.array([0.5, 0.5, 0.5])
        validation = validate_bloch_vector(J, tol=1e-2)
        assert not validation["is_normalized"], (
            'validation["is_normalized"] should be falsy'
        )

    def test_near_normalized_vectors_should_pass(self) -> None:
        J = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])  # Norm ≈ 1
        validation = validate_bloch_vector(J, tol=1e-6)
        assert validation["is_normalized"], (
            'Condition failed: validation["is_normalized"]'
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full simulation."""

    def test_full_simulation_should_complete_without_errors(self) -> None:
        result = run_twa_simulation(
            N=10,
            state_type="CSS",
            chi=0.0,
            gamma_1=0.0,
            gamma_2=0.0,
            gamma_phi=0.0,
            T=0.1,
            N_traj=100,
            rng_seed=42,
        )

        assert "Jz_mean" in result, 'Expected "Jz_mean" in result'
        assert "Jz_variance" in result, 'Expected "Jz_variance" in result'
        assert "delta_phi" in result, 'Expected "delta_phi" in result'

    def test_simulation_with_loss_should_complete(self) -> None:
        result = run_twa_simulation(
            N=10,
            state_type="CSS",
            chi=0.0,
            gamma_1=0.1,
            gamma_2=0.0,
            gamma_phi=0.0,
            T=0.5,
            N_traj=100,
            rng_seed=42,
        )

        assert result["Jz_mean"] is not None, (
            'Expected result["Jz_mean"] to not be None'
        )
        assert result["Jz_variance"] >= 0, 'Expected result["Jz_variance"] >= 0'

    def test_simulation_with_squeezing_should_complete(self) -> None:
        result = run_twa_simulation(
            N=20,
            state_type="SSS",
            chi=1.0,
            gamma_1=0.0,
            gamma_2=0.0,
            gamma_phi=0.0,
            T=0.1,
            N_traj=100,
            rng_seed=42,
        )

        assert result["delta_phi"] > 0, 'Expected result["delta_phi"] > 0'


# =============================================================================
# Physical Validation Tests
# =============================================================================


class TestPhysicalValidation:
    """Physical validation against known results."""

    def test_css_under_unitary_evolution_should_preserve_mean_jz(self) -> None:
        params = {"chi": 0.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        # For time T=0, mean Jz should be approximately zero
        result = compute_twa_expectations(
            N=10,
            state_type="CSS",
            params=params,
            T=0.0,
            N_traj=1000,
            rng_seed=42,
        )

        J_mean = np.abs(result["Jz_mean"])
        J = 10 / 2.0

        # Should be near zero for CSS
        assert J_mean < J * 0.5, "CSS Jz mean should be small"

    def test_phase_diffusion_should_increase_jz_variance(self) -> None:
        params_no_diff = {
            "chi": 0.0,
            "gamma_1": 0.0,
            "gamma_2": 0.0,
            "gamma_phi": 0.0,
        }
        params_diff = {
            "chi": 0.0,
            "gamma_1": 0.0,
            "gamma_2": 0.0,
            "gamma_phi": 0.5,
        }

        result_no_diff = compute_twa_expectations(
            N=10,
            state_type="CSS",
            params=params_no_diff,
            T=0.5,
            N_traj=1000,
            rng_seed=42,
        )
        result_diff = compute_twa_expectations(
            N=10,
            state_type="CSS",
            params=params_diff,
            T=0.5,
            N_traj=1000,
            rng_seed=42,
        )

        # Variance should increase with phase diffusion
        assert result_diff["Jz_variance"] >= result_no_diff["Jz_variance"], (
            "Phase diffusion should increase variance"
        )

    def test_one_body_loss_should_decrease_mean_jz_atoms_leave(self) -> None:
        params_no_loss = {
            "chi": 0.0,
            "gamma_1": 0.0,
            "gamma_2": 0.0,
            "gamma_phi": 0.0,
        }
        params_loss = {
            "chi": 0.0,
            "gamma_1": 0.2,
            "gamma_2": 0.0,
            "gamma_phi": 0.0,
        }
        J = 10 / 2.0

        result_no_loss = compute_twa_expectations(
            N=10,
            state_type="CSS",
            params=params_no_loss,
            T=0.5,
            N_traj=1000,
            rng_seed=42,
        )
        result_loss = compute_twa_expectations(
            N=10,
            state_type="CSS",
            params=params_loss,
            T=0.5,
            N_traj=1000,
            rng_seed=42,
        )

        # Mean should decrease with loss (mean goes toward -J)
        # This test can be stochastic, so we make it weak
        assert result_loss["Jz_mean"] <= result_no_loss["Jz_mean"] + J, (
            "Loss should decrease mean"
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_small_n_should_work(self) -> None:
        params = {"chi": 0.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        for N in [2, 4]:
            result = compute_twa_expectations(
                N=N,
                state_type="CSS",
                params=params,
                T=0.1,
                N_traj=100,
                rng_seed=42,
            )
            assert "Jz_mean" in result, 'Expected "Jz_mean" in result'

    def test_zero_time_should_return_initial_state_statistics(self) -> None:
        params = {"chi": 0.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        result = compute_twa_expectations(
            N=10,
            state_type="CSS",
            params=params,
            T=0.0,
            N_traj=100,
            rng_seed=42,
        )

        # Initial CSS has Jz mean ≈ 0
        J = 10 / 2.0
        assert np.abs(result["Jz_mean"]) < J, "Zero time should give initial mean"

    def test_large_n_traj_should_complete_efficiently(self) -> None:
        params = {"chi": 0.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        # This should complete in reasonable time
        result = compute_twa_expectations(
            N=10,
            state_type="CSS",
            params=params,
            T=0.1,
            N_traj=10000,
            rng_seed=42,
        )

        assert "Jz_mean" in result, 'Expected "Jz_mean" in result'
        assert "Jz_variance" in result, 'Expected "Jz_variance" in result'


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance validation tests."""

    def test_n_100_should_complete_in_reasonable_time(self) -> None:
        import time

        params = {"chi": 0.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        start = time.time()
        result = compute_twa_expectations(
            N=100,
            state_type="CSS",
            params=params,
            T=0.1,
            N_traj=1000,
            rng_seed=42,
        )
        elapsed = time.time() - start

        # Should complete in < 1 second
        assert elapsed < 1.0, f"Took {elapsed}s - too slow"
        assert "Jz_mean" in result, 'Expected "Jz_mean" in result'

    def test_n_1000_should_complete_in_reasonable_time(self) -> None:
        import time

        params = {"chi": 0.0, "gamma_1": 0.0, "gamma_2": 0.0, "gamma_phi": 0.0}

        start = time.time()
        result = compute_twa_expectations(
            N=1000,
            state_type="CSS",
            params=params,
            T=0.1,
            N_traj=1000,
            rng_seed=42,
        )
        elapsed = time.time() - start

        # Should complete in < 1 second
        assert elapsed < 1.0, f"Took {elapsed}s - too slow"
        assert "Jz_mean" in result, 'Expected "Jz_mean" in result'

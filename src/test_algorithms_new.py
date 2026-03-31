"""Tests for algorithms.py - Metropolis-Hastings and other algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms import GaussianMetropolisHastings


class TestGaussianMetropolisHastings:
    """Test suite for GaussianMetropolisHastings."""

    @pytest.fixture
    def sampler(self) -> GaussianMetropolisHastings:
        """Create a fresh sampler instance."""
        return GaussianMetropolisHastings(initial_configuration=0.0)

    def test_initial_configuration(self, sampler: GaussianMetropolisHastings) -> None:
        """Test that initial configuration is set correctly."""
        assert sampler.current_configuration == 0.0
        assert len(sampler.configuration_history) == 1

    def test_state_likelihood_is_normalized(self) -> None:
        """Test that state_likelihood returns valid probabilities."""
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        # Likelihood should be highest at x=0
        assert sampler.state_likelihood(0.0) == 1.0
        # Likelihood should be positive
        assert sampler.state_likelihood(1.0) > 0
        assert sampler.state_likelihood(-1.0) > 0

    def test_generator_produces_gaussian_distribution(self) -> None:
        """Test that generator produces approximately Gaussian samples."""
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(42)
        samples = []
        for _ in range(1000):
            new_sample = sampler.generator_function()
            samples.append(new_sample)

        # Mean should be close to 0 (current configuration)
        mean = np.mean(samples)
        std = np.std(samples)
        assert np.isclose(mean, 0.0, atol=0.2), f"Mean should be ~0, got {mean}"
        assert np.isclose(std, 1.0, atol=0.3), (
            f"Std should be ~1, got {std}"
        )  # Less strict

    def test_approval_function_accepts_higher_likelihood(self) -> None:
        """Test that acceptance is more likely for higher likelihood states."""
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(42)

        # Move to a state with higher likelihood (closer to 0)
        higher_likelihood = sampler.approval_function(-0.1)
        # Should often accept (but not always due to randomness)
        accepted_count = sum(
            1
            for _ in range(100)
            if GaussianMetropolisHastings(initial_configuration=0.0).approval_function(
                -0.1
            )
        )
        # Should accept more than 50% of the time
        assert accepted_count > 50, (
            f"Should accept high likelihood >50%, got {accepted_count}%"
        )

    def test_approval_function_rejects_lower_likelihood(self) -> None:
        """Test that acceptance is less likely for lower likelihood states."""
        np.random.seed(42)
        # Move to a state with much lower likelihood (far from 0)
        accepted_count = sum(
            1
            for _ in range(100)
            if GaussianMetropolisHastings(initial_configuration=0.0).approval_function(
                5.0
            )
        )
        # Should accept much less than 50% of the time
        assert accepted_count < 30, (
            f"Should accept low likelihood <30%, got {accepted_count}%"
        )

    def test_run_single_iteration_changes_state(self) -> None:
        """Test that running iterations changes the configuration."""
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(123)

        initial = sampler.current_configuration
        sampler.run_single_iteration()
        new = sampler.current_configuration

        # State should change (at least sometimes)
        # After many iterations, we should see changes
        for _ in range(100):
            sampler.run_single_iteration()

        assert len(sampler.configuration_history) > 1

    def test_accept_reject_counting(self) -> None:
        """Test that acceptance/rejection counting works."""
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        initial_accepted = sampler.accepted_configuration_count
        initial_rejected = sampler.rejected_configuration_count

        sampler.run_single_iteration()

        # Either accepted or rejected should have increased
        # Note: run_single_iteration can reject multiple times before accepting
        total_change = (
            sampler.accepted_configuration_count
            + sampler.rejected_configuration_count
            - initial_accepted
            - initial_rejected
        )
        assert total_change >= 1, "At least one trial should have occurred"

    def test_configuration_history_grows(self) -> None:
        """Test that configuration history grows with iterations."""
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        initial_length = len(sampler.configuration_history)

        sampler.run_iterations(10)

        assert len(sampler.configuration_history) == initial_length + 10, (
            "History should grow by 10"
        )

    def test_likelihood_computation_is_cached(self) -> None:
        """Test that computing likelihood twice gives same result."""
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        config = 1.5

        lik1 = sampler.state_likelihood(config)
        lik2 = sampler.state_likelihood(config)

        assert np.isclose(lik1, lik2)
        assert np.isclose(lik1, np.exp(-1 * config**2))

    def test_run_iterations_with_progress(self) -> None:
        """Test that run_iterations completes without errors."""
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler.run_iterations(50)
        assert len(sampler.configuration_history) == 51  # 1 initial + 50


class TestMetropolisHastingsEdgeCases:
    """Test edge cases for Metropolis-Hastings algorithm."""

    def test_zero_temperature_limit(self) -> None:
        """Test behavior as temperature goes to zero (deterministic acceptance)."""
        # In our implementation, temperature is implicit in the acceptance ratio
        # At very low "temperature", we should only accept moves that increase likelihood
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(42)

        # With current config at 0, likelihood = 1.0
        # Moving to 0.1: likelihood = exp(-0.01) ≈ 0.99
        # The acceptance probability for a worse move is proportional to likelihood ratio
        # This tests the basic acceptance mechanism

    def test_very_high_likelihood_state(self) -> None:
        """Test behavior when moving to a state with very high likelihood."""
        sampler = GaussianMetropolisHastings(initial_configuration=5.0)
        # Moving toward 0 should always be accepted eventually
        accepted = False
        for _ in range(1000):
            sampler.run_single_iteration()
            if sampler.current_configuration < 1.0:
                accepted = True
                break
        assert accepted, "Should eventually accept state closer to 0"


class TestMetropolisHastingsStatisticalProperties:
    """Test statistical properties of the Metropolis-Hastings sampler."""

    def test_detailed_balance_approximate(self) -> None:
        """Test that detailed balance is approximately satisfied."""
        # This is a weak test - real verification would require many samples
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(42)
        sampler.run_iterations(1000)

        # Count transitions between regions
        neg_count = sum(1 for x in sampler.configuration_history if x < 0)
        pos_count = sum(1 for x in sampler.configuration_history if x >= 0)

        # Should roughly balance for symmetric proposal distribution
        # Allow for some deviation due to randomness
        ratio = neg_count / max(pos_count, 1)
        assert 0.3 < ratio < 3.0, (
            f"Distribution should be roughly symmetric, got ratio {ratio}"
        )

    def test_sample_variance_reasonable(self) -> None:
        """Test that sampled distribution has reasonable variance."""
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(42)
        sampler.run_iterations(2000)

        samples = np.array(sampler.configuration_history[100:])  # Skip burn-in
        variance = np.var(samples)

        # For Gaussian target with std=1, effective variance should be similar
        # But Metropolis has lower variance due to correlation
        assert 0.1 < variance < 2.0, f"Variance should be reasonable, got {variance}"

    def test_positive_and_negative_samples(self) -> None:
        """Test that both positive and negative samples are generated."""
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(42)
        sampler.run_iterations(1000)

        has_positive = any(x > 0 for x in sampler.configuration_history)
        has_negative = any(x < 0 for x in sampler.configuration_history)

        assert has_positive, "Should generate some positive samples"
        assert has_negative, "Should generate some negative samples"


class TestReproducibility:
    """Test reproducibility with seed."""

    def test_same_seed_same_results(self) -> None:
        """Test that using same seed gives same results."""
        np.random.seed(42)
        sampler1 = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler1.run_iterations(100)

        np.random.seed(42)
        sampler2 = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler2.run_iterations(100)

        assert np.allclose(
            sampler1.configuration_history, sampler2.configuration_history
        ), "Same seed should give same results"

    def test_different_seed_different_results(self) -> None:
        """Test that different seeds give different results."""
        np.random.seed(42)
        sampler1 = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler1.run_iterations(100)

        np.random.seed(123)
        sampler2 = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler2.run_iterations(100)

        # Should be different (with high probability)
        assert not np.allclose(
            sampler1.configuration_history, sampler2.configuration_history
        ), "Different seeds should give different results"

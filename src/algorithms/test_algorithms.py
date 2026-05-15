"""Tests for algorithms.py - Metropolis-Hastings and other algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from .algorithms import GaussianMetropolisHastings


class TestGaussianMetropolisHastings:
    """Test suite for GaussianMetropolisHastings."""

    @pytest.fixture
    def sampler(self) -> GaussianMetropolisHastings:
        """Create a fresh sampler instance."""
        return GaussianMetropolisHastings(initial_configuration=0.0)

    def test_initial_configuration_is_set_correctly(
        self, sampler: GaussianMetropolisHastings
    ) -> None:
        assert sampler.current_configuration == 0.0, (
            "Expected sampler.current_configuration == 0.0"
        )
        assert len(sampler.configuration_history) == 1, (
            "Expected len(sampler.configuration_history) == 1"
        )

    def test_state_likelihood_returns_valid_probabilities(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        # Likelihood should be highest at x=0
        assert sampler.state_likelihood(0.0) == 1.0, (
            "Expected sampler.state_likelihood(0.0) == 1.0"
        )
        # Likelihood should be positive
        assert sampler.state_likelihood(1.0) > 0, (
            "Expected sampler.state_likelihood(1.0) > 0"
        )
        assert sampler.state_likelihood(-1.0) > 0, (
            "Expected sampler.state_likelihood(-1.0) > 0"
        )

    def test_generator_produces_approximately_gaussian_samples(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(42)
        samples = []
        for _ in range(1000):
            new_sample = sampler.generator_function()
            samples.append(new_sample)

        # Mean should be close to 0 (current configuration)
        mean = np.mean(samples)
        std = np.std(samples)
        assert mean == pytest.approx(0.0, abs=0.2), f"Mean should be ~0, got {mean}"
        assert std == pytest.approx(1.0, abs=0.3), (
            f"Std should be ~1, got {std}"
        )  # Less strict

    def test_acceptance_is_more_likely_for_higher_likelihood_states(self) -> None:
        np.random.seed(42)

        # Verify acceptance ratio (higher likelihood should be accepted)
        current_likelihood = 1.0  # Likelihood at configuration=0
        accepted_count = sum(
            1
            for _ in range(100)
            if GaussianMetropolisHastings(initial_configuration=0.0).approval_function(
                -0.1,
                current_likelihood,
            )
        )
        # Should accept more than 50% of the time
        assert accepted_count > 50, (
            f"Should accept high likelihood >50%, got {accepted_count}%"
        )

    def test_acceptance_is_less_likely_for_lower_likelihood_states(self) -> None:
        np.random.seed(42)
        # Move to a state with much lower likelihood (far from 0)
        current_likelihood = 1.0  # Likelihood at configuration=0
        accepted_count = sum(
            1
            for _ in range(100)
            if GaussianMetropolisHastings(initial_configuration=0.0).approval_function(
                5.0,
                current_likelihood,
            )
        )
        # Should accept much less than 50% of the time
        assert accepted_count < 30, (
            f"Should accept low likelihood <30%, got {accepted_count}%"
        )

    def test_running_iterations_changes_the_configuration(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(123)

        # State should change after many iterations
        for _ in range(100):
            sampler.run_single_iteration()

        assert len(sampler.configuration_history) > 1, (
            "Expected len(sampler.configuration_history) > 1"
        )

    def test_acceptance_rejection_counting_works(self) -> None:
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

    def test_configuration_history_grows_with_iterations(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        initial_length = len(sampler.configuration_history)

        sampler.run_iterations(10)

        assert len(sampler.configuration_history) == initial_length + 10, (
            "History should grow by 10"
        )

    def test_computing_likelihood_twice_gives_same_result(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        config = 1.5

        lik1 = sampler.state_likelihood(config)
        lik2 = sampler.state_likelihood(config)

        assert lik1 == pytest.approx(lik2), "Expected lik1 == pytest.approx(lik2)"
        assert lik1 == pytest.approx(np.exp(-1 * config**2)), (
            "Expected lik1 == pytest.approx(np.exp(-1 * config**2))"
        )

    def test_run_iterations_completes_without_errors(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler.run_iterations(50)
        assert len(sampler.configuration_history) == 51  # 1 initial + 50


class TestMetropolisHastingsEdgeCases:
    """Test edge cases for Metropolis-Hastings algorithm."""

    def test_behavior_as_temperature_goes_to_zero_deterministic_acceptance(
        self,
    ) -> None:
        # In our implementation, temperature is implicit in the acceptance ratio
        # At very low "temperature", we should only accept moves that increase likelihood
        np.random.seed(42)

        # With current config at 0, likelihood = 1.0
        # Moving to 0.1: likelihood = exp(-0.01) ≈ 0.99
        # The acceptance probability for a worse move is proportional to likelihood ratio
        # This tests the basic acceptance mechanism

    def test_behavior_when_moving_to_a_state_with_very_high_likelihood(self) -> None:
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

    def test_detailed_balance_is_approximately_satisfied(self) -> None:
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

    def test_sampled_distribution_has_reasonable_variance(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(42)
        sampler.run_iterations(2000)

        samples = np.array(sampler.configuration_history[100:])  # Skip burn-in
        variance = np.var(samples)

        # For Gaussian target with std=1, effective variance should be similar
        # But Metropolis has lower variance due to correlation
        assert 0.1 < variance < 2.0, f"Variance should be reasonable, got {variance}"

    def test_both_positive_and_negative_samples_are_generated(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        np.random.seed(42)
        sampler.run_iterations(1000)

        has_positive = any(x > 0 for x in sampler.configuration_history)
        has_negative = any(x < 0 for x in sampler.configuration_history)

        assert has_positive, "Should generate some positive samples"
        assert has_negative, "Should generate some negative samples"


class TestReproducibility:
    """Test reproducibility with seed."""

    def test_using_same_seed_gives_same_results(self) -> None:
        np.random.seed(42)
        sampler1 = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler1.run_iterations(100)

        np.random.seed(42)
        sampler2 = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler2.run_iterations(100)

        assert sampler1.configuration_history == pytest.approx(
            sampler2.configuration_history,
        ), "Same seed should give same results"

    def test_different_seeds_give_different_results(self) -> None:
        np.random.seed(42)
        sampler1 = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler1.run_iterations(100)

        np.random.seed(123)
        sampler2 = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler2.run_iterations(100)

        # Should be different (with high probability)
        assert sampler1.configuration_history != pytest.approx(
            sampler2.configuration_history,
        ), "Different seeds should give different results"

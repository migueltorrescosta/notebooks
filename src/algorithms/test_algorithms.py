"""Tests for algorithms.py - Metropolis-Hastings and other algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from .algorithms import GaussianMetropolisHastings


class TestGaussianMetropolisHastings:
    @pytest.fixture
    def sampler(self) -> GaussianMetropolisHastings:
        return GaussianMetropolisHastings(initial_configuration=0.0)

    def test_given_initial_configuration_then_sampler_holds_it(
        self, sampler: GaussianMetropolisHastings
    ) -> None:
        assert sampler.current_configuration == 0.0
        assert len(sampler.configuration_history) == 1

    def test_given_origin_then_likelihood_is_maximal(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        assert sampler.state_likelihood(0.0) == 1.0
        assert sampler.state_likelihood(1.0) > 0
        assert sampler.state_likelihood(-1.0) > 0

    def test_given_gaussian_generator_then_samples_are_normally_distributed(
        self,
    ) -> None:
        rng = np.random.default_rng(42)
        sampler = GaussianMetropolisHastings(initial_configuration=0.0, rng=rng)
        samples = [sampler.generator_function() for _ in range(1000)]

        mean = np.mean(samples)
        std = np.std(samples)
        assert mean == pytest.approx(0.0, abs=0.2)
        assert std == pytest.approx(1.0, abs=0.3)

    def test_given_higher_likelihood_then_acceptance_is_more_likely(self) -> None:
        rng = np.random.default_rng(42)

        current_likelihood = 1.0
        accepted_count = sum(
            1
            for _ in range(100)
            if GaussianMetropolisHastings(initial_configuration=0.0, rng=rng).approval_function(
                -0.1,
                current_likelihood,
            )
        )
        assert accepted_count > 50

    def test_given_lower_likelihood_then_acceptance_is_less_likely(self) -> None:
        rng = np.random.default_rng(42)
        current_likelihood = 1.0
        accepted_count = sum(
            1
            for _ in range(100)
            if GaussianMetropolisHastings(initial_configuration=0.0, rng=rng).approval_function(
                5.0,
                current_likelihood,
            )
        )
        assert accepted_count < 30

    def test_given_iterations_then_configuration_evolves(self) -> None:
        rng = np.random.default_rng(123)
        sampler = GaussianMetropolisHastings(initial_configuration=0.0, rng=rng)

        for _ in range(100):
            sampler.run_single_iteration()

        assert len(sampler.configuration_history) > 1

    def test_given_iteration_then_counters_increment(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        initial_accepted = sampler.accepted_configuration_count
        initial_rejected = sampler.rejected_configuration_count

        sampler.run_single_iteration()

        total_change = (
            sampler.accepted_configuration_count
            + sampler.rejected_configuration_count
            - initial_accepted
            - initial_rejected
        )
        assert total_change >= 1

    def test_given_iterations_then_history_grows(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        initial_length = len(sampler.configuration_history)

        sampler.run_iterations(10)

        assert len(sampler.configuration_history) == initial_length + 10

    def test_given_repeated_calls_then_likelihood_is_deterministic(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        config = 1.5

        lik1 = sampler.state_likelihood(config)
        lik2 = sampler.state_likelihood(config)

        assert lik1 == pytest.approx(lik2)
        assert lik1 == pytest.approx(np.exp(-1 * config**2))

    def test_given_iteration_count_then_run_completes(self) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=0.0)
        sampler.run_iterations(50)
        assert len(sampler.configuration_history) == 51


class TestEdgeCases:
    def test_given_low_temperature_then_acceptance_is_deterministic(self) -> None:
        """Placeholder: acceptance becomes deterministic in low-temperature limit."""

    def test_given_high_likelihood_move_then_acceptance_eventually_succeeds(
        self,
    ) -> None:
        sampler = GaussianMetropolisHastings(initial_configuration=5.0)
        accepted = False
        for _ in range(1000):
            sampler.run_single_iteration()
            if sampler.current_configuration < 1.0:
                accepted = True
                break
        assert accepted


class TestStatisticalProperties:
    def test_given_symmetric_proposal_then_detailed_balance_holds_approximately(
        self,
    ) -> None:
        rng = np.random.default_rng(42)
        sampler = GaussianMetropolisHastings(initial_configuration=0.0, rng=rng)
        sampler.run_iterations(1000)

        neg_count = sum(1 for x in sampler.configuration_history if x < 0)
        pos_count = sum(1 for x in sampler.configuration_history if x >= 0)

        ratio = neg_count / max(pos_count, 1)
        assert 0.3 < ratio < 3.0

    def test_given_gaussian_target_then_variance_is_reasonable(self) -> None:
        rng = np.random.default_rng(42)
        sampler = GaussianMetropolisHastings(initial_configuration=0.0, rng=rng)
        sampler.run_iterations(2000)

        samples = np.array(sampler.configuration_history[100:])
        variance = np.var(samples)

        assert 0.1 < variance < 2.0

    def test_given_many_samples_then_both_signs_appear(self) -> None:
        rng = np.random.default_rng(42)
        sampler = GaussianMetropolisHastings(initial_configuration=0.0, rng=rng)
        sampler.run_iterations(1000)

        has_positive = any(x > 0 for x in sampler.configuration_history)
        has_negative = any(x < 0 for x in sampler.configuration_history)

        assert has_positive
        assert has_negative


class TestReproducibility:
    def test_given_same_seed_then_results_are_identical(self) -> None:
        rng1 = np.random.default_rng(42)
        sampler1 = GaussianMetropolisHastings(initial_configuration=0.0, rng=rng1)
        sampler1.run_iterations(100)

        rng2 = np.random.default_rng(42)
        sampler2 = GaussianMetropolisHastings(initial_configuration=0.0, rng=rng2)
        sampler2.run_iterations(100)

        assert sampler1.configuration_history == pytest.approx(
            sampler2.configuration_history,
        )

    def test_given_different_seeds_then_results_differ(self) -> None:
        rng1 = np.random.default_rng(42)
        sampler1 = GaussianMetropolisHastings(initial_configuration=0.0, rng=rng1)
        sampler1.run_iterations(100)

        rng2 = np.random.default_rng(123)
        sampler2 = GaussianMetropolisHastings(initial_configuration=0.0, rng=rng2)
        sampler2.run_iterations(100)

        assert sampler1.configuration_history != pytest.approx(
            sampler2.configuration_history,
        )

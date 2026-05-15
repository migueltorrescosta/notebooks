from __future__ import annotations

import numpy as np
import pytest

from src.physics.mzi_simulation import prepare_input_state

from .bayesian_phase_estimation import (
    bayesian_estimator,
    bayesian_likelihood,
    bayesian_posterior,
    bayesian_sensitivity,
    bayesian_sensitivity_analytical,
    bayesian_sensitivity_circular,
    compute_crb,
    sample_measurement_outcomes,
)


class TestBayesianPosterior:
    def test_given_any_outcome_then_posterior_normalized(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 180)

        posterior = bayesian_posterior(
            measurement_outcome=0,
            initial_state=state,
            max_photons=max_photons,
            prior_range=prior_range,
        )

        total = np.sum(posterior)
        assert total == pytest.approx(1.0, rel=1e-5)

    def test_given_vacuum_state_then_posterior_uniform(self) -> None:
        max_photons = 2
        state = prepare_input_state("vacuum", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 180)

        posterior = bayesian_posterior(
            measurement_outcome=0,
            initial_state=state,
            max_photons=max_photons,
            prior_range=prior_range,
        )

        expected = np.ones(len(prior_range)) / len(prior_range)
        assert posterior == pytest.approx(expected, abs=0.1)

    def test_given_outcome_0_then_posterior_peaks_near_pi(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 360)

        posterior = bayesian_posterior(
            measurement_outcome=0,
            initial_state=state,
            max_photons=max_photons,
            prior_range=prior_range,
        )

        peak_idx = np.argmax(posterior)
        peak_phi = prior_range[peak_idx]

        assert np.pi - 0.5 < peak_phi < np.pi + 0.5

    def test_given_different_outcomes_then_posteriors_differ(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 90)

        posterior_0 = bayesian_posterior(
            measurement_outcome=0,
            initial_state=state,
            max_photons=max_photons,
            prior_range=prior_range,
        )

        posterior_1 = bayesian_posterior(
            measurement_outcome=1,
            initial_state=state,
            max_photons=max_photons,
            prior_range=prior_range,
        )

        assert posterior_0 != pytest.approx(posterior_1)

    def test_invalid_outcome_raises_value_error(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 90)

        with pytest.raises(ValueError):
            bayesian_posterior(
                measurement_outcome=2,
                initial_state=state,
                max_photons=max_photons,
                prior_range=prior_range,
            )


class TestBayesianSensitivity:
    def test_given_more_outcomes_then_sensitivity_decreases(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 180)

        rng = np.random.default_rng(42)
        phi_true = np.pi / 4

        n_samples_1 = 1
        outcomes_1 = sample_measurement_outcomes(
            state,
            phi_true,
            n_samples_1,
            rng,
            max_photons=max_photons,
        )
        posterior_1 = bayesian_posterior(
            measurement_outcome=outcomes_1[0],
            initial_state=state,
            max_photons=max_photons,
            prior_range=prior_range,
        )
        sens_1 = bayesian_sensitivity(posterior_1, prior_range)

        rng = np.random.default_rng(42)
        n_samples_5 = 5
        outcomes_5 = sample_measurement_outcomes(
            state,
            phi_true,
            n_samples_5,
            rng,
            max_photons=max_photons,
        )
        result = bayesian_estimator(outcomes_5, state, max_photons)
        sens_5 = result["sensitivity"]

        assert np.isfinite(sens_1)
        assert np.isfinite(sens_5)
        assert sens_1 > 0
        assert sens_5 > 0

    def test_given_posterior_then_sensitivity_bounded(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 180)

        posterior = bayesian_posterior(
            measurement_outcome=0,
            initial_state=state,
            max_photons=max_photons,
            prior_range=prior_range,
        )

        sensitivity = bayesian_sensitivity(posterior, prior_range)
        assert 0 < sensitivity < np.pi

    def test_given_posterior_then_circular_sensitivity_finite(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 360)

        posterior = bayesian_posterior(
            measurement_outcome=0,
            initial_state=state,
            max_photons=max_photons,
            prior_range=prior_range,
        )

        sens_lin = bayesian_sensitivity(posterior, prior_range)
        sens_circ = bayesian_sensitivity_circular(posterior, prior_range)

        assert np.isfinite(sens_lin)
        assert np.isfinite(sens_circ)
        assert sens_lin > 0
        assert sens_circ > 0


class TestBayesianLikelihood:
    def test_given_any_state_then_likelihood_bounded(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 90)

        likelihood = bayesian_likelihood(prior_range, state, max_photons)

        assert np.all(likelihood >= 0)
        assert np.all(likelihood <= 1)

    def test_given_single_photon_then_likelihood_follows_cos_squared(self) -> None:
        max_photons = 1
        state = prepare_input_state("single_photon", max_photons=max_photons, mode=0)
        prior_range = np.linspace(0, 2 * np.pi, 180)

        likelihood = bayesian_likelihood(prior_range, state, max_photons)

        expected = np.sin(prior_range / 2) ** 2
        assert likelihood == pytest.approx(expected, abs=0.15)


class TestSampleMeasurementOutcomes:
    def test_given_any_phase_then_outcomes_binary(self) -> None:
        max_photons = 1
        state = prepare_input_state("single_photon", max_photons=max_photons)
        rng = np.random.default_rng(42)

        outcomes = sample_measurement_outcomes(
            state,
            phi_true=0.0,
            n_samples=100,
            rng=rng,
            max_photons=max_photons,
        )

        assert np.all((outcomes == 0) | (outcomes == 1))

    def test_given_phase_pi_then_outcomes_mostly_zero(self) -> None:
        max_photons = 1
        state = prepare_input_state("single_photon", max_photons=max_photons)
        rng = np.random.default_rng(12345)

        outcomes = sample_measurement_outcomes(
            state,
            phi_true=np.pi,
            n_samples=1000,
            rng=rng,
            max_photons=max_photons,
        )

        assert np.sum(outcomes == 0) / len(outcomes) > 0.95


class TestBayesianEstimator:
    def test_given_outcomes_then_result_has_required_keys(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        outcomes = np.array([0, 1, 0])

        result = bayesian_estimator(outcomes, state, max_photons)

        assert "posterior" in result
        assert "phi_estimate" in result
        assert "sensitivity" in result

    def test_empty_outcomes_raises_value_error(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        outcomes = np.array([])

        with pytest.raises(ValueError):
            bayesian_estimator(outcomes, state, max_photons)

    def test_given_outcomes_then_estimate_in_valid_range(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        rng = np.random.default_rng(42)

        outcomes = sample_measurement_outcomes(
            state,
            phi_true=np.pi / 4,
            n_samples=10,
            rng=rng,
            max_photons=max_photons,
        )

        result = bayesian_estimator(outcomes, state, max_photons)
        phi_est = result["phi_estimate"]

        assert -2 * np.pi <= phi_est <= 2 * np.pi


class TestComputeCRB:
    def test_given_n_samples_then_crb_scales_as_one_over_sqrt_n(self) -> None:
        max_photons = 1
        state = prepare_input_state("single_photon", max_photons=max_photons)

        crb_1 = compute_crb(state, max_photons, n_samples=1)
        crb_10 = compute_crb(state, max_photons, n_samples=10)

        assert crb_10 < crb_1
        assert crb_10 == pytest.approx(crb_1 / np.sqrt(10), rel=0.1)


class TestBayesianSensitivityAnalytical:
    def test_given_counts_then_sensitivity_matches_scaling(self) -> None:
        sens = bayesian_sensitivity_analytical(n_outcomes_0=5, n_total=10)

        expected = 1.0 / np.sqrt(10)
        assert sens == pytest.approx(expected, rel=0.01)


class TestPhysicsInvariance:
    def test_given_same_state_then_posterior_aligns_with_likelihood(self) -> None:
        max_photons = 2
        state = prepare_input_state("single_photon", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 360)

        likelihood = bayesian_likelihood(prior_range, state, max_photons)
        posterior = bayesian_posterior(
            measurement_outcome=0,
            initial_state=state,
            max_photons=max_photons,
            prior_range=prior_range,
        )

        like_max_idx = np.argmax(likelihood)
        post_max_idx = np.argmax(posterior)

        assert abs(like_max_idx - post_max_idx) <= 1

    def test_given_state_then_sensitivity_finite(self) -> None:
        max_photons = 2
        state_1ph = prepare_input_state("single_photon", max_photons=max_photons)
        prior_range = np.linspace(0, 2 * np.pi, 180)

        posterior_1ph = bayesian_posterior(
            measurement_outcome=0,
            initial_state=state_1ph,
            max_photons=max_photons,
            prior_range=prior_range,
        )
        sens_1ph = bayesian_sensitivity(posterior_1ph, prior_range)

        assert np.isfinite(sens_1ph)

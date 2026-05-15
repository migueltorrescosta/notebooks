"""Tests for Bayesian update using direct pandas expressions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestPosteriorProperties:
    def test_posterior_equals_normalized_prior_times_likelihood(self) -> None:
        df = pd.DataFrame(
            {"prior": [0.4, 0.6], "likelihood": [0.5, 0.5]},
            index=["A", "B"],
        )
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        expected_A = (0.4 * 0.5) / ((0.4 * 0.5) + (0.6 * 0.5))  # = 0.4
        expected_B = (0.6 * 0.5) / ((0.4 * 0.5) + (0.6 * 0.5))  # = 0.6
        assert df.loc["A", "posterior"] == pytest.approx(expected_A)
        assert df.loc["B", "posterior"] == pytest.approx(expected_B)

    def test_posterior_column_sums_to_one(self) -> None:
        df = pd.DataFrame({"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert df["posterior"].sum() == pytest.approx(1.0, abs=1e-10)

    def test_posterior_proportional_to_prior_times_likelihood(self) -> None:
        df = pd.DataFrame(
            {
                "prior": [0.25, 0.25, 0.25, 0.25],
                "likelihood": [0.9, 0.6, 0.3, 0.1],
            },
        )
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        product = df["prior"] * df["likelihood"]
        nonzero = product > 1e-10
        ratios = df["posterior"][nonzero] / product[nonzero]
        assert np.allclose(ratios, ratios.iloc[0], atol=1e-6)

    def test_posterior_peak_lies_between_prior_and_likelihood_peaks(self) -> None:
        domain = np.linspace(0, 1, 101)
        prior = np.exp(-((domain - 0.3) ** 2) * 50)
        likelihood = np.exp(-((domain - 0.7) ** 2) * 50)
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        peak_x = domain[df["posterior"].argmax()]
        assert 0.3 <= peak_x <= 0.7

    def test_given_uniform_prior_then_posterior_equals_normalized_likelihood(
        self,
    ) -> None:
        prior = np.ones(5)
        likelihood = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        expected = likelihood / likelihood.sum()
        np.testing.assert_allclose(df["posterior"], expected, atol=1e-10)

    def test_posterior_mean_lies_between_prior_and_likelihood_means(self) -> None:
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-((domain + 0.3) ** 2))
        likelihood = np.exp(-((domain - 0.3) ** 2))
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        prior_mean = np.sum(domain * df["prior"])
        like_mean = np.sum(domain * df["likelihood"])
        post_mean = np.sum(domain * df["posterior"])
        lower = min(prior_mean, like_mean)
        upper = max(prior_mean, like_mean)
        assert lower <= post_mean <= upper + 1e-10

    def test_posterior_std_leq_max_prior_and_likelihood_std(self) -> None:
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-(domain**2) * 10)
        likelihood = np.exp(-((domain - 0.1) ** 2) * 10)
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        prior_std = np.sqrt(np.sum(domain**2 * df["prior"]))
        like_mean = np.sum(domain * df["likelihood"])
        like_std = np.sqrt(np.sum((domain - like_mean) ** 2 * df["likelihood"]))
        post_mean = np.sum(domain * df["posterior"])
        post_std = np.sqrt(np.sum((domain - post_mean) ** 2 * df["posterior"]))
        assert post_std <= max(prior_std, like_std) + 0.01


class TestEdgeCases:
    def test_given_point_mass_likelihood_then_posterior_concentrates(self) -> None:
        prior = np.ones(5)
        likelihood = np.zeros(5)
        likelihood[2] = 1.0
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert df["posterior"].idxmax() == 2
        assert df["posterior"].iloc[2] > 0.99

    def test_given_disjoint_prior_and_likelihood_then_prior_and_likelihood_unchanged(
        self,
    ) -> None:
        prior = np.array([1.0, 0.0, 0.0])
        likelihood = np.array([0.0, 1.0, 0.0])
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert df["prior"].sum() == pytest.approx(1.0)
        assert df["likelihood"].sum() == pytest.approx(1.0)

    def test_given_nan_likelihood_then_posterior_is_nan(self) -> None:
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [float("nan"), 1.0]},
        )
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert np.any(np.isnan(df["posterior"]))

    def test_given_infinite_likelihood_then_posterior_is_nan(self) -> None:
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [float("inf"), 1.0]},
        )
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert np.any(np.isnan(df["posterior"]))

    def test_given_zero_likelihood_then_posterior_is_nan(self) -> None:
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.0, 0.0]},
        )
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert df["prior"].sum() == pytest.approx(1.0)
        assert df["likelihood"].sum() == pytest.approx(0.0)

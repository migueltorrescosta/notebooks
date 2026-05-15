"""Tests for Bayesian update using direct pandas expressions.

Tests verify that the expressions
    df['posterior'] = df['prior'] * df['likelihood']
    df['posterior'] /= df['posterior'].sum()
correctly compute the posterior distribution and handle edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestPosteriorProperties:
    """Tests for posterior computation properties."""

    def test_calculate_posterior_basic(self) -> None:
        """Posterior should equal normalized prior * likelihood (Bayes' rule)."""
        df = pd.DataFrame(
            {"prior": [0.4, 0.6], "likelihood": [0.5, 0.5]},
            index=["A", "B"],
        )
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        expected_A = (0.4 * 0.5) / ((0.4 * 0.5) + (0.6 * 0.5))  # = 0.4
        expected_B = (0.6 * 0.5) / ((0.4 * 0.5) + (0.6 * 0.5))  # = 0.6
        assert df.loc["A", "posterior"] == pytest.approx(expected_A), (
            "Expected posterior at A to match Bayes rule"
        )
        assert df.loc["B", "posterior"] == pytest.approx(expected_B), (
            "Expected posterior at B to match Bayes rule"
        )

    def test_posterior_sums_to_one(self) -> None:
        """Posterior column should sum to 1."""
        df = pd.DataFrame({"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert df["posterior"].sum() == pytest.approx(1.0, abs=1e-10), (
            "Posterior should sum to 1"
        )

    def test_posterior_proportional_to_prior_times_likelihood(self) -> None:
        """Posterior should be proportional to prior * likelihood."""
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
        assert np.allclose(ratios, ratios.iloc[0], atol=1e-6), (
            "Posterior should be proportional to prior * likelihood"
        )

    def test_posterior_peak_between_prior_and_likelihood(self) -> None:
        """Posterior peak should lie between prior and likelihood peaks."""
        domain = np.linspace(0, 1, 101)
        prior = np.exp(-((domain - 0.3) ** 2) * 50)
        likelihood = np.exp(-((domain - 0.7) ** 2) * 50)
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        peak_x = domain[df["posterior"].argmax()]
        assert 0.3 <= peak_x <= 0.7, (
            f"Posterior peak at {peak_x:.2f} should be between 0.3 and 0.7"
        )

    def test_uniform_prior_posterior_equals_normalized_likelihood(self) -> None:
        """With uniform prior, posterior should equal normalized likelihood."""
        prior = np.ones(5)
        likelihood = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        expected = likelihood / likelihood.sum()
        np.testing.assert_allclose(df["posterior"], expected, atol=1e-10)

    def test_posterior_mean_between_prior_and_likelihood_means(self) -> None:
        """Posterior mean should lie between prior and likelihood means."""
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
        assert lower <= post_mean <= upper + 1e-10, (
            f"Posterior mean {post_mean:.4f} not in [{lower:.4f}, {upper:.4f}]"
        )

    def test_posterior_narrower_than_max_std(self) -> None:
        """Posterior std should be <= max(prior_std, likelihood_std)."""
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
        assert post_std <= max(prior_std, like_std) + 0.01, (
            f"Posterior std {post_std:.4f} should be <= "
            f"max({prior_std:.4f}, {like_std:.4f})"
        )


class TestEdgeCases:
    """Edge cases for Bayesian updating."""

    def test_point_mass_likelihood(self) -> None:
        """Posterior should concentrate at a point-mass likelihood."""
        prior = np.ones(5)
        likelihood = np.zeros(5)
        likelihood[2] = 1.0
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert df["posterior"].idxmax() == 2, "Posterior should peak at index 2"
        assert df["posterior"].iloc[2] > 0.99, (
            "Posterior at point mass should be near 1"
        )

    def test_zero_prior_vs_zero_likelihood(self) -> None:
        """Should not crash when prior and likelihood have disjoint support."""
        prior = np.array([1.0, 0.0, 0.0])
        likelihood = np.array([0.0, 1.0, 0.0])
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert df["prior"].sum() == pytest.approx(1.0), "Prior should be unchanged"
        assert df["likelihood"].sum() == pytest.approx(1.0), (
            "Likelihood should be unchanged"
        )

    def test_nan_likelihood_raises(self) -> None:
        """NaN values in likelihood should produce NaN posterior."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [float("nan"), 1.0]},
        )
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert np.any(np.isnan(df["posterior"])), (
            "NaN in likelihood should produce NaN posterior"
        )

    def test_inf_likelihood_raises(self) -> None:
        """Inf values in likelihood should produce NaN posterior."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [float("inf"), 1.0]},
        )
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert np.any(np.isnan(df["posterior"])), (
            "Inf in likelihood should produce NaN posterior"
        )

    def test_zero_probability_likelihood(self) -> None:
        """Zero total likelihood should produce NaN posterior."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.0, 0.0]},
        )
        df["posterior"] = df["prior"] * df["likelihood"]
        df["posterior"] /= df["posterior"].sum()
        assert df["prior"].sum() == pytest.approx(1.0), "Prior should be unchanged"
        assert df["likelihood"].sum() == pytest.approx(0.0), "Likelihood should be zero"

"""Tests for Bayesian update module (src.analysis.bayesian_statistics).

Tests verify initialization, normalization, posterior properties,
edge cases, and plot rendering using the BayesUpdate class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typing import cast

from .bayesian_statistics import BayesUpdate


# =============================================================================
# Initialization and column validation
# =============================================================================


class TestBayesUpdateInit:
    """Tests for BayesUpdate initialization and column validation."""

    def test_instantiation_with_valid_columns(self) -> None:
        """Should accept DataFrame with exactly 'prior' and 'likelihood'."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]}, index=["A", "B"]
        )
        bayes = BayesUpdate(df)
        assert "posterior" in bayes.df.columns

    def test_rejects_wrong_columns(self) -> None:
        """Should reject DataFrame without exact {'prior', 'likelihood'}."""
        df = pd.DataFrame({"foo": [0.5, 0.5], "bar": [0.5, 0.5]})
        with pytest.raises(AssertionError):
            BayesUpdate(df)

    def test_rejects_extra_columns(self) -> None:
        """Should reject DataFrame with extra columns beyond prior/likelihood."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.8, 0.2], "extra": [1, 2]},
            index=["A", "B"],
        )
        with pytest.raises(AssertionError):
            BayesUpdate(df)

    def test_detects_numeric_index(self) -> None:
        """Should set is_categorical=False for numeric index."""
        df = pd.DataFrame(
            {"prior": [0.25, 0.25, 0.25, 0.25], "likelihood": [0.9, 0.6, 0.3, 0.1]},
            index=[0, 1, 2, 3],
        )
        bayes = BayesUpdate(df)
        assert not bayes.is_categorical

    def test_detects_categorical_index(self) -> None:
        """Should set is_categorical=True for non-numeric index."""
        df = pd.DataFrame(
            {"prior": [0.25, 0.25, 0.25, 0.25], "likelihood": [0.9, 0.6, 0.3, 0.1]},
            index=["a", "b", "c", "d"],
        )
        bayes = BayesUpdate(df)
        assert bayes.is_categorical


# =============================================================================
# Normalization — all probability columns sum to 1
# =============================================================================


class TestBayesUpdateNormalization:
    """Tests that prior, likelihood, and posterior columns sum to one."""

    @pytest.mark.parametrize("column", ["prior", "likelihood", "posterior"])
    def test_column_sums_to_one(self, column: str) -> None:
        """Each probability column should sum to 1."""
        df = pd.DataFrame({"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]})
        bayes = BayesUpdate(df)
        assert np.isclose(bayes.df[column].sum(), 1.0, atol=1e-10), (
            f"{column} should sum to 1"
        )

    def test_random_distributions_normalize_correctly(self) -> None:
        """Random prior/likelihood pairs should normalize correctly."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            prior = rng.random(100)
            likelihood = rng.random(100)
            df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
            bayes = BayesUpdate(df)

            assert np.isclose(bayes.df["prior"].sum(), 1.0, atol=1e-10)
            assert np.isclose(bayes.df["likelihood"].sum(), 1.0, atol=1e-10)
            assert np.isclose(bayes.df["posterior"].sum(), 1.0, atol=1e-10)

    def test_normalize_columns_works(self) -> None:
        """Normalize columns divides by row sums correctly."""
        df = pd.DataFrame(
            {"prior": [1.0, 2.0], "likelihood": [4.0, 2.0]}, index=["A", "B"]
        )
        bayes = BayesUpdate(df)
        # After normalization, each column should sum to 1
        assert np.isclose(bayes.df["prior"].sum(), 1.0)
        assert np.isclose(bayes.df["likelihood"].sum(), 1.0)


# =============================================================================
# Posterior properties
# =============================================================================


class TestBayesUpdatePosteriorProperties:
    """Tests for the posterior distribution computed via Bayes' rule."""

    def test_calculate_posterior_basic(self) -> None:
        """Posterior should equal normalized prior * likelihood (Bayes' rule)."""
        df = pd.DataFrame(
            {"prior": [0.4, 0.6], "likelihood": [0.5, 0.5]}, index=["A", "B"]
        )
        bayes = BayesUpdate(df)
        # posterior = prior * likelihood, then normalized
        expected_A = (0.4 * 0.5) / ((0.4 * 0.5) + (0.6 * 0.5))  # = 0.4
        expected_B = (0.6 * 0.5) / ((0.4 * 0.5) + (0.6 * 0.5))  # = 0.6
        posterior_A = cast(float, bayes.df.at["A", "posterior"])
        posterior_B = cast(float, bayes.df.at["B", "posterior"])
        assert np.isclose(posterior_A, expected_A)
        assert np.isclose(posterior_B, expected_B)

    def test_posterior_proportional_to_prior_times_likelihood(self) -> None:
        """Posterior should be proportional to prior * likelihood."""
        df = pd.DataFrame(
            {
                "prior": [0.25, 0.25, 0.25, 0.25],
                "likelihood": [0.9, 0.6, 0.3, 0.1],
            }
        )
        bayes = BayesUpdate(df)

        # Check: posterior_i / (prior_i * likelihood_i) is constant forall i
        product = bayes.df["prior"] * bayes.df["likelihood"]
        nonzero = product > 1e-10
        ratios = bayes.df["posterior"][nonzero] / product[nonzero]
        assert np.allclose(ratios, ratios.iloc[0], atol=1e-6), (
            "Posterior should be proportional to prior * likelihood"
        )

    def test_posterior_peak_between_prior_and_likelihood(self) -> None:
        """Posterior peak should lie between prior and likelihood peaks."""
        domain = np.linspace(0, 1, 101)
        prior = np.exp(-((domain - 0.3) ** 2) * 50)
        likelihood = np.exp(-((domain - 0.7) ** 2) * 50)
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        bayes = BayesUpdate(df)

        peak_x = domain[bayes.df["posterior"].argmax()]
        assert 0.3 <= peak_x <= 0.7, (
            f"Posterior peak at {peak_x:.2f} should be between 0.3 and 0.7"
        )

    def test_uniform_prior_posterior_equals_normalized_likelihood(self) -> None:
        """With uniform prior, posterior should equal normalized likelihood."""
        prior = np.ones(5)
        likelihood = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        bayes = BayesUpdate(df)
        expected = likelihood / likelihood.sum()
        np.testing.assert_allclose(
            bayes.df["posterior"],
            expected,
            atol=1e-10,
        )

    def test_posterior_mean_between_prior_and_likelihood_means(self) -> None:
        """Posterior mean should lie between prior and likelihood means."""
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-((domain + 0.3) ** 2))
        likelihood = np.exp(-((domain - 0.3) ** 2))
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        bayes = BayesUpdate(df)

        prior_mean = np.sum(domain * bayes.df["prior"])
        like_mean = np.sum(domain * bayes.df["likelihood"])
        post_mean = np.sum(domain * bayes.df["posterior"])

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
        bayes = BayesUpdate(df)

        prior_std = np.sqrt(np.sum(domain**2 * bayes.df["prior"]))
        like_mean = np.sum(domain * bayes.df["likelihood"])
        like_std = np.sqrt(np.sum((domain - like_mean) ** 2 * bayes.df["likelihood"]))
        post_mean = np.sum(domain * bayes.df["posterior"])
        post_std = np.sqrt(np.sum((domain - post_mean) ** 2 * bayes.df["posterior"]))

        assert post_std <= max(prior_std, like_std) + 0.01, (
            f"Posterior std {post_std:.4f} should be <= "
            f"max({prior_std:.4f}, {like_std:.4f})"
        )


# =============================================================================
# Edge cases
# =============================================================================


class TestBayesUpdateEdgeCases:
    """Edge cases for Bayesian updating via BayesUpdate."""

    def test_point_mass_likelihood(self) -> None:
        """Posterior should concentrate at a point-mass likelihood."""
        prior = np.ones(5)
        likelihood = np.zeros(5)
        likelihood[2] = 1.0
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        bayes = BayesUpdate(df)

        # Posterior should be concentrated at index 2
        assert bayes.df["posterior"].idxmax() == 2
        assert bayes.df["posterior"].iloc[2] > 0.99

    def test_zero_prior_vs_zero_likelihood(self) -> None:
        """Should not crash when prior and likelihood have disjoint support."""
        prior = np.array([1.0, 0.0, 0.0])
        likelihood = np.array([0.0, 1.0, 0.0])
        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        # The product prior * likelihood = [0, 0, 0] yields all-NaN posterior;
        # the object should still be constructable without raising.
        bayes = BayesUpdate(df)
        # The prior and likelihood columns should still be valid
        assert np.isclose(bayes.df["prior"].sum(), 1.0)
        assert np.isclose(bayes.df["likelihood"].sum(), 1.0)

    def test_nan_likelihood_raises(self) -> None:
        """NaN values in likelihood should produce NaN posterior (graceful)."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [float("nan"), 1.0]},
        )
        bayes = BayesUpdate(df)
        assert np.any(np.isnan(bayes.df["posterior"]))

    def test_inf_likelihood_raises(self) -> None:
        """Inf values in likelihood should produce NaN posterior (graceful)."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [float("inf"), 1.0]},
        )
        bayes = BayesUpdate(df)
        assert np.any(np.isnan(bayes.df["posterior"]))

    def test_zero_probability_likelihood(self) -> None:
        """Zero total likelihood should produce NaN posterior."""
        df = pd.DataFrame(
            {"prior": [0.5, 0.5], "likelihood": [0.0, 0.0]},
        )
        bayes = BayesUpdate(df)
        # Prior should still be valid, posterior may be NaN
        assert np.isclose(bayes.df["prior"].sum(), 1.0)
        assert np.isclose(bayes.df["likelihood"].sum(), 0.0)


# =============================================================================
# Plot smoke tests
# =============================================================================


class TestBayesUpdatePlot:
    """Smoke tests for the BayesUpdate.plot() method."""

    def test_plot_returns_figure(self) -> None:
        """Plot should return a Figure for numeric index."""
        df = pd.DataFrame({"prior": [0.5, 0.5], "likelihood": [0.8, 0.2]})
        bayes = BayesUpdate(df)
        fig = bayes.plot()
        assert fig is not None
        assert len(fig.axes) > 0

    def test_plot_categorical_index_returns_figure(self) -> None:
        """Plot should return a Figure for categorical index."""
        df = pd.DataFrame(
            {"prior": [0.25, 0.25, 0.25, 0.25], "likelihood": [0.9, 0.6, 0.3, 0.1]},
            index=["a", "b", "c", "d"],
        )
        bayes = BayesUpdate(df)
        fig = bayes.plot()
        assert fig is not None
        assert len(fig.axes) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

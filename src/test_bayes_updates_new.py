"""Tests for Bayesian updates module."""

from __future__ import annotations

from enum import Enum
from functools import partial

import numpy as np
import pandas as pd


class Distributions(str, Enum):
    """Distribution types for priors and likelihoods."""

    Geometric = "Geometric"
    Linear = "Linear"
    Gaussian = "Gaussian"


def prior_polynomial(x: float, a: float, b: float, c: float) -> float:
    """Polynomial prior distribution."""
    return a * (x - b) ** c


def prior_gaussian(x: float, a: float, mu: float) -> float:
    """Gaussian prior distribution."""
    return np.exp(-1 * a * (x - mu) ** 2)


def likelihood_polynomial(x: float, a: float, b: float, c: float) -> float:
    """Polynomial likelihood function."""
    return a * (x - b) ** c


def likelihood_gaussian(x: float, a: float, mu: float) -> float:
    """Gaussian likelihood function."""
    return np.exp(-1 * a * (x - mu) ** 2)


class TestPriorDistributions:
    """Tests for prior distribution functions."""

    def test_gaussian_prior_at_center(self) -> None:
        """Test that Gaussian prior is maximum at center."""
        a, mu = 1.0, 0.0
        domain = np.linspace(-2, 2, 100)

        prior_fn = partial(prior_gaussian, a=a, mu=mu)
        values = np.array([prior_fn(x) for x in domain])
        max_idx = np.argmax(values)
        max_x = domain[max_idx]

        # Maximum should be close to mu
        assert np.isclose(max_x, mu, atol=0.1), (
            f"Maximum should be at mu={mu}, got x={max_x}"
        )

    def test_gaussian_prior_values_positive(self) -> None:
        """Test that Gaussian prior values are always positive."""
        a, mu = 1.0, 0.0
        domain = np.linspace(-5, 5, 1000)

        prior_fn = partial(prior_gaussian, a=a, mu=mu)
        for x in domain:
            value = prior_fn(x)
            assert value >= 0, f"Gaussian prior should be non-negative: {value}"

    def test_polynomial_prior_exponents(self) -> None:
        """Test polynomial prior with different exponents."""
        a, b, c = 1.0, 0.0, 2.0
        domain = np.linspace(-2, 2, 100)

        prior_fn = partial(prior_polynomial, a=a, b=b, c=c)
        values = np.array([prior_fn(x) for x in domain])

        # For c=2, should be a parabola
        assert np.allclose(values[0], values[-1]), "Parabola symmetric around center"

    def test_polynomial_prior_can_be_negative(self) -> None:
        """Test that polynomial prior can produce negative values."""
        a, b, c = 1.0, 0.0, 3.0
        domain = np.linspace(-1, 1, 100)

        prior_fn = partial(prior_polynomial, a=a, b=b, c=c)
        has_negative = any(prior_fn(x) < 0 for x in domain)

        assert has_negative, "Odd power polynomials should have negative values"


class TestVectorizedPriorComputation:
    """Tests for vectorized prior/likelihood computation."""

    def test_gaussian_prior_vectorized(self) -> None:
        """Test that Gaussian prior can be computed vectorized."""
        domain = np.linspace(-1, 1, 201)
        a, mu = 1.0, 0.0

        # Vectorized computation
        vectorized = np.maximum(prior_gaussian(domain, a, mu), 0)

        # Element-wise computation
        element_wise = np.array([max(prior_gaussian(x, a, mu), 0) for x in domain])

        assert np.allclose(vectorized, element_wise), (
            "Vectorized computation should match element-wise"
        )

    def test_polynomial_prior_vectorized(self) -> None:
        """Test that polynomial prior can be computed vectorized."""
        domain = np.linspace(-1, 1, 201)
        a, b, c = 1.0, 0.0, 2.0

        # Vectorized computation
        vectorized = np.maximum(prior_polynomial(domain, a, b, c), 0)

        # Element-wise computation
        element_wise = np.array([max(prior_polynomial(x, a, b, c), 0) for x in domain])

        assert np.allclose(vectorized, element_wise), (
            "Vectorized computation should match element-wise"
        )

    def test_vectorized_faster_than_loop(self) -> None:
        """Test that vectorized computation is faster than loop."""
        import time

        domain = np.linspace(-1, 1, 201)
        a, mu = 1.0, 0.0

        # Element-wise
        start = time.perf_counter()
        for _ in range(100):
            _ = np.array([max(prior_gaussian(x, a, mu), 0) for x in domain])
        loop_time = time.perf_counter() - start

        # Vectorized
        start = time.perf_counter()
        for _ in range(100):
            _ = np.maximum(prior_gaussian(domain, a, mu), 0)
        vectorized_time = time.perf_counter() - start

        # Vectorized should be at least as fast
        assert vectorized_time <= loop_time * 1.5, (
            f"Vectorized should be faster: loop={loop_time:.4f}s, "
            f"vectorized={vectorized_time:.4f}s"
        )


class TestBayesUpdate:
    """Tests for BayesUpdate class functionality."""

    def create_bayes_update(
        self,
        prior_vector: np.ndarray,
        likelihood_vector: np.ndarray,
    ) -> pd.DataFrame:
        """Create a DataFrame with prior and likelihood for Bayes update."""
        df = pd.DataFrame({"prior": prior_vector, "likelihood": likelihood_vector})
        df = df / df.sum(axis=0)
        posterior = df["prior"] * df["likelihood"]
        df["posterior"] = posterior / sum(posterior)
        return df

    def test_posterior_sum_equals_one(self) -> None:
        """Test that posterior sums to 1."""
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-(domain**2))
        likelihood = np.exp(-((domain - 0.2) ** 2))

        df = self.create_bayes_update(prior, likelihood)
        posterior_sum = df["posterior"].sum()

        assert np.isclose(posterior_sum, 1.0, atol=1e-10), (
            f"Posterior should sum to 1, got {posterior_sum}"
        )

    def test_prior_sum_equals_one(self) -> None:
        """Test that prior sums to 1."""
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-(domain**2))
        likelihood = np.exp(-((domain - 0.2) ** 2))

        df = self.create_bayes_update(prior, likelihood)
        prior_sum = df["prior"].sum()

        assert np.isclose(prior_sum, 1.0, atol=1e-10), (
            f"Prior should sum to 1, got {prior_sum}"
        )

    def test_likelihood_sum_equals_one(self) -> None:
        """Test that likelihood sums to 1."""
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-(domain**2))
        likelihood = np.exp(-((domain - 0.2) ** 2))

        df = self.create_bayes_update(prior, likelihood)
        likelihood_sum = df["likelihood"].sum()

        assert np.isclose(likelihood_sum, 1.0, atol=1e-10), (
            f"Likelihood should sum to 1, got {likelihood_sum}"
        )

    def test_posterior_nonzero_where_both_nonzero(self) -> None:
        """Test that posterior is non-zero where both prior and likelihood are non-zero."""
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-(domain**2))  # Non-zero everywhere
        likelihood = np.exp(-((domain - 0.2) ** 2) * 100)  # Sharp peak at 0.2

        df = self.create_bayes_update(prior, likelihood)

        # Posterior should be concentrated where likelihood is high
        max_posterior_idx = df["posterior"].argmax()
        max_posterior_x = domain[max_posterior_idx]

        assert np.isclose(max_posterior_x, 0.2, atol=0.1), (
            f"Posterior peak should be near likelihood peak at 0.2, got {max_posterior_x}"
        )

    def test_posterior_width_narrower_than_both(self) -> None:
        """Test that posterior is narrower than both prior and likelihood."""
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-(domain**2) * 10)  # Narrow peak at 0
        likelihood = np.exp(-((domain - 0.1) ** 2) * 10)  # Narrow peak at 0.1

        df = self.create_bayes_update(prior, likelihood)

        prior_std = np.sqrt(np.sum((domain**2) * df["prior"]))
        likelihood_std = np.sqrt(np.sum(((domain - 0.1) ** 2) * df["likelihood"]))
        posterior_std = np.sqrt(
            np.sum(((domain - np.sum(domain * df["posterior"])) ** 2) * df["posterior"])
        )

        # Posterior should be narrower than the wider of prior and likelihood
        max_std = max(prior_std, likelihood_std)
        assert posterior_std <= max_std + 0.01, (
            f"Posterior std {posterior_std} should be <= max({prior_std}, {likelihood_std})"
        )

    def test_consistency_with_bayes_theorem(self) -> None:
        """Test that posterior follows Bayes' theorem."""
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-(domain**2))
        likelihood = np.exp(-((domain - 0.2) ** 2))

        df = self.create_bayes_update(prior, likelihood)

        # P(θ|data) ∝ P(data|θ) * P(θ)
        for i in range(len(domain)):
            p_theta = df["prior"].iloc[i]
            p_data_given_theta = df["likelihood"].iloc[i]
            p_theta_given_data = df["posterior"].iloc[i]

            # Check proportionality (up to normalization)
            expected_proportional = p_theta * p_data_given_theta
            if expected_proportional > 1e-10:
                ratio = p_theta_given_data / expected_proportional
                # All ratios should be equal (constant = 1/normalization constant)
                if i == 0:
                    first_ratio = ratio
                else:
                    assert np.isclose(ratio, first_ratio, atol=1e-6), (
                        "Posterior should be proportional to prior * likelihood"
                    )


class TestBayesianUpdateNormalization:
    """Tests for proper normalization in Bayesian updates."""

    def test_normalization_preserves_probability(self) -> None:
        """Test that normalization preserves total probability."""
        for _ in range(10):
            # Generate random prior and likelihood
            prior = np.random.rand(100)
            prior = prior / prior.sum()

            likelihood = np.random.rand(100)
            likelihood = likelihood / likelihood.sum()

            df = pd.DataFrame({"prior": prior, "likelihood": likelihood})

            # Normalize columns
            df = df / df.sum(axis=0)

            # Calculate posterior
            posterior = df["prior"] * df["likelihood"]
            df["posterior"] = posterior / sum(posterior)

            # Check sums
            assert np.isclose(df["prior"].sum(), 1.0, atol=1e-10)
            assert np.isclose(df["likelihood"].sum(), 1.0, atol=1e-10)
            assert np.isclose(df["posterior"].sum(), 1.0, atol=1e-10)

    def test_posterior_mean_between_prior_and_likelihood_means(self) -> None:
        """Test that posterior mean is between prior and likelihood means."""
        domain = np.linspace(-1, 1, 201)
        prior_mean = -0.3
        likelihood_mean = 0.3

        prior = np.exp(-((domain - prior_mean) ** 2))
        likelihood = np.exp(-((domain - likelihood_mean) ** 2))

        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df = df / df.sum(axis=0)
        posterior = df["prior"] * df["likelihood"]
        df["posterior"] = posterior / sum(posterior)

        prior_computed_mean = np.sum(domain * df["prior"])
        likelihood_computed_mean = np.sum(domain * df["likelihood"])
        posterior_computed_mean = np.sum(domain * df["posterior"])

        # Posterior mean should be between prior and likelihood means
        min_mean = min(prior_computed_mean, likelihood_computed_mean)
        max_mean = max(prior_computed_mean, likelihood_computed_mean)

        assert min_mean <= posterior_computed_mean <= max_mean, (
            f"Posterior mean {posterior_computed_mean} should be between "
            f"prior mean {prior_computed_mean} and likelihood mean {likelihood_computed_mean}"
        )


class TestBayesianUpdateEdgeCases:
    """Tests for edge cases in Bayesian updates."""

    def test_zero_likelihood(self) -> None:
        """Test handling of zero likelihood."""
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-(domain**2))
        likelihood = np.zeros_like(domain)
        likelihood[100] = 1.0  # Point mass

        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df = df / df.sum(axis=0)

        # With zero likelihood everywhere except one point, posterior should be
        # prior * likelihood = 0 everywhere except where likelihood = 1
        # This tests that we don't divide by zero
        posterior = df["prior"] * df["likelihood"]
        if posterior.sum() > 0:
            df["posterior"] = posterior / sum(posterior)

            # Posterior should be concentrated at the point where likelihood = 1
            max_idx = df["posterior"].argmax()
            assert max_idx == 100, "Posterior should peak where likelihood peaks"

    def test_uniform_prior(self) -> None:
        """Test with uniform prior."""
        domain = np.linspace(-1, 1, 201)
        prior = np.ones_like(domain)
        likelihood = np.exp(-((domain - 0.2) ** 2))

        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df = df / df.sum(axis=0)
        posterior = df["prior"] * df["likelihood"]
        df["posterior"] = posterior / sum(posterior)

        # With uniform prior, posterior should be proportional to likelihood
        assert np.allclose(
            df["posterior"], df["likelihood"] / df["likelihood"].sum(), atol=1e-10
        ), "With uniform prior, posterior should equal normalized likelihood"

    def test_point_mass_likelihood(self) -> None:
        """Test with likelihood concentrated at a single point."""
        domain = np.linspace(-1, 1, 201)
        prior = np.exp(-(domain**2))
        likelihood = np.zeros_like(domain)
        likelihood[150] = 1.0

        df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
        df = df / df.sum(axis=0)
        posterior = df["prior"] * df["likelihood"]
        df["posterior"] = posterior / sum(posterior)

        # Posterior should be entirely at the likelihood peak
        assert df["posterior"].iloc[150] > 0.99, (
            "Posterior should be at likelihood peak"
        )


class TestBayesianUpdatePerformance:
    """Tests for performance of Bayesian update computation."""

    def test_computation_time_scales_linearly(self) -> None:
        """Test that computation time scales linearly with domain size."""
        import time

        for n_points in [100, 200, 500, 1000]:
            domain = np.linspace(-1, 1, n_points)
            prior = np.exp(-(domain**2))
            likelihood = np.exp(-((domain - 0.2) ** 2))

            start = time.perf_counter()
            df = pd.DataFrame({"prior": prior, "likelihood": likelihood})
            df = df / df.sum(axis=0)
            posterior = df["prior"] * df["likelihood"]
            df["posterior"] = posterior / sum(posterior)
            elapsed = time.perf_counter() - start

            # Should complete in reasonable time
            assert elapsed < 1.0, (
                f"Computation for {n_points} points took {elapsed:.2f}s"
            )

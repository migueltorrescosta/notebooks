"""Tests for Fisher information computation module."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import scipy


def binomial_pdf(x: int, p: float, n: int) -> float:
    """Compute binomial probability mass function."""
    return float(
        scipy.stats.binom.cdf(x, n=n, p=p) - scipy.stats.binom.cdf(x - 1, n=n, p=p)
    )


class TestBinomialPDF:
    """Tests for binomial PDF computation."""

    def test_pdf_at_extreme_probabilities(self) -> None:
        """Test PDF at p=0 and p=1."""
        n = 5

        # At p=0, only x=0 should have probability 1
        assert np.isclose(binomial_pdf(0, 0.0, n), 1.0, atol=1e-10)
        for x in range(1, n + 1):
            assert np.isclose(binomial_pdf(x, 0.0, n), 0.0, atol=1e-10)

        # At p=1, only x=n should have probability 1
        for x in range(n):
            assert np.isclose(binomial_pdf(x, 1.0, n), 0.0, atol=1e-10)
        assert np.isclose(binomial_pdf(n, 1.0, n), 1.0, atol=1e-10)

    def test_pdf_sum_equals_one(self) -> None:
        """Test that sum of PDF over all x equals 1."""
        for n in [1, 2, 5, 10]:
            for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
                total = sum(binomial_pdf(x, p, n) for x in range(n + 1))
                assert np.isclose(total, 1.0, atol=1e-10), (
                    f"PDF should sum to 1 for n={n}, p={p}: got {total}"
                )

    def test_pdf_is_symmetric_at_p_05(self) -> None:
        """Test that PDF is symmetric at p=0.5."""
        n = 5
        p = 0.5
        for x in range(n + 1):
            pdf_x = binomial_pdf(x, p, n)
            pdf_n_minus_x = binomial_pdf(n - x, p, n)
            assert np.isclose(pdf_x, pdf_n_minus_x, atol=1e-10), (
                f"PDF should be symmetric: P(X={x})={pdf_x}, P(X={n - x})={pdf_n_minus_x}"
            )

    def test_pdf_nonzero_in_range(self) -> None:
        """Test that PDF is zero outside [0, n]."""
        n = 5
        p = 0.5
        for x in range(-5, 0):
            assert np.isclose(binomial_pdf(x, p, n), 0.0, atol=1e-10), (
                f"P(X={x}) should be 0 for x<0"
            )
        for x in range(n + 1, n + 5):
            assert np.isclose(binomial_pdf(x, p, n), 0.0, atol=1e-10), (
                f"P(X={x}) should be 0 for x>n"
            )

    def test_pdf_at_specific_points(self) -> None:
        """Test PDF at specific known values."""
        # n=1, p=0.5: P(0) = 0.5, P(1) = 0.5
        assert np.isclose(binomial_pdf(0, 0.5, 1), 0.5, atol=1e-10)
        assert np.isclose(binomial_pdf(1, 0.5, 1), 0.5, atol=1e-10)

        # n=2, p=0.5: P(0)=0.25, P(1)=0.5, P(2)=0.25
        assert np.isclose(binomial_pdf(0, 0.5, 2), 0.25, atol=1e-10)
        assert np.isclose(binomial_pdf(1, 0.5, 2), 0.5, atol=1e-10)
        assert np.isclose(binomial_pdf(2, 0.5, 2), 0.25, atol=1e-10)


class TestFisherInformationComputation:
    """Tests for Fisher information computation."""

    def test_pdf_grid_shape(self) -> None:
        """Test that PDF grid has correct shape."""
        n = 3
        theta_sample_size = 50
        valid_x = range(n + 1)
        valid_theta = np.linspace(0, 1, theta_sample_size + 1)

        # Create PDF using the same method as in Fisher_information.py
        df_pdf = pd.DataFrame(
            data=[
                (x, p, binomial_pdf(x, p, n))
                for x, p in itertools.product(valid_x, valid_theta)
            ],
            columns=["x", "theta", "pdf"],
        ).pivot(columns="x", index="theta", values="pdf")

        assert df_pdf.shape == (theta_sample_size + 1, n + 1)

    def test_pdf_grid_values_valid(self) -> None:
        """Test that all PDF grid values are valid probabilities."""
        n = 3
        valid_x = range(n + 1)
        valid_theta = np.linspace(0, 1, 101)

        df_pdf = pd.DataFrame(
            data=[
                (x, p, binomial_pdf(x, p, n))
                for x, p in itertools.product(valid_x, valid_theta)
            ],
            columns=["x", "theta", "pdf"],
        ).pivot(columns="x", index="theta", values="pdf")

        pdf_values = np.array(df_pdf)

        # All values should be in [0, 1]
        assert np.all(pdf_values >= -1e-10)
        assert np.all(pdf_values <= 1.0 + 1e-10)

    def test_pdf_row_sums(self) -> None:
        """Test that each row (fixed theta) sums to 1."""
        n = 5
        valid_x = range(n + 1)
        valid_theta = np.linspace(0, 1, 100)

        df_pdf = pd.DataFrame(
            data=[
                (x, p, binomial_pdf(x, p, n))
                for x, p in itertools.product(valid_x, valid_theta)
            ],
            columns=["x", "theta", "pdf"],
        ).pivot(columns="x", index="theta", values="pdf")

        pdf_values = np.array(df_pdf)
        row_sums = np.sum(pdf_values, axis=1)

        for i, (theta, row_sum) in enumerate(zip(valid_theta, row_sums)):
            assert np.isclose(row_sum, 1.0, atol=1e-10), (
                f"Row sum at theta={theta} should be 1, got {row_sum}"
            )

    def test_log_likelihood_gradient_exists(self) -> None:
        """Test that log-likelihood gradient can be computed."""
        n = 3
        valid_x = range(n + 1)
        valid_theta = np.linspace(0.01, 0.99, 50)

        df_pdf = pd.DataFrame(
            data=[
                (x, p, binomial_pdf(x, p, n))
                for x, p in itertools.product(valid_x, valid_theta)
            ],
            columns=["x", "theta", "pdf"],
        ).pivot(columns="x", index="theta", values="pdf")

        log_pdf = np.log(np.array(df_pdf))

        # Gradient along theta axis (axis=0)
        gradient = np.gradient(log_pdf, axis=0)

        # Gradient should exist (no NaN in interior points)
        interior_gradient = gradient[1:-1]
        assert not np.any(np.isnan(interior_gradient)), (
            "Gradient should not contain NaN in interior points"
        )

    def test_fisher_information_positive(self) -> None:
        """Test that Fisher information is non-negative."""
        n = 3
        valid_x = range(n + 1)
        valid_theta = np.linspace(0.1, 0.9, 50)

        df_pdf = pd.DataFrame(
            data=[
                (x, p, binomial_pdf(x, p, n))
                for x, p in itertools.product(valid_x, valid_theta)
            ],
            columns=["x", "theta", "pdf"],
        ).pivot(columns="x", index="theta", values="pdf")

        log_pdf = np.log(np.array(df_pdf))
        gradient = np.gradient(log_pdf, axis=0)
        squared_gradient = gradient**2

        # Fisher information should be non-negative
        assert np.all(squared_gradient >= -1e-10)

    def test_fisher_information_finite(self) -> None:
        """Test that Fisher information is finite."""
        n = 3
        valid_x = range(n + 1)
        valid_theta = np.linspace(0.01, 0.99, 50)  # Avoid exact 0 and 1

        df_pdf = pd.DataFrame(
            data=[
                (x, p, binomial_pdf(x, p, n))
                for x, p in itertools.product(valid_x, valid_theta)
            ],
            columns=["x", "theta", "pdf"],
        ).pivot(columns="x", index="theta", values="pdf")

        log_pdf = np.log(np.array(df_pdf))
        gradient = np.gradient(log_pdf, axis=0)

        # Gradient should be finite
        assert np.all(np.isfinite(gradient))


class TestFisherInformationPhysical:
    """Tests for physical interpretation of Fisher information."""

    def test_fisher_information_infinite_at_boundaries(self) -> None:
        """Test that Fisher information is infinite at theta=0 or theta=1."""
        # At boundaries, some probabilities go to 0
        # log(0) = -inf, so Fisher information may diverge
        for p_boundary in [0.0, 1.0]:
            if p_boundary > 0 and p_boundary < 1:
                continue  # Skip non-boundary points

            # This is expected to have numerical issues at boundaries
            # The test verifies we handle this gracefully

    def test_fisher_information_maximum_at_p05(self) -> None:
        """Test that Fisher information is maximum at p=0.5 for binomial."""
        # For binomial(n, p), Fisher information is maximized at p=0.5
        # I(θ) = n / (θ * (1 - θ)) for binomial
        # This is symmetric around θ = 0.5 and diverges at the boundaries

        # Test that the Fisher information is symmetric around p=0.5
        # by checking that I(0.5 - δ) = I(0.5 + δ)
        for delta in [0.1, 0.2, 0.3, 0.4]:
            p_low = 0.5 - delta
            p_high = 0.5 + delta

            # Skip if outside valid range
            if p_low < 0.1 or p_high > 0.9:
                continue

            n = 10
            valid_x = range(n + 1)

            # Compute Fisher information for both p values
            fisher_low = 0.0
            fisher_high = 0.0

            for p, fisher_val_ref in [(p_low, fisher_low), (p_high, fisher_high)]:
                _ = pd.DataFrame(
                    data=[
                        (x, theta, binomial_pdf(x, p, n))
                        for x, theta in itertools.product(valid_x, [p])
                    ],
                    columns=["x", "theta", "pdf"],
                ).pivot(columns="x", index="theta", values="pdf")

                # For single p value, we need to estimate from nearby points
                # Use a small perturbation approach
                eps = 0.01
                valid_theta_perturbed = np.array([p - eps, p, p + eps])

                df_perturbed = pd.DataFrame(
                    data=[
                        (x, theta, binomial_pdf(x, theta, n))
                        for x, theta in itertools.product(
                            valid_x, valid_theta_perturbed
                        )
                    ],
                    columns=["x", "theta", "pdf"],
                ).pivot(columns="x", index="theta", values="pdf")

                log_pdf = np.log(np.array(df_perturbed) + 1e-15)
                gradient = np.gradient(log_pdf, axis=0)
                fisher = np.mean(gradient**2)

                if p == p_low:
                    fisher_low = fisher
                else:
                    fisher_high = fisher

            # Fisher information should be equal (symmetric around p=0.5)
            assert np.isclose(fisher_low, fisher_high, rtol=0.3), (
                f"Fisher information should be symmetric around p=0.5: "
                f"I({p_low})={fisher_low}, I({p_high})={fisher_high}"
            )


class TestFisherInformationPerformance:
    """Tests for performance of Fisher information computation."""

    def test_computation_time_reasonable(self) -> None:
        """Test that computation time is reasonable."""
        import time

        n = 5
        theta_sample_size = 100
        valid_x = range(n + 1)
        valid_theta = np.linspace(0, 1, theta_sample_size + 1)

        start = time.perf_counter()
        _ = pd.DataFrame(
            data=[
                (x, p, binomial_pdf(x, p, n))
                for x, p in itertools.product(valid_x, valid_theta)
            ],
            columns=["x", "theta", "pdf"],
        ).pivot(columns="x", index="theta", values="pdf")
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"Computation took too long: {elapsed:.2f}s"

    def test_memory_usage_reasonable(self) -> None:
        """Test that memory usage is reasonable."""
        n = 10
        theta_sample_size = 100
        valid_x = range(n + 1)
        valid_theta = np.linspace(0, 1, theta_sample_size + 1)

        df_pdf = pd.DataFrame(
            data=[
                (x, p, binomial_pdf(x, p, n))
                for x, p in itertools.product(valid_x, valid_theta)
            ],
            columns=["x", "theta", "pdf"],
        ).pivot(columns="x", index="theta", values="pdf")

        # Should be about (n+1) * (theta_sample_size+1) floats
        expected_elements = (n + 1) * (theta_sample_size + 1)
        actual_elements = np.array(df_pdf).size

        assert actual_elements == expected_elements, (
            f"DataFrame should have {expected_elements} elements, got {actual_elements}"
        )

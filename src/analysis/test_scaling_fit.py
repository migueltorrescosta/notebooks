"""Tests for scaling exponent fit utilities (src.analysis.scaling_fit)."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.scaling_fit import (
    compute_scaling_exponent,
    fit_scaling_exponent,
)


class TestComputeScalingExponent:
    def test_sql_scaling(self) -> None:
        N = np.array([4, 8, 16, 32, 64])
        delta_phi = 1.0 / np.sqrt(N)
        alpha = compute_scaling_exponent(N, delta_phi)
        assert np.isclose(alpha, -0.5, atol=0.01)

    def test_heisenberg_scaling(self) -> None:
        N = np.array([4, 8, 16, 32, 64])
        delta_phi = 1.0 / N
        alpha = compute_scaling_exponent(N, delta_phi)
        assert np.isclose(alpha, -1.0, atol=0.01)

    def test_no_scaling(self) -> None:
        N = np.array([4, 8, 16, 32, 64])
        delta_phi = np.ones_like(N, dtype=float)
        alpha = compute_scaling_exponent(N, delta_phi)
        assert np.isclose(alpha, 0.0, atol=0.01)

    def test_returns_nan_for_invalid_fit(self) -> None:
        N = np.array([1, 2])  # Too few points (min_N defaults to 4)
        delta_phi = np.array([1.0, 0.5])
        alpha = compute_scaling_exponent(N, delta_phi)
        assert np.isnan(alpha)

    def test_returns_nan_for_empty_data(self) -> None:
        N = np.array([], dtype=float)
        delta_phi = np.array([], dtype=float)
        with pytest.raises(ValueError):
            compute_scaling_exponent(N, delta_phi)

    def test_matches_fit_scaling_exponent(self) -> None:
        rng = np.random.default_rng(42)
        N = np.array([4, 8, 16, 32, 64, 128])
        delta_phi = 1.0 / np.sqrt(N) * (1 + 0.02 * rng.normal(size=len(N)))

        alpha_simple = compute_scaling_exponent(N, delta_phi)
        result = fit_scaling_exponent(N, delta_phi)

        assert np.isclose(alpha_simple, result.alpha, atol=1e-10)

    def test_works_with_pandas_series(self) -> None:
        N = pd.Series([4.0, 8.0, 16.0, 32.0])
        delta = 1.0 / np.sqrt(N)
        alpha = compute_scaling_exponent(N, delta)
        assert np.isclose(alpha, -0.5, atol=0.01)

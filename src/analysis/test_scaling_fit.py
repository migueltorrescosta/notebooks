"""Tests for scaling exponent fit utilities (src.analysis.scaling_fit)."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.scaling_fit import (
    _filter_fit_points,
    _perform_loglog_fit,
    _validate_fit_inputs,
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


class TestFitValidation:
    """Tests for private helpers extracted from ``fit_scaling_exponent``."""

    def test_validate_inputs_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            _validate_fit_inputs(np.array([1, 2]), np.array([1.0]))

    def test_validate_inputs_empty(self) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            _validate_fit_inputs(np.array([]), np.array([]))

    def test_validate_inputs_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            _validate_fit_inputs(np.array([1.0, np.nan]), np.array([1.0, 2.0]))

    def test_validate_inputs_inf(self) -> None:
        with pytest.raises(ValueError, match=r"NaN|Infinite|Inf"):
            _validate_fit_inputs(np.array([1.0, 2.0]), np.array([1.0, np.inf]))

    def test_filter_fit_points_all_valid(self) -> None:
        N = np.array([4, 8, 16, 32])
        delta = np.array([1.0, 0.7, 0.5, 0.3])
        prepared, warnings = _filter_fit_points(N, delta, min_N=4)
        assert prepared is not None
        N_fit, _ = prepared
        assert len(N_fit) == 4
        assert warnings == []

    def test_filter_fit_points_excludes_small_n(self) -> None:
        N = np.array([1, 2, 4, 8])
        delta = np.array([2.0, 1.5, 1.0, 0.7])
        prepared, warnings = _filter_fit_points(N, delta, min_N=4)
        assert prepared is not None
        N_fit, _ = prepared
        assert list(N_fit) == [4, 8]
        assert warnings == []

    def test_filter_fit_points_all_excluded(self) -> None:
        N = np.array([1, 2, 3])
        delta = np.array([2.0, 1.5, 1.0])
        prepared, warnings = _filter_fit_points(N, delta, min_N=4)
        assert prepared is None
        assert warnings == []

    def test_filter_fit_points_excludes_non_positive_delta(self) -> None:
        N = np.array([4, 8, 16])
        delta = np.array([1.0, 0.0, -0.5])
        prepared, warnings = _filter_fit_points(N, delta, min_N=4)
        assert prepared is not None
        N_fit, _ = prepared
        assert list(N_fit) == [4]
        assert len(warnings) == 1
        assert "non-positive" in warnings[0]

    def test_perform_loglog_fit_sql_scaling(self) -> None:
        N = np.array([4, 8, 16, 32, 64])
        delta = 1.0 / np.sqrt(N)
        result = _perform_loglog_fit(N, delta, R_squared_threshold=0.9, warnings=[])
        assert result.valid
        assert np.isclose(result.alpha, -0.5, atol=0.01)
        assert result.R_squared > 0.99

    def test_perform_loglog_fit_low_r_squared_warning(self) -> None:
        rng = np.random.default_rng(42)
        N = np.array([4, 8, 16, 32, 64])
        delta = 0.5 + 0.5 * rng.random(len(N))  # Noisy, no clear scaling
        warnings: list[str] = []
        result = _perform_loglog_fit(N, delta, R_squared_threshold=0.9, warnings=warnings)
        assert result.valid
        assert len(result.warnings) >= 1 or any("R²" in w for w in warnings)

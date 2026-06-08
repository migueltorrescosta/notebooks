"""Tests for multi_mzi_scaling module.

Covers:
- ScalingAnalysisResult dataclass creation, DataFrame, and Parquet roundtrip
- fit_scaling_exponents log-log regression correctness
- Edge cases: insufficient data, NaN/inf filtering
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.multi_mzi_scaling import (
    ScalingAnalysisResult,
    fit_scaling_exponents,
)


class TestScalingAnalysisResult:
    def test_creation_with_defaults(self) -> None:
        result = ScalingAnalysisResult()
        assert len(result.theta_values) == 0
        assert result.sql_exponent == -0.5

    def test_to_dataframe_contains_expected_columns(self) -> None:
        result = ScalingAnalysisResult(
            theta_values=np.array([0.5, 1.0]),
            exponents=np.array([-0.5, -0.8]),
            prefactors=np.array([2.0, 3.0]),
            r_squared=np.array([0.99, 0.95]),
            sql_exponent=-0.5,
        )
        df = result.to_dataframe()
        assert list(df.columns) == [
            "theta",
            "exponent",
            "prefactor",
            "r_squared",
            "sql_exponent",
        ]
        assert len(df) == 2

    def test_parquet_roundtrip_preserves_all_fields(self) -> None:
        result = ScalingAnalysisResult(
            theta_values=np.array([0.5, 1.0, 1.5]),
            exponents=np.array([-0.5, -0.8, -1.0]),
            prefactors=np.array([2.0, 3.0, 4.0]),
            r_squared=np.array([0.99, 0.95, 0.97]),
            sql_exponent=-0.5,
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        try:
            saved = result.save_parquet(path)
            assert saved == path
            assert path.exists()

            loaded = ScalingAnalysisResult.from_parquet(path)
            np.testing.assert_array_equal(loaded.theta_values, result.theta_values)
            np.testing.assert_array_equal(loaded.exponents, result.exponents)
            np.testing.assert_array_equal(loaded.prefactors, result.prefactors)
            np.testing.assert_array_equal(loaded.r_squared, result.r_squared)
            assert loaded.sql_exponent == result.sql_exponent
        finally:
            path.unlink(missing_ok=True)

    def test_from_parquet_raises_on_missing_columns(self) -> None:
        df_bad = pd.DataFrame({"theta": [0.5], "exponent": [-0.5]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = Path(f.name)
        try:
            df_bad.to_parquet(path, index=False)
            with pytest.raises(ValueError, match="missing required columns"):
                ScalingAnalysisResult.from_parquet(path)
        finally:
            path.unlink(missing_ok=True)


class TestFitScalingExponents:
    def test_perfect_sql_scaling(self) -> None:
        """If Δθ = 2 / sqrt(N), then α = -0.5 and C = 2."""
        theta_values = np.full(20, 0.5)
        N_values = np.array(
            [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48, 54, 60, 70, 80, 100]
        )
        delta_theta_opt = 2.0 / np.sqrt(N_values)

        result = fit_scaling_exponents(theta_values, N_values, delta_theta_opt)

        assert len(result.theta_values) == 1
        assert result.exponents[0] == pytest.approx(-0.5, abs=0.01)
        assert result.prefactors[0] == pytest.approx(2.0, abs=0.05)
        assert result.r_squared[0] > 0.99

    def test_heisenberg_scaling(self) -> None:
        """If Δθ = 3 / N, then α = -1.0 and C = 3."""
        theta_values = np.full(15, 1.0)
        N_values = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60])
        delta_theta_opt = 3.0 / N_values

        result = fit_scaling_exponents(theta_values, N_values, delta_theta_opt)

        assert len(result.theta_values) == 1
        assert result.exponents[0] == pytest.approx(-1.0, abs=0.02)
        assert result.prefactors[0] == pytest.approx(3.0, abs=0.1)
        assert result.r_squared[0] > 0.99

    def test_multiple_theta_values(self) -> None:
        """Separate exponents for each θ value."""
        N_vals = np.array([5, 10, 20, 40, 80])
        N_values = np.tile(N_vals, 2)
        theta_values = np.array([0.5] * 5 + [1.0] * 5)
        delta_theta_opt = np.concatenate(
            [
                2.0 / np.sqrt(N_vals),
                3.0 / N_vals,
            ]
        )

        result = fit_scaling_exponents(theta_values, N_values, delta_theta_opt)

        assert len(result.theta_values) == 2
        assert result.theta_values[0] == pytest.approx(0.5)
        assert result.exponents[0] == pytest.approx(-0.5, abs=0.01)
        assert result.theta_values[1] == pytest.approx(1.0)
        assert result.exponents[1] == pytest.approx(-1.0, abs=0.02)

    def test_insufficient_data_returns_nan(self) -> None:
        """Less than 3 finite points returns NaN exponent."""
        theta_values = np.array([0.5, 0.5])
        N_values = np.array([2, 4])
        delta_theta_opt = np.array([1.0, 0.5])

        result = fit_scaling_exponents(theta_values, N_values, delta_theta_opt)

        assert np.isnan(result.exponents[0])

    def test_nan_and_inf_are_filtered(self) -> None:
        """Non-finite delta_theta values are excluded from the fit."""
        theta_values = np.full(6, 0.5)
        N_values = np.array([2, 4, 8, 16, 32, 64])
        delta_theta_opt = np.array([1.0, np.inf, 0.5, np.nan, 0.25, 0.125])

        result = fit_scaling_exponents(theta_values, N_values, delta_theta_opt)

        assert np.isfinite(result.exponents[0])

    def test_zero_or_negative_delta_filtered(self) -> None:
        """Zero/negative delta values should be excluded from the fit."""
        theta_values = np.full(6, 0.5)
        N_values = np.array([2, 4, 8, 16, 32, 64])
        delta_theta_opt = np.array([1.0, 0.0, -0.5, 0.25, 0.125, 0.0625])

        result = fit_scaling_exponents(theta_values, N_values, delta_theta_opt)

        assert np.isfinite(result.exponents[0])

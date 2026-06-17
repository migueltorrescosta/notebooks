"""Tests for multi_mzi_scaling module.

Covers:
- ScalingAnalysisResult dataclass creation, DataFrame, and Parquet roundtrip
- fit_scaling_exponents log-log regression correctness
- Edge cases: insufficient data, NaN/inf filtering
- plot_scaling_exponents figure creation
- generate_scaling_analysis pipeline orchestration
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.analysis.multi_mzi_scaling import (
    ScalingAnalysisResult,
    fit_scaling_exponents,
    generate_scaling_analysis,
    plot_scaling_exponents,
)
from src.utils.serialization import ParquetSerializable

# ── Helper: minimal sweep result for testing ──────────────────────────────


@dataclass
class _FakeSweepResult(ParquetSerializable):
    """Minimal sweep result used in ``generate_scaling_analysis`` tests."""

    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    N_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    delta_omega_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    t_hold: float = 10.0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "N",
        "delta_omega_opt",
        "t_hold",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "omega": self.omega_values,
                "N": self.N_values,
                "delta_omega_opt": self.delta_omega_opt,
                "t_hold": [self.t_hold] * len(self.omega_values),
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> _FakeSweepResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            omega_values=df["omega"].to_numpy(dtype=float),
            N_values=df["N"].to_numpy(dtype=int),
            delta_omega_opt=df["delta_omega_opt"].to_numpy(dtype=float),
            t_hold=float(df["t_hold"].iloc[0]),
        )


class TestScalingAnalysisResult:
    def test_creation_with_defaults(self) -> None:
        result = ScalingAnalysisResult()
        assert len(result.omega_values) == 0
        assert result.sql_exponent == -0.5

    def test_to_dataframe_contains_expected_columns(self) -> None:
        result = ScalingAnalysisResult(
            omega_values=np.array([0.5, 1.0]),
            exponents=np.array([-0.5, -0.8]),
            prefactors=np.array([2.0, 3.0]),
            r_squared=np.array([0.99, 0.95]),
            sql_exponent=-0.5,
        )
        df = result.to_dataframe()
        assert list(df.columns) == [
            "omega",
            "exponent",
            "prefactor",
            "r_squared",
            "sql_exponent",
        ]
        assert len(df) == 2

    def test_parquet_roundtrip_preserves_all_fields(self) -> None:
        result = ScalingAnalysisResult(
            omega_values=np.array([0.5, 1.0, 1.5]),
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
            np.testing.assert_array_equal(loaded.omega_values, result.omega_values)
            np.testing.assert_array_equal(loaded.exponents, result.exponents)
            np.testing.assert_array_equal(loaded.prefactors, result.prefactors)
            np.testing.assert_array_equal(loaded.r_squared, result.r_squared)
            assert loaded.sql_exponent == result.sql_exponent
        finally:
            path.unlink(missing_ok=True)

    def test_from_parquet_raises_on_missing_columns(self) -> None:
        df_bad = pd.DataFrame({"omega": [0.5], "exponent": [-0.5]})
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
        omega_values = np.full(20, 0.5)
        N_values = np.array(
            [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48, 54, 60, 70, 80, 100]
        )
        delta_omega_opt = 2.0 / np.sqrt(N_values)

        result = fit_scaling_exponents(omega_values, N_values, delta_omega_opt)

        assert len(result.omega_values) == 1
        assert result.exponents[0] == pytest.approx(-0.5, abs=0.01)
        assert result.prefactors[0] == pytest.approx(2.0, abs=0.05)
        assert result.r_squared[0] > 0.99

    def test_heisenberg_scaling(self) -> None:
        """If Δθ = 3 / N, then α = -1.0 and C = 3."""
        omega_values = np.full(15, 1.0)
        N_values = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60])
        delta_omega_opt = 3.0 / N_values

        result = fit_scaling_exponents(omega_values, N_values, delta_omega_opt)

        assert len(result.omega_values) == 1
        assert result.exponents[0] == pytest.approx(-1.0, abs=0.02)
        assert result.prefactors[0] == pytest.approx(3.0, abs=0.1)
        assert result.r_squared[0] > 0.99

    def test_multiple_omega_values(self) -> None:
        """Separate exponents for each θ value."""
        N_vals = np.array([5, 10, 20, 40, 80])
        N_values = np.tile(N_vals, 2)
        omega_values = np.array([0.5] * 5 + [1.0] * 5)
        delta_omega_opt = np.concatenate(
            [
                2.0 / np.sqrt(N_vals),
                3.0 / N_vals,
            ]
        )

        result = fit_scaling_exponents(omega_values, N_values, delta_omega_opt)

        assert len(result.omega_values) == 2
        assert result.omega_values[0] == pytest.approx(0.5)
        assert result.exponents[0] == pytest.approx(-0.5, abs=0.01)
        assert result.omega_values[1] == pytest.approx(1.0)
        assert result.exponents[1] == pytest.approx(-1.0, abs=0.02)

    def test_insufficient_data_returns_nan(self) -> None:
        """Less than 3 finite points returns NaN exponent."""
        omega_values = np.array([0.5, 0.5])
        N_values = np.array([2, 4])
        delta_omega_opt = np.array([1.0, 0.5])

        result = fit_scaling_exponents(omega_values, N_values, delta_omega_opt)

        assert np.isnan(result.exponents[0])

    def test_nan_and_inf_are_filtered(self) -> None:
        """Non-finite delta_omega values are excluded from the fit."""
        omega_values = np.full(6, 0.5)
        N_values = np.array([2, 4, 8, 16, 32, 64])
        delta_omega_opt = np.array([1.0, np.inf, 0.5, np.nan, 0.25, 0.125])

        result = fit_scaling_exponents(omega_values, N_values, delta_omega_opt)

        assert np.isfinite(result.exponents[0])

    def test_zero_or_negative_delta_filtered(self) -> None:
        """Zero/negative delta values should be excluded from the fit."""
        omega_values = np.full(6, 0.5)
        N_values = np.array([2, 4, 8, 16, 32, 64])
        delta_omega_opt = np.array([1.0, 0.0, -0.5, 0.25, 0.125, 0.0625])

        result = fit_scaling_exponents(omega_values, N_values, delta_omega_opt)

        assert np.isfinite(result.exponents[0])


class TestPlotScalingExponents:
    def test_creates_svg_file(self, tmp_path: Path) -> None:
        """Smoke test: plots and saves an SVG."""
        scaling = ScalingAnalysisResult(
            omega_values=np.array([0.5, 1.0, 2.0]),
            exponents=np.array([-0.5, -0.6, -0.8]),
            prefactors=np.array([2.0, 2.5, 3.0]),
            r_squared=np.array([0.99, 0.98, 0.97]),
        )
        svg_path = tmp_path / "test-scaling.svg"
        result = plot_scaling_exponents(scaling, svg_path)
        assert result == svg_path
        assert svg_path.exists()
        assert svg_path.stat().st_size > 0

    def test_with_nan_values_does_not_crash(self, tmp_path: Path) -> None:
        """Plot survives NaN exponents and R² values."""
        scaling = ScalingAnalysisResult(
            omega_values=np.array([0.5, 1.0, 2.0]),
            exponents=np.array([np.nan, -0.6, np.nan]),
            prefactors=np.array([np.nan, 2.5, np.nan]),
            r_squared=np.array([np.nan, 0.98, np.nan]),
        )
        svg_path = tmp_path / "test-scaling-nan.svg"
        result = plot_scaling_exponents(scaling, svg_path)
        assert result == svg_path
        assert svg_path.exists()

    def test_empty_result_creates_skeleton(self, tmp_path: Path) -> None:
        """Empty scaling result produces a skeleton figure (no crash)."""
        scaling = ScalingAnalysisResult()
        svg_path = tmp_path / "test-scaling-empty.svg"
        result = plot_scaling_exponents(scaling, svg_path)
        assert result == svg_path
        assert svg_path.exists()

    def test_sql_and_hl_lines_always_present(self, tmp_path: Path) -> None:
        """SQL and HL reference lines are drawn even with no valid data."""
        scaling = ScalingAnalysisResult()
        svg_path = tmp_path / "test-scaling-lines.svg"
        plot_scaling_exponents(scaling, svg_path)
        content = svg_path.read_text()
        assert "α = −0.5" in content or "alpha = -0.5" in content
        assert "α = −1.0" in content or "alpha = -1.0" in content


class TestGenerateScalingAnalysis:
    def test_generates_scaling_from_sweep(self, tmp_path: Path) -> None:
        """Happy path: sweep → scaling Parquet + figure."""
        sweep_path = tmp_path / "sweep.parquet"
        scaling_path = tmp_path / "scaling.parquet"
        fig_path = tmp_path / "scaling.svg"

        # Create sweep data with known SQL scaling
        N_vals = np.array([2, 4, 8, 16, 32, 64], dtype=int)
        sweep = _FakeSweepResult(
            omega_values=np.full(6, 0.5),
            N_values=N_vals,
            delta_omega_opt=2.0 / np.sqrt(N_vals),
        )
        sweep.save_parquet(sweep_path)

        result = generate_scaling_analysis(
            force=True,
            parquet_path=sweep_path,
            scaling_path=scaling_path,
            fig_path=fig_path,
            result_cls=_FakeSweepResult,
            label="test-scaling",
        )

        assert result is not None
        assert len(result.omega_values) == 1
        assert result.exponents[0] == pytest.approx(-0.5, abs=0.01)
        assert scaling_path.exists()
        assert fig_path.exists()

    def test_cache_hit_loads_existing(self, tmp_path: Path) -> None:
        """Second call without ``force`` loads cached scaling result."""
        sweep_path = tmp_path / "sweep.parquet"
        scaling_path = tmp_path / "scaling.parquet"

        N_vals = np.array([2, 4, 8, 16, 32, 64], dtype=int)
        sweep = _FakeSweepResult(
            omega_values=np.full(6, 1.0),
            N_values=N_vals,
            delta_omega_opt=3.0 / N_vals,
        )
        sweep.save_parquet(sweep_path)

        # First call — compute and save
        result1 = generate_scaling_analysis(
            force=True,
            parquet_path=sweep_path,
            scaling_path=scaling_path,
            result_cls=_FakeSweepResult,
        )

        # Second call — load from cache
        result2 = generate_scaling_analysis(
            force=False,
            parquet_path=sweep_path,
            scaling_path=scaling_path,
            result_cls=_FakeSweepResult,
        )

        assert result1 is not None
        assert result2 is not None
        np.testing.assert_array_equal(result2.omega_values, result1.omega_values)
        np.testing.assert_array_equal(result2.exponents, result1.exponents)
        assert result2.exponents[0] == pytest.approx(-1.0, abs=0.02)

    def test_force_recomputes(self, tmp_path: Path) -> None:
        """With ``force=True``, cached scaling is overwritten."""
        sweep_path = tmp_path / "sweep.parquet"
        scaling_path = tmp_path / "scaling.parquet"

        N_vals = np.array([2, 4, 8, 16, 32, 64], dtype=int)
        sweep = _FakeSweepResult(
            omega_values=np.full(6, 0.5),
            N_values=N_vals,
            delta_omega_opt=1.0 / np.sqrt(N_vals),
        )
        sweep.save_parquet(sweep_path)

        # Create a fake cached scaling with wrong exponents
        cached = ScalingAnalysisResult(
            omega_values=np.array([0.5]),
            exponents=np.array([0.0]),
            prefactors=np.array([1.0]),
            r_squared=np.array([1.0]),
        )
        cached.save_parquet(scaling_path)

        # Force recompute — should overwrite cached
        result = generate_scaling_analysis(
            force=True,
            parquet_path=sweep_path,
            scaling_path=scaling_path,
            result_cls=_FakeSweepResult,
        )

        assert result is not None
        assert result.exponents[0] == pytest.approx(-0.5, abs=0.01)
        assert result.exponents[0] != 0.0

    def test_no_sweep_data_returns_none(self, tmp_path: Path) -> None:
        """When sweep Parquet does not exist, returns ``None``."""
        sweep_path = tmp_path / "nonexistent-sweep.parquet"
        scaling_path = tmp_path / "scaling.parquet"

        result = generate_scaling_analysis(
            force=True,
            parquet_path=sweep_path,
            scaling_path=scaling_path,
            result_cls=_FakeSweepResult,
        )

        assert result is None
        assert not scaling_path.exists()

    def test_skip_plot_when_fig_path_none(self, tmp_path: Path) -> None:
        """Setting ``fig_path=None`` skips plotting."""
        sweep_path = tmp_path / "sweep.parquet"
        scaling_path = tmp_path / "scaling.parquet"

        N_vals = np.array([2, 4, 8, 16, 32, 64], dtype=int)
        sweep = _FakeSweepResult(
            omega_values=np.full(6, 0.5),
            N_values=N_vals,
            delta_omega_opt=2.0 / np.sqrt(N_vals),
        )
        sweep.save_parquet(sweep_path)

        result = generate_scaling_analysis(
            force=True,
            parquet_path=sweep_path,
            scaling_path=scaling_path,
            fig_path=None,
            result_cls=_FakeSweepResult,
        )

        assert result is not None
        assert scaling_path.exists()

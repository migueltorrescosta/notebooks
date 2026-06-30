"""Tests for the N-scaling visualisation module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
import pandas as pd
import pytest

from src.visualization.scaling_plots import (
    plot_n_scaling_optimal_params,
    plot_n_scaling_ratio,
    plot_n_scaling_ratio_comparison,
    plot_n_scaling_sensitivity,
    plot_n_scaling_single_omega,
)


@pytest.fixture
def make_df() -> pd.DataFrame:
    """Create a minimal test DataFrame with the expected columns."""
    data = []
    for N in [1, 2, 5, 10]:
        for omega in [0.1, 0.2]:
            sql = 1.0 / (np.sqrt(N) * 10.0)
            delta = sql * 0.5  # ratio = 2.0
            data.append(
                {
                    "N": N,
                    "omega": omega,
                    "delta_omega_opt": delta,
                    "sql": sql,
                    "ratio": 2.0,
                    "a_x_opt": float(N),
                    "a_y_opt": 0.0,
                    "a_z_opt": 0.0,
                    "a_zz_opt": float(N),
                    "expectation_Jz": 0.0,
                    "variance_Jz": 0.25,
                    "t_hold": 10.0,
                    "fd_step": 1e-6,
                    "success": 1,
                    "nfev": 50,
                },
            )
    return pd.DataFrame(data)


def test_plot_n_scaling_ratio_creates_file(
    make_df: pd.DataFrame, tmp_path: Path
) -> None:
    p = tmp_path / "ratio.svg"
    result = plot_n_scaling_ratio(make_df, p)
    assert result.exists()
    assert result.suffix == ".svg"


def test_plot_n_scaling_sensitivity_creates_file(
    make_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    p = tmp_path / "sensitivity.svg"
    result = plot_n_scaling_sensitivity(make_df, p)
    assert result.exists()
    assert result.suffix == ".svg"


def test_plot_n_scaling_optimal_params_creates_file(
    make_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    p = tmp_path / "params.svg"
    result = plot_n_scaling_optimal_params(make_df, p)
    assert result.exists()
    assert result.suffix == ".svg"


def test_empty_dataframe_skips(make_df: pd.DataFrame, tmp_path: Path) -> None:
    empty = make_df.iloc[0:0]
    p = tmp_path / "empty.svg"
    result = plot_n_scaling_ratio(empty, p)
    # Should still "succeed" (return path) even with no data
    assert result == p


def test_plot_n_scaling_single_omega_creates_file(
    make_df: pd.DataFrame, tmp_path: Path
) -> None:
    p = tmp_path / "single-omega.svg"
    result = plot_n_scaling_single_omega(make_df, omega_fixed=0.1, save_path=p)
    assert result.exists()
    assert result.suffix == ".svg"


def test_plot_n_scaling_single_omega_with_2n_sql(
    make_df: pd.DataFrame, tmp_path: Path
) -> None:
    p = tmp_path / "single-omega-2n.svg"
    result = plot_n_scaling_single_omega(
        make_df, omega_fixed=0.2, save_path=p, include_2n_sql=True
    )
    assert result.exists()


def test_plot_n_scaling_single_omega_empty(
    make_df: pd.DataFrame, tmp_path: Path
) -> None:
    p = tmp_path / "single-omega-empty.svg"
    # Request an ω not in the fixture
    result = plot_n_scaling_single_omega(make_df, omega_fixed=99.0, save_path=p)
    assert result == p  # returns path even with no data


def test_plot_n_scaling_single_omega_explicit_t_hold(
    make_df: pd.DataFrame, tmp_path: Path
) -> None:
    p = tmp_path / "single-omega-t.svg"
    result = plot_n_scaling_single_omega(
        make_df, omega_fixed=0.1, save_path=p, t_hold=5.0
    )
    assert result.exists()


# ── Additional edge-case and function coverage ────────────────────────────────


def test_plot_n_scaling_sensitivity_empty(tmp_path: Path) -> None:
    """Empty DataFrame triggers the early-return path (lines 116-117)."""
    empty = pd.DataFrame(columns=["N", "omega", "delta_omega_opt"])
    p = tmp_path / "sens-empty.svg"
    result = plot_n_scaling_sensitivity(empty, p)
    assert result == p


def test_plot_n_scaling_optimal_params_empty(tmp_path: Path) -> None:
    """Empty DataFrame triggers the early-return path (lines 193-194)."""
    empty = pd.DataFrame(
        columns=["N", "omega", "a_x_opt", "a_y_opt", "a_z_opt", "a_zz_opt"]
    )
    p = tmp_path / "params-empty.svg"
    result = plot_n_scaling_optimal_params(empty, p)
    assert result == p


def test_plot_n_scaling_ratio_comparison_basic(
    make_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Basic ratio comparison plot with two DataFrames."""
    p = tmp_path / "comparison.svg"
    # Use the same df for multi and fixed to exercise both code paths
    result = plot_n_scaling_ratio_comparison(make_df, make_df, p)
    assert result.exists()
    assert result.suffix == ".svg"


def test_plot_n_scaling_ratio_comparison_empty_multi(tmp_path: Path) -> None:
    """When multi-particle df is empty, function returns early (line 261)."""
    empty = pd.DataFrame(columns=["N", "omega", "ratio"])
    fixed = pd.DataFrame(columns=["N", "omega", "ratio"])
    p = tmp_path / "comparison-empty.svg"
    result = plot_n_scaling_ratio_comparison(empty, fixed, p)
    assert result == p


def test_plot_n_scaling_ratio_comparison_different_omegas(tmp_path: Path) -> None:
    """Multi-particle df with data, fixed df with different data."""
    multi_data = []
    fixed_data = []
    for N in [1, 2, 4]:
        for omega in [0.5, 1.0]:
            multi_data.append({"N": N, "omega": omega, "ratio": 2.0 + N * 0.1})
            fixed_data.append({"N": N, "omega": omega, "ratio": 1.0 + N * 0.05})
    df_multi = pd.DataFrame(multi_data)
    df_fixed = pd.DataFrame(fixed_data)
    p = tmp_path / "comparison-omegas.svg"
    result = plot_n_scaling_ratio_comparison(df_multi, df_fixed, p)
    assert result.exists()

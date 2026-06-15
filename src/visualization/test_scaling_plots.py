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
    plot_n_scaling_sensitivity,
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

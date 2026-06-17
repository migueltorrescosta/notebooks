"""Tests for the N-scaling sweep figure generation orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.analysis.n_scaling_sweep import generate_n_scaling_plots
from src.utils.serialization import ParquetSerializable


@dataclass
class _MockSweepResult(ParquetSerializable):
    """Minimal sweep result for testing."""

    N_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_omega_opt: np.ndarray = field(default_factory=lambda: np.array([]))
    t_hold: float = 10.0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "omega",
        "delta_omega_opt",
        "t_hold",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "N": self.N_values,
                "omega": self.omega_values,
                "delta_omega_opt": self.delta_omega_opt,
                "t_hold": np.full(len(self.N_values), self.t_hold),
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> _MockSweepResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            N_values=df["N"].to_numpy(dtype=int),
            omega_values=df["omega"].to_numpy(dtype=float),
            delta_omega_opt=df["delta_omega_opt"].to_numpy(dtype=float),
            t_hold=float(df["t_hold"].iloc[0]),
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path


@pytest.fixture
def mock_sweep(tmp_path: Path) -> Path:
    """Create a mock sweep Parquet file and return its path."""
    N_vals = np.array([1, 2, 5, 10, 1, 2, 5, 10], dtype=int)
    omega_vals = np.array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0])
    sql = 1.0 / (np.sqrt(N_vals) * 10.0)
    delta = sql * 0.5
    result = _MockSweepResult(
        N_values=N_vals,
        omega_values=omega_vals,
        delta_omega_opt=delta,
        t_hold=10.0,
    )
    p = tmp_path / "sweep.parquet"
    result.save_parquet(p)
    return p


def test_generate_n_scaling_plots_creates_figures(
    mock_sweep: Path, tmp_path: Path
) -> None:
    fig_pairs = [
        (0.1, tmp_path / "figures" / "n-scaling-omega0.1.svg"),
        (1.0, tmp_path / "figures" / "n-scaling-omega1.0.svg"),
    ]
    generate_n_scaling_plots(
        parquet_path=mock_sweep,
        result_cls=_MockSweepResult,
        omega_fig_pairs=fig_pairs,
    )
    assert fig_pairs[0][1].exists()
    assert fig_pairs[1][1].exists()


def test_generate_n_scaling_plots_skip_no_sweep(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "nonexistent.parquet"
    fig_pairs = [
        (0.1, tmp_path / "figures" / "n-scaling-omega0.1.svg"),
    ]
    generate_n_scaling_plots(
        parquet_path=missing,
        result_cls=_MockSweepResult,
        omega_fig_pairs=fig_pairs,
    )
    assert not fig_pairs[0][1].exists()


def test_generate_n_scaling_plots_force_overwrites(
    mock_sweep: Path, tmp_path: Path
) -> None:
    fig_p = tmp_path / "n-scaling-omega0.1.svg"
    fig_p.parent.mkdir(parents=True, exist_ok=True)
    fig_p.write_text("old")

    generate_n_scaling_plots(
        force=True,
        parquet_path=mock_sweep,
        result_cls=_MockSweepResult,
        omega_fig_pairs=[(0.1, fig_p)],
    )
    content = fig_p.read_text()
    assert "old" not in content


def test_generate_n_scaling_plots_skip_existing(
    mock_sweep: Path, tmp_path: Path
) -> None:
    fig_p1 = tmp_path / "n-scaling-omega0.1.svg"
    fig_p2 = tmp_path / "n-scaling-omega1.0.svg"
    fig_p1.parent.mkdir(parents=True, exist_ok=True)
    fig_p1.write_text("keep")

    generate_n_scaling_plots(
        force=False,
        parquet_path=mock_sweep,
        result_cls=_MockSweepResult,
        omega_fig_pairs=[(0.1, fig_p1), (1.0, fig_p2)],
    )
    assert fig_p1.read_text() == "keep"
    assert fig_p2.exists()


def test_generate_n_scaling_plots_with_2n_sql(
    mock_sweep: Path, tmp_path: Path
) -> None:
    fig_p = tmp_path / "n-scaling-omega0.1.svg"
    generate_n_scaling_plots(
        parquet_path=mock_sweep,
        result_cls=_MockSweepResult,
        omega_fig_pairs=[(0.1, fig_p)],
        include_2n_sql=True,
    )
    assert fig_p.exists()

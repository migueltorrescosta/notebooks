"""
Result dataclasses for the driven-ancilla metrology protocol.

Each dataclass implements ParquetSerializable for self-describing
save/load roundtrips.

References:
- Report ``reports/20260518/Ancilla-Drive-Enhanced-Metrology.md``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from src.utils.serialization import ParquetSerializable


@dataclass
class DriveDecoupledBaselineResult(ParquetSerializable):
    """Result from evaluating the decoupled baseline (a_x = a_y = a_z = a_zz = 0).

    Attributes:
        t_hold_value: The holding-time value used.
        delta_omega: Computed Δω at the decoupled configuration.
        sql: SQL = 1/t_hold value (time-based SQL; contrast with particle-number SQL 1/√N).
        omega_value: The ω value at which the baseline was evaluated.
    """

    t_hold_value: float
    delta_omega: float
    sql: float
    omega_value: float = 1.0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "t_hold",
        "delta_omega",
        "sql",
        "omega_value",
        "ratio",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "t_hold": [self.t_hold_value],
                "delta_omega": [self.delta_omega],
                "sql": [self.sql],
                "omega_value": [self.omega_value],
                "ratio": [
                    self.delta_omega / self.sql if self.sql > 0 else float("nan")
                ],
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> DriveDecoupledBaselineResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            t_hold_value=float(df["t_hold"].iloc[0]),
            delta_omega=float(df["delta_omega"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            omega_value=float(df["omega_value"].iloc[0]),
        )


@dataclass
class Drive2DSliceResult(ParquetSerializable):
    """Result from a 2D parameter slice scan over (a_drive, a_zz).

    Attributes:
        drive_values: Array of drive coefficient values (a_x, a_y, or a_z).
        azz_values: Array of a_zz (interaction) values.
        delta_omega_grid: 2D array of Δω values, shape
            (len(drive_values), len(azz_values)).
        omega_value: The ω value at which the scan was performed.
        slice_type: 'ax', 'ay', or 'az'.
        sql: SQL = 1/t_hold reference value (time-based SQL).
    """

    drive_values: np.ndarray
    azz_values: np.ndarray
    delta_omega_grid: np.ndarray
    omega_value: float
    slice_type: str = "ax"
    sql: float = 0.1

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "drive",
        "azz",
        "delta_omega",
        "omega_value",
        "slice_type",
        "sql",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Melt the 2D array into a long-format DataFrame."""
        n_d = len(self.drive_values)
        n_a = len(self.azz_values)
        rows: list[dict[str, float | str]] = [
            {
                "drive": float(self.drive_values[i]),
                "azz": float(self.azz_values[j]),
                "delta_omega": float(self.delta_omega_grid[i, j]),
                "omega_value": float(self.omega_value),
                "slice_type": str(self.slice_type),
                "sql": float(self.sql),
            }
            for i in range(n_d)
            for j in range(n_a)
        ]
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> Drive2DSliceResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        drive_unique = sorted(df["drive"].unique())
        azz_unique = sorted(df["azz"].unique())
        n_d = len(drive_unique)
        n_a = len(azz_unique)
        grid = np.full((n_d, n_a), np.nan, dtype=float)
        for _, row in df.iterrows():
            i = drive_unique.index(row["drive"])
            j = azz_unique.index(row["azz"])
            grid[i, j] = row["delta_omega"]
        omega_value = float(df["omega_value"].iloc[0])
        slice_type = str(df["slice_type"].iloc[0])
        sql = float(df["sql"].iloc[0])
        return cls(
            drive_values=np.array(drive_unique, dtype=float),
            azz_values=np.array(azz_unique, dtype=float),
            delta_omega_grid=grid,
            omega_value=omega_value,
            slice_type=slice_type,
            sql=sql,
        )


@dataclass
class DriveRandomSearchResult(ParquetSerializable):
    """Result from a 4D random search over (a_x, a_y, a_z, a_zz).

    Attributes:
        samples: Array of shape (N, 4) with sampled parameter values.
        delta_omega_values: Array of shape (N,) with Δω for each sample.
        best_params: The (a_x, a_y, a_z, a_zz) that gave minimal Δω.
        best_delta_omega: The minimal Δω found.
        omega_value: ω at which the search was performed.
        sql: SQL = 1/t_hold reference (time-based SQL).
    """

    samples: np.ndarray
    delta_omega_values: np.ndarray
    best_params: tuple[float, float, float, float]
    best_delta_omega: float
    omega_value: float = 1.0
    sql: float = 0.1
    t_hold: float = 10.0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "delta_omega",
        "omega_value",
        "sql",
        "t_hold",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.samples)
        return pd.DataFrame(
            {
                "a_x": self.samples[:, 0],
                "a_y": self.samples[:, 1],
                "a_z": self.samples[:, 2],
                "a_zz": self.samples[:, 3],
                "delta_omega": self.delta_omega_values,
                "omega_value": [self.omega_value] * n,
                "sql": [self.sql] * n,
                "t_hold": [self.t_hold] * n,
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> DriveRandomSearchResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        samples = df[["a_x", "a_y", "a_z", "a_zz"]].to_numpy(dtype=float)
        deltas = df["delta_omega"].to_numpy(dtype=float)
        best_idx = int(np.argmin(deltas))
        return cls(
            samples=samples,
            delta_omega_values=deltas,
            best_params=(
                float(samples[best_idx, 0]),
                float(samples[best_idx, 1]),
                float(samples[best_idx, 2]),
                float(samples[best_idx, 3]),
            ),
            best_delta_omega=float(deltas[best_idx]),
            omega_value=float(df["omega_value"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            t_hold=float(df["t_hold"].iloc[0]),
        )


@dataclass
class DriveNelderMeadResult(ParquetSerializable):
    """Result of a single Nelder--Mead run for the driven-ancilla protocol.

    Attributes:
        delta_omega_opt: Best sensitivity Δω found.
        params_opt: Optimal 4-element parameter vector (a_x, a_y, a_z, a_zz).
        omega_true: True ω used for this optimisation.
        success: Whether the optimiser reported success.
        nfev: Number of function evaluations.
        message: Optimiser message.
        expectation_Jz: ⟨J_z^S⟩ at the optimal operating point.
        variance_Jz: Var(J_z^S) at the optimal operating point.
        history: Objective function values at each iteration.
    """

    delta_omega_opt: float
    params_opt: np.ndarray
    omega_true: float
    success: bool
    nfev: int
    message: str = ""
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    history: list[float] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "delta_omega",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "omega_true",
        "success",
        "nfev",
        "expectation_Jz",
        "variance_Jz",
        "message",
        "history_json",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a_x": [float(self.params_opt[0])],
                "a_y": [float(self.params_opt[1])],
                "a_z": [float(self.params_opt[2])],
                "a_zz": [float(self.params_opt[3])],
                "delta_omega": [self.delta_omega_opt],
                "omega_true": [self.omega_true],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "message": [self.message],
                "history_json": [json.dumps(self.history)],
            },
        )

    def _save_sidecars(self, path: Path) -> None:
        """Save optimisation history as a sidecar Parquet file."""
        history_path = path.with_stem(path.stem + "-history")
        pd.DataFrame({"history": [self.history]}).to_parquet(history_path, index=False)

    @classmethod
    def from_parquet(cls, path: str | Path) -> DriveNelderMeadResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        history_path = Path(path).with_stem(Path(path).stem + "-history")
        if history_path.exists():
            history = list(pd.read_parquet(history_path)["history"].iloc[0])
        else:
            history = []
        return cls(
            delta_omega_opt=float(df["delta_omega"].iloc[0]),
            params_opt=np.array(
                [
                    float(df["a_x"].iloc[0]),
                    float(df["a_y"].iloc[0]),
                    float(df["a_z"].iloc[0]),
                    float(df["a_zz"].iloc[0]),
                ]
            ),
            omega_true=float(df["omega_true"].iloc[0]),
            success=bool(int(df["success"].iloc[0])),
            nfev=int(df["nfev"].iloc[0]),
            message=str(df["message"].iloc[0]),
            expectation_Jz=float(df["expectation_Jz"].iloc[0]),
            variance_Jz=float(df["variance_Jz"].iloc[0]),
            history=history,
        )


@dataclass
class DriveOmegaScanResult(ParquetSerializable):
    """Results of an ω scan over driven-ancilla parameters.

    Attributes:
        omega_values: Array of ω values scanned.
        best_params_per_omega: List of optimal (a_x, a_y, a_z, a_zz) tuples.
        best_delta_omega_per_omega: Optimal Δω for each ω value.
        sql_values: SQL = 1/t_hold for each ω (time-based SQL).
        expectation_Jz_per_omega: ⟨J_z^S⟩ at each optimal point.
        variance_Jz_per_omega: Var(J_z^S) at each optimal point.
        all_results: All Nelder-Mead results keyed by ω (for spread analysis).
    """

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "best_delta_omega",
        "sql",
        "ratio",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "expectation_Jz",
        "variance_Jz",
    ]

    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    best_params_per_omega: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )
    best_delta_omega_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_values: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz_per_omega: np.ndarray = field(default_factory=lambda: np.array([]))
    all_results: dict[float, list[DriveNelderMeadResult]] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for i, omega in enumerate(self.omega_values):
            sql = float(self.sql_values[i]) if i < len(self.sql_values) else 0.1
            best = (
                self.best_delta_omega_per_omega[i]
                if i < len(self.best_delta_omega_per_omega)
                else float("inf")
            )
            params = (
                self.best_params_per_omega[i]
                if i < len(self.best_params_per_omega)
                else (0.0, 0.0, 0.0, 0.0)
            )
            exp_jz = (
                float(self.expectation_Jz_per_omega[i])
                if i < len(self.expectation_Jz_per_omega)
                else 0.0
            )
            var_jz = (
                float(self.variance_Jz_per_omega[i])
                if i < len(self.variance_Jz_per_omega)
                else 0.0
            )
            rows.append(
                {
                    "omega": float(omega),
                    "best_delta_omega": best,
                    "sql": sql,
                    "ratio": best / sql
                    if np.isfinite(best) and sql > 0
                    else float("inf"),
                    "a_x": float(params[0]),
                    "a_y": float(params[1]),
                    "a_z": float(params[2]),
                    "a_zz": float(params[3]),
                    "expectation_Jz": exp_jz,
                    "variance_Jz": var_jz,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> DriveOmegaScanResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        omegas = df["omega"].to_numpy(dtype=float)
        best = df["best_delta_omega"].to_numpy(dtype=float)
        sql = df["sql"].to_numpy(dtype=float)
        exps = df["expectation_Jz"].to_numpy(dtype=float)
        vars_ = df["variance_Jz"].to_numpy(dtype=float)
        params_list: list[tuple[float, float, float, float]] = []
        for _, row in df.iterrows():
            params_list.append(
                (
                    float(row["a_x"]),
                    float(row["a_y"]),
                    float(row["a_z"]),
                    float(row["a_zz"]),
                )
            )
        return cls(
            omega_values=omegas,
            best_params_per_omega=params_list,
            best_delta_omega_per_omega=best,
            sql_values=sql,
            expectation_Jz_per_omega=exps,
            variance_Jz_per_omega=vars_,
        )

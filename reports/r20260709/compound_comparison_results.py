"""
Result dataclasses for the Bounded-Compound Comparison.

Stores all input parameters alongside computed results for
self-describing Parquet serialization.  Each dataclass implements
``ParquetSerializable`` for round-trip save/load.

References:
- Report ``reports/r20260709/Compound-Comparison.md``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.serialization import ParquetSerializable


@dataclass
class DecoupledBaselineResult(ParquetSerializable):
    """Decoupled baseline result for both scenarios.

    Stores delta omega for Scenario A and B at the standard MZI encoding
    point (a_z=1, all other coefficients zero). Both should equal 1/t_hold.
    """

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "scenario",
        "delta_omega",
        "sql",
        "ratio_to_sql",
        "t_hold",
    ]

    scenarios: list[str]
    delta_omega_values: np.ndarray
    sql_values: np.ndarray
    ratio_to_sql_values: np.ndarray
    t_hold_value: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "scenario": self.scenarios,
                "delta_omega": self.delta_omega_values,
                "sql": self.sql_values,
                "ratio_to_sql": self.ratio_to_sql_values,
                "t_hold": [self.t_hold_value] * len(self.scenarios),
            }
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> DecoupledBaselineResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            scenarios=list(df["scenario"]),
            delta_omega_values=df["delta_omega"].to_numpy(dtype=float),
            sql_values=df["sql"].to_numpy(dtype=float),
            ratio_to_sql_values=df["ratio_to_sql"].to_numpy(dtype=float),
            t_hold_value=float(df["t_hold"].iloc[0]),
        )


@dataclass
class ScenarioACompoundResult(ParquetSerializable):
    """Result for Scenario A (system-only ω-modulated drive).

    Stores all input parameters alongside computed results for
    self-describing Parquet serialization.
    """

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "best_delta_omega",
        "sql",
        "a_x",
        "a_y",
        "a_z",
        "t_hold",
        "expectation_Jz",
        "variance_Jz",
    ]

    omega_values: np.ndarray
    best_delta_omega_per_omega: np.ndarray
    best_params_per_omega: list[tuple[float, float, float]]
    sql_values: np.ndarray
    t_hold_value: float
    expectation_Jz_per_omega: np.ndarray
    variance_Jz_per_omega: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for i, omega in enumerate(self.omega_values):
            best = (
                float(self.best_delta_omega_per_omega[i])
                if i < len(self.best_delta_omega_per_omega)
                else float("inf")
            )
            sql = (
                float(self.sql_values[i]) if i < len(self.sql_values) else float("nan")
            )
            params = (
                self.best_params_per_omega[i]
                if i < len(self.best_params_per_omega)
                else (0.0, 0.0, 0.0)
            )
            rows.append(
                {
                    "omega": float(omega),
                    "best_delta_omega": best,
                    "sql": sql,
                    "ratio_to_sql": best / sql
                    if np.isfinite(best) and sql > 0
                    else float("inf"),
                    "a_x": float(params[0]),
                    "a_y": float(params[1]),
                    "a_z": float(params[2]),
                    "t_hold": float(self.t_hold_value),
                    "expectation_Jz": (
                        float(self.expectation_Jz_per_omega[i])
                        if i < len(self.expectation_Jz_per_omega)
                        else 0.0
                    ),
                    "variance_Jz": (
                        float(self.variance_Jz_per_omega[i])
                        if i < len(self.variance_Jz_per_omega)
                        else 0.0
                    ),
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> ScenarioACompoundResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            omega_values=df["omega"].to_numpy(dtype=float),
            best_delta_omega_per_omega=df["best_delta_omega"].to_numpy(dtype=float),
            best_params_per_omega=[
                (float(r["a_x"]), float(r["a_y"]), float(r["a_z"]))
                for _, r in df.iterrows()
            ],
            sql_values=df["sql"].to_numpy(dtype=float),
            t_hold_value=float(df["t_hold"].iloc[0]),
            expectation_Jz_per_omega=df["expectation_Jz"].to_numpy(dtype=float),
            variance_Jz_per_omega=df["variance_Jz"].to_numpy(dtype=float),
        )


@dataclass
class CompoundRatioResult(ParquetSerializable):
    """Comparison result between Scenario A and Scenario B.

    Stores the compound ratio R_compound = Δω_A / Δω_B at each ω.
    """

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "delta_omega_A",
        "delta_omega_B",
        "compound_ratio",
        "sql",
        "ratio_A_to_sql",
        "ratio_B_to_sql",
    ]

    omega_values: np.ndarray
    delta_omega_A: np.ndarray
    delta_omega_B: np.ndarray
    compound_ratio: np.ndarray  # R = Δω_A / Δω_B
    sql_values: np.ndarray
    ratio_A_to_sql: np.ndarray  # R_A = Δω_SQL / Δω_A
    ratio_B_to_sql: np.ndarray  # R_B = Δω_SQL / Δω_B

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "omega": self.omega_values,
                "delta_omega_A": self.delta_omega_A,
                "delta_omega_B": self.delta_omega_B,
                "compound_ratio": self.compound_ratio,
                "sql": self.sql_values,
                "ratio_A_to_sql": self.ratio_A_to_sql,
                "ratio_B_to_sql": self.ratio_B_to_sql,
            }
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> CompoundRatioResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            omega_values=df["omega"].to_numpy(dtype=float),
            delta_omega_A=df["delta_omega_A"].to_numpy(dtype=float),
            delta_omega_B=df["delta_omega_B"].to_numpy(dtype=float),
            compound_ratio=df["compound_ratio"].to_numpy(dtype=float),
            sql_values=df["sql"].to_numpy(dtype=float),
            ratio_A_to_sql=df["ratio_A_to_sql"].to_numpy(dtype=float),
            ratio_B_to_sql=df["ratio_B_to_sql"].to_numpy(dtype=float),
        )


@dataclass
class FixedParameterCompoundRatioResult(ParquetSerializable):
    """Fixed-parameter compound ratio between Scenario A and Scenario B.

    At each ω, evaluates Scenario B at Scenario A's optimal (a_x, a_y, a_z)
    with Scenario B's optimal a_zz.  This isolates the interaction-only
    contribution: how much does a_zz improve B when the drive parameters
    are held at A's optimum?

    The free-optimisation ratio (CompoundRatioResult) compares independently
    optimised results and measures total compound advantage.  This fixed-
    parameter ratio measures the marginal gain from the Ising interaction
    alone, at A's optimal drive parameters.
    """

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "delta_omega_A_opt",
        "a_x_A",
        "a_y_A",
        "a_z_A",
        "a_zz_B",
        "delta_omega_B_fixed",
        "fixed_ratio",
        "sql",
    ]

    omega_values: np.ndarray
    delta_omega_A_opt: np.ndarray
    a_x_A: np.ndarray
    a_y_A: np.ndarray
    a_z_A: np.ndarray
    a_zz_B: np.ndarray
    delta_omega_B_fixed: np.ndarray
    fixed_ratio: np.ndarray  # R_fixed = Δω_A^opt / Δω_B(at A's params, B's a_zz)
    sql_values: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "omega": self.omega_values,
                "delta_omega_A_opt": self.delta_omega_A_opt,
                "a_x_A": self.a_x_A,
                "a_y_A": self.a_y_A,
                "a_z_A": self.a_z_A,
                "a_zz_B": self.a_zz_B,
                "delta_omega_B_fixed": self.delta_omega_B_fixed,
                "fixed_ratio": self.fixed_ratio,
                "sql": self.sql_values,
            }
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> FixedParameterCompoundRatioResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            omega_values=df["omega"].to_numpy(dtype=float),
            delta_omega_A_opt=df["delta_omega_A_opt"].to_numpy(dtype=float),
            a_x_A=df["a_x_A"].to_numpy(dtype=float),
            a_y_A=df["a_y_A"].to_numpy(dtype=float),
            a_z_A=df["a_z_A"].to_numpy(dtype=float),
            a_zz_B=df["a_zz_B"].to_numpy(dtype=float),
            delta_omega_B_fixed=df["delta_omega_B_fixed"].to_numpy(dtype=float),
            fixed_ratio=df["fixed_ratio"].to_numpy(dtype=float),
            sql_values=df["sql"].to_numpy(dtype=float),
        )

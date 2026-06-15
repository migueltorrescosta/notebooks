"""
Shared N-scaling result dataclasses for ancilla-drive metrology reports.

Provides ``NScalingResult`` (per-(N, ω) optimisation result) and
``NScalingScanResult`` (collection across a grid) with Parquet roundtrip
and fail-fast deserialization.

Used by reports #20260611 and #20260612.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd


@dataclass
class NScalingResult:
    """Result from optimising the ω-modulated protocol for a single (N, ω) pair.

    Attributes:
        N: Number of system particles.
        omega: Phase rate value.
        delta_omega_opt: Best sensitivity Δω found.
        sql: SQL = 1/(√N × t_hold).
        ratio: SQL / Δω_opt (ratio > 1 means beating SQL).
        a_x_opt: Optimal J_x^A drive coefficient.
        a_y_opt: Optimal J_y^A drive coefficient.
        a_z_opt: Optimal J_z^A drive coefficient.
        a_zz_opt: Optimal Ising interaction coefficient.
        expectation_Jz: ⟨J_z^S⟩ at the optimal operating point.
        variance_Jz: Var(J_z^S) at the optimal operating point.
        t_hold: Holding time (fixed at 10.0).
        fd_step: Finite-difference step.
        success: Whether Nelder-Mead reported success.
        nfev: Number of function evaluations.
    """

    N: int
    omega: float
    delta_omega_opt: float
    sql: float
    ratio: float
    a_x_opt: float
    a_y_opt: float
    a_z_opt: float
    a_zz_opt: float
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    t_hold: float = 10.0
    fd_step: float = 1e-6
    success: bool = False
    nfev: int = 0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "omega",
        "delta_omega_opt",
        "sql",
        "ratio",
        "a_x_opt",
        "a_y_opt",
        "a_z_opt",
        "a_zz_opt",
        "expectation_Jz",
        "variance_Jz",
        "t_hold",
        "fd_step",
        "success",
        "nfev",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "N": [self.N],
                "omega": [self.omega],
                "delta_omega_opt": [self.delta_omega_opt],
                "sql": [self.sql],
                "ratio": [self.ratio],
                "a_x_opt": [self.a_x_opt],
                "a_y_opt": [self.a_y_opt],
                "a_z_opt": [self.a_z_opt],
                "a_zz_opt": [self.a_zz_opt],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "t_hold": [self.t_hold],
                "fd_step": [self.fd_step],
                "success": [int(self.success)],
                "nfev": [self.nfev],
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> NScalingResult:
        path = Path(path)
        df = pd.read_parquet(path)
        missing = {c for c in cls._PARQUET_COLUMNS if c not in df.columns}
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Re-run the simulation that generated it."
            )
        row = df.iloc[0]
        return cls(
            N=int(row["N"]),
            omega=float(row["omega"]),
            delta_omega_opt=float(row["delta_omega_opt"]),
            sql=float(row["sql"]),
            ratio=float(row["ratio"]),
            a_x_opt=float(row["a_x_opt"]),
            a_y_opt=float(row["a_y_opt"]),
            a_z_opt=float(row["a_z_opt"]),
            a_zz_opt=float(row["a_zz_opt"]),
            expectation_Jz=float(row["expectation_Jz"]),
            variance_Jz=float(row["variance_Jz"]),
            t_hold=float(row["t_hold"]),
            fd_step=float(row["fd_step"]),
            success=bool(int(row["success"])),
            nfev=int(row["nfev"]),
        )


@dataclass
class NScalingScanResult:
    """Collection of N-scaling results for a grid of (N, ω) pairs.

    Attributes:
        results: List of per-(N, ω) NScalingResult.
        N_values: Array of N values (sorted).
        omega_values: Array of ω values (sorted).
    """

    results: list[NScalingResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=NScalingResult._PARQUET_COLUMNS)
        return pd.concat([r.to_dataframe() for r in self.results], ignore_index=True)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> NScalingScanResult:
        path = Path(path)
        df = pd.read_parquet(path)
        required = set(NScalingResult._PARQUET_COLUMNS)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Re-run the simulation that generated it."
            )
        results: list[NScalingResult] = []
        for _, row in df.iterrows():
            results.append(
                NScalingResult(
                    N=int(row["N"]),
                    omega=float(row["omega"]),
                    delta_omega_opt=float(row["delta_omega_opt"]),
                    sql=float(row["sql"]),
                    ratio=float(row["ratio"]),
                    a_x_opt=float(row["a_x_opt"]),
                    a_y_opt=float(row["a_y_opt"]),
                    a_z_opt=float(row["a_z_opt"]),
                    a_zz_opt=float(row["a_zz_opt"]),
                    expectation_Jz=float(row["expectation_Jz"]),
                    variance_Jz=float(row["variance_Jz"]),
                    t_hold=float(row["t_hold"]),
                    fd_step=float(row["fd_step"]),
                    success=bool(int(row["success"])),
                    nfev=int(row["nfev"]),
                ),
            )
        return cls(results=results)

    @property
    def N_values(self) -> np.ndarray:
        return np.array(sorted({r.N for r in self.results}))

    @property
    def omega_values(self) -> np.ndarray:
        return np.array(sorted({r.omega for r in self.results}))

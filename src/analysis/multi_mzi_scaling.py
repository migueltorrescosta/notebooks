"""
Scaling analysis for multi-particle MZI simulations.

Provides the :class:`ScalingAnalysisResult` dataclass (with Parquet roundtrip)
and :func:`fit_scaling_exponents` for log-log linear regression of
Δθ = C·N^α at each θ value.

This module was promoted from duplicated definitions in three reports
(20260522, 20260523, 20260525) and should be reused by any
future multi-MZI scaling analysis.

Physical Model:
    Δθ(θ, N) ∝ N^{α(θ)}    ⇒    log Δθ = α·log N + log C

Conventions:
    - SQL exponent: α = -0.5 (Δθ ∝ 1/√N)
    - Heisenberg limit: α = -1.0 (Δθ ∝ 1/N)
    - N is particle number per subsystem (dimensionless)
    - θ is the unknown phase rate (dimensionless)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ScalingAnalysisResult:
    """Log-log fit results for Δθ ∝ N^α at each θ value.

    Attributes:
        theta_values: θ values analysed.
        exponents: Exponent α from Δθ ∝ N^α for each θ.
        prefactors: Prefactor C in Δθ = C N^α for each θ.
        r_squared: R² goodness-of-fit for each θ.
        sql_exponent: SQL exponent = -0.5 (reference).
    """

    theta_values: np.ndarray = field(default_factory=lambda: np.array([]))
    exponents: np.ndarray = field(default_factory=lambda: np.array([]))
    prefactors: np.ndarray = field(default_factory=lambda: np.array([]))
    r_squared: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_exponent: float = -0.5

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "theta": self.theta_values,
                "exponent": self.exponents,
                "prefactor": self.prefactors,
                "r_squared": self.r_squared,
                "sql_exponent": [self.sql_exponent] * len(self.theta_values),
            },
        )

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> ScalingAnalysisResult:
        df = pd.read_parquet(path)
        required = {"theta", "exponent", "prefactor", "r_squared", "sql_exponent"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: {sorted(missing)}. "
                "Regenerate the file with the current code."
            )
        return cls(
            theta_values=df["theta"].to_numpy(dtype=float),
            exponents=df["exponent"].to_numpy(dtype=float),
            prefactors=df["prefactor"].to_numpy(dtype=float),
            r_squared=df["r_squared"].to_numpy(dtype=float),
            sql_exponent=float(df["sql_exponent"].iloc[0]),
        )


def fit_scaling_exponents(
    theta_values: np.ndarray,
    N_values: np.ndarray,
    delta_theta_opt: np.ndarray,
) -> ScalingAnalysisResult:
    """Fit Δθ = C N^α at each θ value via log-log linear regression.

    Performs a log-log linear fit per unique θ:
        log(Δθ) = α log(N) + log(C)

    Args:
        theta_values: Array of θ values (same length as N_values and delta_theta_opt).
        N_values: Array of particle numbers (same length).
        delta_theta_opt: Array of optimal sensitivities (same length).

    Returns:
        ScalingAnalysisResult with exponent α and prefactor C at each θ.
    """
    theta_vals = np.unique(theta_values)
    exponents = np.full(len(theta_vals), np.nan, dtype=float)
    prefactors = np.full(len(theta_vals), np.nan, dtype=float)
    r_squared_vals = np.full(len(theta_vals), np.nan, dtype=float)

    for i, theta in enumerate(theta_vals):
        mask = np.isclose(theta_values, theta)
        N_vals_at_theta = N_values[mask].astype(float)
        delta_vals = delta_theta_opt[mask]

        # Filter finite values
        finite_mask = np.isfinite(delta_vals) & (delta_vals > 0) & (N_vals_at_theta > 0)
        N_finite = N_vals_at_theta[finite_mask]
        delta_finite = delta_vals[finite_mask]

        if len(N_finite) < 3:
            continue

        log_N = np.log(N_finite)
        log_delta = np.log(delta_finite)

        # Linear fit
        A = np.vstack([log_N, np.ones_like(log_N)]).T
        coeffs, *_ = np.linalg.lstsq(A, log_delta, rcond=None)
        alpha = coeffs[0]
        log_C = coeffs[1]

        exponents[i] = alpha
        prefactors[i] = np.exp(log_C)

        # R² calculation
        if len(log_delta) > 2:
            ss_res = np.sum((log_delta - A @ coeffs) ** 2)
            ss_tot = np.sum((log_delta - np.mean(log_delta)) ** 2)
            r_squared_vals[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return ScalingAnalysisResult(
        theta_values=theta_vals,
        exponents=exponents,
        prefactors=prefactors,
        r_squared=r_squared_vals,
    )

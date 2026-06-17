"""
Scaling analysis for multi-particle MZI simulations.

Provides the :class:`ScalingAnalysisResult` dataclass (with Parquet roundtrip),
:func:`fit_scaling_exponents` for log-log linear regression,
:func:`plot_scaling_exponents` for visualisation, and
:func:`generate_scaling_analysis` for the cache-check → compute → save → plot
pipeline.

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
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.serialization import ParquetSerializable


@dataclass
class ScalingAnalysisResult(ParquetSerializable):
    """Log-log fit results for Δθ ∝ N^α at each ω value.

    Attributes:
        omega_values: ω values analysed.
        exponents: Exponent α from Δθ ∝ N^α for each θ.
        prefactors: Prefactor C in Δθ = C N^α for each θ.
        r_squared: R² goodness-of-fit for each θ.
        sql_exponent: SQL exponent = -0.5 (reference).
    """

    omega_values: np.ndarray = field(default_factory=lambda: np.array([]))
    exponents: np.ndarray = field(default_factory=lambda: np.array([]))
    prefactors: np.ndarray = field(default_factory=lambda: np.array([]))
    r_squared: np.ndarray = field(default_factory=lambda: np.array([]))
    sql_exponent: float = -0.5

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "exponent",
        "prefactor",
        "r_squared",
        "sql_exponent",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "omega": self.omega_values,
                "exponent": self.exponents,
                "prefactor": self.prefactors,
                "r_squared": self.r_squared,
                "sql_exponent": [self.sql_exponent] * len(self.omega_values),
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> ScalingAnalysisResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        return cls(
            omega_values=df["omega"].to_numpy(dtype=float),
            exponents=df["exponent"].to_numpy(dtype=float),
            prefactors=df["prefactor"].to_numpy(dtype=float),
            r_squared=df["r_squared"].to_numpy(dtype=float),
            sql_exponent=float(df["sql_exponent"].iloc[0]),
        )


def fit_scaling_exponents(
    omega_values: np.ndarray,
    N_values: np.ndarray,
    delta_omega_opt: np.ndarray,
) -> ScalingAnalysisResult:
    """Fit Δθ = C N^α at each ω value via log-log linear regression.

    Performs a log-log linear fit per unique ω:
        log(Δθ) = α log(N) + log(C)

    Args:
        omega_values: Array of ω values (same length as N_values and delta_omega_opt).
        N_values: Array of particle numbers (same length).
        delta_omega_opt: Array of optimal sensitivities (same length).

    Returns:
        ScalingAnalysisResult with exponent α and prefactor C at each ω.
    """
    omega_vals = np.unique(omega_values)
    exponents = np.full(len(omega_vals), np.nan, dtype=float)
    prefactors = np.full(len(omega_vals), np.nan, dtype=float)
    r_squared_vals = np.full(len(omega_vals), np.nan, dtype=float)

    for i, omega in enumerate(omega_vals):
        mask = np.isclose(omega_values, omega)
        N_vals_at_theta = N_values[mask].astype(float)
        delta_vals = delta_omega_opt[mask]

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
        omega_values=omega_vals,
        exponents=exponents,
        prefactors=prefactors,
        r_squared=r_squared_vals,
    )


def plot_scaling_exponents(
    scaling: ScalingAnalysisResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot the scaling exponent α vs ω from log-log fits.

    Produces a two-panel figure:
      - Left: exponent α vs ω with SQL (−0.5) and HL (−1.0) reference lines.
      - Right: R² goodness-of-fit vs ω.

    Args:
        scaling: Scaling analysis result.
        save_path: Output SVG path.
        figsize: Figure size in inches (width, height).

    Returns:
        Path to the saved SVG file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: exponent vs ω
    valid_exp = np.isfinite(scaling.exponents)
    if np.any(valid_exp):
        ax1.plot(
            scaling.omega_values[valid_exp],
            scaling.exponents[valid_exp],
            "o-",
            color="C1",
            markersize=6,
            linewidth=1.5,
        )
    ax1.axhline(
        y=-0.5,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="SQL (α = −0.5)",
    )
    ax1.axhline(
        y=-1.0,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label="HL (α = −1.0)",
    )
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"Scaling exponent $\alpha$")
    ax1.set_title("Exponent $\\alpha$ from\n$\\Delta\\omega = C N^{\\alpha}$")
    ax1.legend(fontsize=9)

    # Right: R² vs ω
    valid_r2 = np.isfinite(scaling.r_squared)
    if np.any(valid_r2):
        ax2.plot(
            scaling.omega_values[valid_r2],
            scaling.r_squared[valid_r2],
            "s-",
            color="C2",
            markersize=6,
            linewidth=1.5,
        )
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$R^2$")
    ax2.set_title("Goodness of Fit")
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_scaling_analysis(
    force: bool = False,
    *,
    parquet_path: Path,
    scaling_path: Path,
    fig_path: Path | None = None,
    result_cls: type[Any],
    plot_fn: Callable[..., Any] | None = plot_scaling_exponents,
    label: str = "scaling analysis",
) -> ScalingAnalysisResult | None:
    """Generate scaling exponent analysis from a sweep result.

    Orchestrates the cache-check → load → fit → save → plot pipeline.

    If the sweep Parquet does not exist, the function prints a warning and
    returns ``None``.  If the scaling Parquet already exists and ``force`` is
    ``False``, it is loaded and returned from cache (no recomputation).

    Args:
        force: Recompute even if a cached scaling Parquet exists.
        parquet_path: Path to the input sweep result Parquet.
        scaling_path: Path for the output scaling result Parquet.
        fig_path: Optional path for the scaling-exponents figure.
        result_cls: The sweep result class with ``from_parquet``.
        plot_fn: Function to plot the scaling result (called with the scaling
            result and ``fig_path``). Set to ``None`` to skip plotting.
            Defaults to :func:`plot_scaling_exponents`.
        label: Human-readable label for console output.

    Returns:
        The loaded or computed ``ScalingAnalysisResult``, or ``None`` if the
        sweep Parquet was not found.
    """
    if not parquet_path.exists():
        print(f"[skip] Sweep data not found at {parquet_path}; run sweep first")
        return None

    if scaling_path.exists() and not force:
        scaling = ScalingAnalysisResult.from_parquet(scaling_path)
        print(f"[skip] {scaling_path.name} exists (use --force to overwrite)")
    else:
        print(f"[run]  Computing {label}...")
        result = result_cls.from_parquet(parquet_path)
        scaling = fit_scaling_exponents(
            result.omega_values,
            result.N_values,
            result.delta_omega_opt,
        )
        scaling.save_parquet(scaling_path)
        print(f"[save] {scaling_path}")

    if plot_fn is not None and fig_path is not None:
        plot_fn(scaling, fig_path)
        print(f"[fig]  {fig_path}")

    return scaling

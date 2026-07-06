r"""
Cavity-Enhanced TMSV Mach-Zehnder Interferometer simulation.

Combines cavity finesse enhancement with TMSV input (#20260625) to
achieve a prefactor improvement while preserving the TMSV scaling exponent.

Usage:
    uv run python reports/20260629/cavity_enhanced_tmsv_mzi.py --force
    uv run python reports/20260629/cavity_enhanced_tmsv_mzi.py --force --only F=10
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

from src.analysis.scaling_fit import fit_scaling_exponent
from src.physics.hilbert_space import resource_value_to_truncation
from src.physics.mzi_simulation import (
    beam_splitter_unitary,
    compute_mzi_sensitivity_grid,
)
from src.physics.mzi_states import make_two_mode_squeezed_vacuum
from src.physics.sv_qfi import compute_tmsv_captured_norm
from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable

# Force non-interactive backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

sns.set_theme(style="whitegrid")

# ── Constants ─────────────────────────────────────────────────────────────────

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260629"
H_t: float = 10.0  # Base holding time

# Path helpers
_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)

# Parameter sweep ranges
MEAN_TOTAL_RANGE: list[float] = [
    float(n) for n in range(2, 41, 2)
]  # ⟨N⟩ = 2..40 even (20 points)
FINESSE_RANGE: list[float] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

N_OMEGA_POINTS: int = 200  # ω points per F value
CFI_EPSILON: float = 1e-6  # Central difference step
PROB_FLOOR: float = 1e-15  # CFI denominator regularisation
MAX_TRUNC: int = 250  # Maximum photons per mode (explicit, never defaulted)
TRUNC_MULTIPLIER: float = 8.0  # Multiplier for TMSV truncation; variance convergence needs M ~ 8×⟨N⟩
MIN_TRUNC: int = 20  # Minimum truncation (N=2 needs M≥20 for CFI/QFI > 0.95)


# Figure settings
COLORMAP = "viridis"


# =============================================================================
# ω Grid Construction
# =============================================================================


def _omega_grid_finesse(finesse: float, n_points: int = N_OMEGA_POINTS) -> np.ndarray:
    r"""Construct an adaptive ω grid for a given finesse, covering the first quarter-wave.

    Uses quadratic spacing (:math:`\omega \propto t^2`) to cluster more points near
    :math:`\omega=0` where the CFI peak is narrow.

    :math:`\omega_{\max} = \pi / (2 \cdot \mathcal{F} \cdot H_t)`

    Args:
        finesse: Cavity finesse :math:`\mathcal{F}`.
        n_points: Number of grid points (default N_OMEGA_POINTS).

    Returns:
        Array of ω values with quadratic spacing.
    """
    omega_max = np.pi / (2.0 * finesse * H_t)
    # Quadratic spacing: more points near ω=0 for CFI peak resolution
    t = np.linspace(0.0, 1.0, n_points)
    return omega_max * t**2


# =============================================================================
# Single (N, F) Sensitivity Computation
# =============================================================================


def generate_single_cavity_point(
    mean_total: float,
    finesse: float,
    max_photons: int | None = None,
    t_hold: float = H_t,
) -> dict | None:
    r"""Run a ω-scan for a single (⟨N⟩, ℱ) combination.

    Args:
        mean_total: Total mean photon number :math:`\langle N \rangle`.
        finesse: Cavity finesse :math:`\mathcal{F}`.
        max_photons: Truncation per mode (auto-computed if None).
        t_hold: Base holding time.

    Returns:
        Dictionary with arrays keyed by column names, or None on failure.
    """
    try:
        if max_photons is None:
            # Variance convergence needs M ~ 8×⟨N⟩; enforce minimum for small N
            _max_trunc = MAX_TRUNC
            max_photons = max(
                resource_value_to_truncation(
                    mean_total,
                    "tmsv",
                    trunc_multiplier=TRUNC_MULTIPLIER,
                    max_trunc=_max_trunc,
                ),
                MIN_TRUNC,
            )

        # Effective holding time
        t_hold_eff = finesse * t_hold

        # Prepare TMSV state
        state = make_two_mode_squeezed_vacuum(mean_total, max_photons)

        # ω grid for this finesse
        omega_grid = _omega_grid_finesse(finesse)

        # Pre-compute BS matrix once (reused across all ω)
        bs = beam_splitter_unitary(np.pi / 4, 0.0, max_photons)

        # Run sensitivity grid
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            max_photons,
            t_hold=t_hold_eff,
            skip_bs1=False,
            cfi_epsilon=CFI_EPSILON,
            prob_floor=PROB_FLOOR,
            bs=bs,
        )

        omega_arr = np.asarray(result["omega_values"], dtype=float)
        fisher_classical_arr = np.asarray(result["fisher_classical"], dtype=float)
        delta_omega_c_arr = np.asarray(result["delta_omega_c"], dtype=float)
        delta_omega_q = float(result["delta_omega_q"])
        fisher_quantum = float(result["fisher_quantum"])

        # SQL reference at physical photon number: 1 / (t_hold * sqrt(mean_total))
        sql_val = 1.0 / (t_hold * np.sqrt(mean_total))
        # SQL at effective resources: each photon reused ℱ times
        # Δω_SQL^(cav) = 1 / (H_t * sqrt(ℱ * ⟨N⟩)) per report formula
        sql_eff_val = 1.0 / (t_hold * np.sqrt(finesse * mean_total))

        # Captured norm
        captured_norm = compute_tmsv_captured_norm(mean_total, max_photons)

        n_omega = len(omega_arr)

        return {
            "mean_total": np.full(n_omega, mean_total, dtype=float),
            "finesse": np.full(n_omega, finesse, dtype=float),
            "omega_values": omega_arr,
            "cfi_values": fisher_classical_arr,
            "qfi_bound": np.full(n_omega, fisher_quantum, dtype=float),
            "delta_omega_c": delta_omega_c_arr,
            "delta_omega_q": np.full(n_omega, delta_omega_q, dtype=float),
            "delta_omega_sql": np.full(n_omega, sql_val, dtype=float),
            "delta_omega_sql_eff": np.full(n_omega, sql_eff_val, dtype=float),
            "t_hold": np.full(n_omega, t_hold, dtype=float),
            "t_hold_eff": np.full(n_omega, t_hold_eff, dtype=float),
            "truncation_M": np.full(n_omega, float(max_photons), dtype=float),
            "captured_norm": np.full(n_omega, captured_norm, dtype=float),
        }
    except (ValueError, AssertionError) as exc:
        warnings.warn(
            f"Failed at ⟨N⟩={mean_total}, ℱ={finesse}: {exc}",
            stacklevel=2,
        )
        return None


# =============================================================================
# Data Containers
# =============================================================================


@dataclass
class CavityTmsvScalingFit:
    r"""Scaling-fit results for cavity-enhanced TMSV MZI.

    Attributes:
        finesse_values: Array of finesse values in the fit.
        alpha_values: Fitted :math:`\alpha` per finesse value.
        C_values: Fitted prefactor :math:`C` per finesse value.
        alpha_overall: Fitted :math:`\alpha` from combined data.
        alpha_overall_err: Standard error of combined :math:`\alpha`.
        beta: Prefactor scaling exponent :math:`\beta`
            (:math:`\Delta\omega_{\min} \propto \mathcal{F}^{-\beta}`).
        beta_err: Standard error of :math:`\beta`.
        C0: Prefactor :math:`C_0` in :math:`\Delta\omega_{\min} = C_0 \mathcal{F}^{-\beta}`.
        valid: Whether the fits are physically valid.
        warnings_list: Warnings from the fitting process.
    """

    finesse_values: np.ndarray
    alpha_values: np.ndarray
    C_values: np.ndarray
    alpha_overall: float = float("nan")
    alpha_overall_err: float = float("nan")
    beta: float = float("nan")
    beta_err: float = float("nan")
    C0: float = float("nan")
    valid: bool = True
    warnings_list: list[str] = field(default_factory=list)


@dataclass
class CavityTmsvSensitivityResult(ParquetSerializable):
    r"""Raw sweep data for cavity-enhanced TMSV MZI.

    Stores one row per (⟨N⟩, ℱ, ω) combination in long format.

    Attributes:
        mean_total: Array of total mean photon numbers.
        finesse: Array of cavity finesse values.
        omega_values: Array of :math:`\omega` values.
        cfi_values: Classical Fisher Information at each point.
        qfi_bound: Quantum Fisher Information at each point.
        delta_omega_c: :math:`\Delta\omega_C` at each point.
        delta_omega_q: :math:`\Delta\omega_Q` at each point.
        delta_omega_sql: :math:`\Delta\omega_{\text{SQL}}` at physical ⟨N⟩.
        delta_omega_sql_eff: :math:`\Delta\omega_{\text{SQL}}` at effective resources.
        t_hold: Base holding time :math:`H_t`.
        t_hold_eff: Effective holding time :math:`\mathcal{F} \cdot H_t`.
        truncation_M: Truncation per mode.
        captured_norm: Fraction of TMSV norm captured by truncation.
    """

    mean_total: np.ndarray
    finesse: np.ndarray
    omega_values: np.ndarray
    cfi_values: np.ndarray
    qfi_bound: np.ndarray
    delta_omega_c: np.ndarray
    delta_omega_q: np.ndarray
    delta_omega_sql: np.ndarray
    delta_omega_sql_eff: np.ndarray
    t_hold: np.ndarray
    t_hold_eff: np.ndarray
    truncation_M: np.ndarray
    captured_norm: np.ndarray

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "mean_total",
        "finesse",
        "omega_values",
        "cfi_values",
        "qfi_bound",
        "delta_omega_c",
        "delta_omega_q",
        "delta_omega_sql",
        "delta_omega_sql_eff",
        "t_hold",
        "t_hold_eff",
        "truncation_M",
        "captured_norm",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to long-format DataFrame (one row per (N, F, ω))."""
        return pd.DataFrame(
            {
                "mean_total": self.mean_total,
                "finesse": self.finesse,
                "omega_values": self.omega_values,
                "cfi_values": self.cfi_values,
                "qfi_bound": self.qfi_bound,
                "delta_omega_c": self.delta_omega_c,
                "delta_omega_q": self.delta_omega_q,
                "delta_omega_sql": self.delta_omega_sql,
                "delta_omega_sql_eff": self.delta_omega_sql_eff,
                "t_hold": self.t_hold,
                "t_hold_eff": self.t_hold_eff,
                "truncation_M": self.truncation_M,
                "captured_norm": self.captured_norm,
            }
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> CavityTmsvSensitivityResult:
        """Construct from a DataFrame (must contain all required columns)."""
        cls._validate_columns(df)
        return cls(
            mean_total=df["mean_total"].to_numpy(dtype=float),
            finesse=df["finesse"].to_numpy(dtype=float),
            omega_values=df["omega_values"].to_numpy(dtype=float),
            cfi_values=df["cfi_values"].to_numpy(dtype=float),
            qfi_bound=df["qfi_bound"].to_numpy(dtype=float),
            delta_omega_c=df["delta_omega_c"].to_numpy(dtype=float),
            delta_omega_q=df["delta_omega_q"].to_numpy(dtype=float),
            delta_omega_sql=df["delta_omega_sql"].to_numpy(dtype=float),
            delta_omega_sql_eff=df["delta_omega_sql_eff"].to_numpy(dtype=float),
            t_hold=df["t_hold"].to_numpy(dtype=float),
            t_hold_eff=df["t_hold_eff"].to_numpy(dtype=float),
            truncation_M=df["truncation_M"].to_numpy(dtype=float),
            captured_norm=df["captured_norm"].to_numpy(dtype=float),
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> CavityTmsvSensitivityResult:
        """Load from a Parquet file."""
        df = pd.read_parquet(path)
        return cls.from_dataframe(df)


# =============================================================================
# Full Data Generation
# =============================================================================


def _row_dicts_to_result(rows: list[dict]) -> CavityTmsvSensitivityResult:
    """Combine a list of row-dicts (each returned by ``generate_single_cavity_point``)
    into a single ``CavityTmsvSensitivityResult``."""
    if not rows:
        raise ValueError("No valid data rows to combine")
    # Filter None entries
    valid_rows = [r for r in rows if r is not None]
    if not valid_rows:
        raise ValueError("No valid data rows after filtering None entries")
    # Concatenate dictionaries of arrays
    keys = list(valid_rows[0].keys())
    concatenated = {k: np.concatenate([r[k] for r in valid_rows]) for k in keys}
    return CavityTmsvSensitivityResult(
        mean_total=concatenated["mean_total"],
        finesse=concatenated["finesse"],
        omega_values=concatenated["omega_values"],
        cfi_values=concatenated["cfi_values"],
        qfi_bound=concatenated["qfi_bound"],
        delta_omega_c=concatenated["delta_omega_c"],
        delta_omega_q=concatenated["delta_omega_q"],
        delta_omega_sql=concatenated["delta_omega_sql"],
        delta_omega_sql_eff=concatenated["delta_omega_sql_eff"],
        t_hold=concatenated["t_hold"],
        t_hold_eff=concatenated["t_hold_eff"],
        truncation_M=concatenated["truncation_M"],
        captured_norm=concatenated["captured_norm"],
    )


def _resolve_finesse_range(
    finesse_range: list[float],
    only: str | None,
) -> list[float]:
    """Apply ``--only F=...`` filter to the finesse range."""
    if only is not None and only.startswith("F="):
        target_f = float(only.split("=")[1])
        filtered = [f for f in finesse_range if abs(f - target_f) < 1e-10]
        if not filtered:
            raise ValueError(f"No matching finesse for --only {only}")
        return filtered
    return finesse_range


def generate_full_data(
    mean_total_range: list[float] | None = None,
    finesse_range: list[float] | None = None,
    force: bool = False,
    only: str | None = None,
) -> CavityTmsvSensitivityResult:
    """Generate full cavity-enhanced TMSV sensitivity data.

    Args:
        mean_total_range: List of ⟨N⟩ values (default: MEAN_TOTAL_RANGE).
        finesse_range: List of ℱ values (default: FINESSE_RANGE).
        force: If True, re-generate even if Parquet exists.
        only: If set (e.g. ``"F=10"``), only run for matching finesse.

    Returns:
        Combined sensitivity data across all (⟨N⟩, ℱ) combinations.
    """
    if mean_total_range is None:
        mean_total_range = MEAN_TOTAL_RANGE
    if finesse_range is None:
        finesse_range = FINESSE_RANGE
    finesse_range = _resolve_finesse_range(finesse_range, only)

    rows: list[dict] = []

    for idx, Fi in enumerate(finesse_range, start=1):
        print(f"  Sweeping ℱ={Fi} ({idx}/{len(finesse_range)})", flush=True)
        for Ni in mean_total_range:
            row_data = generate_single_cavity_point(Ni, Fi)
            if row_data is not None:
                rows.append(row_data)
            beam_splitter_unitary.cache_clear()
            _malloc_trim()

    if not rows:
        raise RuntimeError("No valid data generated")

    return _row_dicts_to_result(rows)


def _malloc_trim() -> None:
    """Try to release freed memory back to the OS (glibc only)."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _maybe_generate_full_data(
    force: bool = False,
    only: str | None = None,
    override_pq_path: Path | None = None,
) -> CavityTmsvSensitivityResult:
    """Load or generate cavity-enhanced TMSV sensitivity data.

    Args:
        force: Re-generate even if Parquet exists.
        only: If set, only run for matching finesse.
        override_pq_path: If set, use this path instead of the default.

    Returns:
        Sensitivity data.
    """
    pq_path = override_pq_path or _parquet_path("cavity_tmsv_sensitivity")

    if pq_path.exists() and not force:
        print(f"Loading existing data from {pq_path}")
        return CavityTmsvSensitivityResult.from_parquet(pq_path)

    print("Generating cavity-enhanced TMSV data")
    print(f"  ⟨N⟩ range: {MEAN_TOTAL_RANGE[0]}..{MEAN_TOTAL_RANGE[-1]}")
    print(f"  ℱ range: {FINESSE_RANGE[0]}..{FINESSE_RANGE[-1]}")
    print(f"  ω points per ℱ: {N_OMEGA_POINTS}")

    data = generate_full_data(force=force, only=only)
    data.save_parquet(pq_path)
    print(f"  Saved to {pq_path}")
    return data


# =============================================================================
# Best Sensitivity Extraction and Scaling Analysis
# =============================================================================


def _best_sensitivity_per_config(
    data: CavityTmsvSensitivityResult,
) -> pd.DataFrame:
    """Find the best (minimum) Δω_C for each (⟨N⟩, ℱ) combination.

    Args:
        data: Raw sensitivity data.

    Returns:
        DataFrame with columns: mean_total, finesse, best_delta_omega_c,
        best_omega, best_cfi, delta_omega_q, delta_omega_sql, ratio_to_sql.
    """
    df = data.to_dataframe()
    # Find row with minimum delta_omega_c per (mean_total, finesse) group
    idx = df.groupby(["mean_total", "finesse"])["delta_omega_c"].idxmin()
    best = df.loc[idx, :].copy()
    best = best.rename(
        columns={
            "delta_omega_c": "best_delta_omega_c",
            "omega_values": "best_omega",
            "cfi_values": "best_cfi",
        }
    )
    best["ratio_to_sql"] = best["best_delta_omega_c"] / best["delta_omega_sql"]
    best["ratio_to_sql_eff"] = best["best_delta_omega_c"] / best["delta_omega_sql_eff"]
    return best.reset_index(drop=True)


def _fit_alpha_per_finesse(
    best_df: pd.DataFrame,
    finesse_vals: list[float],
    min_N: float = 4.0,
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[str]]:
    """Fit α at each finesse value.

    Args:
        best_df: DataFrame from ``_best_sensitivity_per_config``.
        finesse_vals: Sorted list of unique finesse values.
        min_N: Minimum resource value for exponent fits.

    Returns:
        Tuple of (alpha_list, alpha_err_list, C_list, C_err_list,
        R2_list, fit_warnings).
    """
    alpha_list: list[float] = []
    alpha_err_list: list[float] = []
    C_list: list[float] = []
    C_err_list: list[float] = []
    R2_list: list[float] = []
    fit_warnings: list[str] = []

    for Fi in finesse_vals:
        subset = best_df[best_df["finesse"] == Fi]
        R_vals = subset["mean_total"].to_numpy(dtype=float)
        dt_vals = subset["best_delta_omega_c"].to_numpy(dtype=float)

        finite = np.isfinite(dt_vals) & (dt_vals > 0)
        if np.sum(finite) < 3:
            fit_warnings.append(f"ℱ={Fi}: insufficient finite points for fit")
            alpha_list.append(float("nan"))
            alpha_err_list.append(float("nan"))
            C_list.append(float("nan"))
            C_err_list.append(float("nan"))
            R2_list.append(float("nan"))
            continue

        result = fit_scaling_exponent(R_vals[finite], dt_vals[finite], min_N=int(min_N))
        if result.valid:
            alpha_list.append(result.alpha)
            alpha_err_list.append(result.alpha_err)
            C_list.append(result.C)
            C_err_list.append(result.C_err)
            R2_list.append(result.R_squared)
        else:
            fit_warnings.append(f"ℱ={Fi}: fit invalid — {result.warnings}")
            alpha_list.append(float("nan"))
            alpha_err_list.append(float("nan"))
            C_list.append(float("nan"))
            C_err_list.append(float("nan"))
            R2_list.append(float("nan"))

    return alpha_list, alpha_err_list, C_list, C_err_list, R2_list, fit_warnings


def _fit_beta_prefactor(best_df: pd.DataFrame) -> dict:
    """Fit β at each ⟨N⟩ with ≥3 finite finesse points, then aggregate.

    Returns:
        Dictionary with aggregated ``beta``, ``beta_err``, ``C0``,
        ``C0_err``, ``beta_R_squared``, and ``beta_warnings``.
    """
    N_vals = sorted(best_df["mean_total"].unique())

    beta_list: list[float] = []
    beta_err_list: list[float] = []
    C0_list: list[float] = []
    C0_err_list: list[float] = []
    R2_list: list[float] = []

    for N_val in N_vals:
        subset = best_df[
            (best_df["mean_total"] == N_val)
            & np.isfinite(best_df["best_delta_omega_c"])
        ]
        if len(subset) < 3:
            continue

        F_vals = subset["finesse"].to_numpy(dtype=float)
        dt_F = subset["best_delta_omega_c"].to_numpy(dtype=float)
        pos = (F_vals > 0) & (dt_F > 0) & np.isfinite(dt_F)
        if np.sum(pos) < 3:
            continue

        log_F = np.log(F_vals[pos])
        log_dt = np.log(dt_F[pos])
        linreg = stats.linregress(log_F, log_dt)

        beta_list.append(float(-linreg.slope))
        beta_err_list.append(float(linreg.stderr))
        C0_list.append(float(np.exp(linreg.intercept)))
        C0_err_list.append(
            float(np.exp(linreg.intercept)) * float(linreg.intercept_stderr)
        )
        R2_list.append(float(linreg.rvalue**2))

    if len(beta_list) >= 1:
        beta_arr = np.array(beta_list)
        beta_err_arr = np.array(beta_err_list)
        C0_arr = np.array(C0_list)
        R2_arr = np.array(R2_list)

        # Use std across N if multiple values, else mean of individual stderr
        beta_aggregated = float(np.mean(beta_arr))
        beta_err_aggregated = (
            float(np.std(beta_arr, ddof=1))
            if len(beta_arr) > 1
            else float(np.mean(beta_err_arr))
        )
        C0_aggregated = float(np.mean(C0_arr))
        C0_err_aggregated = (
            float(np.std(C0_arr, ddof=1))
            if len(C0_arr) > 1
            else float(np.mean(C0_err_list))
        )

        return {
            "beta": beta_aggregated,
            "beta_err": beta_err_aggregated,
            "C0": C0_aggregated,
            "C0_err": C0_err_aggregated,
            "beta_R_squared": float(np.mean(R2_arr)),
            "beta_warnings": [],
        }

    return {
        "beta": float("nan"),
        "beta_err": float("nan"),
        "C0": float("nan"),
        "C0_err": float("nan"),
        "beta_R_squared": float("nan"),
        "beta_warnings": ["No ⟨N⟩ with ≥3 finite finesse points for β fit"],
    }


def _fit_scaling_per_finesse(
    best_df: pd.DataFrame,
    min_N: float = 4.0,
) -> CavityTmsvScalingFit:
    """Fit the scaling exponent α for each finesse value.

    Also fits the overall α from all data combined, and the prefactor
    scaling exponent β from Δω_min ∝ ℱ^{-β} aggregated across all ⟨N⟩.

    Args:
        best_df: DataFrame from ``_best_sensitivity_per_config``.
        min_N: Minimum resource value for exponent fits.

    Returns:
        CavityTmsvScalingFit with per-finesse and overall fits.
    """
    finesse_vals = sorted(best_df["finesse"].unique())

    # 1. Per-finesse α fits
    alpha_list, _alpha_err_list, C_list, _C_err_list, _R2_list, fit_warnings = (
        _fit_alpha_per_finesse(best_df, finesse_vals, min_N)
    )

    # 2. Overall α fit: pool all data
    all_R = best_df["mean_total"].to_numpy(dtype=float)
    all_dt = best_df["best_delta_omega_c"].to_numpy(dtype=float)
    all_finite = np.isfinite(all_dt) & (all_dt > 0) & (all_R >= min_N)
    if np.sum(all_finite) >= 3:
        overall_result = fit_scaling_exponent(
            all_R[all_finite], all_dt[all_finite], min_N=int(min_N)
        )
        alpha_overall = overall_result.alpha if overall_result.valid else float("nan")
        alpha_overall_err = (
            overall_result.alpha_err if overall_result.valid else float("nan")
        )
    else:
        alpha_overall = float("nan")
        alpha_overall_err = float("nan")
        fit_warnings.append("Insufficient points for overall α fit")

    # 3. β prefactor fit aggregated across ⟨N⟩
    beta_result = _fit_beta_prefactor(best_df)
    fit_warnings.extend(beta_result["beta_warnings"])

    return CavityTmsvScalingFit(
        finesse_values=np.array(finesse_vals, dtype=float),
        alpha_values=np.array(alpha_list, dtype=float),
        C_values=np.array(C_list, dtype=float),
        alpha_overall=alpha_overall,
        alpha_overall_err=alpha_overall_err,
        beta=beta_result["beta"],
        beta_err=beta_result["beta_err"],
        C0=beta_result["C0"],
        valid=len(fit_warnings) == 0,
        warnings_list=fit_warnings,
    )


# =============================================================================
# Plot Functions
# =============================================================================


def plot_delta_omega_overlay(
    data: CavityTmsvSensitivityResult,
    finesse: float,
    selected_N: list[float] | None = None,
    save_path: str | Path | None = None,
) -> Path:
    r"""Plot Δω_C vs ω for selected ⟨N⟩ values at a fixed finesse.

    Args:
        data: Raw sensitivity data.
        finesse: Which finesse value to plot.
        selected_N: Which ⟨N⟩ values to include (auto-sampled if None).
        save_path: Output SVG path.

    Returns:
        Path to saved SVG.
    """
    df = data.to_dataframe()
    sub = df[np.isclose(df["finesse"], finesse, rtol=1e-10)]

    n_N = len(sub["mean_total"].unique())
    if selected_N is None:
        step = max(1, n_N // 7)
        selected_N = sorted(sub["mean_total"].unique())[::step]

    if save_path is None:
        save_path = _fig_path(f"delta_omega_F{finesse}")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.colormaps[COLORMAP]
    colors = cmap(np.linspace(0.15, 0.85, len(selected_N)))

    for idx, N_val in enumerate(selected_N):
        mask = np.isclose(sub["mean_total"], N_val, rtol=1e-10)
        pt = sub[mask].sort_values("omega_values")
        omega_v = pt["omega_values"].to_numpy()
        dt_c = pt["delta_omega_c"].to_numpy()
        dt_q = pt["delta_omega_q"].iloc[0]

        c_finite = np.isfinite(dt_c)
        if np.any(c_finite):
            ax.semilogy(
                omega_v[c_finite],
                dt_c[c_finite],
                color=colors[idx],
                linewidth=1.5,
                label=rf"$\langle N\rangle$={N_val:.0f}  $\Delta\omega_{{\mathrm{{C}}}}$",
            )
        ax.axhline(
            y=float(dt_q),
            color=colors[idx],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(rf"Cavity-Enhanced TMSV MZI ($\mathcal{{F}} = {finesse}$)")

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best", ncol=1)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scaling(
    data: CavityTmsvSensitivityResult,
    scaling_fit: CavityTmsvScalingFit | None = None,
    selected_F: Sequence[float] | None = None,
    save_path: str | Path | None = None,
) -> Path:
    r"""Log-log plot of best Δω vs ⟨N⟩ for multiple finesse values.

    Args:
        data: Raw sensitivity data.
        scaling_fit: Optional scaling fit results (adds fit lines).
        selected_F: Which finesse values to include (auto-sampled if None).
        save_path: Output SVG path.

    Returns:
        Path to saved SVG.
    """
    best_df = _best_sensitivity_per_config(data)
    all_F = sorted(best_df["finesse"].unique())

    if selected_F is None:
        step = max(1, len(all_F) // 10)
        selected_F = all_F[::step]

    if save_path is None:
        save_path = _fig_path("scaling")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Reference lines
    R_ref = np.logspace(np.log10(2), np.log10(40), 100)
    ax.plot(
        R_ref,
        1.0 / (H_t * R_ref**1.0),
        "k--",
        alpha=0.3,
        label=r"$\propto 1/\langle N\rangle$ (Heisenberg)",
    )
    ax.plot(
        R_ref,
        1.0 / (H_t * np.sqrt(R_ref)),
        "k:",
        alpha=0.3,
        label=r"$\propto 1/\sqrt{\langle N\rangle}$ (SQL)",
    )

    colours = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "X"]

    for i, Fi in enumerate(selected_F):
        colour = colours[i % len(colours)]
        marker = markers[i % len(markers)]
        sub = best_df[np.isclose(best_df["finesse"], Fi, rtol=1e-10)]

        R_vals = sub["mean_total"].to_numpy()
        dt_best = sub["best_delta_omega_c"].to_numpy()
        dt_q = sub["delta_omega_q"].to_numpy()

        # QFI bound (dashed)
        ax.loglog(
            R_vals,
            dt_q,
            f"{colour}--",
            alpha=0.4,
            label=rf"$\mathcal{{F}}={Fi}$ QFI",
        )
        # Best Δω_C (solid)
        ax.loglog(
            R_vals,
            dt_best,
            f"{colour}{marker}-",
            label=rf"$\mathcal{{F}}={Fi}$ best $\Delta\omega_{{\mathrm{{C}}}}$",
        )

        # Add fit line if available
        if scaling_fit is not None and scaling_fit.valid:
            mask_f = np.isclose(scaling_fit.finesse_values, Fi, rtol=1e-10)
            if np.any(mask_f):
                idx_f = int(np.where(mask_f)[0][0])
                alpha_val = scaling_fit.alpha_values[idx_f]
                C_val = scaling_fit.C_values[idx_f]
                if np.isfinite(alpha_val) and np.isfinite(C_val):
                    R_fit = np.array(R_vals, dtype=float)
                    dt_fit = C_val * R_fit**alpha_val
                    ax.loglog(
                        R_fit,
                        dt_fit,
                        f"{colour}--",
                        alpha=0.7,
                        linewidth=1.5,
                        label=rf"$\mathcal{{F}}={Fi}$: $\alpha={alpha_val:.3f}$",
                    )

    ax.set_xlabel(r"$\langle N\rangle$ (total mean photon number)")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title("Cavity-Enhanced TMSV MZI Scaling")

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best", ncol=1)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_prefactor_scaling(
    data: CavityTmsvSensitivityResult,
    scaling_fit: CavityTmsvScalingFit | None = None,
    selected_N: list[float] | None = None,
    save_path: str | Path | None = None,
) -> Path:
    r"""Log-log plot of best Δω vs ℱ at selected ⟨N⟩ values.

    Args:
        data: Raw sensitivity data.
        scaling_fit: Optional scaling fit results (adds β fit line).
        selected_N: Which ⟨N⟩ values to include (auto-sampled if None).
        save_path: Output SVG path.

    Returns:
        Path to saved SVG.
    """
    best_df = _best_sensitivity_per_config(data)
    all_N = sorted(best_df["mean_total"].unique())

    if selected_N is None:
        step = max(1, len(all_N) // 5)
        selected_N = all_N[::step]

    if save_path is None:
        save_path = _fig_path("prefactor_scaling")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Reference: Δω ∝ 1/ℱ
    F_ref = np.logspace(0, 3, 100)
    ax.plot(
        F_ref,
        1.0 / F_ref,
        "k--",
        alpha=0.3,
        label=r"$\propto 1/\mathcal{F}$ (predicted)",
    )

    colours = ["C0", "C1", "C2", "C3", "C4"]
    markers = ["o", "s", "^", "D", "v"]

    for i, N_val in enumerate(selected_N):
        colour = colours[i % len(colours)]
        marker = markers[i % len(markers)]
        sub = best_df[np.isclose(best_df["mean_total"], N_val, rtol=1e-10)]
        F_vals = sub["finesse"].to_numpy()
        dt_best = sub["best_delta_omega_c"].to_numpy()

        ax.loglog(
            F_vals,
            dt_best,
            f"{colour}{marker}-",
            label=rf"$\langle N\rangle={N_val:.0f}$",
        )

    # β fit line at the largest ⟨N⟩
    if scaling_fit is not None and scaling_fit.valid and np.isfinite(scaling_fit.beta):
        F_line = np.logspace(0, 3, 100)
        dt_line = scaling_fit.C0 * F_line ** (-scaling_fit.beta)
        ax.loglog(
            F_line,
            dt_line,
            "r--",
            alpha=0.7,
            linewidth=2,
            label=rf"$\beta = {scaling_fit.beta:.3f} \pm {scaling_fit.beta_err:.3f}$",
        )

    ax.set_xlabel(r"Cavity finesse $\mathcal{F}$")
    ax.set_ylabel(r"Best $\Delta\omega$")
    ax.set_title("Prefactor Scaling with Cavity Finesse")

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _generate_plots(
    data: CavityTmsvSensitivityResult,
    scaling_fit: CavityTmsvScalingFit | None = None,
    force: bool = False,
    only: str | None = None,
    selected_F: Sequence[float] | None = None,
) -> None:
    """Generate all plots from the computed sensitivity data.

    Args:
        data: Raw sensitivity data.
        scaling_fit: Optional scaling fit results.
        force: Re-generate even if SVG exists.
        only: If set (e.g. ``"F=10"``), only plot for matching finesse.
        selected_F: Which finesse values to include in overlays and scaling
            plot.  Auto-selects from ``data`` if ``None``.
    """
    # Auto-select finesse values from data if not provided
    if selected_F is None:
        best_df = _best_sensitivity_per_config(data)
        all_F = sorted(best_df["finesse"].unique())
        step = max(1, len(all_F) // 4)
        selected_F = all_F[::step]
    else:
        selected_F = list(selected_F)

    # Apply --only filter
    if only is not None and only.startswith("F="):
        target_f = float(only.split("=")[1])
        selected_F = [f for f in selected_F if abs(f - target_f) < 1e-10]

    # Δω overlay plots for selected finesse values
    _plot_each_overlay(data, selected_F, force)

    # Scaling plot
    scaling_path = _fig_path("scaling")
    if not scaling_path.exists() or force:
        plot_scaling(
            data,
            scaling_fit=scaling_fit,
            selected_F=selected_F,
            save_path=scaling_path,
        )
        print(f"  Plotted {scaling_path}")

    # Prefactor scaling plot
    prefactor_path = _fig_path("prefactor_scaling")
    if not prefactor_path.exists() or force:
        plot_prefactor_scaling(data, scaling_fit=scaling_fit, save_path=prefactor_path)
        print(f"  Plotted {prefactor_path}")


def _plot_each_overlay(
    data: CavityTmsvSensitivityResult,
    selected_F: Sequence[float],
    force: bool,
) -> None:
    """Plot a delta-omega overlay for every finesse in *selected_F*."""
    for Fi in selected_F:
        save_path = _fig_path(f"delta_omega_F{Fi}")
        if not save_path.exists() or force:
            plot_delta_omega_overlay(data, finesse=Fi, save_path=save_path)
            print(f"  Plotted {save_path}")


# =============================================================================
# Main Pipeline
# =============================================================================


def generate_all(
    force: bool = False,
    only: str | None = None,
    override_pq_path: Path | None = None,
) -> tuple[CavityTmsvSensitivityResult, CavityTmsvScalingFit]:
    """Generate all data and figures for the report.

    Args:
        force: If True, re-generate even if Parquet files exist.
        only: If set (e.g. ``"F=10"``), only run for matching finesse.
        override_pq_path: If set, use this path instead of the default
            (useful for tests to avoid overwriting shared data).

    Returns:
        Tuple of (sensitivity_data, scaling_fit).
    """
    data = _maybe_generate_full_data(
        force=force,
        only=only,
        override_pq_path=override_pq_path,
    )

    print("Fitting scaling exponents...")
    best_df = _best_sensitivity_per_config(data)
    scaling_fit = _fit_scaling_per_finesse(best_df)

    print(f"  Overall α (combined): {scaling_fit.alpha_overall:.4f}")
    print(f"  Prefactor β: {scaling_fit.beta:.4f} ± {scaling_fit.beta_err:.4f}")

    _generate_plots(data, scaling_fit=scaling_fit, force=force, only=only)

    return data, scaling_fit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cavity-Enhanced TMSV MZI sensitivity simulation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate data and figures even if files exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only run for specific finesse (e.g. 'F=10')",
    )
    parser.add_argument(
        "--pq-path",
        type=str,
        default=None,
        help="Override Parquet path (useful for isolated test runs)",
    )
    args = parser.parse_args()

    pq_path = Path(args.pq_path) if args.pq_path else None
    generate_all(force=args.force, only=args.only, override_pq_path=pq_path)


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    main()

"""
Scaling Exponent Fit Utilities for the Unified Sensitivity Survey.

Provides log-log linear regression to extract α in Δφ ∝ N^α,
with quality metrics (R², α_err, prefactor C).

Physical Model:
- Sensitivity scaling: Δφ = C · N^α
- Standard Quantum Limit (SQL): α = -1/2 (Δφ ∝ 1/√N)
- Heisenberg Limit: α = -1 (Δφ ∝ 1/N)
- log-log linear regression: log(Δφ) = log(C) + α · log(N)

Units:
- Dimensionless throughout. N is particle number.
- α and C are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# Result Container
# =============================================================================


@dataclass
class ScalingFitResult:
    """Result of a scaling exponent fit.

    Attributes:
        alpha: Scaling exponent α in Δφ = C·N^α.
        alpha_err: Standard error of α from covariance matrix.
        C: Prefactor C in Δφ = C·N^α.
        C_err: Standard error of C from covariance matrix.
        R_squared: Goodness-of-fit (coefficient of determination).
        N_values: N values used in the fit (after filtering).
        delta_phi_values: Sensitivity values used in the fit.
        valid: Whether the fit succeeded (sufficient points, finite results).
        warnings: Any warnings (e.g., too few points, low R², negative values).

    """

    alpha: float
    alpha_err: float
    C: float
    C_err: float
    R_squared: float
    N_values: np.ndarray
    delta_phi_values: np.ndarray
    valid: bool = True
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Core Fitting
# =============================================================================


def fit_scaling_exponent(
    N: np.ndarray,
    delta_phi: np.ndarray,
    min_N: int = 4,
    R_squared_threshold: float = 0.9,
) -> ScalingFitResult:
    """Fit Δφ = C·N^α via log-log linear regression.

    Performs a log-log linear fit via scipy.stats.linregress to extract the
    scaling exponent α and prefactor C.

    The fitting procedure:
        1. Filters out N < min_N to avoid finite-size artifacts
        2. Removes non-positive values (log undefined)
        3. Performs log-log linear regression via scipy.stats.linregress
        4. Computes R² = rvalue² (square of Pearson correlation coefficient)
        5. Extracts α, α_err, C, C_err from slope, stderr, and intercept_stderr

    Args:
        N: Array of particle/atom numbers. Must be positive.
        delta_phi: Array of phase sensitivity values. Must be positive.
        min_N: Minimum N to include in the fit. Excluding small N avoids
            finite-size artifacts where scaling may not yet be asymptotic.
            Default: 4.
        R_squared_threshold: Warning threshold for R². If the fit R² is
            below this value, a warning is added to the result. Default: 0.9.

    Returns:
        ScalingFitResult with fit parameters, quality metrics, and any warnings.

    Raises:
        ValueError: If input arrays are empty or have different lengths.
        ValueError: If any N or delta_phi values are NaN or infinite.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> N = np.array([4, 8, 16, 32, 64])
        >>> # SQL scaling: Δφ = 1/√N
        >>> delta_phi = 1.0 / np.sqrt(N) * (1 + 0.01 * rng.normal(size=len(N)))
        >>> result = fit_scaling_exponent(N, delta_phi, min_N=4)
        >>> np.isclose(result.alpha, -0.5, atol=0.05)
        True
        >>> result.R_squared > 0.90
        True

    """
    # --- Input validation ---
    N_arr = np.asarray(N, dtype=float).ravel()
    delta_arr = np.asarray(delta_phi, dtype=float).ravel()

    if len(N_arr) != len(delta_arr):
        raise ValueError(
            f"N and delta_phi must have same length, "
            f"got {len(N_arr)} and {len(delta_arr)}",
        )
    if len(N_arr) == 0:
        raise ValueError("Input arrays must not be empty")

    if np.any(np.isnan(N_arr)) or np.any(np.isinf(N_arr)):
        raise ValueError("N contains NaN or infinite values")
    if np.any(np.isnan(delta_arr)) or np.any(np.isinf(delta_arr)):
        raise ValueError("delta_phi contains NaN or infinite values")

    warnings: list[str] = []

    # --- Filter valid points ---
    # Exclude N < min_N (finite-size artifacts)
    mask = N_arr >= min_N

    if not np.any(mask):
        return ScalingFitResult(
            alpha=0.0,
            alpha_err=0.0,
            C=0.0,
            C_err=0.0,
            R_squared=0.0,
            N_values=N_arr,
            delta_phi_values=delta_arr,
            valid=False,
            warnings=["No data points satisfy N >= min_N"],
        )

    N_fit = N_arr[mask]
    delta_fit = delta_arr[mask]

    # Exclude non-positive values (log is undefined)
    pos_mask = delta_fit > 0
    if not np.all(pos_mask):
        warnings.append(f"Excluded {np.sum(~pos_mask)} non-positive delta_phi values")
        N_fit = N_fit[pos_mask]
        delta_fit = delta_fit[pos_mask]

    if len(N_fit) < 3:
        return ScalingFitResult(
            alpha=0.0,
            alpha_err=0.0,
            C=0.0,
            C_err=0.0,
            R_squared=0.0,
            N_values=N_arr,
            delta_phi_values=delta_arr,
            valid=False,
            warnings=[
                "Too few valid points after filtering (need >= 3 for covariance)",
            ],
        )

    # --- Log-log linear regression via scipy.stats.linregress ---
    log_N = np.log(N_fit)
    log_delta = np.log(delta_fit)

    try:
        res = stats.linregress(log_N, log_delta)
    except ValueError as e:
        return ScalingFitResult(
            alpha=0.0,
            alpha_err=0.0,
            C=0.0,
            C_err=0.0,
            R_squared=0.0,
            N_values=N_fit,
            delta_phi_values=delta_fit,
            valid=False,
            warnings=[f"Linear fit failed: {e}"],
        )

    alpha = res.slope
    log_C = res.intercept
    alpha_err = res.stderr
    log_C_err = res.intercept_stderr
    R_squared = res.rvalue**2

    # Prefactor C = exp(log_C), with relative error propagation
    C_val = np.exp(log_C)
    C_err = C_val * log_C_err  # dC = exp(log_C) * d(log_C)

    # --- Quality checks ---
    if R_squared < R_squared_threshold:
        warnings.append(f"R² = {R_squared:.4f} < threshold {R_squared_threshold:.2f}")

    if alpha_err / max(abs(alpha), 1e-10) > 1.0:
        warnings.append(
            f"Relative error in α is large "
            f"(α_err/|α| = {alpha_err / max(abs(alpha), 1e-10):.2f})",
        )

    return ScalingFitResult(
        alpha=float(alpha),
        alpha_err=float(alpha_err),
        C=float(C_val),
        C_err=float(C_err),
        R_squared=float(R_squared),
        N_values=N_fit,
        delta_phi_values=delta_fit,
        valid=True,
        warnings=warnings,
    )


# =============================================================================
# Quality Validation
# =============================================================================


def validate_fit_quality(result: ScalingFitResult) -> bool:
    """Check if fit quality is acceptable.

    A fit is considered acceptable if:
    - result.valid is True (enough points, fit succeeded)
    - R² >= 0.5 (at least moderate correlation)
    - α_err / |α| < 1.0 (relative error bounded)
    - No more than 2 warnings (excessive warnings indicate problems)

    Args:
        result: The ScalingFitResult to validate.

    Returns:
        True if the fit quality is acceptable for further analysis.

    """
    if not result.valid:
        return False

    if result.R_squared < 0.5:
        return False

    # Check that relative error is not catastrophic
    alpha_abs = max(abs(result.alpha), 1e-10)
    if result.alpha_err / alpha_abs >= 1.0:
        return False

    # Too many warnings indicate reliability concerns
    return not len(result.warnings) > 2


def compare_exponents(
    results: dict[str, ScalingFitResult],
) -> pd.DataFrame:
    """Compare scaling exponents across different models in a table.

    Constructs a DataFrame comparing α, α_err, C, R², and fit validity
    for each model in the input dictionary.

    Args:
        results: Dictionary mapping model names/labels to their
            ScalingFitResult objects.

    Returns:
        DataFrame with columns:
            model, alpha, alpha_err, C, R_squared, valid, n_warnings, n_points
        Sorted by alpha (most negative first = best scaling).

    Example:
        >>> N = np.array([4, 8, 16, 32, 64])
        >>> sql = 1.0 / np.sqrt(N)
        >>> hl = 1.0 / N
        >>> r_sql = fit_scaling_exponent(N, sql)
        >>> r_hl = fit_scaling_exponent(N, hl)
        >>> df = compare_exponents({"SQL": r_sql, "Heisenberg": r_hl})
        >>> len(df)
        2
        >>> df.loc["Heisenberg", "alpha"]  # Should be -1
        -1.0

    """
    rows = []
    for label, result in results.items():
        rows.append(
            {
                "model": label,
                "alpha": result.alpha,
                "alpha_err": result.alpha_err,
                "C": result.C,
                "C_err": result.C_err,
                "R_squared": result.R_squared,
                "valid": result.valid,
                "n_warnings": len(result.warnings),
                "n_points": len(result.N_values),
                "warnings": "; ".join(result.warnings) if result.warnings else "",
            },
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("alpha", ascending=True).reset_index(drop=True)

    return df

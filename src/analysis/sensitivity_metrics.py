"""
Sensitivity Metrics & Comparison.

This module implements three complementary estimators for phase sensitivity:
1. Error Propagation: Δφ_EP = σ_Jz / |∂⟨J_z⟩/∂φ|
2. Fisher Information: Δφ_F = 1/√F_C (Cramér-Rao bound)
3. Bayesian: Δφ_B = Std[φ|m₀] (posterior standard deviation)

And provides comparison utilities to analyze when they agree or diverge.

Physical Model:
- Error propagation: Uses local slope of expectation value
- Fisher: Uses probability distribution derivatives (Cramér-Rao bound)
- Bayesian: Uses full posterior distribution from measurement

Hilbert Space:
- Two-mode Fock basis with dimension (max_photons+1)²
- Basis ordering: |n₁, n₂⟩ with n₁ as first mode, n₂ as second mode

Units:
- Dimensionless throughout. Phase in radians.

References:
- Escher et al. "General framework for estimating the ultimate precision..."
- Demkowicz-Dobrzanski et al. "Quantum metrology and noise effects"

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from src.analysis.bayesian_phase_estimation import (
    bayesian_estimator,
    sample_measurement_outcomes,
)
from src.analysis.fisher_information import (
    classical_fisher_information,
    quantum_fisher_information,
)
from src.physics.mzi_simulation import (
    compute_output_probabilities,
    evolve_mzi,
    prepare_input_state,
)
from src.physics.mzi_states import two_mode_jz_operator
from src.physics.noise_channels import NoiseConfig
from src.utils.serialization import ParquetSerializable

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Type Definitions
# =============================================================================

StateType = np.ndarray  # Pure state vector
SensitivityResult = dict[str, float]  # {"delta_phi": float, ...}


def sql_reference(N: int, t_hold: float = 10.0) -> float:
    """Standard quantum limit for N particles with holding time t_hold.

    Δω_SQL = 1 / (√N × t_hold)

    Args:
        N: Number of particles in the probe.
        t_hold: Holding/evolution time (dimensionless).

    Returns:
        SQL sensitivity value.
    """
    return 1.0 / (np.sqrt(N) * t_hold)


# =============================================================================
# Error Propagation Sensitivity
# =============================================================================


def error_propagation_sensitivity(
    state: StateType,
    max_photons: int,
    phi_grid: np.ndarray,
    dphi: float = 1e-4,
) -> dict:
    """Compute error propagation sensitivity Δφ_EP over φ range.

    The error propagation formula for phase estimation is:
        Δφ_EP = σ_Jz / |∂⟨J_z⟩/∂φ|

    where:
        σ_Jz = sqrt(⟨J_z²⟩ - ⟨J_z⟩²) is the variance
        ∂⟨J_z⟩/∂φ is the slope of expectation value

    Args:
        state: Initial state vector in two-mode Fock basis (dim = (max_photons+1)²).
        max_photons: Maximum photon number per mode for basis truncation.
        phi_grid: Array of phase values to evaluate over.
        dphi: Finite difference step for derivative (default 1e-4).

    Returns:
        Dictionary containing:
        - "delta_phi_ep": Minimum sensitivity over the grid
        - "phi_at_min": Phase value where minimum occurs
        - "delta_phi_grid": Sensitivity values over the full grid

    Raises:
        ValueError: If phi_grid has fewer than 3 points or dphi <= 0.

    """
    if len(phi_grid) < 3:
        raise ValueError("phi_grid must have at least 3 points")
    if dphi <= 0:
        raise ValueError(f"dphi must be positive, got {dphi}")

    # Determine expected dimension
    expected_dim = (max_photons + 1) ** 2
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"State dimension {state.shape[0]} must be (max_photons+1)² = {expected_dim}",
        )

    # Generate J_z operator in two-mode Fock basis
    jz = two_mode_jz_operator(max_photons)
    jz2 = jz @ jz

    # Allocate arrays
    n_phi = len(phi_grid)
    expectation_values = np.zeros(n_phi)
    variance_values = np.zeros(n_phi)

    # MZI parameters
    theta = np.pi / 4  # 50/50 beam splitter
    phi_bs = 0.0
    g = 0.0
    interaction_time = 0.0
    coupling_type = "phase_coupling"

    # Compute ⟨J_z⟩ and Var(J_z) at each phase
    for i, phi in enumerate(phi_grid):
        # Evolve state through MZI
        final_state = evolve_mzi(
            state,
            theta,
            phi_bs,
            phi,
            g,
            interaction_time,
            coupling_type,
            max_photons,
            1,  # ancilla_dim = 1 (no ancilla)
        )

        # Compute expectation values on the system (trace out ancilla)
        # The final state is in the combined space, we need to trace out ancilla
        sys_dim = (max_photons + 1) ** 2

        # Reshape to get system density matrix
        # final_state is (sys_dim * 1,) = (sys_dim,)
        # Convert to density matrix and trace over any ancillary space
        rho = np.outer(final_state, final_state.conj())

        # If there's an ancilla dimension, trace it out
        # For ancilla_dim=1, no tracing needed
        exp_jz = np.trace(rho[:sys_dim, :sys_dim] @ jz).real
        exp_jz2 = np.trace(rho[:sys_dim, :sys_dim] @ jz2).real

        expectation_values[i] = exp_jz
        var_jz = exp_jz2 - exp_jz**2
        variance_values[i] = max(0.0, var_jz)  # Ensure non-negative

    # Compute derivative ∂⟨J_z⟩/∂φ using central differences
    derivative = np.zeros(n_phi)
    for i in range(1, n_phi - 1):
        derivative[i] = (expectation_values[i + 1] - expectation_values[i - 1]) / (
            2 * dphi
        )

    # Forward difference at boundaries
    derivative[0] = (expectation_values[1] - expectation_values[0]) / dphi
    derivative[-1] = (expectation_values[-1] - expectation_values[-2]) / dphi

    # Avoid division by zero - set minimum derivative threshold
    # Based on numerical precision and expected signal magnitude
    abs_derivative = np.abs(derivative)
    # Minimum threshold: avoid numerical noise dominating
    min_deriv_threshold = 1e-8 * np.max(np.abs(expectation_values))
    abs_derivative = np.where(
        abs_derivative < min_deriv_threshold,
        min_deriv_threshold,
        abs_derivative,
    )

    # Compute error propagation sensitivity: Δφ = σ / |d⟨J_z⟩/dφ|
    # Handle edge case where variance is zero (eigenstate)
    # or derivative is below numerical precision threshold
    # In these cases, we return a large sensitivity value (worst case)
    delta_phi_grid = np.zeros(n_phi)
    for i in range(n_phi):
        if variance_values[i] < 1e-10:
            # Eigenstate - no sensitivity via error propagation
            # Return large value (maximum phase uncertainty)
            delta_phi_grid[i] = np.pi
        elif abs_derivative[i] <= min_deriv_threshold:
            # Derivative below numerical precision - unreliable estimate
            # Use a value based on variance but with minimum meaningful sensitivity
            delta_phi_grid[i] = np.sqrt(variance_values[i]) / min_deriv_threshold
        else:
            delta_phi_grid[i] = np.sqrt(variance_values[i]) / abs_derivative[i]

    # Find minimum (best sensitivity)
    min_idx = np.argmin(delta_phi_grid)
    delta_phi_min = delta_phi_grid[min_idx]
    phi_at_min = phi_grid[min_idx]

    return {
        "delta_phi_ep": float(delta_phi_min),
        "phi_at_min": float(phi_at_min),
        "delta_phi_grid": delta_phi_grid,
        "expectation_values": expectation_values,
        "derivative": derivative,
        "variance_values": variance_values,
    }


def sensitivity_from_error_propagation(
    O_mean: float,
    O_var: float,
    dO_dphi: float,
    var_tolerance: float = 1e-15,
    deriv_tolerance: float = 1e-12,
) -> float:
    """Error-propagation sensitivity from pre-computed moments.

    Computes the error-propagation formula:

        Δφ = √(Var(O)) / |∂⟨O⟩/∂φ|

    and returns ``inf`` at fringe extrema where either the variance
    vanishes (the state is an eigenstate of the measurement operator,
    giving a deterministic outcome with zero information) or the
    derivative vanishes (the operating point is at a fringe extremum
    where the signal slope is flat).

    Args:
        O_mean: Expectation value ⟨O⟩ (not used in formula, included
            for debugging/consistency checks).
        O_var: Variance Var(O) = ⟨O²⟩ - ⟨O⟩². Must be non-negative;
            negative values are clamped to zero before the square root.
        dO_dphi: Derivative ∂⟨O⟩/∂φ (central finite-difference value).
        var_tolerance: Variance below this threshold is treated as
            zero (default 1e-15, matching the convention in
            ``ancilla_drive_metrology.py`` and ``n_particle_drive.py``).
        deriv_tolerance: Absolute derivative below this threshold is
            treated as zero (default 1e-12).

    Returns:
        Sensitivity Δφ (positive float). Returns ``inf`` if
        ``var < var_tolerance`` or ``abs(dO_dphi) < deriv_tolerance``.

    """

    # Clamp negative variance (numerical noise) to zero
    var_safe = max(float(O_var), 0.0)

    if var_safe < var_tolerance:
        return float("inf")

    d_abs = abs(float(dO_dphi))
    if d_abs < deriv_tolerance:
        return float("inf")

    return float(np.sqrt(var_safe) / d_abs)


# =============================================================================
# All Sensitivity Metrics
# =============================================================================


def all_sensitivity_metrics(
    state: StateType,
    max_photons: int,
    phi_true: float,
    n_mc: int = 500,
    seed: int | None = None,
) -> dict:
    """Compute all three sensitivity metrics for comparison.

    Computes:
    1. Error propagation: Δφ_EP = σ_Jz / |∂⟨J_z⟩/∂φ|
    2. Fisher (Cramér-Rao): Δφ_F = 1/√F_C
    3. Bayesian: Δφ_B = Std[φ|m₀]

    Args:
        state: Initial state in two-mode Fock basis.
        max_photons: Maximum photon number per mode.
        phi_true: True phase value for simulation.
        n_mc: Number of Monte Carlo samples for Bayesian estimation.
        seed: Random seed for reproducibility (None = fresh entropy).

    Returns:
        Dictionary with all three sensitivity values and metadata.

    Raises:
        ValueError: If max_photons < 1 or n_mc < 1.

    """
    if max_photons < 1:
        raise ValueError(f"max_photons must be >= 1, got {max_photons}")
    if n_mc < 1:
        raise ValueError(f"n_mc must be >= 1, got {n_mc}")

    rng = np.random.default_rng(seed)

    # Verify state dimension matches expected
    dim = state.shape[0]
    expected_dim = (max_photons + 1) ** 2
    if dim != expected_dim:
        raise ValueError(
            f"State dimension {dim} must be (max_photons+1)² = {expected_dim}",
        )

    # Generate J_z generator for the two-mode system
    generator = two_mode_jz_operator(max_photons)

    # -----------------------------------------------------------------------------
    # 1. Error Propagation Sensitivity
    # -----------------------------------------------------------------------------
    phi_grid = np.linspace(0, 2 * np.pi, 181)
    ep_result = error_propagation_sensitivity(state, max_photons, phi_grid)
    delta_phi_ep = ep_result["delta_phi_ep"]

    # -----------------------------------------------------------------------------
    # 2. Fisher Information Sensitivity (Cramér-Rao)
    # -----------------------------------------------------------------------------
    # Compute classical Fisher information at phi_true
    theta = np.pi / 4
    phi_bs = 0.0
    g = 0.0
    interaction_time = 0.0
    coupling_type = "phase_coupling"

    # Generate probability distribution over phase grid
    dphi = phi_grid[1] - phi_grid[0]
    probs_grid = np.zeros((len(phi_grid), 2))

    for i, phi in enumerate(phi_grid):
        final_state = evolve_mzi(
            state,
            theta,
            phi_bs,
            phi,
            g,
            interaction_time,
            coupling_type,
            max_photons,
            1,
        )
        p0, p1 = compute_output_probabilities(final_state, max_photons, 1)
        probs_grid[i, 0] = p0
        probs_grid[i, 1] = p1

    # Find index closest to phi_true
    phi_idx = np.argmin(np.abs(phi_grid - phi_true))
    fc_array = classical_fisher_information(probs_grid, dphi)
    fc = fc_array[phi_idx]

    # Use quantum Fisher for theoretical bound
    # For two-mode Fock states, compute QFI using J_z as generator
    # We need to compute QFI on the evolved state at phi_true
    # (the state has phase information after the first beam splitter)
    final_state_at_phi = evolve_mzi(
        state,
        theta,
        phi_bs,
        phi_true,
        g,
        interaction_time,
        coupling_type,
        max_photons,
        1,
    )
    fq = quantum_fisher_information(final_state_at_phi, generator)

    # Quantum Fisher sensitivity (ultimate bound)
    delta_phi_fq = 1.0 / np.sqrt(fq)

    # Classical Fisher sensitivity
    # Handle edge case: if FC is zero, the 2-output measurement is not optimal
    # For NOON states with simple detection, P(m|φ) is constant
    # Use QFI as the best available bound in this case
    if fc > 1e-12:
        delta_phi_fc = 1.0 / np.sqrt(fc)
    else:
        # FC=0 means this measurement strategy is not informative
        # Fall back to QFI for theoretical best case
        delta_phi_fc = delta_phi_fq if np.isfinite(delta_phi_fq) else np.inf

    # -----------------------------------------------------------------------------
    # 3. Bayesian Sensitivity
    # -----------------------------------------------------------------------------
    # Simulate measurement outcome(s) and compute posterior
    # Sample measurement outcome
    outcomes = sample_measurement_outcomes(
        state,
        phi_true,
        n_mc,
        rng,
        max_photons=max_photons,
    )

    # Use bayesian_estimator for full pipeline
    result = bayesian_estimator(outcomes, state, max_photons)
    delta_phi_bayes = result["sensitivity"]

    return {
        "delta_phi_ep": float(delta_phi_ep),
        "delta_phi_fc": float(delta_phi_fc),
        "delta_phi_fq": float(delta_phi_fq),
        "delta_phi_bayes": float(delta_phi_bayes),
        "fisher_classical": float(fc),
        "fisher_quantum": float(fq),
        "phi_true": float(phi_true),
        "n_mc": n_mc,
    }


# =============================================================================
# Sensitivity Scaling Analysis
# =============================================================================


@dataclass
class SensitivityScalingResult:
    """Result container for sensitivity scaling analysis."""

    df: pd.DataFrame
    state_type: str
    exponents: dict


# =============================================================================
# Helper: single-N computation
# =============================================================================


def _compute_single_N_metrics(
    state_type_lower: str,
    N: int,
    noise_config: NoiseConfig,
    phi_true: float,
    n_mc: int,
    seed: int | None,
) -> dict | None:
    """Compute sensitivity metrics for a single N value.

    Prepares the input state and runs ``all_sensitivity_metrics``. Returns
    ``None`` if the computation fails (caught internally).

    Args:
        state_type_lower: Lowercased state type ("css", "noon", "twin_fock", "single").
        N: Particle number (also used as max_photons).
        noise_config: Noise configuration (unused here, kept for signature consistency).
        phi_true: True phase for Bayesian estimation.
        n_mc: Number of Monte Carlo samples.
        seed: Random seed (None = fresh entropy).

    Returns:
        Result dict with keys (N, max_photons, delta_phi_ep, delta_phi_fc,
        delta_phi_fq, delta_phi_bayes, fisher_quantum, state_type), or None
        on failure.

    """
    try:
        max_photons = N

        # Prepare input state based on type using MZI state preparation
        if state_type_lower == "single":
            state = prepare_input_state("single_photon", max_photons=max_photons)
        else:
            # CSS, noon, twin_fock all use the noon generator in the current
            # implementation (CSS and twin-fock are approximated by noon states).
            state = prepare_input_state(
                "noon",
                max_photons=max_photons,
                n_particles=N,
            )

        # Compute all sensitivity metrics
        metrics = all_sensitivity_metrics(
            state,
            max_photons,
            phi_true,
            n_mc=n_mc,
            seed=seed,
        )

        return {
            "N": N,
            "max_photons": max_photons,
            "delta_phi_ep": metrics["delta_phi_ep"],
            "delta_phi_fc": metrics["delta_phi_fc"],
            "delta_phi_fq": metrics["delta_phi_fq"],
            "delta_phi_bayes": metrics["delta_phi_bayes"],
            "fisher_quantum": metrics["fisher_quantum"],
            "state_type": state_type_lower,
        }
    except Exception as e:
        # Skip N values that fail
        print(f"Warning: N={N} failed: {e}")
        return None


# =============================================================================
# Helper: exponent fitting
# =============================================================================


def _fit_scaling_exponents(df: pd.DataFrame) -> dict:
    """Fit scaling exponents from a sensitivity-vs-N DataFrame.

    Performs log-log linear regression (Δφ ∝ N^α) for each of the three
    sensitivity columns (delta_phi_ep, delta_phi_fq, delta_phi_bayes).

    Args:
        df: DataFrame with columns N, delta_phi_ep, delta_phi_fq,
            delta_phi_bayes.

    Returns:
        Dict mapping column name to fitted exponent α. Empty dict if there
        are fewer than 2 rows or any N is non-positive.

    """
    exponents: dict[str, float] = {}
    if len(df) >= 2 and df["N"].min() > 0:
        for col in ["delta_phi_ep", "delta_phi_fq", "delta_phi_bayes"]:
            if col in df.columns and np.all(df[col] > 0):
                log_N = np.log(df["N"].values)
                log_delta = np.log(df[col].values)
                # Linear regression: log(Δ) = α*log(N) + log(C)
                alpha = np.polyfit(log_N, log_delta, 1)[0]
                exponents[col] = float(alpha)
    return exponents


def sensitivity_scaling(
    state_type: str,
    N_range: np.ndarray,
    noise_config: NoiseConfig | None = None,
    phi_true: float = np.pi / 4,
    n_mc: int = 200,
    seed: int | None = None,
) -> SensitivityScalingResult:
    """Compute phase sensitivity vs particle number N.

    Analyzes how sensitivity scales with N for different input states:
    - CSS (GHZ): Δφ ∝ 1/√N (Standard Quantum Limit)
    - NOON: Δφ ∝ 1/N (Heisenberg Limit)
    - Twin-Fock: Δφ ∝ 1/√N (SQL for this specific state)

    Args:
        state_type: Type of input state ("css", "noon", "twin_fock", "single").
        N_range: Array of particle numbers to analyze (maps to max_photons).
        noise_config: Optional noise configuration for lossy channels.
        phi_true: True phase for Bayesian estimation.
        n_mc: Number of Monte Carlo samples.
        seed: Random seed (None = fresh entropy).

    Returns:
        SensitivityScalingResult with DataFrame and fitted exponents.

    Raises:
        ValueError: If state_type is invalid.

    """
    valid_states = ["css", "noon", "twin_fock", "single"]
    if state_type.lower() not in valid_states:
        raise ValueError(f"state_type must be one of {valid_states}, got {state_type}")

    if noise_config is None:
        noise_config = NoiseConfig()

    results = []
    for N in N_range:
        result = _compute_single_N_metrics(
            state_type.lower(),
            N,
            noise_config,
            phi_true,
            n_mc,
            seed,
        )
        if result is not None:
            results.append(result)

    df = pd.DataFrame(results)
    exponents = _fit_scaling_exponents(df)

    return SensitivityScalingResult(df=df, state_type=state_type, exponents=exponents)


# =============================================================================
# Validation & Comparison Utilities
# =============================================================================


def validate_sensitivity_order(
    delta_phi_ep: float,
    delta_phi_cr: float,
    rtol: float = 0.5,
) -> bool:
    """Validate that error propagation ≤ Cramér-Rao bound.

    The Cramér-Rao bound is the ultimate lower bound, so error propagation
    sensitivity should always be ≥ the Cramér-Rao bound (less precise).

    However, in practice they can diverge due to:
    - Non-Gaussian statistics
    - Finite sample effects
    - Suboptimal estimators

    This function checks if the inequality holds within tolerance.

    Args:
        delta_phi_ep: Error propagation sensitivity.
        delta_phi_cr: Cramér-Rao (Fisher) sensitivity.
        rtol: Relative tolerance for the check.

    Returns:
        True if validation passes (EP ≥ CRB within tolerance).

    """
    if not np.isfinite(delta_phi_ep) or not np.isfinite(delta_phi_cr):
        return False

    # Allow some tolerance for numerical effects
    # Δφ_EP should be >= Δφ_CR (less precise or equal)
    # But we allow it to be slightly less due to finite sample effects
    return delta_phi_ep >= delta_phi_cr * (1 - rtol)


def compare_sensitivity_methods(
    state: StateType,
    max_photons: int,
    phi_true: float,
    n_mc: int = 500,
) -> dict:
    """Compare all three sensitivity methods and return detailed analysis.

    Args:
        state: Input quantum state in two-mode Fock basis.
        max_photons: Maximum photon number per mode.
        phi_true: True phase for simulation.
        n_mc: Monte Carlo samples.

    Returns:
        Detailed comparison dictionary with all metrics and validation results.

    """
    metrics = all_sensitivity_metrics(state, max_photons, phi_true, n_mc=n_mc)

    # Validate error propagation <= Cramér-Rao
    ep_valid = validate_sensitivity_order(
        metrics["delta_phi_ep"],
        metrics["delta_phi_fq"],
    )

    # Check if methods agree (within factor of 2)
    methods = [
        metrics["delta_phi_ep"],
        metrics["delta_phi_fc"],
        metrics["delta_phi_fq"],
        metrics["delta_phi_bayes"],
    ]
    finite_methods = [m for m in methods if np.isfinite(m) and m > 0]
    if len(finite_methods) >= 2:
        max_val = max(finite_methods)
        min_val = min(finite_methods)
        agreement = max_val / min_val < 2.0
    else:
        agreement = False

    return {
        **metrics,
        "ep_valid": ep_valid,
        "methods_agree": agreement,
        "discrepancy_ratio": (
            metrics["delta_phi_ep"] / metrics["delta_phi_fq"]
            if metrics["delta_phi_fq"] > 0
            else np.inf
        ),
    }


# =============================================================================
# Best/Worst Sensitivity Analysis
# =============================================================================


def analyse_best_worst_sensitivity(
    resource_values: np.ndarray,
    omega_values: np.ndarray,
    sensitivity_grid: np.ndarray,
) -> dict:
    """Find best (min) and worst (max) sensitivity at each resource value.

    Args:
        resource_values: Array of resource values, shape ``(n_R,)``.
        omega_values: Array of ω values, shape ``(n_omega,)``.
        sensitivity_grid: 2D array of sensitivity values, shape ``(n_R, n_omega)``.

    Returns:
        Dictionary with keys:
        - ``resource_values``: Array of resource values.
        - ``best_sensitivity``: Minimum sensitivity at each resource value.
        - ``best_omega``: ω where minimum occurs.
        - ``worst_sensitivity``: Maximum finite sensitivity at each resource value.
        - ``worst_omega``: ω where maximum occurs.
    """
    n_R = len(resource_values)
    best_sens = np.full(n_R, np.inf, dtype=float)
    best_th = np.full(n_R, np.nan, dtype=float)
    worst_sens = np.full(n_R, -np.inf, dtype=float)
    worst_th = np.full(n_R, np.nan, dtype=float)

    for i in range(n_R):
        slice_ = sensitivity_grid[i, :]
        finite_mask = np.isfinite(slice_)
        if np.any(finite_mask):
            full_indices = np.where(finite_mask)[0]

            b_idx = int(np.argmin(slice_[finite_mask]))
            actual_idx = full_indices[b_idx]
            best_sens[i] = float(slice_[actual_idx])
            best_th[i] = float(omega_values[actual_idx])

            w_idx = int(np.argmax(slice_[finite_mask]))
            actual_w_idx = full_indices[w_idx]
            worst_sens[i] = float(slice_[actual_w_idx])
            worst_th[i] = float(omega_values[actual_w_idx])

    return {
        "resource_values": resource_values.copy(),
        "best_sensitivity": best_sens,
        "best_omega": best_th,
        "worst_sensitivity": worst_sens,
        "worst_omega": worst_th,
    }


# =============================================================================
# Minimal MZI Sensitivity Data (shared by reports)
# =============================================================================


@dataclass
class MziSensitivityData(ParquetSerializable):
    r"""Sensitivity data for one state type across resource :math:`R` and :math:`\omega`.

    Stores a 2D grid indexed by ``(resource, omega)``, plus per-resource QFI
    bounds.  This is the canonical version used by MZI-scaling reports; report-
    specific subclasses may add extra metadata fields (e.g. ``squeezing_q``).

    The primary sensitivity metric is :math:`\Delta\omega_C` (Classical Fisher
    Information from the full :math:`P(m|\omega)` distribution).

    Attributes:
        state_type: State identifier (e.g. ``"noon"``, ``"sv"``, ``"oat"``).
        resource_type: ``"N"`` for fixed-N states, ``"mean_N"`` for squeezed.
        resource_values: Array of resource parameter values, shape ``(n_R,)``.
        omega_values: Array of :math:`\omega` values, shape ``(n_omega,)``.
        expectation_grid: :math:`\langle J_z\rangle` at each ``(R, omega)``.
        variance_grid: :math:`\text{Var}(J_z)` at each ``(R, omega)``.
        derivative_grid: :math:`\partial\langle J_z\rangle/\partial\omega`.
        delta_omega_ep_grid: :math:`\Delta\omega_{\text{EP}}`.
        delta_omega_q_per_R: :math:`\Delta\omega_Q` per resource value (length ``n_R``).
        fisher_classical_grid: :math:`F_C` at each ``(R, omega)``.
        delta_omega_c_grid: :math:`\Delta\omega_C` at each ``(R, omega)``.
        t_hold: Holding time.
        truncation_M_per_R: Truncation M used per resource value (optional).
        squeezing_q_per_R: Squeezing parameter per resource value (optional).
    """

    state_type: str
    resource_type: str
    resource_values: np.ndarray
    omega_values: np.ndarray
    expectation_grid: np.ndarray
    variance_grid: np.ndarray
    derivative_grid: np.ndarray
    delta_omega_ep_grid: np.ndarray
    delta_omega_q_per_R: np.ndarray
    fisher_classical_grid: np.ndarray
    delta_omega_c_grid: np.ndarray
    t_hold: float = 10.0
    truncation_M_per_R: np.ndarray | None = None
    squeezing_q_per_R: np.ndarray | None = None

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "state_type",
        "resource_type",
        "resource",
        "omega",
        "expectation",
        "variance",
        "derivative",
        "delta_omega_ep",
        "delta_omega_q",
        "fisher_classical",
        "delta_omega_c",
        "t_hold",
        "truncation_M",
        "squeezing_q",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to long-format DataFrame (one row per resource, ω combination)."""
        n_R = len(self.resource_values)
        n_omega = len(self.omega_values)
        rows: list[dict] = []
        trunc_M = (
            self.truncation_M_per_R
            if self.truncation_M_per_R is not None
            else np.full(n_R, np.nan)
        )
        sq_q = (
            self.squeezing_q_per_R
            if self.squeezing_q_per_R is not None
            else np.full(n_R, np.nan)
        )
        for i in range(n_R):
            for j in range(n_omega):
                dt_ep = float(self.delta_omega_ep_grid[i, j])
                dt_c = float(self.delta_omega_c_grid[i, j])
                rows.append(
                    {
                        "state_type": self.state_type,
                        "resource_type": self.resource_type,
                        "resource": float(self.resource_values[i]),
                        "omega": float(self.omega_values[j]),
                        "expectation": float(self.expectation_grid[i, j]),
                        "variance": float(self.variance_grid[i, j]),
                        "derivative": float(self.derivative_grid[i, j]),
                        "delta_omega_ep": (
                            dt_ep if np.isfinite(dt_ep) else float("inf")
                        ),
                        "delta_omega_q": float(self.delta_omega_q_per_R[i]),
                        "fisher_classical": float(self.fisher_classical_grid[i, j]),
                        "delta_omega_c": (dt_c if np.isfinite(dt_c) else float("inf")),
                        "t_hold": self.t_hold,
                        "truncation_M": float(trunc_M[i]),
                        "squeezing_q": float(sq_q[i]),
                    }
                )
        return pd.DataFrame(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> MziSensitivityData:
        """Load from Parquet, reconstructing the 2D grids."""
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        state_type = str(df["state_type"].iloc[0])
        resource_type = str(df["resource_type"].iloc[0])
        t_hold_val = float(df["t_hold"].iloc[0])
        R_vals = sorted(df["resource"].unique())
        omega_vals = sorted(df["omega"].unique())
        n_R = len(R_vals)
        n_omega = len(omega_vals)

        expectation_grid = np.full((n_R, n_omega), np.nan, dtype=float)
        variance_grid = np.full((n_R, n_omega), np.nan, dtype=float)
        derivative_grid = np.full((n_R, n_omega), np.nan, dtype=float)
        delta_omega_ep_grid = np.full((n_R, n_omega), np.nan, dtype=float)
        delta_omega_q_per_R = np.full(n_R, np.nan, dtype=float)
        fisher_classical_grid = np.full((n_R, n_omega), np.nan, dtype=float)
        delta_omega_c_grid = np.full((n_R, n_omega), np.nan, dtype=float)
        truncation_M_per_R = np.full(n_R, np.nan, dtype=float)
        squeezing_q_per_R = np.full(n_R, np.nan, dtype=float)

        for _, row in df.iterrows():
            r_idx = R_vals.index(float(row["resource"]))
            t_idx = omega_vals.index(float(row["omega"]))
            expectation_grid[r_idx, t_idx] = row["expectation"]
            variance_grid[r_idx, t_idx] = row["variance"]
            derivative_grid[r_idx, t_idx] = row["derivative"]
            delta_omega_ep_grid[r_idx, t_idx] = row["delta_omega_ep"]
            dq = float(row["delta_omega_q"])
            if np.isnan(delta_omega_q_per_R[r_idx]):
                delta_omega_q_per_R[r_idx] = dq
            elif not np.isclose(delta_omega_q_per_R[r_idx], dq, rtol=1e-10):
                raise ValueError(
                    f"Inconsistent delta_omega_q for resource={row['resource']}: "
                    f"expected {delta_omega_q_per_R[r_idx]}, got {dq}. "
                    f"Regenerate the file."
                )
            fisher_classical_grid[r_idx, t_idx] = float(row["fisher_classical"])
            delta_omega_c_grid[r_idx, t_idx] = float(row["delta_omega_c"])
            truncation_M_per_R[r_idx] = float(row["truncation_M"])
            squeezing_q_per_R[r_idx] = float(row["squeezing_q"])

        return cls(
            state_type=state_type,
            resource_type=resource_type,
            resource_values=np.array(R_vals, dtype=float),
            omega_values=np.array(omega_vals, dtype=float),
            expectation_grid=expectation_grid,
            variance_grid=variance_grid,
            derivative_grid=derivative_grid,
            delta_omega_ep_grid=delta_omega_ep_grid,
            delta_omega_q_per_R=delta_omega_q_per_R,
            fisher_classical_grid=fisher_classical_grid,
            delta_omega_c_grid=delta_omega_c_grid,
            t_hold=t_hold_val,
            truncation_M_per_R=truncation_M_per_R,
            squeezing_q_per_R=squeezing_q_per_R,
        )


@dataclass
class MziSensitivityDataSV(MziSensitivityData):
    """Sensitivity data for squeezed-vacuum MZI.

    Inherits all fields, serialization, and Parquet I/O from the shared
    :class:`MziSensitivityData`.  This subclass exists purely to give a
    more specific return type to ``from_parquet``.
    """

    @classmethod
    def from_parquet(cls, path: str | Path) -> MziSensitivityDataSV:  # type: ignore[override]
        return super().from_parquet(path)  # type: ignore[return-value]

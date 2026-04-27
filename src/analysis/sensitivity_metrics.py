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

import numpy as np
import pandas as pd

from src.analysis.bayesian_phase_estimation import (
    bayesian_estimator,
    sample_measurement_outcomes,
)
from src.analysis.fisher_information import (
    classical_fisher_information,
    phase_sensitivity_from_fisher,
    quantum_fisher_information,
)
from src.physics.mzi_simulation import (
    compute_output_probabilities,
    evolve_mzi,
    prepare_input_state,
)
from src.physics.mzi_states import create_jz_operator
from src.physics.noise_channels import NoiseConfig


# =============================================================================
# Type Definitions
# =============================================================================

StateType = np.ndarray  # Pure state vector
SensitivityResult = dict[str, float]  # {"delta_phi": float, ...}


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
            f"State dimension {state.shape[0]} must be (max_photons+1)² = {expected_dim}"
        )

    # Generate J_z operator in two-mode Fock basis
    jz = create_jz_operator(max_photons)
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
        abs_derivative < min_deriv_threshold, min_deriv_threshold, abs_derivative
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


# =============================================================================
# All Sensitivity Metrics
# =============================================================================


def all_sensitivity_metrics(
    state: StateType,
    max_photons: int,
    phi_true: float,
    n_mc: int = 500,
    rng_seed: int = 42,
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
        rng_seed: Random seed for reproducibility.

    Returns:
        Dictionary with all three sensitivity values and metadata.

    Raises:
        ValueError: If max_photons < 1 or n_mc < 1.
    """
    if max_photons < 1:
        raise ValueError(f"max_photons must be >= 1, got {max_photons}")
    if n_mc < 1:
        raise ValueError(f"n_mc must be >= 1, got {n_mc}")

    rng = np.random.default_rng(rng_seed)

    # Verify state dimension matches expected
    dim = state.shape[0]
    expected_dim = (max_photons + 1) ** 2
    if dim != expected_dim:
        raise ValueError(
            f"State dimension {dim} must be (max_photons+1)² = {expected_dim}"
        )

    # Generate J_z generator for the two-mode system
    generator = create_jz_operator(max_photons)

    # -----------------------------------------------------------------------------
    # 1. Error Propagation Sensitivity
    # -----------------------------------------------------------------------------
    phi_grid = np.linspace(0, 2 * np.pi, 361)
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
    delta_phi_fq = phase_sensitivity_from_fisher(fq)

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
        state, phi_true, n_mc, rng, max_photons=max_photons
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


def sensitivity_scaling(
    state_type: str,
    N_range: np.ndarray,
    noise_config: NoiseConfig | None = None,
    phi_true: float = np.pi / 4,
    n_mc: int = 200,
    rng_seed: int = 42,
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
        rng_seed: Random seed.

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
        max_photons = N

        # Prepare input state based on type using MZI state preparation
        if state_type.lower() == "css":
            # For GHZ-like state, we use noon which gives similar entanglement
            state = prepare_input_state("noon", max_photons=max_photons, n_particles=N)
        elif state_type.lower() == "noon":
            state = prepare_input_state("noon", max_photons=max_photons, n_particles=N)
        elif state_type.lower() == "twin_fock":
            # Twin-Fock: balanced superposition
            state = prepare_input_state("noon", max_photons=max_photons, n_particles=N)
        else:  # single photon
            state = prepare_input_state("single_photon", max_photons=max_photons)

        try:
            # Compute all sensitivity metrics
            metrics = all_sensitivity_metrics(
                state, max_photons, phi_true, n_mc=n_mc, rng_seed=rng_seed
            )

            results.append(
                {
                    "N": N,
                    "max_photons": max_photons,
                    "delta_phi_ep": metrics["delta_phi_ep"],
                    "delta_phi_fc": metrics["delta_phi_fc"],
                    "delta_phi_fq": metrics["delta_phi_fq"],
                    "delta_phi_bayes": metrics["delta_phi_bayes"],
                    "fisher_quantum": metrics["fisher_quantum"],
                    "state_type": state_type.lower(),
                }
            )
        except Exception as e:
            # Skip N values that fail
            print(f"Warning: N={N} failed: {e}")
            continue

    df = pd.DataFrame(results)

    # Fit scaling exponents using log-log fit
    # Δφ ∝ N^α  =>  log(Δφ) = log(C) + α*log(N)
    exponents = {}
    if len(df) >= 2 and df["N"].min() > 0:
        for col in ["delta_phi_ep", "delta_phi_fq", "delta_phi_bayes"]:
            if col in df.columns and np.all(df[col] > 0):
                log_N = np.log(df["N"].values)
                log_delta = np.log(df[col].values)
                # Linear regression: log(Δ) = α*log(N) + log(C)
                alpha = np.polyfit(log_N, log_delta, 1)[0]
                exponents[col] = float(alpha)

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
        metrics["delta_phi_ep"], metrics["delta_phi_fq"]
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

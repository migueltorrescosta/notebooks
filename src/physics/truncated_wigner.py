"""
Truncated Wigner Approximation (TWA) for Bose-Einstein Condensate Interferometry.

This module implements a semi-classical phase-space method for efficient
simulation of large-N BEC interferometers without exponential Hilbert
space growth.

Physical Model:
- SU(2) Wigner function on Bloch sphere
- Stochastic differential equations for Bloch vector (x, y, z) = (⟨J_x⟩/J, ⟨J_y⟩/J, ⟨J_z⟩/J)
- Quantum jumps modeled as noise terms
- Average over n_traj trajectories

The TWA provides O(1) complexity scaling vs O(N) for full quantum simulations,
enabling practical simulation of large atom numbers.

Conventions:
- Phase convention: standard quantum mechanics
- Units: dimensionless throughout (ℏ = 1)
- Time in dimensionless units
- The `phi` parameter in `sample_wigner_sphere` allows specifying
  the initial phase angle for CSS states (default 0.0)

Reference:
    Law et al. "Collisional dynamics of tunable Bose-Einstein condensates"
    Physics Review Letters 87, 110403 (2001)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TWAConfig:
    """Configuration for Truncated Wigner Approximation simulation.

    Attributes:
        N: Total atom number.
        chi: One-axis twisting (OAT) squeezing strength.
        gamma_1: One-body loss rate.
        gamma_2: Two-body loss rate.
        gamma_phi: Phase diffusion rate.

    """

    N: int
    chi: float = 0.0  # OAT squeezing strength
    gamma_1: float = 0.0  # one-body loss rate
    gamma_2: float = 0.0  # two-body loss rate
    gamma_phi: float = 0.0  # phase diffusion rate


# =============================================================================
# Wigner Function Sampling
# =============================================================================


def sample_wigner_sphere(
    N: int,
    state_type: str,
    rng: np.random.Generator,
    phi_Bloch: float = 0.0,
) -> np.ndarray:
    """Sample initial Bloch vector from Wigner function.

    Samples a point on the Bloch sphere from the Wigner distribution
    corresponding to the specified quantum state.

    The Wigner function for SU(2) coherent states is a delta function on the
    sphere surface, while for number states it has a broader distribution.

    Args:
        N: Total atom number (determines spin J = N/2).
        state_type: State type ('CSS', 'SSS', 'NOON').
        rng: Random number generator.
        phi_Bloch: Phase angle for CSS (default 0.0).

    Returns:
        Bloch vector of shape (3,) with components (x, y, z).

    Raises:
        ValueError: If state_type is not supported.

    """
    if state_type == "CSS":
        # Coherent Spin State (CSS) - coherent state at angle phi
        # Wigner distribution: delta function at specified angle on sphere
        # For coherent state |α⟩ with α = sqrt(N/2) * e^{i*phi}:
        #   r = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)) with θ = π/2 for equator
        # Standard CSS at φ has Bloch vector on equator
        J = N / 2.0
        sigma = 1.0 / np.sqrt(2 * J)

        # Add small noise based on quantum uncertainty
        x = np.cos(phi_Bloch) + rng.normal(0, sigma)
        y = np.sin(phi_Bloch) + rng.normal(0, sigma)
        z = rng.normal(0, sigma)

        # Normalize to sphere surface
        norm = np.sqrt(x**2 + y**2 + z**2)
        if norm > 1e-10:
            x, y, z = x / norm, y / norm, z / norm
        else:
            x, y, z = np.cos(phi_Bloch), np.sin(phi_Bloch), 0.0

        return np.array([x, y, z])

    if state_type == "SSS":
        # Squeezed Spin State (SSS) - Gaussian narrowed in one direction
        # Wigner distribution: Gaussian with different widths
        # Standard deviation varies with direction
        J = N / 2.0

        # Standard deviation for CSS: σ = 1/√(2J) ≈ 1/√N for large N
        sigma = 1.0 / np.sqrt(2 * J)

        # Squeeze in x-direction (reduce fluctuations)
        squeeze_factor = 0.5  # Standard value
        sigma_x = sigma * squeeze_factor
        sigma_z = sigma / squeeze_factor

        # Sample from Gaussian in x and z
        x = rng.normal(0, sigma_x)
        z = rng.normal(0, sigma_z)
        y = 0.0  # No y component for standard states

        # Normalize to sphere surface (Bloch vector length = 1)
        norm = np.sqrt(x**2 + y**2 + z**2)
        if norm > 1e-10:
            # Project onto sphere surface
            x = x / norm
            y = y / norm
            z = z / norm

        return np.array([x, y, z])

    if state_type == "NOON":
        # NOON state - bimodal distribution (anti-diagonal in Dicke basis)
        # Wigner distribution: two peaks separated on sphere
        # WARNING: TWA is not reliable for NOON states
        J = N / 2.0
        sigma = 1.0 / np.sqrt(2 * J)

        # Two peaks at z = ±1 (all atoms in mode a or mode b)
        if rng.random() < 0.5:
            # Peak at +z
            x = rng.normal(0, sigma * 0.2)
            y = rng.normal(0, sigma * 0.2)
            z = rng.normal(1.0, sigma * 0.2)
        else:
            # Peak at -z
            x = rng.normal(0, sigma * 0.2)
            y = rng.normal(0, sigma * 0.2)
            z = rng.normal(-1.0, sigma * 0.2)

        # Normalize to sphere
        norm = np.sqrt(x**2 + y**2 + z**2)
        if norm > 1e-10:
            x = x / norm
            y = y / norm
            z = z / norm

        return np.array([x, y, z])

    raise ValueError(
        f"Unknown state type: {state_type}. Supported: 'CSS', 'SSS', 'NOON'",
    )


# =============================================================================
# Stochastic Differential Equations
# =============================================================================


def _euler_maruyama_step(
    J_vec: np.ndarray,
    dt: float,
    J: float,
    chi: float,
    gamma_1: float,
    gamma_2: float,
    gamma_phi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Single Euler-Maruyama integration step for the TWA Bloch vector SDEs.

    Computes the deterministic drift (nonlinear OAT Hamiltonian) and applies
    stochastic noise from one-body loss, two-body loss, and phase diffusion.
    Normalises the result back to the Bloch sphere surface.

    Args:
        J_vec: Current Bloch vector of shape (3,) with components (x, y, z).
        dt: Timestep for integration.
        J: Total spin J = N/2.
        chi: OAT squeezing strength.
        gamma_1: One-body loss rate.
        gamma_2: Two-body loss rate.
        gamma_phi: Phase diffusion rate.
        rng: Random number generator.

    Returns:
        Updated Bloch vector of shape (3,).

    """
    x, y, z = J_vec

    # Deterministic drift (unitary evolution)
    dx_deterministic = chi * (y * z - y)
    dy_deterministic = -chi * x * z
    dz_deterministic = 0.0

    # Add deterministic update
    x_new = x + dt * dx_deterministic
    y_new = y + dt * dy_deterministic
    z_new = z + dt * dz_deterministic

    # One-body loss noise (quantum jumps)
    if gamma_1 > 0:
        mean_n = J * (1 + z)
        if mean_n > 0:
            dW1 = rng.normal(0, np.sqrt(dt))
            noise_1 = np.sqrt(gamma_1 * max(mean_n, 0) / J) * dW1
            z_new -= noise_1 * np.sqrt(dt)

    # Two-body loss noise
    if gamma_2 > 0:
        mean_n = J * (1 + z)
        if mean_n > 0:
            dW2 = rng.normal(0, np.sqrt(dt))
            noise_2 = np.sqrt(gamma_2 * max(mean_n, 0) ** 2 / J) * dW2
            z_new -= noise_2 * np.sqrt(dt)

    # Phase diffusion noise
    if gamma_phi > 0:
        dW_phi = rng.normal(0, np.sqrt(dt))
        noise_phi = np.sqrt(gamma_phi) * dW_phi
        y_new += noise_phi * np.sqrt(dt)

    # Normalize to Bloch sphere surface
    norm = np.sqrt(x_new**2 + y_new**2 + z_new**2)
    if norm > 1e-10:
        x_new = x_new / norm
        y_new = y_new / norm
        z_new = z_new / norm
    else:
        x_new, y_new, z_new = 0.0, 0.0, 1.0

    return np.array([x_new, y_new, z_new])


def wigner_sde_trajectory(
    J_init: np.ndarray,
    params: dict,
    T_evo: float,
    dt: float,
    rng: np.random.Generator,
    store_trajectory: bool = False,
) -> dict:
    """Propagate single trajectory via stochastic differential equations.

    Uses Euler-Maruyama method to integrate the SDEs for the
    Bloch vector components. Includes:
    - Unitary evolution (nonlinear)
    - One-body loss (quantum jumps)
    - Two-body loss
    - Phase diffusion

    The SDEs for the Bloch vector (x, y, z) = (⟨J_x⟩/J, ⟨J_y⟩/J, ⟨J_z⟩/J) are:

    dx/dt = χ*(y*z - y) + noise terms
    dy/dt = -χ*x*z + noise terms
    dz/dt = noise terms

    For the TWA, we use the truncation that replaces quantum
    operators with c-numbers: J_i → J * r_i where r = (x, y, z).

    Args:
        J_init: Initial Bloch vector of shape (3,).
        params: Dictionary with keys:
            - N: atom number
            - chi: OAT squeezing strength
            - gamma_1: one-body loss rate
            - gamma_2: two-body loss rate
            - gamma_phi: phase diffusion rate
        T_evo: Total evolution time.
        dt: Timestep for integration.
        rng: Random number generator.
        store_trajectory: Whether to store full trajectory.

    Returns:
        Dictionary with:
            - J_final: Final Bloch vector
            - trajectory: Full trajectory if store_trajectory=True

    """
    N = params.get("N", 10)
    chi = params.get("chi", 0.0)
    gamma_1 = params.get("gamma_1", 0.0)
    gamma_2 = params.get("gamma_2", 0.0)
    gamma_phi = params.get("gamma_phi", 0.0)

    J = N / 2.0  # Total spin

    # Number of steps
    num_steps = max(1, int(np.ceil(T_evo / dt)))
    dt = T_evo / num_steps  # Adjust to exactly hit T_evo

    # Initialize
    J_vec = J_init.copy()

    # Store trajectory if requested
    if store_trajectory:
        traj = np.zeros((num_steps + 1, 3))
        traj[0] = J_vec
    else:
        traj = None

    # SDE integration via Euler-Maruyama
    for step in range(num_steps):
        J_vec = _euler_maruyama_step(
            J_vec, dt, J, chi, gamma_1, gamma_2, gamma_phi, rng,
        )

        # Store if requested
        if store_trajectory and traj is not None:
            traj[step + 1] = J_vec

    return {"J_final": J_vec, "trajectory": traj}


# =============================================================================
# TWA Expectation Values
# =============================================================================


def compute_twa_expectations(
    N: int,
    state_type: str,
    params: dict,
    T_evo: float,
    n_traj: int = 5000,
    seed: int | None = None,
    dt: float = 0.01,
    store_trajectories: bool = False,
) -> dict:
    """Compute ⟨J_z⟩ and Var(J_z) via Truncated Wigner Approximation.

    Runs n_traj stochastic trajectories and averages to compute
    expectation values. This is the key method that enables
    efficient simulation of large-N systems.

    Args:
        N: Total atom number.
        state_type: Initial state type ('CSS', 'SSS', 'NOON').
        params: Dictionary with keys:
            - chi: OAT squeezing strength
            - gamma_1: one-body loss rate
            - gamma_2: two-body loss rate
            - gamma_phi: phase diffusion rate
        T_evo: Evolution time.
        n_traj: Number of trajectories to average.
        seed: Random seed for reproducibility (None = fresh entropy).
        dt: Timestep for SDE integration.
        store_trajectories: Whether to store all trajectories.

    Returns:
        Dictionary with:
            - Jz_mean: Mean ⟨J_z⟩ (scaled to [-N/2, N/2])
            - Jz_variance: Variance Var(J_z)
            - Jz_std: Standard deviation ΔJ_z
            - J_total_mean: Mean total spin |⟨J⟩|
            - trajectories: All trajectories if store_trajectories=True

    """
    # Warning for NOON states
    if state_type == "NOON":
        import warnings

        warnings.warn(
            "NOON state simulation via TWA is not reliable. "
            "Consider using full quantum simulation instead.",
            UserWarning,
            stacklevel=2,
        )

    # Initialize random generator
    rng = np.random.default_rng(seed)

    # Build params with N
    full_params = {"N": N, **params}

    # Sample initial conditions and propagate trajectories
    Jz_samples = []
    J_total_samples = []
    all_trajectories = []

    for _traj_idx in range(n_traj):
        # Sample initial Bloch vector
        J_init = sample_wigner_sphere(N, state_type, rng)

        # Propagate trajectory
        result = wigner_sde_trajectory(J_init, full_params, T_evo, dt, rng)

        J_final = result["J_final"]
        x, y, z = J_final

        # Scale: J_z = J * z where J = N/2
        J = N / 2.0
        Jz_samples.append(J * z)

        # Total spin magnitude
        J_total = np.sqrt(x**2 + y**2 + z**2) * J
        J_total_samples.append(J_total)

        if store_trajectories and result["trajectory"] is not None:
            all_trajectories.append(result["trajectory"])

    # Convert to arrays
    Jz_samples_arr = np.array(Jz_samples)
    J_total_samples_arr = np.array(J_total_samples)

    # Compute statistics
    Jz_mean = np.mean(Jz_samples_arr)
    Jz_variance = np.var(Jz_samples_arr)
    Jz_std = np.std(Jz_samples_arr)
    J_total_mean = np.mean(J_total_samples_arr)

    # Return results
    result_dict = {
        "Jz_mean": Jz_mean,
        "Jz_variance": Jz_variance,
        "Jz_std": Jz_std,
        "J_total_mean": J_total_mean,
    }

    if store_trajectories:
        result_dict["trajectories"] = np.array(all_trajectories)  # type: ignore[assignment]

    return result_dict


# =============================================================================
# Phase Estimation Sensitivity
# =============================================================================


def compute_phase_sensitivity(
    N: int,
    state_type: str,
    params: dict,
    T_evo: float,
    n_traj: int = 5000,
    seed: int | None = None,
) -> dict:
    """Compute phase estimation sensitivity via TWA.

    Estimates the phase uncertainty Δφ from the variance of J_z.

    For interferometry, the phase sensitivity scales as:
    - Standard quantum limit: Δφ_SQL = 1/√N
    - Heisenberg limit: Δφ_HL = 1/N (with entanglement)

    Args:
        N: Total atom number.
        state_type: Initial state type.
        params: Evolution parameters.
        T_evo: Evolution time.
        n_traj: Number of trajectories.
        seed: Random seed (None = fresh entropy).

    Returns:
        Dictionary with phase sensitivity metrics.

    """
    result = compute_twa_expectations(
        N=N,
        state_type=state_type,
        params=params,
        T_evo=T_evo,
        n_traj=n_traj,
        seed=seed,
    )

    J = N / 2.0
    Jz_variance = result["Jz_variance"]

    # Phase sensitivity: Δφ = ΔJz / |∂⟨Jz⟩/∂φ|
    # For linear phase accumulation: Δφ = ΔJz / (d⟨Jz⟩/dt * T)
    # Simplified: use variance directly

    # If Jz variance is small, phase sensitivity is high
    delta_phi = np.sqrt(Jz_variance) / J if Jz_variance > 0 else np.inf

    # SQL and HL for comparison
    delta_phi_sql = 1.0 / np.sqrt(N)
    delta_phi_hl = 1.0 / N

    return {
        "delta_phi": delta_phi,
        "delta_phi_sql": delta_phi_sql,
        "delta_phi_hl": delta_phi_hl,
        "Jz_variance": Jz_variance,
        "Jz_mean": result["Jz_mean"],
    }


# =============================================================================
# Validation
# =============================================================================


def validate_bloch_vector(J: np.ndarray, tol: float = 1e-6) -> dict:
    """Validate Bloch vector properties.

    Args:
        J: Bloch vector of shape (3,).
        tolerance: Numerical tolerance.

    Returns:
        Dictionary with validation results.

    """
    x, y, z = J
    norm = np.sqrt(x**2 + y**2 + z**2)

    return {
        "norm": norm,
        "is_normalized": np.isclose(norm, 1.0, atol=tol),
        "x": x,
        "y": y,
        "z": z,
    }


# =============================================================================
# Convenience Functions
# =============================================================================


def run_twa_simulation(
    N: int,
    state_type: str = "CSS",
    chi: float = 0.0,
    gamma_1: float = 0.0,
    gamma_2: float = 0.0,
    gamma_phi: float = 0.0,
    T_evo: float = 1.0,
    n_traj: int = 5000,
    seed: int | None = None,
) -> dict:
    """Run complete TWA simulation.

    Convenience function that performs all steps.

    Args:
        N: Total atom number.
        state_type: Initial state type ('CSS', 'SSS', 'NOON').
        chi: OAT squeezing strength.
        gamma_1: One-body loss rate.
        gamma_2: Two-body loss rate.
        gamma_phi: Phase diffusion rate.
        T_evo: Evolution time.
        n_traj: Number of trajectories.
        seed: Random seed (None = fresh entropy).

    Returns:
        Dictionary with simulation results.

    """
    params = {
        "chi": chi,
        "gamma_1": gamma_1,
        "gamma_2": gamma_2,
        "gamma_phi": gamma_phi,
    }

    # Compute expectations
    exp_result = compute_twa_expectations(
        N=N,
        state_type=state_type,
        params=params,
        T_evo=T_evo,
        n_traj=n_traj,
        seed=seed,
    )

    # Compute phase sensitivity (use different stream for independence)
    sens_seed = (seed + 1) if seed is not None else None
    sens_result = compute_phase_sensitivity(
        N=N,
        state_type=state_type,
        params=params,
        T_evo=T_evo,
        n_traj=n_traj,
        seed=sens_seed,
    )

    return {
        "N": N,
        "state_type": state_type,
        "params": params,
        "T_evo": T_evo,
        "n_traj": n_traj,
        "Jz_mean": exp_result["Jz_mean"],
        "Jz_variance": exp_result["Jz_variance"],
        "Jz_std": exp_result["Jz_std"],
        "J_total_mean": exp_result["J_total_mean"],
        "delta_phi": sens_result["delta_phi"],
        "delta_phi_sql": sens_result["delta_phi_sql"],
        "delta_phi_hl": sens_result["delta_phi_hl"],
    }


# =============================================================================
# Validation vs Lindblad
# =============================================================================


def compare_with_lindblad(
    N: int,
    state_type: str,
    params: dict,
    T_evo: float,
    n_traj: int = 5000,
    seed: int | None = None,
    tolerance: float = 0.05,
) -> dict:
    """Compare TWA results with full quantum (Lindblad) simulation.

    Validates TWA approximation by comparing with exact quantum
    mechanical results from Lindblad master equation.

    Args:
        N: Total atom number (small N <= 20 for tractability).
        state_type: Initial state type.
        params: Dissipation parameters.
        T_evo: Evolution time.
        n_traj: Number of TWA trajectories.
        seed: Random seed (None = fresh entropy).
        tolerance: Acceptable relative difference (default 5%).

    Returns:
        Dictionary with comparison results.

    """
    try:
        from src.evolution.lindblad_solver import (
            LindbladConfig,
            create_coherent_state,
            evolve_lindblad,
        )
        from src.physics.dicke_basis import jz_operator
        from src.utils.enums import OperatorBasis

        # Limit N for computational feasibility
        max_N = 20
        N = min(N, max_N)
    except ImportError:
        return {
            "error": "Lindblad module not available",
            "twa_Jz_mean": None,
            "lindblad_Jz_mean": None,
            "relative_error": None,
        }

    # Get TWA result
    twa_result = compute_twa_expectations(
        N=N,
        state_type=state_type,
        params=params,
        T_evo=T_evo,
        n_traj=n_traj,
        seed=seed,
    )

    # Get Lindblad result (full quantum)
    # Create initial state
    chi = params.get("chi", 0.0)
    gamma_1 = params.get("gamma_1", 0.0)
    gamma_2 = params.get("gamma_2", 0.0)
    gamma_phi = params.get("gamma_phi", 0.0)

    config = LindbladConfig(
        N=N,
        chi=chi,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_phi=gamma_phi,
    )

    # Initial coherent state in Fock basis
    # For CSS, use coherent state with proper phase
    alpha = np.sqrt(N / 2.0)  # Approximately N/2 atoms
    psi0 = create_coherent_state(alpha + 0j, N, truncation=3)
    rho0 = np.outer(psi0, psi0.conj())

    # Evolve
    dt = 0.01
    if T_evo > 0:
        rho_final = evolve_lindblad(rho0, config, T_evo, dt)

        # Compute Jz expectation
        J_z = jz_operator(N, basis=OperatorBasis.FOCK)
        lindblad_Jz_mean = np.real(np.trace(rho_final @ J_z))
    else:
        # At t=0, compute initial expectation
        J_z = jz_operator(N, basis=OperatorBasis.FOCK)
        lindblad_Jz_mean = np.real(np.trace(rho0 @ J_z))

    # Compare
    twa_Jz_mean = twa_result["Jz_mean"]

    if np.abs(lindblad_Jz_mean) > 1e-6:
        relative_error = np.abs(twa_Jz_mean - lindblad_Jz_mean) / np.abs(
            lindblad_Jz_mean,
        )
    else:
        # Use absolute error if Lindblad is near zero
        relative_error = np.abs(twa_Jz_mean - lindblad_Jz_mean)

    return {
        "N": N,
        "state_type": state_type,
        "twa_Jz_mean": twa_Jz_mean,
        "lindblad_Jz_mean": lindblad_Jz_mean,
        "relative_error": relative_error,
        "within_tolerance": relative_error < tolerance,
    }

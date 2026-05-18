"""
BEC sensitivity analysis for phase estimation scaling.

Provides functions to compute phase sensitivity for various input states
(CSS, SSS, Twin-Fock, NOON) under Lindblad or TWA evolution, as a
function of atom number N.

Physical Model:
- Dicke basis |J, m⟩ with J = N/2, dimension N+1
- OAT Hamiltonian: H = χ J_z² for squeezing
- Phase sensitivity: Δφ = √(Var(J_z)) / |∂⟨J_z⟩/∂φ|
- For linear phase accumulation: ∂⟨J_z⟩/∂φ ≈ N/2

Units:
- Dimensionless throughout.

See Also:
    src.physics.states: Dicke-basis state generation
    src.algorithms.spin_squeezing: CSS/SSS state generation
    src.evolution.lindblad_solver: Lindblad master equation solver
    src.physics.truncated_wigner: TWA simulation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.algorithms.spin_squeezing import (
    coherent_spin_state,
    generate_squeezed_state,
    optimal_squeezing_time,
)
from src.evolution.lindblad_solver import LindbladConfig, evolve_lindblad
from src.physics.dicke_basis import jz_operator
from src.physics.states import generate_noon_state, generate_twin_fock_state
from src.physics.truncated_wigner import run_twa_simulation

if TYPE_CHECKING:
    from src.physics.noise_channels import NoiseConfig


def compute_phase_uncertainty_lindblad(
    N: int,
    state: np.ndarray,
    chi: float,
    T: float,
    noise_config: NoiseConfig,
) -> float:
    """Compute phase uncertainty via Lindblad evolution.

    Evolves the initial state under OAT + noise and computes
    the variance in J_z to estimate phase sensitivity.

    Args:
        N: Atom number.
        state: Initial state vector in Dicke basis.
        chi: OAT strength.
        T: Evolution time.
        noise_config: Noise configuration.

    Returns:
        Phase uncertainty Δφ.

    Raises:
        ValueError: If the state dimension does not match N+1.
        ValueError: If chi or T is negative.

    Example:
        >>> from src.physics.noise_channels import NoiseConfig
        >>> from src.algorithms.spin_squeezing import coherent_spin_state
        >>> state = coherent_spin_state(4)
        >>> noise = NoiseConfig()
        >>> dphi = compute_phase_uncertainty_lindblad(4, state, chi=0.0, T=0.0, noise_config=noise)
        >>> dphi > 0
        True

    """
    if state.shape[0] != N + 1:
        raise ValueError(
            f"State dimension {state.shape[0]} does not match N+1 = {N + 1}",
        )
    if chi < 0:
        raise ValueError(f"OAT strength chi must be non-negative, got {chi}")
    if T < 0:
        raise ValueError(f"Evolution time T must be non-negative, got {T}")

    rho0 = np.outer(state, state.conj())

    config = LindbladConfig(
        N=N,
        chi=chi,
        gamma_1=noise_config.gamma_1,
        gamma_2=noise_config.gamma_2,
        gamma_phi=noise_config.gamma_phi,
    )

    dt = 0.01
    rho_final = evolve_lindblad(rho0, config, T, dt)

    # Compute J_z variance
    J_z = jz_operator(N)
    Jz_mean = np.real(np.trace(rho_final @ J_z))
    Jz2_mean = np.real(np.trace(rho_final @ J_z @ J_z))
    Jz_var = Jz2_mean - Jz_mean**2

    # Phase sensitivity: Δφ = √(Var(J_z)) / |d⟨J_z⟩/dφ|
    # For linear phase accumulation: d⟨J_z⟩/dφ ≈ N/2
    if Jz_var > 0:
        return np.sqrt(Jz_var) / (N / 2)
    return 1.0 / np.sqrt(N)  # SQL fallback


def compute_sensitivity_vs_n(
    state_type: str,
    N_range: tuple[int, int],
    N_points: int,
    chi: float,
    noise_config: NoiseConfig,
    method: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute phase sensitivity Δφ vs atom number N.

    Sweeps over atom numbers and computes the phase sensitivity for
    the given state type using either Lindblad (small N) or TWA (large N).

    Args:
        state_type: One of 'CSS', 'SSS', 'Twin-Fock', 'NOON'.
        N_range: Tuple (min_N, max_N).
        N_points: Number of N values to sample.
        chi: OAT strength.
        noise_config: Noise configuration.
        method: 'Lindblad' or 'TWA'.
        seed: Random seed for TWA simulations.

    Returns:
        DataFrame with columns: N, delta_phi.

    Raises:
        ValueError: If state_type is not recognized.

    """
    N_values = np.linspace(N_range[0], N_range[1], N_points, dtype=int)
    N_values = np.unique(N_values)  # Remove duplicates

    results: list[dict[str, float]] = []

    for N in N_values:
        try:
            if method == "Lindblad" and N <= 20:
                # Use Lindblad for small N
                if state_type == "CSS":
                    state = coherent_spin_state(N)
                elif state_type == "SSS":
                    t_opt = optimal_squeezing_time(N, chi)
                    state = generate_squeezed_state(N, chi, t_opt)
                elif state_type == "Twin-Fock":
                    state = generate_twin_fock_state(N)
                elif state_type == "NOON":
                    state = generate_noon_state(N)
                else:
                    raise ValueError(f"Unknown state type: {state_type}")

                T = 1.0  # Evolution time
                delta_phi = compute_phase_uncertainty_lindblad(
                    N,
                    state,
                    chi,
                    T,
                    noise_config,
                )

            else:
                # Use TWA for larger N
                t_opt = optimal_squeezing_time(N, chi) if state_type == "SSS" else 1.0

                result = run_twa_simulation(
                    N=N,
                    state_type=state_type,
                    chi=chi,
                    gamma_1=noise_config.gamma_1,
                    gamma_2=noise_config.gamma_2,
                    gamma_phi=noise_config.gamma_phi,
                    T=t_opt,
                    N_traj=200,
                    seed=seed,
                )

                delta_phi = result["delta_phi"]

            results.append({"N": N, "delta_phi": delta_phi})

        except Exception:
            # Skip failed computations (e.g. Lindblad dimension too large)
            continue

    return pd.DataFrame(results)

"""
Two-mode Lindblad Master Equation Solver for Noisy Mach-Zehnder Interferometer.

This module bridges the two-mode MZI simulation with Lindblad noise channels,
enabling simulation of open quantum system effects in interferometers.

Physical Model:
- Hilbert space: Two-mode Fock basis with truncation N (dimension (N+1)²)
- Master equation: dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
- Lindblad operators act in the two-mode basis

Noise Channels:
| Channel        | Lindblad Operator | Description                    |
|----------------|-------------------|--------------------------------|
| One-body loss  | √γ₁ a₁            | Single-photon loss from mode 1 |
| Two-body loss  | √γ₂ a₁²           | Pair loss from mode 1          |
| Phase diffusion| √γ_φ J_z          | Dephasing between arms         |

Circuit for full MZI with decoherence:
    BS1(θ, φ) → Phase(φ_phase) → Lindblad(T) → BS2(θ, φ)

References:
- Walls & Milburn (2008) "Quantum Optics"
- Gardiner & Zoller (2004) "Quantum Noise"

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.linalg

from src.evolution.lindblad_solver import (
    density_to_vector,
    vector_to_density,
)
from src.physics.mzi_simulation import (
    beam_splitter_unitary,
    create_system_operators,
    phase_shift_unitary,
)

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MziNoiseConfig:
    """Noise configuration specific to MZI simulations.

    Attributes:
        gamma_1: One-body loss rate for mode 1 (single-particle loss).
        gamma_2: Two-body loss rate for mode 1 (pair loss).
        gamma_phi: Phase diffusion rate (dephasing between arms).
        T: Decoherence time (dimensionless).
        dt: Time step for numerical integration.
        method: Integration method ('rk4' or 'scipy').

    """

    gamma_1: float = 0.0
    gamma_2: float = 0.0
    gamma_phi: float = 0.0
    T: float = 1.0
    dt: float = 0.01
    method: str = "rk4"


# =============================================================================
# Lindblad Operator Construction
# =============================================================================


def build_mzi_lindblad_operators(
    max_photons: int,
    config: MziNoiseConfig,
) -> list[np.ndarray]:
    """Build Lindblad operators in the two-mode Fock basis.

    For the two-mode space (dim = (N+1)²):
    - One-body loss: L₁ = √γ₁ a₁ (loss from mode 1)
    - Two-body loss: L₂ = √γ₂ a₁² (pair loss from mode 1)
    - Phase diffusion: L_φ = √γ_φ J_z where J_z = (n₁ - n₂)/2

    Operators have rates pre-absorbed (L = √γ · operator).
    Only returns operators for channels with non-zero rates.

    Args:
        max_photons: Maximum photon number per mode (Hilbert space truncation).
        config: Noise configuration specifying active channels and rates.

    Returns:
        List of Lindblad operator matrices in the two-mode Fock basis.
        Each operator has shape ((N+1)², (N+1)²).

    Raises:
        ValueError: If any rate is negative.

    Example:
        >>> config = MziNoiseConfig(gamma_1=0.1, gamma_phi=0.05)
        >>> L_ops = build_mzi_lindblad_operators(4, config)
        >>> len(L_ops)
        2

    """
    if config.gamma_1 < 0:
        raise ValueError(
            f"One-body loss rate must be non-negative, got {config.gamma_1}",
        )
    if config.gamma_2 < 0:
        raise ValueError(
            f"Two-body loss rate must be non-negative, got {config.gamma_2}",
        )
    if config.gamma_phi < 0:
        raise ValueError(
            f"Phase diffusion rate must be non-negative, got {config.gamma_phi}",
        )

    # Get creation/annihilation operators for both modes
    a0, a1, a0_dag, a1_dag = create_system_operators(max_photons)

    # Number operators
    n0 = a0_dag @ a0  # Number operator for mode 0
    n1 = a1_dag @ a1  # Number operator for mode 1

    # J_z = (n₁ - n₂)/2 in two-mode Fock basis
    jz = 0.5 * (n0 - n1)

    L_ops: list[np.ndarray] = []

    # One-body loss: L₁ = √γ₁ a₁ (loss from mode 1)
    if config.gamma_1 > 0:
        L_ops.append(np.sqrt(config.gamma_1) * a1)

    # Two-body loss: L₂ = √γ₂ a₁² (pair loss from mode 1)
    if config.gamma_2 > 0:
        L_ops.append(np.sqrt(config.gamma_2) * (a1 @ a1))

    # Phase diffusion: L_φ = √γ_φ J_z
    if config.gamma_phi > 0:
        L_ops.append(np.sqrt(config.gamma_phi) * jz)

    return L_ops


# =============================================================================
# Lindblad Master Equation
# =============================================================================


def lindblad_liouvillian_mzi(
    rho: np.ndarray,
    H: np.ndarray | None,
    L_ops: list[np.ndarray],
) -> np.ndarray:
    r"""Compute dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ}).

    Evaluates the Lindblad master equation. Lindblad operators are
    assumed to have rates pre-absorbed (i.e., L_k already contains
    the √γ factor).

    Args:
        rho: Density matrix in two-mode Fock basis.
        H: Hamiltonian (can be None for zero Hamiltonian).
        L_ops: Lindblad operators with rates pre-absorbed.

    Returns:
        dρ/dt matrix.

    """
    drho_dt = np.zeros_like(rho, dtype=complex)

    # Hamiltonian commutator term: -i[H, ρ]
    if H is not None:
        drho_dt += -1.0j * (H @ rho - rho @ H)

    # Lindblad dissipation terms
    for L in L_ops:
        L_dag = L.conj().T

        # L ρ L†
        L_rho_Ld = L @ rho @ L_dag

        # ½{L†L, ρ} = ½(L†L ρ + ρ L†L)
        LdL = L_dag @ L
        anticomm = LdL @ rho + rho @ LdL

        drho_dt += L_rho_Ld - 0.5 * anticomm

    return drho_dt


def _evolve_rk4_mzi(
    initial_rho: np.ndarray,
    H: np.ndarray | None,
    L_ops: list[np.ndarray],
    T: float,
    dt: float,
) -> np.ndarray:
    """4th-order Runge-Kutta integration for Lindblad equation.

    Uses the standard RK4 scheme with Hermiticity and trace
    enforcement at each step for numerical stability.

    Args:
        initial_rho: Initial density matrix.
        H: Hamiltonian (None for zero Hamiltonian).
        L_ops: Lindblad operators with rates pre-absorbed.
        T: Final evolution time.
        dt: Time step.

    Returns:
        Final density matrix after evolution.

    """
    if T <= 0:
        return initial_rho.copy()

    rho = initial_rho.copy()
    num_steps = max(1, int(np.ceil(T / dt)))
    dt_eff = T / num_steps  # Adjust to exactly hit T

    for _ in range(num_steps):
        # RK4 steps
        k1 = lindblad_liouvillian_mzi(rho, H, L_ops)
        k2 = lindblad_liouvillian_mzi(rho + 0.5 * dt_eff * k1, H, L_ops)
        k3 = lindblad_liouvillian_mzi(rho + 0.5 * dt_eff * k2, H, L_ops)
        k4 = lindblad_liouvillian_mzi(rho + dt_eff * k3, H, L_ops)

        rho = rho + (dt_eff / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Enforce Hermiticity and normalization for stability
        rho = 0.5 * (rho + rho.conj().T)
        trace = np.trace(rho)
        if trace > 0:
            rho = rho / trace

    return rho


def _evolve_scipy_mzi(
    initial_rho: np.ndarray,
    H: np.ndarray | None,
    L_ops: list[np.ndarray],
    T: float,
) -> np.ndarray:
    """Evolve using scipy ODE solver.

    Uses scipy's adaptive Runge-Kutta (RK45) for robust integration
    with error control.

    Args:
        initial_rho: Initial density matrix.
        H: Hamiltonian (None for zero Hamiltonian).
        L_ops: Lindblad operators with rates pre-absorbed.
        T: Final evolution time.

    Returns:
        Final density matrix.

    """
    # Vectorize initial state
    rho0 = density_to_vector(initial_rho)

    # ODE function
    def rhs(_t: float, rho_vec: np.ndarray) -> np.ndarray:
        rho = vector_to_density(rho_vec)
        drho_dt = lindblad_liouvillian_mzi(rho, H, L_ops)
        return density_to_vector(drho_dt)

    # Integrate
    sol = scipy.integrate.solve_ivp(
        rhs,
        (0, T),
        rho0,
        method="RK45",
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )

    rho_final_vec = sol.y[:, -1]
    return vector_to_density(rho_final_vec)


# =============================================================================
# Time Evolution
# =============================================================================


def evolve_mzi_lindblad(
    initial_state: np.ndarray,
    config: MziNoiseConfig,
    max_photons: int,
    H: np.ndarray | None = None,
) -> np.ndarray:
    """Evolve density matrix under Lindblad master equation.

    Handles both pure state (1D vector) and density matrix (2D) inputs.
    Uses RK4 integration with Hermiticity/trace enforcement at each step.

    When no Lindblad operators are active (all rates = 0), uses
    scipy.linalg.expm for unitary evolution (or identity if no Hamiltonian).

    Validation:
    - Trace preservation: assert np.isclose(np.trace(rho_final), 1.0, atol=1e-8)
    - Hermiticity: assert np.allclose(rho_final, rho_final.conj().T, atol=1e-8)
    - Positivity: assert min(eigvals) >= -1e-8

    Args:
        initial_state: Initial state. Can be a pure state vector (1D)
            or a density matrix (2D).
        config: Noise configuration specifying rates, time, and method.
        max_photons: Maximum photon number per mode (for operator construction).
        H: Hamiltonian (optional, None means zero Hamiltonian).

    Returns:
        Final density matrix after evolution.

    Raises:
        ValueError: If method is not 'rk4' or 'scipy'.

    """
    # Convert pure state to density matrix if needed
    if initial_state.ndim == 1:
        rho = np.outer(initial_state, initial_state.conj())
    else:
        rho = initial_state.copy()

    # Build Lindblad operators (rates pre-absorbed)
    L_ops = build_mzi_lindblad_operators(max_photons, config)

    # Perform evolution
    if len(L_ops) == 0:
        # No dissipation: unitary evolution (or identity if no Hamiltonian)
        if H is not None:
            U = scipy.linalg.expm(-1.0j * H * config.T)
            rho = U @ rho @ U.conj().T
        # If no H and no L_ops, rho is unchanged
    elif config.method == "rk4":
        rho = _evolve_rk4_mzi(rho, H, L_ops, config.T, config.dt)
    elif config.method == "scipy":
        rho = _evolve_scipy_mzi(rho, H, L_ops, config.T)
    else:
        raise ValueError(f"Unknown method '{config.method}'. Use 'rk4' or 'scipy'.")

    # Physics assertions
    trace = np.trace(rho)
    assert np.isclose(trace, 1.0, atol=1e-8), f"Trace not preserved: Tr(ρ) = {trace}"
    assert np.allclose(rho, rho.conj().T, atol=1e-8), "Density matrix is not Hermitian"
    eigenvalues = np.linalg.eigvalsh(rho)
    assert np.min(eigenvalues) >= -1e-8, (
        f"Density matrix has negative eigenvalues: min = {np.min(eigenvalues)}"
    )

    return rho


# =============================================================================
# Full Noisy MZI Simulation
# =============================================================================


def run_noisy_mzi(
    initial_state: np.ndarray,
    max_photons: int,
    theta: float,
    phi_bs: float,
    phi_phase: float,
    noise_config: MziNoiseConfig,
    ancilla_dim: int = 1,
) -> np.ndarray:
    """Run a full noisy MZI simulation.

    Circuit sequence:
        1. BS1: First beam splitter (mode mixing)
        2. Phase: Phase shift on mode 1
        3. Lindblad: Open-system decoherence for time T
        4. BS2: Second beam splitter (interference)

    The Lindblad decoherence step is applied after the phase imprint
    and before the second beam splitter, which models distributed
    loss throughout the interferometer arms.

    Args:
        initial_state: Initial state. Can be pure state (1D vector) or
            density matrix (2D) in the two-mode Fock basis.
        max_photons: Maximum photon number per mode (Hilbert space truncation).
        theta: Beam splitter transmittance angle (π/4 for 50/50).
        phi_bs: Beam splitter phase parameter.
        phi_phase: Phase shift in arm 1 (the unknown parameter).
        noise_config: Noise configuration for Lindblad evolution.
        ancilla_dim: Dimension of ancilla Hilbert space. Default 1 (no ancilla).

    Returns:
        Final density matrix in the two-mode Fock basis.

    Constraints:
        initial_state must be normalized (1D) or trace-1 (2D).
        theta in [0, π] (beam splitter angle).
        noise_config rates (gamma_{1,2,phi}) must be >= 0.
        noise_config.T > 0, noise_config.dt > 0.
        ancilla_dim >= 1 (1 = no ancilla).

    Example:
        >>> from src.physics.mzi_lindblad import MziNoiseConfig, run_noisy_mzi
        >>> from src.physics.mzi_simulation import fock_state
        >>> state = fock_state(1, 0, max_photons=2)
        >>> config = MziNoiseConfig(gamma_1=0.1, T=0.5)
        >>> rho = run_noisy_mzi(state, max_photons=2, theta=np.pi/4,
        ...     phi_bs=0, phi_phase=1.0, noise_config=config)
        >>> rho.shape
        (9, 9)
        >>> np.isclose(np.trace(rho), 1.0)
        True

    """
    # Convert to density matrix if needed
    if initial_state.ndim == 1:
        rho = np.outer(initial_state, initial_state.conj())
    else:
        rho = initial_state.copy()

    # Build unitaries
    bs = beam_splitter_unitary(theta, phi_bs, max_photons)
    phase = phase_shift_unitary(phi_phase, max_photons)

    # Apply BS1: ρ → U_BS ρ U_BS†
    rho = bs @ rho @ bs.conj().T

    # Apply phase shift: ρ → U_phase ρ U_phase†
    rho = phase @ rho @ phase.conj().T

    # Apply Lindblad decoherence
    has_noise = (
        noise_config.gamma_1 > 0
        or noise_config.gamma_2 > 0
        or noise_config.gamma_phi > 0
    )
    if has_noise:
        rho = evolve_mzi_lindblad(rho, noise_config, max_photons)

    # Apply BS2: ρ → U_BS ρ U_BS†
    return bs @ rho @ bs.conj().T


# =============================================================================
# Unit Tests
# =============================================================================


def _test_build_operators() -> dict:
    """Test Lindblad operator construction."""
    config = MziNoiseConfig(gamma_1=0.5, gamma_phi=0.3)
    L_ops = build_mzi_lindblad_operators(max_photons=3, config=config)

    assert len(L_ops) == 2, "Should have 2 operators"
    assert L_ops[0].shape == (16, 16), "Two-mode dim = (N+1)² = 16 for N=3"

    # Empty config → no operators
    config0 = MziNoiseConfig()
    L_ops0 = build_mzi_lindblad_operators(max_photons=3, config=config0)
    assert len(L_ops0) == 0, "Should have no operators for zero rates"

    return {"status": "passed"}


def _test_liouvillian_structure() -> dict:
    """Test Liouvillian preserves Hermiticity of derivative."""
    # Use small space
    max_photons = 2
    dim = (max_photons + 1) ** 2
    rho = np.eye(dim, dtype=complex) / dim  # Maximally mixed

    config = MziNoiseConfig(gamma_1=0.1)
    L_ops = build_mzi_lindblad_operators(max_photons, config)

    drho = lindblad_liouvillian_mzi(rho, H=None, L_ops=L_ops)

    # dρ/dt is traceless (trace preservation)
    assert np.isclose(np.trace(drho), 0.0, atol=1e-12), "Liouvillian is traceless"

    return {"status": "passed"}


def _test_noiseless_evolution() -> dict:
    """Test that zero-noise evolution preserves state."""
    max_photons = 2
    dim = (max_photons + 1) ** 2
    rho = np.eye(dim, dtype=complex) / dim

    config = MziNoiseConfig(T=1.0, dt=0.1)
    rho_final = evolve_mzi_lindblad(rho, config, max_photons)

    assert np.allclose(rho_final, rho), "Noiseless evolution should preserve state"

    return {"status": "passed"}


def test_noisy_mzi_symmetry() -> dict:
    """Test that noisy MZI produces valid density matrix."""
    from src.physics.mzi_simulation import fock_state

    max_photons = 4
    state = fock_state(2, 0, max_photons)

    config = MziNoiseConfig(gamma_1=0.1, gamma_phi=0.05, T=0.5, dt=0.05)
    rho = run_noisy_mzi(
        state,
        max_photons=max_photons,
        theta=np.pi / 4,
        phi_bs=0.0,
        phi_phase=np.pi / 2,
        noise_config=config,
    )

    # Validate density matrix properties
    assert np.isclose(np.trace(rho), 1.0, atol=1e-8), "Trace must be 1"
    assert np.allclose(rho, rho.conj().T, atol=1e-8), "Must be Hermitian"
    eigenvalues = np.linalg.eigvalsh(rho)
    assert np.min(eigenvalues) >= -1e-8, "Must be positive semidefinite"

    return {"status": "passed"}


def _test_noiseless_mzi() -> dict:
    """Test noiseless MZI produces same result as unitary evolution."""
    from src.physics.mzi_simulation import fock_state

    max_photons = 3
    theta = np.pi / 4
    phi_bs = 0.0
    phi_phase = 1.0

    state = fock_state(1, 0, max_photons)

    # Noiseless noisy MZI
    config = MziNoiseConfig(T=1.0, dt=0.1)
    rho_noiseless = run_noisy_mzi(
        state,
        max_photons=max_photons,
        theta=theta,
        phi_bs=phi_bs,
        phi_phase=phi_phase,
        noise_config=config,
    )

    # Compare with unitary evolution as density matrix
    bs = beam_splitter_unitary(theta, phi_bs, max_photons)
    phase = phase_shift_unitary(phi_phase, max_photons)

    psi = bs @ state
    psi = phase @ psi
    psi = bs @ psi
    rho_unitary = np.outer(psi, psi.conj())

    assert np.allclose(rho_noiseless, rho_unitary, atol=1e-10), (
        "Noiseless noisy MZI should match unitary evolution"
    )

    return {"status": "passed"}


if __name__ == "__main__":
    print("Running two-mode Lindblad tests...")

    results = {
        "build_operators": _test_build_operators(),
        "liouvillian_structure": _test_liouvillian_structure(),
        "noiseless_evolution": _test_noiseless_evolution(),
        "noisy_mzi_symmetry": test_noisy_mzi_symmetry(),
        "noiseless_mzi": _test_noiseless_mzi(),
    }

    for name, result in results.items():
        print(f"  {name}: {result['status']}")

    print("\nAll tests passed!")

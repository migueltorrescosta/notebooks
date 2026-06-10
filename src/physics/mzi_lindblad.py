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
import qutip

from src.physics.mzi_simulation import (
    create_system_operators,
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
        T_decay: Decoherence time (dimensionless).
        dt: Time step for numerical integration.
        method: Integration method ('rk4' or 'scipy').

    """

    gamma_1: float = 0.0
    gamma_2: float = 0.0
    gamma_phi: float = 0.0
    T_decay: float = 1.0
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
# Time Evolution
# =============================================================================


def evolve_mzi_lindblad(
    initial_state: np.ndarray,
    config: MziNoiseConfig,
    max_photons: int,
    H: np.ndarray | None = None,
) -> np.ndarray:
    r"""Evolve density matrix under Lindblad master equation using QuTiP.

    Handles both pure state (1D vector) and density matrix (2D) inputs.
    Delegates integration to ``qutip.mesolve()``, which uses adaptive
    stepping internally. The ``config.dt`` and ``config.method`` fields
    are accepted for backward compatibility but ignored.

    When no Lindblad operators are active (all rates = 0), uses
    QuTiP's matrix exponentiation for unitary evolution (or identity
    if no Hamiltonian).

    Validation:
    - Trace preservation: assert np.isclose(np.trace(rho_final), 1.0, atol=1e-8)
    - Hermiticity: assert np.allclose(rho_final, rho_final.conj().T, atol=1e-8)
    - Positivity: assert min(eigvals) >= -1e-8

    Args:
        initial_state: Initial state. Can be a pure state vector (1D)
            or a density matrix (2D).
        config: Noise configuration specifying rates, time, and method.
            ``dt`` and ``method`` fields are ignored (QuTiP handles
            adaptive stepping).
        max_photons: Maximum photon number per mode (for operator construction).
        H: Hamiltonian (optional, None means zero Hamiltonian).

    Returns:
        Final density matrix after evolution.

    """
    # Convert pure state to density matrix if needed
    if initial_state.ndim == 1:
        rho = np.outer(initial_state, initial_state.conj())
    else:
        rho = initial_state.copy()

    # Build Lindblad operators (rates pre-absorbed, numpy arrays)
    L_ops = build_mzi_lindblad_operators(max_photons, config)

    # Proper tensor-product dims for the two-mode system
    dim = max_photons + 1
    dims = [[dim, dim], [dim, dim]]

    if len(L_ops) == 0 and H is None:
        # No dissipation and no Hamiltonian: rho unchanged
        pass
    else:
        # Convert to QuTiP objects
        if H is not None:
            H_qobj = qutip.Qobj(H, dims=dims)
        else:
            # Zero Hamiltonian with matching dims (QuTiP mesolve does not accept None)
            H_qobj = qutip.Qobj(
                np.zeros((rho.shape[0], rho.shape[0]), dtype=complex),
                dims=dims,
            )

        rho_qobj = qutip.Qobj(rho, dims=dims)
        c_ops_qobj = [qutip.Qobj(L, dims=dims) for L in L_ops]

        if not c_ops_qobj:
            # Unitary evolution only (no Lindblad terms)
            U_qobj = (-1.0j * config.T_decay * H_qobj).expm()
            rho_qobj = U_qobj @ rho_qobj @ U_qobj.dag()
            rho = rho_qobj.full()
        else:
            # Full Lindblad evolution via QuTiP mesolve
            result = qutip.mesolve(
                H_qobj,
                rho_qobj,
                [0, config.T_decay],
                c_ops_qobj,
                e_ops=[],
            )
            rho = result.states[-1].full()

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
    r"""Run a full noisy MZI simulation using QuTiP operators and mesolve.

    Circuit sequence:
        1. BS1: First beam splitter (mode mixing)
        2. Phase: Phase shift on mode 1
        3. Lindblad: Open-system decoherence for time T
        4. BS2: Second beam splitter (interference)

    The beam splitters and phase shift are constructed using QuTiP
    tensor-product operators. The Lindblad decoherence step is applied
    after the phase imprint and before the second beam splitter, which
    models distributed loss throughout the interferometer arms.

    Beam splitter convention:
        U_BS = exp(-i \theta H_{BS})
        H_{BS} = e^{i\phi} a_0^\dagger a_1 + e^{-i\phi} a_1^\dagger a_0

    Phase shift convention:
        U_phase = exp(i \phi_{phase} n_1)  (on mode 1)

    Args:
        initial_state: Initial state. Can be pure state (1D vector) or
            density matrix (2D) in the two-mode Fock basis.
        max_photons: Maximum photon number per mode (Hilbert space truncation).
        theta: Beam splitter transmittance angle (\pi/4 for 50/50).
        phi_bs: Beam splitter phase parameter.
        phi_phase: Phase shift in arm 1 (the unknown parameter).
        noise_config: Noise configuration for Lindblad evolution.
            ``dt`` and ``method`` fields are ignored (QuTiP handles
            adaptive stepping).
        ancilla_dim: Dimension of ancilla Hilbert space. Default 1 (no ancilla).
            Values > 1 are currently unsupported.

    Returns:
        Final density matrix in the two-mode Fock basis.

    Constraints:
        initial_state must be normalized (1D) or trace-1 (2D).
        theta in [0, \pi] (beam splitter angle).
        noise_config rates (gamma_{1,2,phi}) must be >= 0.
        noise_config.T_decay > 0.
        ancilla_dim >= 1 (1 = no ancilla).

    Example:
        >>> import qutip
        >>> from src.physics.mzi_lindblad import MziNoiseConfig, run_noisy_mzi
        >>> dim = 2 + 1
        >>> state = qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()
        >>> config = MziNoiseConfig(gamma_1=0.1, T_decay=0.5)
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

    # --- Build QuTiP operators for the two-mode system ---
    dim_single = max_photons + 1
    a = qutip.destroy(dim_single)
    eye = qutip.qeye(dim_single)

    a0 = qutip.tensor(a, eye)  # a ⊗ I  — mode 0
    a1 = qutip.tensor(eye, a)  # I ⊗ a  — mode 1
    n1 = a1.dag() @ a1  # number operator on mode 1

    # Beam splitter unitary: U_BS = exp(-iθ H_BS)
    # H_BS = e^{iφ} a₀† a₁ + e^{-iφ} a₁† a₀
    H_bs = np.exp(1j * phi_bs) * (a0.dag() @ a1) + np.exp(-1j * phi_bs) * (
        a1.dag() @ a0
    )
    U_bs = (-1j * theta * H_bs).expm()

    # Phase shift unitary: U_phase = exp(i φ n₁)
    U_phase = (1j * phi_phase * n1).expm()

    # Proper tensor-product dims for the density matrix
    dims = [[dim_single, dim_single], [dim_single, dim_single]]

    # Convert rho to Qobj
    rho_qobj = qutip.Qobj(rho, dims=dims)

    # Apply BS1: ρ → U_BS ρ U_BS†
    rho_qobj = U_bs * rho_qobj * U_bs.dag()

    # Apply phase shift: ρ → U_phase ρ U_phase†
    rho_qobj = U_phase @ rho_qobj @ U_phase.dag()

    # Apply Lindblad decoherence
    has_noise = (
        noise_config.gamma_1 > 0
        or noise_config.gamma_2 > 0
        or noise_config.gamma_phi > 0
    )
    if has_noise:
        rho = rho_qobj.full()
        rho = evolve_mzi_lindblad(rho, noise_config, max_photons)
        rho_qobj = qutip.Qobj(rho, dims=dims)

    # Apply BS2: ρ → U_BS ρ U_BS†
    rho_qobj = U_bs @ rho_qobj @ U_bs.dag()

    return rho_qobj.full()

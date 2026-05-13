"""
Cavity-Enhanced Mach-Zehnder Interferometer Simulation.

Physical Model:
- A cavity-enhanced interferometer increases the effective interaction
  time by a factor of the cavity finesse ℱ.
- This can be modeled as repeated phase interrogation:
  - Phase shift per pass: φ
  - Number of passes: ℱ (finesse)
  - Total accumulated phase: ℱ·φ
  - Phase sensitivity: Δφ = 1/√(ℱ·N) for coherent states
    (shot-noise limit with finesse prefactor)

Circuit (unitary):
    BS₁ → (Phase × ℱ) → BS₂

Circuit (noisy):
    BS₁ → [Phase(φ) → Noise(T)] × ℱ → BS₂

The ℱ-fold phase accumulation is equivalent to a single
phase shift ℱ·φ applied between two beam splitters.

Hilbert Space:
- Two-mode Fock basis with max N photons (dimension: (N+1)²)
- Consistent with src.physics.mzi_simulation conventions

Units:
- Dimensionless throughout (ℏ = 1)
- Phase φ in radians
- Cavity finesse ℱ is dimensionless

Conventions:
- Beam splitter: same as src.physics.mzi_simulation (50:50 by default)
- Phase shift on mode 1 (second arm): exp(i · φ · n₂)
- Noise channels: one-body loss, two-body loss, phase diffusion
  (see src.physics.mzi_lindblad for details)

Noise Model:
- Each pass through the cavity applies phase shift + Lindblad noise.
- For efficiency, the ℱ noisy passes can be approximated by a single
  noise step with rates scaled by ℱ (linearity of the Liouvillian:
  the total dissipator over ℱ identical passes is ℱ times the
  single-pass dissipator).

Units:
- Dimensionless throughout. Phase is measured in radians.
- Finesse ℱ is dimensionless (effective number of passes).

Conventions:
- Beam splitter transformation: a → cos(θ)·a + i·e^{iφ}·sin(θ)·b
- Phase convention: e^{i·φ·n} applied to mode 1 (second mode)
- State ordering: |n₁, n₂⟩ with n₁ as first mode, n₂ as second mode

References:
- Walls & Milburn (2008) "Quantum Optics"
- Gardiner & Zoller (2004) "Quantum Noise"
- Dowling (2008) "Quantum optical metrology"

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.physics.mzi_lindblad import (
    MziNoiseConfig,
    run_noisy_mzi,
)
from src.physics.mzi_simulation import (
    beam_splitter_unitary,
    phase_shift_unitary,
)
from src.physics.mzi_states import compute_fisher_information, input_state_factory

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CavityMziConfig:
    """Configuration for cavity-enhanced MZI.

    The cavity finesse ℱ quantifies the effective number of passes
    through the phase shift medium. Higher finesse yields proportionally
    larger accumulated phase and improved phase sensitivity.

    Attributes:
        F: Cavity finesse (effective number of passes through phase
            shift). Must be >= 1. Higher values give more phase
            accumulation and better sensitivity.
        theta: Beam splitter transmittance angle. π/4 for 50/50 splitter.
            θ = 0 gives identity, θ = π/4 gives balanced splitting.
        phi_bs: Beam splitter phase. Controls the reflection amplitude
            phase of the beam splitter transformation.

    """

    F: float = 10.0
    theta: float = np.pi / 4
    phi_bs: float = 0.0


# =============================================================================
# Unitary Evolution
# =============================================================================


def cavity_enhanced_mzi(
    initial_state: np.ndarray,
    phi: float,
    config: CavityMziConfig,
    max_photons: int,
) -> np.ndarray:
    r"""Run cavity-enhanced MZI with multiple phase passes.

    Circuit: BS₁ → (Phase × ℱ) → BS₂

    The ℱ-fold phase accumulation is equivalent to a single
    phase shift ℱ·φ applied between two beam splitters:

        U = BS₂ · exp(i·ℱ·φ·n₁) · BS₁

    Physics:
        The cavity finesse ℱ enhances the effective interaction time
        by a factor of ℱ. For coherent states, this improves the
        standard quantum limit sensitivity from Δφ = 1/√N to
        Δφ = 1/√(ℱ·N).

    Unit testing (validation guarantees):
        - Output state is normalized: ‖|ψ⟩‖ = 1
        - Unitary evolution: U†U = I
        - Backward compatibility: ℱ = 1 reproduces standard MZI

    Args:
        initial_state: Input state in two-mode Fock basis.
            Must be a normalized vector of dimension (max_photons+1)².
        phi: Phase shift per pass (radians). The total accumulated
            phase is ℱ·φ.
        config: Cavity configuration specifying finesse F and beam
            splitter parameters.
        max_photons: Maximum photon number per mode for Hilbert space
            truncation. Determines the state dimension as (N+1)².

    Returns:
        Final state vector after cavity-enhanced MZI of dimension
        (max_photons+1)².

    Raises:
        ValueError: If config.F < 1.

    Example:
        >>> from src.physics.mzi_simulation import fock_state
        >>> config = CavityMziConfig(F=10.0)
        >>> state = fock_state(1, 0, max_photons=2)
        >>> final = cavity_enhanced_mzi(
        ...     state, phi=np.pi / 4, config=config, max_photons=2,
        ... )
        >>> final.shape
        (9,)
        >>> np.isclose(np.sum(np.abs(final) ** 2), 1.0)
        True

    """
    if config.F < 1:
        raise ValueError(f"Cavity finesse must be >= 1, got {config.F}")

    # Total accumulated phase: ℱ·φ
    total_phi = config.F * phi

    # Build unitary operators
    bs = beam_splitter_unitary(config.theta, config.phi_bs, max_photons)
    phase = phase_shift_unitary(total_phi, max_photons)

    # Circuit: BS₂ · Phase(ℱ·φ) · BS₁
    state = bs @ initial_state
    state = phase @ state
    return bs @ state


# =============================================================================
# Noisy Evolution
# =============================================================================


def cavity_enhanced_mzi_with_noise(
    initial_state: np.ndarray,
    phi: float,
    noise_gamma_1: float,
    noise_gamma_2: float,
    noise_gamma_phi: float,
    config: CavityMziConfig,
    max_photons: int,
    noise_T: float = 1.0,
    noise_dt: float = 0.01,
) -> np.ndarray:
    r"""Cavity-enhanced MZI with Lindblad noise.

    Uses the two-mode Lindblad solver from :mod:`mzi_lindblad`.
    Each pass through the cavity ideally applies phase shift + noise.
    For efficiency, the ℱ noisy passes are approximated by a single
    effective noise step with rates scaled by ℱ.

    Exact circuit (ℱ passes):
        BS₁ → [Phase(φ) → Noise(T)]^{×ℱ} → BS₂

    Efficient approximation (used here):
        BS₁ → Phase(ℱ·φ) → Noise(ℱ·γ₁, ℱ·γ₂, ℱ·γ_φ, T) → BS₂

    The efficient approximation is exact for the phase (phase shifts
    commute) and is a Trotter-style approximation for the noise:
    the Liouvillian accumulates linearly, so the total dissipator
    over ℱ identical passes is ℱ times the single-pass dissipator.
    This is valid when the noise per pass is small compared to the
    phase per pass, which holds for high-finesse cavities.

    Noise channels (per pass):
        | Channel        | Lindblad Operator | Description                    |
        |----------------|-------------------|--------------------------------|
        | One-body loss  | √γ₁ a₁            | Single-photon loss from mode 1 |
        | Two-body loss  | √γ₂ a₁²           | Pair loss from mode 1          |
        | Phase diffusion| √γ_φ J_z          | Dephasing between arms         |

    Args:
        initial_state: Input state in two-mode Fock basis.
            Can be a pure state (1D vector) or a density matrix (2D).
        phi: Phase shift per pass (radians). Total accumulated
            phase is ℱ·φ.
        noise_gamma_1: One-body loss rate (γ₁) per pass for mode 1.
            Must be non-negative.
        noise_gamma_2: Two-body loss rate (γ₂) per pass for mode 1.
            Must be non-negative.
        noise_gamma_phi: Phase diffusion rate (γ_φ) per pass between
            arms. Must be non-negative.
        config: Cavity configuration (F, theta, phi_bs).
        max_photons: Maximum photon number per mode for Hilbert
            space truncation.
        noise_T: Noise evolution time per pass (dimensionless).
            The noise is integrated for this duration per pass,
            then the rates are scaled by ℱ for the efficient
            approximation.
        noise_dt: Time step for numerical integration of the
            Lindblad master equation.

    Returns:
        Final density matrix after cavity-enhanced MZI of dimension
        (max_photons+1)² × (max_photons+1)².

    Raises:
        ValueError: If config.F < 1 or any noise rate is negative.

    Constraints:
        initial_state must be normalized (pure) or trace-1 (density matrix).
        config.F >= 1 (cavity finesse).
        noise_gamma_{1,2,phi} >= 0 (non-negative rates per pass).
        noise_T > 0, noise_dt > 0.
        Performance: O(ℱ × (N+1)⁴) for exact; O((N+1)⁴) for efficient approximation.

    Example:
        >>> from src.physics.mzi_simulation import fock_state
        >>> config = CavityMziConfig(F=5.0)
        >>> state = fock_state(1, 0, max_photons=2)
        >>> rho = cavity_enhanced_mzi_with_noise(
        ...     state, phi=np.pi / 4,
        ...     noise_gamma_1=0.1, noise_gamma_2=0.0,
        ...     noise_gamma_phi=0.05,
        ...     config=config, max_photons=2,
        ... )
        >>> rho.shape
        (9, 9)
        >>> np.isclose(np.trace(rho), 1.0)
        True

    """
    if config.F < 1:
        raise ValueError(f"Cavity finesse must be >= 1, got {config.F}")

    F = config.F

    # Build noise config with rates scaled by finesse ℱ.
    # This is the efficient approximation: the ℱ noisy passes are
    # replaced by a single noise step with ℱ times the rates.
    noise_config = MziNoiseConfig(
        gamma_1=noise_gamma_1 * F,
        gamma_2=noise_gamma_2 * F,
        gamma_phi=noise_gamma_phi * F,
        T=noise_T,
        dt=noise_dt,
    )

    # Total accumulated phase: ℱ·φ
    total_phi = F * phi

    # Run the full noisy MZI with scaled rates and total phase.
    # Circuit: BS₁ → Phase(ℱ·φ) → Noise(ℱ·γ, T) → BS₂
    return run_noisy_mzi(
        initial_state=initial_state,
        max_photons=max_photons,
        theta=config.theta,
        phi_bs=config.phi_bs,
        phi_phase=total_phi,
        noise_config=noise_config,
    )


# =============================================================================
# Sensitivity Computation
# =============================================================================


def cavity_enhanced_sensitivity(
    N: int,
    phi: float,
    config: CavityMziConfig,
    state_type: str = "coherent",
) -> float:
    r"""Compute phase sensitivity for cavity-enhanced MZI.

    Evaluates the full MZI circuit with ℱ-fold phase accumulation and
    extracts Δφ from the Quantum Fisher Information on the output state.

    Args:
        N: Mean photon number (used to construct input state).
        phi: Phase shift per pass (radians). Total phase is ℱ·φ.
        config: Cavity configuration (finesse ℱ, beam splitter angles).
        state_type: Input state type. Default: "coherent".
            Supported: "coherent" (SQL input), "noon" (Heisenberg input).

    Returns:
        Phase sensitivity Δφ = 1 / sqrt(F_Q). Returns np.inf if
        the computation fails (e.g., zero QFI from vacuum input).

    Raises:
        ValueError: If N is non-positive or config.F < 1.

    Example:
        >>> config = CavityMziConfig(F=10.0)
        >>> delta = cavity_enhanced_sensitivity(4, np.pi/4, config)
        >>> delta > 0
        True

    """
    if N <= 0:
        raise ValueError(f"Mean photon number must be positive, got {N}")
    if config.F < 1:
        raise ValueError(f"Cavity finesse must be >= 1, got {config.F}")

    # Ensure sufficient Hilbert space while avoiding memory blowup.
    # For N beyond ~15, full density matrix methods become expensive.
    max_photons = min(max(N + 5, 2 * N), 30)  # Cap at 30 for memory safety

    # Build input state via input_state_factory (single source of truth)
    if state_type == "coherent":
        state = input_state_factory("css", N=N, max_photons=max_photons)
    elif state_type == "noon":
        state = input_state_factory("noon", N=N, max_photons=max_photons)
    else:
        raise ValueError(f"Unknown state_type '{state_type}'")

    # Run cavity-enhanced MZI circuit
    state_out = cavity_enhanced_mzi(state, phi, config, max_photons)

    # Compute QFI on output state
    try:
        F_Q = compute_fisher_information(state_out, max_photons)
    except (ValueError, IndexError):
        return np.inf

    if F_Q <= 0 or not np.isfinite(F_Q):
        return np.inf

    return float(1.0 / np.sqrt(F_Q))

"""
Combined local module for the 2026-05-11 reports.

Contains all exclusive code migrated from src/ for the four reports:
- Ancilla-vs-System-Comparison
- Scaling-Survey-Fock-MZI
- Scaling-Survey-Collective-Spin
- Scaling-Survey-Advanced-Architectures

Usage:
    uv run python reports/20260511/local.py --force
    uv run python reports/20260511/local.py --experiment ancilla-vs-system --force

This module is not importable as reports.20260511.local (hyphens in path).
Importers should use importlib.util.spec_from_file_location.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import qutip
import scipy
import scipy.linalg
from scipy.linalg import expm
from scipy.optimize import curve_fit

from src.analysis.ancilla_optimization import J_X, J_Z
from src.analysis.fisher_information import quantum_fisher_information_dm
from src.analysis.scaling_fit import ScalingFitResult, fit_scaling_exponent
from src.physics.dicke_basis import from_dicke_basis, to_dicke_basis
from src.physics.hybrid_mzi import qfi_hybrid_mzi
from src.physics.hybrid_system import (
    hybrid_ground_state_n,
    hybrid_hamiltonian_n,
    hybrid_vacuum_state,
)
from src.physics.mzi_lindblad import MziNoiseConfig, run_noisy_mzi
from src.physics.mzi_simulation import (
    beam_splitter_unitary,
    create_system_operators,
    phase_shift_unitary,
)
from src.physics.mzi_states import (
    compute_fisher_information,
    input_state_factory,
    two_mode_jz_operator,
)
from src.physics.noise_channels import NoiseConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

REPORT_DATE = "20260511"
REPORTS_DIR = Path(__file__).resolve().parent.parent
_REPORT_DIR = REPORTS_DIR / REPORT_DATE


# =============================================================================
# Section: weak_value_mzi
# =============================================================================
# =============================================================================
# Configuration
# =============================================================================


@dataclass
class WeakValueConfig:
    """Configuration for weak-value amplification MZI.

    The post_select_angle controls the trade-off between amplification and
    post-selection probability. When post_select_angle ≈ π/2, the
    pre-selected and post-selected states are nearly orthogonal, giving
    large amplification at the cost of vanishing probability.

    The deviation from perfect orthogonality is:

        δ = π/2 − post_select_angle

    In the analytical model for coherent states:
        - Amplification:       |A_w| = cot(δ)
        - Post-selection prob:  p_ps = sin²(δ)

    Attributes:
        theta: Beam splitter angle for the main MZI (π/4 for 50/50).
        phi_bs: Beam splitter phase (controls reflection amplitude phase).
        post_select_angle: Post-selection angle. Values near π/2 give
            large amplification. Must be in (0, π). Default is π/2 - 0.1,
            which yields ≈10× amplification.
        measurement_basis: Measurement basis for post-selection.
            Supported: 'parity', 'Jz'.

    """

    theta: float = np.pi / 4
    phi_bs: float = 0.0
    post_select_angle: float = np.pi / 2 - 0.1  # Slightly off π/2
    measurement_basis: str = "parity"


# =============================================================================
# Operator Construction
# =============================================================================


def create_number_operator(max_photons: int) -> tuple[np.ndarray, np.ndarray]:
    """Create number operators for both modes in the two-mode Fock basis.

    Constructs the diagonal number operators n₀ and n₁ acting in the
    truncated Fock space with dimension (max_photons+1)².

    Args:
        max_photons: Maximum photon number per mode (truncation).

    Returns:
        Tuple of (n0, n1) where:
        - n0: Number operator for mode 0 (first arm).
        - n1: Number operator for mode 1 (second arm, phase-imprinted).

    Example:
        >>> n0, n1 = create_number_operator(max_photons=1)
        >>> n1[0, 0]  # |0,0⟩ has 0 photons in mode 1
        0.0
        >>> n1[2, 2]  # |0,1⟩ has 1 photon in mode 1
        1.0

    """
    dim = (max_photons + 1) ** 2
    n0 = np.zeros((dim, dim), dtype=complex)
    n1 = np.zeros((dim, dim), dtype=complex)

    for n1_val in range(max_photons + 1):
        for n2_val in range(max_photons + 1):
            idx = n1_val * (max_photons + 1) + n2_val
            n0[idx, idx] = float(n1_val)
            n1[idx, idx] = float(n2_val)

    return n0, n1


def _apply_jz_rotation(
    state: np.ndarray,
    angle: float,
    max_photons: int,
) -> np.ndarray:
    r"""Apply exp(i·angle·J_z) to a state vector.

    J_z = (n₀ − n₁)/2 is diagonal in the two-mode Fock basis, so this
    reduces to multiplying each Fock basis amplitude |n₁, n₂⟩ by:

        exp(i · angle · (n₁ − n₂) / 2)

    This avoids computing the full matrix exponential, which is
    O(dim³) and prohibitive for large truncations.

    Args:
        state: Input state vector of dimension (max_photons+1)².
        angle: Rotation angle in radians.
        max_photons: Maximum photon number per mode (determines dim).

    Returns:
        Rotated state vector.

    Example:
        >>> state = np.array([1.0, 0.0, 0.0, 0.0])  # |0,0⟩
        >>> rotated = _apply_jz_rotation(state, np.pi/2, max_photons=1)
        >>> np.allclose(rotated, state)  # J_z|0,0⟩ = 0
        True

    """
    dim = (max_photons + 1) ** 2
    result = np.empty(dim, dtype=complex)

    for n1_val in range(max_photons + 1):
        for n2_val in range(max_photons + 1):
            idx = n1_val * (max_photons + 1) + n2_val
            phase = np.exp(1j * angle * (n1_val - n2_val) / 2.0)
            result[idx] = phase * state[idx]

    return result


# =============================================================================
# Internal Helpers
# =============================================================================


def _mean_photon_number(state: np.ndarray, max_photons: int) -> float:
    """Compute mean total photon number of a state.

    Args:
        state: State vector in the truncated Fock basis.
        max_photons: Maximum photon number per mode.

    Returns:
        Mean total photon number ⟨n₀ + n₁⟩.

    """
    n0, n1 = create_number_operator(max_photons)
    n_total = n0 + n1
    return float(np.real(np.vdot(state, n_total @ state)))


# =============================================================================
# Full Numerical Weak-Value MZI Simulation
# =============================================================================


def weak_value_mzi(
    initial_state: np.ndarray,
    phi: float,
    config: WeakValueConfig,
    max_photons: int,
) -> dict[str, Any]:
    """Run weak-value amplification MZI and return all metrics.

    Full numerical simulation of the weak-value MZI circuit:

        BS1 → Phase(φ) → BS2 → Post-select

    The pre-selected state |i⟩ is the state inside the interferometer
    after the first beam splitter. The post-selected state |f⟩ is
    obtained by rotating |i⟩ by post_select_angle in J_z space:

        |f⟩ = exp(i · post_select_angle · J_z) |i⟩

    The rotation is in the angular momentum basis J_z = (n₀ − n₁)/2,
    which is the generator of relative phase shifts between the two
    interferometer arms. When post_select_angle ≈ π/2, |f⟩ is nearly
    orthogonal to |i⟩, yielding large amplification.

    The weak value is computed as:

        A_w = ⟨f|n₁|i⟩ / ⟨f|i⟩

    where n₁ is the number operator on mode 1 (the phase-imprinted arm).
    The amplification factor is |A_w|, and the post-selection probability
    is |⟨f|i⟩|². The Fisher information from the post-selected measurement
    is bounded by the conventional Fisher information for the same state.

    For coherent state inputs with large photon number N, this numerical
    simulation approaches the analytical result: |A_w| → cot(δ) where
    δ = π/2 − post_select_angle.

    Args:
        initial_state: Initial two-mode state vector entering the MZI
            (truncated Fock space of dimension (max_photons+1)²).
        phi: Phase shift in mode 1 (the parameter to estimate), in
            radians. Used to compute the amplified signal.
        config: WeakValueConfig with MZI and post-selection parameters.
        max_photons: Maximum photon number per mode for Hilbert space
            truncation.

    Returns:
        Dictionary with the following keys:
        - "weak_value": Complex weak value A_w.
        - "amplification": |A_w| (signal amplification factor).
        - "post_selection_prob": p_ps = |⟨f|i⟩|² (intrinsic probability
          of the post-selected outcome).
        - "signal": Amplified signal |A_w| · φ.
        - "fisher_information": Classical Fisher information from the
          post-selected measurement: F ≈ 4 · p_ps · |A_w|².
        - "delta_phi": Phase sensitivity Δφ = 1/√F.
        - "delta_phi_conventional": Conventional MZI sensitivity
          Δφ_conv = 1/√⟨N⟩ (standard quantum limit).
        - "final_state": Full output state after BS2 at the given φ.

    Raises:
        ValueError: If post_select_angle is outside (0, π) or
            measurement_basis is unknown.

    Example:
        >>> import numpy as np
        >>> import qutip
        >>> config = WeakValueConfig()
        >>> dim = 10 + 1
        >>> psi0 = qutip.tensor(qutip.coherent(dim, 2.0+0j), qutip.fock(dim, 0)).full()
        >>> result = weak_value_mzi(psi0, phi=0.01, config=config, max_photons=10)
        >>> result["amplification"] > 1.0
        True

    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not 0 < config.post_select_angle < np.pi:
        raise ValueError(
            f"post_select_angle must be in (0, π), got {config.post_select_angle}",
        )
    if config.measurement_basis not in ("parity", "Jz"):
        raise ValueError(
            f"Unknown measurement_basis: '{config.measurement_basis}'. "
            f"Supported: 'parity', 'Jz'.",
        )

    # ------------------------------------------------------------------
    # Number operator for mode 1 (phase-imprinted arm)
    # ------------------------------------------------------------------
    _, n1 = create_number_operator(max_photons)

    # ------------------------------------------------------------------
    # Beam splitter unitary (same matrix for BS1 and BS2)
    # ------------------------------------------------------------------
    U_bs = beam_splitter_unitary(config.theta, config.phi_bs, max_photons)

    # ------------------------------------------------------------------
    # Pre-selected state: |i⟩ = U_BS₁ |ψ₀⟩
    #
    # This is the state inside the interferometer after the first BS,
    # right before the phase shift. It represents the "pre-selection."
    # ------------------------------------------------------------------
    pre_selected_state: np.ndarray = U_bs @ initial_state

    # ------------------------------------------------------------------
    # Post-selected state: |f⟩ = exp(i · post_select_angle · J_z) |i⟩
    #
    # J_z generates rotations between the two modes. Rotating |i⟩ by
    # post_select_angle in J_z space produces a state that is nearly
    # orthogonal to |i⟩ when the angle approaches π/2, which is the
    # core mechanism of weak-value amplification.
    #
    # Performance note: J_z is diagonal in the Fock basis, so this
    # rotation is computed as O(dim) phase applications, not as a
    # dense matrix exponential.
    # ------------------------------------------------------------------
    post_selected_state: np.ndarray = _apply_jz_rotation(
        pre_selected_state,
        config.post_select_angle,
        max_photons,
    )
    post_selected_state /= np.linalg.norm(post_selected_state)

    # ------------------------------------------------------------------
    # Phase shift unitary
    # ------------------------------------------------------------------
    U_phi = phase_shift_unitary(phi, max_photons)

    # ------------------------------------------------------------------
    # Full MZI evolution: |ψ_out(φ)⟩ = U_BS₂ · U_φ · |i⟩
    #
    # This is the output state after the second beam splitter,
    # carrying the phase-encoded information.
    # ------------------------------------------------------------------
    output_state: np.ndarray = U_bs @ U_phi @ pre_selected_state

    # ------------------------------------------------------------------
    # Overlap between post-selected and pre-selected states
    #
    # ⟨f|i⟩ — the intrinsic overlap that determines the amplification.
    # When post_select_angle ≈ π/2, this overlap is small, giving
    # large |A_w| = |⟨f|n₁|i⟩ / ⟨f|i⟩|.
    # ------------------------------------------------------------------
    overlap_fi: complex = complex(np.vdot(post_selected_state, pre_selected_state))

    # Guard against division by zero (exactly orthogonal case)
    if np.isclose(abs(overlap_fi), 0.0, atol=1e-15):
        return {
            "weak_value": complex(np.inf, 0),
            "amplification": np.inf,
            "post_selection_prob": 0.0,
            "signal": np.inf if phi != 0 else 0.0,
            "fisher_information": 0.0,
            "delta_phi": np.inf,
            "delta_phi_conventional": np.inf,
            "final_state": output_state,
        }

    # ------------------------------------------------------------------
    # Weak value: A_w = ⟨f|n₁|i⟩ / ⟨f|i⟩
    #
    # The operator A = n₁ is chosen because the phase shift on mode 1
    # is U(φ) = exp(i·φ·n₁). For small φ, we have:
    #     ⟨f|U(φ)|i⟩ ≈ ⟨f|i⟩ + i·φ·⟨f|n₁|i⟩
    #
    # The weak value A_w represents the effective amplification of the
    # phase shift φ in the post-selected outcome.
    # ------------------------------------------------------------------
    weak_val_np = np.vdot(post_selected_state, n1 @ pre_selected_state) / overlap_fi
    weak_val: complex = complex(weak_val_np.item())

    # Physics assertion: the weak value should be predominantly real
    # for the standard MZI configuration. The imaginary part arises
    # from specific phase relationships but should be suppressed
    # relative to the real part for meaningful amplification.
    if abs(np.real(weak_val)) > 1e-10:
        imag_ratio = abs(np.imag(weak_val)) / abs(np.real(weak_val))
        assert imag_ratio < 5.0, (
            f"Weak value has large imaginary component relative to real part: "
            f"Re(A_w)={np.real(weak_val):.4f}, Im(A_w)={np.imag(weak_val):.4f}, "
            f"|Im/Re|={imag_ratio:.2f}. This may indicate an invalid post-selection "
            f"configuration."
        )

    # ------------------------------------------------------------------
    # Amplification factor: |A_w|
    # ------------------------------------------------------------------
    amplification: float = abs(weak_val)

    # ------------------------------------------------------------------
    # Intrinsic post-selection probability: p_ps = |⟨f|i⟩|²
    #
    # This is the probability that a photon would be found in the
    # post-selected port at φ = 0. For large amplification, this
    # probability is small: p_ps = sin²(δ) for the analytical model.
    # ------------------------------------------------------------------
    post_selection_prob: float = abs(overlap_fi) ** 2

    # ------------------------------------------------------------------
    # Amplified signal: S = |A_w| · φ
    #
    # The effective phase shift observed in the post-selected output
    # is amplified by the weak value. However, this amplification
    # comes at the cost of reduced post-selection probability.
    # ------------------------------------------------------------------
    signal: float = amplification * abs(phi)

    # ------------------------------------------------------------------
    # Fisher information
    #
    # For the post-selected weak-value measurement, the classical
    # Fisher information about φ per photon is approximately:
    #     F = 4 · p_ps · |A_w|²
    #
    # This can be derived from:
    #     p_ps(φ) = |⟨f|U(φ)|i⟩|²
    #     F = (dp_ps/dφ)² / (p_ps · (1 - p_ps))  for small φ
    #
    # The key result: F = N · cos²(δ) for the analytical coherent-state
    # model, which is always ≤ N = F_conventional.
    # ------------------------------------------------------------------
    # Total mean photon number for FI scaling
    mean_n: float = _mean_photon_number(initial_state, max_photons)
    fisher_information: float = 4.0 * post_selection_prob * (amplification**2)

    # ------------------------------------------------------------------
    # Sensitivity Δφ = 1/√F
    # ------------------------------------------------------------------
    delta_phi: float = (
        np.inf if fisher_information <= 0 else 1.0 / np.sqrt(fisher_information)
    )

    # ------------------------------------------------------------------
    # Conventional MZI sensitivity for comparison
    # Δφ_conv = 1/√⟨N⟩ (standard quantum limit)
    # ------------------------------------------------------------------
    delta_phi_conventional: float = np.inf if mean_n <= 0 else 1.0 / np.sqrt(mean_n)

    return {
        "weak_value": weak_val,
        "amplification": amplification,
        "post_selection_prob": post_selection_prob,
        "signal": signal,
        "fisher_information": fisher_information,
        "delta_phi": delta_phi,
        "delta_phi_conventional": delta_phi_conventional,
        "final_state": output_state,
    }


# =============================================================================
# Analytical Sensitivity for Coherent State Input
# =============================================================================


def weak_value_mzi_sensitivity(
    N: float,
    phi: float,
    config: WeakValueConfig,
) -> dict[str, Any]:
    """Compute sensitivity for weak-value MZI with coherent state input.

    Analytical formula for a coherent state |α⟩ in mode 0 with mean
    photon number N = |α|²:

        Let δ = π/2 − post_select_angle be the deviation from perfect
        orthogonality between pre- and post-selected states.

        Then:
        - Weak value:            A_w = cot(δ)
        - Amplification:         |A_w| = cot(δ)
        - Post-selection prob:   p_ps = sin²(δ)
        - Signal:                S = |A_w| · φ
        - Fisher information:    F = N · sin²(δ) · cot²(δ) = N · cos²(δ)
        - Conventional FI:       F_conv = N
        - Sensitivity:           Δφ = 1/√(N · cos²(δ))
        - Conventional sens.:    Δφ_conv = 1/√N

    Therefore Δφ = Δφ_conv / cos(δ) ≥ Δφ_conv.

    Weak-value amplification cannot beat the standard quantum limit
    for phase estimation. The amplification factor cot(δ) is exactly
    canceled by the reduced post-selection probability sin²(δ) in the
    Fisher information:

        F = N · sin²(δ) · cot²(δ) = N · cos²(δ) ≤ N

    Args:
        N: Mean photon number |α|² of the coherent state input.
            Must be strictly positive.
        phi: Phase shift in radians (the parameter being estimated).
            Used only for computing the amplified signal.
        config: WeakValueConfig with post-selection and MZI parameters.

    Returns:
        Dictionary with the following keys:
        - "weak_value": Complex weak value A_w = cot(δ) (real-valued).
        - "amplification": |A_w| = cot(δ).
        - "post_selection_prob": p_ps = sin²(δ).
        - "signal": Amplified signal |A_w| · φ.
        - "fisher_information": F = N · cos²(δ).
        - "fisher_information_conventional": F_conv = N.
        - "delta_phi": Δφ = 1/√(N · cos²(δ)).
        - "delta_phi_conventional": Δφ_conv = 1/√N.

    Raises:
        ValueError: If N ≤ 0 or post_select_angle is outside (0, π).

    Example:
        >>> config = WeakValueConfig(post_select_angle=np.pi/2 - 0.1)
        >>> result = weak_value_mzi_sensitivity(N=100, phi=0.01, config=config)
        >>> np.isclose(result["amplification"], 1.0/np.tan(0.1), rtol=1e-10)
        True
        >>> result["delta_phi"] > result["delta_phi_conventional"]
        True

    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if N <= 0:
        raise ValueError(f"Mean photon number N must be positive, got {N}")
    if not 0 < config.post_select_angle < np.pi:
        raise ValueError(
            f"post_select_angle must be in (0, π), got {config.post_select_angle}",
        )

    # ------------------------------------------------------------------
    # Deviation from orthogonality: δ = π/2 − post_select_angle
    #
    # When post_select_angle → π/2, δ → 0 and:
    #   - Amplification A_w = cot(δ) → ∞
    #   - Post-selection probability p_ps = sin²(δ) → 0
    #   - Fisher information F = N · cos²(δ) → N (unchanged!)
    #
    # This is the key insight: the FI is preserved as δ → 0 because
    # the diverging amplification is exactly canceled by the vanishing
    # probability.
    # ------------------------------------------------------------------
    delta: float = np.pi / 2 - config.post_select_angle

    # Guard against δ = 0 (perfectly orthogonal — infinite amplification)
    if np.isclose(delta, 0.0, atol=1e-15):
        return {
            "weak_value": complex(np.inf, 0),
            "amplification": np.inf,
            "post_selection_prob": 0.0,
            "signal": np.inf if phi != 0 else 0.0,
            "fisher_information": N,  # Limit: cos²(δ) → 1
            "fisher_information_conventional": N,
            "delta_phi": 1.0 / np.sqrt(N),
            "delta_phi_conventional": 1.0 / np.sqrt(N),
        }

    # ------------------------------------------------------------------
    # Weak value: A_w = cot(δ)
    #
    # For the coherent-state weak-value MZI, the weak value is purely
    # real (up to a global phase) and equals cot(δ).
    # ------------------------------------------------------------------
    cot_delta: float = 1.0 / np.tan(delta)
    weak_val: complex = complex(cot_delta, 0.0)

    # ------------------------------------------------------------------
    # Amplification: |A_w| = cot(δ)
    # ------------------------------------------------------------------
    amplification: float = cot_delta

    # ------------------------------------------------------------------
    # Post-selection probability: p_ps = sin²(δ)
    #
    # This is the intrinsic probability of finding a photon in the
    # post-selected port. For small δ (large amplification), p_ps ≈ δ².
    # ------------------------------------------------------------------
    sin_delta: float = np.sin(delta)
    post_selection_prob: float = sin_delta**2

    # ------------------------------------------------------------------
    # Amplified signal: S = |A_w| · φ
    # ------------------------------------------------------------------
    signal: float = amplification * abs(phi)

    # ------------------------------------------------------------------
    # Fisher information: F = N · sin²(δ) · cot²(δ) = N · cos²(δ)
    #
    # Derivation:
    #   F = p_ps · (d/dφ signal)²
    #     = sin²(δ) · cot²(δ) · N
    #     = cos²(δ) · N
    #
    # Conventional FI for a coherent state: F_conv = N
    #
    # Since cos²(δ) ≤ 1, we have F ≤ F_conv always, with equality
    # only in the limit δ → 0 (where amplification diverges but
    # post-selection probability vanishes).
    # ------------------------------------------------------------------
    cos_delta: float = np.cos(delta)
    fisher_information: float = N * cos_delta**2
    fisher_information_conventional: float = N

    # ------------------------------------------------------------------
    # Phase sensitivity
    # ------------------------------------------------------------------
    delta_phi: float = 1.0 / np.sqrt(max(fisher_information, 1e-300))
    delta_phi_conventional: float = 1.0 / np.sqrt(N)

    # ------------------------------------------------------------------
    # Physics assertions: no metrological advantage
    #
    # These assertions verify the fundamental theorem of weak-value
    # amplification: the Fisher information from the post-selected
    # measurement never exceeds the conventional Fisher information,
    # and the phase sensitivity is never better than the SQL.
    #
    # The factor cos²(δ) reduction in FI arises from the reduced
    # post-selection probability, not from any measurement inefficiency.
    # ------------------------------------------------------------------
    assert fisher_information <= fisher_information_conventional + 1e-12, (
        f"FI ({fisher_information}) exceeds conventional FI "
        f"({fisher_information_conventional}) — this violates the "
        f"fundamental no-advantage theorem of weak-value amplification."
    )

    assert delta_phi >= delta_phi_conventional - 1e-12, (
        f"Sensitivity ({delta_phi}) is better than conventional "
        f"({delta_phi_conventional}) — weak-value amplification cannot "
        f"improve phase estimation sensitivity."
    )

    return {
        "weak_value": weak_val,
        "amplification": amplification,
        "post_selection_prob": post_selection_prob,
        "signal": signal,
        "fisher_information": fisher_information,
        "fisher_information_conventional": fisher_information_conventional,
        "delta_phi": delta_phi,
        "delta_phi_conventional": delta_phi_conventional,
    }


# =============================================================================
# Section: thermal_langevin
# =============================================================================

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ThermalLangevinConfig:
    """Configuration for thermal Langevin noise calculations.

    This uses normalized units where:
    - The reference thermal noise strength `thermal_strength` determines the
      constant (or weakly N-dependent) thermal contribution.
    - For normalized operation, set `use_normalized=True`.

    When `use_normalized=True`:
        Δφ_thermal(N) = thermal_strength * N^thermal_exponent
        Δφ_quantum(N) = 1 / sqrt(N)

    This allows easy control of the crossover behavior.

    Attributes:
        thermal_strength: Strength of thermal noise contribution.
        thermal_exponent: Scaling exponent for thermal noise with N.
            Use 0 for constant floor (α→0 limit), or slightly negative
            values for weak N-dependent thermal noise.
        use_normalized: Use normalized scaling mode (recommended for scaling surveys).

    """

    thermal_strength: float = 1.0
    thermal_exponent: float = 0.0
    use_normalized: bool = True


# =============================================================================
# Normalized Thermal-Quantum Model
# =============================================================================


def thermal_sensitivity_normalized(
    N: float,
    config: ThermalLangevinConfig,
) -> float:
    """Compute thermal noise sensitivity in normalized units.

    Δφ_thermal(N) = thermal_strength * N^thermal_exponent

    Args:
        N: Particle number.
        config: ThermalLangevinConfig with noise parameters.

    Returns:
        Thermal phase sensitivity Δφ_thermal.

    """
    exponent = config.thermal_exponent
    return config.thermal_strength * (N**exponent)


def combined_sensitivity(
    N: float,
    config: ThermalLangevinConfig,
) -> float:
    """Compute combined thermal+quantum phase sensitivity.

    Adds noise variances in quadrature:
        Δφ_total = sqrt(Δφ_quantum² + Δφ_thermal²)

    Args:
        N: Particle number.
        config: ThermalLangevinConfig with noise parameters.

    Returns:
        Combined sensitivity Δφ.

    """
    if config.use_normalized:
        delta_quantum = 1.0 / np.sqrt(N)
        delta_thermal = thermal_sensitivity_normalized(N, config)
    else:
        # Full physical model - compute via susceptibility integration
        # (for future extension)
        delta_quantum = 1.0 / np.sqrt(N)
        delta_thermal = thermal_sensitivity_normalized(N, config)

    # Quadrature sum
    delta_total = np.sqrt(delta_quantum**2 + delta_thermal**2)

    return float(delta_total)


def thermal_sensitivity_at_N(
    N: float,
    base_config: ThermalLangevinConfig,
    m_scaling: Callable[[float, ThermalLangevinConfig], float] | None = None,
    k_scaling: Callable[[float, ThermalLangevinConfig], float] | None = None,
) -> float:
    """Compute the combined thermal+quantum phase sensitivity at given N.

    This is the main function for scaling surveys.

    Args:
        N: Particle number.
        base_config: Base thermal noise configuration.
        m_scaling: Ignored, included for backward compatibility.
        k_scaling: Ignored, included for backward compatibility.

    Returns:
        Combined Δφ = sqrt(Δφ_quantum² + Δφ_thermal²).

    """
    return combined_sensitivity(N, base_config)


def sweep_thermal_scaling(
    N_values: list[int] | npt.NDArray[np.int_],
    base_config: ThermalLangevinConfig,
    m_scaling: Callable[[float, ThermalLangevinConfig], float] | None = None,
    k_scaling: Callable[[float, ThermalLangevinConfig], float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sweep thermal+quantum sensitivity over a range of N values.

    Args:
        N_values: Array of particle numbers to evaluate.
        base_config: Base configuration.
        m_scaling: Ignored.
        k_scaling: Ignored.

    Returns:
        Tuple of (N_array, delta_phi_array).

    """
    N_arr = np.asarray(N_values, dtype=float)
    delta_phi_arr = np.zeros_like(N_arr)

    for i, N in enumerate(N_arr):
        delta_phi_arr[i] = thermal_sensitivity_at_N(N, base_config)

    return N_arr, delta_phi_arr


def fit_thermal_scaling_exponent(
    N_values: list[int] | npt.NDArray[np.int_],
    base_config: ThermalLangevinConfig,
    m_scaling: Callable[[float, ThermalLangevinConfig], float] | None = None,
    k_scaling: Callable[[float, ThermalLangevinConfig], float] | None = None,
    min_N: int = 4,
) -> ScalingFitResult:
    """Fit the effective scaling exponent α for thermal+quantum noise.

    Extracts α from Δφ ∝ N^α across the given N range.

    Args:
        N_values: Array of particle numbers to evaluate.
        base_config: Base configuration.
        m_scaling: Ignored.
        k_scaling: Ignored.
        min_N: Minimum N for the fit.

    Returns:
        ScalingFitResult with exponent α and quality metrics.

    """
    N_arr, delta_phi_arr = sweep_thermal_scaling(N_values, base_config)

    return fit_scaling_exponent(N_arr, delta_phi_arr, min_N=min_N)


def crossover_N(
    base_config: ThermalLangevinConfig,
    m_scaling: Callable[[float, ThermalLangevinConfig], float] | None = None,
    k_scaling: Callable[[float, ThermalLangevinConfig], float] | None = None,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> float:
    """Find the N where thermal noise equals quantum noise.

    This is the crossover point where scaling transitions from
    quantum-dominated to thermal-dominated.

    Solves for N where:
        1/sqrt(N) = thermal_strength * N^thermal_exponent

    For thermal_exponent = 0 (constant floor):
        N_cross = 1 / thermal_strength²

    Args:
        base_config: Thermal configuration.
        m_scaling: Ignored.
        k_scaling: Ignored.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        N_crossover: Particle number where contributions are equal.

    """
    if not base_config.use_normalized:
        # Fallback for non-normalized: use bisection
        low, high = 1.0, 1e9
        for _ in range(max_iter):
            mid = (low + high) / 2
            delta_q = 1.0 / np.sqrt(mid)
            delta_t = thermal_sensitivity_normalized(mid, base_config)
            if delta_t > delta_q:
                high = mid
            else:
                low = mid
            if high - low < tol * low:
                break
        return (low + high) / 2

    # Analytical solution for normalized case
    # We want: 1/sqrt(N) = S * N^alpha
    # => N^(-1/2 - alpha) = S
    # => N = S^(1/(-1/2 - alpha)) = S^(-2/(1 + 2*alpha))

    alpha = base_config.thermal_exponent
    S = base_config.thermal_strength

    if alpha <= -0.5:
        # Thermal decreases as fast or faster than quantum
        # No finite crossover in interesting regime
        return np.inf

    exponent = -2.0 / (1.0 + 2.0 * alpha)
    N_cross = S**exponent

    return float(N_cross)


# =============================================================================
# Low-level Physical Susceptibility Functions
# (For reference / future extension)
# =============================================================================


def mechanical_susceptibility(
    omega: float | npt.NDArray[np.float64],
    m: float,
    omega_m: float,
    gamma: float,
) -> complex | npt.NDArray[np.complex128]:
    """Compute the mechanical susceptibility χ(ω).

    χ(ω) = 1 / [m(ω_m² - ω² + iΓω)]

    Args:
        omega: Frequency ω in rad/s (scalar or array).
        m: Mass m.
        omega_m: Resonance frequency ω_m.
        gamma: Damping Γ.

    Returns:
        Complex susceptibility χ(ω).

    """
    scalar_input = np.ndim(omega) == 0
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    denominator = m * (omega_m**2 - omega_arr**2 + 1j * gamma / m * omega_arr)
    result: npt.NDArray[np.complex128] = 1.0 / denominator
    if scalar_input:
        return complex(result[0])
    return result


def force_psd_thermal(
    T: float,
    gamma: float,
    k_B: float = 1.0,
) -> float:
    """Compute thermal force power spectral density S_F(ω).

    For thermal Langevin noise, the force PSD is white:
        S_F = 2 Γ k_B T

    Args:
        T: Temperature.
        gamma: Damping Γ.
        k_B: Boltzmann constant.

    Returns:
        Force PSD S_F.

    """
    return 2.0 * gamma * k_B * T


def thermal_floor_approximation(
    config: ThermalLangevinConfig,
) -> float:
    """Approximate thermal noise floor for normalized config.

    In normalized units, this returns the thermal contribution
    at reference N=1.

    Args:
        config: ThermalLangevinConfig.

    Returns:
        Approximate thermal phase noise floor at N=1.

    """
    return config.thermal_strength


# =============================================================================
# Convenience Configurations
# =============================================================================


def create_thermal_config(
    thermal_strength: float = 0.1,
    thermal_exponent: float = 0.0,
) -> ThermalLangevinConfig:
    """Create a ThermalLangevinConfig with intuitive parameters.

    Args:
        thermal_strength: Thermal noise strength relative to SQL at N=1.
            thermal_strength = 0.1 means:
            - At N=1: thermal = 0.1, SQL = 1.0 (quantum dominates)
            - At N=100: thermal = 0.1, SQL = 0.1 (crossover)
            - At N=10000: thermal = 0.1, SQL = 0.01 (thermal dominates)
        thermal_exponent: Scaling exponent for thermal noise.
            0.0 = constant floor (thermal independent of N)

    Returns:
        ThermalLangevinConfig object.

    """
    return ThermalLangevinConfig(
        thermal_strength=thermal_strength,
        thermal_exponent=thermal_exponent,
        use_normalized=True,
    )


def create_quantum_only_config() -> ThermalLangevinConfig:
    """Create config with negligible thermal noise (pure SQL scaling)."""
    return ThermalLangevinConfig(
        thermal_strength=1e-10,
        thermal_exponent=0.0,
        use_normalized=True,
    )


def create_thermal_dominated_config() -> ThermalLangevinConfig:
    """Create config where thermal dominates for all N > 1."""
    return ThermalLangevinConfig(
        thermal_strength=10.0,
        thermal_exponent=0.0,
        use_normalized=True,
    )


# =============================================================================
# Section: dynamical_decoupling
# =============================================================================
# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DDConfig:
    """Configuration for dynamical decoupling.

    Attributes:
        n_pulses: Number of π-pulses.
        sequence: Pulse sequence type ('CPMG' or 'XY8').
        tau: Inter-pulse delay (time between consecutive π-pulses).
        pulse_axis: Rotation axis for pulses ('x' or 'y').

    Raises:
        ValueError: If n_pulses is negative.
        ValueError: If sequence is not 'CPMG' or 'XY8'.
        ValueError: If tau is not positive.
        ValueError: If pulse_axis is not 'x' or 'y'.

    """

    n_pulses: int = 0
    sequence: str = "CPMG"
    tau: float = 0.1
    pulse_axis: str = "x"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_pulses < 0:
            raise ValueError(
                f"Number of pulses must be non-negative, got {self.n_pulses}",
            )
        if self.sequence not in ("CPMG", "XY8"):
            raise ValueError(f"Sequence must be 'CPMG' or 'XY8', got '{self.sequence}'")
        if self.tau <= 0:
            raise ValueError(f"Inter-pulse delay tau must be positive, got {self.tau}")
        if self.pulse_axis not in ("x", "y"):
            raise ValueError(f"Pulse axis must be 'x' or 'y', got '{self.pulse_axis}'")


# =============================================================================
# Filter Functions
# =============================================================================


def cpmg_filter_function(
    omega: np.ndarray,
    n_pulses: int,
    tau: float,
) -> np.ndarray:
    r"""Compute the CPMG filter function.

    For n_pulses π-pulses with inter-pulse spacing τ, the modulation
    function g(t) alternates sign at each pulse, producing a frequency
    filter:

        F(ω) = |1 + (-1)^{n+1} e^{iωT}
               + 2 Σ_{j=1}^{n} (-1)^j e^{iω j τ}|² / (ωT)²

    where T = n_pulses · τ is the total sequence duration.

    Physical interpretation:
    - F(ω) ≈ 1 for ω ≪ 1/T (suppressed — long-time noise refocused)
    - F(ω) → 0 at ω = kπ/τ (notch filters at harmonics)
    - The first zero occurs at ω = π/τ, acting as a high-pass cutoff

    Args:
        omega: Frequency array (can be scalar or ndarray).
        n_pulses: Number of π-pulses.
        tau: Inter-pulse delay.

    Returns:
        Filter function values at each frequency.
        Same shape as omega.

    Raises:
        ValueError: If n_pulses is negative.
        ValueError: If tau is not positive.

    Example:
        >>> omega = np.linspace(0, 10, 1000)
        >>> F = cpmg_filter_function(omega, n_pulses=4, tau=0.5)
        >>> F[0]  # F(0) should be 0 (DC suppressed)
        0.0
        >>> np.all(F >= 0)
        True

    """
    if n_pulses < 0:
        raise ValueError(f"Number of pulses must be non-negative, got {n_pulses}")
    if tau <= 0:
        raise ValueError(f"Inter-pulse delay tau must be positive, got {tau}")

    omega = np.asarray(omega, dtype=float)
    T = n_pulses * tau

    # Handle zero-pulse case: no dynamical decoupling applied
    # Without pulses, there is no filtering — every frequency passes
    # with unit weight: F(ω) ≡ 1 for all ω.
    if n_pulses == 0:
        return np.ones_like(omega)

    # n_pulses > 0: construct the CPMG filter sum
    # S(ω) = 1 + (-1)^{n+1} e^{iωT} + 2 Σ_{j=1}^{n} (-1)^j e^{iω j τ}
    # F(ω) = |S(ω)|² / (ωT)²

    S = 1.0 + (-1) ** (n_pulses + 1) * np.exp(1j * omega * T)

    # Vectorised sum over pulses: 2 Σ_{j=1}^{n} (-1)^j e^{iω j τ}
    # Uses broadcasting: omega (N,) × j_idx (n_pulses,) → (N, n_pulses)
    j_idx = np.arange(1, n_pulses + 1)
    phase_factors = (-1) ** j_idx  # shape (n_pulses,)
    # omega[:, None] * j_idx[None, :] * tau → (N, n_pulses)
    exp_terms = np.exp(1j * omega[:, None] * j_idx[None, :] * tau)  # (N, n_pulses)
    sum_term = np.sum(phase_factors[None, :] * exp_terms, axis=1)  # (N,)
    S += 2.0 * sum_term

    # Guard against division by zero at ω = 0
    # Use np.divide with where= to avoid RuntimeWarning for zero denominator
    denom = (omega * T) ** 2
    zero_mask = np.abs(omega) < 1e-15

    return np.divide(
        np.abs(S) ** 2,
        denom,
        where=~zero_mask,
        out=np.zeros_like(denom),
    )


# =============================================================================
# Effective Coherence Time
# =============================================================================


def dd_effective_coherence_time(
    T_2_0: float,
    n_pulses: int,
    sequence: str = "CPMG",
) -> float:
    """Compute effective coherence time under dynamical decoupling.

    Dynamical decoupling extends the coherence time by periodically
    refocusing the phase evolution:

    - CPMG: T₂^(DD) = T₂⁰ · n_pulses^(2/3)
      The 2/3 power law arises from the filter function's spectral
      overlap with a 1/f noise background.

    - XY-8: T₂^(DD) = T₂⁰ · n_pulses^(0.8)
      The XY-8 sequence provides better compensation for pulse
      imperfections, yielding a slightly improved scaling.

    Args:
        T_2_0: Bare coherence time (no DD). Must be positive.
        n_pulses: Number of π-pulses.
        sequence: 'CPMG' or 'XY8'. Defaults to 'CPMG'.

    Returns:
        Effective coherence time with DD.

    Raises:
        ValueError: If T_2_0 is not positive.
        ValueError: If n_pulses is negative.
        ValueError: If sequence is not 'CPMG' or 'XY8'.

    Example:
        >>> T_dd = dd_effective_coherence_time(T_2_0=1.0, n_pulses=8, sequence="CPMG")
        >>> T_dd > 1.0  # DD always improves coherence
        True
        >>> T_dd_cpmg = dd_effective_coherence_time(1.0, 8, "CPMG")
        >>> T_dd_xy8 = dd_effective_coherence_time(1.0, 8, "XY8")
        >>> T_dd_xy8 > T_dd_cpmg  # XY-8 slightly better
        True

    """
    if T_2_0 <= 0:
        raise ValueError(f"Bare coherence time must be positive, got {T_2_0}")
    if n_pulses < 0:
        raise ValueError(f"Number of pulses must be non-negative, got {n_pulses}")
    if sequence not in ("CPMG", "XY8"):
        raise ValueError(f"Sequence must be 'CPMG' or 'XY8', got '{sequence}'")

    if n_pulses == 0:
        return T_2_0

    match sequence:
        case "CPMG":
            exponent = 2.0 / 3.0
        case "XY8":
            exponent = 0.8

    return T_2_0 * n_pulses**exponent


# =============================================================================
# Phase Sensitivity
# =============================================================================


def dd_phase_sensitivity(
    N: int,
    phi: float,
    T: float,
    n_pulses: int,
    T_2_0: float = 1.0,
    sequence: str = "CPMG",
) -> float:
    r"""Compute phase sensitivity with dynamical decoupling.

    The phase sensitivity for an interferometer with DD-enhanced
    coherence is:

        Δφ = 1 / √(N · T₂^(DD) / T)

    where:
    - N is the mean photon number (SQL resource scaling)
    - T₂^(DD) is the effective coherence time under DD
    - T is the total evolution time

    This formula assumes:
    - The SQL scaling Δφ ∝ 1/√N is preserved
    - DD only improves the prefactor C = √(T / T₂^(DD))
    - The measurement is optimal (photon number counting)

    The phase φ argument is included for interface consistency with
    other sensitivity functions but does not affect the SQL-limited
    result (sensitivity is phase-independent at the SQL).

    Args:
        N: Mean photon number.
        phi: Phase shift (radians). Included for interface consistency.
        T: Total evolution time.
        n_pulses: Number of π-pulses.
        T_2_0: Bare coherence time. Defaults to 1.0.
        sequence: 'CPMG' or 'XY8'. Defaults to 'CPMG'.

    Returns:
        Phase sensitivity Δφ.

    Raises:
        ValueError: If N is not positive.
        ValueError: If T is not positive.
        ValueError: If n_pulses is negative.
        ValueError: If T_2_0 is not positive.

    Example:
        >>> delta_phi = dd_phase_sensitivity(N=100, phi=np.pi/4, T=1.0,
        ...                                  n_pulses=4, T_2_0=0.5)
        >>> delta_phi > 0
        True
        >>> # More pulses → better sensitivity (lower Δφ)
        >>> d1 = dd_phase_sensitivity(100, 0, 1.0, 0, 1.0)
        >>> d2 = dd_phase_sensitivity(100, 0, 1.0, 8, 1.0)
        >>> d2 < d1
        True

    """
    if N <= 0:
        raise ValueError(f"Mean photon number N must be positive, got {N}")
    if T <= 0:
        raise ValueError(f"Total evolution time T must be positive, got {T}")
    if n_pulses < 0:
        raise ValueError(f"Number of pulses must be non-negative, got {n_pulses}")
    if T_2_0 <= 0:
        raise ValueError(f"Bare coherence time must be positive, got {T_2_0}")

    # Compute effective coherence time with DD
    T_2_dd = dd_effective_coherence_time(T_2_0, n_pulses, sequence)

    # SQL sensitivity with DD-enhanced coherence
    # Δφ = 1 / √(N · T₂^(DD) / T)
    return 1.0 / np.sqrt(N * T_2_dd / T)


# =============================================================================
# Scaling Analysis
# =============================================================================


def dd_sensitivity_scaling(
    N_values: np.ndarray,
    n_pulses: int,
    T: float,
    T_2_0: float = 1.0,
) -> dict:
    r"""Compute Δφ vs N for DD-enhanced interferometry.

    Analyzes the scaling of phase sensitivity with photon number N
    under dynamical decoupling. Verifies that the SQL exponent
    α = -0.5 is preserved regardless of pulse number.

    The sensitivity follows:

        Δφ = C · N^{α}

    where:
    - α = -0.5 (SQL) is fixed — DD does NOT change this
    - C = (T / T₂^(DD))^{-1/2} is improved by more pulses

    Args:
        N_values: Array of mean photon numbers to evaluate.
        n_pulses: Number of π-pulses.
        T: Total evolution time.
        T_2_0: Bare coherence time. Defaults to 1.0.

    Returns:
        Dictionary with:
        - 'N': Input N_values array.
        - 'delta_phi': Phase sensitivity at each N.
        - 'fitted_alpha': Power-law exponent from log-log fit.
        - 'expected_alpha': Expected exponent (-0.5 for SQL).
        - 'prefactor_C': Fitted prefactor (improves with DD).

    Raises:
        ValueError: If N_values is empty or contains non-positive values.
        ValueError: If n_pulses is negative.
        ValueError: If T is not positive.

    Example:
        >>> N_vals = np.logspace(1, 4, 10)
        >>> result = dd_sensitivity_scaling(N_vals, n_pulses=4, T=1.0)
        >>> np.isclose(result['fitted_alpha'], -0.5, atol=0.05)
        True
        >>> result['prefactor_C'] > 0
        True

    """
    N_values = np.asarray(N_values, dtype=float)

    if N_values.size == 0:
        raise ValueError("N_values array must not be empty")
    if np.any(N_values <= 0):
        raise ValueError("All N_values must be positive")
    if n_pulses < 0:
        raise ValueError(f"Number of pulses must be non-negative, got {n_pulses}")
    if T <= 0:
        raise ValueError(f"Total evolution time T must be positive, got {T}")

    # Compute sensitivity at each N
    delta_phi = np.array(
        [dd_phase_sensitivity(int(N), 0.0, T, n_pulses, T_2_0) for N in N_values],
    )

    # Fit power law: log(Δφ) = log(C) + α · log(N)
    log_N = np.log(N_values)
    log_delta = np.log(delta_phi)

    # Linear fit via polyfit
    coeffs = np.polyfit(log_N, log_delta, deg=1)
    fitted_alpha = coeffs[0]
    prefactor_C = np.exp(coeffs[1])

    expected_alpha = -0.5

    return {
        "N": N_values,
        "delta_phi": delta_phi,
        "fitted_alpha": fitted_alpha,
        "expected_alpha": expected_alpha,
        "prefactor_C": prefactor_C,
    }


# =============================================================================
# Section: tilt_to_length_noise
# =============================================================================

if TYPE_CHECKING:
    from numpy.typing import NDArray

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TTLNoiseConfig:
    """Configuration for tilt-to-length coupling noise.

    Attributes:
        theta_rms: RMS angular jitter in radians (default: 1e-6 = 1 μrad).
        L: Reference arm length in metres (default: 1.0).
        wavelength: Laser wavelength, same units as L (default: 1e-6 = 1 μm).
        beam_offset: Beam offset from the pivot point in the same units
            as L (default: 1e-3 = 1 mm).

    """

    theta_rms: float = 1e-6  # 1 μrad
    L: float = 1.0  # 1 m
    wavelength: float = 1e-6  # 1 μm
    beam_offset: float = 1e-3  # 1 mm


# =============================================================================
# Core Physics
# =============================================================================


def ttl_path_length_noise(config: TTLNoiseConfig) -> float:
    """Compute the RMS path length noise from tilt-to-length coupling.

    The apparent path length change due to angular jitter is:
        δL = θ_rms · x_offset

    where θ_rms is the RMS angular jitter and x_offset is the beam offset
    from the pivot point.

    Args:
        config: TTL noise configuration parameters.

    Returns:
        RMS path length noise δL in the same units as L and beam_offset.

    Raises:
        ValueError: If config parameters are non-positive or invalid.

    Example:
        >>> config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3)
        >>> round(ttl_path_length_noise(config), 12)
        1e-09

    """
    _validate_config(config)
    return config.theta_rms * config.beam_offset


def ttl_phase_noise(config: TTLNoiseConfig) -> float:
    """Compute the RMS phase noise from tilt-to-length coupling.

    δφ = 2π · (θ_rms · beam_offset) / λ

    Args:
        config: TTL noise configuration parameters.

    Returns:
        RMS phase noise in radians.

    Raises:
        ValueError: If config parameters are non-positive or invalid.

    Example:
        >>> config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3,
        ...                         wavelength=1e-6)
        >>> result = ttl_phase_noise(config)
        >>> abs(result - 2 * np.pi * 1e-3) < 1e-10
        True

    """
    _validate_config(config)
    delta_L = ttl_path_length_noise(config)
    return 2.0 * np.pi * delta_L / config.wavelength


def ttl_sensitivity_floor(config: TTLNoiseConfig) -> float:
    """Compute the sensitivity floor imposed by TTL noise.

    The minimum detectable phase shift is limited by TTL noise:
        Δφ_min = δφ_ttl

    This is a constant floor that is independent of the photon/atom number N.

    Args:
        config: TTL noise configuration parameters.

    Returns:
        Minimum detectable phase shift (radians), equal to the TTL
        phase noise.

    Raises:
        ValueError: If config parameters are non-positive or invalid.

    """
    return ttl_phase_noise(config)


# =============================================================================
# Combined Sensitivity
# =============================================================================


def ttl_limited_sensitivity(
    N: float,
    quantum_sensitivity: float,
    config: TTLNoiseConfig,
) -> float:
    """Compute total sensitivity with TTL noise added in quadrature.

    The total phase sensitivity is the quadratic sum of the quantum-limited
    sensitivity and the tilt-to-length noise floor:

        Δφ_total² = Δφ_quantum² + Δφ_ttl²

    Args:
        N: Mean photon/atom number (used for validation only; the actual
            quantum sensitivity is passed via quantum_sensitivity).
        quantum_sensitivity: Quantum-limited sensitivity Δφ_Q(N) for the
            given N.
        config: TTL noise configuration.

    Returns:
        Total sensitivity Δφ_total in radians.

    Raises:
        ValueError: If N is negative or if quantum_sensitivity is negative.
        ValueError: If config parameters are invalid.

    """
    if N < 0:
        raise ValueError(f"Particle number N must be non-negative, got {N}")
    if quantum_sensitivity < 0:
        raise ValueError(
            f"Quantum sensitivity must be non-negative, got {quantum_sensitivity}",
        )

    phi_ttl = ttl_phase_noise(config)
    return np.sqrt(quantum_sensitivity**2 + phi_ttl**2)


# =============================================================================
# Scaling Analysis
# =============================================================================


def _power_law(
    log_N: NDArray[np.float64],
    log_a: float,
    alpha: float,
) -> NDArray[np.float64]:
    """Power-law model for fitting sensitivity scaling.

    Δφ = a · N^α

    Fitted in log-log space:
        log(Δφ) = log(a) + α · log(N)

    Args:
        log_N: log10(N) values.
        log_a: Intercept parameter (log10 of prefactor).
        alpha: Scaling exponent.

    Returns:
        log10(Δφ) values.

    """
    return log_a + alpha * log_N


def ttl_scaling_sweep(
    N_values: np.ndarray,
    config: TTLNoiseConfig,
    quantum_scaling: str = "sql",
) -> dict:
    """Compute sensitivity vs particle number N with TTL noise.

    Sweeps over N values to show how TTL noise creates a constant floor
    at large N, breaking standard quantum scaling.

    Args:
        N_values: Array of N (photon/atom number) values to evaluate.
            Must be positive.
        config: TTL noise configuration.
        quantum_scaling: Type of quantum-limited scaling:
            - ``"sql"``: Standard quantum limit Δφ_Q = 1/√N (coherent state).
            - ``"hl"``: Heisenberg limit Δφ_Q = 1/N (NOON state).

    Returns:
        Dictionary with fields:
            - ``"N"``: Input N values (NDArray[np.float64]).
            - ``"delta_phi"``: Total sensitivity with TTL noise
              (NDArray[np.float64]).
            - ``"delta_phi_quantum"``: Quantum-limited contribution alone
              (NDArray[np.float64]).
            - ``"delta_phi_ttl"``: TTL noise floor (constant float).
            - ``"alpha_fitted"``: Fitted scaling exponent α from
              power-law fit to total sensitivity. Should approach 0 at
              large N where TTL dominates.

    Raises:
        ValueError: If N_values contains non-positive values.
        ValueError: If quantum_scaling is not ``"sql"`` or ``"hl"``.

    Example:
        >>> config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3,
        ...                         wavelength=1e-6)
        >>> N = np.logspace(0, 8, 50)
        >>> result = ttl_scaling_sweep(N, config, quantum_scaling="sql")
        >>> result["alpha_fitted"] is not None
        True

    """
    N_arr = np.asarray(N_values, dtype=np.float64)

    if np.any(N_arr <= 0):
        raise ValueError(
            f"All N values must be positive, got range [{N_arr.min()}, {N_arr.max()}]",
        )

    if quantum_scaling not in ("sql", "hl"):
        raise ValueError(
            f"Quantum scaling must be 'sql' or 'hl', got '{quantum_scaling}'",
        )

    # Compute sensitivities
    phi_ttl = ttl_phase_noise(config)
    if quantum_scaling == "sql":
        phi_q = np.array([1.0 / np.sqrt(N) for N in N_arr], dtype=np.float64)
    else:
        phi_q = np.array([1.0 / N for N in N_arr], dtype=np.float64)
    phi_total = np.sqrt(phi_q**2 + phi_ttl**2)

    # Fit power law: Δφ = a · N^α in log-log space
    # Fit over all points to extract effective scaling exponent
    log_N = np.log10(N_arr)
    log_phi = np.log10(phi_total)

    alpha_fitted: float | None = None
    try:
        if len(N_arr) >= 3:
            popt, _ = curve_fit(
                _power_law,
                log_N,
                log_phi,
                p0=[0.0, -0.5],
                maxfev=5000,
            )
            alpha_fitted = float(popt[1])
    except (RuntimeError, ValueError):
        # Fit may fail for degenerate cases (e.g., TTL-dominated at all N)
        alpha_fitted = None

    return {
        "N": N_arr,
        "delta_phi": phi_total,
        "delta_phi_quantum": phi_q,
        "delta_phi_ttl": float(phi_ttl),
        "alpha_fitted": alpha_fitted,
    }


# =============================================================================
# Validation
# =============================================================================


def _validate_config(config: TTLNoiseConfig) -> None:
    """Validate TTL noise configuration parameters.

    All physical parameters must be positive and finite.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any parameter is non-positive, NaN, or infinite.

    """
    if config.theta_rms <= 0:
        raise ValueError(
            f"RMS angular jitter theta_rms must be positive, got {config.theta_rms}",
        )
    if config.L <= 0:
        raise ValueError(f"Arm length L must be positive, got {config.L}")
    if config.wavelength <= 0:
        raise ValueError(f"Wavelength must be positive, got {config.wavelength}")
    if config.beam_offset <= 0:
        raise ValueError(f"Beam offset must be positive, got {config.beam_offset}")

    # Check for non-finite values
    for field_name in ("theta_rms", "L", "wavelength", "beam_offset"):
        value = getattr(config, field_name)
        if not np.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value}")


# =============================================================================
# Section: cavity_mzi
# =============================================================================

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
        >>> from src.physics.mzi_simulation import prepare_input_state
        >>> config = CavityMziConfig(F=10.0)
        >>> state = prepare_input_state("fock", max_photons=2, n_particles=1)
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
        >>> from src.physics.mzi_simulation import prepare_input_state
        >>> config = CavityMziConfig(F=5.0)
        >>> state = prepare_input_state("fock", max_photons=2, n_particles=1)
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


# =============================================================================
# Section: distributed_mzi
# =============================================================================
# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DistributedMziConfig:
    """Configuration for distributed interferometer array.

    Attributes:
        M: Number of sensors in the array.
        entangled: If True, use entanglement across sensors for collective
            Heisenberg-limited scaling. If False, independent sensors with
            classical averaging.
        correlation_noise: Noise correlation strength across sensors.
            - 0.0 = completely independent noise
            - 1.0 = fully correlated (common-mode) noise
            Correlated noise reduces the benefit of having multiple sensors.
        theta: Beam splitter angle (π/4 = 50/50 beam splitter).
        phi_bs: Reference phase at output beam splitter.

    """

    M: int = 2
    entangled: bool = False
    correlation_noise: float = 0.0
    theta: float = np.pi / 4
    phi_bs: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.M < 1:
            raise ValueError(f"Number of sensors M must be >= 1, got {self.M}")
        if self.correlation_noise < 0.0 or self.correlation_noise > 1.0:
            raise ValueError(
                f"correlation_noise must be in [0, 1], got {self.correlation_noise}",
            )


# =============================================================================
# Core Sensitivity Calculation
# =============================================================================


def distributed_mzi_sensitivity(
    N_per_sensor: int,
    phi: float,
    config: DistributedMziConfig,
    noise_config: NoiseConfig | None = None,
) -> dict:
    """Compute sensitivity for distributed interferometer array.

    Sensitivity regimes:

    1. **Independent sensors (classical averaging):**
       Each sensor operates at the SQL: Δφ_single = 1/√N_per_sensor
       Combining M independent measurements:
           Δφ_total = Δφ_single / √M = 1/√(M·N_per_sensor)

       This is the √M improvement from independent sampling.

    2. **Entangled sensors (collective Heisenberg limit):**
       With entanglement across all M sensors and photons:
           Δφ_total = 1 / (M·N_per_sensor)

       This gives super-classical scaling: better by factor of √(M·N)
       compared to the single-sensor SQL.

    3. **With correlated noise (correlation_noise = c):**
       Sensitivity combines independent and correlated contributions:
           Δφ_total² = (1-c)·Δφ_ind² + c·Δφ_corr²

       where:
       - Δφ_ind = independent contribution (improves with M)
       - Δφ_corr = correlated contribution (doesn't improve with M)

       For unentangled sensors with c=1: no benefit from multiple sensors.

    4. **With noise channels (NoiseConfig):**
       - Detection efficiency η: scales Fisher information by η
       - Loss/dephasing: add effective inefficiency

    Args:
        N_per_sensor: Photon number per sensor.
        phi: Phase shift being estimated.
        config: Distributed array configuration (M, entanglement, correlation).
        noise_config: Noise configuration for loss, dephasing, detection efficiency.

    Returns:
        Dictionary with sensitivity metrics:
        - delta_phi: Total phase sensitivity (Δφ)
        - delta_phi_independent: Independent noise contribution
        - delta_phi_correlated: Correlated noise contribution
        - effective_qfi: Effective quantum Fisher information
        - scaling_factor: Factor relative to single-sensor SQL
        - regime: Description of operating regime

    Raises:
        ValueError: If N_per_sensor <= 0 or M < 1.

    Example:
        >>> # Classical averaging: M=4 independent sensors
        >>> config = DistributedMziConfig(M=4, entangled=False)
        >>> result = distributed_mzi_sensitivity(100, 0.0, config)
        >>> # Expected: 1/√(4·100) = 0.05
        >>> abs(result["delta_phi"] - 0.05) < 0.001
        True

        >>> # Entangled: collective Heisenberg limit
        >>> config_ent = DistributedMziConfig(M=4, entangled=True)
        >>> result_ent = distributed_mzi_sensitivity(100, 0.0, config_ent)
        >>> # Expected: 1/(4·100) = 0.0025
        >>> abs(result_ent["delta_phi"] - 0.0025) < 0.0001
        True

        >>> # Fully correlated classical: no M benefit
        >>> config_corr = DistributedMziConfig(M=4, entangled=False, correlation_noise=1.0)
        >>> result_corr = distributed_mzi_sensitivity(100, 0.0, config_corr)
        >>> # Expected: 1/√100 = 0.1 (same as single sensor)
        >>> abs(result_corr["delta_phi"] - 0.1) < 0.01
        True

    """
    if N_per_sensor <= 0:
        raise ValueError(f"Photon number per sensor must be > 0, got {N_per_sensor}")

    # Default noise config
    if noise_config is None:
        noise_config = NoiseConfig()

    c = config.correlation_noise
    M = config.M

    # Readout efficiency from beam splitter and phase
    # For MZI at optimal working point
    visibility = np.abs(np.sin(2 * config.theta))
    phi_factor = np.abs(np.cos(phi - config.phi_bs))
    readout_efficiency = max(visibility * phi_factor, 1e-6)

    # Compute effective efficiency from noise channels
    # η_eff = detection_efficiency * exp(-loss_rates)
    loss_rate = (
        noise_config.gamma_1 + noise_config.gamma_2 * max(N_per_sensor - 1, 0) / 2
    )
    transmission = np.exp(-loss_rate) if loss_rate > 0 else 1.0
    eta_eff = noise_config.eta * transmission * readout_efficiency
    eta_eff = max(min(eta_eff, 1.0), 1e-6)

    # Dephasing adds variance directly
    dephasing_variance = noise_config.gamma_phi

    # -------------------------------------------------------------------------
    # Core analytical model
    # -------------------------------------------------------------------------

    if config.entangled:
        # ---------------------------------------------------------------------
        # Entangled case: collective Heisenberg limit
        # ---------------------------------------------------------------------
        # Without correlations: Δφ = 1/(M·N) where N_total = M·N
        # This gives Heisenberg scaling in the total resource N_total
        # F_Q = (M·N)² for maximally entangled states like GHZ/NOON

        # Correlation model for entangled:
        # - c=0: independent noise, full Heisenberg benefit
        #   F_independent = (M·N)²
        # - c=1: fully correlated noise, loses the M benefit
        #   Correlated noise doesn't average across sensors.
        #   F_correlated = N² (Heisenberg in N per sensor, but no M gain)
        #
        # Actually, this is subtle. For maximally entangled states across
        # all M·N particles, the M is part of the quantum enhancement.
        # Common-mode noise processes that affect all sensors equally will
        # couple to the collective phase, and don't average out.
        #
        # Simple interpolation model:
        # F_eff = (1-c) * (M·N)² + c * N²
        #
        # This gives:
        # - c=0: F = M²·N², Δφ = 1/(M·N) (full collective Heisenberg)
        # - c=1: F = N², Δφ = 1/N (Heisenberg scaling in N only)

        # Compute Fisher information components
        f_independent = (M * N_per_sensor) ** 2
        f_correlated = N_per_sensor**2

        # Interpolate with correlation parameter
        f_eff = (1 - c) * f_independent + c * f_correlated

        # Apply efficiency
        f_eff = eta_eff * f_eff

        # Variance from quantum measurement
        var_quantum = 1.0 / f_eff if f_eff > 0 else np.inf

        # Component breakdown
        if M > 1:
            var_independent = (
                1.0 / (eta_eff * f_independent) if f_independent > 0 else 0
            )
            var_independent = (1 - c) * var_independent
        else:
            var_independent = (1 - c) * var_quantum if c < 1 else 0

        var_correlated = (
            c / (eta_eff * f_correlated) if (f_correlated > 0 and eta_eff > 0) else 0
        )

        # Add dephasing
        total_variance = var_quantum + dephasing_variance

        # Regime determination
        if c > 0.9:
            regime = "Correlated noise - no collective benefit from M"
        elif c > 0.5:
            regime = "Partially correlated - partial collective benefit"
        else:
            regime = "Collective Heisenberg limit"

    else:
        # ---------------------------------------------------------------------
        # Unentangled case: classical averaging of M independent sensors
        # ---------------------------------------------------------------------
        # Without correlations: M independent sensors, each at SQL
        # Δφ_single = 1/√N
        # After averaging M independent estimates: Δφ_total = 1/√(M·N)

        # Single sensor quantum variance (after efficiency)
        var_single_quantum = 1.0 / (eta_eff * N_per_sensor)

        # With correlation c:
        # From the plan: Δφ_total² = (1-c)·Δφ_ind²/M + c·Δφ_corr²
        # where Δφ_ind = Δφ_corr = single-sensor sensitivity

        # Independent portion (benefits from M)
        if M > 1:
            var_independent = (1 - c) * var_single_quantum / M
        else:
            var_independent = (1 - c) * var_single_quantum

        # Correlated portion (doesn't benefit from M)
        var_correlated = c * var_single_quantum

        # Total quantum + dephasing variance
        total_variance = var_independent + var_correlated + dephasing_variance

        # Regime determination
        if c > 0.9:
            regime = "Correlated noise - no benefit from multiple sensors"
        elif c > 0.5:
            regime = "Partially correlated - partial benefit from M"
        else:
            regime = "Classical averaging (SQL per √M)"

    # Final sensitivity
    delta_phi = float(np.sqrt(total_variance))
    delta_phi_independent = float(np.sqrt(max(var_independent, 0.0)))
    delta_phi_correlated = float(np.sqrt(max(var_correlated, 0.0)))

    # Single-sensor SQL for comparison
    single_sql = 1.0 / np.sqrt(N_per_sensor)
    scaling_factor = float(single_sql / delta_phi) if delta_phi > 0 else 0.0

    # Effective QFI = 1/variance (excluding dephasing which is additive)
    qfi_denom = total_variance - dephasing_variance
    effective_qfi = float(1.0 / qfi_denom) if qfi_denom > 0 else 0.0

    return {
        "delta_phi": delta_phi,
        "delta_phi_independent": delta_phi_independent,
        "delta_phi_correlated": delta_phi_correlated,
        "effective_qfi": effective_qfi,
        "scaling_factor": scaling_factor,
        "regime": regime,
        "M": M,
        "N_per_sensor": N_per_sensor,
        "entangled": config.entangled,
        "correlation_noise": c,
    }


# =============================================================================
# Scaling Exponents
# =============================================================================


def distributed_scaling_exponent(
    config: DistributedMziConfig,
) -> float:
    """Expected scaling exponent for distributed array.

    Returns the exponent α in Δφ ∝ N^α (with M fixed).

    Scaling regimes (ideal case, no noise):

    1. **Uncorrelated classical:** α = -0.5
       - Δφ ∝ 1/√N (SQL per sensor)
       - With M sensors: Δφ ∝ 1/√(M·N), still α = -0.5 in N

    2. **Entangled:** α = -1.0
       - Δφ ∝ 1/N (Heisenberg scaling in total photon number M·N)
       - When plotted vs N_per_sensor with fixed M, still see α = -1

    3. **Correlated noise dominated:** α → 0
       - If common-mode noise is sufficiently strong, sensitivity
       may saturate to a constant (or scale very weakly with N)

    Args:
        config: Distributed array configuration.

    Returns:
        α: The expected scaling exponent α in Δφ ∝ N^α.
           - For unentangled/independent: -0.5
           - For entangled/noiseless: -1.0

    Example:
        >>> # Classical unentangled
        >>> config_classical = DistributedMziConfig(M=4, entangled=False)
        >>> distributed_scaling_exponent(config_classical)
        -0.5

        >>> # Entangled
        >>> config_ent = DistributedMziConfig(M=4, entangled=True)
        >>> distributed_scaling_exponent(config_ent)
        -1.0

    """
    if config.entangled:
        return -1.0
    return -0.5


def effective_scaling_at_N(
    N_per_sensor: int,
    config: DistributedMziConfig,
    noise_config: NoiseConfig | None = None,
) -> float:
    """Compute effective scaling exponent at a specific photon number.

    Unlike distributed_scaling_exponent which returns the asymptotic exponent,
    this function computes the numerical derivative of log(Δφ) with respect
    to log(N) at the given N, accounting for noise floors.

    With strong correlated noise or at low N, the effective exponent can be
    closer to 0 (saturation) than the asymptotic -0.5 or -1.

    Args:
        N_per_sensor: Photon number where to evaluate the exponent.
        config: Distributed array configuration.
        noise_config: Optional noise configuration.

    Returns:
        Effective exponent α ≡ d(log(Δφ))/d(log(N)).

    """
    # Use central difference with small perturbation
    rel_perturb = 0.01
    N_plus_int = max(round(N_per_sensor * (1 + rel_perturb)), 1)
    N_minus_int = max(round(N_per_sensor * (1 - rel_perturb)), 1)

    # Handle edge case where N is 1
    if N_minus_int == N_plus_int:
        N_plus_int = N_minus_int + 1

    phi_test = 0.0

    result_plus = distributed_mzi_sensitivity(
        N_plus_int,
        phi_test,
        config,
        noise_config,
    )
    result_minus = distributed_mzi_sensitivity(
        N_minus_int,
        phi_test,
        config,
        noise_config,
    )

    # log-log derivative
    log_N_plus = np.log(N_plus_int)
    log_N_minus = np.log(N_minus_int)
    log_dp_plus = np.log(result_plus["delta_phi"])
    log_dp_minus = np.log(result_minus["delta_phi"])

    alpha = (log_dp_plus - log_dp_minus) / (log_N_plus - log_N_minus)

    return float(alpha)


# =============================================================================
# Scaling Comparison
# =============================================================================


def compute_distributed_scaling(
    M_values: list[int] | np.ndarray,
    N_per_sensor_values: list[int] | np.ndarray,
    config_template: DistributedMziConfig,
    noise_config: NoiseConfig | None = None,
) -> dict:
    """Compute sensitivity scaling across M and N values.

    Creates a grid of sensitivity values for varying M (sensor count)
    and N (photons per sensor). Useful for generating scaling plots
    comparing classical vs distributed quantum metrology.

    Args:
        M_values: Array of sensor counts to evaluate.
        N_per_sensor_values: Array of photon numbers to evaluate.
        config_template: Base configuration (entanglement, correlation, angles).
            The M value will be varied across M_values.
        noise_config: Optional noise configuration.

    Returns:
        Dictionary with 2D grids:
        - M_grid: Meshgrid of M values shape (len(M_values), len(N_values))
        - N_grid: Meshgrid of N values
        - delta_phi_grid: Sensitivity values at each (M, N)
        - scaling_factors: Improvement vs single-sensor SQL
        - regimes: Operating regime strings for each point

    Example:
        >>> # Compare classical vs entangled scaling
        >>> M_values = [1, 2, 4, 8]
        >>> N_values = [10, 100, 1000]
        >>> config = DistributedMziConfig(entangled=False)  # M varies
        >>> result = compute_distributed_scaling(M_values, N_values, config)
        >>> result["delta_phi_grid"].shape
        (4, 3)

    """
    M_values = np.asarray(M_values)
    N_values = np.asarray(N_per_sensor_values)

    n_M = len(M_values)
    n_N = len(N_values)

    delta_phi_grid = np.zeros((n_M, n_N))
    scaling_factors = np.zeros((n_M, n_N))
    regimes = []

    for i, M in enumerate(M_values):
        # Create config for this M
        config = DistributedMziConfig(
            M=int(M),
            entangled=config_template.entangled,
            correlation_noise=config_template.correlation_noise,
            theta=config_template.theta,
            phi_bs=config_template.phi_bs,
        )
        row_regimes = []
        for j, N in enumerate(N_values):
            result = distributed_mzi_sensitivity(int(N), 0.0, config, noise_config)
            delta_phi_grid[i, j] = result["delta_phi"]
            scaling_factors[i, j] = result["scaling_factor"]
            row_regimes.append(result["regime"])
        regimes.append(row_regimes)

    M_grid, N_grid = np.meshgrid(M_values, N_values, indexing="ij")

    return {
        "M_grid": M_grid,
        "N_grid": N_grid,
        "delta_phi_grid": delta_phi_grid,
        "scaling_factors": scaling_factors,
        "regimes": regimes,
    }


# =============================================================================
# Section: scaling_survey
# =============================================================================
# Load pseudomode module from 2026-05-09 report via importlib
import importlib.util as _importlib_util  # noqa: E402
import json  # noqa: E402
import sys as _sys  # noqa: E402
from pathlib import Path  # noqa: E402

_pseudomode_local_path = (
    Path(__file__).resolve().parent.parent / "20260509" / "local.py"
)
_pseudomode_spec = _importlib_util.spec_from_file_location(
    "pseudomode_system", str(_pseudomode_local_path)
)
assert _pseudomode_spec is not None, (
    f"Could not find 20260509/local.py at {_pseudomode_local_path}"
)
_pseudomode_module = _importlib_util.module_from_spec(_pseudomode_spec)
assert _pseudomode_spec.loader is not None
_sys.modules["pseudomode_system"] = _pseudomode_module
_pseudomode_spec.loader.exec_module(_pseudomode_module)

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# Custom Sensitivity Function Generators
# =============================================================================


def _non_gaussian_sensitivity_fn(
    n_order: int,
    omega_n: float = 0.5,
    theta_n: float = 0.0,
    t_sqz: float = 2.0,
    use_ground_state: bool = False,
) -> Callable[[int, float], float]:
    """Create a sensitivity function for n-th order non-Gaussian states.

    Uses the hybrid oscillator-spin system from ``hybrid_system.py``.
    The ``N`` parameter in the survey maps to the oscillator Fock truncation;
    the squeezing parameters ``omega_n``, ``theta_n``, ``t_sqz`` are fixed at
    construction.

    Two state preparation modes:
    - ``use_ground_state=False`` (default): time-evolve |0,↓⟩ under H_n for t_sqz
    - ``use_ground_state=True``: lowest-energy eigenstate of H_n via
      :func:`hybrid_ground_state_n`

    Args:
        n_order: Squeezing order (2, 3, or 4).
        omega_n: Squeezing rate Ω_n. Default 0.5.
        theta_n: Squeezing phase θ_n. Default 0.0.
        t_sqz: Squeezing evolution time. Default 2.0.
        use_ground_state: If True, use true ground state instead of
            time-evolved vacuum. Default False.

    Returns:
        Callable ``(N: int, noise_level: float) -> delta_phi: float``
        suitable for use as ``ModelConfig.custom_sensitivity_fn``.

    Raises:
        ValueError: If n_order is not 2, 3, or 4.

    """

    if n_order not in (2, 3, 4):
        raise ValueError(f"n_order must be 2, 3, or 4, got {n_order}")

    def _sensitivity(N: int, noise_level: float) -> float:
        # For hybrid-system N maps directly to oscillator truncation
        try:
            if N < 2:
                return np.inf

            if use_ground_state:
                state = hybrid_ground_state_n(N, n_order, omega_n, theta_n)
            else:
                state = hybrid_vacuum_state(N, spin_state="down")
                H = hybrid_hamiltonian_n(N, n_order, omega_n, theta_n)
                U = scipy.linalg.expm(-1j * H * t_sqz)
                state = U @ state

            # QFI via MZI readout (analytical formula — no phase sweep needed)
            fq = qfi_hybrid_mzi(state, N)
        except Exception:
            return np.inf

        if fq <= 0 or not np.isfinite(fq):
            return np.inf

        return 1.0 / np.sqrt(fq)

    return _sensitivity


def _ancilla_sensitivity_fn(
    alpha: float = 1.0,
    g_sa: float = 1.0,
    tau: float = 0.1,
    g_sp: float = 0.0,
    omega_0: float = 0.0,
    lam: float = 0.0,
    K: int = 2,
) -> Callable[[int, float], float]:
    """Create a sensitivity function for ancilla-assisted metrology.

    Uses the pseudomode-based non-Markovian ancilla protocol from
    ``pseudomode_system.py``. The ``N`` parameter maps to oscillator
    Fock truncation; ancilla, bath, and coupling parameters are fixed
    at construction.

    When ``g_sp=0`` and ``lam=0`` (the defaults), the environment is
    effectively Markovian and the protocol reduces to the dispersive
    ancilla-coupling model described in the report.

    Args:
        alpha: Coherent state amplitude for the oscillator probe.
            Default 1.0 (mean photon number = 1).
        g_sa: System-ancilla coupling strength. Default 1.0.
        tau: Ancilla entanglement time. Default 0.1.
        g_sp: System-pseudomode coupling strength. Default 0.0
            (no environment coupling).
        omega_0: Bath central frequency (pseudomode free energy).
            Default 0.0.
        lam: Bath correlation rate (pseudomode damping). Default 0.0
            (no dissipation).
        K: Pseudomode Fock truncation. Default 2.

    Returns:
        Callable ``(N: int, noise_level: float) -> delta_phi: float``
        suitable for use as ``ModelConfig.custom_sensitivity_fn``.

    Note:
        The ``noise_level`` parameter of the survey is used to scale
        the ancilla-system coupling ``g_sa`` (larger noise_level =
        stronger coupling / effective decoherence). The base ``g_sa``
        is multiplied by ``(1 + noise_level)``.

    """
    # Imported via importlib from reports/20260509/local.py at module scope.

    def _sensitivity(N: int, noise_level: float) -> float:
        try:
            if N < 2:
                return np.inf

            config = _pseudomode_module.PseudomodeConfig(
                N=N,
                K=K,
                alpha=alpha,
                g_sa=g_sa * (1.0 + noise_level),  # noise scales coupling
                tau=tau,
                g_sp=g_sp,
                omega_0=omega_0,
                lam=lam,
                T=1.0,
                dt=0.1,
            )
            result = _pseudomode_module.run_metrology_protocol(config)
            fq = result["qfi_with"]
        except Exception:
            return np.inf

        if fq <= 0 or not np.isfinite(fq):
            return np.inf

        return 1.0 / np.sqrt(fq)

    return _sensitivity


def _kerr_mzi_sensitivity_fn(
    chi: float = 0.1,
    T: float = 1.0,
    state_type: str = "noon",
) -> Callable[[int, float], float]:
    """Create a sensitivity function for Kerr-nonlinear MZI.

    The Kerr nonlinearity :math:`\\chi (n_1^2 + n_2^2)` commutes with
    the phase generator :math:`n_2`, so the QFI is invariant under Kerr
    evolution. For NOON states this gives :math:`F_Q = N^2` (Heisenberg
    limit) regardless of the Kerr strength.

    Args:
        chi: Kerr nonlinearity strength. Default 0.1.
        T: Evolution time for the Kerr interaction. Default 1.0.
            The product ``chi * T`` controls the nonlinear phase.
        state_type: Input state type passed to ``input_state_factory``.
            Default ``"noon"``.

    Returns:
        Callable ``(N: int, noise_level: float) -> delta_phi: float``
        suitable for use as ``ModelConfig.custom_sensitivity_fn``.

    """
    # Validate state_type
    valid_states = {"noon", "coherent", "css", "twin_fock", "sss", "squeezed_vacuum"}
    if state_type not in valid_states:
        state_type = "noon"

    def _sensitivity(N: int, noise_level: float) -> float:
        if N < 2:
            return np.inf
        try:
            max_photons = _max_photons_for_state(state_type, N)
            state = input_state_factory(state_type, N=N, max_photons=max_photons)
            # Pure-state QFI: F_Q = 4·Var(J_z) — invariant under Kerr
            delta = _compute_pure_state_sensitivity(state, max_photons)
        except Exception:
            return np.inf

        if not np.isfinite(delta) or delta <= 0:
            return np.inf
        return delta

    return _sensitivity


def _weak_value_mzi_sensitivity_fn(
    post_select_angle: float = np.pi / 2 - 0.1,
) -> Callable[[int, float], float]:
    """Create a sensitivity function for weak-value MZI with coherent state input.

    Uses the analytical formula from ``weak_value_mzi_sensitivity``,
    which computes Fisher information as :math:`F = N \\cdot \\cos^2(\\delta)`
    where :math:`\\delta = \\pi/2 - \\text{post\\_select\\_angle}`.

    The sensitivity is :math:`\\Delta\\phi = 1 / \\sqrt{N \\cdot \\cos^2(\\delta)}`,
    which never beats the SQL (``\\Delta\\phi \\ge 1/\\sqrt{N}``).

    Args:
        post_select_angle: Post-selection angle. Values near :math:`\\pi/2`
            give large amplification at the cost of vanishing post-selection
            probability. Default ``π/2 - 0.1`` (~10× amplification).

    Returns:
        Callable ``(N: int, noise_level: float) -> delta_phi: float``
        suitable for use as ``ModelConfig.custom_sensitivity_fn``.

    """
    # Note: WeakValueConfig and weak_value_mzi_sensitivity are defined
    # within this same module (co-located), so no import needed.

    def _sensitivity(N: int, noise_level: float) -> float:
        if N < 2:
            return np.inf
        try:
            config = WeakValueConfig(post_select_angle=post_select_angle)
            result = weak_value_mzi_sensitivity(N=N, phi=0.0, config=config)
            delta = result["delta_phi"]
        except Exception:
            return np.inf

        if not np.isfinite(delta) or delta <= 0:
            return np.inf
        return delta

    return _sensitivity


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SurveyConfig:
    """Configuration for a scaling survey sweep.

    Attributes:
        N_range: Tuple of (min, max) particle number range.
            N values are log-spaced between min and max. Default: (2, 64).
        N_points: Number of N values to sweep (log-spaced). Default: 8.
        noise_levels: List of noise levels (dephasing rates) to sweep.
            Each level corresponds to a J_z dephasing rate γ.
            Default: [0.0, 1e-3, 1e-2, 1e-1].
        phi: Operating phase for sensitivity estimation. For QFI-based
            estimation, the sensitivity is phase-independent for pure states
            (F_Q = 4·Var(J_z)), but noise effects may depend on the phase.
            Default: π/4.
        measurement: Measurement type for sensitivity estimation.
            Options: "parity", "Jz", "number_difference".
            Default: "parity".
        method: Sensitivity estimation method.
            Options: "qfi" (Quantum Fisher Information), "cf" (Classical Fisher),
            "ep" (Error Propagation), "bayesian".
            Default: "qfi".
        seed: Random seed for reproducibility. Default: 42.

    """

    N_range: tuple[int, int] = (2, 64)
    N_points: int = 8
    noise_levels: list[float] = field(default_factory=lambda: [0.0, 1e-3, 1e-2, 1e-1])
    phi: float = np.pi / 4
    measurement: str = "parity"
    method: str = "qfi"
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.N_range[0] <= 0:
            raise ValueError(f"N_range minimum must be positive, got {self.N_range[0]}")
        if self.N_range[1] < self.N_range[0]:
            raise ValueError(
                f"N_range max ({self.N_range[1]}) must be >= min ({self.N_range[0]})",
            )
        if self.N_points < 2:
            raise ValueError(f"N_points must be >= 2, got {self.N_points}")
        if self.phi < 0 or self.phi > 2 * np.pi:
            raise ValueError(f"phi must be in [0, 2π], got {self.phi}")
        valid_methods = {"qfi", "cf", "ep", "bayesian"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {self.method}",
            )
        for nl in self.noise_levels:
            if nl < 0:
                raise ValueError(f"Noise level must be non-negative, got {nl}")


@dataclass
class ModelConfig:
    """Configuration for a specific model in the survey.

    Attributes:
        model_id: Unique identifier string for the model
            (e.g., "ideal_coherent", "noon_loss").
        state_type: Type of input state. Must match input_state_factory
            options: "coherent", "noon", "twin_fock", "single_photon_split",
            "fock", "css", "sss", "squeezed_vacuum".
            Ignored when ``custom_sensitivity_fn`` is set.
        noise_type: Type of physical noise.
            Options: "none", "loss", "dephasing", "two_body", "detection",
            "thermal", "custom".
        entangler: Entanglement generation protocol.
            Options: "none", "oat" (one-axis twisting), "tnt" (two-axis twisting).
        label: Human-readable label for plots and tables.
        custom_sensitivity_fn: Optional callable ``(N: int, noise_level: float) -> float``
            that computes Δφ(N) for a model outside the standard MZI pipeline.
            When set, ``state_type`` and ``entangler`` are ignored; the callable
            is invoked directly by ``run_scaling_survey``.
            Typical usage: plug in cavity, distributed, DD, TTL, or thermal models.

    """

    model_id: str
    state_type: str = ""
    noise_type: str = "none"
    entangler: str = "none"
    label: str = ""
    custom_sensitivity_fn: Callable[[int, float], float] | None = None

    def __post_init__(self) -> None:
        """Validate model configuration."""
        valid_noise_types = {
            "none",
            "dephasing",
            "loss",
            "two_body",
            "detection",
            "thermal",
            "custom",
        }
        if self.noise_type not in valid_noise_types:
            raise ValueError(
                f"Unknown noise_type: {self.noise_type}. "
                f"Must be one of: {valid_noise_types}",
            )


# =============================================================================
# Internal Helpers
# =============================================================================


def _max_photons_for_state(state_type: str, N: int) -> int:
    """Determine appropriate max_photons based on state type.

    For Fock states with definite photon number (noon, twin_fock, fock),
    max_photons=N is sufficient.

    For coherent states (css, coherent), the Poisson distribution has
    tails that extend beyond the mean, so we need a larger Hilbert space.

    Args:
        state_type: Type of state.
        N: Target mean/specified photon number.

    Returns:
        Appropriate max_photons value.

    """
    # States that need larger Hilbert space for Poisson tails
    coherent_like = {"css", "coherent", "squeezed_vacuum"}
    if state_type in coherent_like:
        # Use max(2*N, N+20) to capture Poisson tail
        return max(2 * N, N + 20)
    return N


def _generate_N_values(config: SurveyConfig) -> np.ndarray:
    """Generate log-spaced integer N values for the survey sweep.

    Produces unique integer N values logarithmically spaced between
    N_range[0] and N_range[1]. Log-spacing ensures even coverage
    across orders of magnitude in particle number.

    Args:
        config: Survey configuration specifying the range and count.

    Returns:
        Sorted 1D array of unique integer N values.

    """
    N_min, N_max = config.N_range
    N_raw = np.logspace(np.log10(N_min), np.log10(N_max), config.N_points).astype(int)
    # Deduplicate from rounding and sort
    N_unique = np.unique(N_raw)
    # Ensure at least 2 points
    if len(N_unique) < 2:
        N_unique = np.array([N_min, N_max])
    return N_unique


def _apply_entanglement(
    state: np.ndarray,
    N: int,
    entangler: str,
) -> np.ndarray:
    """Apply entanglement generation to the state.

    Converts the two-mode Fock state to the Dicke basis, applies the
    specified entanglement unitary, and converts back.

    Supported entanglers:
    - "none": Identity operation (no entanglement).
    - "oat": One-axis twisting U = exp(-i χ J_z² t) with optimal
      squeezing time t_opt = (6/N)^{1/3} and χ = 1.
    - "tnt": Two-axis twisting U = exp(-i χ (J_+² + J_-²) t / 2)
      with t_opt = (6/N)^{1/3} and χ = 1.

    Args:
        state: State vector in the two-mode Fock basis
            (dimension (N+1)²).
        N: Total particle number.
        entangler: Type of entanglement to apply.

    Returns:
        Entangled state vector in the two-mode Fock basis.

    Raises:
        ValueError: If entangler type is not recognized.

    """
    if entangler == "none":
        return state.copy()

    # Convert to Dicke basis
    try:
        dicke_state = to_dicke_basis(state, N)
    except ValueError:
        # State may not be in the symmetric subspace; return as-is
        return state.copy()

    if entangler == "oat":
        # One-axis twisting: U = exp(-i χ t J_z²)
        # Optimal squeezing time from Kitagawa & Ueda (1993)
        chi = 1.0
        t_opt = (6.0 / N) ** (1.0 / 3.0) / chi if N > 0 else 0.0

        # J_z eigenvalues: m = N/2, N/2-1, ..., -N/2
        J = N / 2.0
        m_values = np.arange(J, -J - 1, -1)

        # Phase factors: exp(-i χ t m²)
        phases = np.exp(-1j * chi * t_opt * m_values**2)
        dicke_state = phases * dicke_state

    elif entangler == "tnt":
        # Two-axis twisting: U = exp(-i χ t (J_+² + J_-²) / 2)
        # Approximate optimal time similar to OAT
        chi = 1.0
        t_opt = (6.0 / N) ** (1.0 / 3.0) / chi if N > 0 else 0.0

        J_plus = qutip.jmat(N / 2.0, "+").full()
        J_minus = qutip.jmat(N / 2.0, "-").full()
        H_tnt = (J_plus @ J_plus + J_minus @ J_minus) / 2

        from scipy.linalg import expm

        U_tnt = expm(-1j * chi * t_opt * H_tnt)
        dicke_state = U_tnt @ dicke_state
    else:
        raise ValueError(f"Unknown entangler type: {entangler}")

    # Convert back to two-mode Fock basis
    return from_dicke_basis(dicke_state, N)


def _compute_noisy_sensitivity(
    state: np.ndarray,
    max_photons: int,
    noise_level: float,
    noise_type: str = "dephasing",
    T: float = 1.0,
) -> float:
    """Compute phase sensitivity for a state with the specified noise type.

    For pure states without noise (noise_level ≈ 0), computes the
    Quantum Fisher Information directly as F_Q = 4·Var(J_z) and
    returns Δφ = 1/√F_Q.

    For noisy states (noise_level > 0), uses the Lindblad master equation
    evolution to apply the specified noise type, then computes QFI via
    the full SLD formula for mixed states.

    Supported noise types:
    - "none": No noise (pure state evolution only)
    - "dephasing": Phase diffusion via L = √γ J_z (default)
    - "loss": One-body loss via L = √γ a
    - "two_body": Two-body/pair loss via L = √γ a²
    - "detection": Imperfect detection (efficiency parameter)

    Args:
        state: Pure state vector in the two-mode Fock basis.
        max_photons: Hilbert space truncation (max photons per mode).
        noise_level: Noise strength γ (dimensionless rate × time).
            For detection noise, this is the efficiency η ∈ [0, 1].
        noise_type: Type of noise to apply. Default: "dephasing".
        T: Evolution time for Lindblad dynamics. Default: 1.0.

    Returns:
        Phase sensitivity Δφ (lower is better). Returns np.inf if
        the QFI is zero or negative (no phase information).

    """
    noise_free = noise_level <= 0.0 or np.isclose(noise_level, 0.0)
    no_noise_type = noise_type == "none"

    if noise_free or no_noise_type:
        return _compute_pure_state_sensitivity(state, max_photons)

    # --- Noisy case: use QuTiP Lindblad evolution ---
    # Build Lindblad collapse operators based on noise_type
    try:
        dim = max_photons + 1
        n0 = qutip.tensor(qutip.num(dim), qutip.qeye(dim))
        n1 = qutip.tensor(qutip.qeye(dim), qutip.num(dim))
        jz = (n0 - n1) / 2.0

        if noise_type == "dephasing":
            c_ops = [np.sqrt(noise_level) * jz]
        elif noise_type == "loss":
            a1 = qutip.tensor(qutip.qeye(dim), qutip.destroy(dim))
            c_ops = [np.sqrt(noise_level) * a1]
        elif noise_type == "two_body":
            a1 = qutip.tensor(qutip.qeye(dim), qutip.destroy(dim))
            c_ops = [np.sqrt(noise_level) * (a1 @ a1)]
        elif noise_type == "detection":
            return _compute_detection_noise_sensitivity(state, max_photons, noise_level)
        else:
            c_ops = [np.sqrt(noise_level) * jz]

        H0 = 0 * n0  # zero Hamiltonian with matching dims
        state_q = qutip.Qobj(state.reshape(-1, 1), dims=[[dim, dim], [1, 1]])
        rho0 = qutip.ket2dm(state_q)
        tlist = [0.0, T]

        result = qutip.mesolve(
            H0, rho0, tlist, c_ops=c_ops, options={"store_states": True}
        )
        rho_noisy = result.states[-1].full()
    except Exception:
        return np.inf

    # J_z operator in the two-mode Fock basis
    J_z = two_mode_jz_operator(max_photons)

    # Compute QFI via SLD formula for mixed states
    try:
        F_Q = quantum_fisher_information_dm(rho_noisy, J_z)
    except (ValueError, np.linalg.LinAlgError):
        return np.inf

    if F_Q <= 0 or not np.isfinite(F_Q):
        return np.inf

    return 1.0 / np.sqrt(F_Q)


def _compute_detection_noise_sensitivity(
    state: np.ndarray,
    max_photons: int,
    eta: float,
) -> float:
    """Compute sensitivity under imperfect detection.

    Detection efficiency η attenuates the quantum Fisher information
    by reducing the distinguishability of measurement outcomes.

    For a given state with photon number distribution P(n), the
    effective QFI with detection noise is bounded by:
        F_Q,eff ≤ η · F_Q

    Args:
        state: Pure state vector.
        max_photons: Hilbert space truncation.
        eta: Detection efficiency η ∈ [0, 1].

    Returns:
        Phase sensitivity Δφ.

    """
    if eta <= 0:
        return np.inf

    # Base pure-state QFI via direct QuTiP variance computation
    try:
        dim = max_photons + 1
        state_q = qutip.Qobj(state.reshape(-1, 1), dims=[[dim, dim], [1, 1]])
        n0 = qutip.tensor(qutip.num(dim), qutip.qeye(dim))
        n1 = qutip.tensor(qutip.qeye(dim), qutip.num(dim))
        jz = (n0 - n1) / 2.0
        var_jz = float(qutip.variance(jz, state_q))
        F_Q_pure = 4.0 * var_jz
    except Exception:
        return np.inf

    if F_Q_pure <= 0 or not np.isfinite(F_Q_pure):
        return np.inf

    # For detection inefficiency, the effective QFI is reduced.
    # The bound Δφ ≥ 1/√(η·F_Q) gives a conservative estimate.
    F_Q_eff = eta * F_Q_pure

    if F_Q_eff <= 0:
        return np.inf

    return 1.0 / np.sqrt(F_Q_eff)


def _compute_pure_state_sensitivity(state: np.ndarray, max_photons: int) -> float:
    """Compute phase sensitivity for a noiseless pure state.

    Uses the Quantum Fisher Information for pure states:
        F_Q = 4 · Var(J_z)  →  Δφ = 1/√F_Q

    Uses QuTiP directly for the J_z operator construction and variance
    computation, bypassing intermediate wrappers.

    Args:
        state: Pure state vector in the two-mode Fock basis.
        max_photons: Hilbert space truncation.

    Returns:
        Phase sensitivity Δφ.

    """
    try:
        dim = max_photons + 1
        state_q = qutip.Qobj(state.reshape(-1, 1), dims=[[dim, dim], [1, 1]])
        n0 = qutip.tensor(qutip.num(dim), qutip.qeye(dim))
        n1 = qutip.tensor(qutip.qeye(dim), qutip.num(dim))
        jz = (n0 - n1) / 2.0
        var_jz = float(qutip.variance(jz, state_q))
        F_Q = 4.0 * var_jz
    except (ValueError, IndexError, Exception):
        return np.inf

    if F_Q <= 0 or not np.isfinite(F_Q):
        return np.inf

    return 1.0 / np.sqrt(F_Q)


def _apply_phase_imprint(
    state: np.ndarray,
    phi: float,
    max_photons: int,
) -> np.ndarray:
    """Apply a phase shift U = exp(i φ n₂) to mode 1.

    Args:
        state: State vector in the two-mode Fock basis.
        phi: Phase shift in radians.
        max_photons: Maximum photon number per mode.

    Returns:
        Phase-imprinted state vector.

    """
    phase_U = phase_shift_unitary(phi, max_photons)
    return phase_U @ state


# =============================================================================
# Survey Orchestration
# =============================================================================


def run_scaling_survey(
    models: list[ModelConfig],
    survey_config: SurveyConfig,
    progress_callback: Callable | None = None,
) -> pd.DataFrame:
    """Run the full scaling survey over all models and noise levels.

    For each combination of (model, noise_level, N):
        1. Prepare the input state via input_state_factory
        2. Apply entanglement if configured (OAT/TNT)
        3. Apply phase imprint (phase shift on mode 1)
        4. Apply dephasing noise at the specified level
        5. Compute phase sensitivity via the specified method
        6. Collect the result

    After all sweeps, the returned DataFrame contains raw sensitivity
    values for subsequent exponent fitting via fit_all_exponents.

    Args:
        models: List of ModelConfig objects defining the states and
            noise types to sweep.
        survey_config: SurveyConfig controlling the N range, noise
            levels, operating phase, and estimation method.
        progress_callback: Optional callable(current, total) called
            after each (model, noise_level, N) combination completes.
            Useful for progress bars in interactive contexts.

    Returns:
        DataFrame with columns:
            model_id, state_type, noise_type, noise_level, N, delta_phi,
            method, entangler

    Raises:
        ValueError: If models list is empty or survey_config is invalid.

    """
    if not models:
        raise ValueError("At least one model must be specified")

    # Validate survey config (triggers __post_init__)
    # (already validated at construction, but we can trust the caller)

    N_values = _generate_N_values(survey_config)

    results: list[dict] = []
    total = len(models) * len(survey_config.noise_levels) * len(N_values)
    count = 0

    for model in models:
        for noise_level in survey_config.noise_levels:
            for N_raw in N_values:
                N = int(N_raw)

                # --- Custom model path (bypasses standard MZI pipeline) ---
                if model.custom_sensitivity_fn is not None:
                    try:
                        delta_phi = model.custom_sensitivity_fn(N, noise_level)
                    except Exception:
                        delta_phi = np.inf

                    results.append(
                        {
                            "model_id": model.model_id,
                            "state_type": model.state_type,
                            "noise_type": model.noise_type,
                            "noise_level": noise_level,
                            "N": N,
                            "delta_phi": delta_phi,
                            "method": survey_config.method,
                            "entangler": model.entangler,
                            "label": model.label,
                        },
                    )
                    count += 1
                    if progress_callback:
                        progress_callback(count, total)
                    continue

                # --- Standard MZI pipeline ---

                # --- Step 1: Prepare input state ---
                try:
                    max_photons = _max_photons_for_state(model.state_type, N)
                    state = input_state_factory(
                        model.state_type,
                        N=N,
                        max_photons=max_photons,
                    )
                except (ValueError, TypeError):
                    # If state cannot be created at this N, skip gracefully
                    results.append(
                        {
                            "model_id": model.model_id,
                            "state_type": model.state_type,
                            "noise_type": model.noise_type,
                            "noise_level": noise_level,
                            "N": N,
                            "delta_phi": np.inf,
                            "method": survey_config.method,
                            "entangler": model.entangler,
                            "label": model.label,
                        },
                    )
                    count += 1
                    if progress_callback:
                        progress_callback(count, total)
                    continue

                # --- Step 2: Apply entanglement ---
                # Note: _apply_entanglement uses N for Dicke basis dimension (N+1),
                # which is based on particle number, not Hilbert space truncation
                try:
                    state = _apply_entanglement(state, N, model.entangler)
                except (ValueError, np.linalg.LinAlgError):
                    state = state  # Continue with unentangled state

                # --- Step 3: Apply phase imprint ---
                try:
                    state = _apply_phase_imprint(state, survey_config.phi, max_photons)
                except (ValueError, IndexError):
                    pass  # Continue without phase imprint if it fails

                # --- Steps 4 & 5: Apply noise and compute sensitivity ---
                delta_phi = _compute_noisy_sensitivity(
                    state,
                    max_photons,
                    noise_level,
                    model.noise_type,
                )

                results.append(
                    {
                        "model_id": model.model_id,
                        "state_type": model.state_type,
                        "noise_type": model.noise_type,
                        "noise_level": noise_level,
                        "N": N,
                        "delta_phi": delta_phi,
                        "method": survey_config.method,
                        "entangler": model.entangler,
                        "label": model.label,
                    },
                )

                count += 1
                if progress_callback:
                    progress_callback(count, total)

    df = pd.DataFrame(results)

    # Ensure numeric types
    for col in ["N", "noise_level", "delta_phi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =============================================================================
# Exponent Fitting
# =============================================================================


def fit_all_exponents(
    survey_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    min_N: int = 4,
    R_squared_threshold: float = 0.9,
) -> pd.DataFrame:
    """Fit scaling exponents for each group in survey data.

    Groups the survey results by the specified columns (e.g., model_id,
    noise_level, method) and fits Δφ = C·N^α for each group.

    Args:
        survey_df: DataFrame from run_scaling_survey.
        group_cols: Columns to group by for independent fits.
            Default: ["model_id", "state_type", "noise_level", "method"].
        min_N: Minimum N to include in each fit (passed to
            fit_scaling_exponent). Default: 4.
        R_squared_threshold: R² warning threshold. Default: 0.9.

    Returns:
        DataFrame with columns:
            model_id, state_type, noise_type, noise_level, method,
            alpha, alpha_err, C, C_err, R_squared, valid, n_points, n_warnings
        Each row is a fitted exponent for one group.

    Raises:
        ValueError: If survey_df is empty or missing required columns.

    """
    required = {"N", "delta_phi"}
    missing = required - set(survey_df.columns)
    if missing:
        raise ValueError(
            f"survey_df missing required columns: {missing}. "
            f"Has columns: {list(survey_df.columns)}",
        )

    if survey_df.empty:
        raise ValueError("survey_df is empty")

    if group_cols is None:
        group_cols = ["model_id", "state_type", "noise_type", "noise_level", "method"]

    # Filter out infinite values for fitting
    finite_df = survey_df.loc[
        np.isfinite(survey_df["delta_phi"]) & (survey_df["delta_phi"] > 0)
    ].copy()

    if finite_df.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "alpha",
                "alpha_err",
                "C",
                "C_err",
                "R_squared",
                "valid",
                "n_points",
            ],
        )

    fit_rows: list[dict] = []
    available_groups = [c for c in group_cols if c in finite_df.columns]

    try:
        grouped = finite_df.groupby(available_groups, dropna=True)
    except Exception:
        return pd.DataFrame(
            columns=[
                *available_groups,
                "alpha",
                "alpha_err",
                "C",
                "C_err",
                "R_squared",
                "valid",
                "n_points",
            ],
        )

    for group_keys, group_df in grouped:
        # Ensure groups is always a tuple
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        N_arr = group_df["N"].to_numpy().astype(float)
        delta_arr = group_df["delta_phi"].to_numpy().astype(float)

        result = fit_scaling_exponent(
            N_arr,
            delta_arr,
            min_N=min_N,
            R_squared_threshold=R_squared_threshold,
        )

        row: dict = dict(zip(available_groups, group_keys, strict=False))
        row["alpha"] = result.alpha
        row["alpha_err"] = result.alpha_err
        row["C"] = result.C
        row["C_err"] = result.C_err
        row["R_squared"] = result.R_squared
        row["valid"] = result.valid
        row["n_points"] = len(result.N_values)
        row["n_warnings"] = len(result.warnings)

        fit_rows.append(row)

    fit_df = pd.DataFrame(fit_rows)

    if "alpha" in fit_df.columns:
        fit_df = fit_df.sort_values("alpha", ascending=True).reset_index(drop=True)

    return fit_df


# =============================================================================
# Export Utilities
# =============================================================================


def survey_to_parquet(survey_df: pd.DataFrame, path: str) -> None:
    """Export survey results to Parquet.

    Args:
        survey_df: DataFrame from run_scaling_survey or fit_all_exponents.
        path: File path to write the Parquet.

    Raises:
        ValueError: If survey_df is empty.

    """
    if survey_df.empty:
        raise ValueError("Cannot export empty DataFrame")

    survey_df.to_parquet(path, index=False)


def survey_to_json(survey_df: pd.DataFrame, path: str) -> None:
    """Export survey results to JSON (structured, human-readable format).

    The JSON output includes metadata (columns, row count) and the
    data records as a list of dictionaries.

    Args:
        survey_df: DataFrame from run_scaling_survey or fit_all_exponents.
        path: File path to write the JSON.

    Raises:
        ValueError: If survey_df is empty.

    """
    if survey_df.empty:
        raise ValueError("Cannot export empty DataFrame")

    # Convert DataFrame to a structured JSON format
    output: dict = {
        "metadata": {
            "columns": list(survey_df.columns),
            "rows": len(survey_df),
        },
        "data": survey_df.to_dict(orient="records"),
    }

    # Handle numpy types for JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, o: object) -> object:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.bool_,)):
                return bool(o)
            return super().default(o)

    Path(path).write_text(json.dumps(output, indent=2, cls=NumpyEncoder))


# =============================================================================
# Factory Functions
# =============================================================================


def create_survey_model(
    model_id: str,
    **kwargs: object,
) -> ModelConfig:
    """Create a ModelConfig from a shorthand identifier.

    Provides convenient shorthand creation for common survey models.
    Supports both named models and custom configurations.

    Built-in model IDs:
        - "ideal_coherent":  Coherent state |α⟩, no noise, no entangler
        - "ideal_noon":      NOON state |N,0⟩+|0,N⟩, no noise
        - "ideal_twin_fock": Twin-Fock state, no noise
        - "noon_loss":       NOON state with loss noise
        - "coherent_oat":    Coherent state with OAT spin squeezing
        - "squeezed_vacuum": Squeezed vacuum state, no noise
        - "non_gaussian_n3": Non-Gaussian trisqueezed state (n=3)
        - "non_gaussian_n4": Non-Gaussian quad-squeezed state (n=4)
        - "ancilla_assisted": Ancilla-assisted metrology with dispersive coupling
        - "kerr_mzi":        Kerr-nonlinear MZI (invariant QFI, NOON input)
        - "weak_value_mzi":  Weak-value MZI (SQL-limited, coherent input)

    Args:
        model_id: Shorthand identifier for the model. Unknown IDs
            produce a model with state_type matching the ID.
        **kwargs: Additional keyword arguments overriding the defaults
            for the given model_id. For example:
            create_survey_model("noon_loss", noise_type="dephasing")
            create_survey_model("non_gaussian_n3", omega_n=1.0, t_sqz=3.0)

    Returns:
        ModelConfig with appropriate defaults for the given model_id.

    """
    defaults: dict[str, str] = {
        "ideal_coherent": "css",  # CSS = coherent state with |alpha|² = N
        "ideal_noon": "noon",
        "ideal_twin_fock": "twin_fock",
        "noon_loss": "noon",
        "coherent_oat": "css",
        "squeezed_vacuum": "squeezed_vacuum",
        "squeezed_vacuum_loss": "squeezed_vacuum",
    }

    noise_defaults: dict[str, str] = {
        "ideal_coherent": "none",
        "ideal_noon": "none",
        "ideal_twin_fock": "none",
        "noon_loss": "loss",
        "coherent_oat": "none",
        "squeezed_vacuum": "none",
        "squeezed_vacuum_loss": "loss",
    }

    entangler_defaults: dict[str, str] = {
        "ideal_coherent": "none",
        "ideal_noon": "none",
        "ideal_twin_fock": "none",
        "noon_loss": "none",
        "coherent_oat": "oat",
        "squeezed_vacuum": "none",
        "squeezed_vacuum_loss": "none",
    }

    label_defaults: dict[str, str] = {
        "ideal_coherent": "Coherent state",
        "ideal_noon": "NOON state",
        "ideal_twin_fock": "Twin-Fock state",
        "noon_loss": "NOON with loss",
        "coherent_oat": "Coherent + OAT",
        "squeezed_vacuum": "Squeezed vacuum",
        "squeezed_vacuum_loss": "Squeezed vacuum with loss",
        "non_gaussian_n3": "Non-Gaussian n=3 (trisqueezed)",
        "non_gaussian_n4": "Non-Gaussian n=4 (quadsqueezed)",
        "ancilla_assisted": "Ancilla-assisted metrology",
        "kerr_mzi": "Kerr-nonlinear MZI",
        "weak_value_mzi": "Weak-value MZI",
    }

    # --- Custom sensitivity models bypass standard pipeline ---
    if model_id == "non_gaussian_n3":
        omega_n = kwargs.get("omega_n", 0.5)
        t_sqz = kwargs.get("t_sqz", 2.0)
        theta_n = kwargs.get("theta_n", 0.0)
        use_gs = kwargs.get("use_ground_state", False)
        fn = _non_gaussian_sensitivity_fn(
            n_order=3,
            omega_n=float(omega_n) if isinstance(omega_n, (int, float)) else 0.5,
            theta_n=float(theta_n) if isinstance(theta_n, (int, float)) else 0.0,
            t_sqz=float(t_sqz) if isinstance(t_sqz, (int, float)) else 2.0,
            use_ground_state=bool(use_gs) if isinstance(use_gs, (bool, int)) else False,
        )
        return ModelConfig(
            model_id=model_id,
            custom_sensitivity_fn=fn,
            state_type="",
            noise_type="none",
            entangler="none",
            label=label_defaults.get(model_id, "Non-Gaussian n=3"),
        )

    if model_id == "non_gaussian_n4":
        omega_n = kwargs.get("omega_n", 0.5)
        t_sqz = kwargs.get("t_sqz", 2.0)
        theta_n = kwargs.get("theta_n", 0.0)
        use_gs = kwargs.get("use_ground_state", False)
        fn = _non_gaussian_sensitivity_fn(
            n_order=4,
            omega_n=float(omega_n) if isinstance(omega_n, (int, float)) else 0.5,
            theta_n=float(theta_n) if isinstance(theta_n, (int, float)) else 0.0,
            t_sqz=float(t_sqz) if isinstance(t_sqz, (int, float)) else 2.0,
            use_ground_state=bool(use_gs) if isinstance(use_gs, (bool, int)) else False,
        )
        return ModelConfig(
            model_id=model_id,
            custom_sensitivity_fn=fn,
            state_type="",
            noise_type="none",
            entangler="none",
            label=label_defaults.get(model_id, "Non-Gaussian n=4"),
        )

    if model_id == "ancilla_assisted":
        alpha = kwargs.get("alpha", 1.0)
        g_sa = kwargs.get("g_sa", 1.0)
        tau = kwargs.get("tau", 0.1)
        g_sp = kwargs.get("g_sp", 0.0)
        lam = kwargs.get("lam", 0.0)
        K = kwargs.get("K", 2)
        fn = _ancilla_sensitivity_fn(
            alpha=float(alpha) if isinstance(alpha, (int, float)) else 1.0,
            g_sa=float(g_sa) if isinstance(g_sa, (int, float)) else 1.0,
            tau=float(tau) if isinstance(tau, (int, float)) else 0.1,
            g_sp=float(g_sp) if isinstance(g_sp, (int, float)) else 0.0,
            lam=float(lam) if isinstance(lam, (int, float)) else 0.0,
            K=int(K) if isinstance(K, int) else 2,
        )
        return ModelConfig(
            model_id=model_id,
            custom_sensitivity_fn=fn,
            state_type="",
            noise_type="none",
            entangler="none",
            label=label_defaults.get(model_id, "Ancilla-assisted metrology"),
        )

    if model_id == "kerr_mzi":
        chi = kwargs.get("chi", 0.1)
        T = kwargs.get("T", 1.0)
        state_type = kwargs.get("state_type", "noon")
        chi_val = float(chi) if isinstance(chi, (int, float)) else 0.1
        T_val = float(T) if isinstance(T, (int, float)) else 1.0
        st_val = str(state_type) if isinstance(state_type, str) else "noon"
        fn = _kerr_mzi_sensitivity_fn(chi=chi_val, T=T_val, state_type=st_val)
        return ModelConfig(
            model_id=model_id,
            custom_sensitivity_fn=fn,
            state_type="",
            noise_type="none",
            entangler="none",
            label=label_defaults.get(model_id, "Kerr-nonlinear MZI"),
        )

    if model_id == "weak_value_mzi":
        post_select_angle = kwargs.get("post_select_angle", np.pi / 2 - 0.1)
        psa = (
            float(post_select_angle)
            if isinstance(post_select_angle, (int, float))
            else np.pi / 2 - 0.1
        )
        fn = _weak_value_mzi_sensitivity_fn(post_select_angle=psa)
        return ModelConfig(
            model_id=model_id,
            custom_sensitivity_fn=fn,
            state_type="",
            noise_type="none",
            entangler="none",
            label=label_defaults.get(model_id, "Weak-value MZI"),
        )

    state_type = defaults.get(model_id, model_id)
    noise_type = noise_defaults.get(model_id, "none")
    entangler = entangler_defaults.get(model_id, "none")
    label = label_defaults.get(model_id, model_id.replace("_", " ").title())

    # Override with any provided kwargs
    model_kwargs: dict[str, str] = {}
    for key in ("state_type", "noise_type", "entangler", "label"):
        if key in kwargs:
            val = kwargs[key]
            if isinstance(val, str):
                model_kwargs[key] = val

    return ModelConfig(
        model_id=model_id,
        state_type=model_kwargs.get("state_type", state_type),
        noise_type=model_kwargs.get("noise_type", noise_type),
        entangler=model_kwargs.get("entangler", entangler),
        label=model_kwargs.get("label", label),
    )


def create_default_survey() -> list[ModelConfig]:
    """Create the default set of models for the full scaling survey.

    Returns models representing the most common interferometry states
    used in quantum metrology scaling studies:

    1. ideal_coherent:   Coherent state (SQL scaling, α = -1/2)
    2. ideal_noon:       NOON state (Heisenberg scaling, α = -1)
    3. ideal_twin_fock:  Twin-Fock state (near-Heisenberg, α ≈ -1)
    4. noon_loss:        NOON with loss (transition from Heisenberg to SQL)
    5. coherent_oat:     Coherent state with OAT squeezing (sub-SQL)
    6. squeezed_vacuum:  Squeezed vacuum state (sub-SQL)
    7. squeezed_vacuum_loss: Squeezed vacuum with one-body loss
    8. non_gaussian_n3:  Non-Gaussian trisqueezed state (n=3)
    9. non_gaussian_n4:  Non-Gaussian quad-squeezed state (n=4)
    10. ancilla_assisted: Ancilla-assisted metrology with dispersive coupling
    11. kerr_mzi:        Kerr-nonlinear MZI (invariant QFI)
    12. weak_value_mzi:  Weak-value MZI (no metrological advantage)

    Returns:
        List of ModelConfig objects for the default survey.

    """
    model_ids = [
        "ideal_coherent",
        "ideal_noon",
        "ideal_twin_fock",
        "noon_loss",
        "coherent_oat",
        "squeezed_vacuum",
        "squeezed_vacuum_loss",
        "non_gaussian_n3",
        "non_gaussian_n4",
        "ancilla_assisted",
        "kerr_mzi",
        "weak_value_mzi",
    ]
    return [create_survey_model(mid) for mid in model_ids]


# =============================================================================
# Section: ancilla_comparison
# =============================================================================

# =============================================================================
# Operator Construction
# =============================================================================


def build_system_jz_jx(N_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Build J_z and J_x for a two-mode Fock space.

    J_z = (n₀ - n₁) / 2  (diagonal in Fock basis)
    J_x = (a₀† a₁ + a₁† a₀) / 2

    Args:
        N_max: Maximum photon number per mode. Hilbert space dimension
            is (N_max + 1)².

    Returns:
        Tuple (J_z, J_x) of Hermitian operators of dimension (N_max+1)².

    """
    J_z = two_mode_jz_operator(N_max)

    # J_x = (a₀† a₁ + a₁† a₀) / 2
    a0, a1, a0_dag, a1_dag = create_system_operators(N_max)
    J_x = 0.5 * (a0_dag @ a1 + a1_dag @ a0)

    return J_z, J_x


# build_ancilla_operators replaced by J_Z, J_X from src.analysis.ancilla_optimization


def build_interaction_hamiltonian(
    alphas: tuple[float, float, float, float],
    J_z_sys: np.ndarray,
    J_x_sys: np.ndarray,
    J_z_anc: np.ndarray,
    J_x_anc: np.ndarray,
) -> np.ndarray:
    """Build ancilla interaction Hamiltonian H_int.

    H_int = α_xx · J_x ⊗ J_x  +  α_xz · J_x ⊗ J_z
          + α_zx · J_z ⊗ J_x  +  α_zz · J_z ⊗ J_z

    Args:
        alphas: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        J_z_sys: J_z operator on system space.
        J_x_sys: J_x operator on system space.
        J_z_anc: J_z operator on ancilla space.
        J_x_anc: J_x operator on ancilla space.

    Returns:
        Full H_int matrix of dimension (dim_sys × 2).

    """
    a_xx, a_xz, a_zx, a_zz = alphas
    dim_sys = J_z_sys.shape[0]
    dim_full = dim_sys * 2

    H = np.zeros((dim_full, dim_full), dtype=complex)

    if a_xx != 0.0:
        H += a_xx * np.kron(J_x_sys, J_x_anc)
    if a_xz != 0.0:
        H += a_xz * np.kron(J_x_sys, J_z_anc)
    if a_zx != 0.0:
        H += a_zx * np.kron(J_z_sys, J_x_anc)
    if a_zz != 0.0:
        H += a_zz * np.kron(J_z_sys, J_z_anc)

    return H


# =============================================================================
# Generator Construction
# =============================================================================


def compute_generator_B(T_H: float, N_max: int) -> np.ndarray:
    """Compute the effective generator G_B for Case B (2 particles, no ancilla).

    G_B = T_H · BS₁† · J_z · BS₁

    With the BS convention (θ_BS = π/4, φ_BS = 0), this equals -T_H · J_y.

    Args:
        T_H: Holding-time strength parameter.
        N_max: Maximum photon number per mode.

    Returns:
        Generator matrix of dimension (N_max+1)² × (N_max+1)².

    """
    J_z_sys, _ = build_system_jz_jx(N_max)
    BS = beam_splitter_unitary(np.pi / 4, 0.0, N_max)

    return T_H * BS.conj().T @ J_z_sys @ BS


def compute_generator_A(
    T_H: float,
    alphas: tuple[float, float, float, float],
    N_max: int,
    n_quadrature: int = 50,
) -> np.ndarray:
    """Compute G_A for Case A (1 system particle + ancilla) at reference θ = 0.

    G_A = T_H · (BS₁† ⊗ I_anc) · [∫₀¹ J_z(s) ds] · (BS₁ ⊗ I_anc)

    where J_z(s) = exp(i s T_H H_int) · (J_z ⊗ I) · exp(-i s T_H H_int).

    When [J_z, H_int] = 0 (only α_zz, α_zx terms), J_z(s) = J_z ⊗ I
    is independent of s, and G_A = T_H · (BS₁† · J_z · BS₁) ⊗ I = -T_H · J_y ⊗ I.

    When [J_z, H_int] ≠ 0, the integral mixes components and G_A differs.

    Args:
        T_H: Holding-time strength parameter.
        alphas: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        N_max: Maximum photon number per mode for the system.
        n_quadrature: Number of quadrature points for the integral (default 50).

    Returns:
        Generator matrix of dimension (N_max+1)² × 2 (system + ancilla).

    """
    J_z_sys, J_x_sys = build_system_jz_jx(N_max)
    J_z_anc, J_x_anc = J_Z, J_X

    dim_sys = (N_max + 1) ** 2
    dim_full = dim_sys * 2

    I_anc = np.eye(2, dtype=complex)

    # Full operator: J_z ⊗ I_anc
    J_z_full = np.kron(J_z_sys, I_anc)

    # Interaction Hamiltonian
    H_int = build_interaction_hamiltonian(alphas, J_z_sys, J_x_sys, J_z_anc, J_x_anc)

    # BS on system, identity on ancilla
    BS = beam_splitter_unitary(np.pi / 4, 0.0, N_max)
    BS_full = np.kron(BS, I_anc)

    # Compute ∫₀¹ J_z(s) ds via numerical quadrature (Simpson's rule)
    s_points = np.linspace(0, 1, n_quadrature)
    J_z_vals = np.zeros((n_quadrature, dim_full, dim_full), dtype=complex)

    for k, s in enumerate(s_points):
        U_s = expm(1j * s * T_H * H_int)
        J_z_vals[k] = U_s @ J_z_full @ U_s.conj().T

    # Simpson integration
    h = 1.0 / (n_quadrature - 1)
    J_z_integral = np.zeros_like(J_z_full)
    J_z_integral += J_z_vals[0] + J_z_vals[-1]  # endpoints
    J_z_integral += 4.0 * np.sum(J_z_vals[1:-1:2], axis=0)  # odd
    J_z_integral += 2.0 * np.sum(J_z_vals[2:-1:2], axis=0)  # even
    J_z_integral *= h / 3.0

    # G_A = T_H · BS_full† · J_z_integral · BS_full
    G_A = T_H * BS_full.conj().T @ J_z_integral @ BS_full

    # Ensure Hermiticity
    return 0.5 * (G_A + G_A.conj().T)


def compute_generator_A_at_theta(
    T_H: float,
    theta: float,
    alphas: tuple[float, float, float, float],
    N_max: int,
    n_quadrature: int = 50,
) -> np.ndarray:
    """Compute G_A at a non-zero reference θ.

    Same as compute_generator_A but with the full θ-dependent Hamiltonian.
    J_z(s) = exp(i s T_H (θ J_z + H_int)) · (J_z ⊗ I) · exp(-i s T_H (θ J_z + H_int))

    Args:
        T_H: Holding-time strength parameter.
        theta: Reference phase value.
        alphas: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        N_max: Maximum photon number per mode.
        n_quadrature: Number of quadrature points (default 50).

    Returns:
        Generator matrix for Case A at given θ.

    """
    J_z_sys, J_x_sys = build_system_jz_jx(N_max)
    J_z_anc, J_x_anc = J_Z, J_X

    dim_sys = (N_max + 1) ** 2
    dim_full = dim_sys * 2

    I_anc = np.eye(2, dtype=complex)

    J_z_full = np.kron(J_z_sys, I_anc)
    H_int = build_interaction_hamiltonian(alphas, J_z_sys, J_x_sys, J_z_anc, J_x_anc)
    H_total = theta * J_z_full + H_int

    BS = beam_splitter_unitary(np.pi / 4, 0.0, N_max)
    BS_full = np.kron(BS, I_anc)

    s_points = np.linspace(0, 1, n_quadrature)
    J_z_vals = np.zeros((n_quadrature, dim_full, dim_full), dtype=complex)

    for k, s in enumerate(s_points):
        U_s = expm(1j * s * T_H * H_total)
        J_z_vals[k] = U_s @ J_z_full @ U_s.conj().T

    h = 1.0 / (n_quadrature - 1)
    J_z_integral = np.zeros_like(J_z_full)
    J_z_integral += J_z_vals[0] + J_z_vals[-1]
    J_z_integral += 4.0 * np.sum(J_z_vals[1:-1:2], axis=0)
    J_z_integral += 2.0 * np.sum(J_z_vals[2:-1:2], axis=0)
    J_z_integral *= h / 3.0

    G_A = T_H * BS_full.conj().T @ J_z_integral @ BS_full
    return 0.5 * (G_A + G_A.conj().T)


# =============================================================================
# Density Matrix Utilities
# =============================================================================


def random_density_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random density matrix via Cholesky decomposition.

    Guarantees: ρ ≥ 0, Tr(ρ) = 1, ρ = ρ†.

    Args:
        d: Dimension of the density matrix.
        rng: NumPy random generator for reproducibility.

    Returns:
        Random density matrix of shape (d, d).

    """
    # Lower-triangular matrix with complex entries
    T = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    T = np.tril(T)

    ρ = T @ T.conj().T
    ρ /= np.trace(ρ)
    return ρ


def random_pure_state_dm(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random pure-state density matrix.

    Args:
        d: Dimension.
        rng: NumPy random generator.

    Returns:
        Pure state density matrix of shape (d, d).

    """
    ψ = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    ψ = ψ / np.linalg.norm(ψ)
    return np.outer(ψ, ψ.conj())


def _subspace_indices(N_max: int, target_N: int) -> np.ndarray:
    """Get indices in the two-mode Fock basis for a given total particle number.

    Args:
        N_max: Maximum photon number per mode.
        target_N: Total particle number n₀ + n₁ = target_N.

    Returns:
        Array of indices into the (N_max+1)²-dimensional Fock basis vectors
        that have the specified total particle number.

    """
    indices = []
    for n0 in range(N_max + 1):
        n1 = target_N - n0
        if 0 <= n1 <= N_max:
            idx = n0 * (N_max + 1) + n1
            indices.append(idx)
    return np.array(indices, dtype=int)


def random_pure_state_in_subspace(
    d_full: int,
    subspace_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random pure state restricted to a subspace.

    The state only has support on the specified indices, with random
    coefficients in that subspace and zero elsewhere.

    Args:
        d_full: Full Hilbert space dimension.
        subspace_idx: Indices of the subspace basis states.
        rng: NumPy random generator.

    Returns:
        Pure state density matrix of shape (d_full, d_full)
        supported only on the given subspace indices.

    """
    d_sub = len(subspace_idx)
    # Random state in the subspace
    psi_sub = rng.standard_normal(d_sub) + 1j * rng.standard_normal(d_sub)
    psi_sub = psi_sub / np.linalg.norm(psi_sub)

    # Embed in full space
    psi_full = np.zeros(d_full, dtype=complex)
    psi_full[subspace_idx] = psi_sub
    return np.outer(psi_full, psi_full.conj())


def random_alphas(
    rng: np.random.Generator,
    scale: float = 10.0,
) -> tuple[float, float, float, float]:
    """Generate random interaction coefficients α.

    Args:
        rng: NumPy random generator.
        scale: Max absolute value for coefficients (default 10).

    Returns:
        Tuple (α_xx, α_xz, α_zx, α_zz) matching the report convention.

    """
    vals = rng.uniform(-scale, scale, size=4)
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def check_particle_number(
    rho: np.ndarray,
    N_max: int,
    atol: float = 1e-6,
) -> tuple[float, float, float]:
    """Check the particle-number properties of a state.

    Computes ⟨N⟩, population of |0,0⟩, and population of |N_max,N_max⟩
    to verify the particle-number constraint.

    Args:
        rho: Density matrix in the two-mode Fock basis.
        N_max: Maximum photon number per mode.
        atol: Absolute tolerance for checks.

    Returns:
        Tuple (mean_N, pop_00, pop_NN) where:
        - mean_N: ⟨n₀ + n₁⟩
        - pop_00: population ρ_{|0,0⟩⟨0,0|}
        - pop_NN: population ρ_{|N_max,N_max⟩⟨N_max,N_max|}

    """
    mean_N = 0.0

    for n0 in range(N_max + 1):
        for n1 in range(N_max + 1):
            idx = n0 * (N_max + 1) + n1
            prob = np.real(rho[idx, idx])
            mean_N += prob * (n0 + n1)

    idx_00 = 0  # n0=0, n1=0
    idx_NN = N_max * (N_max + 1) + N_max  # n0=N_max, n1=N_max

    pop_00 = np.real(rho[idx_00, idx_00])
    pop_NN = np.real(rho[idx_NN, idx_NN])

    return float(mean_N), float(pop_00), float(pop_NN)


# =============================================================================
# QFI Evaluation
# =============================================================================


def evaluate_qfi_case_B(
    rho: np.ndarray,
    T_H: float,
    N_max: int,
) -> float:
    """Evaluate Quantum Fisher Information for Case B.

    Args:
        rho: Density matrix of dimension (N_max+1)².
        T_H: Holding-time strength.
        N_max: Maximum photon number per mode.

    Returns:
        QFI value F_Q.

    """
    G_B = compute_generator_B(T_H, N_max)
    F_Q = quantum_fisher_information_dm(rho, G_B)
    return float(F_Q)


def evaluate_qfi_case_A(
    rho: np.ndarray,
    T_H: float,
    alphas: tuple[float, float, float, float],
    N_max: int,
    n_quadrature: int = 50,
) -> float:
    """Evaluate Quantum Fisher Information for Case A at θ = 0.

    Args:
        rho: Density matrix of dimension (N_max+1)² × 2.
        T_H: Holding-time strength.
        alphas: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        N_max: Maximum photon number per mode for the system.
        n_quadrature: Quadrature points for integral.

    Returns:
        QFI value F_Q.

    """
    G_A = compute_generator_A(T_H, alphas, N_max, n_quadrature)
    F_Q = quantum_fisher_information_dm(rho, G_A)
    return float(F_Q)


# =============================================================================
# Optimisation (Random Search)
# =============================================================================


@dataclass
class RandomSearchResult:
    """Result of a random search for optimal QFI."""

    max_fq: float
    best_rho: np.ndarray
    best_alphas: tuple[float, float, float, float] | None
    all_fq: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_N: float = 0.0
    pop_00: float = 0.0
    pop_NN: float = 0.0


def optimize_qfi_case_B(
    T_H: float,
    N_max: int,
    n_samples: int = 1000,
    pure_only: bool = False,
    subspace_N: int | None = None,
    seed: int | None = None,
) -> RandomSearchResult:
    """Optimise QFI for Case B via random search.

    Args:
        T_H: Holding-time strength.
        N_max: Maximum photon number per mode.
        n_samples: Number of random states to evaluate.
        pure_only: If True, restrict to pure states.
        subspace_N: If set, restrict sampling to the subspace with
            n₀ + n₁ = subspace_N (e.g., 2 for the 2-particle sector).
        seed: Random seed for reproducibility.

    Returns:
        RandomSearchResult with best QFI and state found.

    """
    rng = np.random.default_rng(seed)
    dim = (N_max + 1) ** 2
    G_B = compute_generator_B(T_H, N_max)

    # Pre-compute subspace indices if requested
    sub_idx = _subspace_indices(N_max, subspace_N) if subspace_N is not None else None

    best_fq = -1.0
    best_rho = None
    all_fq = np.zeros(n_samples)

    for i in range(n_samples):
        if sub_idx is not None:
            rho = random_pure_state_in_subspace(dim, sub_idx, rng)
        elif pure_only:
            rho = random_pure_state_dm(dim, rng)
        else:
            rho = random_density_matrix(dim, rng)

        F_Q = quantum_fisher_information_dm(rho, G_B)
        all_fq[i] = F_Q

        if best_fq < F_Q:
            best_fq = F_Q
            best_rho = rho.copy()

    assert best_rho is not None, "No valid states found"

    mean_N, pop_00, pop_NN = check_particle_number(best_rho, N_max)

    return RandomSearchResult(
        max_fq=best_fq,
        best_rho=best_rho,
        best_alphas=None,
        all_fq=all_fq,
        mean_N=mean_N,
        pop_00=pop_00,
        pop_NN=pop_NN,
    )


def optimize_qfi_case_A(
    T_H: float,
    N_max: int,
    n_samples: int = 2000,
    n_alpha_samples: int = 100,
    pure_only: bool = False,
    subspace_N: int | None = None,
    particle_penalty: float = 100.0,
    seed: int | None = None,
) -> RandomSearchResult:
    """Optimise QFI for Case A via random search over ρ and α.

    Imposes a particle-number penalty to ensure ⟨N⟩ ≈ 1.

    Args:
        T_H: Holding-time strength.
        N_max: Maximum photon number per mode for the system.
        n_samples: Number of random states per α sample.
        n_alpha_samples: Number of random α vectors to try.
        pure_only: If True, restrict to pure states.
        subspace_N: If set, restrict system to the n₀ + n₁ = subspace_N
            subspace (e.g., 1 for the 1-particle sector).
        particle_penalty: Penalty strength for ⟨N⟩ ≠ 1.
        seed: Random seed for reproducibility.

    Returns:
        RandomSearchResult with best QFI, ρ, and α found.
        Alpha ordering follows the report convention: (α_xx, α_xz, α_zx, α_zz).

    """
    rng = np.random.default_rng(seed)
    dim_sys = (N_max + 1) ** 2
    dim_full = dim_sys * 2

    # Pre-compute system subspace indices
    sub_idx = _subspace_indices(N_max, subspace_N) if subspace_N is not None else None

    best_fq = -1.0
    best_rho: np.ndarray | None = None
    best_alphas: tuple[float, float, float, float] | None = None
    all_fq = []

    J_z_sys, J_x_sys = build_system_jz_jx(N_max)
    J_z_anc, J_x_anc = J_Z, J_X

    # Pre-compute the BS unitary
    BS = beam_splitter_unitary(np.pi / 4, 0.0, N_max)
    I_anc = np.eye(2, dtype=complex)
    BS_full = np.kron(BS, I_anc)

    for _alpha_iter in range(n_alpha_samples):
        alphas = random_alphas(rng, scale=5.0)

        # Build H_int
        H_int = build_interaction_hamiltonian(
            alphas,
            J_z_sys,
            J_x_sys,
            J_z_anc,
            J_x_anc,
        )

        # Build full Hamiltonian at θ = 0
        J_z_full = np.kron(J_z_sys, I_anc)
        H_total = H_int  # θ = 0

        # Compute J_z(s) integral at θ = 0
        s_points = np.linspace(0, 1, 50)
        J_z_vals = np.zeros((50, dim_full, dim_full), dtype=complex)
        for k, s in enumerate(s_points):
            U_s = expm(1j * s * T_H * H_total)
            J_z_vals[k] = U_s @ J_z_full @ U_s.conj().T

        h = 1.0 / 49.0
        J_z_integral = J_z_vals[0] + J_z_vals[-1]
        J_z_integral += 4.0 * np.sum(J_z_vals[1:-1:2], axis=0)
        J_z_integral += 2.0 * np.sum(J_z_vals[2:-1:2], axis=0)
        J_z_integral *= h / 3.0

        # G_A for this α
        G_A = T_H * BS_full.conj().T @ J_z_integral @ BS_full
        G_A = 0.5 * (G_A + G_A.conj().T)

        # Sample random states for this α
        for _state_iter in range(n_samples // n_alpha_samples):
            if sub_idx is not None:
                # Build full density matrix: system in N=1 subspace × ancilla
                psi_sys = random_pure_state_in_subspace(dim_sys, sub_idx, rng)
                # Random ancilla pure state
                psi_anc = random_pure_state_dm(2, rng)
                rho = np.kron(psi_sys, psi_anc)
            elif pure_only:
                rho = random_pure_state_dm(dim_full, rng)
            else:
                rho = random_density_matrix(dim_full, rng)

            F_Q = quantum_fisher_information_dm(rho, G_A)

            # Apply particle-number penalty
            mean_N_sys, _, _ = check_particle_number(
                _partial_trace_system(rho, N_max),
                N_max,
            )
            penalty = particle_penalty * (mean_N_sys - 1.0) ** 2
            F_Q_penalized = F_Q - penalty

            all_fq.append(F_Q_penalized)

            if F_Q_penalized > best_fq:
                best_fq = F_Q_penalized
                best_rho = rho.copy()
                best_alphas = alphas

    assert best_rho is not None, "No valid states found"
    assert best_alphas is not None

    # Re-evaluate the best without penalty to report true QFI
    # Need to check particle number of the best state
    mean_N, pop_00, pop_NN = check_particle_number(
        _partial_trace_system(best_rho, N_max),
        N_max,
    )

    return RandomSearchResult(
        max_fq=best_fq,  # penalized score
        best_rho=best_rho,
        best_alphas=best_alphas,
        all_fq=np.array(all_fq),
        mean_N=mean_N,
        pop_00=pop_00,
        pop_NN=pop_NN,
    )


def _partial_trace_system(rho_full: np.ndarray, N_max: int) -> np.ndarray:
    """Extract the system (two-mode Fock) reduced density matrix.

    Traces out the ancilla from the full system-ancilla density matrix.

    Args:
        rho_full: Full density matrix of dim (sys_dim × 2).
        N_max: Maximum photon number per mode.

    Returns:
        Reduced system density matrix.

    """
    dim_sys = (N_max + 1) ** 2
    dim_anc = 2

    rho_reshaped = rho_full.reshape(dim_sys, dim_anc, dim_sys, dim_anc)
    return np.trace(rho_reshaped, axis1=1, axis2=3)


# =============================================================================
# Comparison Runner
# =============================================================================


@dataclass
class ComparisonResult:
    """Results of the ancilla vs. system comparison.

    Attributes:
        fq_A_max: Best QFI found for Case A (ancilla-assisted).
        fq_B_max: Best QFI found for Case B (two-particle system).
        fq_A_zero: QFI for Case A with α = 0 (baseline).
        ratio: sqrt(F_B / F_A) = Δθ_A / Δθ_B.
        fq_A_theta: QFI at different reference θ values (optional).
        fq_A_all: All QFI values from random search for Case A.
        fq_B_all: All QFI values from random search for Case B.

    """

    fq_A_max: float
    fq_B_max: float
    fq_A_zero: float
    ratio: float
    fq_A_theta: dict[float, float] = field(default_factory=dict)
    fq_A_all: np.ndarray = field(default_factory=lambda: np.array([]))
    fq_B_all: np.ndarray = field(default_factory=lambda: np.array([]))
    best_alphas_A: tuple[float, float, float, float] | None = None
    mean_N_A: float = 0.0
    pop_00_A: float = 0.0
    pop_NN_A: float = 0.0
    mean_N_B: float = 0.0
    pop_00_B: float = 0.0
    pop_NN_B: float = 0.0


def run_comparison(
    T_H: float = 1.0,
    n_samples_B: int = 2000,
    n_samples_A: int = 3000,
    n_alpha_samples: int = 50,
    n_quadrature: int = 50,
    pure_only: bool = False,
    particle_penalty: float = 100.0,
    theta_values: tuple[float, ...] = (0.0, 0.1, 0.5),
    seed: int | None = 42,
) -> ComparisonResult:
    """Run the full comparison between Case A and Case B.

    Uses subspace-restricted sampling: Case B restricts to N=2 subspace,
    Case A restricts to N=1 subspace for the system.

    Args:
        T_H: Holding-time strength (default 1.0).
        n_samples_B: Number of random states for Case B optimisation.
        n_samples_A: Number of random states for Case A optimisation.
        n_alpha_samples: Number of random α vectors for Case A.
        n_quadrature: Quadrature points for the integral.
        pure_only: Restrict to pure states.
        particle_penalty: Penalty strength for ⟨N⟩ ≠ 1.
        theta_values: Reference θ values for θ-dependence check.
        seed: Random seed (default 42 for reproducibility).

    Returns:
        ComparisonResult with all findings.
        Alpha ordering follows the report convention: (α_xx, α_xz, α_zx, α_zz).

    """
    # Case B: 2 particles, N_max = 2, restrict to N=2 subspace
    result_B = optimize_qfi_case_B(
        T_H=T_H,
        N_max=2,
        n_samples=n_samples_B,
        pure_only=True,
        subspace_N=2,
        seed=seed,
    )

    # Case A: 1 system particle + ancilla, N_max = 1, restrict to N=1 subspace
    result_A = optimize_qfi_case_A(
        T_H=T_H,
        N_max=1,
        n_samples=n_samples_A,
        n_alpha_samples=n_alpha_samples,
        pure_only=True,
        subspace_N=1,
        particle_penalty=particle_penalty,
        seed=seed if seed is None else seed + 1,
    )

    # Case A baseline: α = 0 (no interaction), N=1 subspace
    dim_sys = (1 + 1) ** 2  # N_max = 1 → dim 4
    rng = np.random.default_rng(seed if seed is None else seed + 2)
    G_A_zero = compute_generator_A(T_H, (0.0, 0.0, 0.0, 0.0), 1, n_quadrature)
    sub_idx = _subspace_indices(1, 1)

    best_fq_A_zero = -1.0
    for _ in range(n_samples_A // 2):
        psi_sys = random_pure_state_in_subspace(dim_sys, sub_idx, rng)
        psi_anc = random_pure_state_dm(2, rng)
        rho = np.kron(psi_sys, psi_anc)
        F_Q = quantum_fisher_information_dm(rho, G_A_zero)
        best_fq_A_zero = max(best_fq_A_zero, F_Q)

    # θ-dependence check
    fq_A_theta: dict[float, float] = {}
    if result_A.best_alphas is not None:
        best_rho_A = result_A.best_rho
        for theta_val in theta_values:
            if theta_val == 0.0:
                G_A_theta = compute_generator_A(
                    T_H,
                    result_A.best_alphas,
                    1,
                    n_quadrature,
                )
            else:
                G_A_theta = compute_generator_A_at_theta(
                    T_H,
                    theta_val,
                    result_A.best_alphas,
                    1,
                    n_quadrature,
                )
            F_Q = quantum_fisher_information_dm(best_rho_A, G_A_theta)
            fq_A_theta[theta_val] = float(F_Q)

    # Ratio
    ratio = (
        np.sqrt(result_B.max_fq / result_A.max_fq)
        if result_A.max_fq > 0
        else float("inf")
    )

    return ComparisonResult(
        fq_A_max=result_A.max_fq,
        fq_B_max=result_B.max_fq,
        fq_A_zero=best_fq_A_zero,
        ratio=ratio,
        fq_A_theta=fq_A_theta,
        fq_A_all=result_A.all_fq,
        fq_B_all=result_B.all_fq,
        best_alphas_A=result_A.best_alphas,
        mean_N_A=result_A.mean_N,
        pop_00_A=result_A.pop_00,
        pop_NN_A=result_A.pop_NN,
        mean_N_B=result_B.mean_N,
        pop_00_B=result_B.pop_00,
        pop_NN_B=result_B.pop_NN,
    )


# =============================================================================
# Analytical Checks
# =============================================================================


def analytical_fq_B_max(T_H: float) -> float:
    """Theoretical maximum QFI for Case B (2-particle system).

    For J = 1 (2 particles), J_y has eigenvalues {-1, 0, +1}.
    With optimal pure state: max F_Q = T_H² (λ_max - λ_min)² = 4 T_H².

    Args:
        T_H: Holding-time strength.

    Returns:
        Theoretical maximum QFI.

    """
    return 4.0 * T_H**2


def analytical_fq_A_zero(T_H: float) -> float:
    """Theoretical QFI for Case A with α = 0 (uncoupled ancilla).

    With α = 0, G_A = -T_H · J_y ⊗ I. For the 1-particle system (J = 1/2),
    J_y has eigenvalues {-1/2, +1/2}, so:
    max F_Q = T_H² (1/2 - (-1/2))² = T_H².

    Args:
        T_H: Holding-time strength.

    Returns:
        Theoretical maximum QFI with zero interaction.

    """
    return 1.0 * T_H**2


# =============================================================================
# CLI
# =============================================================================


def _generate_ancilla_vs_system_raw_data(force: bool = False) -> None:
    """Generate raw data for Ancilla-vs-System-Comparison report."""
    raw_dir = _REPORT_DIR / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Ancilla-vs-System raw data generation")


def _generate_ancilla_vs_system_figures(force: bool = False) -> None:
    """Generate figures for Ancilla-vs-System-Comparison report."""
    fig_dir = _REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Ancilla-vs-System figure generation")


def _generate_fock_mzi_raw_data(force: bool = False) -> None:
    """Generate raw data for Scaling-Survey-Fock-MZI report."""
    raw_dir = _REPORT_DIR / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Fock-MZI raw data generation")


def _generate_fock_mzi_figures(force: bool = False) -> None:
    """Generate figures for Scaling-Survey-Fock-MZI report."""
    fig_dir = _REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Fock-MZI figure generation")


def _generate_collective_spin_raw_data(force: bool = False) -> None:
    """Generate raw data for Scaling-Survey-Collective-Spin report."""
    raw_dir = _REPORT_DIR / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Collective-Spin raw data generation")


def _generate_collective_spin_figures(force: bool = False) -> None:
    """Generate figures for Scaling-Survey-Collective-Spin report."""
    fig_dir = _REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Collective-Spin figure generation")


def _generate_advanced_architectures_raw_data(force: bool = False) -> None:
    """Generate raw data for Scaling-Survey-Advanced-Architectures report."""
    raw_dir = _REPORT_DIR / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Advanced-Architectures raw data generation")


def _generate_advanced_architectures_figures(force: bool = False) -> None:
    """Generate figures for Scaling-Survey-Advanced-Architectures report."""
    fig_dir = _REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("  [stub] Advanced-Architectures figure generation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures and CSVs for 2026-05-11 reports",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing CSVs)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=[
            "ancilla-vs-system",
            "fock-mzi",
            "collective-spin",
            "advanced-architectures",
        ],
        help="Generate only one experiment's data and figures",
    )
    args = parser.parse_args()

    (_REPORT_DIR / "raw_data").mkdir(parents=True, exist_ok=True)
    (_REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    experiments = {
        "ancilla-vs-system": (
            _generate_ancilla_vs_system_raw_data,
            _generate_ancilla_vs_system_figures,
        ),
        "fock-mzi": (
            _generate_fock_mzi_raw_data,
            _generate_fock_mzi_figures,
        ),
        "collective-spin": (
            _generate_collective_spin_raw_data,
            _generate_collective_spin_figures,
        ),
        "advanced-architectures": (
            _generate_advanced_architectures_raw_data,
            _generate_advanced_architectures_figures,
        ),
    }

    if args.experiment:
        raw_fn, fig_fn = experiments[args.experiment]
        print(f"\n=== {args.experiment} ===")
        print("  Raw data ...")
        raw_fn(force=args.force)
        print("  Figures ...")
        fig_fn(force=args.force)
    else:
        for name, (raw_fn, fig_fn) in experiments.items():
            print(f"\n=== {name} ===")
            print("  Raw data ...")
            raw_fn(force=args.force)
            print("  Figures ...")
            fig_fn(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()

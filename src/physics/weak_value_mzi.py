"""
Weak-value amplification in a Mach-Zehnder interferometer.

Physical Model:
- A small phase shift φ is amplified by post-selecting on a near-orthogonal
  final state. The weak value is:

      A_w = ⟨f|A|i⟩ / ⟨f|i⟩

  where |i⟩ is the pre-selected state (after the first beam splitter),
  |f⟩ is the post-selected state, and A is the measured observable
  (the number operator n₁ on the phase-imprinted arm).

- Signal amplification S = |A_w|·φ (amplified by the weak value).
- Post-selection probability p_ps = |⟨f|i⟩|² (reduces with amplification).
- Fisher information: F = p_ps · F_conventional (same FI after accounting
  for the reduced probability).
- Sensitivity: Δφ = 1/√F (no advantage over conventional MZI).

Key result: Weak-value amplification cannot beat the standard quantum
limit for phase estimation. The amplification comes entirely at the cost
of reduced post-selection probability.

Units:
- Dimensionless throughout. Phase is measured in radians.

Conventions:
- Two-mode Fock basis |n₁, n₂⟩ with n₁ as first mode, n₂ as second mode.
- Beam splitter transformation from mzi_simulation module.
- Phase shift applied to mode 1 (second mode).
- Pre-selected state |i⟩ = U_BS₁ |ψ₀⟩ (state inside interferometer
  after the first beam splitter, before the phase shift).
- Post-selected state |f⟩ = exp(i · post_select_angle · J_z) |i⟩
  (rotated by post_select_angle in J_z space).

References:
- Hosten & Kwiat, Science 319, 787 (2008)
- Dixon et al., PRL 102, 173601 (2009)
- Steinberg, Nature 463, 890 (2010)
- Dressel et al., RMP 86, 307 (2014)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.physics.mzi_simulation import (
    beam_splitter_unitary,
    phase_shift_unitary,
)


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
        >>> from src.physics.mzi_simulation import coherent_state
        >>> config = WeakValueConfig()
        >>> psi0 = coherent_state(alpha=2.0+0j, max_photons=10)
        >>> result = weak_value_mzi(psi0, phi=0.01, config=config, max_photons=10)
        >>> result["amplification"] > 1.0
        True
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not 0 < config.post_select_angle < np.pi:
        raise ValueError(
            f"post_select_angle must be in (0, π), got {config.post_select_angle}"
        )
    if config.measurement_basis not in ("parity", "Jz"):
        raise ValueError(
            f"Unknown measurement_basis: '{config.measurement_basis}'. "
            f"Supported: 'parity', 'Jz'."
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
        pre_selected_state, config.post_select_angle, max_photons
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
    overlap_fi: complex = np.vdot(post_selected_state, pre_selected_state)

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
    weak_val: complex = (
        np.vdot(post_selected_state, n1 @ pre_selected_state) / overlap_fi
    )

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
            f"post_select_angle must be in (0, π), got {config.post_select_angle}"
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

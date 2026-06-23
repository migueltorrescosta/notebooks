"""Weak-value amplification MZI simulation.

Implements the weak-value amplification protocol for the Mach-Zehnder
interferometer, including both full numerical simulation and analytical
sensitivity formulas for coherent state inputs.

Extracted from src/analysis/scaling_survey.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.physics.mzi_simulation import beam_splitter_unitary, phase_shift_unitary


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


# Operator Construction


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


# Internal Helpers


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


def _compute_pre_post_states(
    initial_state: np.ndarray,
    config: WeakValueConfig,
    U_bs: np.ndarray,
    max_photons: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pre-selected and post-selected states for weak-value MZI.

    The pre-selected state |i⟩ is the state after the first beam splitter.
    The post-selected state |f⟩ is obtained by rotating |i⟩ by
    post_select_angle in J_z space. When the angle approaches :math:`\\pi/2`,
    |f⟩ is nearly orthogonal to |i⟩, yielding large amplification.

    J_z is diagonal in the Fock basis, so the rotation is computed as
    O(dim) phase applications, not as a dense matrix exponential.
    """
    pre_selected_state: np.ndarray = U_bs @ initial_state
    post_selected_state: np.ndarray = _apply_jz_rotation(
        pre_selected_state,
        config.post_select_angle,
        max_photons,
    )
    post_selected_state /= np.linalg.norm(post_selected_state)
    return pre_selected_state, post_selected_state


def _compute_weak_value(
    post_selected_state: np.ndarray,
    pre_selected_state: np.ndarray,
    n1: np.ndarray,
    overlap_fi: complex,
) -> complex:
    """Compute the weak value from pre/post-selected states.

    A_w = :math:`\\langle f | n_1 | i \\rangle / \\langle f | i \\rangle`
    represents the effective amplification of the phase shift in the
    post-selected outcome.
    """
    weak_val_np = np.vdot(post_selected_state, n1 @ pre_selected_state) / overlap_fi
    weak_val: complex = complex(weak_val_np.item())

    # Physics assertion: the weak value should be predominantly real
    # for the standard MZI configuration.
    if abs(np.real(weak_val)) > 1e-10:
        imag_ratio = abs(np.imag(weak_val)) / abs(np.real(weak_val))
        assert imag_ratio < 5.0, (
            f"Weak value has large imaginary component relative to real part: "
            f"Re(A_w)={np.real(weak_val):.4f}, Im(A_w)={np.imag(weak_val):.4f}, "
            f"|Im/Re|={imag_ratio:.2f}. This may indicate an invalid post-selection "
            f"configuration."
        )
    return weak_val


def _build_weak_value_result(
    weak_val: complex,
    overlap_fi: complex,
    mean_n: float,
    phi_phase: float,
    output_state: np.ndarray,
) -> dict[str, Any]:
    """Build result dict for weak-value MZI simulation.

    Computes all derived metrics from the core weak-value quantities:
    amplification, post-selection probability, amplified signal, Fisher
    information, and phase sensitivities.
    """
    amplification: float = abs(weak_val)
    post_selection_prob: float = abs(overlap_fi) ** 2
    signal: float = amplification * abs(phi_phase)
    fisher_information: float = 4.0 * post_selection_prob * (amplification**2)

    delta_phi: float = (
        np.inf if fisher_information <= 0 else 1.0 / np.sqrt(fisher_information)
    )
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


# Full Numerical Weak-Value MZI Simulation


def weak_value_mzi(
    initial_state: np.ndarray,
    phi_phase: float,
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
        phi_phase: Phase shift in mode 1 (the parameter to estimate), in
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
        >>> result = weak_value_mzi(psi0, phi_phase=0.01, config=config, max_photons=10)
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
    # Pre-selected and post-selected states
    # ------------------------------------------------------------------
    pre_selected_state, post_selected_state = _compute_pre_post_states(
        initial_state,
        config,
        U_bs,
        max_photons,
    )

    # ------------------------------------------------------------------
    # Phase shift and output state
    # ------------------------------------------------------------------
    U_phi = phase_shift_unitary(phi_phase, max_photons)
    output_state: np.ndarray = U_bs @ U_phi @ pre_selected_state

    # ------------------------------------------------------------------
    # Overlap between post-selected and pre-selected states
    #
    # ⟨f|i⟩ determines the amplification. When post_select_angle ≈ π/2,
    # this overlap is small, giving large |A_w|.
    # ------------------------------------------------------------------
    overlap_fi: complex = complex(np.vdot(post_selected_state, pre_selected_state))

    # Guard against division by zero (exactly orthogonal case)
    if np.isclose(abs(overlap_fi), 0.0, atol=1e-15):
        return {
            "weak_value": complex(np.inf, 0),
            "amplification": np.inf,
            "post_selection_prob": 0.0,
            "signal": np.inf if phi_phase != 0 else 0.0,
            "fisher_information": 0.0,
            "delta_phi": np.inf,
            "delta_phi_conventional": np.inf,
            "final_state": output_state,
        }

    # ------------------------------------------------------------------
    # Weak value: A_w = ⟨f|n₁|i⟩ / ⟨f|i⟩
    # ------------------------------------------------------------------
    weak_val: complex = _compute_weak_value(
        post_selected_state,
        pre_selected_state,
        n1,
        overlap_fi,
    )

    # ------------------------------------------------------------------
    # Mean photon number and derived metrics
    # ------------------------------------------------------------------
    mean_n: float = _mean_photon_number(initial_state, max_photons)

    return _build_weak_value_result(
        weak_val, overlap_fi, mean_n, phi_phase, output_state
    )


# Analytical Sensitivity for Coherent State Input


def _compute_analytical_sensitivity_metrics(
    N: float,
    delta: float,
    phi_phase: float,
) -> dict[str, Any]:
    """Compute analytical sensitivity metrics for weak-value MZI.

    Closed-form expressions for all sensitivity metrics with a coherent
    state input and deviation :math:`\\delta = \\pi/2 - \\text{post\\_select\\_angle}`
    from perfect orthogonality.

    The key result: Fisher information F = N · cos²(δ) ≤ N, proving that
    weak-value amplification cannot beat the standard quantum limit.
    """
    cot_delta: float = 1.0 / np.tan(delta)
    weak_val: complex = complex(cot_delta, 0.0)
    amplification: float = cot_delta
    sin_delta: float = np.sin(delta)
    post_selection_prob: float = sin_delta**2
    signal: float = amplification * abs(phi_phase)
    cos_delta: float = np.cos(delta)
    fisher_information: float = N * cos_delta**2
    fisher_information_conventional: float = N
    delta_phi: float = 1.0 / np.sqrt(max(fisher_information, 1e-300))
    delta_phi_conventional: float = 1.0 / np.sqrt(N)

    # Physics assertions: no metrological advantage
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


def weak_value_mzi_sensitivity(
    N: float,
    phi_phase: float,
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
        phi_phase: Phase shift in radians (the parameter being estimated).
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
        >>> result = weak_value_mzi_sensitivity(N=100, phi_phase=0.01, config=config)
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
            "signal": np.inf if phi_phase != 0 else 0.0,
            "fisher_information": N,  # Limit: cos²(δ) → 1
            "fisher_information_conventional": N,
            "delta_phi": 1.0 / np.sqrt(N),
            "delta_phi_conventional": 1.0 / np.sqrt(N),
        }

    # Normal path: analytical computation
    return _compute_analytical_sensitivity_metrics(N, delta, phi_phase)

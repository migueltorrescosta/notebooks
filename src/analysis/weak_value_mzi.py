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

    """

    theta: float = np.pi / 4
    phi_bs: float = 0.0
    post_select_angle: float = np.pi / 2 - 0.1  # Slightly off π/2




# Internal Helpers




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

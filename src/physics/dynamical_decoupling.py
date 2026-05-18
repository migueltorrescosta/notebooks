"""
Dynamical Decoupling for Quantum Interferometry.

This module implements dynamical decoupling (DD) pulse sequences to
suppress low-frequency noise and extend coherence times in quantum
interferometers. It provides filter functions, effective coherence
times, phase sensitivity calculations, and scaling analysis.

Physical Model:
- Dynamical decoupling uses periodic π-pulses to refocus phase evolution
- The pulse sequence filters out low-frequency noise, effectively
  extending the coherence time T₂
- For CPMG: T₂^(DD) ≈ T₂⁰ · n_pulses^(2/3)
- For XY-8: T₂^(DD) ≈ T₂⁰ · n_pulses^(0.8) (slightly better isotropy)
- DD improves the prefactor C but does NOT change the SQL scaling
  exponent α = -0.5
- Phase sensitivity: Δφ = 1/√(N · T₂^(DD) / T)
- The filter function F(ω) quantifies the spectral suppression

Hilbert Space:
- Not applicable at the filter-function / sensitivity-scaling level
- Quantum state dynamics assumed to live in a spin-1/2 or two-level subspace
- For full state simulation, see src.evolution.lindblad_solver

Units:
- Dimensionless throughout (ℏ = 1)
- Frequencies ω in rad/sample (dimensionless when multiplied by pulse spacing)
- Coherence times T₂ in dimensionless units consistent with time t

Conventions:
- Filter function: F(ω) = |y(ω)|² where y(ω) = ∫ g(t) e^{iωt} dt / T
- Modulation function g(t) = ±1 alternates sign at each π-pulse
- CPMG sequence: all π-pulses have same phase (x-axis)
- XY-8 sequence: π-pulses alternate phase (x→y→x→y→...) for isotropy

Pulse Sequences:
| Sequence | Power Law | Best For                      |
|----------|-----------|-------------------------------|
| CPMG     | n^(2/3)   | General noise suppression     |
| XY-8     | n^(0.8)   | Non-Markovian / anisotropic   |

Units:
- Dimensionless throughout (ℏ = 1)
- Time in arbitrary units
- Frequency in inverse time units
- Phase in radians

References:
- Carr & Purcell (1954) "Effects of Diffusion on Free Precession in
  Nuclear Magnetic Resonance Experiments"
- Meiboom & Gill (1958) "Modified Spin-Echo Method for Measuring
  Nuclear Relaxation Times"
- Gullion, Baker & Conradi (1990) "New, compensated Carr-Purcell
  sequences" (XY-8)

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

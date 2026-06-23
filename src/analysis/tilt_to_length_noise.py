"""Tilt-to-length (TTL) noise models for interferometry.

Implements tilt-to-length coupling noise, combined quantum+TTL sensitivity,
and scaling analysis for gravitational-wave detectors and long-baseline
interferometers.

Extracted from src/analysis/scaling_survey.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


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


# Core Physics


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


# Combined Sensitivity


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


# Scaling Analysis


def _power_law(
    log_N: npt.NDArray[np.float64],
    log_a: float,
    alpha: float,
) -> npt.NDArray[np.float64]:
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


# Validation


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

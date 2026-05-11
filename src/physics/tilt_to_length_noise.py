"""
Tilt-to-Length (TTL) Coupling Noise Model.

This module implements the tilt-to-length coupling noise model, a systematic
noise floor that arises in interferometric measurements when angular jitter
causes an apparent path length change.

Physical Model:
    Angular jitter θ causes an apparent path length change:
        δL = θ_rms · x_offset

    where θ_rms is the root-mean-square angular jitter and x_offset is the
    beam offset from the pivot point. This path length change translates to
    phase noise:
        δφ = 2π · δL / λ

    where λ is the laser wavelength.

Impact on Sensitivity Scaling:
    - At large N (where quantum noise is small), TTL noise creates a constant
      noise floor, and the scaling exponent α → 0.
    - At small N, the standard quantum limit (SQL) scaling α = -0.5 still holds.

Units:
    - θ_rms: radians
    - L, beam_offset, wavelength: any consistent unit (e.g., metres)
    - Phase: radians

References:
    - LISA Science Requirements Document (ESA L3 mission)
    - "Tilt-to-length coupling in LISA" — Hartig et al. (2022)
    - "Laser Interferometry for LISA" — Bender et al. (2008)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


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
            f"Quantum sensitivity must be non-negative, got {quantum_sensitivity}"
        )

    phi_ttl = ttl_phase_noise(config)
    return np.sqrt(quantum_sensitivity**2 + phi_ttl**2)


def _quantum_sensitivity_sql(N: float) -> float:
    """Standard quantum limit scaling: Δφ_Q = 1/√N.

    Args:
        N: Mean photon/atom number.

    Returns:
        SQL-limited phase sensitivity.

    Raises:
        ValueError: If N is non-positive.
    """
    if N <= 0:
        raise ValueError(f"Particle number N must be positive for SQL, got {N}")
    return 1.0 / np.sqrt(N)


def _quantum_sensitivity_hl(N: float) -> float:
    """Heisenberg-limited scaling: Δφ_Q = 1/N.

    Args:
        N: Mean photon/atom number.

    Returns:
        HL-limited phase sensitivity.

    Raises:
        ValueError: If N is non-positive.
    """
    if N <= 0:
        raise ValueError(f"Particle number N must be positive for HL, got {N}")
    return 1.0 / N


# =============================================================================
# Scaling Analysis
# =============================================================================


def _power_law(
    log_N: NDArray[np.float64], log_a: float, alpha: float
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
            f"All N values must be positive, got range [{N_arr.min()}, {N_arr.max()}]"
        )

    if quantum_scaling not in ("sql", "hl"):
        raise ValueError(
            f"Quantum scaling must be 'sql' or 'hl', got '{quantum_scaling}'"
        )

    # Select quantum scaling function
    if quantum_scaling == "sql":
        quantum_fn = _quantum_sensitivity_sql
    else:
        quantum_fn = _quantum_sensitivity_hl

    # Compute sensitivities
    phi_ttl = ttl_phase_noise(config)
    phi_q = np.array([quantum_fn(N) for N in N_arr], dtype=np.float64)
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
            f"RMS angular jitter theta_rms must be positive, got {config.theta_rms}"
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
# Unit Tests
# =============================================================================


def test_ttl_path_length_noise() -> dict:
    """Test that path length noise follows δL = θ_rms · x_offset.

    Returns:
        Dictionary with test results.
    """
    # Simple case: 1 μrad × 1 mm = 1 nm
    config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3)
    result = ttl_path_length_noise(config)
    expected = 1e-9
    assert abs(result - expected) < 1e-12, f"Expected {expected}, got {result}"
    return {"status": "passed"}


def test_ttl_phase_noise_consistency() -> dict:
    """Test that phase noise relates to path length via δφ = 2πδL/λ.

    Returns:
        Dictionary with test results.
    """
    config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6)
    delta_L = ttl_path_length_noise(config)
    phi = ttl_phase_noise(config)
    expected = 2.0 * np.pi * delta_L / config.wavelength
    assert abs(phi - expected) < 1e-12, (
        f"Phase noise {phi} does not match 2πδL/λ = {expected}"
    )
    return {"status": "passed"}


def test_sensitivity_floor_equals_phase_noise() -> dict:
    """Test that sensitivity floor equals TTL phase noise.

    Returns:
        Dictionary with test results.
    """
    config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3)
    floor = ttl_sensitivity_floor(config)
    phase = ttl_phase_noise(config)
    assert abs(floor - phase) < 1e-12, (
        f"Sensitivity floor {floor} differs from phase noise {phase}"
    )
    return {"status": "passed"}


def test_limited_sensitivity_quadrature_sum() -> dict:
    """Test that TTL-limited sensitivity is quadratic sum of noises.

    When quantum sensitivity ≫ TTL noise, total ≈ quantum.
    When TTL noise ≫ quantum sensitivity, total ≈ TTL.

    Returns:
        Dictionary with test results.
    """
    config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3)

    # Large quantum noise regime: total ≈ quantum
    large_quantum = 100.0
    total_large = ttl_limited_sensitivity(10.0, large_quantum, config)
    assert abs(total_large - large_quantum) / large_quantum < 0.01, (
        "Large quantum noise regime failed"
    )

    # Small quantum noise regime: total ≈ TTL
    small_quantum = 1e-10
    phi_ttl = ttl_phase_noise(config)
    total_small = ttl_limited_sensitivity(10.0, small_quantum, config)
    assert abs(total_small - phi_ttl) / phi_ttl < 0.01, "TTL-dominated regime failed"

    return {"status": "passed"}


def test_scaling_sweep_sql_low_N() -> dict:
    """Test that at low N, scaling approaches SQL (α ≈ -0.5).

    Returns:
        Dictionary with test results.
    """
    config = TTLNoiseConfig(theta_rms=1e-12, beam_offset=1e-6, wavelength=1e-6)
    N = np.logspace(0, 3, 20)
    result = ttl_scaling_sweep(N, config, quantum_scaling="sql")

    assert result["alpha_fitted"] is not None, "Alpha fit returned None"
    # At low N with tiny TTL, α should be close to -0.5
    assert abs(result["alpha_fitted"] - (-0.5)) < 0.05, (
        f"Expected α ≈ -0.5 for SQL regime, got {result['alpha_fitted']}"
    )
    return {"status": "passed"}


def test_scaling_sweep_hl_low_N() -> dict:
    """Test that at low N, HL scaling gives α ≈ -1.0.

    Returns:
        Dictionary with test results.
    """
    config = TTLNoiseConfig(theta_rms=1e-12, beam_offset=1e-6, wavelength=1e-6)
    N = np.logspace(0, 3, 20)
    result = ttl_scaling_sweep(N, config, quantum_scaling="hl")

    assert result["alpha_fitted"] is not None, "Alpha fit returned None"
    assert abs(result["alpha_fitted"] - (-1.0)) < 0.05, (
        f"Expected α ≈ -1.0 for HL regime, got {result['alpha_fitted']}"
    )
    return {"status": "passed"}


def test_scaling_sweep_returns_dict() -> dict:
    """Test that scaling sweep returns the expected dictionary structure.

    Returns:
        Dictionary with test results.
    """
    config = TTLNoiseConfig()
    N = np.logspace(0, 6, 10)
    result = ttl_scaling_sweep(N, config, quantum_scaling="sql")

    expected_keys = {
        "N",
        "delta_phi",
        "delta_phi_quantum",
        "delta_phi_ttl",
        "alpha_fitted",
    }
    assert set(result.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(result.keys())}"
    )
    assert isinstance(result["N"], np.ndarray)
    assert isinstance(result["delta_phi"], np.ndarray)
    assert isinstance(result["delta_phi_quantum"], np.ndarray)
    assert isinstance(result["delta_phi_ttl"], float)
    assert len(result["N"]) == len(N)
    return {"status": "passed"}


def test_config_validation_raises_on_zero_theta() -> dict:
    """Test that zero angular jitter raises ValueError.

    Returns:
        Dictionary with test results.
    """
    import pytest as _pytest

    config = TTLNoiseConfig(theta_rms=0.0)
    with _pytest.raises(ValueError, match="positive"):
        ttl_phase_noise(config)
    return {"status": "passed"}


def test_config_validation_raises_on_negative_wavelength() -> dict:
    """Test that negative wavelength raises ValueError.

    Returns:
        Dictionary with test results.
    """
    import pytest as _pytest

    config = TTLNoiseConfig(wavelength=-1e-6)
    with _pytest.raises(ValueError, match="positive"):
        ttl_phase_noise(config)
    return {"status": "passed"}


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    # Run unit tests
    print("Running tilt-to-length noise tests...")

    results = {
        "ttl_path_length_noise": test_ttl_path_length_noise(),
        "ttl_phase_noise_consistency": test_ttl_phase_noise_consistency(),
        "sensitivity_floor_equals_phase_noise": test_sensitivity_floor_equals_phase_noise(),
        "limited_sensitivity_quadrature_sum": test_limited_sensitivity_quadrature_sum(),
        "scaling_sweep_sql_low_N": test_scaling_sweep_sql_low_N(),
        "scaling_sweep_hl_low_N": test_scaling_sweep_hl_low_N(),
        "scaling_sweep_returns_dict": test_scaling_sweep_returns_dict(),
        "config_validation_raises_on_zero_theta": test_config_validation_raises_on_zero_theta(),
        "config_validation_raises_on_negative_wavelength": (
            test_config_validation_raises_on_negative_wavelength()
        ),
    }

    for test_name, result in results.items():
        print(f"  {test_name}: {result['status']}")

    # Demonstrate scaling behaviour
    print("\n--- Scaling Analysis ---")
    config = TTLNoiseConfig(theta_rms=1e-6, beam_offset=1e-3, wavelength=1e-6)
    N = np.logspace(0, 8, 50)
    result = ttl_scaling_sweep(N, config, quantum_scaling="sql")
    print(f"  TTL phase noise floor: {result['delta_phi_ttl']:.6e} rad")
    print(f"  Fitted scaling exponent α: {result['alpha_fitted']:.4f}")
    print("  (Expected α → 0 at large N when TTL dominates)")

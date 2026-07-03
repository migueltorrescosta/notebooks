"""Thermal Langevin noise models for quantum metrology scaling surveys.

Implements normalized and physical thermal noise models, combined
thermal+quantum sensitivity, and scaling analysis utilities.

Extracted from src/analysis/scaling_survey.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.analysis.scaling_fit import ScalingFitResult, fit_scaling_exponent


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


# Normalized Thermal-Quantum Model


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
) -> float:
    """Compute the combined thermal+quantum phase sensitivity at given N.

    This is the main function for scaling surveys.

    Args:
        N: Particle number.
        base_config: Base thermal noise configuration.

    Returns:
        Combined Δφ = sqrt(Δφ_quantum² + Δφ_thermal²).

    """
    return combined_sensitivity(N, base_config)


def sweep_thermal_scaling(
    N_values: list[int] | npt.NDArray[np.int_],
    base_config: ThermalLangevinConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sweep thermal+quantum sensitivity over a range of N values.

    Args:
        N_values: Array of particle numbers to evaluate.
        base_config: Base configuration.

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
    min_N: int = 4,
) -> ScalingFitResult:
    """Fit the effective scaling exponent α for thermal+quantum noise.

    Extracts α from Δφ ∝ N^α across the given N range.

    Args:
        N_values: Array of particle numbers to evaluate.
        base_config: Base configuration.
        min_N: Minimum N for the fit.

    Returns:
        ScalingFitResult with exponent α and quality metrics.

    """
    N_arr, delta_phi_arr = sweep_thermal_scaling(N_values, base_config)

    return fit_scaling_exponent(N_arr, delta_phi_arr, min_N=min_N)


def crossover_N(
    base_config: ThermalLangevinConfig,
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


# Low-level Physical Susceptibility Functions
# (For reference / future extension)


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
    temp: float,
    gamma: float,
    k_B: float = 1.0,
) -> float:
    """Compute thermal force power spectral density S_F(ω).

    For thermal Langevin noise, the force PSD is white:
        S_F = 2 Γ k_B temp

    Args:
        temp: Temperature.
        gamma: Damping Γ.
        k_B: Boltzmann constant.

    Returns:
        Force PSD S_F.

    """
    return 2.0 * gamma * k_B * temp


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


# Convenience Configurations


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

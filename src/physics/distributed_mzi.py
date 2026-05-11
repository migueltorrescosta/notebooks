"""
Distributed Array Interferometer Model.

This module implements an array of M Mach-Zehnder interferometers (MZIs)
with optional entanglement across sensors for distributed quantum metrology.

Physical Model:
- M independent MZI sensors, each with N photons
- Classical averaging: Δφ_total = Δφ_single / √M (SQL per √M improvement)
- Entangled sensors: Δφ_total = 1/(M·N) (collective Heisenberg limit)
- Correlated noise degrades quantum advantage:
  Δφ_total² = (1-c)·Δφ_ind²/M + c·Δφ_corr²

Scaling Regimes:
- Uncorrelated classical: Δφ ∝ N^(-0.5)
- Entangled (no correlated noise): Δφ ∝ N^(-1.0)
- Correlated noise dominated: Δφ → constant (N^0)

References:
- Giovannetti et al. "Quantum metrology" (2011)
- Demkowicz-Dobrzanski et al. "Quantum metrology with nonclassical states" (2015)
- Escher et al. "General framework for estimating the ultimate precision" (2011)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.physics.noise_channels import NoiseConfig


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
                f"correlation_noise must be in [0, 1], got {self.correlation_noise}"
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
    else:
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
    N_plus_int = max(int(round(N_per_sensor * (1 + rel_perturb))), 1)
    N_minus_int = max(int(round(N_per_sensor * (1 - rel_perturb))), 1)

    # Handle edge case where N is 1
    if N_minus_int == N_plus_int:
        N_plus_int = N_minus_int + 1

    phi_test = 0.0

    result_plus = distributed_mzi_sensitivity(
        N_plus_int, phi_test, config, noise_config
    )
    result_minus = distributed_mzi_sensitivity(
        N_minus_int, phi_test, config, noise_config
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

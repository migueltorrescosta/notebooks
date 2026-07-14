"""
Sensitivity Analysis Physics Module.

This module contains the core physics logic for sensitivity analysis:
- Rabi frequency computation
- Sensitivity calculations to parameters
- Heatmap generation

Physical Model:
- Reduced system-ancilla model with single Fock state
- H = (-J_S sigma_x + delta_S sigma_z) + alpha_x sigma_x J_z + alpha_z sigma_z J_z

Units:
- Dimensionless throughout.
"""

import numpy as np

from src.utils.sensitivity import (
    compute_observable,  # noqa: F401
    compute_rabi_frequency,  # noqa: F401
    sensitivity,
)
from src.utils.validators import validate_sensitivity

# Alias for backward compatibility
validate_sensitivity = validate_sensitivity


def compute_sensitivity_grid(
    n: int,
    k: int,
    j_s: float,
    delta_s: float,
    alpha_x_range: np.ndarray,
    alpha_z_range: np.ndarray,
    t: float,
) -> dict[str, np.ndarray]:
    """Compute sensitivity over a grid of α_x, α_z values.

    Args:
        n: Ancillary dimension.
        k: Level.
        j_s: System parameter.
        delta_s: System parameter.
        alpha_x_range: Range of α_x values.
        alpha_z_range: Range of α_z values.
        t: Time.

    Returns:
        Dictionary with sensitivity grids.

    """
    omega_grid = np.zeros((len(alpha_x_range), len(alpha_z_range)))
    sens_j_grid = np.zeros_like(omega_grid)
    sens_delta_grid = np.zeros_like(omega_grid)

    for i, ax in enumerate(alpha_x_range):
        for j, az in enumerate(alpha_z_range):
            result = sensitivity(n, k, j_s, delta_s, ax, az, t)
            omega_grid[i, j] = result["omega_k"]
            sens_j_grid[i, j] = result["sensitivity_to_j"]
            sens_delta_grid[i, j] = result["sensitivity_to_delta"]

    return {
        "omega_k": omega_grid,
        "sensitivity_to_j": sens_j_grid,
        "sensitivity_to_delta": sens_delta_grid,
        "alpha_x": alpha_x_range,
        "alpha_z": alpha_z_range,
    }


# =============================================================================
# Observable Calculation
# =============================================================================

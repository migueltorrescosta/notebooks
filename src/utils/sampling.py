"""
Random sampling utilities for parameter sweeps and initial guesses.

Provides Marsaglia's method for uniform 3-ball sampling and a 6D
configuration sampler used by the free-ancilla protocol.
"""

from __future__ import annotations

import numpy as np


def sample_ball_3d(
    rng: np.random.Generator,
    radius: float = 1.0,
) -> np.ndarray:
    """Sample a 3D vector uniformly from a ball of given *radius*.

    Uses Marsaglia's method: sample a 3D Gaussian direction and a
    radial distance :math:`r = R \\cdot u^{1/3}` where
    :math:`u \\sim U[0, 1]`.

    Args:
        rng: NumPy random generator.
        radius: Radius of the 3-ball (default 1.0).

    Returns:
        Length-3 array :math:`(x, y, z)` uniformly distributed in the ball.
    """
    v = rng.normal(size=3)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        v = np.array([1.0, 0.0, 0.0])
        norm_v = np.float64(1.0)
    r = radius * (rng.uniform(0.0, 1.0) ** (1.0 / 3.0))
    return v * r / norm_v


def sample_6d_config(
    rng: np.random.Generator,
    *,
    drive_radius: float = 1.0,
    azz_bounds: tuple[float, float] = (-0.5, 0.5),
) -> tuple[float, float, float, float, float, float]:
    """Sample a 6D parameter configuration for the free-ancilla protocol.

    Sampling scheme:
    - :math:`\\theta_A \\sim U[0, \\pi]`
    - :math:`\\phi_A \\sim U[0, 2\\pi)`
    - :math:`(a_x, a_y, a_z)` from the 3-ball :math:`\\|\\mathbf{a}\\| \\le` *drive_radius*
      (Marsaglia's method for uniform volume)
    - :math:`a_{zz} \\sim U[\\text{azz_bounds[0]}, \\text{azz_bounds[1]}]`

    Args:
        rng: NumPy random generator.
        drive_radius: Radius of the 3-ball for drive coefficients.
        azz_bounds: (min, max) for the Ising coupling.

    Returns:
        Tuple ``(theta_A, phi_A, a_x, a_y, a_z, a_zz)``.
    """
    theta_A = float(rng.uniform(0.0, np.pi))
    phi_A = float(rng.uniform(0.0, 2.0 * np.pi))

    a_vec = sample_ball_3d(rng, radius=drive_radius)
    a_x = float(a_vec[0])
    a_y = float(a_vec[1])
    a_z = float(a_vec[2])

    a_zz = float(rng.uniform(azz_bounds[0], azz_bounds[1]))

    return theta_A, phi_A, a_x, a_y, a_z, a_zz

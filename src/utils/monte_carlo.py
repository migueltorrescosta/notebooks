"""
Monte Carlo sampling utilities for the norm-ball sensitivity landscape.

Provides two sampling methods for the 3-ball of drive vectors:

- **Marsaglia's method** (``marsaglia_ball_sample``): uniform *volume*
  distribution, producing P(r) ∝ r². Most samples are at large radii.
- **Stratified method** (``stratified_ball_sample``): uniform *linear* density
  in radius r = ‖a‖, with uniform direction on the 2-sphere. Every radial
  stratum receives equal sample count, resolving the small-‖a‖ regime.

Both return the same ``(drive_samples, azz_samples)`` contract for drop-in
interchangeability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StratifiedBallConfig:
    """Configuration for stratified ball sampling.

    Attributes:
        n_strata: Number of equal-width radial strata in [0, R].
        n_per_stratum: Number of samples to draw per stratum.
    """

    n_strata: int = 10
    n_per_stratum: int = 500


def marsaglia_ball_sample(
    rng: np.random.Generator,
    n_samp: int,
    R: float,
    azz_lo: float,
    azz_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate uniform-volume samples from the 3-ball of radius *R*.

    Uses Marsaglia's method: 3 i.i.d. standard normal variates normalised
    to the unit 2-sphere, then scaled by ``R * u^(1/3)`` where
    ``u ~ U[0, 1]``.  The resulting radial distribution satisfies
    ``P(‖a‖ ≤ r) = (r / R)³``.

    Args:
        rng: NumPy random generator.
        n_samp: Number of samples.
        R: Ball radius.
        azz_lo: Lower bound for the independent a_zz sample.
        azz_hi: Upper bound for the independent a_zz sample.

    Returns:
        Tuple ``(drive_samples, azz_samples)`` where ``drive_samples`` has
        shape ``(n_samp, 3)`` (columns [a_x, a_y, a_z]) and ``azz_samples``
        has shape ``(n_samp,)``.
    """
    # 3 i.i.d. standard normals → direction on unit 2-sphere
    z = rng.normal(0.0, 1.0, size=(n_samp, 3))
    sphere_norm = np.sqrt(np.sum(z**2, axis=1))
    sphere_norm = np.maximum(sphere_norm, 1e-300)  # avoid division by zero
    z_unit = z / sphere_norm[:, np.newaxis]

    # Radial scaling for uniform volume
    u = rng.uniform(0.0, 1.0, size=n_samp)
    r_scaled = R * (u ** (1.0 / 3.0))
    drive_samples = z_unit * r_scaled[:, np.newaxis]

    # a_zz sampled uniformly and independently
    azz_samples = rng.uniform(azz_lo, azz_hi, size=n_samp)

    return drive_samples, azz_samples


def stratified_ball_sample(
    rng: np.random.Generator,
    n_per_stratum: int,
    n_strata: int,
    R: float,
    azz_lo: float,
    azz_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate stratified samples from the 3-ball of radius *R*.

    Divides ``[0, R]`` into *n_strata* equal-width radial bins and draws
    *n_per_stratum* points uniformly within each bin.  Direction is uniform
    on the unit 2-sphere (Marsaglia method).  Total samples returned is
    ``n_strata × n_per_stratum``.

    Unlike :func:`marsaglia_ball_sample`, this method gives every radial
    interval equal sample density, making it suitable for resolving the
    small-‖a‖ regime.

    Args:
        rng: NumPy random generator.
        n_per_stratum: Samples per radial stratum.
        n_strata: Number of radial strata.
        R: Ball radius.
        azz_lo: Lower bound for a_zz.
        azz_hi: Upper bound for a_zz.

    Returns:
        Tuple ``(drive_samples, azz_samples)`` where ``drive_samples`` has
        shape ``(n_strata × n_per_stratum, 3)`` and ``azz_samples`` has
        shape ``(n_strata × n_per_stratum,)``.
    """
    n_total = n_strata * n_per_stratum
    r_bounds = np.linspace(0.0, R, n_strata + 1)  # n_strata + 1 edges

    drive_buffers: list[np.ndarray] = []
    azz_buffers: list[np.ndarray] = []

    for i in range(n_strata):
        r_lo = float(r_bounds[i])
        r_hi = float(r_bounds[i + 1])

        # Uniform direction on unit 2-sphere
        z = rng.normal(0.0, 1.0, size=(n_per_stratum, 3))
        sphere_norm = np.sqrt(np.sum(z**2, axis=1))
        sphere_norm = np.maximum(sphere_norm, 1e-300)
        direction = z / sphere_norm[:, np.newaxis]

        # Uniform radius within [r_lo, r_hi]
        radii = rng.uniform(r_lo, r_hi, size=n_per_stratum)

        drive_buffers.append(direction * radii[:, np.newaxis])
        azz_buffers.append(rng.uniform(azz_lo, azz_hi, size=n_per_stratum))

    drive_samples = np.concatenate(drive_buffers, axis=0)
    azz_samples = np.concatenate(azz_buffers, axis=0)

    assert drive_samples.shape == (n_total, 3), (
        f"Expected ({n_total}, 3), got {drive_samples.shape}"
    )
    assert azz_samples.shape == (n_total,), (
        f"Expected ({n_total},), got {azz_samples.shape}"
    )

    return drive_samples, azz_samples

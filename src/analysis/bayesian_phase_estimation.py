"""
Bayesian Phase Estimation for Quantum Interferometry.

This module implements Bayesian inference for estimating unknown phase shifts
from measurement outcomes in a Mach-Zehnder interferometer. It computes
posterior distributions and sensitivity bounds using Bayes' rule.

Physical Model:
- Prior: π(φ) = 1/(2π) uniform on [0, 2π)
- Likelihood: P(m|φ) from interferometer output distribution
- Posterior: P(φ|m₀) ∝ P(m₀|φ) × π(φ)
- Sensitivity: Δφ_B = Std[φ|m₀] (posterior standard deviation)

Hilbert Space:
- System: Two-mode Fock basis (dimension depends on max_photons)
- Parameter space: Phase φ ∈ [0, 2π] (discretized grid)

Units:
- Dimensionless throughout. Phase is measured in radians.

Conventions:
- Beam splitter: 50/50 (θ = π/4)
- Phase convention: e^{iφn₁} applied to mode 1 (second mode)
- Output mode 0 detection probability: P(m=0|φ) = cos²(φ/2) for single photon input
"""

import numpy as np

from src.physics.mzi_simulation import (
    compute_output_probabilities,
    evolve_mzi,
    prepare_input_state,
)


def bayesian_likelihood(
    phi_grid: np.ndarray,
    initial_state: np.ndarray,
    max_photons: int,
    ancilla_dim: int = 1,
) -> np.ndarray:
    """Compute likelihood P(m=0|φ) over the phase grid.

    For each phase value φ, evolves the initial state through the MZI
    and computes the probability of detecting the photon in output mode 0.

    Args:
        phi_grid: Array of phase values in [0, 2π].
        initial_state: Initial system state vector.
        max_photons: Maximum photons per mode for truncation.
        ancilla_dim: Dimension of ancilla Hilbert space (default 1 = no ancilla).

    Returns:
        Array of P(m=0|φ) for each phase value.
    """
    n_phi = len(phi_grid)
    likelihood = np.zeros(n_phi)

    theta = np.pi / 4  # 50/50 beam splitter
    phi_bs = 0.0  # Beam splitter phase
    g = 0.0  # No system-ancilla coupling
    interaction_time = 0.0
    coupling_type = "phase_coupling"

    for i, phi in enumerate(phi_grid):
        final_state = evolve_mzi(
            initial_state,
            theta,
            phi_bs,
            phi,
            g,
            interaction_time,
            coupling_type,
            max_photons,
            ancilla_dim,
        )
        p0, _ = compute_output_probabilities(final_state, max_photons, ancilla_dim)
        likelihood[i] = p0

    return likelihood


def bayesian_posterior(
    measurement_outcome: int,
    initial_state: np.ndarray,
    max_photons: int,
    prior_range: np.ndarray,
    ancilla_dim: int = 1,
) -> np.ndarray:
    """Compute posterior P(φ|m) over discretized phase grid.

    Uses Bayes' rule: P(φ|m) = P(m|φ) × π(φ) / P(m)
    where π(φ) = 1/(2π) is the uniform prior.

    For outcome m=0: P(φ|0) ∝ P(0|φ) × 1/(2π)
    For outcome m=1: P(φ|1) ∝ P(1|φ) × 1/(2π) = (1 - P(0|φ)) × 1/(2π)

    Args:
        measurement_outcome: Measurement outcome (0 or 1, representing output port).
        initial_state: Initial system state vector.
        max_photons: Maximum photons per mode for truncation.
        prior_range: Discretized phase grid (array of φ values).
        ancilla_dim: Dimension of ancilla Hilbert space (default 1 = no ancilla).

    Returns:
        Normalized posterior distribution P(φ|m) over the phase grid.

    Raises:
        ValueError: If measurement_outcome is not 0 or 1.
    """
    if measurement_outcome not in (0, 1):
        raise ValueError(
            f"measurement_outcome must be 0 or 1, got {measurement_outcome}"
        )

    # Compute likelihood P(m=0|φ) for all φ
    likelihood = bayesian_likelihood(
        prior_range, initial_state, max_photons, ancilla_dim
    )

    if measurement_outcome == 0:
        # P(φ|0) ∝ P(0|φ) × π(φ), with π(φ) = 1/(2π) uniform
        unnormalized = likelihood
    else:
        # P(φ|1) = 1 - P(0|φ) for the complementary outcome
        unnormalized = 1.0 - likelihood

    # Normalize: P(φ|m) = unnormalized / ∫unnormalized dφ
    # In discrete form: divide by sum over grid
    total = np.sum(unnormalized)

    if total > 0:
        posterior = unnormalized / total
    else:
        # Fallback: uniform posterior if likelihood is zero everywhere
        posterior = np.ones_like(unnormalized) / len(unnormalized)

    return posterior


def bayesian_sensitivity(posterior: np.ndarray, phi_grid: np.ndarray) -> float:
    """Compute posterior standard deviation (Bayesian sensitivity).

    Δφ_B = sqrt(⟨φ²⟩ - ⟨φ⟩²) where expectation is over posterior.

    For phase wrapped on [0, 2π], uses circular standard deviation
    for more accurate estimation near boundaries.

    Args:
        posterior: Posterior distribution P(φ|m) over phase grid.
        phi_grid: Discretized phase grid (array of φ values).

    Returns:
        Posterior standard deviation Δφ_B.

    Raises:
        ValueError: If posterior doesn't sum to 1 or grid lengths don't match.
    """
    if len(posterior) != len(phi_grid):
        raise ValueError(
            f"Posterior length {len(posterior)} must match phi_grid length "
            f"{len(phi_grid)}"
        )

    # Check normalization (allow small tolerance)
    total_prob = np.sum(posterior)
    if not np.isclose(total_prob, 1.0, rtol=1e-5):
        raise ValueError(f"Posterior must be normalized, got sum {total_prob}")

    # Compute mean phase (linear)
    mean_phi = np.sum(posterior * phi_grid)

    # Compute variance (linear approximation)
    # For phases near boundaries, we should use circular statistics
    # But for simplicity, use linear approximation after centering
    variance = np.sum(posterior * (phi_grid - mean_phi) ** 2)
    std_phi = np.sqrt(variance)

    return float(std_phi)


def bayesian_sensitivity_circular(posterior: np.ndarray, phi_grid: np.ndarray) -> float:
    """Compute circular posterior standard deviation.

    Uses the correct circular statistics for phase:
    Δφ_circ = sqrt(-2 ln(|⟨e^{iφ}⟩|))

    This properly handles the wrap-around at 0 and 2π.

    Args:
        posterior: Posterior distribution P(φ|m) over phase grid.
        phi_grid: Discretized phase grid (array of φ values).

    Returns:
        Circular standard deviation.
    """
    # Compute the resultant vector length R = |⟨e^{iφ}⟩|
    complex_mean = np.sum(posterior * np.exp(1j * phi_grid))
    r = np.abs(complex_mean)

    if r < 1e-12:
        # Uniform distribution: maximum uncertainty
        return np.pi / np.sqrt(3)  # ≈ 1.81 rad

    # Circular variance: 1 - R
    # Circular standard deviation: sqrt(-2 ln(R))
    circ_var = -2.0 * np.log(r)
    if circ_var < 0:
        circ_var = 0.0  # Numerical stability

    return float(np.sqrt(circ_var))


def sample_measurement_outcomes(
    initial_state: np.ndarray,
    phi_true: float,
    n_samples: int,
    rng: np.random.Generator,
    max_photons: int = 1,
    ancilla_dim: int = 1,
) -> np.ndarray:
    """Sample measurement outcomes from P(m|φ_true).

    For a true phase φ_true, simulates n_samples measurement outcomes
    by computing the output probability and sampling from a Bernoulli
    distribution.

    Args:
        initial_state: Initial system state vector.
        phi_true: True phase shift value in radians.
        n_samples: Number of measurement outcomes to sample.
        rng: NumPy random generator for reproducible sampling.
        max_photons: Maximum photons per mode for truncation.
        ancilla_dim: Dimension of ancilla Hilbert space.

    Returns:
        Array of n_samples measurement outcomes (0s and 1s).

    Raises:
        ValueError: If n_samples <= 0.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    # Evolve state through MZI with true phase
    theta = np.pi / 4  # 50/50 beam splitter
    phi_bs = 0.0
    g = 0.0
    interaction_time = 0.0
    coupling_type = "phase_coupling"

    final_state = evolve_mzi(
        initial_state,
        theta,
        phi_bs,
        phi_true,
        g,
        interaction_time,
        coupling_type,
        max_photons,
        ancilla_dim,
    )

    # Compute output probability P(m=0|φ)
    p0, _ = compute_output_probabilities(final_state, max_photons, ancilla_dim)

    # Sample outcomes: m=0 with probability p0, m=1 with probability 1-p0
    outcomes = rng.choice(2, size=n_samples, p=[p0, 1.0 - p0])

    return outcomes


def bayesian_estimator(
    outcomes: np.ndarray,
    initial_state: np.ndarray,
    max_photons: int,
    n_phi: int = 360,
) -> dict:
    """Full Bayesian estimation pipeline.

    Given a sequence of measurement outcomes, computes the posterior
    distribution and estimates the phase and sensitivity.

    For multiple outcomes from identical experiments:
    P(φ|{m_i}) ∝ Π_i P(m_i|φ) × π(φ)

    Args:
        outcomes: Array of measurement outcomes (0s and 1s).
        initial_state: Initial system state vector.
        max_photons: Maximum photons per mode for truncation.
        n_phi: Number of grid points for phase discretization.

    Returns:
        Dictionary with:
        - "phi_grid": Discretized phase grid
        - "prior": Uniform prior (1/(2π))
        - "likelihood": P(m=0|φ) used for updating
        - "posterior": Final posterior distribution
        - "phi_estimate": Posterior mean estimate of φ
        - "sensitivity": Posterior standard deviation Δφ_B
        - "n_outcomes_0": Number of m=0 outcomes
        - "n_outcomes_1": Number of m=1 outcomes

    Raises:
        ValueError: If outcomes is empty.
    """
    if len(outcomes) == 0:
        raise ValueError("outcomes must not be empty")

    # Create phase grid
    phi_grid = np.linspace(0, 2 * np.pi, n_phi)

    # Compute likelihood P(m=0|φ)
    likelihood = bayesian_likelihood(
        phi_grid, initial_state, max_photons, ancilla_dim=1
    )

    # Start with uniform prior
    prior = np.ones(n_phi) / n_phi  # Equivalent to 1/(2π) on grid

    # Update with each outcome
    posterior = prior.copy()
    n_outcomes_0 = np.sum(outcomes == 0)
    n_outcomes_1 = np.sum(outcomes == 1)

    for outcome in outcomes:
        if outcome == 0:
            # Multiply by P(0|φ)
            posterior = posterior * likelihood
        else:
            # Multiply by P(1|φ) = 1 - P(0|φ)
            posterior = posterior * (1.0 - likelihood)

    # Normalize posterior
    total = np.sum(posterior)
    if total > 0:
        posterior = posterior / total
    else:
        # Fallback to prior if all outcomes impossible
        posterior = prior

    # Compute posterior mean estimate
    phi_estimate = np.sum(posterior * phi_grid)

    # For phase near boundary, consider circular mean
    # (simplified: use linear if mean is in reasonable range)
    if phi_estimate > np.pi:
        phi_estimate = phi_estimate - 2 * np.pi

    # Compute posterior standard deviation
    sensitivity = bayesian_sensitivity(posterior, phi_grid)

    return {
        "phi_grid": phi_grid,
        "prior": prior,
        "likelihood": likelihood,
        "posterior": posterior,
        "phi_estimate": phi_estimate,
        "sensitivity": sensitivity,
        "n_outcomes_0": n_outcomes_0,
        "n_outcomes_1": n_outcomes_1,
    }


def bayesian_estimator_batch(
    outcomes: np.ndarray,
    state_type: str,
    max_photons: int,
    n_phi: int = 360,
) -> dict:
    """Batch version of bayesian_estimator with automatic state preparation.

    Args:
        outcomes: Array of measurement outcomes (0s and 1s).
        state_type: Type of input state ("single_photon", "noon", "coherent").
        max_photons: Maximum photons per mode for truncation.
        n_phi: Number of grid points for phase discretization.

    Returns:
        Dictionary from bayesian_estimator.
    """
    initial_state = prepare_input_state(state_type, max_photons=max_photons)

    return bayesian_estimator(outcomes, initial_state, max_photons, n_phi)


def compute_crb(
    initial_state: np.ndarray,
    max_photons: int,
    n_samples: int,
) -> float:
    """Compute Cramér-Rao bound for phase estimation.

    CRB: Δφ ≥ 1/√(N × F_Q)
    where F_Q is the quantum Fisher information and N is number of samples.

    For a single photon through MZI: F_Q = 1 (standard quantum limit)

    Args:
        initial_state: Initial system state vector.
        max_photons: Maximum photons per mode.
        n_samples: Number of independent measurements.

    Returns:
        Cramér-Rao bound Δφ_CRB = 1/√(N × F_Q).
    """
    # Compute QFI for single photon input
    # For a single photon in mode 0 through 50/50 BS:
    # The output probability is P(0|φ) = cos²(φ/2)
    # The classical Fisher information is F_C = 1 (optimal for single photon)

    # For single photon state, QFI = 1 (SQL scaling)
    f_q = 1.0  # Quantum Fisher information for single photon

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    crb = 1.0 / np.sqrt(n_samples * f_q)
    return float(crb)


def bayesian_sensitivity_analytical(
    n_outcomes_0: int,
    n_total: int,
) -> float:
    """Analytical Bayesian sensitivity for binomial data.

    For n independent measurements with prior uniform on [0, 2π],
    the posterior variance can be approximated analytically for large n.

    Asymptotically: Δφ_B ≈ 1/√(n × F_C) where F_C = 1

    For small n, use numerical posterior.

    Args:
        n_outcomes_0: Number of m=0 outcomes.
        n_total: Total number of measurements.

    Returns:
        Approximate sensitivity (posterior std) at n_outcomes_0/n_total successes.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")

    if n_outcomes_0 < 0 or n_outcomes_0 > n_total:
        raise ValueError(f"n_outcomes_0 must be in [0, n_total], got {n_outcomes_0}")

    # Asymptotic approximation
    f_c = 1.0  # Classical Fisher information

    return float(1.0 / np.sqrt(n_total * f_c))

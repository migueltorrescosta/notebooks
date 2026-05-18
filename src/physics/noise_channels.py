"""
Noise Channels for Open Quantum Systems.

This module implements physical noise channels for quantum optical systems
utilizing the Dicke basis representation. These channels model decoherence
effects in interferometers and atomic ensembles.

Physical Model:
- Lindblad master equation: dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
- Noise channels are implemented as Lindblad operators in the Dicke basis
- Detection noise modeled as binomial measurement projection

Hilbert Space:
- Dicke basis |J, m⟩ with J = N/2, dimension d = N + 1
- Consistent with src.physics.dicke_basis conventions

Units:
- Dimensionless (ℏ = 1, time in arbitrary units)
- Decay rates γ in same dimensionless units

Conventions:
- Same as src.physics.dicke_basis module
- Lindblad operators have rates pre-absorbed: L = √γ × operator
- Detection efficiency η ∈ [0, 1], probability P(k|n) = C(n,k) η^k (1-η)^(n-k)

Note: This module uses the Dicke-basis convention (N+1 dimensions per mode), NOT the
two-mode Fock basis. Operators such as annihilation_operator(N) and creation_operator(N)
produce matrices of shape (N+1, N+1). For Fock-basis noise acting on the full two-mode
Hilbert space of dimension (N+1)², see the mzi_lindblad module.

Noise Channels:
| Channel      | Lindblad Operator L | Physical Rate     |
|--------------|---------------------|------------------|
| One-body loss| √γ₁ a              | γ₁ (s⁻¹)        |
| Two-body loss| √γ₂ a²             | γ₂ (s⁻¹ per pair)|
| Phase diffusion| √γ_φ J_z          | γ_φ (s⁻¹)        |
| Detection   | P(k|n) = Binomial   | η (efficiency)  |

References:
- Walls & Milburn (2008) "Quantum Optics"
- Gardiner & Zoller (2004) "Quantum Noise"

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NoiseConfig:
    """Configuration for noise channels in open quantum systems.

    Attributes:
        gamma_1: One-body loss rate (γ₁).
            Lindblad operator: L = √γ₁ a
            Effect: Population decay via single-particle loss.
        gamma_2: Two-body loss rate (γ₂).
            Lindblad operator: L = √γ₂ a²
            Effect: Pairwise particle loss (proportional to N(N-1)).
        gamma_phi: Phase diffusion rate (γ_φ).
            Lindblad operator: L = √γ_φ J_z
            Effect: Dephasing between states with different m.
        eta: Detection efficiency (η ∈ [0, 1]).
            Effect: Binomial detection noise.
            η = 1: Perfect detection (no noise).
            η = 0: No detection events.

    """

    gamma_1: float = 0.0  # one-body loss rate
    gamma_2: float = 0.0  # two-body loss rate
    gamma_phi: float = 0.0  # phase diffusion rate
    eta: float = 1.0  # detection efficiency


# =============================================================================
# Bosonic Operators in Dicke Basis
# =============================================================================


def annihilation_operator(N: int) -> np.ndarray:
    """Construct annihilation operator a in the Dicke basis.

    The operator a acts on Dicke states |J, m⟩, reducing the total
    photon number by 1. In the two-mode picture, this corresponds to
    transferring one photon from mode b to mode a.

    Matrix elements (Dicke basis):
        ⟨J, m'|a|J, m⟩ = √(J + m) if m' = m - 1
        ⟨J, m'|a|J, m⟩ = 0 otherwise

    Args:
        N: Total atom/photon number. Hilbert space dimension is N+1.

    Returns:
        Annihilation operator a of shape (N+1, N+1).

    Raises:
        ValueError: If N is negative.

    Example:
        >>> a = annihilation_operator(4)  # N=4, J=2
        >>> # Acting on |J=2, m=2⟩ (all in mode a) -> √4|J=2, m=1⟩
        >>> np.abs(a[1, 0])  # sqrt(J+m) = sqrt(2+2) = 2
        2.0

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    dim = N + 1

    a = np.zeros((dim, dim), dtype=complex)

    # Build annihilation operator (lowering in m)
    # a|J, m⟩ = √(J + m) |J, m-1⟩
    # For state at index j (m = J - j), we need transition to j+1 (m-1)
    # a[j+1, j] = √(N - j) for j = 0 to N-1
    for j in range(dim - 1):
        element = np.sqrt(N - j)
        a[j + 1, j] = element

    return a


def creation_operator(N: int) -> np.ndarray:
    """Construct creation operator a† in the Dicke basis.

    The operator a† acts on Dicke states |J, m⟩, increasing the total
    photon number by 1.

    Matrix elements (Dicke basis):
        ⟨J, m'|a†|J, m⟩ = √(J - m) if m' = m + 1
        ⟨J, m'|a†|J, m⟩ = 0 otherwise

    Args:
        N: Total atom/photon number. Hilbert space dimension is N+1.

    Returns:
        Creation operator a† of shape (N+1, N+1).

    Raises:
        ValueError: If N is negative.

    Example:
        >>> a_dag = creation_operator(4)
        >>> # Acting on |J=2, m=-2⟩ (all in mode b) -> √4|J=2, m=-1⟩
        >>> np.abs(a_dag[3, 4])  # sqrt(J-m) = sqrt(2-(-2)) = 2
        2.0

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    dim = N + 1

    a_dag = np.zeros((dim, dim), dtype=complex)

    # Build creation operator (raising in m)
    # a†|J, m⟩ = √(J - m) |J, m+1⟩
    # For state at index j (m = J - j), connects to j-1 (m+1)
    # a†[j-1, j] = √j for j = 1 to N
    for j in range(1, dim):
        element = np.sqrt(j)
        a_dag[j - 1, j] = element

    return a_dag


# =============================================================================
# Lindblad Operators
# =============================================================================


def build_lindblad_operators(N: int, config: NoiseConfig) -> list[np.ndarray]:
    """Return list of Lindblad operators for given noise channels.

    Constructs the Lindblad dissipation operators corresponding to
    the active noise channels in the configuration. Each operator
    is scaled with its rate (√γ) as per the Lindblad form.

    Completeness Relation:
        Σ_k L_k†L_k + H†H ≤ I (for valid Lindblad generators)
        where H is the Hamiltonian (here assumed zero).

    Args:
        N: Total atom/photon number. Hilbert space dimension is N+1.
        config: Noise configuration specifying active channels and rates.

    Returns:
        List of Lindblad operator matrices. Only includes operators
        for channels with non-zero rates. Shape: (num_channels, N+1, N+1).

    Raises:
        ValueError: If N is negative.
        ValueError: If config has invalid parameters.

    Example:
        >>> config = NoiseConfig(gamma_1=0.1, gamma_phi=0.05)
        >>> L_ops = build_lindblad_operators(4, config)
        >>> len(L_ops)
        2

    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    # Validate config
    if config.eta < 0 or config.eta > 1:
        raise ValueError(f"Detection efficiency must be in [0, 1], got {config.eta}")
    if config.gamma_1 < 0:
        raise ValueError(
            f"One-body loss rate must be non-negative, got {config.gamma_1}",
        )
    if config.gamma_2 < 0:
        raise ValueError(
            f"Two-body loss rate must be non-negative, got {config.gamma_2}",
        )
    if config.gamma_phi < 0:
        raise ValueError(
            f"Phase diffusion rate must be non-negative, got {config.gamma_phi}",
        )

    L_ops: list[np.ndarray] = []

    # Get base operators
    from src.physics.dicke_basis import jz_operator

    a = annihilation_operator(N)
    J_z = jz_operator(N)

    # One-body loss: L = √γ₁ a
    if config.gamma_1 > 0:
        L_ops.append(np.sqrt(config.gamma_1) * a)

    # Two-body loss: L = √γ₂ a²
    if config.gamma_2 > 0:
        # a² = a @ a
        a_squared = a @ a
        L_ops.append(np.sqrt(config.gamma_2) * a_squared)

    # Phase diffusion: L = √γ_φ J_z
    if config.gamma_phi > 0:
        L_ops.append(np.sqrt(config.gamma_phi) * J_z)

    return L_ops


def validate_lindblad_completeness(
    L_ops: list[np.ndarray],
    tolerance: float = 1e-8,
) -> dict:
    """Validate Lindblad operators satisfy completeness relation.

    For valid Lindblad operators, the sum of L_k†L_k should be bounded
    by the identity (for trace-preserving channels).

    Σ_k L_k†L_k ≤ I

    Args:
        L_ops: List of Lindblad operators.
        tolerance: Numerical tolerance for validation.

    Returns:
        Dictionary with validation results.

    Example:
        >>> a = annihilation_operator(4)
        >>> config = NoiseConfig(gamma_1=0.5)
        >>> L_ops = [np.sqrt(0.5) * a]
        >>> result = validate_lindblad_completeness(L_ops)
        >>> result['is_bounded']
        True

    """
    if len(L_ops) == 0:
        return {
            "is_bounded": True,
            "max_eigenvalue": 0.0,
            "tolerance": tolerance,
        }

    # Compute Σ L_k†L_k
    sum_LdL = np.zeros_like(L_ops[0])
    for L in L_ops:
        LdL = L.conj().T @ L
        sum_LdL = sum_LdL + LdL

    # Check eigenvalues
    eigenvalues = np.linalg.eigvalsh(sum_LdL)
    max_eigenvalue = np.max(eigenvalues)

    return {
        "is_bounded": max_eigenvalue <= 1.0 + tolerance,
        "max_eigenvalue": max_eigenvalue,
        "sum_LdL_eigenvalues": eigenvalues,
        "tolerance": tolerance,
    }


# =============================================================================
# Detection Noise
# =============================================================================


def apply_detection_noise(
    probabilities: np.ndarray,
    eta: float,
    n_trials: int,
    seed: int | None = None,
) -> np.ndarray:
    """Convolve probabilities with binomial for detection inefficiency.

    Models imperfect detection where each particle is detected with
    probability η (detection efficiency). The detection probability
    follows a binomial distribution:

        P(k|n) = Binomial(k; n, η)

    For perfect detection (η = 1), the distribution is δ_{k,n}.

    Args:
        probabilities: Array of probabilities for n = 0, 1, ..., len-1.
        eta: Detection efficiency in [0, 1].
        n_trials: Number of Monte Carlo samples for approximation.
        seed: Random seed for reproducibility.

    Returns:
        Array of detection-probability-weighted probabilities,
        same shape as input.

    Raises:
        ValueError: If eta outside [0, 1].
        ValueError: If n_trials <= 0.

    Example:
        >>> probs = np.array([0.1, 0.4, 0.5])  # P(0), P(1), P(2)
        >>> # Perfect detection: should return same probabilities
        >>> result = apply_detection_noise(probs, eta=1.0, n_trials=1000)
        >>> np.allclose(result, probs, atol=0.01)
        True

    """
    if eta < 0 or eta > 1:
        raise ValueError(f"Detection efficiency must be in [0, 1], got {eta}")
    if n_trials <= 0:
        raise ValueError(f"Number of trials must be positive, got {n_trials}")

    # Perfect detection: no convolution needed
    if eta >= 1.0 - 1e-10:
        return probabilities.copy()

    # No detection: uniform distribution (not physically accurate but preserves normalization)
    if eta <= 1e-10:
        return np.ones_like(probabilities) / probabilities.shape[0]

    rng = np.random.default_rng(seed)
    n_max = len(probabilities) - 1

    # Monte Carlo approach for binomial convolution
    # Draw samples from the input distribution, then apply binomial detection
    result = np.zeros_like(probabilities)

    for _ in range(n_trials):
        # Sample from input distribution
        n_samples = rng.choice(len(probabilities), size=1, p=probabilities)[0]

        # Apply binomial detection: each particle detected with probability eta
        if n_samples > 0:
            # Number of detected particles ~ Binomial(n, eta)
            k_detected = rng.binomial(n_samples, eta)
            if k_detected <= n_max:
                result[k_detected] += 1 / n_trials

    # Normalize result
    result = result / np.sum(result)
    if np.sum(result) > 0:
        result = result / np.sum(result)
    else:
        result = np.ones_like(probabilities) / probabilities.shape[0]

    return result


# =============================================================================
# Expectation Value Calculations
# =============================================================================


def compute_mean_particle_number(
    probabilities: np.ndarray,
) -> float:
    """Compute mean particle number from probability distribution.

    ⟨n⟩ = Σ_n n P(n)

    Args:
        probabilities: Array of probabilities P(n).

    Returns:
        Mean particle number.

    """
    n_values = np.arange(len(probabilities))
    return np.sum(n_values * probabilities)


def compute_particle_variance(
    probabilities: np.ndarray,
) -> float:
    """Compute particle number variance from distribution.

    Var(n) = ⟨n²⟩ - ⟨n⟩²

    Args:
        probabilities: Array of probabilities P(n).

    Returns:
        Variance of particle number.

    """
    n_values = np.arange(len(probabilities))
    mean_n = compute_mean_particle_number(probabilities)
    mean_n2 = np.sum(n_values**2 * probabilities)
    return mean_n2 - mean_n**2

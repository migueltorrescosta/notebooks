"""
Wigner Function Computation for Oscillator States.

Computes the Wigner function W(x,p) for a single bosonic mode
in the truncated Fock basis.

Physical Model:
- Wigner function: W(α) = (1/π) ⟨α| ρ |α⟩ where |α⟩ is a coherent state
- For density matrix ρ = Σ ρ_mn |m⟩⟨n|:
  W(x,p) = (2/π) Σ_{m,n} ρ_mn (-1)^n ψ_m(x,p) ψ_n(x,p)
  where ψ_n are harmonic oscillator eigenfunctions

Units:
- Dimensionless quadrature variables x, p.
- Wigner function normalized: ∫∫ W(x,p) dx dp = 1.
"""

import numpy as np
import scipy.special


def wigner_function_single(
    rho_osc: np.ndarray,
    x_range: np.ndarray,
    p_range: np.ndarray,
) -> np.ndarray:
    """Compute Wigner function for single-mode density matrix.

    Uses the correct formula for truncated Fock basis:
    W(x,p) = (2/π) exp(-x²-p²) Σ_{m,n} ρ_mn (-1)^n ψ_m(x,p) ψ_n(x,p)
    where ψ_n(x,p) are harmonic oscillator eigenfunctions.

    The eigenfunctions are:
    ψ_n(x,p) = <x+ip|n> = exp(-(x²+p²)/2) * (x+ip)^n / √n!

    This gives:
    W(x,p) = (2/π) exp(-2(x²+p²)) Σ_{m,n} ρ_mn (-1)^n (x-ip)^m (x+ip)^n / √(m!n!)

    Args:
        rho_osc: Density matrix of oscillator (dim N+1, N+1).
        x_range: Array of x quadrature values.
        p_range: Array of p quadrature values.

    Returns:
        2D array W[x_idx, p_idx] of Wigner function values.

    Raises:
        ValueError: If rho_osc is not square.
    """
    if rho_osc.ndim != 2 or rho_osc.shape[0] != rho_osc.shape[1]:
        raise ValueError(f"rho_osc must be square, got shape {rho_osc.shape}")

    dim = rho_osc.shape[0]
    nx = len(x_range)
    np_ = len(p_range)

    W = np.zeros((nx, np_), dtype=float)

    # Precompute factorials
    fact = np.array([scipy.special.factorial(n) for n in range(dim)])

    for ix, x in enumerate(x_range):
        for ip, p in enumerate(p_range):
            # Complex amplitude α = x + i p
            alpha = x + 1j * p
            alpha_conj = alpha.conjugate()
            r_sq = x**2 + p**2

            # Precompute (α)^n / √n! and (α*)^m / √m!
            alpha_pow_n = np.array([
                (alpha ** n) / np.sqrt(fact[n]) for n in range(dim)
            ])
            alpha_conj_pow_m = np.array([
                (alpha_conj ** m) / np.sqrt(fact[m]) for m in range(dim)
            ])

            # W = (2/π) exp(-2r²) Σ_{m,n} ρ_mn (-1)^n (α*)^m α^n / √(m!n!)
            coeff = (2.0 / np.pi) * np.exp(-2 * r_sq)

            total = 0.0
            for m in range(dim):
                for n in range(dim):
                    total += rho_osc[m, n] * (-1)**n * alpha_conj_pow_m[m] * alpha_pow_n[n]

            W[ix, ip] = coeff * np.real(total)

    return W


def wigner_from_hybrid_state(
    hybrid_state: np.ndarray,
    N: int,
    x_range: np.ndarray,
    p_range: np.ndarray,
    spin_component: str = "down",
) -> np.ndarray:
    """Extract oscillator density matrix from hybrid state and compute Wigner.

    Args:
        hybrid_state: State vector of shape (2(N+1),) - hybrid oscillator+spin.
        N: Maximum photon number.
        x_range: Array of x quadrature values.
        p_range: Array of p quadrature values.
        spin_component: Which spin to trace ("down" for |↓⟩, "up" for |↑⟩).

    Returns:
        2D array W[x_idx, p_idx].

    Raises:
        ValueError: If spin_component is invalid.
    """
    dim_osc = N + 1
    dim_hybrid = 2 * dim_osc

    if hybrid_state.shape != (dim_hybrid,):
        raise ValueError(f"hybrid_state must have shape ({dim_hybrid},), got {hybrid_state.shape}")

    # Extract oscillator state for given spin component
    if spin_component == "down":
        osc_state = hybrid_state[::2]  # Even indices
    elif spin_component == "up":
        osc_state = hybrid_state[1::2]  # Odd indices
    else:
        raise ValueError(f"Unknown spin_component: {spin_component}")

    # Check if state is pure or mixed (for now assume pure from state vector)
    rho_osc = np.outer(osc_state, osc_state.conj())

    return wigner_function_single(rho_osc, x_range, p_range)


def wigner_minimum(W: np.ndarray) -> float:
    """Find minimum Wigner value (negative values indicate non-Gaussianity).

    Args:
        W: Wigner function array.

    Returns:
        Minimum value of W.
    """
    return float(np.min(W))


def wigner_is_negative(W: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if Wigner function has negative values.

    Args:
        W: Wigner function array.
        tol: Tolerance for considering negative.

    Returns:
        True if min(W) < -tol.
    """
    return wigner_minimum(W) < -tol

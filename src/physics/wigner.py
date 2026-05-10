"""
Wigner Function Computation for Oscillator States.

Computes the Wigner function W(x,p) for a single bosonic mode
in the truncated Fock basis.

Physical Model:
The Wigner function for a single-mode density matrix ρ in the Fock basis
is computed using the displaced-parity formula:

    W(α) = (2/π) Tr[ρ D†(α) (-1)^(a†a) D(α)]

where D(α) = exp(αa† - α*a) is the displacement operator and α = x + ip.
In the Fock basis, the matrix elements are expressed through associated
Laguerre polynomials (Gerry & Knight, Intro. Quantum Optics):

For m ≥ n:
    ⟨m| D(2α) (-1)^(a†a) |n⟩ = (-1)^n √(n!/m!) (2α)^(m-n) exp(-2|α|²)
                                × L_n^{(m-n)}(4|α|²)

For m < n:
    ⟨m| D(2α) (-1)^(a†a) |n⟩ = (-1)^n √(m!/n!) (-2α*)^(n-m) exp(-2|α|²)
                                × L_m^{(n-m)}(4|α|²)

The Wigner function is then:
    W(x,p) = (2/π) Σ_{m,n} ρ_mn ⟨m| D(2α) (-1)^(a†a) |n⟩

Units:
- Dimensionless quadrature variables x, p (convention: α = x + ip,
  so [x, p] = i/2).
- Wigner function normalized: ∫∫ W(x,p) dx dp = 1.
"""

import numpy as np
import scipy.special


def _laguerre_value(n: int, alpha: int, x: float) -> float:
    """Evaluate associated Laguerre polynomial L_n^(alpha)(x).

    Args:
        n: Degree of the Laguerre polynomial.
        alpha: Laguerre parameter (upper index).
        x: Evaluation point.

    Returns:
        L_n^(alpha)(x) as a float.

    Raises:
        ValueError: If n < 0 or alpha < 0.
    """
    if n < 0 or alpha < 0:
        raise ValueError(f"n={n} and alpha={alpha} must be non-negative")
    if n == 0:
        return 1.0
    # eval_genlaguerre(n, alpha, x) evaluates L_n^(alpha)(x)
    return float(scipy.special.eval_genlaguerre(n, alpha, x))


def wigner_function_single(
    rho_osc: np.ndarray,
    x_range: np.ndarray,
    p_range: np.ndarray,
) -> np.ndarray:
    """Compute Wigner function for single-mode density matrix.

    Uses the displaced-parity formula with associated Laguerre polynomials.
    This is the correct formula for arbitrary Fock states, unlike the
    simplified (and incorrect) formula W ∝ α^m (α*)^n / √(m!n!).

    For quadrature convention α = x + ip:
      W(x,p) = (2/π) Σ_{m,n} ρ_mn ⟨m| D(2α) (-1)^(a†a) |n⟩

    where ⟨m| D(2α) (-1)^(a†a) |n⟩ is given by expressions involving
    associated Laguerre polynomials L_k^(d)(4|α|²).

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

    # Precompute factorials for sqrt(n!/m!) factors
    fact = np.array([float(scipy.special.factorial(n)) for n in range(dim)])

    # Precompute sign factors (-1)^n
    parity_sign = np.array([(-1) ** n for n in range(dim)])

    # Precompute sqrt of factorial ratios: sqrt(n! / m!)
    sqrt_fact_ratio = np.zeros((dim, dim), dtype=float)
    for m in range(dim):
        for n in range(dim):
            if m >= n:
                sqrt_fact_ratio[m, n] = np.sqrt(fact[n] / fact[m])
            else:
                sqrt_fact_ratio[m, n] = np.sqrt(fact[m] / fact[n])

    for ix, x in enumerate(x_range):
        for ip, p in enumerate(p_range):
            alpha = x + 1j * p
            r_sq = x**2 + p**2
            tworm = 4.0 * r_sq  # 4|α|² = argument to Laguerre poly

            if r_sq > 50.0:
                # Far outside: exponential decay makes W ≈ 0
                W[ix, ip] = 0.0
                continue

            two_mag = 2.0 * np.sqrt(r_sq)  # |2α|
            exp_factor = np.exp(-2.0 * r_sq)

            total = 0.0 + 0.0j

            for m in range(dim):
                for n in range(dim):
                    if abs(rho_osc[m, n]) < 1e-15:
                        continue

                    sign = parity_sign[n]
                    ratio = sqrt_fact_ratio[m, n]

                    if m >= n:
                        k = m - n
                        # (2α)^k = (2r)^k * exp(i*k*theta)
                        # Use polar representation for numerical stability
                        term_mag = two_mag**k
                        term_phase = np.exp(1j * k * np.angle(alpha))
                        lag = _laguerre_value(n, k, tworm)
                    else:
                        k = n - m
                        # (-2α*)^k = (-2r)^k * exp(-i*k*theta)
                        # = (2r)^k * exp(i*pi*k) * exp(-i*k*theta)
                        # = (2r)^k * (-1)^k * exp(-i*k*theta)
                        term_mag = two_mag**k
                        term_phase = ((-1.0) ** k) * np.exp(-1j * k * np.angle(alpha))
                        lag = _laguerre_value(m, k, tworm)

                    # W_mn contribution
                    contrib = sign * ratio * term_mag * term_phase * lag * exp_factor
                    total += rho_osc[m, n] * contrib

            W[ix, ip] = (2.0 / np.pi) * np.real(total)

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
        raise ValueError(
            f"hybrid_state must have shape ({dim_hybrid},), got {hybrid_state.shape}"
        )

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

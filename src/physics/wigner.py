"""
Wigner Function Computation for Oscillator States.

Computes the Wigner function W(x,p) for a single bosonic mode
in the truncated Fock basis, delegating to qutip.wigner.

Physical Model:
The Wigner function for a single-mode density matrix ρ in the Fock basis
is computed using methods implemented in QuTiP.

Hilbert Space:
- Single bosonic mode in truncated Fock basis |n⟩, n = 0…N
- Dimension: N + 1
- State represented as density matrix ρ of dimension (N+1) × (N+1)

Units:
- Dimensionless throughout (ℏ = 1)
- Phase space coordinates α = x + ip are dimensionless
- Quadrature convention: W(x,p) normalized so ∫∫ W(x,p) dx dp = 1
- Uses g=2 in qutip.wigner to match the convention α = x + ip,
  i.e., [x, p] = i/2, and the Wigner function formula
  W(α) = (2/π) Tr[ρ D†(α) (-1)^(a†a) D(α)].
"""

import numpy as np
import qutip


def wigner_function_single(
    rho_osc: np.ndarray,
    x_range: np.ndarray,
    p_range: np.ndarray,
) -> np.ndarray:
    """Compute Wigner function for single-mode density matrix.

    Delegates to qutip.wigner with g=2 to match the α = x + ip convention
    and normalization ∫∫ W(x,p) dx dp = 1.

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

    rho_qobj = qutip.Qobj(rho_osc)
    # qutip.wigner returns (len(p), len(x)); transpose to (len(x), len(p))
    return qutip.wigner(rho_qobj, x_range, p_range, g=2).T


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
            f"hybrid_state must have shape ({dim_hybrid},), got {hybrid_state.shape}",
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
    rho_qobj = qutip.Qobj(rho_osc)

    # qutip.wigner returns (len(p), len(x)); transpose to (len(x), len(p))
    return qutip.wigner(rho_qobj, x_range, p_range, g=2).T


def wigner_is_negative(W: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if Wigner function has negative values.

    Args:
        W: Wigner function array.
        tol: Tolerance for considering negative.

    Returns:
        True if min(W) < -tol.

    """
    return float(np.min(W)) < -tol

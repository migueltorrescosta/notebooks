"""
Pure linear-algebra utilities for quantum state analysis.

Provides expectation-value, variance, and Fisher-information functions
that depend only on numpy and carry no physics-specific logic.  These
live in ``utils`` so that both ``physics`` and ``analysis`` modules can
import them without creating upward dependency cycles.
"""

from __future__ import annotations

import numpy as np


def compute_expectation_and_variance(
    psi: np.ndarray,
    operator: np.ndarray,
) -> tuple[float, float]:
    """Compute <psi|O|psi> and Var(O) = <O^2> - <O>^2 for a pure state.

    Args:
        psi: Normalised state vector.
        operator: Hermitian operator matrix.

    Returns:
        Tuple (expectation, variance).

    """
    exp_val = np.real(psi.conj() @ operator @ psi)
    op_sq = operator @ operator
    exp_sq = np.real(psi.conj() @ op_sq @ psi)
    raw_var = exp_sq - exp_val**2
    # Clamp residual round-off noise (~1e-16) to zero.
    var_val = max(0.0, raw_var)
    return float(exp_val), float(var_val)


# ---------------------------------------------------------------------------
# Fisher information
# ---------------------------------------------------------------------------


def classical_fisher_information_single(
    p_plus: np.ndarray,
    p_minus: np.ndarray,
    dphi: float,
    p_at_theta: np.ndarray | None = None,
    prob_floor: float = 1e-12,
) -> float:
    """Compute F_C for a single phase value given neighboring probabilities.

    F_C = Sum [P(m|phi+dphi/2) - P(m|phi-dphi/2)]^2 / (dphi^2 * P(m|phi))

    When ``p_at_theta`` is provided, it is used in the denominator
    (textbook definition).  Otherwise the average of ``p_plus`` and
    ``p_minus`` is used (centered approximation, more numerically
    stable for grid-based computations).

    Args:
        p_plus: P(m|phi + dphi/2) for each outcome.
        p_minus: P(m|phi - dphi/2) for each outcome.
        dphi: Phase step size.
        p_at_theta: Optional P(m|phi) at the evaluation point.
        prob_floor: Minimum probability threshold.

    Returns:
        Classical Fisher Information value.

    Raises:
        ValueError: If dphi <= 0 or arrays don't match.

    """
    if dphi <= 0:
        raise ValueError(f"dphi must be positive, got {dphi}")

    if p_plus.shape != p_minus.shape:
        raise ValueError("Probability arrays must have same shape")

    if p_at_theta is not None and p_at_theta.shape != p_plus.shape:
        raise ValueError("p_at_theta must have the same shape as p_plus")

    # Central difference derivative
    deriv = (p_plus - p_minus) / (2 * dphi)

    # Denominator: use p_at_theta if provided, otherwise average
    if p_at_theta is not None:
        denom = p_at_theta
    else:
        denom = 0.5 * (p_plus + p_minus)

    # F_C = Sum (dP/dphi)^2 / P
    mask = denom > prob_floor
    if not np.any(mask):
        return 0.0

    return float(np.sum(deriv[mask] ** 2 / denom[mask]))


def _validate_qfi_inputs(rho: np.ndarray, generator: np.ndarray) -> None:
    """Raise ValueError if rho and generator are invalid QFI inputs."""
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"rho must be a square matrix, got shape {rho.shape}")
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError(
            f"generator must be a square matrix, got shape {generator.shape}",
        )
    if rho.shape != generator.shape:
        raise ValueError(
            f"Dimension mismatch: rho {rho.shape} vs generator {generator.shape}",
        )


def _sld_zero_pairs(
    eigenvectors: np.ndarray,
    pos_idx: np.ndarray,
    zero_idx: np.ndarray,
    eigenvalues: np.ndarray,
    generator: np.ndarray,
) -> float:
    """Sum SLD contributions from positive-zero eigenvalue pairs."""
    fq = 0.0
    for p in range(len(pos_idx)):
        i = pos_idx[p]
        vi = eigenvectors[:, i]
        for q in range(len(zero_idx)):
            j = zero_idx[q]
            vj = eigenvectors[:, j]

            gij = np.vdot(vi, generator @ vj)
            gij_abs_sq = np.abs(gij) ** 2

            weight = 4.0 * eigenvalues[i]
            fq += weight * gij_abs_sq
    return fq


def quantum_fisher_information_dm(rho: np.ndarray, generator: np.ndarray) -> float:
    """Compute Quantum Fisher Information for a mixed state.

    Uses the symmetric logarithmic derivative (SLD) formula:
        F_Q = 4 Sum_{i<j} (l_i - l_j)^2 / (l_i + l_j) . |<i|G|j>|^2

    For pure states (rho = |psi><psi|, only one l > 0), this reduces to
    4*Var(G), handled as a special case for numerical stability.

    Args:
        rho: Density matrix (dim, dim).
        generator: Phase generator Hermitian operator (dim, dim).

    Returns:
        Quantum Fisher Information value F_Q.

    Raises:
        ValueError: If rho or generator are not matrices or dimensions don't match.

    """
    rho = np.asarray(rho, dtype=complex)
    generator = np.asarray(generator, dtype=complex)

    _validate_qfi_inputs(rho, generator)

    eigenvalues, eigenvectors = np.linalg.eigh(rho)

    # Split into positive and zero eigenvalue subspaces
    rank_tol = 1e-12
    pos_idx = np.where(eigenvalues > rank_tol)[0]
    zero_idx = np.where(eigenvalues <= rank_tol)[0]
    n_pos = len(pos_idx)

    if n_pos == 0:
        return 0.0

    if n_pos == 1:
        # Pure state: F_Q = 4*Var(G)
        i = pos_idx[0]
        state = eigenvectors[:, i]
        g_exp = np.vdot(state, generator @ state).real
        g2_exp = np.vdot(state, generator @ generator @ state).real
        var_g = max(0.0, g2_exp - g_exp**2)
        return float(4.0 * var_g)

    # Mixed state: SLD double sum
    fq = 0.0

    # Term 1: positive-positive pairs
    for p in range(n_pos):
        i = pos_idx[p]
        vi = eigenvectors[:, i]
        for q in range(p + 1, n_pos):
            j = pos_idx[q]
            vj = eigenvectors[:, j]

            gij = np.vdot(vi, generator @ vj)
            gij_abs_sq = np.abs(gij) ** 2

            lambda_sum = eigenvalues[i] + eigenvalues[j]
            if lambda_sum > 1e-12:
                lambda_diff_sq = (eigenvalues[i] - eigenvalues[j]) ** 2
                weight = 4.0 * lambda_diff_sq / lambda_sum
                fq += weight * gij_abs_sq

    # Term 2: positive-zero pairs
    fq += _sld_zero_pairs(eigenvectors, pos_idx, zero_idx, eigenvalues, generator)

    return float(np.real(fq))

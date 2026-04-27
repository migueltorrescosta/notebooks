"""
Spin Squeezing via One-Axis Twisting (OAT) Hamiltonian.

This module implements spin-squeezed state generation using the OAT
Hamiltonian H = χ J_z², a key resource for sub-SQL phase estimation
in quantum metrology.

Physical Model:
- Hilbert space: Dicke basis |J, m⟩ with J = N/2, dimension d = N + 1
- OAT Hamiltonian: H = χ J_z² (diagonal in Dicke basis)
- Evolution: U(t) = exp(-i χ t J_z²) = diag(exp(-i χ t m²))
- CSS: |J, -J⟩_z with Var(J_x) = Var(J_y) = N/4 (at SQL)
- Squeezing parameter: ξ = √(Var_perp / (N/4))

Conventions:
- Phase convention: standard quantum mechanics (no extra phases)
- Units: dimensionless throughout
- Basis ordering: m = N/2, N/2-1, ..., -N/2 (descending)

Theoretical Results:
- CSS has ξ = 1 (at standard quantum limit)
- Optimal squeezing time: t_opt ≈ (6/N)^(1/3)/χ
- Minimum squeezing: ξ_min ≈ c N^(-2/3) for large N

References:
- Kitagawa & Ueda (1993). Squeezed spin states.
- Wineland et al. (1992). Measurement-induced quantum-state
  engineering and atom interferometry.

Example:
    >>> from src.algorithms.spin_squeezing import squeezing_parameter, coherent_spin_state
    >>> N = 100
    >>> css = coherent_spin_state(N)  # |J, -J>_z state
    >>> xi = squeezing_parameter(css, N)  # = 1.0 exactly
    >>> print(f"CSS squeezing parameter: {xi:.6f}")
"""

from __future__ import annotations

import numpy as np


def coherent_spin_state(N: int) -> np.ndarray:
    """Generate the coherent spin state (CSS) pointing along +x.

    Returns |J, -J⟩_x via rotation of |J, -J⟩_z around y-axis:
        |J, -J⟩_x = exp(-i π/2 × J_y) |J, -J⟩_z

    This state has:
    - ⟨J_x⟩ = -J (maximum polarization in x)
    - ⟨J_y⟩ = ⟨J_z⟩ = 0 (symmetric in perpendicular directions)
    - Var(J_x) = 0 (minimum along mean direction)
    - Var(J_y) = Var(J_z) = N/4 (at standard quantum limit)

    This is the correct initial state for OAT squeezing:
    - OAT evolution U = exp(-i χ t J_z²) creates quantum correlations
    - Results in reduced perpendicular variance → ξ < 1

    Args:
        N: Total number of two-level atoms. Must be non-negative.

    Returns:
        Complex state vector of dimension (N+1), the |J, -J⟩_x state.

    Raises:
        ValueError: If N is negative.

    Example:
        >>> from src.algorithms.spin_squeezing import squeezing_parameter
        >>> N = 100
        >>> css = coherent_spin_state(N)
        >>> xi = squeezing_parameter(css, N)
        >>> np.isclose(xi, 1.0, atol=1e-6)  # CSS has ξ ≈ 1
        True
    """
    from scipy.linalg import expm
    from src.physics.dicke_basis import jy_operator

    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")

    dim = N + 1

    # Start with |J, -J⟩_z (eigenstate of J_z with eigenvalue -J)
    z_state = np.zeros(dim, dtype=complex)
    z_state[-1] = 1.0

    # Rotate by +π/2 around y-axis to get |J, -J⟩_x
    J_y = jy_operator(N)
    theta = np.pi / 2
    R_y = expm(-1j * theta * J_y)

    return R_y @ z_state


def one_axis_twist(
    initial_state: np.ndarray,
    N: int,
    chi: float,
    t: float,
) -> np.ndarray:
    """Apply one-axis twisting (OAT) evolution.

    The OAT Hamiltonian is H = χ J_z², which is diagonal in the Dicke
    basis. The unitary evolution is:
        U(t)|J, m⟩ = exp(-i χ t m²) |J, m⟩

    This creates quantum correlations (entanglement) between atoms,
    leading to spin squeezing characterized by reduced variance in one
    quadrature at the expense of increased variance in the orthogonal
    quadrature.

    Args:
        initial_state: State vector in Dicke basis of dimension (N+1).
        N: Total number of two-level atoms.
        chi: Nonlinear coupling strength (proportional to
            interactions between atoms).
        t: Evolution time.

    Returns:
        Evolved state vector in Dicke basis.

    Raises:
        ValueError: If N is negative or state dimension is wrong.
        ValueError: If chi or t is negative.

    Example:
        >>> N = 10
        >>> css = coherent_spin_state(N)
        >>> # Apply OAT for short time
        >>> squeezed = one_axis_twist(css, N, chi=1.0, t=0.1)
        >>> # Squeezing parameter should be < 1 after OAT
        >>> xi = squeezing_parameter(squeezed, N)
        >>> print(f"Squeezing parameter: {xi:.4f}")
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")
    if chi < 0:
        raise ValueError(f"Coupling strength chi must be non-negative, got {chi}")
    if t < 0:
        raise ValueError(f"Evolution time t must be non-negative, got {t}")
    if initial_state.shape[0] != N + 1:
        raise ValueError(
            f"State dimension {initial_state.shape[0]} does not match N+1 = {N + 1}"
        )

    # J_z eigenvalues: m = N/2, N/2-1, ..., -N/2
    J = N / 2.0
    m_values = np.arange(J, -J - 1, -1)

    # Phase factors: exp(-i χ t m²)
    # Using broadcasting for efficiency - multiply each amplitude
    # by its corresponding phase factor
    phases = np.exp(-1j * chi * t * m_values**2)

    # Apply diagonal unitary: result[m] = initial_state[m] * exp(-i χ t m²)
    return phases * initial_state


def squeezing_parameter(state: np.ndarray, N: int) -> float:
    r"""Compute the squeezing parameter ξ.

    The squeezing parameter measures quantum noise reduction in phase estimation
    relative to the standard quantum limit (SQL). For a coherent spin state (CSS),
    ξ = 1. A spin-squeezed state has ξ < 1.

    The parameter is defined as:
        ξ = √(Var_min(J_perp) / (N/4))

    where Var_min(J_perp) is the minimum variance in the perpendicular directions
    to the mean spin axis, and N/4 is the standard quantum limit.

    For CSS aligned along z (the OAT eigenbasis):
    - ⟨J_z⟩ = -N/2
    - Var(J_z) = 0 (parallel to mean direction)
    - Var(J_x) = Var(J_y) = N/4 (perpendicular = at SQL)
    - So ξ = 1

    After OAT (H = χ J_z²):
    - State develops correlations reducing the perpendicular variance
    - ξ < 1 indicates squeezing achieved

    For optimal OAT: ξ_min ∝ N^(-2/3) at t_opt ≈ (6/N)^(1/3)/χ

    Args:
        state: State vector in Dicke basis of dimension (N+1).
        N: Total number of two-level atoms.

    Returns:
        Squeezing parameter ξ (dimensionless). ξ < 1 indicates squeezing.

    Raises:
        ValueError: If N is negative or state dimension is wrong.
        ValueError: If state is not normalized.

    Example:
        >>> N = 100
        >>> css = coherent_spin_state(N)
        >>> xi = squeezing_parameter(css, N)
        >>> np.isclose(xi, 1.0, atol=1e-6)  # CSS has ξ ≈ 1
        True
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")
    if state.shape[0] != N + 1:
        raise ValueError(
            f"State dimension {state.shape[0]} does not match N+1 = {N + 1}"
        )

    # Normalize state
    state = state / np.linalg.norm(state)

    # Density matrix
    rho = np.outer(state, state.conj())

    # Import J operators from dicke_basis
    from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator

    J_x = jx_operator(N)
    J_y = jy_operator(N)
    J_z = jz_operator(N)

    # Mean values
    jx_mean = np.real(np.trace(rho @ J_x))
    jy_mean = np.real(np.trace(rho @ J_y))
    jz_mean = np.real(np.trace(rho @ J_z))

    # Normalize mean spin direction
    r = np.array([jx_mean, jy_mean, jz_mean])
    r_norm = np.linalg.norm(r)
    if r_norm > 0:
        r_unit = r / r_norm
    else:
        r_unit = np.array([1.0, 0.0, 0.0])  # fallback

    # Build perpendicular basis to mean spin direction
    # Use z-axis as reference (unless mean spin is along z)
    if abs(r_unit[2]) < 0.99:
        perp_z = np.array([0.0, 0.0, 1.0])
    else:
        perp_z = np.array([1.0, 0.0, 0.0])

    perp_x = np.cross(r_unit, perp_z)
    perp_x = perp_x / np.linalg.norm(perp_x)
    perp_y = np.cross(r_unit, perp_x)

    # Scan over angles in the perpendicular plane to find minimum variance
    min_perp_var = float("inf")
    n_angles = 180

    for k in range(n_angles):
        angle = k * np.pi / n_angles
        # Rotated perpendicular direction
        J_perp_op = np.cos(angle) * (
            perp_x[0] * J_x + perp_x[1] * J_y + perp_x[2] * J_z
        ) + np.sin(angle) * (perp_y[0] * J_x + perp_y[1] * J_y + perp_y[2] * J_z)

        perp_mean = np.real(np.trace(rho @ J_perp_op))
        perp_var = np.real(np.trace(rho @ J_perp_op @ J_perp_op)) - perp_mean**2

        if perp_var < min_perp_var:
            min_perp_var = perp_var

    # SQL variance
    sql_var = N / 4.0

    # Minimum perpendicular variance (squeezing occurs in one direction)
    min_perp_var = max(min_perp_var, 0.0)

    # Compute squeezing parameter: ratio to SQL
    if min_perp_var <= 0:
        return 1.0

    xi = np.sqrt(min_perp_var / sql_var)
    return np.real(xi)


def optimal_squeezing_time(N: int, chi: float) -> float:
    """Compute the optimal squeezing time for OAT.

    The optimal evolution time for maximum squeezing is given by:
        t_opt ≈ (6/N)^(1/3) / χ

    This is derived from maximizing the squeezing parameter for the
    one-axis twisting Hamiltonian. At this time, the squeezing
    degree reaches its minimum value (maximum squeezing):
        ξ_min ∝ N^(-2/3)

    Args:
        N: Total number of two-level atoms. Must be positive.
        chi: Nonlinear coupling strength (positive).

    Returns:
        Optimal evolution time t_opt.

    Raises:
        ValueError: If N ≤ 0 or chi ≤ 0.

    Example:
        >>> N = 100
        >>> t_opt = optimal_squeezing_time(N, chi=1.0)
        >>> print(f"Optimal time: {t_opt:.4f}")
    """
    if N <= 0:
        raise ValueError(f"Number of atoms N must be positive, got {N}")
    if chi <= 0:
        raise ValueError(f"Coupling strength chi must be positive, got {chi}")

    # t_opt = (6/N)^(1/3) / chi
    t_opt = (6.0 / N) ** (1.0 / 3.0) / chi

    return t_opt


def generate_squeezed_state(N: int, chi: float, t: float) -> np.ndarray:
    """Generate a spin-squeezed state via one-axis twisting.

    This function generates a spin-squeezed state by first preparing
    a coherent spin state (CSS) and then applying the OAT evolution
    for time t.

    Args:
        N: Total number of two-level atoms.
        chi: Nonlinear coupling strength.
        t: Evolution time for OAT. Use t=0 for CSS.

    Returns:
        State vector in Dicke basis. For t=0, returns CSS (ξ=1).
        For t>0, returns squeezed state (ξ<1 if t is appropriate).

    Raises:
        ValueError: If N is negative, chi or t is negative.

    Example:
        >>> N = 100
        >>> t_opt = optimal_squeezing_time(N, chi=1.0)
        >>> squeezed = generate_squeezed_state(N, chi=1.0, t=t_opt)
        >>> xi = squeezing_parameter(squeezed, N)
        >>> print(f"Squeezing parameter at t_opt: {xi:.4f}")
    """
    if N < 0:
        raise ValueError(f"Number of atoms N must be non-negative, got {N}")
    if chi < 0:
        raise ValueError(f"Coupling strength chi must be non-negative, got {chi}")
    if t < 0:
        raise ValueError(f"Evolution time t must be non-negative, got {t}")

    # Generate CSS and apply OAT
    css = coherent_spin_state(N)
    return one_axis_twist(css, N, chi, t)

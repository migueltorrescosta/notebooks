"""
Comparison of Ancilla-Assisted vs. Two-Particle Probe in Mach-Zehnder Interferometer.

Implements the following configurations at fixed total resource of 2 particles:

- **Configuration A (ancilla-assisted)**: 1 particle in the two-mode interferometer
  system + 1 spin-½ ancilla particle coupled via an optimised
  J_z/J_x ⊗ J_z/J_x interaction during the holding time.

- **Configuration B (two-system-particle)**: 2 particles in the two-mode
  interferometer system, no ancilla.

Physical Model:
- Two-mode bosonic system with J_z = (n₀ - n₁)/2, J_x = (a₀†a₁ + a₁†a₀)/2
- Spin-½ ancilla with J_z = σ_z/2, J_x = σ_x/2
- MZI circuit: BS1 → [Hold: exp(-i T_H (θ J_z + H_int))] → BS2
- Beam splitter: U_BS = exp(-i (π/2) J_x) (50/50, φ_BS = 0)
- Under this convention: BS† J_z BS = -J_y

Units:
- Dimensionless throughout. θ is the unknown phase (radians).
- T_H: holding-time strength parameter.
- α coefficients: real coupling strengths.

References:
- Giovannetti, Lloyd, Maccone, Nat. Photonics 5, 222 (2011)
- Demkowicz-Dobrzanski & Maccone, Phys. Rev. Lett. 113, 250801 (2014)
- Paris, Int. J. Quantum Inf. 7, 125 (2009)
- Yurke, McCall, Klauder, Phys. Rev. A 33, 4033 (1986)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from src.analysis.fisher_information import quantum_fisher_information_dm
from src.physics.mzi_simulation import beam_splitter_unitary, create_system_operators


# =============================================================================
# Operator Construction
# =============================================================================


def build_system_jz_jx(N_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Build J_z and J_x for a two-mode Fock space.

    J_z = (n₀ - n₁) / 2  (diagonal in Fock basis)
    J_x = (a₀† a₁ + a₁† a₀) / 2

    Args:
        N_max: Maximum photon number per mode. Hilbert space dimension
            is (N_max + 1)².

    Returns:
        Tuple (J_z, J_x) of Hermitian operators of dimension (N_max+1)².
    """
    dim = (N_max + 1) ** 2

    # J_z: diagonal
    J_z = np.zeros((dim, dim), dtype=complex)
    for n0 in range(N_max + 1):
        for n1 in range(N_max + 1):
            idx = n0 * (N_max + 1) + n1
            J_z[idx, idx] = (n0 - n1) / 2.0

    # J_x = (a₀† a₁ + a₁† a₀) / 2
    a0, a1, a0_dag, a1_dag = create_system_operators(N_max)
    J_x = 0.5 * (a0_dag @ a1 + a1_dag @ a0)

    return J_z, J_x


def build_ancilla_operators() -> tuple[np.ndarray, np.ndarray]:
    """Build J_z and J_x for spin-½ ancilla.

    J_z = σ_z / 2,  J_x = σ_x / 2

    Returns:
        Tuple (J_z_anc, J_x_anc) of 2×2 Hermitian operators.
    """
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    return sigma_z / 2.0, sigma_x / 2.0


def build_interaction_hamiltonian(
    alphas: tuple[float, float, float, float],
    J_z_sys: np.ndarray,
    J_x_sys: np.ndarray,
    J_z_anc: np.ndarray,
    J_x_anc: np.ndarray,
) -> np.ndarray:
    """Build ancilla interaction Hamiltonian H_int.

    H_int = α_zz · J_z ⊗ J_z  +  α_zx · J_z ⊗ J_x
          + α_xz · J_x ⊗ J_z  +  α_xx · J_x ⊗ J_x

    Args:
        alphas: (α_zz, α_zx, α_xz, α_xx) coupling coefficients.
        J_z_sys: J_z operator on system space.
        J_x_sys: J_x operator on system space.
        J_z_anc: J_z operator on ancilla space.
        J_x_anc: J_x operator on ancilla space.

    Returns:
        Full H_int matrix of dimension (dim_sys × 2).
    """
    a_zz, a_zx, a_xz, a_xx = alphas
    dim_sys = J_z_sys.shape[0]
    dim_full = dim_sys * 2

    H = np.zeros((dim_full, dim_full), dtype=complex)

    if a_zz != 0.0:
        H += a_zz * np.kron(J_z_sys, J_z_anc)
    if a_zx != 0.0:
        H += a_zx * np.kron(J_z_sys, J_x_anc)
    if a_xz != 0.0:
        H += a_xz * np.kron(J_x_sys, J_z_anc)
    if a_xx != 0.0:
        H += a_xx * np.kron(J_x_sys, J_x_anc)

    return H


# =============================================================================
# Generator Construction
# =============================================================================


def compute_generator_B(T_H: float, N_max: int) -> np.ndarray:
    """Compute the effective generator G_B for Case B (2 particles, no ancilla).

    G_B = T_H · BS₁† · J_z · BS₁

    With the BS convention (θ_BS = π/4, φ_BS = 0), this equals -T_H · J_y.

    Args:
        T_H: Holding-time strength parameter.
        N_max: Maximum photon number per mode.

    Returns:
        Generator matrix of dimension (N_max+1)² × (N_max+1)².
    """
    J_z_sys, _ = build_system_jz_jx(N_max)
    BS = beam_splitter_unitary(np.pi / 4, 0.0, N_max)

    G_B = T_H * BS.conj().T @ J_z_sys @ BS
    return G_B


def compute_generator_A(
    T_H: float,
    alphas: tuple[float, float, float, float],
    N_max: int,
    n_quadrature: int = 50,
) -> np.ndarray:
    """Compute G_A for Case A (1 system particle + ancilla) at reference θ = 0.

    G_A = T_H · (BS₁† ⊗ I_anc) · [∫₀¹ J_z(s) ds] · (BS₁ ⊗ I_anc)

    where J_z(s) = exp(i s T_H H_int) · (J_z ⊗ I) · exp(-i s T_H H_int).

    When [J_z, H_int] = 0 (only α_zz, α_zx terms), J_z(s) = J_z ⊗ I
    is independent of s, and G_A = T_H · (BS₁† · J_z · BS₁) ⊗ I = -T_H · J_y ⊗ I.

    When [J_z, H_int] ≠ 0, the integral mixes components and G_A differs.

    Args:
        T_H: Holding-time strength parameter.
        alphas: (α_zz, α_zx, α_xz, α_xx) coupling coefficients.
        N_max: Maximum photon number per mode for the system.
        n_quadrature: Number of quadrature points for the integral (default 50).

    Returns:
        Generator matrix of dimension (N_max+1)² × 2 (system + ancilla).
    """
    J_z_sys, J_x_sys = build_system_jz_jx(N_max)
    J_z_anc, J_x_anc = build_ancilla_operators()

    dim_sys = (N_max + 1) ** 2
    dim_full = dim_sys * 2

    I_anc = np.eye(2, dtype=complex)

    # Full operator: J_z ⊗ I_anc
    J_z_full = np.kron(J_z_sys, I_anc)

    # Interaction Hamiltonian
    H_int = build_interaction_hamiltonian(alphas, J_z_sys, J_x_sys, J_z_anc, J_x_anc)

    # BS on system, identity on ancilla
    BS = beam_splitter_unitary(np.pi / 4, 0.0, N_max)
    BS_full = np.kron(BS, I_anc)

    # Compute ∫₀¹ J_z(s) ds via numerical quadrature (Simpson's rule)
    s_points = np.linspace(0, 1, n_quadrature)
    J_z_vals = np.zeros((n_quadrature, dim_full, dim_full), dtype=complex)

    for k, s in enumerate(s_points):
        U_s = expm(1j * s * T_H * H_int)
        J_z_vals[k] = U_s @ J_z_full @ U_s.conj().T

    # Simpson integration
    h = 1.0 / (n_quadrature - 1)
    J_z_integral = np.zeros_like(J_z_full)
    J_z_integral += J_z_vals[0] + J_z_vals[-1]  # endpoints
    J_z_integral += 4.0 * np.sum(J_z_vals[1:-1:2], axis=0)  # odd
    J_z_integral += 2.0 * np.sum(J_z_vals[2:-1:2], axis=0)  # even
    J_z_integral *= h / 3.0

    # G_A = T_H · BS_full† · J_z_integral · BS_full
    G_A = T_H * BS_full.conj().T @ J_z_integral @ BS_full

    # Ensure Hermiticity
    G_A = 0.5 * (G_A + G_A.conj().T)

    return G_A


def compute_generator_A_at_theta(
    T_H: float,
    theta: float,
    alphas: tuple[float, float, float, float],
    N_max: int,
    n_quadrature: int = 50,
) -> np.ndarray:
    """Compute G_A at a non-zero reference θ.

    Same as compute_generator_A but with the full θ-dependent Hamiltonian.
    J_z(s) = exp(i s T_H (θ J_z + H_int)) · (J_z ⊗ I) · exp(-i s T_H (θ J_z + H_int))

    Args:
        T_H: Holding-time strength parameter.
        theta: Reference phase value.
        alphas: (α_zz, α_zx, α_xz, α_xx) coupling coefficients.
        N_max: Maximum photon number per mode.
        n_quadrature: Number of quadrature points (default 50).

    Returns:
        Generator matrix for Case A at given θ.
    """
    J_z_sys, J_x_sys = build_system_jz_jx(N_max)
    J_z_anc, J_x_anc = build_ancilla_operators()

    dim_sys = (N_max + 1) ** 2
    dim_full = dim_sys * 2

    I_anc = np.eye(2, dtype=complex)

    J_z_full = np.kron(J_z_sys, I_anc)
    H_int = build_interaction_hamiltonian(alphas, J_z_sys, J_x_sys, J_z_anc, J_x_anc)
    H_total = theta * J_z_full + H_int

    BS = beam_splitter_unitary(np.pi / 4, 0.0, N_max)
    BS_full = np.kron(BS, I_anc)

    s_points = np.linspace(0, 1, n_quadrature)
    J_z_vals = np.zeros((n_quadrature, dim_full, dim_full), dtype=complex)

    for k, s in enumerate(s_points):
        U_s = expm(1j * s * T_H * H_total)
        J_z_vals[k] = U_s @ J_z_full @ U_s.conj().T

    h = 1.0 / (n_quadrature - 1)
    J_z_integral = np.zeros_like(J_z_full)
    J_z_integral += J_z_vals[0] + J_z_vals[-1]
    J_z_integral += 4.0 * np.sum(J_z_vals[1:-1:2], axis=0)
    J_z_integral += 2.0 * np.sum(J_z_vals[2:-1:2], axis=0)
    J_z_integral *= h / 3.0

    G_A = T_H * BS_full.conj().T @ J_z_integral @ BS_full
    G_A = 0.5 * (G_A + G_A.conj().T)
    return G_A


# =============================================================================
# Density Matrix Utilities
# =============================================================================


def random_density_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random density matrix via Cholesky decomposition.

    Guarantees: ρ ≥ 0, Tr(ρ) = 1, ρ = ρ†.

    Args:
        d: Dimension of the density matrix.
        rng: NumPy random generator for reproducibility.

    Returns:
        Random density matrix of shape (d, d).
    """
    # Lower-triangular matrix with complex entries
    T = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    T = np.tril(T)

    ρ = T @ T.conj().T
    ρ /= np.trace(ρ)
    return ρ


def random_pure_state_dm(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random pure-state density matrix.

    Args:
        d: Dimension.
        rng: NumPy random generator.

    Returns:
        Pure state density matrix of shape (d, d).
    """
    ψ = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    ψ = ψ / np.linalg.norm(ψ)
    return np.outer(ψ, ψ.conj())


def _subspace_indices(N_max: int, target_N: int) -> np.ndarray:
    """Get indices in the two-mode Fock basis for a given total particle number.

    Args:
        N_max: Maximum photon number per mode.
        target_N: Total particle number n₀ + n₁ = target_N.

    Returns:
        Array of indices into the (N_max+1)²-dimensional Fock basis vectors
        that have the specified total particle number.
    """
    indices = []
    for n0 in range(N_max + 1):
        n1 = target_N - n0
        if 0 <= n1 <= N_max:
            idx = n0 * (N_max + 1) + n1
            indices.append(idx)
    return np.array(indices, dtype=int)


def random_pure_state_in_subspace(
    d_full: int,
    subspace_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random pure state restricted to a subspace.

    The state only has support on the specified indices, with random
    coefficients in that subspace and zero elsewhere.

    Args:
        d_full: Full Hilbert space dimension.
        subspace_idx: Indices of the subspace basis states.
        rng: NumPy random generator.

    Returns:
        Pure state density matrix of shape (d_full, d_full)
        supported only on the given subspace indices.
    """
    d_sub = len(subspace_idx)
    # Random state in the subspace
    psi_sub = rng.standard_normal(d_sub) + 1j * rng.standard_normal(d_sub)
    psi_sub = psi_sub / np.linalg.norm(psi_sub)

    # Embed in full space
    psi_full = np.zeros(d_full, dtype=complex)
    psi_full[subspace_idx] = psi_sub
    return np.outer(psi_full, psi_full.conj())


def random_alphas(
    rng: np.random.Generator, scale: float = 10.0
) -> tuple[float, float, float, float]:
    """Generate random interaction coefficients α.

    Args:
        rng: NumPy random generator.
        scale: Max absolute value for coefficients (default 10).

    Returns:
        Tuple (α_zz, α_zx, α_xz, α_xx).
    """
    vals = rng.uniform(-scale, scale, size=4)
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def check_particle_number(
    rho: np.ndarray,
    N_max: int,
    atol: float = 1e-6,
) -> tuple[float, float, float]:
    """Check the particle-number properties of a state.

    Computes ⟨N⟩, population of |0,0⟩, and population of |N_max,N_max⟩
    to verify the particle-number constraint.

    Args:
        rho: Density matrix in the two-mode Fock basis.
        N_max: Maximum photon number per mode.
        atol: Absolute tolerance for checks.

    Returns:
        Tuple (mean_N, pop_00, pop_NN) where:
        - mean_N: ⟨n₀ + n₁⟩
        - pop_00: population ρ_{|0,0⟩⟨0,0|}
        - pop_NN: population ρ_{|N_max,N_max⟩⟨N_max,N_max|}
    """
    mean_N = 0.0

    for n0 in range(N_max + 1):
        for n1 in range(N_max + 1):
            idx = n0 * (N_max + 1) + n1
            prob = np.real(rho[idx, idx])
            mean_N += prob * (n0 + n1)

    idx_00 = 0  # n0=0, n1=0
    idx_NN = N_max * (N_max + 1) + N_max  # n0=N_max, n1=N_max

    pop_00 = np.real(rho[idx_00, idx_00])
    pop_NN = np.real(rho[idx_NN, idx_NN])

    return float(mean_N), float(pop_00), float(pop_NN)


# =============================================================================
# QFI Evaluation
# =============================================================================


def evaluate_qfi_case_B(
    rho: np.ndarray,
    T_H: float,
    N_max: int,
) -> float:
    """Evaluate Quantum Fisher Information for Case B.

    Args:
        rho: Density matrix of dimension (N_max+1)².
        T_H: Holding-time strength.
        N_max: Maximum photon number per mode.

    Returns:
        QFI value F_Q.
    """
    G_B = compute_generator_B(T_H, N_max)
    F_Q = quantum_fisher_information_dm(rho, G_B)
    return float(F_Q)


def evaluate_qfi_case_A(
    rho: np.ndarray,
    T_H: float,
    alphas: tuple[float, float, float, float],
    N_max: int,
    n_quadrature: int = 50,
) -> float:
    """Evaluate Quantum Fisher Information for Case A at θ = 0.

    Args:
        rho: Density matrix of dimension (N_max+1)² × 2.
        T_H: Holding-time strength.
        alphas: (α_zz, α_zx, α_xz, α_xx) coupling coefficients.
        N_max: Maximum photon number per mode for the system.
        n_quadrature: Quadrature points for integral.

    Returns:
        QFI value F_Q.
    """
    G_A = compute_generator_A(T_H, alphas, N_max, n_quadrature)
    F_Q = quantum_fisher_information_dm(rho, G_A)
    return float(F_Q)


# =============================================================================
# Optimisation (Random Search)
# =============================================================================


@dataclass
class RandomSearchResult:
    """Result of a random search for optimal QFI."""

    max_fq: float
    best_rho: np.ndarray
    best_alphas: tuple[float, float, float, float] | None
    all_fq: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_N: float = 0.0
    pop_00: float = 0.0
    pop_NN: float = 0.0


def optimize_qfi_case_B(
    T_H: float,
    N_max: int,
    n_samples: int = 1000,
    pure_only: bool = False,
    subspace_N: int | None = None,
    seed: int | None = None,
) -> RandomSearchResult:
    """Optimise QFI for Case B via random search.

    Args:
        T_H: Holding-time strength.
        N_max: Maximum photon number per mode.
        n_samples: Number of random states to evaluate.
        pure_only: If True, restrict to pure states.
        subspace_N: If set, restrict sampling to the subspace with
            n₀ + n₁ = subspace_N (e.g., 2 for the 2-particle sector).
        seed: Random seed for reproducibility.

    Returns:
        RandomSearchResult with best QFI and state found.
    """
    rng = np.random.default_rng(seed)
    dim = (N_max + 1) ** 2
    G_B = compute_generator_B(T_H, N_max)

    # Pre-compute subspace indices if requested
    sub_idx = _subspace_indices(N_max, subspace_N) if subspace_N is not None else None

    best_fq = -1.0
    best_rho = None
    all_fq = np.zeros(n_samples)

    for i in range(n_samples):
        if sub_idx is not None:
            rho = random_pure_state_in_subspace(dim, sub_idx, rng)
        elif pure_only:
            rho = random_pure_state_dm(dim, rng)
        else:
            rho = random_density_matrix(dim, rng)

        F_Q = quantum_fisher_information_dm(rho, G_B)
        all_fq[i] = F_Q

        if F_Q > best_fq:
            best_fq = F_Q
            best_rho = rho.copy()

    assert best_rho is not None, "No valid states found"

    mean_N, pop_00, pop_NN = check_particle_number(best_rho, N_max)

    return RandomSearchResult(
        max_fq=best_fq,
        best_rho=best_rho,
        best_alphas=None,
        all_fq=all_fq,
        mean_N=mean_N,
        pop_00=pop_00,
        pop_NN=pop_NN,
    )


def optimize_qfi_case_A(
    T_H: float,
    N_max: int,
    n_samples: int = 2000,
    n_alpha_samples: int = 100,
    pure_only: bool = False,
    subspace_N: int | None = None,
    particle_penalty: float = 100.0,
    seed: int | None = None,
) -> RandomSearchResult:
    """Optimise QFI for Case A via random search over ρ and α.

    Imposes a particle-number penalty to ensure ⟨N⟩ ≈ 1.

    Args:
        T_H: Holding-time strength.
        N_max: Maximum photon number per mode for the system.
        n_samples: Number of random states per α sample.
        n_alpha_samples: Number of random α vectors to try.
        pure_only: If True, restrict to pure states.
        subspace_N: If set, restrict system to the n₀ + n₁ = subspace_N
            subspace (e.g., 1 for the 1-particle sector).
        particle_penalty: Penalty strength for ⟨N⟩ ≠ 1.
        seed: Random seed for reproducibility.

    Returns:
        RandomSearchResult with best QFI, ρ, and α found.
    """
    rng = np.random.default_rng(seed)
    dim_sys = (N_max + 1) ** 2
    dim_full = dim_sys * 2

    # Pre-compute system subspace indices
    sub_idx = _subspace_indices(N_max, subspace_N) if subspace_N is not None else None

    best_fq = -1.0
    best_rho: np.ndarray | None = None
    best_alphas: tuple[float, float, float, float] | None = None
    all_fq = []

    J_z_sys, J_x_sys = build_system_jz_jx(N_max)
    J_z_anc, J_x_anc = build_ancilla_operators()

    # Pre-compute the BS unitary
    BS = beam_splitter_unitary(np.pi / 4, 0.0, N_max)
    I_anc = np.eye(2, dtype=complex)
    BS_full = np.kron(BS, I_anc)

    for _alpha_iter in range(n_alpha_samples):
        alphas = random_alphas(rng, scale=5.0)

        # Build H_int
        H_int = build_interaction_hamiltonian(
            alphas, J_z_sys, J_x_sys, J_z_anc, J_x_anc
        )

        # Build full Hamiltonian at θ = 0
        J_z_full = np.kron(J_z_sys, I_anc)
        H_total = H_int  # θ = 0

        # Compute J_z(s) integral at θ = 0
        s_points = np.linspace(0, 1, 50)
        J_z_vals = np.zeros((50, dim_full, dim_full), dtype=complex)
        for k, s in enumerate(s_points):
            U_s = expm(1j * s * T_H * H_total)
            J_z_vals[k] = U_s @ J_z_full @ U_s.conj().T

        h = 1.0 / 49.0
        J_z_integral = J_z_vals[0] + J_z_vals[-1]
        J_z_integral += 4.0 * np.sum(J_z_vals[1:-1:2], axis=0)
        J_z_integral += 2.0 * np.sum(J_z_vals[2:-1:2], axis=0)
        J_z_integral *= h / 3.0

        # G_A for this α
        G_A = T_H * BS_full.conj().T @ J_z_integral @ BS_full
        G_A = 0.5 * (G_A + G_A.conj().T)

        # Sample random states for this α
        for _state_iter in range(n_samples // n_alpha_samples):
            if sub_idx is not None:
                # Build full density matrix: system in N=1 subspace × ancilla
                psi_sys = random_pure_state_in_subspace(dim_sys, sub_idx, rng)
                # Random ancilla pure state
                psi_anc = random_pure_state_dm(2, rng)
                rho = np.kron(psi_sys, psi_anc)
            elif pure_only:
                rho = random_pure_state_dm(dim_full, rng)
            else:
                rho = random_density_matrix(dim_full, rng)

            F_Q = quantum_fisher_information_dm(rho, G_A)

            # Apply particle-number penalty
            mean_N_sys, _, _ = check_particle_number(
                _partial_trace_system(rho, N_max), N_max
            )
            penalty = particle_penalty * (mean_N_sys - 1.0) ** 2
            F_Q_penalized = F_Q - penalty

            all_fq.append(F_Q_penalized)

            if F_Q_penalized > best_fq:
                best_fq = F_Q_penalized
                best_rho = rho.copy()
                best_alphas = alphas

    assert best_rho is not None, "No valid states found"
    assert best_alphas is not None

    # Re-evaluate the best without penalty to report true QFI
    # Need to check particle number of the best state
    mean_N, pop_00, pop_NN = check_particle_number(
        _partial_trace_system(best_rho, N_max), N_max
    )

    return RandomSearchResult(
        max_fq=best_fq,  # penalized score
        best_rho=best_rho,
        best_alphas=best_alphas,
        all_fq=np.array(all_fq),
        mean_N=mean_N,
        pop_00=pop_00,
        pop_NN=pop_NN,
    )


def _partial_trace_system(rho_full: np.ndarray, N_max: int) -> np.ndarray:
    """Extract the system (two-mode Fock) reduced density matrix.

    Traces out the ancilla from the full system-ancilla density matrix.

    Args:
        rho_full: Full density matrix of dim (sys_dim × 2).
        N_max: Maximum photon number per mode.

    Returns:
        Reduced system density matrix.
    """
    dim_sys = (N_max + 1) ** 2
    dim_anc = 2

    rho_reshaped = rho_full.reshape(dim_sys, dim_anc, dim_sys, dim_anc)
    rho_sys = np.trace(rho_reshaped, axis1=1, axis2=3)
    return rho_sys


# =============================================================================
# Comparison Runner
# =============================================================================


@dataclass
class ComparisonResult:
    """Results of the ancilla vs. system comparison.

    Attributes:
        fq_A_max: Best QFI found for Case A (ancilla-assisted).
        fq_B_max: Best QFI found for Case B (two-particle system).
        fq_A_zero: QFI for Case A with α = 0 (baseline).
        ratio: sqrt(F_B / F_A) = Δθ_A / Δθ_B.
        fq_A_theta: QFI at different reference θ values (optional).
        fq_A_all: All QFI values from random search for Case A.
        fq_B_all: All QFI values from random search for Case B.
    """

    fq_A_max: float
    fq_B_max: float
    fq_A_zero: float
    ratio: float
    fq_A_theta: dict[float, float] = field(default_factory=dict)
    fq_A_all: np.ndarray = field(default_factory=lambda: np.array([]))
    fq_B_all: np.ndarray = field(default_factory=lambda: np.array([]))
    best_alphas_A: tuple[float, float, float, float] | None = None
    mean_N_A: float = 0.0
    pop_00_A: float = 0.0
    pop_NN_A: float = 0.0
    mean_N_B: float = 0.0
    pop_00_B: float = 0.0
    pop_NN_B: float = 0.0


def run_comparison(
    T_H: float = 1.0,
    n_samples_B: int = 2000,
    n_samples_A: int = 3000,
    n_alpha_samples: int = 50,
    n_quadrature: int = 50,
    pure_only: bool = False,
    particle_penalty: float = 100.0,
    theta_values: tuple[float, ...] = (0.0, 0.1, 0.5),
    seed: int | None = 42,
) -> ComparisonResult:
    """Run the full comparison between Case A and Case B.

    Uses subspace-restricted sampling: Case B restricts to N=2 subspace,
    Case A restricts to N=1 subspace for the system.

    Args:
        T_H: Holding-time strength (default 1.0).
        n_samples_B: Number of random states for Case B optimisation.
        n_samples_A: Number of random states for Case A optimisation.
        n_alpha_samples: Number of random α vectors for Case A.
        n_quadrature: Quadrature points for the integral.
        pure_only: Restrict to pure states.
        particle_penalty: Penalty strength for ⟨N⟩ ≠ 1.
        theta_values: Reference θ values for θ-dependence check.
        seed: Random seed (default 42 for reproducibility).

    Returns:
        ComparisonResult with all findings.
    """
    # Case B: 2 particles, N_max = 2, restrict to N=2 subspace
    result_B = optimize_qfi_case_B(
        T_H=T_H,
        N_max=2,
        n_samples=n_samples_B,
        pure_only=True,
        subspace_N=2,
        seed=seed,
    )

    # Case A: 1 system particle + ancilla, N_max = 1, restrict to N=1 subspace
    result_A = optimize_qfi_case_A(
        T_H=T_H,
        N_max=1,
        n_samples=n_samples_A,
        n_alpha_samples=n_alpha_samples,
        pure_only=True,
        subspace_N=1,
        particle_penalty=particle_penalty,
        seed=seed if seed is None else seed + 1,
    )

    # Case A baseline: α = 0 (no interaction), N=1 subspace
    dim_sys = (1 + 1) ** 2  # N_max = 1 → dim 4
    rng = np.random.default_rng(seed if seed is None else seed + 2)
    G_A_zero = compute_generator_A(T_H, (0.0, 0.0, 0.0, 0.0), 1, n_quadrature)
    sub_idx = _subspace_indices(1, 1)

    best_fq_A_zero = -1.0
    for _ in range(n_samples_A // 2):
        psi_sys = random_pure_state_in_subspace(dim_sys, sub_idx, rng)
        psi_anc = random_pure_state_dm(2, rng)
        rho = np.kron(psi_sys, psi_anc)
        F_Q = quantum_fisher_information_dm(rho, G_A_zero)
        if F_Q > best_fq_A_zero:
            best_fq_A_zero = F_Q

    # θ-dependence check
    fq_A_theta: dict[float, float] = {}
    if result_A.best_alphas is not None:
        best_rho_A = result_A.best_rho
        for theta_val in theta_values:
            if theta_val == 0.0:
                G_A_theta = compute_generator_A(
                    T_H, result_A.best_alphas, 1, n_quadrature
                )
            else:
                G_A_theta = compute_generator_A_at_theta(
                    T_H, theta_val, result_A.best_alphas, 1, n_quadrature
                )
            F_Q = quantum_fisher_information_dm(best_rho_A, G_A_theta)
            fq_A_theta[theta_val] = float(F_Q)

    # Ratio
    ratio = (
        np.sqrt(result_B.max_fq / result_A.max_fq)
        if result_A.max_fq > 0
        else float("inf")
    )

    return ComparisonResult(
        fq_A_max=result_A.max_fq,
        fq_B_max=result_B.max_fq,
        fq_A_zero=best_fq_A_zero,
        ratio=ratio,
        fq_A_theta=fq_A_theta,
        fq_A_all=result_A.all_fq,
        fq_B_all=result_B.all_fq,
        best_alphas_A=result_A.best_alphas,
        mean_N_A=result_A.mean_N,
        pop_00_A=result_A.pop_00,
        pop_NN_A=result_A.pop_NN,
        mean_N_B=result_B.mean_N,
        pop_00_B=result_B.pop_00,
        pop_NN_B=result_B.pop_NN,
    )


# =============================================================================
# Analytical Checks
# =============================================================================


def analytical_fq_B_max(T_H: float) -> float:
    """Theoretical maximum QFI for Case B (2-particle system).

    For J = 1 (2 particles), J_y has eigenvalues {-1, 0, +1}.
    With optimal pure state: max F_Q = T_H² (λ_max - λ_min)² = 4 T_H².

    Args:
        T_H: Holding-time strength.

    Returns:
        Theoretical maximum QFI.
    """
    return 4.0 * T_H**2


def analytical_fq_A_zero(T_H: float) -> float:
    """Theoretical QFI for Case A with α = 0 (uncoupled ancilla).

    With α = 0, G_A = -T_H · J_y ⊗ I. For the 1-particle system (J = 1/2),
    J_y has eigenvalues {-1/2, +1/2}, so:
    max F_Q = T_H² (1/2 - (-1/2))² = T_H².

    Args:
        T_H: Holding-time strength.

    Returns:
        Theoretical maximum QFI with zero interaction.
    """
    return 1.0 * T_H**2

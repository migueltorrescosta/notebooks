"""
Ancilla-vs-system comparison for interferometric phase estimation.

Implements the full pipeline described in the 2026-05-11 ancilla comparison
report, comparing:
  - **Case A**: One system particle + ancilla qubit (2 particles total)
  - **Case B**: Two system particles alone (no ancilla)

Both cases use the same total particle number N=2, enabling a fair comparison
of the metrological advantage provided by the ancilla.

Functions are organised into:
  1. Operator construction (J_z, J_x, interaction H)
  2. Generator computation (G_A, G_B)
  3. Density matrix utilities (random states, particle-number checks)
  4. QFI evaluation
  5. Random search optimisation (Case A and Case B)
  6. Comparison runner
  7. Analytical upper bounds

Hilbert space conventions:
  - Two-mode Fock basis: index = n0 * (N_max + 1) + n1
  - System-ancilla combined: kron(system, ancilla) of dimension (N_max+1)^2 * 2
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import expm

from src.analysis.ancilla_optimization import J_X, J_Z
from src.analysis.fisher_information import quantum_fisher_information_dm
from src.physics.beam_splitter import bs_fock
from src.physics.mzi_simulation import create_system_operators
from src.physics.mzi_states import two_mode_jz_operator

# =============================================================================
# Operator Construction
# =============================================================================


def build_system_jz_jx(N_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Build J_z and J_x for a two-mode Fock space.

    J_z = (n_0 - n_1) / 2  (diagonal in Fock basis)
    J_x = (a_0^dagger a_1 + a_1^dagger a_0) / 2

    Args:
        N_max: Maximum photon number per mode. Hilbert space dimension
            is (N_max + 1)^2.

    Returns:
        Tuple (J_z, J_x) of Hermitian operators of dimension (N_max+1)^2.

    """
    J_z = two_mode_jz_operator(N_max)

    a0, a1, a0_dag, a1_dag = create_system_operators(N_max)
    J_x = 0.5 * (a0_dag @ a1 + a1_dag @ a0)

    return J_z, J_x


def build_interaction_hamiltonian(
    alphas: tuple[float, float, float, float],
    J_z_sys: np.ndarray,
    J_x_sys: np.ndarray,
    J_z_anc: np.ndarray,
    J_x_anc: np.ndarray,
) -> np.ndarray:
    """Build ancilla interaction Hamiltonian H_int.

    H_int = alpha_xx * J_x x J_x  +  alpha_xz * J_x x J_z
          + alpha_zx * J_z x J_x  +  alpha_zz * J_z x J_z

    Args:
        alphas: (alpha_xx, alpha_xz, alpha_zx, alpha_zz) coupling coefficients.
        J_z_sys: J_z operator on system space.
        J_x_sys: J_x operator on system space.
        J_z_anc: J_z operator on ancilla space.
        J_x_anc: J_x operator on ancilla space.

    Returns:
        Full H_int matrix of dimension (dim_sys * 2).

    """
    a_xx, a_xz, a_zx, a_zz = alphas
    dim_sys = J_z_sys.shape[0]
    dim_full = dim_sys * 2

    H = np.zeros((dim_full, dim_full), dtype=complex)

    if a_xx != 0.0:
        H += a_xx * np.kron(J_x_sys, J_x_anc)
    if a_xz != 0.0:
        H += a_xz * np.kron(J_x_sys, J_z_anc)
    if a_zx != 0.0:
        H += a_zx * np.kron(J_z_sys, J_x_anc)
    if a_zz != 0.0:
        H += a_zz * np.kron(J_z_sys, J_z_anc)

    return H


# =============================================================================
# Generator Construction
# =============================================================================


def compute_generator_B(T_hold: float, N_max: int) -> np.ndarray:
    """Compute the effective generator G_B for Case B (2 particles, no ancilla).

    G_B = T_hold * BS^dagger * J_z * BS

    With the BS convention (theta_BS = pi/4, phi_BS = 0), this equals -T_hold * J_y.

    Args:
        T_hold: Holding-time strength parameter.
        N_max: Maximum photon number per mode.

    Returns:
        Generator matrix of dimension (N_max+1)^2 x (N_max+1)^2.

    """
    J_z_sys, _ = build_system_jz_jx(N_max)
    BS = bs_fock(np.pi / 4, 0.0, N_max)

    return T_hold * BS.conj().T @ J_z_sys @ BS


def compute_generator_A(
    T_hold: float,
    alphas: tuple[float, float, float, float],
    N_max: int,
    n_quadrature: int = 50,
) -> np.ndarray:
    """Compute G_A for Case A (1 system particle + ancilla) at reference theta = 0.

    G_A = T_hold * (BS^dagger x I_anc) * [integral_0^1 J_z(s) ds] * (BS x I_anc)

    where J_z(s) = exp(i s T_hold H_int) * (J_z x I) * exp(-i s T_hold H_int).

    When [J_z, H_int] = 0 (only alpha_zz, alpha_zx terms), J_z(s) = J_z x I
    is independent of s, and G_A = T_hold * (BS^dagger * J_z * BS) x I = -T_hold * J_y x I.

    When [J_z, H_int] != 0, the integral mixes components and G_A differs.

    Args:
        T_hold: Holding-time strength parameter.
        alphas: (alpha_xx, alpha_xz, alpha_zx, alpha_zz) coupling coefficients.
        N_max: Maximum photon number per mode for the system.
        n_quadrature: Number of quadrature points for the integral (default 50).

    Returns:
        Generator matrix of dimension (N_max+1)^2 * 2 (system + ancilla).

    """
    J_z_sys, J_x_sys = build_system_jz_jx(N_max)
    J_z_anc, J_x_anc = J_Z, J_X

    dim_sys = (N_max + 1) ** 2
    dim_full = dim_sys * 2

    I_anc = np.eye(2, dtype=complex)

    # Full operator: J_z x I_anc
    J_z_full = np.kron(J_z_sys, I_anc)

    # Interaction Hamiltonian
    H_int = build_interaction_hamiltonian(alphas, J_z_sys, J_x_sys, J_z_anc, J_x_anc)

    # BS on system, identity on ancilla
    BS = bs_fock(np.pi / 4, 0.0, N_max)
    BS_full = np.kron(BS, I_anc)

    # Compute integral_0^1 J_z(s) ds via numerical quadrature (Simpson's rule)
    s_points = np.linspace(0, 1, n_quadrature)
    J_z_vals = np.zeros((n_quadrature, dim_full, dim_full), dtype=complex)

    for k, s in enumerate(s_points):
        U_s = expm(1j * s * T_hold * H_int)
        J_z_vals[k] = U_s @ J_z_full @ U_s.conj().T

    # Simpson integration
    h = 1.0 / (n_quadrature - 1)
    J_z_integral = np.zeros_like(J_z_full)
    J_z_integral += J_z_vals[0] + J_z_vals[-1]  # endpoints
    J_z_integral += 4.0 * np.sum(J_z_vals[1:-1:2], axis=0)  # odd
    J_z_integral += 2.0 * np.sum(J_z_vals[2:-1:2], axis=0)  # even
    J_z_integral *= h / 3.0

    # G_A = T_hold * BS_full^dagger * J_z_integral * BS_full
    G_A = T_hold * BS_full.conj().T @ J_z_integral @ BS_full

    # Ensure Hermiticity
    return 0.5 * (G_A + G_A.conj().T)


def compute_generator_A_at_omega(
    T_hold: float,
    omega: float,
    alphas: tuple[float, float, float, float],
    N_max: int,
    n_quadrature: int = 50,
) -> np.ndarray:
    """Compute G_A at a non-zero reference omega.

    Same as compute_generator_A but with the full omega-dependent Hamiltonian:
      J_z(s) = exp(i s T_hold (omega * J_z + H_int))
               * (J_z x I)
               * exp(-i s T_hold (omega * J_z + H_int))

    Args:
        T_hold: Holding-time strength parameter.
        omega: Reference phase rate value.
        alphas: (alpha_xx, alpha_xz, alpha_zx, alpha_zz) coupling coefficients.
        N_max: Maximum photon number per mode.
        n_quadrature: Number of quadrature points (default 50).

    Returns:
        Generator matrix for Case A at given omega.

    """
    J_z_sys, J_x_sys = build_system_jz_jx(N_max)
    J_z_anc, J_x_anc = J_Z, J_X

    dim_sys = (N_max + 1) ** 2
    dim_full = dim_sys * 2

    I_anc = np.eye(2, dtype=complex)

    J_z_full = np.kron(J_z_sys, I_anc)
    H_int = build_interaction_hamiltonian(alphas, J_z_sys, J_x_sys, J_z_anc, J_x_anc)
    H_total = omega * J_z_full + H_int

    BS = bs_fock(np.pi / 4, 0.0, N_max)
    BS_full = np.kron(BS, I_anc)

    s_points = np.linspace(0, 1, n_quadrature)
    J_z_vals = np.zeros((n_quadrature, dim_full, dim_full), dtype=complex)

    for k, s in enumerate(s_points):
        U_s = expm(1j * s * T_hold * H_total)
        J_z_vals[k] = U_s @ J_z_full @ U_s.conj().T

    h = 1.0 / (n_quadrature - 1)
    J_z_integral = np.zeros_like(J_z_full)
    J_z_integral += J_z_vals[0] + J_z_vals[-1]
    J_z_integral += 4.0 * np.sum(J_z_vals[1:-1:2], axis=0)
    J_z_integral += 2.0 * np.sum(J_z_vals[2:-1:2], axis=0)
    J_z_integral *= h / 3.0

    G_A = T_hold * BS_full.conj().T @ J_z_integral @ BS_full
    return 0.5 * (G_A + G_A.conj().T)


# =============================================================================
# Density Matrix Utilities
# =============================================================================


def random_density_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random density matrix via Cholesky decomposition.

    Guarantees: rho >= 0, Tr(rho) = 1, rho = rho^dagger.

    Args:
        d: Dimension of the density matrix.
        rng: NumPy random generator for reproducibility.

    Returns:
        Random density matrix of shape (d, d).

    """
    # Lower-triangular matrix with complex entries
    T = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    T = np.tril(T)

    rho = T @ T.conj().T
    rho /= np.trace(rho)
    return rho


def random_pure_state_dm(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random pure-state density matrix.

    Args:
        d: Dimension.
        rng: NumPy random generator.

    Returns:
        Pure state density matrix of shape (d, d).

    """
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def _subspace_indices(N_max: int, target_N: int) -> np.ndarray:
    """Get indices in the two-mode Fock basis for a given total particle number.

    Args:
        N_max: Maximum photon number per mode.
        target_N: Total particle number n_0 + n_1 = target_N.

    Returns:
        Array of indices into the (N_max+1)^2-dimensional Fock basis vectors
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
    rng: np.random.Generator,
    scale: float = 10.0,
) -> tuple[float, float, float, float]:
    """Generate random interaction coefficients alpha.

    Args:
        rng: NumPy random generator.
        scale: Max absolute value for coefficients (default 10).

    Returns:
        Tuple (alpha_xx, alpha_xz, alpha_zx, alpha_zz).

    """
    vals = rng.uniform(-scale, scale, size=4)
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def check_particle_number(
    rho: np.ndarray,
    N_max: int,
    atol: float = 1e-6,
) -> tuple[float, float, float]:
    """Check the particle-number properties of a state.

    Computes <N>, population of |0,0>, and population of |N_max,N_max>
    to verify the particle-number constraint.

    Args:
        rho: Density matrix in the two-mode Fock basis.
        N_max: Maximum photon number per mode.
        atol: Absolute tolerance for checks.

    Returns:
        Tuple (mean_N, pop_00, pop_NN) where:
        - mean_N: <n_0 + n_1>
        - pop_00: population rho_{|0,0><0,0|}
        - pop_NN: population rho_{|N_max,N_max><N_max,N_max|}

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
    T_hold: float,
    N_max: int,
) -> float:
    """Evaluate Quantum Fisher Information for Case B.

    Args:
        rho: Density matrix of dimension (N_max+1)^2.
        T_hold: Holding-time strength.
        N_max: Maximum photon number per mode.

    Returns:
        QFI value F_Q.

    """
    G_B = compute_generator_B(T_hold, N_max)
    F_Q = quantum_fisher_information_dm(rho, G_B)
    return float(F_Q)


def evaluate_qfi_case_A(
    rho: np.ndarray,
    T_hold: float,
    alphas: tuple[float, float, float, float],
    N_max: int,
    n_quadrature: int = 50,
) -> float:
    """Evaluate Quantum Fisher Information for Case A at theta = 0.

    Args:
        rho: Density matrix of dimension (N_max+1)^2 * 2.
        T_hold: Holding-time strength.
        alphas: (alpha_xx, alpha_xz, alpha_zx, alpha_zz) coupling coefficients.
        N_max: Maximum photon number per mode for the system.
        n_quadrature: Quadrature points for integral.

    Returns:
        QFI value F_Q.

    """
    G_A = compute_generator_A(T_hold, alphas, N_max, n_quadrature)
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

    def to_dataframe(self) -> pd.DataFrame:
        """Single-row DataFrame with scalar fields (arrays go to sidecar files)."""
        data: dict[str, Any] = {
            "max_fq": [self.max_fq],
            "mean_N": [self.mean_N],
            "pop_00": [self.pop_00],
            "pop_NN": [self.pop_NN],
            "best_rho_dim": [self.best_rho.shape[0]],
        }
        if self.best_alphas is not None:
            data["alpha_xx"] = [self.best_alphas[0]]
            data["alpha_xz"] = [self.best_alphas[1]]
            data["alpha_zx"] = [self.best_alphas[2]]
            data["alpha_zz"] = [self.best_alphas[3]]
        else:
            data["alpha_xx"] = [float("nan")]
            data["alpha_xz"] = [float("nan")]
            data["alpha_zx"] = [float("nan")]
            data["alpha_zz"] = [float("nan")]
        return pd.DataFrame(data)

    def save_parquet(self, path: str | Path) -> Path:
        """Save to main Parquet with sidecar files for best_rho and all_fq."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        stem = path.stem
        rho_path = path.with_stem(stem + "-best-rho")
        n_dim = self.best_rho.shape[0]
        rows: list[dict[str, object]] = []
        rows.extend(
            {
                "row": i,
                "col": j,
                "real": np.real(self.best_rho[i, j]),
                "imag": np.imag(self.best_rho[i, j]),
            }
            for i in range(n_dim)
            for j in range(n_dim)
        )
        pd.DataFrame(rows).to_parquet(rho_path, index=False)
        fq_path = path.with_stem(stem + "-all-fq")
        pd.DataFrame({"fq_values": self.all_fq}).to_parquet(fq_path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> RandomSearchResult:
        """Reconstruct from a Parquet file written by save_parquet()."""
        path = Path(path)
        df = pd.read_parquet(path)
        required = {
            "max_fq",
            "mean_N",
            "pop_00",
            "pop_NN",
            "best_rho_dim",
            "alpha_xx",
            "alpha_xz",
            "alpha_zx",
            "alpha_zz",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        stem = path.stem
        rho_path = path.with_stem(stem + "-best-rho")
        rho_df = pd.read_parquet(rho_path)
        dim = int(df["best_rho_dim"].iloc[0])
        best_rho = np.zeros((dim, dim), dtype=complex)
        for _, r in rho_df.iterrows():
            best_rho[int(r["row"]), int(r["col"])] = complex(r["real"], r["imag"])
        fq_path = path.with_stem(stem + "-all-fq")
        all_fq = pd.read_parquet(fq_path)["fq_values"].to_numpy(dtype=float)
        alpha_xx = float(df["alpha_xx"].iloc[0])
        best_alphas = None
        if not np.isnan(alpha_xx):
            best_alphas = (
                alpha_xx,
                float(df["alpha_xz"].iloc[0]),
                float(df["alpha_zx"].iloc[0]),
                float(df["alpha_zz"].iloc[0]),
            )
        return cls(
            max_fq=float(df["max_fq"].iloc[0]),
            best_rho=best_rho,
            best_alphas=best_alphas,
            all_fq=all_fq,
            mean_N=float(df["mean_N"].iloc[0]),
            pop_00=float(df["pop_00"].iloc[0]),
            pop_NN=float(df["pop_NN"].iloc[0]),
        )


def optimize_qfi_case_B(
    T_hold: float,
    N_max: int,
    n_samples: int = 1000,
    pure_only: bool = False,
    subspace_N: int | None = None,
    seed: int | None = None,
) -> RandomSearchResult:
    """Optimise QFI for Case B via random search.

    Args:
        T_hold: Holding-time strength.
        N_max: Maximum photon number per mode.
        n_samples: Number of random states to evaluate.
        pure_only: If True, restrict to pure states.
        subspace_N: If set, restrict sampling to the subspace with
            n_0 + n_1 = subspace_N (e.g., 2 for the 2-particle sector).
        seed: Random seed for reproducibility.

    Returns:
        RandomSearchResult with best QFI and state found.

    """
    rng = np.random.default_rng(seed)
    dim = (N_max + 1) ** 2
    G_B = compute_generator_B(T_hold, N_max)

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

        if best_fq < F_Q:
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
    T_hold: float,
    N_max: int,
    n_samples: int = 2000,
    n_alpha_samples: int = 100,
    pure_only: bool = False,
    subspace_N: int | None = None,
    particle_penalty: float = 100.0,
    seed: int | None = None,
) -> RandomSearchResult:
    """Optimise QFI for Case A via random search over rho and alpha.

    Imposes a particle-number penalty to ensure <N> approx 1.

    Args:
        T_hold: Holding-time strength.
        N_max: Maximum photon number per mode for the system.
        n_samples: Number of random states per alpha sample.
        n_alpha_samples: Number of random alpha vectors to try.
        pure_only: If True, restrict to pure states.
        subspace_N: If set, restrict system to the n_0 + n_1 = subspace_N
            subspace (e.g., 1 for the 1-particle sector).
        particle_penalty: Penalty strength for <N> != 1.
        seed: Random seed for reproducibility.

    Returns:
        RandomSearchResult with best QFI, rho, and alpha found.
        Alpha ordering follows the report convention: (alpha_xx, alpha_xz,
        alpha_zx, alpha_zz).

    """
    rng = np.random.default_rng(seed)
    dim_sys = (N_max + 1) ** 2
    dim_full = dim_sys * 2

    # Pre-compute system subspace indices
    sub_idx = _subspace_indices(N_max, subspace_N) if subspace_N is not None else None

    best_fq = -1.0
    best_rho: np.ndarray | None = None
    best_alphas: tuple[float, float, float, float] | None = None
    all_fq: list[float] = []

    J_z_sys, J_x_sys = build_system_jz_jx(N_max)
    J_z_anc, J_x_anc = J_Z, J_X

    # Pre-compute the BS unitary
    BS = bs_fock(np.pi / 4, 0.0, N_max)
    I_anc = np.eye(2, dtype=complex)
    BS_full = np.kron(BS, I_anc)

    for _alpha_iter in range(n_alpha_samples):
        alphas = random_alphas(rng, scale=5.0)

        # Build H_int
        H_int = build_interaction_hamiltonian(
            alphas,
            J_z_sys,
            J_x_sys,
            J_z_anc,
            J_x_anc,
        )

        # Build full Hamiltonian at theta = 0
        J_z_full = np.kron(J_z_sys, I_anc)
        H_total = H_int  # theta = 0

        # Compute J_z(s) integral at theta = 0
        s_points = np.linspace(0, 1, 50)
        J_z_vals = np.zeros((50, dim_full, dim_full), dtype=complex)
        for k, s in enumerate(s_points):
            U_s = expm(1j * s * T_hold * H_total)
            J_z_vals[k] = U_s @ J_z_full @ U_s.conj().T

        h = 1.0 / 49.0
        J_z_integral = J_z_vals[0] + J_z_vals[-1]
        J_z_integral += 4.0 * np.sum(J_z_vals[1:-1:2], axis=0)
        J_z_integral += 2.0 * np.sum(J_z_vals[2:-1:2], axis=0)
        J_z_integral *= h / 3.0

        # G_A for this alpha
        G_A = T_hold * BS_full.conj().T @ J_z_integral @ BS_full
        G_A = 0.5 * (G_A + G_A.conj().T)

        # Sample random states for this alpha
        for _state_iter in range(n_samples // n_alpha_samples):
            if sub_idx is not None:
                # Build full density matrix: system in N=1 subspace x ancilla
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
                _partial_trace_system(rho, N_max),
                N_max,
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
    mean_N, pop_00, pop_NN = check_particle_number(
        _partial_trace_system(best_rho, N_max),
        N_max,
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
        rho_full: Full density matrix of dim (sys_dim * 2).
        N_max: Maximum photon number per mode.

    Returns:
        Reduced system density matrix.

    """
    dim_sys = (N_max + 1) ** 2
    dim_anc = 2

    rho_reshaped = rho_full.reshape(dim_sys, dim_anc, dim_sys, dim_anc)
    return np.trace(rho_reshaped, axis1=1, axis2=3)


# =============================================================================
# Comparison Runner
# =============================================================================


@dataclass
class ComparisonResult:
    """Results of the ancilla vs. system comparison.

    Attributes:
        fq_A_max: Best QFI found for Case A (ancilla-assisted).
        fq_B_max: Best QFI found for Case B (two-particle system).
        fq_A_zero: QFI for Case A with alpha = 0 (baseline).
        ratio: sqrt(F_B / F_A) = Dtheta_A / Dtheta_B.
        fq_A_omega: QFI at different reference omega values (optional).

    """

    fq_A_max: float
    fq_B_max: float
    fq_A_zero: float
    ratio: float
    fq_A_omega: dict[float, float] = field(default_factory=dict)
    fq_A_all: np.ndarray = field(default_factory=lambda: np.array([]))
    fq_B_all: np.ndarray = field(default_factory=lambda: np.array([]))
    best_alphas_A: tuple[float, float, float, float] | None = None
    mean_N_A: float = 0.0
    pop_00_A: float = 0.0
    pop_NN_A: float = 0.0
    mean_N_B: float = 0.0
    pop_00_B: float = 0.0
    pop_NN_B: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Single-row DataFrame with scalar fields (arrays go to sidecar files)."""
        data: dict[str, Any] = {
            "fq_A_max": [self.fq_A_max],
            "fq_B_max": [self.fq_B_max],
            "fq_A_zero": [self.fq_A_zero],
            "ratio": [self.ratio],
            "fq_A_omega": [json.dumps(self.fq_A_omega)],
            "mean_N_A": [self.mean_N_A],
            "pop_00_A": [self.pop_00_A],
            "pop_NN_A": [self.pop_NN_A],
            "mean_N_B": [self.mean_N_B],
            "pop_00_B": [self.pop_00_B],
            "pop_NN_B": [self.pop_NN_B],
        }
        if self.best_alphas_A is not None:
            data["alpha_xx"] = [self.best_alphas_A[0]]
            data["alpha_xz"] = [self.best_alphas_A[1]]
            data["alpha_zx"] = [self.best_alphas_A[2]]
            data["alpha_zz"] = [self.best_alphas_A[3]]
        else:
            data["alpha_xx"] = [float("nan")]
            data["alpha_xz"] = [float("nan")]
            data["alpha_zx"] = [float("nan")]
            data["alpha_zz"] = [float("nan")]
        return pd.DataFrame(data)

    def save_parquet(self, path: str | Path) -> Path:
        """Save to main Parquet with sidecar files for fq_A_all and fq_B_all."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        stem = path.stem
        fq_a_path = path.with_stem(stem + "-fq-A-all")
        pd.DataFrame({"fq_values": self.fq_A_all}).to_parquet(fq_a_path, index=False)
        fq_b_path = path.with_stem(stem + "-fq-B-all")
        pd.DataFrame({"fq_values": self.fq_B_all}).to_parquet(fq_b_path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> ComparisonResult:
        """Reconstruct from a Parquet file written by save_parquet()."""
        path = Path(path)
        df = pd.read_parquet(path)
        required = {
            "fq_A_max",
            "fq_B_max",
            "fq_A_zero",
            "ratio",
            "fq_A_omega",
            "mean_N_A",
            "pop_00_A",
            "pop_NN_A",
            "mean_N_B",
            "pop_00_B",
            "pop_NN_B",
            "alpha_xx",
            "alpha_xz",
            "alpha_zx",
            "alpha_zz",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        stem = path.stem
        fq_a_path = path.with_stem(stem + "-fq-A-all")
        fq_A_all = pd.read_parquet(fq_a_path)["fq_values"].to_numpy(dtype=float)
        fq_b_path = path.with_stem(stem + "-fq-B-all")
        fq_B_all = pd.read_parquet(fq_b_path)["fq_values"].to_numpy(dtype=float)
        alpha_xx = float(df["alpha_xx"].iloc[0])
        best_alphas_A = None
        if not np.isnan(alpha_xx):
            best_alphas_A = (
                alpha_xx,
                float(df["alpha_xz"].iloc[0]),
                float(df["alpha_zx"].iloc[0]),
                float(df["alpha_zz"].iloc[0]),
            )
        return cls(
            fq_A_max=float(df["fq_A_max"].iloc[0]),
            fq_B_max=float(df["fq_B_max"].iloc[0]),
            fq_A_zero=float(df["fq_A_zero"].iloc[0]),
            ratio=float(df["ratio"].iloc[0]),
            fq_A_omega=json.loads(str(df["fq_A_omega"].iloc[0])),
            fq_A_all=fq_A_all,
            fq_B_all=fq_B_all,
            best_alphas_A=best_alphas_A,
            mean_N_A=float(df["mean_N_A"].iloc[0]),
            pop_00_A=float(df["pop_00_A"].iloc[0]),
            pop_NN_A=float(df["pop_NN_A"].iloc[0]),
            mean_N_B=float(df["mean_N_B"].iloc[0]),
            pop_00_B=float(df["pop_00_B"].iloc[0]),
            pop_NN_B=float(df["pop_NN_B"].iloc[0]),
        )


def run_comparison(
    T_hold: float = 1.0,
    n_samples_B: int = 2000,
    n_samples_A: int = 3000,
    n_alpha_samples: int = 50,
    n_quadrature: int = 50,
    pure_only: bool = False,
    particle_penalty: float = 100.0,
    omega_values: tuple[float, ...] = (0.0, 0.1, 0.5),
    seed: int | None = 42,
) -> ComparisonResult:
    """Run the full comparison between Case A and Case B.

    Uses subspace-restricted sampling: Case B restricts to N=2 subspace,
    Case A restricts to N=1 subspace for the system.

    Args:
        T_hold: Holding-time strength (default 1.0).
        n_samples_B: Number of random states for Case B optimisation.
        n_samples_A: Number of random states for Case A optimisation.
        n_alpha_samples: Number of random alpha vectors for Case A.
        n_quadrature: Quadrature points for the integral.
        pure_only: Restrict to pure states.
        particle_penalty: Penalty strength for <N> != 1.
        omega_values: Reference omega values for omega-dependence check.
        seed: Random seed (default 42 for reproducibility).

    Returns:
        ComparisonResult with all findings.
        Alpha ordering follows: (alpha_xx, alpha_xz, alpha_zx, alpha_zz).

    """
    # Case B: 2 particles, N_max = 2, restrict to N=2 subspace
    result_B = optimize_qfi_case_B(
        T_hold=T_hold,
        N_max=2,
        n_samples=n_samples_B,
        pure_only=True,
        subspace_N=2,
        seed=seed,
    )

    # Case A: 1 system particle + ancilla, N_max = 1, restrict to N=1 subspace
    result_A = optimize_qfi_case_A(
        T_hold=T_hold,
        N_max=1,
        n_samples=n_samples_A,
        n_alpha_samples=n_alpha_samples,
        pure_only=True,
        subspace_N=1,
        particle_penalty=particle_penalty,
        seed=seed if seed is None else seed + 1,
    )

    # Case A baseline: alpha = 0 (no interaction), N=1 subspace
    dim_sys = (1 + 1) ** 2  # N_max = 1 -> dim 4
    rng = np.random.default_rng(seed if seed is None else seed + 2)
    G_A_zero = compute_generator_A(T_hold, (0.0, 0.0, 0.0, 0.0), 1, n_quadrature)
    sub_idx = _subspace_indices(1, 1)

    best_fq_A_zero = -1.0
    for _ in range(n_samples_A // 2):
        psi_sys = random_pure_state_in_subspace(dim_sys, sub_idx, rng)
        psi_anc = random_pure_state_dm(2, rng)
        rho = np.kron(psi_sys, psi_anc)
        F_Q = quantum_fisher_information_dm(rho, G_A_zero)
        best_fq_A_zero = max(best_fq_A_zero, F_Q)

    # omega-dependence check
    fq_A_omega: dict[float, float] = {}
    if result_A.best_alphas is not None:
        best_rho_A = result_A.best_rho
        for omega_val in omega_values:
            if omega_val == 0.0:
                G_A_omega = compute_generator_A(
                    T_hold,
                    result_A.best_alphas,
                    1,
                    n_quadrature,
                )
            else:
                G_A_omega = compute_generator_A_at_omega(
                    T_hold,
                    omega_val,
                    result_A.best_alphas,
                    1,
                    n_quadrature,
                )
            F_Q = quantum_fisher_information_dm(best_rho_A, G_A_omega)
            fq_A_omega[omega_val] = float(F_Q)

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
        fq_A_omega=fq_A_omega,
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


def analytical_fq_B_max(T_hold: float) -> float:
    """Theoretical maximum QFI for Case B (2-particle system).

    For J = 1 (2 particles), J_y has eigenvalues {-1, 0, +1}.
    With optimal pure state: max F_Q = T_hold^2 (lambda_max - lambda_min)^2
    = 4 * T_hold^2.

    Args:
        T_hold: Holding-time strength.

    Returns:
        Theoretical maximum QFI.

    """
    return 4.0 * T_hold**2


def analytical_fq_A_zero(T_hold: float) -> float:
    """Theoretical QFI for Case A with alpha = 0 (uncoupled ancilla).

    With alpha = 0, G_A = -T_hold * J_y x I. For the 1-particle system (J = 1/2),
    J_y has eigenvalues {-1/2, +1/2}, so:
    max F_Q = T_hold^2 (1/2 - (-1/2))^2 = T_hold^2.

    Args:
        T_hold: Holding-time strength.

    Returns:
        Theoretical maximum QFI with zero interaction.

    """
    return 1.0 * T_hold**2

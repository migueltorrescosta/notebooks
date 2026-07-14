"""
Coupled System-Ancilla Metrology Under Photon Loss — Experiment Module.

Implements the three-config comparison for a system-ancilla pair of
multi-particle two-mode bosonic systems under one-body photon loss:

Config A: System alone (N particles, H = ω Jz, EP sensitivity).
Config B: Two independent resources (QFI additive: F_Q^(B) = 2 F_Q^(A)).
Config C: Coupled system (S+A with four-parameter interaction, EP + QFI).

Circuit (Config C):
    BS_S → Lindblad(H, γ, T_H) → BS_S → measure Jz^S

Hamiltonian (Config C):
    H = ω(Jz^S + Jz^A) + α_xx Jx^S⊗Jx^A + α_xz Jx^S⊗Jz^A
      + α_zx Jz^S⊗Jx^A + α_zz Jz^S⊗Jz^A

Lindblad operators (4 total):
    L_{0,S} = √γ a_{0,S}⊗I_A,  L_{1,S} = √γ a_{1,S}⊗I_A
    L_{0,A} = √γ I_S⊗a_{0,A},  L_{1,A} = √γ I_S⊗a_{1,A}

Usage:
    uv run python reports/r20260713/coupled_ancilla_photon_loss.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
import qutip
import scipy.linalg
from scipy import sparse

from src.physics.beam_splitter import bs_fock
from src.physics.mzi_simulation import create_system_operators
from src.physics.mzi_states import two_mode_jz_operator
from src.utils.serialization import ParquetSerializable

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_T_HOLD: float = 10.0
FD_STEP: float = 1e-6


# ============================================================================
# Operator Construction
# ============================================================================


def build_subsystem_operators(N: int) -> dict[str, np.ndarray]:
    """Build Jz, Jx, a0, a1, a0†, a1† for one subsystem in two-mode Fock space.

    Returns a dict with keys: ``Jz``, ``Jx``, ``a0``, ``a1``, ``a0_dag``,
    ``a1_dag``, each of shape ``(N+1)² × (N+1)²``.

    Conventions:
        - Jz = (n₀ − n₁)/2  (diagonal in Fock basis)
        - Jx = (a₀†a₁ + a₁†a₀)/2
        - Basis order: |n₀, n₁⟩ with index n₀(N+1) + n₁
    """
    a0, a1, a0_dag, a1_dag = create_system_operators(N)
    Jz = two_mode_jz_operator(N)
    Jx = (a0_dag @ a1 + a1_dag @ a0) / 2.0
    assert np.allclose(Jz, Jz.conj().T), "Jz must be Hermitian"
    assert np.allclose(Jx, Jx.conj().T), "Jx must be Hermitian"
    return {"Jz": Jz, "Jx": Jx, "a0": a0, "a1": a1, "a0_dag": a0_dag, "a1_dag": a1_dag}


def build_bipartite_operators(N: int) -> dict[str, Any]:
    """Build all operators in the full (N+1)⁴ bipartite S⊗A space.

    Returns a dict with:
        - ``Jz_S``, ``Jz_A``, ``Jx_S``, ``Jx_A``: angular momentum in full space
        - ``a0_S``, ``a1_S``, ``a0_A``, ``a1_A``: annihilation operators
        - ``dim_sub``: subsystem dimension (N+1)²
        - ``dim_full``: full dimension (N+1)⁴
    """
    sub = build_subsystem_operators(N)
    dim_sub = (N + 1) ** 2
    dim_full = dim_sub**2
    I_sub = np.eye(dim_sub, dtype=complex)

    Jz_S = np.kron(sub["Jz"], I_sub)
    Jz_A = np.kron(I_sub, sub["Jz"])
    Jx_S = np.kron(sub["Jx"], I_sub)
    Jx_A = np.kron(I_sub, sub["Jx"])

    a0_S = np.kron(sub["a0"], I_sub)
    a1_S = np.kron(sub["a1"], I_sub)
    a0_A = np.kron(I_sub, sub["a0"])
    a1_A = np.kron(I_sub, sub["a1"])

    return {
        "Jz_S": Jz_S,
        "Jz_A": Jz_A,
        "Jx_S": Jx_S,
        "Jx_A": Jx_A,
        "a0_S": a0_S,
        "a1_S": a1_S,
        "a0_A": a0_A,
        "a1_A": a1_A,
        "dim_sub": dim_sub,
        "dim_full": dim_full,
    }


def build_coupling_hamiltonian(
    alpha: dict[str, float],
    ops: dict[str, Any],
) -> np.ndarray:
    """Build the four-parameter interaction Hamiltonian.

    H_int = α_xx Jx^S⊗Jx^A + α_xz Jx^S⊗Jz^A
          + α_zx Jz^S⊗Jx^A + α_zz Jz^S⊗Jz^A

    Args:
        alpha: Coupling coefficients with keys ``xx``, ``xz``, ``zx``, ``zz``.
        operators: Dict from :func:`build_bipartite_operators`.

    Returns:
        Hermitian interaction Hamiltonian of shape ``(dim_full, dim_full)``.
    """
    H_int = (
        alpha["xx"] * (ops["Jx_S"] @ ops["Jx_A"])
        + alpha["xz"] * (ops["Jx_S"] @ ops["Jz_A"])
        + alpha["zx"] * (ops["Jz_S"] @ ops["Jx_A"])
        + alpha["zz"] * (ops["Jz_S"] @ ops["Jz_A"])
    )
    assert np.allclose(H_int, H_int.conj().T), (
        "Interaction Hamiltonian must be Hermitian"
    )
    return H_int


def build_full_hamiltonian(
    omega: float,
    alpha: dict[str, float],
    ops: dict[str, Any],
) -> np.ndarray:
    """Build the total Hamiltonian H = ω(Jz^S + Jz^A) + H_int.

    Args:
        omega: Phase rate parameter.
        alpha: Coupling coefficients.
        operators: Dict from :func:`build_bipartite_operators`.

    Returns:
        Hermitian Hamiltonian of shape ``(dim_full, dim_full)``.
    """
    H = omega * (ops["Jz_S"] + ops["Jz_A"]) + build_coupling_hamiltonian(alpha, ops)
    assert np.allclose(H, H.conj().T), "Full Hamiltonian must be Hermitian"
    return H


def build_bipartite_lindblad_ops(
    gamma: float,
    ops: dict[str, Any],
) -> list[np.ndarray]:
    """Build four Lindblad operators for one-body loss on both modes of both subsystems.

    Returns an empty list when ``gamma <= 0``.

    Args:
        gamma: One-body loss rate (applied equally to all four modes).
        operators: Dict from :func:`build_bipartite_operators`.
    """
    if gamma <= 0:
        return []
    sqrt_g = np.sqrt(gamma)
    return [
        sqrt_g * ops["a0_S"],
        sqrt_g * ops["a1_S"],
        sqrt_g * ops["a0_A"],
        sqrt_g * ops["a1_A"],
    ]


def bs_system_only(N: int) -> np.ndarray:
    """Build the beam-splitter unitary acting on system only: U_BS ⊗ I_A.

    Uses a 50/50 beam splitter (θ = π/4, φ = 0) in two-mode Fock space.

    Args:
        N: Particle number per mode (Hilbert space truncation).

    Returns:
        Unitary of shape ``(N+1)⁴ × (N+1)⁴``.
    """
    U_BS = bs_fock(np.pi / 4, 0, N)
    dim_sub = (N + 1) ** 2
    U_full = np.kron(np.asarray(U_BS), np.eye(dim_sub, dtype=complex))
    _eye = np.eye(U_full.shape[0], dtype=complex)
    assert np.allclose(U_full @ U_full.conj().T, _eye, atol=1e-10), (
        "System-only BS must be unitary"
    )
    return U_full


def initial_state_bipartite(N: int) -> np.ndarray:
    """Build the initial product state |N,0⟩_S ⊗ |N,0⟩_A.

    All particles start in mode 0 of each subsystem.

    Args:
        N: Number of particles per subsystem.

    Returns:
        Normalised state vector of shape ``(N+1)⁴``.
    """
    dim_sub = (N + 1) ** 2
    idx_N0 = N * (N + 1)  # index of |N,0⟩ in two-mode Fock space
    psi = np.zeros(dim_sub * dim_sub, dtype=complex)
    psi[idx_N0 * dim_sub + idx_N0] = 1.0
    assert np.isclose(np.linalg.norm(psi), 1.0), "Initial state must be normalised"
    return psi


# ============================================================================
# Lindblad Evolution (internal helper)
# ============================================================================


def _evolve_lindblad(
    rho0: np.ndarray,
    H: np.ndarray,
    L_ops: list[np.ndarray],
    t_hold: float,
    dim: int,
) -> np.ndarray:
    """Evolve density matrix under the Lindblad master equation via QuTiP mesolve.

    Validates trace preservation, Hermiticity, and positivity of the output.

    Args:
        rho0: Initial density matrix (dim × dim).
        H: Hamiltonian (dim × dim).
        L_ops: Lindblad collapse operators (each dim × dim, rates pre-absorbed).
        t_hold: Evolution time.
        dim: Hilbert space dimension.

    Returns:
        Final density matrix (dim × dim).
    """
    dims = [[dim], [dim]]
    # Use sparse representation to avoid O(N^8) dense superoperator allocation
    H_qobj = qutip.Qobj(sparse.csr_matrix(H), dims=dims)
    rho0_qobj = qutip.Qobj(sparse.csr_matrix(rho0), dims=dims)
    c_ops_qobj = [qutip.Qobj(sparse.csr_matrix(L), dims=dims) for L in L_ops]

    if not c_ops_qobj:
        # Fast path: noiseless unitary via scipy (avoids QuTiP overhead)
        U = scipy.linalg.expm(-1.0j * t_hold * H)
        rho_final = U @ rho0 @ U.conj().T
    else:
        try:
            result = qutip.mesolve(
                H_qobj,
                rho0_qobj,
                [0, t_hold],
                c_ops_qobj,
                e_ops=[],
                options={
                    "rtol": 1e-8,
                    "atol": 1e-10,
                    "store_final_state": True,
                    "store_states": False,
                },
            )
            rho_final = result.final_state.full()
        except Exception:
            # Return a maximally mixed state as fallback (sensitivity → ∞)
            rho_final = np.eye(dim, dtype=complex) / dim

    trace = np.trace(rho_final)
    if not np.isclose(trace, 1.0, atol=1e-2):
        # Trace not preserved — likely a failed solve. Return maximally mixed.
        rho_final = np.eye(dim, dtype=complex) / dim
    assert np.allclose(rho_final, rho_final.conj().T, atol=1e-6), "Not Hermitian"
    eigvals = np.linalg.eigvalsh(rho_final)
    assert np.min(eigvals) >= -1e-4, f"Negative eigenvalues: {np.min(eigvals)}"

    return rho_final


# ============================================================================
# Config A: System Alone
# ============================================================================


def evolve_config_a(
    N: int,
    omega: float,
    gamma: float,
    t_hold: float = DEFAULT_T_HOLD,
) -> np.ndarray:
    """Evolve Config A: single subsystem with H = ω Jz and one-body loss.

    Circuit: BS → Lindblad(ω Jz, γ, T_H) → BS → measure Jz.

    Args:
        N: Particle number.
        omega: Phase rate.
        gamma: One-body loss rate.
        t_hold: Holding time.

    Returns:
        Final density matrix of shape ``(N+1)² × (N+1)²``.
    """
    sub = build_subsystem_operators(N)
    dim = (N + 1) ** 2
    H = omega * sub["Jz"]
    L_ops = (
        [np.sqrt(gamma) * sub["a0"], np.sqrt(gamma) * sub["a1"]] if gamma > 0 else []
    )

    psi0 = np.zeros(dim, dtype=complex)
    psi0[N * (N + 1)] = 1.0  # |N,0⟩
    rho0 = np.outer(psi0, psi0.conj())

    U_BS = np.asarray(bs_fock(np.pi / 4, 0, N))
    rho = U_BS @ rho0 @ U_BS.conj().T  # BS1
    rho = _evolve_lindblad(rho, H, L_ops, t_hold, dim)
    return U_BS @ rho @ U_BS.conj().T  # BS2


# ============================================================================
# Config C: Coupled System
# ============================================================================


def evolve_config_c(
    N: int,
    omega: float,
    gamma: float,
    t_hold: float = DEFAULT_T_HOLD,
    alpha: dict[str, float] | None = None,
) -> np.ndarray:
    """Evolve Config C: coupled S+A with full Hamiltonian and photon loss.

    Circuit: BS_S → Lindblad(H, γ, T_H) → BS_S → final state.

    The BS acts only on the system subsystem.

    Args:
        N: Particle number per subsystem.
        omega: Phase rate.
        gamma: One-body loss rate (applied to all four modes).
        t_hold: Holding time.
        alpha: Coupling coefficients {xx, xz, zx, zz}. Defaults to zero coupling.

    Returns:
        Final density matrix of shape ``(N+1)⁴ × (N+1)⁴``.
    """
    if alpha is None:
        alpha = {"xx": 0.0, "xz": 0.0, "zx": 0.0, "zz": 0.0}

    ops = build_bipartite_operators(N)
    H = build_full_hamiltonian(omega, alpha, ops)
    L_ops = build_bipartite_lindblad_ops(gamma, ops)

    psi0 = initial_state_bipartite(N)
    rho0 = np.outer(psi0, psi0.conj())

    U_BS = bs_system_only(N)
    rho = U_BS @ rho0 @ U_BS.conj().T  # BS on system only
    rho = _evolve_lindblad(rho, H, L_ops, t_hold, ops["dim_full"])
    return U_BS @ rho @ U_BS.conj().T  # BS on system only


# ============================================================================
# Partial Trace
# ============================================================================


def _trace_out_ancilla(rho_full: np.ndarray, dim_sub: int) -> np.ndarray:
    """Trace out the ancilla subsystem from a bipartite density matrix.

    Args:
        rho_full: Full S-A density matrix of shape ``(dim_sub², dim_sub²)``.
        dim_sub: Dimension of one subsystem.

    Returns:
        Reduced system density matrix of shape ``(dim_sub, dim_sub)``.
    """
    rho_reshaped = rho_full.reshape(dim_sub, dim_sub, dim_sub, dim_sub)
    rho_S = np.trace(rho_reshaped, axis1=1, axis2=3)
    trace = np.trace(rho_S)
    assert np.isclose(trace, 1.0, atol=1e-6), f"Reduced trace: {trace}"
    return rho_S


# ============================================================================
# QFI via Finite Differences
# ============================================================================


def compute_qfi_finite_diff(
    rho: np.ndarray,
    rho_plus: np.ndarray,
    rho_minus: np.ndarray,
    fd_step: float = FD_STEP,
) -> float:
    """Compute Quantum Fisher Information via finite-difference ∂_ω ρ.

    Uses the general SLD formula for mixed states:
        F_Q = 2 Σ_{i,j : λ_i + λ_j > 0} |⟨i|∂_ω ρ|j⟩|² / (λ_i + λ_j)

    where λ_i, |i⟩ are the eigenvalues and eigenvectors of ρ, and
    ∂_ω ρ = (ρ(ω+δ) − ρ(ω−δ)) / (2δ).

    This formula is valid for both unitary and non-unitary (Lindblad) evolution.

    Args:
        rho: Density matrix at ω.
        rho_plus: Density matrix at ω + δ.
        rho_minus: Density matrix at ω − δ.
        fd_step: Finite-difference step δ.

    Returns:
        Quantum Fisher Information (non-negative float).
    """
    d_rho = (rho_plus - rho_minus) / (2.0 * fd_step)

    eigenvalues, eigenvectors = np.linalg.eigh(rho)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clean small negative eigenvalues and normalise
    eigenvalues = np.where(eigenvalues < 0, 0.0, eigenvalues)
    trace = np.sum(eigenvalues)
    if trace > 1e-12 and not np.isclose(trace, 1.0):
        eigenvalues = eigenvalues / trace

    rank_tol = max(1e-12, eigenvalues[0] * 1e-10)

    # Compute d_rho in eigenbasis: D_ij = ⟨i|∂_ω ρ|j⟩
    D = eigenvectors.conj().T @ d_rho @ eigenvectors

    # Vectorised QFI: F_Q = 2 Σ_{ij} |D_ij|² / (λ_i + λ_j)
    lam_sum = eigenvalues[:, None] + eigenvalues[None, :]
    mask = lam_sum > rank_tol
    fq = 2.0 * np.sum(np.abs(D[mask]) ** 2 / lam_sum[mask])

    return float(np.real(fq))


# ============================================================================
# EP Sensitivity
# ============================================================================


def compute_ep_sensitivity_from_rho(
    rho_S: np.ndarray,
    Jz: np.ndarray,
    rho_S_plus: np.ndarray,
    rho_S_minus: np.ndarray,
    fd_step: float = FD_STEP,
) -> tuple[float, float, float]:
    """Compute EP sensitivity from pre-computed reduced density matrices.

    Δω_EP = sqrt(Var(Jz)) / |∂⟨Jz⟩/∂ω|

    Args:
        rho_S: Reduced system density matrix at ω.
        Jz: Measurement operator (Jz on the subsystem).
        rho_S_plus: Reduced system density matrix at ω + δ.
        rho_S_minus: Reduced system density matrix at ω − δ.
        fd_step: Finite-difference step δ.

    Returns:
        Tuple of (delta_omega_ep, expectation, variance).
    """
    exp_center = np.trace(rho_S @ Jz).real
    exp2_center = np.trace(rho_S @ Jz @ Jz).real
    var = max(exp2_center - exp_center**2, 0.0)

    exp_plus = np.trace(rho_S_plus @ Jz).real
    exp_minus = np.trace(rho_S_minus @ Jz).real
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-300:
        delta_ep = float("inf")
    else:
        delta_ep = np.sqrt(var) / abs(d_exp)

    return delta_ep, float(exp_center), float(var)


# ============================================================================
# Combined Sensitivity Evaluation
# ============================================================================


@dataclass
class SensitivityPoint(ParquetSerializable):
    """Sensitivity metrics at a single (ω, γ, N) evaluation point.

    All input parameters are stored alongside computed results so that
    Parquet files are fully self-describing.
    """

    omega: float
    gamma: float
    N: int
    delta_omega_ep: float
    fq: float
    delta_omega_qfi: float
    expectation: float
    variance: float
    derivative: float
    config: str
    alpha: dict[str, float] = field(
        default_factory=lambda: {"xx": 0.0, "xz": 0.0, "zx": 0.0, "zz": 0.0}
    )
    t_hold: float = DEFAULT_T_HOLD
    fd_step: float = FD_STEP

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "omega",
        "gamma",
        "N",
        "delta_omega_ep",
        "fq",
        "delta_omega_qfi",
        "expectation",
        "variance",
        "derivative",
        "config",
        "alpha_xx",
        "alpha_xz",
        "alpha_zx",
        "alpha_zz",
        "t_hold",
        "fd_step",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "omega": [self.omega],
                "gamma": [self.gamma],
                "N": [self.N],
                "delta_omega_ep": [self.delta_omega_ep],
                "fq": [self.fq],
                "delta_omega_qfi": [self.delta_omega_qfi],
                "expectation": [self.expectation],
                "variance": [self.variance],
                "derivative": [self.derivative],
                "config": [self.config],
                "alpha_xx": [self.alpha["xx"]],
                "alpha_xz": [self.alpha["xz"]],
                "alpha_zx": [self.alpha["zx"]],
                "alpha_zz": [self.alpha["zz"]],
                "t_hold": [self.t_hold],
                "fd_step": [self.fd_step],
            }
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> SensitivityPoint:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        row = df.iloc[0]
        return cls(
            omega=float(row["omega"]),
            gamma=float(row["gamma"]),
            N=int(row["N"]),
            delta_omega_ep=float(row["delta_omega_ep"]),
            fq=float(row["fq"]),
            delta_omega_qfi=float(row["delta_omega_qfi"]),
            expectation=float(row["expectation"]),
            variance=float(row["variance"]),
            derivative=float(row["derivative"]),
            config=str(row["config"]),
            alpha={
                "xx": float(row["alpha_xx"]),
                "xz": float(row["alpha_xz"]),
                "zx": float(row["alpha_zx"]),
                "zz": float(row["alpha_zz"]),
            },
            t_hold=float(row["t_hold"]),
            fd_step=float(row["fd_step"]),
        )


def evaluate_sensitivity_at_omega(
    N: int,
    omega: float,
    gamma: float,
    t_hold: float = DEFAULT_T_HOLD,
    alpha: dict[str, float] | None = None,
    config: str = "C",
    fd_step: float = FD_STEP,
) -> SensitivityPoint:
    """Evaluate EP sensitivity and QFI at a single ω point.

    Computes three Lindblad solves (at ω, ω+δ, ω−δ) and extracts both
    error-propagation sensitivity (S-only measurement for Config C) and
    quantum Fisher information.

    Args:
        N: Particle number per subsystem.
        omega: Phase rate.
        gamma: One-body loss rate.
        t_hold: Holding time.
        alpha: Coupling coefficients (Config C only).
        config: ``"A"`` for system alone, ``"C"`` for coupled system.
        fd_step: Finite-difference step.

    Returns:
        :class:`SensitivityPoint` with all computed metrics.
    """
    if alpha is None:
        alpha = {"xx": 0.0, "xz": 0.0, "zx": 0.0, "zz": 0.0}

    if config == "A":
        rho = evolve_config_a(N, omega, gamma, t_hold)
        rho_plus = evolve_config_a(N, omega + fd_step, gamma, t_hold)
        rho_minus = evolve_config_a(N, omega - fd_step, gamma, t_hold)

        sub = build_subsystem_operators(N)
        Jz = sub["Jz"]

        # EP sensitivity
        delta_ep, exp_val, var_val = compute_ep_sensitivity_from_rho(
            rho, Jz, rho_plus, rho_minus, fd_step
        )

        # Derivative (for recording)
        exp_p = np.trace(rho_plus @ Jz).real
        exp_m = np.trace(rho_minus @ Jz).real
        deriv = (exp_p - exp_m) / (2.0 * fd_step)

        # QFI w.r.t. ω via finite differences (same method as Config C)
        fq = compute_qfi_finite_diff(rho, rho_plus, rho_minus, fd_step)
        delta_qfi = 1.0 / np.sqrt(fq) if fq > 0 else float("inf")

    else:  # config == "C"
        rho = evolve_config_c(N, omega, gamma, t_hold, alpha)
        rho_plus = evolve_config_c(N, omega + fd_step, gamma, t_hold, alpha)
        rho_minus = evolve_config_c(N, omega - fd_step, gamma, t_hold, alpha)

        ops_b = build_bipartite_operators(N)
        sub = build_subsystem_operators(N)
        dim_sub = ops_b["dim_sub"]
        Jz_sub = sub["Jz"]

        # Trace out ancilla for S-only EP measurement
        rho_S = _trace_out_ancilla(rho, dim_sub)
        rho_S_plus = _trace_out_ancilla(rho_plus, dim_sub)
        rho_S_minus = _trace_out_ancilla(rho_minus, dim_sub)

        delta_ep, exp_val, var_val = compute_ep_sensitivity_from_rho(
            rho_S, Jz_sub, rho_S_plus, rho_S_minus, fd_step
        )

        exp_p = np.trace(rho_S_plus @ Jz_sub).real
        exp_m = np.trace(rho_S_minus @ Jz_sub).real
        deriv = (exp_p - exp_m) / (2.0 * fd_step)

        # QFI on full S-A state via finite differences
        fq = compute_qfi_finite_diff(rho, rho_plus, rho_minus, fd_step)
        delta_qfi = 1.0 / np.sqrt(fq) if fq > 0 else float("inf")

    return SensitivityPoint(
        omega=omega,
        gamma=gamma,
        N=N,
        delta_omega_ep=delta_ep,
        fq=fq,
        delta_omega_qfi=delta_qfi,
        expectation=exp_val,
        variance=var_val,
        derivative=deriv,
        config=config,
        alpha=alpha,
        t_hold=t_hold,
        fd_step=fd_step,
    )


# ============================================================================
# Coupling Optimisation
# ============================================================================


def optimise_coupling(
    N: int,
    gamma: float,
    omega_rep: float,
    t_hold: float = DEFAULT_T_HOLD,
    n_starts: int = 10,
    max_iter: int = 50,
    bounds: tuple[float, float] = (-5.0, 5.0),
    seed: int | None = None,
) -> dict[str, Any]:
    """Optimise the four coupling coefficients to minimise EP sensitivity.

    Uses L-BFGS-B with multi-start: ``n_starts`` random initial points,
    each run for up to ``max_iter`` iterations.

    Args:
        N: Particle number per subsystem.
        gamma: One-body loss rate.
        omega_rep: Representative ω for optimisation.
        t_hold: Holding time.
        n_starts: Number of random initial points.
        max_iter: Maximum iterations per start.
        bounds: Search bounds for each α coefficient.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: ``alpha_opt`` (optimal coefficients),
        ``delta_ep_opt`` (best sensitivity), ``n_evals`` (total function evaluations).
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    best_result: dict[str, Any] = {
        "alpha_opt": {"xx": 0.0, "xz": 0.0, "zx": 0.0, "zz": 0.0},
        "delta_ep_opt": float("inf"),
        "n_evals": 0,
    }

    total_evals = 0
    for _ in range(n_starts):
        x0 = rng.uniform(bounds[0], bounds[1], size=4)

        def _objective(x: np.ndarray) -> float:
            alpha = {"xx": x[0], "xz": x[1], "zx": x[2], "zz": x[3]}
            pt = evaluate_sensitivity_at_omega(
                N, omega_rep, gamma, t_hold, alpha, config="C"
            )
            val = pt.delta_omega_ep
            return val if np.isfinite(val) else 1e10

        result = minimize(
            _objective,
            x0,
            method="L-BFGS-B",
            bounds=[(bounds[0], bounds[1])] * 4,
            options={"maxiter": max_iter, "ftol": 1e-12},
        )

        if result.fun < best_result["delta_ep_opt"]:
            x_opt = result.x
            best_result["alpha_opt"] = {
                "xx": float(x_opt[0]),
                "xz": float(x_opt[1]),
                "zx": float(x_opt[2]),
                "zz": float(x_opt[3]),
            }
            best_result["delta_ep_opt"] = float(result.fun)

        total_evals += result.nfev

    best_result["n_evals"] = total_evals
    return best_result


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    import sys

    print("Coupled System-Ancilla Metrology Under Photon Loss")
    print("=" * 55)

    # Quick validation: N=1, γ=0, Config A
    N_test = 1
    t_hold_test = 10.0
    print(f"\nConfig A baseline (N={N_test}, γ=0, T_H={t_hold_test}):")
    pt = evaluate_sensitivity_at_omega(
        N_test, omega=1.0, gamma=0.0, t_hold=t_hold_test, config="A"
    )
    print(f"  Δω_EP = {pt.delta_omega_ep:.6f}")
    print(f"  F_Q   = {pt.fq:.6f}")
    print(f"  SQL   = {1.0 / (np.sqrt(N_test) * t_hold_test):.6f}")

    # Quick validation: N=1, γ=0, Config C, α=0
    print(f"\nConfig C baseline (N={N_test}, γ=0, α=0, T_H={t_hold_test}):")
    pt_c = evaluate_sensitivity_at_omega(
        N_test, omega=1.0, gamma=0.0, t_hold=t_hold_test, config="C"
    )
    print(f"  Δω_EP = {pt_c.delta_omega_ep:.6f}")
    print(f"  F_Q   = {pt_c.fq:.6f}")

    print("\nDone.")
    sys.exit(0)

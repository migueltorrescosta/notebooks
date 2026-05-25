"""
Local module for the 2026-05-24 Phase-Diffusion Robustness of the Drive Protocol report.

Contains all code exclusive to this report:
- Liouvillian construction for Lindblad master equation with phase diffusion
- Noisy circuit evolution (BS → Lindblad hold → BS)
- Sensitivity computation for mixed states
- 4D random search + Nelder–Mead optimisation with phase noise
- Noise scan over (θ, γ_φ) pairs
- Exclusive plot functions
- Data and figure generation pipeline
- CLI entry point for standalone execution

Usage:
    uv run python reports/2026-05-24/local.py --force

This module is **not** importable as ``reports.2026-05-24.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as _mp
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from scipy.linalg import expm
from scipy.optimize import minimize

# Ensure project root is on sys.path for shared-module imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force non-interactive matplotlib backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

from src.analysis.ancilla_drive_metrology import (  # noqa: E402
    system_only_bs_unitary,
)
from src.analysis.ancilla_optimization import (  # noqa: E402
    build_two_qubit_operators,
)

sns.set_theme(style="whitegrid")

# ============================================================================
# Physical constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_T_H: float = 10.0  # Holding time (SQL = 0.1)
SQL_REFERENCE: float = 1.0 / DEFAULT_T_H  # Δθ_SQL = 0.1
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Range for all coefficients
FD_STEP: float = 1e-6  # Finite-difference step for theta derivative

I_4 = np.eye(4, dtype=complex)

# Initial state: |00⟩ (pure, density matrix form)
DEFAULT_RHO0: np.ndarray = np.zeros((4, 4), dtype=complex)
DEFAULT_RHO0[0, 0] = 1.0

# Theta and gamma_phi values for the scan
THETA_VALS: list[float] = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
GAMMA_PHI_VALS: list[float] = list(np.logspace(-4, 1, 15).tolist())

# ============================================================================
# Hamiltonian Construction (same as θ-modulated drive protocol)
# ============================================================================


def build_noise_drive_hamiltonian(
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the ancilla drive Hamiltonian with θ modulation.

    H_A = θ (a_x J_x^A + a_y J_y^A + a_z J_z^A)

    Args:
        theta: Unknown phase rate parameter.
        a_x: Coefficient for J_x^A.
        a_y: Coefficient for J_y^A.
        a_z: Coefficient for J_z^A.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    H = theta * H
    return 0.5 * (H + H.conj().T)


def build_noise_iszz_interaction(
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the Ising-type system--ancilla interaction.

    H_int = a_zz J_z^S ⊗ J_z^A = a_zz (σ_z/2) ⊗ (σ_z/2)

    Args:
        a_zz: Interaction coupling coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_zz != 0.0:
        H += a_zz * (ops["Jz_S"] @ ops["Jz_A"])
    return H


def build_noise_hold_hamiltonian(
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian with θ-modulated ancilla drive.

    H = θ J_z^S + θ (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A

    Args:
        theta: Unknown phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = theta * ops["Jz_S"]
    H += build_noise_drive_hamiltonian(theta, a_x, a_y, a_z, ops)
    H += build_noise_iszz_interaction(a_zz, ops)
    return 0.5 * (H + H.conj().T)


# ============================================================================
# Liouvillian Construction (Lindblad Master Equation)
# ============================================================================


def build_phase_diffusion_operators(
    gamma_phi: float,
    ops: dict[str, np.ndarray],
) -> list[np.ndarray]:
    """Build the phase diffusion Lindblad operators.

    L_S = √γ_φ J_z^S,  L_A = √γ_φ J_z^A

    Args:
        gamma_phi: Phase diffusion rate.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        List of Lindblad operators [L_S, L_A] (each 4×4).
    """
    if gamma_phi <= 0.0:
        return []
    sqrt_g = np.sqrt(gamma_phi)
    return [
        sqrt_g * ops["Jz_S"],
        sqrt_g * ops["Jz_A"],
    ]


def build_liouvillian(
    H: np.ndarray,
    lindblad_ops: list[np.ndarray],
) -> np.ndarray:
    """Construct the Liouvillian superoperator in vectorised form.

    Using column-major vectorisation (``vec(ρ)`` stacks columns):
        ℒ = -i(I ⊗ H - H^T ⊗ I)
            + Σ_k [L_k^* ⊗ L_k - 1/2(I ⊗ L_k^† L_k + (L_k^† L_k)^T ⊗ I)]

    The vectorised density matrix evolves as:
        vec(ρ(t)) = exp(ℒ t) vec(ρ(0))

    Args:
        H: Hamiltonian matrix (n × n Hermitian).
        lindblad_ops: List of Lindblad jump operators (each n × n).

    Returns:
        Liouvillian matrix (n² × n²).
    """
    d = H.shape[0]
    I_d = np.eye(d, dtype=complex)

    # Unitary part: vec(-i[H, ρ]) = -i(I ⊗ H - H^T ⊗ I) vec(ρ)
    L = -1j * (np.kron(I_d, H) - np.kron(H.T, I_d))

    # Dissipative part
    for Lk in lindblad_ops:
        Lk_dag = Lk.conj().T
        Lk_dag_Lk = Lk_dag @ Lk
        # vec(L_k ρ L_k^†) = (L_k^* ⊗ L_k) vec(ρ)
        L += np.kron(Lk.conj(), Lk)
        # -½ vec({L_k^† L_k, ρ}) = -½(I ⊗ L_k^† L_k + (L_k^† L_k)^T ⊗ I) vec(ρ)
        L -= 0.5 * (np.kron(I_d, Lk_dag_Lk) + np.kron(Lk_dag_Lk.T, I_d))

    return L


# ============================================================================
# Density Matrix Utilities
# ============================================================================


def vectorise_rho(rho: np.ndarray) -> np.ndarray:
    """Vectorise a density matrix (column-major stacking).

    Args:
        rho: n × n density matrix.

    Returns:
        n²-vector.
    """
    return rho.reshape(-1, order="F")


def unvectorise_rho(vec: np.ndarray) -> np.ndarray:
    """Unvectorise a density matrix.

    Args:
        vec: n²-vector.

    Returns:
        n × n density matrix.
    """
    d = int(np.sqrt(vec.shape[0]))
    return vec.reshape(d, d, order="F")


def density_expectation(rho: np.ndarray, op: np.ndarray) -> float:
    """Compute ⟨op⟩ = Tr(op ρ) for a density matrix.

    Args:
        rho: n × n density matrix.
        op: n × n operator (Hermitian).

    Returns:
        Real expectation value.
    """
    return float(np.real(np.trace(op @ rho)))


def density_variance(rho: np.ndarray, op: np.ndarray) -> float:
    """Compute Var(op) = ⟨op²⟩ - ⟨op⟩² for a density matrix.

    Args:
        rho: n × n density matrix.
        op: n × n operator (Hermitian).

    Returns:
        Variance (non-negative).
    """
    exp_val = density_expectation(rho, op)
    exp_sq = density_expectation(rho, op @ op)
    raw_var = exp_sq - exp_val**2
    assert raw_var >= -1e-12, f"Unphysical negative variance: {raw_var:.2e}"
    return float(max(0.0, raw_var))


def validate_density(rho: np.ndarray, atol: float = 1e-8) -> None:
    """Validate basic properties of a density matrix.

    Args:
        rho: Density matrix to validate.
        atol: Absolute tolerance for trace and Hermiticity.

    Raises:
        AssertionError: If any property fails.
    """
    d = rho.shape[0]
    assert rho.shape == (d, d), f"Density matrix must be square, got {rho.shape}"
    trace = float(np.real(np.trace(rho)))
    assert np.isclose(trace, 1.0, atol=atol), (
        f"Trace not preserved: Tr(ρ) = {trace:.2e}"
    )
    assert np.allclose(rho, rho.conj().T, atol=atol), "Density matrix not Hermitian"
    evals = np.linalg.eigvalsh(rho)
    assert np.all(evals >= -atol), (
        f"Density matrix has negative eigenvalues: min={float(np.min(evals)):.2e}"
    )


# ============================================================================
# Noisy Circuit Evolution
# ============================================================================


def evolve_noisy_drive_circuit(
    rho0: np.ndarray,
    T_BS: float,
    T_H: float,
    theta: float,
    gamma_phi: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full noisy MZI circuit with phase diffusion.

    ρ_final = U_BS_S · exp(ℒ T_H)(U_BS_S · ρ₀ · U_BS_S^†) · U_BS_S^†

    where ℒ is the Lindblad Liouvillian with phase diffusion.

    Args:
        rho0: Initial 4×4 density matrix.
        T_BS: Beam-splitter duration (both BS identical).
        T_H: Holding time.
        theta: Phase rate parameter.
        gamma_phi: Phase diffusion rate.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.

    Returns:
        Final 4×4 density matrix.
    """
    assert np.isclose(np.trace(rho0), 1.0, atol=1e-12), (
        "Initial density matrix must have trace 1"
    )

    # BS1
    U_bs = system_only_bs_unitary(T_BS)
    rho = U_bs @ rho0 @ U_bs.conj().T

    # Hold with Lindblad evolution (always use Liouvillian for consistency)
    H = build_noise_hold_hamiltonian(theta, a_x, a_y, a_z, a_zz, ops)
    lindblad_ops = build_phase_diffusion_operators(gamma_phi, ops)
    L = build_liouvillian(H, lindblad_ops)

    # Vectorise, exponentiate, unvectorise
    rho_vec = vectorise_rho(rho)
    rho_vec_evolved = expm(L * T_H) @ rho_vec
    rho = unvectorise_rho(rho_vec_evolved)

    # BS2
    rho = U_bs @ rho @ U_bs.conj().T

    # Validate the final density matrix
    validate_density(rho)

    return rho


# ============================================================================
# Sensitivity Computation
# ============================================================================


def compute_noisy_sensitivity(
    rho0: np.ndarray,
    T_BS: float,
    T_H: float,
    theta_true: float,
    gamma_phi: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
) -> float:
    """Compute the error-propagation sensitivity Δθ with phase diffusion.

    Δθ = sqrt(Var(J_z^S)) / |∂⟨J_z^S⟩/∂θ|

    The derivative is computed via central finite differences at θ±δ,
    re-evaluating the full noisy circuit at each point.

    Args:
        rho0: Initial 4×4 density matrix.
        T_BS: Beam-splitter duration.
        T_H: Holding-time strength.
        theta_true: True phase rate parameter.
        gamma_phi: Phase diffusion rate.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.
        fd_step: Finite-difference step size (default 1e-6).

    Returns:
        Sensitivity Δθ (positive float). Returns inf if derivative is zero.
    """
    meas_op = ops["Jz_S"]

    # Evaluate at theta_true
    rho = evolve_noisy_drive_circuit(
        rho0, T_BS, T_H, theta_true, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )
    var = density_variance(rho, meas_op)

    # Central finite difference for ∂⟨J_z^S⟩/∂θ
    rho_plus = evolve_noisy_drive_circuit(
        rho0, T_BS, T_H, theta_true + fd_step, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )
    rho_minus = evolve_noisy_drive_circuit(
        rho0, T_BS, T_H, theta_true - fd_step, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )

    exp_plus = density_expectation(rho_plus, meas_op)
    exp_minus = density_expectation(rho_minus, meas_op)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


def compute_noisy_sensitivity_with_diagnostics(
    rho0: np.ndarray,
    T_BS: float,
    T_H: float,
    theta_true: float,
    gamma_phi: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
) -> tuple[float, float, float, float]:
    """Compute Δθ with diagnostics.

    Returns:
        Tuple (delta_theta, expectation_Jz, variance_Jz, d_exp_d_theta).
    """
    meas_op = ops["Jz_S"]

    # Evaluate at theta_true
    rho = evolve_noisy_drive_circuit(
        rho0, T_BS, T_H, theta_true, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )
    exp_val = density_expectation(rho, meas_op)
    var = density_variance(rho, meas_op)

    # Central finite difference for ∂⟨J_z^S⟩/∂θ
    rho_plus = evolve_noisy_drive_circuit(
        rho0, T_BS, T_H, theta_true + fd_step, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )
    rho_minus = evolve_noisy_drive_circuit(
        rho0, T_BS, T_H, theta_true - fd_step, gamma_phi, a_x, a_y, a_z, a_zz, ops
    )

    exp_plus = density_expectation(rho_plus, meas_op)
    exp_minus = density_expectation(rho_minus, meas_op)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf"), exp_val, var, d_exp

    delta_theta = float(np.sqrt(var) / abs(d_exp))
    return delta_theta, exp_val, var, d_exp


# ============================================================================
# Dataclass: DriveNoiseScanResult
# ============================================================================


@dataclass
class DriveNoiseScanResult:
    """Result from a (θ, γ_φ) noise scan with re-optimised parameters.

    Attributes:
        theta_values: Array of θ values scanned.
        gamma_phi_values: Array of γ_φ values scanned.
        best_params_per_pair: List of optimal (a_x, a_y, a_z, a_zz) tuples
            indexed as [i_theta * n_gamma + i_gamma].
        delta_theta_per_pair: Δθ at the optimal parameters for each (θ, γ_φ),
            shaped (n_theta, n_gamma).
        expectation_Jz_per_pair: ⟨J_z^S⟩ at each optimal point,
            shaped (n_theta, n_gamma).
        variance_Jz_per_pair: Var(J_z^S) at each optimal point,
            shaped (n_theta, n_gamma).
        d_exp_d_theta_per_pair: ∂⟨J_z^S⟩/∂θ at each optimal point,
            shaped (n_theta, n_gamma).
        sql: SQL = 1/T_H reference value.
        T_H: Holding time used.
        n_random: Number of random search points per (θ, γ_φ) pair.
        n_nm_refine: Number of Nelder-Mead refinements per pair.
        maxiter: Maximum Nelder-Mead iterations.
        bounds_lo: Lower bound for all optimisation parameters.
        bounds_hi: Upper bound for all optimisation parameters.
        fd_step: Finite-difference step for derivative computation.
        seed: Base random seed.
    """

    theta_values: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma_phi_values: np.ndarray = field(default_factory=lambda: np.array([]))
    best_params_per_pair: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )
    delta_theta_per_pair: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_Jz_per_pair: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_Jz_per_pair: np.ndarray = field(default_factory=lambda: np.array([]))
    d_exp_d_theta_per_pair: np.ndarray = field(default_factory=lambda: np.array([]))
    sql: float = SQL_REFERENCE
    T_H: float = DEFAULT_T_H
    n_random: int = 1000
    n_nm_refine: int = 25
    maxiter: int = 5000
    bounds_lo: float = -5.0
    bounds_hi: float = 5.0
    fd_step: float = FD_STEP
    seed: int = 42

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten into a long-format DataFrame with one row per (θ, γ_φ)."""
        n_theta = len(self.theta_values)
        n_gamma = len(self.gamma_phi_values)
        rows: list[dict[str, float | str]] = []
        for i in range(n_theta):
            for j in range(n_gamma):
                idx = i * n_gamma + j
                params = (
                    self.best_params_per_pair[idx]
                    if idx < len(self.best_params_per_pair)
                    else (0.0, 0.0, 0.0, 0.0)
                )
                dt = (
                    float(self.delta_theta_per_pair[i, j])
                    if self.delta_theta_per_pair.size > 0
                    else float("inf")
                )
                exp_val = (
                    float(self.expectation_Jz_per_pair[i, j])
                    if self.expectation_Jz_per_pair.size > 0
                    else 0.0
                )
                var_val = (
                    float(self.variance_Jz_per_pair[i, j])
                    if self.variance_Jz_per_pair.size > 0
                    else 0.0
                )
                d_exp = (
                    float(self.d_exp_d_theta_per_pair[i, j])
                    if self.d_exp_d_theta_per_pair.size > 0
                    else 0.0
                )
                rows.append(
                    {
                        "theta": float(self.theta_values[i]),
                        "gamma_phi": float(self.gamma_phi_values[j]),
                        "a_x": float(params[0]),
                        "a_y": float(params[1]),
                        "a_z": float(params[2]),
                        "a_zz": float(params[3]),
                        "delta_theta": dt,
                        "expectation_Jz": exp_val,
                        "variance_Jz": var_val,
                        "d_exp_d_theta": d_exp,
                        "sql": float(self.sql),
                        "T_H": float(self.T_H),
                        "n_random": int(self.n_random),
                        "n_nm_refine": int(self.n_nm_refine),
                        "maxiter": int(self.maxiter),
                        "bounds_lo": float(self.bounds_lo),
                        "bounds_hi": float(self.bounds_hi),
                        "fd_step": float(self.fd_step),
                        "seed": int(self.seed),
                    }
                )
        return pd.DataFrame(rows)

    def save_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> DriveNoiseScanResult:
        df = pd.read_parquet(path)
        required = {
            "theta",
            "gamma_phi",
            "a_x",
            "a_y",
            "a_z",
            "a_zz",
            "delta_theta",
            "expectation_Jz",
            "variance_Jz",
            "d_exp_d_theta",
            "sql",
            "T_H",
            "n_random",
            "n_nm_refine",
            "maxiter",
            "bounds_lo",
            "bounds_hi",
            "fd_step",
            "seed",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: {sorted(missing)}. "
                "Regenerate the file with the current code."
            )
        theta_vals = df["theta"].unique()
        gamma_vals = df["gamma_phi"].unique()
        theta_vals.sort()
        gamma_vals.sort()
        n_theta = len(theta_vals)
        n_gamma = len(gamma_vals)

        # Build the lookup dict
        lookup: dict[tuple[float, float], dict[str, float]] = {}
        for _, row in df.iterrows():
            key = (float(row["theta"]), float(row["gamma_phi"]))
            lookup[key] = {
                "a_x": float(row["a_x"]),
                "a_y": float(row["a_y"]),
                "a_z": float(row["a_z"]),
                "a_zz": float(row["a_zz"]),
                "delta_theta": float(row["delta_theta"]),
                "expectation_Jz": float(row["expectation_Jz"]),
                "variance_Jz": float(row["variance_Jz"]),
                "d_exp_d_theta": float(row["d_exp_d_theta"]),
            }

        params_list: list[tuple[float, float, float, float]] = []
        dt_arr = np.full((n_theta, n_gamma), np.inf, dtype=float)
        exp_arr = np.zeros((n_theta, n_gamma), dtype=float)
        var_arr = np.zeros((n_theta, n_gamma), dtype=float)
        d_exp_arr = np.zeros((n_theta, n_gamma), dtype=float)

        for i, t in enumerate(theta_vals):
            for j, g in enumerate(gamma_vals):
                entry = lookup.get((t, g), {})
                params_list.append(
                    (
                        entry.get("a_x", 0.0),
                        entry.get("a_y", 0.0),
                        entry.get("a_z", 0.0),
                        entry.get("a_zz", 0.0),
                    )
                )
                dt_arr[i, j] = entry.get("delta_theta", float("inf"))
                exp_arr[i, j] = entry.get("expectation_Jz", 0.0)
                var_arr[i, j] = entry.get("variance_Jz", 0.0)
                d_exp_arr[i, j] = entry.get("d_exp_d_theta", 0.0)

        return cls(
            theta_values=np.array(theta_vals, dtype=float),
            gamma_phi_values=np.array(gamma_vals, dtype=float),
            best_params_per_pair=params_list,
            delta_theta_per_pair=dt_arr,
            expectation_Jz_per_pair=exp_arr,
            variance_Jz_per_pair=var_arr,
            d_exp_d_theta_per_pair=d_exp_arr,
            sql=float(df["sql"].iloc[0]),
            T_H=float(df["T_H"].iloc[0]),
            n_random=int(df["n_random"].iloc[0]),
            n_nm_refine=int(df["n_nm_refine"].iloc[0]),
            maxiter=int(df["maxiter"].iloc[0]),
            bounds_lo=float(df["bounds_lo"].iloc[0]),
            bounds_hi=float(df["bounds_hi"].iloc[0]),
            fd_step=float(df["fd_step"].iloc[0]),
            seed=int(df["seed"].iloc[0]),
        )


# ============================================================================
# Decoupled Baseline (noisy)
# ============================================================================


def compute_noisy_decoupled_baseline(
    gamma_phi: float = 0.0,
    T_H: float = DEFAULT_T_H,
    theta_true: float = 1.0,
) -> float:
    """Compute the decoupled baseline sensitivity with phase diffusion.

    At (a_x = a_y = a_z = a_zz = 0), the circuit reduces to a standard
    single-qubit MZI with |1,0⟩ input and 50/50 BS. Phase diffusion on
    both qubits does not affect the SQL baseline because there is no
    entanglement to dephase.

    Args:
        gamma_phi: Phase diffusion rate.
        T_H: Holding-time strength.
        theta_true: True phase rate.

    Returns:
        Δθ (should equal 1/T_H = SQL for any γ_φ).
    """
    ops = build_two_qubit_operators()
    return compute_noisy_sensitivity(
        DEFAULT_RHO0,
        DEFAULT_T_BS,
        T_H,
        theta_true,
        gamma_phi,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )


# ============================================================================
# Objective Function for Nelder-Mead
# ============================================================================


def noisy_sensitivity_objective(
    params: np.ndarray,
    theta_true: float,
    gamma_phi: float,
    ops: dict[str, np.ndarray],
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    fd_step: float = FD_STEP,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δθ under phase diffusion.

    Fixed configuration: |00⟩ initial state, fixed T_BS, fixed T_H.
    params = [a_x, a_y, a_z, a_zz] (4 elements).

    Args:
        params: 4-element parameter vector.
        theta_true: True phase rate.
        gamma_phi: Phase diffusion rate.
        ops: Two-qubit operators.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        fd_step: Finite-difference step.
        bounds: (min, max) for all parameters.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        Δθ (plus infinite penalty if bounds violated).
    """
    ax = float(params[0])
    ay = float(params[1])
    az = float(params[2])
    azz = float(params[3])

    # Bound enforcement
    lo, hi = bounds
    penalty = 0.0
    for val in (ax, ay, az, azz):
        if val < lo:
            penalty += penalty_scale * (lo - val) ** 2
        if val > hi:
            penalty += penalty_scale * (val - hi) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    return compute_noisy_sensitivity(
        DEFAULT_RHO0,
        T_BS,
        T_H,
        theta_true,
        gamma_phi,
        ax,
        ay,
        az,
        azz,
        ops,
        fd_step,
    )


# ============================================================================
# Random Search
# ============================================================================


def run_noisy_random_search(
    theta: float,
    gamma_phi: float,
    n_samples: int = 500,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]:
    """Random search over the 4D parameter space at fixed (θ, γ_φ).

    Args:
        theta: Phase rate value.
        gamma_phi: Phase diffusion rate.
        n_samples: Number of random points to evaluate.
        bounds: (min, max) for all four coefficients.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        seed: Random seed for reproducibility.

    Returns:
        Tuple (samples, delta_theta_values, best_params, best_delta_theta).
    """
    rng = np.random.default_rng(seed)
    ops = build_two_qubit_operators()
    lo, hi = bounds

    samples = rng.uniform(lo, hi, size=(n_samples, 4))
    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        ax = float(samples[i, 0])
        ay = float(samples[i, 1])
        az = float(samples[i, 2])
        azz = float(samples[i, 3])

        dtheta = compute_noisy_sensitivity(
            DEFAULT_RHO0,
            T_BS,
            T_H,
            theta,
            gamma_phi,
            ax,
            ay,
            az,
            azz,
            ops,
        )
        deltas[i] = dtheta

    best_idx = int(np.argmin(deltas))
    best_params: tuple[float, float, float, float] = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
    )
    best_delta = float(deltas[best_idx])

    return samples, deltas, best_params, best_delta


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


def run_noisy_nelder_mead(
    theta_true: float,
    gamma_phi: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
) -> dict[str, Any]:
    """Run Nelder-Mead optimisation for the noisy protocol at fixed (θ, γ_φ).

    Args:
        theta_true: True phase rate parameter.
        gamma_phi: Phase diffusion rate.
        x0: Initial 4-parameter vector [ax, ay, az, azz]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder-Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        bounds: (min, max) for all four parameters.
        T_H: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        Dict with keys: 'delta_theta_opt', 'params_opt', 'success',
        'nfev', 'message', 'expectation_Jz', 'variance_Jz', 'd_exp_d_theta'.
    """
    ops = build_two_qubit_operators()

    if x0 is None:
        rng = np.random.default_rng(seed)
        lo, hi = bounds
        x0 = rng.uniform(lo, hi, size=4)
    else:
        x0 = np.asarray(x0, dtype=float)
        assert x0.shape == (4,), f"x0 must have 4 elements, got {x0.shape}"

    def objective(p: np.ndarray) -> float:
        return noisy_sensitivity_objective(
            p,
            theta_true,
            gamma_phi,
            ops,
            T_H=T_H,
            T_BS=T_BS,
            bounds=bounds,
        )

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Compute diagnostics at the optimal point
    _, exp_val, var_val, d_exp = compute_noisy_sensitivity_with_diagnostics(
        DEFAULT_RHO0,
        T_BS,
        T_H,
        theta_true,
        gamma_phi,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )

    return {
        "delta_theta_opt": float(result.fun),
        "params_opt": opt_params,
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "message": str(result.message),
        "expectation_Jz": exp_val,
        "variance_Jz": var_val,
        "d_exp_d_theta": d_exp,
    }


# ============================================================================
# Noise Scan: (θ, γ_φ) Grid
# ============================================================================


def _run_single_noise_pair(
    theta: float,
    gamma_phi: float,
    n_random: int = 1000,
    n_nm_refine: int = 25,
    seed: int | None = 42,
    maxiter: int = 5000,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
) -> dict[str, Any]:
    """Run random search + Nelder-Mead refinement for a single (θ, γ_φ) pair.

    Args:
        theta: Phase rate value.
        gamma_phi: Phase diffusion rate.
        n_random: Number of random search points.
        n_nm_refine: Number of Nelder-Mead refinements.
        seed: Base random seed.
        maxiter: Maximum Nelder-Mead iterations.
        bounds: (min, max) for all four coefficients.
        T_H: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        Dict with results for this (θ, γ_φ) pair.
    """
    base_seed = seed if seed is not None else 42
    seed_raw = base_seed + int(theta * 1000) + int(100 * np.log10(gamma_phi + 1e-10))
    seed_val = abs(seed_raw)  # ensure non-negative for numpy

    # Stage 1: Random search
    samples, deltas, best_params, best_delta = run_noisy_random_search(
        theta,
        gamma_phi,
        n_samples=n_random,
        bounds=bounds,
        T_H=T_H,
        T_BS=T_BS,
        seed=seed_val,
    )

    # Stage 2: Nelder-Mead refinement from top candidates
    sorted_indices = np.argsort(deltas)
    top_indices = sorted_indices[: min(n_nm_refine, n_random)]

    nm_best_delta = best_delta
    nm_best_params = best_params
    nm_best_diag: dict[str, float] = {
        "expectation_Jz": 0.0,
        "variance_Jz": 0.0,
        "d_exp_d_theta": 0.0,
    }

    for rank, idx in enumerate(top_indices):
        x0 = samples[idx].copy()
        nm_result = run_noisy_nelder_mead(
            theta_true=theta,
            gamma_phi=gamma_phi,
            x0=x0,
            seed=seed_val + 10000 + rank,
            maxiter=maxiter,
            bounds=bounds,
            T_H=T_H,
            T_BS=T_BS,
        )

        dt = float(nm_result["delta_theta_opt"])
        if np.isfinite(dt) and dt < nm_best_delta:
            nm_best_delta = dt
            nm_best_params = (
                float(nm_result["params_opt"][0]),
                float(nm_result["params_opt"][1]),
                float(nm_result["params_opt"][2]),
                float(nm_result["params_opt"][3]),
            )
            nm_best_diag = {
                "expectation_Jz": float(nm_result["expectation_Jz"]),
                "variance_Jz": float(nm_result["variance_Jz"]),
                "d_exp_d_theta": float(nm_result["d_exp_d_theta"]),
            }

    return {
        "theta": theta,
        "gamma_phi": gamma_phi,
        "a_x": nm_best_params[0],
        "a_y": nm_best_params[1],
        "a_z": nm_best_params[2],
        "a_zz": nm_best_params[3],
        "delta_theta": nm_best_delta,
        "expectation_Jz": nm_best_diag["expectation_Jz"],
        "variance_Jz": nm_best_diag["variance_Jz"],
        "d_exp_d_theta": nm_best_diag["d_exp_d_theta"],
    }


def run_noise_scan(
    theta_values: list[float] | np.ndarray,
    gamma_phi_values: list[float] | np.ndarray,
    n_random: int = 1000,
    n_nm_refine: int = 25,
    seed: int | None = 42,
    maxiter: int = 5000,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
) -> DriveNoiseScanResult:
    """Scan over (θ, γ_φ) pairs with random search + Nelder-Mead refinement.

    For each (θ, γ_φ) pair:
    1. Run `n_random` random evaluations in the 4D parameter space.
    2. Select the best `n_nm_refine` points.
    3. Run Nelder-Mead refinement from each selected point.
    4. Record the best overall result.

    Args:
        theta_values: θ values to scan.
        gamma_phi_values: γ_φ values to scan.
        n_random: Number of random search points per pair.
        n_nm_refine: Number of Nelder-Mead refinements per pair.
        seed: Base random seed.
        maxiter: Maximum Nelder-Mead iterations.
        bounds: (min, max) for all parameters.
        T_H: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        DriveNoiseScanResult with optimal parameters and sensitivities.
    """
    theta_arr = np.asarray(theta_values, dtype=float)
    gamma_arr = np.asarray(gamma_phi_values, dtype=float)
    n_theta = len(theta_arr)
    n_gamma = len(gamma_arr)

    all_results: list[dict[str, Any]] = []

    for theta in theta_arr:
        for gamma in gamma_arr:
            result = _run_single_noise_pair(
                theta,
                gamma,
                n_random=n_random,
                n_nm_refine=n_nm_refine,
                seed=seed,
                maxiter=maxiter,
                bounds=bounds,
                T_H=T_H,
                T_BS=T_BS,
            )
            all_results.append(result)

    # Build the result structure
    params_list: list[tuple[float, float, float, float]] = []
    dt_arr = np.full((n_theta, n_gamma), np.inf, dtype=float)
    exp_arr = np.zeros((n_theta, n_gamma), dtype=float)
    var_arr = np.zeros((n_theta, n_gamma), dtype=float)
    d_exp_arr = np.zeros((n_theta, n_gamma), dtype=float)

    for r in all_results:
        t_val = float(r["theta"])
        g_val = float(r["gamma_phi"])
        i = int(np.where(theta_arr == t_val)[0][0])
        j = int(np.where(gamma_arr == g_val)[0][0])
        params_list.append(
            (
                float(r["a_x"]),
                float(r["a_y"]),
                float(r["a_z"]),
                float(r["a_zz"]),
            )
        )
        dt_arr[i, j] = float(r["delta_theta"])
        exp_arr[i, j] = float(r.get("expectation_Jz", 0.0))
        var_arr[i, j] = float(r.get("variance_Jz", 0.0))
        d_exp_arr[i, j] = float(r.get("d_exp_d_theta", 0.0))

    bounds_lo_f, bounds_hi_f = bounds
    return DriveNoiseScanResult(
        theta_values=theta_arr,
        gamma_phi_values=gamma_arr,
        best_params_per_pair=params_list,
        delta_theta_per_pair=dt_arr,
        expectation_Jz_per_pair=exp_arr,
        variance_Jz_per_pair=var_arr,
        d_exp_d_theta_per_pair=d_exp_arr,
        sql=1.0 / T_H,
        T_H=T_H,
        n_random=n_random,
        n_nm_refine=n_nm_refine,
        maxiter=maxiter,
        bounds_lo=float(bounds_lo_f),
        bounds_hi=float(bounds_hi_f),
        fd_step=FD_STEP,
        seed=seed if seed is not None else 42,
    )


# ============================================================================
# Fixed-Parameter Scan (using noise-free optimal params)
# ============================================================================


def evaluate_fixed_params_scan(
    theta_values: list[float] | np.ndarray,
    gamma_phi_values: list[float] | np.ndarray,
    noise_free_params: dict[float, tuple[float, float, float, float]],
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
) -> np.ndarray:
    """Evaluate Δθ at noise-free optimal params for each (θ, γ_φ) pair.

    Args:
        theta_values: θ values.
        gamma_phi_values: γ_φ values.
        noise_free_params: Dict mapping θ → (a_x*, a_y*, a_z*, a_zz*)
            from the noise-free (γ_φ = 0) optimum.
        T_H: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        Array of Δθ values, shape (n_theta, n_gamma).
    """
    ops = build_two_qubit_operators()
    theta_arr = np.asarray(theta_values, dtype=float)
    gamma_arr = np.asarray(gamma_phi_values, dtype=float)
    n_theta = len(theta_arr)
    n_gamma = len(gamma_arr)

    result = np.full((n_theta, n_gamma), np.inf, dtype=float)

    for i, theta in enumerate(theta_arr):
        if theta not in noise_free_params:
            continue
        ax, ay, az, azz = noise_free_params[theta]
        for j, gamma in enumerate(gamma_arr):
            dt = compute_noisy_sensitivity(
                DEFAULT_RHO0,
                T_BS,
                T_H,
                theta,
                gamma,
                ax,
                ay,
                az,
                azz,
                ops,
            )
            result[i, j] = dt

    return result


# ============================================================================
# Exclusive Plot Functions
# ============================================================================


def plot_noise_sensitivity_heatmap(
    result: DriveNoiseScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 7),
) -> Path:
    """Heatmap of Δθ/Δθ_SQL as a function of θ and γ_φ.

    Args:
        result: DriveNoiseScanResult.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ratio = result.delta_theta_per_pair / result.sql

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        result.theta_values,
        result.gamma_phi_values,
        ratio.T,
        shading="auto",
        cmap="RdYlBu_r",
        norm=LogNorm(
            vmin=max(np.nanmin(ratio[ratio > 0]), 1e-2),
            vmax=max(np.nanmax(ratio[ratio < np.inf]), 10),
        ),
    )
    fig.colorbar(im, ax=ax, label=r"$\Delta\theta / \Delta\theta_{\mathrm{SQL}}$")

    # Contour line at Δθ = SQL
    cs = ax.contour(
        result.theta_values,
        result.gamma_phi_values,
        ratio.T,
        levels=[1.0],
        colors="k",
        linewidths=1.5,
        linestyles="--",
    )
    ax.clabel(cs, fmt=r"SQL", fontsize=10)

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\gamma_\phi$")
    ax.set_yscale("log")
    ax.set_title(
        "Sensitivity ratio vs phase diffusion rate\n"
        "(re-optimised $(a_x, a_y, a_z, a_{zz})$ per point)"
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_noise_sensitivity_curves(
    result: DriveNoiseScanResult,
    theta_subset: list[float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Line plot of Δθ vs γ_φ for selected θ values.

    Args:
        result: DriveNoiseScanResult.
        theta_subset: List of θ values to plot. If None, uses all.
        save_path: Output SVG path. If None, does not save.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG, or a dummy path if not saved.
    """
    if theta_subset is None:
        theta_subset = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    fig, ax = plt.subplots(figsize=figsize)

    # SQL reference line
    ax.axhline(
        y=result.sql,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=rf"SQL = {result.sql:.4f}",
    )

    _viridis_cmap = colormaps.get_cmap("viridis")
    colors = _viridis_cmap(np.linspace(0, 1, len(theta_subset)))

    for theta_val, color in zip(theta_subset, colors, strict=False):
        # Find the index
        idx = np.where(np.isclose(result.theta_values, theta_val, atol=1e-6))[0]
        if len(idx) == 0:
            continue
        i = idx[0]

        valid = np.isfinite(result.delta_theta_per_pair[i, :])
        if np.any(valid):
            ax.plot(
                result.gamma_phi_values[valid],
                result.delta_theta_per_pair[i, valid],
                "o-",
                color=color,
                markersize=5,
                linewidth=1.5,
                label=rf"$\theta$ = {theta_val:.1f}",
            )

    ax.set_xlabel(r"$\gamma_\phi$")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Sensitivity vs phase diffusion rate at fixed $\\theta$")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return Path()


def plot_noise_optimal_params(
    result: DriveNoiseScanResult,
    theta_subset: list[float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> Path:
    """Plot optimal parameters as a function of γ_φ for selected θ values.

    Args:
        result: DriveNoiseScanResult.
        theta_subset: List of θ values to plot. If None, uses all.
        save_path: Output SVG path.
        figsize: Figure size (width, height).

    Returns:
        Path to saved SVG.
    """
    if theta_subset is None:
        theta_subset = [0.1, 0.5, 1.0, 2.0, 5.0]

    param_names = [r"$a_x^*$", r"$a_y^*$", r"$a_z^*$", r"$a_{zz}^*$"]

    n_params = 4
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)

    _viridis_cmap = colormaps.get_cmap("viridis")
    colors = _viridis_cmap(np.linspace(0, 1, len(theta_subset)))

    for param_idx in range(n_params):
        ax = axes[param_idx]
        for theta_val, color in zip(theta_subset, colors, strict=False):
            idx = np.where(np.isclose(result.theta_values, theta_val, atol=1e-6))[0]
            if len(idx) == 0:
                continue
            i = idx[0]

            # Extract the parameter values for this theta across all gamma
            param_vals = []
            for j in range(len(result.gamma_phi_values)):
                pair_idx = i * len(result.gamma_phi_values) + j
                params = result.best_params_per_pair[pair_idx]
                param_vals.append(params[param_idx])

            param_arr = np.array(param_vals)
            valid = np.isfinite(result.delta_theta_per_pair[i, :]) & np.isfinite(
                param_arr
            )
            if np.any(valid):
                ax.plot(
                    result.gamma_phi_values[valid],
                    param_arr[valid],
                    "o-",
                    color=color,
                    markersize=4,
                    linewidth=1.2,
                    label=rf"$\theta$ = {theta_val:.1f}" if param_idx == 0 else "",
                )

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel(param_names[param_idx])
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(r"$\gamma_\phi$")
    axes[0].set_title(
        "Optimal parameters vs phase diffusion rate\n"
        "(re-optimised per $\\theta$, $\\gamma_\\phi$ pair)"
    )
    axes[0].legend(fontsize=8, loc="upper left", ncol=len(theta_subset))

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return Path()


# ============================================================================
# Data / Figure Generation Pipeline
# ============================================================================

REPORTS_DIR = PROJECT_ROOT / "reports"
DATE_TAG = "2026-05-24"


def _parquet_path(name: str) -> Path:
    return REPORTS_DIR / DATE_TAG / "raw_data" / f"{DATE_TAG}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return REPORTS_DIR / DATE_TAG / "figures" / f"{DATE_TAG}-{name}.svg"


# ── Parallel dispatch helper ──────────────────────────────────────────────


def _parallel_map(
    worker_fn: Any,
    items: list[Any],
    desc: str = "Processing",
    max_workers: int | None = None,
) -> None:
    """Run *worker_fn(item)* for each *item* in parallel via process pool.

    Each worker is a top-level function that performs its own file I/O.
    Results are implicitly persisted to disk by the worker.

    Args:
        worker_fn: Callable taking a single item argument.
        items: Iterable of items.
        desc: Short description for progress logging.
        max_workers: Number of subprocess workers (default: CPU count).
    """
    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 1)
    item_list = list(items)
    print(f"  [parallel] {desc}: {len(item_list)} items, {max_workers} workers")

    mp_ctx = _mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_item = {executor.submit(worker_fn, item): item for item in item_list}
        for future in concurrent.futures.as_completed(fut_to_item):
            item = fut_to_item[future]
            try:
                future.result()
            except Exception as exc:
                print(f"  [ERROR] item={item}: {exc}")
                raise


# ── Generator functions ───────────────────────────────────────────────────


def _run_single_pair_worker(args: dict[str, Any]) -> None:
    """Worker function for parallel noise scan.

    Runs one (θ, γ_φ) pair and saves to a separate Parquet file.
    """
    theta = args["theta"]
    gamma_phi = args["gamma_phi"]
    force = args["force"]
    n_random = args.get("n_random", 1000)
    n_nm_refine = args.get("n_nm_refine", 25)
    seed = args.get("seed", 42)

    tag = f"noise-pair-theta{theta}-gamma{gamma_phi:.6e}"
    csv_p = _parquet_path(tag)

    if csv_p.exists() and not force:
        print(f"  [skip] {csv_p.name} exists (use --force to overwrite)")
        return

    print(f"  [run]  (θ={theta}, γ_φ={gamma_phi:.2e})...")
    result = _run_single_noise_pair(
        theta,
        gamma_phi,
        n_random=n_random,
        n_nm_refine=n_nm_refine,
        seed=seed,
    )

    # Save individual result as a single-row DataFrame
    row = pd.DataFrame(
        [
            {
                "theta": theta,
                "gamma_phi": gamma_phi,
                "a_x": result["a_x"],
                "a_y": result["a_y"],
                "a_z": result["a_z"],
                "a_zz": result["a_zz"],
                "delta_theta": result["delta_theta"],
                "expectation_Jz": result.get("expectation_Jz", 0.0),
                "variance_Jz": result.get("variance_Jz", 0.0),
                "d_exp_d_theta": result.get("d_exp_d_theta", 0.0),
                "sql": SQL_REFERENCE,
                "T_H": DEFAULT_T_H,
                "n_random": n_random,
                "n_nm_refine": n_nm_refine,
                "maxiter": 5000,
                "bounds_lo": DRIVE_BOUNDS[0],
                "bounds_hi": DRIVE_BOUNDS[1],
                "fd_step": FD_STEP,
                "seed": seed,
            }
        ]
    )
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    row.to_parquet(csv_p, index=False)
    print(f"  [save] {csv_p.name}")


def generate_noise_scan(force: bool = False) -> None:
    """Run the full (θ, γ_φ) noise scan in parallel.

    Each (θ, γ_φ) pair is computed independently and saved to a separate
    Parquet file. After all pairs complete, results are aggregated into a
    single DriveNoiseScanResult and saved.
    """
    agg_p = _parquet_path("noise-scan")

    if agg_p.exists() and not force:
        print(f"[skip] {agg_p.name} exists (use --force to overwrite)")
        result = DriveNoiseScanResult.from_parquet(agg_p)
    else:
        n_theta = len(THETA_VALS)
        n_gamma = len(GAMMA_PHI_VALS)
        n_total = n_theta * n_gamma
        print(f"[run]  Noise scan: {n_total} (θ, γ_φ) pairs (parallel)")

        # Build worker arguments
        args_list = [
            {
                "theta": theta,
                "gamma_phi": gamma_phi,
                "force": force,
                "n_random": 1000,
                "n_nm_refine": 25,
                "seed": 42,
            }
            for theta in THETA_VALS
            for gamma_phi in GAMMA_PHI_VALS
        ]

        _parallel_map(
            _run_single_pair_worker,
            args_list,
            desc=f"noise scan ({n_total} pairs)",
        )

        # Aggregate results from individual Parquet files
        print("  [aggregate] Building aggregated result...")
        all_rows: list[pd.DataFrame] = []
        for theta in THETA_VALS:
            for gamma_phi in GAMMA_PHI_VALS:
                tag = f"noise-pair-theta{theta}-gamma{gamma_phi:.6e}"
                csv_p = _parquet_path(tag)
                if csv_p.exists():
                    all_rows.append(pd.read_parquet(csv_p))

        if not all_rows:
            print("  [WARNING] No individual results found.")
            return

        df_agg = pd.concat(all_rows, ignore_index=True)

        # Build the aggregated result
        theta_arr = np.array(THETA_VALS, dtype=float)
        gamma_arr = np.array(GAMMA_PHI_VALS, dtype=float)
        n_theta = len(theta_arr)
        n_gamma = len(gamma_arr)

        params_list: list[tuple[float, float, float, float]] = []
        dt_arr = np.full((n_theta, n_gamma), np.inf, dtype=float)
        exp_arr = np.zeros((n_theta, n_gamma), dtype=float)
        var_arr = np.zeros((n_theta, n_gamma), dtype=float)
        d_exp_arr = np.zeros((n_theta, n_gamma), dtype=float)

        lookup: dict[tuple[float, float], dict[str, float]] = {}
        for _, row in df_agg.iterrows():
            key = (float(row["theta"]), float(row["gamma_phi"]))
            lookup[key] = {
                "a_x": float(row["a_x"]),
                "a_y": float(row["a_y"]),
                "a_z": float(row["a_z"]),
                "a_zz": float(row["a_zz"]),
                "delta_theta": float(row["delta_theta"]),
                "expectation_Jz": float(row.get("expectation_Jz", 0.0)),
                "variance_Jz": float(row.get("variance_Jz", 0.0)),
                "d_exp_d_theta": float(row.get("d_exp_d_theta", 0.0)),
            }

        for i, t in enumerate(theta_arr):
            for j, g in enumerate(gamma_arr):
                entry = lookup.get((t, g), {})
                params_list.append(
                    (
                        entry.get("a_x", 0.0),
                        entry.get("a_y", 0.0),
                        entry.get("a_z", 0.0),
                        entry.get("a_zz", 0.0),
                    )
                )
                dt_arr[i, j] = entry.get("delta_theta", float("inf"))
                exp_arr[i, j] = entry.get("expectation_Jz", 0.0)
                var_arr[i, j] = entry.get("variance_Jz", 0.0)
                d_exp_arr[i, j] = entry.get("d_exp_d_theta", 0.0)

        # Read hyperparameters from the first row of the aggregated data
        first_row = df_agg.iloc[0]
        result = DriveNoiseScanResult(
            theta_values=theta_arr,
            gamma_phi_values=gamma_arr,
            best_params_per_pair=params_list,
            delta_theta_per_pair=dt_arr,
            expectation_Jz_per_pair=exp_arr,
            variance_Jz_per_pair=var_arr,
            d_exp_d_theta_per_pair=d_exp_arr,
            sql=SQL_REFERENCE,
            T_H=DEFAULT_T_H,
            n_random=int(first_row.get("n_random", 1000)),
            n_nm_refine=int(first_row.get("n_nm_refine", 25)),
            maxiter=int(first_row.get("maxiter", 5000)),
            bounds_lo=float(first_row.get("bounds_lo", -5.0)),
            bounds_hi=float(first_row.get("bounds_hi", 5.0)),
            fd_step=float(first_row.get("fd_step", FD_STEP)),
            seed=int(first_row.get("seed", 42)),
        )
        result.save_parquet(agg_p)
        print(f"[save] {agg_p}")

    # Generate figures
    generate_noise_figures(result)


def generate_noise_figures(result: DriveNoiseScanResult | None = None) -> None:
    """Generate all figures from an existing or loaded result."""
    if result is None:
        agg_p = _parquet_path("noise-scan")
        if not agg_p.exists():
            print("[skip] noise-scan.parquet does not exist; run 'noise-scan' first")
            return
        result = DriveNoiseScanResult.from_parquet(agg_p)

    # Heatmap
    fig_p = _fig_path("noise-sensitivity-heatmap")
    plot_noise_sensitivity_heatmap(result, fig_p)
    print(f"[fig]  {fig_p}")

    # Sensitivity curves
    fig_p = _fig_path("noise-sensitivity-curves")
    plot_noise_sensitivity_curves(result, save_path=fig_p)
    print(f"[fig]  {fig_p}")

    # Optimal parameters
    fig_p = _fig_path("noise-optimal-params")
    plot_noise_optimal_params(result, save_path=fig_p)
    print(f"[fig]  {fig_p}")


def generate_noise_decoupled_baseline(force: bool = False) -> None:
    """Verify the decoupled baseline under phase diffusion."""
    csv_p = _parquet_path("noise-decoupled-baseline")

    if csv_p.exists() and not force:
        print(f"[skip] {csv_p.name} exists (use --force to overwrite)")
    else:
        print("[run]  Computing decoupled baseline under phase diffusion...")
        gamma_test = [0.0, 1e-4, 1e-2, 1.0, 10.0]
        rows = []
        for g in gamma_test:
            dt = compute_noisy_decoupled_baseline(gamma_phi=g)
            rows.append(
                {
                    "gamma_phi": g,
                    "delta_theta": dt,
                    "sql": SQL_REFERENCE,
                    "T_H": DEFAULT_T_H,
                }
            )
        df = pd.DataFrame(rows)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(csv_p, index=False)
        print(f"[save] {csv_p}")


def generate_noise_validation(force: bool = False) -> None:
    """Run validation checks: trace, Hermiticity, positivity."""
    print("[run]  Running validation checks...")
    ops = build_two_qubit_operators()

    # Test a small set of (θ, γ_φ) pairs
    test_thetas = [0.1, 1.0, 5.0]
    test_gammas = [0.0, 1e-4, 1e-2, 1.0]
    test_params_list = [
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 1.0),
        (0.0, 3.0, 0.0, -2.0),
    ]

    n_pass = 0
    n_total = 0
    for theta in test_thetas:
        for gamma in test_gammas:
            for params in test_params_list:
                ax, ay, az, azz = params
                try:
                    rho = evolve_noisy_drive_circuit(
                        DEFAULT_RHO0,
                        DEFAULT_T_BS,
                        DEFAULT_T_H,
                        theta,
                        gamma,
                        ax,
                        ay,
                        az,
                        azz,
                        ops,
                    )
                    validate_density(rho)
                    n_pass += 1
                except AssertionError as e:
                    print(f"  [FAIL] θ={theta}, γ_φ={gamma}, params={params}: {e}")
                n_total += 1

    print(f"  Validation: {n_pass}/{n_total} passed")

    # Also test sensitivity is finite
    n_sens_pass = 0
    n_sens_total = 0
    for theta in test_thetas:
        for gamma in test_gammas:
            for params in [test_params_list[0], test_params_list[1]]:
                ax, ay, az, azz = params
                dt = compute_noisy_sensitivity(
                    DEFAULT_RHO0,
                    DEFAULT_T_BS,
                    DEFAULT_T_H,
                    theta,
                    gamma,
                    ax,
                    ay,
                    az,
                    azz,
                    ops,
                )
                n_sens_total += 1
                if np.isfinite(dt):
                    n_sens_pass += 1

    print(f"  Sensitivity valid: {n_sens_pass}/{n_sens_total} ")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026-05-24 report figures and Parquet data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations (overwrite existing Parquets)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Generate only one dataset (e.g. 'noise-decoupled-baseline')",
    )
    args = parser.parse_args()

    # Ensure directories exist
    (REPORTS_DIR / DATE_TAG / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / DATE_TAG / "figures").mkdir(parents=True, exist_ok=True)

    def _run_all(force: bool = False) -> None:
        generate_noise_decoupled_baseline(force=force)
        generate_noise_scan(force=force)

    tasks = {
        "noise-decoupled-baseline": generate_noise_decoupled_baseline,
        "noise-scan": generate_noise_scan,
        "noise-validation": generate_noise_validation,
        "figures": lambda force=args.force: generate_noise_figures(),
        "all": _run_all,
    }

    if args.only:
        if args.only not in tasks:
            print(f"Unknown dataset '{args.only}'. Options: {list(tasks.keys())}")
            sys.exit(1)
        tasks[args.only](force=args.force)
    else:
        for name, func in tasks.items():
            if name == "all":
                continue
            print(f"\n=== {name} ===")
            func(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()

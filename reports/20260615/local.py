"""
Local module for the 2026-06-15 Non-Linear Measurement (Parity and CFI) report.

Replaces the linear J_z^S measurement in the ω-modulated drive protocol (#20260519)
with two non-linear strategies: parity Π_S = exp(iπ J_z^S) and full-distribution
Classical Fisher Information (CFI) from the J_z^S eigenbasis.

Circuit: BS_S → Hold → BS_S → measure.

Three S-only measurement protocols:
  - Linear (baseline): O = J_z^S
  - Parity: Π_S = exp(iπ J_z^S) (even N only)
  - CFI: F_C(ω) from P(m_S|ω)

Two stages:
  - Stage A: Evaluate all three protocols at #20260613's joint-optimal params.
  - Stage B: Re-optimise the 4D parameter space independently per protocol.

Usage:
    uv run python reports/20260615/local.py --force
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

from src.analysis.ancilla_optimization import compute_expectation_and_variance
from src.analysis.fisher_information import classical_fisher_information_single
from src.analysis.sensitivity_metrics import sql_reference
from src.physics.dicke_basis import jz_operator
from src.physics.n_particle_drive import (
    build_n_particle_operators,
    compute_n_particle_sensitivity,
    evolve_n_particle_circuit,
    n_particle_initial_state,
)
from src.utils.enums import OperatorBasis
from src.utils.paths import fig_path, parquet_path
from src.utils.serialization import ParquetSerializable

# ============================================================================
# Constants
# ============================================================================

# ω values matching #20260613
OMEGA_VALS: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0]

# N values matching #20260613 (subset: where joint-opt data exists)
STAGE_A_N_VALS: list[int] = [1, 2, 3, 4, 5, 8, 10, 15, 20]

# N values for Stage B (reduced range)
STAGE_B_N_VALS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Even N values for parity (only Hermitian at even N)
EVEN_N_VALS: list[int] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Random search and NM parameters
N_RANDOM: int = 1000
N_NM_REFINE: int = 30
NM_MAXITER: int = 5000

# Local copies of shared constants (originally in src/physics/n_particle_drive.py)
FD_STEP: float = 1e-6  # Finite-difference step
T_HOLD: float = 10.0  # Holding time (fixed)
T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Parameter bounds

# Probability floor for CFI
PROB_FLOOR: float = 1e-12

# Protocol type identifiers
PROTOCOL_LINEAR: str = "linear"
PROTOCOL_PARITY: str = "parity"
PROTOCOL_CFI: str = "cfi"

if TYPE_CHECKING:
    from collections.abc import Callable

# ============================================================================
# Path Helpers
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260615"
JOINT_REPORT_DATE = "20260613"


def _parquet_path(name: str) -> Path:
    return parquet_path(REPORTS_DIR, REPORT_DATE, name)


def _fig_path(name: str) -> Path:
    return fig_path(REPORTS_DIR, REPORT_DATE, name)


def _try_load_scan_cache(out_path: Path, force: bool) -> NonLinearScanResult | None:
    """Return cached scan result if available and not forced."""
    if out_path.exists() and not force:
        print(f"[load] {out_path.name} exists, loading...")
        return NonLinearScanResult.from_parquet(out_path)
    return None


def _skip_parity_odd(proto: str, N: int) -> bool:
    """Parity protocol is not Hermitian for odd N."""
    return proto == PROTOCOL_PARITY and N % 2 == 1


# ============================================================================
# Parity Operator Construction
# ============================================================================


def build_parity_operator(N: int) -> np.ndarray:
    """Build the parity operator Π_S = exp(iπ J_z^S) ⊗ I_2.

    For even N, J_z^S has integer eigenvalues m_S ∈ {-N/2, ..., N/2},
    so exp(iπ m_S) = (-1)^{m_S} ∈ {±1}. The operator is Hermitian and
    satisfies Π_S² = I_tot.

    For odd N, J_z^S has half-integer eigenvalues, giving exp(iπ m_S) = ±i
    — the operator is anti-Hermitian. A ValueError is raised.

    Args:
        N: Number of system particles.

    Returns:
        2(N+1) × 2(N+1) Hermitian matrix (even N only).

    Raises:
        ValueError: If N is odd (parity is anti-Hermitian).
    """
    if N % 2 == 1:
        raise ValueError(
            f"Parity operator is not Hermitian for odd N={N}. "
            "Use only even N (N/2 integer, so J_z^S eigenvalues are half-integer)."
        )

    # Π_S^sys = exp(iπ J_z^S) — compute in the Dicke basis (system only)
    # Use jz_operator directly since extracting from the kron product is fragile
    d_sys = N + 1
    Jz_S_dicke = jz_operator(N, basis=OperatorBasis.DICKE)
    pi_Jz_sys = 1j * np.pi * Jz_S_dicke
    parity_sys = expm(pi_Jz_sys)

    # Extend to full space: Π_S ⊗ I_2
    d_tot = 2 * d_sys
    I_2 = np.eye(2, dtype=complex)
    parity = np.kron(parity_sys, I_2).astype(complex)

    # Validate Hermiticity
    assert np.allclose(parity, parity.conj().T, atol=1e-12), (
        f"Parity operator not Hermitian for N={N}"
    )
    # Validate Π² = I
    assert np.allclose(parity @ parity, np.eye(d_tot, dtype=complex), atol=1e-12), (
        f"Parity operator does not square to identity for N={N}"
    )

    return parity


# ============================================================================
# J_z Probability Distribution
# ============================================================================


def compute_jz_projectors(N: int, ops: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Pre-compute projectors |m_S⟩⟨m_S| ⊗ I_2 for each J_z^S eigenvalue.

    These are used to compute P(m_S|ω) = ⟨ψ_final|ℙ_{m_S}|ψ_final⟩.

    Args:
        N: Number of system particles.
        ops: Operators from build_n_particle_operators(N).

    Returns:
        List of (d_tot, d_tot) projectors, one per m_S ∈ {-J_S, ..., J_S}.
        Length = N+1.
    """
    d_sys = N + 1
    d_tot = 2 * d_sys
    I_2 = np.eye(2, dtype=complex)

    # The J_z^S operator is block-diagonal in the Dicke basis.
    # The eigenstates are |m_S⟩ for m_S = -J_S, -J_S+1, ..., J_S.
    # In our basis ordering, m_S descends from +J_S to -J_S.
    # So index i corresponds to m = +J_S - i.

    projectors: list[np.ndarray] = []
    for i in range(d_sys):
        proj_sys = np.zeros((d_sys, d_sys), dtype=complex)
        proj_sys[i, i] = 1.0
        proj_full = np.kron(proj_sys, I_2).astype(complex)
        assert proj_full.shape == (d_tot, d_tot), (
            f"Projector shape {proj_full.shape} != ({d_tot}, {d_tot})"
        )
        projectors.append(proj_full)

    return projectors


def compute_jz_probability_distribution(
    psi: np.ndarray,
    projectors: list[np.ndarray],
) -> np.ndarray:
    """Compute P(m_S|ω) from the final state vector.

    P(m_S) = ⟨ψ|ℙ_{m_S}|ψ⟩ for each m_S.

    Args:
        psi: Final state vector (must be normalised).
        projectors: Projectors from compute_jz_projectors.

    Returns:
        Array P(m_S) of length len(projectors), summing to 1.
    """
    n_outcomes = len(projectors)
    probs = np.zeros(n_outcomes, dtype=float)
    for i, proj in enumerate(projectors):
        p = np.real(psi.conj() @ proj @ psi)
        # Clamp tiny negatives from numerical noise
        probs[i] = max(0.0, p)

    total = np.sum(probs)
    assert np.isclose(total, 1.0, atol=1e-10), (
        f"Probabilities sum to {total}, expected 1.0"
    )
    return probs / total  # Re-normalise for numerical safety


# ============================================================================
# Sensitivity Computation (Three Protocols)
# ============================================================================


def compute_parity_sensitivity(
    N: int,
    psi0: np.ndarray,
    T_bs: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = FD_STEP,
) -> tuple[float, float]:
    """Compute sensitivity using the parity measurement Π_S.

    Δω_Π = sqrt(1 - ⟨Π⟩²) / |∂⟨Π⟩/∂ω|

    For the dichotomic ±1 parity operator, Var(Π) = 1 - ⟨Π⟩².

    Args:
        N: Number of system particles (must be even).
        psi0: Initial state vector.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_n_particle_operators(N).
        fd_step: Finite-difference step size.

    Returns:
        Tuple (Δω_Π, ⟨Π⟩). Δω_Π = inf if singular.

    Raises:
        ValueError: If N is odd.
    """
    parity_op = build_parity_operator(N)

    # Evolve at omega_true
    psi = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_par, var_par = compute_expectation_and_variance(psi, parity_op)

    # For dichotomic ±1: Var(Π) = 1 - ⟨Π⟩²
    var_par_exact = 1.0 - exp_par**2
    assert np.isclose(var_par, var_par_exact, atol=1e-10), (
        f"Var(Π) = {var_par:.6e}, 1-⟨Π⟩² = {var_par_exact:.6e}"
    )
    var_par = max(var_par_exact, 0.0)

    # Central finite difference for ∂⟨Π⟩/∂ω
    psi_plus = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_plus = np.real(psi_plus.conj() @ parity_op @ psi_plus)
    exp_minus = np.real(psi_minus.conj() @ parity_op @ psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12 or var_par < 1e-15:
        return float("inf"), exp_par

    delta_omega = float(np.sqrt(var_par) / abs(d_exp))
    return delta_omega, exp_par


def compute_cfi_sensitivity(
    N: int,
    psi0: np.ndarray,
    T_bs: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    projectors: list[np.ndarray] | None = None,
    fd_step: float = FD_STEP,
    prob_floor: float = PROB_FLOOR,
) -> float:
    """Compute sensitivity using full-distribution CFI from J_z^S eigenbasis.

    1. Evolve at ω ± fd_step to get P(m_S|ω ± fd_step).
    2. Compute F_C(ω) = Σ [P(m_S|ω+ε) - P(m_S|ω-ε)]² / [4ε² P(m_S|ω)].
    3. Δω_C = 1 / √F_C.

    Args:
        N: Number of system particles.
        psi0: Initial state vector.
        T_bs: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Operators from build_n_particle_operators(N).
        projectors: Pre-computed projectors from compute_jz_projectors.
            If None, they are computed here.
        fd_step: Finite-difference step for CFI derivative.
        prob_floor: Minimum probability floor to avoid division by zero.

    Returns:
        Δω_C (positive float). Returns inf if F_C = 0.
    """
    if projectors is None:
        projectors = compute_jz_projectors(N, ops)

    # Evolve at ω, ω+ε, ω-ε
    psi = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_plus = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_n_particle_circuit(
        N,
        psi0,
        T_bs,
        t_hold,
        omega_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )

    # Compute probability distributions
    p_at_omega = compute_jz_probability_distribution(psi, projectors)
    p_plus = compute_jz_probability_distribution(psi_plus, projectors)
    p_minus = compute_jz_probability_distribution(psi_minus, projectors)

    # CFI at this single ω value
    fc = classical_fisher_information_single(
        p_plus,
        p_minus,
        fd_step,
        p_at_theta=p_at_omega,
        prob_floor=prob_floor,
    )

    if fc <= 1e-15:
        return float("inf")

    return float(1.0 / np.sqrt(fc))


def compute_protocol_sensitivity(
    protocol: str,
    N: int,
    psi0: np.ndarray,
    T_bs: float,
    t_hold: float,
    omega_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    projectors: list[np.ndarray] | None = None,
    fd_step: float = FD_STEP,
) -> tuple[float, float]:
    """Dispatch to the appropriate sensitivity function.

    For parity, also returns ⟨Π⟩. For linear and CFI, returns 0.0
    as the second value (no parity expectation).

    Args:
        protocol: One of 'linear', 'parity', 'cfi'.
        All other args: Same as the individual sensitivity functions.

    Returns:
        Tuple (Δω, extra) where extra is ⟨Π⟩ for parity, 0.0 otherwise.

    Raises:
        ValueError: If protocol is unknown.
    """
    if protocol == PROTOCOL_LINEAR:
        dw = compute_n_particle_sensitivity(
            N,
            psi0,
            T_bs,
            t_hold,
            omega_true,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
            fd_step=fd_step,
            meas_op=ops["Jz_S"],
        )
        return dw, 0.0
    if protocol == PROTOCOL_PARITY:
        dw, exp_par = compute_parity_sensitivity(
            N,
            psi0,
            T_bs,
            t_hold,
            omega_true,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
            fd_step=fd_step,
        )
        return dw, exp_par
    if protocol == PROTOCOL_CFI:
        dw = compute_cfi_sensitivity(
            N,
            psi0,
            T_bs,
            t_hold,
            omega_true,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
            projectors=projectors,
            fd_step=fd_step,
        )
        return dw, 0.0
    raise ValueError(
        f"Unknown protocol '{protocol}'. "
        f"Must be one of {[PROTOCOL_LINEAR, PROTOCOL_PARITY, PROTOCOL_CFI]}."
    )


# ============================================================================
# Decoupled Baseline Verification
# ============================================================================


def verify_decoupled_baseline(
    N_values: list[int] | None = None,
    omega_values: list[float] | None = None,
    rtol: float = 1e-10,
) -> dict[tuple[int, float], dict[str, bool | None]]:
    """Verify protocols at zero drive: CFI recovers SQL, parity is well-behaved.

    At a_x = a_y = a_z = a_zz = 0, the full J_z probability distribution
    (CFI) should recover Δω = 1/(√N T_H). The single-observable parity
    measurement Π_S is sub-optimal and does NOT recover the SQL.

    Args:
        N_values: List of N values (default: Stage A N values).
        omega_values: List of ω values (default: all OMEGA_VALS).
        rtol: Relative tolerance.

    Returns:
        Nested dict: {(N, ω): {'linear': bool|None, 'parity': bool|None,
                               'cfi': bool|None}}.
    """
    if N_values is None:
        N_values = STAGE_A_N_VALS
    if omega_values is None:
        omega_values = OMEGA_VALS

    results: dict[tuple[int, float], dict[str, bool | None]] = {}

    for N in N_values:
        ops = build_n_particle_operators(N)
        psi0 = n_particle_initial_state(N)
        projectors = compute_jz_projectors(N, ops)
        sql = sql_reference(N)

        for omega in omega_values:
            # Linear J_z measurement recovers the SQL at decoupled params
            dw_lin = compute_n_particle_sensitivity(
                N,
                psi0,
                T_BS,
                T_HOLD,
                omega,
                0.0,
                0.0,
                0.0,
                0.0,
                ops,
                meas_op=ops["Jz_S"],
            )
            lin_ok = bool(np.isclose(dw_lin, sql, rtol=rtol))

            # Parity (even N only): well-behaved but does NOT equal SQL
            # (parity collapses the full Dicke-basis distribution to a
            # binary ±1 outcome, losing Fisher information)
            par_ok: bool | None = None
            if N % 2 == 0:
                dw_par, exp_par = compute_parity_sensitivity(
                    N,
                    psi0,
                    T_BS,
                    T_HOLD,
                    omega,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    ops,
                )
                # Must be finite, positive, and real expectation
                par_ok = bool(
                    np.isfinite(dw_par)
                    and dw_par > 0.0
                    and np.isclose(np.imag(exp_par), 0.0, atol=1e-10)
                )

            # CFI from the full J_z distribution recovers the SQL
            dw_cfi = compute_cfi_sensitivity(
                N,
                psi0,
                T_BS,
                T_HOLD,
                omega,
                0.0,
                0.0,
                0.0,
                0.0,
                ops,
                projectors=projectors,
            )
            cfi_ok = bool(np.isclose(dw_cfi, sql, rtol=rtol))

            results[(N, omega)] = {
                "linear": lin_ok,
                "parity": par_ok,
                "cfi": cfi_ok,
            }

    return results


# ============================================================================
# Stage A: Fixed-Parameter Evaluation Using #20260613 Joint-Optimal Params
# ============================================================================


def load_joint_optimal_params(N: int, omega: float) -> dict[str, float] | None:
    """Load optimal drive parameters from the #20260613 joint measurement scan.

    Reads the per-omega Parquet file and returns the (a_x, a_y, a_z, a_zz)
    that were optimal for the joint measurement M(ψ) at the given (N, ω).

    Args:
        N: Number of system particles.
        omega: Phase rate value.

    Returns:
        Dict with keys 'a_x', 'a_y', 'a_z', 'a_zz' (all floats),
        or None if no data found.
    """
    # Try per-omega file first
    parquet_path = (
        REPORTS_DIR
        / JOINT_REPORT_DATE
        / "raw_data"
        / f"{JOINT_REPORT_DATE}-n-scaling-omega-{omega}.parquet"
    )
    if not parquet_path.exists():
        # Fall back to combined file
        parquet_path = REPORTS_DIR / JOINT_REPORT_DATE / "raw_data"
        candidates = list(
            parquet_path.glob(f"{JOINT_REPORT_DATE}-n-scaling-joint*.parquet")
        )
        if not candidates:
            return None
        parquet_path = candidates[0]

    df = pd.read_parquet(parquet_path)

    # Filter by N and omega
    mask = (np.isclose(df["N"].to_numpy().astype(float), float(N))) & (
        np.isclose(df["omega"].to_numpy().astype(float), float(omega))
    )
    if not np.any(mask):
        return None

    row = df[mask].iloc[0]
    return {
        "a_x": float(row["a_x_opt"]),
        "a_y": float(row["a_y_opt"]),
        "a_z": float(row["a_z_opt"]),
        "a_zz": float(row["a_zz_opt"]),
    }


def evaluate_protocols_at_params(
    N: int,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray] | None = None,
    psi0: np.ndarray | None = None,
    projectors: list[np.ndarray] | None = None,
) -> dict[str, float]:
    """Evaluate all three S-only protocols at given drive parameters.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        a_x: J_x^A drive coefficient.
        a_y: J_y^A drive coefficient.
        a_z: J_z^A drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Pre-computed operators (built here if None).
        psi0: Initial state (built here if None).
        projectors: Pre-computed J_z projectors (built here if None).

    Returns:
        Dict with keys:
            'delta_omega_lin', 'delta_omega_parity', 'delta_omega_cfi',
            'parity_expectation', 'sql'.
    """
    if ops is None:
        ops = build_n_particle_operators(N)
    if psi0 is None:
        psi0 = n_particle_initial_state(N)
    if projectors is None:
        projectors = compute_jz_projectors(N, ops)

    sql = sql_reference(N)

    # Linear
    dw_lin = compute_n_particle_sensitivity(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
        meas_op=ops["Jz_S"],
    )

    # Parity (even N only)
    dw_par = float("nan")
    par_exp = float("nan")
    if N % 2 == 0:
        dw_par, par_exp = compute_parity_sensitivity(
            N,
            psi0,
            T_BS,
            T_HOLD,
            omega,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
        )

    # CFI
    dw_cfi = compute_cfi_sensitivity(
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
        projectors=projectors,
    )

    return {
        "delta_omega_lin": dw_lin,
        "delta_omega_parity": dw_par,
        "delta_omega_cfi": dw_cfi,
        "parity_expectation": par_exp,
        "sql": sql,
    }


def _compute_ratio(sql: float, delta_omega: float) -> float:
    """Compute SQL / Δω ratio, returning inf for invalid inputs."""
    if np.isfinite(delta_omega) and delta_omega > 0:
        return sql / delta_omega
    return float("inf")


def _evaluate_stage_a_point(
    N: int,
    omega: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    projectors: list[np.ndarray],
    params: dict[str, float] | None,
) -> NonLinearResult | None:
    """Evaluate all three protocols at a single (N, ω) point.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        ops: Pre-computed operators.
        psi0: Initial state.
        projectors: Pre-computed J_z projectors.
        params: Drive parameters dict with keys 'a_x', 'a_y', 'a_z', 'a_zz'.

    Returns:
        NonLinearResult, or None if params are missing.
    """
    if params is None:
        print(f"[skip] No joint-opt params for N={N}, omega={omega}")
        return None

    evals = evaluate_protocols_at_params(
        N,
        omega,
        params["a_x"],
        params["a_y"],
        params["a_z"],
        params["a_zz"],
        ops=ops,
        psi0=psi0,
        projectors=projectors,
    )

    sql = evals["sql"]
    dw_lin = evals["delta_omega_lin"]
    dw_par = evals["delta_omega_parity"]
    dw_cfi = evals["delta_omega_cfi"]

    res = NonLinearResult(
        N=N,
        omega=omega,
        a_x=params["a_x"],
        a_y=params["a_y"],
        a_z=params["a_z"],
        a_zz=params["a_zz"],
        delta_omega_lin=dw_lin,
        delta_omega_parity=dw_par,
        delta_omega_cfi=dw_cfi,
        ratio_lin=_compute_ratio(sql, dw_lin),
        ratio_parity=_compute_ratio(sql, dw_par),
        ratio_cfi=_compute_ratio(sql, dw_cfi),
        parity_expectation=evals["parity_expectation"],
        sql=sql,
        stage="A",
        success=True,
    )
    print(
        f"  [A] N={N}, omega={omega}: lin={dw_lin:.6e}, par={dw_par:.6e}, cfi={dw_cfi:.6e}"
    )
    return res


def _run_stage_a_loop(
    N_values: list[int], omega_values: list[float]
) -> list[NonLinearResult]:
    """Nested loop over N and omega evaluating all three protocols."""
    results: list[NonLinearResult] = []
    for N in N_values:
        ops = build_n_particle_operators(N)
        psi0 = n_particle_initial_state(N)
        projectors = compute_jz_projectors(N, ops)
        for omega in omega_values:
            params = load_joint_optimal_params(N, omega)
            res = _evaluate_stage_a_point(N, omega, ops, psi0, projectors, params)
            if res is not None:
                results.append(res)
    return results


def generate_stage_a_scan(
    omega_values: list[float] | None = None,
    N_values: list[int] | None = None,
    force: bool = False,
) -> NonLinearScanResult:
    """Run Stage A: evaluate all three protocols at #20260613 joint-optimal params.

    Args:
        omega_values: List of ω values (default: all OMEGA_VALS).
        N_values: List of N values (default: STAGE_A_N_VALS).
        force: If True, re-run even if Parquet exists.

    Returns:
        NonLinearScanResult with all evaluated points.
    """
    _omega_values = OMEGA_VALS if omega_values is None else omega_values
    _N_values = STAGE_A_N_VALS if N_values is None else N_values

    out_path = _parquet_path("stage-a-scan")
    cached = _try_load_scan_cache(out_path, force)
    if cached is not None:
        return cached

    results = _run_stage_a_loop(_N_values, _omega_values)

    scan_result = NonLinearScanResult(results=results)
    scan_result.save_parquet(out_path)
    print(f"[save] Stage A scan saved to {out_path}")
    return scan_result


# ============================================================================
# Stage B: Re-optimised per Protocol
# ============================================================================


def _sensitivity_objective(
    params: np.ndarray,
    N: int,
    omega_true: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    protocol: str,
    projectors: list[np.ndarray] | None = None,
    fd_step: float = FD_STEP,
) -> float:
    """Objective function for NM optimisation: minimise Δω.

    Args:
        params: 4-element array [a_x, a_y, a_z, a_zz].
        All other args: Passed through to compute_protocol_sensitivity.

    Returns:
        Δω value (positive float, possibly inf).
    """
    a_x = float(params[0])
    a_y = float(params[1])
    a_z = float(params[2])
    a_zz = float(params[3])

    # Check bounds and apply quadratic penalty
    lo, hi = DRIVE_BOUNDS
    penalty = 0.0
    for val in params:
        if val < lo:
            penalty += (lo - val) ** 2
        elif val > hi:
            penalty += (val - hi) ** 2

    dw, _ = compute_protocol_sensitivity(
        protocol,
        N,
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
        projectors=projectors,
        fd_step=fd_step,
    )

    if not np.isfinite(dw):
        return 1e10  # Large penalty for singular points

    return float(dw) + 1e3 * penalty


def non_linear_random_search(
    N: int,
    omega: float,
    protocol: str,
    n_samples: int = N_RANDOM,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Random search over the 4D parameter space for a given protocol.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        protocol: One of 'linear', 'parity', 'cfi'.
        n_samples: Number of random points.
        bounds: (min, max) for all drive parameters.
        seed: Random seed.

    Returns:
        Tuple (samples, delta_omega_values, best_params).
    """
    rng = np.random.default_rng(seed)
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)
    projectors = compute_jz_projectors(N, ops)
    lo, hi = bounds

    samples = rng.uniform(lo, hi, size=(n_samples, 4))
    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        ax = float(samples[i, 0])
        ay = float(samples[i, 1])
        az = float(samples[i, 2])
        azz = float(samples[i, 3])
        dw, _ = compute_protocol_sensitivity(
            protocol,
            N,
            psi0,
            T_BS,
            T_HOLD,
            omega,
            ax,
            ay,
            az,
            azz,
            ops,
            projectors=projectors,
        )
        deltas[i] = dw

    best_idx = int(np.argmin(deltas))
    best_params = (
        float(samples[best_idx, 0]),
        float(samples[best_idx, 1]),
        float(samples[best_idx, 2]),
        float(samples[best_idx, 3]),
    )

    return samples, deltas, best_params


def run_non_linear_nelder_mead(
    N: int,
    omega_true: float,
    x0: tuple[float, float, float, float],
    protocol: str,
    ops: dict[str, np.ndarray] | None = None,
    psi0: np.ndarray | None = None,
    projectors: list[np.ndarray] | None = None,
    maxiter: int = NM_MAXITER,
    fd_step: float = FD_STEP,
) -> dict:
    """Run Nelder-Mead optimisation for a given protocol.

    Args:
        N: Number of system particles.
        omega_true: True phase rate.
        x0: Initial 4D parameter vector (a_x, a_y, a_z, a_zz).
        protocol: One of 'linear', 'parity', 'cfi'.
        ops: Pre-computed operators.
        psi0: Initial state.
        projectors: Pre-computed J_z projectors.
        maxiter: Maximum NM iterations.
        fd_step: Finite-difference step.

    Returns:
        Dict with 'x_opt', 'fun_opt', 'success', 'nfev'.
    """
    if ops is None:
        ops = build_n_particle_operators(N)
    if psi0 is None:
        psi0 = n_particle_initial_state(N)
    if projectors is None:
        projectors = compute_jz_projectors(N, ops)

    result = minimize(
        _sensitivity_objective,
        x0=np.array(x0, dtype=float),
        args=(N, omega_true, ops, psi0, protocol, projectors, fd_step),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-8, "adaptive": True},
    )

    return {
        "x_opt": (
            float(result.x[0]),
            float(result.x[1]),
            float(result.x[2]),
            float(result.x[3]),
        ),
        "fun_opt": float(result.fun),
        "success": bool(result.success),
        "nfev": int(result.nfev),
    }


def run_single_non_linear_n_omega(
    N: int,
    omega: float,
    protocol: str,
    n_random: int = N_RANDOM,
    n_nm_refine: int = N_NM_REFINE,
    nm_maxiter: int = NM_MAXITER,
    seed: int | None = 42,
) -> NonLinearResult:
    """Full optimisation pipeline for a single (N, ω, protocol) triple.

    1. 4D random search over all drive parameters.
    2. Take top n_nm_refine points and refine each with Nelder-Mead.
    3. Return the best result found.

    Args:
        N: Number of system particles.
        omega: Phase rate value.
        protocol: One of 'linear', 'parity', 'cfi'.
        n_random: Number of random search points.
        n_nm_refine: Number of NM refinements from top points.
        nm_maxiter: Maximum iterations per NM run.
        seed: Random seed.

    Returns:
        NonLinearResult with the best result found.
    """
    ops = build_n_particle_operators(N)
    psi0 = n_particle_initial_state(N)
    projectors = compute_jz_projectors(N, ops)
    sql = sql_reference(N)

    # Stage 1: Random search
    samples, deltas, _best_params_rs = non_linear_random_search(
        N,
        omega,
        protocol,
        n_samples=n_random,
        seed=seed,
    )

    # Stage 2: NM refinement from top points
    top_k = min(n_nm_refine, len(deltas))
    top_indices = np.argsort(deltas)[:top_k]

    best_dw = float(deltas[top_indices[0]])
    best_a = (
        float(samples[top_indices[0], 0]),
        float(samples[top_indices[0], 1]),
        float(samples[top_indices[0], 2]),
        float(samples[top_indices[0], 3]),
    )
    best_success = False
    best_nfev = 0

    for idx in top_indices:
        x0 = (
            float(samples[idx, 0]),
            float(samples[idx, 1]),
            float(samples[idx, 2]),
            float(samples[idx, 3]),
        )
        nm_result = run_non_linear_nelder_mead(
            N,
            omega,
            x0,
            protocol,
            ops=ops,
            psi0=psi0,
            projectors=projectors,
            maxiter=nm_maxiter,
        )
        if nm_result["fun_opt"] < best_dw:
            best_dw = nm_result["fun_opt"]
            best_a = nm_result["x_opt"]
            best_success = nm_result["success"]
            best_nfev = nm_result["nfev"]

    # Evaluate all three protocols at the best params found
    evals = evaluate_protocols_at_params(
        N,
        omega,
        best_a[0],
        best_a[1],
        best_a[2],
        best_a[3],
        ops=ops,
        psi0=psi0,
        projectors=projectors,
    )

    dw_lin = evals["delta_omega_lin"]
    dw_par = evals["delta_omega_parity"]
    dw_cfi = evals["delta_omega_cfi"]

    return NonLinearResult(
        N=N,
        omega=omega,
        a_x=best_a[0],
        a_y=best_a[1],
        a_z=best_a[2],
        a_zz=best_a[3],
        delta_omega_lin=dw_lin,
        delta_omega_parity=dw_par,
        delta_omega_cfi=dw_cfi,
        ratio_lin=sql / dw_lin if np.isfinite(dw_lin) and dw_lin > 0 else float("inf"),
        ratio_parity=sql / dw_par
        if np.isfinite(dw_par) and dw_par > 0
        else float("inf"),
        ratio_cfi=sql / dw_cfi if np.isfinite(dw_cfi) and dw_cfi > 0 else float("inf"),
        parity_expectation=evals["parity_expectation"],
        sql=sql,
        stage="B",
        success=best_success,
        nfev=best_nfev,
    )


def _generate_stage_b_protocol(
    proto: str,
    omega_values: list[float],
    N_values: list[int],
    force: bool,
) -> list[NonLinearResult]:
    """Run Stage B optimisation for a single protocol.

    Args:
        proto: Protocol identifier ('linear', 'parity', 'cfi').
        omega_values: List of ω values.
        N_values: List of N values.
        force: If True, re-run even if Parquet exists.

    Returns:
        List of NonLinearResult for this protocol.
    """
    suffix = f"stage-b-{proto}"
    out_path = _parquet_path(suffix)

    cached = _try_load_scan_cache(out_path, force)
    if cached is not None:
        return cached.results

    results: list[NonLinearResult] = []
    for N in N_values:
        if _skip_parity_odd(proto, N):
            print(f"  [skip] Parity not Hermitian for odd N={N}")
            continue

        for o in omega_values:
            print(f"  [B] Protocol={proto}, N={N}, omega={o}: running...")
            res = run_single_non_linear_n_omega(N, o, proto)
            results.append(res)
            print(
                f"    -> Delta omega = {res.delta_omega_lin:.6e} (lin), "
                f"{res.delta_omega_parity:.6e} (par), "
                f"{res.delta_omega_cfi:.6e} (cfi)"
            )

    scan_result = NonLinearScanResult(results=results)
    scan_result.save_parquet(out_path)
    print(f"[save] Stage B ({proto}) scan saved to {out_path}")
    return results


def generate_stage_b_scan(
    omega_values: list[float] | None = None,
    N_values: list[int] | None = None,
    protocol: str | None = None,
    force: bool = False,
) -> NonLinearScanResult:
    """Run Stage B: re-optimise per protocol.

    Args:
        omega_values: List of ω values (default: [0.2, 1.0]).
        N_values: List of N values (default: STAGE_B_N_VALS).
        protocol: Which protocol to optimise. If None, do all three.
        force: If True, re-run even if Parquet exists.

    Returns:
        NonLinearScanResult with all evaluated points.
    """
    if omega_values is None:
        omega_values = [0.2, 1.0]
    if N_values is None:
        N_values = STAGE_B_N_VALS
    if protocol is None:
        protocols = [PROTOCOL_LINEAR, PROTOCOL_PARITY, PROTOCOL_CFI]
    else:
        protocols = [protocol]

    all_results: list[NonLinearResult] = []
    for proto in protocols:
        all_results.extend(
            _generate_stage_b_protocol(proto, omega_values, N_values, force)
        )

    return NonLinearScanResult(results=all_results)


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class NonLinearResult(ParquetSerializable):
    """Result from evaluating all three S-only protocols at a single (N, ω) point.

    Attributes:
        N: Number of system particles.
        omega: Phase rate value.
        a_x: J_x^A drive coefficient used.
        a_y: J_y^A drive coefficient used.
        a_z: J_z^A drive coefficient used.
        a_zz: Ising interaction coefficient used.
        delta_omega_lin: Sensitivity via linear J_z^S measurement.
        delta_omega_parity: Sensitivity via parity Π_S (NaN for odd N).
        delta_omega_cfi: Sensitivity via full-distribution CFI.
        ratio_lin: SQL / delta_omega_lin.
        ratio_parity: SQL / delta_omega_parity (NaN for odd N).
        ratio_cfi: SQL / delta_omega_cfi.
        parity_expectation: ⟨Π_S⟩ at the operating point.
        sql: Standard Quantum Limit = 1/(√N × T_HOLD).
        t_hold: Holding time.
        fd_step: Finite-difference step size.
        success: Whether optimisation succeeded.
        nfev: Number of function evaluations.
        stage: 'A' (fixed params) or 'B' (re-optimised).
    """

    N: int
    omega: float
    a_x: float
    a_y: float
    a_z: float
    a_zz: float
    delta_omega_lin: float
    delta_omega_parity: float
    delta_omega_cfi: float
    ratio_lin: float
    ratio_parity: float
    ratio_cfi: float
    parity_expectation: float
    sql: float
    t_hold: float = T_HOLD
    fd_step: float = FD_STEP
    success: bool = True
    nfev: int = 0
    stage: str = "A"

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "N",
        "omega",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "delta_omega_lin",
        "delta_omega_parity",
        "delta_omega_cfi",
        "ratio_lin",
        "ratio_parity",
        "ratio_cfi",
        "parity_expectation",
        "sql",
        "t_hold",
        "fd_step",
        "success",
        "nfev",
        "stage",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "N": [self.N],
                "omega": [self.omega],
                "a_x": [self.a_x],
                "a_y": [self.a_y],
                "a_z": [self.a_z],
                "a_zz": [self.a_zz],
                "delta_omega_lin": [self.delta_omega_lin],
                "delta_omega_parity": [self.delta_omega_parity],
                "delta_omega_cfi": [self.delta_omega_cfi],
                "ratio_lin": [self.ratio_lin],
                "ratio_parity": [self.ratio_parity],
                "ratio_cfi": [self.ratio_cfi],
                "parity_expectation": [self.parity_expectation],
                "sql": [self.sql],
                "t_hold": [self.t_hold],
                "fd_step": [self.fd_step],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "stage": [self.stage],
            }
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> NonLinearResult:
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        row = df.iloc[0]
        return cls(
            N=int(row["N"]),
            omega=float(row["omega"]),
            a_x=float(row["a_x"]),
            a_y=float(row["a_y"]),
            a_z=float(row["a_z"]),
            a_zz=float(row["a_zz"]),
            delta_omega_lin=float(row["delta_omega_lin"]),
            delta_omega_parity=float(row["delta_omega_parity"]),
            delta_omega_cfi=float(row["delta_omega_cfi"]),
            ratio_lin=float(row["ratio_lin"]),
            ratio_parity=float(row["ratio_parity"]),
            ratio_cfi=float(row["ratio_cfi"]),
            parity_expectation=float(row["parity_expectation"]),
            sql=float(row["sql"]),
            t_hold=float(row["t_hold"]),
            fd_step=float(row["fd_step"]),
            success=bool(int(row["success"])),
            nfev=int(row["nfev"]),
            stage=str(row["stage"]),
        )


@dataclass
class NonLinearScanResult(ParquetSerializable):
    """Collection of NonLinearResult across a grid."""

    _PARQUET_COLUMNS: ClassVar[list[str]] = NonLinearResult._PARQUET_COLUMNS

    results: list[NonLinearResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=NonLinearResult._PARQUET_COLUMNS)
        return pd.concat([r.to_dataframe() for r in self.results], ignore_index=True)

    @classmethod
    def from_parquet(cls, path: str | Path) -> NonLinearScanResult:
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        results: list[NonLinearResult] = []
        for _, row in df.iterrows():
            results.append(
                NonLinearResult(
                    N=int(row["N"]),
                    omega=float(row["omega"]),
                    a_x=float(row["a_x"]),
                    a_y=float(row["a_y"]),
                    a_z=float(row["a_z"]),
                    a_zz=float(row["a_zz"]),
                    delta_omega_lin=float(row["delta_omega_lin"]),
                    delta_omega_parity=float(row["delta_omega_parity"]),
                    delta_omega_cfi=float(row["delta_omega_cfi"]),
                    ratio_lin=float(row["ratio_lin"]),
                    ratio_parity=float(row["ratio_parity"]),
                    ratio_cfi=float(row["ratio_cfi"]),
                    parity_expectation=float(row["parity_expectation"]),
                    sql=float(row["sql"]),
                    t_hold=float(row["t_hold"]),
                    fd_step=float(row["fd_step"]),
                    success=bool(int(row["success"])),
                    nfev=int(row["nfev"]),
                    stage=str(row["stage"]),
                ),
            )
        return cls(results=results)


# ============================================================================
# Plotting
# ============================================================================


def _get_df_from_result(result: NonLinearScanResult) -> pd.DataFrame:
    """Convert scan result to a plottable DataFrame."""
    return result.to_dataframe()


def plot_protocol_comparison(
    result: NonLinearScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (12, 8),
) -> Path:
    """Plot all three protocol sensitivities vs N, coloured by ω.

    One panel for each protocol. Shows SQL and HL reference lines.

    Args:
        result: Scan result with protocol data.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = result.to_dataframe()
    if df.empty:
        print("[skip] No data to plot.")
        return save_path

    protocols = [
        ("delta_omega_lin", r"$\Delta\omega_{\mathrm{lin}}$", "Linear $J_z^S$"),
        ("delta_omega_parity", "$\\Delta\\omega_{\\Pi}$", "Parity $\\Pi_S$"),
        ("delta_omega_cfi", r"$\Delta\omega_{\mathrm{CFI}}$", "CFI $F_C^{(S)}$"),
    ]

    omega_values = sorted(df["omega"].unique())
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_values)))

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, (col, ylabel, title) in zip(axes, protocols, strict=False):
        for omega_val, colour in zip(omega_values, colours, strict=False):
            sub = df[np.isclose(df["omega"], omega_val)]
            sub = sub.sort_values("N")
            vals = sub[col].to_numpy()
            valid = np.isfinite(vals) & (vals > 0)
            if np.any(valid):
                ax.loglog(
                    sub["N"].to_numpy()[valid],
                    vals[valid],
                    "o-",
                    color=colour,
                    label=rf"$\omega={omega_val:.1f}$",
                    markersize=5,
                    linewidth=1.2,
                )

        N_range = np.linspace(1, 20, 100)
        t_hold = T_HOLD
        sql_line = 1.0 / (np.sqrt(N_range) * t_hold)
        ax.loglog(N_range, sql_line, "k--", alpha=0.7, linewidth=1.2, label="SQL")
        hl_line = 1.0 / (N_range * t_hold)
        ax.loglog(N_range, hl_line, "k:", alpha=0.5, linewidth=1.0, label="HL")
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, title=r"$\omega$")
        ax.set_xlim(left=0.5)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_ratio_comparison(
    result: NonLinearScanResult,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 6),
) -> Path:
    """Plot SQL-violation ratio R vs N for all three protocols at a single ω.

    Args:
        result: Scan result with protocol data.
        save_path: Output SVG path.
        figsize: Figure size.

    Returns:
        Path to saved SVG.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = result.to_dataframe()
    if df.empty:
        print("[skip] No data to plot.")
        return save_path

    protocols = [
        ("ratio_lin", r"$R_{\mathrm{lin}}$", "solid"),
        ("ratio_parity", r"$R_{\Pi}$", "dashed"),
        ("ratio_cfi", r"$R_{\mathrm{CFI}}$", "dotted"),
    ]

    omega_values = sorted(df["omega"].unique())
    colours = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(omega_values)))

    fig, ax = plt.subplots(figsize=figsize)

    for omega_val, colour in zip(omega_values, colours, strict=False):
        sub = df[np.isclose(df["omega"], omega_val)]
        sub = sub.sort_values("N")
        for col_name, label, ls in protocols:
            vals = sub[col_name].to_numpy()
            valid = np.isfinite(vals) & (vals > 0)
            if np.any(valid):
                ax.plot(
                    sub["N"].to_numpy()[valid],
                    vals[valid],
                    "o-",
                    color=colour,
                    linestyle=ls,
                    label=rf"$\omega={omega_val:.1f}$, {label}",
                    markersize=4,
                    linewidth=1.0,
                )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="SQL (R=1)")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$R(N) = \Delta\omega_{\mathrm{SQL}} / \Delta\omega$")
    ax.set_title("SQL-violation ratio comparison")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(left=0.5)
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# CLI Entry Point
# ============================================================================


# ── Dispatch targets for CLI ────────────────────────────────────────────


def _run_decoupled_baseline(*, force: bool = False) -> None:
    """Verify decoupled baseline (no plotting output)."""
    verify_decoupled_baseline()


def _run_stage_a(*, force: bool = False) -> None:
    """Run Stage A and save results (no plotting output)."""
    generate_stage_a_scan(force=force)


def _run_stage_b_linear(*, force: bool = False) -> None:
    """Re-optimise for linear measurement and generate comparison plot."""
    result = generate_stage_b_scan(protocol=PROTOCOL_LINEAR, force=force)
    if len(result.results) > 0:
        plot_protocol_comparison(result, _fig_path("stage-b-linear-comparison"))


def _run_stage_b_parity(*, force: bool = False) -> None:
    """Re-optimise for parity measurement and generate comparison plot."""
    result = generate_stage_b_scan(protocol=PROTOCOL_PARITY, force=force)
    if len(result.results) > 0:
        plot_protocol_comparison(result, _fig_path("stage-b-parity-comparison"))


def _run_stage_b_cfi(*, force: bool = False) -> None:
    """Re-optimise for CFI measurement and generate comparison plot."""
    result = generate_stage_b_scan(protocol=PROTOCOL_CFI, force=force)
    if len(result.results) > 0:
        plot_protocol_comparison(result, _fig_path("stage-b-cfi-comparison"))


def _run_stage_b_all(*, force: bool = False) -> None:
    """Re-optimise for all protocols and generate comparison + ratio plots."""
    result = generate_stage_b_scan(force=force)
    if len(result.results) > 0:
        plot_protocol_comparison(result, _fig_path("stage-b-all-comparison"))
        plot_ratio_comparison(result, _fig_path("stage-b-all-ratios"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-Linear Measurement (Parity and CFI) on omega-Modulated Drive",
    )
    parser.add_argument("--force", action="store_true", help="Re-run all scans")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=[
            "decoupled-baseline",
            "stage-a",
            "stage-b-linear",
            "stage-b-parity",
            "stage-b-cfi",
            "stage-b-all",
        ],
        help="Run only a specific generator",
    )
    args = parser.parse_args()

    (REPORTS_DIR / REPORT_DATE / "raw_data").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / REPORT_DATE / "figures").mkdir(parents=True, exist_ok=True)

    dispatch: dict[str, tuple[str, Callable[..., None]]] = {
        "decoupled-baseline": ("Decoupled Baseline", _run_decoupled_baseline),
        "stage-a": ("Stage A (Fixed-Parameter)", _run_stage_a),
        "stage-b-linear": ("Stage B (Linear)", _run_stage_b_linear),
        "stage-b-parity": ("Stage B (Parity)", _run_stage_b_parity),
        "stage-b-cfi": ("Stage B (CFI)", _run_stage_b_cfi),
        "stage-b-all": ("Stage B (All)", _run_stage_b_all),
    }

    keys = [args.only] if args.only else list(dispatch.keys())

    for key in keys:
        title, func = dispatch[key]
        print(f"\n{'=' * 60}")
        print(f"Running: {title}")
        print(f"{'=' * 60}")
        func(force=args.force)


if __name__ == "__main__":
    main()

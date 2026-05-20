"""
Ancilla-Drive Phase-Modulated Metrology: Beating the SQL by Exposing the
Ancilla Drive to the Unknown Phase.

Implements the driven-ancilla metrology protocol described in
``reports/2026-05-19-Ancilla-Drive-Phase-Modulated-Metrology.md``.

Physical Model:
- Two qubits (system S + ancilla A), each a spin-1/2 (single-particle subspace).
- Basis: {|00⟩, |01⟩, |10⟩, |11⟩} where |0⟩ = |1,0⟩ (particle in mode 0).
- Circuit: BS_S → Hold → BS_S, where BS_S acts only on the system qubit.
- Hold Hamiltonian:
    H = θ J_z^S + H_A + H_int
    H_A = θ (a_x J_x^A + a_y J_y^A + a_z J_z^A)   **θ-modulated** ancilla drive
    H_int = a_zz J_z^S ⊗ J_z^A                     (Ising interaction)
- Initial state: |00⟩ (both qubits in |1,0⟩).
- Measurement: J_z^S on the system qubit.
- Sensitivity: Δθ via error propagation (central finite differences).

Key difference from prior work (2026-05-18, fixed-drive protocol):
    In the fixed-drive protocol, H_A = a_x J_x^A + a_y J_y^A + a_z J_z^A was
    independent of θ. Here H_A = θ (a_x J_x^A + a_y J_y^A + a_z J_z^A) is
    modulated by θ itself, giving ∂H/∂θ = J_z^S + H_A^norm, an extra channel
    for θ-information to flow through the ancilla.

Units:
- Dimensionless throughout. θ is the unknown phase rate.
- T_H: holding-time strength (dimensionless).
- a_x, a_y, a_z, a_zz: real coefficients.

References:
- Report 2026-05-19-Ancilla-Drive-Phase-Modulated-Metrology.md
- Report ``reports/2026-05-18/2026-05-18-Ancilla-Drive-Enhanced-Metrology.md`` (fixed-drive precursor)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

# Reuse shared primitives from ancilla_optimization
from src.analysis.ancilla_optimization import (
    I_2,
    bs_unitary,
    build_two_qubit_operators,
    compute_expectation_and_variance,
)

# Reuse result dataclasses from the fixed-drive protocol (identical structure)
from src.analysis.ancilla_drive_metrology import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveNelderMeadResult,
    DriveRandomSearchResult,
    DriveThetaScanResult,
)

I_4 = np.eye(4, dtype=complex)


# ============================================================================
# Operator Construction
# ============================================================================


def system_only_bs_unitary(T: float) -> np.ndarray:
    """Single-qubit beam-splitter on the system, identity on the ancilla.

    U = U_BS(T) ⊗ I_2 = exp(-i T J_x^S) ⊗ I_2

    A 50/50 beam splitter corresponds to T = π/2.

    Args:
        T: Beam-splitter duration.

    Returns:
        4×4 unitary matrix.
    """
    U_sys = bs_unitary(T)
    U = np.kron(U_sys, I_2)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"System-only BS unitary not unitary for T={T}"
    )
    return U


def build_phase_modulated_drive_hamiltonian(
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the θ-modulated ancilla drive Hamiltonian.

    H_A = θ (a_x J_x^A + a_y J_y^A + a_z J_z^A)

    The critical difference from the fixed-drive protocol is the leading θ
    factor: the ancilla drive scales with the unknown phase, creating a
    parametric amplification effect in ∂⟨J_z^S⟩/∂θ.

    Args:
        theta: Unknown phase rate parameter (scales the whole drive).
        a_x: Coefficient for J_x^A.
        a_y: Coefficient for J_y^A.
        a_z: Coefficient for J_z^A.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian matrix representing the θ-modulated ancilla drive.
    """
    H = np.zeros((4, 4), dtype=complex)
    if a_x != 0.0:
        H += a_x * ops["Jx_A"]
    if a_y != 0.0:
        H += a_y * ops["Jy_A"]
    if a_z != 0.0:
        H += a_z * ops["Jz_A"]
    H = theta * H  # θ-modulation: entire drive scales with the unknown phase
    return 0.5 * (H + H.conj().T)


def build_iszz_interaction(
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
        # J_z^S ⊗ J_z^A = (J_z ⊗ I_2) @ (I_2 ⊗ J_z)
        H += a_zz * (ops["Jz_S"] @ ops["Jz_A"])
    return H


def build_phase_modulated_hold_hamiltonian(
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the total holding Hamiltonian with θ-modulated ancilla drive.

    H = θ J_z^S + H_A + H_int
      = θ J_z^S + θ (a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A
      = θ [J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A] + a_zz J_z^S ⊗ J_z^A

    The θ factor on the drive terms means ∂H/∂θ = J_z^S + a_x J_x^A + a_y J_y^A
    + a_z J_z^A, which includes ancilla operators. This extra contribution to
    the derivative is the key mechanism for potential SQL violation.

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
    H += build_phase_modulated_drive_hamiltonian(theta, a_x, a_y, a_z, ops)
    H += build_iszz_interaction(a_zz, ops)
    return 0.5 * (H + H.conj().T)


def phase_modulated_hold_unitary(
    T_H: float,
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Holding-time unitary for the θ-modulated ancilla protocol.

    U_hold(T_H) = exp(-i T_H H)
    where H = θ J_z^S + θ(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S ⊗ J_z^A.

    Args:
        T_H: Holding-time strength.
        theta: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 unitary matrix.
    """
    H = build_phase_modulated_hold_hamiltonian(theta, a_x, a_y, a_z, a_zz, ops)
    U = expm(-1j * T_H * H)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
        f"Phase-modulated hold unitary not unitary for T_H={T_H}, θ={theta}"
    )
    return U


def evolve_phase_modulated_circuit(
    psi0: np.ndarray,
    T_BS: float,
    T_H: float,
    theta: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full θ-modulated ancilla MZI circuit.

    |ψ_final⟩ = U_BS_S · U_hold(T_H) · U_BS_S · |ψ₀⟩

    The hold unitary uses the θ-modulated H_A = θ (a_x J_x^A + ...).

    Args:
        psi0: Initial 4-vector (must be normalised).
        T_BS: Beam-splitter duration (both BS identical).
        T_H: Holding-time strength.
        theta: Phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    assert np.isclose(np.linalg.norm(psi0), 1.0), "Initial state must be normalised"

    U_bs = system_only_bs_unitary(T_BS)
    psi = U_bs @ psi0
    psi = phase_modulated_hold_unitary(T_H, theta, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_bs @ psi

    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


def compute_phase_modulated_sensitivity(
    psi0: np.ndarray,
    T_BS: float,
    T_H: float,
    theta_true: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
    meas_op: np.ndarray | None = None,
) -> float:
    """Compute the error-propagation sensitivity Δθ.

    Δθ = sqrt(Var(O)) / |∂⟨O⟩/∂θ|

    where O is the measurement operator (default: J_z^S).

    IMPORTANT: Because θ now appears in both H_S (= θ J_z^S) and H_A
    (= θ (a_x J_x^A + a_y J_y^A + a_z J_z^A)), the central finite-difference
    step captures the FULL θ-dependence (both channels) automatically —
    the circuit is re-evaluated at θ ± δ, and both H_S and H_A change.

    Args:
        psi0: Initial 4-vector (product state).
        T_BS: Beam-splitter duration.
        T_H: Holding-time strength.
        theta_true: True phase rate parameter.
        a_x: Ancilla J_x drive coefficient.
        a_y: Ancilla J_y drive coefficient.
        a_z: Ancilla J_z drive coefficient.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators (must contain 'Jz_S').
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator. Defaults to ops['Jz_S'] (S-only).

    Returns:
        Sensitivity Δθ (positive float). Returns inf if derivative is zero
        (fringe extremum).
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    # Evaluate at theta_true
    psi = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    _, var = compute_expectation_and_variance(psi, meas_op)

    # Central finite difference for ∂⟨O⟩/∂θ
    psi_plus = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true + fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    psi_minus = evolve_phase_modulated_circuit(
        psi0,
        T_BS,
        T_H,
        theta_true - fd_step,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
    )
    exp_plus = np.real(psi_plus.conj() @ meas_op @ psi_plus)
    exp_minus = np.real(psi_minus.conj() @ meas_op @ psi_minus)
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


# ============================================================================
# Default configuration
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_T_H: float = 10.0  # Holding time (SQL = 0.1)
DEFAULT_PSI0: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩
SQL_REFERENCE: float = 1.0 / DEFAULT_T_H  # Δθ_SQL = 0.1
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Range for all coefficients


# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_phase_modulated_decoupled_baseline(
    T_H: float = DEFAULT_T_H,
    theta_true: float = 1.0,
) -> DriveDecoupledBaselineResult:
    """Compute the decoupled baseline sensitivity Δθ.

    At (a_x = a_y = a_z = a_zz = 0), the θ-modulated ancilla circuit reduces
    to a standard single-qubit MZI with |1,0⟩ input and 50/50 BS,
    giving Δθ = 1/T_H. The θ factor in H_A is irrelevant when all a_k = 0.

    Args:
        T_H: Holding-time strength.
        theta_true: True phase rate.

    Returns:
        DriveDecoupledBaselineResult.
    """
    ops = build_two_qubit_operators()
    dtheta = compute_phase_modulated_sensitivity(
        DEFAULT_PSI0,
        DEFAULT_T_BS,
        T_H,
        theta_true,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )
    return DriveDecoupledBaselineResult(
        T_H_value=T_H,
        delta_theta=dtheta,
        sql=1.0 / T_H,
    )


# ============================================================================
# 2D Slice Scan
# ============================================================================


def phase_modulated_2d_slice(
    theta: float,
    drive_range: tuple[float, float] = DRIVE_BOUNDS,
    azz_range: tuple[float, float] = DRIVE_BOUNDS,
    n_drive: int = 201,
    n_azz: int = 201,
    slice_type: str = "ax",
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
) -> Drive2DSliceResult:
    """Run a 2D slice scan over (a_drive, a_zz) with θ-modulated ancilla drive.

    For slice_type='ax': varies a_x (with a_y = a_z = 0).
    For slice_type='ay': varies a_y (with a_x = a_z = 0).

    Args:
        theta: Phase rate value.
        drive_range: (min, max) for the drive coefficient.
        azz_range: (min, max) for the interaction coefficient.
        n_drive: Number of drive-coefficient points.
        n_azz: Number of a_zz points.
        slice_type: 'ax' or 'ay'.
        T_H: Holding time (default 10).
        T_BS: Beam-splitter duration (default π/2).

    Returns:
        Drive2DSliceResult with the sensitivity grid.
    """
    if slice_type not in ("ax", "ay"):
        raise ValueError(f"slice_type must be 'ax' or 'ay', got {slice_type}")

    ops = build_two_qubit_operators()
    drive_vals = np.linspace(drive_range[0], drive_range[1], n_drive)
    azz_vals = np.linspace(azz_range[0], azz_range[1], n_azz)
    grid = np.full((n_drive, n_azz), np.inf, dtype=float)

    for i, d_val in enumerate(drive_vals):
        for j, a_val in enumerate(azz_vals):
            if slice_type == "ax":
                ax, ay, az = d_val, 0.0, 0.0
            else:
                ax, ay, az = 0.0, d_val, 0.0

            dtheta = compute_phase_modulated_sensitivity(
                DEFAULT_PSI0,
                T_BS,
                T_H,
                theta,
                ax,
                ay,
                az,
                a_val,
                ops,
            )
            grid[i, j] = dtheta

    return Drive2DSliceResult(
        drive_values=drive_vals,
        azz_values=azz_vals,
        delta_theta_grid=grid,
        theta_value=theta,
        slice_type=slice_type,
        sql=1.0 / T_H,
    )


# ============================================================================
# 4D Random Search
# ============================================================================


def phase_modulated_random_search(
    theta: float,
    n_samples: int = 500,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    seed: int | None = 42,
) -> DriveRandomSearchResult:
    """Random search over the 4D parameter space (a_x, a_y, a_z, a_zz).

    Uses the θ-modulated ancilla drive H_A = θ (a_x J_x^A + ...).

    Args:
        theta: Phase rate value.
        n_samples: Number of random points to evaluate.
        bounds: (min, max) for all four coefficients.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        seed: Random seed for reproducibility.

    Returns:
        DriveRandomSearchResult with all samples and best found.
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

        dtheta = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            T_BS,
            T_H,
            theta,
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

    return DriveRandomSearchResult(
        samples=samples,
        delta_theta_values=deltas,
        best_params=best_params,
        best_delta_theta=float(deltas[best_idx]),
        theta_value=theta,
        sql=1.0 / T_H,
        T_H=T_H,
    )


# ============================================================================
# Nelder--Mead Optimisation
# ============================================================================


def phase_modulated_sensitivity_objective(
    params: np.ndarray,
    theta_true: float,
    ops: dict[str, np.ndarray],
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    fd_step: float = 1e-6,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    penalty_scale: float = 1e6,
) -> float:
    """Objective function for minimising Δθ in the θ-modulated protocol.

    Fixed configuration: |00⟩ initial state, fixed T_BS, fixed T_H.
    params = [a_x, a_y, a_z, a_zz] (4 elements).

    Args:
        params: 4-element parameter vector.
        theta_true: True phase rate.
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

    return compute_phase_modulated_sensitivity(
        DEFAULT_PSI0,
        T_BS,
        T_H,
        theta_true,
        ax,
        ay,
        az,
        azz,
        ops,
        fd_step,
    )


def run_phase_modulated_nelder_mead(
    theta_true: float,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
    track_history: bool = False,
) -> DriveNelderMeadResult:
    """Run Nelder--Mead optimisation for the θ-modulated ancilla protocol.

    Args:
        theta_true: True phase rate parameter.
        x0: Initial 4-parameter vector [ax, ay, az, azz]. Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder--Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder--Mead parameters.
        bounds: (min, max) for all four parameters.
        T_H: Holding time.
        T_BS: Beam-splitter duration.
        track_history: If True, record objective values per iteration.

    Returns:
        DriveNelderMeadResult.
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
        return phase_modulated_sensitivity_objective(
            p,
            theta_true,
            ops,
            T_H=T_H,
            T_BS=T_BS,
            bounds=bounds,
        )

    history: list[float] = []

    def callback(_x: np.ndarray) -> None:
        if track_history:
            val = objective(_x)
            history.append(val)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback if track_history else None,
        options={
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Compute diagnostics at the optimal point
    psi_final = evolve_phase_modulated_circuit(
        DEFAULT_PSI0,
        T_BS,
        T_H,
        theta_true,
        float(opt_params[0]),
        float(opt_params[1]),
        float(opt_params[2]),
        float(opt_params[3]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])

    return DriveNelderMeadResult(
        delta_theta_opt=float(result.fun),
        params_opt=opt_params,
        theta_true=theta_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        history=history.copy(),
    )


# ============================================================================
# θ Scan with Random Search + Nelder--Mead Refinement
# ============================================================================


def run_phase_modulated_theta_scan(
    theta_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 50,
    seed: int | None = 42,
    maxiter: int = 5000,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    T_H: float = DEFAULT_T_H,
    T_BS: float = DEFAULT_T_BS,
) -> DriveThetaScanResult:
    """Scan over θ values with 4D random search and Nelder--Mead refinement.

    For each θ:
    1. Run `n_random` random evaluations in the 4D parameter space.
    2. Select the best `n_nm_refine` points.
    3. Run Nelder--Mead refinement from each selected point.
    4. Record the best overall result.

    Args:
        theta_values: θ values to scan.
        n_random: Number of random search points per θ.
        n_nm_refine: Number of Nelder--Mead refinements per θ.
        seed: Base random seed (incremented per θ).
        maxiter: Maximum Nelder--Mead iterations.
        bounds: (min, max) for all parameters.
        T_H: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        DriveThetaScanResult with optimal parameters and sensitivities.
    """
    theta_arr = np.asarray(theta_values, dtype=float)
    base_seed = seed if seed is not None else 42

    best_params_list: list[tuple[float, float, float, float]] = []
    best_deltas: list[float] = []
    sql_vals: list[float] = []
    exp_vals: list[float] = []
    var_vals: list[float] = []
    all_results_dict: dict[float, list[DriveNelderMeadResult]] = {}

    for theta in theta_arr:
        # Stage 1: Random search
        rs_result = phase_modulated_random_search(
            theta,
            n_samples=n_random,
            bounds=bounds,
            T_H=T_H,
            T_BS=T_BS,
            seed=base_seed + int(theta * 1000),
        )

        # Sort random-search results by Δθ, take top n_nm_refine
        sorted_indices = np.argsort(rs_result.delta_theta_values)
        top_indices = sorted_indices[:n_nm_refine]

        # Stage 2: Nelder--Mead refinement from each top point
        nm_results: list[DriveNelderMeadResult] = []
        for rank, idx in enumerate(top_indices):
            x0 = rs_result.samples[idx].copy()
            nm = run_phase_modulated_nelder_mead(
                theta_true=theta,
                x0=x0,
                seed=base_seed + int(theta * 1000) + 10000 + rank,
                maxiter=maxiter,
                bounds=bounds,
                T_H=T_H,
                T_BS=T_BS,
                track_history=False,
            )
            nm_results.append(nm)

        # Sort Nelder--Mead results by Δθ
        nm_results.sort(key=lambda r: r.delta_theta_opt)
        best_nm = nm_results[0]

        best_params_list.append(
            (
                float(best_nm.params_opt[0]),
                float(best_nm.params_opt[1]),
                float(best_nm.params_opt[2]),
                float(best_nm.params_opt[3]),
            )
        )
        best_deltas.append(best_nm.delta_theta_opt)
        sql_vals.append(1.0 / T_H)
        exp_vals.append(best_nm.expectation_Jz)
        var_vals.append(best_nm.variance_Jz)
        all_results_dict[float(theta)] = nm_results

    return DriveThetaScanResult(
        theta_values=theta_arr,
        best_params_per_theta=best_params_list,
        best_delta_theta_per_theta=np.array(best_deltas, dtype=float),
        sql_values=np.array(sql_vals, dtype=float),
        expectation_Jz_per_theta=np.array(exp_vals, dtype=float),
        variance_Jz_per_theta=np.array(var_vals, dtype=float),
        all_results=all_results_dict,
    )

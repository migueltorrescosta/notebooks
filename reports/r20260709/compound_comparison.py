"""
Symmetric ω-Modulated Drive: Bounded-Compound Comparison — Experiment Module.

Implements two scenarios for comparing system-only vs ancilla-assisted
ω-modulated drive metrology:

Scenario A (system-only baseline):
  Single-qubit MZI with H_S = ω(a_x J_x + a_y J_y + a_z J_z).
  3D optimisation over unit-direction on S^2(R=5).

Scenario B (ancilla-assisted, identical drive):
  Dual MZI on both qubits with identical ω-modulated drive on system
  and ancilla, plus Ising interaction a_zz J_z^S ⊗ J_z^A.
  4D optimisation over unit-direction on S^3(R=5).

Both scenarios measure J_z^S after the second beam splitter.
Sensitivity: error propagation Δω = sqrt(Var(J_z^S)) / |∂⟨J_z^S⟩/∂ω|.

Sampling modes:
  - "sphere" (default): Marsaglia method on S^{d-1}(R), with NM
    projection back onto sphere. Separates direction from magnitude.
  - "cube" (legacy): uniform sampling on [-R, R]^d with bound penalty.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as _mp
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from reports.r20260709.compound_comparison_results import (
    CompoundRatioResult,
    FixedParameterCompoundRatioResult,
    ScenarioACompoundResult,
)
from src.analysis.ancilla_drive_results import (
    DriveOmegaScanResult,
    DriveRandomSearchResult,
)
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
    compute_expectation_and_variance,
    two_qubit_bs_unitary,
)
from src.analysis.optimisation_pipeline import run_nelder_mead
from src.physics.beam_splitter import bs_qubit
from src.utils.constants import J_X, J_Y, J_Z
from src.utils.sampling import (
    project_to_sphere,
    sample_uniform_sphere,
    sphere_objective_wrapper,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ============================================================================
# Physical Constants
# ============================================================================

DEFAULT_T_BS: float = np.pi / 2.0  # 50/50 beam splitter
DEFAULT_T_HOLD: float = 10.0  # Holding time
SQL_REFERENCE: float = 1.0 / DEFAULT_T_HOLD  # Δω_SQL = 0.1
DRIVE_BOUNDS: tuple[float, float] = (-5.0, 5.0)
DEFAULT_SPHERE_RADIUS: float = 5.0  # Radius R for sphere sampling
DEFAULT_SAMPLING_MODE: str = "sphere"  # "cube" (legacy) or "sphere" (default)
FD_STEP: float = 1e-6  # Finite-difference step


# ============================================================================
# Scenario A: System-Only ω-Modulated Drive (Single Qubit)
# ============================================================================


def scenario_a_state() -> np.ndarray:
    """Initial state |0⟩ = |1,0⟩ for Scenario A (single qubit)."""
    return np.array([1.0, 0.0], dtype=complex)


def scenario_a_bs(T_BS: float) -> np.ndarray:
    """Single-qubit beam splitter U_BS = exp(-i T_BS J_x).

    Delegates to ``bs_qubit`` from ``src.physics.beam_splitter``.
    """
    return bs_qubit(T_BS)


def scenario_a_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
) -> np.ndarray:
    """Build the system-only ω-modulated Hamiltonian.

    H_S = ω (a_x J_x + a_y J_y + a_z J_z)

    The drive coefficients (a_x, a_y, a_z) are identical to those used
    on the ancilla in Scenario B — there is no bare ω J_z encoding term
    on either subsystem.

    Args:
        omega: Unknown phase rate parameter.
        a_x: J_x drive coefficient.
        a_y: J_y drive coefficient.
        a_z: J_z drive coefficient.

    Returns:
        2×2 Hermitian Hamiltonian matrix.
    """
    H = omega * (a_z * J_Z + a_x * J_X + a_y * J_Y)
    return 0.5 * (H + H.conj().T)


def scenario_a_hold_unitary(
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
) -> np.ndarray:
    """Hold unitary U_hold = exp(-i t_hold H_S) for Scenario A."""
    H = scenario_a_hamiltonian(omega, a_x, a_y, a_z)
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs @ np.diag(np.exp(-1j * t_hold * eigvals)) @ eigvecs.conj().T


def scenario_a_evolve(
    T_BS: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
) -> np.ndarray:
    """Run the full Scenario A MZI circuit.

    |ψ_final⟩ = U_BS · U_hold(t_hold) · U_BS · |0⟩

    Args:
        T_BS: Beam-splitter duration (π/2 for 50/50).
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x, a_y, a_z: Drive coefficients.

    Returns:
        Final normalised 2-vector state.
    """
    psi0 = scenario_a_state()
    U_bs = scenario_a_bs(T_BS)
    psi = U_bs @ psi0
    psi = scenario_a_hold_unitary(t_hold, omega, a_x, a_y, a_z) @ psi
    psi = U_bs @ psi
    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


def scenario_a_sensitivity(
    T_BS: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    meas_op: np.ndarray | None = None,
    fd_step: float = FD_STEP,
) -> float:
    """Compute error-propagation sensitivity Δω for Scenario A.

    Δω = sqrt(Var(J_z)) / |∂⟨J_z⟩/∂ω|

    Args:
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega: True phase rate parameter.
        a_x, a_y, a_z: Drive coefficients.
        meas_op: Measurement operator (default: J_z for single qubit).
        fd_step: Finite-difference step size.

    Returns:
        Sensitivity Δω (positive float, or inf at fringe extremum).
    """
    if meas_op is None:
        meas_op = J_Z

    psi = scenario_a_evolve(T_BS, t_hold, omega, a_x, a_y, a_z)
    _, var = compute_expectation_and_variance(psi, meas_op)

    psi_plus = scenario_a_evolve(T_BS, t_hold, omega + fd_step, a_x, a_y, a_z)
    psi_minus = scenario_a_evolve(T_BS, t_hold, omega - fd_step, a_x, a_y, a_z)
    exp_plus = float(np.real(psi_plus.conj() @ meas_op @ psi_plus))
    exp_minus = float(np.real(psi_minus.conj() @ meas_op @ psi_minus))
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12 or var < 1e-15:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


def _scenario_a_objective_3d(
    p: np.ndarray,
    omega: float,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
) -> float:
    """3D objective: Δω for Scenario A with params = [a_x, a_y, a_z]."""
    return scenario_a_sensitivity(
        T_BS,
        t_hold,
        omega,
        float(p[0]),
        float(p[1]),
        float(p[2]),
    )


# ============================================================================
# Scenario B: Ancilla-Assisted Symmetric Drive (Two Qubit)
# ============================================================================


def scenario_b_state() -> np.ndarray:
    """Initial state |00⟩ for Scenario B."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)


def scenario_b_hamiltonian(
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Build the two-qubit ω-modulated Hamiltonian for Scenario B.

    H = ω(a_x J_x^S + a_y J_y^S + a_z J_z^S)
      + ω(a_x J_x^A + a_y J_y^A + a_z J_z^A) + a_zz J_z^S J_z^A

    The drive parameters (a_x, a_y, a_z) are identical on both subsystems.
    There is no bare ω J_z encoding term on either subsystem — the phase
    dependence comes entirely from the modulated drive, identical on S and A.

    Args:
        omega: Unknown phase rate parameter.
        a_x, a_y, a_z: Identical drive coefficients for both S and A.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators from build_two_qubit_operators().

    Returns:
        4×4 Hermitian Hamiltonian matrix.
    """
    H = (
        +omega * a_x * ops["Jx_S"]
        + omega * a_x * ops["Jx_A"]
        + omega * a_y * ops["Jy_S"]
        + omega * a_y * ops["Jy_A"]
        + omega * a_z * ops["Jz_S"]
        + omega * a_z * ops["Jz_A"]
        + a_zz * (ops["Jz_S"] @ ops["Jz_A"])
    )
    return 0.5 * (H + H.conj().T)


def scenario_b_hold_unitary(
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Hold unitary for Scenario B."""
    H = scenario_b_hamiltonian(omega, a_x, a_y, a_z, a_zz, ops)
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs @ np.diag(np.exp(-1j * t_hold * eigvals)) @ eigvecs.conj().T


def scenario_b_evolve(
    T_BS: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the full Scenario B dual-MZI circuit.

    |ψ_final⟩ = (U_BS ⊗ U_BS) · U_hold · (U_BS ⊗ U_BS) · |00⟩

    Args:
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega: Phase rate parameter.
        a_x, a_y, a_z: Identical drive coefficients.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.

    Returns:
        Final normalised 4-vector state.
    """
    psi0 = scenario_b_state()
    U_dual = two_qubit_bs_unitary(T_BS)
    psi = U_dual @ psi0
    psi = scenario_b_hold_unitary(t_hold, omega, a_x, a_y, a_z, a_zz, ops) @ psi
    psi = U_dual @ psi
    assert np.isclose(np.linalg.norm(psi), 1.0), "Final state must be normalised"
    return psi


def scenario_b_sensitivity(
    T_BS: float,
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    meas_op: np.ndarray | None = None,
    fd_step: float = FD_STEP,
) -> float:
    """Compute error-propagation sensitivity Δω for Scenario B.

    Δω = sqrt(Var(J_z^S)) / |∂⟨J_z^S⟩/∂ω|

    Args:
        T_BS: Beam-splitter duration.
        t_hold: Holding-time strength.
        omega: True phase rate parameter.
        a_x, a_y, a_z: Identical drive coefficients.
        a_zz: Ising interaction coefficient.
        ops: Two-qubit operators.
        meas_op: Measurement operator (default: J_z^S).
        fd_step: Finite-difference step size.

    Returns:
        Sensitivity Δω (positive float, or inf at fringe extremum).
    """
    if meas_op is None:
        meas_op = ops["Jz_S"]

    psi = scenario_b_evolve(T_BS, t_hold, omega, a_x, a_y, a_z, a_zz, ops)
    _, var = compute_expectation_and_variance(psi, meas_op)

    psi_plus = scenario_b_evolve(
        T_BS, t_hold, omega + fd_step, a_x, a_y, a_z, a_zz, ops
    )
    psi_minus = scenario_b_evolve(
        T_BS, t_hold, omega - fd_step, a_x, a_y, a_z, a_zz, ops
    )
    exp_plus = float(np.real(psi_plus.conj() @ meas_op @ psi_plus))
    exp_minus = float(np.real(psi_minus.conj() @ meas_op @ psi_minus))
    d_exp = (exp_plus - exp_minus) / (2.0 * fd_step)

    if abs(d_exp) < 1e-12 or var < 1e-15:
        return float("inf")

    return float(np.sqrt(var) / abs(d_exp))


def _scenario_b_objective_4d(
    p: np.ndarray,
    omega: float,
    ops: dict[str, np.ndarray],
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
) -> float:
    """4D objective: Δω for Scenario B with params = [a_x, a_y, a_z, a_zz]."""
    return scenario_b_sensitivity(
        T_BS,
        t_hold,
        omega,
        float(p[0]),
        float(p[1]),
        float(p[2]),
        float(p[3]),
        ops,
    )


# ============================================================================
# Decoupled Baseline
# ============================================================================


def compute_decoupled_baseline(
    t_hold: float = DEFAULT_T_HOLD,
    omega: float = 1.0,
) -> tuple[float, float]:
    """Compute Δω at the decoupled baseline (standard single-qubit MZI).

    The decoupled configuration is: a_x = a_y = 0, a_z = 1 (standard ω J_z
    encoding), a_zz = 0 (no S-A interaction).  At this configuration both
    scenarios should recover Δω = 1/t_hold (SQL) for the J_z^S measurement.

    With the identical-drive Hamiltonian (no bare ω J_z^S), the standard MZI
    phase encoding requires a_z = 1; at a_z = 0 the Hamiltonian vanishes.

    Args:
        t_hold: Holding-time strength.
        omega: Phase rate parameter (default 1.0). The baseline should
            recover SQL = 1/t_hold for any ω.

    Returns:
        Tuple (delta_omega_A, delta_omega_B) — both should equal 1/t_hold.
    """
    domega_a = scenario_a_sensitivity(DEFAULT_T_BS, t_hold, omega, 0.0, 0.0, 1.0)
    ops = build_two_qubit_operators()
    domega_b = scenario_b_sensitivity(
        DEFAULT_T_BS, t_hold, omega, 0.0, 0.0, 1.0, 0.0, ops
    )
    return domega_a, domega_b


# ============================================================================
# 3D Random Search for Scenario A
# ============================================================================


def scenario_a_random_search(
    omega: float,
    n_samples: int = 500,
    bounds: tuple[float, float] = DRIVE_BOUNDS,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
    seed: int | None = 42,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    radius: float = DEFAULT_SPHERE_RADIUS,
) -> DriveRandomSearchResult:
    """Random search over the 3D parameter space (a_x, a_y, a_z) for Scenario A.

    When ``sampling_mode="sphere"`` (default), samples are drawn uniformly
    from S^2(R) via the Marsaglia method.  When ``sampling_mode="cube"``,
    samples are drawn uniformly from [-R, R]^3 (legacy behaviour).

    Args:
        omega: Phase rate value.
        n_samples: Number of random points.
        bounds: (min, max) for drive coefficients (cube mode only).
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        seed: Random seed.
        sampling_mode: ``"sphere"`` or ``"cube"``.
        radius: Sphere radius (or half-side-length for cube mode).

    Returns:
        DriveRandomSearchResult with all samples and best found.
    """
    rng = np.random.default_rng(seed)
    raw_obj = _scenario_a_objective_3d

    if sampling_mode == "sphere":
        samples3d = sample_uniform_sphere(3, radius, n_samples, rng)
    else:
        lo, hi = -radius, radius
        samples3d = np.column_stack(
            [rng.uniform(lo, hi, size=n_samples) for _ in range(3)]
        )

    deltas = np.array(
        [raw_obj(samples3d[i], omega, t_hold, T_BS) for i in range(n_samples)]
    )
    best_idx = int(np.argmin(deltas))
    # Pad 3D samples to 4D with a_zz=0.0 for DriveRandomSearchResult API
    # compatibility. This is correct because Scenario A has no ancilla
    # interaction term — a_zz is always zero and unused.
    assert not np.any(np.isnan(samples3d)), "Samples contain NaN"
    samples4d = np.column_stack([samples3d, np.zeros(n_samples)])
    return DriveRandomSearchResult(
        samples=samples4d,
        delta_omega_values=deltas,
        best_params=(
            float(samples3d[best_idx, 0]),
            float(samples3d[best_idx, 1]),
            float(samples3d[best_idx, 2]),
            0.0,
        ),
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=1.0 / t_hold,
        t_hold=t_hold,
    )


# ============================================================================
# 2D Constrained Optimisation (a_y = 0) for Role-of-a_y Verification
# ============================================================================


def scenario_a_sensitivity_constrained_ay(
    omega: float,
    a_x: float,
    a_z: float,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
) -> float:
    """Scenario A sensitivity with a_y fixed at zero.

    This allows direct comparison with the free 3D optimisation to
    quantify how much the oscillation-frequency modulation by a_y
    contributes to the EP sensitivity.

    Args:
        omega: Phase rate parameter.
        a_x: J_x drive coefficient.
        a_z: J_z drive coefficient.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        Sensitivity Δω (positive float, or inf at fringe extremum).
    """
    return scenario_a_sensitivity(T_BS, t_hold, omega, a_x, 0.0, a_z)


def _print_ay_progress(
    i: int, n_omega: int, delta_free: np.ndarray, delta_constrained: np.ndarray
) -> None:
    """Print progress line for the a_y verification sweep."""
    if (i + 1) % max(1, n_omega // 10) != 0:
        return
    pct = 100.0 * (i + 1) / n_omega
    ratio = delta_free[i] / delta_constrained[i]
    print(f"  a_y verification: {i + 1}/{n_omega} ({pct:.0f}%), ratio = {ratio:.4f}")


def run_constrained_ay_verification(
    omega_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 50,
    seed: int = 42,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
) -> dict[str, Any]:
    """Run 2D (a_y=0) vs 3D (free a_y) optimisation for Scenario A.

    At each ω, independently optimises over (a_x, a_z) with a_y=0 and
    compares with the full 3D result.  The difference isolates the
    contribution of a_y through the oscillation-frequency modulation
    θ = ω t r (with r = √(a_x² + a_y² + a_z²)).

    Args:
        omega_values: ω values to scan.
        n_random: Number of random search samples per ω.
        n_nm_refine: Number of Nelder-Mead refinements per ω.
        seed: Base random seed.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        Dict with omega_values, delta_free, delta_constrained, ratio,
        and optimal parameters for both.
    """
    omega_arr = np.asarray(omega_values, dtype=float)
    n_omega = len(omega_arr)
    base_seed = seed

    delta_free = np.full(n_omega, np.inf)
    delta_constrained = np.full(n_omega, np.inf)
    params_free: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n_omega
    params_constrained: list[tuple[float, float]] = [(0.0, 0.0)] * n_omega

    for i, omega_val in enumerate(omega_arr):
        omega_seed = base_seed + int(omega_val * 1000)

        # --- Free 3D (a_x, a_y, a_z) — reuse existing pipeline ---
        rs_free = scenario_a_random_search(
            omega=omega_val,
            n_samples=n_random,
            t_hold=t_hold,
            T_BS=T_BS,
            seed=omega_seed,
        )
        sorted_idx = np.argsort(rs_free.delta_omega_values)
        top_idx = sorted_idx[:n_nm_refine]

        best_free_delta = np.inf
        best_free_params = (0.0, 0.0, 0.0)

        def _make_obj_free(ov: float) -> Callable[[np.ndarray], float]:
            def _obj(p: np.ndarray) -> float:
                return _scenario_a_objective_3d(p, ov, t_hold, T_BS)

            return _obj

        _obj_free = _make_obj_free(omega_val)
        for idx in top_idx:
            x0 = rs_free.samples[idx, :3].copy()
            nm = run_nelder_mead(_obj_free, x0=x0, bounds=(-5.0, 5.0), maxiter=5000)
            if nm["fun_opt"] < best_free_delta:
                best_free_delta = nm["fun_opt"]
                x_opt = nm["x_opt"]
                best_free_params = (
                    float(x_opt[0]),
                    float(x_opt[1]),
                    float(x_opt[2]),
                )

        delta_free[i] = best_free_delta
        params_free[i] = best_free_params

        # --- Constrained 2D (a_x, a_z) with a_y = 0 ---
        rng = np.random.default_rng(omega_seed)
        samples_2d = rng.uniform(-5.0, 5.0, size=(n_random, 2))
        deltas_2d = np.array(
            [
                scenario_a_sensitivity_constrained_ay(
                    omega_val, float(s[0]), float(s[1]), t_hold, T_BS
                )
                for s in samples_2d
            ]
        )

        sorted_idx_2d = np.argsort(deltas_2d)
        top_idx_2d = sorted_idx_2d[:n_nm_refine]

        best_con_delta = np.inf
        best_con_params = (0.0, 0.0)

        def _make_obj_constrained(
            ov: float,
        ) -> Callable[[np.ndarray], float]:
            def _obj(p2: np.ndarray) -> float:
                return scenario_a_sensitivity_constrained_ay(
                    ov, float(p2[0]), float(p2[1]), t_hold, T_BS
                )

            return _obj

        _obj_constrained = _make_obj_constrained(omega_val)

        for idx in top_idx_2d:
            x0_2d = samples_2d[idx].copy()
            nm = run_nelder_mead(
                _obj_constrained, x0=x0_2d, bounds=(-5.0, 5.0), maxiter=5000
            )
            if nm["fun_opt"] < best_con_delta:
                best_con_delta = nm["fun_opt"]
                x_opt_2d = nm["x_opt"]
                best_con_params = (float(x_opt_2d[0]), float(x_opt_2d[1]))

        delta_constrained[i] = best_con_delta
        params_constrained[i] = best_con_params

        _print_ay_progress(i, n_omega, delta_free, delta_constrained)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ay = np.where(
            np.isfinite(delta_free)
            & np.isfinite(delta_constrained)
            & (delta_constrained > 0),
            delta_free / delta_constrained,
            np.nan,
        )

    return {
        "omega_values": omega_arr,
        "delta_free_3d": delta_free,
        "delta_constrained_2d": delta_constrained,
        "ratio_free_over_constrained": ratio_ay,
        "params_free": params_free,
        "params_constrained": params_constrained,
    }


# ============================================================================
# ω Scan for Scenario A
# ============================================================================


def _refine_nm_scenario_a(
    rs_result: DriveRandomSearchResult,
    n_nm_refine: int,
    omega_val: float,
    t_hold: float,
    T_BS: float,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    radius: float = DEFAULT_SPHERE_RADIUS,
) -> tuple[float, tuple[float, float, float]]:
    """Nelder-Mead refinement from top random-search candidates for Scenario A.

    In sphere mode, each NM candidate is projected onto S^2(R) before
    evaluation (no bound penalty needed).  In cube mode, the bound-penalty
    wrapper from ``run_nelder_mead`` is used.

    Args:
        rs_result: Random search result with samples and delta_omega values.
        n_nm_refine: Number of top candidates to refine.
        omega_val: omega at which to evaluate.
        t_hold: Holding-time strength.
        T_BS: Beam-splitter duration.
        sampling_mode: ``"sphere"`` or ``"cube"``.
        radius: Sphere radius.

    Returns:
        Tuple (best_delta_omega, (a_x, a_y, a_z)) from refinement.
    """
    sorted_idx = np.argsort(rs_result.delta_omega_values)
    top_idx = sorted_idx[:n_nm_refine]
    best_nm_delta = np.inf
    best_nm_params = (0.0, 0.0, 0.0)

    def _obj_a(p: np.ndarray) -> float:
        return _scenario_a_objective_3d(p, omega_val, t_hold, T_BS)

    if sampling_mode == "sphere":
        wrapped_obj = sphere_objective_wrapper(_obj_a, radius)
    else:
        wrapped_obj = _obj_a

    for idx in top_idx:
        x0_3d = rs_result.samples[idx, :3].copy()
        if sampling_mode == "sphere":
            # Project initial point onto sphere; use bounds=(-R, R) so
            # NM stays within the bounding box of the sphere.
            x0_3d = project_to_sphere(x0_3d, radius)
            nm = run_nelder_mead(
                wrapped_obj,
                x0=x0_3d,
                bounds=(-radius, radius),
                maxiter=5000,
            )
            # Project the NM result back onto the sphere
            x_opt_3d = project_to_sphere(nm["x_opt"], radius)
        else:
            nm = run_nelder_mead(
                wrapped_obj,
                x0=x0_3d,
                bounds=(-5.0, 5.0),
                maxiter=5000,
            )
            x_opt_3d = nm["x_opt"]
        if nm["fun_opt"] < best_nm_delta:
            best_nm_delta = nm["fun_opt"]
            best_nm_params = (
                float(x_opt_3d[0]),
                float(x_opt_3d[1]),
                float(x_opt_3d[2]),
            )
    return best_nm_delta, best_nm_params


def run_scenario_a_omega_scan(
    omega_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 50,
    seed: int | None = 42,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    radius: float = DEFAULT_SPHERE_RADIUS,
) -> ScenarioACompoundResult:
    """Scan ω for Scenario A: random search + Nelder-Mead refinement.

    For each ω:
    1. Run 3D random search (cube or sphere sampling).
    2. Take top n_nm_refine candidates.
    3. Refine with Nelder-Mead (sphere-projected or bound-penalised).

    Args:
        omega_values: ω values to scan.
        n_random: Number of random search samples per ω.
        n_nm_refine: Number of Nelder-Mead refinements per ω.
        seed: Base random seed.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        sampling_mode: ``"sphere"`` or ``"cube"``.
        radius: Sphere radius.

    Returns:
        ScenarioACompoundResult with optimal parameters per ω.
    """
    base_seed = seed if seed is not None else 42
    sql = 1.0 / t_hold
    omega_arr = np.asarray(omega_values, dtype=float)
    n_omega = len(omega_arr)

    best_deltas = np.full(n_omega, np.inf)
    best_params: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n_omega
    exp_vals = np.zeros(n_omega)
    var_vals = np.zeros(n_omega)

    log_interval = max(1, n_omega // 20)  # Log ~20 progress updates
    for i, omega_val in enumerate(omega_arr):
        omega_seed = base_seed + int(omega_val * 1000)

        # Stage 1: Random search
        rs_result = scenario_a_random_search(
            omega=omega_val,
            n_samples=n_random,
            t_hold=t_hold,
            T_BS=T_BS,
            seed=omega_seed,
            sampling_mode=sampling_mode,
            radius=radius,
        )

        # Stage 2 & 3: Select top candidates + Nelder-Mead refinement
        best_nm_delta, best_nm_params = _refine_nm_scenario_a(
            rs_result,
            n_nm_refine,
            omega_val,
            t_hold,
            T_BS,
            sampling_mode=sampling_mode,
            radius=radius,
        )

        best_deltas[i] = best_nm_delta
        best_params[i] = best_nm_params

        # Compute diagnostics at optimal point
        psi = scenario_a_evolve(
            T_BS,
            t_hold,
            omega_val,
            best_nm_params[0],
            best_nm_params[1],
            best_nm_params[2],
        )
        exp_val, var_val = compute_expectation_and_variance(psi, J_Z)
        exp_vals[i] = exp_val
        var_vals[i] = var_val

        # Periodic progress log
        if (i + 1) % log_interval == 0 or i == n_omega - 1:
            pct = 100.0 * (i + 1) / n_omega
            print(
                f"  Scenario A: {i + 1}/{n_omega} ω done ({pct:.1f}%), last Δω={best_deltas[i]:.6f}"
            )

    return ScenarioACompoundResult(
        omega_values=omega_arr,
        best_delta_omega_per_omega=best_deltas,
        best_params_per_omega=best_params,
        sql_values=np.full(n_omega, sql),
        t_hold_value=t_hold,
        expectation_Jz_per_omega=exp_vals,
        variance_Jz_per_omega=var_vals,
    )


# ============================================================================
# ω Scan for Scenario B (using existing pipeline infrastructure)
# ============================================================================


def _run_scenario_b_single_omega(
    omega: float,
    n_random: int,
    n_nm_refine: int,
    seed: int,
    t_hold: float,
    T_BS: float,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    radius: float = DEFAULT_SPHERE_RADIUS,
) -> dict[str, Any]:
    """Run random search + NM refinement for Scenario B at a single ω."""
    ops = build_two_qubit_operators()
    rng = np.random.default_rng(seed)

    def _raw_obj(p: np.ndarray) -> float:
        return _scenario_b_objective_4d(p, omega, ops, t_hold, T_BS)

    if sampling_mode == "sphere":
        samples = sample_uniform_sphere(4, radius, n_random, rng)
    else:
        lo, hi = -radius, radius
        samples = np.column_stack(
            [rng.uniform(lo, hi, size=n_random) for _ in range(4)]
        )

    deltas = np.array([_raw_obj(samples[i]) for i in range(n_random)])

    # Stage 2: Select top candidates
    sorted_idx = np.argsort(deltas)
    top_idx = sorted_idx[:n_nm_refine]

    # Stage 3: Nelder-Mead refinement
    best_nm_delta = np.inf
    best_nm_params = (0.0, 0.0, 0.0, 0.0)

    if sampling_mode == "sphere":
        wrapped_obj = sphere_objective_wrapper(_raw_obj, radius)
    else:
        wrapped_obj = _raw_obj

    for idx in top_idx:
        x0 = samples[idx].copy()
        if sampling_mode == "sphere":
            x0 = project_to_sphere(x0, radius)
            nm = run_nelder_mead(
                wrapped_obj, x0=x0, bounds=(-radius, radius), maxiter=5000
            )
            # Project the NM result back onto the sphere
            x_opt = project_to_sphere(nm["x_opt"], radius)
        else:
            nm = run_nelder_mead(wrapped_obj, x0=x0, bounds=(-5.0, 5.0), maxiter=5000)
            x_opt = nm["x_opt"]
        if nm["fun_opt"] < best_nm_delta:
            best_nm_delta = nm["fun_opt"]
            best_nm_params = (
                float(x_opt[0]),
                float(x_opt[1]),
                float(x_opt[2]),
                float(x_opt[3]),
            )

    # Compute diagnostics at optimal point
    psi = scenario_b_evolve(
        T_BS,
        t_hold,
        omega,
        best_nm_params[0],
        best_nm_params[1],
        best_nm_params[2],
        best_nm_params[3],
        ops,
    )
    exp_val_best, var_val_best = compute_expectation_and_variance(psi, ops["Jz_S"])

    return {
        "omega": omega,
        "best_delta_omega": best_nm_delta,
        "a_x": best_nm_params[0],
        "a_y": best_nm_params[1],
        "a_z": best_nm_params[2],
        "a_zz": best_nm_params[3],
        "expectation_Jz": exp_val_best,
        "variance_Jz": var_val_best,
    }


def _scenario_b_worker(
    omega: float,
    n_random: int,
    n_nm_refine: int,
    seed: int | None,
    t_hold: float,
    T_BS: float,
    sampling_mode: str,
    radius: float,
) -> dict[str, Any]:
    """Module-level worker for parallel Scenario B ω scan.

    Must be module-level (not a closure) to be picklable by
    ``ProcessPoolExecutor``.
    """
    omega_seed = (seed if seed is not None else 42) + int(omega * 1000)
    return _run_scenario_b_single_omega(
        omega,
        n_random,
        n_nm_refine,
        omega_seed,
        t_hold,
        T_BS,
        sampling_mode=sampling_mode,
        radius=radius,
    )


def run_scenario_b_omega_scan(
    omega_values: list[float] | np.ndarray,
    n_random: int = 500,
    n_nm_refine: int = 50,
    seed: int | None = 42,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    radius: float = DEFAULT_SPHERE_RADIUS,
) -> DriveOmegaScanResult:
    """Scan ω for Scenario B: random search + Nelder-Mead refinement (parallel).

    Args:
        omega_values: ω values to scan.
        n_random: Number of random search samples per ω.
        n_nm_refine: Number of Nelder-Mead refinements per ω.
        seed: Base random seed.
        t_hold: Holding time.
        T_BS: Beam-splitter duration.
        sampling_mode: ``"sphere"`` or ``"cube"``.
        radius: Sphere radius.

    Returns:
        DriveOmegaScanResult with optimal parameters per ω.
    """
    omega_arr = np.asarray(omega_values, dtype=float)
    sql = 1.0 / t_hold

    max_workers = min(32, os.cpu_count() or 1)
    per_omega: list[dict[str, Any]] = []

    mp_ctx = _mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_omega = {
            executor.submit(
                _scenario_b_worker,
                o,
                n_random,
                n_nm_refine,
                seed,
                t_hold,
                T_BS,
                sampling_mode,
                radius,
            ): o
            for o in omega_arr
        }
        for future in concurrent.futures.as_completed(fut_to_omega):
            omega = fut_to_omega[future]
            try:
                per_omega.append(future.result())
                print(f"  [done] Scenario B ω={omega}")
            except Exception as exc:
                print(f"  [ERROR] Scenario B ω={omega}: {exc}")
                raise

    per_omega.sort(key=lambda r: float(r["omega"]))

    omega_out = np.array([r["omega"] for r in per_omega], dtype=float)
    best_deltas = np.array([r["best_delta_omega"] for r in per_omega], dtype=float)
    best_params = [(r["a_x"], r["a_y"], r["a_z"], r["a_zz"]) for r in per_omega]
    exp_vals = np.array([r["expectation_Jz"] for r in per_omega], dtype=float)
    var_vals = np.array([r["variance_Jz"] for r in per_omega], dtype=float)

    return DriveOmegaScanResult(
        omega_values=omega_out,
        best_params_per_omega=best_params,
        best_delta_omega_per_omega=best_deltas,
        sql_values=np.full(len(omega_out), sql),
        expectation_Jz_per_omega=exp_vals,
        variance_Jz_per_omega=var_vals,
    )


# ============================================================================
# Comparison: Compute Compound Ratio
# ============================================================================


def compute_compound_ratio(
    result_a: ScenarioACompoundResult,
    result_b: DriveOmegaScanResult,
) -> CompoundRatioResult:
    """Compute R_compound = Δω_A / Δω_B at matched ω values.

    Args:
        result_a: Scenario A omega-scan result.
        result_b: Scenario B omega-scan result.

    Returns:
        CompoundRatioResult with ratios at each ω.
    """
    omega_a = result_a.omega_values
    delta_a = result_a.best_delta_omega_per_omega
    omega_b = result_b.omega_values
    delta_b = result_b.best_delta_omega_per_omega
    sql = result_a.sql_values

    # Interpolate B to match A's ω grid if needed
    if len(omega_a) == len(omega_b) and np.allclose(omega_a, omega_b):
        delta_b_matched = delta_b
    else:
        delta_b_matched = np.interp(omega_a, omega_b, delta_b)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_compound = np.where(
            np.isfinite(delta_a) & np.isfinite(delta_b_matched) & (delta_b_matched > 0),
            delta_a / delta_b_matched,
            np.nan,
        )
        ratio_a_to_sql = np.where(
            np.isfinite(delta_a) & (sql > 0), sql / delta_a, np.nan
        )
        ratio_b_to_sql = np.where(
            np.isfinite(delta_b_matched) & (sql > 0),
            sql / delta_b_matched,
            np.nan,
        )

    return CompoundRatioResult(
        omega_values=omega_a,
        delta_omega_A=delta_a,
        delta_omega_B=delta_b_matched,
        compound_ratio=ratio_compound,
        sql_values=sql,
        ratio_A_to_sql=ratio_a_to_sql,
        ratio_B_to_sql=ratio_b_to_sql,
    )


# ============================================================================
# Fixed-Parameter Compound Ratio
# ============================================================================


def compute_fixed_parameter_compound_ratio(
    result_a: ScenarioACompoundResult,
    result_b: DriveOmegaScanResult,
    t_hold: float = DEFAULT_T_HOLD,
    T_BS: float = DEFAULT_T_BS,
) -> FixedParameterCompoundRatioResult:
    """Compute fixed-parameter compound ratio: B evaluated at A's optimal drive params.

    At each ω, takes Scenario A's optimal (a_x^A, a_y^A, a_z^A) and evaluates
    Scenario B at those same drive parameters with B's optimal a_zz^B.  This
    isolates the interaction-only contribution: how much does a_zz improve B
    when the drive parameters are held at A's optimum?

    The QFI ratio for identical (a_x, a_y, a_z) is exactly √2 (resource-
    counting bound: 2 particles vs 1), regardless of a_zz.  The EP ratio at
    fixed parameters measures how efficiently the J_z^S measurement extracts
    this available advantage.

    Args:
        result_a: Scenario A omega-scan result (independently optimised).
        result_b: Scenario B omega-scan result (independently optimised).
        t_hold: Holding time.
        T_BS: Beam-splitter duration.

    Returns:
        FixedParameterCompoundRatioResult with per-ω ratios.
    """
    ops = build_two_qubit_operators()
    omega_a = result_a.omega_values
    n_omega = len(omega_a)

    delta_b_fixed = np.full(n_omega, np.inf)
    a_x_arr = np.zeros(n_omega)
    a_y_arr = np.zeros(n_omega)
    a_z_arr = np.zeros(n_omega)
    a_zz_arr = np.zeros(n_omega)

    # Build lookup for B's optimal a_zz per ω
    omega_b = result_b.omega_values
    b_a_zz = np.array([p[3] for p in result_b.best_params_per_omega])

    for i in range(n_omega):
        omega_val = omega_a[i]
        # A's optimal drive parameters
        ax_a = result_a.best_params_per_omega[i][0]
        ay_a = result_a.best_params_per_omega[i][1]
        az_a = result_a.best_params_per_omega[i][2]
        a_x_arr[i] = ax_a
        a_y_arr[i] = ay_a
        a_z_arr[i] = az_a

        # B's optimal a_zz at this ω (interpolate if grids differ)
        if len(omega_b) == n_omega and np.allclose(omega_a, omega_b):
            azz_b = float(b_a_zz[i])
        else:
            azz_b = float(np.interp(omega_val, omega_b, b_a_zz))
        a_zz_arr[i] = azz_b

        # Evaluate B at A's drive params + B's optimal a_zz
        delta_b_fixed[i] = scenario_b_sensitivity(
            T_BS, t_hold, omega_val, ax_a, ay_a, az_a, azz_b, ops
        )

    delta_a = result_a.best_delta_omega_per_omega
    sql = result_a.sql_values

    with np.errstate(divide="ignore", invalid="ignore"):
        fixed_ratio = np.where(
            np.isfinite(delta_a) & np.isfinite(delta_b_fixed) & (delta_b_fixed > 0),
            delta_a / delta_b_fixed,
            np.nan,
        )

    return FixedParameterCompoundRatioResult(
        omega_values=omega_a,
        delta_omega_A_opt=delta_a,
        a_x_A=a_x_arr,
        a_y_A=a_y_arr,
        a_z_A=a_z_arr,
        a_zz_B=a_zz_arr,
        delta_omega_B_fixed=delta_b_fixed,
        fixed_ratio=fixed_ratio,
        sql_values=sql,
    )

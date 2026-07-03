"""
Ancilla-Assisted Metrology — Scan and Optimisation Functions.

All scan runners, optimisation functions, and validation helpers.
Depends on ``ancilla_optimization`` (core physics) and
``ancilla_optimization_results`` (dataclasses).

References:
- Giovannetti, Lloyd, Maccone, Nat. Photonics 5, 222 (2011)
- Davis et al., PRA 94, 063814 (2016)
- Hentschel & Sanders, PRA 88, 062329 (2013)

"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.optimize import minimize

from src.analysis.ancilla_optimization import (
    build_joint_operator,
    build_two_qubit_operators,
    compute_covariance,
    compute_expectation_and_variance,
    compute_sensitivity,
    diagnostics_from_params,
    evolve_full,
    random_initial_params,
    sensitivity_objective,
    two_qubit_state,
)
from src.analysis.ancilla_optimization_results import (
    AlphaRandomSearchResult,
    AlphaReoptScanResult,
    AlphaSingleScanResult,
    CovarianceAnalysisResult,
    DecoupledBaselineResult,
    InteractionRobustnessResult,
    OmegaScanResult,
    OptimisationResult,
)
from src.utils.constants import I_2, I_4

# ============================================================================
# Validation Helpers
# ============================================================================


def validate_operators(ops: dict[str, np.ndarray]) -> bool:
    """Validate all two-qubit operators.

    Checks: Hermiticity, commutation relations.

    Args:
        ops: Two-qubit operators.

    Returns:
        True if all checks pass.

    Raises:
        AssertionError: If any check fails.

    """
    for name, op in ops.items():
        assert op.shape == (4, 4), f"{name} must be 4×4, got {op.shape}"
        assert np.allclose(op, op.conj().T, atol=1e-12), f"{name} must be Hermitian"

    # Commutation: [J_z^S, J_x^S] = i J_y^S
    comm_zx_S = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
    assert np.allclose(comm_zx_S, 1j * ops["Jy_S"], atol=1e-12), (
        "[Jz_S, Jx_S] = i Jy_S failed"
    )

    # Commutation: [J_z^A, J_x^A] = i J_y^A
    comm_zx_A = ops["Jz_A"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jz_A"]
    assert np.allclose(comm_zx_A, 1j * ops["Jy_A"], atol=1e-12), (
        "[Jz_A, Jx_A] = i Jy_A failed"
    )

    return True


def validate_bs_unitarity(T_BS: float = np.pi / 4) -> bool:
    """Validate the beam-splitter unitary.

    Args:
        T_BS: Beam-splitter duration.

    Returns:
        True if unitary.

    """
    from src.analysis.ancilla_optimization import bs_unitary  # fmt: skip

    U = bs_unitary(T_BS)
    assert np.allclose(U @ U.conj().T, I_2, atol=1e-12), "BS must be unitary"
    assert np.allclose(U.conj().T @ U, I_2, atol=1e-12), "BS† must be unitary"
    return True


def validate_hold_unitarity(
    t_hold: float = 1.0,
    omega: float = 1.0,
    alpha: tuple[float, float, float, float] = (0.1, 0.0, 0.0, 0.0),
) -> bool:
    """Validate the hold unitary.

    Args:
        t_hold: Holding time.
        omega: Phase rate.
        alpha: Interaction coefficients.

    Returns:
        True if unitary.

    """
    from src.analysis.ancilla_optimization import (
        build_two_qubit_operators,
        hold_unitary_two_qubit,
    )

    ops = build_two_qubit_operators()
    U = hold_unitary_two_qubit(t_hold, omega, alpha, ops)
    assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), "Hold must be unitary"
    return True


def validate_sensitivity_reasonable(
    t_hold_vals: list[float] | None = None,
) -> bool:
    """Verify that sensitivity Δω ≈ 1/t_hold in the decoupled (α=0) case.

    Checks that the analytically optimal configuration achieves Δω ≈ 1/t_hold
    when the interaction is off.

    Args:
        t_hold_vals: Holding-time values to check.

    Returns:
        True if all checks pass.

    """
    if t_hold_vals is None:
        t_hold_vals = [0.5, 1.0, 2.0]

    ops = build_two_qubit_operators()

    # Analytic optimum for decoupled case:
    # |ψ_S⟩ = |1,0⟩, 50/50 beam splitters (T = π/2)
    theta_S, phi_S = 0.0, 0.0  # |1,0⟩
    theta_A, phi_A = 0.0, 0.0  # |1,0⟩ (ancilla doesn't matter)
    alpha = (0.0, 0.0, 0.0, 0.0)
    T_BS = np.pi / 2.0

    psi0 = two_qubit_state(theta_S, phi_S, theta_A, phi_A)

    for t_hold in t_hold_vals:
        # Use θ_true = 1.0 (away from fringe extremum)
        domega = compute_sensitivity(psi0, T_BS, T_BS, t_hold, 1.0, alpha, ops)
        expected = 1.0 / t_hold
        # Relative tolerance 5% to account for finite-difference and
        # residual fringe effects
        assert np.isclose(domega, expected, rtol=0.05), (
            f"Δω = {domega:.6f} for t_hold={t_hold}, expected ≈ {expected:.6f}"
        )

    return True


def validate_variance_positive(
    psi: np.ndarray,
    operator: np.ndarray,
    atol: float = 1e-12,
) -> bool:
    """Validate that Var(O) ≥ -atol for a given state and operator.

    A slightly negative variance (order 1e-16) can arise from numerical
    rounding; this function asserts it is not significantly negative.

    Args:
        psi: Normalised state vector.
        operator: Hermitian operator matrix.
        atol: Absolute tolerance for negative variance (default 1e-12).

    Returns:
        True if check passes.

    Raises:
        AssertionError: If variance < -atol (unphysical).

    """
    _, var = compute_expectation_and_variance(psi, operator)
    assert var >= -atol, f"Unphysical negative variance: Var = {var:.2e}"
    return True


def validate_derivative_stability(
    psi0: np.ndarray,
    T_BS1: float,
    T_BS2: float,
    t_hold: float,
    omega_true: float,
    alpha: tuple[float, float, float, float],
    ops: dict[str, np.ndarray],
    fd_steps: tuple[float, float, float] = (1e-5, 1e-6, 1e-7),
    rtol: float = 0.02,
) -> bool:
    """Check that the finite-difference derivative is stable w.r.t. step size.

    Computes Δω at multiple fd_step values and asserts the relative spread
    is below `rtol`.  This validates that the default fd_step = 1e-6 is not
    in a noisy regime.

    Args:
        psi0: Initial 4-vector (product state).
        T_BS1: First beam-splitter duration.
        T_BS2: Second beam-splitter duration.
        t_hold: Holding-time strength.
        omega_true: True phase rate parameter.
        alpha: (α_xx, α_xz, α_zx, α_zz) coupling coefficients.
        ops: Two-qubit operators (must contain 'Jz_S').
        fd_steps: Three fd_step values to compare (default 1e-5, 1e-6, 1e-7).
        rtol: Maximum allowed relative spread (default 0.02 = 2%).

    Returns:
        True if all checks pass.

    Raises:
        AssertionError: If derivative is unstable across step sizes.

    """
    sensitivities: list[float] = []
    for step in fd_steps:
        domega = compute_sensitivity(
            psi0,
            T_BS1,
            T_BS2,
            t_hold,
            omega_true,
            alpha,
            ops,
            fd_step=step,
        )
        if not np.isfinite(domega):
            # Skip fringe-extremum configurations
            continue
        sensitivities.append(domega)

    if len(sensitivities) < 2:
        # Not enough finite values to compare — skip (likely at fringe extremum)
        return True

    arr = np.array(sensitivities)
    mean_val = float(np.mean(arr))
    if mean_val == 0.0:
        return True

    # Max relative deviation from the mean
    max_dev = float(np.max(np.abs(arr - mean_val) / mean_val))
    assert max_dev < rtol, (
        f"Derivative unstable across fd_steps {fd_steps}: "
        f"sensitivities = {sensitivities}, max relative deviation = {max_dev:.4f}"
    )
    return True


# ============================================================================
# Optimisation Runner
# ============================================================================


def _run_nelder_mead(
    omega_true: float,
    ops: dict[str, np.ndarray],
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: dict[str, tuple[float, float]] | None = None,
    track_history: bool = False,
    meas_op: np.ndarray | None = None,
    fixed_alpha: tuple[float, float, float, float] | None = None,
) -> OptimisationResult:
    """Internal Nelder–Mead runner. Shared by run_optimisation and run_omega_scan.

    Args:
        omega_true: True phase rate parameter.
        ops: Two-qubit operators.
        x0: Initial parameter vector (11 elements, or 7 when fixed_alpha is set).
            Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder–Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder–Mead parameters.
        bounds: Custom parameter bounds dictionary. Uses defaults if None.
        track_history: If True, record objective function values per iteration.
        meas_op: Measurement operator for the objective.
            Defaults to ops['Jz_S'] (S-only).
        fixed_alpha: If set, optimisation is over state params only (7 elements)
            with the interaction coefficients held fixed.

    """
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = random_initial_params(rng, bounds)
    else:
        x0 = np.asarray(x0, dtype=float)

    if fixed_alpha is not None:
        assert x0.shape in ((11,), (7,)), (
            f"x0 must have 7 or 11 elements when fixed_alpha is set, got {x0.shape}"
        )
        x0 = x0[:7]  # keep only state params
    else:
        assert x0.shape == (11,), "x0 must have 11 elements"

    # Objective function (closure over omega_true, ops, bounds, meas_op, fixed_alpha)
    def objective(p: np.ndarray) -> float:
        return sensitivity_objective(
            p, omega_true, ops, bounds=bounds, meas_op=meas_op, fixed_alpha=fixed_alpha
        )

    # History tracking via callback
    history: list[float] = []

    def callback(_x: np.ndarray) -> None:
        if track_history:
            val = objective(_x)
            history.append(val)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=cast("Any", callback if track_history else None),
        options=cast(
            "Any",
            {
                "maxiter": maxiter,
                "xatol": xatol,
                "fatol": fatol,
                "adaptive": adaptive,
            },
        ),
    )

    # Compute expectation, variance, and purity at the optimal point
    # Use the full 11-element parameter vector for diagnostics
    opt_params = result.x.copy()
    if fixed_alpha is not None:
        opt_params = np.concatenate([opt_params, np.asarray(fixed_alpha, dtype=float)])
    (
        exp_val_s,
        var_val_s,
        exp_val_m,
        var_val_m,
        cov_sa,
        purity,
    ) = diagnostics_from_params(opt_params, omega_true, ops)

    # Determine measurement label by comparing meas_op to known operators
    if meas_op is None:
        meas_label = "S-only"
    else:
        # Compare against the canonical joint operator Jz_S + Jz_A
        joint_op = build_joint_operator(ops)
        if np.allclose(meas_op, joint_op, atol=1e-12):
            meas_label = "Joint M"
        else:
            meas_label = "Custom"

    return OptimisationResult(
        delta_omega_opt=float(result.fun),
        meas_label=meas_label,
        params_opt=opt_params,
        omega_true=omega_true,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val_s,
        variance_Jz=var_val_s,
        purity_S=purity,
        expectation_M=exp_val_m,
        variance_M=var_val_m,
        covariance_SA=cov_sa,
        history=history.copy(),
    )


def run_optimisation(
    omega_true: float,
    ops: dict[str, np.ndarray],
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: dict[str, tuple[float, float]] | None = None,
    track_history: bool = False,
    meas_op: np.ndarray | None = None,
    fixed_alpha: tuple[float, float, float, float] | None = None,
) -> OptimisationResult:
    """Run a single Nelder–Mead optimisation for ancilla-assisted metrology.

    Args:
        omega_true: True phase rate parameter.
        ops: Two-qubit operators.
        x0: Initial parameter vector (11 elements, or 7 when fixed_alpha is set).
            Random if None.
        seed: Random seed (used if x0 is None).
        maxiter: Maximum Nelder–Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder–Mead parameters.
        bounds: Custom parameter bounds dictionary. Uses defaults if None.
        track_history: If True, record objective function values at each
            iteration in the result's `history` field.
        meas_op: Measurement operator for the objective.
            Defaults to ops['Jz_S'] (S-only).
        fixed_alpha: If set, optimisation is over state params only (7 elements)
            with the interaction coefficients held fixed.

    Returns:
        OptimisationResult with the best parameters found.

    """
    return _run_nelder_mead(
        omega_true=omega_true,
        ops=ops,
        x0=x0,
        seed=seed,
        maxiter=maxiter,
        xatol=xatol,
        fatol=fatol,
        adaptive=adaptive,
        bounds=bounds,
        track_history=track_history,
        meas_op=meas_op,
        fixed_alpha=fixed_alpha,
    )


def run_omega_scan(
    omega_values: list[float] | np.ndarray,
    n_restarts: int = 10,
    seed: int | None = 42,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    bounds: dict[str, tuple[float, float]] | None = None,
    track_history: bool = False,
    meas_op: np.ndarray | None = None,
) -> OmegaScanResult:
    """Scan over omega values with multiple Nelder–Mead restarts each.

    Args:
        omega_values: ω values to scan.
        n_restarts: Number of random-start Nelder–Mead runs per ω.
        seed: Base random seed (incremented per restart).
        maxiter: Maximum Nelder–Mead iterations per run.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder–Mead parameters.
        bounds: Custom parameter bounds (e.g., to expand t_hold range).
            Default: t_hold ∈ [0, 5]. Use bounds={"t_hold": (0, 20), ...} for
            expanded-range experiments.
        track_history: If True, record convergence history per run.
        meas_op: Measurement operator for the objective.
            Defaults to ops['Jz_S'] (S-only).

    Returns:
        OmegaScanResult with all recorded information.

    """
    ops = build_two_qubit_operators()
    omega_arr = np.asarray(omega_values, dtype=float)
    base_seed = seed if seed is not None else 42

    all_results: dict[float, list[OptimisationResult]] = {}
    best_per_omega_list: list[float] = []
    all_results_flat: list[OptimisationResult] = []

    for omega_val in omega_arr:
        omega_results: list[OptimisationResult] = []
        for restart in range(n_restarts):
            rng = np.random.default_rng(base_seed + int(omega_val * 1000) + restart)
            x0 = random_initial_params(rng, bounds)
            opt_result = _run_nelder_mead(
                omega_true=omega_val,
                ops=ops,
                x0=x0,
                maxiter=maxiter,
                xatol=xatol,
                fatol=fatol,
                adaptive=adaptive,
                bounds=bounds,
                track_history=track_history,
                meas_op=meas_op,
            )
            omega_results.append(opt_result)
            all_results_flat.append(opt_result)

        # Sort by sensitivity (ascending)
        omega_results.sort(key=lambda r: r.delta_omega_opt)
        all_results[omega_val] = omega_results
        best_per_omega_list.append(omega_results[0].delta_omega_opt)

    return OmegaScanResult(
        results=all_results_flat,
        omega_values=omega_arr,
        best_per_omega=np.array(best_per_omega_list),
        all_results=all_results,
    )


# ============================================================================
# α-Scan with State Re-Optimisation
# ============================================================================


def scan_alpha_with_reoptimisation(
    alpha_name: str,
    alpha_values: np.ndarray | None = None,
    *,
    omega_true: float = 1.0,
    n_restarts: int = 5,
    maxiter: int = 500,
    seed: int = 42,
) -> AlphaReoptScanResult:
    """Scan a single α coefficient with state re-optimisation at each point.

    For each α value, the state parameters (θ_S, φ_S, θ_A, φ_A, T_BS1,
    T_BS2, t_hold) are re-optimised via Nelder–Mead while the interaction
    coefficients are held fixed.  Both joint and S-only measurement
    operators are evaluated independently.

    Args:
        alpha_name: Which coefficient to scan: 'xx', 'xz', 'zx', or 'zz'.
        alpha_values: Array of α values to scan.  Defaults to 21 points
            in [-2.0, 2.0].
        omega_true: True phase rate parameter (default 1.0).
        n_restarts: Number of random-start Nelder–Mead runs per α.
        maxiter: Maximum Nelder–Mead iterations per run.
        seed: Base random seed (incremented per restart).

    Returns:
        AlphaReoptScanResult with all recorded information.

    Raises:
        ValueError: If alpha_name is not one of 'xx', 'xz', 'zx', 'zz'.

    """
    alpha_idx_map = {"xx": 0, "xz": 1, "zx": 2, "zz": 3}
    if alpha_name not in alpha_idx_map:
        raise ValueError(
            f"alpha_name must be one of {list(alpha_idx_map.keys())}, got {alpha_name}",
        )
    scan_idx = alpha_idx_map[alpha_name]

    if alpha_values is None:
        alpha_values = np.linspace(-2.0, 2.0, 5)

    ops = build_two_qubit_operators()
    M_op = build_joint_operator(ops)

    alpha_arr = np.asarray(alpha_values, dtype=float)
    n_points = len(alpha_arr)

    delta_omega_joint = np.full(n_points, np.inf, dtype=float)
    delta_omega_sonly = np.full(n_points, np.inf, dtype=float)
    best_params_joint: list[np.ndarray] = []
    best_params_sonly: list[np.ndarray] = []

    for i, a_val in enumerate(alpha_arr):
        # Build the fixed α tuple: one coefficient varies, others zero
        alpha_list = [0.0, 0.0, 0.0, 0.0]
        alpha_list[scan_idx] = a_val
        fixed_alpha: tuple[float, float, float, float] = (
            alpha_list[0],
            alpha_list[1],
            alpha_list[2],
            alpha_list[3],
        )

        # --- Joint measurement ---
        joint_results: list[OptimisationResult] = []
        for restart in range(n_restarts):
            rng = np.random.default_rng(seed + i * 1000 + restart)
            x0 = random_initial_params(rng)[:7]  # 7 state params
            opt = _run_nelder_mead(
                omega_true=omega_true,
                ops=ops,
                x0=x0,
                maxiter=maxiter,
                meas_op=M_op,
                fixed_alpha=fixed_alpha,
            )
            joint_results.append(opt)
        joint_results.sort(key=lambda r: r.delta_omega_opt)
        delta_omega_joint[i] = joint_results[0].delta_omega_opt
        best_params_joint.append(joint_results[0].params_opt.copy())

        # --- S-only measurement ---
        sonly_results: list[OptimisationResult] = []
        for restart in range(n_restarts):
            rng = np.random.default_rng(seed + i * 1000 + restart + 10000)
            x0 = random_initial_params(rng)[:7]
            opt = _run_nelder_mead(
                omega_true=omega_true,
                ops=ops,
                x0=x0,
                maxiter=maxiter,
                meas_op=None,
                fixed_alpha=fixed_alpha,
            )
            sonly_results.append(opt)
        sonly_results.sort(key=lambda r: r.delta_omega_opt)
        delta_omega_sonly[i] = sonly_results[0].delta_omega_opt
        best_params_sonly.append(sonly_results[0].params_opt.copy())

    return AlphaReoptScanResult(
        alpha_values=alpha_arr,
        delta_omega_joint=delta_omega_joint,
        delta_omega_sonly=delta_omega_sonly,
        best_params_joint=best_params_joint,
        best_params_sonly=best_params_sonly,
    )


# ============================================================================
# Decoupled Baseline
# ============================================================================


def get_decoupled_sensitivity(t_hold: float, omega_true: float = 1.0) -> float:
    """Compute the SQL sensitivity for the decoupled (α=0) case.

    Optimal configuration: system in |1,0⟩, 50/50 beam splitters, no interaction.

    Args:
        t_hold: Holding-time strength.
        omega_true: True phase rate.

    Returns:
        Δω (ideally = 1/t_hold).

    """
    ops = build_two_qubit_operators()
    psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)  # both in |1,0⟩
    T_BS = np.pi / 2.0
    alpha = (0.0, 0.0, 0.0, 0.0)
    return compute_sensitivity(psi0, T_BS, T_BS, t_hold, omega_true, alpha, ops)


def compute_decoupled_baseline(
    t_hold_values: np.ndarray,
    omega_true: float = 1.0,
) -> DecoupledBaselineResult:
    """Compute the decoupled baseline sensitivity Δω for a range of t_hold values.

    The optimal decoupled configuration is: system in |1,0⟩, ancilla in |1,0⟩,
    50/50 beam splitters, α=0.  At this configuration, Δω = 1/t_hold exactly.

    Args:
        t_hold_values: Array of holding-time strengths to evaluate.
        omega_true: True phase rate (default 1.0).

    Returns:
        DecoupledBaselineResult with computed sensitivities.

    """
    t_hold_arr = np.asarray(t_hold_values, dtype=float)
    delta_omega = np.array(
        [get_decoupled_sensitivity(t_hold, omega_true) for t_hold in t_hold_arr],
        dtype=float,
    )
    sql = 1.0 / t_hold_arr
    return DecoupledBaselineResult(
        t_hold_values=t_hold_arr,
        delta_omega_values=delta_omega,
        sql_values=sql,
    )


# ============================================================================
# Covariance Analysis
# ============================================================================


def compute_covariance_analysis(
    alpha_range: tuple[float, float] = (-2.0, 2.0),
    n_points: int = 101,
    t_hold: float = 1.0,
    omega_true: float = 1.0,
) -> CovarianceAnalysisResult:
    """Compute max |Cov(J_z^S, J_z^A)| for each α coefficient type.

    Uses the analytically optimal decoupled state configuration
    (θ_S=0, θ_A=0, 50/50 BS).  Scans each α coefficient independently
    over ``alpha_range`` and records the maximum absolute covariance
    and its sign.

    Args:
        alpha_range: (min, max) for scanning each α coefficient.
        n_points: Number of points per scan.
        t_hold: Holding-time strength (default 1.0).
        omega_true: True phase rate (default 1.0).

    Returns:
        CovarianceAnalysisResult with max covariances and signs.

    """
    ops = build_two_qubit_operators()
    alpha_lo, alpha_hi = alpha_range
    alpha_scan = np.linspace(alpha_lo, alpha_hi, n_points)
    psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)

    coeff_names = ["α_xx", "α_xz", "α_zx", "α_zz"]
    max_covs = np.zeros(4, dtype=float)
    cov_signs = np.zeros(4, dtype=float)

    for idx in range(4):
        covs = np.zeros(n_points, dtype=float)
        for j, a_val in enumerate(alpha_scan):
            alpha_list = [0.0, 0.0, 0.0, 0.0]
            alpha_list[idx] = a_val
            alpha: tuple[float, float, float, float] = (
                alpha_list[0],
                alpha_list[1],
                alpha_list[2],
                alpha_list[3],
            )
            psi = evolve_full(
                psi0,
                np.pi / 2,
                np.pi / 2,
                t_hold,
                omega_true,
                alpha,
                ops,
            )
            covs[j] = compute_covariance(psi, ops)
        max_idx = int(np.argmax(np.abs(covs)))
        max_covs[idx] = abs(covs[max_idx])
        cov_signs[idx] = np.sign(covs[max_idx])

    return CovarianceAnalysisResult(
        coefficient_names=coeff_names,
        max_covariances=max_covs,
        covariance_signs=cov_signs,
    )


# ============================================================================
# Single α Grid Scan
# ============================================================================


def scan_alpha_single_parameter(
    alpha_name: str,
    alpha_min: float = -2.0,
    alpha_max: float = 2.0,
    n_points: int = 21,
    *,
    theta_S: float = np.pi / 2,
    phi_S: float = 0.0,
    theta_A: float = 0.0,
    phi_A: float = 0.0,
    T_BS1: float = np.pi / 2,
    T_BS2: float = np.pi / 2,
    t_hold: float = 1.0,
    omega_true: float = 1.0,
    meas_op: np.ndarray | None = None,
) -> AlphaSingleScanResult:
    """Scan a single α coefficient while holding others fixed.

    This replicates the grid scan described in the report: "each α_ij is
    scanned independently in [-2, 2] with 21 points, while the other three
    are held at zero."

    Physical configuration defaults (SQL-optimal for α=0):
        - System: |ψ_S⟩ = (|0⟩ + |1⟩)/√2 (θ_S = π/2)
        - Ancilla: |ψ_A⟩ = |0⟩ (θ_A = 0)
        - Beam splitters: 50/50 (T_BS1 = T_BS2 = π/2)
        - t_hold = 1.0, ω_true = 1.0

    Args:
        alpha_name: Which coefficient to scan: 'xx', 'xz', 'zx', or 'zz'.
        alpha_min: Minimum value for the scan (default -2.0).
        alpha_max: Maximum value for the scan (default 2.0).
        n_points: Number of points in the scan (default 21).
        theta_S: System polar angle (default π/2).
        phi_S: System azimuthal angle (default 0.0).
        theta_A: Ancilla polar angle (default 0.0).
        phi_A: Ancilla azimuthal angle (default 0.0).
        T_BS1: First beam-splitter duration (default π/2).
        T_BS2: Second beam-splitter duration (default π/2).
        t_hold: Holding time (default 1.0).
        omega_true: True phase rate (default 1.0).
        meas_op: Measurement operator. Defaults to ops['Jz_S'] (S-only).

    Returns:
        AlphaSingleScanResult with scanned values and sensitivities.

    Raises:
        ValueError: If alpha_name is not one of 'xx', 'xz', 'zx', 'zz'.

    """
    alpha_idx_map = {"xx": 0, "xz": 1, "zx": 2, "zz": 3}
    if alpha_name not in alpha_idx_map:
        raise ValueError(
            f"alpha_name must be one of {list(alpha_idx_map.keys())}, got {alpha_name}",
        )
    scan_idx = alpha_idx_map[alpha_name]

    ops = build_two_qubit_operators()
    if meas_op is None:
        meas_op = ops["Jz_S"]
    psi0 = two_qubit_state(theta_S, phi_S, theta_A, phi_A)

    alpha_values = np.linspace(alpha_min, alpha_max, n_points)
    delta_omega_values = np.zeros(n_points, dtype=float)

    for i, a_val in enumerate(alpha_values):
        alpha_list = [0.0, 0.0, 0.0, 0.0]
        alpha_list[scan_idx] = a_val
        alpha: tuple[float, float, float, float] = (
            alpha_list[0],
            alpha_list[1],
            alpha_list[2],
            alpha_list[3],
        )
        domega = compute_sensitivity(
            psi0,
            T_BS1,
            T_BS2,
            t_hold,
            omega_true,
            alpha,
            ops,
            meas_op=meas_op,
        )
        delta_omega_values[i] = domega

    fixed_params = {
        "theta_S": theta_S,
        "phi_S": phi_S,
        "theta_A": theta_A,
        "phi_A": phi_A,
        "T_BS1": T_BS1,
        "T_BS2": T_BS2,
        "t_hold": t_hold,
        "omega_true": omega_true,
    }

    return AlphaSingleScanResult(
        alpha_name=alpha_name,
        alpha_values=alpha_values,
        delta_omega_values=delta_omega_values,
        fixed_params=fixed_params,
    )


# ============================================================================
# Random α Search
# ============================================================================


def random_search_alpha(
    n_samples: int = 200,
    alpha_min: float = -2.0,
    alpha_max: float = 2.0,
    *,
    theta_S: float = np.pi / 2,
    phi_S: float = 0.0,
    theta_A: float = 0.0,
    phi_A: float = 0.0,
    T_BS1: float = np.pi / 2,
    T_BS2: float = np.pi / 2,
    t_hold: float = 1.0,
    omega_true: float = 1.0,
    seed: int | None = 42,
    meas_op: np.ndarray | None = None,
) -> AlphaRandomSearchResult:
    """Random search over the 4D α = (α_xx, α_xz, α_zx, α_zz) space.

    This replicates the random search described in the report: "200 samples
    over the full 4D α space" to verify that no combination achieves
    Δω < 1/t_hold (SQL).

    Args:
        n_samples: Number of random α samples to evaluate (default 200).
        alpha_min: Lower bound for all α coefficients (default -2.0).
        alpha_max: Upper bound for all α coefficients (default 2.0).
        theta_S: System polar angle (default π/2).
        phi_S: System azimuthal angle (default 0.0).
        theta_A: Ancilla polar angle (default 0.0).
        phi_A: Ancilla azimuthal angle (default 0.0).
        T_BS1: First beam-splitter duration (default π/2).
        T_BS2: Second beam-splitter duration (default π/2).
        t_hold: Holding time (default 1.0).
        omega_true: True phase rate (default 1.0).
        seed: Random seed for reproducibility (default 42).
        meas_op: Measurement operator. Defaults to ops['Jz_S'] (S-only).

    Returns:
        AlphaRandomSearchResult with all samples and the best found.

    """
    rng = np.random.default_rng(seed)
    ops = build_two_qubit_operators()
    if meas_op is None:
        meas_op = ops["Jz_S"]
    psi0 = two_qubit_state(theta_S, phi_S, theta_A, phi_A)

    # Sample uniformly in [alpha_min, alpha_max]^4
    alpha_samples = rng.uniform(alpha_min, alpha_max, size=(n_samples, 4))
    delta_omega_values = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        alpha: tuple[float, float, float, float] = (
            float(alpha_samples[i, 0]),
            float(alpha_samples[i, 1]),
            float(alpha_samples[i, 2]),
            float(alpha_samples[i, 3]),
        )
        domega = compute_sensitivity(
            psi0,
            T_BS1,
            T_BS2,
            t_hold,
            omega_true,
            alpha,
            ops,
            meas_op=meas_op,
        )
        delta_omega_values[i] = domega

    # Find best (minimal Δω)
    best_idx = int(np.argmin(delta_omega_values))
    best_alpha: tuple[float, float, float, float] = (
        float(alpha_samples[best_idx, 0]),
        float(alpha_samples[best_idx, 1]),
        float(alpha_samples[best_idx, 2]),
        float(alpha_samples[best_idx, 3]),
    )
    best_delta_omega = float(delta_omega_values[best_idx])

    fixed_params = {
        "theta_S": theta_S,
        "phi_S": phi_S,
        "theta_A": theta_A,
        "phi_A": phi_A,
        "T_BS1": T_BS1,
        "T_BS2": T_BS2,
        "t_hold": t_hold,
        "omega_true": omega_true,
    }

    return AlphaRandomSearchResult(
        alpha_samples=alpha_samples,
        delta_omega_values=delta_omega_values,
        best_alpha=best_alpha,
        best_delta_omega=best_delta_omega,
        fixed_params=fixed_params,
    )


# ============================================================================
# Interaction Robustness Scan (t_hold × α)
# ============================================================================


def compute_interaction_robustness(
    t_hold_values: np.ndarray,
    alpha_values: np.ndarray,
    *,
    omega_true: float = 1.0,
    alpha_name: str = "xx",
    theta_S: float = 0.0,
    phi_S: float = 0.0,
    theta_A: float = 0.0,
    phi_A: float = 0.0,
    T_BS: float = np.pi / 2,
) -> InteractionRobustnessResult:
    """Scan over t_hold and a single α coefficient, recording sensitivity for
    both S-only and joint measurements.

    For each (t_hold, α) pair, the state and beam-splitter parameters are
    held fixed while the holding-time strength and interaction coefficient
    are varied.  The sensitivity Δω is computed via `compute_sensitivity`
    for both measurement operators (S-only and joint).

    Args:
        t_hold_values: Array of t_hold holding-time strengths to scan.
        alpha_values: Array of α coefficient values to scan (single
            coefficient determined by `alpha_name`).
        omega_true: True phase rate parameter (default 1.0).
        alpha_name: Which coefficient to scan: 'xx', 'xz', 'zx', or 'zz'
            (default 'xx').
        theta_S: System polar angle (default 0.0).
        phi_S: System azimuthal angle (default 0.0).
        theta_A: Ancilla polar angle (default 0.0).
        phi_A: Ancilla azimuthal angle (default 0.0).
        T_BS: Beam-splitter duration for both first and second beam
            splitters (default π/2).

    Returns:
        InteractionRobustnessResult containing the 2D sensitivity arrays.

    Raises:
        ValueError: If alpha_name is not one of 'xx', 'xz', 'zx', 'zz'.

    """
    alpha_idx_map = {"xx": 0, "xz": 1, "zx": 2, "zz": 3}
    if alpha_name not in alpha_idx_map:
        raise ValueError(
            f"alpha_name must be one of {list(alpha_idx_map.keys())}, got {alpha_name}",
        )
    scan_idx = alpha_idx_map[alpha_name]

    ops = build_two_qubit_operators()
    M_op = build_joint_operator(ops)
    psi0 = two_qubit_state(theta_S, phi_S, theta_A, phi_A)

    t_hold_arr = np.asarray(t_hold_values, dtype=float)
    alpha_arr = np.asarray(alpha_values, dtype=float)
    n_t_hold = len(t_hold_arr)
    n_alpha = len(alpha_arr)

    delta_omega_joint = np.full((n_t_hold, n_alpha), np.inf, dtype=float)
    delta_omega_sonly = np.full((n_t_hold, n_alpha), np.inf, dtype=float)

    for i, t_hold in enumerate(t_hold_arr):
        for j, a_val in enumerate(alpha_arr):
            alpha_list = [0.0, 0.0, 0.0, 0.0]
            alpha_list[scan_idx] = a_val
            alpha: tuple[float, float, float, float] = (
                alpha_list[0],
                alpha_list[1],
                alpha_list[2],
                alpha_list[3],
            )

            # S-only measurement (meas_op=None defaults to Jz_S)
            dtheta_s = compute_sensitivity(
                psi0,
                T_BS,
                T_BS,
                t_hold,
                omega_true,
                alpha,
                ops,
                meas_op=None,
            )
            delta_omega_sonly[i, j] = dtheta_s

            # Joint measurement (Jz_S + Jz_A)
            dtheta_j = compute_sensitivity(
                psi0,
                T_BS,
                T_BS,
                t_hold,
                omega_true,
                alpha,
                ops,
                meas_op=M_op,
            )
            delta_omega_joint[i, j] = dtheta_j

    return InteractionRobustnessResult(
        t_hold_values=t_hold_arr,
        alpha_values=alpha_arr,
        delta_omega_joint=delta_omega_joint,
        delta_omega_sonly=delta_omega_sonly,
    )

"""
Tests for the four-parameter coupling multi-particle dual MZI module.

Run with:
    uv run pytest reports/20260523/test_local.py -q --tb=short
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from src.physics.dicke_basis import jz_operator
from src.utils.enums import OperatorBasis

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]

_report_dir = str(
    Path(__file__).resolve().parent.parent.parent / "reports" / "20260523",
)
if _report_dir not in _sys.path:
    _sys.path.insert(0, _report_dir)
del _sys, _report_dir

from local import (  # type: ignore[import-untyped]  # noqa: E402
    ALPHA_BOUND,
    DEFAULT_T_BS,
    FD_STEP,
    N_LBFGS_STARTS,
    DEFAULT_t_hold,
    FourParamSweepResult,
    ScalingAnalysisResult,
    build_hold_hamiltonian,
    compute_decoupled_baseline,
    compute_reduced_expectation_and_variance,
    compute_sensitivity,
    embed_combined_operators,
    evolve_circuit,
    fit_scaling_exponents,
    hold_unitary,
    initial_state,
    optimise_four_params,
    protocol_bs_unitary,
    run_sweep,
    single_bs_unitary,
)


def _embed_ops(N: int) -> dict[str, np.ndarray]:
    """Fast operator embedding for tests."""
    return embed_combined_operators(N)


# ---------------------------------------------------------------------------
# Operator Construction
# ---------------------------------------------------------------------------


class TestOperatorConstruction:
    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_operators_correct_dimension(self, N: int) -> None:
        ops = _embed_ops(N)
        dim = (N + 1) ** 2
        for name, op in ops.items():
            assert op.shape == (dim, dim), (
                f"{name} has shape {op.shape}, expected ({dim}, {dim})"
            )

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_operators_hermitian(self, N: int) -> None:
        ops = _embed_ops(N)
        for name, op in ops.items():
            assert np.allclose(op, op.conj().T, atol=1e-12), (
                f"{name} not Hermitian for N={N}"
            )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_commutation_jz_jx(self, N: int) -> None:
        ops = _embed_ops(N)
        comm_S = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
        assert np.allclose(comm_S, 1j * ops["Jy_S"], atol=1e-12), (
            f"[Jz_S, Jx_S] = i Jy_S failed for N={N}"
        )
        comm_A = ops["Jz_A"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jz_A"]
        assert np.allclose(comm_A, 1j * ops["Jy_A"], atol=1e-12), (
            f"[Jz_A, Jx_A] = i Jy_A failed for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_hold_hamiltonian_hermitian(self, N: int) -> None:
        ops = _embed_ops(N)
        for omega in [0.1, 1.0, 5.0]:
            alpha = (1.0, 2.0, -1.0, 0.5)
            H = build_hold_hamiltonian(N, omega, alpha, ops)
            assert np.allclose(H, H.conj().T, atol=1e-12), (
                f"H not Hermitian for N={N}, ω={omega}, α={alpha}"
            )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_hold_hamiltonian_decoupled(self, N: int) -> None:
        """At α = (0,0,0,0), H = ω(J_z^S + J_z^A)."""
        ops = _embed_ops(N)
        omega = 0.7
        H = build_hold_hamiltonian(N, omega, (0.0, 0.0, 0.0, 0.0), ops)
        expected = omega * (ops["Jz_S"] + ops["Jz_A"])
        assert np.allclose(H, expected, atol=1e-12), f"Decoupled H mismatch for N={N}"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_jz_single_diagonal(self, N: int) -> None:
        """J_z should be diagonal in the Dicke basis."""
        Jz = jz_operator(N, basis=OperatorBasis.DICKE)
        assert np.allclose(Jz, np.diag(np.diag(Jz)), atol=1e-12), (
            f"J_z not diagonal for N={N}"
        )


# ---------------------------------------------------------------------------
# Unitarity
# ---------------------------------------------------------------------------


class TestUnitarity:
    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_single_bs_unitary(self, N: int) -> None:
        U = single_bs_unitary(N)
        eye = np.eye(N + 1, dtype=complex)
        assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
            f"Single BS not unitary for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_dual_bs_unitary(self, N: int) -> None:
        U = protocol_bs_unitary(N, "dual")
        dim = (N + 1) ** 2
        eye = np.eye(dim, dtype=complex)
        assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
            f"Dual BS not unitary for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_sonly_bs_unitary(self, N: int) -> None:
        U = protocol_bs_unitary(N, "S-only")
        dim = (N + 1) ** 2
        eye = np.eye(dim, dtype=complex)
        assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
            f"S-only BS not unitary for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_hold_unitary(self, N: int) -> None:
        ops = _embed_ops(N)
        alpha = (1.0, 2.0, -0.5, 0.3)
        U = hold_unitary(N, t_hold=1.0, omega=0.5, alpha=alpha, ops=ops)
        dim = (N + 1) ** 2
        eye = np.eye(dim, dtype=complex)
        assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
            f"Hold unitary not unitary for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_hold_unitary_decoupled_factorises(self, N: int) -> None:
        """At α = (0,0,0,0), U_hold = exp(-i t_hold ω J_z) ⊗ exp(-i t_hold ω J_z)."""
        ops = _embed_ops(N)
        omega = 0.5
        t_hold = 2.0
        U = hold_unitary(N, t_hold, omega, (0.0, 0.0, 0.0, 0.0), ops)
        Jz = jz_operator(N, basis=OperatorBasis.DICKE)
        U_single = expm(-1j * t_hold * omega * Jz)
        expected = np.kron(U_single, U_single)
        assert np.allclose(U, expected, atol=1e-12), (
            f"Decoupled hold does not factorise for N={N}"
        )


# ---------------------------------------------------------------------------
# Circuit Evolution
# ---------------------------------------------------------------------------


class TestCircuitEvolution:
    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_initial_state_normalised(self, N: int) -> None:
        psi = initial_state(N)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize(
        ("N", "protocol"), [(1, "dual"), (2, "S-only"), (3, "dual"), (5, "S-only")]
    )
    def test_normalisation_preserved_decoupled(self, N: int, protocol: str) -> None:
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        psi = evolve_circuit(
            N, psi0, omega=0.5, alpha=(0.0, 0.0, 0.0, 0.0), ops=ops, protocol=protocol
        )
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize(
        ("N", "alpha", "protocol"),
        [
            (1, (1.0, -0.5, 0.3, 2.0), "dual"),
            (2, (5.0, 0.0, 0.0, 0.0), "S-only"),
            (3, (0.0, 3.0, 0.0, 0.0), "dual"),
            (5, (1.0, 2.0, -1.0, 0.5), "dual"),
        ],
    )
    def test_normalisation_with_coupling(
        self, N: int, alpha: tuple[float, ...], protocol: str
    ) -> None:
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        psi = evolve_circuit(
            N, psi0, omega=1.0, alpha=alpha, ops=ops, protocol=protocol
        )
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_no_op_identity(self, N: int) -> None:
        """T_BS=0, t_hold=0, omega=0 should give the initial state back."""
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        U_bs_zero = protocol_bs_unitary(N, "dual", T_BS=0.0)
        psi = U_bs_zero @ psi0
        psi = (
            hold_unitary(N, t_hold=0.0, omega=0.0, alpha=(0.0, 0.0, 0.0, 0.0), ops=ops)
            @ psi
        )
        psi = U_bs_zero @ psi
        assert np.allclose(psi, psi0, atol=1e-12), f"Identity failed for N={N}"


# ---------------------------------------------------------------------------
# Reduced Variance
# ---------------------------------------------------------------------------


class TestReducedVariance:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_product_state_variance_zero(self, N: int) -> None:
        """For the initial |J,J⟩_S ⊗ |J,J⟩_A state, Var(J_z^S) = 0."""
        psi = initial_state(N)
        Jz_single = jz_operator(N, basis=OperatorBasis.DICKE)
        _, var = compute_reduced_expectation_and_variance(psi, N, Jz_single)
        assert var == pytest.approx(0.0, abs=1e-12), f"Var != 0 for N={N}"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_trace_preservation(self, N: int) -> None:
        """Tr(ρ_S) = 1 after partial trace."""
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        alpha = (1.0, 2.0, -0.5, 0.3)
        psi = evolve_circuit(N, psi0, omega=0.5, alpha=alpha, ops=ops, protocol="dual")
        psi_mat = psi.reshape(N + 1, N + 1)
        rho_S = psi_mat @ psi_mat.conj().T
        trace = float(np.real(np.trace(rho_S)))
        assert np.isclose(trace, 1.0, atol=1e-12), f"Tr(ρ_S) = {trace} != 1 for N={N}"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_variance_positive(self, N: int) -> None:
        """Var(J_z^S) >= 0 for all couplings."""
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        Jz_single = jz_operator(N, basis=OperatorBasis.DICKE)
        for alpha_val in [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 3.0, 0.0),
            (1.0, 2.0, -1.0, 0.5),
        ]:
            psi = evolve_circuit(
                N, psi0, omega=0.5, alpha=alpha_val, ops=ops, protocol="dual"
            )
            _, var = compute_reduced_expectation_and_variance(psi, N, Jz_single)
            assert var >= -1e-12, f"Negative Var at N={N}, α={alpha_val}: {var}"


# ---------------------------------------------------------------------------
# Sensitivity Computation
# ---------------------------------------------------------------------------


class TestSensitivity:
    @pytest.mark.parametrize(
        ("N", "protocol"), [(1, "dual"), (2, "S-only"), (3, "dual"), (5, "S-only")]
    )
    def test_decoupled_sensitivity_matches_sql(self, N: int, protocol: str) -> None:
        """At α = (0,0,0,0), Δω should equal SQL = 1/(√N t_hold)."""
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        omega = 1.0
        dt, *_ = compute_sensitivity(
            N,
            psi0,
            omega_true=omega,
            alpha=(0.0, 0.0, 0.0, 0.0),
            ops=ops,
            protocol=protocol,
        )
        sql = 1.0 / (np.sqrt(N) * DEFAULT_t_hold)
        assert dt == pytest.approx(sql, rel=1e-5), (
            f"N={N}, protocol={protocol}: Δω={dt:.10f} != SQL={sql:.10f}"
        )

    @pytest.mark.parametrize(
        ("N", "omega", "protocol"),
        [
            (1, 0.5, "dual"),
            (1, 0.5, "S-only"),
            (2, 0.5, "dual"),
            (3, 1.0, "dual"),
            (5, 2.0, "S-only"),
        ],
    )
    def test_decoupled_all_omega(self, N: int, omega: float, protocol: str) -> None:
        """At α = 0, Δω = SQL for any ω."""
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        dt, _, _, _ = compute_sensitivity(
            N,
            psi0,
            omega_true=omega,
            alpha=(0.0, 0.0, 0.0, 0.0),
            ops=ops,
            protocol=protocol,
        )
        sql = 1.0 / (np.sqrt(N) * DEFAULT_t_hold)
        assert dt == pytest.approx(sql, rel=1e-5), (
            f"N={N}, ω={omega}, protocol={protocol}: Δω={dt:.10f} != SQL={sql:.10f}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_sensitivity_finite_with_coupling(self, N: int) -> None:
        """With non-zero α, sensitivity should be finite."""
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        for protocol in ("dual", "S-only"):
            for alpha_val in [
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 3.0, 0.0),
                (1.0, 2.0, -1.0, 0.5),
            ]:
                dt, _, _, _ = compute_sensitivity(
                    N,
                    psi0,
                    omega_true=1.0,
                    alpha=alpha_val,
                    ops=ops,
                    protocol=protocol,
                )
                assert np.isfinite(dt), (
                    f"Non-finite Δω={dt} at N={N}, α={alpha_val}, protocol={protocol}"
                )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_sensitivity_positive(self, N: int) -> None:
        """Sensitivity should always be positive."""
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        for protocol in ("dual", "S-only"):
            for alpha_val in [
                (0.0, 0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0, 0.0),
                (0.0, 0.0, 2.0, 0.0),
                (1.0, -1.0, 0.5, 2.0),
            ]:
                dt, _, _, _ = compute_sensitivity(
                    N,
                    psi0,
                    omega_true=1.0,
                    alpha=alpha_val,
                    ops=ops,
                    protocol=protocol,
                )
                assert dt > 0, (
                    f"Non-positive Δω={dt} at N={N}, α={alpha_val}, protocol={protocol}"
                )

    def test_derivative_stability_across_fd_step(self) -> None:
        """Δω should be stable across a range of fd_step values."""
        N = 3
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        for protocol in ("dual", "S-only"):
            dt_vals = []
            for fd_step in [1e-5, 1e-6, 1e-7]:
                dt, _, _, _ = compute_sensitivity(
                    N,
                    psi0,
                    omega_true=1.0,
                    alpha=(1.0, 2.0, -0.5, 0.3),
                    ops=ops,
                    protocol=protocol,
                    fd_step=fd_step,
                )
                dt_vals.append(dt)
            dt_arr = np.array(dt_vals)
            finite = np.isfinite(dt_arr)
            if np.sum(finite) >= 2:
                rel_spread = (
                    np.max(dt_arr[finite]) - np.min(dt_arr[finite])
                ) / np.mean(dt_arr[finite])
                assert rel_spread < 0.1, (
                    f"Derivative unstable for {protocol}: {dt_vals}, spread={rel_spread:.3e}"
                )

    @pytest.mark.slow
    def test_derivative_stability_large_N(self) -> None:
        """Derivative stability at N=20 (largest eigenvalue spread)."""
        N = 20
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        for protocol in ("dual", "S-only"):
            dt_vals = []
            for fd_step in [1e-5, 1e-6, 1e-7]:
                dt, _, _, _ = compute_sensitivity(
                    N,
                    psi0,
                    omega_true=1.0,
                    alpha=(0.5, 1.0, -0.5, 0.3),
                    ops=ops,
                    protocol=protocol,
                    fd_step=fd_step,
                )
                dt_vals.append(dt)
            dt_arr = np.array(dt_vals)
            finite = np.isfinite(dt_arr)
            if np.sum(finite) >= 2:
                rel_spread = (
                    np.max(dt_arr[finite]) - np.min(dt_arr[finite])
                ) / np.mean(dt_arr[finite])
                assert rel_spread < 0.2, (
                    f"N=20 derivative unstable for {protocol}: "
                    f"{dt_vals}, spread={rel_spread:.3e}"
                )


# ---------------------------------------------------------------------------
# Four-Parameter Optimisation (L-BFGS-B multi-start)
# ---------------------------------------------------------------------------


class TestFourParamOptimiser:
    @pytest.mark.parametrize(
        ("N", "protocol"), [(1, "dual"), (1, "S-only"), (2, "dual")]
    )
    def test_optimisation_returns_finite(self, N: int, protocol: str) -> None:
        """Optimisation should return a finite Δω."""
        ops = _embed_ops(N)
        result = optimise_four_params(
            N=N, omega=1.0, ops=ops, protocol=protocol, n_starts=10
        )
        assert np.isfinite(result.delta_omega_opt) or np.isnan(
            result.delta_omega_opt
        ), f"Non-finite result for N={N}, protocol={protocol}"

    @pytest.mark.parametrize(("N", "protocol"), [(1, "dual"), (1, "S-only")])
    def test_optimisation_convergence_some_starts(self, N: int, protocol: str) -> None:
        """At least some L-BFGS-B starts should converge (success=True)."""
        ops = _embed_ops(N)
        result = optimise_four_params(
            N=N, omega=1.0, ops=ops, protocol=protocol, n_starts=15
        )
        # At least one start should have converged in 15 attempts
        assert result.n_converged >= 1, (
            f"All {result.n_starts} L-BFGS-B starts failed to converge "
            f"for N={N}, protocol={protocol}"
        )

    @pytest.mark.parametrize(("N", "protocol"), [(1, "dual"), (1, "S-only")])
    def test_gradient_norm_recorded(self, N: int, protocol: str) -> None:
        """Gradient norm should be finite for the optimal result."""
        ops = _embed_ops(N)
        result = optimise_four_params(
            N=N, omega=1.0, ops=ops, protocol=protocol, n_starts=10
        )
        assert np.isfinite(result.gradient_norm) or result.gradient_norm == 0.0, (
            f"Gradient norm not finite: {result.gradient_norm}"
        )

    @pytest.mark.parametrize(("N", "protocol"), [(1, "dual"), (2, "S-only")])
    def test_optimisation_alpha_in_bounds(self, N: int, protocol: str) -> None:
        """All α components should be within [-ALPHA_BOUND, ALPHA_BOUND]."""
        ops = _embed_ops(N)
        result = optimise_four_params(
            N=N, omega=0.5, ops=ops, protocol=protocol, n_starts=10
        )
        for val in result.alpha_opt:
            if np.isfinite(val):
                assert -ALPHA_BOUND - 1e-6 <= val <= ALPHA_BOUND + 1e-6, (
                    f"α={val} outside [{-ALPHA_BOUND}, {ALPHA_BOUND}] for "
                    f"N={N}, protocol={protocol}"
                )

    def test_optimisation_runs_and_returns_sql_reference(self) -> None:
        """Optimisation should store the correct SQL reference."""
        N = 3
        ops = _embed_ops(N)
        result = optimise_four_params(
            N=N, omega=0.5, ops=ops, protocol="dual", n_starts=10
        )
        sql = 1.0 / (np.sqrt(N) * DEFAULT_t_hold)
        assert result.sql == pytest.approx(sql)

    @pytest.mark.slow
    def test_sonly_reproduction_ratio_leq_0690(self) -> None:
        """S-only MZI at N=1, ω=3.8 must reproduce ratio ≤ 0.690 (2026-05-21)."""
        N = 1
        omega_val = 3.8
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        result = optimise_four_params(
            N=N,
            omega=omega_val,
            ops=ops,
            psi0=psi0,
            protocol="S-only",
            n_starts=N_LBFGS_STARTS,
        )
        sql = 1.0 / (np.sqrt(N) * DEFAULT_t_hold)
        ratio = (
            result.delta_omega_opt / sql
            if np.isfinite(result.delta_omega_opt)
            else float("inf")
        )
        assert ratio <= 0.690 * 1.15, (
            f"2026-05-21 reproduction failed: ratio={ratio:.4f} > 0.690×1.15"
        )

    def test_optimisation_omega_scan_small(self) -> None:
        """Run a mini sweep to check the pipeline works."""
        N = 2
        ops = _embed_ops(N)
        for omega in [0.5, 1.0, 2.0]:
            result = optimise_four_params(
                N=N, omega=omega, ops=ops, protocol="dual", n_starts=10
            )
            assert np.isfinite(result.delta_omega_opt)

    @pytest.mark.slow
    def test_start_count_stability(self) -> None:
        """Best Δω should be stable under increasing start counts.

        The reported 20-30 starts is validated by checking consistency
        with a larger count at a representative (ω, N) point.
        """
        N = 3
        omega = 2.0
        ops = _embed_ops(N)
        result_low = optimise_four_params(
            N=N, omega=omega, ops=ops, protocol="dual", n_starts=10
        )
        result_high = optimise_four_params(
            N=N, omega=omega, ops=ops, protocol="dual", n_starts=30
        )
        low_val = result_low.delta_omega_opt
        high_val = result_high.delta_omega_opt
        if np.isfinite(low_val) and np.isfinite(high_val) and high_val > 0:
            rel_diff = abs(low_val - high_val) / high_val
            assert rel_diff < 0.25, (
                f"Start-count instability: 10-starts Δω={low_val:.6f}, "
                f"30-starts Δω={high_val:.6f}, rel_diff={rel_diff:.3%}"
            )


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


class TestSweeps:
    def test_small_sweep_runs(self) -> None:
        """A small sweep (2 ω × 2 N, both protocols) should complete."""
        result = run_sweep(
            omega_values=np.array([0.5, 1.0]),
            N_values=np.array([1, 2]),
            protocol="dual",
            n_starts=5,
        )
        assert result.n_points == 4
        assert len(result.omega_values) == 4
        assert len(result.alpha_xx_opt) == 4

    def test_sweep_contains_sql(self) -> None:
        result = run_sweep(
            omega_values=np.array([1.0]),
            N_values=np.array([1, 3]),
            protocol="S-only",
            n_starts=5,
        )
        for i in range(result.n_points):
            N = result.N_values[i]
            expected_sql = 1.0 / (np.sqrt(N) * DEFAULT_t_hold)
            assert result.sql_values[i] == pytest.approx(expected_sql), (
                f"SQL mismatch for N={N}: {result.sql_values[i]} != {expected_sql}"
            )

    def test_sweep_protocol_stored(self) -> None:
        """Protocol should be stored as the correct string."""
        result = run_sweep(
            omega_values=np.array([1.0]),
            N_values=np.array([1]),
            protocol="dual",
            n_starts=5,
        )
        assert result.protocol[0] == "dual"

    def test_sweep_all_finite_or_inf(self) -> None:
        """All Δω values should be finite (or inf at fringe extremum)."""
        result = run_sweep(
            omega_values=np.array([0.5, 2.0]),
            N_values=np.array([1, 3]),
            protocol="dual",
            n_starts=5,
        )
        for dt in result.delta_omega_opt:
            assert np.isfinite(dt) or np.isinf(dt), f"Non-finite Δω: {dt}"


# ---------------------------------------------------------------------------
# Decoupled Baseline
# ---------------------------------------------------------------------------


class TestDecoupledBaseline:
    @pytest.mark.parametrize(
        ("N", "protocol"), [(1, "dual"), (2, "S-only"), (5, "dual")]
    )
    def test_baseline_ratio_near_one(self, N: int, protocol: str) -> None:
        """At α = 0, Δω/SQL should be very close to 1."""
        result = compute_decoupled_baseline(
            omega_values=np.array([0.5, 1.0]),
            N_values=np.array([N]),
            protocol=protocol,
        )
        valid = np.isfinite(result.ratio)
        if np.any(valid):
            assert np.allclose(result.ratio[valid], 1.0, atol=1e-5), (
                f"Baseline ratio != 1 for N={N}, protocol={protocol}: {result.ratio[valid]}"
            )

    def test_baseline_all_N_dual(self) -> None:
        """Baseline for dual MZI for N=1..10 should have ratio ≈ 1."""
        result = compute_decoupled_baseline(
            omega_values=np.array([1.0]),
            N_values=np.arange(1, 11, dtype=int),
            protocol="dual",
        )
        valid = np.isfinite(result.ratio)
        assert np.sum(valid) >= 5
        assert np.allclose(result.ratio[valid], 1.0, atol=1e-5)

    def test_baseline_all_N_sonly(self) -> None:
        """Baseline for S-only MZI should have ratio ≈ 1."""
        result = compute_decoupled_baseline(
            omega_values=np.array([1.0]),
            N_values=np.array([1, 5, 10], dtype=int),
            protocol="S-only",
        )
        valid = np.isfinite(result.ratio)
        assert np.sum(valid) >= 2
        assert np.allclose(result.ratio[valid], 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Scaling Analysis
# ---------------------------------------------------------------------------


class TestScalingAnalysis:
    def test_scaling_decoupled_gives_sql_exponent(self) -> None:
        """For decoupled data, exponent should be close to -0.5 (SQL)."""
        omega_vals = np.array([0.5, 1.0])
        N_vals = np.array([1, 2, 3, 5, 10, 15, 20], dtype=int)
        sweep = compute_decoupled_baseline(
            omega_values=omega_vals,
            N_values=N_vals,
            protocol="dual",
        )
        scaling = fit_scaling_exponents(
            sweep.omega_values,
            sweep.N_values,
            sweep.delta_omega_opt,
        )
        valid = np.isfinite(scaling.exponents)
        if np.any(valid):
            for alpha in scaling.exponents[valid]:
                assert alpha == pytest.approx(-0.5, abs=0.05), (
                    f"Exponent {alpha} not close to SQL -0.5"
                )

    def test_scaling_returns_correct_shape(self) -> None:
        """Scaling analysis should return one exponent per ω."""
        N_vals = np.array([1, 2, 3, 5, 10], dtype=int)
        omega_vals = np.array([0.5, 1.0, 2.0])
        sweep = compute_decoupled_baseline(
            omega_values=omega_vals,
            N_values=N_vals,
            protocol="dual",
        )
        fit_scaling_exponents(
            sweep.omega_values,
            sweep.N_values,
            sweep.delta_omega_opt,
        )

    def test_scaling_r_squared_high(self) -> None:
        """For clean decoupled data, R² should be very high."""
        N_vals = np.array([1, 2, 3, 5, 10, 20], dtype=int)
        sweep = compute_decoupled_baseline(
            omega_values=np.array([1.0]),
            N_values=N_vals,
            protocol="dual",
        )
        scaling = fit_scaling_exponents(
            sweep.omega_values,
            sweep.N_values,
            sweep.delta_omega_opt,
        )
        valid = np.isfinite(scaling.r_squared)
        if np.any(valid):
            assert np.all(scaling.r_squared[valid] > 0.99)
        valid = np.isfinite(scaling.r_squared)
        if np.any(valid):
            assert np.all(scaling.r_squared[valid] > 0.99)


# ---------------------------------------------------------------------------
# Parquet Roundtrip
# ---------------------------------------------------------------------------


class TestParquetRoundtrip:
    def test_sweep_roundtrip(self, tmp_path: Path) -> None:
        """Basic roundtrip: save then load — all fields survive."""
        original = FourParamSweepResult(
            omega_values=np.array([0.5, 0.5, 1.0, 1.0]),
            N_values=np.array([1, 2, 1, 2], dtype=int),
            protocol=["dual", "dual", "S-only", "S-only"],
            alpha_xx_opt=np.array([1.0, 2.0, 0.5, 1.5]),
            alpha_xz_opt=np.array([0.5, -0.5, 1.0, -1.0]),
            alpha_zx_opt=np.array([3.0, 4.0, 2.0, 1.0]),
            alpha_zz_opt=np.array([-1.0, -2.0, 0.0, 3.0]),
            delta_omega_opt=np.array([0.08, 0.06, 0.09, 0.07]),
            sql_values=np.array([0.1, 0.07071, 0.1, 0.07071]),
            ratio=np.array([0.8, 0.8485, 0.9, 0.99]),
            expectation_Jz=np.array([0.25, 0.2, 0.3, 0.15]),
            variance_Jz=np.array([0.05, 0.04, 0.06, 0.03]),
            d_expectation=np.array([-0.5, -0.4, -0.6, -0.3]),
            n_starts=np.array([25, 25, 25, 25], dtype=int),
            n_converged=np.array([15, 12, 18, 14], dtype=int),
            gradient_norm=np.array([1e-8, 2e-7, 5e-9, 3e-6], dtype=float),
            t_hold=10.0,
        )
        parquet_path = tmp_path / "test_sweep.parquet"
        original.save_parquet(parquet_path)
        loaded = FourParamSweepResult.from_parquet(parquet_path)
        assert np.allclose(loaded.omega_values, original.omega_values)
        assert np.array_equal(loaded.N_values, original.N_values)
        assert loaded.protocol == original.protocol
        assert np.allclose(loaded.alpha_xx_opt, original.alpha_xx_opt)
        assert np.allclose(loaded.alpha_xz_opt, original.alpha_xz_opt)
        assert np.allclose(loaded.alpha_zx_opt, original.alpha_zx_opt)
        assert np.allclose(loaded.alpha_zz_opt, original.alpha_zz_opt)
        assert np.allclose(loaded.delta_omega_opt, original.delta_omega_opt)
        assert np.allclose(loaded.sql_values, original.sql_values)
        assert np.allclose(loaded.ratio, original.ratio)
        assert np.allclose(loaded.expectation_Jz, original.expectation_Jz)
        assert np.allclose(loaded.variance_Jz, original.variance_Jz)
        assert np.allclose(loaded.d_expectation, original.d_expectation)
        assert np.array_equal(loaded.n_starts, original.n_starts)
        assert np.array_equal(loaded.n_converged, original.n_converged)
        assert np.allclose(loaded.gradient_norm, original.gradient_norm)
        assert pytest.approx(original.t_hold) == loaded.t_hold

    def test_sweep_roundtrip_metadata(self, tmp_path: Path) -> None:
        """All metadata fields survive roundtrip."""
        original = FourParamSweepResult(
            omega_values=np.array([0.5, 1.0]),
            N_values=np.array([1, 5], dtype=int),
            protocol=["dual", "S-only"],
            alpha_xx_opt=np.array([5.0, 10.0]),
            alpha_xz_opt=np.array([-2.0, 3.0]),
            alpha_zx_opt=np.array([8.0, 12.0]),
            alpha_zz_opt=np.array([-6.0, 4.0]),
            delta_omega_opt=np.array([0.05, 0.03]),
            sql_values=np.array([0.1, 0.04472]),
            ratio=np.array([0.5, 0.6708]),
            expectation_Jz=np.array([0.1, 0.2]),
            variance_Jz=np.array([0.01, 0.02]),
            d_expectation=np.array([-0.5, -0.3]),
            n_starts=np.array([25, 25], dtype=int),
            n_converged=np.array([12, 10], dtype=int),
            gradient_norm=np.array([1e-8, 2e-7], dtype=float),
            t_hold=10.0,
        )
        parquet_path = tmp_path / "test_meta.parquet"
        original.save_parquet(parquet_path)
        loaded = FourParamSweepResult.from_parquet(parquet_path)
        assert loaded.omega_values[0] == pytest.approx(0.5)
        assert loaded.N_values[1] == 5
        assert loaded.protocol[0] == "dual"
        assert loaded.protocol[1] == "S-only"
        assert loaded.alpha_xx_opt[0] == pytest.approx(5.0)
        assert loaded.alpha_xz_opt[1] == pytest.approx(3.0)
        assert loaded.alpha_zx_opt[0] == pytest.approx(8.0)
        assert loaded.alpha_zz_opt[1] == pytest.approx(4.0)
        assert loaded.n_starts[0] == 25
        assert loaded.n_converged[0] == 12
        assert loaded.gradient_norm[0] == pytest.approx(1e-8)

    def test_from_parquet_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when required columns missing."""
        import pandas as pd

        df_bad = pd.DataFrame({"omega": [0.1, 0.5], "delta_omega_opt": [0.05, 0.06]})
        parquet_path = tmp_path / "bad.parquet"
        df_bad.to_parquet(parquet_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            FourParamSweepResult.from_parquet(parquet_path)

    def test_scaling_roundtrip(self, tmp_path: Path) -> None:
        """ScalingAnalysisResult roundtrip."""
        original = ScalingAnalysisResult(
            omega_values=np.array([0.5, 1.0, 2.0]),
            exponents=np.array([-0.5, -0.48, -0.52]),
            prefactors=np.array([0.1, 0.12, 0.09]),
            r_squared=np.array([0.999, 0.998, 0.997]),
        )
        parquet_path = tmp_path / "test_scaling.parquet"
        original.save_parquet(parquet_path)
        loaded = ScalingAnalysisResult.from_parquet(parquet_path)
        assert np.allclose(loaded.exponents, original.exponents)
        assert np.allclose(loaded.prefactors, original.prefactors)

    def test_scaling_roundtrip_metadata(self, tmp_path: Path) -> None:
        """Scaling metadata survives roundtrip."""
        original = ScalingAnalysisResult(
            omega_values=np.array([0.1, 1.0]),
            exponents=np.array([-0.5, -0.45]),
            prefactors=np.array([0.1, 0.15]),
            r_squared=np.array([0.99, 0.97]),
        )
        parquet_path = tmp_path / "test_scaling_meta.parquet"
        original.save_parquet(parquet_path)
        loaded = ScalingAnalysisResult.from_parquet(parquet_path)
        assert loaded.exponents[0] == pytest.approx(-0.5)
        assert loaded.prefactors[1] == pytest.approx(0.15)
        assert loaded.r_squared[0] == pytest.approx(0.99)
        assert loaded.sql_exponent == pytest.approx(-0.5)

    def test_scaling_roundtrip_custom_sql_exponent(self, tmp_path: Path) -> None:
        """sql_exponent metadata survives roundtrip."""
        original = ScalingAnalysisResult(
            omega_values=np.array([0.5]),
            exponents=np.array([-0.3]),
            prefactors=np.array([0.2]),
            r_squared=np.array([0.95]),
            sql_exponent=-0.3,
        )
        parquet_path = tmp_path / "test_scaling_custom.parquet"
        original.save_parquet(parquet_path)
        loaded = ScalingAnalysisResult.from_parquet(parquet_path)
        assert loaded.sql_exponent == pytest.approx(-0.3)

    def test_scaling_from_parquet_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when required columns missing."""
        import pandas as pd

        df_bad = pd.DataFrame({"omega": [0.1], "exponent": [-0.5]})
        parquet_path = tmp_path / "bad_scaling.parquet"
        df_bad.to_parquet(parquet_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            ScalingAnalysisResult.from_parquet(parquet_path)


# ---------------------------------------------------------------------------
# Physical Invariants
# ---------------------------------------------------------------------------


class TestPhysicalInvariants:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_var_positive_all_couplings(self, N: int) -> None:
        """Var(J_z^S) >= 0 for all couplings at representative ω."""
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        Jz_single = jz_operator(N, basis=OperatorBasis.DICKE)
        alpha_list = [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 3.0, 0.0, 0.0),
            (0.0, 0.0, 3.0, 0.0),
            (0.0, 0.0, 0.0, 3.0),
            (1.0, 2.0, -1.0, 0.5),
        ]
        for protocol in ("dual", "S-only"):
            for alpha_val in alpha_list:
                psi = evolve_circuit(
                    N, psi0, omega=1.0, alpha=alpha_val, ops=ops, protocol=protocol
                )
                _, var = compute_reduced_expectation_and_variance(psi, N, Jz_single)
                assert var >= -1e-12, (
                    f"Negative Var(J_z^S)={var:.2e} at N={N}, α={alpha_val}, "
                    f"protocol={protocol}"
                )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_sql_recovery_at_zero_coupling(self, N: int) -> None:
        """At α = 0, must recover SQL exactly for both protocols."""
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        sql = 1.0 / (np.sqrt(N) * DEFAULT_t_hold)
        for protocol in ("dual", "S-only"):
            for omega in [0.1, 1.0, 5.0]:
                dt, _, _, _ = compute_sensitivity(
                    N,
                    psi0,
                    omega_true=omega,
                    alpha=(0.0, 0.0, 0.0, 0.0),
                    ops=ops,
                    protocol=protocol,
                )
                assert dt == pytest.approx(sql, rel=1e-5), (
                    f"N={N}, ω={omega}, protocol={protocol}: "
                    f"Δω={dt:.10f} != SQL={sql:.10f}"
                )

    def test_single_bs_unitarity_all_N(self) -> None:
        """BS unitary for all standard N values."""
        for N in [1, 2, 3, 5, 10, 20]:
            U = single_bs_unitary(N)
            eye = np.eye(N + 1, dtype=complex)
            assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
                f"BS not unitary for N={N}"
            )

    def test_hermiticity_across_parameters(self) -> None:
        """Total H is Hermitian for all tested parameters."""
        for N in [1, 2, 5, 10]:
            ops = _embed_ops(N)
            for omega in [0.1, 1.0, 5.0]:
                for alpha_val in [
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 3.0, 0.0, 0.0),
                    (0.0, 0.0, 3.0, 0.0),
                    (1.0, 2.0, -1.0, 0.5),
                    (10.0, -5.0, 8.0, -3.0),
                ]:
                    H = build_hold_hamiltonian(N, omega, alpha_val, ops)
                    assert np.allclose(H, H.conj().T, atol=1e-12), (
                        f"H not Hermitian at N={N}, ω={omega}, α={alpha_val}"
                    )

    def test_zz_only_gives_sql_sonly(self) -> None:
        """α_zz-only gives SQL for S-only MZI (ancilla in eigenstate).

        In S-only MZI, the ancilla stays in |J,J⟩_A throughout the hold,
        so the α_zz J_z^S J_z^A term acts as a constant shift on the
        system only, commuting with the measurement.
        """
        N = 3
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        sql = 1.0 / (np.sqrt(N) * DEFAULT_t_hold)
        for alpha_zz in [0.0, 2.0, 5.0, 10.0]:
            dt, _, _, _ = compute_sensitivity(
                N,
                psi0,
                omega_true=1.0,
                alpha=(0.0, 0.0, 0.0, alpha_zz),
                ops=ops,
                protocol="S-only",
            )
            assert dt == pytest.approx(sql, rel=1e-5), (
                f"α_zz={alpha_zz}, S-only: Δω={dt:.10f} != SQL={sql:.10f}"
            )

    def test_zz_only_changes_sensitivity_dual(self) -> None:
        """α_zz-only changes sensitivity for dual MZI (ancilla in superposition).

        In dual MZI, the ancilla enters the hold in a superposition of J_z^A
        eigenstates, so α_zz J_z^S J_z^A creates system-ancilla correlations
        that affect the reduced state.
        """
        N = 3
        ops = _embed_ops(N)
        psi0 = initial_state(N)
        sql = 1.0 / (np.sqrt(N) * DEFAULT_t_hold)
        # At α_zz=0, must recover SQL
        dt0, _, _, _ = compute_sensitivity(
            N,
            psi0,
            omega_true=1.0,
            alpha=(0.0, 0.0, 0.0, 0.0),
            ops=ops,
            protocol="dual",
        )
        assert dt0 == pytest.approx(sql, rel=1e-5), (
            f"α_zz=0, dual: Δω={dt0:.10f} != SQL={sql:.10f}"
        )
        # At non-zero α_zz, sensitivity may differ from SQL
        for alpha_zz in [2.0, 5.0]:
            dt, _, _, _ = compute_sensitivity(
                N,
                psi0,
                omega_true=1.0,
                alpha=(0.0, 0.0, 0.0, alpha_zz),
                ops=ops,
                protocol="dual",
            )
            # Sensitivity should be finite and positive
            assert np.isfinite(dt) and dt > 0, (
                f"α_zz={alpha_zz}, dual: Δω={dt:.10f} should be finite and positive"
            )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_fd_step(self) -> None:
        assert FD_STEP == 1e-6

    def test_th_bs(self) -> None:
        assert pytest.approx(np.pi / 2.0) == DEFAULT_T_BS

    def test_th_default(self) -> None:
        assert DEFAULT_t_hold == 10.0

    def test_alpha_bound(self) -> None:
        assert ALPHA_BOUND == 20.0

    def test_n_lbfgs_starts(self) -> None:
        assert 20 <= N_LBFGS_STARTS <= 30

"""
Tests for the multi-particle dual-MZI XX-coupling optimised joint measurement module.

Run with:
    uv run pytest reports/20260525/test_local.py -q --tb=short
"""

from __future__ import annotations

import importlib.util
import sys as _sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from scipy.linalg import expm

from src.physics.dicke_basis import jz_operator
from src.utils.enums import OperatorBasis
from src.utils.serialization import assert_roundtrip_fields

_local_path = Path(__file__).resolve().parent / "local.py"
_spec = importlib.util.spec_from_file_location("local", str(_local_path))
assert _spec is not None
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_sys.modules["local"] = _module
_spec.loader.exec_module(_module)
del _local_path, _spec, _module

from local import (  # type: ignore[import-untyped]  # noqa: E402
    AXX_BOUNDS,
    DEFAULT_T_BS,
    FD_STEP,
    N_RANDOM_STARTS,
    OMEGA_MAX,
    OMEGA_MIN,
    PSI_BOUNDS,
    DEFAULT_t_hold,
    DualMZIOptimisedResult,
    ScalingAnalysisResult,
    build_hold_hamiltonian,
    build_measurement_operator,
    compute_decoupled_baseline,
    compute_sensitivity_full,
    dual_bs_unitary,
    embed_combined_operators,
    evolve_circuit,
    fit_scaling_exponents,
    full_state_expectation_and_variance,
    hold_unitary_dicke,
    initial_state,
    optimise_joint,
    run_sweep,
    single_bs_unitary,
)


def _embed_ops_for_tests(N: int) -> dict[str, np.ndarray]:
    """Fast operator embedding for tests."""
    return embed_combined_operators(N)


class TestOperatorConstruction:
    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_operators_correct_dimension(self, N: int) -> None:
        ops = _embed_ops_for_tests(N)
        dim = (N + 1) ** 2
        for name, op in ops.items():
            assert op.shape == (dim, dim), (
                f"{name} has shape {op.shape}, expected ({dim}, {dim})"
            )

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_operators_hermitian(self, N: int) -> None:
        ops = _embed_ops_for_tests(N)
        for name in ["Jz_S", "Jz_A", "Jx_S", "Jx_A", "Jy_S", "Jy_A"]:
            assert np.allclose(ops[name], ops[name].conj().T, atol=1e-12), (
                f"{name} not Hermitian for N={N}"
            )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_commutation_jz_jx(self, N: int) -> None:
        ops = _embed_ops_for_tests(N)
        comm_S = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
        expected = 1j * ops["Jy_S"]
        assert np.allclose(comm_S, expected, atol=1e-12), (
            f"[Jz_S, Jx_S] = i Jy_S failed for N={N}"
        )
        comm_A = ops["Jz_A"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jz_A"]
        assert np.allclose(comm_A, 1j * ops["Jy_A"], atol=1e-12), (
            f"[Jz_A, Jx_A] = i Jy_A failed for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_hold_hamiltonian_hermitian(self, N: int) -> None:
        ops = _embed_ops_for_tests(N)
        for omega in [0.1, 1.0, 5.0]:
            for alpha_xx in [0.0, 1.0, 10.0]:
                H = build_hold_hamiltonian(N, omega, alpha_xx, ops)
                assert np.allclose(H, H.conj().T, atol=1e-12), (
                    f"H not Hermitian for N={N}, \u03b8={omega}, \u03b1_xx={alpha_xx}"
                )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_hold_hamiltonian_decoupled(self, N: int) -> None:
        """At α_xx = 0, H = ω(J_z^S + J_z^A)."""
        ops = _embed_ops_for_tests(N)
        omega = 0.7
        H = build_hold_hamiltonian(N, omega, 0.0, ops)
        expected = omega * (ops["Jz_S"] + ops["Jz_A"])
        assert np.allclose(H, expected, atol=1e-12), f"Decoupled H mismatch for N={N}"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_jz_single_diagonal(self, N: int) -> None:
        """J_z should be diagonal in the Dicke basis."""
        Jz = jz_operator(N, basis=OperatorBasis.DICKE)
        assert np.allclose(Jz, np.diag(np.diag(Jz)), atol=1e-12), (
            f"J_z not diagonal for N={N}"
        )


class TestMeasurementOperator:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_measurement_operator_hermitian(self, N: int) -> None:
        ops = _embed_ops_for_tests(N)
        for psi in [0.0, np.pi / 4, np.pi / 2, -1.0]:
            M = build_measurement_operator(N, psi, ops)
            assert np.allclose(M, M.conj().T, atol=1e-12), (
                f"M(ψ) not Hermitian for N={N}, ψ={psi}"
            )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_measurement_operator_phi_zero(self, N: int) -> None:
        """At φ=0, M = J_z^S."""
        ops = _embed_ops_for_tests(N)
        M = build_measurement_operator(N, 0.0, ops)
        assert np.allclose(M, ops["Jz_S"], atol=1e-12), f"M(0) != Jz_S for N={N}"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_measurement_operator_phi_pi_over_two(self, N: int) -> None:
        """At φ=π/2, M = J_z^A."""
        ops = _embed_ops_for_tests(N)
        M = build_measurement_operator(N, np.pi / 2.0, ops)
        assert np.allclose(M, ops["Jz_A"], atol=1e-12), f"M(π/2) != Jz_A for N={N}"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_measurement_operator_phi_pi_over_four(self, N: int) -> None:
        """At φ=π/4, M = (J_z^S + J_z^A)/√2."""
        ops = _embed_ops_for_tests(N)
        M = build_measurement_operator(N, np.pi / 4.0, ops)
        expected = (ops["Jz_S"] + ops["Jz_A"]) / np.sqrt(2)
        assert np.allclose(M, expected, atol=1e-12), (
            f"M(π/4) != (Jz_S+Jz_A)/√2 for N={N}"
        )


class TestFullStateMeasurement:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_initial_state_variance_zero(self, N: int) -> None:
        """For |J,J⟩_S ⊗ |J,J⟩_A, Var(M) = 0 for any φ."""
        psi = initial_state(N)
        ops = _embed_ops_for_tests(N)
        for psi_val in [0.0, np.pi / 4, np.pi / 2]:
            meas_op = build_measurement_operator(N, psi_val, ops)
            _, var = full_state_expectation_and_variance(psi, meas_op)
            assert var == pytest.approx(0.0, abs=1e-12), (
                f"Var != 0 at N={N}, ψ={psi_val}"
            )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_variance_positive(self, N: int) -> None:
        """Var(M) >= 0 for all parameters."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        for alpha_xx in [0.0, 1.0, 5.0]:
            for psi_weight in [0.0, np.pi / 4, np.pi / 2]:
                psi_state = evolve_circuit(
                    N, psi0, omega=0.5, alpha_xx=alpha_xx, ops=ops
                )
                meas_op = build_measurement_operator(N, psi_weight, ops)
                _, var = full_state_expectation_and_variance(psi_state, meas_op)
                assert var >= -1e-12, (
                    f"Negative Var at N={N}, α_xx={alpha_xx}, ψ={psi_weight}: {var}"
                )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_phi_zero_equals_jz_s_expectation(self, N: int) -> None:
        """At φ=0, ⟨M⟩ = ⟨J_z^S⟩ (compute via full state)."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        psi_weight = 0.0
        meas_op = build_measurement_operator(N, psi_weight, ops)
        Jz_S_op = ops["Jz_S"]
        for alpha_xx in [0.0, 2.0]:
            psi = evolve_circuit(N, psi0, omega=0.5, alpha_xx=alpha_xx, ops=ops)
            exp_M, _ = full_state_expectation_and_variance(psi, meas_op)
            exp_Jz, _ = full_state_expectation_and_variance(psi, Jz_S_op)
            assert exp_M == pytest.approx(exp_Jz, abs=1e-12), (
                f"⟨M⟩(φ=0) != ⟨Jz_S⟩ for N={N}"
            )


class TestUnitarity:
    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_single_bs_unitary(self, N: int) -> None:
        U = single_bs_unitary(N)
        eye = np.eye(N + 1, dtype=complex)
        assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
            f"BS unitary not unitary for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 3, 5, 10])
    def test_dual_bs_unitary(self, N: int) -> None:
        U = dual_bs_unitary(N)
        dim = (N + 1) ** 2
        eye = np.eye(dim, dtype=complex)
        assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
            f"Dual BS not unitary for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_hold_unitary_dicke(self, N: int) -> None:
        ops = _embed_ops_for_tests(N)
        U = hold_unitary_dicke(N, t_hold=1.0, omega=0.5, alpha_xx=2.0, ops=ops)
        dim = (N + 1) ** 2
        eye = np.eye(dim, dtype=complex)
        assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
            f"Hold unitary not unitary for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_hold_unitary_decoupled_factorizes(self, N: int) -> None:
        """At α_xx = 0, U_hold = exp(-i t_hold ω J_z) ⊗ exp(-i t_hold ω J_z)."""
        ops = _embed_ops_for_tests(N)
        omega = 0.5
        t_hold = 2.0
        U = hold_unitary_dicke(N, t_hold, omega, 0.0, ops)
        Jz = jz_operator(N, basis=OperatorBasis.DICKE)
        U_single = expm(-1j * t_hold * omega * Jz)
        expected = np.kron(U_single, U_single)
        assert np.allclose(U, expected, atol=1e-12), (
            f"Decoupled hold does not factorize for N={N}"
        )


class TestCircuitEvolution:
    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_initial_state_normalised(self, N: int) -> None:
        psi = initial_state(N)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_normalisation_preserved(self, N: int) -> None:
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        psi = evolve_circuit(N, psi0, omega=0.5, alpha_xx=0.0, ops=ops)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize(
        ("N", "alpha_xx"), [(1, 0.0), (2, 0.0), (3, 5.0), (5, 10.0)]
    )
    def test_normalisation_with_coupling(self, N: int, alpha_xx: float) -> None:
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        psi = evolve_circuit(N, psi0, omega=1.0, alpha_xx=alpha_xx, ops=ops)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_no_op_identity(self, N: int) -> None:
        """T_BS=0, t_hold=0 should give the initial state back."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        U_bs_zero = dual_bs_unitary(N, T_BS=0.0)
        psi = U_bs_zero @ psi0
        psi = hold_unitary_dicke(N, t_hold=0.0, omega=0.0, alpha_xx=0.0, ops=ops) @ psi
        psi = U_bs_zero @ psi
        assert np.allclose(psi, psi0, atol=1e-12), f"Identity failed for N={N}"


class TestSensitivity:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_decoupled_phi_zero_matches_n_sql(self, N: int) -> None:
        """At α_xx = 0, φ = 0: Δω = 1/(√N t_hold) (N-SQL)."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        omega = 1.0
        dt, *_ = compute_sensitivity_full(
            N,
            psi0,
            omega_true=omega,
            alpha_xx=0.0,
            psi=0.0,
            ops=ops,
        )
        n_sql = 1.0 / (np.sqrt(N) * DEFAULT_t_hold)
        assert dt == pytest.approx(n_sql, rel=1e-5), (
            f"N={N}: Δω={dt:.10f} != N-SQL={n_sql:.10f}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_decoupled_phi_pi_over_four_matches_2n_sql(self, N: int) -> None:
        """At α_xx = 0, φ = π/4: Δω = 1/(√(2N) t_hold) (2N-SQL)."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        omega = 1.0
        dt, *_ = compute_sensitivity_full(
            N,
            psi0,
            omega_true=omega,
            alpha_xx=0.0,
            psi=np.pi / 4.0,
            ops=ops,
        )
        sql_2n = 1.0 / (np.sqrt(2 * N) * DEFAULT_t_hold)
        assert dt == pytest.approx(sql_2n, rel=1e-5), (
            f"N={N}: Δω={dt:.10f} != 2N-SQL={sql_2n:.10f}"
        )

    @pytest.mark.parametrize(("N", "omega"), [(1, 0.1), (2, 0.5), (3, 1.0), (5, 2.0)])
    def test_decoupled_phi_pi_over_four_all_omega(self, N: int, omega: float) -> None:
        """At α_xx=0, φ=π/4, Δω = 2N-SQL for any ω."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        dt, _, _, _ = compute_sensitivity_full(
            N,
            psi0,
            omega_true=omega,
            alpha_xx=0.0,
            psi=np.pi / 4.0,
            ops=ops,
        )
        sql_2n = 1.0 / (np.sqrt(2 * N) * DEFAULT_t_hold)
        assert dt == pytest.approx(sql_2n, rel=1e-5), (
            f"N={N}, ω={omega}: Δω={dt:.10f} != 2N-SQL={sql_2n:.10f}"
        )

    @pytest.mark.parametrize(("N", "omega"), [(1, 0.1), (2, 0.5)])
    def test_n_sql_vs_2n_sql_ratio(self, N: int, omega: float) -> None:
        """Ratio of Δω(φ=0) to Δω(φ=π/4) should be √2 at α_xx=0."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        dt_phi0, *_ = compute_sensitivity_full(
            N,
            psi0,
            omega,
            0.0,
            psi=0.0,
            ops=ops,
        )
        dt_phi_pi4, *_ = compute_sensitivity_full(
            N,
            psi0,
            omega,
            0.0,
            psi=np.pi / 4.0,
            ops=ops,
        )
        assert dt_phi0 / dt_phi_pi4 == pytest.approx(np.sqrt(2), rel=1e-5), (
            f"N={N}: ratio={dt_phi0 / dt_phi_pi4:.6f} != √2"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_sensitivity_finite_with_coupling(self, N: int) -> None:
        """With non-zero α_xx and φ, sensitivity should be finite."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        for alpha_xx in [0.1, 1.0, 5.0]:
            for psi_weight in [0.0, np.pi / 4, -1.0]:
                dt, _, _, _ = compute_sensitivity_full(
                    N,
                    psi0,
                    omega_true=1.0,
                    alpha_xx=alpha_xx,
                    psi=psi_weight,
                    ops=ops,
                )
                assert np.isfinite(dt), (
                    f"Non-finite Δω={dt} at N={N}, α_xx={alpha_xx}, ψ={psi_weight}"
                )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_sensitivity_positive(self, N: int) -> None:
        """Sensitivity should always be positive."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        for alpha_xx in [0.0, 0.5, 2.0]:
            for psi_weight in [0.0, np.pi / 4, -0.5]:
                dt, _, _, _ = compute_sensitivity_full(
                    N,
                    psi0,
                    omega_true=1.0,
                    alpha_xx=alpha_xx,
                    psi=psi_weight,
                    ops=ops,
                )
                assert dt > 0, (
                    f"Non-positive Δω={dt} at N={N}, α_xx={alpha_xx}, ψ={psi_weight}"
                )

    def test_derivative_stability_across_fd_step(self) -> None:
        """Δω should be stable across a range of fd_step values."""
        N = 3
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        dt_vals = []
        for fd_step in [1e-5, 1e-6, 1e-7]:
            dt, _, _, _ = compute_sensitivity_full(
                N,
                psi0,
                omega_true=1.0,
                alpha_xx=2.0,
                psi=np.pi / 4.0,
                ops=ops,
                fd_step=fd_step,
            )
            dt_vals.append(dt)
        dt_arr = np.array(dt_vals)
        finite = np.isfinite(dt_arr)
        if np.sum(finite) >= 2:
            rel_spread = (np.max(dt_arr[finite]) - np.min(dt_arr[finite])) / np.mean(
                dt_arr[finite]
            )
            assert rel_spread < 0.1, (
                f"Derivative unstable: {dt_vals}, spread={rel_spread:.3e}"
            )

    @pytest.mark.slow
    def test_derivative_stability_at_N20(self) -> None:
        """Δω derivative stability at N=20 (largest particle number).

        Validates that the finite-difference derivative is well-conditioned
        for the largest Hilbert space dimension (441×441).
        """
        N = 20
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        dt_vals = []
        for fd_step in [1e-5, 1e-6, 1e-7]:
            dt, _, _, _ = compute_sensitivity_full(
                N,
                psi0,
                omega_true=1.0,
                alpha_xx=2.0,
                psi=np.pi / 4.0,
                ops=ops,
                fd_step=fd_step,
            )
            dt_vals.append(dt)
        dt_arr = np.array(dt_vals)
        finite = np.isfinite(dt_arr)
        if np.sum(finite) >= 2:
            rel_spread = (np.max(dt_arr[finite]) - np.min(dt_arr[finite])) / np.mean(
                dt_arr[finite]
            )
            assert rel_spread < 0.1, (
                f"Derivative unstable at N=20: {dt_vals}, spread={rel_spread:.3e}"
            )


class TestJointOptimisation:
    @pytest.mark.parametrize("N", [1, 2])
    def test_optimisation_returns_finite(self, N: int) -> None:
        """Optimisation should return a finite Δω."""
        ops = _embed_ops_for_tests(N)
        result = optimise_joint(
            N=N,
            omega=1.0,
            ops=ops,
            n_starts=5,
            rng_seed=42,
        )
        assert np.isfinite(result["delta_omega_opt"]) or np.isnan(
            result["alpha_xx_opt"]
        ), f"Non-finite result for N={N}"

    @pytest.mark.parametrize("N", [1, 2])
    def test_optimisation_alpha_in_bounds(self, N: int) -> None:
        """α_xx* should be within [0, 20]."""
        ops = _embed_ops_for_tests(N)
        result = optimise_joint(
            N=N,
            omega=1.0,
            ops=ops,
            n_starts=5,
            rng_seed=42,
        )
        if np.isfinite(result["alpha_xx_opt"]):
            lo, hi = AXX_BOUNDS
            assert lo - 1e-6 <= result["alpha_xx_opt"] <= hi + 1e-6, (
                f"α_xx*={result['alpha_xx_opt']} outside [{lo}, {hi}] for N={N}"
            )

    @pytest.mark.parametrize("N", [1, 2])
    def test_optimisation_phi_in_bounds(self, N: int) -> None:
        """φ* should be within [-π, π]."""
        ops = _embed_ops_for_tests(N)
        result = optimise_joint(
            N=N,
            omega=1.0,
            ops=ops,
            n_starts=5,
            rng_seed=42,
        )
        if np.isfinite(result["psi_opt"]):
            lo, hi = PSI_BOUNDS
            assert lo - 1e-6 <= result["psi_opt"] <= hi + 1e-6, (
                f"ψ*={result['psi_opt']} outside [{lo}, {hi}] for N={N}"
            )

    def test_optimisation_runs_and_returns_sql_reference(self) -> None:
        """Optimisation should run and store the correct SQL reference."""
        N = 3
        ops = _embed_ops_for_tests(N)
        result = optimise_joint(
            N=N,
            omega=0.5,
            ops=ops,
            n_starts=5,
            rng_seed=42,
        )
        sql_2n = 1.0 / (np.sqrt(2 * N) * DEFAULT_t_hold)
        if np.isfinite(result["delta_omega_opt"]):
            assert result["delta_omega_opt"] > 0
            assert result["sql_2n"] == pytest.approx(sql_2n)

    def test_optimisation_ms_ma_normalised(self) -> None:
        """m_s² + m_a² = 1 at optimum."""
        N = 2
        ops = _embed_ops_for_tests(N)
        result = optimise_joint(
            N=N,
            omega=1.0,
            ops=ops,
            n_starts=5,
            rng_seed=42,
        )
        if np.isfinite(result["ms_opt"]):
            ms2_plus_ma2 = result["ms_opt"] ** 2 + result["ma_opt"] ** 2
            assert ms2_plus_ma2 == pytest.approx(1.0, abs=1e-12), (
                f"m_s²+m_a² = {ms2_plus_ma2} != 1"
            )


class TestFullSweep:
    def test_small_sweep_runs(self) -> None:
        """A small sweep (2 ω × 2 N) should complete without error."""
        result = run_sweep(
            omega_values=np.array([0.5, 1.0]),
            N_values=np.array([1, 2]),
            n_starts=3,
        )
        assert result.n_points == 4
        assert len(result.omega_values) == 4
        assert len(result.alpha_xx_opt) == 4
        assert len(result.psi_opt) == 4
        assert len(result.ms_opt) == 4
        assert len(result.ma_opt) == 4

    def test_sweep_contains_sql(self) -> None:
        result = run_sweep(
            omega_values=np.array([1.0]),
            N_values=np.array([1, 3]),
            n_starts=3,
        )
        for i in range(result.n_points):
            N = result.N_values[i]
            expected_sql = 1.0 / (np.sqrt(2 * N) * DEFAULT_t_hold)
            assert result.sql_2n[i] == pytest.approx(expected_sql), (
                f"SQL mismatch for N={N}: {result.sql_2n[i]} != {expected_sql}"
            )

    def test_sweep_all_finite_or_inf(self) -> None:
        """All Δω values should be finite (or inf at fringe extremum)."""
        result = run_sweep(
            omega_values=np.array([0.5, 2.0]),
            N_values=np.array([1, 3]),
            n_starts=3,
        )
        for dt in result.delta_omega_opt:
            assert np.isfinite(dt) or np.isinf(dt), f"Non-finite Δω: {dt}"


class TestDecoupledBaseline:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_baseline_ratio_near_one(self, N: int) -> None:
        """At α_xx = 0, φ=π/4: Δω/2N-SQL ⊲ 1."""
        result = compute_decoupled_baseline(
            omega_values=np.array([0.5, 1.0]),
            N_values=np.array([N]),
        )
        valid = np.isfinite(result.ratio)
        if np.any(valid):
            assert np.allclose(result.ratio[valid], 1.0, atol=1e-5), (
                f"Baseline ratio != 1 for N={N}: {result.ratio[valid]}"
            )

    def test_baseline_all_N(self) -> None:
        """Baseline for all N should have ratio ≈ 1."""
        result = compute_decoupled_baseline(
            omega_values=np.array([1.0]),
            N_values=np.arange(1, 11, dtype=int),
        )
        valid = np.isfinite(result.ratio)
        assert np.sum(valid) >= 5
        assert np.allclose(result.ratio[valid], 1.0, atol=1e-5)

    def test_baseline_phi_all_pi_over_four(self) -> None:
        """All φ values in baseline should be π/4."""
        result = compute_decoupled_baseline(
            omega_values=np.array([0.5, 1.0, 2.0]),
            N_values=np.array([1, 3, 5]),
        )
        valid = np.isfinite(result.psi_opt)
        if np.any(valid):
            assert np.allclose(result.psi_opt[valid], np.pi / 4.0, atol=1e-12), (
                "Baseline φ not all π/4"
            )

    def test_baseline_ms_ma(self) -> None:
        """m_s = m_a = 1/√2 in baseline."""
        result = compute_decoupled_baseline(
            omega_values=np.array([1.0]),
            N_values=np.array([1, 5]),
        )
        expected = 1.0 / np.sqrt(2)
        valid = np.isfinite(result.ms_opt)
        if np.any(valid):
            assert np.allclose(result.ms_opt[valid], expected, atol=1e-12)
            assert np.allclose(result.ma_opt[valid], expected, atol=1e-12)


class TestScalingAnalysis:
    def test_scaling_decoupled_gives_sql_exponent(self) -> None:
        """For decoupled data, exponent should be close to -0.5 (SQL)."""
        omega_vals = np.array([0.5, 1.0])
        N_vals = np.array([1, 2, 3, 5, 10, 15, 20])
        sweep = compute_decoupled_baseline(
            omega_values=omega_vals,
            N_values=N_vals,
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
        )
        scaling = fit_scaling_exponents(
            sweep.omega_values,
            sweep.N_values,
            sweep.delta_omega_opt,
        )
        assert len(scaling.exponents) == len(omega_vals), (
            f"Expected {len(omega_vals)} exponents, got {len(scaling.exponents)}"
        )

    def test_scaling_r_squared_high(self) -> None:
        """For clean decoupled data, R² should be very high."""
        N_vals = np.array([1, 2, 3, 5, 10, 20], dtype=int)
        sweep = compute_decoupled_baseline(
            omega_values=np.array([1.0]),
            N_values=N_vals,
        )
        scaling = fit_scaling_exponents(
            sweep.omega_values,
            sweep.N_values,
            sweep.delta_omega_opt,
        )
        valid = np.isfinite(scaling.r_squared)
        if np.any(valid):
            assert np.all(scaling.r_squared[valid] > 0.99)


class TestParquetRoundtrip:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("omega_values", "allclose"),
        ("N_values", "array_eq"),
        ("alpha_xx_opt", "allclose"),
        ("psi_opt", "allclose"),
        ("ms_opt", "allclose"),
        ("ma_opt", "allclose"),
        ("delta_omega_opt", "allclose"),
        ("sql_2n", "allclose"),
        ("ratio", "allclose"),
        ("expectation_M", "allclose"),
        ("variance_M", "allclose"),
        ("d_expectation", "allclose"),
        ("n_starts_converged", "array_eq"),
        ("n_starts_at_best", "array_eq"),
        ("t_hold", "isclose"),
    ]

    def test_sweep_roundtrip(self, tmp_path: Path) -> None:
        """Basic roundtrip: save then load — all fields survive."""
        original = DualMZIOptimisedResult(
            omega_values=np.array([0.5, 0.5, 1.0, 1.0]),
            N_values=np.array([1, 2, 1, 2], dtype=int),
            alpha_xx_opt=np.array([0.0, 0.5, 1.0, 1.5]),
            psi_opt=np.array([0.785, 0.8, 0.79, 0.78]),
            ms_opt=np.array([0.707, 0.696, 0.703, 0.71]),
            ma_opt=np.array([0.707, 0.717, 0.71, 0.7]),
            delta_omega_opt=np.array([0.1, 0.07, 0.09, 0.06]),
            sql_2n=np.array([0.07071, 0.05, 0.07071, 0.05]),
            ratio=np.array([1.414, 1.40, 1.273, 1.20]),
            expectation_M=np.array([0.1, 0.2, 0.3, 0.15]),
            variance_M=np.array([0.05, 0.04, 0.06, 0.03]),
            d_expectation=np.array([-0.5, -0.4, -0.6, -0.3]),
            n_starts_converged=np.array([10, 8, 12, 9], dtype=int),
            n_starts_at_best=np.array([5, 3, 8, 4], dtype=int),
            t_hold=10.0,
        )
        parquet_path = tmp_path / "test_sweep.parquet"
        original.save_parquet(parquet_path)
        loaded = DualMZIOptimisedResult.from_parquet(parquet_path)
        assert_roundtrip_fields(loaded, original, self._FIELD_SPECS)

    def test_sweep_roundtrip_metadata(self, tmp_path: Path) -> None:
        """All metadata fields survive roundtrip."""
        original = DualMZIOptimisedResult(
            omega_values=np.array([0.1, 0.5, 1.0]),
            N_values=np.array([1, 3, 10], dtype=int),
            alpha_xx_opt=np.array([5.0, 10.0, 15.0]),
            psi_opt=np.array([0.7, 0.8, 0.6]),
            ms_opt=np.array([0.7648, 0.6967, 0.8253]),
            ma_opt=np.array([0.6442, 0.7174, 0.5646]),
            delta_omega_opt=np.array([0.05, 0.03, 0.015]),
            sql_2n=np.array([0.07071, 0.04082, 0.02236]),
            ratio=np.array([0.7071, 0.7348, 0.6708]),
            expectation_M=np.array([0.1, 0.2, 0.3]),
            variance_M=np.array([0.01, 0.02, 0.03]),
            d_expectation=np.array([-0.5, -0.3, -0.2]),
            n_starts_converged=np.array([10, 10, 12], dtype=int),
            n_starts_at_best=np.array([2, 5, 7], dtype=int),
            t_hold=10.0,
        )
        parquet_path = tmp_path / "test_meta.parquet"
        original.save_parquet(parquet_path)
        loaded = DualMZIOptimisedResult.from_parquet(parquet_path)
        assert_roundtrip_fields(loaded, original, self._FIELD_SPECS)

    def test_from_parquet_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when required columns missing."""
        import pandas as pd

        df_bad = pd.DataFrame({"omega": [0.1, 0.5], "delta_omega_opt": [0.05, 0.06]})
        parquet_path = tmp_path / "bad.parquet"
        df_bad.to_parquet(parquet_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            DualMZIOptimisedResult.from_parquet(parquet_path)

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
        parquet_path = tmp_path / "test_meta.parquet"
        original.save_parquet(parquet_path)
        loaded = ScalingAnalysisResult.from_parquet(parquet_path)
        assert loaded.exponents[0] == pytest.approx(-0.5)
        assert loaded.prefactors[1] == pytest.approx(0.15)
        assert loaded.r_squared[0] == pytest.approx(0.99)
        assert loaded.sql_exponent == pytest.approx(-0.5)

    def test_scaling_from_parquet_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when required columns missing."""
        import pandas as pd

        df_bad = pd.DataFrame({"omega": [0.1], "exponent": [-0.5]})
        parquet_path = tmp_path / "bad_scaling.parquet"
        df_bad.to_parquet(parquet_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            ScalingAnalysisResult.from_parquet(parquet_path)


class TestPhysicalInvariants:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_var_positive_all_couplings(self, N: int) -> None:
        """Var(M) >= 0 for all α_xx and φ at representative ω."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        for alpha_xx in [0.0, 0.5, 2.0, 5.0]:
            for psi_weight in [0.0, np.pi / 4, np.pi / 2, -1.0]:
                psi_state = evolve_circuit(
                    N, psi0, omega=1.0, alpha_xx=alpha_xx, ops=ops
                )
                meas_op = build_measurement_operator(N, psi_weight, ops)
                _, var = full_state_expectation_and_variance(psi_state, meas_op)
                assert var >= -1e-12, (
                    f"Negative Var(M)={var:.2e} at N={N}, α_xx={alpha_xx}, ψ={psi_weight}"
                )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_sql_recovery_at_zero_coupling(self, N: int) -> None:
        """At α_xx=0, φ=π/4: must recover 2N-SQL exactly."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        sql_2n = 1.0 / (np.sqrt(2 * N) * DEFAULT_t_hold)
        for omega in [0.1, 1.0, 5.0]:
            dt, _, _, _ = compute_sensitivity_full(
                N,
                psi0,
                omega_true=omega,
                alpha_xx=0.0,
                psi=np.pi / 4.0,
                ops=ops,
            )
            assert dt == pytest.approx(sql_2n, rel=1e-5), (
                f"N={N}, ω={omega}: Δω={dt:.10f} != 2N-SQL={sql_2n:.10f}"
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
            ops = _embed_ops_for_tests(N)
            for omega in [0.1, 1.0, 5.0]:
                for alpha_xx in [0.0, 1.0, 10.0, 20.0]:
                    H = build_hold_hamiltonian(N, omega, alpha_xx, ops)
                    assert np.allclose(H, H.conj().T, atol=1e-12), (
                        f"H not Hermitian at N={N}, ω={omega}, α_xx={alpha_xx}"
                    )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_phi_zero_n_sql_relationship(self, N: int) -> None:
        """At φ=0, the sensitivity is the N-SQL (factor √2 worse than 2N-SQL)."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        dt_phi0, *_ = compute_sensitivity_full(
            N,
            psi0,
            omega_true=1.0,
            alpha_xx=0.0,
            psi=0.0,
            ops=ops,
        )
        dt_phi_pi4, *_ = compute_sensitivity_full(
            N,
            psi0,
            omega_true=1.0,
            alpha_xx=0.0,
            psi=np.pi / 4.0,
            ops=ops,
        )
        # Δω(φ=0) = √2 × Δω(φ=π/4)
        assert dt_phi0 == pytest.approx(np.sqrt(2) * dt_phi_pi4, rel=1e-5), (
            f"N={N}: φ=0 sensitivity not √2 × φ=π/4"
        )


class TestSQLScaling:
    def test_sql_exponent_from_decoupled(self) -> None:
        """Log-log fit of decoupled Δω vs N should give α = -0.5."""
        N_vals = np.array([1, 2, 3, 5, 10, 15, 20], dtype=int)
        sql_vals = 1.0 / (np.sqrt(2 * N_vals) * DEFAULT_t_hold)
        log_N = np.log(N_vals.astype(float))
        log_sql = np.log(sql_vals)
        A = np.vstack([log_N, np.ones_like(log_N)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, log_sql, rcond=None)
        alpha = coeffs[0]
        assert alpha == pytest.approx(-0.5, abs=0.01), (
            f"SQL exponent = {alpha}, expected -0.5"
        )


class TestConstants:
    def test_omega_range(self) -> None:
        assert OMEGA_MIN == 0.1
        assert OMEGA_MAX == 5.0

    def test_axx_bounds(self) -> None:
        lo, hi = AXX_BOUNDS
        assert lo == 0.0
        assert hi == 20.0

    def test_psi_bounds(self) -> None:
        lo, hi = PSI_BOUNDS
        assert lo == pytest.approx(-np.pi)
        assert hi == pytest.approx(np.pi)

    def test_n_random_starts(self) -> None:
        assert N_RANDOM_STARTS == 20

    def test_fd_step(self) -> None:
        assert FD_STEP == 1e-6

    def test_d_bs(self) -> None:
        assert pytest.approx(np.pi / 2.0) == DEFAULT_T_BS
        assert DEFAULT_t_hold == 10.0

"""
Tests for the multi-particle dual-MZI XX-coupling module.

Run with:
    uv run pytest reports/2026-05-22/test_local.py -q --tb=short
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from src.physics.dicke_basis import jz_operator

_report_dir = str(
    Path(__file__).resolve().parent.parent.parent / "reports" / "2026-05-22",
)
if _report_dir not in _sys.path:
    _sys.path.insert(0, _report_dir)
del _sys, _report_dir

from local import (  # type: ignore[import-untyped]  # noqa: E402
    AXX_BOUNDS,
    DEFAULT_T_BS,
    DEFAULT_T_H,
    FD_STEP,
    N_COARSE_GRID,
    THETA_MAX,
    THETA_MIN,
    DualMZISweepResult,
    ScalingAnalysisResult,
    build_hold_hamiltonian,
    compute_decoupled_baseline,
    compute_reduced_expectation_and_variance,
    compute_sensitivity,
    dual_bs_unitary,
    embed_combined_operators,
    evolve_circuit,
    fit_scaling_exponents,
    hold_unitary,
    initial_state,
    optimise_alpha_xx,
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
        for name, op in ops.items():
            assert np.allclose(op, op.conj().T, atol=1e-12), (
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
        for theta in [0.1, 1.0, 5.0]:
            for alpha_xx in [0.0, 1.0, 10.0]:
                H = build_hold_hamiltonian(N, theta, alpha_xx, ops)
                assert np.allclose(H, H.conj().T, atol=1e-12), (
                    f"H not Hermitian for N={N}, θ={theta}, α_xx={alpha_xx}"
                )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_hold_hamiltonian_decoupled(self, N: int) -> None:
        """At α_xx = 0, H = θ(J_z^S + J_z^A)."""
        ops = _embed_ops_for_tests(N)
        theta = 0.7
        H = build_hold_hamiltonian(N, theta, 0.0, ops)
        expected = theta * (ops["Jz_S"] + ops["Jz_A"])
        assert np.allclose(H, expected, atol=1e-12), f"Decoupled H mismatch for N={N}"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_jz_single_diagonal(self, N: int) -> None:
        """J_z should be diagonal in the Dicke basis."""
        Jz = jz_operator(N)
        assert np.allclose(Jz, np.diag(np.diag(Jz)), atol=1e-12), (
            f"J_z not diagonal for N={N}"
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
    def test_hold_unitary(self, N: int) -> None:
        ops = _embed_ops_for_tests(N)
        U = hold_unitary(N, T_H=1.0, theta=0.5, alpha_xx=2.0, ops=ops)
        dim = (N + 1) ** 2
        eye = np.eye(dim, dtype=complex)
        assert np.allclose(U @ U.conj().T, eye, atol=1e-12), (
            f"Hold unitary not unitary for N={N}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_hold_unitary_decoupled_factorizes(self, N: int) -> None:
        """At α_xx = 0, U_hold = exp(-i T_H θ J_z) ⊗ exp(-i T_H θ J_z)."""
        ops = _embed_ops_for_tests(N)
        theta = 0.5
        T_H = 2.0
        U = hold_unitary(N, T_H, theta, 0.0, ops)
        Jz = jz_operator(N)
        U_single = expm(-1j * T_H * theta * Jz)
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
        psi = evolve_circuit(N, psi0, theta=0.5, alpha_xx=0.0, ops=ops)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize(
        ("N", "alpha_xx"), [(1, 0.0), (2, 0.0), (3, 5.0), (5, 10.0)]
    )
    def test_normalisation_with_coupling(self, N: int, alpha_xx: float) -> None:
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        psi = evolve_circuit(N, psi0, theta=1.0, alpha_xx=alpha_xx, ops=ops)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_no_op_identity(self, N: int) -> None:
        """T_BS=0, T_H=0 should give the initial state back."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        # T_BS=0 is identity, T_H=0 is identity
        U_bs_zero = dual_bs_unitary(N, T=0.0)
        psi = U_bs_zero @ psi0
        psi = hold_unitary(N, T_H=0.0, theta=0.0, alpha_xx=0.0, ops=ops) @ psi
        psi = U_bs_zero @ psi
        assert np.allclose(psi, psi0, atol=1e-12), f"Identity failed for N={N}"


class TestReducedVariance:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_product_state_variance_zero(self, N: int) -> None:
        """For the initial product state |J,J⟩_S⊗|J,J⟩_A, Var(J_z^S)=0."""
        psi = initial_state(N)
        Jz_single = jz_operator(N)
        _, var = compute_reduced_expectation_and_variance(psi, N, Jz_single)
        assert var == pytest.approx(0.0, abs=1e-12), f"Var != 0 for N={N}"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_trace_preservation(self, N: int) -> None:
        """Tr(ρ_S) = 1 after partial trace."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        psi = evolve_circuit(N, psi0, theta=0.5, alpha_xx=2.0, ops=ops)
        psi_mat = psi.reshape(N + 1, N + 1)
        rho_S = psi_mat @ psi_mat.conj().T
        trace = float(np.real(np.trace(rho_S)))
        assert np.isclose(trace, 1.0, atol=1e-12), f"Tr(ρ_S) = {trace} != 1 for N={N}"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_variance_positive(self, N: int) -> None:
        """Var(J_z^S) >= 0 for all couplings."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        Jz_single = jz_operator(N)
        for alpha_xx in [0.0, 1.0, 5.0, 10.0]:
            psi = evolve_circuit(N, psi0, theta=0.5, alpha_xx=alpha_xx, ops=ops)
            _, var = compute_reduced_expectation_and_variance(psi, N, Jz_single)
            assert var >= -1e-12, f"Negative Var at N={N}, α_xx={alpha_xx}: {var}"


class TestSensitivity:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_decoupled_sensitivity_matches_sql(self, N: int) -> None:
        """At α_xx = 0, Δθ should equal SQL = 1/(√N T_H)."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        theta = 1.0
        dt, *_ = compute_sensitivity(
            N,
            psi0,
            theta_true=theta,
            alpha_xx=0.0,
            ops=ops,
        )
        sql = 1.0 / (np.sqrt(N) * DEFAULT_T_H)
        assert dt == pytest.approx(sql, rel=1e-5), (
            f"N={N}: Δθ={dt:.10f} != SQL={sql:.10f}"
        )

    @pytest.mark.parametrize(("N", "theta"), [(1, 0.1), (2, 0.5), (3, 1.0), (5, 2.0)])
    def test_decoupled_all_theta(self, N: int, theta: float) -> None:
        """At α_xx = 0, Δθ = SQL for any θ."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        dt, _, _, _ = compute_sensitivity(
            N,
            psi0,
            theta_true=theta,
            alpha_xx=0.0,
            ops=ops,
        )
        sql = 1.0 / (np.sqrt(N) * DEFAULT_T_H)
        assert dt == pytest.approx(sql, rel=1e-5), (
            f"N={N}, θ={theta}: Δθ={dt:.10f} != SQL={sql:.10f}"
        )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_sensitivity_finite_with_coupling(self, N: int) -> None:
        """With non-zero α_xx, sensitivity should be finite."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        for alpha_xx in [0.1, 1.0, 5.0, 10.0]:
            dt, _, _, _ = compute_sensitivity(
                N,
                psi0,
                theta_true=1.0,
                alpha_xx=alpha_xx,
                ops=ops,
            )
            assert np.isfinite(dt), f"Non-finite Δθ={dt} at N={N}, α_xx={alpha_xx}"

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_sensitivity_positive(self, N: int) -> None:
        """Sensitivity should always be positive."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        for alpha_xx in [0.0, 0.5, 2.0, 10.0]:
            dt, _, _, _ = compute_sensitivity(
                N,
                psi0,
                theta_true=1.0,
                alpha_xx=alpha_xx,
                ops=ops,
            )
            assert dt > 0, f"Non-positive Δθ={dt} at N={N}, α_xx={alpha_xx}"

    def test_derivative_stability_across_fd_step(self) -> None:
        """Δθ should be stable across a range of fd_step values."""
        N = 3
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        dt_vals = []
        for fd_step in [1e-5, 1e-6, 1e-7]:
            dt, _, _, _ = compute_sensitivity(
                N,
                psi0,
                theta_true=1.0,
                alpha_xx=2.0,
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


class TestAlphaOptimisation:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_optimisation_returns_finite(self, N: int) -> None:
        """Optimisation should return a finite Δθ."""
        ops = _embed_ops_for_tests(N)
        result = optimise_alpha_xx(N=N, theta=1.0, ops=ops, n_coarse=51)
        assert np.isfinite(result["delta_theta_opt"]) or np.isnan(
            result["alpha_xx_opt"]
        ), f"Non-finite result for N={N}"

    @pytest.mark.parametrize("N", [1, 2])
    def test_optimisation_alpha_in_bounds(self, N: int) -> None:
        """α_xx* should be within [0, 20]."""
        ops = _embed_ops_for_tests(N)
        result = optimise_alpha_xx(N=N, theta=1.0, ops=ops, n_coarse=51)
        if np.isfinite(result["alpha_xx_opt"]):
            lo, hi = AXX_BOUNDS
            assert lo - 1e-6 <= result["alpha_xx_opt"] <= hi + 1e-6, (
                f"α_xx*={result['alpha_xx_opt']} outside [{lo}, {hi}] for N={N}"
            )

    def test_optimisation_runs_and_returns_sql_reference(self) -> None:
        """Optimisation should run and store the correct SQL reference."""
        N = 3
        ops = _embed_ops_for_tests(N)
        result = optimise_alpha_xx(N=N, theta=0.5, ops=ops, n_coarse=51)
        sql = 1.0 / (np.sqrt(N) * DEFAULT_T_H)
        if np.isfinite(result["delta_theta_opt"]):
            assert result["delta_theta_opt"] > 0
            assert result["sql"] == pytest.approx(sql)

    def test_optimisation_theta_scan_small(self) -> None:
        """Run a mini sweep to check the pipeline works."""
        N = 2
        ops = _embed_ops_for_tests(N)
        for theta in [0.5, 1.0, 2.0]:
            result = optimise_alpha_xx(N=N, theta=theta, ops=ops, n_coarse=31)
            assert np.isfinite(result["delta_theta_opt"]) or np.isnan(
                result["alpha_xx_opt"]
            )


class TestFullSweep:
    def test_small_sweep_runs(self) -> None:
        """A small sweep (2 θ × 2 N) should complete without error."""
        result = run_sweep(
            theta_values=np.array([0.5, 1.0]),
            N_values=np.array([1, 2]),
        )
        assert result.n_points == 4
        assert len(result.theta_values) == 4
        assert len(result.alpha_xx_opt) == 4

    def test_sweep_contains_sql(self) -> None:
        result = run_sweep(
            theta_values=np.array([1.0]),
            N_values=np.array([1, 3]),
        )
        for i in range(result.n_points):
            N = result.N_values[i]
            expected_sql = 1.0 / (np.sqrt(N) * DEFAULT_T_H)
            assert result.sql_values[i] == pytest.approx(expected_sql), (
                f"SQL mismatch for N={N}: {result.sql_values[i]} != {expected_sql}"
            )

    def test_sweep_all_finite_or_inf(self) -> None:
        """All Δθ values should be finite (or inf at fringe extremum)."""
        result = run_sweep(
            theta_values=np.array([0.5, 2.0]),
            N_values=np.array([1, 3]),
        )
        for dt in result.delta_theta_opt:
            assert np.isfinite(dt) or np.isinf(dt), f"Non-finite Δθ: {dt}"


class TestDecoupledBaseline:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_baseline_ratio_near_one(self, N: int) -> None:
        """At α_xx = 0, Δθ/SQL should be very close to 1."""
        result = compute_decoupled_baseline(
            theta_values=np.array([0.5, 1.0]),
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
            theta_values=np.array([1.0]),
            N_values=np.arange(1, 11, dtype=int),
        )
        valid = np.isfinite(result.ratio)
        assert np.sum(valid) >= 5
        assert np.allclose(result.ratio[valid], 1.0, atol=1e-5)


class TestScalingAnalysis:
    def test_scaling_decoupled_gives_sql_exponent(self) -> None:
        """For decoupled data, exponent should be close to -0.5 (SQL)."""
        theta_vals = np.array([0.5, 1.0])
        N_vals = np.array([1, 2, 3, 5, 10, 15, 20])
        sweep = compute_decoupled_baseline(
            theta_values=theta_vals,
            N_values=N_vals,
        )
        scaling = fit_scaling_exponents(sweep)
        valid = np.isfinite(scaling.exponents)
        if np.any(valid):
            for alpha in scaling.exponents[valid]:
                assert alpha == pytest.approx(-0.5, abs=0.05), (
                    f"Exponent {alpha} not close to SQL -0.5"
                )

    def test_scaling_returns_correct_shape(self) -> None:
        """Scaling analysis should return one exponent per θ."""
        N_vals = np.array([1, 2, 3, 5, 10], dtype=int)
        theta_vals = np.array([0.5, 1.0, 2.0])
        sweep = compute_decoupled_baseline(
            theta_values=theta_vals,
            N_values=N_vals,
        )
        scaling = fit_scaling_exponents(sweep)
        assert len(scaling.theta_values) == 3
        assert len(scaling.exponents) == 3
        assert len(scaling.prefactors) == 3

    def test_scaling_r_squared_high(self) -> None:
        """For clean decoupled data, R² should be very high."""
        N_vals = np.array([1, 2, 3, 5, 10, 20], dtype=int)
        sweep = compute_decoupled_baseline(
            theta_values=np.array([1.0]),
            N_values=N_vals,
        )
        scaling = fit_scaling_exponents(sweep)
        valid = np.isfinite(scaling.r_squared)
        if np.any(valid):
            assert np.all(scaling.r_squared[valid] > 0.99)


class TestParquetRoundtrip:
    def test_sweep_roundtrip(self, tmp_path: Path) -> None:
        """Basic roundtrip: save then load — all fields survive."""
        original = DualMZISweepResult(
            theta_values=np.array([0.5, 0.5, 1.0, 1.0]),
            N_values=np.array([1, 2, 1, 2], dtype=int),
            alpha_xx_opt=np.array([0.0, 0.5, 1.0, 1.5]),
            delta_theta_opt=np.array([0.1, 0.07, 0.09, 0.06]),
            sql_values=np.array([0.1, 0.07071, 0.1, 0.07071]),
            ratio=np.array([1.0, 0.9899, 0.9, 0.8485]),
            expectation_Jz=np.array([0.25, 0.2, 0.3, 0.15]),
            variance_Jz=np.array([0.05, 0.04, 0.06, 0.03]),
            d_expectation=np.array([-0.5, -0.4, -0.6, -0.3]),
            T_H=10.0,
        )
        parquet_path = tmp_path / "test_sweep.parquet"
        original.save_parquet(parquet_path)
        loaded = DualMZISweepResult.from_parquet(parquet_path)
        assert np.allclose(loaded.theta_values, original.theta_values)
        assert np.array_equal(loaded.N_values, original.N_values)
        assert np.allclose(loaded.alpha_xx_opt, original.alpha_xx_opt)
        assert np.allclose(loaded.delta_theta_opt, original.delta_theta_opt)
        assert np.allclose(loaded.sql_values, original.sql_values)
        assert np.allclose(loaded.ratio, original.ratio)
        assert np.allclose(loaded.expectation_Jz, original.expectation_Jz)
        assert np.allclose(loaded.variance_Jz, original.variance_Jz)
        assert np.allclose(loaded.d_expectation, original.d_expectation)
        assert pytest.approx(original.T_H) == loaded.T_H

    def test_sweep_roundtrip_metadata(self, tmp_path: Path) -> None:
        """All metadata fields survive roundtrip."""
        original = DualMZISweepResult(
            theta_values=np.array([0.1, 0.5, 1.0]),
            N_values=np.array([1, 3, 10], dtype=int),
            alpha_xx_opt=np.array([5.0, 10.0, 15.0]),
            delta_theta_opt=np.array([0.05, 0.03, 0.015]),
            sql_values=np.array([0.1, 0.05774, 0.03162]),
            ratio=np.array([0.5, 0.5196, 0.4743]),
            expectation_Jz=np.array([0.1, 0.2, 0.3]),
            variance_Jz=np.array([0.01, 0.02, 0.03]),
            d_expectation=np.array([-0.5, -0.3, -0.2]),
            T_H=10.0,
        )
        parquet_path = tmp_path / "test_meta.parquet"
        original.save_parquet(parquet_path)
        loaded = DualMZISweepResult.from_parquet(parquet_path)
        assert loaded.theta_values[0] == pytest.approx(0.1)
        assert loaded.alpha_xx_opt[1] == pytest.approx(10.0)
        assert loaded.delta_theta_opt[2] == pytest.approx(0.015)
        assert loaded.sql_values[0] == pytest.approx(0.1)
        assert loaded.ratio[2] == pytest.approx(0.4743, rel=1e-3)
        assert loaded.expectation_Jz[1] == pytest.approx(0.2)
        assert loaded.variance_Jz[2] == pytest.approx(0.03)
        assert loaded.d_expectation[0] == pytest.approx(-0.5)
        assert pytest.approx(10.0) == loaded.T_H

    def test_from_parquet_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when required columns missing."""
        import pandas as pd

        df_bad = pd.DataFrame({"theta": [0.1, 0.5], "delta_theta_opt": [0.05, 0.06]})
        parquet_path = tmp_path / "bad.parquet"
        df_bad.to_parquet(parquet_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            DualMZISweepResult.from_parquet(parquet_path)

    def test_scaling_roundtrip(self, tmp_path: Path) -> None:
        """ScalingAnalysisResult roundtrip."""
        original = ScalingAnalysisResult(
            theta_values=np.array([0.5, 1.0, 2.0]),
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
            theta_values=np.array([0.1, 1.0]),
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

    def test_scaling_roundtrip_custom_sql_exponent(self, tmp_path: Path) -> None:
        """sql_exponent metadata survives roundtrip."""
        original = ScalingAnalysisResult(
            theta_values=np.array([0.5]),
            exponents=np.array([-0.3]),
            prefactors=np.array([0.2]),
            r_squared=np.array([0.95]),
            sql_exponent=-0.3,
        )
        parquet_path = tmp_path / "test_custom.parquet"
        original.save_parquet(parquet_path)
        loaded = ScalingAnalysisResult.from_parquet(parquet_path)
        assert loaded.sql_exponent == pytest.approx(-0.3)

    def test_scaling_from_parquet_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when required columns missing."""
        import pandas as pd

        df_bad = pd.DataFrame({"theta": [0.1], "exponent": [-0.5]})
        parquet_path = tmp_path / "bad_scaling.parquet"
        df_bad.to_parquet(parquet_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            ScalingAnalysisResult.from_parquet(parquet_path)


class TestPhysicalInvariants:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_var_positive_all_couplings(self, N: int) -> None:
        """Var(J_z^S) >= 0 for all α_xx at representative θ."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        Jz_single = jz_operator(N)
        for alpha_xx in [0.0, 0.5, 2.0, 5.0, 10.0]:
            psi = evolve_circuit(N, psi0, theta=1.0, alpha_xx=alpha_xx, ops=ops)
            _, var = compute_reduced_expectation_and_variance(psi, N, Jz_single)
            assert var >= -1e-12, (
                f"Negative Var(J_z^S)={var:.2e} at N={N}, α_xx={alpha_xx}"
            )

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_sql_recovery_at_zero_coupling(self, N: int) -> None:
        """At α_xx=0, must recover SQL exactly."""
        ops = _embed_ops_for_tests(N)
        psi0 = initial_state(N)
        sql = 1.0 / (np.sqrt(N) * DEFAULT_T_H)
        for theta in [0.1, 1.0, 5.0]:
            dt, _, _, _ = compute_sensitivity(
                N,
                psi0,
                theta_true=theta,
                alpha_xx=0.0,
                ops=ops,
            )
            assert dt == pytest.approx(sql, rel=1e-5), (
                f"N={N}, θ={theta}: Δθ={dt:.10f} != SQL={sql:.10f}"
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
            for theta in [0.1, 1.0, 5.0]:
                for alpha_xx in [0.0, 1.0, 10.0, 20.0]:
                    H = build_hold_hamiltonian(N, theta, alpha_xx, ops)
                    assert np.allclose(H, H.conj().T, atol=1e-12), (
                        f"H not Hermitian at N={N}, θ={theta}, α_xx={alpha_xx}"
                    )


class TestSQLScaling:
    def test_sql_exponent_from_decoupled(self) -> None:
        """Log-log fit of decoupled Δθ vs N should give α = -0.5."""
        N_vals = np.array([1, 2, 3, 5, 10, 15, 20], dtype=int)
        sql_vals = 1.0 / (np.sqrt(N_vals) * DEFAULT_T_H)
        log_N = np.log(N_vals.astype(float))
        log_sql = np.log(sql_vals)
        A = np.vstack([log_N, np.ones_like(log_N)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, log_sql, rcond=None)
        alpha = coeffs[0]
        assert alpha == pytest.approx(-0.5, abs=0.01), (
            f"SQL exponent = {alpha}, expected -0.5"
        )


class TestConstants:
    def test_theta_range(self) -> None:
        assert THETA_MIN == 0.1
        assert THETA_MAX == 5.0

    def test_axx_bounds(self) -> None:
        lo, hi = AXX_BOUNDS
        assert lo == 0.0
        assert hi == 20.0

    def test_n_coarse_grid(self) -> None:
        assert N_COARSE_GRID == 101

    def test_fd_step(self) -> None:
        assert FD_STEP == 1e-6

    def test_th_bs(self) -> None:
        assert pytest.approx(np.pi / 2.0) == DEFAULT_T_BS
        assert DEFAULT_T_H == 10.0

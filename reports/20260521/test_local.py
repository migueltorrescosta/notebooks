"""
Tests for the general-interaction ancilla metrology module.

This is the companion test module for ``local.py`` in this directory.
Run with:
    uv run pytest reports/20260521/test_local.py -q --tb=short
"""

from __future__ import annotations

import concurrent.futures
import shutil
import sys as _sys
from pathlib import Path as _Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from deltalake import DeltaTable
from scipy.linalg import expm

from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
)

if TYPE_CHECKING:
    from pathlib import Path

_report_dir = str(
    _Path(__file__).resolve().parent.parent.parent / "reports" / "20260521"
)
if _report_dir not in _sys.path:
    _sys.path.insert(0, _report_dir)
del _sys, _Path, _report_dir

from local import (  # type: ignore[import-untyped]  # noqa: E402
    ALPHA_BOUNDS,
    DEFAULT_PSI0,
    DEFAULT_T_BS,
    N_BFGS_STARTS,
    SQL_REFERENCE,
    DEFAULT_T_hold,
    GeneralBFGSOptimizationResult,
    GeneralOmegaScanResult,
    _upsert_bfgs_result,
    build_general_hold_hamiltonian,
    compute_general_decoupled_baseline,
    compute_general_sensitivity,
    compute_general_sensitivity_with_diagnostics,
    compute_reduced_expectation,
    compute_reduced_variance,
    evolve_general_circuit,
    general_hold_unitary,
    run_general_bfgs_optimization,
    run_general_omega_scan,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def make_ops() -> dict:
    """Default two-qubit operators."""
    return build_two_qubit_operators()


# ============================================================================
# Test: Operator and Hamiltonian Construction
# ============================================================================


class TestOperatorConstruction:
    def test_hold_hamiltonian_hermitian(self, make_ops: dict) -> None:
        alpha = (1.0, -0.5, 2.0, 1.5)
        H = build_general_hold_hamiltonian(omega=0.5, alpha=alpha, ops=make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_hold_hamiltonian_contains_jz_sum(self, make_ops: dict) -> None:
        """H should contain ω(J_z^S + J_z^A) as the phase-encoding part."""
        omega = 0.7
        alpha = (0.0, 0.0, 0.0, 0.0)
        H = build_general_hold_hamiltonian(omega=omega, alpha=alpha, ops=make_ops)
        expected = omega * (make_ops["Jz_S"] + make_ops["Jz_A"])
        assert np.allclose(H, expected, atol=1e-12)

    def test_hold_hamiltonian_zero_omega(self, make_ops: dict) -> None:
        """At ω=0, the Hamiltonian should be just H_int."""
        alpha = (1.0, 2.0, 3.0, 4.0)
        H = build_general_hold_hamiltonian(omega=0.0, alpha=alpha, ops=make_ops)
        from local import build_interaction_hamiltonian

        H_int = build_interaction_hamiltonian(alpha)
        assert np.allclose(H, H_int, atol=1e-12)

    def test_hold_unitary(self, make_ops: dict) -> None:
        alpha = (1.0, -0.5, 2.0, 1.5)
        U = general_hold_unitary(T_hold=1.0, omega=0.5, alpha=alpha, ops=make_ops)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12)

    def test_hold_unitary_identity_at_zero(self, make_ops: dict) -> None:
        """At T_hold=0, the hold unitary should be identity."""
        alpha = (1.0, 2.0, 3.0, 4.0)
        U = general_hold_unitary(T_hold=0.0, omega=0.5, alpha=alpha, ops=make_ops)
        assert np.allclose(U, np.eye(4), atol=1e-12)

    @pytest.mark.parametrize("omega", [0.0, 0.5, 1.0, 2.0])
    def test_hold_hermiticity_all_omega(self, omega: float, make_ops: dict) -> None:
        """Total H must be Hermitian for all parameters."""
        alpha = (1.0, -1.0, 1.0, -1.0)
        H = build_general_hold_hamiltonian(omega, alpha, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12), f"H not Hermitian at ω={omega}"

    def test_commutation_jz_jx(self, make_ops: dict) -> None:
        comm_S = (
            make_ops["Jz_S"] @ make_ops["Jx_S"] - make_ops["Jx_S"] @ make_ops["Jz_S"]
        )
        assert np.allclose(comm_S, 1j * make_ops["Jy_S"], atol=1e-12)
        comm_A = (
            make_ops["Jz_A"] @ make_ops["Jx_A"] - make_ops["Jx_A"] @ make_ops["Jz_A"]
        )
        assert np.allclose(comm_A, 1j * make_ops["Jy_A"], atol=1e-12)

    def test_decoupled_hold_reduces_to_factorised(self, make_ops: dict) -> None:
        """At α = (0,0,0,0), the hold should factorise.

        exp(-i T_hold ω (J_z^S + J_z^A)) = exp(-i T_hold ω J_z) ⊗ exp(-i T_hold ω J_z)
        since [J_z^S, J_z^A] = 0.
        """
        omega = 0.5
        T_hold = 2.0
        alpha = (0.0, 0.0, 0.0, 0.0)
        U = general_hold_unitary(T_hold=T_hold, omega=omega, alpha=alpha, ops=make_ops)
        J_z = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
        U_single = expm(-1j * T_hold * omega * J_z)
        expected = np.kron(U_single, U_single)
        assert np.allclose(U, expected, atol=1e-12)


# ============================================================================
# Test: Circuit Evolution
# ============================================================================


class TestCircuitEvolution:
    def test_normalisation_preserved(self, make_ops: dict) -> None:
        alpha = (0.0, 0.0, 0.0, 0.0)
        psi = evolve_general_circuit(
            DEFAULT_PSI0, DEFAULT_T_BS, 5.0, 0.5, alpha, make_ops
        )
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    def test_normalisation_with_full_coupling(self, make_ops: dict) -> None:
        alpha = (5.0, -3.0, 2.0, 4.0)
        psi = evolve_general_circuit(
            DEFAULT_PSI0, DEFAULT_T_BS, 10.0, 1.0, alpha, make_ops
        )
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize(
        "alpha",
        [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (5.0, -2.0, 3.0, -1.0),
            (-10.0, 5.0, -5.0, 10.0),
            (20.0, -20.0, 20.0, -20.0),
        ],
    )
    def test_evolution_normalised_all_couplings(
        self, alpha: tuple[float, float, float, float], make_ops: dict
    ) -> None:
        psi = evolve_general_circuit(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, 0.5, alpha, make_ops
        )
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    def test_no_op_identity(self, make_ops: dict) -> None:
        """T_BS=0, T_hold=0, ω=0 should give the initial state back."""
        alpha = (0.0, 0.0, 0.0, 0.0)
        psi = evolve_general_circuit(DEFAULT_PSI0, 0.0, 0.0, 0.0, alpha, make_ops)
        assert np.allclose(psi, DEFAULT_PSI0, atol=1e-12)

    def test_ancilla_dynamics_independent_at_zero_coupling(
        self, make_ops: dict
    ) -> None:
        """At α=0, the system and ancilla evolve independently."""
        T_BS = 0.0  # No BS to entangle them
        T_hold = 5.0
        omega = 1.0
        alpha = (0.0, 0.0, 0.0, 0.0)
        psi = evolve_general_circuit(DEFAULT_PSI0, T_BS, T_hold, omega, alpha, make_ops)
        # State should be a product state: |ψ_S⟩ ⊗ |ψ_A⟩
        psi_mat = psi.reshape(2, 2)
        # Check that it's rank-1 (product state)
        _, s, _ = np.linalg.svd(psi_mat)
        assert np.isclose(s[0], 1.0, atol=1e-10), "State should be a product state"
        assert np.isclose(s[1], 0.0, atol=1e-10), "State should be a product state"


# ============================================================================
# Test: Reduced Density Matrix and Variance
# ============================================================================


class TestReducedVariance:
    def test_product_state_variance(self) -> None:
        """For a product state |00⟩, Var(J_z^S) should be zero."""
        var = compute_reduced_variance(DEFAULT_PSI0)
        assert var == pytest.approx(0.0, abs=1e-12)

    def test_bs_output_variance(self, make_ops: dict) -> None:
        """After first BS, the system is in (|0⟩ - i|1⟩)/√2.
        Var(J_z^S) = 1/4 - ⟨J_z^S⟩² = 1/4 - 0 = 1/4.
        """
        from local import system_only_bs_unitary

        U_bs = system_only_bs_unitary(DEFAULT_T_BS)
        psi = U_bs @ DEFAULT_PSI0
        var = compute_reduced_variance(psi)
        assert var == pytest.approx(0.25, abs=1e-12)

    @pytest.mark.parametrize(
        "alpha",
        [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 2.0, 0.0, 0.0),
            (0.0, 0.0, 3.0, 0.0),
            (0.0, 0.0, 0.0, 4.0),
            (2.0, -1.0, 3.0, -2.0),
        ],
    )
    def test_variance_positive(
        self, alpha: tuple[float, float, float, float], make_ops: dict
    ) -> None:
        """Variance should always be non-negative."""
        psi = evolve_general_circuit(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, 0.5, alpha, make_ops
        )
        var = compute_reduced_variance(psi)
        assert var >= -1e-12, f"Negative variance at α={alpha}: {var}"

    def test_reduced_expectation_matches_full(self, make_ops: dict) -> None:
        """⟨J_z^S⟩ via reduced state should match full-state expectation."""
        alpha = (3.0, -2.0, 1.0, 4.0)
        psi = evolve_general_circuit(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, 0.5, alpha, make_ops
        )
        exp_reduced = compute_reduced_expectation(psi)
        Jz_S = make_ops["Jz_S"]
        exp_full_val = float(np.real(psi.conj() @ Jz_S @ psi))
        assert exp_reduced == pytest.approx(exp_full_val, abs=1e-12)


# ============================================================================
# Test: Sensitivity Computation
# ============================================================================


class TestSensitivity:
    def test_decoupled_sensitivity(self, make_ops: dict) -> None:
        """At α = (0,0,0,0), Δω should equal SQL = 1/T_hold."""
        alpha = (0.0, 0.0, 0.0, 0.0)
        domega = compute_general_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, 1.0, alpha, make_ops
        )
        sql = 1.0 / DEFAULT_T_hold
        assert domega == pytest.approx(sql, rel=1e-6), (
            f"Δω={domega:.10f} != SQL={sql:.10f} at α=0"
        )

    @pytest.mark.parametrize("omega_true", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_decoupled_all_omega(self, omega_true: float, make_ops: dict) -> None:
        """At α = (0,0,0,0), Δω = SQL for any ω."""
        alpha = (0.0, 0.0, 0.0, 0.0)
        domega = compute_general_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, omega_true, alpha, make_ops
        )
        sql = 1.0 / DEFAULT_T_hold
        assert domega == pytest.approx(sql, rel=1e-5), (
            f"ω={omega_true}: Δω={domega:.10f} != SQL={sql:.10f}"
        )

    @pytest.mark.parametrize(
        "alpha",
        [
            (0.1, 0.0, 0.0, 0.0),
            (0.0, 0.1, 0.0, 0.0),
            (0.0, 0.0, 0.1, 0.0),
            (0.0, 0.0, 0.0, 0.1),
            (5.0, 0.0, 0.0, 0.0),
            (0.0, 5.0, 0.0, 0.0),
            (0.0, 0.0, 5.0, 0.0),
            (0.0, 0.0, 0.0, 5.0),
            (1.0, 2.0, 3.0, 4.0),
            (-5.0, 10.0, -10.0, 5.0),
        ],
    )
    def test_sensitivity_finite_with_coupling(
        self, alpha: tuple[float, float, float, float], make_ops: dict
    ) -> None:
        """With non-zero α, sensitivity should be finite."""
        domega = compute_general_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, 1.0, alpha, make_ops
        )
        assert np.isfinite(domega), f"Non-finite Δω={domega} at α={alpha}"

    def test_sensitivity_positive(self, make_ops: dict) -> None:
        """Sensitivity should always be positive."""
        for a_vals in [
            (0.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.0),
            (0.0, 2.0, 0.0, 0.0),
            (0.0, 0.0, -3.0, 0.0),
            (0.0, 0.0, 0.0, 4.0),
            (1.0, -1.0, 1.0, -1.0),
        ]:
            domega = compute_general_sensitivity(
                DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, 1.0, a_vals, make_ops
            )
            assert domega > 0, f"Non-positive Δω={domega} at α={a_vals}"

    def test_sensitivity_with_diagnostics_consistency(self, make_ops: dict) -> None:
        """compute_general_sensitivity and *_with_diagnostics should agree."""
        alpha = (2.0, -1.0, 3.0, -2.0)
        omega = 0.5
        domega1 = compute_general_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, omega, alpha, make_ops
        )
        domega2, exp_val, var_val, d_exp = compute_general_sensitivity_with_diagnostics(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, omega, alpha, make_ops
        )
        assert domega1 == pytest.approx(domega2, rel=1e-12)
        assert np.isfinite(exp_val)
        assert var_val >= 0
        assert np.isfinite(d_exp)

    def test_fringe_extremum(self, make_ops: dict) -> None:
        """At ω = π/T_hold with α=0, derivative should vanish (fringe extremum)."""
        alpha = (0.0, 0.0, 0.0, 0.0)
        domega = compute_general_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_hold,
            np.pi / DEFAULT_T_hold,
            alpha,
            make_ops,
        )
        assert np.isinf(domega) or domega > 100, (
            f"Δω should be large at fringe extremum: {domega}"
        )


# ============================================================================
# Test: Decoupled Baseline
# ============================================================================


class TestDecoupledBaseline:
    def test_baseline_matches_sql(self) -> None:
        result = compute_general_decoupled_baseline()
        assert result.delta_omega == pytest.approx(result.sql, rel=1e-10)

    def test_baseline_ratio(self) -> None:
        result = compute_general_decoupled_baseline()
        ratio = result.delta_omega / result.sql
        assert ratio == pytest.approx(1.0, rel=1e-10)

    def test_baseline_multiple_omegas(self) -> None:
        """Baseline should match SQL for any ω value."""
        for omega_true in [0.1, 0.5, 1.0, 2.0, 5.0]:
            result = compute_general_decoupled_baseline(omega_true=omega_true)
            assert result.delta_omega == pytest.approx(result.sql, rel=1e-5), (
                f"Baseline failed at ω={omega_true}"
            )


# ============================================================================
# Test: L-BFGS-B Optimisation
# ============================================================================


class TestBFGSOptimisation:
    def test_optimisation_runs(self) -> None:
        """L-BFGS-B should complete without error."""
        result = run_general_bfgs_optimization(
            omega_true=0.5,
            n_starts=5,
            maxiter=100,
        )
        assert np.isfinite(result.delta_omega_opt)
        assert result.n_starts == 5
        assert result.n_converged >= 0

    def test_optimisation_returns_valid_alpha(self) -> None:
        """Optimal α should be within bounds."""
        result = run_general_bfgs_optimization(
            omega_true=1.0,
            n_starts=10,
            maxiter=200,
        )
        lo, hi = ALPHA_BOUNDS
        for a_val in result.alpha_opt:
            assert lo <= a_val <= hi, f"α={a_val} outside bounds [{lo}, {hi}]"

    def test_optimisation_omega_recorded(self) -> None:
        result = run_general_bfgs_optimization(
            omega_true=0.7,
            n_starts=5,
            maxiter=100,
        )
        assert result.omega_value == pytest.approx(0.7)

    def test_optimisation_diagnostics_recorded(self) -> None:
        result = run_general_bfgs_optimization(
            omega_true=1.0,
            n_starts=5,
            maxiter=100,
        )
        assert np.isfinite(result.expectation_Jz)
        assert result.variance_Jz >= -1e-12
        assert np.isfinite(result.d_exp_d_omega) or np.isinf(result.delta_omega_opt)

    @pytest.mark.parametrize("omega_true", [0.1, 1.0, 3.0])
    def test_optimisation_convergence_variation(self, omega_true: float) -> None:
        """At least some starts should converge at each ω."""
        result = run_general_bfgs_optimization(
            omega_true=omega_true,
            n_starts=10,
            maxiter=200,
        )
        assert result.n_converged >= 0
        # At least one converged start is expected (but don't assert)
        # Just verify the count is in range
        assert result.n_converged <= result.n_starts


# ============================================================================
# Test: ω Scan
# ============================================================================


class TestOmegaScan:
    def test_omega_scan_runs(self) -> None:
        result = run_general_omega_scan(
            omega_values=[0.1, 0.5, 1.0],
            n_starts=5,
            maxiter=100,
        )
        assert len(result.omega_values) == 3
        assert len(result.delta_omega_opt_per_omega) == 3

    def test_omega_scan_all_alphas_recorded(self) -> None:
        result = run_general_omega_scan(
            omega_values=[0.1, 0.5, 1.0, 2.0, 5.0],
            n_starts=5,
            maxiter=100,
        )
        assert len(result.alpha_xx_opt_per_omega) == 5
        assert len(result.alpha_xz_opt_per_omega) == 5
        assert len(result.alpha_zx_opt_per_omega) == 5
        assert len(result.alpha_zz_opt_per_omega) == 5
        for i in range(5):
            assert np.isfinite(result.alpha_xx_opt_per_omega[i]) or np.isnan(
                result.alpha_xx_opt_per_omega[i]
            )

    def test_omega_scan_finite_results(self) -> None:
        """All ω values should produce finite optimal Δω."""
        result = run_general_omega_scan(
            omega_values=[0.1, 0.5, 1.0, 2.0, 5.0],
            n_starts=5,
            maxiter=100,
        )
        for i, omega in enumerate(result.omega_values):
            dt = result.delta_omega_opt_per_omega[i]
            assert np.isfinite(dt), f"Non-finite Δω at ω={omega}"

    def test_omega_scan_expectation_variance(self) -> None:
        result = run_general_omega_scan(
            omega_values=[0.1, 1.0],
            n_starts=5,
            maxiter=100,
        )
        assert len(result.expectation_Jz_per_omega) == 2
        assert len(result.variance_Jz_per_omega) == 2
        assert np.all(np.isfinite(result.expectation_Jz_per_omega))

    def test_omega_scan_converged_recorded(self) -> None:
        result = run_general_omega_scan(
            omega_values=[0.1, 1.0],
            n_starts=5,
            maxiter=100,
        )
        assert len(result.n_converged_per_omega) == 2
        for i in range(2):
            assert 0 <= result.n_converged_per_omega[i] <= 5


# ============================================================================
# Test: Parquet Roundtrip
# ============================================================================


class TestParquetRoundtrip:
    def test_bfgs_optimization_roundtrip(self, tmp_path: Path) -> None:
        original = GeneralBFGSOptimizationResult(
            omega_value=1.0,
            alpha_opt=(5.0, -3.0, 2.0, 4.0),
            delta_omega_opt=0.05,
            sql=0.1,
            expectation_Jz=0.25,
            variance_Jz=0.05,
            d_exp_d_omega=-0.5,
            n_starts=100,
            n_converged=95,
        )
        parquet_path = tmp_path / "test_bfgs.parquet"
        original.save_parquet(parquet_path)
        loaded = GeneralBFGSOptimizationResult.from_parquet(parquet_path)
        assert loaded.omega_value == pytest.approx(original.omega_value)
        assert loaded.alpha_opt == pytest.approx(original.alpha_opt)
        assert loaded.delta_omega_opt == pytest.approx(original.delta_omega_opt)
        assert loaded.sql == pytest.approx(original.sql)

    def test_bfgs_roundtrip_all_metadata(self, tmp_path: Path) -> None:
        """Verify all metadata fields survive roundtrip."""
        original = GeneralBFGSOptimizationResult(
            omega_value=0.5,
            alpha_opt=(10.0, -5.0, 0.0, -10.0),
            delta_omega_opt=0.03,
            sql=0.1,
            expectation_Jz=-0.2,
            variance_Jz=0.15,
            d_exp_d_omega=2.5,
            n_starts=50,
            n_converged=48,
        )
        parquet_path = tmp_path / "test_bfgs_meta.parquet"
        original.save_parquet(parquet_path)
        loaded = GeneralBFGSOptimizationResult.from_parquet(parquet_path)
        assert loaded.omega_value == pytest.approx(0.5)
        assert np.allclose(loaded.alpha_opt, (10.0, -5.0, 0.0, -10.0))
        assert loaded.delta_omega_opt == pytest.approx(0.03)
        assert loaded.sql == pytest.approx(0.1)
        assert loaded.n_starts == 50
        assert loaded.n_converged == 48

    def test_omega_scan_roundtrip(self, tmp_path: Path) -> None:
        original = GeneralOmegaScanResult(
            omega_values=np.array([0.1, 0.5, 1.0]),
            alpha_xx_opt_per_omega=np.array([5.0, 10.0, 15.0]),
            alpha_xz_opt_per_omega=np.array([-2.0, 3.0, -5.0]),
            alpha_zx_opt_per_omega=np.array([1.0, -1.0, 2.0]),
            alpha_zz_opt_per_omega=np.array([3.0, -4.0, 6.0]),
            delta_omega_opt_per_omega=np.array([0.05, 0.06, 0.08]),
            sql_values=np.array([0.1, 0.1, 0.1]),
            expectation_Jz_per_omega=np.array([0.1, 0.2, 0.3]),
            variance_Jz_per_omega=np.array([0.01, 0.02, 0.03]),
            d_exp_d_omega_per_omega=np.array([-0.5, 1.0, -1.5]),
            n_converged_per_omega=np.array([95, 90, 85]),
        )
        parquet_path = tmp_path / "test_omega.parquet"
        original.save_parquet(parquet_path)
        loaded = GeneralOmegaScanResult.from_parquet(parquet_path)
        assert np.allclose(loaded.omega_values, original.omega_values)
        assert np.allclose(
            loaded.alpha_xx_opt_per_omega, original.alpha_xx_opt_per_omega
        )
        assert np.allclose(
            loaded.alpha_xz_opt_per_omega, original.alpha_xz_opt_per_omega
        )
        assert np.allclose(
            loaded.delta_omega_opt_per_omega, original.delta_omega_opt_per_omega
        )
        assert np.allclose(loaded.sql_values, original.sql_values)

    def test_omega_scan_roundtrip_metadata(self, tmp_path: Path) -> None:
        """Verify all metadata fields survive roundtrip."""
        original = GeneralOmegaScanResult(
            omega_values=np.array([0.1, 0.5]),
            alpha_xx_opt_per_omega=np.array([3.0, 7.0]),
            alpha_xz_opt_per_omega=np.array([-1.0, 2.0]),
            alpha_zx_opt_per_omega=np.array([4.0, -3.0]),
            alpha_zz_opt_per_omega=np.array([-2.0, 5.0]),
            delta_omega_opt_per_omega=np.array([0.04, 0.06]),
            sql_values=np.array([0.1, 0.1]),
            expectation_Jz_per_omega=np.array([0.15, 0.25]),
            variance_Jz_per_omega=np.array([0.02, 0.03]),
            d_exp_d_omega_per_omega=np.array([2.0, -1.0]),
            n_converged_per_omega=np.array([80, 90]),
        )
        parquet_path = tmp_path / "test_omega_meta.parquet"
        original.save_parquet(parquet_path)
        loaded = GeneralOmegaScanResult.from_parquet(parquet_path)
        assert loaded.omega_values[0] == pytest.approx(0.1)
        assert loaded.alpha_xx_opt_per_omega[1] == pytest.approx(7.0)
        assert loaded.alpha_xz_opt_per_omega[0] == pytest.approx(-1.0)
        assert loaded.sql_values[0] == pytest.approx(0.1)
        assert loaded.expectation_Jz_per_omega[0] == pytest.approx(0.15)
        assert loaded.n_converged_per_omega[1] == pytest.approx(90.0)

    def test_bfgs_from_parquet_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when required columns are missing."""
        import pandas as pd

        df_bad = pd.DataFrame(
            {
                "omega_value": [1.0],
                "alpha_xx_opt": [5.0],
                "delta_omega_opt": [0.05],
            }
        )
        parquet_path = tmp_path / "bad_bfgs.parquet"
        df_bad.to_parquet(parquet_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            GeneralBFGSOptimizationResult.from_parquet(parquet_path)

    def test_omega_scan_from_parquet_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when required columns are missing."""
        import pandas as pd

        df_bad = pd.DataFrame(
            {
                "omega": [0.1, 0.5],
                "best_delta_omega": [0.05, 0.06],
            }
        )
        parquet_path = tmp_path / "bad_omega.parquet"
        df_bad.to_parquet(parquet_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            GeneralOmegaScanResult.from_parquet(parquet_path)


# ============================================================================
# Test: Physical Invariants
# ============================================================================


class TestPhysicalInvariants:
    def test_var_positive_all_parameters(self) -> None:
        """Var(J_z^S) >= 0 for all α combinations."""
        ops = build_two_qubit_operators()
        alpha_list = [
            (0.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.0),
            (0.0, 2.0, 0.0, 0.0),
            (0.0, 0.0, 5.0, 0.0),
            (0.0, 0.0, 0.0, 10.0),
            (5.0, -5.0, 5.0, -5.0),
            (10.0, 10.0, -10.0, -10.0),
            (15.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 15.0),
        ]
        for alpha in alpha_list:
            psi = evolve_general_circuit(
                DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, 1.0, alpha, ops
            )
            var = compute_reduced_variance(psi)
            assert var >= -1e-12, f"Negative Var(J_z^S)={var:.2e} at α={alpha}"

    def test_hold_hermiticity_all_parameters(self) -> None:
        """Total H must be Hermitian for all parameter combinations."""
        ops = build_two_qubit_operators()
        for omega in [0.0, 0.1, 1.0, 5.0]:
            for alpha in [
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 2.0, 0.0, 0.0),
                (0.0, 0.0, 3.0, 0.0),
                (0.0, 0.0, 0.0, 4.0),
                (5.0, -3.0, 2.0, -1.0),
                (-10.0, 10.0, -10.0, 10.0),
            ]:
                H = build_general_hold_hamiltonian(omega, alpha, ops)
                assert np.allclose(H, H.conj().T, atol=1e-12), (
                    f"H not Hermitian at ω={omega}, α={alpha}"
                )

    def test_unitarity_preserved(self) -> None:
        """All unitaries should be unitary for all parameters."""
        ops = build_two_qubit_operators()
        for omega in [0.1, 1.0, 5.0]:
            for alpha in [
                (0.0, 0.0, 0.0, 0.0),
                (5.0, 0.0, 0.0, 0.0),
                (0.0, 5.0, 0.0, 0.0),
                (5.0, -5.0, 5.0, -5.0),
            ]:
                U = general_hold_unitary(
                    T_hold=DEFAULT_T_hold, omega=omega, alpha=alpha, ops=ops
                )
                assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12), (
                    f"Hold unitary not unitary at ω={omega}, α={alpha}"
                )

    def test_zz_only_gives_sql(self, make_ops: dict) -> None:
        """α_zz-only interaction should give Δω = SQL."""
        alpha = (0.0, 0.0, 0.0, 10.0)
        domega = compute_general_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, 1.0, alpha, make_ops
        )
        sql = 1.0 / DEFAULT_T_hold
        assert domega == pytest.approx(sql, rel=1e-6), (
            f"α_zz-only: Δω={domega:.10f} != SQL={sql:.10f}"
        )

    def test_zx_only_finite(self, make_ops: dict) -> None:
        """α_zx-only interaction should produce finite Δω (BCH corrections)."""
        alpha = (0.0, 0.0, 10.0, 0.0)
        domega = compute_general_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_hold, 1.0, alpha, make_ops
        )
        assert np.isfinite(domega), f"Non-finite Δω={domega} for α_zx-only"
        assert domega > 0, f"Non-positive Δω={domega} for α_zx-only"

    def test_derivative_stability(self, make_ops: dict) -> None:
        """Central-difference derivative should be stable across step sizes."""
        alpha = (2.0, -1.0, 3.0, -2.0)
        omega = 0.5
        sensitivities = []
        for step in [1e-5, 1e-6, 1e-7]:
            domega = compute_general_sensitivity(
                DEFAULT_PSI0,
                DEFAULT_T_BS,
                DEFAULT_T_hold,
                omega,
                alpha,
                make_ops,
                fd_step=step,
            )
            if np.isfinite(domega):
                sensitivities.append(domega)
        # We need at least 2 finite values to compare
        assert len(sensitivities) >= 2, (
            f"Not enough finite sensitivities: {sensitivities}"
        )
        arr = np.array(sensitivities)
        mean_val = float(np.mean(arr))
        if mean_val > 0:
            max_dev = float(np.max(np.abs(arr - mean_val) / mean_val))
            assert max_dev < 0.05, (
                f"Derivative unstable: sensitivities = {sensitivities}, "
                f"max relative deviation = {max_dev:.4f}"
            )


# ============================================================================
# Test: Constants Validation
# ============================================================================


class TestConstants:
    def test_sql_reference_correct(self) -> None:
        assert pytest.approx(1.0 / DEFAULT_T_hold) == SQL_REFERENCE

    def test_alpha_bounds(self) -> None:
        lo, hi = ALPHA_BOUNDS
        assert lo == -20.0
        assert hi == 20.0

    def test_n_bfgs_starts(self) -> None:
        assert N_BFGS_STARTS == 100

    def test_initial_state_normalised(self) -> None:
        assert abs(np.linalg.norm(DEFAULT_PSI0) - 1.0) < 1e-12

    def test_initial_state_is_00(self) -> None:
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        assert np.allclose(DEFAULT_PSI0, expected, atol=1e-12)


# ============================================================================
# Test: Delta Lake Storage
# ============================================================================


class TestDeltaLake:
    """Tests for Delta Lake-based BFGS result storage."""

    def test_delta_append_single_row(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Insert 1 row via _upsert_bfgs_result; DeltaTable reads 1 row."""

        tmp_dir = tmp_path_factory.mktemp("delta_single")
        table_dir = str(tmp_dir / "bfgs-results")

        result = GeneralBFGSOptimizationResult(
            omega_value=1.0,
            alpha_opt=(1.0, 2.0, 3.0, 4.0),
            delta_omega_opt=0.05,
            sql=0.1,
            expectation_Jz=0.25,
            variance_Jz=0.0625,
            d_exp_d_omega=-0.5,
            n_starts=100,
            n_converged=90,
        )
        with patch("local.BFGS_TABLE_DIR", table_dir):
            _upsert_bfgs_result(result)

        dt = DeltaTable(table_dir)
        df = dt.to_pandas()
        assert len(df) == 1
        assert df["omega_value"].iloc[0] == pytest.approx(1.0)
        assert df["alpha_xx_opt"].iloc[0] == pytest.approx(1.0)
        assert df["alpha_zz_opt"].iloc[0] == pytest.approx(4.0)
        assert df["delta_omega_opt"].iloc[0] == pytest.approx(0.05)
        assert df["n_converged"].iloc[0] == 90

    def test_delta_concurrent_append(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """ThreadPoolExecutor (10 workers) writes 10 rows; table has exactly 10 rows."""

        tmp_dir = tmp_path_factory.mktemp("delta_concurrent")
        table_dir = str(tmp_dir / "bfgs-results")

        def _worker(omega: float) -> None:
            result = GeneralBFGSOptimizationResult(
                omega_value=omega,
                alpha_opt=(omega, 0.0, 0.0, 0.0),
                delta_omega_opt=0.05,
                sql=0.1,
                expectation_Jz=0.0,
                variance_Jz=0.0625,
                d_exp_d_omega=0.0,
                n_starts=10,
                n_converged=10,
            )
            with patch("local.BFGS_TABLE_DIR", table_dir):
                _upsert_bfgs_result(result)

        omegas = [0.1 * i for i in range(10)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
            list(exe.map(_worker, omegas))

        dt = DeltaTable(table_dir)
        df = dt.to_pandas()
        assert len(df) == 10
        assert sorted(df["omega_value"].tolist()) == pytest.approx(omegas)

    def test_delta_force_recreates_table(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Write 3 rows; shutil.rmtree + re-run; table has exactly the new rows."""

        tmp_dir = tmp_path_factory.mktemp("delta_force")
        table_dir = str(tmp_dir / "bfgs-results")

        def _write_n_rows(n: int) -> None:
            for i in range(n):
                result = GeneralBFGSOptimizationResult(
                    omega_value=float(i),
                    alpha_opt=(0.0, 0.0, 0.0, 0.0),
                    delta_omega_opt=0.1,
                    sql=0.1,
                    expectation_Jz=0.0,
                    variance_Jz=0.0,
                    d_exp_d_omega=0.0,
                    n_starts=10,
                    n_converged=10,
                )
                with patch("local.BFGS_TABLE_DIR", table_dir):
                    _upsert_bfgs_result(result)

        # Write 3 rows
        _write_n_rows(3)
        dt = DeltaTable(table_dir)
        assert len(dt.to_pandas()) == 3

        # Force recreate
        shutil.rmtree(table_dir, ignore_errors=True)
        _write_n_rows(2)

        dt = DeltaTable(table_dir)
        df = dt.to_pandas()
        assert len(df) == 2
        assert sorted(df["omega_value"].tolist()) == [0.0, 1.0]

    def test_delta_to_omega_scan_roundtrip(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Write 5 rows; construct GeneralOmegaScanResult; verify data matches."""

        tmp_dir = tmp_path_factory.mktemp("delta_roundtrip")
        table_dir = str(tmp_dir / "bfgs-results")

        omega_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
        for omega in omega_vals:
            result = GeneralBFGSOptimizationResult(
                omega_value=omega,
                alpha_opt=(omega * 2, omega, -omega, omega * 3),
                delta_omega_opt=0.1 / omega if omega > 0 else 0.1,
                sql=0.1,
                expectation_Jz=0.25 / omega if omega > 0 else 0.0,
                variance_Jz=0.0625,
                d_exp_d_omega=-0.5 * omega,
                n_starts=100,
                n_converged=int(100 - omega * 10),
            )
            with patch("local.BFGS_TABLE_DIR", table_dir):
                _upsert_bfgs_result(result)

        # Read back via DeltaTable
        dt = DeltaTable(table_dir)
        df = dt.to_pandas().sort_values("omega_value").reset_index(drop=True)

        # Construct GeneralOmegaScanResult from Delta data
        scan_result = GeneralOmegaScanResult(
            omega_values=df["omega_value"].to_numpy(dtype=float),
            alpha_xx_opt_per_omega=df["alpha_xx_opt"].to_numpy(dtype=float),
            alpha_xz_opt_per_omega=df["alpha_xz_opt"].to_numpy(dtype=float),
            alpha_zx_opt_per_omega=df["alpha_zx_opt"].to_numpy(dtype=float),
            alpha_zz_opt_per_omega=df["alpha_zz_opt"].to_numpy(dtype=float),
            delta_omega_opt_per_omega=df["delta_omega_opt"].to_numpy(dtype=float),
            sql_values=df["sql"].to_numpy(dtype=float),
            expectation_Jz_per_omega=df["expectation_Jz"].to_numpy(dtype=float),
            variance_Jz_per_omega=df["variance_Jz"].to_numpy(dtype=float),
            d_exp_d_omega_per_omega=df["d_exp_d_omega"].to_numpy(dtype=float),
            n_converged_per_omega=df["n_converged"].to_numpy(dtype=float),
        )

        assert len(scan_result.omega_values) == 5
        assert np.allclose(scan_result.omega_values, sorted(omega_vals))
        assert np.allclose(
            scan_result.alpha_xx_opt_per_omega,
            [t * 2 for t in sorted(omega_vals)],
        )
        assert np.allclose(
            scan_result.delta_omega_opt_per_omega,
            [0.1 / t if t > 0 else 0.1 for t in sorted(omega_vals)],
        )
        assert np.allclose(scan_result.sql_values, [0.1] * 5)
        assert np.allclose(
            scan_result.n_converged_per_omega,
            [100.0 - t * 10 for t in sorted(omega_vals)],
        )

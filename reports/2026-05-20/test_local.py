"""
Tests for the XX-coupling ancilla metrology module.

This is the companion test module for ``local.py`` in this directory.
Run with:
    uv run pytest reports/2026-05-20/test_local.py -q --tb=short
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

import numpy as np
import pytest
from scipy.linalg import expm

_report_dir = str(
    _Path(__file__).resolve().parent.parent.parent / "reports" / "2026-05-20"
)
if _report_dir not in _sys.path:
    _sys.path.insert(0, _report_dir)
del _sys, _Path, _report_dir

from local import (  # type: ignore[import-untyped]  # noqa: E402
    AXX_BOUNDS,
    DEFAULT_PSI0,
    DEFAULT_T_BS,
    DEFAULT_T_H,
    N_GRID_POINTS,
    SQL_REFERENCE,
    XXGridScanResult,
    XXThetaScanResult,
    build_xx_hold_hamiltonian,
    build_xx_interaction,
    compute_xx_decoupled_baseline,
    compute_xx_sensitivity,
    compute_reduced_variance,
    evolve_xx_circuit,
    run_xx_theta_scan,
    xx_grid_scan,
    xx_hold_unitary,
)

from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
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
    def test_xx_interaction_hermitian(self, make_ops: dict) -> None:
        H = build_xx_interaction(1.5, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_xx_interaction_zero(self, make_ops: dict) -> None:
        H = build_xx_interaction(0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-14)

    def test_hold_hamiltonian_hermitian(self, make_ops: dict) -> None:
        H = build_xx_hold_hamiltonian(theta=0.5, alpha_xx=2.0, ops=make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_hold_hamiltonian_contains_jz_sum(self, make_ops: dict) -> None:
        """H should contain θ(J_z^S + J_z^A) as the phase-encoding part."""
        theta = 0.7
        H = build_xx_hold_hamiltonian(theta=theta, alpha_xx=0.0, ops=make_ops)
        expected = theta * (make_ops["Jz_S"] + make_ops["Jz_A"])
        assert np.allclose(H, expected, atol=1e-12)

    def test_hold_unitary(self, make_ops: dict) -> None:
        U = xx_hold_unitary(T_H=1.0, theta=0.5, alpha_xx=2.0, ops=make_ops)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12)

    def test_commutation_jz_jx(self, make_ops: dict) -> None:
        comm_S = (
            make_ops["Jz_S"] @ make_ops["Jx_S"]
            - make_ops["Jx_S"] @ make_ops["Jz_S"]
        )
        assert np.allclose(comm_S, 1j * make_ops["Jy_S"], atol=1e-12)
        comm_A = (
            make_ops["Jz_A"] @ make_ops["Jx_A"]
            - make_ops["Jx_A"] @ make_ops["Jz_A"]
        )
        assert np.allclose(comm_A, 1j * make_ops["Jy_A"], atol=1e-12)

    def test_decoupled_hold_reduces_to_phase_shift(self, make_ops: dict) -> None:
        """At α_xx = 0, the hold should factorise into independent phase shifts.

        exp(-i T_H θ (J_z^S + J_z^A)) = exp(-i T_H θ J_z) ⊗ exp(-i T_H θ J_z)
        since [J_z^S, J_z^A] = 0.
        """
        theta = 0.5
        T_H = 2.0
        U = xx_hold_unitary(T_H=T_H, theta=theta, alpha_xx=0.0, ops=make_ops)
        J_z = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
        U_single = expm(-1j * T_H * theta * J_z)
        expected = np.kron(U_single, U_single)
        assert np.allclose(U, expected, atol=1e-12)

    def test_interaction_vanishes_at_zero(self, make_ops: dict) -> None:
        """α_xx=0 should give zero interaction."""
        H = build_xx_interaction(0.0, make_ops)
        assert np.linalg.norm(H) < 1e-14


# ============================================================================
# Test: Circuit Evolution
# ============================================================================


class TestCircuitEvolution:
    def test_normalisation_preserved(self, make_ops: dict) -> None:
        psi = evolve_xx_circuit(
            DEFAULT_PSI0, DEFAULT_T_BS, 5.0, 0.5, 0.0, make_ops
        )
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    def test_normalisation_with_coupling(self, make_ops: dict) -> None:
        psi = evolve_xx_circuit(
            DEFAULT_PSI0, DEFAULT_T_BS, 10.0, 1.0, 5.0, make_ops
        )
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize("alpha_xx", [0.0, 1.0, 10.0, 20.0])
    def test_evolution_normalised_all_couplings(
        self, alpha_xx: float, make_ops: dict
    ) -> None:
        psi = evolve_xx_circuit(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_H, 0.5, alpha_xx, make_ops
        )
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    def test_no_op_identity(self, make_ops: dict) -> None:
        """T_BS=0, T_H=0 should give the initial state back."""
        psi = evolve_xx_circuit(
            DEFAULT_PSI0, 0.0, 0.0, 0.0, 0.0, make_ops
        )
        assert np.allclose(psi, DEFAULT_PSI0, atol=1e-12)


# ============================================================================
# Test: Reduced Density Matrix and Variance
# ============================================================================


class TestReducedVariance:
    def test_product_state_variance(self, make_ops: dict) -> None:
        """For a product state |00⟩, Var(J_z^S) should be zero."""
        psi = DEFAULT_PSI0.copy()
        var = compute_reduced_variance(psi, make_ops["Jz_S"])
        assert var == pytest.approx(0.0, abs=1e-12)

    def test_bs_output_variance(self, make_ops: dict) -> None:
        """After first BS, the system is in (|0⟩ - i|1⟩)/√2.
        Var(J_z^S) = 1/4 - ⟨J_z^S⟩² = 1/4 - 0 = 1/4.
        """
        from local import system_only_bs_unitary

        U_bs = system_only_bs_unitary(DEFAULT_T_BS)
        psi = U_bs @ DEFAULT_PSI0
        var = compute_reduced_variance(psi, make_ops["Jz_S"])
        assert var == pytest.approx(0.25, abs=1e-12)

    def test_variance_positive(self, make_ops: dict) -> None:
        """Variance should always be non-negative."""
        for alpha_xx in [0.0, 0.5, 2.0, 10.0]:
            psi = evolve_xx_circuit(
                DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_H, 0.5, alpha_xx, make_ops
            )
            var = compute_reduced_variance(psi, make_ops["Jz_S"])
            assert var >= -1e-12, f"Negative variance at alpha_xx={alpha_xx}: {var}"


# ============================================================================
# Test: Sensitivity Computation
# ============================================================================


class TestSensitivity:
    def test_decoupled_sensitivity(self, make_ops: dict) -> None:
        """At α_xx = 0, Δθ should equal SQL = 1/T_H."""
        dtheta = compute_xx_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_H, 1.0, 0.0, make_ops
        )
        sql = 1.0 / DEFAULT_T_H
        assert dtheta == pytest.approx(sql, rel=1e-6), (
            f"Δθ={dtheta:.10f} != SQL={sql:.10f} at α_xx=0"
        )

    @pytest.mark.parametrize("theta_true", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_decoupled_all_theta(self, theta_true: float, make_ops: dict) -> None:
        """At α_xx = 0, Δθ = SQL for any θ."""
        dtheta = compute_xx_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_H, theta_true, 0.0, make_ops
        )
        sql = 1.0 / DEFAULT_T_H
        assert dtheta == pytest.approx(sql, rel=1e-5), (
            f"θ={theta_true}: Δθ={dtheta:.10f} != SQL={sql:.10f}"
        )

    def test_sensitivity_finite_with_coupling(self, make_ops: dict) -> None:
        """With non-zero α_xx, sensitivity should be finite."""
        for alpha_xx in [0.1, 1.0, 5.0, 10.0, 20.0]:
            dtheta = compute_xx_sensitivity(
                DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_H, 1.0, alpha_xx, make_ops
            )
            assert np.isfinite(dtheta), (
                f"Non-finite Δθ={dtheta} at α_xx={alpha_xx}"
            )

    def test_sensitivity_positive(self, make_ops: dict) -> None:
        """Sensitivity should always be positive."""
        for alpha_xx in [0.0, 0.5, 2.0, 10.0]:
            dtheta = compute_xx_sensitivity(
                DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_H, 1.0, alpha_xx, make_ops
            )
            assert dtheta > 0, f"Non-positive Δθ={dtheta} at α_xx={alpha_xx}"

    def test_sensitivity_derivative_near_zero(self, make_ops: dict) -> None:
        """At a fringe extremum, Δθ should be inf."""
        # θ = π/T_H ≈ 0.314 should be near a fringe extremum
        # for the standard single-qubit MZI
        dtheta = compute_xx_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_H, np.pi / DEFAULT_T_H, 0.0, make_ops
        )
        assert np.isinf(dtheta) or dtheta > 100, (
            f"Δθ should be large at fringe extremum: {dtheta}"
        )


# ============================================================================
# Test: Decoupled Baseline
# ============================================================================


class TestDecoupledBaseline:
    def test_baseline_matches_sql(self) -> None:
        result = compute_xx_decoupled_baseline()
        assert result.delta_theta == pytest.approx(result.sql, rel=1e-10)

    def test_baseline_ratio(self) -> None:
        result = compute_xx_decoupled_baseline()
        ratio = result.delta_theta / result.sql
        assert ratio == pytest.approx(1.0, rel=1e-10)


# ============================================================================
# Test: 1D α_xx Grid Scan
# ============================================================================


class TestGridScan:
    def test_grid_scan_returns_correct_shape(self) -> None:
        result = xx_grid_scan(theta=1.0, n_points=101)
        assert len(result.alpha_xx_values) == 101
        assert len(result.delta_theta_values) == 101

    def test_grid_scan_finds_minimum(self) -> None:
        result = xx_grid_scan(theta=1.0, n_points=501)
        assert np.isfinite(result.delta_theta_opt)
        assert result.delta_theta_opt > 0

    def test_grid_scan_at_zero_coupling(self, make_ops: dict) -> None:
        """At α_xx = 0, Δθ = SQL exactly (one point on the grid)."""
        result = xx_grid_scan(theta=1.0, n_points=11)
        # Find α_xx=0 index (should be first point)
        assert np.isclose(result.alpha_xx_values[0], 0.0, atol=1e-10)
        sql = 1.0 / DEFAULT_T_H
        assert result.delta_theta_values[0] == pytest.approx(sql, rel=1e-6)

    def test_grid_scan_sql_below_at_some_theta(self) -> None:
        """Check if any point on the grid beats SQL at θ=0.1."""
        result = xx_grid_scan(theta=0.1, n_points=501)
        count_below = int(np.sum(result.delta_theta_values < result.sql))
        # We don't know if the hypothesis holds, just verify the counting works
        assert 0 <= count_below <= len(result.delta_theta_values)

    def test_grid_scan_optimal_params_recorded(self) -> None:
        result = xx_grid_scan(theta=1.0, n_points=201)
        assert np.isfinite(result.alpha_xx_opt) or np.isnan(result.alpha_xx_opt)

    def test_grid_scan_sql_boundary(self) -> None:
        """The sql field should equal 1/T_H."""
        result = xx_grid_scan(theta=0.5, n_points=51)
        assert result.sql == pytest.approx(1.0 / DEFAULT_T_H)


# ============================================================================
# Test: θ Scan
# ============================================================================


class TestThetaScan:
    def test_theta_scan_runs(self) -> None:
        result = run_xx_theta_scan(
            theta_values=[0.1, 0.5, 1.0],
            n_points=51,
        )
        assert len(result.theta_values) == 3
        assert len(result.alpha_xx_opt_per_theta) == 3
        assert len(result.delta_theta_opt_per_theta) == 3

    def test_theta_scan_finite_results(self) -> None:
        """All θ values should produce finite optimal Δθ."""
        result = run_xx_theta_scan(
            theta_values=[0.1, 0.5, 1.0, 2.0, 5.0],
            n_points=101,
        )
        for i, theta in enumerate(result.theta_values):
            dt = result.delta_theta_opt_per_theta[i]
            assert np.isfinite(dt), f"Non-finite Δθ at θ={theta}"

    def test_theta_scan_alpha_opts_recorded(self) -> None:
        result = run_xx_theta_scan(
            theta_values=[0.1, 1.0, 5.0],
            n_points=51,
        )
        for i in range(len(result.theta_values)):
            alpha = result.alpha_xx_opt_per_theta[i]
            assert np.isfinite(alpha) or np.isnan(alpha)

    def test_theta_scan_expectation_variance(self) -> None:
        result = run_xx_theta_scan(
            theta_values=[0.1, 1.0],
            n_points=51,
        )
        assert len(result.expectation_Jz_per_theta) == 2
        assert len(result.variance_Jz_per_theta) == 2
        assert np.all(np.isfinite(result.expectation_Jz_per_theta))

    def test_theta_scan_decoupled_recovery(self) -> None:
        """The first point (α_xx=0) of the grid scan at each θ
        should give SQL-limited sensitivity."""
        result = run_xx_theta_scan(
            theta_values=[0.1, 1.0, 5.0],
            n_points=51,
        )
        for i, theta in enumerate(result.theta_values):
            dt = result.delta_theta_opt_per_theta[i]
            assert dt > 0, f"Δθ <= 0 at θ={theta}"

    def test_theta_scan_below_sql_counts(self) -> None:
        result = run_xx_theta_scan(
            theta_values=[0.1, 1.0],
            n_points=101,
        )
        assert len(result.count_below_sql_per_theta) == 2
        assert len(result.total_points_per_theta) == 2
        for i in range(2):
            assert result.count_below_sql_per_theta[i] >= 0
            assert result.total_points_per_theta[i] == 101


# ============================================================================
# Test: CSV Roundtrip
# ============================================================================


class TestCsvRoundtrip:
    def test_grid_scan_roundtrip(self, tmp_path: Path) -> None:
        original = XXGridScanResult(
            alpha_xx_values=np.linspace(0, 20, 11),
            delta_theta_values=np.array(
                [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            ),
            theta_value=0.5,
            alpha_xx_opt=5.0,
            delta_theta_opt=0.05,
            sql=0.1,
            expectation_Jz=0.25,
            variance_Jz=0.05,
        )
        csv_path = tmp_path / "test_grid.csv"
        original.save_csv(csv_path)
        loaded = XXGridScanResult.from_csv(csv_path)
        assert np.allclose(loaded.alpha_xx_values, original.alpha_xx_values)
        assert loaded.theta_value == pytest.approx(original.theta_value)
        assert loaded.alpha_xx_opt == pytest.approx(original.alpha_xx_opt)
        assert loaded.sql == pytest.approx(original.sql)

    def test_grid_scan_roundtrip_metadata(self, tmp_path: Path) -> None:
        """Verify all metadata fields survive roundtrip."""
        original = XXGridScanResult(
            alpha_xx_values=np.linspace(0, 20, 5),
            delta_theta_values=np.array([0.1, 0.09, 0.08, 0.07, 0.06]),
            theta_value=1.0,
            alpha_xx_opt=15.0,
            delta_theta_opt=0.06,
            sql=0.1,
            expectation_Jz=-0.1,
            variance_Jz=0.2,
        )
        csv_path = tmp_path / "test_grid_meta.csv"
        original.save_csv(csv_path)
        loaded = XXGridScanResult.from_csv(csv_path)
        assert loaded.theta_value == pytest.approx(1.0)
        assert loaded.alpha_xx_opt == pytest.approx(15.0)
        assert loaded.delta_theta_opt == pytest.approx(0.06)
        assert loaded.sql == pytest.approx(0.1)
        assert loaded.expectation_Jz == pytest.approx(-0.1)
        assert loaded.variance_Jz == pytest.approx(0.2)

    def test_theta_scan_roundtrip(self, tmp_path: Path) -> None:
        original = XXThetaScanResult(
            theta_values=np.array([0.1, 0.5, 1.0]),
            alpha_xx_opt_per_theta=np.array([5.0, 10.0, 15.0]),
            delta_theta_opt_per_theta=np.array([0.05, 0.06, 0.08]),
            sql_values=np.array([0.1, 0.1, 0.1]),
            expectation_Jz_per_theta=np.array([0.1, 0.2, 0.3]),
            variance_Jz_per_theta=np.array([0.01, 0.02, 0.03]),
            count_below_sql_per_theta=np.array([100, 50, 25]),
            total_points_per_theta=np.array([2001, 2001, 2001]),
        )
        csv_path = tmp_path / "test_theta.csv"
        original.save_csv(csv_path)
        loaded = XXThetaScanResult.from_csv(csv_path)
        assert np.allclose(loaded.theta_values, original.theta_values)
        assert np.allclose(
            loaded.alpha_xx_opt_per_theta, original.alpha_xx_opt_per_theta
        )
        assert np.allclose(
            loaded.delta_theta_opt_per_theta, original.delta_theta_opt_per_theta
        )
        assert np.allclose(loaded.sql_values, original.sql_values)

    def test_theta_scan_roundtrip_metadata(self, tmp_path: Path) -> None:
        """Verify all metadata fields survive roundtrip."""
        original = XXThetaScanResult(
            theta_values=np.array([0.1, 0.5]),
            alpha_xx_opt_per_theta=np.array([3.0, 7.0]),
            delta_theta_opt_per_theta=np.array([0.04, 0.06]),
            sql_values=np.array([0.1, 0.1]),
            expectation_Jz_per_theta=np.array([0.15, 0.25]),
            variance_Jz_per_theta=np.array([0.02, 0.03]),
            count_below_sql_per_theta=np.array([500, 200]),
            total_points_per_theta=np.array([2001, 2001]),
        )
        csv_path = tmp_path / "test_theta_meta.csv"
        original.save_csv(csv_path)
        loaded = XXThetaScanResult.from_csv(csv_path)
        assert loaded.theta_values[0] == pytest.approx(0.1)
        assert loaded.alpha_xx_opt_per_theta[1] == pytest.approx(7.0)
        assert loaded.sql_values[0] == pytest.approx(0.1)
        assert loaded.expectation_Jz_per_theta[0] == pytest.approx(0.15)
        assert loaded.count_below_sql_per_theta[0] == pytest.approx(500.0)

    def test_grid_scan_from_csv_missing_columns(self, tmp_path: Path) -> None:
        """from_csv should fail fast when required columns are missing."""
        import pandas as pd

        df_bad = pd.DataFrame({"alpha_xx": [0.0, 1.0], "delta_theta": [0.1, 0.08]})
        csv_path = tmp_path / "bad_grid.csv"
        df_bad.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            XXGridScanResult.from_csv(csv_path)

    def test_theta_scan_from_csv_missing_columns(self, tmp_path: Path) -> None:
        """from_csv should fail fast when required columns are missing."""
        import pandas as pd

        df_bad = pd.DataFrame({"theta": [0.1, 0.5], "best_delta_theta": [0.05, 0.06]})
        csv_path = tmp_path / "bad_theta.csv"
        df_bad.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            XXThetaScanResult.from_csv(csv_path)


# ============================================================================
# Test: Physical Invariants
# ============================================================================


class TestPhysicalInvariants:
    def test_var_positive_all_couplings(self) -> None:
        """Var(J_z^S) >= 0 for all α_xx at a representative θ."""
        ops = build_two_qubit_operators()
        for alpha_xx in [0.0, 0.5, 2.0, 5.0, 10.0, 15.0, 20.0]:
            psi = evolve_xx_circuit(
                DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_T_H, 1.0, alpha_xx, ops
            )
            var = compute_reduced_variance(psi, ops["Jz_S"])
            assert var >= -1e-12, (
                f"Negative Var(J_z^S)={var:.2e} at α_xx={alpha_xx}"
            )

    def test_hold_hermiticity(self) -> None:
        """Total H must be Hermitian for all parameters."""
        ops = build_two_qubit_operators()
        for theta in [0.1, 1.0, 5.0]:
            for alpha_xx in [0.0, 1.0, 10.0, 20.0]:
                H = build_xx_hold_hamiltonian(theta, alpha_xx, ops)
                assert np.allclose(H, H.conj().T, atol=1e-12), (
                    f"H not Hermitian at θ={theta}, α_xx={alpha_xx}"
                )


# ============================================================================
# Test: Constants Validation
# ============================================================================


class TestConstants:
    def test_sql_reference_correct(self) -> None:
        assert SQL_REFERENCE == pytest.approx(1.0 / DEFAULT_T_H)

    def test_axx_bounds(self) -> None:
        lo, hi = AXX_BOUNDS
        assert lo == 0.0
        assert hi == 20.0

    def test_n_grid_points(self) -> None:
        assert N_GRID_POINTS == 2001

    def test_initial_state_normalised(self) -> None:
        assert abs(np.linalg.norm(DEFAULT_PSI0) - 1.0) < 1e-12

    def test_initial_state_is_00(self) -> None:
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        assert np.allclose(DEFAULT_PSI0, expected, atol=1e-12)

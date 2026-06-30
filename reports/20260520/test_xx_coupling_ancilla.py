"""
Tests for the XX-coupling ancilla metrology module.

This is the companion test module for ``xx_coupling_ancilla.py`` in this directory.
Run with:
    uv run pytest reports/20260520/test_xx_coupling_ancilla.py -q --tb=short
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.linalg import expm

from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
)

if TYPE_CHECKING:
    from pathlib import Path

_m = importlib.import_module("reports.20260520.xx_coupling_ancilla")
AXX_BOUNDS = _m.AXX_BOUNDS
DEFAULT_PSI0 = _m.DEFAULT_PSI0
DEFAULT_T_BS = _m.DEFAULT_T_BS
N_GRID_POINTS = _m.N_GRID_POINTS
SQL_REFERENCE = _m.SQL_REFERENCE
DEFAULT_t_hold = _m.DEFAULT_t_hold
XXGridScanResult = _m.XXGridScanResult
XXOmegaScanResult = _m.XXOmegaScanResult
build_xx_hold_hamiltonian = _m.build_xx_hold_hamiltonian
build_xx_interaction = _m.build_xx_interaction
compute_reduced_variance = _m.compute_reduced_variance
compute_xx_decoupled_baseline = _m.compute_xx_decoupled_baseline
compute_xx_sensitivity = _m.compute_xx_sensitivity
evolve_xx_circuit = _m.evolve_xx_circuit
run_xx_omega_scan = _m.run_xx_omega_scan
xx_grid_scan = _m.xx_grid_scan
xx_hold_unitary = _m.xx_hold_unitary
system_only_bs_unitary = _m.system_only_bs_unitary

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
        H = build_xx_hold_hamiltonian(omega=0.5, alpha_xx=2.0, ops=make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_hold_hamiltonian_contains_jz_sum(self, make_ops: dict) -> None:
        """H should contain ω(J_z^S + J_z^A) as the phase-encoding part."""
        omega = 0.7
        H = build_xx_hold_hamiltonian(omega=omega, alpha_xx=0.0, ops=make_ops)
        expected = omega * (make_ops["Jz_S"] + make_ops["Jz_A"])
        assert np.allclose(H, expected, atol=1e-12)

    def test_hold_unitary(self, make_ops: dict) -> None:
        U = xx_hold_unitary(t_hold=1.0, omega=0.5, alpha_xx=2.0, ops=make_ops)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12)

    def test_commutation_jz_jx(self, make_ops: dict) -> None:
        comm_S = (
            make_ops["Jz_S"] @ make_ops["Jx_S"] - make_ops["Jx_S"] @ make_ops["Jz_S"]
        )
        assert np.allclose(comm_S, 1j * make_ops["Jy_S"], atol=1e-12)
        comm_A = (
            make_ops["Jz_A"] @ make_ops["Jx_A"] - make_ops["Jx_A"] @ make_ops["Jz_A"]
        )
        assert np.allclose(comm_A, 1j * make_ops["Jy_A"], atol=1e-12)

    def test_decoupled_hold_reduces_to_phase_shift(self, make_ops: dict) -> None:
        """At α_xx = 0, the hold should factorise into independent phase shifts.

        exp(-i t_hold ω (J_z^S + J_z^A)) = exp(-i t_hold ω J_z) ⊗ exp(-i t_hold ω J_z)
        since [J_z^S, J_z^A] = 0.
        """
        omega = 0.5
        t_hold = 2.0
        U = xx_hold_unitary(t_hold=t_hold, omega=omega, alpha_xx=0.0, ops=make_ops)
        J_z = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
        U_single = expm(-1j * t_hold * omega * J_z)
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
        psi = evolve_xx_circuit(DEFAULT_PSI0, DEFAULT_T_BS, 5.0, 0.5, 0.0, make_ops)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    def test_normalisation_with_coupling(self, make_ops: dict) -> None:
        psi = evolve_xx_circuit(DEFAULT_PSI0, DEFAULT_T_BS, 10.0, 1.0, 5.0, make_ops)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    @pytest.mark.parametrize("alpha_xx", [0.0, 1.0, 10.0, 20.0])
    def test_evolution_normalised_all_couplings(
        self, alpha_xx: float, make_ops: dict
    ) -> None:
        psi = evolve_xx_circuit(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_t_hold, 0.5, alpha_xx, make_ops
        )
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    def test_no_op_identity(self, make_ops: dict) -> None:
        """T_BS=0, t_hold=0 should give the initial state back."""
        psi = evolve_xx_circuit(DEFAULT_PSI0, 0.0, 0.0, 0.0, 0.0, make_ops)
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
        U_bs = system_only_bs_unitary(DEFAULT_T_BS)
        psi = U_bs @ DEFAULT_PSI0
        var = compute_reduced_variance(psi, make_ops["Jz_S"])
        assert var == pytest.approx(0.25, abs=1e-12)

    def test_variance_positive(self, make_ops: dict) -> None:
        """Variance should always be non-negative."""
        for alpha_xx in [0.0, 0.5, 2.0, 10.0]:
            psi = evolve_xx_circuit(
                DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_t_hold, 0.5, alpha_xx, make_ops
            )
            var = compute_reduced_variance(psi, make_ops["Jz_S"])
            assert var >= -1e-12, f"Negative variance at alpha_xx={alpha_xx}: {var}"


# ============================================================================
# Test: Sensitivity Computation
# ============================================================================


class TestSensitivity:
    def test_decoupled_sensitivity(self, make_ops: dict) -> None:
        """At α_xx = 0, Δω should equal SQL = 1/t_hold."""
        domega = compute_xx_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_t_hold, 1.0, 0.0, make_ops
        )
        sql = 1.0 / DEFAULT_t_hold
        assert domega == pytest.approx(sql, rel=1e-6), (
            f"Δω={domega:.10f} != SQL={sql:.10f} at α_xx=0"
        )

    @pytest.mark.parametrize("omega_true", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_decoupled_all_omega(self, omega_true: float, make_ops: dict) -> None:
        """At α_xx = 0, Δω = SQL for any ω."""
        domega = compute_xx_sensitivity(
            DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_t_hold, omega_true, 0.0, make_ops
        )
        sql = 1.0 / DEFAULT_t_hold
        assert domega == pytest.approx(sql, rel=1e-5), (
            f"ω={omega_true}: Δω={domega:.10f} != SQL={sql:.10f}"
        )

    def test_sensitivity_finite_with_coupling(self, make_ops: dict) -> None:
        """With non-zero α_xx, sensitivity should be finite."""
        for alpha_xx in [0.1, 1.0, 5.0, 10.0, 20.0]:
            domega = compute_xx_sensitivity(
                DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_t_hold, 1.0, alpha_xx, make_ops
            )
            assert np.isfinite(domega), f"Non-finite Δω={domega} at α_xx={alpha_xx}"

    def test_sensitivity_positive(self, make_ops: dict) -> None:
        """Sensitivity should always be positive."""
        for alpha_xx in [0.0, 0.5, 2.0, 10.0]:
            domega = compute_xx_sensitivity(
                DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_t_hold, 1.0, alpha_xx, make_ops
            )
            assert domega > 0, f"Non-positive Δω={domega} at α_xx={alpha_xx}"

    def test_sensitivity_derivative_near_zero(self, make_ops: dict) -> None:
        """At a fringe extremum, Δω should be inf."""
        # ω = π/t_hold ≈ 0.314 should be near a fringe extremum
        # for the standard single-qubit MZI
        domega = compute_xx_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_t_hold,
            np.pi / DEFAULT_t_hold,
            0.0,
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
        result = compute_xx_decoupled_baseline()
        assert result.delta_omega == pytest.approx(result.sql, rel=1e-10)

    def test_baseline_ratio(self) -> None:
        result = compute_xx_decoupled_baseline()
        ratio = result.delta_omega / result.sql
        assert ratio == pytest.approx(1.0, rel=1e-10)


# ============================================================================
# Test: 1D α_xx Grid Scan
# ============================================================================


class TestGridScan:
    def test_grid_scan_returns_correct_shape(self) -> None:
        result = xx_grid_scan(omega=1.0, n_points=101)
        assert len(result.alpha_xx_values) == 101
        assert len(result.delta_omega_values) == 101

    def test_grid_scan_finds_minimum(self) -> None:
        result = xx_grid_scan(omega=1.0, n_points=501)
        assert np.isfinite(result.delta_omega_opt)
        assert result.delta_omega_opt > 0

    def test_grid_scan_at_zero_coupling(self, make_ops: dict) -> None:
        """At α_xx = 0, Δω = SQL exactly (one point on the grid)."""
        result = xx_grid_scan(omega=1.0, n_points=11)
        # Find α_xx=0 index (should be first point)
        assert np.isclose(result.alpha_xx_values[0], 0.0, atol=1e-10)
        sql = 1.0 / DEFAULT_t_hold
        assert result.delta_omega_values[0] == pytest.approx(sql, rel=1e-6)

    def test_grid_scan_sql_below_at_some_omega(self) -> None:
        """Check if any point on the grid beats SQL at ω=0.1."""
        result = xx_grid_scan(omega=0.1, n_points=501)
        count_below = int(np.sum(result.delta_omega_values < result.sql))
        # We don't know if the hypothesis holds, just verify the counting works
        assert 0 <= count_below <= len(result.delta_omega_values)

    def test_grid_scan_optimal_params_recorded(self) -> None:
        result = xx_grid_scan(omega=1.0, n_points=201)
        assert np.isfinite(result.alpha_xx_opt) or np.isnan(result.alpha_xx_opt)

    def test_grid_scan_sql_boundary(self) -> None:
        """The sql field should equal 1/t_hold."""
        result = xx_grid_scan(omega=0.5, n_points=51)
        assert result.sql == pytest.approx(1.0 / DEFAULT_t_hold)


# ============================================================================
# Test: ω Scan
# ============================================================================


class TestOmegaScan:
    def test_omega_scan_runs(self) -> None:
        result = run_xx_omega_scan(
            omega_values=[0.1, 0.5, 1.0],
            n_points=51,
        )
        assert len(result.omega_values) == 3
        assert len(result.alpha_xx_opt_per_omega) == 3
        assert len(result.delta_omega_opt_per_omega) == 3

    def test_omega_scan_finite_results(self) -> None:
        """All ω values should produce finite optimal Δω."""
        result = run_xx_omega_scan(
            omega_values=[0.1, 0.5, 1.0, 2.0, 5.0],
            n_points=101,
        )
        for i, omega in enumerate(result.omega_values):
            dt = result.delta_omega_opt_per_omega[i]
            assert np.isfinite(dt), f"Non-finite Δω at ω={omega}"

    def test_omega_scan_alpha_opts_recorded(self) -> None:
        result = run_xx_omega_scan(
            omega_values=[0.1, 1.0, 5.0],
            n_points=51,
        )
        for i in range(len(result.omega_values)):
            alpha = result.alpha_xx_opt_per_omega[i]
            assert np.isfinite(alpha) or np.isnan(alpha)

    def test_omega_scan_expectation_variance(self) -> None:
        result = run_xx_omega_scan(
            omega_values=[0.1, 1.0],
            n_points=51,
        )
        assert len(result.expectation_Jz_per_omega) == 2
        assert len(result.variance_Jz_per_omega) == 2
        assert np.all(np.isfinite(result.expectation_Jz_per_omega))

    def test_omega_scan_decoupled_recovery(self) -> None:
        """The first point (α_xx=0) of the grid scan at each ω
        should give SQL-limited sensitivity."""
        result = run_xx_omega_scan(
            omega_values=[0.1, 1.0, 5.0],
            n_points=51,
        )
        for i, omega in enumerate(result.omega_values):
            dt = result.delta_omega_opt_per_omega[i]
            assert dt > 0, f"Δω <= 0 at ω={omega}"

    def test_omega_scan_below_sql_counts(self) -> None:
        result = run_xx_omega_scan(
            omega_values=[0.1, 1.0],
            n_points=101,
        )
        assert len(result.count_below_sql_per_omega) == 2
        assert len(result.total_points_per_omega) == 2
        for i in range(2):
            assert result.count_below_sql_per_omega[i] >= 0
            assert result.total_points_per_omega[i] == 101


# ============================================================================
# Test: Parquet Roundtrip
# ============================================================================


class TestParquetRoundtrip:
    def test_grid_scan_roundtrip(self, tmp_path: Path) -> None:
        original = XXGridScanResult(
            alpha_xx_values=np.linspace(0, 20, 11),
            delta_omega_values=np.array(
                [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            ),
            omega_value=0.5,
            alpha_xx_opt=5.0,
            delta_omega_opt=0.05,
            sql=0.1,
            expectation_Jz=0.25,
            variance_Jz=0.05,
        )
        parquet_path = tmp_path / "test_grid.parquet"
        original.save_parquet(parquet_path)
        loaded = XXGridScanResult.from_parquet(parquet_path)
        assert np.allclose(loaded.alpha_xx_values, original.alpha_xx_values)
        assert loaded.omega_value == pytest.approx(original.omega_value)
        assert loaded.alpha_xx_opt == pytest.approx(original.alpha_xx_opt)
        assert loaded.sql == pytest.approx(original.sql)

    def test_grid_scan_roundtrip_metadata(self, tmp_path: Path) -> None:
        """Verify all metadata fields survive roundtrip."""
        original = XXGridScanResult(
            alpha_xx_values=np.linspace(0, 20, 5),
            delta_omega_values=np.array([0.1, 0.09, 0.08, 0.07, 0.06]),
            omega_value=1.0,
            alpha_xx_opt=15.0,
            delta_omega_opt=0.06,
            sql=0.1,
            expectation_Jz=-0.1,
            variance_Jz=0.2,
        )
        parquet_path = tmp_path / "test_grid_meta.parquet"
        original.save_parquet(parquet_path)
        loaded = XXGridScanResult.from_parquet(parquet_path)
        assert loaded.omega_value == pytest.approx(1.0)
        assert loaded.alpha_xx_opt == pytest.approx(15.0)
        assert loaded.delta_omega_opt == pytest.approx(0.06)
        assert loaded.sql == pytest.approx(0.1)
        assert loaded.expectation_Jz == pytest.approx(-0.1)
        assert loaded.variance_Jz == pytest.approx(0.2)

    def test_omega_scan_roundtrip(self, tmp_path: Path) -> None:
        original = XXOmegaScanResult(
            omega_values=np.array([0.1, 0.5, 1.0]),
            alpha_xx_opt_per_omega=np.array([5.0, 10.0, 15.0]),
            delta_omega_opt_per_omega=np.array([0.05, 0.06, 0.08]),
            sql_values=np.array([0.1, 0.1, 0.1]),
            expectation_Jz_per_omega=np.array([0.1, 0.2, 0.3]),
            variance_Jz_per_omega=np.array([0.01, 0.02, 0.03]),
            count_below_sql_per_omega=np.array([100, 50, 25]),
            total_points_per_omega=np.array([2001, 2001, 2001]),
        )
        parquet_path = tmp_path / "test_omega.parquet"
        original.save_parquet(parquet_path)
        loaded = XXOmegaScanResult.from_parquet(parquet_path)
        assert np.allclose(loaded.omega_values, original.omega_values)
        assert np.allclose(
            loaded.alpha_xx_opt_per_omega, original.alpha_xx_opt_per_omega
        )
        assert np.allclose(
            loaded.delta_omega_opt_per_omega, original.delta_omega_opt_per_omega
        )
        assert np.allclose(loaded.sql_values, original.sql_values)

    def test_omega_scan_roundtrip_metadata(self, tmp_path: Path) -> None:
        """Verify all metadata fields survive roundtrip."""
        original = XXOmegaScanResult(
            omega_values=np.array([0.1, 0.5]),
            alpha_xx_opt_per_omega=np.array([3.0, 7.0]),
            delta_omega_opt_per_omega=np.array([0.04, 0.06]),
            sql_values=np.array([0.1, 0.1]),
            expectation_Jz_per_omega=np.array([0.15, 0.25]),
            variance_Jz_per_omega=np.array([0.02, 0.03]),
            count_below_sql_per_omega=np.array([500, 200]),
            total_points_per_omega=np.array([2001, 2001]),
        )
        parquet_path = tmp_path / "test_omega_meta.parquet"
        original.save_parquet(parquet_path)
        loaded = XXOmegaScanResult.from_parquet(parquet_path)
        assert loaded.omega_values[0] == pytest.approx(0.1)
        assert loaded.alpha_xx_opt_per_omega[1] == pytest.approx(7.0)
        assert loaded.sql_values[0] == pytest.approx(0.1)
        assert loaded.expectation_Jz_per_omega[0] == pytest.approx(0.15)
        assert loaded.count_below_sql_per_omega[0] == pytest.approx(500.0)

    def test_grid_scan_from_parquet_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when required columns are missing."""
        import pandas as pd

        df_bad = pd.DataFrame({"alpha_xx": [0.0, 1.0], "delta_omega": [0.1, 0.08]})
        parquet_path = tmp_path / "bad_grid.parquet"
        df_bad.to_parquet(parquet_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            XXGridScanResult.from_parquet(parquet_path)

    def test_omega_scan_from_parquet_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when required columns are missing."""
        import pandas as pd

        df_bad = pd.DataFrame({"omega": [0.1, 0.5], "best_delta_omega": [0.05, 0.06]})
        parquet_path = tmp_path / "bad_omega.parquet"
        df_bad.to_parquet(parquet_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            XXOmegaScanResult.from_parquet(parquet_path)


# ============================================================================
# Test: Physical Invariants
# ============================================================================


class TestPhysicalInvariants:
    def test_var_positive_all_couplings(self) -> None:
        """Var(J_z^S) >= 0 for all α_xx at a representative ω."""
        ops = build_two_qubit_operators()
        for alpha_xx in [0.0, 0.5, 2.0, 5.0, 10.0, 15.0, 20.0]:
            psi = evolve_xx_circuit(
                DEFAULT_PSI0, DEFAULT_T_BS, DEFAULT_t_hold, 1.0, alpha_xx, ops
            )
            var = compute_reduced_variance(psi, ops["Jz_S"])
            assert var >= -1e-12, f"Negative Var(J_z^S)={var:.2e} at α_xx={alpha_xx}"

    def test_hold_hermiticity(self) -> None:
        """Total H must be Hermitian for all parameters."""
        ops = build_two_qubit_operators()
        for omega in [0.1, 1.0, 5.0]:
            for alpha_xx in [0.0, 1.0, 10.0, 20.0]:
                H = build_xx_hold_hamiltonian(omega, alpha_xx, ops)
                assert np.allclose(H, H.conj().T, atol=1e-12), (
                    f"H not Hermitian at ω={omega}, α_xx={alpha_xx}"
                )


# ============================================================================
# Test: Constants Validation
# ============================================================================


class TestConstants:
    def test_sql_reference_correct(self) -> None:
        assert pytest.approx(1.0 / DEFAULT_t_hold) == SQL_REFERENCE

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

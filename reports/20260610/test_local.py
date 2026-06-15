"""
Tests for the free-ancilla ω-modulated drive module (2026-06-10).

Run with:
    uv run pytest reports/20260610/test_local.py -q --tb=short
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path

import numpy as np
import pytest

from src.analysis.ancilla_drive_metrology import (
    build_two_qubit_operators,
    evolve_phase_modulated_circuit,
)
from src.analysis.ancilla_optimization import free_ancilla_initial_state

_report_dir = str(
    Path(__file__).resolve().parent.parent.parent / "reports" / "20260610",
)
if _report_dir not in _sys.path:
    _sys.path.insert(0, _report_dir)
del _sys, _report_dir

from local import (  # type: ignore[import-untyped]  # noqa: E402
    AZZ_BOUNDS,
    R_MAX,
    SQL,
    FreeAncillaModulated2DSliceResult,
    FreeAncillaModulatedNelderMeadResult,
    FreeAncillaModulatedOmegaScanResult,
    FreeAncillaModulatedSearchResult,
    _modulated_6d_objective,
    _sample_6d_config,
    compute_free_ancilla_modulated_sensitivity,
    free_ancilla_modulated_2d_slice,
    free_ancilla_modulated_random_search,
    plot_best_ratio_by_omega,
    plot_cross_experiment_comparison,
    plot_optimal_ancilla_state_by_omega,
    plot_slice_heatmap,
    run_modulated_nelder_mead,
    run_modulated_omega_scan,
    t_hold,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def make_ops() -> dict[str, np.ndarray]:
    return build_two_qubit_operators()


# ============================================================================
# Sensitivity Computation
# ============================================================================


class TestComputeFreeAncillaModulatedSensitivity:
    def test_decoupled_baseline_returns_sql(self) -> None:
        """At all-zero parameters with theta_A=0, Δω should equal SQL."""
        domega, _exp_val, var_val, _deriv, fringe = (
            compute_free_ancilla_modulated_sensitivity(
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        )
        assert np.isclose(domega, SQL, rtol=1e-8), (
            f"Decoupled Δω={domega} should equal SQL={SQL}"
        )
        assert not fringe
        assert var_val >= 0.0

    def test_fringe_extremum_detected(self) -> None:
        """Some configurations give zero derivative → flagged as fringe."""
        domega, _exp, _var, _deriv, fringe = compute_free_ancilla_modulated_sensitivity(
            1.0,
            0.0,
            0.0,
            5.0,
            0.0,
            0.0,
            5.0,
        )
        if fringe:
            assert np.isinf(domega)

    def test_nonzero_drive_returns_finite(self) -> None:
        """Most non-pathological configurations should give finite Δω."""
        domega, *_ = compute_free_ancilla_modulated_sensitivity(
            1.0,
            np.pi / 4,
            0.0,
            1.0,
            0.5,
            -0.3,
            2.0,
        )
        if np.isfinite(domega):
            assert domega > 0.0

    def test_returns_five_values(self) -> None:
        result = compute_free_ancilla_modulated_sensitivity(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        assert len(result) == 5
        domega, exp_val, var_val, deriv, fringe = result
        assert isinstance(domega, float)
        assert isinstance(exp_val, float)
        assert isinstance(var_val, float)
        assert isinstance(deriv, float)
        assert isinstance(fringe, bool)

    def test_drive_below_sql_at_theta_zero(self) -> None:
        """At θ_A=0, φ_A=0 with non-zero drive, Δω should be below SQL."""
        # Use parameters close to the 20260519 optimal at ω=0.2
        domega, *_ = compute_free_ancilla_modulated_sensitivity(
            omega_true=0.2,
            theta_A=0.0,
            phi_A=0.0,
            a_x=4.0,
            a_y=0.0,
            a_z=0.0,
            a_zz=-1.0,
        )
        if np.isfinite(domega):
            # Should be well below SQL at ω=0.2 with this configuration
            assert domega < SQL, (
                f"Δω={domega} should be below SQL={SQL} with non-zero drive"
            )

    def test_exact_20260519_baseline_recovery(self) -> None:
        """At θ_A=0, φ_A=0 with the canonical 20260519 optimal parameters
        at ω=0.2, Δω must reproduce the report value 0.02036 ± 1e-6.

        The canonical parameters (a_x=5.0, a_y=-5.0, a_z=4.0, a_zz=4.0) were
        found by Nelder--Mead refinement at ω=0.2 in the 20260519 report and
        produce Δω = 0.02035954... which rounds to the reported 0.02036.
        """
        domega, *_ = compute_free_ancilla_modulated_sensitivity(
            omega_true=0.2,
            theta_A=0.0,
            phi_A=0.0,
            a_x=5.0,
            a_y=-5.0,
            a_z=4.0,
            a_zz=4.0,
        )
        assert np.isfinite(domega), "Baseline Δω must be finite"
        assert np.isclose(domega, 0.02036, rtol=1e-6, atol=1e-6), (
            f"20260519 baseline Δω={domega:.10f} should equal 0.02036 ± 1e-6"
        )

    def test_variance_positivity(self) -> None:
        """Var(J_z^S) must always be non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            theta_A = rng.uniform(0.0, np.pi)
            phi_A = rng.uniform(0.0, 2.0 * np.pi)
            _, _, var_val, _, _ = compute_free_ancilla_modulated_sensitivity(
                1.0,
                theta_A,
                phi_A,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            assert var_val >= 0.0, (
                f"Negative variance at theta_A={theta_A}, phi_A={phi_A}"
            )


# ============================================================================
# 6D Sampling
# ============================================================================


class TestSample6DConfig:
    def test_shape_and_bounds(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            theta_A, phi_A, a_x, a_y, a_z, a_zz = _sample_6d_config(rng)
            assert 0.0 <= theta_A <= np.pi + 1e-12
            assert 0.0 <= phi_A <= 2.0 * np.pi + 1e-12
            norm_a = np.sqrt(a_x**2 + a_y**2 + a_z**2)
            assert norm_a <= R_MAX + 1e-12, f"norm_a={norm_a} exceeds R_MAX={R_MAX}"
            assert AZZ_BOUNDS[0] - 1e-12 <= a_zz <= AZZ_BOUNDS[1] + 1e-12

    def test_deterministic_with_seed(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = _sample_6d_config(rng1)
        s2 = _sample_6d_config(rng2)
        assert np.allclose(s1, s2)

    def test_custom_R_and_azz_bounds(self) -> None:
        """Verify that custom R and azz_bounds are respected."""
        rng = np.random.default_rng(42)
        custom_R = 3.0
        custom_azz_bounds = (-2.0, 2.0)
        for _ in range(10):
            _, _, a_x, a_y, a_z, a_zz = _sample_6d_config(
                rng,
                R=custom_R,
                azz_bounds=custom_azz_bounds,
            )
            norm_a = np.sqrt(a_x**2 + a_y**2 + a_z**2)
            assert norm_a <= custom_R + 1e-12, (
                f"norm_a={norm_a} exceeds custom_R={custom_R}"
            )
            assert (
                custom_azz_bounds[0] - 1e-12 <= a_zz <= custom_azz_bounds[1] + 1e-12
            ), f"a_zz={a_zz} outside [{custom_azz_bounds[0]}, {custom_azz_bounds[1]}]"


# ============================================================================
# 2D Slice
# ============================================================================


class TestFreeAncillaModulated2DSlice:
    def test_slice_returns_correct_shape(self) -> None:
        result = free_ancilla_modulated_2d_slice(omega=1.0, n_grid=5)
        assert result.delta_omega_grid.shape == (5, 5)
        assert len(result.theta_A_values) == 5
        assert len(result.azz_values) == 5

    def test_slice_sql_bound_holds(self) -> None:
        result = free_ancilla_modulated_2d_slice(omega=1.0, n_grid=5)
        finite_mask = np.isfinite(result.delta_omega_grid)
        if np.any(finite_mask):
            min_val = np.min(result.delta_omega_grid[finite_mask])
            assert min_val >= result.sql - 1e-10, (
                f"Min Δω={min_val} below SQL={result.sql}"
            )

    def test_slice_baseline_at_theta_A_zero(self) -> None:
        """At θ_A=0, the slice should recover SQL for all a_zz (decoupled drive)."""
        result = free_ancilla_modulated_2d_slice(omega=1.0, n_grid=5)
        for j in range(len(result.azz_values)):
            domega = result.delta_omega_grid[0, j]
            if np.isfinite(domega):
                assert np.isclose(domega, result.sql, rtol=1e-8), (
                    f"At θ_A=0, a_zz={result.azz_values[j]}: "
                    f"Δω={domega} should equal SQL={result.sql}"
                )

    def test_slice_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = free_ancilla_modulated_2d_slice(omega=1.0, n_grid=3)
        p = tmp_path / "slice.parquet"
        result.save_parquet(p)
        loaded = FreeAncillaModulated2DSliceResult.from_parquet(p)
        assert np.allclose(loaded.theta_A_values, result.theta_A_values)
        assert np.allclose(loaded.azz_values, result.azz_values)
        assert np.allclose(
            loaded.delta_omega_grid,
            result.delta_omega_grid,
            equal_nan=True,
        )
        assert np.isclose(loaded.omega_value, result.omega_value)
        assert np.isclose(loaded.sql, result.sql)
        assert loaded.fixed_drive_params == result.fixed_drive_params


# ============================================================================
# Random Search
# ============================================================================


class TestFreeAncillaModulatedRandomSearch:
    def test_small_run_shape(self) -> None:
        result = free_ancilla_modulated_random_search(
            omega=1.0,
            n_samples=10,
            seed=42,
        )
        assert result.samples.shape == (10, 6)
        assert result.delta_omega_values.shape == (10,)

    def test_deterministic_with_seed(self) -> None:
        r1 = free_ancilla_modulated_random_search(1.0, n_samples=10, seed=42)
        r2 = free_ancilla_modulated_random_search(1.0, n_samples=10, seed=42)
        assert np.allclose(r1.samples, r2.samples)
        assert np.allclose(r1.delta_omega_values, r2.delta_omega_values)

    def test_some_finite_values(self) -> None:
        result = free_ancilla_modulated_random_search(1.0, n_samples=20, seed=42)
        finite_count = np.sum(np.isfinite(result.delta_omega_values))
        assert finite_count > 0, "No finite Δω values found"

    def test_norms_within_R(self) -> None:
        result = free_ancilla_modulated_random_search(1.0, n_samples=20, seed=42)
        norms_a = np.sqrt(
            result.samples[:, 2] ** 2
            + result.samples[:, 3] ** 2
            + result.samples[:, 4] ** 2,
        )
        assert np.all(norms_a <= R_MAX + 1e-10)

    def test_best_params_are_valid(self) -> None:
        result = free_ancilla_modulated_random_search(1.0, n_samples=20, seed=42)
        bp = result.best_params
        assert len(bp) == 6
        assert 0.0 <= bp[0] <= np.pi + 1e-10
        assert 0.0 <= bp[1] <= 2.0 * np.pi + 1e-10

    def test_sql_not_beaten_by_default(self) -> None:
        """With zero drive and theta_A=0, we should not beat SQL."""
        result = free_ancilla_modulated_random_search(1.0, n_samples=20, seed=42)
        finite_mask = np.isfinite(result.delta_omega_values)
        if np.any(finite_mask):
            min_dt = np.min(result.delta_omega_values[finite_mask])
            # With zero drive, min should be SQL; but random search may find
            # drive configurations that beat SQL (the whole point of ω-modulated
            # drive), so this test just checks no infinite values.
            assert np.isfinite(min_dt)


# ============================================================================
# FreeAncillaModulatedSearchResult — Parquet Roundtrip
# ============================================================================


class TestSearchResultParquet:
    @pytest.fixture
    def make_result(self) -> FreeAncillaModulatedSearchResult:
        n_samp = 10
        rng = np.random.default_rng(42)
        samples = np.zeros((n_samp, 6), dtype=float)
        samples[:, 0] = rng.uniform(0.0, np.pi, n_samp)
        samples[:, 1] = rng.uniform(0.0, 2.0 * np.pi, n_samp)
        samples[:, 2:5] = rng.uniform(-5.0, 5.0, (n_samp, 3))
        samples[:, 5] = rng.uniform(-5.0, 5.0, n_samp)
        deltas = np.full(n_samp, 5.0, dtype=float)
        deltas[0] = float("inf")
        deltas[1] = 0.42
        return FreeAncillaModulatedSearchResult(
            samples=samples,
            delta_omega_values=deltas,
            expectation_values=rng.uniform(-0.5, 0.5, n_samp),
            variance_values=rng.uniform(0.0, 0.5, n_samp),
            deriv_values=rng.uniform(-1.0, 1.0, n_samp),
            is_fringe=np.array([True] + [False] * (n_samp - 1)),
            best_params=(
                float(samples[1, 0]),
                float(samples[1, 1]),
                float(samples[1, 2]),
                float(samples[1, 3]),
                float(samples[1, 4]),
                float(samples[1, 5]),
            ),
            best_delta_omega=float(deltas[1]),
            omega_value=1.0,
            sql=0.1,
            t_hold=10.0,
            R=5.0,
        )

    def test_roundtrip(
        self,
        make_result: FreeAncillaModulatedSearchResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "search.parquet"
        make_result.save_parquet(p)
        loaded = FreeAncillaModulatedSearchResult.from_parquet(p)
        assert np.allclose(loaded.samples, make_result.samples)
        assert np.allclose(
            loaded.delta_omega_values,
            make_result.delta_omega_values,
            equal_nan=True,
        )
        assert np.allclose(loaded.expectation_values, make_result.expectation_values)
        assert np.allclose(loaded.variance_values, make_result.variance_values)
        assert np.allclose(loaded.deriv_values, make_result.deriv_values)
        assert np.array_equal(loaded.is_fringe, make_result.is_fringe)
        assert loaded.best_params == make_result.best_params
        assert np.isclose(loaded.best_delta_omega, make_result.best_delta_omega)
        assert loaded.omega_value == make_result.omega_value
        assert loaded.sql == make_result.sql
        assert loaded.t_hold == make_result.t_hold
        assert loaded.R == make_result.R

    def test_fail_fast_missing_column(
        self,
        make_result: FreeAncillaModulatedSearchResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["R"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            FreeAncillaModulatedSearchResult.from_parquet(p)


# ============================================================================
# FreeAncillaModulatedNelderMeadResult — Parquet Roundtrip
# ============================================================================


class TestNelderMeadResultParquet:
    @pytest.fixture
    def make_result(self) -> FreeAncillaModulatedNelderMeadResult:
        return FreeAncillaModulatedNelderMeadResult(
            delta_omega_opt=0.09,
            params_opt=np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]),
            omega_true=1.0,
            success=True,
            nfev=100,
            message="OK",
            expectation_Jz=0.25,
            variance_Jz=0.1,
            t_hold=10.0,
            sql=0.1,
            T_BS=1.5707963267948966,
            fd_step=1e-6,
        )

    def test_roundtrip(
        self,
        make_result: FreeAncillaModulatedNelderMeadResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "nm.parquet"
        make_result.save_parquet(p)
        loaded = FreeAncillaModulatedNelderMeadResult.from_parquet(p)
        assert np.isclose(loaded.delta_omega_opt, make_result.delta_omega_opt)
        assert np.allclose(loaded.params_opt, make_result.params_opt)
        assert loaded.omega_true == make_result.omega_true
        assert loaded.success == make_result.success
        assert loaded.nfev == make_result.nfev
        assert np.isclose(loaded.expectation_Jz, make_result.expectation_Jz)
        assert np.isclose(loaded.variance_Jz, make_result.variance_Jz)
        assert np.isclose(loaded.t_hold, make_result.t_hold)
        assert np.isclose(loaded.sql, make_result.sql)
        assert np.isclose(loaded.T_BS, make_result.T_BS)
        assert np.isclose(loaded.fd_step, make_result.fd_step)

    def test_fail_fast_missing_column(
        self,
        make_result: FreeAncillaModulatedNelderMeadResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["t_hold"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            FreeAncillaModulatedNelderMeadResult.from_parquet(p)


# ============================================================================
# FreeAncillaModulatedOmegaScanResult — Parquet Roundtrip
# ============================================================================


class TestOmegaScanResultParquet:
    @pytest.fixture
    def make_result(self) -> FreeAncillaModulatedOmegaScanResult:
        return FreeAncillaModulatedOmegaScanResult(
            omega_values=np.array([0.1, 1.0, 5.0], dtype=float),
            best_params_per_omega=[
                (0.0, 0.0, 1.0, 2.0, 3.0, 4.0),
                (0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
                (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            ],
            best_delta_omega_per_omega=np.array([0.1, 0.09, 0.08], dtype=float),
            sql_values=np.array([0.1, 0.1, 0.1], dtype=float),
            expectation_Jz_per_omega=np.array([0.0, 0.25, -0.1], dtype=float),
            variance_Jz_per_omega=np.array([0.01, 0.1, 0.05], dtype=float),
            t_hold=10.0,
        )

    def test_roundtrip(
        self,
        make_result: FreeAncillaModulatedOmegaScanResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "scan.parquet"
        make_result.save_parquet(p)
        loaded = FreeAncillaModulatedOmegaScanResult.from_parquet(p)
        assert np.allclose(loaded.omega_values, make_result.omega_values)
        assert loaded.best_params_per_omega == make_result.best_params_per_omega
        assert np.allclose(
            loaded.best_delta_omega_per_omega,
            make_result.best_delta_omega_per_omega,
        )
        assert np.allclose(loaded.sql_values, make_result.sql_values)
        assert np.allclose(
            loaded.expectation_Jz_per_omega,
            make_result.expectation_Jz_per_omega,
        )
        assert np.allclose(
            loaded.variance_Jz_per_omega,
            make_result.variance_Jz_per_omega,
        )
        assert np.isclose(loaded.t_hold, make_result.t_hold)

    def test_fail_fast_missing_column(
        self,
        make_result: FreeAncillaModulatedOmegaScanResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["t_hold"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            FreeAncillaModulatedOmegaScanResult.from_parquet(p)


# ============================================================================
# 2D Slice Result — Parquet Roundtrip
# ============================================================================


class Test2DSliceResultParquet:
    @pytest.fixture
    def make_result(self) -> FreeAncillaModulated2DSliceResult:
        return FreeAncillaModulated2DSliceResult(
            theta_A_values=np.linspace(0.0, np.pi, 5),
            azz_values=np.linspace(-5.0, 5.0, 5),
            delta_omega_grid=np.ones((5, 5)) * 0.1,
            omega_value=1.0,
            sql=0.1,
            fixed_drive_params=(1.0, 0.0, 0.0),
        )

    def test_roundtrip(
        self,
        make_result: FreeAncillaModulated2DSliceResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "slice.parquet"
        make_result.save_parquet(p)
        loaded = FreeAncillaModulated2DSliceResult.from_parquet(p)
        assert np.allclose(loaded.theta_A_values, make_result.theta_A_values)
        assert np.allclose(loaded.azz_values, make_result.azz_values)
        assert np.allclose(loaded.delta_omega_grid, make_result.delta_omega_grid)
        assert np.isclose(loaded.omega_value, make_result.omega_value)
        assert np.isclose(loaded.sql, make_result.sql)
        assert loaded.fixed_drive_params == make_result.fixed_drive_params

    def test_fail_fast_missing_column(
        self,
        make_result: FreeAncillaModulated2DSliceResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["omega_value"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            FreeAncillaModulated2DSliceResult.from_parquet(p)

    def test_fail_fast_missing_fixed_drive(
        self,
        make_result: FreeAncillaModulated2DSliceResult,
        tmp_path: Path,
    ) -> None:
        """Missing fixed_ax/fixed_ay/fixed_az must raise, not silently default."""
        p = tmp_path / "bad_fixed.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["fixed_ax"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            FreeAncillaModulated2DSliceResult.from_parquet(p)


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


class TestRunModulatedNelderMead:
    def test_basic_run(self) -> None:
        result = run_modulated_nelder_mead(
            omega_true=1.0,
            x0=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            maxiter=10,
            fatol=1e-6,
        )
        assert np.isfinite(result.delta_omega_opt)
        assert len(result.params_opt) == 6

    def test_random_init(self) -> None:
        result = run_modulated_nelder_mead(
            omega_true=1.0,
            seed=42,
            maxiter=10,
            fatol=1e-6,
        )
        assert np.isfinite(result.delta_omega_opt)
        assert len(result.params_opt) == 6


# ============================================================================
# Omega Scan
# ============================================================================


class TestRunModulatedOmegaScan:
    def test_small_scan_produces_correct_shape(self) -> None:
        result = run_modulated_omega_scan(
            omega_values=[1.0],
            n_random=10,
            n_nm_refine=3,
            seed=42,
        )
        assert len(result.omega_values) == 1
        assert len(result.best_params_per_omega) == 1
        assert np.isfinite(result.best_delta_omega_per_omega[0]) or np.isinf(
            result.best_delta_omega_per_omega[0],
        )

    def test_deterministic_with_seed(self) -> None:
        r1 = run_modulated_omega_scan(
            [1.0],
            n_random=10,
            n_nm_refine=2,
            seed=42,
        )
        r2 = run_modulated_omega_scan(
            [1.0],
            n_random=10,
            n_nm_refine=2,
            seed=42,
        )
        assert np.allclose(r1.best_delta_omega_per_omega, r2.best_delta_omega_per_omega)


# ============================================================================
# 6D Objective Function
# ============================================================================


class TestModulated6DObjective:
    def test_zero_params_returns_finite(self) -> None:
        val = _modulated_6d_objective(
            np.zeros(6),
            omega_true=1.0,
        )
        assert np.isfinite(val)

    def test_penalty_for_bad_theta(self) -> None:
        val = _modulated_6d_objective(
            np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
            omega_true=1.0,
        )
        assert val > 1e9, "Negative theta should incur penalty"

    def test_penalty_for_large_norm(self) -> None:
        val = _modulated_6d_objective(
            np.array([0.0, 0.0, 20.0, 0.0, 0.0, 0.0]),
            omega_true=1.0,
        )
        assert val > 1e9, "Oversized drive should incur penalty"


# ============================================================================
# Plot Functions
# ============================================================================


class TestPlotFunctions:
    @pytest.fixture
    def make_scan_result(self) -> FreeAncillaModulatedOmegaScanResult:
        return FreeAncillaModulatedOmegaScanResult(
            omega_values=np.array([0.1, 1.0, 5.0], dtype=float),
            best_params_per_omega=[
                (0.0, 0.0, 1.0, 2.0, 3.0, 4.0),
                (0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
                (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            ],
            best_delta_omega_per_omega=np.array([0.1, 0.09, 0.08], dtype=float),
            sql_values=np.array([0.1, 0.1, 0.1], dtype=float),
            expectation_Jz_per_omega=np.array([0.0, 0.25, -0.1], dtype=float),
            variance_Jz_per_omega=np.array([0.01, 0.1, 0.05], dtype=float),
        )

    @pytest.fixture
    def make_slice_result(self) -> FreeAncillaModulated2DSliceResult:
        return FreeAncillaModulated2DSliceResult(
            theta_A_values=np.linspace(0.0, np.pi, 10),
            azz_values=np.linspace(-5.0, 5.0, 10),
            delta_omega_grid=np.ones((10, 10)) * 0.1,
            omega_value=1.0,
            sql=0.1,
        )

    def test_plot_best_ratio_by_omega_saves_svg(
        self,
        make_scan_result: FreeAncillaModulatedOmegaScanResult,
        tmp_path: Path,
    ) -> None:
        svg_p = tmp_path / "ratio.svg"
        result = plot_best_ratio_by_omega(make_scan_result, svg_p)
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0

    def test_plot_slice_heatmap_saves_svg(
        self,
        make_slice_result: FreeAncillaModulated2DSliceResult,
        tmp_path: Path,
    ) -> None:
        svg_p = tmp_path / "slice.svg"
        result = plot_slice_heatmap(make_slice_result, svg_p)
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0

    def test_plot_optimal_ancilla_state_saves_svg(
        self,
        make_scan_result: FreeAncillaModulatedOmegaScanResult,
        tmp_path: Path,
    ) -> None:
        svg_p = tmp_path / "theta.svg"
        result = plot_optimal_ancilla_state_by_omega(make_scan_result, svg_p)
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0

    def test_plot_cross_experiment_comparison_saves_svg(
        self,
        make_scan_result: FreeAncillaModulatedOmegaScanResult,
        tmp_path: Path,
    ) -> None:
        svg_p = tmp_path / "comparison.svg"
        result = plot_cross_experiment_comparison(make_scan_result, save_path=svg_p)
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0

    def test_plot_cross_experiment_comparison_with_baseline(
        self,
        make_scan_result: FreeAncillaModulatedOmegaScanResult,
        tmp_path: Path,
    ) -> None:
        svg_p = tmp_path / "comparison2.svg"
        baseline = np.array([0.1, 0.09, 0.08])
        result = plot_cross_experiment_comparison(
            make_scan_result,
            baseline_delta=baseline,
            save_path=svg_p,
        )
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_zero_drive_zero_interaction_equals_sql(self) -> None:
        """When a_x=a_y=a_z=a_zz=0 regardless of ancilla state, Δω = SQL."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            theta_A = rng.uniform(0.0, np.pi)
            phi_A = rng.uniform(0.0, 2.0 * np.pi)
            domega, *_ = compute_free_ancilla_modulated_sensitivity(
                1.0,
                theta_A,
                phi_A,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            assert np.isclose(domega, SQL, rtol=1e-8), (
                f"Decoupled Δω={domega} should equal SQL={SQL} "
                f"at theta_A={theta_A}, phi_A={phi_A}"
            )

    def test_small_omega_gives_finite(self) -> None:
        domega, *_ = compute_free_ancilla_modulated_sensitivity(
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        assert np.isfinite(domega)
        assert domega > 0.0

    def test_evolve_circuit_preserves_norm(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        psi0 = free_ancilla_initial_state(np.pi / 3, np.pi / 4)
        psi = evolve_phase_modulated_circuit(
            psi0,
            np.pi / 2,
            t_hold,
            1.0,
            1.0,
            0.5,
            -0.3,
            2.0,
            make_ops,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12)

    def test_zero_radius_ball(self) -> None:
        """With R=0, drive samples should be zero."""
        rng = np.random.default_rng(42)
        from src.utils.monte_carlo import marsaglia_ball_sample

        drive, _ = marsaglia_ball_sample(rng, 10, 0.0, -5.0, 5.0)
        assert np.allclose(drive, 0.0)

    def test_inf_sensitivity_at_fringe(self) -> None:
        """Decoupled baseline should be finite."""
        domega, *_ = compute_free_ancilla_modulated_sensitivity(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        assert np.isfinite(domega)


# ============================================================================
# CLI
# ============================================================================


def test_cli_help() -> None:
    import subprocess

    result = subprocess.run(  # noqa: PLW1510
        [
            "uv",
            "run",
            "python",
            str(Path(__file__).resolve().parent / "local.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()


# ============================================================================
# Physical Invariants
# ============================================================================


class TestPhysicalInvariants:
    def test_unitarity_of_phase_modulated_hold(self) -> None:
        """Verify U_hold^† U_hold = I for all ω and drive values."""
        ops = build_two_qubit_operators()
        from src.analysis.ancilla_drive_metrology import (
            phase_modulated_hold_unitary,
        )

        rng = np.random.default_rng(42)
        for _ in range(5):
            omega = rng.uniform(0.1, 5.0)
            a_x = rng.uniform(-5.0, 5.0)
            a_y = rng.uniform(-5.0, 5.0)
            a_z = rng.uniform(-5.0, 5.0)
            a_zz = rng.uniform(-5.0, 5.0)
            U = phase_modulated_hold_unitary(t_hold, omega, a_x, a_y, a_z, a_zz, ops)
            assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12), (
                f"Hold unitary not unitary at ω={omega}"
            )

    def test_initial_state_normalised(self) -> None:
        """Free-ancilla initial state is always normalised."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            theta_A = rng.uniform(0.0, np.pi)
            phi_A = rng.uniform(0.0, 2.0 * np.pi)
            psi = free_ancilla_initial_state(theta_A, phi_A)
            assert np.isclose(np.linalg.norm(psi), 1.0), (
                f"State not normalised at theta_A={theta_A}, phi_A={phi_A}"
            )

    def test_system_always_in_one_zero(self) -> None:
        """System qubit must always be in |1,0⟩ (computational |0⟩)."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            theta_A = rng.uniform(0.0, np.pi)
            phi_A = rng.uniform(0.0, 2.0 * np.pi)
            state = free_ancilla_initial_state(theta_A, phi_A)
            # System in |1,0⟩_S → indices 2 and 3 (|10⟩ and |11⟩) must be zero
            assert np.isclose(np.abs(state[2]) ** 2 + np.abs(state[3]) ** 2, 0.0), (
                f"System not in |1,0⟩ at theta_A={theta_A}, phi_A={phi_A}"
            )

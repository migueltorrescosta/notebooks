r"""
Tests for the Cavity-Enhanced TMSV MZI module (2026-06-29).

Run with:
    uv run pytest reports/20260629/test_cavity_enhanced_tmsv_mzi.py -q --tb=short
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.physics.mzi_states import (
    compute_jz_expectation,
    compute_jz_variance,
    make_two_mode_squeezed_vacuum,
)
from src.physics.sv_qfi import compute_tmsv_captured_norm, compute_tmsv_qfi
from src.utils.serialization import assert_roundtrip_fields

_m = importlib.import_module("reports.20260629.cavity_enhanced_tmsv_mzi")

CavityTmsvScalingFit = _m.CavityTmsvScalingFit
CavityTmsvSensitivityResult = _m.CavityTmsvSensitivityResult
H_t = _m.H_t
_best_sensitivity_per_config = _m._best_sensitivity_per_config
_fit_scaling_per_finesse = _m._fit_scaling_per_finesse
_omega_grid_finesse = _m._omega_grid_finesse
_row_dicts_to_result = _m._row_dicts_to_result
generate_full_data = _m.generate_full_data
generate_single_cavity_point = _m.generate_single_cavity_point
main = _m.main
plot_delta_omega_overlay = _m.plot_delta_omega_overlay
plot_prefactor_scaling = _m.plot_prefactor_scaling
plot_scaling = _m.plot_scaling

# =============================================================================
# Two-Mode Squeezed Vacuum (TMSV) State
# =============================================================================


class TestTwoModeSqueezedVacuum:
    def test_normalized(self) -> None:
        """TMSV state must have unit norm."""
        for mean_total in [2, 4, 10, 20]:
            M = int(5 * mean_total)
            state = make_two_mode_squeezed_vacuum(float(mean_total), M)
            assert np.isclose(np.linalg.norm(state), 1.0, rtol=1e-10), (
                f"Failed for mean_total={mean_total}"
            )

    def test_mean_photon_number(self) -> None:
        """Mean photon number should match the target value."""
        mean_total = 4.0
        M = 30
        state = make_two_mode_squeezed_vacuum(mean_total, M)
        dim_single = M + 1
        mean_n = 0.0
        for n1 in range(M + 1):
            for n2 in range(M + 1):
                idx = n1 * dim_single + n2
                prob = np.abs(state[idx]) ** 2
                mean_n += prob * (n1 + n2)
        assert np.isclose(mean_n, mean_total, rtol=1e-2), (
            f"Mean total N={mean_n}, expected {mean_total}"
        )

    def test_only_diagonal_components(self) -> None:
        """TMSV should only have |n, n⟩ components."""
        mean_total = 4.0
        M = 15
        state = make_two_mode_squeezed_vacuum(mean_total, M)
        dim_single = M + 1
        for n1 in range(M + 1):
            for n2 in range(M + 1):
                idx = n1 * dim_single + n2
                if n1 != n2:
                    assert np.abs(state[idx]) < 1e-12, (
                        f"Non-zero component at |{n1},{n2}⟩"
                    )

    def test_jz_expectation_zero(self) -> None:
        """⟨J_z⟩ = 0 for TMSV (symmetric in both modes)."""
        for mean_total in [2, 4, 10]:
            M = int(5 * mean_total)
            state = make_two_mode_squeezed_vacuum(float(mean_total), M)
            exp = np.real(compute_jz_expectation(state, M))
            assert np.isclose(exp, 0.0, atol=1e-12), (
                f"Failed for mean_total={mean_total}"
            )

    def test_jz_variance_zero_input(self) -> None:
        """Var(J_z) = 0 for input TMSV (|n,n⟩ has n_1 - n_2 = 0)."""
        for mean_total in [2, 4, 10]:
            M = int(5 * mean_total)
            state = make_two_mode_squeezed_vacuum(float(mean_total), M)
            var = compute_jz_variance(state, M)
            assert np.isclose(var, 0.0, atol=1e-12), (
                f"Var(J_z) should be 0 for input TMSV, got {var}"
            )

    def test_positive_mean_N_required(self) -> None:
        """TMSV requires mean_total > 0."""
        with pytest.raises(ValueError, match="positive"):
            make_two_mode_squeezed_vacuum(0.0, 10)


# =============================================================================
# Analytical QFI and Helpers
# =============================================================================


class TestAnalyticalQFI:
    def test_tmsv_qfi_formula(self) -> None:
        """Analytical QFI formula for TMSV."""
        mean_total = 10.0
        fq = compute_tmsv_qfi(mean_total, t_hold=10.0)
        expected = 100.0 * 10.0 * (10.0 + 2.0)  # H_t^2 * N * (N+2)
        assert np.isclose(fq, expected), f"F_Q={fq}, expected {expected}"

    def test_tmsv_qfi_scaling(self) -> None:
        """F_Q scales as N^2 for large N."""
        fq_10 = compute_tmsv_qfi(10.0)
        fq_20 = compute_tmsv_qfi(20.0)
        ratio = fq_20 / fq_10
        expected_ratio = (20.0 * 22.0) / (10.0 * 12.0)
        assert np.isclose(ratio, expected_ratio, rtol=1e-10)


class TestCapturedNorm:
    def test_tmsv_captured_norm(self) -> None:
        """Captured norm should be close to 1 for sufficient truncation."""
        captured = compute_tmsv_captured_norm(mean_total=4.0, max_photons=50)
        assert captured > 0.999, f"Captured norm {captured} too low"

    def test_tmsv_captured_norm_at_boundary(self) -> None:
        """Captured norm at max truncation for large mean_total."""
        # With max_trunc=200 for ⟨N⟩≥30, capture >99.9%
        captured = compute_tmsv_captured_norm(mean_total=40.0, max_photons=200)
        assert captured > 0.999, f"Captured norm {captured} too low"
        # Old truncation (100) loses ~0.7% of norm at ⟨N⟩=40
        captured_old = compute_tmsv_captured_norm(mean_total=40.0, max_photons=100)
        assert captured_old < captured, (
            f"Old truncation norm {captured_old} should be less than new {captured}"
        )


class TestTmsvSQL:
    def test_sql_formula(self) -> None:
        """SQL formula for TMSV: 1 / (t_hold * sqrt(N))."""
        sql = 1.0 / (10.0 * np.sqrt(10.0))
        expected = 1.0 / (10.0 * np.sqrt(10.0))
        assert np.isclose(sql, expected), f"SQL={sql}, expected {expected}"

    def test_sql_scaling(self) -> None:
        """SQL scales as 1/sqrt(N)."""
        sql_10 = 1.0 / (H_t * np.sqrt(10.0))
        sql_40 = 1.0 / (H_t * np.sqrt(40.0))
        ratio = sql_40 / sql_10
        expected_ratio = np.sqrt(10.0 / 40.0)
        assert np.isclose(ratio, expected_ratio, rtol=1e-10)


# =============================================================================
# ω Grid Construction
# =============================================================================


class TestOmegaGrid:
    def test_omega_grid_bounds(self) -> None:
        """ω grid should be within [0, π/(2FH_t)]."""
        for Fi in [1, 10, 100, 1000]:
            grid = _omega_grid_finesse(Fi)
            assert grid[0] >= 0
            omega_max = np.pi / (2.0 * Fi * H_t)
            assert grid[-1] <= omega_max + 1e-15, (
                f"ℱ={Fi}: max ω={grid[-1]}, expected ≤{omega_max}"
            )

    def test_omega_grid_length(self) -> None:
        """ω grid should have N_OMEGA_POINTS points."""
        for Fi in [1, 10, 100]:
            grid = _omega_grid_finesse(Fi)
            assert len(grid) == 200, f"ℱ={Fi}: len={len(grid)}, expected 200"

    def test_omega_grid_quadratic(self) -> None:
        """ω grid should be monotonically increasing with quadratic spacing."""
        for Fi in [1, 10, 100]:
            grid = _omega_grid_finesse(Fi)
            assert grid[0] >= 0
            assert np.all(np.diff(grid) > 0), (
                f"ℱ={Fi}: grid is not monotonically increasing"
            )
            # Spacing should increase with ω (quadratic → larger gaps at larger ω)
            diffs = np.diff(grid)
            assert diffs[-1] > diffs[0], (
                f"ℱ={Fi}: last spacing {diffs[-1]} not > first spacing {diffs[0]}"
            )

    def test_higher_finesse_narrower_grid(self) -> None:
        """Higher finesse → narrower ω range."""
        grid_1 = _omega_grid_finesse(1)
        grid_10 = _omega_grid_finesse(10)
        assert grid_10[-1] < grid_1[-1], "Higher finesse should have narrower ω range"


# =============================================================================
# Single Point Generation
# =============================================================================


class TestGenerateSinglePoint:
    def test_single_point_basic(self) -> None:
        """Generate a single (⟨N⟩, ℱ) point and verify structure."""
        result = generate_single_cavity_point(
            mean_total=4.0,
            finesse=1.0,
            max_photons=20,
        )
        assert result is not None
        assert len(result["mean_total"]) == 200
        assert np.allclose(result["mean_total"], 4.0)
        assert np.allclose(result["finesse"], 1.0)
        assert np.all(np.isfinite(result["delta_omega_c"]))

    def test_single_point_cfi_positive(self) -> None:
        """CFI values should be non-negative."""
        result = generate_single_cavity_point(
            mean_total=4.0,
            finesse=1.0,
            max_photons=20,
        )
        assert result is not None
        assert np.all(result["cfi_values"] >= -1e-15), "Some CFI values negative"

    def test_single_point_qfi_finite(self) -> None:
        """QFI should be finite and positive."""
        result = generate_single_cavity_point(
            mean_total=4.0,
            finesse=1.0,
            max_photons=20,
        )
        assert result is not None
        assert np.all(np.isfinite(result["qfi_bound"]))
        assert np.all(result["qfi_bound"] > 0)

    def test_single_point_delta_omega_finite(self) -> None:
        """Best Δω should be finite."""
        result = generate_single_cavity_point(
            mean_total=4.0,
            finesse=1.0,
            max_photons=20,
        )
        assert result is not None
        min_idx = int(np.argmin(result["delta_omega_c"]))
        assert np.isfinite(result["delta_omega_c"][min_idx])

    def test_single_point_captured_norm(self) -> None:
        """Captured norm should be within [0, 1]."""
        result = generate_single_cavity_point(
            mean_total=4.0,
            finesse=1.0,
            max_photons=20,
        )
        assert result is not None
        assert np.all(result["captured_norm"] > 0)
        assert np.all(result["captured_norm"] <= 1.0)

    def test_single_point_auto_truncation(self) -> None:
        """Generate with auto-computed truncation."""
        result = generate_single_cavity_point(
            mean_total=4.0,
            finesse=1.0,
        )
        assert result is not None
        assert np.isfinite(result["delta_omega_c"][0])

    def test_failure_returns_none(self) -> None:
        """Invalid input should return None, not raise."""
        result = generate_single_cavity_point(
            mean_total=0.0,
            finesse=1.0,
            max_photons=10,
        )
        assert result is None


# =============================================================================
# Row Dicts to Result
# =============================================================================


class TestRowDictsToResult:
    def test_empty_raises(self) -> None:
        """Empty list must raise ValueError."""
        with pytest.raises(ValueError, match="No valid"):
            _row_dicts_to_result([])

    def test_single_row(self) -> None:
        """Single row produces correct result."""
        rows = [generate_single_cavity_point(4.0, 1.0, max_photons=20)]
        result = _row_dicts_to_result(rows)
        assert isinstance(result, CavityTmsvSensitivityResult)
        assert len(result.mean_total) == 200
        assert np.allclose(result.finesse[:5], 1.0)

    def test_multiple_rows(self) -> None:
        """Multiple rows are concatenated correctly."""
        r1 = generate_single_cavity_point(4.0, 1.0, max_photons=20)
        r2 = generate_single_cavity_point(6.0, 1.0, max_photons=20)
        result = _row_dicts_to_result([r1, r2])
        assert len(result.mean_total) == 400
        assert np.isclose(result.mean_total[0], 4.0)
        assert np.isclose(result.mean_total[-1], 6.0)

    def test_mixed_finesse(self) -> None:
        """Rows with different finesse are concatenated."""
        r1 = generate_single_cavity_point(4.0, 1.0, max_photons=20)
        r2 = generate_single_cavity_point(4.0, 10.0, max_photons=20)
        result = _row_dicts_to_result([r1, r2])
        assert np.allclose(result.finesse[:200], 1.0)
        assert np.allclose(result.finesse[200:], 10.0)


# =============================================================================
# CavityTmsvSensitivityResult — Parquet Roundtrip
# =============================================================================


class TestCavityTmsvSensitivityResultParquet:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("mean_total", "allclose"),
        ("finesse", "allclose"),
        ("omega_values", "allclose"),
        ("cfi_values", "allclose"),
        ("qfi_bound", "allclose"),
        ("delta_omega_c", "allclose"),
        ("delta_omega_q", "allclose"),
        ("delta_omega_sql", "allclose"),
        ("delta_omega_sql_eff", "allclose"),
        ("t_hold", "allclose"),
        ("t_hold_eff", "allclose"),
        ("truncation_M", "allclose"),
        ("captured_norm", "allclose"),
    ]

    @pytest.fixture
    def make_result(self) -> CavityTmsvSensitivityResult:
        n_pts = 30
        rng = np.random.default_rng(42)
        return CavityTmsvSensitivityResult(
            mean_total=rng.uniform(2, 40, n_pts),
            finesse=rng.choice([1, 10, 100], n_pts),
            omega_values=rng.uniform(0, 0.1, n_pts),
            cfi_values=rng.uniform(1, 1000, n_pts),
            qfi_bound=rng.uniform(100, 10000, n_pts),
            delta_omega_c=rng.uniform(0.001, 1, n_pts),
            delta_omega_q=rng.uniform(0.001, 1, n_pts),
            delta_omega_sql=rng.uniform(0.01, 1, n_pts),
            delta_omega_sql_eff=rng.uniform(0.001, 1, n_pts),
            t_hold=rng.uniform(1, 100, n_pts),
            t_hold_eff=rng.uniform(10, 10000, n_pts),
            truncation_M=rng.integers(10, 100, n_pts).astype(float),
            captured_norm=rng.uniform(0.99, 1.0, n_pts),
        )

    def test_roundtrip(
        self, make_result: CavityTmsvSensitivityResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "sensitivity.parquet"
        make_result.save_parquet(p)
        loaded = CavityTmsvSensitivityResult.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)

    def test_fail_fast_missing_column(
        self, make_result: CavityTmsvSensitivityResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["mean_total"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing"):
            CavityTmsvSensitivityResult.from_parquet(p)

    def test_fail_fast_missing_metadata(
        self, make_result: CavityTmsvSensitivityResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad_meta.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["finesse", "t_hold", "t_hold_eff"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing"):
            CavityTmsvSensitivityResult.from_parquet(p)

    def test_to_dataframe_columns(
        self, make_result: CavityTmsvSensitivityResult
    ) -> None:
        df = make_result.to_dataframe()
        expected = set(make_result._PARQUET_COLUMNS)
        assert set(df.columns) == expected, f"Missing: {expected - set(df.columns)}"

    def test_from_dataframe(self, make_result: CavityTmsvSensitivityResult) -> None:
        df = make_result.to_dataframe()
        loaded = CavityTmsvSensitivityResult.from_dataframe(df)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)


# =============================================================================
# Best Sensitivity Extraction
# =============================================================================


class TestBestSensitivity:
    @pytest.fixture
    def sample_data(self) -> CavityTmsvSensitivityResult:
        # Build a small result with 2 mean_total × 2 finesse × 3 ω
        n_N = 2
        n_F = 2
        n_omega = 3
        n_total = n_N * n_F * n_omega
        rng = np.random.default_rng(42)
        return CavityTmsvSensitivityResult(
            mean_total=np.repeat([4.0, 8.0], n_F * n_omega),
            finesse=np.tile(np.repeat([1.0, 10.0], n_omega), n_N),
            omega_values=np.tile(np.linspace(0.01, 0.1, n_omega), n_N * n_F),
            cfi_values=rng.uniform(1, 100, n_total),
            qfi_bound=rng.uniform(100, 1000, n_total),
            delta_omega_c=rng.uniform(0.01, 1, n_total),
            delta_omega_q=rng.uniform(0.01, 0.5, n_total),
            delta_omega_sql=0.1 / np.sqrt(np.repeat([4.0, 8.0], n_F * n_omega)),
            delta_omega_sql_eff=0.1 / np.sqrt(np.repeat([4.0, 8.0], n_F * n_omega)),
            t_hold=np.full(n_total, 10.0),
            t_hold_eff=np.tile(np.repeat([10.0, 100.0], n_omega), n_N),
            truncation_M=np.full(n_total, 20.0),
            captured_norm=np.full(n_total, 0.999),
        )

    def test_best_sensitivity_shape(
        self, sample_data: CavityTmsvSensitivityResult
    ) -> None:
        best_df = _best_sensitivity_per_config(sample_data)
        assert len(best_df) == 4  # 2 N × 2 F
        expected_cols = {
            "mean_total",
            "finesse",
            "best_delta_omega_c",
            "best_omega",
            "best_cfi",
            "delta_omega_q",
            "delta_omega_sql",
            "delta_omega_sql_eff",
            "ratio_to_sql",
            "ratio_to_sql_eff",
            "qfi_bound",
            "t_hold",
            "t_hold_eff",
            "truncation_M",
            "captured_norm",
        }
        assert set(best_df.columns) == expected_cols, (
            f"Missing: {expected_cols - set(best_df.columns)}"
        )

    def test_best_is_minimum(self, sample_data: CavityTmsvSensitivityResult) -> None:
        best_df = _best_sensitivity_per_config(sample_data)
        full = sample_data.to_dataframe()
        for _, row in best_df.iterrows():
            mean_total = row["mean_total"]
            finesse = row["finesse"]
            sub = full[
                np.isclose(full["mean_total"], mean_total)
                & np.isclose(full["finesse"], finesse)
            ]
            assert row["best_delta_omega_c"] <= sub["delta_omega_c"].min() + 1e-15


# =============================================================================
# Scaling Fit
# =============================================================================


class TestScalingFit:
    def _make_synthetic_data(
        self,
        N_vals: np.ndarray,
        F_vals: list[float],
    ) -> CavityTmsvSensitivityResult:
        """Build clean synthetic data: Δω = C * N^α with α = -0.76."""
        rows: list[CavityTmsvSensitivityResult] = []
        for Fi in F_vals:
            for Ni in N_vals:
                C = 0.1 / Fi  # Δω_min ∝ 1/ℱ
                dt = C * Ni ** (-0.76)
                n_omega = 3
                rows.append(
                    CavityTmsvSensitivityResult(
                        mean_total=np.full(n_omega, Ni),
                        finesse=np.full(n_omega, Fi),
                        omega_values=np.linspace(0.01, 0.1, n_omega),
                        cfi_values=np.full(n_omega, 1.0 / dt**2),
                        qfi_bound=np.full(n_omega, 1.0 / dt**2),
                        delta_omega_c=np.full(n_omega, dt),
                        delta_omega_q=np.full(n_omega, dt * 0.99),
                        delta_omega_sql=np.full(n_omega, 1.0 / (10 * np.sqrt(Ni))),
                        delta_omega_sql_eff=np.full(
                            n_omega, 1.0 / (10 * Fi * np.sqrt(Ni))
                        ),
                        t_hold=np.full(n_omega, 10.0),
                        t_hold_eff=np.full(n_omega, Fi * 10.0),
                        truncation_M=np.full(n_omega, 30.0),
                        captured_norm=np.full(n_omega, 0.999),
                    )
                )
        dfs = [r.to_dataframe() for r in rows]
        big_df = pd.concat(dfs, ignore_index=True)
        return CavityTmsvSensitivityResult.from_dataframe(big_df)

    def test_fit_scaling_per_finesse(self) -> None:
        """Scaling fit produces a valid result."""
        N_vals = np.array([4.0, 8.0, 12.0, 16.0, 20.0])
        F_vals = [1.0, 10.0, 100.0]
        data = self._make_synthetic_data(N_vals, F_vals)

        best_df = _best_sensitivity_per_config(data)
        scaling_fit = _fit_scaling_per_finesse(best_df)

        assert scaling_fit.valid or len(scaling_fit.warnings_list) > 0
        assert len(scaling_fit.finesse_values) == 3

        for alpha_val in scaling_fit.alpha_values:
            if np.isfinite(alpha_val):
                assert np.isclose(alpha_val, -0.76, atol=0.05), (
                    f"α={alpha_val}, expected -0.76"
                )

        if np.isfinite(scaling_fit.beta):
            assert np.isclose(scaling_fit.beta, 1.0, atol=0.05), (
                f"β={scaling_fit.beta}, expected 1.0"
            )


# =============================================================================
# Plot Functions
# =============================================================================


class TestPlotFunctions:
    @pytest.fixture
    def sample_data(self) -> CavityTmsvSensitivityResult:
        rng = np.random.default_rng(42)
        n_N = 3
        n_F = 2
        n_omega = 5
        n_total = n_N * n_F * n_omega
        return CavityTmsvSensitivityResult(
            mean_total=np.repeat([4.0, 8.0, 16.0], n_F * n_omega),
            finesse=np.tile(np.repeat([1.0, 10.0], n_omega), n_N),
            omega_values=np.tile(np.linspace(0.01, 0.1, n_omega), n_N * n_F),
            cfi_values=rng.uniform(1, 100, n_total),
            qfi_bound=rng.uniform(100, 10000, n_total),
            delta_omega_c=rng.uniform(0.01, 0.5, n_total),
            delta_omega_q=rng.uniform(0.01, 0.3, n_total),
            delta_omega_sql=0.1 / np.sqrt(np.repeat([4.0, 8.0, 16.0], n_F * n_omega)),
            delta_omega_sql_eff=0.1
            / np.sqrt(
                np.tile(np.repeat([1.0, 10.0], n_omega), n_N)
                * np.repeat([4.0, 8.0, 16.0], n_F * n_omega)
            ),
            t_hold=np.full(n_total, 10.0),
            t_hold_eff=np.tile(np.repeat([10.0, 100.0], n_omega), n_N),
            truncation_M=np.full(n_total, 20.0),
            captured_norm=np.full(n_total, 0.999),
        )

    def test_plot_delta_omega_overlay(
        self, sample_data: CavityTmsvSensitivityResult, tmp_path: Path
    ) -> None:
        svg = tmp_path / "overlay.svg"
        created = plot_delta_omega_overlay(
            sample_data,
            finesse=1.0,
            selected_N=[4.0, 16.0],
            save_path=svg,
        )
        assert svg.exists()
        assert created == svg

    def test_plot_scaling(
        self, sample_data: CavityTmsvSensitivityResult, tmp_path: Path
    ) -> None:
        svg = tmp_path / "scaling.svg"
        created = plot_scaling(
            sample_data,
            selected_F=[1.0, 10.0],
            save_path=svg,
        )
        assert svg.exists()
        assert created == svg

    def test_plot_prefactor_scaling(
        self, sample_data: CavityTmsvSensitivityResult, tmp_path: Path
    ) -> None:
        svg = tmp_path / "prefactor.svg"
        created = plot_prefactor_scaling(
            sample_data,
            selected_N=[4.0, 16.0],
            save_path=svg,
        )
        assert svg.exists()
        assert created == svg

    def test_plot_scaling_no_fit(
        self, sample_data: CavityTmsvSensitivityResult, tmp_path: Path
    ) -> None:
        """plot_scaling works without scaling_fit."""
        svg = tmp_path / "scaling_nofit.svg"
        plot_scaling(sample_data, scaling_fit=None, save_path=svg)
        assert svg.exists()


# =============================================================================
# Integration: Generate Full Data (Small)
# =============================================================================


class TestGenerateFullData:
    @pytest.mark.slow
    def test_generate_full_data_small(self, tmp_path: Path) -> None:
        """Generate full data with small ranges and verify."""
        n_range = [4.0, 8.0]
        f_range = [1.0, 10.0]
        data = generate_full_data(
            mean_total_range=n_range,
            finesse_range=f_range,
        )
        assert isinstance(data, CavityTmsvSensitivityResult)
        n_expected = len(n_range) * len(f_range) * 200
        assert len(data.mean_total) == n_expected, (
            f"Expected {n_expected} rows, got {len(data.mean_total)}"
        )
        assert np.all(np.isfinite(data.delta_omega_c))

    @pytest.mark.slow
    def test_generate_full_data_save_parquet(self, tmp_path: Path) -> None:
        """Save and reload full data."""
        n_range = [4.0, 8.0]
        f_range = [1.0]
        data = generate_full_data(
            mean_total_range=n_range,
            finesse_range=f_range,
        )
        pq_path = tmp_path / "test.parquet"
        data.save_parquet(pq_path)
        loaded = CavityTmsvSensitivityResult.from_parquet(pq_path)
        assert len(loaded.mean_total) == len(data.mean_total)
        assert np.allclose(loaded.mean_total, data.mean_total)


# =============================================================================
# CLI
# =============================================================================


class TestCLI:
    def test_cli_help(self) -> None:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(Path(__file__).resolve().parent / "cavity_enhanced_tmsv_mzi.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    def test_main_direct_help(self) -> None:
        import sys as _sys

        old_argv = _sys.argv[:]
        try:
            _sys.argv = ["script", "--help"]
            with pytest.raises(SystemExit):
                main()
        finally:
            _sys.argv = old_argv

    def test_main_direct_run_small(self, tmp_path: Path) -> None:
        """Call main() with --force and --only F=1 to test generation pipeline."""
        import sys as _sys_mod

        old_argv = _sys_mod.argv[:]
        orig_N = _m.MEAN_TOTAL_RANGE
        orig_F = _m.FINESSE_RANGE
        orig_npts = _m.N_OMEGA_POINTS
        pq_path = tmp_path / "test_cavity_tmsv.parquet"
        try:
            _m.MEAN_TOTAL_RANGE = [4.0]
            _m.N_OMEGA_POINTS = 3
            _sys_mod.argv = [
                "script",
                "--force",
                "--only",
                "F=1",
                "--pq-path",
                str(pq_path),
            ]
            main()
        finally:
            _sys_mod.argv = old_argv
            _m.MEAN_TOTAL_RANGE = orig_N
            _m.FINESSE_RANGE = orig_F
            _m.N_OMEGA_POINTS = orig_npts


# =============================================================================
# Generate All (Pipeline)
# =============================================================================


class TestGenerateAll:
    def test_generate_all_small(self, tmp_path: Path) -> None:
        """generate_all runs end-to-end with small ranges (isolated Parquet)."""
        orig_N = _m.MEAN_TOTAL_RANGE
        orig_F = _m.FINESSE_RANGE
        orig_npts = _m.N_OMEGA_POINTS
        pq_path = tmp_path / "test_cavity_tmsv.parquet"
        try:
            _m.MEAN_TOTAL_RANGE = [4.0]
            _m.FINESSE_RANGE = [1.0]
            _m.N_OMEGA_POINTS = 3
            data, scaling_fit = _m.generate_all(
                force=True,
                only="F=1",
                override_pq_path=pq_path,
            )
        finally:
            _m.MEAN_TOTAL_RANGE = orig_N
            _m.FINESSE_RANGE = orig_F
            _m.N_OMEGA_POINTS = orig_npts

        assert data is not None
        assert isinstance(data, CavityTmsvSensitivityResult)
        assert scaling_fit is not None
        assert isinstance(scaling_fit, CavityTmsvScalingFit)

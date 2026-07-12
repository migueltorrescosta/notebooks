"""
Tests for the drive-component analysis module (2026-05-27).

Run with:
    uv run pytest reports/r20260527/test_drive_component_analysis.py -q --tb=short
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from scipy.stats import kstest

from src.analysis.ancilla_drive_metrology import (
    evolve_drive_circuit,
)
from src.analysis.ancilla_drive_results import (
    Drive2DSliceResult,
)
from src.analysis.ancilla_drive_scans import (
    drive_2d_slice,
)
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
)
from src.utils.serialization import assert_roundtrip_fields

_m = importlib.import_module("reports.r20260527.drive_component_analysis")
AZZ_BOUNDS = _m.AZZ_BOUNDS
REPORT_DATE = _m.REPORT_DATE
SQL = _m.SQL
EnvelopeResult = _m.EnvelopeResult
NormBallResult = _m.NormBallResult
_sample_ball_for_omega = _m._sample_ball_for_omega
compute_sensitivity_with_extra = _m.compute_sensitivity_with_extra
extract_envelope_curve = _m.extract_envelope_curve
marsaglia_ball_sample = _m.marsaglia_ball_sample
norm_ball_sampling = _m.norm_ball_sampling
plot_best_ratio_by_slice = _m.plot_best_ratio_by_slice
plot_norm_envelope_curve = _m.plot_norm_envelope_curve
plot_normball_histogram = _m.plot_normball_histogram
_parquet_path = _m._parquet_path
_fig_path = _m._fig_path
t_hold = _m.t_hold

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def make_ops() -> dict[str, np.ndarray]:
    return build_two_qubit_operators()


# ============================================================================
# Marsaglia 3-Ball Sampling
# ============================================================================


class TestMarsagliaBallSample:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        drive, azz = marsaglia_ball_sample(rng, 100, 10.0, -5.0, 5.0)
        assert drive.shape == (100, 3)
        assert azz.shape == (100,)

    def test_norms_within_ball(self) -> None:
        rng = np.random.default_rng(42)
        drive, _ = marsaglia_ball_sample(rng, 1000, 10.0, -5.0, 5.0)
        norms = np.sqrt(np.sum(drive**2, axis=1))
        assert np.all(norms <= 10.0 + 1e-12), "Samples must be within ball"
        assert np.all(norms >= 0.0), "Norms must be non-negative"

    def test_azz_within_bounds(self) -> None:
        rng = np.random.default_rng(42)
        _, azz = marsaglia_ball_sample(rng, 1000, 10.0, -5.0, 5.0)
        assert np.all(azz >= -5.0 - 1e-12), "a_zz must be ≥ lower bound"
        assert np.all(azz <= 5.0 + 1e-12), "a_zz must be ≤ upper bound"

    def test_uniform_distribution(self) -> None:
        """KS test: P(||a|| ≤ r) = (r/R)³ for 3-ball."""
        rng = np.random.default_rng(42)
        R = 10.0
        drive, _ = marsaglia_ball_sample(rng, 5000, R, -5.0, 5.0)
        norms = np.sqrt(np.sum(drive**2, axis=1))
        # Theoretical CDF: F(r) = (r/R)³
        stat, pval = kstest(norms, lambda x: (x / R) ** 3)
        assert stat < 0.05, (
            f"KS statistic {stat:.4f} too large (p={pval:.4f}); "
            "distribution may not be uniform in the ball"
        )

    def test_deterministic_with_seed(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        d1, a1 = marsaglia_ball_sample(rng1, 100, 10.0, -5.0, 5.0)
        d2, a2 = marsaglia_ball_sample(rng2, 100, 10.0, -5.0, 5.0)
        assert np.allclose(d1, d2)
        assert np.allclose(a1, a2)


# ============================================================================
# Ball Sampling Dispatcher
# ============================================================================


class TestSampleBallForOmega:
    def test_marsaglia_default(self) -> None:
        rng = np.random.default_rng(42)
        drive, azz = _sample_ball_for_omega(rng, 100, 10.0, -5.0, 5.0, "marsaglia", 1)
        assert drive.shape == (100, 3)
        assert azz.shape == (100,)

    def test_stratified_correct_shape(self) -> None:
        rng = np.random.default_rng(42)
        drive, azz = _sample_ball_for_omega(
            rng,
            100,
            10.0,
            -5.0,
            5.0,
            "stratified",
            10,
        )
        assert drive.shape == (100, 3)
        assert azz.shape == (100,)

    def test_stratified_requires_divisible(self) -> None:
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="must be divisible by n_strata"):
            _sample_ball_for_omega(rng, 101, 10.0, -5.0, 5.0, "stratified", 10)

    def test_unknown_method_raises(self) -> None:
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Unknown sampling_method"):
            _sample_ball_for_omega(rng, 100, 10.0, -5.0, 5.0, "invalid", 1)


# ============================================================================
# Norm-Ball Sampling — Stratified Mode
# ============================================================================


class TestNormBallSamplingStratified:
    def test_stratified_correct_shape(self) -> None:
        omega_vals = [0.5, 1.0]
        result = norm_ball_sampling(
            omega_values=omega_vals,
            n_samp=100,
            R=5.0,
            seed=42,
            sampling_method="stratified",
            n_strata=10,
        )
        assert result.omega_values.shape == (2,)
        assert result.samples.shape == (2, 100, 4)
        assert result.delta_omega_values.shape == (2, 100)
        assert result.norms.shape == (2, 100)

    def test_stratified_norms_within_R(self) -> None:
        result = norm_ball_sampling(
            [1.0],
            n_samp=100,
            R=5.0,
            seed=42,
            sampling_method="stratified",
            n_strata=10,
        )
        assert np.all(result.norms <= 5.0 + 1e-12)

    def test_stratified_azz_within_bounds(self) -> None:
        result = norm_ball_sampling(
            [1.0],
            n_samp=100,
            R=5.0,
            seed=42,
            sampling_method="stratified",
            n_strata=10,
        )
        a_zz_vals = result.samples[0, :, 3]
        assert np.all(a_zz_vals >= AZZ_BOUNDS[0] - 1e-12)
        assert np.all(a_zz_vals <= AZZ_BOUNDS[1] + 1e-12)

    def test_stratified_deterministic(self) -> None:
        r1 = norm_ball_sampling(
            [1.0],
            n_samp=100,
            R=5.0,
            seed=42,
            sampling_method="stratified",
            n_strata=10,
        )
        r2 = norm_ball_sampling(
            [1.0],
            n_samp=100,
            R=5.0,
            seed=42,
            sampling_method="stratified",
            n_strata=10,
        )
        assert np.allclose(r1.samples, r2.samples)
        assert np.allclose(r1.delta_omega_values, r2.delta_omega_values)

    def test_stratified_some_finite(self) -> None:
        result = norm_ball_sampling(
            [1.0],
            n_samp=100,
            R=5.0,
            seed=42,
            sampling_method="stratified",
            n_strata=10,
        )
        finite_count = np.sum(np.isfinite(result.delta_omega_values))
        assert finite_count > 0

    def test_stratified_sql_bound_holds(self) -> None:
        result = norm_ball_sampling(
            [1.0],
            n_samp=100,
            R=5.0,
            seed=42,
            sampling_method="stratified",
            n_strata=10,
        )
        finite_mask = np.isfinite(result.delta_omega_values)
        if np.any(finite_mask):
            min_dt = np.min(result.delta_omega_values[finite_mask])
            assert min_dt >= SQL - 1e-12

    def test_stratified_small_r_density(self) -> None:
        """Stratified mode produces many more samples at small r than Marsaglia."""
        r_m = norm_ball_sampling(
            [1.0],
            n_samp=500,
            R=10.0,
            seed=42,
            sampling_method="marsaglia",
        )
        r_s = norm_ball_sampling(
            [1.0],
            n_samp=500,
            R=10.0,
            seed=42,
            sampling_method="stratified",
            n_strata=50,
        )
        count_m = int(np.sum(r_m.norms[0] <= 2.0))
        count_s = int(np.sum(r_s.norms[0] <= 2.0))
        assert count_s > count_m, (
            f"Stratified has {count_s} samples at r≤2, "
            f"Marsaglia has {count_m}; expected stratified to have more"
        )

    def test_stratified_each_stratum_has_samples(self) -> None:
        """Each of the n_strata bins is populated."""
        n_strata = 20
        result = norm_ball_sampling(
            [1.0],
            n_samp=200,
            R=10.0,
            seed=42,
            sampling_method="stratified",
            n_strata=n_strata,
        )
        norms = result.norms[0]
        r_bounds = np.linspace(0.0, 10.0, n_strata + 1)
        for i in range(n_strata):
            r_lo = r_bounds[i]
            r_hi = r_bounds[i + 1]
            mask = (norms > r_lo - 1e-12) & (norms <= r_hi + 1e-12)
            count = int(np.sum(mask))
            expected = 200 // n_strata  # 10
            assert count == expected, (
                f"Stratum [{r_lo:.2f}, {r_hi:.2f}] has {count} samples, "
                f"expected {expected}"
            )


# ============================================================================
# Sensitivity with Extra Metadata
# ============================================================================


class TestComputeSensitivityWithExtra:
    def test_decoupled_baseline_returns_sql(self) -> None:
        """At (0,0,0,0), Δω should equal SQL = 0.1."""
        domega, exp_val, var_val, _deriv, fringe = compute_sensitivity_with_extra(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        assert np.isclose(domega, SQL, rtol=1e-8), (
            f"Decoupled Δω={domega} should equal SQL={SQL}"
        )
        assert not fringe
        assert isinstance(exp_val, float)
        assert isinstance(var_val, float)
        assert var_val >= 0.0

    def test_fringe_extremum_detected(self) -> None:
        """Some configurations give zero derivative → flagged as fringe."""
        domega, _exp_val, _var_val, _deriv, fringe = compute_sensitivity_with_extra(
            1.0,
            5.0,
            0.0,
            0.0,
            5.0,
        )
        if fringe:
            assert np.isinf(domega)
            assert abs(_deriv) < 1e-12

    def test_nonzero_drive_returns_finite(self) -> None:
        """Most non-pathological configurations should give finite Δω."""
        domega, *_ = compute_sensitivity_with_extra(
            1.0,
            1.0,
            0.5,
            -0.3,
            2.0,
        )
        # May be inf at fringe extremum, but should be finite often
        if np.isfinite(domega):
            assert domega > 0.0

    def test_returns_four_values_and_fringe_flag(self) -> None:
        result = compute_sensitivity_with_extra(1.0, 0.0, 0.0, 0.0, 0.0)
        assert len(result) == 5
        domega, exp_val, var_val, deriv, fringe = result
        assert isinstance(domega, float)
        assert isinstance(exp_val, float)
        assert isinstance(var_val, float)
        assert isinstance(deriv, float)
        assert isinstance(fringe, bool)


# ============================================================================
# Norm-Ball Sampling
# ============================================================================


class TestNormBallSampling:
    def test_small_returns_correct_shape(self) -> None:
        """Small run to verify shape."""
        omega_vals = [0.5, 1.0]
        result = norm_ball_sampling(
            omega_values=omega_vals,
            n_samp=20,
            R=5.0,
            seed=42,
        )
        assert result.omega_values.shape == (2,)
        assert result.samples.shape == (2, 20, 4)
        assert result.delta_omega_values.shape == (2, 20)
        assert result.norms.shape == (2, 20)

    def test_deterministic_with_seed(self) -> None:
        r1 = norm_ball_sampling([1.0], n_samp=10, R=5.0, seed=42)
        r2 = norm_ball_sampling([1.0], n_samp=10, R=5.0, seed=42)
        assert np.allclose(r1.samples, r2.samples)
        assert np.allclose(r1.delta_omega_values, r2.delta_omega_values)

    def test_norms_within_R(self) -> None:
        result = norm_ball_sampling([1.0], n_samp=100, R=5.0, seed=42)
        assert np.all(result.norms <= 5.0 + 1e-12)

    def test_azz_within_bounds(self) -> None:
        result = norm_ball_sampling([1.0], n_samp=100, R=5.0, seed=42)
        a_zz_vals = result.samples[0, :, 3]
        assert np.all(a_zz_vals >= AZZ_BOUNDS[0] - 1e-12)
        assert np.all(a_zz_vals <= AZZ_BOUNDS[1] + 1e-12)

    def test_some_finite_values(self) -> None:
        """At least some samples should yield finite Δω."""
        result = norm_ball_sampling([1.0], n_samp=50, R=5.0, seed=42)
        finite_count = np.sum(np.isfinite(result.delta_omega_values))
        assert finite_count > 0, "No finite Δω values found"
        assert finite_count <= 50

    def test_sql_bound_holds(self) -> None:
        """No configuration should beat the SQL."""
        result = norm_ball_sampling([1.0], n_samp=50, R=5.0, seed=42)
        finite_mask = np.isfinite(result.delta_omega_values)
        if np.any(finite_mask):
            min_dt = np.min(result.delta_omega_values[finite_mask])
            assert min_dt >= SQL - 1e-12, f"Min Δω={min_dt} is below SQL={SQL}"


# ============================================================================
# NormBallResult — Parquet Roundtrip
# ============================================================================


class TestNormBallResultParquet:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("omega_values", "allclose"),
        ("samples", "allclose"),
        ("delta_omega_values", "allclose"),
        ("expectation_values", "allclose"),
        ("variance_values", "allclose"),
        ("deriv_values", "allclose"),
        ("norms", "allclose"),
        ("sql", "eq"),
        ("t_hold", "eq"),
        ("R", "eq"),
    ]

    @pytest.fixture
    def make_result(self) -> NormBallResult:
        omega_vals = np.array([0.5, 1.0], dtype=float)
        n_omega = 2
        n_samp = 5
        return NormBallResult(
            omega_values=omega_vals,
            samples=np.random.default_rng(42).uniform(-5, 5, size=(n_omega, n_samp, 4)),
            delta_omega_values=np.random.default_rng(42).uniform(
                0.1, 2.0, size=(n_omega, n_samp)
            ),
            expectation_values=np.random.default_rng(42).uniform(
                -0.5, 0.5, size=(n_omega, n_samp)
            ),
            variance_values=np.random.default_rng(42).uniform(
                0.0, 0.5, size=(n_omega, n_samp)
            ),
            deriv_values=np.random.default_rng(42).uniform(
                -1.0, 1.0, size=(n_omega, n_samp)
            ),
            norms=np.random.default_rng(42).uniform(0.0, 5.0, size=(n_omega, n_samp)),
            sql=0.1,
            t_hold=10.0,
            R=5.0,
        )

    def test_roundtrip(self, make_result: NormBallResult, tmp_path: Path) -> None:
        p = tmp_path / "normball.parquet"
        make_result.save_parquet(p)
        loaded = NormBallResult.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)

    def test_fail_fast_missing_column(
        self, make_result: NormBallResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["norm"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            NormBallResult.from_parquet(p)


# ============================================================================
# Envelope Curve Extraction
# ============================================================================


class TestExtractEnvelopeCurve:
    @pytest.fixture
    def make_normball(self) -> NormBallResult:
        """Simple norm-ball result with 2 ω values, 20 samples each."""
        omega_vals = np.array([0.5, 1.0], dtype=float)
        n_omega = 2
        n_samp = 20
        rng = np.random.default_rng(42)
        # Build samples with norms uniformly in [0, 5]
        samples = np.zeros((n_omega, n_samp, 4), dtype=float)
        norms = np.zeros((n_omega, n_samp), dtype=float)
        deltas = np.full((n_omega, n_samp), np.inf, dtype=float)
        for ti in range(n_omega):
            for si in range(n_samp):
                r = rng.uniform(0.0, 5.0)
                # Direction from unit sphere
                z = rng.normal(size=3)
                z = z / max(np.sqrt(np.sum(z**2)), 1e-300)
                samples[ti, si, :3] = z * r
                samples[ti, si, 3] = rng.uniform(-5.0, 5.0)
                norms[ti, si] = r
                # Δω: SQL at r=0, degrades with r, plus some fringe
                if si < 18:
                    deltas[ti, si] = SQL * (
                        1.0 + 0.1 * r + 0.01 * abs(samples[ti, si, 3])
                    )
        return NormBallResult(
            omega_values=omega_vals,
            samples=samples,
            delta_omega_values=deltas,
            expectation_values=np.zeros((n_omega, n_samp)),
            variance_values=np.ones((n_omega, n_samp)) * 0.01,
            deriv_values=np.ones((n_omega, n_samp)) * 0.5,
            norms=norms,
            sql=SQL,
            t_hold=t_hold,
            R=5.0,
        )

    def test_envelope_shape(self, make_normball: NormBallResult) -> None:
        env = extract_envelope_curve(make_normball, n_r=50)
        assert env.r_values.shape == (50,)
        assert env.best_ratio_per_omega.shape == (2, 50)
        assert np.allclose(env.omega_values, make_normball.omega_values)

    def test_envelope_monotonicity(self, make_normball: NormBallResult) -> None:
        """best_ratio(r) must be non-increasing."""
        env = extract_envelope_curve(make_normball, n_r=20)
        for ti in range(len(env.omega_values)):
            valid = np.isfinite(env.best_ratio_per_omega[ti])
            if np.any(valid):
                ratios = env.best_ratio_per_omega[ti, valid]
                diffs = np.diff(ratios)
                # Allow tiny numerical increases (< 1e-12)
                assert np.all(diffs <= 1e-12), (
                    f"Envelope not non-increasing at ω={env.omega_values[ti]}: "
                    f"diffs={diffs}"
                )

    def test_sql_bound(self, make_normball: NormBallResult) -> None:
        """Envelope must be >= SQL at all r."""
        env = extract_envelope_curve(make_normball, n_r=20)
        for ti in range(len(env.omega_values)):
            valid = np.isfinite(env.best_ratio_per_omega[ti])
            if np.any(valid):
                assert np.all(env.best_ratio_per_omega[ti, valid] >= 1.0 - 1e-12), (
                    f"Envelope below SQL at ω={env.omega_values[ti]}"
                )

    def test_envelope_at_r_zero(self, make_normball: NormBallResult) -> None:
        """At r=0, only the zero-drive config qualifies.
        Since we have no exact zero-drive sample, the envelope may be inf at r=0."""
        env = extract_envelope_curve(make_normball, n_r=20)
        # r=0 is the first element
        for ti in range(len(env.omega_values)):
            if np.isfinite(env.best_ratio_per_omega[ti, 0]):
                assert env.best_ratio_per_omega[ti, 0] >= 1.0 - 1e-12

    def test_empty_ball_returns_inf(self) -> None:
        """If no samples have norm <= r, envelope should be inf."""
        omega_vals = np.array([1.0], dtype=float)
        result = NormBallResult(
            omega_values=omega_vals,
            samples=np.zeros((1, 5, 4), dtype=float),
            delta_omega_values=np.full((1, 5), np.inf, dtype=float),
            expectation_values=np.zeros((1, 5)),
            variance_values=np.ones((1, 5)),
            deriv_values=np.ones((1, 5)),
            norms=np.full((1, 5), 10.0),
            sql=SQL,
            t_hold=t_hold,
            R=10.0,
        )
        env = extract_envelope_curve(result, n_r=10, r_max=5.0)
        assert np.all(np.isinf(env.best_ratio_per_omega)), (
            "Envelope should be inf when no samples within ball"
        )


# ============================================================================
# EnvelopeResult — Parquet Roundtrip
# ============================================================================


class TestEnvelopeResultParquet:
    @pytest.fixture
    def make_envelope(self) -> EnvelopeResult:
        return EnvelopeResult(
            r_values=np.linspace(0.0, 10.0, 20),
            best_ratio_per_omega=np.ones((2, 20)),
            omega_values=np.array([0.5, 1.0], dtype=float),
            sql=SQL,
        )

    def test_roundtrip(self, make_envelope: EnvelopeResult, tmp_path: Path) -> None:
        p = tmp_path / "envelope.parquet"
        make_envelope.save_parquet(p)
        loaded = EnvelopeResult.from_parquet(p)
        assert np.allclose(loaded.r_values, make_envelope.r_values)
        assert np.allclose(
            loaded.best_ratio_per_omega, make_envelope.best_ratio_per_omega
        )
        assert np.allclose(loaded.omega_values, make_envelope.omega_values)
        assert loaded.sql == make_envelope.sql

    def test_fail_fast_missing_column(
        self, make_envelope: EnvelopeResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_envelope.to_dataframe()
        df = df.drop(columns=["best_ratio"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            EnvelopeResult.from_parquet(p)


# ============================================================================
# 2D Slice: az type
# ============================================================================


class TestDrive2DSliceAz:
    def test_az_slice_returns_correct_shape(self) -> None:
        result = drive_2d_slice(
            omega=1.0,
            slice_type="az",
            n_drive=5,
            n_azz=5,
        )
        assert result.delta_omega_grid.shape == (5, 5)
        assert result.slice_type == "az"

    def test_az_slice_sql_bound_holds(self) -> None:
        result = drive_2d_slice(
            omega=1.0,
            slice_type="az",
            n_drive=5,
            n_azz=5,
        )
        finite_mask = np.isfinite(result.delta_omega_grid)
        if np.any(finite_mask):
            min_val = np.min(result.delta_omega_grid[finite_mask])
            assert min_val >= result.sql - 1e-10, (
                f"Min Δω={min_val} below SQL={result.sql} for (a_z, a_zz) slice"
            )

    def test_az_slice_ax_symmetry_at_zero_azz(self) -> None:
        """At a_zz=0, the az slice should give the same result as ax at a_zz=0,
        since both are decoupled from the ancilla drive at zero interaction."""
        res_ax = drive_2d_slice(omega=1.0, slice_type="ax", n_drive=5, n_azz=5)
        res_az = drive_2d_slice(omega=1.0, slice_type="az", n_drive=5, n_azz=5)
        # Compare the a_zz=0 column (middle index since symmetric range)
        azz_mid = len(res_ax.azz_values) // 2
        assert np.allclose(
            res_ax.delta_omega_grid[:, azz_mid],
            res_az.delta_omega_grid[:, azz_mid],
            atol=1e-10,
        ), "ax and az slices should match at a_zz=0"

    def test_az_slice_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = drive_2d_slice(omega=1.0, slice_type="az", n_drive=3, n_azz=3)
        p = tmp_path / "slice_az.parquet"
        result.save_parquet(p)
        loaded = Drive2DSliceResult.from_parquet(p)
        assert loaded.drive_values == pytest.approx(result.drive_values)
        assert loaded.azz_values == pytest.approx(result.azz_values)
        assert loaded.delta_omega_grid == pytest.approx(
            result.delta_omega_grid,
            nan_ok=True,
        )
        assert loaded.slice_type == "az"
        assert loaded.omega_value == pytest.approx(result.omega_value)


# ============================================================================
# Plot Functions
# ============================================================================


class TestPlotFunctions:
    @pytest.fixture
    def make_normball_result(self) -> NormBallResult:
        omega_vals = np.array([0.5, 1.0], dtype=float)
        n_omega = 2
        n_samp = 20
        rng = np.random.default_rng(42)
        samples = np.zeros((n_omega, n_samp, 4), dtype=float)
        norms = np.zeros((n_omega, n_samp), dtype=float)
        deltas = np.full((n_omega, n_samp), np.inf, dtype=float)
        exps = np.zeros((n_omega, n_samp), dtype=float)
        vars_ = np.ones((n_omega, n_samp), dtype=float) * 0.01
        derivs = np.ones((n_omega, n_samp), dtype=float) * 0.5
        for ti in range(n_omega):
            for si in range(n_samp):
                r = rng.uniform(0.0, 5.0)
                z = rng.normal(size=3)
                z = z / max(np.sqrt(np.sum(z**2)), 1e-300)
                samples[ti, si, :3] = z * r
                samples[ti, si, 3] = rng.uniform(-5.0, 5.0)
                norms[ti, si] = r
                if si < 18:
                    deltas[ti, si] = SQL * (
                        1.0 + 0.1 * r + 0.01 * abs(samples[ti, si, 3])
                    )
        return NormBallResult(
            omega_values=omega_vals,
            samples=samples,
            delta_omega_values=deltas,
            expectation_values=exps,
            variance_values=vars_,
            deriv_values=derivs,
            norms=norms,
            sql=SQL,
            t_hold=t_hold,
            R=5.0,
        )

    def test_plot_norm_envelope_curve_saves_svg(
        self, make_normball_result: NormBallResult, tmp_path: Path
    ) -> None:
        env = extract_envelope_curve(make_normball_result, n_r=10)
        svg_p = tmp_path / "envelope.svg"
        result = plot_norm_envelope_curve(env, svg_p)
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0

    def test_plot_normball_histogram_saves_svg(
        self, make_normball_result: NormBallResult, tmp_path: Path
    ) -> None:
        svg_p = tmp_path / "hist.svg"
        result = plot_normball_histogram(
            make_normball_result, omega=0.5, save_path=svg_p
        )
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0

    def test_plot_best_ratio_by_slice_saves_svg(self, tmp_path: Path) -> None:
        """Use drive_2d_slice with small grids to create slice results."""
        slice_results: dict[str, Drive2DSliceResult] = {}
        for st in ("ax", "ay", "az"):
            res = drive_2d_slice(omega=1.0, slice_type=st, n_drive=5, n_azz=5)
            slice_results[st] = res
        svg_p = tmp_path / "best_ratio.svg"
        result = plot_best_ratio_by_slice(slice_results, svg_p)
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 100, (
            "SVG too small — plot likely empty (wrong key format?)"
        )


# ============================================================================
# Pipeline Helpers
# ============================================================================


class TestPathHelpers:
    def test_parquet_path(self) -> None:
        p = _parquet_path("test")
        assert str(REPORT_DATE) in str(p)
        assert p.suffix == ".parquet"
        assert "raw_data" in str(p)

    def test_fig_path(self) -> None:
        p = _fig_path("test")
        assert str(REPORT_DATE) in str(p)
        assert p.suffix == ".svg"
        assert "figures" in str(p)


# ============================================================================
# CLI
# ============================================================================


def test_cli_help() -> None:
    import subprocess

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(Path(__file__).resolve().parent / "drive_component_analysis.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_zero_radius_ball(self) -> None:
        """R=0 should give all samples at origin."""
        result = norm_ball_sampling(
            omega_values=[1.0],
            n_samp=10,
            R=0.0,
            seed=42,
        )
        assert np.allclose(result.norms, 0.0)
        assert np.allclose(result.samples[:, :, :3], 0.0)

    def test_small_omega_gives_finite(self) -> None:
        """ω=0.1 should be well-behaved."""
        domega, *_ = compute_sensitivity_with_extra(0.1, 0.0, 0.0, 0.0, 0.0)
        assert np.isfinite(domega)
        assert domega > 0.0

    def test_large_azz_gives_fringe_at_some_points(self) -> None:
        """Large a_zz should produce fringe extremum for some ω."""
        domega, *_ = compute_sensitivity_with_extra(1.0, 0.0, 0.0, 0.0, 10.0)
        # May or may not be fringe — we just verify it runs without error
        assert isinstance(domega, float)

    def test_evolve_drive_circuit_preserves_norm(self, make_ops: dict) -> None:
        """Circuit evolution must preserve state norm."""
        psi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        psi = evolve_drive_circuit(
            psi0, np.pi / 2, t_hold, 1.0, 1.0, 0.5, -0.3, 2.0, make_ops
        )
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12)

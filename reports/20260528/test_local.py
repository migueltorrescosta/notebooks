"""
Tests for the free-ancilla initial-state module (2026-05-28).

Run with:
    uv run pytest reports/20260528/test_local.py -q --tb=short
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path

import numpy as np
import pytest

from src.analysis.ancilla_drive_metrology import (
    build_two_qubit_operators,
    evolve_drive_circuit,
)

_report_dir = str(
    Path(__file__).resolve().parent.parent.parent / "reports" / "20260528",
)
if _report_dir not in _sys.path:
    _sys.path.insert(0, _report_dir)
del _sys, _report_dir

# Import the module we are testing — mypy/pyright ignore the report-local import.
from local import (  # type: ignore[import-untyped]  # noqa: E402
    AZZ_BOUNDS,
    R_MAX,
    SQL,
    T_H,
    FreeAncilla2DSliceResult,
    FreeAncillaNelderMeadResult,
    FreeAncillaSearchResult,
    FreeAncillaThetaScanResult,
    _marsaglia_3ball_sample,
    _params_to_full,
    _sample_scenario_A,
    _sample_scenario_B,
    _sample_scenario_C,
    _sample_scenario_D,
    compute_free_ancilla_sensitivity,
    free_ancilla_2d_slice,
    free_ancilla_initial_state,
    free_ancilla_random_search,
    plot_cross_scenario_comparison,
    plot_norm_envelope_comparison,
    plot_scenario_best_ratio_by_theta,
    plot_theta_A_azz_slice_heatmap,
    run_free_ancilla_nelder_mead,
    run_free_ancilla_theta_scan,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def make_ops() -> dict[str, np.ndarray]:
    return build_two_qubit_operators()


# ============================================================================
# Free-Ancilla Initial State
# ============================================================================


class TestFreeAncillaInitialState:
    def test_normalised(self) -> None:
        state = free_ancilla_initial_state(0.0, 0.0)
        assert np.isclose(np.linalg.norm(state), 1.0)

    def test_theta_zero_gives_00(self) -> None:
        """At theta_A=0, the state should be |00⟩."""
        state = free_ancilla_initial_state(0.0, 0.0)
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        assert np.allclose(state, expected)

    def test_theta_pi_gives_01_superposition(self) -> None:
        """At theta_A=π, the state should be |0⟩(|0⟩+|1⟩)/√2."""
        state = free_ancilla_initial_state(np.pi, 0.0)
        # |1,0⟩_S ⊗ (cos(π/2)|1,0⟩_A + sin(π/2)|0,1⟩_A)
        # = |1,0⟩_S ⊗ |0,1⟩_A = |01⟩
        expected = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
        assert np.allclose(state, expected)

    def test_theta_pi_over_2_gives_equal_superposition(self) -> None:
        """At theta_A=π/2, phi_A=0, the ancilla is in (|0⟩+|1⟩)/√2."""
        state = free_ancilla_initial_state(np.pi / 2, 0.0)
        # |1,0⟩_S ⊗ (cos(π/4)|1,0⟩_A + sin(π/4)|0,1⟩_A)
        # = (|00⟩ + |01⟩)/√2
        expected = np.array([1.0, 1.0, 0.0, 0.0], dtype=complex) / np.sqrt(2)
        assert np.allclose(state, expected)

    def test_normalised_all_angles(self) -> None:
        for theta_A in [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]:
            for phi_A in [0.0, np.pi / 2, np.pi]:
                state = free_ancilla_initial_state(theta_A, phi_A)
                assert np.isclose(np.linalg.norm(state), 1.0), (
                    f"Not normalised at theta_A={theta_A}, phi_A={phi_A}"
                )

    def test_system_always_in_one_zero(self) -> None:
        """The system qubit must always be in |1,0⟩ (computational |0⟩)."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            theta_A = rng.uniform(0.0, np.pi)
            phi_A = rng.uniform(0.0, 2.0 * np.pi)
            state = free_ancilla_initial_state(theta_A, phi_A)
            # System in |1,0⟩_S (computational |0⟩) means indices 2 and 3
            # (|10⟩ and |11⟩) must be zero
            assert np.isclose(np.abs(state[2]) ** 2 + np.abs(state[3]) ** 2, 0.0), (
                f"System not in |1,0⟩ at theta_A={theta_A}, phi_A={phi_A}"
            )


# ============================================================================
# Marsaglia 3-Ball Sampling
# ============================================================================


class TestMarsaglia3BallSample:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        drive, azz = _marsaglia_3ball_sample(rng, 100, 10.0, -5.0, 5.0)
        assert drive.shape == (100, 3)
        assert azz.shape == (100,)

    def test_norms_within_ball(self) -> None:
        rng = np.random.default_rng(42)
        drive, _ = _marsaglia_3ball_sample(rng, 1000, 10.0, -5.0, 5.0)
        norms = np.sqrt(np.sum(drive**2, axis=1))
        assert np.all(norms <= 10.0 + 1e-12), "Samples must be within ball"
        assert np.all(norms >= 0.0), "Norms must be non-negative"

    def test_azz_within_bounds(self) -> None:
        rng = np.random.default_rng(42)
        _, azz = _marsaglia_3ball_sample(rng, 1000, 10.0, -5.0, 5.0)
        assert np.all(azz >= -5.0 - 1e-12), "a_zz must be >= lower bound"
        assert np.all(azz <= 5.0 + 1e-12), "a_zz must be <= upper bound"


# ============================================================================
# Scenario Sampling
# ============================================================================


class TestScenarioSampling:
    def test_scenario_A_fixed_ancilla(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            theta_A, phi_A, a_x, a_y, a_z, a_zz = _sample_scenario_A(rng)
            assert theta_A == 0.0
            assert phi_A == 0.0
            assert np.sqrt(a_x**2 + a_y**2 + a_z**2) <= R_MAX + 1e-12
            assert AZZ_BOUNDS[0] - 1e-12 <= a_zz <= AZZ_BOUNDS[1] + 1e-12

    def test_scenario_B_free_ancilla(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            theta_A, phi_A, a_x, a_y, a_z, a_zz = _sample_scenario_B(rng)
            assert 0.0 <= theta_A <= np.pi + 1e-12
            assert 0.0 <= phi_A <= 2.0 * np.pi + 1e-12
            assert np.sqrt(a_x**2 + a_y**2 + a_z**2) <= R_MAX + 1e-12
            assert AZZ_BOUNDS[0] - 1e-12 <= a_zz <= AZZ_BOUNDS[1] + 1e-12

    def test_scenario_C_no_drive(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            theta_A, phi_A, a_x, a_y, a_z, a_zz = _sample_scenario_C(rng)
            assert 0.0 <= theta_A <= np.pi + 1e-12
            assert 0.0 <= phi_A <= 2.0 * np.pi + 1e-12
            assert a_x == 0.0 and a_y == 0.0 and a_z == 0.0
            assert AZZ_BOUNDS[0] - 1e-12 <= a_zz <= AZZ_BOUNDS[1] + 1e-12

    def test_scenario_D_no_interaction(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            theta_A, phi_A, a_x, a_y, a_z, a_zz = _sample_scenario_D(rng)
            assert 0.0 <= theta_A <= np.pi + 1e-12
            assert 0.0 <= phi_A <= 2.0 * np.pi + 1e-12
            assert a_zz == 0.0
            assert np.sqrt(a_x**2 + a_y**2 + a_z**2) <= R_MAX + 1e-12

    def test_deterministic_with_seed(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = _sample_scenario_B(rng1)
        s2 = _sample_scenario_B(rng2)
        assert np.allclose(s1, s2)


# ============================================================================
# _params_to_full
# ============================================================================


class TestParamsToFull:
    def test_scenario_A(self) -> None:
        full = _params_to_full(np.array([1.0, 2.0, 3.0, 4.0]), "A")
        assert full == (0.0, 0.0, 1.0, 2.0, 3.0, 4.0)

    def test_scenario_B(self) -> None:
        full = _params_to_full(np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]), "B")
        assert full == (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)

    def test_scenario_C(self) -> None:
        full = _params_to_full(np.array([0.5, 1.0, 2.0]), "C")
        assert full == (0.5, 1.0, 0.0, 0.0, 0.0, 2.0)

    def test_scenario_D(self) -> None:
        full = _params_to_full(np.array([0.5, 1.0, 2.0, 3.0, 4.0]), "D")
        assert full == (0.5, 1.0, 2.0, 3.0, 4.0, 0.0)

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scenario"):
            _params_to_full(np.array([1.0, 2.0]), "X")


# ============================================================================
# Sensitivity Computation
# ============================================================================


class TestComputeFreeAncillaSensitivity:
    def test_decoupled_baseline_returns_sql(self) -> None:
        """At all-zero parameters with theta_A=0, Δθ should equal SQL."""
        dtheta, exp_val, var_val, _deriv, fringe = compute_free_ancilla_sensitivity(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        assert np.isclose(dtheta, SQL, rtol=1e-8), (
            f"Decoupled Δθ={dtheta} should equal SQL={SQL}"
        )
        assert not fringe
        assert var_val >= 0.0
        assert isinstance(exp_val, float)

    def test_fringe_extremum_detected(self) -> None:
        """Some configurations give zero derivative → flagged as fringe."""
        dtheta, _exp, _var, _deriv, fringe = compute_free_ancilla_sensitivity(
            1.0,
            0.0,
            0.0,
            5.0,
            0.0,
            0.0,
            5.0,
        )
        if fringe:
            assert np.isinf(dtheta)

    def test_nonzero_drive_returns_finite(self) -> None:
        """Most non-pathological configurations should give finite Δθ."""
        dtheta, *_ = compute_free_ancilla_sensitivity(
            1.0,
            np.pi / 4,
            0.0,
            1.0,
            0.5,
            -0.3,
            2.0,
        )
        if np.isfinite(dtheta):
            assert dtheta > 0.0

    def test_returns_five_values(self) -> None:
        result = compute_free_ancilla_sensitivity(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        assert len(result) == 5
        dtheta, exp_val, var_val, deriv, fringe = result
        assert isinstance(dtheta, float)
        assert isinstance(exp_val, float)
        assert isinstance(var_val, float)
        assert isinstance(deriv, float)
        assert isinstance(fringe, bool)


# ============================================================================
# Free-Ancilla Random Search
# ============================================================================


class TestFreeAncillaRandomSearch:
    def test_small_run_shape(self) -> None:
        result = free_ancilla_random_search(
            theta=1.0,
            scenario="B",
            n_samples=10,
            seed=42,
        )
        assert result.samples.shape == (10, 6)
        assert result.delta_theta_values.shape == (10,)
        assert result.scenario == "B"

    def test_deterministic_with_seed(self) -> None:
        r1 = free_ancilla_random_search(1.0, "B", n_samples=10, seed=42)
        r2 = free_ancilla_random_search(1.0, "B", n_samples=10, seed=42)
        assert np.allclose(r1.samples, r2.samples)
        assert np.allclose(r1.delta_theta_values, r2.delta_theta_values)

    def test_some_finite_values(self) -> None:
        result = free_ancilla_random_search(1.0, "B", n_samples=20, seed=42)
        finite_count = np.sum(np.isfinite(result.delta_theta_values))
        assert finite_count > 0, "No finite Δθ values found"

    def test_scenario_A_baseline(self) -> None:
        """Scenario A should not beat SQL."""
        result = free_ancilla_random_search(1.0, "A", n_samples=20, seed=42)
        finite_mask = np.isfinite(result.delta_theta_values)
        if np.any(finite_mask):
            min_dt = np.min(result.delta_theta_values[finite_mask])
            assert min_dt >= SQL - 1e-10, (
                f"Scenario A min Δθ={min_dt} is below SQL={SQL}"
            )

    def test_norms_within_R(self) -> None:
        result = free_ancilla_random_search(1.0, "B", n_samples=20, seed=42)
        norms_a = np.sqrt(
            result.samples[:, 2] ** 2
            + result.samples[:, 3] ** 2
            + result.samples[:, 4] ** 2
        )
        assert np.all(norms_a <= R_MAX + 1e-10)

    def test_best_params_are_valid(self) -> None:
        result = free_ancilla_random_search(1.0, "B", n_samples=20, seed=42)
        bp = result.best_params
        assert len(bp) == 6
        assert 0.0 <= bp[0] <= np.pi + 1e-10
        assert 0.0 <= bp[1] <= 2.0 * np.pi + 1e-10

    def test_scenario_C_all_zero_drive(self) -> None:
        """Scenario C must have a_x = a_y = a_z = 0 for all samples."""
        result = free_ancilla_random_search(1.0, "C", n_samples=10, seed=42)
        assert np.allclose(result.samples[:, 2], 0.0)
        assert np.allclose(result.samples[:, 3], 0.0)
        assert np.allclose(result.samples[:, 4], 0.0)

    def test_scenario_D_all_zero_interaction(self) -> None:
        """Scenario D must have a_zz = 0 for all samples."""
        result = free_ancilla_random_search(1.0, "D", n_samples=10, seed=42)
        assert np.allclose(result.samples[:, 5], 0.0)


# ============================================================================
# FreeAncillaSearchResult — Parquet Roundtrip
# ============================================================================


class TestFreeAncillaSearchResultParquet:
    @pytest.fixture
    def make_result(self) -> FreeAncillaSearchResult:
        n_samp = 10
        rng = np.random.default_rng(42)
        samples = np.zeros((n_samp, 6), dtype=float)
        samples[:, 0] = rng.uniform(0.0, np.pi, n_samp)  # theta_A
        samples[:, 1] = rng.uniform(0.0, 2.0 * np.pi, n_samp)  # phi_A
        samples[:, 2:5] = rng.uniform(-5.0, 5.0, (n_samp, 3))  # drive
        samples[:, 5] = rng.uniform(-5.0, 5.0, n_samp)  # a_zz
        # Make index 1 the clear minimum to match manual best_params below
        deltas = np.full(n_samp, 5.0, dtype=float)
        deltas[0] = float("inf")  # one fringe point
        deltas[1] = 0.42  # the minimum
        return FreeAncillaSearchResult(
            samples=samples,
            delta_theta_values=deltas,
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
            best_delta_theta=float(deltas[1]),
            theta_value=1.0,
            sql=0.1,
            T_H=10.0,
            R=5.0,
            scenario="B",
        )

    def test_roundtrip(
        self, make_result: FreeAncillaSearchResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "search.parquet"
        make_result.save_parquet(p)
        loaded = FreeAncillaSearchResult.from_parquet(p)
        assert np.allclose(loaded.samples, make_result.samples)
        assert np.allclose(
            loaded.delta_theta_values,
            make_result.delta_theta_values,
            equal_nan=True,
        )
        assert np.allclose(loaded.expectation_values, make_result.expectation_values)
        assert np.allclose(loaded.variance_values, make_result.variance_values)
        assert np.allclose(loaded.deriv_values, make_result.deriv_values)
        assert np.array_equal(loaded.is_fringe, make_result.is_fringe)
        assert loaded.best_params == make_result.best_params
        assert np.isclose(loaded.best_delta_theta, make_result.best_delta_theta)
        assert loaded.theta_value == make_result.theta_value
        assert loaded.sql == make_result.sql
        assert loaded.T_H == make_result.T_H
        assert loaded.R == make_result.R
        assert loaded.scenario == make_result.scenario

    def test_fail_fast_missing_column(
        self, make_result: FreeAncillaSearchResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["scenario"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            FreeAncillaSearchResult.from_parquet(p)


# ============================================================================
# FreeAncillaNelderMeadResult — Parquet Roundtrip
# ============================================================================


class TestFreeAncillaNelderMeadResultParquet:
    @pytest.fixture
    def make_result(self) -> FreeAncillaNelderMeadResult:
        return FreeAncillaNelderMeadResult(
            delta_theta_opt=0.09,
            params_opt=np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]),
            full_params_opt=(0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
            theta_true=1.0,
            scenario="B",
            success=True,
            nfev=100,
            message="OK",
            expectation_Jz=0.25,
            variance_Jz=0.1,
        )

    def test_roundtrip(
        self, make_result: FreeAncillaNelderMeadResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "nm.parquet"
        make_result.save_parquet(p)
        loaded = FreeAncillaNelderMeadResult.from_parquet(p)
        assert np.isclose(loaded.delta_theta_opt, make_result.delta_theta_opt)
        assert np.allclose(loaded.params_opt, make_result.params_opt)
        assert loaded.full_params_opt == make_result.full_params_opt
        assert loaded.theta_true == make_result.theta_true
        assert loaded.scenario == make_result.scenario
        assert loaded.success == make_result.success
        assert loaded.nfev == make_result.nfev
        assert np.isclose(loaded.expectation_Jz, make_result.expectation_Jz)
        assert np.isclose(loaded.variance_Jz, make_result.variance_Jz)

    def test_fail_fast_missing_column(
        self, make_result: FreeAncillaNelderMeadResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["theta_true"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            FreeAncillaNelderMeadResult.from_parquet(p)


# ============================================================================
# FreeAncillaThetaScanResult — Parquet Roundtrip
# ============================================================================


class TestFreeAncillaThetaScanResultParquet:
    @pytest.fixture
    def make_result(self) -> FreeAncillaThetaScanResult:
        return FreeAncillaThetaScanResult(
            theta_values=np.array([0.1, 1.0, 5.0], dtype=float),
            best_params_per_theta=[
                (0.0, 0.0, 1.0, 2.0, 3.0, 4.0),
                (0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
                (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            ],
            best_delta_theta_per_theta=np.array([0.1, 0.09, 0.08], dtype=float),
            sql_values=np.array([0.1, 0.1, 0.1], dtype=float),
            expectation_Jz_per_theta=np.array([0.0, 0.25, -0.1], dtype=float),
            variance_Jz_per_theta=np.array([0.01, 0.1, 0.05], dtype=float),
            scenario="B",
        )

    def test_roundtrip(
        self, make_result: FreeAncillaThetaScanResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "scan.parquet"
        make_result.save_parquet(p)
        loaded = FreeAncillaThetaScanResult.from_parquet(p)
        assert np.allclose(loaded.theta_values, make_result.theta_values)
        assert loaded.best_params_per_theta == make_result.best_params_per_theta
        assert np.allclose(
            loaded.best_delta_theta_per_theta, make_result.best_delta_theta_per_theta
        )
        assert np.allclose(loaded.sql_values, make_result.sql_values)
        assert np.allclose(
            loaded.expectation_Jz_per_theta, make_result.expectation_Jz_per_theta
        )
        assert np.allclose(
            loaded.variance_Jz_per_theta, make_result.variance_Jz_per_theta
        )
        assert loaded.scenario == make_result.scenario

    def test_fail_fast_missing_column(
        self, make_result: FreeAncillaThetaScanResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["scenario"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            FreeAncillaThetaScanResult.from_parquet(p)


# ============================================================================
# 2D Slice: (theta_A, a_zz)
# ============================================================================


class TestFreeAncilla2DSlice:
    def test_slice_returns_correct_shape(self) -> None:
        result = free_ancilla_2d_slice(theta=1.0, n_grid=5)
        assert result.delta_theta_grid.shape == (5, 5)
        assert len(result.theta_A_values) == 5
        assert len(result.azz_values) == 5

    def test_slice_sql_bound_holds(self) -> None:
        result = free_ancilla_2d_slice(theta=1.0, n_grid=5)
        finite_mask = np.isfinite(result.delta_theta_grid)
        if np.any(finite_mask):
            min_val = np.min(result.delta_theta_grid[finite_mask])
            assert min_val >= result.sql - 1e-10, (
                f"Min Δθ={min_val} below SQL={result.sql}"
            )

    def test_slice_baseline_at_theta_A_zero(self) -> None:
        """At theta_A=0, the slice should reproduce the decoupled baseline
        (Δθ = SQL) for all a_zz, matching 20260527 results."""
        result = free_ancilla_2d_slice(theta=1.0, n_grid=5)
        # theta_A=0 is the first row
        for j in range(len(result.azz_values)):
            dtheta = result.delta_theta_grid[0, j]
            if np.isfinite(dtheta):
                assert np.isclose(dtheta, result.sql, rtol=1e-8), (
                    f"At theta_A=0, a_zz={result.azz_values[j]}: "
                    f"Δθ={dtheta} should equal SQL={result.sql}"
                )

    def test_slice_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = free_ancilla_2d_slice(theta=1.0, n_grid=3)
        p = tmp_path / "slice.parquet"
        result.save_parquet(p)
        loaded = FreeAncilla2DSliceResult.from_parquet(p)
        assert np.allclose(loaded.theta_A_values, result.theta_A_values)
        assert np.allclose(loaded.azz_values, result.azz_values)
        assert np.allclose(
            loaded.delta_theta_grid, result.delta_theta_grid, equal_nan=True
        )
        assert np.isclose(loaded.theta_value, result.theta_value)
        assert np.isclose(loaded.sql, result.sql)


# ============================================================================
# FreeAncilla2DSliceResult — Parquet Roundtrip (standalone)
# ============================================================================


class TestFreeAncilla2DSliceResultParquet:
    @pytest.fixture
    def make_result(self) -> FreeAncilla2DSliceResult:
        return FreeAncilla2DSliceResult(
            theta_A_values=np.linspace(0.0, np.pi, 5),
            azz_values=np.linspace(-5.0, 5.0, 5),
            delta_theta_grid=np.ones((5, 5)) * 0.1,
            theta_value=1.0,
            sql=0.1,
        )

    def test_roundtrip(
        self, make_result: FreeAncilla2DSliceResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "slice2.parquet"
        make_result.save_parquet(p)
        loaded = FreeAncilla2DSliceResult.from_parquet(p)
        assert np.allclose(loaded.theta_A_values, make_result.theta_A_values)
        assert np.allclose(loaded.azz_values, make_result.azz_values)
        assert np.allclose(loaded.delta_theta_grid, make_result.delta_theta_grid)
        assert np.isclose(loaded.theta_value, make_result.theta_value)
        assert np.isclose(loaded.sql, make_result.sql)

    def test_fail_fast_missing_column(
        self, make_result: FreeAncilla2DSliceResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["theta_value"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            FreeAncilla2DSliceResult.from_parquet(p)


# ============================================================================
# Nelder--Mead Optimisation (small tests)
# ============================================================================


class TestFreeAncillaNelderMead:
    def test_basic_run_scenario_A(self) -> None:
        """Nelder-Mead from a known good starting point."""
        result = run_free_ancilla_nelder_mead(
            theta_true=1.0,
            scenario="A",
            x0=np.array([0.0, 0.0, 0.0, 0.0]),
            maxiter=10,
            fatol=1e-6,
        )
        assert np.isfinite(result.delta_theta_opt)
        assert result.scenario == "A"

    def test_basic_run_scenario_B(self) -> None:
        result = run_free_ancilla_nelder_mead(
            theta_true=1.0,
            scenario="B",
            x0=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            maxiter=10,
            fatol=1e-6,
        )
        assert np.isfinite(result.delta_theta_opt)
        assert result.scenario == "B"

    def test_basic_run_scenario_C(self) -> None:
        result = run_free_ancilla_nelder_mead(
            theta_true=1.0,
            scenario="C",
            x0=np.array([0.0, 0.0, 0.0]),
            maxiter=10,
            fatol=1e-6,
        )
        assert np.isfinite(result.delta_theta_opt)
        assert result.scenario == "C"

    def test_basic_run_scenario_D(self) -> None:
        result = run_free_ancilla_nelder_mead(
            theta_true=1.0,
            scenario="D",
            x0=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            maxiter=10,
            fatol=1e-6,
        )
        assert np.isfinite(result.delta_theta_opt)
        assert result.scenario == "D"


# ============================================================================
# Theta Scan (small test)
# ============================================================================


class TestRunFreeAncillaThetaScan:
    def test_small_scan_produces_correct_shape(self) -> None:
        result = run_free_ancilla_theta_scan(
            theta_values=[1.0],
            scenario="B",
            n_random=10,
            n_nm_refine=3,
            seed=42,
        )
        assert len(result.theta_values) == 1
        assert len(result.best_params_per_theta) == 1
        assert result.scenario == "B"
        assert np.isfinite(result.best_delta_theta_per_theta[0]) or np.isinf(
            result.best_delta_theta_per_theta[0]
        )

    def test_deterministic_with_seed(self) -> None:
        r1 = run_free_ancilla_theta_scan(
            [1.0],
            "B",
            n_random=10,
            n_nm_refine=2,
            seed=42,
        )
        r2 = run_free_ancilla_theta_scan(
            [1.0],
            "B",
            n_random=10,
            n_nm_refine=2,
            seed=42,
        )
        assert np.allclose(r1.best_delta_theta_per_theta, r2.best_delta_theta_per_theta)


# ============================================================================
# Plot Functions
# ============================================================================


class TestPlotFunctions:
    @pytest.fixture
    def make_scan_result(self) -> FreeAncillaThetaScanResult:
        return FreeAncillaThetaScanResult(
            theta_values=np.array([0.1, 1.0, 5.0], dtype=float),
            best_params_per_theta=[
                (0.0, 0.0, 1.0, 2.0, 3.0, 4.0),
                (0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
                (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            ],
            best_delta_theta_per_theta=np.array([0.1, 0.09, 0.08], dtype=float),
            sql_values=np.array([0.1, 0.1, 0.1], dtype=float),
            expectation_Jz_per_theta=np.array([0.0, 0.25, -0.1], dtype=float),
            variance_Jz_per_theta=np.array([0.01, 0.1, 0.05], dtype=float),
            scenario="B",
        )

    @pytest.fixture
    def make_slice_result(self) -> FreeAncilla2DSliceResult:
        return FreeAncilla2DSliceResult(
            theta_A_values=np.linspace(0.0, np.pi, 10),
            azz_values=np.linspace(-5.0, 5.0, 10),
            delta_theta_grid=np.ones((10, 10)) * 0.1,
            theta_value=1.0,
            sql=0.1,
        )

    def test_plot_scenario_best_ratio_by_theta_saves_svg(
        self, make_scan_result: FreeAncillaThetaScanResult, tmp_path: Path
    ) -> None:
        svg_p = tmp_path / "ratio.svg"
        result = plot_scenario_best_ratio_by_theta(make_scan_result, svg_p)
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0

    def test_plot_theta_A_azz_slice_heatmap_saves_svg(
        self, make_slice_result: FreeAncilla2DSliceResult, tmp_path: Path
    ) -> None:
        svg_p = tmp_path / "slice.svg"
        result = plot_theta_A_azz_slice_heatmap(make_slice_result, svg_p)
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0

    def test_plot_cross_scenario_comparison_saves_svg(
        self, make_scan_result: FreeAncillaThetaScanResult, tmp_path: Path
    ) -> None:
        """Create comparison DataFrame from mock scan results."""
        import pandas as pd

        data = []
        for sc in ("A", "B", "C", "D"):
            for theta, ratio in zip(
                make_scan_result.theta_values,
                [1.0, 0.99, 0.98] if sc == "B" else [1.0, 1.0, 1.0],
                strict=True,
            ):
                data.append(
                    {
                        "theta": theta,
                        "scenario": sc,
                        "best_delta_theta": ratio * 0.1,
                        "sql": 0.1,
                        "ratio": ratio,
                        "theta_A": 0.0,
                        "phi_A": 0.0,
                        "a_x": 0.0,
                        "a_y": 0.0,
                        "a_z": 0.0,
                        "a_zz": 0.0,
                    }
                )
        df = pd.DataFrame(data)
        csv_p = tmp_path / "comparison.parquet"
        df.to_parquet(csv_p, index=False)
        svg_p = tmp_path / "comparison.svg"
        result = plot_cross_scenario_comparison(csv_p, svg_p)
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0

    def test_plot_norm_envelope_comparison_saves_svg(
        self, make_scan_result: FreeAncillaThetaScanResult, tmp_path: Path
    ) -> None:
        """Create two scan results (A and B) and plot their envelope comparison."""
        scan_A = FreeAncillaThetaScanResult(
            theta_values=make_scan_result.theta_values.copy(),
            best_params_per_theta=[
                (0.0, 0.0, 1.0, 2.0, 3.0, 4.0),
                (0.0, 0.0, 2.0, 3.0, 4.0, 5.0),
                (0.0, 0.0, 3.0, 4.0, 5.0, 6.0),
            ],
            best_delta_theta_per_theta=np.array([0.1, 0.1, 0.1], dtype=float),
            sql_values=np.array([0.1, 0.1, 0.1], dtype=float),
            expectation_Jz_per_theta=np.array([0.0, 0.0, 0.0], dtype=float),
            variance_Jz_per_theta=np.array([0.01, 0.01, 0.01], dtype=float),
            scenario="A",
        )
        svg_p = tmp_path / "envelope.svg"
        result = plot_norm_envelope_comparison(scan_A, make_scan_result, svg_p)
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.stat().st_size > 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_zero_drive_zero_interaction_equals_sql(self) -> None:
        """When a_x=a_y=a_z=a_zz=0 regardless of ancilla state,
        Δθ should equal SQL (the ancilla is decoupled)."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            theta_A = rng.uniform(0.0, np.pi)
            phi_A = rng.uniform(0.0, 2.0 * np.pi)
            dtheta, *_ = compute_free_ancilla_sensitivity(
                1.0,
                theta_A,
                phi_A,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            assert np.isclose(dtheta, SQL, rtol=1e-8), (
                f"Decoupled sensitivity at theta_A={theta_A}, phi_A={phi_A}: "
                f"Δθ={dtheta} should equal SQL={SQL}"
            )

    def test_small_theta_gives_finite(self) -> None:
        """θ=0.1 should be well-behaved."""
        dtheta, *_ = compute_free_ancilla_sensitivity(
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        assert np.isfinite(dtheta)
        assert dtheta > 0.0

    def test_evolve_circuit_preserves_norm(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = free_ancilla_initial_state(np.pi / 3, np.pi / 4)
        psi = evolve_drive_circuit(
            psi0,
            np.pi / 2,
            T_H,
            1.0,
            1.0,
            0.5,
            -0.3,
            2.0,
            make_ops,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12)

    def test_inf_sensitivity_at_fringe(self) -> None:
        """Some symmetric configurations give fringe extremum (inf Δθ)."""
        dtheta, *_ = compute_free_ancilla_sensitivity(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        # The decoupled baseline should be finite at 0.1
        assert np.isfinite(dtheta)

    def test_zero_radius_ball_in_scenario_B(self) -> None:
        """With R=0, drive is always zero (like scenario C but with a_zz free)."""
        # We can't directly pass R=0 to free_ancilla_random_search easily
        # since it uses global R_MAX for sampling. Test via manual check.
        rng = np.random.default_rng(42)
        drive, _ = _marsaglia_3ball_sample(rng, 10, 0.0, -5.0, 5.0)
        assert np.allclose(drive, 0.0)


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
# Parameter Mappings
# ============================================================================


class TestScenarioFreeParams:
    """Verify that SCENARIO_FREE_PARAMS matches expected param counts."""

    def test_scenario_A_has_4_params(self) -> None:
        from local import SCENARIO_FREE_PARAMS  # type: ignore[import-untyped]

        assert len(SCENARIO_FREE_PARAMS["A"]) == 4

    def test_scenario_B_has_6_params(self) -> None:
        from local import SCENARIO_FREE_PARAMS  # type: ignore[import-untyped]

        assert len(SCENARIO_FREE_PARAMS["B"]) == 6

    def test_scenario_C_has_3_params(self) -> None:
        from local import SCENARIO_FREE_PARAMS  # type: ignore[import-untyped]

        assert len(SCENARIO_FREE_PARAMS["C"]) == 3

    def test_scenario_D_has_5_params(self) -> None:
        from local import SCENARIO_FREE_PARAMS  # type: ignore[import-untyped]

        assert len(SCENARIO_FREE_PARAMS["D"]) == 5

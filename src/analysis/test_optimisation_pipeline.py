"""Tests for :mod:`src.analysis.optimisation_pipeline`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pytest

from src.analysis.optimisation_pipeline import (
    TwoPhaseConfig,
    penalized_objective,
    run_nelder_mead,
    run_omega_scan,
    run_random_search,
    run_two_phase_pipeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_quadratic(offset: float = 0.0) -> Callable:
    """Simple quadratic centred at the origin (minimum = offset at zero)."""

    def obj(p: np.ndarray) -> float:
        return float(np.sum(p**2)) + offset

    return obj


def _make_rosenbrock() -> Callable:
    """2D Rosenbrock valley (minimum = 0 at [1, 1])."""

    def obj(p: np.ndarray) -> float:
        x, y = float(p[0]), float(p[1])
        return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2

    return obj


# ---------------------------------------------------------------------------
# penalized_objective
# ---------------------------------------------------------------------------


class TestPenalizedObjective:
    def test_no_penalty_inside_bounds(self) -> None:
        obj = _make_quadratic()
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        val = penalized_objective(np.array([1.0, 1.0]), obj, bounds)
        assert val == pytest.approx(2.0)

    def test_penalty_outside_bounds(self) -> None:
        obj = _make_quadratic()
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        val = penalized_objective(np.array([3.0, 0.0]), obj, bounds)
        # penalty = 1e6 * (3 - 2)^2 = 1e6
        assert val > 1e6

    def test_penalty_below_bounds(self) -> None:
        obj = _make_quadratic()
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        val = penalized_objective(np.array([-3.0, 0.0]), obj, bounds)
        assert val > 1e6

    def test_mixed_compliance(self) -> None:
        obj = _make_quadratic()
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        val = penalized_objective(np.array([3.0, 1.0]), obj, bounds)
        # Only first param violates: penalty = 1e6 * 1^2 = 1e6
        # Returns 1e10 + penalty when violated
        assert val == pytest.approx(1e10 + 1e6)

    def test_custom_penalty_scale(self) -> None:
        obj = _make_quadratic()
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        val = penalized_objective(
            np.array([3.0, 0.0]),
            obj,
            bounds,
            penalty_scale=100.0,
        )
        # penalty = 100 * 1^2 = 100; returns 1e10 + penalty
        assert val == pytest.approx(1e10 + 100.0)


# ---------------------------------------------------------------------------
# run_random_search
# ---------------------------------------------------------------------------


class TestRunRandomSearch:
    def test_returns_correct_shape(self) -> None:
        obj = _make_quadratic()
        samples, values = run_random_search(obj, n_params=4, n_samples=100, seed=42)
        assert samples.shape == (100, 4)
        assert values.shape == (100,)

    def test_deterministic_with_seed(self) -> None:
        obj = _make_quadratic()
        s1, v1 = run_random_search(obj, n_params=2, n_samples=50, seed=123)
        s2, v2 = run_random_search(obj, n_params=2, n_samples=50, seed=123)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(v1, v2)

    def test_minimum_found_near_origin(self) -> None:
        obj = _make_quadratic()
        _samples, values = run_random_search(
            obj,
            n_params=2,
            n_samples=10000,
            bounds=(-5.0, 5.0),
            seed=42,
        )
        best_idx = int(np.argmin(values))
        best_val = values[best_idx]
        # With 10k samples in [-5,5]^2, the minimum should be < 0.1
        assert best_val < 0.1

    def test_per_dimension_bounds(self) -> None:
        obj = _make_quadratic()
        per_dim_bounds = [(-10.0, -9.0), (9.0, 10.0)]
        samples, _values = run_random_search(
            obj,
            n_params=2,
            n_samples=100,
            bounds=per_dim_bounds,
            seed=42,
        )
        # All a_0 should be in [-10, -9], all a_1 in [9, 10]
        assert np.all(samples[:, 0] <= -9.0)
        assert np.all(samples[:, 0] >= -10.0)
        assert np.all(samples[:, 1] <= 10.0)
        assert np.all(samples[:, 1] >= 9.0)

    def test_penalized_outside_bounds(self) -> None:
        """All samples in [-5,5] so bound penalty is zero for quadratic."""
        obj = _make_quadratic()
        _samples, values = run_random_search(
            obj,
            n_params=2,
            n_samples=50,
            bounds=(-5.0, 5.0),
            seed=42,
        )
        assert np.all(np.isfinite(values))
        assert np.all(values < 1e6)  # no penalties triggered


# ---------------------------------------------------------------------------
# run_nelder_mead
# ---------------------------------------------------------------------------


class TestRunNelderMead:
    def test_quadratic_1d(self) -> None:
        obj = _make_quadratic()
        result = run_nelder_mead(obj, x0=np.array([10.0]), maxiter=5000)
        assert result["success"]
        assert result["fun_opt"] == pytest.approx(0.0, abs=1e-6)
        assert result["x_opt"][0] == pytest.approx(0.0, abs=1e-4)

    def test_rosenbrock_2d(self) -> None:
        obj = _make_rosenbrock()
        result = run_nelder_mead(obj, x0=np.array([0.0, 0.0]), maxiter=10000)
        assert result["success"]
        assert result["fun_opt"] == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(result["x_opt"], [1.0, 1.0], atol=1e-4)

    def test_bounds_enforced(self) -> None:
        obj = _make_quadratic()
        # Start outside bounds - NM should be penalised back inside
        result = run_nelder_mead(
            obj,
            x0=np.array([5.0, 5.0]),
            bounds=(-2.0, 2.0),
            maxiter=5000,
        )
        assert np.all(np.abs(result["x_opt"]) <= 2.0 + 1e-4)

    def test_track_history(self) -> None:
        obj = _make_quadratic()
        result = run_nelder_mead(
            obj,
            x0=np.array([5.0]),
            maxiter=100,
            track_history=True,
        )
        assert len(result["history"]) > 0
        assert all(isinstance(v, float) for v in result["history"])

    def test_deterministic_with_seed(self) -> None:
        """Nelder-Mead is deterministic from same x0 (no seed dependence)."""
        obj = _make_quadratic()
        r1 = run_nelder_mead(obj, x0=np.array([3.0, 4.0]), maxiter=5000)
        r2 = run_nelder_mead(obj, x0=np.array([3.0, 4.0]), maxiter=5000)
        assert r1["fun_opt"] == pytest.approx(r2["fun_opt"])
        np.testing.assert_allclose(r1["x_opt"], r2["x_opt"])


# ---------------------------------------------------------------------------
# Two-phase pipeline (callback-based)
# ---------------------------------------------------------------------------


@dataclass
class _FakeRSResult:
    samples: np.ndarray
    delta_omega_values: np.ndarray


@dataclass
class _FakeNMResult:
    delta_omega_opt: float
    params_opt: np.ndarray


class TestRunTwoPhasePipeline:
    def test_basic_pipeline(self) -> None:
        obj = _make_quadratic()

        def rs_fn(n_samples: int, seed: int, **kw: Any) -> _FakeRSResult:
            samples, values = run_random_search(
                obj,
                n_params=2,
                n_samples=n_samples,
                seed=seed,
            )
            return _FakeRSResult(samples=samples, delta_omega_values=values)

        def nm_fn(x0: np.ndarray, seed: int, **kw: Any) -> _FakeNMResult:
            res = run_nelder_mead(obj, x0=x0, maxiter=1000)
            return _FakeNMResult(
                delta_omega_opt=res["fun_opt"],
                params_opt=res["x_opt"],
            )

        best, all_nm = run_two_phase_pipeline(
            rs_fn,
            nm_fn,
            config=TwoPhaseConfig(n_random=200, n_nm_refine=5, seed=42),
        )

        assert isinstance(best, _FakeNMResult)
        assert len(all_nm) == 5
        # Best result should be close to 0 (quadratic minimum)
        assert best.delta_omega_opt < 0.01

    def test_pipeline_deterministic(self) -> None:
        obj = _make_quadratic()

        def rs_fn(n_samples: int, seed: int, **kw: Any) -> _FakeRSResult:
            samples, values = run_random_search(
                obj,
                n_params=2,
                n_samples=n_samples,
                seed=seed,
            )
            return _FakeRSResult(samples=samples, delta_omega_values=values)

        def nm_fn(x0: np.ndarray, seed: int, **kw: Any) -> _FakeNMResult:
            res = run_nelder_mead(obj, x0=x0, maxiter=1000)
            return _FakeNMResult(
                delta_omega_opt=res["fun_opt"],
                params_opt=res["x_opt"],
            )

        cfg = TwoPhaseConfig(n_random=200, n_nm_refine=5, seed=42)
        best1, _ = run_two_phase_pipeline(rs_fn, nm_fn, cfg)
        best2, _ = run_two_phase_pipeline(rs_fn, nm_fn, cfg)
        assert best1.delta_omega_opt == pytest.approx(best2.delta_omega_opt)

    def test_kwargs_forwarded(self) -> None:
        """Verify that rs_kwargs and nm_kwargs are passed to the callbacks."""

        def rs_fn(n_samples: int, seed: int, **kw: Any) -> _FakeRSResult:
            assert kw.get("custom_arg") == "hello"
            return _FakeRSResult(
                samples=np.zeros((n_samples, 2)),
                delta_omega_values=np.full(n_samples, 1.0),
            )

        def nm_fn(x0: np.ndarray, seed: int, **kw: Any) -> _FakeNMResult:
            assert kw.get("custom_arg") == "hello"
            return _FakeNMResult(delta_omega_opt=0.5, params_opt=x0)

        best, _ = run_two_phase_pipeline(
            rs_fn,
            nm_fn,
            config=TwoPhaseConfig(n_random=10, n_nm_refine=2, seed=42),
            rs_kwargs={"custom_arg": "hello"},
            nm_kwargs={"custom_arg": "hello"},
        )
        assert best.delta_omega_opt == 0.5


# ---------------------------------------------------------------------------
# Omega scan wrapper
# ---------------------------------------------------------------------------


class TestRunOmegaScan:
    def test_basic_omega_scan(self) -> None:
        obj = _make_quadratic()

        def rs_fn(n_samples: int, seed: int, **kw: Any) -> _FakeRSResult:
            samples, values = run_random_search(
                obj,
                n_params=2,
                n_samples=n_samples,
                seed=seed,
            )
            return _FakeRSResult(samples=samples, delta_omega_values=values)

        def nm_fn(x0: np.ndarray, seed: int, **kw: Any) -> _FakeNMResult:
            # Verify omega kwarg is passed
            assert "omega" in kw or "omega_true" in kw
            res = run_nelder_mead(obj, x0=x0, maxiter=500)
            return _FakeNMResult(
                delta_omega_opt=res["fun_opt"],
                params_opt=res["x_opt"],
            )

        omega_vals = [0.1, 0.5, 1.0]
        best_results, all_results = run_omega_scan(
            omega_vals,
            rs_fn,
            nm_fn,
            config=TwoPhaseConfig(n_random=100, n_nm_refine=3, seed=42),
        )

        assert len(best_results) == len(omega_vals)
        assert len(all_results) == len(omega_vals)
        for i, _omega in enumerate(omega_vals):
            assert len(all_results[i]) == 3  # n_nm_refine
            assert best_results[i].delta_omega_opt < 0.1

    def test_custom_omega_keys(self) -> None:
        collected: list[float] = []

        def rs_fn(n_samples: int, seed: int, **kw: Any) -> _FakeRSResult:
            collected.append(kw.get("my_omega", -1.0))
            return _FakeRSResult(
                samples=np.zeros((n_samples, 2)),
                delta_omega_values=np.full(n_samples, 1.0),
            )

        def nm_fn(x0: np.ndarray, seed: int, **kw: Any) -> _FakeNMResult:
            return _FakeNMResult(delta_omega_opt=0.5, params_opt=x0)

        run_omega_scan(
            [0.1, 0.5],
            rs_fn,
            nm_fn,
            config=TwoPhaseConfig(n_random=5, n_nm_refine=1, seed=42),
            rs_omega_key="my_omega",
        )
        assert collected == pytest.approx([0.1, 0.5])

    def test_deterministic_omega_scan(self) -> None:
        obj = _make_quadratic()

        def rs_fn(n_samples: int, seed: int, **kw: Any) -> _FakeRSResult:
            samples, values = run_random_search(
                obj,
                n_params=2,
                n_samples=n_samples,
                seed=seed,
            )
            return _FakeRSResult(samples=samples, delta_omega_values=values)

        def nm_fn(x0: np.ndarray, seed: int, **kw: Any) -> _FakeNMResult:
            res = run_nelder_mead(obj, x0=x0, maxiter=500)
            return _FakeNMResult(
                delta_omega_opt=res["fun_opt"],
                params_opt=res["x_opt"],
            )

        cfg = TwoPhaseConfig(n_random=100, n_nm_refine=3, seed=42)
        b1, _ = run_omega_scan([0.1, 0.5], rs_fn, nm_fn, cfg)
        b2, _ = run_omega_scan([0.1, 0.5], rs_fn, nm_fn, cfg)
        for r1, r2 in zip(b1, b2, strict=False):
            assert r1.delta_omega_opt == pytest.approx(r2.delta_omega_opt)


# ---------------------------------------------------------------------------
# TwoPhaseConfig
# ---------------------------------------------------------------------------


class TestTwoPhaseConfig:
    def test_n_params_from_tuple_bounds(self) -> None:
        cfg = TwoPhaseConfig(bounds=(-5.0, 5.0))
        # tuple bounds default: n_params = 4
        assert cfg.n_params == 4

    def test_n_params_from_list_bounds(self) -> None:
        cfg = TwoPhaseConfig(bounds=[(-5.0, 5.0), (0.0, 1.0), (-1.0, 1.0)])
        assert cfg.n_params == 3

    def test_default_seed_is_42(self) -> None:
        cfg = TwoPhaseConfig()
        assert cfg.seed == 42

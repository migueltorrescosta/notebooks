"""Unit tests for the optimization physics module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from .optimization import (
    MINIMIZERS,
    TEST_FUNCTIONS,
    generate_surface,
    himmelblau,
    normalize_benchmark_results,
    rastrigin,
    rosenbrock,
    sphere,
)


class TestTestFunctions:
    def test_given_sphere_then_minimum_at_origin(self) -> None:
        assert sphere(0.0, 0.0) == pytest.approx(0.0)

    def test_given_rastrigin_then_minimum_at_origin(self) -> None:
        assert rastrigin(0.0, 0.0) == pytest.approx(0.0)

    def test_given_rosenbrock_then_minimum_at_ones(self) -> None:
        assert rosenbrock(1.0, 1.0) == pytest.approx(0.0, abs=0.01)

    def test_given_himmelblau_then_values_are_finite(self) -> None:
        val1 = himmelblau(3.0, 2.0)
        val2 = himmelblau(-3.0, -3.0)
        assert np.isfinite(val1)
        assert np.isfinite(val2)


class TestFunctionDictionary:
    @pytest.mark.parametrize(
        ("name", "func"),
        list(TEST_FUNCTIONS.items()),
        ids=list(TEST_FUNCTIONS.keys()),
    )
    def test_given_function_then_is_callable(
        self, name: str, func: Callable[[float, float], float]
    ) -> None:
        assert callable(func)

    @pytest.mark.parametrize(
        ("name", "func"),
        list(TEST_FUNCTIONS.items()),
        ids=list(TEST_FUNCTIONS.keys()),
    )
    def test_given_function_then_returns_finite_for_scalar(
        self, name: str, func: Callable[[float, float], float]
    ) -> None:
        result = func(0.0, 1.0)
        assert np.isfinite(result)


class TestMinimizers:
    @pytest.mark.parametrize(
        ("name", "minimizer"),
        list(MINIMIZERS.items()),
        ids=list(MINIMIZERS.keys()),
    )
    def test_given_minimizer_then_is_callable(
        self, name: str, minimizer: Callable[..., dict[str, Any]]
    ) -> None:
        assert callable(minimizer)

    def test_given_nelder_mead_then_finds_sphere_minimum(self) -> None:
        minimizer = MINIMIZERS["Nelder-Mead"]
        result = minimizer(sphere, np.array([2.0, 2.0]))
        assert np.linalg.norm(result["x"]) < 0.1


class TestSurfaceGeneration:
    def test_given_grid_then_surface_has_correct_shape(self) -> None:
        surf = generate_surface("Sphere", (-3, 3), (-3, 3), 11)
        assert surf.shape == (11, 11)

    def test_given_sphere_then_surface_minimum_near_center(self) -> None:
        surf = generate_surface("Sphere", (-3, 3), (-3, 3), 101)
        center = surf.shape[0] // 2
        assert surf[center, center] < 0.1


class TestNormalization:
    def test_given_benchmark_results_then_values_in_unit_interval(self) -> None:
        results = [
            {"function": "Sphere", "minimizer": "BFGS", "minimum": 0.0},
            {"function": "Sphere", "minimizer": "Nelder-Mead", "minimum": 1.0},
            {"function": "Rastrigin", "minimizer": "BFGS", "minimum": 0.0},
            {"function": "Rastrigin", "minimizer": "Nelder-Mead", "minimum": 10.0},
        ]
        values = normalize_benchmark_results(results)
        assert np.all(values >= 0)
        assert np.all(values <= 1)

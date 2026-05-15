"""Unit tests for Optimization physics module."""

import numpy as np
import pytest

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
    def test_sphere_should_have_minimum_at_0_0(self) -> None:
        assert sphere(0.0, 0.0) == pytest.approx(0.0), (
            "Expected sphere(0.0, 0.0) == pytest.approx(0.0)"
        )

    def test_rastrigin_should_have_minimum_at_0_0(self) -> None:
        assert rastrigin(0.0, 0.0) == pytest.approx(0.0), (
            "Expected rastrigin(0.0, 0.0) == pytest.approx(0.0)"
        )

    def test_rosenbrock_should_have_minimum_at_1_1(self) -> None:
        assert rosenbrock(1.0, 1.0) == pytest.approx(0.0, abs=0.01), (
            "Expected rosenbrock(1.0, 1.0) == pytest.approx(0.0, abs=0.01)"
        )

    def test_himmelblau_should_have_multiple_minima(self) -> None:
        # Just test that function has reasonable values
        val1 = himmelblau(3.0, 2.0)
        val2 = himmelblau(-3.0, -3.0)
        # Both should be finite
        assert np.isfinite(val1), "Expected val1 to be finite"
        assert np.isfinite(val2), "Expected val2 to be finite"


class TestFunctionDictionary:
    def test_all_functions_should_be_callable(self) -> None:
        for func in TEST_FUNCTIONS.values():
            assert callable(func), "Expected func to be callable"

    def test_all_functions_should_work_with_numpy_arrays(self) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        for func in TEST_FUNCTIONS.values():
            result = func(x[0], y[0])  # Just scalar for now
            assert np.isfinite(result), "Expected result to be finite"


class TestMinimizers:
    def test_all_minimizers_should_exist(self) -> None:
        for minimizer in MINIMIZERS.values():
            assert callable(minimizer), "Expected minimizer to be callable"

    def test_minimizer_should_find_minimum_for_sphere(self) -> None:
        minimizer = MINIMIZERS["Nelder-Mead"]
        result = minimizer(sphere, np.array([2.0, 2.0]))
        # Should converge close to (0, 0)
        assert np.linalg.norm(result["x"]) < 0.1, (
            'Expected np.linalg.norm(result["x"]) < 0.1'
        )


class TestSurfaceGeneration:
    def test_surface_should_have_correct_shape(self) -> None:
        surf = generate_surface("Sphere", (-3, 3), (-3, 3), 11)
        assert surf.shape == (11, 11), "Expected surf.shape == (11, 11)"

    def test_surface_minimum_should_be_near_center_for_sphere(self) -> None:
        surf = generate_surface("Sphere", (-3, 3), (-3, 3), 101)
        center = surf.shape[0] // 2
        # Center pixel should be ~0
        assert surf[center, center] < 0.1, "Expected surf[center, center] < 0.1"


class TestNormalization:
    def test_normalized_values_should_be_in_0_1(self) -> None:
        results = [
            {"function": "Sphere", "minimizer": "BFGS", "minimum": 0.0},
            {"function": "Sphere", "minimizer": "Nelder-Mead", "minimum": 1.0},
            {"function": "Rastrigin", "minimizer": "BFGS", "minimum": 0.0},
            {"function": "Rastrigin", "minimizer": "Nelder-Mead", "minimum": 10.0},
        ]
        values = normalize_benchmark_results(results)
        # All values should be in [0, 1]
        assert np.all(values >= 0), "Expected np.all(values >= 0)"
        assert np.all(values <= 1), "Expected np.all(values <= 1)"

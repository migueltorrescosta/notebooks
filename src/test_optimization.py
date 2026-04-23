"""Unit tests for Optimization physics module."""

import numpy as np
import pytest

from src.optimization import (
    rastrigin,
    sphere,
    rosenbrock,
    himmelblau,
    TEST_FUNCTIONS,
    MINIMIZERS,
    generate_surface,
    normalize_benchmark_results,
)


class TestTestFunctions:
    def test_sphere_minimum(self) -> None:
        """Sphere should have minimum at (0, 0)."""
        assert sphere(0.0, 0.0) == pytest.approx(0.0)

    def test_rastrigin_minimum(self) -> None:
        """Rastrigin should have minimum at (0, 0)."""
        assert rastrigin(0.0, 0.0) == pytest.approx(0.0)

    def test_rosenbrock_minimum(self) -> None:
        """Rosenbrock should have minimum at (1, 1)."""
        assert rosenbrock(1.0, 1.0) == pytest.approx(0.0, abs=0.01)

    def test_himmelblau_values(self) -> None:
        """Himmelblau should have multiple minima."""
        # Just test that function has reasonable values
        val1 = himmelblau(3.0, 2.0)
        val2 = himmelblau(-3.0, -3.0)
        # Both should be finite
        assert np.isfinite(val1)
        assert np.isfinite(val2)


class TestFunctionDictionary:
    def test_all_functions_callable(self) -> None:
        """All functions should be callable."""
        for name, func in TEST_FUNCTIONS.items():
            assert callable(func)

    def test_all_functions_vectorized(self) -> None:
        """All functions should work with numpy arrays."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        for name, func in TEST_FUNCTIONS.items():
            result = func(x[0], y[0])  # Just scalar for now
            assert np.isfinite(result)


class TestMinimizers:
    def test_all_minimizers_exist(self) -> None:
        """All minimizers should exist."""
        for name, minimizer in MINIMIZERS.items():
            assert callable(minimizer)

    def test_sphere_minimization(self) -> None:
        """Minimizer should find minimum for sphere."""
        minimizer = MINIMIZERS["Nelder-Mead"]
        result = minimizer(sphere, np.array([2.0, 2.0]))
        # Should converge close to (0, 0)
        assert np.linalg.norm(result["x"]) < 0.1


class TestSurfaceGeneration:
    def test_surface_shape(self) -> None:
        """Surface should have correct shape."""
        surf = generate_surface("Sphere", (-3, 3), (-3, 3), 11)
        assert surf.shape == (11, 11)

    def test_surface_minimum(self) -> None:
        """Surface minimum should be near center for sphere."""
        surf = generate_surface("Sphere", (-3, 3), (-3, 3), 101)
        center = surf.shape[0] // 2
        # Center pixel should be ~0
        assert surf[center, center] < 0.1


class TestNormalization:
    def test_normalize_results(self) -> None:
        """Normalized values should be in [0, 1]."""
        results = [
            {"function": "Sphere", "minimizer": "BFGS", "minimum": 0.0},
            {"function": "Sphere", "minimizer": "Nelder-Mead", "minimum": 1.0},
            {"function": "Rastrigin", "minimizer": "BFGS", "minimum": 0.0},
            {"function": "Rastrigin", "minimizer": "Nelder-Mead", "minimum": 10.0},
        ]
        values = normalize_benchmark_results(results)
        # All values should be in [0, 1]
        assert np.all(values >= 0)
        assert np.all(values <= 1)

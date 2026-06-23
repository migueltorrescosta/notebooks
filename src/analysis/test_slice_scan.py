"""Tests for :mod:`src.analysis.slice_scan`."""

from __future__ import annotations

import numpy as np

from src.analysis.slice_scan import (
    _resolve_workers,
    parallel_grid_scan,
    sequential_grid_scan,
)

# ---------------------------------------------------------------------------
# _resolve_workers
# ---------------------------------------------------------------------------


class TestResolveWorkers:
    def test_default_returns_at_least_one(self) -> None:
        assert _resolve_workers(None) >= 1

    def test_negative_one_returns_at_least_one(self) -> None:
        assert _resolve_workers(-1) >= 1

    def test_specific_value(self) -> None:
        assert _resolve_workers(4) == 4

    def test_clamps_to_one(self) -> None:
        assert _resolve_workers(0) == 1
        assert _resolve_workers(-5) == 1


# ---------------------------------------------------------------------------
# sequential_grid_scan
# ---------------------------------------------------------------------------


class TestSequentialGridScan:
    def test_returns_correct_shape(self) -> None:
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 20)
        grid = sequential_grid_scan(x, y, lambda a, b: a + b)
        assert grid.shape == (10, 20)

    def test_known_values(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0])
        grid = sequential_grid_scan(x, y, lambda a, b: a * b)
        expected = np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]])
        np.testing.assert_allclose(grid, expected)

    def test_single_point(self) -> None:
        x = np.array([3.0])
        y = np.array([5.0])
        grid = sequential_grid_scan(x, y, lambda a, b: a + b)
        assert grid.shape == (1, 1)
        assert grid[0, 0] == 8.0

    def test_handles_inf_sensitivity(self) -> None:
        def _inf_at_origin(x: float, y: float) -> float:
            return np.inf if x == 0.0 and y == 0.0 else x + y

        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        grid = sequential_grid_scan(x, y, _inf_at_origin)
        assert grid[0, 0] == np.inf
        assert np.isfinite(grid[1, 1])

    def test_deterministic(self) -> None:
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 30)
        g1 = sequential_grid_scan(x, y, lambda a, b: a**2 + b**2)
        g2 = sequential_grid_scan(x, y, lambda a, b: a**2 + b**2)
        np.testing.assert_array_equal(g1, g2)


# ---------------------------------------------------------------------------
# parallel_grid_scan
# ---------------------------------------------------------------------------


def _add_worker(args: tuple) -> tuple[int, np.ndarray]:
    """Module-level worker: returns (start_idx, grid) for x + y."""
    x_chunk, y_vals, start_idx, _kw = args
    n_x = len(x_chunk)
    n_y = len(y_vals)
    grid = np.full((n_x, n_y), np.inf, dtype=float)
    for i in range(n_x):
        for j in range(n_y):
            grid[i, j] = x_chunk[i] + y_vals[j]
    return start_idx, grid


def _quadratic_worker(args: tuple) -> tuple[int, np.ndarray]:
    """Module-level worker: returns (start_idx, grid) for x^2 + y^2."""
    x_chunk, y_vals, start_idx, _kw = args
    n_x = len(x_chunk)
    n_y = len(y_vals)
    grid = np.full((n_x, n_y), np.inf, dtype=float)
    for i in range(n_x):
        for j in range(n_y):
            grid[i, j] = x_chunk[i] ** 2 + y_vals[j] ** 2
    return start_idx, grid


def _inf_worker(args: tuple) -> tuple[int, np.ndarray]:
    """Worker that returns inf at origin."""
    x_chunk, y_vals, start_idx, _kw = args
    n_x = len(x_chunk)
    n_y = len(y_vals)
    grid = np.full((n_x, n_y), np.inf, dtype=float)
    for i in range(n_x):
        for j in range(n_y):
            if x_chunk[i] == 0.0 and y_vals[j] == 0.0:
                grid[i, j] = np.inf
            else:
                grid[i, j] = x_chunk[i] + y_vals[j]
    return start_idx, grid


class TestParallelGridScan:
    def test_returns_correct_shape(self) -> None:
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 20)
        grid = parallel_grid_scan(x, y, _add_worker, n_jobs=2)
        assert grid.shape == (10, 20)

    def test_sequential_path_single_worker(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0])
        grid = parallel_grid_scan(x, y, _add_worker, n_jobs=1)
        expected = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]])
        np.testing.assert_allclose(grid, expected)

    def test_parallel_path_matches_sequential(self) -> None:
        x = np.linspace(-2, 2, 51)
        y = np.linspace(-2, 2, 61)
        seq = sequential_grid_scan(x, y, lambda a, b: a**2 + b**2)
        par = parallel_grid_scan(x, y, _quadratic_worker, n_jobs=4)
        np.testing.assert_allclose(par, seq)

    def test_single_point(self) -> None:
        x = np.array([3.0])
        y = np.array([5.0])
        grid = parallel_grid_scan(x, y, _add_worker, n_jobs=1)
        assert grid.shape == (1, 1)
        assert grid[0, 0] == 8.0

    def test_handles_inf_sensitivity(self) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        grid = parallel_grid_scan(x, y, _inf_worker, n_jobs=2)
        assert grid[0, 0] == np.inf
        assert np.isfinite(grid[1, 1])

    def test_deterministic(self) -> None:
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 31)
        g1 = parallel_grid_scan(x, y, _add_worker, n_jobs=2)
        g2 = parallel_grid_scan(x, y, _add_worker, n_jobs=2)
        np.testing.assert_array_equal(g1, g2)

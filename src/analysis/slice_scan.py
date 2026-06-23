"""
Generic 2D grid-scan utilities for sensitivity landscapes.

Provides shared infrastructure for the common pattern seen across 5+ report
report experiment modules (was ``local.py``):

    1. Create ``(n_x, n_y)`` linspace arrays for two parameters.
    2. Evaluate a sensitivity function over the full 2D grid.
    3. Optionally parallelise across x-axis chunks via
       ``concurrent.futures.ProcessPoolExecutor``.

Usage::

    from src.analysis.slice_scan import parallel_grid_scan, sequential_grid_scan

    # Sequential evaluation (simple nested loop)
    def _my_sensitivity(x, y):
        return compute_foo(x, y, omega=0.5, t_hold=1.0)

    grid = sequential_grid_scan(x_vals, y_vals, _my_sensitivity)

    # Parallel evaluation (module-level worker for pickling)
    def _my_worker(args):
        x_chunk, y_vals, start_idx, kw = args
        grid = np.full((len(x_chunk), len(y_vals)), np.inf)
        for i, x in enumerate(x_chunk):
            for j, y in enumerate(y_vals):
                grid[i, j] = compute_foo(x, y, **kw)
        return start_idx, grid

    grid = parallel_grid_scan(x_vals, y_vals, _my_worker, n_jobs=4, omega=0.5)
"""

from __future__ import annotations

import concurrent.futures
import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def sequential_grid_scan(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    sensitivity_fn: Callable[[float, float], float],
) -> np.ndarray:
    """Evaluate *sensitivity_fn* over every ``(x, y)`` point sequentially.

    Args:
        x_vals: 1D array of x-axis values.
        y_vals: 1D array of y-axis values.
        sensitivity_fn: Callable ``(x, y) -> float`` returning the sensitivity
            (typically :math:`\\Delta\\omega`) at each grid point.

    Returns:
        2D array of shape ``(len(x_vals), len(y_vals))``.
    """
    n_x = len(x_vals)
    n_y = len(y_vals)
    grid = np.full((n_x, n_y), np.inf, dtype=float)
    for i in range(n_x):
        for j in range(n_y):
            grid[i, j] = sensitivity_fn(x_vals[i], y_vals[j])
    return grid


def parallel_grid_scan(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    worker_fn: Callable[[tuple], tuple[int, np.ndarray]],
    n_jobs: int | None = None,
    **fixed_kwargs: Any,
) -> np.ndarray:
    """Evaluate a 2D grid in parallel using ``ProcessPoolExecutor``.

    Chunks *x_vals* across multiple worker processes. Each worker receives
    ``(x_chunk, y_vals, start_idx, fixed_kwargs_dict)`` and returns
    ``(start_idx, chunk_grid)``.

    Args:
        x_vals: 1D array of x-axis values (split across workers).
        y_vals: 1D array of y-axis values (each worker iterates fully).
        worker_fn: Module-level (picklable) callable accepting a single
            ``tuple`` argument of the form
            ``(x_chunk, y_vals, start_idx, fixed_kwargs_dict)``
            and returning ``(start_idx, chunk_grid)``.
        n_jobs: Number of parallel workers. ``None`` or ``-1`` = all CPUs.
            ``1`` = sequential (but still uses the worker function).
        **fixed_kwargs: Additional keyword arguments forwarded to each
            worker inside ``fixed_kwargs_dict``.

    Returns:
        2D array of shape ``(len(x_vals), len(y_vals))``.
    """
    n_x = len(x_vals)
    n_y = len(y_vals)
    n_workers = _resolve_workers(n_jobs)

    indices = np.arange(n_x)
    chunks = np.array_split(indices, n_workers)
    worker_args_list: list[tuple] = [
        (x_vals[chunk], y_vals, int(chunk[0]), fixed_kwargs) for chunk in chunks
    ]

    grid = np.full((n_x, n_y), np.inf, dtype=float)

    if n_workers == 1:
        # Sequential path via single worker call
        _start_idx, chunk_grid = worker_fn(worker_args_list[0])
        grid = chunk_grid
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
        ) as executor:
            futures = {
                executor.submit(worker_fn, args): args for args in worker_args_list
            }
            for future in concurrent.futures.as_completed(futures):
                start_idx, chunk_grid = future.result()
                n_chunk = chunk_grid.shape[0]
                grid[start_idx : start_idx + n_chunk, :] = chunk_grid

    return grid


def _resolve_workers(n_jobs: int | None) -> int:
    """Resolve the number of worker processes."""
    if n_jobs is None or n_jobs == -1:
        return max(1, os.cpu_count() or 4)
    return max(1, n_jobs)

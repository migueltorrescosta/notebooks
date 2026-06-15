"""Tests for the parallel execution utility."""

from __future__ import annotations

from src.utils.parallel import parallel_map


def square(x: int) -> int:
    return x * x


def test_parallel_map_returns_results_in_order() -> None:
    result = parallel_map(square, [1, 2, 3, 4, 5], max_workers=2)
    assert result == [1, 4, 9, 16, 25]


def test_parallel_map_empty_list() -> None:
    result = parallel_map(square, [], max_workers=1)
    assert result == []


def test_parallel_map_single_item() -> None:
    result = parallel_map(square, [7], max_workers=1)
    assert result == [49]

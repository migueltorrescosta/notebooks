"""
Parallel execution utilities for quantum metrology simulations.

Provides a generic ``parallel_map`` function for distributing independent
workloads across a process pool, used by multiple report pipelines.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as _mp
import os
from collections.abc import Callable


def parallel_map(
    worker_fn: Callable,
    items: list,
    desc: str = "Processing",
    max_workers: int | None = None,
) -> list:
    """Run worker_fn(item) for each item in parallel via process pool.

    Args:
        worker_fn: Callable taking a single item argument.
        items: Iterable of items to process.
        desc: Short description for progress logging.
        max_workers: Number of subprocess workers (default: CPU count).

    Returns:
        List of results in the same order as items.
    """
    if max_workers is None:
        max_workers = min(32, os.cpu_count() or 1)
    item_list = list(items)
    print(f"  [parallel] {desc}: {len(item_list)} items, {max_workers} workers")

    mp_ctx = _mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    ) as executor:
        fut_to_idx = {
            executor.submit(worker_fn, item): i for i, item in enumerate(item_list)
        }
        results: list = [None] * len(item_list)
        for future in concurrent.futures.as_completed(fut_to_idx):
            idx = fut_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                print(f"  [ERROR] item={item_list[idx]}: {exc}")
                raise
    return results

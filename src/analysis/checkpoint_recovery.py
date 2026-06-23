"""
Generic checkpoint recovery utilities for N-scaling scans.

Provides shared infrastructure for the common pattern seen across 5+ report
report experiment modules (was ``local.py``):

    1. Iterate over ``N_*.parquet`` checkpoint files in a directory.
    2. Read each checkpoint and build result objects from rows with finite
       ``delta_omega_opt`` values, tracking which keys are completed.
    3. Group pending items by a caller-defined key function.
    4. Dispatch each group via :func:`src.utils.parallel.parallel_map`.
    5. Collect results, skip non-finite sensitivities, save per-group
       checkpoint Parquet files.

Usage (callback-based)::

    from src.analysis.checkpoint_recovery import load_checkpoints, run_pending_groups

    # Define how to build a result from a Parquet row
    def _build_from_row(row):
        return MyResult(N=int(row["N"]), omega=float(row["omega"]), ...)

    completed, results = load_checkpoints(
        checkpoint_dir=Path("raw_data/checkpoints"),
        build_result_from_row=_build_from_row,
        key_columns=["N", "omega"],
    )

    # Define how to group pending items and name checkpoints
    def _group_key(item):
        N, omega = item
        return N

    def _ckpt_name(key):
        return checkpoint_dir / f"N_{key:03d}.parquet"

    new_results = run_pending_groups(
        items=pending_items,
        checkpoint_dir=checkpoint_dir,
        worker_fn=_my_worker,
        build_result_from_dict=_build_from_dict,
        group_key_fn=_group_key,
        checkpoint_name_fn=_ckpt_name,
        scan_result_cls=MyScanResult,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.utils.parallel import parallel_map

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    from pathlib import Path


def load_checkpoints(
    checkpoint_dir: Path,
    build_result_from_row: Callable[..., Any],
    key_columns: list[str],
) -> tuple[set[tuple[Any, ...]], list[Any]]:
    """Load existing checkpoints from ``N_*.parquet`` files.

    Scans *checkpoint_dir* for ``N_*.parquet`` files, reads each row,
    and if the row has a finite ``delta_omega_opt`` value, records it as
    completed and builds a result object via *build_result_from_row*.

    Args:
        checkpoint_dir: Directory containing ``N_*.parquet`` checkpoint files.
        build_result_from_row: Callable ``row_dict -> result`` that constructs a
            result dataclass from a dictionary of column values.
            The dictionary is ``dict(name=value, ...)`` from
            ``df.iloc[idx].to_dict()``.
        key_columns: Column names that uniquely identify a completed run
            (e.g. ``["N", "omega"]`` or ``["N", "M", "omega"]``).

    Returns:
        Tuple ``(completed_set, results_list)`` where *completed_set* contains
        ``tuple(row[col] for col in key_columns)`` for each finite result,
        and *results_list* contains the corresponding result objects.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[Any, ...]] = set()
    checkpoint_results: list[Any] = []

    for ckpt_file in sorted(checkpoint_dir.glob("N_*.parquet")):
        try:
            df_ckpt = pd.read_parquet(ckpt_file)
            # Fail fast if required columns are missing
            _check_required_columns(df_ckpt, key_columns, ckpt_file)

            for _, row in df_ckpt.iterrows():
                delta = float(row.get("delta_omega_opt", np.inf))
                if not np.isfinite(delta):
                    continue
                key = tuple(row[col] for col in key_columns)
                # Normalise integer types for hashable keys
                key = tuple(
                    int(v)
                    if isinstance(v, (np.integer, int))
                    else float(v)
                    if isinstance(v, (np.floating, float))
                    else v
                    for v in key
                )
                completed.add(key)
                checkpoint_results.append(build_result_from_row(row.to_dict()))
        except (
            pd.errors.EmptyDataError,
            FileNotFoundError,
            KeyError,
            ValueError,
        ) as exc:
            print(f"  [warn] Could not load checkpoint {ckpt_file}: {exc}")

    return completed, checkpoint_results


def _check_required_columns(
    df: pd.DataFrame,
    key_columns: list[str],
    ckpt_file: Path,
) -> None:
    """Verify that all required columns exist in the checkpoint DataFrame."""
    required = set(key_columns) | {"delta_omega_opt"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Checkpoint {ckpt_file} missing columns: {missing}. "
            "Re-run the simulation that generated it."
        )


def run_pending_groups(
    items_to_run: list[Any],
    checkpoint_dir: Path,
    worker_fn: Callable[[Any], dict[str, Any]],
    build_result_from_dict: Callable[[dict[str, Any]], Any],
    group_key_fn: Callable[[Any], Hashable],
    checkpoint_name_fn: Callable[[Hashable], Path],
    scan_result_cls: type,
) -> list[Any]:
    """Run pending items grouped by key, saving per-group checkpoints.

    Groups *items_to_run* by ``group_key_fn(item)``. For each group:
    if the checkpoint file exists (via ``checkpoint_name_fn(key)``), skip it.
    Otherwise, dispatch the group via :func:`~src.utils.parallel.parallel_map`
    using *worker_fn*, collect results, skip those with non-finite
    ``delta_omega_opt``, build result objects via *build_result_from_dict*,
    save the checkpoint via *scan_result_cls*, and extend the results list.

    Args:
        items_to_run: List of items to process (e.g. ``(N, omega)`` tuples).
        checkpoint_dir: Directory for saving ``N_*.parquet`` checkpoint files.
        worker_fn: Callable taking a single item, returning a dict with at
            least ``"delta_omega_opt"`` and all fields needed by
            *build_result_from_dict*.
        build_result_from_dict: Callable ``rdict -> result`` that constructs
            a result dataclass from the dict returned by *worker_fn*.
        group_key_fn: Callable ``item -> hashable key`` for grouping
            (typically extracts ``N`` or ``(N, N_A)``).
        checkpoint_name_fn: Callable ``key -> Path`` for the checkpoint file
            to save (e.g. ``lambda N: dir / f"N_{N:03d}.parquet"``).
        scan_result_cls: Dataclass type that wraps a list of result objects
            and has a ``save_parquet(path)`` method.

    Returns:
        List of newly completed result objects.
    """
    # Group by key
    by_key: dict[Hashable, list[Any]] = {}
    for item in items_to_run:
        key = group_key_fn(item)
        by_key.setdefault(key, []).append(item)

    checkpoint_results: list[Any] = []

    keys: list[Hashable] = list(by_key.keys())
    for key in sorted(keys, key=_safe_sort_key):
        group_items = by_key[key]
        ckpt_path = checkpoint_name_fn(key)
        if ckpt_path.exists():
            print(f"  [ckpt] {ckpt_path.name} already done, skipping")
            continue

        print(
            f"  [batch] {ckpt_path.stem}: {len(group_items)} items (parallel)",
        )
        batch_results = parallel_map(
            worker_fn,
            group_items,
            desc=str(ckpt_path.stem),
        )

        ckpt_list: list[Any] = []
        for rdict in batch_results:
            delta = rdict.get("delta_omega_opt", np.inf)
            if not np.isfinite(delta):
                continue
            ckpt_list.append(build_result_from_dict(rdict))

        if ckpt_list:
            ckpt_list.sort(key=_omega_sort_key)
            scan_result = scan_result_cls(results=ckpt_list)
            scan_result.save_parquet(ckpt_path)
            checkpoint_results.extend(ckpt_list)
            print(f"    [ckpt] saved {ckpt_path.name}")

    return checkpoint_results


def _safe_sort_key(key: Hashable) -> tuple:
    """Convert a hashable key to a sortable tuple.

    Handles single values and tuples of mixed types gracefully.
    """
    if isinstance(key, tuple):
        return tuple(
            (0, v) if isinstance(v, (int, np.integer)) else (1, v) for v in key
        )
    return ((0, key) if isinstance(key, (int, np.integer)) else (1, key),)


def _omega_sort_key(result: Any) -> float:
    """Extract omega from a result for sorting checkpoints.

    Tries ``omega``, then ``omega_value``, then falls back to 0.0.
    """
    if hasattr(result, "omega"):
        return float(result.omega)
    if hasattr(result, "omega_value"):
        return float(result.omega_value)
    return 0.0

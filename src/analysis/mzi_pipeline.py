r"""Shared MZI data-pipeline functions for report modules.

Provides helpers for the common pattern of:

- Generating a single-resource scan with :mod:`safe_generate_scan`
- Iterating over resource values (:func:`generate_full_data`)
- Concatenating scan results (:func:`concatenate_scan_results`)
- Caching results as Parquet (:func:`maybe_generate_full_data`)

All functions operate on :class:`~src.analysis.sensitivity_metrics.MziSensitivityData`
(or its subclasses).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from src.analysis.sensitivity_metrics import MziSensitivityData

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T", bound=MziSensitivityData)


def safe_generate_scan[T: MziSensitivityData](
    generator_fn: Callable[..., T | None],
    *args: Any,
    fail_label: str = "",
    **kwargs: Any,
) -> T | None:
    """Run *generator_fn* and return its result, or ``None`` on failure.

    Catches :exc:`ValueError` and :exc:`AssertionError`, logs a warning
    with the optional *fail_label*, and returns ``None``.

    Args:
        generator_fn: Callable that returns a data object or ``None``.
        *args: Forwarded to *generator_fn*.
        fail_label: Human-readable label for warning messages.
        **kwargs: Forwarded to *generator_fn*.

    Returns:
        Data object or ``None`` on failure.
    """
    try:
        return generator_fn(*args, **kwargs)
    except (ValueError, AssertionError) as exc:
        prefix = f" [{fail_label}]" if fail_label else ""
        print(f"Warning{prefix}: {exc}")
        return None


def collect_metadata_per_r[T: MziSensitivityData](
    scan_results: list[T],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract ``truncation_M`` and ``squeezing_q`` arrays from scan results.

    Args:
        scan_results: List of single-resource scan results.

    Returns:
        Tuple ``(truncation_Ms, squeezing_qs)`` where the second element
        is ``None`` if every result has ``squeezing_q_per_R is None``.
    """
    trunc_Ms: list[float] = []
    sq_qs: list[float | None] = []
    has_sq_q = False
    for r in scan_results:
        trunc_val = (
            float(r.truncation_M_per_R[0])
            if r.truncation_M_per_R is not None
            else float("nan")
        )
        sq_val: float | None = (
            float(r.squeezing_q_per_R[0]) if r.squeezing_q_per_R is not None else None
        )
        trunc_Ms.append(trunc_val)
        sq_qs.append(sq_val)
        if sq_val is not None:
            has_sq_q = True

    sq_array: np.ndarray | None = (
        np.array([v if v is not None else float("nan") for v in sq_qs], dtype=float)
        if has_sq_q
        else None
    )
    return np.array(trunc_Ms, dtype=float), sq_array


def concatenate_scan_results[T: MziSensitivityData](
    scan_results: list[T],
    state_type: str,
    t_hold: float,
) -> T:
    """Concatenate a list of single-resource scan results.

    Args:
        scan_results: List of single-resource scan results (each with one
            resource value). At least one entry is required.
        state_type: State-type label for the combined result.
        t_hold: Holding time.

    Returns:
        Combined data object of the same type as the inputs.
    """
    resource_type = scan_results[0].resource_type
    omega_values = scan_results[0].omega_values
    trunc_Ms, sq_qs = collect_metadata_per_r(scan_results)

    def _cat(field: str) -> np.ndarray:
        return np.concatenate([getattr(r, field) for r in scan_results])

    cls = type(scan_results[0])
    return cls(
        state_type=state_type,
        resource_type=resource_type,
        resource_values=_cat("resource_values"),
        omega_values=omega_values,
        expectation_grid=_cat("expectation_grid"),
        variance_grid=_cat("variance_grid"),
        derivative_grid=_cat("derivative_grid"),
        delta_omega_ep_grid=_cat("delta_omega_ep_grid"),
        delta_omega_q_per_R=_cat("delta_omega_q_per_R"),
        fisher_classical_grid=_cat("fisher_classical_grid"),
        delta_omega_c_grid=_cat("delta_omega_c_grid"),
        t_hold=t_hold,
        truncation_M_per_R=trunc_Ms,
        squeezing_q_per_R=sq_qs,
    )


def generate_full_data[T: MziSensitivityData](
    state_type: str,
    resource_range: list[float] | list[int],
    omega_grid: np.ndarray,
    generator_fn: Callable[..., T | None],
    t_hold: float = 10.0,
) -> T:
    r"""Generate sensitivity data for all resource values in a range.

    Args:
        state_type: State type key (e.g. ``"sv"``, ``"oat"``).
        resource_range: List of resource parameter values.
        omega_grid: :math:`\omega` values to scan.
        generator_fn: Callable that accepts a float resource value
            and returns a single-resource data object or ``None``.
        t_hold: Holding time.

    Returns:
        Combined data object with all resource values.

    Raises:
        RuntimeError: If no valid resource values were generated.
    """
    scan_results: list[T] = []
    for idx, R in enumerate(resource_range):
        print(
            f"  Sweeping {state_type} R={R} ({idx + 1}/{len(resource_range)})...",
            flush=True,
        )
        scan = safe_generate_scan(
            generator_fn,
            R,
            omega_grid,
            fail_label=f"{state_type} R={R}",
            t_hold=t_hold,
        )
        if scan is not None:
            scan_results.append(scan)

    if not scan_results:
        raise RuntimeError(f"No valid resource values for state_type={state_type}")

    return concatenate_scan_results(scan_results, state_type, t_hold)




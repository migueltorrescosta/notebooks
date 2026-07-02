"""
Generic 2D (N, ω) coupling-sweep orchestration for XX-coupling reports.

Provides shared infrastructure for the common pattern seen across three
multi-particle XX-coupling reports (20260522, 20260523, 20260525):

    1. Resolve default ω and N sweep ranges.
    2. Loop over (N, ω) pairs, calling a per-point optimisation function.
    3. Collect returned values into named arrays.
    4. Return a dict of arrays ready for dataclass construction.

Usage::

    from src.analysis.coupling_sweep import resolve_sweep_defaults, run_sweep_base

    def per_point(N, omega):
        result = my_optimiser(N, omega, ...)
        return {"delta_omega": result.delta, "sql": result.sql,
                "alpha_xx_opt": result.alpha}

    data = run_sweep_base(omega_values, N_values, per_point)
    result = MySweepResult(
        omega_values=data["omegas"],
        N_values=data["Ns"],
        delta_omega_opt=data["delta_opts"],
        ...
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def resolve_sweep_defaults(
    omega_values: np.ndarray | None,
    N_values: np.ndarray | None,
    *,
    default_omega_range: tuple[float, float, float] = (0.1, 5.0, 0.1),
    default_N_range: tuple[int, int] = (1, 20),
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve default ω and N value arrays for a sweep.

    Args:
        omega_values: Provided ω values (or ``None`` for default range).
        N_values: Provided N values (or ``None`` for default range).
        default_omega_range: ``(start, stop, step)`` for ``np.arange``.
        default_N_range: ``(start, stop)`` inclusive for ``np.arange``.

    Returns:
        Tuple ``(omega_values, N_values)`` as 1D arrays.
    """
    if omega_values is None:
        start, stop, step = default_omega_range
        omega_values = np.arange(start, stop + 1e-9, step)
    if N_values is None:
        n_start, n_stop = default_N_range
        N_values = np.arange(n_start, n_stop + 1, dtype=int)
    return omega_values, N_values


def _store_result_values(
    data: dict[str, np.ndarray],
    extra_fields: dict[str, tuple[Any, np.dtype]],
    idx: int,
    result: dict[str, Any],
    total: int,
) -> None:
    """Store per-point result values into data arrays, allocating on first use."""
    for key, value in result.items():
        if key in ("delta_omega", "sql"):
            continue
        if key == "ratio":
            v = float(value) if np.isfinite(value) else np.inf
            _set_or_init(data, key, idx, v, extra_fields, total)
        else:
            _set_or_init(data, key, idx, value, extra_fields, total)

    # Compute ratio if not provided
    if "ratio" not in result:
        d_val = data["delta_opts"][idx]
        s_val = data["sqls"][idx]
        if np.isfinite(d_val) and s_val > 0:
            _set_or_init(data, "ratio", idx, float(d_val / s_val), extra_fields, total)
        else:
            _set_or_init(data, "ratio", idx, np.inf, extra_fields, total)


def run_sweep_base(
    omega_values: np.ndarray,
    N_values: np.ndarray,
    per_point_fn: Callable[[int, float], dict[str, Any]],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Run a generic 2D (N, ω) sweep over coupling parameters.

    For each ``(N, ω)`` pair, calls ``per_point_fn(N, ω)`` and stores
    the returned values into named arrays. The function pre-allocates
    ``omegas``, ``Ns``, ``delta_opts``, and ``sqls`` arrays. All other
    keys in the per-point dict are allocated on first encounter.

    The ``per_point_fn`` must return at minimum ``"delta_omega"`` and
    ``"sql"``.  Common extra keys: ``"expectation_Jz"``, ``"variance_Jz"``,
    ``"d_expectation"``, ``"alpha_xx_opt"``, ``"ratio"`` (if not provided,
    it is computed as ``delta_omega / sql``).

    Args:
        omega_values: ω values to sweep (1D array).
        N_values: N values to sweep (1D integer array).
        per_point_fn: Callable ``(N, ω) → dict`` of results.
        progress_callback: Optional ``(current, total)`` callback.

    Returns:
        Dict mapping field names to 1D numpy arrays, one entry per
        ``(N, ω)`` pair in row-major order (ω varies fastest).
    """
    n_omega = len(omega_values)
    n_N = len(N_values)
    total = n_omega * n_N

    # Pre-allocate core arrays
    data: dict[str, np.ndarray] = {
        "omegas": np.zeros(total, dtype=float),
        "Ns": np.zeros(total, dtype=int),
        "delta_opts": np.full(total, np.inf, dtype=float),
        "sqls": np.zeros(total, dtype=float),
    }

    # Track which extra keys were discovered
    extra_fields: dict[str, tuple[Any, np.dtype]] = {}

    idx = 0
    for N in N_values:
        for omega in omega_values:
            result = per_point_fn(int(N), float(omega))

            data["omegas"][idx] = float(omega)
            data["Ns"][idx] = int(N)
            data["delta_opts"][idx] = float(result.get("delta_omega", np.inf))
            data["sqls"][idx] = float(result.get("sql", 0.0))

            _store_result_values(data, extra_fields, idx, result, total)

            idx += 1
            if progress_callback is not None:
                progress_callback(idx, total)

    return data


def _set_or_init(
    data: dict[str, np.ndarray],
    key: str,
    idx: int,
    value: Any,
    extra_fields: dict[str, tuple[Any, np.dtype]],
    total: int,
) -> None:
    """Set *data[key][idx] = value*, allocating the array if first encounter."""
    if key not in data:
        # Determine dtype from first non-None value
        val = value
        if val is None:
            arr = np.full(total, np.nan, dtype=float)
        elif isinstance(val, bool):
            arr = np.full(total, False, dtype=bool)
        elif isinstance(val, str):
            arr = np.empty(total, dtype=object)
            arr[:] = ""
        elif isinstance(val, int):
            arr = np.full(total, 0, dtype=int)
        else:
            arr = np.full(total, np.nan, dtype=float)
        data[key] = arr
        extra_fields[key] = (value, data[key].dtype)

    if value is not None:
        try:
            data[key][idx] = value
        except (ValueError, TypeError):
            if isinstance(value, str):
                data[key][idx] = value
            else:
                data[key][idx] = float(value)

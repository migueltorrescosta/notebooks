"""
Generic two-phase optimisation pipeline: random search + Nelder-Mead refinement.

Provides shared infrastructure for the common pattern used across 11+ report
report experiment modules (was ``local.py``):

    1. Uniform random search over a D-dimensional parameter space.
    2. Sort results by sensitivity, take top-k points.
    3. Nelder-Mead refinement (``scipy.optimize.minimize``, method="Nelder-Mead")
       from each top-k point.
    4. Return the best refined result.

Usage (callback-based)::

    best_nm, all_nm = run_two_phase_pipeline(
        random_search_fn=my_random_search,
        nm_fn=my_nelder_mead,
        config=TwoPhaseConfig(n_random=500, n_nm_refine=50),
        rs_kwargs={"N": N, "omega": omega},
        nm_kwargs={"N": N, "omega_true": omega},
    )

or with the generic helper functions::

    def raw_obj(p):
        return my_sensitivity(p, ...)

    samples, deltas = run_random_search(raw_obj, n_params=4, ...)
    nm_result = run_nelder_mead(raw_obj, x0=best_sample, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.optimize import minimize

from src.analysis.ancilla_drive_results import (
    DriveNelderMeadResult,
    DriveRandomSearchResult,
)
from src.analysis.ancilla_optimization import compute_expectation_and_variance

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------


def make_4d_objective(
    sensitivity_fn: Callable[..., float],
    *,
    psi0: np.ndarray,
    T_BS: float,
    t_hold: float,
    omega: float,
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
    meas_op: np.ndarray | None = None,
) -> Callable[[np.ndarray], float]:
    """Build a 4D-parameter objective function for Nelder-Mead.

    The returned closure accepts a 4-element array ``p = [ax, ay, az, azz]``
    and calls ``sensitivity_fn(psi0, T_BS, t_hold, omega, ax, ay, az, azz,
    ops, fd_step, meas_op)``.

    Args:
        sensitivity_fn: Function with signature
            ``(psi0, T_BS, t_hold, omega, ax, ay, az, azz, ops, fd_step, ...)``.
        psi0: Initial state vector.
        T_BS: Beam-splitter duration.
        t_hold: Holding time.
        omega: Phase rate parameter.
        ops: Operator dictionary.
        fd_step: Finite-difference step size (default 1e-6).
        meas_op: Measurement operator (default None = use ops['Jz_S']).

    Returns:
        Callable[[np.ndarray], float] — the objective function.
    """

    def _raw_objective(p: np.ndarray) -> float:
        return sensitivity_fn(
            psi0,
            T_BS,
            t_hold,
            omega,
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(p[3]),
            ops,
            fd_step,
            meas_op,
        )

    return _raw_objective


@dataclass
class TwoPhaseConfig:
    """Configuration for the two-phase (random search + Nelder-Mead) pipeline.

    Attributes:
        n_random: Number of random-search samples per pipeline run.
        n_nm_refine: Number of top random points to refine with Nelder-Mead.
        nm_maxiter: Maximum iterations per Nelder-Mead run.
        nm_xatol: Absolute parameter tolerance for Nelder-Mead.
        nm_fatol: Absolute function-value tolerance for Nelder-Mead.
        nm_adaptive: Use adaptive Nelder-Mead parameters.
        seed: Default random seed (overridable per call).
        bounds: Parameter bounds. A single ``(lo, hi)`` tuple applies to all
            dimensions. A list of ``(lo, hi)`` tuples gives per-dimension
            bounds (length must match ``n_params``).
        penalty_scale: Scale factor for quadratic bound-violation penalty.
    """

    n_random: int = 500
    n_nm_refine: int = 50
    nm_maxiter: int = 5000
    nm_xatol: float = 1e-8
    nm_fatol: float = 1e-8
    nm_adaptive: bool = True
    seed: int | None = 42
    bounds: tuple[float, float] | list[tuple[float, float]] = (-5.0, 5.0)
    penalty_scale: float = 1e6

    @property
    def n_params(self) -> int:
        """Infer parameter count from bounds configuration."""
        if isinstance(self.bounds, tuple):
            return 4  # common default; override explicitly if different
        return len(self.bounds)


# ---------------------------------------------------------------------------
# Bound-penalty helper
# ---------------------------------------------------------------------------


def _expand_bounds(
    bounds: tuple[float, float] | list[tuple[float, float]],
    n_params: int,
) -> list[tuple[float, float]]:
    """Normalise bounds to a per-parameter list of (lo, hi) tuples."""
    if isinstance(bounds, tuple):
        return [bounds] * n_params
    return list(bounds)


def penalized_objective(
    params: np.ndarray,
    raw_objective: Callable[[np.ndarray], float],
    bounds_per_param: list[tuple[float, float]],
    penalty_scale: float = 1e6,
) -> float:
    """Evaluate *raw_objective* with quadratic bound-violation penalties.

    Returns ``1e10 + penalty`` if any bound is violated (strong signal to the
    Nelder-Mead optimiser to avoid out-of-bounds regions).
    """
    penalty = 0.0
    for val, (lo, hi) in zip(params, bounds_per_param, strict=False):
        if val < lo:
            penalty += penalty_scale * (lo - val) ** 2
        elif val > hi:
            penalty += penalty_scale * (val - hi) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    return raw_objective(params)


# ---------------------------------------------------------------------------
# Generic random search
# ---------------------------------------------------------------------------


def run_random_search(
    objective_fn: Callable[[np.ndarray], float],
    n_params: int,
    n_samples: int = 500,
    bounds: tuple[float, float] | list[tuple[float, float]] = (-5.0, 5.0),
    seed: int | None = 42,
    penalty_scale: float = 1e6,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random search over the D-dimensional parameter space.

    For each sample, evaluates *objective_fn* and records the result.
    Bound violations are penalised via :func:`penalized_objective`.

    Args:
        objective_fn: Scalar function ``f(params) -> float``.
        n_params: Dimensionality of the parameter space.
        n_samples: Number of random points to evaluate.
        bounds: Single ``(lo, hi)`` or per-parameter list of bounds.
        seed: RNG seed.
        penalty_scale: Bound-penalty scale factor.

    Returns:
        Tuple ``(samples, values)`` — both ``(n_samples,)``-shaped arrays.
    """
    rng = np.random.default_rng(seed)
    bounds_list = _expand_bounds(bounds, n_params)

    samples = np.empty((n_samples, n_params), dtype=float)
    for j, (lo, hi) in enumerate(bounds_list):
        samples[:, j] = rng.uniform(lo, hi, size=n_samples)

    values = np.full(n_samples, np.inf, dtype=float)
    for i in range(n_samples):
        values[i] = penalized_objective(
            samples[i],
            objective_fn,
            bounds_list,
            penalty_scale,
        )

    return samples, values


# ---------------------------------------------------------------------------
# Generic Nelder-Mead wrapper
# ---------------------------------------------------------------------------


def run_nelder_mead(
    objective_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
    penalty_scale: float = 1e6,
    maxiter: int = 5000,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    track_history: bool = False,
) -> dict[str, Any]:
    """Run Nelder-Mead optimisation (``scipy.optimize.minimize``).

    The objective is wrapped with bound-violation penalties so that
    ``scipy.optimize.minimize`` (which does not natively support bounds for
    Nelder-Mead) stays within the feasible region.

    Args:
        objective_fn: Raw (unpenalised) scalar function.
        x0: Starting point.
        bounds: Parameter bounds. If ``None``, no penalty is applied.
        penalty_scale: Bound-penalty scale factor.
        maxiter: Maximum iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function-value tolerance.
        adaptive: Use adaptive Nelder-Mead parameters.
        track_history: If True, record objective values per iteration.

    Returns:
        Dict with keys ``x_opt``, ``fun_opt``, ``success``, ``nfev``,
        ``message``, and ``history`` (list of objective values).
    """
    x0 = np.asarray(x0, dtype=float)

    wrapped_obj: Callable[[np.ndarray], float]

    if bounds is not None:
        bounds_list = _expand_bounds(bounds, len(x0))

        def _wrapped(p: np.ndarray) -> float:
            return penalized_objective(p, objective_fn, bounds_list, penalty_scale)

        wrapped_obj = _wrapped
    else:
        wrapped_obj = objective_fn

    history: list[float] = []

    def _callback(_x: np.ndarray) -> None:
        if track_history:
            history.append(wrapped_obj(_x))

    result = minimize(
        wrapped_obj,
        x0=x0,
        method="Nelder-Mead",
        callback=cast("Any", _callback if track_history else None),
        options=cast(
            "Any",
            {
                "maxiter": maxiter,
                "xatol": xatol,
                "fatol": fatol,
                "adaptive": adaptive,
            },
        ),
    )

    return {
        "x_opt": np.asarray(result.x, dtype=float),
        "fun_opt": float(result.fun),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "message": str(result.message),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Two-phase pipeline (callback-based)
# ---------------------------------------------------------------------------


def run_two_phase_pipeline(
    random_search_fn: Callable[..., Any],
    nm_fn: Callable[..., Any],
    config: TwoPhaseConfig,
    seed: int | None = None,
    rs_kwargs: dict[str, Any] | None = None,
    nm_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, list[Any]]:
    """Run a generic two-phase optimisation pipeline.

    Stage 1 — call ``random_search_fn(n_samples, seed, **rs_kwargs)``.
        The return value must have ``.samples`` (ndarray) and
        ``.delta_omega_values`` (ndarray) attributes.

    Stage 2 — sort by ``delta_omega_values``, select top
    ``config.n_nm_refine`` points.

    Stage 3 — for each top point, call
    ``nm_fn(x0=point, seed=..., **nm_kwargs)``.
        The return value must have a ``.delta_omega_opt`` attribute.

    Stage 4 — sort Nelder-Mead results by ``.delta_omega_opt``.

    Args:
        random_search_fn: Callable for the random-search stage.
        nm_fn: Callable for a single Nelder-Mead refinement.
        config: Pipeline configuration.
        seed: Base seed. Falls back to ``config.seed``.
        rs_kwargs: Extra keyword arguments forwarded to *random_search_fn*.
        nm_kwargs: Extra keyword arguments forwarded to *nm_fn*.

    Returns:
        Tuple ``(best_result, all_results_sorted)`` where *all_results_sorted*
        is a list of all Nelder-Mead results sorted by ascending
        ``.delta_omega_opt``.
    """
    base_seed = (
        seed if seed is not None else (config.seed if config.seed is not None else 42)
    )

    # Stage 1: Random search
    rs_result = random_search_fn(
        n_samples=config.n_random,
        seed=base_seed,
        **(rs_kwargs or {}),
    )

    samples = rs_result.samples  # type: ignore[attr-defined]
    delta_values = rs_result.delta_omega_values  # type: ignore[attr-defined]

    # Stage 2: Sort and select top points
    sorted_indices = np.argsort(delta_values)
    top_indices = sorted_indices[: config.n_nm_refine]

    # Stage 3: Nelder-Mead refinement
    nm_results: list[Any] = []
    for rank, idx in enumerate(top_indices):
        x0 = samples[idx].copy()
        nm = nm_fn(
            x0=x0,
            seed=base_seed + 10000 + rank,
            **(nm_kwargs or {}),
        )
        nm_results.append(nm)

    # Stage 4: Sort by result quality
    nm_results.sort(key=lambda r: r.delta_omega_opt)
    return nm_results[0], nm_results


# ---------------------------------------------------------------------------
# Omega-scan wrapper
# ---------------------------------------------------------------------------


def run_omega_scan(
    omega_values: list[float] | np.ndarray,
    random_search_fn: Callable[..., Any],
    nm_fn: Callable[..., Any],
    config: TwoPhaseConfig,
    seed: int | None = None,
    rs_kwargs: dict[str, Any] | None = None,
    nm_kwargs: dict[str, Any] | None = None,
    rs_omega_key: str = "omega",
    nm_omega_key: str = "omega_true",
) -> tuple[list[Any], list[list[Any]]]:
    """Scan over ω values using the two-phase pipeline per ω.

    For each ω in *omega_values*:

    1. Add ``{rs_omega_key: ω}`` to the random-search kwargs.
    2. Add ``{nm_omega_key: ω}`` to the Nelder-Mead kwargs.
    3. Run :func:`run_two_phase_pipeline`.
    4. Record the best result and all results.

    Args:
        omega_values: ω values to scan.
        random_search_fn: Callable for random search.
        nm_fn: Callable for Nelder-Mead refinement.
        config: Pipeline configuration.
        seed: Base seed (incremented by ``int(ω * 1000)`` per ω).
        rs_kwargs: Static kwargs for random search (same for all ω).
        nm_kwargs: Static kwargs for Nelder-Mead (same for all ω).
        rs_omega_key: Key under which the current ω is added to RS kwargs.
        nm_omega_key: Key under which the current ω is added to NM kwargs.

    Returns:
        Tuple ``(best_results, all_results_per_omega)`` where *best_results*
        is a list of the best NM result for each ω, and
        *all_results_per_omega* is a list of lists of all NM results per ω.
    """
    omega_arr = np.asarray(omega_values, dtype=float)
    base_seed = (
        seed if seed is not None else (config.seed if config.seed is not None else 42)
    )

    best_results: list[Any] = []
    all_results: list[list[Any]] = []

    base_rs_kwargs = dict(rs_kwargs or {})
    base_nm_kwargs = dict(nm_kwargs or {})

    for omega_val in omega_arr:
        rs_kw = dict(base_rs_kwargs)
        rs_kw[rs_omega_key] = float(omega_val)
        nm_kw = dict(base_nm_kwargs)
        nm_kw[nm_omega_key] = float(omega_val)

        omega_seed = base_seed + int(omega_val * 1000)

        best_nm, all_nm = run_two_phase_pipeline(
            random_search_fn=random_search_fn,
            nm_fn=nm_fn,
            config=config,
            seed=omega_seed,
            rs_kwargs=rs_kw,
            nm_kwargs=nm_kw,
        )
        best_results.append(best_nm)
        all_results.append(all_nm)

    return best_results, all_results


# ---------------------------------------------------------------------------
# Result-wrapping helpers (promoted from report local.py files)
# ---------------------------------------------------------------------------


def build_rs_result(
    raw_objective: Callable[[np.ndarray], float],
    n_samples: int,
    seed: int,
    *,
    omega: float,
    sql: float,
    t_hold: float,
    bounds: tuple[float, float] = (-5.0, 5.0),
) -> DriveRandomSearchResult:
    """4D random search → DriveRandomSearchResult.

    Runs :func:`run_random_search` over the 4D parameter space
    ``(a_x, a_y, a_z, a_zz)`` and wraps the result in a
    :class:`~src.analysis.ancilla_drive_results.DriveRandomSearchResult`.

    Args:
        raw_objective: Raw (unpenalised) objective ``f(params) -> Δω``.
        n_samples: Number of random points to evaluate.
        seed: RNG seed.
        omega: ω value at which the search was performed.
        sql: SQL reference value for the result.
        t_hold: Holding time for the result.
        bounds: Parameter bounds ``(lo, hi)`` (same for all 4 dimensions).

    Returns:
        DriveRandomSearchResult with samples, deltas, and best found.
    """
    samples, deltas = run_random_search(
        raw_objective,
        n_params=4,
        n_samples=n_samples,
        bounds=bounds,
        seed=seed,
    )
    best_idx = int(np.argmin(deltas))
    return DriveRandomSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        best_params=(
            float(samples[best_idx, 0]),
            float(samples[best_idx, 1]),
            float(samples[best_idx, 2]),
            float(samples[best_idx, 3]),
        ),
        best_delta_omega=float(deltas[best_idx]),
        omega_value=omega,
        sql=sql,
        t_hold=t_hold,
    )


def build_nm_result(
    raw_objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    omega: float,
    ops: dict[str, np.ndarray],
    psi0: np.ndarray,
    evolve_fn: Callable[
        [np.ndarray, float, float, float, float, dict[str, np.ndarray]],
        np.ndarray,
    ],
    t_hold: float,
    maxiter: int = 5000,
    bounds: tuple[float, float] = (-5.0, 5.0),
) -> DriveNelderMeadResult:
    """Nelder-Mead refinement → DriveNelderMeadResult.

    Runs :func:`run_nelder_mead` from a given starting point, evaluates
    the final evolved state with *evolve_fn* to extract measurement
    statistics, and wraps the result in a
    :class:`~src.analysis.ancilla_drive_results.DriveNelderMeadResult`.

    The *evolve_fn* callback must have the signature::

        evolve_fn(psi0, a_x, a_y, a_z, a_zz, ops) -> evolved_state

    Each report wraps its own circuit evolution function (which may
    carry ``N``, ``T_BS``, ``T_HOLD``, etc. as closures) to match this
    interface.

    Args:
        raw_objective: Raw (unpenalised) objective ``f(params) -> Δω``.
        x0: Nelder-Mead starting point (4-element vector).
        omega: True ω value.
        ops: Operator dict (must contain ``'Jz_S'``).
        psi0: Initial state vector.
        evolve_fn: Callable that applies the full circuit evolution.
        t_hold: Holding time (for result metadata).
        maxiter: Maximum Nelder-Mead iterations.
        bounds: Parameter bounds ``(lo, hi)``.

    Returns:
        DriveNelderMeadResult with optimal params and diagnostics.
    """
    nm = run_nelder_mead(
        raw_objective,
        x0=x0,
        bounds=bounds,
        maxiter=maxiter,
        track_history=False,
    )
    opt_p = nm["x_opt"]
    psi_final = evolve_fn(
        psi0,
        float(opt_p[0]),
        float(opt_p[1]),
        float(opt_p[2]),
        float(opt_p[3]),
        ops,
    )
    exp_val, var_val = compute_expectation_and_variance(psi_final, ops["Jz_S"])
    return DriveNelderMeadResult(
        delta_omega_opt=nm["fun_opt"],
        params_opt=opt_p,
        omega_true=omega,
        success=nm["success"],
        nfev=nm["nfev"],
        message=nm["message"],
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        history=nm["history"],
    )

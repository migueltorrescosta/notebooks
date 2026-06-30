r"""
Local module for the 2026-06-25 Squeezed-Vacuum MZI with Parity Measurement report.

Contains all code exclusive to this report:

- Parity distribution computation P(\pm 1|\omega) from output state
- Parity-based CFI sensitivity grid (replaces number-difference with parity)
- MZI evolution reuse from ``src.physics.mzi_simulation``
- Parquet-serializable dataclass (reuses ``MziSensitivityDataSV``)
- Log-log scaling exponent fitting
- Plot functions for Δω vs ω and Δω vs resource parameter
- CLI pipeline for generating all data and figures

Usage:

    uv run python reports/20260625/squeezed_vacuum_parity.py --force

This module is **not** importable as ``reports.20260625.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import squeezed_vacuum_parity``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
from pathlib import Path

import numpy as np
import seaborn as sns

from src.analysis import mzi_pipeline
from src.analysis.sensitivity_metrics import MziSensitivityDataSV
from src.physics.hilbert_space import resource_value_to_truncation
from src.physics.mzi_simulation import (
    beam_splitter_unitary,
    compute_mzi_sensitivity_grid,
)
from src.physics.mzi_states import input_state_factory
from src.utils.paths import report_path_fn
from src.visualization import mzi_plots

# Force non-interactive backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

sns.set_theme(style="whitegrid")

# ============================================================================
# Constants
# ============================================================================

_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
_REPORT_DATE = "20260625"
t_hold: float = 10.0  # Holding time
SV_N_RANGE: list[float] = [float(n) for n in range(1, 21)]
_parquet_path, _fig_path = report_path_fn(_REPORTS_DIR, _REPORT_DATE)

TRUNC_MULTIPLIER: float = 5.0  # Truncation multiplier for SV
MAX_TRUNC: int = 80  # Maximum truncation per mode

# Parameter sweep ranges
OMEGA_RANGE: tuple[float, float] = (0.1, 5.0)
OMEGA_STEP: float = 0.01

# Finite difference step for CFI
CFI_EPSILON: float = 1e-6
PROB_FLOOR: float = 1e-15


# ============================================================================
# Parity Distribution
# ============================================================================


def compute_parity_distribution(
    state_out: np.ndarray,
    max_photons: int,
) -> np.ndarray:
    r"""Compute :math:`P(\pm 1|\omega)` from the output state.

    The parity operator is :math:`\Pi = (-1)^{n_2}` (photon-number parity at
    output port 2).  The two-outcome distribution is:

        :math:`P(+1|\omega) = \sum_{n_2\text{ even}} P(n_1, n_2)`,
        :math:`P(-1|\omega) = \sum_{n_2\text{ odd}} P(n_1, n_2)`.

    The parity expectation is :math:`\langle\Pi\rangle = P(+1) - P(-1)`.

    Args:
        state_out: Output state vector in the two-mode Fock basis,
            dimension ``(max_photons+1)^2``.
        max_photons: Maximum photon number per mode.

    Returns:
        Array of shape ``(2,)`` with ``[P(+1), P(-1)]``.
    """
    P = np.zeros(2, dtype=float)
    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            idx = n1 * (max_photons + 1) + n2
            prob = np.real(state_out[idx].conj() * state_out[idx])
            if n2 % 2 == 0:
                P[0] += prob  # even → +1
            else:
                P[1] += prob  # odd → -1
    return P


# ============================================================================
# Parity Sensitivity Grid
# ============================================================================


def compute_parity_sensitivity_grid(
    initial_state: np.ndarray,
    omega_grid: np.ndarray,
    max_photons: int,
    t_hold: float = 10.0,
    skip_bs1: bool = False,
    cfi_epsilon: float = CFI_EPSILON,
    prob_floor: float = PROB_FLOOR,
    bs: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    r"""Compute :math:`\Delta\omega_C`, :math:`\Delta\omega_{\text{EP}}` and
    :math:`\Delta\omega_Q` across a :math:`\omega` grid using parity measurement.

    Delegates to :func:`compute_mzi_sensitivity_grid` with a custom
    ``distribution_fn`` (parity distribution) and ``observable_fn``
    (parity expectation and variance).

    For each :math:`\omega_i`:
        1. Evolve the state through the MZI
        2. Compute the parity distribution :math:`P(\pm 1|\omega_i)` from the
           output state (2-outcome distribution)
        3. Compute :math:`\langle\Pi\rangle_{\text{out}}` and
           :math:`\text{Var}(\Pi)_{\text{out}} = 1 - \langle\Pi\rangle^2`
        4. Compute :math:`\partial P(\pm 1|\omega)/\partial\omega` via
           central finite differences with step ``cfi_epsilon``
        5. :math:`F_C^\Pi = \sum_{k=\pm 1} (\partial P_k/\partial\omega)^2 / P_k`
        6. :math:`\Delta\omega_C = 1/\sqrt{F_C^\Pi}`

    The QFI bound :math:`\Delta\omega_Q = 1/\sqrt{F_Q}` is computed from the
    probe state using :math:`F_Q = 4 t_hold^2 \text{Var}(J_z)_{\text{probe}}`,
    independent of :math:`\omega`.

    Args:
        initial_state: Input state in the two-mode Fock basis.
        omega_grid: Array of :math:`\omega` values to evaluate.
        max_photons: Maximum photon number per mode.
        t_hold: Holding time.
        skip_bs1: If True, omit BS1 from both probe and evolution.
        cfi_epsilon: Finite-difference step for CFI derivatives.
        prob_floor: Minimum probability floor for CFI denominator.
        bs: Pre-computed beam-splitter unitary. If None, computed fresh.

    Returns:
        Dictionary with keys:
        - ``omega_values``: The input :math:`\omega` grid.
        - ``expectation_values``: :math:`\langle\Pi\rangle_{\text{out}}`.
        - ``variance_values``: :math:`\text{Var}(\Pi)_{\text{out}} = 1 - \langle\Pi\rangle^2`.
        - ``derivative_values``: :math:`\partial\langle\Pi\rangle/\partial\omega`.
        - ``delta_omega_ep``: :math:`\Delta\omega_{\text{EP}}` (error propagation from parity).
        - ``delta_omega_q``: :math:`\Delta\omega_Q` (scalar, :math:`\omega`-independent).
        - ``fisher_quantum``: :math:`F_Q` (scalar).
        - ``fisher_classical``: :math:`F_C^\Pi(\omega)` (array).
        - ``delta_omega_c``: :math:`\Delta\omega_C(\omega)` (array).
    """

    def _parity_observable(state: np.ndarray, _max_photons: int) -> tuple[float, float]:
        P = compute_parity_distribution(state, _max_photons)
        exp = P[0] - P[1]
        var = 1.0 - exp**2
        return exp, var

    return compute_mzi_sensitivity_grid(
        initial_state,
        omega_grid,
        max_photons,
        t_hold=t_hold,
        skip_bs1=skip_bs1,
        cfi_epsilon=cfi_epsilon,
        prob_floor=prob_floor,
        bs=bs,
        distribution_fn=compute_parity_distribution,
        observable_fn=_parity_observable,
    )


# ============================================================================
# Generate Omega Scan (Single Resource Value)
# ============================================================================


def generate_single_omega_scan(
    resource_value: float,
    omega_grid: np.ndarray,
    max_photons: int | None = None,
    t_hold: float = t_hold,
) -> MziSensitivityDataSV:
    r"""Run a :math:`\omega` scan for a single resource value using parity measurement.

    Args:
        resource_value: Mean photon number :math:`\langle N \rangle`.
        omega_grid: :math:`\omega` values to scan.
        max_photons: Hilbert space truncation (auto-computed if None).
        t_hold: Holding time.

    Returns:
        MziSensitivityDataSV with one resource value.
    """
    if max_photons is None:
        max_photons = resource_value_to_truncation(
            resource_value, "sv", trunc_multiplier=TRUNC_MULTIPLIER, max_trunc=MAX_TRUNC
        )

    resource_type = "mean_N"
    state_type = "sv_parity"
    # NOTE: skip_bs1 must be False for parity measurement to recover sensitivity.
    # With skip_bs1=True, the SV state Σ c_n |2n,0⟩ has total-photon-number
    # superselection: after BS2, each output Fock state |n1,n2⟩ couples to exactly
    # one input component (n = (n1+n2)/2).  Since Π = (-1)^{n₂} is diagonal in
    # the Fock basis, there are no cross-term coherences between different n,
    # and ⟨Π⟩ becomes ω-independent (⟨Π⟩ = 1/cosh(r)).  This gives F_C^Π = 0,
    # exactly like the number-difference measurement.
    # With skip_bs1=False, BS1 creates path entanglement that allows the phase
    # shift (J_z generator) to produce ω-dependent parity oscillations.
    skip_bs1 = False

    # Prepare the SV state
    r_val = float(np.arcsinh(np.sqrt(max(resource_value, 1.0))))
    state = input_state_factory(
        "squeezed_vacuum",
        round(resource_value),
        max_photons,
        r=r_val,
    )

    # Pre-compute BS matrix once and reuse
    bs = beam_splitter_unitary(np.pi / 4, 0.0, max_photons)

    # Compute sensitivity grid using parity measurement
    result = compute_parity_sensitivity_grid(
        state,
        omega_grid,
        max_photons,
        t_hold=t_hold,
        skip_bs1=skip_bs1,
        bs=bs,
    )

    omega_arr = np.asarray(result["omega_values"], dtype=float)

    return MziSensitivityDataSV(
        state_type=state_type,
        resource_type=resource_type,
        resource_values=np.array([resource_value], dtype=float),
        omega_values=omega_arr,
        expectation_grid=np.atleast_2d(np.asarray(result["expectation_values"])),
        variance_grid=np.atleast_2d(np.asarray(result["variance_values"])),
        derivative_grid=np.atleast_2d(np.asarray(result["derivative_values"])),
        delta_omega_ep_grid=np.atleast_2d(np.asarray(result["delta_omega_ep"])),
        delta_omega_q_per_R=np.array([float(result["delta_omega_q"])]),
        fisher_classical_grid=np.atleast_2d(np.asarray(result["fisher_classical"])),
        delta_omega_c_grid=np.atleast_2d(np.asarray(result["delta_omega_c"])),
        t_hold=t_hold,
        truncation_M_per_R=np.array([max_photons], dtype=float),
        squeezing_q_per_R=None,
    )


# ============================================================================
# Generate Full Data (All Resource Values)
# ============================================================================


def _generate_single_resource_data(
    resource_value: float,
    omega_grid: np.ndarray,
    t_hold: float = t_hold,
) -> MziSensitivityDataSV | None:
    r"""Run omega scan for a single resource value, returning None on failure.

    Delegates to :func:`safe_generate_scan` with a closure that computes
    the truncation and calls :func:`generate_single_omega_scan`.

    Args:
        resource_value: Mean photon number :math:`\langle N \rangle`.
        omega_grid: :math:`\omega` values to scan.
        t_hold: Holding time.

    Returns:
        MziSensitivityDataSV with one resource value, or None on failure.
    """

    def _gen_one(
        R: float, og: np.ndarray, t_hold: float
    ) -> MziSensitivityDataSV | None:
        _t0 = time.perf_counter()
        max_photons = resource_value_to_truncation(
            R, "sv", trunc_multiplier=TRUNC_MULTIPLIER, max_trunc=MAX_TRUNC
        )
        result = generate_single_omega_scan(
            R,
            og,
            max_photons=max_photons,
            t_hold=t_hold,
        )
        _elapsed = time.perf_counter() - _t0
        print(
            f"    SV R={R} (M={max_photons}, {len(og)} omega points): {_elapsed:.1f}s",
            flush=True,
        )
        # Free the beam-splitter matrix from the LRU cache after each R value.
        beam_splitter_unitary.cache_clear()
        # Force glibc to release freed memory back to the OS.
        import ctypes

        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass
        return result

    return mzi_pipeline.safe_generate_scan(
        _gen_one,
        resource_value,
        omega_grid,
        t_hold,
        fail_label=f"SV R={resource_value}",
    )


def generate_full_data(
    resource_range: list[float],
    omega_grid: np.ndarray,
    t_hold: float = t_hold,
) -> MziSensitivityDataSV:
    r"""Generate sensitivity data for all resource values in a range.

    Delegates to the shared :func:`pipeline_generate_full_data`.

    Args:
        resource_range: List of mean photon number values.
        omega_grid: :math:`\omega` values to scan.
        t_hold: Holding time.

    Returns:
        MziSensitivityDataSV with all resource values.
    """

    def _gen(R: float, og: np.ndarray, t_hold: float) -> MziSensitivityDataSV | None:
        return _generate_single_resource_data(R, og, t_hold=t_hold)

    return mzi_pipeline.generate_full_data(
        "sv_parity",
        resource_range,
        omega_grid,
        _gen,
        t_hold=t_hold,
    )


def generate_full_data_parallel(
    resource_range: list[float],
    omega_grid: np.ndarray,
    t_hold: float = t_hold,
    max_workers: int | None = None,
) -> MziSensitivityDataSV:
    r"""Generate sensitivity data for all resource values in parallel.

    Uses :class:`~concurrent.futures.ProcessPoolExecutor` to parallelise
    across resource values.  Falls back to sequential execution when
    *max_workers=1*.

    Args:
        resource_range: List of mean photon number values.
        omega_grid: :math:`\omega` values to scan.
        t_hold: Holding time.
        max_workers: Number of parallel worker processes.
            Defaults to ``os.cpu_count()``.

    Returns:
        :class:`MziSensitivityDataSV` with all resource values.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    if max_workers <= 1:
        return generate_full_data(resource_range, omega_grid, t_hold=t_hold)

    scan_results: list[MziSensitivityDataSV] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
    ) as pool:
        future_map: dict[concurrent.futures.Future, float] = {}
        for R in resource_range:
            fut = pool.submit(
                _generate_single_resource_data,
                R,
                omega_grid,
                t_hold,
            )
            future_map[fut] = R

        for fut in concurrent.futures.as_completed(future_map):
            R = future_map[fut]
            try:
                result = fut.result()
                if result is not None:
                    scan_results.append(result)
                    print(f"  SV R={R} completed successfully", flush=True)
            except Exception as exc:
                print(f"  SV R={R} failed: {exc}", flush=True)

    if not scan_results:
        raise RuntimeError("No valid resource values for sv_parity")

    return mzi_pipeline.concatenate_scan_results(
        scan_results,
        "sv_parity",
        t_hold,
    )


def _maybe_generate_full_data(
    r_range: list[float],
    label: str,
    omega_grid: np.ndarray,
    force: bool,
    override_pq_path: Path | None = None,
) -> MziSensitivityDataSV:
    r"""Load or generate sensitivity data for the SV parity measurement.

    Uses :func:`generate_full_data_parallel` for generation, with
    :class:`~concurrent.futures.ProcessPoolExecutor` to parallelise
    across resource values.  Caches results as Parquet.

    Args:
        r_range: List of resource values.
        label: Human-readable name for logging.
        omega_grid: :math:`\omega` grid.
        force: Re-generate even if Parquet exists.
        override_pq_path: If set, use this path instead of the default
            production path.  Intended for tests.

    Returns:
        MziSensitivityDataSV.
    """
    pq_path = override_pq_path or _parquet_path("sv_parity_sensitivity")

    if pq_path.exists() and not force:
        print(f"Loading existing data for {label} from {pq_path}")
        return MziSensitivityDataSV.from_parquet(pq_path)

    print(
        f"Generating {label} data (R={r_range[0]}..{r_range[-1]})"
        f" using {os.cpu_count() or 1} parallel workers"
    )
    _t0 = time.perf_counter()
    data = generate_full_data_parallel(r_range, omega_grid, t_hold=10.0)
    _elapsed = time.perf_counter() - _t0
    # Normalise elapsed to the same format as _generate_single_resource_data
    print(f"  Total generation finished in {_elapsed:.1f}s")
    data.save_parquet(pq_path)
    print(f"  Saved to {pq_path}")
    return data


# ============================================================================
# Plot Functions
# ============================================================================


def plot_delta_omega_overlay(
    data: MziSensitivityDataSV,
    selected_R: list[float] | None = None,
    save_path: str | Path | None = None,
) -> Path:
    """Overlay Δω_C and Δω_Q vs ω for multiple resource values on a single panel.

    Delegates to the shared :func:`mzi_plots.plot_delta_omega_overlay` with
    :func:`_fig_path` for default save-path resolution.

    Args:
        data: Sensitivity data containing all resource values.
        selected_R: Which resource values to include (auto-selected if None).
        save_path: Output SVG path. Auto-generated if None.

    Returns:
        Path to saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("sv_parity_delta_omega_comparison")
    return mzi_plots.plot_delta_omega_overlay(
        data,
        selected_R=selected_R,
        save_path=save_path,
        title="SV + Parity Measurement --- Phase Sensitivity vs $\\omega$",
    )


def plot_scaling(
    data: MziSensitivityDataSV,
    save_path: str | Path | None = None,
    N_min_fit: float = 4.0,
) -> Path:
    """Log-log plot of best Δω vs resource parameter with analytical QFI bound and fit.

    Delegates to the shared :func:`mzi_plots.plot_scaling` with
    :func:`_fig_path` for default save-path resolution.

    Args:
        data: Sensitivity data for SV parity measurement.
        save_path: Output SVG path.
        N_min_fit: Minimum resource value for exponent fits.

    Returns:
        Path to saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("sv_parity_scaling")
    return mzi_plots.plot_scaling(
        [data],
        ["SV+Parity"],
        save_path=save_path,
        N_min_fit=N_min_fit,
        xlabel="Mean photon number $\\langle N \\rangle$",
        title="SV + Parity Measurement --- Phase Sensitivity Scaling",
    )


# ============================================================================
# Main Pipeline
# ============================================================================


def generate_all(
    force: bool = False,
    override_pq_path: Path | None = None,
) -> MziSensitivityDataSV:
    """Generate all data and figures for the report.

    Args:
        force: If True, re-generate even if Parquet files exist.
        override_pq_path: If set, use this path instead of the default
            production path.  Intended for tests.

    Returns:
        MziSensitivityDataSV with all resource values.
    """
    omega_grid = np.arange(OMEGA_RANGE[0], OMEGA_RANGE[1] + OMEGA_STEP / 2, OMEGA_STEP)

    print("Generating SV parity measurement data")
    _t0 = time.perf_counter()
    data = _maybe_generate_full_data(
        SV_N_RANGE,
        "SV+Parity",
        omega_grid,
        force,
        override_pq_path=override_pq_path,
    )
    _elapsed = time.perf_counter() - _t0
    print(f"Total pipeline: {_elapsed:.1f}s")

    # Generate plots
    overlay_path = _fig_path("sv_parity_delta_omega_comparison")
    if not overlay_path.exists() or force:
        plot_delta_omega_overlay(data, save_path=overlay_path)
        print(f"  Plotted {overlay_path}")

    scaling_path = _fig_path("sv_parity_scaling")
    if not scaling_path.exists() or force:
        plot_scaling(data, save_path=scaling_path)
        print(f"  Plotted {scaling_path}")

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Squeezed-Vacuum MZI with Parity Measurement",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate data and figures even if files exist",
    )
    args = parser.parse_args()

    generate_all(force=args.force)


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    main()

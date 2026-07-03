r"""
Local module for the 2026-06-25 Heisenberg-Limit MZI: Squeezed Vacuum & OAT report.

Contains all code exclusive to this report:
- Single-mode squeezed vacuum (SV) input state
- Two-mode squeezed vacuum (TMSV) input state
- OAT spin-squeezed state preparation with Dicke-to-Fock mapping
- Ancilla-free MZI evolution (reused from #20260601)
- Sensitivity computation via Classical Fisher Information (CFI) from
  the full number-difference distribution P(m|ω), plus QFI bound
- Parquet-serializable dataclass for grid results
- OAT optimal q-parameter scan
- Log-log scaling exponent fitting
- Plot functions for Δω vs ω and Δω vs resource parameter
- CLI pipeline for generating all data and figures

Usage:
    uv run python reports/20260625/heisenberg_limit_mzi_sq_oat.py --force
    uv run python reports/20260625/heisenberg_limit_mzi_sq_oat.py --only sv
    uv run python reports/20260625/heisenberg_limit_mzi_sq_oat.py --only oat

This module is importable via ``importlib.import_module("reports.20260625.heisenberg_limit_mzi_sq_oat")``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import seaborn as sns

from src.analysis import mzi_pipeline
from src.analysis.fisher_information import classical_fisher_information_single

# Re-export shared analysis functions for backward compatibility with local imports.
from src.analysis.sensitivity_metrics import (
    MziSensitivityDataSV,
)
from src.physics.hilbert_space import resource_value_to_truncation
from src.physics.mzi_distribution import output_number_diff_distribution
from src.physics.mzi_simulation import (
    beam_splitter_unitary,
    compute_mzi_sensitivity_grid,
    simple_mzi_evolution,
)
from src.physics.mzi_states import (
    input_state_factory,
    make_two_mode_squeezed_vacuum,
    standard_twin_fock_state,
)
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

OAT_N_Q_POINTS: int = 20  # Number of q points per OAT N value
OAT_CHI: float = 1.0  # OAT coupling strength

# Parameter sweep ranges
TMSV_N_RANGE: list[float] = [
    float(n) for n in range(2, 41, 2)
]  # Total ⟨N⟩ = 2..40 even
OAT_N_RANGE: list[int] = list(range(2, 41, 2))  # Even N=2..40
OMEGA_RANGE: tuple[float, float] = (0.01, 5.0)
OMEGA_STEP: float = 0.01

# Scaling fit


# ============================================================================
# Two-Mode Squeezed Vacuum State
# ============================================================================
# (make_two_mode_squeezed_vacuum imported from src.physics.mzi_states)


# ============================================================================
# Dicke-to-Fock Mapping
# ============================================================================


def _dicke_to_fock(dicke_state: np.ndarray, N: int) -> np.ndarray:
    r"""Map a Dicke-basis state to the two-mode Fock basis.

    The Schwinger mapping is:
        :math:`|J, m\rangle \leftrightarrow |J+m, J-m\rangle`
    where :math:`J = N/2`.

    The Dicke basis ordering is :math:`m = N/2, N/2-1, \dots, -N/2` (descending).

    Args:
        dicke_state: State vector in the Dicke basis, dimension ``(N+1,)``.
        N: Total particle number.

    Returns:
        Normalised state vector in the two-mode Fock basis, dimension ``(N+1)^2``.

    Raises:
        ValueError: If dicke_state dimension does not match N+1.
    """
    if dicke_state.shape[0] != N + 1:
        raise ValueError(
            f"Dicke state dimension {dicke_state.shape[0]} != N+1 = {N + 1}"
        )
    dim_single = N + 1
    dim = dim_single**2
    fock = np.zeros(dim, dtype=complex)
    J = N / 2.0

    # Descending m: J, J-1, ..., -J
    m_values = np.arange(J, -J - 1, -1)
    for idx, m in enumerate(m_values):
        n1 = int(J + m)  # = J + m in {0, 1, ..., N}
        n2 = int(J - m)  # = J - m in {0, 1, ..., N}
        fock_idx = n1 * dim_single + n2
        fock[fock_idx] = dicke_state[idx]

    # Normalise (should already be normalised, but numerical safety)
    norm = np.linalg.norm(fock)
    if norm > 0:
        fock /= norm

    return fock


# ============================================================================
# OAT Spin-Squeezed State
# ============================================================================


def _make_oat_state(N: int, q: float) -> np.ndarray:
    r"""Create an OAT spin-squeezed state in the two-mode Fock basis.

    The state is prepared by:
        1. CSS :math:`|J, -J\rangle_x` in the Dicke basis (coherent spin state
           aligned along -x, which is equivalent to BS1 applied to |N, 0⟩)
        2. OAT evolution :math:`\exp(-i q J_z^2)` (when q > 0)
        3. Dicke-to-Fock mapping via the Schwinger representation

    Args:
        N: Total particle number (must be even >= 2).
        q: OAT parameter :math:`q = \chi t` (use q=0 for CSS baseline).

    Returns:
        Normalised state vector in the two-mode Fock basis, dimension ``(N+1)^2``.

    Raises:
        ValueError: If N is not even or N < 2.
    """
    if N % 2 != 0:
        raise ValueError(f"N must be even for OAT state, got {N}")
    if N < 2:
        raise ValueError(f"N must be >= 2 for OAT state, got {N}")

    from src.algorithms.spin_squeezing import (
        coherent_spin_state,
        one_axis_twist,
    )

    # 1. CSS in Dicke basis (|J, -J⟩_x — equivalent to BS1 on |N, 0⟩)
    css = coherent_spin_state(N)

    # 2. OAT evolution
    if q > 0:
        oat_dicke = one_axis_twist(css, N, chi=OAT_CHI, t=q)
    else:
        oat_dicke = css.copy()

    # 3. Dicke → Fock mapping
    return _dicke_to_fock(oat_dicke, N)


# ============================================================================
# State Preparation Dispatch
# ============================================================================


def _prepare_state(
    state_type: str,
    resource_value: float,
    max_photons: int,
    q: float = 0.0,
) -> np.ndarray:
    r"""Dispatch state creation by type.

    Args:
        state_type: ``"sv"``, ``"tmsv"``, ``"oat"``, ``"noon"``, or ``"twin_fock_std"``.
        resource_value: Resource parameter (mean N for SV/TMSV, N for OAT).
        max_photons: Hilbert space truncation per mode.
        q: OAT parameter (only used for ``"oat"``).

    Returns:
        State vector in the two-mode Fock basis.

    Raises:
        ValueError: If state_type is unknown.
    """
    match state_type:
        case "sv":
            return input_state_factory(
                "squeezed_vacuum",
                round(resource_value),
                max_photons,
                r=float(np.arcsinh(np.sqrt(max(resource_value, 1.0)))),
            )
        case "tmsv":
            return make_two_mode_squeezed_vacuum(resource_value, max_photons)
        case "oat":
            return _make_oat_state(round(resource_value), q)
        case "noon":
            return input_state_factory("noon", round(resource_value), max_photons)
        case "twin_fock_std":
            return standard_twin_fock_state(round(resource_value), max_photons)
        case _:
            raise ValueError(f"Unknown state_type: {state_type}")


# ============================================================================
# BS1 Skip Logic
# ============================================================================


def _use_skip_bs1(state_type: str) -> bool:
    r"""Determine whether BS1 should be skipped for a given state type.

    SV skips BS1 because the squeezed vacuum is already the optimal probe
    (the analytical QFI formula :math:`F_Q = 2\langle N\rangle(\langle N\rangle+1)`
    holds for the SV state directly, not after a beam splitter).

    OAT and NOON also skip BS1 because BS1 is integrated into their preparation.
    TMSV does NOT skip BS1 --- the two-mode squeezed vacuum needs BS1 to generate
    a probe state with non-zero :math:`\text{Var}(J_z)`.
    """
    return state_type in ("sv", "oat", "noon")


# MziSensitivityDataSV is imported from src.analysis.sensitivity_metrics


# ============================================================================
# OAT q-Scan Result Dataclass
# ============================================================================


@dataclass
class OATQScanResult:
    r"""Result of an OAT q-parameter scan.

    Attributes:
        q_values: Array of scanned q values.
        fc_values: Array of :math:`F_C(q)` at each q.
        q_opt: q value that maximizes :math:`F_C`.
        fc_opt: Maximum :math:`F_C` value.
    """

    q_values: np.ndarray
    fc_values: np.ndarray
    q_opt: float
    fc_opt: float


# ============================================================================
# OAT Optimal q Scan
# ============================================================================


def _oat_optimal_q_estimate(N: int) -> float:
    r"""Estimate the optimal OAT parameter :math:`q_{\text{opt}}`.

    From Kitagawa-Ueda theory:
        :math:`q_{\text{opt}} \approx (6/N)^{1/3}`

    Args:
        N: Total particle number.

    Returns:
        Estimated optimal q.
    """
    return (6.0 / N) ** (1.0 / 3.0)


def _oat_q_grid(N: int, n_points: int = OAT_N_Q_POINTS) -> np.ndarray:
    r"""Generate a logarithmic q-grid centered around :math:`q_{\text{opt}}`.

    The grid spans :math:`[10^{-3}, 10^{1}]` with ``n_points`` values on a
    log scale.

    Args:
        N: Total particle number (used for centering).
        n_points: Number of q points.

    Returns:
        Array of q values.
    """
    q_opt = _oat_optimal_q_estimate(N)
    # Span 2 decades below and 2 decades above q_opt, clamped to [1e-3, 10]
    q_min = max(1e-3, q_opt / 10.0)
    q_max = min(10.0, q_opt * 10.0)
    return np.logspace(np.log10(q_min), np.log10(q_max), n_points)


def scan_oat_q(
    N: int,
    omega: float,
    max_photons: int,
    n_points: int = OAT_N_Q_POINTS,
    t_hold: float = t_hold,
) -> OATQScanResult:
    r"""Scan the OAT parameter q for a given N and ω to find the value
    that maximizes the Classical Fisher Information :math:`F_C`.

    Args:
        N: Total particle number.
        omega: Phase parameter :math:`\omega`.
        max_photons: Hilbert space truncation per mode.
        n_points: Number of q points to scan.
        t_hold: Holding time.

    Returns:
        OATQScanResult with ``q_values``, ``fc_values``, ``q_opt``, ``fc_opt``.
    """
    q_grid = _oat_q_grid(N, n_points)
    fc_values = np.zeros(n_points, dtype=float)

    bs = beam_splitter_unitary(np.pi / 4, 0.0, max_photons)

    for i, q in enumerate(q_grid):
        # Prepare OAT state with this q
        oat_state = _make_oat_state(N, q)

        # Evolve through MZI
        state_out = simple_mzi_evolution(
            oat_state,
            omega,
            max_photons,
            t_hold=t_hold,
            skip_bs1=True,
            bs=bs,
        )

        # Compute P(m|ω) and derivatives for CFI
        state_plus = simple_mzi_evolution(
            oat_state,
            omega + 1e-6,
            max_photons,
            t_hold=t_hold,
            skip_bs1=True,
            bs=bs,
        )
        state_minus = simple_mzi_evolution(
            oat_state,
            omega - 1e-6,
            max_photons,
            t_hold=t_hold,
            skip_bs1=True,
            bs=bs,
        )

        P_omega = output_number_diff_distribution(state_out, max_photons)
        P_plus = output_number_diff_distribution(state_plus, max_photons)
        P_minus = output_number_diff_distribution(state_minus, max_photons)

        fc_values[i] = classical_fisher_information_single(
            P_plus,
            P_minus,
            1e-6,
            p_at_theta=P_omega,
            prob_floor=1e-15,
        )

    # Find optimal q
    best_idx = int(np.argmax(fc_values))
    q_opt = float(q_grid[best_idx])
    fc_opt = float(fc_values[best_idx])

    return OATQScanResult(
        q_values=q_grid,
        fc_values=fc_values,
        q_opt=q_opt,
        fc_opt=fc_opt,
    )


# ============================================================================
# Generate Omega Scan (Single Resource Value)
# ============================================================================


def generate_single_omega_scan(
    state_type: str,
    resource_value: float,
    omega_grid: np.ndarray,
    max_photons: int | None = None,
    t_hold: float = t_hold,
    q: float | None = None,
) -> MziSensitivityDataSV:
    r"""Run a :math:`\omega` scan for a single resource value.

    .. note::

        For ``"oat"``, an explicit ``q > 0`` is required. If ``q`` is
        ``None``, a :exc:`ValueError` is raised. OAT sensitivity is
        independent of q because :math:`[\exp(-i q J_z^2), J_z] = 0`,
        so ``q=0`` (CSS baseline) is always the appropriate choice.

    Args:
        state_type: ``"sv"``, ``"tmsv"``, or ``"oat"``.
        resource_value: Resource parameter (mean N for SV/TMSV, N for OAT).
        omega_grid: :math:`\omega` values to scan.
        max_photons: Hilbert space truncation (auto-computed if None).
        t_hold: Holding time.
        q: OAT parameter (only used for ``"oat"``). Must be provided
            for ``"oat"`` states. Use ``q=0.0`` for the CSS baseline.

    Returns:
        MziSensitivityDataSV with one resource value.

    Raises:
        ValueError: If ``state_type="oat"`` and ``q`` is ``None``.
    """
    if max_photons is None:
        max_photons = resource_value_to_truncation(resource_value, state_type)

    # Determine resource type label
    if state_type in ("oat", "noon", "twin_fock_std"):
        resource_type = "N"
    else:
        resource_type = "mean_N"

    skip_bs1 = _use_skip_bs1(state_type)

    # For OAT, require an explicit q (sensitivity is q-independent
    # because [exp(-i q J_z^2), J_z] = 0).
    q_float: float = 0.0
    if state_type == "oat":
        if q is None:
            raise ValueError(
                "An explicit q parameter is required for OAT states. "
                "Use q=0.0 for the CSS baseline. "
                "OAT sensitivity is independent of q because "
                "exp(-i q J_z^2) commutes with J_z."
            )
        q_float = float(q)
    elif q is not None:
        q_float = float(q)

    # Prepare the state
    state = _prepare_state(state_type, resource_value, max_photons, q=q_float)

    # Pre-compute BS matrix once and reuse
    bs = beam_splitter_unitary(np.pi / 4, 0.0, max_photons)

    # Compute sensitivity grid
    result = compute_mzi_sensitivity_grid(
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
        squeezing_q_per_R=np.array([q], dtype=float),
    )


# ============================================================================
# Generate Full Data (All Resource Values)
# ============================================================================


def _generate_single_resource_data(
    state_type: str,
    resource_value: float,
    omega_grid: np.ndarray,
    t_hold: float = t_hold,
    q: float = 0.0,
) -> MziSensitivityDataSV | None:
    r"""Run omega scan for a single resource value, returning None on failure.

    Delegates to :func:`mzi_pipeline.safe_generate_scan` with a closure that computes
    the truncation and calls :func:`generate_single_omega_scan`.

    Args:
        state_type: ``"sv"``, ``"tmsv"``, or ``"oat"``.
        resource_value: Resource parameter.
        omega_grid: :math:`\omega` values to scan.
        t_hold: Holding time.
        q: OAT parameter (only used for ``"oat"``).

    Returns:
        MziSensitivityDataSV with one resource value, or None on failure.
    """

    def _gen_one(
        R: float, og: np.ndarray, t_hold: float
    ) -> MziSensitivityDataSV | None:
        max_photons = resource_value_to_truncation(R, state_type)
        result = generate_single_omega_scan(
            state_type,
            R,
            og,
            max_photons=max_photons,
            t_hold=t_hold,
            q=q,
        )
        # Free the beam-splitter matrix from the LRU cache after each R value.
        # Without this, all cached BS matrices (M=5..80) accumulate ~2.5 GB,
        # causing OOM crashes around SV R=14 (M=70).
        beam_splitter_unitary.cache_clear()
        # Force glibc to release freed memory back to the OS.  BS construction
        # (bs_fock) creates dense operators + dense expm, allocating 5-8 GB
        # temporarily.  Python's allocator does not return this to the OS,
        # so RSS grows monotonically across R values until OOM.
        import ctypes

        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass  # Non-glibc systems may lack malloc_trim
        return result

    return mzi_pipeline.safe_generate_scan(
        _gen_one,
        resource_value,
        omega_grid,
        t_hold,
        fail_label=f"{state_type} R={resource_value}",
    )


def _run_worker(
    state_type: str,
    R: float,
    omega_start: float,
    omega_end: float,
    omega_step: float,
    t_hold: float,
    output_path: Path,
) -> None:
    r"""Run a single R value in a **subprocess** to avoid OOM.

    Each worker process is a fresh Python interpreter; when it finishes, the OS
    fully reclaims all memory (BS construction temporaries, expm arenas, etc.).
    """
    # The worker script runs inline (-c) to avoid serialising large arrays.
    # Uses short-circuit ``and`` instead of ``if`` because ``-c`` scripts
    # cannot contain keyword statements after semicolons.
    # Use PYTHONPATH instead of sys.path.insert so the subprocess can
    # find this module via flat import (heisenberg_limit_mzi_sq_oat).
    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = str(Path(__file__).parent)
    script = (
        "import os,numpy as np;"
        f"os.environ.setdefault('MPLBACKEND','Agg');"
        f"from heisenberg_limit_mzi_sq_oat import generate_single_omega_scan;"
        f"from src.physics.mzi_simulation import beam_splitter_unitary;"
        f"from src.physics.hilbert_space import resource_value_to_truncation;"
        f"beam_splitter_unitary.cache_clear();"
        f"M=resource_value_to_truncation({R},{state_type!r});"
        f"og=np.arange({omega_start},{omega_end}+{omega_step}/2,{omega_step});"
        f"r=generate_single_omega_scan({state_type!r},{R},og,max_photons=M,t_hold={t_hold});"
        f"r is not None and r.save_parquet({str(output_path)!r})"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        env=worker_env,
        check=True,
        timeout=7200,
    )


def generate_full_data(
    state_type: str,
    resource_range: list[float] | list[int],
    omega_grid: np.ndarray,
    t_hold: float = t_hold,
) -> MziSensitivityDataSV:
    r"""Generate sensitivity data for all resource values in a range.

    For state types with :math:`M \ge 30` (SV and TMSV at large R), each R value
    is run in a **subprocess** to avoid OOM from BS construction (``bs_fock``
    creates dense operators + dense expm, allocating 5--8 GB temporarily, and
    glibc does not release the freed arenas back to the OS).

    Args:
        state_type: ``"sv"``, ``"tmsv"``, or ``"oat"``.
        resource_range: List of resource parameter values.
        omega_grid: :math:`\omega` values to scan.
        t_hold: Holding time.

    Returns:
        MziSensitivityDataSV with all resource values.
    """
    # OAT uses M ≤ 40 (BS ~45 MB), fine in-process.
    # SV/TMSV can use M up to 80 (BS ~688 MB, BS construction peak ~8 GB)
    # so we run each R value in a separate process for large M.
    if state_type == "oat":

        def _gen(
            R: float, og: np.ndarray, t_hold: float
        ) -> MziSensitivityDataSV | None:
            return _generate_single_resource_data(state_type, R, og, t_hold=t_hold)

        return mzi_pipeline.generate_full_data(
            state_type,
            resource_range,
            omega_grid,
            _gen,
            t_hold=t_hold,
        )

    # SV / TMSV: use subprocess workers
    omega_start = omega_grid[0]
    omega_step = omega_grid[1] - omega_grid[0]
    omega_end = omega_grid[-1]

    scan_results: list[MziSensitivityDataSV] = []
    tmpdir = Path(tempfile.mkdtemp())
    try:
        for idx, R in enumerate(resource_range):
            print(
                f"  Sweeping {state_type} R={R} ({idx + 1}/{len(resource_range)})...",
                flush=True,
            )
            tf = tmpdir / f"R_{R}.parquet"
            # Retry up to 3 times with a 60s pause; OOM-killed workers can
            # be transient when the system is under memory pressure.
            for attempt in range(3):
                try:
                    _run_worker(
                        state_type, R, omega_start, omega_end, omega_step, t_hold, tf
                    )
                    break
                except subprocess.CalledProcessError:
                    if attempt < 2:
                        import time as _time

                        _time.sleep(60)
                        print(f"  Retrying {state_type} R={R}...", flush=True)
                        continue
                    raise
            if tf.exists():
                scan_results.append(MziSensitivityDataSV.from_parquet(tf))
            # Brief pause between large-R workers so the OS can reclaim memory
            # and reduce OOM pressure from concurrent page reclaim / compaction.
            if R >= 10:
                import time as _time

                _time.sleep(30)

        if not scan_results:
            raise RuntimeError(f"No valid resource values for {state_type}")

        return mzi_pipeline.concatenate_scan_results(scan_results, state_type, t_hold)
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def _maybe_generate_full_data(
    st: str,
    r_range: list[float] | list[int],
    label: str,
    omega_grid: np.ndarray,
    force: bool,
    only: str | None,
    override_pq_path: Path | None = None,
) -> MziSensitivityDataSV | None:
    r"""Load or generate sensitivity data for one state type.

    Delegates to the report's :func:`generate_full_data` (with subprocess
    workers for SV/TMSV) instead of the shared
    :func:`mzi_pipeline.maybe_generate_full_data`, so that BS construction
    memory is fully reclaimed between R values.

    Args:
        st: State type key (e.g. ``"sv"``, ``"tmsv"``, ``"oat"``).
        r_range: List of resource values.
        label: Human-readable name for logging.
        omega_grid: :math:`\omega` grid.
        force: Re-generate even if Parquet exists.
        only: If set, only load/generate for matching state type.
        override_pq_path: If set, use this path instead of the default
            production path.  Intended for tests.

    Returns:
        MziSensitivityDataSV or None if filtered out by ``only``.
    """
    if only is not None and st != only:
        return None

    pq_path = override_pq_path or _parquet_path(f"{st}_sensitivity")

    if pq_path.exists() and not force:
        print(f"Loading existing data for {label} from {pq_path}")
        return MziSensitivityDataSV.from_parquet(pq_path)

    print(f"Generating {label} data (R={r_range[0]}..{r_range[-1]})")
    data = generate_full_data(st, r_range, omega_grid, t_hold=10.0)
    data.save_parquet(pq_path)
    print(f"  Saved to {pq_path}")
    return data


# ============================================================================
# Scaling Exponent Fitting
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
        save_path = _fig_path(f"{data.state_type}_delta_omega_comparison")
    return mzi_plots.plot_delta_omega_overlay(
        data,
        selected_R=selected_R,
        save_path=save_path,
        title=f"{data.state_type.upper()} --- Phase Sensitivity vs $\\omega$",
    )


def plot_scaling(
    data_list: list[MziSensitivityDataSV | None],
    labels: list[str],
    save_path: str | Path | None = None,
    N_min_fit: float = 4.0,
) -> Path:
    """Log-log plot of best Δω vs resource parameter with analytical QFI bounds and fits.

    Delegates to the shared :func:`mzi_plots.plot_scaling` with
    :func:`_fig_path` for default save-path resolution.

    Args:
        data_list: List of sensitivity data (entries may be None).
        labels: Display labels for each data entry.
        save_path: Output SVG path.
        N_min_fit: Minimum resource value for exponent fits.

    Returns:
        Path to saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("scaling_comparison")
    return mzi_plots.plot_scaling(
        data_list,
        labels,
        save_path=save_path,
        N_min_fit=N_min_fit,
    )


# ============================================================================
# Plot Orchestration
# ============================================================================


def _maybe_plot_delta_omega_overlays(
    results: dict[str, MziSensitivityDataSV],
    state_configs: list[tuple[str, list[float] | list[int], str]],
    force: bool,
    only: str | None,
) -> None:
    """Plot Δω overlay figures for each state type."""
    mzi_plots.maybe_plot_delta_omega_overlays(
        results,
        state_configs,
        force,
        only,
        _fig_path,
    )


def _maybe_plot_scaling_comparison(
    results: dict[str, MziSensitivityDataSV],
    force: bool,
) -> None:
    """Plot combined scaling comparison."""
    mzi_plots.maybe_plot_scaling_comparison(
        results,
        force,
        _fig_path,
        data_keys=["sv", "tmsv", "oat"],
        data_labels=["SV", "TMSV", "OAT"],
    )


def _generate_plots(
    results: dict[str, MziSensitivityDataSV],
    state_configs: list[tuple[str, list[float] | list[int], str]],
    force: bool,
    only: str | None,
) -> None:
    """Generate all plots from the computed sensitivity data."""
    mzi_plots.generate_plots(
        results,
        state_configs,
        force,
        only,
        _fig_path,
        scaling_data_keys=["sv", "tmsv", "oat"],
        scaling_data_labels=["SV", "TMSV", "OAT"],
    )


# ============================================================================
# Analytical QFI Verification Functions
# ============================================================================


def _verify_tmsv_qfi(mean_total: float, var_probe: float) -> bool:
    r"""Verify that Var(J_z) satisfies the TMSV analytical formula.

    :math:`\text{Var}(J_z)_{\text{probe}} = \langle N \rangle (\langle N \rangle + 2) / 4`

    Args:
        mean_total: Total mean photon number.
        var_probe: Computed variance of J_z from the probe state.

    Returns:
        True if the variance matches within tolerance.
    """
    expected_var = mean_total * (mean_total + 2.0) / 4.0
    return bool(np.isclose(var_probe, expected_var, rtol=1e-4))


def _verify_oat_q0_qfi(N: int, var_probe: float) -> bool:
    r"""Verify that OAT with q=0 (CSS) achieves SQL-level QFI.

    :math:`\text{Var}(J_z)_{\text{probe}} = N/4` for CSS.

    Args:
        N: Total particle number.
        var_probe: Computed variance of J_z from the probe state.

    Returns:
        True if the variance matches SQL within tolerance.
    """
    expected_var = N / 4.0
    return bool(np.isclose(var_probe, expected_var, rtol=1e-4))


# ============================================================================
# Main Pipeline
# ============================================================================


def generate_all(
    force: bool = False,
    only: str | None = None,
) -> dict[str, MziSensitivityDataSV]:
    """Generate all data and figures for the report.

    Args:
        force: If True, re-generate even if Parquet files exist.
        only: If set, only generate for this state type
            ("sv", "tmsv", or "oat").

    Returns:
        Dict mapping state_type to MziSensitivityDataSV.
    """
    omega_grid = np.arange(OMEGA_RANGE[0], OMEGA_RANGE[1] + OMEGA_STEP / 2, OMEGA_STEP)

    results: dict[str, MziSensitivityDataSV] = {}

    state_configs: list[tuple[str, list[float] | list[int], str]] = [
        ("sv", SV_N_RANGE, "Single-mode SV"),
        ("tmsv", TMSV_N_RANGE, "Two-mode SV"),
        ("oat", OAT_N_RANGE, "OAT"),
    ]

    for st, r_range, label in state_configs:
        data = _maybe_generate_full_data(st, r_range, label, omega_grid, force, only)
        if data is not None:
            results[st] = data

    _generate_plots(results, state_configs, force, only)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heisenberg-Limit MZI: Squeezed Vacuum & OAT sensitivity simulation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate data and figures even if files exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["sv", "tmsv", "oat"],
        default=None,
        help="Only generate data for this state type",
    )
    args = parser.parse_args()

    generate_all(force=args.force, only=args.only)


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    main()

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

This module is **not** importable as ``reports.20260625.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.analysis.fisher_information import classical_fisher_information_single
from src.analysis.scaling_fit import (
    fit_scaling_exponent,
)

# Re-export shared analysis functions for backward compatibility with local imports.
from src.analysis.sensitivity_metrics import (
    MziSensitivityData,
    analyse_best_worst_sensitivity,
)
from src.physics.mzi_distribution import output_number_diff_distribution
from src.physics.mzi_simulation import (
    beam_splitter_unitary,
    compute_mzi_sensitivity_grid,
    simple_mzi_evolution,
)
from src.physics.mzi_states import (
    input_state_factory,
    standard_twin_fock_state,
)
from src.utils.paths import report_path_fn

# Force non-interactive backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

sns.set_theme(style="whitegrid")

# ============================================================================
# Constants
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260625"
t_hold: float = 10.0  # Holding time
TRUNC_MULTIPLIER: float = 5.0  # Truncation multiplier for SV/TMSV
MAX_TRUNC: int = 80  # Maximum truncation per mode
OAT_N_Q_POINTS: int = 20  # Number of q points per OAT N value
OAT_CHI: float = 1.0  # OAT coupling strength

# Parameter sweep ranges
SV_N_RANGE: list[float] = [float(n) for n in range(1, 21)]  # ⟨N⟩ = 1..20
TMSV_N_RANGE: list[float] = [
    float(n) for n in range(2, 41, 2)
]  # Total ⟨N⟩ = 2..40 even
OAT_N_RANGE: list[int] = list(range(2, 41, 2))  # Even N=2..40
OMEGA_RANGE: tuple[float, float] = (0.1, 5.0)
OMEGA_STEP: float = 0.1

# Scaling fit
SV_ALPHA_EXPECTED: float = -1.0  # SV → Heisenberg
TMSV_ALPHA_EXPECTED: float = -1.0  # TMSV → Heisenberg
OAT_ALPHA_EXPECTED: float = -0.5  # OAT → SQL (invariant under J_z²)
ALPHA_TOL: float = 0.05  # Tolerance on fitted α for PASS/FAIL determination


# ============================================================================
# Path Helpers
# ============================================================================

_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ============================================================================
# Two-Mode Squeezed Vacuum State
# ============================================================================


def _make_two_mode_squeezed_vacuum(mean_total: float, max_photons: int) -> np.ndarray:
    r"""Create a two-mode squeezed vacuum state.

    The TMSV state is:
        :math:`|\psi\rangle = \sum_{n=0}^\infty \frac{\tanh^n(r)}{\cosh(r)} |n, n\rangle`

    The total mean photon number is :math:`\langle N \rangle = 2\sinh^2(r)`.

    Args:
        mean_total: Target total mean photon number :math:`\langle N \rangle`.
        max_photons: Maximum photon number per mode (truncation).

    Returns:
        Normalised state vector of dimension ``(max_photons+1)^2``.

    Raises:
        ValueError: If mean_total <= 0.
    """
    if mean_total <= 0:
        raise ValueError(f"Total mean photon number must be positive, got {mean_total}")
    r = float(np.arcsinh(np.sqrt(mean_total / 2.0)))
    dim_single = max_photons + 1
    dim = dim_single**2
    state = np.zeros(dim, dtype=complex)

    tanh_r = np.tanh(r)
    sech_r = 1.0 / np.cosh(r)

    for n in range(max_photons + 1):
        c_n = sech_r * (tanh_r**n)
        idx = n * dim_single + n  # |n, n⟩
        state[idx] = c_n

    # Check truncation convergence BEFORE normalisation
    raw_norm = np.linalg.norm(state)
    if raw_norm < 0.999:
        warnings.warn(
            f"TMSV truncation at M={max_photons} captures only "
            f"{raw_norm:.4f} of the total norm "
            f"(target mean_total={mean_total}). Increase max_photons.",
            stacklevel=2,
        )

    # Normalise (truncation may cause slight norm loss)
    if raw_norm > 0:
        state /= raw_norm

    return state


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


def _resource_value_to_truncation(resource_value: float, state_type: str) -> int:
    r"""Compute appropriate Hilbert space truncation for a given resource value.

    For SV: resource_value = mean photon number :math:`\langle N \rangle`.
    For TMSV: resource_value = total mean photon number :math:`\langle N \rangle_{\text{total}}`.
    For OAT: truncation = N (exact).

    Args:
        resource_value: The resource parameter.
        state_type: ``"sv"``, ``"tmsv"``, or ``"oat"``.

    Returns:
        Truncation M (max photons per mode).
    """
    if state_type == "oat":
        return int(resource_value)
    # For SV/TMSV: use truncation multiplier
    return min(int(np.ceil(TRUNC_MULTIPLIER * resource_value)), MAX_TRUNC)


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
            return _make_two_mode_squeezed_vacuum(resource_value, max_photons)
        case "oat":
            return _make_oat_state(round(resource_value), q)
        case "noon":
            return input_state_factory("noon", round(resource_value), max_photons)
        case "twin_fock_std":
            return standard_twin_fock_state(round(resource_value), max_photons)
        case _:
            raise ValueError(f"Unknown state_type: {state_type}")


# ============================================================================
# Analytical QFI Computation
# ============================================================================


def _compute_sv_qfi(mean_N: float, t_hold: float = t_hold) -> float:
    r"""Analytical QFI for single-mode squeezed vacuum.

    :math:`F_Q = 2 \cdot t_hold^2 \cdot \langle N \rangle (\langle N \rangle + 1)`

    Args:
        mean_N: Mean photon number :math:`\langle N \rangle = \sinh^2(r)`.
        t_hold: Holding time.

    Returns:
        Quantum Fisher information.
    """
    return 2.0 * t_hold**2 * mean_N * (mean_N + 1.0)


def _compute_tmsv_qfi(mean_total: float, t_hold: float = t_hold) -> float:
    r"""Analytical QFI for two-mode squeezed vacuum.

    :math:`F_Q = t_hold^2 \cdot \langle N \rangle (\langle N \rangle + 2)`

    where :math:`\langle N \rangle = 2\sinh^2(r)` is the total mean photon number.

    Args:
        mean_total: Total mean photon number :math:`\langle N \rangle`.
        t_hold: Holding time.

    Returns:
        Quantum Fisher information.
    """
    return t_hold**2 * mean_total * (mean_total + 2.0)


# ============================================================================
# Truncation Convergence Check
# ============================================================================


def _compute_sv_captured_norm(mean_N: float, max_photons: int) -> float:
    r"""Analytical norm captured by SV truncated at ``max_photons`` per mode.

    The single-mode squeezed vacuum :math:`S(r)\vert 0\rangle_1 \otimes \vert 0\rangle_2`
    only populates states :math:`\vert 2n, 0\rangle`.  The probability of the
    :math:`2n`-photon component satisfies the recurrence:

    .. math::

        P(0) = \frac{1}{\cosh(r)},\qquad
        P(2n) = \frac{2n-1}{2n}\,\tanh^{2}(r)\,P(2n-2)

    The captured norm is :math:`\sum_{n=0}^{\lfloor M/2\rfloor} P(2n)` where
    :math:`M =` ``max_photons`` and :math:`\langle N \rangle = \sinh^{2}(r)`.

    Args:
        mean_N: Mean photon number :math:`\langle N \rangle = \sinh^{2}(r)`.
        max_photons: Truncation per mode (maximum photon number).

    Returns:
        Fraction of total norm captured in :math:`[0, 1]`.
    """
    r = float(np.arcsinh(np.sqrt(mean_N)))
    tanh_sq = np.tanh(r) ** 2
    cosh_r = np.cosh(r)
    max_n = max_photons // 2  # only even n are populated

    prob = 1.0 / cosh_r  # P(0)
    captured = prob

    for n in range(1, max_n + 1):
        prob *= (2.0 * n - 1.0) / (2.0 * n) * tanh_sq
        captured += prob

    return float(captured)


def _check_truncation_convergence(
    state: np.ndarray | None = None,
    threshold: float = 0.999,
    *,
    mean_total: float | None = None,
    mean_n: float | None = None,
    max_photons: int | None = None,
) -> bool:
    r"""Check that the truncated Hilbert space captures enough norm.

    For SV states, the analytical truncation error is computed using the
    photon-number recurrence of the squeezed vacuum *before* renormalisation
    (which would otherwise hide the truncation loss).

    For TMSV states, the analytical truncation error is the geometric series:

    .. math::

        \sum_{n=0}^{M} \frac{\tanh^{2n}(r)}{\cosh^{2}(r)}
        = 1 - \tanh^{2(M+1)}(r)

    Args:
        state: State vector (only used as fallback when analytical parameters
            are not provided).
        threshold: Minimum captured fraction (default 0.999).
        mean_total: Total mean photon number for TMSV (analytical check).
        mean_n: Mean photon number for SV (analytical check).
        max_photons: Truncation per mode (required for analytical checks).

    Returns:
        True if the captured norm fraction >= threshold.

    Raises:
        ValueError: If neither analytical parameters nor state are provided.
    """
    if mean_n is not None and max_photons is not None:
        # Analytical check for SV before renormalisation
        captured = _compute_sv_captured_norm(mean_n, max_photons)
        return captured >= threshold
    if mean_total is not None and max_photons is not None:
        # Analytical check for TMSV before renormalisation
        r = float(np.arcsinh(np.sqrt(mean_total / 2.0)))
        tanh_r = np.tanh(r)
        captured = 1.0 - tanh_r ** (2 * (max_photons + 1))
        return captured >= threshold
    if state is not None:
        return bool(np.linalg.norm(state) >= threshold)
    raise ValueError(
        "Must provide (mean_n, max_photons) for SV, "
        "(mean_total, max_photons) for TMSV, or state."
    )


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


# ============================================================================
# MziSensitivityDataSV Dataclass
# ============================================================================


@dataclass
class MziSensitivityDataSV(MziSensitivityData):
    """Sensitivity data for squeezed-vacuum or OAT MZI.

    Inherits all fields, serialization, and Parquet I/O from the shared
    :class:`MziSensitivityData` in ``src.analysis.sensitivity_metrics``.
    """

    @classmethod
    def from_parquet(cls, path: str | Path) -> MziSensitivityDataSV:
        return super().from_parquet(path)  # type: ignore[return-value]


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
        max_photons = _resource_value_to_truncation(resource_value, state_type)

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

    Args:
        state_type: ``"sv"``, ``"tmsv"``, or ``"oat"``.
        resource_value: Resource parameter.
        omega_grid: :math:`\omega` values to scan.
        t_hold: Holding time.
        q: OAT parameter (only used for ``"oat"``).

    Returns:
        MziSensitivityDataSV with one resource value, or None on failure.
    """
    try:
        max_photons = _resource_value_to_truncation(resource_value, state_type)
        return generate_single_omega_scan(
            state_type,
            resource_value,
            omega_grid,
            max_photons=max_photons,
            t_hold=t_hold,
            q=q,
        )
    except (ValueError, AssertionError) as exc:
        print(f"Warning: resource={resource_value} failed for {state_type}: {exc}")
        return None


def _collect_metadata_per_r(
    scan_results: list[MziSensitivityDataSV],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract truncation_M and squeezing_q arrays from scan results."""
    trunc_Ms: list[float] = []
    sq_qs: list[float] = []
    for r in scan_results:
        trunc_val = (
            r.truncation_M_per_R[0] if r.truncation_M_per_R is not None else np.nan
        )
        sq_val = r.squeezing_q_per_R[0] if r.squeezing_q_per_R is not None else np.nan
        trunc_Ms.append(float(trunc_val))
        sq_qs.append(float(sq_val))
    return np.array(trunc_Ms, dtype=float), np.array(sq_qs, dtype=float)


def _concatenate_scan_results(
    scan_results: list[MziSensitivityDataSV],
    state_type: str,
    t_hold: float,
) -> MziSensitivityDataSV:
    """Concatenate a list of single-resource scan results."""
    resource_type = scan_results[0].resource_type
    omega_values = scan_results[0].omega_values
    trunc_Ms, sq_qs = _collect_metadata_per_r(scan_results)

    def _cat(field: str) -> np.ndarray:
        return np.concatenate([getattr(r, field) for r in scan_results])

    return MziSensitivityDataSV(
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


def generate_full_data(
    state_type: str,
    resource_range: list[float] | list[int],
    omega_grid: np.ndarray,
    t_hold: float = t_hold,
) -> MziSensitivityDataSV:
    r"""Generate sensitivity data for all resource values in a range.

    Args:
        state_type: ``"sv"``, ``"tmsv"``, or ``"oat"``.
        resource_range: List of resource parameter values.
        omega_grid: :math:`\omega` values to scan.
        t_hold: Holding time.

    Returns:
        MziSensitivityDataSV with all resource values.
    """
    scan_results: list[MziSensitivityDataSV] = []
    for idx, R in enumerate(resource_range):
        print(
            f"  Sweeping {state_type} R={R} ({idx + 1}/{len(resource_range)})...",
            flush=True,
        )
        scan = _generate_single_resource_data(
            state_type, float(R), omega_grid, t_hold=t_hold
        )
        if scan is not None:
            scan_results.append(scan)

    if not scan_results:
        raise RuntimeError(f"No valid resource values for state_type={state_type}")

    return _concatenate_scan_results(scan_results, state_type, t_hold)


def _maybe_generate_full_data(
    st: str,
    r_range: list[float] | list[int],
    label: str,
    omega_grid: np.ndarray,
    force: bool,
    only: str | None,
) -> MziSensitivityDataSV | None:
    r"""Load or generate sensitivity data for one state type.

    Args:
        st: State type key (e.g. ``"sv"``, ``"tmsv"``, ``"oat"``).
        r_range: List of resource values.
        label: Human-readable name for logging.
        omega_grid: :math:`\omega` grid.
        force: Re-generate even if Parquet exists.
        only: If set, only load/generate for matching state type.

    Returns:
        MziSensitivityDataSV or None if filtered out by ``only``.
    """
    if only is not None and st != only:
        return None

    pq_path = _parquet_path(f"{st}_sensitivity")
    if pq_path.exists() and not force:
        print(f"Loading existing data for {label} from {pq_path}")
        return MziSensitivityDataSV.from_parquet(pq_path)

    print(f"Generating {label} sensitivity data (R={r_range[0]}..{r_range[-1]})")
    data = generate_full_data(st, r_range, omega_grid, t_hold=t_hold)
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

    Args:
        data: Sensitivity data containing all resource values.
        selected_R: Which resource values to include (auto-selected if None).
        save_path: Output SVG path. Auto-generated if None.

    Returns:
        Path to saved SVG.
    """
    if selected_R is None:
        n_R = len(data.resource_values)
        step = max(1, n_R // 7)
        selected_R = [float(data.resource_values[i]) for i in range(0, n_R, step)]

    if save_path is None:
        save_path = _fig_path(f"{data.state_type}_delta_omega_comparison")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_label = data.state_type.upper()
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.colormaps["viridis"]
    colors = cmap(np.linspace(0.15, 0.85, len(selected_R)))

    for idx, R_val in enumerate(selected_R):
        match = np.where(np.isclose(data.resource_values, R_val, rtol=1e-10))[0]
        if len(match) == 0:
            continue
        r_idx = match[0]
        omega = data.omega_values
        dt_c = data.delta_omega_c_grid[r_idx, :]
        dt_q = data.delta_omega_q_per_R[r_idx]

        c_finite = np.isfinite(dt_c)
        if np.any(c_finite):
            ax.semilogy(
                omega[c_finite],
                dt_c[c_finite],
                color=colors[idx],
                linewidth=1.5,
                label=rf"R={R_val}  $\Delta\omega_{{\mathrm{{C}}}}$",
            )

        ax.axhline(
            y=float(dt_q),
            color=colors[idx],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title(rf"{state_label} — Phase Sensitivity vs $\omega$")
    ax.legend(fontsize=8, loc="best", ncol=1)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scaling(
    data_list: list[MziSensitivityDataSV | None],
    labels: list[str],
    save_path: str | Path | None = None,
    N_min_fit: float = 4.0,
) -> Path:
    """Log-log plot of best Δω vs resource parameter with analytical QFI bounds and fits.

    Overlays multiple state types on a single figure.

    Args:
        data_list: List of sensitivity data (entries may be None).
        labels: Display labels for each data entry.
        save_path: Output SVG path.
        N_min_fit: Minimum resource value for exponent fits.

    Returns:
        Path to saved SVG.
    """
    if all(d is None for d in data_list):
        raise ValueError("At least one data set must be provided")

    if save_path is None:
        save_path = _fig_path("scaling_comparison")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Reference lines — use a generic resource axis
    R_ref = np.logspace(0, 1.5, 50)
    ax.plot(
        R_ref,
        1.0 / (t_hold * R_ref),
        "k--",
        alpha=0.4,
        label=r"$\propto 1/R$ (Heisenberg)",
    )
    ax.plot(
        R_ref,
        1.0 / (t_hold * np.sqrt(R_ref)),
        "k:",
        alpha=0.4,
        label=r"$\propto 1/\sqrt{R}$ (SQL)",
    )

    colours = ["C0", "C1", "C2", "C3", "C4"]
    markers = ["o", "s", "^", "D", "v"]

    for i, (data, label) in enumerate(zip(data_list, labels, strict=False)):
        if data is None:
            continue
        colour = colours[i % len(colours)]
        marker = markers[i % len(markers)]

        # QFI bound
        ax.loglog(
            data.resource_values,
            data.delta_omega_q_per_R,
            f"{colour}--",
            alpha=0.5,
            label=f"{label} QFI bound",
        )

        # Best Δω_C at each resource value
        analysis = analyse_best_worst_sensitivity(
            data.resource_values,
            data.omega_values,
            data.delta_omega_c_grid,
        )
        R_vals = analysis["resource_values"]
        best_dt_c = analysis["best_sensitivity"]
        finite = np.isfinite(best_dt_c)
        if np.any(finite):
            ax.loglog(
                R_vals[finite],
                best_dt_c[finite],
                f"{colour}{marker}-",
                label=rf"{label} best $\Delta\omega_{{\mathrm{{C}}}}$",
            )

        # Fit exponent
        best_c_finite = np.array(best_dt_c, dtype=float)
        fit_result = fit_scaling_exponent(
            np.array(R_vals, dtype=float), best_c_finite, min_N=int(N_min_fit)
        )
        if fit_result.valid:
            N_fit = fit_result.N_values
            delta_fit = fit_result.C * N_fit**fit_result.alpha
            ax.loglog(
                N_fit,
                delta_fit,
                f"{colour}--",
                alpha=0.7,
                linewidth=1.5,
                label=f"{label}: "
                rf"$\alpha = {fit_result.alpha:.3f}$",
            )

    ax.set_xlabel("Resource parameter $R$")
    ax.set_ylabel(r"$\Delta\omega$")
    ax.set_title("Phase Sensitivity Scaling in Standard MZI")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


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
    for st, _r_range, _label in state_configs:
        if only is not None and st != only:
            continue
        data = results.get(st)
        if data is None:
            continue
        overlay_path = _fig_path(f"{st}_delta_omega_comparison")
        if not overlay_path.exists() or force:
            plot_delta_omega_overlay(data, save_path=overlay_path)
            print(f"  Plotted {overlay_path}")


def _maybe_plot_scaling_comparison(
    results: dict[str, MziSensitivityDataSV],
    force: bool,
) -> None:
    """Plot combined scaling comparison."""
    path = _fig_path("scaling_comparison")
    if not path.exists() or force:
        plot_scaling(
            [
                results.get("sv"),
                results.get("tmsv"),
                results.get("oat"),
            ],
            ["SV", "TMSV", "OAT"],
            save_path=path,
        )
        print(f"  Plotted {path}")


def _generate_plots(
    results: dict[str, MziSensitivityDataSV],
    state_configs: list[tuple[str, list[float] | list[int], str]],
    force: bool,
    only: str | None,
) -> None:
    """Generate all plots from the computed sensitivity data."""
    _maybe_plot_delta_omega_overlays(results, state_configs, force, only)
    _maybe_plot_scaling_comparison(results, force)


# ============================================================================
# Analytical QFI Verification Functions
# ============================================================================


def _verify_sv_qfi(mean_N: float, var_probe: float) -> bool:
    r"""Verify that Var(J_z) satisfies the SV analytical formula.

    :math:`\text{Var}(J_z)_{\text{probe}} = \langle N \rangle (\langle N \rangle + 1) / 2`

    Args:
        mean_N: Mean photon number.
        var_probe: Computed variance of J_z from the probe state.

    Returns:
        True if the variance matches the analytical formula within tolerance.
    """
    expected_var = mean_N * (mean_N + 1.0) / 2.0
    return bool(np.isclose(var_probe, expected_var, rtol=1e-4))


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

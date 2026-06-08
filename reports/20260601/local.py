r"""
Local module for the 2026-06-01 Heisenberg-Limit MZI: NOON & Twin-Fock report.

Contains all code exclusive to this report:
- Standard Twin-Fock |N/2, N/2⟩ state (unlike the uniform-superposition Twin-Fock in src/)
- Simplified ancilla-free MZI evolution (BS1 → phase → BS2)
- Sensitivity computation via Classical Fisher Information (CFI) from
  the full number-difference distribution P(m|θ), plus QFI bound
- Parquet-serializable dataclass for grid results
- Log-log scaling exponent fitting
- Plot functions for Δθ vs θ and Δθ vs N
- CLI pipeline for generating all data and figures

Usage:
    uv run python reports/20260601/local.py --force
    uv run python reports/20260601/local.py --only noon
    uv run python reports/20260601/local.py --only twin_fock_std

This module is **not** importable as ``reports.20260601.local`` (the directory
name contains hyphens).  Instead, importers add the report directory to
``sys.path`` and do ``import local``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.fisher_information import (
    classical_fisher_information_single,
)
from src.analysis.scaling_fit import (
    ScalingFitResult,
)
from src.analysis.scaling_fit import (
    fit_scaling_exponent as module_fit_scaling_exponent,
)
from src.physics.mzi_simulation import beam_splitter_unitary
from src.physics.mzi_states import (
    input_state_factory,
    two_mode_jz_operator,
)

# Force non-interactive backend before any plotting imports.
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

sns.set_theme(style="whitegrid")

# ============================================================================
# Constants
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260601"
H_T: float = 10.0  # Holding time
BS_THETA: float = np.pi / 4  # 50/50 beam splitter
BS_PHI: float = 0.0  # Beam splitter phase
CFI_EPSILON: float = 1e-6  # Finite-difference step for CFI derivative
EP_EPSILON: float = 1e-6  # Finite-difference step for error-propagation derivative
EP_DERIV_REL_FLOOR: float = 1e-8  # Relative floor for |d⟨J_z⟩/dθ| to avoid /0
PROB_FLOOR: float = 1e-15  # Minimum probability for CFI denominator

# Parameter sweep ranges
NOON_N_RANGE: list[int] = list(range(1, 41))  # 1..40
TF_N_RANGE: list[int] = list(range(2, 41, 2))  # Even 2..40
THETA_RANGE: tuple[float, float] = (0.1, 5.0)
THETA_STEP: float = 0.1

# Scaling fit
ALPHA_EXPECTED_NOON: float = -1.0
ALPHA_TOL: float = 0.02


# ============================================================================
# Path Helpers
# ============================================================================


def _parquet_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "raw_data" / f"{REPORT_DATE}-{name}.parquet"


def _fig_path(name: str) -> Path:
    return REPORTS_DIR / REPORT_DATE / "figures" / f"{REPORT_DATE}-{name}.svg"


# ============================================================================
# Standard Twin-Fock |N/2, N/2⟩ State
# ============================================================================


def _make_standard_twin_fock_state(N: int, max_photons: int) -> np.ndarray:
    r"""Create the standard Twin-Fock state :math:`|N/2, N/2\rangle`.

    This is a single two-mode Fock state with exactly N/2 photons in each
    mode. It differs from the uniform-superposition Twin-Fock in
    ``src.physics.mzi_states.twin_fock_state``.

    Args:
        N: Total photon number (must be even).
        max_photons: Maximum photon number per mode (Hilbert space truncation).

    Returns:
        Normalised state vector of dimension ``(max_photons+1)^2``.

    Raises:
        ValueError: If N is odd (|N/2, N/2⟩ not defined).
    """
    if N % 2 != 0:
        raise ValueError(f"Standard Twin-Fock |N/2,N/2⟩ requires even N, got N={N}")
    dim = (max_photons + 1) ** 2
    state = np.zeros(dim, dtype=complex)
    n = N // 2
    idx = n * (max_photons + 1) + n  # |N/2, N/2⟩
    state[idx] = 1.0
    return state


# ============================================================================
# State Preparation Dispatch
# ============================================================================


def _prepare_state(state_type: str, N: int, max_photons: int) -> np.ndarray:
    """Dispatch state creation by type.

    Args:
        state_type: ``"noon"`` or ``"twin_fock_std"``.
        N: Total photon number.
        max_photons: Hilbert space truncation per mode.

    Returns:
        State vector in the two-mode Fock basis.

    Raises:
        ValueError: If state_type is unknown.
    """
    match state_type:
        case "noon":
            return input_state_factory("noon", N, max_photons)
        case "twin_fock_std":
            return _make_standard_twin_fock_state(N, max_photons)
        case _:
            raise ValueError(f"Unknown state_type: {state_type}")


# ============================================================================
# Simplified Ancilla-Free MZI Evolution
# ============================================================================


def _apply_phase_shift(
    state: np.ndarray,
    phi: float,
    max_photons: int,
) -> np.ndarray:
    r"""Apply :math:`\exp(i \phi n_2)` as an element-wise multiplication.

    This is O(d) instead of O(d²) required by a full matrix-vector multiply,
    and avoids the O(d³) unitarity assertion inside ``phase_shift_unitary``.

    Args:
        state: State vector in the two-mode Fock basis, shape ``((M+1)^2,)``.
        phi: Phase angle in radians.
        max_photons: Maximum photon number per mode.

    Returns:
        State with phase applied (mutated copy).
    """
    dim = max_photons + 1
    state_2d = state.reshape(dim, dim)
    # Phase = exp(i·phi·n₂), broadcast over columns (n₂ index)
    n2 = np.arange(dim, dtype=float)
    phase_factors = np.exp(1j * phi * n2)  # shape (dim,)
    state_2d *= phase_factors[None, :]  # broadcast over rows
    return state_2d.ravel()


def simple_mzi_evolution(
    initial_state: np.ndarray,
    theta: float,
    max_photons: int,
    H_t: float = H_T,
    skip_bs1: bool = False,
    bs: np.ndarray | None = None,
) -> np.ndarray:
    r"""Evolve a state through a standard MZI with no ancilla.

    Circuit: BS1(:math:`\pi/4`) → Phase(:math:`H_t \cdot \theta`) → BS2(:math:`\pi/4`)

    When ``skip_bs1=True``, the first BS is omitted (the input state is already
    path-entangled and is used directly as the probe). This is used for the NOON
    state, which is its own optimal probe.

    The phase shift is :math:`\exp(i \cdot H_t \cdot \theta \cdot n_2)`, which
    produces the same relative phase as :math:`\exp(-i \theta H_t J_z)` up to
    an irrelevant global phase.

    The phase shift is applied as an O(d) element-wise multiplication (rather than
    calling :func:`phase_shift_unitary`) to avoid O(d³) unitarity checks.

    Args:
        initial_state: Input state in the two-mode Fock basis.
        theta: Unknown phase parameter :math:`\theta`.
        max_photons: Maximum photon number per mode.
        H_t: Holding time (sensitivity amplification factor).
        skip_bs1: If True, omit the first beam splitter (NOON convention).
        bs: Pre-computed beam-splitter unitary. If None, computed fresh.

    Returns:
        Output state after the full MZI circuit.
    """
    if bs is None:
        bs = beam_splitter_unitary(BS_THETA, BS_PHI, max_photons)
    phi = theta * H_t

    if skip_bs1:
        state = _apply_phase_shift(initial_state.copy(), phi, max_photons)
    else:
        state = bs @ initial_state  # BS1
        state = _apply_phase_shift(state, phi, max_photons)  # Phase shift
    return bs @ state  # BS2


# ============================================================================
# Number-Difference Distribution P(m|θ)
# ============================================================================


def output_number_diff_distribution(
    state_out: np.ndarray,
    max_photons: int,
) -> np.ndarray:
    r"""Compute :math:`P(m|\theta)` where :math:`m = n_1 - n_2`.

    From the output state, collect the probability of each number-difference
    outcome :math:`m \in \{-M, \dots, M\}` with :math:`M = \text{max\_photons}`.

    Args:
        state_out: Output state vector in the two-mode Fock basis.
        max_photons: Maximum photon number per mode.

    Returns:
        Array of shape ``(2 * max_photons + 1,)`` indexed by ``m + max_photons``.
    """
    n_outcomes = 2 * max_photons + 1
    P = np.zeros(n_outcomes, dtype=float)
    offset = max_photons
    for n1 in range(max_photons + 1):
        for n2 in range(max_photons + 1):
            idx = n1 * (max_photons + 1) + n2
            prob = np.real(state_out[idx].conj() * state_out[idx])
            m = n1 - n2
            P[m + offset] += prob
    return P


# ============================================================================
# Classical Fisher Information from P(m|θ)
# ============================================================================


def compute_fisher_classical(
    P_theta: np.ndarray,
    P_plus: np.ndarray,
    P_minus: np.ndarray,
    epsilon: float = CFI_EPSILON,
) -> float:
    r"""Compute the Classical Fisher Information at a single :math:`\theta`.

    Delegates to :func:`src.analysis.fisher_information.classical_fisher_information_single`
    with ``p_at_theta=P_theta`` for the textbook denominator convention.

    .. math::

        F_C(\theta) = \sum_m \frac{(\partial P(m|\theta)/\partial\theta)^2}{P(m|\theta)}

    Args:
        P_theta: :math:`P(m|\theta)` — distribution at the evaluation point.
        P_plus: :math:`P(m|\theta+\varepsilon)` — distribution at forward point.
        P_minus: :math:`P(m|\theta-\varepsilon)` — distribution at backward point.
        epsilon: Finite-difference step.

    Returns:
        Classical Fisher information :math:`F_C(\theta)`.
    """
    return classical_fisher_information_single(
        P_plus,
        P_minus,
        epsilon,
        p_at_theta=P_theta,
        prob_floor=PROB_FLOOR,
    )


# ============================================================================
# MZI Sensitivity Grid Computation (with CFI)
# ============================================================================


def compute_mzi_sensitivity_grid(
    initial_state: np.ndarray,
    theta_grid: np.ndarray,
    max_photons: int,
    H_t: float = H_T,
    skip_bs1: bool = False,
) -> dict[str, np.ndarray | float]:
    r"""Compute :math:`\Delta\theta_C`, :math:`\Delta\theta_{\text{EP}}` and
    :math:`\Delta\theta_Q` across a :math:`\theta` grid.

    For each :math:`\theta_i`:
        1. Evolve the state through the MZI
        2. Compute :math:`P(m|\theta_i)` from the output state
        3. Compute :math:`\langle J_z\rangle_{\text{out}}` and
           :math:`\text{Var}(J_z)_{\text{out}}`
        4. Compute :math:`\partial P(m|\theta)/\partial\theta` via
           central finite differences with step :math:`\varepsilon = 10^{-6}`
        5. :math:`F_C = \sum_m (\partial P/\partial\theta)^2 / P`
        6. :math:`\Delta\theta_C = 1/\sqrt{F_C}`

    The QFI bound :math:`\Delta\theta_Q = 1/\sqrt{F_Q}` is computed from the
    probe state using :math:`F_Q = 4 H_t^2 \text{Var}(J_z)_{\text{probe}}`,
    independent of :math:`\theta`. When ``skip_bs1=True``, the probe state
    is the initial state itself (used for NOON, which is already path-entangled).

    Args:
        initial_state: Input state in the two-mode Fock basis.
        theta_grid: Array of :math:`\theta` values to evaluate.
        max_photons: Maximum photon number per mode.
        H_t: Holding time.
        skip_bs1: If True, omit BS1 from both probe and evolution (NOON convention).

    Returns:
        Dictionary with keys:
        - ``theta_values``: The input :math:`\theta` grid.
        - ``expectation_values``: :math:`\langle J_z\rangle_{\text{out}}`.
        - ``variance_values``: :math:`\text{Var}(J_z)_{\text{out}}`.
        - ``derivative_values``: :math:`\partial\langle J_z\rangle/\partial\theta`.
        - ``delta_theta_ep``: :math:`\Delta\theta_{\text{EP}}` (error propagation).
        - ``delta_theta_q``: :math:`\Delta\theta_Q` (scalar, :math:`\theta`-independent).
        - ``fisher_quantum``: :math:`F_Q` (scalar).
        - ``fisher_classical``: :math:`F_C(\theta)` (array, primary sensitivity metric).
        - ``delta_theta_c``: :math:`\Delta\theta_C(\theta)` (array).
    """
    n_theta = len(theta_grid)
    jz = two_mode_jz_operator(max_photons)
    jz2 = jz @ jz

    # Pre-compute beam splitter (same for all θ)
    bs = beam_splitter_unitary(BS_THETA, BS_PHI, max_photons)

    # Compute the probe state for QFI bound
    if skip_bs1:
        probe_state = initial_state.copy()
    else:
        probe_state = bs @ initial_state
    mean_probe = np.conj(probe_state) @ jz @ probe_state
    mean_sq_probe = np.conj(probe_state) @ jz2 @ probe_state
    var_probe = float(np.real(mean_sq_probe - mean_probe**2))
    fq = 4.0 * H_t**2 * var_probe
    delta_theta_q = 1.0 / np.sqrt(fq) if fq > 0 else float("inf")

    # Evolve state at each θ and compute statistics
    expectation_values = np.zeros(n_theta, dtype=float)
    variance_values = np.zeros(n_theta, dtype=float)
    derivative_values = np.zeros(n_theta, dtype=float)
    fisher_classical = np.full(n_theta, np.nan, dtype=float)

    # Cache P(m|θ_i) for each θ_i
    P_grid = np.zeros((n_theta, 2 * max_photons + 1), dtype=float)

    def _jz_expectation(state: np.ndarray) -> float:
        r"""Inline ⟨ψ|J_z|ψ⟩ using precomputed jz (avoids O(d²) reconstruction)."""
        return float(np.real(np.conj(state) @ jz @ state))

    def _jz_variance(state: np.ndarray) -> float:
        r"""Inline Var(J_z) using precomputed jz, jz2 (avoids O(d³) jz@jz)."""
        mean = np.conj(state) @ jz @ state
        mean_sq = np.conj(state) @ jz2 @ state
        return float(np.real(mean_sq - mean**2))

    for i, theta in enumerate(theta_grid):
        phi = theta * H_t
        if skip_bs1:
            state = _apply_phase_shift(initial_state.copy(), phi, max_photons)
        else:
            state = bs @ initial_state  # BS1
            state = _apply_phase_shift(state, phi, max_photons)  # Phase
        state = bs @ state  # BS2

        # J_z statistics (use inlined helpers to avoid O(d³) jz@jz)
        exp_val = _jz_expectation(state)
        var_val = _jz_variance(state)

        expectation_values[i] = exp_val
        variance_values[i] = var_val

        # Full distribution P(m|θ)
        P_grid[i] = output_number_diff_distribution(state, max_photons)

        # Evolve at θ ± ε for both CFI and EP derivative
        state_plus = simple_mzi_evolution(
            initial_state,
            theta + CFI_EPSILON,
            max_photons,
            H_t=H_t,
            skip_bs1=skip_bs1,
            bs=bs,
        )
        state_minus = simple_mzi_evolution(
            initial_state,
            theta - CFI_EPSILON,
            max_photons,
            H_t=H_t,
            skip_bs1=skip_bs1,
            bs=bs,
        )

        # EP derivative from θ±ε states (consistent small-ε step)
        exp_plus = _jz_expectation(state_plus)
        exp_minus = _jz_expectation(state_minus)
        derivative_values[i] = (exp_plus - exp_minus) / (2.0 * CFI_EPSILON)

        # CFI from P(m|θ±ε) distributions
        P_plus = output_number_diff_distribution(state_plus, max_photons)
        P_minus = output_number_diff_distribution(state_minus, max_photons)
        fisher_classical[i] = compute_fisher_classical(
            P_grid[i],
            P_plus,
            P_minus,
            epsilon=CFI_EPSILON,
        )

    # Error-propagation sensitivity (secondary metric)
    abs_deriv = np.abs(derivative_values)
    max_exp = np.max(np.abs(expectation_values))
    min_deriv = EP_DERIV_REL_FLOOR * max_exp if max_exp > 0 else 1e-12
    abs_deriv = np.maximum(abs_deriv, min_deriv)

    delta_theta_ep = np.sqrt(variance_values) / abs_deriv
    delta_theta_c = 1.0 / np.sqrt(np.maximum(fisher_classical, 1e-300))

    return {
        "theta_values": theta_grid,
        "expectation_values": expectation_values,
        "variance_values": variance_values,
        "derivative_values": derivative_values,
        "delta_theta_ep": delta_theta_ep,
        "delta_theta_q": float(delta_theta_q),
        "fisher_quantum": float(fq),
        "fisher_classical": fisher_classical,
        "delta_theta_c": delta_theta_c,
    }


# ============================================================================
# MziSensitivityData Dataclass
# ============================================================================


@dataclass
class MziSensitivityData:
    r"""All sensitivity data for one state type across :math:`N` and :math:`\theta`.

    Stores a 2D grid indexed by ``(N, theta)``, plus per-:math:`N` QFI bounds.

    The primary sensitivity metric is :math:`\Delta\theta_C` (Classical Fisher
    Information from the full :math:`P(m|\theta)` distribution). The error-
    propagation :math:`\Delta\theta_{\text{EP}}` is retained as a secondary
    diagnostic.

    Attributes:
        state_type: ``"noon"`` or ``"twin_fock_std"``.
        N_values: Array of :math:`N` values, shape ``(n_N,)``.
        theta_values: Array of :math:`\theta` values, shape ``(n_theta,)``.
        expectation_grid: :math:`\langle J_z\rangle` at each ``(N, theta)``.
        variance_grid: :math:`\text{Var}(J_z)` at each ``(N, theta)``.
        derivative_grid: :math:`\partial\langle J_z\rangle/\partial\theta`.
        delta_theta_ep_grid: :math:`\Delta\theta_{\text{EP}}`.
        delta_theta_q_per_N: :math:`\Delta\theta_Q` per :math:`N` (length ``n_N``).
        fisher_classical_grid: :math:`F_C` at each ``(N, theta)``.
        delta_theta_c_grid: :math:`\Delta\theta_C` at each ``(N, theta)``.
        H_t: Holding time.
    """

    state_type: str
    N_values: np.ndarray
    theta_values: np.ndarray
    expectation_grid: np.ndarray
    variance_grid: np.ndarray
    derivative_grid: np.ndarray
    delta_theta_ep_grid: np.ndarray
    delta_theta_q_per_N: np.ndarray
    fisher_classical_grid: np.ndarray
    delta_theta_c_grid: np.ndarray
    H_t: float = H_T

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to long-format DataFrame (one row per N, θ combination)."""
        n_N = len(self.N_values)
        n_theta = len(self.theta_values)
        rows: list[dict] = []
        for i in range(n_N):
            for j in range(n_theta):
                dt_ep = float(self.delta_theta_ep_grid[i, j])
                dt_c = float(self.delta_theta_c_grid[i, j])
                rows.append(
                    {
                        "state_type": self.state_type,
                        "N": int(self.N_values[i]),
                        "theta": float(self.theta_values[j]),
                        "expectation": float(self.expectation_grid[i, j]),
                        "variance": float(self.variance_grid[i, j]),
                        "derivative": float(self.derivative_grid[i, j]),
                        "delta_theta_ep": (
                            dt_ep if np.isfinite(dt_ep) else float("inf")
                        ),
                        "delta_theta_q": float(self.delta_theta_q_per_N[i]),
                        "fisher_classical": float(self.fisher_classical_grid[i, j]),
                        "delta_theta_c": (dt_c if np.isfinite(dt_c) else float("inf")),
                        "H_t": self.H_t,
                    }
                )
        return pd.DataFrame(rows)

    def save_parquet(self, path: str | Path) -> Path:
        """Save to Parquet via long-format DataFrame."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> MziSensitivityData:
        """Load from Parquet, reconstructing the 2D grids."""
        df = pd.read_parquet(path)
        required = {
            "state_type",
            "N",
            "theta",
            "expectation",
            "variance",
            "derivative",
            "delta_theta_ep",
            "delta_theta_q",
            "fisher_classical",
            "delta_theta_c",
            "H_t",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet at {path} is missing required columns: "
                f"{sorted(missing)}. Regenerate the file with the current code."
            )
        state_type = str(df["state_type"].iloc[0])
        H_t = float(df["H_t"].iloc[0])
        N_vals = sorted(df["N"].unique())
        theta_vals = sorted(df["theta"].unique())
        n_N = len(N_vals)
        n_theta = len(theta_vals)

        # Build lookup for reconstruction
        expectation_grid = np.full((n_N, n_theta), np.nan, dtype=float)
        variance_grid = np.full((n_N, n_theta), np.nan, dtype=float)
        derivative_grid = np.full((n_N, n_theta), np.nan, dtype=float)
        delta_theta_ep_grid = np.full((n_N, n_theta), np.nan, dtype=float)
        delta_theta_q_per_N = np.full(n_N, np.nan, dtype=float)
        fisher_classical_grid = np.full((n_N, n_theta), np.nan, dtype=float)
        delta_theta_c_grid = np.full((n_N, n_theta), np.nan, dtype=float)

        for _, row in df.iterrows():
            n_idx = N_vals.index(int(row["N"]))
            t_idx = theta_vals.index(float(row["theta"]))
            expectation_grid[n_idx, t_idx] = row["expectation"]
            variance_grid[n_idx, t_idx] = row["variance"]
            derivative_grid[n_idx, t_idx] = row["derivative"]
            delta_theta_ep_grid[n_idx, t_idx] = row["delta_theta_ep"]
            # All rows for the same N must have the same QFI bound
            dq = float(row["delta_theta_q"])
            if np.isnan(delta_theta_q_per_N[n_idx]):
                delta_theta_q_per_N[n_idx] = dq
            elif not np.isclose(delta_theta_q_per_N[n_idx], dq, rtol=1e-10):
                raise ValueError(
                    f"Inconsistent delta_theta_q for N={row['N']}: "
                    f"expected {delta_theta_q_per_N[n_idx]}, got {dq}. "
                    f"Regenerate the file."
                )
            fisher_classical_grid[n_idx, t_idx] = float(row["fisher_classical"])
            delta_theta_c_grid[n_idx, t_idx] = float(row["delta_theta_c"])

        return cls(
            state_type=state_type,
            N_values=np.array(N_vals, dtype=int),
            theta_values=np.array(theta_vals, dtype=float),
            expectation_grid=expectation_grid,
            variance_grid=variance_grid,
            derivative_grid=derivative_grid,
            delta_theta_ep_grid=delta_theta_ep_grid,
            delta_theta_q_per_N=delta_theta_q_per_N,
            fisher_classical_grid=fisher_classical_grid,
            delta_theta_c_grid=delta_theta_c_grid,
            H_t=H_t,
        )


# ============================================================================
# Generate Theta Scan (Single N)
# ============================================================================


def generate_theta_scan(
    state_type: str,
    N: int,
    theta_grid: np.ndarray,
    max_photons: int | None = None,
    H_t: float = H_T,
) -> MziSensitivityData:
    r"""Run a :math:`\theta` scan for a single :math:`N` value.

    Args:
        state_type: ``"noon"`` or ``"twin_fock_std"``.
        N: Total photon number.
        theta_grid: :math:`\theta` values to scan.
        max_photons: Hilbert space truncation (defaults to ``N``).
        H_t: Holding time.

    Returns:
        MziSensitivityData with one N value.
    """
    if max_photons is None:
        max_photons = N
    state = _prepare_state(state_type, N, max_photons)
    skip_bs1 = state_type == "noon"
    result = compute_mzi_sensitivity_grid(
        state,
        theta_grid,
        max_photons,
        H_t=H_t,
        skip_bs1=skip_bs1,
    )

    theta_arr = np.asarray(result["theta_values"], dtype=float)
    return MziSensitivityData(
        state_type=state_type,
        N_values=np.array([N], dtype=int),
        theta_values=theta_arr,
        expectation_grid=np.atleast_2d(np.asarray(result["expectation_values"])),
        variance_grid=np.atleast_2d(np.asarray(result["variance_values"])),
        derivative_grid=np.atleast_2d(np.asarray(result["derivative_values"])),
        delta_theta_ep_grid=np.atleast_2d(np.asarray(result["delta_theta_ep"])),
        delta_theta_q_per_N=np.array([float(result["delta_theta_q"])]),
        fisher_classical_grid=np.atleast_2d(np.asarray(result["fisher_classical"])),
        delta_theta_c_grid=np.atleast_2d(np.asarray(result["delta_theta_c"])),
        H_t=H_t,
    )


# ============================================================================
# Generate Full Data (All N)
# ============================================================================


def generate_full_data(
    state_type: str,
    N_range: list[int],
    theta_grid: np.ndarray,
    H_t: float = H_T,
) -> MziSensitivityData:
    r"""Generate sensitivity data for all :math:`N` in a range.

    Args:
        state_type: ``"noon"`` or ``"twin_fock_std"``.
        N_range: List of :math:`N` values.
        theta_grid: :math:`\theta` values to scan.
        H_t: Holding time.

    Returns:
        MziSensitivityData with all N values.
    """
    scan_results: list[MziSensitivityData] = []
    for idx, N in enumerate(N_range):
        max_photons = N
        try:
            print(
                f"  Sweeping {state_type} N={N} ({idx + 1}/{len(N_range)})...",
                flush=True,
            )
            scan = generate_theta_scan(
                state_type,
                N,
                theta_grid,
                max_photons=max_photons,
                H_t=H_t,
            )
            scan_results.append(scan)
        except (ValueError, AssertionError) as exc:
            print(f"Warning: N={N} failed for {state_type}: {exc}")
            continue

    if not scan_results:
        raise RuntimeError(f"No valid N values for state_type={state_type}")

    # Concatenate along N axis
    return MziSensitivityData(
        state_type=state_type,
        N_values=np.concatenate([r.N_values for r in scan_results]).astype(int),
        theta_values=scan_results[0].theta_values,
        expectation_grid=np.concatenate([r.expectation_grid for r in scan_results]),
        variance_grid=np.concatenate([r.variance_grid for r in scan_results]),
        derivative_grid=np.concatenate([r.derivative_grid for r in scan_results]),
        delta_theta_ep_grid=np.concatenate(
            [r.delta_theta_ep_grid for r in scan_results]
        ),
        delta_theta_q_per_N=np.concatenate(
            [r.delta_theta_q_per_N for r in scan_results]
        ),
        fisher_classical_grid=np.concatenate(
            [r.fisher_classical_grid for r in scan_results]
        ),
        delta_theta_c_grid=np.concatenate([r.delta_theta_c_grid for r in scan_results]),
        H_t=H_t,
    )


# ============================================================================
# Scaling Exponent Fitting
# ============================================================================


def fit_scaling_exponent(
    N_values: np.ndarray,
    delta_theta_values: np.ndarray,
    N_min: int = 4,
) -> ScalingFitResult:
    r"""Fit scaling exponent :math:`\alpha` from :math:`\Delta\theta \propto N^\alpha`.

    Delegates to :func:`src.analysis.scaling_fit.fit_scaling_exponent` for the
    actual regression (``scipy.stats.linregress``), returning a
    :class:`src.analysis.scaling_fit.ScalingFitResult` with error bars and
    quality metrics.

    Args:
        N_values: Array of :math:`N` values.
        delta_theta_values: Array of :math:`\Delta\theta` values.
        N_min: Minimum :math:`N` for the fit (excludes small-:math:`N` transients).

    Returns:
        ScalingFitResult with fitted exponent, prefactor, error estimates,
        and quality diagnostics.
    """
    # Filter out non-positive / non-finite sensitivities
    mask = np.isfinite(delta_theta_values) & (delta_theta_values > 0)
    N_filtered = np.asarray(N_values, dtype=float)[mask]
    delta_filtered = np.asarray(delta_theta_values, dtype=float)[mask]

    if len(N_filtered) < 3:
        # Too few points — return invalid result (same pattern as module)
        return ScalingFitResult(
            alpha=0.0,
            alpha_err=0.0,
            C=0.0,
            C_err=0.0,
            R_squared=0.0,
            N_values=N_filtered,
            delta_phi_values=delta_filtered,
            valid=False,
            warnings=["Insufficient valid points for fit"],
        )

    return module_fit_scaling_exponent(
        N_filtered,
        delta_filtered,
        min_N=N_min,
    )


# ============================================================================
# Analyse Best/Worst Sensitivity from a 2D Grid
# ============================================================================


def analyse_best_worst_sensitivity(
    N_values: np.ndarray,
    theta_values: np.ndarray,
    sensitivity_grid: np.ndarray,
) -> dict:
    """Find best (min) and worst (max) sensitivity at each N.

    Args:
        N_values: Array of N values, shape ``(n_N,)``.
        theta_values: Array of θ values, shape ``(n_theta,)``.
        sensitivity_grid: 2D array of sensitivity values, shape ``(n_N, n_theta)``.

    Returns:
        Dictionary with keys:
        - ``N_values``: Array of N values.
        - ``best_sensitivity``: Minimum sensitivity at each N.
        - ``best_theta``: θ where minimum occurs.
        - ``worst_sensitivity``: Maximum finite sensitivity at each N.
        - ``worst_theta``: θ where maximum occurs.
    """
    n_N = len(N_values)
    best_sens = np.full(n_N, np.inf, dtype=float)
    best_th = np.full(n_N, np.nan, dtype=float)
    worst_sens = np.full(n_N, -np.inf, dtype=float)
    worst_th = np.full(n_N, np.nan, dtype=float)

    for i in range(n_N):
        slice_ = sensitivity_grid[i, :]
        finite_mask = np.isfinite(slice_)
        if np.any(finite_mask):
            full_indices = np.where(finite_mask)[0]

            # Best (minimum)
            b_idx = int(np.argmin(slice_[finite_mask]))
            actual_idx = full_indices[b_idx]
            best_sens[i] = float(slice_[actual_idx])
            best_th[i] = float(theta_values[actual_idx])

            # Worst (maximum finite)
            w_idx = int(np.argmax(slice_[finite_mask]))
            actual_w_idx = full_indices[w_idx]
            worst_sens[i] = float(slice_[actual_w_idx])
            worst_th[i] = float(theta_values[actual_w_idx])

    return {
        "N_values": N_values.copy(),
        "best_sensitivity": best_sens,
        "best_theta": best_th,
        "worst_sensitivity": worst_sens,
        "worst_theta": worst_th,
    }


# ============================================================================
# Plot Functions
# ============================================================================


def plot_delta_theta_overlay(
    data: MziSensitivityData,
    selected_N: list[int] | None = None,
    save_path: str | Path | None = None,
) -> Path:
    """Overlay Δθ_C and Δθ_Q vs θ for multiple N values on a single panel.

    Each N gets a unique colour from the *viridis* colormap.  Solid lines
    show Δθ_C, dashed horizontal lines show the corresponding QFI bound.
    The y-axis uses a log scale so that different N are clearly separated.

    Args:
        data: Sensitivity data containing all N values.
        selected_N: Which N to include (defaults to
            ``[1, 2, 4, 10, 20, 30, 40]`` for NOON,
            ``[2, 4, 10, 20, 30, 40]`` for Twin-Fock).
        save_path: Output SVG path.  Auto-generated if None.

    Returns:
        Path to saved SVG.
    """
    if selected_N is None:
        if data.state_type == "noon":
            selected_N = [1, 2, 4, 10, 20, 30, 40]
        else:
            selected_N = [2, 4, 10, 20, 30, 40]

    if save_path is None:
        save_path = _fig_path(f"{data.state_type}_delta_theta_comparison")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_label = data.state_type.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.colormaps["viridis"]
    colors = cmap(np.linspace(0.15, 0.85, len(selected_N)))

    for idx, N_val in enumerate(selected_N):
        match = np.where(data.N_values == N_val)[0]
        if len(match) == 0:
            continue
        n_idx = match[0]

        theta = data.theta_values
        dt_c = data.delta_theta_c_grid[n_idx, :]
        dt_q = data.delta_theta_q_per_N[n_idx]

        # Δθ_C (solid line)
        c_finite = np.isfinite(dt_c)
        if np.any(c_finite):
            ax.semilogy(
                theta[c_finite],
                dt_c[c_finite],
                color=colors[idx],
                linewidth=1.5,
                label=rf"N={N_val}  $\Delta\theta_{{\mathrm{{C}}}}$",
            )

        # Δθ_Q (dashed horizontal line)
        ax.axhline(
            y=float(dt_q),
            color=colors[idx],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title(f"{state_label} — Phase Sensitivity vs $\theta$")
    ax.legend(fontsize=8, loc="best", ncol=1)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_standard_deviation_comparison(
    data_noon: MziSensitivityData | None,
    data_tf: MziSensitivityData | None,
    save_path: str | Path | None = None,
) -> Path:
    r"""Overlaid line plot of the probe variance :math:`\text{Var}(J_z)` vs N.

    The probe variance is computed from the stored QFI bound:

    .. math::

        \text{Var}(J_z)_{\text{probe}} = \frac{1}{4 H_t^2 \cdot \Delta\theta_Q^2}

    For NOON this gives :math:`\text{Var}(J_z) = N^2/4` (Heisenberg-limited input).
    For Twin-Fock after BS1 this gives
    :math:`\text{Var}(J_z) = N(N+2)/8` (near-Heisenberg).

    The y-axis shows :math:`\text{Var}(J_z)` on a log-log scale.

    Args:
        data_noon: NOON sensitivity data (or None).
        data_tf: Twin-Fock sensitivity data (or None).
        save_path: Output SVG path.

    Returns:
        Path to saved SVG.
    """
    if save_path is None:
        save_path = _fig_path("variance_histogram")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colours = {"noon": "C0", "twin_fock_std": "C1"}
    labels_display = {"noon": "NOON", "twin_fock_std": "Twin-Fock"}
    markers = {"noon": "o", "twin_fock_std": "s"}

    for data, label_st in [
        (data_noon, "noon"),
        (data_tf, "twin_fock_std"),
    ]:
        if data is None:
            continue
        # Probe variance Var(J_z) = 1 / (4 * H_t² * Δθ_Q²)
        var = 1.0 / (4.0 * H_T**2 * data.delta_theta_q_per_N**2)
        ax.loglog(
            data.N_values,
            var,
            color=colours[label_st],
            marker=markers[label_st],
            linewidth=1.5,
            label=labels_display[label_st],
        )

    ax.set_xlabel("Total photon number $N$")
    ax.set_ylabel(r"$\mathrm{Var}(J_z)$ (probe variance)")
    ax.set_title(r"Probe Variance $\mathrm{Var}(J_z)$ vs $N$")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_scaling(
    data_noon: MziSensitivityData | None,
    data_tf: MziSensitivityData | None,
    save_path: str | Path | None = None,
    N_min_fit: int = 4,
) -> Path:
    """Log-log plot of best Δθ vs N with analytical QFI bounds and fits.

    Overlays NOON and Twin-Fock scaling data (from Δθ_C) on a single figure.

    Args:
        data_noon: NOON sensitivity data (or None).
        data_tf: Twin-Fock sensitivity data (or None).
        save_path: Output SVG path.
        N_min_fit: Minimum N for exponent fits.

    Returns:
        Path to saved SVG.
    """
    if data_noon is None and data_tf is None:
        raise ValueError("At least one data set must be provided")

    if save_path is None:
        save_path = _fig_path("scaling_comparison")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Reference lines
    N_ref = np.logspace(0, 1.5, 50)
    ax.plot(
        N_ref,
        1.0 / (H_T * N_ref),
        "k--",
        alpha=0.4,
        label=r"$\propto 1/N$ (Heisenberg)",
    )
    ax.plot(
        N_ref,
        1.0 / (H_T * np.sqrt(N_ref)),
        "k:",
        alpha=0.4,
        label=r"$\propto 1/\sqrt{N}$ (SQL)",
    )

    colours = {"noon": "C0", "twin_fock_std": "C1"}
    markers = {"noon": "o", "twin_fock_std": "s"}

    for data, label_name in [(data_noon, "NOON"), (data_tf, "Twin-Fock")]:
        if data is None:
            continue
        colour = colours[data.state_type]
        marker = markers[data.state_type]

        # QFI bound
        ax.loglog(
            data.N_values,
            data.delta_theta_q_per_N,
            f"{colour}--",
            alpha=0.5,
            label=f"{label_name} QFI bound",
        )

        # Best Δθ_C at each N
        analysis = analyse_best_worst_sensitivity(
            data.N_values,
            data.theta_values,
            data.delta_theta_c_grid,
        )
        N_vals = analysis["N_values"]
        best_dt_c = analysis["best_sensitivity"]
        finite = np.isfinite(best_dt_c)
        if np.any(finite):
            ax.loglog(
                N_vals[finite],
                best_dt_c[finite],
                f"{colour}{marker}-",
                label=rf"{label_name} best $\Delta\theta_{{\mathrm{{C}}}}$",
            )

        # Fit exponent to Δθ_C
        fit_result = fit_scaling_exponent(N_vals, best_dt_c, N_min=N_min_fit)
        if fit_result.valid:
            N_fit = fit_result.N_values
            delta_fit = fit_result.C * N_fit**fit_result.alpha
            ax.loglog(
                N_fit,
                delta_fit,
                f"{colour}--",
                alpha=0.7,
                linewidth=1.5,
                label=f"{label_name}: "
                rf"$\alpha = {fit_result.alpha:.3f}$",
            )

    ax.set_xlabel("Total photon number $N$")
    ax.set_ylabel(r"$\Delta\theta$")
    ax.set_title("Phase Sensitivity Scaling in Standard MZI")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_expectation_vs_theta_grid(
    data_noon: MziSensitivityData | None,
    data_tf: MziSensitivityData | None,
    save_path: str | Path | None = None,
) -> Path:
    """Plot ⟨J_z⟩ vs θ for NOON — N=1 (varying) and a representative N>1 (flat).

    This simplification works because for NOON with N ≥ 2 the output
    expectation ⟨J_z⟩ is identically zero at all θ, while N=1 shows the
    familiar single-photon MZI fringe.  The ±σ band is shown as a shaded
    region.

    Shows a single panel with at most two overlaid curves.
    """
    if data_noon is None:
        raise ValueError("NOON data is required for the simplified expectation grid")

    if save_path is None:
        save_path = _fig_path("expectation_grid")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    theta = data_noon.theta_values

    # N=1: varies sinusoidally
    idx_1 = np.where(data_noon.N_values == 1)[0]
    if len(idx_1) > 0:
        i1 = idx_1[0]
        exp_1 = data_noon.expectation_grid[i1, :]
        var_1 = data_noon.variance_grid[i1, :]
        ax.plot(theta, exp_1, "C0-", linewidth=1.5, label=r"N=1: $\langle J_z \rangle$")
        ax.fill_between(
            theta,
            exp_1 - np.sqrt(var_1),
            exp_1 + np.sqrt(var_1),
            alpha=0.15,
            color="C0",
            label=r"N=1: $\pm\sigma$",
        )

    # Representative N > 1 (use the largest available)
    mask = data_noon.N_values > 1
    if np.any(mask):
        large_idx = np.where(mask)[0][-1]  # largest N
        N_large = int(data_noon.N_values[large_idx])
        exp_large = data_noon.expectation_grid[large_idx, :]
        var_large = data_noon.variance_grid[large_idx, :]
        ax.plot(
            theta,
            exp_large,
            "C1-",
            linewidth=1.5,
            label=rf"N={N_large}: $\langle J_z \rangle$",
        )
        ax.fill_between(
            theta,
            exp_large - np.sqrt(var_large),
            exp_large + np.sqrt(var_large),
            alpha=0.15,
            color="C1",
            label=rf"N={N_large}: $\pm\sigma$",
        )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\langle J_z \rangle$")
    ax.set_title("NOON — Output Expectation (N=1 vs N>1)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Main Pipeline
# ============================================================================


def generate_all(
    force: bool = False,
    only: str | None = None,
) -> dict[str, MziSensitivityData]:
    """Generate all data and figures for the report.

    Args:
        force: If True, re-generate even if Parquet files exist.
        only: If set, only generate for this state type ("noon" or "twin_fock_std").

    Returns:
        Dict mapping state_type to MziSensitivityData.
    """
    theta_grid = np.arange(THETA_RANGE[0], THETA_RANGE[1] + THETA_STEP / 2, THETA_STEP)

    results: dict[str, MziSensitivityData] = {}

    state_configs: list[tuple[str, list[int], str]] = [
        ("noon", NOON_N_RANGE, "NOON"),
        ("twin_fock_std", TF_N_RANGE, "Twin-Fock"),
    ]

    for st, n_range, label in state_configs:
        if only is not None and st != only:
            continue

        pq_path = _parquet_path(f"{st}_sensitivity")
        if pq_path.exists() and not force:
            print(f"Loading existing data for {label} from {pq_path}")
            data = MziSensitivityData.from_parquet(pq_path)
        else:
            print(
                f"Generating {label} sensitivity data (N={n_range[0]}..{n_range[-1]})"
            )
            data = generate_full_data(st, n_range, theta_grid, H_t=H_T)
            data.save_parquet(pq_path)
            print(f"  Saved to {pq_path}")

        results[st] = data

    # --- Δθ overlay figures (one per state) ---
    for st, n_range, _label in state_configs:
        if only is not None and st != only:
            continue
        data = results.get(st)
        if data is None:
            continue
        overlay_path = _fig_path(f"{st}_delta_theta_comparison")
        if not overlay_path.exists() or force:
            if st == "noon":
                sel_N = [1, 2, 4, 10, 20, 30, 40]
            else:
                sel_N = [2, 4, 10, 20, 30, 40]
            plot_delta_theta_overlay(data, selected_N=sel_N, save_path=overlay_path)
            print(f"  Plotted {overlay_path}")

    # --- Probe standard deviation comparison ---
    hist_path = _fig_path("variance_histogram")
    if not hist_path.exists() or force:
        plot_standard_deviation_comparison(
            results.get("noon"),
            results.get("twin_fock_std"),
            save_path=hist_path,
        )
        print(f"  Plotted {hist_path}")

    # --- Simplified expectation grid (NOON-only: N=1 + N>1) ---
    exp_path = _fig_path("expectation_grid")
    if not exp_path.exists() or force:
        plot_expectation_vs_theta_grid(
            results.get("noon"),
            results.get("twin_fock_std"),
            save_path=exp_path,
        )
        print(f"  Plotted {exp_path}")

    # --- Combined scaling plot ---
    scaling_path = _fig_path("scaling_comparison")
    if not scaling_path.exists() or force:
        plot_scaling(
            results.get("noon"),
            results.get("twin_fock_std"),
            save_path=scaling_path,
        )
        print(f"  Plotted {scaling_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heisenberg-Limit MZI: NOON & Twin-Fock sensitivity simulation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate data and figures even if files exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["noon", "twin_fock_std"],
        default=None,
        help="Only generate data for this state type",
    )
    args = parser.parse_args()

    generate_all(force=args.force, only=args.only)


if __name__ == "__main__":
    main()

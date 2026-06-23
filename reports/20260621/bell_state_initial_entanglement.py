r"""
Local module for the 2026-06-21 Bell-State Initial S--A Entanglement report.

Tests whether a maximally entangled Bell state :math:`|\Phi^+\rangle` as the
initial S--A state can circumvent the :math:`J=1/2` bound established for
product initial states in the :math:`\omega`-independent drive protocol.

Circuit (identical to #20260528 except initial state):
    BS_S (50/50) :math:`\to` Hold (:math:`T_H=10`) :math:`\to` BS_S (50/50)
    :math:`\to` measure :math:`J_z^S` on system

Hold Hamiltonian:
    :math:`H = \omega J_z^S + a_x J_x^A + a_y J_y^A + a_z J_z^A + a_{zz} J_z^S J_z^A`

Initial states:
    - Bell: :math:`|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}`
    - Product: :math:`|00\rangle` (baseline, from #20260528)

Scenarios:
    A: Bell + :math:`(a_x, a_y, a_z, a_{zz})` all free
    B: Bell + :math:`(a_{zz})` free, no drive (:math:`a_x=a_y=a_z=0`)
    C: Bell + :math:`(a_x, a_y, a_z)` free, no interaction (:math:`a_{zz}=0`)
    D: Product :math:`|00\rangle` + :math:`(a_x, a_y, a_z, a_{zz})` free (data from #20260528)
    E: (Contingent) Bell + :math:`\omega`-modulated drive + :math:`a_{zz}` free

Usage:
    uv run python reports/20260621/bell_state_initial_entanglement.py
    uv run python reports/20260621/bell_state_initial_entanglement.py --force
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analysis.ancilla_drive_metrology import (
    compute_drive_sensitivity,
    compute_drive_sensitivity_with_details,
)
from src.analysis.ancilla_optimization import build_two_qubit_operators
from src.analysis.optimisation_pipeline import (
    TwoPhaseConfig,
    run_two_phase_pipeline,
)
from src.analysis.sensitivity_metrics import sql_reference
from src.utils.paths import report_path_fn
from src.utils.serialization import ParquetSerializable

# ============================================================================
# Constants
# ============================================================================

T_HOLD: float = 10.0  # Holding time (fixed)
T_BS: float = np.pi / 2.0  # 50/50 beam splitter
FD_STEP: float = 1e-6  # Finite-difference step
DRIVE_RADIUS: float = 10.0  # 3-ball radius for (a_x, a_y, a_z)
AZZ_BOUNDS: tuple[float, float] = (-5.0, 5.0)  # Bounds for a_zz

# Sampling hyperparameters
N_RANDOM: int = 1000  # Random search samples per (scenario, omega)
N_NM_REFINE: int = 15  # Nelder-Mead refinements per (scenario, omega)
NM_MAXITER: int = 5000

# Phase-rate values (50 points from 0.1 to 5.0, step 0.1)
OMEGA_VALS: list[float] = [round(0.1 * i, 1) for i in range(1, 51)]


class Scenario(StrEnum):
    """Experiment scenario labels."""

    A = "A"  # Bell + drive + interaction
    B = "B"  # Bell + interaction only
    C = "C"  # Bell + drive only
    D = "D"  # Product baseline (data from #20260528)
    E = "E"  # Bell + ω-modulated (contingent)


# ============================================================================
# Path Helpers
# ============================================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORT_DATE = "20260621"


_parquet_path, _fig_path = report_path_fn(REPORTS_DIR, REPORT_DATE)


# ============================================================================
# State Preparation
# ============================================================================


def bell_state_phi_plus() -> np.ndarray:
    r"""Construct the Bell state :math:`|\Phi^+\rangle`.

    :math:`|\Phi^+\rangle = \frac{1}{\sqrt{2}}\bigl(|00\rangle + |11\rangle\bigr)`

    In the computational basis :math:`\{|00\rangle, |01\rangle, |10\rangle,
    |11\rangle\}`, this is the vector :math:`[1, 0, 0, 1]^\top / \sqrt{2}`.

    Returns:
        Normalised complex 4-vector.
    """
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0 / np.sqrt(2.0)  # |00⟩
    state[3] = 1.0 / np.sqrt(2.0)  # |11⟩
    assert np.isclose(np.linalg.norm(state), 1.0), "Bell state must be normalised"
    return state


def bell_state_phi_minus() -> np.ndarray:
    r"""Construct the Bell state :math:`|\Phi^-\rangle`.

    :math:`|\Phi^-\rangle = \frac{1}{\sqrt{2}}\bigl(|00\rangle - |11\rangle\bigr)`

    Returns:
        Normalised complex 4-vector.
    """
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0 / np.sqrt(2.0)  # |00⟩
    state[3] = -1.0 / np.sqrt(2.0)  # -|11⟩
    assert np.isclose(np.linalg.norm(state), 1.0), "Bell state must be normalised"
    return state


def product_state_00() -> np.ndarray:
    r"""Construct the product state :math:`|00\rangle`.

    :math:`|00\rangle = |1,0\rangle_S \otimes |1,0\rangle_A`

    Returns:
        Normalised 4-vector with 1 at index 0.
    """
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0
    return state


# ============================================================================
# Initial State Selector
# ============================================================================


def get_initial_state(scenario: Scenario) -> np.ndarray:
    """Return the initial state for the given scenario.

    Args:
        scenario: Experiment scenario.

    Returns:
        Normalised 4-vector.
    """
    if scenario == Scenario.D:
        return product_state_00()
    return bell_state_phi_plus()


def verify_decoupled_baseline(
    scenarios: list[Scenario] | None = None,
    omega_values: list[float] | None = None,
    rtol: float = 1e-10,
) -> dict[tuple[str, float], bool]:
    """Verify decoupled baseline for all (scenario, omega).

    At zero drive and zero interaction, the circuit reduces to a standard MZI.
    For the product initial state ``|00>`` (Scenario D), the sensitivity must
    equal :math:`\\Delta\\omega = 1/T_\\text{HOLD}`.

    For Bell-state scenarios (A, B, C), the system reduced density matrix is
    maximally mixed :math:`\\mathbb{1}_2/2`, so :math:`\\langle J_z^S\\rangle`
    is identically zero for all :math:`\\omega` at zero drive/interaction.
    The error-propagation sensitivity therefore diverges (fringe extremum).
    This is the correct physical result and is checked separately.

    Args:
        scenarios: List of scenarios (default: A, B, C, D).
        omega_values: List of :math:`\\omega` values (default: all OMEGA_VALS).
        rtol: Relative tolerance for comparison.

    Returns:
        Dict mapping ``(scenario, omega) -> PASS/FAIL (True/False)``.
        For Bell-state scenarios, `True` means the sensitivity is infinite
        (fringe extremum), which is the expected behavior.
    """
    if scenarios is None:
        scenarios = [Scenario.A, Scenario.B, Scenario.C, Scenario.D]
    if omega_values is None:
        omega_values = OMEGA_VALS
    sql_ref = sql_reference(1, T_HOLD)  # 0.1
    results: dict[tuple[str, float], bool] = {}
    for sc in scenarios:
        for omega in omega_values:
            ops_dec = build_two_qubit_operators()
            psi0_dec = get_initial_state(sc)
            delta = compute_drive_sensitivity(
                psi0_dec, T_BS, T_HOLD, omega, 0.0, 0.0, 0.0, 0.0, ops_dec
            )
            if sc == Scenario.D:
                # Product state: expect exact SQL
                results[(sc.value, omega)] = bool(
                    np.isclose(delta, sql_ref, rtol=rtol),
                )
            else:
                # Bell state: expect fringe due to maximally mixed marginal
                results[(sc.value, omega)] = bool(
                    not np.isfinite(delta) or delta > 1e6,
                )
    return results


# ============================================================================
# Parameter Sampling
# ============================================================================


def sample_drive_vector(
    rng: np.random.Generator, R: float = DRIVE_RADIUS
) -> np.ndarray:
    """Sample a 3D drive vector uniformly from a 3-ball of radius R.

    Uses Marsaglia's method for uniform sampling within the ball.

    Args:
        rng: NumPy random generator.
        R: Radius of the 3-ball.

    Returns:
        Array of shape (3,) with (a_x, a_y, a_z).
    """
    v = rng.normal(size=3)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        return np.zeros(3)
    r = R * (rng.uniform(0.0, 1.0) ** (1.0 / 3.0))
    return v * r / norm_v


def sample_scenario_config(
    scenario: Scenario,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample parameter configuration for the given scenario.

    Args:
        scenario: Experiment scenario.
        rng: NumPy random generator.

    Returns:
        1D array of free parameters:
            - Scenario A: [a_x, a_y, a_z, a_zz] (4 params)
            - Scenario B: [a_zz] (1 param)
            - Scenario C: [a_x, a_y, a_z] (3 params)
            - Scenario D: [a_x, a_y, a_z, a_zz] (4 params)
            - Scenario E: [a_x, a_y, a_z, a_zz] (4 params)
    """
    if scenario in (Scenario.A, Scenario.D, Scenario.E):
        drive = sample_drive_vector(rng)
        a_zz = rng.uniform(AZZ_BOUNDS[0], AZZ_BOUNDS[1])
        return np.array([drive[0], drive[1], drive[2], a_zz], dtype=float)
    if scenario == Scenario.B:
        a_zz = rng.uniform(AZZ_BOUNDS[0], AZZ_BOUNDS[1])
        return np.array([a_zz], dtype=float)
    if scenario == Scenario.C:
        drive = sample_drive_vector(rng)
        return np.array([drive[0], drive[1], drive[2]], dtype=float)
    raise ValueError(f"Unknown scenario: {scenario}")


def config_to_params(
    config: np.ndarray,
    scenario: Scenario,
) -> tuple[float, float, float, float]:
    """Unpack a scenario parameter vector into (a_x, a_y, a_z, a_zz).

    Args:
        config: Parameter vector for the scenario.
        scenario: Experiment scenario (determines parameter layout).

    Returns:
        Tuple ``(a_x, a_y, a_z, a_zz)``.
    """
    if scenario in (Scenario.A, Scenario.D, Scenario.E):
        return (float(config[0]), float(config[1]), float(config[2]), float(config[3]))
    if scenario == Scenario.B:
        return (0.0, 0.0, 0.0, float(config[0]))
    if scenario == Scenario.C:
        return (float(config[0]), float(config[1]), float(config[2]), 0.0)
    raise ValueError(f"Unknown scenario: {scenario}")


# ============================================================================
# Random Search
# ============================================================================


@dataclass
class BellRandomSearchResult(ParquetSerializable):
    r"""Result from a random search for a given scenario and omega.

    Attributes:
        samples: Array of shape ``(N, D)`` with sampled parameter values.
        delta_omega_values: Array of shape ``(N,)`` with :math:`\Delta\omega`.
        best_params: Best parameter tuple.
        best_delta_omega: Minimal :math:`\Delta\omega` found.
        scenario: Scenario label.
        omega_value: :math:`\omega` value.
        sql: SQL reference value.
        t_hold: Holding time.
    """

    samples: np.ndarray
    delta_omega_values: np.ndarray
    best_params: tuple[float, ...]
    best_delta_omega: float
    scenario: str = "A"
    omega_value: float = 1.0
    sql: float = 0.1
    t_hold: float = 10.0

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "delta_omega",
        "scenario",
        "omega_value",
        "sql",
        "t_hold",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.samples)
        ncols = self.samples.shape[1]
        # Build column names dynamically
        if ncols == 4:
            col_names = ["a_x", "a_y", "a_z", "a_zz"]
        elif ncols == 1:
            col_names = ["a_zz"]
        elif ncols == 3:
            col_names = ["a_x", "a_y", "a_z"]
        else:
            col_names = [f"param_{i}" for i in range(ncols)]
        data: dict[str, object] = {}
        for i, cname in enumerate(col_names):
            data[cname] = self.samples[:, i]
        # Pad missing columns with NaN for uniformity
        for col in ["a_x", "a_y", "a_z", "a_zz"]:
            if col not in data:
                data[col] = np.full(n, np.nan)
        data["delta_omega"] = self.delta_omega_values
        data["scenario"] = [self.scenario] * n
        data["omega_value"] = [self.omega_value] * n
        data["sql"] = [self.sql] * n
        data["t_hold"] = [self.t_hold] * n
        return pd.DataFrame(data)

    @classmethod
    def from_parquet(cls, path: str | Path) -> BellRandomSearchResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        samples = df[["a_x", "a_y", "a_z", "a_zz"]].to_numpy(dtype=float)
        deltas = df["delta_omega"].to_numpy(dtype=float)
        best_idx = int(np.argmin(deltas))
        return cls(
            samples=samples,
            delta_omega_values=deltas,
            best_params=(
                float(samples[best_idx, 0])
                if not np.isnan(samples[best_idx, 0])
                else 0.0,
                float(samples[best_idx, 1])
                if not np.isnan(samples[best_idx, 1])
                else 0.0,
                float(samples[best_idx, 2])
                if not np.isnan(samples[best_idx, 2])
                else 0.0,
                float(samples[best_idx, 3])
                if not np.isnan(samples[best_idx, 3])
                else 0.0,
            ),
            best_delta_omega=float(deltas[best_idx]),
            scenario=str(df["scenario"].iloc[0]),
            omega_value=float(df["omega_value"].iloc[0]),
            sql=float(df["sql"].iloc[0]),
            t_hold=float(df["t_hold"].iloc[0]),
        )


def run_random_search(
    scenario: Scenario,
    omega: float,
    n_samples: int = N_RANDOM,
    seed: int | None = 42,
) -> BellRandomSearchResult:
    """Random search over the free parameters for a given (scenario, omega).

    Args:
        scenario: Experiment scenario.
        omega: Phase rate value.
        n_samples: Number of random points to evaluate.
        seed: Random seed for reproducibility.

    Returns:
        BellRandomSearchResult with all samples and best found.
    """
    rng = np.random.default_rng(seed)
    ops = build_two_qubit_operators()
    psi0 = get_initial_state(scenario)

    # Determine parameter dimensionality
    if scenario == Scenario.B:
        n_dims = 1
    elif scenario == Scenario.C:
        n_dims = 3
    else:
        n_dims = 4

    samples = np.zeros((n_samples, n_dims), dtype=float)
    deltas = np.full(n_samples, np.inf, dtype=float)

    for i in range(n_samples):
        config = sample_scenario_config(scenario, rng)
        samples[i] = config
        a_x, a_y, a_z, a_zz = config_to_params(config, scenario)
        domega = compute_drive_sensitivity(
            psi0, T_BS, T_HOLD, omega, a_x, a_y, a_z, a_zz, ops
        )
        deltas[i] = domega

    best_idx = int(np.argmin(deltas))
    best_config = samples[best_idx]
    best_a_x, best_a_y, best_a_z, best_a_zz = config_to_params(best_config, scenario)

    return BellRandomSearchResult(
        samples=samples,
        delta_omega_values=deltas,
        best_params=(best_a_x, best_a_y, best_a_z, best_a_zz),
        best_delta_omega=float(deltas[best_idx]),
        scenario=scenario.value,
        omega_value=omega,
        sql=sql_reference(1, T_HOLD),
        t_hold=T_HOLD,
    )


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


def sensitivity_objective(
    params: np.ndarray,
    scenario: Scenario,
    omega_true: float,
    ops: dict[str, np.ndarray],
    t_hold: float = T_HOLD,
    fd_step: float = FD_STEP,
    penalty_scale: float = 1e6,
) -> float:
    r"""Objective function for minimising :math:`\Delta\omega`.

    The bound constraints are enforced by quadratic penalties:
    - :math:`\|(a_x, a_y, a_z)\| \le R` (3-ball constraint)
    - :math:`a_{zz} \in [-5, 5]`

    Args:
        params: Parameter vector (dimensionality depends on scenario).
        scenario: Experiment scenario.
        omega_true: True phase rate.
        ops: Two-qubit operators.
        t_hold: Holding time.
        fd_step: Finite-difference step.
        penalty_scale: Scale for bound-violation penalty.

    Returns:
        :math:`\Delta\omega` (plus infinite penalty if bounds violated).
    """
    penalty = 0.0
    a_x, a_y, a_z, a_zz = config_to_params(params, scenario)

    # Bound enforcement: drive vector in 3-ball
    if scenario != Scenario.B:
        drive_norm = math.sqrt(a_x**2 + a_y**2 + a_z**2)
        if drive_norm > DRIVE_RADIUS:
            penalty += penalty_scale * (drive_norm - DRIVE_RADIUS) ** 2

    # Bound enforcement: a_zz in [-5, 5]
    if scenario != Scenario.C:
        if a_zz < AZZ_BOUNDS[0]:
            penalty += penalty_scale * (AZZ_BOUNDS[0] - a_zz) ** 2
        elif a_zz > AZZ_BOUNDS[1]:
            penalty += penalty_scale * (a_zz - AZZ_BOUNDS[1]) ** 2

    if penalty > 0.0:
        return float(1e10 + penalty)

    psi0 = get_initial_state(scenario)
    return compute_drive_sensitivity(
        psi0, T_BS, T_HOLD, omega_true, a_x, a_y, a_z, a_zz, ops
    )


@dataclass
class BellNelderMeadResult(ParquetSerializable):
    r"""Result of a single Nelder--Mead run.

    Attributes:
        delta_omega_opt: Best sensitivity :math:`\Delta\omega` found.
        params_opt: Optimal parameter vector.
        omega_true: True :math:`\omega` used for this optimisation.
        scenario: Scenario label.
        success: Whether the optimiser reported success.
        nfev: Number of function evaluations.
        message: Optimiser message.
        expectation_Jz: :math:`\langle J_z^S\rangle` at the optimum.
        variance_Jz: :math:`\mathrm{Var}(J_z^S)` at the optimum.
        d_exp: :math:`\partial\langle J_z^S\rangle/\partial\omega` at optimum.
        is_fringe: Whether the optimum is at a fringe extremum.
        history: Objective function values at each iteration.
    """

    delta_omega_opt: float
    params_opt: np.ndarray
    omega_true: float
    scenario: str
    success: bool
    nfev: int
    message: str = ""
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    d_exp: float = 0.0
    is_fringe: bool = False
    history: list[float] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "scenario",
        "delta_omega",
        "omega_true",
        "a_x",
        "a_y",
        "a_z",
        "a_zz",
        "success",
        "nfev",
        "expectation_Jz",
        "variance_Jz",
        "d_exp",
        "is_fringe",
        "message",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        a_x, a_y, a_z, a_zz = self._unpack_params()
        return pd.DataFrame(
            {
                "scenario": [self.scenario],
                "delta_omega": [self.delta_omega_opt],
                "omega_true": [self.omega_true],
                "a_x": [a_x],
                "a_y": [a_y],
                "a_z": [a_z],
                "a_zz": [a_zz],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "d_exp": [self.d_exp],
                "is_fringe": [int(self.is_fringe)],
                "message": [self.message],
            },
        )

    def _unpack_params(self) -> tuple[float, float, float, float]:
        """Unpack params_opt into (a_x, a_y, a_z, a_zz).

        Delegates to :func:`config_to_params`.
        """
        return config_to_params(self.params_opt, Scenario(self.scenario))

    def _save_sidecars(self, path: Path) -> None:
        if not self.history:
            return
        history_path = path.with_stem(path.stem + "-history")
        pd.DataFrame({"history": [self.history]}).to_parquet(history_path, index=False)

    @classmethod
    def from_parquet(cls, path: str | Path) -> BellNelderMeadResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        history_path = Path(path).with_stem(Path(path).stem + "-history")
        if history_path.exists():
            history = list(pd.read_parquet(history_path)["history"].iloc[0])
        else:
            history = []
        row = df.iloc[0]
        scenario_str = str(row["scenario"])
        # Determine parameter dimensionality from scenario
        if scenario_str == Scenario.B.value:
            params_opt = np.array([float(row["a_zz"])])
        elif scenario_str == Scenario.C.value:
            params_opt = np.array(
                [
                    float(row["a_x"]),
                    float(row["a_y"]),
                    float(row["a_z"]),
                ]
            )
        else:
            params_opt = np.array(
                [
                    float(row["a_x"]),
                    float(row["a_y"]),
                    float(row["a_z"]),
                    float(row["a_zz"]),
                ]
            )
        return cls(
            delta_omega_opt=float(row["delta_omega"]),
            params_opt=params_opt,
            omega_true=float(row["omega_true"]),
            scenario=scenario_str,
            success=bool(int(row["success"])),
            nfev=int(row["nfev"]),
            message=str(row["message"]),
            expectation_Jz=float(row["expectation_Jz"]),
            variance_Jz=float(row["variance_Jz"]),
            d_exp=float(row["d_exp"]),
            is_fringe=bool(int(row["is_fringe"])),
            history=history,
        )


def run_nelder_mead(
    scenario: Scenario,
    omega_true: float,
    ops: dict[str, np.ndarray] | None = None,
    x0: np.ndarray | None = None,
    seed: int | None = None,
    maxiter: int = NM_MAXITER,
    xatol: float = 1e-8,
    fatol: float = 1e-8,
    adaptive: bool = True,
    track_history: bool = False,
) -> BellNelderMeadResult:
    """Run Nelder--Mead optimisation.

    Args:
        scenario: Experiment scenario.
        omega_true: True phase rate parameter.
        ops: Two-qubit operators (built if ``None``).
        x0: Initial parameter vector. Randomly sampled if ``None``.
        seed: Random seed (used if ``x0`` is ``None``).
        maxiter: Maximum Nelder--Mead iterations.
        xatol: Absolute parameter tolerance.
        fatol: Absolute function tolerance.
        adaptive: Use adaptive Nelder--Mead parameters.
        track_history: If ``True``, record objective values per iteration.

    Returns:
        BellNelderMeadResult.
    """
    if ops is None:
        ops = build_two_qubit_operators()

    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = _random_starting_point(scenario, rng)
    else:
        x0 = np.asarray(x0, dtype=float)

    def objective(p: np.ndarray) -> float:
        return sensitivity_objective(p, scenario, omega_true, ops)

    history: list[float] = []

    def callback(_x: np.ndarray) -> None:
        if track_history:
            val = objective(_x)
            history.append(val)

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        callback=callback if track_history else None,
        options={
            "maxiter": maxiter,
            "xatol": xatol,
            "fatol": fatol,
            "adaptive": adaptive,
        },
    )

    opt_params = result.x.copy()

    # Compute diagnostics at the optimal point
    a_x, a_y, a_z, a_zz = _unpack_opt_params(opt_params, scenario)
    psi0 = get_initial_state(scenario)
    _delta, exp_val, var_val, d_exp_val, is_fringe_val = (
        compute_drive_sensitivity_with_details(
            psi0,
            T_BS,
            T_HOLD,
            omega_true,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
        )
    )

    return BellNelderMeadResult(
        delta_omega_opt=float(result.fun),
        params_opt=opt_params,
        omega_true=omega_true,
        scenario=scenario.value,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
        expectation_Jz=exp_val,
        variance_Jz=var_val,
        d_exp=d_exp_val,
        is_fringe=is_fringe_val,
        history=history.copy(),
    )


def _random_starting_point(
    scenario: Scenario,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a random starting point for Nelder-Mead."""
    return sample_scenario_config(scenario, rng)


def _unpack_opt_params(
    params: np.ndarray,
    scenario: Scenario,
) -> tuple[float, float, float, float]:
    """Unpack optimal params into (a_x, a_y, a_z, a_zz).

    Delegates to :func:`config_to_params`.
    """
    return config_to_params(params, scenario)


# ============================================================================
# Per-(Scenario, Omega) Optimisation
# ============================================================================


@dataclass
class BellOptimisationResult(ParquetSerializable):
    """Optimisation result for a single (scenario, omega) pair.

    Attributes:
        scenario: Scenario label.
        omega: Phase rate value.
        delta_omega_opt: Best sensitivity found.
        sql: SQL reference value (0.1 for N=1, T_H=10).
        ratio: SQL / Δω_opt (ratio > 1 means beating SQL).
        a_x_opt: Optimal J_x^A drive coefficient.
        a_y_opt: Optimal J_y^A drive coefficient.
        a_z_opt: Optimal J_z^A drive coefficient.
        a_zz_opt: Optimal Ising interaction coefficient.
        expectation_Jz: ⟨J_z^S⟩ at the optimum.
        variance_Jz: Var(J_z^S) at the optimum.
        drive_norm: Norm of optimal drive vector.
        d_exp: ∂⟨J_z^S⟩/∂ω at optimum.
        is_fringe: Whether the optimum is at a fringe extremum.
        success: Whether Nelder--Mead reported success.
        nfev: Number of function evaluations.
        t_hold: Holding time (fixed at 10.0).
        fd_step: Finite-difference step.
    """

    scenario: str
    omega: float
    delta_omega_opt: float
    sql: float
    ratio: float
    a_x_opt: float
    a_y_opt: float
    a_z_opt: float
    a_zz_opt: float
    expectation_Jz: float = 0.0
    variance_Jz: float = 0.0
    drive_norm: float = 0.0
    d_exp: float = 0.0
    is_fringe: bool = False
    success: bool = True
    nfev: int = 0
    t_hold: float = T_HOLD
    fd_step: float = FD_STEP

    _PARQUET_COLUMNS: ClassVar[list[str]] = [
        "scenario",
        "omega",
        "delta_omega_opt",
        "sql",
        "ratio",
        "a_x_opt",
        "a_y_opt",
        "a_z_opt",
        "a_zz_opt",
        "expectation_Jz",
        "variance_Jz",
        "drive_norm",
        "d_exp",
        "is_fringe",
        "success",
        "nfev",
        "t_hold",
        "fd_step",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "scenario": [self.scenario],
                "omega": [self.omega],
                "delta_omega_opt": [self.delta_omega_opt],
                "sql": [self.sql],
                "ratio": [self.ratio],
                "a_x_opt": [self.a_x_opt],
                "a_y_opt": [self.a_y_opt],
                "a_z_opt": [self.a_z_opt],
                "a_zz_opt": [self.a_zz_opt],
                "expectation_Jz": [self.expectation_Jz],
                "variance_Jz": [self.variance_Jz],
                "drive_norm": [self.drive_norm],
                "d_exp": [self.d_exp],
                "is_fringe": [int(self.is_fringe)],
                "success": [int(self.success)],
                "nfev": [self.nfev],
                "t_hold": [self.t_hold],
                "fd_step": [self.fd_step],
            },
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> BellOptimisationResult:
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        row = df.iloc[0]
        return cls(
            scenario=str(row["scenario"]),
            omega=float(row["omega"]),
            delta_omega_opt=float(row["delta_omega_opt"]),
            sql=float(row["sql"]),
            ratio=float(row["ratio"]),
            a_x_opt=float(row["a_x_opt"]),
            a_y_opt=float(row["a_y_opt"]),
            a_z_opt=float(row["a_z_opt"]),
            a_zz_opt=float(row["a_zz_opt"]),
            expectation_Jz=float(row["expectation_Jz"]),
            variance_Jz=float(row["variance_Jz"]),
            drive_norm=float(row["drive_norm"]),
            d_exp=float(row["d_exp"]),
            is_fringe=bool(int(row["is_fringe"])),
            success=bool(int(row["success"])),
            nfev=int(row["nfev"]),
            t_hold=float(row["t_hold"]),
            fd_step=float(row["fd_step"]),
        )


@dataclass
class BellScanResult(ParquetSerializable):
    """Collection of (scenario, omega) optimisation results."""

    results: list[BellOptimisationResult] = field(default_factory=list)

    _PARQUET_COLUMNS: ClassVar[list[str]] = BellOptimisationResult._PARQUET_COLUMNS

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=self._PARQUET_COLUMNS)
        return pd.concat([r.to_dataframe() for r in self.results], ignore_index=True)

    @classmethod
    def from_parquet(cls, path: str | Path) -> BellScanResult:
        path = Path(path)
        df = pd.read_parquet(path)
        cls._validate_columns(df)
        results: list[BellOptimisationResult] = []
        for _, row in df.iterrows():
            results.append(
                BellOptimisationResult(
                    scenario=str(row["scenario"]),
                    omega=float(row["omega"]),
                    delta_omega_opt=float(row["delta_omega_opt"]),
                    sql=float(row["sql"]),
                    ratio=float(row["ratio"]),
                    a_x_opt=float(row["a_x_opt"]),
                    a_y_opt=float(row["a_y_opt"]),
                    a_z_opt=float(row["a_z_opt"]),
                    a_zz_opt=float(row["a_zz_opt"]),
                    expectation_Jz=float(row["expectation_Jz"]),
                    variance_Jz=float(row["variance_Jz"]),
                    drive_norm=float(row["drive_norm"]),
                    d_exp=float(row["d_exp"]),
                    is_fringe=bool(int(row["is_fringe"])),
                    success=bool(int(row["success"])),
                    nfev=int(row["nfev"]),
                    t_hold=float(row["t_hold"]),
                    fd_step=float(row["fd_step"]),
                ),
            )
        return cls(results=results)

    @property
    def scenario_values(self) -> np.ndarray:
        return np.array(sorted({r.scenario for r in self.results}))

    @property
    def omega_values(self) -> np.ndarray:
        return np.array(sorted({r.omega for r in self.results}))


# ============================================================================
# Full Pipeline: Random Search + NM Refinement
# ============================================================================


def run_single_scenario_omega(
    scenario: Scenario,
    omega: float,
    seed: int | None = 42,
) -> BellOptimisationResult:
    """Run the full optimisation pipeline for a single (scenario, omega) pair.

    Uses the shared two-phase pipeline (:func:`run_two_phase_pipeline`) for
    random search + Nelder--Mead refinement.

    1. Random search (1000 samples).
    2. Nelder--Mead refinement from top 15 points.
       - Skipped when all random-search samples are fringe (Scenario C),
         since the landscape is flat (no a_zz interaction → no sensitivity).

    Args:
        scenario: Experiment scenario.
        omega: Phase rate value.
        seed: Base random seed (incremented per call).

    Returns:
        BellOptimisationResult with the optimal parameters and sensitivity.
    """
    base_seed = seed if seed is not None else 42
    ops = build_two_qubit_operators()

    # Stage 1: Random search (also used for fringe detection)
    rs_result = run_random_search(scenario, omega, n_samples=N_RANDOM, seed=base_seed)

    # If all random samples are fringe (Δω > 1e6), skip NM refinement.
    # This happens for Scenario C (no interaction, a_zz=0), where
    # independent S/A evolution gives zero ⟨J_z^S⟩ for all parameters.
    all_fringe = all(
        not np.isfinite(d) or d > 1e6 for d in rs_result.delta_omega_values
    )

    if all_fringe:
        sql_val = sql_reference(1, T_HOLD)
        a_x, a_y, a_z, a_zz = rs_result.best_params
        drive_norm = math.sqrt(a_x**2 + a_y**2 + a_z**2)
        return BellOptimisationResult(
            scenario=scenario.value,
            omega=omega,
            delta_omega_opt=float("inf"),
            sql=sql_val,
            ratio=float("nan"),
            a_x_opt=a_x,
            a_y_opt=a_y,
            a_z_opt=a_z,
            a_zz_opt=a_zz,
            expectation_Jz=0.0,
            variance_Jz=0.25,
            drive_norm=drive_norm,
            d_exp=0.0,
            is_fringe=True,
            success=True,
            nfev=N_RANDOM,
        )

    # Wrapper callables for the shared pipeline
    def rs_fn(n_samples, seed, **kw):
        return run_random_search(scenario, omega, n_samples=n_samples, seed=seed)

    def nm_fn(x0, seed, **kw):
        return run_nelder_mead(scenario, omega_true=omega, ops=ops, x0=x0, seed=seed)

    config = TwoPhaseConfig(
        n_random=N_RANDOM,
        n_nm_refine=N_NM_REFINE,
        nm_maxiter=NM_MAXITER,
    )

    best_nm, _all_nm = run_two_phase_pipeline(
        random_search_fn=rs_fn,
        nm_fn=nm_fn,
        config=config,
        seed=base_seed,
    )

    sql_val = sql_reference(1, T_HOLD)
    a_x, a_y, a_z, a_zz = _unpack_opt_params(best_nm.params_opt, scenario)
    drive_norm = math.sqrt(a_x**2 + a_y**2 + a_z**2)

    ratio = (
        sql_val / best_nm.delta_omega_opt
        if np.isfinite(best_nm.delta_omega_opt) and best_nm.delta_omega_opt > 0
        else float("nan")
    )

    return BellOptimisationResult(
        scenario=scenario.value,
        omega=omega,
        delta_omega_opt=best_nm.delta_omega_opt,
        sql=sql_val,
        ratio=ratio,
        a_x_opt=a_x,
        a_y_opt=a_y,
        a_z_opt=a_z,
        a_zz_opt=a_zz,
        expectation_Jz=best_nm.expectation_Jz,
        variance_Jz=best_nm.variance_Jz,
        drive_norm=drive_norm,
        d_exp=best_nm.d_exp,
        is_fringe=best_nm.is_fringe,
        success=best_nm.success,
        nfev=best_nm.nfev,
    )


# ============================================================================
# Scenario Definitions
# ============================================================================

# Active scenarios to run (A, B, C, D)
ACTIVE_SCENARIOS: list[Scenario] = [Scenario.A, Scenario.B, Scenario.C, Scenario.D]


def _worker(args: tuple[str, float]) -> dict[str, int | float | str]:
    """Worker for parallel scan.

    Args:
        args: Tuple ``(scenario_str, omega)``.

    Returns:
        Dict of result data.
    """
    scenario = Scenario(args[0])
    omega_val = args[1]
    print(f"  [run] Scenario {scenario.value}, ω={omega_val}")
    result = run_single_scenario_omega(scenario, omega_val)
    return {
        "scenario": result.scenario,
        "omega": result.omega,
        "delta_omega_opt": result.delta_omega_opt,
        "sql": result.sql,
        "ratio": result.ratio,
        "a_x_opt": result.a_x_opt,
        "a_y_opt": result.a_y_opt,
        "a_z_opt": result.a_z_opt,
        "a_zz_opt": result.a_zz_opt,
        "expectation_Jz": result.expectation_Jz,
        "variance_Jz": result.variance_Jz,
        "drive_norm": result.drive_norm,
        "d_exp": result.d_exp,
        "is_fringe": int(result.is_fringe),
        "success": int(result.success),
        "nfev": result.nfev,
    }


# ============================================================================
# Data Generation
# ============================================================================


def generate_decoupled_baseline(force: bool = False) -> None:
    """Verify the decoupled baseline and save results.

    Args:
        force: Re-run even if Parquet exists.
    """
    parquet_p = _parquet_path("decoupled-baseline")

    if parquet_p.exists() and not force:
        print(f"[skip] {parquet_p.name} exists (use --force to overwrite)")
        return

    print("[run] Computing decoupled baseline verification...")
    verifications = verify_decoupled_baseline()
    rows: list[dict[str, float | int | str]] = []
    for (sc_str, omega), passed in verifications.items():
        sql_ref = sql_reference(1, T_HOLD)
        ops_dec = build_two_qubit_operators()
        psi0_dec = get_initial_state(Scenario(sc_str))
        delta = compute_drive_sensitivity(
            psi0_dec, T_BS, T_HOLD, omega, 0.0, 0.0, 0.0, 0.0, ops_dec
        )
        rows.append(
            {
                "scenario": sc_str,
                "omega": omega,
                "delta_omega": delta,
                "sql": sql_ref,
                "ratio": sql_ref / delta if delta > 0 else float("nan"),
                "pass": str(passed),
            },
        )
    df = pd.DataFrame(rows)
    parquet_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_p, index=False)
    print(f"[save] {parquet_p}")


def _process_scan_item(
    item: tuple[str, float],
    idx: int,
    total: int,
    t_start: float,
    results: list[BellOptimisationResult],
) -> None:
    """Process a single (scenario, omega) item for the scenario scan.

    Handles try/except around the worker call, prints a progress bar with ETA,
    and appends non-fringe results to ``results`` in-place.
    """
    sc_str, omega_val = item
    t_item = time.monotonic()
    try:
        rdict = _worker(item)
    except Exception as exc:
        print(f"\n  [ERROR] S={sc_str}, ω={omega_val}: {exc}")
        return
    elapsed = time.monotonic() - t_item
    delta = rdict["delta_omega_opt"]
    ratio_str = f"Δω={delta:.6f}" if np.isfinite(delta) else "Δω=∞"
    frac = (idx + 1) / total
    nbars = int(frac * 40)
    bar = "█" * nbars + "░" * (40 - nbars)
    total_elapsed = time.monotonic() - t_start
    remaining = total_elapsed / (idx + 1) * (total - idx - 1) if idx > 0 else 0
    print(
        f"  [{bar}] {idx + 1:3d}/{total} "
        f"S={sc_str} ω={omega_val:3.1f} {ratio_str} "
        f"({elapsed:.1f}s, ETA {remaining:.0f}s)",
        flush=True,
    )
    if not np.isfinite(delta):
        return
    results.append(
        BellOptimisationResult(
            scenario=str(rdict["scenario"]),
            omega=float(rdict["omega"]),
            delta_omega_opt=float(rdict["delta_omega_opt"]),
            sql=float(rdict["sql"]),
            ratio=float(rdict["ratio"]),
            a_x_opt=float(rdict["a_x_opt"]),
            a_y_opt=float(rdict["a_y_opt"]),
            a_z_opt=float(rdict["a_z_opt"]),
            a_zz_opt=float(rdict["a_zz_opt"]),
            expectation_Jz=float(rdict["expectation_Jz"]),
            variance_Jz=float(rdict["variance_Jz"]),
            drive_norm=float(rdict["drive_norm"]),
            d_exp=float(rdict["d_exp"]),
            is_fringe=bool(rdict["is_fringe"]),
            success=bool(rdict["success"]),
            nfev=int(rdict["nfev"]),
        ),
    )


def generate_scenario_scan(force: bool = False) -> None:
    """Full (scenario, omega) scan.

    Each (scenario, omega) pair runs:
      1. Random search with 1000 points.
      2. Nelder--Mead refinement from top 15 points.

    Runs serially with a progress ticker.

    Args:
        force: Re-run even if Parquet exists.
    """
    parquet_p = _parquet_path("scenario-omega-scan")

    if parquet_p.exists() and not force:
        print(f"[skip] {parquet_p.name} exists (use --force to overwrite)")
        summary = BellScanResult.from_parquet(parquet_p)
    else:
        if force:
            parquet_p.unlink(missing_ok=True)

        items_to_run = [
            (sc.value, omega) for sc in ACTIVE_SCENARIOS for omega in OMEGA_VALS
        ]
        total = len(items_to_run)
        print(f"[run] Scenario scan: {total} (scenario, omega) pairs (serial)")

        results: list[BellOptimisationResult] = []
        t_start = time.monotonic()
        for idx, item in enumerate(items_to_run):
            _process_scan_item(item, idx, total, t_start, results)
        total_time = time.monotonic() - t_start
        print(
            f"  Total: {total_time:.0f}s for {total} pairs ({total_time / total:.1f}s avg)"
        )
        summary = BellScanResult(results=results)
        summary.save_parquet(parquet_p)
        print(f"[save] {parquet_p}")

    # Generate figures
    df = summary.to_dataframe()
    if not df.empty:
        _generate_figures(df)


def _generate_figures(df: pd.DataFrame) -> None:
    """Generate standard figures from the scan results.

    Creates three ω-axis plots per scenario:
    1. Sensitivity Δω vs ω (with SQL reference line).
    2. Ratio SQL/Δω vs ω (with ratio=1 line).
    3. Optimal parameters (drive norm, |a_zz|) vs ω.

    Args:
        df: Dataframe with scan results.
    """
    import matplotlib.pyplot as plt

    for sc in sorted(df["scenario"].unique()):
        df_sc = df[df["scenario"] == sc].sort_values("omega")

        # --- 1. Sensitivity vs ω ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_sc["omega"], df_sc["delta_omega_opt"], "o-", label=f"Scenario {sc}")
        ax.axhline(
            y=df_sc["sql"].iloc[0],
            color="gray",
            ls="--",
            label=f"SQL = {df_sc['sql'].iloc[0]:.3f}",
        )
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$\Delta\omega_{\mathrm{opt}}$")
        ax.set_title(f"Sensitivity vs ω — Scenario {sc}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = _fig_path(f"sensitivity-scenario-{sc}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig]  {path}")

        # --- 2. Ratio vs ω ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_sc["omega"], df_sc["ratio"], "s-", label=f"Scenario {sc}")
        ax.axhline(y=1.0, color="gray", ls="--", label="SQL (ratio=1)")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$R = \mathrm{SQL} / \Delta\omega_{\mathrm{opt}}$")
        ax.set_title(f"SQL-violation Ratio vs ω — Scenario {sc}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = _fig_path(f"ratio-scenario-{sc}")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig]  {path}")

        # --- 3. Optimal parameters vs ω ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            df_sc["omega"],
            df_sc["drive_norm"],
            "o-",
            label=r"$\| \vec{a}_{\mathrm{drive}} \|$",
        )
        ax.plot(df_sc["omega"], np.abs(df_sc["a_zz_opt"]), "s-", label=r"$|a_{zz}|$")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel("Optimal parameter value")
        ax.set_title(f"Optimal Parameters vs ω — Scenario {sc}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = _fig_path(f"optimal-params-scenario-{sc}")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig]  {path}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point for generating all data and figures."""
    parser = argparse.ArgumentParser(
        description="Bell-State Initial S--A Entanglement Report (20260621)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all simulations even if Parquet files exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only a specific generator (e.g., 'decoupled-baseline')",
    )
    args = parser.parse_args()

    force = args.force

    generators: dict[str, tuple[str, str, bool]] = {
        "decoupled-baseline": (
            "Decoupled Baseline Verification",
            "generate_decoupled_baseline",
            False,
        ),
        "scenario-scan": (
            "(Scenario, ω) Full Scan",
            "generate_scenario_scan",
            True,
        ),
    }

    if args.only:
        if args.only not in generators:
            print(f"Unknown generator: {args.only}")
            print(f"Available: {list(generators.keys())}")
            import sys

            sys.exit(1)
        gen_list = [args.only]
    else:
        gen_list = list(generators.keys())

    # Ensure raw_data and figures directories exist
    for d in ["raw_data", "figures"]:
        (REPORTS_DIR / REPORT_DATE / d).mkdir(parents=True, exist_ok=True)

    for key in gen_list:
        name, func_name, _ = generators[key]
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        func = globals()[func_name]
        func(force=force)


if __name__ == "__main__":
    main()

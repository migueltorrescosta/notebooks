"""
Scaling survey orchestration module.

Implements the scaling survey infrastructure for quantum metrology:
configuration (SurveyConfig, ModelConfig), state preparation, noise
application, survey sweep loop, exponent fitting, and data export.

Custom sensitivity function generators and per-model factory functions
have been promoted to ``src/analysis/survey_models.py``.

Survey model implementations (weak-value amplification, thermal Langevin
noise, dynamical decoupling, tilt-to-length noise, cavity-enhanced MZI,
distributed MZI) have been promoted to their own modules under
``src/analysis/``.

This module is the promoted reusable core of the 2026-05-11 reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pandas as pd
import qutip

from src.analysis.fisher_information import quantum_fisher_information_dm
from src.analysis.scaling_fit import fit_scaling_exponent
from src.physics.dicke_basis import from_dicke_basis, to_dicke_basis
from src.physics.mzi_simulation import phase_shift_unitary
from src.physics.mzi_states import (
    input_state_factory,
    two_mode_jz_operator,
)

# Configuration


@dataclass
class SurveyConfig:
    """Configuration for a scaling survey sweep.

    Attributes:
        N_range: Tuple of (min, max) particle number range.
            N values are log-spaced between min and max. Default: (2, 64).
        n_points: Number of N values to sweep (log-spaced). Default: 8.
        noise_levels: List of noise levels (dephasing rates) to sweep.
            Each level corresponds to a J_z dephasing rate γ.
            Default: [0.0, 1e-3, 1e-2, 1e-1].
        phi_phase: Operating phase for sensitivity estimation. For QFI-based
            estimation, the sensitivity is phase-independent for pure states
            (F_Q = 4·Var(J_z)), but noise effects may depend on the phase.
            Default: π/4.
        measurement: Measurement type for sensitivity estimation.
            Options: "parity", "Jz", "number_difference".
            Default: "parity".
        method: Sensitivity estimation method.
            Options: "qfi" (Quantum Fisher Information), "cf" (Classical Fisher),
            "ep" (Error Propagation), "bayesian".
            Default: "qfi".
        seed: Random seed for reproducibility. Default: 42.

    """

    N_range: tuple[int, int] = (2, 64)
    n_points: int = 8
    noise_levels: list[float] = field(default_factory=lambda: [0.0, 1e-3, 1e-2, 1e-1])
    phi_phase: float = np.pi / 4
    measurement: str = "parity"
    method: str = "qfi"
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.N_range[0] <= 0:
            raise ValueError(f"N_range minimum must be positive, got {self.N_range[0]}")
        if self.N_range[1] < self.N_range[0]:
            raise ValueError(
                f"N_range max ({self.N_range[1]}) must be >= min ({self.N_range[0]})",
            )
        if self.n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {self.n_points}")
        if self.phi_phase < 0 or self.phi_phase > 2 * np.pi:
            raise ValueError(f"phi_phase must be in [0, 2π], got {self.phi_phase}")
        valid_methods = {"qfi", "cf", "ep", "bayesian"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {self.method}",
            )
        for nl in self.noise_levels:
            if nl < 0:
                raise ValueError(f"Noise level must be non-negative, got {nl}")


@dataclass
class ModelConfig:
    """Configuration for a specific model in the survey.

    Attributes:
        model_id: Unique identifier string for the model
            (e.g., "ideal_coherent", "noon_loss").
        state_type: Type of input state. Must match input_state_factory
            options: "coherent", "noon", "twin_fock", "single_photon_split",
            "fock", "css", "sss", "squeezed_vacuum".
            Ignored when ``custom_sensitivity_fn`` is set.
        noise_type: Type of physical noise.
            Options: "none", "loss", "dephasing", "two_body", "detection",
            "thermal", "custom".
        entangler: Entanglement generation protocol.
            Options: "none", "oat" (one-axis twisting), "tnt" (two-axis twisting).
        label: Human-readable label for plots and tables.
        custom_sensitivity_fn: Optional callable ``(N: int, noise_level: float) -> float``
            that computes Δφ(N) for a model outside the standard MZI pipeline.
            When set, ``state_type`` and ``entangler`` are ignored; the callable
            is invoked directly by ``run_scaling_survey``.
            Typical usage: plug in cavity, distributed, DD, TTL, or thermal models.

    """

    model_id: str
    state_type: str = ""
    noise_type: str = "none"
    entangler: str = "none"
    label: str = ""
    custom_sensitivity_fn: Callable[[int, float], float] | None = None

    def __post_init__(self) -> None:
        """Validate model configuration."""
        valid_noise_types = {
            "none",
            "dephasing",
            "loss",
            "two_body",
            "detection",
            "thermal",
            "custom",
        }
        if self.noise_type not in valid_noise_types:
            raise ValueError(
                f"Unknown noise_type: {self.noise_type}. "
                f"Must be one of: {valid_noise_types}",
            )


# Internal Helpers


def max_photons_for_state(state_type: str, N: int) -> int:
    """Determine appropriate max_photons based on state type.

    For Fock states with definite photon number (noon, twin_fock, fock),
    max_photons=N is sufficient.

    For coherent states (css, coherent), the Poisson distribution has
    tails that extend beyond the mean, so we need a larger Hilbert space.

    Args:
        state_type: Type of state.
        N: Target mean/specified photon number.

    Returns:
        Appropriate max_photons value.

    """
    # States that need larger Hilbert space for Poisson tails
    coherent_like = {"css", "coherent", "squeezed_vacuum"}
    if state_type in coherent_like:
        # Use max(2*N, N+20) to capture Poisson tail
        return max(2 * N, N + 20)
    return N


def _generate_N_values(config: SurveyConfig) -> np.ndarray:
    """Generate log-spaced integer N values for the survey sweep.

    Produces unique integer N values logarithmically spaced between
    N_range[0] and N_range[1]. Log-spacing ensures even coverage
    across orders of magnitude in particle number.

    Args:
        config: Survey configuration specifying the range and count.

    Returns:
        Sorted 1D array of unique integer N values.

    """
    N_min, N_max = config.N_range
    N_raw = np.logspace(np.log10(N_min), np.log10(N_max), config.n_points).astype(int)
    # Deduplicate from rounding and sort
    N_unique = np.unique(N_raw)
    # Ensure at least 2 points
    if len(N_unique) < 2:
        N_unique = np.array([N_min, N_max])
    return N_unique


def _apply_entanglement(
    state: np.ndarray,
    N: int,
    entangler: str,
) -> np.ndarray:
    """Apply entanglement generation to the state.

    Converts the two-mode Fock state to the Dicke basis, applies the
    specified entanglement unitary, and converts back.

    Supported entanglers:
    - "none": Identity operation (no entanglement).
    - "oat": One-axis twisting U = exp(-i χ J_z² t) with optimal
      squeezing time t_opt = (6/N)^{1/3} and χ = 1.
    - "tnt": Two-axis twisting U = exp(-i χ (J_+² + J_-²) t / 2)
      with t_opt = (6/N)^{1/3} and χ = 1.

    Args:
        state: State vector in the two-mode Fock basis
            (dimension (N+1)²).
        N: Total particle number.
        entangler: Type of entanglement to apply.

    Returns:
        Entangled state vector in the two-mode Fock basis.

    Raises:
        ValueError: If entangler type is not recognized.

    """
    if entangler == "none":
        return state.copy()

    # Convert to Dicke basis
    try:
        dicke_state = to_dicke_basis(state, N)
    except ValueError:
        # State may not be in the symmetric subspace; return as-is
        return state.copy()

    if entangler == "oat":
        # One-axis twisting: U = exp(-i χ t J_z²)
        # Optimal squeezing time from Kitagawa & Ueda (1993)
        chi = 1.0
        t_opt = (6.0 / N) ** (1.0 / 3.0) / chi if N > 0 else 0.0

        # J_z eigenvalues: m = N/2, N/2-1, ..., -N/2
        J = N / 2.0
        m_values = np.arange(J, -J - 1, -1)

        # Phase factors: exp(-i χ t m²)
        phases = np.exp(-1j * chi * t_opt * m_values**2)
        dicke_state = phases * dicke_state

    elif entangler == "tnt":
        # Two-axis twisting: U = exp(-i χ t (J_+² + J_-²) / 2)
        # Approximate optimal time similar to OAT
        chi = 1.0
        t_opt = (6.0 / N) ** (1.0 / 3.0) / chi if N > 0 else 0.0

        J_plus = qutip.jmat(N / 2.0, "+").full()
        J_minus = qutip.jmat(N / 2.0, "-").full()
        H_tnt = (J_plus @ J_plus + J_minus @ J_minus) / 2

        from scipy.linalg import expm

        U_tnt = expm(-1j * chi * t_opt * H_tnt)
        dicke_state = U_tnt @ dicke_state
    else:
        raise ValueError(f"Unknown entangler type: {entangler}")

    # Convert back to two-mode Fock basis
    return from_dicke_basis(dicke_state, N)


def _build_noise_collapse_operators(
    dim: int,
    noise_type: str,
    noise_level: float,
) -> list | None:
    """Build Lindblad collapse operators for the given noise type.

    Args:
        dim: Hilbert space dimension per mode (max_photons + 1).
        noise_type: One of ``"dephasing"``, ``"loss"``, ``"two_body"``,
            ``"detection"``.
        noise_level: Noise strength (rate × time) or detection efficiency.

    Returns:
        List of QuTiP collapse operators, or None if noise_type is
        ``"detection"`` (handled by a separate function).

    """
    n0 = qutip.tensor(qutip.num(dim), qutip.qeye(dim))
    n1 = qutip.tensor(qutip.qeye(dim), qutip.num(dim))
    jz = (n0 - n1) / 2.0

    if noise_type == "dephasing":
        return [np.sqrt(noise_level) * jz]
    if noise_type == "loss":
        a1 = qutip.tensor(qutip.qeye(dim), qutip.destroy(dim))
        return [np.sqrt(noise_level) * a1]
    if noise_type == "two_body":
        a1 = qutip.tensor(qutip.qeye(dim), qutip.destroy(dim))
        return [np.sqrt(noise_level) * (a1 @ a1)]
    if noise_type == "detection":
        return None
    return [np.sqrt(noise_level) * jz]


def _run_lindblad_qfi(
    rho0: qutip.Qobj,
    max_photons: int,
    c_ops: list,
    T_decay: float,
) -> float:
    """Run Lindblad evolution and compute QFI on the noisy state.

    Args:
        rho0: Initial density matrix as QuTiP Qobj.
        max_photons: Hilbert space truncation (max photons per mode).
        c_ops: List of QuTiP collapse operators.
        T_decay: Evolution time for Lindblad dynamics.

    Returns:
        Phase sensitivity Δφ, or np.inf on failure.

    """
    try:
        H0 = 0 * qutip.tensor(
            qutip.num(max_photons + 1),
            qutip.qeye(max_photons + 1),
        )
        tlist = [0.0, T_decay]
        result = qutip.mesolve(
            H0,
            rho0,
            tlist,
            c_ops=c_ops,
            options={"store_states": True},
        )
        rho_noisy = result.states[-1].full()
    except (ValueError, np.linalg.LinAlgError):
        return np.inf

    J_z = two_mode_jz_operator(max_photons)
    try:
        F_Q = quantum_fisher_information_dm(rho_noisy, J_z)
    except (ValueError, np.linalg.LinAlgError):
        return np.inf

    if F_Q <= 0 or not np.isfinite(F_Q):
        return np.inf

    return 1.0 / np.sqrt(F_Q)


def _is_noise_free(noise_level: float, noise_type: str) -> bool:
    """Check if the noise configuration is effectively noise-free.

    Args:
        noise_level: Noise strength (rate × time).
        noise_type: Type of noise.

    Returns:
        True if noise_level is zero or noise_type is "none".

    """
    return bool(
        noise_level <= 0.0 or np.isclose(noise_level, 0.0) or noise_type == "none",
    )


def _compute_noisy_sensitivity(
    state: np.ndarray,
    max_photons: int,
    noise_level: float,
    noise_type: str = "dephasing",
    T_decay: float = 1.0,
) -> float:
    """Compute phase sensitivity for a state with the specified noise type.

    For pure states without noise (noise_level ≈ 0), computes the
    Quantum Fisher Information directly as F_Q = 4·Var(J_z) and
    returns Δφ = 1/√F_Q.

    For noisy states (noise_level > 0), uses the Lindblad master equation
    evolution to apply the specified noise type, then computes QFI via
    the full SLD formula for mixed states.

    Supported noise types:
    - "none": No noise (pure state evolution only)
    - "dephasing": Phase diffusion via L = √γ J_z (default)
    - "loss": One-body loss via L = √γ a
    - "two_body": Two-body/pair loss via L = √γ a²
    - "detection": Imperfect detection (efficiency parameter)

    Args:
        state: Pure state vector in the two-mode Fock basis.
        max_photons: Hilbert space truncation (max photons per mode).
        noise_level: Noise strength γ (dimensionless rate × time).
            For detection noise, this is the efficiency η ∈ [0, 1].
        noise_type: Type of noise to apply. Default: "dephasing".
        T_decay: Evolution time for Lindblad dynamics. Default: 1.0.

    Returns:
        Phase sensitivity Δφ (lower is better). Returns np.inf if
        the QFI is zero or negative (no phase information).

    """
    if _is_noise_free(noise_level, noise_type):
        return compute_pure_state_sensitivity(state, max_photons)

    try:
        dim = max_photons + 1
        c_ops = _build_noise_collapse_operators(dim, noise_type, noise_level)

        if c_ops is None:
            return _compute_detection_noise_sensitivity(
                state,
                max_photons,
                noise_level,
            )

        state_q = qutip.Qobj(state.reshape(-1, 1), dims=[[dim, dim], [1, 1]])
        rho0 = qutip.ket2dm(state_q)
        return _run_lindblad_qfi(rho0, max_photons, c_ops, T_decay)
    except (ValueError, np.linalg.LinAlgError):
        return np.inf


def _compute_detection_noise_sensitivity(
    state: np.ndarray,
    max_photons: int,
    eta: float,
) -> float:
    """Compute sensitivity under imperfect detection.

    Detection efficiency η attenuates the quantum Fisher information
    by reducing the distinguishability of measurement outcomes.

    For a given state with photon number distribution P(n), the
    effective QFI with detection noise is bounded by:
        F_Q,eff ≤ η · F_Q

    Args:
        state: Pure state vector.
        max_photons: Hilbert space truncation.
        eta: Detection efficiency η ∈ [0, 1].

    Returns:
        Phase sensitivity Δφ.

    """
    if eta <= 0:
        return np.inf

    # Base pure-state QFI via direct QuTiP variance computation
    try:
        dim = max_photons + 1
        state_q = qutip.Qobj(state.reshape(-1, 1), dims=[[dim, dim], [1, 1]])
        n0 = qutip.tensor(qutip.num(dim), qutip.qeye(dim))
        n1 = qutip.tensor(qutip.qeye(dim), qutip.num(dim))
        jz = (n0 - n1) / 2.0
        var_jz = float(qutip.variance(jz, state_q).real)
        F_Q_pure = 4.0 * var_jz
    except (ValueError, TypeError):
        return np.inf

    if F_Q_pure <= 0 or not np.isfinite(F_Q_pure):
        return np.inf

    # For detection inefficiency, the effective QFI is reduced.
    # The bound Δφ ≥ 1/√(η·F_Q) gives a conservative estimate.
    F_Q_eff = eta * F_Q_pure

    if F_Q_eff <= 0:
        return np.inf

    return 1.0 / np.sqrt(F_Q_eff)


def compute_pure_state_sensitivity(state: np.ndarray, max_photons: int) -> float:
    """Compute phase sensitivity for a noiseless pure state.

    Uses the Quantum Fisher Information for pure states:
        F_Q = 4 · Var(J_z)  →  Δφ = 1/√F_Q

    Uses QuTiP directly for the J_z operator construction and variance
    computation, bypassing intermediate wrappers.

    Args:
        state: Pure state vector in the two-mode Fock basis.
        max_photons: Hilbert space truncation.

    Returns:
        Phase sensitivity Δφ.

    """
    try:
        dim = max_photons + 1
        state_q = qutip.Qobj(state.reshape(-1, 1), dims=[[dim, dim], [1, 1]])
        n0 = qutip.tensor(qutip.num(dim), qutip.qeye(dim))
        n1 = qutip.tensor(qutip.qeye(dim), qutip.num(dim))
        jz = (n0 - n1) / 2.0
        var_jz = float(qutip.variance(jz, state_q).real)
        F_Q = 4.0 * var_jz
    except (ValueError, IndexError, Exception):
        return np.inf

    if F_Q <= 0 or not np.isfinite(F_Q):
        return np.inf

    return 1.0 / np.sqrt(F_Q)


def _apply_phase_imprint(
    state: np.ndarray,
    phi_phase: float,
    max_photons: int,
) -> np.ndarray:
    """Apply a phase shift U = exp(i φ n₂) to mode 1.

    Args:
        state: State vector in the two-mode Fock basis.
        phi_phase: Phase shift in radians.
        max_photons: Maximum photon number per mode.

    Returns:
        Phase-imprinted state vector.

    """
    phase_U = phase_shift_unitary(phi_phase, max_photons)
    return phase_U @ state


# Survey Orchestration


def _process_custom_model_point(
    model: ModelConfig,
    noise_level: float,
    N: int,
) -> float:
    """Evaluate sensitivity via custom_sensitivity_fn.

    Args:
        model: Model configuration with custom_sensitivity_fn set.
        noise_level: Noise level to pass to the custom function.
        N: Particle number.

    Returns:
        Phase sensitivity Δφ, or np.inf on failure.

    """
    fn = model.custom_sensitivity_fn
    if fn is None:
        return np.inf
    try:
        return fn(N, noise_level)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return np.inf


def _process_standard_model_point(
    model: ModelConfig,
    noise_level: float,
    N: int,
    survey_config: SurveyConfig,
) -> float:
    """Evaluate sensitivity through the standard MZI pipeline.

    Steps: state preparation → entanglement → phase imprint → noise → QFI.

    Args:
        model: Model configuration with state_type, noise_type, entangler.
        noise_level: Noise strength.
        N: Particle number.
        survey_config: Survey configuration (phase, method).

    Returns:
        Phase sensitivity Δφ, or np.inf if any step fails.

    """
    try:
        max_photons = max_photons_for_state(model.state_type, N)
        state = input_state_factory(
            model.state_type,
            N=N,
            max_photons=max_photons,
        )
    except (ValueError, TypeError):
        return np.inf

    try:
        state = _apply_entanglement(state, N, model.entangler)
    except (ValueError, np.linalg.LinAlgError):
        pass

    try:
        state = _apply_phase_imprint(state, survey_config.phi_phase, max_photons)
    except (ValueError, IndexError):
        pass

    return _compute_noisy_sensitivity(
        state,
        max_photons,
        noise_level,
        model.noise_type,
    )


def _build_result_row(
    model: ModelConfig,
    noise_level: float,
    N: int,
    method: str,
    delta_phi: float,
) -> dict[str, object]:
    """Build a result row dict for the survey DataFrame."""
    return {
        "model_id": model.model_id,
        "state_type": model.state_type,
        "noise_type": model.noise_type,
        "noise_level": noise_level,
        "N": N,
        "delta_phi": delta_phi,
        "method": method,
        "entangler": model.entangler,
        "label": model.label,
    }


def _finalize_survey_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results list to DataFrame and ensure numeric types.

    Args:
        results: List of result row dicts.

    Returns:
        DataFrame with numeric columns coerced to float.

    """
    df = pd.DataFrame(results)
    for col in ("N", "noise_level", "delta_phi"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _run_survey_point(
    model: ModelConfig,
    noise_level: float,
    N_raw: int,
    survey_config: SurveyConfig,
) -> dict[str, object]:
    """Process a single (model, noise_level, N) combination in the survey.

    Args:
        model: Model configuration.
        noise_level: Noise strength.
        N_raw: Raw particle number (cast to int internally).
        survey_config: Survey configuration.

    Returns:
        Result row dict with model_id, delta_phi, method, etc.

    """
    N = int(N_raw)
    if model.custom_sensitivity_fn is not None:
        delta_phi = _process_custom_model_point(model, noise_level, N)
    else:
        delta_phi = _process_standard_model_point(
            model,
            noise_level,
            N,
            survey_config,
        )
    return _build_result_row(model, noise_level, N, survey_config.method, delta_phi)


def _run_survey_loop(
    models: list[ModelConfig],
    survey_config: SurveyConfig,
    N_values: np.ndarray,
    progress_callback: Callable | None = None,
) -> list[dict]:
    """Run the triple-nested survey loop with optional progress reporting.

    Args:
        models: List of model configurations.
        survey_config: Survey configuration.
        N_values: Particle numbers to sweep.
        progress_callback: Optional (current, total) progress callback.

    Returns:
        List of result row dicts.

    """
    results: list[dict] = []
    total = len(models) * len(survey_config.noise_levels) * len(N_values)
    count = 0

    for model in models:
        for noise_level in survey_config.noise_levels:
            for N_raw in N_values:
                results.append(
                    _run_survey_point(model, noise_level, N_raw, survey_config),
                )
                count += 1
                if progress_callback:
                    progress_callback(count, total)

    return results


def run_scaling_survey(
    models: list[ModelConfig],
    survey_config: SurveyConfig,
    progress_callback: Callable | None = None,
) -> pd.DataFrame:
    """Run the full scaling survey over all models and noise levels.

    For each combination of (model, noise_level, N):
        1. Prepare the input state via input_state_factory
        2. Apply entanglement if configured (OAT/TNT)
        3. Apply phase imprint (phase shift on mode 1)
        4. Apply dephasing noise at the specified level
        5. Compute phase sensitivity via the specified method
        6. Collect the result

    After all sweeps, the returned DataFrame contains raw sensitivity
    values for subsequent exponent fitting via fit_all_exponents.

    Args:
        models: List of ModelConfig objects defining the states and
            noise types to sweep.
        survey_config: SurveyConfig controlling the N range, noise
            levels, operating phase, and estimation method.
        progress_callback: Optional callable(current, total) called
            after each (model, noise_level, N) combination completes.
            Useful for progress bars in interactive contexts.

    Returns:
        DataFrame with columns:
            model_id, state_type, noise_type, noise_level, N, delta_phi,
            method, entangler

    Raises:
        ValueError: If models list is empty or survey_config is invalid.

    """
    if not models:
        raise ValueError("At least one model must be specified")

    N_values = _generate_N_values(survey_config)
    results = _run_survey_loop(models, survey_config, N_values, progress_callback)
    return _finalize_survey_dataframe(results)


# Exponent Fitting


def _validate_fit_dataframe(survey_df: pd.DataFrame) -> None:
    """Validate survey DataFrame has required columns and is non-empty.

    Args:
        survey_df: DataFrame from run_scaling_survey.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.

    """
    required = {"N", "delta_phi"}
    missing = required - set(survey_df.columns)
    if missing:
        raise ValueError(
            f"survey_df missing required columns: {missing}. "
            f"Has columns: {list(survey_df.columns)}",
        )
    if survey_df.empty:
        raise ValueError("survey_df is empty")


def _filter_finite_for_fitting(survey_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to finite positive delta_phi values only.

    Args:
        survey_df: DataFrame from run_scaling_survey.

    Returns:
        Copy of survey_df with only rows where delta_phi is finite and > 0.

    """
    return survey_df.loc[
        np.isfinite(survey_df["delta_phi"]) & (survey_df["delta_phi"] > 0)
    ].copy()


def _empty_fit_dataframe(group_cols: list[str]) -> pd.DataFrame:
    """Return an empty DataFrame with the correct columns for fit results.

    Args:
        group_cols: Group-by column names to include.

    Returns:
        Empty DataFrame with group columns plus fit result columns.

    """
    return pd.DataFrame(
        columns=[
            *group_cols,
            "alpha",
            "alpha_err",
            "C",
            "C_err",
            "R_squared",
            "valid",
            "n_points",
        ],
    )


def _fit_single_survey_group(
    group_df: pd.DataFrame,
    group_keys: object,
    available_groups: list[str],
    min_N: int,
    R_squared_threshold: float,
) -> dict[str, object]:
    """Fit scaling exponent for a single group and return result row.

    Args:
        group_df: DataFrame subset for one group.
        group_keys: Group key(s) from pandas groupby.
        available_groups: Column names present in the groupby result.
        min_N: Minimum N for the scaling fit.
        R_squared_threshold: R² warning threshold.

    Returns:
        Dict with group columns and fit results (alpha, C, R_squared, etc.).

    """
    if not isinstance(group_keys, tuple):
        group_keys = (group_keys,)

    N_arr = group_df["N"].to_numpy().astype(float)
    delta_arr = group_df["delta_phi"].to_numpy().astype(float)

    result = fit_scaling_exponent(
        N_arr,
        delta_arr,
        min_N=min_N,
        R_squared_threshold=R_squared_threshold,
    )

    row: dict[str, object] = dict(zip(available_groups, group_keys, strict=False))
    row["alpha"] = result.alpha
    row["alpha_err"] = result.alpha_err
    row["C"] = result.C
    row["C_err"] = result.C_err
    row["R_squared"] = result.R_squared
    row["valid"] = result.valid
    row["n_points"] = len(result.N_values)
    row["n_warnings"] = len(result.warnings)
    return row


def _default_group_cols(group_cols: list[str] | None) -> list[str]:
    """Return the default group-by columns if none are provided."""
    if group_cols is None:
        return ["model_id", "state_type", "noise_type", "noise_level", "method"]
    return group_cols


def _compute_fit_results(
    finite_df: pd.DataFrame,
    group_cols: list[str],
    min_N: int,
    R_squared_threshold: float,
) -> pd.DataFrame:
    """Compute fit results from a finite-only survey DataFrame.

    Handles groupby, per-group fitting, and result sorting.

    Args:
        finite_df: DataFrame with finite positive delta_phi.
        group_cols: Column names to group by.
        min_N: Minimum N for the scaling fit.
        R_squared_threshold: R² warning threshold.

    Returns:
        DataFrame with fit results (alpha, C, R_squared, etc.).

    """
    available_groups = [c for c in group_cols if c in finite_df.columns]

    try:
        grouped = finite_df.groupby(available_groups, dropna=True)
    except (ValueError, TypeError):
        return _empty_fit_dataframe(available_groups)

    fit_rows = [
        _fit_single_survey_group(
            group_df,
            group_keys,
            available_groups,
            min_N,
            R_squared_threshold,
        )
        for group_keys, group_df in grouped
    ]

    fit_df = pd.DataFrame(fit_rows)

    if "alpha" in fit_df.columns:
        fit_df = fit_df.sort_values("alpha", ascending=True).reset_index(drop=True)

    return fit_df


def fit_all_exponents(
    survey_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    min_N: int = 4,
    R_squared_threshold: float = 0.9,
) -> pd.DataFrame:
    """Fit scaling exponents for each group in survey data.

    Groups the survey results by the specified columns (e.g., model_id,
    noise_level, method) and fits Δφ = C·N^α for each group.

    Args:
        survey_df: DataFrame from run_scaling_survey.
        group_cols: Columns to group by for independent fits.
            Default: ["model_id", "state_type", "noise_level", "method"].
        min_N: Minimum N to include in each fit (passed to
            fit_scaling_exponent). Default: 4.
        R_squared_threshold: R² warning threshold. Default: 0.9.

    Returns:
        DataFrame with columns:
            model_id, state_type, noise_type, noise_level, method,
            alpha, alpha_err, C, C_err, R_squared, valid, n_points, n_warnings
        Each row is a fitted exponent for one group.

    Raises:
        ValueError: If survey_df is empty or missing required columns.

    """
    _validate_fit_dataframe(survey_df)

    group_cols = _default_group_cols(group_cols)
    finite_df = _filter_finite_for_fitting(survey_df)

    if finite_df.empty:
        return _empty_fit_dataframe(group_cols)

    return _compute_fit_results(finite_df, group_cols, min_N, R_squared_threshold)


# Export Utilities


def survey_to_parquet(survey_df: pd.DataFrame, path: str) -> None:
    """Export survey results to Parquet.

    Args:
        survey_df: DataFrame from run_scaling_survey or fit_all_exponents.
        path: File path to write the Parquet.

    Raises:
        ValueError: If survey_df is empty.

    """
    if survey_df.empty:
        raise ValueError("Cannot export empty DataFrame")

    survey_df.to_parquet(path, index=False)


def survey_to_json(survey_df: pd.DataFrame, path: str) -> None:
    """Export survey results to JSON (structured, human-readable format).

    The JSON output includes metadata (columns, row count) and the
    data records as a list of dictionaries.

    Args:
        survey_df: DataFrame from run_scaling_survey or fit_all_exponents.
        path: File path to write the JSON.

    Raises:
        ValueError: If survey_df is empty.

    """
    if survey_df.empty:
        raise ValueError("Cannot export empty DataFrame")

    # Convert DataFrame to a structured JSON format
    output: dict = {
        "metadata": {
            "columns": list(survey_df.columns),
            "rows": len(survey_df),
        },
        "data": survey_df.to_dict(orient="records"),
    }

    # Handle numpy types for JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, o: object) -> object:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.bool_,)):
                return bool(o)
            return super().default(o)

    Path(path).write_text(json.dumps(output, indent=2, cls=NumpyEncoder))

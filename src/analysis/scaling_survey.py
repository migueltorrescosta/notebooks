"""
Unified Scaling Survey Orchestrator for Interferometry Sensitivity.

Sweeps over a grid of (model, N, noise level) and extracts
scaling exponents for each combination. Provides a composable
pipeline for sensitivity scaling analysis.

Pipeline:
    input_state(N, state_type) → entangle(state) → phase_imprint(state, phi)
    → decohere(state, noise_config) → sensitivity_estimator(state, method)
    → extract_exponent(N_array, delta_phi_array)

Physical Model:
- Two-mode bosonic interferometer states
- Optional spin squeezing (OAT) for entanglement generation
- Dephasing noise modelled as J_z-coupling Lindblad channel
- Phase sensitivity computed via Quantum Fisher Information (QFI)

Units:
- Dimensionless throughout. Phase in radians.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import qutip

from src.analysis.fisher_information import quantum_fisher_information_dm
from src.analysis.scaling_fit import fit_scaling_exponent
from src.physics.dicke_basis import from_dicke_basis, to_dicke_basis
from src.physics.hybrid_mzi import qfi_hybrid_mzi
from src.physics.hybrid_system import (
    hybrid_ground_state_n,
    hybrid_hamiltonian_n,
    hybrid_vacuum_state,
)
from src.physics.mzi_simulation import phase_shift_unitary
from src.physics.mzi_states import (
    input_state_factory,
    two_mode_jz_operator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# Custom Sensitivity Function Generators
# =============================================================================


def _non_gaussian_sensitivity_fn(
    n_order: int,
    omega_n: float = 0.5,
    theta_n: float = 0.0,
    t_sqz: float = 2.0,
    use_ground_state: bool = False,
) -> Callable[[int, float], float]:
    """Create a sensitivity function for n-th order non-Gaussian states.

    Uses the hybrid oscillator-spin system from ``hybrid_system.py``.
    The ``N`` parameter in the survey maps to the oscillator Fock truncation;
    the squeezing parameters ``omega_n``, ``theta_n``, ``t_sqz`` are fixed at
    construction.

    Two state preparation modes:
    - ``use_ground_state=False`` (default): time-evolve |0,↓⟩ under H_n for t_sqz
    - ``use_ground_state=True``: lowest-energy eigenstate of H_n via
      :func:`hybrid_ground_state_n`

    Args:
        n_order: Squeezing order (2, 3, or 4).
        omega_n: Squeezing rate Ω_n. Default 0.5.
        theta_n: Squeezing phase θ_n. Default 0.0.
        t_sqz: Squeezing evolution time. Default 2.0.
        use_ground_state: If True, use true ground state instead of
            time-evolved vacuum. Default False.

    Returns:
        Callable ``(N: int, noise_level: float) -> delta_phi: float``
        suitable for use as ``ModelConfig.custom_sensitivity_fn``.

    Raises:
        ValueError: If n_order is not 2, 3, or 4.

    """
    import scipy.linalg

    if n_order not in (2, 3, 4):
        raise ValueError(f"n_order must be 2, 3, or 4, got {n_order}")

    def _sensitivity(N: int, noise_level: float) -> float:
        # For hybrid-system N maps directly to oscillator truncation
        try:
            if N < 2:
                return np.inf

            if use_ground_state:
                state = hybrid_ground_state_n(N, n_order, omega_n, theta_n)
            else:
                state = hybrid_vacuum_state(N, spin_state="down")
                H = hybrid_hamiltonian_n(N, n_order, omega_n, theta_n)
                U = scipy.linalg.expm(-1j * H * t_sqz)
                state = U @ state

            # QFI via MZI readout (analytical formula — no phase sweep needed)
            fq = qfi_hybrid_mzi(state, N)
        except Exception:
            return np.inf

        if fq <= 0 or not np.isfinite(fq):
            return np.inf

        return 1.0 / np.sqrt(fq)

    return _sensitivity


def _ancilla_sensitivity_fn(
    alpha: float = 1.0,
    g_sa: float = 1.0,
    tau: float = 0.1,
    g_sp: float = 0.0,
    omega_0: float = 0.0,
    lam: float = 0.0,
    K: int = 2,
) -> Callable[[int, float], float]:
    """Create a sensitivity function for ancilla-assisted metrology.

    Uses the pseudomode-based non-Markovian ancilla protocol from
    ``pseudomode_system.py``. The ``N`` parameter maps to oscillator
    Fock truncation; ancilla, bath, and coupling parameters are fixed
    at construction.

    When ``g_sp=0`` and ``lam=0`` (the defaults), the environment is
    effectively Markovian and the protocol reduces to the dispersive
    ancilla-coupling model described in the article.

    Args:
        alpha: Coherent state amplitude for the oscillator probe.
            Default 1.0 (mean photon number = 1).
        g_sa: System-ancilla coupling strength. Default 1.0.
        tau: Ancilla entanglement time. Default 0.1.
        g_sp: System-pseudomode coupling strength. Default 0.0
            (no environment coupling).
        omega_0: Bath central frequency (pseudomode free energy).
            Default 0.0.
        lam: Bath correlation rate (pseudomode damping). Default 0.0
            (no dissipation).
        K: Pseudomode Fock truncation. Default 2.

    Returns:
        Callable ``(N: int, noise_level: float) -> delta_phi: float``
        suitable for use as ``ModelConfig.custom_sensitivity_fn``.

    Note:
        The ``noise_level`` parameter of the survey is used to scale
        the ancilla-system coupling ``g_sa`` (larger noise_level =
        stronger coupling / effective decoherence). The base ``g_sa``
        is multiplied by ``(1 + noise_level)``.

    """
    from src.physics.pseudomode_system import (
        PseudomodeConfig,
        run_metrology_protocol,
    )

    def _sensitivity(N: int, noise_level: float) -> float:
        try:
            if N < 2:
                return np.inf

            config = PseudomodeConfig(
                N=N,
                K=K,
                alpha=alpha,
                g_sa=g_sa * (1.0 + noise_level),  # noise scales coupling
                tau=tau,
                g_sp=g_sp,
                omega_0=omega_0,
                lam=lam,
                T=1.0,
                dt=0.1,
            )
            result = run_metrology_protocol(config)
            fq = result["qfi_with"]
        except Exception:
            return np.inf

        if fq <= 0 or not np.isfinite(fq):
            return np.inf

        return 1.0 / np.sqrt(fq)

    return _sensitivity


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SurveyConfig:
    """Configuration for a scaling survey sweep.

    Attributes:
        N_range: Tuple of (min, max) particle number range.
            N values are log-spaced between min and max. Default: (2, 64).
        N_points: Number of N values to sweep (log-spaced). Default: 8.
        noise_levels: List of noise levels (dephasing rates) to sweep.
            Each level corresponds to a J_z dephasing rate γ.
            Default: [0.0, 1e-3, 1e-2, 1e-1].
        phi: Operating phase for sensitivity estimation. For QFI-based
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
    N_points: int = 8
    noise_levels: list[float] = field(default_factory=lambda: [0.0, 1e-3, 1e-2, 1e-1])
    phi: float = np.pi / 4
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
        if self.N_points < 2:
            raise ValueError(f"N_points must be >= 2, got {self.N_points}")
        if self.phi < 0 or self.phi > 2 * np.pi:
            raise ValueError(f"phi must be in [0, 2π], got {self.phi}")
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


# =============================================================================
# Internal Helpers
# =============================================================================


def _max_photons_for_state(state_type: str, N: int) -> int:
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
    N_raw = np.logspace(np.log10(N_min), np.log10(N_max), config.N_points).astype(int)
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


def _compute_noisy_sensitivity(
    state: np.ndarray,
    max_photons: int,
    noise_level: float,
    noise_type: str = "dephasing",
    T: float = 1.0,
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
        T: Evolution time for Lindblad dynamics. Default: 1.0.

    Returns:
        Phase sensitivity Δφ (lower is better). Returns np.inf if
        the QFI is zero or negative (no phase information).

    """
    noise_free = noise_level <= 0.0 or np.isclose(noise_level, 0.0)
    no_noise_type = noise_type == "none"

    if noise_free or no_noise_type:
        return _compute_pure_state_sensitivity(state, max_photons)

    # --- Noisy case: use QuTiP Lindblad evolution ---
    # Build Lindblad collapse operators based on noise_type
    try:
        dim = max_photons + 1
        n0 = qutip.tensor(qutip.num(dim), qutip.qeye(dim))
        n1 = qutip.tensor(qutip.qeye(dim), qutip.num(dim))
        jz = (n0 - n1) / 2.0

        if noise_type == "dephasing":
            c_ops = [np.sqrt(noise_level) * jz]
        elif noise_type == "loss":
            a1 = qutip.tensor(qutip.qeye(dim), qutip.destroy(dim))
            c_ops = [np.sqrt(noise_level) * a1]
        elif noise_type == "two_body":
            a1 = qutip.tensor(qutip.qeye(dim), qutip.destroy(dim))
            c_ops = [np.sqrt(noise_level) * (a1 * a1)]
        elif noise_type == "detection":
            return _compute_detection_noise_sensitivity(state, max_photons, noise_level)
        else:
            c_ops = [np.sqrt(noise_level) * jz]

        H0 = 0 * n0  # zero Hamiltonian with matching dims
        state_q = qutip.Qobj(state.reshape(-1, 1), dims=[[dim, dim], [1, 1]])
        rho0 = qutip.ket2dm(state_q)
        tlist = [0.0, T]

        result = qutip.mesolve(
            H0, rho0, tlist, c_ops=c_ops, options={"store_states": True}
        )
        rho_noisy = result.states[-1].full()
    except Exception:
        return np.inf

    # J_z operator in the two-mode Fock basis
    J_z = two_mode_jz_operator(max_photons)

    # Compute QFI via SLD formula for mixed states
    try:
        F_Q = quantum_fisher_information_dm(rho_noisy, J_z)
    except (ValueError, np.linalg.LinAlgError):
        return np.inf

    if F_Q <= 0 or not np.isfinite(F_Q):
        return np.inf

    return 1.0 / np.sqrt(F_Q)


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
        var_jz = float(qutip.variance(jz, state_q))
        F_Q_pure = 4.0 * var_jz
    except Exception:
        return np.inf

    if F_Q_pure <= 0 or not np.isfinite(F_Q_pure):
        return np.inf

    # For detection inefficiency, the effective QFI is reduced.
    # The bound Δφ ≥ 1/√(η·F_Q) gives a conservative estimate.
    F_Q_eff = eta * F_Q_pure

    if F_Q_eff <= 0:
        return np.inf

    return 1.0 / np.sqrt(F_Q_eff)


def _compute_pure_state_sensitivity(state: np.ndarray, max_photons: int) -> float:
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
        var_jz = float(qutip.variance(jz, state_q))
        F_Q = 4.0 * var_jz
    except (ValueError, IndexError, Exception):
        return np.inf

    if F_Q <= 0 or not np.isfinite(F_Q):
        return np.inf

    return 1.0 / np.sqrt(F_Q)


def _apply_phase_imprint(
    state: np.ndarray,
    phi: float,
    max_photons: int,
) -> np.ndarray:
    """Apply a phase shift U = exp(i φ n₂) to mode 1.

    Args:
        state: State vector in the two-mode Fock basis.
        phi: Phase shift in radians.
        max_photons: Maximum photon number per mode.

    Returns:
        Phase-imprinted state vector.

    """
    phase_U = phase_shift_unitary(phi, max_photons)
    return phase_U @ state


# =============================================================================
# Survey Orchestration
# =============================================================================


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

    # Validate survey config (triggers __post_init__)
    # (already validated at construction, but we can trust the caller)

    N_values = _generate_N_values(survey_config)

    results: list[dict] = []
    total = len(models) * len(survey_config.noise_levels) * len(N_values)
    count = 0

    for model in models:
        for noise_level in survey_config.noise_levels:
            for N_raw in N_values:
                N = int(N_raw)

                # --- Custom model path (bypasses standard MZI pipeline) ---
                if model.custom_sensitivity_fn is not None:
                    try:
                        delta_phi = model.custom_sensitivity_fn(N, noise_level)
                    except Exception:
                        delta_phi = np.inf

                    results.append(
                        {
                            "model_id": model.model_id,
                            "state_type": model.state_type,
                            "noise_type": model.noise_type,
                            "noise_level": noise_level,
                            "N": N,
                            "delta_phi": delta_phi,
                            "method": survey_config.method,
                            "entangler": model.entangler,
                            "label": model.label,
                        },
                    )
                    count += 1
                    if progress_callback:
                        progress_callback(count, total)
                    continue

                # --- Standard MZI pipeline ---

                # --- Step 1: Prepare input state ---
                try:
                    max_photons = _max_photons_for_state(model.state_type, N)
                    state = input_state_factory(
                        model.state_type,
                        N=N,
                        max_photons=max_photons,
                    )
                except (ValueError, TypeError):
                    # If state cannot be created at this N, skip gracefully
                    results.append(
                        {
                            "model_id": model.model_id,
                            "state_type": model.state_type,
                            "noise_type": model.noise_type,
                            "noise_level": noise_level,
                            "N": N,
                            "delta_phi": np.inf,
                            "method": survey_config.method,
                            "entangler": model.entangler,
                            "label": model.label,
                        },
                    )
                    count += 1
                    if progress_callback:
                        progress_callback(count, total)
                    continue

                # --- Step 2: Apply entanglement ---
                # Note: _apply_entanglement uses N for Dicke basis dimension (N+1),
                # which is based on particle number, not Hilbert space truncation
                try:
                    state = _apply_entanglement(state, N, model.entangler)
                except (ValueError, np.linalg.LinAlgError):
                    state = state  # Continue with unentangled state

                # --- Step 3: Apply phase imprint ---
                try:
                    state = _apply_phase_imprint(state, survey_config.phi, max_photons)
                except (ValueError, IndexError):
                    pass  # Continue without phase imprint if it fails

                # --- Steps 4 & 5: Apply noise and compute sensitivity ---
                delta_phi = _compute_noisy_sensitivity(
                    state,
                    max_photons,
                    noise_level,
                    model.noise_type,
                )

                results.append(
                    {
                        "model_id": model.model_id,
                        "state_type": model.state_type,
                        "noise_type": model.noise_type,
                        "noise_level": noise_level,
                        "N": N,
                        "delta_phi": delta_phi,
                        "method": survey_config.method,
                        "entangler": model.entangler,
                        "label": model.label,
                    },
                )

                count += 1
                if progress_callback:
                    progress_callback(count, total)

    df = pd.DataFrame(results)

    # Ensure numeric types
    for col in ["N", "noise_level", "delta_phi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =============================================================================
# Exponent Fitting
# =============================================================================


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
    required = {"N", "delta_phi"}
    missing = required - set(survey_df.columns)
    if missing:
        raise ValueError(
            f"survey_df missing required columns: {missing}. "
            f"Has columns: {list(survey_df.columns)}",
        )

    if survey_df.empty:
        raise ValueError("survey_df is empty")

    if group_cols is None:
        group_cols = ["model_id", "state_type", "noise_type", "noise_level", "method"]

    # Filter out infinite values for fitting
    finite_df = survey_df.loc[
        np.isfinite(survey_df["delta_phi"]) & (survey_df["delta_phi"] > 0)
    ].copy()

    if finite_df.empty:
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

    fit_rows: list[dict] = []
    available_groups = [c for c in group_cols if c in finite_df.columns]

    try:
        grouped = finite_df.groupby(available_groups, dropna=True)
    except Exception:
        return pd.DataFrame(
            columns=[
                *available_groups,
                "alpha",
                "alpha_err",
                "C",
                "C_err",
                "R_squared",
                "valid",
                "n_points",
            ],
        )

    for group_keys, group_df in grouped:
        # Ensure groups is always a tuple
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

        row: dict = dict(zip(available_groups, group_keys, strict=False))
        row["alpha"] = result.alpha
        row["alpha_err"] = result.alpha_err
        row["C"] = result.C
        row["C_err"] = result.C_err
        row["R_squared"] = result.R_squared
        row["valid"] = result.valid
        row["n_points"] = len(result.N_values)
        row["n_warnings"] = len(result.warnings)

        fit_rows.append(row)

    fit_df = pd.DataFrame(fit_rows)

    if "alpha" in fit_df.columns:
        fit_df = fit_df.sort_values("alpha", ascending=True).reset_index(drop=True)

    return fit_df


# =============================================================================
# Export Utilities
# =============================================================================


def survey_to_csv(survey_df: pd.DataFrame, path: str) -> None:
    """Export survey results to CSV.

    Args:
        survey_df: DataFrame from run_scaling_survey or fit_all_exponents.
        path: File path to write the CSV.

    Raises:
        ValueError: If survey_df is empty.

    """
    if survey_df.empty:
        raise ValueError("Cannot export empty DataFrame")

    survey_df.to_csv(path, index=False)


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
        def default(self, obj: object) -> object:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    Path(path).write_text(json.dumps(output, indent=2, cls=NumpyEncoder))


# =============================================================================
# Factory Functions
# =============================================================================


def create_survey_model(
    model_id: str,
    **kwargs: object,
) -> ModelConfig:
    """Create a ModelConfig from a shorthand identifier.

    Provides convenient shorthand creation for common survey models.
    Supports both named models and custom configurations.

    Built-in model IDs:
        - "ideal_coherent":  Coherent state |α⟩, no noise, no entangler
        - "ideal_noon":      NOON state |N,0⟩+|0,N⟩, no noise
        - "ideal_twin_fock": Twin-Fock state, no noise
        - "noon_loss":       NOON state with loss noise
        - "coherent_oat":    Coherent state with OAT spin squeezing
        - "squeezed_vacuum": Squeezed vacuum state, no noise
        - "non_gaussian_n3": Non-Gaussian trisqueezed state (n=3)
        - "non_gaussian_n4": Non-Gaussian quad-squeezed state (n=4)
        - "ancilla_assisted": Ancilla-assisted metrology with dispersive coupling

    Args:
        model_id: Shorthand identifier for the model. Unknown IDs
            produce a model with state_type matching the ID.
        **kwargs: Additional keyword arguments overriding the defaults
            for the given model_id. For example:
            create_survey_model("noon_loss", noise_type="dephasing")
            create_survey_model("non_gaussian_n3", omega_n=1.0, t_sqz=3.0)

    Returns:
        ModelConfig with appropriate defaults for the given model_id.

    """
    defaults: dict[str, str] = {
        "ideal_coherent": "css",  # CSS = coherent state with |alpha|² = N
        "ideal_noon": "noon",
        "ideal_twin_fock": "twin_fock",
        "noon_loss": "noon",
        "coherent_oat": "css",
        "squeezed_vacuum": "squeezed_vacuum",
        "squeezed_vacuum_loss": "squeezed_vacuum",
    }

    noise_defaults: dict[str, str] = {
        "ideal_coherent": "none",
        "ideal_noon": "none",
        "ideal_twin_fock": "none",
        "noon_loss": "loss",
        "coherent_oat": "none",
        "squeezed_vacuum": "none",
        "squeezed_vacuum_loss": "loss",
    }

    entangler_defaults: dict[str, str] = {
        "ideal_coherent": "none",
        "ideal_noon": "none",
        "ideal_twin_fock": "none",
        "noon_loss": "none",
        "coherent_oat": "oat",
        "squeezed_vacuum": "none",
        "squeezed_vacuum_loss": "none",
    }

    label_defaults: dict[str, str] = {
        "ideal_coherent": "Coherent state",
        "ideal_noon": "NOON state",
        "ideal_twin_fock": "Twin-Fock state",
        "noon_loss": "NOON with loss",
        "coherent_oat": "Coherent + OAT",
        "squeezed_vacuum": "Squeezed vacuum",
        "squeezed_vacuum_loss": "Squeezed vacuum with loss",
        "non_gaussian_n3": "Non-Gaussian n=3 (trisqueezed)",
        "non_gaussian_n4": "Non-Gaussian n=4 (quadsqueezed)",
        "ancilla_assisted": "Ancilla-assisted metrology",
    }

    # --- Custom sensitivity models bypass standard pipeline ---
    if model_id == "non_gaussian_n3":
        omega_n = kwargs.get("omega_n", 0.5)
        t_sqz = kwargs.get("t_sqz", 2.0)
        theta_n = kwargs.get("theta_n", 0.0)
        use_gs = kwargs.get("use_ground_state", False)
        fn = _non_gaussian_sensitivity_fn(
            n_order=3,
            omega_n=float(omega_n) if isinstance(omega_n, (int, float)) else 0.5,
            theta_n=float(theta_n) if isinstance(theta_n, (int, float)) else 0.0,
            t_sqz=float(t_sqz) if isinstance(t_sqz, (int, float)) else 2.0,
            use_ground_state=bool(use_gs) if isinstance(use_gs, (bool, int)) else False,
        )
        return ModelConfig(
            model_id=model_id,
            custom_sensitivity_fn=fn,
            state_type="",
            noise_type="none",
            entangler="none",
            label=label_defaults.get(model_id, "Non-Gaussian n=3"),
        )

    if model_id == "non_gaussian_n4":
        omega_n = kwargs.get("omega_n", 0.5)
        t_sqz = kwargs.get("t_sqz", 2.0)
        theta_n = kwargs.get("theta_n", 0.0)
        use_gs = kwargs.get("use_ground_state", False)
        fn = _non_gaussian_sensitivity_fn(
            n_order=4,
            omega_n=float(omega_n) if isinstance(omega_n, (int, float)) else 0.5,
            theta_n=float(theta_n) if isinstance(theta_n, (int, float)) else 0.0,
            t_sqz=float(t_sqz) if isinstance(t_sqz, (int, float)) else 2.0,
            use_ground_state=bool(use_gs) if isinstance(use_gs, (bool, int)) else False,
        )
        return ModelConfig(
            model_id=model_id,
            custom_sensitivity_fn=fn,
            state_type="",
            noise_type="none",
            entangler="none",
            label=label_defaults.get(model_id, "Non-Gaussian n=4"),
        )

    if model_id == "ancilla_assisted":
        alpha = kwargs.get("alpha", 1.0)
        g_sa = kwargs.get("g_sa", 1.0)
        tau = kwargs.get("tau", 0.1)
        g_sp = kwargs.get("g_sp", 0.0)
        lam = kwargs.get("lam", 0.0)
        K = kwargs.get("K", 2)
        fn = _ancilla_sensitivity_fn(
            alpha=float(alpha) if isinstance(alpha, (int, float)) else 1.0,
            g_sa=float(g_sa) if isinstance(g_sa, (int, float)) else 1.0,
            tau=float(tau) if isinstance(tau, (int, float)) else 0.1,
            g_sp=float(g_sp) if isinstance(g_sp, (int, float)) else 0.0,
            lam=float(lam) if isinstance(lam, (int, float)) else 0.0,
            K=int(K) if isinstance(K, int) else 2,
        )
        return ModelConfig(
            model_id=model_id,
            custom_sensitivity_fn=fn,
            state_type="",
            noise_type="none",
            entangler="none",
            label=label_defaults.get(model_id, "Ancilla-assisted metrology"),
        )

    state_type = defaults.get(model_id, model_id)
    noise_type = noise_defaults.get(model_id, "none")
    entangler = entangler_defaults.get(model_id, "none")
    label = label_defaults.get(model_id, model_id.replace("_", " ").title())

    # Override with any provided kwargs
    model_kwargs: dict[str, str] = {}
    for key in ("state_type", "noise_type", "entangler", "label"):
        if key in kwargs:
            val = kwargs[key]
            if isinstance(val, str):
                model_kwargs[key] = val

    return ModelConfig(
        model_id=model_id,
        state_type=model_kwargs.get("state_type", state_type),
        noise_type=model_kwargs.get("noise_type", noise_type),
        entangler=model_kwargs.get("entangler", entangler),
        label=model_kwargs.get("label", label),
    )


def create_default_survey() -> list[ModelConfig]:
    """Create the default set of models for the full scaling survey.

    Returns models representing the most common interferometry states
    used in quantum metrology scaling studies:

    1. ideal_coherent:   Coherent state (SQL scaling, α = -1/2)
    2. ideal_noon:       NOON state (Heisenberg scaling, α = -1)
    3. ideal_twin_fock:  Twin-Fock state (near-Heisenberg, α ≈ -1)
    4. noon_loss:        NOON with loss (transition from Heisenberg to SQL)
    5. coherent_oat:     Coherent state with OAT squeezing (sub-SQL)
    6. squeezed_vacuum:  Squeezed vacuum state (sub-SQL)
    7. non_gaussian_n3:  Non-Gaussian trisqueezed state (n=3)
    8. non_gaussian_n4:  Non-Gaussian quad-squeezed state (n=4)
    9. ancilla_assisted: Ancilla-assisted metrology with dispersive coupling

    Returns:
        List of ModelConfig objects for the default survey.

    """
    model_ids = [
        "ideal_coherent",
        "ideal_noon",
        "ideal_twin_fock",
        "noon_loss",
        "coherent_oat",
        "squeezed_vacuum",
        "squeezed_vacuum_loss",
        "non_gaussian_n3",
        "non_gaussian_n4",
        "ancilla_assisted",
    ]
    return [create_survey_model(mid) for mid in model_ids]

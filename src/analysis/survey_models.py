"""
Survey model definitions for the scaling survey.

Provides custom sensitivity function generators for non-standard
MZI models (non-Gaussian, ancilla-assisted, Kerr, weak-value MZI),
type coercion helpers, per-model factory functions, and the public
``create_survey_model`` / ``create_default_survey`` API.

Extracted from ``src/analysis/scaling_survey.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import scipy.linalg

from src.analysis.scaling_survey import (
    ModelConfig,
    compute_pure_state_sensitivity,
    max_photons_for_state,
)
from src.analysis.weak_value_mzi import WeakValueConfig, weak_value_mzi_sensitivity
from src.physics.hybrid_mzi import qfi_hybrid_mzi
from src.physics.hybrid_system import (
    hybrid_ground_state_n,
    hybrid_hamiltonian_n,
    hybrid_vacuum_state,
)
from src.physics.mzi_states import input_state_factory
from src.physics.pseudomode_system import (
    PseudomodeConfig,
    run_metrology_protocol,
)


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
        omega_n: Squeezing rate Omega_n. Default 0.5.
        theta_n: Squeezing phase theta_n. Default 0.0.
        t_sqz: Squeezing evolution time. Default 2.0.
        use_ground_state: If True, use true ground state instead of
            time-evolved vacuum. Default False.

    Returns:
        Callable ``(N: int, noise_level: float) -> delta_phi: float``
        suitable for use as ``ModelConfig.custom_sensitivity_fn``.

    Raises:
        ValueError: If n_order is not 2, 3, or 4.

    """

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
        except (ValueError, np.linalg.LinAlgError, TypeError):
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
    ancilla-coupling model described in the report.

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
                T_decay=1.0,
                dt=0.1,
            )
            result = run_metrology_protocol(config)
            fq = result["qfi_with"]
        except (KeyError, ValueError, TypeError):
            return np.inf

        if fq <= 0 or not np.isfinite(fq):
            return np.inf

        return 1.0 / np.sqrt(fq)

    return _sensitivity


def _kerr_mzi_sensitivity_fn(
    K: float = 0.1,
    T_kerr: float = 1.0,
    state_type: str = "noon",
) -> Callable[[int, float], float]:
    """Create a sensitivity function for Kerr-nonlinear MZI.

    The Kerr nonlinearity :math:`K (n_1^2 + n_2^2)` commutes with
    the phase generator :math:`n_2`, so the QFI is invariant under Kerr
    evolution. For NOON states this gives :math:`F_Q = N^2` (Heisenberg
    limit) regardless of the Kerr strength.

    Args:
        K: Kerr nonlinearity strength. Default 0.1.
        T_kerr: Evolution time for the Kerr interaction. Default 1.0.
            The product ``K * T_kerr`` controls the nonlinear phase.
        state_type: Input state type passed to ``input_state_factory``.
            Default ``"noon"``.

    Returns:
        Callable ``(N: int, noise_level: float) -> delta_phi: float``
        suitable for use as ``ModelConfig.custom_sensitivity_fn``.

    """
    # Validate state_type
    valid_states = {"noon", "coherent", "css", "twin_fock", "sss", "squeezed_vacuum"}
    if state_type not in valid_states:
        state_type = "noon"

    def _sensitivity(N: int, noise_level: float) -> float:
        if N < 2:
            return np.inf
        try:
            max_photons = max_photons_for_state(state_type, N)
            state = input_state_factory(state_type, N=N, max_photons=max_photons)
            # Pure-state QFI: F_Q = 4·Var(J_z) — invariant under Kerr
            delta = compute_pure_state_sensitivity(state, max_photons)
        except (ValueError, np.linalg.LinAlgError, TypeError):
            return np.inf

        if not np.isfinite(delta) or delta <= 0:
            return np.inf
        return delta

    return _sensitivity


def _weak_value_mzi_sensitivity_fn(
    post_select_angle: float = np.pi / 2 - 0.1,
) -> Callable[[int, float], float]:
    """Create a sensitivity function for weak-value MZI with coherent state input.

    Uses the analytical formula from ``weak_value_mzi_sensitivity``,
    which computes Fisher information as :math:`F = N \\cdot \\cos^2(\\delta)`
    where :math:`\\delta = \\pi/2 - \\text{post\\_select\\_angle}`.

    The sensitivity is :math:`\\Delta\\phi = 1 / \\sqrt{N \\cdot \\cos^2(\\delta)}`,
    which never beats the SQL (``\\Delta\\phi \\ge 1/\\sqrt{N}``).

    Args:
        post_select_angle: Post-selection angle. Values near :math:`\\pi/2`
            give large amplification at the cost of vanishing post-selection
            probability. Default ``pi/2 - 0.1`` (~10x amplification).

    Returns:
        Callable ``(N: int, noise_level: float) -> delta_phi: float``
        suitable for use as ``ModelConfig.custom_sensitivity_fn``.

    """

    def _sensitivity(N: int, noise_level: float) -> float:
        if N < 2:
            return np.inf
        try:
            config = WeakValueConfig(post_select_angle=post_select_angle)
            result = weak_value_mzi_sensitivity(N=N, phi_phase=0.0, config=config)
            delta = result["delta_phi"]
        except (KeyError, ValueError, TypeError):
            return np.inf

        if not np.isfinite(delta) or delta <= 0:
            return np.inf
        return delta

    return _sensitivity


# ── Type coercion helpers for kwarg parsing ──────────────────────────────


def _to_float(val: object, default: float) -> float:
    """Coerce *val* to float, falling back to *default* on type mismatch."""
    if isinstance(val, (int, float)):
        return float(val)
    return default


def _to_int(val: object, default: int) -> int:
    """Coerce *val* to int, falling back to *default* on type mismatch."""
    if isinstance(val, int):
        return val
    return default


def _to_str(val: object, default: str) -> str:
    """Coerce *val* to str, falling back to *default* on type mismatch."""
    if isinstance(val, str):
        return val
    return default


def _to_bool(val: object, default: bool) -> bool:
    """Coerce *val* to bool, falling back to *default* on type mismatch."""
    if isinstance(val, (bool, int)):
        return bool(val)
    return default


# ── Per-model factory functions ──────────────────────────────────────────


def _factory_non_gaussian(
    n_order: int,
    model_id: str,
    kwargs: dict[str, object],
    labels: dict[str, str] | None = None,
) -> ModelConfig:
    """Factory for non-Gaussian state models (n=3, n=4)."""
    _labels = labels if labels is not None else {}
    fn = _non_gaussian_sensitivity_fn(
        n_order=n_order,
        omega_n=_to_float(kwargs.get("omega_n", 0.5), 0.5),
        theta_n=_to_float(kwargs.get("theta_n", 0.0), 0.0),
        t_sqz=_to_float(kwargs.get("t_sqz", 2.0), 2.0),
        use_ground_state=_to_bool(kwargs.get("use_ground_state", False), False),
    )
    return ModelConfig(
        model_id=model_id,
        custom_sensitivity_fn=fn,
        state_type="",
        noise_type="none",
        entangler="none",
        label=_labels.get(model_id, f"Non-Gaussian n={n_order}"),
    )


def _factory_ancilla_assisted(
    model_id: str,
    kwargs: dict[str, object],
    labels: dict[str, str] | None = None,
) -> ModelConfig:
    """Factory for ancilla-assisted metrology model."""
    _labels = labels if labels is not None else {}
    fn = _ancilla_sensitivity_fn(
        alpha=_to_float(kwargs.get("alpha", 1.0), 1.0),
        g_sa=_to_float(kwargs.get("g_sa", 1.0), 1.0),
        tau=_to_float(kwargs.get("tau", 0.1), 0.1),
        g_sp=_to_float(kwargs.get("g_sp", 0.0), 0.0),
        lam=_to_float(kwargs.get("lam", 0.0), 0.0),
        K=_to_int(kwargs.get("K", 2), 2),
    )
    return ModelConfig(
        model_id=model_id,
        custom_sensitivity_fn=fn,
        state_type="",
        noise_type="none",
        entangler="none",
        label=_labels.get(model_id, "Ancilla-assisted metrology"),
    )


def _factory_kerr_mzi(
    model_id: str,
    kwargs: dict[str, object],
    labels: dict[str, str] | None = None,
) -> ModelConfig:
    """Factory for Kerr-nonlinear MZI model."""
    _labels = labels if labels is not None else {}
    fn = _kerr_mzi_sensitivity_fn(
        K=_to_float(kwargs.get("K", 0.1), 0.1),
        T_kerr=_to_float(kwargs.get("T_kerr", 1.0), 1.0),
        state_type=_to_str(kwargs.get("state_type", "noon"), "noon"),
    )
    return ModelConfig(
        model_id=model_id,
        custom_sensitivity_fn=fn,
        state_type="",
        noise_type="none",
        entangler="none",
        label=_labels.get(model_id, "Kerr-nonlinear MZI"),
    )


def _factory_weak_value_mzi(
    model_id: str,
    kwargs: dict[str, object],
    labels: dict[str, str] | None = None,
) -> ModelConfig:
    """Factory for weak-value MZI model."""
    _labels = labels if labels is not None else {}
    fn = _weak_value_mzi_sensitivity_fn(
        post_select_angle=_to_float(
            kwargs.get("post_select_angle", np.pi / 2 - 0.1),
            np.pi / 2 - 0.1,
        ),
    )
    return ModelConfig(
        model_id=model_id,
        custom_sensitivity_fn=fn,
        state_type="",
        noise_type="none",
        entangler="none",
        label=_labels.get(model_id, "Weak-value MZI"),
    )


def _factory_standard(
    model_id: str,
    kwargs: dict[str, object],
    state_types: dict[str, str] | None = None,
    noise_types: dict[str, str] | None = None,
    entanglers: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> ModelConfig:
    """Factory for standard (lookup-dict-based) survey models.

    Unknown model IDs fall through to this factory and produce a model
    with ``state_type`` matching the ID.
    """
    _state_types = state_types if state_types is not None else {}
    _noise_types = noise_types if noise_types is not None else {}
    _entanglers = entanglers if entanglers is not None else {}
    _labels = labels if labels is not None else {}

    state_type = _state_types.get(model_id, model_id)
    noise_type = _noise_types.get(model_id, "none")
    entangler = _entanglers.get(model_id, "none")
    label = _labels.get(model_id, model_id.replace("_", " ").title())

    # Override with any provided string kwargs
    overrides: dict[str, str] = {}
    for key in ("state_type", "noise_type", "entangler", "label"):
        val = kwargs.get(key)
        if isinstance(val, str):
            overrides[key] = val

    return ModelConfig(
        model_id=model_id,
        state_type=overrides.get("state_type", state_type),
        noise_type=overrides.get("noise_type", noise_type),
        entangler=overrides.get("entangler", entangler),
        label=overrides.get("label", label),
    )


# ── Public API ───────────────────────────────────────────────────────────


def create_survey_model(
    model_id: str,
    **kwargs: object,
) -> ModelConfig:
    """Create a ModelConfig from a shorthand identifier.

    Provides convenient shorthand creation for common survey models.
    Supports both named models and custom configurations.

    Built-in model IDs:
        - "ideal_coherent":  Coherent state |alpha>, no noise, no entangler
        - "ideal_noon":      NOON state |N,0>+|0,N>, no noise
        - "ideal_twin_fock": Twin-Fock state, no noise
        - "noon_loss":       NOON state with loss noise
        - "coherent_oat":    Coherent state with OAT spin squeezing
        - "squeezed_vacuum": Squeezed vacuum state, no noise
        - "non_gaussian_n3": Non-Gaussian trisqueezed state (n=3)
        - "non_gaussian_n4": Non-Gaussian quad-squeezed state (n=4)
        - "ancilla_assisted": Ancilla-assisted metrology with dispersive coupling
        - "kerr_mzi":        Kerr-nonlinear MZI (invariant QFI, NOON input)
        - "weak_value_mzi":  Weak-value MZI (SQL-limited, coherent input)

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
    # ── Lookup tables (local to avoid module-level constants) ─────────
    state_types: dict[str, str] = {
        "ideal_coherent": "css",  # CSS = coherent state with |alpha|^2 = N
        "ideal_noon": "noon",
        "ideal_twin_fock": "twin_fock",
        "noon_loss": "noon",
        "coherent_oat": "css",
        "squeezed_vacuum": "squeezed_vacuum",
        "squeezed_vacuum_loss": "squeezed_vacuum",
    }
    noise_types: dict[str, str] = {
        "ideal_coherent": "none",
        "ideal_noon": "none",
        "ideal_twin_fock": "none",
        "noon_loss": "loss",
        "coherent_oat": "none",
        "squeezed_vacuum": "none",
        "squeezed_vacuum_loss": "loss",
    }
    entanglers: dict[str, str] = {
        "ideal_coherent": "none",
        "ideal_noon": "none",
        "ideal_twin_fock": "none",
        "noon_loss": "none",
        "coherent_oat": "oat",
        "squeezed_vacuum": "none",
        "squeezed_vacuum_loss": "none",
    }
    labels: dict[str, str] = {
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
        "kerr_mzi": "Kerr-nonlinear MZI",
        "weak_value_mzi": "Weak-value MZI",
    }

    # ── Local dispatch dict ─────────────────────────────────────────
    factories: dict[str, Callable[[str, dict[str, object]], ModelConfig]] = {
        "non_gaussian_n3": lambda mid, kw: _factory_non_gaussian(
            3, mid, kw, labels=labels
        ),
        "non_gaussian_n4": lambda mid, kw: _factory_non_gaussian(
            4, mid, kw, labels=labels
        ),
        "ancilla_assisted": lambda mid, kw: _factory_ancilla_assisted(
            mid, kw, labels=labels
        ),
        "kerr_mzi": lambda mid, kw: _factory_kerr_mzi(mid, kw, labels=labels),
        "weak_value_mzi": lambda mid, kw: _factory_weak_value_mzi(
            mid, kw, labels=labels
        ),
    }

    factory = factories.get(model_id)
    if factory is not None:
        return factory(model_id, kwargs)

    return _factory_standard(
        model_id,
        kwargs,
        state_types=state_types,
        noise_types=noise_types,
        entanglers=entanglers,
        labels=labels,
    )


def create_default_survey() -> list[ModelConfig]:
    """Create the default set of models for the full scaling survey.

    Returns models representing the most common interferometry states
    used in quantum metrology scaling studies:

    1. ideal_coherent:   Coherent state (SQL scaling, alpha = -1/2)
    2. ideal_noon:       NOON state (Heisenberg scaling, alpha = -1)
    3. ideal_twin_fock:  Twin-Fock state (near-Heisenberg, alpha approx -1)
    4. noon_loss:        NOON with loss (transition from Heisenberg to SQL)
    5. coherent_oat:     Coherent state with OAT squeezing (sub-SQL)
    6. squeezed_vacuum:  Squeezed vacuum state (sub-SQL)
    7. squeezed_vacuum_loss: Squeezed vacuum with one-body loss
    8. non_gaussian_n3:  Non-Gaussian trisqueezed state (n=3)
    9. non_gaussian_n4:  Non-Gaussian quad-squeezed state (n=4)
    10. ancilla_assisted: Ancilla-assisted metrology with dispersive coupling
    11. kerr_mzi:        Kerr-nonlinear MZI (invariant QFI)
    12. weak_value_mzi:  Weak-value MZI (no metrological advantage)

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
        "kerr_mzi",
        "weak_value_mzi",
    ]
    return [create_survey_model(mid) for mid in model_ids]

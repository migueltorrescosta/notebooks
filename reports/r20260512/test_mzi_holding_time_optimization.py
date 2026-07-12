"""Tests for the 20260512 report local module.

Tests verify the single-particle MZI scaling functions and the
ancilla validate_hold_unitarity function that were migrated from
src/ to reports/r20260512/mzi_holding_time_optimization.py.

The module is loaded via importlib because Python identifiers cannot start with digits,
making ``from reports.r20260512.mzi_holding_time_optimization import ...`` a SyntaxError.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# Functions promoted to src/ — imported directly from shared modules.
from src.analysis.ancilla_optimization_scans import validate_hold_unitarity
from src.analysis.scaling_fit import fit_scaling_exponent
from src.physics.mzi_simulation import beam_splitter_unitary
from src.physics.mzi_states import (
    compute_jz_variance,
    two_mode_jz_operator,
)
from src.physics.single_particle_mzi import (
    build_holding_unitary,
    compute_analytical_derivative,
    compute_delta_omega_from_propagation,
    compute_numerical_derivative,
    compute_sensitivity_sweep,
    evolve_single_particle_mzi,
    run_validation,
)

# ── Load mzi_holding_time_optimization.py via importlib ──────────────────────────────────────────────
_report_local = importlib.import_module(
    "reports.r20260512.mzi_holding_time_optimization"
)


def _fock_state(n0: int, n1: int) -> np.ndarray:
    """Test helper: create a Fock basis state in the 4D (max_photons=1) space."""
    s = np.zeros(4, dtype=complex)
    s[n0 * 2 + n1] = 1.0
    return s


# Shared parameter combinations reused across multiple tests
_MZI_PARAMS = [
    (0.5, 0.1),
    (0.5, 1.0),
    (0.5, 10.0),
    (1.0, 0.1),
    (1.0, 1.0),
    (1.0, 10.0),
    (2.0, 0.1),
    (2.0, 1.0),
    (2.0, 10.0),
]

_MZI_PARAMS_WIDE = [
    (0.5, 0.1),
    (0.5, 1.0),
    (0.5, 10.0),
    (0.5, 50.0),
    (1.0, 0.1),
    (1.0, 1.0),
    (1.0, 10.0),
    (1.0, 50.0),
    (2.0, 0.1),
    (2.0, 1.0),
    (2.0, 10.0),
    (2.0, 50.0),
]


def test_jz_eigenvalues_for_physical_states() -> None:
    jz = two_mode_jz_operator(1)
    assert jz.shape == (4, 4)
    assert float(
        np.real(np.conj(_fock_state(1, 0)) @ jz @ _fock_state(1, 0))
    ) == pytest.approx(0.5)
    assert float(
        np.real(np.conj(_fock_state(0, 1)) @ jz @ _fock_state(0, 1))
    ) == pytest.approx(-0.5)
    assert float(
        np.real(np.conj(_fock_state(0, 0)) @ jz @ _fock_state(0, 0))
    ) == pytest.approx(0.0)
    assert float(
        np.real(np.conj(_fock_state(1, 1)) @ jz @ _fock_state(1, 1))
    ) == pytest.approx(0.0)


def test_beam_splitter_is_unitary() -> None:
    u_bs = beam_splitter_unitary(np.pi / 4.0, 0.0, max_photons=1)
    assert u_bs @ u_bs.conj().T == pytest.approx(np.eye(4), abs=1e-12)


def test_given_fock_10_then_bs_produces_balanced_superposition() -> None:
    u_bs = beam_splitter_unitary(np.pi / 4.0, 0.0, max_photons=1)
    psi = u_bs @ _fock_state(1, 0)
    expected = np.zeros(4, dtype=complex)
    expected[2] = 1.0 / np.sqrt(2)
    expected[1] = -1j / np.sqrt(2)
    assert psi == pytest.approx(expected, abs=1e-12)


def test_holding_unitary_is_unitary() -> None:
    jz = two_mode_jz_operator(1)
    u_hold = build_holding_unitary(omega=1.0, t_hold=1.0, jz=jz)
    assert u_hold @ u_hold.conj().T == pytest.approx(np.eye(4), abs=1e-12)


@pytest.mark.parametrize(
    ("omega", "t_hold"),
    _MZI_PARAMS,
    ids=[f"ω={t}, t_hold={h}" for t, h in _MZI_PARAMS],
)
def test_given_mzi_circuit_then_state_remains_normalized(
    omega: float, t_hold: float
) -> None:
    u_bs = beam_splitter_unitary(np.pi / 4.0, 0.0, max_photons=1)
    jz = two_mode_jz_operator(1)
    psi = evolve_single_particle_mzi(omega, t_hold, u_bs, jz)
    assert np.linalg.norm(psi) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("omega", "t_hold"),
    _MZI_PARAMS_WIDE,
    ids=[f"ω={t}, t_hold={h}" for t, h in _MZI_PARAMS_WIDE],
)
def test_given_error_propagation_then_delta_omega_equals_one_over_t_h(
    omega: float, t_hold: float
) -> None:
    if abs(np.sin(omega * t_hold)) < 1e-6:
        pytest.skip("Singular point at fringe extremum")
    u_bs = beam_splitter_unitary(np.pi / 4.0, 0.0, max_photons=1)
    jz = two_mode_jz_operator(1)
    dt_a, *_ = compute_delta_omega_from_propagation(
        t_hold, omega, u_bs, jz, use_numerical=False
    )
    assert dt_a == pytest.approx(1.0 / t_hold, rel=1e-12)


@pytest.mark.parametrize(
    ("omega", "t_hold"),
    _MZI_PARAMS,
    ids=[f"ω={t}, t_hold={h}" for t, h in _MZI_PARAMS],
)
def test_given_mzi_circuit_then_jz_expectation_matches_cos(
    omega: float, t_hold: float
) -> None:
    u_bs = beam_splitter_unitary(np.pi / 4.0, 0.0, max_photons=1)
    jz = two_mode_jz_operator(1)
    psi = evolve_single_particle_mzi(omega, t_hold, u_bs, jz)
    jz_mean = float(np.real(np.conj(psi) @ jz @ psi))
    assert jz_mean == pytest.approx(-0.5 * np.cos(omega * t_hold), abs=1e-12)


@pytest.mark.parametrize(
    ("omega", "t_hold"),
    _MZI_PARAMS,
    ids=[f"ω={t}, t_hold={h}" for t, h in _MZI_PARAMS],
)
def test_given_mzi_circuit_then_jz_variance_matches_sin_squared(
    omega: float, t_hold: float
) -> None:
    u_bs = beam_splitter_unitary(np.pi / 4.0, 0.0, max_photons=1)
    jz = two_mode_jz_operator(1)
    psi = evolve_single_particle_mzi(omega, t_hold, u_bs, jz)
    jz_var = compute_jz_variance(psi, max_photons=1)
    assert jz_var == pytest.approx(0.25 * (np.sin(omega * t_hold) ** 2), abs=1e-12)


@pytest.mark.parametrize(
    ("omega", "t_hold"),
    _MZI_PARAMS,
    ids=[f"ω={t}, t_hold={h}" for t, h in _MZI_PARAMS],
)
def test_given_analytical_derivative_then_matches_expected_form(
    omega: float, t_hold: float
) -> None:
    d_jz = compute_analytical_derivative(t_hold, omega)
    assert d_jz == pytest.approx(0.5 * t_hold * np.sin(omega * t_hold), abs=1e-12)


@pytest.mark.parametrize(
    ("omega", "t_hold"),
    _MZI_PARAMS_WIDE,
    ids=[f"ω={t}, t_hold={h}" for t, h in _MZI_PARAMS_WIDE],
)
def test_given_numerical_and_analytical_derivatives_then_agree_within_tolerance(
    omega: float, t_hold: float
) -> None:
    u_bs = beam_splitter_unitary(np.pi / 4.0, 0.0, max_photons=1)
    jz = two_mode_jz_operator(1)
    d_a = compute_analytical_derivative(t_hold, omega)
    d_n = compute_numerical_derivative(omega, t_hold, u_bs, jz, fd_step=1e-6)
    if abs(d_a) < 1e-15 and abs(d_n) < 1e-15:
        pytest.skip("Both derivatives zero at fringe extremum")
    denom = max(abs(d_a), 1e-15)
    rel_diff = abs(d_a - d_n) / denom
    assert rel_diff < 1e-6, (
        f"Derivative mismatch at ω={omega}, t_hold={t_hold}: "
        f"analytical={d_a:.10e}, numerical={d_n:.10e}, "
        f"rel_diff={rel_diff:.2e}"
    )


def test_given_non_singular_point_then_all_validation_checks_pass() -> None:
    result = run_validation(omega=1.0, t_hold=1.0)
    assert result["state_normalized"]
    assert result["bs_unitary"]
    assert result["delta_omega_matches_theory"]
    assert result["derivative_match"]


def test_given_fringe_extremum_then_point_is_flagged() -> None:
    """At sin(ω t_hold) = 0, the point should be flagged as fringe extremum.

    The analytical limit Δω → 1/t_hold holds via continuity (L'Hôpital's rule),
    but direct numerical evaluation hits 0/0, so delta_omega_matches_theory
    is expected to be False at exactly the singular point.
    """
    t_h = np.pi
    u_bs = beam_splitter_unitary(np.pi / 4.0, 0.0, max_photons=1)
    jz = two_mode_jz_operator(1)
    dt_a, _, _, _, is_fringe = compute_delta_omega_from_propagation(
        t_h,
        1.0,
        u_bs,
        jz,
        use_numerical=False,
    )
    assert is_fringe
    assert not np.isfinite(dt_a) or dt_a > 1e6


def test_given_sensitivity_sweep_then_dataframe_has_expected_columns() -> None:
    df = compute_sensitivity_sweep(omega=1.0, n_points=10)
    expected_cols = [
        "t_hold",
        "omega",
        "jz_mean",
        "jz_var",
        "d_jz_analytical",
        "d_jz_numerical",
        "delta_omega_analytical",
        "delta_omega_numerical",
        "delta_omega_theory",
        "is_fringe_extremum",
        "abs_sin",
    ]
    for col in expected_cols:
        assert col in df.columns
    assert len(df) == 10
    assert np.all(np.diff(df["t_hold"].to_numpy()) > 0)


def test_given_sensitivity_sweep_then_non_fringe_points_match_theory() -> None:
    df = compute_sensitivity_sweep(omega=1.0, n_points=20)
    non_fringe = df[~df["is_fringe_extremum"]]
    for _, row in non_fringe.iterrows():
        assert row["delta_omega_analytical"] == pytest.approx(
            row["delta_omega_theory"],
            rel=1e-12,
        )


def test_given_sensitivity_sweep_then_fringe_points_detected() -> None:
    df = compute_sensitivity_sweep(omega=1.0, n_points=50)
    fringe_points = df[df["is_fringe_extremum"]]
    for _, row in fringe_points.iterrows():
        assert abs(np.sin(row["t_hold"])) < 1e-6


def test_scaling_exponent_from_log_log_fit_is_minus_one() -> None:
    df = compute_sensitivity_sweep(omega=1.0, n_points=50)
    clean = df[~df["is_fringe_extremum"]]
    result = fit_scaling_exponent(
        np.asarray(clean["t_hold"]), np.asarray(clean["delta_omega_analytical"])
    )
    assert result.valid
    assert -1.005 <= result.alpha <= -0.995
    assert result.R_squared > 0.999


@pytest.mark.parametrize(
    "omega", [0.5, 1.0, 2.0, 3.0], ids=["ω=0.5", "ω=1.0", "ω=2.0", "ω=3.0"]
)
def test_scaling_exponent_is_minus_one_for_various_omega(omega: float) -> None:
    df = compute_sensitivity_sweep(omega=omega, n_points=50)
    clean = df[~df["is_fringe_extremum"]]
    result = fit_scaling_exponent(
        np.asarray(clean["t_hold"]), np.asarray(clean["delta_omega_analytical"])
    )
    assert result.valid
    assert -1.005 <= result.alpha <= -0.995
    assert result.R_squared > 0.999


def test_scaling_exponent_using_numerical_derivatives_is_minus_one() -> None:
    df = compute_sensitivity_sweep(omega=1.0, n_points=50)
    clean = df[~df["is_fringe_extremum"]]
    result = fit_scaling_exponent(
        np.asarray(clean["t_hold"]), np.asarray(clean["delta_omega_numerical"])
    )
    assert result.valid
    assert -1.01 <= result.alpha <= -0.99


def test_excluding_fringe_points_improves_fit_quality() -> None:
    df = compute_sensitivity_sweep(omega=1.0, n_points=100)
    # Fit with all points (including fringe extrema)
    result_all = fit_scaling_exponent(
        np.asarray(df["t_hold"]), np.asarray(df["delta_omega_analytical"])
    )
    # Fit excluding fringe points
    clean = df[~df["is_fringe_extremum"]]
    result_excl = fit_scaling_exponent(
        np.asarray(clean["t_hold"]), np.asarray(clean["delta_omega_analytical"])
    )
    assert result_excl.valid
    assert result_excl.R_squared >= result_all.R_squared - 0.001


def test_given_tiny_t_h_then_sensitivity_diverges() -> None:
    u_bs = beam_splitter_unitary(np.pi / 4.0, 0.0, max_photons=1)
    jz = two_mode_jz_operator(1)
    dt_a, *_ = compute_delta_omega_from_propagation(
        1e-10, 1.0, u_bs, jz, use_numerical=False
    )
    assert dt_a > 1e8


def test_given_large_t_h_then_sensitivity_approaches_zero() -> None:
    u_bs = beam_splitter_unitary(np.pi / 4.0, 0.0, max_photons=1)
    jz = two_mode_jz_operator(1)
    dt_a, *_ = compute_delta_omega_from_propagation(
        100.0, 1.0, u_bs, jz, use_numerical=False
    )
    assert dt_a < 0.02


# =============================================================================
# Ancilla report: validate_hold_unitarity
# =============================================================================


def test_validate_hold_unitarity_default() -> None:
    """Default call should pass (returns True)."""
    assert validate_hold_unitarity() is True


def test_validate_hold_unitarity_zero_interaction() -> None:
    """Zero interaction should still be unitary."""
    assert (
        validate_hold_unitarity(t_hold=2.0, omega=0.5, alpha=(0.0, 0.0, 0.0, 0.0))
        is True
    )


def test_validate_hold_unitarity_various_params() -> None:
    """Various parameter combinations should produce unitary matrices."""
    for t_hold in [0.5, 1.0, 3.0]:
        for omega in [0.1, 1.0, 2.0]:
            assert validate_hold_unitarity(t_hold=t_hold, omega=omega) is True


def test_validate_hold_unitarity_negative_alpha() -> None:
    """Negative alpha values should still produce a unitary."""
    alpha = (-0.5, 0.3, -0.2, 0.1)
    assert validate_hold_unitarity(t_hold=1.0, omega=1.0, alpha=alpha) is True

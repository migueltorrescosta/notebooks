"""Tests for the 2026-05-12 report local module.

Tests verify the single-particle MZI scaling functions and the
ancilla validate_hold_unitarity function that were migrated from
src/ to reports/2026-05-12/local.py.

Because the report directory name contains hyphens, we load the
module via importlib to avoid Python import syntax issues.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# ── Load local.py via importlib ──────────────────────────────────────────────
_local_path = Path(__file__).resolve().parent / "local.py"
_spec = importlib.util.spec_from_file_location("report_local", str(_local_path))
assert _spec is not None, f"Could not find local.py at {_local_path}"
_report_local = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_report_local)

# Bind functions to local names for ergonomic access
build_beam_splitter = _report_local.build_beam_splitter
build_holding_unitary = _report_local.build_holding_unitary
compute_analytical_derivative = _report_local.compute_analytical_derivative
compute_delta_theta_from_propagation = (
    _report_local.compute_delta_theta_from_propagation
)
compute_numerical_derivative = _report_local.compute_numerical_derivative
compute_sensitivity_sweep = _report_local.compute_sensitivity_sweep
compute_variance_jz = _report_local.compute_variance_jz
evolve_single_particle_mzi = _report_local.evolve_single_particle_mzi
fit_scaling_exponent = _report_local.fit_scaling_exponent
fock_state = _report_local.fock_state
run_validation = _report_local.run_validation
validate_hold_unitarity = _report_local.validate_hold_unitarity

# Shared utility from src/ (not migrated)
from src.physics.mzi_states import two_mode_jz_operator  # noqa: E402

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
        np.real(np.conj(fock_state(1, 0)) @ jz @ fock_state(1, 0))
    ) == pytest.approx(0.5)
    assert float(
        np.real(np.conj(fock_state(0, 1)) @ jz @ fock_state(0, 1))
    ) == pytest.approx(-0.5)
    assert float(
        np.real(np.conj(fock_state(0, 0)) @ jz @ fock_state(0, 0))
    ) == pytest.approx(0.0)
    assert float(
        np.real(np.conj(fock_state(1, 1)) @ jz @ fock_state(1, 1))
    ) == pytest.approx(0.0)


def test_beam_splitter_is_unitary() -> None:
    u_bs = build_beam_splitter()
    assert u_bs @ u_bs.conj().T == pytest.approx(np.eye(4), abs=1e-12)


def test_given_fock_10_then_bs_produces_balanced_superposition() -> None:
    u_bs = build_beam_splitter()
    psi = u_bs @ fock_state(1, 0)
    expected = np.zeros(4, dtype=complex)
    expected[2] = 1.0 / np.sqrt(2)
    expected[1] = -1j / np.sqrt(2)
    assert psi == pytest.approx(expected, abs=1e-12)


def test_holding_unitary_is_unitary() -> None:
    jz = two_mode_jz_operator(1)
    u_hold = build_holding_unitary(theta=1.0, t_h=1.0, jz=jz)
    assert u_hold @ u_hold.conj().T == pytest.approx(np.eye(4), abs=1e-12)


@pytest.mark.parametrize(
    ("theta", "t_h"),
    _MZI_PARAMS,
    ids=[f"θ={t}, T_H={h}" for t, h in _MZI_PARAMS],
)
def test_given_mzi_circuit_then_state_remains_normalized(
    theta: float, t_h: float
) -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    psi = evolve_single_particle_mzi(theta, t_h, u_bs, jz)
    assert np.linalg.norm(psi) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("theta", "t_h"),
    _MZI_PARAMS_WIDE,
    ids=[f"θ={t}, T_H={h}" for t, h in _MZI_PARAMS_WIDE],
)
def test_given_error_propagation_then_delta_theta_equals_one_over_t_h(
    theta: float, t_h: float
) -> None:
    if abs(np.sin(theta * t_h)) < 1e-6:
        pytest.skip("Singular point at fringe extremum")
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    dt_a, *_ = compute_delta_theta_from_propagation(
        t_h, theta, u_bs, jz, use_numerical=False
    )
    assert dt_a == pytest.approx(1.0 / t_h, rel=1e-12)


@pytest.mark.parametrize(
    ("theta", "t_h"),
    _MZI_PARAMS,
    ids=[f"θ={t}, T_H={h}" for t, h in _MZI_PARAMS],
)
def test_given_mzi_circuit_then_jz_expectation_matches_cos(
    theta: float, t_h: float
) -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    psi = evolve_single_particle_mzi(theta, t_h, u_bs, jz)
    jz_mean = float(np.real(np.conj(psi) @ jz @ psi))
    assert jz_mean == pytest.approx(-0.5 * np.cos(theta * t_h), abs=1e-12)


@pytest.mark.parametrize(
    ("theta", "t_h"),
    _MZI_PARAMS,
    ids=[f"θ={t}, T_H={h}" for t, h in _MZI_PARAMS],
)
def test_given_mzi_circuit_then_jz_variance_matches_sin_squared(
    theta: float, t_h: float
) -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    psi = evolve_single_particle_mzi(theta, t_h, u_bs, jz)
    jz_var = compute_variance_jz(psi, jz)
    assert jz_var == pytest.approx(0.25 * (np.sin(theta * t_h) ** 2), abs=1e-12)


@pytest.mark.parametrize(
    ("theta", "t_h"),
    _MZI_PARAMS,
    ids=[f"θ={t}, T_H={h}" for t, h in _MZI_PARAMS],
)
def test_given_analytical_derivative_then_matches_expected_form(
    theta: float, t_h: float
) -> None:
    d_jz = compute_analytical_derivative(t_h, theta)
    assert d_jz == pytest.approx(0.5 * t_h * np.sin(theta * t_h), abs=1e-12)


@pytest.mark.parametrize(
    ("theta", "t_h"),
    _MZI_PARAMS_WIDE,
    ids=[f"θ={t}, T_H={h}" for t, h in _MZI_PARAMS_WIDE],
)
def test_given_numerical_and_analytical_derivatives_then_agree_within_tolerance(
    theta: float, t_h: float
) -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    d_a = compute_analytical_derivative(t_h, theta)
    d_n = compute_numerical_derivative(theta, t_h, u_bs, jz, delta=1e-6)
    if abs(d_a) < 1e-15 and abs(d_n) < 1e-15:
        pytest.skip("Both derivatives zero at fringe extremum")
    denom = max(abs(d_a), 1e-15)
    rel_diff = abs(d_a - d_n) / denom
    assert rel_diff < 1e-6, (
        f"Derivative mismatch at θ={theta}, T_H={t_h}: "
        f"analytical={d_a:.10e}, numerical={d_n:.10e}, "
        f"rel_diff={rel_diff:.2e}"
    )


def test_given_non_singular_point_then_all_validation_checks_pass() -> None:
    result = run_validation(theta=1.0, t_h=1.0)
    assert result["state_normalized"]
    assert result["bs_unitary"]
    assert result["delta_theta_matches_theory"]
    assert result["derivative_match"]


def test_given_fringe_extremum_then_point_is_flagged() -> None:
    """At sin(θ T_H) = 0, the point should be flagged as fringe extremum.

    The analytical limit Δθ → 1/T_H holds via continuity (L'Hôpital's rule),
    but direct numerical evaluation hits 0/0, so delta_theta_matches_theory
    is expected to be False at exactly the singular point.
    """
    t_h = np.pi
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    dt_a, _, _, _, is_fringe = compute_delta_theta_from_propagation(
        t_h,
        1.0,
        u_bs,
        jz,
        use_numerical=False,
    )
    assert is_fringe
    assert not np.isfinite(dt_a) or dt_a > 1e6


def test_given_sensitivity_sweep_then_dataframe_has_expected_columns() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=10)
    expected_cols = [
        "T_H",
        "theta",
        "jz_mean",
        "jz_var",
        "d_jz_analytical",
        "d_jz_numerical",
        "delta_theta_analytical",
        "delta_theta_numerical",
        "delta_theta_theory",
        "is_fringe_extremum",
        "abs_sin",
    ]
    for col in expected_cols:
        assert col in df.columns
    assert len(df) == 10
    assert np.all(np.diff(df["T_H"].to_numpy()) > 0)


def test_given_sensitivity_sweep_then_non_fringe_points_match_theory() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=20)
    non_fringe = df[~df["is_fringe_extremum"]]
    for _, row in non_fringe.iterrows():
        assert row["delta_theta_analytical"] == pytest.approx(
            row["delta_theta_theory"],
            rel=1e-12,
        )


def test_given_sensitivity_sweep_then_fringe_points_detected() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=50)
    fringe_points = df[df["is_fringe_extremum"]]
    for _, row in fringe_points.iterrows():
        assert abs(np.sin(row["T_H"])) < 1e-6


def test_scaling_exponent_from_log_log_fit_is_minus_one() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=50)
    alpha, r_sq, _fit_df = fit_scaling_exponent(df)
    assert np.isfinite(alpha)
    assert -1.005 <= alpha <= -0.995
    assert r_sq > 0.999


@pytest.mark.parametrize(
    "theta", [0.5, 1.0, 2.0, 3.0], ids=["θ=0.5", "θ=1.0", "θ=2.0", "θ=3.0"]
)
def test_scaling_exponent_is_minus_one_for_various_theta(theta: float) -> None:
    df = compute_sensitivity_sweep(theta=theta, n_points=50)
    alpha, r_sq, _ = fit_scaling_exponent(df)
    assert np.isfinite(alpha)
    assert -1.005 <= alpha <= -0.995
    assert r_sq > 0.999


def test_scaling_exponent_using_numerical_derivatives_is_minus_one() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=50)
    alpha, _r_sq, _ = fit_scaling_exponent(df, column="delta_theta_numerical")
    assert np.isfinite(alpha)
    assert -1.01 <= alpha <= -0.99


def test_excluding_fringe_points_improves_fit_quality() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=100)
    _alpha_all, r_sq_all, _ = fit_scaling_exponent(df, exclude_fringe=False)
    alpha_excl, r_sq_excl, _ = fit_scaling_exponent(df, exclude_fringe=True)
    assert np.isfinite(alpha_excl)
    assert r_sq_excl >= r_sq_all - 0.001


def test_given_tiny_t_h_then_sensitivity_diverges() -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    dt_a, *_ = compute_delta_theta_from_propagation(
        1e-10, 1.0, u_bs, jz, use_numerical=False
    )
    assert dt_a > 1e8


def test_given_large_t_h_then_sensitivity_approaches_zero() -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    dt_a, *_ = compute_delta_theta_from_propagation(
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
        validate_hold_unitarity(T_H=2.0, theta=0.5, alpha=(0.0, 0.0, 0.0, 0.0)) is True
    )


def test_validate_hold_unitarity_various_params() -> None:
    """Various parameter combinations should produce unitary matrices."""
    for T_H in [0.5, 1.0, 3.0]:
        for theta in [0.1, 1.0, 2.0]:
            assert validate_hold_unitarity(T_H=T_H, theta=theta) is True


def test_validate_hold_unitarity_negative_alpha() -> None:
    """Negative alpha values should still produce a unitary."""
    alpha = (-0.5, 0.3, -0.2, 0.1)
    assert validate_hold_unitarity(T_H=1.0, theta=1.0, alpha=alpha) is True

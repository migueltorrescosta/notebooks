"""
Tests for the single-particle MZI holding-time scaling module.

Tests verify:
1. Operator construction correctness (J_z, BS, unitarity)
2. State normalization through the MZI circuit
3. Analytical Δθ = 1/T_H formula
4. Numerical derivative matches analytical
5. Full sensitivity sweep produces expected scaling
"""

import numpy as np
import pytest

from src.physics.mzi_states import two_mode_jz_operator
from src.physics.single_particle_mzi_scaling import (
    build_beam_splitter,
    build_holding_unitary,
    compute_analytical_derivative,
    compute_delta_theta_from_propagation,
    compute_numerical_derivative,
    compute_sensitivity_sweep,
    compute_variance_jz,
    evolve_single_particle_mzi,
    fit_scaling_exponent,
    fock_state,
    run_validation,
)

# =============================================================================
# Operator Construction
# =============================================================================


def test_j_z_must_be_diagonal_with_eigenvalues_1_2_for_physical_states() -> None:
    jz = two_mode_jz_operator(1)
    assert jz.shape == (4, 4), "Expected jz.shape == (4, 4)"
    # |1,0⟩ eigenvalue = +1/2
    state_10 = fock_state(1, 0)
    assert float(np.real(np.conj(state_10) @ jz @ state_10)) == pytest.approx(0.5), (
        "Expected ⟨J_z⟩ = 0.5 for |1,0⟩"
    )
    # |0,1⟩ eigenvalue = -1/2
    state_01 = fock_state(0, 1)
    assert float(np.real(np.conj(state_01) @ jz @ state_01)) == pytest.approx(-0.5), (
        "Expected ⟨J_z⟩ = -0.5 for |0,1⟩"
    )
    # |0,0⟩ eigenvalue = 0
    state_00 = fock_state(0, 0)
    assert float(np.real(np.conj(state_00) @ jz @ state_00)) == pytest.approx(0.0), (
        "Expected ⟨J_z⟩ = 0.0 for |0,0⟩"
    )
    # |1,1⟩ eigenvalue = 0
    state_11 = fock_state(1, 1)
    assert float(np.real(np.conj(state_11) @ jz @ state_11)) == pytest.approx(0.0), (
        "Expected ⟨J_z⟩ = 0.0 for |1,1⟩"
    )


def test_beam_splitter_must_be_unitary_u_bs_u_bs_i() -> None:
    u_bs = build_beam_splitter()
    result = u_bs @ u_bs.conj().T
    assert result == pytest.approx(np.eye(4), abs=1e-12), (
        "Expected result == pytest.approx(np.eye(4), abs=1e-12)"
    )


def test_bs_on_1_0_must_produce_balanced_superposition() -> None:
    u_bs = build_beam_splitter()
    state_in = fock_state(1, 0)
    psi = u_bs @ state_in
    # After 50:50 BS: (|1,0⟩ - i|0,1⟩)/√2
    expected = np.zeros(4, dtype=complex)
    expected[2] = 1.0 / np.sqrt(2)  # |1,0⟩ index = 1*2+0 = 2
    expected[1] = -1j / np.sqrt(2)  # |0,1⟩ index = 0*2+1 = 1
    assert psi == pytest.approx(expected, abs=1e-12), (
        "Expected psi == pytest.approx(expected, abs=1e-12)"
    )


def test_u_hold_must_be_unitary() -> None:
    jz = two_mode_jz_operator(1)
    u_hold = build_holding_unitary(theta=1.0, t_h=1.0, jz=jz)
    assert u_hold @ u_hold.conj().T == pytest.approx(np.eye(4), abs=1e-12), (
        "Expected u_hold @ u_hold.conj().T == pytest.approx(np.eye(4), abs=1e-12)"
    )


# =============================================================================
# State Evolution
# =============================================================================


def test_full_mzi_circuit_must_preserve_state_norm() -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    for theta in [0.5, 1.0, 2.0]:
        for t_h in [0.1, 1.0, 10.0]:
            psi = evolve_single_particle_mzi(theta, t_h, u_bs, jz)
            assert np.linalg.norm(psi) == pytest.approx(1.0), (
                f"Norm violation: θ={theta}, T_H={t_h}"
            )


# =============================================================================
# Analytical Results
# =============================================================================


def test_from_error_propagation_must_equal_exactly_1_t_h() -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    for theta in [0.5, 1.0, 2.0]:
        for t_h in [0.1, 1.0, 10.0, 50.0]:
            dt_a, _jz_mean, _jz_var, _d_jz, _ = compute_delta_theta_from_propagation(
                t_h,
                theta,
                u_bs,
                jz,
                use_numerical=False,
            )
            if abs(np.sin(theta * t_h)) < 1e-6:
                continue  # skip fringe extrema
            assert dt_a == pytest.approx(1.0 / t_h, rel=1e-12), (
                f"Analytical Δθ = {dt_a:.6e}, expected {1.0 / t_h:.6e}, "
                f"θ={theta}, T_H={t_h}"
            )


def test_j_z_must_match_cos_t_h() -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    for theta in [0.5, 1.0, 2.0]:
        for t_h in [0.1, 1.0, 10.0]:
            psi = evolve_single_particle_mzi(theta, t_h, u_bs, jz)
            jz_mean = float(np.real(np.conj(psi) @ jz @ psi))
            expected = -0.5 * np.cos(theta * t_h)
            assert jz_mean == pytest.approx(expected, abs=1e-12), (
                f"⟨J_z⟩ = {jz_mean:.6e}, expected {expected:.6e}"
            )


def test_var_j_z_must_match_sin_t_h() -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    for theta in [0.5, 1.0, 2.0]:
        for t_h in [0.1, 1.0, 10.0]:
            psi = evolve_single_particle_mzi(theta, t_h, u_bs, jz)
            jz_var = compute_variance_jz(psi, jz)
            expected = 0.25 * (np.sin(theta * t_h) ** 2)
            assert jz_var == pytest.approx(expected, abs=1e-12), (
                f"Var(J_z) = {jz_var:.6e}, expected {expected:.6e}"
            )


def test_j_z_must_match_t_h_2_sin_t_h() -> None:
    for theta in [0.5, 1.0, 2.0]:
        for t_h in [0.1, 1.0, 10.0]:
            d_jz = compute_analytical_derivative(t_h, theta)
            expected = 0.5 * t_h * np.sin(theta * t_h)
            assert d_jz == pytest.approx(expected, abs=1e-12), (
                f"∂⟨J_z⟩/∂θ = {d_jz:.6e}, expected {expected:.6e}"
            )


# =============================================================================
# Numerical Derivative
# =============================================================================


def test_numerical_and_analytical_derivatives_must_agree_to_1e_6_relative() -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    for theta in [0.5, 1.0, 2.0]:
        for t_h in [0.1, 1.0, 10.0, 50.0]:
            d_a = compute_analytical_derivative(t_h, theta)
            d_n = compute_numerical_derivative(theta, t_h, u_bs, jz, delta=1e-6)
            if abs(d_a) < 1e-15 and abs(d_n) < 1e-15:
                continue  # both zero at fringe extrema
            denom = max(abs(d_a), 1e-15)
            rel_diff = abs(d_a - d_n) / denom
            assert rel_diff < 1e-6, (
                f"Derivative mismatch at θ={theta}, T_H={t_h}: "
                f"analytical={d_a:.10e}, numerical={d_n:.10e}, "
                f"rel_diff={rel_diff:.2e}"
            )


# =============================================================================
# Validation
# =============================================================================


def test_all_validation_checks_must_pass_at_a_non_singular_point() -> None:
    result = run_validation(theta=1.0, t_h=1.0)
    assert result["state_normalized"], 'Condition failed: result["state_normalized"]'
    assert result["bs_unitary"], 'Condition failed: result["bs_unitary"]'
    assert result["delta_theta_matches_theory"], (
        'Condition failed: result["delta_theta_matches_theory"]'
    )
    assert result["derivative_match"], 'Condition failed: result["derivative_match"]'


def test_validation_at_fringe_extremum() -> None:
    """At sin(θ T_H) = 0, the point should be flagged as fringe extremum.

    The analytical limit Δθ → 1/T_H holds via continuity (L'Hôpital's rule),
    but direct numerical evaluation hits 0/0, so delta_theta_matches_theory
    is expected to be False at exactly the singular point.
    """
    t_h = np.pi  # θ=1, T_H=π → sin(π) ≈ 0
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)

    # Compute delta_theta; it will be inf/NaN at exact fringe extremum
    dt_a, _, _, _, is_fringe = compute_delta_theta_from_propagation(
        t_h,
        1.0,
        u_bs,
        jz,
        use_numerical=False,
    )
    assert is_fringe, "Should detect fringe extremum"
    # At exact fringe extremum, 0/0 leads to inf (singular)
    assert not np.isfinite(dt_a) or dt_a > 1e6, (
        "Delta theta diverges at fringe extremum"
    )


# =============================================================================
# Sensitivity Sweep
# =============================================================================


def test_sensitivity_sweep_must_produce_a_dataframe_with_expected_columns() -> None:
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
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) == 10, "Expected len(df) == 10"
    # T_H should be increasing (log-spaced)
    assert np.all(np.diff(df["T_H"].to_numpy()) > 0), (
        'Expected np.all(np.diff(df["T_H"].to_numpy()) > 0)'
    )


def test_from_sweep_must_match_1_t_h_for_all_non_fringe_points() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=20)
    non_fringe = df[~df["is_fringe_extremum"]]
    for _, row in non_fringe.iterrows():
        assert row["delta_theta_analytical"] == pytest.approx(
            row["delta_theta_theory"],
            rel=1e-12,
        ), (
            'Expected row["delta_theta_analytical"] == pytest.approx(row["delta_theta_theory"], rel=1e-12)'
        )


def test_fringe_extremum_detection_should_flag_points_near_sin_t_h_0() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=50)
    # Points near θ T_H = nπ should be flagged
    # θ=1, so T_H = nπ ≈ {3.14, 6.28, ...}
    fringe_points = df[df["is_fringe_extremum"]]
    if len(fringe_points) > 0:
        for _, row in fringe_points.iterrows():
            assert abs(np.sin(row["T_H"])) < 1e-6, (
                f"False fringe detection at T_H={row['T_H']:.4f}, "
                f"sin={np.sin(row['T_H']):.2e}"
            )


# =============================================================================
# Scaling Exponent Fit
# =============================================================================


def test_scaling_exponent_must_be_1_from_log_log_fit() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=50)
    alpha, r_sq, _fit_df = fit_scaling_exponent(df)
    assert np.isfinite(alpha), "Alpha must be finite"
    assert -1.005 <= alpha <= -0.995, (
        f"Scaling exponent α = {alpha:.6f}, expected -1.000 ± 0.005"
    )
    assert r_sq > 0.999, f"R² = {r_sq:.6f}, expected > 0.999"


def test_scaling_exponent_must_be_1_for_different_true_values() -> None:
    for theta in [0.5, 1.0, 2.0, 3.0]:
        df = compute_sensitivity_sweep(theta=theta, n_points=50)
        alpha, r_sq, _ = fit_scaling_exponent(df)
        assert np.isfinite(alpha), f"Non-finite α={alpha} at θ={theta}"
        assert -1.005 <= alpha <= -0.995, (
            f"α = {alpha:.6f} at θ={theta}, expected -1.000 ± 0.005"
        )
        assert r_sq > 0.999, f"R² = {r_sq:.6f} at θ={theta}"


def test_scaling_exponent_using_numerical_derivatives_must_also_be_1() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=50)
    alpha, _r_sq, _ = fit_scaling_exponent(df, column="delta_theta_numerical")
    assert np.isfinite(alpha), "Alpha must be finite (numerical)"
    assert -1.01 <= alpha <= -0.99, (
        f"Numerical scaling exponent α = {alpha:.6f}, expected -1.000 ± 0.010"
    )


def test_excluding_fringe_points_should_produce_a_cleaner_fit() -> None:
    df = compute_sensitivity_sweep(theta=1.0, n_points=100)
    # Fit without excluding fringe points — some may be NaN/inf
    _alpha_all, r_sq_all, _ = fit_scaling_exponent(df, exclude_fringe=False)
    alpha_excl, r_sq_excl, _ = fit_scaling_exponent(df, exclude_fringe=True)
    # Both should still be reasonable given analytical formula
    assert np.isfinite(alpha_excl), "Expected alpha_excl to be finite"
    assert r_sq_excl >= r_sq_all - 0.001  # exclusion should not hurt


# =============================================================================
# Edge Cases
# =============================================================================


def test_at_t_h_0_sensitivity_diverges() -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    dt_a, _, _, _, _ = compute_delta_theta_from_propagation(
        1e-10,
        1.0,
        u_bs,
        jz,
        use_numerical=False,
    )
    assert dt_a > 1e8, f"Δθ should be large for tiny T_H, got {dt_a:.2e}"


def test_at_large_t_h_sensitivity_should_approach_zero() -> None:
    u_bs = build_beam_splitter()
    jz = two_mode_jz_operator(1)
    dt_a, _, _, _, _ = compute_delta_theta_from_propagation(
        100.0,
        1.0,
        u_bs,
        jz,
        use_numerical=False,
    )
    assert dt_a < 0.02, f"Δθ should be small for large T_H, got {dt_a:.4f}"

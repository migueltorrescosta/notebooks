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


def test_jz_operator_diagonal() -> None:
    """J_z must be diagonal with eigenvalues ±1/2 for physical states."""
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


def test_beam_splitter_unitarity() -> None:
    """Beam splitter must be unitary: U_BS U_BS^† = I."""
    u_bs = build_beam_splitter()
    result = u_bs @ u_bs.conj().T
    assert result == pytest.approx(np.eye(4), abs=1e-12), (
        "Expected result == pytest.approx(np.eye(4), abs=1e-12)"
    )


def test_beam_splitter_acts_on_subspace() -> None:
    """BS on |1,0⟩ must produce balanced superposition."""
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


def test_holding_unitary_unitarity() -> None:
    """U_hold must be unitary."""
    jz = two_mode_jz_operator(1)
    u_hold = build_holding_unitary(theta=1.0, t_h=1.0, jz=jz)
    assert u_hold @ u_hold.conj().T == pytest.approx(np.eye(4), abs=1e-12), (
        "Expected u_hold @ u_hold.conj().T == pytest.approx(np.eye(4), abs=1e-12)"
    )


# =============================================================================
# State Evolution
# =============================================================================


def test_evolution_preserves_norm() -> None:
    """Full MZI circuit must preserve state norm."""
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


def test_analytical_formula() -> None:
    """Δθ from error propagation must equal exactly 1/T_H."""
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


def test_jz_expectation_analytical() -> None:
    """⟨J_z⟩ must match -½ cos(θ T_H)."""
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


def test_jz_variance_analytical() -> None:
    """Var(J_z) must match ¼ sin²(θ T_H)."""
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


def test_analytical_derivative_formula() -> None:
    """∂⟨J_z⟩/∂θ must match (T_H/2) sin(θ T_H)."""
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


def test_numerical_derivative_matches_analytical() -> None:
    """Numerical and analytical derivatives must agree to 1e-6 relative."""
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


def test_validation_passes() -> None:
    """All validation checks must pass at a non-singular point."""
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


def test_sweep_produces_dataframe() -> None:
    """Sensitivity sweep must produce a DataFrame with expected columns."""
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


def test_sweep_delta_theta_matches_theory() -> None:
    """Δθ from sweep must match 1/T_H for all non-fringe points."""
    df = compute_sensitivity_sweep(theta=1.0, n_points=20)
    non_fringe = df[~df["is_fringe_extremum"]]
    for _, row in non_fringe.iterrows():
        assert row["delta_theta_analytical"] == pytest.approx(
            row["delta_theta_theory"],
            rel=1e-12,
        ), (
            'Expected row["delta_theta_analytical"] == pytest.approx(row["delta_theta_theory"], rel=1e-12)'
        )


def test_fringe_detection() -> None:
    """Fringe extremum detection should flag points near sin(θ T_H) ≈ 0."""
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


def test_scaling_exponent_is_minus_one() -> None:
    """Scaling exponent must be α = -1 from log-log fit."""
    df = compute_sensitivity_sweep(theta=1.0, n_points=50)
    alpha, r_sq, _fit_df = fit_scaling_exponent(df)
    assert np.isfinite(alpha), "Alpha must be finite"
    assert -1.005 <= alpha <= -0.995, (
        f"Scaling exponent α = {alpha:.6f}, expected -1.000 ± 0.005"
    )
    assert r_sq > 0.999, f"R² = {r_sq:.6f}, expected > 0.999"


def test_scaling_exponent_independent_of_theta() -> None:
    """Scaling exponent must be α ≈ -1 for different true θ values."""
    for theta in [0.5, 1.0, 2.0, 3.0]:
        df = compute_sensitivity_sweep(theta=theta, n_points=50)
        alpha, r_sq, _ = fit_scaling_exponent(df)
        assert np.isfinite(alpha), f"Non-finite α={alpha} at θ={theta}"
        assert -1.005 <= alpha <= -0.995, (
            f"α = {alpha:.6f} at θ={theta}, expected -1.000 ± 0.005"
        )
        assert r_sq > 0.999, f"R² = {r_sq:.6f} at θ={theta}"


def test_scaling_exponent_numerical_derivative() -> None:
    """Scaling exponent using numerical derivatives must also be α ≈ -1."""
    df = compute_sensitivity_sweep(theta=1.0, n_points=50)
    alpha, _r_sq, _ = fit_scaling_exponent(df, column="delta_theta_numerical")
    assert np.isfinite(alpha), "Alpha must be finite (numerical)"
    assert -1.01 <= alpha <= -0.99, (
        f"Numerical scaling exponent α = {alpha:.6f}, expected -1.000 ± 0.010"
    )


def test_fringe_exclusion_improves_fit() -> None:
    """Excluding fringe points should produce a cleaner fit."""
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


def test_zero_holding_time_limit() -> None:
    """At T_H → 0, sensitivity diverges (Δθ → ∞)."""
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


def test_large_holding_time() -> None:
    """At large T_H, sensitivity should approach zero."""
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

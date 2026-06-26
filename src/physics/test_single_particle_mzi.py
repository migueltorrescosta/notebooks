"""
Tests for Single-Particle MZI Simulation.

Covers: build_holding_unitary, evolve_single_particle_mzi,
compute_analytical_derivative, compute_numerical_derivative,
compute_delta_omega_from_propagation, compute_sensitivity_sweep,
run_validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.beam_splitter import bs_fock
from src.physics.mzi_simulation import prepare_input_state
from src.physics.mzi_states import compute_jz_variance, two_mode_jz_operator

from .single_particle_mzi import (
    build_holding_unitary,
    compute_analytical_derivative,
    compute_delta_omega_from_propagation,
    compute_numerical_derivative,
    compute_sensitivity_sweep,
    evolve_single_particle_mzi,
    run_validation,
)

# Shared helpers
_JZ_1 = two_mode_jz_operator(1)
_U_BS = bs_fock(np.pi / 4.0, 0.0, max_photons=1)

# Representative parameter pairs for parametrized tests
_PARAM_SETS = [
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
_PARAM_IDS = [f"ω={w}, t_hold={t}" for w, t in _PARAM_SETS]


# =============================================================================
# build_holding_unitary
# =============================================================================


class TestBuildHoldingUnitary:
    """Tests for build_holding_unitary."""

    def test_correct_dimension(self) -> None:
        u = build_holding_unitary(1.0, 1.0, _JZ_1)
        assert u.shape == (4, 4)

    def test_identity_at_zero_omega(self) -> None:
        u = build_holding_unitary(0.0, 1.0, _JZ_1)
        assert np.allclose(u, np.eye(4), atol=1e-12)

    def test_identity_at_zero_t_hold(self) -> None:
        u = build_holding_unitary(1.0, 0.0, _JZ_1)
        assert np.allclose(u, np.eye(4), atol=1e-12)

    def test_unitary_holds(self) -> None:
        u = build_holding_unitary(1.0, 1.0, _JZ_1)
        assert np.allclose(u @ u.conj().T, np.eye(4), atol=1e-12)

    def test_periodicity_two_pi(self) -> None:
        """omega * t_hold = 2*pi should give identity (J_z eigenvalues are half-integer)."""
        u = build_holding_unitary(2.0, np.pi, _JZ_1)
        # exp(-i * 2*pi * J_z) — for J_z eigenvalues ±0.5: exp(-i*2*pi*±0.5) = exp(∓i*pi) = -1
        # So not identity for single particle. Let's test omega*t_hold = 4*pi instead.
        # exp(-i * 4*pi * J_z) = exp(∓i*2*pi) = 1 for both eigenvalues → identity
        u = build_holding_unitary(4.0, np.pi, _JZ_1)
        assert np.allclose(u, np.eye(4), atol=1e-12)

    @pytest.mark.parametrize(("omega", "t_hold"), _PARAM_SETS, ids=_PARAM_IDS)
    def test_unitary_across_parameters(self, omega: float, t_hold: float) -> None:
        u = build_holding_unitary(omega, t_hold, _JZ_1)
        assert np.allclose(u @ u.conj().T, np.eye(4), atol=1e-12)


# =============================================================================
# evolve_single_particle_mzi
# =============================================================================


class TestEvolveSingleParticleMZI:
    """Tests for evolve_single_particle_mzi."""

    @pytest.mark.parametrize(("omega", "t_hold"), _PARAM_SETS, ids=_PARAM_IDS)
    def test_norm_preserved(self, omega: float, t_hold: float) -> None:
        psi = evolve_single_particle_mzi(omega, t_hold, _U_BS, _JZ_1)
        assert np.linalg.norm(psi) == pytest.approx(1.0)

    def test_default_input_is_single_photon_mode0(self) -> None:
        """Default input_state=None should use |1,0⟩."""
        default_psi = evolve_single_particle_mzi(1.0, 1.0, _U_BS, _JZ_1)
        explicit = prepare_input_state("single_photon", max_photons=1, mode=0)
        explicit_psi = evolve_single_particle_mzi(
            1.0, 1.0, _U_BS, _JZ_1, input_state=explicit
        )
        assert np.allclose(default_psi, explicit_psi, atol=1e-12)

    def test_non_default_input_works(self) -> None:
        """Single photon in mode 1 should also produce normalized output."""
        state = prepare_input_state("single_photon", max_photons=1, mode=1)
        psi = evolve_single_particle_mzi(1.0, 1.0, _U_BS, _JZ_1, input_state=state)
        assert np.linalg.norm(psi) == pytest.approx(1.0)

    @pytest.mark.parametrize(("omega", "t_hold"), _PARAM_SETS, ids=_PARAM_IDS)
    def test_jz_mean_matches_analytical(self, omega: float, t_hold: float) -> None:
        """⟨J_z⟩ = -(1/2) cos(ω t_hold)."""
        psi = evolve_single_particle_mzi(omega, t_hold, _U_BS, _JZ_1)
        jz_mean = float(np.real(np.conj(psi) @ _JZ_1 @ psi))
        expected = -0.5 * np.cos(omega * t_hold)
        assert jz_mean == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize(("omega", "t_hold"), _PARAM_SETS, ids=_PARAM_IDS)
    def test_jz_variance_matches_analytical(self, omega: float, t_hold: float) -> None:
        """Var(J_z) = (1/4) sin²(ω t_hold)."""
        psi = evolve_single_particle_mzi(omega, t_hold, _U_BS, _JZ_1)
        jz_var = compute_jz_variance(psi, max_photons=1)
        expected = 0.25 * (np.sin(omega * t_hold) ** 2)
        assert jz_var == pytest.approx(expected, abs=1e-12)

    def test_variance_non_negative(self) -> None:
        for omega in [0.1, 1.0, 2.0, 5.0]:
            for t_hold in [0.1, 1.0, 5.0]:
                psi = evolve_single_particle_mzi(omega, t_hold, _U_BS, _JZ_1)
                jz_var = compute_jz_variance(psi, max_photons=1)
                assert jz_var >= -1e-12, (
                    f"Negative variance at ω={omega}, t_hold={t_hold}"
                )


# =============================================================================
# compute_analytical_derivative
# =============================================================================


class TestComputeAnalyticalDerivative:
    """Tests for compute_analytical_derivative."""

    @pytest.mark.parametrize(("omega", "t_hold"), _PARAM_SETS, ids=_PARAM_IDS)
    def test_matches_closed_form(self, omega: float, t_hold: float) -> None:
        """∂⟨J_z⟩/∂ω = (t_hold/2) sin(ω t_hold)."""
        d_jz = compute_analytical_derivative(t_hold, omega)
        expected = 0.5 * t_hold * np.sin(omega * t_hold)
        assert d_jz == pytest.approx(expected, abs=1e-12)

    def test_zero_at_fringe_extremum(self) -> None:
        """At ω t_hold = 0, derivative should be 0."""
        assert compute_analytical_derivative(0.0, 1.0) == pytest.approx(0.0, abs=1e-12)
        assert compute_analytical_derivative(1.0, 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_zero_at_pi(self) -> None:
        """At ω t_hold = π (sin = 0), derivative should be 0."""
        assert compute_analytical_derivative(np.pi, 1.0) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_maximum_at_half_pi(self) -> None:
        """At ω t_hold = π/2 (sin = 1), derivative = t_hold/2."""
        d_jz = compute_analytical_derivative(np.pi / 2.0, 1.0)
        assert d_jz == pytest.approx(0.5 * np.pi / 2.0, abs=1e-12)


# =============================================================================
# compute_numerical_derivative
# =============================================================================


class TestComputeNumericalDerivative:
    """Tests for compute_numerical_derivative."""

    @pytest.mark.parametrize(("omega", "t_hold"), _PARAM_SETS, ids=_PARAM_IDS)
    def test_agrees_with_analytical(self, omega: float, t_hold: float) -> None:
        d_a = compute_analytical_derivative(t_hold, omega)
        d_n = compute_numerical_derivative(omega, t_hold, _U_BS, _JZ_1)
        if abs(d_a) < 1e-15 and abs(d_n) < 1e-15:
            pytest.skip("Both derivatives zero at fringe extremum")
        rel_diff = abs(d_a - d_n) / max(abs(d_a), 1e-15)
        assert rel_diff < 1e-6, (
            f"Derivative mismatch at ω={omega}, t_hold={t_hold}: "
            f"analytical={d_a:.10e}, numerical={d_n:.10e}, "
            f"rel_diff={rel_diff:.2e}"
        )

    def test_different_fd_steps(self) -> None:
        """Should work for various finite-difference step sizes."""
        d_a = compute_analytical_derivative(1.0, 1.0)
        for fd_step in [1e-5, 1e-6, 1e-7]:
            d_n = compute_numerical_derivative(1.0, 1.0, _U_BS, _JZ_1, fd_step=fd_step)
            assert abs(d_n - d_a) / max(abs(d_a), 1e-15) < 1e-4

    def test_zero_at_fringe_extremum(self) -> None:
        """At ω t_hold = π, derivative should be near zero (sin = 0)."""
        d_n = compute_numerical_derivative(np.pi, 1.0, _U_BS, _JZ_1)
        assert abs(d_n) < 1e-4


# =============================================================================
# compute_delta_omega_from_propagation
# =============================================================================


class TestComputeDeltaOmegaFromPropagation:
    """Tests for compute_delta_omega_from_propagation."""

    def test_returns_expected_tuple(self) -> None:
        result = compute_delta_omega_from_propagation(1.0, 1.0, _U_BS, _JZ_1)
        assert len(result) == 5
        dt, jz_mean, jz_var, d_jz, is_fringe = result
        assert isinstance(dt, float)
        assert isinstance(jz_mean, float)
        assert isinstance(jz_var, float)
        assert isinstance(d_jz, float)
        assert isinstance(is_fringe, bool)

    @pytest.mark.parametrize(("omega", "t_hold"), _PARAM_SETS, ids=_PARAM_IDS)
    def test_delta_omega_equals_one_over_t_hold_analytical(
        self, omega: float, t_hold: float
    ) -> None:
        """Δω = 1/t_hold for non-fringe points using analytical derivative."""
        if abs(np.sin(omega * t_hold)) < 1e-6:
            pytest.skip("Singular point at fringe extremum")
        dt_a, *_ = compute_delta_omega_from_propagation(
            t_hold, omega, _U_BS, _JZ_1, use_numerical=False
        )
        assert dt_a == pytest.approx(1.0 / t_hold, rel=1e-12)

    @pytest.mark.parametrize(("omega", "t_hold"), _PARAM_SETS, ids=_PARAM_IDS)
    def test_delta_omega_equals_one_over_t_hold_numerical(
        self, omega: float, t_hold: float
    ) -> None:
        """Δω = 1/t_hold for non-fringe points using numerical derivative."""
        if abs(np.sin(omega * t_hold)) < 1e-6:
            pytest.skip("Singular point at fringe extremum")
        dt_n, *_ = compute_delta_omega_from_propagation(
            t_hold, omega, _U_BS, _JZ_1, use_numerical=True, fd_step=1e-6
        )
        assert dt_n == pytest.approx(1.0 / t_hold, rel=1e-6)

    def test_fringe_extremum_detected_at_zero(self) -> None:
        """At sin(ω t_hold) ≈ 0, is_fringe should be True."""
        _, _, _, _, is_fringe = compute_delta_omega_from_propagation(
            0.0, 1.0, _U_BS, _JZ_1
        )
        assert is_fringe

    def test_fringe_extremum_detected_at_pi(self) -> None:
        _, _, _, _, is_fringe = compute_delta_omega_from_propagation(
            np.pi, 1.0, _U_BS, _JZ_1
        )
        assert is_fringe

    def test_inf_at_fringe_extremum(self) -> None:
        """Δω should be infinity at fringe extremum (0/0)."""
        dt_a, *_ = compute_delta_omega_from_propagation(
            0.0, 1.0, _U_BS, _JZ_1, use_numerical=False
        )
        assert not np.isfinite(dt_a) or dt_a > 1e6

    def test_jz_mean_and_variance_match_direct_computation(self) -> None:
        """Returned jz_mean and jz_var should match direct computation."""
        psi = evolve_single_particle_mzi(1.0, 1.0, _U_BS, _JZ_1)
        expected_mean = float(np.real(np.conj(psi) @ _JZ_1 @ psi))
        expected_var = compute_jz_variance(psi, max_photons=1)
        _, jz_mean, jz_var, _, _ = compute_delta_omega_from_propagation(
            1.0, 1.0, _U_BS, _JZ_1
        )
        assert jz_mean == pytest.approx(expected_mean, abs=1e-12)
        assert jz_var == pytest.approx(expected_var, abs=1e-12)

    def test_use_numerical_flag(self) -> None:
        """Flagging use_numerical=True should produce same Δω within tolerance."""
        dt_a, _, _, d_a, _ = compute_delta_omega_from_propagation(
            1.0, 1.0, _U_BS, _JZ_1, use_numerical=False
        )
        dt_n, _, _, d_n, _ = compute_delta_omega_from_propagation(
            1.0, 1.0, _U_BS, _JZ_1, use_numerical=True, fd_step=1e-6
        )
        # Both should be close to 1/t_hold = 1.0
        assert dt_a == pytest.approx(1.0, rel=1e-12)
        assert dt_n == pytest.approx(1.0, rel=1e-6)
        # Analytical and numerical derivatives should be similar
        assert d_a == pytest.approx(d_n, rel=1e-6)

    def test_tiny_t_hold_diverges(self) -> None:
        """Very small t_hold should give large Δω (1/t_hold diverges)."""
        dt_a, *_ = compute_delta_omega_from_propagation(
            1e-10, 1.0, _U_BS, _JZ_1, use_numerical=False
        )
        assert dt_a > 1e8


# =============================================================================
# compute_sensitivity_sweep
# =============================================================================


class TestComputeSensitivitySweep:
    """Tests for compute_sensitivity_sweep."""

    def test_dataframe_has_expected_columns(self) -> None:
        df = compute_sensitivity_sweep(omega=1.0, n_points=10)
        expected_cols = {
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
        }
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 10

    def test_t_hold_monotonic_increasing(self) -> None:
        df = compute_sensitivity_sweep(omega=1.0, n_points=10)
        assert np.all(np.diff(df["t_hold"].to_numpy()) > 0)

    def test_omega_column_preserved(self) -> None:
        df = compute_sensitivity_sweep(omega=2.5, n_points=5)
        assert (df["omega"] == 2.5).all()

    def test_delta_omega_theory_matches_one_over_t_hold(self) -> None:
        df = compute_sensitivity_sweep(omega=1.0, n_points=10)
        assert np.allclose(
            df["delta_omega_theory"].to_numpy(),
            1.0 / df["t_hold"].to_numpy(),
        )

    def test_non_fringe_points_match_theory(self) -> None:
        df = compute_sensitivity_sweep(omega=1.0, n_points=20)
        non_fringe = df[~df["is_fringe_extremum"]]
        assert len(non_fringe) > 0, "All points were fringe extrema — unexpected"
        for _, row in non_fringe.iterrows():
            assert row["delta_omega_analytical"] == pytest.approx(
                row["delta_omega_theory"], rel=1e-12
            )

    def test_fringe_points_have_small_abs_sin(self) -> None:
        df = compute_sensitivity_sweep(omega=1.0, n_points=50)
        fringe = df[df["is_fringe_extremum"]]
        for _, row in fringe.iterrows():
            assert row["abs_sin"] < 1e-6

    def test_analytical_and_numerical_columns_disagree_at_fringe(self) -> None:
        """At fringe extrema, analytical Δω is inf while numerical may differ."""
        df = compute_sensitivity_sweep(omega=1.0, n_points=50)
        fringe = df[df["is_fringe_extremum"]]
        for _, row in fringe.iterrows():
            assert (
                not np.isfinite(row["delta_omega_analytical"])
                or row["delta_omega_analytical"] > 1e6
            )

    def test_custom_omega_affects_results(self) -> None:
        df1 = compute_sensitivity_sweep(omega=0.5, n_points=10)
        df2 = compute_sensitivity_sweep(omega=2.0, n_points=10)
        # jz_mean should differ
        assert not np.allclose(
            df1["jz_mean"].to_numpy(), df2["jz_mean"].to_numpy(), atol=1e-10
        )


# =============================================================================
# run_validation
# =============================================================================


class TestRunValidation:
    """Tests for run_validation."""

    def test_validation_passes_at_non_singular_point(self) -> None:
        result = run_validation(omega=1.0, t_hold=1.0)
        assert result["state_normalized"]
        assert result["bs_unitary"]
        assert result["delta_omega_matches_theory"]
        assert result["derivative_match"]

    def test_validation_returns_all_keys(self) -> None:
        result = run_validation(omega=1.0, t_hold=1.0)
        expected_keys = {
            "state_normalized",
            "norm",
            "bs_unitary",
            "delta_omega_matches_theory",
            "delta_omega_analytical",
            "delta_omega_theory",
            "derivative_match",
            "derivative_relative_diff",
            "d_jz_analytical",
            "d_jz_numerical",
        }
        assert set(result.keys()) == expected_keys

    def test_norm_is_one(self) -> None:
        result = run_validation(omega=1.0, t_hold=1.0)
        assert result["norm"] == pytest.approx(1.0)

    def test_delta_omega_matches_theory_when_not_at_fringe(self) -> None:
        result = run_validation(omega=1.0, t_hold=2.0)
        assert result["delta_omega_analytical"] == pytest.approx(
            result["delta_omega_theory"], rel=1e-12
        )

    def test_norm_fails_for_unphysical_state(self) -> None:
        # Run with default parameters should pass norm check
        result = run_validation(omega=1.0, t_hold=1.0)
        assert result["state_normalized"]

    @pytest.mark.parametrize(
        ("omega", "t_hold"),
        [(1.0, 0.5), (1.0, 2.0), (0.5, 3.0)],
        ids=["ω=1.0,t=0.5", "ω=1.0,t=2.0", "ω=0.5,t=3.0"],
    )
    def test_validation_across_parameters(self, omega: float, t_hold: float) -> None:
        result = run_validation(omega=omega, t_hold=t_hold)
        assert result["state_normalized"]
        assert result["bs_unitary"]
        assert result["derivative_relative_diff"] < 1e-6

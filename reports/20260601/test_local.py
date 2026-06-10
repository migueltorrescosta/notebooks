r"""
Tests for the Heisenberg-Limit MZI: NOON & Twin-Fock module (2026-06-01).

Run with:
    uv run pytest reports/20260601/test_local.py -q --tb=short
"""

from __future__ import annotations

import subprocess
import sys as _sys
from pathlib import Path

import numpy as np
import pytest

from src.physics.mzi_simulation import beam_splitter_unitary
from src.physics.mzi_states import (
    compute_jz_expectation,
    compute_jz_variance,
    input_state_factory,
)

_report_dir = str(
    Path(__file__).resolve().parent.parent.parent / "reports" / "20260601",
)
if _report_dir not in _sys.path:
    _sys.path.insert(0, _report_dir)
del _sys, _report_dir

from local import (  # type: ignore[import-untyped]  # noqa: E402
    MziSensitivityData,
    _make_standard_twin_fock_state,
    analyse_best_worst_sensitivity,
    compute_fisher_classical,
    compute_mzi_sensitivity_grid,
    fit_scaling_exponent,
    output_number_diff_distribution,
    simple_mzi_evolution,
    t_hold,
)

# ============================================================================
# Standard Twin-Fock |N/2, N/2⟩ State
# ============================================================================


class TestStandardTwinFockState:
    def test_normalized(self) -> None:
        """The standard Twin-Fock state must have unit norm."""
        for N in [2, 4, 6, 10]:
            state = _make_standard_twin_fock_state(N, N)
            assert np.isclose(np.linalg.norm(state), 1.0), f"Failed for N={N}"

    def test_jz_expectation_zero(self) -> None:
        r"""⟨J_z⟩ = 0 for |N/2, N/2⟩ before any BS."""
        for N in [2, 4, 6, 10]:
            state = _make_standard_twin_fock_state(N, N)
            exp = np.real(compute_jz_expectation(state, N))
            assert np.isclose(exp, 0.0, atol=1e-12), f"Failed for N={N}"

    def test_jz_variance_zero_before_bs(self) -> None:
        r"""Var(J_z) = 0 for |N/2, N/2⟩ before any BS (number eigenstate)."""
        for N in [2, 4, 6, 10]:
            state = _make_standard_twin_fock_state(N, N)
            var = compute_jz_variance(state, N)
            assert np.isclose(var, 0.0, atol=1e-12), f"Failed for N={N}"

    def test_fock_index_correct(self) -> None:
        r"""The state must be exactly |N/2, N/2⟩ in the Fock basis."""
        N = 4
        state = _make_standard_twin_fock_state(N, N)
        dim = (N + 1) ** 2
        expected = np.zeros(dim, dtype=complex)
        idx = (N // 2) * (N + 1) + (N // 2)  # |2, 2⟩
        expected[idx] = 1.0
        assert np.allclose(state, expected)

    def test_odd_N_raises(self) -> None:
        r"""Odd N is invalid for |N/2, N/2⟩."""
        with pytest.raises(ValueError, match="even"):
            _make_standard_twin_fock_state(3, 5)

    def test_max_photons_larger_than_N(self) -> None:
        """State works when max_photons > N (padding with extra basis states)."""
        state = _make_standard_twin_fock_state(4, 10)
        assert np.isclose(np.linalg.norm(state), 1.0)
        # Dimension should be (10+1)^2 = 121
        assert len(state) == 121


# ============================================================================
# Twin-Fock After BS1: Correct Variance
# ============================================================================


class TestTwinFockAfterBS1:
    @pytest.mark.parametrize("N", [2, 4, 6, 10])
    def test_variance_after_bs1(self, N: int) -> None:
        r"""After a 50/50 BS, |N/2,N/2⟩ has Var(J_z) = N(N+2)/8.

        Note: this differs from the N/4 scaling that would hold for a
        different BS convention. The codebase's BS gives N(N+2)/8.
        """
        state = _make_standard_twin_fock_state(N, N)
        bs = beam_splitter_unitary(np.pi / 4, 0.0, N)
        state_bs1 = bs @ state
        var = compute_jz_variance(state_bs1, N)
        expected = N * (N + 2) / 8.0
        assert np.isclose(var, expected, rtol=1e-10), (
            f"N={N}: Var(J_z)={var}, expected {expected}"
        )


# ============================================================================
# Simple MZI Evolution
# ============================================================================


class TestSimpleMziEvolution:
    @pytest.mark.parametrize("N", [1, 2, 4, 6])
    def test_norm_preserved_noon(self, N: int) -> None:
        """MZI evolution must preserve norm for NOON state."""
        state = input_state_factory("noon", N, N)
        final = simple_mzi_evolution(
            state,
            omega=1.0,
            max_photons=N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        assert np.isclose(np.linalg.norm(final), 1.0, rtol=1e-10), f"Failed for N={N}"

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_norm_preserved_twin_fock_std(self, N: int) -> None:
        """MZI evolution must preserve norm for standard Twin-Fock."""
        state = _make_standard_twin_fock_state(N, N)
        final = simple_mzi_evolution(
            state,
            omega=1.0,
            max_photons=N,
            t_hold=t_hold,
            skip_bs1=False,
        )
        assert np.isclose(np.linalg.norm(final), 1.0, rtol=1e-10), f"Failed for N={N}"

    def test_zero_phase_norm_preserved(self) -> None:
        r"""At θ=0, the MZI output must have unit norm."""
        N = 4
        state = input_state_factory("noon", N, N)
        final = simple_mzi_evolution(
            state,
            omega=0.0,
            max_photons=N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        assert np.isclose(np.linalg.norm(final), 1.0, rtol=1e-10)

    @pytest.mark.parametrize("omega", [0.1, 0.5, 1.0, 2.0])
    def test_output_variance_nonnegative(self, omega: float) -> None:
        """Output variance of J_z must be non-negative."""
        N = 4
        state = input_state_factory("noon", N, N)
        final = simple_mzi_evolution(
            state,
            omega=omega,
            max_photons=N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        var = compute_jz_variance(final, N)
        assert var >= -1e-12, f"Negative variance: {var}"


# ============================================================================
# NOON QFI Bound Validation (skip BS1: probe = NOON state)
# ============================================================================


class TestNOONQFI:
    @pytest.mark.parametrize("N", [1, 2, 4, 6, 10])
    def test_var_jz_of_input(self, N: int) -> None:
        r"""Var(J_z) = N²/4 for NOON input state (the probe, since skip_bs1)."""
        state = input_state_factory("noon", N, N)
        var = compute_jz_variance(state, N)
        assert np.isclose(var, N**2 / 4, rtol=1e-10), (
            f"N={N}: Var={var}, expected {N**2 / 4}"
        )

    @pytest.mark.parametrize("N", [1, 2, 4, 6, 10])
    def test_qfi_bound(self, N: int) -> None:
        r"""F_Q = t_hold² N², Δθ_Q = 1/(t_hold N) for NOON (probe = input, skip BS1)."""
        state = input_state_factory("noon", N, N)
        var = compute_jz_variance(state, N)
        fq = 4.0 * t_hold**2 * var
        expected_fq = t_hold**2 * N**2
        assert np.isclose(fq, expected_fq, rtol=1e-10), (
            f"N={N}: F_Q={fq}, expected {expected_fq}"
        )
        delta_q = 1.0 / np.sqrt(fq)
        expected_delta_q = 1.0 / (t_hold * N)
        assert np.isclose(delta_q, expected_delta_q, rtol=1e-10), (
            f"N={N}: Δθ_Q={delta_q}, expected {expected_delta_q}"
        )

    @pytest.mark.parametrize("N", [2, 4, 6, 10])
    def test_probe_unchanged_by_bs(self, N: int) -> None:
        r"""NOON probe with skip_bs1=True is the input itself (no BS1 applied)."""
        state = input_state_factory("noon", N, N)
        result = compute_mzi_sensitivity_grid(
            state,
            np.linspace(0.1, 5.0, 5),
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        var_from_qfi = result["fisher_quantum"] / (4.0 * t_hold**2)
        expected_var = N**2 / 4.0
        assert np.isclose(var_from_qfi, expected_var, rtol=1e-10), (
            f"N={N}: probe Var(J_z)={var_from_qfi}, expected {expected_var}"
        )


# ============================================================================
# Twin-Fock QFI Bound Validation (with BS1)
# ============================================================================


class TestTwinFockStdQFI:
    @pytest.mark.parametrize("N", [2, 4, 6, 10])
    def test_qfi_bound(self, N: int) -> None:
        r"""F_Q = t_hold² · N(N+2)/2, Δθ_Q = 1 / (t_hold · √(N(N+2)/2)) for TF after BS1."""
        state = _make_standard_twin_fock_state(N, N)
        bs = beam_splitter_unitary(np.pi / 4, 0.0, N)
        state_bs1 = bs @ state
        var = compute_jz_variance(state_bs1, N)
        expected_var = N * (N + 2) / 8.0
        assert np.isclose(var, expected_var, rtol=1e-10), (
            f"N={N}: Var_probe(J_z)={var}, expected {expected_var}"
        )
        fq = 4.0 * t_hold**2 * var
        expected_fq = t_hold**2 * N * (N + 2) / 2.0
        assert np.isclose(fq, expected_fq, rtol=1e-10), (
            f"N={N}: F_Q={fq}, expected {expected_fq}"
        )
        delta_q = 1.0 / np.sqrt(fq)
        expected_delta_q = 1.0 / (t_hold * np.sqrt(N * (N + 2) / 2.0))
        assert np.isclose(delta_q, expected_delta_q, rtol=1e-10), (
            f"N={N}: Δθ_Q={delta_q}, expected {expected_delta_q}"
        )


# ============================================================================
# MZI Sensitivity Grid Computation
# ============================================================================


class TestComputeMziSensitivityGrid:
    def test_returns_required_keys(self) -> None:
        """Result dict must contain all expected arrays."""
        N = 4
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        expected_keys = {
            "omega_values",
            "expectation_values",
            "variance_values",
            "derivative_values",
            "delta_omega_ep",
            "delta_omega_q",
            "fisher_quantum",
            "fisher_classical",
            "delta_omega_c",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_shapes_match(self) -> None:
        """All grid arrays must have the same length as omega_grid."""
        N = 4
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        n_omega = len(omega_grid)
        for key in [
            "expectation_values",
            "variance_values",
            "derivative_values",
            "delta_omega_ep",
            "fisher_classical",
            "delta_omega_c",
        ]:
            assert len(result[key]) == n_omega, (
                f"Key '{key}' has length {len(result[key])}, expected {n_omega}"
            )
        assert np.isscalar(result["delta_omega_q"])
        assert np.isscalar(result["fisher_quantum"])

    def test_variance_nonnegative(self) -> None:
        """Variance must be ≥ 0 at all θ."""
        N = 4
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        assert np.all(result["variance_values"] >= -1e-12)

    def test_finite_delta_omega_ep(self) -> None:
        r"""Δθ_EP should be finite at most θ values."""
        N = 4
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        finite_count = np.sum(np.isfinite(result["delta_omega_ep"]))
        assert finite_count >= len(omega_grid) // 2, (
            f"Only {finite_count}/{len(omega_grid)} finite Δθ_EP values"
        )

    def test_cramer_rao_inequality_noon(self) -> None:
        r"""Δθ_EP ≥ Δθ_Q for NOON (Cramér-Rao bound)."""
        N = 4
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        delta_q = result["delta_omega_q"]
        for ep_val in result["delta_omega_ep"]:
            if np.isfinite(ep_val):
                assert ep_val >= delta_q - 1e-10, (
                    f"Δθ_EP={ep_val} < Δθ_Q={delta_q} violates Cramér-Rao bound"
                )

    def test_cramer_rao_inequality_twin_fock(self) -> None:
        r"""Δθ_EP ≥ Δθ_Q for standard Twin-Fock (Cramér-Rao bound)."""
        N = 4
        state = _make_standard_twin_fock_state(N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=False,
        )
        delta_q = result["delta_omega_q"]
        for ep_val in result["delta_omega_ep"]:
            if np.isfinite(ep_val):
                assert ep_val >= delta_q - 1e-10, (
                    f"Δθ_EP={ep_val} < Δθ_Q={delta_q} violates Cramér-Rao bound"
                )

    def test_noon_qfi_invariant_omega(self) -> None:
        r"""Δθ_Q is θ-independent for NOON (pure state, fixed generator)."""
        N = 4
        state = input_state_factory("noon", N, N)
        grid1 = np.linspace(0.1, 2.0, 5)
        grid2 = np.linspace(3.0, 5.0, 5)
        r1 = compute_mzi_sensitivity_grid(
            state,
            grid1,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        r2 = compute_mzi_sensitivity_grid(
            state,
            grid2,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        assert np.isclose(r1["delta_omega_q"], r2["delta_omega_q"], rtol=1e-10)

    # --- CFI-specific tests ---

    def test_cfi_positivity(self) -> None:
        r"""F_C(θ) must be ≥ 0 at all θ."""
        N = 4
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        assert np.all(result["fisher_classical"] >= -1e-12), (
            "Some F_C values are negative"
        )

    def test_cfi_finite_delta_omega_c(self) -> None:
        r"""Δθ_C should be finite at most θ values (full distribution)."""
        N = 4
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        finite_c = np.sum(np.isfinite(result["delta_omega_c"]))
        finite_ep = np.sum(np.isfinite(result["delta_omega_ep"]))
        # CFI approach should give at least as many finite points as EP
        assert finite_c >= finite_ep, (
            f"Δθ_C has {finite_c} finite values vs Δθ_EP has {finite_ep}"
        )
        assert finite_c >= len(omega_grid) // 2, (
            f"Only {finite_c}/{len(omega_grid)} finite Δθ_C values"
        )

    @pytest.mark.parametrize("N", [1, 2, 4])
    def test_cfi_cramer_rao_noon(self, N: int) -> None:
        r"""Δθ_C ≥ Δθ_Q for NOON (quantum Cramér-Rao bound)."""
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        delta_q = result["delta_omega_q"]
        for c_val in result["delta_omega_c"]:
            if np.isfinite(c_val):
                assert c_val >= delta_q - 1e-10, (
                    f"N={N}: Δθ_C={c_val} < Δθ_Q={delta_q} violates Cramér-Rao bound"
                )

    @pytest.mark.parametrize("N", [2, 4])
    def test_cfi_cramer_rao_twin_fock(self, N: int) -> None:
        r"""Δθ_C ≥ Δθ_Q for standard Twin-Fock (quantum Cramér-Rao bound)."""
        state = _make_standard_twin_fock_state(N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=False,
        )
        delta_q = result["delta_omega_q"]
        for c_val in result["delta_omega_c"]:
            if np.isfinite(c_val):
                assert c_val >= delta_q - 1e-10, (
                    f"N={N}: Δθ_C={c_val} < Δθ_Q={delta_q} violates Cramér-Rao bound"
                )

    def test_cfi_epsilon_robustness(self) -> None:
        r"""F_C computed with different ε (1e-6 vs 1e-7) should be similar."""
        N = 2
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.1, 5.0, 10)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )

        alt_eps = 1e-7
        for i in [0, 3, 7]:
            omega = omega_grid[i]
            fc_default = float(result["fisher_classical"][i])
            if not np.isfinite(fc_default):
                continue

            state_out = simple_mzi_evolution(
                state,
                omega,
                N,
                t_hold=t_hold,
                skip_bs1=True,
            )
            state_plus = simple_mzi_evolution(
                state,
                omega + alt_eps,
                N,
                t_hold=t_hold,
                skip_bs1=True,
            )
            state_minus = simple_mzi_evolution(
                state,
                omega - alt_eps,
                N,
                t_hold=t_hold,
                skip_bs1=True,
            )
            P_omega = output_number_diff_distribution(state_out, N)
            P_plus = output_number_diff_distribution(state_plus, N)
            P_minus = output_number_diff_distribution(state_minus, N)
            fc_alt = compute_fisher_classical(
                P_omega,
                P_plus,
                P_minus,
                epsilon=alt_eps,
            )

            if np.isfinite(fc_alt):
                assert np.isclose(fc_default, fc_alt, rtol=0.02), (
                    f"θ={omega}: F_C(1e-6)={fc_default}, F_C(1e-7)={fc_alt}, "
                    f"rel diff={abs(fc_default - fc_alt) / max(fc_default, fc_alt):.6e}"
                )


# ============================================================================
# NOON vs Twin-Fock QFI Comparison
# ============================================================================


class TestNOONvsTwinFockQFI:
    @pytest.mark.parametrize("N", [4, 6, 10, 20])
    def test_noon_beats_twin_fock(self, N: int) -> None:
        r"""NOON QFI should exceed Twin-Fock QFI by factor ~2N/(N+2) ≈ 2."""
        noon_state = input_state_factory("noon", N, N)
        tf_state = _make_standard_twin_fock_state(N, N)
        bs = beam_splitter_unitary(np.pi / 4, 0.0, N)

        # NOON QFI from probe = input (skip BS1)
        var_noon = compute_jz_variance(noon_state, N)
        fq_noon = 4.0 * t_hold**2 * var_noon

        # Twin-Fock QFI from probe = BS1 @ input
        tf_probe = bs @ tf_state
        var_tf = compute_jz_variance(tf_probe, N)
        fq_tf = 4.0 * t_hold**2 * var_tf

        # NOON should be better by factor N² / (N(N+2)/2) = 2N/(N+2) ≈ 2
        expected_ratio = N**2 / (N * (N + 2) / 2.0)
        computed_ratio = fq_noon / fq_tf
        assert np.isclose(computed_ratio, expected_ratio, rtol=1e-10), (
            f"N={N}: F_Q_noon/F_Q_tf={computed_ratio}, expected {expected_ratio}"
        )


# ============================================================================
# Scaling Exponent Fitting
# ============================================================================


class TestFitScalingExponent:
    def test_noon_exponent(self) -> None:
        r"""NOON states should give α ≈ -1.0 (Heisenberg)."""
        N_vals = np.array([2, 4, 6, 10, 14, 20], dtype=float)
        delta_vals = 1.0 / (t_hold * N_vals)
        result = fit_scaling_exponent(N_vals, delta_vals)
        assert result.valid, f"Fit invalid: {result.warnings}"
        assert np.isclose(result.alpha, -1.0, atol=0.02), (
            f"NOON exponent α={result.alpha}, expected -1.0"
        )
        assert np.isclose(result.C, 1.0 / t_hold, rtol=0.02)

    def test_twin_fock_heisenberg_exponent(self) -> None:
        r"""Twin-Fock |N/2,N/2⟩ after BS1 gives α ≈ -1.0 (near-Heisenberg).

        The correct QFI is F_Q = t_hold²·N(N+2)/2, so the scaling exponent
        is δ = -1.0 for large N (Heisenberg-like), not -0.5 (SQL) as the
        report initially estimated. The prefactor √2 makes TF a factor
        of √2 worse than NOON at the same N.

        The N+2 correction gives α = -1 + 1/(N+2), which biases the fit
        by about 0.03 at N=10 and 0.005 at N=200. We use a relaxed
        tolerance (atol=0.05) to account for this finite-N effect.
        """
        N_vals = np.array([10, 20, 40, 80, 120, 200], dtype=float)
        # Δθ_Q = 1 / (t_hold · √(N(N+2)/2)) → ∝ 1/N for large N
        delta_vals = 1.0 / (t_hold * np.sqrt(N_vals * (N_vals + 2) / 2.0))
        result = fit_scaling_exponent(N_vals, delta_vals, N_min=10)
        assert result.valid, f"Fit invalid: {result.warnings}"
        assert np.isclose(result.alpha, -1.0, atol=0.05), (
            f"TF exponent α={result.alpha}, expected ~-1.0"
        )
        # The asymptotic value is -1.0; finite-N correction makes it
        # slightly shallower: α = -1 + 1/(N+2) → α ≈ -0.97 for this range
        assert result.alpha > -0.98, (
            f"TF exponent α={result.alpha} is too steep (should be ≈ -0.97)"
        )

    def test_fit_with_noise(self) -> None:
        """Fit should be robust to small numerical noise."""
        rng = np.random.default_rng(42)
        N_vals = np.array([2, 4, 6, 10, 14, 20], dtype=float)
        true_alpha = -1.0
        true_C = 1.0 / t_hold
        delta_vals = true_C * N_vals**true_alpha
        delta_vals *= 1.0 + 0.01 * rng.normal(size=len(N_vals))
        result = fit_scaling_exponent(N_vals, delta_vals)
        assert result.valid, f"Fit invalid: {result.warnings}"
        assert np.isclose(result.alpha, true_alpha, atol=0.05), (
            f"Noisy NOON exponent α={result.alpha}, expected {true_alpha}"
        )

    def test_min_N_filter(self) -> None:
        """N_min parameter should exclude small N from fit."""
        N_vals = np.array([1, 2, 4, 6, 10, 20], dtype=float)
        delta_vals = 1.0 / (t_hold * N_vals)
        result = fit_scaling_exponent(N_vals, delta_vals, N_min=4)
        assert result.valid, f"Fit invalid: {result.warnings}"
        assert len(result.N_values) == 4, (
            f"Expected 4 fit points (N>=4), got {len(result.N_values)}"
        )
        assert np.all(result.N_values >= 4)

    def test_insufficient_points(self) -> None:
        """Fewer than 3 points should return invalid result (valid=False)."""
        N_vals = np.array([2], dtype=float)
        delta_vals = np.array([0.1])
        result = fit_scaling_exponent(N_vals, delta_vals)
        assert not result.valid, "Fit should be invalid with only 1 point"


# ============================================================================
# MziSensitivityData — Parquet Roundtrip
# ============================================================================


class TestMziSensitivityDataParquet:
    @pytest.fixture
    def make_result(self) -> MziSensitivityData:
        n_n = 3
        n_t = 5
        rng_ep = np.random.default_rng(45)
        rng_c = np.random.default_rng(46)
        return MziSensitivityData(
            state_type="noon",
            N_values=np.array([2, 4, 6], dtype=int),
            omega_values=np.linspace(0.1, 5.0, n_t),
            expectation_grid=np.random.default_rng(42).uniform(-1, 1, (n_n, n_t)),
            variance_grid=np.random.default_rng(43).uniform(0, 0.5, (n_n, n_t)),
            derivative_grid=np.random.default_rng(44).uniform(-2, 2, (n_n, n_t)),
            delta_omega_ep_grid=rng_ep.uniform(0.01, 1, (n_n, n_t)),
            delta_omega_q_per_N=np.array([0.05, 0.025, 0.0167]),
            fisher_classical_grid=rng_c.uniform(1, 100, (n_n, n_t)),
            delta_omega_c_grid=rng_ep.uniform(0.01, 1, (n_n, n_t)),
            t_hold=t_hold,
        )

    def test_roundtrip(self, make_result: MziSensitivityData, tmp_path: Path) -> None:
        p = tmp_path / "sensitivity.parquet"
        make_result.save_parquet(p)
        loaded = MziSensitivityData.from_parquet(p)
        assert loaded.state_type == make_result.state_type
        assert np.allclose(loaded.N_values, make_result.N_values)
        assert np.allclose(loaded.omega_values, make_result.omega_values)
        assert np.allclose(loaded.expectation_grid, make_result.expectation_grid)
        assert np.allclose(loaded.variance_grid, make_result.variance_grid)
        assert np.allclose(loaded.derivative_grid, make_result.derivative_grid)
        assert np.allclose(loaded.delta_omega_ep_grid, make_result.delta_omega_ep_grid)
        assert np.allclose(loaded.delta_omega_q_per_N, make_result.delta_omega_q_per_N)
        assert np.allclose(
            loaded.fisher_classical_grid, make_result.fisher_classical_grid
        )
        assert np.allclose(loaded.delta_omega_c_grid, make_result.delta_omega_c_grid)
        assert np.isclose(loaded.t_hold, make_result.t_hold)

    def test_fail_fast_missing_column(
        self, make_result: MziSensitivityData, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["state_type"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            MziSensitivityData.from_parquet(p)

    def test_fail_fast_missing_cfi_column(
        self, make_result: MziSensitivityData, tmp_path: Path
    ) -> None:
        p = tmp_path / "bad_cfi.parquet"
        df = make_result.to_dataframe()
        df = df.drop(columns=["fisher_classical"])
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            MziSensitivityData.from_parquet(p)


# ============================================================================
# Integration: Generate Theta Scan
# ============================================================================


class TestGenerateOmegaScan:
    def test_generate_noon_scan(self) -> None:
        """Generate a small θ scan for NOON and verify basic properties."""
        from local import generate_omega_scan

        omega_grid = np.linspace(0.1, 5.0, 5)
        result = generate_omega_scan(
            "noon", N=4, omega_grid=omega_grid, max_photons=4, t_hold=t_hold
        )
        assert result.state_type == "noon"
        assert result.N_values[0] == 4
        assert len(result.omega_values) == 5
        assert np.all(np.isfinite(result.delta_omega_ep_grid[0]))
        assert np.all(np.isfinite(result.fisher_classical_grid[0]))

    def test_generate_twin_fock_scan(self) -> None:
        """Generate a small θ scan for standard Twin-Fock."""
        from local import generate_omega_scan

        omega_grid = np.linspace(0.1, 5.0, 5)
        result = generate_omega_scan(
            "twin_fock_std", N=4, omega_grid=omega_grid, max_photons=4, t_hold=t_hold
        )
        assert result.state_type == "twin_fock_std"
        assert result.N_values[0] == 4
        assert len(result.omega_values) == 5
        assert np.all(np.isfinite(result.fisher_classical_grid[0]))


# ============================================================================
# Output Number-Difference Distribution P(m|θ)
# ============================================================================


class TestOutputNumberDiffDistribution:
    def test_normalized(self) -> None:
        """Sum of P(m|θ) = 1 for any output state."""
        N = 4
        state = input_state_factory("noon", N, N)
        out = simple_mzi_evolution(
            state, omega=0.5, max_photons=N, t_hold=t_hold, skip_bs1=True
        )
        P = output_number_diff_distribution(out, N)
        assert np.isclose(np.sum(P), 1.0, rtol=1e-12), f"Sum={np.sum(P)}"

    def test_nonnegative(self) -> None:
        """All P(m) ≥ 0."""
        N = 4
        state = input_state_factory("noon", N, N)
        out = simple_mzi_evolution(
            state, omega=0.5, max_photons=N, t_hold=t_hold, skip_bs1=True
        )
        P = output_number_diff_distribution(out, N)
        assert np.all(P >= -1e-15), "Some probabilities are negative"

    def test_shape(self) -> None:
        """Array shape = (2*max_photons+1,)."""
        for M in [2, 4, 6]:
            state = input_state_factory("noon", M, M)
            out = simple_mzi_evolution(
                state, omega=0.5, max_photons=M, t_hold=t_hold, skip_bs1=True
            )
            P = output_number_diff_distribution(out, M)
            assert P.shape == (2 * M + 1,), (
                f"max_photons={M}: shape={P.shape}, expected {(2 * M + 1,)}"
            )

    def test_number_difference_fock(self) -> None:
        """For a Fock state |n1,n2⟩, P(m) = 1 at m = n1-n2 and 0 elsewhere."""
        M = 6
        n1, n2 = 4, 2
        dim = (M + 1) ** 2
        state = np.zeros(dim, dtype=complex)
        state[n1 * (M + 1) + n2] = 1.0 + 0.0j
        P = output_number_diff_distribution(state, M)
        offset = M
        expected_m = n1 - n2
        assert np.isclose(P[expected_m + offset], 1.0, rtol=1e-12)
        assert np.sum(P) == 1.0

    def test_number_difference_symmetric(self) -> None:
        r"""For (|2,0⟩+|0,2⟩)/√2 (NOON), P(m) peaks at m=±2 with p=0.5 each."""
        M = 4  # max_photons large enough for |2,0⟩ and |0,2⟩
        dim = (M + 1) ** 2
        state = np.zeros(dim, dtype=complex)
        idx_20 = 2 * (M + 1) + 0  # |2,0⟩
        idx_02 = 0 * (M + 1) + 2  # |0,2⟩
        state[idx_20] = 1.0 / np.sqrt(2)
        state[idx_02] = 1.0 / np.sqrt(2)
        P = output_number_diff_distribution(state, M)
        offset = M
        # m = +2
        assert np.isclose(P[2 + offset], 0.5, rtol=1e-12)
        # m = -2
        assert np.isclose(P[-2 + offset], 0.5, rtol=1e-12)
        # All other entries are zero
        nonzero = np.where(P > 1e-15)[0]
        assert len(nonzero) == 2


# ============================================================================
# Classical Fisher Information
# ============================================================================


class TestClassicalFisher:
    def test_cfi_positivity(self) -> None:
        r"""F_C ≥ 0 for a simple test case."""
        eps = 1e-6
        P_omega = np.array([0.5, 0.0, 0.5])
        P_plus = np.array([0.5 + 5e-6, 0.0, 0.5 - 5e-6])
        P_minus = np.array([0.5 - 5e-6, 0.0, 0.5 + 5e-6])
        fc = compute_fisher_classical(P_omega, P_plus, P_minus, epsilon=eps)
        assert fc >= 0.0

    def test_cfi_known_noon_n1(self) -> None:
        r"""For NOON N=1 at θ=0, F_C = t_hold² = 100."""
        t_hold_val = 10.0
        eps = 1e-6
        # At θ=0: P(m=±1) = 0.5, P(m=0) = 0
        P_omega = np.array([0.5, 0.0, 0.5])
        # P(θ+ε): probability shifts toward m=+1
        P_plus = np.array([0.5 - t_hold_val * eps / 2, 0.0, 0.5 + t_hold_val * eps / 2])
        # P(θ-ε): probability shifts toward m=-1
        P_minus = np.array(
            [0.5 + t_hold_val * eps / 2, 0.0, 0.5 - t_hold_val * eps / 2]
        )
        fc = compute_fisher_classical(P_omega, P_plus, P_minus, epsilon=eps)
        expected_fc = t_hold_val**2  # F_Q = t_hold²·N² with N=1
        assert np.isclose(fc, expected_fc, rtol=1e-5), (
            f"F_C={fc}, expected {expected_fc}"
        )

    def test_cfi_vanishes_at_null(self) -> None:
        """When P_omega = P_plus = P_minus, F_C = 0."""
        eps = 1e-6
        P = np.array([0.5, 0.0, 0.5])
        fc = compute_fisher_classical(P, P, P, epsilon=eps)
        assert np.isclose(fc, 0.0, atol=1e-20), f"F_C={fc}, expected 0"

    def test_cfi_epsilon_invariance(self) -> None:
        r"""F_C computed with different ε should be similar (within 1%)."""
        t_hold_val = 10.0
        P_omega = np.array([0.5, 0.0, 0.5])

        fc_values = []
        for eps in [1e-6, 1e-7, 1e-8]:
            P_plus = np.array(
                [
                    0.5 - t_hold_val * eps / 2,
                    0.0,
                    0.5 + t_hold_val * eps / 2,
                ]
            )
            P_minus = np.array(
                [
                    0.5 + t_hold_val * eps / 2,
                    0.0,
                    0.5 - t_hold_val * eps / 2,
                ]
            )
            fc = compute_fisher_classical(P_omega, P_plus, P_minus, epsilon=eps)
            fc_values.append(fc)

        for i in range(1, len(fc_values)):
            if fc_values[0] > 0 and fc_values[i] > 0:
                rel_diff = abs(fc_values[i] - fc_values[0]) / fc_values[0]
                assert rel_diff < 0.01, (
                    f"ε={[1e-6, 1e-7, 1e-8][i]}: rel_diff={rel_diff:.6e}"
                )

    def test_cfi_protected_division(self) -> None:
        r"""When P(m) = 0 for some outcomes, those terms contribute 0 (no /0)."""
        P_omega = np.array([0.5, 0.0, 0.5])
        # Non-zero dP/dθ in the middle entry (P=0) would cause /0 if unprotected
        P_plus = np.array([0.6, 0.2, 0.2])
        P_minus = np.array([0.4, -0.2, 0.8])
        fc = compute_fisher_classical(P_omega, P_plus, P_minus, epsilon=1e-6)
        assert np.isfinite(fc), "F_C should be finite (no division by zero)"
        assert fc >= 0.0


# ============================================================================
# Analyse Best/Worst Sensitivity
# ============================================================================


class TestAnalyseBestWorstSensitivity:
    def test_returns_dict_keys(self) -> None:
        r"""analyse_best_worst_sensitivity returns expected keys."""
        N_vals = np.array([2, 4])
        omega_vals = np.linspace(0.1, 5.0, 10)
        grid = np.random.default_rng(42).uniform(0.01, 1, (2, 10))
        result = analyse_best_worst_sensitivity(N_vals, omega_vals, grid)
        expected_keys = {
            "N_values",
            "best_sensitivity",
            "best_omega",
            "worst_sensitivity",
            "worst_omega",
        }
        assert result.keys() == expected_keys, (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_best_is_minimum(self) -> None:
        r"""Best sensitivity ≤ worst sensitivity at each N."""
        N_vals = np.array([2, 4, 6])
        omega_vals = np.linspace(0.1, 5.0, 20)
        rng = np.random.default_rng(42)
        grid = rng.uniform(0.01, 1, (3, 20))
        result = analyse_best_worst_sensitivity(N_vals, omega_vals, grid)
        assert np.all(result["best_sensitivity"] <= result["worst_sensitivity"] + 1e-15)

    def test_simple_case(self) -> None:
        r"""With a sinusoidal sensitivity, best/worst detection is correct."""
        N_vals = np.array([4])
        omega_vals = np.linspace(0, 2 * np.pi, 100)
        # sensitivity = 2 + sin(θ): best=1 (θ≈3π/2), worst=3 (θ≈π/2)
        sens = 2.0 + np.sin(omega_vals)
        sensitivity_grid = sens.reshape(1, -1)
        result = analyse_best_worst_sensitivity(N_vals, omega_vals, sensitivity_grid)
        best_idx = int(np.argmin(sens))
        worst_idx = int(np.argmax(sens))
        assert np.isclose(
            result["best_sensitivity"][0],
            sens[best_idx],
            rtol=1e-10,
        ), "Best sensitivity mismatch"
        assert np.isclose(
            result["best_omega"][0],
            omega_vals[best_idx],
            rtol=1e-10,
        ), "Best omega mismatch"
        assert np.isclose(
            result["worst_sensitivity"][0],
            sens[worst_idx],
            rtol=1e-10,
        ), "Worst sensitivity mismatch"
        assert np.isclose(
            result["worst_omega"][0],
            omega_vals[worst_idx],
            rtol=1e-10,
        ), "Worst omega mismatch"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_noon_N_1_works(self) -> None:
        r"""NOON with N=1 is (|1,0⟩+|0,1⟩)/√2, should work fine."""
        N = 1
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        assert np.isfinite(result["delta_omega_q"])
        assert result["delta_omega_q"] > 0

    def test_twin_fock_min_N(self) -> None:
        r"""The smallest standard Twin-Fock is |1,1⟩ (N=2)."""
        N = 2
        state = _make_standard_twin_fock_state(N, N)
        omega_grid = np.linspace(0.1, 5.0, 5)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=False,
        )
        assert np.isfinite(result["delta_omega_q"])
        assert result["delta_omega_q"] > 0

    def test_no_sensitivity_at_fringe(self) -> None:
        r"""At θ where derivative ≈ 0, Δθ_EP should be very large or inf."""
        N = 4
        state = input_state_factory("noon", N, N)
        omega_grid = np.linspace(0.0, np.pi, 50)
        result = compute_mzi_sensitivity_grid(
            state,
            omega_grid,
            N,
            t_hold=t_hold,
            skip_bs1=True,
        )
        max_dt = np.max(result["delta_omega_ep"][np.isfinite(result["delta_omega_ep"])])
        assert max_dt > 10 * result["delta_omega_q"], (
            f"Max Δθ_EP={max_dt} not far above QFI bound={result['delta_omega_q']}"
        )

    def test_noon_n2_variance_input(self) -> None:
        r"""Var(n₂) = N²/4 for NOON input (used as probe with skip_bs1)."""
        N = 4
        state = input_state_factory("noon", N, N)
        # Compute number operator expectation
        n_op_mat = np.diag(
            np.array([idx % (N + 1) for idx in range((N + 1) ** 2)], dtype=float)
        )
        exp_n = np.real(state.conj() @ n_op_mat @ state)
        exp_n2 = np.real(state.conj() @ (n_op_mat @ n_op_mat) @ state)
        var_n = exp_n2 - exp_n**2
        assert np.isclose(var_n, N**2 / 4, rtol=1e-10), (
            f"N={N}: Var(n₂)={var_n}, expected {N**2 / 4}"
        )


# ============================================================================
# CLI
# ============================================================================


class TestCLI:
    def test_cli_help(self) -> None:
        result = subprocess.run(  # noqa: PLW1510
            [
                "uv",
                "run",
                "python",
                str(Path(__file__).resolve().parent / "local.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

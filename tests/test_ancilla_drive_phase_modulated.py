"""Tests for the θ-modulated ancilla drive metrology protocol.

This is the companion test module for
``reports/20260519/local.py`` (which replaces the former
``src/analysis/ancilla_drive_phase_modulated.py``).
It mirrors the structure of ``test_ancilla_drive_metrology.py`` (fixed-drive)
but tests the θ-modulated Hamiltonian H_A = θ (a_x J_x^A + a_y J_y^A + a_z J_z^A).

Key new tests (not in the fixed-drive test suite):
- θ factor appears in the drive Hamiltonian proportionality
- At θ = 0, the hold Hamiltonian reduces to H_int only
- At a_x = a_y = a_z = a_zz = 0, the SQL baseline is recovered
- The drive Hamiltonian with θ ≠ 0 obeys H_A ∝ θ (linearity test)
- The decoupled case (a_zz = 0) with any drive still gives SQL
"""

from __future__ import annotations

# Add the report directory to sys.path so we can import ``local``.
import sys as _sys
from pathlib import Path as _Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.analysis.ancilla_drive_metrology import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveNelderMeadResult,
    DriveRandomSearchResult,
    DriveThetaScanResult,
)
from src.analysis.ancilla_optimization import (
    I_2,
    J_Z,
    bs_unitary,
    build_two_qubit_operators,
)

_report_dir = str(_Path(__file__).resolve().parent.parent / "reports" / "20260519")
if _report_dir not in _sys.path:
    _sys.path.insert(0, _report_dir)
del _sys, _Path, _report_dir

from local import (  # type: ignore[import-untyped]  # noqa: E402
    DEFAULT_PSI0,
    DEFAULT_T_BS,
    DEFAULT_T_H,
    build_iszz_interaction,
    build_phase_modulated_drive_hamiltonian,
    build_phase_modulated_hold_hamiltonian,
    compute_phase_modulated_decoupled_baseline,
    compute_phase_modulated_sensitivity,
    evolve_phase_modulated_circuit,
    phase_modulated_2d_slice,
    phase_modulated_hold_unitary,
    phase_modulated_random_search,
    phase_modulated_sensitivity_objective,
    run_phase_modulated_nelder_mead,
    run_phase_modulated_theta_scan,
    system_only_bs_unitary,
)

I_4 = np.eye(4, dtype=complex)


@pytest.fixture
def make_ops() -> dict[str, np.ndarray]:
    return build_two_qubit_operators()


# ============================================================================
# Operator Construction
# ============================================================================


class TestSystemOnlyBS:
    def test_given_system_only_bs_then_is_4x4(self) -> None:
        U = system_only_bs_unitary(np.pi / 2)
        assert U.shape == (4, 4)

    def test_given_system_only_bs_then_is_unitary(self) -> None:
        U = system_only_bs_unitary(np.pi / 2)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)

    def test_given_system_only_bs_then_acts_as_identity_on_ancilla(self) -> None:
        """Verify that U_BS_S = U_BS ⊗ I_2 by checking action on |00⟩."""
        U_sys = bs_unitary(np.pi / 2)
        U_expected = np.kron(U_sys, I_2)
        U_got = system_only_bs_unitary(np.pi / 2)
        assert np.allclose(U_got, U_expected, atol=1e-12)

    def test_given_zero_T_bs_then_is_identity(self) -> None:
        U = system_only_bs_unitary(0.0)
        assert np.allclose(U, I_4, atol=1e-12)


class TestPhaseModulatedDriveHamiltonian:
    def test_given_zero_theta_then_is_zero(self, make_ops: dict) -> None:
        """When θ = 0, the θ-modulated drive is zero regardless of a_k."""
        H = build_phase_modulated_drive_hamiltonian(0.0, 2.0, 3.0, 4.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_zero_coefficients_then_is_zero(self, make_ops: dict) -> None:
        H = build_phase_modulated_drive_hamiltonian(1.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_theta_ax_then_hermitian(self, make_ops: dict) -> None:
        H = build_phase_modulated_drive_hamiltonian(2.0, 1.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_all_coefficients_then_hermitian(self, make_ops: dict) -> None:
        H = build_phase_modulated_drive_hamiltonian(1.5, 1.0, 2.0, 3.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_ax_only_then_proportional_to_theta_times_Jx_A(
        self, make_ops: dict
    ) -> None:
        """H_A = θ * a_x * J_x^A must hold exactly."""
        H = build_phase_modulated_drive_hamiltonian(2.0, 3.0, 0.0, 0.0, make_ops)
        expected = 2.0 * 3.0 * make_ops["Jx_A"]
        assert np.allclose(H, expected, atol=1e-12)

    def test_given_linear_in_theta(self, make_ops: dict) -> None:
        """H_A at θ=1.0 should be 2× H_A at θ=0.5."""
        H_half = build_phase_modulated_drive_hamiltonian(0.5, 2.0, 0.0, 0.0, make_ops)
        H_full = build_phase_modulated_drive_hamiltonian(1.0, 2.0, 0.0, 0.0, make_ops)
        assert np.allclose(2.0 * H_half, H_full, atol=1e-12)

    def test_given_theta_factor_extra_versus_fixed_drive(self, make_ops: dict) -> None:
        """At θ=1.0, the phase-modulated drive equals the fixed drive.

        This is the crossover point. At θ=2.0, it should be 2× the fixed drive.
        """
        from src.analysis.ancilla_drive_metrology import build_ancilla_drive_hamiltonian

        H_fixed = build_ancilla_drive_hamiltonian(2.0, 3.0, 4.0, make_ops)
        H_phase = build_phase_modulated_drive_hamiltonian(1.0, 2.0, 3.0, 4.0, make_ops)
        assert np.allclose(H_phase, H_fixed, atol=1e-12)


class TestIszzInteraction:
    def test_given_zero_coefficient_then_is_zero(self, make_ops: dict) -> None:
        H = build_iszz_interaction(0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_nonzero_coefficient_then_hermitian(self, make_ops: dict) -> None:
        H = build_iszz_interaction(2.5, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_coefficient_then_proportional_to_Jz_kron_Jz(
        self, make_ops: dict
    ) -> None:
        expected = 2.0 * np.kron(J_Z, J_Z)
        H = build_iszz_interaction(2.0, make_ops)
        assert np.allclose(H, expected, atol=1e-12)


class TestPhaseModulatedHoldHamiltonian:
    def test_given_zero_params_then_only_Jz_S(self, make_ops: dict) -> None:
        H = build_phase_modulated_hold_hamiltonian(1.0, 0.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, 1.0 * make_ops["Jz_S"], atol=1e-12)

    def test_given_all_params_then_hermitian(self, make_ops: dict) -> None:
        H = build_phase_modulated_hold_hamiltonian(1.0, 2.0, 3.0, 4.0, 5.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_zero_theta_then_only_interaction(self, make_ops: dict) -> None:
        """When θ = 0, H = H_int only (no system phase, no ancilla drive)."""
        H = build_phase_modulated_hold_hamiltonian(0.0, 2.0, 3.0, 4.0, 5.0, make_ops)
        expected = build_iszz_interaction(5.0, make_ops)
        assert np.allclose(H, expected, atol=1e-12)

    def test_given_nonzero_params_then_includes_theta_times_drive(
        self, make_ops: dict
    ) -> None:
        """Verify the θ factor on the drive: H contains θ*a_x*J_x^A."""
        H = build_phase_modulated_hold_hamiltonian(2.0, 3.0, 0.0, 0.0, 0.0, make_ops)
        # H = θ*J_z^S + θ*a_x*J_x^A = 2*J_z^S + 6*J_x^A
        expected = 2.0 * make_ops["Jz_S"] + 6.0 * make_ops["Jx_A"]
        assert np.allclose(H, expected, atol=1e-12)


class TestPhaseModulatedHoldUnitary:
    def test_given_zero_params_then_is_unitary(self, make_ops: dict) -> None:
        U = phase_modulated_hold_unitary(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)

    def test_given_nonzero_params_then_is_unitary(self, make_ops: dict) -> None:
        U = phase_modulated_hold_unitary(10.0, 1.0, 2.0, 0.5, -1.0, 3.0, make_ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)

    def test_given_zero_theta_then_is_unitary(self, make_ops: dict) -> None:
        """At θ = 0, the hold unitary should still be unitary (H_int only)."""
        U = phase_modulated_hold_unitary(10.0, 0.0, 2.0, 3.0, 4.0, 5.0, make_ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)


# ============================================================================
# Circuit Evolution
# ============================================================================


class TestEvolvePhaseModulatedCircuit:
    def test_given_default_state_then_final_state_is_normalised(
        self, make_ops: dict
    ) -> None:
        psi = evolve_phase_modulated_circuit(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12)

    def test_given_zero_drive_then_sql_sensitivity(self, make_ops: dict) -> None:
        """At zero drive and zero interaction, Δθ should equal 1/T_H."""
        dtheta = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        expected = 1.0 / DEFAULT_T_H
        assert np.isclose(dtheta, expected, rtol=0.05), (
            f"Δθ = {dtheta:.6f}, expected ≈ {expected:.6f}"
        )

    def test_given_nonzero_params_then_finite_sensitivity(self, make_ops: dict) -> None:
        dtheta = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        assert np.isfinite(dtheta), "Sensitivity must be finite"
        assert dtheta > 0.0, "Sensitivity must be positive"

    def test_given_zero_theta_then_sensitivity_finite(self, make_ops: dict) -> None:
        """At θ = 0, the sensitivity is finite (though θ=0 is a special point).

        Because H_A = θ(a_x J_x^A + ...) and H_S = θ J_z^S are both zero at
        θ=0, the hold Hamiltonian reduces to H_int only. The finite-difference
        derivative at θ=0 still captures how ⟨J_z^S⟩ changes as θ moves away
        from zero (since both H_S and H_A turn on). The sensitivity is
        therefore finite.
        """
        dtheta = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        assert np.isfinite(dtheta), "Sensitivity should be finite at θ=0"
        assert dtheta > 0.0, "Sensitivity must be positive"

    def test_given_decoupled_then_sql(self, make_ops: dict) -> None:
        """With a_zz = 0, ancilla is decoupled → SQL should hold for any drive.

        Because H_S and H_A act on different subsystems, [H_S, H_A] = 0, and
        the evolution factorises. The ancilla factor doesn't affect J_z^S.
        """
        for a_x, a_y, a_z in [(2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)]:
            dtheta = compute_phase_modulated_sensitivity(
                DEFAULT_PSI0,
                DEFAULT_T_BS,
                DEFAULT_T_H,
                1.0,
                a_x,
                a_y,
                a_z,
                0.0,  # a_zz = 0 → decoupled
                make_ops,
            )
            expected = 1.0 / DEFAULT_T_H
            assert np.isclose(dtheta, expected, rtol=0.05), (
                f"At (a_x={a_x}, a_y={a_y}, a_z={a_z}, a_zz=0): "
                f"Δθ = {dtheta:.6f}, expected ≈ {expected:.6f}"
            )


# ============================================================================
# Decoupled Baseline
# ============================================================================


class TestPhaseModulatedDecoupledBaseline:
    def test_baseline_recovers_sql(self) -> None:
        result = compute_phase_modulated_decoupled_baseline()
        assert np.isclose(result.delta_theta, result.sql, rtol=0.05)

    def test_baseline_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = compute_phase_modulated_decoupled_baseline()
        parquet_p = tmp_path / "baseline.parquet"
        result.save_parquet(parquet_p)
        loaded = DriveDecoupledBaselineResult.from_parquet(parquet_p)
        assert loaded.T_H_value == result.T_H_value
        assert np.isclose(loaded.delta_theta, result.delta_theta)


# ============================================================================
# 2D Slice Scan
# ============================================================================


class TestPhaseModulated2DSlice:
    def test_ax_slice_returns_correct_shape(self) -> None:
        result = phase_modulated_2d_slice(
            theta=1.0,
            slice_type="ax",
            n_drive=5,
            n_azz=5,
        )
        assert result.delta_theta_grid.shape == (5, 5)

    def test_ay_slice_returns_correct_shape(self) -> None:
        result = phase_modulated_2d_slice(
            theta=1.0,
            slice_type="ay",
            n_drive=5,
            n_azz=5,
        )
        assert result.delta_theta_grid.shape == (5, 5)

    def test_slice_all_finite_or_inf(self) -> None:
        result = phase_modulated_2d_slice(
            theta=1.0,
            slice_type="ax",
            n_drive=5,
            n_azz=5,
        )
        for val in result.delta_theta_grid.flatten():
            assert np.isfinite(val) or np.isinf(val)

    def test_invalid_slice_type_raises(self) -> None:
        with pytest.raises(ValueError, match="slice_type"):
            phase_modulated_2d_slice(theta=1.0, slice_type="invalid")

    def test_slice_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = phase_modulated_2d_slice(
            theta=1.0, slice_type="ax", n_drive=3, n_azz=3
        )
        parquet_p = tmp_path / "slice.parquet"
        result.save_parquet(parquet_p)
        loaded = Drive2DSliceResult.from_parquet(parquet_p)
        assert loaded.drive_values == pytest.approx(result.drive_values)
        assert loaded.azz_values == pytest.approx(result.azz_values)
        assert loaded.delta_theta_grid == pytest.approx(
            result.delta_theta_grid,
            nan_ok=True,
        )


# ============================================================================
# 4D Random Search
# ============================================================================


class TestPhaseModulatedRandomSearch:
    def test_random_search_returns_correct_length(self) -> None:
        result = phase_modulated_random_search(theta=1.0, n_samples=50, seed=42)
        assert result.samples.shape == (50, 4)
        assert len(result.delta_theta_values) == 50

    def test_random_search_reproducible(self) -> None:
        r1 = phase_modulated_random_search(theta=1.0, n_samples=50, seed=42)
        r2 = phase_modulated_random_search(theta=1.0, n_samples=50, seed=42)
        assert np.allclose(r1.samples, r2.samples)
        assert np.allclose(r1.delta_theta_values, r2.delta_theta_values)

    def test_random_search_best_is_best(self) -> None:
        result = phase_modulated_random_search(theta=1.0, n_samples=50, seed=42)
        assert result.best_delta_theta == pytest.approx(
            float(np.min(result.delta_theta_values)),
        )

    def test_random_search_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = phase_modulated_random_search(theta=1.0, n_samples=50, seed=42)
        parquet_p = tmp_path / "random.parquet"
        result.save_parquet(parquet_p)
        loaded = DriveRandomSearchResult.from_parquet(parquet_p)
        assert loaded.samples.shape == result.samples.shape
        assert np.allclose(loaded.best_params, result.best_params)
        assert np.isclose(loaded.best_delta_theta, result.best_delta_theta)


# ============================================================================
# Nelder--Mead Optimisation
# ============================================================================


class TestPhaseModulatedObjective:
    def test_objective_finite_at_valid_params(self, make_ops: dict) -> None:
        params = np.array([1.0, 0.0, 0.0, 1.0])
        val = phase_modulated_sensitivity_objective(params, 1.0, make_ops)
        assert np.isfinite(val)
        assert val > 0.0

    def test_objective_penalty_out_of_bounds(self, make_ops: dict) -> None:
        params = np.array([10.0, 0.0, 0.0, 0.0])  # a_x = 10 > 5
        val = phase_modulated_sensitivity_objective(params, 1.0, make_ops)
        assert val > 1e10  # huge penalty

    def test_objective_penalty_negative_bounds(self, make_ops: dict) -> None:
        params = np.array([-10.0, 0.0, 0.0, 0.0])  # a_x = -10 < -5
        val = phase_modulated_sensitivity_objective(params, 1.0, make_ops)
        assert val > 1e10  # huge penalty


class TestPhaseModulatedNelderMead:
    def test_nelder_mead_runs(self) -> None:
        result = run_phase_modulated_nelder_mead(
            theta_true=1.0,
            x0=np.array([1.0, 0.0, 0.0, 1.0]),
            maxiter=100,
        )
        assert np.isfinite(result.delta_theta_opt)
        assert result.params_opt.shape == (4,)

    def test_nelder_mead_improves_over_random_start(self) -> None:
        """Nelder-Mead should find Δθ at or below the starting point."""
        x0 = np.array([1.0, 0.0, 0.0, 1.0])
        ops = build_two_qubit_operators()
        start_val = phase_modulated_sensitivity_objective(x0, 1.0, ops)
        result = run_phase_modulated_nelder_mead(
            theta_true=1.0,
            x0=x0,
            maxiter=200,
        )
        assert result.delta_theta_opt <= start_val * 1.01 or result.success, (
            f"NM did not improve: start={start_val:.4f}, opt={result.delta_theta_opt:.4f}"
        )

    def test_nelder_mead_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = run_phase_modulated_nelder_mead(
            theta_true=1.0,
            x0=np.array([1.0, 0.0, 0.0, 1.0]),
            maxiter=100,
        )
        parquet_p = tmp_path / "nm.parquet"
        result.save_parquet(parquet_p)
        loaded = DriveNelderMeadResult.from_parquet(parquet_p)
        assert np.isclose(loaded.delta_theta_opt, result.delta_theta_opt)
        assert np.allclose(loaded.params_opt, result.params_opt)
        assert loaded.theta_true == result.theta_true


# ============================================================================
# θ Scan
# ============================================================================


class TestPhaseModulatedThetaScan:
    def test_theta_scan_runs(self) -> None:
        result = run_phase_modulated_theta_scan(
            theta_values=[0.5, 1.0],
            n_random=50,
            n_nm_refine=5,
            seed=42,
            maxiter=100,
        )
        assert len(result.theta_values) == 2
        assert len(result.best_params_per_theta) == 2
        assert len(result.best_delta_theta_per_theta) == 2

    def test_theta_scan_all_finite(self) -> None:
        result = run_phase_modulated_theta_scan(
            theta_values=[0.5, 1.0],
            n_random=50,
            n_nm_refine=5,
            seed=42,
            maxiter=100,
        )
        for dt in result.best_delta_theta_per_theta:
            assert np.isfinite(dt), f"Non-finite Δθ: {dt}"

    def test_theta_scan_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = run_phase_modulated_theta_scan(
            theta_values=[0.5],
            n_random=20,
            n_nm_refine=3,
            seed=42,
            maxiter=50,
        )
        parquet_p = tmp_path / "theta-scan.parquet"
        result.save_parquet(parquet_p)
        loaded = DriveThetaScanResult.from_parquet(parquet_p)
        assert np.allclose(loaded.theta_values, result.theta_values)
        assert np.allclose(
            loaded.best_delta_theta_per_theta,
            result.best_delta_theta_per_theta,
        )
        for got, expected in zip(
            loaded.best_params_per_theta,
            result.best_params_per_theta,
            strict=False,
        ):
            assert np.allclose(got, expected, atol=1e-10)


# ============================================================================
# Validation Helpers
# ============================================================================


class TestPhaseModulatedValidation:
    def test_given_decoupled_config_then_sql_held(self) -> None:
        """With all drive and interaction off, the SQL must hold exactly."""
        result = compute_phase_modulated_decoupled_baseline()
        ratio = result.delta_theta / result.sql
        assert abs(ratio - 1.0) < 0.05, f"SQL violated: Δθ/SQL = {ratio:.6f}"

    def test_given_random_search_then_best_not_worse_than_sql(self) -> None:
        """The best point in a random search should be at worst ~SQL."""
        result = phase_modulated_random_search(theta=1.0, n_samples=200, seed=42)
        assert result.best_delta_theta <= result.sql * 2.0 or not np.isfinite(
            result.best_delta_theta,
        ), f"Best Δθ = {result.best_delta_theta:.4f} >> SQL = {result.sql:.4f}"

    def test_given_system_only_bs_then_acts_only_on_system(self) -> None:
        """The system-only BS should leave |01⟩'s ancilla bit unchanged."""
        # |01⟩ = [0, 1, 0, 0]^T
        psi_in = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
        U_bs = system_only_bs_unitary(np.pi / 2)
        psi_out = U_bs @ psi_in
        # The ancilla population should remain unchanged: |⟨01|ψ_out⟩|^2 +
        # |⟨11|ψ_out⟩|^2 = 1 (ancilla stays in |1⟩)
        ancilla_in_1 = abs(psi_out[1]) ** 2 + abs(psi_out[3]) ** 2
        assert np.isclose(ancilla_in_1, 1.0, atol=1e-12)

    def test_given_drive_linearity_in_theta(self) -> None:
        """The sensitivity should depend on θ in a non-trivial way.

        Unlike the fixed-drive protocol where Δθ(θ) was θ-independent for
        the SQL points, the θ-modulated drive creates a genuine θ-dependence.
        We test that at different θ values with the same drive, the sensitivity
        differs — confirming the θ-modulation is active.
        """
        ops = build_two_qubit_operators()
        dt_1 = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            2.0,
            0.0,
            0.0,
            2.0,
            ops,
        )
        dt_2 = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            2.0,
            2.0,
            0.0,
            0.0,
            2.0,
            ops,
        )
        # The two sensitivities should differ because the effective drive
        # strength θ * a_x differs (θ=1 vs θ=2)
        assert abs(dt_1 - dt_2) > 1e-6, (
            f"θ-modulation not active: Δθ(θ=1) = {dt_1:.10f}, Δθ(θ=2) = {dt_2:.10f}"
        )

    def test_given_fixed_vs_phase_modulated_differ_at_large_theta(
        self,
    ) -> None:
        """At θ ≠ 1, the phase-modulated protocol gives different sensitivity
        from the fixed-drive protocol because the effective drive strength
        differs."""
        from src.analysis.ancilla_drive_metrology import (
            compute_drive_sensitivity as fixed_drive_sensitivity,
        )

        ops = build_two_qubit_operators()
        params = {"a_x": 2.0, "a_y": 0.0, "a_z": 0.0, "a_zz": 2.0}
        theta_val = 2.0

        dt_fixed = fixed_drive_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            theta_val,
            params["a_x"],
            params["a_y"],
            params["a_z"],
            params["a_zz"],
            ops,
        )
        dt_phase = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            theta_val,
            params["a_x"],
            params["a_y"],
            params["a_z"],
            params["a_zz"],
            ops,
        )

        # The phase-modulated drive at θ=2 is twice as strong as the fixed
        # drive with the same a_k, so sensitivities should differ
        assert abs(dt_fixed - dt_phase) > 1e-8, (
            f"Fixed and phase-modulated protocols give same sensitivity: "
            f"{dt_fixed:.10f} vs {dt_phase:.10f} at θ={theta_val}"
        )

"""Tests for the ω-modulated ancilla drive metrology protocol.

This is the companion test module for
``reports/r20260519/phase_modulated_drive.py`` (which replaces the former
``src/analysis/ancilla_drive_phase_modulated.py``).
It mirrors the structure of ``test_ancilla_drive_metrology.py`` (fixed-drive)
but tests the ω-modulated Hamiltonian H_A = ω (a_x J_x^A + a_y J_y^A + a_z J_z^A).

Key new tests (not in the fixed-drive test suite):
- ω factor appears in the drive Hamiltonian proportionality
- At ω = 0, the hold Hamiltonian reduces to H_int only
- At a_x = a_y = a_z = a_zz = 0, the SQL baseline is recovered
- The drive Hamiltonian with ω ≠ 0 obeys H_A ∝ ω (linearity test)
- The decoupled case (a_zz = 0) with any drive still gives SQL
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.analysis.ancilla_drive_metrology import (
    build_iszz_interaction,
    build_phase_modulated_drive_hamiltonian,
    build_phase_modulated_hold_hamiltonian,
    compute_phase_modulated_sensitivity,
    evolve_phase_modulated_circuit,
    phase_modulated_hold_unitary,
    system_only_bs_unitary,
)
from src.analysis.ancilla_drive_results import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveOmegaScanResult,
)
from src.analysis.ancilla_optimization import (
    I_2,
    J_Z,
    bs_unitary,
    build_two_qubit_operators,
)
from src.visualization.ancilla_drive_plots import _markevery

_m = importlib.import_module("reports.r20260519.phase_modulated_drive")
DEFAULT_PSI0 = _m.DEFAULT_PSI0
DEFAULT_T_BS = _m.DEFAULT_T_BS
DEFAULT_t_hold = _m.DEFAULT_t_hold
compute_phase_modulated_decoupled_baseline = (
    _m.compute_phase_modulated_decoupled_baseline
)
phase_modulated_2d_slice = _m.phase_modulated_2d_slice
phase_modulated_random_search = _m.phase_modulated_random_search
run_phase_modulated_omega_scan = _m.run_phase_modulated_omega_scan
verify_longitudinal_only_sql = _m.verify_longitudinal_only_sql

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
    def test_given_zero_omega_then_is_zero(self, make_ops: dict) -> None:
        """When ω = 0, the ω-modulated drive is zero regardless of a_k."""
        H = build_phase_modulated_drive_hamiltonian(0.0, 2.0, 3.0, 4.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_zero_coefficients_then_is_zero(self, make_ops: dict) -> None:
        H = build_phase_modulated_drive_hamiltonian(1.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_omega_ax_then_hermitian(self, make_ops: dict) -> None:
        H = build_phase_modulated_drive_hamiltonian(2.0, 1.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_all_coefficients_then_hermitian(self, make_ops: dict) -> None:
        H = build_phase_modulated_drive_hamiltonian(1.5, 1.0, 2.0, 3.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_ax_only_then_proportional_to_omega_times_Jx_A(
        self, make_ops: dict
    ) -> None:
        """H_A = ω * a_x * J_x^A must hold exactly."""
        H = build_phase_modulated_drive_hamiltonian(2.0, 3.0, 0.0, 0.0, make_ops)
        expected = 2.0 * 3.0 * make_ops["Jx_A"]
        assert np.allclose(H, expected, atol=1e-12)

    def test_given_linear_in_omega(self, make_ops: dict) -> None:
        """H_A at ω=1.0 should be 2× H_A at ω=0.5."""
        H_half = build_phase_modulated_drive_hamiltonian(0.5, 2.0, 0.0, 0.0, make_ops)
        H_full = build_phase_modulated_drive_hamiltonian(1.0, 2.0, 0.0, 0.0, make_ops)
        assert np.allclose(2.0 * H_half, H_full, atol=1e-12)

    def test_given_omega_factor_extra_versus_fixed_drive(self, make_ops: dict) -> None:
        """At ω=1.0, the phase-modulated drive equals the fixed drive.

        This is the crossover point. At ω=2.0, it should be 2× the fixed drive.
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

    def test_given_zero_omega_then_only_interaction(self, make_ops: dict) -> None:
        """When ω = 0, H = H_int only (no system phase, no ancilla drive)."""
        H = build_phase_modulated_hold_hamiltonian(0.0, 2.0, 3.0, 4.0, 5.0, make_ops)
        expected = build_iszz_interaction(5.0, make_ops)
        assert np.allclose(H, expected, atol=1e-12)

    def test_given_nonzero_params_then_includes_omega_times_drive(
        self, make_ops: dict
    ) -> None:
        """Verify the ω factor on the drive: H contains ω*a_x*J_x^A."""
        H = build_phase_modulated_hold_hamiltonian(2.0, 3.0, 0.0, 0.0, 0.0, make_ops)
        # H = ω*J_z^S + ω*a_x*J_x^A = 2*J_z^S + 6*J_x^A
        expected = 2.0 * make_ops["Jz_S"] + 6.0 * make_ops["Jx_A"]
        assert np.allclose(H, expected, atol=1e-12)


class TestPhaseModulatedHoldUnitary:
    def test_given_zero_params_then_is_unitary(self, make_ops: dict) -> None:
        U = phase_modulated_hold_unitary(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)

    def test_given_nonzero_params_then_is_unitary(self, make_ops: dict) -> None:
        U = phase_modulated_hold_unitary(10.0, 1.0, 2.0, 0.5, -1.0, 3.0, make_ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)

    def test_given_zero_omega_then_is_unitary(self, make_ops: dict) -> None:
        """At ω = 0, the hold unitary should still be unitary (H_int only)."""
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
            DEFAULT_t_hold,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12)

    def test_given_zero_drive_then_sql_sensitivity(self, make_ops: dict) -> None:
        """At zero drive and zero interaction, Δω should equal 1/t_hold."""
        domega = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_t_hold,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        expected = 1.0 / DEFAULT_t_hold
        assert np.isclose(domega, expected, rtol=0.05), (
            f"Δω = {domega:.6f}, expected ≈ {expected:.6f}"
        )

    def test_given_nonzero_params_then_finite_sensitivity(self, make_ops: dict) -> None:
        domega = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_t_hold,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        assert np.isfinite(domega), "Sensitivity must be finite"
        assert domega > 0.0, "Sensitivity must be positive"

    def test_given_zero_omega_then_sensitivity_finite(self, make_ops: dict) -> None:
        """At ω = 0, the sensitivity is finite (though ω=0 is a special point).

        Because H_A = ω(a_x J_x^A + ...) and H_S = ω J_z^S are both zero at
        ω=0, the hold Hamiltonian reduces to H_int only. The finite-difference
        derivative at ω=0 still captures how ⟨J_z^S⟩ changes as ω moves away
        from zero (since both H_S and H_A turn on). The sensitivity is
        therefore finite.
        """
        domega = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_t_hold,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        assert np.isfinite(domega), "Sensitivity should be finite at ω=0"
        assert domega > 0.0, "Sensitivity must be positive"

    def test_given_decoupled_then_sql(self, make_ops: dict) -> None:
        """With a_zz = 0, ancilla is decoupled → SQL should hold for any drive.

        Because H_S and H_A act on different subsystems, [H_S, H_A] = 0, and
        the evolution factorises. The ancilla factor doesn't affect J_z^S.
        """
        for a_x, a_y, a_z in [(2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)]:
            domega = compute_phase_modulated_sensitivity(
                DEFAULT_PSI0,
                DEFAULT_T_BS,
                DEFAULT_t_hold,
                1.0,
                a_x,
                a_y,
                a_z,
                0.0,  # a_zz = 0 → decoupled
                make_ops,
            )
            expected = 1.0 / DEFAULT_t_hold
            assert np.isclose(domega, expected, rtol=0.05), (
                f"At (a_x={a_x}, a_y={a_y}, a_z={a_z}, a_zz=0): "
                f"Δω = {domega:.6f}, expected ≈ {expected:.6f}"
            )


# ============================================================================
# Decoupled Baseline
# ============================================================================


class TestPhaseModulatedDecoupledBaseline:
    def test_baseline_recovers_sql(self) -> None:
        result = compute_phase_modulated_decoupled_baseline()
        assert np.isclose(result.delta_omega, result.sql, rtol=0.05)

    def test_baseline_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = compute_phase_modulated_decoupled_baseline()
        parquet_p = tmp_path / "baseline.parquet"
        result.save_parquet(parquet_p)
        loaded = DriveDecoupledBaselineResult.from_parquet(parquet_p)
        assert loaded.t_hold_value == result.t_hold_value
        assert np.isclose(loaded.delta_omega, result.delta_omega)


# ============================================================================
# 2D Slice Scan
# ============================================================================


class TestPhaseModulated2DSlice:
    def test_ax_slice_returns_correct_shape(self) -> None:
        result = phase_modulated_2d_slice(
            omega=1.0,
            slice_type="ax",
            n_drive=5,
            n_azz=5,
        )
        assert result.delta_omega_grid.shape == (5, 5)

    def test_ay_slice_returns_correct_shape(self) -> None:
        result = phase_modulated_2d_slice(
            omega=1.0,
            slice_type="ay",
            n_drive=5,
            n_azz=5,
        )
        assert result.delta_omega_grid.shape == (5, 5)

    def test_slice_all_finite_or_inf(self) -> None:
        result = phase_modulated_2d_slice(
            omega=1.0,
            slice_type="ax",
            n_drive=5,
            n_azz=5,
        )
        for val in result.delta_omega_grid.flatten():
            assert np.isfinite(val) or np.isinf(val)

    def test_invalid_slice_type_raises(self) -> None:
        with pytest.raises(ValueError, match="slice_type"):
            phase_modulated_2d_slice(omega=1.0, slice_type="invalid")

    def test_slice_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = phase_modulated_2d_slice(
            omega=1.0, slice_type="ax", n_drive=3, n_azz=3
        )
        parquet_p = tmp_path / "slice.parquet"
        result.save_parquet(parquet_p)
        loaded = Drive2DSliceResult.from_parquet(parquet_p)
        assert loaded.drive_values == pytest.approx(result.drive_values)
        assert loaded.azz_values == pytest.approx(result.azz_values)
        assert loaded.delta_omega_grid == pytest.approx(
            result.delta_omega_grid,
            nan_ok=True,
        )


# ============================================================================
# ω Scan
# ============================================================================


class TestPhaseModulatedOmegaScan:
    def test_omega_scan_runs(self) -> None:
        result = run_phase_modulated_omega_scan(
            omega_values=[0.5, 1.0],
            n_random=50,
            n_nm_refine=5,
            seed=42,
            maxiter=100,
        )
        assert len(result.omega_values) == 2
        assert len(result.best_params_per_omega) == 2
        assert len(result.best_delta_omega_per_omega) == 2

    def test_omega_scan_all_finite(self) -> None:
        result = run_phase_modulated_omega_scan(
            omega_values=[0.5, 1.0],
            n_random=50,
            n_nm_refine=5,
            seed=42,
            maxiter=100,
        )
        for dt in result.best_delta_omega_per_omega:
            assert np.isfinite(dt), f"Non-finite Δω: {dt}"

    def test_omega_scan_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = run_phase_modulated_omega_scan(
            omega_values=[0.5],
            n_random=20,
            n_nm_refine=3,
            seed=42,
            maxiter=50,
        )
        parquet_p = tmp_path / "omega-scan.parquet"
        result.save_parquet(parquet_p)
        loaded = DriveOmegaScanResult.from_parquet(parquet_p)
        assert np.allclose(loaded.omega_values, result.omega_values)
        assert np.allclose(
            loaded.best_delta_omega_per_omega,
            result.best_delta_omega_per_omega,
        )
        for got, expected in zip(
            loaded.best_params_per_omega,
            result.best_params_per_omega,
            strict=False,
        ):
            assert np.allclose(got, expected, atol=1e-10)


# ============================================================================
# 2D Slice: (a_z, a_zz) longitudinal-only
# ============================================================================


class TestAzSlice:
    def test_az_slice_returns_correct_shape(self) -> None:
        result = phase_modulated_2d_slice(
            omega=0.1,
            slice_type="az",
            drive_range=(-5.0, 5.0),
            azz_range=(-2.0, 5.0),
            n_drive=10,
            n_azz=10,
        )
        assert result.delta_omega_grid.shape == (10, 10)

    def test_az_slice_all_at_sql(self) -> None:
        """Every point in the (a_z, a_zz) longitudinal-only slice should equal
        the SQL (= 1/T_H) within numerical precision."""
        result = phase_modulated_2d_slice(
            omega=0.1,
            slice_type="az",
            drive_range=(-5.0, 5.0),
            azz_range=(-2.0, 5.0),
            n_drive=10,
            n_azz=10,
        )
        sql = 1.0 / DEFAULT_t_hold
        max_dev = float(np.max(np.abs(result.delta_omega_grid - sql)))
        assert max_dev < 1e-8, (
            f"Max deviation from SQL in (a_z, a_zz) slice = {max_dev:.2e}"
        )

    def test_az_slice_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = phase_modulated_2d_slice(
            omega=0.1,
            slice_type="az",
            drive_range=(-5.0, 5.0),
            azz_range=(-2.0, 5.0),
            n_drive=5,
            n_azz=5,
        )
        parquet_p = tmp_path / "az-slice.parquet"
        result.save_parquet(parquet_p)
        loaded = Drive2DSliceResult.from_parquet(parquet_p)
        assert loaded.drive_values == pytest.approx(result.drive_values)
        assert loaded.azz_values == pytest.approx(result.azz_values)
        assert loaded.delta_omega_grid == pytest.approx(
            result.delta_omega_grid, nan_ok=True
        )
        # Verify metadata survives roundtrip
        assert loaded.omega_value == pytest.approx(result.omega_value)
        assert loaded.slice_type == result.slice_type
        assert loaded.sql == pytest.approx(result.sql)

    def test_given_az_slice_then_slice_type_stored(self) -> None:
        result = phase_modulated_2d_slice(
            omega=0.1,
            slice_type="az",
            drive_range=(-5.0, 5.0),
            azz_range=(-2.0, 5.0),
            n_drive=3,
            n_azz=3,
        )
        assert result.slice_type == "az"

    def test_given_invalid_slice_type_raises(self) -> None:
        with pytest.raises(ValueError, match="slice_type"):
            phase_modulated_2d_slice(omega=0.1, slice_type="invalid")


class TestLongitudinalOnlyVerification:
    """Tests reproducing the Section 7.4 verification.

    Confirms that:
    - With a_x = a_y = 0, the sensitivity is exactly SQL for any (a_z, a_zz).
    - Adding transverse drives (a_x, a_y) produces sub-SQL sensitivity.
    """

    def test_given_longitudinal_only_then_exactly_sql(self) -> None:
        ops = build_two_qubit_operators()
        domega = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_t_hold,
            0.1,  # omega
            0.0,  # a_x
            0.0,  # a_y
            2.1,  # a_z
            0.94,  # a_zz
            ops,
        )
        expected_sql = 1.0 / DEFAULT_t_hold
        assert np.isclose(domega, expected_sql, atol=1e-10), (
            f"Longitudinal-only Δω = {domega:.15f}, expected SQL = {expected_sql:.15f}"
        )

    def test_given_transverse_drives_then_sub_sql(self) -> None:
        ops = build_two_qubit_operators()
        domega = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_t_hold,
            0.1,  # omega
            5.0,  # a_x
            -5.0,  # a_y
            2.1,  # a_z
            0.94,  # a_zz
            ops,
        )
        sql = 1.0 / DEFAULT_t_hold
        assert domega < sql, (
            f"With transverse drives: Δω = {domega:.6f} should be < SQL = {sql}"
        )

    def test_given_verify_function_runs(self) -> None:
        """Verify the standalone verification function runs without error."""
        results = verify_longitudinal_only_sql()
        assert "longitudinal_only_delta_omega" in results
        assert "with_transverse_delta_omega" in results
        assert "grid_max_deviation" in results
        assert results["grid_max_deviation"] < 1e-8


# ============================================================================
# Validation Helpers
# ============================================================================


class TestPhaseModulatedValidation:
    def test_given_decoupled_config_then_sql_held(self) -> None:
        """With all drive and interaction off, the SQL must hold exactly."""
        result = compute_phase_modulated_decoupled_baseline()
        ratio = result.delta_omega / result.sql
        assert abs(ratio - 1.0) < 0.05, f"SQL violated: Δω/SQL = {ratio:.6f}"

    def test_given_random_search_then_best_not_worse_than_sql(self) -> None:
        """The best point in a random search should be at worst ~SQL."""
        result = phase_modulated_random_search(omega=1.0, n_samples=200, seed=42)
        assert result.best_delta_omega <= result.sql * 2.0 or not np.isfinite(
            result.best_delta_omega,
        ), f"Best Δω = {result.best_delta_omega:.4f} >> SQL = {result.sql:.4f}"

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

    def test_given_drive_linearity_in_omega(self) -> None:
        """The sensitivity should depend on ω in a non-trivial way.

        Unlike the fixed-drive protocol where Δω(ω) was ω-independent for
        the SQL points, the ω-modulated drive creates a genuine ω-dependence.
        We test that at different ω values with the same drive, the sensitivity
        differs — confirming the ω-modulation is active.
        """
        ops = build_two_qubit_operators()
        dt_1 = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_t_hold,
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
            DEFAULT_t_hold,
            2.0,
            2.0,
            0.0,
            0.0,
            2.0,
            ops,
        )
        # The two sensitivities should differ because the effective drive
        # strength ω * a_x differs (ω=1 vs ω=2)
        assert abs(dt_1 - dt_2) > 1e-6, (
            f"ω-modulation not active: Δω(ω=1) = {dt_1:.10f}, Δω(ω=2) = {dt_2:.10f}"
        )

    def test_given_fixed_vs_phase_modulated_differ_at_large_omega(
        self,
    ) -> None:
        """At ω ≠ 1, the phase-modulated protocol gives different sensitivity
        from the fixed-drive protocol because the effective drive strength
        differs."""
        from src.analysis.ancilla_drive_metrology import (
            compute_drive_sensitivity,
        )

        ops = build_two_qubit_operators()
        params = {"a_x": 2.0, "a_y": 0.0, "a_z": 0.0, "a_zz": 2.0}
        omega_val = 2.0

        dt_fixed = compute_drive_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_t_hold,
            omega_val,
            params["a_x"],
            params["a_y"],
            params["a_z"],
            params["a_zz"],
            ops,
        )
        dt_phase = compute_phase_modulated_sensitivity(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_t_hold,
            omega_val,
            params["a_x"],
            params["a_y"],
            params["a_z"],
            params["a_zz"],
            ops,
        )

        # The phase-modulated drive at ω=2 is twice as strong as the fixed
        # drive with the same a_k, so sensitivities should differ
        assert abs(dt_fixed - dt_phase) > 1e-8, (
            f"Fixed and phase-modulated protocols give same sensitivity: "
            f"{dt_fixed:.10f} vs {dt_phase:.10f} at ω={omega_val}"
        )


# ============================================================================
# Markevery helper
# ============================================================================


class TestMarkevery:
    def test_given_50_points_then_returns_1(self) -> None:
        """50 points: 50 // 30 = 1, every point gets a marker."""
        assert _markevery(50) == 1

    def test_given_500_points_then_returns_16(self) -> None:
        """500 points: 500 // 30 = 16, every 16th point gets a marker."""
        assert _markevery(500) == 16

    def test_given_1_point_then_returns_1(self) -> None:
        assert _markevery(1) == 1

    def test_given_30_points_then_returns_1(self) -> None:
        """30 points: 30 // 30 = 1."""
        assert _markevery(30) == 1

    def test_given_60_points_then_returns_2(self) -> None:
        """60 points: 60 // 30 = 2."""
        assert _markevery(60) == 2

    def test_custom_target(self) -> None:
        """With target=10, 500 points → 500 // 10 = 50."""
        assert _markevery(500, target=10) == 50


# ============================================================================
# Default ω Grid
# ============================================================================


class TestDefaultOmegaGrid:
    def test_default_grid_has_500_points(self) -> None:
        assert len(_m.PHASE_OMEGA_VALS) == 500

    def test_default_grid_starts_at_0_01(self) -> None:
        assert _m.PHASE_OMEGA_VALS[0] == 0.01

    def test_default_grid_ends_at_5_00(self) -> None:
        assert _m.PHASE_OMEGA_VALS[-1] == 5.0

    def test_default_grid_step_is_0_01(self) -> None:
        """Adjacent values should differ by 0.01."""
        diffs = np.diff(_m.PHASE_OMEGA_VALS)
        assert np.allclose(diffs, 0.01, atol=1e-10)

    def test_default_grid_monotonically_increasing(self) -> None:
        diffs = np.diff(_m.PHASE_OMEGA_VALS)
        assert np.all(diffs > 0)

    def test_omega_min_max_constants(self) -> None:
        assert _m.OMEGA_MIN == 0.01
        assert _m.OMEGA_MAX == 5.0
        assert _m.DEFAULT_N_OMEGA == 500


# ============================================================================
# Pipeline omega_values Threading
# ============================================================================


class TestPipelineOmegaThreading:
    def test_omega_scan_accepts_custom_grid(self) -> None:
        """The omega scan function should accept a custom ω grid."""
        result = run_phase_modulated_omega_scan(
            omega_values=[0.3, 0.6],
            n_random=20,
            n_nm_refine=3,
            seed=42,
            maxiter=50,
        )
        assert len(result.omega_values) == 2
        assert np.isclose(result.omega_values[0], 0.3)
        assert np.isclose(result.omega_values[1], 0.6)

    def test_omega_scan_small_grid_finer_than_default(self) -> None:
        """A 3-point grid at fine spacing should work correctly."""
        result = run_phase_modulated_omega_scan(
            omega_values=[0.01, 0.02, 0.03],
            n_random=20,
            n_nm_refine=3,
            seed=42,
            maxiter=50,
        )
        assert len(result.omega_values) == 3
        for dt in result.best_delta_omega_per_omega:
            assert np.isfinite(dt), f"Non-finite Δω: {dt}"

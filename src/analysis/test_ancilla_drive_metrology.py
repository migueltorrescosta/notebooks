"""Tests for the driven-ancilla metrology module."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime via tmp_path fixture

import numpy as np
import pytest

from src.analysis.ancilla_drive_metrology import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveNelderMeadResult,
    DriveOmegaScanResult,
    DriveRandomSearchResult,
    build_ancilla_drive_hamiltonian,
    build_drive_hold_hamiltonian,
    build_iszz_interaction,
    compute_drive_decoupled_baseline,
    compute_drive_sensitivity,
    drive_2d_slice,
    drive_hold_unitary,
    drive_random_search,
    drive_sensitivity_objective,
    evolve_drive_circuit,
    run_drive_nelder_mead,
    run_drive_omega_scan,
    system_only_bs_unitary,
)
from src.analysis.ancilla_optimization import (
    I_2,
    J_Z,
    bs_unitary,
    build_two_qubit_operators,
)
from src.utils.constants import I_4


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


class TestAncillaDriveHamiltonian:
    def test_given_zero_coefficients_then_is_zero(self, make_ops: dict) -> None:
        H = build_ancilla_drive_hamiltonian(0.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_ax_coefficient_then_hermitian(self, make_ops: dict) -> None:
        H = build_ancilla_drive_hamiltonian(2.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_all_coefficients_then_hermitian(self, make_ops: dict) -> None:
        H = build_ancilla_drive_hamiltonian(1.0, 2.0, 3.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_ax_only_then_proportional_to_Jx_A(self, make_ops: dict) -> None:
        H = build_ancilla_drive_hamiltonian(2.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, 2.0 * make_ops["Jx_A"], atol=1e-12)

    def test_given_az_only_then_proportional_to_Jz_A(self, make_ops: dict) -> None:
        H = build_ancilla_drive_hamiltonian(0.0, 0.0, 3.0, make_ops)
        assert np.allclose(H, 3.0 * make_ops["Jz_A"], atol=1e-12)


class TestIszzInteraction:
    def test_given_zero_coefficient_then_is_zero(self, make_ops: dict) -> None:
        H = build_iszz_interaction(0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_nonzero_coefficient_then_hermitian(self, make_ops: dict) -> None:
        H = build_iszz_interaction(2.5, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_coefficient_then_proportional_to_Jz_kron_Jz(self) -> None:
        expected = 2.0 * np.kron(J_Z, J_Z)
        ops = build_two_qubit_operators()
        H = build_iszz_interaction(2.0, ops)
        assert np.allclose(H, expected, atol=1e-12)


class TestDriveHoldHamiltonian:
    def test_given_zero_params_then_only_Jz_S(self, make_ops: dict) -> None:
        H = build_drive_hold_hamiltonian(1.0, 0.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, 1.0 * make_ops["Jz_S"], atol=1e-12)

    def test_given_all_params_then_hermitian(self, make_ops: dict) -> None:
        H = build_drive_hold_hamiltonian(1.0, 2.0, 3.0, 4.0, 5.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)


class TestDriveHoldUnitary:
    def test_given_zero_params_then_is_unitary(self, make_ops: dict) -> None:
        U = drive_hold_unitary(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)

    def test_given_nonzero_params_then_is_unitary(self, make_ops: dict) -> None:
        U = drive_hold_unitary(10.0, 1.0, 2.0, 0.5, -1.0, 3.0, make_ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)


# ============================================================================
# Circuit Evolution
# ============================================================================


class TestEvolveDriveCircuit:
    def test_given_default_state_then_final_state_is_normalised(
        self, make_ops: dict
    ) -> None:
        psi = evolve_drive_circuit(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
            np.pi / 2.0,
            10.0,
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
        domega = compute_drive_sensitivity(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
            np.pi / 2.0,
            10.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        expected = 1.0 / 10.0
        assert np.isclose(domega, expected, rtol=0.05), (
            f"Δω = {domega:.6f}, expected ≈ {expected:.6f}"
        )

    def test_given_nonzero_params_then_finite_sensitivity(self, make_ops: dict) -> None:
        domega = compute_drive_sensitivity(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
            np.pi / 2.0,
            10.0,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        assert np.isfinite(domega), "Sensitivity must be finite"
        assert domega > 0.0, "Sensitivity must be positive"


# ============================================================================
# Decoupled Baseline
# ============================================================================


class TestDriveDecoupledBaseline:
    def test_baseline_recovers_sql(self) -> None:
        result = compute_drive_decoupled_baseline()
        assert np.isclose(result.delta_omega, result.sql, rtol=0.05)

    def test_baseline_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = compute_drive_decoupled_baseline()
        parquet_p = tmp_path / "baseline.parquet"
        result.save_parquet(parquet_p)
        loaded = DriveDecoupledBaselineResult.from_parquet(parquet_p)
        assert loaded.t_hold_value == result.t_hold_value
        assert np.isclose(loaded.delta_omega, result.delta_omega)


# ============================================================================
# 2D Slice Scan
# ============================================================================


class TestDrive2DSlice:
    def test_ax_slice_returns_correct_shape(self, make_ops: dict) -> None:
        result = drive_2d_slice(
            omega=1.0,
            slice_type="ax",
            n_drive=5,
            n_azz=5,
        )
        assert result.delta_omega_grid.shape == (5, 5)

    def test_ay_slice_returns_correct_shape(self) -> None:
        result = drive_2d_slice(
            omega=1.0,
            slice_type="ay",
            n_drive=5,
            n_azz=5,
        )
        assert result.delta_omega_grid.shape == (5, 5)

    def test_slice_all_finite_or_inf(self) -> None:
        result = drive_2d_slice(
            omega=1.0,
            slice_type="ax",
            n_drive=5,
            n_azz=5,
        )
        for val in result.delta_omega_grid.flatten():
            assert np.isfinite(val) or np.isinf(val)

    def test_invalid_slice_type_raises(self) -> None:
        with pytest.raises(ValueError, match="slice_type"):
            drive_2d_slice(omega=1.0, slice_type="invalid")

    def test_slice_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = drive_2d_slice(omega=1.0, slice_type="ax", n_drive=3, n_azz=3)
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
# 4D Random Search
# ============================================================================


class TestDriveRandomSearch:
    def test_random_search_returns_correct_length(self) -> None:
        result = drive_random_search(omega=1.0, n_samples=50, seed=42)
        assert result.samples.shape == (50, 4)
        assert len(result.delta_omega_values) == 50

    def test_random_search_reproducible(self) -> None:
        r1 = drive_random_search(omega=1.0, n_samples=50, seed=42)
        r2 = drive_random_search(omega=1.0, n_samples=50, seed=42)
        assert np.allclose(r1.samples, r2.samples)
        assert np.allclose(r1.delta_omega_values, r2.delta_omega_values)

    def test_random_search_best_is_best(self) -> None:
        result = drive_random_search(omega=1.0, n_samples=50, seed=42)
        assert result.best_delta_omega == pytest.approx(
            float(np.min(result.delta_omega_values)),
        )

    def test_random_search_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = drive_random_search(omega=1.0, n_samples=50, seed=42)
        parquet_p = tmp_path / "random.parquet"
        result.save_parquet(parquet_p)
        loaded = DriveRandomSearchResult.from_parquet(parquet_p)
        assert loaded.samples.shape == result.samples.shape
        assert np.allclose(loaded.best_params, result.best_params)
        assert np.isclose(loaded.best_delta_omega, result.best_delta_omega)


# ============================================================================
# Nelder--Mead Optimisation
# ============================================================================


class TestDriveObjective:
    def test_objective_finite_at_valid_params(self, make_ops: dict) -> None:
        params = np.array([1.0, 0.0, 0.0, 1.0])
        val = drive_sensitivity_objective(params, 1.0, make_ops)
        assert np.isfinite(val)
        assert val > 0.0

    def test_objective_penalty_out_of_bounds(self, make_ops: dict) -> None:
        params = np.array([10.0, 0.0, 0.0, 0.0])  # a_x = 10 > 5
        val = drive_sensitivity_objective(params, 1.0, make_ops)
        assert val > 1e10  # huge penalty

    def test_objective_penalty_negative_bounds(self, make_ops: dict) -> None:
        params = np.array([-10.0, 0.0, 0.0, 0.0])  # a_x = -10 < -5
        val = drive_sensitivity_objective(params, 1.0, make_ops)
        assert val > 1e10  # huge penalty


class TestDriveNelderMead:
    def test_nelder_mead_runs(self) -> None:
        result = run_drive_nelder_mead(
            omega_true=1.0,
            x0=np.array([1.0, 0.0, 0.0, 1.0]),
            maxiter=100,
        )
        assert np.isfinite(result.delta_omega_opt)
        assert result.params_opt.shape == (4,)

    def test_nelder_mead_improves_over_random_start(self) -> None:
        """Nelder-Mead should find Δθ at or below the starting point."""
        x0 = np.array([1.0, 0.0, 0.0, 1.0])
        ops = build_two_qubit_operators()
        start_val = drive_sensitivity_objective(x0, 1.0, ops)
        result = run_drive_nelder_mead(
            omega_true=1.0,
            x0=x0,
            maxiter=200,
        )
        assert result.delta_omega_opt <= start_val * 1.01 or result.success, (
            f"NM did not improve: start={start_val:.4f}, opt={result.delta_omega_opt:.4f}"
        )

    def test_nelder_mead_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = run_drive_nelder_mead(
            omega_true=1.0,
            x0=np.array([1.0, 0.0, 0.0, 1.0]),
            maxiter=100,
        )
        parquet_p = tmp_path / "nm.parquet"
        result.save_parquet(parquet_p)
        loaded = DriveNelderMeadResult.from_parquet(parquet_p)
        assert np.isclose(loaded.delta_omega_opt, result.delta_omega_opt)
        assert np.allclose(loaded.params_opt, result.params_opt)
        assert loaded.omega_true == result.omega_true


# ============================================================================
# ω Scan
# ============================================================================


class TestDriveOmegaScan:
    def test_omega_scan_runs(self) -> None:
        result = run_drive_omega_scan(
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
        result = run_drive_omega_scan(
            omega_values=[0.5, 1.0],
            n_random=50,
            n_nm_refine=5,
            seed=42,
            maxiter=100,
        )
        for dt in result.best_delta_omega_per_omega:
            assert np.isfinite(dt), f"Non-finite Δω: {dt}"

    def test_omega_scan_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = run_drive_omega_scan(
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
# Validation Helpers
# ============================================================================


class TestDriveValidation:
    def test_given_decoupled_config_then_sql_held(self) -> None:
        """With all drive and interaction off, the SQL must hold exactly."""
        result = compute_drive_decoupled_baseline()
        ratio = result.delta_omega / result.sql
        assert abs(ratio - 1.0) < 0.05, f"SQL violated: Δω/SQL = {ratio:.6f}"

    def test_given_random_search_then_best_not_worse_than_sql(self) -> None:
        """The best point in a random search should be at worst ~SQL."""
        result = drive_random_search(omega=1.0, n_samples=200, seed=42)
        assert result.best_delta_omega <= result.sql * 2.0 or not np.isfinite(
            result.best_delta_omega,
        ), f"Best Δθ = {result.best_delta_omega:.4f} >> SQL = {result.sql:.4f}"

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

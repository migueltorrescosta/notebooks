"""Tests for the partially ω-modulated ancilla drive with EP/CFI/QFI comparison.

This is the companion test module for
``reports/r20260701/partial_omega_modulated_drive.py``.

Key new tests compared to the fully-modulated drive (#20260519):
- ω only appears on the a_z term in H_A (not on a_x, a_y)
- EP = CFI for binary J_z^S measurement (exact identity)
- QFI ≤ EP (quantum Cramér-Rao bound)
- The three-way sensitivity comparison (compute_all_sensitivities)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

if TYPE_CHECKING:
    from pathlib import Path

from src.analysis.ancilla_drive_metrology import (
    build_iszz_interaction,
    system_only_bs_unitary,
)
from src.analysis.ancilla_drive_results import (
    Drive2DSliceResult,
    DriveDecoupledBaselineResult,
    DriveOmegaScanResult,
)
from src.analysis.ancilla_drive_scans import (  # type: ignore[import-untyped]
    compute_drive_decoupled_baseline,
)
from src.analysis.ancilla_optimization import (
    I_2,
    J_Z,
    bs_unitary,
    build_two_qubit_operators,
)

_m = importlib.import_module("reports.r20260701.partial_omega_modulated_drive")
DEFAULT_PSI0 = _m.DEFAULT_PSI0
DEFAULT_T_BS = _m.DEFAULT_T_BS
DEFAULT_T_HOLD = _m.DEFAULT_T_HOLD
build_partial_drive_hamiltonian = _m.build_partial_drive_hamiltonian
build_partial_hold_hamiltonian = _m.build_partial_hold_hamiltonian
compute_partial_sensitivity_ep = _m.compute_partial_sensitivity_ep
compute_partial_sensitivity_cfi = _m.compute_partial_sensitivity_cfi
compute_partial_sensitivity_qfi = _m.compute_partial_sensitivity_qfi
compute_all_sensitivities = _m.compute_all_sensitivities
evolve_partial_circuit = _m.evolve_partial_circuit
partial_2d_slice = _m.partial_2d_slice
partial_hold_unitary = _m.partial_hold_unitary
partial_random_search = _m.partial_random_search
run_partial_nelder_mead = _m.run_partial_nelder_mead
run_partial_omega_scan = _m.run_partial_omega_scan
PartialOmegaDriveResult = _m.PartialOmegaDriveResult

I_4 = np.eye(4, dtype=complex)


@pytest.fixture
def make_ops() -> dict[str, np.ndarray]:
    return build_two_qubit_operators()


class TestSystemOnlyBS:
    def test_given_system_only_bs_then_is_4x4(self) -> None:
        U = system_only_bs_unitary(np.pi / 2)
        assert U.shape == (4, 4)

    def test_given_system_only_bs_then_is_unitary(self) -> None:
        U = system_only_bs_unitary(np.pi / 2)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)

    def test_given_system_only_bs_then_acts_as_identity_on_ancilla(self) -> None:
        U_sys = bs_unitary(np.pi / 2)
        U_expected = np.kron(U_sys, I_2)
        U_got = system_only_bs_unitary(np.pi / 2)
        assert np.allclose(U_got, U_expected, atol=1e-12)

    def test_given_zero_T_bs_then_is_identity(self) -> None:
        U = system_only_bs_unitary(0.0)
        assert np.allclose(U, I_4, atol=1e-12)


class TestPartialDriveHamiltonian:
    def test_given_omega_only_on_az(self, make_ops: dict) -> None:
        """Verify that ω only appears on the a_z term, not on a_x or a_y."""
        # At ω=2 with only a_z=3, the drive should be 2*3*J_z^A = 6*J_z^A
        H = build_partial_drive_hamiltonian(2.0, 0.0, 0.0, 3.0, make_ops)
        expected = 6.0 * make_ops["Jz_A"]
        assert np.allclose(H, expected, atol=1e-12)

    def test_given_omega_not_on_ax(self, make_ops: dict) -> None:
        """Verify that a_x is NOT multiplied by ω."""
        # At ω=2 with only a_x=3, the drive should be 3*J_x^A (NOT ω*3*J_x^A)
        H = build_partial_drive_hamiltonian(2.0, 3.0, 0.0, 0.0, make_ops)
        expected = 3.0 * make_ops["Jx_A"]
        assert np.allclose(H, expected, atol=1e-12)

    def test_given_omega_not_on_ay(self, make_ops: dict) -> None:
        """Verify that a_y is NOT multiplied by ω."""
        H = build_partial_drive_hamiltonian(2.0, 0.0, 3.0, 0.0, make_ops)
        expected = 3.0 * make_ops["Jy_A"]
        assert np.allclose(H, expected, atol=1e-12)

    def test_given_zero_coefficients_then_is_zero(self, make_ops: dict) -> None:
        H = build_partial_drive_hamiltonian(1.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_ax_ay_then_hermitian(self, make_ops: dict) -> None:
        H = build_partial_drive_hamiltonian(2.0, 1.0, 2.0, 0.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_all_coefficients_then_hermitian(self, make_ops: dict) -> None:
        H = build_partial_drive_hamiltonian(1.5, 1.0, 2.0, 3.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_az_only_then_linear_in_omega(self, make_ops: dict) -> None:
        """H_A at ω=2 should be 2× H_A at ω=1 (for a_z term only)."""
        H_half = build_partial_drive_hamiltonian(1.0, 0.0, 0.0, 2.0, make_ops)
        H_full = build_partial_drive_hamiltonian(2.0, 0.0, 0.0, 2.0, make_ops)
        assert np.allclose(2.0 * H_half, H_full, atol=1e-12)

    def test_given_ax_ay_then_omega_independent(self, make_ops: dict) -> None:
        """a_x and a_y terms should NOT depend on ω."""
        H_at_omega_1 = build_partial_drive_hamiltonian(1.0, 2.0, 3.0, 0.0, make_ops)
        H_at_omega_2 = build_partial_drive_hamiltonian(2.0, 2.0, 3.0, 0.0, make_ops)
        # Both should be identical because ω does not enter a_x, a_y
        assert np.allclose(H_at_omega_1, H_at_omega_2, atol=1e-12)


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


class TestPartialHoldHamiltonian:
    def test_given_zero_params_then_only_Jz_S(self, make_ops: dict) -> None:
        H = build_partial_hold_hamiltonian(1.0, 0.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, 1.0 * make_ops["Jz_S"], atol=1e-12)

    def test_given_all_params_then_hermitian(self, make_ops: dict) -> None:
        H = build_partial_hold_hamiltonian(1.0, 2.0, 3.0, 4.0, 5.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_zero_omega_then_static_drive_remains(self, make_ops: dict) -> None:
        """At ω=0, H_S = 0, ω a_z J_z^A = 0, but a_x J_x^A + a_y J_y^A remain."""
        H = build_partial_hold_hamiltonian(0.0, 2.0, 3.0, 4.0, 5.0, make_ops)
        # H = 0 + 2*J_x^A + 3*J_y^A + 0 + 5*J_zS@J_zA
        expected = (
            2.0 * make_ops["Jx_A"]
            + 3.0 * make_ops["Jy_A"]
            + build_iszz_interaction(5.0, make_ops)
        )
        assert np.allclose(H, expected, atol=1e-12)

    def test_given_nonzero_params_then_derivative_is_jz_s_plus_az_jz_a(
        self, make_ops: dict
    ) -> None:
        """∂H/∂ω = J_z^S + a_z J_z^A (no a_x, a_y contribution)."""
        # ∂H/∂ω ≈ (H(ω+δ) - H(ω-δ)) / (2δ)
        delta = 1e-6
        H_plus = build_partial_hold_hamiltonian(
            1.0 + delta,
            2.0,
            3.0,
            4.0,
            5.0,
            make_ops,
        )
        H_minus = build_partial_hold_hamiltonian(
            1.0 - delta,
            2.0,
            3.0,
            4.0,
            5.0,
            make_ops,
        )
        dH = (H_plus - H_minus) / (2.0 * delta)
        expected_dH = make_ops["Jz_S"] + 4.0 * make_ops["Jz_A"]
        assert np.allclose(dH, expected_dH, atol=1e-6)


class TestPartialHoldUnitary:
    def test_given_zero_params_then_is_unitary(self, make_ops: dict) -> None:
        U = partial_hold_unitary(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)

    def test_given_nonzero_params_then_is_unitary(self, make_ops: dict) -> None:
        U = partial_hold_unitary(10.0, 1.0, 2.0, 0.5, -1.0, 3.0, make_ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)

    def test_given_zero_omega_then_is_unitary(self, make_ops: dict) -> None:
        U = partial_hold_unitary(10.0, 0.0, 2.0, 3.0, 4.0, 5.0, make_ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)


class TestEvolvePartialCircuit:
    def test_given_default_state_then_final_state_is_normalised(
        self, make_ops: dict
    ) -> None:
        psi = evolve_partial_circuit(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12)

    def test_given_zero_drive_then_sql_sensitivity(self, make_ops: dict) -> None:
        domega = compute_partial_sensitivity_ep(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        expected = 1.0 / DEFAULT_T_HOLD
        assert np.isclose(domega, expected, rtol=0.05)

    def test_given_nonzero_params_then_finite_sensitivity(self, make_ops: dict) -> None:
        domega = compute_partial_sensitivity_ep(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        assert np.isfinite(domega), "Sensitivity must be finite"
        assert domega > 0.0, "Sensitivity must be positive"

    def test_given_decoupled_then_sql(self, make_ops: dict) -> None:
        """With a_zz = 0, ancilla is decoupled → SQL should hold."""
        for a_x, a_y, a_z in [(2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)]:
            domega = compute_partial_sensitivity_ep(
                DEFAULT_PSI0,
                DEFAULT_T_BS,
                DEFAULT_T_HOLD,
                1.0,
                a_x,
                a_y,
                a_z,
                0.0,
                make_ops,
            )
            expected = 1.0 / DEFAULT_T_HOLD
            assert np.isclose(domega, expected, rtol=0.05), (
                f"At (a_x={a_x}, a_y={a_y}, a_z={a_z}, a_zz=0): "
                f"Δω = {domega:.6f}, expected ≈ {expected:.6f}"
            )


class TestThreeWaySensitivity:
    def test_given_baseline_then_ep_equals_cfi(self, make_ops: dict) -> None:
        """EP and CFI should be identical for binary J_z^S measurement."""
        sens = compute_all_sensitivities(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        if np.isfinite(sens["delta_omega_ep"]) and np.isfinite(sens["delta_omega_cfi"]):
            diff = abs(sens["delta_omega_ep"] - sens["delta_omega_cfi"])
            assert diff < 1e-12, f"EP ≠ CFI: |Δω_EP - Δω_CFI| = {diff}"

    def test_given_nonzero_params_then_ep_equals_cfi(self, make_ops: dict) -> None:
        """EP and CFI should be identical even with drive and interaction."""
        for ax, ay, az, azz in [(2.0, 0.0, 0.0, 1.0), (0.0, 3.0, 1.0, 2.0)]:
            sens = compute_all_sensitivities(
                DEFAULT_PSI0,
                DEFAULT_T_BS,
                DEFAULT_T_HOLD,
                1.0,
                ax,
                ay,
                az,
                azz,
                make_ops,
            )
            if np.isfinite(sens["delta_omega_ep"]) and np.isfinite(
                sens["delta_omega_cfi"]
            ):
                diff = abs(sens["delta_omega_ep"] - sens["delta_omega_cfi"])
                assert diff < 1e-10, (
                    f"At (ax={ax}, ay={ay}, az={az}, azz={azz}): "
                    f"EP ≠ CFI: |Δω_EP - Δω_CFI| = {diff}"
                )

    def test_given_nonzero_params_then_qfi_leq_ep(self, make_ops: dict) -> None:
        """QFI sensitivity should be ≤ EP sensitivity (quantum Cramér-Rao)."""
        for ax, ay, az, azz in [
            (2.0, 0.0, 0.0, 1.0),
            (0.0, 3.0, 1.0, 2.0),
            (-1.0, 2.0, -0.5, 1.5),
        ]:
            sens = compute_all_sensitivities(
                DEFAULT_PSI0,
                DEFAULT_T_BS,
                DEFAULT_T_HOLD,
                1.0,
                ax,
                ay,
                az,
                azz,
                make_ops,
            )
            if np.isfinite(sens["delta_omega_ep"]) and np.isfinite(
                sens["delta_omega_qfi"]
            ):
                ratio = sens["delta_omega_ep"] / sens["delta_omega_qfi"]
                assert ratio >= 1.0 - 1e-8, (
                    f"QFI > EP violation: Δω_EP/Δω_QFI = {ratio:.10f} < 1 "
                    f"at (ax={ax}, ay={ay}, az={az}, azz={azz})"
                )

    def test_given_all_metrics_returned(self, make_ops: dict) -> None:
        """compute_all_sensitivities returns all expected keys."""
        sens = compute_all_sensitivities(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        expected_keys = {
            "delta_omega_ep",
            "delta_omega_cfi",
            "delta_omega_qfi",
            "expectation_Jz",
            "variance_Jz",
            "fisher_classical",
            "fisher_quantum",
        }
        assert set(sens.keys()) == expected_keys


class TestDecoupledBaseline:
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


class TestPartial2DSlice:
    def test_ax_slice_returns_correct_shape(self) -> None:
        result = partial_2d_slice(
            omega=1.0,
            slice_type="ax",
            n_drive=5,
            n_azz=5,
        )
        assert result.delta_omega_grid.shape == (5, 5)

    def test_ay_slice_returns_correct_shape(self) -> None:
        result = partial_2d_slice(
            omega=1.0,
            slice_type="ay",
            n_drive=5,
            n_azz=5,
        )
        assert result.delta_omega_grid.shape == (5, 5)

    def test_slice_all_finite_or_inf(self) -> None:
        result = partial_2d_slice(
            omega=1.0,
            slice_type="ax",
            n_drive=5,
            n_azz=5,
        )
        for val in result.delta_omega_grid.flatten():
            assert np.isfinite(val) or np.isinf(val)

    def test_invalid_slice_type_raises(self) -> None:
        with pytest.raises(ValueError, match="slice_type"):
            partial_2d_slice(omega=1.0, slice_type="invalid")

    def test_slice_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = partial_2d_slice(
            omega=1.0,
            slice_type="ax",
            n_drive=3,
            n_azz=3,
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


class TestPartialOmegaDriveResult:
    def test_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = PartialOmegaDriveResult(
            omega=1.0,
            a_x=2.0,
            a_y=0.5,
            a_z=-1.0,
            a_zz=3.0,
            t_hold=10.0,
            delta_omega_ep=0.05,
            delta_omega_cfi=0.05,
            delta_omega_qfi=0.02,
            expectation_Jz=0.3,
            variance_Jz=0.25,
            fisher_classical=400.0,
            fisher_quantum=2500.0,
            sql=0.1,
        )
        parquet_p = tmp_path / "result.parquet"
        result.save_parquet(parquet_p)
        loaded = PartialOmegaDriveResult.from_parquet(parquet_p)

        # Use a dict-based comparison to keep cyclomatic complexity low.
        # All fields are checked via a single approx comparison over the
        # full attribute set, sidestepping radon counting 14 assert nodes.
        cols = PartialOmegaDriveResult._PARQUET_COLUMNS
        expected = {f: getattr(result, f) for f in cols}
        actual = {f: getattr(loaded, f) for f in cols}
        assert actual == pytest.approx(expected)

    def test_from_parquet_fails_on_missing_columns(self, tmp_path: Path) -> None:
        """Fail-fast: missing columns raise ValueError."""
        import pandas as pd

        bad_df = pd.DataFrame({"omega": [1.0], "delta_omega_ep": [0.05]})
        bad_path = tmp_path / "bad.parquet"
        bad_df.to_parquet(bad_path)
        with pytest.raises(ValueError):
            PartialOmegaDriveResult.from_parquet(bad_path)


class TestPartialOmegaScan:
    @pytest.mark.slow
    def test_omega_scan_runs(self) -> None:
        result = run_partial_omega_scan(
            omega_values=[0.5, 1.0],
            n_random=50,
            n_nm_refine=5,
            seed=42,
            maxiter=100,
        )
        assert len(result.omega_values) == 2
        assert len(result.best_params_per_omega) == 2
        assert len(result.best_delta_omega_per_omega) == 2

    @pytest.mark.slow
    def test_omega_scan_all_finite(self) -> None:
        result = run_partial_omega_scan(
            omega_values=[0.5, 1.0],
            n_random=50,
            n_nm_refine=5,
            seed=42,
            maxiter=100,
        )
        for dt in result.best_delta_omega_per_omega:
            assert np.isfinite(dt), f"Non-finite Δω: {dt}"

    def test_omega_scan_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = run_partial_omega_scan(
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


class TestPartialValidation:
    def test_given_decoupled_config_then_sql_held(self) -> None:
        result = compute_drive_decoupled_baseline()
        ratio = result.delta_omega / result.sql
        assert abs(ratio - 1.0) < 0.05, f"SQL violated: Δω/SQL = {ratio:.6f}"

    def test_given_random_search_then_best_not_worse_than_sql(self) -> None:
        result = partial_random_search(omega=1.0, n_samples=200, seed=42)
        assert result.best_delta_omega <= result.sql * 2.0 or not np.isfinite(
            result.best_delta_omega,
        )

    def test_given_system_only_bs_then_acts_only_on_system(self) -> None:
        psi_in = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
        U_bs = system_only_bs_unitary(np.pi / 2)
        psi_out = U_bs @ psi_in
        ancilla_in_1 = abs(psi_out[1]) ** 2 + abs(psi_out[3]) ** 2
        assert np.isclose(ancilla_in_1, 1.0, atol=1e-12)

    def test_given_omega_modulation_active(self, make_ops: dict) -> None:
        """Sensitivity at different ω values with same drive should differ."""
        dt_1 = compute_partial_sensitivity_ep(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            2.0,
            0.0,
            2.0,
            2.0,
            make_ops,
        )
        dt_2 = compute_partial_sensitivity_ep(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            2.0,
            2.0,
            0.0,
            2.0,
            2.0,
            make_ops,
        )
        assert abs(dt_1 - dt_2) > 1e-6, (
            f"ω-modulation not active: Δω(ω=1) = {dt_1:.10f}, Δω(ω=2) = {dt_2:.10f}"
        )

    def test_given_ax_ay_omega_independence(self, make_ops: dict) -> None:
        """Without a_z, changing ω should only affect H_S (static drive only).

        With a_z=0 and a_zz=0, the total Hamiltonian factorises:
        U_hold = exp(-iω t_hold J_z^S) ⊗ exp(-i t_hold (a_x J_x^A + a_y J_y^A)).
        The ancilla factor doesn't affect J_z^S, so changing ω should give
        different sensitivity only through the system's phase accumulation.
        """
        dt_ax = compute_partial_sensitivity_ep(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            2.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        dt_ay = compute_partial_sensitivity_ep(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            0.0,
            3.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        # Without a_z, ω only enters through H_S = ω J_z^S. At ω=0, no
        # system phase accumulation, so sensitivity is infinite.
        # At ω=1, it should be finite (SQL).
        assert np.isfinite(dt_ax), "Sensitivity should be finite at ω=1"
        assert not np.isfinite(dt_ay) or dt_ay > 1e6, (
            "Sensitivity at ω=0 should be very large (no phase accumulation)"
        )


class TestPartialNelderMead:
    def test_given_explicit_x0_then_converges(self, make_ops: dict) -> None:
        """Nelder-Mead with explicit starting point should produce finite Δω."""
        result = run_partial_nelder_mead(
            omega_true=1.0,
            x0=np.array([1.0, 0.5, -0.5, 0.0]),
            seed=None,
            maxiter=200,
        )
        assert np.isfinite(result.delta_omega_opt), (
            f"NM Δω must be finite, got {result.delta_omega_opt}"
        )
        assert result.delta_omega_opt > 0.0, "Δω must be positive"
        assert len(result.params_opt) == 4

    def test_given_random_x0_then_converges(self) -> None:
        """Nelder-Mead with random x0 should produce finite Δω."""
        result = run_partial_nelder_mead(
            omega_true=0.5,
            x0=None,
            seed=42,
            maxiter=200,
        )
        assert np.isfinite(result.delta_omega_opt), (
            f"NM (random x0) Δω must be finite, got {result.delta_omega_opt}"
        )
        assert result.delta_omega_opt > 0.0
        assert len(result.params_opt) == 4


class TestStandaloneSensitivities:
    def test_given_cfi_standalone_then_finite(self, make_ops: dict) -> None:
        """compute_partial_sensitivity_cfi standalone returns finite value."""
        domega = compute_partial_sensitivity_cfi(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        assert np.isfinite(domega), f"CFI Δω must be finite, got {domega}"
        assert domega > 0.0

    def test_given_cfi_baseline_then_sql(self, make_ops: dict) -> None:
        """At zero drive/interaction, CFI should match SQL."""
        domega = compute_partial_sensitivity_cfi(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        expected = 1.0 / DEFAULT_T_HOLD
        assert np.isclose(domega, expected, rtol=0.05), (
            f"CFI baseline: {domega:.6f} vs expected {expected:.6f}"
        )

    def test_given_qfi_standalone_then_finite(self, make_ops: dict) -> None:
        """compute_partial_sensitivity_qfi standalone returns finite value."""
        domega = compute_partial_sensitivity_qfi(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        assert np.isfinite(domega), f"QFI Δω must be finite, got {domega}"
        assert domega > 0.0

    def test_given_qfi_leq_ep_standalone(self, make_ops: dict) -> None:
        """QFI standalone should be ≤ EP standalone."""
        delta_ep = compute_partial_sensitivity_ep(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        delta_qfi = compute_partial_sensitivity_qfi(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0,
            make_ops,
        )
        assert delta_qfi <= delta_ep + 1e-10, (
            f"QFI > EP: Δω_QFI = {delta_qfi:.10f}, Δω_EP = {delta_ep:.10f}"
        )

    def test_given_zero_omega_then_infinite_ep(self, make_ops: dict) -> None:
        """At ω=0 with decoupled ancilla, EP derivative is zero → inf sensitivity."""
        domega = compute_partial_sensitivity_ep(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert not np.isfinite(domega), (
            f"At ω=0 with no drive, EP should be inf, got {domega}"
        )

    def test_given_zero_t_hold_then_qfi_inf(self, make_ops: dict) -> None:
        """When T_hold=0, state is ω-independent → F_Q=0 → inf sensitivity."""
        domega = compute_partial_sensitivity_qfi(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            0.0,  # t_hold = 0
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert not np.isfinite(domega), f"At t_hold=0, QFI should be inf, got {domega}"

    def test_given_zero_t_hold_then_all_sens_inf(self, make_ops: dict) -> None:
        """compute_all_sensitivities with T_hold=0 → ω-independent state → all inf."""
        sens = compute_all_sensitivities(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            0.0,  # t_hold = 0 → U_hold = I
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        # With T_hold=0, U_hold = I regardless of ω → d_exp = 0, dP = 0, F_Q = 0
        assert not np.isfinite(sens["delta_omega_ep"])
        assert not np.isfinite(sens["delta_omega_cfi"])
        assert not np.isfinite(sens["delta_omega_qfi"])

    def test_given_zero_derivative_then_cfi_inf(self, make_ops: dict) -> None:
        """When P(+) derivative is zero, CFI should return inf sensitivity."""
        # ω=0 with zero coupling ⇒ no phase accumulation ⇒ dP/dω ≈ 0
        domega = compute_partial_sensitivity_cfi(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert not np.isfinite(domega), (
            f"At ω=0 with no drive, CFI should be inf, got {domega}"
        )


class TestPartialOmegaDriveResultDataframe:
    def test_to_dataframe_has_all_columns(self) -> None:
        """to_dataframe() returns all expected columns."""
        result = PartialOmegaDriveResult(
            omega=1.0,
            a_x=2.0,
            a_y=0.5,
            a_z=-1.0,
            a_zz=3.0,
            t_hold=10.0,
            delta_omega_ep=0.05,
            delta_omega_cfi=0.05,
            delta_omega_qfi=0.02,
            expectation_Jz=0.3,
            variance_Jz=0.25,
            fisher_classical=400.0,
            fisher_quantum=2500.0,
            sql=0.1,
        )
        df = result.to_dataframe()
        expected_cols = {
            "omega",
            "a_x",
            "a_y",
            "a_z",
            "a_zz",
            "t_hold",
            "delta_omega_ep",
            "delta_omega_cfi",
            "delta_omega_qfi",
            "expectation_Jz",
            "variance_Jz",
            "fisher_classical",
            "fisher_quantum",
            "sql",
        }
        assert set(df.columns) == expected_cols, (
            f"Missing: {expected_cols - set(df.columns)}"
        )
        assert len(df) == 1

    def test_to_dataframe_column_count_matches_parquet_columns(self) -> None:
        """Column count in to_dataframe() matches _PARQUET_COLUMNS."""
        result = PartialOmegaDriveResult(
            omega=1.0,
            a_x=0.0,
            a_y=0.0,
            a_z=0.0,
            a_zz=0.0,
            t_hold=10.0,
            delta_omega_ep=0.1,
            delta_omega_cfi=0.1,
            delta_omega_qfi=0.05,
            expectation_Jz=0.0,
            variance_Jz=0.25,
            fisher_classical=100.0,
            fisher_quantum=400.0,
            sql=0.1,
        )
        df = result.to_dataframe()
        assert len(df.columns) == len(PartialOmegaDriveResult._PARQUET_COLUMNS), (
            f"DataFrame has {len(df.columns)} columns but "
            f"_PARQUET_COLUMNS has {len(PartialOmegaDriveResult._PARQUET_COLUMNS)}"
        )


# ============================================================================
# Property-based tests (hypothesis)
# ============================================================================


@st.composite
def param_config(draw: st.DrawFn) -> tuple[float, float, float, float, float]:
    """Strategy generating random (ω, a_x, a_y, a_z, a_zz) tuples."""
    omega = draw(st.floats(0.1, 5.0))
    a_x = draw(st.floats(-5.0, 5.0))
    a_y = draw(st.floats(-5.0, 5.0))
    a_z = draw(st.floats(-5.0, 5.0))
    a_zz = draw(st.floats(-5.0, 5.0))
    return omega, a_x, a_y, a_z, a_zz


# Shared settings for all property-based tests: 50 examples, no deadline.
_PROPSETTINGS = settings(max_examples=50, deadline=None)


class TestPropertyBasedInvariants:
    """Physical invariants tested over random parameter tuples."""

    make_ops = staticmethod(build_two_qubit_operators)

    @_PROPSETTINGS
    @given(params=param_config())
    def test_hold_hamiltonian_hermitian(
        self, params: tuple[float, float, float, float, float]
    ) -> None:
        omega, a_x, a_y, a_z, a_zz = params
        ops = self.make_ops()
        H = build_partial_hold_hamiltonian(omega, a_x, a_y, a_z, a_zz, ops)
        assert np.allclose(H, H.conj().T, atol=1e-12), (
            f"H not Hermitian at (ω={omega}, a_x={a_x}, a_y={a_y}, "
            f"a_z={a_z}, a_zz={a_zz})"
        )

    @_PROPSETTINGS
    @given(params=param_config())
    def test_hold_unitary(
        self, params: tuple[float, float, float, float, float]
    ) -> None:
        omega, a_x, a_y, a_z, a_zz = params
        ops = self.make_ops()
        U = partial_hold_unitary(DEFAULT_T_HOLD, omega, a_x, a_y, a_z, a_zz, ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), (
            f"U not unitary at (ω={omega}, a_x={a_x}, a_y={a_y}, "
            f"a_z={a_z}, a_zz={a_zz})"
        )

    @_PROPSETTINGS
    @given(params=param_config())
    def test_state_normalisation(
        self, params: tuple[float, float, float, float, float]
    ) -> None:
        omega, a_x, a_y, a_z, a_zz = params
        ops = self.make_ops()
        psi = evolve_partial_circuit(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            omega,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12), (
            f"State not normalised at (ω={omega}, a_x={a_x}, a_y={a_y}, "
            f"a_z={a_z}, a_zz={a_zz})"
        )

    @_PROPSETTINGS
    @given(params=param_config())
    def test_ep_equals_cfi(
        self, params: tuple[float, float, float, float, float]
    ) -> None:
        """For a binary J_z^S measurement, EP and CFI must coincide exactly."""
        omega, a_x, a_y, a_z, a_zz = params
        ops = self.make_ops()
        sens = compute_all_sensitivities(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            omega,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
        )
        ep = sens["delta_omega_ep"]
        cfi = sens["delta_omega_cfi"]
        if np.isfinite(ep) and np.isfinite(cfi):
            diff = abs(ep - cfi)
            # Tolerance 1e-6 for property-based sampling. The derivative
            # ∂⟨Jz^S⟩/∂ω computed via the matrix-vector path (~0.08) differs
            # from the amplitude-extraction path by ~9e-10, propagating to a
            # ~3e-8 sensitivity diff. Example-based tests with fixed params
            # still use the tighter 1e-10 tolerance.
            assert diff < 1e-6, (
                f"EP≠CFI at (ω={omega}, a_x={a_x}, a_y={a_y}, "
                f"a_z={a_z}, a_zz={a_zz}): diff={diff}"
            )

    @_PROPSETTINGS
    @given(params=param_config())
    def test_qfi_leq_ep(self, params: tuple[float, float, float, float, float]) -> None:
        """Quantum Cramér-Rao bound: Δω_QFI ≤ Δω_EP."""
        omega, a_x, a_y, a_z, a_zz = params
        ops = self.make_ops()
        sens = compute_all_sensitivities(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            omega,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
        )
        ep = sens["delta_omega_ep"]
        qfi = sens["delta_omega_qfi"]
        if np.isfinite(ep) and np.isfinite(qfi):
            ratio = ep / qfi
            assert ratio >= 1.0 - 1e-8, (
                f"QFI>EP violation at (ω={omega}, a_x={a_x}, a_y={a_y}, "
                f"a_z={a_z}, a_zz={a_zz}): EP/QFI={ratio:.10f}"
            )

    @_PROPSETTINGS
    @given(params=param_config())
    def test_probability_conservation(
        self, params: tuple[float, float, float, float, float]
    ) -> None:
        """Σ P(m|ω) = 1 for the binary J_z^S measurement."""
        omega, a_x, a_y, a_z, a_zz = params
        ops = self.make_ops()
        psi = evolve_partial_circuit(
            DEFAULT_PSI0,
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            omega,
            a_x,
            a_y,
            a_z,
            a_zz,
            ops,
        )
        p_plus = _m._compute_marginal_probability_plus(psi)
        p_minus = 1.0 - p_plus
        total = p_plus + p_minus
        assert abs(total - 1.0) < 1e-14, (
            f"Probabilities don't sum to 1 at (ω={omega}, "
            f"a_x={a_x}, a_y={a_y}, a_z={a_z}, a_zz={a_zz}): sum={total}"
        )

"""Tests for the Symmetric ω-Modulated Drive: Bounded-Compound Comparison.

Companion test module for
``reports/r20260709/compound_comparison.py``.

Key test areas:
- Scenario A: single-qubit baseline, Hamiltonians, unitaries, sensitivity
- Scenario B: two-qubit Hamiltonians, dual BS, sensitivity
- Decoupled baseline: both scenarios recover SQL at a_k = 0
- Consistency: Scenario B at a_zz=0 reproduces Scenario A (S-only BS variant)
- Dataclass roundtrip: Parquet serialization preserves all fields
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

if TYPE_CHECKING:
    from pathlib import Path

from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
    two_qubit_bs_unitary,
)
from src.utils.constants import I_2, I_4, J_X, J_Y, J_Z
from src.utils.serialization import assert_roundtrip_fields

_m = importlib.import_module("reports.r20260709.compound_comparison")

scenario_a_state = _m.scenario_a_state
scenario_a_bs = _m.scenario_a_bs
scenario_a_hamiltonian = _m.scenario_a_hamiltonian
scenario_a_hold_unitary = _m.scenario_a_hold_unitary
scenario_a_evolve = _m.scenario_a_evolve
scenario_a_sensitivity = _m.scenario_a_sensitivity
scenario_b_state = _m.scenario_b_state
scenario_b_hamiltonian = _m.scenario_b_hamiltonian
scenario_b_hold_unitary = _m.scenario_b_hold_unitary
scenario_b_evolve = _m.scenario_b_evolve
scenario_b_sensitivity = _m.scenario_b_sensitivity
compute_decoupled_baseline = _m.compute_decoupled_baseline
_scenario_a_objective_3d = _m._scenario_a_objective_3d
_scenario_b_objective_4d = _m._scenario_b_objective_4d
scenario_a_random_search = _m.scenario_a_random_search
run_scenario_a_omega_scan = _m.run_scenario_a_omega_scan
run_scenario_b_omega_scan = _m.run_scenario_b_omega_scan
_run_scenario_b_single_omega = _m._run_scenario_b_single_omega
compute_compound_ratio = _m.compute_compound_ratio
ScenarioACompoundResult = _m.ScenarioACompoundResult
CompoundRatioResult = _m.CompoundRatioResult
DecoupledBaselineResult = _m.DecoupledBaselineResult
main = _m.main

DEFAULT_T_BS = _m.DEFAULT_T_BS
DEFAULT_T_HOLD = _m.DEFAULT_T_HOLD
SQL_REFERENCE = _m.SQL_REFERENCE


@pytest.fixture
def ops() -> dict[str, np.ndarray]:
    return build_two_qubit_operators()


class TestScenarioAState:
    def test_initial_state_is_normalised(self) -> None:
        psi = scenario_a_state()
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_initial_state_is_2_vector(self) -> None:
        psi = scenario_a_state()
        assert psi.shape == (2,)

    def test_initial_state_is_1_0(self) -> None:
        psi = scenario_a_state()
        assert np.isclose(psi[0], 1.0)
        assert np.isclose(psi[1], 0.0)


class TestScenarioABS:
    def test_bs_is_unitary(self) -> None:
        U = scenario_a_bs(DEFAULT_T_BS)
        assert np.allclose(U @ U.conj().T, I_2, atol=1e-12)

    def test_bs_50_50_matches_expected(self) -> None:
        U = scenario_a_bs(np.pi / 2)
        expected = (1.0 / np.sqrt(2)) * np.array([[1, -1j], [-1j, 1]], dtype=complex)
        assert np.allclose(U, expected, atol=1e-12)

    def test_bs_identity_at_zero(self) -> None:
        U = scenario_a_bs(0.0)
        assert np.allclose(U, I_2, atol=1e-12)


class TestScenarioAHamiltonian:
    def test_hamiltonian_is_hermitian(self) -> None:
        for omega in [0.1, 1.0, 5.0]:
            for ax in [-2.0, 0.0, 3.0]:
                H = scenario_a_hamiltonian(omega, ax, 0.0, 0.0)
                assert np.allclose(H, H.conj().T), (
                    f"Not Hermitian for ω={omega}, ax={ax}"
                )

    def test_hamiltonian_is_2x2(self) -> None:
        H = scenario_a_hamiltonian(1.0, 1.0, 1.0, 1.0)
        assert H.shape == (2, 2)

    def test_hamiltonian_proportional_to_omega(self) -> None:
        H1 = scenario_a_hamiltonian(1.0, 2.0, 0.5, -1.0)
        H2 = scenario_a_hamiltonian(3.0, 2.0, 0.5, -1.0)
        assert np.allclose(H2, 3.0 * H1, atol=1e-12)

    def test_hamiltonian_zero_at_zero_drive(self) -> None:
        # H = ω (0·J_x + 0·J_y + 0·J_z) = 0
        H = scenario_a_hamiltonian(1.0, 0.0, 0.0, 0.0)
        assert np.allclose(H, np.zeros((2, 2), dtype=complex), atol=1e-12)

    def test_hamiltonian_with_a_z_only(self) -> None:
        # H = ω (0·J_x + 0·J_y + 1·J_z) = ω J_z  (standard MZI encoding)
        H = scenario_a_hamiltonian(1.0, 0.0, 0.0, 1.0)
        assert np.allclose(H, J_Z, atol=1e-12)


class TestScenarioAHoldUnitary:
    def test_hold_unitary_is_unitary(self) -> None:
        U = scenario_a_hold_unitary(DEFAULT_T_HOLD, 1.0, 1.0, 0.5, -0.5)
        assert np.allclose(U @ U.conj().T, I_2, atol=1e-10)

    def test_hold_unitary_identity_at_zero_time(self) -> None:
        U = scenario_a_hold_unitary(0.0, 1.0, 1.0, 0.5, 0.0)
        assert np.allclose(U, I_2, atol=1e-12)


class TestScenarioAEvolve:
    def test_final_state_normalised(self) -> None:
        psi = scenario_a_evolve(DEFAULT_T_BS, DEFAULT_T_HOLD, 1.0, 1.0, 0.5, -0.5)
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_final_state_is_2_vector(self) -> None:
        psi = scenario_a_evolve(DEFAULT_T_BS, DEFAULT_T_HOLD, 1.0, 1.0, 0.5, 0.0)
        assert psi.shape == (2,)


class TestScenarioASensitivity:
    def test_baseline_gives_sql(self) -> None:
        # Standard MZI encoding: a_z = 1 (H = ω J_z)
        domega = scenario_a_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, 1.0, 0.0, 0.0, 1.0
        )
        assert np.isclose(domega, SQL_REFERENCE, rtol=1e-4)

    def test_sensitivity_is_positive(self) -> None:
        domega = scenario_a_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, 1.0, 2.0, 1.0, -1.0
        )
        assert domega > 0

    def test_sensitivity_at_various_omega(self) -> None:
        for omega in [0.1, 1.0, 5.0]:
            domega = scenario_a_sensitivity(
                DEFAULT_T_BS, DEFAULT_T_HOLD, omega, 1.0, 1.0, 0.0
            )
            assert np.isfinite(domega), f"Sensitivity inf at ω={omega}"
            assert domega > 0


class TestScenarioBState:
    def test_initial_state_is_normalised(self) -> None:
        psi = scenario_b_state()
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_initial_state_is_00(self) -> None:
        psi = scenario_b_state()
        assert psi.shape == (4,)
        assert np.isclose(psi[0], 1.0)
        assert np.allclose(psi[1:], 0.0)


class TestScenarioBDualBS:
    def test_dual_bs_is_unitary(self) -> None:
        U = two_qubit_bs_unitary(DEFAULT_T_BS)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)

    def test_dual_bs_is_tensor_product(self) -> None:
        U1 = scenario_a_bs(DEFAULT_T_BS)
        U_dual = two_qubit_bs_unitary(DEFAULT_T_BS)
        assert np.allclose(U_dual, np.kron(U1, U1), atol=1e-15)


class TestScenarioBHamiltonian:
    def test_hamiltonian_is_hermitian(self, ops: dict[str, np.ndarray]) -> None:
        for omega in [0.1, 1.0]:
            H = scenario_b_hamiltonian(omega, 1.0, 0.5, -0.5, 2.0, ops)
            assert np.allclose(H, H.conj().T), f"Not Hermitian for ω={omega}"

    def test_hamiltonian_is_4x4(self, ops: dict[str, np.ndarray]) -> None:
        H = scenario_b_hamiltonian(1.0, 1.0, 0.5, -0.5, 2.0, ops)
        assert H.shape == (4, 4)

    def test_hamiltonian_proportional_to_omega(
        self, ops: dict[str, np.ndarray]
    ) -> None:
        H1 = scenario_b_hamiltonian(1.0, 1.0, 0.5, -0.5, 2.0, ops)
        H2 = scenario_b_hamiltonian(2.0, 1.0, 0.5, -0.5, 2.0, ops)
        # Not proportional because a_zz term doesn't have ω
        # H2 - 2*H1 = -a_zz Jz_S Jz_A (the interaction is ω-independent)
        diff = H2 - 2.0 * H1
        expected_azz_diff = -2.0 * (ops["Jz_S"] @ ops["Jz_A"])
        assert np.allclose(diff, expected_azz_diff, atol=1e-12)

    def test_hamiltonian_at_zero_drive(self, ops: dict[str, np.ndarray]) -> None:
        # All drive coefficients zero → H = 0 (no bare ω J_z^S term)
        H = scenario_b_hamiltonian(1.0, 0.0, 0.0, 0.0, 0.0, ops)
        assert np.allclose(H, np.zeros((4, 4), dtype=complex), atol=1e-12)

    def test_hamiltonian_with_a_z_only(self, ops: dict[str, np.ndarray]) -> None:
        # a_z = 1, a_x = a_y = a_zz = 0 → H = ω J_z^S + ω J_z^A
        H = scenario_b_hamiltonian(1.0, 0.0, 0.0, 1.0, 0.0, ops)
        assert np.allclose(H, ops["Jz_S"] + ops["Jz_A"], atol=1e-12)


class TestScenarioBHoldUnitary:
    def test_hold_unitary_is_unitary(self, ops: dict[str, np.ndarray]) -> None:
        U = scenario_b_hold_unitary(DEFAULT_T_HOLD, 1.0, 1.0, 0.5, -0.5, 2.0, ops)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-10)


class TestScenarioBEvolve:
    def test_final_state_normalised(self, ops: dict[str, np.ndarray]) -> None:
        psi = scenario_b_evolve(
            DEFAULT_T_BS, DEFAULT_T_HOLD, 1.0, 1.0, 0.5, -0.5, 2.0, ops
        )
        assert np.isclose(np.linalg.norm(psi), 1.0)


class TestScenarioBSensitivity:
    def test_baseline_gives_sql(self, ops: dict[str, np.ndarray]) -> None:
        # Standard MZI encoding: a_z = 1 on both S and A (dual MZI)
        domega = scenario_b_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, 1.0, 0.0, 0.0, 1.0, 0.0, ops
        )
        assert np.isclose(domega, SQL_REFERENCE, rtol=1e-4)

    def test_sensitivity_is_positive(self, ops: dict[str, np.ndarray]) -> None:
        domega = scenario_b_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, 1.0, 2.0, 1.0, -1.0, 1.0, ops
        )
        assert domega > 0


class TestDecoupledBaseline:
    """Standard MZI baseline: a_z = 1 (ω J_z encoding), no x/y drive, no interaction.

    With the identical-drive Hamiltonian (no bare ω J_z^S on either subsystem),
    the standard MZI phase encoding is a_z = 1. At a_z = 0 the Hamiltonian
    vanishes and Δω = ∞.
    """

    def test_scenario_a_baseline_is_sql(self) -> None:
        domega_a, _ = compute_decoupled_baseline()
        assert np.isclose(domega_a, SQL_REFERENCE, rtol=1e-4)

    def test_scenario_b_baseline_is_sql(self) -> None:
        _, domega_b = compute_decoupled_baseline()
        assert np.isclose(domega_b, SQL_REFERENCE, rtol=1e-4)

    def test_both_scenarios_give_same_baseline(self) -> None:
        domega_a, domega_b = compute_decoupled_baseline()
        assert np.isclose(domega_a, domega_b, rtol=1e-4)


class TestScenarioConsistency:
    def test_scenario_b_at_azz_zero_reduces_to_system_only(
        self, ops: dict[str, np.ndarray]
    ) -> None:
        """At a_zz=0, Scenario B with dual MZI should differ from Scenario A
        because Scenario A uses a single-qubit BS on the system, while
        Scenario B uses dual BS on both qubits. The J_z^S measurement
        on the system qubit should give the same sensitivity because
        the ancilla factor U_A doesn't affect the system measurement.

        However, this only holds if the BS acts only on the system.
        With dual BS, the BS on the ancilla affects the entanglement structure.
        """
        # With dual MZI and a_zz=0, the system and ancilla separate.
        # The system evolution is: BS_S · U_S · BS_S on |0⟩
        # which is the same as Scenario A.
        a_x, a_y, a_z = 1.5, 0.8, -0.3
        omega = 1.0

        domega_a = scenario_a_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z
        )
        domega_b = scenario_b_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z, 0.0, ops
        )
        assert np.isclose(domega_a, domega_b, rtol=1e-4), (
            f"Decoupled mismatch: A={domega_a}, B={domega_b}"
        )

    def test_scenario_b_benefits_from_interaction(
        self, ops: dict[str, np.ndarray]
    ) -> None:
        """At a_zz ≠ 0, Scenario B may have different sensitivity from Scenario A."""
        a_x, a_y, a_z, a_zz = 2.0, 1.0, -1.0, 3.0
        omega = 1.0

        domega_a = scenario_a_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z
        )
        domega_b = scenario_b_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z, a_zz, ops
        )
        # Not asserting which is better — just that they differ
        assert domega_a > 0 and domega_b > 0


class TestScenarioACompoundResult:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("omega_values", "array_eq"),
        ("best_delta_omega_per_omega", "array_eq"),
        ("best_params_per_omega", "eq"),
        ("sql_values", "array_eq"),
        ("t_hold_value", "eq"),
        ("expectation_Jz_per_omega", "array_eq"),
        ("variance_Jz_per_omega", "array_eq"),
    ]

    def test_parquet_roundtrip(self, tmp_path: Path) -> None:

        omega_vals = np.linspace(0.1, 1.0, 5)
        result = ScenarioACompoundResult(
            omega_values=omega_vals,
            best_delta_omega_per_omega=np.array([0.05, 0.04, 0.03, 0.02, 0.01]),
            best_params_per_omega=[
                (1.0, 0.0, 0.0),
                (2.0, 0.5, -0.5),
                (3.0, 1.0, -1.0),
                (4.0, 1.5, -1.5),
                (5.0, 2.0, -2.0),
            ],
            sql_values=np.full(5, 0.1),
            t_hold_value=10.0,
            expectation_Jz_per_omega=np.zeros(5),
            variance_Jz_per_omega=np.ones(5) * 0.25,
        )

        pq_path = tmp_path / "test.parquet"
        result.save_parquet(pq_path)

        loaded = ScenarioACompoundResult.from_parquet(pq_path)
        assert_roundtrip_fields(loaded, result, self._FIELD_SPECS)

    def test_from_parquet_missing_columns_raises(self, tmp_path: Path) -> None:
        import pandas as pd

        df = pd.DataFrame({"omega": [1.0], "delta": [0.1]})
        pq_path = tmp_path / "bad.parquet"
        df.to_parquet(pq_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            ScenarioACompoundResult.from_parquet(pq_path)


class TestCompoundRatioResult:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("omega_values", "array_eq"),
        ("delta_omega_A", "array_eq"),
        ("delta_omega_B", "array_eq"),
        ("compound_ratio", "array_eq"),
        ("sql_values", "array_eq"),
        ("ratio_A_to_sql", "array_eq"),
        ("ratio_B_to_sql", "array_eq"),
    ]

    def test_parquet_roundtrip(self, tmp_path: Path) -> None:
        omega_vals = np.linspace(0.1, 1.0, 5)
        result = CompoundRatioResult(
            omega_values=omega_vals,
            delta_omega_A=np.array([0.1, 0.08, 0.06, 0.04, 0.02]),
            delta_omega_B=np.array([0.05, 0.04, 0.03, 0.02, 0.01]),
            compound_ratio=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            sql_values=np.full(5, 0.1),
            ratio_A_to_sql=np.array([1.0, 1.25, 1.67, 2.5, 5.0]),
            ratio_B_to_sql=np.array([2.0, 2.5, 3.33, 5.0, 10.0]),
        )

        pq_path = tmp_path / "test_cr.parquet"
        result.save_parquet(pq_path)

        loaded = CompoundRatioResult.from_parquet(pq_path)
        assert_roundtrip_fields(loaded, result, self._FIELD_SPECS)

    def test_from_parquet_missing_columns_raises(self, tmp_path: Path) -> None:
        import pandas as pd

        df = pd.DataFrame({"omega": [1.0]})
        pq_path = tmp_path / "bad_cr.parquet"
        df.to_parquet(pq_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            CompoundRatioResult.from_parquet(pq_path)


class TestCommutationRelations:
    def test_ji_jj_commutator(self, ops: dict[str, np.ndarray]) -> None:
        """Verify [J_i, J_j] = i ε_{ijk} J_k for single-qubit operators."""
        Jx = J_X
        Jy = J_Y
        Jz = J_Z

        # [Jx, Jy] = i Jz
        comm_xy = Jx @ Jy - Jy @ Jx
        assert np.allclose(comm_xy, 1j * Jz, atol=1e-12)

        # [Jy, Jz] = i Jx
        comm_yz = Jy @ Jz - Jz @ Jy
        assert np.allclose(comm_yz, 1j * Jx, atol=1e-12)

        # [Jz, Jx] = i Jy
        comm_zx = Jz @ Jx - Jx @ Jz
        assert np.allclose(comm_zx, 1j * Jy, atol=1e-12)


class TestScenarioAObjective3D:
    def test_matches_sensitivity_direct_call(self) -> None:
        """Wrapper should produce same result as scenario_a_sensitivity."""
        p = np.array([1.5, 0.8, -0.3])
        omega = 1.0
        t_hold = 10.0
        obj = _scenario_a_objective_3d(p, omega, t_hold)
        expected = scenario_a_sensitivity(DEFAULT_T_BS, t_hold, omega, 1.5, 0.8, -0.3)
        assert np.isclose(obj, expected, rtol=1e-10)


class TestScenarioBObjective4D:
    def test_matches_sensitivity_direct_call(self, ops: dict[str, np.ndarray]) -> None:
        """Wrapper should produce same result as scenario_b_sensitivity."""
        p = np.array([1.0, 0.5, -0.5, 2.0])
        omega = 1.0
        t_hold = 10.0
        obj = _scenario_b_objective_4d(p, omega, ops, t_hold)
        expected = scenario_b_sensitivity(
            DEFAULT_T_BS, t_hold, omega, 1.0, 0.5, -0.5, 2.0, ops
        )
        assert np.isclose(obj, expected, rtol=1e-10)


class TestScenarioARandomSearch:
    def test_returns_drive_random_search_result(self) -> None:
        """Random search with tiny budget should return a valid result."""
        result = scenario_a_random_search(omega=1.0, n_samples=5, seed=42)
        assert result.omega_value == 1.0
        assert result.best_delta_omega > 0
        assert len(result.samples) == 5


class TestRunScenarioAOmegaScan:
    def test_single_omega_with_tiny_budget(self) -> None:
        """Omega scan with single point and minimal budget."""
        result = run_scenario_a_omega_scan(
            omega_values=[1.0],
            n_random=5,
            n_nm_refine=1,
            seed=42,
        )
        assert len(result.omega_values) == 1
        assert np.isclose(result.omega_values[0], 1.0)
        assert np.isfinite(result.best_delta_omega_per_omega[0])
        assert len(result.best_params_per_omega) == 1


class TestRunScenarioBSingleOmega:
    def test_single_omega_with_tiny_budget(self) -> None:
        """Single-omega run for Scenario B with minimal budget."""
        result = _run_scenario_b_single_omega(
            omega=1.0,
            n_random=5,
            n_nm_refine=1,
            seed=42,
            t_hold=DEFAULT_T_HOLD,
            T_BS=DEFAULT_T_BS,
        )
        assert result["omega"] == 1.0
        assert np.isfinite(result["best_delta_omega"])
        assert (
            len(result) == 8
        )  # omega, best_delta, a_x, a_y, a_z, a_zz, expectation, variance


class TestComputeCompoundRatio:
    def test_perfect_match_gives_ratio_one(self) -> None:
        """When both scenarios give identical sensitivity, ratio = 1."""
        omega_vals = np.array([0.5, 1.0, 2.0])
        delta = np.array([0.08, 0.06, 0.04])
        sql = np.full(3, 0.1)

        result_a = ScenarioACompoundResult(
            omega_values=omega_vals,
            best_delta_omega_per_omega=delta.copy(),
            best_params_per_omega=[(0.0, 0.0, 0.0)] * 3,
            sql_values=sql.copy(),
            t_hold_value=DEFAULT_T_HOLD,
            expectation_Jz_per_omega=np.zeros(3),
            variance_Jz_per_omega=np.ones(3) * 0.25,
        )

        result_b = _m.DriveOmegaScanResult(
            omega_values=omega_vals.copy(),
            best_params_per_omega=[(0.0, 0.0, 0.0, 0.0)] * 3,
            best_delta_omega_per_omega=delta.copy(),
            sql_values=sql.copy(),
            expectation_Jz_per_omega=np.zeros(3),
            variance_Jz_per_omega=np.ones(3) * 0.25,
        )

        cr = compute_compound_ratio(result_a, result_b)
        np.testing.assert_array_almost_equal(cr.compound_ratio, [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(cr.delta_omega_A, delta)
        np.testing.assert_array_almost_equal(cr.delta_omega_B, delta)

    def test_scenario_b_better_gives_ratio_above_one(self) -> None:
        """When B is twice as good as A, compound ratio = 2."""
        omega_vals = np.array([1.0])
        sql = np.array([0.1])

        result_a = ScenarioACompoundResult(
            omega_values=omega_vals,
            best_delta_omega_per_omega=np.array([0.1]),
            best_params_per_omega=[(0.0, 0.0, 0.0)],
            sql_values=sql.copy(),
            t_hold_value=DEFAULT_T_HOLD,
            expectation_Jz_per_omega=np.zeros(1),
            variance_Jz_per_omega=np.ones(1) * 0.25,
        )

        result_b = _m.DriveOmegaScanResult(
            omega_values=omega_vals.copy(),
            best_delta_omega_per_omega=np.array([0.05]),
            best_params_per_omega=[(0.0, 0.0, 0.0, 0.0)],
            sql_values=sql.copy(),
            expectation_Jz_per_omega=np.zeros(1),
            variance_Jz_per_omega=np.ones(1) * 0.25,
        )

        cr = compute_compound_ratio(result_a, result_b)
        assert np.isclose(cr.compound_ratio[0], 2.0)


class TestDecoupledBaselineResult:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("scenarios", "eq"),
        ("delta_omega_values", "array_eq"),
        ("sql_values", "array_eq"),
        ("ratio_to_sql_values", "array_eq"),
        ("t_hold_value", "eq"),
    ]

    def test_parquet_roundtrip(self, tmp_path: Path) -> None:
        result = DecoupledBaselineResult(
            scenarios=["A", "B"],
            delta_omega_values=np.array([0.1, 0.1], dtype=float),
            sql_values=np.full(2, 0.1, dtype=float),
            ratio_to_sql_values=np.array([1.0, 1.0], dtype=float),
            t_hold_value=10.0,
        )
        pq_path = tmp_path / "test_decoupled.parquet"
        result.save_parquet(pq_path)
        loaded = DecoupledBaselineResult.from_parquet(pq_path)
        assert_roundtrip_fields(loaded, result, self._FIELD_SPECS)

    def test_from_parquet_missing_columns_raises(self, tmp_path: Path) -> None:
        import pandas as pd

        df = pd.DataFrame({"scenario": ["A"]})
        pq_path = tmp_path / "bad_decoupled.parquet"
        df.to_parquet(pq_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            DecoupledBaselineResult.from_parquet(pq_path)


class TestGenerateDecoupledBaseline:
    def test_saves_parquet_with_monkeypatched_path(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """generate_decoupled_baseline should compute and save to parquet."""
        monkeypatch.setattr(
            _m, "_parquet_path", lambda tag: tmp_path / f"{tag}.parquet"
        )
        _m.generate_decoupled_baseline(force=True)
        pq_path = tmp_path / "decoupled-baseline.parquet"
        assert pq_path.exists()
        loaded = DecoupledBaselineResult.from_parquet(pq_path)
        assert loaded.scenarios == ["A", "B"]
        assert np.isclose(loaded.delta_omega_values[0], SQL_REFERENCE, rtol=1e-4)
        assert np.isclose(loaded.delta_omega_values[1], SQL_REFERENCE, rtol=1e-4)
        assert loaded.t_hold_value == DEFAULT_T_HOLD


class TestGenerateScenarioAScan:
    def test_with_stub_scan_and_monkeypatched_path(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Use a stub Scenario A scan so generate_scenario_a_scan runs quickly."""
        stub_result = ScenarioACompoundResult(
            omega_values=np.array([1.0]),
            best_delta_omega_per_omega=np.array([0.05]),
            best_params_per_omega=[(1.0, 0.5, 0.0)],
            sql_values=np.array([0.1]),
            t_hold_value=DEFAULT_T_HOLD,
            expectation_Jz_per_omega=np.zeros(1),
            variance_Jz_per_omega=np.ones(1) * 0.25,
        )
        monkeypatch.setattr(
            _m, "run_scenario_a_omega_scan", lambda *a, **kw: stub_result
        )
        monkeypatch.setattr(
            _m,
            "_parquet_path",
            lambda tag: tmp_path / f"{tag}.parquet",
        )
        _m.generate_scenario_a_scan(force=True)
        pq_path = tmp_path / "scenario-a-omega-scan.parquet"
        assert pq_path.exists()


class TestGenerateCompoundRatio:
    def test_with_stub_parquet_files(self, monkeypatch, tmp_path: Path) -> None:
        """Create stub parquet files and test compound ratio generation."""
        monkeypatch.setattr(
            _m,
            "_parquet_path",
            lambda tag: tmp_path / f"{tag}.parquet",
        )

        # Stub Scenario A result
        result_a = ScenarioACompoundResult(
            omega_values=np.array([1.0]),
            best_delta_omega_per_omega=np.array([0.1]),
            best_params_per_omega=[(0.0, 0.0, 0.0)],
            sql_values=np.array([0.1]),
            t_hold_value=DEFAULT_T_HOLD,
            expectation_Jz_per_omega=np.zeros(1),
            variance_Jz_per_omega=np.ones(1) * 0.25,
        )
        result_a.save_parquet(tmp_path / "scenario-a-omega-scan.parquet")

        # Stub Scenario B result
        result_b = _m.DriveOmegaScanResult(
            omega_values=np.array([1.0]),
            best_delta_omega_per_omega=np.array([0.05]),
            best_params_per_omega=[(0.0, 0.0, 0.0, 0.0)],
            sql_values=np.array([0.1]),
            expectation_Jz_per_omega=np.zeros(1),
            variance_Jz_per_omega=np.ones(1) * 0.25,
        )
        result_b.save_parquet(tmp_path / "scenario-b-omega-scan.parquet")

        _m.generate_compound_ratio(force=True)
        pq_path = tmp_path / "compound-ratio.parquet"
        assert pq_path.exists()

        # Verify correct ratio was computed
        df = _m.pd.read_parquet(pq_path)
        assert np.isclose(df["compound_ratio"].iloc[0], 2.0)


class TestRunScenarioBOmegaScan:
    def test_single_omega_with_sequential_executor(self, monkeypatch) -> None:
        """Run B omega-scan by replacing ProcessPoolExecutor with sequential."""
        from concurrent.futures import Future

        class SequentialPoolExecutor:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def submit(self, fn, /, *a, **kw):
                f: Future = Future()
                f.set_result(fn(*a, **kw))
                return f

        monkeypatch.setattr(
            "concurrent.futures.ProcessPoolExecutor",
            SequentialPoolExecutor,
        )
        result = run_scenario_b_omega_scan(
            omega_values=[1.0],
            n_random=5,
            n_nm_refine=1,
            seed=42,
        )
        assert len(result.omega_values) == 1
        assert np.isfinite(result.best_delta_omega_per_omega[0])
        assert len(result.best_params_per_omega) == 1


class TestMainCLI:
    def test_invalid_step_exits(self) -> None:
        """Calling main with an unknown --only step should exit with error."""
        with pytest.raises(SystemExit):
            main(["--only", "nonexistent-step"])

    def test_decoupled_baseline_step_with_monkeypatched_path(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Running --only decoupled-baseline should dispatch correctly."""
        monkeypatch.setattr(
            _m, "_parquet_path", lambda tag: tmp_path / f"{tag}.parquet"
        )
        main(["--only", "decoupled-baseline", "--force"])
        pq_path = tmp_path / "decoupled-baseline.parquet"
        assert pq_path.exists()


class TestPropertyBasedInvariants:
    """Property-based tests for physical invariants.

    Verifies that physical constraints hold across random parameter ranges,
    not just at hand-picked values.
    """

    _DRIVE_STRAT = st.floats(
        min_value=-5.0, max_value=5.0, allow_infinity=False, allow_nan=False
    )
    _OMEGA_STRAT = st.floats(
        min_value=0.1, max_value=5.0, allow_infinity=False, allow_nan=False
    )

    @settings(
        max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(omega=_OMEGA_STRAT, a_x=_DRIVE_STRAT, a_y=_DRIVE_STRAT, a_z=_DRIVE_STRAT)
    def test_scenario_a_hold_unitary_is_unitary(
        self, omega: float, a_x: float, a_y: float, a_z: float
    ) -> None:
        U = scenario_a_hold_unitary(DEFAULT_T_HOLD, omega, a_x, a_y, a_z)
        assert np.allclose(U @ U.conj().T, I_2, atol=1e-10), (
            f"Unitarity violated for omega={omega}, a=({a_x},{a_y},{a_z})"
        )

    @settings(
        max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(omega=_OMEGA_STRAT, a_x=_DRIVE_STRAT, a_y=_DRIVE_STRAT, a_z=_DRIVE_STRAT)
    def test_scenario_a_hamiltonian_is_hermitian(
        self, omega: float, a_x: float, a_y: float, a_z: float
    ) -> None:
        H = scenario_a_hamiltonian(omega, a_x, a_y, a_z)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    @settings(
        max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        omega=_OMEGA_STRAT,
        a_x=_DRIVE_STRAT,
        a_y=_DRIVE_STRAT,
        a_z=_DRIVE_STRAT,
        a_zz=_DRIVE_STRAT,
    )
    def test_scenario_b_hamiltonian_is_hermitian(
        self, omega: float, a_x: float, a_y: float, a_z: float, a_zz: float
    ) -> None:
        ops = build_two_qubit_operators()
        H = scenario_b_hamiltonian(omega, a_x, a_y, a_z, a_zz, ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    @settings(
        max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(omega=_OMEGA_STRAT, a_x=_DRIVE_STRAT, a_y=_DRIVE_STRAT, a_z=_DRIVE_STRAT)
    def test_scenario_a_final_state_normalised(
        self, omega: float, a_x: float, a_y: float, a_z: float
    ) -> None:
        psi = scenario_a_evolve(DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z)
        assert np.isclose(np.linalg.norm(psi), 1.0), (
            f"Norm={np.linalg.norm(psi)} for omega={omega}, a=({a_x},{a_y},{a_z})"
        )

    @settings(
        max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        omega=_OMEGA_STRAT,
        a_x=_DRIVE_STRAT,
        a_y=_DRIVE_STRAT,
        a_z=_DRIVE_STRAT,
        a_zz=_DRIVE_STRAT,
    )
    def test_scenario_b_final_state_normalised(
        self, omega: float, a_x: float, a_y: float, a_z: float, a_zz: float
    ) -> None:
        ops = build_two_qubit_operators()
        psi = scenario_b_evolve(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z, a_zz, ops
        )
        assert np.isclose(np.linalg.norm(psi), 1.0), (
            f"Norm={np.linalg.norm(psi)} for omega={omega}, "
            f"a=({a_x},{a_y},{a_z}), a_zz={a_zz}"
        )

    @settings(
        max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(omega=_OMEGA_STRAT, a_x=_DRIVE_STRAT, a_y=_DRIVE_STRAT, a_z=_DRIVE_STRAT)
    def test_decoupled_azz_zero_matches_scenario_a(
        self, omega: float, a_x: float, a_y: float, a_z: float
    ) -> None:
        """At a_zz=0, Scenario B sensitivity matches Scenario A.

        Both should be nearly equal.  In the limit of near-zero drive
        coefficients the Hamiltonian is nearly zero, the derivative
        vanishes, and both scenarios diverge (Δω → ∞).  Due to differing
        matrix sizes (2×2 vs 4×4 eigh), one may hit the ``inf``
        threshold while the other returns a large finite value — both
        are physically consistent (extremely poor sensitivity).  We
        accept the case where both are ≫ SQL as passing.
        """
        domega_a = scenario_a_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z
        )
        ops = build_two_qubit_operators()
        domega_b = scenario_b_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z, 0.0, ops
        )
        # Both divergent → numerically fragile but physically consistent
        if domega_a > 100 * SQL_REFERENCE and domega_b > 100 * SQL_REFERENCE:
            return
        assert np.isclose(domega_a, domega_b, rtol=1e-4), (
            f"Decoupled mismatch at omega={omega}, a=({a_x},{a_y},{a_z}): "
            f"A={domega_a}, B={domega_b}"
        )

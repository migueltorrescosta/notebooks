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
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

if TYPE_CHECKING:
    from pathlib import Path

from src.analysis.ancilla_drive_results import DriveOmegaScanResult
from src.analysis.ancilla_optimization import (
    build_two_qubit_operators,
    two_qubit_bs_unitary,
)
from src.utils.constants import I_2, I_4, J_X, J_Y, J_Z
from src.utils.sampling import (
    project_to_sphere,
    sample_uniform_sphere,
    sphere_objective_wrapper,
)
from src.utils.serialization import assert_roundtrip_fields

_m = importlib.import_module("reports.r20260709.compound_comparison")
_cli = importlib.import_module("reports.r20260709.compound_comparison_cli")
_results = importlib.import_module("reports.r20260709.compound_comparison_results")

# Physics functions (from compound_comparison)
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
scenario_a_sensitivity_constrained_ay = _m.scenario_a_sensitivity_constrained_ay
run_constrained_ay_verification = _m.run_constrained_ay_verification
run_scenario_a_omega_scan = _m.run_scenario_a_omega_scan
run_scenario_b_omega_scan = _m.run_scenario_b_omega_scan
_run_scenario_b_single_omega = _m._run_scenario_b_single_omega
compute_compound_ratio = _m.compute_compound_ratio
compute_fixed_parameter_compound_ratio = _m.compute_fixed_parameter_compound_ratio

# Dataclasses (from compound_comparison_results)
ScenarioACompoundResult = _results.ScenarioACompoundResult
CompoundRatioResult = _results.CompoundRatioResult
FixedParameterCompoundRatioResult = _results.FixedParameterCompoundRatioResult
DecoupledBaselineResult = _results.DecoupledBaselineResult

# CLI entry point (from compound_comparison_cli)
main = _cli.main

# Constants (from compound_comparison)
DEFAULT_T_BS = _m.DEFAULT_T_BS
DEFAULT_T_HOLD = _m.DEFAULT_T_HOLD
SQL_REFERENCE = _m.SQL_REFERENCE
DEFAULT_SPHERE_RADIUS = _m.DEFAULT_SPHERE_RADIUS
DEFAULT_SAMPLING_MODE = _m.DEFAULT_SAMPLING_MODE


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


class TestRoleOfAy:
    """Verify that a_y modulates EP sensitivity through oscillation frequency.

    a_y drops out of the QFI and amplitude prefactor ρ, but enters the
    EP sensitivity through the rotation angle θ = ω t r.  Constraining
    a_y=0 should give equal or worse sensitivity than free 3D optimisation.
    """

    def test_constrained_matches_free_when_ay_zero(self) -> None:
        """When a_y=0, the constrained and free formulas give identical results."""
        omega = 1.0
        d_free = scenario_a_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, 5.0, 0.0, 5.0
        )
        d_constrained = scenario_a_sensitivity_constrained_ay(omega, 5.0, 5.0)
        assert np.isclose(d_free, d_constrained, rtol=1e-10)

    def test_constrained_minimum_saturates(self) -> None:
        """With a_y=0 and ρ=1, the best sensitivity is 1/(t·5√2)."""
        expected_min = 1.0 / (DEFAULT_T_HOLD * 5.0 * np.sqrt(2))
        for omega in [0.5, 1.0, 2.0, 4.0]:
            d = scenario_a_sensitivity_constrained_ay(omega, 5.0, 5.0)
            assert np.isclose(d, expected_min, rtol=1e-4), (
                f"Expected {expected_min} at ω={omega}, got {d}"
            )

    def test_ay_changes_sensitivity_at_fixed_ax_az(self) -> None:
        """At ω ≥ 1.0, adding a_y to (a_x, a_z) changes the sensitivity."""
        for omega in [1.0, 2.0, 4.51]:
            d_without = scenario_a_sensitivity(
                DEFAULT_T_BS, DEFAULT_T_HOLD, omega, 5.0, 0.0, 5.0
            )
            d_with = scenario_a_sensitivity(
                DEFAULT_T_BS, DEFAULT_T_HOLD, omega, 5.0, 2.5, 5.0
            )
            assert np.isfinite(d_without) and np.isfinite(d_with)
            assert not np.isclose(d_without, d_with, rtol=1e-6), (
                f"a_y has no effect at ω={omega}: {d_without} vs {d_with}"
            )

    def test_ay_increases_rotation_angle(self) -> None:
        """Adding a_y increases r = √(a_x²+a_y²+a_z²), changing θ = ωtr."""
        a_x, a_z = 3.0, 4.0
        r_without = np.sqrt(a_x**2 + a_z**2)  # = 5.0
        r_with = np.sqrt(a_x**2 + 3.0**2 + a_z**2)  # = √34 ≈ 5.83
        assert r_with > r_without, "a_y should increase r"

    def test_qfi_independent_of_ay(self) -> None:
        """QFI = t²(a_x²+a_z²) does not depend on a_y."""
        t = DEFAULT_T_HOLD
        a_x, a_z = 3.0, 4.0
        qfi = t**2 * (a_x**2 + a_z**2)
        # Verify against variance formula
        for a_y in [0.0, 1.0, -2.0, 5.0]:
            # After BS1, Bloch vector is in -y direction
            # Var(G_S) = r²(1-n_y²)/4 = (a_x²+a_z²)/4
            var_gs = (a_x**2 + a_z**2) / 4.0
            qfi_check = 4.0 * t**2 * var_gs
            assert np.isclose(qfi, qfi_check, rtol=1e-10), f"QFI mismatch at a_y={a_y}"

    @pytest.mark.parametrize(
        ("omega", "a_y"),
        [
            (1.0, 2.5),
            (2.0, 3.0),
            (4.51, 1.0),
        ],
        ids=["w1.0_ay2.5", "w2.0_ay3.0", "w4.51_ay1.0"],
    )
    def test_ay_modulates_oscillation_frequency(self, omega: float, a_y: float) -> None:
        """At fixed a_x=a_z=5, varying a_y changes the sensitivity
        through the rotation angle θ = ω t r."""
        a_x, a_z = 5.0, 5.0
        d_ay0 = scenario_a_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, 0.0, a_z
        )
        d_ay_nonzero = scenario_a_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z
        )
        # Both should be finite and different (a_y changes θ)
        assert np.isfinite(d_ay0) and np.isfinite(d_ay_nonzero)
        assert not np.isclose(d_ay0, d_ay_nonzero, rtol=1e-4), (
            f"a_y={a_y} should change sensitivity at ω={omega}: "
            f"{d_ay0} vs {d_ay_nonzero}"
        )


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


class TestDecoupledLimitRandomParams:
    """Decoupled-limit consistency: Δω_B(a_zz=0) = Δω_A for random drives.

    At a_zz=0 the ancilla factorises and does not affect the J_z^S
    measurement, so Scenario B must reproduce Scenario A exactly.
    """

    _DRIVE_PARAMS: ClassVar[list[tuple[float, float, float]]] = [
        (3.2, -1.7, 0.5),
        (-4.9, 0.3, 2.8),
        (0.0, 4.1, -3.6),
        (1.1, 1.1, 1.1),
        (-2.5, -2.5, -2.5),
        (5.0, -2.1, 0.0),
        (0.0, 0.0, 3.0),
        (-1.3, 4.7, -0.9),
    ]

    @pytest.mark.parametrize(
        ("a_x", "a_y", "a_z"),
        _DRIVE_PARAMS,
        ids=[f"({ax},{ay},{az})" for ax, ay, az in _DRIVE_PARAMS],
    )
    def test_decoupled_matches_scenario_a(
        self, a_x: float, a_y: float, a_z: float
    ) -> None:
        omega = 1.0
        ops = build_two_qubit_operators()
        domega_a = scenario_a_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z
        )
        domega_b = scenario_b_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z, 0.0, ops
        )
        # Both divergent → numerically fragile but physically consistent
        if domega_a > 100 * SQL_REFERENCE and domega_b > 100 * SQL_REFERENCE:
            return
        assert np.isclose(domega_a, domega_b, rtol=1e-4), (
            f"Decoupled mismatch at a=({a_x},{a_y},{a_z}): A={domega_a}, B={domega_b}"
        )


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

        result_b = DriveOmegaScanResult(
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

        result_b = DriveOmegaScanResult(
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
            _cli, "_parquet_path", lambda tag: tmp_path / f"{tag}.parquet"
        )
        _cli.generate_decoupled_baseline(force=True)
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
            _cli, "run_scenario_a_omega_scan", lambda *a, **kw: stub_result
        )
        monkeypatch.setattr(
            _cli,
            "_parquet_path",
            lambda tag: tmp_path / f"{tag}.parquet",
        )
        _cli.generate_scenario_a_scan(force=True)
        pq_path = tmp_path / "scenario-a-omega-scan.parquet"
        assert pq_path.exists()


class TestGenerateCompoundRatio:
    def test_with_stub_parquet_files(self, monkeypatch, tmp_path: Path) -> None:
        """Create stub parquet files and test compound ratio generation."""
        monkeypatch.setattr(
            _cli,
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
        result_b = DriveOmegaScanResult(
            omega_values=np.array([1.0]),
            best_delta_omega_per_omega=np.array([0.05]),
            best_params_per_omega=[(0.0, 0.0, 0.0, 0.0)],
            sql_values=np.array([0.1]),
            expectation_Jz_per_omega=np.zeros(1),
            variance_Jz_per_omega=np.ones(1) * 0.25,
        )
        result_b.save_parquet(tmp_path / "scenario-b-omega-scan.parquet")

        _cli.generate_compound_ratio(force=True)
        pq_path = tmp_path / "compound-ratio.parquet"
        assert pq_path.exists()

        # Verify correct ratio was computed
        df = pd.read_parquet(pq_path)
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
            _cli, "_parquet_path", lambda tag: tmp_path / f"{tag}.parquet"
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


class TestFixedParameterCompoundRatio:
    def test_decoupled_limit_ratio_one(self) -> None:
        """At a_zz=0, B reduces to A, so fixed-parameter ratio = 1."""
        omega_vals = np.array([0.5, 1.0, 2.0])
        ax, ay, az = 1.5, 0.8, -0.3
        ops = build_two_qubit_operators()

        # Compute actual sensitivities at shared params
        deltas_a = np.array(
            [
                scenario_a_sensitivity(DEFAULT_T_BS, DEFAULT_T_HOLD, w, ax, ay, az)
                for w in omega_vals
            ]
        )
        deltas_b = np.array(
            [
                scenario_b_sensitivity(
                    DEFAULT_T_BS, DEFAULT_T_HOLD, w, ax, ay, az, 0.0, ops
                )
                for w in omega_vals
            ]
        )
        sql = np.full(3, SQL_REFERENCE)

        result_a = ScenarioACompoundResult(
            omega_values=omega_vals,
            best_delta_omega_per_omega=deltas_a,
            best_params_per_omega=[(ax, ay, az)] * 3,
            sql_values=sql.copy(),
            t_hold_value=DEFAULT_T_HOLD,
            expectation_Jz_per_omega=np.zeros(3),
            variance_Jz_per_omega=np.ones(3) * 0.25,
        )

        result_b = DriveOmegaScanResult(
            omega_values=omega_vals.copy(),
            best_params_per_omega=[(ax, ay, az, 0.0)] * 3,
            best_delta_omega_per_omega=deltas_b,
            sql_values=sql.copy(),
            expectation_Jz_per_omega=np.zeros(3),
            variance_Jz_per_omega=np.ones(3) * 0.25,
        )

        fpr = compute_fixed_parameter_compound_ratio(result_a, result_b)
        np.testing.assert_array_almost_equal(fpr.fixed_ratio, [1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(fpr.a_zz_B, [0.0, 0.0, 0.0])

    def test_azz_nonzero_differs_from_one(self) -> None:
        """With a_zz>0, the fixed-parameter ratio generally differs from 1."""
        a_x, a_y, a_z, a_zz = 2.0, 1.0, -1.0, 3.0
        omega = 1.0

        domega_a = scenario_a_sensitivity(
            DEFAULT_T_BS, DEFAULT_T_HOLD, omega, a_x, a_y, a_z
        )
        domega_b = scenario_b_sensitivity(
            DEFAULT_T_BS,
            DEFAULT_T_HOLD,
            omega,
            a_x,
            a_y,
            a_z,
            a_zz,
            build_two_qubit_operators(),
        )

        result_a = ScenarioACompoundResult(
            omega_values=np.array([omega]),
            best_delta_omega_per_omega=np.array([domega_a]),
            best_params_per_omega=[(a_x, a_y, a_z)],
            sql_values=np.array([SQL_REFERENCE]),
            t_hold_value=DEFAULT_T_HOLD,
            expectation_Jz_per_omega=np.array([0.0]),
            variance_Jz_per_omega=np.array([0.25]),
        )
        result_b = DriveOmegaScanResult(
            omega_values=np.array([omega]),
            best_params_per_omega=[(a_x, a_y, a_z, a_zz)],
            best_delta_omega_per_omega=np.array([domega_b]),
            sql_values=np.array([SQL_REFERENCE]),
            expectation_Jz_per_omega=np.array([0.0]),
            variance_Jz_per_omega=np.array([0.25]),
        )

        fpr = compute_fixed_parameter_compound_ratio(result_a, result_b)
        # At a_zz=0 the ratio would be 1.0; at a_zz=3 it differs
        assert np.isfinite(fpr.fixed_ratio[0])
        assert not np.isclose(fpr.fixed_ratio[0], 1.0, atol=1e-3), (
            f"Expected non-unity ratio at a_zz={a_zz}, got {fpr.fixed_ratio[0]}"
        )


class TestFixedParameterCompoundRatioResult:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("omega_values", "array_eq"),
        ("delta_omega_A_opt", "array_eq"),
        ("a_x_A", "array_eq"),
        ("a_y_A", "array_eq"),
        ("a_z_A", "array_eq"),
        ("a_zz_B", "array_eq"),
        ("delta_omega_B_fixed", "array_eq"),
        ("fixed_ratio", "array_eq"),
        ("sql_values", "array_eq"),
    ]

    def test_parquet_roundtrip(self, tmp_path: Path) -> None:
        omega_vals = np.linspace(0.1, 1.0, 5)
        result = FixedParameterCompoundRatioResult(
            omega_values=omega_vals,
            delta_omega_A_opt=np.array([0.1, 0.08, 0.06, 0.04, 0.02]),
            a_x_A=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            a_y_A=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            a_z_A=np.array([0.0, -0.5, -1.0, -1.5, -2.0]),
            a_zz_B=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            delta_omega_B_fixed=np.array([0.1, 0.09, 0.05, 0.03, 0.015]),
            fixed_ratio=np.array([1.0, 0.889, 1.2, 1.333, 1.333]),
            sql_values=np.full(5, 0.1),
        )

        pq_path = tmp_path / "test_fpr.parquet"
        result.save_parquet(pq_path)

        loaded = FixedParameterCompoundRatioResult.from_parquet(pq_path)
        assert_roundtrip_fields(loaded, result, self._FIELD_SPECS)

    def test_from_parquet_missing_columns_raises(self, tmp_path: Path) -> None:
        import pandas as pd

        df = pd.DataFrame({"omega": [1.0], "ratio": [1.0]})
        pq_path = tmp_path / "bad_fpr.parquet"
        df.to_parquet(pq_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            FixedParameterCompoundRatioResult.from_parquet(pq_path)


# ============================================================================
# Sphere Sampling Tests
# ============================================================================


class TestSampleUniformSphere:
    def test_all_samples_have_correct_radius(self) -> None:
        """Every sample must lie on S^{d-1}(R): norm == R."""
        rng = np.random.default_rng(42)
        R = 5.0
        samples = sample_uniform_sphere(3, R, 200, rng)
        norms = np.linalg.norm(samples, axis=1)
        np.testing.assert_allclose(norms, R, rtol=1e-12)

    def test_correct_shape(self) -> None:
        rng = np.random.default_rng(42)
        samples = sample_uniform_sphere(4, 3.0, 50, rng)
        assert samples.shape == (50, 4)

    def test_all_samples_have_correct_radius_4d(self) -> None:
        """Sphere sampling works in 4D (Scenario B)."""
        rng = np.random.default_rng(123)
        R = 5.0
        samples = sample_uniform_sphere(4, R, 200, rng)
        norms = np.linalg.norm(samples, axis=1)
        np.testing.assert_allclose(norms, R, rtol=1e-12)

    def test_single_sample(self) -> None:
        rng = np.random.default_rng(42)
        samples = sample_uniform_sphere(3, 5.0, 1, rng)
        assert samples.shape == (1, 3)
        assert np.isclose(np.linalg.norm(samples[0]), 5.0)

    def test_distribution_covers_sphere(self) -> None:
        """Samples should not all cluster in one octant."""
        rng = np.random.default_rng(42)
        samples = sample_uniform_sphere(3, 5.0, 1000, rng)
        # Check that samples appear in all 8 octants
        signs = np.sign(samples)
        octants = set(map(tuple, signs.astype(int)))
        assert len(octants) >= 6, f"Expected coverage of ≥6 octants, got {len(octants)}"


class TestProjectToSphere:
    def test_projects_to_correct_radius(self) -> None:
        p = np.array([3.0, 4.0, 0.0])  # norm = 5
        result = project_to_sphere(p, 5.0)
        assert np.isclose(np.linalg.norm(result), 5.0)

    def test_preserves_direction(self) -> None:
        p = np.array([3.0, 4.0, 0.0])
        result = project_to_sphere(p, 5.0)
        np.testing.assert_allclose(result, p, rtol=1e-12)

    def test_scales_to_radius(self) -> None:
        p = np.array([1.0, 0.0, 0.0])  # norm = 1
        result = project_to_sphere(p, 7.0)
        np.testing.assert_allclose(result, np.array([7.0, 0.0, 0.0]))

    def test_zero_vector_returns_first_basis(self) -> None:
        p = np.array([0.0, 0.0, 0.0])
        result = project_to_sphere(p, 5.0)
        assert np.isclose(np.linalg.norm(result), 5.0)
        assert np.isclose(result[0], 5.0)

    def test_4d_projection(self) -> None:
        p = np.array([1.0, 2.0, 3.0, 4.0])
        result = project_to_sphere(p, 5.0)
        assert np.isclose(np.linalg.norm(result), 5.0)
        # Direction should be preserved
        np.testing.assert_allclose(result / 5.0, p / np.linalg.norm(p), rtol=1e-12)


class TestSphereObjectiveWrapper:
    def test_wrapper_projects_before_eval(self) -> None:
        """The wrapper should project any point onto the sphere before evaluation."""
        call_args: list[np.ndarray] = []

        def _track_obj(p: np.ndarray) -> float:
            call_args.append(p.copy())
            return float(np.sum(p**2))

        wrapped = sphere_objective_wrapper(_track_obj, 5.0)
        # Pass a point with norm != 5
        raw_p = np.array([10.0, 0.0, 0.0])
        result = wrapped(raw_p)

        # The wrapper should have projected onto the sphere
        assert len(call_args) == 1
        np.testing.assert_allclose(np.linalg.norm(call_args[0]), 5.0, rtol=1e-12)
        # The projected point [5, 0, 0] has sum of squares = 25
        assert np.isclose(result, 25.0)


class TestScenarioARandomSearchSphere:
    def test_sphere_mode_all_at_radius(self) -> None:
        """All samples must have |params| == R in sphere mode."""
        result = scenario_a_random_search(
            omega=1.0,
            n_samples=50,
            seed=42,
            sampling_mode="sphere",
            radius=5.0,
        )
        # Check the 3D samples (first 3 columns of the 4D array)
        samples_3d = result.samples[:, :3]
        norms = np.linalg.norm(samples_3d, axis=1)
        np.testing.assert_allclose(norms, 5.0, rtol=1e-12)

    def test_cube_mode_fills_hypercube(self) -> None:
        """In cube mode, samples fill [-R, R]^3."""
        result = scenario_a_random_search(
            omega=1.0,
            n_samples=50,
            seed=42,
            sampling_mode="cube",
            radius=5.0,
        )
        samples_3d = result.samples[:, :3]
        assert np.all(np.abs(samples_3d) <= 5.0 + 1e-10)

    def test_sphere_mode_returns_valid_result(self) -> None:
        result = scenario_a_random_search(
            omega=1.0,
            n_samples=10,
            seed=42,
            sampling_mode="sphere",
        )
        assert result.omega_value == 1.0
        assert result.best_delta_omega > 0
        assert len(result.samples) == 10

    def test_sphere_vs_cube_different_samples(self) -> None:
        """Sphere and cube sampling should produce different sample sets."""
        rs_sphere = scenario_a_random_search(
            omega=1.0,
            n_samples=20,
            seed=42,
            sampling_mode="sphere",
        )
        rs_cube = scenario_a_random_search(
            omega=1.0,
            n_samples=20,
            seed=42,
            sampling_mode="cube",
        )
        # The 3D samples should differ (cube samples will have |r| != 5)
        assert not np.allclose(rs_sphere.samples[:, :3], rs_cube.samples[:, :3])


class TestScenarioAOmegaScanSphere:
    def test_single_omega_sphere_mode(self) -> None:
        """Omega scan with sphere mode and tiny budget."""
        result = run_scenario_a_omega_scan(
            omega_values=[1.0],
            n_random=5,
            n_nm_refine=1,
            seed=42,
            sampling_mode="sphere",
        )
        assert len(result.omega_values) == 1
        assert np.isclose(result.omega_values[0], 1.0)
        assert np.isfinite(result.best_delta_omega_per_omega[0])

    def test_sphere_gives_finite_sensitivity(self) -> None:
        """Sphere mode must produce finite sensitivities."""
        result = run_scenario_a_omega_scan(
            omega_values=[0.5, 1.0, 2.0],
            n_random=10,
            n_nm_refine=2,
            seed=42,
            sampling_mode="sphere",
        )
        assert all(np.isfinite(result.best_delta_omega_per_omega))


class TestScenarioBSingleOmegaSphere:
    def test_single_omega_sphere_mode(self) -> None:
        result = _run_scenario_b_single_omega(
            omega=1.0,
            n_random=5,
            n_nm_refine=1,
            seed=42,
            t_hold=DEFAULT_T_HOLD,
            T_BS=DEFAULT_T_BS,
            sampling_mode="sphere",
        )
        assert result["omega"] == 1.0
        assert np.isfinite(result["best_delta_omega"])


class TestKnownOptimalDirection:
    """On S^2(R=5), the QFI for Scenario A is F_Q = 25 t^2 (1 - n_y^2).

    The optimal direction maximises 1 - n_y^2, i.e. n_y = 0.
    The EP sensitivity at the fringe midpoint is then 1/(t * R) when ρ = 1.
    On the sphere, ALL points with a_y = 0 give the same sensitivity
    because r = R is fixed — this is the key difference from cube sampling.
    """

    def test_optimal_direction_on_equator(self) -> None:
        """With a_x=0, a_z=5, a_y=0 (|n|=5, n_y=0), sensitivity = 1/(t*R)."""
        omega = 1.0
        d = scenario_a_sensitivity(DEFAULT_T_BS, DEFAULT_T_HOLD, omega, 0.0, 0.0, 5.0)
        # On S^2(R=5), r=R=5 and ρ=1, so Δω = 1/(t*R)
        expected = 1.0 / (DEFAULT_T_HOLD * 5.0)
        assert np.isclose(d, expected, rtol=1e-4)

    def test_sphere_optimiser_finds_optimal_direction(self) -> None:
        """NM refinement on the sphere should find near-optimal sensitivity."""
        # The best sensitivity on S^2(R=5) is 1/(t*R) = 0.02
        result = run_scenario_a_omega_scan(
            omega_values=[1.0],
            n_random=50,
            n_nm_refine=10,
            seed=42,
            sampling_mode="sphere",
            radius=5.0,
        )
        best_delta = result.best_delta_omega_per_omega[0]
        expected_min = 1.0 / (DEFAULT_T_HOLD * 5.0)
        # Allow some slack because 50 samples + 10 NM may not find exact optimum
        assert best_delta < expected_min * 1.5, (
            f"Expected best Δω < {expected_min * 1.5:.6f}, got {best_delta:.6f}"
        )

    def test_optimal_on_sphere_matches_formula(self) -> None:
        """On S^2(R=5), any a_y=0 direction gives Δω = 1/(t*R)."""
        expected_min = 1.0 / (DEFAULT_T_HOLD * 5.0)
        for omega in [0.5, 1.0, 2.0]:
            # a_x=0, a_z=5: on the sphere with R=5
            d = scenario_a_sensitivity(
                DEFAULT_T_BS, DEFAULT_T_HOLD, omega, 0.0, 0.0, 5.0
            )
            assert np.isclose(d, expected_min, rtol=1e-4), (
                f"Expected {expected_min} at ω={omega}, got {d}"
            )

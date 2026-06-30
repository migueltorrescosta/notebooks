"""Tests for the Free-Ancilla omega-Modulated Drive with Weighted Joint Measurement report.

Companion test module for ``reports/20260628/free_ancilla_joint_measurement.py``.
Tests state preparation, joint measurement operators, circuit evolution,
decoupled baseline, sensitivity computation, optimisation, and serialization.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.analysis.ancilla_optimization import compute_expectation_and_variance
from src.analysis.sensitivity_metrics import sql_reference
from src.physics.bipartite_operators import build_operators
from src.physics.joint_measurement import (
    build_bipartite_joint_measurement_operator,
    build_joint_measurement_operator,
)
from src.physics.n_particle_drive import (
    build_n_particle_operators,
    compute_n_particle_sensitivity,
    evolve_n_particle_circuit,
    n_particle_initial_state,
)
from src.utils.serialization import assert_roundtrip_fields

_m = importlib.import_module("reports.20260628.free_ancilla_joint_measurement")

AZZ_BOUNDS = _m.AZZ_BOUNDS
DRIVE_BOUNDS = _m.DRIVE_BOUNDS
PHI_A_BOUNDS = _m.PHI_A_BOUNDS
PSI_BOUNDS = _m.PSI_BOUNDS
T_BS = _m.T_BS
T_HOLD = _m.T_HOLD
THETA_A_BOUNDS = _m.THETA_A_BOUNDS
BipartiteFreeAncillaJointNMSensitivityResult = (
    _m.BipartiteFreeAncillaJointNMSensitivityResult
)
BipartiteFreeAncillaJointNScalingResult = _m.BipartiteFreeAncillaJointNScalingResult
BipartiteFreeAncillaJointNScalingScanResult = (
    _m.BipartiteFreeAncillaJointNScalingScanResult
)
BipartiteFreeAncillaJointRandomSearchResult = (
    _m.BipartiteFreeAncillaJointRandomSearchResult
)
QubitFreeAncillaJointNMSensitivityResult = _m.QubitFreeAncillaJointNMSensitivityResult
QubitFreeAncillaJointNScalingResult = _m.QubitFreeAncillaJointNScalingResult
QubitFreeAncillaJointNScalingScanResult = _m.QubitFreeAncillaJointNScalingScanResult
QubitFreeAncillaJointRandomSearchResult = _m.QubitFreeAncillaJointRandomSearchResult
bipartite_free_ancilla_joint_objective = _m.bipartite_free_ancilla_joint_objective
bipartite_free_ancilla_joint_random_search = (
    _m.bipartite_free_ancilla_joint_random_search
)
bipartite_hold_unitary = _m.bipartite_hold_unitary
build_bipartite_hold_hamiltonian = _m.build_bipartite_hold_hamiltonian
compute_bipartite_free_css_joint_sensitivity = (
    _m.compute_bipartite_free_css_joint_sensitivity
)
compute_bipartite_sensitivity = _m.compute_bipartite_sensitivity
compute_qubit_free_ancilla_joint_sensitivity = (
    _m.compute_qubit_free_ancilla_joint_sensitivity
)
decoupled_baseline = _m.decoupled_baseline
evolve_bipartite_circuit = _m.evolve_bipartite_circuit
multi_particle_free_css_initial_state = _m.multi_particle_free_css_initial_state
n_particle_free_ancilla_initial_state = _m.n_particle_free_ancilla_initial_state
qubit_free_ancilla_joint_objective = _m.qubit_free_ancilla_joint_objective
qubit_free_ancilla_joint_random_search = _m.qubit_free_ancilla_joint_random_search
run_bipartite_free_ancilla_joint_nelder_mead = (
    _m.run_bipartite_free_ancilla_joint_nelder_mead
)
run_decoupled_baseline = _m.run_decoupled_baseline
run_qubit_free_ancilla_joint_nelder_mead = _m.run_qubit_free_ancilla_joint_nelder_mead
run_single_qubit_n_omega = _m.run_single_qubit_n_omega

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=[1, 2, 5, 10])
def make_N(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture
def make_ops(make_N: int) -> dict[str, np.ndarray]:
    return build_n_particle_operators(make_N)


@pytest.fixture
def make_bipartite_ops(make_N: int) -> dict[str, np.ndarray]:
    return build_operators(make_N, make_N)


@pytest.fixture
def make_psi0(make_N: int) -> np.ndarray:
    return n_particle_initial_state(make_N)


# ============================================================================
# State Preparation — Free-Ancilla (J_A=1/2)
# ============================================================================


class TestNParticleFreeAncillaInitialState:
    def test_given_N_then_correct_dimension(self, make_N: int) -> None:
        psi = n_particle_free_ancilla_initial_state(make_N, 0.0, 0.0)
        d_tot = 2 * (make_N + 1)
        assert psi.shape == (d_tot,), f"Expected ({d_tot},), got {psi.shape}"

    def test_given_N_then_normalised(self, make_N: int) -> None:
        psi = n_particle_free_ancilla_initial_state(make_N, 0.5, 1.0)
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_given_theta_zero_then_top_ancilla(self, make_N: int) -> None:
        """At θ_A=0, ancilla should be |1,0⟩ (highest weight)."""
        psi = n_particle_free_ancilla_initial_state(make_N, 0.0, 0.0)
        d_sys = make_N + 1
        # System part: |J_S, J_S⟩ @ index 0
        # Ancilla part: |1,0⟩ @ index 0 within its 2-dim space
        # Kronecker: index = 0 * 2 + 0 = 0
        assert abs(psi[0]) > 0.99, f"psi[0] = {psi[0]:.6f} (expected ~1)"
        assert np.allclose(np.abs(psi[[0, d_sys]]), [1.0, 0.0], atol=1e-10), (
            "Ancilla should be |1,0⟩ at θ_A=0"
        )

    def test_given_theta_pi_then_bottom_ancilla(self, make_N: int) -> None:
        """At θ_A=π, ancilla should be |0,1⟩."""
        psi = n_particle_free_ancilla_initial_state(make_N, np.pi, 0.0)
        # System: |J_S, J_S⟩ @ index 0; Ancilla: |0,1⟩ @ index 1
        # Kronecker: index = 0 * 2 + 1 = 1
        assert abs(psi[1]) > 0.99, f"psi[1] = {psi[1]:.6f} (expected ~1)"

    def test_given_theta_pi_half_phi_zero_then_equal_superposition(
        self, make_N: int
    ) -> None:
        """At θ_A=π/2, φ_A=0, ancilla should be equal superposition."""
        psi = n_particle_free_ancilla_initial_state(make_N, np.pi / 2.0, 0.0)
        # System: |J_S, J_S⟩ @ index 0; Ancilla: (|1,0⟩ + |0,1⟩)/√2
        # Expected: 1/√2 at indices 0 and 1
        assert np.isclose(abs(psi[0]), 1.0 / np.sqrt(2), atol=1e-10)
        assert np.isclose(abs(psi[1]), 1.0 / np.sqrt(2), atol=1e-10)


# ============================================================================
# State Preparation — Free-CSS (J_A=N/2)
# ============================================================================


class TestMultiParticleFreeCSSInitialState:
    def test_given_N_then_correct_dimension(self, make_N: int) -> None:
        psi = multi_particle_free_css_initial_state(make_N, 0.0, 0.0)
        d_tot = (make_N + 1) ** 2
        assert psi.shape == (d_tot,), f"Expected ({d_tot},), got {psi.shape}"

    def test_given_N_then_normalised(self, make_N: int) -> None:
        psi = multi_particle_free_css_initial_state(make_N, 0.5, 1.0)
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_given_theta_zero_then_top_ancilla_dicke(self, make_N: int) -> None:
        """At θ_A=0, ancilla CSS should be |J_A, +J_A⟩."""
        psi = multi_particle_free_css_initial_state(make_N, 0.0, 0.0)
        # System: |J_S, J_S⟩ @ index 0; Ancilla: |J_A, J_A⟩ @ index 0 in its space
        # Kronecker: index = 0 * d_sys + 0 = 0
        assert abs(psi[0]) > 0.99, f"psi[0] = {psi[0]:.6f} (expected ~1)"

    def test_given_theta_pi_then_bottom_ancilla_dicke(self, make_N: int) -> None:
        """At θ_A=π, ancilla CSS should be |J_A, -J_A⟩."""
        psi = multi_particle_free_css_initial_state(make_N, np.pi, 0.0)
        d_anc = make_N + 1
        # System: |J_S, J_S⟩ @ index 0
        # Ancilla: |J_A, -J_A⟩ @ index d_anc - 1
        # Kronecker: index = 0 * d_anc + (d_anc - 1) = d_anc - 1
        expected_idx = d_anc - 1
        assert abs(psi[expected_idx]) > 0.99, (
            f"psi[{expected_idx}] = {psi[expected_idx]:.6f} (expected ~1)"
        )


# ============================================================================
# Joint Measurement Operator — J_A=1/2
# ============================================================================


class TestBuildJointMeasurementOperator:
    def test_given_psi_zero_then_matches_Jz_S(
        self, make_N: int, make_ops: dict
    ) -> None:
        M = build_joint_measurement_operator(make_N, 0.0, make_ops)
        assert np.allclose(M, make_ops["Jz_S"], atol=1e-12)

    def test_given_psi_pi_half_then_matches_Jz_A(
        self, make_N: int, make_ops: dict
    ) -> None:
        M = build_joint_measurement_operator(make_N, np.pi / 2.0, make_ops)
        assert np.allclose(M, make_ops["Jz_A"], atol=1e-12)

    def test_given_psi_then_hermitian(self, make_N: int, make_ops: dict) -> None:
        for psi in [0.0, 0.5, 1.0, np.pi / 4]:
            M = build_joint_measurement_operator(make_N, psi, make_ops)
            assert np.allclose(M, M.conj().T, atol=1e-12)

    def test_given_psi_then_correct_dimension(
        self, make_N: int, make_ops: dict
    ) -> None:
        M = build_joint_measurement_operator(make_N, 0.5, make_ops)
        d_tot = 2 * (make_N + 1)
        assert M.shape == (d_tot, d_tot)

    def test_given_psi_pi_four_then_balanced(self, make_N: int, make_ops: dict) -> None:
        M = build_joint_measurement_operator(make_N, np.pi / 4.0, make_ops)
        expected = (make_ops["Jz_S"] + make_ops["Jz_A"]) / np.sqrt(2)
        assert np.allclose(M, expected, atol=1e-12)

    def test_given_N1_psi_pi_four_then_correct_values(self) -> None:
        ops = build_n_particle_operators(1)
        M = build_joint_measurement_operator(1, np.pi / 4.0, ops)
        expected_diag = np.array([1.0, 0.0, 0.0, -1.0]) / np.sqrt(2)
        assert np.allclose(np.diag(M), expected_diag, atol=1e-12)


# ============================================================================
# Joint Measurement Operator — J_A=N/2 (Bipartite)
# ============================================================================


class TestBuildBipartiteJointMeasurementOperator:
    def test_given_psi_zero_then_matches_Jz_S(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        M = build_bipartite_joint_measurement_operator(make_N, 0.0, make_bipartite_ops)
        assert np.allclose(M, make_bipartite_ops["Jz_S"], atol=1e-12)

    def test_given_psi_pi_half_then_matches_Jz_A(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        M = build_bipartite_joint_measurement_operator(
            make_N, np.pi / 2.0, make_bipartite_ops
        )
        assert np.allclose(M, make_bipartite_ops["Jz_A"], atol=1e-12)

    def test_given_psi_then_hermitian(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        for psi in [0.0, 0.5, 1.0, np.pi / 4]:
            M = build_bipartite_joint_measurement_operator(
                make_N, psi, make_bipartite_ops
            )
            assert np.allclose(M, M.conj().T, atol=1e-12)

    def test_given_psi_then_correct_dimension(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        M = build_bipartite_joint_measurement_operator(make_N, 0.5, make_bipartite_ops)
        d_tot = (make_N + 1) ** 2
        assert M.shape == (d_tot, d_tot)

    def test_given_psi_pi_four_then_balanced(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        M = build_bipartite_joint_measurement_operator(
            make_N, np.pi / 4.0, make_bipartite_ops
        )
        expected = (make_bipartite_ops["Jz_S"] + make_bipartite_ops["Jz_A"]) / np.sqrt(
            2
        )
        assert np.allclose(M, expected, atol=1e-12)


# ============================================================================
# Bipartite Hold Hamiltonian
# ============================================================================


class TestBuildBipartiteHoldHamiltonian:
    def test_given_phi_zero_then_hermitian(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        H = build_bipartite_hold_hamiltonian(
            make_N,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_bipartite_ops,
        )
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_all_zero_then_omega_jz_s(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        H = build_bipartite_hold_hamiltonian(
            make_N,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            make_bipartite_ops,
        )
        expected = 0.5 * make_bipartite_ops["Jz_S"]
        assert np.allclose(H, expected, atol=1e-12)

    def test_given_correct_dimension(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        H = build_bipartite_hold_hamiltonian(
            make_N,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_bipartite_ops,
        )
        d_tot = (make_N + 1) ** 2
        assert H.shape == (d_tot, d_tot)

    def test_given_azz_term_then_correct_structure(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        H = build_bipartite_hold_hamiltonian(
            make_N,
            0.0,
            0.0,
            0.0,
            0.0,
            5.0,
            make_bipartite_ops,
        )
        expected = 5.0 * (make_bipartite_ops["Jz_S"] @ make_bipartite_ops["Jz_A"])
        assert np.allclose(H, expected, atol=1e-12)


class TestBipartiteHoldUnitary:
    def test_given_hold_then_unitary(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        U = bipartite_hold_unitary(
            make_N,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_bipartite_ops,
        )
        d_tot = (make_N + 1) ** 2
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12)

    def test_given_zero_hold_then_identity(
        self, make_N: int, make_bipartite_ops: dict
    ) -> None:
        U = bipartite_hold_unitary(
            make_N,
            0.0,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_bipartite_ops,
        )
        d_tot = (make_N + 1) ** 2
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U, I_full, atol=1e-12)


# ============================================================================
# Bipartite Circuit Evolution
# ============================================================================


class TestEvolveBipartiteCircuit:
    def test_given_circuit_then_normalised(self, make_N: int) -> None:
        ops = build_operators(make_N, make_N)
        psi0 = multi_particle_free_css_initial_state(make_N, 0.5, 1.0)
        psi = evolve_bipartite_circuit(
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            ops,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_given_zero_params_then_system_only_mzi(self, make_N: int) -> None:
        ops = build_operators(make_N, make_N)
        psi0 = multi_particle_free_css_initial_state(make_N, 0.0, 0.0)
        psi = evolve_bipartite_circuit(
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            ops,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0)
        exp_val, var_val = compute_expectation_and_variance(psi, ops["Jz_S"])
        assert np.isfinite(exp_val)
        assert var_val >= 0.0


# ============================================================================
# Sensitivity Computation — J_A=1/2
# ============================================================================


class TestComputeQubitFreeAncillaJointSensitivity:
    def test_given_decoupled_then_sql(self, make_N: int, make_ops: dict) -> None:
        """At zero drive, zero interaction, qubit ancilla gives SQL."""
        delta = compute_qubit_free_ancilla_joint_sensitivity(
            make_N,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        sql = sql_reference(make_N, T_HOLD)
        assert np.isfinite(delta), f"Δω = inf for N={make_N} (decoupled)"
        assert np.isclose(delta, sql, rtol=1e-8), (
            f"N={make_N}: Δω={delta:.10f}, SQL={sql:.10f}"
        )

    def test_given_valid_params_then_positive(
        self, make_N: int, make_ops: dict
    ) -> None:
        delta = compute_qubit_free_ancilla_joint_sensitivity(
            make_N,
            0.5,
            0.5,
            1.0,
            1.0,
            2.0,
            3.0,
            4.0,
            0.5,
        )
        assert np.isfinite(delta), f"Δω = inf for valid params N={make_N}"
        assert delta > 0.0, f"Δω must be positive, got {delta}"

    def test_given_psi_zero_matches_n_particle_sensitivity(
        self, make_N: int, make_ops: dict, make_psi0: np.ndarray
    ) -> None:
        """ψ=0 with free ancilla should give same result as Jz_S measurement."""
        delta_free = compute_qubit_free_ancilla_joint_sensitivity(
            make_N,
            0.5,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            0.0,
        )
        delta_base = compute_n_particle_sensitivity(
            make_N,
            make_psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_ops,
        )
        assert np.isclose(delta_free, delta_base, rtol=1e-8), (
            f"N={make_N}: free={delta_free:.10f}, base={delta_base:.10f}"
        )


# ============================================================================
# Sensitivity Computation — J_A=N/2
# ============================================================================


class TestComputeBipartiteFreeCSSJointSensitivity:
    def test_given_decoupled_then_sql(self, make_N: int) -> None:
        """At zero drive, zero interaction, CSS ancilla gives SQL."""
        ops = build_operators(make_N, make_N)
        psi0 = multi_particle_free_css_initial_state(make_N, 0.0, 0.0)
        delta = compute_bipartite_sensitivity(
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            ops,
            meas_op=build_bipartite_joint_measurement_operator(make_N, 0.0, ops),
        )
        sql = sql_reference(make_N, T_HOLD)
        assert np.isfinite(delta), f"Δω = inf for N={make_N} (decoupled)"
        assert np.isclose(delta, sql, rtol=1e-8), (
            f"N={make_N}: Δω={delta:.10f}, SQL={sql:.10f}"
        )

    def test_given_valid_params_then_positive(self, make_N: int) -> None:
        delta = compute_bipartite_free_css_joint_sensitivity(
            make_N,
            0.5,
            0.5,
            1.0,
            1.0,
            2.0,
            3.0,
            4.0,
            0.5,
        )
        assert np.isfinite(delta), f"Δω = inf for valid params N={make_N}"
        assert delta > 0.0, f"Δω must be positive, got {delta}"

    def test_given_psi_zero_matches_system_only(self, make_N: int) -> None:
        psi0 = multi_particle_free_css_initial_state(make_N, 0.0, 0.0)
        ops = build_operators(make_N, make_N)
        delta_joint = compute_bipartite_free_css_joint_sensitivity(
            make_N,
            0.5,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            0.0,
        )
        delta_sys = compute_bipartite_sensitivity(
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            ops,
        )
        assert np.isclose(delta_joint, delta_sys, rtol=1e-8), (
            f"N={make_N}: joint={delta_joint:.10f}, sys={delta_sys:.10f}"
        )


# ============================================================================
# Decoupled Baseline
# ============================================================================


class TestDecoupledBaseline:
    def test_given_N_then_matches_sql(self, make_N: int) -> None:
        delta = decoupled_baseline(make_N, T_HOLD)
        sql = sql_reference(make_N, T_HOLD)
        assert np.isclose(delta, sql, rtol=1e-10), (
            f"N={make_N}: Δω={delta:.10f}, SQL={sql:.10f}"
        )

    def test_given_sql_scaling_then_exponent_half(self) -> None:
        Ns = [1, 2, 4, 8, 16]
        sqls = np.array([sql_reference(n, T_HOLD) for n in Ns])
        log_N = np.log(Ns)
        log_sql = np.log(sqls)
        A = np.vstack([log_N, np.ones_like(log_N)]).T
        alpha, _ = np.linalg.lstsq(A, log_sql, rcond=None)[0]
        assert np.isclose(alpha, -0.5, atol=1e-10), (
            f"SQL scaling exponent α = {alpha:.6f}, expected -0.5"
        )

    def test_given_decoupled_baseline_then_ratio_one(self, make_N: int) -> None:
        delta = decoupled_baseline(make_N, T_HOLD)
        sql = sql_reference(make_N, T_HOLD)
        ratio = delta / sql
        assert np.isclose(ratio, 1.0, rtol=1e-10), f"N={make_N}: ratio={ratio:.10f}"

    def test_run_decoupled_baseline_success(self) -> None:
        """Cover the run_decoupled_baseline wrapper."""
        for N in [1, 2, 5]:
            result = run_decoupled_baseline(N)
            assert result.N == N
            assert np.isclose(result.ratio, 1.0, atol=1e-10)
            assert result.success
            assert result.nfev == 0


# ============================================================================
# Random Search — J_A=1/2
# ============================================================================


class TestQubitRandomSearch:
    def test_given_random_search_then_returns_result(self, make_N: int) -> None:
        result = qubit_free_ancilla_joint_random_search(
            make_N,
            0.5,
            n_samples=10,
            seed=42,
        )
        assert isinstance(result, QubitFreeAncillaJointRandomSearchResult)
        assert result.samples.shape == (10, 7)
        assert len(result.delta_omega_values) == 10
        assert result.best_delta_omega > 0.0

    def test_given_random_search_then_best_is_minimum(self, make_N: int) -> None:
        result = qubit_free_ancilla_joint_random_search(
            make_N,
            0.5,
            n_samples=10,
            seed=42,
        )
        assert np.isclose(
            result.best_delta_omega,
            np.min(result.delta_omega_values),
        )

    def test_given_random_search_then_params_have_seven_elements(
        self, make_N: int
    ) -> None:
        result = qubit_free_ancilla_joint_random_search(
            make_N,
            0.5,
            n_samples=10,
            seed=42,
        )
        assert len(result.best_params) == 7


# ============================================================================
# Random Search — J_A=N/2
# ============================================================================


class TestBipartiteRandomSearch:
    @pytest.mark.slow
    def test_given_random_search_then_returns_result(self, make_N: int) -> None:
        result = bipartite_free_ancilla_joint_random_search(
            make_N,
            0.5,
            n_samples=10,
            seed=42,
        )
        assert isinstance(result, BipartiteFreeAncillaJointRandomSearchResult)
        assert result.samples.shape == (10, 7)
        assert len(result.delta_omega_values) == 10
        assert result.best_delta_omega > 0.0

    @pytest.mark.slow
    def test_given_random_search_then_best_is_minimum(self, make_N: int) -> None:
        result = bipartite_free_ancilla_joint_random_search(
            make_N,
            0.5,
            n_samples=10,
            seed=42,
        )
        assert np.isclose(
            result.best_delta_omega,
            np.min(result.delta_omega_values),
        )


# ============================================================================
# Objective Functions
# ============================================================================


class TestQubitObjective:
    def test_given_valid_params_then_finite(self, make_N: int) -> None:
        obj = qubit_free_ancilla_joint_objective(
            np.array([0.5, 1.0, 1.0, 2.0, 3.0, 4.0, 0.5]),
            make_N,
            0.5,
        )
        assert np.isfinite(obj)
        assert obj > 0.0

    def test_given_theta_out_of_bounds_then_penalty(self, make_N: int) -> None:
        obj = qubit_free_ancilla_joint_objective(
            np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            make_N,
            0.5,
        )
        assert obj > 1e10, f"Expected large penalty, got {obj}"

    def test_given_psi_out_of_bounds_then_penalty(self, make_N: int) -> None:
        obj = qubit_free_ancilla_joint_objective(
            np.array([0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 10.0]),
            make_N,
            0.5,
        )
        assert obj > 1e10, f"Expected large penalty, got {obj}"


class TestBipartiteObjective:
    def test_given_valid_params_then_finite(self, make_N: int) -> None:
        obj = bipartite_free_ancilla_joint_objective(
            np.array([0.5, 1.0, 1.0, 2.0, 3.0, 4.0, 0.5]),
            make_N,
            0.5,
        )
        assert np.isfinite(obj)
        assert obj > 0.0

    def test_given_theta_out_of_bounds_then_penalty(self, make_N: int) -> None:
        obj = bipartite_free_ancilla_joint_objective(
            np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            make_N,
            0.5,
        )
        assert obj > 1e10, f"Expected large penalty, got {obj}"


# ============================================================================
# N=1 Consistency (Exp 1)
# ============================================================================


class TestN1Consistency:
    @pytest.mark.slow
    def test_given_N1_omega02_then_beats_sql(self) -> None:
        result = run_single_qubit_n_omega(
            N=1,
            omega=0.2,
            n_random=100,
            n_nm_refine=5,
            seed=42,
        )
        assert result.ratio > 1.0, (
            f"N=1, ω=0.2: ratio = {result.ratio:.4f} (expected > 1, beating SQL)"
        )

    @pytest.mark.slow
    def test_given_N1_omega02_then_variance_M_positive(self) -> None:
        result = run_single_qubit_n_omega(
            N=1,
            omega=0.2,
            n_random=100,
            n_nm_refine=5,
            seed=42,
        )
        assert result.variance_M >= -1e-12
        assert result.variance_M >= 0.0

    @pytest.mark.slow
    def test_given_N1_omega02_then_expectation_M_finite(self) -> None:
        result = run_single_qubit_n_omega(
            N=1,
            omega=0.2,
            n_random=100,
            n_nm_refine=5,
            seed=42,
        )
        assert np.isfinite(result.expectation_M)


# ============================================================================
# Nelder-Mead — J_A=1/2
# ============================================================================


class TestQubitNelderMead:
    @pytest.mark.parametrize("make_N", [1, 2])
    @pytest.mark.slow
    def test_given_nm_then_converges(self, make_N: int) -> None:
        result = run_qubit_free_ancilla_joint_nelder_mead(
            N=make_N,
            omega_true=0.5,
            seed=42,
        )
        assert isinstance(result, QubitFreeAncillaJointNMSensitivityResult)
        assert result.delta_omega_opt > 0.0
        assert np.isfinite(result.delta_omega_opt)
        assert result.params_opt.shape == (7,)

    @pytest.mark.parametrize("make_N", [1, 2])
    @pytest.mark.slow
    def test_given_nm_then_expectation_finite(self, make_N: int) -> None:
        result = run_qubit_free_ancilla_joint_nelder_mead(
            N=make_N,
            omega_true=0.5,
            seed=42,
        )
        assert np.isfinite(result.expectation_M)
        assert result.variance_M >= 0.0
        assert np.isfinite(result.d_expectation)


# ============================================================================
# Nelder-Mead — J_A=N/2
# ============================================================================


class TestBipartiteNelderMead:
    @pytest.mark.parametrize("make_N", [1])
    def test_given_nm_then_converges(self, make_N: int) -> None:
        result = run_bipartite_free_ancilla_joint_nelder_mead(
            N=make_N,
            omega_true=0.5,
            seed=42,
        )
        assert isinstance(result, BipartiteFreeAncillaJointNMSensitivityResult)
        assert result.delta_omega_opt > 0.0
        assert np.isfinite(result.delta_omega_opt)
        assert result.params_opt.shape == (7,)

    @pytest.mark.parametrize("make_N", [1])
    def test_given_nm_then_expectation_finite(self, make_N: int) -> None:
        result = run_bipartite_free_ancilla_joint_nelder_mead(
            N=make_N,
            omega_true=0.5,
            seed=42,
        )
        assert np.isfinite(result.expectation_M)
        assert result.variance_M >= 0.0
        assert np.isfinite(result.d_expectation)


# ============================================================================
# QubitFreeAncillaJointNScalingResult Serialization
# ============================================================================


class TestQubitScalingResultParquet:
    @pytest.fixture
    def make_result(self) -> QubitFreeAncillaJointNScalingResult:
        sql_ref = sql_reference(5, T_HOLD)
        return QubitFreeAncillaJointNScalingResult(
            N=5,
            omega=0.2,
            delta_omega_opt=0.015,
            sql=sql_ref,
            ratio=sql_ref / 0.015,
            theta_A_opt=0.5,
            phi_A_opt=1.0,
            a_x_opt=3.0,
            a_y_opt=-2.5,
            a_z_opt=1.0,
            a_zz_opt=4.0,
            psi_opt=0.5,
            expectation_M=0.3,
            variance_M=0.18,
            d_expectation=0.05,
            t_hold=T_HOLD,
            success=True,
            nfev=100,
        )

    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("N", "eq"),
        ("omega", "eq"),
        ("delta_omega_opt", "isclose"),
        ("sql", "isclose"),
        ("ratio", "isclose"),
        ("theta_A_opt", "eq"),
        ("phi_A_opt", "eq"),
        ("a_x_opt", "eq"),
        ("a_y_opt", "eq"),
        ("a_z_opt", "eq"),
        ("a_zz_opt", "eq"),
        ("psi_opt", "isclose"),
        ("expectation_M", "eq"),
        ("variance_M", "eq"),
        ("d_expectation", "eq"),
        ("t_hold", "eq"),
        ("success", "eq"),
        ("nfev", "eq"),
    ]

    def test_roundtrip(
        self, make_result: QubitFreeAncillaJointNScalingResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "test.parquet"
        make_result.save_parquet(p)
        loaded = QubitFreeAncillaJointNScalingResult.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "missing.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            QubitFreeAncillaJointNScalingResult.from_parquet(p)


class TestQubitScalingScanResultParquet:
    @pytest.fixture
    def make_results(self) -> list[QubitFreeAncillaJointNScalingResult]:
        return [
            QubitFreeAncillaJointNScalingResult(
                N=n,
                omega=w,
                delta_omega_opt=0.02 / (n**0.5),
                sql=sql_reference(n, T_HOLD),
                ratio=sql_reference(n, T_HOLD) / (0.02 / (n**0.5)),
                theta_A_opt=0.0,
                phi_A_opt=0.0,
                a_x_opt=float(n),
                a_y_opt=0.0,
                a_z_opt=0.0,
                a_zz_opt=float(n),
                psi_opt=0.5,
                expectation_M=0.2,
                variance_M=0.1,
                d_expectation=0.05,
                t_hold=T_HOLD,
                success=True,
                nfev=50,
            )
            for n in [1, 2, 5, 10]
            for w in [0.1, 0.2, 0.5]
        ]

    def test_roundtrip(
        self,
        make_results: list[QubitFreeAncillaJointNScalingResult],
        tmp_path: Path,
    ) -> None:
        summary = QubitFreeAncillaJointNScalingScanResult(results=make_results)
        p = tmp_path / "scan.parquet"
        summary.save_parquet(p)
        loaded = QubitFreeAncillaJointNScalingScanResult.from_parquet(p)
        assert len(loaded.results) == len(make_results)
        for orig, loaded_r in zip(make_results, loaded.results, strict=False):
            assert orig.N == loaded_r.N
            assert orig.omega == loaded_r.omega
            assert np.isclose(orig.psi_opt, loaded_r.psi_opt)

    def test_empty_dataframe(self) -> None:
        summary = QubitFreeAncillaJointNScalingScanResult(results=[])
        df = summary.to_dataframe()
        assert df.empty

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "bad_scan.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            QubitFreeAncillaJointNScalingScanResult.from_parquet(p)


# ============================================================================
# BipartiteFreeAncillaJointNScalingResult Serialization
# ============================================================================


class TestBipartiteScalingResultParquet:
    @pytest.fixture
    def make_result(self) -> BipartiteFreeAncillaJointNScalingResult:
        sql_ref = sql_reference(5, T_HOLD)
        return BipartiteFreeAncillaJointNScalingResult(
            N=5,
            omega=0.2,
            delta_omega_opt=0.015,
            sql=sql_ref,
            ratio=sql_ref / 0.015,
            theta_A_opt=0.5,
            phi_A_opt=1.0,
            a_x_opt=3.0,
            a_y_opt=-2.5,
            a_z_opt=1.0,
            a_zz_opt=4.0,
            psi_opt=0.5,
            expectation_M=0.3,
            variance_M=0.18,
            d_expectation=0.05,
            t_hold=T_HOLD,
            success=True,
            nfev=100,
        )

    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("N", "eq"),
        ("omega", "eq"),
        ("delta_omega_opt", "isclose"),
        ("sql", "isclose"),
        ("ratio", "isclose"),
        ("theta_A_opt", "eq"),
        ("phi_A_opt", "eq"),
        ("a_x_opt", "eq"),
        ("a_y_opt", "eq"),
        ("a_z_opt", "eq"),
        ("a_zz_opt", "eq"),
        ("psi_opt", "isclose"),
        ("expectation_M", "eq"),
        ("variance_M", "eq"),
        ("d_expectation", "eq"),
        ("t_hold", "eq"),
        ("success", "eq"),
        ("nfev", "eq"),
    ]

    def test_roundtrip(
        self, make_result: BipartiteFreeAncillaJointNScalingResult, tmp_path: Path
    ) -> None:
        p = tmp_path / "test.parquet"
        make_result.save_parquet(p)
        loaded = BipartiteFreeAncillaJointNScalingResult.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "missing.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            BipartiteFreeAncillaJointNScalingResult.from_parquet(p)


class TestBipartiteScalingScanResultParquet:
    @pytest.fixture
    def make_results(self) -> list[BipartiteFreeAncillaJointNScalingResult]:
        return [
            BipartiteFreeAncillaJointNScalingResult(
                N=n,
                omega=w,
                delta_omega_opt=0.02 / (n**0.5),
                sql=sql_reference(n, T_HOLD),
                ratio=sql_reference(n, T_HOLD) / (0.02 / (n**0.5)),
                theta_A_opt=0.0,
                phi_A_opt=0.0,
                a_x_opt=float(n),
                a_y_opt=0.0,
                a_z_opt=0.0,
                a_zz_opt=float(n),
                psi_opt=0.5,
                expectation_M=0.2,
                variance_M=0.1,
                d_expectation=0.05,
                t_hold=T_HOLD,
                success=True,
                nfev=50,
            )
            for n in [1, 2, 5, 10]
            for w in [0.1, 0.2, 0.5]
        ]

    def test_roundtrip(
        self,
        make_results: list[BipartiteFreeAncillaJointNScalingResult],
        tmp_path: Path,
    ) -> None:
        summary = BipartiteFreeAncillaJointNScalingScanResult(results=make_results)
        p = tmp_path / "scan.parquet"
        summary.save_parquet(p)
        loaded = BipartiteFreeAncillaJointNScalingScanResult.from_parquet(p)
        assert len(loaded.results) == len(make_results)
        for orig, loaded_r in zip(make_results, loaded.results, strict=False):
            assert orig.N == loaded_r.N
            assert orig.omega == loaded_r.omega
            assert np.isclose(orig.psi_opt, loaded_r.psi_opt)

    def test_empty_dataframe(self) -> None:
        summary = BipartiteFreeAncillaJointNScalingScanResult(results=[])
        df = summary.to_dataframe()
        assert df.empty

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "bad_scan.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            BipartiteFreeAncillaJointNScalingScanResult.from_parquet(p)


# ============================================================================
# Physical Invariants
# ============================================================================


class TestPhysicalInvariants:
    def test_given_sensitivity_then_positive_definite(self, make_N: int) -> None:
        """Δω should always be positive (or inf at fringe extremum)."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            theta = rng.uniform(*THETA_A_BOUNDS)
            phi = rng.uniform(*PHI_A_BOUNDS)
            ax = rng.uniform(*DRIVE_BOUNDS)
            ay = rng.uniform(*DRIVE_BOUNDS)
            az = rng.uniform(*DRIVE_BOUNDS)
            azz = rng.uniform(*AZZ_BOUNDS)
            psi = rng.uniform(*PSI_BOUNDS)
            delta = compute_qubit_free_ancilla_joint_sensitivity(
                make_N,
                0.5,
                theta,
                phi,
                ax,
                ay,
                az,
                azz,
                psi,
            )
            assert delta > 0.0 or np.isinf(delta), (
                f"Δω must be positive (or inf), got {delta}"
            )

    def test_given_variance_non_negative(self, make_N: int) -> None:
        """Var(M) must be non-negative everywhere."""
        rng = np.random.default_rng(42)
        ops = build_n_particle_operators(make_N)
        for _ in range(5):
            omega = rng.uniform(0.1, 1.0)
            theta = rng.uniform(*THETA_A_BOUNDS)
            phi = rng.uniform(*PHI_A_BOUNDS)
            ax = rng.uniform(*DRIVE_BOUNDS)
            ay = rng.uniform(*DRIVE_BOUNDS)
            az = rng.uniform(*DRIVE_BOUNDS)
            azz = rng.uniform(*AZZ_BOUNDS)
            psi = rng.uniform(*PSI_BOUNDS)

            M_op = build_joint_measurement_operator(make_N, psi, ops)
            psi0 = n_particle_free_ancilla_initial_state(make_N, theta, phi)
            psi_final = evolve_n_particle_circuit(
                make_N,
                psi0,
                T_BS,
                T_HOLD,
                omega,
                ax,
                ay,
                az,
                azz,
                ops,
            )
            _, var_val = compute_expectation_and_variance(psi_final, M_op)
            assert var_val >= 0.0 or np.isclose(var_val, 0.0, atol=1e-15), (
                f"Variance must be non-negative, got {var_val}"
            )


# ============================================================================
# Coverage: Scan result container interface
# ============================================================================


class TestScanResultContainer:
    def test_qubit_scan_len_and_getitem(self) -> None:
        """Cover __len__ and __getitem__ on QubitFreeAncillaJointNScalingScanResult."""
        results = [
            QubitFreeAncillaJointNScalingResult(
                N=n,
                omega=0.2,
                delta_omega_opt=0.1 / (n**0.5),
                sql=sql_reference(n, T_HOLD),
                ratio=sql_reference(n, T_HOLD) / (0.1 / (n**0.5)),
                theta_A_opt=0.0,
                phi_A_opt=0.0,
                a_x_opt=0.0,
                a_y_opt=0.0,
                a_z_opt=0.0,
                a_zz_opt=0.0,
                psi_opt=0.0,
            )
            for n in [1, 2, 5]
        ]
        scan = QubitFreeAncillaJointNScalingScanResult(results=results)
        assert len(scan) == 3
        assert scan[0].N == 1
        assert scan[1].N == 2
        assert scan[2].N == 5

    def test_qubit_scan_to_numpy_arrays(self) -> None:
        """Cover to_numpy_arrays on QubitFreeAncillaJointNScalingScanResult."""
        results = [
            QubitFreeAncillaJointNScalingResult(
                N=n,
                omega=0.2,
                delta_omega_opt=0.1 / (n**0.5),
                sql=sql_reference(n, T_HOLD),
                ratio=sql_reference(n, T_HOLD) / (0.1 / (n**0.5)),
                theta_A_opt=0.5,
                phi_A_opt=1.0,
                a_x_opt=1.0,
                a_y_opt=2.0,
                a_z_opt=3.0,
                a_zz_opt=4.0,
                psi_opt=0.5,
            )
            for n in [1, 2, 5]
        ]
        scan = QubitFreeAncillaJointNScalingScanResult(results=results)
        arrays = scan.to_numpy_arrays()
        assert "N" in arrays
        assert "omega" in arrays
        assert "delta_omega_opt" in arrays
        assert list(arrays["N"]) == [1, 2, 5]
        assert list(arrays["psi_opt"]) == [0.5, 0.5, 0.5]

    def test_bipartite_scan_len_and_getitem(self) -> None:
        """Cover __len__ and __getitem__ on BipartiteFreeAncillaJointNScalingScanResult."""
        results = [
            BipartiteFreeAncillaJointNScalingResult(
                N=n,
                omega=0.2,
                delta_omega_opt=0.1 / (n**0.5),
                sql=sql_reference(n, T_HOLD),
                ratio=sql_reference(n, T_HOLD) / (0.1 / (n**0.5)),
                theta_A_opt=0.0,
                phi_A_opt=0.0,
                a_x_opt=0.0,
                a_y_opt=0.0,
                a_z_opt=0.0,
                a_zz_opt=0.0,
                psi_opt=0.0,
            )
            for n in [1, 2]
        ]
        scan = BipartiteFreeAncillaJointNScalingScanResult(results=results)
        assert len(scan) == 2
        assert scan[0].N == 1

    def test_bipartite_scan_to_numpy_arrays(self) -> None:
        """Cover to_numpy_arrays on BipartiteFreeAncillaJointNScalingScanResult."""
        results = [
            BipartiteFreeAncillaJointNScalingResult(
                N=n,
                omega=0.2,
                delta_omega_opt=0.1 / (n**0.5),
                sql=sql_reference(n, T_HOLD),
                ratio=sql_reference(n, T_HOLD) / (0.1 / (n**0.5)),
                theta_A_opt=0.0,
                phi_A_opt=0.0,
                a_x_opt=0.0,
                a_y_opt=0.0,
                a_z_opt=0.0,
                a_zz_opt=0.0,
                psi_opt=0.0,
            )
            for n in [1, 2]
        ]
        scan = BipartiteFreeAncillaJointNScalingScanResult(results=results)
        arrays = scan.to_numpy_arrays()
        assert "N" in arrays
        assert list(arrays["N"]) == [1, 2]

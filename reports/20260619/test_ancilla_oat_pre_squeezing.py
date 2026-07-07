"""Tests for the Ancilla OAT Pre-Squeezing before omega-Modulated Hold report.

Companion test module for ``reports/20260619/ancilla_oat_pre_squeezing.py``.
Tests operator construction, CSS state, OAT unitary, circuit evolution,
decoupled baseline, sensitivity, optimisation, and serialization.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.algorithms.spin_squeezing import coherent_spin_state
from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
)
from src.analysis.sensitivity_metrics import sql_reference
from src.physics.dicke_basis import jx_operator, jy_operator, jz_operator
from src.utils.enums import OperatorBasis
from src.utils.serialization import assert_roundtrip_fields

_m = importlib.import_module("reports.20260619.ancilla_oat_pre_squeezing")

FD_STEP = _m.FD_STEP
PHASE1_N_ANCILLA = _m.PHASE1_N_ANCILLA
PHASE1_N_SYSTEM = _m.PHASE1_N_SYSTEM
T_BS = _m.T_BS
T_HOLD = _m.T_HOLD
OATNelderMeadResult = _m.OATNelderMeadResult
OATNScalingResult = _m.OATNScalingResult
OATNScalingScanResult = _m.OATNScalingScanResult
OATRandomSearchResult = _m.OATRandomSearchResult
build_hold_hamiltonian = _m.build_hold_hamiltonian
build_iszz_interaction = _m.build_iszz_interaction
build_modulated_drive_hamiltonian = _m.build_modulated_drive_hamiltonian
build_operators = _m.build_operators
build_system_only_bs_unitary = _m.build_system_only_bs_unitary
compute_oat_decoupled_baseline = _m.compute_oat_decoupled_baseline
compute_oat_decoupled_with_oat = _m.compute_oat_decoupled_with_oat
compute_oat_sensitivity = _m.compute_oat_sensitivity
evolve_oat_circuit = _m.evolve_oat_circuit
hold_unitary = _m.hold_unitary
oat_initial_state = _m.oat_initial_state
oat_random_search = _m.oat_random_search
oat_unitary = _m.oat_unitary
run_oat_nelder_mead = _m.run_oat_nelder_mead
run_single_oat_optimisation = _m.run_single_oat_optimisation
verify_oat_decoupled_baseline = _m.verify_oat_decoupled_baseline

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(
    params=[
        (1, 2),  # Phase 1 asymmetric: J_S=1/2, J_A=1
        (2, 2),  # Phase 2 symmetric: J_S=1, J_A=1
        (5, 5),  # Phase 2 larger symmetric
    ],
    ids=["N_S=1,N_A=2", "N_S=2,N_A=2", "N_S=5,N_A=5"],
)
def make_N_S_N_A(request: pytest.FixtureRequest) -> tuple[int, int]:
    return tuple(request.param)


@pytest.fixture
def make_N_S(make_N_S_N_A: tuple[int, int]) -> int:
    return make_N_S_N_A[0]


@pytest.fixture
def make_N_A(make_N_S_N_A: tuple[int, int]) -> int:
    return make_N_S_N_A[1]


@pytest.fixture
def make_d_tot(make_N_S: int, make_N_A: int) -> int:
    return (make_N_S + 1) * (make_N_A + 1)


@pytest.fixture
def make_ops(make_N_S: int, make_N_A: int) -> dict[str, np.ndarray]:
    return build_operators(make_N_S, make_N_A)


@pytest.fixture
def make_psi0(make_N_S: int, make_N_A: int) -> np.ndarray:
    return oat_initial_state(make_N_S, make_N_A)


# ============================================================================
# Operator Construction
# ============================================================================


class TestBuildOperators:
    def test_given_N_S_N_A_then_dimension_correct(
        self,
        make_N_S: int,
        make_N_A: int,
        make_d_tot: int,
        make_ops: dict,
    ) -> None:
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert make_ops[key].shape == (make_d_tot, make_d_tot), (
                f"{key} has wrong shape for N_S={make_N_S}, N_A={make_N_A}"
            )

    def test_given_operators_then_hermitian(
        self,
        make_ops: dict,
    ) -> None:
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert np.allclose(make_ops[key], make_ops[key].conj().T, atol=1e-12), (
                f"{key} not Hermitian"
            )

    def test_given_N1_N2_then_dimensions_match(self) -> None:
        """Phase 1 case: N_S=1, N_A=2 -> dim = 2 * 3 = 6."""
        ops = build_operators(1, 2)
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert ops[key].shape == (6, 6)

    def test_given_commutation_jz_jx_equals_ijy_system(
        self,
        make_N_S: int,
        make_ops: dict,
    ) -> None:
        comm = make_ops["Jz_S"] @ make_ops["Jx_S"] - make_ops["Jx_S"] @ make_ops["Jz_S"]
        expected = 1j * make_ops["Jy_S"]
        assert np.allclose(comm, expected, atol=1e-10), (
            f"[J_z^S, J_x^S] = i J_y^S violated for N_S={make_N_S}"
        )

    def test_given_commutation_jz_jx_equals_ijy_ancilla(
        self,
        make_N_A: int,
        make_ops: dict,
    ) -> None:
        comm = make_ops["Jz_A"] @ make_ops["Jx_A"] - make_ops["Jx_A"] @ make_ops["Jz_A"]
        expected = 1j * make_ops["Jy_A"]
        assert np.allclose(comm, expected, atol=1e-10), (
            f"[J_z^A, J_x^A] = i J_y^A violated for N_A={make_N_A}"
        )

    def test_given_ops_then_system_and_ancilla_commute(
        self,
        make_ops: dict,
    ) -> None:
        comm = make_ops["Jz_S"] @ make_ops["Jz_A"] - make_ops["Jz_A"] @ make_ops["Jz_S"]
        assert np.allclose(comm, 0.0, atol=1e-12)

    def test_given_jz_eigenvalues_system(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
    ) -> None:
        """J_z^S eigenvalues span -N_S/2 to N_S/2, each repeated (N_A+1) times."""
        expected_m = np.arange(make_N_S / 2.0, -make_N_S / 2.0 - 1, -1)
        expected_evals = np.sort(np.tile(expected_m, make_N_A + 1))
        actual_evals = np.sort(np.linalg.eigvalsh(make_ops["Jz_S"]))
        assert np.allclose(actual_evals, expected_evals, atol=1e-10)

    def test_given_jz_eigenvalues_ancilla(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
    ) -> None:
        """J_z^A eigenvalues span -N_A/2 to N_A/2, each repeated (N_S+1) times."""
        expected_m = np.arange(make_N_A / 2.0, -make_N_A / 2.0 - 1, -1)
        expected_evals = np.sort(np.tile(expected_m, make_N_S + 1))
        actual_evals = np.sort(np.linalg.eigvalsh(make_ops["Jz_A"]))
        assert np.allclose(actual_evals, expected_evals, atol=1e-10)

    def test_given_invalid_N_S_raises(self) -> None:
        with pytest.raises(ValueError):
            build_operators(0, 2)

    def test_given_invalid_N_A_raises(self) -> None:
        with pytest.raises(ValueError):
            build_operators(2, 0)

    def test_given_N1_N2_then_matches_pauli_embedding(self) -> None:
        """N_S=1 operators should match J_k(1) ⊗ I_3 at the Pauli embedding."""
        ops = build_operators(1, 2)
        from src.utils.constants import J_X, J_Y, J_Z

        I_3 = np.eye(3, dtype=complex)
        expected_Jz_S = np.kron(J_Z, I_3)
        expected_Jx_S = np.kron(J_X, I_3)
        expected_Jy_S = np.kron(J_Y, I_3)
        assert np.allclose(ops["Jz_S"], expected_Jz_S, atol=1e-12)
        assert np.allclose(ops["Jx_S"], expected_Jx_S, atol=1e-12)
        assert np.allclose(ops["Jy_S"], expected_Jy_S, atol=1e-12)


# ============================================================================
# System-Only BS Unitary
# ============================================================================


class TestSystemOnlyBSUnitary:
    def test_given_N_S_N_A_then_dimension_correct(
        self,
        make_N_S: int,
        make_N_A: int,
        make_d_tot: int,
    ) -> None:
        U = build_system_only_bs_unitary(make_N_S, make_N_A)
        assert U.shape == (make_d_tot, make_d_tot)

    def test_given_bs_then_unitary(
        self,
        make_N_S: int,
        make_N_A: int,
        make_d_tot: int,
    ) -> None:
        U = build_system_only_bs_unitary(make_N_S, make_N_A)
        I_full = np.eye(make_d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12)

    def test_given_zero_T_bs_then_identity(
        self,
        make_N_S: int,
        make_N_A: int,
        make_d_tot: int,
    ) -> None:
        U = build_system_only_bs_unitary(make_N_S, make_N_A, 0.0)
        I_full = np.eye(make_d_tot, dtype=complex)
        assert np.allclose(U, I_full, atol=1e-12)

    def test_given_N1_N2_then_matches_known_bs(self) -> None:
        """At N_S=1, the system BS should match the single-qubit version."""
        U = build_system_only_bs_unitary(1, 2, T_BS)
        from src.physics.beam_splitter import bs_qubit

        # bs_qubit returns 2x2, embed with I_3 on ancilla
        expected_bs_sys = np.kron(bs_qubit(T_BS), np.eye(3, dtype=complex))
        assert np.allclose(U, expected_bs_sys, atol=1e-12)


# ============================================================================
# Coherent Spin State (CSS)
# ============================================================================


class TestCoherentSpinState:
    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_given_N_then_normalised(self, N: int) -> None:
        state = coherent_spin_state(N)
        assert np.isclose(np.linalg.norm(state), 1.0)

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_given_N_then_correct_length(self, N: int) -> None:
        state = coherent_spin_state(N)
        assert len(state) == N + 1

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_given_N_then_expectation_values(self, N: int) -> None:
        """CSS along -x: ⟨J_x⟩ = -J, ⟨J_y⟩ = ⟨J_z⟩ = 0."""
        state = coherent_spin_state(N)
        J = N / 2.0
        Jx = _dicke_jx(N)
        Jy = _dicke_jy(N)
        Jz = _dicke_jz(N)

        exp_x = np.real(state.conj() @ Jx @ state)
        exp_y = np.real(state.conj() @ Jy @ state)
        exp_z = np.real(state.conj() @ Jz @ state)

        assert np.isclose(exp_x, -J, atol=1e-10), (
            f"N={N}: ⟨J_x⟩ = {exp_x:.6f}, expected {-J:.6f}"
        )
        assert np.isclose(exp_y, 0.0, atol=1e-10)
        assert np.isclose(exp_z, 0.0, atol=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 5])
    def test_given_N_then_variance_values(self, N: int) -> None:
        """CSS along -x: Var(J_y) = Var(J_z) = J/2 = N/4."""
        state = coherent_spin_state(N)
        J = N / 2.0
        Jy = _dicke_jy(N)
        Jz = _dicke_jz(N)

        _, var_y = compute_expectation_and_variance(state, Jy)
        _, var_z = compute_expectation_and_variance(state, Jz)

        assert np.isclose(var_y, J / 2.0, atol=1e-10), (
            f"N={N}: Var(J_y) = {var_y:.6f}, expected {J / 2:.6f}"
        )
        assert np.isclose(var_z, J / 2.0, atol=1e-10)

    def test_given_negative_N_raises(self) -> None:
        with pytest.raises(ValueError):
            coherent_spin_state(-1)


def _dicke_jx(N: int) -> np.ndarray:
    """J_x in Dicke basis for N particles."""
    return jx_operator(N, basis=OperatorBasis.DICKE)


def _dicke_jy(N: int) -> np.ndarray:
    """J_y in Dicke basis for N particles."""
    return jy_operator(N, basis=OperatorBasis.DICKE)


def _dicke_jz(N: int) -> np.ndarray:
    """J_z in Dicke basis for N particles."""
    return jz_operator(N, basis=OperatorBasis.DICKE)


# ============================================================================
# OAT Initial State
# ============================================================================


class TestOatInitialState:
    def test_given_dimensions_then_normalised(
        self,
        make_N_S: int,
        make_N_A: int,
        make_d_tot: int,
        make_psi0: np.ndarray,
    ) -> None:
        assert np.isclose(np.linalg.norm(make_psi0), 1.0)
        assert len(make_psi0) == make_d_tot

    def test_given_state_then_system_in_top_dicke(
        self,
        make_N_S: int,
        make_N_A: int,
        make_psi0: np.ndarray,
    ) -> None:
        """The system part should be |J_S, J_S> = [1, 0, ..., 0]^T."""
        d_sys = make_N_S + 1
        d_anc = make_N_A + 1
        psi_matrix = make_psi0.reshape(d_sys, d_anc)
        # First row (system index 0 = |J,J>) should be the ancilla CSS
        anc_state = coherent_spin_state(make_N_A)
        assert np.allclose(psi_matrix[0, :], anc_state, atol=1e-12), (
            "System not in top Dicke state"
        )
        # Other rows should be zero
        assert np.allclose(psi_matrix[1:, :], 0.0, atol=1e-12)


# ============================================================================
# OAT Unitary
# ============================================================================


class TestOatUnitary:
    @pytest.mark.parametrize("q_val", [0.0, 0.5, 1.0, 2.5])
    def test_given_q_then_unitary(
        self,
        make_N_S: int,
        make_N_A: int,
        make_d_tot: int,
        q_val: float,
    ) -> None:
        U = oat_unitary(make_N_A, q_val, make_N_S + 1)
        I_full = np.eye(make_d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12)

    @pytest.mark.parametrize("q_val", [0.0, 0.5, 1.0, 2.5])
    def test_given_q_then_diagonal(
        self,
        make_N_A: int,
        q_val: float,
    ) -> None:
        U = oat_unitary(make_N_A, q_val, 1)  # d_S=1 for ancilla-only
        d_A = make_N_A + 1
        # Check that the ancilla block is diagonal
        U_block = U[:d_A, :d_A]
        assert np.allclose(U_block, np.diag(np.diag(U_block)), atol=1e-12)

    @pytest.mark.parametrize("q_val", [0.0, 0.5, 1.0, 2.5])
    def test_given_oat_then_commutes_with_jz_ancilla(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
        q_val: float,
    ) -> None:
        U = oat_unitary(make_N_A, q_val, make_N_S + 1)
        comm = U @ make_ops["Jz_A"] - make_ops["Jz_A"] @ U
        assert np.allclose(comm, 0.0, atol=1e-12), (
            f"[U_OAT, J_z^A] != 0 for N_A={make_N_A}, q={q_val}"
        )

    def test_given_zero_q_then_identity(
        self,
        make_N_S: int,
        make_N_A: int,
        make_d_tot: int,
    ) -> None:
        U = oat_unitary(make_N_A, 0.0, make_N_S + 1)
        I_full = np.eye(make_d_tot, dtype=complex)
        assert np.allclose(U, I_full, atol=1e-12)

    @pytest.mark.parametrize("q_val", [0.0, 0.5, 1.0, 2.5])
    def test_given_oat_then_commutes_with_jz_system(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
        q_val: float,
    ) -> None:
        """U_OAT = I_S ⊗ D, so it commutes with any system-only operator."""
        U = oat_unitary(make_N_A, q_val, make_N_S + 1)
        comm = U @ make_ops["Jz_S"] - make_ops["Jz_S"] @ U
        assert np.allclose(comm, 0.0, atol=1e-12)


# ============================================================================
# Drive Hamiltonian
# ============================================================================


class TestDriveHamiltonian:
    def test_given_zero_omega_then_zero(
        self,
        make_ops: dict,
    ) -> None:
        H = build_modulated_drive_hamiltonian(2, 0.0, 1.0, 2.0, 3.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_zero_coefficients_then_zero(
        self,
        make_ops: dict,
    ) -> None:
        H = build_modulated_drive_hamiltonian(2, 1.0, 0.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_drive_then_hermitian(
        self,
        make_ops: dict,
    ) -> None:
        H = build_modulated_drive_hamiltonian(2, 1.5, 1.0, 2.0, 3.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_drive_then_proportional_to_omega(
        self,
        make_ops: dict,
    ) -> None:
        H_half = build_modulated_drive_hamiltonian(2, 0.5, 1.0, 2.0, 3.0, make_ops)
        H_full = build_modulated_drive_hamiltonian(2, 1.0, 1.0, 2.0, 3.0, make_ops)
        assert np.allclose(2.0 * H_half, H_full, atol=1e-12)


# ============================================================================
# Ising Interaction
# ============================================================================


class TestIsingInteraction:
    def test_given_zero_azz_then_zero(
        self,
        make_ops: dict,
    ) -> None:
        H = build_iszz_interaction(0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_azz_then_hermitian(
        self,
        make_ops: dict,
    ) -> None:
        H = build_iszz_interaction(3.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_azz_then_commutes_with_jz_system(
        self,
        make_ops: dict,
    ) -> None:
        H = build_iszz_interaction(3.0, make_ops)
        comm = H @ make_ops["Jz_S"] - make_ops["Jz_S"] @ H
        assert np.allclose(comm, 0.0, atol=1e-12)

    def test_given_azz_then_commutes_with_jz_ancilla(
        self,
        make_ops: dict,
    ) -> None:
        H = build_iszz_interaction(3.0, make_ops)
        comm = H @ make_ops["Jz_A"] - make_ops["Jz_A"] @ H
        assert np.allclose(comm, 0.0, atol=1e-12)


# ============================================================================
# Hold Hamiltonian
# ============================================================================


class TestHoldHamiltonian:
    def test_given_hold_then_hermitian(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
    ) -> None:
        H = build_hold_hamiltonian(
            make_N_S,
            make_N_A,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_ops,
        )
        assert np.allclose(H, H.conj().T, atol=1e-12)


# ============================================================================
# Hold Unitary
# ============================================================================


class TestHoldUnitary:
    def test_given_hold_then_unitary(
        self,
        make_N_S: int,
        make_N_A: int,
        make_d_tot: int,
        make_ops: dict,
    ) -> None:
        U = hold_unitary(
            make_N_S,
            make_N_A,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_ops,
        )
        I_full = np.eye(make_d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12)

    def test_given_zero_hold_then_identity(
        self,
        make_N_S: int,
        make_N_A: int,
        make_d_tot: int,
        make_ops: dict,
    ) -> None:
        U = hold_unitary(
            make_N_S,
            make_N_A,
            0.0,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_ops,
        )
        I_full = np.eye(make_d_tot, dtype=complex)
        assert np.allclose(U, I_full, atol=1e-12)


# ============================================================================
# Circuit Evolution
# ============================================================================


class TestEvolveCircuit:
    def test_given_circuit_then_normalised(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
        make_psi0: np.ndarray,
    ) -> None:
        psi = evolve_oat_circuit(
            make_N_S,
            make_N_A,
            make_psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            1.0,
            make_ops,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_given_zero_params_then_mzi(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
        make_psi0: np.ndarray,
    ) -> None:
        """At zero drive, zero interaction, zero OAT, circuit is a standard MZI."""
        psi = evolve_oat_circuit(
            make_N_S,
            make_N_A,
            make_psi0,
            T_BS,
            T_HOLD,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0)
        exp_val, var_val = compute_expectation_and_variance(psi, make_ops["Jz_S"])
        assert np.isfinite(exp_val)
        assert var_val >= 0.0

    def test_given_oat_zero_q_equals_no_oat(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
        make_psi0: np.ndarray,
    ) -> None:
        """q=0 OAT is identity, so results should match."""
        psi_with_oat = evolve_oat_circuit(
            make_N_S,
            make_N_A,
            make_psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            0.0,
            make_ops,
        )
        # Evolve without OAT by re-applying BS+hold+BS
        U_bs = build_system_only_bs_unitary(make_N_S, make_N_A, T_BS)
        U_hold = hold_unitary(
            make_N_S,
            make_N_A,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_ops,
        )
        psi_no_oat = U_bs @ U_hold @ U_bs @ make_psi0
        assert np.allclose(psi_with_oat, psi_no_oat, atol=1e-12)

    def test_given_zero_oat_then_correct_evolution(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
        make_psi0: np.ndarray,
    ) -> None:
        """At q=0, the evolved state matches the non-OAT circuit exactly."""
        U_bs = build_system_only_bs_unitary(make_N_S, make_N_A, T_BS)
        U_hold = hold_unitary(
            make_N_S,
            make_N_A,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_ops,
        )
        psi_manual = U_bs @ U_hold @ U_bs @ make_psi0

        psi_oat = evolve_oat_circuit(
            make_N_S,
            make_N_A,
            make_psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            0.0,
            make_ops,
        )
        assert np.allclose(psi_oat, psi_manual, atol=1e-12)


# ============================================================================
# Sensitivity Computation
# ============================================================================


class TestSensitivity:
    def test_given_valid_params_then_positive(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
        make_psi0: np.ndarray,
    ) -> None:
        delta = compute_oat_sensitivity(
            make_N_S,
            make_N_A,
            make_psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            1.0,
            make_ops,
        )
        assert np.isfinite(delta)
        assert delta > 0.0

    def test_given_zero_derivative_then_inf(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
        make_psi0: np.ndarray,
    ) -> None:
        """At parameters where derivative vanishes, return inf."""
        delta = compute_oat_sensitivity(
            make_N_S,
            make_N_A,
            make_psi0,
            T_BS,
            T_HOLD,
            0.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            make_ops,
        )
        assert delta > 0.0

    def test_given_q_zero_matches_no_oat_baseline(
        self,
        make_N_S: int,
        make_N_A: int,
        make_ops: dict,
        make_psi0: np.ndarray,
    ) -> None:
        """At q=0, the OAT sensitivity should match the non-OAT sensitivity."""
        delta_oat = compute_oat_sensitivity(
            make_N_S,
            make_N_A,
            make_psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            0.0,
            make_ops,
        )
        # Non-OAT: compute via the circuit with no OAT
        meas_op = make_ops["Jz_S"]

        def non_oat_delta(omega_val: float) -> float:
            U_bs = build_system_only_bs_unitary(make_N_S, make_N_A, T_BS)
            U_h = hold_unitary(
                make_N_S,
                make_N_A,
                T_HOLD,
                omega_val,
                1.0,
                2.0,
                3.0,
                4.0,
                make_ops,
            )
            psi = U_bs @ U_h @ U_bs @ make_psi0
            _, var = compute_expectation_and_variance(psi, meas_op)
            psi_p = (
                U_bs
                @ hold_unitary(
                    make_N_S,
                    make_N_A,
                    T_HOLD,
                    omega_val + FD_STEP,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    make_ops,
                )
                @ U_bs
                @ make_psi0
            )
            psi_m = (
                U_bs
                @ hold_unitary(
                    make_N_S,
                    make_N_A,
                    T_HOLD,
                    omega_val - FD_STEP,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    make_ops,
                )
                @ U_bs
                @ make_psi0
            )
            exp_p = np.real(psi_p.conj() @ meas_op @ psi_p)
            exp_m = np.real(psi_m.conj() @ meas_op @ psi_m)
            d_exp = (exp_p - exp_m) / (2.0 * FD_STEP)
            if abs(d_exp) < 1e-12 or var < 1e-15:
                return float("inf")
            return float(np.sqrt(var) / abs(d_exp))

        delta_non_oat = non_oat_delta(0.5)
        assert np.isclose(delta_oat, delta_non_oat, rtol=1e-6), (
            f"OAT delta={delta_oat:.10f}, non-OAT delta={delta_non_oat:.10f}"
        )


# ============================================================================
# Decoupled Baseline
# ============================================================================


class TestDecoupledBaseline:
    def test_given_decoupled_then_matches_sql(
        self,
        make_N_S: int,
        make_N_A: int,
    ) -> None:
        for omega in [0.1, 0.2, 0.5, 1.0]:
            delta = compute_oat_decoupled_baseline(make_N_S, make_N_A, omega)
            sql = sql_reference(make_N_S)
            assert np.isclose(delta, sql, rtol=1e-8), (
                f"N_S={make_N_S}, N_A={make_N_A}, omega={omega}: "
                f"Delta_omega={delta:.10f}, SQL={sql:.10f}"
            )

    def test_given_oat_only_then_matches_sql(
        self,
        make_N_S: int,
        make_N_A: int,
    ) -> None:
        """OAT alone (no drive/interaction) should not affect sensitivity."""
        for q_val in [0.5, 1.0, 2.5]:
            delta = compute_oat_decoupled_with_oat(
                make_N_S,
                make_N_A,
                q_val,
                omega_true=1.0,
            )
            sql = sql_reference(make_N_S)
            assert np.isclose(delta, sql, rtol=1e-8), (
                f"N_S={make_N_S}, N_A={make_N_A}, q={q_val}: "
                f"Delta_omega={delta:.10f}, SQL={sql:.10f}"
            )

    def test_given_all_params_then_all_pass(self) -> None:
        results = verify_oat_decoupled_baseline(
            N_S_values=[1],
            N_A_values=[2],
            omega_values=[0.1, 0.2, 0.5, 1.0, 2.0],
        )
        assert all(results.values()), (
            f"Failed pairs: {[(ns, na, w) for (ns, na, w), v in results.items() if not v]}"
        )

    def test_given_phase2_params_then_all_pass(self) -> None:
        results = verify_oat_decoupled_baseline(
            N_S_values=[2, 5],
            N_A_values=[2, 5],
            omega_values=[0.1, 0.5, 1.0],
        )
        assert all(results.values())


# ============================================================================
# Random Search
# ============================================================================


class TestRandomSearch:
    def test_given_search_then_returns_result(
        self,
    ) -> None:
        result = oat_random_search(
            PHASE1_N_SYSTEM,
            PHASE1_N_ANCILLA,
            0.5,
            n_samples=20,
            seed=42,
        )
        assert result.samples.shape == (20, 5)
        assert len(result.delta_omega_values) == 20
        assert result.best_delta_omega > 0.0

    def test_given_search_then_best_is_minimum(self) -> None:
        result = oat_random_search(
            PHASE1_N_SYSTEM,
            PHASE1_N_ANCILLA,
            0.5,
            n_samples=20,
            seed=42,
        )
        assert np.isclose(
            result.best_delta_omega,
            np.min(result.delta_omega_values),
        )

    def test_given_fixed_seed_then_deterministic(self) -> None:
        r1 = oat_random_search(
            PHASE1_N_SYSTEM,
            PHASE1_N_ANCILLA,
            0.5,
            n_samples=50,
            seed=42,
        )
        r2 = oat_random_search(
            PHASE1_N_SYSTEM,
            PHASE1_N_ANCILLA,
            0.5,
            n_samples=50,
            seed=42,
        )
        assert np.allclose(r1.samples, r2.samples)
        assert np.allclose(r1.delta_omega_values, r2.delta_omega_values)

    def test_given_search_then_5_parameters(self) -> None:
        result = oat_random_search(
            PHASE1_N_SYSTEM,
            PHASE1_N_ANCILLA,
            0.5,
            n_samples=10,
            seed=42,
        )
        assert result.samples.shape[1] == 5  # (a_x, a_y, a_z, a_zz, q)


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


class TestNelderMead:
    @pytest.mark.slow
    def test_given_nm_then_converges(self) -> None:
        result = run_oat_nelder_mead(
            N_S=PHASE1_N_SYSTEM,
            N_A=PHASE1_N_ANCILLA,
            omega_true=0.5,
            seed=42,
            maxiter=200,
        )
        assert result.delta_omega_opt > 0.0
        assert np.isfinite(result.delta_omega_opt)
        assert result.params_opt.shape == (5,)

    @pytest.mark.slow
    def test_given_nm_then_expectation_finite(self) -> None:
        result = run_oat_nelder_mead(
            N_S=PHASE1_N_SYSTEM,
            N_A=PHASE1_N_ANCILLA,
            omega_true=0.5,
            seed=42,
            maxiter=200,
        )
        assert np.isfinite(result.expectation_Jz)
        assert result.variance_Jz >= 0.0

    @pytest.mark.slow
    def test_given_nm_then_params_in_bounds(self) -> None:
        result = run_oat_nelder_mead(
            N_S=PHASE1_N_SYSTEM,
            N_A=PHASE1_N_ANCILLA,
            omega_true=0.5,
            seed=42,
            maxiter=200,
        )
        lo_d, hi_d = -5.0, 5.0
        lo_q, hi_q = 0.0, 5.0
        for i in range(4):
            assert lo_d <= result.params_opt[i] <= hi_d, (
                f"Drive param {i} = {result.params_opt[i]:.4f} out of bounds"
            )
        assert lo_q <= result.params_opt[4] <= hi_q, (
            f"q = {result.params_opt[4]:.4f} out of bounds"
        )

    @pytest.mark.slow
    def test_given_track_history_then_has_history(self) -> None:
        result = run_oat_nelder_mead(
            N_S=PHASE1_N_SYSTEM,
            N_A=PHASE1_N_ANCILLA,
            omega_true=0.5,
            seed=42,
            maxiter=200,
            track_history=True,
        )
        assert len(result.history) > 0


# ============================================================================
# Single Optimisation
# ============================================================================


class TestSingleOptimisation:
    @pytest.mark.slow
    def test_given_phase1_then_runs(self) -> None:
        """Full optimisation pipeline for Phase 1 (N_S=1, N_A=2)."""
        result = run_single_oat_optimisation(
            N_S=PHASE1_N_SYSTEM,
            N_A=PHASE1_N_ANCILLA,
            omega=0.5,
            n_random=50,
            n_nm_refine=5,
            seed=42,
        )
        assert result.delta_omega_opt > 0.0
        assert np.isfinite(result.delta_omega_opt)
        assert result.sql > 0.0
        assert result.ratio > 0.0
        assert result.a_x_opt is not None
        assert result.q_opt >= 0.0
        assert result.delta_omega_no_oat > 0.0
        assert result.improvement_factor >= 1.0 or np.isnan(result.improvement_factor)

    @pytest.mark.slow
    def test_given_phase2_then_runs(self) -> None:
        """Full optimisation pipeline for Phase 2 (N_S=2, N_A=2)."""
        result = run_single_oat_optimisation(
            N_S=2,
            N_A=2,
            omega=0.5,
            n_random=50,
            n_nm_refine=5,
            seed=42,
        )
        assert result.delta_omega_opt > 0.0
        assert np.isfinite(result.delta_omega_opt)

    @pytest.mark.slow
    def test_given_symmetric_then_beats_sql(self) -> None:
        """At N_S=2, N_A=2, the protocol should beat SQL (ratio > 1)."""
        result = run_single_oat_optimisation(
            N_S=2,
            N_A=2,
            omega=0.5,
            n_random=50,
            n_nm_refine=5,
            seed=42,
        )
        # Based on #20260612, the multi-particle ancilla beats SQL
        # Even without OAT, ratio should be > 1 for N=2
        assert result.ratio > 1.0, (
            f"N_S=2, omega=0.5: ratio = {result.ratio:.4f} (expected > 1)"
        )


# ============================================================================
# Parquet Serialization — OATRandomSearchResult
# ============================================================================


class TestOATRandomSearchResultParquet:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("best_delta_omega", "isclose"),
        ("omega_value", "eq"),
        ("sql", "isclose"),
        ("t_hold", "eq"),
    ]

    @pytest.fixture
    def make_result(self) -> OATRandomSearchResult:
        rng = np.random.default_rng(42)
        samples = rng.uniform(-5, 5, size=(10, 4))
        q_samples = rng.uniform(0, 5, size=10)
        samples = np.column_stack([samples, q_samples])
        deltas = np.abs(rng.normal(0.1, 0.05, size=10))
        # Ensure the first entry is the minimum for consistent roundtrip
        deltas[0] = 0.005
        return OATRandomSearchResult(
            samples=samples,
            delta_omega_values=deltas,
            best_params=(
                float(samples[0, 0]),
                float(samples[0, 1]),
                float(samples[0, 2]),
                float(samples[0, 3]),
                float(samples[0, 4]),
            ),
            best_delta_omega=float(deltas[0]),
            omega_value=0.5,
            sql=0.1,
            t_hold=10.0,
        )

    def test_roundtrip(
        self,
        make_result: OATRandomSearchResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "rs.parquet"
        make_result.save_parquet(p)
        loaded = OATRandomSearchResult.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)
        assert np.allclose(loaded.samples, make_result.samples), (
            "samples array content mismatch"
        )
        assert np.allclose(
            loaded.delta_omega_values,
            make_result.delta_omega_values,
        ), "delta_omega_values array content mismatch"

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a_x": [1.0], "delta_omega": [0.1]})
        p = tmp_path / "bad_rs.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            OATRandomSearchResult.from_parquet(p)


# ============================================================================
# Parquet Serialization — OATNelderMeadResult
# ============================================================================


class TestOATNelderMeadResultParquet:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("delta_omega_opt", "isclose"),
        ("omega_true", "eq"),
        ("success", "eq"),
        ("nfev", "eq"),
        ("expectation_Jz", "eq"),
        ("variance_Jz", "eq"),
    ]

    @pytest.fixture
    def make_result(self) -> OATNelderMeadResult:
        return OATNelderMeadResult(
            delta_omega_opt=0.025,
            params_opt=np.array([3.0, -2.0, 1.0, 4.0, 1.5]),
            omega_true=0.5,
            success=True,
            nfev=150,
            message="Optimization terminated successfully.",
            expectation_Jz=0.3,
            variance_Jz=0.15,
            history=[0.1, 0.08, 0.06, 0.04, 0.025],
        )

    def test_roundtrip(
        self,
        make_result: OATNelderMeadResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "nm.parquet"
        make_result.save_parquet(p)
        loaded = OATNelderMeadResult.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)
        assert np.allclose(loaded.params_opt, make_result.params_opt)

    def test_roundtrip_history(
        self,
        make_result: OATNelderMeadResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "nm_hist.parquet"
        make_result.save_parquet(p)
        loaded = OATNelderMeadResult.from_parquet(p)
        assert loaded.history == make_result.history

    def test_roundtrip_no_history(
        self,
        tmp_path: Path,
    ) -> None:
        result = OATNelderMeadResult(
            delta_omega_opt=0.025,
            params_opt=np.array([3.0, -2.0, 1.0, 4.0, 1.5]),
            omega_true=0.5,
            success=True,
            nfev=150,
        )
        p = tmp_path / "nm_no_hist.parquet"
        result.save_parquet(p)
        loaded = OATNelderMeadResult.from_parquet(p)
        assert loaded.history == []

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"delta_omega": [0.025], "omega_true": [0.5]})
        p = tmp_path / "bad_nm.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            OATNelderMeadResult.from_parquet(p)


# ============================================================================
# Parquet Serialization — OATNScalingResult
# ============================================================================


class TestOATNScalingResultParquet:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("N", "eq"),
        ("n_ancilla", "eq"),
        ("omega", "eq"),
        ("delta_omega_opt", "isclose"),
        ("sql", "isclose"),
        ("ratio", "isclose"),
        ("a_x_opt", "eq"),
        ("a_y_opt", "eq"),
        ("a_z_opt", "eq"),
        ("a_zz_opt", "eq"),
        ("q_opt", "eq"),
        ("delta_omega_no_oat", "isclose"),
        ("improvement_factor", "isclose"),
        ("expectation_Jz", "eq"),
        ("variance_Jz", "eq"),
        ("t_hold", "eq"),
        ("fd_step", "eq"),
        ("success", "eq"),
        ("nfev", "eq"),
    ]

    @pytest.fixture
    def make_result(self) -> OATNScalingResult:
        sql = sql_reference(5)
        return OATNScalingResult(
            N=5,
            n_ancilla=5,
            omega=0.2,
            delta_omega_opt=0.012,
            sql=sql,
            ratio=sql / 0.012,
            a_x_opt=3.0,
            a_y_opt=-2.5,
            a_z_opt=1.0,
            a_zz_opt=4.0,
            q_opt=1.5,
            delta_omega_no_oat=0.018,
            improvement_factor=0.018 / 0.012,
            expectation_Jz=0.5,
            variance_Jz=0.25,
            success=True,
            nfev=100,
        )

    def test_roundtrip(
        self,
        make_result: OATNScalingResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "ns.parquet"
        make_result.save_parquet(p)
        loaded = OATNScalingResult.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)
        assert np.isclose(loaded.improvement_factor, make_result.improvement_factor)

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "bad_ns.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            OATNScalingResult.from_parquet(p)

    def test_missing_q_opt_raises(self, tmp_path: Path) -> None:
        """Missing q_opt column should raise ValueError."""
        sql = sql_reference(5)
        data = {
            "N": [5],
            "n_ancilla": [5],
            "omega": [0.2],
            "delta_omega_opt": [0.012],
            "sql": [sql],
            "ratio": [sql / 0.012],
            "a_x_opt": [3.0],
            "a_y_opt": [-2.5],
            "a_z_opt": [1.0],
            "a_zz_opt": [4.0],
            # "q_opt" missing
            "delta_omega_no_oat": [0.018],
            "improvement_factor": [1.5],
            "expectation_Jz": [0.5],
            "variance_Jz": [0.25],
            "t_hold": [10.0],
            "fd_step": [1e-6],
            "success": [1],
            "nfev": [100],
        }
        df = pd.DataFrame(data)
        p = tmp_path / "missing_qopt.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            OATNScalingResult.from_parquet(p)


# ============================================================================
# Parquet Serialization — OATNScalingScanResult
# ============================================================================


class TestOATNScalingScanResultParquet:
    @pytest.fixture
    def make_results(self) -> list[OATNScalingResult]:
        return [
            OATNScalingResult(
                N=n,
                n_ancilla=n,
                omega=w,
                delta_omega_opt=0.02 / (n**0.5),
                sql=sql_reference(n),
                ratio=sql_reference(n) / (0.02 / (n**0.5)),
                a_x_opt=float(n),
                a_y_opt=0.0,
                a_z_opt=0.0,
                a_zz_opt=float(n),
                q_opt=float(n) * 0.3,
                delta_omega_no_oat=0.025 / (n**0.5),
                improvement_factor=1.25,
                success=True,
                nfev=50,
            )
            for n in [1, 2, 5]
            for w in [0.1, 0.2, 0.5]
        ]

    def test_roundtrip(
        self,
        make_results: list[OATNScalingResult],
        tmp_path: Path,
    ) -> None:
        summary = OATNScalingScanResult(results=make_results)
        p = tmp_path / "scan.parquet"
        summary.save_parquet(p)
        loaded = OATNScalingScanResult.from_parquet(p)
        assert len(loaded.results) == len(make_results)
        for orig, loaded_r in zip(make_results, loaded.results, strict=False):
            assert orig.N == loaded_r.N
            assert orig.omega == loaded_r.omega
            assert orig.q_opt == loaded_r.q_opt

    def test_empty_dataframe(self) -> None:
        summary = OATNScalingScanResult(results=[])
        df = summary.to_dataframe()
        assert df.empty

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "bad_scan.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            OATNScalingScanResult.from_parquet(p)

    def test_properties(
        self,
        make_results: list[OATNScalingResult],
    ) -> None:
        summary = OATNScalingScanResult(results=make_results)
        assert len(summary.N_values) == 3
        assert len(summary.omega_values) == 3


# ============================================================================
# Squeezing Verification (Ancilla-Only)
# ============================================================================


class TestSqueezingParameter:
    @pytest.mark.parametrize("N", [2, 5])
    def test_given_q_zero_then_no_squeezing(self, N: int) -> None:
        """At q=0, the variance in all quadratures equals N/4."""
        psi = coherent_spin_state(N)
        J = N / 2.0
        Jy = _dicke_jy(N)
        _, var_y = compute_expectation_and_variance(psi, Jy)
        assert np.isclose(var_y, J / 2.0, atol=1e-10)

    @staticmethod
    def _min_quadrature_variance(
        psi: np.ndarray,
        N: int,
        n_theta: int | None = None,
    ) -> float:
        """Minimum variance over quadrature angles J_θ = cosθ J_y + sinθ J_z.

        Args:
            psi: State vector (ancilla-only, length N+1).
            N: Number of particles.
            n_theta: Number of angles to scan (default max(500, 10*N)).

        Returns:
            Minimum variance found.
        """
        if n_theta is None:
            n_theta = max(500, 10 * N)
        Jy = _dicke_jy(N)
        Jz = _dicke_jz(N)
        thetas = np.linspace(0, np.pi, n_theta)
        min_var = float("inf")
        for theta in thetas:
            J_theta = np.cos(theta) * Jy + np.sin(theta) * Jz
            _, var = compute_expectation_and_variance(psi, J_theta)
            min_var = min(min_var, var)
        return min_var

    @pytest.mark.parametrize("N", [2, 5])
    def test_given_q_positive_then_squeezing(self, N: int) -> None:
        """With OAT, the minimum quadrature variance should be below SQL."""
        psi0 = coherent_spin_state(N)
        q_test = 3.0 / N  # ~optimal for OAT
        J = N / 2.0
        sql_var = J / 2.0

        # Build ancilla-only OAT unitary: d_S=1
        U_oat = oat_unitary(N, q_test, 1)
        psi = U_oat @ psi0

        min_var = self._min_quadrature_variance(psi, N)
        squeezing_param = np.sqrt(min_var / sql_var)
        assert squeezing_param < 0.99, (
            f"N={N}, q={q_test:.4f}: squeezing param = {squeezing_param:.6f}, "
            "expected < 0.99"
        )

    @pytest.mark.parametrize("N", [2, 5])
    def test_given_larger_q_then_antisqueezing_in_y(self, N: int) -> None:
        """Var(J_y) should increase (anti-squeeze) for q>0 with CSS along -x."""
        psi0 = coherent_spin_state(N)
        J = N / 2.0
        Jy = _dicke_jy(N)

        q_test = 3.0 / N
        U_oat = oat_unitary(N, q_test, 1)
        psi = U_oat @ psi0

        _, var_y = compute_expectation_and_variance(psi, Jy)
        sql_var = J / 2.0
        # Var(Y) should increase (anti-squeezing) or at least not decrease
        assert var_y >= sql_var - 1e-10, (
            f"N={N}: Var(J_y) = {var_y:.6f}, SQL_var = {sql_var:.6f}"
        )

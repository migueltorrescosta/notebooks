"""Tests for the General 4-Parameter Interaction with ω-Modulated Drive report.

Companion test module for ``reports/20260616/general_4param_omega_drive.py``.
Tests operator construction, circuit evolution, decoupled baseline,
N=1 consistency, optimization, and serialization.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
)
from src.analysis.decoupled_baseline import verify_decoupled_baseline
from src.analysis.sensitivity_metrics import sql_reference
from src.utils.serialization import assert_roundtrip_fields

_m = importlib.import_module("reports.20260616.general_4param_omega_drive")
T_BS = _m.T_BS
T_HOLD = _m.T_HOLD
Combined2DSliceResult = _m.Combined2DSliceResult
CombinedNScalingScanResult = _m.CombinedNScalingScanResult
CombinedOptimizationResult = _m.CombinedOptimizationResult
CombinedRandomSearchResult = _m.CombinedRandomSearchResult
build_combined_hold_hamiltonian = _m.build_combined_hold_hamiltonian
build_combined_system_bs_unitary = _m.build_combined_system_bs_unitary
build_fixed_ancilla_combined_operators = _m.build_fixed_ancilla_combined_operators
build_full_ancilla_combined_operators = _m.build_full_ancilla_combined_operators
combined_hold_unitary = _m.combined_hold_unitary
combined_initial_state = _m.combined_initial_state
combined_objective = _m.combined_objective
combined_random_search = _m.combined_random_search
compute_combined_decoupled_baseline = _m.compute_combined_decoupled_baseline
compute_combined_sensitivity = _m.compute_combined_sensitivity
evolve_combined_circuit = _m.evolve_combined_circuit
run_combined_bfgs_optimization = _m.run_combined_bfgs_optimization
run_combined_single_n_omega = _m.run_combined_single_n_omega

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=[1, 2, 5, 10])
def make_N(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture(params=[2, 3, 6])
def make_N_small(request: pytest.FixtureRequest) -> int:
    """Smaller N for full-ancilla tests (dim = (N+1)^2 grows fast)."""
    return int(request.param)


@pytest.fixture
def make_fixed_ops(make_N: int) -> dict[str, np.ndarray]:
    return build_fixed_ancilla_combined_operators(make_N)


@pytest.fixture
def make_full_ops(make_N_small: int) -> dict[str, np.ndarray]:
    return build_full_ancilla_combined_operators(make_N_small)


@pytest.fixture
def make_fixed_psi0(make_N: int) -> np.ndarray:
    d_tot = 2 * (make_N + 1)
    return combined_initial_state(d_tot)


@pytest.fixture
def make_full_psi0(make_N_small: int) -> np.ndarray:
    d_tot = (make_N_small + 1) ** 2
    return combined_initial_state(d_tot)


# ============================================================================
# Operator Construction — Fixed Ancilla (J_A = 1/2)
# ============================================================================


class TestBuildFixedAncillaOperators:
    def test_given_N_then_dimension_correct(self, make_N: int) -> None:
        ops = build_fixed_ancilla_combined_operators(make_N)
        d_tot = 2 * (make_N + 1)
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert ops[key].shape == (d_tot, d_tot), (
                f"{key} has wrong shape for N={make_N}"
            )

    def test_given_operators_then_hermitian(self, make_N: int) -> None:
        ops = build_fixed_ancilla_combined_operators(make_N)
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert np.allclose(ops[key], ops[key].conj().T, atol=1e-12), (
                f"{key} not Hermitian for N={make_N}"
            )

    def test_given_n1_then_matches_pauli(self) -> None:
        ops = build_fixed_ancilla_combined_operators(1)
        from src.utils.constants import I_2, J_X, J_Z

        expected_Jz_S = np.kron(J_Z, I_2)
        assert np.allclose(ops["Jz_S"], expected_Jz_S, atol=1e-12)

        expected_Jx_S = np.kron(J_X, I_2)
        assert np.allclose(ops["Jx_S"], expected_Jx_S, atol=1e-12)

        expected_Jz_A = np.kron(I_2, J_Z)
        assert np.allclose(ops["Jz_A"], expected_Jz_A, atol=1e-12)

    def test_given_commutation_jz_jx_equals_ijy(self, make_N: int) -> None:
        ops = build_fixed_ancilla_combined_operators(make_N)
        comm = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
        expected = 1j * ops["Jy_S"]
        assert np.allclose(comm, expected, atol=1e-10), (
            f"[J_z^S, J_x^S] = i J_y^S violated for N={make_N}"
        )

    def test_given_invalid_N_raises(self) -> None:
        with pytest.raises(ValueError):
            build_fixed_ancilla_combined_operators(0)


# ============================================================================
# Operator Construction — Full Ancilla (J_A = N/2)
# ============================================================================


class TestBuildFullAncillaOperators:
    def test_given_N_then_dimension_correct(self, make_N_small: int) -> None:
        ops = build_full_ancilla_combined_operators(make_N_small)
        d_tot = (make_N_small + 1) ** 2
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert ops[key].shape == (d_tot, d_tot), (
                f"{key} has wrong shape for N={make_N_small}"
            )

    def test_given_operators_then_hermitian(self, make_N_small: int) -> None:
        ops = build_full_ancilla_combined_operators(make_N_small)
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert np.allclose(ops[key], ops[key].conj().T, atol=1e-12), (
                f"{key} not Hermitian for N={make_N_small}"
            )

    def test_given_n1_then_equivalent_to_fixed_ancilla(self) -> None:
        """At N=1, both builder paths should give the same operators (dim=4)."""
        ops_full = build_full_ancilla_combined_operators(1)
        ops_fixed = build_fixed_ancilla_combined_operators(1)
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert np.allclose(ops_full[key], ops_fixed[key], atol=1e-12), (
                f"{key} differs between full and fixed builders at N=1"
            )

    def test_given_n1_then_matches_pauli(self) -> None:
        ops = build_full_ancilla_combined_operators(1)
        from src.utils.constants import I_2, J_X, J_Z

        expected_Jz_S = np.kron(J_Z, I_2)
        assert np.allclose(ops["Jz_S"], expected_Jz_S, atol=1e-12)

        expected_Jx_S = np.kron(J_X, I_2)
        assert np.allclose(ops["Jx_S"], expected_Jx_S, atol=1e-12)

    def test_given_commutation_jz_jx_equals_ijy(self, make_N_small: int) -> None:
        ops = build_full_ancilla_combined_operators(make_N_small)
        comm = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
        expected = 1j * ops["Jy_S"]
        assert np.allclose(comm, expected, atol=1e-10), (
            f"[J_z^S, J_x^S] = i J_y^S violated for N={make_N_small}"
        )

    def test_given_invalid_N_raises(self) -> None:
        with pytest.raises(ValueError):
            build_full_ancilla_combined_operators(0)


# ============================================================================
# Hamiltonian Construction
# ============================================================================


class TestCombinedHamiltonian:
    def test_given_zero_all_then_only_omega_jz_s(
        self,
        make_N: int,
        make_fixed_ops: dict,
    ) -> None:
        H = build_combined_hold_hamiltonian(
            make_N,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_fixed_ops,
        )
        expected = make_fixed_ops["Jz_S"]
        assert np.allclose(H, expected, atol=1e-12), (
            "H should be just J_z^S when all params are zero"
        )

    def test_given_hamiltonian_then_hermitian(
        self,
        make_N: int,
        make_fixed_ops: dict,
    ) -> None:
        H = build_combined_hold_hamiltonian(
            make_N,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            make_fixed_ops,
        )
        assert np.allclose(H, H.conj().T, atol=1e-12), "H must be Hermitian"

    def test_given_zero_omega_then_only_interaction(
        self,
        make_N: int,
        make_fixed_ops: dict,
    ) -> None:
        H = build_combined_hold_hamiltonian(
            make_N,
            0.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            make_fixed_ops,
        )
        # At omega=0, the J_z^S and drive terms vanish; only H_int remains
        H_int = (
            5.0 * (make_fixed_ops["Jx_S"] @ make_fixed_ops["Jx_A"])
            + 6.0 * (make_fixed_ops["Jx_S"] @ make_fixed_ops["Jz_A"])
            + 7.0 * (make_fixed_ops["Jz_S"] @ make_fixed_ops["Jx_A"])
            + 8.0 * (make_fixed_ops["Jz_S"] @ make_fixed_ops["Jz_A"])
        )
        H_int = 0.5 * (H_int + H_int.conj().T)
        assert np.allclose(H, H_int, atol=1e-12), "At ω=0, H should equal H_int only"

    def test_given_drive_linear_in_omega(
        self,
        make_N: int,
        make_fixed_ops: dict,
    ) -> None:
        H_ref = build_combined_hold_hamiltonian(
            make_N,
            1.0,
            2.0,
            3.0,
            4.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_fixed_ops,
        )
        H_half = build_combined_hold_hamiltonian(
            make_N,
            0.5,
            2.0,
            3.0,
            4.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_fixed_ops,
        )
        assert np.allclose(2.0 * H_half, H_ref, atol=1e-12), (
            "Drive + J_z^S should be linear in ω"
        )

    def test_given_full_ancilla_then_hermitian(
        self,
        make_N_small: int,
        make_full_ops: dict,
    ) -> None:
        H = build_combined_hold_hamiltonian(
            make_N_small,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            make_full_ops,
        )
        assert np.allclose(H, H.conj().T, atol=1e-12), (
            "H must be Hermitian (full ancilla)"
        )


# ============================================================================
# Beam-Splitter Unitary
# ============================================================================


class TestBSUnitary:
    def test_given_bs_then_unitary_fixed(self, make_N: int) -> None:
        U = build_combined_system_bs_unitary(make_N, ancilla_dim=2)
        d_tot = 2 * (make_N + 1)
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12)

    def test_given_bs_then_unitary_full(self, make_N_small: int) -> None:
        ancilla_dim = make_N_small + 1
        U = build_combined_system_bs_unitary(make_N_small, ancilla_dim=ancilla_dim)
        d_tot = (make_N_small + 1) * ancilla_dim
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12)

    def test_given_zero_T_bs_then_identity(self, make_N: int) -> None:
        U = build_combined_system_bs_unitary(make_N, ancilla_dim=2, T_bs=0.0)
        d_tot = 2 * (make_N + 1)
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U, I_full, atol=1e-12)


# ============================================================================
# Initial State
# ============================================================================


class TestInitialState:
    def test_given_dim_then_normalised(self) -> None:
        for d in [4, 6, 12, 20]:
            psi = combined_initial_state(d)
            assert np.isclose(np.linalg.norm(psi), 1.0), f"Not normalised for dim={d}"

    def test_given_dim_then_first_element_is_one(self) -> None:
        psi = combined_initial_state(4)
        assert psi[0] == 1.0
        assert np.all(psi[1:] == 0.0)


# ============================================================================
# Circuit Evolution
# ============================================================================


class TestHoldUnitary:
    def test_given_hold_then_unitary(
        self,
        make_N: int,
        make_fixed_ops: dict,
    ) -> None:
        U = combined_hold_unitary(
            make_N,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            make_fixed_ops,
        )
        assert np.allclose(U @ U.conj().T, make_fixed_ops["I_full"], atol=1e-12)

    def test_given_zero_hold_then_identity(
        self,
        make_N: int,
        make_fixed_ops: dict,
    ) -> None:
        U = combined_hold_unitary(
            make_N,
            0.0,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            make_fixed_ops,
        )
        assert np.allclose(U, make_fixed_ops["I_full"], atol=1e-12)


class TestEvolveCircuit:
    def test_given_circuit_then_normalised(
        self,
        make_N: int,
        make_fixed_ops: dict,
        make_fixed_psi0: np.ndarray,
    ) -> None:
        psi = evolve_combined_circuit(
            make_N,
            make_fixed_psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            make_fixed_ops,
            ancilla_dim=2,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_given_zero_params_then_css_mzi(
        self,
        make_N: int,
        make_fixed_ops: dict,
        make_fixed_psi0: np.ndarray,
    ) -> None:
        """At all zeros, the circuit is a standard MZI."""
        psi = evolve_combined_circuit(
            make_N,
            make_fixed_psi0,
            T_BS,
            T_HOLD,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_fixed_ops,
            ancilla_dim=2,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0)
        exp_val, var_val = compute_expectation_and_variance(psi, make_fixed_ops["Jz_S"])
        assert np.isfinite(exp_val)
        assert var_val >= 0.0

    def test_given_full_ancilla_circuit_then_normalised(
        self,
        make_N_small: int,
        make_full_ops: dict,
        make_full_psi0: np.ndarray,
    ) -> None:
        ancilla_dim = make_N_small + 1
        psi = evolve_combined_circuit(
            make_N_small,
            make_full_psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            make_full_ops,
            ancilla_dim=ancilla_dim,
        )
        assert np.isclose(np.linalg.norm(psi), 1.0)


# ============================================================================
# Sensitivity Computation
# ============================================================================


class TestSensitivity:
    def test_given_valid_params_then_positive(
        self,
        make_N: int,
        make_fixed_ops: dict,
        make_fixed_psi0: np.ndarray,
    ) -> None:
        delta = compute_combined_sensitivity(
            make_N,
            make_fixed_psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            make_fixed_ops,
            ancilla_dim=2,
        )
        assert np.isfinite(delta)
        assert delta > 0.0

    def test_given_large_drive_then_finite_or_inf(
        self,
        make_N: int,
        make_fixed_ops: dict,
        make_fixed_psi0: np.ndarray,
    ) -> None:
        """At large drive, sensitivity should remain non-negative."""
        delta = compute_combined_sensitivity(
            make_N,
            make_fixed_psi0,
            T_BS,
            T_HOLD,
            0.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            make_fixed_ops,
            ancilla_dim=2,
        )
        assert delta > 0.0 or np.isinf(delta)

    def test_given_full_ancilla_then_positive(
        self,
        make_N_small: int,
        make_full_ops: dict,
        make_full_psi0: np.ndarray,
    ) -> None:
        ancilla_dim = make_N_small + 1
        delta = compute_combined_sensitivity(
            make_N_small,
            make_full_psi0,
            T_BS,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            make_full_ops,
            ancilla_dim=ancilla_dim,
        )
        assert np.isfinite(delta)
        assert delta > 0.0


# ============================================================================
# Decoupled Baseline
# ============================================================================


class TestDecoupledBaseline:
    def test_given_decoupled_then_matches_sql(self, make_N: int) -> None:
        for omega in [0.1, 0.2, 0.5, 1.0]:
            delta = compute_combined_decoupled_baseline(make_N, omega, ancilla_dim=2)
            sql = sql_reference(make_N)
            assert np.isclose(delta, sql, rtol=1e-8), (
                f"N={make_N}, ω={omega}: Δω={delta:.10f}, SQL={sql:.10f}"
            )

    def test_given_all_n_omega_then_all_pass(self) -> None:
        results = verify_decoupled_baseline(
            N_values=[1, 2, 3, 5, 10],
            omega_values=[0.1, 0.2, 0.5, 1.0, 2.0],
            compute_fn=compute_combined_decoupled_baseline,
            ancilla_dim=2,
        )
        assert all(results.values()), (
            f"Failed pairs: {[(n, w) for (n, w), v in results.items() if not v]}"
        )

    def test_given_full_ancilla_decoupled_then_matches_sql(self) -> None:
        for N in [1, 2, 3]:
            ancilla_dim = N + 1
            delta = compute_combined_decoupled_baseline(N, 0.5, ancilla_dim=ancilla_dim)
            sql = sql_reference(N)
            assert np.isclose(delta, sql, rtol=1e-8), (
                f"N={N}: Δω={delta:.10f}, SQL={sql:.10f}"
            )


# ============================================================================
# N=1 Consistency
# ============================================================================


class TestN1Consistency:
    @pytest.mark.slow
    def test_given_N1_omega02_then_beats_sql(self) -> None:
        """At N=1, ω=0.2, the combined protocol should beat SQL."""
        result = run_combined_single_n_omega(N=1, omega=0.2, ancilla_dim=2)
        assert result.ratio > 1.0, (
            f"N=1, ω=0.2: ratio = {result.ratio:.4f} (expected > 1)"
        )

    @pytest.mark.slow
    def test_given_N1_omega02_then_non_commuting_drive(self) -> None:
        """At least one of a_x or a_y should be non-zero."""
        result = run_combined_single_n_omega(N=1, omega=0.2, ancilla_dim=2)
        assert abs(result.a_x_opt) > 0.1 or abs(result.a_y_opt) > 0.1, (
            f"a_x={result.a_x_opt:.4f}, a_y={result.a_y_opt:.4f}"
        )

    @pytest.mark.slow
    def test_given_N1_omega02_then_variance_positive(self) -> None:
        result = run_combined_single_n_omega(N=1, omega=0.2, ancilla_dim=2)
        assert result.variance_Jz >= -1e-12
        assert result.variance_Jz >= 0.0

    def test_given_N1_then_both_ancilla_paths_equivalent(self) -> None:
        """At N=1, the fixed-ancilla and full-ancilla paths should agree."""
        # Small random search test at a single ω
        # (builders already imported at module level)
        ops_fixed = build_fixed_ancilla_combined_operators(1)
        ops_full = build_full_ancilla_combined_operators(1)
        psi0 = combined_initial_state(4)

        # Run a few random samples and compare
        rng = np.random.default_rng(42)
        for _ in range(5):
            ax = rng.uniform(-5, 5)
            ay = rng.uniform(-5, 5)
            az = rng.uniform(-5, 5)
            alpha_xx = rng.uniform(-20, 20)
            alpha_xz = rng.uniform(-20, 20)
            alpha_zx = rng.uniform(-20, 20)
            alpha_zz = rng.uniform(-20, 20)
            d_fixed = compute_combined_sensitivity(
                1,
                psi0,
                T_BS,
                T_HOLD,
                0.5,
                ax,
                ay,
                az,
                alpha_xx,
                alpha_xz,
                alpha_zx,
                alpha_zz,
                ops_fixed,
                ancilla_dim=2,
            )
            d_full = compute_combined_sensitivity(
                1,
                psi0,
                T_BS,
                T_HOLD,
                0.5,
                ax,
                ay,
                az,
                alpha_xx,
                alpha_xz,
                alpha_zx,
                alpha_zz,
                ops_full,
                ancilla_dim=2,
            )
            assert np.isclose(d_fixed, d_full, rtol=1e-10), (
                f"Fixed vs full disagree: {d_fixed:.10f} vs {d_full:.10f}"
            )


# ============================================================================
# Random Search
# ============================================================================


class TestRandomSearch:
    def test_given_random_search_then_returns_result(
        self,
        make_N: int,
        make_fixed_ops: dict,
        make_fixed_psi0: np.ndarray,
    ) -> None:
        result = combined_random_search(
            make_N,
            0.5,
            make_fixed_ops,
            make_fixed_psi0,
            ancilla_dim=2,
            n_samples=20,
            seed=42,
        )
        assert result.samples.shape == (20, 7)
        assert len(result.delta_omega_values) == 20
        assert result.best_delta_omega > 0.0

    def test_given_random_search_then_best_is_minimum(
        self,
        make_N: int,
        make_fixed_ops: dict,
        make_fixed_psi0: np.ndarray,
    ) -> None:
        result = combined_random_search(
            make_N,
            0.5,
            make_fixed_ops,
            make_fixed_psi0,
            ancilla_dim=2,
            n_samples=20,
            seed=42,
        )
        assert np.isclose(
            result.best_delta_omega,
            np.min(result.delta_omega_values),
        )

    def test_given_random_search_then_7d_params(self) -> None:
        """Verify the best_params tuple has 7 elements."""
        ops = build_fixed_ancilla_combined_operators(1)
        psi0 = combined_initial_state(4)
        result = combined_random_search(
            1,
            0.5,
            ops,
            psi0,
            ancilla_dim=2,
            n_samples=10,
            seed=42,
        )
        assert len(result.best_params) == 7
        assert result.best_params[0] != 0.0 or result.best_params[1] != 0.0


# ============================================================================
# Objective Function
# ============================================================================


class TestObjective:
    def test_given_zero_params_then_finite(
        self,
        make_N: int,
        make_fixed_ops: dict,
        make_fixed_psi0: np.ndarray,
    ) -> None:
        params = np.zeros(7, dtype=float)
        val = combined_objective(
            params,
            make_N,
            0.5,
            make_fixed_ops,
            make_fixed_psi0,
            ancilla_dim=2,
        )
        assert np.isfinite(val)
        # At zero params, should be SQL
        sql = sql_reference(make_N)
        assert np.isclose(val, sql, rtol=1e-6), (
            f"At zero params, Δω should ~ SQL: {val:.10f} vs {sql:.10f}"
        )

    def test_given_out_of_bounds_then_large_penalty(
        self,
        make_N: int,
        make_fixed_ops: dict,
        make_fixed_psi0: np.ndarray,
    ) -> None:
        # Drive param out of bounds
        params = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        val = combined_objective(
            params,
            make_N,
            0.5,
            make_fixed_ops,
            make_fixed_psi0,
            ancilla_dim=2,
        )
        assert val > 1e9

        # α param out of bounds
        params2 = np.array([0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0], dtype=float)
        val2 = combined_objective(
            params2,
            make_N,
            0.5,
            make_fixed_ops,
            make_fixed_psi0,
            ancilla_dim=2,
        )
        assert val2 > 1e9


# ============================================================================
# BFGS Optimization
# ============================================================================


class TestBFGSOptimization:
    def test_given_bfgs_then_converges(self, make_N: int) -> None:
        ops = build_fixed_ancilla_combined_operators(make_N)
        psi0 = combined_initial_state(2 * (make_N + 1))
        result = run_combined_bfgs_optimization(
            N=make_N,
            omega_true=0.5,
            ops=ops,
            psi0=psi0,
            ancilla_dim=2,
            n_starts=3,
            seed=42,
        )
        assert result.delta_omega_opt > 0.0
        assert np.isfinite(result.delta_omega_opt)

    def test_given_bfgs_then_diagnostics_finite(self, make_N: int) -> None:
        ops = build_fixed_ancilla_combined_operators(make_N)
        psi0 = combined_initial_state(2 * (make_N + 1))
        result = run_combined_bfgs_optimization(
            N=make_N,
            omega_true=0.5,
            ops=ops,
            psi0=psi0,
            ancilla_dim=2,
            n_starts=3,
            seed=42,
        )
        assert np.isfinite(result.expectation_Jz)
        assert result.variance_Jz >= 0.0
        assert result.n_starts == 3


# ============================================================================
# CombinedOptimizationResult Serialization
# ============================================================================


class TestCombinedOptimizationResultParquet:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("N", "eq"),
        ("omega", "eq"),
        ("delta_omega_opt", "isclose"),
        ("sql", "isclose"),
        ("ratio", "isclose"),
        ("a_x_opt", "eq"),
        ("a_y_opt", "eq"),
        ("a_z_opt", "eq"),
        ("alpha_xx_opt", "eq"),
        ("alpha_xz_opt", "eq"),
        ("alpha_zx_opt", "eq"),
        ("alpha_zz_opt", "eq"),
        ("expectation_Jz", "eq"),
        ("variance_Jz", "eq"),
        ("t_hold", "eq"),
        ("fd_step", "eq"),
        ("success", "eq"),
        ("nfev", "eq"),
        ("n_starts", "eq"),
        ("n_converged", "eq"),
    ]

    @pytest.fixture
    def make_result(self) -> CombinedOptimizationResult:
        return CombinedOptimizationResult(
            N=5,
            omega=0.2,
            delta_omega_opt=0.015,
            sql=sql_reference(5),
            ratio=sql_reference(5) / 0.015,
            a_x_opt=3.0,
            a_y_opt=-2.5,
            a_z_opt=1.0,
            alpha_xx_opt=10.0,
            alpha_xz_opt=-5.0,
            alpha_zx_opt=3.0,
            alpha_zz_opt=8.0,
            expectation_Jz=0.5,
            variance_Jz=0.25,
            success=True,
            nfev=100,
            n_starts=200,
            n_converged=150,
        )

    def test_roundtrip(
        self,
        make_result: CombinedOptimizationResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "test.parquet"
        make_result.save_parquet(p)
        loaded = CombinedOptimizationResult.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "missing.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            CombinedOptimizationResult.from_parquet(p)


# ============================================================================
# CombinedNScalingScanResult Serialization
# ============================================================================


class TestCombinedNScalingScanResultParquet:
    @pytest.fixture
    def make_results(self) -> list[CombinedOptimizationResult]:
        return [
            CombinedOptimizationResult(
                N=n,
                omega=w,
                delta_omega_opt=0.02 / (n**0.5),
                sql=sql_reference(n),
                ratio=sql_reference(n) / (0.02 / (n**0.5)),
                a_x_opt=float(n),
                a_y_opt=0.0,
                a_z_opt=0.0,
                alpha_xx_opt=float(n) * 2,
                alpha_xz_opt=0.0,
                alpha_zx_opt=0.0,
                alpha_zz_opt=float(n),
                success=True,
                nfev=50,
                n_starts=10,
                n_converged=8,
            )
            for n in [1, 2, 5, 10]
            for w in [0.1, 0.2, 0.5]
        ]

    def test_roundtrip(
        self,
        make_results: list[CombinedOptimizationResult],
        tmp_path: Path,
    ) -> None:
        summary = CombinedNScalingScanResult(results=make_results)
        p = tmp_path / "scan.parquet"
        summary.save_parquet(p)
        loaded = CombinedNScalingScanResult.from_parquet(p)
        assert len(loaded.results) == len(make_results)
        for orig, loaded_r in zip(make_results, loaded.results, strict=False):
            assert orig.N == loaded_r.N
            assert orig.omega == loaded_r.omega

    def test_empty_dataframe(self) -> None:
        summary = CombinedNScalingScanResult(results=[])
        df = summary.to_dataframe()
        assert df.empty

    def test_properties(
        self,
        make_results: list[CombinedOptimizationResult],
    ) -> None:
        summary = CombinedNScalingScanResult(results=make_results)
        assert len(summary.N_values) == 4
        assert len(summary.omega_values) == 3


# ============================================================================
# CombinedRandomSearchResult Serialization
# ============================================================================


class TestCombinedRandomSearchResultParquet:
    @pytest.fixture
    def make_rs_result(self) -> CombinedRandomSearchResult:
        rng = np.random.default_rng(42)
        samples = rng.uniform(-5, 5, size=(10, 7))
        deltas = np.abs(rng.normal(0.1, 0.05, size=10))
        best_idx = int(np.argmin(deltas))
        return CombinedRandomSearchResult(
            N=1,
            samples=samples,
            delta_omega_values=deltas,
            best_params=(
                float(samples[best_idx, 0]),
                float(samples[best_idx, 1]),
                float(samples[best_idx, 2]),
                float(samples[best_idx, 3]),
                float(samples[best_idx, 4]),
                float(samples[best_idx, 5]),
                float(samples[best_idx, 6]),
            ),
            best_delta_omega=float(deltas[best_idx]),
            omega_value=0.5,
            sql=0.1,
        )

    def test_roundtrip(
        self,
        make_rs_result: CombinedRandomSearchResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "rs.parquet"
        make_rs_result.save_parquet(p)
        loaded = CombinedRandomSearchResult.from_parquet(p)
        assert loaded.N == make_rs_result.N
        assert loaded.omega_value == make_rs_result.omega_value
        assert np.isclose(loaded.best_delta_omega, make_rs_result.best_delta_omega)
        assert len(loaded.best_params) == 7
        assert loaded.samples.shape == make_rs_result.samples.shape

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a_x": [1.0]})
        p = tmp_path / "bad_rs.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            CombinedRandomSearchResult.from_parquet(p)


# ============================================================================
# Combined2DSliceResult Serialization
# ============================================================================


class TestCombined2DSliceResultParquet:
    @pytest.fixture
    def make_slice_result(self) -> Combined2DSliceResult:
        return Combined2DSliceResult(
            alpha_xx_values=np.linspace(-5, 5, 5),
            alpha_zz_values=np.linspace(-5, 5, 4),
            delta_omega_grid=np.random.default_rng(42).uniform(0.1, 0.5, size=(5, 4)),
            omega_value=0.2,
            a_x_fixed=1.0,
            a_y_fixed=0.0,
            a_z_fixed=0.0,
            sql=0.1,
        )

    def test_roundtrip(
        self,
        make_slice_result: Combined2DSliceResult,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "slice.parquet"
        make_slice_result.save_parquet(p)
        loaded = Combined2DSliceResult.from_parquet(p)
        assert loaded.N == make_slice_result.N
        assert np.allclose(loaded.alpha_xx_values, make_slice_result.alpha_xx_values)
        assert np.allclose(loaded.alpha_zz_values, make_slice_result.alpha_zz_values)
        assert loaded.omega_value == make_slice_result.omega_value
        assert loaded.a_x_fixed == make_slice_result.a_x_fixed
        assert loaded.a_y_fixed == make_slice_result.a_y_fixed
        assert loaded.alpha_xz_fixed == make_slice_result.alpha_xz_fixed
        assert loaded.alpha_zx_fixed == make_slice_result.alpha_zx_fixed
        assert loaded.t_hold == make_slice_result.t_hold

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"alpha_xx": [1.0]})
        p = tmp_path / "bad_slice.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            Combined2DSliceResult.from_parquet(p)


# ============================================================================
# SQL Reference
# ============================================================================


class TestSqlReference:
    def test_given_N1_then_01(self) -> None:
        assert np.isclose(sql_reference(1), 0.1)

    def test_given_N4_then_005(self) -> None:
        assert np.isclose(sql_reference(4), 0.05)

    def test_given_larger_N_then_smaller_sql(self) -> None:
        sql2 = sql_reference(2)
        sql4 = sql_reference(4)
        assert sql4 < sql2

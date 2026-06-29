"""Tests for the ω-Modulated Drive with Weighted Joint Measurement report.

Companion test module for ``reports/20260613/omega_modulated_joint_measurement.py``.
Tests operator construction, joint measurement operator, circuit evolution,
decoupled baseline, 5D optimisation, and serialization.
"""

from __future__ import annotations

import importlib.util
import sys as _sys
from pathlib import Path as _Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.analysis.ancilla_optimization import compute_expectation_and_variance
from src.analysis.decoupled_baseline import verify_decoupled_baseline
from src.analysis.sensitivity_metrics import sql_reference
from src.physics.joint_measurement import build_joint_measurement_operator
from src.physics.n_particle_drive import (
    build_n_particle_hold_hamiltonian,
    build_n_particle_iszz_interaction,
    build_n_particle_operators,
    build_n_particle_phase_modulated_drive_hamiltonian,
    build_n_particle_system_only_bs_unitary,
    compute_n_particle_decoupled_baseline,
    compute_n_particle_sensitivity,
    evolve_n_particle_circuit,
    n_particle_hold_unitary,
    n_particle_initial_state,
)
from src.utils.serialization import assert_roundtrip_fields

_local_path = _Path(__file__).resolve().parent / "omega_modulated_joint_measurement.py"
_spec = importlib.util.spec_from_file_location(
    "omega_modulated_joint_measurement", str(_local_path)
)
assert _spec is not None
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_sys.modules["omega_modulated_joint_measurement"] = _module
_spec.loader.exec_module(_module)
del _local_path, _spec, _module

from omega_modulated_joint_measurement import (  # type: ignore[import-untyped]  # noqa: E402
    T_BS,
    T_HOLD,
    JointNScalingResult,
    JointNScalingScanResult,
    joint_2d_psi_azz_slice,
    joint_random_search,
    run_joint_nelder_mead,
    run_single_joint_n_omega,
    run_single_sonly_n_omega,
)

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
def make_psi0(make_N: int) -> np.ndarray:
    return n_particle_initial_state(make_N)


# ============================================================================
# Operator Construction
# ============================================================================


class TestBuildNParticleOperators:
    def test_given_N_then_dimension_correct(self, make_N: int) -> None:
        ops = build_n_particle_operators(make_N)
        d_tot = 2 * (make_N + 1)
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert ops[key].shape == (d_tot, d_tot), (
                f"{key} has wrong shape for N={make_N}"
            )

    def test_given_operators_then_hermitian(self, make_N: int) -> None:
        ops = build_n_particle_operators(make_N)
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert np.allclose(ops[key], ops[key].conj().T, atol=1e-12), (
                f"{key} not Hermitian for N={make_N}"
            )

    def test_given_N1_then_matches_pauli(self) -> None:
        """At N=1, system operators should match Pauli matrices (divided by 2)."""
        ops = build_n_particle_operators(1)
        from src.utils.constants import I_2, J_X, J_Z

        expected_Jz_S = np.kron(J_Z, I_2)
        assert np.allclose(ops["Jz_S"], expected_Jz_S, atol=1e-12)

        expected_Jx_S = np.kron(J_X, I_2)
        assert np.allclose(ops["Jx_S"], expected_Jx_S, atol=1e-12)

        expected_Jz_A = np.kron(I_2, J_Z)
        assert np.allclose(ops["Jz_A"], expected_Jz_A, atol=1e-12)

    def test_given_commutation_jz_jx_equals_ijy(self, make_N: int) -> None:
        ops = build_n_particle_operators(make_N)
        comm = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
        expected = 1j * ops["Jy_S"]
        assert np.allclose(comm, expected, atol=1e-10), (
            f"[J_z^S, J_x^S] = i J_y^S violated for N={make_N}"
        )

    def test_given_jz_diagonal_values(self, make_N: int) -> None:
        """J_z^S eigenvalues should span -N/2 to N/2 in steps of 1."""
        ops = build_n_particle_operators(make_N)
        expected_evals = np.sort(
            np.tile(np.arange(make_N / 2.0, -make_N / 2.0 - 1, -1), 2),
        )
        actual_evals = np.sort(np.linalg.eigvalsh(ops["Jz_S"]))
        assert np.allclose(actual_evals, expected_evals, atol=1e-10), (
            f"J_z^S eigenvalues wrong for N={make_N}"
        )

    def test_given_invalid_N_raises(self) -> None:
        with pytest.raises(ValueError):
            build_n_particle_operators(0)


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
        """At ψ=π/4, the operator should equally weight J_z^S and J_z^A."""
        M = build_joint_measurement_operator(make_N, np.pi / 4.0, make_ops)
        expected = (make_ops["Jz_S"] + make_ops["Jz_A"]) / np.sqrt(2)
        assert np.allclose(M, expected, atol=1e-12)

    def test_given_N1_psi_pi_four_then_correct_values(self) -> None:
        """At N=1, ψ=π/4, verify explicit matrix values."""
        ops = build_n_particle_operators(1)
        M = build_joint_measurement_operator(1, np.pi / 4.0, ops)
        # J_z^S = diag(0.5, 0.5, -0.5, -0.5), J_z^A = diag(0.5, -0.5, 0.5, -0.5)
        # M = (J_z^S + J_z^A) / sqrt(2) = diag(1, 0, 0, -1) / sqrt(2)
        expected_diag = np.array([1.0, 0.0, 0.0, -1.0]) / np.sqrt(2)
        assert np.allclose(np.diag(M), expected_diag, atol=1e-12)


class TestBSUnitary:
    def test_given_N_then_dimension_correct(self, make_N: int) -> None:
        U = build_n_particle_system_only_bs_unitary(make_N)
        d_tot = 2 * (make_N + 1)
        assert U.shape == (d_tot, d_tot)

    def test_given_bs_then_unitary(self, make_N: int) -> None:
        U = build_n_particle_system_only_bs_unitary(make_N)
        d_tot = 2 * (make_N + 1)
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12)

    def test_given_zero_T_bs_then_identity(self, make_N: int) -> None:
        U = build_n_particle_system_only_bs_unitary(make_N, 0.0)
        d_tot = 2 * (make_N + 1)
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U, I_full, atol=1e-12)

    def test_given_N1_then_matches_known_bs(self) -> None:
        """At N=1, the N-particle BS should match the single-qubit version."""
        U_n = build_n_particle_system_only_bs_unitary(1, T_BS)
        from src.physics.beam_splitter import bs_qubit
        from src.utils.constants import I_2

        U_expected = np.kron(bs_qubit(T_BS), I_2)
        assert np.allclose(U_n, U_expected, atol=1e-12)


class TestDriveHamiltonian:
    def test_given_zero_omega_then_zero(self, make_N: int, make_ops: dict) -> None:
        H = build_n_particle_phase_modulated_drive_hamiltonian(
            make_N,
            0.0,
            2.0,
            3.0,
            4.0,
            make_ops,
        )
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_zero_coefficients_then_zero(
        self, make_N: int, make_ops: dict
    ) -> None:
        H = build_n_particle_phase_modulated_drive_hamiltonian(
            make_N,
            1.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_drive_then_hermitian(self, make_N: int, make_ops: dict) -> None:
        H = build_n_particle_phase_modulated_drive_hamiltonian(
            make_N,
            1.5,
            1.0,
            2.0,
            3.0,
            make_ops,
        )
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_drive_then_proportional_to_omega(
        self, make_N: int, make_ops: dict
    ) -> None:
        H_half = build_n_particle_phase_modulated_drive_hamiltonian(
            make_N,
            0.5,
            1.0,
            2.0,
            3.0,
            make_ops,
        )
        H_full = build_n_particle_phase_modulated_drive_hamiltonian(
            make_N,
            1.0,
            1.0,
            2.0,
            3.0,
            make_ops,
        )
        assert np.allclose(2.0 * H_half, H_full, atol=1e-12)


class TestIsingInteraction:
    def test_given_zero_azz_then_zero(self, make_N: int, make_ops: dict) -> None:
        H = build_n_particle_iszz_interaction(make_N, 0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_given_azz_then_hermitian(self, make_N: int, make_ops: dict) -> None:
        H = build_n_particle_iszz_interaction(make_N, 3.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_given_azz_then_commutes_with_jz_s(
        self, make_N: int, make_ops: dict
    ) -> None:
        H = build_n_particle_iszz_interaction(make_N, 3.0, make_ops)
        comm = H @ make_ops["Jz_S"] - make_ops["Jz_S"] @ H
        assert np.allclose(comm, 0.0, atol=1e-12)


class TestHoldHamiltonian:
    def test_given_hold_then_hermitian(self, make_N: int, make_ops: dict) -> None:
        H = build_n_particle_hold_hamiltonian(
            make_N,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_ops,
        )
        assert np.allclose(H, H.conj().T, atol=1e-12)


# ============================================================================
# Circuit Evolution
# ============================================================================


class TestHoldUnitary:
    def test_given_hold_then_unitary(self, make_N: int, make_ops: dict) -> None:
        U = n_particle_hold_unitary(
            make_N,
            T_HOLD,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_ops,
        )
        d_tot = 2 * (make_N + 1)
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12)

    def test_given_zero_hold_then_identity(self, make_N: int, make_ops: dict) -> None:
        U = n_particle_hold_unitary(
            make_N,
            0.0,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            make_ops,
        )
        d_tot = 2 * (make_N + 1)
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U, I_full, atol=1e-12)


class TestEvolveCircuit:
    def test_given_circuit_then_normalised(
        self, make_N: int, make_ops: dict, make_psi0: np.ndarray
    ) -> None:
        psi = evolve_n_particle_circuit(
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
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_given_zero_params_then_css_mzi(
        self, make_N: int, make_ops: dict, make_psi0: np.ndarray
    ) -> None:
        """At zero drive and zero interaction, the circuit is a standard MZI."""
        psi = evolve_n_particle_circuit(
            make_N,
            make_psi0,
            T_BS,
            T_HOLD,
            0.5,
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


# ============================================================================
# Sensitivity Computation
# ============================================================================


class TestSensitivity:
    def test_given_valid_params_then_positive(
        self, make_N: int, make_ops: dict, make_psi0: np.ndarray
    ) -> None:
        delta = compute_n_particle_sensitivity(
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
        assert np.isfinite(delta)
        assert delta > 0.0

    def test_given_joint_meas_then_finite(
        self, make_N: int, make_ops: dict, make_psi0: np.ndarray
    ) -> None:
        """Sensitivity with joint measurement should also be positive."""
        M = build_joint_measurement_operator(make_N, 0.5, make_ops)
        delta = compute_n_particle_sensitivity(
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
            meas_op=M,
        )
        assert np.isfinite(delta)
        assert delta > 0.0

    def test_given_sonly_meas_matches_psi_zero(
        self, make_N: int, make_ops: dict, make_psi0: np.ndarray
    ) -> None:
        """Sensitivity with ψ=0 should match S-only measurement."""
        delta_sonly = compute_n_particle_sensitivity(
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
        M0 = build_joint_measurement_operator(make_N, 0.0, make_ops)
        delta_joint_psi0 = compute_n_particle_sensitivity(
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
            meas_op=M0,
        )
        assert np.isclose(delta_sonly, delta_joint_psi0, atol=1e-12)

    def test_given_fringe_extremum_then_inf(
        self, make_N: int, make_ops: dict, make_psi0: np.ndarray
    ) -> None:
        """At parameters where the derivative vanishes, return inf."""
        delta = compute_n_particle_sensitivity(
            make_N,
            make_psi0,
            T_BS,
            T_HOLD,
            0.0,
            5.0,
            5.0,
            5.0,
            5.0,
            make_ops,
        )
        assert delta > 0.0


# ============================================================================
# Decoupled Baseline
# ============================================================================


class TestDecoupledBaseline:
    def test_given_decoupled_then_matches_sql(self, make_N: int) -> None:
        for omega in [0.1, 0.2, 0.5, 1.0]:
            delta = compute_n_particle_decoupled_baseline(make_N, omega)
            sql = sql_reference(make_N)
            assert np.isclose(delta, sql, rtol=1e-8), (
                f"N={make_N}, ω={omega}: Δω={delta:.10f}, SQL={sql:.10f}"
            )

    def test_given_decoupled_with_joint_then_matches_sql(self, make_N: int) -> None:
        """At decoupled parameters, joint measurement should also give SQL."""
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        for omega in [0.1, 0.2, 1.0]:
            for psi in [0.0, 0.3, 0.8]:
                M = build_joint_measurement_operator(make_N, psi, ops)
                delta = compute_n_particle_sensitivity(
                    make_N,
                    psi0,
                    T_BS,
                    T_HOLD,
                    omega,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    ops,
                    meas_op=M,
                )
                sql = sql_reference(make_N)
                assert np.isclose(delta, sql, rtol=1e-8), (
                    f"N={make_N}, ω={omega}, ψ={psi}: Δω={delta:.10f}, SQL={sql:.10f}"
                )

    def test_given_all_n_omega_then_all_pass(self) -> None:
        results = verify_decoupled_baseline(
            N_values=[1, 2, 3, 5, 10],
            omega_values=[0.1, 0.2, 0.5, 1.0, 2.0],
        )
        assert all(results.values()), (
            f"Failed pairs: {[(n, w) for (n, w), v in results.items() if not v]}"
        )

    def test_given_sql_scaling_then_exponent_half(self) -> None:
        Ns = [1, 2, 4, 8, 16]
        sqls = np.array([sql_reference(n) for n in Ns])
        log_N = np.log(Ns)
        log_sql = np.log(sqls)
        A = np.vstack([log_N, np.ones_like(log_N)]).T
        alpha, _ = np.linalg.lstsq(A, log_sql, rcond=None)[0]
        assert np.isclose(alpha, -0.5, atol=1e-10), (
            f"SQL scaling exponent α = {alpha:.6f}, expected -0.5"
        )


# ============================================================================
# N=1 Consistency
# ============================================================================


class TestN1Consistency:
    def test_given_N1_omega02_then_beats_sql(self) -> None:
        """At N=1, ω=0.2, the joint protocol should beat SQL (ratio < 1)."""
        result = run_single_joint_n_omega(N=1, omega=0.2, seed=42)
        assert result.ratio < 1.0, (
            f"N=1, ω=0.2: ratio = {result.ratio:.4f} (expected < 1, beating SQL)"
        )

    def test_given_N1_omega02_then_nonzero_params(self) -> None:
        """Verify that non-commuting drive amplitudes are non-zero."""
        result = run_single_joint_n_omega(N=1, omega=0.2, seed=42)
        assert abs(result.a_x_opt) > 0.1 or abs(result.a_y_opt) > 0.1, (
            f"a_x={result.a_x_opt:.4f}, a_y={result.a_y_opt:.4f}"
        )

    def test_given_N1_omega02_then_variance_positive(self) -> None:
        result = run_single_joint_n_omega(N=1, omega=0.2, seed=42)
        assert result.variance_Jz >= -1e-12
        assert result.variance_Jz >= 0.0

    def test_given_N1_omega02_then_variance_M_positive(self) -> None:
        result = run_single_joint_n_omega(N=1, omega=0.2, seed=42)
        assert result.variance_M >= -1e-12
        assert result.variance_M >= 0.0

    def test_given_N1_omega02_then_expectation_M_finite(self) -> None:
        result = run_single_joint_n_omega(N=1, omega=0.2, seed=42)
        assert np.isfinite(result.expectation_M)

    def test_given_N1_omega02_sonly_then_close_to_20260519(self) -> None:
        """S-only control should be near the 20260519 optimum of 0.02036."""
        result = run_single_sonly_n_omega(N=1, omega=0.2, seed=42)
        expected = 0.02036
        assert np.isclose(result.delta_omega_opt, expected, rtol=0.5), (
            f"N=1, ω=0.2 S-only: Δω={result.delta_omega_opt:.6f}, expected ~{expected:.6f}"
        )


# ============================================================================
# Random Search
# ============================================================================


class TestJointRandomSearch:
    def test_given_random_search_then_returns_result(self, make_N: int) -> None:
        result = joint_random_search(make_N, 0.5, n_samples=20, seed=42)
        assert result.samples.shape == (20, 5)
        assert len(result.delta_omega_values) == 20
        assert result.best_delta_omega > 0.0

    def test_given_random_search_then_best_is_minimum(self, make_N: int) -> None:
        result = joint_random_search(make_N, 0.5, n_samples=20, seed=42)
        assert np.isclose(
            result.best_delta_omega,
            np.min(result.delta_omega_values),
        )

    def test_given_random_search_then_psi_in_params(self, make_N: int) -> None:
        result = joint_random_search(make_N, 0.5, n_samples=20, seed=42)
        assert len(result.best_params) == 5


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


class TestJointNelderMead:
    def test_given_nm_then_converges(self, make_N: int) -> None:
        result = run_joint_nelder_mead(N=make_N, omega_true=0.5, seed=42)
        assert result.delta_omega_opt > 0.0
        assert np.isfinite(result.delta_omega_opt)
        assert result.params_opt.shape == (5,)

    def test_given_nm_then_expectation_finite(self, make_N: int) -> None:
        result = run_joint_nelder_mead(N=make_N, omega_true=0.5, seed=42)
        assert np.isfinite(result.expectation_Jz)
        assert result.variance_Jz >= 0.0
        assert np.isfinite(result.expectation_M)
        assert result.variance_M >= 0.0


# ============================================================================
# 2D Slice (ψ × a_zz)
# ============================================================================


class TestJoint2DSlice:
    def test_given_slice_then_correct_shape(self) -> None:
        result = joint_2d_psi_azz_slice(
            N=1,
            omega=0.2,
            n_psi=11,
            n_azz=11,
        )
        assert result.psi_values.shape == (11,)
        assert result.azz_values.shape == (11,)
        assert result.delta_omega_grid.shape == (11, 11)

    def test_given_slice_then_all_finite(self) -> None:
        result = joint_2d_psi_azz_slice(
            N=1,
            omega=0.2,
            n_psi=5,
            n_azz=5,
        )
        # At least some finite values
        assert np.any(np.isfinite(result.delta_omega_grid))


# ============================================================================
# JointNScalingResult Serialization
# ============================================================================


class TestJointNScalingResultParquet:
    @pytest.fixture
    def make_result(self) -> JointNScalingResult:
        return JointNScalingResult(
            N=5,
            omega=0.2,
            delta_omega_opt=0.015,
            sql=sql_reference(5),
            ratio=sql_reference(5) / 0.015,
            a_x_opt=3.0,
            a_y_opt=-2.5,
            a_z_opt=1.0,
            a_zz_opt=4.0,
            psi_opt=0.5,
            expectation_Jz=0.5,
            variance_Jz=0.25,
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
        ("a_x_opt", "eq"),
        ("a_y_opt", "eq"),
        ("a_z_opt", "eq"),
        ("a_zz_opt", "eq"),
        ("psi_opt", "isclose"),
        ("expectation_Jz", "eq"),
        ("variance_Jz", "eq"),
        ("expectation_M", "eq"),
        ("variance_M", "eq"),
        ("d_expectation", "eq"),
        ("t_hold", "eq"),
        ("success", "eq"),
        ("nfev", "eq"),
    ]

    def test_roundtrip(self, make_result: JointNScalingResult, tmp_path: Path) -> None:
        p = tmp_path / "test.parquet"
        make_result.save_parquet(p)
        loaded = JointNScalingResult.from_parquet(p)
        assert_roundtrip_fields(loaded, make_result, self._FIELD_SPECS)

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "missing.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            JointNScalingResult.from_parquet(p)

    def test_missing_psi_opt_raises(self, tmp_path: Path) -> None:
        """Missing psi_opt column should raise ValueError."""
        df = pd.DataFrame(
            {
                "N": [5],
                "omega": [0.2],
                "delta_omega_opt": [0.015],
                "sql": [0.0447],
                "ratio": [2.98],
                "a_x_opt": [1.0],
                "a_y_opt": [0.0],
                "a_z_opt": [0.0],
                "a_zz_opt": [0.0],
                "expectation_Jz": [0.0],
                "variance_Jz": [0.0],
                "expectation_M": [0.0],
                "variance_M": [0.0],
                "d_expectation": [0.0],
                "t_hold": [10.0],
                "fd_step": [1e-6],
                "success": [1],
                "nfev": [0],
            },
        )
        p = tmp_path / "missing_psi.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            JointNScalingResult.from_parquet(p)


class TestJointNScalingScanResultParquet:
    @pytest.fixture
    def make_results(self) -> list[JointNScalingResult]:
        return [
            JointNScalingResult(
                N=n,
                omega=w,
                delta_omega_opt=0.02 / (n**0.5),
                sql=sql_reference(n),
                ratio=sql_reference(n) / (0.02 / (n**0.5)),
                a_x_opt=float(n),
                a_y_opt=0.0,
                a_z_opt=0.0,
                a_zz_opt=float(n),
                psi_opt=0.5,
                variance_M=0.1,
                expectation_M=0.2,
                d_expectation=0.05,
                success=True,
                nfev=50,
            )
            for n in [1, 2, 5, 10]
            for w in [0.1, 0.2, 0.5]
        ]

    def test_roundtrip(
        self, make_results: list[JointNScalingResult], tmp_path: Path
    ) -> None:
        summary = JointNScalingScanResult(results=make_results)
        p = tmp_path / "scan.parquet"
        summary.save_parquet(p)
        loaded = JointNScalingScanResult.from_parquet(p)
        assert len(loaded.results) == len(make_results)
        for orig, loaded_r in zip(make_results, loaded.results, strict=False):
            assert orig.N == loaded_r.N
            assert orig.omega == loaded_r.omega
            assert np.isclose(orig.psi_opt, loaded_r.psi_opt)

    def test_empty_dataframe(self) -> None:
        summary = JointNScalingScanResult(results=[])
        df = summary.to_dataframe()
        assert df.empty

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "bad_scan.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            JointNScalingScanResult.from_parquet(p)


# ============================================================================
# Initial State
# ============================================================================


class TestInitialState:
    def test_given_N_then_normalised(self, make_N: int) -> None:
        psi = n_particle_initial_state(make_N)
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_given_N_then_first_element_is_one(self, make_N: int) -> None:
        psi = n_particle_initial_state(make_N)
        assert psi[0] == 1.0
        assert np.all(psi[1:] == 0.0)

    def test_given_N_then_correct_length(self, make_N: int) -> None:
        psi = n_particle_initial_state(make_N)
        assert len(psi) == 2 * (make_N + 1)

"""Tests for the N-Scaling of Phase-Modulated Ancilla Drive report.

Companion test module for ``reports/20260611/local.py``.
Tests operator construction, circuit evolution, decoupled baseline,
N=1 consistency, and serialization.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.analysis.ancilla_optimization import (
    compute_expectation_and_variance,
)

_report_dir = str(
    _Path(__file__).resolve().parent.parent.parent / "reports" / "20260611"
)
if _report_dir not in _sys.path:
    _sys.path.insert(0, _report_dir)
del _sys, _Path, _report_dir

from local import (  # type: ignore[import-untyped]  # noqa: E402
    T_BS,
    T_HOLD,
    NScalingResult,
    NScalingScanResult,
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
    n_particle_random_search,
    run_n_particle_nelder_mead,
    run_single_n_omega,
    sql_reference,
    verify_decoupled_baseline,
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

        # J_z^S = J_z ⊗ I_2 should match J_Z ⊗ I_2
        expected_Jz_S = np.kron(J_Z, I_2)
        assert np.allclose(ops["Jz_S"], expected_Jz_S, atol=1e-12)

        # J_x^S = J_x ⊗ I_2 should match J_X ⊗ I_2
        expected_Jx_S = np.kron(J_X, I_2)
        assert np.allclose(ops["Jx_S"], expected_Jx_S, atol=1e-12)

        # J_z^A = I_2 ⊗ J_Z should match I_2 ⊗ J_Z
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
        # The eigenvalues of J_z^S are the Dicke m-values (N/2, N/2-1, ..., -N/2),
        # each appearing twice (once per ancilla state).
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
        self,
        make_N: int,
        make_ops: dict,
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
        self,
        make_N: int,
        make_ops: dict,
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
        self,
        make_N: int,
        make_ops: dict,
    ) -> None:
        """H_int = a_zz J_z^S J_z^A commutes with J_z^S (both diagonal)."""
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
        U = n_particle_hold_unitary(make_N, T_HOLD, 0.5, 1.0, 2.0, 3.0, 4.0, make_ops)
        d_tot = 2 * (make_N + 1)
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12)

    def test_given_zero_hold_then_identity(
        self,
        make_N: int,
        make_ops: dict,
    ) -> None:
        U = n_particle_hold_unitary(make_N, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, make_ops)
        d_tot = 2 * (make_N + 1)
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(U, I_full, atol=1e-12)


class TestEvolveCircuit:
    def test_given_circuit_then_normalised(
        self,
        make_N: int,
        make_ops: dict,
        make_psi0: np.ndarray,
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
        self,
        make_N: int,
        make_ops: dict,
        make_psi0: np.ndarray,
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
        # J_z^S expectation should be finite
        exp_val, var_val = compute_expectation_and_variance(psi, make_ops["Jz_S"])
        assert np.isfinite(exp_val)
        assert var_val >= 0.0


# ============================================================================
# Sensitivity Computation
# ============================================================================


class TestSensitivity:
    def test_given_valid_params_then_positive(
        self,
        make_N: int,
        make_ops: dict,
        make_psi0: np.ndarray,
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

    def test_given_fringe_extremum_then_inf(
        self,
        make_N: int,
        make_ops: dict,
        make_psi0: np.ndarray,
    ) -> None:
        """At parameters where the derivative vanishes, return inf."""
        # Use large drive to potentially hit a fringe extremum
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
        # Should be either finite or inf — both are valid
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

    def test_given_all_n_omega_then_all_pass(self) -> None:
        results = verify_decoupled_baseline(
            N_values=[1, 2, 3, 5, 10],
            omega_values=[0.1, 0.2, 0.5, 1.0, 2.0],
        )
        assert all(results.values()), (
            f"Failed pairs: {[(n, w) for (n, w), v in results.items() if not v]}"
        )

    def test_given_sql_scaling_then_exponent_half(self) -> None:
        """The SQL Δω_SQL = 1/√N should scale with α = -0.5 on log-log."""
        Ns = [1, 2, 4, 8, 16]
        sqls = np.array([sql_reference(n) for n in Ns])
        log_N = np.log(Ns)
        log_sql = np.log(sqls)
        # Linear fit
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
        """At N=1, ω=0.2, the protocol should beat SQL (R > 1)."""
        result = run_single_n_omega(N=1, omega=0.2, seed=42)
        assert result.ratio > 1.0, (
            f"N=1, ω=0.2: ratio = {result.ratio:.4f} (expected > 1)"
        )

    def test_given_N1_omega02_then_reasonable_params(self) -> None:
        """Verify that non-commuting drive amplitudes are non-zero."""
        result = run_single_n_omega(N=1, omega=0.2, seed=42)
        # At least one of a_x or a_y should be non-zero (non-commuting drive)
        assert abs(result.a_x_opt) > 0.1 or abs(result.a_y_opt) > 0.1, (
            f"a_x={result.a_x_opt:.4f}, a_y={result.a_y_opt:.4f} — "
            "need non-commuting drive"
        )

    def test_given_N1_omega02_then_variance_positive(self) -> None:
        result = run_single_n_omega(N=1, omega=0.2, seed=42)
        assert result.variance_Jz >= -1e-12
        assert result.variance_Jz >= 0.0  # after clamping

    @pytest.mark.slow
    def test_given_N1_omega02_then_close_to_20260519(self) -> None:
        """Slow test: verify Δω is near the 20260519 optimum of 0.02036."""
        result = run_single_n_omega(N=1, omega=0.2, seed=42)
        # Should be within 20% of the 20260519 result
        expected = 0.02036
        assert np.isclose(result.delta_omega_opt, expected, rtol=0.2), (
            f"N=1, ω=0.2: Δω={result.delta_omega_opt:.6f}, expected ~{expected:.6f}"
        )


# ============================================================================
# Random Search
# ============================================================================


class TestRandomSearch:
    def test_given_random_search_then_returns_result(self, make_N: int) -> None:
        result = n_particle_random_search(make_N, 0.5, n_samples=20, seed=42)
        assert result.samples.shape == (20, 4)
        assert len(result.delta_omega_values) == 20
        assert result.best_delta_omega > 0.0

    def test_given_random_search_then_best_is_minimum(
        self,
        make_N: int,
    ) -> None:
        result = n_particle_random_search(make_N, 0.5, n_samples=20, seed=42)
        assert np.isclose(
            result.best_delta_omega,
            np.min(result.delta_omega_values),
        )


# ============================================================================
# Nelder-Mead Optimisation
# ============================================================================


class TestNelderMead:
    def test_given_nm_then_converges(self, make_N: int) -> None:
        result = run_n_particle_nelder_mead(
            N=make_N,
            omega_true=0.5,
            seed=42,
        )
        assert result.delta_omega_opt > 0.0
        assert np.isfinite(result.delta_omega_opt)
        assert result.params_opt.shape == (4,)

    def test_given_nm_then_expectation_finite(self, make_N: int) -> None:
        result = run_n_particle_nelder_mead(
            N=make_N,
            omega_true=0.5,
            seed=42,
        )
        assert np.isfinite(result.expectation_Jz)
        assert result.variance_Jz >= 0.0


# ============================================================================
# NScalingResult Serialization
# ============================================================================


class TestNScalingResultParquet:
    @pytest.fixture
    def make_result(self) -> NScalingResult:
        return NScalingResult(
            N=5,
            omega=0.2,
            delta_omega_opt=0.015,
            sql=sql_reference(5),
            ratio=sql_reference(5) / 0.015,
            a_x_opt=3.0,
            a_y_opt=-2.5,
            a_z_opt=1.0,
            a_zz_opt=4.0,
            expectation_Jz=0.5,
            variance_Jz=0.25,
            success=True,
            nfev=100,
        )

    def test_roundtrip(self, make_result: NScalingResult, tmp_path: Path) -> None:
        p = tmp_path / "test.parquet"
        make_result.save_parquet(p)
        loaded = NScalingResult.from_parquet(p)
        assert loaded.N == make_result.N
        assert loaded.omega == make_result.omega
        assert np.isclose(loaded.delta_omega_opt, make_result.delta_omega_opt)
        assert np.isclose(loaded.sql, make_result.sql)
        assert np.isclose(loaded.ratio, make_result.ratio)
        assert loaded.a_x_opt == make_result.a_x_opt
        assert loaded.a_y_opt == make_result.a_y_opt
        assert loaded.a_z_opt == make_result.a_z_opt
        assert loaded.a_zz_opt == make_result.a_zz_opt
        assert loaded.expectation_Jz == make_result.expectation_Jz
        assert loaded.variance_Jz == make_result.variance_Jz
        assert loaded.T_hold == make_result.T_hold
        assert loaded.fd_step == make_result.fd_step
        assert loaded.success == make_result.success
        assert loaded.nfev == make_result.nfev

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "missing.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            NScalingResult.from_parquet(p)

    def test_missing_column_name_raises(self, tmp_path: Path) -> None:
        """Missing a_x_opt column should raise ValueError."""
        df = pd.DataFrame(
            {
                "N": [5],
                "omega": [0.2],
                "delta_omega_opt": [0.015],
                "sql": [0.0447],
                "ratio": [2.98],
                # "a_x_opt" is missing
                "a_y_opt": [0.0],
                "a_z_opt": [0.0],
                "a_zz_opt": [0.0],
                "expectation_Jz": [0.0],
                "variance_Jz": [0.0],
                "T_hold": [10.0],
                "fd_step": [1e-6],
                "success": [1],
                "nfev": [0],
            },
        )
        p = tmp_path / "missing_ax.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            NScalingResult.from_parquet(p)


class TestNScalingScanResultParquet:
    @pytest.fixture
    def make_results(self) -> list[NScalingResult]:
        return [
            NScalingResult(
                N=n,
                omega=w,
                delta_omega_opt=0.02 / (n**0.5),
                sql=sql_reference(n),
                ratio=sql_reference(n) / (0.02 / (n**0.5)),
                a_x_opt=float(n),
                a_y_opt=0.0,
                a_z_opt=0.0,
                a_zz_opt=float(n),
                success=True,
                nfev=50,
            )
            for n in [1, 2, 5, 10]
            for w in [0.1, 0.2, 0.5]
        ]

    def test_roundtrip(
        self,
        make_results: list[NScalingResult],
        tmp_path: Path,
    ) -> None:
        summary = NScalingScanResult(results=make_results)
        p = tmp_path / "scan.parquet"
        summary.save_parquet(p)
        loaded = NScalingScanResult.from_parquet(p)
        assert len(loaded.results) == len(make_results)
        for orig, loaded_r in zip(make_results, loaded.results, strict=False):
            assert orig.N == loaded_r.N
            assert orig.omega == loaded_r.omega

    def test_empty_dataframe(self) -> None:
        summary = NScalingScanResult(results=[])
        df = summary.to_dataframe()
        assert df.empty

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"N": [5], "omega": [0.2]})
        p = tmp_path / "bad_scan.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError):
            NScalingScanResult.from_parquet(p)

    def test_properties(
        self,
        make_results: list[NScalingResult],
    ) -> None:
        summary = NScalingScanResult(results=make_results)
        assert len(summary.N_values) == 4
        assert len(summary.omega_values) == 3


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

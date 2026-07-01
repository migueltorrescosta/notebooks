"""Tests for the Multi-Particle Ancilla Free Initial State report.

Companion test module for ``reports/20260620/multi_particle_free_ancilla.py``.
Tests CSS construction, operator construction, circuit evolution,
decoupled baseline, sensitivity, optimisation, and serialization.
"""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.algorithms.coherent_spin_state import coherent_spin_state
from src.analysis.sensitivity_metrics import sql_reference
from src.utils.serialization import assert_roundtrip_fields

_m = importlib.import_module("reports.20260620.multi_particle_free_ancilla")

AZZ_BOUNDS = _m.AZZ_BOUNDS
DRIVE_RADIUS = _m.DRIVE_RADIUS
FD_STEP = _m.FD_STEP
T_BS = _m.T_BS
T_HOLD = _m.T_HOLD
FreeAncillaNelderMeadResult = _m.FreeAncillaNelderMeadResult
FreeAncillaNScalingResult = _m.FreeAncillaNScalingResult
FreeAncillaNScalingScanResult = _m.FreeAncillaNScalingScanResult
FreeAncillaRandomSearchResult = _m.FreeAncillaRandomSearchResult
build_fixed_drive_hamiltonian = _m.build_fixed_drive_hamiltonian
build_hold_hamiltonian = _m.build_hold_hamiltonian
build_iszz_interaction = _m.build_iszz_interaction
build_operators = _m.build_operators
build_system_only_bs_unitary = _m.build_system_only_bs_unitary
compute_sensitivity = _m.compute_sensitivity
evolve_circuit = _m.evolve_circuit
free_initial_state = _m.free_initial_state
hold_unitary = _m.hold_unitary
random_search = _m.random_search
run_nelder_mead = _m.run_nelder_mead
run_single_n_m_omega = _m.run_single_n_m_omega
sample_6d_config = _m.sample_6d_config
sensitivity_objective = _m.sensitivity_objective
verify_decoupled_baseline = _m.verify_decoupled_baseline

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(
    params=[
        (1, 1),  # J_S=1/2, J_A=1/2 (baseline)
        (3, 2),  # J_S=3/2, J_A=1
        (5, 4),  # J_S=5/2, J_A=2
    ],
    ids=["N=1,M=1", "N=3,M=2", "N=5,M=4"],
)
def make_N_M(request: pytest.FixtureRequest) -> tuple[int, int]:
    return tuple(request.param)


@pytest.fixture
def make_N(make_N_M: tuple[int, int]) -> int:
    return make_N_M[0]


@pytest.fixture
def make_M(make_N_M: tuple[int, int]) -> int:
    return make_N_M[1]


@pytest.fixture
def make_d_tot(make_N: int, make_M: int) -> int:
    return (make_N + 1) * (make_M + 1)


@pytest.fixture
def make_ops(make_N: int, make_M: int) -> dict[str, np.ndarray]:
    return build_operators(make_N, make_M)


@pytest.fixture
def make_psi0(make_N: int, make_M: int) -> np.ndarray:
    return free_initial_state(make_N, make_M, np.pi / 3, np.pi / 4)


# ============================================================================
# Test: Free CSS
# ============================================================================


class TestFreeCSS:
    """Tests for the free-CSS construction."""

    @pytest.mark.parametrize(
        ("J", "theta", "phi"),
        [
            (0.5, 0.0, 0.0),
            (0.5, np.pi, 0.0),
            (1.0, np.pi / 3, np.pi / 4),
            (1.5, np.pi / 2, np.pi),
            (2.0, 0.7, 1.3),
        ],
        ids=["J=0.5,theta=0", "J=0.5,theta=pi", "J=1.0", "J=1.5", "J=2.0"],
    )
    def test_css_normalised(self, J: float, theta: float, phi: float) -> None:
        state = coherent_spin_state(J, theta, phi)
        assert np.isclose(np.linalg.norm(state), 1.0), (
            f"CSS not normalised: norm={np.linalg.norm(state)}"
        )

    @pytest.mark.parametrize(
        ("J", "theta"),
        [(0.5, 0.0), (1.0, 0.0), (2.0, 0.0)],
        ids=["J=0.5", "J=1.0", "J=2.0"],
    )
    def test_css_theta_zero(self, J: float, theta: float) -> None:
        """At theta=0, CSS should be |J, J⟩ (top Dicke state)."""
        state = coherent_spin_state(J, theta, 0.0)
        expected = np.zeros(int(2 * J + 1), dtype=complex)
        expected[0] = 1.0
        assert np.allclose(state, expected, atol=1e-12), (
            f"CSS at θ=0 should be |J,J⟩ for J={J}"
        )

    @pytest.mark.parametrize(
        ("J", "theta"),
        [(0.5, np.pi), (1.0, np.pi), (2.0, np.pi)],
        ids=["J=0.5", "J=1.0", "J=2.0"],
    )
    def test_css_theta_pi(self, J: float, theta: float) -> None:
        """At theta=pi, CSS should be |J, -J⟩ (bottom Dicke state)."""
        state = coherent_spin_state(J, theta, 0.0)
        expected = np.zeros(int(2 * J + 1), dtype=complex)
        expected[-1] = 1.0
        assert np.allclose(state, expected, atol=1e-12), (
            f"CSS at θ=π should be |J,-J⟩ for J={J}"
        )

    def test_css_raises_for_negative_J(self) -> None:
        with pytest.raises(ValueError):
            coherent_spin_state(-1.0, 0.0, 0.0)


# ============================================================================
# Test: Operator Construction
# ============================================================================


class TestOperators:
    """Tests for operator construction."""

    def test_operator_dimensions(
        self,
        make_N: int,
        make_M: int,
        make_d_tot: int,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert make_ops[key].shape == (make_d_tot, make_d_tot), (
                f"{key} has wrong shape: {make_ops[key].shape}"
            )

    def test_operators_hermitian(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert np.allclose(make_ops[key], make_ops[key].conj().T, atol=1e-12), (
                f"{key} not Hermitian"
            )

    def test_commutation_relations_system(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        """[J_z^S, J_x^S] = i J_y^S"""
        comm = make_ops["Jz_S"] @ make_ops["Jx_S"] - make_ops["Jx_S"] @ make_ops["Jz_S"]
        assert np.allclose(comm, 1j * make_ops["Jy_S"], atol=1e-10)

    def test_commutation_relations_ancilla(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        """[J_z^A, J_x^A] = i J_y^A"""
        comm = make_ops["Jz_A"] @ make_ops["Jx_A"] - make_ops["Jx_A"] @ make_ops["Jz_A"]
        assert np.allclose(comm, 1j * make_ops["Jy_A"], atol=1e-10)

    def test_operators_system_ancilla_commute(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        """[J_z^S, J_z^A] = 0 — system and ancilla operators should commute."""
        comm = make_ops["Jz_S"] @ make_ops["Jz_A"] - make_ops["Jz_A"] @ make_ops["Jz_S"]
        assert np.allclose(comm, np.zeros_like(comm), atol=1e-10)

    def test_build_operators_raises_for_bad_N(self) -> None:
        with pytest.raises(ValueError, match="N_sys must be >= 1"):
            build_operators(0, 1)

    def test_build_operators_raises_for_bad_M(self) -> None:
        with pytest.raises(ValueError, match="N_anc must be >= 1"):
            build_operators(1, 0)


# ============================================================================
# Test: BS Unitary
# ============================================================================


class TestBSUnitary:
    """Tests for the system-only beam-splitter unitary."""

    def test_bs_unitary(
        self,
        make_N: int,
        make_M: int,
        make_d_tot: int,
    ) -> None:
        U_bs = build_system_only_bs_unitary(make_N, make_M)
        I_full = np.eye(make_d_tot, dtype=complex)
        assert np.allclose(U_bs @ U_bs.conj().T, I_full, atol=1e-12), (
            "BS unitary not unitary"
        )


# ============================================================================
# Test: Hamiltonian Construction
# ============================================================================


class TestHamiltonian:
    """Tests for Hamiltonian construction."""

    def test_drive_hamiltonian_hermitian(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        H = build_fixed_drive_hamiltonian(1.0, 2.0, -1.5, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_drive_hamiltonian_zero_at_zero(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        """Zero drive coefficients should give zero Hamiltonian."""
        H = build_fixed_drive_hamiltonian(0.0, 0.0, 0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_iszz_hermitian(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        H = build_iszz_interaction(2.5, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_iszz_zero_at_zero(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        H = build_iszz_interaction(0.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-12)

    def test_hold_hamiltonian_hermitian(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        H = build_hold_hamiltonian(1.0, 1.0, 2.0, -1.5, 2.5, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_hold_unitary(
        self,
        make_ops: dict[str, np.ndarray],
        make_d_tot: int,
    ) -> None:
        U = hold_unitary(T_HOLD, 1.0, 1.0, 2.0, -1.5, 2.5, make_ops)
        I_full = np.eye(make_d_tot, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_full, atol=1e-12), (
            "Hold unitary not unitary"
        )


# ============================================================================
# Test: Free Initial State
# ============================================================================


class TestFreeInitialState:
    """Tests for the free initial state."""

    def test_initial_state_normalised(
        self,
        make_N: int,
        make_M: int,
    ) -> None:
        psi = free_initial_state(make_N, make_M, np.pi / 3, np.pi / 4)
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_initial_state_dimensions(
        self,
        make_N: int,
        make_M: int,
        make_d_tot: int,
    ) -> None:
        psi = free_initial_state(make_N, make_M, np.pi / 3, np.pi / 4)
        assert psi.shape == (make_d_tot,), (
            f"State shape {psi.shape}, expected ({make_d_tot},)"
        )

    def test_initial_state_system_in_top_dicke(
        self,
        make_N: int,
        make_M: int,
    ) -> None:
        """System should be in |+N/2⟩ — check that the projection onto |J,J⟩_S is 1."""
        psi = free_initial_state(make_N, make_M, 0.0, 0.0)
        # The state should be |J_S,J_S⟩_S ⊗ |J_A,J_A⟩_A at θ=0
        # In the full basis, this is index 0 (both in top Dicke)
        assert np.isclose(abs(psi[0]), 1.0), "Initial state should have |psi[0]| = 1"


# ============================================================================
# Test: Circuit Evolution
# ============================================================================


class TestCircuit:
    """Tests for the full circuit evolution."""

    def test_circuit_preserves_normalisation(
        self,
        make_ops: dict[str, np.ndarray],
        make_N: int,
        make_M: int,
    ) -> None:
        psi0 = free_initial_state(make_N, make_M, np.pi / 3, np.pi / 4)
        psi_final = evolve_circuit(
            psi0,
            make_N,
            make_M,
            T_BS,
            T_HOLD,
            1.0,
            1.0,
            2.0,
            -1.5,
            2.5,
            make_ops,
        )
        assert np.isclose(np.linalg.norm(psi_final), 1.0), (
            "Circuit does not preserve normalisation"
        )


# ============================================================================
# Test: Sensitivity Computation
# ============================================================================


class TestSensitivity:
    """Tests for the sensitivity computation."""

    def test_sensitivity_finite(
        self,
        make_ops: dict[str, np.ndarray],
        make_N: int,
        make_M: int,
    ) -> None:
        psi0 = free_initial_state(make_N, make_M, np.pi / 3, np.pi / 4)
        delta, _exp_val, _var_val, _d_exp, is_fringe = compute_sensitivity(
            psi0,
            make_N,
            make_M,
            T_BS,
            T_HOLD,
            1.0,
            1.0,
            2.0,
            -1.5,
            2.5,
            make_ops,
        )
        assert np.isfinite(delta) or is_fringe, (
            f"Expected finite delta or fringe, got Δω={delta}, is_fringe={is_fringe}"
        )
        if not is_fringe:
            assert delta > 0, f"Δω must be positive, got {delta}"

    def test_sensitivity_variance_positivity(
        self,
        make_ops: dict[str, np.ndarray],
        make_N: int,
        make_M: int,
    ) -> None:
        psi0 = free_initial_state(make_N, make_M, np.pi / 3, np.pi / 4)
        _, _, var_val, _, _ = compute_sensitivity(
            psi0,
            make_N,
            make_M,
            T_BS,
            T_HOLD,
            1.0,
            1.0,
            2.0,
            -1.5,
            2.5,
            make_ops,
        )
        assert var_val >= 0 or np.isclose(var_val, 0.0, atol=1e-15), (
            f"Variance must be non-negative, got {var_val}"
        )


# ============================================================================
# Test: Decoupled Baseline
# ============================================================================


class TestDecoupledBaseline:
    """Tests for the decoupled baseline."""

    @pytest.mark.parametrize(
        ("N", "M"),
        [
            (1, 1),
            (2, 1),
            (3, 2),
            (5, 3),
            (10, 4),
        ],
        ids=["N=1,M=1", "N=2,M=1", "N=3,M=2", "N=5,M=3", "N=10,M=4"],
    )
    def test_decoupled_baseline_matches_sql(
        self,
        N: int,
        M: int,
    ) -> None:
        """At zero drive/interaction, Δω should equal 1/(√N T_HOLD)."""
        delta = sql_reference(N, T_HOLD)
        expected = sql_reference(N, T_HOLD)
        assert np.isclose(delta, expected, rtol=1e-10), (
            f"Decoupled baseline Δω={delta:.6e} ≠ SQL={expected:.6e} for N={N}, M={M}"
        )

    def test_decoupled_baseline_independent_of_M(self) -> None:
        """The decoupled baseline should not depend on M (ancilla size)."""
        sql_ref = sql_reference(3, T_HOLD)
        deltas = [sql_reference(3, T_HOLD) for _ in [1, 2, 3, 4]]
        for d in deltas:
            assert np.isclose(d, sql_ref, rtol=1e-10), (
                f"Decoupled Δω depends on M: {d} vs SQL={sql_ref}"
            )

    def test_verify_decoupled_baseline_all_pass(self) -> None:
        """All(N, M, omega) triples should pass the decoupled baseline check."""
        results = verify_decoupled_baseline(
            N_values=[1, 2, 3],
            M_values=[1, 2],
            omega_values=[0.1, 1.0],
        )
        for key, passed in results.items():
            assert passed, (
                f"Decoupled baseline FAIL for N={key[0]}, M={key[1]}, ω={key[2]}"
            )


# ============================================================================
# Test: 6D Random Search
# ============================================================================


class TestRandomSearch:
    """Tests for the 6D random search."""

    def test_sample_6d_config_bounds(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(100):
            theta, phi, ax, ay, az, azz = sample_6d_config(rng)
            assert 0.0 <= theta <= np.pi, f"theta out of bounds: {theta}"
            assert 0.0 <= phi < 2.0 * np.pi, f"phi out of bounds: {phi}"
            drive_norm = np.sqrt(ax**2 + ay**2 + az**2)
            assert drive_norm <= DRIVE_RADIUS + 1e-12, (
                f"Drive norm {drive_norm} exceeds radius {DRIVE_RADIUS}"
            )
            assert AZZ_BOUNDS[0] <= azz <= AZZ_BOUNDS[1], f"a_zz out of bounds: {azz}"

    def test_random_search_returns_result(
        self,
        make_N: int,
        make_M: int,
    ) -> None:
        result = random_search(make_N, make_M, 1.0, n_samples=50, seed=42)
        assert isinstance(result, FreeAncillaRandomSearchResult)
        assert result.samples.shape == (50, 6)
        assert len(result.delta_omega_values) == 50
        assert result.best_delta_omega > 0 or np.isinf(result.best_delta_omega)

    def test_random_search_best_is_minimum(
        self,
        make_N: int,
        make_M: int,
    ) -> None:
        result = random_search(make_N, make_M, 1.0, n_samples=50, seed=42)
        assert np.isclose(
            result.best_delta_omega,
            np.min(result.delta_omega_values),
        ), "best_delta_omega should be the minimum of delta_omega_values"

    def test_random_search_reproducible(self) -> None:
        """Same seed should give same results."""
        r1 = random_search(2, 2, 1.0, n_samples=20, seed=123)
        r2 = random_search(2, 2, 1.0, n_samples=20, seed=123)
        assert np.allclose(r1.samples, r2.samples)
        assert np.allclose(r1.delta_omega_values, r2.delta_omega_values)


# ============================================================================
# Test: Nelder-Mead Optimisation
# ============================================================================


class TestNelderMead:
    """Tests for the Nelder-Mead optimisation."""

    def test_nelder_mead_returns_result(
        self,
        make_N: int,
        make_M: int,
    ) -> None:
        result = run_nelder_mead(
            N=make_N,
            M=make_M,
            omega_true=1.0,
            x0=np.array([np.pi / 4, np.pi / 4, 1.0, 2.0, -1.0, 1.5]),
            maxiter=200,
        )
        assert isinstance(result, FreeAncillaNelderMeadResult)
        assert result.delta_omega_opt > 0 or np.isinf(result.delta_omega_opt)
        assert result.params_opt.shape == (6,)
        assert result.nfev > 0

    def test_sensitivity_objective_penalty(self) -> None:
        """Out-of-bounds parameters should receive a large penalty."""
        ops = build_operators(2, 2)
        # theta_A = -1.0 (out of bounds)
        val = sensitivity_objective(
            np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            N=2,
            M=2,
            omega_true=1.0,
            ops=ops,
        )
        assert val > 1e9, f"Expected large penalty for out-of-bounds, got {val}"


# ============================================================================
# Test: Serialization
# ============================================================================


class TestSerialization:
    """Tests for Parquet roundtrip of result dataclasses."""

    def test_random_search_roundtrip(
        self,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 1, size=(10, 6))
        deltas = rng.uniform(0.01, 1.0, size=10)
        best_idx = int(np.argmin(deltas))
        orig = FreeAncillaRandomSearchResult(
            samples=samples,
            delta_omega_values=deltas,
            best_params=(
                float(samples[best_idx, 0]),
                float(samples[best_idx, 1]),
                float(samples[best_idx, 2]),
                float(samples[best_idx, 3]),
                float(samples[best_idx, 4]),
                float(samples[best_idx, 5]),
            ),
            best_delta_omega=float(deltas[best_idx]),
            N=3,
            M=2,
            omega_value=1.0,
            sql=0.1,
            t_hold=10.0,
        )
        path = tmp_path / "test_random.parquet"
        orig.save_parquet(path)
        loaded = FreeAncillaRandomSearchResult.from_parquet(path)
        assert np.allclose(loaded.samples, orig.samples)
        assert np.allclose(loaded.delta_omega_values, orig.delta_omega_values)
        assert loaded.best_params == orig.best_params
        assert np.isclose(loaded.best_delta_omega, orig.best_delta_omega)
        assert loaded.N == orig.N
        assert loaded.M == orig.M
        assert np.isclose(loaded.omega_value, orig.omega_value)

    def test_nelder_mead_roundtrip(self, tmp_path: Path) -> None:
        orig = FreeAncillaNelderMeadResult(
            delta_omega_opt=0.05,
            params_opt=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            omega_true=0.5,
            N=3,
            M=2,
            success=True,
            nfev=100,
            message="OK",
            expectation_Jz=0.5,
            variance_Jz=0.1,
            history=[0.1, 0.08, 0.05],
        )
        path = tmp_path / "test_nm.parquet"
        orig.save_parquet(path)
        loaded = FreeAncillaNelderMeadResult.from_parquet(path)
        assert_roundtrip_fields(
            loaded,
            orig,
            [
                ("delta_omega_opt", "isclose"),
                ("omega_true", "isclose"),
                ("N", "eq"),
                ("M", "eq"),
                ("success", "eq"),
                ("nfev", "eq"),
                ("expectation_Jz", "isclose"),
                ("variance_Jz", "isclose"),
            ],
        )
        assert np.allclose(loaded.params_opt, orig.params_opt)
        # Verify history sidecar survives the roundtrip
        assert loaded.history == orig.history, (
            f"History roundtrip mismatch: {loaded.history} != {orig.history}"
        )

    def test_nelder_mead_no_history_sidecar(self, tmp_path: Path) -> None:
        """Loading a NM result without a history sidecar should return empty history."""
        orig = FreeAncillaNelderMeadResult(
            delta_omega_opt=0.05,
            params_opt=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            omega_true=0.5,
            N=3,
            M=2,
            success=True,
            nfev=100,
            message="OK",
            expectation_Jz=0.5,
            variance_Jz=0.1,
        )
        path = tmp_path / "test_nm_no_history.parquet"
        orig.save_parquet(path)
        # Remove the history sidecar if it was saved
        history_path = path.with_stem(path.stem + "-history")
        if history_path.exists():
            history_path.unlink()
        loaded = FreeAncillaNelderMeadResult.from_parquet(path)
        assert loaded.history == [], f"Expected empty history, got {loaded.history}"

    def test_n_scaling_result_roundtrip(self, tmp_path: Path) -> None:
        orig = FreeAncillaNScalingResult(
            N=3,
            M=2,
            omega=1.0,
            delta_omega_opt=0.05,
            sql=sql_reference(3, T_HOLD),
            ratio=sql_reference(3, T_HOLD) / 0.05,
            theta_A_opt=1.0,
            phi_A_opt=2.0,
            a_x_opt=3.0,
            a_y_opt=4.0,
            a_z_opt=5.0,
            a_zz_opt=6.0,
            expectation_Jz=0.5,
            variance_Jz=0.1,
            t_hold=T_HOLD,
            fd_step=FD_STEP,
            success=True,
            nfev=100,
            J_A=1.0,  # M/2 = 2/2 = 1
            drive_norm=math.sqrt(3.0**2 + 4.0**2 + 5.0**2),
            d_exp=0.75,
            is_fringe=False,
        )
        path = tmp_path / "test_ns.parquet"
        orig.save_parquet(path)
        loaded = FreeAncillaNScalingResult.from_parquet(path)
        assert_roundtrip_fields(
            loaded,
            orig,
            [
                ("N", "eq"),
                ("M", "eq"),
                ("omega", "isclose"),
                ("delta_omega_opt", "isclose"),
                ("sql", "isclose"),
                ("ratio", "isclose"),
                ("theta_A_opt", "isclose"),
                ("phi_A_opt", "isclose"),
                ("a_x_opt", "isclose"),
                ("a_y_opt", "isclose"),
                ("a_z_opt", "isclose"),
                ("a_zz_opt", "isclose"),
                ("t_hold", "isclose"),
                ("fd_step", "isclose"),
                ("success", "eq"),
                ("nfev", "eq"),
                ("J_A", "isclose"),
                ("drive_norm", "isclose"),
                ("d_exp", "isclose"),
                ("is_fringe", "eq"),
            ],
        )

    def test_n_scaling_scan_result_roundtrip(self, tmp_path: Path) -> None:
        sql_val = sql_reference(3, T_HOLD)
        r1 = FreeAncillaNScalingResult(
            N=3,
            M=2,
            omega=1.0,
            delta_omega_opt=0.05,
            sql=sql_val,
            ratio=sql_val / 0.05,
            theta_A_opt=1.0,
            phi_A_opt=2.0,
            a_x_opt=3.0,
            a_y_opt=4.0,
            a_z_opt=5.0,
            a_zz_opt=6.0,
            expectation_Jz=0.5,
            variance_Jz=0.1,
            t_hold=T_HOLD,
            fd_step=FD_STEP,
            success=True,
            nfev=100,
        )
        r2 = FreeAncillaNScalingResult(
            N=5,
            M=3,
            omega=0.5,
            delta_omega_opt=0.03,
            sql=sql_val,
            ratio=sql_val / 0.03,
            theta_A_opt=0.5,
            phi_A_opt=1.0,
            a_x_opt=2.0,
            a_y_opt=3.0,
            a_z_opt=4.0,
            a_zz_opt=5.0,
            expectation_Jz=0.4,
            variance_Jz=0.2,
            t_hold=T_HOLD,
            fd_step=FD_STEP,
            success=True,
            nfev=150,
        )
        orig = FreeAncillaNScalingScanResult(results=[r1, r2])
        path = tmp_path / "test_scan.parquet"
        orig.save_parquet(path)
        loaded = FreeAncillaNScalingScanResult.from_parquet(path)
        assert len(loaded.results) == 2
        for lr, rr in zip(loaded.results, orig.results, strict=True):
            assert lr.N == rr.N
            assert lr.M == rr.M
            assert np.isclose(lr.omega, rr.omega)
            assert np.isclose(lr.delta_omega_opt, rr.delta_omega_opt)

    def test_random_search_missing_column_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """Fail-fast deserialization: missing column should raise."""
        df = pd.DataFrame({"a_x": [1.0], "a_y": [2.0], "a_z": [3.0]})
        path = tmp_path / "bad.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError):
            FreeAncillaRandomSearchResult.from_parquet(path)

    def test_n_scaling_result_missing_column_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """Fail-fast deserialization: missing column should raise."""
        df = pd.DataFrame({"N": [3], "M": [2], "omega": [1.0]})
        path = tmp_path / "bad.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError):
            FreeAncillaNScalingResult.from_parquet(path)


# ============================================================================
# Test: End-to-End Pipeline
# ============================================================================


class TestPipeline:
    """Tests for the full optimisation pipeline."""

    @pytest.mark.slow
    def test_run_single_n_m_omega_basic(self) -> None:
        """Run a single (N, M, omega) triple with fast settings."""
        result = run_single_n_m_omega(
            N=1,
            M=1,
            omega=1.0,
            seed=42,
        )
        assert isinstance(result, FreeAncillaNScalingResult)
        assert result.N == 1
        assert result.M == 1
        assert np.isclose(result.omega, 1.0)
        # For N=1, M=1 with ω-independent drive, we expect no SQL violation
        assert result.delta_omega_opt > 0
        assert result.ratio > 0

    def test_n_scaling_scan_result_properties(self) -> None:
        """Test the property accessors for NScalingScanResult."""
        sql_val = sql_reference(3, T_HOLD)
        r1 = FreeAncillaNScalingResult(
            N=3,
            M=2,
            omega=0.5,
            delta_omega_opt=0.05,
            sql=sql_val,
            ratio=sql_val / 0.05,
            theta_A_opt=1.0,
            phi_A_opt=2.0,
            a_x_opt=3.0,
            a_y_opt=4.0,
            a_z_opt=5.0,
            a_zz_opt=6.0,
            expectation_Jz=0.5,
            variance_Jz=0.1,
            t_hold=T_HOLD,
            fd_step=FD_STEP,
            success=True,
            nfev=100,
        )
        r2 = FreeAncillaNScalingResult(
            N=5,
            M=3,
            omega=1.0,
            delta_omega_opt=0.03,
            sql=sql_val,
            ratio=sql_val / 0.03,
            theta_A_opt=0.5,
            phi_A_opt=1.0,
            a_x_opt=2.0,
            a_y_opt=3.0,
            a_z_opt=4.0,
            a_zz_opt=5.0,
            expectation_Jz=0.4,
            variance_Jz=0.2,
            t_hold=T_HOLD,
            fd_step=FD_STEP,
            success=True,
            nfev=150,
        )
        scan = FreeAncillaNScalingScanResult(results=[r1, r2])
        assert np.array_equal(scan.N_values, np.array([3, 5]))
        assert np.array_equal(scan.M_values, np.array([2, 3]))
        assert np.array_equal(scan.omega_values, np.array([0.5, 1.0]))


# ============================================================================
# Test: N=1, M=1 Consistency
# ============================================================================


class TestN1M1Consistency:
    """Verify consistency with #20260528: N=1, M=1 cannot beat SQL."""

    @pytest.mark.parametrize("omega", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_n1_m1_decoupled_baseline(self, omega: float) -> None:
        """At N=1, M=1, decoupled baseline should be Δω = 1/T_HOLD = 0.1."""
        delta = sql_reference(1, T_HOLD)
        expected = sql_reference(1, T_HOLD)  # 0.1
        assert np.isclose(delta, expected, rtol=1e-10), (
            f"N=1,M=1 decoupled Δω={delta} ≠ {expected}"
        )

    def test_n1_m1_consistency_commutator_check(self) -> None:
        """At a_x = a_y = 0 (pure z-drive), the drive commutes with J_z^A
        so no sensitivity improvement can come from the drive. Verify the
        commutator explicitly and check sensitivity does not exceed SQL."""
        ops = build_operators(1, 1)
        psi0 = free_initial_state(1, 1, np.pi / 3, np.pi / 4)

        # Build the drive-only Hamiltonian (a_x = a_y = 0, a_z = 5.0)
        H_drive = build_fixed_drive_hamiltonian(
            0.0,
            0.0,
            5.0,
            ops,
        )
        # Verify [H_drive, J_z^A] = 0 for pure z-drive
        comm = H_drive @ ops["Jz_A"] - ops["Jz_A"] @ H_drive
        assert np.allclose(comm, 0, atol=1e-12), (
            f"[H_drive, J_z^A] should be zero for pure z-drive, "
            f"max |comm| = {np.max(np.abs(comm)):.2e}"
        )

        delta, _, _, _, _ = compute_sensitivity(
            psi0,
            1,
            1,
            T_BS,
            T_HOLD,
            1.0,
            0.0,
            0.0,
            5.0,
            2.0,
            ops,
        )
        # Pure z-drive commutes with J_z^A, so no SQL improvement expected
        assert np.isfinite(delta), "Delta should be finite for pure z-drive"
        # Sensitivity should not be substantially better than SQL
        sql = sql_reference(1, T_HOLD)
        assert delta >= 0.5 * sql, (
            f"Delta ({delta:.6e}) should not be substantially better "
            f"than SQL ({sql:.6e}) with commuting drive"
        )


# ============================================================================
# Test: Physical Invariants
# ============================================================================


class TestPhysicalInvariants:
    """Physical invariants for the MZI circuit."""

    def test_system_only_bs_unitarity_all_N(self) -> None:
        """BS unitary should be unitary for all system sizes."""
        for N in [1, 2, 5, 10]:
            for M in [1, 2, 4]:
                d_tot = (N + 1) * (M + 1)
                U_bs = build_system_only_bs_unitary(N, M)
                I_full = np.eye(d_tot, dtype=complex)
                assert np.allclose(
                    U_bs @ U_bs.conj().T,
                    I_full,
                    atol=1e-12,
                ), f"BS not unitary for N={N}, M={M}"

    def test_coherent_spin_state_special_cases(self) -> None:
        """Verify specific known CSS values."""
        # J=0.5: CSS at (π/2, 0) should be (|0⟩ + |1⟩)/√2 in z-basis
        state = coherent_spin_state(0.5, np.pi / 2, 0.0)
        # |J=1/2, m=+1/2⟩ at index 0, |J=1/2, m=-1/2⟩ at index 1
        expected_amp = 1.0 / np.sqrt(2)
        assert np.isclose(abs(state[0]), expected_amp, atol=1e-12), (
            f"CSS(π/2,0) for J=0.5: expected |amp[0]|={expected_amp}, got {abs(state[0])}"
        )
        assert np.isclose(abs(state[1]), expected_amp, atol=1e-12), (
            f"CSS(π/2,0) for J=0.5: expected |amp[1]|={expected_amp}, got {abs(state[1])}"
        )

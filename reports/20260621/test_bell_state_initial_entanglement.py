"""Tests for the Bell-State Initial S--A Entanglement report.

Companion test module for ``reports/20260621/bell_state_initial_entanglement.py``.
Tests Bell state preparation, operator construction, circuit evolution,
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

from src.analysis.ancilla_drive_metrology import (
    compute_drive_sensitivity,
    compute_drive_sensitivity_with_details,
)
from src.analysis.ancilla_optimization import (
    build_hold_hamiltonian,
    build_two_qubit_operators,
    hold_unitary_two_qubit,
)
from src.analysis.sensitivity_metrics import sql_reference
from src.utils.serialization import assert_roundtrip_fields

_m = importlib.import_module("reports.20260621.bell_state_initial_entanglement")

ACTIVE_SCENARIOS = _m.ACTIVE_SCENARIOS
AZZ_BOUNDS = _m.AZZ_BOUNDS
DRIVE_RADIUS = _m.DRIVE_RADIUS
T_BS = _m.T_BS
T_HOLD = _m.T_HOLD
BellNelderMeadResult = _m.BellNelderMeadResult
BellOptimisationResult = _m.BellOptimisationResult
BellRandomSearchResult = _m.BellRandomSearchResult
BellScanResult = _m.BellScanResult
Scenario = _m.Scenario
bell_state_phi_minus = _m.bell_state_phi_minus
bell_state_phi_plus = _m.bell_state_phi_plus
config_to_params = _m.config_to_params
get_initial_state = _m.get_initial_state
product_state_00 = _m.product_state_00
run_nelder_mead = _m.run_nelder_mead
run_random_search = _m.run_random_search
run_single_scenario_omega = _m.run_single_scenario_omega
sample_drive_vector = _m.sample_drive_vector
sample_scenario_config = _m.sample_scenario_config
sensitivity_objective = _m.sensitivity_objective
verify_decoupled_baseline = _m.verify_decoupled_baseline

# ============================================================================
# Test Helpers (replacements for functions removed from experiment module)
# ============================================================================


def _compute_sensitivity(
    psi: np.ndarray,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
    fd_step: float = 1e-6,
) -> tuple[float, float, float, float, bool]:
    """Compute sensitivity with details, wrapping compute_drive_sensitivity_with_details.

    Inserts T_BS and T_HOLD into the call signature.
    """
    return compute_drive_sensitivity_with_details(
        psi,
        T_BS,
        T_HOLD,
        omega,
        a_x,
        a_y,
        a_z,
        a_zz,
        ops,
        fd_step=fd_step,
    )


def _hold_unitary(
    t_hold: float,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Hold unitary, wrapping hold_unitary_two_qubit with individual drive params.

    Converts individual (a_x, a_y, a_z, a_zz) into the alpha tuple.
    """
    return hold_unitary_two_qubit(
        t_hold,
        omega,
        (a_x, a_y, a_z, a_zz),
        ops,
    )


def _evolve_circuit(
    psi: np.ndarray,
    omega: float,
    a_x: float,
    a_y: float,
    a_z: float,
    a_zz: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray:
    """Full circuit: BS -> hold -> BS, returning final state.

    Uses build_hold_hamiltonian for the hold step and scipy expm
    for the beam-splitter rotation on the system.
    """
    from scipy.linalg import expm

    U_bs = expm(-1j * T_BS * ops["Jx_S"])
    H = build_hold_hamiltonian(omega, (a_x, a_y, a_z, a_zz), ops)
    U_hold = expm(-1j * T_HOLD * H)
    return U_bs @ U_hold @ U_bs @ psi


def _compute_decoupled_baseline(
    scenario: Scenario,
    omega_true: float = 1.0,
) -> float:
    """Decoupled baseline sensitivity for a given scenario at zero drive.

    For product state (Scenario D), this equals 1/T_HOLD = 0.1.
    For Bell states, this is infinite (fringe extremum).
    """
    psi0 = get_initial_state(scenario)
    ops = build_two_qubit_operators()
    return compute_drive_sensitivity(
        psi0,
        T_BS,
        T_HOLD,
        omega_true,
        0.0,
        0.0,
        0.0,
        0.0,
        ops,
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def make_ops() -> dict[str, np.ndarray]:
    from src.analysis.ancilla_optimization import build_two_qubit_operators

    return build_two_qubit_operators()


@pytest.fixture
def make_bell() -> np.ndarray:
    return bell_state_phi_plus()


@pytest.fixture
def make_product() -> np.ndarray:
    return product_state_00()


# ============================================================================
# Test: Bell State
# ============================================================================


class TestBellState:
    """Tests for Bell state construction."""

    def test_bell_phi_plus_normalised(self) -> None:
        state = bell_state_phi_plus()
        assert np.isclose(np.linalg.norm(state), 1.0), (
            f"Bell state not normalised: norm={np.linalg.norm(state)}"
        )

    def test_bell_phi_plus_structure(self) -> None:
        """|Φ⁺⟩ should have equal amplitudes on |00⟩ and |11⟩."""
        state = bell_state_phi_plus()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0 / np.sqrt(2.0)
        expected[3] = 1.0 / np.sqrt(2.0)
        assert np.allclose(state, expected, atol=1e-15), (
            f"Bell state structure wrong: {state}"
        )

    def test_bell_phi_minus_normalised(self) -> None:
        state = bell_state_phi_minus()
        assert np.isclose(np.linalg.norm(state), 1.0)

    def test_bell_phi_minus_structure(self) -> None:
        """|Φ⁻⟩ should have equal amplitudes with opposite sign on |00⟩ and |11⟩."""
        state = bell_state_phi_minus()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0 / np.sqrt(2.0)
        expected[3] = -1.0 / np.sqrt(2.0)
        assert np.allclose(state, expected, atol=1e-15)

    def test_bell_entanglement(self) -> None:
        """Bell state should have |⟨Φ⁺|Φ⁺⟩| = 1 (pure) and partial trace 0.5 I."""
        state = bell_state_phi_plus()
        rho = np.outer(state, state.conj())
        # Partial trace over ancilla: rho_S = Tr_A(rho)
        rho_s = np.zeros((2, 2), dtype=complex)
        for a_idx in range(2):
            for s1 in range(2):
                for s2 in range(2):
                    i = s1 * 2 + a_idx
                    j = s2 * 2 + a_idx
                    rho_s[s1, s2] += rho[i, j]
        # Reduced system state should be maximally mixed
        expected_rho_s = 0.5 * np.eye(2, dtype=complex)
        assert np.allclose(rho_s, expected_rho_s, atol=1e-12), (
            f"Bell state reduced system not maximally mixed:\n{rho_s}"
        )

    def test_jz_statistics_bell(self) -> None:
        """Bell state: ⟨J_z^S⟩ = 0, Var(J_z^S) = 1/4."""
        from src.analysis.ancilla_optimization import (
            build_two_qubit_operators,
            compute_expectation_and_variance,
        )

        ops = build_two_qubit_operators()
        state = bell_state_phi_plus()
        exp_val, var_val = compute_expectation_and_variance(state, ops["Jz_S"])
        assert np.isclose(exp_val, 0.0, atol=1e-15), (
            f"⟨J_z^S⟩ should be 0, got {exp_val}"
        )
        assert np.isclose(var_val, 0.25, atol=1e-15), (
            f"Var(J_z^S) should be 1/4, got {var_val}"
        )

    def test_jz_covariance_bell(self) -> None:
        """Bell state: Cov(J_z^S, J_z^A) = +1/4 (maximal positive correlation)."""
        from src.analysis.ancilla_optimization import (
            build_two_qubit_operators,
            compute_expectation_and_variance,
        )

        ops = build_two_qubit_operators()
        state = bell_state_phi_plus()
        # Compute ⟨J_z^S J_z^A⟩
        jz_s_jz_a = ops["Jz_S"] @ ops["Jz_A"]
        exp_jz_s_jz_a = np.real(state.conj() @ jz_s_jz_a @ state)
        exp_jz_s, _ = compute_expectation_and_variance(state, ops["Jz_S"])
        exp_jz_a, _ = compute_expectation_and_variance(state, ops["Jz_A"])
        cov = exp_jz_s_jz_a - exp_jz_s * exp_jz_a
        assert np.isclose(cov, 0.25, atol=1e-15), (
            f"Cov(J_z^S, J_z^A) should be +1/4, got {cov}"
        )

    def test_product_state_00(self) -> None:
        state = product_state_00()
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        assert np.allclose(state, expected, atol=1e-15)
        assert np.isclose(np.linalg.norm(state), 1.0)


# ============================================================================
# Test: State Selector
# ============================================================================


class TestStateSelector:
    """Tests for get_initial_state."""

    def test_bell_for_scenario_a(self) -> None:
        state = get_initial_state(Scenario.A)
        expected = bell_state_phi_plus()
        assert np.allclose(state, expected, atol=1e-15)

    def test_bell_for_scenario_b(self) -> None:
        state = get_initial_state(Scenario.B)
        expected = bell_state_phi_plus()
        assert np.allclose(state, expected, atol=1e-15)

    def test_bell_for_scenario_c(self) -> None:
        state = get_initial_state(Scenario.C)
        expected = bell_state_phi_plus()
        assert np.allclose(state, expected, atol=1e-15)

    def test_product_for_scenario_d(self) -> None:
        state = get_initial_state(Scenario.D)
        expected = product_state_00()
        assert np.allclose(state, expected, atol=1e-15)


# ============================================================================
# Test: Hamiltonian Construction
# ============================================================================


class TestHamiltonian:
    """Tests for Hamiltonian construction."""

    def test_hold_hamiltonian_hermitian(self, make_ops: dict[str, np.ndarray]) -> None:
        H = build_hold_hamiltonian(1.0, (1.0, 2.0, -1.5, 2.5), make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_hold_hamiltonian_zero_drive(self, make_ops: dict[str, np.ndarray]) -> None:
        """With zero drive and interaction, H = omega * J_z^S."""
        H = build_hold_hamiltonian(1.0, (0.0, 0.0, 0.0, 0.0), make_ops)
        expected = 1.0 * make_ops["Jz_S"]
        assert np.allclose(H, expected, atol=1e-12), (
            "Zero-drive hold Hamiltonian should be omega * J_z^S"
        )

    def test_hold_unitary(self, make_ops: dict[str, np.ndarray]) -> None:
        U = _hold_unitary(T_HOLD, 1.0, 1.0, 2.0, -1.5, 2.5, make_ops)
        assert np.allclose(U @ U.conj().T, np.eye(4, dtype=complex), atol=1e-12), (
            "Hold unitary not unitary"
        )


# ============================================================================
# Test: Circuit Evolution
# ============================================================================


class TestCircuit:
    """Tests for the full circuit evolution."""

    def test_circuit_preserves_normalisation(
        self,
        make_ops: dict[str, np.ndarray],
        make_bell: np.ndarray,
    ) -> None:
        psi_final = _evolve_circuit(make_bell, 1.0, 1.0, 2.0, -1.5, 2.5, make_ops)
        assert np.isclose(np.linalg.norm(psi_final), 1.0), (
            "Circuit does not preserve normalisation"
        )

    def test_circuit_bell_product_same_at_zero(
        self,
        make_ops: dict[str, np.ndarray],
    ) -> None:
        """At zero params, both Bell and product states should evolve the same way
        up to a different initial state (just the BS transformations)."""
        psi_bell = _evolve_circuit(
            bell_state_phi_plus(), 0.0, 0.0, 0.0, 0.0, 0.0, make_ops
        )
        psi_product = _evolve_circuit(
            product_state_00(), 0.0, 0.0, 0.0, 0.0, 0.0, make_ops
        )
        # Both should be normalised
        assert np.isclose(np.linalg.norm(psi_bell), 1.0)
        assert np.isclose(np.linalg.norm(psi_product), 1.0)
        # They differ because Bell state has additional |11⟩ component
        # but both should evolve unitarily (assert normalisation only)


# ============================================================================
# Test: Sensitivity Computation
# ============================================================================


class TestSensitivity:
    """Tests for the sensitivity computation."""

    def test_sensitivity_finite(
        self,
        make_ops: dict[str, np.ndarray],
        make_bell: np.ndarray,
    ) -> None:
        delta, _exp, _var, _dexp, is_fringe = _compute_sensitivity(
            make_bell,
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
        make_bell: np.ndarray,
    ) -> None:
        _, _, var_val, _, _ = _compute_sensitivity(
            make_bell,
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

    SQL_REF = sql_reference(1, T_HOLD)  # 0.1

    @pytest.mark.parametrize("scenario", [Scenario.A, Scenario.B, Scenario.C])
    def test_decoupled_baseline_bell_is_fringe(self, scenario: Scenario) -> None:
        r"""Bell-state decoupled baselines should be fringe (infinite sensitivity).
        The system reduced density matrix is maximally mixed, making
        :math:`\langle J_z^S\rangle` identically zero for all :math:`\omega`."""
        delta = _compute_decoupled_baseline(scenario, omega_true=1.0)
        assert not np.isfinite(delta) or delta > 1e6, (
            f"Bell baseline should be fringe for {scenario}, got Δω={delta:.6e}"
        )

    def test_decoupled_baseline_product_matches_sql(self) -> None:
        """Product state |00⟩ should give exact SQL at zero params."""
        delta = _compute_decoupled_baseline(Scenario.D, omega_true=1.0)
        assert np.isclose(delta, self.SQL_REF, rtol=1e-10), (
            f"Product baseline Δω={delta:.6e} ≠ SQL={self.SQL_REF:.6e}"
        )

    @pytest.mark.parametrize("omega", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_decoupled_baseline_product_all_omega(self, omega: float) -> None:
        """Product state decoupled baseline should match SQL for all ω."""
        delta = _compute_decoupled_baseline(Scenario.D, omega_true=omega)
        assert np.isclose(delta, self.SQL_REF, rtol=1e-10), (
            f"Scenario D, ω={omega}: Δω={delta:.6e} ≠ SQL={self.SQL_REF:.6e}"
        )

    def test_verify_decoupled_baseline_bell_fringe(self) -> None:
        """Bell scenarios should be reported as PASS (= fringe expected)."""
        results = verify_decoupled_baseline(
            scenarios=[Scenario.A, Scenario.B, Scenario.C],
            omega_values=[0.1, 1.0, 5.0],
        )
        for key, passed in results.items():
            assert passed, f"Decoupled baseline FAIL for S={key[0]}, ω={key[1]}"

    def test_verify_decoupled_baseline_product_sql(self) -> None:
        """Product scenario should be reported as PASS (= exact SQL)."""
        results = verify_decoupled_baseline(
            scenarios=[Scenario.D],
            omega_values=[0.1, 1.0, 5.0],
        )
        for key, passed in results.items():
            assert passed, f"Product baseline FAIL for S={key[0]}, ω={key[1]}"


# ============================================================================
# Test: Parameter Sampling
# ============================================================================


class TestSampling:
    """Tests for parameter sampling."""

    def test_sample_drive_vector_bounds(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(100):
            drive = sample_drive_vector(rng)
            assert drive.shape == (3,)
            norm = np.linalg.norm(drive)
            assert norm <= DRIVE_RADIUS + 1e-12, (
                f"Drive norm {norm} exceeds radius {DRIVE_RADIUS}"
            )

    def test_sample_drive_vector_reproducible(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        d1 = sample_drive_vector(rng1)
        d2 = sample_drive_vector(rng2)
        assert np.allclose(d1, d2)

    def test_sample_scenario_a(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            config = sample_scenario_config(Scenario.A, rng)
            assert config.shape == (4,)
            drive_norm = np.linalg.norm(config[:3])
            assert drive_norm <= DRIVE_RADIUS + 1e-12
            assert AZZ_BOUNDS[0] <= config[3] <= AZZ_BOUNDS[1]

    def test_sample_scenario_b(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            config = sample_scenario_config(Scenario.B, rng)
            assert config.shape == (1,)
            assert AZZ_BOUNDS[0] <= config[0] <= AZZ_BOUNDS[1]

    def test_sample_scenario_c(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            config = sample_scenario_config(Scenario.C, rng)
            assert config.shape == (3,)
            drive_norm = np.linalg.norm(config)
            assert drive_norm <= DRIVE_RADIUS + 1e-12

    def test_config_to_params_a(self) -> None:
        config = np.array([1.0, 2.0, 3.0, 4.0])
        a_x, a_y, a_z, a_zz = config_to_params(config, Scenario.A)
        assert (a_x, a_y, a_z, a_zz) == (1.0, 2.0, 3.0, 4.0)

    def test_config_to_params_b(self) -> None:
        config = np.array([4.0])
        a_x, a_y, a_z, a_zz = config_to_params(config, Scenario.B)
        assert (a_x, a_y, a_z, a_zz) == (0.0, 0.0, 0.0, 4.0)

    def test_config_to_params_c(self) -> None:
        config = np.array([1.0, 2.0, 3.0])
        a_x, a_y, a_z, a_zz = config_to_params(config, Scenario.C)
        assert (a_x, a_y, a_z, a_zz) == (1.0, 2.0, 3.0, 0.0)


# ============================================================================
# Test: Random Search
# ============================================================================


class TestRandomSearch:
    """Tests for the random search."""

    @pytest.mark.parametrize("scenario", [Scenario.A, Scenario.B, Scenario.C])
    def test_random_search_returns_result(self, scenario: Scenario) -> None:
        result = run_random_search(scenario, 1.0, n_samples=30, seed=42)
        assert isinstance(result, BellRandomSearchResult)
        n_dims = {"A": 4, "B": 1, "C": 3}[scenario.value]
        assert result.samples.shape == (30, n_dims)
        assert len(result.delta_omega_values) == 30
        assert result.best_delta_omega > 0 or np.isinf(result.best_delta_omega)

    def test_random_search_best_is_minimum(self) -> None:
        result = run_random_search(Scenario.A, 1.0, n_samples=30, seed=42)
        assert np.isclose(
            result.best_delta_omega,
            np.min(result.delta_omega_values),
        ), "best_delta_omega should be the minimum of delta_omega_values"

    def test_random_search_reproducible(self) -> None:
        r1 = run_random_search(Scenario.A, 1.0, n_samples=20, seed=123)
        r2 = run_random_search(Scenario.A, 1.0, n_samples=20, seed=123)
        assert np.allclose(r1.samples, r2.samples)
        assert np.allclose(r1.delta_omega_values, r2.delta_omega_values)

    def test_random_search_reproducible_scenario_b(self) -> None:
        r1 = run_random_search(Scenario.B, 1.0, n_samples=20, seed=123)
        r2 = run_random_search(Scenario.B, 1.0, n_samples=20, seed=123)
        assert np.allclose(r1.samples, r2.samples)
        assert np.allclose(r1.delta_omega_values, r2.delta_omega_values)


# ============================================================================
# Test: Nelder-Mead Optimisation
# ============================================================================


class TestNelderMead:
    """Tests for the Nelder-Mead optimisation."""

    def test_nelder_mead_returns_result(self, make_ops: dict[str, np.ndarray]) -> None:
        result = run_nelder_mead(
            scenario=Scenario.A,
            omega_true=1.0,
            ops=make_ops,
            x0=np.array([1.0, 2.0, -1.0, 1.5]),
            maxiter=200,
        )
        assert isinstance(result, BellNelderMeadResult)
        assert result.delta_omega_opt > 0 or np.isinf(result.delta_omega_opt)
        assert len(result.params_opt) == 4
        assert result.nfev > 0

    def test_nelder_mead_scenario_b(self, make_ops: dict[str, np.ndarray]) -> None:
        result = run_nelder_mead(
            scenario=Scenario.B,
            omega_true=1.0,
            ops=make_ops,
            x0=np.array([1.5]),
            maxiter=200,
        )
        assert isinstance(result, BellNelderMeadResult)
        assert len(result.params_opt) == 1

    def test_nelder_mead_scenario_c(self, make_ops: dict[str, np.ndarray]) -> None:
        result = run_nelder_mead(
            scenario=Scenario.C,
            omega_true=1.0,
            ops=make_ops,
            x0=np.array([1.0, 2.0, -1.0]),
            maxiter=200,
        )
        assert isinstance(result, BellNelderMeadResult)
        assert len(result.params_opt) == 3

    def test_sensitivity_objective_penalty(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        """Out-of-bounds parameters should receive a large penalty."""
        # a_zz = 10.0 (out of bounds, max is 5.0)
        val = sensitivity_objective(
            np.array([0.0, 0.0, 0.0, 10.0]),
            Scenario.A,
            omega_true=1.0,
            ops=make_ops,
        )
        assert val > 1e9, f"Expected large penalty for out-of-bounds, got {val}"


# ============================================================================
# Test: Serialization
# ============================================================================


class TestSerialization:
    """Tests for Parquet roundtrip of result dataclasses."""

    def test_random_search_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1, 1, size=(10, 4))
        deltas = rng.uniform(0.01, 1.0, size=10)
        best_idx = int(np.argmin(deltas))
        orig = BellRandomSearchResult(
            samples=samples,
            delta_omega_values=deltas,
            best_params=tuple(float(samples[best_idx, i]) for i in range(4)),
            best_delta_omega=float(deltas[best_idx]),
            scenario="A",
            omega_value=1.0,
            sql=0.1,
            t_hold=10.0,
        )
        path = tmp_path / "test_random.parquet"
        orig.save_parquet(path)
        loaded = BellRandomSearchResult.from_parquet(path)
        assert np.allclose(loaded.samples, orig.samples)
        assert np.allclose(loaded.delta_omega_values, orig.delta_omega_values)
        assert loaded.best_params == orig.best_params
        assert np.isclose(loaded.best_delta_omega, orig.best_delta_omega)
        assert loaded.scenario == orig.scenario
        assert np.isclose(loaded.omega_value, orig.omega_value)

    def test_nelder_mead_roundtrip(self, tmp_path: Path) -> None:
        orig = BellNelderMeadResult(
            delta_omega_opt=0.05,
            params_opt=np.array([1.0, 2.0, 3.0, 4.0]),
            omega_true=0.5,
            scenario="A",
            success=True,
            nfev=100,
            message="OK",
            expectation_Jz=0.5,
            variance_Jz=0.1,
            history=[0.1, 0.08, 0.05],
        )
        path = tmp_path / "test_nm.parquet"
        orig.save_parquet(path)
        loaded = BellNelderMeadResult.from_parquet(path)
        assert_roundtrip_fields(
            loaded,
            orig,
            [
                ("delta_omega_opt", "isclose"),
                ("omega_true", "isclose"),
                ("scenario", "eq"),
                ("success", "eq"),
                ("nfev", "eq"),
                ("expectation_Jz", "isclose"),
                ("variance_Jz", "isclose"),
            ],
        )
        assert np.allclose(loaded.params_opt, orig.params_opt)
        assert loaded.history == orig.history, (
            f"History roundtrip mismatch: {loaded.history} != {orig.history}"
        )

    def test_nelder_mead_roundtrip_scenario_b(self, tmp_path: Path) -> None:
        orig = BellNelderMeadResult(
            delta_omega_opt=0.05,
            params_opt=np.array([2.5]),
            omega_true=0.5,
            scenario="B",
            success=True,
            nfev=80,
            message="OK",
            expectation_Jz=0.3,
            variance_Jz=0.2,
        )
        path = tmp_path / "test_nm_b.parquet"
        orig.save_parquet(path)
        loaded = BellNelderMeadResult.from_parquet(path)
        assert_roundtrip_fields(
            loaded,
            orig,
            [
                ("delta_omega_opt", "isclose"),
                ("omega_true", "isclose"),
                ("scenario", "eq"),
                ("success", "eq"),
            ],
        )
        assert np.allclose(loaded.params_opt, orig.params_opt)
        # Scenario B params should map correctly
        assert len(loaded.params_opt) == 1

    def test_nelder_mead_no_history_sidecar(self, tmp_path: Path) -> None:
        """Loading a NM result without a history sidecar should return empty history."""
        orig = BellNelderMeadResult(
            delta_omega_opt=0.05,
            params_opt=np.array([1.0, 2.0, 3.0, 4.0]),
            omega_true=0.5,
            scenario="A",
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
        loaded = BellNelderMeadResult.from_parquet(path)
        assert loaded.history == [], f"Expected empty history, got {loaded.history}"

    def test_optimisation_result_roundtrip(self, tmp_path: Path) -> None:
        orig = BellOptimisationResult(
            scenario="A",
            omega=1.0,
            delta_omega_opt=0.05,
            sql=sql_reference(1, T_HOLD),
            ratio=sql_reference(1, T_HOLD) / 0.05,
            a_x_opt=1.0,
            a_y_opt=2.0,
            a_z_opt=3.0,
            a_zz_opt=4.0,
            expectation_Jz=0.5,
            variance_Jz=0.1,
            drive_norm=math.sqrt(1.0**2 + 2.0**2 + 3.0**2),
            d_exp=0.75,
            is_fringe=False,
            success=True,
            nfev=100,
        )
        path = tmp_path / "test_opt.parquet"
        orig.save_parquet(path)
        loaded = BellOptimisationResult.from_parquet(path)
        assert_roundtrip_fields(
            loaded,
            orig,
            [
                ("scenario", "eq"),
                ("omega", "isclose"),
                ("delta_omega_opt", "isclose"),
                ("sql", "isclose"),
                ("ratio", "isclose"),
                ("a_x_opt", "isclose"),
                ("a_y_opt", "isclose"),
                ("a_z_opt", "isclose"),
                ("a_zz_opt", "isclose"),
                ("expectation_Jz", "isclose"),
                ("variance_Jz", "isclose"),
                ("drive_norm", "isclose"),
                ("d_exp", "isclose"),
                ("is_fringe", "eq"),
                ("success", "eq"),
                ("nfev", "eq"),
                ("t_hold", "isclose"),
                ("fd_step", "isclose"),
            ],
        )

    def test_scan_result_roundtrip(self, tmp_path: Path) -> None:
        sql_val = sql_reference(1, T_HOLD)
        r1 = BellOptimisationResult(
            scenario="A",
            omega=0.5,
            delta_omega_opt=0.05,
            sql=sql_val,
            ratio=sql_val / 0.05,
            a_x_opt=1.0,
            a_y_opt=2.0,
            a_z_opt=3.0,
            a_zz_opt=4.0,
            expectation_Jz=0.5,
            variance_Jz=0.1,
            drive_norm=math.sqrt(1.0**2 + 2.0**2 + 3.0**2),
        )
        r2 = BellOptimisationResult(
            scenario="B",
            omega=1.0,
            delta_omega_opt=0.03,
            sql=sql_val,
            ratio=sql_val / 0.03,
            a_x_opt=0.0,
            a_y_opt=0.0,
            a_z_opt=0.0,
            a_zz_opt=2.5,
            expectation_Jz=0.4,
            variance_Jz=0.2,
            drive_norm=0.0,
        )
        orig = BellScanResult(results=[r1, r2])
        path = tmp_path / "test_scan.parquet"
        orig.save_parquet(path)
        loaded = BellScanResult.from_parquet(path)
        assert len(loaded.results) == 2
        for lr, rr in zip(loaded.results, orig.results, strict=True):
            assert lr.scenario == rr.scenario
            assert np.isclose(lr.omega, rr.omega)
            assert np.isclose(lr.delta_omega_opt, rr.delta_omega_opt)

    def test_scan_result_properties(self) -> None:
        sql_val = sql_reference(1, T_HOLD)
        r1 = BellOptimisationResult(
            scenario="A",
            omega=0.5,
            delta_omega_opt=0.05,
            sql=sql_val,
            ratio=sql_val / 0.05,
            a_x_opt=1.0,
            a_y_opt=2.0,
            a_z_opt=3.0,
            a_zz_opt=4.0,
            expectation_Jz=0.5,
            variance_Jz=0.1,
            drive_norm=math.sqrt(14.0),
        )
        r2 = BellOptimisationResult(
            scenario="B",
            omega=1.0,
            delta_omega_opt=0.03,
            sql=sql_val,
            ratio=sql_val / 0.03,
            a_x_opt=0.0,
            a_y_opt=0.0,
            a_z_opt=0.0,
            a_zz_opt=2.5,
            expectation_Jz=0.4,
            variance_Jz=0.2,
            drive_norm=0.0,
        )
        scan = BellScanResult(results=[r1, r2])
        assert set(scan.scenario_values) == {"A", "B"}
        assert set(scan.omega_values) == {0.5, 1.0}

    def test_random_search_missing_column_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a_x": [1.0], "a_y": [2.0]})
        path = tmp_path / "bad.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError):
            BellRandomSearchResult.from_parquet(path)

    def test_optimisation_result_missing_column_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"scenario": ["A"], "omega": [1.0]})
        path = tmp_path / "bad.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError):
            BellOptimisationResult.from_parquet(path)

    def test_nelder_mead_missing_column_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"scenario": ["A"], "delta_omega": [0.05]})
        path = tmp_path / "bad.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError):
            BellNelderMeadResult.from_parquet(path)


# ============================================================================
# Test: Physical Invariants
# ============================================================================


class TestPhysicalInvariants:
    """Physical invariants for the MZI circuit."""

    def test_bs_unitary(self) -> None:
        from src.analysis.ancilla_drive_metrology import system_only_bs_unitary

        U_bs = system_only_bs_unitary(T_BS)
        I_4_mat = np.eye(4, dtype=complex)
        assert np.allclose(U_bs @ U_bs.conj().T, I_4_mat, atol=1e-12), (
            "BS unitary not unitary"
        )

    def test_commutation_system_ancilla(self, make_ops: dict[str, np.ndarray]) -> None:
        """[J_z^S, J_z^A] = 0."""
        comm = make_ops["Jz_S"] @ make_ops["Jz_A"] - make_ops["Jz_A"] @ make_ops["Jz_S"]
        assert np.allclose(comm, 0.0, atol=1e-12)

    def test_su2_commutation(self, make_ops: dict[str, np.ndarray]) -> None:
        """[J_z^S, J_x^S] = i J_y^S."""
        comm = make_ops["Jz_S"] @ make_ops["Jx_S"] - make_ops["Jx_S"] @ make_ops["Jz_S"]
        assert np.allclose(comm, 1j * make_ops["Jy_S"], atol=1e-10)

    def test_hold_unitary(self, make_ops: dict[str, np.ndarray]) -> None:
        U = _hold_unitary(T_HOLD, 1.0, 1.0, 2.0, -1.5, 2.5, make_ops)
        assert np.allclose(U @ U.conj().T, np.eye(4, dtype=complex), atol=1e-12), (
            "Hold unitary not unitary"
        )

    def test_derivative_stability(self, make_ops: dict[str, np.ndarray]) -> None:
        """Finite-difference derivative should be stable over δ ∈ [1e-7, 1e-5]."""
        psi0 = bell_state_phi_plus()
        omega = 1.0
        a_x, a_y, a_z, a_zz = 1.0, 2.0, -1.5, 2.5
        steps = [1e-7, 1e-6, 1e-5]
        derivatives = []
        for step in steps:
            _, _, _, d_exp, _ = _compute_sensitivity(
                psi0,
                omega,
                a_x,
                a_y,
                a_z,
                a_zz,
                make_ops,
                fd_step=step,
            )
            if np.isfinite(d_exp):
                derivatives.append(d_exp)
        if len(derivatives) >= 2:
            max_rel = max(
                abs(d - derivatives[0]) / max(abs(derivatives[0]), 1e-15)
                for d in derivatives
            )
            assert max_rel < 1e-3, (
                f"Derivative unstable: relative variation {max_rel:.2e}"
            )


# ============================================================================
# Test: Scenario Definitions
# ============================================================================


class TestScenarioDefinitions:
    """Tests for scenario constants."""

    def test_active_scenarios(self) -> None:
        assert Scenario.A in ACTIVE_SCENARIOS
        assert Scenario.B in ACTIVE_SCENARIOS
        assert Scenario.C in ACTIVE_SCENARIOS

    def test_scenario_values(self) -> None:
        assert Scenario.A.value == "A"
        assert Scenario.B.value == "B"
        assert Scenario.C.value == "C"
        assert Scenario.D.value == "D"
        assert Scenario.E.value == "E"


# ============================================================================
# Test: End-to-End Pipeline
# ============================================================================


class TestPipeline:
    """Tests for the full optimisation pipeline."""

    @pytest.mark.slow
    def test_run_single_scenario_a(self) -> None:
        """Run scenario A pipeline (uses defaults from local)."""
        result = run_single_scenario_omega(Scenario.A, 1.0, seed=42)
        assert isinstance(result, BellOptimisationResult)
        assert result.scenario == "A"
        assert np.isclose(result.omega, 1.0)
        assert result.delta_omega_opt > 0
        assert result.ratio > 0

    @pytest.mark.slow
    def test_run_single_scenario_b(self) -> None:
        result = run_single_scenario_omega(Scenario.B, 1.0, seed=42)
        assert isinstance(result, BellOptimisationResult)
        assert result.scenario == "B"

    @pytest.mark.slow
    def test_run_single_scenario_c(self) -> None:
        result = run_single_scenario_omega(Scenario.C, 1.0, seed=42)
        assert isinstance(result, BellOptimisationResult)
        assert result.scenario == "C"

    def test_scan_result_empty(self) -> None:
        scan = BellScanResult()
        assert scan.to_dataframe().empty
        assert len(scan.results) == 0

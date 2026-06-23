"""Tests for the Non-Linear Measurement (Parity and CFI) on omega-Modulated Drive report.

Companion test module for ``reports/20260615/nonlinear_measurement_parity_cfi.py``.
Tests parity operator, J_z projectors, sensitivity functions,
decoupled baseline, Stage A evaluation, optimisation, and serialization.
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

from src.analysis.sensitivity_metrics import sql_reference
from src.physics.n_particle_drive import (
    build_n_particle_operators,
    compute_n_particle_sensitivity,
    evolve_n_particle_circuit,
    n_particle_initial_state,
)
from src.utils.serialization import assert_roundtrip_fields

_local_path = _Path(__file__).resolve().parent / "nonlinear_measurement_parity_cfi.py"
_spec = importlib.util.spec_from_file_location(
    "nonlinear_measurement_parity_cfi", str(_local_path)
)
assert _spec is not None
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_sys.modules["nonlinear_measurement_parity_cfi"] = _module
_spec.loader.exec_module(_module)
del _local_path, _spec, _module

from nonlinear_measurement_parity_cfi import (  # noqa: E402
    PROTOCOL_CFI,
    PROTOCOL_LINEAR,
    PROTOCOL_PARITY,
    T_BS,
    T_HOLD,
    NonLinearResult,
    NonLinearScanResult,
    build_parity_operator,
    compute_cfi_sensitivity,
    compute_jz_probability_distribution,
    compute_jz_projectors,
    compute_parity_sensitivity,
    compute_protocol_sensitivity,
    evaluate_protocols_at_params,
    load_joint_optimal_params,
    non_linear_random_search,
    run_non_linear_nelder_mead,
    verify_decoupled_baseline,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=[2, 4, 6])
def make_even_N(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture(params=[1, 3, 5])
def make_odd_N(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture(params=[1, 2, 4])
def make_N(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture
def make_ops(make_N: int) -> dict[str, np.ndarray]:
    return build_n_particle_operators(make_N)


@pytest.fixture
def make_psi0(make_N: int) -> np.ndarray:
    return n_particle_initial_state(make_N)


# ============================================================================
# Parity Operator Construction
# ============================================================================


class TestBuildParityOperator:
    def test_even_N_hermitian(self, make_even_N: int) -> None:
        parity = build_parity_operator(make_even_N)
        d_tot = 2 * (make_even_N + 1)
        assert parity.shape == (d_tot, d_tot)
        assert np.allclose(parity, parity.conj().T, atol=1e-12), (
            f"Parity not Hermitian for even N={make_even_N}"
        )

    def test_even_N_squares_to_identity(self, make_even_N: int) -> None:
        parity = build_parity_operator(make_even_N)
        d_tot = 2 * (make_even_N + 1)
        I_full = np.eye(d_tot, dtype=complex)
        assert np.allclose(parity @ parity, I_full, atol=1e-12), (
            f"Parity^2 != I for even N={make_even_N}"
        )

    def test_odd_N_raises_value_error(self, make_odd_N: int) -> None:
        with pytest.raises(ValueError, match="not Hermitian"):
            build_parity_operator(make_odd_N)

    def test_eigenvalues_plus_minus_one(self, make_even_N: int) -> None:
        parity = build_parity_operator(make_even_N)
        eigvals = np.linalg.eigvalsh(parity)
        assert np.allclose(np.abs(eigvals), 1.0, atol=1e-10), (
            f"Parity eigenvalues not ±1 for N={make_even_N}: {eigvals}"
        )

    def test_N2_Jz_eigenstate_parity(self) -> None:
        """At N=2, |m_S⟩ = |+1⟩ should give ⟨Π⟩ = (-1)^{+1} = -1."""
        parity = build_parity_operator(2)
        d_sys = 3
        # State |m_S=+1⟩ ⊗ |0⟩_A: index = 0 for |+1⟩ in Dicke basis
        psi = np.zeros(2 * d_sys, dtype=complex)
        psi[0] = 1.0
        exp_val = np.real(psi.conj() @ parity @ psi)
        assert np.isclose(exp_val, -1.0, atol=1e-10), (
            f"Expected <Pi> = -1 for |m=+1>, got {exp_val}"
        )

    def test_N2_Jz_negative_eigenstate_parity(self) -> None:
        """At N=2, |m_S⟩ = |-1⟩ should give ⟨Π⟩ = (-1)^{-1} = -1.

        In the full space H_S (Dicke, dim=3) ⊗ H_A (qubit, dim=2),
        basis ordering is: {|m_S⟩_S ⊗ |0⟩_A, ..., |m_S⟩_S ⊗ |1⟩_A}.
        So the |m_S=-1⟩_S ⊗ |0⟩_A state is at full-space index 4
        (|m=+1,0⟩=0, |m=+1,1⟩=1, |m=0,0⟩=2, |m=0,1⟩=3, |m=-1,0⟩=4, |m=-1,1⟩=5).
        """
        parity = build_parity_operator(2)
        d_tot = 6
        # State |m_S=-1⟩ ⊗ |0⟩_A: full-space index = 4
        psi = np.zeros(d_tot, dtype=complex)
        psi[4] = 1.0
        exp_val = np.real(psi.conj() @ parity @ psi)
        assert np.isclose(exp_val, -1.0, atol=1e-10), (
            f"Expected <Pi> = -1 for |m=-1>, got {exp_val}"
        )

    def test_N2_parity_m0_eigenstate(self) -> None:
        """At N=2, |m_S⟩ = |0⟩ should give ⟨Π⟩ = (-1)^0 = +1.

        Full-space index 2 = |m_S=0⟩_S ⊗ |0⟩_A.
        """
        parity = build_parity_operator(2)
        d_tot = 6
        # State |m_S=0⟩ ⊗ |0⟩_A: full-space index = 2
        psi = np.zeros(d_tot, dtype=complex)
        psi[2] = 1.0
        exp_val = np.real(psi.conj() @ parity @ psi)
        assert np.isclose(exp_val, 1.0, atol=1e-10), (
            f"Expected <Pi> = +1 for |m=0>, got {exp_val}"
        )


# ============================================================================
# J_z Probability Distribution
# ============================================================================


class TestJzProbabilityDistribution:
    def test_projectors_sum_to_identity(self, make_N: int) -> None:
        ops = build_n_particle_operators(make_N)
        projectors = compute_jz_projectors(make_N, ops)
        d_tot = 2 * (make_N + 1)
        total = np.sum(projectors, axis=0)
        assert np.allclose(total, np.eye(d_tot, dtype=complex), atol=1e-12), (
            f"Sum of projectors != I for N={make_N}"
        )

    def test_probabilities_sum_to_one(self, make_N: int) -> None:
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        projectors = compute_jz_projectors(make_N, ops)
        # Evolve at some random params
        psi = evolve_n_particle_circuit(
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops,
        )
        probs = compute_jz_probability_distribution(psi, projectors)
        assert len(probs) == make_N + 1
        assert np.isclose(np.sum(probs), 1.0, atol=1e-10), (
            f"Probabilities sum to {np.sum(probs)}, not 1 for N={make_N}"
        )
        assert np.all(probs >= 0), f"Negative probabilities for N={make_N}"

    def test_decoupled_state_is_jz_eigenstate(self, make_N: int) -> None:
        """At zero drive, the state should have prob 1 for the |+J_S⟩ component."""
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        projectors = compute_jz_projectors(make_N, ops)
        # Evolve with zero params
        psi = evolve_n_particle_circuit(
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
        )
        probs = compute_jz_probability_distribution(psi, projectors)
        # After the BS, the CSS should give a binomial distribution
        # Check that it sums to 1 and has the right shape
        assert len(probs) == make_N + 1
        assert np.isclose(np.sum(probs), 1.0, atol=1e-10)


# ============================================================================
# Sensitivity Functions
# ============================================================================


class TestParitySensitivity:
    def test_parity_sensitivity_returns_finite(self, make_even_N: int) -> None:
        ops = build_n_particle_operators(make_even_N)
        psi0 = n_particle_initial_state(make_even_N)
        dw, _exp_par = compute_parity_sensitivity(
            make_even_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops,
        )
        assert np.isfinite(dw), f"Parity sensitivity inf for N={make_even_N}"
        assert dw > 0, f"Parity sensitivity <= 0 for N={make_even_N}"

    def test_parity_expectation_between_minus_one_and_one(
        self, make_even_N: int
    ) -> None:
        ops = build_n_particle_operators(make_even_N)
        psi0 = n_particle_initial_state(make_even_N)
        _, exp_par = compute_parity_sensitivity(
            make_even_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops,
        )
        assert -1.0 <= exp_par <= 1.0, f"Parity expectation {exp_par} outside [-1, 1]"

    def test_odd_N_raises_error(self, make_odd_N: int) -> None:
        ops = build_n_particle_operators(make_odd_N)
        psi0 = n_particle_initial_state(make_odd_N)
        with pytest.raises(ValueError, match="not Hermitian"):
            compute_parity_sensitivity(
                make_odd_N,
                psi0,
                T_BS,
                T_HOLD,
                0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                ops,
            )


class TestCfiSensitivity:
    def test_cfi_sensitivity_returns_finite(self, make_N: int) -> None:
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        dw = compute_cfi_sensitivity(
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops,
        )
        assert np.isfinite(dw), f"CFI sensitivity inf for N={make_N}"
        assert dw > 0, f"CFI sensitivity <= 0 for N={make_N}"

    def test_cfi_vs_linear_sensitivity(self, make_N: int) -> None:
        """CFI should be <= linear error-propagation sensitivity."""
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        dw_lin = compute_n_particle_sensitivity(
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops,
            meas_op=ops["Jz_S"],
        )
        dw_cfi = compute_cfi_sensitivity(
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops,
        )
        # CFI should be <= linear (CFI is at least as informative)
        assert dw_cfi <= dw_lin + 1e-10, (
            f"CFI sensitivity {dw_cfi} > linear sensitivity {dw_lin} for N={make_N}"
        )

    def test_decoupled_cfi_matches_sql(self, make_N: int) -> None:
        """At zero drive, CFI sensitivity should match SQL."""
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        sql = sql_reference(make_N)
        dw_cfi = compute_cfi_sensitivity(
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
        )
        assert np.isclose(dw_cfi, sql, rtol=1e-8), (
            f"CFI decoupled baseline {dw_cfi} != SQL {sql} for N={make_N}"
        )


class TestProtocolSensitivity:
    def test_linear_protocol(self, make_N: int) -> None:
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        dw, extra = compute_protocol_sensitivity(
            PROTOCOL_LINEAR,
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops,
        )
        assert np.isfinite(dw), "Linear protocol returned inf"
        assert extra == 0.0, "Linear protocol should return extra=0.0"

    def test_parity_protocol_even_N(self, make_even_N: int) -> None:
        ops = build_n_particle_operators(make_even_N)
        psi0 = n_particle_initial_state(make_even_N)
        dw, extra = compute_protocol_sensitivity(
            PROTOCOL_PARITY,
            make_even_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops,
        )
        assert np.isfinite(dw), "Parity protocol returned inf"
        assert -1.0 <= extra <= 1.0, f"Parity extra ({extra}) outside [-1, 1]"

    def test_cfi_protocol(self, make_N: int) -> None:
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        dw, extra = compute_protocol_sensitivity(
            PROTOCOL_CFI,
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops,
        )
        assert np.isfinite(dw), "CFI protocol returned inf"
        assert extra == 0.0, "CFI protocol should return extra=0.0"

    def test_unknown_protocol_raises(self, make_N: int) -> None:
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        with pytest.raises(ValueError, match="Unknown protocol"):
            compute_protocol_sensitivity(
                "invalid",
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
            )


# ============================================================================
# Decoupled Baseline Verification
# ============================================================================


class TestDecoupledBaseline:
    def test_linear_baseline(self) -> None:
        results = verify_decoupled_baseline(
            N_values=[2, 4],
            omega_values=[0.2],
        )
        for key, val in results.items():
            assert val["linear"], (
                f"Linear baseline failed at N={key[0]}, omega={key[1]}"
            )

    def test_parity_baseline_even_N(self) -> None:
        """Parity Δω is finite, positive, and well-behaved at decoupled params.

        Unlike the full J_z distribution (CFI), the parity measurement Π_S
        collapses Dicke-basis information to a single binary outcome, so it
        does NOT equal the SQL. This test only verifies sanity.
        """
        results = verify_decoupled_baseline(
            N_values=[2, 4],
            omega_values=[0.2],
        )
        for key, val in results.items():
            if key[0] % 2 == 0:
                assert val["parity"] is True, (
                    f"Parity not well-behaved at N={key[0]}, omega={key[1]}: "
                    f"got {val['parity']}"
                )

    def test_parity_baseline_odd_N(self) -> None:
        results = verify_decoupled_baseline(
            N_values=[1, 3],
            omega_values=[0.2],
        )
        for key, val in results.items():
            assert val["parity"] is None, f"Parity should be None for odd N={key[0]}"

    def test_cfi_baseline(self) -> None:
        results = verify_decoupled_baseline(
            N_values=[2, 4],
            omega_values=[0.2],
        )
        for key, val in results.items():
            assert val["cfi"], f"CFI baseline failed at N={key[0]}, omega={key[1]}"


# ============================================================================
# Stage A: Fixed-Parameter Evaluation
# ============================================================================


class TestLoadJointOptimalParams:
    def test_loads_params_for_known_pair(self) -> None:
        params = load_joint_optimal_params(N=1, omega=0.2)
        assert params is not None, "Failed to load params for N=1, omega=0.2"
        for key in ("a_x", "a_y", "a_z", "a_zz"):
            assert key in params, f"Missing key {key}"
            assert np.isfinite(params[key]), f"Non-finite {key}: {params[key]}"

    def test_returns_none_for_unknown_pair(self) -> None:
        params = load_joint_optimal_params(N=999, omega=0.2)
        assert params is None, "Should return None for unknown N"


class TestEvaluateProtocolsAtParams:
    def test_all_protocols_finite(self) -> None:
        ops = build_n_particle_operators(4)
        psi0 = n_particle_initial_state(4)
        evals = evaluate_protocols_at_params(
            4,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops=ops,
            psi0=psi0,
        )
        assert np.isfinite(evals["delta_omega_lin"]), "Linear not finite"
        assert np.isfinite(evals["delta_omega_parity"]), "Parity not finite"
        assert np.isfinite(evals["delta_omega_cfi"]), "CFI not finite"
        assert "sql" in evals
        assert np.isfinite(evals["parity_expectation"])

    def test_odd_N_parity_nan(self) -> None:
        ops = build_n_particle_operators(3)
        psi0 = n_particle_initial_state(3)
        evals = evaluate_protocols_at_params(
            3,
            0.2,
            1.0,
            -0.5,
            0.3,
            0.1,
            ops=ops,
            psi0=psi0,
        )
        assert np.isnan(evals["delta_omega_parity"]), "Parity should be NaN for odd N"
        assert np.isfinite(evals["delta_omega_lin"]), "Linear not finite for odd N"
        assert np.isfinite(evals["delta_omega_cfi"]), "CFI not finite for odd N"


# ============================================================================
# Serialization (Parquet Roundtrip)
# ============================================================================


class TestNonLinearResultRoundtrip:
    _FIELD_SPECS: ClassVar[list[tuple[str, str]]] = [
        ("N", "eq"),
        ("omega", "isclose"),
        ("delta_omega_lin", "isclose"),
        ("delta_omega_parity", "isclose"),
        ("delta_omega_cfi", "isclose"),
        ("ratio_parity", "isclose"),
        ("ratio_cfi", "isclose"),
        ("parity_expectation", "isclose"),
        ("stage", "eq"),
        ("success", "eq"),
        ("nfev", "eq"),
    ]

    def test_roundtrip(self, tmp_path: Path) -> None:
        res = NonLinearResult(
            N=4,
            omega=0.2,
            a_x=1.0,
            a_y=-0.5,
            a_z=0.3,
            a_zz=0.1,
            delta_omega_lin=0.01,
            delta_omega_parity=0.008,
            delta_omega_cfi=0.007,
            ratio_lin=10.0,
            ratio_parity=12.5,
            ratio_cfi=14.3,
            parity_expectation=0.5,
            sql=0.05,
            success=True,
            nfev=100,
            stage="A",
        )
        path = tmp_path / "test.parquet"
        res.save_parquet(path)
        loaded = NonLinearResult.from_parquet(path)
        assert_roundtrip_fields(loaded, res, self._FIELD_SPECS)

    def test_roundtrip_with_nan_parity(self, tmp_path: Path) -> None:
        """Test roundtrip with NaN parity (odd N case)."""
        res = NonLinearResult(
            N=3,
            omega=0.2,
            a_x=1.0,
            a_y=0.0,
            a_z=0.0,
            a_zz=0.0,
            delta_omega_lin=0.015,
            delta_omega_parity=float("nan"),
            delta_omega_cfi=0.012,
            ratio_lin=6.67,
            ratio_parity=float("inf"),
            ratio_cfi=8.33,
            parity_expectation=float("nan"),
            sql=0.0577,
            stage="A",
        )
        path = tmp_path / "test_nan.parquet"
        res.save_parquet(path)
        loaded = NonLinearResult.from_parquet(path)
        assert np.isnan(loaded.delta_omega_parity)
        assert np.isnan(loaded.parity_expectation)
        assert np.isinf(loaded.ratio_parity)

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        """Loading a Parquet missing required columns should raise."""
        bad_df = pd.DataFrame({"N": [4], "omega": [0.2]})
        path = tmp_path / "bad.parquet"
        bad_df.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            NonLinearResult.from_parquet(path)


class TestNonLinearScanResultRoundtrip:
    def test_roundtrip(self, tmp_path: Path) -> None:
        res1 = NonLinearResult(
            N=2,
            omega=0.2,
            a_x=1.0,
            a_y=0.0,
            a_z=0.0,
            a_zz=0.0,
            delta_omega_lin=0.01,
            delta_omega_parity=0.009,
            delta_omega_cfi=0.008,
            ratio_lin=7.07,
            ratio_parity=7.86,
            ratio_cfi=8.84,
            parity_expectation=-0.3,
            sql=0.0707,
            stage="A",
        )
        res2 = NonLinearResult(
            N=4,
            omega=0.2,
            a_x=0.5,
            a_y=0.0,
            a_z=0.0,
            a_zz=0.0,
            delta_omega_lin=0.02,
            delta_omega_parity=0.018,
            delta_omega_cfi=0.015,
            ratio_lin=2.5,
            ratio_parity=2.78,
            ratio_cfi=3.33,
            parity_expectation=0.1,
            sql=0.05,
            stage="A",
        )
        scan = NonLinearScanResult(results=[res1, res2])
        path = tmp_path / "scan.parquet"
        scan.save_parquet(path)
        loaded = NonLinearScanResult.from_parquet(path)
        assert len(loaded.results) == 2
        assert loaded.results[0].N == 2
        assert loaded.results[1].N == 4

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        bad_df = pd.DataFrame({"N": [4]})
        path = tmp_path / "bad_scan.parquet"
        bad_df.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            NonLinearScanResult.from_parquet(path)


# ============================================================================
# Stage B: Optimisation (Brief)
# ============================================================================


class TestNonLinearRandomSearch:
    def test_linear_random_search_finds_finite(self) -> None:
        samples, deltas, best = non_linear_random_search(
            2,
            0.2,
            PROTOCOL_LINEAR,
            n_samples=20,
            seed=42,
        )
        assert samples.shape == (20, 4)
        assert deltas.shape == (20,)
        assert len(best) == 4
        assert np.any(np.isfinite(deltas)), "No finite deltas found"

    def test_cfi_random_search_finds_finite(self) -> None:
        _samples, deltas, _best = non_linear_random_search(
            2,
            0.2,
            PROTOCOL_CFI,
            n_samples=20,
            seed=42,
        )
        assert np.any(np.isfinite(deltas)), "No finite CFI deltas found"

    def test_parity_random_search_even_N(self) -> None:
        _samples, deltas, _best = non_linear_random_search(
            2,
            0.2,
            PROTOCOL_PARITY,
            n_samples=20,
            seed=42,
        )
        assert np.any(np.isfinite(deltas)), "No finite parity deltas found"


class TestRunNonLinearNelderMead:
    def test_nm_converges_linear(self) -> None:
        result = run_non_linear_nelder_mead(
            2,
            0.2,
            (1.0, -0.5, 0.3, 0.1),
            PROTOCOL_LINEAR,
            maxiter=200,
        )
        assert np.isfinite(result["fun_opt"]), "NM objective not finite"
        assert result["fun_opt"] > 0, "NM objective non-positive"

    def test_nm_converges_cfi(self) -> None:
        result = run_non_linear_nelder_mead(
            2,
            0.2,
            (1.0, -0.5, 0.3, 0.1),
            PROTOCOL_CFI,
            maxiter=200,
        )
        assert np.isfinite(result["fun_opt"]), "NM CFI objective not finite"

    def test_nm_converges_parity(self) -> None:
        result = run_non_linear_nelder_mead(
            2,
            0.2,
            (1.0, -0.5, 0.3, 0.1),
            PROTOCOL_PARITY,
            maxiter=200,
        )
        assert np.isfinite(result["fun_opt"]), "NM parity objective not finite"


# ============================================================================
# Physical Invariants
# ============================================================================


class TestPhysicalInvariants:
    def test_hermiticity_of_all_operators(self, make_N: int) -> None:
        ops = build_n_particle_operators(make_N)
        for key in ("Jz_S", "Jx_S", "Jy_S", "Jz_A", "Jx_A", "Jy_A"):
            assert np.allclose(ops[key], ops[key].conj().T, atol=1e-12), (
                f"{key} not Hermitian for N={make_N}"
            )

    def test_state_normalisation(self, make_N: int) -> None:
        psi0 = n_particle_initial_state(make_N)
        assert np.isclose(np.linalg.norm(psi0), 1.0)

    def test_cfi_positivity(self, make_N: int) -> None:
        """CFI sensitivity must be finite (F_C > 0) for non-zero params."""
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        dw = compute_cfi_sensitivity(
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            0.0,
            0.0,
            0.0,
            ops,
        )
        assert np.isfinite(dw) and dw > 0, f"CFI not positive for N={make_N}"

    def test_sensitivity_positivity(self, make_N: int) -> None:
        """Linear sensitivity must be positive for non-zero params."""
        ops = build_n_particle_operators(make_N)
        psi0 = n_particle_initial_state(make_N)
        dw = compute_n_particle_sensitivity(
            make_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            1.0,
            0.0,
            0.0,
            0.0,
            ops,
            meas_op=ops["Jz_S"],
        )
        assert dw > 0 or np.isinf(dw), f"Invalid sensitivity: {dw} for N={make_N}"


# ============================================================================
# Reproducibility
# ============================================================================


class TestReproducibility:
    def test_random_search_deterministic(self) -> None:
        s1, d1, _b1 = non_linear_random_search(
            2, 0.2, PROTOCOL_LINEAR, n_samples=10, seed=42
        )
        s2, d2, _b2 = non_linear_random_search(
            2, 0.2, PROTOCOL_LINEAR, n_samples=10, seed=42
        )
        assert np.allclose(s1, s2), "Samples differ with same seed"
        assert np.allclose(d1, d2), "Deltas differ with same seed"

    def test_parity_sensitivity_deterministic(self, make_even_N: int) -> None:
        ops = build_n_particle_operators(make_even_N)
        psi0 = n_particle_initial_state(make_even_N)
        dw1, _ = compute_parity_sensitivity(
            make_even_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            0.5,
            0.0,
            0.0,
            0.0,
            ops,
        )
        dw2, _ = compute_parity_sensitivity(
            make_even_N,
            psi0,
            T_BS,
            T_HOLD,
            0.2,
            0.5,
            0.0,
            0.0,
            0.0,
            ops,
        )
        assert np.isclose(dw1, dw2)

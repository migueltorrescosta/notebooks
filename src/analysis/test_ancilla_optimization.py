"""
Unit tests for the Ancilla-Assisted Metrology Optimisation module.

Covers operator construction, state preparation, beam-splitter unitaries,
holding Hamiltonian/unitary, circuit evolution, sensitivity computation,
objective function, optimisation, bounds, and alpha-coefficient scans.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from src.analysis.ancilla_optimization import (
    I_2,
    AlphaRandomSearchResult,
    AlphaSingleScanResult,
    OptimisationResult,
    ThetaScanResult,
    bs_unitary,
    build_hold_hamiltonian,
    build_interaction_hamiltonian,
    build_two_qubit_operators,
    compute_convergence_metric,
    compute_expectation_and_variance,
    compute_reduced_purity,
    compute_sensitivity,
    evolve_full,
    get_decoupled_sensitivity,
    get_default_bounds,
    hold_unitary,
    random_initial_params,
    random_search_alpha,
    run_optimisation,
    scan_alpha_single_parameter,
    sensitivity_objective,
    single_qubit_state,
    two_qubit_bs_unitary,
    two_qubit_state,
    validate_derivative_stability,
    validate_operators,
    validate_sensitivity_reasonable,
    validate_variance_positive,
)

I_4 = np.eye(4, dtype=complex)


def _default_params() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0, 0.0, 0.0, 0.0, 0.0])


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ops() -> dict[str, np.ndarray]:
    """Two-qubit operators for tests."""
    return build_two_qubit_operators()


# ============================================================================
# Operator Construction
# ============================================================================


class TestOperatorConstruction:
    """Two-qubit operator Hermiticity, dimensions, commutation."""

    def test_shape(self, ops: dict[str, np.ndarray]) -> None:
        for name, op in ops.items():
            assert op.shape == (4, 4), f"{name} shape {op.shape}"

    def test_hermitian(self, ops: dict[str, np.ndarray]) -> None:
        for op in ops.values():
            assert op == pytest.approx(op.conj().T, abs=1e-12)

    def test_jz_diagonal(self, ops: dict[str, np.ndarray]) -> None:
        for name in ["Jz_S", "Jz_A"]:
            assert ops[name] == pytest.approx(np.diag(np.diag(ops[name])))

    def test_jz_eigenvalues(self, ops: dict[str, np.ndarray]) -> None:
        for name in ["Jz_S", "Jz_A"]:
            assert sorted(np.linalg.eigvalsh(ops[name])) == pytest.approx(
                [-0.5, -0.5, 0.5, 0.5]
            )

    def test_commutation_jz_jx(self, ops: dict[str, np.ndarray]) -> None:
        comm_S = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
        assert comm_S == pytest.approx(1j * ops["Jy_S"], abs=1e-12)
        comm_A = ops["Jz_A"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jz_A"]
        assert comm_A == pytest.approx(1j * ops["Jy_A"], abs=1e-12)

    def test_interaction_zero(self) -> None:
        assert build_interaction_hamiltonian((0.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)

    def test_interaction_hermitian(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            alpha = tuple(rng.uniform(-2, 2, size=4))
            H = build_interaction_hamiltonian(alpha)
            assert pytest.approx(H.conj().T, abs=1e-12) == H


# ============================================================================
# State Preparation
# ============================================================================


class TestStatePreparation:
    """Single- and two-qubit state parameterisation, purity."""

    @pytest.mark.parametrize(
        ("theta", "phi"), [(0.0, 0.0), (np.pi, 0.0), (np.pi / 2, np.pi)]
    )
    def test_single_qubit_normalised(self, theta: float, phi: float) -> None:
        assert np.linalg.norm(single_qubit_state(theta, phi)) == pytest.approx(1.0)

    def test_zero_theta(self) -> None:
        assert single_qubit_state(0.0, 0.0) == pytest.approx(
            np.array([1.0, 0.0], dtype=complex)
        )

    def test_pi_theta(self) -> None:
        assert single_qubit_state(np.pi, 0.0) == pytest.approx(
            np.array([0.0, 1.0], dtype=complex)
        )

    def test_two_qubit_product_structure(self) -> None:
        psi = two_qubit_state(0.3, 0.7, 1.2, 2.5)
        expected = np.kron(single_qubit_state(0.3, 0.7), single_qubit_state(1.2, 2.5))
        assert psi == pytest.approx(expected)
        assert np.linalg.norm(psi) == pytest.approx(1.0)

    # --- Reduced purity ---
    @pytest.mark.parametrize(
        ("psi", "expected"),
        [
            (np.array([1.0, 0.0, 0.0, 0.0], dtype=complex), 1.0),  # |00⟩
            (np.array([0.0, 1.0, 0.0, 0.0], dtype=complex), 1.0),  # |01⟩
            (np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2), 0.5),  # |Φ⁺⟩
            (np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2), 0.5),  # |Ψ⁺⟩
        ],
    )
    def test_reduced_purity_known(self, psi: np.ndarray, expected: float) -> None:
        assert compute_reduced_purity(psi) == pytest.approx(expected, abs=1e-12)

    def test_product_state_general(self) -> None:
        assert compute_reduced_purity(
            two_qubit_state(0.7, 1.2, 0.3, 2.8)
        ) == pytest.approx(1.0, abs=1e-12)

    def test_purity_through_circuit(self, ops: dict[str, np.ndarray]) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        psi = evolve_full(
            psi0, np.pi / 2, np.pi / 2, 1.0, 1.0, (0.0, 0.0, 0.0, 0.0), ops
        )
        assert compute_reduced_purity(psi) == pytest.approx(1.0, abs=1e-10)

    def test_purity_clamped_range(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            psi /= np.linalg.norm(psi)
            assert 0.5 <= compute_reduced_purity(psi) <= 1.0


# ============================================================================
# Beam-Splitter
# ============================================================================


class TestBeamSplitter:
    """Beam-splitter unitaries: unitarity, special cases, tensor structure."""

    @pytest.mark.parametrize("T", [0.0, 0.5, np.pi / 4, np.pi / 2, np.pi])
    def test_unitary(self, T: float) -> None:
        U = bs_unitary(T)
        assert pytest.approx(I_2, abs=1e-12) == U @ U.conj().T
        assert pytest.approx(I_2, abs=1e-12) == U.conj().T @ U

    @pytest.mark.parametrize("T", [0.0, 0.3, np.pi / 4, np.pi / 2, np.pi])
    def test_smoke(self, T: float) -> None:
        assert bs_unitary(T).shape == (2, 2)

    def test_zero_time(self) -> None:
        assert bs_unitary(0.0) == pytest.approx(I_2)

    def test_half_pi(self) -> None:
        expected = np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2)
        assert bs_unitary(np.pi / 2.0) == pytest.approx(expected, abs=1e-12)

    def test_two_qubit_unitary(self) -> None:
        U = two_qubit_bs_unitary(np.pi / 4)
        assert pytest.approx(I_4, abs=1e-12) == U @ U.conj().T
        assert pytest.approx(I_4, abs=1e-12) == U.conj().T @ U

    def test_two_qubit_tensor_structure(self) -> None:
        T = 0.7
        assert two_qubit_bs_unitary(T) == pytest.approx(
            np.kron(bs_unitary(T), bs_unitary(T))
        )


# ============================================================================
# Holding Hamiltonian & Unitary
# ============================================================================


class TestHold:
    """Holding Hamiltonian Hermiticity and unitary evolution."""

    def test_hamiltonian_hermitian(self, ops: dict[str, np.ndarray]) -> None:
        H = build_hold_hamiltonian(1.0, (0.1, 0.2, 0.3, 0.4), ops)
        assert pytest.approx(H.conj().T, abs=1e-12) == H

    def test_unitary(self, ops: dict[str, np.ndarray]) -> None:
        U = hold_unitary(1.0, 1.0, (0.1, 0.0, 0.0, 0.0), ops)
        assert pytest.approx(I_4, abs=1e-12) == U @ U.conj().T

    @pytest.mark.parametrize("T_H", [0.0, 0.5, 2.0])
    def test_matches_exact(self, T_H: float, ops: dict[str, np.ndarray]) -> None:
        theta, alpha = 1.0, (0.1, 0.2, -0.1, 0.3)
        H = build_hold_hamiltonian(theta, alpha, ops)
        assert hold_unitary(T_H, theta, alpha, ops) == pytest.approx(
            expm(-1j * T_H * H), abs=1e-12
        )

    def test_zero_hold_identity(self, ops: dict[str, np.ndarray]) -> None:
        assert hold_unitary(0.0, 1.0, (0.1, 0.0, 0.0, 0.0), ops) == pytest.approx(
            I_4, abs=1e-12
        )

    @pytest.mark.parametrize("T_H", [0.0, 0.5, 1.0, 2.0])
    def test_smoke(self, T_H: float, ops: dict[str, np.ndarray]) -> None:
        assert hold_unitary(T_H, 1.0, (0.1, 0.0, -0.2, 0.3), ops).shape == (4, 4)


# ============================================================================
# Full Circuit Evolution
# ============================================================================


class TestCircuitEvolution:
    """Full MZI circuit: norm preservation, unitarity."""

    @pytest.mark.parametrize("T_H", [0.0, 0.5, 1.0])
    def test_normalisation_preserved(
        self, T_H: float, ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        psi = evolve_full(
            psi0, np.pi / 4, np.pi / 4, T_H, 1.0, (0.0, 0.0, 0.0, 0.0), ops
        )
        assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-12)

    def test_no_hold_no_bs_identity(self, ops: dict[str, np.ndarray]) -> None:
        psi0 = two_qubit_state(0.5, 0.3, 1.2, 0.8)
        psi = evolve_full(psi0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0), ops)
        assert psi == pytest.approx(psi0, abs=1e-12)

    @pytest.mark.parametrize("random_state", [True, False])
    def test_unitarity_of_evolution(
        self, random_state: bool, ops: dict[str, np.ndarray]
    ) -> None:
        rng = np.random.default_rng(42)
        v1 = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        v1 /= np.linalg.norm(v1)
        v2 = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        v2 /= np.linalg.norm(v2)

        if random_state:
            v2 -= np.vdot(v1, v2) * v1
            v2 /= np.linalg.norm(v2)
            psi0_1, psi0_2 = v1, v2
        else:
            psi0_1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
            psi0_2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)

        inner_before = np.vdot(psi0_1, psi0_2)
        alpha = (0.3, -0.1, 0.2, 0.0)
        psi1 = evolve_full(psi0_1, 0.8, 0.6, 1.5, 2.0, alpha, ops)
        psi2 = evolve_full(psi0_2, 0.8, 0.6, 1.5, 2.0, alpha, ops)
        assert np.vdot(psi1, psi2) == pytest.approx(inner_before, abs=1e-12)


# ============================================================================
# Sensitivity Computation
# ============================================================================


class TestSensitivity:
    """Error-propagation sensitivity: SQL, fringe extrema, validation."""

    _alpha_zero = (0.0, 0.0, 0.0, 0.0)

    def test_expectation_variance_consistency(self, ops: dict[str, np.ndarray]) -> None:
        psi = two_qubit_state(0.5, 0.3, 0.8, 1.2)
        exp_val, var_val = compute_expectation_and_variance(psi, ops["Jz_S"])
        exp_direct = np.real(psi.conj() @ ops["Jz_S"] @ psi)
        var_direct = (
            np.real(psi.conj() @ (ops["Jz_S"] @ ops["Jz_S"]) @ psi) - exp_direct**2
        )
        assert exp_val == pytest.approx(exp_direct)
        assert var_val == pytest.approx(max(0.0, var_direct))

    @pytest.mark.parametrize("T_H", [0.5, 1.0, 2.0])
    def test_decoupled_sensitivity_sql(
        self, T_H: float, ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        dtheta = compute_sensitivity(
            psi0, np.pi / 2, np.pi / 2, T_H, 1.0, self._alpha_zero, ops
        )
        assert dtheta == pytest.approx(1.0 / T_H, rel=0.05)

    def test_fringe_extremum_returns_inf(self, ops: dict[str, np.ndarray]) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        assert np.isinf(
            compute_sensitivity(
                psi0, np.pi / 2, np.pi / 2, 1.0, np.pi, self._alpha_zero, ops
            )
        )

    @pytest.mark.parametrize(
        ("T_H", "theta"), itertools.product([0.5, 1.0, 2.0], [0.5, 1.0, 1.5])
    )
    def test_finite_away_from_fringe(
        self, T_H: float, theta: float, ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        dtheta = compute_sensitivity(
            psi0, np.pi / 2, np.pi / 2, T_H, theta, self._alpha_zero, ops
        )
        assert np.isfinite(dtheta) and dtheta > 0

    @pytest.mark.parametrize("T_H", [0.5, 1.0, 2.0])
    def test_get_decoupled_sensitivity_sql(self, T_H: float) -> None:
        assert get_decoupled_sensitivity(T_H, theta_true=1.0) == pytest.approx(
            1.0 / T_H, rel=0.05
        )

    @pytest.mark.parametrize(
        ("theta_true", "T_H"),
        list(itertools.product([0.3, 0.7, 1.0, 1.3, 1.7], [0.5, 1.0, 1.5, 2.0])),
    )
    def test_decoupled_sensitivity_analytical(
        self, theta_true: float, T_H: float, ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        dtheta = compute_sensitivity(
            psi0, np.pi / 2, np.pi / 2, T_H, theta_true, self._alpha_zero, ops
        )
        assert dtheta == pytest.approx(1.0 / T_H, rel=5e-3)

    def test_variance_nonnegative(self, ops: dict[str, np.ndarray]) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            psi /= np.linalg.norm(psi)
            assert validate_variance_positive(psi, ops["Jz_S"]) is True

    # --- Validation helpers ---
    def test_validate_sensitivity_reasonable(self) -> None:
        assert validate_sensitivity_reasonable() is True

    def test_validate_operators_raises_on_bad(self, ops: dict[str, np.ndarray]) -> None:
        ops["Jz_S"] = np.zeros((4, 4))
        with pytest.raises(AssertionError):
            validate_operators(ops)

    def test_validate_variance_positive_passes(
        self, ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        psi = evolve_full(psi0, np.pi / 2, np.pi / 2, 1.0, 1.0, self._alpha_zero, ops)
        assert validate_variance_positive(psi, ops["Jz_S"]) is True

    def test_validate_derivative_stability_passes(
        self, ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        assert (
            validate_derivative_stability(
                psi0, np.pi / 2, np.pi / 2, 1.0, 1.0, self._alpha_zero, ops
            )
            is True
        )

    def test_validate_derivative_stability_at_fringe(
        self, ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        assert (
            validate_derivative_stability(
                psi0, np.pi / 2, np.pi / 2, 1.0, np.pi, self._alpha_zero, ops
            )
            is True
        )


# ============================================================================
# Objective Function
# ============================================================================


class TestObjective:
    """Nelder–Mead objective: validity, SQL match, bounds, smoothness."""

    def test_valid_params_finite(self, ops: dict[str, np.ndarray]) -> None:
        val = sensitivity_objective(_default_params(), theta_true=1.0, ops=ops)
        assert np.isfinite(val) and val > 0

    def test_matches_sql(self, ops: dict[str, np.ndarray]) -> None:
        assert sensitivity_objective(
            _default_params(), theta_true=1.0, ops=ops
        ) == pytest.approx(1.0, rel=0.05)

    def test_penalty_out_of_bounds_theta(self, ops: dict[str, np.ndarray]) -> None:
        params = _default_params().copy()
        params[0] = 4.0  # theta_S out of [0, π]
        assert sensitivity_objective(params, theta_true=1.0, ops=ops) > 1e9

    def test_penalty_out_of_bounds_alpha(self, ops: dict[str, np.ndarray]) -> None:
        params = _default_params().copy()
        params[7] = 5.0  # alpha_xx out of [-2, 2]
        assert sensitivity_objective(params, theta_true=1.0, ops=ops) > 1e9

    def test_smoothness(self, ops: dict[str, np.ndarray]) -> None:
        base = _default_params()
        val_base = sensitivity_objective(base, theta_true=1.0, ops=ops)
        for i in range(11):
            perturbed = base.copy()
            perturbed[i] += 1e-6
            assert (
                abs(
                    sensitivity_objective(perturbed, theta_true=1.0, ops=ops) - val_base
                )
                < 1.0
            )


# ============================================================================
# Optimisation
# ============================================================================


class TestOptimisation:
    """Nelder–Mead optimisation interface, result dataclasses, convergence."""

    def test_run_returns_result_type(self, ops: dict[str, np.ndarray]) -> None:
        result = run_optimisation(
            theta_true=1.0, ops=ops, x0=_default_params(), maxiter=10
        )
        assert isinstance(result, OptimisationResult)
        assert result.theta_true == 1.0

    def test_run_returns_valid_params(self, ops: dict[str, np.ndarray]) -> None:
        result = run_optimisation(
            theta_true=1.0, ops=ops, x0=_default_params(), maxiter=10
        )
        assert result.params_opt.shape == (11,)
        assert not np.isnan(result.delta_theta_opt)
        assert 0.5 <= result.purity_S <= 1.0

    def test_result_dataclass(self) -> None:
        r = OptimisationResult(0.5, np.zeros(11), 1.0, True, 100, "OK")
        assert r.delta_theta_opt == 0.5
        assert r.success is True
        r2 = OptimisationResult(
            0.3,
            np.ones(11),
            2.0,
            False,
            50,
            "test",
            expectation_Jz=0.25,
            variance_Jz=0.01,
            purity_S=0.75,
        )
        assert r2.expectation_Jz == pytest.approx(0.25)
        assert r2.purity_S == pytest.approx(0.75)

    def test_theta_scan_result(self) -> None:
        r = ThetaScanResult(
            results=[],
            theta_values=np.array([0.5, 1.0]),
            best_per_theta=np.array([0.6, 0.3]),
            all_results={},
        )
        assert len(r.theta_values) == 2
        assert r.best_per_theta[0] == pytest.approx(0.6)

    @pytest.mark.slow
    def test_explores_t_h(self, ops: dict[str, np.ndarray]) -> None:
        result = run_optimisation(
            theta_true=1.0, ops=ops, x0=_default_params(), maxiter=200
        )
        T_H_opt = result.params_opt[6]
        assert T_H_opt > 1.5
        assert result.delta_theta_opt == pytest.approx(1.0 / T_H_opt, rel=0.15)

    # --- Convergence metric ---
    def test_convergence_fewer_than_two(self) -> None:
        r = OptimisationResult(0.5, np.zeros(11), 1.0, True, 10, "ok")
        assert compute_convergence_metric([r]) == 0.0

    def test_convergence_all_inf(self) -> None:
        results = [
            OptimisationResult(float("inf"), np.zeros(11), 1.0, True, 10, "ok")
            for _ in range(2)
        ]
        assert compute_convergence_metric(results) == 0.0

    def test_convergence_small_spread(self) -> None:
        results = [
            OptimisationResult(v, np.zeros(11), 1.0, True, 10, "ok")
            for v in [0.51, 0.52, 0.50, 0.53]
        ]
        metric = compute_convergence_metric(results)
        assert 0.0 < metric < 0.10


# ============================================================================
# Bounds
# ============================================================================


class TestBounds:
    """Default bounds, custom bounds, random initial params, history."""

    @pytest.mark.parametrize("key", ["theta", "phi", "T_BS", "T_H", "alpha"])
    def test_default_bounds_structure(self, key: str) -> None:
        bounds = get_default_bounds()
        assert key in bounds
        assert isinstance(bounds[key], tuple) and len(bounds[key]) == 2

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("theta", (0.0, np.pi)),
            ("phi", (0.0, 2.0 * np.pi)),
            ("T_BS", (0.0, np.pi)),
            ("T_H", (0.0, 5.0)),
            ("alpha", (-2.0, 2.0)),
        ],
    )
    def test_default_bounds_values(
        self, key: str, expected: tuple[float, float]
    ) -> None:
        assert get_default_bounds()[key] == expected

    def test_random_initial_params_shape(self) -> None:
        assert random_initial_params(np.random.default_rng(42)).shape == (11,)

    @pytest.mark.parametrize("seed", range(5))
    def test_random_initial_params_within_bounds(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        params = random_initial_params(rng)
        lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ub = np.array([np.pi, 2 * np.pi, np.pi, 2 * np.pi, np.pi, np.pi, 5.0])
        assert np.all((lb <= params[:7]) & (params[:7] <= ub))
        assert np.all(np.abs(params[7:]) <= 2.0)

    def test_random_initial_params_respects_custom_bounds(self) -> None:
        rng = np.random.default_rng(42)
        custom_bounds = get_default_bounds()
        custom_bounds["T_H"] = (0.0, 20.0)
        for _ in range(50):
            params = random_initial_params(rng, custom_bounds)
            assert params.shape == (11,)
            assert 0.0 <= params[6] <= 20.0

    # --- History ---
    def test_optimisation_result_history(self) -> None:
        r = OptimisationResult(0.5, np.zeros(11), 1.0, True, 100, "OK")
        assert hasattr(r, "history")
        assert r.history == []

    def test_optimisation_result_history_settable(self) -> None:
        r = OptimisationResult(
            0.5, np.zeros(11), 1.0, True, 100, "OK", history=[1.0, 0.8, 0.6, 0.5]
        )
        assert r.history == [1.0, 0.8, 0.6, 0.5]

    def test_track_history(self, ops: dict[str, np.ndarray]) -> None:
        result_no_track = run_optimisation(
            theta_true=1.0,
            ops=ops,
            x0=_default_params(),
            maxiter=20,
            track_history=False,
        )
        assert result_no_track.history == []
        result_with_track = run_optimisation(
            theta_true=1.0,
            ops=ops,
            x0=_default_params(),
            maxiter=20,
            track_history=True,
        )
        assert len(result_with_track.history) > 0
        assert all(np.isfinite(v) and v > 0 for v in result_with_track.history)


# ============================================================================
# α-Coefficient Scans
# ============================================================================


class TestAlphaScans:
    """Grid scan and random search over α coefficients."""

    def test_single_parameter_xx(self) -> None:
        result = scan_alpha_single_parameter(
            "xx", alpha_min=-0.5, alpha_max=0.5, n_points=5
        )
        assert isinstance(result, AlphaSingleScanResult)
        assert result.alpha_name == "xx"
        assert result.alpha_values.shape == (5,)
        assert np.all(np.isfinite(result.delta_theta_values))

    @pytest.mark.parametrize("name", ["xx", "xz", "zx", "zz"])
    def test_single_parameter_all_names(self, name: str) -> None:
        result = scan_alpha_single_parameter(
            name, alpha_min=-0.1, alpha_max=0.1, n_points=3
        )
        assert result.alpha_name == name

    def test_single_parameter_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError):
            scan_alpha_single_parameter("invalid")

    def test_single_parameter_sql_at_zero(self) -> None:
        result = scan_alpha_single_parameter(
            "xx", alpha_min=-0.2, alpha_max=0.2, n_points=5, T_H=1.0, theta_true=1.0
        )
        idx_mid = 2
        assert result.alpha_values[idx_mid] == 0.0
        assert result.delta_theta_values[idx_mid] == pytest.approx(1.0, rel=0.1)

    def test_random_search_basic(self) -> None:
        result = random_search_alpha(n_samples=10, seed=42)
        assert isinstance(result, AlphaRandomSearchResult)
        assert result.alpha_samples.shape == (10, 4)
        assert len(result.delta_theta_values) == 10
        assert np.isfinite(result.best_delta_theta)

    def test_random_search_bounds(self) -> None:
        result = random_search_alpha(
            n_samples=50, alpha_min=-1.0, alpha_max=1.0, seed=42
        )
        for i in range(50):
            for j in range(4):
                assert -1.0 <= result.alpha_samples[i, j] <= 1.0

    def test_random_search_reproducible(self) -> None:
        result1 = random_search_alpha(n_samples=20, seed=123)
        result2 = random_search_alpha(n_samples=20, seed=123)
        assert result1.alpha_samples == pytest.approx(result2.alpha_samples)
        assert result1.delta_theta_values == pytest.approx(result2.delta_theta_values)

    @pytest.mark.slow
    def test_alpha_nonzero_should_never_beat_sql_when_measuring_jz_s(self) -> None:
        T_H, sql = 1.0, 1.0
        for name in ["xx", "xz", "zx", "zz"]:
            result = scan_alpha_single_parameter(
                name, alpha_min=-1.5, alpha_max=1.5, n_points=11, T_H=T_H
            )
            finite = np.isfinite(result.delta_theta_values)
            assert np.min(result.delta_theta_values[finite]) >= sql - 1e-8

        result_rand = random_search_alpha(n_samples=100, T_H=T_H, seed=42)
        finite = np.isfinite(result_rand.delta_theta_values)
        assert np.min(result_rand.delta_theta_values[finite]) >= sql - 1e-8

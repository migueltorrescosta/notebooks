from __future__ import annotations

import itertools
from pathlib import Path  # noqa: TC003 — used at runtime via tmp_path fixture

import numpy as np
import pytest
from scipy.linalg import expm

from src.analysis.ancilla_optimization import (
    I_2,
    AlphaRandomSearchResult,
    AlphaReoptScanResult,
    AlphaSingleScanResult,
    CovarianceAnalysisResult,
    DecoupledBaselineResult,
    InteractionRobustnessResult,
    OmegaScanResult,
    OptimisationResult,
    bs_unitary,
    build_hold_hamiltonian,
    build_interaction_hamiltonian,
    build_joint_operator,
    build_two_qubit_operators,
    compute_convergence_metric,
    compute_covariance,
    compute_expectation_and_variance,
    compute_interaction_robustness,
    compute_reduced_purity,
    compute_sensitivity,
    evolve_full,
    get_decoupled_sensitivity,
    get_default_bounds,
    hold_unitary_two_qubit,
    random_initial_params,
    random_search_alpha,
    run_optimisation,
    scan_alpha_single_parameter,
    scan_alpha_with_reoptimisation,
    sensitivity_objective,
    single_qubit_state,
    two_qubit_bs_unitary,
    two_qubit_state,
    validate_derivative_stability,
    validate_operators,
    validate_sensitivity_reasonable,
    validate_variance_positive,
)
from src.utils.constants import I_4


def _make_default_params() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def make_ops() -> dict[str, np.ndarray]:
    return build_two_qubit_operators()


class TestOperatorConstruction:
    def test_given_two_qubit_operators_then_all_have_shape_4x4(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        for op in make_ops.values():
            assert op.shape == (4, 4)

    def test_given_two_qubit_operators_then_all_are_hermitian(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        for op in make_ops.values():
            assert op == pytest.approx(op.conj().T, abs=1e-12)

    def test_given_jz_operators_then_are_diagonal(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        for name in ["Jz_S", "Jz_A"]:
            assert make_ops[name] == pytest.approx(np.diag(np.diag(make_ops[name])))

    def test_given_jz_operators_then_eigenvalues_are_correct(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        for name in ["Jz_S", "Jz_A"]:
            assert sorted(np.linalg.eigvalsh(make_ops[name])) == pytest.approx(
                [-0.5, -0.5, 0.5, 0.5]
            )

    def test_commutation_jz_jx(self, make_ops: dict[str, np.ndarray]) -> None:
        comm_S = (
            make_ops["Jz_S"] @ make_ops["Jx_S"] - make_ops["Jx_S"] @ make_ops["Jz_S"]
        )
        assert comm_S == pytest.approx(1j * make_ops["Jy_S"], abs=1e-12)
        comm_A = (
            make_ops["Jz_A"] @ make_ops["Jx_A"] - make_ops["Jx_A"] @ make_ops["Jz_A"]
        )
        assert comm_A == pytest.approx(1j * make_ops["Jy_A"], abs=1e-12)

    def test_given_zero_coefficients_then_hamiltonian_is_zero(self) -> None:
        assert build_interaction_hamiltonian((0.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)

    @pytest.mark.parametrize("seed", range(10), ids=[f"seed_{s}" for s in range(10)])
    def test_given_random_coefficients_then_hamiltonian_is_hermitian(
        self, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        alpha = tuple(rng.uniform(-2, 2, size=4))
        H = build_interaction_hamiltonian(alpha)
        assert pytest.approx(H.conj().T, abs=1e-12) == H


class TestJointOperator:
    def test_joint_operator_is_hermitian(self, make_ops: dict[str, np.ndarray]) -> None:
        M = build_joint_operator(make_ops)
        assert pytest.approx(M.conj().T, abs=1e-12) == M

    def test_joint_operator_eigenvalues(self, make_ops: dict[str, np.ndarray]) -> None:
        M = build_joint_operator(make_ops)
        assert sorted(np.linalg.eigvalsh(M)) == pytest.approx(
            [-1.0, 0.0, 0.0, 1.0], abs=1e-12
        )

    def test_joint_operator_equals_sum(self, make_ops: dict[str, np.ndarray]) -> None:
        M = build_joint_operator(make_ops)
        assert pytest.approx(make_ops["Jz_S"] + make_ops["Jz_A"], abs=1e-12) == M


class TestCovariance:
    def test_product_state_zero_covariance(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        cov = compute_covariance(psi, make_ops)
        assert cov == pytest.approx(0.0, abs=1e-12)

    def test_bell_state_nonzero_covariance(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi = np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
        cov = compute_covariance(psi, make_ops)
        assert cov == pytest.approx(-0.25, abs=1e-12)

    def test_variance_identity(self, make_ops: dict[str, np.ndarray]) -> None:
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        psi /= np.linalg.norm(psi)

        M = build_joint_operator(make_ops)
        _, var_M = compute_expectation_and_variance(psi, M)
        _, var_S = compute_expectation_and_variance(psi, make_ops["Jz_S"])
        _, var_A = compute_expectation_and_variance(psi, make_ops["Jz_A"])
        cov = compute_covariance(psi, make_ops)

        assert var_M == pytest.approx(var_S + var_A + 2.0 * cov, abs=1e-12)


class TestStatePreparation:
    @pytest.mark.parametrize(
        ("theta", "phi"), [(0.0, 0.0), (np.pi, 0.0), (np.pi / 2, np.pi)]
    )
    def test_single_qubit_normalised(self, theta: float, phi: float) -> None:
        assert np.linalg.norm(single_qubit_state(theta, phi)) == pytest.approx(1.0)

    def test_given_zero_theta_then_state_is_up(self) -> None:
        assert single_qubit_state(0.0, 0.0) == pytest.approx(
            np.array([1.0, 0.0], dtype=complex)
        )

    def test_given_pi_theta_then_state_is_down(self) -> None:
        assert single_qubit_state(np.pi, 0.0) == pytest.approx(
            np.array([0.0, 1.0], dtype=complex)
        )

    def test_two_qubit_product_structure(self) -> None:
        psi = two_qubit_state(0.3, 0.7, 1.2, 2.5)
        expected = np.kron(single_qubit_state(0.3, 0.7), single_qubit_state(1.2, 2.5))
        assert psi == pytest.approx(expected)
        assert np.linalg.norm(psi) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("psi", "expected"),
        [
            (np.array([1.0, 0.0, 0.0, 0.0], dtype=complex), 1.0),
            (np.array([0.0, 1.0, 0.0, 0.0], dtype=complex), 1.0),
            (np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2), 0.5),
            (np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2), 0.5),
        ],
        ids=["00", "01", "Phi+", "Psi+"],
    )
    def test_reduced_purity_known(self, psi: np.ndarray, expected: float) -> None:
        assert compute_reduced_purity(psi) == pytest.approx(expected, abs=1e-12)

    def test_product_state_general(self) -> None:
        assert compute_reduced_purity(
            two_qubit_state(0.7, 1.2, 0.3, 2.8)
        ) == pytest.approx(1.0, abs=1e-12)

    def test_purity_through_circuit(self, make_ops: dict[str, np.ndarray]) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        psi = evolve_full(
            psi0, np.pi / 2, np.pi / 2, 1.0, 1.0, (0.0, 0.0, 0.0, 0.0), make_ops
        )
        assert compute_reduced_purity(psi) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("seed", range(20), ids=[f"seed_{s}" for s in range(20)])
    def test_given_random_state_then_purity_in_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        psi /= np.linalg.norm(psi)
        assert 0.5 <= compute_reduced_purity(psi) <= 1.0


class TestBeamSplitter:
    @pytest.mark.parametrize("T_BS", [0.0, 0.5, np.pi / 4, np.pi / 2, np.pi])
    def test_given_bs_unitary_then_is_unitary(self, T_BS: float) -> None:
        U = bs_unitary(T_BS)
        assert pytest.approx(I_2, abs=1e-12) == U @ U.conj().T
        assert pytest.approx(I_2, abs=1e-12) == U.conj().T @ U

    @pytest.mark.parametrize("T_BS", [0.0, 0.3, np.pi / 4, np.pi / 2, np.pi])
    def test_given_bs_unitary_then_shape_is_2x2(self, T_BS: float) -> None:
        assert bs_unitary(T_BS).shape == (2, 2)

    def test_given_zero_angle_then_identity(self) -> None:
        assert bs_unitary(0.0) == pytest.approx(I_2)

    def test_given_half_pi_angle_then_known_matrix(self) -> None:
        expected = np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2)
        assert bs_unitary(np.pi / 2.0) == pytest.approx(expected, abs=1e-12)

    def test_two_qubit_unitary(self) -> None:
        U = two_qubit_bs_unitary(np.pi / 4)
        assert pytest.approx(I_4, abs=1e-12) == U @ U.conj().T
        assert pytest.approx(I_4, abs=1e-12) == U.conj().T @ U

    def test_two_qubit_tensor_structure(self) -> None:
        T_BS = 0.7
        assert two_qubit_bs_unitary(T_BS) == pytest.approx(
            np.kron(bs_unitary(T_BS), bs_unitary(T_BS))
        )


class TestHold:
    def test_hamiltonian_hermitian(self, make_ops: dict[str, np.ndarray]) -> None:
        H = build_hold_hamiltonian(1.0, (0.1, 0.2, 0.3, 0.4), make_ops)
        assert pytest.approx(H, abs=1e-12) == H.conj().T

    def test_given_hold_unitary_then_is_unitary(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        U = hold_unitary_two_qubit(1.0, 1.0, (0.1, 0.0, 0.0, 0.0), make_ops)
        assert pytest.approx(I_4, abs=1e-12) == U @ U.conj().T

    @pytest.mark.parametrize("T_hold", [0.0, 0.5, 2.0])
    def test_given_hold_unitary_then_matches_scipy_exponential(
        self, T_hold: float, make_ops: dict[str, np.ndarray]
    ) -> None:
        omega, alpha = 1.0, (0.1, 0.2, -0.1, 0.3)
        H = build_hold_hamiltonian(omega, alpha, make_ops)
        assert hold_unitary_two_qubit(T_hold, omega, alpha, make_ops) == pytest.approx(
            expm(-1j * T_hold * H), abs=1e-12
        )

    def test_zero_hold_identity(self, make_ops: dict[str, np.ndarray]) -> None:
        assert hold_unitary_two_qubit(
            0.0, 1.0, (0.1, 0.0, 0.0, 0.0), make_ops
        ) == pytest.approx(I_4, abs=1e-12)

    @pytest.mark.parametrize("T_hold", [0.0, 0.5, 1.0, 2.0])
    def test_given_hold_unitary_then_shape_is_4x4(
        self, T_hold: float, make_ops: dict[str, np.ndarray]
    ) -> None:
        assert hold_unitary_two_qubit(
            T_hold, 1.0, (0.1, 0.0, -0.2, 0.3), make_ops
        ).shape == (4, 4)


class TestCircuitEvolution:
    @pytest.mark.parametrize("T_hold", [0.0, 0.5, 1.0])
    def test_normalisation_preserved(
        self, T_hold: float, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        psi = evolve_full(
            psi0, np.pi / 4, np.pi / 4, T_hold, 1.0, (0.0, 0.0, 0.0, 0.0), make_ops
        )
        assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-12)

    def test_no_hold_no_bs_identity(self, make_ops: dict[str, np.ndarray]) -> None:
        psi0 = two_qubit_state(0.5, 0.3, 1.2, 0.8)
        psi = evolve_full(psi0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0), make_ops)
        assert psi == pytest.approx(psi0, abs=1e-12)

    @pytest.mark.parametrize("random_state", [True, False], ids=["random", "basis"])
    def test_unitarity_of_evolution(
        self, random_state: bool, make_ops: dict[str, np.ndarray]
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
        psi1 = evolve_full(psi0_1, 0.8, 0.6, 1.5, 2.0, alpha, make_ops)
        psi2 = evolve_full(psi0_2, 0.8, 0.6, 1.5, 2.0, alpha, make_ops)
        assert np.vdot(psi1, psi2) == pytest.approx(inner_before, abs=1e-12)


class TestSensitivity:
    _alpha_zero = (0.0, 0.0, 0.0, 0.0)

    def test_expectation_variance_consistency(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi = two_qubit_state(0.5, 0.3, 0.8, 1.2)
        exp_val, var_val = compute_expectation_and_variance(psi, make_ops["Jz_S"])
        exp_direct = np.real(psi.conj() @ make_ops["Jz_S"] @ psi)
        var_direct = (
            np.real(psi.conj() @ (make_ops["Jz_S"] @ make_ops["Jz_S"]) @ psi)
            - exp_direct**2
        )
        assert exp_val == pytest.approx(exp_direct)
        assert var_val == pytest.approx(max(0.0, var_direct))

    @pytest.mark.parametrize("T_hold", [0.5, 1.0, 2.0])
    def test_decoupled_sensitivity_sql(
        self, T_hold: float, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        domega = compute_sensitivity(
            psi0, np.pi / 2, np.pi / 2, T_hold, 1.0, self._alpha_zero, make_ops
        )
        assert domega == pytest.approx(1.0 / T_hold, rel=0.05)

    def test_fringe_extremum_returns_inf(self, make_ops: dict[str, np.ndarray]) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        assert np.isinf(
            compute_sensitivity(
                psi0, np.pi / 2, np.pi / 2, 1.0, np.pi, self._alpha_zero, make_ops
            )
        )

    @pytest.mark.parametrize(
        ("T_hold", "omega_true"), itertools.product([0.5, 1.0, 2.0], [0.5, 1.0, 1.5])
    )
    def test_finite_away_from_fringe(
        self, T_hold: float, omega_true: float, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        domega = compute_sensitivity(
            psi0, np.pi / 2, np.pi / 2, T_hold, omega_true, self._alpha_zero, make_ops
        )
        assert np.isfinite(domega) and domega > 0

    @pytest.mark.parametrize("T_hold", [0.5, 1.0, 2.0])
    def test_get_decoupled_sensitivity_sql(self, T_hold: float) -> None:
        assert get_decoupled_sensitivity(T_hold, omega_true=1.0) == pytest.approx(
            1.0 / T_hold, rel=0.05
        )

    @pytest.mark.parametrize(
        ("omega_true", "T_hold"),
        list(itertools.product([0.3, 0.7, 1.0, 1.3, 1.7], [0.5, 1.0, 1.5, 2.0])),
    )
    def test_decoupled_sensitivity_analytical(
        self, omega_true: float, T_hold: float, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        domega = compute_sensitivity(
            psi0, np.pi / 2, np.pi / 2, T_hold, omega_true, self._alpha_zero, make_ops
        )
        assert domega == pytest.approx(1.0 / T_hold, rel=5e-3)

    @pytest.mark.parametrize("seed", range(20), ids=[f"seed_{s}" for s in range(20)])
    def test_variance_nonnegative(
        self, seed: int, make_ops: dict[str, np.ndarray]
    ) -> None:
        rng = np.random.default_rng(seed)
        psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        psi /= np.linalg.norm(psi)
        assert validate_variance_positive(psi, make_ops["Jz_S"]) is True

    def test_validate_sensitivity_reasonable(self) -> None:
        assert validate_sensitivity_reasonable() is True

    def test_validate_operators_raises_on_bad(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        make_ops["Jz_S"] = np.zeros((4, 4))
        with pytest.raises(AssertionError):
            validate_operators(make_ops)

    def test_validate_variance_positive_passes(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        psi = evolve_full(
            psi0, np.pi / 2, np.pi / 2, 1.0, 1.0, self._alpha_zero, make_ops
        )
        assert validate_variance_positive(psi, make_ops["Jz_S"]) is True

    def test_validate_derivative_stability_passes(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        assert (
            validate_derivative_stability(
                psi0, np.pi / 2, np.pi / 2, 1.0, 1.0, self._alpha_zero, make_ops
            )
            is True
        )

    def test_validate_derivative_stability_at_fringe(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        assert (
            validate_derivative_stability(
                psi0, np.pi / 2, np.pi / 2, 1.0, np.pi, self._alpha_zero, make_ops
            )
            is True
        )

    @pytest.mark.parametrize("T_hold", [0.5, 1.0, 2.0])
    def test_joint_measurement_sensitivity_sql(
        self, T_hold: float, make_ops: dict[str, np.ndarray]
    ) -> None:
        """Joint measurement must achieve Δθ = 1/T_hold at α=0, matching SQL."""
        M_op = build_joint_operator(make_ops)
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        T_BS = np.pi / 2.0
        omega_true = 1.0
        alpha = (0.0, 0.0, 0.0, 0.0)
        domega = compute_sensitivity(
            psi0, T_BS, T_BS, T_hold, omega_true, alpha, make_ops, meas_op=M_op
        )
        assert domega == pytest.approx(1.0 / T_hold, rel=0.05), (
            f"Joint measurement Δθ={domega:.6f} for T_hold={T_hold},"
            f" expected SQL={1.0 / T_hold:.6f}"
        )


class TestObjective:
    def test_valid_params_finite(self, make_ops: dict[str, np.ndarray]) -> None:
        val = sensitivity_objective(
            _make_default_params(), omega_true=1.0, ops=make_ops
        )
        assert np.isfinite(val) and val > 0

    def test_matches_sql(self, make_ops: dict[str, np.ndarray]) -> None:
        assert sensitivity_objective(
            _make_default_params(), omega_true=1.0, ops=make_ops
        ) == pytest.approx(1.0, rel=0.05)

    def test_penalty_out_of_bounds_theta(self, make_ops: dict[str, np.ndarray]) -> None:
        params = _make_default_params().copy()
        params[0] = 4.0
        assert sensitivity_objective(params, omega_true=1.0, ops=make_ops) > 1e9

    def test_penalty_out_of_bounds_alpha(self, make_ops: dict[str, np.ndarray]) -> None:
        params = _make_default_params().copy()
        params[7] = 5.0
        assert sensitivity_objective(params, omega_true=1.0, ops=make_ops) > 1e9

    @pytest.mark.parametrize("idx", range(11), ids=[f"param_{i}" for i in range(11)])
    def test_given_small_perturbation_then_objective_changes_smoothly(
        self, idx: int, make_ops: dict[str, np.ndarray]
    ) -> None:
        base = _make_default_params()
        val_base = sensitivity_objective(base, omega_true=1.0, ops=make_ops)
        perturbed = base.copy()
        perturbed[idx] += 1e-6
        assert (
            abs(
                sensitivity_objective(perturbed, omega_true=1.0, ops=make_ops)
                - val_base
            )
            < 1.0
        )


class TestOptimisation:
    def test_run_returns_result_type(self, make_ops: dict[str, np.ndarray]) -> None:
        result = run_optimisation(
            omega_true=1.0, ops=make_ops, x0=_make_default_params(), maxiter=10
        )
        assert isinstance(result, OptimisationResult)
        assert result.omega_true == 1.0

    def test_run_returns_valid_params(self, make_ops: dict[str, np.ndarray]) -> None:
        result = run_optimisation(
            omega_true=1.0, ops=make_ops, x0=_make_default_params(), maxiter=10
        )
        assert result.params_opt.shape == (11,)
        assert not np.isnan(result.delta_omega_opt)
        assert 0.5 <= result.purity_S <= 1.0

    def test_result_dataclass(self) -> None:
        r = OptimisationResult(
            delta_omega_opt=0.5,
            params_opt=np.zeros(11),
            omega_true=1.0,
            success=True,
            nfev=100,
            message="OK",
            meas_label="S-only",
        )
        assert r.delta_omega_opt == 0.5
        assert r.success is True
        r2 = OptimisationResult(
            delta_omega_opt=0.3,
            params_opt=np.ones(11),
            omega_true=2.0,
            success=False,
            nfev=50,
            message="test",
            expectation_Jz=0.25,
            variance_Jz=0.01,
            purity_S=0.75,
            meas_label="S-only",
        )
        assert r2.expectation_Jz == pytest.approx(0.25)
        assert r2.purity_S == pytest.approx(0.75)

    def test_omega_scan_result(self) -> None:
        r = OmegaScanResult(
            results=[],
            omega_values=np.array([0.5, 1.0]),
            best_per_omega=np.array([0.6, 0.3]),
            all_results={},
        )
        assert len(r.omega_values) == 2
        assert r.best_per_omega[0] == pytest.approx(0.6)

    @pytest.mark.slow
    def test_explores_t_h(self, make_ops: dict[str, np.ndarray]) -> None:
        result = run_optimisation(
            omega_true=1.0, ops=make_ops, x0=_make_default_params(), maxiter=200
        )
        T_hold_opt = result.params_opt[6]
        assert T_hold_opt > 1.5
        assert result.delta_omega_opt == pytest.approx(1.0 / T_hold_opt, rel=0.15)

    def test_convergence_fewer_than_two(self) -> None:
        r = OptimisationResult(
            delta_omega_opt=0.5,
            params_opt=np.zeros(11),
            omega_true=1.0,
            success=True,
            nfev=10,
            message="ok",
            meas_label="S-only",
        )
        assert compute_convergence_metric([r]) == 0.0

    def test_convergence_all_inf(self) -> None:
        results = [
            OptimisationResult(
                delta_omega_opt=float("inf"),
                params_opt=np.zeros(11),
                omega_true=1.0,
                success=True,
                nfev=10,
                message="ok",
                meas_label="S-only",
            )
            for _ in range(2)
        ]
        assert compute_convergence_metric(results) == 0.0

    def test_convergence_small_spread(self) -> None:
        results = [
            OptimisationResult(
                delta_omega_opt=v,
                params_opt=np.zeros(11),
                omega_true=1.0,
                success=True,
                nfev=10,
                message="ok",
                meas_label="S-only",
            )
            for v in [0.51, 0.52, 0.50, 0.53]
        ]
        metric = compute_convergence_metric(results)
        assert 0.0 < metric < 0.10


class TestBounds:
    @pytest.mark.parametrize("key", ["bloch_theta", "phi", "T_BS", "T_hold", "alpha"])
    def test_default_bounds_structure(self, key: str) -> None:
        bounds = get_default_bounds()
        assert key in bounds
        assert isinstance(bounds[key], tuple) and len(bounds[key]) == 2

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("bloch_theta", (0.0, np.pi)),
            ("phi", (0.0, 2.0 * np.pi)),
            ("T_BS", (0.0, np.pi)),
            ("T_hold", (0.0, 5.0)),
            ("alpha", (-2.0, 2.0)),
        ],
    )
    def test_default_bounds_values(
        self, key: str, expected: tuple[float, float]
    ) -> None:
        assert get_default_bounds()[key] == expected

    def test_random_initial_params_shape(self) -> None:
        assert random_initial_params(np.random.default_rng(42)).shape == (11,)

    @pytest.mark.parametrize("seed", range(5), ids=[f"seed_{s}" for s in range(5)])
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
        custom_bounds["T_hold"] = (0.0, 20.0)
        for _ in range(50):
            params = random_initial_params(rng, custom_bounds)
            assert params.shape == (11,)
            assert 0.0 <= params[6] <= 20.0

    def test_optimisation_result_history(self) -> None:
        r = OptimisationResult(
            delta_omega_opt=0.5,
            params_opt=np.zeros(11),
            omega_true=1.0,
            success=True,
            nfev=100,
            message="OK",
            meas_label="S-only",
        )
        assert hasattr(r, "history")
        assert r.history == []

    def test_optimisation_result_history_settable(self) -> None:
        r = OptimisationResult(
            0.5,
            np.zeros(11),
            1.0,
            True,
            100,
            "OK",
            meas_label="S-only",
            history=[1.0, 0.8, 0.6, 0.5],
        )
        assert r.history == [1.0, 0.8, 0.6, 0.5]

    def test_track_history(self, make_ops: dict[str, np.ndarray]) -> None:
        result_no_track = run_optimisation(
            omega_true=1.0,
            ops=make_ops,
            x0=_make_default_params(),
            maxiter=20,
            track_history=False,
        )
        assert result_no_track.history == []
        result_with_track = run_optimisation(
            omega_true=1.0,
            ops=make_ops,
            x0=_make_default_params(),
            maxiter=20,
            track_history=True,
        )
        assert len(result_with_track.history) > 0
        assert all(np.isfinite(v) and v > 0 for v in result_with_track.history)


class TestAlphaScans:
    def test_single_parameter_xx(self) -> None:
        result = scan_alpha_single_parameter(
            "xx", alpha_min=-0.5, alpha_max=0.5, n_points=5
        )
        assert isinstance(result, AlphaSingleScanResult)
        assert result.alpha_name == "xx"
        assert result.alpha_values.shape == (5,)
        assert np.all(np.isfinite(result.delta_omega_values))

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
            "xx", alpha_min=-0.2, alpha_max=0.2, n_points=5, T_hold=1.0, omega_true=1.0
        )
        idx_mid = 2
        assert result.alpha_values[idx_mid] == 0.0
        assert result.delta_omega_values[idx_mid] == pytest.approx(1.0, rel=0.1)

    def test_random_search_basic(self) -> None:
        result = random_search_alpha(n_samples=10, seed=42)
        assert isinstance(result, AlphaRandomSearchResult)
        assert result.alpha_samples.shape == (10, 4)
        assert len(result.delta_omega_values) == 10
        assert np.isfinite(result.best_delta_omega)

    @pytest.mark.parametrize("seed", range(50), ids=[f"seed_{s}" for s in range(50)])
    def test_random_search_bounds(self, seed: int) -> None:
        result = random_search_alpha(
            n_samples=1, alpha_min=-1.0, alpha_max=1.0, seed=seed
        )
        assert np.all((result.alpha_samples >= -1.0) & (result.alpha_samples <= 1.0))

    def test_random_search_reproducible(self) -> None:
        result1 = random_search_alpha(n_samples=20, seed=123)
        result2 = random_search_alpha(n_samples=20, seed=123)
        assert result1.alpha_samples == pytest.approx(result2.alpha_samples)
        assert result1.delta_omega_values == pytest.approx(result2.delta_omega_values)

    @pytest.mark.parametrize("name", ["xx", "xz", "zx", "zz"])
    @pytest.mark.slow
    def test_alpha_nonzero_does_not_beat_sql_when_measuring_jz_s(
        self, name: str
    ) -> None:
        T_hold, sql = 1.0, 1.0
        result = scan_alpha_single_parameter(
            name, alpha_min=-1.5, alpha_max=1.5, n_points=11, T_hold=T_hold
        )
        finite = np.isfinite(result.delta_omega_values)
        assert np.min(result.delta_omega_values[finite]) >= sql - 1e-8

    @pytest.mark.slow
    def test_random_search_does_not_beat_sql(self) -> None:
        result = random_search_alpha(n_samples=100, T_hold=1.0, seed=42)
        finite = np.isfinite(result.delta_omega_values)
        assert np.min(result.delta_omega_values[finite]) >= 1.0 - 1e-8

    def test_alpha_reopt_scan_result_dataclass(self) -> None:
        r = AlphaReoptScanResult(
            alpha_values=np.array([-1.0, 0.0, 1.0]),
            delta_omega_joint=np.array([0.5, 0.6, 0.7]),
            delta_omega_sonly=np.array([0.8, 0.9, 1.0]),
            best_params_joint=[np.zeros(11), np.ones(11), np.ones(11) * 2],
            best_params_sonly=[np.zeros(11), np.ones(11), np.ones(11) * 3],
        )
        assert len(r.alpha_values) == 3
        assert r.delta_omega_joint[0] == pytest.approx(0.5)
        assert r.delta_omega_sonly[1] == pytest.approx(0.9)
        assert r.best_params_joint[1].shape == (11,)

    def test_sensitivity_objective_with_fixed_alpha(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        params_7 = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0])
        fixed = (0.0, 0.0, 0.0, 0.0)
        val = sensitivity_objective(
            params_7, omega_true=1.0, ops=make_ops, fixed_alpha=fixed
        )
        assert np.isfinite(val) and val > 0

    def test_optimisation_with_fixed_alpha(
        self, make_ops: dict[str, np.ndarray]
    ) -> None:
        x0_7 = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0])
        fixed = (0.0, 0.0, 0.0, 0.0)
        result = run_optimisation(
            omega_true=1.0, ops=make_ops, x0=x0_7, maxiter=10, fixed_alpha=fixed
        )
        assert isinstance(result, OptimisationResult)
        assert result.params_opt.shape == (11,)
        # Alpha components should match the fixed values
        assert result.params_opt[7:] == pytest.approx([0.0, 0.0, 0.0, 0.0])

    def test_scan_alpha_with_reoptimisation_basic(self) -> None:
        result = scan_alpha_with_reoptimisation(
            "xx",
            alpha_values=np.array([-0.5, 0.0, 0.5]),
            n_restarts=2,
            maxiter=20,
            seed=42,
        )
        assert isinstance(result, AlphaReoptScanResult)
        assert len(result.alpha_values) == 3
        assert result.alpha_values[1] == pytest.approx(0.0)
        assert len(result.delta_omega_joint) == 3
        assert len(result.delta_omega_sonly) == 3
        assert len(result.best_params_joint) == 3
        assert len(result.best_params_sonly) == 3
        assert np.all(np.isfinite(result.delta_omega_joint))
        assert np.all(np.isfinite(result.delta_omega_sonly))

    @pytest.mark.parametrize("name", ["xx", "xz", "zx", "zz"])
    def test_scan_alpha_with_reoptimisation_all_names(self, name: str) -> None:
        result = scan_alpha_with_reoptimisation(
            name,
            alpha_values=np.array([-0.2, 0.0, 0.2]),
            n_restarts=2,
            maxiter=20,
            seed=42,
        )
        assert result.alpha_values.shape == (3,)

    def test_scan_alpha_with_reoptimisation_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError):
            scan_alpha_with_reoptimisation("invalid")

    def test_scan_alpha_with_reoptimisation_default_alpha_values(self) -> None:
        result = scan_alpha_with_reoptimisation("xx", n_restarts=2, maxiter=10, seed=42)
        # Default: 21 points in [-2, 2]
        assert result.alpha_values.shape == (21,)
        assert result.alpha_values[0] == pytest.approx(-2.0)
        assert result.alpha_values[-1] == pytest.approx(2.0)


class TestInteractionRobustness:
    def test_result_dataclass_defaults(self) -> None:
        r = InteractionRobustnessResult()
        assert len(r.T_hold_values) == 0
        assert len(r.alpha_values) == 0
        assert r.delta_omega_joint.shape == (0,)
        assert r.delta_omega_sonly.shape == (0,)

    def test_result_dataclass_with_values(self) -> None:
        T_hold = np.array([0.5, 1.0])
        alpha = np.array([-1.0, 0.0, 1.0])
        domega_j = np.array([[0.8, 0.6, 0.8], [0.4, 0.3, 0.4]])
        domega_s = np.array([[1.0, 0.9, 1.0], [0.5, 0.4, 0.5]])
        r = InteractionRobustnessResult(
            T_hold_values=T_hold,
            alpha_values=alpha,
            delta_omega_joint=domega_j,
            delta_omega_sonly=domega_s,
        )
        assert r.T_hold_values == pytest.approx(T_hold)
        assert r.alpha_values == pytest.approx(alpha)
        assert r.delta_omega_joint == pytest.approx(domega_j)
        assert r.delta_omega_sonly == pytest.approx(domega_s)
        assert r.delta_omega_joint.shape == (2, 3)

    def test_basic_smoke(self) -> None:
        T_hold_vals = np.array([0.5, 1.0])
        alpha_vals = np.array([-0.2, 0.0, 0.2])
        result = compute_interaction_robustness(
            T_hold_vals,
            alpha_vals,
            omega_true=1.0,
            alpha_name="xx",
        )
        assert isinstance(result, InteractionRobustnessResult)
        assert result.T_hold_values == pytest.approx(T_hold_vals)
        assert result.alpha_values == pytest.approx(alpha_vals)
        assert result.delta_omega_joint.shape == (2, 3)
        assert result.delta_omega_sonly.shape == (2, 3)
        assert np.all(np.isfinite(result.delta_omega_joint))
        assert np.all(np.isfinite(result.delta_omega_sonly))

    def test_sql_at_zero_alpha(self) -> None:
        """At α=0, both joint and S-only sensitivity ≈ 1/T_hold."""
        T_hold_vals = np.array([0.5, 1.0, 2.0])
        alpha_vals = np.array([0.0])
        result = compute_interaction_robustness(
            T_hold_vals,
            alpha_vals,
            omega_true=1.0,
            alpha_name="xx",
            theta_S=0.0,
            phi_S=0.0,
            theta_A=0.0,
            phi_A=0.0,
            T_BS=np.pi / 2,
        )
        for i, T_hold in enumerate(T_hold_vals):
            expected = 1.0 / T_hold
            assert result.delta_omega_sonly[i, 0] == pytest.approx(expected, rel=0.05)
            assert result.delta_omega_joint[i, 0] == pytest.approx(expected, rel=0.05)

    @pytest.mark.parametrize("name", ["xx", "xz", "zx", "zz"])
    def test_all_alpha_names(self, name: str) -> None:
        T_hold_vals = np.array([1.0])
        alpha_vals = np.array([-0.1, 0.0, 0.1])
        result = compute_interaction_robustness(
            T_hold_vals,
            alpha_vals,
            omega_true=1.0,
            alpha_name=name,
        )
        assert result.delta_omega_joint.shape == (1, 3)

    def test_invalid_alpha_name_raises(self) -> None:
        T_hold_vals = np.array([1.0])
        alpha_vals = np.array([0.0])
        with pytest.raises(ValueError):
            compute_interaction_robustness(
                T_hold_vals,
                alpha_vals,
                alpha_name="invalid",
            )

    def test_larger_scan(self) -> None:
        T_hold_vals = np.linspace(0.5, 2.0, 4)
        alpha_vals = np.linspace(-1.0, 1.0, 5)
        result = compute_interaction_robustness(
            T_hold_vals,
            alpha_vals,
            omega_true=1.0,
            alpha_name="zz",
        )
        assert result.delta_omega_joint.shape == (4, 5)
        assert result.delta_omega_sonly.shape == (4, 5)
        assert np.all(np.isfinite(result.delta_omega_joint))
        assert np.all(np.isfinite(result.delta_omega_sonly))
        # At α=0 (index 2), sensitivity should approximately equal 1/T_hold
        for i, T_hold in enumerate(T_hold_vals):
            expected = 1.0 / T_hold
            assert result.delta_omega_sonly[i, 2] == pytest.approx(expected, rel=0.05)
            assert result.delta_omega_joint[i, 2] == pytest.approx(expected, rel=0.05)


class TestParquetRoundtrip:
    """Verify that each plottable dataclass can roundtrip through Parquet."""

    def test_decoupled_baseline_roundtrip(self, tmp_path: Path) -> None:
        T_hold = np.array([0.5, 1.0, 2.0, 5.0])
        sql = 1.0 / T_hold
        original = DecoupledBaselineResult(
            T_hold_values=T_hold,
            delta_omega_values=sql.copy(),
            sql_values=sql.copy(),
        )
        parquet_path = tmp_path / "test.parquet"
        original.save_parquet(parquet_path)
        loaded = DecoupledBaselineResult.from_parquet(parquet_path)
        assert loaded.T_hold_values == pytest.approx(original.T_hold_values)
        assert loaded.delta_omega_values == pytest.approx(original.delta_omega_values)
        assert loaded.sql_values == pytest.approx(original.sql_values)

    def test_omega_scan_roundtrip(self, tmp_path: Path) -> None:
        original = OmegaScanResult(
            omega_values=np.array([0.5, 1.0, 2.0]),
            best_per_omega=np.array([0.2, 0.3, 0.4]),
            all_results={},
        )
        parquet_path = tmp_path / "test.parquet"
        original.save_parquet(parquet_path)
        loaded = OmegaScanResult.from_parquet(parquet_path)
        assert loaded.omega_values == pytest.approx(original.omega_values)
        assert loaded.best_per_omega == pytest.approx(original.best_per_omega)

    def test_alpha_reopt_roundtrip(self, tmp_path: Path) -> None:
        original = AlphaReoptScanResult(
            alpha_values=np.array([-1.0, 0.0, 1.0]),
            delta_omega_joint=np.array([0.5, 0.6, 0.7]),
            delta_omega_sonly=np.array([0.8, 0.9, 1.0]),
        )
        parquet_path = tmp_path / "test.parquet"
        original.save_parquet(parquet_path)
        loaded = AlphaReoptScanResult.from_parquet(parquet_path)
        assert loaded.alpha_values == pytest.approx(original.alpha_values)
        assert loaded.delta_omega_joint == pytest.approx(original.delta_omega_joint)
        assert loaded.delta_omega_sonly == pytest.approx(original.delta_omega_sonly)

    def test_alpha_single_roundtrip(self, tmp_path: Path) -> None:
        original = AlphaSingleScanResult(
            alpha_name="xx",
            alpha_values=np.linspace(-1.0, 1.0, 5),
            delta_omega_values=np.ones(5),
        )
        parquet_path = tmp_path / "test.parquet"
        original.save_parquet(parquet_path)
        loaded = AlphaSingleScanResult.from_parquet(parquet_path)
        assert loaded.alpha_name == original.alpha_name
        assert loaded.alpha_values == pytest.approx(original.alpha_values)
        assert loaded.delta_omega_values == pytest.approx(original.delta_omega_values)

    def test_alpha_random_search_roundtrip(self, tmp_path: Path) -> None:
        original = AlphaRandomSearchResult(
            alpha_samples=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            delta_omega_values=np.array([0.5, 0.6]),
            best_alpha=(1.0, 0.0, 0.0, 0.0),
            best_delta_omega=0.5,
        )
        parquet_path = tmp_path / "test.parquet"
        original.save_parquet(parquet_path)
        loaded = AlphaRandomSearchResult.from_parquet(parquet_path)
        assert loaded.alpha_samples == pytest.approx(original.alpha_samples)
        assert loaded.delta_omega_values == pytest.approx(original.delta_omega_values)
        assert loaded.best_delta_omega == pytest.approx(original.best_delta_omega)

    def test_interaction_robustness_roundtrip(self, tmp_path: Path) -> None:
        original = InteractionRobustnessResult(
            T_hold_values=np.array([0.5, 1.0]),
            alpha_values=np.array([-1.0, 0.0, 1.0]),
            delta_omega_joint=np.array([[0.8, 0.6, 0.8], [0.4, 0.3, 0.4]]),
            delta_omega_sonly=np.array([[1.0, 0.9, 1.0], [0.5, 0.4, 0.5]]),
        )
        parquet_path = tmp_path / "test.parquet"
        original.save_parquet(parquet_path)
        loaded = InteractionRobustnessResult.from_parquet(parquet_path)
        assert loaded.T_hold_values == pytest.approx(original.T_hold_values)
        assert loaded.alpha_values == pytest.approx(original.alpha_values)
        assert loaded.delta_omega_joint == pytest.approx(original.delta_omega_joint)
        assert loaded.delta_omega_sonly == pytest.approx(original.delta_omega_sonly)

    def test_covariance_analysis_roundtrip(self, tmp_path: Path) -> None:
        original = CovarianceAnalysisResult(
            coefficient_names=["α_xx", "α_xz", "α_zx", "α_zz"],
            max_covariances=np.array([0.12, 0.12, 0.10, 0.10]),
            covariance_signs=np.array([1.0, 1.0, 1.0, -1.0]),
        )
        parquet_path = tmp_path / "test.parquet"
        original.save_parquet(parquet_path)
        loaded = CovarianceAnalysisResult.from_parquet(parquet_path)
        assert loaded.coefficient_names == original.coefficient_names
        assert loaded.max_covariances == pytest.approx(original.max_covariances)
        assert loaded.covariance_signs == pytest.approx(original.covariance_signs)

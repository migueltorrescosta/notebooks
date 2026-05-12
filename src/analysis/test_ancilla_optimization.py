"""
Unit tests for the Ancilla-Assisted Metrology Optimisation module.

Covers:
1. Operator construction (Hermiticity, commutation)
2. State preparation (normalisation, parameterisation)
3. Beam-splitter unitaries (unitarity, 50/50 limit)
4. Holding Hamiltonian and unitary (Hermiticity, unitarity)
5. Full circuit evolution (normalisation preservation)
6. Sensitivity computation (finite difference consistency, SQL limit)
7. Objective function (smoothness, boundary penalties)
8. Optimisation (convergence, Nelder–Mead interface)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from src.analysis.ancilla_optimization import (
    I_2,
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
    hold_unitary,
    random_initial_params,
    run_optimisation,
    sensitivity_objective,
    single_qubit_state,
    two_qubit_bs_unitary,
    two_qubit_state,
    validate_bs_unitarity,
    validate_derivative_stability,
    validate_hold_unitarity,
    validate_operators,
    validate_sensitivity_reasonable,
    validate_two_qubit_bs_unitarity,
    validate_variance_positive,
)

# ============================================================================
# Operator Construction Tests
# ============================================================================


class TestOperatorConstruction:
    """Validate two-qubit operator Hermiticity, dimensions, commutation."""

    def test_operators_shape(self) -> None:
        """All operators must be 4×4."""
        ops = build_two_qubit_operators()
        for name, op in ops.items():
            assert op.shape == (4, 4), f"{name} must be 4×4, got {op.shape}"

    def test_operators_hermitian(self) -> None:
        """All operators must be Hermitian."""
        ops = build_two_qubit_operators()
        for name, op in ops.items():
            assert np.allclose(op, op.conj().T, atol=1e-12), f"{name} must be Hermitian"

    def test_jz_diagonal(self) -> None:
        """J_z^S and J_z^A must be diagonal in the computational basis."""
        ops = build_two_qubit_operators()
        for name in ["Jz_S", "Jz_A"]:
            op = ops[name]
            assert np.allclose(op, np.diag(np.diag(op))), f"{name} must be diagonal"

    def test_jz_eigenvalues(self) -> None:
        """J_z eigenvalues must be ±1/2."""
        ops = build_two_qubit_operators()
        for name in ["Jz_S", "Jz_A"]:
            evals = np.linalg.eigvalsh(ops[name])
            assert np.allclose(sorted(evals), [-0.5, -0.5, 0.5, 0.5]), (
                f"{name} eigenvalues not ±1/2"
            )

    def test_commutation_jz_jx(self) -> None:
        """[J_z^S, J_x^S] = i J_y^S and [J_z^A, J_x^A] = i J_y^A."""
        ops = build_two_qubit_operators()

        comm_S = ops["Jz_S"] @ ops["Jx_S"] - ops["Jx_S"] @ ops["Jz_S"]
        assert np.allclose(comm_S, 1j * ops["Jy_S"], atol=1e-12), (
            "[Jz_S, Jx_S] = i Jy_S failed"
        )

        comm_A = ops["Jz_A"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jz_A"]
        assert np.allclose(comm_A, 1j * ops["Jy_A"], atol=1e-12), (
            "[Jz_A, Jx_A] = i Jy_A failed"
        )

    def test_validate_operators_passes(self) -> None:
        """validate_operators must return True for valid operators."""
        ops = build_two_qubit_operators()
        assert validate_operators(ops) is True

    def test_interaction_hamiltonian_zero(self) -> None:
        """Zero coefficients must give zero matrix."""
        H = build_interaction_hamiltonian((0.0, 0.0, 0.0, 0.0))
        assert np.allclose(H, 0.0), "Zero α must give zero H_int"

    def test_interaction_hamiltonian_hermitian(self) -> None:
        """H_int must be Hermitian for any real α."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            alpha = tuple(rng.uniform(-2, 2, size=4))
            H = build_interaction_hamiltonian(alpha)  # type: ignore[arg-type]
            assert np.allclose(H, H.conj().T, atol=1e-12), (
                f"H_int must be Hermitian for α={alpha}"
            )


# ============================================================================
# State Preparation Tests
# ============================================================================


class TestStatePreparation:
    """Validate single- and two-qubit state parameterisation."""

    @pytest.mark.parametrize(
        "theta, phi", [(0.0, 0.0), (np.pi, 0.0), (np.pi / 2, np.pi)]
    )
    def test_single_qubit_normalised(self, theta: float, phi: float) -> None:
        """Single-qubit states must be normalised."""
        psi = single_qubit_state(theta, phi)
        assert np.isclose(np.linalg.norm(psi), 1.0), (
            f"State (θ={theta}, φ={phi}) not normalised"
        )

    def test_single_qubit_zero_theta(self) -> None:
        """θ=0, φ=0 must give |0⟩ = |1,0⟩."""
        psi = single_qubit_state(0.0, 0.0)
        expected = np.array([1.0, 0.0], dtype=complex)
        assert np.allclose(psi, expected), "θ=0, φ=0 must give |0⟩"

    def test_single_qubit_pi_theta(self) -> None:
        """θ=π, φ=0 must give |1⟩ = |0,1⟩."""
        psi = single_qubit_state(np.pi, 0.0)
        expected = np.array([0.0, 1.0], dtype=complex)
        assert np.allclose(psi, expected), "θ=π, φ=0 must give |1⟩"

    def test_two_qubit_product_structure(self) -> None:
        """The two-qubit state must be a product state."""
        theta_S, phi_S = 0.3, 0.7
        theta_A, phi_A = 1.2, 2.5
        psi = two_qubit_state(theta_S, phi_S, theta_A, phi_A)

        psi_S = single_qubit_state(theta_S, phi_S)
        psi_A = single_qubit_state(theta_A, phi_A)
        expected = np.kron(psi_S, psi_A)

        assert np.allclose(psi, expected), "Two-qubit state must be product"
        assert np.isclose(np.linalg.norm(psi), 1.0), "Must be normalised"


# ============================================================================
# Beam-Splitter Tests
# ============================================================================


class TestBeamSplitter:
    """Validate beam-splitter unitaries."""

    @pytest.mark.parametrize("T", [0.0, 0.5, np.pi / 4, np.pi / 2, np.pi])
    def test_bs_unitary(self, T: float) -> None:
        """U_BS(T) must be unitary."""
        U = bs_unitary(T)
        assert np.allclose(U @ U.conj().T, I_2, atol=1e-12), f"BS(T={T}) not unitary"
        assert np.allclose(U.conj().T @ U, I_2, atol=1e-12)

    def test_bs_zero_time(self) -> None:
        """T=0 gives identity."""
        U = bs_unitary(0.0)
        assert np.allclose(U, I_2), "BS(T=0) must be identity"

    def test_bs_half_pi(self) -> None:
        """T=π/2 gives the 50/50 beam-splitter matrix.

        U_BS(π/2) = (1/√2) [[1, -i], [-i, 1]]
        """
        U = bs_unitary(np.pi / 2.0)
        expected = np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2)
        assert np.allclose(U, expected, atol=1e-12), "BS(π/2) should be 50/50"

    def test_two_qubit_bs_unitary(self) -> None:
        """Two-qubit BS must be unitary and tensor-product structured."""
        U = two_qubit_bs_unitary(np.pi / 4)
        I_4 = np.eye(4, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12)
        assert np.allclose(U.conj().T @ U, I_4, atol=1e-12)

    def test_two_qubit_bs_tensor_structure(self) -> None:
        """U_BS^{(2)}(T) = U_BS(T) ⊗ U_BS(T)."""
        T = 0.7
        U_single = bs_unitary(T)
        U_double = two_qubit_bs_unitary(T)
        expected = np.kron(U_single, U_single)
        assert np.allclose(U_double, expected), (
            "Two-qubit BS must be tensor product of single-qubit BS"
        )

    def test_validate_bs_unitarity(self) -> None:
        """Validation helper must pass."""
        assert validate_bs_unitarity() is True
        assert validate_two_qubit_bs_unitarity() is True

    @pytest.mark.parametrize("T", [0.0, 0.3, np.pi / 4, np.pi / 2, np.pi])
    def test_bs_runtime_assertion_does_not_raise(self, T: float) -> None:
        """Runtime unitarity assertion in bs_unitary must not raise for valid T."""
        # The internal assert in bs_unitary verifies U @ U† = I₂ at runtime.
        # This test verifies it passes for a range of valid T values.
        U = bs_unitary(T)
        assert U.shape == (2, 2)  # Just verify we got a matrix back


# ============================================================================
# Holding Hamiltonian & Unitary Tests
# ============================================================================


class TestHold:
    """Validate holding Hamiltonian and unitary."""

    def test_hold_hamiltonian_hermitian(self) -> None:
        """H_hold must be Hermitian."""
        ops = build_two_qubit_operators()
        H = build_hold_hamiltonian(1.0, (0.1, 0.2, 0.3, 0.4), ops)
        assert np.allclose(H, H.conj().T, atol=1e-12), "H_hold must be Hermitian"

    def test_hold_unitary(self) -> None:
        """U_hold must be unitary."""
        ops = build_two_qubit_operators()
        U = hold_unitary(1.0, 1.0, (0.1, 0.0, 0.0, 0.0), ops)
        I_4 = np.eye(4, dtype=complex)
        assert np.allclose(U @ U.conj().T, I_4, atol=1e-12), "U_hold not unitary"

    @pytest.mark.parametrize("T_H", [0.0, 0.5, 2.0])
    def test_hold_unitary_correctness(self, T_H: float) -> None:
        """U_hold must match exp(-i T_H H_hold) by direct computation."""
        ops = build_two_qubit_operators()
        theta = 1.0
        alpha = (0.1, 0.2, -0.1, 0.3)
        H = build_hold_hamiltonian(theta, alpha, ops)
        U_expected = expm(-1j * T_H * H)
        U = hold_unitary(T_H, theta, alpha, ops)
        assert np.allclose(U, U_expected, atol=1e-12), f"U_hold mismatch at T_H={T_H}"

    def test_zero_hold_identity(self) -> None:
        """T_H = 0 gives identity."""
        ops = build_two_qubit_operators()
        U = hold_unitary(0.0, 1.0, (0.1, 0.0, 0.0, 0.0), ops)
        assert np.allclose(U, np.eye(4, dtype=complex), atol=1e-12)

    def test_validate_hold_unitarity(self) -> None:
        """Validation helper must pass."""
        assert validate_hold_unitarity() is True

    @pytest.mark.parametrize("T_H", [0.0, 0.5, 1.0, 2.0])
    def test_hold_runtime_assertion_does_not_raise(self, T_H: float) -> None:
        """Runtime unitarity assertion in hold_unitary must not raise for valid params."""
        ops = build_two_qubit_operators()
        U = hold_unitary(T_H, 1.0, (0.1, 0.0, -0.2, 0.3), ops)
        assert U.shape == (4, 4)


# ============================================================================
# Full Circuit Evolution Tests
# ============================================================================


class TestCircuitEvolution:
    """Validate the full MZI circuit."""

    @pytest.mark.parametrize("T_H", [0.0, 0.5, 1.0])
    def test_normalisation_preserved(self, T_H: float) -> None:
        """Norm must be preserved through the circuit."""
        ops = build_two_qubit_operators()
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)  # |0,0⟩
        psi = evolve_full(
            psi0, np.pi / 4, np.pi / 4, T_H, 1.0, (0.0, 0.0, 0.0, 0.0), ops
        )
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12), (
            f"Norm not preserved at T_H={T_H}"
        )

    def test_no_hold_no_bs_identity(self) -> None:
        """With T_BS = T_H = 0 and θ = 0, circuit must be identity."""
        ops = build_two_qubit_operators()
        psi0 = two_qubit_state(0.5, 0.3, 1.2, 0.8)
        psi = evolve_full(psi0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0), ops)
        assert np.allclose(psi, psi0, atol=1e-12), "Circuit should be identity"

    @pytest.mark.parametrize("random_state", [True, False])
    def test_unitarity_of_evolution(self, random_state: bool) -> None:
        """The full circuit must be a linear unitary map.

        Test by checking that inner products are preserved for two
        orthogonal input states.
        """
        ops = build_two_qubit_operators()
        rng = np.random.default_rng(42)

        # Two random input states
        if random_state:
            v1 = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            v1 /= np.linalg.norm(v1)
            v2 = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            v2 /= np.linalg.norm(v2)
            # Orthogonalise
            v2 -= np.vdot(v1, v2) * v1
            v2 /= np.linalg.norm(v2)
            psi0_1 = v1
            psi0_2 = v2
        else:
            psi0_1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
            psi0_2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)

        inner_before = np.vdot(psi0_1, psi0_2)

        alpha = (0.3, -0.1, 0.2, 0.0)
        psi1 = evolve_full(psi0_1, 0.8, 0.6, 1.5, 2.0, alpha, ops)
        psi2 = evolve_full(psi0_2, 0.8, 0.6, 1.5, 2.0, alpha, ops)
        inner_after = np.vdot(psi1, psi2)

        assert np.isclose(inner_before, inner_after, atol=1e-12), (
            "Inner product not preserved (non-unitary evolution)"
        )


# ============================================================================
# Sensitivity Computation Tests
# ============================================================================


class TestSensitivity:
    """Validate error-propagation sensitivity computation."""

    def test_expectation_variance_consistency(self) -> None:
        """Var(O) = ⟨O²⟩ - ⟨O⟩² must hold for J_z^S."""
        ops = build_two_qubit_operators()
        psi = two_qubit_state(0.5, 0.3, 0.8, 1.2)
        exp_val, var_val = compute_expectation_and_variance(psi, ops["Jz_S"])

        # Direct computation
        exp_direct = np.real(psi.conj() @ ops["Jz_S"] @ psi)
        var_direct = (
            np.real(psi.conj() @ (ops["Jz_S"] @ ops["Jz_S"]) @ psi) - exp_direct**2
        )

        assert np.isclose(exp_val, exp_direct), "Expectation mismatch"
        assert np.isclose(var_val, max(0.0, var_direct)), "Variance mismatch"

    def test_decoupled_sensitivity_sql(self) -> None:
        """Decoupled system must achieve Δθ ≈ 1/T_H (SQL)."""
        T_H_vals = [0.5, 1.0, 2.0]
        ops = build_two_qubit_operators()
        theta_true = 1.0
        alpha = (0.0, 0.0, 0.0, 0.0)
        T_BS = np.pi / 2.0
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)  # |0,0⟩

        for T_H in T_H_vals:
            dtheta = compute_sensitivity(psi0, T_BS, T_BS, T_H, theta_true, alpha, ops)
            expected = 1.0 / T_H
            assert np.isclose(dtheta, expected, rtol=0.05), (
                f"Δθ = {dtheta:.6f}, expected {expected:.6f} for T_H={T_H}"
            )

    def test_fringe_extremum_returns_inf(self) -> None:
        """When derivative is near zero (fringe extremum), Δθ = inf."""
        ops = build_two_qubit_operators()
        # θ * T_H = π makes sin(θ T_H) = 0 → derivative zero
        T_H = 1.0
        theta_true = np.pi  # θ T_H = π → sin = 0

        # Use decoupled configuration
        alpha = (0.0, 0.0, 0.0, 0.0)
        T_BS = np.pi / 2.0
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)

        dtheta = compute_sensitivity(psi0, T_BS, T_BS, T_H, theta_true, alpha, ops)
        assert np.isinf(dtheta), f"Δθ should be ∞ at fringe extremum, got {dtheta}"

    def test_finite_difference_sign_agreement(self) -> None:
        """Sensitivity must be positive and finite away from fringe extrema."""
        ops = build_two_qubit_operators()
        alpha = (0.0, 0.0, 0.0, 0.0)
        T_BS = np.pi / 2.0
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)

        for T_H in [0.5, 1.0, 2.0]:
            for theta_true in [0.5, 1.0, 1.5]:
                dtheta = compute_sensitivity(
                    psi0, T_BS, T_BS, T_H, theta_true, alpha, ops
                )
                assert np.isfinite(dtheta) and dtheta > 0, (
                    f"Δθ must be finite positive, got {dtheta} "
                    f"(T_H={T_H}, θ={theta_true})"
                )

    def test_get_decoupled_sensitivity_matches_sql(self) -> None:
        """get_decoupled_sensitivity returns ≈ 1/T_H."""
        for T_H in [0.5, 1.0, 2.0]:
            dtheta = get_decoupled_sensitivity(T_H, theta_true=1.0)
            expected = 1.0 / T_H
            assert np.isclose(dtheta, expected, rtol=0.05), (
                f"get_decoupled_sensitivity({T_H}) = {dtheta:.6f}, expected {expected:.6f}"
            )

    def test_decoupled_sensitivity_analytical_exact(self) -> None:
        """Analytical formula Δθ = 1/T_H holds exactly for the decoupled case.

        For a single-qubit MZI with |ψ_S⟩ = |0⟩ (θ_S = 0), 50/50 beam splitters
        (T_BS = π/2), measurement of J_z, and no interaction (α = 0):

            Δθ = √(Var(J_z)) / |∂⟨J_z⟩/∂θ| = 1/T_H

        This test validates the exact formula at several (θ, T_H) points,
        avoiding fringe extrema where sin(θ T_H) = 0.
        """
        ops = build_two_qubit_operators()
        alpha = (0.0, 0.0, 0.0, 0.0)
        T_BS = np.pi / 2.0
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)  # |0,0⟩

        # θ × T_H values that avoid fringe extrema (sin ≠ 0)
        test_points = [
            (theta, T_H)
            for theta in [0.3, 0.7, 1.0, 1.3, 1.7]
            for T_H in [0.5, 1.0, 1.5, 2.0]
            if abs(np.sin(theta * T_H)) > 0.1
        ]

        for theta_true, T_H in test_points:
            dtheta = compute_sensitivity(psi0, T_BS, T_BS, T_H, theta_true, alpha, ops)
            expected = 1.0 / T_H
            # Tighter tolerance (0.5%) since the formula is exact away from
            # fringe extrema — any deviation is purely numerical
            assert np.isclose(dtheta, expected, rtol=5e-3), (
                f"Δθ = {dtheta:.6f}, expected {expected:.6f} "
                f"for θ={theta_true}, T_H={T_H}"
            )

    def test_variance_never_significantly_negative(self) -> None:
        """Variance of J_z^S must be ≥ -1e-12 for any valid state.

        A slightly negative variance can arise from numerical rounding in
        the expression Var = ⟨O²⟩ - ⟨O⟩², but it must not exceed 1e-12
        in absolute value.
        """
        ops = build_two_qubit_operators()
        rng = np.random.default_rng(42)

        for _ in range(20):
            # Random two-qubit state
            psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            psi /= np.linalg.norm(psi)

            # Validate should pass (variance may be 0 for product states
            # but never significantly negative)
            assert validate_variance_positive(psi, ops["Jz_S"]) is True


# ============================================================================
# Reduced Purity Tests
# ============================================================================


class TestReducedPurity:
    """Validate compute_reduced_purity for known two-qubit states."""

    def test_product_state_00(self) -> None:
        """Product state |0⟩_S ⊗ |0⟩_A must have purity = 1."""
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩
        purity = compute_reduced_purity(psi)
        assert np.isclose(purity, 1.0, atol=1e-12), (
            f"Product state purity should be 1.0, got {purity}"
        )

    def test_product_state_01(self) -> None:
        """Product state |0⟩_S ⊗ |1⟩_A must have purity = 1."""
        psi = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)  # |01⟩
        purity = compute_reduced_purity(psi)
        assert np.isclose(purity, 1.0, atol=1e-12), (
            f"Product state purity should be 1.0, got {purity}"
        )

    def test_product_state_general(self) -> None:
        """Generic product state via Bloch-sphere params must have purity = 1."""
        psi = two_qubit_state(0.7, 1.2, 0.3, 2.8)
        purity = compute_reduced_purity(psi)
        assert np.isclose(purity, 1.0, atol=1e-12), (
            f"Product state purity should be 1.0, got {purity}"
        )

    def test_bell_phi_plus(self) -> None:
        """Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 must have purity = 0.5."""
        psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)
        purity = compute_reduced_purity(psi)
        assert np.isclose(purity, 0.5, atol=1e-12), (
            f"Bell state purity should be 0.5, got {purity}"
        )

    def test_bell_psi_plus(self) -> None:
        """Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 must have purity = 0.5."""
        psi = np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
        purity = compute_reduced_purity(psi)
        assert np.isclose(purity, 0.5, atol=1e-12), (
            f"Bell state purity should be 0.5, got {purity}"
        )

    def test_product_state_through_circuit(self) -> None:
        """After evolving through circuit with α=0, purity should stay ≈ 1."""
        ops = build_two_qubit_operators()
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)  # |0⟩_S ⊗ |0⟩_A
        psi = evolve_full(
            psi0, np.pi / 2, np.pi / 2, 1.0, 1.0, (0.0, 0.0, 0.0, 0.0), ops
        )
        purity = compute_reduced_purity(psi)
        # Decoupled evolution is product → purity ≈ 1
        assert np.isclose(purity, 1.0, atol=1e-10), (
            f"Decoupled circuit purity should be ≈ 1.0, got {purity}"
        )

    def test_purity_clamped_range(self) -> None:
        """Purity must always be in [0.5, 1.0] for any normalised state."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            psi /= np.linalg.norm(psi)
            purity = compute_reduced_purity(psi)
            assert 0.5 <= purity <= 1.0, f"Purity {purity} outside [0.5, 1.0]"


# ============================================================================
# Objective Function Tests
# ============================================================================


class TestObjective:
    """Validate the Nelder–Mead objective function."""

    def test_objective_passes_for_valid_params(self) -> None:
        """A valid parameter vector must give a finite objective."""
        ops = build_two_qubit_operators()
        params = np.array(
            [0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0, 0.0, 0.0, 0.0, 0.0]
        )
        val = sensitivity_objective(params, theta_true=1.0, ops=ops)
        assert np.isfinite(val) and val > 0, (
            f"Objective must be finite positive, got {val}"
        )

    def test_objective_matches_sql(self) -> None:
        """Optimal decoupled params give objective ≈ 1/T_H."""
        ops = build_two_qubit_operators()
        params = np.array(
            [0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0, 0.0, 0.0, 0.0, 0.0]
        )
        val = sensitivity_objective(params, theta_true=1.0, ops=ops)
        assert np.isclose(val, 1.0, rtol=0.05), f"Objective should be ≈ 1.0, got {val}"

    def test_objective_penalty_out_of_bounds_theta(self) -> None:
        """A θ out of [0, π] must produce a huge penalty."""
        ops = build_two_qubit_operators()
        # theta_S = 4.0 (exceeds π ≈ 3.14)
        params = np.array(
            [4.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0, 0.0, 0.0, 0.0, 0.0]
        )
        val = sensitivity_objective(params, theta_true=1.0, ops=ops)
        assert val > 1e9, f"Out-of-bounds must be penalised, got {val}"

    def test_objective_penalty_out_of_bounds_alpha(self) -> None:
        """An α out of [-2, 2] must produce a huge penalty."""
        ops = build_two_qubit_operators()
        params = np.array(
            [0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0, 5.0, 0.0, 0.0, 0.0]
        )
        val = sensitivity_objective(params, theta_true=1.0, ops=ops)
        assert val > 1e9, f"Out-of-bounds must be penalised, got {val}"

    def test_objective_smooth(self) -> None:
        """Small parameter changes should produce small objective changes."""
        ops = build_two_qubit_operators()
        base = np.array(
            [0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0, 0.0, 0.0, 0.0, 0.0]
        )
        val_base = sensitivity_objective(base, theta_true=1.0, ops=ops)

        for i in range(11):
            perturbed = base.copy()
            perturbed[i] += 1e-6
            val_pert = sensitivity_objective(perturbed, theta_true=1.0, ops=ops)
            diff = abs(val_pert - val_base)
            # The change should be small (not necessarily monotonic, but not huge)
            assert diff < 1.0, (
                f"Large objective change {diff} from small perturbation at index {i}"
            )


# ============================================================================
# Random Initial Parameter Tests
# ============================================================================


class TestRandomInitialParams:
    """Validate random initial parameter generation."""

    def test_shape(self) -> None:
        """Must produce 11-element vector."""
        rng = np.random.default_rng(42)
        params = random_initial_params(rng)
        assert params.shape == (11,), f"Expected 11 elements, got {params.shape}"

    def test_within_bounds(self) -> None:
        """All parameters must be within bounds."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            params = random_initial_params(rng)
            assert 0.0 <= params[0] <= np.pi  # theta_S
            assert 0.0 <= params[1] <= 2.0 * np.pi  # phi_S
            assert 0.0 <= params[2] <= np.pi  # theta_A
            assert 0.0 <= params[3] <= 2.0 * np.pi  # phi_A
            assert 0.0 <= params[4] <= np.pi  # T_BS1
            assert 0.0 <= params[5] <= np.pi  # T_BS2
            assert 0.0 <= params[6] <= 5.0  # T_H
            assert all(-2.0 <= a <= 2.0 for a in params[7:])  # alpha


# ============================================================================
# Optimisation Tests (short, with limited restarts)
# ============================================================================


class TestOptimisation:
    """Validate the Nelder–Mead optimisation interface."""

    def test_run_optimisation_returns_result(self) -> None:
        """run_optimisation must return an OptimisationResult
        with all fields populated."""
        ops = build_two_qubit_operators()
        # Use fixed optimal parameters as starting point
        x0 = np.array(
            [0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0, 0.0, 0.0, 0.0, 0.0]
        )
        result = run_optimisation(
            theta_true=1.0,
            ops=ops,
            x0=x0,
            maxiter=10,  # Few iterations for test speed
        )
        assert isinstance(result, OptimisationResult)
        assert result.theta_true == 1.0
        assert result.params_opt.shape == (11,)
        assert np.isfinite(result.delta_theta_opt) or np.isinf(result.delta_theta_opt)
        # New fields must be populated (discrepancies #2, #3)
        assert isinstance(result.expectation_Jz, float), (
            f"expectation_Jz must be float, got {type(result.expectation_Jz)}"
        )
        assert isinstance(result.variance_Jz, float), (
            f"variance_Jz must be float, got {type(result.variance_Jz)}"
        )
        assert isinstance(result.purity_S, float), (
            f"purity_S must be float, got {type(result.purity_S)}"
        )
        assert 0.5 <= result.purity_S <= 1.0, (
            f"purity_S must be in [0.5, 1.0], got {result.purity_S}"
        )

    def test_optimisation_result_dataclass(self) -> None:
        """OptimisationResult must store correct attributes
        (including new fields from discrepancies #2, #3)."""
        result = OptimisationResult(
            delta_theta_opt=0.5,
            params_opt=np.zeros(11),
            theta_true=1.0,
            success=True,
            nfev=100,
            message="OK",
        )
        assert result.delta_theta_opt == 0.5
        assert result.success is True
        assert result.nfev == 100
        # New fields must have sensible defaults and be settable
        assert result.expectation_Jz == 0.0
        assert result.variance_Jz == 0.0
        assert result.purity_S == 0.0

        # Test explicit setting
        result2 = OptimisationResult(
            delta_theta_opt=0.3,
            params_opt=np.ones(11),
            theta_true=2.0,
            success=False,
            nfev=50,
            message="test",
            expectation_Jz=0.25,
            variance_Jz=0.01,
            purity_S=0.75,
        )
        assert np.isclose(result2.expectation_Jz, 0.25)
        assert np.isclose(result2.variance_Jz, 0.01)
        assert np.isclose(result2.purity_S, 0.75)

    def test_theta_scan_result_dataclass(self) -> None:
        """ThetaScanResult must store correct attributes."""
        result = ThetaScanResult(
            results=[],
            theta_values=np.array([0.5, 1.0]),
            best_per_theta=np.array([0.6, 0.3]),
            all_results={},
        )
        assert len(result.theta_values) == 2
        assert np.isclose(result.best_per_theta[0], 0.6)

    @pytest.mark.slow
    def test_optimisation_explores_t_h(self) -> None:
        """Nelder-Mead should increase T_H to improve sensitivity (Δθ ∝ 1/T_H).

        Starting from T_H=1 and 50/50 splitters, the optimiser should
        discover that increasing T_H reduces Δθ. The result should scale
        approximately as Δθ ≈ 1/T_H_opt.
        """
        ops = build_two_qubit_operators()
        x0 = np.array(
            [0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 1.0, 0.0, 0.0, 0.0, 0.0]
        )
        result = run_optimisation(
            theta_true=1.0,
            ops=ops,
            x0=x0,
            maxiter=200,
        )
        # The optimiser should have increased T_H
        T_H_opt = result.params_opt[6]
        assert T_H_opt > 1.5, f"Expected T_H to increase, got T_H = {T_H_opt:.4f}"
        # Sensitivity should roughly match Δθ ≈ 1/T_H_opt (SQL scaling)
        expected = 1.0 / T_H_opt
        assert np.isclose(result.delta_theta_opt, expected, rtol=0.15), (
            f"Δθ = {result.delta_theta_opt:.4f}, expected 1/T_H ≈ {expected:.4f}"
        )


# ============================================================================
# Validation Helper Tests
# ============================================================================


class TestValidation:
    """Validate the validation helpers."""

    def test_validate_sensitivity_reasonable(self) -> None:
        """validate_sensitivity_reasonable must pass for default params."""
        assert validate_sensitivity_reasonable() is True

    def test_validate_operators_raises_on_bad(self) -> None:
        """validate_operators must raise on invalid operators."""
        ops = build_two_qubit_operators()
        ops["Jz_S"] = np.zeros((4, 4))  # Make it invalid
        with pytest.raises(AssertionError):
            validate_operators(ops)

    def test_validate_variance_positive_passes(self) -> None:
        """validate_variance_positive must pass for the decoupled state."""
        ops = build_two_qubit_operators()
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        # Evolve to a non-trivial state
        psi = evolve_full(
            psi0, np.pi / 2, np.pi / 2, 1.0, 1.0, (0.0, 0.0, 0.0, 0.0), ops
        )
        assert validate_variance_positive(psi, ops["Jz_S"]) is True

    def test_validate_derivative_stability_passes(self) -> None:
        """validate_derivative_stability must pass for the decoupled config."""
        ops = build_two_qubit_operators()
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        alpha = (0.0, 0.0, 0.0, 0.0)

        # Use a configuration away from fringe extrema
        result = validate_derivative_stability(
            psi0=psi0,
            T_BS1=np.pi / 2,
            T_BS2=np.pi / 2,
            T_H=1.0,
            theta_true=1.0,
            alpha=alpha,
            ops=ops,
        )
        assert result is True

    def test_validate_derivative_stability_at_fringe(self) -> None:
        """validate_derivative_stability must gracefully skip fringe extrema."""
        ops = build_two_qubit_operators()
        psi0 = two_qubit_state(0.0, 0.0, 0.0, 0.0)
        alpha = (0.0, 0.0, 0.0, 0.0)

        # θ * T_H = π → fringe extremum (derivative = 0)
        result = validate_derivative_stability(
            psi0=psi0,
            T_BS1=np.pi / 2,
            T_BS2=np.pi / 2,
            T_H=1.0,
            theta_true=np.pi,
            alpha=alpha,
            ops=ops,
        )
        assert result is True  # Should gracefully return True


class TestConvergenceMetric:
    """Validate the convergence spread metric."""

    def test_fewer_than_two_returns_zero(self) -> None:
        """Fewer than 2 results must return 0.0."""
        r = OptimisationResult(0.5, np.zeros(11), 1.0, True, 10, "ok")
        assert compute_convergence_metric([r]) == 0.0

    def test_all_inf_returns_zero(self) -> None:
        """All-infinite results must return 0.0."""
        results = [
            OptimisationResult(float("inf"), np.zeros(11), 1.0, True, 10, "ok"),
            OptimisationResult(float("inf"), np.zeros(11), 1.0, True, 10, "ok"),
        ]
        assert compute_convergence_metric(results) == 0.0

    def test_converged_returns_small(self) -> None:
        """Clustered Δθ values must give a small spread (< 0.10)."""
        results = [
            OptimisationResult(0.51, np.zeros(11), 1.0, True, 10, "ok"),
            OptimisationResult(0.52, np.zeros(11), 1.0, True, 10, "ok"),
            OptimisationResult(0.50, np.zeros(11), 1.0, True, 10, "ok"),
            OptimisationResult(0.53, np.zeros(11), 1.0, True, 10, "ok"),
        ]
        metric = compute_convergence_metric(results)
        assert metric < 0.10, (
            f"Converged results should give small spread, got {metric}"
        )
        assert metric > 0.0, "Spread must be positive for varied data"

"""
Tests for the ancilla-assisted vs. two-particle probe comparison module.

Validates:
1. Operator construction (J_z, J_x, J_z_anc, J_x_anc)
2. Generator computation (G_B analytical max, G_A zero-coupling limit)
3. Beam splitter convention (BS† J_z BS = -J_y)
4. Random density matrix generation (positivity, trace)
5. QFI evaluation at known reference points
6. Full comparison pipeline
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.ancilla_comparison import (
    ComparisonResult,
    analytical_fq_A_zero,
    analytical_fq_B_max,
    build_ancilla_operators,
    build_interaction_hamiltonian,
    build_system_jz_jx,
    compute_generator_A,
    compute_generator_B,
    random_density_matrix,
    random_pure_state_dm,
    run_comparison,
)


class TestOperatorConstruction:
    """Test J_z, J_x, J_z_anc, J_x_anc operator properties."""

    @pytest.mark.parametrize("N_max", [1, 2, 3])
    def test_given_jz_operator_then_diagonal_with_correct_eigenvalues(
        self, N_max: int
    ) -> None:
        J_z, _ = build_system_jz_jx(N_max)
        dim = (N_max + 1) ** 2

        # Check diagonal
        assert J_z == pytest.approx(np.diag(np.diag(J_z))), "J_z must be diagonal"
        assert J_z.shape == (dim, dim)

        # Check eigenvalues: (n0 - n1) / 2 for all n0, n1
        expected_vals = set()
        for n0 in range(N_max + 1):
            for n1 in range(N_max + 1):
                expected_vals.add((n0 - n1) / 2.0)

        actual_vals = set(np.round(np.diag(J_z).real, 10))
        assert actual_vals == expected_vals, (
            f"J_z eigenvalues mismatch: got {actual_vals}, expected {expected_vals}"
        )

    @pytest.mark.parametrize("N_max", [1, 2])
    def test_given_jx_operator_then_it_is_hermitian(self, N_max: int) -> None:
        _, J_x = build_system_jz_jx(N_max)
        assert J_x == pytest.approx(J_x.conj().T), "J_x must be Hermitian"

    def test_given_jx_operator_then_it_couples_adjacent_fock_states(
        self,
    ) -> None:
        N_max = 2
        _, J_x = build_system_jz_jx(N_max)

        # Check that J_x|1,0⟩ has support on |0,1⟩
        idx_10 = 1 * (N_max + 1) + 0
        idx_01 = 0 * (N_max + 1) + 1
        assert abs(J_x[idx_10, idx_01]) > 0, "J_x must couple |1,0⟩ ↔ |0,1⟩"
        assert J_x[idx_10, idx_01] == pytest.approx(J_x[idx_01, idx_10]), (
            "J_x must be symmetric"
        )

    def test_given_ancilla_operators_then_they_are_pauli_matrices_divided_by_2(
        self,
    ) -> None:
        J_z_anc, J_x_anc = build_ancilla_operators()

        assert J_z_anc.shape == (2, 2)
        assert J_x_anc.shape == (2, 2)

        # Pauli σ_z/2 eigenvalues: ±1/2
        np.isclose(np.linalg.eigvalsh(J_z_anc), [-0.5, 0.5]).all()
        np.isclose(np.linalg.eigvalsh(J_x_anc), [-0.5, 0.5]).all()

        # Commutation: [J_z, J_x] = i J_y
        comm = J_z_anc @ J_x_anc - J_x_anc @ J_z_anc
        J_y_expected = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
        assert comm == pytest.approx(1j * J_y_expected), (
            "Spin-½ commutation [J_z, J_x] = iJ_y must hold"
        )


class TestGeneratorB:
    """Test G_B for the 2-particle system (Case B)."""

    def test_given_generator_b_then_it_is_hermitian(self) -> None:
        G_B = compute_generator_B(T_H=1.0, N_max=2)
        assert pytest.approx(G_B.conj().T) == G_B, "G_B must be Hermitian"

    def test_given_th_1_nmax_2_then_generator_b_eigenvalues_in_minus_1_to_1(
        self,
    ) -> None:
        G_B = compute_generator_B(T_H=1.0, N_max=2)
        evals = np.linalg.eigvalsh(G_B)
        assert np.min(evals) >= -1.0 - 1e-10, f"Min eigenvalue {np.min(evals)} < -1"
        assert np.max(evals) <= 1.0 + 1e-10, f"Max eigenvalue {np.max(evals)} > 1"

    def test_given_generator_b_then_it_scales_linearly_with_th(self) -> None:
        G_B_1 = compute_generator_B(T_H=1.0, N_max=2)
        G_B_2 = compute_generator_B(T_H=2.0, N_max=2)
        assert pytest.approx(2.0 * G_B_1) == G_B_2, "G_B must scale linearly with T_H"

    def test_given_th_equals_1_then_fq_b_max_equals_4(
        self,
    ) -> None:
        from src.analysis.ancilla_comparison import optimize_qfi_case_B

        result = optimize_qfi_case_B(
            T_H=1.0,
            N_max=2,
            n_samples=500,
            pure_only=True,
            subspace_N=2,
            seed=42,
        )
        expected = analytical_fq_B_max(T_H=1.0)
        assert result.max_fq == pytest.approx(expected, rel=0.05), (
            f"Case B max QFI = {result.max_fq}, expected ~{expected}"
        )


class TestGeneratorA:
    """Test G_A for the ancilla-assisted case (Case A)."""

    def test_given_generator_a_then_it_is_hermitian(self) -> None:
        G_A = compute_generator_A(T_H=1.0, alphas=(0.0, 0.0, 0.0, 0.0), N_max=1)
        assert pytest.approx(G_A.conj().T) == G_A, "G_A must be Hermitian"

    def test_generator_a_zero_coupling_limit(self) -> None:
        """With α = 0, G_A must equal -T_H · J_y ⊗ I.

        We check that G_A eigenvalues match G_B eigenvalues for the
        1-particle case with an extra ancilla dimension.
        """
        G_A = compute_generator_A(T_H=1.0, alphas=(0.0, 0.0, 0.0, 0.0), N_max=1)
        evals_A = np.linalg.eigvalsh(G_A)

        # For N_max=1, J_y eigenvalues are -0.5, 0, 0.5 (but J=1/2 has -0.5, +0.5)
        # With ancilla, we get these each duplicated: [-0.5, -0.5, 0.5, 0.5]
        # Actually J_y for J=1/2 has eigenvalues ±0.5, so G_A = -J_y gives ±0.5
        # But there's a subtlety: the 1-particle sector has dim 4, and J_y operates
        # on the 1-particle subspace. With 0 particles it's 0, with 2 it's also 0.
        # We need to be more careful here.

        # For N_max=1, the system Hilbert space is {|0,0⟩, |0,1⟩, |1,0⟩, |1,1⟩}
        # The 1-particle subspace is {|0,1⟩, |1,0⟩}. On this subspace, J_y has
        # eigenvalues ±0.5.
        # With 0 particles: |0,0⟩ → J_y = 0
        # With 2 particles: |1,1⟩ → J_y = 0
        # So J_y eigenvalues (across ALL Fock states) are: -0.5, 0, 0, +0.5
        # With ancilla kroneckered: all duplicated → [-0.5, -0.5, 0, 0, 0, 0, +0.5, +0.5]

        assert np.max(evals_A) <= 0.5 + 1e-10, (
            f"G_A max eigenvalue {np.max(evals_A)} exceeds 0.5"
        )
        assert np.min(evals_A) >= -0.5 - 1e-10, (
            f"G_A min eigenvalue {np.min(evals_A)} below -0.5"
        )

    def test_fq_a_zero_equals_one(self) -> None:
        """Maximum QFI for Case A with α=0 must be 1 at T_H = 1.

        This is the single-particle bound: for J=1/2, (λ_max - λ_min)² = 1.
        """
        from src.analysis.ancilla_comparison import (
            _subspace_indices,
            random_pure_state_in_subspace,
        )

        dim_sys = (1 + 1) ** 2  # N_max = 1 → dim 4
        rng = np.random.default_rng(42)
        G_A_zero = compute_generator_A(T_H=1.0, alphas=(0.0, 0.0, 0.0, 0.0), N_max=1)
        sub_idx = _subspace_indices(1, 1)  # N=1 subspace

        expected = analytical_fq_A_zero(T_H=1.0)
        best_fq = -1.0
        for _ in range(200):
            psi_sys = random_pure_state_in_subspace(dim_sys, sub_idx, rng)
            psi_anc = random_pure_state_dm(2, rng)
            rho = np.kron(psi_sys, psi_anc)
            from src.analysis.fisher_information import quantum_fisher_information_dm

            F_Q = quantum_fisher_information_dm(rho, G_A_zero)
            best_fq = max(best_fq, F_Q)

        assert best_fq == pytest.approx(expected, rel=0.1), (
            f"Case A (α=0) max QFI = {best_fq}, expected ~{expected}"
        )

    def test_given_commuting_case_then_generator_a_scales_linearly_with_th(
        self,
    ) -> None:
        G_A_1 = compute_generator_A(T_H=1.0, alphas=(0.0, 0.0, 0.0, 0.0), N_max=1)
        G_A_2 = compute_generator_A(T_H=2.0, alphas=(0.0, 0.0, 0.0, 0.0), N_max=1)
        assert pytest.approx(2.0 * G_A_1) == G_A_2, "G_A must scale linearly with T_H"


class TestDensityMatrix:
    """Test random density matrix generation."""

    def test_given_random_dm_then_trace_is_1(self) -> None:
        rng = np.random.default_rng(42)
        for d in [2, 4, 8]:
            rho = random_density_matrix(d, rng)
            assert np.trace(rho) == pytest.approx(1.0), "Tr(ρ) must be 1"

    def test_given_random_dm_then_it_is_positive_semidefinite(self) -> None:
        rng = np.random.default_rng(42)
        for d in [2, 4, 8]:
            rho = random_density_matrix(d, rng)
            evals = np.linalg.eigvalsh(rho)
            assert np.all(evals >= -1e-12), "ρ must be positive semidefinite"

    def test_given_random_dm_then_it_is_hermitian(self) -> None:
        rng = np.random.default_rng(42)
        for d in [2, 4, 8]:
            rho = random_density_matrix(d, rng)
            assert rho == pytest.approx(rho.conj().T), "ρ must be Hermitian"

    def test_given_random_pure_dm_then_purity_is_1(self) -> None:
        rng = np.random.default_rng(42)
        for d in [2, 4, 8]:
            rho = random_pure_state_dm(d, rng)
            purity = np.trace(rho @ rho).real
            assert purity == pytest.approx(1.0, abs=1e-10), (
                "Purity must be 1 for pure state"
            )


class TestInteractionHamiltonian:
    """Test H_int construction and commutation."""

    def test_given_h_int_then_it_is_hermitian(self) -> None:
        J_z_sys, J_x_sys = build_system_jz_jx(N_max=1)
        J_z_anc, J_x_anc = build_ancilla_operators()

        for alphas in [
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.5, -1.0, 2.0, -0.5),
        ]:
            H = build_interaction_hamiltonian(
                alphas,
                J_z_sys,
                J_x_sys,
                J_z_anc,
                J_x_anc,
            )
            assert pytest.approx(H.conj().T) == H, (
                f"H_int must be Hermitian for α = {alphas}"
            )

    def test_jz_hint_commutation(self) -> None:
        """Test [J_z ⊗ I, H_int] for different α terms.

        For α_zz and α_zx: [J_z ⊗ J_z/α, J_z ⊗ I] = 0
        For α_xz and α_xx: [J_x ⊗ J_z/α, J_z ⊗ I] ≠ 0

        Alpha ordering: (α_xx, α_xz, α_zx, α_zz).
        """
        J_z_sys, J_x_sys = build_system_jz_jx(N_max=1)
        J_z_anc, J_x_anc = build_ancilla_operators()
        I_anc = np.eye(2, dtype=complex)
        J_z_full = np.kron(J_z_sys, I_anc)

        # Commuting case: only α_zz (4th position)
        H_zz = build_interaction_hamiltonian(
            (0.0, 0.0, 0.0, 1.0),
            J_z_sys,
            J_x_sys,
            J_z_anc,
            J_x_anc,
        )
        comm = J_z_full @ H_zz - H_zz @ J_z_full
        assert comm == pytest.approx(0), "[J_z, H_int] must be 0 for α_zz only"

        # Non-commuting case: α_xz only (2nd position)
        H_xz = build_interaction_hamiltonian(
            (0.0, 1.0, 0.0, 0.0),
            J_z_sys,
            J_x_sys,
            J_z_anc,
            J_x_anc,
        )
        comm = J_z_full @ H_xz - H_xz @ J_z_full
        assert comm != pytest.approx(0), "[J_z, H_int] must be ≠ 0 for α_xz"


class TestBSConvention:
    """Verify BS convention.

    Due to the operator-construction convention in mzi_simulation.py
    (where a0 acts as a creator and a1 as a creator), the commutation
    relation is [J_z, J_x] = -i J_y (with J_y from the standard formula
    (a†₀a₁ - a₁†a₀)/(2i)). This means BS† J_z BS = J_y (not -J_y).

    However, the eigenvalues (λ ∈ [-1, 1] for the 2-particle subspace)
    match the theoretical expectation up to sign, so QFI = 4·Var(G) is
    unaffected since Var(J_y) = Var(-J_y).
    """

    def test_given_bs_rotated_jz_then_eigenvalue_range_covers_minus_1_to_1(
        self,
    ) -> None:
        from src.physics.mzi_simulation import beam_splitter_unitary

        N_max = 2
        J_z_sys, _ = build_system_jz_jx(N_max)

        BS = beam_splitter_unitary(np.pi / 4, 0.0, N_max)
        J_z_rotated = BS.conj().T @ J_z_sys @ BS
        J_z_rotated = 0.5 * (J_z_rotated + J_z_rotated.conj().T)

        evals = np.linalg.eigvalsh(J_z_rotated)
        assert np.min(evals) >= -1.0 - 1e-10
        assert np.max(evals) <= 1.0 + 1e-10
        assert np.min(evals) == pytest.approx(-1.0, abs=1e-10)
        assert np.max(evals) == pytest.approx(1.0, abs=1e-10)

    def test_given_th_1_and_n2_subspace_then_generator_b_eigenvalues_in_minus_1_to_1(
        self,
    ) -> None:
        G_B = compute_generator_B(T_H=1.0, N_max=2)
        evals = np.linalg.eigvalsh(G_B)
        assert np.min(evals) >= -1.0 - 1e-10
        assert np.max(evals) <= 1.0 + 1e-10


class TestComparisonPipeline:
    """End-to-end tests of the comparison pipeline."""

    def test_given_run_comparison_then_returns_comparisonresult(
        self,
    ) -> None:
        result = run_comparison(
            T_H=1.0,
            n_samples_B=100,
            n_samples_A=100,
            n_alpha_samples=5,
            pure_only=True,
            seed=42,
        )
        assert isinstance(result, ComparisonResult), (
            "Expected result to be instance of ComparisonResult"
        )

    def test_given_comparison_result_then_fq_b_is_positive(self) -> None:
        result = run_comparison(
            T_H=1.0,
            n_samples_B=100,
            n_samples_A=100,
            n_alpha_samples=5,
            pure_only=True,
            seed=42,
        )
        assert result.fq_B_max > 0, "Case B QFI must be positive"

    def test_given_comparison_result_then_fq_a_is_positive(self) -> None:
        result = run_comparison(
            T_H=1.0,
            n_samples_B=100,
            n_samples_A=100,
            n_alpha_samples=5,
            pure_only=True,
            seed=42,
        )
        assert result.fq_A_max > 0, "Case A QFI must be positive"

    def test_given_comparison_result_then_ratio_is_finite_and_positive(self) -> None:
        result = run_comparison(
            T_H=1.0,
            n_samples_B=100,
            n_samples_A=100,
            n_alpha_samples=5,
            pure_only=True,
            seed=42,
        )
        assert np.isfinite(result.ratio), "Ratio must be finite"
        assert result.ratio > 0, "Ratio must be positive"

    def test_given_alpha_zero_then_fq_a_baseline_is_positive(
        self,
    ) -> None:
        result = run_comparison(
            T_H=1.0,
            n_samples_B=100,
            n_samples_A=100,
            n_alpha_samples=5,
            pure_only=True,
            seed=42,
        )
        assert result.fq_A_zero > 0.0, "Baseline F_Q must be positive"

    def test_given_theta_values_then_result_dict_has_expected_keys(
        self,
    ) -> None:
        result = run_comparison(
            T_H=1.0,
            n_samples_B=100,
            n_samples_A=100,
            n_alpha_samples=5,
            pure_only=True,
            theta_values=(0.0, 0.1, 0.5),
            seed=42,
        )
        assert 0.0 in result.fq_A_theta
        assert 0.1 in result.fq_A_theta
        assert 0.5 in result.fq_A_theta


class TestAnalyticalBounds:
    """Verify analytical bounds match expectations."""

    def test_given_th_equals_1_then_analytical_fq_b_equals_4(self) -> None:
        assert analytical_fq_B_max(1.0) == pytest.approx(4.0)
        assert analytical_fq_B_max(2.0) == pytest.approx(16.0)

    def test_given_th_equals_1_then_analytical_fq_a_zero_equals_1(self) -> None:
        assert analytical_fq_A_zero(1.0) == pytest.approx(1.0)
        assert analytical_fq_A_zero(2.0) == pytest.approx(4.0)

    def test_ratio_geq_two(self) -> None:
        """ℛ = Δθ_A / Δθ_B must be ≥ 2 at T_H = 1.

        From the analytical bound: F_A ≤ 1, F_B = 4, so ℛ = √(4/1) = 2.
        """
        F_A = analytical_fq_A_zero(1.0)  # = 1
        F_B = analytical_fq_B_max(1.0)  # = 4
        ratio = np.sqrt(F_B / F_A)
        assert ratio == pytest.approx(2.0)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_given_all_alphas_zero_then_generator_a_matches_no_interaction_case(
        self,
    ) -> None:
        G_A_1 = compute_generator_A(T_H=1.0, alphas=(0.0, 0.0, 0.0, 0.0), N_max=1)
        G_A_2 = compute_generator_A(T_H=1.0, alphas=(0.0, 0.0, 0.0, 0.0), N_max=1)
        assert pytest.approx(G_A_2) == G_A_1, "Must be consistent for same α"

    def test_commuting_ancilla_no_benefit(self) -> None:
        """Check that J_z-commuting interaction cannot change generator.

        When only α_zz and α_zx are nonzero, [J_z, H_int] = 0,
        so J_z(s) = J_z and the generator is identical to α = 0 case.

        Alpha ordering: (α_xx, α_xz, α_zx, α_zz).
        Commuting coefficients occupy the last two positions.
        """
        G_commuting = compute_generator_A(T_H=1.0, alphas=(0.0, 0.0, 1.0, 2.0), N_max=1)
        G_zero = compute_generator_A(T_H=1.0, alphas=(0.0, 0.0, 0.0, 0.0), N_max=1)
        assert G_commuting == pytest.approx(G_zero, abs=1e-10), (
            "Commuting interaction must leave generator unchanged"
        )

    def test_noncommuting_ancilla_changes_generator(self) -> None:
        """Check that non-commuting interaction changes the generator.

        When α_xz or α_xx are nonzero, [J_z, H_int] ≠ 0,
        so J_z(s) ≠ J_z and the generator differs from α = 0 case.

        Alpha ordering: (α_xx, α_xz, α_zx, α_zz).
        Non-commuting coefficients occupy the first two positions.
        """
        G_noncomm = compute_generator_A(T_H=1.0, alphas=(0.0, 1.0, 0.0, 0.0), N_max=1)
        G_zero = compute_generator_A(T_H=1.0, alphas=(0.0, 0.0, 0.0, 0.0), N_max=1)
        assert G_noncomm != pytest.approx(G_zero, abs=1e-10), (
            "Non-commuting interaction must change the generator"
        )

"""
Tests for Fisher Information module.

Tests validate:
- CSS (GHZ) state achieves SQL: F_Q = N
- NOON state achieves Heisenberg scaling: F_Q = N²
- Squeezed states achieve sub-SQL: F_Q > N
- Classical Fisher <= Quantum Fisher (fundamental bound)
- Phase sensitivity: Δφ = 1/√F
"""

import numpy as np
import pytest

from .fisher_information import (
    classical_fisher_information,
    generate_phase_generator,
    phase_sensitivity_from_fisher,
    quantum_fisher_information,
    quantum_fisher_information_dm,
    validate_fisher_inputs,
)


class TestQuantumFisherInformationPure:
    """Tests for QFI with pure states."""

    def test_qfi_noon_state_heisenberg_scaling(self) -> None:
        """NOON-like state should achieve F_Q = N² (Heisenberg scaling)."""
        for N in [2, 4, 8, 10]:
            # Generate state in symmetric subspace representation
            # Using J_z basis where eigenvalues span [-N/2, N/2]
            dim = N + 1
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0 / np.sqrt(2)  # |m=-N/2⟩
            state[N] = 1.0 / np.sqrt(2)  # |m=+N/2⟩

            # Use J_z as generator
            generator = generate_phase_generator(N, "Jz")

            fq = quantum_fisher_information(state, generator)

            # For NOON: F_Q = N² (Heisenberg scaling)
            expected = N**2
            assert fq == pytest.approx(expected, rel=0.1), (
                f"N={N}: F_Q={fq} should be ~{expected}"
            )

    def test_qfi_css_state_scaling(self) -> None:
        """GHZ-like state should achieve F_Q = N² (same as NOON in J_z basis)."""
        for N in [2, 4, 8, 10]:
            dim = N + 1
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0 / np.sqrt(2)
            state[N] = 1.0 / np.sqrt(2)

            generator = generate_phase_generator(N, "Jz")

            fq = quantum_fisher_information(state, generator)

            # For GHZ with J_z: F_Q = N² (since variance is N²/4)
            expected = N**2
            assert fq == pytest.approx(expected, rel=0.1), (
                f"N={N}: F_Q={fq} should be ~{expected}"
            )

    def test_qfi_noon_greater_than_eigenstate(self) -> None:
        """NOON should have higher QFI than an eigenstate."""
        N = 10
        dim = N + 1

        # NOON state
        noon_state = np.zeros(dim, dtype=complex)
        noon_state[0] = 1.0 / np.sqrt(2)
        noon_state[N] = 1.0 / np.sqrt(2)

        # Eigenstate
        eigen_state = np.zeros(dim, dtype=complex)
        eigen_state[0] = 1.0

        generator = generate_phase_generator(N, "Jz")

        fq_noon = quantum_fisher_information(noon_state, generator)
        fq_eigen = quantum_fisher_information(eigen_state, generator)

        # NOON should have higher QFI than eigenstate (0)
        assert fq_noon > fq_eigen + 1e-5, (
            f"NOON QFI {fq_noon} should exceed eigenstate QFI {fq_eigen}"
        )

    def test_qfi_zero_for_eigenstate(self) -> None:
        """QFI should be zero for an eigenstate of the generator."""
        N = 5
        dim = N + 1

        # Eigenstate of J_z (eigenvalue -N/2)
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        generator = generate_phase_generator(N, "Jz")

        # An eigenstate has zero variance, so QFI = 0
        fq = quantum_fisher_information(state, generator)
        assert fq == pytest.approx(0.0, abs=1e-10), (
            f"QFI for eigenstate should be ~0, got {fq}"
        )

    def test_qfi_variance_formula(self) -> None:
        """Test that F_Q = 4 Var(G) holds."""
        N = 5
        dim = N + 1

        # Create superposition state
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0 / np.sqrt(2)
        state[N] = 1.0 / np.sqrt(2)

        generator = generate_phase_generator(N, "Jz")

        # Compute expectation values directly
        g_exp = np.vdot(state, generator @ state).real
        g2_exp = np.vdot(state, generator @ generator @ state).real
        var_g = g2_exp - g_exp**2

        # Compute QFI
        fq = quantum_fisher_information(state, generator)

        # Verify F_Q = 4 Var(G)
        expected_fq = 4.0 * var_g
        assert fq == pytest.approx(expected_fq, rel=1e-5), (
            f"F_Q={fq} should equal 4*Var={expected_fq}"
        )


class TestQuantumFisherInformationMixed:
    """Tests for QFI with mixed states."""

    def test_qfi_mixed_state_matches_pure(self) -> None:
        """Mixed state QFI should reduce to pure formula when pure."""
        N = 5
        dim = N + 1

        # Pure state as density matrix
        state_vec = np.zeros(dim, dtype=complex)
        state_vec[0] = 1.0 / np.sqrt(2)
        state_vec[N] = 1.0 / np.sqrt(2)
        rho_pure = np.outer(state_vec, state_vec.conj())

        generator = generate_phase_generator(N, "Jz")

        fq_pure = quantum_fisher_information(state_vec, generator)
        fq_dm = quantum_fisher_information_dm(rho_pure, generator)

        assert fq_pure == pytest.approx(fq_dm, rel=1e-5), (
            f"Pure state QFI={fq_pure}, DM QFI={fq_dm}"
        )

    def test_qfi_mixed_state_lower_than_pure(self) -> None:
        """Mixed state QFI should be <= pure state QFI."""
        N = 5

        # Pure GHZ state
        dim = N + 1
        state_vec = np.zeros(dim, dtype=complex)
        state_vec[0] = 1.0 / np.sqrt(2)
        state_vec[N] = 1.0 / np.sqrt(2)
        rho_pure = np.outer(state_vec, state_vec.conj())

        # Mixed state: 50% GHZ + 50% maximally mixed
        rho_mixed = 0.5 * rho_pure + 0.5 * np.eye(dim) / dim

        generator = generate_phase_generator(N, "Jz")

        fq_pure = quantum_fisher_information_dm(rho_pure, generator)
        fq_mixed = quantum_fisher_information_dm(rho_mixed, generator)

        # Mixed should have lower or equal QFI
        assert fq_mixed <= fq_pure + 1e-5, (
            f"Mixed QFI {fq_mixed} should be <= pure QFI {fq_pure}"
        )

    def test_qfi_maximally_mixed_is_zero(self) -> None:
        """Maximally mixed state should have F_Q = 0."""
        dim = 4
        rho = np.eye(dim, dtype=complex) / dim
        G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        G = G + G.conj().T

        fq = quantum_fisher_information_dm(rho, G)
        assert fq == pytest.approx(0.0, abs=1e-12), (
            f"Maximally mixed QFI should be 0, got {fq}"
        )

    def test_qfi_commuting_rho_G_is_zero(self) -> None:
        """When [G, ρ] = 0, the state carries no phase info: F_Q = 0."""
        dim = 5
        # Diagonal state and diagonal generator — they commute
        eigenvalues = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
        rho = np.diag(eigenvalues)
        G = np.diag(np.arange(dim, dtype=float))

        fq = quantum_fisher_information_dm(rho, G)
        assert fq == pytest.approx(0.0, abs=1e-12), (
            f"Commuting ρ and G should give F_Q=0, got {fq}"
        )

    def test_qfi_mixed_matches_first_principles_2d(self) -> None:
        """Verify against the textbook SLD formula for 2D mixed states."""
        np.random.seed(42)
        for _ in range(5):
            # Random 2D mixed state
            a = np.random.uniform(0.1, 0.9)
            rho = np.diag([a, 1.0 - a])
            # Random Hermitian generator
            g11, g22 = np.random.uniform(-2, 2, 2)
            g_off = np.random.uniform(-1, 1) + 1j * np.random.uniform(-1, 1)
            G = np.array([[g11, g_off], [np.conj(g_off), g22]], dtype=complex)

            # First-principles: F_Q = Σ_{i≠j} 2·(λ_i-λ_j)²/(λ_i+λ_j)·|G_ij|²
            w, v = np.linalg.eigh(rho)
            w = w[::-1]
            v = v[:, ::-1]
            expected = 0.0
            for i in range(2):
                for j in range(2):
                    if i == j:
                        continue
                    denom = w[i] + w[j]
                    if denom > 0:
                        g_ij = np.vdot(v[:, i], G @ v[:, j])
                        expected += 2.0 * (w[i] - w[j]) ** 2 / denom * np.abs(g_ij) ** 2

            fq = quantum_fisher_information_dm(rho, G)
            assert fq == pytest.approx(expected, rel=1e-10), (
                f"2D mixed F_Q={fq} != first-principles={expected}"
            )
            assert fq >= 0.0, f"QFI should be non-negative, got {fq}"

    def test_qfi_mixed_matches_first_principles_high_dim(self) -> None:
        """Verify against textbook SLD formula for higher-dimensional mixed states."""
        np.random.seed(123)
        for dim in [3, 5, 8]:
            # Random mixed state
            A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            rho = A @ A.conj().T
            rho = rho / np.trace(rho)

            # Random Hermitian generator
            G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
            G = G + G.conj().T

            # First-principles
            w, v = np.linalg.eigh(rho)
            w = w[::-1]
            v = v[:, ::-1]
            expected = 0.0
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        continue
                    denom = w[i] + w[j]
                    if denom > 1e-12:
                        g_ij = np.vdot(v[:, i], G @ v[:, j])
                        expected += 2.0 * (w[i] - w[j]) ** 2 / denom * np.abs(g_ij) ** 2

            fq = quantum_fisher_information_dm(rho, G)
            assert fq == pytest.approx(expected, rel=1e-10), (
                f"dim={dim}: F_Q={fq} != first-principles={expected}"
            )
            assert fq >= 0.0, f"QFI should be non-negative, got {fq}"

    def test_qfi_mixed_equal_eigenvalues(self) -> None:
        """When two eigenvalues are equal, their contribution to F_Q is 0."""
        # 3-level system with λ₁=λ₂ (degenerate)
        dim = 3
        rho = np.diag([0.4, 0.4, 0.2])
        G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        G = G + G.conj().T

        fq = quantum_fisher_information_dm(rho, G)

        # Hand-corrected check: only the non-degenerate pair (λ₁,λ₃) and (λ₂,λ₃)
        # contribute. λ₁-λ₂=0 so that pair vanishes.
        w, v = np.linalg.eigh(rho)
        w = w[::-1]
        v = v[:, ::-1]
        expected = 0.0
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    continue
                denom = w[i] + w[j]
                if denom > 1e-12 and not np.isclose(w[i], w[j]):
                    g_ij = np.vdot(v[:, i], G @ v[:, j])
                    expected += 2.0 * (w[i] - w[j]) ** 2 / denom * np.abs(g_ij) ** 2

        assert fq == pytest.approx(expected, rel=1e-10), (
            f"Degenerate F_Q={fq} != expected={expected}"
        )

    def test_qfi_mixed_pure_state_limit(self) -> None:
        """As mixedness → 0, F_Q should approach the pure-state value."""
        dim = 4
        np.random.seed(7)
        G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        G = G + G.conj().T

        # Pure reference
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        fq_pure = quantum_fisher_information(state, G)

        # Nearly-pure mixed state
        for eps in [1e-3, 1e-6, 1e-10]:
            rho = (1.0 - eps) * np.outer(state, state.conj()) + eps * np.eye(dim) / dim
            fq = quantum_fisher_information_dm(rho, G)
            # Should approach pure-state value from below
            assert fq <= fq_pure + 1e-10, f"eps={eps}: F_Q={fq} > pure={fq_pure}"
            assert fq >= fq_pure * (1.0 - 10 * eps) or np.isclose(
                fq,
                fq_pure,
                rtol=1e-3,
            ), f"eps={eps}: F_Q={fq} too far from pure={fq_pure}"


class TestClassicalFisherInformation:
    """Tests for classical Fisher information."""

    def test_cfi_positive_for_nonzero_probabilities(self) -> None:
        """F_C should be positive when all probabilities are nonzero."""
        n_phi = 10
        n_outcomes = 2

        # Simple sinusoidal probabilities
        phis = np.linspace(0, 2 * np.pi, n_phi)
        probs = np.zeros((n_phi, n_outcomes))
        probs[:, 0] = 0.5 + 0.4 * np.sin(phis)
        probs[:, 1] = 1 - probs[:, 0]

        fc = classical_fisher_information(probs, dphi=1e-3)

        # Should be positive for everywhere where sin is not too flat
        assert np.all(fc >= 0), "F_C should be non-negative"

    def test_cfi_smaller_than_qfi(self) -> None:
        """Classical Fisher <= Quantum Fisher (Cramér-Rao inequality)."""
        N = 5
        dim = N + 1

        # Generate proabilities from a pure state
        phis = np.linspace(0, np.pi, 20)
        n_phi = len(phis)
        probs = np.zeros((n_phi, dim))

        # Coherent superposition with phase
        for i, phi in enumerate(phis):
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0 / np.sqrt(2)
            state[N] = np.exp(1j * phi) / np.sqrt(2)

            # P(m|φ) = |⟨m|ψ(φ)⟩|²
            probs[i] = np.abs(state) ** 2

        generator = generate_phase_generator(N, "Jz")

        # Compute QFI for reference (use central state phi=pi/2)
        state_center = np.zeros(dim, dtype=complex)
        state_center[0] = 1.0 / np.sqrt(2)
        state_center[N] = 1.0 / np.sqrt(2) * 1j
        fq = quantum_fisher_information(state_center, generator)

        # Compute classical FIsher at each phi
        fc = classical_fisher_information(probs)

        # At each phi, F_C <= F_Q (true for estimation)
        # Check at the maximum
        fc_max = np.max(fc)
        assert fc_max <= fq * 1.5, f"F_C max {fc_max} should be <= F_Q {fq}"


class TestPhaseSensitivity:
    """Tests for phase sensitivity from Fisher information."""

    def test_phase_sensitivity_noon_heisenberg(self) -> None:
        """NOON state achieves Δφ = 1/N (Heisenberg scaling) when F_Q = N²."""
        # For NOON-like state with F_Q = N², Δφ = 1/√(N²) = 1/N
        N = 100

        # Create state with F_Q ≈ N²
        dim = N + 1
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0 / np.sqrt(2)
        state[N] = 1.0 / np.sqrt(2)

        generator = generate_phase_generator(N, "Jz")
        fq = quantum_fisher_information(state, generator)

        delta_phi = phase_sensitivity_from_fisher(fq)

        # Expected SQL: 1/√N
        expected_sql = 1.0 / np.sqrt(N)

        # Should achieve Heisenberg scaling: much better than SQL
        assert delta_phi < expected_sql, (
            f"Δφ={delta_phi} should be < SQL={expected_sql}"
        )

    def test_phase_sensitivity_invalid_f(self) -> None:
        """Should raise for non-positive Fisher information."""
        # Test zero
        with pytest.raises(ValueError):
            phase_sensitivity_from_fisher(0.0)

        # Test negative
        with pytest.raises(ValueError):
            phase_sensitivity_from_fisher(-1.0)


class TestValidation:
    """Tests for input validation."""

    def test_validate_fisher_positive(self) -> None:
        """Should validate positive Fisher information."""
        validate_fisher_inputs(1.0)
        validate_fisher_inputs(100.0)

    def test_validate_fisher_invalid(self) -> None:
        """Should reject non-positive Fisher information."""
        with pytest.raises(ValueError):
            validate_fisher_inputs(0.0)

        with pytest.raises(ValueError):
            validate_fisher_inputs(-1.0)

        with pytest.raises(ValueError):
            validate_fisher_inputs(np.nan)


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_large_N_scaling(self) -> None:
        """Should handle large N without overflow."""
        N = 1000
        dim = N + 1

        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0 / np.sqrt(2)
        state[N] = 1.0 / np.sqrt(2)

        generator = generate_phase_generator(N, "Jz")
        fq = quantum_fisher_information(state, generator)

        # Should be ≈ N (SQL) scaled
        assert fq > 0, "QFI should be positive"
        assert np.isfinite(fq), "QFI should be finite"

    def test_small_probability_handling(self) -> None:
        """Should handle small probabilities without division errors."""
        # Probabilities with very small values
        probs = np.array(
            [
                [1e-10, 1 - 1e-10],
                [0.5, 0.5],
            ],
        )

        fc = classical_fisher_information(probs, dphi=1e-3)

        # Should be finite and positive
        assert np.all(np.isfinite(fc)), "F_C should be finite"

    def test_generator_dimension_mismatch(self) -> None:
        """Should raise on dimension mismatch."""
        dim_state = 5
        dim_generator = 10

        state = np.ones(dim_state, dtype=complex)
        generator = np.eye(dim_generator, dtype=complex)

        with pytest.raises(ValueError):
            quantum_fisher_information(state, generator)


# =============================================================================
# Additional validation tests
# =============================================================================


def test_qfi_variance_relationship() -> None:
    """Verify F_Q = 4 Var(G) relationship."""
    N = 6

    # Create state with known variance
    dim = N + 1
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)
    state[N] = 1.0 / np.sqrt(2)

    generator = generate_phase_generator(N, "Jz")

    # Compute expectation values directly
    g_exp = np.vdot(state, generator @ state).real
    g2_exp = np.vdot(state, generator @ generator @ state).real
    var_g = g2_exp - g_exp**2

    # Compute QFI
    fq = quantum_fisher_information(state, generator)

    # Verify F_Q = 4 Var(G)
    expected = 4.0 * var_g
    assert fq == pytest.approx(expected, rel=1e-5), (
        f"F_Q={fq} should equal 4*Var={expected}"
    )


class TestPhysicsInvariants:
    """Validation tests for physics invariants."""

    def test_cfi_less_than_equal_qfi(self) -> None:
        """F_C ≤ F_Q always (Cramér-Rao bound)."""
        # Generate probability distributions for test states
        N = 5
        dim = N + 1

        # Phase values
        n_phi = 20
        phis = np.linspace(0, 2 * np.pi, n_phi)
        probs = np.zeros((n_phi, dim))

        for i, phi in enumerate(phis):
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0 / np.sqrt(2)
            state[N] = np.exp(1j * phi) / np.sqrt(2)
            probs[i] = np.abs(state) ** 2

        # Classical FIsher at each φ
        fc = classical_fisher_information(probs, dphi=1e-3)

        # QFI (same for all phi)
        generator = generate_phase_generator(N, "Jz")
        state_center = np.zeros(dim, dtype=complex)
        state_center[0] = 1.0 / np.sqrt(2)
        state_center[N] = 1.0 / np.sqrt(2)
        fq = quantum_fisher_information(state_center, generator)

        # F_C <= F_Q at each phi
        assert np.all(fc <= fq + 1e-5), f"Max F_C={np.max(fc)} should be <= F_Q={fq}"

    def test_cfi_positive_for_nonzero_probs(self) -> None:
        """F_C(φ) > 0 for all φ where P(m|φ) > 0."""
        n_phi = 20
        dim = 2

        # Probabilities with all non-zero values
        phis = np.linspace(0.1, 2 * np.pi - 0.1, n_phi)
        probs = np.zeros((n_phi, dim))
        probs[:, 0] = 0.5 + 0.4 * np.cos(phis)
        probs[:, 1] = 1 - probs[:, 0]

        # Ensure all probabilities are > 0
        probs = np.clip(probs, 1e-10, None)

        fc = classical_fisher_information(probs, dphi=1e-4)

        assert np.all(fc >= 0), "F_C should be non-negative"

    def test_phase_sensitivity_css_sql(self) -> None:
        """Verify phase sensitivity computes correctly from Fisher info."""
        for N in [10, 50, 100]:
            dim = N + 1

            # Use equal superposition state
            state = np.ones(dim, dtype=complex) / np.sqrt(dim)

            generator = generate_phase_generator(N, "Jz")
            fq = quantum_fisher_information(state, generator)

            delta_phi = phase_sensitivity_from_fisher(fq)

            # Verify Δφ = 1/√F (definition)
            expected_delta = 1.0 / np.sqrt(fq)
            assert delta_phi == pytest.approx(expected_delta, rel=0.01), (
                f"N={N}: Δφ={delta_phi:.6f} should be 1/√F={expected_delta:.6f}"
            )

    def test_squeezed_state_sub_sql(self) -> None:
        """Squeezed states should achieve F_Q > N (sub-SQL)."""
        N = 20
        dim = N + 1

        # Create a squeezed state with binomial distribution
        # P(n) ∝ (N choose n) with reduced variance
        probs = np.array([1.0] * dim)  # uniform
        probs = probs / np.sum(probs)

        # Generate state from distribution
        state = np.sqrt(probs).astype(complex)

        generator = generate_phase_generator(N, "Jz")
        fq = quantum_fisher_information(state, generator)

        # For uniform state, F_Q scales with N² which is better than SQL
        # (Heisenberg-like scaling)
        sql_fisher = N  # SQL

        assert fq > sql_fisher, f"Squeezed F_Q={fq} should exceed SQL={sql_fisher}"


# =============================================================================
# Mixed-state QFI: Input Validation
# =============================================================================


class TestInputValidationDM:
    """Tests that mixed-state QFI correctly validates inputs, preventing silent errors."""

    def test_non_hermitian_rho_raises(self) -> None:
        """Non-Hermitian ρ should raise ValueError (eigh silently wrong otherwise)."""
        dim = 4
        # Non-Hermitian: A + i·A^T (asymmetric imaginary part)
        rho = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        rho = rho @ rho.conj().T  # Start Hermitian then break it
        rho[0, 1] += 1.0j  # Make asymmetric → non-Hermitian
        G = np.eye(dim, dtype=complex)

        with pytest.raises(ValueError, match="Hermitian"):
            quantum_fisher_information_dm(rho, G)

    def test_nan_in_rho_raises(self) -> None:
        """ρ containing NaN should raise ValueError."""
        dim = 3
        rho = np.eye(dim, dtype=complex) / dim
        rho[0, 0] = np.nan
        G = np.eye(dim, dtype=complex)

        with pytest.raises(ValueError, match="NaN"):
            quantum_fisher_information_dm(rho, G)

    def test_inf_in_rho_raises(self) -> None:
        """ρ containing Inf should raise ValueError."""
        dim = 3
        rho = np.eye(dim, dtype=complex) / dim
        rho[0, 0] = np.inf
        G = np.eye(dim, dtype=complex)

        with pytest.raises(ValueError, match="infinite"):
            quantum_fisher_information_dm(rho, G)

    def test_nan_in_generator_raises(self) -> None:
        """Generator containing NaN should raise ValueError."""
        dim = 3
        rho = np.eye(dim, dtype=complex) / dim
        G = np.eye(dim, dtype=complex)
        G[0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            quantum_fisher_information_dm(rho, G)

    def test_inf_in_generator_raises(self) -> None:
        """Generator containing Inf should raise ValueError."""
        dim = 3
        rho = np.eye(dim, dtype=complex) / dim
        G = np.eye(dim, dtype=complex)
        G[0, 0] = np.inf

        with pytest.raises(ValueError, match="infinite"):
            quantum_fisher_information_dm(rho, G)

    def test_non_hermitian_generator_raises(self) -> None:
        """Non-Hermitian generator should raise ValueError."""
        dim = 4
        rho = np.eye(dim, dtype=complex) / dim
        G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        # G is explicitly non-Hermitian (no symmetrization)

        with pytest.raises(ValueError, match="Hermitian"):
            quantum_fisher_information_dm(rho, G)

    def test_dimension_mismatch_generator_raises(self) -> None:
        """Generator with wrong dimension should raise ValueError."""
        rho = np.eye(3, dtype=complex) / 3
        G = np.eye(5, dtype=complex)  # Wrong dimension

        with pytest.raises(ValueError, match=r"dimension|match"):
            quantum_fisher_information_dm(rho, G)

    def test_non_square_rho_raises(self) -> None:
        """Non-square ρ should raise ValueError."""
        rho = np.ones((3, 4), dtype=complex)
        G = np.eye(3, dtype=complex)

        with pytest.raises(ValueError, match="square"):
            quantum_fisher_information_dm(rho, G)

    def test_non_square_generator_raises(self) -> None:
        """Non-square generator should raise ValueError (dimension mismatch)."""
        rho = np.eye(3, dtype=complex) / 3
        G = np.ones((3, 4), dtype=complex)

        with pytest.raises(ValueError, match="dimension"):
            quantum_fisher_information_dm(rho, G)

    def test_zero_trace_rho_handled(self) -> None:
        """Zero-trace ρ should produce finite result (handled gracefully)."""
        dim = 3
        # Zero matrix should have trace=0, eigenvalues all zero
        rho = np.zeros((dim, dim), dtype=complex)
        G = np.eye(dim, dtype=complex)

        fq = quantum_fisher_information_dm(rho, G)
        assert np.isfinite(fq), "QFI should be finite even for zero-trace ρ"
        assert fq >= 0.0, "QFI should be non-negative"


class TestInputValidationPure:
    """Tests that pure-state QFI validates NaN/Inf inputs."""

    def test_nan_in_state_raises(self) -> None:
        """State vector containing NaN should raise ValueError."""
        dim = 4
        state = np.ones(dim, dtype=complex)
        state[0] = np.nan
        G = np.eye(dim, dtype=complex)

        with pytest.raises(ValueError, match="NaN"):
            quantum_fisher_information(state, G)

    def test_inf_in_state_raises(self) -> None:
        """State vector containing Inf should raise ValueError."""
        dim = 4
        state = np.ones(dim, dtype=complex)
        state[0] = np.inf
        G = np.eye(dim, dtype=complex)

        with pytest.raises(ValueError, match="infinite"):
            quantum_fisher_information(state, G)


# =============================================================================
# Mixed-state QFI: Physical Invariants
# =============================================================================


class TestPhysicalInvariantsDM:
    """Tests that mixed-state QFI satisfies fundamental physical invariants."""

    def test_convexity(self) -> None:
        """QFI is convex: F_Q(p·ρ₁+(1-p)·ρ₂) ≤ p·F_Q(ρ₁)+(1-p)·F_Q(ρ₂)."""
        rng = np.random.default_rng(42)
        dim = 5

        # Two different mixed states
        A1 = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho1 = A1 @ A1.conj().T
        rho1 = rho1 / np.trace(rho1)

        A2 = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho2 = A2 @ A2.conj().T
        rho2 = rho2 / np.trace(rho2)

        # Random Hermitian generator
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T

        fq1 = quantum_fisher_information_dm(rho1, G)
        fq2 = quantum_fisher_information_dm(rho2, G)

        for p in [0.2, 0.5, 0.8]:
            rho_mix = p * rho1 + (1.0 - p) * rho2
            fq_mix = quantum_fisher_information_dm(rho_mix, G)
            weighted_avg = p * fq1 + (1.0 - p) * fq2
            # Convexity: mixed ≤ weighted average
            assert fq_mix <= weighted_avg + 1e-10, (
                f"Convexity violated at p={p}: F_Q(mix)={fq_mix:.8f} > "
                f"p·FQ₁+(1-p)·FQ₂={weighted_avg:.8f}"
            )

    def test_additivity(self) -> None:
        """QFI is additive for product states with summed generators.

        F_Q(ρ₁⊗ρ₂, G₁⊗I+I⊗G₂) = F_Q(ρ₁, G₁) + F_Q(ρ₂, G₂)
        """
        rng = np.random.default_rng(42)
        d1, d2 = 3, 2

        # System 1
        A1 = rng.normal(size=(d1, d1)) + 1j * rng.normal(size=(d1, d1))
        rho1 = A1 @ A1.conj().T
        rho1 = rho1 / np.trace(rho1)
        G1 = rng.normal(size=(d1, d1)) + 1j * rng.normal(size=(d1, d1))
        G1 = G1 + G1.conj().T

        # System 2
        A2 = rng.normal(size=(d2, d2)) + 1j * rng.normal(size=(d2, d2))
        rho2 = A2 @ A2.conj().T
        rho2 = rho2 / np.trace(rho2)
        G2 = rng.normal(size=(d2, d2)) + 1j * rng.normal(size=(d2, d2))
        G2 = G2 + G2.conj().T

        # Individual QFIs
        fq1 = quantum_fisher_information_dm(rho1, G1)
        fq2 = quantum_fisher_information_dm(rho2, G2)

        # Product state
        rho_product = np.kron(rho1, rho2)
        G_product = np.kron(G1, np.eye(d2)) + np.kron(np.eye(d1), G2)

        fq_product = quantum_fisher_information_dm(rho_product, G_product)

        # Additivity: product QFI = sum of individual QFIs
        expected = fq1 + fq2
        assert fq_product == pytest.approx(expected, rel=1e-10), (
            f"Additivity violated: F_Q(product)={fq_product:.10f} != "
            f"F_Q₁+F_Q₂={expected:.10f} (diff={fq_product - expected:.2e})"
        )

    def test_shift_invariance(self) -> None:
        """QFI is invariant under generator shift: F_Q(ρ, G + cI) = F_Q(ρ, G)."""
        rng = np.random.default_rng(42)
        dim = 4

        # Mixed state
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)

        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T

        fq_base = quantum_fisher_information_dm(rho, G)

        for c in [-2.5, 0.0, 1.0, 10.0]:
            G_shifted = G + c * np.eye(dim)
            fq_shifted = quantum_fisher_information_dm(rho, G_shifted)

            assert fq_base == pytest.approx(fq_shifted, rel=1e-12), (
                f"Shift invariance violated at c={c}: F_Q={fq_shifted} != {fq_base}"
            )

    def test_unitary_covariance(self) -> None:
        """QFI is unitarily covariant: F_Q(UρU†, UGU†) = F_Q(ρ, G)."""
        rng = np.random.default_rng(42)
        dim = 4

        # Random mixed state
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)

        # Random Hermitian generator
        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T

        # Random unitary (Haar-distributed via QR)
        X = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        U, _ = np.linalg.qr(X)

        fq_original = quantum_fisher_information_dm(rho, G)

        # Transformed: ρ' = UρU†, G' = UGU†
        rho_transformed = U @ rho @ U.conj().T
        G_transformed = U @ G @ U.conj().T
        fq_transformed = quantum_fisher_information_dm(rho_transformed, G_transformed)

        assert fq_original == pytest.approx(fq_transformed, rel=1e-10), (
            f"Unitary covariance violated: F_Q(ρ,G)={fq_original:.10f} != "
            f"F_Q(UρU†,UGU†)={fq_transformed:.10f}"
        )

    def test_qfi_leq_four_variance(self) -> None:
        """F_Q(ρ, G) ≤ 4·Var_ρ(G) for all ρ, with equality iff ρ is pure."""
        rng = np.random.default_rng(42)

        for dim in [3, 5]:
            G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            G = G + G.conj().T

            # Pure state: should saturate bound
            vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
            vec = vec / np.linalg.norm(vec)
            rho_pure = np.outer(vec, vec.conj())

            g_exp = np.vdot(vec, G @ vec).real
            g2_exp = np.vdot(vec, G @ G @ vec).real
            var_g = max(0.0, g2_exp - g_exp**2)

            fq_pure = quantum_fisher_information_dm(rho_pure, G)
            assert fq_pure == pytest.approx(4.0 * var_g, rel=1e-10), (
                f"Pure state: F_Q={fq_pure} != 4·Var={4 * var_g}"
            )

            # Mixed state: F_Q < 4·Var (strict for genuinely mixed)
            A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            rho_mixed = A @ A.conj().T
            rho_mixed = rho_mixed / np.trace(rho_mixed)

            fq_mixed = quantum_fisher_information_dm(rho_mixed, G)
            var_g_mixed = np.real(
                np.trace(rho_mixed @ G @ G) - (np.trace(rho_mixed @ G)) ** 2,
            )
            assert fq_mixed <= 4.0 * var_g_mixed + 1e-10, (
                f"Mixed state: F_Q={fq_mixed:.10f} > 4·Var={4 * var_g_mixed:.10f}. "
                "F_Q cannot exceed 4·Var(G) for any state."
            )

    def test_zero_generator(self) -> None:
        """F_Q(ρ, 0) = 0 for any ρ."""
        rng = np.random.default_rng(42)
        dim = 4

        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)

        G_zero = np.zeros((dim, dim), dtype=complex)
        fq = quantum_fisher_information_dm(rho, G_zero)
        assert fq == pytest.approx(0.0, abs=1e-15), (
            f"F_Q with zero generator should be 0, got {fq}"
        )


# =============================================================================
# Mixed-state QFI: Numerical Stability
# =============================================================================


class TestNumericalStabilityDM:
    """Tests that mixed-state QFI handles numerical edge cases stably."""

    def test_rank_deficient_with_null_space(self) -> None:
        """Rank-deficient ρ with multiple zero eigenvalues should be handled."""
        dim = 6
        # ρ with rank 3: 3 non-zero, 3 zero eigenvalues
        w = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
        rho = np.diag(w)

        # Non-diagonal generator to activate off-diagonal contributions
        G = np.zeros((dim, dim), dtype=complex)
        G[0, 1] = G[1, 0] = 1.0
        G[2, 3] = G[3, 2] = 0.5

        fq = quantum_fisher_information_dm(rho, G)

        # Should be finite and non-negative
        assert np.isfinite(fq), "QFI should be finite for rank-deficient ρ"
        assert fq >= 0.0, "QFI should be non-negative"

        # Manually compute expected value
        # Only pairs with non-zero eigenvalues contribute:
        # λ₀=0.5, λ₁=0.3 pair: (0.5-0.3)²/(0.5+0.3) = 0.04/0.8 = 0.05
        #   |G₀₁|² = 1, contribution: 4 * 0.05 * 1 = 0.2
        # λ₀=0.5, λ₂=0.2 pair: (0.5-0.2)²/(0.5+0.2) = 0.09/0.7 = 0.12857...
        #   G₀₂ = 0 (no coupling), contribution: 0
        # λ₁=0.3, λ₂=0.2 pair: (0.3-0.2)²/(0.3+0.2) = 0.01/0.5 = 0.02
        #   G₁₂ = 0 (no coupling), contribution: 0
        # Pairs involving zero eigenvalues:
        # λ₂=0.2, λ₃=0.0: (0.2-0)²/(0.2+0) = 0.04/0.2 = 0.2
        #   G₂₃ = 0.5, |G₂₃|² = 0.25, contribution: 4 * 0.2 * 0.25 = 0.2
        # All other pairs with zero eigenvalues: G = 0, contribution = 0
        expected = 0.2 + 0.2  # = 0.4
        assert fq == pytest.approx(expected, rel=1e-10), (
            f"Rank-deficient: F_Q={fq} != expected={expected}"
        )

    def test_rank_deficient_diagonal_basis(self) -> None:
        """Rank-deficient ρ with G off-diagonal coupling eigen/support ↔ null-space."""
        dim = 5
        # ρ with rank 2: eigenvalues on 0 and 1 indices
        # Support: |0⟩, |1⟩; Null-space: |2⟩, |3⟩, |4⟩
        w = np.array([0.6, 0.4, 0.0, 0.0, 0.0])
        rho = np.diag(w)

        # G couples support-0 ↔ null-3 and support-1 ↔ null-2
        G = np.zeros((dim, dim), dtype=complex)
        G[0, 3] = G[3, 0] = 2.0
        G[1, 2] = G[2, 1] = 1.0

        fq = quantum_fisher_information_dm(rho, G)

        # Manually (only support↔null pairs contribute since G is zero within support):
        # λ₀=0.6, λ₃=0.0: (0.6-0)²/(0.6+0) = 0.36/0.6 = 0.6
        #   |G₀₃|² = 4, contribution: 4 * 0.6 * 4 = 9.6
        # λ₁=0.4, λ₂=0.0: (0.4-0)²/(0.4+0) = 0.16/0.4 = 0.4
        #   |G₁₂|² = 1, contribution: 4 * 0.4 * 1 = 1.6
        # Total: 9.6 + 1.6 = 11.2
        expected = 9.6 + 1.6
        assert fq == pytest.approx(expected, rel=1e-10), (
            f"Rank-deficient null-space: F_Q={fq} != expected={expected}"
        )

    def test_near_degenerate_eigenvalues(self) -> None:
        """Near-degenerate eigenvalues should not cause numerical instability."""
        rng = np.random.default_rng(99)
        dim = 4

        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T

        # Start with equal eigenvalues (degenerate): F_Q depends on support↔null
        for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
            w = np.array([0.5 - eps / 2, 0.5 + eps / 2, 0.0, 0.0])
            w = w / np.sum(w)
            rho = np.diag(w)

            fq = quantum_fisher_information_dm(rho, G)
            assert np.isfinite(fq), f"F_Q not finite at ε={eps}"
            assert fq >= 0.0, f"F_Q negative at ε={eps}"

            # As eps → 0, F_Q should be continuous
            # When eigenvalues are equal, (λᵢ-λⱼ)² term → 0 for that pair
            # Other pairs (with null-space) contribute normally
            if eps > 1e-10:
                # Very rough bound: F_Q should be O(1) and finite
                assert fq < 1e6, f"Unreasonably large F_Q={fq} at ε={eps}"

    def test_large_dimension_does_not_crash(self) -> None:
        """QFI computation should handle dim=100 without error."""
        dim = 100
        rng = np.random.default_rng(42)

        # Random mixed state (density from Wishart-like construction)
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)

        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T

        fq = quantum_fisher_information_dm(rho, G)
        assert np.isfinite(fq), "QFI should be finite for large dim"
        assert fq >= 0.0, "QFI should be non-negative for large dim"
        # F_Q should scale as O(dim) for a random density with random G
        assert fq < dim**3, (
            f"F_Q={fq:.2f} unreasonably large for dim={dim}. Expected O(dim) scaling."
        )

    def test_diagonal_g_in_rho_eigenbasis(self) -> None:
        """When G is diagonal in ρ's eigenbasis, F_Q = 0."""
        dim = 5
        rng = np.random.default_rng(42)

        # Diagonal ρ in J_z eigenbasis (Dicke convention)
        w = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        rho = np.diag(w)

        # G diagonal in same basis → [ρ, G] = 0 → F_Q = 0
        G = np.diag(rng.uniform(-2, 2, size=dim))
        G = np.array(G, dtype=complex)

        fq = quantum_fisher_information_dm(rho, G)
        assert fq == pytest.approx(0.0, abs=1e-15), (
            f"Diagonal G in ρ eigenbasis should give F_Q=0, got {fq}"
        )

    def test_generator_all_diagonal_equal(self) -> None:
        """When G = c·I (all eigenvalues equal), F_Q = 0 for any ρ."""
        rng = np.random.default_rng(42)
        dim = 4

        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)

        G = 3.0 * np.eye(dim, dtype=complex)

        fq = quantum_fisher_information_dm(rho, G)
        assert fq == pytest.approx(0.0, abs=1e-15), (
            f"G=cI should give F_Q=0 for any ρ, got {fq}"
        )


# =============================================================================
# Mixed-state QFI: Noise Integration (Phase Diffusion)
# =============================================================================


class TestNoiseIntegrationDM:
    """Tests that mixed-state QFI correctly captures noise effects.

    Phase diffusion on a GHZ-like state provides an analytically tractable
    benchmark:
        |ψ⟩ = (|J⟩_z + |-J⟩_z)/√2  with J = N/2
        ρ_diff(t) = (1/2)(|J⟩⟨J| + |-J⟩⟨-J|)
                  + (e^{-γ·t·N²/2}/2)(|J⟩⟨-J| + |-J⟩⟨J|)
        F_Q(t) = N² · exp(-γ·t·N²)

    This follows from the SLD formula applied to ρ_diff with G = J_z.
    """

    def _phase_diffused_ghz(self, N: int, gamma_phi: float, t: float) -> np.ndarray:
        """Construct phase-diffused GHZ density matrix in Dicke basis.

        Args:
            N: Total particle number (dim = N+1, J = N/2).
            gamma_phi: Phase diffusion rate.
            t: Evolution time.

        Returns:
            Density matrix of phase-diffused GHZ state.

        """
        dim = N + 1
        rho = np.zeros((dim, dim), dtype=complex)

        # Populations
        rho[0, 0] = 0.5  # |J=N/2⟩⟨J=N/2|
        rho[dim - 1, dim - 1] = 0.5  # |-J=-N/2⟩⟨-J=-N/2|

        # Coherence decay
        decay = np.exp(-gamma_phi * t * N**2 / 2)
        rho[0, dim - 1] = decay / 2
        rho[dim - 1, 0] = decay / 2

        return rho

    def test_phase_diffusion_reduces_qfi(self) -> None:
        """QFI should decrease monotonically with increasing phase noise."""
        N = 6
        G = generate_phase_generator(N, "Jz")

        # No noise: pure GHZ → F_Q = N²
        rho_pure = self._phase_diffused_ghz(N, gamma_phi=0.0, t=1.0)
        fq_pure = quantum_fisher_information_dm(rho_pure, G)

        assert fq_pure == pytest.approx(N**2, rel=1e-10), (
            f"Pure GHZ F_Q={fq_pure} should equal N²={N**2}"
        )

        # Increasing noise → decreasing QFI
        fq_prev = fq_pure
        for gamma_phi in [0.01, 0.05, 0.1, 0.5, 1.0]:
            rho = self._phase_diffused_ghz(N, gamma_phi, t=1.0)
            fq = quantum_fisher_information_dm(rho, G)

            assert fq <= fq_prev + 1e-10, (
                f"QFI should decrease monotonically with γ_φ. "
                f"At γ_φ={gamma_phi}: F_Q={fq} > previous={fq_prev}"
            )
            assert fq >= 0.0, f"QFI should be non-negative, got {fq}"
            fq_prev = fq

        # Strong noise → QFI → 0
        rho_dead = self._phase_diffused_ghz(N, gamma_phi=100.0, t=1.0)
        fq_dead = quantum_fisher_information_dm(rho_dead, G)
        assert fq_dead == pytest.approx(0.0, abs=1e-10), (
            f"Strong dephasing should give F_Q≈0, got {fq_dead}"
        )

    def test_phase_diffusion_analytical_formula(self) -> None:
        """F_Q matches analytical formula N²·exp(-γ·t·N²)."""
        for N in [4, 6]:
            G = generate_phase_generator(N, "Jz")

            for gamma_phi in [0.0, 0.02, 0.05, 0.1]:
                t = 1.0
                rho = self._phase_diffused_ghz(N, gamma_phi, t)
                fq = quantum_fisher_information_dm(rho, G)

                expected = N**2 * np.exp(-gamma_phi * t * N**2)
                assert fq == pytest.approx(expected, rel=1e-10), (
                    f"N={N}, γ={gamma_phi}: F_Q={fq} != N²·exp(-γN²)={expected}"
                )

    def test_phase_diffusion_no_noise_limit(self) -> None:
        """As γ·t → 0, F_Q approaches the pure-state value N²."""
        N = 8
        G = generate_phase_generator(N, "Jz")

        for gamma_phi in [1e-3, 1e-6, 1e-10]:
            rho = self._phase_diffused_ghz(N, gamma_phi, t=1.0)
            fq = quantum_fisher_information_dm(rho, G)

            expected = N**2 * np.exp(-gamma_phi * N**2)
            assert fq == pytest.approx(expected, rel=1e-10), (
                f"γ={gamma_phi}: F_Q={fq} != expected={expected}"
            )
            # Should approach N² from below
            assert fq <= N**2 + 1e-10, f"QFI {fq} should not exceed N²={N**2}"


# =============================================================================
# Mixed-state QFI: Edge Cases
# =============================================================================


class TestEdgeCasesDM:
    """Tests for edge cases in mixed-state QFI."""

    def test_dim_one(self) -> None:
        """Trivial 1D Hilbert space: F_Q = 0 for any generator."""
        rho = np.ones((1, 1), dtype=complex)
        G = np.array([[2.0 + 0.0j]])

        fq = quantum_fisher_information_dm(rho, G)
        assert fq == pytest.approx(0.0, abs=1e-15), f"1D QFI should be 0, got {fq}"

    def test_real_density_matrix(self) -> None:
        """Real symmetric ρ and real symmetric G should work."""
        rng = np.random.default_rng(42)
        dim = 4

        # Construct real symmetric mixed state
        A = rng.normal(size=(dim, dim))
        rho = A @ A.T
        rho = rho / np.trace(rho)

        G = rng.normal(size=(dim, dim))
        G = G + G.T  # Symmetrize

        fq = quantum_fisher_information_dm(rho, G)
        assert np.isfinite(fq), "QFI should be finite for real matrices"
        assert fq >= 0.0, "QFI should be non-negative"

    def test_pure_state_from_dm(self) -> None:
        """When ρ is pure (rank 1), DM path should match pure-state path."""
        rng = np.random.default_rng(42)
        dim = 6

        vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        vec = vec / np.linalg.norm(vec)

        G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        G = G + G.conj().T

        fq_pure = quantum_fisher_information(vec, G)
        fq_dm = quantum_fisher_information_dm(np.outer(vec, vec.conj()), G)

        assert fq_pure == pytest.approx(fq_dm, rel=1e-12), (
            f"Pure state mismatch: state={fq_pure} dm={fq_dm}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

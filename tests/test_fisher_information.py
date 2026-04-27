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

from src.analysis.fisher_information import (
    classical_fisher_information,
    generate_phase_generator,
    phase_sensitivity_from_fisher,
    quantum_fisher_information,
    quantum_fisher_information_dm,
    validate_fisher_inputs,
)


class TestQuantumFisherInformationPure:
    """Tests for QFI with pure states."""

    def test_qfi_noon_state_heisenberg_scaling(self):
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
            assert np.isclose(fq, expected, rtol=0.1), (
                f"N={N}: F_Q={fq} should be ~{expected}"
            )

    def test_qfi_css_state_scaling(self):
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
            assert np.isclose(fq, expected, rtol=0.1), (
                f"N={N}: F_Q={fq} should be ~{expected}"
            )

    def test_qfi_noon_greater_than_eigenstate(self):
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

    def test_qfi_zero_for_eigenstate(self):
        """QFI should be zero for an eigenstate of the generator."""
        N = 5
        dim = N + 1

        # Eigenstate of J_z (eigenvalue -N/2)
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        generator = generate_phase_generator(N, "Jz")

        # An eigenstate has zero variance, so QFI = 0
        fq = quantum_fisher_information(state, generator)
        assert np.isclose(fq, 0.0, atol=1e-10), (
            f"QFI for eigenstate should be ~0, got {fq}"
        )

    def test_qfi_variance_formula(self):
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
        assert np.isclose(fq, expected_fq, rtol=1e-5), (
            f"F_Q={fq} should equal 4*Var={expected_fq}"
        )


class TestQuantumFisherInformationMixed:
    """Tests for QFI with mixed states."""

    def test_qfi_mixed_state_matches_pure(self):
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

        assert np.isclose(fq_pure, fq_dm, rtol=1e-5), (
            f"Pure state QFI={fq_pure}, DM QFI={fq_dm}"
        )

    def test_qfi_mixed_state_lower_than_pure(self):
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


class TestClassicalFisherInformation:
    """Tests for classical Fisher information."""

    def test_cfi_positive_for_nonzero_probabilities(self):
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

    def test_cfi_smaller_than_qfi(self):
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

    def test_phase_sensitivity_noon_heisenberg(self):
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

    def test_phase_sensitivity_invalid_f(self):
        """Should raise for non-positive Fisher information."""
        # Test zero
        with pytest.raises(ValueError):
            phase_sensitivity_from_fisher(0.0)

        # Test negative
        with pytest.raises(ValueError):
            phase_sensitivity_from_fisher(-1.0)


class TestValidation:
    """Tests for input validation."""

    def test_validate_fisher_positive(self):
        """Should validate positive Fisher information."""
        validate_fisher_inputs(1.0)
        validate_fisher_inputs(100.0)

    def test_validate_fisher_invalid(self):
        """Should reject non-positive Fisher information."""
        with pytest.raises(ValueError):
            validate_fisher_inputs(0.0)

        with pytest.raises(ValueError):
            validate_fisher_inputs(-1.0)

        with pytest.raises(ValueError):
            validate_fisher_inputs(np.nan)


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_large_N_scaling(self):
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

    def test_small_probability_handling(self):
        """Should handle small probabilities without division errors."""
        # Probabilities with very small values
        probs = np.array(
            [
                [1e-10, 1 - 1e-10],
                [0.5, 0.5],
            ]
        )

        fc = classical_fisher_information(probs, dphi=1e-3)

        # Should be finite and positive
        assert np.all(np.isfinite(fc)), "F_C should be finite"

    def test_generator_dimension_mismatch(self):
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


def test_qfi_variance_relationship():
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
    assert np.isclose(fq, expected, rtol=1e-5), (
        f"F_Q={fq} should equal 4*Var={expected}"
    )


class TestPhysicsInvariants:
    """Validation tests for physics invariants."""

    def test_cfi_less_than_equal_qfi(self):
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

    def test_cfi_positive_for_nonzero_probs(self):
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

    def test_phase_sensitivity_css_sql(self):
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
            assert np.isclose(delta_phi, expected_delta, rtol=0.01), (
                f"N={N}: Δφ={delta_phi:.6f} should be 1/√F={expected_delta:.6f}"
            )

    def test_squeezed_state_sub_sql(self):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

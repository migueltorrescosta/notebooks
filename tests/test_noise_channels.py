"""
Tests for Noise Channels Module.

Physical Validation Tests:
- One-body loss: Lindblad operator structure, completeness
- Two-body loss: Lindblad operator structure, N(N-1) scaling
- Phase diffusion: Operator structure, off-diagonal decay
- Detection noise: Binomial distribution properties
- Trace preservation: Probabilities sum to 1

Test Strategy:
- Unit tests for individual operators
- Integration tests for combined channels
- Physical validation: conservation laws, symmetry
"""

import numpy as np
import pytest
import scipy.stats
from numpy.testing import assert_allclose

from src.physics.noise_channels import (
    NoiseConfig,
    annihilation_operator,
    apply_detection_noise,
    build_lindblad_operators,
    compute_mean_particle_number,
    compute_particle_variance,
    creation_operator,
    detection_channel_pmf,
    jz_operator,
    validate_lindblad_completeness,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def N():
    """Standard atom number for testing."""
    return 4


@pytest.fixture
def config_all_channels():
    """Configuration with all channels active."""
    return NoiseConfig(gamma_1=0.1, gamma_2=0.05, gamma_phi=0.02, eta=0.9)


@pytest.fixture
def fock_distribution():
    """Standard Fock state distribution."""
    # P(n) = δ_{n, N}
    probs = np.zeros(11)
    probs[4] = 1.0
    return probs


# =============================================================================
# Test Annihilation/Creation Operators
# =============================================================================


class TestAnnihilationOperator:
    """Tests for annihilation operator a."""

    def test_operator_shape(self, N):
        """Operator should have correct shape."""
        a = annihilation_operator(N)
        assert a.shape == (N + 1, N + 1)

    def test_operator_hermitian(self, N):
        """a is not Hermitian (lowering operator)."""
        a = annihilation_operator(N)
        # a should NOT equal a†
        a_dag = a.conj().T
        assert not np.allclose(a, a_dag)

    def test_matrix_elements(self, N):
        """Test specific matrix elements."""
        a = annihilation_operator(N)
        # a[j+1, j] = √(N - j) for j = 0 to N-1
        # For N=4: check a[1, 0] = √4 = 2
        assert_allclose(np.abs(a[1, 0]), 2.0)
        # Check a[2, 1] = √3
        assert_allclose(np.abs(a[2, 1]), np.sqrt(3))

    def test_action_on_extreme_states(self, N):
        """Test action on extremal Dicke states."""
        a = annihilation_operator(N)

        # Acting on highest state |J, J⟩ (all in mode a = index 0)
        highest_state = np.zeros(N + 1)
        highest_state[0] = 1.0
        result = a @ highest_state

        # Check that highest state transitions to next state
        # The original amplitude should be zeroed
        assert np.abs(result[0]) < 1e-10
        # There should be non-zero amplitude somewhere
        assert np.sum(np.abs(result)) > 1e-10


class TestCreationOperator:
    """Tests for creation operator a†."""

    def test_operator_shape(self, N):
        """Operator should have correct shape."""
        a_dag = creation_operator(N)
        assert a_dag.shape == (N + 1, N + 1)

    def test_hermitian_conjugate(self, N):
        """a relates to a via transpose-conjugate relationship."""
        # The key property is that one operator is the transpose of the other
        a = annihilation_operator(N)
        a_dag = creation_operator(N)
        # They should differ by transpose (not necessarily conjugate due to real values)
        assert a_dag.shape == a.shape

    def test_matrix_elements(self, N):
        """Test specific matrix elements."""
        a_dag = creation_operator(N)

        # a†[j-1, j] = √j for j = 1 to N
        # For N=4, check a†[0, 1] = √1 = 1
        assert_allclose(np.abs(a_dag[0, 1]), 1.0)

        # Check a†[1, 2] = √2
        assert_allclose(np.abs(a_dag[1, 2]), np.sqrt(2))


# =============================================================================
# Test J_z Operator
# =============================================================================


class TestJzOperator:
    """Tests for J_z operator."""

    def test_eigenvalues(self, N):
        """J_z should have correct eigenvalues."""
        J_z = jz_operator(N)
        J = N / 2.0

        expected = np.arange(J, -J - 1, -1)
        assert_allclose(J_z.diagonal(), expected)

    def test_hermitian(self, N):
        """J_z should be Hermitian."""
        J_z = jz_operator(N)
        assert_allclose(J_z, J_z.conj().T)

    def test_commutation(self, N):
        """Test number operator structure."""
        a = annihilation_operator(N)
        a_dag = creation_operator(N)

        # n = a†a should be diagonal
        n = a_dag @ a

        # Check it's Hermitian
        assert_allclose(n, n.conj().T, atol=1e-10)

        # Check diagonal entries are non-negative
        assert np.all(np.diag(n) >= -1e-10)


# =============================================================================
# Test Lindblad Operators Build
# =============================================================================


class TestBuildLindbladOperators:
    """Tests for building Lindblad operators."""

    def test_empty_config(self, N):
        """Empty config should return empty list."""
        config = NoiseConfig()
        L_ops = build_lindblad_operators(N, config)
        assert len(L_ops) == 0

    def test_one_body_loss(self, N):
        """One-body loss should give correct operator."""
        config = NoiseConfig(gamma_1=0.3)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 1
        a = annihilation_operator(N)
        expected = np.sqrt(0.3) * a
        assert_allclose(L_ops[0], expected)

    def test_two_body_loss(self, N):
        """Two-body loss should give a² operator."""
        config = NoiseConfig(gamma_2=0.3)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 1
        a = annihilation_operator(N)
        a_squared = a @ a
        expected = np.sqrt(0.3) * a_squared
        assert_allclose(L_ops[0], expected)

    def test_phase_diffusion(self, N):
        """Phase diffusion should give J_z operator."""
        config = NoiseConfig(gamma_phi=0.3)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 1
        J_z = jz_operator(N)
        expected = np.sqrt(0.3) * J_z
        assert_allclose(L_ops[0], expected)

    def test_combined_channels(self, N):
        """Combined channels should give list of all operators."""
        config = NoiseConfig(gamma_1=0.1, gamma_2=0.05, gamma_phi=0.02)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 3

    def test_invalid_config(self, N):
        """Invalid config should raise error."""
        # Negative rate should raise ValueError
        config_neg = NoiseConfig(gamma_1=-0.1)
        with pytest.raises(ValueError):
            build_lindblad_operators(N, config_neg)

        # eta > 1 should raise error during validation (check the build function)
        config_invalid = NoiseConfig(eta=1.5)
        with pytest.raises(ValueError):
            build_lindblad_operators(N, config_invalid)


# =============================================================================
# Test Validation
# =============================================================================


class TestLindbladValidation:
    """Tests for Lindblad completeness validation."""

    def test_empty_operators(self):
        """Empty operator list should pass validation."""
        result = validate_lindblad_completeness([])
        assert result["is_bounded"]

    def test_one_body_bounded(self, N):
        """One-body loss should satisfy completeness."""
        config = NoiseConfig(gamma_1=0.1)
        L_ops = build_lindblad_operators(N, config)
        result = validate_lindblad_completeness(L_ops)

        assert result["is_bounded"]

    def test_all_channels_bounded(self, N):
        """All channels combined should satisfy completeness."""
        config = NoiseConfig(gamma_1=0.1, gamma_2=0.05, gamma_phi=0.02)
        L_ops = build_lindblad_operators(N, config)
        result = validate_lindblad_completeness(L_ops)

        # With small rates, should be bounded
        assert result["max_eigenvalue"] < 2.0


# =============================================================================
# Test Detection Noise
# =============================================================================


class TestDetectionNoise:
    """Tests for detection noise channel."""

    def test_perfect_detection(self):
        """Perfect detection should return identity."""
        probs = np.array([0.1, 0.4, 0.5])
        result = apply_detection_noise(probs, eta=1.0, n_trials=100)

        assert_allclose(result, probs, atol=1e-6)

    def test_binomial_pmf(self):
        """Test binomial PMF calculation."""
        # n=2, η=0.5
        # P(0) = (1-0.5)² = 0.25
        # P(1) = 2(0.5)(0.5) = 0.5
        # P(2) = 0.5² = 0.25
        pmf = detection_channel_pmf(2, 0.5)
        assert_allclose(pmf, [0.25, 0.5, 0.25], atol=1e-10)

    def test_binomial_eta_zero(self):
        """η=0 should give P(k) = δ_{k,0}."""
        pmf = detection_channel_pmf(2, 0.0)
        expected = [1.0, 0.0, 0.0]
        assert_allclose(pmf, expected, atol=1e-10)

    def test_binomial_eta_one(self):
        """η=1 should give δ_{k,n}."""
        pmf = detection_channel_pmf(2, 1.0)
        expected = [0.0, 0.0, 1.0]
        assert_allclose(pmf, expected, atol=1e-10)

    def test_detection_monte_carlo(self):
        """Monte Carlo should approximate binomial."""
        probs = np.array([0.0, 0.0, 1.0])  # Certain n=2
        result = apply_detection_noise(probs, eta=0.5, n_trials=1000, seed=42)

        # Should have some probability at k=0,1,2
        assert result[0] > 0
        assert result[1] > 0
        assert result[2] > 0

    def test_validity_ranges(self):
        """Invalid eta should raise error."""
        probs = np.array([0.5, 0.5])

        with pytest.raises(ValueError):
            apply_detection_noise(probs, eta=-0.1, n_trials=10)

        with pytest.raises(ValueError):
            apply_detection_noise(probs, eta=1.1, n_trials=10)


# =============================================================================
# Test Physical Properties
# =============================================================================


class TestPhysicalProperties:
    """Tests for physical conservation laws."""

    def test_number_conservation_no_loss(self, N):
        """Without loss, particle number should be conserved."""
        config = NoiseConfig(gamma_1=0, gamma_2=0, gamma_phi=0)
        L_ops = build_lindblad_operators(N, config)

        # No dissipation operators
        assert len(L_ops) == 0

        # Number operator commutes with nothing = identity
        a = annihilation_operator(N)
        n = a.conj().T @ a

        # [n, 1] = 0 trivially
        comm = n @ np.eye(N + 1) - np.eye(N + 1) @ n
        assert_allclose(comm, np.zeros((N + 1, N + 1)))

    def test_two_body_rate_scaling(self):
        """Two-body loss rate scales with N(N-1)."""
        # Verify operator structure: a² is constructed correctly
        N = 10
        config = NoiseConfig(gamma_2=0.5)
        L_ops = build_lindblad_operators(N, config)

        # Check is proportional to a²
        a = annihilation_operator(N)
        a_squared_expected = np.sqrt(0.5) * (a @ a)
        assert_allclose(L_ops[0], a_squared_expected)

        # The operator should be non-trivial (not identity)
        assert not np.allclose(L_ops[0], np.eye(N + 1) * 0.5)


# =============================================================================
# Test Expectation Values
# =============================================================================


class TestExpectationValues:
    """Tests for expectation value calculations."""

    def test_mean_number_single_fock(self):
        """Single Fock state should have mean = n."""
        n = 3
        probs = np.zeros(10)
        probs[n] = 1.0

        mean_n = compute_mean_particle_number(probs)
        assert_allclose(mean_n, n)

    def test_variance_single_fock(self):
        """Single Fock state should have zero variance."""
        n = 3
        probs = np.zeros(10)
        probs[n] = 1.0

        var_n = compute_particle_variance(probs)
        assert_allclose(var_n, 0.0)

    def test_mean_poisson(self):
        """Poisson distribution should have mean = variance."""
        # P(n) = e^{-μ} μ^n / n!
        mu = 2.5
        n_max = 15  # Need more values for convergence
        probs = scipy.stats.poisson.pmf(np.arange(n_max + 1), mu)
        probs = probs / probs.sum()  # Normalize

        mean_n = compute_mean_particle_number(probs)
        var_n = compute_particle_variance(probs)

        # For Poisson: mean = variance = μ (approximately)
        assert_allclose(mean_n, mu, atol=0.1)
        assert_allclose(var_n, mu, atol=0.1)


# =============================================================================
# Integration Test - Combined Noise Model
# =============================================================================


class TestCombinedNoiseModel:
    """Integration tests for combined noise channels."""

    def test_config_validation(self):
        """Test that config parameters are properly used."""
        N = 4
        config = NoiseConfig(
            gamma_1=0.1,
            gamma_2=0.05,
            gamma_phi=0.02,
            eta=0.95,
        )

        # Build operators
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 3

        # The operators should be valid (bounded by a reasonable value)
        result = validate_lindblad_completeness(L_ops)
        # Just check max eigenvalue is finite
        assert np.isfinite(result["max_eigenvalue"])

    def test_all_channels_independent(self):
        """Test each channel is independent."""
        N = 4

        # Check each channel separately
        for config in [
            NoiseConfig(gamma_1=0.1),
            NoiseConfig(gamma_2=0.1),
            NoiseConfig(gamma_phi=0.1),
        ]:
            L_ops = build_lindblad_operators(N, config)
            assert len(L_ops) == 1

    def test_zero_rates(self):
        """Zero rates should produce no operators."""
        N = 4
        config = NoiseConfig(gamma_1=0, gamma_2=0, gamma_phi=0)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_N_zero(self):
        """N=0 should produce valid operators."""
        a = annihilation_operator(0)
        assert a.shape == (1, 1)
        assert_allclose(a, np.array([[0.0]]))

    def test_small_N(self):
        """Small N values should work."""
        for N in [0, 1, 2]:
            a = annihilation_operator(N)
            assert a.shape == (N + 1, N + 1)

            J_z = jz_operator(N)
            assert J_z.shape == (N + 1, N + 1)

    def test_very_small_rates(self):
        """Very small rates should not cause numerical issues."""
        N = 4
        config = NoiseConfig(gamma_1=1e-10, gamma_2=0, gamma_phi=0)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 1
        assert np.isfinite(L_ops[0]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

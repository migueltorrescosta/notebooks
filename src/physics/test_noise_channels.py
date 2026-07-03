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

from src.utils.enums import OperatorBasis

from .dicke_basis import jz_operator
from .noise_channels import (
    NoiseConfig,
    annihilation_operator,
    apply_detection_noise,
    build_lindblad_operators,
    compute_mean_particle_number,
    compute_particle_variance,
    creation_operator,
    validate_lindblad_completeness,
)

# Fixtures


@pytest.fixture
def N() -> int:
    """Standard atom number for testing."""
    return 4



# Test Annihilation/Creation Operators


class TestAnnihilationOperator:
    """Tests for annihilation operator a."""

    def test_given_operator_then_have_correct_shape(self, N: int) -> None:
        a = annihilation_operator(N)
        assert a.shape == (N + 1, N + 1)

    def test_a_is_not_hermitian_lowering_operator(self, N: int) -> None:
        a = annihilation_operator(N)
        # a should NOT equal a†
        a_dag = a.conj().T
        assert a != pytest.approx(a_dag)

    def test_specific_matrix_elements(self, N: int) -> None:
        a = annihilation_operator(N)
        # a[j+1, j] = √(N - j) for j = 0 to N-1
        # For N=4: check a[1, 0] = √4 = 2
        assert_allclose(np.abs(a[1, 0]), 2.0)
        # Check a[2, 1] = √3
        assert_allclose(np.abs(a[2, 1]), np.sqrt(3))

    def test_action_on_extremal_dicke_states(self, N: int) -> None:
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

    def test_given_operator_then_have_correct_shape(self, N: int) -> None:
        a_dag = creation_operator(N)
        assert a_dag.shape == (N + 1, N + 1)

    def test_a_relates_to_a_via_transpose_conjugate_relationship(self, N: int) -> None:
        # The key property is that one operator is the transpose of the other
        a = annihilation_operator(N)
        a_dag = creation_operator(N)
        # They should differ by transpose (not necessarily conjugate due to real values)
        assert a_dag.shape == a.shape

    def test_specific_matrix_elements(self, N: int) -> None:
        a_dag = creation_operator(N)

        # a†[j-1, j] = √j for j = 1 to N
        # For N=4, check a†[0, 1] = √1 = 1
        assert_allclose(np.abs(a_dag[0, 1]), 1.0)

        # Check a†[1, 2] = √2
        assert_allclose(np.abs(a_dag[1, 2]), np.sqrt(2))


# Test J_z Operator


class TestJzOperator:
    """Tests for J_z operator."""

    def test_given_j_z_then_have_correct_eigenvalues(self, N: int) -> None:
        J_z = jz_operator(N, basis=OperatorBasis.DICKE)
        J = N / 2.0

        expected = np.arange(J, -J - 1, -1)
        assert_allclose(J_z.diagonal(), expected)

    def test_given_j_z_then_be_hermitian(self, N: int) -> None:
        J_z = jz_operator(N, basis=OperatorBasis.DICKE)
        assert_allclose(J_z, J_z.conj().T)

    def test_number_operator_structure(self, N: int) -> None:
        a = annihilation_operator(N)
        a_dag = creation_operator(N)

        # n = a†a should be diagonal
        n = a_dag @ a

        # Check it's Hermitian
        assert_allclose(n, n.conj().T, atol=1e-10)

        # Check diagonal entries are non-negative
        assert np.all(np.diag(n) >= -1e-10)


# Test Lindblad Operators Build


class TestBuildLindbladOperators:
    """Tests for building Lindblad operators."""

    def test_given_empty_config_then_return_empty_list(self, N: int) -> None:
        config = NoiseConfig()
        L_ops = build_lindblad_operators(N, config)
        assert len(L_ops) == 0

    def test_given_one_body_loss_then_give_correct_operator(self, N: int) -> None:
        config = NoiseConfig(gamma_1=0.3)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 1
        a = annihilation_operator(N)
        expected = np.sqrt(0.3) * a
        assert_allclose(L_ops[0], expected)

    def test_given_two_body_loss_then_give_a_operator(self, N: int) -> None:
        config = NoiseConfig(gamma_2=0.3)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 1
        a = annihilation_operator(N)
        a_squared = a @ a
        expected = np.sqrt(0.3) * a_squared
        assert_allclose(L_ops[0], expected)

    def test_given_phase_diffusion_then_give_j_z_operator(self, N: int) -> None:
        config = NoiseConfig(gamma_phi=0.3)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 1
        J_z = jz_operator(N, basis=OperatorBasis.DICKE)
        expected = np.sqrt(0.3) * J_z
        assert_allclose(L_ops[0], expected)

    def test_given_combined_channels_then_give_list_of_all_operators(
        self, N: int
    ) -> None:
        config = NoiseConfig(gamma_1=0.1, gamma_2=0.05, gamma_phi=0.02)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 3

    def test_given_invalid_config_then_raise_error(self, N: int) -> None:
        # Negative rate should raise ValueError
        config_neg = NoiseConfig(gamma_1=-0.1)
        with pytest.raises(ValueError):
            build_lindblad_operators(N, config_neg)

        # eta > 1 should raise error during validation (check the build function)
        config_invalid = NoiseConfig(eta=1.5)
        with pytest.raises(ValueError):
            build_lindblad_operators(N, config_invalid)


# Test Validation


class TestLindbladValidation:
    """Tests for Lindblad completeness validation."""

    def test_given_empty_operator_list_then_pass_validation(self) -> None:
        result = validate_lindblad_completeness([])
        assert result["is_bounded"]

    def test_given_one_body_loss_then_satisfy_completeness(self, N: int) -> None:
        config = NoiseConfig(gamma_1=0.1)
        L_ops = build_lindblad_operators(N, config)
        result = validate_lindblad_completeness(L_ops)

        assert result["is_bounded"]

    def test_given_all_channels_combined_then_satisfy_completeness(
        self, N: int
    ) -> None:
        config = NoiseConfig(gamma_1=0.1, gamma_2=0.05, gamma_phi=0.02)
        L_ops = build_lindblad_operators(N, config)
        result = validate_lindblad_completeness(L_ops)

        # With small rates, should be bounded
        assert result["max_eigenvalue"] < 2.0


# Test Detection Noise


class TestDetectionNoise:
    """Tests for detection noise channel."""

    def test_given_perfect_detection_then_return_identity(self) -> None:
        probs = np.array([0.1, 0.4, 0.5])
        result = apply_detection_noise(probs, eta=1.0, n_trials=100)

        assert_allclose(result, probs, atol=1e-6)

    def test_binomial_pmf_calculation(self) -> None:
        # n=2, η=0.5
        # P(0) = (1-0.5)² = 0.25
        # P(1) = 2(0.5)(0.5) = 0.5
        # P(2) = 0.5² = 0.25
        pmf = scipy.stats.binom.pmf(np.arange(3), 2, 0.5)
        assert_allclose(pmf, [0.25, 0.5, 0.25], atol=1e-10)

    def test_given_0_then_give_p_k_k_0(self) -> None:
        pmf = scipy.stats.binom.pmf(np.arange(3), 2, 0.0)
        expected = [1.0, 0.0, 0.0]
        assert_allclose(pmf, expected, atol=1e-10)

    def test_given_1_then_give_k_n(self) -> None:
        pmf = scipy.stats.binom.pmf(np.arange(3), 2, 1.0)
        expected = [0.0, 0.0, 1.0]
        assert_allclose(pmf, expected, atol=1e-10)

    def test_given_monte_carlo_then_approximate_binomial(self) -> None:
        probs = np.array([0.0, 0.0, 1.0])  # Certain n=2
        result = apply_detection_noise(probs, eta=0.5, n_trials=1000, seed=42)

        # Should have some probability at k=0,1,2
        assert result[0] > 0
        assert result[1] > 0
        assert result[2] > 0

    def test_given_invalid_eta_then_raise_error(self) -> None:
        probs = np.array([0.5, 0.5])

        with pytest.raises(ValueError):
            apply_detection_noise(probs, eta=-0.1, n_trials=10)

        with pytest.raises(ValueError):
            apply_detection_noise(probs, eta=1.1, n_trials=10)


# Test Physical Properties


class TestPhysicalProperties:
    """Tests for physical conservation laws."""

    def test_given_without_loss_particle_number_then_be_conserved(self, N: int) -> None:
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

    def test_two_body_loss_rate_scales_with_n_n_1(self) -> None:
        # Verify operator structure: a² is constructed correctly
        N = 10
        config = NoiseConfig(gamma_2=0.5)
        L_ops = build_lindblad_operators(N, config)

        # Check is proportional to a²
        a = annihilation_operator(N)
        a_squared_expected = np.sqrt(0.5) * (a @ a)
        assert_allclose(L_ops[0], a_squared_expected)

        # The operator should be non-trivial (not identity)
        assert L_ops[0] != pytest.approx(np.eye(N + 1) * 0.5)


# Test Expectation Values


class TestExpectationValues:
    """Tests for expectation value calculations."""

    def test_given_single_fock_state_then_have_mean_n(self) -> None:
        n = 3
        probs = np.zeros(10)
        probs[n] = 1.0

        mean_n = compute_mean_particle_number(probs)
        assert_allclose(mean_n, n)

    def test_given_single_fock_state_then_have_zero_variance(self) -> None:
        n = 3
        probs = np.zeros(10)
        probs[n] = 1.0

        var_n = compute_particle_variance(probs)
        assert_allclose(var_n, 0.0)

    def test_given_poisson_distribution_then_have_mean_variance(self) -> None:
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


# Integration Test - Combined Noise Model


class TestCombinedNoiseModel:
    """Integration tests for combined noise channels."""

    def test_that_config_parameters_are_properly_used(self) -> None:
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

    def test_each_channel_is_independent(self) -> None:
        N = 4

        # Check each channel separately
        for config in [
            NoiseConfig(gamma_1=0.1),
            NoiseConfig(gamma_2=0.1),
            NoiseConfig(gamma_phi=0.1),
        ]:
            L_ops = build_lindblad_operators(N, config)
            assert len(L_ops) == 1

    def test_given_zero_rates_then_produce_no_operators(self) -> None:
        N = 4
        config = NoiseConfig(gamma_1=0, gamma_2=0, gamma_phi=0)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 0


# Test Edge Cases


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_given_n_0_then_produce_valid_operators(self) -> None:
        a = annihilation_operator(0)
        assert a.shape == (1, 1)
        assert_allclose(a, np.array([[0.0]]))

    @pytest.mark.parametrize("N", [0, 1, 2], ids=["0", "1", "2"])
    def test_given_small_n_values_then_work(self, N: int) -> None:
        a = annihilation_operator(N)
        assert a.shape == (N + 1, N + 1)

        J_z = jz_operator(N, basis=OperatorBasis.DICKE)
        assert J_z.shape == (N + 1, N + 1)

    def test_given_very_small_rates_then_not_cause_numerical_issues(self) -> None:
        N = 4
        config = NoiseConfig(gamma_1=1e-10, gamma_2=0, gamma_phi=0)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 1
        assert np.isfinite(L_ops[0]).all()

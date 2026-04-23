"""Unit tests for Delta Estimation physics module."""

import numpy as np
import pytest

from src.delta_estimation import (
    DeltaEstimationConfig,
    generate_initial_state,
    generate_hamiltonian,
    evolve_density_matrix,
    compute_observables,
    full_calculation,
    validate_state,
    validate_hamiltonian,
)


class TestStatePreparation:
    def test_generate_initial_state_dimensions(self) -> None:
        """Initial state should have correct dimensions."""
        for dim in [2, 5, 10]:
            state = generate_initial_state(dim, 0)
            assert state.shape == (2 * dim, 2 * dim)

    def test_generate_initial_state_trace(self) -> None:
        """Initial state should have trace 1."""
        dim = 5
        state = generate_initial_state(dim, 2)
        assert np.isclose(np.trace(state), 1.0)

    def test_generate_initial_state_invalid_state(self) -> None:
        """Should raise for invalid initial state."""
        with pytest.raises(ValueError):
            generate_initial_state(5, 10)


class TestHamiltonian:
    def test_hamiltonian_hermitian(self) -> None:
        """Hamiltonian should be Hermitian."""
        config = DeltaEstimationConfig()
        H = generate_hamiltonian(config)
        assert np.allclose(H, H.conj().T)

    def test_hamiltonian_shape(self) -> None:
        """Hamiltonian should have correct dimensions."""
        config = DeltaEstimationConfig(ancillary_dimension=5)
        H = generate_hamiltonian(config)
        assert H.shape == (10, 10)


class TestEvolution:
    def test_evolved_state_normalized(self) -> None:
        """Evolved state should remain normalized."""
        config = DeltaEstimationConfig()
        H = generate_hamiltonian(config)
        rho0 = generate_initial_state(
            config.ancillary_dimension, config.ancillary_initial_state
        )
        rho_t = evolve_density_matrix(H, rho0, time=1.0)
        # For pure state evolution, trace should be 1
        assert np.isclose(np.trace(rho_t), 1.0)


class TestObservables:
    def test_observables_sum_to_one(self) -> None:
        """Populations should sum to 1."""
        rho = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
        obs = compute_observables(rho)
        assert np.isclose(obs["pop_0"] + obs["pop_1"], 1.0)


class TestFullCalculation:
    def test_full_calculation_returns_dict(self) -> None:
        """Should return correctly shaped dictionary."""
        config = DeltaEstimationConfig(t=0.0)
        result = full_calculation(config)
        assert "time" in result
        assert "<0|rho_system_t|0>" in result
        assert "expected_sigma_z" in result

    def test_full_calculation_at_t0(self) -> None:
        """At t=0, should return initial state."""
        config = DeltaEstimationConfig(t=0.0)
        result = full_calculation(config)
        # At t=0, population in |0⟩ should be 1
        assert result["<0|rho_system_t|0>"] == pytest.approx(1.0, abs=0.01)


class TestValidation:
    def test_validate_valid_state(self) -> None:
        """Valid density matrix should pass."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert validate_state(rho, (2, 2))

    def test_validate_invalid_trace(self) -> None:
        """State with wrong trace should fail."""
        rho = np.array([[0.5, 0], [0, 0]], dtype=complex)
        assert not validate_state(rho, (2, 2))

    def test_validate_hamiltonian(self) -> None:
        """Valid Hamiltonian should pass."""
        config = DeltaEstimationConfig()
        H = generate_hamiltonian(config)
        assert validate_hamiltonian(H)

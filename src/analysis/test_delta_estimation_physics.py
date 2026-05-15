"""Unit tests for Delta Estimation physics module."""

from __future__ import annotations

import numpy as np
import pytest

from .delta_estimation import (
    DeltaEstimationConfig,
    compute_observables,
    evolve_density_matrix,
    full_calculation,
    generate_hamiltonian,
    generate_initial_state,
    validate_hamiltonian,
    validate_state,
)


class TestStatePreparation:
    @pytest.mark.parametrize("dim", [2, 5, 10])
    def test_given_dim_d_then_state_shape_is_2d_x_2d(self, dim: int) -> None:
        state = generate_initial_state(dim, 0)
        assert state.shape == (2 * dim, 2 * dim)

    def test_initial_state_has_trace_one(self) -> None:
        dim = 5
        state = generate_initial_state(dim, 2)
        assert np.trace(state) == pytest.approx(1.0)

    def test_given_invalid_ancilla_state_then_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            generate_initial_state(5, 10)


class TestHamiltonian:
    def test_hamiltonian_is_hermitian(self) -> None:
        config = DeltaEstimationConfig()
        H = generate_hamiltonian(config)
        assert pytest.approx(H.conj().T) == H

    def test_given_dim_5_then_hamiltonian_shape_is_10x10(self) -> None:
        config = DeltaEstimationConfig(ancillary_dimension=5)
        H = generate_hamiltonian(config)
        assert H.shape == (10, 10)


class TestEvolution:
    def test_evolved_state_remains_normalized(self) -> None:
        config = DeltaEstimationConfig()
        H = generate_hamiltonian(config)
        rho0 = generate_initial_state(
            config.ancillary_dimension,
            config.ancillary_initial_state,
        )
        rho_t = evolve_density_matrix(H, rho0, time=1.0)
        assert np.trace(rho_t) == pytest.approx(1.0)


class TestObservables:
    def test_populations_sum_to_one(self) -> None:
        rho = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
        obs = compute_observables(rho)
        assert obs["pop_0"] + obs["pop_1"] == pytest.approx(1.0)


class TestFullCalculation:
    def test_full_calculation_returns_expected_fields(self) -> None:
        config = DeltaEstimationConfig(t=0.0)
        result = full_calculation(config)
        assert "time" in result
        assert "<0|rho_system_t|0>" in result
        assert "expected_sigma_z" in result

    def test_given_zero_time_then_returns_initial_state(self) -> None:
        config = DeltaEstimationConfig(t=0.0)
        result = full_calculation(config)
        assert result["<0|rho_system_t|0>"] == pytest.approx(1.0, abs=0.01)


class TestValidation:
    def test_given_valid_density_matrix_then_passes(self) -> None:
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert validate_state(rho, (2, 2))

    def test_given_wrong_trace_then_fails(self) -> None:
        rho = np.array([[0.5, 0], [0, 0]], dtype=complex)
        assert not validate_state(rho, (2, 2))

    def test_given_valid_hamiltonian_then_passes(self) -> None:
        config = DeltaEstimationConfig()
        H = generate_hamiltonian(config)
        assert validate_hamiltonian(H)

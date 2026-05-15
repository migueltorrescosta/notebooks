"""Unit tests for Delta Estimation physics module."""

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
    def test_initial_state_should_have_correct_dimensions(self) -> None:
        for dim in [2, 5, 10]:
            state = generate_initial_state(dim, 0)
            assert state.shape == (2 * dim, 2 * dim), (
                "Expected state.shape == (2 * dim, 2 * dim)"
            )

    def test_initial_state_should_have_trace_1(self) -> None:
        dim = 5
        state = generate_initial_state(dim, 2)
        assert np.trace(state) == pytest.approx(1.0), (
            "Expected np.trace(state) == pytest.approx(1.0)"
        )

    def test_initial_state_should_raise_for_invalid_state(self) -> None:
        with pytest.raises(ValueError):
            generate_initial_state(5, 10)


class TestHamiltonian:
    def test_hamiltonian_should_be_hermitian(self) -> None:
        config = DeltaEstimationConfig()
        H = generate_hamiltonian(config)
        assert pytest.approx(H.conj().T) == H, "Expected H == pytest.approx(H.conj().T)"

    def test_hamiltonian_should_have_correct_dimensions(self) -> None:
        config = DeltaEstimationConfig(ancillary_dimension=5)
        H = generate_hamiltonian(config)
        assert H.shape == (10, 10), "Expected H.shape == (10, 10)"


class TestEvolution:
    def test_evolved_state_should_remain_normalized(self) -> None:
        config = DeltaEstimationConfig()
        H = generate_hamiltonian(config)
        rho0 = generate_initial_state(
            config.ancillary_dimension,
            config.ancillary_initial_state,
        )
        rho_t = evolve_density_matrix(H, rho0, time=1.0)
        # For pure state evolution, trace should be 1
        assert np.trace(rho_t) == pytest.approx(1.0), (
            "Expected np.trace(rho_t) == pytest.approx(1.0)"
        )


class TestObservables:
    def test_populations_should_sum_to_1(self) -> None:
        rho = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
        obs = compute_observables(rho)
        assert obs["pop_0"] + obs["pop_1"] == pytest.approx(1.0), (
            'Expected obs["pop_0"] + obs["pop_1"] == pytest.approx(1.0)'
        )


class TestFullCalculation:
    def test_full_calculation_should_return_correctly_shaped_dict(self) -> None:
        config = DeltaEstimationConfig(t=0.0)
        result = full_calculation(config)
        assert "time" in result, 'Expected "time" in result'
        assert "<0|rho_system_t|0>" in result, (
            'Expected "<0|rho_system_t|0 > " in result'
        )
        assert "expected_sigma_z" in result, 'Expected "expected_sigma_z" in result'

    def test_full_calculation_at_t0_should_return_initial_state(self) -> None:
        config = DeltaEstimationConfig(t=0.0)
        result = full_calculation(config)
        # At t=0, population in |0⟩ should be 1
        assert result["<0|rho_system_t|0>"] == pytest.approx(1.0, abs=0.01), (
            'Expected result["<0|rho_system_t|0>"] == pytest.approx(1.0, abs=0.01)'
        )


class TestValidation:
    def test_validate_should_pass_for_valid_density_matrix(self) -> None:
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert validate_state(rho, (2, 2)), (
            "Condition failed: validate_state(rho, (2, 2))"
        )

    def test_validate_should_fail_for_wrong_trace(self) -> None:
        rho = np.array([[0.5, 0], [0, 0]], dtype=complex)
        assert not validate_state(rho, (2, 2)), (
            "validate_state(rho, (2, 2)) should be falsy"
        )

    def test_validate_should_pass_for_valid_hamiltonian(self) -> None:
        config = DeltaEstimationConfig()
        H = generate_hamiltonian(config)
        assert validate_hamiltonian(H), "Condition failed: validate_hamiltonian(H)"

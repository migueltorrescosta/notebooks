"""
Tests for MZI with Ancilla simulation module.
"""

import numpy as np
import pytest

from src.utils.validators import validate_unitary

from .mzi_simulation import (
    beam_splitter_unitary,
    coherent_state,
    compute_interference_fringe,
    compute_output_probabilities,
    create_system_operators,
    evolve_mzi,
    fock_state_n,
    noon_state,
    phase_shift_unitary,
    prepare_input_state,
    single_photon_state,
    system_ancilla_interaction_unitary,
    vacuum_state,
    validate_state,
)


class TestFockStates:
    """Test Fock state creation."""

    def test_vacuum_state(self) -> None:
        state = vacuum_state(max_photons=2)
        assert validate_state(state), "Condition failed: validate_state(state)"
        assert state[0] == pytest.approx(1.0)  # |0,0> is first state

    def test_single_photon_state_mode0(self) -> None:
        state = single_photon_state(mode=0, max_photons=2)
        assert validate_state(state), "Condition failed: validate_state(state)"
        # |1,0> should have amplitude 1

    def test_single_photon_state_mode1(self) -> None:
        state = single_photon_state(mode=1, max_photons=2)
        assert validate_state(state), "Condition failed: validate_state(state)"

    def test_noon_state(self) -> None:
        state = noon_state(N=2, max_photons=3)
        assert validate_state(state), "Condition failed: validate_state(state)"
        # Should be (|2,0> + |0,2>)/sqrt(2)

    def test_coherent_state(self) -> None:
        state = coherent_state(alpha=0.5 + 0.5j, max_photons=5)
        assert validate_state(state), "Condition failed: validate_state(state)"

    def test_fock_state_n(self) -> None:
        state = fock_state_n(n=3, max_photons=5)
        assert validate_state(state), "Condition failed: validate_state(state)"


class TestOperators:
    """Test operator creation and properties."""

    def test_system_operators_dimensions(self) -> None:
        a0, a1, _a0_dag, _a1_dag = create_system_operators(max_photons=2)
        dim = (2 + 1) ** 2
        assert a0.shape == (dim, dim), "Expected a0.shape == (dim, dim)"
        assert a1.shape == (dim, dim), "Expected a1.shape == (dim, dim)"

    def test_beam_splitter_preserves_norm(self) -> None:
        # Beam splitter should preserve norm of any state
        bs = beam_splitter_unitary(theta=np.pi / 4, phi=0.0, max_photons=2)
        # Test with vacuum
        vac = vacuum_state(2)
        vac_out = bs @ vac
        assert np.sum(np.abs(vac_out) ** 2) == pytest.approx(1.0, abs=1e-6), (
            "Expected np.sum(np.abs(vac_out) ** 2) == pytest.approx(1.0, abs=1e-6)"
        )
        # Test with single photon
        sp = single_photon_state(0, 2)
        sp_out = bs @ sp
        assert np.sum(np.abs(sp_out) ** 2) == pytest.approx(1.0, abs=1e-6), (
            "Expected np.sum(np.abs(sp_out) ** 2) == pytest.approx(1.0, abs=1e-6)"
        )

    def test_phase_shift_unitarity(self) -> None:
        ps = phase_shift_unitary(phi=np.pi / 2, max_photons=2)
        assert validate_unitary(ps, tol=1e-10), (
            "Condition failed: validate_unitary(ps, tol=1e-10)"
        )

    def test_ancilla_coupling_hermitian(self) -> None:
        H = system_ancilla_interaction_unitary(
            g=1.0,
            interaction_time=1.0,
            coupling_type="phase_coupling",
            max_photons=2,
            ancilla_dim=3,
        )
        # Check that exp(-i*H) is unitary
        assert validate_unitary(H, tol=1e-6), (
            "Condition failed: validate_unitary(H, tol=1e-6)"
        )


class TestEvolution:
    """Test state evolution."""

    def test_vacuum_produces_vacuum(self) -> None:
        state = vacuum_state(max_photons=1)
        evolved = evolve_mzi(
            initial_system_state=state,
            theta=np.pi / 4,
            phi_bs=0.0,
            phi_phase=0.0,
            g=0.0,
            interaction_time=0.0,
            coupling_type="phase_coupling",
            max_photons=1,
            ancilla_dim=2,
        )
        # Vacuum should remain vacuum (no phase to apply)
        assert validate_state(evolved), "Condition failed: validate_state(evolved)"

    def test_evolution_conserves_probability(self) -> None:
        state = vacuum_state(max_photons=2)
        evolved = evolve_mzi(
            initial_system_state=state,
            theta=np.pi / 4,
            phi_bs=0.0,
            phi_phase=np.pi,
            g=0.5,
            interaction_time=0.1,
            coupling_type="phase_coupling",
            max_photons=2,
            ancilla_dim=3,
        )
        prob = np.sum(np.abs(evolved) ** 2)
        assert prob == pytest.approx(1.0), "Expected prob == pytest.approx(1.0)"

    def test_output_probabilities_sum_to_one(self) -> None:
        state = vacuum_state(max_photons=2)
        evolved = evolve_mzi(
            initial_system_state=state,
            theta=np.pi / 4,
            phi_bs=0.0,
            phi_phase=np.pi,
            g=0.0,
            interaction_time=0.0,
            coupling_type="phase_coupling",
            max_photons=2,
            ancilla_dim=2,
        )
        p0, p1 = compute_output_probabilities(evolved, max_photons=2, ancilla_dim=2)
        assert p0 + p1 == pytest.approx(1.0), "Expected p0 + p1 == pytest.approx(1.0)"


class TestInterference:
    """Test interference pattern computation."""

    def test_interference_fringe_shape(self) -> None:
        state = vacuum_state(max_photons=2)
        phases = np.linspace(0, 2 * np.pi, 50)
        fringe = compute_interference_fringe(
            phase_range=phases,
            initial_system_state=state,
            theta=np.pi / 4,
            phi_bs=0.0,
            g=0.0,
            interaction_time=0.0,
            coupling_type="phase_coupling",
            max_photons=2,
            ancilla_dim=2,
        )
        assert len(fringe) == len(phases), "Expected len(fringe) == len(phases)"
        # All probabilities should be in [0, 1]
        assert np.all(fringe >= 0) and np.all(fringe <= 1), (
            "Expected np.all(fringe >= 0) and np.all(fringe <= 1)"
        )

    def test_interference_with_noon_state(self) -> None:
        state = noon_state(N=2, max_photons=3)
        phases = np.linspace(0, 2 * np.pi, 20)
        fringe = compute_interference_fringe(
            phase_range=phases,
            initial_system_state=state,
            theta=np.pi / 4,
            phi_bs=0.0,
            g=0.0,
            interaction_time=0.0,
            coupling_type="phase_coupling",
            max_photons=3,
            ancilla_dim=2,
        )
        assert len(fringe) == 20, "Expected len(fringe) == 20"


class TestInputStatePreparation:
    """Test input state preparation function."""

    def test_prepare_vacuum(self) -> None:
        state = prepare_input_state("vacuum", max_photons=2)
        assert validate_state(state), "Condition failed: validate_state(state)"

    def test_prepare_single_photon(self) -> None:
        state = prepare_input_state("single_photon", max_photons=2, mode=0)
        assert validate_state(state), "Condition failed: validate_state(state)"

    def test_prepare_coherent(self) -> None:
        state = prepare_input_state("coherent", max_photons=5, alpha=0.5)
        assert validate_state(state), "Condition failed: validate_state(state)"

    def test_prepare_fock(self) -> None:
        state = prepare_input_state("fock", max_photons=5, n_particles=3)
        assert validate_state(state), "Condition failed: validate_state(state)"

    def test_prepare_noon(self) -> None:
        state = prepare_input_state("noon", max_photons=5, n_particles=2)
        assert validate_state(state), "Condition failed: validate_state(state)"


class TestValidation:
    """Test validation functions."""

    def test_validate_normalized_state(self) -> None:
        state = np.array([1, 0, 0], dtype=complex)
        assert validate_state(state), "Condition failed: validate_state(state)"

    def test_validate_unnormalized_state(self) -> None:
        state = np.array([1, 1, 1], dtype=complex)
        assert not validate_state(state), "validate_state(state) should be falsy"

    def test_validate_unitary_matrix(self) -> None:
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        assert validate_unitary(U), "Condition failed: validate_unitary(U)"

    def test_validate_non_unitary(self) -> None:
        U = np.array([[1, 1], [0, 1]], dtype=complex)
        assert not validate_unitary(U), "validate_unitary(U) should be falsy"

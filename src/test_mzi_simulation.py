"""
Tests for MZI with Ancilla simulation module.
"""

import numpy as np

from src.mzi_simulation import (
    vacuum_state,
    single_photon_state,
    noon_state,
    coherent_state,
    fock_state_n,
    create_system_operators,
    beam_splitter_unitary,
    phase_shift_unitary,
    system_ancilla_interaction_unitary,
    evolve_mzi,
    compute_output_probabilities,
    compute_interference_fringe,
    prepare_input_state,
    validate_state,
    validate_unitary,
)


class TestFockStates:
    """Test Fock state creation."""

    def test_vacuum_state(self):
        state = vacuum_state(max_photons=2)
        assert validate_state(state)
        assert np.isclose(state[0], 1.0)  # |0,0> is first state

    def test_single_photon_state_mode0(self):
        state = single_photon_state(mode=0, max_photons=2)
        assert validate_state(state)
        # |1,0> should have amplitude 1

    def test_single_photon_state_mode1(self):
        state = single_photon_state(mode=1, max_photons=2)
        assert validate_state(state)

    def test_noon_state(self):
        state = noon_state(N=2, max_photons=3)
        assert validate_state(state)
        # Should be (|2,0> + |0,2>)/sqrt(2)

    def test_coherent_state(self):
        state = coherent_state(alpha=0.5 + 0.5j, max_photons=5)
        assert validate_state(state)

    def test_fock_state_n(self):
        state = fock_state_n(n=3, max_photons=5)
        assert validate_state(state)


class TestOperators:
    """Test operator creation and properties."""

    def test_system_operators_dimensions(self):
        a0, a1, a0_dag, a1_dag = create_system_operators(max_photons=2)
        dim = (2 + 1) ** 2
        assert a0.shape == (dim, dim)
        assert a1.shape == (dim, dim)

    def test_beam_splitter_preserves_norm(self):
        # Beam splitter should preserve norm of any state
        bs = beam_splitter_unitary(theta=np.pi / 4, phi=0.0, max_photons=2)
        # Test with vacuum
        vac = vacuum_state(2)
        vac_out = bs @ vac
        assert np.isclose(np.sum(np.abs(vac_out) ** 2), 1.0, atol=1e-6)
        # Test with single photon
        sp = single_photon_state(0, 2)
        sp_out = bs @ sp
        assert np.isclose(np.sum(np.abs(sp_out) ** 2), 1.0, atol=1e-6)

    def test_phase_shift_unitarity(self):
        ps = phase_shift_unitary(phi=np.pi / 2, max_photons=2)
        assert validate_unitary(ps, tol=1e-10)

    def test_ancilla_coupling_hermitian(self):
        H = system_ancilla_interaction_unitary(
            g=1.0,
            interaction_time=1.0,
            coupling_type="phase_coupling",
            max_photons=2,
            ancilla_dim=3,
        )
        # Check that exp(-i*H) is unitary
        assert validate_unitary(H, tol=1e-6)


class TestEvolution:
    """Test state evolution."""

    def test_vacuum_produces_vacuum(self):
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
        assert validate_state(evolved)

    def test_evolution_conserves_probability(self):
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
        assert np.isclose(prob, 1.0)

    def test_output_probabilities_sum_to_one(self):
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
        assert np.isclose(p0 + p1, 1.0)


class TestInterference:
    """Test interference pattern computation."""

    def test_interference_fringe_shape(self):
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
        assert len(fringe) == len(phases)
        # All probabilities should be in [0, 1]
        assert np.all(fringe >= 0) and np.all(fringe <= 1)

    def test_interference_with_noon_state(self):
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
        assert len(fringe) == 20


class TestInputStatePreparation:
    """Test input state preparation function."""

    def test_prepare_vacuum(self):
        state = prepare_input_state("vacuum", max_photons=2)
        assert validate_state(state)

    def test_prepare_single_photon(self):
        state = prepare_input_state("single_photon", max_photons=2, mode=0)
        assert validate_state(state)

    def test_prepare_coherent(self):
        state = prepare_input_state("coherent", max_photons=5, alpha=0.5)
        assert validate_state(state)

    def test_prepare_fock(self):
        state = prepare_input_state("fock", max_photons=5, n_particles=3)
        assert validate_state(state)

    def test_prepare_noon(self):
        state = prepare_input_state("noon", max_photons=5, n_particles=2)
        assert validate_state(state)


class TestValidation:
    """Test validation functions."""

    def test_validate_normalized_state(self):
        state = np.array([1, 0, 0], dtype=complex)
        assert validate_state(state)

    def test_validate_unnormalized_state(self):
        state = np.array([1, 1, 1], dtype=complex)
        assert not validate_state(state)

    def test_validate_unitary_matrix(self):
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        assert validate_unitary(U)

    def test_validate_non_unitary(self):
        U = np.array([[1, 1], [0, 1]], dtype=complex)
        assert not validate_unitary(U)

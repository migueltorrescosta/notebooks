"""
Tests for MZI with Ancilla simulation module.
"""

import numpy as np
import pytest
import qutip

from src.physics.mzi_states import standard_twin_fock_state
from src.utils.validators import validate_state_mzi

from .mzi_simulation import (
    apply_phase_shift_mzi,
    beam_splitter_unitary,
    compute_interference_fringe,
    compute_mzi_sensitivity_grid,
    compute_output_probabilities,
    create_system_operators,
    evolve_mzi,
    noon_state,
    phase_shift_unitary,
    prepare_input_state,
    simple_mzi_evolution,
    system_ancilla_interaction_unitary,
)


# Test helpers: construct two-mode states via QuTiP (not wrappers of removed functions)
def _make_vacuum(max_photons: int) -> np.ndarray:
    dim = max_photons + 1
    return qutip.tensor(qutip.fock(dim, 0), qutip.fock(dim, 0)).full().ravel()


def _make_single_photon(mode: int, max_photons: int) -> np.ndarray:
    dim = max_photons + 1
    if mode == 0:
        return qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()
    return qutip.tensor(qutip.fock(dim, 0), qutip.fock(dim, 1)).full().ravel()


class TestStateCreation:
    """Test state creation via QuTiP and noon_state."""

    def test_vacuum_state(self) -> None:
        state = _make_vacuum(max_photons=2)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"
        assert state[0] == pytest.approx(1.0)  # |0,0> is first state

    def test_single_photon_state_mode0(self) -> None:
        state = _make_single_photon(mode=0, max_photons=2)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_single_photon_state_mode1(self) -> None:
        state = _make_single_photon(mode=1, max_photons=2)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_noon_state(self) -> None:
        state = noon_state(N=2, max_photons=3)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"
        # Should be (|2,0> + |0,2>)/sqrt(2)

    def test_coherent_state(self) -> None:
        dim = 5 + 1
        state = (
            qutip.tensor(qutip.coherent(dim, 0.5), qutip.fock(dim, 0)).full().ravel()
        )
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_fock_state_n(self) -> None:
        dim = 5 + 1
        state = qutip.tensor(qutip.fock(dim, 3), qutip.fock(dim, 0)).full().ravel()
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"


class TestOperators:
    """Test operator creation and properties."""

    def test_system_operators_dimensions(self) -> None:
        a0, a1, _a0_dag, _a1_dag = create_system_operators(max_photons=2)
        dim = (2 + 1) ** 2
        assert a0.shape == (dim, dim)
        assert a1.shape == (dim, dim)

    def test_beam_splitter_preserves_norm(self) -> None:
        # Beam splitter should preserve norm of any state
        bs = beam_splitter_unitary(theta=np.pi / 4, phi_bs=0.0, max_photons=2)
        # Test with vacuum
        vac = _make_vacuum(2)
        vac_out = bs @ vac
        assert np.sum(np.abs(vac_out) ** 2) == pytest.approx(1.0, abs=1e-6), (
            "Expected np.sum(np.abs(vac_out) ** 2) == pytest.approx(1.0, abs=1e-6)"
        )
        # Test with single photon
        sp = _make_single_photon(0, 2)
        sp_out = bs @ sp
        assert np.sum(np.abs(sp_out) ** 2) == pytest.approx(1.0, abs=1e-6), (
            "Expected np.sum(np.abs(sp_out) ** 2) == pytest.approx(1.0, abs=1e-6)"
        )

    def test_phase_shift_unitarity(self) -> None:
        ps = phase_shift_unitary(phi_phase=np.pi / 2, max_photons=2)
        assert np.allclose(ps @ ps.conj().T, np.eye(ps.shape[0]), atol=1e-10), (
            "Phase shift unitary must satisfy U U† = I"
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
        assert np.allclose(H @ H.conj().T, np.eye(H.shape[0]), atol=1e-6), (
            "Ancilla coupling unitary must satisfy U U† = I"
        )


class TestEvolution:
    """Test state evolution."""

    def test_vacuum_produces_vacuum(self) -> None:
        state = _make_vacuum(max_photons=1)
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
        assert validate_state_mzi(evolved), (
            "Condition failed: validate_state_mzi(evolved)"
        )

    def test_evolution_conserves_probability(self) -> None:
        state = _make_vacuum(max_photons=2)
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
        assert prob == pytest.approx(1.0)

    def test_output_probabilities_sum_to_one(self) -> None:
        state = _make_vacuum(max_photons=2)
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
        assert p0 + p1 == pytest.approx(1.0)


class TestInterference:
    """Test interference pattern computation."""

    def test_interference_fringe_shape(self) -> None:
        state = _make_vacuum(max_photons=2)
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
        assert len(fringe) == 20


class TestInputStatePreparation:
    """Test input state preparation function."""

    def test_prepare_vacuum(self) -> None:
        state = prepare_input_state("vacuum", max_photons=2)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_prepare_make_single_photon(self) -> None:
        state = prepare_input_state("single_photon", max_photons=2, mode=0)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_prepare_coherent(self) -> None:
        state = prepare_input_state("coherent", max_photons=5, alpha=0.5)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_prepare_fock(self) -> None:
        state = prepare_input_state("fock", max_photons=5, n_particles=3)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_prepare_noon(self) -> None:
        state = prepare_input_state("noon", max_photons=5, n_particles=2)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"


class TestValidation:
    """Test validation functions."""

    def test_validate_normalized_state(self) -> None:
        state = np.array([1, 0, 0], dtype=complex)
        assert validate_state_mzi(state), "Condition failed: validate_state_mzi(state)"

    def test_validate_unnormalized_state(self) -> None:
        state = np.array([1, 1, 1], dtype=complex)
        assert not validate_state_mzi(state), (
            "validate_state_mzi(state) should be falsy"
        )


class TestAncillaFreeMzi:
    """Tests for ancilla-free MZI functions promoted from report modules."""

    def test_apply_phase_shift_preserves_norm(self) -> None:
        """Phase shift must preserve state norm."""
        state = qutip.tensor(qutip.fock(3, 1), qutip.fock(3, 0)).full().ravel()
        shifted = apply_phase_shift_mzi(state, phi=0.5, max_photons=2)
        assert np.isclose(np.linalg.norm(shifted), 1.0, rtol=1e-10)

    def test_apply_phase_shift_vacuum_invariant(self) -> None:
        """Vacuum state should be invariant under any phase shift."""
        vac = qutip.tensor(qutip.fock(3, 0), qutip.fock(3, 0)).full().ravel()
        shifted = apply_phase_shift_mzi(vac, phi=np.pi, max_photons=2)
        assert np.allclose(shifted, vac, atol=1e-15)

    def test_simple_mzi_evolution_preserves_norm(self) -> None:
        """MZI evolution must preserve state norm for all inputs."""
        for max_photons in [1, 2, 3]:
            vac = (
                qutip.tensor(
                    qutip.fock(max_photons + 1, 0), qutip.fock(max_photons + 1, 0)
                )
                .full()
                .ravel()
            )
            out = simple_mzi_evolution(vac, omega=0.5, max_photons=max_photons)
            assert np.isclose(np.linalg.norm(out), 1.0, rtol=1e-10), (
                f"Failed at M={max_photons}"
            )

    def test_simple_mzi_evolution_skip_bs1(self) -> None:
        """When skip_bs1=True, only phase + BS2 are applied."""
        max_photons = 3
        state = (
            qutip.tensor(qutip.fock(max_photons + 1, 1), qutip.fock(max_photons + 1, 0))
            .full()
            .ravel()
        )
        # Compare: full vs skip_bs1
        full = simple_mzi_evolution(
            state.copy(), omega=0.5, max_photons=max_photons, skip_bs1=False
        )
        skipped = simple_mzi_evolution(
            state.copy(), omega=0.5, max_photons=max_photons, skip_bs1=True
        )
        # They should differ (skip_bs1 removes the first BS)
        assert not np.allclose(full, skipped, atol=1e-10)

    def test_simple_mzi_phi0_equals_bs_only(self) -> None:
        """At omega=0, evolution should be equivalent to applying BS2@BS1."""
        max_photons = 2
        state = (
            qutip.tensor(qutip.fock(max_photons + 1, 1), qutip.fock(max_photons + 1, 0))
            .full()
            .ravel()
        )
        bs = beam_splitter_unitary(np.pi / 4, 0.0, max_photons)
        out_mzi = simple_mzi_evolution(state, omega=0.0, max_photons=max_photons)
        out_bs2_bs1 = bs @ bs @ state  # BS1 then BS2 (both same at π/4)
        assert np.allclose(out_mzi, out_bs2_bs1, atol=1e-10)

    def test_compute_mzi_sensitivity_grid_return_keys(self) -> None:
        """The return dict must contain all expected keys."""
        max_photons = 2
        vac = (
            qutip.tensor(qutip.fock(max_photons + 1, 0), qutip.fock(max_photons + 1, 0))
            .full()
            .ravel()
        )
        omega_grid = np.linspace(0.1, 1.0, 5)
        result = compute_mzi_sensitivity_grid(vac, omega_grid, max_photons, t_hold=1.0)
        expected_keys = {
            "omega_values",
            "expectation_values",
            "variance_values",
            "derivative_values",
            "delta_omega_ep",
            "delta_omega_q",
            "fisher_quantum",
            "fisher_classical",
            "delta_omega_c",
        }
        assert set(result.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_compute_mzi_sensitivity_grid_qfi_bound(self) -> None:
        """QFI bound must be consistent with the J_z variance."""
        max_photons = 2
        state = (
            qutip.tensor(qutip.fock(max_photons + 1, 1), qutip.fock(max_photons + 1, 0))
            .full()
            .ravel()
        )
        bs = beam_splitter_unitary(np.pi / 4, 0.0, max_photons)
        probe = bs @ state
        # Compute Var(J_z) manually
        jz_local = np.diag(
            np.array(
                [
                    (n1 - n2) / 2.0
                    for n1 in range(max_photons + 1)
                    for n2 in range(max_photons + 1)
                ]
            )
        )
        mean = np.conj(probe) @ jz_local @ probe
        var = np.real(np.conj(probe) @ (jz_local @ jz_local) @ probe - mean**2)
        expected_fq = 4.0 * 10.0**2 * var
        omega_grid = np.linspace(0.1, 1.0, 3)
        result = compute_mzi_sensitivity_grid(
            state, omega_grid, max_photons, t_hold=10.0
        )
        assert np.isclose(result["fisher_quantum"], expected_fq, rtol=1e-10), (
            f"QFI mismatch: {result['fisher_quantum']} vs {expected_fq}"
        )

    def test_simple_mzi_reuses_bs(self) -> None:
        """When bs is passed explicitly, it should be reused."""
        max_photons = 2
        state = (
            qutip.tensor(qutip.fock(max_photons + 1, 1), qutip.fock(max_photons + 1, 0))
            .full()
            .ravel()
        )
        bs = beam_splitter_unitary(np.pi / 4, 0.0, max_photons)
        out1 = simple_mzi_evolution(state, omega=0.5, max_photons=max_photons, bs=bs)
        out2 = simple_mzi_evolution(state, omega=0.5, max_photons=max_photons, bs=bs)
        assert np.allclose(out1, out2, atol=1e-10)

    def test_twin_fock_simple_mzi_roundtrip(self) -> None:
        """Standard twin-fock state through MZI at omega=0 should be invariant."""
        N = 4
        max_photons = N
        state = standard_twin_fock_state(N, max_photons)
        out = simple_mzi_evolution(state, omega=0.0, max_photons=max_photons)
        bs = beam_splitter_unitary(np.pi / 4, 0.0, max_photons)
        expected = bs @ bs @ state
        assert np.allclose(out, expected, atol=1e-10), "MZI at omega=0 = BS2@BS1"

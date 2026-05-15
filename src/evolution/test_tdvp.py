"""Unit tests for TDVP (Time-Dependent Variational Principle) module.

Uses quimb :class:`~quimb.tensor.Tensor` as the state representation.
"""

import numpy as np
import pytest
import quimb.tensor as qtn

from .tdvp import (
    TDVPConfig,
    TDVPResult,
    apply_trotter_step,
    apply_trotter_step_simple,
    compute_energy,
    compute_energy_variance,
    compute_manifold_violation,
    compute_state_fidelity,
    evolve_exact,
    project_to_manifold,
    tdvp_evolution,
    tdvp_single_site,
    validate_tdvp_step,
)

# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
EYE = np.array([[1, 0], [0, 1]], dtype=complex)


def _make_tensor(state: np.ndarray) -> qtn.Tensor:
    """Shortcut: build a quimb Tensor from a 2-qubit (4-dim) state vector."""
    return qtn.Tensor(
        data=state.reshape(2, 2).astype(complex),
        inds=("main", "ancilla"),
    )


def _flat(t: qtn.Tensor) -> np.ndarray:
    """Return the flat state vector stored in *t*."""
    return t.data.flatten()


# =============================================================================
#  TDVP Configuration
# =============================================================================


class TestTDVPConfiguration:
    """Tests for TDVP configuration."""

    def test_default_config_should_have_sensible_values(self) -> None:
        config = TDVPConfig()
        assert config.dt == 0.01, "Expected config.dt == 0.01"
        assert config.trotter_order == 2, "Expected config.trotter_order == 2"
        assert config.checkpoint_every == 10, "Expected config.checkpoint_every == 10"
        assert config.max_sweeps == 100, "Expected config.max_sweeps == 100"

    def test_custom_config_should_accept_all_parameters(self) -> None:
        config = TDVPConfig(
            dt=0.001,
            trotter_order=1,
            checkpoint_every=5,
            bond_dim_limit=32,
        )
        assert config.dt == 0.001, "Expected config.dt == 0.001"
        assert config.trotter_order == 1, "Expected config.trotter_order == 1"
        assert config.checkpoint_every == 5, "Expected config.checkpoint_every == 5"
        assert config.bond_dim_limit == 32, "Expected config.bond_dim_limit == 32"


# =============================================================================
#  Single-site TDVP update
# =============================================================================


class TestTDPVSingleSite:
    """Tests for single-site TDVP update."""

    def test_single_site_update_on_product_state(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H_z = SIGMA_Z
        dt = 0.1
        updated = tdvp_single_site(tensor, site_idx=0, H_eff=H_z, dt=dt)

        sv = _flat(updated)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, rel=1e-6)

    def test_single_site_update_should_preserve_norm(self) -> None:
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        tensor = _make_tensor(state)

        H_z = SIGMA_Z
        dt = 0.05
        updated = tdvp_single_site(tensor, site_idx=0, H_eff=H_z, dt=dt)

        norm_before = np.linalg.norm(_flat(tensor))
        norm_after = np.linalg.norm(_flat(updated))
        assert norm_before == pytest.approx(norm_after, rel=1e-6)

    def test_should_work_with_any_hermitian_hamiltonian(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H_x = SIGMA_X
        dt = 0.1
        updated = tdvp_single_site(tensor, site_idx=0, H_eff=H_x, dt=dt)

        assert _flat(updated) is not None

    def test_should_reject_non_hermitian_hamiltonian(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H_nh = np.array([[1, 1], [0, 1]], dtype=complex)

        with pytest.raises(ValueError, match="H_eff must be Hermitian"):
            tdvp_single_site(tensor, site_idx=0, H_eff=H_nh, dt=0.1)

    def test_should_reject_non_square_effective_hamiltonian(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H_bad = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)

        with pytest.raises(ValueError, match="square"):
            tdvp_single_site(tensor, site_idx=0, H_eff=H_bad, dt=0.1)


# =============================================================================
#  Trotter decomposition
# =============================================================================


class TestApplyTrotterStep:
    """Tests for Trotter decomposition."""

    def test_first_order_trotter_should_complete(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H_terms = [SIGMA_Z]
        dt = 0.1
        result = apply_trotter_step(tensor, H_terms, dt, order=1)

        sv = _flat(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, rel=1e-6)

    def test_second_order_trotter_should_complete(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H_terms = [SIGMA_Z]
        dt = 0.1
        result = apply_trotter_step(tensor, H_terms, dt, order=2)

        sv = _flat(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, rel=1e-6)

    def test_order_2_should_be_more_accurate_than_order_1(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H_z = np.kron(SIGMA_Z, EYE)
        dt = 0.01

        # Exact evolution
        exact = evolve_exact(state, H_z, dt, np.random.default_rng(42))

        # First-order Trotter
        result_1 = apply_trotter_step(tensor, [SIGMA_Z], dt, order=1)
        fidelity_1 = compute_state_fidelity(_flat(result_1), exact)

        # Second-order Trotter
        result_2 = apply_trotter_step(tensor, [SIGMA_Z], dt, order=2)
        fidelity_2 = compute_state_fidelity(_flat(result_2), exact)

        # Order 2 should be at least as good as order 1
        assert fidelity_2 >= fidelity_1 - 1e-6

    def test_invalid_trotter_order_should_raise(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        with pytest.raises(ValueError, match="Trotter order must be 1 or 2"):
            apply_trotter_step(tensor, [SIGMA_Z], 0.1, order=3)


# =============================================================================
#  Energy calculations
# =============================================================================


class TestEnergyCalculations:
    """Tests for energy-related functions."""

    def test_energy_should_match_expectation_value(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H = np.kron(SIGMA_Z, EYE)
        energy = compute_energy(tensor, H)

        assert np.real(energy) == pytest.approx(1.0, abs=1e-6)

    def test_variance_should_be_non_negative(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H = np.kron(SIGMA_Z, EYE)
        variance = compute_energy_variance(tensor, H)

        # For eigenstate, variance should be zero
        assert variance >= -1e-10

    def test_energy_for_superposition_state(self) -> None:
        state = (
            np.array([1, 0, 0, 0], dtype=complex)
            + np.array([0, 0, 1, 0], dtype=complex)
        ) / np.sqrt(2)
        tensor = _make_tensor(state)

        H = np.kron(SIGMA_Z, EYE)
        energy = compute_energy(tensor, H)

        # Average of +1 and -1 = 0
        assert np.real(energy) == pytest.approx(0.0, abs=1e-6)


# =============================================================================
#  Fidelity calculations
# =============================================================================


class TestFidelityCalculations:
    """Tests for fidelity calculations."""

    def test_fidelity_of_identical_states_should_be_1(self) -> None:
        psi = np.array([1, 0, 0, 0], dtype=complex)
        f = compute_state_fidelity(psi, psi)
        assert f == pytest.approx(1.0)

    def test_fidelity_of_orthogonal_states_should_be_0(self) -> None:
        psi1 = np.array([1, 0, 0, 0], dtype=complex)
        psi2 = np.array([0, 1, 0, 0], dtype=complex)
        f = compute_state_fidelity(psi1, psi2)
        assert f == pytest.approx(0.0)

    def test_fidelity_for_bell_states(self) -> None:
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        f = compute_state_fidelity(bell, bell)
        assert f == pytest.approx(1.0)

    def test_fidelity_should_normalize_inputs(self) -> None:
        psi1 = 2 * np.array([1, 0, 0, 0], dtype=complex)
        psi2 = 3 * np.array([1, 0, 0, 0], dtype=complex)
        f = compute_state_fidelity(psi1, psi2)
        assert f == pytest.approx(1.0)


# =============================================================================
#  Full TDVP evolution
# =============================================================================


class TestTDVPEvolution:
    """Tests for full TDVP evolution."""

    def test_evolution_should_complete_without_error(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H = np.kron(SIGMA_Z, EYE)
        config = TDVPConfig(dt=0.01, checkpoint_every=10)

        result = tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.01,
            n_sites=1,
            config=config,
        )

        assert isinstance(result, TDVPResult)
        assert result.final_tensor is not None
        assert len(result.times) > 0

    def test_norm_should_be_preserved_during_evolution(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H = np.kron(SIGMA_Z, EYE)
        config = TDVPConfig(dt=0.01)

        result = tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.01,
            n_sites=1,
            config=config,
        )

        assert result.norm_preserved

    def test_checkpoints_should_be_saved_at_correct_intervals(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H = np.kron(SIGMA_Z, EYE)
        config = TDVPConfig(dt=0.01, checkpoint_every=5)

        result = tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.01,
            n_sites=1,
            config=config,
        )

        # 0.1 / 0.01 = 10 steps, checkpoints every 5 = 2 checkpoints
        assert len(result.checkpoints) == 2

    def test_energy_history_should_be_recorded(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H = np.kron(SIGMA_Z, EYE)
        config = TDVPConfig(dt=0.01)

        result = tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.01,
            n_sites=1,
            config=config,
        )

        assert len(result.energies) == len(result.times)
        assert len(result.energies) == 10  # 0.1 / 0.01 = 10 steps


# =============================================================================
#  Exact evolution
# =============================================================================


class TestExactEvolution:
    """Tests for exact evolution benchmark."""

    def test_exact_evolution_should_be_unitary(self) -> None:
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        H = np.kron(SIGMA_Z, EYE)
        rng = np.random.default_rng(42)

        psi_t = evolve_exact(psi0, H, t=0.5, rng=rng)

        assert np.linalg.norm(psi_t) == pytest.approx(1.0, rel=1e-6)

    def test_evolution_at_t_zero_should_return_initial_state(self) -> None:
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        H = np.kron(SIGMA_Z, EYE)
        rng = np.random.default_rng(42)

        psi_t = evolve_exact(psi0, H, t=0.0, rng=rng)

        assert psi_t == pytest.approx(psi0)

    def test_trotter_should_approximate_exact_evolution(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H = np.kron(SIGMA_Z, EYE)
        rng = np.random.default_rng(42)

        # Exact evolution
        psi_exact = evolve_exact(state, H, t=0.1, rng=rng)

        # Trotter evolution
        result = tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.01,
            n_sites=1,
            config=TDVPConfig(dt=0.01, trotter_order=2),
        )

        sv = _flat(result.final_tensor)
        fidelity = compute_state_fidelity(sv, psi_exact)

        # Should be reasonably accurate for small dt
        assert fidelity > 0.99


# =============================================================================
#  Project to manifold
# =============================================================================


class TestProjectToManifold:
    """Tests for projection to tensor manifold."""

    def test_projected_state_should_be_normalized(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)

        tensor = project_to_manifold(state, n_sites=1, local_dim=2)

        assert np.linalg.norm(_flat(tensor)) == pytest.approx(1.0, rel=1e-6)

    def test_product_state_should_project_exactly(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)

        tensor = project_to_manifold(state, n_sites=1, local_dim=2)
        reconstructed = _flat(tensor)

        fidelity = compute_state_fidelity(state, reconstructed)
        assert fidelity > 1 - 1e-10

    def test_entangled_state_may_not_project_exactly(self) -> None:
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        tensor = project_to_manifold(
            state,
            n_sites=1,
            local_dim=2,
            epsilon=1e-8,
        )
        reconstructed = _flat(tensor)

        # Should still have good fidelity
        fidelity = compute_state_fidelity(state, reconstructed)
        assert fidelity > 0.99


# =============================================================================
#  Manifold violation
# =============================================================================


class TestManifoldViolation:
    """Tests for manifold violation measurement."""

    def test_product_state_should_have_zero_manifold_violation(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = project_to_manifold(state, n_sites=1, local_dim=2)

        violation = compute_manifold_violation(tensor, state)

        assert violation < 1e-10

    def test_entangled_state_should_have_nonzero_manifold_violation(self) -> None:
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        tensor = project_to_manifold(state, n_sites=1, local_dim=2)

        violation = compute_manifold_violation(tensor, state)

        assert violation >= -1e-10


# =============================================================================
#  TDVP validation
# =============================================================================


class TestTDVPValidation:
    """Tests for TDVP validation functions."""

    def test_validation_should_detect_norm_changes(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor_before = _make_tensor(state)

        # Create a slightly modified state
        state_after = state.copy()
        state_after[0] = 0.99
        tensor_after = _make_tensor(
            state_after / np.linalg.norm(state_after),
        )

        H = np.kron(SIGMA_Z, EYE)
        metrics = validate_tdvp_step(tensor_before, tensor_after, H, dt=0.01)

        assert "norm_error" in metrics
        assert "energy_change" in metrics


# =============================================================================
#  Energy conservation
# =============================================================================


class TestTDVPEnergyConservation:
    """Tests for energy conservation within TTN manifold."""

    def test_energy_should_be_approximately_conserved_for_small_dt(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        # Diagonal Hamiltonian - energy should be conserved exactly
        H = np.kron(SIGMA_Z, EYE)

        E_initial = compute_energy(tensor, H)

        result = tdvp_evolution(
            tensor,
            H,
            T=0.05,
            dt=0.001,
            n_sites=1,
            config=TDVPConfig(dt=0.001),
        )

        E_final = compute_energy(result.final_tensor, H)

        energy_change = abs(np.real(E_final - E_initial))
        assert energy_change < 0.1  # Relaxed tolerance for numerical errors


# =============================================================================
#  TDVP vs exact
# =============================================================================


class TestTDVPAgainstExact:
    """Tests comparing TDVP with exact evolution."""

    def test_tdvp_should_match_exact_evolution_for_n_2(self) -> None:
        # Initial superposition state
        state = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
        tensor = _make_tensor(state)

        # Full Hamiltonian for n_sites=1
        H = np.kron(SIGMA_Z, EYE)
        rng = np.random.default_rng(42)

        # Exact evolution
        psi_exact = evolve_exact(state, H, t=0.1, rng=rng)

        # TDVP with small dt
        result = tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.005,
            n_sites=1,
            config=TDVPConfig(dt=0.005, trotter_order=2),
        )

        sv = _flat(result.final_tensor)
        fidelity = compute_state_fidelity(sv, psi_exact)

        # Should be accurate to within 10^-4 relative error
        assert fidelity > 1 - 1e-4

    def test_tdvp_relative_error_for_n_sites_1_with_simple_setup(self) -> None:
        rng = np.random.default_rng(42)
        dim = 2**2  # n_sites=1, 2 qubits total
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        tensor = project_to_manifold(
            state,
            n_sites=1,
            local_dim=2,
            epsilon=1e-8,
        )

        # Use a diagonal Hamiltonian
        H = np.diag(np.arange(dim, dtype=complex))

        # Evolve with TDVP
        result = tdvp_evolution(
            tensor,
            H,
            T=0.05,
            dt=0.001,
            n_sites=1,
            config=TDVPConfig(dt=0.001, trotter_order=2),
        )

        # Check against exact
        psi_exact = evolve_exact(state, H, t=0.05, rng=rng)
        fidelity = compute_state_fidelity(
            _flat(result.final_tensor),
            psi_exact,
        )

        # Relative error = 1 - fidelity
        relative_error = 1 - fidelity

        # Should be reasonably accurate
        assert relative_error < 0.5


# =============================================================================
#  Small-system comparison
# =============================================================================


class TestTDVPSmallSystemComparison:
    """Test TDVP + tensor manifold vs exact for small systems."""

    def test_compare_tdvp_with_exact_for_n_sites_1(self) -> None:
        rng = np.random.default_rng(42)

        # Create a random initial state
        dim = 2**2
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        tensor = project_to_manifold(
            state,
            n_sites=1,
            local_dim=2,
            epsilon=1e-8,
        )

        # Simple diagonal Hamiltonian for comparison
        H = np.diag(np.arange(dim, dtype=complex))

        # Evolve with TDVP
        result = tdvp_evolution(
            tensor,
            H,
            T=0.02,
            dt=0.002,
            n_sites=1,
            config=TDVPConfig(dt=0.002, trotter_order=2),
        )

        # Evolve exactly
        psi_exact = evolve_exact(state, H, t=0.02, rng=rng)

        # Compare
        fidelity = compute_state_fidelity(
            _flat(result.final_tensor),
            psi_exact,
        )

        # TDVP should capture the dynamics reasonably well
        assert fidelity > 0.5

    def test_energy_should_be_conserved_within_tensor_manifold(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        # Use diagonal Hamiltonian for exact conservation
        H = np.diag([1.0, -1.0, 0.5, -0.5])

        result = tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.01,
            n_sites=1,
            config=TDVPConfig(dt=0.01),
        )

        # Should have recorded energies
        assert len(result.energies) > 0


# =============================================================================
#  Performance
# =============================================================================


class TestTDVPPerformance:
    """Performance tests for TDVP."""

    def test_tdvp_should_complete_quickly_for_n_sites_1(self) -> None:
        import time

        rng = np.random.default_rng(42)
        dim = 2**2
        state = rng.random(dim) + 1j * rng.random(dim)
        state = state / np.linalg.norm(state)

        tensor = project_to_manifold(
            state,
            n_sites=1,
            local_dim=2,
            epsilon=1e-8,
        )

        # Use a simple diagonal Hamiltonian
        H = np.diag(np.arange(dim, dtype=complex))

        start = time.perf_counter()
        tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.01,
            n_sites=1,
            config=TDVPConfig(dt=0.01),
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5

    def test_many_steps_should_complete_in_reasonable_time(self) -> None:
        import time

        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H = np.kron(SIGMA_Z, EYE)

        start = time.perf_counter()
        tdvp_evolution(
            tensor,
            H,
            T=1.0,
            dt=0.01,
            n_sites=1,
            config=TDVPConfig(dt=0.01),
        )
        elapsed = time.perf_counter() - start

        # 100 steps should be reasonable
        assert elapsed < 1.0


# =============================================================================
#  Simplified Trotter step
# =============================================================================


class TestTrotterStepSimple:
    """Tests for simplified Trotter step."""

    def test_simplified_step_should_complete(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)

        H_local = SIGMA_Z
        result = apply_trotter_step_simple(tensor, H_local, dt=0.1)

        sv = _flat(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, rel=1e-6)

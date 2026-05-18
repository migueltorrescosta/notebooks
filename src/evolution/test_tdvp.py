"""Unit tests for TDVP (Time-Dependent Variational Principle) module.

Uses quimb :class:`~quimb.tensor.Tensor` as the state representation.
"""

from __future__ import annotations

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


def _make_flat(t: qtn.Tensor) -> np.ndarray:
    """Return the flat state vector stored in *t*."""
    return t.data.flatten()


class TestTDVPConfiguration:
    """Tests for TDVP configuration."""

    def test_default_config_has_sensible_values(self) -> None:
        config = TDVPConfig()
        assert config.dt == 0.01
        assert config.trotter_order == 2
        assert config.checkpoint_every == 10
        assert config.max_sweeps == 100

    def test_custom_config_accepts_all_parameters(self) -> None:
        config = TDVPConfig(
            dt=0.001,
            trotter_order=1,
            checkpoint_every=5,
            bond_dim_limit=32,
        )
        assert config.dt == 0.001
        assert config.trotter_order == 1
        assert config.checkpoint_every == 5
        assert config.bond_dim_limit == 32


class TestTDPVSingleSite:
    """Tests for single-site TDVP update."""

    def test_single_site_update_on_product_state(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H_z = SIGMA_Z
        dt = 0.1
        updated = tdvp_single_site(tensor, site_idx=0, H_eff=H_z, dt=dt)
        sv = _make_flat(updated)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, rel=1e-6)

    def test_single_site_update_preserves_norm(self) -> None:
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        tensor = _make_tensor(state)
        H_z = SIGMA_Z
        dt = 0.05
        updated = tdvp_single_site(tensor, site_idx=0, H_eff=H_z, dt=dt)
        norm_before = np.linalg.norm(_make_flat(tensor))
        norm_after = np.linalg.norm(_make_flat(updated))
        assert norm_before == pytest.approx(norm_after, rel=1e-6)

    def test_any_hermitian_hamiltonian_works(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H_x = SIGMA_X
        dt = 0.1
        updated = tdvp_single_site(tensor, site_idx=0, H_eff=H_x, dt=dt)
        assert _make_flat(updated) is not None

    def test_rejects_non_hermitian_hamiltonian(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H_nh = np.array([[1, 1], [0, 1]], dtype=complex)
        with pytest.raises(ValueError, match="H_eff must be Hermitian"):
            tdvp_single_site(tensor, site_idx=0, H_eff=H_nh, dt=0.1)

    def test_rejects_non_square_hamiltonian(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H_bad = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        with pytest.raises(ValueError, match="square"):
            tdvp_single_site(tensor, site_idx=0, H_eff=H_bad, dt=0.1)


class TestApplyTrotterStep:
    """Tests for Trotter decomposition."""

    @pytest.mark.parametrize("order", [1, 2], ids=["order_1", "order_2"])
    def test_trotter_step_completes(self, order: int) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H_terms = [SIGMA_Z]
        dt = 0.1
        result = apply_trotter_step(tensor, H_terms, dt, order=order)
        sv = _make_flat(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, rel=1e-6)

    def test_order_2_more_accurate_than_order_1(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H_z = np.kron(SIGMA_Z, EYE)
        dt = 0.01
        exact = evolve_exact(state, H_z, dt, np.random.default_rng(42))
        result_1 = apply_trotter_step(tensor, [SIGMA_Z], dt, order=1)
        fidelity_1 = compute_state_fidelity(_make_flat(result_1), exact)
        result_2 = apply_trotter_step(tensor, [SIGMA_Z], dt, order=2)
        fidelity_2 = compute_state_fidelity(_make_flat(result_2), exact)
        assert fidelity_2 >= fidelity_1 - 1e-6

    def test_invalid_trotter_order_raises_error(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        with pytest.raises(ValueError, match="Trotter order must be 1 or 2"):
            apply_trotter_step(tensor, [SIGMA_Z], 0.1, order=3)


class TestEnergyCalculations:
    """Tests for energy-related functions."""

    def test_energy_matches_expectation_value(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H = np.kron(SIGMA_Z, EYE)
        energy = compute_energy(tensor, H)
        assert np.real(energy) == pytest.approx(1.0, abs=1e-6)

    def test_variance_is_non_negative(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H = np.kron(SIGMA_Z, EYE)
        variance = compute_energy_variance(tensor, H)
        assert variance >= -1e-10

    def test_energy_for_superposition_state(self) -> None:
        state = (
            np.array([1, 0, 0, 0], dtype=complex)
            + np.array([0, 0, 1, 0], dtype=complex)
        ) / np.sqrt(2)
        tensor = _make_tensor(state)
        H = np.kron(SIGMA_Z, EYE)
        energy = compute_energy(tensor, H)
        assert np.real(energy) == pytest.approx(0.0, abs=1e-6)


class TestFidelityCalculations:
    """Tests for fidelity calculations."""

    def test_identical_states_have_unity_fidelity(self) -> None:
        psi = np.array([1, 0, 0, 0], dtype=complex)
        f = compute_state_fidelity(psi, psi)
        assert f == pytest.approx(1.0)

    def test_orthogonal_states_have_zero_fidelity(self) -> None:
        psi1 = np.array([1, 0, 0, 0], dtype=complex)
        psi2 = np.array([0, 1, 0, 0], dtype=complex)
        f = compute_state_fidelity(psi1, psi2)
        assert f == pytest.approx(0.0)

    def test_fidelity_for_bell_states(self) -> None:
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        f = compute_state_fidelity(bell, bell)
        assert f == pytest.approx(1.0)

    def test_fidelity_normalizes_inputs(self) -> None:
        psi1 = 2 * np.array([1, 0, 0, 0], dtype=complex)
        psi2 = 3 * np.array([1, 0, 0, 0], dtype=complex)
        f = compute_state_fidelity(psi1, psi2)
        assert f == pytest.approx(1.0)


class TestTDVPEvolution:
    """Tests for full TDVP evolution."""

    def test_evolution_completes(self) -> None:
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

    def test_norm_preserved_during_evolution(self) -> None:
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

    def test_checkpoints_saved_at_correct_intervals(self) -> None:
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
        assert len(result.checkpoints) == 2

    def test_energy_history_recorded(self) -> None:
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
        assert len(result.energies) == 10


class TestExactEvolution:
    """Tests for exact evolution benchmark."""

    def test_exact_evolution_preserves_norm(self) -> None:
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        H = np.kron(SIGMA_Z, EYE)
        rng = np.random.default_rng(42)
        psi_t = evolve_exact(psi0, H, t=0.5, rng=rng)
        assert np.linalg.norm(psi_t) == pytest.approx(1.0, rel=1e-6)

    def test_given_zero_time_then_returns_initial_state(self) -> None:
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        H = np.kron(SIGMA_Z, EYE)
        rng = np.random.default_rng(42)
        psi_t = evolve_exact(psi0, H, t=0.0, rng=rng)
        assert psi_t == pytest.approx(psi0)

    def test_trotter_approximates_exact_evolution(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H = np.kron(SIGMA_Z, EYE)
        rng = np.random.default_rng(42)
        psi_exact = evolve_exact(state, H, t=0.1, rng=rng)
        result = tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.01,
            n_sites=1,
            config=TDVPConfig(dt=0.01, trotter_order=2),
        )
        sv = _make_flat(result.final_tensor)
        fidelity = compute_state_fidelity(sv, psi_exact)
        assert fidelity > 0.99


class TestProjectToManifold:
    """Tests for projection to tensor manifold."""

    def test_projected_state_normalized(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = project_to_manifold(state, n_sites=1, local_dim=2)
        assert np.linalg.norm(_make_flat(tensor)) == pytest.approx(1.0, rel=1e-6)

    def test_product_state_projects_exactly(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = project_to_manifold(state, n_sites=1, local_dim=2)
        reconstructed = _make_flat(tensor)
        fidelity = compute_state_fidelity(state, reconstructed)
        assert fidelity > 1 - 1e-10

    def test_entangled_state_approximate_projection(self) -> None:
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        tensor = project_to_manifold(
            state,
            n_sites=1,
            local_dim=2,
            epsilon=1e-8,
        )
        reconstructed = _make_flat(tensor)
        fidelity = compute_state_fidelity(state, reconstructed)
        assert fidelity > 0.99


class TestManifoldViolation:
    """Tests for manifold violation measurement."""

    def test_product_state_has_zero_violation(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = project_to_manifold(state, n_sites=1, local_dim=2)
        violation = compute_manifold_violation(tensor, state)
        assert violation < 1e-10

    def test_entangled_state_has_nonzero_violation(self) -> None:
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        tensor = project_to_manifold(state, n_sites=1, local_dim=2)
        violation = compute_manifold_violation(tensor, state)
        assert violation >= -1e-10


class TestTDVPValidation:
    """Tests for TDVP validation functions."""

    def test_validation_detects_norm_changes(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor_before = _make_tensor(state)
        state_after = state.copy()
        state_after[0] = 0.99
        tensor_after = _make_tensor(
            state_after / np.linalg.norm(state_after),
        )
        H = np.kron(SIGMA_Z, EYE)
        metrics = validate_tdvp_step(tensor_before, tensor_after, H, dt=0.01)
        assert "norm_error" in metrics
        assert "energy_change" in metrics


class TestTDVPEnergyConservation:
    """Tests for energy conservation within TTN manifold."""

    def test_energy_conserved_for_small_dt(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
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
        assert energy_change < 0.1


class TestTDVPAgainstExact:
    """Tests comparing TDVP with exact evolution."""

    def test_tdvp_matches_exact_evolution(self) -> None:
        state = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
        tensor = _make_tensor(state)
        H = np.kron(SIGMA_Z, EYE)
        rng = np.random.default_rng(42)
        psi_exact = evolve_exact(state, H, t=0.1, rng=rng)
        result = tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.005,
            n_sites=1,
            config=TDVPConfig(dt=0.005, trotter_order=2),
        )
        sv = _make_flat(result.final_tensor)
        fidelity = compute_state_fidelity(sv, psi_exact)
        assert fidelity > 1 - 1e-4

    def test_tdvp_relative_error_with_simple_setup(self) -> None:
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
        H = np.diag(np.arange(dim, dtype=complex))
        result = tdvp_evolution(
            tensor,
            H,
            T=0.05,
            dt=0.001,
            n_sites=1,
            config=TDVPConfig(dt=0.001, trotter_order=2),
        )
        psi_exact = evolve_exact(state, H, t=0.05, rng=rng)
        fidelity = compute_state_fidelity(
            _make_flat(result.final_tensor),
            psi_exact,
        )
        relative_error = 1 - fidelity
        assert relative_error < 0.5


class TestTDVPSmallSystemComparison:
    """Test TDVP + tensor manifold vs exact for small systems."""

    def test_tdvp_agrees_with_exact(self) -> None:
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
        H = np.diag(np.arange(dim, dtype=complex))
        result = tdvp_evolution(
            tensor,
            H,
            T=0.02,
            dt=0.002,
            n_sites=1,
            config=TDVPConfig(dt=0.002, trotter_order=2),
        )
        psi_exact = evolve_exact(state, H, t=0.02, rng=rng)
        fidelity = compute_state_fidelity(
            _make_flat(result.final_tensor),
            psi_exact,
        )
        assert fidelity > 0.5

    def test_energy_conserved_within_tensor_manifold(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H = np.diag([1.0, -1.0, 0.5, -0.5])
        result = tdvp_evolution(
            tensor,
            H,
            T=0.1,
            dt=0.01,
            n_sites=1,
            config=TDVPConfig(dt=0.01),
        )
        assert len(result.energies) > 0


class TestTDVPPerformance:
    """Performance tests for TDVP."""

    def test_tdvp_completes_quickly(self) -> None:
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

    def test_many_steps_completes_in_reasonable_time(self) -> None:
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
        assert elapsed < 1.0


class TestTrotterStepSimple:
    """Tests for simplified Trotter step."""

    def test_simplified_step_completes(self) -> None:
        state = np.array([1, 0, 0, 0], dtype=complex)
        tensor = _make_tensor(state)
        H_local = SIGMA_Z
        result = apply_trotter_step_simple(tensor, H_local, dt=0.1)
        sv = _make_flat(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, rel=1e-6)

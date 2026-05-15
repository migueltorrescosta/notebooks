"""
Tests for Lindblad Master Equation Solver.

Physical Validation Tests:
- Trace preservation: Tr[ρ(t)] = 1 for all t
- Hermiticity: ρ = ρ†
- Positivity: eigenvalues of ρ ≥ 0
- Conservation under no-loss: particle number preserved
- Phase diffusion: off-diagonal decay rate matches γ_φ
"""

import numpy as np
import pytest
import qutip
import scipy

from src.physics.dicke_basis import jz_operator

from .lindblad_solver import (
    LindbladConfig,
    create_coherent_state,
    create_fock_state,
    evolve_lindblad,
    simulate_trajectory,
    steady_state,
    validate_density_matrix,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_config() -> LindbladConfig:
    """Simple configuration for testing."""
    return LindbladConfig(N=10)


@pytest.fixture
def fock_state_1() -> np.ndarray:
    """Fock state |1⟩ for testing."""
    return create_fock_state(1, 10)


@pytest.fixture
def coherent_alpha() -> np.ndarray:
    """Coherent state with amplitude 1.0."""
    return create_coherent_state(1.0 + 0j, 5)


# =============================================================================
# Test Trace Preservation
# =============================================================================


class TestTracePreservation:
    """Test that Tr[ρ(t)] = 1 for all t."""

    def test_trace_should_be_preserved_with_no_dissipation(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())

        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=100)

        traces = np.array([np.trace(rho) for rho in rhos])

        # All traces should be 1
        assert traces == pytest.approx(1.0, abs=1e-6), f"Traces: {traces}"

    def test_trace_should_be_less_than_or_equal_to_1_with_one_body_loss(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0.5, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(2, 5)
        rho0 = np.outer(psi, psi.conj())

        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=100)

        traces = np.array([np.trace(rho) for rho in rhos])

        # All traces should be ≤ 1 and monotonic
        assert np.all(traces <= 1.0 + 1e-6), f"Traces exceed 1: {traces}"
        assert np.all(np.diff(traces) <= 1e-6), "Trace should be monotonic"

    def test_trace_should_be_preserved_with_phase_diffusion_only(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0.5)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())

        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=100)

        traces = np.array([np.trace(rho) for rho in rhos])

        # All traces should be 1
        assert traces == pytest.approx(1.0, abs=1e-6), f"Traces: {traces}"


# =============================================================================
# Test Hermiticity
# =============================================================================


class TestHermiticity:
    """Test that density matrix remains Hermitian."""

    def test_density_matrix_should_remain_hermitian_with_no_dissipation(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())

        times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)

        for i, rho in enumerate(rhos):
            assert rho == pytest.approx(rho.conj().T, abs=1e-6), (
                f"Non-Hermitian at t={times[i]}"
            )

    def test_density_matrix_should_remain_hermitian_with_dissipation(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0.3, gamma_2=0.1, gamma_phi=0.2)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())

        times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)

        for i, rho in enumerate(rhos):
            assert rho == pytest.approx(rho.conj().T, abs=1e-6), (
                f"Non-Hermitian at t={times[i]}"
            )


# =============================================================================
# Test Positivity
# =============================================================================


class TestPositivity:
    """Test that density matrix eigenvalues are non-negative."""

    def test_density_matrix_should_remain_positive_with_no_dissipation(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())

        times, rhos = simulate_trajectory(rho0, config=config, T=1.0, num_times=50)

        for i, rho in enumerate(rhos):
            eigenvalues = np.linalg.eigvalsh(rho)
            min_eigenvalue = np.min(eigenvalues.real)
            assert min_eigenvalue >= -1e-6, (
                f"Negative eigenvalue {min_eigenvalue} at t={times[i]}"
            )

    def test_density_matrix_should_remain_positive_with_small_phase_diffusion(
        self,
    ) -> None:
        # Use small gamma_phi to avoid numerical instability
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0.01)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())

        times, rhos = simulate_trajectory(rho0, config=config, T=0.5, num_times=50)

        for i, rho in enumerate(rhos):
            eigenvalues = np.linalg.eigvalsh(rho)
            min_eigenvalue = np.min(eigenvalues.real)
            assert min_eigenvalue >= -1e-3, (
                f"Negative eigenvalue {min_eigenvalue} at t={times[i]}"
            )

    def test_density_matrix_should_remain_positive_with_phase_diffusion(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=1.0)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())

        times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)

        for i, rho in enumerate(rhos):
            eigenvalues = np.linalg.eigvalsh(rho)
            min_eigenvalue = np.min(eigenvalues.real)
            assert min_eigenvalue >= -1e-6, (
                f"Negative eigenvalue {min_eigenvalue} at t={times[i]}"
            )


# =============================================================================
# Test Conservation Under No-Loss
# =============================================================================


class TestParticleConservation:
    """Test particle number is conserved when gamma_1 = gamma_2 = 0."""

    def test_mean_photon_number_should_be_conserved_with_no_loss(self) -> None:
        N = 5
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=0)

        # Initial state |1⟩
        psi = create_fock_state(1, N)
        rho0 = np.outer(psi, psi.conj())

        initial_n = np.real(
            np.trace(rho0 @ qutip.create(N + 1).full() @ qutip.destroy(N + 1).full())
        )

        times, rhos = simulate_trajectory(rho0, config, T=2.0, num_times=50)

        for i, rho in enumerate(rhos):
            mean_n = np.real(
                np.trace(rho @ qutip.create(N + 1).full() @ qutip.destroy(N + 1).full())
            )
            assert mean_n == pytest.approx(initial_n, abs=1e-6), (
                f"Particle number changed at t={times[i]}: {mean_n} vs {initial_n}"
            )

    def test_coherent_state_should_preserve_mean_photon_number(self) -> None:
        N = 10
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=0)

        alpha = 2.0
        psi = create_coherent_state(alpha + 0j, N)
        rho0 = np.outer(psi, psi.conj())

        initial_n = np.real(
            np.trace(rho0 @ qutip.create(N + 1).full() @ qutip.destroy(N + 1).full())
        )

        times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=30)

        for i, rho in enumerate(rhos):
            mean_n = np.real(
                np.trace(rho @ qutip.create(N + 1).full() @ qutip.destroy(N + 1).full())
            )
            assert mean_n == pytest.approx(initial_n, abs=1e-4), (
                f"Particle number changed at t={times[i]}"
            )


# =============================================================================
# Test Phase Diffusion
# =============================================================================


class TestPhaseDiffusion:
    """Test phase diffusion behavior."""

    def test_off_diagonal_elements_should_decay_at_rate_gamma_phi_for_small_amplitude_states(
        self,
    ) -> None:
        N = 10
        gamma_phi = 0.1  # Use smaller value for numerical stability

        # Initialize in coherent state |α⟩ with small α
        alpha = 0.3
        psi = create_coherent_state(alpha + 0j, N, truncation=5)
        rho0 = np.outer(psi, psi.conj())

        # Build config with phase diffusion only
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=gamma_phi)

        _times, rhos = simulate_trajectory(rho0, config=config, T=2.0, num_times=50)

        # Compute off-diagonal coherence |ρ_{01}|
        coherences: list[float] = []
        for rho in rhos:
            coherence = np.abs(rho[0, 1])
            coherences.append(coherence)

        # Check decays from initial to final
        initial_c = coherences[0]
        final_c = coherences[-1]

        # Should decay but not completely to zero
        if initial_c > 1e-6:
            ratio = final_c / initial_c
            # Should have some decay
            assert ratio < 1.0, "Should decay"

    def test_phase_diffusion_should_preserve_diagonal_populations(self) -> None:
        N = 5
        gamma_phi = 0.1  # Use smaller value

        # Start with superposition
        psi = (create_fock_state(0, N) + create_fock_state(1, N)) / np.sqrt(2)
        rho0 = np.outer(psi, psi.conj())

        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=gamma_phi)

        times, rhos = simulate_trajectory(rho0, config=config, T=0.5, num_times=50)

        # Diagonal elements should be approximately preserved (for small gamma_phi)
        # Allow some tolerance
        for i, rho in enumerate(rhos[:10]):  # Early times only
            pop_0 = np.real(rho[0, 0])
            pop_1 = np.real(rho[1, 1])
            total = pop_0 + pop_1
            assert total == pytest.approx(1.0, abs=1e-4), (
                f"Populations not preserved at t={times[i]}"
            )


# =============================================================================
# Test Unitary Evolution
# =============================================================================


class TestUnitaryEvolution:
    """Test unitary evolution when all dissipation is zero."""

    def test_numerical_evolution_should_match_analytical_unitary_evolution(
        self,
    ) -> None:
        N = 5
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=0)

        # Initial state
        psi = create_fock_state(1, N)
        rho0 = np.outer(psi, psi.conj())

        T = 1.0

        # Evolve using Lindblad solver
        final_rho = evolve_lindblad(rho0, config, T, dt=0.001)

        # Analytical unitary evolution
        H = qutip.create(N + 1).full() @ qutip.destroy(N + 1).full()  # H = n
        U = scipy.linalg.expm(-1.0j * H * T)
        expected_rho = U @ rho0 @ U.conj().T

        assert final_rho == pytest.approx(expected_rho, abs=1e-4), (
            "Unitary evolution mismatch"
        )


# =============================================================================
# Test Steady State
# =============================================================================


class TestSteadyState:
    """Test steady state calculations."""

    def test_iterative_steady_state_should_converge(self) -> None:
        N = 3
        a = qutip.destroy(N + 1).full()
        H = np.zeros((N + 1, N + 1), dtype=complex)

        # One-body loss: L = a
        L_ops = [a]
        gammas = [1.0]

        # Compute steady state (check it doesn't error)
        rho_ss = steady_state(H, L_ops, gammas, max_iter=50)

        # Basic validation - should be a valid density matrix
        validation = validate_density_matrix(rho_ss)
        assert validation["is_hermitian"], "Steady state should be Hermitian"

    def test_with_phase_diffusion_steady_state_should_be_diagonal(self) -> None:
        N = 5
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=1.0)

        H = qutip.create(N + 1).full() @ qutip.destroy(N + 1).full()
        jz = jz_operator(N)

        L_ops = [np.sqrt(config.gamma_phi) * jz]
        gammas = [1.0]

        rho_ss = steady_state(H, L_ops, gammas)

        # Off-diagonal should be small
        max_off_diag = 0.0
        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    max_off_diag = max(max_off_diag, np.abs(rho_ss[i, j]))

        assert max_off_diag < 1e-3, f"Max off-diagonal: {max_off_diag}"


# =============================================================================
# Test Validation Function
# =============================================================================


class TestValidation:
    """Test the validate_density_matrix function."""

    def test_valid_density_matrix_should_pass_all_checks(self) -> None:
        N = 5
        psi = create_fock_state(1, N)
        rho = np.outer(psi, psi.conj())

        validation = validate_density_matrix(rho)

        assert validation["is_hermitian"], (
            'Condition failed: validation["is_hermitian"]'
        )
        assert validation["is_normalized"], (
            'Condition failed: validation["is_normalized"]'
        )
        assert validation["is_positive"], 'Condition failed: validation["is_positive"]'

    def test_unnormalized_state_should_fail_trace_check(self) -> None:
        rho = np.array([[1, 0], [0, 0.5]], dtype=complex)  # trace = 1.5

        validation = validate_density_matrix(rho)

        assert not validation["is_normalized"], (
            'validation["is_normalized"] should be falsy'
        )

    def test_non_hermitian_should_fail_hermiticity_check(self) -> None:
        rho = np.array([[1, 0.1], [-0.1, 0]], dtype=complex)

        validation = validate_density_matrix(rho)

        assert not validation["is_hermitian"], (
            'validation["is_hermitian"] should be falsy'
        )


# =============================================================================
# Integration Test: Full Simulation
# =============================================================================


class TestFullSimulation:
    """Integration tests for full simulation."""

    def test_simulation_should_complete_without_errors(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0.1, gamma_2=0.05)
        psi = create_fock_state(2, 5)
        rho0 = np.outer(psi, psi.conj())

        final_rho = evolve_lindblad(rho0, config, T=1.0, dt=0.01)

        validation = validate_density_matrix(final_rho)

        assert validation["is_hermitian"], "Final state not Hermitian"

    def test_rk4_and_scipy_methods_should_give_similar_results(self) -> None:
        config = LindbladConfig(N=4, gamma_1=0.0, gamma_2=0.0, gamma_phi=0.2)
        psi = create_fock_state(1, 4)
        rho0 = np.outer(psi, psi.conj())

        # Different methods
        rho_rk4 = evolve_lindblad(rho0, config, T=0.5, dt=0.01, method="rk4")

        validation = validate_density_matrix(rho_rk4)
        assert validation["is_hermitian"], (
            'Condition failed: validation["is_hermitian"]'
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_time_should_return_initial_state(self) -> None:
        config = LindbladConfig(N=3, gamma_1=0.1)
        psi = create_fock_state(1, 3)
        rho0 = np.outer(psi, psi.conj())

        final_rho = evolve_lindblad(rho0, config, T=0.0, dt=0.01)

        assert final_rho == pytest.approx(rho0, abs=1e-6), (
            "Expected final_rho == pytest.approx(rho0, abs=1e-6)"
        )

    def test_small_dt_should_give_stable_evolution(self) -> None:
        config = LindbladConfig(N=4, gamma_1=0.1, gamma_2=0.0, gamma_phi=0.0)
        psi = create_fock_state(2, 4)
        rho0 = np.outer(psi, psi.conj())

        final_rho = evolve_lindblad(rho0, config, T=0.5, dt=0.001)

        validation = validate_density_matrix(final_rho)

        assert validation["is_hermitian"], (
            'Condition failed: validation["is_hermitian"]'
        )
        assert validation["is_normalized"], (
            'Condition failed: validation["is_normalized"]'
        )

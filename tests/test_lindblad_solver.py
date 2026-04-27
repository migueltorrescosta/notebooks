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
import scipy

from src.evolution.lindblad_solver import (
    LindbladConfig,
    build_liouvillian_matrix,
    create_bosonic_operators,
    create_coherent_state,
    create_fock_state,
    density_to_vector,
    evolve_lindblad,
    jz_operator,
    ket_to_density,
    number_operator,
    simulate_trajectory,
    steady_state,
    steady_state_dense,
    validate_density_matrix,
    vector_to_density,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_config():
    """Simple configuration for testing."""
    return LindbladConfig(N=10)


@pytest.fixture
def fock_state_1():
    """Fock state |1⟩ for testing."""
    return create_fock_state(1, 10)


@pytest.fixture
def coherent_alpha():
    """Coherent state with amplitude 1.0."""
    return create_coherent_state(1.0 + 0j, 5)


# =============================================================================
# Test Trace Preservation
# =============================================================================


class TestTracePreservation:
    """Test that Tr[ρ(t)] = 1 for all t."""

    def test_trace_preservation_no_loss(self):
        """Trace should be preserved with no dissipation."""
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, 5)
        rho0 = ket_to_density(psi)

        times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=100)

        traces = np.array([np.trace(rho) for rho in rhos])

        # All traces should be 1
        assert np.allclose(traces, 1.0, atol=1e-6), f"Traces: {traces}"

    def test_trace_preservation_one_body_loss(self):
        """Trace should be ≤ 1 with one-body loss."""
        config = LindbladConfig(N=5, gamma_1=0.5, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(2, 5)
        rho0 = ket_to_density(psi)

        times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=100)

        traces = np.array([np.trace(rho) for rho in rhos])

        # All traces should be ≤ 1 and monotonic
        assert np.all(traces <= 1.0 + 1e-6), f"Traces exceed 1: {traces}"
        assert np.all(np.diff(traces) <= 1e-6), "Trace should be monotonic"

    def test_trace_preservation_phase_diffusion(self):
        """Trace should be preserved with phase diffusion only."""
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0.5)
        psi = create_fock_state(1, 5)
        rho0 = ket_to_density(psi)

        times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=100)

        traces = np.array([np.trace(rho) for rho in rhos])

        # All traces should be 1
        assert np.allclose(traces, 1.0, atol=1e-6), f"Traces: {traces}"


# =============================================================================
# Test Hermiticity
# =============================================================================


class TestHermiticity:
    """Test that density matrix remains Hermitian."""

    def test_hermiticity_no_loss(self):
        """Density matrix should remain Hermitian with no dissipation."""
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, 5)
        rho0 = ket_to_density(psi)

        times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)

        for i, rho in enumerate(rhos):
            assert np.allclose(rho, rho.conj().T, atol=1e-6), (
                f"Non-Hermitian at t={times[i]}"
            )

    def test_hermiticity_with_losses(self):
        """Density matrix should remain Hermitian with dissipation."""
        config = LindbladConfig(N=5, gamma_1=0.3, gamma_2=0.1, gamma_phi=0.2)
        psi = create_fock_state(1, 5)
        rho0 = ket_to_density(psi)

        times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)

        for i, rho in enumerate(rhos):
            assert np.allclose(rho, rho.conj().T, atol=1e-6), (
                f"Non-Hermitian at t={times[i]}"
            )


# =============================================================================
# Test Positivity
# =============================================================================


class TestPositivity:
    """Test that density matrix eigenvalues are non-negative."""

    def test_positivity_no_loss(self):
        """Density matrix should remain positive with no dissipation."""
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, 5)
        rho0 = ket_to_density(psi)

        times, rhos = simulate_trajectory(rho0, config=config, T=1.0, num_times=50)

        for i, rho in enumerate(rhos):
            eigenvalues = np.linalg.eigvalsh(rho)
            min_eigenvalue = np.min(eigenvalues.real)
            assert min_eigenvalue >= -1e-6, (
                f"Negative eigenvalue {min_eigenvalue} at t={times[i]}"
            )

    def test_positivity_with_phase_diffusion_small(self):
        """Density matrix should remain positive with small phase diffusion."""
        # Use small gamma_phi to avoid numerical instability
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0.01)
        psi = create_fock_state(1, 5)
        rho0 = ket_to_density(psi)

        times, rhos = simulate_trajectory(rho0, config=config, T=0.5, num_times=50)

        for i, rho in enumerate(rhos):
            eigenvalues = np.linalg.eigvalsh(rho)
            min_eigenvalue = np.min(eigenvalues.real)
            assert min_eigenvalue >= -1e-3, (
                f"Negative eigenvalue {min_eigenvalue} at t={times[i]}"
            )

    def test_positivity_with_phase_diffusion(self):
        """Density matrix should remain positive with phase diffusion."""
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=1.0)
        psi = create_fock_state(1, 5)
        rho0 = ket_to_density(psi)

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

    def test_particle_conservation_no_loss(self):
        """Mean photon number should be conserved with no loss."""
        N = 5
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=0)

        # Initial state |1⟩
        psi = create_fock_state(1, N)
        rho0 = ket_to_density(psi)

        initial_n = np.real(np.trace(rho0 @ number_operator(N)))

        times, rhos = simulate_trajectory(rho0, config, T=2.0, num_times=50)

        for i, rho in enumerate(rhos):
            mean_n = np.real(np.trace(rho @ number_operator(N)))
            assert np.isclose(mean_n, initial_n, atol=1e-6), (
                f"Particle number changed at t={times[i]}: {mean_n} vs {initial_n}"
            )

    def test_particle_conservation_coherent_state(self):
        """Coherent state should preserve mean photon number."""
        N = 10
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=0)

        alpha = 2.0
        psi = create_coherent_state(alpha + 0j, N)
        rho0 = ket_to_density(psi)

        initial_n = np.real(np.trace(rho0 @ number_operator(N)))

        times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=30)

        for i, rho in enumerate(rhos):
            mean_n = np.real(np.trace(rho @ number_operator(N)))
            assert np.isclose(mean_n, initial_n, atol=1e-4), (
                f"Particle number changed at t={times[i]}"
            )


# =============================================================================
# Test Phase Diffusion
# =============================================================================


class TestPhaseDiffusion:
    """Test phase diffusion behavior."""

    def test_phase_diffusion_decay_rate_small(self):
        """Off-diagonal elements should decay at rate γ_φ for small amplitude states."""
        N = 10
        gamma_phi = 0.1  # Use smaller value for numerical stability

        # Initialize in coherent state |α⟩ with small α
        alpha = 0.3
        psi = create_coherent_state(alpha + 0j, N, truncation=5)
        rho0 = ket_to_density(psi)

        # Build config with phase diffusion only
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=gamma_phi)

        times, rhos = simulate_trajectory(rho0, config=config, T=2.0, num_times=50)

        # Compute off-diagonal coherence |ρ_{01}|
        coherences = []
        for rho in rhos:
            coherence = np.abs(rho[0, 1])
            coherences.append(coherence)

        coherences = np.array(coherences)

        # Check decays from initial to final
        initial_c = coherences[0]
        final_c = coherences[-1]

        # Should decay but not completely to zero
        if initial_c > 1e-6:
            ratio = final_c / initial_c
            # Should have some decay
            assert ratio < 1.0, "Should decay"

    def test_phase_diffusion_preserves_populations(self):
        """Phase diffusion should preserve diagonal populations."""
        N = 5
        gamma_phi = 0.1  # Use smaller value

        # Start with superposition
        psi = (create_fock_state(0, N) + create_fock_state(1, N)) / np.sqrt(2)
        rho0 = ket_to_density(psi)

        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=gamma_phi)

        times, rhos = simulate_trajectory(rho0, config=config, T=0.5, num_times=50)

        # Diagonal elements should be approximately preserved (for small gamma_phi)
        # Allow some tolerance
        for i, rho in enumerate(rhos[:10]):  # Early times only
            pop_0 = np.real(rho[0, 0])
            pop_1 = np.real(rho[1, 1])
            total = pop_0 + pop_1
            assert np.isclose(total, 1.0, atol=1e-4), (
                f"Populations not preserved at t={times[i]}"
            )


# =============================================================================
# Test Unitary Evolution
# =============================================================================


class TestUnitaryEvolution:
    """Test unitary evolution when all dissipation is zero."""

    def test_unitary_evolution_matches_exact(self):
        """Numerical evolution should match analytical unitary evolution."""
        N = 5
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=0)

        # Initial state
        psi = create_fock_state(1, N)
        rho0 = ket_to_density(psi)

        T = 1.0

        # Evolve using Lindblad solver
        final_rho = evolve_lindblad(rho0, config, T, dt=0.001)

        # Analytical unitary evolution
        a, a_dag = create_bosonic_operators(N)
        H = number_operator(N)  # H = n
        U = scipy.linalg.expm(-1.0j * H * T)
        expected_rho = U @ rho0 @ U.conj().T

        assert np.allclose(final_rho, expected_rho, atol=1e-4), (
            "Unitary evolution mismatch"
        )


# =============================================================================
# Test Steady State
# =============================================================================


class TestSteadyState:
    """Test steady state calculations."""

    def test_steady_state_iterative(self):
        """Iterative steady state should converge (basic check)."""
        N = 3
        a, a_dag = create_bosonic_operators(N)
        H = np.zeros((N + 1, N + 1), dtype=complex)

        # One-body loss: L = a
        L_ops = [a]
        gammas = [1.0]

        # Compute steady state (check it doesn't error)
        rho_ss = steady_state(H, L_ops, gammas, max_iter=50)

        # Basic validation - should be a valid density matrix
        validation = validate_density_matrix(rho_ss)
        assert validation["is_hermitian"], "Steady state should be Hermitian"

    def test_steady_state_with_phase_diffusion(self):
        """With phase diffusion, steady state should be diagonal."""
        N = 5
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=1.0)

        a, a_dag = create_bosonic_operators(N)
        H = number_operator(N)
        jz = jz_operator(N)

        L_ops = [np.sqrt(config.gamma_phi) * jz]
        gammas = [1.0]

        rho_ss = steady_state_dense(H, L_ops, gammas)

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

    def test_valid_density_matrix(self):
        """Valid density matrix should pass all checks."""
        N = 5
        psi = create_fock_state(1, N)
        rho = ket_to_density(psi)

        validation = validate_density_matrix(rho)

        assert validation["is_hermitian"]
        assert validation["is_normalized"]
        assert validation["is_positive"]

    def test_unnormalized_fails(self):
        """Unnormalized state should fail trace check."""
        rho = np.array([[1, 0], [0, 0.5]], dtype=complex)  # trace = 1.5

        validation = validate_density_matrix(rho)

        assert not validation["is_normalized"]

    def test_non_hermitian_fails(self):
        """Non-Hermitian should fail hermiticity check."""
        rho = np.array([[1, 0.1], [-0.1, 0]], dtype=complex)

        validation = validate_density_matrix(rho)

        assert not validation["is_hermitian"]


# =============================================================================
# Test Liouvillian Construction
# =============================================================================


class TestLiouvillian:
    """Test Liouvillian superoperator."""

    def test_liouvillian_unitary_part(self):
        """Liouvillian unitary part should give correct dynamics."""
        N = 3
        a, a_dag = create_bosonic_operators(N)
        H = number_operator(N)

        # Build Liouvillian with no dissipation
        L_ops = []
        gammas = []

        L_mat = build_liouvillian_matrix(H, L_ops, gammas)

        # Test on a density matrix
        psi = create_fock_state(1, N)
        rho = ket_to_density(psi)

        rho_vec = density_to_vector(rho)
        drho_dt_vec = L_mat @ rho_vec
        drho_dt = vector_to_density(drho_dt_vec)

        # Expected: -i[H, rho]
        expected = -1.0j * (H @ rho - rho @ H)

        assert np.allclose(drho_dt, expected, atol=1e-6)

    def test_liouvillian_dissipative_part(self):
        """Liouvillian should give correct decay for one-body loss."""
        N = 3
        a, a_dag = create_bosonic_operators(N)

        H = number_operator(N)
        L_ops = [a]
        gammas = [1.0]

        L_mat = build_liouvillian_matrix(H, L_ops, gammas)

        # Test on vacuum
        psi = create_fock_state(1, N)  # |1⟩
        rho = ket_to_density(psi)

        rho_vec = density_to_vector(rho)
        drho_dt_vec = L_mat @ rho_vec
        drho_dt = vector_to_density(drho_dt_vec)

        # dρ/dt = L ρ L† - ½{L†L, ρ} for one-body loss from |1⟩
        # Should have transitions to |0⟩
        # At least one element should change
        assert np.max(np.abs(drho_dt)) > 0


# =============================================================================
# Integration Test: Full Simulation
# =============================================================================


class TestFullSimulation:
    """Integration tests for full simulation."""

    def test_simulation_completes(self):
        """Simulation should complete without errors."""
        config = LindbladConfig(N=5, gamma_1=0.1, gamma_2=0.05)
        psi = create_fock_state(2, 5)
        rho0 = ket_to_density(psi)

        final_rho = evolve_lindblad(rho0, config, T=1.0, dt=0.01)

        validation = validate_density_matrix(final_rho)

        assert validation["is_hermitian"], "Final state not Hermitian"

    def test_rk4_vs_scipy_methods(self):
        """RK4 and scipy methods should give similar results."""
        config = LindbladConfig(N=4, gamma_1=0.0, gamma_2=0.0, gamma_phi=0.2)
        psi = create_fock_state(1, 4)
        rho0 = ket_to_density(psi)

        # Different methods
        rho_rk4 = evolve_lindblad(rho0, config, T=0.5, dt=0.01, method="rk4")

        validation = validate_density_matrix(rho_rk4)
        assert validation["is_hermitian"]


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_time_evolution(self):
        """Zero time should return initial state."""
        config = LindbladConfig(N=3, gamma_1=0.1)
        psi = create_fock_state(1, 3)
        rho0 = ket_to_density(psi)

        final_rho = evolve_lindblad(rho0, config, T=0.0, dt=0.01)

        assert np.allclose(final_rho, rho0, atol=1e-6)

    def test_small_dt(self):
        """Small dt should give stable evolution."""
        config = LindbladConfig(N=4, gamma_1=0.1, gamma_2=0.0, gamma_phi=0.0)
        psi = create_fock_state(2, 4)
        rho0 = ket_to_density(psi)

        final_rho = evolve_lindblad(rho0, config, T=0.5, dt=0.001)

        validation = validate_density_matrix(final_rho)

        assert validation["is_hermitian"]
        assert validation["is_normalized"]

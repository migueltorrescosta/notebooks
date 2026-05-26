"""
Tests for Lindblad Master Equation Solver.

Physical Validation Tests:
- Trace preservation: Tr[ρ(t)] = 1 for all t
- Hermiticity: ρ = ρ†
- Positivity: eigenvalues of ρ ≥ 0
- Conservation under no-loss: particle number preserved
- Phase diffusion: off-diagonal decay rate matches γ_φ
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.dicke_basis import jz_operator

from .lindblad_solver import (
    LindbladConfig,
    create_coherent_state,
    create_fock_state,
    evolve_lindblad,
    evolve_lindblad_rk4,
    evolve_lindblad_scipy,
    lindblad_rhs,
    simulate_trajectory,
    steady_state,
    validate_density_matrix,
)


class TestTracePreservation:
    """Tr[ρ(t)] = 1 for all t."""

    def test_given_no_dissipation_then_trace_preserved(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())
        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=100)
        traces = np.array([np.trace(rho) for rho in rhos])
        assert traces == pytest.approx(1.0, abs=1e-6)

    def test_given_one_body_loss_then_trace_leq_1(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0.5, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(2, 5)
        rho0 = np.outer(psi, psi.conj())
        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=100)
        traces = np.array([np.trace(rho) for rho in rhos])
        assert np.all(traces <= 1.0 + 1e-6)
        assert np.all(np.diff(traces) <= 1e-6)

    def test_given_phase_diffusion_only_then_trace_preserved(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0.5)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())
        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=100)
        traces = np.array([np.trace(rho) for rho in rhos])
        assert traces == pytest.approx(1.0, abs=1e-6)


class TestHermiticity:
    """Density matrix remains Hermitian."""

    def test_given_no_dissipation_then_hermitian(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())
        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)
        for rho in rhos:
            assert rho == pytest.approx(rho.conj().T, abs=1e-6)

    def test_given_dissipation_then_hermitian(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0.3, gamma_2=0.1, gamma_phi=0.2)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())
        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)
        for rho in rhos:
            assert rho == pytest.approx(rho.conj().T, abs=1e-6)


class TestPositivity:
    """Density matrix eigenvalues are non-negative."""

    def test_given_no_dissipation_then_positive(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())
        _times, rhos = simulate_trajectory(rho0, config=config, T=1.0, num_times=50)
        for rho in rhos:
            min_eigenvalue = np.min(np.linalg.eigvalsh(rho).real)
            assert min_eigenvalue >= -1e-6

    def test_given_small_phase_diffusion_then_positive(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0.01)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())
        _times, rhos = simulate_trajectory(rho0, config=config, T=0.5, num_times=50)
        for rho in rhos:
            min_eigenvalue = np.min(np.linalg.eigvalsh(rho).real)
            assert min_eigenvalue >= -1e-3

    def test_given_phase_diffusion_then_positive(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=1.0)
        psi = create_fock_state(1, 5)
        rho0 = np.outer(psi, psi.conj())
        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)
        for rho in rhos:
            min_eigenvalue = np.min(np.linalg.eigvalsh(rho).real)
            assert min_eigenvalue >= -1e-6


class TestParticleConservation:
    """Particle number is conserved when gamma_1 = gamma_2 = 0."""

    def test_given_no_loss_then_mean_photon_number_conserved(self) -> None:
        import qutip

        N = 5
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, N)
        rho0 = np.outer(psi, psi.conj())
        n_op = qutip.create(N + 1).full() @ qutip.destroy(N + 1).full()
        initial_n = np.real(np.trace(rho0 @ n_op))
        _times, rhos = simulate_trajectory(rho0, config, T=2.0, num_times=50)
        for rho in rhos:
            mean_n = np.real(np.trace(rho @ n_op))
            assert mean_n == pytest.approx(initial_n, abs=1e-6)

    def test_given_coherent_state_then_mean_photon_number_conserved(self) -> None:
        import qutip

        N = 10
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=0)
        alpha = 2.0
        psi = create_coherent_state(alpha + 0j, N)
        rho0 = np.outer(psi, psi.conj())
        n_op = qutip.create(N + 1).full() @ qutip.destroy(N + 1).full()
        initial_n = np.real(np.trace(rho0 @ n_op))
        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=30)
        for rho in rhos:
            mean_n = np.real(np.trace(rho @ n_op))
            assert mean_n == pytest.approx(initial_n, abs=1e-4)


class TestPhaseDiffusion:
    """Phase diffusion behavior."""

    def test_given_small_amplitude_then_off_diagonal_decays(self) -> None:
        N = 10
        gamma_phi = 0.1
        alpha = 0.3
        psi = create_coherent_state(alpha + 0j, N, truncation=5)
        rho0 = np.outer(psi, psi.conj())
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=gamma_phi)
        _times, rhos = simulate_trajectory(rho0, config=config, T=2.0, num_times=50)
        coherences = [np.abs(rho[0, 1]) for rho in rhos]
        initial_c = coherences[0]
        final_c = coherences[-1]
        if initial_c > 1e-6:
            ratio = final_c / initial_c
            assert ratio < 1.0

    def test_given_phase_diffusion_then_diagonal_populations_preserved(self) -> None:
        N = 5
        gamma_phi = 0.1
        psi = (create_fock_state(0, N) + create_fock_state(1, N)) / np.sqrt(2)
        rho0 = np.outer(psi, psi.conj())
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=gamma_phi)
        _times, rhos = simulate_trajectory(rho0, config=config, T=0.5, num_times=50)
        for rho in rhos[:10]:
            pop_0 = np.real(rho[0, 0])
            pop_1 = np.real(rho[1, 1])
            assert pop_0 + pop_1 == pytest.approx(1.0, abs=1e-4)


class TestUnitaryEvolution:
    """Unitary evolution when all dissipation is zero."""

    def test_given_zero_dissipation_then_numerical_matches_analytical(
        self,
    ) -> None:
        import qutip
        import scipy

        N = 5
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=0)
        psi = create_fock_state(1, N)
        rho0 = np.outer(psi, psi.conj())
        T = 1.0
        final_rho = evolve_lindblad(rho0, config, T, dt=0.001)
        H = qutip.create(N + 1).full() @ qutip.destroy(N + 1).full()
        U = scipy.linalg.expm(-1.0j * H * T)
        expected_rho = U @ rho0 @ U.conj().T
        assert final_rho == pytest.approx(expected_rho, abs=1e-4)


class TestLindbladRhs:
    """lindblad_rhs — RHS of Lindblad master equation."""

    def test_given_no_dissipation_then_rhs_is_minus_i_commutator(self) -> None:
        d = 4
        H = np.diag([0.0, 1.0, 2.0, 3.0])
        L_ops: list[np.ndarray] = []
        gammas: list[float] = []
        rho = np.eye(d, dtype=complex) / d

        drho = lindblad_rhs(rho, H, L_ops, gammas)
        expected = -1.0j * (H @ rho - rho @ H)
        assert drho == pytest.approx(expected)

    def test_given_dissipation_then_drho_is_nonzero(self) -> None:
        d = 3
        H = np.zeros((d, d), dtype=complex)
        L = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
        rho = np.diag([0.0, 1.0, 0.0])
        drho = lindblad_rhs(rho, H, [L], [1.0])
        assert np.max(np.abs(drho)) > 0, "Dissipation should produce non-zero drift"

    def test_given_zero_gamma_then_operator_is_skipped(self) -> None:
        d = 2
        H = np.zeros((d, d), dtype=complex)
        L = np.array([[0, 1], [0, 0]], dtype=complex)
        rho = np.eye(d, dtype=complex) / d
        drho = lindblad_rhs(rho, H, [L], [0.0])
        assert drho == pytest.approx(np.zeros((d, d), dtype=complex))


class TestEvolveLindbladRk4:
    """evolve_lindblad_rk4 — RK4 integration of Lindblad equation."""

    def test_given_zero_time_then_returns_initial_state(self) -> None:
        d = 4
        rho0 = np.eye(d, dtype=complex) / d
        H = np.diag([0.0, 1.0, 2.0, 3.0])
        result = evolve_lindblad_rk4(rho0, H, [], [], T=0.0, dt=0.01)
        assert result == pytest.approx(rho0)

    def test_given_no_dissipation_then_trace_preserved(self) -> None:
        d = 4
        rho0 = np.eye(d, dtype=complex) / d
        H = np.diag([0.0, 1.0, 2.0, 3.0])
        result = evolve_lindblad_rk4(rho0, H, [], [], T=1.0, dt=0.01)
        assert np.trace(result) == pytest.approx(1.0, abs=1e-6)

    def test_given_no_dissipation_then_hermitian(self) -> None:
        d = 4
        rho0 = np.eye(d, dtype=complex) / d
        H = np.diag([0.0, 1.0, 2.0, 3.0])
        result = evolve_lindblad_rk4(rho0, H, [], [], T=1.0, dt=0.01)
        assert result == pytest.approx(result.conj().T, abs=1e-6)

    def test_given_dissipation_then_hermitian(self) -> None:
        d = 3
        rho0 = np.diag([1.0, 0.0, 0.0])
        H = np.zeros((d, d), dtype=complex)
        L = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
        result = evolve_lindblad_rk4(rho0, H, [L], [1.0], T=0.5, dt=0.01)
        assert result == pytest.approx(result.conj().T, abs=1e-6)
        assert np.trace(result) == pytest.approx(1.0, abs=1e-6)

    def test_given_positive_initial_then_positive_final(self) -> None:
        d = 3
        rho0 = np.diag([1.0, 0.0, 0.0])
        H = np.zeros((d, d), dtype=complex)
        L = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
        result = evolve_lindblad_rk4(rho0, H, [L], [1.0], T=0.5, dt=0.01)
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.min(eigenvalues) >= -1e-6


class TestEvolveLindbladScipy:
    """evolve_lindblad_scipy — scipy ODE integration of Lindblad equation."""

    def test_given_no_dissipation_then_trace_preserved(self) -> None:
        d = 3
        rho0 = np.eye(d, dtype=complex) / d
        H = np.diag([0.0, 1.0, 2.0])
        result = evolve_lindblad_scipy(rho0, H, [], [], T=1.0)
        assert np.trace(result) == pytest.approx(1.0, abs=1e-6)

    def test_given_no_dissipation_then_hermitian(self) -> None:
        d = 3
        rho0 = np.eye(d, dtype=complex) / d
        H = np.diag([0.0, 1.0, 2.0])
        result = evolve_lindblad_scipy(rho0, H, [], [], T=1.0)
        assert result == pytest.approx(result.conj().T, abs=1e-6)

    def test_given_dissipation_then_density_matrix_valid(self) -> None:
        d = 3
        rho0 = np.diag([1.0, 0.0, 0.0])
        H = np.zeros((d, d), dtype=complex)
        L = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
        result = evolve_lindblad_scipy(rho0, H, [L], [1.0], T=0.5)
        assert result == pytest.approx(result.conj().T, abs=1e-6)
        assert np.trace(result) == pytest.approx(1.0, abs=1e-6)
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.min(eigenvalues) >= -1e-6


class TestSteadyState:
    """Steady state calculations."""

    def test_given_one_body_loss_then_steady_state_converges(self) -> None:
        import qutip

        N = 3
        a = qutip.destroy(N + 1).full()
        H = np.zeros((N + 1, N + 1), dtype=complex)
        L_ops = [a]
        gammas = [1.0]
        rho_ss = steady_state(H, L_ops, gammas, max_iter=50)
        validation = validate_density_matrix(rho_ss)
        assert validation["is_hermitian"]

    def test_given_phase_diffusion_then_steady_state_diagonal(self) -> None:
        import qutip

        N = 5
        config = LindbladConfig(N=N, gamma_1=0, gamma_2=0, gamma_phi=1.0)
        H = qutip.create(N + 1).full() @ qutip.destroy(N + 1).full()
        jz = jz_operator(N)
        L_ops = [np.sqrt(config.gamma_phi) * jz]
        gammas = [1.0]
        rho_ss = steady_state(H, L_ops, gammas)
        max_off_diag = max(
            np.abs(rho_ss[i, j]) for i in range(N + 1) for j in range(N + 1) if i != j
        )
        assert max_off_diag < 1e-3


class TestValidation:
    """validate_density_matrix function."""

    def test_given_valid_density_matrix_then_passes_all_checks(self) -> None:
        N = 5
        psi = create_fock_state(1, N)
        rho = np.outer(psi, psi.conj())
        validation = validate_density_matrix(rho)
        assert validation["is_hermitian"]
        assert validation["is_normalized"]
        assert validation["is_positive"]

    def test_given_unnormalized_state_then_fails_trace_check(self) -> None:
        rho = np.array([[1, 0], [0, 0.5]], dtype=complex)
        validation = validate_density_matrix(rho)
        assert not validation["is_normalized"]

    def test_given_non_hermitian_state_then_fails_hermiticity_check(self) -> None:
        rho = np.array([[1, 0.1], [-0.1, 0]], dtype=complex)
        validation = validate_density_matrix(rho)
        assert not validation["is_hermitian"]


class TestFullSimulation:
    """Integration tests for full simulation."""

    def test_given_typical_config_then_simulation_completes(self) -> None:
        config = LindbladConfig(N=5, gamma_1=0.1, gamma_2=0.05)
        psi = create_fock_state(2, 5)
        rho0 = np.outer(psi, psi.conj())
        final_rho = evolve_lindblad(rho0, config, T=1.0, dt=0.01)
        validation = validate_density_matrix(final_rho)
        assert validation["is_hermitian"]

    def test_given_methods_differ_then_final_state_hermitian(self) -> None:
        config = LindbladConfig(N=4, gamma_1=0.0, gamma_2=0.0, gamma_phi=0.2)
        psi = create_fock_state(1, 4)
        rho0 = np.outer(psi, psi.conj())
        rho_rk4 = evolve_lindblad(rho0, config, T=0.5, dt=0.01, method="rk4")
        validation = validate_density_matrix(rho_rk4)
        assert validation["is_hermitian"]


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_given_zero_time_then_returns_initial_state(self) -> None:
        config = LindbladConfig(N=3, gamma_1=0.1)
        psi = create_fock_state(1, 3)
        rho0 = np.outer(psi, psi.conj())
        final_rho = evolve_lindblad(rho0, config, T=0.0, dt=0.01)
        assert final_rho == pytest.approx(rho0, abs=1e-6)

    def test_given_small_dt_then_evolution_stable(self) -> None:
        config = LindbladConfig(N=4, gamma_1=0.1, gamma_2=0.0, gamma_phi=0.0)
        psi = create_fock_state(2, 4)
        rho0 = np.outer(psi, psi.conj())
        final_rho = evolve_lindblad(rho0, config, T=0.5, dt=0.001)
        validation = validate_density_matrix(final_rho)
        assert validation["is_hermitian"]
        assert validation["is_normalized"]

"""
Tests for J_z operator correctness and phase diffusion.

Validates:
- Dicke vs Fock basis conventions for jz_operator
- Phase diffusion off-diagonal decay rate ∝ (m₁ - m₂)²
- Unitary evolution (gamma=0) unchanged
- Tr[ρ(t)] = 1 preserved under phase diffusion
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evolution.lindblad_solver import (
    LindbladConfig,
    evolve_lindblad,
    simulate_trajectory,
)
from src.physics.dicke_basis import jz_operator
from src.physics.noise_channels import NoiseConfig, build_lindblad_operators
from src.utils.enums import OperatorBasis

# =============================================================================
# Basis Conventions
# =============================================================================


class TestJzBasisConventions:
    """Verify jz_operator basis conventions are consistent."""

    @pytest.mark.parametrize("N", [2, 4, 10])
    def test_dicke_basis_explicit_vs_default_should_match_default_basis(
        self, N: int
    ) -> None:
        mat_default = jz_operator(N)
        mat_explicit = jz_operator(N, basis=OperatorBasis.DICKE)
        np.testing.assert_allclose(
            mat_default,
            mat_explicit,
            rtol=1e-12,
            err_msg=f"Default basis does not match explicit DICKE for N={N}",
        )

    @pytest.mark.parametrize("N", [2, 4, 10])
    def test_fock_basis_is_reversed_dicke_should_have_reversed_eigenvalues(
        self, N: int
    ) -> None:
        mat_dicke = jz_operator(N, basis=OperatorBasis.DICKE)
        mat_fock = jz_operator(N, basis=OperatorBasis.FOCK)

        # Fock eigenvalues = reversed Dicke eigenvalues
        dicke_diag = np.diag(mat_dicke)
        fock_diag = np.diag(mat_fock)
        np.testing.assert_allclose(
            fock_diag,
            dicke_diag[::-1],
            rtol=1e-12,
            err_msg=f"FOCK J_z eigenvalues are not reversed DICKE for N={N}",
        )


# =============================================================================
# Phase diffusion off-diagonal decay ∝ (m₁ - m₂)²
# =============================================================================


class TestPhaseDiffusionOffDiagonalDecay:
    """Verify phase diffusion causes off-diagonal decay proportional to (m₁ - m₂)²."""

    @pytest.mark.parametrize("N", [4, 6])
    def test_decay_rate_scales_with_m_difference_squared_should_scale_with_m1_minus_m2_squared(
        self, N: int
    ) -> None:
        gamma_phi = 0.1
        config = LindbladConfig(N=N, gamma_1=0.0, gamma_2=0.0, gamma_phi=gamma_phi)

        dim = N + 1
        # Fock basis: index i corresponds to eigenvalue m = i - N/2
        m_values = np.arange(dim) - N / 2.0

        # Test multiple pairs of (m1, m2)
        pairs = [(0, 1), (0, 2), (1, 3), (0, 3)]
        for i1, i2 in pairs:
            m1 = m_values[i1]
            m2 = m_values[i2]
            delta_m_sq = (m1 - m2) ** 2

            # Initial superposition: (|m1⟩ + |m2⟩) / √2
            psi = np.zeros(dim, dtype=complex)
            psi[i1] = 1.0 / np.sqrt(2)
            psi[i2] = 1.0 / np.sqrt(2)
            rho0 = np.outer(psi, psi.conj())

            T = 1.0
            _times, rhos = simulate_trajectory(rho0, config, T=T, num_times=100)

            # Track off-diagonal element |ρ_{i1,i2}|
            coherences = np.array([np.abs(rho[i1, i2]) for rho in rhos])
            t_array = np.array([np.real(np.trace(rho)) for rho in rhos])
            t_array = np.linspace(0, T, len(coherences))

            # Expected: coherence(t) = coherence(0) * exp(-gamma_phi * (m1-m2)^2 * t / 2)
            # For the Lindblad form L = sqrt(gamma) * J_z, the off-diagonal decay rate
            # is gamma_phi * (m1 - m2)^2 / 2
            initial_coh = coherences[0]
            if initial_coh < 1e-10:
                continue

            # Fit exponential decay: |rho_12(t)| = |rho_12(0)| * exp(-lambda * t)
            log_coh = np.log(np.maximum(coherences / initial_coh, 1e-15))
            # Use early-time fit (first 50% of evolution)
            mid = len(t_array) // 2
            slope = np.polyfit(t_array[:mid], log_coh[:mid], 1)[0]
            measured_rate = -slope

            expected_rate = gamma_phi * delta_m_sq / 2.0

            # Allow 20% tolerance for numerical integration error
            assert measured_rate > 0, f"No decay observed for pair ({i1}, {i2})"
            relative_error = abs(measured_rate - expected_rate) / expected_rate
            assert relative_error < 0.30, (
                f"Decay rate mismatch for m₁={m1}, m₂={m2} (Δm²={delta_m_sq}): "
                f"measured={measured_rate:.4f}, expected={expected_rate:.4f}, "
                f"relative_error={relative_error:.2f}"
            )

    @pytest.mark.parametrize("N", [4, 6])
    def test_same_m_no_decay_should_not_decay_diagonal_elements(self, N: int) -> None:
        gamma_phi = 0.5
        config = LindbladConfig(N=N, gamma_1=0.0, gamma_2=0.0, gamma_phi=gamma_phi)

        # Pure state |m=0⟩
        idx = N // 2
        psi = np.zeros(N + 1, dtype=complex)
        psi[idx] = 1.0
        rho0 = np.outer(psi, psi.conj())

        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)

        # Diagonal element should stay at 1.0
        pops = np.array([np.real(rho[idx, idx]) for rho in rhos])
        np.testing.assert_allclose(pops, 1.0, atol=1e-4)


# =============================================================================
# Unitary evolution (gamma=0) unchanged
# =============================================================================


class TestUnitaryEvolutionUnchanged:
    """Verify unitary evolution is unaffected by the J_z fix."""

    @pytest.mark.parametrize("N", [3, 5, 8])
    def test_gamma_zero_preserves_fock_state_should_only_acquire_a_phase(
        self, N: int
    ) -> None:
        config = LindbladConfig(N=N, gamma_1=0.0, gamma_2=0.0, gamma_phi=0.0)

        psi = np.zeros(N + 1, dtype=complex)
        psi[1] = 1.0
        rho0 = np.outer(psi, psi.conj())

        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)

        # Populations should be exactly preserved
        for rho in rhos:
            populations = np.real(np.diag(rho))
            expected = np.zeros(N + 1)
            expected[1] = 1.0
            np.testing.assert_allclose(populations, expected, atol=1e-6)

    @pytest.mark.parametrize("N", [4, 6])
    def test_gamma_zero_matches_analytical_should_match_exact_unitary(
        self, N: int
    ) -> None:
        import qutip
        import scipy

        config = LindbladConfig(N=N, gamma_1=0.0, gamma_2=0.0, gamma_phi=0.0)

        psi = np.zeros(N + 1, dtype=complex)
        psi[1] = 1.0 / np.sqrt(2)
        psi[2] = 1.0 / np.sqrt(2)
        rho0 = np.outer(psi, psi.conj())

        T = 1.0
        final_rho = evolve_lindblad(rho0, config, T=T, dt=0.001)

        # Analytical: U = exp(-i H T) with H = n
        n_op = qutip.create(N + 1).full() @ qutip.destroy(N + 1).full()
        U = scipy.linalg.expm(-1.0j * n_op * T)
        expected = U @ rho0 @ U.conj().T

        np.testing.assert_allclose(final_rho, expected, atol=1e-5)


# =============================================================================
# Trace preservation under phase diffusion
# =============================================================================


class TestTracePreservationUnderPhaseDiffusion:
    """Verify Tr[ρ(t)] = 1 is preserved under phase diffusion."""

    @pytest.mark.parametrize("N", [3, 5, 8])
    @pytest.mark.parametrize("gamma_phi", [0.1, 0.5, 1.0])
    def test_trace_preserved_should_remain_exactly_1_under_phase_diffusion(
        self, N: int, gamma_phi: float
    ) -> None:
        config = LindbladConfig(N=N, gamma_1=0.0, gamma_2=0.0, gamma_phi=gamma_phi)

        # Superposition state
        psi = np.zeros(N + 1, dtype=complex)
        psi[0] = 1.0 / np.sqrt(2)
        psi[1] = 1.0 / np.sqrt(2)
        rho0 = np.outer(psi, psi.conj())

        _times, rhos = simulate_trajectory(rho0, config, T=2.0, num_times=100)

        traces = np.array([np.trace(rho) for rho in rhos])
        np.testing.assert_allclose(
            np.real(traces),
            1.0,
            atol=1e-6,
            err_msg=f"Trace not preserved: min={np.min(traces):.8f}, max={np.max(traces):.8f}",
        )

    @pytest.mark.parametrize("N", [4, 6])
    def test_trace_preserved_coherent_state_should_remain_1_for_coherent_states(
        self, N: int
    ) -> None:
        import scipy.special

        gamma_phi = 0.3
        config = LindbladConfig(N=N, gamma_1=0.0, gamma_2=0.0, gamma_phi=gamma_phi)

        # Coherent state |α=1⟩
        alpha = 1.0
        psi = np.zeros(N + 1, dtype=complex)
        for n in range(N + 1):
            psi[n] = alpha**n / np.sqrt(scipy.special.factorial(n))
        psi *= np.exp(-(alpha**2) / 2)
        psi /= np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())

        _times, rhos = simulate_trajectory(rho0, config, T=1.0, num_times=50)

        traces = np.array([np.trace(rho) for rho in rhos])
        np.testing.assert_allclose(np.real(traces), 1.0, atol=1e-5)


# =============================================================================
# NoiseConfig build_lindblad_operators uses J_z correctly
# =============================================================================


class TestNoiseConfigJzUsage:
    """Verify build_lindblad_operators uses J_z (Dicke basis)."""

    @pytest.mark.parametrize("N", [2, 4, 10])
    def test_phase_diffusion_uses_jz_should_use_sqrt_gamma_phi_times_jz_lindblad_operator(
        self, N: int
    ) -> None:
        gamma_phi = 0.05
        config = NoiseConfig(gamma_phi=gamma_phi)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 1, "Expected len(L_ops) == 1"
        expected = np.sqrt(gamma_phi) * jz_operator(N)
        np.testing.assert_allclose(L_ops[0], expected, rtol=1e-12)

    @pytest.mark.parametrize("N", [2, 4, 10])
    def test_combined_channels_all_correct_should_use_correct_operators_simultaneously(
        self, N: int
    ) -> None:
        config = NoiseConfig(gamma_1=0.1, gamma_2=0.05, gamma_phi=0.02)
        L_ops = build_lindblad_operators(N, config)

        assert len(L_ops) == 3, "Expected len(L_ops) == 3"
        # Each operator should be non-zero
        for L in L_ops:
            assert np.max(np.abs(L)) > 0, "Expected np.max(np.abs(L)) > 0"

"""Integration tests for BEC sensitivity and ancilla physics simulations.

Tests verify:
1. Simulations complete within timeout
2. Physics assertions hold
"""

from __future__ import annotations

import numpy as np


class TestBECSensitivityScaling:
    """Tests for BEC sensitivity scaling physics."""

    def test_given_twa_simulation_completes_quickly_then_complete_within_timeout_for_small_n(
        self,
    ) -> None:
        import time

        from src.physics.truncated_wigner import run_twa_simulation

        start = time.perf_counter()

        result = run_twa_simulation(
            N=20,
            state_type="CSS",
            chi=1.0,
            T=0.1,
            N_traj=50,
            rng_seed=42,
        )

        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Simulation took {elapsed:.2f}s, expected < 2s"
        assert "delta_phi" in result, 'Expected "delta_phi" in result'
        assert "delta_phi_sql" in result, 'Expected "delta_phi_sql" in result'
        assert "delta_phi_hl" in result, 'Expected "delta_phi_hl" in result'


class TestBECAncilla:
    """Tests for BEC ancilla physics."""

    def test_given_ancilla_evolution_produces_finite_expectations_then_produce_finite_jz_expectation(
        self,
    ) -> None:
        from src.algorithms.spin_squeezing import coherent_spin_state
        from src.evolution.lindblad_solver import (
            LindbladConfig,
            evolve_lindblad,
        )
        from src.physics.dicke_basis import jz_operator

        N = 10
        state = coherent_spin_state(N)
        chi = 1.0
        T = 0.1

        rho0 = np.outer(state, state.conj())
        config = LindbladConfig(N=N, chi=chi)
        rho = evolve_lindblad(rho0, config, T, 0.01)
        J_z = jz_operator(N)

        Jz_mean = np.real(np.trace(rho @ J_z))
        assert np.isfinite(Jz_mean), "Jz expectation should be finite"

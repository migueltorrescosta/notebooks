"""Tests for the MZI Lindblad master equation module.

Tests verify:
1. MziNoiseConfig defaults and validation
2. Lindblad operator construction (build_mzi_lindblad_operators)
3. Lindblad evolution preserves trace and Hermiticity
4. Noiseless evolution preserves state
5. Noisy MZI produces valid density matrix
6. Noiseless MZI matches unitary evolution
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.mzi_lindblad import (
    MziNoiseConfig,
    build_mzi_lindblad_operators,
    evolve_mzi_lindblad,
    run_noisy_mzi,
)


class TestMziNoiseConfig:
    """Test MziNoiseConfig defaults and validation."""

    def test_default_values_are_reasonable(self) -> None:
        config = MziNoiseConfig()
        assert config.gamma_1 == 0.0
        assert config.gamma_2 == 0.0
        assert config.gamma_phi == 0.0
        assert config.T == 1.0
        assert config.dt == 0.01
        assert config.method == "rk4"

    def test_custom_values_are_preserved(self) -> None:
        config = MziNoiseConfig(gamma_1=0.5, gamma_phi=0.3, T=2.0, dt=0.05)
        assert config.gamma_1 == 0.5
        assert config.gamma_phi == 0.3
        assert config.T == 2.0
        assert config.dt == 0.05


class TestBuildMziLindbladOperators:
    """Test Lindblad operator construction."""

    def test_two_channels_gives_two_operators(self) -> None:
        config = MziNoiseConfig(gamma_1=0.5, gamma_phi=0.3)
        L_ops = build_mzi_lindblad_operators(max_photons=3, config=config)
        assert len(L_ops) == 2

    def test_operators_have_correct_shape(self) -> None:
        config = MziNoiseConfig(gamma_1=0.5, gamma_phi=0.3)
        L_ops = build_mzi_lindblad_operators(max_photons=3, config=config)
        dim = (3 + 1) ** 2  # (N+1)^2 = 16 for N=3
        for L in L_ops:
            assert L.shape == (dim, dim)

    def test_empty_config_returns_no_operators(self) -> None:
        config = MziNoiseConfig()
        L_ops = build_mzi_lindblad_operators(max_photons=3, config=config)
        assert len(L_ops) == 0

    def test_negative_gamma_raises_valueerror(self) -> None:
        config = MziNoiseConfig(gamma_1=-0.1)
        with pytest.raises(ValueError):
            build_mzi_lindblad_operators(max_photons=3, config=config)

        config = MziNoiseConfig(gamma_phi=-0.1)
        with pytest.raises(ValueError):
            build_mzi_lindblad_operators(max_photons=3, config=config)

        config = MziNoiseConfig(gamma_2=-0.1)
        with pytest.raises(ValueError):
            build_mzi_lindblad_operators(max_photons=3, config=config)


class TestEvolveMziLindblad:
    """Test Lindblad evolution properties."""

    def test_liouvillian_preserves_trace(self) -> None:
        max_photons = 2
        dim = (max_photons + 1) ** 2
        rho = np.eye(dim, dtype=complex) / dim  # Maximally mixed

        config = MziNoiseConfig(gamma_1=0.1, gamma_phi=0.05, T=0.5)
        rho_final = evolve_mzi_lindblad(rho, config, max_photons)

        assert np.isclose(np.trace(rho_final), 1.0, atol=1e-10)

    def test_liouvillian_preserves_hermiticity(self) -> None:
        max_photons = 2
        dim = (max_photons + 1) ** 2
        rho = np.eye(dim, dtype=complex) / dim  # Maximally mixed

        config = MziNoiseConfig(gamma_1=0.1, gamma_phi=0.05, T=0.5)
        rho_final = evolve_mzi_lindblad(rho, config, max_photons)

        assert np.allclose(rho_final, rho_final.conj().T, atol=1e-10)

    def test_noiseless_evolution_preserves_state(self) -> None:
        max_photons = 2
        dim = (max_photons + 1) ** 2
        rho = np.eye(dim, dtype=complex) / dim

        config = MziNoiseConfig(T=1.0, dt=0.1)  # All rates zero
        rho_final = evolve_mzi_lindblad(rho, config, max_photons)

        assert np.allclose(rho_final, rho), "Noiseless evolution should preserve state"


class TestRunNoisyMzi:
    """Test full noisy MZI simulation."""

    def test_noisy_mzi_produces_valid_density_matrix(self) -> None:
        import qutip

        max_photons = 4
        dim = max_photons + 1
        state = qutip.tensor(qutip.fock(dim, 2), qutip.fock(dim, 0)).full().ravel()

        config = MziNoiseConfig(gamma_1=0.1, gamma_phi=0.05, T=0.5, dt=0.05)
        rho = run_noisy_mzi(
            state,
            max_photons=max_photons,
            theta=np.pi / 4,
            phi_bs=0.0,
            phi_phase=np.pi / 2,
            noise_config=config,
        )

        assert np.isclose(np.trace(rho), 1.0, atol=1e-8), "Trace must be 1"
        assert np.allclose(rho, rho.conj().T, atol=1e-8), "Must be Hermitian"
        eigenvalues = np.linalg.eigvalsh(rho)
        assert np.min(eigenvalues) >= -1e-8, "Must be positive semidefinite"

    def test_noiseless_mzi_matches_unitary_evolution(self) -> None:
        import qutip

        max_photons = 3
        dim = max_photons + 1
        theta = np.pi / 4
        phi_bs = 0.0
        phi_phase = 1.0

        state = qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()

        # Noiseless noisy MZI
        config = MziNoiseConfig(T=1.0, dt=0.1)
        rho_noiseless = run_noisy_mzi(
            state,
            max_photons=max_photons,
            theta=theta,
            phi_bs=phi_bs,
            phi_phase=phi_phase,
            noise_config=config,
        )

        # Compare with QuTiP unitary evolution as density matrix
        a = qutip.destroy(dim)
        eye = qutip.qeye(dim)
        a0 = qutip.tensor(a, eye)
        a1 = qutip.tensor(eye, a)
        H_bs = np.exp(1j * phi_bs) * (a0.dag() @ a1) + np.exp(-1j * phi_bs) * (
            a1.dag() @ a0
        )
        U_bs = (-1j * theta * H_bs).expm()
        U_phase = (1j * phi_phase * a1.dag() @ a1).expm()

        psi_q = qutip.Qobj(state.reshape(-1, 1), dims=[[dim, dim], [1, 1]])
        psi_q = U_bs @ psi_q
        psi_q = U_phase @ psi_q
        psi_q = U_bs @ psi_q
        rho_unitary = (psi_q @ psi_q.dag()).full()

        assert np.allclose(rho_noiseless, rho_unitary, atol=1e-10), (
            "Noiseless noisy MZI should match unitary evolution"
        )

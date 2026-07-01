"""Tests for the MZI Lindblad master equation module.

Tests verify:
1. MziNoiseConfig defaults and validation
2. Lindblad operator construction (build_mzi_lindblad_operators)
3. Noisy MZI produces valid density matrix
4. Noiseless MZI matches unitary evolution
"""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.mzi_lindblad import (
    MziNoiseConfig,
    build_mzi_lindblad_operators,
    run_noisy_mzi,
    run_noisy_mzi_hamiltonian,
)


class TestMziNoiseConfig:
    """Test MziNoiseConfig defaults and validation."""

    def test_default_values_are_reasonable(self) -> None:
        config = MziNoiseConfig()
        assert config.gamma_1 == 0.0
        assert config.gamma_2 == 0.0
        assert config.gamma_phi == 0.0
        assert config.T_decay == 1.0
        assert config.dt == 0.01
        assert config.method == "rk4"

    def test_custom_values_are_preserved(self) -> None:
        config = MziNoiseConfig(gamma_1=0.5, gamma_phi=0.3, T_decay=2.0, dt=0.05)
        assert config.gamma_1 == 0.5
        assert config.gamma_phi == 0.3
        assert config.T_decay == 2.0
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


class TestRunNoisyMzi:
    """Test full noisy MZI simulation."""

    def test_noisy_mzi_produces_valid_density_matrix(self) -> None:
        import qutip

        max_photons = 4
        dim = max_photons + 1
        state = qutip.tensor(qutip.fock(dim, 2), qutip.fock(dim, 0)).full().ravel()

        config = MziNoiseConfig(gamma_1=0.1, gamma_phi=0.05, T_decay=0.5, dt=0.05)
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
        config = MziNoiseConfig(T_decay=1.0, dt=0.1)
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


class TestRunNoisyMziHamiltonian:
    """Test the Hamiltonian-based phase-encoding MZI."""

    def test_produces_valid_density_matrix(self) -> None:
        """Final ρ must be trace-1, Hermitian, and positive."""
        import qutip

        max_photons = 2
        dim = max_photons + 1
        state = qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()

        config = MziNoiseConfig(gamma_1=0.1, gamma_phi=0.05, T_decay=0.5)
        rho = run_noisy_mzi_hamiltonian(
            state,
            max_photons=max_photons,
            theta=np.pi / 4,
            phi_bs=0.0,
            omega=1.0,
            noise_config=config,
        )

        assert np.isclose(np.trace(rho), 1.0, atol=1e-8), "Trace must be 1"
        assert np.allclose(rho, rho.conj().T, atol=1e-8), "Must be Hermitian"
        eigenvalues = np.linalg.eigvalsh(rho)
        assert np.min(eigenvalues) >= -1e-8, "Must be positive semidefinite"

    def test_noiseless_matches_unitary(self) -> None:
        """At γ=0, must match explicit unitary exp(-i ω t_hold J_z)."""
        import qutip

        max_photons = 1
        dim = max_photons + 1
        theta = np.pi / 4
        phi_bs = 0.0
        omega = 0.5
        t_hold = 2.0

        state_v = qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()

        config = MziNoiseConfig(T_decay=t_hold)
        rho_ham = run_noisy_mzi_hamiltonian(
            state_v,
            max_photons=max_photons,
            theta=theta,
            phi_bs=phi_bs,
            omega=omega,
            noise_config=config,
        )

        # Explicit unitary: BS1 → U_ϕ → BS2
        a = qutip.destroy(dim)
        eye = qutip.qeye(dim)
        a0 = qutip.tensor(a, eye)
        a1 = qutip.tensor(eye, a)
        n0 = a0.dag() @ a0
        n1 = a1.dag() @ a1
        jz = 0.5 * (n0 - n1)

        H_bs_exp = np.exp(1j * phi_bs) * (a0.dag() @ a1) + np.exp(-1j * phi_bs) * (
            a1.dag() @ a0
        )
        U_bs = (-1j * theta * H_bs_exp).expm()
        U_phi = (-1j * omega * t_hold * jz).expm()

        psi = qutip.Qobj(state_v.reshape(-1, 1), dims=[[dim, dim], [1, 1]])
        psi = U_bs @ psi
        psi = U_phi @ psi
        psi = U_bs @ psi
        rho_explicit = (psi @ psi.dag()).full()

        assert np.allclose(rho_ham, rho_explicit, atol=1e-10), (
            "Noiseless Hamiltonian MZI should match explicit unitary evolution"
        )

    def test_sql_recovery_at_mid_fringe(self) -> None:
        """Noiseless must recover ⟨J_z⟩ = -0.5·cos(ω·t_hold)."""
        import qutip

        max_photons = 1
        dim = max_photons + 1
        omega = 1.0
        t_hold = 1.0
        state_v = qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()

        config = MziNoiseConfig(T_decay=t_hold)
        rho = run_noisy_mzi_hamiltonian(
            state_v,
            max_photons=max_photons,
            theta=np.pi / 4,
            phi_bs=0.0,
            omega=omega,
            noise_config=config,
        )
        a = qutip.destroy(dim)
        eye = qutip.qeye(dim)
        a0 = qutip.tensor(a, eye)
        a1 = qutip.tensor(eye, a)
        n0 = a0.dag() @ a0
        n1 = a1.dag() @ a1
        jz = 0.5 * (n0 - n1)
        jz_mean = float(np.real(np.trace(rho @ jz.full())))

        expected = -0.5 * np.cos(omega * t_hold)
        assert np.isclose(jz_mean, expected, atol=1e-10), (
            f"\u27e8J_z\u27e9 = {jz_mean}, expected {expected}"
        )

    def test_noise_degradation_monotonic_in_rate(self) -> None:
        """Δω/SQL must increase monotonically with γ_φ."""
        import qutip

        max_photons = 1
        dim = max_photons + 1
        t_hold = 1.0
        omega = 1.0
        state_v = qutip.tensor(qutip.fock(dim, 1), qutip.fock(dim, 0)).full().ravel()

        a = qutip.destroy(dim)
        eye = qutip.qeye(dim)
        a0 = qutip.tensor(a, eye)
        a1 = qutip.tensor(eye, a)
        n0 = a0.dag() @ a0
        n1 = a1.dag() @ a1
        jz = 0.5 * (n0 - n1)

        prev_ratio = -1.0
        for gamma in [0.0, 0.1, 0.5]:
            config = MziNoiseConfig(gamma_phi=gamma, T_decay=t_hold)
            rho = run_noisy_mzi_hamiltonian(
                state_v,
                max_photons=max_photons,
                theta=np.pi / 4,
                phi_bs=0.0,
                omega=omega,
                noise_config=config,
            )
            rho_plus = run_noisy_mzi_hamiltonian(
                state_v,
                max_photons=max_photons,
                theta=np.pi / 4,
                phi_bs=0.0,
                omega=omega + 1e-6,
                noise_config=config,
            )
            rho_minus = run_noisy_mzi_hamiltonian(
                state_v,
                max_photons=max_photons,
                theta=np.pi / 4,
                phi_bs=0.0,
                omega=omega - 1e-6,
                noise_config=config,
            )
            jz_mean = float(np.real(np.trace(rho @ jz.full())))
            jz_mean_p = float(np.real(np.trace(rho_plus @ jz.full())))
            jz_mean_m = float(np.real(np.trace(rho_minus @ jz.full())))
            d_jz = (jz_mean_p - jz_mean_m) / 2e-6
            jz_var = float(np.real(np.trace(rho @ (jz @ jz).full()))) - jz_mean**2
            denom = abs(d_jz)
            if denom < 1e-12:
                delta_omega = float("inf")
            else:
                delta_omega = np.sqrt(max(jz_var, 0.0)) / denom
            ratio = delta_omega / (1.0 / t_hold)

            assert ratio >= prev_ratio - 1e-10, (
                f"\u03b3={gamma}: ratio {ratio:.6f} < previous {prev_ratio:.6f}"
            )
            prev_ratio = ratio

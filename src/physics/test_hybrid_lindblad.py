"""
Tests for Hybrid Oscillator-Spin Lindblad Solver.

Physical Validation Tests:
- Hamiltonian Hermiticity: H = H†
- Lindblad operator structure: correct tensor product form
- Trace preservation: Tr[ρ(t)] = 1 for unitary, ≤ 1 with loss
- Hermiticity: ρ = ρ† at all times
- Positivity: eigenvalues of ρ ≥ 0
- Squeezing: n=2 should produce Gaussian squeezing
- Wigner negativity: n≥3 should show Wigner negativity
"""

import numpy as np
import pytest
import scipy

from .hybrid_lindblad import (
    HybridLindbladConfig,
    apply_squeezing,
    build_hybrid_hamiltonian,
    build_hybrid_lindblad_operators,
    evolve_hybrid_lindblad,
    lindblad_rhs,
    run_hybrid_simulation,
    validate_hybrid_density_matrix,
)
from .hybrid_system import (
    hybrid_vacuum_state,
    hybrid_mean_photon,
)


# =============================================================================
# Test Configuration
# =============================================================================


class TestHybridLindbladConfig:
    """Test configuration dataclass."""

    def test_default_values(self) -> None:
        """Default values should be reasonable."""
        config = HybridLindbladConfig(N=5)
        assert config.N == 5
        assert config.n == 2
        assert config.omega_n == 1.0
        assert config.theta_n == 0.0
        assert config.phi == 0.0
        assert config.gamma_1 == 0.0
        assert config.gamma_2 == 0.0
        assert config.gamma_phi == 0.0
        assert config.t_squeeze == 1.0

    def test_custom_values(self) -> None:
        """Custom values should be preserved."""
        config = HybridLindbladConfig(
            N=10,
            n=3,
            omega_n=0.5,
            theta_n=np.pi / 4,
            phi=0.1,
            gamma_1=0.01,
            gamma_phi=0.02,
            t_squeeze=2.0,
        )
        assert config.N == 10
        assert config.n == 3
        assert config.omega_n == 0.5
        assert config.phi == 0.1
        assert config.gamma_1 == 0.01


# =============================================================================
# Test Hamiltonian Construction
# =============================================================================


class TestBuildHybridHamiltonian:
    """Test Hamiltonian construction for different squeezing orders."""

    def test_hamiltonian_shape(self) -> None:
        """Hamiltonian should have correct shape."""
        for n in [2, 3, 4]:
            config = HybridLindbladConfig(N=5, n=n, omega_n=1.0)
            H = build_hybrid_hamiltonian(config)
            dim = 2 * (5 + 1)  # 2(N+1)
            assert H.shape == (dim, dim), f"n={n}: shape {H.shape} != {(dim, dim)}"

    def test_hamiltonian_hermiticity(self) -> None:
        """Hamiltonian should be Hermitian."""
        for n in [2, 3, 4]:
            config = HybridLindbladConfig(N=5, n=n, omega_n=1.0)
            H = build_hybrid_hamiltonian(config)
            assert np.allclose(H, H.conj().T), f"n={n}: H is not Hermitian"

    def test_hamiltonian_zero_omega(self) -> None:
        """Zero squeezing rate should give zero Hamiltonian."""
        config = HybridLindbladConfig(N=5, n=2, omega_n=0.0)
        H = build_hybrid_hamiltonian(config)
        assert np.allclose(H, 0), "Zero omega_n should give zero Hamiltonian"

    def test_hamiltonian_n2_vs_n3(self) -> None:
        """n=2 and n=3 should give different Hamiltonians."""
        config2 = HybridLindbladConfig(N=5, n=2, omega_n=1.0)
        config3 = HybridLindbladConfig(N=5, n=3, omega_n=1.0)
        H2 = build_hybrid_hamiltonian(config2)
        H3 = build_hybrid_hamiltonian(config3)
        assert not np.allclose(H2, H3), "n=2 and n=3 should differ"


# =============================================================================
# Test Lindblad Operator Construction
# =============================================================================


class TestBuildHybridLindbladOperators:
    """Test Lindblad operator construction."""

    def test_no_dissipation(self) -> None:
        """No dissipation should give empty lists."""
        config = HybridLindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 0
        assert len(gammas) == 0

    def test_one_body_loss(self) -> None:
        """One-body loss should add one operator."""
        config = HybridLindbladConfig(N=5, gamma_1=0.1)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 1
        assert len(gammas) == 1
        # Check shape: should be 2(N+1) x 2(N+1)
        dim = 2 * (5 + 1)
        assert L_ops[0].shape == (dim, dim)

    def test_phase_diffusion(self) -> None:
        """Phase diffusion should add one operator."""
        config = HybridLindbladConfig(N=5, gamma_phi=0.1)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 1
        assert len(gammas) == 1

    def test_multiple_channels(self) -> None:
        """Multiple channels should add multiple operators."""
        config = HybridLindbladConfig(N=5, gamma_1=0.1, gamma_2=0.05, gamma_phi=0.02)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 3
        assert len(gammas) == 3

    def test_operator_structure_one_body(self) -> None:
        """One-body loss: L = √γ₁ (a ⊗ I₂)."""
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        L_ops, _ = build_hybrid_lindblad_operators(config)

        # Check that L only acts on oscillator (spin unchanged)
        # For spin down (even indices), L should act as a
        # For spin up (odd indices), L should act as a
        L = L_ops[0]
        dim_osc = N + 1

        # Extract oscillator part for spin down
        L_down = L[::2, ::2][:dim_osc, :dim_osc]
        # Extract oscillator part for spin up
        L_up = L[1::2, 1::2][:dim_osc, :dim_osc]

        # Both should be equal to √γ₁ * a
        a = np.zeros((dim_osc, dim_osc), dtype=complex)
        for n in range(1, dim_osc):
            a[n - 1, n] = np.sqrt(n)

        assert np.allclose(L_down, np.sqrt(0.1) * a), "Spin down part incorrect"
        assert np.allclose(L_up, np.sqrt(0.1) * a), "Spin up part incorrect"


# =============================================================================
# Test Lindblad RHS
# =============================================================================


class TestLindbladRHS:
    """Test Lindblad right-hand side computation."""

    def test_unitary_only(self) -> None:
        """With no dissipation, should give -i[H, rho]."""
        N = 5
        config = HybridLindbladConfig(N=N, n=2, omega_n=1.0)
        H = build_hybrid_hamiltonian(config)
        L_ops, gammas = [], []

        dim = 2 * (N + 1)
        rho = np.eye(dim, dtype=complex) / dim  # Maximally mixed state

        drho = lindblad_rhs(rho, H, L_ops, gammas)

        # Should be -i[H, rho]
        expected = -1.0j * (H @ rho - rho @ H)
        assert np.allclose(drho, expected), "RHS incorrect for unitary case"

    def test_with_dissipation(self) -> None:
        """With dissipation, should have non-zero drift."""
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        H = np.zeros((2 * (N + 1), 2 * (N + 1)), dtype=complex)
        L_ops, gammas = build_hybrid_lindblad_operators(config)

        dim = 2 * (N + 1)
        # Use Fock state |1, down> (has a particle to lose)
        rho = np.zeros((dim, dim), dtype=complex)
        rho[2, 2] = 1.0  # Index for |1,down>: n=1, s=0 -> 1*2+0=2

        drho = lindblad_rhs(rho, H, L_ops, gammas)

        # Should be non-zero due to dissipation (particle can be lost)
        assert np.max(np.abs(drho)) > 0, "Dissipation should cause evolution"


# =============================================================================
# Test Evolution
# =============================================================================


class TestEvolveHybridLindblad:
    """Test time evolution under Lindblad equation."""

    def test_zero_time(self) -> None:
        """Zero evolution time should return initial state."""
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T=0.0, dt=0.01)

        # Convert to density matrix for comparison
        rho0 = np.outer(psi0, psi0.conj())
        assert np.allclose(rho_final, rho0), "Zero time should return initial state"

    def test_unitary_evolution(self) -> None:
        """With no dissipation, should match unitary evolution."""
        N = 5
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, t_squeeze=1.0)
        config.gamma_1 = 0.0
        config.gamma_phi = 0.0

        psi0 = hybrid_vacuum_state(N, spin_state="down")

        # Analytical: U = exp(-iH*t)
        H = build_hybrid_hamiltonian(config)
        U = scipy.linalg.expm(-1.0j * H * 1.0)
        rho_expected = U @ np.outer(psi0, psi0.conj()) @ U.conj().T

        # Numerical
        rho_final = evolve_hybrid_lindblad(psi0, config, T=1.0, dt=0.001, method="rk4")

        assert np.allclose(rho_final, rho_expected, atol=1e-4), (
            "Unitary evolution mismatch"
        )

    def test_trace_preservation_no_loss(self) -> None:
        """Trace should be preserved with no dissipation."""
        N = 5
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, gamma_1=0, gamma_phi=0)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T=1.0, dt=0.01)

        trace = np.trace(rho_final)
        assert np.isclose(trace, 1.0, atol=1e-6), f"Trace should be 1, got {trace}"

    def test_trace_lessthan_equal_with_loss(self) -> None:
        """Trace should be ≤ 1 with particle loss."""
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.2)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T=1.0, dt=0.01)

        trace = np.trace(rho_final)
        assert trace <= 1.0 + 1e-6, f"Trace should be ≤ 1, got {trace}"

    def test_hermiticity(self) -> None:
        """Density matrix should remain Hermitian."""
        N = 5
        config = HybridLindbladConfig(
            N=N, n=2, omega_n=0.3, gamma_1=0.1, gamma_phi=0.05
        )
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T=0.5, dt=0.01)

        assert np.allclose(rho_final, rho_final.conj().T, atol=1e-6), (
            "Final state not Hermitian"
        )

    def test_positivity(self) -> None:
        """Eigenvalues should be non-negative."""
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T=0.5, dt=0.01)

        eigenvalues = np.linalg.eigvalsh(rho_final)
        min_ev = np.min(eigenvalues.real)
        assert min_ev >= -1e-6, f"Negative eigenvalue: {min_ev}"


# =============================================================================
# Test Squeezing Protocol
# =============================================================================


class TestApplySqueezing:
    """Test squeezing protocol."""

    def test_squeezing_creates_photons(self) -> None:
        """Squeezing should increase mean photon number."""
        N = 10
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, t_squeeze=1.0)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        # Initial photon number
        n_initial = hybrid_mean_photon(psi0, N)

        # Apply squeezing
        psi_sq = apply_squeezing(config, psi0)
        n_squeezed = hybrid_mean_photon(psi_sq, N)

        assert n_squeezed > n_initial, (
            f"Squeezing should increase <n>: {n_initial} -> {n_squeezed}"
        )

    def test_squeezing_norm_preservation(self) -> None:
        """Squeezing should preserve norm."""
        N = 10
        config = HybridLindbladConfig(N=N, n=3, omega_n=0.3, t_squeeze=2.0)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        psi_sq = apply_squeezing(config, psi0)
        norm = np.sum(np.abs(psi_sq) ** 2)

        assert np.isclose(norm, 1.0, atol=1e-6), f"Norm not preserved: {norm}"


# =============================================================================
# Test Validation
# =============================================================================


class TestValidateHybridDensityMatrix:
    """Test density matrix validation."""

    def test_valid_density_matrix(self) -> None:
        """Valid density matrix should pass all checks."""
        N = 5
        dim = 2 * (N + 1)
        rho = np.eye(dim, dtype=complex) / dim  # Maximally mixed

        result = validate_hybrid_density_matrix(rho)

        assert result["is_hermitian"]
        assert result["is_normalized"]
        assert result["is_positive"]

    def test_pure_state(self) -> None:
        """Pure state should be valid."""
        N = 5
        psi = hybrid_vacuum_state(N, spin_state="down")
        rho = np.outer(psi, psi.conj())

        result = validate_hybrid_density_matrix(rho)

        assert result["is_hermitian"]
        assert result["is_normalized"]
        assert result["is_positive"]

    def test_non_hermitian_fails(self) -> None:
        """Non-Hermitian matrix should fail."""
        N = 5
        dim = 2 * (N + 1)
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 1] = 1.0  # Not Hermitian

        result = validate_hybrid_density_matrix(rho)

        assert not result["is_hermitian"]


# =============================================================================
# Test Complete Simulation
# =============================================================================


class TestRunHybridSimulation:
    """Test complete simulation runs."""

    def test_simulation_completes(self) -> None:
        """Simulation should complete without errors."""
        config = HybridLindbladConfig(
            N=5, n=2, omega_n=0.5, t_squeeze=0.5, gamma_1=0.01
        )
        result = run_hybrid_simulation(config)

        assert "final_state" in result
        assert "validation" in result

        validation = result["validation"]
        assert validation["is_hermitian"], "Final state not Hermitian"
        assert validation["is_positive"], "Final state not positive"

    def test_simulation_n3(self) -> None:
        """Simulation with n=3 should work."""
        config = HybridLindbladConfig(N=8, n=3, omega_n=0.3, t_squeeze=1.0, gamma_1=0.0)
        result = run_hybrid_simulation(config)

        assert result["final_state"] is not None
        validation = result["validation"]
        assert validation["is_hermitian"]


# =============================================================================
# Test Wigner Negativity (n≥3)
# =============================================================================


class TestWignerNegativity:
    """Test that n≥3 states show Wigner negativity."""

    def test_n2_no_negativity(self) -> None:
        """n=2 (Gaussian) should not have Wigner negativity."""
        N = 10
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, t_squeeze=1.0)
        psi_sq = apply_squeezing(config)

        # Extract oscillator density matrix (trace out spin)
        rho_osc = self._extract_oscillator_density(psi_sq, N)

        # Compute Wigner function
        from .wigner import wigner_function_single

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)
        W = wigner_function_single(rho_osc, x, p)

        min_W = np.min(W)
        # Gaussian states have W >= 0
        assert min_W >= -1e-3, f"n=2 should not have strong negativity: {min_W}"

    def test_n3_negativity(self) -> None:
        """n=3 (non-Gaussian) should show Wigner negativity."""
        N = 10
        config = HybridLindbladConfig(N=N, n=3, omega_n=0.3, t_squeeze=2.0)
        psi_sq = apply_squeezing(config)

        # Extract oscillator density matrix
        rho_osc = self._extract_oscillator_density(psi_sq, N)

        # Compute Wigner function
        from .wigner import wigner_function_single

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)
        W = wigner_function_single(rho_osc, x, p)

        min_W = np.min(W)
        # Non-Gaussian states should have W < 0
        assert min_W < -1e-3, f"n=3 should have Wigner negativity: min(W)={min_W}"

    def _extract_oscillator_density(
        self, hybrid_state: np.ndarray, N: int
    ) -> np.ndarray:
        """Extract oscillator density matrix from hybrid state."""
        dim_osc = N + 1
        rho_hybrid = np.outer(hybrid_state, hybrid_state.conj())
        # Reshape to (dim_osc, 2, dim_osc, 2) and trace over spin
        rho_reshaped = rho_hybrid.reshape(dim_osc, 2, dim_osc, 2)
        rho_osc = np.trace(rho_reshaped, axis1=1, axis2=3)
        return rho_osc


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_small_N(self) -> None:
        """Should work with N=1."""
        config = HybridLindbladConfig(N=1, n=2, omega_n=0.5)
        psi0 = hybrid_vacuum_state(1, spin_state="down")
        rho = evolve_hybrid_lindblad(psi0, config, T=0.1, dt=0.01)
        assert rho.shape == (4, 4)  # 2*(1+1) = 4

    def test_large_gamma(self) -> None:
        """Large gamma should transfer population to vacuum (particle loss)."""
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=10.0)
        # Create a coherent state with photons (not vacuum)
        from .hybrid_system import hybrid_coherent_state, hybrid_mean_photon

        psi0 = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")

        # Check initial photon number
        initial_n = hybrid_mean_photon(psi0, N)
        print(f"Initial <n>: {initial_n}")

        rho = evolve_hybrid_lindblad(psi0, config, T=1.0, dt=0.01)

        # Check final photon number (should be much lower)
        # Need to compute expectation of n_op ⊗ I_spin
        dim_osc = N + 1
        n_op = np.zeros((dim_osc, dim_osc), dtype=complex)
        for n in range(dim_osc):
            n_op[n, n] = n
        n_hybrid = np.kron(n_op, np.eye(2, dtype=complex))

        final_n = np.real(np.trace(rho @ n_hybrid))
        print(f"Final <n>: {final_n}")

        # Photon number should decrease significantly
        assert final_n < initial_n * 0.5, (
            f"Photon number should decrease: {initial_n} -> {final_n}"
        )

        # Trace should be preserved (Lindblad eq preserves trace)
        trace = np.trace(rho)
        assert np.isclose(trace, 1.0, atol=1e-6), f"Trace should be 1, got {trace}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

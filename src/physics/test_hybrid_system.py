"""
Tests for Hybrid Oscillator-Spin System module.

Physical Validation:
- Spin operators are Hermitian and satisfy Pauli algebra
- Oscillator operators satisfy [a, a†] = 1
- Hybrid Hamiltonian is Hermitian
- State normalization is preserved
- Adaptive truncation produces reasonable bounds
"""

import numpy as np
import pytest
import scipy

from .hybrid_system import (
    spin_operator_x,
    spin_operator_y,
    spin_operator_z,
    spin_operator_phi,
    oscillator_annihilation,
    oscillator_creation,
    oscillator_number,
    oscillator_power,
    hybrid_operator,
    hybrid_hamiltonian_n,
    hybrid_ground_state_n,
    hybrid_vacuum_state,
    hybrid_coherent_state,
    adaptive_truncation,
    hybrid_mean_photon,
    validate_hybrid_state,
    validate_hybrid_unitary,
)


# =============================================================================
# Test Spin Operators
# =============================================================================


class TestSpinOperators:
    """Test Pauli matrix properties."""

    def test_sigma_x_hermitian(self) -> None:
        sx = spin_operator_x()
        assert np.allclose(sx, sx.conj().T)

    def test_sigma_y_hermitian(self) -> None:
        sy = spin_operator_y()
        assert np.allclose(sy, sy.conj().T)

    def test_sigma_z_hermitian(self) -> None:
        sz = spin_operator_z()
        assert np.allclose(sz, sz.conj().T)

    def test_sigma_x_square(self) -> None:
        sx = spin_operator_x()
        assert np.allclose(sx @ sx, np.eye(2))

    def test_sigma_z_eigenvalues(self) -> None:
        sz = spin_operator_z()
        eigenvalues = np.linalg.eigvalsh(sz)
        assert np.allclose(eigenvalues, [-1, 1])

    def test_sigma_phi_hermitian(self) -> None:
        for phi in [0.0, np.pi / 4, np.pi / 2]:
            sphi = spin_operator_phi(phi)
            assert np.allclose(sphi, sphi.conj().T, atol=1e-10)

    def test_sigma_phi_normalization(self) -> None:
        """σ_φ should have eigenvalues ±1."""
        sphi = spin_operator_phi(np.pi / 3)
        eigenvalues = np.linalg.eigvalsh(sphi)
        assert np.allclose(np.abs(eigenvalues), [1, 1], atol=1e-10)


# =============================================================================
# Test Oscillator Operators
# =============================================================================


class TestOscillatorOperators:
    """Test bosonic operator properties."""

    def test_annihilation_dimension(self) -> None:
        a = oscillator_annihilation(N=5)
        assert a.shape == (6, 6)

    def test_creation_dimension(self) -> None:
        a_dag = oscillator_creation(N=5)
        assert a_dag.shape == (6, 6)

    def test_commutation_relation(self) -> None:
        """[a, a†] ≈ I (with truncation effect at boundary).

        In truncated Fock space, a|N⟩ = 0 (since |N+1⟩ is truncated).
        And a†|N⟩ = 0 (since |N+1⟩ is truncated).
        So [a, a†]|N⟩ = -a†a|N⟩ = -N|N⟩ instead of |N⟩.

        Actually, let me compute it properly:
        [a, a†]|n⟩ = a a†|n⟩ - a† a|n⟩
        For n < N: a a†|n⟩ = (n+1)|n⟩, a†a|n⟩ = n|n⟩, so [a,a†]|n⟩ = |n⟩
        For n = N: a a†|N⟩ = 0 (truncated), a†a|N⟩ = N|N⟩, so [a,a†]|N⟩ = -N|N⟩
        """
        N = 5
        a = oscillator_annihilation(N)
        a_dag = oscillator_creation(N)
        commutator = a @ a_dag - a_dag @ a

        # Compute expected [a, a†] in truncated basis
        expected = np.eye(N + 1, dtype=complex)
        expected[N, N] = -N  # Truncation effect: [a,a†]|N⟩ = -N|N⟩

        assert np.allclose(commutator, expected, atol=1e-10)

    def test_number_operator_diagonal(self) -> None:
        N = 5
        n = oscillator_number(N)
        # Should be diagonal with values 0, 1, ..., N
        expected_diag = np.arange(N + 1, dtype=complex)
        assert np.allclose(np.diag(n), expected_diag)

    def test_annihilation_action(self) -> None:
        """a|n⟩ = √n |n-1⟩."""
        N = 5
        a = oscillator_annihilation(N)
        # Check a|3⟩ = √3 |2⟩
        state_3 = np.zeros(N + 1, dtype=complex)
        state_3[3] = 1.0
        result = a @ state_3
        expected = np.zeros(N + 1, dtype=complex)
        expected[2] = np.sqrt(3)
        assert np.allclose(result, expected)

    def test_oscillator_power(self) -> None:
        """Test a^n computation."""
        N = 5
        a = oscillator_annihilation(N)
        a2 = oscillator_power(a, 2)
        # a^2|2⟩ = a(√2|1⟩) = √2·√1|0⟩ = √2|0⟩
        state_2 = np.zeros(N + 1, dtype=complex)
        state_2[2] = 1.0
        result = a2 @ state_2
        expected = np.zeros(N + 1, dtype=complex)
        expected[0] = np.sqrt(2)
        assert np.allclose(result, expected)

    def test_oscillator_power_zero(self) -> None:
        """a^0 should be identity."""
        N = 5
        a = oscillator_annihilation(N)
        a0 = oscillator_power(a, 0)
        assert np.allclose(a0, np.eye(N + 1))


# =============================================================================
# Test Hybrid Operators
# =============================================================================


class TestHybridOperators:
    """Test hybrid system operator construction."""

    def test_hybrid_operator_dimension(self) -> None:
        N = 5
        a = oscillator_annihilation(N)
        sz = spin_operator_z()
        op = hybrid_operator(a, sz, N)
        assert op.shape == (2 * (N + 1), 2 * (N + 1))

    def test_hybrid_operator_hermitian(self) -> None:
        N = 5
        a = oscillator_annihilation(N)
        a_dag = oscillator_creation(N)
        # a + a† is Hermitian
        x = a + a_dag
        sx = spin_operator_x()
        op = hybrid_operator(x, sx, N)
        assert np.allclose(op, op.conj().T, atol=1e-10)


# =============================================================================
# Test Hamiltonian Construction
# =============================================================================


class TestHybridHamiltonian:
    """Test n-th order squeezing Hamiltonian."""

    def test_hamiltonian_hermitian_n2(self) -> None:
        N = 5
        H = hybrid_hamiltonian_n(N, n=2, omega_n=1.0, theta_n=0.0)
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_hamiltonian_hermitian_n3(self) -> None:
        N = 5
        H = hybrid_hamiltonian_n(N, n=3, omega_n=1.0, theta_n=0.0)
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_hamiltonian_hermitian_n4(self) -> None:
        N = 5
        H = hybrid_hamiltonian_n(N, n=4, omega_n=1.0, theta_n=0.0)
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_hamiltonian_dimension(self) -> None:
        for n in [2, 3, 4]:
            N = 5
            H = hybrid_hamiltonian_n(N, n=n, omega_n=1.0, theta_n=0.0)
            assert H.shape == (2 * (N + 1), 2 * (N + 1))

    def test_hamiltonian_zero_omega(self) -> None:
        """H should be zero when omega_n = 0."""
        N = 5
        H = hybrid_hamiltonian_n(N, n=2, omega_n=0.0, theta_n=0.0)
        assert np.allclose(H, 0, atol=1e-10)

    def test_invalid_order_raises(self) -> None:
        N = 5
        with pytest.raises(ValueError, match="Unsupported order"):
            hybrid_hamiltonian_n(N, n=5, omega_n=1.0, theta_n=0.0)


# =============================================================================
# Test State Preparation
# =============================================================================


class TestStatePreparation:
    """Test hybrid state creation."""

    def test_vacuum_state_down(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        assert validate_hybrid_state(state, N)
        # Should be |0,↓⟩ = index 0
        assert np.isclose(state[0], 1.0)

    def test_vacuum_state_up(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="up")
        assert validate_hybrid_state(state, N)
        # Should be |0,↑⟩ = index 1
        assert np.isclose(state[1], 1.0)

    def test_coherent_state_normalized(self) -> None:
        N = 10
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        assert validate_hybrid_state(state, N)

    def test_coherent_state_amplitude(self) -> None:
        N = 10
        alpha = 1.0
        state = hybrid_coherent_state(N, alpha=alpha, spin_state="down")
        # Mean photon should be approximately |α|² = 1
        mean_n = hybrid_mean_photon(state, N)
        assert np.isclose(mean_n, 1.0, atol=0.1)

    def test_invalid_spin_state_raises(self) -> None:
        N = 5
        with pytest.raises(ValueError, match="Unknown spin_state"):
            hybrid_vacuum_state(N, spin_state="invalid")


# =============================================================================
# Test Adaptive Truncation
# =============================================================================


class TestAdaptiveTruncation:
    """Test adaptive truncation formula."""

    def test_vacuum_input(self) -> None:
        """Vacuum (alpha=0) with small r_n should give small N."""
        N = adaptive_truncation(alpha=0j, r_n=0.1, n=2, N_max=100)
        assert N > 0
        assert N <= 100

    def test_coherent_input(self) -> None:
        """Coherent state with |α|²=4 should give N ≥ 4."""
        N = adaptive_truncation(alpha=2.0 + 0j, r_n=0.0, n=2, N_max=100)
        assert N >= 4

    def test_rn_increases_truncation(self) -> None:
        """Larger r_n should give larger N."""
        N1 = adaptive_truncation(alpha=0j, r_n=1.0, n=2, N_max=100)
        N2 = adaptive_truncation(alpha=0j, r_n=5.0, n=2, N_max=100)
        assert N2 >= N1

    def test_respects_max(self) -> None:
        """N should not exceed N_max."""
        N = adaptive_truncation(alpha=100j, r_n=100.0, n=4, N_max=50)
        assert N <= 50

    def test_higher_order_gives_larger_N(self) -> None:
        """Higher order n should give larger N at same r_n (wider safety margin)."""
        N2 = adaptive_truncation(alpha=0j, r_n=1.0, n=2, N_max=200)
        N3 = adaptive_truncation(alpha=0j, r_n=1.0, n=3, N_max=200)
        N4 = adaptive_truncation(alpha=0j, r_n=1.0, n=4, N_max=200)
        assert N4 >= N3 >= N2


# =============================================================================
# Test Expectation Values
# =============================================================================


class TestExpectationValues:
    """Test expectation value computations."""

    def test_mean_photon_vacuum(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        mean_n = hybrid_mean_photon(state, N)
        assert np.isclose(mean_n, 0.0, atol=1e-10)

    def test_mean_photon_coherent(self) -> None:
        N = 10
        alpha = 2.0
        state = hybrid_coherent_state(N, alpha=alpha, spin_state="down")
        mean_n = hybrid_mean_photon(state, N)
        assert np.isclose(mean_n, alpha**2, rtol=1e-2)


# =============================================================================
# Test Validation Functions
# =============================================================================


class TestValidation:
    """Test validation functions."""

    def test_validate_good_state(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        assert validate_hybrid_state(state, N)

    def test_validate_wrong_dim(self) -> None:
        state = np.array([1.0, 0.0, 0.0])  # Wrong dimension
        assert not validate_hybrid_state(state, N=5)

    def test_validate_unnormalized(self) -> None:
        N = 5
        state = np.ones(2 * (N + 1), dtype=complex)
        assert not validate_hybrid_state(state, N)

    def test_validate_unitary_good(self) -> None:
        U = np.eye(4, dtype=complex)
        assert validate_hybrid_unitary(U)

    def test_validate_unitary_bad(self) -> None:
        U = np.array([[1, 1], [0, 1]], dtype=complex)
        assert not validate_hybrid_unitary(U)


# =============================================================================
# Test Unitary Evolution
# =============================================================================


class TestUnitaryEvolution:
    """Test time evolution under Hamiltonian."""

    def test_evolution_preserves_norm(self) -> None:
        """Unitary evolution should preserve norm."""
        N = 5
        H = hybrid_hamiltonian_n(N, n=2, omega_n=0.5, theta_n=0.0)
        # U = exp(-iHt)
        U = scipy.linalg.expm(-1j * H * 1.0)
        assert validate_hybrid_unitary(U, tol=1e-8)

        state = hybrid_vacuum_state(N, spin_state="down")
        evolved = U @ state
        assert np.isclose(np.sum(np.abs(evolved) ** 2), 1.0, atol=1e-6)

    def test_evolution_with_scipy(self) -> None:
        """Test that evolution works with scipy.linalg.expm."""
        N = 4
        H = hybrid_hamiltonian_n(N, n=3, omega_n=0.3, theta_n=np.pi / 4)
        T = 0.5
        U = scipy.linalg.expm(-1j * H * T)
        assert U.shape == (2 * (N + 1), 2 * (N + 1))


# =============================================================================
# Test Ground State
# =============================================================================


class TestHybridGroundState:
    """Tests for hybrid_ground_state_n."""

    def test_ground_state_dimension(self) -> None:
        """Ground state should have correct dimension."""
        N = 6
        for n_order in (2, 3, 4):
            gs = hybrid_ground_state_n(N, n=n_order, omega_n=0.5, theta_n=0.0)
            expected_dim = 2 * (N + 1)
            assert gs.shape == (expected_dim,), (
                f"n={n_order}: expected dim {expected_dim}, got {gs.shape}"
            )

    def test_ground_state_normalized(self) -> None:
        """Ground state should be normalised to 1."""
        N = 6
        for n_order in (2, 3, 4):
            gs = hybrid_ground_state_n(N, n=n_order, omega_n=0.5, theta_n=0.0)
            assert np.isclose(np.linalg.norm(gs), 1.0, atol=1e-10), (
                f"n={n_order}: norm not preserved"
            )

    def test_ground_state_is_lowest_eigenvector(self) -> None:
        """Ground state should be the lowest eigenvector of H_n."""
        N = 5
        n_order = 2
        omega_n = 1.0
        theta_n = 0.0
        H = hybrid_hamiltonian_n(N, n_order, omega_n, theta_n)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        gs = hybrid_ground_state_n(N, n_order, omega_n, theta_n)

        # The ground state should match the eigenvector for the smallest
        # eigenvalue (up to global phase)
        expected_gs = eigenvectors[:, 0]
        overlap = np.abs(np.vdot(expected_gs, gs))
        assert np.isclose(overlap, 1.0, atol=1e-10), (
            f"Overlap with lowest eigenvector: {overlap:.2e}"
        )

    def test_ground_state_energy(self) -> None:
        """Ground state energy should be the minimum eigenvalue."""
        N = 5
        for n_order in (2, 3, 4):
            H = hybrid_hamiltonian_n(N, n_order, omega_n=0.5, theta_n=0.0)
            eigenvalues = np.linalg.eigvalsh(H)
            gs = hybrid_ground_state_n(N, n_order, omega_n=0.5, theta_n=0.0)
            gs_energy = np.real(np.vdot(gs, H @ gs))
            assert np.isclose(gs_energy, eigenvalues[0], atol=1e-10), (
                f"n={n_order}: GS energy {gs_energy:.6e} != min eigenvalue {eigenvalues[0]:.6e}"
            )

    def test_invalid_order_raises(self) -> None:
        """Invalid order should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported order"):
            hybrid_ground_state_n(N=5, n=5, omega_n=0.5, theta_n=0.0)

    def test_ground_state_differs_from_vacuum(self) -> None:
        """For non-trivial omega_n, the ground state should differ from |0,↓⟩."""
        N = 5
        gs = hybrid_ground_state_n(N, n=2, omega_n=1.0, theta_n=0.0)
        vac = hybrid_vacuum_state(N, spin_state="down")
        overlap = np.abs(np.vdot(vac, gs))
        # For sufficiently strong coupling, the ground state is not vacuum
        assert overlap < 1.0, "Ground state should differ from bare vacuum"

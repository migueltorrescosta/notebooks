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
    adaptive_truncation,
    evolve_hybrid_state,
    hybrid_coherent_state,
    hybrid_ground_state_n,
    hybrid_hamiltonian_n,
    hybrid_mean_photon,
    hybrid_operator,
    hybrid_vacuum_state,
    oscillator_annihilation,
    oscillator_creation,
    oscillator_number,
    oscillator_power,
    spin_operator_phi,
    spin_operator_x,
    spin_operator_y,
    spin_operator_z,
    validate_hybrid_state,
    validate_hybrid_unitary,
)

# Test Spin Operators


class TestSpinOperators:
    """Test Pauli matrix properties."""

    def test_sigma_x_hermitian(self) -> None:
        sx = spin_operator_x()
        assert sx == pytest.approx(sx.conj().T), (
            "Expected sx == pytest.approx(sx.conj().T)"
        )

    def test_sigma_y_hermitian(self) -> None:
        sy = spin_operator_y()
        assert sy == pytest.approx(sy.conj().T), (
            "Expected sy == pytest.approx(sy.conj().T)"
        )

    def test_sigma_z_hermitian(self) -> None:
        sz = spin_operator_z()
        assert sz == pytest.approx(sz.conj().T), (
            "Expected sz == pytest.approx(sz.conj().T)"
        )

    def test_sigma_x_square(self) -> None:
        sx = spin_operator_x()
        assert sx @ sx == pytest.approx(np.eye(2)), (
            "Expected sx @ sx == pytest.approx(np.eye(2))"
        )

    def test_sigma_z_eigenvalues(self) -> None:
        sz = spin_operator_z()
        eigenvalues = np.linalg.eigvalsh(sz)
        assert eigenvalues == pytest.approx([-1, 1]), (
            "Expected eigenvalues == pytest.approx([-1, 1])"
        )

    @pytest.mark.parametrize(
        "phi", [0.0, np.pi / 4, np.pi / 2], ids=["0", "pi/4", "pi/2"]
    )
    def test_sigma_phi_hermitian(self, phi: float) -> None:
        sphi = spin_operator_phi(phi)
        assert sphi == pytest.approx(sphi.conj().T, abs=1e-10), (
            "Expected sphi == pytest.approx(sphi.conj().T, abs=1e-10)"
        )

    def test_given_sigma_phi_then_eigenvalues_are_1(self) -> None:
        sphi = spin_operator_phi(np.pi / 3)
        eigenvalues = np.linalg.eigvalsh(sphi)
        assert np.abs(eigenvalues) == pytest.approx([1, 1], abs=1e-10), (
            "Expected np.abs(eigenvalues) == pytest.approx([1, 1], abs=1e-10)"
        )


# Test Oscillator Operators


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

        assert commutator == pytest.approx(expected, abs=1e-10), (
            "Expected commutator == pytest.approx(expected, abs=1e-10)"
        )

    def test_number_operator_diagonal(self) -> None:
        N = 5
        n = oscillator_number(N)
        # Should be diagonal with values 0, 1, ..., N
        expected_diag = np.arange(N + 1, dtype=complex)
        assert np.diag(n) == pytest.approx(expected_diag), (
            "Expected np.diag(n) == pytest.approx(expected_diag)"
        )

    def test_a_n_n_n_1(self) -> None:
        N = 5
        a = oscillator_annihilation(N)
        # Check a|3⟩ = √3 |2⟩
        state_3 = np.zeros(N + 1, dtype=complex)
        state_3[3] = 1.0
        result = a @ state_3
        expected = np.zeros(N + 1, dtype=complex)
        expected[2] = np.sqrt(3)
        assert result == pytest.approx(expected), (
            "Expected result == pytest.approx(expected)"
        )

    def test_a_n_computation(self) -> None:
        N = 5
        a = oscillator_annihilation(N)
        a2 = oscillator_power(a, 2)
        # a^2|2⟩ = a(√2|1⟩) = √2·√1|0⟩ = √2|0⟩
        state_2 = np.zeros(N + 1, dtype=complex)
        state_2[2] = 1.0
        result = a2 @ state_2
        expected = np.zeros(N + 1, dtype=complex)
        expected[0] = np.sqrt(2)
        assert result == pytest.approx(expected), (
            "Expected result == pytest.approx(expected)"
        )

    def test_given_a_0_then_be_identity(self) -> None:
        N = 5
        a = oscillator_annihilation(N)
        a0 = oscillator_power(a, 0)
        assert a0 == pytest.approx(np.eye(N + 1)), (
            "Expected a0 == pytest.approx(np.eye(N + 1))"
        )


# Test Hybrid Operators


class TestHybridOperators:
    """Test hybrid system operator construction."""

    def test_hybrid_operator_dimension(self) -> None:
        N = 5
        a = oscillator_annihilation(N)
        sz = spin_operator_z()
        op = hybrid_operator(a, sz, N)
        assert op.shape == (2 * (N + 1), 2 * (N + 1)), (
            "Expected op.shape == (2 * (N + 1), 2 * (N + 1))"
        )

    def test_hybrid_operator_hermitian(self) -> None:
        N = 5
        a = oscillator_annihilation(N)
        a_dag = oscillator_creation(N)
        # a + a† is Hermitian
        x = a + a_dag
        sx = spin_operator_x()
        op = hybrid_operator(x, sx, N)
        assert op == pytest.approx(op.conj().T, abs=1e-10), (
            "Expected op == pytest.approx(op.conj().T, abs=1e-10)"
        )


# Test Hamiltonian Construction


class TestHybridHamiltonian:
    """Test n-th order squeezing Hamiltonian."""

    def test_hamiltonian_hermitian_n2(self) -> None:
        N = 5
        H = hybrid_hamiltonian_n(N, n=2, omega_n=1.0, theta_n=0.0)
        assert pytest.approx(H.conj().T, abs=1e-10) == H, (
            "Expected H == pytest.approx(H.conj().T, abs=1e-10)"
        )

    def test_hamiltonian_hermitian_n3(self) -> None:
        N = 5
        H = hybrid_hamiltonian_n(N, n=3, omega_n=1.0, theta_n=0.0)
        assert pytest.approx(H.conj().T, abs=1e-10) == H, (
            "Expected H == pytest.approx(H.conj().T, abs=1e-10)"
        )

    def test_hamiltonian_hermitian_n4(self) -> None:
        N = 5
        H = hybrid_hamiltonian_n(N, n=4, omega_n=1.0, theta_n=0.0)
        assert pytest.approx(H.conj().T, abs=1e-10) == H, (
            "Expected H == pytest.approx(H.conj().T, abs=1e-10)"
        )

    @pytest.mark.parametrize(
        "n", [2, 3, 4], ids=["2", "3", "4"]
    )
    def test_hamiltonian_dimension(self, n: int) -> None:
        N = 5
        H = hybrid_hamiltonian_n(N, n=n, omega_n=1.0, theta_n=0.0)
        assert H.shape == (2 * (N + 1), 2 * (N + 1)), (
            "Expected H.shape == (2 * (N + 1), 2 * (N + 1))"
        )

    def test_given_h_then_be_zero_when_omega_n_0(self) -> None:
        N = 5
        H = hybrid_hamiltonian_n(N, n=2, omega_n=0.0, theta_n=0.0)
        assert pytest.approx(0, abs=1e-10) == H, (
            "Expected H == pytest.approx(0, abs=1e-10)"
        )

    def test_invalid_order_raises(self) -> None:
        N = 5
        with pytest.raises(ValueError, match="Unsupported order"):
            hybrid_hamiltonian_n(N, n=5, omega_n=1.0, theta_n=0.0)


# Test State Preparation


class TestStatePreparation:
    """Test hybrid state creation."""

    def test_vacuum_state_down(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        assert validate_hybrid_state(state, N), (
            "Condition failed: validate_hybrid_state(state, N)"
        )
        # Should be |0,↓⟩ = index 0
        assert state[0] == pytest.approx(1.0)

    def test_vacuum_state_up(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="up")
        assert validate_hybrid_state(state, N), (
            "Condition failed: validate_hybrid_state(state, N)"
        )
        # Should be |0,↑⟩ = index 1
        assert state[1] == pytest.approx(1.0)

    def test_coherent_state_normalized(self) -> None:
        N = 10
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        assert validate_hybrid_state(state, N), (
            "Condition failed: validate_hybrid_state(state, N)"
        )

    def test_coherent_state_amplitude(self) -> None:
        N = 10
        alpha = 1.0
        state = hybrid_coherent_state(N, alpha=alpha, spin_state="down")
        # Mean photon should be approximately |α|² = 1
        mean_n = hybrid_mean_photon(state, N)
        assert mean_n == pytest.approx(1.0, abs=0.1), (
            "Expected mean_n == pytest.approx(1.0, abs=0.1)"
        )

    def test_invalid_spin_state_raises(self) -> None:
        N = 5
        with pytest.raises(ValueError, match="Unknown spin_state"):
            hybrid_vacuum_state(N, spin_state="invalid")


# Test Adaptive Truncation


class TestAdaptiveTruncation:
    """Test adaptive truncation formula."""

    def test_given_vacuum_alpha_0_with_small_r_n_then_give_small_n(self) -> None:
        N = adaptive_truncation(alpha=0j, r_n=0.1, n=2, N_max=100)
        assert N > 0
        assert N <= 100

    def test_given_coherent_state_with_4_then_give_n_4(self) -> None:
        N = adaptive_truncation(alpha=2.0 + 0j, r_n=0.0, n=2, N_max=100)
        assert N >= 4

    def test_given_larger_r_n_then_give_larger_n(self) -> None:
        N1 = adaptive_truncation(alpha=0j, r_n=1.0, n=2, N_max=100)
        N2 = adaptive_truncation(alpha=0j, r_n=5.0, n=2, N_max=100)
        assert N2 >= N1

    def test_given_n_then_not_exceed_n_max(self) -> None:
        N = adaptive_truncation(alpha=100j, r_n=100.0, n=4, N_max=50)
        assert N <= 50

    def test_given_higher_order_n_then_give_larger_n_at_same_r_n_wider_safety_margin(
        self,
    ) -> None:
        N2 = adaptive_truncation(alpha=0j, r_n=1.0, n=2, N_max=200)
        N3 = adaptive_truncation(alpha=0j, r_n=1.0, n=3, N_max=200)
        N4 = adaptive_truncation(alpha=0j, r_n=1.0, n=4, N_max=200)
        assert N4 >= N3 >= N2


# Test Expectation Values


class TestExpectationValues:
    """Test expectation value computations."""

    def test_mean_photon_vacuum(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        mean_n = hybrid_mean_photon(state, N)
        assert mean_n == pytest.approx(0.0, abs=1e-10), (
            "Expected mean_n == pytest.approx(0.0, abs=1e-10)"
        )

    def test_mean_photon_coherent(self) -> None:
        N = 10
        alpha = 2.0
        state = hybrid_coherent_state(N, alpha=alpha, spin_state="down")
        mean_n = hybrid_mean_photon(state, N)
        assert mean_n == pytest.approx(alpha**2, rel=1e-2), (
            "Expected mean_n == pytest.approx(alpha**2, rel=1e-2)"
        )


# Test Validation Functions


class TestValidation:
    """Test validation functions."""

    def test_validate_good_state(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        assert validate_hybrid_state(state, N), (
            "Condition failed: validate_hybrid_state(state, N)"
        )

    def test_validate_wrong_dim(self) -> None:
        state = np.array([1.0, 0.0, 0.0])  # Wrong dimension
        assert not validate_hybrid_state(state, N=5), (
            "validate_hybrid_state(state, N=5) should be falsy"
        )

    def test_validate_unnormalized(self) -> None:
        N = 5
        state = np.ones(2 * (N + 1), dtype=complex)
        assert not validate_hybrid_state(state, N), (
            "validate_hybrid_state(state, N) should be falsy"
        )

    def test_validate_unitary_good(self) -> None:
        U = np.eye(4, dtype=complex)
        assert validate_hybrid_unitary(U), (
            "Condition failed: validate_hybrid_unitary(U)"
        )

    def test_validate_unitary_bad(self) -> None:
        U = np.array([[1, 1], [0, 1]], dtype=complex)
        assert not validate_hybrid_unitary(U), (
            "validate_hybrid_unitary(U) should be falsy"
        )


# Test Unitary Evolution


class TestUnitaryEvolution:
    """Test time evolution under Hamiltonian."""

    def test_given_unitary_evolution_then_preserve_norm(self) -> None:
        N = 5
        H = hybrid_hamiltonian_n(N, n=2, omega_n=0.5, theta_n=0.0)
        # U = exp(-iHt)
        U = scipy.linalg.expm(-1j * H * 1.0)
        assert validate_hybrid_unitary(U, tol=1e-8), (
            "Condition failed: validate_hybrid_unitary(U, tol=1e-8)"
        )

        state = hybrid_vacuum_state(N, spin_state="down")
        evolved = U @ state
        assert np.sum(np.abs(evolved) ** 2) == pytest.approx(1.0, abs=1e-6), (
            "Expected np.sum(np.abs(evolved) ** 2) == pytest.approx(1.0, abs=1e-6)"
        )

    def test_that_evolution_works_with_scipy_linalg_expm(self) -> None:
        N = 4
        H = hybrid_hamiltonian_n(N, n=3, omega_n=0.3, theta_n=np.pi / 4)
        T = 0.5
        U = scipy.linalg.expm(-1j * H * T)
        assert U.shape == (2 * (N + 1), 2 * (N + 1)), (
            "Expected U.shape == (2 * (N + 1), 2 * (N + 1))"
        )

    # --- evolve_hybrid_state ---

    def test_given_evolve_then_preserve_norm_for_all_orders(self) -> None:
        N = 5
        for n_order in (2, 3, 4):
            initial = hybrid_vacuum_state(N, spin_state="down")
            evolved = evolve_hybrid_state(
                N=N,
                n=n_order,
                omega_n=0.5,
                theta_n=0.0,
                t=1.0,
                initial_state=initial,
            )
            norm = np.sum(np.abs(evolved) ** 2)
            assert norm == pytest.approx(1.0, abs=1e-6), (
                f"n={n_order}: norm {norm:.6e} != 1"
            )

    def test_given_evolve_with_zero_time_then_return_initial(self) -> None:
        N = 5
        initial = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        evolved = evolve_hybrid_state(
            N=N,
            n=2,
            omega_n=0.5,
            theta_n=0.0,
            t=0.0,
            initial_state=initial,
        )
        assert np.allclose(evolved, initial, atol=1e-10), (
            "Evolved state should match initial at t=0"
        )

    def test_given_evolve_then_have_correct_dimension(self) -> None:
        N = 5
        initial = hybrid_vacuum_state(N, spin_state="down")
        evolved = evolve_hybrid_state(
            N=N,
            n=2,
            omega_n=0.5,
            theta_n=0.0,
            t=1.0,
            initial_state=initial,
        )
        assert evolved.shape == (2 * (N + 1),), (
            f"Expected shape {(2 * (N + 1),)}, got {evolved.shape}"
        )

    def test_given_evolve_with_zero_omega_then_not_change_state(self) -> None:
        N = 5
        initial = hybrid_coherent_state(N, alpha=1.5 + 0j, spin_state="down")
        evolved = evolve_hybrid_state(
            N=N,
            n=2,
            omega_n=0.0,
            theta_n=0.0,
            t=1.0,
            initial_state=initial,
        )
        assert np.allclose(evolved, initial, atol=1e-10), (
            "Evolved state should match initial when omega_n=0"
        )


# Test Ground State


class TestHybridGroundState:
    """Tests for hybrid_ground_state_n."""

    def test_given_ground_state_then_have_correct_dimension(self) -> None:
        N = 6
        for n_order in (2, 3, 4):
            gs = hybrid_ground_state_n(N, n=n_order, omega_n=0.5, theta_n=0.0)
            expected_dim = 2 * (N + 1)
            assert gs.shape == (expected_dim,), (
                f"n={n_order}: expected dim {expected_dim}, got {gs.shape}"
            )

    def test_given_ground_state_then_be_normalised_to_1(self) -> None:
        N = 6
        for n_order in (2, 3, 4):
            gs = hybrid_ground_state_n(N, n=n_order, omega_n=0.5, theta_n=0.0)
            assert np.linalg.norm(gs) == pytest.approx(1.0, abs=1e-10), (
                f"n={n_order}: norm not preserved"
            )

    def test_given_ground_state_then_be_the_lowest_eigenvector_of_h_n(self) -> None:
        N = 5
        n_order = 2
        omega_n = 1.0
        theta_n = 0.0
        H = hybrid_hamiltonian_n(N, n_order, omega_n, theta_n)
        _eigenvalues, eigenvectors = np.linalg.eigh(H)
        gs = hybrid_ground_state_n(N, n_order, omega_n, theta_n)

        # The ground state should match the eigenvector for the smallest
        # eigenvalue (up to global phase)
        expected_gs = eigenvectors[:, 0]
        overlap = np.abs(np.vdot(expected_gs, gs))
        assert overlap == pytest.approx(1.0, abs=1e-10), (
            f"Overlap with lowest eigenvector: {overlap:.2e}"
        )

    def test_given_ground_state_energy_then_be_the_minimum_eigenvalue(self) -> None:
        N = 5
        for n_order in (2, 3, 4):
            H = hybrid_hamiltonian_n(N, n_order, omega_n=0.5, theta_n=0.0)
            eigenvalues = np.linalg.eigvalsh(H)
            gs = hybrid_ground_state_n(N, n_order, omega_n=0.5, theta_n=0.0)
            gs_energy = np.real(np.vdot(gs, H @ gs))
            assert gs_energy == pytest.approx(eigenvalues[0], abs=1e-10), (
                f"n={n_order}: GS energy {gs_energy:.6e} != min eigenvalue {eigenvalues[0]:.6e}"
            )

    def test_given_invalid_order_then_raise_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Unsupported order"):
            hybrid_ground_state_n(N=5, n=5, omega_n=0.5, theta_n=0.0)

    def test_given_for_non_trivial_omega_n_the_ground_state_then_differ_from_0(
        self,
    ) -> None:
        N = 5
        gs = hybrid_ground_state_n(N, n=2, omega_n=1.0, theta_n=0.0)
        vac = hybrid_vacuum_state(N, spin_state="down")
        overlap = np.abs(np.vdot(vac, gs))
        # For sufficiently strong coupling, the ground state is not vacuum
        assert overlap < 1.0, "Ground state should differ from bare vacuum"

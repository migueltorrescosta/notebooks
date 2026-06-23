"""
Tests for Hybrid Oscillator-Spin System module.

Physical Validation:
- Spin operators are Hermitian and satisfy Pauli algebra
- Oscillator operators satisfy [a, a†] = 1
- Hybrid Hamiltonian is Hermitian
- State normalization is preserved

Note: Tests for adaptive_truncation, hybrid_mean_photon, evolve_hybrid_state,
validate_hybrid_state, validate_hybrid_unitary, hybrid_coherent_state have been
migrated to reports/20260507/test_high_order_squeezing.py.
"""

import numpy as np
import pytest

from .hybrid_system import (
    hybrid_ground_state_n,
    hybrid_hamiltonian_n,
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

    @pytest.mark.parametrize("n", [2, 3, 4], ids=["2", "3", "4"])
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
    """Test hybrid state creation (vacuum only; coherent tests migrated)."""

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

    def test_invalid_spin_state_raises(self) -> None:
        N = 5
        with pytest.raises(ValueError, match="Unknown spin_state"):
            hybrid_vacuum_state(N, spin_state="invalid")


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

"""
Tests for Pseudomode Non-Markovian System.

Physical Validation Tests:
- Configuration validation and defaults
- Operator construction: Hermiticity, correct dimensions
- Hamiltonian: shape, Hermiticity, zero-coupling limits
- Lindblad operators: shape, structure, zero-rate handling
- State preparation: normalization, correct amplitudes
- Ancilla entanglement: unitarity, norm preservation
- Partial trace: trace conservation, dimension reduction
- Evolution: trace preservation, Hermiticity, positivity
- QFI computation: ℛ(0) = 1, ℛ(T) ≤ 1, cyclic invariance
- Pseudomode occupancy truncation check
"""

import numpy as np
import pytest
import scipy

from .pseudomode_system import (
    PseudomodeConfig,
    create_pseudomode_operators,
    pseudomode_number_operator,
    tripartite_operator,
    build_pseudomode_hamiltonian,
    build_pseudomode_lindblad_operators,
    pseudomode_initial_state,
    apply_ancilla_entanglement,
    evolve_pseudomode,
    trace_out_pseudomode,
    trace_out_spin,
    trace_out_spin_and_pseudomode,
    compute_qfi_with_ancilla,
    compute_qfi_without_ancilla,
    run_metrology_protocol,
    validate_pseudomode_density,
    check_pseudomode_occupancy,
    qfi_preservation_ratio,
)


# =============================================================================
# Test Configuration
# =============================================================================


class TestPseudomodeConfig:
    """Test configuration dataclass."""

    def test_default_values(self) -> None:
        """Default values should be reasonable."""
        config = PseudomodeConfig(N=5, K=3)
        assert config.N == 5
        assert config.K == 3
        assert config.alpha == 1.0
        assert config.g_sa == 1.0
        assert config.tau == 0.1
        assert config.g_sp == 0.5
        assert config.omega_0 == 0.0
        assert config.lam == 1.0
        assert config.T == 2.0
        assert config.dt == 0.01

    def test_custom_values(self) -> None:
        """Custom values should be preserved."""
        config = PseudomodeConfig(
            N=10,
            K=5,
            alpha=2.0,
            g_sa=0.5,
            tau=0.2,
            g_sp=1.0,
            omega_0=0.5,
            lam=0.1,
            T=5.0,
            dt=0.005,
        )
        assert config.N == 10
        assert config.K == 5
        assert config.alpha == 2.0
        assert config.g_sa == 0.5
        assert config.tau == 0.2
        assert config.g_sp == 1.0
        assert config.omega_0 == 0.5
        assert config.lam == 0.1
        assert config.T == 5.0
        assert config.dt == 0.005

    def test_negative_N_raises(self) -> None:
        """Negative N should raise ValueError."""
        with pytest.raises(ValueError):
            PseudomodeConfig(N=-1, K=3)

    def test_negative_K_raises(self) -> None:
        """Negative K should raise ValueError."""
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=-1)

    def test_negative_dt_raises(self) -> None:
        """Negative dt should raise ValueError."""
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, dt=-0.1)

    def test_zero_dt_raises(self) -> None:
        """Zero dt should raise ValueError."""
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, dt=0.0)

    def test_negative_T_raises(self) -> None:
        """Negative T should raise ValueError."""
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, T=-1.0)

    def test_negative_tau_raises(self) -> None:
        """Negative tau should raise ValueError."""
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, tau=-0.1)


# =============================================================================
# Test Pseudomode Operators
# =============================================================================


class TestCreatePseudomodeOperators:
    """Test pseudomode ladder operator construction."""

    def test_shape(self) -> None:
        """Operators should have correct shape."""
        for K in [0, 1, 3, 10]:
            b, b_dag = create_pseudomode_operators(K)
            assert b.shape == (K + 1, K + 1), f"K={K}: b shape {b.shape}"
            assert b_dag.shape == (K + 1, K + 1), f"K={K}: b_dag shape {b_dag.shape}"

    def test_commutation(self) -> None:
        """Should satisfy [b, b†] = I except last element (truncation)."""
        K = 10
        b, b_dag = create_pseudomode_operators(K)
        commutator = b @ b_dag - b_dag @ b
        expected = np.eye(K + 1, dtype=complex)
        # Truncated Fock space: the last diagonal element of [b,b†] is -K,
        # because b† maps |K⟩ → 0 (beyond truncation).
        # All other elements should equal identity.
        expected[K, K] = -K  # Truncation artifact
        assert np.allclose(commutator, expected, atol=1e-10), "[b, b†] != truncated I"

    def test_annihilation_action(self) -> None:
        """b|k⟩ = √k |k-1⟩."""
        K = 5
        b, _ = create_pseudomode_operators(K)
        for k in range(K + 1):
            ket = np.zeros(K + 1, dtype=complex)
            ket[k] = 1.0
            result = b @ ket
            if k > 0:
                assert np.isclose(result[k - 1], np.sqrt(k)), f"b|{k}>: wrong amplitude"
            else:
                assert np.allclose(result, 0), "b|0>: should be 0"

    def test_negative_K_raises(self) -> None:
        """Negative K should raise ValueError."""
        with pytest.raises(ValueError):
            create_pseudomode_operators(-1)


class TestPseudomodeNumberOperator:
    """Test pseudomode number operator."""

    def test_shape(self) -> None:
        """Should have correct shape."""
        for K in [0, 1, 5]:
            n = pseudomode_number_operator(K)
            assert n.shape == (K + 1, K + 1)

    def test_diagonal_values(self) -> None:
        """b†b|k⟩ = k|k⟩."""
        K = 5
        n = pseudomode_number_operator(K)
        for k in range(K + 1):
            assert np.isclose(n[k, k], k), f"n[{k},{k}] should be {k}"

    def test_consistency_with_operators(self) -> None:
        """b†b should equal b† @ b."""
        K = 5
        b, b_dag = create_pseudomode_operators(K)
        n_from_ops = b_dag @ b
        n_diag = pseudomode_number_operator(K)
        assert np.allclose(n_from_ops, n_diag), "b†b != number_operator"


# =============================================================================
# Test Tripartite Operator Construction
# =============================================================================


class TestTripartiteOperator:
    """Test tripartite operator construction."""

    def test_correct_dimensions(self) -> None:
        """Should produce correct total dimension."""
        N, K = 5, 3
        dim_total = 2 * (N + 1) * (K + 1)
        op = tripartite_operator(np.eye(N + 1), np.eye(2), np.eye(K + 1), N, K)
        assert op.shape == (dim_total, dim_total), (
            f"Expected ({dim_total}, {dim_total}), got {op.shape}"
        )

    def test_identity_product(self) -> None:
        """I_osc ⊗ I_spin ⊗ I_pm should equal I_total."""
        N, K = 5, 3
        dim_total = 2 * (N + 1) * (K + 1)
        op = tripartite_operator(np.eye(N + 1), np.eye(2), np.eye(K + 1), N, K)
        assert np.allclose(op, np.eye(dim_total)), (
            "Kronecker product of identities should be identity"
        )

    def test_zero_operator(self) -> None:
        """Zero input should give zero output."""
        N, K = 5, 3
        op = tripartite_operator(
            np.zeros((N + 1, N + 1)), np.eye(2), np.eye(K + 1), N, K
        )
        assert np.allclose(op, 0), "Zero osc_op should give zero operator"

    def test_dimension_mismatch_raises(self) -> None:
        """Dimension mismatch should raise AssertionError."""
        N, K = 5, 3
        with pytest.raises(AssertionError):
            tripartite_operator(np.eye(N + 2), np.eye(2), np.eye(K + 1), N, K)

    def test_hermitian_product(self) -> None:
        """Hermitian inputs should give Hermitian output."""
        N, K = 5, 3
        A = np.random.randn(N + 1, N + 1) + 1j * np.random.randn(N + 1, N + 1)
        A = A + A.conj().T  # Make Hermitian
        B = np.array([[1, 0], [0, -1]], dtype=complex)  # sigma_z
        C = pseudomode_number_operator(K)

        op = tripartite_operator(A, B, C, N, K)
        assert np.allclose(op, op.conj().T, atol=1e-10), (
            "Hermitian inputs should produce Hermitian output"
        )


# =============================================================================
# Test Hamiltonian Construction
# =============================================================================


class TestBuildPseudomodeHamiltonian:
    """Test Hamiltonian construction."""

    def test_shape(self) -> None:
        """Hamiltonian should have correct shape."""
        config = PseudomodeConfig(N=5, K=3)
        dim_total = 2 * (5 + 1) * (3 + 1)
        H = build_pseudomode_hamiltonian(config)
        assert H.shape == (dim_total, dim_total), (
            f"Expected ({dim_total}, {dim_total}), got {H.shape}"
        )

    def test_hermiticity(self) -> None:
        """Hamiltonian should be Hermitian."""
        config = PseudomodeConfig(N=5, K=3)
        H = build_pseudomode_hamiltonian(config)
        assert np.allclose(H, H.conj().T, atol=1e-10), "H is not Hermitian"

    def test_hermiticity_no_sa(self) -> None:
        """Decoherence Hamiltonian (no H_sa) should be Hermitian."""
        config = PseudomodeConfig(N=5, K=3)
        H = build_pseudomode_hamiltonian(config, include_sa=False)
        assert np.allclose(H, H.conj().T, atol=1e-10), "H_dec is not Hermitian"

    def test_zero_coupling(self) -> None:
        """Zero couplings should give zero (or identity-free) Hamiltonian."""
        config = PseudomodeConfig(N=5, K=3, g_sa=0.0, g_sp=0.0, omega_0=0.0)
        H = build_pseudomode_hamiltonian(config)
        assert np.allclose(H, 0, atol=1e-10), (
            "Zero couplings should give zero Hamiltonian"
        )

    def test_include_vs_exclude_sa_differ(self) -> None:
        """Hamiltonians with and without H_sa should differ."""
        config = PseudomodeConfig(N=5, K=3, g_sa=1.0)
        H_with = build_pseudomode_hamiltonian(config, include_sa=True)
        H_without = build_pseudomode_hamiltonian(config, include_sa=False)
        assert not np.allclose(H_with, H_without), (
            "H_sa inclusion should change the Hamiltonian"
        )


# =============================================================================
# Test Lindblad Operators
# =============================================================================


class TestBuildPseudomodeLindbladOperators:
    """Test Lindblad operator construction."""

    def test_no_dissipation_zero_lam(self) -> None:
        """Zero lambda should give empty lists."""
        config = PseudomodeConfig(N=5, K=3, lam=0.0)
        L_ops, gammas = build_pseudomode_lindblad_operators(config)
        assert len(L_ops) == 0
        assert len(gammas) == 0

    def test_no_dissipation_negative_lam(self) -> None:
        """Negative lambda should give empty lists."""
        config = PseudomodeConfig(N=5, K=3, lam=-0.1)
        L_ops, gammas = build_pseudomode_lindblad_operators(config)
        assert len(L_ops) == 0

    def test_dissipation_with_positive_lam(self) -> None:
        """Positive lambda should add one operator."""
        config = PseudomodeConfig(N=5, K=3, lam=1.0)
        L_ops, gammas = build_pseudomode_lindblad_operators(config)
        assert len(L_ops) == 1
        assert len(gammas) == 1
        assert gammas[0] == 1.0

    def correct_shape(self) -> None:
        """Lindblad operator should have correct shape."""
        config = PseudomodeConfig(N=5, K=3, lam=1.0)
        dim_total = 2 * (5 + 1) * (3 + 1)
        L_ops, _ = build_pseudomode_lindblad_operators(config)
        assert L_ops[0].shape == (dim_total, dim_total)

    def test_operator_structure(self) -> None:
        """L = √λ · I_osc ⊗ I_spin ⊗ b."""
        N, K, lam = 3, 2, 4.0
        config = PseudomodeConfig(N=N, K=K, lam=lam)
        b, _ = create_pseudomode_operators(K)
        I_osc = np.eye(N + 1)
        I_spin = np.eye(2)

        # Expected: sqrt(lam) * I_osc ⊗ I_spin ⊗ b
        expected = np.sqrt(lam) * np.kron(np.kron(I_osc, I_spin), b)

        L_ops, _ = build_pseudomode_lindblad_operators(config)
        assert np.allclose(L_ops[0], expected, atol=1e-10), (
            "Lindblad operator structure incorrect"
        )


# =============================================================================
# Test State Preparation
# =============================================================================


class TestPseudomodeInitialState:
    """Test initial state preparation."""

    def test_shape(self) -> None:
        """State should have correct shape."""
        config = PseudomodeConfig(N=5, K=3)
        dim_total = 2 * (5 + 1) * (3 + 1)
        state = pseudomode_initial_state(config)
        assert state.shape == (dim_total,), (
            f"Expected ({dim_total},), got {state.shape}"
        )

    def test_normalization(self) -> None:
        """State should be normalized (within truncation error)."""
        # Use N large enough for each alpha to minimise truncation
        test_cases = [(0.0, 5), (0.5, 10), (1.0, 10), (2.0, 25)]
        for alpha, N in test_cases:
            config = PseudomodeConfig(N=N, K=3, alpha=alpha)
            state = pseudomode_initial_state(config)
            norm = np.sum(np.abs(state) ** 2)
            # Allow ~1e-3 tolerance for truncation effects
            assert np.isclose(norm, 1.0, atol=1e-3), (
                f"alpha={alpha}, N={N}: norm = {norm}"
            )

    def test_spin_down(self) -> None:
        """Spin should be in |↓> (s=0)."""
        config = PseudomodeConfig(N=5, K=3)
        state = pseudomode_initial_state(config)
        dim_pm = config.K + 1

        # Check that only s=0 components are non-zero
        for n in range(config.N + 1):
            for k in range(dim_pm):
                idx_up = (n * 2 + 1) * dim_pm + k
                assert np.abs(state[idx_up]) == 0, (
                    f"Spin up at n={n},k={k} should be zero"
                )

    def test_pm_vacuum(self) -> None:
        """Pseudomode should be in |0> (k=0)."""
        config = PseudomodeConfig(N=5, K=3)
        state = pseudomode_initial_state(config)
        dim_pm = config.K + 1

        # Check that only k=0 components are non-zero
        for n in range(config.N + 1):
            for s in range(2):
                for k in range(1, dim_pm):
                    idx = (n * 2 + s) * dim_pm + k
                    assert np.abs(state[idx]) == 0, (
                        f"PM k={k} at n={n},s={s} should be zero"
                    )

    def test_coherent_amplitudes(self) -> None:
        """Oscillator coherent state amplitudes should be correct."""
        alpha = 1.0
        config = PseudomodeConfig(N=10, K=3, alpha=alpha)
        state = pseudomode_initial_state(config)

        dim_pm = config.K + 1
        for n in range(config.N + 1):
            expected = (
                alpha**n
                / np.sqrt(scipy.special.factorial(n))
                * np.exp(-(np.abs(alpha) ** 2) / 2)
            )
            idx = n * 2 * dim_pm  # s=0, k=0
            assert np.isclose(state[idx], expected, atol=1e-10), (
                f"n={n}: amplitude mismatch"
            )


# =============================================================================
# Test Ancilla Entanglement
# =============================================================================


class TestApplyAncillaEntanglement:
    """Test ancilla entanglement unitary."""

    def test_norm_preservation(self) -> None:
        """Unitary evolution should preserve norm."""
        config = PseudomodeConfig(N=5, K=3, g_sa=1.0, tau=0.2)
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)

        norm_before = np.sum(np.abs(state) ** 2)
        norm_after = np.sum(np.abs(entangled) ** 2)
        assert np.isclose(norm_before, norm_after, atol=1e-10), (
            "Norm not preserved by entanglement unitary"
        )

    def test_zero_tau_identity(self) -> None:
        """Zero tau should return the same state."""
        config = PseudomodeConfig(N=5, K=3, g_sa=1.0, tau=0.0)
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)

        assert np.allclose(entangled, state, atol=1e-10), (
            "Zero tau should give identity evolution"
        )

    def test_zero_coupling_identity(self) -> None:
        """Zero coupling should return the same state."""
        config = PseudomodeConfig(N=5, K=3, g_sa=0.0, tau=0.2)
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)

        assert np.allclose(entangled, state, atol=1e-10), (
            "Zero g_sa should give identity evolution"
        )

    def test_generates_spin_population(self) -> None:
        """Entanglement should create spin-up population from spin-down."""
        config = PseudomodeConfig(N=5, K=3, g_sa=2.0, tau=0.5)
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)

        dim_pm = config.K + 1
        # Check that some spin-up (s=1) components are non-zero
        has_spin_up = False
        for n in range(config.N + 1):
            for k in range(dim_pm):
                idx_up = (n * 2 + 1) * dim_pm + k
                if np.abs(entangled[idx_up]) > 1e-6:
                    has_spin_up = True
                    break

        assert has_spin_up, "Ancilla entanglement should populate spin-up states"

    def test_wrong_dimension_raises(self) -> None:
        """Wrong state dimension should raise AssertionError."""
        config = PseudomodeConfig(N=5, K=3)
        bad_state = np.zeros(10, dtype=complex)
        with pytest.raises(AssertionError):
            apply_ancilla_entanglement(bad_state, config)


# =============================================================================
# Test Partial Trace
# =============================================================================


class TestPartialTrace:
    """Test partial trace operations."""

    def _make_test_density(self, N: int, K: int) -> np.ndarray:
        """Create a random valid density matrix for testing."""
        dim = 2 * (N + 1) * (K + 1)
        # Random pure state
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi = psi / np.linalg.norm(psi)
        return np.outer(psi, psi.conj())

    def test_trace_out_pm_shape(self) -> None:
        """Trace out pseudomode should reduce dimension."""
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        rho_os = trace_out_pseudomode(rho, N, K)
        expected_dim = 2 * (N + 1)
        assert rho_os.shape == (expected_dim, expected_dim), (
            f"Expected ({expected_dim}, {expected_dim}), got {rho_os.shape}"
        )

    def test_trace_out_spin_shape(self) -> None:
        """Trace out spin should reduce dimension."""
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        rho_op = trace_out_spin(rho, N, K)
        expected_dim = (N + 1) * (K + 1)
        assert rho_op.shape == (expected_dim, expected_dim), (
            f"Expected ({expected_dim}, {expected_dim}), got {rho_op.shape}"
        )

    def test_trace_out_both_shape(self) -> None:
        """Trace out both should give oscillator-only density."""
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        rho_osc = trace_out_spin_and_pseudomode(rho, N, K)
        expected_dim = N + 1
        assert rho_osc.shape == (expected_dim, expected_dim), (
            f"Expected ({expected_dim}, {expected_dim}), got {rho_osc.shape}"
        )

    def test_trace_equality(self) -> None:
        """Double trace should equal single combined trace."""
        N, K = 5, 3
        rho = self._make_test_density(N, K)

        # Method 1: trace spin, then trace pm
        rho_op = trace_out_spin(rho, N, K)
        rho_osc_a = np.trace(
            rho_op.reshape(N + 1, K + 1, N + 1, K + 1),
            axis1=1,
            axis2=3,
        )

        # Method 2: combined trace
        rho_osc_b = trace_out_spin_and_pseudomode(rho, N, K)

        assert np.allclose(rho_osc_a, rho_osc_b, atol=1e-10), (
            "Combined trace should match sequential traces"
        )

    def test_trace_conservation(self) -> None:
        """Partial trace should conserve total trace."""
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        original_trace = np.trace(rho)

        rho_os = trace_out_pseudomode(rho, N, K)
        assert np.isclose(np.trace(rho_os), original_trace, atol=1e-10), (
            "Trace not conserved after tracing out pseudomode"
        )

    def test_hermiticity_preserved(self) -> None:
        """Partial trace should preserve Hermiticity."""
        N, K = 5, 3
        rho = self._make_test_density(N, K)
        rho_os = trace_out_pseudomode(rho, N, K)
        assert np.allclose(rho_os, rho_os.conj().T, atol=1e-10), (
            "Reduced density not Hermitian"
        )


# =============================================================================
# Test Lindblad Evolution
# =============================================================================


class TestEvolvePseudomode:
    """Test Lindblad evolution."""

    def test_zero_time(self) -> None:
        """Zero evolution time should return initial state."""
        config = PseudomodeConfig(N=5, K=3, T=0.0, lam=0.5)
        psi = pseudomode_initial_state(config)
        rho = evolve_pseudomode(psi, config, method="rk4")
        rho_expected = np.outer(psi, psi.conj())
        assert np.allclose(rho, rho_expected, atol=1e-6), (
            "Zero time should return initial state"
        )

    def test_unitary_evolution_no_lam(self) -> None:
        """With no dissipation (lam=0), should match unitary evolution."""
        config = PseudomodeConfig(N=5, K=3, lam=0.0, g_sp=0.5, T=0.5)
        psi = pseudomode_initial_state(config)

        # Analytical unitary
        H = build_pseudomode_hamiltonian(config, include_sa=False)
        U = scipy.linalg.expm(-1.0j * H * config.T)
        rho_expected = U @ np.outer(psi, psi.conj()) @ U.conj().T

        rho_final = evolve_pseudomode(psi, config, method="rk4")

        assert np.allclose(rho_final, rho_expected, atol=1e-4), (
            "Unitary evolution mismatch"
        )

    def test_trace_preservation(self) -> None:
        """Trace should be preserved (Lindblad is trace-preserving)."""
        config = PseudomodeConfig(N=5, K=3, lam=0.5, g_sp=0.3, T=0.5)
        psi = pseudomode_initial_state(config)
        rho_final = evolve_pseudomode(psi, config, method="rk4")
        trace = np.trace(rho_final)
        assert np.isclose(trace, 1.0, atol=1e-6), f"Trace should be 1, got {trace}"

    def test_hermiticity(self) -> None:
        """Density matrix should remain Hermitian."""
        config = PseudomodeConfig(N=5, K=3, lam=0.5, g_sp=0.3, T=0.5)
        psi = pseudomode_initial_state(config)
        rho_final = evolve_pseudomode(psi, config, method="rk4")
        assert np.allclose(rho_final, rho_final.conj().T, atol=1e-6), (
            "Final state not Hermitian"
        )

    def test_positivity(self) -> None:
        """Eigenvalues should be non-negative."""
        config = PseudomodeConfig(N=5, K=3, lam=0.5, g_sp=0.3, T=0.5)
        psi = pseudomode_initial_state(config)
        rho_final = evolve_pseudomode(psi, config, method="rk4")
        eigenvalues = np.linalg.eigvalsh(rho_final)
        min_ev = np.min(eigenvalues.real)
        assert min_ev >= -1e-6, f"Negative eigenvalue: {min_ev}"

    def test_scipy_method_matches_rk4(self) -> None:
        """Scipy solver should give similar results to RK4."""
        config = PseudomodeConfig(N=4, K=2, lam=0.3, g_sp=0.2, T=0.2)
        psi = pseudomode_initial_state(config)

        rho_rk4 = evolve_pseudomode(psi, config, method="rk4")
        rho_scipy = evolve_pseudomode(psi, config, method="scipy")

        assert np.allclose(rho_rk4, rho_scipy, atol=1e-4), (
            "RK4 and scipy solvers disagree"
        )

    def test_unknown_method_raises(self) -> None:
        """Unknown method should raise ValueError."""
        config = PseudomodeConfig(N=5, K=3)
        psi = pseudomode_initial_state(config)
        with pytest.raises(ValueError):
            evolve_pseudomode(psi, config, method="invalid")


# =============================================================================
# Test QFI Computation
# =============================================================================


class TestQFIComputation:
    """Test QFI computation for the tripartite system."""

    def test_qfi_initial_nonzero(self) -> None:
        """Initial QFI should be nonzero for alpha > 0."""
        config = PseudomodeConfig(N=10, K=3, alpha=1.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        psi = pseudomode_initial_state(config)
        rho = np.outer(psi, psi.conj())

        qfi = compute_qfi_with_ancilla(rho, config.N, config.K)
        assert qfi > 0, f"Initial QFI should be positive, got {qfi}"

    def test_qfi_coherent_state_formula(self) -> None:
        """For |α>|↓>, QFI = 4|α|² with generator a†a ⊗ I_spin.

        For a coherent state |α>, Var(a†a) = |α|², so F_Q = 4|α|².
        Uses large N to minimise truncation effects on variance.
        """
        config = PseudomodeConfig(N=30, K=3, alpha=2.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        psi = pseudomode_initial_state(config)
        rho = np.outer(psi, psi.conj())

        qfi = compute_qfi_with_ancilla(rho, config.N, config.K)
        expected = 4.0 * config.alpha**2
        assert np.isclose(qfi, expected, rtol=1e-3), f"QFI {qfi} != 4|α|² = {expected}"

    def test_qfi_zero_alpha(self) -> None:
        """For |0>|↓>, QFI should be 0 (vacuum has no phase info)."""
        config = PseudomodeConfig(N=5, K=3, alpha=0.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        psi = pseudomode_initial_state(config)
        rho = np.outer(psi, psi.conj())

        qfi = compute_qfi_with_ancilla(rho, config.N, config.K)
        assert np.isclose(qfi, 0.0, atol=1e-10), (
            f"QFI for vacuum should be 0, got {qfi}"
        )

    def test_qfi_without_ancilla_equals_with_at_zero_gsa(self) -> None:
        """With g_sa=0, QFI with and without ancilla should match.

        When there's no ancilla entanglement, tracing out spin has no effect.
        """
        config = PseudomodeConfig(N=10, K=3, alpha=1.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        psi = pseudomode_initial_state(config)
        rho = np.outer(psi, psi.conj())

        qfi_with = compute_qfi_with_ancilla(rho, config.N, config.K)
        qfi_without = compute_qfi_without_ancilla(rho, config.N, config.K)

        assert np.isclose(qfi_with, qfi_without, atol=1e-10), (
            "With g_sa=0, ancilla should not change QFI"
        )

    def test_qfi_with_and_without_differ_with_gsa(self) -> None:
        """With g_sa > 0, QFI with ancilla may differ from without.

        After ancilla entanglement, spin carries phase information,
        so tracing it out should reduce QFI (or leave it unchanged).
        """
        config = PseudomodeConfig(
            N=10, K=3, alpha=1.0, g_sa=2.0, tau=0.3, g_sp=0.0, lam=0.0, T=0.0
        )
        # Just initial state (no evolution) — but we entangle
        psi = pseudomode_initial_state(config)
        psi_ent = apply_ancilla_entanglement(psi, config)
        rho = np.outer(psi_ent, psi_ent.conj())

        qfi_with = compute_qfi_with_ancilla(rho, config.N, config.K)
        qfi_without = compute_qfi_without_ancilla(rho, config.N, config.K)

        # With ancilla should have at least as much QFI
        assert qfi_with >= qfi_without - 1e-10, (
            "QFI with ancilla should be >= QFI without ancilla"
        )


# =============================================================================
# Test Metrology Protocol
# =============================================================================


class TestRunMetrologyProtocol:
    """Test full metrology protocol."""

    def test_protocol_completes(self) -> None:
        """Protocol should run without errors."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=1.0, tau=0.2, g_sp=0.3, lam=0.5, T=0.5
        )
        result = run_metrology_protocol(config)

        assert "rho_final" in result
        assert "qfi_with" in result
        assert "qfi_without" in result
        assert "qfi_initial" in result
        assert "ratio_with" in result
        assert "ratio_without" in result
        assert "pm_occupancy" in result
        assert "validation" in result

    def test_ratio_at_least_zero(self) -> None:
        """Preservation ratios should be between 0 and 1."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.3, lam=0.5, T=0.5
        )
        result = run_metrology_protocol(config)

        assert 0 <= result["ratio_with"] <= 1.0 + 1e-6, (
            f"ratio_with = {result['ratio_with']} outside [0, 1]"
        )
        assert 0 <= result["ratio_without"] <= 1.0 + 1e-6, (
            f"ratio_without = {result['ratio_without']} outside [0, 1]"
        )

    def test_qfi_decreases_with_time(self) -> None:
        """QFI should decrease (or stay same) with decoherence."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.3, lam=0.5, T=0.5
        )
        result = run_metrology_protocol(config)

        assert result["qfi_with"] <= result["qfi_initial"] + 1e-6, (
            "QFI should not increase under decoherence"
        )

    def test_validation_passes(self) -> None:
        """Final density matrix should pass all checks."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.3, lam=0.5, T=0.5
        )
        result = run_metrology_protocol(config)

        validation = result["validation"]
        assert validation["is_hermitian"], "Final state not Hermitian"
        assert validation["is_normalized"], "Final state not normalized"
        assert validation["is_positive"], "Final state not positive"

    def test_ratio_with_geq_without(self) -> None:
        """With ancilla should preserve at least as much QFI as without."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=2.0, tau=0.3, g_sp=0.3, lam=0.5, T=0.5
        )
        result = run_metrology_protocol(config)

        # Ancilla should help (or at least not hurt)
        assert result["ratio_with"] >= result["ratio_without"] - 1e-6, (
            "Ancilla should not reduce QFI preservation"
        )


# =============================================================================
# Test Validation
# =============================================================================


class TestValidatePseudomodeDensity:
    """Test density matrix validation."""

    def test_valid_density(self) -> None:
        """Valid density matrix should pass all checks."""
        dim = 2 * (5 + 1) * (3 + 1)
        rho = np.eye(dim, dtype=complex) / dim  # Maximally mixed

        result = validate_pseudomode_density(rho)
        assert result["is_hermitian"]
        assert result["is_normalized"]
        assert result["is_positive"]

    def test_pure_state(self) -> None:
        """Pure state should be valid (allow truncation tolerance)."""
        # Use large N to minimise coherent state truncation
        config = PseudomodeConfig(N=20, K=3, alpha=1.0)
        psi = pseudomode_initial_state(config)
        rho = np.outer(psi, psi.conj())

        result = validate_pseudomode_density(rho, tolerance=1e-4)
        assert result["is_hermitian"]
        assert result["is_normalized"]
        assert result["is_positive"]

    def test_non_hermitian_fails(self) -> None:
        """Non-Hermitian matrix should fail."""
        dim = 10
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 1] = 1.0  # Not Hermitian

        result = validate_pseudomode_density(rho)
        assert not result["is_hermitian"]

    def test_not_normalized_fails(self) -> None:
        """Non-normalized matrix should fail."""
        dim = 10
        rho = np.eye(dim, dtype=complex) / dim + 0.1 * np.eye(dim, dtype=complex)

        result = validate_pseudomode_density(rho)
        assert not result["is_normalized"]


# =============================================================================
# Test Pseudomode Occupancy
# =============================================================================


class TestCheckPseudomodeOccupancy:
    """Test pseudomode truncation check."""

    def test_vacuum_occupancy_zero(self) -> None:
        """Vacuum pseudomode should have zero occupancy."""
        config = PseudomodeConfig(N=5, K=3)
        psi = pseudomode_initial_state(config)
        rho = np.outer(psi, psi.conj())

        occ, is_safe = check_pseudomode_occupancy(rho, config.N, config.K)
        assert np.isclose(occ, 0.0, atol=1e-10), (
            f"Occupancy should be 0 for vacuum, got {occ}"
        )
        assert is_safe, "Vacuum should be safe"

    def test_vacuum_is_safe(self) -> None:
        """Vacuum should always be safe (occupancy << 0.8*K)."""
        for K in [1, 3, 5, 10]:
            config = PseudomodeConfig(N=5, K=K)
            psi = pseudomode_initial_state(config)
            rho = np.outer(psi, psi.conj())

            _, is_safe = check_pseudomode_occupancy(rho, config.N, config.K)
            assert is_safe, f"K={K}: vacuum should be safe"


# =============================================================================
# Test QFI Preservation Ratio
# =============================================================================


class TestQFIPreservationRatio:
    """Test QFI preservation ratio computation."""

    def test_ratio_at_T0(self) -> None:
        """At T=0, preservation ratio should be 1."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.0, lam=0.0, T=0.0
        )
        psi = pseudomode_initial_state(config)
        psi_ent = apply_ancilla_entanglement(psi, config)
        rho = np.outer(psi_ent, psi_ent.conj())

        fq_initial = compute_qfi_with_ancilla(rho, config.N, config.K)
        ratio = qfi_preservation_ratio(rho, fq_initial, config.N, config.K)

        assert np.isclose(ratio, 1.0, atol=1e-6), (
            f"At T=0, ratio should be 1, got {ratio}"
        )

    def test_ratio_zero_fq_initial(self) -> None:
        """Zero initial QFI should give ratio 0."""
        config = PseudomodeConfig(N=5, K=3, alpha=0.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        psi = pseudomode_initial_state(config)
        rho = np.outer(psi, psi.conj())

        ratio = qfi_preservation_ratio(rho, 0.0, config.N, config.K)
        assert ratio == 0.0, f"Zero initial QFI should give 0 ratio, got {ratio}"

    def test_with_and_without_ratio(self) -> None:
        """Test both with_ancilla=True and with_ancilla=False."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.0, lam=0.0, T=0.0
        )
        psi = pseudomode_initial_state(config)
        psi_ent = apply_ancilla_entanglement(psi, config)
        rho = np.outer(psi_ent, psi_ent.conj())

        fq_initial = compute_qfi_with_ancilla(rho, config.N, config.K)

        ratio_with = qfi_preservation_ratio(
            rho, fq_initial, config.N, config.K, with_ancilla=True
        )
        ratio_without = qfi_preservation_ratio(
            rho, fq_initial, config.N, config.K, with_ancilla=False
        )

        assert np.isclose(ratio_with, 1.0, atol=1e-6)
        assert ratio_without >= 0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_small_N_and_K(self) -> None:
        """Should work with N=1, K=1."""
        config = PseudomodeConfig(
            N=1, K=1, alpha=0.5, g_sa=0.5, g_sp=0.1, lam=0.1, T=0.1
        )
        result = run_metrology_protocol(config)
        validation = result["validation"]
        assert validation["is_normalized"], "Small system should preserve trace"

    def test_zero_g_sp(self) -> None:
        """Zero system-pseudomode coupling should give no decoherence."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.0, lam=1.0, T=0.5
        )
        result = run_metrology_protocol(config)
        # Without system-pm coupling, pseudomode is isolated so no decoherence
        assert np.isclose(result["ratio_with"], 1.0, atol=1e-4), (
            "Zero g_sp should give no decoherence"
        )

    def test_zero_lam_produces_valid_result(self) -> None:
        """Zero lambda (no dissipation) should produce a valid result.

        Note: With λ=0 the dynamics are fully coherent (reversible), and the
        QFI may oscillate rather than decay monotonically (see article §
        ``Likely Failure Conditions'', condition 5).  It is NOT always true
        that λ=0 preserves more QFI than a finite λ at a given time T.
        """
        config = PseudomodeConfig(
            N=15, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.3, lam=0.0, T=0.5
        )
        result = run_metrology_protocol(config)

        assert result["validation"]["is_normalized"], "Zero lam should preserve trace"
        assert result["validation"]["is_hermitian"], (
            "Zero lam should produce Hermitian state"
        )
        assert result["ratio_with"] >= 0, "QFI ratio should be non-negative"
        assert result["qfi_with"] <= result["qfi_initial"] + 1e-6, (
            "QFI should not increase under any dynamics"
        )

    def test_high_alpha_needs_larger_N(self) -> None:
        """Large alpha should work with sufficient N.

        For |α|² = 9, the Poisson tail extends to n ≈ 9 + 5√10 ≈ 25.
        """
        config = PseudomodeConfig(N=40, K=5, alpha=3.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        psi = pseudomode_initial_state(config)
        norm = np.sum(np.abs(psi) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-6), f"Large alpha state norm = {norm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

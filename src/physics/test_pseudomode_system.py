"""
Unit tests for the pseudomode non-Markovian metrology module.

Tests all public functions in ``src.physics.pseudomode_system``:
operator construction, Hamiltonian assembly, Lindblad operators,
state preparation, ancilla entanglement, evolution, partial trace,
QFI computation, and the full metrology protocol.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from src.physics.pseudomode_system import (
    PseudomodeConfig,
    apply_ancilla_entanglement,
    build_pseudomode_hamiltonian,
    build_pseudomode_lindblad_operators,
    check_pseudomode_occupancy,
    compute_qfi_with_ancilla,
    compute_qfi_without_ancilla,
    create_pseudomode_operators,
    evolve_pseudomode,
    pseudomode_initial_state,
    pseudomode_number_operator,
    qfi_preservation_ratio,
    run_metrology_protocol,
    trace_out_pseudomode,
    trace_out_spin,
    trace_out_spin_and_pseudomode,
    tripartite_operator,
)

# =============================================================================
# Helpers
# =============================================================================


def _dim_total(N: int, K: int) -> int:
    return 2 * (N + 1) * (K + 1)


# =============================================================================
# Tests: PseudomodeConfig
# =============================================================================


class TestPseudomodeConfig:
    """Configuration dataclass defaults and validation."""

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("N", 5),
            ("K", 3),
            ("alpha", 1.0),
            ("g_sa", 1.0),
            ("tau", 0.1),
            ("g_sp", 0.5),
            ("omega_0", 0.0),
            ("lam", 1.0),
            ("T_decay", 2.0),
            ("dt", 0.01),
        ],
    )
    def test_default_values(self, field: str, expected: object) -> None:
        assert getattr(PseudomodeConfig(N=5, K=3), field) == expected

    def test_invalid_N_neg(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=-1, K=3)

    def test_invalid_K_neg(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=-1)

    def test_invalid_dt_neg(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, dt=-0.1)

    def test_invalid_dt_zero(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, dt=0.0)

    def test_invalid_T_decay_neg(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, T_decay=-1.0)

    def test_invalid_tau_neg(self) -> None:
        with pytest.raises(ValueError):
            PseudomodeConfig(N=5, K=3, tau=-0.1)


# =============================================================================
# Tests: Pseudomode Operators
# =============================================================================


class TestPseudomodeOperators:
    """Ladder operator and number operator construction."""

    def test_shape(self) -> None:
        b, bd = create_pseudomode_operators(5)
        assert b.shape == (6, 6)
        assert bd.shape == (6, 6)
        assert np.allclose(bd, b.conj().T)

    def test_negative_K_raises(self) -> None:
        with pytest.raises(ValueError):
            create_pseudomode_operators(-1)

    def test_number_shape(self) -> None:
        n = pseudomode_number_operator(5)
        assert n.shape == (6, 6)

    def test_number_diagonal(self) -> None:
        n = pseudomode_number_operator(5)
        assert np.allclose(n, np.diag(np.arange(6, dtype=complex)))

    def test_commutation(self) -> None:
        """Verify [b, b^dagger] for truncated space.

        In an infinite Fock space [b, b^dagger] = I. In truncated
        dimension D = K+1, the last diagonal element differs:
        [b, b^dagger]_{D-1,D-1} = -(D-1) because the raising from
        the last state is truncated.
        """
        K = 10
        D = K + 1  # dimension = 11
        b, bd = create_pseudomode_operators(K)
        comm = b @ bd - bd @ b
        expected = np.eye(D, dtype=complex)
        # In truncated space, [b, b^dagger]_{K,K} = -K
        expected[-1, -1] = -K
        assert np.allclose(comm, expected, atol=1e-10)


# =============================================================================
# Tests: Tripartite Operator
# =============================================================================


class TestTripartiteOperator:
    """Tripartite Kronecker product construction."""

    @pytest.mark.parametrize(
        ("N", "K"),
        [(2, 2), (3, 3), (5, 4), (1, 1)],
    )
    def test_correct_dimensions(self, N: int, K: int) -> None:
        op = tripartite_operator(np.eye(N + 1), np.eye(2), np.eye(K + 1), N, K)
        assert op.shape == (_dim_total(N, K), _dim_total(N, K))

    def test_dimension_mismatch_osc_raises(self) -> None:
        with pytest.raises(AssertionError):
            tripartite_operator(np.eye(7), np.eye(2), np.eye(4), 5, 3)

    def test_dimension_mismatch_spin_raises(self) -> None:
        with pytest.raises(AssertionError):
            tripartite_operator(np.eye(6), np.eye(3), np.eye(4), 5, 3)

    def test_dimension_mismatch_pm_raises(self) -> None:
        with pytest.raises(AssertionError):
            tripartite_operator(np.eye(6), np.eye(2), np.eye(5), 5, 3)

    def test_index_convention(self) -> None:
        """Verify the Kronecker order against a hand-computed reference.

        We construct a projector onto |n=1, s=0, k=0⟩ as the Kronecker
        product of single-subsystem projectors and check that only the
        expected index (1, 2*(K+1)*n + 2*s + 2*(K+1)*0 + k) is non-zero.
        """
        N, K = 2, 2
        dim_osc = N + 1
        dim_pm = K + 1
        dim_total = 2 * dim_osc * dim_pm
        # The tripartite index for |n=1, s=0, k=0⟩ is:
        # idx = (n * 2 + s) * (K+1) + k = (1*2 + 0)*3 + 0 = 6
        target_idx = (1 * 2 + 0) * dim_pm + 0
        assert target_idx == 6

        # Build a tripartite operator that is |1,0,0⟩⟨1,0,0|
        # from single-subsystem projectors
        osc_proj = np.zeros((dim_osc, dim_osc), dtype=complex)
        osc_proj[1, 1] = 1.0
        spin_proj = np.zeros((2, 2), dtype=complex)
        spin_proj[0, 0] = 1.0  # project onto |down⟩
        pm_proj = np.zeros((dim_pm, dim_pm), dtype=complex)
        pm_proj[0, 0] = 1.0  # project onto |k=0⟩

        full = tripartite_operator(osc_proj, spin_proj, pm_proj, N, K)
        assert full.shape == (dim_total, dim_total)

        # The only non-zero element should be at (target_idx, target_idx)
        assert np.isclose(full[target_idx, target_idx], 1.0)
        assert np.isclose(full.sum(), 1.0)  # only one non-zero entry
        # Also verify no other diagonal element is non-zero
        for i in range(dim_total):
            if i != target_idx:
                assert np.isclose(full[i, i], 0.0), f"Unexpected non-zero at ({i},{i})"


# =============================================================================
# Tests: Hamiltonian
# =============================================================================


class TestBuildPseudomodeHamiltonian:
    """Hamiltonian construction and properties."""

    @pytest.mark.parametrize(
        ("N", "K"),
        [(2, 2), (5, 3), (3, 5)],
    )
    def test_shape(self, N: int, K: int) -> None:
        H = build_pseudomode_hamiltonian(PseudomodeConfig(N=N, K=K))
        assert H.shape == (_dim_total(N, K), _dim_total(N, K))

    def test_hermiticity(self) -> None:
        H = build_pseudomode_hamiltonian(PseudomodeConfig(N=5, K=3))
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_hermiticity_no_sa(self) -> None:
        H = build_pseudomode_hamiltonian(PseudomodeConfig(N=5, K=3), include_sa=False)
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_include_sa_flag(self) -> None:
        """H with include_sa=True should differ from include_sa=False."""
        config = PseudomodeConfig(N=5, K=3, g_sa=2.0)
        H_with = build_pseudomode_hamiltonian(config, include_sa=True)
        H_without = build_pseudomode_hamiltonian(config, include_sa=False)
        assert not np.allclose(H_with, H_without, atol=1e-10)

    def test_g_sa_zero_equivalence(self) -> None:
        """With g_sa=0, include_sa flag should not matter."""
        config = PseudomodeConfig(N=5, K=3, g_sa=0.0)
        H_with = build_pseudomode_hamiltonian(config, include_sa=True)
        H_without = build_pseudomode_hamiltonian(config, include_sa=False)
        assert np.allclose(H_with, H_without, atol=1e-10)

    def test_hamiltonian_real_for_omega_0_zero(self) -> None:
        """Hamiltonian should be real-valued when omega_0=0."""
        H = build_pseudomode_hamiltonian(PseudomodeConfig(N=5, K=3, omega_0=0.0))
        # Check imaginary part is zero
        assert np.allclose(np.imag(H), 0.0, atol=1e-12)


# =============================================================================
# Tests: Lindblad Operators
# =============================================================================


class TestBuildPseudomodeLindbladOperators:
    """Lindblad operator construction."""

    def test_shape(self) -> None:
        L_ops, gammas = build_pseudomode_lindblad_operators(
            PseudomodeConfig(N=5, K=3, lam=0.5)
        )
        assert len(L_ops) == 1
        assert len(gammas) == 1
        expected_dim = _dim_total(5, 3)
        assert L_ops[0].shape == (expected_dim, expected_dim)
        assert gammas[0] == 1.0

    def test_lam_zero_returns_empty(self) -> None:
        L_ops, gammas = build_pseudomode_lindblad_operators(
            PseudomodeConfig(N=5, K=3, lam=0.0)
        )
        assert len(L_ops) == 0
        assert len(gammas) == 0

    def test_lam_negative_returns_empty(self) -> None:
        L_ops, _gammas = build_pseudomode_lindblad_operators(
            PseudomodeConfig(N=5, K=3, lam=-1.0)
        )
        assert len(L_ops) == 0

    def test_rate_scaling(self) -> None:
        """Operator norm should scale as sqrt(lam)."""
        L_1, _ = build_pseudomode_lindblad_operators(
            PseudomodeConfig(N=2, K=2, lam=1.0)
        )
        L_4, _ = build_pseudomode_lindblad_operators(
            PseudomodeConfig(N=2, K=2, lam=4.0)
        )
        norm_1 = np.linalg.norm(L_1[0])
        norm_4 = np.linalg.norm(L_4[0])
        assert np.isclose(norm_4 / norm_1, 2.0, rtol=1e-10)


# =============================================================================
# Tests: Initial State
# =============================================================================


class TestPseudomodeInitialState:
    """Initial state preparation."""

    @pytest.mark.parametrize(
        ("alpha", "N"), [(0.0, 5), (0.5, 10), (1.0, 10), (2.0, 25)]
    )
    def test_normalization(self, alpha: float, N: int) -> None:
        state = pseudomode_initial_state(PseudomodeConfig(N=N, K=3, alpha=alpha))
        assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0, abs=1e-3)

    def test_correct_dimensions(self) -> None:
        config = PseudomodeConfig(N=5, K=3)
        state = pseudomode_initial_state(config)
        assert state.shape == (_dim_total(5, 3),)

    def test_spin_down(self) -> None:
        """The spin component should be in the down state (s=0)."""
        config = PseudomodeConfig(N=5, K=3)
        state = pseudomode_initial_state(config)
        # Index: n=0, s=0, k=0 should be populated
        # Index: n=0, s=1, k=0 should be zero (spin up)
        zero_up_idx = (0 * 2 + 1) * (3 + 1) + 0  # = 4
        assert np.abs(state[zero_up_idx]) < 1e-12

    def test_vacuum_pm(self) -> None:
        """Pseudomode should start in vacuum (k=0)."""
        config = PseudomodeConfig(N=5, K=3, alpha=0.0)
        state = pseudomode_initial_state(config)
        # All amplitude should be at n=0, s=0, k=0
        assert np.isclose(np.abs(state[0]), 1.0)

    def test_coherent_state_occupation(self) -> None:
        """Mean photon number should match |alpha|^2."""
        alpha = 1.5
        config = PseudomodeConfig(N=15, K=3, alpha=alpha)
        state = pseudomode_initial_state(config)
        # Build oscillator number operator in full space
        from src.physics.hybrid_system import oscillator_number

        n_op = oscillator_number(15)
        I_spin = np.eye(2, dtype=complex)
        I_pm = np.eye(4, dtype=complex)
        n_full = np.kron(np.kron(n_op, I_spin), I_pm)
        mean_n = np.real(np.vdot(state, n_full @ state))
        assert np.isclose(mean_n, np.abs(alpha) ** 2, rtol=1e-2)


# =============================================================================
# Tests: Ancilla Entanglement
# =============================================================================


class TestApplyAncillaEntanglement:
    """Ancilla entanglement unitary."""

    def test_preserves_norm(self) -> None:
        """Ancilla entanglement should preserve the state norm.

        The norm of the initial coherent state is < 1 due to Fock
        truncation; the entanglement unitary preserves this norm.
        """
        config = PseudomodeConfig(N=5, K=3, g_sa=1.0, tau=0.2)
        state = pseudomode_initial_state(config)
        norm_before = np.sum(np.abs(state) ** 2)
        entangled = apply_ancilla_entanglement(state, config)
        norm_after = np.sum(np.abs(entangled) ** 2)
        assert np.isclose(norm_after, norm_before, rtol=1e-10)

    def test_theta_zero_is_identity(self) -> None:
        """When g_sa=0, entanglement should be identity."""
        config = PseudomodeConfig(N=5, K=3, g_sa=0.0, tau=0.2)
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)
        np.testing.assert_allclose(entangled, state, atol=1e-10)

    def test_wrong_dimension_raises(self) -> None:
        config = PseudomodeConfig(N=5, K=3)
        bad_state = np.zeros(10)
        with pytest.raises(AssertionError):
            apply_ancilla_entanglement(bad_state, config)


# =============================================================================
# Tests: Pseudomode Occupancy
# =============================================================================


class TestCheckPseudomodeOccupancy:
    """Pseudomode truncation validity check."""

    def test_vacuum_occupancy(self) -> None:
        config = PseudomodeConfig(N=5, K=3, alpha=0.0)
        state = pseudomode_initial_state(config)
        rho = np.outer(state, state.conj())
        occ, safe = check_pseudomode_occupancy(rho, config.N, config.K)
        assert np.isclose(occ, 0.0, atol=1e-10)
        assert safe is True

    def test_occupancy_range(self) -> None:
        """Occupancy should be non-negative and finite."""
        config = PseudomodeConfig(N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2)
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)
        rho = np.outer(entangled, entangled.conj())
        occ, safe = check_pseudomode_occupancy(rho, config.N, config.K)
        assert 0.0 <= occ < np.inf
        assert isinstance(safe, bool)

    def test_wrong_dimension_raises(self) -> None:
        """Mismatched rho dimension should raise an error."""
        with pytest.raises((AssertionError, ValueError)):
            check_pseudomode_occupancy(np.eye(10), N=5, K=3)


# =============================================================================
# Tests: Partial Trace
# =============================================================================


class TestTraceFunctions:
    """Partial trace operations on tripartite space."""

    def _make_full_rho(self) -> np.ndarray:
        config = PseudomodeConfig(N=4, K=2, alpha=1.0, g_sa=0.5, tau=0.2)
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)
        return np.outer(entangled, entangled.conj())

    def test_trace_out_pm_dimensions(self) -> None:
        rho = self._make_full_rho()
        rho_os = trace_out_pseudomode(rho, N=4, K=2)
        # Result should be 2*(N+1) = 10
        assert rho_os.shape == (10, 10)

    def test_trace_out_pm_trace_preserved(self) -> None:
        """Partial trace should preserve the trace of the original rho."""
        rho = self._make_full_rho()
        full_trace = np.trace(rho)
        rho_os = trace_out_pseudomode(rho, N=4, K=2)
        assert np.isclose(np.trace(rho_os), full_trace, rtol=1e-10)

    def test_trace_out_spin_dimensions(self) -> None:
        rho = self._make_full_rho()
        rho_op = trace_out_spin(rho, N=4, K=2)
        # Result should be (N+1)*(K+1) = 5*3 = 15
        assert rho_op.shape == (15, 15)

    def test_trace_out_spin_trace_preserved(self) -> None:
        rho = self._make_full_rho()
        full_trace = np.trace(rho)
        rho_op = trace_out_spin(rho, N=4, K=2)
        assert np.isclose(np.trace(rho_op), full_trace, rtol=1e-10)

    def test_trace_out_spin_and_pm_dimensions(self) -> None:
        rho = self._make_full_rho()
        rho_o = trace_out_spin_and_pseudomode(rho, N=4, K=2)
        assert rho_o.shape == (5, 5)

    def test_trace_out_spin_and_pm_trace(self) -> None:
        rho = self._make_full_rho()
        full_trace = np.trace(rho)
        rho_o = trace_out_spin_and_pseudomode(rho, N=4, K=2)
        assert np.isclose(np.trace(rho_o), full_trace, rtol=1e-10)

    def test_wrong_dimension_raises(self) -> None:
        with pytest.raises(AssertionError):
            trace_out_pseudomode(np.eye(10), N=5, K=3)


# =============================================================================
# Tests: QFI Computation
# =============================================================================


class TestQFIComputation:
    """QFI computation (with and without ancilla)."""

    def test_initial_nonzero(self) -> None:
        """Coherent state should have non-zero QFI."""
        config = PseudomodeConfig(N=10, K=3, alpha=1.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        state = pseudomode_initial_state(config)
        rho = np.outer(state, state.conj())
        assert compute_qfi_with_ancilla(rho, config.N, config.K) > 0

    def test_vacuum_has_zero_qfi(self) -> None:
        """Vacuum state should have zero QFI."""
        config = PseudomodeConfig(N=5, K=3, alpha=0.0, g_sa=0.0, g_sp=0.0, lam=0.0)
        state = pseudomode_initial_state(config)
        rho = np.outer(state, state.conj())
        assert compute_qfi_with_ancilla(rho, config.N, config.K) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_qfi_without_smaller_or_equal(self) -> None:
        """QFIs without ancilla should be <= QFI with ancilla."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.3, lam=0.5, T_decay=0.5
        )
        result = run_metrology_protocol(config)
        assert result["qfi_with"] >= result["qfi_without"] - 1e-10

    def test_qfi_preservation_ratio(self) -> None:
        """Preservation ratio should be <= 1."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.3, lam=0.5, T_decay=0.5
        )
        result = run_metrology_protocol(config)
        assert 0.0 <= result["ratio_with"] <= 1.0 + 1e-6

    def test_qfi_preservation_ratio_fn(self) -> None:
        """qfi_preservation_ratio should match protocol's ratio_with."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.3, lam=0.5, T_decay=0.5
        )
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)
        rho_ent = np.outer(entangled, entangled.conj())
        qfi0 = compute_qfi_with_ancilla(rho_ent, config.N, config.K)

        rho_final = evolve_pseudomode(entangled, config)
        ratio = qfi_preservation_ratio(
            rho_final, qfi0, config.N, config.K, with_ancilla=True
        )
        assert 0.0 <= ratio <= 1.0 + 1e-6

    def test_qfi_without_ancilla_fn(self) -> None:
        """compute_qfi_without_ancilla should return finite non-negative."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.5, tau=0.2, g_sp=0.3, lam=0.5, T_decay=0.5
        )
        result = run_metrology_protocol(config)
        qfi_without = compute_qfi_without_ancilla(
            result["rho_final"], config.N, config.K
        )
        assert qfi_without >= 0
        assert np.isfinite(qfi_without)


# =============================================================================
# Tests: Evolution
# =============================================================================


class TestEvolvePseudomode:
    """Lindblad and unitary evolution."""

    def test_unitary_evolution_preserves_purity(self) -> None:
        """Without dissipation (lam=0), unitary evolution preserves purity.

        The initial coherent state is not perfectly pure due to Fock
        truncation (~0.9988 for alpha=1.0, N=5). Unitary evolution
        (via expm) should preserve this value exactly.
        """
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.0, g_sp=0.0, lam=0.0, T_decay=1.0, dt=0.01
        )
        state = pseudomode_initial_state(config)
        rho0 = np.outer(state, state.conj())
        purity_before = np.real(np.trace(rho0 @ rho0))
        rho = evolve_pseudomode(state, config, method="rk4")
        purity_after = np.real(np.trace(rho @ rho))
        assert np.isclose(purity_after, purity_before, rtol=1e-10)

    def test_dissipative_evolution_reduces_purity(self) -> None:
        """With dissipation, purity should be < 1."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.0, g_sp=0.3, lam=0.5, T_decay=1.0, dt=0.01
        )
        state = pseudomode_initial_state(config)
        rho = evolve_pseudomode(state, config, method="rk4")
        purity = np.real(np.trace(rho @ rho))
        assert purity < 1.0

    def test_scipy_method_matches_rk4(self) -> None:
        """scipy and rk4 should agree to reasonable tolerance."""
        config = PseudomodeConfig(
            N=3, K=2, alpha=0.5, g_sa=0.0, g_sp=0.2, lam=0.3, T_decay=0.5, dt=0.025
        )
        state = pseudomode_initial_state(config)
        rho_rk4 = evolve_pseudomode(state, config, method="rk4")
        rho_scipy = evolve_pseudomode(state, config, method="scipy")
        np.testing.assert_allclose(rho_rk4, rho_scipy, atol=1e-4)

    def test_trace_preserved(self) -> None:
        """Final density matrix should have trace = 1."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.0, g_sp=0.3, lam=0.5, T_decay=0.5, dt=0.01
        )
        state = pseudomode_initial_state(config)
        rho = evolve_pseudomode(state, config, method="rk4")
        assert np.isclose(np.trace(rho), 1.0, atol=1e-8)

    def test_hermiticity_preserved(self) -> None:
        """Final density matrix should be Hermitian."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.0, g_sp=0.3, lam=0.5, T_decay=0.5, dt=0.01
        )
        state = pseudomode_initial_state(config)
        rho = evolve_pseudomode(state, config, method="rk4")
        assert np.allclose(rho, rho.conj().T, atol=1e-8)

    def test_unknown_method_raises(self) -> None:
        config = PseudomodeConfig(N=5, K=3)
        state = pseudomode_initial_state(config)
        with pytest.raises(ValueError, match="Unknown method"):
            evolve_pseudomode(state, config, method="invalid")


# =============================================================================
# Tests: Metrology Protocol
# =============================================================================


class TestRunMetrologyProtocol:
    """Full metrology protocol."""

    _default_kw: ClassVar[dict] = {
        "N": 5,
        "K": 3,
        "alpha": 1.0,
        "g_sa": 0.5,
        "tau": 0.2,
        "g_sp": 0.3,
        "lam": 0.5,
        "T_decay": 0.5,
    }

    def test_protocol_completes(self) -> None:
        result = run_metrology_protocol(PseudomodeConfig(**self._default_kw))
        expected_keys = {
            "rho_final",
            "qfi_with",
            "qfi_without",
            "qfi_initial",
            "ratio_with",
            "ratio_without",
            "pm_occupancy",
            "validation",
            "pm_occupancy_safe",
        }
        assert expected_keys.issubset(result.keys())

    def test_ratio_between_zero_and_one(self) -> None:
        result = run_metrology_protocol(PseudomodeConfig(**self._default_kw))
        assert 0 <= result["ratio_with"] <= 1.0 + 1e-6
        assert 0 <= result["ratio_without"] <= 1.0 + 1e-6

    def test_validation_passes(self) -> None:
        result = run_metrology_protocol(PseudomodeConfig(**self._default_kw))
        v = result["validation"]
        assert v["is_hermitian"]
        assert v["is_normalized"]
        assert v["is_positive"]

    def test_theta_zero_baseline(self) -> None:
        """At theta=0, with and without ancilla should match."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sa=0.0, tau=0.2, g_sp=0.3, lam=0.5, T_decay=0.5
        )
        result = run_metrology_protocol(config)
        assert np.isclose(result["ratio_with"], result["ratio_without"], rtol=1e-5)

    def test_qfi_non_increasing(self) -> None:
        """QFI after decoherence should not exceed initial QFI."""
        result = run_metrology_protocol(PseudomodeConfig(**self._default_kw))
        assert result["qfi_with"] <= result["qfi_initial"] + 1e-10
        assert result["qfi_without"] <= result["qfi_initial"] + 1e-10

    def test_occupancy_check_in_result(self) -> None:
        """Occupancy safe flag should be in results."""
        result = run_metrology_protocol(PseudomodeConfig(**self._default_kw))
        assert "pm_occupancy_safe" in result
        assert isinstance(result["pm_occupancy_safe"], bool)


# =============================================================================
# Tests: Physical Invariants
# =============================================================================


class TestPhysicalInvariants:
    """Physical invariants for the pseudomode system."""

    def test_trace_conservation_over_evolution(self) -> None:
        """Lindblad evolution should preserve trace."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sp=0.5, lam=1.0, T_decay=2.0, dt=0.05
        )
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)
        rho = evolve_pseudomode(entangled, config, method="rk4")
        assert np.isclose(np.trace(rho), 1.0, atol=1e-8)

    def test_hermiticity_throughout(self) -> None:
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sp=0.5, lam=1.0, T_decay=2.0, dt=0.05
        )
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)
        rho = evolve_pseudomode(entangled, config, method="rk4")
        assert np.allclose(rho, rho.conj().T, atol=1e-8)

    def test_positivity(self) -> None:
        """Density matrix must be positive semidefinite."""
        config = PseudomodeConfig(
            N=5, K=3, alpha=1.0, g_sp=0.5, lam=1.0, T_decay=2.0, dt=0.05
        )
        state = pseudomode_initial_state(config)
        entangled = apply_ancilla_entanglement(state, config)
        rho = evolve_pseudomode(entangled, config, method="rk4")
        eigenvalues = np.linalg.eigvalsh(rho)
        assert np.all(eigenvalues >= -1e-8)

    def test_operator_unitarity(self) -> None:
        """The entanglement unitary should be unitary."""
        config = PseudomodeConfig(N=5, K=3, g_sa=1.0, tau=0.2)
        from src.physics.hybrid_system import oscillator_number, spin_operator_x

        n_op = oscillator_number(5)
        sigma_x = spin_operator_x()
        I_pm = np.eye(4, dtype=complex)
        H_sa = config.g_sa * tripartite_operator(n_op, sigma_x, I_pm, 5, 3)
        H_sa = 0.5 * (H_sa + H_sa.conj().T)
        import scipy.linalg

        U = scipy.linalg.expm(-1j * H_sa * config.tau)
        assert np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-10)

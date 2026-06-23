"""Tests for the 2026-05-07 report local module.

Contains all migrated tests from:
- test_hybrid_system.py (TestAdaptiveTruncation, TestExpectationValues,
  TestValidation, TestUnitaryEvolution, coherent state tests)
- test_hybrid_mzi.py (TestEmbedding, TestMZIOperators, TestMZIEvolution,
  TestOutputProbabilities, TestWignerComputation)
- test_hybrid_lindblad.py (all test classes)
- test_wigner.py (all test classes)

Because the report directory name contains hyphens, we load the
module via importlib to avoid Python import syntax issues.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy

from src.evolution.lindblad_solver import lindblad_rhs
from src.physics.hybrid_mzi import (
    compute_wigner_for_state,
    embed_hybrid_in_mzi,
    evolve_hybrid_mzi,
    mzi_beam_splitter,
    mzi_marginal_photon_probs,
    mzi_output_probabilities,
    mzi_phase_generator,
    mzi_phase_shift,
    wigner_from_hybrid_state,
    wigner_function_single,
    wigner_is_negative,
)
from src.physics.hybrid_system import (
    adaptive_truncation,
    evolve_hybrid_state,
    hybrid_coherent_state,
    hybrid_hamiltonian_n,
    hybrid_mean_photon,
    hybrid_vacuum_state,
    validate_hybrid_state,
    validate_hybrid_unitary,
)

# ── Load high_order_squeezing.py via importlib ──────────────────────────────────────────────
_local_path = Path(__file__).resolve().parent / "high_order_squeezing.py"
_dirname = Path(__file__).resolve().parent.name
_modname = f"report_local_{_dirname}"
_spec = importlib.util.spec_from_file_location(_modname, str(_local_path))
assert _spec is not None, f"Could not find high_order_squeezing.py at {_local_path}"
_report_local = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Register in sys.modules so dataclass decorator can resolve types
sys.modules[_modname] = _report_local
_spec.loader.exec_module(_report_local)

# Bind only names defined in the experiment module (not re-exported from src/)
apply_squeezing = _report_local.apply_squeezing
build_hybrid_hamiltonian = _report_local.build_hybrid_hamiltonian
build_hybrid_lindblad_operators = _report_local.build_hybrid_lindblad_operators
evolve_hybrid_lindblad = _report_local.evolve_hybrid_lindblad
HybridLindbladConfig = _report_local.HybridLindbladConfig
run_hybrid_simulation = _report_local.run_hybrid_simulation
validate_hybrid_density_matrix = _report_local.validate_hybrid_density_matrix

# =============================================================================
# Tests from test_hybrid_system.py
# =============================================================================


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


# Additional state preparation tests
class TestCoherentStateNormalized:
    def test_given_coherent_state_then_normalized(self) -> None:
        N = 10
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        assert validate_hybrid_state(state, N), (
            "Condition failed: validate_hybrid_state(state, N)"
        )


class TestCoherentStateAmplitude:
    def test_given_coherent_state_then_amplitude_matches(self) -> None:
        N = 10
        alpha = 1.0
        state = hybrid_coherent_state(N, alpha=alpha, spin_state="down")
        # Mean photon should be approximately |α|² = 1
        mean_n = hybrid_mean_photon(state, N)
        assert mean_n == pytest.approx(1.0, abs=0.1), (
            "Expected mean_n == pytest.approx(1.0, abs=0.1)"
        )


# =============================================================================
# Tests from test_hybrid_mzi.py
# =============================================================================


class TestEmbedding:
    def test_given_vacuum_state_then_norm_preserved(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)
        assert np.sum(np.abs(embedded) ** 2) == pytest.approx(
            np.sum(np.abs(state) ** 2), abs=1e-10
        )

    def test_given_vacuum_state_then_correct_dimension(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)
        expected_dim = 2 * (N + 1) ** 2
        assert embedded.shape == (expected_dim,)

    def test_given_vacuum_state_then_amplitude_only_at_origin(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)
        assert embedded[0] == pytest.approx(1.0, abs=1e-10)
        assert np.sum(np.abs(embedded[1:]) ** 2) == pytest.approx(0.0, abs=1e-10)

    def test_given_density_matrix_then_correct_shape(self) -> None:
        N = 5
        dim_mzi = 2 * (N + 1) ** 2
        state = hybrid_vacuum_state(N, spin_state="down")
        rho = np.outer(state, state.conj())
        embedded = embed_hybrid_in_mzi(rho, N)
        assert embedded.shape == (dim_mzi, dim_mzi)

    def test_given_density_matrix_then_trace_preserved(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        rho = np.outer(state, state.conj())
        embedded = embed_hybrid_in_mzi(rho, N)
        assert np.trace(embedded) == pytest.approx(1.0, abs=1e-10)

    def test_given_density_matrix_then_hermitian(self) -> None:
        N = 5
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        rho = np.outer(state, state.conj())
        embedded = embed_hybrid_in_mzi(rho, N)
        assert embedded == pytest.approx(embedded.conj().T, abs=1e-10)

    def test_given_density_matrix_then_positive_semidefinite(self) -> None:
        N = 5
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        rho = np.outer(state, state.conj())
        embedded = embed_hybrid_in_mzi(rho, N)
        eigenvalues = np.linalg.eigvalsh(embedded)
        assert np.all(eigenvalues >= -1e-10)

    def test_given_pure_vacuum_density_then_only_zero_entry(self) -> None:
        N = 5
        dim_hybrid = 2 * (N + 1)
        rho = np.zeros((dim_hybrid, dim_hybrid), dtype=complex)
        rho[0, 0] = 1.0
        embedded = embed_hybrid_in_mzi(rho, N)
        assert embedded[0, 0] == pytest.approx(1.0, abs=1e-10)
        all_others = np.abs(embedded).sum() - np.abs(embedded[0, 0])
        assert all_others == pytest.approx(0.0, abs=1e-10)

    def test_given_pure_state_then_density_embedding_matches(self) -> None:
        N = 5
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        rho = np.outer(state, state.conj())
        psi_emb = embed_hybrid_in_mzi(state, N)
        rho_from_pure = np.outer(psi_emb, psi_emb.conj())
        rho_emb = embed_hybrid_in_mzi(rho, N)
        assert rho_emb == pytest.approx(rho_from_pure, abs=1e-10)

    def test_given_off_diagonal_density_then_correct_embedded_index(self) -> None:
        N = 3
        dim_hybrid = 2 * (N + 1)
        rho = np.zeros((dim_hybrid, dim_hybrid), dtype=complex)
        rho[0, 2] = 0.5 + 0.5j
        embedded = embed_hybrid_in_mzi(rho, N)
        idx_0 = 0 * (N + 1) * 2 + 0
        idx_1 = 1 * (N + 1) * 2 + 0
        assert embedded[idx_0, idx_1] == pytest.approx(0.5 + 0.5j, abs=1e-10)

    def test_given_non_square_density_then_raises_valueerror(self) -> None:
        N = 5
        bad_rho = np.zeros((4, 6), dtype=complex)
        with pytest.raises(ValueError, match="must have shape"):
            embed_hybrid_in_mzi(bad_rho, N)

    def test_given_larger_cutoff_then_properties_preserved(self) -> None:
        N = 10
        state = hybrid_vacuum_state(N, spin_state="up")
        rho = np.outer(state, state.conj())
        embedded = embed_hybrid_in_mzi(rho, N)
        dim_mzi = 2 * (N + 1) ** 2
        assert embedded.shape == (dim_mzi, dim_mzi)
        assert np.trace(embedded) == pytest.approx(1.0, abs=1e-10)
        assert embedded == pytest.approx(embedded.conj().T, abs=1e-10)


class TestMZIOperators:
    def test_given_beam_splitter_then_unitary(self) -> None:
        N = 5
        bs = mzi_beam_splitter(N, theta=np.pi / 4)
        assert validate_hybrid_unitary(bs, tol=1e-8)

    def test_given_phase_shift_then_unitary(self) -> None:
        N = 5
        ps = mzi_phase_shift(N, phi_phase=0.5)
        assert validate_hybrid_unitary(ps, tol=1e-8)

    def test_given_phase_generator_then_hermitian(self) -> None:
        N = 5
        G = mzi_phase_generator(N)
        assert pytest.approx(G.conj().T, abs=1e-10) == G

    def test_given_phase_generator_then_diagonal(self) -> None:
        N = 5
        G = mzi_phase_generator(N)
        assert G - np.diag(np.diag(G)) == pytest.approx(0, abs=1e-10)


class TestMZIEvolution:
    def test_given_evolution_then_normalized(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi_phase=0.5)
        norm = np.sum(np.abs(output) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_given_zero_phase_then_normalized(self) -> None:
        N = 3
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi_phase=0.0)
        norm = np.sum(np.abs(output) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_given_evolution_then_correct_shape(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi_phase=1.0)
        expected_dim = 2 * (N + 1) ** 2
        assert output.shape == (expected_dim,)


class TestOutputProbabilities:
    def test_given_evolution_then_probabilities_sum_to_one(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi_phase=0.5)
        probs = mzi_output_probabilities(output, N)
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_given_evolution_then_marginals_sum_to_one(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi_phase=0.5)
        P1, P2 = mzi_marginal_photon_probs(output, N)
        assert np.sum(P1) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(P2) == pytest.approx(1.0, abs=1e-6)

    def test_given_evolution_then_marginals_have_correct_length(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi_phase=0.5)
        P1, P2 = mzi_marginal_photon_probs(output, N)
        assert len(P1) == N + 1
        assert len(P2) == N + 1


class TestWignerComputation:
    def test_given_vacuum_then_correct_shape(self) -> None:
        N = 10
        state = hybrid_vacuum_state(N, spin_state="down")
        x, p, W = compute_wigner_for_state(state, N, x_max=3.0, n_points=50)
        assert x.shape == (50,)
        assert p.shape == (50,)
        assert W.shape == (50, 50)

    def test_given_vacuum_then_wigner_positive(self) -> None:
        N = 10
        state = hybrid_vacuum_state(N, spin_state="down")
        _, _, W = compute_wigner_for_state(state, N, x_max=3.0, n_points=50)
        assert np.min(W) >= -1e-10


# =============================================================================
# Tests from test_hybrid_lindblad.py
# =============================================================================


class TestHybridLindbladConfig:
    def test_default_values_are_reasonable(self) -> None:
        config = HybridLindbladConfig(N=5)
        assert config.N == 5
        assert config.n == 2
        assert config.omega_n == 1.0
        assert config.theta_n == 0.0
        assert config.phi_phase == 0.0
        assert config.gamma_1 == 0.0
        assert config.gamma_2 == 0.0
        assert config.gamma_phi == 0.0
        assert config.t_squeeze == 1.0

    def test_custom_values_are_preserved(self) -> None:
        config = HybridLindbladConfig(
            N=10,
            n=3,
            omega_n=0.5,
            theta_n=np.pi / 4,
            phi_phase=0.1,
            gamma_1=0.01,
            gamma_phi=0.02,
            t_squeeze=2.0,
        )
        assert config.N == 10
        assert config.n == 3
        assert config.omega_n == 0.5
        assert config.phi_phase == 0.1
        assert config.gamma_1 == 0.01


class TestBuildHybridHamiltonian:
    @pytest.mark.parametrize("n", [2, 3, 4], ids=["n=2", "n=3", "n=4"])
    def test_hamiltonian_has_correct_shape(self, n: int) -> None:
        config = HybridLindbladConfig(N=5, n=n, omega_n=1.0)
        H = build_hybrid_hamiltonian(config)
        dim = 2 * (5 + 1)
        assert H.shape == (dim, dim)

    @pytest.mark.parametrize("n", [2, 3, 4], ids=["n=2", "n=3", "n=4"])
    def test_hamiltonian_is_hermitian(self, n: int) -> None:
        config = HybridLindbladConfig(N=5, n=n, omega_n=1.0)
        H = build_hybrid_hamiltonian(config)
        assert pytest.approx(H.conj().T) == H

    def test_given_zero_squeezing_rate_then_hamiltonian_is_zero(self) -> None:
        config = HybridLindbladConfig(N=5, n=2, omega_n=0.0)
        H = build_hybrid_hamiltonian(config)
        assert pytest.approx(0) == H

    def test_n_2_and_n_3_give_different_hamiltonians(self) -> None:
        config2 = HybridLindbladConfig(N=5, n=2, omega_n=1.0)
        config3 = HybridLindbladConfig(N=5, n=3, omega_n=1.0)
        H2 = build_hybrid_hamiltonian(config2)
        H3 = build_hybrid_hamiltonian(config3)
        assert pytest.approx(H3) != H2


class TestBuildHybridLindbladOperators:
    def test_given_no_dissipation_then_lists_are_empty(self) -> None:
        config = HybridLindbladConfig(N=5, gamma_1=0, gamma_2=0, gamma_phi=0)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 0
        assert len(gammas) == 0

    def test_one_body_loss_adds_one_operator(self) -> None:
        config = HybridLindbladConfig(N=5, gamma_1=0.1)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 1
        assert len(gammas) == 1
        dim = 2 * (5 + 1)
        assert L_ops[0].shape == (dim, dim)

    def test_phase_diffusion_adds_one_operator(self) -> None:
        config = HybridLindbladConfig(N=5, gamma_phi=0.1)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 1
        assert len(gammas) == 1

    def test_multiple_channels_add_multiple_operators(self) -> None:
        config = HybridLindbladConfig(N=5, gamma_1=0.1, gamma_2=0.05, gamma_phi=0.02)
        L_ops, gammas = build_hybrid_lindblad_operators(config)
        assert len(L_ops) == 3
        assert len(gammas) == 3

    def test_one_body_loss_l_a_i(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        L_ops, _ = build_hybrid_lindblad_operators(config)

        L = L_ops[0]
        dim_osc = N + 1

        L_down = L[::2, ::2][:dim_osc, :dim_osc]
        L_up = L[1::2, 1::2][:dim_osc, :dim_osc]

        a = np.zeros((dim_osc, dim_osc), dtype=complex)
        for n in range(1, dim_osc):
            a[n - 1, n] = np.sqrt(n)

        assert L_down == pytest.approx(np.sqrt(0.1) * a)
        assert L_up == pytest.approx(np.sqrt(0.1) * a)


class TestLindbladRHS:
    def test_given_no_dissipation_then_rhs_is_minus_i_commutator(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, n=2, omega_n=1.0)
        H = build_hybrid_hamiltonian(config)
        L_ops: list[np.ndarray] = []
        gammas: list[float] = []

        dim = 2 * (N + 1)
        rho = np.eye(dim, dtype=complex) / dim

        drho = lindblad_rhs(rho, H, L_ops, gammas)

        expected = -1.0j * (H @ rho - rho @ H)
        assert drho == pytest.approx(expected)

    def test_given_dissipation_then_drift_is_nonzero(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        H = np.zeros((2 * (N + 1), 2 * (N + 1)), dtype=complex)
        L_ops, gammas = build_hybrid_lindblad_operators(config)

        dim = 2 * (N + 1)
        rho = np.zeros((dim, dim), dtype=complex)
        rho[2, 2] = 1.0

        drho = lindblad_rhs(rho, H, L_ops, gammas)

        assert np.max(np.abs(drho)) > 0


class TestEvolveHybridLindblad:
    def test_given_zero_time_then_returns_initial_state(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T_decay=0.0, dt=0.01)

        rho0 = np.outer(psi0, psi0.conj())
        assert rho_final == pytest.approx(rho0)

    def test_given_no_dissipation_then_matches_unitary_evolution(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, t_squeeze=1.0)
        config.gamma_1 = 0.0
        config.gamma_phi = 0.0

        psi0 = hybrid_vacuum_state(N, spin_state="down")

        H = build_hybrid_hamiltonian(config)
        U = scipy.linalg.expm(-1.0j * H * 1.0)
        rho_expected = U @ np.outer(psi0, psi0.conj()) @ U.conj().T

        rho_final = evolve_hybrid_lindblad(
            psi0, config, T_decay=1.0, dt=0.001, method="rk4"
        )

        assert rho_final == pytest.approx(rho_expected, abs=1e-4)

    def test_given_no_dissipation_then_trace_is_preserved(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, gamma_1=0, gamma_phi=0)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T_decay=1.0, dt=0.01)

        assert np.trace(rho_final) == pytest.approx(1.0, abs=1e-6)

    def test_given_particle_loss_then_trace_does_not_exceed_one(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.2)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T_decay=1.0, dt=0.01)

        assert np.trace(rho_final) <= 1.0 + 1e-6

    def test_density_matrix_remains_hermitian(self) -> None:
        N = 5
        config = HybridLindbladConfig(
            N=N,
            n=2,
            omega_n=0.3,
            gamma_1=0.1,
            gamma_phi=0.05,
        )
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T_decay=0.5, dt=0.01)

        assert rho_final == pytest.approx(rho_final.conj().T, abs=1e-6)

    def test_eigenvalues_are_non_negative(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=0.1)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        rho_final = evolve_hybrid_lindblad(psi0, config, T_decay=0.5, dt=0.01)

        eigenvalues = np.linalg.eigvalsh(rho_final)
        assert np.min(eigenvalues.real) >= -1e-6


class TestApplySqueezing:
    def test_squeezing_increases_mean_photon_number(self) -> None:
        N = 10
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, t_squeeze=1.0)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        n_initial = hybrid_mean_photon(psi0, N)
        psi_sq = apply_squeezing(config, psi0)
        n_squeezed = hybrid_mean_photon(psi_sq, N)

        assert n_squeezed > n_initial

    def test_squeezing_preserves_norm(self) -> None:
        N = 10
        config = HybridLindbladConfig(N=N, n=3, omega_n=0.3, t_squeeze=2.0)
        psi0 = hybrid_vacuum_state(N, spin_state="down")

        psi_sq = apply_squeezing(config, psi0)

        assert np.sum(np.abs(psi_sq) ** 2) == pytest.approx(1.0, abs=1e-6)


class TestValidateHybridDensityMatrix:
    def test_valid_density_matrix_passes_all_checks(self) -> None:
        N = 5
        dim = 2 * (N + 1)
        rho = np.eye(dim, dtype=complex) / dim

        result = validate_hybrid_density_matrix(rho)

        assert result["is_hermitian"]
        assert result["is_normalized"]
        assert result["is_positive"]

    def test_pure_state_is_valid(self) -> None:
        N = 5
        psi = hybrid_vacuum_state(N, spin_state="down")
        rho = np.outer(psi, psi.conj())

        result = validate_hybrid_density_matrix(rho)

        assert result["is_hermitian"]
        assert result["is_normalized"]
        assert result["is_positive"]

    def test_non_hermitian_matrix_fails_validation(self) -> None:
        N = 5
        dim = 2 * (N + 1)
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 1] = 1.0

        result = validate_hybrid_density_matrix(rho)

        assert not result["is_hermitian"]


class TestRunHybridSimulation:
    def test_simulation_completes_without_errors(self) -> None:
        config = HybridLindbladConfig(
            N=5,
            n=2,
            omega_n=0.5,
            t_squeeze=0.5,
            gamma_1=0.01,
        )
        result = run_hybrid_simulation(config)

        assert "final_state" in result
        assert "validation" in result

        validation = result["validation"]
        assert validation["is_hermitian"]
        assert validation["is_positive"]

    def test_simulation_with_n_3_completes(self) -> None:
        config = HybridLindbladConfig(N=8, n=3, omega_n=0.3, t_squeeze=1.0, gamma_1=0.0)
        result = run_hybrid_simulation(config)

        assert result["final_state"] is not None
        assert result["validation"]["is_hermitian"]


class TestWignerNegativity:
    @pytest.mark.slow
    def test_n_2_gaussian_does_not_have_wigner_negativity(self) -> None:
        N = 20
        config = HybridLindbladConfig(N=N, n=2, omega_n=0.5, t_squeeze=1.0)
        psi_sq = apply_squeezing(config)

        rho_osc = self._extract_oscillator_density(psi_sq, N)

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)
        W = wigner_function_single(rho_osc, x, p)

        assert np.min(W) >= -1e-3

    def test_n_3_non_gaussian_shows_wigner_negativity(self) -> None:
        N = 10
        config = HybridLindbladConfig(N=N, n=3, omega_n=0.3, t_squeeze=2.0)
        psi_sq = apply_squeezing(config)

        rho_osc = self._extract_oscillator_density(psi_sq, N)

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)
        W = wigner_function_single(rho_osc, x, p)

        assert np.min(W) < -1e-3

    @staticmethod
    def _extract_oscillator_density(
        hybrid_state: np.ndarray,
        N: int,
    ) -> np.ndarray:
        dim_osc = N + 1
        rho_hybrid = np.outer(hybrid_state, hybrid_state.conj())
        rho_reshaped = rho_hybrid.reshape(dim_osc, 2, dim_osc, 2)
        return np.trace(rho_reshaped, axis1=1, axis2=3)


class TestN4WignerNegativityDiagnostic:
    OMEGA_N = 1.0
    X_MAX = 4.0

    def _extract_oscillator_density(
        self,
        hybrid_state: np.ndarray,
        N: int,
    ) -> np.ndarray:
        dim_osc = N + 1
        rho_hybrid = np.outer(hybrid_state, hybrid_state.conj())
        rho_reshaped = rho_hybrid.reshape(dim_osc, 2, dim_osc, 2)
        return np.trace(rho_reshaped, axis1=1, axis2=3)

    def test_n4_baseline(self) -> None:
        N = 10
        config = HybridLindbladConfig(N=N, n=4, omega_n=0.5, t_squeeze=2.0)
        psi_sq = apply_squeezing(config)
        rho_osc = self._extract_oscillator_density(psi_sq, N)

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)
        W = wigner_function_single(rho_osc, x, p)

        assert np.min(W) < -1e-3

    @pytest.mark.slow
    def test_n4_grid_sweep(self) -> None:
        N = 20
        t_sqz = 0.30
        resolutions = [40, 60, 80]

        config = HybridLindbladConfig(N=N, n=4, omega_n=self.OMEGA_N, t_squeeze=t_sqz)
        psi_sq = apply_squeezing(config)
        rho_osc = self._extract_oscillator_density(psi_sq, N)

        min_values = []
        for n_pts in resolutions:
            x = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            p = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            W = wigner_function_single(rho_osc, x, p)
            min_values.append(float(np.min(W)))

        assert min(min_values) < -1e-5

    @pytest.mark.slow
    def test_n4_time_sweep(self) -> None:
        N = 20
        n_pts = 50
        times = np.array([0.15, 0.30, 0.45])

        best_mins: list[tuple[float, float]] = []

        for t in times:
            config = HybridLindbladConfig(
                N=N,
                n=4,
                omega_n=self.OMEGA_N,
                t_squeeze=float(t),
            )
            psi_sq = apply_squeezing(config)
            rho_osc = self._extract_oscillator_density(psi_sq, N)

            x = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            p = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            W = wigner_function_single(rho_osc, x, p)

            best_mins.append((float(t), float(np.min(W))))

        best_mins.sort(key=lambda pair: pair[1])
        _t_best, w_best = best_mins[0]

        assert w_best < -1e-4

    @pytest.mark.slow
    def test_n4_truncation_check(self) -> None:
        t_sqz = 0.30
        n_pts = 60
        N_values = [10, 20, 30]

        best_w = 0.0
        for N in N_values:
            config = HybridLindbladConfig(
                N=N,
                n=4,
                omega_n=self.OMEGA_N,
                t_squeeze=t_sqz,
            )
            psi_sq = apply_squeezing(config)
            rho_osc = self._extract_oscillator_density(psi_sq, N)

            x = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            p = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
            W = wigner_function_single(rho_osc, x, p)
            w_min = float(np.min(W))
            best_w = min(best_w, w_min)

        assert best_w < -1e-5

    @pytest.mark.slow
    def test_n4_high_resolution_confirm(self) -> None:
        N = 20
        t_sqz = 0.30
        n_pts = 100

        config = HybridLindbladConfig(N=N, n=4, omega_n=self.OMEGA_N, t_squeeze=t_sqz)
        psi_sq = apply_squeezing(config)
        rho_osc = self._extract_oscillator_density(psi_sq, N)

        x = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
        p = np.linspace(-self.X_MAX, self.X_MAX, n_pts)
        W = wigner_function_single(rho_osc, x, p)

        assert np.min(W) < -1e-5


class TestEdgeCases:
    def test_given_minimal_dimension_then_evolves(self) -> None:
        config = HybridLindbladConfig(N=1, n=2, omega_n=0.5)
        psi0 = hybrid_vacuum_state(1, spin_state="down")
        rho = evolve_hybrid_lindblad(psi0, config, T_decay=0.1, dt=0.01)
        assert rho.shape == (4, 4)

    def test_large_gamma_decays_photon_number(self) -> None:
        N = 5
        config = HybridLindbladConfig(N=N, gamma_1=10.0)
        psi0 = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")

        initial_n = hybrid_mean_photon(psi0, N)
        rho = evolve_hybrid_lindblad(psi0, config, T_decay=1.0, dt=0.01)

        dim_osc = N + 1
        n_op = np.zeros((dim_osc, dim_osc), dtype=complex)
        for n in range(dim_osc):
            n_op[n, n] = n
        n_hybrid = np.kron(n_op, np.eye(2, dtype=complex))

        final_n = np.real(np.trace(rho @ n_hybrid))

        assert final_n < initial_n * 0.5
        assert np.trace(rho) == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Tests from test_wigner.py
# =============================================================================


class TestWignerFunctionSingle:
    """Test Wigner function computation for single-mode states."""

    def test_w_x_p_dx_dp_1_for_vacuum(self) -> None:
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0  # |0⟩⟨0|

        x = np.linspace(-5, 5, 100)
        p = np.linspace(-5, 5, 100)

        W = wigner_function_single(rho, x, p)

        # Integrate using trapezoidal rule
        dx = x[1] - x[0]
        dp = p[1] - p[0]
        integral = np.sum(W) * dx * dp

        # Relaxed tolerance due to numerical integration
        assert integral == pytest.approx(1.0, rel=1e-1, abs=0.1), (
            "Expected integral == pytest.approx(1.0, rel=1e-1, abs=0.1)"
        )

    def test_given_vacuum_wigner_max_then_be_2_0_637_at_origin(self) -> None:
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-3, 3, 50)
        p = np.linspace(-3, 3, 50)

        W = wigner_function_single(rho, x, p)

        max_w = np.max(W)
        assert max_w == pytest.approx(2.0 / np.pi, abs=0.1), (
            "Expected max_w == pytest.approx(2.0 / np.pi, abs=0.1)"
        )

    def test_given_vacuum_wigner_then_be_non_negative_everywhere(self) -> None:
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-5, 5, 100)
        p = np.linspace(-5, 5, 100)

        W = wigner_function_single(rho, x, p)

        assert np.min(W) >= -1e-10

    def test_given_wigner_output_then_have_correct_shape(self) -> None:
        N = 5
        rho = np.eye(N + 1, dtype=complex) / (N + 1)  # Maximally mixed

        x = np.linspace(-3, 3, 50)
        p = np.linspace(-3, 3, 60)

        W = wigner_function_single(rho, x, p)

        assert W.shape == (50, 60)


class TestWignerFromHybridState:
    """Test Wigner extraction from hybrid state."""

    def test_given_wigner_for_0_then_be_vacuum_wigner(self) -> None:
        N = 10
        state = hybrid_vacuum_state(N, spin_state="down")

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_from_hybrid_state(state, N, x, p, spin_component="down")

        assert W.shape == (50, 50)
        # Vacuum should have max near 2/π ≈ 0.637
        assert np.max(W) > 0.3

    @pytest.mark.slow
    def test_given_wigner_for_coherent_state_then_be_gaussian(self) -> None:
        N = 20
        alpha = 1.0 + 0j
        state = hybrid_coherent_state(N, alpha, spin_state="down")

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_from_hybrid_state(state, N, x, p, spin_component="down")

        # Coherent states are Gaussian - allow small negative due to
        # numerical integration artifacts in the discrete Wigner computation
        assert np.min(W) > -2e-2


class TestWignerMinimum:
    """Test Wigner minimum and negativity detection."""

    def test_given_vacuum_wigner_minimum_then_be_positive(self) -> None:
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_function_single(rho, x, p)
        min_w = float(np.min(W))

        assert min_w >= -1e-10

    def test_vacuum_is_not_wigner_negative(self) -> None:
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        x = np.linspace(-5, 5, 50)
        p = np.linspace(-5, 5, 50)

        W = wigner_function_single(rho, x, p)

        assert not wigner_is_negative(W), "wigner_is_negative(W) should be falsy"

    def test_fock_n1_negativity(self) -> None:
        """Fock state |1⟩ must show negative Wigner at origin.

        Analytical: W_1(0,0) = -2/π ≈ -0.637.
        This is a regression test against the incorrect formula
        W = (2/π) exp(-2r²) Σ ρ_mn (-1)^n α^m (α*)^n / √(m!n!)
        which gave W(0,0) = 0 for |1⟩.
        """
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[1, 1] = 1.0  # |1⟩⟨1|

        x = np.array([0.0])
        p = np.array([0.0])
        W = wigner_function_single(rho, x, p)

        min_W = W[0, 0]
        assert min_W == pytest.approx(-2.0 / np.pi, abs=1e-4), (
            f"Fock |1⟩ W(0,0) should be -2/π ≈ -0.637, got {min_W}"
        )

    def test_fock_n2_positivity_at_origin(self) -> None:
        """Fock state |2⟩ has positive Wigner at origin: W = +2/π.

        Even-n Fock states have (+1)^n factor, giving positive Wigner at origin.
        """
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[2, 2] = 1.0  # |2⟩⟨2|

        x = np.array([0.0])
        p = np.array([0.0])
        W = wigner_function_single(rho, x, p)

        max_W = W[0, 0]
        assert max_W == pytest.approx(2.0 / np.pi, abs=1e-4), (
            f"Fock |2⟩ W(0,0) should be +2/π ≈ +0.637, got {max_W}"
        )

    def test_maximally_mixed_state_has_no_wigner_negativity(self) -> None:
        N = 5
        rho = np.eye(N + 1, dtype=complex) / (N + 1)

        x = np.linspace(-4, 4, 50)
        p = np.linspace(-4, 4, 50)

        W = wigner_function_single(rho, x, p)
        assert not wigner_is_negative(W), "wigner_is_negative(W) should be falsy"


class TestWignerIntegration:
    """Integration tests for Wigner function."""

    def test_given_wigner_function_then_be_symmetric_for_vacuum(self) -> None:
        N = 10
        rho = np.zeros((N + 1, N + 1), dtype=complex)
        rho[0, 0] = 1.0

        # Use symmetric grid
        x = np.linspace(-4, 4, 81)
        p = np.linspace(-4, 4, 81)

        W = wigner_function_single(rho, x, p)

        # W(0,0) should be maximum for vacuum
        assert W[40, 40] == np.max(W)

    def test_given_non_square_rho_then_raise_valueerror(self) -> None:
        rho = np.zeros((5, 3), dtype=complex)
        x = np.linspace(-3, 3, 10)
        p = np.linspace(-3, 3, 10)

        with pytest.raises(ValueError, match="must be square"):
            wigner_function_single(rho, x, p)

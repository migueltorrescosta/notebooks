"""
Tests for Hybrid MZI Embedding module.

Physical Validation:
- Embedding preserves norm: ‖embedded‖ = ‖hybrid‖
- Beam splitter is unitary
- Phase shift is unitary
- MZI evolution preserves probability
- QFI is non-negative
"""

import numpy as np
import pytest

from .hybrid_mzi import (
    compute_wigner_for_state,
    embed_hybrid_in_mzi,
    evolve_hybrid_mzi,
    extract_oscillator_density,
    mzi_beam_splitter,
    mzi_marginal_photon_probs,
    mzi_output_probabilities,
    mzi_phase_generator,
    mzi_phase_shift,
    qfi_hybrid_mzi,
)
from .hybrid_system import (
    hybrid_coherent_state,
    hybrid_vacuum_state,
    validate_hybrid_unitary,
)

# =============================================================================
# Test Embedding
# =============================================================================


class TestEmbedding:
    """Test hybrid state embedding into MZI space."""

    def test_embedded_hybrid_1_pure_state(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)

        norm_hybrid = np.sum(np.abs(state) ** 2)
        norm_embedded = np.sum(np.abs(embedded) ** 2)

        assert norm_embedded == pytest.approx(norm_hybrid, abs=1e-10), (
            "Expected norm_embedded == pytest.approx(norm_hybrid, abs=1e-10)"
        )

    def test_embedded_state_should_have_correct_dimension_pure_state(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)

        expected_dim = 2 * (N + 1) ** 2
        assert embedded.shape == (expected_dim,), (
            "Expected embedded.shape == (expected_dim,)"
        )

    def test_0_0_should_only_have_amplitude_at_n1_0_n2_0_s_0(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)

        # Index for (n1=0, n2=0, s=0): 0*(N+1)*2 + 0 = 0
        assert embedded[0] == pytest.approx(1.0, abs=1e-10), (
            "Expected embedded[0] == pytest.approx(1.0, abs=1e-10)"
        )
        # All other elements should be zero
        assert np.sum(np.abs(embedded[1:]) ** 2) == pytest.approx(0.0, abs=1e-10), (
            "Expected np.sum(np.abs(embedded[1:]) ** 2) == pytest.approx(0.0, abs=1e-10)"
        )

    # ------------------------------------------------------------------ #
    # Density matrix embedding tests
    # ------------------------------------------------------------------ #

    def test_embedded_density_matrix_should_have_correct_2d_shape(self) -> None:
        N = 5
        dim_mzi = 2 * (N + 1) ** 2

        state = hybrid_vacuum_state(N, spin_state="down")
        rho = np.outer(state, state.conj())
        embedded = embed_hybrid_in_mzi(rho, N)

        assert embedded.shape == (dim_mzi, dim_mzi), (
            "Expected embedded.shape == (dim_mzi, dim_mzi)"
        )

    def test_tr_embedded_tr_hybrid_1(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        rho = np.outer(state, state.conj())

        embedded = embed_hybrid_in_mzi(rho, N)

        assert np.trace(embedded) == pytest.approx(1.0, abs=1e-10), (
            "Expected np.trace(embedded) == pytest.approx(1.0, abs=1e-10)"
        )

    def test_embedded_should_be_hermitian(self) -> None:
        N = 5
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        rho = np.outer(state, state.conj())

        embedded = embed_hybrid_in_mzi(rho, N)

        assert embedded == pytest.approx(embedded.conj().T, abs=1e-10), (
            "Expected embedded == pytest.approx(embedded.conj().T, abs=1e-10)"
        )

    def test_embedded_should_be_positive_semidefinite(self) -> None:
        N = 5
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        rho = np.outer(state, state.conj())

        embedded = embed_hybrid_in_mzi(rho, N)

        eigenvalues = np.linalg.eigvalsh(embedded)
        assert np.all(eigenvalues >= -1e-10), "Expected np.all(eigenvalues >= -1e-10)"

    def test_0_0_should_only_have_nonzero_entry_at_0_0(self) -> None:
        N = 5
        dim_hybrid = 2 * (N + 1)

        rho = np.zeros((dim_hybrid, dim_hybrid), dtype=complex)
        rho[0, 0] = 1.0  # |0,↓⟩⟨0,↓|

        embedded = embed_hybrid_in_mzi(rho, N)

        assert embedded[0, 0] == pytest.approx(1.0, abs=1e-10), (
            "Expected embedded[0, 0] == pytest.approx(1.0, abs=1e-10)"
        )
        all_others = np.abs(embedded).sum() - np.abs(embedded[0, 0])
        assert all_others == pytest.approx(0.0, abs=1e-10), (
            "Expected all_others == pytest.approx(0.0, abs=1e-10)"
        )

    def test_embedding_should_match_emb_emb(self) -> None:
        N = 5
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        rho = np.outer(state, state.conj())

        # Embed pure state
        psi_emb = embed_hybrid_in_mzi(state, N)
        rho_from_pure = np.outer(psi_emb, psi_emb.conj())

        # Embed density matrix
        rho_emb = embed_hybrid_in_mzi(rho, N)

        assert rho_emb == pytest.approx(rho_from_pure, abs=1e-10), (
            "Expected rho_emb == pytest.approx(rho_from_pure, abs=1e-10)"
        )

    def test_off_diagonal_i_j_should_map_to_correct_embedded_position(self) -> None:
        N = 3
        dim_hybrid = 2 * (N + 1)

        # Coherence between |0,↓⟩ and |1,↓⟩
        rho = np.zeros((dim_hybrid, dim_hybrid), dtype=complex)
        rho[0, 2] = 0.5 + 0.5j  # |0,↓⟩⟨1,↓|

        embedded = embed_hybrid_in_mzi(rho, N)

        # |0,↓⟩ → embedded index 0*(N+1)*2 + 0 = 0
        # |1,↓⟩ → embedded index 1*(N+1)*2 + 0 = 2*(N+1) = 8 for N=3
        idx_0 = 0 * (N + 1) * 2 + 0
        idx_1 = 1 * (N + 1) * 2 + 0
        assert embedded[idx_0, idx_1] == pytest.approx(0.5 + 0.5j, abs=1e-10), (
            "Expected embedded[idx_0, idx_1] == pytest.approx(0.5 + 0.5j, abs=1e-10)"
        )

    def test_non_square_density_matrix_should_raise_valueerror(self) -> None:
        N = 5
        bad_rho = np.zeros((4, 6), dtype=complex)
        with pytest.raises(ValueError, match="must have shape"):
            embed_hybrid_in_mzi(bad_rho, N)

    def test_embedding_should_work_for_larger_n(self) -> None:
        N = 10
        state = hybrid_vacuum_state(N, spin_state="up")
        rho = np.outer(state, state.conj())

        embedded = embed_hybrid_in_mzi(rho, N)

        dim_mzi = 2 * (N + 1) ** 2
        assert embedded.shape == (dim_mzi, dim_mzi), (
            "Expected embedded.shape == (dim_mzi, dim_mzi)"
        )
        assert np.trace(embedded) == pytest.approx(1.0, abs=1e-10), (
            "Expected np.trace(embedded) == pytest.approx(1.0, abs=1e-10)"
        )
        assert embedded == pytest.approx(embedded.conj().T, abs=1e-10), (
            "Expected embedded == pytest.approx(embedded.conj().T, abs=1e-10)"
        )


# =============================================================================
# Test MZI Operators
# =============================================================================


class TestMZIOperators:
    """Test MZI operator properties."""

    def test_bs_should_be_unitary(self) -> None:
        N = 5
        bs = mzi_beam_splitter(N, theta=np.pi / 4)
        assert validate_hybrid_unitary(bs, tol=1e-8), (
            "Condition failed: validate_hybrid_unitary(bs, tol=1e-8)"
        )

    def test_phase_shift_should_be_unitary(self) -> None:
        N = 5
        ps = mzi_phase_shift(N, phi=0.5)
        assert validate_hybrid_unitary(ps, tol=1e-8), (
            "Condition failed: validate_hybrid_unitary(ps, tol=1e-8)"
        )

    def test_phase_generator_should_be_hermitian(self) -> None:
        N = 5
        G = mzi_phase_generator(N)
        assert pytest.approx(G.conj().T, abs=1e-10) == G, (
            "Expected G == pytest.approx(G.conj().T, abs=1e-10)"
        )

    def test_g_n_i_should_be_diagonal(self) -> None:
        N = 5
        G = mzi_phase_generator(N)
        # G should be diagonal
        assert G - np.diag(np.diag(G)) == pytest.approx(0, abs=1e-10), (
            "Expected G - np.diag(np.diag(G)) == pytest.approx(0, abs=1e-10)"
        )


# =============================================================================
# Test MZI Evolution
# =============================================================================


class TestMZIEvolution:
    """Test evolution through MZI."""

    def test_output_state_should_be_normalized(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)

        norm = np.sum(np.abs(output) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-6), (
            "Expected norm == pytest.approx(1.0, abs=1e-6)"
        )

    def test_with_0_vacuum_should_remain_same_up_to_bs_transformations(self) -> None:
        N = 3
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.0)

        # Just check it's normalized
        norm = np.sum(np.abs(output) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-6), (
            "Expected norm == pytest.approx(1.0, abs=1e-6)"
        )

    def test_output_should_have_correct_shape(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=1.0)

        expected_dim = 2 * (N + 1) ** 2
        assert output.shape == (expected_dim,), (
            "Expected output.shape == (expected_dim,)"
        )


# =============================================================================
# Test Output Probabilities
# =============================================================================


class TestOutputProbabilities:
    """Test probability computations."""

    def test_sum_of_all_amplitudes_should_be_1(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)
        probs = mzi_output_probabilities(output, N)

        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6), (
            "Expected np.sum(probs) == pytest.approx(1.0, abs=1e-6)"
        )

    def test_marginal_probabilities_should_sum_to_1(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)
        P1, P2 = mzi_marginal_photon_probs(output, N)

        assert np.sum(P1) == pytest.approx(1.0, abs=1e-6), (
            "Expected np.sum(P1) == pytest.approx(1.0, abs=1e-6)"
        )
        assert np.sum(P2) == pytest.approx(1.0, abs=1e-6), (
            "Expected np.sum(P2) == pytest.approx(1.0, abs=1e-6)"
        )

    def test_marginal_arrays_should_have_length_n_1(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)
        P1, P2 = mzi_marginal_photon_probs(output, N)

        assert len(P1) == N + 1, "Expected len(P1) == N + 1"
        assert len(P2) == N + 1, "Expected len(P2) == N + 1"


# =============================================================================
# Test QFI Computation
# =============================================================================


class TestQFIHybridMZI:
    """Test Quantum Fisher Information computation."""

    def test_qfi_should_be_non_negative(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        fq = qfi_hybrid_mzi(state, N)

        assert fq >= 0.0, "Expected fq >= 0.0"

    def test_vacuum_input_should_give_zero_qfi_no_phase_sensitivity(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        fq = qfi_hybrid_mzi(state, N)

        # Vacuum has no phase information
        assert fq == pytest.approx(0.0, abs=1e-6), (
            "Expected fq == pytest.approx(0.0, abs=1e-6)"
        )

    def test_more_photons_should_give_higher_qfi(self) -> None:
        N = 10
        # Coherent state with |α|² = 1
        state1 = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        # Coherent state with |α|² = 4
        state4 = hybrid_coherent_state(N, alpha=2.0 + 0j, spin_state="down")

        fq1 = qfi_hybrid_mzi(state1, N)
        fq4 = qfi_hybrid_mzi(state4, N)

        # More photons → higher QFI
        assert fq4 > fq1, "Expected fq4 > fq1"


# =============================================================================
# Test Density Matrix Extraction
# =============================================================================


class TestDensityExtraction:
    """Test oscillator density matrix extraction."""

    def test_vacuum_should_give_0_0(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        rho_osc = extract_oscillator_density(state, N)

        # Should be |0⟩⟨0|
        assert rho_osc[0, 0] == pytest.approx(1.0), (
            "Expected rho_osc[0, 0] == pytest.approx(1.0)"
        )
        assert np.sum(np.abs(rho_osc[1:, :])) == pytest.approx(0.0, abs=1e-10), (
            "Expected np.sum(np.abs(rho_osc[1:, :])) == pytest.approx(0.0, abs=1e-10)"
        )

    def test_trace_of_extracted_density_should_be_1(self) -> None:
        N = 5
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        rho_osc = extract_oscillator_density(state, N)

        trace = np.trace(rho_osc).real
        # Relaxed tolerance due to numerical truncation
        assert trace == pytest.approx(1.0, rel=1e-3, abs=1e-3), (
            "Expected trace == pytest.approx(1.0, rel=1e-3, abs=1e-3)"
        )


# =============================================================================
# Test Wigner Computation
# =============================================================================


class TestWignerComputation:
    """Test Wigner function computation."""

    def test_wigner_output_should_have_correct_shape(self) -> None:
        N = 10
        state = hybrid_vacuum_state(N, spin_state="down")
        x, p, W = compute_wigner_for_state(state, N, x_max=3.0, n_points=50)

        assert x.shape == (50,), "Expected x.shape == (50,)"
        assert p.shape == (50,), "Expected p.shape == (50,)"
        assert W.shape == (50, 50), "Expected W.shape == (50, 50)"

    def test_vacuum_wigner_should_be_positive(self) -> None:
        N = 10
        state = hybrid_vacuum_state(N, spin_state="down")
        _, _, W = compute_wigner_for_state(state, N, x_max=3.0, n_points=50)

        assert np.min(W) >= -1e-10, "Expected np.min(W) >= -1e-10"

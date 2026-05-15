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
        ps = mzi_phase_shift(N, phi=0.5)
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
        output = evolve_hybrid_mzi(state, N, phi=0.5)
        norm = np.sum(np.abs(output) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_given_zero_phase_then_normalized(self) -> None:
        N = 3
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.0)
        norm = np.sum(np.abs(output) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_given_evolution_then_correct_shape(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=1.0)
        expected_dim = 2 * (N + 1) ** 2
        assert output.shape == (expected_dim,)


class TestOutputProbabilities:
    def test_given_evolution_then_probabilities_sum_to_one(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)
        probs = mzi_output_probabilities(output, N)
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_given_evolution_then_marginals_sum_to_one(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)
        P1, P2 = mzi_marginal_photon_probs(output, N)
        assert np.sum(P1) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(P2) == pytest.approx(1.0, abs=1e-6)

    def test_given_evolution_then_marginals_have_correct_length(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)
        P1, P2 = mzi_marginal_photon_probs(output, N)
        assert len(P1) == N + 1
        assert len(P2) == N + 1


class TestQFIHybridMZI:
    def test_given_vacuum_then_non_negative(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        fq = qfi_hybrid_mzi(state, N)
        assert fq >= 0.0

    def test_given_vacuum_then_qfi_zero(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        fq = qfi_hybrid_mzi(state, N)
        assert fq == pytest.approx(0.0, abs=1e-6)

    def test_given_more_photons_then_qfi_higher(self) -> None:
        N = 10
        state1 = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        state4 = hybrid_coherent_state(N, alpha=2.0 + 0j, spin_state="down")
        fq1 = qfi_hybrid_mzi(state1, N)
        fq4 = qfi_hybrid_mzi(state4, N)
        assert fq4 > fq1


class TestDensityExtraction:
    def test_given_vacuum_state_then_extracted_density_is_vacuum(self) -> None:
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        rho_osc = extract_oscillator_density(state, N)
        assert rho_osc[0, 0] == pytest.approx(1.0)
        assert np.sum(np.abs(rho_osc[1:, :])) == pytest.approx(0.0, abs=1e-10)

    def test_given_coherent_state_then_extracted_density_has_unity_trace(self) -> None:
        N = 5
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        rho_osc = extract_oscillator_density(state, N)
        trace = np.trace(rho_osc).real
        assert trace == pytest.approx(1.0, rel=1e-3, abs=1e-3)


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

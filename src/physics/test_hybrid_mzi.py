"""
Tests for MZI Embedding for Hybrid Oscillator-Spin System.

Covers: embed_hybrid_in_mzi, mzi_beam_splitter, mzi_phase_shift,
mzi_phase_generator, evolve_hybrid_mzi, mzi_output_probabilities,
mzi_marginal_photon_probs, qfi_hybrid_mzi, extract_oscillator_density,
and all Wigner function helpers.
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
    wigner_from_hybrid_state,
    wigner_function_single,
    wigner_is_negative,
)
from .hybrid_system import hybrid_coherent_state, hybrid_vacuum_state


class TestQFIHybridMZI:
    """Tests for QFI computation on the hybrid MZI."""

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

    def test_given_coherent_state_then_qfi_matches_var_n_plus_mean_n(self) -> None:
        """For a coherent state, F_Q = var(n) + mean(n)."""
        N = 10
        state = hybrid_coherent_state(N, alpha=2.0 + 0j, spin_state="down")
        fq = qfi_hybrid_mzi(state, N)
        # For |alpha=2>: mean_n ~ 4, var_n ~ 4, so F_Q ~ 8
        assert fq == pytest.approx(8.0, rel=0.15)

    def test_given_fock_state_then_qfi_matches_n(self) -> None:
        """For a Fock state |n,down>, Var(n)=0 so F_Q = n."""
        N = 5
        state = np.zeros(2 * (N + 1), dtype=complex)
        state[2 * 2] = 1.0  # |2,down> at index 2*2 + 0
        fq = qfi_hybrid_mzi(state, N)
        assert fq == pytest.approx(2.0, abs=1e-10)

    def test_given_spin_up_state_then_qfi_matches(self) -> None:
        """QFI should work regardless of spin state."""
        N = 4
        state = hybrid_vacuum_state(N, spin_state="up")
        fq = qfi_hybrid_mzi(state, N)
        assert fq == pytest.approx(0.0, abs=1e-10)


class TestDensityExtraction:
    """Tests for extracting oscillator density from hybrid state."""

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

    def test_given_up_spin_trace_one(self) -> None:
        N = 3
        state = hybrid_vacuum_state(N, spin_state="up")
        rho_osc = extract_oscillator_density(state, N)
        assert np.trace(rho_osc).real == pytest.approx(1.0, abs=1e-10)
        assert rho_osc[0, 0] == pytest.approx(1.0)

    def test_given_coherent_up_spin_trace_matches_state_norm(self) -> None:
        N = 3
        state = hybrid_coherent_state(N, alpha=1.5 + 0j, spin_state="up")
        input_norm = np.sum(np.abs(state) ** 2)
        rho_osc = extract_oscillator_density(state, N)
        assert np.trace(rho_osc).real == pytest.approx(input_norm, abs=1e-10)

    def test_output_shape(self) -> None:
        N = 4
        state = hybrid_vacuum_state(N, spin_state="down")
        rho = extract_oscillator_density(state, N)
        assert rho.shape == (N + 1, N + 1)


class TestEmbedHybridInMZI:
    """Tests for embedding hybrid state into two-mode MZI space."""

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_pure_state_shape(self, N: int) -> None:
        dim_mzi = 2 * (N + 1) ** 2
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)
        assert embedded.shape == (dim_mzi,)

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_pure_state_norm_preserved(self, N: int) -> None:
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        input_norm = np.sum(np.abs(state) ** 2)
        embedded = embed_hybrid_in_mzi(state, N)
        assert np.sum(np.abs(embedded) ** 2) == pytest.approx(input_norm, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2])
    def test_pure_vacuum_only_first_entry(self, N: int) -> None:
        """Vacuum |0,down> maps to the first entry (n1=0, n2=0, s=0) only."""
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)
        assert embedded[0] == pytest.approx(1.0, abs=1e-10)
        nonzero = np.where(np.abs(embedded) > 1e-10)[0]
        assert len(nonzero) == 1
        assert nonzero[0] == 0

    @pytest.mark.parametrize("N", [1, 2])
    def test_density_matrix_shape(self, N: int) -> None:
        dim_mzi = 2 * (N + 1) ** 2
        state = hybrid_vacuum_state(N, spin_state="down")
        rho = np.outer(state, state.conj())
        embedded = embed_hybrid_in_mzi(rho, N)
        assert embedded.shape == (dim_mzi, dim_mzi)

    @pytest.mark.parametrize("N", [1, 2])
    def test_density_matrix_trace_preserved(self, N: int) -> None:
        state = hybrid_coherent_state(N, alpha=0.5 + 0j, spin_state="up")
        rho = np.outer(state, state.conj())
        embedded = embed_hybrid_in_mzi(rho, N)
        assert np.trace(embedded).real == pytest.approx(np.trace(rho).real, abs=1e-10)

    def test_raises_on_wrong_pure_shape(self) -> None:
        with pytest.raises(ValueError):
            embed_hybrid_in_mzi(np.zeros(10, dtype=complex), N=2)

    def test_raises_on_wrong_dm_shape(self) -> None:
        with pytest.raises(ValueError):
            embed_hybrid_in_mzi(np.zeros((10, 10), dtype=complex), N=2)

    def test_raises_on_wrong_ndim(self) -> None:
        with pytest.raises(ValueError):
            embed_hybrid_in_mzi(np.zeros((2, 2, 2), dtype=complex), N=2)


class TestMZIBeamSplitter:
    """Tests for the MZI beam splitter unitary."""

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_unitarity(self, N: int) -> None:
        bs = mzi_beam_splitter(N)
        dim = bs.shape[0]
        assert np.allclose(bs @ bs.conj().T, np.eye(dim), atol=1e-10)

    @pytest.mark.parametrize("theta", [0.0, np.pi / 8, np.pi / 4, np.pi / 3])
    def test_unitarity_different_angles(self, theta: float) -> None:
        bs = mzi_beam_splitter(N=2, theta=theta)
        dim = bs.shape[0]
        assert np.allclose(bs @ bs.conj().T, np.eye(dim), atol=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_norm_preservation_on_vacuum(self, N: int) -> None:
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)
        bs = mzi_beam_splitter(N)
        out = bs @ embedded
        assert np.sum(np.abs(out) ** 2) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2])
    def test_norm_preservation_on_coherent(self, N: int) -> None:
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)
        input_norm = np.sum(np.abs(embedded) ** 2)
        bs = mzi_beam_splitter(N)
        out = bs @ embedded
        assert np.sum(np.abs(out) ** 2) == pytest.approx(input_norm, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2])
    def test_vacuum_unchanged(self, N: int) -> None:
        """|0,0> is an eigenstate of the BS generator (G|0,0>=0) at any angle."""
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)
        bs = mzi_beam_splitter(N)
        out = bs @ embedded
        assert np.abs(out[0]) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2])
    def test_correct_dimensions(self, N: int) -> None:
        dim_mzi = 2 * (N + 1) ** 2
        bs = mzi_beam_splitter(N)
        assert bs.shape == (dim_mzi, dim_mzi)


class TestMZIPhaseShift:
    """Tests for the MZI phase shift unitary."""

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_unitarity(self, N: int) -> None:
        ps = mzi_phase_shift(N, phi_phase=np.pi / 3)
        dim = ps.shape[0]
        assert np.allclose(ps @ ps.conj().T, np.eye(dim), atol=1e-10)

    def test_identity_at_zero_phase(self) -> None:
        N = 2
        ps = mzi_phase_shift(N, phi_phase=0.0)
        dim = ps.shape[0]
        assert np.allclose(ps, np.eye(dim), atol=1e-10)

    @pytest.mark.parametrize("phi_phase", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_diagonal_phases(self, phi_phase: float) -> None:
        """Diagonal entry for |n1,n2,s> should be exp(i phi_phase * n1)."""
        N = 2
        dim_osc = N + 1
        ps = mzi_phase_shift(N, phi_phase=phi_phase)
        # Check |1,0,down> at index (1*dim_osc + 0)*2 + 0 = 2*dim_osc
        idx = 2 * dim_osc
        expected = np.exp(1j * phi_phase * 1)
        assert ps[idx, idx] == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2])
    def test_correct_dimensions(self, N: int) -> None:
        dim_mzi = 2 * (N + 1) ** 2
        ps = mzi_phase_shift(N, phi_phase=0.5)
        assert ps.shape == (dim_mzi, dim_mzi)


class TestMZIPhaseGenerator:
    """Tests for the MZI phase generator G = n1 ⊗ I."""

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_shape(self, N: int) -> None:
        dim_mzi = 2 * (N + 1) ** 2
        G = mzi_phase_generator(N)
        assert G.shape == (dim_mzi, dim_mzi)

    def test_hermiticity(self) -> None:
        G = mzi_phase_generator(N=2)
        assert np.allclose(G, G.conj().T, atol=1e-10)

    @pytest.mark.parametrize("N", [1, 2])
    def test_diagonal_values(self, N: int) -> None:
        """Generator diagonal = n1 (photon number in mode 1)."""
        dim_osc = N + 1
        G = mzi_phase_generator(N)
        for n1 in range(dim_osc):
            for n2 in range(dim_osc):
                for s in range(2):
                    idx = (n1 * dim_osc + n2) * 2 + s
                    assert G[idx, idx] == pytest.approx(n1, abs=1e-10)


class TestEvolveHybridMZI:
    """Tests for the full MZI circuit evolution."""

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_norm_conservation(self, N: int) -> None:
        state = hybrid_coherent_state(N, alpha=0.8 + 0j, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)
        input_norm = np.sum(np.abs(embedded) ** 2)
        evolved = evolve_hybrid_mzi(state, N, phi_phase=np.pi / 4)
        assert np.sum(np.abs(evolved) ** 2) == pytest.approx(input_norm, abs=1e-10)

    @pytest.mark.parametrize("N", [1, 2])
    def test_vacuum_unchanged_at_zero_phase(self, N: int) -> None:
        """Vacuum at phi=0 evolves to itself: BS^2 = I on vacuum."""
        state = hybrid_vacuum_state(N, spin_state="down")
        evolved = evolve_hybrid_mzi(state, N, phi_phase=0.0)
        embedded = embed_hybrid_in_mzi(state, N)
        assert np.allclose(evolved, embedded, atol=1e-10)

    @pytest.mark.parametrize("N", [1, 2])
    def test_different_phases_give_different_outputs(self, N: int) -> None:
        state = hybrid_coherent_state(N, alpha=0.5 + 0j, spin_state="up")
        out1 = evolve_hybrid_mzi(state, N, phi_phase=0.0)
        out2 = evolve_hybrid_mzi(state, N, phi_phase=np.pi / 2)
        overlap = np.abs(np.dot(out1.conj(), out2))
        assert overlap < 1.0 - 1e-6

    def test_zero_theta_bs_gives_phase_shift_only(self) -> None:
        """With theta=0, BS=I so evolution reduces to phase shift."""
        N = 2
        state = hybrid_coherent_state(N, alpha=0.5 + 0j, spin_state="down")
        evolved = evolve_hybrid_mzi(state, N, phi_phase=np.pi / 3, theta=0.0)
        embedded = embed_hybrid_in_mzi(state, N)
        ps = mzi_phase_shift(N, phi_phase=np.pi / 3)
        expected = ps @ embedded
        assert np.allclose(evolved, expected, atol=1e-10)


class TestMZIOutputProbabilities:
    """Tests for output probability functions."""

    def test_probabilities_sum_to_evolved_norm(self) -> None:
        N = 2
        state = hybrid_coherent_state(N, alpha=0.8 + 0j, spin_state="down")
        evolved = evolve_hybrid_mzi(state, N, phi_phase=np.pi / 4)
        evolved_norm = np.sum(np.abs(evolved) ** 2)
        probs = mzi_output_probabilities(evolved, N)
        assert np.sum(probs) == pytest.approx(evolved_norm, abs=1e-10)

    def test_probabilities_non_negative(self) -> None:
        N = 2
        state = hybrid_coherent_state(N, alpha=0.8 + 0j, spin_state="down")
        evolved = evolve_hybrid_mzi(state, N, phi_phase=np.pi / 4)
        probs = mzi_output_probabilities(evolved, N)
        assert np.all(probs >= -1e-10)

    def test_marginals_sum_to_evolved_norm(self) -> None:
        N = 2
        state = hybrid_coherent_state(N, alpha=0.8 + 0j, spin_state="down")
        evolved = evolve_hybrid_mzi(state, N, phi_phase=np.pi / 4)
        evolved_norm = np.sum(np.abs(evolved) ** 2)
        P1, P2 = mzi_marginal_photon_probs(evolved, N)
        assert np.sum(P1) == pytest.approx(evolved_norm, abs=1e-10)
        assert np.sum(P2) == pytest.approx(evolved_norm, abs=1e-10)

    def test_marginals_non_negative(self) -> None:
        N = 2
        state = hybrid_coherent_state(N, alpha=0.8 + 0j, spin_state="down")
        evolved = evolve_hybrid_mzi(state, N, phi_phase=np.pi / 4)
        P1, P2 = mzi_marginal_photon_probs(evolved, N)
        assert np.all(P1 >= -1e-10)
        assert np.all(P2 >= -1e-10)

    @pytest.mark.parametrize("N", [1, 2])
    def test_vacuum_marginals(self, N: int) -> None:
        """Vacuum at phi=0: output = |0,0,down>, so P(0)=1 for both modes."""
        state = hybrid_vacuum_state(N, spin_state="down")
        evolved = evolve_hybrid_mzi(state, N, phi_phase=0.0)
        P1, P2 = mzi_marginal_photon_probs(evolved, N)
        assert P1[0] == pytest.approx(1.0, abs=1e-10)
        assert P2[0] == pytest.approx(1.0, abs=1e-10)


class TestWignerFunctions:
    """Tests for Wigner function computations."""

    def test_wigner_function_single_raises_on_non_square(self) -> None:
        with pytest.raises(ValueError):
            wigner_function_single(
                np.zeros((3, 4), dtype=complex),
                np.linspace(-5, 5, 10),
                np.linspace(-5, 5, 10),
            )

    def test_wigner_function_single_output_shape(self) -> None:
        N = 3
        state = hybrid_vacuum_state(N, spin_state="down")
        rho_osc = extract_oscillator_density(state, N)
        x = np.linspace(-5, 5, 20)
        p = np.linspace(-5, 5, 20)
        W = wigner_function_single(rho_osc, x, p)
        assert W.shape == (20, 20)

    def test_wigner_from_hybrid_state_raises_on_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            wigner_from_hybrid_state(
                np.zeros(10, dtype=complex),
                N=2,
                x_range=np.linspace(-5, 5, 10),
                p_range=np.linspace(-5, 5, 10),
            )

    def test_wigner_from_hybrid_state_raises_on_invalid_spin(self) -> None:
        N = 2
        state = hybrid_vacuum_state(N, spin_state="down")
        with pytest.raises(ValueError):
            wigner_from_hybrid_state(
                state,
                N,
                x_range=np.linspace(-5, 5, 10),
                p_range=np.linspace(-5, 5, 10),
                spin_component="invalid",
            )

    def test_wigner_from_hybrid_state_shape(self) -> None:
        N = 2
        state = hybrid_vacuum_state(N, spin_state="down")
        x = np.linspace(-5, 5, 15)
        p = np.linspace(-5, 5, 15)
        W = wigner_from_hybrid_state(state, N, x, p)
        assert W.shape == (15, 15)

    def test_wigner_from_hybrid_state_up_spin(self) -> None:
        N = 2
        state = hybrid_vacuum_state(N, spin_state="up")
        x = np.linspace(-5, 5, 10)
        p = np.linspace(-5, 5, 10)
        W = wigner_from_hybrid_state(state, N, x, p, spin_component="up")
        assert W.shape == (10, 10)

    def test_compute_wigner_for_state_output(self) -> None:
        N = 2
        state = hybrid_vacuum_state(N, spin_state="down")
        X, P, W = compute_wigner_for_state(state, N, x_max=5.0, n_points=20)
        assert X.shape == (20,)
        assert P.shape == (20,)
        assert W.shape == (20, 20)

    def test_wigner_is_negative_detects_negativity(self) -> None:
        W = np.array([[0.1, 0.0], [0.0, -0.05]])
        assert wigner_is_negative(W)

    def test_wigner_is_negative_returns_false_for_non_negative(self) -> None:
        W = np.array([[0.1, 0.0], [0.0, 0.05]])
        assert not wigner_is_negative(W)

    def test_wigner_is_negative_respects_tol(self) -> None:
        W = np.array([[-1e-12]])
        # Default tol=1e-10: -1e-12 > -1e-10 so not negative
        assert not wigner_is_negative(W)
        # With tol=1e-15: -1e-12 < -1e-15 so flagged negative
        assert wigner_is_negative(W, tol=1e-15)

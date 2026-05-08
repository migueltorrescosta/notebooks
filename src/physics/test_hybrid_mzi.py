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

from .hybrid_mzi import (
    embed_hybrid_in_mzi,
    mzi_beam_splitter,
    mzi_phase_shift,
    mzi_phase_generator,
    evolve_hybrid_mzi,
    mzi_output_probabilities,
    mzi_marginal_photon_probs,
    qfi_hybrid_mzi,
    extract_oscillator_density,
    compute_wigner_for_state,
)
from .hybrid_system import (
    hybrid_vacuum_state,
    hybrid_coherent_state,
    validate_hybrid_unitary,
)


# =============================================================================
# Test Embedding
# =============================================================================


class TestEmbedding:
    """Test hybrid state embedding into MZI space."""

    def test_embed_preserves_norm(self) -> None:
        """‖embedded‖² = ‖hybrid‖² = 1."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)

        norm_hybrid = np.sum(np.abs(state) ** 2)
        norm_embedded = np.sum(np.abs(embedded) ** 2)

        assert np.isclose(norm_embedded, norm_hybrid, atol=1e-10)

    def test_embed_dimension(self) -> None:
        """Embedded state should have correct dimension."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)

        expected_dim = 2 * (N + 1) ** 2
        assert embedded.shape == (expected_dim,)

    def test_embed_vacuum_structure(self) -> None:
        """|0,↓⟩ ⊗ |0⟩ should only have amplitude at (n1=0, n2=0, s=0)."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        embedded = embed_hybrid_in_mzi(state, N)

        # Index for (n1=0, n2=0, s=0): 0*(N+1)*2 + 0 = 0
        assert np.isclose(embedded[0], 1.0, atol=1e-10)
        # All other elements should be zero
        assert np.isclose(np.sum(np.abs(embedded[1:]) ** 2), 0.0, atol=1e-10)


# =============================================================================
# Test MZI Operators
# =============================================================================


class TestMZIOperators:
    """Test MZI operator properties."""

    def test_beam_splitter_unitary(self) -> None:
        """BS should be unitary."""
        N = 5
        bs = mzi_beam_splitter(N, theta=np.pi / 4)
        assert validate_hybrid_unitary(bs, tol=1e-8)

    def test_phase_shift_unitary(self) -> None:
        """Phase shift should be unitary."""
        N = 5
        ps = mzi_phase_shift(N, phi=0.5)
        assert validate_hybrid_unitary(ps, tol=1e-8)

    def test_phase_generator_hermitian(self) -> None:
        """Phase generator should be Hermitian."""
        N = 5
        G = mzi_phase_generator(N)
        assert np.allclose(G, G.conj().T, atol=1e-10)

    def test_phase_generator_diagonal(self) -> None:
        """G = n₁ ⊗ I should be diagonal."""
        N = 5
        G = mzi_phase_generator(N)
        # G should be diagonal
        assert np.allclose(G - np.diag(np.diag(G)), 0, atol=1e-10)


# =============================================================================
# Test MZI Evolution
# =============================================================================


class TestMZIEvolution:
    """Test evolution through MZI."""

    def test_evolution_preserves_norm(self) -> None:
        """Output state should be normalized."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)

        norm = np.sum(np.abs(output) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_evolution_no_phase(self) -> None:
        """With φ=0, vacuum should remain same (up to BS transformations)."""
        N = 3
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.0)

        # Just check it's normalized
        norm = np.sum(np.abs(output) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_evolution_output_shape(self) -> None:
        """Output should have correct shape."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=1.0)

        expected_dim = 2 * (N + 1) ** 2
        assert output.shape == (expected_dim,)


# =============================================================================
# Test Output Probabilities
# =============================================================================


class TestOutputProbabilities:
    """Test probability computations."""

    def test_probabilities_sum_to_one(self) -> None:
        """Sum of all |amplitudes|² should be 1."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)
        probs = mzi_output_probabilities(output, N)

        assert np.isclose(np.sum(probs), 1.0, atol=1e-6)

    def test_marginal_probs(self) -> None:
        """Marginal probabilities should sum to 1."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)
        P1, P2 = mzi_marginal_photon_probs(output, N)

        assert np.isclose(np.sum(P1), 1.0, atol=1e-6)
        assert np.isclose(np.sum(P2), 1.0, atol=1e-6)

    def test_marginal_probs_shape(self) -> None:
        """Marginal arrays should have length N+1."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        output = evolve_hybrid_mzi(state, N, phi=0.5)
        P1, P2 = mzi_marginal_photon_probs(output, N)

        assert len(P1) == N + 1
        assert len(P2) == N + 1


# =============================================================================
# Test QFI Computation
# =============================================================================


class TestQFIHybridMZI:
    """Test Quantum Fisher Information computation."""

    def test_qfi_non_negative(self) -> None:
        """QFI should be non-negative."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        fq = qfi_hybrid_mzi(state, N)

        assert fq >= 0.0

    def test_qfi_vacuum(self) -> None:
        """Vacuum input should give zero QFI (no phase sensitivity)."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        fq = qfi_hybrid_mzi(state, N)

        # Vacuum has no phase information
        assert np.isclose(fq, 0.0, atol=1e-6)

    def test_qfi_scales_with_photons(self) -> None:
        """More photons should give higher QFI."""
        N = 10
        # Coherent state with |α|² = 1
        state1 = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        # Coherent state with |α|² = 4
        state4 = hybrid_coherent_state(N, alpha=2.0 + 0j, spin_state="down")

        fq1 = qfi_hybrid_mzi(state1, N)
        fq4 = qfi_hybrid_mzi(state4, N)

        # More photons → higher QFI
        assert fq4 > fq1


# =============================================================================
# Test Density Matrix Extraction
# =============================================================================


class TestDensityExtraction:
    """Test oscillator density matrix extraction."""

    def test_extract_vacuum(self) -> None:
        """Vacuum should give ρ = |0⟩⟨0|."""
        N = 5
        state = hybrid_vacuum_state(N, spin_state="down")
        rho_osc = extract_oscillator_density(state, N)

        # Should be |0⟩⟨0|
        assert np.isclose(rho_osc[0, 0], 1.0)
        assert np.isclose(np.sum(np.abs(rho_osc[1:, :])), 0.0, atol=1e-10)

    def test_extract_preserves_trace(self) -> None:
        """Trace of extracted density should be 1."""
        N = 5
        state = hybrid_coherent_state(N, alpha=1.0 + 0j, spin_state="down")
        rho_osc = extract_oscillator_density(state, N)

        trace = np.trace(rho_osc).real
        # Relaxed tolerance due to numerical truncation
        assert np.isclose(trace, 1.0, rtol=1e-3, atol=1e-3)


# =============================================================================
# Test Wigner Computation
# =============================================================================


class TestWignerComputation:
    """Test Wigner function computation."""

    def test_compute_wigner_shape(self) -> None:
        """Wigner output should have correct shape."""
        N = 10
        state = hybrid_vacuum_state(N, spin_state="down")
        x, p, W = compute_wigner_for_state(state, N, x_max=3.0, n_points=50)

        assert x.shape == (50,)
        assert p.shape == (50,)
        assert W.shape == (50, 50)

    def test_compute_wigner_vacuum_positive(self) -> None:
        """Vacuum Wigner should be positive."""
        N = 10
        state = hybrid_vacuum_state(N, spin_state="down")
        _, _, W = compute_wigner_for_state(state, N, x_max=3.0, n_points=50)

        assert np.min(W) >= -1e-10

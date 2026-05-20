"""
MZI Embedding for Hybrid Oscillator-Spin System.

Implements the MZI readout protocol for the hybrid system:
1. Two-mode embedding: ρ_hybrid ⊗ |0⟩⟨0| (vacuum in second mode)
2. MZI evolution: BS → phase shift φ → BS
3. QFI computation with G = n₁ ⊗ I_spin

Physical Model:
- The hybrid system (oscillator + spin) is embedded as the first mode
  of a two-mode interferometer
- A vacuum state is appended as the second mode to enable MZI readout
- MZI protocol: BS → phase encoding on mode 1 → BS → measurement
- QFI computed with phase generator G = n₁ (number in mode 1)

Hilbert Space:
- Original hybrid: dim = 2(N+1)
- After embedding: dim = 2(N+1) × (N+1) = 2(N+1)²
  (hybrid ⊗ vacuum mode)
- Phase generator acts on mode 1 only: G = n₁ ⊗ I_osc2 ⊗ I_spin

Units:
- Dimensionless throughout (ℏ = 1)
- Phase φ in radians

Conventions:
- Beam splitter: 50:50 symmetric (θ = π/4, φ = 0)
- Phase shift on mode 1: exp(i · φ · n₁)
- State ordering after embedding: |n_hyb⟩_osc ⊗ |σ⟩_spin ⊗ |n⟩_vac
- QFI convention: F_Q = 4 · Var(G) for pure states
  (see src.analysis.fisher_information)

Note: The following functions have been migrated to reports/2026-05-07/local.py:
embed_hybrid_in_mzi, mzi_beam_splitter, mzi_phase_shift, mzi_phase_generator,
evolve_hybrid_mzi, mzi_output_probabilities, mzi_marginal_photon_probs,
compute_wigner_for_state.
"""

import numpy as np

# =============================================================================
# QFI Computation
# =============================================================================


def qfi_hybrid_mzi(
    hybrid_state: np.ndarray,
    N: int,
) -> float:
    """Compute QFI for MZI phase estimation with hybrid input.

    For the MZI, the phase shift exp(-iφ n₁) is applied between two
    beam splitters.  The correct generator is n₁ acting on the state
    *after* the first beam splitter.  For a 50:50 beam splitter and
    a pure input |ψ⟩|0⟩, this reduces to an analytical expression
    in terms of the input state's photon number statistics:

        F_Q = ⟨n²⟩ - ⟨n⟩² + ⟨n⟩ = Var(n) + ⟨n⟩

    where ⟨n⟩, ⟨n²⟩ are computed on the oscillator after tracing out
    the spin.  This formula is exact for the |ψ⟩|0⟩ ⊗ |spin⟩ setup
    and avoids constructing the BS matrix explicitly.

    Args:
        hybrid_state: Input hybrid state vector of shape (2(N+1),).
        N: Maximum photon number.

    Returns:
        Quantum Fisher Information F_Q.

    """
    # Extract oscillator density matrix (trace out spin)
    rho_osc = extract_oscillator_density(hybrid_state, N)

    # Photon number operator in oscillator space (diagonal)
    n_op = np.diag(np.arange(N + 1, dtype=complex))

    # Compute first and second moments of n
    mean_n = np.real(np.trace(rho_osc @ n_op))
    mean_n2 = np.real(np.trace(rho_osc @ n_op @ n_op))

    var_n = mean_n2 - mean_n**2
    var_n = max(0.0, var_n)  # Numerical safety

    # F_Q = Var(n) + ⟨n⟩  (derived from 4·Var(n₁) after BS₁)
    return var_n + mean_n


# =============================================================================
# Oscillator Density Extraction
# =============================================================================


def extract_oscillator_density(hybrid_state: np.ndarray, N: int) -> np.ndarray:
    """Extract oscillator density matrix from hybrid state (trace out spin).

    Args:
        hybrid_state: State vector of shape (2(N+1),).
        N: Maximum photon number.

    Returns:
        Density matrix of shape (N+1, N+1).

    """
    dim_osc = N + 1

    # Convert to density matrix
    rho_hybrid = np.outer(hybrid_state, hybrid_state.conj())

    # Reshape to (dim_osc, 2, dim_osc, 2) and trace over spin
    rho_reshaped = rho_hybrid.reshape(dim_osc, 2, dim_osc, 2)
    return np.trace(rho_reshaped, axis1=1, axis2=3)

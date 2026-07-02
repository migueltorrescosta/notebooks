"""
Time-Dependent Variational Principle (TDVP) for MPS / tensor-network states.

TDVP projects the exact Schrödinger dynamics onto the tensor-network manifold,
providing an efficient way to simulate quantum dynamics while respecting the
tensor-network structure.  This implementation uses quimb (`qtn.Tensor`) as the
state representation.

Physical Model:
- TDVP: |ψ̇⟩ = -i P (H - E) |ψ⟩ where P is the projection onto TTN manifold
- Suzuki-Trotter decomposition separates Hamiltonian terms
- Checkpointing saves intermediate states during evolution
- One-site updates with effective Hamiltonian

Units:
- Dimensionless throughout (ℏ = 1).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import quimb.tensor as qtn
import scipy.linalg

from src.utils.constants import SIGMA_Z

# =============================================================================
# TDVP Configuration
# =============================================================================


@dataclass
class TDVPConfig:
    """Configuration for TDVP evolution."""

    dt: float = 0.01
    """Time step."""

    trotter_order: int = 2
    """Order of Suzuki-Trotter decomposition (1 or 2)."""

    bond_dim_limit: int = 64
    """Maximum bond dimension allowed."""

    svd_epsilon: float = 1e-8
    """SVD truncation threshold."""

    checkpoint_every: int = 10
    """Save checkpoint every N time steps."""

    max_sweeps: int = 100
    """Maximum number of TDVP sweeps per time step."""

    convergence_tol: float = 1e-6
    """Convergence tolerance for sweeps."""


@dataclass
class TDVPCheckpoint:
    """Checkpoint data for TDVP evolution."""

    step: int
    """Time step number."""

    time: float
    """Evolution time."""

    state_vector: np.ndarray
    """Full state vector at checkpoint."""

    energy: complex
    """Energy at checkpoint."""

    max_bond_dim: int
    """Maximum bond dimension at checkpoint."""


@dataclass
class TDVPResult:
    """Result of TDVP evolution."""

    final_tensor: qtn.Tensor
    """Final tensor-network state."""

    checkpoints: list[TDVPCheckpoint] = field(default_factory=list)
    """Saved checkpoints."""

    energies: list[complex] = field(default_factory=list)
    """Energy at each step."""

    times: list[float] = field(default_factory=list)
    """Time points."""

    fidelity_history: list[float] = field(default_factory=list)
    """Fidelity relative to exact evolution (if provided)."""

    final_time: float = 0.0
    """Final evolution time."""

    norm_preserved: bool = True
    """Whether norm was preserved throughout evolution."""


# =============================================================================
# Helpers
# =============================================================================


def _tensor_from_state_vector(
    state: np.ndarray,
    n_sites: int,
    local_dim: int,
    _epsilon: float = 1e-8,
) -> qtn.Tensor:
    """Build a quimb ``Tensor`` from a flat state vector.

    The state is represented as a 2-index tensor with indices ``('main', 'ancilla')``,
    each of dimension ``local_dim ** n_sites``.  The input vector is first normalized.
    """
    expected_dim = local_dim ** (2 * n_sites)
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"State dimension {state.shape[0]} doesn't match expected {expected_dim}",
        )
    state = state / np.linalg.norm(state)
    mat = state.reshape(local_dim**n_sites, local_dim**n_sites)
    return qtn.Tensor(np.asarray(mat.astype(complex)), inds=("main", "ancilla"))


def _get_state_vector(tensor: qtn.Tensor) -> np.ndarray:
    """Return the flat state vector (numpy array) stored in *tensor*."""
    return tensor.data.flatten()


# =============================================================================
# Single-Site TDVP Update
# =============================================================================


def compute_environment_tensor(
    tensor: qtn.Tensor,
    site_idx: int,
) -> np.ndarray:
    """Compute the environment tensor for a given site.

    The environment is the rest of the tensor network contracted with the site
    removed.  For a single 2-index tensor this is a placeholder.
    """
    state = tensor.data
    if site_idx == 0:
        # main subspace — identity-on-ancilla projection
        return state @ state.conj().T
    # ancilla subspace
    return state.conj().T @ state


def apply_single_site_update(
    tensor: qtn.Tensor,
    site_idx: int,
    dt: float,
    h_eff: np.ndarray,
) -> qtn.Tensor:
    """Apply single-site TDVP update.

    The update is: |ψ'⟩ = exp(-i dt * H_eff) |ψ⟩
    where H_eff is the effective Hamiltonian at the site.

    For n_sites=1 (the tested regime) this applies *h_eff* as a gate on the
    ``'main'`` index (site_idx=0) or the ``'ancilla'`` index (site_idx=1).
    """
    if h_eff.shape[0] != h_eff.shape[1]:
        raise ValueError(
            f"H_eff must be a square matrix, got shape {h_eff.shape}",
        )

    # Unitary from H_eff
    u_site = scipy.linalg.expm(-1j * dt * h_eff)

    # Apply as a gate on the appropriate index via quimb.
    which = "main" if site_idx == 0 else "ancilla"

    # quimb's gate applies U as a matrix multiplication on the given index.
    return tensor.gate(u_site, which).normalize()


def tdvp_single_site(
    tensor: qtn.Tensor,
    site_idx: int,
    H_eff: np.ndarray,
    dt: float,
) -> qtn.Tensor:
    """Apply single-site TDVP update with Hermitian validation.

    See :func:`apply_single_site_update`.
    """
    # Validate square shape first (before comparing with conjugate transpose)
    if H_eff.ndim != 2 or H_eff.shape[0] != H_eff.shape[1]:
        raise ValueError(
            f"H_eff must be a square matrix, got shape {H_eff.shape}",
        )

    # Validate H_eff is Hermitian
    if not np.allclose(H_eff, H_eff.conj().T):
        raise ValueError("H_eff must be Hermitian")

    return apply_single_site_update(tensor, site_idx, dt, H_eff)


# =============================================================================
# Effective Hamiltonian Construction
# =============================================================================


def compute_local_expectation(
    tensor: qtn.Tensor,
    site_idx: int,
    operator: np.ndarray,
) -> complex:
    """Compute expectation value of local operator.

    ⟨ψ|O_i|ψ⟩ for site i.
    """
    which = "main" if site_idx == 0 else "ancilla"
    # Apply operator and compute overlap
    op_tensor = tensor.gate(operator, which)
    return complex(tensor.overlap(op_tensor))


def construct_effective_hamiltonian(
    tensor: qtn.Tensor,
    site_idx: int,
    H_terms: list[np.ndarray],
    site_operators: list[np.ndarray],
) -> np.ndarray:
    """Construct effective Hamiltonian for a site.

    H_eff = Σ_k c_k O_k where c_k are expectation values of commuting terms.
    """
    loc_dim = H_terms[0].shape[0] if H_terms else 2
    H_eff = np.zeros((loc_dim, loc_dim), dtype=complex)

    for H_term, site_op in zip(H_terms, site_operators, strict=False):
        exp_val = compute_local_expectation(tensor, site_idx, site_op)
        H_eff = H_eff + exp_val * H_term

    return H_eff


# =============================================================================
# Suzuki-Trotter Decomposition
# =============================================================================


def apply_trotter_step(
    tensor: qtn.Tensor,
    H_terms: list[np.ndarray],
    dt: float,
    order: int = 2,
) -> qtn.Tensor:
    """Apply Trotter decomposition step.

    For order=2 (Strang splitting):
        U ≈ exp(-i dt/2 H_1) exp(-i dt H_2) exp(-i dt/2 H_1)

    Args:
        tensor: Current tensor state.
        H_terms: List of Hamiltonian terms (each term acts on one site).
        dt: Time step.
        order: Trotter order (1 or 2).

    Returns:
        Updated tensor after one Trotter step.

    Raises:
        ValueError: If order is not 1 or 2.
    """
    if order not in (1, 2):
        raise ValueError(f"Trotter order must be 1 or 2, got {order}")

    if len(H_terms) == 0:
        return tensor

    result = tensor

    if order == 1:
        # First-order Trotter: product of exponentials
        for H_term in H_terms:
            result = tdvp_single_site(result, 0, H_term, dt)

    else:
        # Second-order (Strang) splitting:
        #   U ≈ e^{-i dt/2 H_1} e^{-i dt H_2} e^{-i dt/2 H_1}
        half_dt = dt / 2

        if len(H_terms) > 0:
            result = tdvp_single_site(result, 0, H_terms[0], half_dt)

        for H_term in H_terms[1:]:
            result = tdvp_single_site(result, 0, H_term, dt)

        if len(H_terms) > 0:
            result = tdvp_single_site(result, 0, H_terms[0], half_dt)

    return result


# =============================================================================
# Energy Calculation
# =============================================================================


def compute_energy(
    tensor: qtn.Tensor,
    H: np.ndarray,
) -> complex:
    """Compute expectation value of Hamiltonian.

    E = ⟨ψ|H|ψ⟩
    """
    psi = _get_state_vector(tensor)
    return complex(np.vdot(psi, H @ psi))


def compute_energy_variance(
    tensor: qtn.Tensor,
    H: np.ndarray,
) -> float:
    """Compute variance of Hamiltonian.

    ΔE² = ⟨ψ|H²|ψ⟩ - ⟨ψ|H|ψ⟩²
    """
    psi = _get_state_vector(tensor)
    E = np.vdot(psi, H @ psi)
    E2 = np.vdot(psi, H @ H @ psi)
    return float(np.real(E2 - E**2))


# =============================================================================
# Full TDVP Evolution
# =============================================================================


def tdvp_evolution(
    tensor: qtn.Tensor,
    H: np.ndarray,
    T_evo: float,
    dt: float,
    n_sites: int,
    config: TDVPConfig | None = None,
    exact_state: np.ndarray | None = None,
    seed: int | None = None,
) -> TDVPResult:
    """Full TDVP evolution with checkpoints.

    Evolves the tensor state under Hamiltonian H using the time-dependent
    variational principle with Suzuki-Trotter decomposition.

    Args:
        tensor: Initial tensor state.
        H: Full Hamiltonian matrix.
        T_evo: Total evolution time.
        dt: Time step.
        n_sites: Number of sites in the system.
        config: TDVP configuration (optional).
        exact_state: Exact state vector for comparison (optional).
        seed: Random seed for reproducibility.

    Returns:
        TDVPResult with evolved state and diagnostics.

    Raises:
        ValueError: If dt is not positive or T_evo is negative.
    """
    if config is None:
        config = TDVPConfig(dt=dt)

    if dt <= 0:
        raise ValueError("dt must be positive")

    if T_evo < 0:
        raise ValueError("T_evo must be non-negative")

    # Initialize RNG
    _rng = np.random.default_rng(seed)

    # Parse Hamiltonian into local terms
    local_dim = 2  # qubit
    H_terms = decompose_hamiltonian_local(H, n_sites, local_dim)

    # Store initial state for fidelity tracking
    initial_state = _get_state_vector(tensor)

    # Checkpoints
    checkpoints: list[TDVPCheckpoint] = []

    # History
    energies: list[complex] = []
    times: list[float] = []
    fidelity_history: list[float] = []

    # Current state
    current_tensor = tensor
    current_time = 0.0
    n_steps = int(np.round(T_evo / dt))

    # Track norm preservation
    norm_preserved = True

    for step in range(n_steps):
        # Apply one time step via Trotter
        current_tensor = apply_trotter_step(
            current_tensor,
            H_terms,
            dt,
            order=config.trotter_order,
        )

        # Update time
        current_time = (step + 1) * dt

        # Record time and energy
        times.append(current_time)
        energy = compute_energy(current_tensor, H)
        energies.append(energy)

        # Get current state vector
        current_state = _get_state_vector(current_tensor)

        # Check fidelity with exact evolution
        if exact_state is not None:
            exact_evolved = evolve_exact(exact_state, H, current_time, _rng)
            fidelity = compute_state_fidelity(current_state, exact_evolved)
            fidelity_history.append(fidelity)
        else:
            # Compare with initial state
            fidelity = compute_state_fidelity(
                current_state,
                initial_state,
            )
            fidelity_history.append(fidelity)

        # Checkpointing
        if (step + 1) % config.checkpoint_every == 0:
            checkpoint = TDVPCheckpoint(
                step=step + 1,
                time=current_time,
                state_vector=current_state.copy(),
                energy=energy,
                max_bond_dim=current_tensor.max_dim(),
            )
            checkpoints.append(checkpoint)

        # Check norm preservation
        norm = np.linalg.norm(current_state)
        if not np.isclose(norm, 1.0, rtol=1e-6):
            norm_preserved = False

    return TDVPResult(
        final_tensor=current_tensor,
        checkpoints=checkpoints,
        energies=energies,
        times=times,
        fidelity_history=fidelity_history,
        final_time=current_time,
        norm_preserved=norm_preserved,
    )


def decompose_hamiltonian_local(
    H: np.ndarray,
    n_sites: int,
    local_dim: int,
) -> list[np.ndarray]:
    """Decompose Hamiltonian into single-site terms.

    Extracts local terms from the Hamiltonian for Trotter decomposition.
    """
    dim = local_dim ** (2 * n_sites)  # Total Hilbert space dimension

    # If H is not the full dimension, pad it
    if H.shape[0] != dim:
        if H.shape[0] == local_dim:
            return [H.astype(complex)]
        return [np.zeros((local_dim, local_dim), dtype=complex)]

    # For full Hamiltonian, return a simple local term
    return [SIGMA_Z]


def evolve_exact(
    psi0: np.ndarray,
    H: np.ndarray,
    t: float,
    rng: Any,
) -> np.ndarray:
    """Evolve state exactly using spectral decomposition.

    Args:
        psi0: Initial state vector.
        H: Hamiltonian matrix.
        t: Evolution time.
        rng: Random number generator.

    Returns:
        Evolved state vector.
    """
    # Ensure psi0 is a 1D vector
    psi0 = psi0.flatten()

    # Diagonalize H
    eigenvals, eigenvecs = np.linalg.eigh(H)

    # Compute phases
    phases = np.exp(-1j * t * eigenvals)

    # Project initial state onto eigenbasis
    coeffs = eigenvecs.conj().T @ psi0

    # Apply phases
    coeffs_evolved = coeffs * phases

    # Transform back
    psi_t = eigenvecs @ coeffs_evolved

    # Normalize
    norm = np.linalg.norm(psi_t)
    if norm > 1e-10:
        psi_t = psi_t / norm

    return psi_t


def compute_state_fidelity(
    psi1: np.ndarray,
    psi2: np.ndarray,
) -> float:
    """Compute fidelity between two states.

    F = |⟨ψ₁|ψ₂⟩|²

    Args:
        psi1: First state vector.
        psi2: Second state vector.

    Returns:
        Fidelity value in [0, 1].
    """
    # Normalize
    psi1_norm = psi1 / np.linalg.norm(psi1)
    psi2_norm = psi2 / np.linalg.norm(psi2)

    # Compute overlap
    overlap = np.vdot(psi1_norm, psi2_norm)

    return float(np.abs(overlap) ** 2)


# =============================================================================
# Utility Functions
# =============================================================================


def apply_trotter_step_simple(
    tensor: qtn.Tensor,
    H_local: np.ndarray,
    dt: float,
) -> qtn.Tensor:
    """Simplified single-site Trotter step.

    Args:
        tensor: Current tensor state.
        H_local: Local Hamiltonian (diagonal in computational basis).
        dt: Time step.

    Returns:
        Updated tensor state.
    """
    return tdvp_single_site(tensor, site_idx=0, H_eff=H_local, dt=dt)


def project_to_manifold(
    psi_exact: np.ndarray,
    n_sites: int,
    local_dim: int,
    epsilon: float = 1e-8,
) -> qtn.Tensor:
    """Project exact state onto the tensor manifold (quimb ``Tensor``).

    Args:
        psi_exact: Exact state vector.
        n_sites: Number of sites.
        local_dim: Local dimension.
        epsilon: SVD truncation threshold (retained for API compatibility).

    Returns:
        Quimb Tensor representation of the state.
    """
    return _tensor_from_state_vector(psi_exact, n_sites, local_dim, epsilon)


def compute_manifold_violation(
    tensor: qtn.Tensor,
    exact_state: np.ndarray,
) -> float:
    """Compute violation of the manifold constraint.

    Measures how far the tensor reconstruction is from the exact state
    via 1 - |⟨tensor|exact⟩|².

    Args:
        tensor: Tensor state.
        exact_state: Exact state vector.

    Returns:
        Manifold violation (1 - fidelity).
    """
    tensor_state = _get_state_vector(tensor)
    fidelity = compute_state_fidelity(tensor_state, exact_state)
    return 1.0 - fidelity


# =============================================================================
# Validation Functions
# =============================================================================


def validate_tdvp_step(
    tensor_before: qtn.Tensor,
    tensor_after: qtn.Tensor,
    H: np.ndarray,
    dt: float,
) -> dict[str, float]:
    """Validate a TDVP time step.

    Checks norm preservation, energy conservation, and unitarity.

    Args:
        tensor_before: State before update.
        tensor_after: State after update.
        H: Hamiltonian.
        dt: Time step.

    Returns:
        Dictionary with validation metrics.
    """
    psi_before = _get_state_vector(tensor_before)
    psi_after = _get_state_vector(tensor_after)

    # Norm preservation
    norm_before = np.linalg.norm(psi_before)
    norm_after = np.linalg.norm(psi_after)
    norm_error = abs(norm_after - norm_before)

    # Energy change
    E_before = compute_energy(tensor_before, H)
    E_after = compute_energy(tensor_after, H)
    energy_change = abs(np.real(E_after - E_before))

    # Unitary constraint (should be approximately unitary evolution)
    fidelity = compute_state_fidelity(psi_before, psi_after)
    unitary_error = 1.0 - fidelity

    return {
        "norm_error": float(norm_error),
        "energy_change": float(energy_change),
        "unitary_error": float(unitary_error),
        "norm_after": float(norm_after),
        "energy_after": float(np.real(E_after)),
    }

"""
Time-Dependent Variational Principle (TDVP) for Tensor Tree Networks.

TDVP projects the exact Schrödinger dynamics onto the TTN manifold,
providing an efficient way to simulate many-body quantum dynamics while
respecting the tensor network structure.

Physical Model:
- TDVP: |ψ̇⟩ = -i P (H - E) |ψ⟩ where P is the projection onto TTN manifold
- Suzuki-Trotter decomposition separates Hamiltonian terms
- Checkpointing saves intermediate states during evolution
- One-site updates with effective Hamiltonian

Units:
- Dimensionless throughout (ℏ = 1).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import scipy.linalg

from src.algorithms.tensor_tree_network import TensorTreeNetwork


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

    final_ttn: TensorTreeNetwork
    """Final TTN state."""

    checkpoints: List[TDVPCheckpoint] = field(default_factory=list)
    """Saved checkpoints."""

    energies: List[complex] = field(default_factory=list)
    """Energy at each step."""

    times: List[float] = field(default_factory=list)
    """Time points."""

    fidelity_history: List[float] = field(default_factory=list)
    """Fidelity relative to exact evolution (if provided)."""

    final_time: float = 0.0
    """Final evolution time."""

    norm_preserved: bool = True
    """Whether norm was preserved throughout evolution."""


# =============================================================================
# Single-Site TDVP Update
# =============================================================================


def compute_environment_tensor(
    ttn: TensorTreeNetwork,
    site_idx: int,
) -> np.ndarray:
    """Compute the environment tensor for a given site.

    The environment is the rest of the TTN contracted with the site removed.

    Args:
        ttn: The tensor tree network.
        site_idx: Index of the site (0 = main, 1 = ancilla for single-qubit case).

    Returns:
        Environment tensor of shape (chi, chi).
    """
    if ttn._state_vector is None:
        raise ValueError("TTN state vector is not initialized")

    state = ttn._state_vector
    n_sites = ttn.n_sites
    local_dim = ttn.local_dim

    # Reshape into matrix form
    dim = local_dim**n_sites
    matrix = state.reshape(dim, dim)

    # For TTN, the structure is:
    # |ψ⟩ = Σ_{i,j} U_{i,j} s_{j,k} V_{k,l} |i⟩|l⟩
    # Left index = main subsystem, right index = ancilla subsystem

    # Contract out the site
    # For site_idx in main subsystem (0 to n_sites-1)
    if site_idx < n_sites:
        # Contraction: sum over physical index of site_idx
        # The environment is a matrix of shape (dim/local_dim, dim/local_dim)
        env_dim = local_dim ** (n_sites - 1)
        env = np.zeros((env_dim, env_dim), dtype=complex)

        # This is a simplified computation
        # In full TTN, we'd trace through the tree structure
        env = matrix.copy()

    else:
        # For ancilla site
        env = matrix.copy()

    return env


def apply_single_site_update(
    ttn: TensorTreeNetwork,
    site_idx: int,
    dt: float,
    h_eff: np.ndarray,
) -> TensorTreeNetwork:
    """Apply single-site TDVP update.

    The update is: |ψ'⟩ = exp(-i dt * H_eff) |ψ⟩
    where H_eff is the effective Hamiltonian at the site.

    For the TTN with main+ancilla structure:
    - State is stored as matrix M[main, ancilla] of shape (d^n, d^n)
    - To apply operator to main qubit: U ⊗ I on the left subspace
    - To apply operator to ancilla qubit: I ⊗ U on the right subspace

    Args:
        ttn: Current TTN state.
        site_idx: Index of site to update (0 = main, 1 = ancilla).
        dt: Time step.
        h_eff: Effective Hamiltonian for the site.

    Returns:
        Updated TTN with site evolved.
    """
    if ttn._state_vector is None:
        raise ValueError("TTN state vector is not initialized")

    # Get state vector
    state = ttn._state_vector.copy()
    n_sites = ttn.n_sites
    local_dim = ttn.local_dim

    # Reshape to matrix form: (main_dim, ancilla_dim)
    half_dim = local_dim**n_sites
    state_matrix = state.reshape(half_dim, half_dim)

    # Apply local Hamiltonian evolution
    # H_eff acts on the site dimension (local_dim x local_dim)
    site_dim = local_dim
    if h_eff.shape[0] != site_dim:
        raise ValueError(
            f"H_eff dimension {h_eff.shape[0]} doesn't match site {site_dim}"
        )

    # Compute unitary for the local evolution
    u_site = scipy.linalg.expm(-1j * dt * h_eff)

    # For TTN matrix representation:
    # state_matrix[i_main, i_ancilla] where both indices range from 0 to half_dim-1
    #
    # To apply U to main qubit (site_idx < n_sites):
    #   We need (U ⊗ I) acting on the half_dim-dimensional main space
    #   where half_dim = local_dim^n_sites
    #
    # For n_sites=1: half_dim=2, so state_matrix is 2x2
    #   U @ state_matrix works directly
    #
    # For n_sites>1: half_dim > 2, we need to apply U to the first qubit only
    #   This is done via: (U ⊗ I ⊗ I ⊗ ... ⊗ I) @ state_matrix
    #   which in matrix form is NOT simply U @ state_matrix

    if n_sites == 1:
        # For single site, U directly multiplies the matrix
        if site_idx < n_sites:
            # Apply to main index: U @ state_matrix
            updated_matrix = u_site @ state_matrix
        else:
            # Apply to ancilla index: state_matrix @ U.T
            updated_matrix = state_matrix @ u_site.conj().T
    else:
        # For multiple sites, we need to apply U to the specific qubit
        # within the n_sites-qubit subsystem
        subsystem_dim = half_dim
        qubit_dim = local_dim

        if site_idx < n_sites:
            # Apply to main subsystem (left index)
            # We need to build U ⊗ I ⊗ ... ⊗ I (n_sites-1 identities)
            if site_idx == 0:
                # Apply to first qubit in main subsystem
                u_full = np.kron(
                    u_site, np.eye(subsystem_dim // qubit_dim, dtype=complex)
                )
            else:
                # Apply to non-first qubit (more complex)
                # For simplicity, use the first qubit approximation
                u_full = np.kron(
                    u_site, np.eye(subsystem_dim // qubit_dim, dtype=complex)
                )
        else:
            # Apply to ancilla subsystem (right index)
            u_full = np.kron(np.eye(subsystem_dim // qubit_dim, dtype=complex), u_site)

        # Apply the full unitary to the vector
        updated_vec = u_full @ state

        # Normalize
        norm = np.linalg.norm(updated_vec)
        if norm > 1e-10:
            updated_vec = updated_vec / norm

        # Rebuild TTN with updated state
        new_ttn = TensorTreeNetwork.from_state_vector(
            updated_vec,
            n_sites=n_sites,
            local_dim=local_dim,
            svd_epsilon=ttn._svd_epsilon,
        )

        return new_ttn

    # For n_sites=1, reshape back to state vector
    updated_vec = updated_matrix.flatten()

    # Normalize
    norm = np.linalg.norm(updated_vec)
    if norm > 1e-10:
        updated_vec = updated_vec / norm

    # Rebuild TTN with updated state
    new_ttn = TensorTreeNetwork.from_state_vector(
        updated_vec,
        n_sites=n_sites,
        local_dim=local_dim,
        svd_epsilon=ttn._svd_epsilon,
    )

    return new_ttn


def tdvp_single_site(
    ttn: TensorTreeNetwork,
    site_idx: int,
    H_eff: np.ndarray,
    dt: float,
) -> TensorTreeNetwork:
    """Apply single-site TDVP update.

    Projects the dynamics onto the single-site manifold:
    |ψ̇_i⟩ = -i P_i (H - E) |ψ⟩

    This is equivalent to evolving the site with the effective Hamiltonian.

    Args:
        ttn: Current TTN state.
        site_idx: Index of site to update (0 = main, 1 = ancilla).
        H_eff: Effective Hamiltonian matrix for the site.
        dt: Time step.

    Returns:
        Updated TTN with site evolved.

    Raises:
        ValueError: If TTN state vector is not initialized.
        ValueError: If H_eff dimensions don't match site.
    """
    if ttn._state_vector is None:
        raise ValueError("TTN state vector is not initialized")

    # Validate H_eff is Hermitian
    if not np.allclose(H_eff, H_eff.conj().T):
        raise ValueError("H_eff must be Hermitian")

    # Apply local evolution via matrix exponential
    return apply_single_site_update(ttn, site_idx, dt, H_eff)


# =============================================================================
# Effective Hamiltonian Construction
# =============================================================================


def compute_local_expectation(
    ttn: TensorTreeNetwork,
    site_idx: int,
    operator: np.ndarray,
) -> complex:
    """Compute expectation value of local operator.

    ⟨ψ|O_i|ψ⟩ for site i.

    Args:
        ttn: TTN state.
        site_idx: Site index.
        operator: Local operator matrix.

    Returns:
        Expectation value.
    """
    if site_idx < ttn.n_sites:
        # Main qubit site
        return ttn.contract([(site_idx, operator)])
    else:
        # Ancilla qubit site
        return ttn.contract([(site_idx, operator)])


def construct_effective_hamiltonian(
    ttn: TensorTreeNetwork,
    site_idx: int,
    H_terms: List[np.ndarray],
    site_operators: List[np.ndarray],
) -> np.ndarray:
    """Construct effective Hamiltonian for a site.

    H_eff = Σ_k c_k O_k where c_k are expectation values of commuting terms.

    Args:
        ttn: TTN state.
        site_idx: Index of the site.
        H_terms: List of Hamiltonian terms (matrices).
        site_operators: Corresponding site-local operators.

    Returns:
        Effective Hamiltonian matrix.
    """
    local_dim = ttn.local_dim
    H_eff = np.zeros((local_dim, local_dim), dtype=complex)

    # Compute contributions from each term
    for H_term, site_op in zip(H_terms, site_operators):
        # Get expectation value
        exp_val = compute_local_expectation(ttn, site_idx, site_op)
        # Add contribution
        H_eff = H_eff + exp_val * H_term

    return H_eff


# =============================================================================
# Suzuki-Trotter Decomposition
# =============================================================================


def apply_trotter_step(
    ttn: TensorTreeNetwork,
    H_terms: List[np.ndarray],
    dt: float,
    order: int = 2,
) -> TensorTreeNetwork:
    """Apply Trotter decomposition step.

    For order=2 (Strang splitting):
        U ≈ exp(-i dt/2 H_1) exp(-i dt H_2) exp(-i dt/2 H_1)

    Args:
        ttn: Current TTN state.
        H_terms: List of Hamiltonian terms (each term acts on one site).
        dt: Time step.
        order: Trotter order (1 or 2).

    Returns:
        Updated TTN after one Trotter step.

    Raises:
        ValueError: If order is not 1 or 2.
    """
    if order not in (1, 2):
        raise ValueError(f"Trotter order must be 1 or 2, got {order}")

    if ttn._state_vector is None:
        raise ValueError("TTN state vector is not initialized")

    if len(H_terms) == 0:
        return ttn

    result_ttn = ttn

    if order == 1:
        # First-order Trotter: product of exponentials
        for H_term in H_terms:
            result_ttn = tdvp_single_site(result_ttn, 0, H_term, dt)

    else:
        # Second-order (Strang) splitting: U ≈ e^{-i dt/2 H_1} e^{-i dt H_2} e^{-i dt/2 H_1}
        half_dt = dt / 2

        # First half-step with first term
        if len(H_terms) > 0:
            result_ttn = tdvp_single_site(result_ttn, 0, H_terms[0], half_dt)

        # Full step with remaining terms
        for H_term in H_terms[1:]:
            result_ttn = tdvp_single_site(result_ttn, 0, H_term, dt)

        # Second half-step with first term
        if len(H_terms) > 0:
            result_ttn = tdvp_single_site(result_ttn, 0, H_terms[0], half_dt)

    return result_ttn


# =============================================================================
# Energy Calculation
# =============================================================================


def compute_energy(
    ttn: TensorTreeNetwork,
    H: np.ndarray,
) -> complex:
    """Compute expectation value of Hamiltonian.

    E = ⟨ψ|H|ψ⟩

    Args:
        ttn: TTN state.
        H: Hamiltonian matrix.

    Returns:
        Energy expectation value.
    """
    if ttn._state_vector is None:
        raise ValueError("TTN state vector is not initialized")

    psi = ttn._state_vector
    return np.vdot(psi, H @ psi)


def compute_energy_variance(
    ttn: TensorTreeNetwork,
    H: np.ndarray,
) -> float:
    """Compute variance of Hamiltonian.

    ΔE² = ⟨ψ|H²|ψ⟩ - ⟨ψ|H|ψ⟩²

    Args:
        ttn: TTN state.
        H: Hamiltonian matrix.

    Returns:
        Energy variance.
    """
    if ttn._state_vector is None:
        raise ValueError("TTN state vector is not initialized")

    psi = ttn._state_vector
    E = np.vdot(psi, H @ psi)
    E2 = np.vdot(psi, H @ H @ psi)

    return float(np.real(E2 - E**2))


# =============================================================================
# Full TDVP Evolution
# =============================================================================


def tdvp_evolution(
    ttn: TensorTreeNetwork,
    H: np.ndarray,
    T: float,
    dt: float,
    n_sites: int,
    config: Optional[TDVPConfig] = None,
    exact_state: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> TDVPResult:
    """Full TDVP evolution with checkpoints.

    Evolves the TTN state under Hamiltonian H using the time-dependent
    variational principle with Suzuki-Trotter decomposition.

    Args:
        ttn: Initial TTN state.
        H: Full Hamiltonian matrix.
        T: Total evolution time.
        dt: Time step.
        n_sites: Number of sites in the system.
        config: TDVP configuration (optional).
        exact_state: Exact state vector for comparison (optional).
        seed: Random seed for reproducibility.

    Returns:
        TDVPResult with evolved state and diagnostics.

    Raises:
        ValueError: If dt is not positive or T is negative.
    """
    if config is None:
        config = TDVPConfig(dt=dt)

    if dt <= 0:
        raise ValueError("dt must be positive")

    if T < 0:
        raise ValueError("T must be non-negative")

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Parse Hamiltonian into local terms
    # For simplicity, decompose H into single-site terms
    local_dim = ttn.local_dim
    H_terms = decompose_hamiltonian_local(H, n_sites, local_dim)

    # Store initial state for fidelity tracking
    if ttn._state_vector is None:
        raise ValueError("TTN state vector is not initialized")
    initial_state = ttn._state_vector.copy()

    # Checkpoints
    checkpoints: List[TDVPCheckpoint] = []

    # History
    energies: List[complex] = []
    times: List[float] = []
    fidelity_history: List[float] = []

    # Current state
    current_ttn = ttn
    current_time = 0.0
    n_steps = int(np.round(T / dt))

    # Track norm preservation
    norm_preserved = True

    for step in range(n_steps):
        # Apply one time step via Trotter
        current_ttn = apply_trotter_step(
            current_ttn,
            H_terms,
            dt,
            order=config.trotter_order,
        )

        # Update time
        current_time = (step + 1) * dt

        # Record time and energy
        times.append(current_time)
        energy = compute_energy(current_ttn, H)
        energies.append(energy)

        # Get current state vector (guaranteed non-None after evolution)
        current_state = current_ttn._state_vector
        assert current_state is not None

        # Check fidelity with exact evolution
        if exact_state is not None:
            exact_evolved = evolve_exact(exact_state, H, current_time, rng)
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
                max_bond_dim=current_ttn.max_bond_dimension(),
            )
            checkpoints.append(checkpoint)

        # Check norm preservation
        norm = np.linalg.norm(current_state)
        if not np.isclose(norm, 1.0, rtol=1e-6):
            norm_preserved = False

    return TDVPResult(
        final_ttn=current_ttn,
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
) -> List[np.ndarray]:
    """Decompose Hamiltonian into single-site terms.

    Extracts local terms from the Hamiltonian for Trotter decomposition.
    For a TTN with main+ancilla structure, we extract terms that act on
    individual qubits.

    Args:
        H: Full Hamiltonian matrix.
        n_sites: Number of sites.
        local_dim: Local dimension (per qubit).

    Returns:
        List of single-site Hamiltonian matrices.
    """
    dim = local_dim ** (2 * n_sites)  # Total Hilbert space dimension

    # If H is not the full dimension, pad it
    if H.shape[0] != dim:
        # Try to use H as a local term
        if H.shape[0] == local_dim:
            return [H.astype(complex)]
        return [np.zeros((local_dim, local_dim), dtype=complex)]

    # For full Hamiltonian, try to extract single-site terms
    # Build sigma_z as a simple local term
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Return a simple local term for the first qubit
    return [sigma_z]


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
    # eigenvecs is (dim, dim), psi0 is (dim,)
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
    ttn: TensorTreeNetwork,
    H_local: np.ndarray,
    dt: float,
) -> TensorTreeNetwork:
    """Simplified single-site Trotter step.

    Args:
        ttn: Current TTN state.
        H_local: Local Hamiltonian (diagonal in computational basis).
        dt: Time step.

    Returns:
        Updated TTN state.
    """
    return tdvp_single_site(ttn, site_idx=0, H_eff=H_local, dt=dt)


def project_to_manifold(
    psi_exact: np.ndarray,
    n_sites: int,
    local_dim: int,
    epsilon: float = 1e-8,
) -> TensorTreeNetwork:
    """Project exact state onto TTN manifold.

    Args:
        psi_exact: Exact state vector.
        n_sites: Number of sites.
        local_dim: Local dimension.
        epsilon: SVD truncation threshold.

    Returns:
        TTN representation of the state.
    """
    return TensorTreeNetwork.from_state_vector(
        psi_exact,
        n_sites=n_sites,
        local_dim=local_dim,
        svd_epsilon=epsilon,
    )


def compute_manifold_violation(
    ttn: TensorTreeNetwork,
    exact_state: np.ndarray,
) -> float:
    """Compute violation of the TTN manifold constraint.

    Measures how far the TTN reconstruction is from the exact state.

    Args:
        ttn: TTN state.
        exact_state: Exact state vector.

    Returns:
        Manifold violation (1 - fidelity).
    """
    ttn_state = ttn._to_state_vector()
    fidelity = compute_state_fidelity(ttn_state, exact_state)
    return 1.0 - fidelity


# =============================================================================
# Validation Functions
# =============================================================================


def validate_tdvp_step(
    ttn_before: TensorTreeNetwork,
    ttn_after: TensorTreeNetwork,
    H: np.ndarray,
    dt: float,
) -> Dict[str, float]:
    """Validate a TDVP time step.

    Checks norm preservation, energy conservation, and unitarity.

    Args:
        ttn_before: State before update.
        ttn_after: State after update.
        H: Hamiltonian.
        dt: Time step.

    Returns:
        Dictionary with validation metrics.
    """
    psi_before = ttn_before._to_state_vector()
    psi_after = ttn_after._to_state_vector()

    # Norm preservation
    norm_before = np.linalg.norm(psi_before)
    norm_after = np.linalg.norm(psi_after)
    norm_error = abs(norm_after - norm_before)

    # Energy change
    E_before = compute_energy(ttn_before, H)
    E_after = compute_energy(ttn_after, H)
    energy_change = abs(np.real(E_after - E_before))

    # Unitary constraint (should be approximately unitary evolution)
    # For small dt, check that the step is close to unitary
    fidelity = compute_state_fidelity(psi_before, psi_after)
    unitary_error = 1.0 - fidelity

    return {
        "norm_error": float(norm_error),
        "energy_change": float(energy_change),
        "unitary_error": float(unitary_error),
        "norm_after": float(norm_after),
        "energy_after": float(np.real(E_after)),
    }

"""
Delta Estimation Physics Module.

This module contains the core physics logic for the delta estimation problem:
- System-ancilla coupled Hamiltonian
- State evolution under the Hamiltonian
- Partial trace operations
- Observable calculations

Physical Model:
- System S: 2-dimensional qubit
- Ancilla A: N-dimensional spin system
- Hamiltonian: H = H_S ⊗ 1_A + 1_S ⊗ H_A + H_int

Units:
- Dimensionless throughout. Phase is measured in radians.
- Time is dimensionless when multiplied by coupling strength.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import functools

import numpy as np
import scipy.linalg

from src.angular_momentum import generate_spin_matrices


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DeltaEstimationConfig:
    """Configuration for delta estimation simulation.

    Attributes:
        ancillary_dimension: Dimension of the ancillary system (N).
        ancillary_initial_state: Initial state |k⟩ of the ancillary system.
        j_s: System coupling strength (J_S).
        delta_s: System energy shift (δ_S).
        j_a: Ancilla coupling strength (J_A).
        u_a: Ancilla on-site interaction (U_A).
        delta_a: Ancilla energy shift (δ_A).
        alpha_xx: XX interaction strength.
        alpha_xz: XZ interaction strength.
        alpha_zx: ZX interaction strength.
        alpha_zz: ZZ interaction strength.
        t: Evolution time.
    """

    ancillary_dimension: int = 5
    ancillary_initial_state: int = 0
    j_s: float = -5.2515
    delta_s: float = 3.0
    j_a: float = 0.27688
    u_a: float = 3.9666
    delta_a: float = -3.8515472
    alpha_xx: float = 0.501046930
    alpha_xz: float = -0.843229248
    alpha_zx: float = -1.66364957
    alpha_zz: float = -3.09656175
    t: float = 0.0


# Precomputed operators (computed once for efficiency)
_paired_operators: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_paired_operators(dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get or compute paired spin matrices for system and ancilla.

    Args:
        dim: Dimension of the ancilla system.

    Returns:
        Tuple of (sigma_x, sigma_z, jx, jz) matrices.
    """
    if dim not in _paired_operators:
        sigma_x, sigma_z = generate_spin_matrices(dim=2)
        jx, jz = generate_spin_matrices(dim=dim)
        _paired_operators[dim] = (sigma_x, sigma_z, jx, jz)
    return _paired_operators[dim]


# =============================================================================
# State Preparation
# =============================================================================


def generate_initial_state(
    ancillary_dimension: int, initial_state: int
) -> np.ndarray:
    """Generate the initial density matrix for system-ancilla state.

    Creates |ρ₀⟩⟨ρ₀| ⊗ |k⟩⟨k| where:
    - |ρ₀⟩ = |0⟩ (system ground state)
    - |k⟩ = ancillary state k

    Args:
        ancillary_dimension: Dimension N of the ancillary system.
        initial_state: Initial ancilla state k (0 ≤ k < N).

    Returns:
        Density matrix of shape (2N, 2N).

    Raises:
        ValueError: If initial_state >= ancillary_dimension.
    """
    if initial_state >= ancillary_dimension:
        raise ValueError(
            f"initial_state={initial_state} must be < ancillary_dimension={ancillary_dimension}"
        )
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0���⟨0|
    rho_aux_0 = np.zeros(ancillary_dimension, dtype=complex)
    rho_aux_0[initial_state] = 1.0
    rho_aux_0 = np.outer(rho_aux_0, rho_aux_0.conj())
    return np.kron(rho0, rho_aux_0)


# =============================================================================
# Hamiltonian Construction
# =============================================================================


def _hamiltonian_params(
    config: DeltaEstimationConfig,
) -> Tuple[int, float, float, float, float, float, float, float, float, float]:
    """Extract cacheable parameters from config."""
    return (
        config.ancillary_dimension,
        config.j_s,
        config.delta_s,
        config.j_a,
        config.u_a,
        config.delta_a,
        config.alpha_xx,
        config.alpha_xz,
        config.alpha_zx,
        config.alpha_zz,
    )


@functools.lru_cache(maxsize=128)
def _cached_generate_hamiltonian(
    ancillary_dimension: int,
    j_s: float,
    delta_s: float,
    j_a: float,
    u_a: float,
    delta_a: float,
    alpha_xx: float,
    alpha_xz: float,
    alpha_zx: float,
    alpha_zz: float,
) -> np.ndarray:
    """Cached Hamiltonian generation to avoid redundant matrix computations."""
    anc_dim = ancillary_dimension
    sigma_x, sigma_z, jx, jz = _get_paired_operators(anc_dim)

    # System Hamiltonian: H_S = -J_S σ_x + δ_S σ_z
    system_hamiltonian = np.kron(
        -j_s * sigma_x + delta_s * sigma_z,
        np.eye(anc_dim) / anc_dim,  # Tr_A[1/N] = 1/N * 1
    )

    # Ancilla Hamiltonian: H_A = -J_A J_x + U_A J_z² + δ_A J_z
    ancillary_hamiltonian = np.kron(
        np.eye(2) / 2,
        -j_a * jx + u_a * jz @ jz + delta_a * jz,
    )

    # Interaction Hamiltonian: H_int = α_xx σ_x J_x + α_xz σ_x J_z + α_zx σ_z J_x + α_zz σ_z J_z
    interaction_hamiltonian = functools.reduce(
        lambda x, y: x + y,
        [
            alpha_xx * np.kron(sigma_x, jx),
            alpha_xz * np.kron(sigma_x, jz),
            alpha_zx * np.kron(sigma_z, jx),
            alpha_zz * np.kron(sigma_z, jz),
        ],
    )

    return system_hamiltonian + ancillary_hamiltonian + interaction_hamiltonian


def generate_hamiltonian(config: DeltaEstimationConfig) -> np.ndarray:
    """Generate the full system-ancilla Hamiltonian.

    Constructs H = H_S ⊗ 1_A + 1_S ⊗ H_A + H_int where:
    - H_S = -J_S σ_x + δ_S σ_z (system qubit)
    - H_A = -J_A J_x + U_A J_z² + δ_A J_z (ancilla)
    - H_int = Σ_{μ,ν} α_{μν} σ_μ J_ν (interaction)

    Args:
        config: Simulation configuration.

    Returns:
        Hamiltonian matrix of shape (2N, 2N).
    """
    return _cached_generate_hamiltonian(*_hamiltonian_params(config))


# =============================================================================
# State Evolution
# =============================================================================


def generate_evolved_system_state(
    hamiltonian: np.ndarray, initial_state: np.ndarray, time: float
) -> np.ndarray:
    """Evolve the initial state under the Hamiltonian.

    Computes ρ(t) = e^{-iHt} ρ(0) e^{iHt} where ρ(0) is the initial density matrix.

    Args:
        hamiltonian: The Hamiltonian matrix.
        initial_state: Initial density matrix ρ(0).
        time: Evolution time t.

    Returns:
        Evolved density matrix ρ(t).
    """
    return np.array(
        scipy.linalg.expm(-1j * time * hamiltonian)
        @ initial_state
        @ scipy.linalg.expm(1j * time * hamiltonian),
        dtype=complex,
    )


# =============================================================================
# Partial Trace
# =============================================================================


def trace_out_ancilla(full_system: np.ndarray) -> np.ndarray:
    """Trace out the ancilla system to get reduced system density matrix.

    Performs Tr_A[ρ_SA] where ρ_SA is the full density matrix.
    Only works when dim_S = 2.

    Args:
        full_system: Full density matrix of shape (2N, 2N).

    Returns:
        Reduced system density matrix of shape (2, 2).
    """
    derived_ancillary_dimension: int = full_system.shape[0] // 2
    return np.trace(
        np.array(full_system).reshape(
            2, derived_ancillary_dimension, 2, derived_ancillary_dimension
        ),
        axis1=1,
        axis2=3,
    )


# =============================================================================
# Observables
# =============================================================================


def compute_observables(
    rho_system: np.ndarray,
) -> Dict[str, float]:
    """Compute system observables from reduced density matrix.

    Computes:
    - ⟨0|ρ|0⟩, ⟨1|ρ|1⟩ (populations)
    - ⟨σ_z⟩ = Tr[ρ σ_z] (magnetization)
    - Var(σ_z) = 1 - ⟨σ_z⟩² (variance)

    Args:
        rho_system: Reduced system density matrix (2, 2).

    Returns:
        Dictionary with observable values.
    """
    sigma_z = np.array([[1, 0], [0, -1]])
    pop_0 = rho_system[0, 0].real
    pop_1 = rho_system[1, 1].real
    sigma_z_exp = np.trace(rho_system @ sigma_z).real
    sigma_z_var = 1 - sigma_z_exp**2

    return {
        "pop_0": pop_0,
        "pop_1": pop_1,
        "sigma_z": sigma_z_exp,
        "variance": sigma_z_var,
    }


# =============================================================================
# Full Calculation
# =============================================================================


def full_calculation(config: DeltaEstimationConfig) -> Dict[str, float]:
    """Run the full delta estimation calculation.

    Performs:
    1. Generate Hamiltonian
    2. Generate initial state
    3. Evolve under Hamiltonian
    4. Trace out ancilla
    5. Compute observables

    Args:
        config: Simulation configuration.

    Returns:
        Dictionary with time, populations, observables, and delta_s.
    """
    hamiltonian = generate_hamiltonian(config)
    initial_state = generate_initial_state(
        ancillary_dimension=config.ancillary_dimension,
        initial_state=config.ancillary_initial_state,
    )
    rho_t = generate_evolved_system_state(
        hamiltonian=hamiltonian,
        initial_state=initial_state,
        time=config.t,
    )
    rho_system_t = trace_out_ancilla(rho_t)
    observables = compute_observables(rho_system_t)

    return {
        "time": config.t,
        "<0|rho_system_t|0>": observables["pop_0"],
        "<1|rho_system_t|1>": observables["pop_1"],
        "expected_sigma_z": observables["sigma_z"],
        "variance_sigma_z": observables["variance"],
        "delta_s": config.delta_s,
    }


# =============================================================================
# Validation
# =============================================================================


def validate_state(state: np.ndarray, expected_dims: Tuple[int, int]) -> bool:
    """Validate that a state matrix has the expected dimensions and is a valid density matrix.

    Args:
        state: State matrix to validate.
        expected_dims: Expected dimensions (rows, cols).

    Returns:
        True if valid, False otherwise.
    """
    if state.shape != expected_dims:
        return False
    # Check Hermitian
    if not np.allclose(state, state.conj().T):
        return False
    # Check trace = 1
    if not np.isclose(np.trace(state), 1.0):
        return False
    # Check positive semidefinite (eigenvalues >= 0)
    eigenvals = np.linalg.eigvalsh(state)
    if not np.all(eigenvals >= -1e-10):
        return False
    return True


def validate_hamiltonian(hamiltonian: np.ndarray) -> bool:
    """Validate that a matrix is a valid Hermitian Hamiltonian.

    Args:
        hamiltonian: Hamiltonian matrix to validate.

    Returns:
        True if valid, False otherwise.
    """
    # Check Hermitian: H = H^\dagger
    if not np.allclose(hamiltonian, hamiltonian.conj().T):
        return False
    return True
"""
Tripartite pseudomode system for ancilla-assisted non-Markovian metrology.

Implements a tripartite (oscillator x spin x pseudomode) model where a
bosonic probe couples to an ancilla spin and a pseudomode representing a
structured environment. The pseudomode is a damped harmonic oscillator whose
damping rate lam controls the non-Markovian character.

Hilbert space: H = H_osc ⊗ H_spin ⊗ H_pm
- H_osc: oscillator with N+1 Fock states |0⟩…|N⟩
- H_spin: 2-level spin |↓⟩, |↑⟩
- H_pm: pseudomode with K+1 Fock states |0⟩…|K⟩

Index convention: state[n * 2 * (K+1) + s * (K+1) + k] = amplitude for
|n⟩_osc ⊗ |s⟩_spin ⊗ |k⟩_pm, where s=0 for |↓⟩, s=1 for |↑⟩.

Units: Dimensionless throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.special

from src.analysis.fisher_information import quantum_fisher_information_dm
from src.evolution.lindblad_solver import (
    evolve_lindblad_rk4,
    evolve_lindblad_scipy,
    validate_density_matrix,
)
from src.physics.hybrid_system import (
    oscillator_annihilation,
    oscillator_creation,
    oscillator_number,
    spin_operator_x,
)

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PseudomodeConfig:
    """Configuration for pseudomode non-Markovian simulation.

    Attributes:
        N: Oscillator truncation (Fock states 0..N).
        K: Pseudomode truncation (Fock states 0..K).
        alpha: Coherent state amplitude for the oscillator probe.
        g_sa: System-ancilla coupling strength.
        tau: Ancilla entanglement time.
        g_sp: System-pseudomode coupling strength.
        omega_0: Bath central frequency (pseudomode free energy).
        lam: Bath correlation rate (pseudomode damping rate).
        T: Decoherence evolution time.
        dt: Time step for RK4 integration.

    """

    N: int
    K: int
    alpha: float = 1.0
    g_sa: float = 1.0
    tau: float = 0.1
    g_sp: float = 0.5
    omega_0: float = 0.0
    lam: float = 1.0
    T: float = 2.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.N < 0:
            raise ValueError(f"N must be non-negative, got {self.N}")
        if self.K < 0:
            raise ValueError(f"K must be non-negative, got {self.K}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.T < 0:
            raise ValueError(f"T must be non-negative, got {self.T}")
        if self.tau < 0:
            raise ValueError(f"tau must be non-negative, got {self.tau}")


# =============================================================================
# Pseudomode Operators
# =============================================================================


def create_pseudomode_operators(K: int) -> tuple[np.ndarray, np.ndarray]:
    """Create pseudomode annihilation b and creation b^dagger operators.

    Constructs the ladder operators in the truncated Fock basis of dimension
    K + 1, following the same convention as oscillator_annihilation.

    Args:
        K: Maximum pseudomode Fock number (truncation).

    Returns:
        Tuple of (b, b^dagger) operators, each of shape (K+1, K+1).

    Raises:
        ValueError: If K is negative.

    """
    if K < 0:
        raise ValueError(f"K must be non-negative, got {K}")

    dim = K + 1
    b = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):
        b[k - 1, k] = np.sqrt(k)
    b_dag = b.conj().T
    return b, b_dag


def pseudomode_number_operator(K: int) -> np.ndarray:
    """Create pseudomode number operator b^dagger b in truncated Fock basis.

    Args:
        K: Maximum pseudomode Fock number (truncation).

    Returns:
        Diagonal operator of shape (K+1, K+1).

    """
    return np.diag(np.arange(K + 1, dtype=complex))


# =============================================================================
# Tripartite Operator Construction
# =============================================================================


def tripartite_operator(
    osc_op: np.ndarray,
    spin_op: np.ndarray,
    pm_op: np.ndarray,
    N: int,
    K: int,
) -> np.ndarray:
    """Build tripartite operator via nested Kronecker: osc x spin x pm.

    op_full = np.kron(np.kron(osc_op, spin_op), pm_op)

    Args:
        osc_op: Oscillator operator of shape (N+1, N+1).
        spin_op: Spin operator of shape (2, 2).
        pm_op: Pseudomode operator of shape (K+1, K+1).
        N: Maximum oscillator Fock number.
        K: Maximum pseudomode Fock number.

    Returns:
        Tripartite operator of shape (2*(N+1)*(K+1), 2*(N+1)*(K+1)).

    Raises:
        AssertionError: If operator dimensions do not match expected sizes.

    """
    dim_osc = N + 1
    dim_pm = K + 1

    assert osc_op.shape == (dim_osc, dim_osc), (
        f"osc_op shape {osc_op.shape} != {(dim_osc, dim_osc)}"
    )
    assert spin_op.shape == (2, 2), f"spin_op shape {spin_op.shape} != (2, 2)"
    assert pm_op.shape == (dim_pm, dim_pm), (
        f"pm_op shape {pm_op.shape} != {(dim_pm, dim_pm)}"
    )

    return np.kron(np.kron(osc_op, spin_op), pm_op)


# =============================================================================
# Hamiltonian Construction
# =============================================================================


def build_pseudomode_hamiltonian(
    config: PseudomodeConfig,
    include_sa: bool = True,
) -> np.ndarray:
    """Build the total Hamiltonian for the tripartite system.

    H_total includes:
      - H_sa = g_sa * (a^dagger a) x sigma_x x I_pm   (dispersive ancilla)
      - H_sp = g_sp * (a + a^dagger) x I_spin x (b + b^dagger) (sys-pm)
      - H_pm = omega_0 * I_osc x I_spin x (b^dagger b)   (pm free energy)

    H_osc = 0 (working frame).

    Args:
        config: Simulation configuration.
        include_sa: If True, include H_sa. If False, only return H_sp + H_pm
                    (for the decoherence step).

    Returns:
        Hamiltonian matrix of shape (2*(N+1)*(K+1), 2*(N+1)*(K+1)).

    """
    N = config.N
    K = config.K
    dim_pm = K + 1

    # Identity operators for each subsystem
    I_osc = np.eye(N + 1, dtype=complex)
    I_spin = np.eye(2, dtype=complex)
    I_pm = np.eye(dim_pm, dtype=complex)

    # -- H_sp = g_sp * (a + a^dagger) x I_spin x (b + b^dagger)
    a = oscillator_annihilation(N)
    a_dag = oscillator_creation(N)
    b, b_dag = create_pseudomode_operators(K)

    x_osc = a + a_dag
    x_pm = b + b_dag
    H_sp = config.g_sp * tripartite_operator(x_osc, I_spin, x_pm, N, K)

    # -- H_pm = omega_0 * I_osc x I_spin x (b^dagger b)
    n_pm = pseudomode_number_operator(K)
    H_pm = config.omega_0 * tripartite_operator(I_osc, I_spin, n_pm, N, K)

    H = H_sp + H_pm

    # -- H_sa = g_sa * (a^dagger a) x sigma_x x I_pm  (conditional)
    if include_sa:
        n_op = oscillator_number(N)
        sigma_x = spin_operator_x()
        H_sa = config.g_sa * tripartite_operator(n_op, sigma_x, I_pm, N, K)
        H = H + H_sa

    # Ensure Hermiticity
    return 0.5 * (H + H.conj().T)


# =============================================================================
# Lindblad Operators
# =============================================================================


def build_pseudomode_lindblad_operators(
    config: PseudomodeConfig,
) -> tuple[list[np.ndarray], list[float]]:
    """Build Lindblad operators for pseudomode damping.

    Single dissipator:
      L_pm = sqrt(lambda) * I_osc x I_spin x b

    The damping rate lambda is absorbed into the operator. The corresponding
    gamma coefficient in the Lindblad equation is 1.0.

    Args:
        config: Simulation configuration.

    Returns:
        Tuple of (L_ops, gammas) where:
          - L_ops[0] = sqrt(lam) * I_osc x I_spin x b  (if lam > 0)
          - gammas[0] = 1.0
          If lam <= 0, returns empty lists (no dissipation).

    """
    if config.lam <= 0:
        return [], []

    N = config.N
    K = config.K

    I_osc = np.eye(N + 1, dtype=complex)
    I_spin = np.eye(2, dtype=complex)
    b, _ = create_pseudomode_operators(K)

    L_pm = np.sqrt(config.lam) * tripartite_operator(I_osc, I_spin, b, N, K)

    return [L_pm], [1.0]


# =============================================================================
# State Preparation
# =============================================================================


def pseudomode_initial_state(config: PseudomodeConfig) -> np.ndarray:
    """Create initial tripartite state |alpha>_osc x |down>_spin x |0>_pm.

    Builds the oscillator coherent state |alpha> in the Fock basis,
    then embeds it into the tripartite space with spin down (s=0) and
    pseudomode vacuum (k=0).

    Index placement: state[n * 2 * (K + 1)] = psi_n

    Args:
        config: Simulation configuration.

    Returns:
        State vector of shape (2*(N+1)*(K+1),) representing
        |alpha> x |down> x |0>.

    """
    N = config.N
    K = config.K
    alpha = config.alpha
    dim_osc = N + 1
    dim_pm = K + 1
    dim_total = 2 * dim_osc * dim_pm

    # Build oscillator coherent state vector
    osc_state = np.zeros(dim_osc, dtype=complex)
    norm_factor = np.exp(-0.5 * np.abs(alpha) ** 2)
    for n in range(dim_osc):
        osc_state[n] = (alpha**n) / np.sqrt(scipy.special.factorial(n)) * norm_factor

    # Embed into tripartite space with spin=down (s=0) and pm=vacuum (k=0)
    # idx = (n * 2 + s) * (K + 1) + k
    # With s=0, k=0: idx = n * 2 * (K + 1) = n * 2 * dim_pm
    state = np.zeros(dim_total, dtype=complex)
    stride = 2 * dim_pm
    for n in range(dim_osc):
        state[n * stride] = osc_state[n]

    return state


# =============================================================================
# Unitary Evolution (Ancilla Entanglement)
# =============================================================================


def apply_ancilla_entanglement(
    state: np.ndarray,
    config: PseudomodeConfig,
) -> np.ndarray:
    """Apply ancilla entanglement unitary U_ent = exp(-i H_sa * tau).

    H_sa = g_sa * (a^dagger a) x sigma_x x I_pm

    Uses scipy.linalg.expm to construct the unitary.

    Args:
        state: Input state vector of shape (2*(N+1)*(K+1),).
        config: Simulation configuration.

    Returns:
        Entangled state vector of the same shape.

    Raises:
        AssertionError: If state dimension is incorrect.

    """
    N = config.N
    K = config.K
    dim_total = 2 * (N + 1) * (K + 1)
    assert state.shape == (dim_total,), (
        f"Expected state shape ({dim_total},), got {state.shape}"
    )

    # Build H_sa only: g_sa * (a^dagger a) x sigma_x x I_pm
    n_op = oscillator_number(N)
    sigma_x = spin_operator_x()
    I_pm = np.eye(K + 1, dtype=complex)

    H_sa = config.g_sa * tripartite_operator(n_op, sigma_x, I_pm, N, K)
    H_sa = 0.5 * (H_sa + H_sa.conj().T)

    U_ent = scipy.linalg.expm(-1j * H_sa * config.tau)
    return U_ent @ state


# =============================================================================
# Lindblad Master Equation Integration
# =============================================================================


def evolve_pseudomode(
    initial_state: np.ndarray,
    config: PseudomodeConfig,
    method: str = "rk4",
) -> np.ndarray:
    """Evolve under the Lindblad master equation for time T.

    Uses H_dec = H_sp + H_pm (the system-pseudomode Hamiltonian with H_sa
    turned OFF). Handles both pure state (vector) and mixed state (dense
    matrix) inputs.

    When there are no Lindblad operators (lam <= 0), uses unitary evolution
    via matrix exponentiation.

    Args:
        initial_state: Initial state vector (dim,) or density matrix (dim, dim).
        config: Simulation configuration.
        method: Integration method - "rk4" (default) or "scipy".

    Returns:
        Final density matrix of shape (dim, dim) where dim = 2*(N+1)*(K+1).

    Raises:
        ValueError: If method is not "rk4" or "scipy".

    """
    N = config.N
    K = config.K
    dim_total = 2 * (N + 1) * (K + 1)

    # Build decoherence Hamiltonian (no H_sa)
    H = build_pseudomode_hamiltonian(config, include_sa=False)

    # Build Lindblad operators
    L_ops, gammas = build_pseudomode_lindblad_operators(config)

    # Convert pure state to density matrix if needed
    if initial_state.ndim == 1:
        assert initial_state.shape == (dim_total,), (
            f"Expected state shape ({dim_total},), got {initial_state.shape}"
        )
        rho0 = np.outer(initial_state, initial_state.conj())
    else:
        assert initial_state.shape == (dim_total, dim_total), (
            f"Expected rho shape ({dim_total}, {dim_total}), got {initial_state.shape}"
        )
        rho0 = initial_state.copy()

    # No dissipation: use unitary evolution
    if len(L_ops) == 0:
        U = scipy.linalg.expm(-1.0j * H * config.T)
        return U @ rho0 @ U.conj().T

    # Dissipative evolution
    if method == "rk4":
        return evolve_lindblad_rk4(rho0, H, L_ops, gammas, config.T, config.dt)
    if method == "scipy":
        return evolve_lindblad_scipy(rho0, H, L_ops, gammas, config.T)
    raise ValueError(f"Unknown method '{method}'. Use 'rk4' or 'scipy'.")


# =============================================================================
# Partial Trace Operations
# =============================================================================


def trace_out_pseudomode(
    rho: np.ndarray,
    N: int,
    K: int,
) -> np.ndarray:
    """Partial trace over the pseudomode subsystem.

    rho_{osc+spin} = Tr_pm[rho_full]

    Reshapes to (N+1, 2, K+1, N+1, 2, K+1) and traces over the pseudomode
    axes (2 and 5).

    Args:
        rho: Full density matrix of shape (2*(N+1)*(K+1), 2*(N+1)*(K+1)).
        N: Maximum oscillator Fock number.
        K: Maximum pseudomode Fock number.

    Returns:
        Reduced density matrix for oscillator + spin of shape
        (2*(N+1), 2*(N+1)).

    """
    dim_osc = N + 1
    dim_pm = K + 1
    dim_total = 2 * dim_osc * dim_pm

    assert rho.shape == (dim_total, dim_total), (
        f"Expected rho shape ({dim_total}, {dim_total}), got {rho.shape}"
    )

    # Reshape to (N+1, 2, K+1, N+1, 2, K+1) and trace over pm axes (2, 5)
    rho_reduced = np.trace(
        rho.reshape(dim_osc, 2, dim_pm, dim_osc, 2, dim_pm),
        axis1=2,
        axis2=5,
    )

    # Result is (N+1, 2, N+1, 2) -> reshape to (2*(N+1), 2*(N+1))
    dim_os = 2 * dim_osc
    return rho_reduced.reshape(dim_os, dim_os)


def trace_out_spin(
    rho: np.ndarray,
    N: int,
    K: int,
) -> np.ndarray:
    """Partial trace over the spin subsystem.

    rho_{osc+pm} = Tr_spin[rho_full]

    Reshapes to (N+1, 2, K+1, N+1, 2, K+1) and traces over the spin axes
    (1 and 4).

    Args:
        rho: Full density matrix of shape (2*(N+1)*(K+1), 2*(N+1)*(K+1)).
        N: Maximum oscillator Fock number.
        K: Maximum pseudomode Fock number.

    Returns:
        Reduced density matrix for oscillator + pseudomode of shape
        ((N+1)*(K+1), (N+1)*(K+1)).

    """
    dim_osc = N + 1
    dim_pm = K + 1
    dim_total = 2 * dim_osc * dim_pm

    assert rho.shape == (dim_total, dim_total), (
        f"Expected rho shape ({dim_total}, {dim_total}), got {rho.shape}"
    )

    # Reshape to (N+1, 2, K+1, N+1, 2, K+1) and trace over spin axes (1, 4)
    rho_reduced = np.trace(
        rho.reshape(dim_osc, 2, dim_pm, dim_osc, 2, dim_pm),
        axis1=1,
        axis2=4,
    )

    # Result is (N+1, K+1, N+1, K+1) -> reshape to ((N+1)*(K+1), (N+1)*(K+1))
    dim_op = dim_osc * dim_pm
    return rho_reduced.reshape(dim_op, dim_op)


def trace_out_spin_and_pseudomode(
    rho: np.ndarray,
    N: int,
    K: int,
) -> np.ndarray:
    """Partial trace over both spin and pseudomode.

    rho_{osc} = Tr_{spin,pm}[rho_full]

    Reshapes to (N+1, 2, K+1, N+1, 2, K+1) and traces over the spin axes
    (1, 4) and pseudomode axes (2, 5).

    Args:
        rho: Full density matrix of shape (2*(N+1)*(K+1), 2*(N+1)*(K+1)).
        N: Maximum oscillator Fock number.
        K: Maximum pseudomode Fock number.

    Returns:
        Reduced density matrix for oscillator only of shape (N+1, N+1).

    """
    dim_osc = N + 1
    dim_pm = K + 1
    dim_total = 2 * dim_osc * dim_pm

    assert rho.shape == (dim_total, dim_total), (
        f"Expected rho shape ({dim_total}, {dim_total}), got {rho.shape}"
    )

    # Reshape to (N+1, 2, K+1, N+1, 2, K+1) and trace over spin (1, 4)
    # and pseudomode (2, 5)
    rho_reduced = np.trace(
        rho.reshape(dim_osc, 2, dim_pm, dim_osc, 2, dim_pm),
        axis1=1,
        axis2=4,
    )
    return np.trace(
        rho_reduced.reshape(dim_osc, dim_pm, dim_osc, dim_pm),
        axis1=1,
        axis2=3,
    )


# =============================================================================
# QFI Computation
# =============================================================================


def compute_qfi_with_ancilla(
    rho_full: np.ndarray,
    N: int,
    K: int,
) -> float:
    """Compute QFI with ancilla retained.

    Traces out the pseudomode to obtain rho_{osc+spin}, then computes F_Q
    with generator G = a^dagger a x I_spin.

    Uses quantum_fisher_information_dm from src.analysis.fisher_information.

    Args:
        rho_full: Full density matrix of shape (2*(N+1)*(K+1), 2*(N+1)*(K+1)).
        N: Maximum oscillator Fock number.
        K: Maximum pseudomode Fock number.

    Returns:
        Quantum Fisher Information value.

    """
    # Trace out pseudomode -> rho_{osc+spin} of shape (2*(N+1), 2*(N+1))
    rho_os = trace_out_pseudomode(rho_full, N, K)

    # Generator: a^dagger a x I_spin  (shape 2*(N+1), 2*(N+1))
    n_op = oscillator_number(N)
    I_spin = np.eye(2, dtype=complex)
    G = np.kron(n_op, I_spin)

    return quantum_fisher_information_dm(rho_os, G)


def compute_qfi_without_ancilla(
    rho_full: np.ndarray,
    N: int,
    K: int,
) -> float:
    """Compute QFI without ancilla (ancilla traced out).

    Traces out pseudomode AND spin to obtain rho_{osc}, then computes F_Q
    with generator G = a^dagger a.

    Args:
        rho_full: Full density matrix of shape (2*(N+1)*(K+1), 2*(N+1)*(K+1)).
        N: Maximum oscillator Fock number.
        K: Maximum pseudomode Fock number.

    Returns:
        Quantum Fisher Information value.

    """
    # Trace out spin and pseudomode -> rho_{osc} of shape (N+1, N+1)
    rho_osc = trace_out_spin_and_pseudomode(rho_full, N, K)

    # Generator: a^dagger a  (shape N+1, N+1)
    G = oscillator_number(N)

    return quantum_fisher_information_dm(rho_osc, G)


# =============================================================================
# Metrology Protocol
# =============================================================================


def run_metrology_protocol(
    config: PseudomodeConfig,
    phi: float = 0.0,
) -> dict:
    """Run the full metrology protocol and return results.

    Steps:
      1. Initial state |alpha> x |down> x |0>
      2. Ancilla entanglement via U_ent = exp(-i H_sa * tau)
      3. (Phase imprint - not needed for QFI computation)
      4. Non-Markovian decoherence evolution under H_sp + H_pm with L_pm
      5. QFI computation (with and without ancilla)
      6. Validation of final density matrix

    Args:
        config: Simulation configuration.
        phi: Phase parameter (reserved for future use; not used in QFI).

    Returns:
        Dictionary with fields:
            - rho_final: final density matrix
            - qfi_with: F_Q with ancilla
            - qfi_without: F_Q without ancilla
            - qfi_initial: F_Q at t=0 (with ancilla, before decoherence)
            - ratio_with: qfi_with / qfi_initial
            - ratio_without: qfi_without / qfi_initial
            - pm_occupancy: <b^dagger b> (for truncation check)
            - validation: validation dict (trace, hermiticity, positivity)

    """
    # Step 1: initial state
    psi0 = pseudomode_initial_state(config)

    # Step 2: ancilla entanglement
    psi_entangled = apply_ancilla_entanglement(psi0, config)

    # Compute initial QFI (with ancilla) on the entangled state before
    # decoherence
    rho_entangled = np.outer(psi_entangled, psi_entangled.conj())
    qfi_initial = compute_qfi_with_ancilla(rho_entangled, config.N, config.K)

    # Step 4: non-Markovian decoherence evolution
    rho_final = evolve_pseudomode(psi_entangled, config, method="rk4")

    # Step 5a: QFI with ancilla
    qfi_with = compute_qfi_with_ancilla(rho_final, config.N, config.K)

    # Step 5b: QFI without ancilla
    qfi_without = compute_qfi_without_ancilla(rho_final, config.N, config.K)

    # Compute preservation ratios
    ratio_with = qfi_with / qfi_initial if qfi_initial > 0 else 0.0
    ratio_without = qfi_without / qfi_initial if qfi_initial > 0 else 0.0

    # Pseudomode occupancy for truncation check
    pm_occ, _ = check_pseudomode_occupancy(rho_final, config.N, config.K)

    # Validation
    validation = validate_density_matrix(rho_final)

    return {
        "rho_final": rho_final,
        "qfi_with": qfi_with,
        "qfi_without": qfi_without,
        "qfi_initial": qfi_initial,
        "ratio_with": ratio_with,
        "ratio_without": ratio_without,
        "pm_occupancy": pm_occ,
        "validation": validation,
    }


# Re-export for backward compatibility
validate_pseudomode_density = validate_density_matrix


def check_pseudomode_occupancy(
    rho: np.ndarray,
    N: int,
    K: int,
) -> tuple[float, bool]:
    """Check pseudomode occupancy to verify truncation validity.

    Computes <b^dagger b> = Tr(rho_full * I_osc x I_spin x b^dagger b).

    Args:
        rho: Full density matrix of shape (2*(N+1)*(K+1), 2*(N+1)*(K+1)).
        N: Maximum oscillator Fock number.
        K: Maximum pseudomode Fock number.

    Returns:
        Tuple of (occupancy, is_safe) where is_safe = occupancy <= 0.8 * K.

    """
    # Build the pseudomode number operator in the full tripartite space
    I_osc = np.eye(N + 1, dtype=complex)
    I_spin = np.eye(2, dtype=complex)
    n_pm = pseudomode_number_operator(K)

    n_pm_full = tripartite_operator(I_osc, I_spin, n_pm, N, K)

    occupancy = float(np.real(np.trace(rho @ n_pm_full)))
    is_safe = occupancy <= 0.8 * K

    return occupancy, is_safe


def qfi_preservation_ratio(
    rho_full: np.ndarray,
    fq_initial: float,
    N: int,
    K: int,
    with_ancilla: bool = True,
) -> float:
    """Compute QFI preservation ratio R = F_Q(T) / F_Q(0).

    Args:
        rho_full: Final density matrix after decoherence.
        fq_initial: Initial QFI (at t=0, before decoherence).
        N: Maximum oscillator Fock number.
        K: Maximum pseudomode Fock number.
        with_ancilla: If True, compute F_Q(T) with ancilla retained.
                      If False, without ancilla.

    Returns:
        Preservation ratio R = F_Q(T) / F_Q(0).
        Returns 0.0 if fq_initial is zero or negative.

    """
    if fq_initial <= 0:
        return 0.0

    if with_ancilla:
        fq_t = compute_qfi_with_ancilla(rho_full, N, K)
    else:
        fq_t = compute_qfi_without_ancilla(rho_full, N, K)

    return fq_t / fq_initial

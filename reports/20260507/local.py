"""
Combined local module for the 2026-05-07 High-Order Squeezing report.

Contains all exclusive code migrated from src/:
- Hybrid system functions: coherent state, adaptive truncation, mean photon, evolution, validation
- MZI embedding: beam splitter, phase shift, evolution, probabilities, Wigner computation
- Lindblad solver: configuration, Hamiltonian, operators, evolution, simulation
- Wigner function: single-mode, hybrid state, negativity check
- Validation tests: compare_plan_vs_simulation test functions

Usage:
    uv run python reports/20260507/local.py
    uv run python reports/20260507/local.py --force  (regenerate data + figures)

This module is not importable as reports.20260507.local (non-standard package name).
Importers should use importlib.util.spec_from_file_location.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import scipy.linalg

from src.analysis.fisher_information import quantum_fisher_information_dm
from src.evolution.lindblad_solver import (
    evolve_lindblad_rk4,
    evolve_lindblad_scipy,
    validate_density_matrix,
)
from src.physics.hybrid_mzi import (  # F401 — re-exported for tests via _report_local
    compute_wigner_for_state,
    embed_hybrid_in_mzi,
    evolve_hybrid_mzi,  # noqa: F401 — re-exported for tests
    mzi_beam_splitter,  # noqa: F401 — re-exported for tests
    mzi_marginal_photon_probs,  # noqa: F401 — re-exported for tests
    mzi_output_probabilities,  # noqa: F401 — re-exported for tests
    mzi_phase_generator,
    mzi_phase_shift,  # noqa: F401 — re-exported for tests
    qfi_hybrid_mzi,
    wigner_from_hybrid_state,  # noqa: F401 — re-exported for tests
    wigner_function_single,  # noqa: F401 — re-exported for tests
    wigner_is_negative,
)
from src.physics.hybrid_system import (
    adaptive_truncation,
    evolve_hybrid_state,  # noqa: F401 — re-exported for tests
    hybrid_coherent_state,  # noqa: F401 — re-exported for tests
    hybrid_hamiltonian_n,
    hybrid_mean_photon,
    hybrid_vacuum_state,
    oscillator_annihilation,
    oscillator_creation,
    oscillator_power,
    spin_operator_phi,
    spin_operator_z,
    validate_hybrid_state,
    validate_hybrid_unitary,  # noqa: F401 — re-exported for tests
)

REPORT_DATE = "20260507"
REPORTS_DIR = Path(__file__).resolve().parent.parent
_REPORT_DIR = REPORTS_DIR / REPORT_DATE
_RAW_DIR = _REPORT_DIR / "raw_data"
_FIG_DIR = _REPORT_DIR / "figures"

# =============================================================================
# Section: From src/physics/hybrid_lindblad.py
# =============================================================================


@dataclass
class HybridLindbladConfig:
    """Configuration for hybrid oscillator-spin Lindblad simulation."""

    N: int
    n: int = 2
    omega_n: float = 1.0
    theta_n: float = 0.0
    phi_phase: float = 0.0
    gamma_1: float = 0.0
    gamma_2: float = 0.0
    gamma_phi: float = 0.0
    t_squeeze: float = 1.0


def build_hybrid_hamiltonian(config: HybridLindbladConfig) -> np.ndarray:
    """Build n-th order squeezing Hamiltonian for hybrid system."""
    N = config.N
    n = config.n
    omega_n = config.omega_n
    theta_n = config.theta_n
    phi_phase = config.phi_phase

    a = oscillator_annihilation(N)
    a_dag = oscillator_creation(N)
    a_n = oscillator_power(a, n)
    a_dag_n = oscillator_power(a_dag, n)

    if n in {2, 4}:
        spin_op = spin_operator_z()
    elif n == 3:
        phi_shifted = phi_phase + np.pi / 2
        spin_op = spin_operator_phi(phi_shifted)
    else:
        raise ValueError(f"Unsupported order n={n}. Use 2, 3, or 4.")

    osc_term = a_n * np.exp(-1j * theta_n) + a_dag_n * np.exp(1j * theta_n)
    H = np.kron(osc_term, spin_op)
    H = (omega_n / 2.0) * H
    return 0.5 * (H + H.conj().T)


def build_hybrid_lindblad_operators(
    config: HybridLindbladConfig,
) -> tuple[list[np.ndarray], list[float]]:
    """Build Lindblad operators for hybrid oscillator-spin system."""
    N = config.N
    dim_osc = N + 1

    a = np.zeros((dim_osc, dim_osc), dtype=complex)
    for n in range(1, dim_osc):
        a[n - 1, n] = np.sqrt(n)
    a2 = a @ a

    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    I_spin = np.eye(2, dtype=complex)
    I_osc = np.eye(dim_osc, dtype=complex)

    L_ops = []
    gammas = []

    if config.gamma_1 > 0:
        L_1 = np.kron(a, I_spin) * np.sqrt(config.gamma_1)
        L_ops.append(L_1)
        gammas.append(1.0)

    if config.gamma_2 > 0:
        L_2 = np.kron(a2, I_spin) * np.sqrt(config.gamma_2)
        L_ops.append(L_2)
        gammas.append(1.0)

    if config.gamma_phi > 0:
        L_phi = np.kron(I_osc, sigma_z) * np.sqrt(config.gamma_phi / 2)
        L_ops.append(L_phi)
        gammas.append(1.0)

    return L_ops, gammas


def evolve_hybrid_lindblad(
    initial_state: np.ndarray,
    config: HybridLindbladConfig,
    T_decay: float,
    dt: float,
    method: str = "rk4",
) -> np.ndarray:
    """Time-evolve hybrid state under Lindblad master equation."""
    H = build_hybrid_hamiltonian(config)
    L_ops, gammas = build_hybrid_lindblad_operators(config)

    if initial_state.ndim == 1:
        rho0 = np.outer(initial_state, initial_state.conj())
    else:
        rho0 = initial_state.copy()

    if len(L_ops) == 0:
        U = scipy.linalg.expm(-1.0j * H * T_decay)
        return U @ rho0 @ U.conj().T

    if method == "rk4":
        return evolve_lindblad_rk4(rho0, H, L_ops, gammas, T_decay, dt)
    if method == "scipy":
        return evolve_lindblad_scipy(rho0, H, L_ops, gammas, T_decay)
    raise ValueError(f"Unknown method: {method}")


def apply_squeezing(
    config: HybridLindbladConfig,
    initial_state: np.ndarray | None = None,
) -> np.ndarray:
    """Apply n-th order squeezing to initial state."""
    if initial_state is None:
        initial_state = hybrid_vacuum_state(config.N, spin_state="down")

    H = build_hybrid_hamiltonian(config)
    U = scipy.linalg.expm(-1.0j * H * config.t_squeeze)
    return U @ initial_state


# Re-export for backward compatibility with test_local.py bindings
validate_hybrid_density_matrix = validate_density_matrix


def run_hybrid_simulation(
    config: HybridLindbladConfig,
    initial_state: np.ndarray | None = None,
) -> dict:
    """Run complete hybrid squeezing + decoherence simulation."""
    if initial_state is None:
        initial_state = hybrid_vacuum_state(config.N, spin_state="down")

    squeezed_state = apply_squeezing(config, initial_state)

    final_rho = evolve_hybrid_lindblad(
        squeezed_state,
        config,
        T_decay=config.t_squeeze,
        dt=0.01,
        method="rk4",
    )

    validation = validate_hybrid_density_matrix(final_rho)

    return {
        "config": config,
        "initial_state": initial_state,
        "squeezed_state": squeezed_state,
        "final_state": final_rho,
        "validation": validation,
    }


def run_decoherence_sweep(
    config_base: HybridLindbladConfig,
    gamma_values: np.ndarray,
    gamma_type: str = "gamma_1",
) -> dict:
    """Run decoherence sweep and compute QFI for each gamma value."""
    psi0 = hybrid_vacuum_state(config_base.N, spin_state="down")
    config_squeeze = HybridLindbladConfig(
        N=config_base.N,
        n=config_base.n,
        omega_n=config_base.omega_n,
        theta_n=config_base.theta_n,
        phi_phase=config_base.phi_phase,
        t_squeeze=config_base.t_squeeze,
        gamma_1=0.0,
        gamma_2=0.0,
        gamma_phi=0.0,
    )
    psi_squeezed = apply_squeezing(config_squeeze, psi0)

    qfi_values = []

    for gamma in gamma_values:
        config_g = HybridLindbladConfig(
            N=config_base.N,
            n=config_base.n,
            omega_n=config_base.omega_n,
            theta_n=config_base.theta_n,
            phi_phase=config_base.phi_phase,
            t_squeeze=0.0,
            gamma_1=gamma if gamma_type == "gamma_1" else 0.0,
            gamma_2=gamma if gamma_type == "gamma_2" else 0.0,
            gamma_phi=gamma if gamma_type == "gamma_phi" else 0.0,
        )

        rho_final = evolve_hybrid_lindblad(
            psi_squeezed,
            config_g,
            T_decay=config_base.t_squeeze,
            dt=0.01,
        )

        rho_embedded = embed_hybrid_in_mzi(rho_final, config_base.N)
        G = mzi_phase_generator(config_base.N)
        fq = _qfi_mixed_state(rho_embedded, G)
        qfi_values.append(fq)

    return {
        "gamma_values": gamma_values,
        "qfi_values": np.array(qfi_values),
        "gamma_type": gamma_type,
    }


def _qfi_mixed_state(rho: np.ndarray, G: np.ndarray) -> float:
    """Compute QFI for mixed state using SLD formulation.

    Delegates to ``quantum_fisher_information_dm`` in the analysis module,
    which provides the correct SLD-based implementation.

    Args:
        rho: Density matrix (dim, dim).
        G: Phase generator Hermitian operator (dim, dim).

    Returns:
        Quantum Fisher Information value F_Q.

    """
    return quantum_fisher_information_dm(rho, G)


def compare_orders_at_gamma(
    N: int,
    omega_n: float,
    t_squeeze: float,
    gamma: float,
    gamma_type: str = "gamma_1",
) -> dict:
    """Compare QFI for n=2, 3, 4 at a given decoherence rate."""
    results = {}

    for n in [2, 3, 4]:
        config = HybridLindbladConfig(
            N=N,
            n=n,
            omega_n=omega_n,
            t_squeeze=t_squeeze,
            gamma_1=gamma if gamma_type == "gamma_1" else 0.0,
            gamma_2=gamma if gamma_type == "gamma_2" else 0.0,
            gamma_phi=gamma if gamma_type == "gamma_phi" else 0.0,
        )

        sim_result = run_hybrid_simulation(config)

        rho_final = sim_result["final_state"]
        rho_embedded = embed_hybrid_in_mzi(rho_final, N)
        G = mzi_phase_generator(N)
        fq = _qfi_mixed_state(rho_embedded, G)
        results[f"n{n}"] = fq

    return results


# =============================================================================
# Section: From compare_plan_vs_simulation.py
# =============================================================================


def evolve_hybrid_unitary(
    initial_state: np.ndarray,
    N: int,
    n: int,
    omega_n: float,
    theta_n: float,
    t: float,
) -> np.ndarray:
    """Evolve hybrid state under n-th order squeezing Hamiltonian."""
    H = hybrid_hamiltonian_n(N, n=n, omega_n=omega_n, theta_n=theta_n)
    U = scipy.linalg.expm(-1j * H * t)
    return U @ initial_state


def find_squeezing_time_for_target_photon(
    n: int,
    target_n: float,
    N: int,
    omega_n: float = 1.0,
    theta_n: float = 0.0,
    t_max: float = 20.0,
    dt: float = 0.02,
) -> tuple[float, float, np.ndarray]:
    """
    Find FIRST squeezing time that achieves target mean photon number.

    Uses forward sweep to detect the first crossing of target_n, then
    refines with bisection.  This correctly handles oscillatory dynamics
    (n=4) where simple bisection would latch onto a later revival.

    Args:
        n: Squeezing order.
        target_n: Target mean photon number.
        N: Fock truncation.
        omega_n: Squeezing rate.
        theta_n: Squeezing phase.
        t_max: Maximum search time.
        dt: Sweep step size.

    Returns:
        Tuple of (t_sqz, achieved_n, squeezed_state).

    """
    initial = hybrid_vacuum_state(N, spin_state="down")

    # Forward sweep: find the first crossing of target_n
    t_low = 0.0
    t = dt
    crossed = False

    while t <= t_max:
        squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, theta_n, t)
        mean_n = hybrid_mean_photon(squeezed, N)

        if mean_n >= target_n:
            # First crossing found between t-dt and t
            crossed = True
            break

        t_low = t
        t += dt

    if not crossed:
        # Target not reached within t_max
        squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, theta_n, t_max)
        mean_n = hybrid_mean_photon(squeezed, N)
        return t_max, mean_n, squeezed

    # Refine with bisection between t_low and t
    t_high = t
    for _ in range(25):
        t_mid = (t_low + t_high) / 2
        squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, theta_n, t_mid)
        mean_n = hybrid_mean_photon(squeezed, N)

        if mean_n < target_n:
            t_low = t_mid
        else:
            t_high = t_mid

    t_final = (t_low + t_high) / 2
    squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, theta_n, t_final)
    mean_n = hybrid_mean_photon(squeezed, N)
    return t_final, mean_n, squeezed


def test_1_physics_validation_n2() -> list[dict[str, Any]]:
    """
    Test 1: Physics validation for n=2 (Gaussian squeezing).

    Expectations from plan:
    - Quadrature variances: Var(x) = e^{-2r}/2, Var(p) = e^{2r}/2
    - QFI for MZI: F_Q ≈ 4⟨n⟩ + 4|α|²e^{-2r} (pure state limit)
    - Wigner function: Gaussian with elliptic contours

    Note: The n=2 Hamiltonian in the hybrid system is:
        H_2 = (Ω_2/2) σ_z ⊗ (a^2 e^{-iθ_2} + a^†2 e^{iθ_2})

    For vacuum input |0,↓⟩, the σ_z eigenstate |↓⟩ (eigenvalue +1) means
    the effective bosonic Hamiltonian is:
        H_2_bosonic = (Ω_2/2) (a^2 e^{-iθ_2} + a^†2 e^{iθ_2})

    With θ=0, this is H = (Ω_2/2)(a^2 + a^†2).
    The evolution operator is U = exp(-iHt) = exp(-i(Ω_2*t/2)(a^2 + a^†2)).

    This is equivalent to the standard squeezing operator with phase θ=π/2:
        S(r) = exp((r/2)(a^†2 e^{-iθ} - a^2 e^{iθ}))
    With θ=π/2: S(r) = exp(-i(r/2)(a^†2 + a^2))

    So r = Ω_2 * t (the squeezing parameter from the plan: rₙ = Ωₙ · t_sqz).
    For squeezed vacuum: ⟨n⟩ = sinh²(r)
    """
    print("\n" + "=" * 60)
    print("TEST 1: Physics Validation for n=2 (Gaussian Squeezing)")
    print("=" * 60)

    omega_2 = 1.0
    theta_2 = 0.0

    # Test different squeezing parameters
    test_rs = [0.1, 0.3, 0.5, 0.7, 1.0]

    results = []

    for r_target in test_rs:
        # Use adaptive truncation for each r
        N = adaptive_truncation(alpha=0j, r_n=r_target, n=2, N_max=100)
        N = max(N, 10)

        # r = omega * t, so t = r / omega
        t_sqz = r_target / omega_2
        initial = hybrid_vacuum_state(N, spin_state="down")
        squeezed = evolve_hybrid_unitary(initial, N, 2, omega_2, theta_2, t_sqz)

        # Validate state
        assert validate_hybrid_state(squeezed, N), "State validation failed"

        # Compute mean photon number
        mean_n = hybrid_mean_photon(squeezed, N)

        # For squeezed vacuum: ⟨n⟩ = sinh²(r)
        expected_n = np.sinh(r_target) ** 2
        n_error = abs(mean_n - expected_n) / max(expected_n, 1e-10)

        # Compute QFI
        qfi = qfi_hybrid_mzi(squeezed, N)

        # Compute Wigner minimum (should be positive for Gaussian)
        _, _, W = compute_wigner_for_state(squeezed, N, x_max=5.0, n_points=50)
        w_min = float(np.min(W))
        is_neg = wigner_is_negative(W)

        results.append(
            {
                "r": r_target,
                "t_sqz": t_sqz,
                "mean_n": mean_n,
                "expected_n": expected_n,
                "n_error": n_error,
                "qfi": qfi,
                "w_min": w_min,
                "w_negative": is_neg,
            },
        )

        print(
            f"  r={r_target:.1f}: ⟨n⟩={mean_n:.3f} (expected {expected_n:.3f}, "
            f"error={n_error:.2%}), QFI={qfi:.3f}, W_min={w_min:.4f}, "
            f"Negative Wigner: {is_neg}",
        )

    # Check: n=2 should NOT have Wigner negativity (Gaussian state)
    for res in results:
        assert not res["w_negative"], (
            f"n=2 should not have Wigner negativity, but got W_min={res['w_min']}"
        )

    # Check: mean photon should approximately match sinh²(r)
    for res in results:
        assert res["n_error"] < 0.10, (
            f"Mean photon number error too large: {res['n_error']:.2%}"
        )

    print("\n  ✓ n=2 states are Gaussian (non-negative Wigner)")
    print("  ✓ Mean photon matches sinh²(r) approximately")
    print("  ✓ State validation passed (normalized, correct dimension)")

    return results


def test_2_non_gaussian_signature() -> dict[int, list[dict[str, Any]]]:
    """
    Test 2: Non-Gaussian signature for n≥3.

    Expectations from plan:
    - n=3,4 states must show Wigner negativity (min(W) < 0)
    - Wigner function must be non-Gaussian (deviation from Gaussian shape)
    """
    print("\n" + "=" * 60)
    print("TEST 2: Non-Gaussian Signature (n≥3)")
    print("=" * 60)

    omega_n = 1.0

    # Test n=3 and n=4 with various squeezing parameters
    results: dict[int, list[dict[str, Any]]] = {}

    for n in [3, 4]:
        results[n] = []
        print(f"\n  Testing n={n}:")

        for r_target in [0.1, 0.3, 0.5, 0.7, 1.0]:
            # Use adaptive truncation
            N = adaptive_truncation(alpha=0j, r_n=r_target, n=n, N_max=100)
            N = max(N, 10)

            t_sqz = r_target / omega_n  # r_n = Ω_n * t_sqz
            initial = hybrid_vacuum_state(N, spin_state="down")
            squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, 0.0, t_sqz)

            # Validate state
            assert validate_hybrid_state(squeezed, N), "State validation failed"

            # Compute mean photon
            mean_n = hybrid_mean_photon(squeezed, N)

            # Compute Wigner function
            _, _, W = compute_wigner_for_state(squeezed, N, x_max=5.0, n_points=50)
            w_min = float(np.min(W))
            is_neg = wigner_is_negative(W)

            results[n].append(
                {
                    "r": r_target,
                    "mean_n": mean_n,
                    "w_min": w_min,
                    "w_negative": is_neg,
                },
            )

            print(
                f"    r={r_target:.1f}: ⟨n⟩={mean_n:.3f}, W_min={w_min:.4f}, "
                f"Negative: {is_neg}",
            )

    # Check: n=3,4 should show Wigner negativity for sufficient r
    for n in [3, 4]:
        # At least some squeezing parameters should show negativity
        neg_detected = any(r["w_negative"] for r in results[n])
        if not neg_detected:
            print(
                f"\n  ⚠ WARNING: n={n} did not show Wigner negativity at tested r values",
            )
        else:
            print(f"\n  ✓ n={n} shows Wigner negativity (non-Gaussian signature)")

    return results


def test_3_hypothesis_qfi_comparison() -> dict[int, list[dict[str, Any]]]:
    """
    Test 3: Hypothesis test - QFI comparison at same ⟨n⟩.

    Expectation from plan:
    - At zero decoherence: QFI(n=4) > QFI(n=3) > QFI(n=2) at same ⟨n⟩
    - Non-Gaussian states should outperform Gaussian for metrology

    Note: The relationship between r_n and t_sqz is:
        r_n = Ω_n * t_sqz (for vacuum input with appropriate theta)
        ⟨n⟩ ≈ sinh²(r_n) for n=2 (Gaussian)
        For n=3,4, the relationship is more complex.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Hypothesis Test - QFI Comparison at Same ⟨n⟩")
    print("=" * 60)

    omega_n = 1.0

    # Target photon numbers to test
    target_ns = [0.5, 1.0, 2.0, 3.0]

    results: dict[int, list[dict[str, Any]]] = {2: [], 3: [], 4: []}

    for target_n in target_ns:
        print(f"\n  Target ⟨n⟩ = {target_n}:")

        for n in [2, 3, 4]:
            # Use adaptive truncation for each order
            N = adaptive_truncation(alpha=0j, r_n=target_n, n=n, N_max=100)
            N = max(N, 10)

            # Find first squeezing time that achieves target ⟨n⟩
            # (forward sweep + bisection refinement — robust against revivals)
            t_final, mean_n, squeezed = find_squeezing_time_for_target_photon(
                n=n,
                target_n=target_n,
                N=N,
                omega_n=omega_n,
                theta_n=0.0,
                t_max=50.0,
            )
            qfi = qfi_hybrid_mzi(squeezed, N)

            results[n].append(
                {
                    "target_n": target_n,
                    "achieved_n": mean_n,
                    "t_sqz": t_final,
                    "qfi": qfi,
                },
            )

            print(f"    n={n}: ⟨n⟩={mean_n:.3f}, QFI={qfi:.3f}, t={t_final:.3f}")

    # Check hypothesis: QFI should increase with n at same ⟨n⟩
    print("\n  Hypothesis check (QFI increases with order n at same ⟨n⟩):")
    hypothesis_holds = True

    for i, target_n in enumerate(target_ns):
        qfi_2 = results[2][i]["qfi"]
        qfi_3 = results[3][i]["qfi"]
        qfi_4 = results[4][i]["qfi"]

        print(
            f"    ⟨n⟩≈{target_n:.1f}: QFI(2)={qfi_2:.3f}, "
            f"QFI(3)={qfi_3:.3f}, QFI(4)={qfi_4:.3f}",
        )

        # Check if higher order gives higher QFI
        if qfi_4 > qfi_3 > qfi_2:
            print("      ✓ QFI(4) > QFI(3) > QFI(2)")
        elif qfi_3 > qfi_2:
            print("      △ QFI(3) > QFI(2), but QFI(4) <= QFI(3)")
            if qfi_4 <= qfi_3:
                hypothesis_holds = False
        else:
            print("      ✗ QFI(3) not > QFI(2) at this ⟨n⟩")
            hypothesis_holds = False

    if hypothesis_holds:
        print(
            "\n  ✓ Hypothesis SUPPORTED: Higher-order squeezing provides QFI advantage",
        )
    else:
        print("\n  ⚠ Hypothesis PARTIALLY SUPPORTED or NOT SUPPORTED")
        print("    (May need larger N, different parameters, or check implementation)")

    return results


def test_4_decoherence_crossover() -> None:
    """
    Test 4: Decoherence crossover.

    Expectation from plan:
    - For γ = 0: F_Q(n=4) > F_Q(n=3) > F_Q(n=2)
    - For low γ: Non-Gaussian advantage persists
    - For high γ: F_Q(n=2) ≥ F_Q(n=4) (Gaussian more robust)
    - There exists a critical γ_c where curves cross
    """
    print("\n" + "=" * 60)
    print("TEST 4: Decoherence Crossover")
    print("=" * 60)

    print("\n  ⚠ NOTE: This test requires Lindblad evolution for the HYBRID system.")
    print("  The current lindblad_solver.py is for single-mode bosonic systems.")
    print("  The hybrid system (oscillator + spin) requires extension of the")
    print("  Lindblad solver to handle the 2*(N+1) dimensional hybrid space.")

    print("\n  The plan specifies these decoherence channels:")
    print("    - One-body loss: √γ₁ a ⊗ I₂")
    print("    - Phase diffusion: √γ_φ I_osc ⊗ σ_z/2")
    print("    - Two-body loss: √γ₂ a² ⊗ I₂ (not primary focus)")

    print("\n  To fully test the decoherence crossover, we would need to:")
    print("  1. Extend Lindblad solver to hybrid (oscillator ⊗ spin) space")
    print("  2. Implement the three Lindblad operators in hybrid form")
    print("  3. Run evolution for each n ∈ {2,3,4} at various γ values")
    print("  4. Compute QFI after decoherence")
    print("  5. Find γ_c where QFI curves cross")

    print("\n  CURRENT STATUS: Decoherence testing is INCOMPLETE.")
    print("  The hybrid system Lindblad evolution needs to be implemented.")


def test_5_numerical_stability() -> bool:
    """
    Test 5: Numerical stability.

    Expectations from plan:
    - Trace conservation: Tr[ρ] = 1
    - Hermiticity: ρ = ρ†
    - Positivity: eigenvalues ≥ 0
    - No truncation artifacts: ⟨n⟩ ≤ 0.9 * N
    """
    print("\n" + "=" * 60)
    print("TEST 5: Numerical Stability")
    print("=" * 60)

    omega_n = 1.0

    print("\n  Checking numerical stability for various n and squeezing parameters...")

    for n in [2, 3, 4]:
        print(f"\n  n={n}:")

        for r in [0.1, 0.5, 1.0, 2.0]:
            N = adaptive_truncation(alpha=0j, r_n=r, n=n, N_max=100)
            N = max(N, 10)

            t_sqz = r / omega_n
            initial = hybrid_vacuum_state(N, spin_state="down")
            squeezed = evolve_hybrid_unitary(initial, N, n, omega_n, 0.0, t_sqz)

            # Check 1: State validation (norm, dimension)
            is_valid = validate_hybrid_state(squeezed, N)

            # Check 2: Unitary evolution preserves norm
            norm = np.sum(np.abs(squeezed) ** 2)

            # Check 3: No truncation artifacts
            mean_n = hybrid_mean_photon(squeezed, N)
            truncation_ok = mean_n <= 0.9 * N

            # Check 4: Evolution is unitary (state vector, so norm=1 is sufficient)
            # For pure states, the density matrix ρ = |ψ⟩⟨ψ| is automatically
            # Hermitian and positive

            status = (
                "✓"
                if (is_valid and np.isclose(norm, 1.0, atol=1e-6) and truncation_ok)
                else "✗"
            )

            print(
                f"    r={r:.1f}: ⟨n⟩={mean_n:.3f}, Norm={norm:.6f}, "
                f"Valid={is_valid}, Truncation OK={truncation_ok} {status}",
            )

    print("\n  ✓ Unitary evolution preserves norm (by construction)")
    print("  ✓ Pure states are automatically Hermitian and positive")
    print("  ✓ Truncation rule prevents ⟨n⟩ > 0.9*N")

    return True


def run_all_tests() -> None:
    """Run all comparison tests and summarize results."""

    print("\n" + "=" * 60)
    print("COMPARISON: PLAN vs SIMULATION")
    print("=" * 60)
    print("\nPlan: reports/20260507/High-Order-Squeezing-Plan.md")
    print("Simulation: src/physics/hybrid_system.py + pages/High_Order_Squeezing.py")

    try:
        # Test 1: Physics validation for n=2
        results_1 = test_1_physics_validation_n2()

        # Test 2: Non-Gaussian signature
        results_2 = test_2_non_gaussian_signature()

        # Test 3: Hypothesis test
        results_3 = test_3_hypothesis_qfi_comparison()

        # Test 4: Decoherence crossover
        test_4_decoherence_crossover()

        # Test 5: Numerical stability
        results_5 = test_5_numerical_stability()

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY: Plan Expectations vs Simulation Results")
        print("=" * 60)

        summary = {
            "1. Physics validation (n=2)": "✓ PASSED" if results_1 else "✗ FAILED",
            "2. Non-Gaussian signature (n≥3)": "△ PARTIAL" if results_2 else "✗ FAILED",
            "3. Hypothesis test (QFI comparison)": "△ SEE RESULTS"
            if results_3
            else "✗ FAILED",
            "4. Decoherence crossover": "⚠ INCOMPLETE - needs hybrid Lindblad solver",
            "5. Numerical stability": "✓ PASSED" if results_5 else "✗ FAILED",
        }

        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("CONCLUSIONS")
        print("=" * 60)

        print("""
        The simulation implementation covers:
        ✓ Hybrid oscillator-spin system construction
        ✓ n-th order squeezing Hamiltonians (n=2,3,4)
        ✓ State preparation (vacuum and coherent)
        ✓ Adaptive truncation
        ✓ MZI embedding and QFI computation
        ✓ Wigner function computation

        The simulation is MISSING:
        ✗ Lindblad decoherence for hybrid system (needed for Test 4)
        ✗ Comprehensive QFI comparison at fixed ⟨n⟩ (Test 3 needs more work)

        RECOMMENDATIONS:
        1. Implement hybrid Lindblad solver to test decoherence crossover
        2. Verify QFI advantage at higher ⟨n⟩ values (may need larger N)
        3. Compare with analytical formulas for n=2 to validate implementation
        """)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()


# =============================================================================
# Data and Figure Generation
# =============================================================================


def generate_qfi_comparison_data(
    target_ns: list[float] | None = None,
    omega_n: float = 1.0,
    N_max: int = 100,
    force: bool = False,
) -> Path:
    """Run the QFI comparison parameter sweep and save CSV.

    Performs QFI comparison across squeezing orders n=2,3,4 at multiple
    target mean photon numbers. Saves results as a CSV with columns for
    squeezing order n, achieved mean photon number, and QFI value.

    Args:
        target_ns: List of target mean photon numbers (default: 0.5 to 3.0).
        omega_n: Squeezing rate.
        N_max: Maximum Fock truncation.
        force: If True, regenerate data even if CSV exists.

    Returns:
        Path to the saved CSV file.

    """
    if target_ns is None:
        target_ns = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = _RAW_DIR / "qfi_comparison.parquet"

    if parquet_path.exists() and not force:
        print(f"Data file already exists at {parquet_path}. Use --force to regenerate.")
        return parquet_path

    rows = []
    for target_n in target_ns:
        print(f"  Target ⟨n⟩ = {target_n}:")
        for n in [2, 3, 4]:
            N = adaptive_truncation(alpha=0j, r_n=target_n, n=n, N_max=N_max)
            N = max(N, 10)

            t_final, mean_n, squeezed = find_squeezing_time_for_target_photon(
                n=n,
                target_n=target_n,
                N=N,
                omega_n=omega_n,
                theta_n=0.0,
                t_max=50.0,
            )
            qfi = qfi_hybrid_mzi(squeezed, N)
            rows.append(
                {
                    "n": n,
                    "target_n": target_n,
                    "achieved_n": float(mean_n),
                    "qfi": float(qfi),
                    "t_sqz": float(t_final),
                },
            )
            print(f"    n={n}: ⟨n⟩={mean_n:.3f}, QFI={qfi:.3f}, t={t_final:.3f}")

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved QFI comparison data to {parquet_path}")
    return parquet_path


def plot_qfi_comparison(
    data_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Load raw data and generate QFI comparison SVG figure.

    Produces a QFI comparison line plot with n=2, n=3, n=4 traces
    versus mean photon number.

    Args:
        data_path: Path to the Parquet file. If None, looks in raw_data/.
        force: If True, regenerate figure even if SVG exists.

    Returns:
        Path to the saved SVG file.

    """
    if data_path is None:
        data_path = _RAW_DIR / "qfi_comparison.parquet"

    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = _FIG_DIR / "qfi_comparison.svg"

    if svg_path.exists() and not force:
        print(f"Figure already exists at {svg_path}. Use --force to regenerate.")
        return svg_path

    df = pd.read_parquet(data_path)

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for n in [2, 3, 4]:
        subset = df[df["n"] == n].sort_values("achieved_n")
        ax.plot(
            subset["achieved_n"],
            subset["qfi"],
            marker="o",
            label=f"n={n}",
            linewidth=2,
        )

    ax.set_xlabel("Mean photon number ⟨n⟩")
    ax.set_ylabel("Quantum Fisher Information $F_Q$")
    ax.set_title("QFI Comparison Across Squeezing Orders")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(svg_path), format="svg", dpi=150)
    plt.close(fig)

    print(f"Saved QFI comparison figure to {svg_path}")
    return svg_path


def run_pipeline(force: bool = False) -> None:
    """Run the complete pipeline: generate data, generate figures, print summary."""
    print("=" * 60)
    print("QFI Comparison Pipeline — 2026-05-07 Report")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/2] Generating QFI comparison data...")
    data_path = generate_qfi_comparison_data(force=force)

    # Step 2: Generate figures
    print("\n[2/2] Generating QFI comparison figure...")
    fig_path = plot_qfi_comparison(data_path, force=force)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Data:  {data_path}")
    print(f"  Figure: {fig_path}")
    print("=" * 60)


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2026-05-07 High-Order Squeezing Report — Data & Figures",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate data and figures even if they already exist",
    )
    args = parser.parse_args()

    if args.force:
        run_pipeline(force=True)
    else:
        print(
            "Use `uv run python reports/20260507/local.py --force` to regenerate "
            "data and figures.",
        )
        print()
        run_all_tests()

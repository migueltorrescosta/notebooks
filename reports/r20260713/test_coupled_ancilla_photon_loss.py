"""Tests for Coupled System-Ancilla Metrology Under Photon Loss.

Companion test module for
``reports/r20260713/coupled_ancilla_photon_loss.py``.

Key test areas:
- Operator construction (dimensions, Hermiticity, eigenvalues, commutation)
- Bipartite operator properties (Kronecker structure, interaction Hamiltonians)
- Beam-splitter on system only (unitarity, identity on ancilla)
- Lindblad evolution invariants (trace, Hermiticity, positivity)
- Config A baseline recovery (SQL at γ=0)
- Config C: QFI additivity at α=0
- QFI-EP inequality
- QFI finite-difference correctness
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from src.utils.serialization import assert_roundtrip_fields

if TYPE_CHECKING:
    from pathlib import Path

_m = importlib.import_module("reports.r20260713.coupled_ancilla_photon_loss")


# Short aliases for the module's public API
build_subsystem_operators = _m.build_subsystem_operators
build_bipartite_operators = _m.build_bipartite_operators
build_coupling_hamiltonian = _m.build_coupling_hamiltonian
build_full_hamiltonian = _m.build_full_hamiltonian
build_bipartite_lindblad_ops = _m.build_bipartite_lindblad_ops
bs_system_only = _m.bs_system_only
initial_state_bipartite = _m.initial_state_bipartite
evolve_config_a = _m.evolve_config_a
evolve_config_c = _m.evolve_config_c
_trace_out_ancilla = _m._trace_out_ancilla
compute_qfi_finite_diff = _m.compute_qfi_finite_diff
compute_ep_sensitivity_from_rho = _m.compute_ep_sensitivity_from_rho
evaluate_sensitivity_at_omega = _m.evaluate_sensitivity_at_omega
SensitivityPoint = _m.SensitivityPoint
optimise_coupling = _m.optimise_coupling


# ============================================================================
# Fixtures
# ============================================================================

_make_alpha_zero = lambda: {"xx": 0.0, "xz": 0.0, "zx": 0.0, "zz": 0.0}  # noqa: E731
_make_alpha_nonzero = lambda: {"xx": 1.0, "xz": 0.5, "zx": 0.3, "zz": 0.7}  # noqa: E731


# ============================================================================
# Subsystem Operator Tests
# ============================================================================


class TestSubsystemOperators:
    """Verify Jz, Jx, a0, a1 in two-mode Fock space."""

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_dimensions(self, N: int) -> None:
        ops = build_subsystem_operators(N)
        dim = (N + 1) ** 2
        for key in ("Jz", "Jx", "a0", "a1", "a0_dag", "a1_dag"):
            assert ops[key].shape == (dim, dim), f"{key} wrong shape"

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_Jz_hermiticity(self, N: int) -> None:
        ops = build_subsystem_operators(N)
        assert np.allclose(ops["Jz"], ops["Jz"].conj().T)

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_Jx_hermiticity(self, N: int) -> None:
        ops = build_subsystem_operators(N)
        assert np.allclose(ops["Jx"], ops["Jx"].conj().T)

    def test_Jz_eigenvalues(self) -> None:
        """Jz eigenvalues in two-mode Fock space are (n0-n1)/2."""
        N = 2
        ops = build_subsystem_operators(N)
        eigvals = sorted(np.linalg.eigvalsh(ops["Jz"]).tolist())
        expected = sorted([(n0 - n1) / 2 for n0 in range(N + 1) for n1 in range(N + 1)])
        assert np.allclose(eigvals, expected)

    def test_commutation_relation(self) -> None:
        """[Jz, Jx] = i Jy (where Jy = (a0†a1 - a1†a0)/(2i))."""
        N = 2
        ops = build_subsystem_operators(N)
        Jy = (ops["a0_dag"] @ ops["a1"] - ops["a1_dag"] @ ops["a0"]) / (2j)
        comm = ops["Jz"] @ ops["Jx"] - ops["Jx"] @ ops["Jz"]
        assert np.allclose(comm, 1j * Jy, atol=1e-10)


# ============================================================================
# Bipartite Operator Tests
# ============================================================================


class TestBipartiteOperators:
    """Verify operators in the full (N+1)⁴ space."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_dimensions(self, N: int) -> None:
        ops = build_bipartite_operators(N)
        dim = (N + 1) ** 4
        assert ops["Jz_S"].shape == (dim, dim)
        assert ops["Jz_A"].shape == (dim, dim)
        assert ops["Jx_S"].shape == (dim, dim)
        assert ops["Jx_A"].shape == (dim, dim)

    def test_Jz_commutation(self) -> None:
        """[Jz_S, Jz_A] = 0 — subsystems are independent."""
        N = 2
        ops = build_bipartite_operators(N)
        comm = ops["Jz_S"] @ ops["Jz_A"] - ops["Jz_A"] @ ops["Jz_S"]
        assert np.allclose(comm, 0, atol=1e-12)

    def test_Jx_commutation(self) -> None:
        """[Jx_S, Jx_A] = 0 — subsystems are independent."""
        N = 1
        ops = build_bipartite_operators(N)
        comm = ops["Jx_S"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jx_S"]
        assert np.allclose(comm, 0, atol=1e-12)

    def test_cross_commutation(self) -> None:
        """[Jz_S, Jx_A] = 0 — different operators on different subsystems."""
        N = 1
        ops = build_bipartite_operators(N)
        comm = ops["Jz_S"] @ ops["Jx_A"] - ops["Jx_A"] @ ops["Jz_S"]
        assert np.allclose(comm, 0, atol=1e-12)

    @pytest.mark.parametrize("N", [1, 2])
    def test_coupling_hamiltonian_hermiticity(self, N: int) -> None:
        ops = build_bipartite_operators(N)
        H_int = build_coupling_hamiltonian(_make_alpha_nonzero(), ops)
        assert np.allclose(H_int, H_int.conj().T)

    @pytest.mark.parametrize("N", [1, 2])
    def test_full_hamiltonian_hermiticity(self, N: int) -> None:
        ops = build_bipartite_operators(N)
        H = build_full_hamiltonian(1.5, _make_alpha_nonzero(), ops)
        assert np.allclose(H, H.conj().T)

    def test_coupling_hamiltonian_linearity(self) -> None:
        """H_int(2α) = 2 H_int(α) — linearity in coupling coefficients."""
        N = 1
        ops = build_bipartite_operators(N)
        alpha1 = _make_alpha_nonzero()
        alpha2 = {k: 2.0 * v for k, v in alpha1.items()}
        H1 = build_coupling_hamiltonian(alpha1, ops)
        H2 = build_coupling_hamiltonian(alpha2, ops)
        assert np.allclose(H2, 2.0 * H1)


# ============================================================================
# Beam Splitter Tests
# ============================================================================


class TestBeamSplitter:
    """Verify BS on system only."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_unitarity(self, N: int) -> None:
        U = bs_system_only(N)
        dim = (N + 1) ** 4
        assert np.allclose(U @ U.conj().T, np.eye(dim), atol=1e-10)

    def test_identity_on_ancilla(self) -> None:
        """BS on system should leave Jz_A unchanged."""
        N = 1
        U = bs_system_only(N)
        ops = build_bipartite_operators(N)
        Jz_A_rot = U @ ops["Jz_A"] @ U.conj().T
        assert np.allclose(Jz_A_rot, ops["Jz_A"], atol=1e-10)

    def test_Jz_S_rotated(self) -> None:
        """BS on system rotates Jz_S (non-trivial transformation)."""
        N = 1
        U = bs_system_only(N)
        ops = build_bipartite_operators(N)
        Jz_S_rot = U @ ops["Jz_S"] @ U.conj().T
        # After 50/50 BS, Jz should rotate to something different
        assert not np.allclose(Jz_S_rot, ops["Jz_S"], atol=1e-6)


# ============================================================================
# Initial State Tests
# ============================================================================


class TestInitialState:
    def test_normalization(self) -> None:
        N = 2
        psi = initial_state_bipartite(N)
        assert np.isclose(np.linalg.norm(psi), 1.0)

    def test_is_product_state(self) -> None:
        """Initial state should be |N,0⟩_S ⊗ |N,0⟩_A."""
        N = 2
        psi = initial_state_bipartite(N)
        dim_sub = (N + 1) ** 2
        idx_N0 = N * (N + 1)  # index of |N,0⟩
        idx_full = idx_N0 * dim_sub + idx_N0
        assert np.isclose(abs(psi[idx_full]) ** 2, 1.0)

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_dimension(self, N: int) -> None:
        psi = initial_state_bipartite(N)
        assert psi.shape == ((N + 1) ** 4,)


# ============================================================================
# Lindblad Operator Tests
# ============================================================================


class TestLindbladOperators:
    def test_count(self) -> None:
        N = 1
        ops = build_bipartite_operators(N)
        L_ops = build_bipartite_lindblad_ops(0.1, ops)
        assert len(L_ops) == 4

    def test_dimensions(self) -> None:
        N = 2
        ops = build_bipartite_operators(N)
        L_ops = build_bipartite_lindblad_ops(0.1, ops)
        dim = (N + 1) ** 4
        for L in L_ops:
            assert L.shape == (dim, dim)

    def test_zero_rate(self) -> None:
        N = 1
        ops = build_bipartite_operators(N)
        L_ops = build_bipartite_lindblad_ops(0.0, ops)
        assert len(L_ops) == 0

    def test_tracelessness(self) -> None:
        """Annihilation operators are traceless."""
        N = 2
        ops = build_bipartite_operators(N)
        L_ops = build_bipartite_lindblad_ops(1.0, ops)
        for L in L_ops:
            assert np.isclose(np.trace(L), 0.0, atol=1e-12)


# ============================================================================
# Evolution Invariant Tests
# ============================================================================


class TestEvolutionInvariants:
    """Trace preservation, Hermiticity, positivity after Lindblad evolution."""

    @pytest.mark.parametrize("gamma", [0.0, 0.01, 0.1])
    def test_config_a_trace(self, gamma: float) -> None:
        rho = evolve_config_a(N=1, omega=1.0, gamma=gamma, t_hold=1.0)
        assert np.isclose(np.trace(rho), 1.0, atol=1e-6)

    @pytest.mark.parametrize("gamma", [0.0, 0.01, 0.1])
    def test_config_a_hermiticity(self, gamma: float) -> None:
        rho = evolve_config_a(N=1, omega=1.0, gamma=gamma, t_hold=1.0)
        assert np.allclose(rho, rho.conj().T, atol=1e-8)

    @pytest.mark.parametrize("gamma", [0.0, 0.01, 0.1])
    def test_config_a_positivity(self, gamma: float) -> None:
        rho = evolve_config_a(N=1, omega=1.0, gamma=gamma, t_hold=1.0)
        eigvals = np.linalg.eigvalsh(rho)
        assert np.min(eigvals) >= -1e-4

    @pytest.mark.parametrize("gamma", [0.0, 0.05])
    def test_config_c_trace(self, gamma: float) -> None:
        alpha = _make_alpha_nonzero()
        rho = evolve_config_c(N=1, omega=1.0, gamma=gamma, t_hold=1.0, alpha=alpha)
        assert np.isclose(np.trace(rho), 1.0, atol=1e-6)

    @pytest.mark.parametrize("gamma", [0.0, 0.05])
    def test_config_c_hermiticity(self, gamma: float) -> None:
        alpha = _make_alpha_nonzero()
        rho = evolve_config_c(N=1, omega=1.0, gamma=gamma, t_hold=1.0, alpha=alpha)
        assert np.allclose(rho, rho.conj().T, atol=1e-8)

    @pytest.mark.parametrize("gamma", [0.0, 0.05])
    def test_config_c_positivity(self, gamma: float) -> None:
        alpha = _make_alpha_nonzero()
        rho = evolve_config_c(N=1, omega=1.0, gamma=gamma, t_hold=1.0, alpha=alpha)
        eigvals = np.linalg.eigvalsh(rho)
        assert np.min(eigvals) >= -1e-4

    def test_partial_trace_preserves_trace(self) -> None:
        N = 1
        alpha = _make_alpha_nonzero()
        rho = evolve_config_c(N=N, omega=1.0, gamma=0.05, t_hold=1.0, alpha=alpha)
        ops = build_bipartite_operators(N)
        rho_S = _trace_out_ancilla(rho, ops["dim_sub"])
        assert np.isclose(np.trace(rho_S), 1.0, atol=1e-6)
        assert np.allclose(rho_S, rho_S.conj().T, atol=1e-8)


# ============================================================================
# Baseline Recovery Tests
# ============================================================================


class TestBaselineRecovery:
    """Verify known analytical results at special parameter values."""

    def test_config_a_noiseless_sql(self) -> None:
        """Config A at γ=0, N=1 should give Δω = 1/T_H."""
        N = 1
        t_hold = 10.0
        omega = 1.0
        fd = 1e-6

        rho = evolve_config_a(N, omega, gamma=0.0, t_hold=t_hold)
        rho_p = evolve_config_a(N, omega + fd, gamma=0.0, t_hold=t_hold)
        rho_m = evolve_config_a(N, omega - fd, gamma=0.0, t_hold=t_hold)

        sub = build_subsystem_operators(N)
        Jz = sub["Jz"]

        delta_ep, _, _ = compute_ep_sensitivity_from_rho(rho, Jz, rho_p, rho_m, fd)
        expected_sql = 1.0 / (np.sqrt(N) * t_hold)
        assert np.isclose(delta_ep, expected_sql, rtol=0.01), (
            f"Δω_EP={delta_ep:.6f}, SQL={expected_sql:.6f}"
        )

    def test_config_a_noiseless_qfi(self) -> None:
        """Config A at γ=0, N=1: F_Q(ω) = N·T² = T² (SQL)."""
        N = 1
        t_hold = 10.0
        omega = 1.0
        fd = 1e-6
        rho = evolve_config_a(N, omega, gamma=0.0, t_hold=t_hold)
        rho_p = evolve_config_a(N, omega + fd, gamma=0.0, t_hold=t_hold)
        rho_m = evolve_config_a(N, omega - fd, gamma=0.0, t_hold=t_hold)
        fq = compute_qfi_finite_diff(rho, rho_p, rho_m, fd)
        # F_Q(ω) = N·T² for noiseless SQL state
        expected_fq = N * t_hold**2
        assert np.isclose(fq, expected_fq, rtol=0.05), (
            f"F_Q={fq:.6f}, expected N·T²={expected_fq:.6f}"
        )

    def test_config_a_noiseless_n2(self) -> None:
        """Config A at γ=0, N=2: Δω = 1/(√2 T_H)."""
        N = 2
        t_hold = 10.0
        omega = 1.0
        fd = 1e-6

        rho = evolve_config_a(N, omega, gamma=0.0, t_hold=t_hold)
        rho_p = evolve_config_a(N, omega + fd, gamma=0.0, t_hold=t_hold)
        rho_m = evolve_config_a(N, omega - fd, gamma=0.0, t_hold=t_hold)

        sub = build_subsystem_operators(N)
        Jz = sub["Jz"]

        delta_ep, _, _ = compute_ep_sensitivity_from_rho(rho, Jz, rho_p, rho_m, fd)
        expected = 1.0 / (np.sqrt(N) * t_hold)
        assert np.isclose(delta_ep, expected, rtol=0.05), (
            f"Δω_EP={delta_ep:.6f}, expected {expected:.6f}"
        )


# ============================================================================
# QFI Finite-Difference Tests
# ============================================================================


class TestQFIFiniteDiff:
    """Verify QFI computation via finite differences."""

    def test_qfi_positivity(self) -> None:
        N = 1
        omega = 1.0
        t_hold = 10.0
        fd = 1e-6
        rho = evolve_config_a(N, omega, gamma=0.0, t_hold=t_hold)
        rho_p = evolve_config_a(N, omega + fd, gamma=0.0, t_hold=t_hold)
        rho_m = evolve_config_a(N, omega - fd, gamma=0.0, t_hold=t_hold)
        fq = compute_qfi_finite_diff(rho, rho_p, rho_m, fd)
        assert fq >= 0, f"F_Q must be non-negative, got {fq}"

    def test_qfi_pure_state_match(self) -> None:
        """For noiseless evolution, finite-diff QFI should match analytical F_Q(ω)=N·T²."""
        N = 1
        omega = 1.0
        t_hold = 10.0
        fd = 1e-6

        rho = evolve_config_a(N, omega, gamma=0.0, t_hold=t_hold)
        rho_p = evolve_config_a(N, omega + fd, gamma=0.0, t_hold=t_hold)
        rho_m = evolve_config_a(N, omega - fd, gamma=0.0, t_hold=t_hold)

        fq_fd = compute_qfi_finite_diff(rho, rho_p, rho_m, fd)
        # Analytical: F_Q(ω) = N·T² for noiseless MZI with SQL input
        fq_expected = N * t_hold**2
        assert np.isclose(fq_fd, fq_expected, rtol=0.02), (
            f"F_Q(fd)={fq_fd:.6f}, expected N·T²={fq_expected:.6f}"
        )

    def test_qfi_with_noise(self) -> None:
        """QFI should be positive even with noise."""
        N = 1
        omega = 1.0
        t_hold = 10.0
        gamma = 0.05
        fd = 1e-6

        rho = evolve_config_a(N, omega, gamma, t_hold)
        rho_p = evolve_config_a(N, omega + fd, gamma, t_hold)
        rho_m = evolve_config_a(N, omega - fd, gamma, t_hold)

        fq = compute_qfi_finite_diff(rho, rho_p, rho_m, fd)
        assert fq >= 0


# ============================================================================
# QFI-EP Inequality Tests
# ============================================================================


class TestQFIEPInequality:
    """Δω_QFI ≤ Δω_EP always (QFI is the optimal bound)."""

    @pytest.mark.slow
    def test_config_a_inequality(self) -> None:
        N = 2
        omega = 1.0
        t_hold = 10.0
        gamma = 0.05
        fd = 1e-6

        rho = evolve_config_a(N, omega, gamma, t_hold)
        rho_p = evolve_config_a(N, omega + fd, gamma, t_hold)
        rho_m = evolve_config_a(N, omega - fd, gamma, t_hold)

        sub = build_subsystem_operators(N)
        Jz = sub["Jz"]

        fq = compute_qfi_finite_diff(rho, rho_p, rho_m, fd)
        delta_qfi = 1.0 / np.sqrt(fq) if fq > 0 else float("inf")

        delta_ep, _, _ = compute_ep_sensitivity_from_rho(rho, Jz, rho_p, rho_m, fd)

        if np.isfinite(delta_qfi) and np.isfinite(delta_ep):
            assert delta_qfi <= delta_ep * 1.02, (
                f"Δω_QFI={delta_qfi:.6f} > Δω_EP={delta_ep:.6f}"
            )

    @pytest.mark.slow
    def test_config_c_inequality(self) -> None:
        N = 1
        omega = 1.0
        t_hold = 10.0
        gamma = 0.05
        fd = 1e-6
        alpha = _make_alpha_nonzero()

        rho = evolve_config_c(N, omega, gamma, t_hold, alpha)
        rho_p = evolve_config_c(N, omega + fd, gamma, t_hold, alpha)
        rho_m = evolve_config_c(N, omega - fd, gamma, t_hold, alpha)

        fq = compute_qfi_finite_diff(rho, rho_p, rho_m, fd)
        delta_qfi = 1.0 / np.sqrt(fq) if fq > 0 else float("inf")

        ops_b = build_bipartite_operators(N)
        sub = build_subsystem_operators(N)
        rho_S = _trace_out_ancilla(rho, ops_b["dim_sub"])
        rho_S_p = _trace_out_ancilla(rho_p, ops_b["dim_sub"])
        rho_S_m = _trace_out_ancilla(rho_m, ops_b["dim_sub"])

        delta_ep, _, _ = compute_ep_sensitivity_from_rho(
            rho_S, sub["Jz"], rho_S_p, rho_S_m, fd
        )

        if np.isfinite(delta_qfi) and np.isfinite(delta_ep):
            assert delta_qfi <= delta_ep * 1.05, (
                f"Δω_QFI={delta_qfi:.6f} > Δω_EP={delta_ep:.6f}"
            )


# ============================================================================
# QFI Additivity at α=0
# ============================================================================


class TestQFIAdditivity:
    """At α=0, the ancilla carries no phase info (eigenstate of Jz_A, no BS).

    Config C at α=0 should give F_Q(C) ≈ F_Q(A) because the ancilla
    contributes zero QFI — its initial state |N,0⟩_A is an eigenstate of Jz_A
    and it receives no beam-splitter. Only with non-zero coupling (α≠0) can
    entanglement transfer ancilla phase information to the measured subsystem.
    """

    def test_config_c_alpha_zero_equals_config_a(self) -> None:
        N = 1
        omega = 1.0
        t_hold = 10.0
        gamma = 0.0
        fd = 1e-6

        # Config A: F_Q(ω) via finite differences
        rho_a = evolve_config_a(N, omega, gamma, t_hold)
        rho_a_p = evolve_config_a(N, omega + fd, gamma, t_hold)
        rho_a_m = evolve_config_a(N, omega - fd, gamma, t_hold)
        fq_a = compute_qfi_finite_diff(rho_a, rho_a_p, rho_a_m, fd)

        # Config C at α=0: F_Q(ω) via finite differences
        alpha_zero = _make_alpha_zero()
        rho_c = evolve_config_c(N, omega, gamma, t_hold, alpha_zero)
        rho_c_p = evolve_config_c(N, omega + fd, gamma, t_hold, alpha_zero)
        rho_c_m = evolve_config_c(N, omega - fd, gamma, t_hold, alpha_zero)
        fq_c = compute_qfi_finite_diff(rho_c, rho_c_p, rho_c_m, fd)

        # At α=0, ancilla |N,0⟩_A is eigenstate of Jz_A with no BS,
        # so F_Q(C) ≈ F_Q(A) — ancilla contributes nothing
        if fq_a > 0:
            ratio = fq_c / fq_a
            assert np.isclose(ratio, 1.0, rtol=0.05), (
                f"F_Q(C)={fq_c:.6f}, F_Q(A)={fq_a:.6f}, ratio={ratio:.4f}"
            )


# ============================================================================
# Coupling Optimisation Tests
# ============================================================================


class TestCouplingOptimisation:
    """Verify optimise_coupling finds a valid minimum."""

    def test_returns_valid_structure(self) -> None:
        """optimise_coupling returns dict with correct keys and finite values."""
        result = optimise_coupling(
            N=1,
            gamma=0.0,
            omega_rep=1.0,
            t_hold=10.0,
            n_starts=2,
            max_iter=5,
            seed=42,
        )
        assert "alpha_opt" in result
        assert "delta_ep_opt" in result
        assert "n_evals" in result
        assert all(k in result["alpha_opt"] for k in ("xx", "xz", "zx", "zz"))
        assert np.isfinite(result["delta_ep_opt"])
        assert result["delta_ep_opt"] > 0
        assert result["n_evals"] > 0

    @pytest.mark.slow
    def test_optimised_better_than_zero_coupling(self) -> None:
        """At γ=0, non-zero transverse coupling improves EP sensitivity."""
        N = 1
        omega_rep = 1.0
        t_hold = 10.0

        # Zero-coupling baseline
        baseline = evaluate_sensitivity_at_omega(
            N, omega_rep, gamma=0.0, t_hold=t_hold, config="C"
        )

        # Optimised coupling
        result = optimise_coupling(
            N=N,
            gamma=0.0,
            omega_rep=omega_rep,
            t_hold=t_hold,
            n_starts=2,
            max_iter=10,
            seed=42,
        )

        assert result["delta_ep_opt"] < baseline.delta_omega_ep, (
            f"Optimised {result['delta_ep_opt']:.6f} not better "
            f"than baseline {baseline.delta_omega_ep:.6f}"
        )


# ============================================================================
# Sensitivity Point Tests
# ============================================================================


class TestSensitivityPoint:
    """Verify evaluate_sensitivity_at_omega returns valid results."""

    def test_config_a_returns_point(self) -> None:
        pt = evaluate_sensitivity_at_omega(
            N=1, omega=1.0, gamma=0.0, t_hold=10.0, config="A"
        )
        assert isinstance(pt, SensitivityPoint)
        assert pt.config == "A"
        assert pt.N == 1
        assert np.isfinite(pt.delta_omega_ep)
        assert pt.fq >= 0

    @pytest.mark.slow
    def test_config_c_returns_point(self) -> None:
        alpha = _make_alpha_nonzero()
        pt = evaluate_sensitivity_at_omega(
            N=1, omega=1.0, gamma=0.05, t_hold=10.0, alpha=alpha, config="C"
        )
        assert isinstance(pt, SensitivityPoint)
        assert pt.config == "C"
        assert pt.N == 1
        assert np.isfinite(pt.delta_omega_ep)
        assert pt.fq >= 0

    @pytest.mark.slow
    def test_config_c_measurement_gap(self) -> None:
        """Δω_EP ≥ Δω_QFI for Config C (measurement gap)."""
        alpha = _make_alpha_nonzero()
        pt = evaluate_sensitivity_at_omega(
            N=1, omega=1.0, gamma=0.05, t_hold=10.0, alpha=alpha, config="C"
        )
        if np.isfinite(pt.delta_omega_qfi):
            assert pt.delta_omega_ep >= pt.delta_omega_qfi * 0.95


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSensitivityPointSerialization:
    """Verify Parquet roundtrip for SensitivityPoint."""

    def test_roundtrip_all_metadata(self, tmp_path: Path) -> None:
        """All input parameters and results survive a Parquet roundtrip."""
        alpha = _make_alpha_nonzero()
        pt = SensitivityPoint(
            omega=1.5,
            gamma=0.05,
            N=2,
            delta_omega_ep=0.083,
            fq=144.0,
            delta_omega_qfi=0.0833,
            expectation=0.42,
            variance=0.18,
            derivative=2.1,
            config="C",
            alpha=alpha,
            t_hold=10.0,
            fd_step=1e-6,
        )

        parquet_path = tmp_path / "test_point.parquet"
        pt.save_parquet(parquet_path)

        loaded = SensitivityPoint.from_parquet(parquet_path)
        assert_roundtrip_fields(
            loaded,
            pt,
            [
                ("omega", "isclose"),
                ("gamma", "isclose"),
                ("N", "eq"),
                ("delta_omega_ep", "isclose"),
                ("fq", "isclose"),
                ("delta_omega_qfi", "isclose"),
                ("expectation", "isclose"),
                ("variance", "isclose"),
                ("derivative", "isclose"),
                ("config", "eq"),
                ("t_hold", "isclose"),
                ("fd_step", "isclose"),
            ],
        )
        assert loaded.alpha["xx"] == pytest.approx(alpha["xx"])
        assert loaded.alpha["xz"] == pytest.approx(alpha["xz"])
        assert loaded.alpha["zx"] == pytest.approx(alpha["zx"])
        assert loaded.alpha["zz"] == pytest.approx(alpha["zz"])

    def test_roundtrip_config_a(self, tmp_path: Path) -> None:
        """Config A roundtrip preserves all fields."""
        pt = SensitivityPoint(
            omega=1.0,
            gamma=0.0,
            N=3,
            delta_omega_ep=0.0577,
            fq=300.0,
            delta_omega_qfi=0.0577,
            expectation=0.0,
            variance=0.5,
            derivative=8.66,
            config="A",
            t_hold=10.0,
            fd_step=1e-6,
        )

        parquet_path = tmp_path / "test_a.parquet"
        pt.save_parquet(parquet_path)
        loaded = SensitivityPoint.from_parquet(parquet_path)

        assert loaded.config == "A"
        assert loaded.N == 3
        assert loaded.alpha == {"xx": 0.0, "xz": 0.0, "zx": 0.0, "zz": 0.0}
        assert loaded.t_hold == 10.0
        assert loaded.fd_step == pytest.approx(1e-6)

    def test_fail_fast_missing_columns(self, tmp_path: Path) -> None:
        """from_parquet raises ValueError when required columns are absent."""
        df = pd.DataFrame({"omega": [1.0], "gamma": [0.0]})
        bad_path = tmp_path / "bad.parquet"
        df.to_parquet(bad_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            SensitivityPoint.from_parquet(bad_path)

    def test_to_dataframe_shape(self) -> None:
        """to_dataframe returns single-row DataFrame with expected columns."""
        pt = SensitivityPoint(
            omega=1.0,
            gamma=0.0,
            N=1,
            delta_omega_ep=0.1,
            fq=100.0,
            delta_omega_qfi=0.1,
            expectation=0.5,
            variance=0.25,
            derivative=1.0,
            config="A",
        )
        df = pt.to_dataframe()
        assert df.shape == (1, len(SensitivityPoint._PARQUET_COLUMNS))
        assert list(df.columns) == SensitivityPoint._PARQUET_COLUMNS


class TestSolverTuningRegression:
    """Ensure BDF + relaxed tolerances reproduce reference values."""

    @pytest.mark.parametrize(
        "N,gamma,expected_ep",
        [
            (1, 0.0, 0.1),       # SQL
            (1, 0.01, 0.11642),  # noise-degraded
            (2, 0.0, 0.07071),   # SQL N=2
        ],
        ids=["sql-N1", "noisy-N1", "sql-N2"],
    )
    def test_config_a_ep_within_tolerance(
        self, N: int, gamma: float, expected_ep: float
    ) -> None:
        pt = evaluate_sensitivity_at_omega(N, 1.0, gamma, 10.0, config="A")
        assert np.isclose(pt.delta_omega_ep, expected_ep, rtol=1e-3), (
            f"EP={pt.delta_omega_ep:.6f}, expected≈{expected_ep:.5f}"
        )

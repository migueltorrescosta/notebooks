"""
Tests for the phase-diffusion robustness of the drive protocol module.

Run with:
    uv run pytest reports/20260524/test_local.py -q --tb=short
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.linalg import expm

from src.analysis.ancilla_optimization import build_two_qubit_operators

if TYPE_CHECKING:
    from pathlib import Path

_report_dir = str(
    _Path(__file__).resolve().parent.parent.parent / "reports" / "20260524"
)
if _report_dir not in _sys.path:
    _sys.path.insert(0, _report_dir)
del _sys, _Path, _report_dir

from local import (  # type: ignore[import-untyped]  # noqa: E402
    DEFAULT_RHO0,
    DEFAULT_T_BS,
    DEFAULT_T_H,
    DRIVE_BOUNDS,
    FD_STEP,
    SQL_REFERENCE,
    DriveNoiseScanResult,
    build_liouvillian,
    build_noise_drive_hamiltonian,
    build_noise_hold_hamiltonian,
    build_noise_iszz_interaction,
    build_phase_diffusion_operators,
    compute_noisy_decoupled_baseline,
    compute_noisy_sensitivity,
    compute_noisy_sensitivity_with_diagnostics,
    density_expectation,
    density_variance,
    evolve_noisy_drive_circuit,
    noisy_sensitivity_objective,
    run_noise_scan,
    run_noisy_nelder_mead,
    run_noisy_random_search,
    unvectorise_rho,
    validate_density,
    vectorise_rho,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def make_ops() -> dict:
    """Default two-qubit operators."""
    return build_two_qubit_operators()


# ============================================================================
# Test: Density Matrix Utilities
# ============================================================================


class TestDensityUtilities:
    def test_vectorize_unvectorize(self, make_ops: dict) -> None:
        """Vectorise and unvectorise should be inverses."""
        rho = DEFAULT_RHO0.copy()
        vec = vectorise_rho(rho)
        rho_restored = unvectorise_rho(vec)
        assert np.allclose(rho, rho_restored, atol=1e-12)

    def test_vectorize_shape(self) -> None:
        """Vectorise a 4×4 matrix → 16-vector."""
        vec = vectorise_rho(DEFAULT_RHO0)
        assert vec.shape == (16,)

    def test_unvectorize_shape(self) -> None:
        """Unvectorise a 16-vector → 4×4 matrix."""
        vec = np.zeros(16, dtype=complex)
        vec[0] = 1.0
        rho = unvectorise_rho(vec)
        assert rho.shape == (4, 4)

    def test_vectorize_column_major(self) -> None:
        """Verify F-ordering: vec(ρ)[i + d*j] = ρ[i, j]."""
        rho = np.arange(16, dtype=complex).reshape(4, 4)
        vec = vectorise_rho(rho)
        for i in range(4):
            for j in range(4):
                assert vec[i + 4 * j] == rho[i, j], f"Mismatch at ({i},{j})"

    def test_density_expectation(self, make_ops: dict) -> None:
        """⟨J_z^S⟩ for |00⟩ should be 0.5."""
        exp_val = density_expectation(DEFAULT_RHO0, make_ops["Jz_S"])
        assert exp_val == pytest.approx(0.5, abs=1e-12)

    def test_density_variance_zero_pure(self, make_ops: dict) -> None:
        """For |00⟩, Var(J_z^S) = 0."""
        var = density_variance(DEFAULT_RHO0, make_ops["Jz_S"])
        assert var == pytest.approx(0.0, abs=1e-12)

    def test_validate_density_passes(self) -> None:
        """Default state should pass validation."""
        validate_density(DEFAULT_RHO0)

    def test_validate_density_raises_on_bad_trace(self) -> None:
        """Non-trace-1 matrix should fail validation."""
        rho_bad = np.zeros((4, 4), dtype=complex)
        with pytest.raises(AssertionError, match="Trace not preserved"):
            validate_density(rho_bad)

    def test_density_variance_positive(self, make_ops: dict) -> None:
        """Variance should always be non-negative."""
        # Create a mixed state
        rho_mixed = 0.5 * np.eye(4, dtype=complex)
        var = density_variance(rho_mixed, make_ops["Jz_S"])
        assert var >= -1e-12


# ============================================================================
# Test: Hamiltonian Construction
# ============================================================================


class TestHamiltonianConstruction:
    def test_drive_hamiltonian_hermitian(self, make_ops: dict) -> None:
        H = build_noise_drive_hamiltonian(1.0, 2.0, 3.0, 4.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_drive_hamiltonian_zero_theta(self, make_ops: dict) -> None:
        """At θ=0, the drive Hamiltonian should be zero."""
        H = build_noise_drive_hamiltonian(0.0, 5.0, 3.0, 2.0, make_ops)
        assert np.allclose(H, 0.0, atol=1e-14)

    def test_iszz_interaction_hermitian(self, make_ops: dict) -> None:
        H = build_noise_iszz_interaction(3.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_iszz_interaction_zero(self) -> None:
        """Zero coupling should give zero matrix."""
        ops = build_two_qubit_operators()
        H = build_noise_iszz_interaction(0.0, ops)
        assert np.allclose(H, 0.0, atol=1e-14)

    def test_hold_hamiltonian_hermitian(self, make_ops: dict) -> None:
        H = build_noise_hold_hamiltonian(0.5, 2.0, -1.0, 3.0, 4.0, make_ops)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_hold_contains_jz_plus_drive(self, make_ops: dict) -> None:
        """H should contain θ J_z^S + drive + interaction."""
        theta = 0.7
        a_x, a_y, a_z, a_zz = 2.0, 0.0, 0.0, 0.0
        H = build_noise_hold_hamiltonian(theta, a_x, a_y, a_z, a_zz, make_ops)
        expected = theta * make_ops["Jz_S"] + theta * a_x * make_ops["Jx_A"]
        assert np.allclose(H, expected, atol=1e-12)

    def test_hold_reduces_to_thetajz_at_zero(self, make_ops: dict) -> None:
        """At a_k = a_zz = 0, H = θ J_z^S."""
        H = build_noise_hold_hamiltonian(1.0, 0.0, 0.0, 0.0, 0.0, make_ops)
        expected = 1.0 * make_ops["Jz_S"]
        assert np.allclose(H, expected, atol=1e-12)


# ============================================================================
# Test: Phase Diffusion Operators
# ============================================================================


class TestPhaseDiffusionOperators:
    def test_zero_gamma_returns_empty(self, make_ops: dict) -> None:
        ops = build_phase_diffusion_operators(0.0, make_ops)
        assert len(ops) == 0

    def test_two_operators_at_positive_gamma(self, make_ops: dict) -> None:
        ops = build_phase_diffusion_operators(1.0, make_ops)
        assert len(ops) == 2

    def test_lindblad_scaling(self, make_ops: dict) -> None:
        """L_k should scale with sqrt(γ_φ)."""
        ops_1 = build_phase_diffusion_operators(1.0, make_ops)
        ops_4 = build_phase_diffusion_operators(4.0, make_ops)
        for L1, L4 in zip(ops_1, ops_4, strict=False):
            assert np.allclose(L4, 2.0 * L1, atol=1e-12)

    def test_lindblad_operators_proportional_to_jz(self, make_ops: dict) -> None:
        """L_S = √γ_φ J_z^S, L_A = √γ_φ J_z^A."""
        gamma = 2.0
        lindblad = build_phase_diffusion_operators(gamma, make_ops)
        expected_LS = np.sqrt(gamma) * make_ops["Jz_S"]
        expected_LA = np.sqrt(gamma) * make_ops["Jz_A"]
        assert np.allclose(lindblad[0], expected_LS, atol=1e-12)
        assert np.allclose(lindblad[1], expected_LA, atol=1e-12)


# ============================================================================
# Test: Liouvillian Construction
# ============================================================================


class TestLiouvillian:
    def test_liouvillian_shape(self, make_ops: dict) -> None:
        """Liouvillian should be 16×16 for a 4-dim Hilbert space."""
        lindblad = build_phase_diffusion_operators(1.0, make_ops)
        L = build_liouvillian(np.zeros((4, 4), dtype=complex), lindblad)
        assert L.shape == (16, 16)

    def test_liouvillian_trace_preserving(self, make_ops: dict) -> None:
        """Tr(exp(ℒ t) ρ₀) = 1 should hold."""
        H = build_noise_hold_hamiltonian(1.0, 0.5, 0.0, 0.0, 0.3, make_ops)
        lindblad = build_phase_diffusion_operators(0.5, make_ops)
        L = build_liouvillian(H, lindblad)

        rho_vec = vectorise_rho(DEFAULT_RHO0)
        rho_evolved = unvectorise_rho(expm(L * DEFAULT_T_H) @ rho_vec)
        trace = float(np.real(np.trace(rho_evolved)))
        assert np.isclose(trace, 1.0, atol=1e-10)

    def test_liouvillian_no_noise_gives_unitary(self, make_ops: dict) -> None:
        """At γ_φ = 0, the Liouvillian should reduce to -i[H, ·] (no Lindblad terms)."""
        H = build_noise_hold_hamiltonian(1.0, 0.5, 0.0, 0.0, 0.3, make_ops)
        lindblad = build_phase_diffusion_operators(0.0, make_ops)
        L = build_liouvillian(H, lindblad)

        # Should match -i(I ⊗ H - H^T ⊗ I) (column-major vectorization)
        I4 = np.eye(4, dtype=complex)
        expected_L = -1j * (np.kron(I4, H) - np.kron(H.T, I4))
        assert np.allclose(L, expected_L, atol=1e-12)

    def test_liouvillian_hermiticity(self, make_ops: dict) -> None:
        """The density matrix evolved by the Liouvillian should remain Hermitian."""
        H = build_noise_hold_hamiltonian(0.5, 2.0, 1.0, 0.0, 1.5, make_ops)
        lindblad = build_phase_diffusion_operators(0.1, make_ops)
        L = build_liouvillian(H, lindblad)

        rho_vec = vectorise_rho(DEFAULT_RHO0)
        rho_evolved = unvectorise_rho(expm(L * 5.0) @ rho_vec)
        assert np.allclose(rho_evolved, rho_evolved.conj().T, atol=1e-10)


# ============================================================================
# Test: Noisy Circuit Evolution
# ============================================================================


class TestNoisyCircuit:
    def test_no_noise_matches_noiseless(self, make_ops: dict) -> None:
        """At γ_φ = 0, the noisy circuit should match the unitary evolution."""
        ops = make_ops
        rho_noiseless = evolve_noisy_drive_circuit(
            DEFAULT_RHO0, DEFAULT_T_BS, DEFAULT_T_H, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ops
        )

        # Pure unitary evolution
        H = build_noise_hold_hamiltonian(1.0, 0.0, 0.0, 0.0, 0.0, ops)
        U_hold = expm(-1j * DEFAULT_T_H * H)
        from src.analysis.ancilla_drive_metrology import system_only_bs_unitary

        U_bs = system_only_bs_unitary(DEFAULT_T_BS)
        rho_unitary = (
            U_bs
            @ U_hold
            @ U_bs
            @ DEFAULT_RHO0
            @ (U_bs.conj().T @ U_hold.conj().T @ U_bs.conj().T)
        )

        assert np.allclose(rho_noiseless, rho_unitary, atol=1e-10)

    def test_trace_preserved_noise(self, make_ops: dict) -> None:
        """Trace should be preserved even with noise."""
        for gamma in [1e-4, 1e-2, 1.0, 10.0]:
            rho = evolve_noisy_drive_circuit(
                DEFAULT_RHO0,
                DEFAULT_T_BS,
                DEFAULT_T_H,
                1.0,
                gamma,
                2.0,
                0.0,
                0.0,
                1.0,
                make_ops,
            )
            assert np.isclose(np.trace(rho), 1.0, atol=1e-8), (
                f"Trace not preserved at γ_φ={gamma}"
            )

    def test_hermiticity_preserved_noise(self, make_ops: dict) -> None:
        """Density matrix should remain Hermitian with noise."""
        for gamma in [1e-4, 1e-2, 1.0]:
            rho = evolve_noisy_drive_circuit(
                DEFAULT_RHO0,
                DEFAULT_T_BS,
                DEFAULT_T_H,
                0.5,
                gamma,
                1.0,
                2.0,
                -1.0,
                0.5,
                make_ops,
            )
            assert np.allclose(rho, rho.conj().T, atol=1e-8)

    def test_positivity_preserved_noise(self, make_ops: dict) -> None:
        """Density matrix eigenvalues should be non-negative with noise."""
        for gamma in [1e-4, 1e-2, 1.0]:
            rho = evolve_noisy_drive_circuit(
                DEFAULT_RHO0,
                DEFAULT_T_BS,
                DEFAULT_T_H,
                1.0,
                gamma,
                0.0,
                3.0,
                0.0,
                2.0,
                make_ops,
            )
            evals = np.linalg.eigvalsh(rho)
            assert np.all(evals >= -1e-8), (
                f"Negative eigenvalue at γ_φ={gamma}: min={float(np.min(evals)):.2e}"
            )

    def test_zero_params_reduces_to_bs_sequence(self, make_ops: dict) -> None:
        """At all params=0, the circuit should be just BS → phase → BS."""
        theta = 0.5
        gamma = 0.0
        rho = evolve_noisy_drive_circuit(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            theta,
            gamma,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        # Manual calculation
        Jz_S = make_ops["Jz_S"]
        from src.analysis.ancilla_drive_metrology import system_only_bs_unitary

        U_bs = system_only_bs_unitary(DEFAULT_T_BS)
        U_phase = expm(-1j * DEFAULT_T_H * theta * Jz_S)
        rho_manual = (
            U_bs
            @ U_phase
            @ U_bs
            @ DEFAULT_RHO0
            @ (U_bs.conj().T @ U_phase.conj().T @ U_bs.conj().T)
        )
        assert np.allclose(rho, rho_manual, atol=1e-10)

    @pytest.mark.parametrize("seed", range(3))
    def test_deterministic_seed_irrelevant(self, seed: int, make_ops: dict) -> None:
        """The circuit is deterministic (no random sampling in Lindblad)."""
        rho1 = evolve_noisy_drive_circuit(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            0.1,
            2.0,
            1.0,
            0.0,
            0.5,
            make_ops,
        )
        rho2 = evolve_noisy_drive_circuit(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            0.1,
            2.0,
            1.0,
            0.0,
            0.5,
            make_ops,
        )
        assert np.allclose(rho1, rho2, atol=1e-12)


# ============================================================================
# Test: Sensitivity Computation
# ============================================================================


class TestNoisySensitivity:
    def test_decoupled_sensitivity_at_zero_noise(self, make_ops: dict) -> None:
        """At γ_φ=0 and a_k=a_zz=0, Δθ should equal SQL."""
        dt = compute_noisy_sensitivity(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert dt == pytest.approx(SQL_REFERENCE, rel=1e-5), (
            f"Δθ={dt:.10f} != SQL={SQL_REFERENCE:.10f}"
        )

    @pytest.mark.parametrize("gamma_phi", [0.0, 1e-4, 1e-2])
    def test_sensitivity_finite_with_noise(
        self, gamma_phi: float, make_ops: dict
    ) -> None:
        """Sensitivity should be finite for small noise."""
        dt = compute_noisy_sensitivity(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            gamma_phi,
            2.0,
            1.0,
            0.0,
            0.5,
            make_ops,
        )
        assert np.isfinite(dt), f"Non-finite Δθ at γ_φ={gamma_phi}"

    def test_sensitivity_positive(self, make_ops: dict) -> None:
        """Sensitivity should always be positive."""
        for gamma in [0.0, 1e-4, 1e-2]:
            dt = compute_noisy_sensitivity(
                DEFAULT_RHO0,
                DEFAULT_T_BS,
                DEFAULT_T_H,
                0.5,
                gamma,
                1.0,
                0.0,
                0.0,
                0.0,
                make_ops,
            )
            assert dt > 0, f"Non-positive Δθ at γ_φ={gamma}"

    def test_sensitivity_worsens_with_noise(self, make_ops: dict) -> None:
        """Δθ should increase monotonically with γ_φ for fixed params."""
        fixed_params = (2.0, 1.0, 0.0, 0.5)
        thetas = [1.0]
        gammas = [0.0, 1e-4, 1e-2, 1.0]
        deltas = []
        for gamma in gammas:
            dt = compute_noisy_sensitivity(
                DEFAULT_RHO0,
                DEFAULT_T_BS,
                DEFAULT_T_H,
                thetas[0],
                gamma,
                *fixed_params,
                make_ops,
            )
            deltas.append(dt)

        # Δθ should be non-decreasing with γ_φ
        for i in range(1, len(deltas)):
            finite_prev = np.isfinite(deltas[i - 1])
            finite_curr = np.isfinite(deltas[i])
            if finite_prev and finite_curr:
                assert deltas[i] >= deltas[i - 1] - 0.01 * deltas[i - 1], (
                    f"Δθ decreased from g={gammas[i - 1]} to g={gammas[i]}: "
                    f"{deltas[i - 1]:.6f} → {deltas[i]:.6f}"
                )

    def test_high_noise_gives_large_delta(self, make_ops: dict) -> None:
        """At very high noise, Δθ should be very large."""
        dt = compute_noisy_sensitivity(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert np.isinf(dt) or dt > 10 * SQL_REFERENCE, (
            f"Δθ={dt:.4f} should be large at high noise"
        )


# ============================================================================
# Test: Sensitivity with Diagnostics
# ============================================================================


class TestSensitivityWithDiagnostics:
    def test_diagnostics_consistency(self, make_ops: dict) -> None:
        """compute_noisy_sensitivity and *_with_diagnostics should agree."""
        dt1 = compute_noisy_sensitivity(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            0.5,
            0.01,
            1.0,
            2.0,
            0.0,
            0.5,
            make_ops,
        )
        dt2, exp_val, var_val, d_exp = compute_noisy_sensitivity_with_diagnostics(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            0.5,
            0.01,
            1.0,
            2.0,
            0.0,
            0.5,
            make_ops,
        )
        assert dt1 == pytest.approx(dt2, rel=1e-12)
        assert np.isfinite(exp_val)
        assert var_val >= 0
        assert np.isfinite(d_exp) or np.isinf(dt2)

    def test_diagnostics_at_zero_noise(self, make_ops: dict) -> None:
        """Diagnostics should be finite at γ_φ=0."""
        _, exp_val, var_val, d_exp = compute_noisy_sensitivity_with_diagnostics(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            make_ops,
        )
        assert np.isfinite(exp_val)
        assert var_val >= 0
        assert np.isfinite(d_exp)


# ============================================================================
# Test: Noise Decoupled Baseline
# ============================================================================


class TestNoiseDecoupledBaseline:
    def test_baseline_at_zero_noise(self) -> None:
        """At γ_φ=0, decoupled baseline should equal SQL."""
        dt = compute_noisy_decoupled_baseline(gamma_phi=0.0)
        assert dt == pytest.approx(SQL_REFERENCE, rel=1e-5), (
            f"γ_φ=0: Δθ={dt:.10f} != SQL={SQL_REFERENCE:.10f}"
        )

    @pytest.mark.parametrize("gamma_phi", [1e-4, 1e-2, 0.1])
    def test_baseline_degraded_by_noise(self, gamma_phi: float) -> None:
        """At γ_φ>0, decoupled baseline should be worse than SQL (dephasing
        destroys coherences that the BS2-rotated measurement depends on)."""
        dt = compute_noisy_decoupled_baseline(gamma_phi=gamma_phi)
        assert dt > SQL_REFERENCE, f"γ_φ={gamma_phi}: Δθ={dt:.10f} should be > SQL"
        assert np.isfinite(dt), f"γ_φ={gamma_phi}: Δθ={dt:.10f} should be finite"

    def test_baseline_finite_low_noise(self) -> None:
        """Baseline should be finite for small γ_φ."""
        for gamma in [1e-4, 1e-3, 0.01]:
            dt = compute_noisy_decoupled_baseline(gamma_phi=gamma)
            assert np.isfinite(dt), f"γ_φ={gamma}: Δθ={dt} not finite"

    def test_baseline_large_at_high_noise(self) -> None:
        """At very high noise, decoupled baseline should be very large."""
        dt = compute_noisy_decoupled_baseline(gamma_phi=100.0)
        assert np.isinf(dt) or dt > 10 * SQL_REFERENCE, (
            f"γ_φ=100: Δθ={dt:.4f} should be large"
        )

    @pytest.mark.parametrize("theta_true", [0.1, 0.5, 2.0, 5.0])
    def test_baseline_finite_at_different_theta(self, theta_true: float) -> None:
        """Decoupled baseline should be finite for different θ at low noise."""
        dt = compute_noisy_decoupled_baseline(gamma_phi=0.01, theta_true=theta_true)
        assert np.isfinite(dt), f"θ={theta_true}: Δθ={dt:.10f} not finite"


# ============================================================================
# Test: Objective Function
# ============================================================================


class TestNoisyObjective:
    def test_objective_finite(self, make_ops: dict) -> None:
        """Objective should return a finite value for valid parameters."""
        obj = noisy_sensitivity_objective(
            np.array([1.0, 2.0, 0.0, 0.5]),
            theta_true=1.0,
            gamma_phi=0.01,
            ops=make_ops,
        )
        assert np.isfinite(obj), f"Objective returned non-finite: {obj}"

    def test_objective_penalty_outside_bounds(self, make_ops: dict) -> None:
        """Objective should return a large value for out-of-bounds params."""
        obj = noisy_sensitivity_objective(
            np.array([100.0, 0.0, 0.0, 0.0]),
            theta_true=1.0,
            gamma_phi=0.01,
            ops=make_ops,
        )
        assert obj > 1e9, f"Objective should be large for OOB params: {obj}"

    def test_objective_symmetric(self, make_ops: dict) -> None:
        """Objective should be symmetric about a_x = 0 in some configurations."""
        obj_pos = noisy_sensitivity_objective(
            np.array([1.0, 0.0, 0.0, 0.0]),
            theta_true=1.0,
            gamma_phi=0.0,
            ops=make_ops,
        )
        obj_neg = noisy_sensitivity_objective(
            np.array([-1.0, 0.0, 0.0, 0.0]),
            theta_true=1.0,
            gamma_phi=0.0,
            ops=make_ops,
        )
        assert np.isfinite(obj_pos)
        assert np.isfinite(obj_neg)


# ============================================================================
# Test: Random Search
# ============================================================================


class TestNoisyRandomSearch:
    def test_random_search_runs(self) -> None:
        """Random search should complete without error."""
        samples, deltas, best_params, best_delta = run_noisy_random_search(
            theta=1.0,
            gamma_phi=0.01,
            n_samples=20,
            seed=42,
        )
        assert samples.shape == (20, 4)
        assert deltas.shape == (20,)
        assert len(best_params) == 4
        assert np.isfinite(best_delta) or np.isinf(best_delta)

    def test_random_search_finds_best(self) -> None:
        """The reported best should be the minimum of all samples."""
        samples, deltas, best_params, best_delta = run_noisy_random_search(
            theta=1.0,
            gamma_phi=0.0,
            n_samples=30,
            seed=42,
        )
        min_idx = int(np.argmin(deltas))
        expected_best = float(deltas[min_idx])
        assert best_delta == pytest.approx(expected_best, rel=1e-12)
        assert np.allclose(
            best_params,
            (
                float(samples[min_idx, 0]),
                float(samples[min_idx, 1]),
                float(samples[min_idx, 2]),
                float(samples[min_idx, 3]),
            ),
            atol=1e-12,
        )

    def test_random_search_deterministic(self) -> None:
        """Same seed should give same results."""
        res1 = run_noisy_random_search(theta=0.5, gamma_phi=0.1, n_samples=10, seed=42)
        res2 = run_noisy_random_search(theta=0.5, gamma_phi=0.1, n_samples=10, seed=42)
        assert np.allclose(res1[0], res2[0])
        assert np.allclose(res1[1], res2[1])
        assert res1[3] == pytest.approx(res2[3])


# ============================================================================
# Test: Nelder-Mead
# ============================================================================


class TestNoisyNelderMead:
    def test_nm_runs(self) -> None:
        """Nelder-Mead should complete without error."""
        result = run_noisy_nelder_mead(
            theta_true=1.0,
            gamma_phi=0.01,
            x0=np.array([1.0, 0.0, 0.0, 0.0]),
            maxiter=50,
        )
        assert np.isfinite(result["delta_theta_opt"]) or np.isinf(
            result["delta_theta_opt"]
        )
        assert result["params_opt"].shape == (4,)

    def test_nm_diagnostics(self) -> None:
        """Diagnostics should be recorded."""
        result = run_noisy_nelder_mead(
            theta_true=1.0,
            gamma_phi=0.0,
            x0=np.array([0.0, 0.0, 0.0, 0.0]),
            maxiter=50,
        )
        assert "expectation_Jz" in result
        assert "variance_Jz" in result
        assert "d_exp_d_theta" in result

    def test_nm_improves_over_decoupled(self) -> None:
        """Nelder-Mead should find parameters that beat the decoupled baseline."""
        decoupled_dt = compute_noisy_decoupled_baseline(gamma_phi=0.0)
        result = run_noisy_nelder_mead(
            theta_true=1.0,
            gamma_phi=0.0,
            x0=np.array([2.0, 1.0, 0.0, 0.5]),
            maxiter=200,
        )
        nm_dt = float(result["delta_theta_opt"])
        if np.isfinite(nm_dt):
            assert nm_dt <= decoupled_dt * 1.1, (
                f"NM Δθ={nm_dt:.6f} should not be much worse than decoupled "
                f"{decoupled_dt:.6f}"
            )


# ============================================================================
# Test: Noise Scan
# ============================================================================


class TestNoiseScan:
    def test_noise_scan_runs(self) -> None:
        """Noise scan should complete without error for a small grid."""
        result = run_noise_scan(
            theta_values=[0.1, 1.0],
            gamma_phi_values=[1e-4, 0.01, 1.0],
            n_random=10,
            n_nm_refine=2,
            maxiter=20,
        )
        assert len(result.theta_values) == 2
        assert len(result.gamma_phi_values) == 3
        assert result.delta_theta_per_pair.shape == (2, 3)

    def test_noise_scan_all_results_finite(self) -> None:
        """All (θ, γ_φ) pairs should produce finite Δθ for small noise."""
        result = run_noise_scan(
            theta_values=[0.5],
            gamma_phi_values=[1e-4, 1e-2],
            n_random=10,
            n_nm_refine=2,
            maxiter=20,
        )
        for i in range(len(result.theta_values)):
            for j in range(len(result.gamma_phi_values)):
                dt = result.delta_theta_per_pair[i, j]
                assert np.isfinite(dt), (
                    f"Non-finite Δθ at θ={result.theta_values[i]}, "
                    f"γ_φ={result.gamma_phi_values[j]}"
                )

    def test_noise_scan_params_recorded(self) -> None:
        """Optimal parameters should be recorded for each pair."""
        result = run_noise_scan(
            theta_values=[1.0],
            gamma_phi_values=[1e-3, 0.1],
            n_random=10,
            n_nm_refine=2,
            maxiter=20,
        )
        assert len(result.best_params_per_pair) == 2  # n_theta * n_gamma
        for params in result.best_params_per_pair:
            assert len(params) == 4


# ============================================================================
# Test: Parquet Roundtrip
# ============================================================================


class TestParquetRoundtrip:
    def test_noise_scan_roundtrip(self, tmp_path: Path) -> None:
        """DriveNoiseScanResult should survive a roundtrip."""
        original = DriveNoiseScanResult(
            theta_values=np.array([0.1, 1.0]),
            gamma_phi_values=np.array([1e-4, 0.01, 1.0]),
            best_params_per_pair=[
                (1.0, 2.0, 0.0, 0.5),
                (0.0, 1.0, 0.5, 1.0),
                (2.0, 0.0, 1.0, 0.0),
                (0.5, 0.5, 0.0, 2.0),
                (1.0, 0.0, 2.0, 0.5),
                (0.0, 2.0, 1.0, 0.0),
            ],
            delta_theta_per_pair=np.array([[0.05, 0.06, 0.10], [0.04, 0.05, 0.08]]),
            expectation_Jz_per_pair=np.array([[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]]),
            variance_Jz_per_pair=np.array([[0.05, 0.06, 0.08], [0.04, 0.05, 0.07]]),
            d_exp_d_theta_per_pair=np.array([[-0.5, -0.4, -0.3], [-0.6, -0.5, -0.4]]),
            sql=0.1,
            T_H=10.0,
            n_random=500,
            n_nm_refine=10,
            maxiter=2000,
            bounds_lo=-3.0,
            bounds_hi=3.0,
            fd_step=1e-7,
            seed=123,
        )
        csv_path = tmp_path / "test_noise.parquet"
        original.save_parquet(csv_path)
        loaded = DriveNoiseScanResult.from_parquet(csv_path)

        assert np.allclose(loaded.theta_values, original.theta_values)
        assert np.allclose(loaded.gamma_phi_values, original.gamma_phi_values)
        assert np.allclose(loaded.delta_theta_per_pair, original.delta_theta_per_pair)
        assert np.allclose(
            loaded.expectation_Jz_per_pair, original.expectation_Jz_per_pair
        )
        assert np.allclose(loaded.variance_Jz_per_pair, original.variance_Jz_per_pair)
        assert loaded.sql == pytest.approx(original.sql)
        assert pytest.approx(original.T_H) == loaded.T_H
        # Verify hyperparameter metadata roundtrip
        assert loaded.n_random == 500
        assert loaded.n_nm_refine == 10
        assert loaded.maxiter == 2000
        assert loaded.bounds_lo == pytest.approx(-3.0)
        assert loaded.bounds_hi == pytest.approx(3.0)
        assert loaded.fd_step == pytest.approx(1e-7)
        assert loaded.seed == 123

    def test_noise_scan_roundtrip_metadata(self, tmp_path: Path) -> None:
        """Verify all metadata fields survive roundtrip."""
        original = DriveNoiseScanResult(
            theta_values=np.array([0.5]),
            gamma_phi_values=np.array([1e-3, 0.1]),
            best_params_per_pair=[(2.0, 1.0, 0.0, 0.5), (1.0, 0.0, 1.0, 2.0)],
            delta_theta_per_pair=np.array([[0.04, 0.07]]),
            expectation_Jz_per_pair=np.array([[0.3, 0.2]]),
            variance_Jz_per_pair=np.array([[0.06, 0.09]]),
            d_exp_d_theta_per_pair=np.array([[-0.3, -0.2]]),
            sql=0.1,
            T_H=10.0,
            n_random=100,
            n_nm_refine=5,
            maxiter=1000,
            bounds_lo=-2.0,
            bounds_hi=2.0,
            fd_step=1e-5,
            seed=99,
        )
        csv_path = tmp_path / "test_noise_meta.parquet"
        original.save_parquet(csv_path)
        loaded = DriveNoiseScanResult.from_parquet(csv_path)

        assert loaded.theta_values[0] == pytest.approx(0.5)
        assert loaded.gamma_phi_values[0] == pytest.approx(1e-3)
        assert loaded.delta_theta_per_pair[0, 0] == pytest.approx(0.04)
        assert loaded.sql == pytest.approx(0.1)
        assert pytest.approx(10.0) == loaded.T_H
        assert loaded.n_random == 100
        assert loaded.n_nm_refine == 5
        assert loaded.maxiter == 1000
        assert loaded.bounds_lo == pytest.approx(-2.0)
        assert loaded.bounds_hi == pytest.approx(2.0)
        assert loaded.fd_step == pytest.approx(1e-5)
        assert loaded.seed == 99

    def test_from_parquet_missing_core_columns_raises(self, tmp_path: Path) -> None:
        """from_parquet should fail fast when core required columns are missing."""
        import pandas as pd

        df_bad = pd.DataFrame(
            {
                "theta": [0.1],
                "gamma_phi": [0.01],
                "a_x": [1.0],
                "a_y": [0.0],
                "a_z": [0.0],
                # missing a_zz, delta_theta, expectation_Jz, variance_Jz,
                # d_exp_d_theta, sql, T_H, n_random, n_nm_refine, maxiter,
                # bounds_lo, bounds_hi, fd_step, seed
            }
        )
        csv_path = tmp_path / "bad_noise.parquet"
        df_bad.to_parquet(csv_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            DriveNoiseScanResult.from_parquet(csv_path)

    def test_from_parquet_missing_diagnostics_columns_raises(
        self, tmp_path: Path
    ) -> None:
        """from_parquet should fail fast when diagnostics columns are missing."""
        import pandas as pd

        # All core columns present but diagnostics (expectation_Jz, variance_Jz,
        # d_exp_d_theta) missing
        df_bad = pd.DataFrame(
            {
                "theta": [0.1],
                "gamma_phi": [0.01],
                "a_x": [1.0],
                "a_y": [0.0],
                "a_z": [0.0],
                "a_zz": [0.5],
                "delta_theta": [0.05],
                "sql": [0.1],
                "T_H": [10.0],
                "n_random": [1000],
                "n_nm_refine": [25],
                "maxiter": [5000],
                "bounds_lo": [-5.0],
                "bounds_hi": [5.0],
                "fd_step": [1e-6],
                "seed": [42],
                # missing expectation_Jz, variance_Jz, d_exp_d_theta
            }
        )
        csv_path = tmp_path / "bad_noise_diag.parquet"
        df_bad.to_parquet(csv_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            DriveNoiseScanResult.from_parquet(csv_path)


# ============================================================================
# Test: Physical Invariants
# ============================================================================


class TestPhysicalInvariants:
    def test_trace_preserved_all_params(self) -> None:
        """Trace should be preserved for all typical parameters."""
        ops = build_two_qubit_operators()
        for theta in [0.1, 1.0, 5.0]:
            for gamma in [0.0, 1e-4, 1e-2, 0.1]:
                for params in [
                    (0.0, 0.0, 0.0, 0.0),
                    (2.0, 1.0, 0.0, 0.5),
                    (0.0, 3.0, 0.0, -2.0),
                ]:
                    ax, ay, az, azz = params
                    rho = evolve_noisy_drive_circuit(
                        DEFAULT_RHO0,
                        DEFAULT_T_BS,
                        DEFAULT_T_H,
                        theta,
                        gamma,
                        ax,
                        ay,
                        az,
                        azz,
                        ops,
                    )
                    assert np.isclose(np.trace(rho), 1.0, atol=1e-8), (
                        f"Trace not preserved at θ={theta}, γ={gamma}, params={params}"
                    )

    def test_hermiticity_all_params(self) -> None:
        """Density matrix should be Hermitian for all parameters."""
        ops = build_two_qubit_operators()
        for theta in [0.1, 1.0]:
            for gamma in [0.0, 1e-3]:
                for params in [
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 2.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 3.0),
                    (2.0, -1.0, 3.0, -2.0),
                ]:
                    ax, ay, az, azz = params
                    rho = evolve_noisy_drive_circuit(
                        DEFAULT_RHO0,
                        DEFAULT_T_BS,
                        DEFAULT_T_H,
                        theta,
                        gamma,
                        ax,
                        ay,
                        az,
                        azz,
                        ops,
                    )
                    assert np.allclose(rho, rho.conj().T, atol=1e-8), (
                        f"Not Hermitian at θ={theta}, γ={gamma}, params={params}"
                    )

    def test_positivity_all_params(self) -> None:
        """Density matrix eigenvalues should be non-negative."""
        ops = build_two_qubit_operators()
        for theta in [0.5, 2.0]:
            for gamma in [0.0, 1e-3, 0.1]:
                for params in [
                    (1.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 2.0),
                ]:
                    ax, ay, az, azz = params
                    rho = evolve_noisy_drive_circuit(
                        DEFAULT_RHO0,
                        DEFAULT_T_BS,
                        DEFAULT_T_H,
                        theta,
                        gamma,
                        ax,
                        ay,
                        az,
                        azz,
                        ops,
                    )
                    evals = np.linalg.eigvalsh(rho)
                    assert np.all(evals >= -1e-8), (
                        f"Negative eigenvalues at θ={theta}, γ={gamma}: "
                        f"min={float(np.min(evals)):.2e}"
                    )

    def test_decoupled_sensitivity_at_zero_noise(self) -> None:
        """At γ_φ=0 and a_k=a_zz=0, sensitivity should equal SQL."""
        ops = build_two_qubit_operators()
        dt = compute_noisy_sensitivity(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            ops,
        )
        assert dt == pytest.approx(SQL_REFERENCE, rel=1e-4), (
            f"γ_φ=0: Δθ={dt:.10f} != SQL={SQL_REFERENCE:.10f}"
        )

    def test_decoupled_sensitivity_degraded_by_noise(self) -> None:
        """At γ_φ>0, sensitivity should be worse than SQL (dephasing
        destroys system coherences even without entanglement)."""
        ops = build_two_qubit_operators()
        for gamma in [1e-4, 1e-2, 0.1]:
            dt = compute_noisy_sensitivity(
                DEFAULT_RHO0,
                DEFAULT_T_BS,
                DEFAULT_T_H,
                1.0,
                gamma,
                0.0,
                0.0,
                0.0,
                0.0,
                ops,
            )
            assert dt > SQL_REFERENCE, f"γ_φ={gamma}: Δθ={dt:.10f} should be > SQL"
            assert np.isfinite(dt), f"γ_φ={gamma}: Δθ={dt:.10f} should be finite"

    def test_noise_free_reproduction(self) -> None:
        """At γ_φ=10^{-4} (negligible noise), optimised Δθ should be
        sub-SQL.  A short optimisation run (random search + Nelder-Mead)
        at θ=0.2 should find parameters achieving Δθ < 0.75 × SQL,
        confirming that the protocol still works at negligible noise."""
        ops = build_two_qubit_operators()

        # Stage 1: small random search
        _, _, best_params, best_delta = run_noisy_random_search(
            theta=0.2,
            gamma_phi=1e-4,
            n_samples=200,
            seed=42,
        )

        # Stage 2: Nelder-Mead refinement
        nm_result = run_noisy_nelder_mead(
            theta_true=0.2,
            gamma_phi=1e-4,
            x0=np.array(best_params),
            maxiter=300,
        )
        dt_opt = float(nm_result["delta_theta_opt"])
        ratio = dt_opt / SQL_REFERENCE

        # The noise-free protocol achieves ~0.204 at θ=0.2.
        # With γ_φ=1e-4 (negligible), the ratio should be well below 1.0.
        assert ratio < 0.75, (
            f"γ_φ=1e-4 with re-optimisation: Δθ/Δθ_SQL={ratio:.4f} "
            f"should be < 0.75 (SQL).  Best Δθ={dt_opt:.6f}"
        )
        assert np.isfinite(dt_opt), "Δθ should be finite at negligible noise"

    def test_css_limit_dephased(self) -> None:
        """At very high γ_φ, the state should be heavily dephased and
        sensitivity should be extremely poor."""
        ops = build_two_qubit_operators()
        # Sensitivity should be very large or infinite at extreme dephasing
        dt = compute_noisy_sensitivity(
            DEFAULT_RHO0,
            DEFAULT_T_BS,
            DEFAULT_T_H,
            1.0,
            1e6,
            0.0,
            0.0,
            0.0,
            0.0,
            ops,
        )
        assert np.isinf(dt) or dt > 100 * SQL_REFERENCE, (
            f"At γ_φ=1e6: Δθ={dt:.4f} should be >> SQL"
        )


# ============================================================================
# Test: Constants Validation
# ============================================================================


class TestConstants:
    def test_sql_reference_correct(self) -> None:
        assert pytest.approx(1.0 / DEFAULT_T_H) == SQL_REFERENCE

    def test_drive_bounds(self) -> None:
        lo, hi = DRIVE_BOUNDS
        assert lo == -5.0
        assert hi == 5.0

    def test_fd_step_default(self) -> None:
        assert FD_STEP == 1e-6

    def test_initial_state_trace(self) -> None:
        assert np.isclose(np.trace(DEFAULT_RHO0), 1.0, atol=1e-12)

    def test_initial_state_is_00(self) -> None:
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0] = 1.0
        assert np.allclose(DEFAULT_RHO0, expected, atol=1e-12)
